"""Pure numpy/scipy WORLD-inspired HNM + True-Envelope vocoder (zero neural weights)."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import wave

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt

from exp_v6_vocoder import SR


BAND_EDGES = ((60.0, 1000.0), (1000.0, 2000.0), (2000.0, 4000.0), (4000.0, 8000.0), (8000.0, 11500.0))
BAND_CENTERS = np.array([500.0, 1500.0, 3000.0, 6000.0, 10000.0], dtype=np.float64)


@dataclass
class VocoderFeatures:
    f0_hz: np.ndarray        # (T,) 0.0 for unvoiced
    voiced: np.ndarray       # (T,) bool
    log_env: np.ndarray      # (T, n_fft//2 + 1) natural log amplitude spectral envelope
    bap: np.ndarray          # (T, 5) band aperiodicity ratio in [0, 1]
    sr: int
    hop_samples: int
    n_samples: int
    n_fft: int = 1024


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read mono 16-bit PCM WAV as float64 in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        ch, width, sr, n = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        if ch != 1 or width != 2:
            raise ValueError(f"Expected mono 16-bit WAV at {path}, got ch={ch}, width={width}")
        raw = wf.readframes(n)
    pcm = np.frombuffer(raw, dtype="<i2")
    return pcm.astype(np.float64) / 32768.0, int(sr)


def write_wav_mono(path: str | Path, samples: np.ndarray, sr: int = SR, peak_norm: float = 0.90) -> float:
    """Write mono 16-bit PCM WAV, peak-normalized if peak > peak_norm or peak_norm is set."""
    arr = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    factor = min(peak_norm / max(peak, 1e-8), 1.0) if peak_norm is not None else 1.0
    clipped = np.clip(arr * factor, -0.999, 0.999)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as wf:
        wf.setparams((1, 2, int(sr), 0, "NONE", "not compressed"))
        wf.writeframes(pcm.tobytes())
    return factor


def _extract_f0_padded(
    xp: np.ndarray,
    n_frames: int,
    pad_half: int,
    sr: int = SR,
    hop: int = 120,
    win_len: int = 960,
    fmin: float = 70.0,
    fmax: float = 420.0,
    min_acf: float = 0.54,
    min_rms: float = 0.008,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized normalized-ACF pitch tracker centered at sample i*hop of original signal."""
    idx = np.arange(win_len)[None, :] + np.arange(n_frames)[:, None] * hop
    frames = xp[idx]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak_rms = float(np.max(rms)) if rms.size else 0.0
    rms_floor = max(min_rms, 0.06 * peak_rms)

    lag_min = max(2, int(sr / fmax))
    lag_max = min(win_len // 2, int(sr / fmin))

    z = frames - np.mean(frames, axis=1, keepdims=True)
    sq = z * z
    csum = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(sq, axis=1)], axis=1)
    e0 = csum[:, win_len]

    lags = np.arange(lag_min - 1, lag_max + 2)
    acf_mat = np.zeros((n_frames, len(lags)), dtype=np.float64)
    for k, d in enumerate(lags):
        num = np.sum(z[:, : win_len - d] * z[:, d:], axis=1)
        den = np.sqrt(np.maximum(csum[:, win_len - d] * (csum[:, win_len] - csum[:, d]), 1e-24))
        acf_mat[:, k] = num / den

    f0_raw = np.zeros(n_frames, dtype=np.float64)
    voiced = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        if rms[i] < rms_floor or e0[i] <= 1e-12:
            continue
        row = acf_mat[i]
        is_peak = (row[1:-1] >= row[:-2]) & (row[1:-1] >= row[2:])
        p_idxs = np.where(is_peak)[0] + 1
        if p_idxs.size == 0:
            continue
        top = float(np.max(row[p_idxs]))
        if top < min_acf:
            continue
        thresh = max(min_acf, top - 0.06)
        valid = p_idxs[row[p_idxs] >= thresh]
        k_best = int(valid[0])
        y0, y1, y2 = row[k_best - 1], row[k_best], row[k_best + 1]
        denom = 2.0 * (y0 - 2.0 * y1 + y2)
        delta = max(-0.5, min(0.5, (y0 - y2) / denom)) if abs(denom) > 1e-12 else 0.0
        f0_raw[i] = sr / (float(lags[k_best]) + delta)
        voiced[i] = True

    # 3-frame median smoothing within contiguous voiced runs to eliminate isolated octave spikes
    f0_smooth = f0_raw.copy()
    for i in range(1, n_frames - 1):
        if voiced[i - 1] and voiced[i] and voiced[i + 1]:
            f0_smooth[i] = float(np.median([f0_raw[i - 1], f0_raw[i], f0_raw[i + 1]]))
    # Drop isolated 1-frame voiced islands
    for i in range(1, n_frames - 1):
        if voiced[i] and not voiced[i - 1] and not voiced[i + 1]:
            voiced[i] = False
            f0_smooth[i] = 0.0
    return f0_smooth, voiced


_BAND_SOS_CACHE: dict[int, list[np.ndarray]] = {}


def _band_sos(sr: int) -> list[np.ndarray]:
    if sr not in _BAND_SOS_CACHE:
        _BAND_SOS_CACHE[sr] = [
            butter(4, [lo, min(hi, sr * 0.49)], btype="band", fs=sr, output="sos")
            for lo, hi in BAND_EDGES
        ]
    return _BAND_SOS_CACHE[sr]


def analyze(x: np.ndarray, sr: int = SR, hop_ms: float = 5.0, n_fft: int = 1024) -> VocoderFeatures:
    """Decompose waveform into F0, voicing, True-Envelope log-spectrum, and 5-band BAP."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    n_fr = 1 + max(0, n - 1) // hop

    pad_half = 480
    xp = np.pad(x, (pad_half, pad_half))
    f0_arr, voiced = _extract_f0_padded(xp, n_fr, pad_half, sr=sr, hop=hop, win_len=2 * pad_half)

    freqs = rfftfreq(n_fft, 1.0 / sr)
    df = sr / n_fft

    # 5 bandpass signals for time-domain fractional-lag aperiodicity
    x_bands = [sosfiltfilt(sos, xp) for sos in _band_sos(sr)]

    log_envs = np.zeros((n_fr, n_fft // 2 + 1), dtype=np.float64)
    baps = np.zeros((n_fr, len(BAND_EDGES)), dtype=np.float64)

    for i in range(n_fr):
        c_xp = i * hop + pad_half
        f0 = float(f0_arr[i])
        is_v = bool(voiced[i] and f0 > 0.0)
        win_len = int(np.clip(round(3.0 * sr / f0), 300, 900)) if is_v else 480
        s = max(0, min(len(xp) - win_len, c_xp - win_len // 2))

        seg_orig = xp[s : s + win_len]
        w = np.hanning(win_len)
        w = w / math.sqrt(float(np.sum(w * w)))

        xw_orig = (seg_orig - np.mean(seg_orig)) * w
        pwr = np.abs(rfft(xw_orig, n=n_fft)) ** 2

        if is_v:
            # CheapTrick-style pitch-adaptive boxcar smoothing across [f - F0/2, f + F0/2]
            half_bins = max(1, int(round(0.5 * f0 / df)))
            k_len = 2 * half_bins + 1
            csum = np.concatenate([[0.0], np.cumsum(np.pad(pwr, (half_bins, half_bins), mode="edge"))])
            pwr_smooth = (csum[k_len:] - csum[:-k_len]) / k_len
            log_s = 0.5 * np.log(np.maximum(pwr_smooth, 1e-18))
            q_cut = int(np.clip(round(0.85 * sr / f0), 24, 96))
        else:
            log_s = 0.5 * np.log(np.maximum(pwr, 1e-18))
            q_cut = 40

        cep = irfft(log_s, n=n_fft)
        cep_lp = np.zeros_like(cep)
        cep_lp[0] = cep[0]
        cep_lp[1:q_cut] = cep[1:q_cut]
        cep_lp[-q_cut + 1 :] = cep[-q_cut + 1 :]
        log_env_orig = rfft(cep_lp, n=n_fft).real

        # Exact Parseval frame-power normalization
        frame_pow = float(np.sum(xw_orig * xw_orig))
        env_lin = np.exp(log_env_orig)
        env_pow = float(env_lin[0] ** 2 + 2.0 * np.sum(env_lin[1:-1] ** 2) + env_lin[-1] ** 2) / n_fft
        if env_pow > 1e-18 and frame_pow > 1e-18:
            log_env_orig += 0.5 * math.log(frame_pow / env_pow)
        log_envs[i] = log_env_orig

        if is_v:
            T0_f = sr / f0
            d0 = int(math.floor(T0_f))
            frac = T0_f - d0
            W = min(360, max(160, int(round(2.0 * T0_f))))
            s0_idx = max(0, min(len(xp) - W - d0 - 2, c_xp - W // 2))
            for b, xb in enumerate(x_bands):
                v0 = xb[s0_idx : s0_idx + W]
                v1 = (1.0 - frac) * xb[s0_idx + d0 : s0_idx + d0 + W] + frac * xb[
                    s0_idx + d0 + 1 : s0_idx + d0 + 1 + W
                ]
                den = math.sqrt(max(1e-24, float(np.sum(v0 * v0)) * float(np.sum(v1 * v1))))
                rb = min(1.0, max(0.0, float(np.sum(v0 * v1)) / den / 0.975))
                baps[i, b] = math.sqrt(max(0.0004, min(0.985, 1.0 - rb)))
        else:
            baps[i, :] = 1.0

    return VocoderFeatures(
        f0_hz=f0_arr,
        voiced=voiced,
        log_env=log_envs,
        bap=baps,
        sr=sr,
        hop_samples=hop,
        n_samples=n,
        n_fft=n_fft,
    )


def synthesize(
    feats: VocoderFeatures,
    seed: int = 72,
    phase_dispersion: bool = True,
) -> np.ndarray:
    """Resynthesize waveform from (f0, voiced, log_env, bap) via pitch-sync OLA + band-noise OLA."""
    sr = feats.sr
    hop = feats.hop_samples
    n = feats.n_samples
    n_fft = feats.n_fft
    n_fr = len(feats.f0_hz)
    freqs = rfftfreq(n_fft, 1.0 / sr)

    # Vectorized minimum-phase conversion across all frames
    ceps = irfft(feats.log_env, n=n_fft, axis=1)
    win_mp = np.zeros(n_fft, dtype=np.float64)
    win_mp[0] = 1.0
    win_mp[1 : n_fft // 2] = 2.0
    win_mp[n_fft // 2] = 1.0
    H_min = np.exp(rfft(ceps * win_mp[None, :], n=n_fft, axis=1))

    # Optional smooth all-pass phase dispersion (> 2.5 kHz) to reduce pulse buzziness
    if phase_dispersion:
        f_disp = np.maximum(0.0, freqs - 2500.0) / max(1.0, (sr * 0.5 - 2500.0))
        phi_disp = np.exp(-1j * (2.2 * (f_disp ** 2) * math.pi))
        H_harm_base = H_min * phi_disp[None, :]
    else:
        H_harm_base = H_min

    # Interpolate 5-band BAP to all n_fft//2+1 bins; suppress sub-F0 & sub-70Hz rumble
    ap_full = np.empty((n_fr, len(freqs)), dtype=np.float64)
    sub70 = freqs < 70.0
    sub70_gain = (freqs[sub70] / 70.0) ** 2
    for i in range(n_fr):
        ap_row = np.clip(np.interp(freqs, BAND_CENTERS, feats.bap[i]), 0.0, 1.0)
        ap_row[sub70] *= sub70_gain
        if feats.voiced[i] and feats.f0_hz[i] > 0.0:
            f_cut = 0.85 * float(feats.f0_hz[i])
            sub_mask = freqs < f_cut
            ap_row[sub_mask] *= np.clip((freqs[sub_mask] / max(1.0, f_cut)) ** 3, 0.0, 1.0)
        ap_full[i] = ap_row
    harm_gain = np.sqrt(np.maximum(0.0, 1.0 - ap_full * ap_full))

    # Harmonic pitch-synchronous overlap-add with fractional-sample delay
    y_harm = np.zeros(n + n_fft, dtype=np.float64)
    w_ang = math.tau * freqs / sr
    pulse_offset = n_fft // 4
    t_cur = 0.0
    while t_cur < n:
        fi = min(n_fr - 1, max(0, int(round(t_cur / hop))))
        f0 = float(feats.f0_hz[fi])
        if feats.voiced[fi] and f0 > 0.0:
            T0 = sr / f0
            idx_int = int(math.floor(t_cur))
            frac = t_cur - idx_int
            # Fractional delay e^{-j * w * frac} eliminates 24 kHz sample-rounding comb loss
            delay_phasor = np.exp(-1j * w_ang * frac)
            spec_h = H_harm_base[fi] * harm_gain[fi] * delay_phasor
            pulse = np.roll(irfft(spec_h, n=n_fft), pulse_offset) * math.sqrt(T0)
            start = idx_int - pulse_offset
            if start >= 0 and start + n_fft <= len(y_harm):
                y_harm[start : start + n_fft] += pulse
            elif start < 0 and start + n_fft > 0:
                y_harm[0 : start + n_fft] += pulse[-start:]
            t_cur += T0
        else:
            t_cur += hop

    # Aperiodic + unvoiced 5 ms COLA noise excitation
    rng = np.random.default_rng(seed)
    y_ap = np.zeros(n + n_fft, dtype=np.float64)
    norm_ola = np.zeros(n + n_fft, dtype=np.float64)
    win_ola = np.hanning(2 * hop)
    noise_all = rng.standard_normal((n_fr, n_fft))
    noise_specs = rfft(noise_all, axis=1)
    ap_frames = irfft(H_min * ap_full * noise_specs, n=n_fft, axis=1)[:, : 2 * hop]
    for i in range(n_fr):
        s = i * hop - hop
        if s < 0 or s + 2 * hop > len(y_ap):
            continue
        y_ap[s : s + 2 * hop] += ap_frames[i] * win_ola
        norm_ola[s : s + 2 * hop] += win_ola * win_ola

    y_ap = y_ap / np.maximum(np.sqrt(norm_ola), 1e-6)
    return (y_harm + y_ap)[:n]
