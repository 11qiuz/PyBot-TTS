"""High-Definition Unvoiced Excitation & >8 kHz Sibilant Preservation for Non-Neural HNM Vocoding."""
from __future__ import annotations

import math

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt

from exp_v6_vocoder import vocoder
from exp_v9_clarity_pro import HOP, N_FFT, SR


BAND_EDGES_V9 = (
    (60.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 8000.0),
    (8000.0, 10000.0),
    (10000.0, 11800.0),
)
BAND_CENTERS_V9 = np.array([500.0, 1500.0, 3000.0, 6000.0, 9000.0, 10900.0], dtype=np.float64)

_SOS_V9_CACHE: dict[int, list[np.ndarray]] = {}
_HP_SOS_CACHE: dict[tuple[int, float], np.ndarray] = {}


def _band_sos_v9(sr: int) -> list[np.ndarray]:
    if sr not in _SOS_V9_CACHE:
        _SOS_V9_CACHE[sr] = [
            butter(4, [lo, min(hi, sr * 0.495)], btype="band", fs=sr, output="sos")
            for lo, hi in BAND_EDGES_V9
        ]
    return _SOS_V9_CACHE[sr]


def _hp_sos(sr: int, cutoff_hz: float) -> np.ndarray:
    key = (int(sr), float(cutoff_hz))
    if key not in _HP_SOS_CACHE:
        _HP_SOS_CACHE[key] = butter(4, min(cutoff_hz, sr * 0.45), btype="high", fs=sr, output="sos")
    return _HP_SOS_CACHE[key]


def analyze_v9(
    x: np.ndarray,
    sr: int = SR,
    hop_ms: float = 5.0,
    n_fft: int = N_FFT,
) -> tuple[vocoder.VocoderFeatures, np.ndarray]:
    """Extract VocoderFeatures with unsmoothed high-res unvoiced envelopes + 6-band BAP (incl. 8-10/10-11.8k).

    Returns `(feats, bap_gt8k)` where `feats` is 100% compatible with TimbreClonerV2/V3
    and `bap_gt8k` has shape `(T, 2)` for `[8-10 kHz, 10-11.8 kHz]`.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    n_fr = 1 + max(0, n - 1) // hop

    pad_half = 480
    xp = np.pad(x, (pad_half, pad_half))
    f0_arr, voiced = vocoder._extract_f0_padded(xp, n_fr, pad_half, sr=sr, hop=hop, win_len=2 * pad_half)

    df = sr / n_fft
    x_bands = [sosfiltfilt(sos, xp) for sos in _band_sos_v9(sr)]

    log_envs = np.zeros((n_fr, n_fft // 2 + 1), dtype=np.float64)
    baps_6 = np.ones((n_fr, len(BAND_EDGES_V9)), dtype=np.float64)

    for i in range(n_fr):
        c_xp = i * hop + pad_half
        f0 = float(f0_arr[i])
        is_v = bool(voiced[i] and f0 > 0.0)
        # Shorter 12 ms window on unvoiced bursts preserves transient sharpness; 3*T0 on voiced
        win_len = int(np.clip(round(3.0 * sr / f0), 300, 900)) if is_v else 288
        s = max(0, min(len(xp) - win_len, c_xp - win_len // 2))

        seg_orig = xp[s : s + win_len]
        w = np.hanning(win_len)
        w = w / math.sqrt(float(np.sum(w * w)))

        xw_orig = (seg_orig - np.mean(seg_orig)) * w
        pwr = np.abs(rfft(xw_orig, n=n_fft)) ** 2

        if is_v:
            half_bins = max(1, int(round(0.5 * f0 / df)))
            k_len = 2 * half_bins + 1
            csum = np.concatenate([[0.0], np.cumsum(np.pad(pwr, (half_bins, half_bins), mode="edge"))])
            pwr_smooth = (csum[k_len:] - csum[:-k_len]) / k_len
            log_s = 0.5 * np.log(np.maximum(pwr_smooth, 1e-18))
            q_cut = int(np.clip(round(0.85 * sr / f0), 24, 96))
        else:
            # High-definition unvoiced envelope: 3-bin light smoothing + q_cut=180 preserves >8 kHz & sibilant notches
            pwr_pad = np.pad(pwr, (1, 1), mode="edge")
            pwr_sm = 0.25 * pwr_pad[:-2] + 0.50 * pwr_pad[1:-1] + 0.25 * pwr_pad[2:]
            log_s = 0.5 * np.log(np.maximum(pwr_sm, 1e-18))
            q_cut = 180

        cep = irfft(log_s, n=n_fft)
        cep_lp = np.zeros_like(cep)
        cep_lp[0] = cep[0]
        cep_lp[1:q_cut] = cep[1:q_cut]
        cep_lp[-q_cut + 1 :] = cep[-q_cut + 1 :]
        log_env_orig = rfft(cep_lp, n=n_fft).real

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
                baps_6[i, b] = math.sqrt(max(0.0004, min(0.985, 1.0 - rb)))
        else:
            baps_6[i, :] = 1.0

    # Collapse 6 bands into standard 5-band bap (mean of 8-10 and 10-11.8 for band 4) while returning bap_gt8k
    baps_5 = np.empty((n_fr, 5), dtype=np.float64)
    baps_5[:, :4] = baps_6[:, :4]
    baps_5[:, 4] = 0.5 * (baps_6[:, 4] + baps_6[:, 5])

    feats = vocoder.VocoderFeatures(
        f0_hz=f0_arr,
        voiced=voiced,
        log_env=log_envs,
        bap=baps_5,
        sr=sr,
        hop_samples=hop,
        n_samples=n,
        n_fft=n_fft,
    )
    return feats, baps_6[:, 4:6]


def synthesize_v9(
    feats: vocoder.VocoderFeatures,
    *,
    seed: int = 72,
    phase_dispersion: bool = True,
    bap_gt8k: np.ndarray | None = None,
    unvoiced_residual_spans: list[tuple[int, int, np.ndarray]] | None = None,
    residual_highpass_hz: float = 1600.0,
) -> np.ndarray:
    """HD HNM synthesis preserving >8 kHz fine aperiodicity and click-free unvoiced bursts."""
    sr = feats.sr
    hop = feats.hop_samples
    n = feats.n_samples
    n_fft = feats.n_fft
    n_fr = len(feats.f0_hz)
    freqs = rfftfreq(n_fft, 1.0 / sr)

    ceps = irfft(feats.log_env, n=n_fft, axis=1)
    win_mp = np.zeros(n_fft, dtype=np.float64)
    win_mp[0] = 1.0
    win_mp[1 : n_fft // 2] = 2.0
    win_mp[n_fft // 2] = 1.0
    H_min = np.exp(rfft(ceps * win_mp[None, :], n=n_fft, axis=1))

    if phase_dispersion:
        f_disp = np.maximum(0.0, freqs - 2500.0) / max(1.0, (sr * 0.5 - 2500.0))
        phi_disp = np.exp(-1j * (2.2 * (f_disp ** 2) * math.pi))
        H_harm_base = H_min * phi_disp[None, :]
    else:
        H_harm_base = H_min

    # Fine 6-band BAP interpolation when bap_gt8k is supplied
    ap_full = np.empty((n_fr, len(freqs)), dtype=np.float64)
    sub70 = freqs < 70.0
    sub70_gain = (freqs[sub70] / 70.0) ** 2
    if bap_gt8k is not None and len(bap_gt8k) == n_fr:
        baps_use = np.concatenate([feats.bap[:, :4], bap_gt8k], axis=1)
        centers_use = BAND_CENTERS_V9
    else:
        baps_use = feats.bap
        centers_use = vocoder.BAND_CENTERS

    for i in range(n_fr):
        ap_row = np.clip(np.interp(freqs, centers_use, baps_use[i]), 0.0, 1.0)
        ap_row[sub70] *= sub70_gain
        if feats.voiced[i] and feats.f0_hz[i] > 0.0:
            f_cut = 0.85 * float(feats.f0_hz[i])
            sub_mask = freqs < f_cut
            ap_row[sub_mask] *= np.clip((freqs[sub_mask] / max(1.0, f_cut)) ** 3, 0.0, 1.0)
        ap_full[i] = ap_row
    harm_gain = np.sqrt(np.maximum(0.0, 1.0 - ap_full * ap_full))

    # Harmonic pitch-sync OLA
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

    # HD Aperiodic & Unvoiced noise excitation (proper power-normalized 5 ms hop COLA)
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

    y = (y_harm + y_ap / np.maximum(np.sqrt(norm_ola), 1e-6))[:n]

    # Optional click-free high-passed (>1.6 kHz) burst/sibilant residual strictly when voiced==False
    if unvoiced_residual_spans:
        hp_sos = _hp_sos(sr, residual_highpass_hz)
        fade_n = max(24, min(72, int(round(0.003 * sr))))  # 3 ms crossfade
        for fr_s, fr_e, raw_w in unvoiced_residual_spans:
            fr_s_c = max(0, min(n_fr, int(fr_s)))
            fr_e_c = max(fr_s_c, min(n_fr, int(fr_e)))
            if fr_e_c <= fr_s_c or np.any(feats.voiced[fr_s_c:fr_e_c]):
                continue
            s0 = fr_s_c * hop
            s1 = min(n, fr_e_c * hop)
            seg_len = s1 - s0
            if seg_len < 72 or len(raw_w) < 72:
                continue
            src = np.asarray(raw_w, dtype=np.float64)
            # High-pass to block any residual LF rumble from donor boundary while keeping sharp >1.6k burst
            if len(src) > 32:
                src_hp = sosfiltfilt(hp_sos, src)
            else:
                src_hp = src.copy()
            patch = (
                src_hp[-seg_len:].copy()
                if len(src_hp) >= seg_len
                else np.interp(np.linspace(0.0, 1.0, seg_len), np.linspace(0.0, 1.0, len(src_hp)), src_hp)
            )
            # Match patch RMS inside >1.6 kHz to synthesized target RMS so timbre adaptation tilt/level holds!
            y_seg_hp = sosfiltfilt(hp_sos, y[s0:s1]) if seg_len > 32 else y[s0:s1]
            r_syn = float(np.sqrt(np.mean(y_seg_hp ** 2)))
            r_pat = float(np.sqrt(np.mean(patch ** 2)))
            if r_syn > 1e-6 and r_pat > 1e-6:
                patch *= r_syn / r_pat
            else:
                continue
            # Keep synthesized <1.6 kHz LF envelope intact + replace >1.6 kHz noise with HD residual
            y_seg_lf = y[s0:s1] - y_seg_hp
            hd_seg = y_seg_lf + patch
            fn = min(fade_n, seg_len // 3)
            if fn > 0:
                w_in = 0.5 * (1.0 - np.cos(math.pi * np.linspace(0.0, 1.0, fn)))
                w_out = 0.5 * (1.0 + np.cos(math.pi * np.linspace(0.0, 1.0, fn)))
                y[s0 : s0 + fn] = (1.0 - w_in) * y[s0 : s0 + fn] + w_in * hd_seg[:fn]
                y[s0 + fn : s1 - fn] = hd_seg[fn : seg_len - fn]
                y[s1 - fn : s1] = (1.0 - w_out) * hd_seg[seg_len - fn :] + w_out * y[s1 - fn : s1]

    return y
