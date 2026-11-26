"""Reusable objective acoustic metrics and WAV sanity checks (24 kHz DSP).

Metrics implemented using numpy/scipy only:
- WAV sanity checks: sample rate, finite samples (no NaN/Inf), clipping, silence, duration.
- LTAS spectral tilt (dB/oct over 80-8000 Hz), spectral centroid, 1-8 kHz and 4-8 kHz band balances.
- Periodicity / HNR (Boersma-style window-normalized autocorrelation) & F0 contour stats.
- LPC (22nd-order AR) formant medians, bandwidths, IQRs, and transition speeds |dF1/dt|, |dF2/dt|.
- DTW-aligned 40-band Mel Log-Spectral Distance (Mel-LSD, dB) and magnitude-log MFCC distance (MCD, dB).
"""
from __future__ import annotations

import math
import pathlib
import wave
from functools import lru_cache

import numpy as np
from scipy.fft import dct, rfft, rfftfreq

DEFAULT_SR = 24000


def read_wav(path: str | pathlib.Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file as float64 in [-1, 1] (mono-averaged if multi-channel)."""
    p = pathlib.Path(path)
    with wave.open(str(p), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif sw == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sw} bytes in {p}")
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1)
    return arr, int(sr)


def check_wav_sanity(
    samples: np.ndarray,
    sr: int,
    *,
    expected_sr: int = DEFAULT_SR,
    min_duration_s: float = 0.05,
    silence_rms_dbfs: float = -60.0,
    clip_threshold: float = 0.999,
    max_clipped_fraction: float = 0.005,
) -> dict:
    """Validate waveform integrity (sample rate, NaN/Inf, silence, clipping, duration)."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    issues: list[str] = []
    if sr != expected_sr:
        issues.append(f"sample_rate={sr} != expected {expected_sr}")
    n = int(x.size)
    duration_s = n / sr if sr > 0 else 0.0
    if n == 0:
        issues.append("empty_audio")
        return {
            "ok": False,
            "issues": issues,
            "n_samples": 0,
            "sr": int(sr),
            "duration_s": 0.0,
            "peak": 0.0,
            "rms": 0.0,
            "rms_dbfs": None,
            "clipped_fraction": 0.0,
        }
    if not np.all(np.isfinite(x)):
        issues.append("non_finite_samples")
        finite_x = x[np.isfinite(x)]
        if finite_x.size == 0:
            return {
                "ok": False,
                "issues": issues,
                "n_samples": n,
                "sr": int(sr),
                "duration_s": round(duration_s, 4),
                "peak": float("nan"),
                "rms": float("nan"),
                "rms_dbfs": None,
                "clipped_fraction": 0.0,
            }
        x_eval = finite_x
    else:
        x_eval = x

    if duration_s < min_duration_s:
        issues.append(f"too_short={duration_s:.3f}s < {min_duration_s:.3f}s")

    peak = float(np.max(np.abs(x_eval)))
    rms = float(np.sqrt(np.mean(x_eval ** 2)))
    rms_dbfs = 20.0 * math.log10(rms) if rms > 1e-12 else -120.0
    if peak <= 1e-5 or rms_dbfs < silence_rms_dbfs:
        issues.append(f"silent_or_near_silent(rms_dbfs={rms_dbfs:.1f})")

    clipped_frac = float(np.mean(np.abs(x_eval) >= clip_threshold))
    if clipped_frac > max_clipped_fraction or peak > 1.0001:
        issues.append(f"clipping(peak={peak:.4f}, clipped_frac={clipped_frac:.4f})")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "n_samples": n,
        "sr": int(sr),
        "duration_s": round(duration_s, 4),
        "peak": round(peak, 5),
        "rms": round(rms, 6),
        "rms_dbfs": round(rms_dbfs, 2),
        "clipped_fraction": round(clipped_frac, 6),
    }


def extract_active_frames(
    x: np.ndarray,
    frame_len: int = 720,
    hop: int = 240,
    db_thresh: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice 1D signal into overlapping frames and mask frames above energy floor."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size < frame_len:
        return np.empty((0, frame_len), dtype=np.float64), np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    n_frames = 1 + (arr.size - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + np.arange(n_frames)[:, None] * hop
    frames = arr[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-18)
    peak = float(np.max(rms)) if rms.size else 0.0
    if peak <= 0.0:
        return frames, np.zeros(n_frames, dtype=bool), rms
    floor = max(0.10 * peak, peak * (10.0 ** (-db_thresh / 20.0)))
    return frames, rms >= floor, rms


def compute_ltas_metrics(x: np.ndarray, sr: int = DEFAULT_SR) -> dict | None:
    """Compute 1/3-octave LTAS tilt (80-8000 Hz), centroid, and band balances."""
    frame_len = max(64, int(round(0.030 * sr)))
    hop = max(32, int(round(0.010 * sr)))
    frames, keep, _ = extract_active_frames(x, frame_len=frame_len, hop=hop, db_thresh=30.0)
    if int(np.sum(keep)) < 4:
        return None
    kept = frames[keep]
    kept = kept - np.mean(kept, axis=1, keepdims=True)
    win = np.hamming(frame_len)[None, :]
    nfft = 2048 if sr <= 24000 else 4096
    spec = np.abs(rfft(kept * win, n=nfft, axis=1))
    mean_mag = np.mean(spec, axis=0)
    mean_pow = np.mean(spec ** 2, axis=0)
    freqs = rfftfreq(nfft, 1.0 / sr)

    hi_limit = min(8000.0, 0.48 * sr)
    grid = []
    f = 80.0
    while f <= hi_limit:
        grid.append(f)
        f *= 2.0 ** (1.0 / 3.0)
    if len(grid) < 4:
        return None
    grid_arr = np.array(grid, dtype=np.float64)
    half = (2.0 ** (1.0 / 3.0) - 1.0) * 0.5 * 1.2
    band_mags = []
    for fc in grid_arr:
        lo, hi = fc * (1.0 - half), fc * (1.0 + half)
        m = (freqs >= lo) & (freqs <= hi)
        if int(np.sum(m)) < 2:
            order = np.argsort(np.abs(freqs - fc))[:2]
            band_mags.append(float(np.mean(mean_mag[order])))
        else:
            band_mags.append(float(np.mean(mean_mag[m])))
    band_mags_arr = np.maximum(np.array(band_mags, dtype=np.float64), 1e-12)
    slope, _ = np.polyfit(np.log2(grid_arr), 20.0 * np.log10(band_mags_arr), 1)

    m_all = (freqs >= 80.0) & (freqs <= min(12000.0, 0.49 * sr))
    p_all = mean_pow[m_all]
    centroid = float(np.sum(freqs[m_all] * p_all) / (np.sum(p_all) + 1e-18))

    m_lo = (freqs >= 80.0) & (freqs < 1000.0)
    m_hi = (freqs >= 1000.0) & (freqs <= hi_limit)
    m_vhi = (freqs >= 4000.0) & (freqs <= hi_limit)
    e_lo = float(np.sum(mean_pow[m_lo]) + 1e-18)
    e_hi = float(np.sum(mean_pow[m_hi]) + 1e-18)
    e_vhi = float(np.sum(mean_pow[m_vhi]) + 1e-18)

    return {
        "tilt_db_oct": float(slope),
        "centroid_hz": centroid,
        "hi_lo_db_1_8k": 10.0 * math.log10(e_hi / e_lo),
        "vhi_lo_db_4_8k": 10.0 * math.log10(e_vhi / e_lo),
    }


def compute_f0_hnr_metrics(
    x: np.ndarray,
    sr: int = DEFAULT_SR,
    fmin: float = 75.0,
    fmax: float = 450.0,
) -> dict | None:
    """Compute F0 contour dynamics and HNR/periodicity on active speech frames."""
    frame_len = max(128, int(round(0.030 * sr)))
    hop = max(32, int(round(0.010 * sr)))
    frames, keep, _ = extract_active_frames(x, frame_len=frame_len, hop=hop, db_thresh=30.0)
    if int(np.sum(keep)) < 4:
        return None
    lag_min = max(2, int(sr / fmax))
    lag_max = min(int(sr / fmin), frame_len - 2)
    if lag_max <= lag_min + 2:
        return None

    win = np.hanning(frame_len)
    win_ac = np.correlate(win, win, mode="full")[frame_len - 1:]
    win_ac = np.maximum(win_ac / win_ac[0], 1e-6)

    f0s: list[float] = []
    r_maxs: list[float] = []
    hnrs: list[float] = []
    is_voiced: list[bool] = []

    for fr, k in zip(frames, keep):
        if not k:
            continue
        fr0 = fr - np.mean(fr)
        xw = fr0 * win
        ac = np.correlate(xw, xw, mode="full")[frame_len - 1:]
        if ac[0] <= 1e-12:
            continue
        nac = ac / (ac[0] * win_ac)
        seg = nac[lag_min:lag_max + 1]
        if seg.size < 3:
            continue
        g_idx = int(np.argmax(seg))
        g_val = float(seg[g_idx])
        best_idx = g_idx
        best_val = g_val

        # Guard against locking onto 2*T0 or 3*T0 subharmonics.
        for div in (2, 3):
            sub_lag = (lag_min + g_idx) // div
            if sub_lag >= lag_min + 1:
                lo = max(0, sub_lag - lag_min - 3)
                hi = min(len(seg), sub_lag - lag_min + 4)
                if hi > lo:
                    s_idx = lo + int(np.argmax(seg[lo:hi]))
                    if seg[s_idx] >= 0.88 * g_val and seg[s_idx] >= 0.55:
                        best_idx = s_idx
                        best_val = float(seg[s_idx])
                        break

        r_clipped = min(max(best_val, 1e-6), 0.9999)
        r_maxs.append(r_clipped)
        if r_clipped >= 0.45:
            if 0 < best_idx < len(seg) - 1:
                y0, y1, y2 = seg[best_idx - 1], seg[best_idx], seg[best_idx + 1]
                denom = y0 - 2.0 * y1 + y2
                delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
                delta = max(-0.5, min(0.5, delta))
            else:
                delta = 0.0
            lag = lag_min + best_idx + delta
            f0s.append(sr / lag)
            hnrs.append(10.0 * math.log10(r_clipped / (1.0 - r_clipped)))
            is_voiced.append(True)
        else:
            is_voiced.append(False)

    if len(f0s) < 3:
        return {
            "voiced_frac": float(np.mean(is_voiced)) if is_voiced else 0.0,
            "r_max_median": float(np.median(r_maxs)) if r_maxs else 0.0,
            "hnr_db_median": None,
            "hnr_db_mean": None,
            "hnr_db_p90": None,
            "f0_median_hz": None,
            "f0_mean_hz": None,
            "f0_iqr_hz": None,
            "f0_iqr_st": None,
            "f0_range_5_95_st": None,
            "f0_std_st": None,
            "f0_slope_abs_st_per_s": None,
        }

    f0_arr = np.array(f0s, dtype=np.float64)
    f0_st = 12.0 * np.log2(f0_arr / np.median(f0_arr))
    df0_st_per_s = np.abs(np.diff(f0_st)) / (hop / sr)
    return {
        "voiced_frac": float(np.mean(is_voiced)),
        "r_max_median": float(np.median(r_maxs)),
        "hnr_db_median": float(np.median(hnrs)),
        "hnr_db_mean": float(np.mean(hnrs)),
        "hnr_db_p90": float(np.percentile(hnrs, 90)),
        "f0_median_hz": float(np.median(f0_arr)),
        "f0_mean_hz": float(np.mean(f0_arr)),
        "f0_iqr_hz": float(np.percentile(f0_arr, 75) - np.percentile(f0_arr, 25)),
        "f0_iqr_st": float(np.percentile(f0_st, 75) - np.percentile(f0_st, 25)),
        "f0_range_5_95_st": float(np.percentile(f0_st, 95) - np.percentile(f0_st, 5)),
        "f0_std_st": float(np.std(f0_st)),
        "f0_slope_abs_st_per_s": float(np.median(df0_st_per_s)) if df0_st_per_s.size else 0.0,
    }


def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray | None:
    if r[0] <= 1e-18:
        return None
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        acc = r[i] + float(np.dot(a[1:i], r[i - 1:0:-1]))
        k = -acc / e
        if abs(k) >= 0.9999:
            return None
        a_new = a.copy()
        a_new[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a_new[i] = k
        a = a_new
        e *= (1.0 - k * k)
        if e <= 1e-18:
            return None
    return a


def compute_formant_metrics(
    x: np.ndarray,
    sr: int = DEFAULT_SR,
    order: int = 22,
    fmin: float = 75.0,
    fmax: float = 450.0,
) -> dict | None:
    """Estimate 22nd-order LPC formant medians, IQRs, bandwidths, and |dF1/dt|, |dF2/dt|."""
    frame_len = max(128, int(round(0.030 * sr)))
    hop = max(32, int(round(0.010 * sr)))
    frames, keep, _ = extract_active_frames(x, frame_len=frame_len, hop=hop, db_thresh=30.0)
    if int(np.sum(keep)) < 4:
        return None

    win = np.hamming(frame_len)
    lag_min = max(2, int(sr / fmax))
    lag_max = min(int(sr / fmin), frame_len - 2)

    f1s: list[float] = []
    f2s: list[float] = []
    f3s: list[float] = []
    bw1s: list[float] = []
    bw2s: list[float] = []
    f_seq: list[tuple[int, float, float]] = []

    for idx, (fr, k) in enumerate(zip(frames, keep)):
        if not k:
            continue
        fr0 = fr - np.mean(fr)
        ac0 = np.correlate(fr0, fr0, mode="full")[frame_len - 1:]
        if ac0[0] <= 1e-12:
            continue
        rmax = float(np.max(ac0[lag_min:lag_max + 1]) / ac0[0])
        if rmax < 0.45:
            continue

        pe = np.empty_like(fr0)
        pe[0] = fr0[0]
        pe[1:] = fr0[1:] - 0.97 * fr0[:-1]
        xw = pe * win
        ac = np.correlate(xw, xw, mode="full")[frame_len - 1:frame_len + order]
        a = _levinson_durbin(ac, order)
        if a is None:
            continue
        roots = np.roots(a)
        roots = roots[np.imag(roots) > 1e-6]
        if roots.size == 0:
            continue
        freqs = np.angle(roots) * (sr / (2.0 * math.pi))
        bws = -sr / math.pi * np.log(np.maximum(np.abs(roots), 1e-6))
        mask = (freqs >= 180.0) & (freqs <= 4500.0) & (bws > 0.0) & (bws < 600.0)
        cand = sorted(zip(freqs[mask], bws[mask]), key=lambda t: t[0])
        if len(cand) >= 2:
            f1, bw1 = float(cand[0][0]), float(cand[0][1])
            f2, bw2 = float(cand[1][0]), float(cand[1][1])
            f3 = float(cand[2][0]) if len(cand) >= 3 else float("nan")
            if 200.0 <= f1 <= 1200.0 and 500.0 <= f2 <= 3200.0 and (f2 - f1) >= 120.0:
                f1s.append(f1)
                f2s.append(f2)
                bw1s.append(bw1)
                bw2s.append(bw2)
                if not math.isnan(f3):
                    f3s.append(f3)
                f_seq.append((idx, f1, f2))

    if len(f_seq) < 4:
        return None

    df1: list[float] = []
    df2: list[float] = []
    dt_step = hop / sr
    for (i0, a1, a2), (i1, b1, b2) in zip(f_seq[:-1], f_seq[1:]):
        gap = i1 - i0
        if 1 <= gap <= 2:
            dt = gap * dt_step
            df1.append(abs(b1 - a1) / dt)
            df2.append(abs(b2 - a2) / dt)

    return {
        "n_formant_frames": len(f_seq),
        "f1_median_hz": float(np.median(f1s)),
        "f2_median_hz": float(np.median(f2s)),
        "f3_median_hz": float(np.median(f3s)) if f3s else None,
        "bw1_median_hz": float(np.median(bw1s)),
        "bw2_median_hz": float(np.median(bw2s)),
        "f1_iqr_hz": float(np.percentile(f1s, 75) - np.percentile(f1s, 25)),
        "f2_iqr_hz": float(np.percentile(f2s, 75) - np.percentile(f2s, 25)),
        "df1_dt_median_hz_s": float(np.median(df1)) if df1 else 0.0,
        "df2_dt_median_hz_s": float(np.median(df2)) if df2 else 0.0,
    }


@lru_cache(maxsize=8)
def _mel_filterbank(sr: int = DEFAULT_SR, nfft: int = 2048, n_mels: int = 40) -> np.ndarray:
    fmin = 50.0
    fmax = min(11000.0, 0.48 * sr)

    def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
        return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)

    m_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(m_pts)
    bins = np.floor((nfft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, nfft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fb


def _extract_mfcc_and_logmel(
    x: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mels: int = 40,
    n_ceps: int = 13,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    frame_len = max(64, int(round(0.025 * sr)))
    hop = max(32, int(round(0.010 * sr)))
    nfft = 2048 if sr <= 24000 else 4096
    frames, keep, _ = extract_active_frames(x, frame_len=frame_len, hop=hop, db_thresh=30.0)
    kept = frames[keep]
    if kept.shape[0] < 4:
        return None, None
    kept = kept - np.mean(kept, axis=1, keepdims=True)
    win = np.hamming(frame_len)[None, :]
    pow_spec = np.abs(rfft(kept * win, n=nfft, axis=1)) ** 2
    fb = _mel_filterbank(sr=sr, nfft=nfft, n_mels=n_mels)
    mel_pow = np.maximum(pow_spec @ fb.T, 1e-12)
    # Magnitude-domain natural log (0.5 * ln(power)) matches standard 10*sqrt(2)/ln(10) MCD scaling.
    log_mel_amp = 0.5 * np.log(mel_pow)
    ceps = dct(log_mel_amp, type=2, axis=1, norm="ortho")
    mel_db = (10.0 / math.log(10.0)) * np.log(mel_pow)
    return ceps[:, 1:n_ceps], mel_db


def _dtw_mean_l2(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    n, m = seq_a.shape[0], seq_b.shape[0]
    diff = seq_a[:, None, :] - seq_b[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    steps = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        d_row = dist[i - 1]
        for j in range(1, m + 1):
            c0 = dp[i - 1, j - 1]
            c1 = dp[i - 1, j]
            c2 = dp[i, j - 1]
            if c0 <= c1 and c0 <= c2:
                dp[i, j] = d_row[j - 1] + c0
                steps[i, j] = steps[i - 1, j - 1] + 1
            elif c1 <= c2:
                dp[i, j] = d_row[j - 1] + c1
                steps[i, j] = steps[i - 1, j] + 1
            else:
                dp[i, j] = d_row[j - 1] + c2
                steps[i, j] = steps[i, j - 1] + 1
    return float(dp[n, m] / max(1, int(steps[n, m])))


def compute_spectral_distances(
    x_a: np.ndarray,
    x_b: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mels: int = 40,
    n_ceps: int = 13,
) -> dict | None:
    """Compute DTW-aligned Mel-LSD (dB) and magnitude-log MFCC MCD (raw & CMVN, dB)."""
    c_a, mel_db_a = _extract_mfcc_and_logmel(x_a, sr=sr, n_mels=n_mels, n_ceps=n_ceps)
    c_b, mel_db_b = _extract_mfcc_and_logmel(x_b, sr=sr, n_mels=n_mels, n_ceps=n_ceps)
    if c_a is None or c_b is None or mel_db_a is None or mel_db_b is None:
        return None

    mcd_factor = 10.0 * math.sqrt(2.0) / math.log(10.0)
    mcd_db = mcd_factor * _dtw_mean_l2(c_a, c_b)
    c_a_cmn = c_a - np.mean(c_a, axis=0, keepdims=True)
    c_b_cmn = c_b - np.mean(c_b, axis=0, keepdims=True)
    mcd_cmn_db = mcd_factor * _dtw_mean_l2(c_a_cmn, c_b_cmn)

    mel_a_norm = mel_db_a - np.mean(mel_db_a, axis=1, keepdims=True)
    mel_b_norm = mel_db_b - np.mean(mel_db_b, axis=1, keepdims=True)
    mel_lsd_db = _dtw_mean_l2(mel_a_norm, mel_b_norm) / math.sqrt(float(n_mels))

    return {
        "mel_lsd_db": float(mel_lsd_db),
        "mcd_db": float(mcd_db),
        "mcd_cmn_db": float(mcd_cmn_db),
    }


def evaluate_audio(
    samples: np.ndarray,
    sr: int = DEFAULT_SR,
    *,
    expected_sr: int = DEFAULT_SR,
    fmin: float = 75.0,
    fmax: float = 450.0,
) -> dict:
    """Run WAV sanity + LTAS + F0/HNR + LPC formant dynamics on an in-memory waveform."""
    x = np.asarray(samples, dtype=np.float64).ravel()
    sanity = check_wav_sanity(x, sr, expected_sr=expected_sr)
    out: dict = {"sanity": sanity}
    if not sanity["ok"] and ("empty_audio" in sanity["issues"] or "non_finite_samples" in sanity["issues"]):
        out.update({"ltas": None, "f0_hnr": None, "formants": None})
        return out
    out["ltas"] = compute_ltas_metrics(x, sr=sr)
    out["f0_hnr"] = compute_f0_hnr_metrics(x, sr=sr, fmin=fmin, fmax=fmax)
    out["formants"] = compute_formant_metrics(x, sr=sr, fmin=fmin, fmax=fmax)
    return out


def evaluate_wav(
    path: str | pathlib.Path,
    *,
    start_sample: int = 0,
    stop_sample: int | None = None,
    expected_sr: int = DEFAULT_SR,
    fmin: float = 75.0,
    fmax: float = 450.0,
) -> dict:
    """Read a WAV file (optionally sliced to [start_sample:stop_sample]) and evaluate it."""
    x, sr = read_wav(path)
    if stop_sample is not None or start_sample > 0:
        x = x[max(0, start_sample):stop_sample]
    res = evaluate_audio(x, sr=sr, expected_sr=expected_sr, fmin=fmin, fmax=fmax)
    res["path"] = str(path)
    return res
