"""Fast numpy/scipy acoustic feature extraction and LPC pole-tracking."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import wave

import numpy as np
from scipy.fft import dct, rfft, rfftfreq
from scipy.io import wavfile

from exp_v5_timbre import SR

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pybot_tts_v2_recorded import engine  # noqa: E402


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit PCM WAV as float64 in [-1, 1]."""
    sr, data = wavfile.read(str(path))
    if data.ndim != 1:
        raise ValueError(f"Expected mono WAV at {path}, got shape {data.shape}")
    if data.dtype == np.int16:
        x = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float64) / 2147483648.0
    else:
        x = data.astype(np.float64)
    return x, int(sr)


def write_wav_mono(path: str | Path, samples: np.ndarray, sr: int = SR, peak_norm: float = 0.90) -> float:
    """Write mono 16-bit PCM WAV, peak-normalized to <= peak_norm (matching v4)."""
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


def _frame_signal(x: np.ndarray, win_len: int, hop: int) -> np.ndarray:
    """Slice 1D signal into overlapping frames (n_frames, win_len)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < win_len:
        pad = np.zeros(win_len, dtype=np.float64)
        pad[: len(x)] = x
        return pad[None, :]
    n_frames = 1 + (len(x) - win_len) // hop
    idx = np.arange(win_len)[None, :] + np.arange(n_frames)[:, None] * hop
    return x[idx]


def _ltas_grid(low_hz: float = 80.0, high_hz: float = 8000.0) -> np.ndarray:
    grid = []
    f = float(low_hz)
    step = 2.0 ** (1.0 / 3.0)
    while f <= high_hz + 1e-9:
        grid.append(f)
        f *= step
    return np.array(grid, dtype=np.float64)


LTAS_GRID = _ltas_grid(80.0, 8000.0)


def extract_ltas(
    x: np.ndarray,
    sr: int = SR,
    win_len: int = 720,
    hop: int = 240,
    low_hz: float = 80.0,
    high_hz: float = 8000.0,
) -> dict:
    """Compute 1/3-octave LTAS envelope and linear dB/oct spectral tilt."""
    frames = _frame_signal(x, win_len, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak = float(np.max(rms)) if rms.size else 0.0
    if peak <= 1e-12:
        grid = _ltas_grid(low_hz, high_hz)
        return {
            "freqs_hz": grid,
            "ltas_db": np.zeros_like(grid),
            "ltas_db_norm": np.zeros_like(grid),
            "tilt_db_per_oct": 0.0,
            "intercept_db": 0.0,
            "n_active_frames": 0,
        }
    floor = max(0.10 * peak, peak * (10.0 ** (-30.0 / 20.0)))
    active = frames[rms >= floor]
    if len(active) < 2:
        active = frames[rms > 0]
    ham = np.hamming(win_len)
    w_frames = (active - np.mean(active, axis=1, keepdims=True)) * ham[None, :]
    n_fft = 2048
    spec = np.abs(rfft(w_frames, n=n_fft, axis=1))
    mean_mag = np.mean(spec, axis=0)
    freqs = rfftfreq(n_fft, 1.0 / sr)

    grid = _ltas_grid(low_hz, high_hz)
    half = (2.0 ** (1.0 / 3.0) - 1.0) * 0.5 * 1.2
    ltas_db = np.empty(len(grid), dtype=np.float64)
    for i, fc in enumerate(grid):
        mask = (freqs >= fc * (1.0 - half)) & (freqs <= fc * (1.0 + half))
        if np.sum(mask) >= 2:
            val = float(np.mean(mean_mag[mask]))
        else:
            j = int(np.argmin(np.abs(freqs - fc)))
            val = float(mean_mag[j])
        ltas_db[i] = 20.0 * math.log10(max(val, 1e-12))

    log2f = np.log2(grid)
    slope, intercept = np.polyfit(log2f, ltas_db, 1)
    ltas_db_norm = ltas_db - float(np.mean(ltas_db))
    return {
        "freqs_hz": grid,
        "ltas_db": ltas_db,
        "ltas_db_norm": ltas_db_norm,
        "tilt_db_per_oct": float(slope),
        "intercept_db": float(intercept),
        "n_active_frames": int(len(active)),
    }


def extract_f0_hnr(
    x: np.ndarray,
    sr: int = SR,
    win_len: int = 960,
    hop: int = 240,
    fmin: float = 75.0,
    fmax: float = 400.0,
    min_acf: float = 0.58,
    min_rms: float = 0.010,
) -> dict:
    """Vectorized normalized-ACF F0 (with parabolic interpolation) and HNR (dB)."""
    frames = _frame_signal(x, win_len, hop)
    n_frames = len(frames)
    times_s = (np.arange(n_frames) * hop + 0.5 * win_len) / sr
    f0_hz = np.full(n_frames, np.nan, dtype=np.float64)
    hnr_db = np.full(n_frames, np.nan, dtype=np.float64)
    corr = np.zeros(n_frames, dtype=np.float64)

    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak_rms = float(np.max(rms)) if rms.size else 0.0
    rms_floor = max(min_rms, 0.08 * peak_rms)

    lag_min = max(2, int(sr / fmax))
    lag_max = min(win_len // 2, int(sr / fmin))

    # Pre-center all frames and compute prefix sums for exact lag denominator
    z = frames - np.mean(frames, axis=1, keepdims=True)
    sq = z * z
    csum = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(sq, axis=1)], axis=1)
    e0 = csum[:, win_len]

    # Compute normalized ACF matrix across lags [lag_min - 1, lag_max + 1]
    lags = np.arange(lag_min - 1, lag_max + 2)
    acf_mat = np.zeros((n_frames, len(lags)), dtype=np.float64)
    for k, d in enumerate(lags):
        num = np.sum(z[:, : win_len - d] * z[:, d:], axis=1)
        e_left = csum[:, win_len - d]
        e_right = csum[:, win_len] - csum[:, d]
        den = np.sqrt(np.maximum(e_left * e_right, 1e-24))
        acf_mat[:, k] = num / den

    for i in range(n_frames):
        if rms[i] < rms_floor or e0[i] <= 1e-12:
            continue
        row = acf_mat[i]
        # interior local peaks in lags[1:-1]
        is_peak = (row[1:-1] >= row[:-2]) & (row[1:-1] >= row[2:])
        peak_idxs = np.where(is_peak)[0] + 1
        if peak_idxs.size == 0:
            corr[i] = float(np.max(row[1:-1]))
            continue
        top = float(np.max(row[peak_idxs]))
        corr[i] = top
        if top < min_acf:
            continue
        # Earliest peak within 0.06 of global peak to reject octave-down errors
        thresh = max(min_acf, top - 0.06)
        valid = peak_idxs[row[peak_idxs] >= thresh]
        k_best = int(valid[0])
        # Parabolic interpolation for sub-sample lag accuracy
        y0, y1, y2 = row[k_best - 1], row[k_best], row[k_best + 1]
        denom = 2.0 * (y0 - 2.0 * y1 + y2)
        delta = 0.0
        if abs(denom) > 1e-12:
            delta = max(-0.5, min(0.5, (y0 - y2) / denom))
        lag_interp = float(lags[k_best]) + delta
        r_peak = max(1e-4, min(0.9997, float(y1)))
        f0_hz[i] = sr / lag_interp
        hnr_db[i] = 10.0 * math.log10(r_peak / (1.0 - r_peak))

    voiced = np.isfinite(f0_hz)
    f0_v = f0_hz[voiced]
    hnr_v = hnr_db[voiced]
    if f0_v.size >= 3:
        f0_med = float(np.median(f0_v))
        f0_st = 12.0 * np.log2(f0_v / f0_med)
        iqr_st = float(np.percentile(f0_st, 75) - np.percentile(f0_st, 25))
        p05_95_st = float(np.percentile(f0_st, 95) - np.percentile(f0_st, 5))
        std_st = float(np.std(f0_st))
        f0_mean = float(np.mean(f0_v))
    elif f0_v.size > 0:
        f0_med = float(np.median(f0_v))
        f0_mean = float(np.mean(f0_v))
        iqr_st = 0.0
        p05_95_st = 0.0
        std_st = 0.0
    else:
        f0_med = np.nan
        f0_mean = np.nan
        iqr_st = np.nan
        p05_95_st = np.nan
        std_st = np.nan

    return {
        "times_s": times_s,
        "f0_hz": f0_hz,
        "hnr_db": hnr_db,
        "corr": corr,
        "voiced_mask": voiced,
        "f0_median_hz": f0_med,
        "f0_mean_hz": f0_mean,
        "f0_iqr_st": iqr_st,
        "f0_p05_p95_st": p05_95_st,
        "f0_std_st": std_st,
        "hnr_median_db": float(np.median(hnr_v)) if hnr_v.size else np.nan,
        "hnr_mean_db": float(np.mean(hnr_v)) if hnr_v.size else np.nan,
        "voiced_fraction": float(np.mean(voiced)) if voiced.size else 0.0,
    }


_MEL_CACHE: dict[tuple[int, int, int, float, float], np.ndarray] = {}


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40, fmin: float = 60.0, fmax: float = 8000.0) -> np.ndarray:
    key = (sr, n_fft, n_mels, fmin, fmax)
    if key in _MEL_CACHE:
        return _MEL_CACHE[key]
    hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel_to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    m_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * f_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bins[m - 1], bins[m], bins[m + 1]
        if f_center > f_left:
            fb[m - 1, f_left:f_center] = (np.arange(f_left, f_center) - f_left) / (f_center - f_left)
        if f_right > f_center:
            fb[m - 1, f_center:f_right] = (f_right - np.arange(f_center, f_right)) / (f_right - f_center)
    _MEL_CACHE[key] = fb
    return fb


def extract_mfcc(
    x: np.ndarray,
    sr: int = SR,
    win_len: int = 720,
    hop: int = 240,
    n_fft: int = 2048,
    n_mels: int = 40,
    n_mfcc: int = 13,
) -> dict:
    """Compute active-frame MFCCs (natural log mel-spectrum + ortho DCT-II)."""
    frames = _frame_signal(x, win_len, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak = float(np.max(rms)) if rms.size else 0.0
    if peak <= 1e-12:
        zeros = np.zeros((1, n_mfcc), dtype=np.float64)
        return {"mfcc": zeros, "mfcc_mean": zeros[0], "mfcc_std": zeros[0], "n_active": 0}
    floor = max(0.08 * peak, peak * (10.0 ** (-30.0 / 20.0)))
    active = frames[rms >= floor]
    if len(active) < 2:
        active = frames
    ham = np.hamming(win_len)
    w_frames = (active - np.mean(active, axis=1, keepdims=True)) * ham[None, :]
    power = np.abs(rfft(w_frames, n=n_fft, axis=1)) ** 2
    fb = _mel_filterbank(sr, n_fft, n_mels=n_mels, fmin=60.0, fmax=min(8000.0, sr * 0.5))
    mel_e = np.maximum(power @ fb.T, 1e-12)
    log_mel = 0.5 * np.log(mel_e)
    log_mel_db = 10.0 * np.log10(mel_e)
    # Zero-mean per frame (removes frame-level loudness offset for spectral shape LSD)
    log_mel_db_norm = log_mel_db - np.mean(log_mel_db, axis=1, keepdims=True)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return {
        "mfcc": mfcc,
        "mfcc_mean": np.mean(mfcc, axis=0),
        "mfcc_std": np.std(mfcc, axis=0),
        "log_mel_db_norm": log_mel_db_norm,
        "n_active": int(len(active)),
    }


def _dtw_mean_l2(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    diff = a[:, None, :] - b[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    ta, tb = dist.shape
    dp = np.full((ta + 1, tb + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    steps = np.zeros((ta + 1, tb + 1), dtype=np.int32)
    for i in range(1, ta + 1):
        row_dp = dp[i]
        prev_dp = dp[i - 1]
        d_row = dist[i - 1]
        for j in range(1, tb + 1):
            c_diag = prev_dp[j - 1]
            c_up = prev_dp[j]
            c_left = row_dp[j - 1]
            if c_diag <= c_up and c_diag <= c_left:
                row_dp[j] = d_row[j - 1] + c_diag
                steps[i, j] = steps[i - 1, j - 1] + 1
            elif c_up <= c_left:
                row_dp[j] = d_row[j - 1] + c_up
                steps[i, j] = steps[i - 1, j] + 1
            else:
                row_dp[j] = d_row[j - 1] + c_left
                steps[i, j] = steps[i, j - 1] + 1
    return float(dp[ta, tb] / max(1, steps[ta, tb]))


def compute_dtw_mcd(mfcc_ref: np.ndarray, mfcc_syn: np.ndarray) -> float:
    """Compute DTW-aligned Mel-Cepstral Distortion (dB) on c1..c12."""
    a = np.asarray(mfcc_ref[:, 1:13], dtype=np.float64)
    b = np.asarray(mfcc_syn[:, 1:13], dtype=np.float64)
    mean_l2 = _dtw_mean_l2(a, b)
    if not math.isfinite(mean_l2):
        return float("nan")
    return float((10.0 * math.sqrt(2.0) / math.log(10.0)) * mean_l2)


def compute_dtw_mel_lsd(log_mel_ref: np.ndarray, log_mel_syn: np.ndarray) -> float:
    """Compute DTW-aligned 40-band Log-Spectral Distance (RMSE per frame in dB)."""
    a = np.asarray(log_mel_ref, dtype=np.float64)
    b = np.asarray(log_mel_syn, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    # Convert Euclidean norm over 40 bands to RMS per band (div by sqrt(n_bands))
    mean_l2 = _dtw_mean_l2(a, b)
    return float(mean_l2 / math.sqrt(a.shape[1]))


def compute_formant_speed(x: np.ndarray, sr: int = SR) -> tuple[float, float]:
    """Mean |dF1/dt| and |dF2/dt| in Hz/s over continuous voiced frames."""
    traj = track_formant_trajectory(x, sr=sr)
    if traj is None or len(traj["times_s"]) < 3:
        return float("nan"), float("nan")
    dt = np.diff(traj["times_s"])
    valid = dt <= 0.025  # consecutive or near-consecutive voiced frames
    if not np.any(valid):
        return float("nan"), float("nan")
    df1 = np.abs(np.diff(traj["f1"])[valid]) / dt[valid]
    df2 = np.abs(np.diff(traj["f2"])[valid]) / dt[valid]
    return float(np.mean(df1)), float(np.mean(df2))


def hz_to_bark_np(hz: np.ndarray | float) -> np.ndarray | float:
    return 13.0 * np.arctan(0.00076 * np.asarray(hz)) + 3.5 * np.arctan((np.asarray(hz) / 7500.0) ** 2)


def lpc_candidates_per_frame(
    x: np.ndarray,
    sr: int = SR,
    order: int = 22,
    win_len: int = 720,
    hop: int = 240,
    preemph: float = 0.97,
) -> list[dict]:
    """Extract LPC candidate poles (f_hz, bw_hz, strength) per frame."""
    frames = _frame_signal(x, win_len, hop)
    ham = np.hamming(win_len)
    out = []
    for i, fr in enumerate(frames):
        fr_rms = float(np.sqrt(np.mean(fr * fr)))
        if fr_rms < 1e-4:
            out.append({"frame": i, "rms": fr_rms, "cands": []})
            continue
        fr0 = fr - np.mean(fr)
        pe = np.empty_like(fr0)
        pe[0] = fr0[0]
        pe[1:] = fr0[1:] - preemph * fr0[:-1]
        w = pe * ham
        r = np.correlate(w, w, mode="full")[win_len - 1 : win_len + order]
        if r[0] <= 1e-12:
            out.append({"frame": i, "rms": fr_rms, "cands": []})
            continue
        a = np.zeros(order + 1, dtype=np.float64)
        a[0] = 1.0
        err = float(r[0])
        ok = True
        for k_ord in range(1, order + 1):
            acc = float(r[k_ord]) + float(np.dot(a[1:k_ord], r[k_ord - 1 : 0 : -1]))
            k_ref = -acc / err
            if not math.isfinite(k_ref) or abs(k_ref) >= 1.0:
                ok = False
                break
            prev = a[1:k_ord].copy()
            a[1:k_ord] = prev + k_ref * prev[::-1]
            a[k_ord] = k_ref
            err *= 1.0 - k_ref * k_ref
            if err <= 1e-15:
                ok = False
                break
        if not ok:
            out.append({"frame": i, "rms": fr_rms, "cands": []})
            continue
        roots = np.roots(a)
        cands = []
        for z in roots:
            if z.imag <= 1e-7:
                continue
            rad = abs(z)
            if not (1e-5 < rad < 1.0):
                continue
            f = float(math.atan2(z.imag, z.real) * sr / math.tau)
            bw = float(-math.log(rad) * sr / math.pi)
            if 180.0 <= f <= 4100.0 and 10.0 <= bw <= 900.0:
                # Evaluate |1/A(e^jw)| at this pole's angle
                w_ang = math.tau * f / sr
                z_inv = np.exp(-1j * w_ang * np.arange(order + 1))
                mag = 1.0 / max(1e-12, float(abs(np.dot(a, z_inv))))
                cands.append((f, bw, mag))
        cands.sort(key=lambda p: p[0])
        out.append({"frame": i, "rms": fr_rms, "cands": cands})
    return out


def track_formant_trajectory(
    x: np.ndarray,
    sr: int = SR,
    order: int = 22,
    win_len: int = 720,
    hop: int = 240,
    min_rms: float = 0.010,
) -> dict | None:
    """Viterbi/DP pole-tracking of smooth (F1, F2, F3, BW1, BW2) over voiced frames."""
    frame_cands = lpc_candidates_per_frame(x, sr=sr, order=order, win_len=win_len, hop=hop)
    f0_info = extract_f0_hnr(x, sr=sr, win_len=max(win_len, 960), hop=hop, min_rms=min_rms)
    n_fr = min(len(frame_cands), len(f0_info["voiced_mask"]))
    if n_fr < 2:
        return None

    # Build valid (F1, F2, F3) triples per voiced frame
    voiced_indices = []
    frame_triples = []
    for i in range(n_fr):
        if not f0_info["voiced_mask"][i]:
            continue
        cands = frame_cands[i]["cands"]
        l1 = [p for p in cands if 200.0 <= p[0] <= 1100.0 and p[1] <= 700.0]
        l2 = [p for p in cands if 700.0 <= p[0] <= 3200.0 and p[1] <= 800.0]
        l3 = [p for p in cands if 1600.0 <= p[0] <= 4100.0 and p[1] <= 900.0]
        triples = []
        for p1 in l1:
            for p2 in l2:
                if p2[0] <= p1[0] + 150.0:
                    continue
                for p3 in l3:
                    if p3[0] <= p2[0] + 180.0:
                        continue
                    score = math.log(max(p1[2], 1e-3)) + math.log(max(p2[2], 1e-3)) + 0.5 * math.log(max(p3[2], 1e-3))
                    bw_pen = 0.002 * (p1[1] + p2[1])
                    # Mild prior penalty favoring canonical formant bands
                    node_cost = -score + bw_pen
                    triples.append((p1[0], p2[0], p3[0], p1[1], p2[1], node_cost))
        if triples:
            # Keep top 12 by node_cost for DP tractability
            triples.sort(key=lambda t: t[5])
            voiced_indices.append(i)
            frame_triples.append(triples[:12])

    if len(voiced_indices) < 2:
        return None

    # Viterbi DP over voiced frames with Bark-continuity transition penalty
    dp_costs = [np.array([t[5] for t in frame_triples[0]], dtype=np.float64)]
    backptrs = []
    for t_idx in range(1, len(frame_triples)):
        prev_tr = frame_triples[t_idx - 1]
        curr_tr = frame_triples[t_idx]
        prev_b = np.array([[hz_to_bark_np(tr[0]), hz_to_bark_np(tr[1]), hz_to_bark_np(tr[2])] for tr in prev_tr])
        curr_b = np.array([[hz_to_bark_np(tr[0]), hz_to_bark_np(tr[1]), hz_to_bark_np(tr[2])] for tr in curr_tr])
        # Transition cost weighted heavily on F1 & F2 continuity
        diff = curr_b[:, None, :] - prev_b[None, :, :]
        trans = 4.0 * (diff[:, :, 0] ** 2) + 3.0 * (diff[:, :, 1] ** 2) + 0.8 * (diff[:, :, 2] ** 2)
        dt_frames = max(1, voiced_indices[t_idx] - voiced_indices[t_idx - 1])
        trans = trans / dt_frames
        total = trans + dp_costs[-1][None, :] + np.array([c[5] for c in curr_tr])[:, None]
        best_prev = np.argmin(total, axis=1)
        dp_costs.append(total[np.arange(len(curr_tr)), best_prev])
        backptrs.append(best_prev)

    # Backtrack
    path_idx = [int(np.argmin(dp_costs[-1]))]
    for bptr in reversed(backptrs):
        path_idx.append(int(bptr[path_idx[-1]]))
    path_idx.reverse()

    chosen = [frame_triples[t][k] for t, k in enumerate(path_idx)]
    times_s = (np.array(voiced_indices, dtype=np.float64) * hop + 0.5 * win_len) / sr
    return {
        "times_s": times_s,
        "f1": np.array([c[0] for c in chosen], dtype=np.float64),
        "f2": np.array([c[1] for c in chosen], dtype=np.float64),
        "f3": np.array([c[2] for c in chosen], dtype=np.float64),
        "bw1": np.array([c[3] for c in chosen], dtype=np.float64),
        "bw2": np.array([c[4] for c in chosen], dtype=np.float64),
    }


TRAJ_GRID = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float64)


def resample_trajectory(traj: dict, n_pts: int = 5) -> np.ndarray | None:
    """Resample tracked F1, F2, BW1, BW2 onto normalized [0..1] grid -> shape (n_pts, 4)."""
    if traj is None or len(traj["times_s"]) < 2:
        return None
    t = traj["times_s"]
    t0, t1 = float(t[0]), float(t[-1])
    if t1 <= t0 + 1e-6:
        return None
    u_src = (t - t0) / (t1 - t0)
    u_dst = np.linspace(0.0, 1.0, n_pts)
    f1 = np.interp(u_dst, u_src, traj["f1"])
    f2 = np.interp(u_dst, u_src, traj["f2"])
    bw1 = np.interp(u_dst, u_src, traj["bw1"])
    bw2 = np.interp(u_dst, u_src, traj["bw2"])
    return np.stack([f1, f2, bw1, bw2], axis=1)


def fit_fold_model(
    train_utterances: list[dict],
    utt_features: dict[str, dict],
    syl_trajs: dict[str, list[tuple[str, np.ndarray]]],
    phoneme_params: dict | None = None,
) -> dict:
    """Fit speaker profiles and formant trajectories strictly from training utterances (no leakage)."""
    speakers: dict[str, dict] = {}
    by_spk: dict[str, list[dict]] = {}
    for u in train_utterances:
        by_spk.setdefault(u["speaker"], []).append(u)

    # Pooled trajectories per final across training utterances
    pooled_final_trajs: dict[str, list[np.ndarray]] = {}
    spk_final_trajs: dict[str, dict[str, list[np.ndarray]]] = {}
    for u in train_utterances:
        uid = u["id"]
        spk = u["speaker"]
        for final_key, arr in syl_trajs.get(uid, []):
            pooled_final_trajs.setdefault(final_key, []).append(arr)
            spk_final_trajs.setdefault(spk, {}).setdefault(final_key, []).append(arr)

    pooled_traj_med = {
        k: np.median(np.stack(vs, axis=0), axis=0).tolist()
        for k, vs in pooled_final_trajs.items()
        if len(vs) >= 1
    }

    for spk, us in sorted(by_spk.items()):
        tilts = []
        ltas_norms = []
        f0_meds = []
        f0_iqrs = []
        f0_ranges = []
        hnr_meds = []
        mfcc_means = []
        for u in us:
            feat = utt_features[u["id"]]
            tilts.append(feat["ltas"]["tilt_db_per_oct"])
            ltas_norms.append(feat["ltas"]["ltas_db_norm"])
            if np.isfinite(feat["f0_hnr"]["f0_median_hz"]):
                f0_meds.append(feat["f0_hnr"]["f0_median_hz"])
            if np.isfinite(feat["f0_hnr"]["f0_iqr_st"]):
                f0_iqrs.append(feat["f0_hnr"]["f0_iqr_st"])
            if np.isfinite(feat["f0_hnr"]["f0_p05_p95_st"]):
                f0_ranges.append(feat["f0_hnr"]["f0_p05_p95_st"])
            if np.isfinite(feat["f0_hnr"]["hnr_median_db"]):
                hnr_meds.append(feat["f0_hnr"]["hnr_median_db"])
            mfcc_means.append(feat["mfcc"]["mfcc_mean"])

        # Speaker-specific trajectory if >= 3 samples, else blend or pooled
        spk_trajs = {}
        for f_key, base_arr in pooled_traj_med.items():
            own = spk_final_trajs.get(spk, {}).get(f_key, [])
            if len(own) >= 3:
                spk_trajs[f_key] = np.median(np.stack(own, axis=0), axis=0).tolist()
            elif len(own) >= 1:
                own_med = np.median(np.stack(own, axis=0), axis=0)
                w = len(own) / 3.0
                spk_trajs[f_key] = (w * own_med + (1.0 - w) * np.array(base_arr)).tolist()
            else:
                spk_trajs[f_key] = base_arr

        speakers[spk] = {
            "speaker": spk,
            "n_train_utterances": len(us),
            "tilt_db_per_oct": float(np.median(tilts)),
            "tilt_mean_db_per_oct": float(np.mean(tilts)),
            "ltas_db_norm": np.mean(np.stack(ltas_norms, axis=0), axis=0).tolist(),
            "f0_median_hz": float(np.median(f0_meds)) if f0_meds else 200.0,
            "f0_iqr_st": float(np.median(f0_iqrs)) if f0_iqrs else 3.0,
            "f0_p05_p95_st": float(np.median(f0_ranges)) if f0_ranges else 7.0,
            "hnr_median_db": float(np.median(hnr_meds)) if hnr_meds else 12.5,
            "mfcc_mean": np.mean(np.stack(mfcc_means, axis=0), axis=0).tolist(),
            "final_trajectories": spk_trajs,
        }

    return {
        "speakers": speakers,
        "pooled_trajectories": pooled_traj_med,
        "train_ids": [u["id"] for u in train_utterances],
    }
