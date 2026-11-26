"""Prosody, CV-contrast, >8 kHz sibilant ratio, and join-click metrics for Phase 5."""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import uniform_filter1d

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_eval.metrics import DEFAULT_SR, _mel_filterbank, evaluate_wav, extract_active_frames, read_wav  # noqa: E402
from exp_v6_eval.fluency import compute_f0_track, compute_fluency_metrics  # noqa: E402
from exp_v8_data.clarity_diag import compute_clarity_metrics  # noqa: E402


def compute_v9_extended_metrics(
    x: np.ndarray,
    sr: int = DEFAULT_SR,
    *,
    fmin: float = 75.0,
    fmax: float = 450.0,
) -> dict:
    """Compute CV contrast, >8 kHz sibilant ratio, F0 declination, rhythm, and click checks."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    frame_len = max(128, int(round(0.030 * sr)))
    hop = max(32, int(round(0.010 * sr)))
    nfft = 2048 if sr <= 24000 else 4096

    frames, keep, rms = extract_active_frames(arr, frame_len=frame_len, hop=hop, db_thresh=30.0)
    track = compute_f0_track(arr, sr=sr, fmin=fmin, fmax=fmax)
    n_c = min(len(keep), len(track["voiced"]))
    keep = keep[:n_c]
    voiced = track["voiced"][:n_c]
    frames = frames[:n_c]
    rms = rms[:n_c]
    f0 = track["f0_hz"][:n_c]
    times = track["times_s"][:n_c]

    win = np.hamming(frame_len)[None, :]
    pow_spec = np.abs(rfft((frames - np.mean(frames, axis=1, keepdims=True)) * win, n=nfft, axis=1)) ** 2
    freqs = rfftfreq(nfft, 1.0 / sr)
    fb = _mel_filterbank(sr=sr, nfft=nfft, n_mels=40)
    mel_db = 10.0 * np.log10(np.maximum(pow_spec @ fb.T, 1e-12))
    mel_norm = mel_db - np.mean(mel_db, axis=1, keepdims=True)

    v_mask = keep & voiced
    u_mask = keep & (~voiced)

    # 1. Consonant-Vowel (CV) energy & spectral contrast
    if int(np.sum(v_mask)) >= 3 and int(np.sum(u_mask)) >= 2:
        v_rms_db = 20.0 * np.log10(np.maximum(rms[v_mask], 1e-12))
        u_rms_db = 20.0 * np.log10(np.maximum(rms[u_mask], 1e-12))
        cv_energy_contrast_db = float(np.median(v_rms_db) - np.median(u_rms_db))
        cv_spectral_dist_db = float(
            np.sqrt(np.mean((np.mean(mel_norm[v_mask], axis=0) - np.mean(mel_norm[u_mask], axis=0)) ** 2))
        )
    else:
        cv_energy_contrast_db = None
        cv_spectral_dist_db = None

    # 2. >8 kHz sibilant energy ratio (8-11.5 kHz vs 2-8 kHz and vs 80-1000 Hz)
    m_lo = (freqs >= 80.0) & (freqs < 1000.0)
    m_2_8 = (freqs >= 2000.0) & (freqs <= 8000.0)
    m_8_115 = (freqs > 8000.0) & (freqs <= min(11500.0, 0.48 * sr))
    e_lo = np.sum(pow_spec[:, m_lo], axis=1) + 1e-18
    e_2_8 = np.sum(pow_spec[:, m_2_8], axis=1) + 1e-18
    e_hi8 = np.sum(pow_spec[:, m_8_115], axis=1) + 1e-18

    if int(np.sum(u_mask)) >= 2 and np.any(m_8_115):
        sib_8_vs_2_8_db = float(np.median(10.0 * np.log10(e_hi8[u_mask] / e_2_8[u_mask])))
        sib_8_vs_lo_db = float(np.median(10.0 * np.log10(e_hi8[u_mask] / e_lo[u_mask])))
    else:
        sib_8_vs_2_8_db = None
        sib_8_vs_lo_db = None
    act_8_vs_2_8_db = float(np.median(10.0 * np.log10(e_hi8[keep] / e_2_8[keep]))) if int(np.sum(keep)) >= 2 and np.any(m_8_115) else None

    # 3. Phrase F0 declination (linear trend over active voiced span)
    if int(np.sum(v_mask)) >= 8:
        t_v = times[v_mask]
        t_span = max(0.05, float(t_v[-1] - t_v[0]))
        t_norm = (t_v - t_v[0]) / t_span
        f0_st = 12.0 * np.log2(f0[v_mask] / np.median(f0[v_mask]))
        if float(np.std(f0_st)) > 1e-4 and float(np.std(t_norm)) > 1e-4:
            corr = float(np.corrcoef(t_norm, f0_st)[0, 1])
            slope_total_st = float(np.polyfit(t_norm, f0_st, 1)[0])
            slope_st_per_s = slope_total_st / t_span
        else:
            corr = 0.0
            slope_st_per_s = 0.0
    else:
        corr = None
        slope_st_per_s = None

    # 4. Phrase-final lengthening proxy (last-third vs first-two-thirds inter-syllable spacing)
    rms_lin = np.maximum(rms, 1e-12)
    rms_sm = uniform_filter1d(rms_lin, size=5)
    pk = float(np.max(rms_sm)) if rms_sm.size else 0.0
    act_idx = np.where(rms_sm >= max(0.08 * pk, pk * (10.0 ** (-28.0 / 20.0))))[0]
    final_lengthening_ratio = None
    if act_idx.size >= 10:
        i0, i1 = int(act_idx[0]), int(act_idx[-1]) + 1
        min_sep = max(2, int(round(0.090 / (hop / sr))))
        cands = [j for j in range(i0 + 1, i1 - 1) if rms_sm[j] > rms_sm[j - 1] and rms_sm[j] >= rms_sm[j + 1] and rms_sm[j] >= 0.18 * pk]
        peaks: list[int] = []
        for p in cands:
            if not peaks or (p - peaks[-1]) >= min_sep:
                peaks.append(p)
            elif rms_sm[p] > rms_sm[peaks[-1]]:
                peaks[-1] = p
        if len(peaks) >= 4:
            ivs = np.diff(np.array(peaks, dtype=np.float64)) * (hop / sr)
            if len(ivs) >= 3:
                final_lengthening_ratio = round(float(ivs[-1] / max(np.median(ivs[:-1]), 1e-4)), 4)

    # 5. Join-seam waveform click detector (2nd difference normalized by local 5ms RMS)
    if arr.size > 240:
        d2 = np.abs(arr[2:] - 2.0 * arr[1:-1] + arr[:-2])
        win_5ms = max(24, int(round(0.005 * sr)))
        local_rms = np.sqrt(uniform_filter1d(arr[1:-1] ** 2, size=win_5ms) + 1e-12)
        global_rms = float(np.sqrt(np.mean(arr ** 2)))
        active_sample_mask = local_rms >= max(1e-4, 0.08 * global_rms)
        ratio = d2[active_sample_mask] / local_rms[active_sample_mask]
        if ratio.size > 0:
            click_max_d2 = float(np.max(ratio))
            click_p999_d2 = float(np.percentile(ratio, 99.9))
            click_outlier_rate = float(np.mean(ratio > 8.0))
        else:
            click_max_d2 = click_p999_d2 = click_outlier_rate = 0.0
    else:
        click_max_d2 = click_p999_d2 = click_outlier_rate = 0.0

    clarity = compute_clarity_metrics(arr, sr=sr, fmin=fmin, fmax=fmax)
    fluency = compute_fluency_metrics(arr, sr=sr, fmin=fmin, fmax=fmax)

    return {
        "cv_energy_contrast_db": round(cv_energy_contrast_db, 4) if cv_energy_contrast_db is not None else None,
        "cv_spectral_dist_db": round(cv_spectral_dist_db, 4) if cv_spectral_dist_db is not None else None,
        "unvoiced_8_115k_vs_2_8k_db": round(sib_8_vs_2_8_db, 4) if sib_8_vs_2_8_db is not None else None,
        "unvoiced_8_115k_vs_lo_db": round(sib_8_vs_lo_db, 4) if sib_8_vs_lo_db is not None else None,
        "active_8_115k_vs_2_8k_db": round(act_8_vs_2_8_db, 4) if act_8_vs_2_8_db is not None else None,
        "f0_declination_corr": round(corr, 4) if corr is not None else None,
        "f0_declination_slope_st_s": round(slope_st_per_s, 4) if slope_st_per_s is not None else None,
        "final_lengthening_ratio": final_lengthening_ratio,
        "click_max_d2_norm": round(click_max_d2, 3),
        "click_p999_d2_norm": round(click_p999_d2, 3),
        "click_outlier_rate_gt8": round(click_outlier_rate, 6),
        "clarity": clarity,
        "fluency": fluency,
    }


def evaluate_v9_wav(
    path: str | pathlib.Path,
    *,
    start_sample: int = 0,
    stop_sample: int | None = None,
    fmin: float = 75.0,
    fmax: float = 450.0,
) -> dict:
    x, sr = read_wav(path)
    if stop_sample is not None or start_sample > 0:
        x = x[max(0, start_sample):stop_sample]
    base = evaluate_wav(path, start_sample=start_sample, stop_sample=stop_sample, fmin=fmin, fmax=fmax)
    ext = compute_v9_extended_metrics(x, sr=sr, fmin=fmin, fmax=fmax)
    return {"path": str(path), **base, "v9": ext}
