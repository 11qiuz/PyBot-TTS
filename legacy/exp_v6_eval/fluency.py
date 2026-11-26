"""Fluency & seam continuity metrics: F0 glitch rate, spectral flux jumps, V/UV smoothness, rhythm."""
from __future__ import annotations

import math
import pathlib
from typing import Sequence

import numpy as np
from scipy.fft import rfft
from scipy.ndimage import uniform_filter1d

from exp_v5_eval.metrics import DEFAULT_SR, _mel_filterbank, extract_active_frames, read_wav


def compute_f0_track(
    x: np.ndarray,
    sr: int = DEFAULT_SR,
    fmin: float = 75.0,
    fmax: float = 450.0,
) -> dict:
    """Compute frame-wise F0 (Hz), periodicity r_max, voiced mask, and RMS dB (10 ms hop)."""
    frame_len = max(128, int(round(0.030 * sr)))
    hop = max(32, int(round(0.010 * sr)))
    frames, keep, rms = extract_active_frames(x, frame_len=frame_len, hop=hop, db_thresh=30.0)
    n_frames = frames.shape[0]
    times_s = (np.arange(n_frames) * hop + frame_len * 0.5) / float(sr)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-12))

    f0_hz = np.full(n_frames, np.nan, dtype=np.float64)
    r_max = np.zeros(n_frames, dtype=np.float64)
    voiced = np.zeros(n_frames, dtype=bool)
    if n_frames == 0:
        return {
            "times_s": times_s,
            "f0_hz": f0_hz,
            "r_max": r_max,
            "voiced": voiced,
            "active": keep,
            "rms_db": rms_db,
            "hop_s": hop / float(sr),
        }

    lag_min = max(2, int(sr / fmax))
    lag_max = min(int(sr / fmin), frame_len - 2)
    win = np.hanning(frame_len)
    win_ac = np.correlate(win, win, mode="full")[frame_len - 1:]
    win_ac = np.maximum(win_ac / win_ac[0], 1e-6)

    for idx in range(n_frames):
        if not keep[idx]:
            continue
        fr0 = frames[idx] - np.mean(frames[idx])
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

        r_clipped = min(max(best_val, 0.0), 0.9999)
        r_max[idx] = r_clipped
        if r_clipped >= 0.45:
            if 0 < best_idx < len(seg) - 1:
                y0, y1, y2 = seg[best_idx - 1], seg[best_idx], seg[best_idx + 1]
                denom = y0 - 2.0 * y1 + y2
                delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
                delta = max(-0.5, min(0.5, delta))
            else:
                delta = 0.0
            f0_hz[idx] = sr / (lag_min + best_idx + delta)
            voiced[idx] = True

    return {
        "times_s": times_s,
        "f0_hz": f0_hz,
        "r_max": r_max,
        "voiced": voiced,
        "active": keep,
        "rms_db": rms_db,
        "hop_s": hop / float(sr),
    }


def compute_fluency_metrics(
    x: np.ndarray,
    sr: int = DEFAULT_SR,
    *,
    fmin: float = 75.0,
    fmax: float = 450.0,
    boundary_times_s: Sequence[float] | None = None,
) -> dict:
    """Measure F0 glitch rate, spectral flux jumps, V/UV smoothness, and rhythm variance."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size < int(0.060 * sr):
        raise ValueError("Signal too short for fluency analysis (< 60 ms)")

    track = compute_f0_track(arr, sr=sr, fmin=fmin, fmax=fmax)
    voiced = track["voiced"]
    active = track["active"]
    f0 = track["f0_hz"]
    rms_db = track["rms_db"]
    times_s = track["times_s"]
    hop_s = track["hop_s"]

    # 1. F0 continuity & glitch rate (> 4 st frame jumps inside voiced regions)
    v_pair = voiced[:-1] & voiced[1:]
    if int(np.sum(v_pair)) >= 1:
        jumps_st = np.abs(12.0 * np.log2(f0[1:][v_pair] / f0[:-1][v_pair]))
        glitches = jumps_st > 4.0
        f0_cont = {
            "n_voiced_transitions": int(jumps_st.size),
            "f0_jump_median_st": round(float(np.median(jumps_st)), 4),
            "f0_jump_p95_st": round(float(np.percentile(jumps_st, 95)), 4),
            "f0_jump_max_st": round(float(np.max(jumps_st)), 4),
            "f0_glitch_count_gt4st": int(np.sum(glitches)),
            "f0_glitch_rate_gt4st": round(float(np.mean(glitches)), 5),
        }
    else:
        f0_cont = {
            "n_voiced_transitions": 0,
            "f0_jump_median_st": None,
            "f0_jump_p95_st": None,
            "f0_jump_max_st": None,
            "f0_glitch_count_gt4st": 0,
            "f0_glitch_rate_gt4st": None,
        }

    # 2. Frame-wise 40-band normalized Mel spectral flux (aligned to same 10ms grid)
    frame_len_mel = max(64, int(round(0.025 * sr)))
    hop = max(32, int(round(0.010 * sr)))
    nfft = 2048 if sr <= 24000 else 4096
    frames_m, keep_m, _ = extract_active_frames(arr, frame_len=frame_len_mel, hop=hop, db_thresh=30.0)
    win_m = np.hamming(frame_len_mel)[None, :]
    fb = _mel_filterbank(sr=sr, nfft=nfft, n_mels=40)
    pow_spec = np.abs(rfft((frames_m - np.mean(frames_m, axis=1, keepdims=True)) * win_m, n=nfft, axis=1)) ** 2
    mel_db = (10.0 / math.log(10.0)) * np.log(np.maximum(pow_spec @ fb.T, 1e-12))
    mel_norm = mel_db - np.mean(mel_db, axis=1, keepdims=True)

    n_common = min(len(active), len(keep_m))
    act_pair = keep_m[:n_common - 1] & keep_m[1:n_common]
    flux_all = np.sqrt(np.mean((mel_norm[1:n_common] - mel_norm[:n_common - 1]) ** 2, axis=1))
    act_flux = flux_all[act_pair]

    if act_flux.size >= 1:
        med_f = float(np.median(act_flux))
        p95_f = float(np.percentile(act_flux, 95))
        flux_stats = {
            "spectral_flux_median_db": round(med_f, 4),
            "spectral_flux_p95_db": round(p95_f, 4),
            "spectral_flux_max_db": round(float(np.max(act_flux)), 4),
            "spectral_flux_p95_to_median": round(p95_f / max(med_f, 1e-6), 4),
        }
    else:
        flux_stats = {
            "spectral_flux_median_db": None,
            "spectral_flux_p95_db": None,
            "spectral_flux_max_db": None,
            "spectral_flux_p95_to_median": None,
        }

    # Optional explicit syllable/word boundary flux ratio
    if boundary_times_s:
        edge_times = 0.5 * (times_s[:n_common - 1] + times_s[1:n_common])
        b_mask = np.zeros(n_common - 1, dtype=bool)
        for bt in boundary_times_s:
            if len(edge_times):
                nearest = int(np.argmin(np.abs(edge_times - float(bt))))
                if abs(edge_times[nearest] - float(bt)) <= 0.025:
                    lo = max(0, nearest - 1)
                    hi = min(len(b_mask), nearest + 2)
                    b_mask[lo:hi] = True
        b_flux = flux_all[act_pair & b_mask]
        i_flux = flux_all[act_pair & (~b_mask)]
        if b_flux.size >= 1 and i_flux.size >= 1:
            b_med = float(np.median(b_flux))
            i_med = float(np.median(i_flux))
            flux_stats["boundary_flux_median_db"] = round(b_med, 4)
            flux_stats["interior_flux_median_db"] = round(i_med, 4)
            flux_stats["boundary_flux_ratio"] = round(b_med / max(i_med, 1e-6), 4)

    # 3. Voiced <-> Unvoiced transition smoothness (on active frames)
    vuv_mask = (voiced[:n_common - 1] != voiced[1:n_common]) & active[:n_common - 1] & active[1:n_common]
    if int(np.sum(vuv_mask)) >= 1:
        vuv_e_steps = np.abs(rms_db[1:n_common][vuv_mask] - rms_db[:n_common - 1][vuv_mask])
        vuv_flux = flux_all[vuv_mask]
        vuv_stats = {
            "n_vuv_transitions": int(vuv_e_steps.size),
            "vuv_energy_step_median_db": round(float(np.median(vuv_e_steps)), 4),
            "vuv_energy_step_p95_db": round(float(np.percentile(vuv_e_steps, 95)), 4),
            "vuv_flux_median_db": round(float(np.median(vuv_flux)), 4),
            "vuv_flux_p95_db": round(float(np.percentile(vuv_flux, 95)), 4),
        }
    else:
        vuv_stats = {
            "n_vuv_transitions": 0,
            "vuv_energy_step_median_db": None,
            "vuv_energy_step_p95_db": None,
            "vuv_flux_median_db": None,
            "vuv_flux_p95_db": None,
        }

    # 4. Speech-rate & pause rhythm variance
    rms_lin = np.power(10.0, rms_db / 20.0)
    rms_smooth = uniform_filter1d(rms_lin, size=max(3, int(round(0.050 / hop_s))))
    pk = float(np.max(rms_smooth)) if rms_smooth.size else 0.0
    speech_mask = rms_smooth >= max(0.08 * pk, pk * (10.0 ** (-28.0 / 20.0)))

    # Trim leading/trailing silence before counting internal pauses
    idx_on = np.where(speech_mask)[0]
    if idx_on.size >= 2:
        i0, i1 = int(idx_on[0]), int(idx_on[-1]) + 1
        inner = speech_mask[i0:i1]
        active_speech_ratio = float(np.mean(inner))
        # Run-length encode silence runs >= 4 frames (40 ms) inside utterance span
        pause_lens: list[float] = []
        run = 0
        for flag in inner:
            if not flag:
                run += 1
            else:
                if run >= 4:
                    pause_lens.append(run * hop_s)
                run = 0
        # Syllabic nucleus rate: local maxima in smoothed RMS above 0.18*pk separated by >= 90ms
        min_sep = max(2, int(round(0.090 / hop_s)))
        cand_peaks = [
            j for j in range(i0 + 1, i1 - 1)
            if rms_smooth[j] > rms_smooth[j - 1]
            and rms_smooth[j] >= rms_smooth[j + 1]
            and rms_smooth[j] >= 0.18 * pk
        ]
        kept_peaks: list[int] = []
        for p in cand_peaks:
            if not kept_peaks or (p - kept_peaks[-1]) >= min_sep:
                kept_peaks.append(p)
            elif rms_smooth[p] > rms_smooth[kept_peaks[-1]]:
                kept_peaks[-1] = p

        span_s = max(hop_s, (i1 - i0) * hop_s)
        syllable_rate_hz = len(kept_peaks) / span_s
        if len(kept_peaks) >= 3:
            iv = np.diff(np.array(kept_peaks, dtype=np.float64)) * hop_s
            syllable_iv_cv = float(np.std(iv) / max(np.mean(iv), 1e-6))
        else:
            syllable_iv_cv = None
        pause_cv = (
            float(np.std(pause_lens) / max(np.mean(pause_lens), 1e-6))
            if len(pause_lens) >= 2
            else None
        )
        rhythm = {
            "utterance_span_s": round(span_s, 4),
            "active_speech_ratio": round(active_speech_ratio, 4),
            "n_internal_pauses": len(pause_lens),
            "pause_dur_median_s": round(float(np.median(pause_lens)), 4) if pause_lens else 0.0,
            "pause_dur_cv": round(pause_cv, 4) if pause_cv is not None else None,
            "syllable_peaks": len(kept_peaks),
            "syllable_rate_hz": round(float(syllable_rate_hz), 3),
            "syllable_interval_cv": round(syllable_iv_cv, 4) if syllable_iv_cv is not None else None,
        }
    else:
        rhythm = {
            "utterance_span_s": 0.0,
            "active_speech_ratio": 0.0,
            "n_internal_pauses": 0,
            "pause_dur_median_s": 0.0,
            "pause_dur_cv": None,
            "syllable_peaks": 0,
            "syllable_rate_hz": 0.0,
            "syllable_interval_cv": None,
        }

    return {
        "f0_continuity": f0_cont,
        "spectral_flux": flux_stats,
        "vuv_transitions": vuv_stats,
        "rhythm": rhythm,
    }


def evaluate_fluency_wav(
    path: str | pathlib.Path,
    *,
    start_sample: int = 0,
    stop_sample: int | None = None,
    fmin: float = 75.0,
    fmax: float = 450.0,
    boundary_times_s: Sequence[float] | None = None,
) -> dict:
    """Read WAV and compute fluency/seam continuity metrics."""
    x, sr = read_wav(path)
    if stop_sample is not None or start_sample > 0:
        x = x[max(0, start_sample):stop_sample]
    res = compute_fluency_metrics(x, sr=sr, fmin=fmin, fmax=fmax, boundary_times_s=boundary_times_s)
    res["path"] = str(path)
    return res
