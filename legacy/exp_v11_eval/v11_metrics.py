"""Phase 7 (v11) Metrics: Syllable Non-Drop Detector, Diphone Boundary Smoothness & EN Rhythm."""
from __future__ import annotations

import math
import pathlib
import sys
from typing import Any

import numpy as np
from scipy.fft import rfft
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_eval.metrics import DEFAULT_SR, _mel_filterbank, check_wav_sanity  # noqa: E402
from exp_v6_eval.fluency import compute_f0_track  # noqa: E402
from exp_v9_eval.v9_metrics import compute_v9_extended_metrics  # noqa: E402
from exp_v10_audit.loudness_audit import _normalize_active_rms, measure_windowed_loudness  # noqa: E402

EN_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "but", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "shall", "may", "might", "must", "that", "this", "these", "those",
    "it", "its", "he", "him", "his", "she", "her", "hers", "they", "them", "their",
    "we", "us", "our", "you", "your", "i", "me", "my", "there", "here", "than",
    "so", "if", "not", "no", "into", "onto", "upon", "about", "over", "under",
})


def _active_speech_span_samples(x: np.ndarray, sr: int = DEFAULT_SR, gate_rel: float = 0.015) -> tuple[int, int]:
    """Find outer active speech span [s0, s1) ignoring leading/trailing low-energy pad."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0, 0
    pk = float(np.max(np.abs(arr)))
    if pk <= 1e-12:
        return 0, len(arr)
    hop = max(24, int(round(0.005 * sr)))
    win = max(48, int(round(0.015 * sr)))
    if len(arr) <= win:
        return 0, len(arr)
    n_f = 1 + (len(arr) - win) // hop
    idx = np.arange(win)[None, :] + np.arange(n_f)[:, None] * hop
    rms = np.sqrt(np.mean(arr[idx] ** 2, axis=1))
    thresh = max(1e-5, gate_rel * float(np.percentile(rms, 95)))
    act = np.where(rms >= thresh)[0]
    if act.size == 0:
        return 0, len(arr)
    return int(act[0] * hop), min(len(arr), int(act[-1] * hop + win))


def _estimate_slot_timings_ms(x: np.ndarray, plan: dict[str, Any], sr: int = DEFAULT_SR) -> list[tuple[float, float]]:
    """Map planned units proportionally onto the active speech span of `x` (in ms)."""
    units = plan.get("units", [])
    if not units:
        return []
    s0, s1 = _active_speech_span_samples(x, sr=sr)
    start_ms = s0 * 1000.0 / sr
    span_ms = max(20.0, (s1 - s0) * 1000.0 / sr)

    raw_slots: list[tuple[float, float]] = []
    cur = 0.0
    for idx, u in enumerate(units):
        d = max(25.0, float(u.get("duration_ms", 160.0)))
        raw_slots.append((cur, cur + d))
        cur += d
        if idx + 1 < len(units):
            cur += max(0.0, float(u.get("pause_after_ms", 0.0)))
    tot_plan_ms = max(1.0, cur)
    scale = span_ms / tot_plan_ms
    return [(start_ms + a * scale, start_ms + b * scale) for a, b in raw_slots]


def detect_dropped_slots(
    x: np.ndarray,
    plan: dict[str, Any],
    sr: int = DEFAULT_SR,
    *,
    slot_timings_ms: list[tuple[float, float]] | None = None,
    active_rms_thresh_dbfs: float = -35.0,
    min_active_ms: float = 25.0,
) -> dict[str, Any]:
    """Detect dropped or collapsed syllables/words against planned timeline slots.

    Audio is normalized to -20 dBFS active RMS before thresholding frames at
    `active_rms_thresh_dbfs` (default -35 dBFS = 15 dB below speech active RMS).
    A slot is flagged dropped if its peak/mean RMS < thresh or active duration < `min_active_ms`.
    """
    units = plan.get("units", [])
    if not units:
        return {
            "n_slots": 0,
            "n_dropped_slots": 0,
            "drop_rate": 0.0,
            "dropped_slots": [],
            "min_slot_active_ms": 0.0,
            "min_slot_rms_dbfs": 0.0,
            "syllable_peak_recall": 1.0,
            "n_detected_peaks": 0,
            "n_expected_syllables": 0,
        }

    xn = _normalize_active_rms(x, target_dbfs=-20.0)
    timings = slot_timings_ms if (slot_timings_ms is not None and len(slot_timings_ms) == len(units)) else _estimate_slot_timings_ms(xn, plan, sr=sr)

    hop_ms = 5.0
    hop = max(24, int(round(hop_ms * 0.001 * sr)))
    win = max(48, int(round(0.015 * sr)))

    dropped: list[dict[str, Any]] = []
    slot_active_ms_list: list[float] = []
    slot_rms_list: list[float] = []

    for idx, (u, (t0_ms, t1_ms)) in enumerate(zip(units, timings)):
        s0 = max(0, int(round(t0_ms * 0.001 * sr)))
        s1 = min(len(xn), max(s0 + 1, int(round(t1_ms * 0.001 * sr))))
        seg = xn[s0:s1]
        slot_dur_ms = (s1 - s0) * 1000.0 / sr
        if len(seg) == 0:
            slot_rms_db = -120.0
            active_ms = 0.0
        else:
            slot_rms_db = 20.0 * math.log10(max(1e-10, float(np.sqrt(np.mean(seg * seg)))))
            if len(seg) >= win:
                n_f = 1 + (len(seg) - win) // hop
                f_idx = np.arange(win)[None, :] + np.arange(n_f)[:, None] * hop
                f_db = 20.0 * np.log10(np.maximum(1e-10, np.sqrt(np.mean(seg[f_idx] ** 2, axis=1))))
                active_ms = float(np.sum(f_db >= active_rms_thresh_dbfs)) * hop_ms
            else:
                active_ms = slot_dur_ms if slot_rms_db >= active_rms_thresh_dbfs else 0.0

        slot_active_ms_list.append(active_ms)
        slot_rms_list.append(slot_rms_db)

        reasons = []
        if slot_rms_db < active_rms_thresh_dbfs:
            reasons.append(f"rms<{active_rms_thresh_dbfs:.1f}dBFS")
        if active_ms < min_active_ms:
            reasons.append(f"active_ms<{min_active_ms:.0f}ms")
        if reasons:
            lbl = str(u.get("token") or u.get("hanzi") or u.get("pinyin") or f"u{idx}")
            dropped.append(
                {
                    "idx": idx,
                    "label": lbl,
                    "lang": str(u.get("lang", "")),
                    "slot_dur_ms": round(slot_dur_ms, 2),
                    "active_ms": round(active_ms, 2),
                    "rms_dbfs": round(slot_rms_db, 2),
                    "reason": ",".join(reasons),
                }
            )

    # Acoustic syllable peak count (20 ms smoothed envelope, 5 ms hop)
    n_expected_syl = 0
    for u in units:
        if u.get("lang") == "zh":
            n_expected_syl += 1
        else:
            arpa = u.get("arpabet") or []
            v_cnt = sum(1 for a in arpa if len(a) >= 2 and a[-1].isdigit())
            n_expected_syl += max(1, v_cnt)

    if len(xn) >= win:
        n_f = 1 + (len(xn) - win) // hop
        f_idx = np.arange(win)[None, :] + np.arange(n_f)[:, None] * hop
        env_db = 20.0 * np.log10(np.maximum(1e-10, np.sqrt(np.mean(xn[f_idx] ** 2, axis=1))))
        sm_db = uniform_filter1d(env_db, size=5)  # 25 ms smoothing
        peaks, _ = find_peaks(sm_db, height=-36.0, prominence=3.0, distance=max(1, int(round(85.0 / hop_ms))))
        n_peaks = int(len(peaks))
    else:
        n_peaks = 0

    return {
        "n_slots": len(units),
        "n_dropped_slots": len(dropped),
        "drop_rate": round(len(dropped) / max(1, len(units)), 4),
        "dropped_slots": dropped,
        "min_slot_active_ms": round(min(slot_active_ms_list) if slot_active_ms_list else 0.0, 2),
        "min_slot_rms_dbfs": round(min(slot_rms_list) if slot_rms_list else 0.0, 2),
        "n_expected_syllables": n_expected_syl,
        "n_detected_peaks": n_peaks,
        "syllable_peak_recall": round(min(1.0, n_peaks / max(1, n_expected_syl)), 4),
    }


def measure_boundary_smoothness(
    x: np.ndarray,
    sr: int = DEFAULT_SR,
    *,
    unit_boundaries_ms: list[float] | None = None,
    log_env: np.ndarray | None = None,
    boundary_frames: list[int] | None = None,
) -> dict[str, Any]:
    """Measure spectral & energy smoothness at unit join points and across all 5 ms frames."""
    arr = _normalize_active_rms(x, target_dbfs=-20.0)
    hop_ms = 5.0
    hop = max(24, int(round(hop_ms * 0.001 * sr)))
    win = max(64, int(round(0.015 * sr)))
    nfft = 1024 if sr <= 24000 else 2048

    if len(arr) < win + hop:
        return {
            "n_joins": 0,
            "join_mel_jump_median_db": None,
            "join_mel_jump_p95_db": None,
            "join_energy_jump_median_db": None,
            "frame5ms_mel_jump_p95_db": 0.0,
            "frame5ms_mel_jump_p99_db": 0.0,
            "feature_join_log_env_l2": None,
        }

    n_f = 1 + (len(arr) - win) // hop
    idx = np.arange(win)[None, :] + np.arange(n_f)[:, None] * hop
    frames = arr[idx]
    w = np.hamming(win)[None, :]
    spec = np.abs(rfft((frames - np.mean(frames, axis=1, keepdims=True)) * w, n=nfft, axis=1)) ** 2
    fb = _mel_filterbank(sr=sr, nfft=nfft, n_mels=40)
    mel_db = 10.0 * np.log10(np.maximum(spec @ fb.T, 1e-12))
    fr_db = 20.0 * np.log10(np.maximum(1e-10, np.sqrt(np.mean(frames ** 2, axis=1))))

    act = fr_db >= -45.0
    step_mel = np.sqrt(np.mean(np.diff(mel_db, axis=0) ** 2, axis=1))
    step_eng = np.abs(np.diff(fr_db))
    act_pairs = act[:-1] & act[1:]
    act_mel_steps = step_mel[act_pairs]

    join_mel_jumps: list[float] = []
    join_eng_jumps: list[float] = []
    if unit_boundaries_ms:
        for t_ms in unit_boundaries_ms:
            fi = int(round(float(t_ms) / hop_ms))
            if 1 <= fi < len(mel_db) and act[fi] and act[fi - 1]:
                # Take local max within +-1 frame of nominal boundary to allow small OLA spread
                lo = max(0, fi - 1)
                hi = min(len(step_mel), fi + 2)
                join_mel_jumps.append(float(np.max(step_mel[lo:hi])))
                join_eng_jumps.append(float(np.max(step_eng[lo:hi])))

    feat_join_l2 = None
    if log_env is not None and boundary_frames:
        vals = []
        for bf in boundary_frames:
            if 1 <= bf < len(log_env):
                d = log_env[bf].astype(np.float64) - log_env[bf - 1].astype(np.float64)
                vals.append(float(np.sqrt(np.mean(d * d))) * 20.0 / math.log(10.0))
        if vals:
            feat_join_l2 = round(float(np.median(vals)), 3)

    return {
        "n_joins": len(join_mel_jumps),
        "join_mel_jump_median_db": round(float(np.median(join_mel_jumps)), 3) if join_mel_jumps else None,
        "join_mel_jump_p95_db": round(float(np.percentile(join_mel_jumps, 95)), 3) if join_mel_jumps else None,
        "join_energy_jump_median_db": round(float(np.median(join_eng_jumps)), 3) if join_eng_jumps else None,
        "frame5ms_mel_jump_p95_db": round(float(np.percentile(act_mel_steps, 95)) if act_mel_steps.size else 0.0, 3),
        "frame5ms_mel_jump_p99_db": round(float(np.percentile(act_mel_steps, 99)) if act_mel_steps.size else 0.0, 3),
        "feature_join_log_env_l2_db": feat_join_l2,
    }


def measure_en_prosody_rhythm(
    x: np.ndarray,
    plan: dict[str, Any],
    sr: int = DEFAULT_SR,
    *,
    unit_timings_ms: list[tuple[float, float]] | None = None,
    fmin: float = 75.0,
    fmax: float = 450.0,
) -> dict[str, Any]:
    """Measure English function-vs-content word duration ratio, RMS contrast, and F0 contrast."""
    units = plan.get("units", [])
    en_items = [(i, u) for i, u in enumerate(units) if u.get("lang") == "en"]
    if not en_items:
        return {
            "n_en_words": 0,
            "n_func_words": 0,
            "n_content_words": 0,
            "func_median_dur_ms": None,
            "content_median_dur_ms": None,
            "func_to_content_dur_ratio": None,
            "content_minus_func_db": None,
            "content_minus_func_f0_st": None,
        }

    xn = _normalize_active_rms(x, target_dbfs=-20.0)
    track = compute_f0_track(xn, sr=sr, fmin=fmin, fmax=fmax)
    f0_hz = track["f0_hz"]
    voiced = track["voiced"]
    t_s = track["times_s"]
    v_f0 = f0_hz[voiced & (f0_hz > 0.0)]
    f0_med = float(np.median(v_f0)) if v_f0.size else 150.0

    use_actual = unit_timings_ms is not None and len(unit_timings_ms) == len(units)
    timings = unit_timings_ms if use_actual else _estimate_slot_timings_ms(xn, plan, sr=sr)

    func_durs, cont_durs = [], []
    func_dbs, cont_dbs = [], []
    func_f0s, cont_f0s = [], []

    for i, u in en_items:
        word = str(u.get("token") or u.get("hanzi") or "").strip().lower()
        is_func = word in EN_FUNCTION_WORDS
        t0_ms, t1_ms = timings[i]
        dur_ms = (t1_ms - t0_ms) if use_actual else float(u.get("duration_ms", t1_ms - t0_ms))
        s0 = max(0, int(round(t0_ms * 0.001 * sr)))
        s1 = min(len(xn), max(s0 + 1, int(round(t1_ms * 0.001 * sr))))
        seg = xn[s0:s1]
        seg_db = 20.0 * math.log10(max(1e-10, float(np.sqrt(np.mean(seg * seg))))) if seg.size else -80.0

        f_mask = (t_s >= t0_ms * 0.001) & (t_s <= t1_ms * 0.001) & voiced & (f0_hz > 0.0)
        seg_f0_st = float(np.median(12.0 * np.log2(f0_hz[f_mask] / f0_med))) if np.any(f_mask) else None

        if is_func:
            func_durs.append(dur_ms)
            func_dbs.append(seg_db)
            if seg_f0_st is not None:
                func_f0s.append(seg_f0_st)
        else:
            cont_durs.append(dur_ms)
            cont_dbs.append(seg_db)
            if seg_f0_st is not None:
                cont_f0s.append(seg_f0_st)

    f_med = float(np.median(func_durs)) if func_durs else None
    c_med = float(np.median(cont_durs)) if cont_durs else None
    ratio = (f_med / c_med) if (f_med is not None and c_med and c_med > 1e-3) else None
    db_diff = (float(np.median(cont_dbs)) - float(np.median(func_dbs))) if (func_dbs and cont_dbs) else None
    f0_diff = (float(np.median(cont_f0s)) - float(np.median(func_f0s))) if (func_f0s and cont_f0s) else None

    return {
        "n_en_words": len(en_items),
        "n_func_words": len(func_durs),
        "n_content_words": len(cont_durs),
        "func_median_dur_ms": round(f_med, 2) if f_med is not None else None,
        "content_median_dur_ms": round(c_med, 2) if c_med is not None else None,
        "func_to_content_dur_ratio": round(ratio, 4) if ratio is not None else None,
        "content_minus_func_db": round(db_diff, 3) if db_diff is not None else None,
        "content_minus_func_f0_st": round(f0_diff, 3) if f0_diff is not None else None,
    }


def evaluate_v11_utterance(
    x: np.ndarray,
    plan: dict[str, Any],
    sr: int = DEFAULT_SR,
    *,
    fmin: float = 75.0,
    fmax: float = 450.0,
    slot_timings_ms: list[tuple[float, float]] | None = None,
    unit_boundaries_ms: list[float] | None = None,
    log_env: np.ndarray | None = None,
    boundary_frames: list[int] | None = None,
) -> dict[str, Any]:
    """Unified Phase 7 (v11) utterance quality summary."""
    sanity = check_wav_sanity(x, sr)
    v9 = compute_v9_extended_metrics(x, sr=sr, fmin=fmin, fmax=fmax)
    loudness = measure_windowed_loudness(x, sr=sr)
    nondrop = detect_dropped_slots(x, plan, sr=sr, slot_timings_ms=slot_timings_ms)
    smoothness = measure_boundary_smoothness(
        x, sr=sr, unit_boundaries_ms=unit_boundaries_ms, log_env=log_env, boundary_frames=boundary_frames
    )
    rhythm = measure_en_prosody_rhythm(x, plan, sr=sr, unit_timings_ms=slot_timings_ms, fmin=fmin, fmax=fmax)
    return {
        "sanity_ok": bool(sanity["ok"]),
        "loudness_jumps": loudness,
        "nondrop": nondrop,
        "boundary_smoothness": smoothness,
        "en_rhythm": rhythm,
        "v9": v9,
    }
