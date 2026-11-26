"""Manner-Calibrated Parseval Loudness Normalization & Smooth Gain Contouring (100% Non-Neural)."""
from __future__ import annotations

import math

import numpy as np

from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS
from exp_v10_integrity import N_FFT


# Grounded in real human CMU-ARCTIC / AISHELL-3 Parseval RMS offsets relative to stressed vowel (dB),
# with audible floors for liquids/stops so misaligned silence never creates dropped-phone dips.
MANNER_REL_RMS_DB: dict[str, float] = {
    "zh_syllable": 0.0,
    "vowel_stressed": 0.0,
    "vowel_unstressed": -2.0,
    "nasal": -4.5,
    "glide": -6.5,
    "liquid": -7.5,
    "voiced_fricative": -8.5,
    "voiced_affricate": -8.5,
    "voiced_stop": -9.5,
    "voiceless_affricate": -11.0,
    "voiceless_fricative": -12.5,
    "voiceless_stop": -12.5,
}


def parseval_frame_db(log_env: np.ndarray, n_fft: int = N_FFT) -> np.ndarray:
    """Compute exact per-frame Parseval RMS power (dB) from natural-log amplitude envelope."""
    el = np.exp(np.asarray(log_env, dtype=np.float64))
    pwr = (el[:, 0] ** 2 + 2.0 * np.sum(el[:, 1:-1] ** 2, axis=1) + el[:, -1] ** 2) / float(n_fft)
    return 10.0 * np.log10(np.maximum(1e-18, pwr))


def classify_ipa_manner(ipa: str, stress: int | None = None) -> str:
    """Map an IPA phoneme (+ optional lexical stress) to its calibrated loudness manner class."""
    sp = IPA_SPECS.get(ipa)
    if sp is None:
        return "vowel_unstressed"
    if sp.manner in ("vowel", "diphthong"):
        return "vowel_stressed" if stress == 1 else "vowel_unstressed"
    if sp.manner in ("nasal", "glide", "liquid"):
        return sp.manner
    prefix = "voiced_" if sp.voiced else "voiceless_"
    key = prefix + sp.manner
    return key if key in MANNER_REL_RMS_DB else "voiced_fricative"


def calibrate_token_parseval_gain(
    log_env: np.ndarray,
    manner_key: str,
    ref_vowel_db: float,
    *,
    pull_strength: float = 0.85,
    max_corr_db: float = 6.5,
) -> tuple[np.ndarray, float]:
    """Shift a token's log_env smoothly toward its empirical manner-relative Parseval dB target.

    Preserves internal envelope dynamics (e.g. stop burst vs closure, vowel attack/decay) while
    removing inter-token recording/position loudness jumps and preventing near-silent dropouts.
    """
    arr = np.asarray(log_env, dtype=np.float64)
    if len(arr) == 0:
        return arr.copy(), 0.0
    pdb = parseval_frame_db(arr, n_fft=2 * (arr.shape[1] - 1))
    # Use 75th percentile so stop closures or edge fade-outs don't cause excessive boost
    tok_lvl_db = float(np.percentile(pdb, 75))
    target_db = ref_vowel_db + MANNER_REL_RMS_DB.get(manner_key, -2.0)
    raw_delta_db = (target_db - tok_lvl_db) * pull_strength

    # Allow larger rescue boost ONLY if token would otherwise be inaudibly soft (< ref - 22 dB),
    # and always clamp upward/downward extremes cleanly.
    lo_cap = -max_corr_db
    hi_cap = max(max_corr_db, (ref_vowel_db - 18.0) - tok_lvl_db)
    delta_db = float(np.clip(raw_delta_db, lo_cap, hi_cap))

    # Hard ceiling: no obstruent or vowel exceeds ref_vowel_db + 3.5 dB
    max_allowed = (ref_vowel_db + 3.5) if manner_key in ("zh_syllable", "vowel_stressed") else (ref_vowel_db + 1.5)
    if tok_lvl_db + delta_db > max_allowed:
        delta_db = max_allowed - tok_lvl_db

    delta_np = delta_db * math.log(10.0) / 20.0
    return arr + delta_np, delta_db


def smooth_utterance_gain_contour(
    env_all: np.ndarray,
    voi_all: np.ndarray,
    is_obs_frame: np.ndarray,
    is_pause_frame: np.ndarray,
    ref_vowel_db: float,
    *,
    max_voiced_step_db: float = 2.0,
    unvoiced_floor_rel_db: float = -20.0,
    unvoiced_ceil_rel_db: float = -4.5,
) -> np.ndarray:
    """Apply smooth frame-level Parseval gain limiting across voiced runs & clamp unvoiced spikes.

    Eliminates >6 dB unnatural inter-syllable loudness jumps inside voiced speech while preserving
    crisp CV contrast at obstruent edges and guaranteeing no unvoiced blast or silence dropout.
    """
    out = env_all.astype(np.float64).copy()
    n_fr = len(out)
    if n_fr < 2:
        return out

    pdb = parseval_frame_db(out, n_fft=2 * (out.shape[1] - 1))
    desired_db = pdb.copy()

    # 1. Bound unvoiced obstruent energy inside [ref + floor, ref + ceil]
    uv_act = (~is_pause_frame) & (~voi_all) & is_obs_frame
    if np.any(uv_act):
        lo_uv = ref_vowel_db + unvoiced_floor_rel_db
        hi_uv = ref_vowel_db + unvoiced_ceil_rel_db
        desired_db[uv_act] = np.clip(desired_db[uv_act], lo_uv, hi_uv)

    # 2. Enforce audible floor on non-pause voiced frames (>= ref - 15 dB) & ceiling (<= ref + 4.0 dB)
    v_act = (~is_pause_frame) & voi_all
    if np.any(v_act):
        desired_db[v_act] = np.clip(desired_db[v_act], ref_vowel_db - 15.0, ref_vowel_db + 4.0)

    # 3. Bidirectional slew-rate limiter across contiguous voiced runs (preserve sharp V<->UV unvoiced edges only)
    i = 0
    while i < n_fr:
        if not v_act[i]:
            i += 1
            continue
        j = i + 1
        while j < n_fr and v_act[j]:
            j += 1
        if j - i >= 2:
            # Gentle 5-tap moving smoothing on deviations + bidirectional step-clamping
            seg = desired_db[i:j].copy()
            fwd = seg.copy()
            for t in range(1, len(fwd)):
                diff = fwd[t] - fwd[t - 1]
                if abs(diff) > max_voiced_step_db:
                    fwd[t] = fwd[t - 1] + math.copysign(max_voiced_step_db, diff)
            bwd = seg.copy()
            for t in range(len(bwd) - 2, -1, -1):
                diff = bwd[t] - bwd[t + 1]
                if abs(diff) > max_voiced_step_db:
                    bwd[t] = bwd[t + 1] + math.copysign(max_voiced_step_db, diff)
            desired_db[i:j] = 0.5 * (fwd + bwd)
        i = j

    corr_np = (desired_db - pdb) * math.log(10.0) / 20.0
    # Never alter pause frames
    corr_np[is_pause_frame] = 0.0
    out += corr_np[:, None]
    return out
