"""Phase 9 (v13) UTAU Deformation: Steady-Nucleus Stretching, Ping-Pong Sustains, Gentle Gain, Seam Join."""
from __future__ import annotations

import math

import numpy as np
from scipy.fft import rfftfreq

from exp_v8_clarity.utau_concat import _resample_1d, _resample_2d
from exp_v10_integrity.gain_calibration import MANNER_REL_RMS_DB, parseval_frame_db
from exp_v13_utau_ceiling import N_FFT, SR

VT_BIN_LO = 3
VT_BIN_HI = 308


def find_steady_vowel_nucleus(log_env_rime: np.ndarray, voiced_rime: np.ndarray) -> tuple[int, int]:
    """Locate the lowest spectral-flux voiced steady vowel nucleus `[nuc_s, nuc_e)` inside a rime."""
    n = len(log_env_rime)
    if n < 8 or int(np.sum(voiced_rime)) < 5:
        return 0, n

    # Protect initial CV glide transition and final nasal/coda tail
    guard_l = min(3, max(1, n // 6))
    guard_r = min(3, max(1, n // 6))
    lo = guard_l
    hi = n - guard_r
    if hi - lo < 4:
        return 0, n

    sub = log_env_rime[:, VT_BIN_LO:VT_BIN_HI].astype(np.float64)
    sub = sub - np.mean(sub, axis=1, keepdims=True)
    # Per-frame transition flux score (average of incoming and outgoing spectral change)
    diffs = np.sqrt(np.mean(np.diff(sub, axis=0) ** 2, axis=1))
    frame_flux = np.zeros(n, dtype=np.float64)
    frame_flux[1:] += 0.5 * diffs
    frame_flux[:-1] += 0.5 * diffs
    frame_flux[0] += 0.5 * diffs[0]
    frame_flux[-1] += 0.5 * diffs[-1]

    # Prefer voiced frames inside [lo, hi) with lowest spectral flux
    w = max(3, min(hi - lo, int(round(0.48 * (hi - lo)))))
    best_cost = float("inf")
    best_span = (lo, lo + w)
    for s in range(lo, hi - w + 1):
        v_frac = float(np.mean(voiced_rime[s : s + w]))
        if v_frac < 0.75:
            continue
        cost = float(np.mean(frame_flux[s : s + w])) - 0.15 * v_frac
        if cost < best_cost:
            best_cost = cost
            best_span = (s, s + w)

    if best_cost == float("inf"):
        return lo, hi
    return best_span


def _pingpong_indices(len_src: int, n_out: int, xfade: int = 2) -> np.ndarray:
    """Generate smooth floating-point palindrome (forward-backward-forward) frame indices."""
    if len_src <= 1 or n_out <= 1:
        return np.zeros(max(1, n_out), dtype=np.float64)
    if n_out <= len_src:
        return np.linspace(0.0, float(len_src - 1), n_out)

    # Palindrome bounce between 0 and len_src - 1 without repeating endpoints abruptly
    fwd = np.arange(len_src, dtype=np.float64)
    if len_src >= 3:
        bwd = np.arange(len_src - 2, 0, -1, dtype=np.float64)
        cycle = np.concatenate([fwd, bwd])
    else:
        cycle = fwd

    # Repeat full palindrome cycles; map n_out smoothly while clamping endpoints to 0 and len_src-1
    # First half -> sustained middle ping-pong -> clean landing on final frame
    n_cyc = max(1, int(math.ceil(n_out / max(1, len(cycle)))))
    tiled = np.tile(cycle, n_cyc)
    # Ensure exit connects cleanly toward the post-nucleus tail by sampling full or half-cycles
    idx_grid = np.linspace(0.0, float(len(tiled) - 1), n_out)
    # Map grid onto cycle positions with endpoint anchored cleanly
    pos = np.interp(idx_grid, np.arange(len(tiled), dtype=np.float64), tiled)
    if xfade > 0 and n_out >= 6:
        # Smooth turnaround corners gently
        sm = pos.copy()
        sm[1:-1] = 0.25 * pos[:-2] + 0.50 * pos[1:-1] + 0.25 * pos[2:]
        pos = sm
    return np.clip(pos, 0.0, float(len_src - 1))


def _sample_by_float_indices_2d(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    k = len(arr)
    if k == 0:
        return np.zeros((len(pos), arr.shape[1]), dtype=np.float64)
    if k == 1:
        return np.repeat(arr.astype(np.float64), len(pos), axis=0)
    idx_lo = np.floor(pos).astype(int)
    idx_hi = np.minimum(k - 1, idx_lo + 1)
    w = (pos - idx_lo)[:, None]
    return (1.0 - w) * arr[idx_lo].astype(np.float64) + w * arr[idx_hi].astype(np.float64)


def _sample_by_float_indices_1d(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    k = len(arr)
    if k == 0:
        return np.zeros(len(pos), dtype=np.float64)
    if k == 1:
        return np.repeat(arr.astype(np.float64), len(pos))
    idx_lo = np.floor(pos).astype(int)
    idx_hi = np.minimum(k - 1, idx_lo + 1)
    w = pos - idx_lo
    return (1.0 - w) * arr[idx_lo].astype(np.float64) + w * arr[idx_hi].astype(np.float64)


def stretch_utau_steady_nucleus(
    env: np.ndarray,
    bap: np.ndarray,
    bap_gt8k: np.ndarray,
    voi: np.ndarray,
    f0_rel: np.ndarray,
    f_split: int,
    n_target: int,
    *,
    target_onset_frames: int | None = None,
    allow_pingpong: bool = True,
    pingpong_thresh_ratio: float = 1.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """UTAU-style time scaling: preserve onset, glide head & coda tail; stretch/loop steady nucleus."""
    n_src = len(env)
    n_target = max(3, int(n_target))
    f_sp = max(0, min(n_src - 2, int(f_split)))

    # 1. Determine onset frame count
    if f_sp > 0:
        if target_onset_frames is not None:
            lo_on = max(1, int(round(0.60 * f_sp)))
            hi_on = min(n_target - 2, max(lo_on, int(round(1.45 * f_sp))))
            n_on = int(np.clip(int(target_onset_frames), lo_on, hi_on))
        else:
            n_on = f_sp
        if n_on > n_target - 2:
            n_on = max(1, n_target - 2)
    else:
        n_on = 0

    n_rime_target = n_target - n_on
    e_on = _resample_2d(env[:f_sp], n_on) if n_on > 0 else np.empty((0, env.shape[1]), dtype=np.float64)
    b_on = np.clip(_resample_2d(bap[:f_sp], n_on), 0.005, 0.995) if n_on > 0 else np.empty((0, bap.shape[1]), dtype=np.float64)
    bg_on = np.clip(_resample_2d(bap_gt8k[:f_sp], n_on), 0.005, 0.995) if n_on > 0 else np.empty((0, bap_gt8k.shape[1]), dtype=np.float64)
    v_on = (_resample_1d(voi[:f_sp].astype(np.float64), n_on) >= 0.45) if n_on > 0 else np.empty(0, dtype=bool)
    f0_on = _resample_1d(f0_rel[:f_sp], n_on) if n_on > 0 else np.empty(0, dtype=np.float64)

    # 2. Split rime into [pre_glide | steady_nucleus | post_tail]
    e_ri = env[f_sp:]
    b_ri = bap[f_sp:]
    bg_ri = bap_gt8k[f_sp:]
    v_ri = voi[f_sp:]
    f0_ri = f0_rel[f_sp:]
    n_ri_src = len(e_ri)

    nuc_s, nuc_e = find_steady_vowel_nucleus(e_ri, v_ri)
    n_pre = nuc_s
    n_post = n_ri_src - nuc_e
    n_nuc_src = nuc_e - nuc_s

    used_pingpong = False
    if n_nuc_src < 3 or n_rime_target <= n_pre + n_post + 2:
        # Too short to preserve unstretched guards cleanly: uniform rime scaling
        e_ri_out = _resample_2d(e_ri, n_rime_target)
        b_ri_out = np.clip(_resample_2d(b_ri, n_rime_target), 0.005, 0.995)
        bg_ri_out = np.clip(_resample_2d(bg_ri, n_rime_target), 0.005, 0.995)
        v_ri_out = _resample_1d(v_ri.astype(np.float64), n_rime_target) >= 0.45
        f0_ri_out = _resample_1d(f0_ri, n_rime_target)
    else:
        n_nuc_target = n_rime_target - n_pre - n_post
        ratio = n_nuc_target / float(max(1, n_nuc_src))
        if allow_pingpong and ratio > pingpong_thresh_ratio and n_nuc_src >= 4 and np.all(v_ri[nuc_s:nuc_e]):
            pos = _pingpong_indices(n_nuc_src, n_nuc_target)
            e_nuc = _sample_by_float_indices_2d(e_ri[nuc_s:nuc_e], pos)
            b_nuc = np.clip(_sample_by_float_indices_2d(b_ri[nuc_s:nuc_e], pos), 0.005, 0.995)
            bg_nuc = np.clip(_sample_by_float_indices_2d(bg_ri[nuc_s:nuc_e], pos), 0.005, 0.995)
            v_nuc = np.ones(n_nuc_target, dtype=bool)
            f0_nuc = _sample_by_float_indices_1d(f0_ri[nuc_s:nuc_e], pos)
            used_pingpong = True
        else:
            e_nuc = _resample_2d(e_ri[nuc_s:nuc_e], n_nuc_target)
            b_nuc = np.clip(_resample_2d(b_ri[nuc_s:nuc_e], n_nuc_target), 0.005, 0.995)
            bg_nuc = np.clip(_resample_2d(bg_ri[nuc_s:nuc_e], n_nuc_target), 0.005, 0.995)
            v_nuc = _resample_1d(v_ri[nuc_s:nuc_e].astype(np.float64), n_nuc_target) >= 0.45
            f0_nuc = _resample_1d(f0_ri[nuc_s:nuc_e], n_nuc_target)

        e_ri_out = np.concatenate([e_ri[:n_pre].astype(np.float64), e_nuc, e_ri[nuc_e:].astype(np.float64)], axis=0)
        b_ri_out = np.clip(np.concatenate([b_ri[:n_pre].astype(np.float64), b_nuc, b_ri[nuc_e:].astype(np.float64)], axis=0), 0.005, 0.995)
        bg_ri_out = np.clip(np.concatenate([bg_ri[:n_pre].astype(np.float64), bg_nuc, bg_ri[nuc_e:].astype(np.float64)], axis=0), 0.005, 0.995)
        v_ri_out = np.concatenate([v_ri[:n_pre].astype(bool), v_nuc, v_ri[nuc_e:].astype(bool)], axis=0)
        f0_ri_out = np.concatenate([f0_ri[:n_pre].astype(np.float64), f0_nuc, f0_ri[nuc_e:].astype(np.float64)], axis=0)

    e_all = np.concatenate([e_on, e_ri_out], axis=0) if n_on > 0 else e_ri_out
    b_all = np.concatenate([b_on, b_ri_out], axis=0) if n_on > 0 else b_ri_out
    bg_all = np.concatenate([bg_on, bg_ri_out], axis=0) if n_on > 0 else bg_ri_out
    v_all = np.concatenate([v_on, v_ri_out], axis=0) if n_on > 0 else v_ri_out
    f0_all = np.concatenate([f0_on, f0_ri_out], axis=0) if n_on > 0 else f0_ri_out
    return e_all, b_all, bg_all, v_all, f0_all, n_on, used_pingpong


def calibrate_gain_gentle(
    log_env: np.ndarray,
    manner_key: str,
    ref_vowel_db: float,
    *,
    pull_strength: float = 0.30,
    max_corr_db: float = 4.0,
) -> tuple[np.ndarray, float]:
    """Dynamics-preserving gain calibration that fixes severe recording outliers without flattening speech."""
    arr = np.asarray(log_env, dtype=np.float64)
    if len(arr) == 0:
        return arr.copy(), 0.0
    pdb = parseval_frame_db(arr, n_fft=2 * (arr.shape[1] - 1))
    tok_lvl_db = float(np.percentile(pdb, 75))
    target_db = ref_vowel_db + MANNER_REL_RMS_DB.get(manner_key, -2.0)
    raw_delta = (target_db - tok_lvl_db) * pull_strength
    lo_cap = -max_corr_db
    hi_cap = max(max_corr_db, (ref_vowel_db - 19.0) - tok_lvl_db)
    delta_db = float(np.clip(raw_delta, lo_cap, hi_cap))
    max_allowed = (ref_vowel_db + 4.5) if manner_key in ("zh_syllable", "vowel_stressed") else (ref_vowel_db + 2.0)
    if tok_lvl_db + delta_db > max_allowed:
        delta_db = max_allowed - tok_lvl_db
    return arr + delta_db * math.log(10.0) / 20.0, delta_db


def smooth_utterance_gain_gentle(
    env_all: np.ndarray,
    voi_all: np.ndarray,
    is_obs_frame: np.ndarray,
    is_pause_frame: np.ndarray,
    ref_vowel_db: float,
    *,
    max_voiced_step_db: float = 3.5,
) -> np.ndarray:
    """Gentle slew-rate limiter preserving natural vowel/tone loudness swings while blocking extreme clicks."""
    from exp_v10_integrity.gain_calibration import smooth_utterance_gain_contour

    return smooth_utterance_gain_contour(
        env_all,
        voi_all,
        is_obs_frame,
        is_pause_frame,
        ref_vowel_db=ref_vowel_db,
        max_voiced_step_db=max_voiced_step_db,
        unvoiced_floor_rel_db=-22.0,
        unvoiced_ceil_rel_db=-3.5,
    )


def join_seam_v13(
    env_a: np.ndarray,
    env_b: np.ndarray,
    bap_a: np.ndarray,
    bap_b: np.ndarray,
    voi_a: np.ndarray,
    voi_b: np.ndarray,
    *,
    obs_cur: bool,
) -> None:
    """Adaptive voiced-only cosine seam join keeping exact block lengths invariant."""
    if len(env_a) < 2 or len(env_b) < 2:
        return
    if obs_cur or not bool(voi_a[-1]) or not bool(voi_b[0]):
        return

    # Measure vocal-tract spectral jump across the boundary
    da = env_a[-1, VT_BIN_LO:VT_BIN_HI].astype(np.float64)
    db = env_b[0, VT_BIN_LO:VT_BIN_HI].astype(np.float64)
    da_n = da - float(np.mean(da))
    db_n = db - float(np.mean(db))
    jump = float(math.sqrt(float(np.mean((da_n - db_n) ** 2))))

    # 1 frame when jump is gentle, 2 frames when larger; never 3 frames (avoids dual-formant smear)
    w = 2 if jump > 0.85 else 1
    w = min(w, len(env_a) // 3, len(env_b) // 3)
    if w < 1:
        return
    # Verify boundary frames are all voiced
    if not (np.all(voi_a[-w:]) and np.all(voi_b[:w])):
        return

    mid_env = 0.5 * (env_a[-1] + env_b[0])
    mid_bap = 0.5 * (bap_a[-1] + bap_b[0])
    if w == 1:
        env_a[-1] = 0.65 * env_a[-1] + 0.35 * mid_env
        env_b[0] = 0.35 * mid_env + 0.65 * env_b[0]
        bap_a[-1] = 0.65 * bap_a[-1] + 0.35 * mid_bap
        bap_b[0] = 0.35 * mid_bap + 0.65 * bap_b[0]
    else:
        u_a = 0.5 * (1.0 - np.cos(math.pi * np.linspace(0.25, 0.50, w)))[:, None]
        env_a[-w:] = (1.0 - u_a) * env_a[-w:] + u_a * mid_env[None, :]
        bap_a[-w:] = (1.0 - u_a) * bap_a[-w:] + u_a * mid_bap[None, :]
        u_b = 0.5 * (1.0 - np.cos(math.pi * np.linspace(0.50, 0.75, w)))[:, None]
        env_b[:w] = (1.0 - u_b) * mid_env[None, :] + u_b * env_b[:w]
        bap_b[:w] = (1.0 - u_b) * mid_bap[None, :] + u_b * bap_b[:w]


_TILT_OFFSET_CACHE: dict[tuple[int, int], np.ndarray] = {}


def compensate_f0_shift_tilt(
    log_env: np.ndarray,
    voiced: np.ndarray,
    delta_st: float,
    *,
    slope_db_oct_per_12st: float = 1.0,
    sr: int = SR,
    n_fft: int = N_FFT,
) -> np.ndarray:
    """Gentle F0-coupled vocal-effort spectral tilt compensation when pitch shifts > 2.5 st."""
    if abs(delta_st) <= 2.5 or not np.any(voiced):
        return log_env
    key = (int(sr), int(n_fft))
    if key not in _TILT_OFFSET_CACHE:
        freqs = rfftfreq(n_fft, 1.0 / sr)
        octs = np.log2(np.clip(freqs, 80.0, 8000.0) / 1000.0)
        octs[freqs < 80.0] = math.log2(80.0 / 1000.0)
        octs[freqs > 8000.0] = math.log2(8000.0 / 1000.0)
        _TILT_OFFSET_CACHE[key] = octs
    octs = _TILT_OFFSET_CACHE[key]

    eff_st = math.copysign(min(12.0, abs(delta_st) - 2.5), delta_st)
    tilt_db = np.clip(slope_db_oct_per_12st * (eff_st / 12.0) * octs, -4.0, 4.0)
    tilt_np = tilt_db * math.log(10.0) / 20.0
    out = log_env.copy()
    out[voiced] += tilt_np[None, :]
    return out
