"""Zero-ML voice cloning trajectory adapter for Stage 5 LarynxControl."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.fft import rfftfreq

from exp_20_pybot_larynx import N_BAP, N_BINS, N_FFT, SR
from exp_20_pybot_larynx.clone.profile import VoiceCloneProfile
from exp_20_pybot_larynx.larynx.synth import LarynxControl
from exp_v6_vocoder import cloner as v6_cloner
from exp_v9_timbre_v3.eq_decoupler import (
    DB_PER_NP,
    decompose_envelope_eq,
    linear_tilt_fit,
    normalize_env_shared_band,
)
from exp_v9_timbre_v3.timbre_cloner_v3 import (
    MANNER_NAMES,
    _apply_bri_contrast,
    _ceps_decode,
    classify_acoustic_manner,
)

_VTLN_PIVOT_HZ = 7500.0


def _warp_log_env_pivot(log_env: np.ndarray, alpha: float, *, sr: int = SR, n_fft: int = N_FFT) -> np.ndarray:
    """Vectorized piecewise-linear VTLN spectral warping matching `engine._warp_log_env`."""
    if abs(alpha - 1.0) < 1e-4 or log_env.shape[0] == 0:
        return log_env.copy()
    scale = float(np.clip(alpha, 0.78, 1.22))
    freqs = rfftfreq(n_fft, 1.0 / sr)
    nyq = 0.5 * sr
    if scale <= 1.0:
        f_out_mid = _VTLN_PIVOT_HZ * scale
        f_src_mid = _VTLN_PIVOT_HZ
    else:
        f_out_mid = _VTLN_PIVOT_HZ
        f_src_mid = _VTLN_PIVOT_HZ / scale

    f_src = np.interp(freqs, (0.0, f_out_mid, nyq), (0.0, f_src_mid, nyq))
    bin_pos = np.clip(f_src * (n_fft / float(sr)), 0.0, float(N_BINS - 1))
    i0 = np.floor(bin_pos).astype(np.intp)
    i1 = np.minimum(N_BINS - 1, i0 + 1)
    w = (bin_pos - i0)[None, :]
    return (1.0 - w) * log_env[:, i0] + w * log_env[:, i1]


def estimate_clone_vtln(src: VoiceCloneProfile, tgt: VoiceCloneProfile) -> tuple[float, dict[str, Any]]:
    """Estimate anatomically regularized VTLN scale from F1/F2/F3 ratios + F0 prior."""
    r1 = float(np.clip(tgt.f1_median_hz / max(100.0, src.f1_median_hz), 0.5, 2.0))
    r2 = float(np.clip(tgt.f2_median_hz / max(300.0, src.f2_median_hz), 0.5, 2.0))
    r3 = float(np.clip(tgt.f3_median_hz / max(1200.0, src.f3_median_hz), 0.5, 2.0))
    alpha_fmt = math.exp(0.25 * math.log(r1) + 0.35 * math.log(r2) + 0.40 * math.log(r3))

    f0_st_diff = 12.0 * math.log2(max(50.0, tgt.f0_median_hz) / max(50.0, src.f0_median_hz))
    alpha_prior = float(np.clip((tgt.f0_median_hz / max(50.0, src.f0_median_hz)) ** 0.14, 0.84, 1.14))
    n_eff = min(int(src.n_formant_frames), int(tgt.n_formant_frames))
    w_f = float(n_eff / (n_eff + 180.0)) if n_eff > 0 else 0.0

    if abs(f0_st_diff) < 4.5:
        shrunk = 1.0 + w_f * (alpha_fmt - 1.0)
        alpha = 1.0 if abs(shrunk - 1.0) < 0.04 else float(np.clip(shrunk, 0.84, 1.16))
    else:
        alpha = w_f * alpha_fmt + (1.0 - w_f) * alpha_prior
        if f0_st_diff <= -4.5:
            alpha = min(alpha, 0.955)
        elif f0_st_diff >= 4.5:
            alpha = max(alpha, 1.025)
        alpha = float(np.clip(alpha, 0.78, 1.22))

    return round(alpha, 4), {
        "alpha": round(alpha, 4),
        "alpha_fmt": round(alpha_fmt, 4),
        "alpha_prior": round(alpha_prior, 4),
        "f0_st_diff": round(f0_st_diff, 2),
        "w_fmt": round(w_f, 4),
        "n_eff": n_eff,
    }


def apply_clone_to_control(
    ctrl: LarynxControl,
    src: VoiceCloneProfile,
    tgt: VoiceCloneProfile,
    *,
    strength: float = 0.85,
    clone_f0: bool = True,
    clone_glottal: bool = True,
    base_coherent_phasor: np.ndarray | None = None,
) -> tuple[LarynxControl, np.ndarray | None, dict[str, Any]]:
    """Transform Stage 5 LarynxControl timbre, F0, glottal waveshape/phase, and BAP toward `tgt`."""
    out = ctrl.copy()
    s = float(np.clip(strength, 0.0, 1.0))
    n_fr = len(out.f0_hz)
    if s <= 1e-4 or n_fr == 0:
        return out, base_coherent_phasor, {"strength": 0.0, "alpha": 1.0}

    sr = int(out.sr)
    n_fft = int(out.n_fft)
    freqs = rfftfreq(n_fft, 1.0 / sr)
    v_mask = out.voiced & (out.f0_hz > 40.0)
    u_mask = ~v_mask

    # 1. F0 log-z-score transfer anchored gently to utterance pitch so ZH & EN match target F0 cleanly
    if clone_f0 and np.any(v_mask):
        src_f0_ref = max(50.0, float(src.f0_median_hz))
        tgt_f0_ref = max(55.0, float(tgt.f0_median_hz))
        ratio_std = float(np.clip(tgt.f0_log_std / max(0.05, src.f0_log_std), 0.65, 1.55))
        log_f0_src = np.log(out.f0_hz[v_mask])
        utt_med_log = float(np.median(log_f0_src)) if log_f0_src.size >= 16 else math.log(src_f0_ref)
        anchor_log = 0.30 * math.log(src_f0_ref) + 0.70 * utt_med_log
        s_f0 = min(1.0, s / 0.85)
        tgt_med_log = anchor_log + s_f0 * (math.log(tgt_f0_ref) - anchor_log)
        std_eff = 1.0 + s_f0 * (ratio_std - 1.0)
        out.f0_hz[v_mask] = np.clip(np.exp(tgt_med_log + std_eff * (log_f0_src - anchor_log)), 58.0, 480.0)

    # 2. Regularized VTLN vocal-tract warp on voiced frames ONLY
    alpha_raw, vtln_info = estimate_clone_vtln(src, tgt)
    alpha_eff = 1.0 + s * (alpha_raw - 1.0)
    if abs(alpha_eff - 1.0) >= 1e-4 and np.any(v_mask):
        out.log_env[v_mask] = _warp_log_env_pivot(out.log_env[v_mask], alpha_eff, sr=sr, n_fft=n_fft)

    # 3. Voiced LTAS + adaptive tilt + OPE + 3-manner + BRI + linear-power preservation
    eq_diag: dict[str, Any] = {}
    manner_weights: dict[str, float] = {}
    d_ope_db = 0.0
    if np.any(v_mask):
        src_env_unwarped = src.voiced_env_mean
        src_env_warped = src_env_unwarped
        warp_tilt_delta_db = 0.0
        if abs(alpha_eff - 1.0) >= 1e-4:
            src_env_warped = _warp_log_env_pivot(src_env_unwarped[None, :], alpha_eff, sr=sr, n_fft=n_fft)[0]
            s_unw, _ = linear_tilt_fit(normalize_env_shared_band(src_env_unwarped, freqs), freqs)
            s_wrp, _ = linear_tilt_fit(normalize_env_shared_band(src_env_warped, freqs), freqs)
            warp_tilt_delta_db = (s_wrp - s_unw) * DB_PER_NP

        f0_shift_oct = abs(math.log2(max(50.0, tgt.f0_median_hz) / max(50.0, src.f0_median_hz)))
        w_tilt = float(np.clip(0.55 - 0.35 * min(1.0, f0_shift_oct / 0.85), 0.20, 0.55))

        base_curve, eq_diag = decompose_envelope_eq(
            src_env=src_env_warped,
            tgt_env=tgt.voiced_env_mean,
            freqs=freqs,
            src_tilt_db_oct=src.tilt_db_per_oct + warp_tilt_delta_db,
            tgt_tilt_db_oct=tgt.tilt_db_per_oct,
            n_fft=n_fft,
            q_timbre=18,
            tilt_anchor_weight=w_tilt,
            eq_mode="blind_tilt_anchor",
            clip_lim=1.10,
        )

        # Wideband extension (>7.2 kHz): keep genuine 24 kHz HF timbre differences instead of flat clamp
        if tgt.effective_bw_hz >= 9000.0:
            src_n = normalize_env_shared_band(src_env_warped, freqs)
            tgt_n = normalize_env_shared_band(tgt.voiced_env_mean, freqs)
            wb_sm = v6_cloner._cepstral_smooth_1d(tgt_n - src_n, n_fft=n_fft, q_max=18)
            wb_sm = np.clip(wb_sm, -1.10, 1.10)
            u_wb = np.clip((freqs - 7200.0) / 1400.0, 0.0, 1.0)
            hi_taper = np.where(freqs > 9500.0, np.clip(1.0 - (freqs - 9500.0) / 2200.0, 0.0, 1.0), 1.0)
            w_wb = 0.5 * (1.0 - np.cos(math.pi * u_wb)) * hi_taper
            base_curve = (1.0 - w_wb) * base_curve + w_wb * wb_sm

        # Glottal open quotient shelf (OPE) + LF Rd shift
        d_ope_db = float(np.clip(tgt.ope_h1_h2_db - src.ope_h1_h2_db, -4.5, 4.5))
        if abs(d_ope_db) > 1e-4:
            u_ope = np.clip((freqs - 400.0) / 1100.0, 0.0, 1.0)
            ope_shelf = (1.0 - 0.5 * (1.0 - np.cos(math.pi * u_ope))) * float(
                np.clip(0.18 * d_ope_db / DB_PER_NP, -0.16, 0.16)
            )
            base_curve = np.clip(base_curve + ope_shelf, -1.15, 1.15)
            if out.rd is not None:
                out.rd[v_mask] = np.clip(
                    out.rd[v_mask] + s * float(np.clip(0.035 * d_ope_db, -0.18, 0.22)),
                    0.55,
                    2.05,
                )

        pre_pwr = np.sum(np.exp(2.0 * out.log_env[v_mask]), axis=1)
        env_shifted = out.log_env[v_mask] + (s * base_curve)[None, :]

        # Shrunk 3-manner q=14 offset deltas
        m_labels = classify_acoustic_manner(ctrl.log_env, ctrl.voiced, ctrl.f0_hz, freqs)
        v_labels = m_labels[np.where(v_mask)[0]]
        for cls_id, cls_name in zip((1, 2, 3), MANNER_NAMES):
            cls_pos = np.where(v_labels == cls_id)[0]
            if cls_pos.size == 0:
                manner_weights[cls_name] = 0.0
                continue
            n_eff = min(int(src.manner_counts.get(cls_name, 0)), int(tgt.manner_counts.get(cls_name, 0)))
            w_c = float(n_eff / (n_eff + 260.0)) if n_eff > 0 else 0.0
            manner_weights[cls_name] = round(w_c, 4)
            if w_c < 0.04:
                continue
            d_src_c = _ceps_decode(src.manner_deltas_q14[cls_name], n_fft=n_fft)
            if abs(alpha_eff - 1.0) >= 1e-4:
                d_src_c = _warp_log_env_pivot(d_src_c[None, :], alpha_eff, sr=sr, n_fft=n_fft)[0]
            d_tgt_c = _ceps_decode(tgt.manner_deltas_q14[cls_name], n_fft=n_fft)
            extra_c = s * w_c * np.clip(normalize_env_shared_band(d_tgt_c - d_src_c, freqs), -0.35, 0.35)
            env_shifted[cls_pos] += extra_c[None, :]

        # Formant peak sharpness contrast (BRI, q=8..28)
        n_bri = min(int(src.n_formant_frames), int(tgt.n_formant_frames))
        w_bri = float(n_bri / (n_bri + 320.0)) if n_bri > 0 else 0.0
        raw_bri = tgt.bri_formant_sharp_db / max(1.0, src.bri_formant_sharp_db)
        bri_ratio = 1.0 + s * w_bri * float(np.clip(raw_bri - 1.0, -0.12, 0.12))
        env_shifted = _apply_bri_contrast(env_shifted, bri_ratio=bri_ratio, n_fft=n_fft)

        post_pwr = np.sum(np.exp(2.0 * env_shifted), axis=1)
        gain_adj = 0.5 * np.log(np.maximum(pre_pwr, 1e-18) / np.maximum(post_pwr, 1e-18))
        out.log_env[v_mask] = env_shifted + gain_adj[:, None]

    # 4. Safe unvoiced consonant slope/level alignment (zero VTLN)
    if np.any(u_mask):
        u_ramp = np.clip((freqs - 800.0) / 800.0, 0.0, 1.0)
        hi_taper = np.where(freqs > 7200.0, np.clip(1.0 - (freqs - 7200.0) / 1800.0, 0.0, 1.0), 1.0)
        d_u_slope = float(np.clip(tgt.unvoiced_tilt_slope - src.unvoiced_tilt_slope, -0.35, 0.35))
        u_shift = np.clip(u_ramp * hi_taper * d_u_slope * np.log2(np.clip(freqs, 1000.0, 7200.0) / 3000.0), -0.30, 0.30)
        d_u_lvl = float(np.clip(tgt.unvoiced_rel_level_np - src.unvoiced_rel_level_np, -0.25, 0.25))
        out.log_env[u_mask] = out.log_env[u_mask] + (s * u_shift)[None, :] + (0.35 * s * d_u_lvl)

    # 5. 6-Band Voiced BAP Transfer in logit space
    if np.any(v_mask):
        tgt_bap = np.asarray(tgt.bap6_voiced_median, dtype=np.float64)
        src_bap = np.asarray(src.bap6_voiced_median, dtype=np.float64)
        if len(tgt_bap) == N_BAP and len(src_bap) == N_BAP:
            d_logit = np.clip(v6_cloner._logit(tgt_bap) - v6_cloner._logit(src_bap), -1.4, 1.4)
            out.bap[v_mask] = np.clip(
                v6_cloner._sigmoid(v6_cloner._logit(out.bap[v_mask]) + (0.75 * s * d_logit)[None, :]),
                0.005,
                0.985,
            )

    # 6. Target speaker chest-band coherent glottal phase blend
    eff_phasor = base_coherent_phasor
    w_ph = 0.0
    if clone_glottal and tgt.glottal_phase_frames >= 8:
        tgt_ph = tgt.reconstruct_coherent_phasor(n_fft=n_fft, sr=sr)
        w_ph = s * float(tgt.glottal_phase_frames / (tgt.glottal_phase_frames + 20.0))
        if base_coherent_phasor is not None and len(base_coherent_phasor) == len(tgt_ph):
            b_ph = np.asarray(base_coherent_phasor, dtype=np.complex128)
            b_ph = b_ph / np.maximum(np.abs(b_ph), 1e-8)
            comb = w_ph * tgt_ph + (1.0 - w_ph) * b_ph
            eff_phasor = comb / np.maximum(np.abs(comb), 1e-8)
        else:
            eff_phasor = tgt_ph

    return out, eff_phasor, {
        "strength": round(s, 3),
        "alpha_eff": round(alpha_eff, 4),
        "vtln_info": vtln_info,
        "w_tilt": round(w_tilt if np.any(v_mask) else 0.0, 3),
        "d_ope_db": round(d_ope_db, 3),
        "w_phasor": round(w_ph, 3),
        "manner_weights": manner_weights,
        "eq_diag": eq_diag,
    }
