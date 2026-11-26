"""Unified Phase-7 Adaptive Timbre Cloner (`TimbreClonerV5`).

Combines:
- `TimbreClonerV3`'s cross-corpus `[250, 7200] Hz` EQ decoupling & controlled tilt anchor
  (solving `bdl <-> SSB0073` and `ZH <-> EN` corpus/bandwidth mismatches)
- `TimbreClonerV4`'s intra-corpus continuous envelope & true-RMS voiced/unvoiced loudness
  calibration (preserving `F->M` spectral tilt and blocking 8-12 dB unvoiced jumps)
- Compact `< 2 KB` UTAU (`GEN / OPE / BRI / BRE`) + acoustic manner controls.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
from scipy.fft import rfftfreq

ROOT_IMPORT = Path(__file__).resolve().parent.parent
if str(ROOT_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_IMPORT))

from exp_v6_vocoder import cloner as v6_cloner, vocoder as v6_voc
from exp_v8_timbre_clone.timbre_cloner_v2 import estimate_auto_vtln
from exp_v9_timbre_v3.eq_decoupler import DB_PER_NP, decompose_envelope_eq, linear_tilt_fit, normalize_env_shared_band
from exp_v9_timbre_v3.timbre_cloner_v3 import (
    MANNER_NAMES,
    _apply_bri_contrast,
    _ceps_decode,
    classify_acoustic_manner,
)
from exp_v10_g2p_and_align.timbre_cloner_v4 import (
    TimbreProfileV4,
    _detect_cross_corpus,
    calibrate_hnm_residual_gain,
    extract_timbre_v4,
)
from exp_v11_align_gmm import SR

TimbreProfileV5 = TimbreProfileV4


def extract_timbre_v5(
    items: list[tuple[np.ndarray, v6_voc.VocoderFeatures]] | list[np.ndarray],
    sr: int = SR,
    name: str = "target_v5",
    corpus: str = "",
) -> TimbreProfileV5:
    """Extract unified V5 profile (wraps `< 2 KB` V3 UTAU descriptor + true V/U RMS & 513-bin intra env)."""
    return extract_timbre_v4(items, sr=sr, name=name, corpus=corpus)


def adapt_features_v5(
    src_feats: v6_voc.VocoderFeatures,
    src_profile: TimbreProfileV5,
    tgt_profile: TimbreProfileV5,
    *,
    vtln_alpha: float | None = None,
    eq_mode: str = "auto",
    src_corpus_env: np.ndarray | None = None,
    tgt_corpus_env: np.ndarray | None = None,
    corpus_eq_keep: float = 0.70,
    use_manner_delta: bool = True,
    use_utau_colors: bool = True,
    max_vu_shift_db: float = 2.0,
) -> tuple[v6_voc.VocoderFeatures, dict]:
    """Adapt HNM/WORLD features with automatic cross-corpus EQ decoupling + V/U RMS calibration."""
    sr = src_feats.sr
    n_fft = src_feats.n_fft
    freqs = rfftfreq(n_fft, 1.0 / sr)
    v_mask = src_feats.voiced & (src_feats.f0_hz > 0.0)
    u_mask = ~v_mask

    src3 = src_profile.v3
    tgt3 = tgt_profile.v3

    is_cross = _detect_cross_corpus(src_profile, tgt_profile) if eq_mode == "auto" else (eq_mode != "intra_decoupled")
    if is_cross:
        active_mode = (
            "corpus_eq_split"
            if (src_corpus_env is not None and tgt_corpus_env is not None)
            else "blind_tilt_anchor"
        )
    else:
        active_mode = "intra_decoupled"

    # 1. F0 log-z-score transfer
    f0_out = np.zeros_like(src_feats.f0_hz)
    if np.any(v_mask):
        ratio_std = max(0.55, min(1.85, tgt3.f0_log_std / max(0.03, src3.f0_log_std)))
        log_f0_src = np.log(src_feats.f0_hz[v_mask])
        log_f0_tgt = math.log(tgt3.f0_median_hz) + ratio_std * (log_f0_src - math.log(src3.f0_median_hz))
        f0_out[v_mask] = np.clip(np.exp(log_f0_tgt), 65.0, 440.0)

    # 2. UTAU GEN (data-driven F1/F2/F3 VTLN strictly on voiced frames)
    vtln_info = estimate_auto_vtln(src3.to_v2_stub(), tgt3.to_v2_stub())
    alpha = float(vtln_info["alpha"]) if vtln_alpha is None else float(vtln_alpha)

    env_out = src_feats.log_env.copy()
    if abs(alpha - 1.0) >= 1e-4 and np.any(v_mask):
        env_out[v_mask] = v6_cloner._vtln_warp_2d(src_feats.log_env[v_mask], alpha, sr=sr, n_fft=n_fft)

    if np.any(v_mask):
        if active_mode == "intra_decoupled":
            src_v = src_profile.voiced_env_raw513
            if abs(alpha - 1.0) >= 1e-4:
                src_v = v6_cloner._vtln_warp_2d(src_v[None, :], alpha, sr=sr, n_fft=n_fft)[0]
            raw_diff = tgt_profile.voiced_env_raw513 - src_v
            base_curve = np.clip(v6_cloner._cepstral_smooth_1d(raw_diff, n_fft=n_fft, q_max=20), -1.15, 1.15)
        else:
            src_v = src3.voiced_env_mean
            warp_tilt_delta_db = 0.0
            if abs(alpha - 1.0) >= 1e-4:
                src_v = v6_cloner._vtln_warp_2d(src_v[None, :], alpha, sr=sr, n_fft=n_fft)[0]
                s_unw, _ = linear_tilt_fit(normalize_env_shared_band(src3.voiced_env_mean, freqs), freqs)
                s_wrp, _ = linear_tilt_fit(normalize_env_shared_band(src_v, freqs), freqs)
                warp_tilt_delta_db = (s_wrp - s_unw) * DB_PER_NP
            base_curve, _ = decompose_envelope_eq(
                src_env=src_v,
                tgt_env=tgt3.voiced_env_mean,
                freqs=freqs,
                src_tilt_db_oct=src3.tilt_db_per_oct + warp_tilt_delta_db,
                tgt_tilt_db_oct=tgt3.tilt_db_per_oct,
                n_fft=n_fft,
                q_timbre=18,
                tilt_anchor_weight=0.85,
                eq_mode=active_mode,
                src_corpus_env=src_corpus_env,
                tgt_corpus_env=tgt_corpus_env,
                corpus_eq_keep=corpus_eq_keep,
            )

        if use_utau_colors:
            d_ope_db = float(np.clip(tgt3.ope_h1_h2_db - src3.ope_h1_h2_db, -4.5, 4.5))
            u_ope = np.clip((freqs - 400.0) / 1100.0, 0.0, 1.0)
            ope_shelf = (1.0 - 0.5 * (1.0 - np.cos(math.pi * u_ope))) * float(np.clip(0.18 * d_ope_db / DB_PER_NP, -0.16, 0.16))
            base_curve = np.clip(base_curve + ope_shelf, -1.20, 1.20)

        if tgt3.name.startswith("male_") and tgt3.f0_median_hz < 0.85 * src3.f0_median_hz:
            f_cut = max(250.0, 1.3 * tgt3.f0_median_hz)
            base_curve[freqs < f_cut] = np.minimum(base_curve[freqs < f_cut], 0.0)

        pre_v_pwr = np.sum(np.exp(2.0 * env_out[v_mask]), axis=1)
        env_shifted = env_out[v_mask] + base_curve[None, :]

        if use_manner_delta:
            m_labels = classify_acoustic_manner(src_feats.log_env, src_feats.voiced, src_feats.f0_hz, freqs)
            v_labels = m_labels[v_mask]
            for cls_id, cls_name in zip((1, 2, 3), MANNER_NAMES):
                cls_pos = np.where(v_labels == cls_id)[0]
                if cls_pos.size == 0:
                    continue
                n_eff = min(src3.manner_counts.get(cls_name, 0), tgt3.manner_counts.get(cls_name, 0))
                w_c = float(n_eff / (n_eff + 350.0)) if n_eff > 0 else 0.0
                if w_c < 0.03:
                    continue
                d_src_c = _ceps_decode(src3.manner_deltas_q14[cls_name], n_fft=n_fft)
                if abs(alpha - 1.0) >= 1e-4:
                    d_src_c = v6_cloner._vtln_warp_2d(d_src_c[None, :], alpha, sr=sr, n_fft=n_fft)[0]
                d_tgt_c = _ceps_decode(tgt3.manner_deltas_q14[cls_name], n_fft=n_fft)
                extra_c = w_c * np.clip(normalize_env_shared_band(d_tgt_c - d_src_c, freqs), -0.38, 0.38)
                env_shifted[cls_pos] = env_shifted[cls_pos] + extra_c[None, :]

        if use_utau_colors:
            n_bri = min(src3.n_formant_frames, tgt3.n_formant_frames)
            w_bri = n_bri / (n_bri + 320.0) if n_bri > 0 else 0.0
            bri_ratio = 1.0 + w_bri * (tgt3.bri_formant_sharp_db / max(1.0, src3.bri_formant_sharp_db) - 1.0)
            env_shifted = _apply_bri_contrast(env_shifted, bri_ratio=bri_ratio, n_fft=n_fft)

        post_v_pwr = np.sum(np.exp(2.0 * env_shifted), axis=1)
        gain_v = 0.5 * np.log(np.maximum(pre_v_pwr, 1e-18) / np.maximum(post_v_pwr, 1e-18))
        env_out[v_mask] = env_shifted + gain_v[:, None]

    # 3b. Unvoiced consonant preservation with per-frame RMS calibration & tight ±2 dB V-U shift
    applied_vu_shift_db = 0.0
    if np.any(u_mask):
        u_ramp = np.clip((freqs - 800.0) / 800.0, 0.0, 1.0)
        hi_taper = np.where(freqs > 7200.0, np.clip(1.0 - (freqs - 7200.0) / 1800.0, 0.0, 1.0), 1.0)
        d_u_slope = float(np.clip(tgt3.unvoiced_tilt_slope - src3.unvoiced_tilt_slope, -0.35, 0.35))
        u_shift = u_ramp * hi_taper * d_u_slope * np.log2(np.clip(freqs, 1000.0, 7200.0) / 3000.0)
        u_shift = np.clip(u_shift, -0.30, 0.30)

        pre_u_pwr = np.sum(np.exp(2.0 * env_out[u_mask]), axis=1)
        u_shaped = env_out[u_mask] + u_shift[None, :]
        post_u_pwr = np.sum(np.exp(2.0 * u_shaped), axis=1)
        gain_u_preserve = 0.5 * np.log(np.maximum(pre_u_pwr, 1e-18) / np.maximum(post_u_pwr, 1e-18))

        gap_delta_db = float(np.clip(0.30 * (tgt_profile.vu_rms_gap_db - src_profile.vu_rms_gap_db), -max_vu_shift_db, max_vu_shift_db))
        applied_vu_shift_db = -gap_delta_db
        env_out[u_mask] = u_shaped + gain_u_preserve[:, None] + (applied_vu_shift_db / DB_PER_NP)

    # 4. UTAU BRE (logit BAP transfer on voiced frames ONLY)
    bap_out = src_feats.bap.copy()
    if np.any(v_mask):
        delta_logit = v6_cloner._logit(np.asarray(tgt3.bap_voiced_median, dtype=np.float64)) - v6_cloner._logit(
            np.asarray(src3.bap_voiced_median, dtype=np.float64)
        )
        bap_out[v_mask] = np.clip(
            v6_cloner._sigmoid(v6_cloner._logit(src_feats.bap[v_mask]) + delta_logit[None, :]),
            0.005,
            0.985,
        )

    out_feats = v6_voc.VocoderFeatures(
        f0_hz=f0_out,
        voiced=src_feats.voiced.copy(),
        log_env=env_out,
        bap=bap_out,
        sr=sr,
        hop_samples=src_feats.hop_samples,
        n_samples=src_feats.n_samples,
        n_fft=n_fft,
    )
    return out_feats, {
        "alpha": round(alpha, 4),
        "routing_mode": active_mode,
        "applied_vu_shift_db": round(applied_vu_shift_db, 3),
        "profile_bytes": tgt_profile.serialized_size_bytes(),
    }


def adapt_features_v5_only(
    src_feats: v6_voc.VocoderFeatures,
    src_profile: TimbreProfileV5,
    tgt_profile: TimbreProfileV5,
) -> v6_voc.VocoderFeatures:
    """Thin wrapper returning VocoderFeatures directly."""
    feats, _ = adapt_features_v5(src_feats, src_profile, tgt_profile)
    return feats
