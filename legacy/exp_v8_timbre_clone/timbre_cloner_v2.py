"""Non-neural decoupled few-shot timbre cloning (Voiced/Unvoiced split + Auto-VTLN + Ridge cepstral).

Addresses the three failure modes observed in ``exp_v6_vocoder.cloner``:
1. Sub-fundamental tilt collapse: applying LTAS delta down to 80 Hz when shifting
   F0 to ~115 Hz double-counts new fundamental energy and steepens 80-8k tilt to -7.5 dB/oct.
   Fix: apply voiced LTAS delta smoothly above ``f_cut = max(250 Hz, 1.3 * F0_tgt)``.
2. Consonant smearing: ``v6`` applied half-strength VTLN to unvoiced fricatives/stops.
   Fix: preserve unvoiced consonant spectra cleanly (bounded gentle HF tilt only).
3. Hardcoded cross-gender VTLN: estimate ``vtln_alpha`` from LPC-22 F1/F2 ratios with
   sample-size shrinkage toward 1.0 so short same-gender prompts don't overfit noise
   while genuine vocal-tract length differences adapt automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_eval import metrics as eval_m
from exp_v6_vocoder import cloner as v6_cloner, vocoder as v6_voc
from exp_v8_timbre_clone import SR


@dataclass
class TimbreProfileV2:
    name: str
    n_utterances: int
    total_voiced_s: float
    f0_median_hz: float
    f0_log_std: float
    f0_iqr_st: float
    tilt_db_per_oct: float
    voiced_tilt_hf_db_oct: float       # slope over 280-6500 Hz (immune to 80-160 Hz F0 shift)
    hnr_median_db: float
    voiced_env_mean: np.ndarray        # (n_bins,) mean-centered > 250 Hz
    bap_voiced_median: np.ndarray      # (5,)
    f1_median_hz: float
    f2_median_hz: float
    f3_median_hz: float
    n_formant_frames: int
    unvoiced_tilt_slope: float         # slope (np/oct) over 1-8 kHz on unvoiced frames
    unvoiced_rel_level_np: float       # unvoiced vs voiced RMS level difference (nepers)
    voiced_ceps_q24: np.ndarray        # (N_v, 24) real cepstrum c[1..24] for Ridge transform
    vtln_anchor_f2_hz: float

    def to_v6_profile(self) -> v6_cloner.FewShotProfile:
        return v6_cloner.FewShotProfile(
            name=self.name,
            n_utterances=self.n_utterances,
            total_voiced_s=self.total_voiced_s,
            f0_median_hz=self.f0_median_hz,
            f0_log_std=self.f0_log_std,
            f0_iqr_st=self.f0_iqr_st,
            tilt_db_per_oct=self.tilt_db_per_oct,
            hnr_median_db=self.hnr_median_db,
            voiced_env_mean=self.voiced_env_mean,
            bap_voiced_median=self.bap_voiced_median,
            vtln_anchor_f2_hz=self.vtln_anchor_f2_hz,
        )


def _voiced_hf_tilt(env_mean: np.ndarray, freqs: np.ndarray, f_lo: float = 280.0, f_hi: float = 6500.0) -> float:
    m = (freqs >= f_lo) & (freqs <= f_hi)
    if int(np.sum(m)) < 4:
        return -3.5
    slope, _ = np.polyfit(np.log2(freqs[m]), 8.6858896 * env_mean[m], 1)
    return float(slope)


def _unvoiced_stats(log_env: np.ndarray, v_mask: np.ndarray, freqs: np.ndarray) -> tuple[float, float]:
    """Compute unvoiced 1-8 kHz slope (nepers/oct) and unvoiced-vs-voiced log level."""
    u_mask = ~v_mask
    if int(np.sum(u_mask)) < 3 or int(np.sum(v_mask)) < 3:
        return 0.0, -1.2
    u_mean = np.mean(log_env[u_mask], axis=0)
    m_hi = (freqs >= 1000.0) & (freqs <= 8000.0)
    slope_np, _ = np.polyfit(np.log2(freqs[m_hi]), u_mean[m_hi], 1)
    v_lvl = float(np.median(np.mean(log_env[v_mask], axis=1)))
    u_lvl = float(np.median(np.mean(log_env[u_mask], axis=1)))
    return float(slope_np), float(np.clip(u_lvl - v_lvl, -3.5, 0.5))


def extract_v2_profile(
    items: list[tuple[np.ndarray, v6_voc.VocoderFeatures]] | list[np.ndarray],
    sr: int = SR,
    name: str = "target_v2",
) -> TimbreProfileV2:
    """Extract a decoupled V/UV + LPC-22 formant + cepstral profile from 1+ utterances."""
    f0_list = []
    env_list = []
    bap_list = []
    tilts = []
    hnrs = []
    f1s = []
    f2s = []
    f3s = []
    n_fmt_total = 0
    u_slopes = []
    u_rels = []
    ceps_list = []
    n_fft = 1024
    hop = 120
    freqs = rfftfreq(n_fft, 1.0 / sr)
    m_above_250 = freqs >= 250.0

    for item in items:
        if isinstance(item, tuple):
            wav, feats = item
        else:
            wav = np.asarray(item, dtype=np.float64)
            feats = v6_voc.analyze(wav, sr=sr)
        n_fft = feats.n_fft
        hop = feats.hop_samples
        freqs = rfftfreq(n_fft, 1.0 / sr)
        m_above_250 = freqs >= 250.0

        v_mask = feats.voiced & (feats.f0_hz > 0.0)
        if np.any(v_mask):
            f0_list.append(feats.f0_hz[v_mask])
            ve = feats.log_env[v_mask]
            # Mean-center using >250 Hz band so sub-fundamental shift doesn't skew DC
            ve_c = ve - np.mean(ve[:, m_above_250], axis=1, keepdims=True)
            env_list.append(ve_c)
            bap_list.append(feats.bap[v_mask])
            cep = irfft(ve_c, n=n_fft, axis=1)[:, 1:25]
            ceps_list.append(cep)

        u_slope, u_rel = _unvoiced_stats(feats.log_env, v_mask, freqs)
        u_slopes.append(u_slope)
        u_rels.append(u_rel)

        lt = eval_m.compute_ltas_metrics(wav, sr=sr)
        if lt and math.isfinite(lt["tilt_db_oct"]):
            tilts.append(float(lt["tilt_db_oct"]))
        fh = eval_m.compute_f0_hnr_metrics(wav, sr=sr)
        if fh and fh.get("hnr_db_median") is not None and math.isfinite(fh["hnr_db_median"]):
            hnrs.append(float(fh["hnr_db_median"]))
        fmt = eval_m.compute_formant_metrics(wav, sr=sr)
        if fmt and fmt.get("f1_median_hz") is not None and fmt.get("f2_median_hz") is not None:
            w_f = int(fmt["n_formant_frames"])
            f1s.append((float(fmt["f1_median_hz"]), w_f))
            f2s.append((float(fmt["f2_median_hz"]), w_f))
            if fmt.get("f3_median_hz") is not None:
                f3s.append((float(fmt["f3_median_hz"]), w_f))
            n_fmt_total += w_f

    all_f0 = np.concatenate(f0_list) if f0_list else np.array([200.0], dtype=np.float64)
    f0_med = float(np.median(all_f0))
    f0_log_std = float(np.std(np.log(all_f0))) if len(all_f0) > 2 else 0.12
    st = 12.0 * np.log2(all_f0 / max(1e-6, f0_med))
    f0_iqr_st = float(np.percentile(st, 75) - np.percentile(st, 25)) if len(all_f0) > 4 else 2.5

    all_env = np.concatenate(env_list, axis=0) if env_list else np.zeros((1, n_fft // 2 + 1))
    voiced_env_mean = np.mean(all_env, axis=0)
    all_bap = np.concatenate(bap_list, axis=0) if bap_list else np.full((1, 5), 0.3)
    bap_med = np.median(all_bap, axis=0)
    all_ceps = np.concatenate(ceps_list, axis=0) if ceps_list else np.zeros((1, 24), dtype=np.float64)

    f1_med = float(np.average([v for v, _ in f1s], weights=[w for _, w in f1s])) if f1s else 500.0
    f2_med = float(np.average([v for v, _ in f2s], weights=[w for _, w in f2s])) if f2s else 1750.0
    f3_med = float(np.average([v for v, _ in f3s], weights=[w for _, w in f3s])) if f3s else 2900.0
    f2_cent = v6_cloner._voiced_f2_centroid(all_env, np.ones(len(all_env), dtype=bool), sr, n_fft)

    return TimbreProfileV2(
        name=name,
        n_utterances=len(items),
        total_voiced_s=float(len(all_f0) * hop / sr),
        f0_median_hz=f0_med,
        f0_log_std=max(0.03, f0_log_std),
        f0_iqr_st=max(0.5, f0_iqr_st),
        tilt_db_per_oct=float(np.median(tilts)) if tilts else -3.5,
        voiced_tilt_hf_db_oct=_voiced_hf_tilt(voiced_env_mean, freqs),
        hnr_median_db=float(np.median(hnrs)) if hnrs else 12.0,
        voiced_env_mean=voiced_env_mean,
        bap_voiced_median=bap_med,
        f1_median_hz=f1_med,
        f2_median_hz=f2_med,
        f3_median_hz=f3_med,
        n_formant_frames=n_fmt_total,
        unvoiced_tilt_slope=float(np.median(u_slopes)) if u_slopes else 0.0,
        unvoiced_rel_level_np=float(np.median(u_rels)) if u_rels else -1.2,
        voiced_ceps_q24=all_ceps,
        vtln_anchor_f2_hz=f2_cent,
    )


def make_synthetic_male_v2_profile(
    base_female: TimbreProfileV2,
    f0_hz: float = 115.0,
    iqr_st: float = 2.4,
    tilt_db_per_oct: float = -4.5,
    vtln_alpha: float = 0.84,
    sr: int = SR,
    n_fft: int = 1024,
) -> TimbreProfileV2:
    """Create a synthetic male baritone V2 profile (F1/F2 scaled by vtln_alpha for auto_vtln testing)."""
    warped_env = v6_cloner._vtln_warp_2d(base_female.voiced_env_mean[None, :], vtln_alpha, sr=sr, n_fft=n_fft)[0]
    freqs = rfftfreq(n_fft, 1.0 / sr)
    delta_slope = tilt_db_per_oct - base_female.voiced_tilt_hf_db_oct
    f_clip = np.clip(freqs, 250.0, 8000.0)
    delta_np = (delta_slope * np.log2(f_clip / 1000.0)) / 8.6858896
    male_env = warped_env + delta_np
    m_above_250 = freqs >= 250.0
    male_env -= float(np.mean(male_env[m_above_250]))
    cep_pad = np.zeros((len(base_female.voiced_ceps_q24), n_fft), dtype=np.float64)
    cep_pad[:, 1:25] = base_female.voiced_ceps_q24
    cep_pad[:, n_fft - 24 :] = base_female.voiced_ceps_q24[:, ::-1]
    ceps_warped = irfft(
        v6_cloner._vtln_warp_2d(
            rfft(cep_pad, n=n_fft, axis=1).real,
            vtln_alpha,
            sr=sr,
            n_fft=n_fft,
        ) + delta_np[None, :],
        n=n_fft,
        axis=1,
    )[:, 1:25]
    return TimbreProfileV2(
        name="male_115hz_v2",
        n_utterances=base_female.n_utterances,
        total_voiced_s=base_female.total_voiced_s,
        f0_median_hz=float(f0_hz),
        f0_log_std=base_female.f0_log_std * (iqr_st / max(0.5, base_female.f0_iqr_st)),
        f0_iqr_st=float(iqr_st),
        tilt_db_per_oct=float(tilt_db_per_oct),
        voiced_tilt_hf_db_oct=float(tilt_db_per_oct),
        hnr_median_db=13.0,
        voiced_env_mean=male_env,
        bap_voiced_median=np.clip(base_female.bap_voiced_median * 0.92, 0.01, 0.98),
        f1_median_hz=base_female.f1_median_hz * vtln_alpha,
        f2_median_hz=base_female.f2_median_hz * vtln_alpha,
        f3_median_hz=base_female.f3_median_hz * vtln_alpha,
        n_formant_frames=base_female.n_formant_frames,
        unvoiced_tilt_slope=base_female.unvoiced_tilt_slope,
        unvoiced_rel_level_np=base_female.unvoiced_rel_level_np,
        voiced_ceps_q24=ceps_warped,
        vtln_anchor_f2_hz=base_female.vtln_anchor_f2_hz * vtln_alpha,
    )


def estimate_auto_vtln(
    src_profile: TimbreProfileV2,
    tgt_profile: TimbreProfileV2,
    *,
    k_shrink: float = 280.0,
    deadband: float = 0.055,
) -> dict:
    """Data-driven VTLN alpha from LPC-22 F1/F2/F3 ratios with shrinkage toward 1.0."""
    r_f1 = tgt_profile.f1_median_hz / max(100.0, src_profile.f1_median_hz)
    r_f2 = tgt_profile.f2_median_hz / max(300.0, src_profile.f2_median_hz)
    r_f3 = tgt_profile.f3_median_hz / max(1200.0, src_profile.f3_median_hz)
    raw_ratio = math.exp(
        0.25 * math.log(max(0.5, min(2.0, r_f1)))
        + 0.35 * math.log(max(0.5, min(2.0, r_f2)))
        + 0.40 * math.log(max(0.5, min(2.0, r_f3)))
    )
    n_eff = min(src_profile.n_formant_frames, tgt_profile.n_formant_frames)
    weight = n_eff / (n_eff + k_shrink) if n_eff > 0 else 0.0
    shrunk = 1.0 + weight * (raw_ratio - 1.0)
    alpha = 1.0 if abs(shrunk - 1.0) < deadband else float(np.clip(shrunk, 0.78, 1.22))
    return {
        "alpha": round(alpha, 4),
        "raw_ratio": round(raw_ratio, 4),
        "shrunk_ratio": round(shrunk, 4),
        "shrink_weight": round(weight, 4),
        "n_eff_formant_frames": int(n_eff),
    }


def _sub_fundamental_ramp(freqs: np.ndarray, tgt_f0_hz: float) -> np.ndarray:
    """Raised-cosine high-pass mask: 0 below 0.85*f_cut, 1 above 1.25*f_cut."""
    f_cut = max(250.0, 1.3 * float(tgt_f0_hz))
    f_lo = 0.85 * f_cut
    f_hi = 1.25 * f_cut
    u = np.clip((freqs - f_lo) / max(1.0, f_hi - f_lo), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(math.pi * u))


def adapt_features_v2(
    src_feats: v6_voc.VocoderFeatures,
    src_profile: TimbreProfileV2,
    tgt_profile: TimbreProfileV2,
    *,
    vtln_alpha: float | None = None,
    q_timbre: int = 20,
) -> tuple[v6_voc.VocoderFeatures, dict]:
    """V/UV-decoupled spectral envelope adaptation + data-driven auto-VTLN + BAP transfer."""
    sr = src_feats.sr
    n_fft = src_feats.n_fft
    freqs = rfftfreq(n_fft, 1.0 / sr)
    v_mask = src_feats.voiced & (src_feats.f0_hz > 0.0)
    u_mask = ~v_mask

    # 1. F0 log-z-score mapping
    f0_out = np.zeros_like(src_feats.f0_hz)
    if np.any(v_mask):
        ratio_std = max(0.55, min(1.85, tgt_profile.f0_log_std / max(0.03, src_profile.f0_log_std)))
        log_f0_src = np.log(src_feats.f0_hz[v_mask])
        log_f0_tgt = math.log(tgt_profile.f0_median_hz) + ratio_std * (
            log_f0_src - math.log(src_profile.f0_median_hz)
        )
        f0_out[v_mask] = np.clip(np.exp(log_f0_tgt), 65.0, 440.0)

    # 2. Data-driven Auto-VTLN (or explicit override)
    vtln_info = estimate_auto_vtln(src_profile, tgt_profile)
    alpha = vtln_info["alpha"] if vtln_alpha is None else float(vtln_alpha)

    env_out = src_feats.log_env.copy()
    # Voiced-only VTLN warp (unvoiced consonants kept unwarped to preserve fricative/stop crispness)
    if abs(alpha - 1.0) >= 1e-4 and np.any(v_mask):
        env_out[v_mask] = v6_cloner._vtln_warp_2d(src_feats.log_env[v_mask], alpha, sr=sr, n_fft=n_fft)

    # 3a. Voiced-only decoupled spectral adaptation with sub-fundamental boost guard
    if np.any(v_mask):
        src_v_norm = src_profile.voiced_env_mean
        if abs(alpha - 1.0) >= 1e-4:
            src_v_norm = v6_cloner._vtln_warp_2d(src_v_norm[None, :], alpha, sr=sr, n_fft=n_fft)[0]
        raw_diff = tgt_profile.voiced_env_mean - src_v_norm
        smooth_diff = v6_cloner._cepstral_smooth_1d(raw_diff, n_fft=n_fft, q_max=q_timbre)

        f_cut = max(250.0, 1.3 * tgt_profile.f0_median_hz)
        applied_curve = np.clip(smooth_diff, -1.15, 1.15)
        # Only clamp sub-fundamental boost on synthetic profiles lacking real low-F0 recordings.
        if tgt_profile.name.startswith("male_") and tgt_profile.f0_median_hz < 0.85 * src_profile.f0_median_hz:
            sub_mask = freqs < f_cut
            applied_curve[sub_mask] = np.minimum(applied_curve[sub_mask], 0.0)

        # Preserve per-frame RMS energy across full voiced band
        pre_pwr = np.sum(np.exp(2.0 * env_out[v_mask]), axis=1)
        env_shifted = env_out[v_mask] + applied_curve[None, :]
        post_pwr = np.sum(np.exp(2.0 * env_shifted), axis=1)
        gain_adj = 0.5 * np.log(np.maximum(pre_pwr, 1e-18) / np.maximum(post_pwr, 1e-18))
        env_out[v_mask] = env_shifted + gain_adj[:, None]

    # 3b. Unvoiced consonant preservation: gentle tilt/level align only above 1 kHz (never VTLN-blurred)
    if np.any(u_mask):
        m_u = freqs >= 1000.0
        u_ramp = np.clip((freqs - 800.0) / 800.0, 0.0, 1.0)
        d_u_slope = float(np.clip(tgt_profile.unvoiced_tilt_slope - src_profile.unvoiced_tilt_slope, -0.45, 0.45))
        u_shift = u_ramp * d_u_slope * np.log2(np.maximum(freqs, 1000.0) / 3000.0)
        u_shift = np.clip(u_shift, -0.35, 0.35)
        d_u_lvl = float(np.clip(tgt_profile.unvoiced_rel_level_np - src_profile.unvoiced_rel_level_np, -0.30, 0.30))
        env_out[u_mask] = env_out[u_mask] + u_shift[None, :] + 0.5 * d_u_lvl

    # 4. Band-wise aperiodicity transfer on voiced frames ONLY
    bap_out = src_feats.bap.copy()
    if np.any(v_mask):
        delta_logit = v6_cloner._logit(tgt_profile.bap_voiced_median) - v6_cloner._logit(src_profile.bap_voiced_median)
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
    return out_feats, {"alpha": alpha, "vtln_info": vtln_info}


def fit_ridge_cepstral_map(
    src_profile: TimbreProfileV2,
    tgt_profile: TimbreProfileV2,
    *,
    n_quantiles: int = 24,
    alpha_reg: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a regularized Ridge affine transform ``c_tgt = W @ c_src + b`` over per-bin quantiles.

    Because few-shot prompts are unpaired across text, frames are aligned via
    per-cepstral-dimension sorted quantiles (matching marginal distributions of c1..c24
    without leaking phonetic sequence).
    """
    c_src = src_profile.voiced_ceps_q24
    c_tgt = tgt_profile.voiced_ceps_q24
    qs = np.linspace(5.0, 95.0, n_quantiles)
    q_src = np.percentile(c_src, qs, axis=0)   # (Q, 24)
    q_tgt = np.percentile(c_tgt, qs, axis=0)   # (Q, 24)

    reg = Ridge(alpha=alpha_reg, fit_intercept=True)
    reg.fit(q_src, q_tgt)
    return reg.coef_, reg.intercept_


def adapt_features_ridge(
    src_feats: v6_voc.VocoderFeatures,
    src_profile: TimbreProfileV2,
    tgt_profile: TimbreProfileV2,
    *,
    alpha_reg: float = 50.0,
    blend: float = 0.65,
) -> tuple[v6_voc.VocoderFeatures, dict]:
    """Non-neural Ridge cepstral transform + F0/BAP transfer (ablation candidate vs v8_decoupled)."""
    n_fft = src_feats.n_fft
    sr = src_feats.sr
    freqs = rfftfreq(n_fft, 1.0 / sr)
    v_mask = src_feats.voiced & (src_feats.f0_hz > 0.0)

    # Start with F0 + BAP from v2 (no VTLN double-warp since Ridge learns spectral shift)
    f0_out = np.zeros_like(src_feats.f0_hz)
    if np.any(v_mask):
        ratio_std = max(0.55, min(1.85, tgt_profile.f0_log_std / max(0.03, src_profile.f0_log_std)))
        log_f0_src = np.log(src_feats.f0_hz[v_mask])
        log_f0_tgt = math.log(tgt_profile.f0_median_hz) + ratio_std * (
            log_f0_src - math.log(src_profile.f0_median_hz)
        )
        f0_out[v_mask] = np.clip(np.exp(log_f0_tgt), 65.0, 440.0)

    env_out = src_feats.log_env.copy()
    if np.any(v_mask):
        W, b = fit_ridge_cepstral_map(src_profile, tgt_profile, alpha_reg=alpha_reg)
        ve = src_feats.log_env[v_mask]
        cep_full = irfft(ve, n=n_fft, axis=1)
        c24 = cep_full[:, 1:25]
        c24_hat = c24 @ W.T + b[None, :]
        # Blend with identity to prevent unstable extrapolation on small 1-shot prompts
        delta_c24 = blend * (c24_hat - c24)
        cep_delta = np.zeros_like(cep_full)
        cep_delta[:, 1:25] = delta_c24
        cep_delta[:, n_fft - 24 :] = delta_c24[:, ::-1]
        spec_delta = rfft(cep_delta, n=n_fft, axis=1).real
        hf_mask = _sub_fundamental_ramp(freqs, tgt_profile.f0_median_hz)
        spec_delta = np.clip(spec_delta * hf_mask[None, :], -1.0, 1.0)

        m_rms = freqs >= max(250.0, 1.3 * tgt_profile.f0_median_hz)
        pre_pwr = np.sum(np.exp(2.0 * ve[:, m_rms]), axis=1)
        env_shifted = ve + spec_delta
        post_pwr = np.sum(np.exp(2.0 * env_shifted[:, m_rms]), axis=1)
        gain_adj = 0.5 * np.log(np.maximum(pre_pwr, 1e-18) / np.maximum(post_pwr, 1e-18))
        env_out[v_mask] = env_shifted + gain_adj[:, None]

    bap_out = src_feats.bap.copy()
    if np.any(v_mask):
        delta_logit = v6_cloner._logit(tgt_profile.bap_voiced_median) - v6_cloner._logit(src_profile.bap_voiced_median)
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
    return out_feats, {"alpha_reg": alpha_reg, "blend": blend}
