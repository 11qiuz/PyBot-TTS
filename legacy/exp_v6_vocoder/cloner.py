"""Zero-neural-weight few-shot voice cloning via F0/IQR mapping, VTLN warp, LTAS envelope transfer, and BAP transfer."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from exp_v5_eval import metrics as eval_m
from exp_v6_vocoder import SR
from exp_v6_vocoder.vocoder import VocoderFeatures, analyze


@dataclass
class FewShotProfile:
    name: str
    n_utterances: int
    total_voiced_s: float
    f0_median_hz: float
    f0_log_std: float
    f0_iqr_st: float
    tilt_db_per_oct: float
    hnr_median_db: float
    voiced_env_mean: np.ndarray   # (n_fft//2 + 1,) mean-centered log amplitude over voiced frames
    bap_voiced_median: np.ndarray # (5,) median band aperiodicity over voiced frames
    vtln_anchor_f2_hz: float      # rough F2-band spectral centroid for optional auto-VTLN


def _voiced_f2_centroid(log_env: np.ndarray, voiced: np.ndarray, sr: int, n_fft: int) -> float:
    """Spectral centroid in the F2 band (800-2800 Hz) over voiced frames (VTLN scale proxy)."""
    if not np.any(voiced):
        return 1750.0
    freqs = rfftfreq(n_fft, 1.0 / sr)
    m = (freqs >= 800.0) & (freqs <= 2800.0)
    pwr = np.exp(2.0 * np.mean(log_env[voiced], axis=0))
    w = pwr[m]
    return float(np.sum(freqs[m] * w) / max(1e-18, float(np.sum(w))))


def extract_fewshot_profile(
    items: list[tuple[np.ndarray, VocoderFeatures]] | list[np.ndarray],
    sr: int = SR,
    name: str = "target",
) -> FewShotProfile:
    """Extract a compact few-shot timbre profile from 1-3 utterances (~3-10 s)."""
    f0_list = []
    env_list = []
    bap_list = []
    tilts = []
    hnrs = []
    n_fft = 1024
    hop = 120

    for item in items:
        if isinstance(item, tuple):
            wav, feats = item
        else:
            wav = np.asarray(item, dtype=np.float64)
            feats = analyze(wav, sr=sr)
        n_fft = feats.n_fft
        hop = feats.hop_samples

        v_mask = feats.voiced & (feats.f0_hz > 0.0)
        if np.any(v_mask):
            f0_list.append(feats.f0_hz[v_mask])
            # Zero-center each frame before averaging so loudness doesn't skew spectral shape
            ve = feats.log_env[v_mask]
            env_list.append(ve - np.mean(ve, axis=1, keepdims=True))
            bap_list.append(feats.bap[v_mask])

        lt = eval_m.compute_ltas_metrics(wav, sr=sr)
        tilts.append(lt["tilt_db_oct"])
        fh = eval_m.compute_f0_hnr_metrics(wav, sr=sr)
        if math.isfinite(fh["hnr_db_median"]):
            hnrs.append(fh["hnr_db_median"])

    all_f0 = np.concatenate(f0_list) if f0_list else np.array([200.0], dtype=np.float64)
    f0_med = float(np.median(all_f0))
    log_f0 = np.log(all_f0)
    f0_log_std = float(np.std(log_f0)) if len(all_f0) > 2 else 0.12
    st = 12.0 * np.log2(all_f0 / max(1e-6, f0_med))
    f0_iqr_st = float(np.percentile(st, 75) - np.percentile(st, 25)) if len(all_f0) > 4 else 2.5

    all_env = np.concatenate(env_list, axis=0) if env_list else np.zeros((1, n_fft // 2 + 1))
    voiced_env_mean = np.mean(all_env, axis=0)
    all_bap = np.concatenate(bap_list, axis=0) if bap_list else np.full((1, 5), 0.3)
    bap_med = np.median(all_bap, axis=0)

    total_voiced_s = float(len(all_f0) * hop / sr)
    f2_cent = _voiced_f2_centroid(all_env, np.ones(len(all_env), dtype=bool), sr, n_fft)

    return FewShotProfile(
        name=name,
        n_utterances=len(items),
        total_voiced_s=total_voiced_s,
        f0_median_hz=f0_med,
        f0_log_std=max(0.03, f0_log_std),
        f0_iqr_st=max(0.5, f0_iqr_st),
        tilt_db_per_oct=float(np.median(tilts)) if tilts else -3.5,
        hnr_median_db=float(np.median(hnrs)) if hnrs else 12.0,
        voiced_env_mean=voiced_env_mean,
        bap_voiced_median=bap_med,
        vtln_anchor_f2_hz=f2_cent,
    )


def make_synthetic_male_profile(
    base_female_profile: FewShotProfile,
    f0_hz: float = 115.0,
    iqr_st: float = 2.4,
    tilt_db_per_oct: float = -4.5,
    vtln_alpha: float = 0.84,
    sr: int = SR,
    n_fft: int = 1024,
) -> FewShotProfile:
    """Construct a physically realistic male baritone target profile for cross-gender stress testing."""
    warped_env = _vtln_warp_2d(base_female_profile.voiced_env_mean[None, :], vtln_alpha, sr=sr, n_fft=n_fft)[0]
    freqs = rfftfreq(n_fft, 1.0 / sr)
    # Apply tilt shift from base female tilt to target male tilt
    delta_slope = tilt_db_per_oct - base_female_profile.tilt_db_per_oct
    f_clip = np.clip(freqs, 80.0, 8000.0)
    delta_np = (delta_slope * np.log2(f_clip / 1000.0)) / 8.6858896
    male_env = warped_env + delta_np
    male_env -= float(np.mean(male_env))
    return FewShotProfile(
        name="male_115hz_stress",
        n_utterances=1,
        total_voiced_s=base_female_profile.total_voiced_s,
        f0_median_hz=float(f0_hz),
        f0_log_std=base_female_profile.f0_log_std * (iqr_st / max(0.5, base_female_profile.f0_iqr_st)),
        f0_iqr_st=float(iqr_st),
        tilt_db_per_oct=float(tilt_db_per_oct),
        hnr_median_db=13.5,
        voiced_env_mean=male_env,
        bap_voiced_median=np.clip(base_female_profile.bap_voiced_median * 0.90, 0.01, 0.98),
        vtln_anchor_f2_hz=base_female_profile.vtln_anchor_f2_hz * vtln_alpha,
    )


def _vtln_warp_2d(log_env: np.ndarray, alpha: float, sr: int = SR, n_fft: int = 1024) -> np.ndarray:
    """Piecewise-linear VTLN spectral warping along frequency axis.

    ``alpha < 1.0`` shifts formants DOWN (longer male vocal tract):
    output at ``f_tgt`` samples source at ``f_src = f_tgt / alpha``.
    """
    if abs(alpha - 1.0) < 1e-4:
        return log_env.copy()
    freqs = rfftfreq(n_fft, 1.0 / sr)
    nyq = sr * 0.5
    f_max_warp = 0.82 * nyq
    # Forward map: f_tgt -> f_src
    # Breakpoint chosen so f_src <= nyq when f_tgt <= f_hi
    if alpha < 1.0:
        f_hi = min(f_max_warp, alpha * 0.90 * nyq)
    else:
        f_hi = f_max_warp
    f_src_hi = f_hi / alpha

    f_src = np.where(
        freqs <= f_hi,
        freqs / alpha,
        f_src_hi + (freqs - f_hi) * (nyq - f_src_hi) / max(1.0, nyq - f_hi),
    )
    f_src = np.clip(f_src, 0.0, nyq)
    bin_pos = f_src / (nyq / (len(freqs) - 1))
    lo = np.floor(bin_pos).astype(np.int32)
    hi = np.minimum(len(freqs) - 1, lo + 1)
    frac = (bin_pos - lo)[None, :]
    return (1.0 - frac) * log_env[:, lo] + frac * log_env[:, hi]


def _cepstral_smooth_1d(curve: np.ndarray, n_fft: int, q_max: int = 14) -> np.ndarray:
    """Smooth spectral difference curve via low-quefrency cepstral liftering (timbre-only, no phonetic detail)."""
    cep = irfft(curve, n=n_fft)
    cep_lp = np.zeros_like(cep)
    cep_lp[0] = cep[0]
    cep_lp[1:q_max] = cep[1:q_max]
    cep_lp[-q_max + 1 :] = cep[-q_max + 1 :]
    return rfft(cep_lp, n=n_fft).real


def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 0.005, 0.995)
    return np.log(pc / (1.0 - pc))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -12.0, 12.0)))


def adapt_features(
    src_feats: VocoderFeatures,
    src_profile: FewShotProfile,
    tgt_profile: FewShotProfile,
    vtln_alpha: float | None = None,
    use_ltas_transfer: bool = True,
    use_bap_transfer: bool = True,
    q_timbre: int = 24,
) -> VocoderFeatures:
    """Adapt source VocoderFeatures to target FewShotProfile (F0 + VTLN + LTAS envelope + BAP)."""
    n_fr, n_bins = src_feats.log_env.shape
    sr = src_feats.sr
    n_fft = src_feats.n_fft

    # 1. F0 log-z-score mapping
    f0_out = np.zeros_like(src_feats.f0_hz)
    v_mask = src_feats.voiced & (src_feats.f0_hz > 0.0)
    if np.any(v_mask):
        ratio_std = max(0.55, min(1.85, tgt_profile.f0_log_std / max(0.03, src_profile.f0_log_std)))
        log_f0_src = np.log(src_feats.f0_hz[v_mask])
        log_f0_tgt = math.log(tgt_profile.f0_median_hz) + ratio_std * (
            log_f0_src - math.log(src_profile.f0_median_hz)
        )
        f0_out[v_mask] = np.clip(np.exp(log_f0_tgt), 65.0, 440.0)

    # 2. VTLN spectral envelope warping
    if vtln_alpha is None:
        raw_alpha = tgt_profile.vtln_anchor_f2_hz / max(100.0, src_profile.vtln_anchor_f2_hz)
        # Damped auto-VTLN: ignore <4% F2 noise jitter (like SSB0502 vs SSB1593)
        alpha = 1.0 if abs(raw_alpha - 1.0) < 0.04 else float(np.clip(raw_alpha, 0.78, 1.22))
    else:
        alpha = float(vtln_alpha)

    env_out = src_feats.log_env.copy()
    if abs(alpha - 1.0) >= 1e-4:
        if np.any(v_mask):
            env_out[v_mask] = _vtln_warp_2d(src_feats.log_env[v_mask], alpha, sr=sr, n_fft=n_fft)
        u_mask = ~v_mask
        if np.any(u_mask):
            # Half-strength VTLN on unvoiced fricatives/stops preserves consonant place
            alpha_u = 0.5 * (1.0 + alpha)
            env_out[u_mask] = _vtln_warp_2d(src_feats.log_env[u_mask], alpha_u, sr=sr, n_fft=n_fft)

    # 3. Smooth LTAS / cepstral spectral-envelope residual transfer (voiced frames)
    if use_ltas_transfer and np.any(v_mask):
        src_v_norm = src_profile.voiced_env_mean
        if abs(alpha - 1.0) >= 1e-4:
            src_v_norm = _vtln_warp_2d(src_v_norm[None, :], alpha, sr=sr, n_fft=n_fft)[0]
        raw_diff = tgt_profile.voiced_env_mean - src_v_norm
        smooth_diff = _cepstral_smooth_1d(raw_diff, n_fft=n_fft, q_max=q_timbre)
        smooth_diff = np.clip(smooth_diff, -1.2, 1.2)  # cap at +/- 10.4 dB

        # Preserve per-frame RMS energy after applying spectral-balance shift
        pre_pwr = np.sum(np.exp(2.0 * env_out[v_mask]), axis=1)
        env_shifted = env_out[v_mask] + smooth_diff[None, :]
        post_pwr = np.sum(np.exp(2.0 * env_shifted), axis=1)
        gain_adj = 0.5 * np.log(np.maximum(pre_pwr, 1e-18) / np.maximum(post_pwr, 1e-18))
        env_out[v_mask] = env_shifted + gain_adj[:, None]

    # 4. Band-wise aperiodicity transfer (logit-space shift over voiced frames)
    bap_out = src_feats.bap.copy()
    if use_bap_transfer and np.any(v_mask):
        delta_logit = _logit(tgt_profile.bap_voiced_median) - _logit(src_profile.bap_voiced_median)
        bap_out[v_mask] = np.clip(_sigmoid(_logit(src_feats.bap[v_mask]) + delta_logit[None, :]), 0.005, 0.985)

    return VocoderFeatures(
        f0_hz=f0_out,
        voiced=src_feats.voiced.copy(),
        log_env=env_out,
        bap=bap_out,
        sr=sr,
        hop_samples=src_feats.hop_samples,
        n_samples=src_feats.n_samples,
        n_fft=n_fft,
    )
