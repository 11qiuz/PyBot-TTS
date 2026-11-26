"""Cross-corpus microphone/channel EQ vs. speaker timbre decoupler (non-neural)."""
from __future__ import annotations

import math

import numpy as np
from scipy.fft import rfftfreq

from exp_v6_vocoder import cloner as v6_cloner
from exp_v9_timbre_v3 import N_FFT, SR

DB_PER_NP = 20.0 / math.log(10.0)


def shared_band_mask(freqs: np.ndarray, f_lo: float = 250.0, f_hi: float = 7200.0) -> np.ndarray:
    """Boolean mask for the 16 kHz-safe shared speech band (avoids sub-F0 & 8-12 kHz Nyquist cliffs)."""
    return (freqs >= f_lo) & (freqs <= f_hi)


def normalize_env_shared_band(
    log_env: np.ndarray,
    freqs: np.ndarray,
    f_lo: float = 250.0,
    f_hi: float = 7200.0,
) -> np.ndarray:
    """Mean-center log_env frames/vectors using only the shared [250, 7200] Hz speech band."""
    m = shared_band_mask(freqs, f_lo=f_lo, f_hi=f_hi)
    if not np.any(m):
        return log_env - np.mean(log_env, axis=-1, keepdims=True)
    if log_env.ndim == 1:
        return log_env - float(np.mean(log_env[m]))
    return log_env - np.mean(log_env[:, m], axis=1, keepdims=True)


def clamp_nyquist_mismatch(
    raw_diff: np.ndarray,
    freqs: np.ndarray,
    f_edge_lo: float = 5800.0,
    f_edge_hi: float = 7200.0,
) -> np.ndarray:
    """Hold [5.8, 7.2] kHz mean diff smoothly above 7.2 kHz so 16k-vs-24k corpus Nyquist cliffs do not ring."""
    out = raw_diff.copy()
    m_ref = (freqs >= f_edge_lo) & (freqs <= f_edge_hi)
    if not np.any(m_ref):
        return out
    ref_val = float(np.mean(raw_diff[m_ref]))
    m_hi = freqs > f_edge_hi
    if np.any(m_hi):
        # Smoothly blend from measured diff at 7.2 kHz to flat ref_val by 8.2 kHz
        u = np.clip((freqs[m_hi] - f_edge_hi) / 1000.0, 0.0, 1.0)
        w = 0.5 * (1.0 - np.cos(math.pi * u))
        out[m_hi] = (1.0 - w) * out[m_hi] + w * ref_val
    return out


def linear_tilt_fit(curve_np: np.ndarray, freqs: np.ndarray, f_lo: float = 80.0, f_hi: float = 8000.0) -> tuple[float, float]:
    """Fit `curve_np ≈ slope_np_oct * log2(f/1000) + intercept` over [f_lo, f_hi]."""
    m = (freqs >= f_lo) & (freqs <= f_hi)
    if int(np.sum(m)) < 4:
        return 0.0, float(np.mean(curve_np))
    x = np.log2(freqs[m] / 1000.0)
    slope, intercept = np.polyfit(x, curve_np[m], 1)
    return float(slope), float(intercept)


def tilt_curve_on_freqs(slope_np_oct: float, freqs: np.ndarray, intercept: float = 0.0) -> np.ndarray:
    """Evaluate `slope_np_oct * log2(clip(f, 80, 8000)/1000) + intercept` smoothly across all FFT bins."""
    f_safe = np.clip(freqs, 80.0, 8000.0)
    return slope_np_oct * np.log2(f_safe / 1000.0) + intercept


def estimate_corpus_baseline(
    speaker_envs: dict[str, np.ndarray],
    *,
    exclude_speaker: str | None = None,
) -> np.ndarray | None:
    """Zero-leakage corpus-mean voiced envelope computed over training speakers excluding `exclude_speaker`."""
    vecs = [v for k, v in speaker_envs.items() if k != exclude_speaker]
    if not vecs:
        return None
    return np.mean(np.stack(vecs, axis=0), axis=0)


def decompose_envelope_eq(
    src_env: np.ndarray,
    tgt_env: np.ndarray,
    freqs: np.ndarray,
    *,
    src_tilt_db_oct: float,
    tgt_tilt_db_oct: float,
    n_fft: int = N_FFT,
    q_timbre: int = 18,
    tilt_anchor_weight: float = 0.85,
    eq_mode: str = "blind_tilt_anchor",
    src_corpus_env: np.ndarray | None = None,
    tgt_corpus_env: np.ndarray | None = None,
    corpus_eq_keep: float = 0.65,
    channel_ridge_damp: float = 0.65,
    clip_lim: float = 1.15,
) -> tuple[np.ndarray, dict]:
    """Separate cross-corpus mic EQ / bandwidth cliffs from speaker tilt & formant shape.

    Modes:
    - ``blind_tilt_anchor`` (default): requires zero corpus labels; clamps >7.2 kHz Nyquist step,
      extracts smooth cepstral difference, strips inconsistent cross-corpus linear slope when
      waveform tilt disagrees with envelope slope, and re-anchors linear tilt directly to
      ``tgt_tilt_db_oct - src_tilt_db_oct``.
    - ``corpus_eq_split``: when ``src_corpus_env`` and ``tgt_corpus_env`` are available,
      separates ``(tgt - tgt_corp) - (src - src_corp)`` (speaker deviation) from
      ``corpus_eq_keep * (tgt_corp - src_corp)`` (corpus mic EQ), while keeping tilt anchor.
    """
    src_n = normalize_env_shared_band(src_env, freqs)
    tgt_n = normalize_env_shared_band(tgt_env, freqs)

    raw_diff = clamp_nyquist_mismatch(tgt_n - src_n, freqs)
    sm_diff = v6_cloner._cepstral_smooth_1d(raw_diff, n_fft=n_fft, q_max=q_timbre)

    # Measure slope implied by smooth_diff vs true waveform LTAS tilt difference
    s_env_np, b_env = linear_tilt_fit(sm_diff, freqs, f_lo=80.0, f_hi=8000.0)
    s_wav_np = float(tgt_tilt_db_oct - src_tilt_db_oct) / DB_PER_NP
    slope_mismatch = abs(s_wav_np - s_env_np)

    corpus_diff_sm = np.zeros_like(sm_diff)
    spk_resid_sm = sm_diff.copy()

    if eq_mode == "corpus_eq_split" and src_corpus_env is not None and tgt_corpus_env is not None:
        sc_n = normalize_env_shared_band(src_corpus_env, freqs)
        tc_n = normalize_env_shared_band(tgt_corpus_env, freqs)
        corp_raw = clamp_nyquist_mismatch(tc_n - sc_n, freqs)
        spk_raw = clamp_nyquist_mismatch((tgt_n - tc_n) - (src_n - sc_n), freqs)
        corpus_diff_sm = v6_cloner._cepstral_smooth_1d(corp_raw, n_fft=n_fft, q_max=q_timbre)
        spk_resid_sm = v6_cloner._cepstral_smooth_1d(spk_raw, n_fft=n_fft, q_max=q_timbre)
        combined = spk_resid_sm + float(corpus_eq_keep) * corpus_diff_sm
        s_env_np, b_env = linear_tilt_fit(combined, freqs, f_lo=80.0, f_hi=8000.0)
        resid_shape = combined - tilt_curve_on_freqs(s_env_np, freqs, intercept=b_env)
        # Re-anchor linear slope: blend target speaker tilt toward expected channel-adjusted tilt
        w_t = float(np.clip(tilt_anchor_weight, 0.0, 1.0))
        s_anchor = float(np.clip(w_t * s_wav_np + (1.0 - w_t) * s_env_np, -0.85, 0.85))
    else:
        # Remove global linear tilt from smooth envelope diff -> zero-slope formant/EQ shape
        resid_shape = sm_diff - tilt_curve_on_freqs(s_env_np, freqs, intercept=b_env)
        # If envelope slope strongly disagrees with waveform tilt (>0.12 np/oct ~= 1.0 dB/oct,
        # signature of cross-corpus mic presence ridges such as ARCTIC 2-4 kHz vs AISHELL-3),
        # softly damp ultra-broadband low-order ridge curvature (q <= 5) while keeping formant detail.
        if slope_mismatch > 0.12 and channel_ridge_damp > 0.0:
            broad_eq = v6_cloner._cepstral_smooth_1d(resid_shape, n_fft=n_fft, q_max=5)
            damping = float(np.clip(channel_ridge_damp * min(1.0, (slope_mismatch - 0.12) / 0.25), 0.0, 0.85))
            resid_shape = resid_shape - damping * broad_eq
        w_t = float(np.clip(tilt_anchor_weight, 0.0, 1.0))
        s_anchor = float(np.clip(w_t * s_wav_np + (1.0 - w_t) * s_env_np, -0.85, 0.85))

    # Taper residual & tilt safely above 7.5 kHz so 16k-vs-24k sources never inject HF hiss
    f_safe = np.clip(freqs, 80.0, 8000.0)
    tilt_component = s_anchor * np.log2(f_safe / 1000.0)
    combined_curve = resid_shape + tilt_component
    # Keep zero DC mean inside shared [250, 7200] Hz band before frame-level RMS match
    m_dc = shared_band_mask(freqs)
    if np.any(m_dc):
        combined_curve = combined_curve - float(np.mean(combined_curve[m_dc]))
    applied = np.clip(combined_curve, -clip_lim, clip_lim)

    diag = {
        "eq_mode": eq_mode,
        "s_env_np_oct": round(s_env_np, 4),
        "s_wav_np_oct": round(s_wav_np, 4),
        "s_anchor_np_oct": round(s_anchor, 4),
        "slope_mismatch_db_oct": round(slope_mismatch * DB_PER_NP, 3),
    }
    return applied, diag
