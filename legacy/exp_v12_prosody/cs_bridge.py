"""Cross-lingual ZH <-> EN 30 ms spectral-tilt & formant-contour seam bridge (`exp_v12_prosody`).

Fixes Phase-7's 20.24 dB `feature_join_log_env_l2_db` discontinuity at language-switch
boundaries (e.g., AISHELL-3 ZH <-> CMU-ARCTIC EN) via a symmetric raised-cosine
spectral-tilt + smooth formant ramp across +-15 ms (30 ms total) on voiced/sonorant
joins, while preserving unvoiced obstruent transients un-smeared.
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
from exp_v12_prosody import N_FFT, SR


def bridge_cs_boundary(
    log_env: np.ndarray,
    voiced: np.ndarray,
    switch_frames: list[int],
    freqs: np.ndarray | None = None,
    *,
    sr: int = SR,
    half_win_frames: int = 3,
    q_formant: int = 16,
    preserve_unvoiced_obstruents: bool = True,
) -> tuple[np.ndarray, dict]:
    """Smooth ``ZH <-> EN`` spectral & Nyquist discontinuities across 30 ms centered at switch frames.

    Args:
        log_env: ``(T, n_bins)`` spectral log-amplitude matrix.
        voiced: ``(T,)`` boolean voicing mask.
        switch_frames: frame indices ``bf`` where language changes between ``bf - 1`` and ``bf``.
        half_win_frames: half-width in 5 ms frames (default ``3`` -> 15 ms each side = 30 ms total).
        q_formant: cepstral order for smooth tilt+formant ramp inside ``[80, 7200] Hz``.
        preserve_unvoiced_obstruents: if True, only smooths ``>7.2 kHz`` Nyquist tail when
            both sides of the seam are unvoiced (never smearing fricative/stop bursts).
    """
    out = np.array(log_env, dtype=np.float64, copy=True)
    n_fr, n_bins = out.shape
    n_fft = (n_bins - 1) * 2
    if freqs is None:
        freqs = rfftfreq(n_fft, 1.0 / sr)

    m_band = freqs <= 7200.0
    m_hi = freqs > 7200.0
    l2_before: list[float] = []
    l2_after: list[float] = []
    bridged_voiced = 0
    bridged_uv_nyquist_only = 0

    for bf in sorted(set(int(f) for f in switch_frames)):
        if bf < 2 or bf >= n_fr - 1:
            continue
        diff_pre = out[bf] - out[bf - 1]
        l2_before.append(float(np.sqrt(np.mean(diff_pre * diff_pre))) * 20.0 / math.log(10.0))

        lo_start = max(0, bf - half_win_frames)
        hi_end = min(n_fr, bf + half_win_frames)
        l_anchor = np.mean(out[max(0, bf - 2) : bf], axis=0)
        r_anchor = np.mean(out[bf : min(n_fr, bf + 2)], axis=0)
        raw_step = r_anchor - l_anchor

        is_uv_pair = (not bool(voiced[bf - 1])) and (not bool(voiced[bf]))
        if preserve_unvoiced_obstruents and is_uv_pair:
            # Keep <7.2 kHz consonant burst intact; softly bridge >7.2 kHz 16k-vs-24k Nyquist step only
            nyq_ramp = np.where(m_hi, raw_step, 0.0)
            for t in range(lo_start, bf):
                w = 0.5 * (t - lo_start + 1) / max(1, bf - lo_start)
                out[t] = out[t] + w * nyq_ramp
            for t in range(bf, hi_end):
                w = 0.5 * (hi_end - t) / max(1, hi_end - bf)
                out[t] = out[t] - w * nyq_ramp
            bridged_uv_nyquist_only += 1
        else:
            # Smooth [0, 7200] Hz tilt+formant contour via q_formant cepstrum + exact >7.2 kHz Nyquist bridge
            inband = np.where(m_band, raw_step, float(np.mean(raw_step[(freqs >= 5800.0) & (freqs <= 7200.0)])) if np.any(m_band) else 0.0)
            sm_step = v6_cloner._cepstral_smooth_1d(inband, n_fft=n_fft, q_max=q_formant)
            # Add residual fine discrepancy weighted near the exact boundary frame so join L2 collapses cleanly
            full_step = np.where(m_band, 0.85 * sm_step + 0.15 * raw_step, raw_step)
            for t in range(lo_start, bf):
                # Raised-cosine ramp from 0 at outer edge to +0.5 at bf-1
                u = (t - lo_start + 1) / max(1, bf - lo_start)
                w = 0.5 * 0.5 * (1.0 - math.cos(math.pi * u))
                # Ensure exact midpoint meet at t == bf-1 (u == 1.0 -> w == 0.5)
                out[t] = out[t] + w * full_step
            for t in range(bf, hi_end):
                u = (hi_end - t) / max(1, hi_end - bf)
                w = 0.5 * 0.5 * (1.0 - math.cos(math.pi * u))
                out[t] = out[t] - w * full_step
            bridged_voiced += 1

        diff_post = out[bf] - out[bf - 1]
        l2_after.append(float(np.sqrt(np.mean(diff_post * diff_post))) * 20.0 / math.log(10.0))

    diag = {
        "n_switches": len(l2_before),
        "bridged_voiced_or_mixed": bridged_voiced,
        "bridged_unvoiced_nyquist_only": bridged_uv_nyquist_only,
        "join_l2_before_db": round(float(np.median(l2_before)), 3) if l2_before else None,
        "join_l2_after_db": round(float(np.median(l2_after)), 3) if l2_after else None,
    }
    return out.astype(log_env.dtype), diag


def find_cs_switch_frames(unit_langs: list[str], unit_end_frames: list[int]) -> list[int]:
    """Return frame indices where consecutive spoken units switch language (``zh <-> en``)."""
    switches: list[int] = []
    for i in range(len(unit_langs) - 1):
        if unit_langs[i] != unit_langs[i + 1] and unit_langs[i] in ("zh", "en") and unit_langs[i + 1] in ("zh", "en"):
            switches.append(int(unit_end_frames[i]))
    return switches


def bridge_cs_utterance_features(
    feats: v6_voc.VocoderFeatures,
    switch_frames: list[int],
    *,
    half_win_frames: int = 3,
) -> tuple[v6_voc.VocoderFeatures, dict]:
    """Apply ``bridge_cs_boundary`` directly to a ``VocoderFeatures`` object."""
    env_new, diag = bridge_cs_boundary(
        feats.log_env,
        feats.voiced,
        switch_frames,
        sr=feats.sr,
        half_win_frames=half_win_frames,
    )
    out = v6_voc.VocoderFeatures(
        f0_hz=feats.f0_hz.copy(),
        voiced=feats.voiced.copy(),
        log_env=env_new,
        bap=feats.bap.copy(),
        sr=feats.sr,
        hop_samples=feats.hop_samples,
        n_samples=feats.n_samples,
        n_fft=feats.n_fft,
    )
    return out, diag
