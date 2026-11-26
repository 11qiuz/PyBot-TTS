"""Spectral sharpness, formant peak-to-valley contrast, and consonant transient clarity metrics."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
from scipy.fft import rfft, rfftfreq

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_v8_clarity import SR  # noqa: E402


def compute_clarity_metrics(x: np.ndarray, sr: int = SR) -> dict[str, float]:
    """Measure spectral peak-to-valley sharpness and 2-8 kHz consonant transient crispness.

    1. ``spectral_contrast_db``: mean per-frame (p90 - p10) dB spread across 150-4000 Hz
       over active voiced frames (captures formant prominence vs cepstral/averaging blur).
    2. ``spectral_std_db``: mean per-frame dB std across 150-4000 Hz.
    3. ``transient_hf_p95_db`` & ``transient_hf_max_db``: 95th-percentile and max positive
       frame-to-frame energy jump (5 ms hop) in the 2-8 kHz consonant band (captures stop
       release bursts & crisp fricative attacks vs smeared splines).
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.size < 480:
        return {
            "spectral_contrast_db": 0.0,
            "spectral_std_db": 0.0,
            "transient_hf_p95_db": 0.0,
            "transient_hf_max_db": 0.0,
            "hf_energy_ratio_2_8k_db": -30.0,
        }

    # 20 ms Hann frames at 5 ms hop
    win_len = 480
    hop = 120
    n_fr = 1 + (len(arr) - win_len) // hop
    idx = np.arange(win_len)[None, :] + np.arange(n_fr)[:, None] * hop
    frames = arr[idx]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak = float(np.max(rms)) if rms.size else 0.0
    if peak <= 1e-9:
        return {
            "spectral_contrast_db": 0.0,
            "spectral_std_db": 0.0,
            "transient_hf_p95_db": 0.0,
            "transient_hf_max_db": 0.0,
            "hf_energy_ratio_2_8k_db": -30.0,
        }

    han = np.hanning(win_len)
    spec = np.abs(rfft((frames - np.mean(frames, axis=1, keepdims=True)) * han[None, :], n=1024, axis=1))
    freqs = rfftfreq(1024, 1.0 / sr)
    spec_db = 20.0 * np.log10(np.maximum(spec, 1e-7))

    # Active frames (>= -24 dB from peak)
    act_mask = rms >= max(1e-4, 0.06 * peak)
    f_band = (freqs >= 150.0) & (freqs <= 4000.0)
    if np.any(act_mask):
        sub_db = spec_db[act_mask][:, f_band]
        contrasts = np.percentile(sub_db, 90, axis=1) - np.percentile(sub_db, 10, axis=1)
        stds = np.std(sub_db, axis=1)
        spectral_contrast_db = float(np.mean(contrasts))
        spectral_std_db = float(np.mean(stds))
    else:
        spectral_contrast_db = 0.0
        spectral_std_db = 0.0

    # 2-8 kHz band energy per 5 ms frame
    hf_mask = (freqs >= 2000.0) & (freqs <= 8000.0)
    lf_mask = (freqs >= 100.0) & (freqs < 2000.0)
    hf_db = 10.0 * np.log10(np.maximum(np.mean(spec[:, hf_mask] ** 2, axis=1), 1e-14))
    lf_db = 10.0 * np.log10(np.maximum(np.mean(spec[:, lf_mask] ** 2, axis=1), 1e-14))

    # Positive onset transients in 2-8 kHz (excluding digital silence below -50 dB of peak)
    floor_db = 20.0 * math.log10(peak) - 50.0
    valid_pair = (hf_db[1:] >= floor_db) | (hf_db[:-1] >= floor_db)
    jumps = np.diff(hf_db)
    pos_jumps = jumps[valid_pair & (jumps > 0.0)]
    if pos_jumps.size > 0:
        transient_p95 = float(np.percentile(pos_jumps, 95))
        transient_max = float(np.max(pos_jumps))
    else:
        transient_p95 = 0.0
        transient_max = 0.0

    hf_ratio = float(np.mean(hf_db[act_mask] - lf_db[act_mask])) if np.any(act_mask) else -30.0
    # Unvoiced-frame HF (2-8 kHz) vs LF (80-1000 Hz) contrast: directly catches muffled vowel-bleed into consonants
    uv_lf_mask = (freqs >= 80.0) & (freqs <= 1000.0)
    uv_lf_db = 10.0 * np.log10(np.maximum(np.mean(spec[:, uv_lf_mask] ** 2, axis=1), 1e-14))
    # Zero-crossing rate proxy for unvoiced active frames (high ZCR + active RMS)
    zcr = np.mean((frames[:, 1:] * frames[:, :-1]) < 0.0, axis=1)
    uv_mask = (rms >= max(1e-4, 0.03 * peak)) & (zcr >= 0.14)
    unvoiced_hf_lf_db = float(np.mean(hf_db[uv_mask] - uv_lf_db[uv_mask])) if np.any(uv_mask) else hf_ratio
    return {
        "spectral_contrast_db": round(spectral_contrast_db, 3),
        "spectral_std_db": round(spectral_std_db, 3),
        "transient_hf_p95_db": round(transient_p95, 3),
        "transient_hf_max_db": round(transient_max, 3),
        "hf_energy_ratio_2_8k_db": round(hf_ratio, 3),
        "unvoiced_hf_lf_db": round(unvoiced_hf_lf_db, 3),
    }
