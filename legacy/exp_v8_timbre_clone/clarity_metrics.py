"""Clarity & consonant-preservation diagnostics for timbre conversion."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
from scipy.fft import rfft, rfftfreq

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_eval import metrics as eval_m
from exp_v6_vocoder import vocoder as v6_voc
from exp_v8_timbre_clone import SR


def compute_clarity_metrics(wav: np.ndarray, sr: int = SR) -> dict:
    """Measure formant peak-to-valley prominence (dB) and unvoiced consonant HF crispness (dB)."""
    feats = v6_voc.analyze(wav, sr=sr)
    freqs = rfftfreq(feats.n_fft, 1.0 / sr)
    v_mask = feats.voiced & (feats.f0_hz > 0.0)

    # Active-energy mask on frame mean log_env so silence frames don't pollute consonant stats
    frame_db = 8.6858896 * np.mean(feats.log_env, axis=1)
    act = frame_db >= (float(np.max(frame_db)) - 30.0) if frame_db.size else np.zeros(0, dtype=bool)
    v_act = v_mask & act
    u_act = (~v_mask) & act

    # 1. Voiced formant peak-to-valley prominence in 300-3500 Hz
    m_fmt = (freqs >= 300.0) & (freqs <= 3500.0)
    if np.any(v_act):
        v_db = 8.6858896 * feats.log_env[v_act][:, m_fmt]
        prom = np.percentile(v_db, 90, axis=1) - np.percentile(v_db, 10, axis=1)
        formant_prom_db = float(np.median(prom))
    else:
        formant_prom_db = 0.0

    # 2. Unvoiced consonant HF ratio (2-8 kHz vs 80-1000 Hz)
    m_lo = (freqs >= 80.0) & (freqs < 1000.0)
    m_hf = (freqs >= 2000.0) & (freqs <= 8000.0)
    if int(np.sum(u_act)) >= 2:
        u_pwr = np.exp(2.0 * feats.log_env[u_act])
        e_lo = np.sum(u_pwr[:, m_lo], axis=1) + 1e-18
        e_hf = np.sum(u_pwr[:, m_hf], axis=1) + 1e-18
        hf_ratio_db = float(np.median(10.0 * np.log10(e_hf / e_lo)))
    else:
        # Fallback over active frames if utterance has few unvoiced frames
        a_pwr = np.exp(2.0 * feats.log_env[act]) if np.any(act) else np.ones((1, len(freqs)))
        e_lo = np.sum(a_pwr[:, m_lo], axis=1) + 1e-18
        e_hf = np.sum(a_pwr[:, m_hf], axis=1) + 1e-18
        hf_ratio_db = float(np.median(10.0 * np.log10(e_hf / e_lo)))

    # 3. High-band >280 Hz LTAS tilt (excludes 80-160 Hz F0 shift artifact)
    ltas = eval_m.compute_ltas_metrics(wav, sr=sr)
    return {
        "formant_prominence_db": round(formant_prom_db, 3),
        "consonant_hf_ratio_db": round(hf_ratio_db, 3),
        "tilt_80_8k_db_oct": round(float(ltas["tilt_db_oct"]), 3) if ltas else None,
        "vhi_lo_4_8k_db": round(float(ltas["vhi_lo_db_4_8k"]), 3) if ltas else None,
        "n_unvoiced_active": int(np.sum(u_act)),
    }
