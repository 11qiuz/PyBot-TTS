"""Rule 4: Dual-regime context-aware coarticulation, cep[0:18] seam bridging, and sustained-vowel micro-dynamics."""
from __future__ import annotations

import math

import numpy as np
from scipy.fft import dct, rfftfreq

from exp_20_pybot_larynx import N_BINS, N_FFT, SR
from exp_20_pybot_larynx.arbiter import (
    NEPER_TO_DB,
    OBSTRUENT_MANNERS,
    RuleProposal,
    UtteranceContext,
    classify_final_group,
    classify_onset_manner,
    classify_onset_place,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl

_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_DB_TO_NEPER = 1.0 / NEPER_TO_DB
_CEP_BRIDGE_K = 18
# Precomputed ortho DCT-II basis (N_BINS, 64) for blur detection (24:64) & cep[0:18] seam bridging
_CEP_DCT64 = dct(np.eye(N_BINS, dtype=np.float64), type=2, norm="ortho", axis=1)[:, :64]
_CEP_DCT18 = _CEP_DCT64[:, :_CEP_BRIDGE_K]
_CEP_DCT24_64 = _CEP_DCT64[:, 24:64]

# 10 C^1 raised-cosine partition-of-unity bands across 0..12 kHz:
# [<300, 300-650, 650-1150, 1150-1800, 1800-2600, 2600-3600, 3600-4900, 4900-6500, 6500-8600, >8600 Hz]
_BAND_CUTOFFS: tuple[tuple[float, float], ...] = (
    (300.0, 120.0),
    (650.0, 180.0),
    (1150.0, 220.0),
    (1800.0, 300.0),
    (2600.0, 350.0),
    (3600.0, 450.0),
    (4900.0, 550.0),
    (6500.0, 700.0),
    (8600.0, 900.0),
)


def _build_smooth_band_basis(freqs: np.ndarray) -> np.ndarray:
    """Build 10 C^1 partition-of-unity frequency band basis functions."""
    steps = []
    for fc, hw in _BAND_CUTOFFS:
        u = np.clip((freqs - fc) / hw, -1.0, 1.0)
        steps.append(0.5 * (1.0 + np.sin(0.5 * math.pi * u)))
    bands = [1.0 - steps[0]]
    for i in range(len(steps) - 1):
        bands.append(steps[i] * (1.0 - steps[i + 1]))
    bands.append(steps[-1])
    return np.stack(bands, axis=0)  # (10, N_BINS)


_BAND_BASIS = _build_smooth_band_basis(_FREQS)


def _to_nepers(table: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float64) * _DB_TO_NEPER for k, v in table.items()}


def _to_nepers_shaped(table: dict[str, list[float]]) -> dict[str, np.ndarray]:
    """Convert dB band vector to zero-mean nepers to shape spectrum without global gain shift."""
    out: dict[str, np.ndarray] = {}
    for k, v in table.items():
        arr = np.asarray(v, dtype=np.float64)
        out[k] = (arr - np.mean(arr)) * _DB_TO_NEPER
    return out


# Empirically measured on 250 TRAIN_IDS utterances vs knowledge_full.npz (in dB across 10 bands).
_RIGHT_FIN_PROFILES: dict[str, np.ndarray] = _to_nepers({
    "a_open_central|alveolar":    [+1.25, +2.24, +2.78, +3.53, +1.87, +0.83, +1.30, +0.45, +0.35, +1.08],
    "a_open_central|bilabial":    [+0.34, +0.15, +1.61, +2.81, -2.30, -3.75, -0.94, -2.40, -4.77, -3.81],
    "a_open_central|dental":      [+1.28, +0.78, -0.71, +1.46, -0.37, -2.92, -2.75, +0.35, +3.10, +4.92],
    "a_open_central|labiodental": [+0.31, -0.68, +3.34, +3.69, -0.73, -2.46, +1.36, +1.75, +1.88, +1.34],
    "a_open_central|palatal":     [+1.73, +1.06, -1.74, -2.34, +3.51, +2.36, -0.66, +2.20, +4.99, +5.28],
    "a_open_central|retroflex":   [+1.89, +2.55, +0.41, +0.03, +6.81, +8.09, +2.30, +3.00, +1.69, +1.04],
    "a_open_central|velar":       [-0.70, -1.10, +0.59, +0.64, -3.34, -4.19, -0.80, -0.85, -0.24, -1.72],
    "a_open_central|zero":        [+1.52, +4.72, +7.05, +4.20, +5.03, +8.82, +9.83, +5.72, +6.55, +6.15],
    "i_front|alveolar":           [+1.18, +1.89, +2.20, +2.94, +1.74, -0.34, -0.32, -1.35, -0.23, +0.50],
    "i_front|bilabial":           [+0.37, -0.44, +0.27, -0.12, -3.64, -6.05, -4.51, -5.56, -8.80, -7.05],
    "i_front|dental":             [+1.43, +1.52, +1.13, +3.95, +1.83, -2.83, -2.02, +2.11, +6.28, +7.73],
    "i_front|labiodental":        [+1.00, +1.72, +4.41, +5.55, +2.05, -0.30, +2.15, +2.61, +1.78, +2.39],
    "i_front|palatal":            [+1.91, +0.51, -0.94, -1.27, +1.95, +2.04, +0.56, +3.63, +7.13, +5.43],
    "i_front|retroflex":          [+2.13, +1.21, -0.27, +0.13, +4.96, +7.18, +2.03, +1.24, +0.63, -0.63],
    "i_front|velar":              [-0.12, +0.28, +2.28, +2.24, -1.57, -2.90, -0.29, +0.45, +0.51, +0.47],
    "i_front|zero":               [+2.50, +4.24, +4.63, +2.78, +3.24, +7.84, +9.75, +5.37, +6.25, +5.46],
    "u_round|alveolar":           [+0.78, +1.85, +2.18, +4.00, +0.96, +1.06, +1.62, +0.87, +0.48, +0.12],
    "u_round|bilabial":           [+0.25, +0.26, +2.64, -0.71, -3.67, -4.14, -0.88, -3.90, -3.97, -4.15],
    "u_round|dental":             [+0.29, -0.09, -0.42, +1.95, -0.90, -1.96, -1.41, +2.75, +4.18, +4.58],
    "u_round|labiodental":        [+0.80, +1.71, +4.71, +1.04, -0.99, -1.29, +2.47, +1.67, +2.50, +2.81],
    "u_round|palatal":            [+1.64, +1.06, -2.05, -0.06, +4.66, +3.02, +0.37, +3.51, +5.18, +4.71],
    "u_round|retroflex":          [+1.55, +1.16, -1.48, +2.45, +6.81, +6.62, -1.07, -1.29, -1.42, -0.65],
    "u_round|velar":              [-1.39, -1.82, +0.61, -2.35, -7.01, -7.06, -2.19, -2.63, -1.71, -2.95],
    "u_round|zero":               [+2.77, +4.21, +4.13, +1.31, +4.25, +7.61, +8.16, +3.85, +6.25, +5.79],
    "v_front_round|alveolar":     [+0.54, +0.51, -0.75, -0.31, -0.79, -2.86, -1.94, -1.15, -1.22, -2.36],
    "v_front_round|bilabial":     [-1.73, -3.19, -3.98, -3.07, -5.70, -7.13, -7.60, -8.34, -11.43, -9.44],
    "v_front_round|palatal":      [+1.41, -0.10, -2.44, -3.64, -1.63, +0.82, +1.42, +4.14, +7.00, +4.81],
    "v_front_round|retroflex":    [+2.29, +1.82, +0.89, +0.29, +4.71, +7.15, +1.87, +0.54, +1.01, -0.17],
    "v_front_round|velar":        [-0.32, -0.08, +1.26, +1.34, -3.32, -4.66, -2.19, -1.79, -1.23, -0.60],
    "v_front_round|zero":         [+2.44, +4.41, +4.56, +1.97, -1.83, +5.52, +8.03, +3.65, +5.11, +4.22],
})

_RIGHT_PLACE_FALLBACK: dict[str, np.ndarray] = _to_nepers({
    "alveolar":    [+1.07, +1.92, +2.22, +3.23, +1.44, +0.28, +0.64, -0.15, +0.09, +0.43],
    "bilabial":    [+0.21, -0.21, +1.08, +0.77, -3.22, -4.77, -2.46, -4.08, -6.28, -5.25],
    "dental":      [+1.09, +0.84, +0.07, +2.54, +0.34, -2.65, -2.13, +1.63, +4.60, +5.92],
    "labiodental": [+0.67, +0.76, +4.06, +3.59, +0.12, -1.43, +1.92, +2.01, +2.02, +2.08],
    "palatal":     [+1.76, +0.76, -1.52, -1.46, +2.82, +2.29, +0.20, +3.20, +6.04, +5.19],
    "retroflex":   [+1.92, +1.67, -0.27, +0.68, +6.01, +7.34, +1.34, +1.15, +0.50, -0.06],
    "velar":       [-0.59, -0.64, +1.32, +0.67, -3.44, -4.33, -0.99, -0.78, -0.33, -1.04],
    "zero":        [+2.18, +4.42, +5.44, +2.98, +3.95, +8.08, +9.38, +5.12, +6.33, +5.75],
})

# Residual right-anticipatory spectral shape measured against Stage 2 context-conditioned base (zero-mean)
_RIGHT_PLC_S2_SHAPED: dict[str, np.ndarray] = _to_nepers_shaped({
    "alveolar":    [+0.12, +1.21, +2.18, +1.87, +1.80, +1.75, +2.17, +1.67, +1.27, +1.58],
    "retroflex":   [+0.41, +0.63, +0.90, +1.18, +1.14, +1.51, +0.69, +0.55, -0.22, +0.38],
    "zero":        [+0.31, +0.49, +0.78, +1.15, +0.87, +1.31, +1.43, +1.03, +0.92, +0.96],
    "palatal":     [+0.52, +1.12, +1.59, +1.40, +2.04, +1.74, +1.33, +0.82, +0.45, +0.75],
    "velar":       [+0.62, +0.39, +1.67, +1.03, +0.23, +0.24, +1.49, +0.50, -0.03, +0.19],
    "bilabial":    [+0.77, +0.14, +1.63, +1.51, -0.11, +0.26, +1.19, +0.66, -0.49, -0.33],
    "dental":      [+0.18, +0.57, +1.08, +1.69, +0.83, +0.23, +0.74, +1.08, +0.73, +1.48],
    "labiodental": [+1.52, +0.39, +2.61, +1.53, -0.43, -0.16, +1.30, +0.63, +0.17, +0.75],
})

# Left coarticulation: spectral shape carryover onto non-obstruent onset & early final.
_LEFT_ON_PROFILES: dict[str, np.ndarray] = _to_nepers_shaped({
    "a_open_central|alveolar":  [+1.19, +1.44, +1.46, +1.07, +1.03, +0.91, +1.14, +1.31, -0.01, +0.60],
    "a_open_central|bilabial":  [+0.07, +1.16, +0.88, +1.41, +1.44, +0.56, +0.47, +1.50, -0.15, +1.03],
    "a_open_central|retroflex": [+0.41, +0.22, +0.39, -0.34, -0.50, -0.58, -0.09, +0.43, +0.80, -0.11],
    "i_front|alveolar":         [+1.36, +1.37, +1.11, +1.10, +1.02, +1.73, +1.07, +0.86, +0.41, +0.70],
    "i_front|bilabial":         [+0.62, +0.16, -0.72, -0.17, +0.35, +0.06, -0.38, +0.76, -0.85, +0.23],
    "i_front|retroflex":        [+2.17, +1.50, +0.76, +0.08, +1.07, +1.13, +0.22, +0.92, +0.79, +0.60],
    "u_round|alveolar":         [+0.99, +0.79, +1.48, +1.52, -0.19, +0.15, -0.75, -0.90, +0.42, -0.81],
    "u_round|bilabial":         [-0.36, +0.38, +1.33, +0.28, -0.38, +1.12, +1.32, +0.82, +0.61, -0.27],
    "u_round|retroflex":        [+0.57, +1.28, +1.47, +2.43, +2.91, +1.11, +1.10, +3.92, +5.40, +3.39],
})

_LEFT_FIN_PROFILES: dict[str, np.ndarray] = _to_nepers_shaped({
    "a_open_central|alveolar":  [-0.26, -0.42, -0.98, -1.26, -0.81, -0.44, -1.04, -0.58, -0.70, -0.23],
    "a_open_central|bilabial":  [-0.41, -0.23, +0.08, -0.06, -0.54, -1.04, +0.22, +0.17, +0.21, -0.63],
    "a_open_central|retroflex": [+0.10, -0.34, +0.28, +0.00, -1.10, -1.83, -1.66, -0.63, -0.59, -0.50],
    "a_open_central|zero":      [+0.04, +0.16, +0.32, +0.10, +0.47, +0.19, -0.13, +0.04, +0.04, +0.55],
    "i_front|alveolar":         [-0.27, -0.77, -0.62, -0.35, -0.31, -0.67, -1.00, -0.59, -0.79, -1.14],
    "i_front|bilabial":         [-0.01, -0.27, +0.02, -0.18, +0.01, -0.12, -0.45, -0.17, -1.27, -0.63],
    "i_front|retroflex":        [+0.23, +0.49, +0.10, -1.13, -0.95, -0.31, -0.92, -0.09, -0.28, -0.20],
    "i_front|zero":             [+0.69, +0.78, +1.44, +0.96, +0.94, +1.28, +1.12, +0.81, +0.91, +0.41],
    "u_round|alveolar":         [-0.15, +0.07, +0.23, +0.00, -0.47, -0.37, -0.02, -0.27, -0.21, -0.03],
    "u_round|bilabial":         [-1.35, -0.96, -0.18, +0.03, -1.05, -0.78, -1.53, -0.78, -1.21, -1.72],
    "u_round|zero":             [+0.90, +0.78, +0.74, +0.54, +0.55, +0.91, +0.62, +0.15, +1.09, +0.54],
})

# Cross-syllable tone-pair carryover onto early final (u <= 0.40) keyed by "<prev_tone>|<cur_tone>".
_TONE_PAIR_PROFILES: dict[str, np.ndarray] = _to_nepers({
    "1|1": [-0.24, +0.22, +0.18, +0.13, -0.15, -0.04, +0.17, +0.01, -0.16, -0.57],
    "1|2": [-0.23, +0.15, +0.87, +0.01, -0.26, -0.33, +0.39, +0.49, +0.35, +0.42],
    "1|3": [+0.36, +0.28, +0.41, -0.07, -0.19, -0.06, -0.14, +0.11, +0.30, +0.25],
    "1|4": [-0.77, +0.05, +0.26, -0.02, -0.57, -0.29, +0.19, +0.09, +0.17, +0.07],
    "1|5": [+0.52, +0.32, +1.32, +0.39, +0.98, +0.26, +1.30, +1.47, +3.10, +2.08],
    "2|1": [-0.19, -0.35, -0.33, -0.17, -0.30, -0.37, -0.41, -0.43, +0.32, -0.73],
    "2|2": [+0.06, -0.08, +0.68, +0.08, +0.04, +0.08, +0.40, +0.41, +1.06, +0.53],
    "2|3": [+0.19, +0.34, +0.70, +0.43, -0.04, -0.27, -0.06, +0.08, +0.67, +0.31],
    "2|4": [-0.05, +0.10, +0.26, -0.10, -0.16, -0.12, +0.08, -0.51, -0.12, -0.33],
    "2|5": [+0.89, +0.43, +0.64, -0.74, +1.09, +0.51, -0.49, +0.28, +1.51, +1.53],
    "3|1": [+0.20, -0.59, -1.12, -0.82, -1.20, -1.36, -2.13, -0.84, -2.33, -1.19],
    "3|2": [-0.66, -1.22, -1.72, -0.98, -1.03, -1.31, -2.45, -2.11, -2.69, -2.13],
    "3|4": [-0.01, -0.97, -0.96, -0.99, -0.80, -0.87, -1.64, -1.09, -1.51, -1.46],
    "3|5": [-0.49, -0.49, -1.05, -0.51, -0.65, -0.08, -1.21, -0.84, -1.05, -1.41],
    "4|1": [-0.35, -0.60, -0.65, -0.26, -0.23, -0.03, -0.35, -0.37, -0.14, -0.04],
    "4|2": [-0.01, -0.38, -0.41, -0.51, -0.27, +0.01, -0.58, -0.25, -0.10, -0.62],
    "4|3": [+0.09, -0.49, -1.35, -1.03, -0.57, -0.53, -1.28, -0.94, -1.19, -0.91],
    "4|4": [+0.14, -0.27, -0.10, -0.06, +0.63, +0.80, +0.39, +0.21, +0.12, +0.07],
    "4|5": [+0.26, +0.19, -0.88, -0.49, -0.12, +0.02, +0.01, -0.14, -0.27, -0.87],
    "5|1": [-0.31, -0.80, -1.10, -1.80, -0.93, -0.32, -1.10, -0.99, -0.94, -1.22],
    "5|2": [-0.71, -1.11, -1.07, -1.05, -0.91, -0.73, -1.21, -0.94, -0.73, -1.01],
    "5|4": [+0.41, -0.42, +0.03, -0.43, +0.12, -0.14, +0.22, -0.19, +0.30, -0.18],
})

_TONE_PAIR_S2_PROFILES: dict[str, np.ndarray] = _to_nepers({
    "1|1": [-0.62, +0.05, +0.16, +0.08, -0.36, -0.32, +0.40, +0.18, +0.43, +0.13],
    "1|2": [-0.24, -0.09, +0.13, -0.06, -0.69, -1.11, -0.48, +0.27, +0.44, -0.31],
    "1|3": [-0.15, -0.09, -0.48, -0.59, -0.34, -0.52, -0.51, +0.01, +0.14, -0.40],
    "1|4": [-0.65, -0.07, -0.42, -0.41, -0.71, -0.62, +0.02, -0.12, +0.25, -0.15],
    "1|5": [+0.24, +0.38, +0.63, +0.60, -0.38, -0.61, -0.16, +0.90, +0.34, +0.76],
    "2|1": [-0.47, -0.54, -0.53, -0.45, -0.37, -0.60, -0.58, -0.28, -0.28, -0.54],
    "2|2": [+0.49, +0.31, +0.54, +0.14, -0.09, -0.07, +0.47, +0.11, +0.72, +0.50],
    "2|3": [-0.01, +0.41, +0.45, +0.21, +0.14, -0.12, +0.08, +0.28, +0.48, +0.27],
    "2|4": [-0.08, +0.24, -0.34, -0.53, -0.62, -0.64, -0.34, -0.47, -0.24, -0.72],
    "2|5": [-0.13, +0.12, +0.96, +0.36, -0.31, -0.45, -0.01, +0.50, +0.62, +0.60],
    "3|1": [+0.03, -0.09, -0.50, -0.31, -0.29, -0.21, -0.72, +0.02, -1.06, -0.49],
    "3|2": [-1.11, -1.51, -1.89, -1.45, -1.64, -1.71, -2.01, -1.46, -2.17, -1.92],
    "3|4": [-0.53, -0.87, -1.52, -0.86, -1.30, -1.08, -1.46, -0.68, -1.56, -1.06],
    "3|5": [-0.62, -0.35, -2.08, -0.73, -1.89, -1.35, -2.30, -1.72, -2.31, -2.53],
    "4|1": [-0.21, -0.25, -0.25, -0.22, +0.09, +0.21, +0.17, +0.21, +0.29, -0.07],
    "4|2": [-0.20, -0.44, -0.63, -0.37, -0.15, -0.26, -0.41, -0.42, -0.49, -0.49],
    "4|3": [-0.06, -0.93, -1.73, -1.26, -1.06, -0.61, -1.34, -1.10, -1.40, -1.12],
    "4|4": [-0.18, -0.54, -1.26, -1.06, -0.30, -0.26, -0.77, -0.39, -0.21, -0.37],
    "4|5": [+0.21, +0.75, -0.44, +0.32, -0.24, -0.04, +0.17, +0.11, -0.50, -0.62],
    "5|1": [-0.23, +0.09, -0.19, -0.07, -0.36, +0.07, +0.06, -0.37, -0.19, -0.76],
    "5|2": [-0.62, -0.91, -1.03, -0.70, -0.88, -0.90, -1.87, -1.31, -1.56, -2.23],
    "5|3": [+0.03, -0.28, -0.90, -0.77, -1.31, -0.65, -1.06, -0.85, -0.61, -0.62],
    "5|4": [+0.34, -0.50, -0.24, -0.13, -0.06, +0.26, -0.43, -0.29, +0.06, -0.13],
})

# Medial glide trajectory profiles (rows: early u=0.15, mid u=0.475, late u=0.825; cols: 10 bands in nepers)
_MEDIAL_CENTERS = np.array([0.15, 0.475, 0.825], dtype=np.float64)
_MEDIAL_PROFILES: dict[str, np.ndarray] = {
    "i_medial": np.asarray([
        [+0.23, -0.14, -0.44, -0.51, +0.14, +0.45, +0.20, -0.04, +0.61, +0.17],
        [+0.10, +0.53, +0.95, +1.37, +1.37, +1.40, +1.34, +0.67, +0.66, +0.54],
        [+0.47, +0.47, +0.67, +0.79, +0.79, +0.81, +0.80, +0.62, +0.90, +0.77],
    ], dtype=np.float64) * _DB_TO_NEPER,
    "u_medial": np.asarray([
        [+0.12, +0.13, -0.09, -0.27, -0.61, -0.48, -0.24, -0.43, -0.09, -0.26],
        [+0.00, +0.47, +1.21, +1.31, +1.35, +1.14, +1.40, +0.96, +0.97, +0.79],
        [-0.01, +0.19, +0.16, +0.35, +0.41, +0.47, +0.36, +0.20, +0.43, +0.42],
    ], dtype=np.float64) * _DB_TO_NEPER,
    "v_medial": np.asarray([
        [+0.29, -0.41, -1.09, -1.10, -1.13, -1.23, -1.33, -1.22, -0.79, -0.84],
        [+0.35, +0.78, +0.91, +0.73, +1.70, +1.01, +0.06, -0.02, +0.00, -0.14],
        [+0.47, +0.25, -0.67, -0.23, +0.37, +0.30, -0.49, -0.67, -0.56, -0.68],
    ], dtype=np.float64) * _DB_TO_NEPER,
}

_MICRO_VEC1 = np.array([+0.5, +0.7, -0.9, -0.5, +0.8, +0.6, -0.4, -0.5, -0.3, -0.2], dtype=np.float64)
_MICRO_VEC2 = np.array([-0.3, +0.4, +0.7, -0.8, -0.6, +0.9, +0.5, -0.3, -0.4, -0.2], dtype=np.float64)

_I_MEDIAL_FINALS = frozenset({"ia", "ie", "iao", "iou", "iu", "ian", "iang", "iong"})
_U_MEDIAL_FINALS = frozenset({"ua", "uo", "uai", "ui", "uei", "uan", "uen", "un", "uang", "ueng"})
_V_MEDIAL_FINALS = frozenset({"ve", "üe", "van", "üan"})


def classify_medial_glide(final: str) -> str:
    """Return 'i_medial', 'u_medial', 'v_medial', or 'none' for Mandarin compound finals."""
    f = str(final).strip()
    if f in _I_MEDIAL_FINALS:
        return "i_medial"
    if f in _U_MEDIAL_FINALS:
        return "u_medial"
    if f in _V_MEDIAL_FINALS:
        return "v_medial"
    return "none"


def _interp_3x10(profile: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Linearly interpolate a (3, 10) phase-by-band profile at normalized phases `u`."""
    uc = np.clip(u, _MEDIAL_CENTERS[0], _MEDIAL_CENTERS[-1])
    hi = np.clip(np.searchsorted(_MEDIAL_CENTERS, uc, side="right"), 1, 2)
    lo = hi - 1
    denom = np.maximum(1e-9, _MEDIAL_CENTERS[hi] - _MEDIAL_CENTERS[lo])
    w = ((uc - _MEDIAL_CENTERS[lo]) / denom)[:, None]
    return (1.0 - w) * profile[lo] + w * profile[hi]


def _is_unblurred_base(base: LarynxControl) -> bool:
    """Detect whether base control already carries unblurred Stage 2 medoid detail (cep[24:64])."""
    vm = np.asarray(base.voiced, dtype=bool) & (np.asarray(base.f0_hz, dtype=np.float64) > 50.0)
    if int(np.sum(vm)) < 5:
        return False
    c24_64 = base.log_env[vm] @ _CEP_DCT24_64
    return float(np.mean(np.std(c24_64, axis=0))) >= 0.42


class RuleBoundaryCoarticulation:
    """Dual-regime coarticulation, adaptive cep[0:18] seam bridging, and sustained-vowel micro-dynamics."""

    name = "boundary_coarticulation"

    def __init__(
        self,
        half_win: int = 3,
        jump_thresh_nep: float = 0.16,
        *,
        right_gain: float = 0.80,
        right_s2_gain: float = 0.45,
        left_on_gain: float = 0.18,
        left_fin_gain: float = 0.20,
        tone_pair_gain: float = 0.60,
        tone_pair_s2_gain: float = 0.40,
        medial_gain: float = 0.65,
        neutral_cent_gain: float = 0.10,
        micro_amp_nep: float = 0.026,
        micro_tilt_db: float = 0.14,
        micro_freq_hz: float = 5.2,
        bridge_blend_max: float = 0.82,
    ) -> None:
        self.half_win = int(half_win)
        self.jump_thresh_nep = float(jump_thresh_nep)
        self.right_gain = float(right_gain)
        self.right_s2_gain = float(right_s2_gain)
        self.left_on_gain = float(left_on_gain)
        self.left_fin_gain = float(left_fin_gain)
        self.tone_pair_gain = float(tone_pair_gain)
        self.tone_pair_s2_gain = float(tone_pair_s2_gain)
        self.medial_gain = float(medial_gain)
        self.neutral_cent_gain = float(neutral_cent_gain)
        self.micro_amp_nep = float(micro_amp_nep)
        self.micro_tilt_db = float(micro_tilt_db)
        self.micro_freq_hz = float(micro_freq_hz)
        self.bridge_blend_max = float(bridge_blend_max)

    def propose(self, ctx: UtteranceContext, base: LarynxControl) -> RuleProposal | None:
        n_fr = len(base.f0_hz)
        syls = ctx.syllables
        if not syls or n_fr == 0:
            return None

        is_s2 = _is_unblurred_base(base)
        mask = np.zeros(n_fr, dtype=bool)
        d_band = np.zeros((n_fr, _BAND_BASIS.shape[0]), dtype=np.float64)
        d_cep18 = np.zeros((n_fr, _CEP_BRIDGE_K), dtype=np.float64)
        d_env = np.zeros((n_fr, N_BINS), dtype=np.float64)
        d_tilt = np.zeros(n_fr, dtype=np.float64)
        has_tilt = False
        w = self.half_win

        for i, s in enumerate(syls):
            a = max(0, min(n_fr, int(s.start_fr)))
            b = max(a, min(n_fr, int(s.end_fr)))
            sp = max(a, min(b, int(s.split_fr)))
            if b - a < 3:
                continue

            f_a = sp if (s.onset and sp > a) else a
            o_man = classify_onset_manner(s.onset)
            o_plc = classify_onset_place(s.onset)
            f_grp = classify_final_group(s.final)
            is_obs = o_man in OBSTRUENT_MANNERS

            # 1. Left coarticulation from previous syllable (protect obstruent closure/burst starts)
            if i > 0:
                sprev = syls[i - 1]
                gap_prev = a - int(sprev.end_fr)
                if 0 <= gap_prev < 3 and float(sprev.pause_after_ms) < 25.0 and (sprev.end_fr - sprev.start_fr) >= 3:
                    p_grp = classify_final_group(sprev.final)
                    if (not is_s2) and (not is_obs) and self.left_on_gain > 0.0 and s.onset and sp > a:
                        prof_lo = _LEFT_ON_PROFILES.get(f"{p_grp}|{o_plc}")
                        if prof_lo is not None:
                            u_on = np.asarray(ctx.phase_u[a:sp], dtype=np.float64)
                            d_band[a:sp] += (self.left_on_gain * (1.0 - 0.5 * u_on))[:, None] * prof_lo[None, :]
                            mask[a:sp] = True

                    if b > f_a:
                        u_f = np.asarray(ctx.phase_u[f_a:b], dtype=np.float64)
                        w_early = np.clip((0.40 - u_f) / 0.40, 0.0, 1.0)
                        e_m = w_early > 0.0
                        if np.any(e_m):
                            idx_e = np.arange(f_a, b)[e_m]
                            if (not is_s2) and (not is_obs) and self.left_fin_gain > 0.0:
                                prof_lf = _LEFT_FIN_PROFILES.get(f"{p_grp}|{o_plc}")
                                if prof_lf is not None:
                                    d_band[idx_e] += (self.left_fin_gain * w_early[e_m])[:, None] * prof_lf[None, :]
                                    mask[idx_e] = True
                            tp_tbl = _TONE_PAIR_S2_PROFILES if is_s2 else _TONE_PAIR_PROFILES
                            tp_g = self.tone_pair_s2_gain if is_s2 else self.tone_pair_gain
                            if tp_g > 0.0:
                                prof_tp = tp_tbl.get(f"{int(sprev.tone)}|{int(s.tone)}")
                                if prof_tp is not None:
                                    d_band[idx_e] += (tp_g * w_early[e_m])[:, None] * prof_tp[None, :]
                                    mask[idx_e] = True

            # 2. Right anticipatory coarticulation on late final (u >= 0.55)
            if i < len(syls) - 1 and b > f_a:
                snxt = syls[i + 1]
                gap_nxt = int(snxt.start_fr) - b
                if 0 <= gap_nxt < 3 and float(s.pause_after_ms) < 25.0 and (snxt.end_fr - snxt.start_fr) >= 3:
                    n_plc = classify_onset_place(snxt.onset)
                    if is_s2:
                        prof_rf = _RIGHT_PLC_S2_SHAPED.get(n_plc)
                        rg = self.right_s2_gain
                    else:
                        prof_rf = _RIGHT_FIN_PROFILES.get(f"{f_grp}|{n_plc}")
                        if prof_rf is None:
                            prof_rf = _RIGHT_PLACE_FALLBACK.get(n_plc)
                        rg = self.right_gain
                    if prof_rf is not None and rg > 0.0:
                        u_f = np.asarray(ctx.phase_u[f_a:b], dtype=np.float64)
                        w_late = np.clip((u_f - 0.55) / 0.45, 0.0, 1.0)
                        l_m = w_late > 0.0
                        if np.any(l_m):
                            idx_l = np.arange(f_a, b)[l_m]
                            d_band[idx_l] += (rg * w_late[l_m])[:, None] * prof_rf[None, :]
                            mask[idx_l] = True

            # 3. Medial glide trajectory shaping (5-bin blurred base only; Stage 2 9-bin medoids capture medials)
            if (not is_s2) and self.medial_gain > 0.0 and b > f_a:
                mk = classify_medial_glide(s.final)
                if mk != "none":
                    u_f = np.asarray(ctx.phase_u[f_a:b], dtype=np.float64)
                    d_band[f_a:b] += self.medial_gain * _interp_3x10(_MEDIAL_PROFILES[mk], u_f)
                    mask[f_a:b] = True

            # 4. Neutral tone centralization (5-bin blurred base only)
            if (not is_s2) and self.neutral_cent_gain > 0.0 and int(s.tone) == 5 and b > f_a:
                u_f = np.asarray(ctx.phase_u[f_a:b], dtype=np.float64)
                w_c = self.neutral_cent_gain * np.clip((u_f - 0.25) / 0.75, 0.0, 1.0)
                if np.any(w_c > 0.0):
                    seg = base.log_env[f_a:b]
                    d_env[f_a:b] += w_c[:, None] * (np.mean(seg, axis=1, keepdims=True) - seg)
                    mask[f_a:b] = True

            # 5. Sustained-vowel physiological micro-dynamics (len >= 20 frames, 0.2 < u < 0.8)
            n_fin = b - f_a
            if (self.micro_amp_nep > 0.0 or self.micro_tilt_db > 0.0) and n_fin >= 20:
                u_f = np.asarray(ctx.phase_u[f_a:b], dtype=np.float64)
                mid_m = (u_f > 0.2) & (u_f < 0.8)
                if np.any(mid_m):
                    idx_m = np.arange(f_a, b)[mid_m]
                    t_sec = np.arange(n_fin, dtype=np.float64)[mid_m] * 0.005
                    env_w = np.sin(math.pi * (u_f[mid_m] - 0.2) / 0.6) ** 2
                    phi0 = 0.7 * float(s.index)
                    w1 = np.sin(2.0 * math.pi * self.micro_freq_hz * t_sec + phi0) * env_w
                    w2 = np.cos(2.0 * math.pi * (self.micro_freq_hz * 0.63) * t_sec + 1.3 * phi0) * env_w
                    if self.micro_amp_nep > 0.0:
                        d_band[idx_m] += (
                            (self.micro_amp_nep * w1)[:, None] * _MICRO_VEC1[None, :]
                            + (0.7 * self.micro_amp_nep * w2)[:, None] * _MICRO_VEC2[None, :]
                        )
                    if self.micro_tilt_db > 0.0:
                        d_tilt[idx_m] += self.micro_tilt_db * w1
                        has_tilt = True
                    mask[idx_m] = True

            # 6. Adaptive cross-syllable cep[0:18] S-curve bridging over ±half_win frames at sonorant seams
            if w > 0 and i < len(syls) - 1:
                snxt = syls[i + 1]
                gap_nxt = int(snxt.start_fr) - b
                if 0 <= gap_nxt < 3 and float(s.pause_after_ms) < 25.0 and (snxt.end_fr - snxt.start_fr) >= 3:
                    if classify_onset_manner(snxt.onset) not in OBSTRUENT_MANNERS:
                        bnd = int(snxt.start_fr)
                        i0 = max(a + 1, bnd - w)
                        i1 = min(int(snxt.end_fr) - 1, bnd + w)
                        if i1 > i0 + 1:
                            left_idx = max(0, i0 - 1)
                            right_idx = min(n_fr - 1, i1)
                            span_env = np.vstack([
                                base.log_env[left_idx : left_idx + 1],
                                base.log_env[i0:i1],
                                base.log_env[right_idx : right_idx + 1],
                            ])
                            span_cep = span_env @ _CEP_DCT18
                            jump = float(np.sqrt(np.sum((span_cep[-1] - span_cep[0]) ** 2) / N_BINS))
                            if jump >= self.jump_thresh_nep:
                                m = i1 - i0
                                u_s = np.linspace(1.0 / (m + 1), m / (m + 1), m)
                                alpha = 0.5 * (1.0 - np.cos(math.pi * u_s))[:, None]
                                target_cep = (1.0 - alpha) * span_cep[0:1] + alpha * span_cep[-1:]
                                bmax = self.bridge_blend_max
                                blend = float(np.clip(0.35 + (bmax - 0.35) * (jump - self.jump_thresh_nep) / 0.45, 0.35, bmax))
                                d_cep18[i0:i1] += blend * (target_cep - span_cep[1:-1])
                                mask[i0:i1] = True

        if not np.any(mask):
            return None

        d_env[mask] += d_band[mask] @ _BAND_BASIS + d_cep18[mask] @ _CEP_DCT18.T
        return RuleProposal(
            module_name=self.name,
            mask=mask,
            delta_log_env=d_env,
            delta_tilt=d_tilt if has_tilt else None,
        )
