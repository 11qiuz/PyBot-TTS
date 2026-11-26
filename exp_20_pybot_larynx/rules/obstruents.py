"""Rule 2: Place/manner-specific obstruent shaping, 513-bin hi-res medoids, and clean VOT cutoffs."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.fft import rfftfreq

from exp_20_pybot_larynx import EXP_DIR, N_BAP, N_BINS, N_CEP, N_FFT, SR
from exp_20_pybot_larynx.arbiter import (
    OBSTRUENT_MANNERS,
    RuleProposal,
    UtteranceContext,
    classify_final_group,
    classify_onset_manner,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env, log_env_to_cep

_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_PHASE_CENTERS = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=np.float64)
_M_1_11 = (_FREQS >= 1000.0) & (_FREQS <= 11000.0)
HI_RES_PATH = EXP_DIR / "rules" / "obstruent_hi_res.npz"
_Q_SPLIT = 40
_Q_BLUR_REF = 24


def _build_smooth_band_basis(freqs: np.ndarray) -> np.ndarray:
    """Build 5 C^1 partition-of-unity frequency band basis functions (<500, 500-2k, 2-4k, 4-8k, >8k)."""
    cutoffs = ((500.0, 150.0), (2000.0, 400.0), (4000.0, 600.0), (8000.0, 1000.0))
    steps = []
    for fc, hw in cutoffs:
        u = np.clip((freqs - fc) / hw, -1.0, 1.0)
        steps.append(0.5 * (1.0 + np.sin(0.5 * math.pi * u)))
    s0, s1, s2, s3 = steps
    return np.stack(
        [
            1.0 - s0,
            s0 * (1.0 - s1),
            s1 * (1.0 - s2),
            s2 * (1.0 - s3),
            s3,
        ],
        axis=0,
    )  # (5, N_BINS)


_BAND_BASIS = _build_smooth_band_basis(_FREQS)

# Non-initial (s.index > 0) 5-phase x 5-band residual profiles (nepers) for 3-bin averaged KBs.
# Rows: u centers [0.10, 0.30, 0.50, 0.70, 0.90]; Cols: [<500, 500-2k, 2-4k, 4-8k, >8k].
_NON_INIT_PROFILES: dict[str, np.ndarray] = {
    "b": np.array([
        [+0.35, +0.22, +0.18, +0.14, +0.15],
        [-0.05, -0.20, -0.18, -0.15, -0.15],
        [-0.20, -0.34, -0.37, -0.34, -0.31],
        [-0.85, -1.00, -0.98, -0.85, -0.78],
        [+0.36, +0.40, +0.37, +0.29, +0.29],
    ], dtype=np.float64),
    "d": np.array([
        [+0.40, +0.28, +0.17, +0.16, +0.16],
        [-0.18, -0.38, -0.44, -0.40, -0.42],
        [-0.24, -0.65, -0.74, -0.68, -0.70],
        [-0.75, -1.10, -1.10, -1.00, -1.04],
        [+0.24, +0.28, +0.27, +0.27, +0.28],
    ], dtype=np.float64),
    "g": np.array([
        [+0.38, +0.08, -0.11, -0.04, +0.00],
        [-0.26, -0.50, -0.50, -0.40, -0.38],
        [-0.34, -0.58, -0.50, -0.41, -0.38],
        [-0.25, +0.20, +0.12, +0.02, +0.07],
        [+0.09, +0.01, -0.03, +0.00, +0.00],
    ], dtype=np.float64),
    "p": np.array([
        [+0.15, -0.19, -0.36, -0.38, -0.39],
        [+0.17, +0.10, +0.11, +0.09, +0.02],
        [+0.18, +0.14, +0.12, +0.11, +0.10],
        [-0.12, +0.12, +0.08, +0.12, +0.13],
        [+0.18, +0.04, -0.10, -0.11, -0.09],
    ], dtype=np.float64),
    "t": np.array([
        [+0.12, -0.42, -0.58, -0.64, -0.65],
        [+0.24, +0.32, +0.50, +0.52, +0.54],
        [+0.05, +0.16, +0.20, +0.24, +0.25],
        [-0.28, +0.02, +0.04, +0.11, +0.11],
        [+0.10, -0.05, -0.16, -0.15, -0.14],
    ], dtype=np.float64),
    "k": np.array([
        [+0.20, -0.36, -0.26, -0.32, -0.33],
        [-0.12, +0.30, +0.25, +0.15, +0.18],
        [+0.10, +0.25, +0.21, +0.17, +0.18],
        [-0.18, -0.01, +0.07, +0.04, -0.03],
        [-0.03, +0.00, +0.01, -0.05, -0.13],
    ], dtype=np.float64),
    "z": np.array([
        [+0.42, +0.04, -0.12, -0.19, -0.29],
        [-0.28, -0.12, -0.06, -0.05, -0.01],
        [-0.21, -0.12, -0.08, -0.09, -0.02],
        [-0.36, -0.09, -0.03, +0.07, +0.34],
        [+0.16, +0.11, +0.09, +0.06, -0.05],
    ], dtype=np.float64),
    "zh": np.array([
        [+0.70, +0.20, -0.06, -0.17, -0.05],
        [-0.02, -0.01, -0.13, -0.16, -0.21],
        [+0.05, +0.06, -0.02, -0.06, -0.09],
        [-0.21, +0.09, +0.25, +0.48, +0.49],
        [+0.14, +0.05, -0.04, -0.10, -0.05],
    ], dtype=np.float64),
    "j": np.array([
        [+0.60, +0.08, -0.06, -0.25, -0.22],
        [-0.16, +0.00, +0.06, +0.11, +0.22],
        [-0.09, +0.00, -0.02, +0.05, +0.15],
        [-0.27, +0.03, +0.04, +0.36, +0.42],
        [+0.12, +0.03, +0.03, -0.14, -0.17],
    ], dtype=np.float64),
    "c": np.array([
        [+0.55, -0.07, -0.11, -0.16, -0.26],
        [-0.30, +0.04, +0.15, +0.24, +0.52],
        [-0.02, +0.04, +0.08, +0.14, +0.33],
        [-0.06, +0.03, +0.13, +0.16, +0.16],
        [+0.08, -0.04, -0.11, -0.16, -0.25],
    ], dtype=np.float64),
    "ch": np.array([
        [+0.36, -0.24, -0.52, -0.58, -0.48],
        [-0.28, +0.20, +0.72, +0.84, +0.60],
        [-0.02, +0.03, +0.30, +0.30, +0.25],
        [-0.20, -0.04, +0.07, -0.09, -0.03],
        [-0.04, -0.05, -0.13, -0.18, -0.14],
    ], dtype=np.float64),
    "q": np.array([
        [+0.52, -0.02, -0.34, -0.62, -0.55],
        [-0.42, -0.02, +0.12, +0.47, +0.52],
        [-0.12, +0.02, +0.04, +0.25, +0.19],
        [-0.08, +0.12, +0.15, +0.24, +0.15],
        [+0.07, +0.00, -0.03, -0.21, -0.24],
    ], dtype=np.float64),
    "f": np.array([
        [+0.42, +0.08, -0.04, -0.09, -0.10],
        [-0.28, -0.13, -0.12, -0.10, -0.06],
        [-0.13, -0.16, -0.16, -0.17, -0.12],
        [-0.38, -0.22, -0.14, -0.13, -0.02],
        [+0.07, +0.05, +0.01, -0.04, -0.12],
    ], dtype=np.float64),
    "s": np.array([
        [+0.68, +0.15, +0.06, +0.11, +0.04],
        [-0.15, -0.02, +0.01, +0.03, +0.17],
        [-0.03, -0.03, -0.06, -0.05, +0.03],
        [-0.26, -0.08, -0.03, +0.04, +0.24],
        [+0.24, +0.13, +0.06, +0.00, -0.17],
    ], dtype=np.float64),
    "sh": np.array([
        [+0.56, +0.03, -0.20, -0.30, -0.25],
        [-0.35, +0.02, +0.24, +0.31, +0.28],
        [-0.14, -0.03, +0.04, +0.07, +0.08],
        [-0.34, -0.06, +0.13, +0.23, +0.22],
        [+0.17, +0.04, -0.08, -0.15, -0.12],
    ], dtype=np.float64),
    "x": np.array([
        [+0.56, +0.12, +0.00, -0.21, -0.25],
        [-0.29, +0.01, +0.05, +0.24, +0.30],
        [-0.09, -0.01, -0.01, +0.08, +0.10],
        [-0.32, -0.04, +0.01, +0.22, +0.21],
        [+0.12, +0.03, +0.01, -0.21, -0.23],
    ], dtype=np.float64),
    "h": np.array([
        [+0.27, -0.07, -0.12, -0.14, -0.12],
        [-0.27, -0.09, -0.11, -0.05, -0.03],
        [-0.07, -0.13, -0.14, -0.09, -0.06],
        [-0.18, -0.17, -0.12, -0.08, -0.03],
        [+0.03, -0.09, -0.14, -0.14, -0.15],
    ], dtype=np.float64),
}

# Stage 2 KB (5-bin medoid + cep[:17] anchor) residual profiles measured on 250 TRAIN_IDS utts.
# Captures remaining b/d/g closure excess and place-specific burst/slope offsets on sharp bases.
_STAGE2_SHARP_PROFILES: dict[str, np.ndarray] = {
    "b": np.array([
        [+0.12, +0.10, +0.10, +0.12, +0.15],
        [-0.04, -0.06, -0.07, -0.05, -0.02],
        [-0.24, -0.52, -0.52, -0.50, -0.50],
        [-1.00, -1.35, -1.40, -1.30, -1.25],
        [-0.25, -0.25, -0.22, -0.16, -0.18],
    ], dtype=np.float64),
    "d": np.array([
        [+0.06, +0.11, +0.11, +0.12, +0.10],
        [-0.15, -0.14, -0.06, -0.09, -0.10],
        [-0.18, -0.44, -0.44, -0.44, -0.40],
        [-0.82, -1.30, -1.30, -1.22, -1.22],
        [-0.20, -0.20, -0.14, -0.20, -0.25],
    ], dtype=np.float64),
    "g": np.array([
        [+0.02, -0.12, -0.06, -0.08, -0.09],
        [-0.18, -0.38, -0.34, -0.30, -0.28],
        [-0.36, -0.68, -0.64, -0.58, -0.54],
        [-0.45, -0.46, -0.38, -0.38, -0.36],
        [-0.10, +0.06, +0.00, -0.02, +0.02],
    ], dtype=np.float64),
    "p": np.array([
        [-0.06, -0.26, -0.28, -0.36, -0.40],
        [-0.22, -0.45, -0.40, -0.40, -0.36],
        [+0.20, +0.02, +0.14, +0.08, +0.06],
        [-0.10, -0.06, +0.00, +0.04, +0.00],
        [-0.24, -0.12, -0.12, -0.02, -0.06],
    ], dtype=np.float64),
    "t": np.array([
        [-0.10, -0.28, -0.28, -0.30, -0.34],
        [-0.08, -0.10, -0.06, -0.02, -0.03],
        [-0.04, +0.01, +0.04, +0.00, +0.03],
        [-0.18, -0.02, -0.02, -0.02, +0.02],
        [-0.12, -0.04, -0.04, -0.04, -0.03],
    ], dtype=np.float64),
    "k": np.array([
        [+0.02, -0.20, -0.04, -0.05, -0.05],
        [-0.22, -0.16, -0.02, -0.08, -0.02],
        [+0.01, +0.02, -0.01, -0.03, +0.03],
        [-0.05, -0.05, -0.02, -0.05, -0.04],
        [-0.03, -0.08, -0.05, -0.04, -0.07],
    ], dtype=np.float64),
    "z": np.array([
        [-0.12, -0.12, -0.14, -0.18, -0.20],
        [-0.22, -0.16, -0.14, -0.15, -0.18],
        [-0.16, -0.18, -0.18, -0.20, -0.20],
        [-0.24, -0.18, -0.15, -0.15, -0.02],
        [-0.08, -0.05, -0.02, +0.01, +0.02],
    ], dtype=np.float64),
    "c": np.array([
        [+0.15, +0.07, +0.03, +0.10, +0.14],
        [-0.09, +0.00, -0.01, +0.05, +0.14],
        [+0.01, -0.01, +0.02, +0.05, +0.15],
        [-0.03, +0.01, +0.02, +0.13, +0.18],
        [+0.00, +0.08, +0.01, +0.04, +0.08],
    ], dtype=np.float64),
    "zh": np.array([
        [+0.08, -0.08, -0.10, -0.14, -0.04],
        [-0.08, -0.22, -0.30, -0.32, -0.32],
        [-0.06, -0.10, -0.28, -0.36, -0.38],
        [-0.15, -0.08, -0.08, -0.01, -0.01],
        [-0.12, +0.02, +0.04, +0.07, +0.05],
    ], dtype=np.float64),
    "ch": np.array([
        [+0.18, -0.06, -0.12, -0.10, -0.08],
        [-0.14, -0.08, +0.08, +0.12, +0.05],
        [-0.04, -0.04, +0.16, +0.18, +0.14],
        [-0.04, -0.04, +0.14, +0.10, +0.07],
        [-0.08, -0.04, +0.05, +0.02, -0.03],
    ], dtype=np.float64),
    "j": np.array([
        [+0.18, +0.02, -0.02, +0.01, +0.02],
        [+0.02, -0.09, -0.11, -0.06, -0.01],
        [+0.01, -0.06, -0.10, -0.07, +0.02],
        [-0.14, -0.07, -0.10, +0.10, +0.18],
        [-0.01, -0.02, -0.07, +0.00, +0.02],
    ], dtype=np.float64),
    "q": np.array([
        [+0.16, +0.04, -0.08, -0.18, -0.11],
        [-0.16, +0.02, -0.02, +0.03, +0.10],
        [-0.09, -0.01, -0.06, +0.08, +0.09],
        [-0.07, +0.00, +0.00, +0.08, +0.09],
        [-0.02, +0.02, +0.02, -0.02, -0.02],
    ], dtype=np.float64),
    "f": np.array([
        [+0.20, -0.02, -0.01, +0.02, +0.01],
        [-0.03, -0.06, -0.02, +0.02, +0.08],
        [+0.05, -0.08, -0.10, -0.06, +0.01],
        [-0.06, -0.15, -0.17, -0.10, +0.00],
        [-0.10, -0.10, -0.08, -0.02, +0.03],
    ], dtype=np.float64),
    "s": np.array([
        [+0.14, +0.02, +0.04, +0.06, +0.01],
        [-0.03, +0.00, +0.06, +0.04, +0.06],
        [+0.00, -0.04, -0.02, -0.03, -0.02],
        [-0.17, -0.10, -0.06, -0.03, +0.02],
        [-0.12, -0.06, -0.03, +0.01, +0.02],
    ], dtype=np.float64),
    "sh": np.array([
        [+0.14, +0.08, -0.05, -0.05, -0.03],
        [-0.06, +0.04, +0.06, +0.10, +0.10],
        [+0.01, +0.00, +0.00, +0.00, +0.02],
        [-0.11, -0.03, +0.02, +0.07, +0.07],
        [-0.05, +0.01, +0.00, +0.04, +0.05],
    ], dtype=np.float64),
    "x": np.array([
        [+0.12, +0.04, -0.08, -0.10, -0.06],
        [-0.04, +0.00, -0.05, +0.02, +0.10],
        [+0.02, -0.05, -0.09, +0.02, +0.08],
        [-0.14, -0.07, -0.05, +0.13, +0.18],
        [-0.02, +0.00, +0.02, +0.06, +0.08],
    ], dtype=np.float64),
    "h": np.array([
        [+0.14, +0.05, +0.02, +0.02, -0.01],
        [+0.02, +0.05, +0.06, +0.05, +0.03],
        [+0.10, +0.06, +0.04, +0.06, +0.03],
        [-0.02, +0.05, +0.07, +0.08, +0.05],
        [+0.02, +0.07, +0.07, +0.06, +0.04],
    ], dtype=np.float64),
}

# Utterance-initial (s.index == 0) onset silence-to-burst ramp profiles (nepers) across [0.1, 0.3, 0.5, 0.7, 0.9]
_INIT_ATTEN_CURVES: dict[str, np.ndarray] = {
    "stop_unasp": np.array([-2.60, -1.85, -1.40, -1.10, -0.05], dtype=np.float64),
    "affr_unasp": np.array([-2.40, -1.75, -1.10, -0.35, -0.05], dtype=np.float64),
    "stop_asp":   np.array([-2.20, -1.40, -0.30, -0.05, +0.00], dtype=np.float64),
    "affr_asp":   np.array([-2.40, -1.45, -0.35, +0.10, +0.00], dtype=np.float64),
    "fric":       np.array([-2.00, -0.90, -0.25, -0.05, +0.00], dtype=np.float64),
}
_STAGE2_INIT_CURVE = np.array([-0.36, -0.30, -0.13, -0.32, -0.05], dtype=np.float64)

# Unvoiced core windows [u_lo, u_hi] for s.index > 0 (avoids false unvoicing on voice-bars / VOT)
_CORE_UV_WINDOWS: dict[str, tuple[float, float]] = {
    "p": (0.20, 0.78),
    "t": (0.20, 0.78),
    "k": (0.20, 0.78),
    "c": (0.20, 0.78),
    "ch": (0.20, 0.78),
    "q": (0.20, 0.78),
    "s": (0.20, 0.78),
    "sh": (0.20, 0.78),
    "x": (0.20, 0.78),
    "z": (0.26, 0.76),
    "zh": (0.26, 0.76),
    "j": (0.26, 0.76),
    "f": (0.26, 0.70),
    "h": (0.26, 0.70),
    "b": (0.26, 0.55),
    "g": (0.26, 0.55),
    "d": (0.32, 0.52),
}


def _interp_5x5(profile: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Linearly interpolate `(5, D)` phase profile at normalized phases `u in [0, 1]`."""
    uc = np.clip(u, _PHASE_CENTERS[0], _PHASE_CENTERS[-1])
    hi = np.clip(np.searchsorted(_PHASE_CENTERS, uc, side="right"), 1, 4)
    lo = hi - 1
    w = ((uc - _PHASE_CENTERS[lo]) / 0.20)[:, None]
    return (1.0 - w) * profile[lo] + w * profile[hi]


def _nearest_5bins(arr5: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Select nearest phase-bin slice from `(5, D)` array without blurring fine spectral peaks."""
    idx = np.argmin(np.abs(u[:, None] - _PHASE_CENTERS[None, :]), axis=1)
    return arr5[idx]


def _interp_1d(curve: np.ndarray, u: np.ndarray) -> np.ndarray:
    uc = np.clip(u, _PHASE_CENTERS[0], _PHASE_CENTERS[-1])
    hi = np.clip(np.searchsorted(_PHASE_CENTERS, uc, side="right"), 1, 4)
    lo = hi - 1
    w = (uc - _PHASE_CENTERS[lo]) / 0.20
    return (1.0 - w) * curve[lo] + w * curve[hi]


def _decompose_medoid_513(env5: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split `(5, 513)` medoid into broad `cep[:40]`, mid `cep[40:64]`, hi `q>=64`, and `cep[24:64]` RMS ref."""
    e = env5.astype(np.float64)
    c64 = log_env_to_cep(e, n_cep=N_CEP)
    e_broad = cep_to_log_env(c64[:, :_Q_SPLIT], n_bins=N_BINS)
    e_c64 = cep_to_log_env(c64, n_bins=N_BINS)
    e_mid = e_c64 - e_broad
    e_hi = e - e_c64
    e_mid24 = e_c64 - cep_to_log_env(c64[:, :_Q_BLUR_REF], n_bins=N_BINS)
    ref_rms24 = np.sqrt(np.mean(e_mid24[:, _M_1_11] ** 2, axis=1))
    return e_broad, e_mid, e_hi, ref_rms24


def _load_hi_res_tables(
    path: Path = HI_RES_PATH,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    if not path.exists():
        return {}, {}
    with np.load(path, allow_pickle=False) as d:
        ofg = {
            str(k): (*_decompose_medoid_513(env5), bap5.astype(np.float64))
            for k, env5, bap5 in zip(d["ofg_k"], d["ofg_env"], d["ofg_bap"])
        }
        o_only = {
            str(k): (*_decompose_medoid_513(env5), bap5.astype(np.float64))
            for k, env5, bap5 in zip(d["o_k"], d["o_env"], d["o_bap"])
        }
    return ofg, o_only


_HI_OFG, _HI_O = _load_hi_res_tables()


class RuleObstruentCrispness:
    """Restore sharp 513-bin unvoiced medoid structure, burst contrast, and clean VOT cutoffs."""

    name = "obstruent_crispness"

    def __init__(
        self,
        shape_gain: float = 0.85,
        init_gain: float = 0.85,
        w_broad_max: float = 0.45,
        w_mid_max: float = 0.90,
        w_hi: float = 1.05,
        comp_alpha: float = 0.35,
        s2_gain: float = 0.95,
    ) -> None:
        self.shape_gain = float(shape_gain)
        self.init_gain = float(init_gain)
        self.w_broad_max = float(w_broad_max)
        self.w_mid_max = float(w_mid_max)
        self.w_hi = float(w_hi)
        self.comp_alpha = float(comp_alpha)
        self.s2_gain = float(s2_gain)

    def propose(self, ctx: UtteranceContext, base: LarynxControl) -> RuleProposal | None:
        n_fr = len(base.f0_hz)
        if n_fr == 0 or not ctx.syllables:
            return None

        mask = np.zeros(n_fr, dtype=bool)
        d_env = np.zeros((n_fr, N_BINS), dtype=np.float64)
        d_bap = np.zeros((n_fr, N_BAP), dtype=np.float64)
        ov = np.full(n_fr, -1, dtype=np.int8)

        spans_info: list[tuple[int, int, object, str]] = []
        obs_frame_mask = np.zeros(n_fr, dtype=bool)
        for s in ctx.syllables:
            onset = str(s.onset)
            manner = classify_onset_manner(onset)
            if manner not in OBSTRUENT_MANNERS:
                continue
            a = max(0, min(n_fr, int(s.start_fr)))
            sp = max(a, min(n_fr, int(s.split_fr)))
            if sp <= a:
                continue
            spans_info.append((a, sp, s, manner))
            obs_frame_mask[a:sp] = True

        if not spans_info:
            return None

        obs_cep_all = np.zeros((n_fr, N_CEP), dtype=np.float64)
        obs_mid24_rms = np.zeros(n_fr, dtype=np.float64)
        if _HI_O:
            c_obs = log_env_to_cep(base.log_env[obs_frame_mask], n_cep=N_CEP)
            obs_cep_all[obs_frame_mask] = c_obs
            e_mid24 = cep_to_log_env(c_obs, n_bins=N_BINS) - cep_to_log_env(c_obs[:, :_Q_BLUR_REF], n_bins=N_BINS)
            obs_mid24_rms[obs_frame_mask] = np.sqrt(np.mean(e_mid24[:, _M_1_11] ** 2, axis=1))

        for a, sp, s, manner in spans_info:
            onset = str(s.onset)
            n_on = sp - a
            u_seg = ctx.phase_u[a:sp]
            idx_seg = np.arange(a, sp)

            # 1. Clean voiced/unvoiced VOT cutoff on the genuinely unvoiced core
            if s.index == 0:
                u_lo, u_hi = 0.0, (0.62 if manner == "stop_unasp" else 0.72)
            else:
                u_lo, u_hi = _CORE_UV_WINDOWS.get(onset, (0.25, 0.75))

            uv_sub = (u_seg >= u_lo) & (u_seg <= u_hi)
            fg = classify_final_group(s.final)
            entry = _HI_OFG.get(f"{onset}|{fg}") or _HI_O.get(onset)

            if np.any(uv_sub):
                idx_uv = idx_seg[uv_sub]
                mask[idx_uv] = True
                ov[idx_uv] = 0
                if entry is not None:
                    bap_target = np.maximum(0.985, _interp_5x5(entry[4], u_seg[uv_sub]))
                    d_bap[idx_uv] = bap_target - base.bap[idx_uv]
                else:
                    d_bap[idx_uv] = 0.99 - base.bap[idx_uv]

            # 2. Detect whether base is blurred (3-bin averaged KB) vs unblurred medoid (Stage 2 KB)
            if entry is not None:
                m_broad5, m_mid5, m_hi5, m_rms24_5, _ = entry
                m_rms_ref = _nearest_5bins(m_rms24_5, u_seg)
                b_rms = float(np.mean(obs_mid24_rms[a:sp]))
                m_rms = float(np.mean(m_rms_ref))
                blur_ratio = float(np.clip(1.0 - b_rms / max(m_rms, 1e-4), 0.0, 1.0))
                is_sharp_base = blur_ratio < 0.25
            else:
                blur_ratio = 1.0
                is_sharp_base = False

            # 3. Closure/burst/initial shaping calibrated separately for blurred vs sharp bases
            if not is_sharp_base:
                if s.index == 0:
                    curve = _INIT_ATTEN_CURVES[manner]
                    scale = self.init_gain * (0.6 if n_on <= 2 else 1.0)
                    mask[idx_seg] = True
                    d_env[idx_seg] += (scale * _interp_1d(curve, u_seg))[:, None]
                else:
                    prof = _NON_INIT_PROFILES.get(onset)
                    if prof is not None:
                        band_deltas = self.shape_gain * _interp_5x5(prof, u_seg)
                        if n_on <= 2:
                            band_deltas *= 0.65
                        mask[idx_seg] = True
                        d_env[idx_seg] += band_deltas @ _BAND_BASIS
            elif self.s2_gain > 0.0:
                if s.index == 0:
                    scale = self.s2_gain * (0.6 if n_on <= 2 else 1.0)
                    mask[idx_seg] = True
                    d_env[idx_seg] += (scale * _interp_1d(_STAGE2_INIT_CURVE, u_seg))[:, None]
                else:
                    prof2 = _STAGE2_SHARP_PROFILES.get(onset)
                    if prof2 is not None:
                        band_deltas = self.s2_gain * _interp_5x5(prof2, u_seg)
                        if n_on <= 2:
                            band_deltas *= 0.65
                        mask[idx_seg] = True
                        d_env[idx_seg] += band_deltas @ _BAND_BASIS

            # 4. Full 513-bin unvoiced medoid fine-structure pull with smooth VOT taper
            if entry is not None:
                is_uv_eff = uv_sub | (~base.voiced[a:sp])
                g_right = np.where(u_seg <= 0.75, 1.0, np.clip((0.90 - u_seg) / 0.15, 0.0, 1.0))
                g_right = 0.5 * (1.0 - np.cos(math.pi * g_right))
                if s.index == 0:
                    g_left = np.ones_like(u_seg)
                else:
                    g_left = np.where(u_seg >= 0.16, 1.0, np.clip((u_seg - 0.04) / 0.12, 0.0, 1.0))
                    g_left = 0.5 * (1.0 - np.cos(math.pi * g_left))
                gate = np.where(is_uv_eff, g_left * g_right, 0.0)

                if np.any(gate > 1e-4):
                    m_broad = _interp_5x5(m_broad5, u_seg)
                    m_mid = _nearest_5bins(m_mid5, u_seg)
                    m_hi = _nearest_5bins(m_hi5, u_seg)

                    b_cur = base.log_env[idx_seg] + d_env[idx_seg]
                    c_cur = log_env_to_cep(b_cur, n_cep=N_CEP)
                    b_broad = cep_to_log_env(c_cur[:, :_Q_SPLIT], n_bins=N_BINS)
                    b_mid = cep_to_log_env(c_cur, n_bins=N_BINS) - b_broad

                    w_broad = 0.0 if is_sharp_base else self.w_broad_max * blur_ratio
                    w_mid = 0.0 if is_sharp_base else self.w_mid_max * blur_ratio
                    b_zm = b_broad - np.mean(b_broad[:, _M_1_11], axis=1, keepdims=True)
                    m_zm = m_broad - np.mean(m_broad[:, _M_1_11], axis=1, keepdims=True)

                    raw_delta = gate[:, None] * (
                        w_broad * (m_zm - b_zm)
                        + w_mid * (m_mid - b_mid)
                        + self.w_hi * m_hi
                    )
                    # Damped Parseval compensation: avoids both power inflation and -1.5 dB Mel-LSD bias
                    if self.comp_alpha > 0.0:
                        p_before = np.exp(2.0 * np.clip(b_cur, -18.0, 6.0)) @ _BAND_BASIS.T
                        p_after = np.exp(2.0 * np.clip(b_cur + raw_delta, -18.0, 6.0)) @ _BAND_BASIS.T
                        corr_5 = 0.5 * np.log(np.maximum(p_before, 1e-18) / np.maximum(p_after, 1e-18))
                        raw_delta += gate[:, None] * (self.comp_alpha * (corr_5 @ _BAND_BASIS))

                    mask[idx_seg] = True
                    d_env[idx_seg] += raw_delta

        if not np.any(mask):
            return None

        return RuleProposal(
            module_name=self.name,
            mask=mask,
            delta_log_env=d_env,
            delta_bap=d_bap,
            override_voiced=ov,
        )
