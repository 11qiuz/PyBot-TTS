"""Stage 8 Physical Spectrum Synthesizer Timbre Engine (`exp_20_pybot_larynx`).

Provides real-time, zero-neural-weight physical spectral sculpting over 513-bin
`LarynxControl` trajectories using LSF-22 source-filter inspection, formant
selectivity/Q-factor liftering, F1/F2 vowel-openness warping, multi-band vocal-tract
resonance EQ, singer's formant/exciter, vocal-fry subharmonics, and voiced vibrato.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from scipy.fft import rfftfreq

from exp_20_pybot_larynx import HOP, N_BINS, N_FFT, SR
from exp_20_pybot_larynx.larynx.spectral_decomp import (
    decompose_frame_source_filter,
    lsf_to_formants,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env, log_env_to_cep

_NEPER_TO_DB = 20.0 / math.log(10.0)
_DB_TO_NEPER = 1.0 / _NEPER_TO_DB
_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_PREVIEW_HZ = np.geomspace(100.0, 10000.0, 48)

# Smooth Hann window over quefrency [10:30] for formant peak sharpening without DC/tilt drift.
_LIFTER_BELL = np.sin(np.linspace(0.15 * math.pi, 0.85 * math.pi, 20, dtype=np.float64))

TIMBRE_BOUNDS: dict[str, tuple[float, float]] = {
    "chest_db": (-8.0, 8.0),
    "warmth_db": (-8.0, 8.0),
    "nasal_twang_db": (-8.0, 8.0),
    "presence_db": (-8.0, 8.0),
    "air_db": (-8.0, 8.0),
    "formant_q": (-0.60, 0.80),
    "vowel_openness": (-1.0, 1.0),
    "vocal_fry": (0.0, 1.0),
    "vibrato_depth_st": (0.0, 0.60),
    "vibrato_rate_hz": (4.0, 7.5),
    "exciter_drive": (0.0, 1.0),
    "bandpass_vintage": (0.0, 1.0),
}

SYNTH_KNOB_KEYS: tuple[str, ...] = tuple(TIMBRE_BOUNDS.keys())


def _clamp_t(val: Any, key: str, default: float) -> float:
    lo, hi = TIMBRE_BOUNDS[key]
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    if v != v:
        return default
    return max(lo, min(hi, v))


@dataclass
class TimbreSynthParams:
    """Physical synthesizer timbre carving parameters (zero ML weights)."""

    patch_id: str = "neutral"
    chest_db: float = 0.0
    warmth_db: float = 0.0
    nasal_twang_db: float = 0.0
    presence_db: float = 0.0
    air_db: float = 0.0
    formant_q: float = 0.0
    vowel_openness: float = 0.0
    vocal_fry: float = 0.0
    vibrato_depth_st: float = 0.0
    vibrato_rate_hz: float = 5.4
    exciter_drive: float = 0.0
    bandpass_vintage: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TimbreSynthParams:
        """Build clamped TimbreSynthParams, layering patch defaults beneath explicit values."""
        d = data or {}
        patch_id = str(d.get("patch_id") or d.get("timbre_patch") or "neutral").strip()
        if patch_id not in TIMBRE_PATCH_MAP:
            patch_id = "neutral"
        base = dict(TIMBRE_PATCH_MAP[patch_id]["params"])
        merged = {**base, **d}
        return cls(
            patch_id=patch_id,
            chest_db=_clamp_t(merged.get("chest_db", 0.0), "chest_db", 0.0),
            warmth_db=_clamp_t(merged.get("warmth_db", 0.0), "warmth_db", 0.0),
            nasal_twang_db=_clamp_t(merged.get("nasal_twang_db", 0.0), "nasal_twang_db", 0.0),
            presence_db=_clamp_t(merged.get("presence_db", 0.0), "presence_db", 0.0),
            air_db=_clamp_t(merged.get("air_db", 0.0), "air_db", 0.0),
            formant_q=_clamp_t(merged.get("formant_q", 0.0), "formant_q", 0.0),
            vowel_openness=_clamp_t(merged.get("vowel_openness", 0.0), "vowel_openness", 0.0),
            vocal_fry=_clamp_t(merged.get("vocal_fry", 0.0), "vocal_fry", 0.0),
            vibrato_depth_st=_clamp_t(merged.get("vibrato_depth_st", 0.0), "vibrato_depth_st", 0.0),
            vibrato_rate_hz=_clamp_t(merged.get("vibrato_rate_hz", 5.4), "vibrato_rate_hz", 5.4),
            exciter_drive=_clamp_t(merged.get("exciter_drive", 0.0), "exciter_drive", 0.0),
            bandpass_vintage=_clamp_t(merged.get("bandpass_vintage", 0.0), "bandpass_vintage", 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def is_neutral(self) -> bool:
        """Return True when every spectral/glottal synth modifier is at neutral 0.0."""
        return (
            abs(self.chest_db) < 1e-4
            and abs(self.warmth_db) < 1e-4
            and abs(self.nasal_twang_db) < 1e-4
            and abs(self.presence_db) < 1e-4
            and abs(self.air_db) < 1e-4
            and abs(self.formant_q) < 1e-4
            and abs(self.vowel_openness) < 1e-4
            and self.vocal_fry < 1e-4
            and self.vibrato_depth_st < 1e-4
            and self.exciter_drive < 1e-4
            and self.bandpass_vintage < 1e-4
        )


TIMBRE_PATCHES: list[dict[str, Any]] = [
    {
        "id": "neutral",
        "name": "原声直通",
        "desc": "关闭合成器额外染色，保留最纯正 Stage 7 电子声带原声",
        "params": {
            "timbre_patch": "neutral",
            "chest_db": 0.0, "warmth_db": 0.0, "nasal_twang_db": 0.0,
            "presence_db": 0.0, "air_db": 0.0,
            "formant_q": 0.0, "vowel_openness": 0.0,
            "vocal_fry": 0.0, "vibrato_depth_st": 0.0, "vibrato_rate_hz": 5.4,
            "exciter_drive": 0.0, "bandpass_vintage": 0.0,
        },
    },
    {
        "id": "radio_magnetic",
        "name": "电台磁性主播",
        "desc": "丰厚胸腔共鸣 + 清晰歌手共振峰 + 轻微句尾气泡音",
        "params": {
            "timbre_patch": "radio_magnetic",
            "chest_db": 3.8, "warmth_db": 1.8, "nasal_twang_db": -1.2,
            "presence_db": 3.0, "air_db": 1.2,
            "formant_q": 0.18, "vowel_openness": -0.10,
            "vocal_fry": 0.22, "vibrato_depth_st": 0.04, "vibrato_rate_hz": 5.2,
            "exciter_drive": 0.28, "bandpass_vintage": 0.0,
        },
    },
    {
        "id": "anime_sweet",
        "name": "清甜二次元",
        "desc": "开阔微笑唇形(F1/F2微上移) + 清亮高频空气感与高Q值聚焦",
        "params": {
            "timbre_patch": "anime_sweet",
            "chest_db": -1.8, "warmth_db": 0.4, "nasal_twang_db": 2.2,
            "presence_db": 3.4, "air_db": 3.8,
            "formant_q": 0.28, "vowel_openness": 0.48,
            "vocal_fry": 0.0, "vibrato_depth_st": 0.12, "vibrato_rate_hz": 5.8,
            "exciter_drive": 0.35, "bandpass_vintage": 0.0,
        },
    },
    {
        "id": "cinematic_epic",
        "name": "电影史诗旁白",
        "desc": "深沉厚重胸声 + 高声门紧实激励 + 戏剧长音微颤",
        "params": {
            "timbre_patch": "cinematic_epic",
            "chest_db": 4.8, "warmth_db": 2.2, "nasal_twang_db": -2.0,
            "presence_db": 3.6, "air_db": 0.8,
            "formant_q": 0.32, "vowel_openness": -0.28,
            "vocal_fry": 0.30, "vibrato_depth_st": 0.16, "vibrato_rate_hz": 5.0,
            "exciter_drive": 0.45, "bandpass_vintage": 0.0,
        },
    },
    {
        "id": "asmr_intimate",
        "name": "近场 ASMR",
        "desc": "贴耳低频暖度 + 柔和气声高空气感 + 轻气泡微擦音质感",
        "params": {
            "timbre_patch": "asmr_intimate",
            "chest_db": 3.2, "warmth_db": 1.6, "nasal_twang_db": -2.5,
            "presence_db": -1.4, "air_db": 4.8,
            "formant_q": -0.18, "vowel_openness": -0.12,
            "vocal_fry": 0.42, "vibrato_depth_st": 0.0, "vibrato_rate_hz": 5.2,
            "exciter_drive": 0.0, "bandpass_vintage": 0.0,
        },
    },
    {
        "id": "cyber_synth",
        "name": "赛博合成器",
        "desc": "极窄高频共振峰Q值 + 明亮鼻腔金属共鸣 + 紧凑电子振音",
        "params": {
            "timbre_patch": "cyber_synth",
            "chest_db": -1.2, "warmth_db": -1.5, "nasal_twang_db": 4.5,
            "presence_db": 4.8, "air_db": 3.2,
            "formant_q": 0.62, "vowel_openness": 0.25,
            "vocal_fry": 0.08, "vibrato_depth_st": 0.28, "vibrato_rate_hz": 6.4,
            "exciter_drive": 0.72, "bandpass_vintage": 0.0,
        },
    },
    {
        "id": "vintage_phone",
        "name": "复古留声/对讲机",
        "desc": "物理带通滤波截断高低频 + 中频鼻音穿透力",
        "params": {
            "timbre_patch": "vintage_phone",
            "chest_db": -5.5, "warmth_db": 2.4, "nasal_twang_db": 4.6,
            "presence_db": 1.5, "air_db": -6.5,
            "formant_q": 0.20, "vowel_openness": 0.0,
            "vocal_fry": 0.12, "vibrato_depth_st": 0.06, "vibrato_rate_hz": 5.5,
            "exciter_drive": 0.35, "bandpass_vintage": 0.92,
        },
    },
]

TIMBRE_PATCH_MAP: dict[str, dict[str, Any]] = {p["id"]: p for p in TIMBRE_PATCHES}


def compute_synth_eq_curve_db(params: TimbreSynthParams, freqs_hz: np.ndarray = _FREQS) -> np.ndarray:
    """Compute the 5-band vocal-tract + exciter + vintage bandpass EQ curve in dB."""
    f = np.asarray(freqs_hz, dtype=np.float64)
    eq = np.zeros_like(f)

    # 1. Chest resonance bell (~240 Hz)
    if abs(params.chest_db) > 1e-4:
        eq += float(params.chest_db) / (1.0 + ((f - 240.0) / 140.0) ** 2)
    # 2. Warmth / throat body bell (~700 Hz)
    if abs(params.warmth_db) > 1e-4:
        eq += float(params.warmth_db) / (1.0 + ((f - 700.0) / 280.0) ** 2)
    # 3. Nasal / twang focus (~1650 Hz bell + soft 950 Hz dip when bright twang is positive)
    if abs(params.nasal_twang_db) > 1e-4:
        tw = float(params.nasal_twang_db)
        eq += tw / (1.0 + ((f - 1650.0) / 420.0) ** 2)
        if tw > 0.5:
            eq -= (0.28 * tw) / (1.0 + ((f - 950.0) / 210.0) ** 2)
    # 4. Singer's formant presence (~3300 Hz)
    if abs(params.presence_db) > 1e-4:
        eq += float(params.presence_db) / (1.0 + ((f - 3300.0) / 720.0) ** 2)
    # 5. High-frequency air shelf (> 5.5 kHz)
    if abs(params.air_db) > 1e-4:
        air_w = 0.5 * (1.0 + np.tanh((f - 6200.0) / 1600.0))
        eq += float(params.air_db) * air_w
    # 6. Exciter upper-harmonic sheen (2.8..5.5 kHz)
    if params.exciter_drive > 1e-4:
        eq += (2.4 * float(params.exciter_drive)) / (1.0 + ((f - 3900.0) / 1350.0) ** 2)
    # 7. Vintage telephone / gramophone physical bandpass
    if params.bandpass_vintage > 1e-4:
        bp = float(params.bandpass_vintage)
        lo_cut = 0.5 * (1.0 - np.tanh((f - 360.0) / 120.0))
        hi_cut = 0.5 * (1.0 + np.tanh((f - 3500.0) / 650.0))
        mid_hump = 2.6 / (1.0 + ((f - 1350.0) / 550.0) ** 2)
        eq += bp * (-16.0 * lo_cut - 22.0 * hi_cut + mid_hump)

    return np.clip(eq, -26.0, 12.0)


def _apply_vowel_openness_warp(log_env_sub: np.ndarray, openness: float) -> np.ndarray:
    """Warp lower-to-mid formant region (200..3000 Hz) cleanly while pinning Nyquist/DC."""
    if log_env_sub.shape[0] == 0 or abs(openness) < 1e-4:
        return log_env_sub
    op = float(np.clip(openness, -1.0, 1.0))
    # Positive openness shifts F1/F2 upward (bright/smiling open mouth); negative shifts down (darker/closed).
    f1_scale = 1.0 + 0.14 * op
    f2_scale = 1.0 + 0.07 * op
    src_f = _FREQS.copy()
    # Smooth anchor piecewise map: (0, 800*f1_scale, 2400*f2_scale, 4200, nyq) -> (0, 800, 2400, 4200, nyq)
    # Wait: to move output formant UP, src_f must be LOWER than _FREQS!
    src_f = np.interp(
        _FREQS,
        (0.0, 800.0 * f1_scale, 2400.0 * f2_scale, 4400.0, SR * 0.5),
        (0.0, 800.0, 2400.0, 4400.0, SR * 0.5),
    )
    bin_pos = np.clip(src_f * (N_FFT / float(SR)), 0.0, float(N_BINS - 1))
    i0 = np.floor(bin_pos).astype(np.intp)
    i1 = np.minimum(N_BINS - 1, i0 + 1)
    alpha = (bin_pos - i0)[None, :]
    return (1.0 - alpha) * log_env_sub[:, i0] + alpha * log_env_sub[:, i1]


def apply_timbre_synth(
    ctrl: LarynxControl,
    params: TimbreSynthParams,
    *,
    seed: int = 123,
) -> tuple[LarynxControl, dict[str, Any]]:
    """Apply physical synthesizer EQ, formant Q, vowel openness, vibrato, exciter & fry."""
    out = ctrl.copy()
    n_fr = len(out.f0_hz)
    if n_fr == 0 or params.is_neutral():
        return out, {"patch_id": params.patch_id, "subharm_strength": 0.0}

    v_mask = out.voiced & (out.f0_hz > 40.0)
    u_mask = ~v_mask

    # 1. Frequency-selective F1/F2 vowel openness warp on voiced frames
    if abs(params.vowel_openness) > 1e-4 and np.any(v_mask):
        out.log_env[v_mask] = _apply_vowel_openness_warp(out.log_env[v_mask], params.vowel_openness)

    # 2. Formant Q cepstral liftering on cep[10:30] (strictly preserves cep[0] DC level)
    if abs(params.formant_q) > 1e-4 and np.any(v_mask):
        q = float(np.clip(params.formant_q, -0.60, 0.80))
        cep_v = log_env_to_cep(out.log_env[v_mask], n_cep=32)
        d_cep = np.zeros_like(cep_v)
        d_cep[:, 10:30] = q * _LIFTER_BELL[None, :] * cep_v[:, 10:30]
        out.log_env[v_mask] = np.clip(
            out.log_env[v_mask] + cep_to_log_env(d_cep, n_bins=N_BINS),
            -22.0,
            10.0,
        )

    # 3. 5-band Vocal-Tract EQ + Exciter + Vintage Bandpass with voiced level guard
    eq_db = compute_synth_eq_curve_db(params, _FREQS)
    if np.any(np.abs(eq_db) > 1e-4):
        eq_nep_v = eq_db * _DB_TO_NEPER
        # Remove net shift in air >6.5 kHz partially on unvoiced frames to avoid harsh hiss
        air_only = (0.5 * (1.0 + np.tanh((_FREQS - 6200.0) / 1600.0))) * float(params.air_db)
        eq_nep_uv = (eq_db - 0.55 * air_only) * _DB_TO_NEPER

        if np.any(v_mask):
            core_band = (_FREQS >= 200.0) & (_FREQS <= 4200.0)
            pre_mean = float(np.mean(out.log_env[v_mask][:, core_band]))
            out.log_env[v_mask] += eq_nep_v[None, :]
            post_mean = float(np.mean(out.log_env[v_mask][:, core_band]))
            drift = post_mean - pre_mean
            # Preserve overall intelligibility loudness within +-2.2 dB unless vintage phone attenuates intentionally
            max_drift = (4.0 if params.bandpass_vintage > 0.3 else 2.2) * _DB_TO_NEPER
            if abs(drift) > max_drift:
                comp = drift - math.copysign(max_drift, drift)
                out.log_env[v_mask] -= comp
                if np.any(u_mask):
                    out.log_env[u_mask] -= 0.5 * comp
        if np.any(u_mask):
            out.log_env[u_mask] += eq_nep_uv[None, :]
        out.log_env = np.clip(out.log_env, -22.0, 10.0)

    # 4. Exciter drive & vocal fry glottal coupling
    if out.rd is None:
        out.rd = np.ones(n_fr, dtype=np.float64)
    if params.exciter_drive > 1e-4:
        drv = float(params.exciter_drive)
        out.rd = np.clip(out.rd - 0.24 * drv, 0.50, 2.20)
        if np.any(v_mask):
            out.bap[v_mask] = np.clip(out.bap[v_mask] - 0.055 * drv, 0.005, 0.995)

    subharm_strength = float(np.clip(0.28 * float(params.vocal_fry), 0.0, 0.28))
    if params.vocal_fry > 1e-4 and np.any(v_mask):
        fry = float(params.vocal_fry)
        out.rd[v_mask] = np.clip(out.rd[v_mask] + 0.18 * fry, 0.50, 2.20)

    # 5. Delayed-onset natural sinusoidal F0 vibrato on sustained voiced syllables (>= 120 ms)
    if params.vibrato_depth_st > 1e-4 and np.any(v_mask):
        depth = float(params.vibrato_depth_st)
        rate = float(params.vibrato_rate_hz)
        rng = np.random.default_rng(int(seed) + 4242)
        phi0 = float(rng.uniform(0.0, math.tau))
        idx = 0
        while idx < n_fr:
            if not v_mask[idx]:
                idx += 1
                continue
            j = idx
            while j < n_fr and v_mask[j]:
                j += 1
            seg_len = j - idx
            if seg_len >= 24:
                t_loc = np.arange(seg_len, dtype=np.float64) * (HOP / float(SR))
                ramp_on = np.clip((np.arange(seg_len, dtype=np.float64) - 6.0) / 14.0, 0.0, 1.0)
                ramp_off = np.clip((seg_len - 1.0 - np.arange(seg_len, dtype=np.float64)) / 8.0, 0.0, 1.0)
                env_v = 0.5 * (1.0 - np.cos(math.pi * np.minimum(ramp_on, ramp_off)))
                vib_st = depth * env_v * np.sin(math.tau * rate * t_loc + phi0 + idx * 0.13)
                out.f0_hz[idx:j] = np.clip(out.f0_hz[idx:j] * (2.0 ** (vib_st / 12.0)), 55.0, 520.0)
            idx = j

    return out, {
        "patch_id": params.patch_id,
        "subharm_strength": round(subharm_strength, 4),
    }


def extract_spectral_profile_preview(
    ctrl: LarynxControl,
    *,
    max_frames: int = 16,
) -> dict[str, Any]:
    """Extract LSF-22 physical formants, tilt, band energy, and 48-pt preview curve."""
    v_idx = np.where(ctrl.voiced & (ctrl.f0_hz > 40.0))[0]
    if v_idx.size == 0:
        return {
            "formants_hz": [],
            "bandwidths_hz": [],
            "tilt_db_oct": 0.0,
            "band_levels_db": {"chest": 0.0, "warmth": 0.0, "nasal": 0.0, "presence": 0.0, "air": 0.0},
            "curve_hz": [round(float(x), 1) for x in _PREVIEW_HZ],
            "curve_db": [0.0 for _ in range(len(_PREVIEW_HZ))],
        }

    step = max(1, v_idx.size // max(1, int(max_frames)))
    sel = v_idx[::step][:max_frames]

    tilts: list[float] = []
    f_lists: list[list[float]] = [[], [], [], []]
    bw_lists: list[list[float]] = [[], [], [], []]
    for t_i in sel:
        sf = decompose_frame_source_filter(ctrl.log_env[t_i], order=22, sr=SR, n_fft=N_FFT)
        tilts.append(float(sf.tilt_db_oct))
        hz, bw = lsf_to_formants(sf.lsf, sr=SR, f_min=180.0, f_max=5200.0)
        for fi in range(min(4, len(hz))):
            f_lists[fi].append(float(hz[fi]))
            bw_lists[fi].append(float(bw[fi]))

    mean_env_db = np.mean(ctrl.log_env[v_idx], axis=0) * _NEPER_TO_DB
    ref_band = (_FREQS >= 200.0) & (_FREQS <= 3500.0)
    ref_db = float(np.mean(mean_env_db[ref_band])) if np.any(ref_band) else float(np.mean(mean_env_db))
    rel_db = mean_env_db - ref_db

    def _band_db(lo: float, hi: float) -> float:
        m = (_FREQS >= lo) & (_FREQS <= hi)
        return round(float(np.mean(rel_db[m])), 2) if np.any(m) else 0.0

    curve_db = np.interp(_PREVIEW_HZ, _FREQS, rel_db)
    formants = [round(float(np.median(lst)), 1) for lst in f_lists if lst]
    bws = [round(float(np.median(lst)), 1) for lst in bw_lists if lst]
    return {
        "formants_hz": formants,
        "bandwidths_hz": bws,
        "tilt_db_oct": round(float(np.median(tilts)), 2) if tilts else 0.0,
        "band_levels_db": {
            "chest": _band_db(120.0, 380.0),
            "warmth": _band_db(450.0, 1000.0),
            "nasal": _band_db(1200.0, 2200.0),
            "presence": _band_db(2500.0, 4500.0),
            "air": _band_db(5500.0, 9500.0),
        },
        "curve_hz": [round(float(x), 1) for x in _PREVIEW_HZ],
        "curve_db": [round(float(x), 2) for x in curve_db],
    }
