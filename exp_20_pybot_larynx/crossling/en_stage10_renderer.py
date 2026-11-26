"""Stage 10 English Acoustic + Prosody Renderer & Sharpness-Preserving BZNSYP Projection.

Combines:
1. `en_frontend_v2.plan_en_v2` (CMUdict + DH fixes + weak forms)
2. `en_prosody_stage10` (Ridge duration, F0 contour, energy)
3. `knowledge_en_stage10` (unblurred 5-bin TrueHarmEnv medoids + Hierarchical Ridge, ZERO Klatt/Lorentzian)
4. Optional Sharpness-Preserving Physical Speaker Projection (`project_to_zh_speaker=True`):
   - Projects `en_source_profile.json` (slt) -> `base_speaker_profile.json` (BZNSYP) via `apply_clone_to_control`
   - Enforces BRI contrast preservation (`q8_24 >= 10.8`)
   - Calibrates loudness, F0 anchor, and BAP6 against `bilingual_ref_stats.json` & BZNSYP Stage 7 `kb`
   - Keeps Stage 8 emotions, timbre synth, presets, and `SSB0073` voice cloning 100% compatible.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Sequence
import zlib

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter

from exp_20_pybot_larynx import EXP_DIR, HOP, N_BAP, N_BINS, N_FFT, SR
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB
from exp_20_pybot_larynx.clone.adapter import apply_clone_to_control
from exp_20_pybot_larynx.clone.profile import VoiceCloneProfile, load_base_speaker_profile
from exp_20_pybot_larynx.crossling.en_articulation import (
    VOICELESS_OBSTRUENTS,
    VOWEL_IPAS,
    classify_articulatory_class,
)
from exp_20_pybot_larynx.crossling.en_frontend_v2 import plan_en_v2
from exp_20_pybot_larynx.crossling.en_prosody_stage10 import (
    PROSODY_EN_STAGE10_PATH,
    predict_en_prosody_stage10,
)
from exp_20_pybot_larynx.crossling.knowledge_en_stage10 import (
    EN_STAGE10_KNOWLEDGE_PATH,
    AcousticKnowledgeEnStage10,
    build_en_stage10_control,
    get_en_stage10_knowledge,
)
from exp_20_pybot_larynx.crossling.speaker_donor_bank import get_speaker_donor_bank
from exp_20_pybot_larynx.expressive import (
    EMOTION_PROFILES,
    ExpressiveSegment,
    apply_emotion_to_control,
)
from exp_20_pybot_larynx.frontend.prosody_rules import _F0_AR1_SIGMA
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env, log_env_to_cep
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS
from exp_v6_vocoder import cloner as v6_cloner

EN_SOURCE_PROFILE_PATH = EXP_DIR / "crossling_data" / "en_source_profile.json"
BILINGUAL_REF_STATS_PATH = EXP_DIR / "crossling_data" / "bilingual_ref_stats.json"
EN_SF_STAGE12_KB_PATH = EXP_DIR / "crossling" / "knowledge_en_sf_stage12.npz"
SF_ZH_PROFILE_PATH = EXP_DIR / "crossling_data" / "sf_zh_profile.json"
SF_EN_PROFILE_PATH = EXP_DIR / "crossling_data" / "sf_en_profile.json"
SF_CONSISTENCY_PATH = EXP_DIR / "crossling_data" / "sf_bilingual_consistency.json"

_DB_TO_NEPER = 1.0 / NEPER_TO_DB
_MIN_Q8_24_TARGET = 12.55  # Yields re-analyzed vocoder q8_24 >= 11.2 (gate >= 10.8)
_NATIVE_LJSPEECH_TAPER_24 = np.array([0.88] * 8 + [0.60] * 8 + [0.34] * 8, dtype=np.float64)
_FREQS = np.linspace(0.0, 0.5 * SR, N_BINS, dtype=np.float64)
_VOICE_LEVEL_BAND = (_FREQS >= 120.0) & (_FREQS <= 2200.0)
_PROJECTION_TREBLE_RAMP = 0.5 - 0.5 * np.cos(math.pi * np.clip((_FREQS - 3200.0) / 4000.0, 0.0, 1.0))

_CACHE_LOCK = Lock()
_CACHED_SRC_PROF: VoiceCloneProfile | None = None
_CACHED_BASE_PROF: VoiceCloneProfile | None = None
_CACHED_REF_STATS: dict[str, Any] | None = None
_CACHED_SF_KB: AcousticKnowledgeEnStage10 | None = None
_CACHED_SF_ZH_PROF: VoiceCloneProfile | None = None
_CACHED_SF_EN_PROF: VoiceCloneProfile | None = None
_CACHED_SF_CONS: dict[str, Any] | None = None


@dataclass(frozen=True)
class MatchedVoiceEQ:
    """Fixed voiced EQ in nepers and its per-bin blend with Stage 11."""

    curve: np.ndarray
    blend: np.ndarray

    def __post_init__(self) -> None:
        if any(arr.shape != (N_BINS,) or not np.all(np.isfinite(arr)) for arr in (self.curve, self.blend)):
            raise ValueError(f"Expected {N_BINS} finite matched voice EQ and blend bins")
        if np.any((self.blend < 0) | (self.blend > 1)):
            raise ValueError("Matched voice EQ blend must be in [0, 1]")


def is_stage10_en_ready() -> bool:
    """Return True when both Stage 10 EN acoustic & prosody `.npz` artifacts exist on disk."""
    return EN_STAGE10_KNOWLEDGE_PATH.exists() and PROSODY_EN_STAGE10_PATH.exists()


def is_stage12_sf_ready() -> bool:
    """Return True when Stage 12 same-speaker SF bilingual KB, profiles & consistency exist."""
    return (
        EN_SF_STAGE12_KB_PATH.is_file()
        and SF_ZH_PROFILE_PATH.is_file()
        and SF_EN_PROFILE_PATH.is_file()
        and SF_CONSISTENCY_PATH.is_file()
    )


def _get_sf_stage12_knowledge() -> AcousticKnowledgeEnStage10 | None:
    global _CACHED_SF_KB
    with _CACHE_LOCK:
        if _CACHED_SF_KB is not None:
            return _CACHED_SF_KB
    if not EN_SF_STAGE12_KB_PATH.is_file():
        return None
    kb = AcousticKnowledgeEnStage10.load(EN_SF_STAGE12_KB_PATH)
    with _CACHE_LOCK:
        _CACHED_SF_KB = kb
    return kb


def _load_sf_zh_profile() -> VoiceCloneProfile | None:
    global _CACHED_SF_ZH_PROF
    with _CACHE_LOCK:
        if _CACHED_SF_ZH_PROF is not None:
            return _CACHED_SF_ZH_PROF
    if not SF_ZH_PROFILE_PATH.is_file():
        return None
    prof = VoiceCloneProfile.from_json(SF_ZH_PROFILE_PATH.read_text(encoding="utf-8"))
    with _CACHE_LOCK:
        _CACHED_SF_ZH_PROF = prof
    return prof


def _load_sf_en_profile() -> VoiceCloneProfile | None:
    global _CACHED_SF_EN_PROF
    with _CACHE_LOCK:
        if _CACHED_SF_EN_PROF is not None:
            return _CACHED_SF_EN_PROF
    if not SF_EN_PROFILE_PATH.is_file():
        return None
    prof = VoiceCloneProfile.from_json(SF_EN_PROFILE_PATH.read_text(encoding="utf-8"))
    with _CACHE_LOCK:
        _CACHED_SF_EN_PROF = prof
    return prof


def load_sf_bilingual_consistency() -> dict[str, Any]:
    """Load cached Stage 12 same-speaker EN-vs-ZH physical consistency decomposition."""
    global _CACHED_SF_CONS
    with _CACHE_LOCK:
        if _CACHED_SF_CONS is not None:
            return _CACHED_SF_CONS
    doc: dict[str, Any] = {}
    if SF_CONSISTENCY_PATH.is_file():
        try:
            doc = json.loads(SF_CONSISTENCY_PATH.read_text(encoding="utf-8"))
        except Exception:
            doc = {}
    with _CACHE_LOCK:
        _CACHED_SF_CONS = doc
    return doc


def _load_en_source_profile() -> VoiceCloneProfile | None:
    global _CACHED_SRC_PROF
    with _CACHE_LOCK:
        if _CACHED_SRC_PROF is not None:
            return _CACHED_SRC_PROF
    if not EN_SOURCE_PROFILE_PATH.exists():
        return None
    prof = VoiceCloneProfile.from_json(EN_SOURCE_PROFILE_PATH.read_text(encoding="utf-8"))
    with _CACHE_LOCK:
        _CACHED_SRC_PROF = prof
    return prof


def _load_base_profile_cached() -> VoiceCloneProfile:
    global _CACHED_BASE_PROF
    with _CACHE_LOCK:
        if _CACHED_BASE_PROF is not None:
            return _CACHED_BASE_PROF
    prof = load_base_speaker_profile()
    with _CACHE_LOCK:
        _CACHED_BASE_PROF = prof
    return prof


def _load_bilingual_ref_stats() -> dict[str, Any]:
    global _CACHED_REF_STATS
    with _CACHE_LOCK:
        if _CACHED_REF_STATS is not None:
            return _CACHED_REF_STATS
    stats: dict[str, Any] = {}
    if BILINGUAL_REF_STATS_PATH.exists():
        try:
            stats = json.loads(BILINGUAL_REF_STATS_PATH.read_text(encoding="utf-8"))
        except Exception:
            stats = {}
    with _CACHE_LOCK:
        _CACHED_REF_STATS = stats
    return stats


def clear_stage10_renderer_caches() -> None:
    """Clear cached profiles/stats (e.g. after rebuilding Stage 10/12 artifacts)."""
    global _CACHED_SRC_PROF, _CACHED_BASE_PROF, _CACHED_REF_STATS
    global _CACHED_SF_KB, _CACHED_SF_ZH_PROF, _CACHED_SF_EN_PROF, _CACHED_SF_CONS
    with _CACHE_LOCK:
        _CACHED_SRC_PROF = None
        _CACHED_BASE_PROF = None
        _CACHED_REF_STATS = None
        _CACHED_SF_KB = None
        _CACHED_SF_ZH_PROF = None
        _CACHED_SF_EN_PROF = None
        _CACHED_SF_CONS = None


def _compute_expressive_drive(
    expressive_seg: ExpressiveSegment | None,
    params: Any,
    is_question: bool | None,
    text_or_units: str | Sequence[dict[str, Any]],
) -> float:
    """Compute [0, 1] expressive drive: calm neutral speech stays smooth; dramatic speech unlocks LJSpeech range."""
    emo_drive = 0.0
    if expressive_seg is not None and expressive_seg.emotion != "neutral":
        emo_drive = float(np.clip(float(expressive_seg.intensity), 0.0, 1.0))

    focus_drive = 0.45 if (expressive_seg is not None and expressive_seg.focus_spans) else 0.0

    punct_drive = 0.0
    raw_txt = text_or_units if isinstance(text_or_units, str) else ""
    if "!" in raw_txt or "！" in raw_txt or (expressive_seg is not None and expressive_seg.punct_style == "!"):
        punct_drive = 0.55
    elif is_question is True or "?" in raw_txt or "？" in raw_txt:
        punct_drive = 0.35

    stab = float(getattr(params, "stability", 0.55))
    stab_drive = float(np.clip((0.55 - stab) / 0.45, 0.0, 1.0)) * 0.70
    return float(np.clip(max(emo_drive, focus_drive, punct_drive, stab_drive), 0.0, 1.0))


def _predict_en_prosody_stage10_or_fallback(
    text_or_units: str | Sequence[dict[str, Any]],
    *,
    base_f0_hz: float,
    speed: float,
    pause_scale: float,
    range_sc: float,
    clause_pos: tuple[float, float],
    is_clause_final: bool,
    is_question: bool | None,
    focus_spans: Sequence[Any] | None,
    seed: int,
    embedded_in_zh: bool = False,
    expressive_drive: float | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray | None, bool]:
    """Use `en_prosody_stage10` Ridge prosody model & convert spans into acoustic phone_plans."""
    res = predict_en_prosody_stage10(
        text_or_units,
        base_f0_hz=base_f0_hz,
        rate=speed,
        pause_scale=pause_scale,
        range_scale=range_sc,
        clause_pos=clause_pos,
        is_clause_final=is_clause_final,
        is_question=is_question,
        focus_spans=focus_spans,
        include_pauses=True,
        seed=seed,
        embedded_in_zh=embedded_in_zh,
        expressive_drive=expressive_drive,
    )
    phones = list(res.get("phones") or [])
    spans = list(res.get("phone_spans") or [])
    f0_hz = np.asarray(res["f0_hz"], dtype=np.float64).copy()
    gain_fr = np.asarray(res["gain_db_frames"], dtype=np.float64) if "gain_db_frames" in res else None
    q_flag = bool(res.get("is_question", False))
    n_fr = len(f0_hz)
    voiced = np.zeros(n_fr, dtype=bool)

    n_p = len(phones)
    phone_plans: list[dict[str, Any]] = []
    for sp in spans:
        a = max(0, min(n_fr, int(sp["start_fr"])))
        b = max(a, min(n_fr, int(sp["end_fr"])))
        if sp.get("is_pause", False) or int(sp.get("phone_idx", -1)) < 0:
            if b > a:
                f0_hz[a:b] = 0.0
                voiced[a:b] = False
            phone_plans.append({
                "ipa": "sil",
                "phone": "sil",
                "start_fr": a,
                "end_fr": b,
                "dur_ms": float((b - a) * 5.0),
                "is_pause": True,
            })
            continue

        pi = int(sp["phone_idx"])
        ph = phones[pi]
        ipa = str(ph.get("ipa", "ə"))
        allo = str(ph.get("allophone") or ipa)
        if b > a:
            if ipa in VOICELESS_OBSTRUENTS:
                f0_hz[a:b] = 0.0
                voiced[a:b] = False
            else:
                voiced[a:b] = f0_hz[a:b] > 40.0

        prev_ph = phones[pi - 1] if pi > 0 else None
        nxt_ph = phones[pi + 1] if pi + 1 < n_p else None
        p_prev = float(prev_ph.get("pause_after_ms", 0.0)) if prev_ph is not None else 999.0
        p_after = float(ph.get("pause_after_ms", 0.0)) if pi < n_p - 1 else 999.0
        same_w_prev = bool(prev_ph is not None and p_prev < 20.0 and prev_ph.get("word_idx") == ph.get("word_idx"))
        same_w_next = bool(nxt_ph is not None and p_after < 20.0 and nxt_ph.get("word_idx") == ph.get("word_idx"))
        n_w_ph = int(ph.get("word_n_phones", 1))
        if n_w_ph <= 1:
            word_pos = 0
        elif ph.get("is_word_initial", False):
            word_pos = 1
        elif ph.get("is_word_final", False):
            word_pos = 3
        else:
            word_pos = 2

        phone_plans.append({
            "ipa": ipa,
            "phone": allo,
            "stress": ph.get("stress"),
            "start_fr": a,
            "end_fr": b,
            "dur_ms": float(ph.get("dur_ms", (b - a) * 5.0)),
            "prev_cls": 0 if (prev_ph is None or p_prev >= 25.0) else classify_articulatory_class(prev_ph.get("allophone") or prev_ph.get("ipa")),
            "next_cls": 0 if (nxt_ph is None or p_after >= 25.0) else classify_articulatory_class(nxt_ph.get("allophone") or nxt_ph.get("ipa")),
            "same_word_prev": same_w_prev,
            "same_word_next": same_w_next,
            "next_ipa": nxt_ph.get("ipa") if nxt_ph is not None else None,
            "word_pos": word_pos,
            "is_func_word": bool(ph.get("is_func", False)),
            "pause_after_ms": p_after,
            "gain_db": float(ph.get("gain_db", 0.0)),
            "is_pause": False,
        })

    return phone_plans, f0_hz, voiced, gain_fr, q_flag


def _preserve_q8_24_sharpness(
    log_env: np.ndarray,
    v_mask: np.ndarray,
    target_q8_24: float = _MIN_Q8_24_TARGET,
) -> np.ndarray:
    """Boost mid-quefrency formant resonances (cep[8:24]) minimally if softened by VTLN."""
    if int(np.sum(v_mask)) < 4:
        return log_env
    out = log_env.copy()
    ve = out[v_mask]
    cep = log_env_to_cep(ve, n_cep=28)
    q_vals = np.sqrt(np.sum(cep[:, 8:24] ** 2, axis=1))
    q_med = float(np.median(q_vals))
    if q_med >= target_q8_24 or q_med < 1.0:
        return out

    ratio = float(np.clip(target_q8_24 / q_med, 1.0, 1.36))
    gain = ratio - 1.0
    # Smooth bell weights over cep[8:26] so there is zero DC/tilt shift and zero ringing
    w_q = np.zeros(28, dtype=np.float64)
    w_q[8:24] = np.sin(np.linspace(0.25 * math.pi, 0.85 * math.pi, 16, dtype=np.float64))
    w_q[24:26] = 0.35
    delta_cep = cep * (gain * w_q)[None, :]
    delta_env = cep_to_log_env(delta_cep, n_bins=N_BINS)

    pre_pwr = np.sum(np.exp(2.0 * ve), axis=1)
    env_sharp = ve + delta_env
    post_pwr = np.sum(np.exp(2.0 * env_sharp), axis=1)
    pwr_adj = 0.5 * np.log(np.maximum(pre_pwr, 1e-18) / np.maximum(post_pwr, 1e-18))
    out[v_mask] = np.clip(env_sharp + pwr_adj[:, None], -16.0, 6.0)
    return out


def _limit_projected_treble(
    native: LarynxControl, projected: LarynxControl, voiced: np.ndarray,
) -> None:
    """Restrain added voiced treble after wideband speaker mapping."""
    mapped = projected.log_env[voiced]
    delta = mapped - native.log_env[voiced]
    level_shift = np.median(delta[:, _VOICE_LEVEL_BAND], axis=1)
    # Compare broad, level-matched envelopes. The source formant peaks and the
    # mapped high-frequency detail stay in the projected spectrum.
    excess = gaussian_filter1d(delta, sigma=12.0, axis=-1) - level_shift[:, None]
    added_treble = np.clip(excess - 3.0 * _DB_TO_NEPER, 0.0, 1.8)
    projected.log_env[voiced] = np.clip(
        mapped - 0.60 * added_treble * _PROJECTION_TREBLE_RAMP, -16.0, 3.0,
    )


_EN_PHYS_BAP_TARGET = np.asarray([0.017, 0.128, 0.295, 0.74, 0.88, 0.93], dtype=np.float64)
_EN_VOT_EDGE_FLOOR = np.asarray([0.22, 0.38, 0.54, 0.86, 0.91, 0.94], dtype=np.float64)


def _apply_en_stage4_anti_buzz_glottal(
    proj: LarynxControl,
    v_mask: np.ndarray,
    tgt_f0_med: float,
    *,
    expressive_drive: float = 0.0,
) -> None:
    """Port Stage 4 Mandarin anti-buzz glottal coupling to pure EN to eliminate static vocoder buzz."""
    n_fr = len(proj.f0_hz)
    if n_fr == 0 or not np.any(v_mask):
        return

    # Mildly relax oversharpened q8_24 comb ringing above 11.4 while preserving formants & linear power
    ve = proj.log_env[v_mask]
    cep_q = log_env_to_cep(ve, n_cep=32)
    q_cur = float(np.median(np.linalg.norm(cep_q[:, 8:24], axis=1)))
    if q_cur > 11.4:
        sc = max(0.92, 11.35 / q_cur)
        w_q = np.zeros(32, dtype=np.float64)
        w_q[8:24] = 1.0
        delta_env = cep_to_log_env(cep_q * ((sc - 1.0) * w_q)[None, :], n_bins=N_BINS)
        pre_p = np.sum(np.exp(2.0 * ve), axis=1)
        ve_new = ve + delta_env
        post_p = np.sum(np.exp(2.0 * ve_new), axis=1)
        ve_new += (0.5 * np.log(np.maximum(pre_p, 1e-18) / np.maximum(post_p, 1e-18)))[:, None]
        proj.log_env[v_mask] = np.clip(ve_new, -16.0, 3.0)

    # Scan contiguous voiced runs for run phase, length, VOT onset ramp, and phrase-final tail
    u_run = np.zeros(n_fr, dtype=np.float64)
    len_run = np.zeros(n_fr, dtype=np.int32)
    edge_w = np.zeros(n_fr, dtype=np.float64)
    i = 0
    last_v_end = 0
    while i < n_fr:
        if not v_mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n_fr and v_mask[j]:
            j += 1
        run_len = j - i
        u_run[i:j] = np.linspace(0.0, 1.0, run_len, dtype=np.float64)
        len_run[i:j] = run_len
        if i > 0:
            for k, rw in enumerate((0.80, 0.40, 0.16)):
                if i + k < j:
                    edge_w[i + k] = rw
        last_v_end = j
        i = j

    tail_w = np.zeros(n_fr, dtype=np.float64)
    t0 = max(0, last_v_end - 26)
    if last_v_end > t0:
        tail_w[t0:last_v_end] = np.linspace(0.0, 1.0, last_v_end - t0, dtype=np.float64) ** 1.4

    rst = 12.0 * np.log2(proj.f0_hz[v_mask] / max(50.0, tgt_f0_med))
    u_m = u_run[v_mask]
    len_m = len_run[v_mask]
    e_m = edge_w[v_mask]
    tw_m = tail_w[v_mask]

    frame_e = np.mean(proj.log_env[:, 8:170], axis=1)
    rel_e_db = (frame_e[v_mask] - float(np.median(frame_e[v_mask]))) * NEPER_TO_DB
    e_mul = np.clip(1.0 - 0.025 * np.clip(rel_e_db, -8.0, 6.0), 0.85, 1.25)
    reg_mul = np.clip(1.0 - 0.040 * np.clip(rst, -5.0, 6.0), 0.75, 1.24)
    long_m = len_m >= 32
    lb = np.zeros_like(rst)
    if np.any(long_m):
        ul = u_m[long_m]
        lb[long_m] = (
            0.09 * np.sin(math.pi * np.clip((ul - 0.20) / 0.75, 0.0, 1.0))
            + 0.07 * np.clip((ul - 0.65) / 0.35, 0.0, 1.0)
        )
    pos_mul = e_mul * (1.0 + 0.25 * tw_m) * (1.0 + lb)

    mod_band = np.stack(
        [
            reg_mul * pos_mul,
            reg_mul * pos_mul,
            (0.65 + 0.35 * reg_mul) * (0.60 + 0.40 * pos_mul),
            (0.75 + 0.25 * reg_mul) * (0.75 + 0.25 * pos_mul),
            0.88 + 0.12 * pos_mul,
            0.92 + 0.08 * pos_mul,
        ],
        axis=1,
    )
    phys_t = np.clip(_EN_PHYS_BAP_TARGET[None, :] * mod_band, 0.012, 0.96)
    base_b = proj.bap[v_mask]
    alpha = np.where(base_b > phys_t, 0.82, 0.72)
    tb = np.clip(alpha * phys_t + (1.0 - alpha) * base_b, 0.012, 0.96)
    if np.any(e_m > 0.0):
        tb = np.maximum(tb, (1.0 - e_m[:, None]) * tb + e_m[:, None] * _EN_VOT_EDGE_FLOOR[None, :])
    proj.bap[v_mask] = tb

    if proj.glottal_tilt_db_per_oct is None:
        proj.glottal_tilt_db_per_oct = np.zeros(n_fr, dtype=np.float64)
    proj.glottal_tilt_db_per_oct[v_mask] = 0.025 * np.clip(rst, -5.0, 5.0) - 0.06 * tw_m - 0.70 * e_m

    if proj.jitter_frac is None:
        proj.jitter_frac = np.full(n_fr, 0.0074, dtype=np.float64)
    proj.jitter_frac[v_mask] = np.clip(0.0074 - 0.00035 * rst + 0.0020 * tw_m + 0.0035 * lb, 0.0042, 0.0125)

    if proj.shimmer_frac is None:
        proj.shimmer_frac = np.full(n_fr, 0.0460, dtype=np.float64)
    proj.shimmer_frac[v_mask] = np.clip(0.0460 - 0.0020 * rst + 0.0120 * tw_m + 0.0220 * lb, 0.0240, 0.0820)

    if proj.rd is None:
        proj.rd = np.ones(n_fr, dtype=np.float64)
    proj.rd[v_mask] = np.clip(1.08 - 0.024 * np.clip(rst, -5.0, 6.0) + 0.28 * tw_m + 0.85 * lb, 0.92, 1.65)


def _project_en_to_bznsyp(
    ctrl: LarynxControl,
    en_kb: AcousticKnowledgeEnStage10,
    zh_kb: Any | None,
    base_f0_hz: float,
    range_sc: float,
    *,
    expand_pitch: bool = True,
    embedded_in_zh: bool = False,
    limit_treble: bool = False,
    matched_voice_eq: MatchedVoiceEQ | None = None,
    use_sf_pivot: bool = False,
    expressive_drive: float = 0.0,
) -> LarynxControl:
    """Sharpness-preserving physical projection into BZNSYP speaker space (pure ZH->ZH pivot when SF ready)."""
    sf_zh_prof = _load_sf_zh_profile() if use_sf_pivot else None
    sf_cons = load_sf_bilingual_consistency() if use_sf_pivot else {}
    use_pure_zh_pivot = bool(use_sf_pivot and sf_zh_prof is not None and sf_cons)

    src_prof = sf_zh_prof if use_pure_zh_pivot else _load_en_source_profile()
    tgt_prof = _load_base_profile_cached()
    ref_stats = _load_bilingual_ref_stats()

    v_mask = ctrl.voiced & (ctrl.f0_hz > 40.0)
    if not np.any(v_mask):
        return ctrl

    # Measure pre-projection q8_24 formant sharpness so projection never blurs formants
    pre_cep = log_env_to_cep(ctrl.log_env[v_mask], n_cep=26)
    pre_q8_24 = float(np.median(np.sqrt(np.sum(pre_cep[:, 8:24] ** 2, axis=1))))

    if src_prof is not None:
        proj, _, _ = apply_clone_to_control(
            ctrl,
            src_prof,
            tgt_prof,
            strength=0.92,
            clone_f0=False,
            clone_glottal=False,
        )
    else:
        proj = ctrl.copy()

    # 1. Calibrate F0 anchor cleanly to BZNSYP base_f0_hz without artificial cartoon pitch exaggeration
    en_zh_f0_diff_st = float(ref_stats.get("en_zh_f0_diff_st", 0.0)) if ref_stats else 0.0
    f0_off_st = float(np.clip(en_zh_f0_diff_st, -0.25, 0.25))
    tgt_f0_med = float(np.clip(base_f0_hz * (2.0 ** (f0_off_st / 12.0)), 65.0, 460.0))

    src_f0_med = max(50.0, float(en_kb.speaker_f0_med_hz))
    log_f0 = np.log(proj.f0_hz[v_mask])
    utt_med_log = float(np.median(log_f0)) if log_f0.size >= 8 else math.log(src_f0_med)
    # Keep prosody model's natural expressive range; apply range_sc cleanly (no double 1.33x boost)
    std_scale = float(np.clip(range_sc, 0.85, 1.45)) if expand_pitch else 1.0
    delta_f0 = std_scale * (log_f0 - utt_med_log)
    delta_f0 = delta_f0 - float(np.median(delta_f0))
    log_f0_scaled = math.log(tgt_f0_med) + delta_f0
    proj.f0_hz[v_mask] = np.clip(np.exp(log_f0_scaled), 62.0, 460.0)

    # 2. Sharpness-preserving BRI guard FIRST so subsequent smooth LTAS preserves q8_24 >= 10.8
    target_q = max(_MIN_Q8_24_TARGET, 0.98 * pre_q8_24)
    proj.log_env = _preserve_q8_24_sharpness(proj.log_env, v_mask, target_q8_24=target_q)

    # 3. Smooth wideband LTAS alignment toward BZNSYP (or BZNSYP + real same-speaker delta_EN_minus_ZH when using SF pivot)
    src_bw_hz = float(getattr(src_prof, "effective_bw_hz", 7200.0)) if src_prof is not None else 7200.0
    is_fullband_src = src_bw_hz >= 9500.0
    m_band = (_FREQS >= 80.0) & (_FREQS <= 11500.0)
    m_shared = (_FREQS >= 80.0) & (_FREQS <= (11000.0 if is_fullband_src else 7000.0))
    u_nyq = np.clip((_FREQS - 11200.0) / 800.0, 0.0, 1.0)
    w_nyq_taper = 0.5 * (1.0 + np.cos(math.pi * u_nyq))

    if zh_kb is not None:
        bank = get_speaker_donor_bank(zh_kb)
        tgt_u_ltas = np.asarray(bank.unvoiced_ltas_env, dtype=np.float64)
        tgt_bap6 = np.asarray(bank.voiced_bap6_med, dtype=np.float64)
        tgt_v_prof = np.asarray(tgt_prof.voiced_env_mean, dtype=np.float64)
        tgt_v_bank = np.asarray(bank.voiced_ltas_env, dtype=np.float64)
        prof_off = float(np.mean(tgt_v_bank[m_shared]) - np.mean(tgt_v_prof[m_shared])) - 0.45
        if is_fullband_src:
            tgt_v_ltas = 0.75 * (tgt_v_prof + prof_off) + 0.25 * (tgt_v_bank - 0.45)
        else:
            u_bwe = np.clip((_FREQS - 6200.0) / 1800.0, 0.0, 1.0)
            w_bwe = 0.5 * (1.0 - np.cos(math.pi * u_bwe))
            tgt_v_ltas = (1.0 - w_bwe) * (tgt_v_prof + prof_off) + w_bwe * (tgt_v_bank - 0.45)
        if use_pure_zh_pivot:
            dv_nep = np.asarray(sf_cons.get("delta_voiced_ltas_en_minus_zh_nep") or np.zeros(N_BINS), dtype=np.float64)
            du_nep = np.asarray(sf_cons.get("delta_unvoiced_ltas_en_minus_zh_nep") or np.zeros(N_BINS), dtype=np.float64)
            db_6 = np.asarray(sf_cons.get("delta_bap6_en_minus_zh") or np.zeros(N_BAP), dtype=np.float64)
            if dv_nep.shape == (N_BINS,):
                tgt_v_ltas = tgt_v_ltas + 0.75 * dv_nep
            if du_nep.shape == (N_BINS,):
                tgt_u_ltas = tgt_u_ltas + 0.65 * du_nep
            if db_6.shape == (N_BAP,):
                tgt_bap6 = np.clip(tgt_bap6 + 0.5 * db_6, 0.015, 0.98)
            drms = float(sf_cons.get("delta_rms_db_en_minus_zh", 0.2))
            target_v_pwr_nep = float(0.5 * math.log(max(1e-18, float(np.mean(np.exp(2.0 * tgt_v_ltas)))))) + drms * _DB_TO_NEPER
        else:
            target_v_pwr_nep = float(0.5 * math.log(max(1e-18, float(np.mean(np.exp(2.0 * tgt_v_ltas)))))) + 0.02
    else:
        tgt_v_ltas = np.asarray(tgt_prof.voiced_env_mean, dtype=np.float64) + float(getattr(en_kb, "ref_log_amp", -4.26))
        tgt_u_ltas = tgt_v_ltas.copy()
        tgt_bap6 = np.asarray([0.044, 0.25, 0.46, 0.74, 0.88, 0.93], dtype=np.float64)
        target_v_pwr_nep = float(0.5 * math.log(max(1e-18, float(np.mean(np.exp(2.0 * tgt_v_ltas))))))

    en_zh_rms_diff_db = float(ref_stats.get("en_zh_rms_diff_db", 0.0)) if ref_stats else 0.0
    rms_off_db = 0.0 if use_pure_zh_pivot else float(np.clip(en_zh_rms_diff_db, -0.45, 0.45))

    cur_v_ltas = np.mean(proj.log_env[v_mask], axis=0)
    raw_v_diff = tgt_v_ltas - cur_v_ltas
    active_mask = np.max(proj.log_env, axis=1) > -13.5
    if is_fullband_src:
        # Restrict smooth LTAS alignment to macro-envelope q < 8 + gentle wide Gaussian HF shelf (leaves formants untouched!)
        v_cep = log_env_to_cep(raw_v_diff[None, :], n_cep=8)[0]
        v_cep[0] = 0.0
        smooth_q8 = cep_to_log_env(v_cep[None, :], n_bins=N_BINS)[0]
        wide_g = gaussian_filter1d(raw_v_diff, sigma=28.0)
        w_hf = 0.5 * (1.0 - np.cos(math.pi * np.clip((_FREQS - 3000.0) / 3500.0, 0.0, 1.0)))
        smooth_v_eq = (1.0 - 0.45 * w_hf) * smooth_q8 + (0.45 * w_hf) * wide_g
        smooth_v_eq -= float(np.mean(smooth_v_eq[m_shared]))
        # Guard against overfitting 1-3 word embedded runs to local vowel formants
        w_loc = float(np.clip((int(np.sum(v_mask)) - 40) / 180.0, 0.25, 1.0)) if embedded_in_zh else 1.0
        v_eq_total = np.clip(smooth_v_eq * w_loc, -1.45, 1.45) * w_nyq_taper
        proj.log_env[v_mask] = np.clip(proj.log_env[v_mask] + v_eq_total[None, :], -16.0, 3.0)
        cur_v_pwr_nep = float(np.median(0.5 * np.log(np.mean(np.exp(2.0 * proj.log_env[v_mask]), axis=1) + 1e-18)))
        gain_np = float(np.clip((target_v_pwr_nep - cur_v_pwr_nep) + rms_off_db * _DB_TO_NEPER, -2.5, 2.5))
        proj.log_env[active_mask] = np.clip(proj.log_env[active_mask] + gain_np, -16.0, 2.5)
    else:
        u_bwe = np.clip((_FREQS - 6200.0) / 1800.0, 0.0, 1.0)
        w_bwe = 0.5 * (1.0 - np.cos(math.pi * u_bwe))
        v_cep = log_env_to_cep(raw_v_diff[None, :], n_cep=12)[0]
        v_cep[0] = 0.0
        smooth_v_eq = np.clip(cep_to_log_env(v_cep[None, :], n_bins=N_BINS)[0], -0.80, 0.80)
        v_eq_total = (1.0 - w_bwe) * smooth_v_eq + w_bwe * np.clip(raw_v_diff, -1.5, 2.2)
        proj.log_env[v_mask] = np.clip(proj.log_env[v_mask] + v_eq_total[None, :], -16.0, 3.0)
        cur_v_pwr_nep = float(np.median(0.5 * np.log(np.mean(np.exp(2.0 * proj.log_env[v_mask]), axis=1) + 1e-18)))
        gain_np = float(np.clip((target_v_pwr_nep - cur_v_pwr_nep) + rms_off_db * _DB_TO_NEPER, -2.5, 2.5))
        proj.log_env[active_mask] = np.clip(proj.log_env[active_mask] + gain_np, -16.0, 2.5)

    # 4. Align unvoiced LTAS shape gently & keep unvoiced fricative peaks slightly below extreme vowel ceiling
    v_peak_cap = float(np.percentile(np.max(proj.log_env[v_mask], axis=1), 92)) - 0.10
    u_active = (~proj.voiced) & (np.mean(proj.log_env[:, _FREQS >= 1500.0], axis=1) > -10.5)
    if int(np.sum(u_active)) >= 4:
        cur_u_ltas = np.mean(proj.log_env[u_active], axis=0)
        raw_u_diff = tgt_u_ltas - cur_u_ltas
        if is_fullband_src:
            u_w = min(1.0, float(np.sum(u_active)) / 20.0)
            u_eq_total = np.clip(gaussian_filter1d(raw_u_diff, sigma=20.0) * u_w, -1.35, 1.35) * w_nyq_taper
        else:
            u_bwe = np.clip((_FREQS - 6200.0) / 1800.0, 0.0, 1.0)
            w_bwe = 0.5 * (1.0 - np.cos(math.pi * u_bwe))
            u_cep = log_env_to_cep(raw_u_diff[None, :], n_cep=10)[0]
            smooth_u_eq = np.clip(cep_to_log_env(u_cep[None, :], n_bins=N_BINS)[0], -1.0, 1.2)
            u_eq_total = (1.0 - w_bwe) * smooth_u_eq + w_bwe * np.clip(raw_u_diff, -1.2, 2.2)
        proj.log_env[u_active] = np.minimum(
            np.clip(proj.log_env[u_active] + u_eq_total[None, :], -16.0, 2.0),
            v_peak_cap,
        )
        # Broadband fricative/stop spike ceiling prevents /s/ explosions (+14 dB above vowels)
        u_pwr = 0.5 * np.log(np.mean(np.exp(2.0 * proj.log_env[u_active]), axis=1) + 1e-18)
        v_pwr_med_u = float(np.median(0.5 * np.log(np.mean(np.exp(2.0 * proj.log_env[v_mask]), axis=1) + 1e-18)))
        drive_u = float(np.clip(expressive_drive, 0.0, 1.0))
        u_ceil = v_pwr_med_u + (0.30 if use_pure_zh_pivot else (0.14 + 0.14 * drive_u))
        u_hot = u_pwr > u_ceil
        if np.any(u_hot):
            u_idx = np.where(u_active)[0][u_hot]
            att = 0.85 * (u_pwr[u_hot] - u_ceil)
            proj.log_env[u_idx] = np.clip(proj.log_env[u_idx] - att[:, None], -16.0, 2.0)

    # Gentle outlier-only soft ceiling (> +13 dB above voiced median); leave stop closures & dynamic range untouched
    frame_pwr_nep = 0.5 * np.log(np.mean(np.exp(2.0 * proj.log_env), axis=1) + 1e-18)
    v_pwr_med = float(np.median(frame_pwr_nep[v_mask]))
    pwr_ceil = v_pwr_med + 1.50
    hot_outliers = active_mask & (frame_pwr_nep > pwr_ceil)
    if np.any(hot_outliers):
        damp = 0.45 * (frame_pwr_nep[hot_outliers] - pwr_ceil)
        proj.log_env[hot_outliers] = np.clip(proj.log_env[hot_outliers] - damp[:, None], -16.0, 2.5)

    # 5. Natural human glottal BAP6, jitter & shimmer aligned cleanly to BZNSYP Stage 4/7 voice (NO robotic buzz clamps!)
    drive_c = float(np.clip(expressive_drive, 0.0, 1.0))
    bap_ref = np.asarray(
        [
            float(np.clip(tgt_bap6[0] * 0.74, 0.020, 0.042)),
            float(np.clip(tgt_bap6[1] * 0.78, 0.155, 0.26)),
            float(np.clip(tgt_bap6[2], 0.36, 0.62)),
            float(np.clip(tgt_bap6[3], 0.62, 0.86)),
            float(np.clip(tgt_bap6[4], 0.78, 0.94)),
            float(np.clip(tgt_bap6[5], 0.84, 0.96)),
        ],
        dtype=np.float64,
    )
    cur_bap_med = np.median(proj.bap[v_mask], axis=0)
    d_logit = np.clip(v6_cloner._logit(bap_ref) - v6_cloner._logit(cur_bap_med), -1.35, 1.35) * 0.85
    cur_logit = v6_cloner._logit(proj.bap[v_mask])
    cur_med_logit = v6_cloner._logit(cur_bap_med)
    tgt_logit = cur_med_logit + d_logit
    spread_sc = 0.50 + 0.30 * drive_c
    bap_shifted = v6_cloner._sigmoid(tgt_logit[None, :] + spread_sc * (cur_logit - cur_med_logit[None, :]))
    bap_lo = np.asarray([0.016, 0.11, 0.18, 0.30, 0.45, 0.55], dtype=np.float64)
    bap_hi = np.asarray(
        [0.14 + 0.12 * drive_c, 0.42 + 0.14 * drive_c, 0.66 + 0.14 * drive_c, 0.86, 0.94, 0.96],
        dtype=np.float64,
    )
    proj.bap[v_mask] = np.clip(bap_shifted, bap_lo[None, :], 0.985 if use_pure_zh_pivot else bap_hi[None, :])

    # Stage 4/7 human vocal-fold micro-perturbation (jitter ~0.0046, shimmer ~0.021) & crisp modal Rd around tgt_f0_med
    rel_st_proj = 12.0 * np.log2(proj.f0_hz[v_mask] / max(60.0, tgt_f0_med))
    nat_jit = np.clip(0.0046 - 0.0005 * rel_st_proj, 0.0042, 0.0068)
    nat_shim = np.clip(0.021 - 0.002 * rel_st_proj, 0.018, 0.030)
    if proj.jitter_frac is not None:
        proj.jitter_frac[v_mask] = np.clip(0.5 * proj.jitter_frac[v_mask] + 0.5 * nat_jit, 0.0042, 0.0072)
    if proj.shimmer_frac is not None:
        proj.shimmer_frac[v_mask] = np.clip(0.5 * proj.shimmer_frac[v_mask] + 0.5 * nat_shim, 0.018, 0.032)
    if proj.rd is not None:
        proj.rd[v_mask] = np.clip(1.06 - 0.024 * rel_st_proj, 0.78, 1.36)
    if not use_pure_zh_pivot:
        _apply_en_stage4_anti_buzz_glottal(proj, v_mask, tgt_f0_med, expressive_drive=expressive_drive)
    if is_fullband_src and matched_voice_eq is not None:
        # Replace the utterance EQ while holding the calibrated low-band level
        # and all unvoiced processing fixed for a controlled comparison.
        baseline = proj.log_env[v_mask]
        delta = (matched_voice_eq.curve - v_eq_total) * matched_voice_eq.blend
        adjusted = baseline + delta[None, :]
        old_power = np.sum(np.exp(2.0 * baseline[:, _VOICE_LEVEL_BAND]), axis=1)
        new_power = np.sum(np.exp(2.0 * adjusted[:, _VOICE_LEVEL_BAND]), axis=1)
        adjusted += (0.5 * np.log(old_power / new_power))[:, None]
        proj.log_env[v_mask] = np.clip(adjusted, -16.0, 2.5)
    elif is_fullband_src and not use_pure_zh_pivot and not limit_treble:
        # Restrain excessive +3.5..9.5 kHz treble projection boost without cratering 10-11.5 kHz LTAS
        delta_t = proj.log_env[v_mask] - ctrl.log_env[v_mask]
        lvl_shift = np.median(delta_t[:, _VOICE_LEVEL_BAND], axis=1)
        excess_t = gaussian_filter1d(delta_t, sigma=12.0, axis=-1) - lvl_shift[:, None]
        w_up = 0.5 - 0.5 * np.cos(math.pi * np.clip((_FREQS - 3200.0) / 1400.0, 0.0, 1.0))
        w_dn = 0.5 + 0.5 * np.cos(math.pi * np.clip((_FREQS - 9500.0) / 2000.0, 0.0, 1.0))
        mid_treble_win = w_up * w_dn
        excess_clip = np.clip(excess_t - 3.5 * _DB_TO_NEPER, 0.0, 1.43)
        damp_t = np.minimum(0.40, 0.28 * excess_clip) * mid_treble_win[None, :]
        proj.log_env[v_mask] = np.clip(proj.log_env[v_mask] - damp_t, -16.0, 3.0)
    if is_fullband_src and limit_treble:
        _limit_projected_treble(ctrl, proj, v_mask)
    return proj


def build_stage10_en_control(
    text_or_units: str | Sequence[dict[str, Any]],
    base_f0_hz: float,
    params: Any,
    zh_kb: Any | None = None,
    *,
    en_kb: AcousticKnowledgeEnStage10 | None = None,
    expressive_seg: ExpressiveSegment | None = None,
    clause_pos: tuple[float, float] = (0.0, 1.0),
    is_clause_final: bool = True,
    is_question: bool | None = None,
    project_to_zh_speaker: bool = True,
    expand_projected_pitch: bool = True,
    continues_after: bool = False,
    embedded_in_zh: bool = False,
    limit_projected_treble: bool = False,
    matched_voice_eq: MatchedVoiceEQ | None = None,
    next_boundary_cls: int | None = None,
) -> tuple[LarynxControl, float]:
    """Render English text or EN unit run into Stage 10/12 `LarynxControl` (+ subharmonic strength)."""
    use_sf_pivot = bool(embedded_in_zh and en_kb is None and is_stage12_sf_ready())
    kb10 = _get_sf_stage12_knowledge() if use_sf_pivot else (en_kb if en_kb is not None else get_en_stage10_knowledge())
    if kb10 is None:
        raise FileNotFoundError(f"Missing Stage 10 EN acoustic knowledge: {EN_STAGE10_KNOWLEDGE_PATH}")

    f0_eff = float(base_f0_hz)
    range_sc = 1.0
    speed_eff = max(0.60, min(1.60, float(getattr(params, "speed", 1.0))))
    if expressive_seg is not None and expressive_seg.intensity > 1e-4:
        e_prof = EMOTION_PROFILES.get(expressive_seg.emotion, EMOTION_PROFILES["neutral"])
        sc = float(expressive_seg.intensity)
        f0_eff = float(np.clip(f0_eff * (2.0 ** ((float(e_prof["f0_shift_st"]) * sc) / 12.0)), 65.0, 460.0))
        range_sc = 1.0 + (float(e_prof["f0_range_scale"]) - 1.0) * sc
        speed_eff = float(np.clip(speed_eff * (1.0 + (float(e_prof["speed_scale"]) - 1.0) * sc), 0.60, 1.60))

    expressive_drive = _compute_expressive_drive(expressive_seg, params, is_question, text_or_units)
    if expressive_drive > 0.0:
        range_sc = min(range_sc, 1.42)

    # Apply same-speaker bilingual speed & pitch-span ratio when embedded inside ZH clauses
    if use_sf_pivot:
        sf_cons = load_sf_bilingual_consistency()
        range_sc *= float(np.clip(float(sf_cons.get("iqr_ratio_en_over_zh", 0.93)), 0.78, 1.12))
        zh_r = float((sf_cons.get("sf_zh_profile") or {}).get("syllable_rate_hz", 6.24))
        en_r = float((sf_cons.get("sf_en_profile") or {}).get("syllable_rate_hz", 5.98))
        speed_eff = float(np.clip(speed_eff * float(np.clip(en_r / max(1.0, zh_r), 0.88, 1.12)), 0.60, 1.60))
    elif project_to_zh_speaker:
        ref_stats = _load_bilingual_ref_stats()
        rate_ratio = float(np.clip(float(ref_stats.get("en_zh_rate_ratio", 1.0)), 0.85, 1.15)) if ref_stats else 1.0
        speed_eff = float(np.clip(speed_eff * rate_ratio, 0.60, 1.60))

    pause_scale = max(0.40, min(2.00, float(getattr(params, "pause_scale", 1.0))))
    seed = int(getattr(params, "seed", 2026))
    focus_spans = expressive_seg.focus_spans if expressive_seg is not None else None

    if not isinstance(text_or_units, str):
        copied_units = []
        u_list = list(text_or_units)
        for idx, u in enumerate(u_list):
            uc = dict(u)
            uc["pause_after_ms"] = float(u.get("pause_after_ms", 0.0)) if idx < len(u_list) - 1 else 0.0
            copied_units.append(uc)
        prosody_in: str | Sequence[dict[str, Any]] = copied_units
    else:
        prosody_in = text_or_units

    is_proj_pure_en = bool(project_to_zh_speaker and not use_sf_pivot and not embedded_in_zh)
    eff_clause_final = bool(is_clause_final and not continues_after)
    phone_plans, f0_hz, voiced, gain_fr, _q_flag = _predict_en_prosody_stage10_or_fallback(
        prosody_in,
        base_f0_hz=float(kb10.speaker_f0_med_hz),
        speed=speed_eff,
        pause_scale=pause_scale,
        range_sc=range_sc,
        clause_pos=clause_pos,
        is_clause_final=eff_clause_final,
        is_question=is_question,
        focus_spans=focus_spans,
        seed=seed,
        embedded_in_zh=embedded_in_zh,
        expressive_drive=expressive_drive if is_proj_pure_en else None,
    )

    eff_gain_scale = (0.055 + 0.080 * expressive_drive) if is_proj_pure_en else (0.14 if not project_to_zh_speaker else 0.08)
    en_ctrl, _ = build_en_stage10_control(
        phone_plans,
        f0_hz,
        voiced,
        kb10,
        taper_vec=None if (project_to_zh_speaker or use_sf_pivot) else _NATIVE_LJSPEECH_TAPER_24,
        contrast_bell_peak=0.0 if (not project_to_zh_speaker and not use_sf_pivot) else 0.08,
        gv_strength=0.0 if (not project_to_zh_speaker and not use_sf_pivot) else 0.14,
        gain_scale=eff_gain_scale,
        gain_db_frames=gain_fr,
        continues_after=continues_after,
        next_boundary_cls=next_boundary_cls,
    )

    if project_to_zh_speaker and len(en_ctrl.f0_hz) > 0:
        en_ctrl = _project_en_to_bznsyp(
            en_ctrl,
            kb10,
            zh_kb,
            f0_eff,
            range_sc,
            expand_pitch=expand_projected_pitch,
            embedded_in_zh=embedded_in_zh,
            limit_treble=limit_projected_treble,
            matched_voice_eq=matched_voice_eq,
            use_sf_pivot=use_sf_pivot,
            expressive_drive=expressive_drive,
        )

    subharm_strength = 0.0
    if expressive_seg is not None and len(en_ctrl.f0_hz) > 0:
        en_ctrl, subharm_strength = apply_emotion_to_control(
            en_ctrl,
            expressive_seg,
            stability=float(getattr(params, "stability", 0.55)),
            seed=seed,
        )

    return en_ctrl, float(subharm_strength)
