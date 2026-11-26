"""Stage 10 Unblurred Contextual English Medoid + Hierarchical Bayesian Ridge Acoustic Engine.

Zero-ML-weight deterministic acoustic model (`allow_pickle=False` `.npz`):
- K <= 4 unblurred 5-bin TrueHarmEnv 64-dim cepstral + 6-band BAP + voicing medoids
  per phone & positional allophone (`l_coda`, `*_coda`), conditioned on
  `(prev_cls, phone, next_cls, stress)`.
- Hierarchical Bayesian Ridge weights predicting `cep[0:24]` blended via a
  sharpness-preserving taper so `cep[24:64]` and `q8_24` formant sharpness stay crisp.
- Catmull-Rom cubic phase interpolation, real 6-band BAP trajectories, VOT edge
  ramping, and Stage 7-style `cep[1:18]` seam bridging across voiced phones & links.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from exp_20_pybot_larynx import (
    EXP_DIR,
    HOP,
    N_BAP,
    N_BINS,
    N_CEP,
)
from exp_20_pybot_larynx.acoustics.knowledge import _interp_phase_bins
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import (
    _BELL_COS_WEIGHT,
    _BELL_Q_END,
    _BELL_Q_START,
    MedoidGroup,
    _load_medoid_table,
)
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import (
    _GV_BELL_W,
    _GV_Q_END,
    _GV_Q_START,
    _interp_phase_bins_cubic,
    _load_gv_table,
)
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import (
    _load_proto_dict,
    _load_weight_dict,
)
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB
from exp_20_pybot_larynx.crossling.en_articulation import (
    ART_CLASSES,
    VOICELESS_OBSTRUENTS,
    VOWEL_IPAS,
    classify_articulatory_class,
    resolve_en_allophone,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS

EN_STAGE10_KNOWLEDGE_PATH = EXP_DIR / "crossling" / "knowledge_en_stage10.npz"
EN_STAGE10_SUMMARY_PATH = EXP_DIR / "crossling" / "knowledge_en_stage10_summary.json"

N_EN_BINS = 5
N_COARSE_CEP_S10 = 24
N_ART_CLS = len(ART_CLASSES)  # 9
D_ACOUSTIC_FEAT_S10 = 61

_EN_BIN_CENTERS = (np.arange(N_EN_BINS, dtype=np.float64) + 0.5) / float(N_EN_BINS)

# Sharpness-preserving tapered blending over cep[0:24]:
# Strongly tracks energy/tilt/coarticulation on cep[0:8], gently guides cep[8:16],
# preserves crisp unblurred medoid formant peaks above cep[16] & 100% of cep[24:64].
DEFAULT_EN_TAPER_24 = np.array(
    [0.86] * 8 + [0.46] * 8 + [0.20] * 8,
    dtype=np.float64,
)

_VOT_EDGE_FLOOR = np.asarray([0.65, 0.75, 0.82, 0.88, 0.92, 0.95], dtype=np.float64)
_ZERO_8 = np.zeros(8, dtype=np.float64)

_STOPS_AFFR = frozenset({"p", "pʰ", "t", "tʰ", "k", "kʰ", "b", "d", "ɡ", "tʃ", "dʒ"})
_OBSTRUENTS = frozenset({
    "p", "pʰ", "t", "tʰ", "k", "kʰ", "b", "d", "ɡ",
    "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h", "tʃ", "dʒ",
})

_KB_LOCK = Lock()
_CACHED_KB_PATH: str | None = None
_CACHED_KB: "AcousticKnowledgeEnStage10 | None" = None


def normalize_stress(stress: int | None, ipa: str) -> int:
    """Normalize stress into {0: unstressed vowel, 1: primary, 2: secondary, 3: consonant}."""
    base = str(ipa).replace("_coda", "").strip()
    if base not in VOWEL_IPAS:
        return 3
    if stress == 1:
        return 1
    if stress == 2:
        return 2
    return 0


def extract_en_stage10_acoustic_features(
    *,
    prev_cls: int,
    next_cls: int,
    stress: int,
    pos_id: int,
    word_pos: int,
    is_func_word: bool,
    is_coda: bool,
    same_word_prev: bool,
    same_word_next: bool,
    dur_ms: float,
    f3_st: tuple[float, float, float] | np.ndarray,
    pause_prev_ms: float,
    pause_after_ms: float,
    prev_phone: str,
    next_phone: str,
    p_end_proto: dict[str, np.ndarray],
    p_start_proto: dict[str, np.ndarray],
) -> np.ndarray:
    """Build deterministic 61-dim physical context vector for Stage 10 EN Ridge models."""
    x = np.zeros(D_ACOUSTIC_FEAT_S10, dtype=np.float64)
    c = 0
    x[c] = 1.0
    c += 1

    x[c + max(0, min(N_ART_CLS - 1, int(prev_cls)))] = 1.0
    c += N_ART_CLS  # -> 10

    x[c + max(0, min(N_ART_CLS - 1, int(next_cls)))] = 1.0
    c += N_ART_CLS  # -> 19

    x[c + max(0, min(3, int(stress)))] = 1.0
    c += 4  # -> 23

    x[c + max(0, min(2, int(pos_id)))] = 1.0
    c += 3  # -> 26

    x[c + max(0, min(3, int(word_pos)))] = 1.0
    c += 4  # -> 30

    x[c] = 1.0 if is_func_word else 0.0
    x[c + 1] = 1.0 if is_coda else 0.0
    x[c + 2] = 1.0 if same_word_prev else 0.0
    x[c + 3] = 1.0 if same_word_next else 0.0
    c += 4  # -> 34

    ld = math.log(max(15.0, float(dur_ms)) / 75.0)
    p0 = float(f3_st[0]) / 5.0
    p1 = float(f3_st[1]) / 5.0
    p2 = float(f3_st[2]) / 5.0
    x[c : c + 11] = (
        ld,
        ld * ld,
        p0,
        p1,
        p2,
        p2 - p0,
        p0 + p2 - 2.0 * p1,
        math.log1p(max(0.0, float(pause_prev_ms)) / 50.0),
        math.log1p(max(0.0, float(pause_after_ms)) / 50.0),
        float(float(pause_prev_ms) >= 25.0),
        float(float(pause_after_ms) >= 25.0),
    )
    c += 11  # -> 45

    pe = _ZERO_8 if (not prev_phone or prev_phone == "sil") else p_end_proto.get(prev_phone, _ZERO_8) / 15.0
    ns = _ZERO_8 if (not next_phone or next_phone == "sil") else p_start_proto.get(next_phone, _ZERO_8) / 15.0
    x[c : c + 8] = pe
    c += 8
    x[c : c + 8] = ns
    return x


@dataclass
class AcousticKnowledgeEnStage10:
    """Stage 10 English unblurred K<=4 contextual medoids + Hierarchical Bayesian Ridge."""

    ctx_pns: dict[str, MedoidGroup]
    ctx_pn: dict[str, MedoidGroup]
    ctx_ps: dict[str, MedoidGroup]
    ctx_p: dict[str, MedoidGroup]
    ctx_base: dict[str, MedoidGroup]
    en_global: tuple[np.ndarray, np.ndarray, np.ndarray]

    # Hierarchical Bayesian Ridge weights: W (61, 5 * C_DIM) and rare residuals d (5, C_DIM)
    w_base: dict[str, np.ndarray]
    w_phone: dict[str, np.ndarray]
    w_ps: dict[str, np.ndarray]
    w_pns: dict[str, np.ndarray]
    d_pns: dict[str, np.ndarray]
    w_bap_ps: dict[str, np.ndarray]

    # Articulatory boundary prototypes (8-dim cep[1:9])
    p_start_proto: dict[str, np.ndarray]
    p_end_proto: dict[str, np.ndarray]

    sil_cep64: np.ndarray
    sil_bap6: np.ndarray
    speaker_f0_med_hz: float
    ref_log_amp: float
    n_train_utts: int
    c_dim: int = N_COARSE_CEP_S10

    gv_ps: dict[str, np.ndarray] = field(default_factory=dict)
    gv_glob: np.ndarray = field(default_factory=lambda: np.ones(40, dtype=np.float64))

    @classmethod
    def load(cls, path: Path | str = EN_STAGE10_KNOWLEDGE_PATH) -> "AcousticKnowledgeEnStage10":
        with np.load(Path(path), allow_pickle=False) as d:
            c_dim = int(d["c_dim"][0]) if "c_dim" in d else N_COARSE_CEP_S10
            return cls(
                ctx_pns=_load_medoid_table(d, "ctx_pns"),
                ctx_pn=_load_medoid_table(d, "ctx_pn"),
                ctx_ps=_load_medoid_table(d, "ctx_ps"),
                ctx_p=_load_medoid_table(d, "ctx_p"),
                ctx_base=_load_medoid_table(d, "ctx_base"),
                en_global=(
                    d["en_glob_cep"].astype(np.float64),
                    d["en_glob_bap"].astype(np.float64),
                    d["en_glob_vp"].astype(np.float64),
                ),
                w_base=_load_weight_dict(d, "w_base"),
                w_phone=_load_weight_dict(d, "w_phone"),
                w_ps=_load_weight_dict(d, "w_ps"),
                w_pns=_load_weight_dict(d, "w_pns"),
                d_pns=_load_weight_dict(d, "d_pns"),
                w_bap_ps=_load_weight_dict(d, "w_bap_ps"),
                p_start_proto=_load_proto_dict(d, "p_start_proto"),
                p_end_proto=_load_proto_dict(d, "p_end_proto"),
                sil_cep64=d["sil_cep64"].astype(np.float64),
                sil_bap6=d["sil_bap6"].astype(np.float64),
                speaker_f0_med_hz=float(d["speaker_f0_med_hz"][0]),
                ref_log_amp=float(d["ref_log_amp"][0]),
                n_train_utts=int(d["n_train_utts"][0]),
                c_dim=c_dim,
                gv_ps=_load_gv_table(d, "gv_ps"),
                gv_glob=d["gv_glob"].astype(np.float64) if "gv_glob" in d else np.ones(40),
            )

    def predict_ridge(
        self,
        prev_cls: int,
        phone: str,
        next_cls: int,
        stress: int,
        x_feat: np.ndarray,
    ) -> np.ndarray | None:
        base = phone.replace("_coda", "")
        k_pns = f"{prev_cls}|{phone}|{next_cls}|{stress}"
        w_exact = self.w_pns.get(k_pns)
        if w_exact is not None:
            return (x_feat @ w_exact).reshape(N_EN_BINS, self.c_dim)

        w_parent = self.w_ps.get(
            f"{phone}|{stress}",
            self.w_phone.get(phone, self.w_base.get(base)),
        )
        if w_parent is None:
            return None
        pred = (x_feat @ w_parent).reshape(N_EN_BINS, self.c_dim)
        d_rare = self.d_pns.get(k_pns)
        if d_rare is not None:
            pred = pred + d_rare
        return pred

    def predict_bap_ridge(self, phone: str, stress: int, x_feat: np.ndarray) -> np.ndarray | None:
        base = phone.replace("_coda", "")
        w = self.w_bap_ps.get(
            f"{phone}|{stress}",
            self.w_bap_ps.get(f"{base}|{stress}", self.w_bap_ps.get(f"{base}|0")),
        )
        if w is None:
            return None
        return np.clip((x_feat @ w).reshape(N_EN_BINS, N_BAP), 0.004, 0.995)

    def lookup_gv(self, phone: str, stress: int) -> np.ndarray:
        base = phone.replace("_coda", "")
        return self.gv_ps.get(f"{phone}|{stress}", self.gv_ps.get(f"{base}|{stress}", self.gv_glob))

    def lookup_candidates(
        self,
        prev_cls: int,
        phone: str,
        next_cls: int,
        stress: int,
    ) -> tuple[MedoidGroup | None, str]:
        g = self.ctx_pns.get(f"{prev_cls}|{phone}|{next_cls}|{stress}")
        if g is not None and g.total_tokens >= 3:
            return g, "ctx_pns"
        g = self.ctx_pn.get(f"{phone}|{next_cls}|{stress}")
        if g is not None and g.total_tokens >= 3:
            return g, "ctx_pn"
        g = self.ctx_ps.get(f"{phone}|{stress}")
        if g is not None and g.total_tokens >= 2:
            return g, "ctx_ps"
        g = self.ctx_p.get(phone)
        if g is not None and g.total_tokens >= 1:
            return g, "ctx_p"
        base = phone.replace("_coda", "")
        g = self.ctx_base.get(base)
        if g is not None and g.total_tokens >= 1:
            return g, "ctx_base"
        return None, "fallback"


def get_en_stage10_knowledge(path: Path | str = EN_STAGE10_KNOWLEDGE_PATH) -> AcousticKnowledgeEnStage10 | None:
    """Thread-safe cached loader for :class:`AcousticKnowledgeEnStage10` (None if missing)."""
    global _CACHED_KB_PATH, _CACHED_KB
    p = Path(path)
    p_str = str(p.resolve()) if p.exists() else str(p)
    with _KB_LOCK:
        if _CACHED_KB is not None and _CACHED_KB_PATH == p_str:
            return _CACHED_KB
    if not p.exists():
        return None
    kb = AcousticKnowledgeEnStage10.load(p)
    with _KB_LOCK:
        _CACHED_KB = kb
        _CACHED_KB_PATH = p_str
    return kb


def clear_en_stage10_cache() -> None:
    """Reset module cache (used after rebuilding `knowledge_en_stage10.npz`)."""
    global _CACHED_KB_PATH, _CACHED_KB
    with _KB_LOCK:
        _CACHED_KB_PATH = None
        _CACHED_KB = None


def _score_en_medoid_ridge_guided(
    grp: MedoidGroup,
    pred_cep: np.ndarray | None,
    prev_cls: int,
    next_cls: int,
    pos_id: int,
    dur_ms: float,
    f0_st: float,
    *,
    w_ridge: float = 0.85,
) -> np.ndarray:
    """Score K <= 4 unblurred medoids by Ridge acoustic proximity + context cues."""
    costs = (
        np.where(grp.ctx_prev == prev_cls, 0.0, 0.16)
        + np.where(grp.ctx_next == next_cls, 0.0, 0.14)
        + np.where(grp.ctx_pos == pos_id, 0.0, 0.10)
        + 0.24 * np.abs(np.log(np.maximum(10.0, dur_ms) / np.maximum(10.0, grp.ctx_dur_ms)))
        + 0.025 * np.abs(f0_st - grp.ctx_f0_st)
        + 0.08 * (1.0 - grp.cluster_sz.astype(np.float64) / float(max(1, int(np.max(grp.cluster_sz)))))
    )
    if pred_cep is not None:
        d_spec = np.sqrt(np.mean((grp.cep[:, :, 1:24] - pred_cep[None, :, 1:24]) ** 2, axis=(1, 2)))
        d_c0 = np.mean(np.abs(grp.cep[:, :, 0] - pred_cep[None, :, 0]), axis=1)
        costs += w_ridge * (d_spec + 0.15 * d_c0)
    return costs


def _smooth_en_voiced_seams(
    cep: np.ndarray,
    voiced: np.ndarray,
    seg_meta: list[tuple[int, int, str, str, bool, float]],
) -> None:
    """Smooth voiced coarticulation across continuant seams without blunting obstruent attacks."""
    n_fr = cep.shape[0]
    for i in range(1, len(seg_meta)):
        prev_a, prev_b, prev_base, _prev_ph, prev_is_pause, prev_pause_ms = seg_meta[i - 1]
        cur_a, cur_b, cur_base, _cur_ph, cur_is_pause, _cur_pause_ms = seg_meta[i]
        if prev_is_pause or cur_is_pause or prev_pause_ms >= 20.0:
            continue
        if abs(cur_a - prev_b) > 1 or cur_a < 2 or cur_a + 1 >= n_fr:
            continue
        # Protect voiceless obstruent attacks completely
        if prev_base in VOICELESS_OBSTRUENTS or cur_base in _OBSTRUENTS:
            continue

        # Smooth 4-frame symmetric blend over cep[1:28] across voiced continuant/sonorant seams
        is_continuant = (prev_base not in _OBSTRUENTS) and (cur_base not in _OBSTRUENTS)
        if (
            is_continuant
            and (prev_b - prev_a) >= 3
            and (cur_b - cur_a) >= 3
            and cur_a >= 3
            and cur_a + 2 < n_fr
            and bool(np.all(voiced[cur_a - 3 : cur_a + 3]))
        ):
            c_l2 = cep[cur_a - 3, 1:28].copy()
            c_r2 = cep[cur_a + 2, 1:28].copy()
            if float(np.sqrt(np.mean((c_r2 - c_l2) ** 2))) >= 0.04:
                cep[cur_a - 2, 1:28] = 0.80 * cep[cur_a - 2, 1:28] + 0.20 * (0.80 * c_l2 + 0.20 * c_r2)
                cep[cur_a - 1, 1:28] = 0.60 * cep[cur_a - 1, 1:28] + 0.40 * (0.60 * c_l2 + 0.40 * c_r2)
                cep[cur_a, 1:28] = 0.60 * cep[cur_a, 1:28] + 0.40 * (0.40 * c_l2 + 0.60 * c_r2)
                cep[cur_a + 1, 1:28] = 0.80 * cep[cur_a + 1, 1:28] + 0.20 * (0.20 * c_l2 + 0.80 * c_r2)
                continue

        if not (voiced[cur_a - 2] and voiced[cur_a - 1] and voiced[cur_a] and voiced[cur_a + 1]):
            continue
        c_left = cep[cur_a - 2, 1:18].copy()
        c_right = cep[cur_a + 1, 1:18].copy()
        if float(np.sqrt(np.mean((c_right - c_left) ** 2))) < 0.05:
            continue
        cep[cur_a - 1, 1:18] = 0.65 * cep[cur_a - 1, 1:18] + 0.35 * (0.67 * c_left + 0.33 * c_right)
        cep[cur_a, 1:18] = 0.65 * cep[cur_a, 1:18] + 0.35 * (0.33 * c_left + 0.67 * c_right)


def build_en_stage10_control(
    phone_plans: list[dict[str, Any]],
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    knowledge: AcousticKnowledgeEnStage10,
    *,
    taper_vec: np.ndarray | None = None,
    contrast_bell_peak: float = 0.12,
    gv_strength: float = 0.22,
    bap_ridge_w: float = 0.35,
    seam_polish: bool = True,
    vot_ramp: bool = True,
    gain_scale: float = 0.35,
    gain_db_frames: np.ndarray | None = None,
    continues_after: bool = False,
    next_boundary_cls: int | None = None,
) -> tuple[LarynxControl, dict[str, Any]]:
    """Build Stage 10 frame-level EN LarynxControl from phone plans, F0 track & knowledge."""
    n_fr = len(f0_hz)
    f0_arr = np.asarray(f0_hz, dtype=np.float64).copy()
    voi_arr = np.asarray(voiced, dtype=bool).copy()

    cep = np.repeat(knowledge.sil_cep64[None, :], n_fr, axis=0)
    bap = np.repeat(knowledge.sil_bap6[None, :], n_fr, axis=0)
    tilt = np.zeros(n_fr, dtype=np.float64)
    jit = np.full(n_fr, 0.0055, dtype=np.float64)
    shim = np.full(n_fr, 0.025, dtype=np.float64)
    rd = np.ones(n_fr, dtype=np.float64)

    hit_counts: dict[str, int] = {
        "ctx_pns": 0,
        "ctx_pn": 0,
        "ctx_ps": 0,
        "ctx_p": 0,
        "ctx_base": 0,
        "fallback": 0,
        "ridge_pns": 0,
        "ridge_parent": 0,
    }

    n_p = len(phone_plans)
    if n_p == 0 or n_fr == 0:
        log_env = np.clip(cep_to_log_env(cep, n_bins=N_BINS) + knowledge.ref_log_amp, -16.0, 6.0)
        return LarynxControl(
            f0_hz=f0_arr,
            voiced=voi_arr,
            log_env=log_env,
            bap=np.clip(bap, 0.004, 0.995),
            glottal_tilt_db_per_oct=tilt,
            jitter_frac=jit,
            shimmer_frac=shim,
            n_samples=n_fr * HOP,
            rd=rd,
        ), {"mode": "en_stage10", "hits": hit_counts}

    ref_f0 = max(50.0, float(knowledge.speaker_f0_med_hz))

    # Pre-resolve phone context & Ridge predictions for non-pause segments
    nonpause_indices = [i for i, p in enumerate(phone_plans) if not p.get("is_pause", False)]
    cands: dict[int, MedoidGroup | None] = {}
    pred_ceps: dict[int, np.ndarray | None] = {}
    pred_baps: dict[int, np.ndarray | None] = {}
    phone_meta: dict[int, tuple[int, int, int, int, int, int, float, float, str, str, float]] = {}

    for pos_np, idx in enumerate(nonpause_indices):
        p = phone_plans[idx]
        a = max(0, min(n_fr, int(p["start_fr"])))
        b = max(a, min(n_fr, int(p["end_fr"])))
        base_ipa = str(p.get("ipa", "ə")).strip()
        same_w_next = bool(p.get("same_word_next", False))
        same_w_prev = bool(p.get("same_word_prev", False))
        next_ipa_raw = p.get("next_ipa")
        phone = str(p.get("phone") or resolve_en_allophone(base_ipa, next_ipa_raw, same_w_next))
        is_coda = phone.endswith("_coda")

        prev_p = phone_plans[idx - 1] if idx > 0 else None
        next_p = phone_plans[idx + 1] if idx + 1 < n_p else None
        pause_prev = float(prev_p.get("pause_after_ms", 0.0)) if (prev_p and not prev_p.get("is_pause", False)) else (999.0 if idx == 0 or (prev_p and prev_p.get("is_pause", False)) else 0.0)
        if prev_p is not None and not prev_p.get("is_pause", False) and (a - int(prev_p["end_fr"])) >= 2:
            pause_prev = max(pause_prev, float(a - int(prev_p["end_fr"])) * 5.0)
        pause_after = float(p.get("pause_after_ms", 0.0))
        is_tail_continued = continues_after and (pos_np == len(nonpause_indices) - 1) and (pause_after < 20.0)
        if next_p is None or next_p.get("is_pause", False):
            if is_tail_continued:
                pause_after = 0.0
            else:
                pause_after = max(pause_after, 999.0 if next_p is None else float(next_p.get("dur_ms", 50.0)))

        prev_phone = "sil" if (prev_p is None or prev_p.get("is_pause", False) or pause_prev >= 25.0) else str(prev_p.get("phone") or prev_p.get("ipa") or "sil")
        if is_tail_continued:
            next_phone = "ə"
            next_cls = int(next_boundary_cls) if next_boundary_cls is not None else 1
        else:
            next_phone = "sil" if (next_p is None or next_p.get("is_pause", False) or pause_after >= 25.0) else str(next_p.get("phone") or next_p.get("ipa") or "sil")
            next_cls = 0 if next_phone == "sil" else int(p.get("next_cls", classify_articulatory_class(next_phone)))

        prev_cls = 0 if prev_phone == "sil" else int(p.get("prev_cls", classify_articulatory_class(prev_phone)))
        stress = normalize_stress(p.get("stress"), base_ipa)
        pos_id = 0 if prev_cls == 0 else (2 if next_cls == 0 else 1)
        word_pos = int(p.get("word_pos", 0))
        is_func = bool(p.get("is_func_word", False))

        dur_ms = float(max(1, b - a)) * 5.0
        v_sub = voi_arr[a:b] & (f0_arr[a:b] > 40.0)
        f0_st = float(12.0 * np.log2(np.median(f0_arr[a:b][v_sub]) / ref_f0)) if np.any(v_sub) else 0.0
        vf = np.where(v_sub)[0] if b > a else np.zeros(0, dtype=np.intp)
        if len(vf) >= 2:
            uv = vf.astype(np.float64) / float(max(1, (b - a) - 1))
            st_v = 12.0 * np.log2(f0_arr[a:b][vf] / ref_f0)
            p3 = np.interp([0.20, 0.50, 0.80], uv, st_v)
            f3_st = (float(p3[0]), float(p3[1]), float(p3[2]))
        elif len(vf) == 1:
            val = float(12.0 * np.log2(f0_arr[a:b][vf[0]] / ref_f0))
            f3_st = (val, val, val)
        else:
            f3_st = (f0_st, f0_st, f0_st)

        x_feat = extract_en_stage10_acoustic_features(
            prev_cls=prev_cls,
            next_cls=next_cls,
            stress=stress,
            pos_id=pos_id,
            word_pos=word_pos,
            is_func_word=is_func,
            is_coda=is_coda,
            same_word_prev=same_w_prev,
            same_word_next=same_w_next,
            dur_ms=dur_ms,
            f3_st=f3_st,
            pause_prev_ms=pause_prev,
            pause_after_ms=pause_after,
            prev_phone=prev_phone,
            next_phone=next_phone,
            p_end_proto=knowledge.p_end_proto,
            p_start_proto=knowledge.p_start_proto,
        )

        pr = knowledge.predict_ridge(prev_cls, phone, next_cls, stress, x_feat)
        pred_ceps[idx] = pr
        if f"{prev_cls}|{phone}|{next_cls}|{stress}" in knowledge.w_pns:
            hit_counts["ridge_pns"] += 1
        elif pr is not None:
            hit_counts["ridge_parent"] += 1
        pred_baps[idx] = knowledge.predict_bap_ridge(phone, stress, x_feat) if bap_ridge_w > 0.0 else None

        grp, tag = knowledge.lookup_candidates(prev_cls, phone, next_cls, stress)
        hit_counts[tag] += 1
        cands[idx] = grp
        phone_meta[idx] = (a, b, prev_cls, next_cls, stress, pos_id, dur_ms, f0_st, base_ipa, phone, pause_prev)

    # Step 1: Ridge-Guided Viterbi DP across non-pause phones
    chosen_med: dict[int, int] = {}
    if nonpause_indices:
        unary_list: list[np.ndarray] = []
        cep_list: list[np.ndarray] = []
        bap_list: list[np.ndarray] = []
        vp_list: list[np.ndarray] = []
        for idx in nonpause_indices:
            _a, _b, prev_c, next_c, _st, pos_id, dur_ms, f0_st, _bi, _ph, _pp = phone_meta[idx]
            grp = cands[idx]
            if grp is None:
                unary_list.append(np.zeros(1, dtype=np.float64))
                cep_list.append(knowledge.en_global[0][None, :, :])
                bap_list.append(knowledge.en_global[1][None, :, :])
                vp_list.append(knowledge.en_global[2][None, :])
            else:
                unary_list.append(
                    _score_en_medoid_ridge_guided(grp, pred_ceps[idx], prev_c, next_c, pos_id, dur_ms, f0_st)
                )
                cep_list.append(grp.cep)
                bap_list.append(grp.bap)
                vp_list.append(grp.vp)

        n_np = len(nonpause_indices)
        dp: list[np.ndarray] = [unary_list[0].copy()]
        back: list[np.ndarray] = []
        for m in range(1, n_np):
            idx_cur = nonpause_indices[m]
            idx_prv = nonpause_indices[m - 1]
            k_prev = len(unary_list[m - 1])
            k_cur = len(unary_list[m])
            cur_a, _cb, prev_c, _nc, _st, _pid, _dur, _f0, cur_base, _ph, pause_prev = phone_meta[idx_cur]
            _pa, prev_b, _pc2, _nc2, _st2, _pid2, _dur2, _f02, prev_base, _ph2, _pp2 = phone_meta[idx_prv]

            contiguous = (prev_c != 0) and (pause_prev < 20.0) and (abs(cur_a - prev_b) <= 1)
            if not contiguous or (cur_base in _STOPS_AFFR) or (prev_base in VOICELESS_OBSTRUENTS):
                trans = np.zeros((k_prev, k_cur), dtype=np.float64)
            else:
                p_end = cep_list[m - 1][:, -1, 1:24]
                p_e0 = cep_list[m - 1][:, -1, :1]
                c_st = cep_list[m][:, 0, 1:24]
                c_e0 = cep_list[m][:, 0, :1]
                d_mat = np.sqrt(np.mean((p_end[:, None, :] - c_st[None, :, :]) ** 2, axis=2))
                d_e = 0.20 * np.abs(p_e0 - c_e0.T)
                trans = 0.42 * (d_mat + d_e)

            total = dp[m - 1][:, None] + trans + unary_list[m][None, :]
            bp = np.argmin(total, axis=0)
            dp.append(total[bp, np.arange(k_cur)])
            back.append(bp)

        best_k = [0] * n_np
        best_k[-1] = int(np.argmin(dp[-1]))
        for m in range(n_np - 2, -1, -1):
            best_k[m] = int(back[m][best_k[m + 1]])
        for m, idx in enumerate(nonpause_indices):
            chosen_med[idx] = best_k[m]

        # Step 2: Sharpness-preserving Tapered Ridge fusion + Catmull-Rom cubic interpolation + GV & bell
        t_vec = DEFAULT_EN_TAPER_24 if taper_vec is None else np.asarray(taper_vec, dtype=np.float64)
        c_dim = min(int(knowledge.c_dim), len(t_vec))
        bell_pk = float(np.clip(contrast_bell_peak, 0.0, 0.35))
        bell_vec = (1.0 + bell_pk * _BELL_COS_WEIGHT)[None, :] if bell_pk > 0.0 else None
        gv_s = float(np.clip(gv_strength, 0.0, 1.0))
        br_w = float(np.clip(bap_ridge_w, 0.0, 1.0))

        seg_meta_seam: list[tuple[int, int, str, str, bool, float]] = []
        for idx_p, p in enumerate(phone_plans):
            if p.get("is_pause", False):
                pa = max(0, min(n_fr, int(p["start_fr"])))
                pb = max(pa, min(n_fr, int(p["end_fr"])))
                if pb > pa:
                    voi_arr[pa:pb] = False
                    f0_arr[pa:pb] = 0.0
                seg_meta_seam.append((pa, pb, "sil", "sil", True, float(p.get("dur_ms", 50.0))))
                continue

            m_pos = nonpause_indices.index(idx_p)
            k_sel = chosen_med[idx_p]
            a, b, _pc, _nc, stress, _pid, _dur, _f0, base_ipa, phone, pause_prev = phone_meta[idx_p]
            seg_meta_seam.append((a, b, base_ipa, phone, False, pause_prev))
            if b <= a:
                continue

            m_cep = cep_list[m_pos][k_sel].copy()
            m_bap = bap_list[m_pos][k_sel].copy()
            m_vp = vp_list[m_pos][k_sel].copy()

            pr = pred_ceps[idx_p]
            if pr is not None:
                av = t_vec[:c_dim][None, :]
                m_cep[:, :c_dim] = av * pr[:, :c_dim] + (1.0 - av) * m_cep[:, :c_dim]

            pb = pred_baps[idx_p]
            if pb is not None and br_w > 0.0:
                m_bap = br_w * pb + (1.0 - br_w) * m_bap

            nf = b - a
            u_ph = np.linspace(0.0, 1.0, nf, dtype=np.float64) if nf > 1 else np.array([0.5], dtype=np.float64)
            cep[a:b, :24] = _interp_phase_bins_cubic(m_cep[:, :24], u_ph, _EN_BIN_CENTERS)
            cep[a:b, 24:] = _interp_phase_bins(m_cep[:, 24:], u_ph, _EN_BIN_CENTERS)
            bap[a:b] = _interp_phase_bins(m_bap, u_ph, _EN_BIN_CENTERS)
            vp_fr = _interp_phase_bins(m_vp[:, None], u_ph, _EN_BIN_CENTERS)[:, 0]

            # Phone-intrinsic voicing gating for stops/fricatives (closure vs burst vs voiced bar)
            spec = IPA_SPECS.get(base_ipa)
            if base_ipa in VOICELESS_OBSTRUENTS:
                voi_arr[a:b] = False
                f0_arr[a:b] = 0.0
            elif spec is not None and spec.manner in ("stop", "affricate", "fricative") and spec.voiced:
                # Manner-aware voicing preservation for voiced obstruents
                is_stop_or_affr = spec.manner in ("stop", "affricate")
                thresh = 0.20 if is_stop_or_affr else 0.18
                v_gate = vp_fr >= thresh
                max_vp = float(np.max(vp_fr)) if nf > 0 else 0.0
                if float(np.mean(voi_arr[a:b])) >= 0.50 and float(np.mean(v_gate)) < 0.50:
                    if is_stop_or_affr and (max_vp >= 0.10 or not is_coda):
                        core_n = max(1, min(nf, int(round(0.65 * nf))))
                        core_mask = np.zeros(nf, dtype=bool)
                        core_mask[:core_n] = True
                        if max_vp >= 0.10:
                            core_cut = max(0.10, 0.55 * max_vp)
                            v_gate = v_gate | (vp_fr >= core_cut) | core_mask
                        else:
                            v_gate = v_gate | core_mask
                    elif (not is_stop_or_affr) and max_vp >= 0.08:
                        v_gate = v_gate | (vp_fr >= max(0.08, 0.45 * max_vp))
                voi_arr[a:b] = voi_arr[a:b] & v_gate
                f0_arr[a:b][~voi_arr[a:b]] = 0.0

            is_sonorant = base_ipa in VOWEL_IPAS or (spec is not None and spec.manner in ("vowel", "diphthong", "glide", "nasal", "liquid"))
            if is_sonorant:
                if bell_vec is not None:
                    cep[a:b, _BELL_Q_START:_BELL_Q_END] *= bell_vec
                if gv_s > 0.0 and nf >= 4:
                    seg_sub = cep[a:b, _GV_Q_START:_GV_Q_END]
                    mu = np.mean(seg_sub, axis=0, keepdims=True)
                    cur_std = np.std(seg_sub, axis=0)
                    tgt_std = knowledge.lookup_gv(phone, stress)[_GV_Q_START:_GV_Q_END]
                    ratio = np.clip(tgt_std / np.maximum(cur_std, 0.15), 0.90, 1.38)
                    scale_q = 1.0 + gv_s * _GV_BELL_W * (ratio - 1.0)
                    cep[a:b, _GV_Q_START:_GV_Q_END] = mu + (seg_sub - mu) * scale_q[None, :]

        if seam_polish:
            _smooth_en_voiced_seams(cep, voi_arr, seg_meta_seam)

    log_env = cep_to_log_env(cep, n_bins=N_BINS) + knowledge.ref_log_amp

    # Step 3: Apply prosodic gain_db gently (smoothed frame curve when available) + glottal/VOT coupling
    g_sc = float(np.clip(gain_scale, 0.0, 1.0))
    if g_sc > 0.0:
        if gain_db_frames is not None and len(gain_db_frames) == n_fr:
            g_arr = np.clip(np.asarray(gain_db_frames, dtype=np.float64), -5.0, 5.0) / NEPER_TO_DB
            log_env = log_env + (g_sc * g_arr)[:, None]
        else:
            for p in phone_plans:
                if p.get("is_pause", False):
                    continue
                g_db = p.get("gain_db")
                if g_db is None or abs(float(g_db)) < 1e-4:
                    continue
                a = max(0, min(n_fr, int(p["start_fr"])))
                b = max(a, min(n_fr, int(p["end_fr"])))
                if b > a:
                    g_nep = float(np.clip(float(g_db), -5.0, 5.0)) / NEPER_TO_DB
                    log_env[a:b] += g_sc * g_nep

    if np.any(voi_arr):
        vm = voi_arr & (f0_arr > 40.0)
        if np.any(vm):
            rel_st = 12.0 * np.log2(f0_arr[vm] / ref_f0)
            rd[vm] = np.clip(1.06 - 0.028 * rel_st, 0.72, 1.50)
            tilt[vm] = np.clip(0.16 * rel_st, -1.25, 1.25)

        if vot_ramp:
            uv_to_v = np.where(voi_arr & np.concatenate([[True], ~voi_arr[:-1]]))[0]
            for v0 in uv_to_v:
                weights = (1.0, 0.55, 0.22)
                for dk, rw in enumerate(weights):
                    fk = v0 + dk
                    if fk < n_fr and voi_arr[fk]:
                        bap[fk] = np.maximum(bap[fk], (1.0 - rw) * bap[fk] + rw * _VOT_EDGE_FLOOR)

    ctrl = LarynxControl(
        f0_hz=f0_arr,
        voiced=voi_arr,
        log_env=np.clip(log_env, -16.0, 6.0),
        bap=np.clip(bap, 0.004, 0.995),
        glottal_tilt_db_per_oct=tilt,
        jitter_frac=jit,
        shimmer_frac=shim,
        n_samples=n_fr * HOP,
        rd=rd,
    )
    return ctrl, {"mode": "en_stage10", "hits": hit_counts}
