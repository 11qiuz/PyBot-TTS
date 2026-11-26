"""Stage 7 Hierarchical Bayesian Ridge + Ridge-Guided Medoid acoustic knowledge & trajectory builder."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np

from exp_20_pybot_larynx import (
    EXP_DIR,
    HOP,
    N_BAP,
    N_BINS,
    N_CEP,
)
from exp_20_pybot_larynx.acoustics.knowledge import _interp_phase_bins, normalize_of
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import (
    _BELL_COS_WEIGHT,
    _BELL_Q_END,
    _BELL_Q_START,
    _COMB_BAP_TP,
    _GLOBAL_COMB_BAP,
    _V9_MED_BAP,
    N_FINAL_BINS_S2,
    N_ONSET_BINS_S2,
    PREV_FG_TO_ID,
    MedoidGroup,
    _load_medoid_table,
    encode_next_place,
    encode_prev_fg,
)
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import (
    _GV_BELL_W,
    _GV_Q_END,
    _GV_Q_START,
    _interp_phase_bins_cubic,
    _load_gv_table,
)
from exp_20_pybot_larynx.arbiter import (
    NEPER_TO_DB,
    OBSTRUENT_MANNERS,
    UtteranceContext,
    classify_final_group,
    classify_onset_manner,
    classify_onset_place,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env

STAGE7_KNOWLEDGE_PATH = EXP_DIR / "acoustics" / "knowledge_stage7.npz"
STAGE7_SUMMARY_PATH = EXP_DIR / "acoustics" / "knowledge_stage7_summary.json"
STAGE7_TRUEENV_PATH = EXP_DIR / "acoustics" / "knowledge_stage7_trueenv.npz"
STAGE7_TRUEENV_SUMMARY_PATH = EXP_DIR / "acoustics" / "knowledge_stage7_trueenv_summary.json"

N_COARSE_CEP_S7 = 28
D_ACOUSTIC_FEAT_S7 = 84
DEFAULT_UV_LEVEL_BOOST_DB_S7 = 0.9

_ONSET_CENTERS_S7 = (np.arange(N_ONSET_BINS_S2, dtype=np.float64) + 0.5) / float(N_ONSET_BINS_S2)
_FINAL_CENTERS_S7 = (np.arange(N_FINAL_BINS_S2, dtype=np.float64) + 0.5) / float(N_FINAL_BINS_S2)

# Sharpness-preserving tapered blending over cep[0:28]:
# strongly tracks coarticulated tilt & low resonances [0:12], gently guides [12:24],
# preserves natural unblurred medoid formant sharpness above cep[24].
DEFAULT_TAPER_28 = np.array(
    [0.92] * 12 + [0.62] * 6 + [0.34] * 6 + [0.16] * 4,
    dtype=np.float64,
)
DEFAULT_TAPER_36 = DEFAULT_TAPER_28  # Backward-compatible alias

OFFGLIDES_VOCAB: tuple[str, ...] = (
    "SIL", "a", "e", "i", "u", "v", "n", "ng", "er", "ir", "iz", "other",
)
_OFFGLIDE_TO_ID: dict[str, int] = {k: i for i, k in enumerate(OFFGLIDES_VOCAB)}

ONSETS_VOCAB: tuple[str, ...] = (
    "SIL", "", "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s",
)
_ONSET_TO_ID: dict[str, int] = {k: i for i, k in enumerate(ONSETS_VOCAB)}

FINAL_GROUPS_VOCAB: tuple[str, ...] = (
    "SIL", "i_front", "u_round", "v_front_round", "a_open_central",
)
_FG_VOCAB_TO_ID: dict[str, int] = {k: i for i, k in enumerate(FINAL_GROUPS_VOCAB)}

_ZERO_8 = np.zeros(8, dtype=np.float64)


def classify_offglide(final: str) -> str:
    """Classify the phonetic coda/offglide of a Mandarin final into 12 articulatory targets."""
    f = str(final or "").strip()
    if not f or f == "SIL":
        return "SIL"
    if f.endswith("ng"):
        return "ng"
    if f.endswith("n"):
        return "n"
    if f == "er":
        return "er"
    if f in ("ir", "ih", "i_zh"):
        return "ir"
    if f in ("iz", "ii", "i_z"):
        return "iz"
    if f.endswith("i"):
        return "i"
    if f.endswith("u") or f.endswith("o"):
        return "u"
    if f in ("v", "ü"):
        return "v"
    if f.endswith("e"):
        return "e"
    if f.endswith("a"):
        return "a"
    return "other"


def extract_stage7_acoustic_features(
    *,
    prev_final: str,
    prev_tone: int,
    next_onset: str,
    next_final: str,
    next_tone: int,
    pos_id: int,
    dur_ms: float,
    on_ms: float,
    f3_st: tuple[float, float, float] | np.ndarray,
    brk: int,
    pause_prev_ms: float,
    pause_after_ms: float,
    f_end_proto: dict[str, np.ndarray],
    o_start_proto: dict[str, np.ndarray],
    f_start_proto: dict[str, np.ndarray],
) -> np.ndarray:
    """Build deterministic 84-dim physical context vector for Stage 7 Ridge acoustic models."""
    x = np.zeros(D_ACOUSTIC_FEAT_S7, dtype=np.float64)
    c = 0
    x[c] = 1.0
    c += 1

    prev_og = "SIL" if prev_final == "SIL" else classify_offglide(prev_final)
    next_fg = "SIL" if next_final == "SIL" else classify_final_group(next_final)

    x[c + _OFFGLIDE_TO_ID.get(prev_og, 11)] = 1.0
    c += len(OFFGLIDES_VOCAB)  # +12 -> 13

    x[c + _ONSET_TO_ID.get(next_onset, 1)] = 1.0
    c += len(ONSETS_VOCAB)  # +23 -> 36

    x[c + _FG_VOCAB_TO_ID.get(next_fg, 4)] = 1.0
    c += len(FINAL_GROUPS_VOCAB)  # +5 -> 41

    x[c + (int(pos_id) % 3)] = 1.0
    c += 3  # -> 44

    x[c + (int(prev_tone) % 6)] = 1.0
    c += 6  # -> 50

    x[c + (int(next_tone) % 6)] = 1.0
    c += 6  # -> 56

    ld = math.log(max(30.0, float(dur_ms)) / 180.0)
    lo = math.log(max(5.0, float(on_ms)) / 50.0) if on_ms > 0.0 else -1.5
    p0 = float(f3_st[0]) / 5.0
    p1 = float(f3_st[1]) / 5.0
    p2 = float(f3_st[2]) / 5.0
    x[c : c + 12] = (
        ld,
        ld * ld,
        lo,
        p0,
        p1,
        p2,
        p2 - p0,
        p0 + p2 - 2.0 * p1,
        float(int(brk) >= 2),
        math.log1p(max(0.0, float(pause_prev_ms)) / 50.0),
        math.log1p(max(0.0, float(pause_after_ms)) / 50.0),
        float(int(brk) >= 1),
    )
    c += 12  # -> 68

    pe = _ZERO_8 if prev_final == "SIL" else f_end_proto.get(prev_final, _ZERO_8) / 15.0
    if next_onset == "SIL":
        ns = _ZERO_8
    elif next_onset and next_onset in o_start_proto:
        ns = o_start_proto[next_onset] / 15.0
    else:
        ns = f_start_proto.get(next_final, _ZERO_8) / 15.0

    x[c : c + 8] = pe
    c += 8
    x[c : c + 8] = ns
    return x


def _load_weight_dict(d: Any, prefix: str) -> dict[str, np.ndarray]:
    k_name = f"{prefix}_keys"
    if k_name not in d:
        return {}
    keys = d[k_name]
    vals = d[f"{prefix}_w"]
    return {str(keys[i]): vals[i].astype(np.float64) for i in range(len(keys))}


def _load_proto_dict(d: Any, prefix: str) -> dict[str, np.ndarray]:
    k_name = f"{prefix}_keys"
    if k_name not in d:
        return {}
    keys = d[k_name]
    vals = d[f"{prefix}_vec"]
    return {str(keys[i]): vals[i].astype(np.float64) for i in range(len(keys))}


@dataclass
class AcousticKnowledgeStage7:
    """Stage 7 hierarchical Bayesian Ridge matrices, rare residuals, unblurred medoids & BAP models."""

    on_of: dict[str, MedoidGroup]
    on_grp: dict[str, MedoidGroup]
    on_only: dict[str, MedoidGroup]
    on_global: tuple[np.ndarray, np.ndarray, np.ndarray]

    fin_oft: dict[str, MedoidGroup]
    fin_pft: dict[str, MedoidGroup]
    fin_ft: dict[str, MedoidGroup]
    fin_f: dict[str, MedoidGroup]
    fin_global: tuple[np.ndarray, np.ndarray, np.ndarray]

    # Hierarchical Bayesian Ridge weights: W (84, T_bins * C_DIM) and rare residual deltas d (T_bins, C_DIM)
    w_ft: dict[str, np.ndarray]
    w_pft: dict[str, np.ndarray]
    w_oft: dict[str, np.ndarray]
    d_oft: dict[str, np.ndarray]

    w_on_o: dict[str, np.ndarray]
    w_on_og: dict[str, np.ndarray]
    w_on_of: dict[str, np.ndarray]
    d_on_of: dict[str, np.ndarray]

    w_bap_ft: dict[str, np.ndarray]

    # Articulatory boundary prototypes (8-dim cep[1:9])
    f_end_proto: dict[str, np.ndarray]
    f_start_proto: dict[str, np.ndarray]
    o_start_proto: dict[str, np.ndarray]

    onset_frac: dict[str, float]
    onset_ms: dict[str, float]
    onset_uv_frac: dict[str, float]
    sil_cep64: np.ndarray
    sil_bap6: np.ndarray
    speaker_f0_med_hz: float
    ref_log_amp: float
    n_train_utts: int
    c_dim: int = N_COARSE_CEP_S7
    is_trueenv: bool = False

    fin_gv_oft: dict[str, np.ndarray] = field(default_factory=dict)
    fin_gv_glob: np.ndarray = field(default_factory=lambda: np.ones(40, dtype=np.float64))

    @classmethod
    def load(cls, path: Path | str = STAGE7_KNOWLEDGE_PATH) -> AcousticKnowledgeStage7:
        with np.load(Path(path), allow_pickle=False) as d:
            on_frac_map = {str(k): float(v) for k, v in zip(d["onset_frac_keys"], d["onset_frac_vals"])}
            on_ms_map = {str(k): float(v) for k, v in zip(d["onset_frac_keys"], d["onset_ms_vals"])}
            on_uv_map = (
                {str(k): float(v) for k, v in zip(d["onset_uv_keys"], d["onset_uv_vals"])}
                if "onset_uv_keys" in d
                else {}
            )
            c_dim = int(d["c_dim"][0]) if "c_dim" in d else N_COARSE_CEP_S7
            is_te = bool(int(d["is_trueenv"][0])) if "is_trueenv" in d else False
            return cls(
                on_of=_load_medoid_table(d, "on_of"),
                on_grp=_load_medoid_table(d, "on_grp"),
                on_only=_load_medoid_table(d, "on_only"),
                on_global=(
                    d["on_glob_cep"].astype(np.float64),
                    d["on_glob_bap"].astype(np.float64),
                    d["on_glob_vp"].astype(np.float64),
                ),
                fin_oft=_load_medoid_table(d, "fin_oft"),
                fin_pft=_load_medoid_table(d, "fin_pft"),
                fin_ft=_load_medoid_table(d, "fin_ft"),
                fin_f=_load_medoid_table(d, "fin_f"),
                fin_global=(
                    d["fin_glob_cep"].astype(np.float64),
                    d["fin_glob_bap"].astype(np.float64),
                    d["fin_glob_vp"].astype(np.float64),
                ),
                w_ft=_load_weight_dict(d, "w_ft"),
                w_pft=_load_weight_dict(d, "w_pft"),
                w_oft=_load_weight_dict(d, "w_oft"),
                d_oft=_load_weight_dict(d, "d_oft"),
                w_on_o=_load_weight_dict(d, "w_on_o"),
                w_on_og=_load_weight_dict(d, "w_on_og"),
                w_on_of=_load_weight_dict(d, "w_on_of"),
                d_on_of=_load_weight_dict(d, "d_on_of"),
                w_bap_ft=_load_weight_dict(d, "w_bap_ft"),
                f_end_proto=_load_proto_dict(d, "f_end_proto"),
                f_start_proto=_load_proto_dict(d, "f_start_proto"),
                o_start_proto=_load_proto_dict(d, "o_start_proto"),
                onset_frac=on_frac_map,
                onset_ms=on_ms_map,
                onset_uv_frac=on_uv_map,
                sil_cep64=d["sil_cep64"].astype(np.float64),
                sil_bap6=d["sil_bap6"].astype(np.float64),
                speaker_f0_med_hz=float(d["speaker_f0_med_hz"][0]),
                ref_log_amp=float(d["ref_log_amp"][0]),
                n_train_utts=int(d["n_train_utts"][0]),
                c_dim=c_dim,
                is_trueenv=is_te,
                fin_gv_oft=_load_gv_table(d, "fin_gv_oft"),
                fin_gv_glob=d["fin_gv_glob"].astype(np.float64) if "fin_gv_glob" in d else np.ones(40),
            )

    def predict_final_ridge(
        self, onset: str, final: str, tone: int, x_feat: np.ndarray,
    ) -> np.ndarray | None:
        k_oft = f"{onset}|{final}|{tone}"
        w_exact = self.w_oft.get(k_oft)
        if w_exact is not None:
            return (x_feat @ w_exact).reshape(N_FINAL_BINS_S2, self.c_dim)

        place = classify_onset_place(onset)
        w_parent = self.w_pft.get(f"{place}|{final}|{tone}", self.w_ft.get(f"{final}|{tone}"))
        if w_parent is None:
            return None
        pred = (x_feat @ w_parent).reshape(N_FINAL_BINS_S2, self.c_dim)
        d_rare = self.d_oft.get(k_oft)
        if d_rare is not None:
            pred = pred + d_rare
        return pred

    def predict_onset_ridge(
        self, onset: str, final: str, x_feat: np.ndarray,
    ) -> np.ndarray | None:
        k_of = f"{onset}|{final}"
        w_exact = self.w_on_of.get(k_of)
        if w_exact is not None:
            return (x_feat @ w_exact).reshape(N_ONSET_BINS_S2, self.c_dim)

        fg = classify_final_group(final)
        w_parent = self.w_on_og.get(f"{onset}|{fg}", self.w_on_o.get(onset))
        if w_parent is None:
            return None
        pred = (x_feat @ w_parent).reshape(N_ONSET_BINS_S2, self.c_dim)
        d_rare = self.d_on_of.get(k_of)
        if d_rare is not None:
            pred = pred + d_rare
        return pred

    def predict_final_bap(
        self, final: str, tone: int, x_feat: np.ndarray,
    ) -> np.ndarray | None:
        w = self.w_bap_ft.get(f"{final}|{tone}")
        if w is None:
            return None
        return np.clip((x_feat @ w).reshape(N_FINAL_BINS_S2, N_BAP), 0.004, 0.995)

    def lookup_final_gv(self, onset: str, final: str, tone: int) -> np.ndarray:
        return self.fin_gv_oft.get(f"{onset}|{final}|{tone}", self.fin_gv_glob)

    def lookup_onset_candidates(self, onset: str, final: str) -> tuple[MedoidGroup | None, str]:
        g = self.on_of.get(f"{onset}|{final}")
        if g is not None and g.total_tokens >= 3:
            return g, "on_of"
        fg = classify_final_group(final)
        g = self.on_grp.get(f"{onset}|{fg}")
        if g is not None and g.total_tokens >= 3:
            return g, "on_grp"
        g = self.on_only.get(onset)
        if g is not None and g.total_tokens >= 1:
            return g, "on_only"
        return None, "on_fallback"

    def lookup_final_candidates(
        self,
        onset: str,
        final: str,
        tone: int,
        *,
        min_exact_tokens: int = 3,
        min_place_tokens: int = 3,
    ) -> tuple[MedoidGroup | None, str]:
        g = self.fin_oft.get(f"{onset}|{final}|{tone}")
        if g is not None and g.total_tokens >= min_exact_tokens:
            return g, "exact_oft"
        place = classify_onset_place(onset)
        g = self.fin_pft.get(f"{place}|{final}|{tone}")
        if g is not None and g.total_tokens >= min_place_tokens:
            return g, "place_ft"
        g = self.fin_ft.get(f"{final}|{tone}")
        if g is not None and g.total_tokens >= 1:
            return g, "ft"
        g = self.fin_f.get(final)
        if g is not None and g.total_tokens >= 1:
            return g, "f"
        return None, "fallback"


def _score_medoid_ridge_guided(
    grp: MedoidGroup,
    pred_cep: np.ndarray | None,
    prev_fg_id: int,
    next_pl_id: int,
    pos_id: int,
    dur_ms: float,
    f0_st: float,
    *,
    w_ridge: float = 0.85,
) -> np.ndarray:
    """Score K unblurred medoids by Ridge acoustic proximity + prosody/context cues."""
    k = grp.cep.shape[0]
    costs = (
        np.where(grp.ctx_prev == prev_fg_id, 0.0, 0.16)
        + np.where(grp.ctx_next == next_pl_id, 0.0, 0.14)
        + np.where(grp.ctx_pos == pos_id, 0.0, 0.12)
        + 0.25 * np.abs(np.log(np.maximum(10.0, dur_ms) / np.maximum(10.0, grp.ctx_dur_ms)))
        + 0.025 * np.abs(f0_st - grp.ctx_f0_st)
        + 0.08 * (1.0 - grp.cluster_sz.astype(np.float64) / float(max(1, int(np.max(grp.cluster_sz)))))
    )
    if pred_cep is not None:
        d_spec = np.sqrt(np.mean((grp.cep[:, :, 1:24] - pred_cep[None, :, 1:24]) ** 2, axis=(1, 2)))
        d_c0 = np.mean(np.abs(grp.cep[:, :, 0] - pred_cep[None, :, 0]), axis=1)
        costs += w_ridge * (d_spec + 0.15 * d_c0)
    return costs


def _smooth_voiced_seams(cep: np.ndarray, syl_meta: list[tuple[Any, ...]]) -> None:
    """Apply bounded 2-frame raised-cosine continuity bridge across non-paused voiced seams."""
    n_fr = cep.shape[0]
    for i in range(1, len(syl_meta)):
        a = int(syl_meta[i][0])
        prev_b = int(syl_meta[i - 1][2])
        is_p_sil = int(syl_meta[i][3]) == PREV_FG_TO_ID["SIL"]
        cur_obs = bool(syl_meta[i][11])
        if is_p_sil or cur_obs or abs(a - prev_b) > 2 or a < 2 or a + 2 >= n_fr:
            continue
        # Gently bridge cep[1:18] over [a-2, a+1] so vowel-to-vowel/sonorant seams have zero step clicks
        c_left = cep[a - 2, 1:18].copy()
        c_right = cep[a + 1, 1:18].copy()
        d_max = float(np.sqrt(np.mean((c_right - c_left) ** 2)))
        if d_max < 0.05:
            continue
        cep[a - 1, 1:18] = 0.65 * cep[a - 1, 1:18] + 0.35 * (0.67 * c_left + 0.33 * c_right)
        cep[a, 1:18] = 0.65 * cep[a, 1:18] + 0.35 * (0.33 * c_left + 0.67 * c_right)


def build_stage7_control(
    ctx: UtteranceContext,
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    knowledge: AcousticKnowledgeStage7,
    *,
    syl_gain_db: np.ndarray | None = None,
    taper_vec: np.ndarray | None = None,
    contrast_bell_peak: float = 0.16,
    gv_strength: float = 0.15,
    bap_w: float = 0.85,
    bap_ridge_w: float = 0.35,
    seam_polish: bool = True,
) -> tuple[LarynxControl, dict[str, Any]]:
    """Build Stage 7 frame-level LarynxControl using Hierarchical Ridge + Ridge-Guided Medoids."""
    n_fr = ctx.n_frames
    f0_arr = np.asarray(f0_hz, dtype=np.float64)
    voi_arr = np.asarray(voiced, dtype=bool)

    cep = np.repeat(knowledge.sil_cep64[None, :], n_fr, axis=0)
    bap = np.repeat(knowledge.sil_bap6[None, :], n_fr, axis=0)

    hit_counts: dict[str, int] = {
        "exact_oft": 0,
        "place_ft": 0,
        "ft": 0,
        "f": 0,
        "fallback": 0,
        "on_of": 0,
        "on_grp": 0,
        "on_only": 0,
        "on_fallback": 0,
        "ridge_oft": 0,
        "ridge_parent": 0,
    }

    syls = ctx.syllables
    n_s = len(syls)
    if n_s == 0:
        log_env = cep_to_log_env(cep, n_bins=N_BINS) + knowledge.ref_log_amp
        ctrl = LarynxControl(
            f0_hz=f0_arr,
            voiced=voi_arr,
            log_env=log_env,
            bap=np.clip(bap, 0.004, 0.995),
            n_samples=n_fr * HOP,
        )
        return ctrl, {"mode": "stage7", "hits": hit_counts}

    ref_f0 = max(50.0, float(knowledge.speaker_f0_med_hz))
    fin_cands: list[MedoidGroup | None] = []
    on_cands: list[MedoidGroup | None] = []
    pred_fins: list[np.ndarray | None] = []
    pred_ons: list[np.ndarray | None] = []
    pred_baps: list[np.ndarray | None] = []
    syl_meta: list[tuple[Any, ...]] = []

    for i, s in enumerate(syls):
        a = max(0, min(n_fr, int(s.start_fr)))
        b = max(a, min(n_fr, int(s.end_fr)))
        sp = max(a, min(b, int(s.split_fr)))
        o, f = normalize_of(s.onset, s.final)
        t = int(s.tone)

        gap_prev = (a - int(syls[i - 1].end_fr)) if i > 0 else 999
        pause_prev = float(syls[i - 1].pause_after_ms) if i > 0 else 999.0
        is_p_sil = i == 0 or gap_prev >= 3 or pause_prev >= 25.0
        if is_p_sil:
            prev_o, prev_f, prev_t = "SIL", "SIL", 0
            prev_fg = "SIL"
        else:
            prev_o, prev_f = normalize_of(syls[i - 1].onset, syls[i - 1].final)
            prev_t = int(syls[i - 1].tone)
            prev_fg = classify_final_group(prev_f)

        gap_next = (int(syls[i + 1].start_fr) - b) if i < n_s - 1 else 999
        pause_cur = float(s.pause_after_ms) if i < n_s - 1 else 999.0
        is_n_sil = i == n_s - 1 or gap_next >= 3 or pause_cur >= 25.0
        if is_n_sil:
            next_o, next_f, next_t = "SIL", "SIL", 0
            next_pl = "SIL"
        else:
            next_o, next_f = normalize_of(syls[i + 1].onset, syls[i + 1].final)
            next_t = int(syls[i + 1].tone)
            next_pl = classify_onset_place(next_o)

        pos_id = 0 if prev_fg == "SIL" else (2 if next_pl == "SIL" else 1)
        dur_ms = float(max(1, b - a)) * 5.0
        on_ms = float(max(0, sp - a)) * 5.0 if (o and sp > a) else 0.0
        v_sub = voi_arr[a:b] & (f0_arr[a:b] > 50.0)
        f0_st = float(12.0 * np.log2(np.median(f0_arr[a:b][v_sub]) / ref_f0)) if np.any(v_sub) else 0.0

        fa = sp if (o and sp > a) else a
        vf = np.where(voi_arr[fa:b] & (f0_arr[fa:b] > 50.0))[0] if b > fa else np.zeros(0, dtype=np.intp)
        if len(vf) >= 2:
            uv = vf.astype(np.float64) / float(max(1, (b - fa) - 1))
            st_v = 12.0 * np.log2(f0_arr[fa:b][vf] / ref_f0)
            p3 = np.interp([0.20, 0.50, 0.80], uv, st_v)
            f3_st = (float(p3[0]), float(p3[1]), float(p3[2]))
        elif len(vf) == 1:
            val = float(12.0 * np.log2(f0_arr[fa:b][vf[0]] / ref_f0))
            f3_st = (val, val, val)
        else:
            f3_st = (f0_st, f0_st, f0_st)

        x_feat = extract_stage7_acoustic_features(
            prev_final=prev_f,
            prev_tone=prev_t,
            next_onset=next_o,
            next_final=next_f,
            next_tone=next_t,
            pos_id=pos_id,
            dur_ms=dur_ms,
            on_ms=on_ms,
            f3_st=f3_st,
            brk=int(s.prosodic_break),
            pause_prev_ms=pause_prev if not is_p_sil else 999.0,
            pause_after_ms=pause_cur if not is_n_sil else 999.0,
            f_end_proto=knowledge.f_end_proto,
            o_start_proto=knowledge.o_start_proto,
            f_start_proto=knowledge.f_start_proto,
        )

        pf = knowledge.predict_final_ridge(o, f, t, x_feat)
        pred_fins.append(pf)
        if f"{o}|{f}|{t}" in knowledge.w_oft:
            hit_counts["ridge_oft"] += 1
        elif pf is not None:
            hit_counts["ridge_parent"] += 1

        po = knowledge.predict_onset_ridge(o, f, x_feat) if (o and sp > a) else None
        pred_ons.append(po)
        pred_baps.append(knowledge.predict_final_bap(f, t, x_feat) if bap_ridge_w > 0.0 else None)

        f_grp, f_tag = knowledge.lookup_final_candidates(o, f, t)
        if b > fa:
            hit_counts[f_tag] += 1
        fin_cands.append(f_grp)

        if o and sp > a:
            o_grp, o_tag = knowledge.lookup_onset_candidates(o, f)
            hit_counts[o_tag] += 1
        else:
            o_grp = None
        on_cands.append(o_grp)

        is_obs = classify_onset_manner(o) in OBSTRUENT_MANNERS
        syl_meta.append((
            a, sp, b,
            encode_prev_fg(prev_fg),
            encode_next_place(next_pl),
            pos_id,
            dur_ms, on_ms, f0_st,
            o, f, is_obs,
        ))

    # Step 1: Ridge-Guided Viterbi DP over K<=12 unblurred final medoids
    unary_costs: list[np.ndarray] = []
    fin_cep_list: list[np.ndarray] = []
    fin_bap_list: list[np.ndarray] = []
    for i in range(n_s):
        _, _, _, prev_id, next_id, pos_id, dur_ms, _, f0_st, _, _, _ = syl_meta[i]
        grp = fin_cands[i]
        if grp is None:
            unary_costs.append(np.zeros(1, dtype=np.float64))
            fin_cep_list.append(knowledge.fin_global[0][None, :, :])
            fin_bap_list.append(knowledge.fin_global[1][None, :, :])
        else:
            unary_costs.append(
                _score_medoid_ridge_guided(
                    grp, pred_fins[i], prev_id, next_id, pos_id, dur_ms, f0_st
                )
            )
            fin_cep_list.append(grp.cep)
            fin_bap_list.append(grp.bap)

    dp: list[np.ndarray] = [unary_costs[0].copy()]
    back: list[np.ndarray] = []
    for i in range(1, n_s):
        k_prev = len(unary_costs[i - 1])
        k_cur = len(unary_costs[i])
        _, _, _, prev_id, _, _, _, _, _, cur_o, _, cur_obs = syl_meta[i]
        prev_c_end = fin_cep_list[i - 1][:, -1, 1:24]
        prev_e_end = fin_cep_list[i - 1][:, -1, :1]

        no_pause = prev_id != PREV_FG_TO_ID["SIL"]
        if not no_pause or cur_obs:
            trans = np.zeros((k_prev, k_cur), dtype=np.float64)
        else:
            if cur_o and on_cands[i] is not None:
                o_c0 = on_cands[i].cep[:, 0, 1:24]
                o_e0 = on_cands[i].cep[:, 0, :1]
                d_mat = np.sqrt(np.mean((prev_c_end[:, None, :] - o_c0[None, :, :]) ** 2, axis=2))
                d_e = 0.20 * np.abs(prev_e_end - o_e0.T)
                min_d = np.min(d_mat + d_e, axis=1)[:, None]
                trans = 0.22 * np.repeat(min_d, k_cur, axis=1)
            else:
                cur_c_start = fin_cep_list[i][:, 0, 1:24]
                cur_e_start = fin_cep_list[i][:, 0, :1]
                d_mat = np.sqrt(np.mean((prev_c_end[:, None, :] - cur_c_start[None, :, :]) ** 2, axis=2))
                d_e = 0.20 * np.abs(prev_e_end - cur_e_start.T)
                trans = 0.44 * (d_mat + d_e)

        total = dp[i - 1][:, None] + trans + unary_costs[i][None, :]
        bp = np.argmin(total, axis=0)
        dp.append(total[bp, np.arange(k_cur)])
        back.append(bp)

    chosen_fin = [0] * n_s
    chosen_fin[-1] = int(np.argmin(dp[-1]))
    for i in range(n_s - 2, -1, -1):
        chosen_fin[i] = int(back[i][chosen_fin[i + 1]])

    # Step 2: Sharpness-preserving Tapered Ridge fusion + cubic interpolation + GV & bell
    t_vec = DEFAULT_TAPER_28 if taper_vec is None else np.asarray(taper_vec, dtype=np.float64)
    c_dim = min(int(knowledge.c_dim), len(t_vec))
    bell_pk = float(np.clip(contrast_bell_peak, 0.0, 0.40))
    bell_vec = (1.0 + bell_pk * _BELL_COS_WEIGHT)[None, :] if bell_pk > 0.0 else None
    gv_s = float(np.clip(gv_strength, 0.0, 1.0))
    b_w = float(np.clip(bap_w, 0.0, 1.0))
    br_w = float(np.clip(bap_ridge_w, 0.0, 1.0))

    for i in range(n_s):
        a, sp, b, prev_id, next_id, pos_id, dur_ms, on_ms, f0_st, o, f, is_obs = syl_meta[i]
        if b <= a:
            continue
        t = int(syls[i].tone)

        f_idx = chosen_fin[i]
        f_cep = fin_cep_list[i][f_idx].copy()
        f_bap = fin_bap_list[i][f_idx].copy()

        pf = pred_fins[i]
        if pf is not None:
            av = t_vec[:c_dim][None, :]
            f_cep[:, :c_dim] = av * pf[:, :c_dim] + (1.0 - av) * f_cep[:, :c_dim]

        pb = pred_baps[i]
        if pb is not None and br_w > 0.0:
            f_bap = br_w * pb + (1.0 - br_w) * f_bap

        if o and sp > a:
            o_grp = on_cands[i]
            if o_grp is None:
                o_cep = knowledge.on_global[0].copy()
                o_bap = knowledge.on_global[1].copy()
            elif o_grp.cep.shape[0] == 1:
                o_cep = o_grp.cep[0].copy()
                o_bap = o_grp.bap[0].copy()
            else:
                o_costs = _score_medoid_ridge_guided(
                    o_grp, pred_ons[i], prev_id, next_id, pos_id, on_ms, f0_st, w_ridge=0.80
                )
                w_fin = 0.20 if is_obs else 0.45
                o_costs += w_fin * np.sqrt(np.mean((o_grp.cep[:, -1, 1:24] - f_cep[None, 0, 1:24]) ** 2, axis=1))
                if i > 0 and prev_id != PREV_FG_TO_ID["SIL"] and not is_obs:
                    prev_end = fin_cep_list[i - 1][chosen_fin[i - 1], -1]
                    o_costs += 0.35 * np.sqrt(np.mean((o_grp.cep[:, 0, 1:24] - prev_end[None, 1:24]) ** 2, axis=1))
                best_o = int(np.argmin(o_costs))
                o_cep = o_grp.cep[best_o].copy()
                o_bap = o_grp.bap[best_o].copy()

            po = pred_ons[i]
            if po is not None:
                av = t_vec[:c_dim][None, :]
                o_cep[:, :c_dim] = av * po[:, :c_dim] + (1.0 - av) * o_cep[:, :c_dim]

            u_on = ctx.phase_u[a:sp]
            cep[a:sp, :24] = _interp_phase_bins_cubic(o_cep[:, :24], u_on, _ONSET_CENTERS_S7)
            cep[a:sp, 24:] = _interp_phase_bins(o_cep[:, 24:], u_on, _ONSET_CENTERS_S7)
            bap[a:sp] = _interp_phase_bins(o_bap, u_on, _ONSET_CENTERS_S7)
            f_a = sp
        else:
            f_a = a

        if b > f_a:
            u_fi = ctx.phase_u[f_a:b]
            cep[f_a:b, :24] = _interp_phase_bins_cubic(f_cep[:, :24], u_fi, _FINAL_CENTERS_S7)
            cep[f_a:b, 24:] = _interp_phase_bins(f_cep[:, 24:], u_fi, _FINAL_CENTERS_S7)
            bap[f_a:b] = _interp_phase_bins(f_bap, u_fi, _FINAL_CENTERS_S7)

            if bell_vec is not None:
                cep[f_a:b, _BELL_Q_START:_BELL_Q_END] *= bell_vec

            if gv_s > 0.0 and (b - f_a) >= 4:
                seg_sub = cep[f_a:b, _GV_Q_START:_GV_Q_END]
                mu = np.mean(seg_sub, axis=0, keepdims=True)
                cur_std = np.std(seg_sub, axis=0)
                tgt_std = knowledge.lookup_final_gv(o, f, t)[_GV_Q_START:_GV_Q_END]
                ratio = np.clip(tgt_std / np.maximum(cur_std, 0.15), 0.90, 1.35)
                scale_q = 1.0 + gv_s * _GV_BELL_W * (ratio - 1.0)
                cep[f_a:b, _GV_Q_START:_GV_Q_END] = mu + (seg_sub - mu) * scale_q[None, :]

        # HNR-calibrated BAP mapping (only when knowledge was extracted from raw v9 scale)
        if not knowledge.is_trueenv and b_w > 0.0 and b > a:
            v_seg = voi_arr[a:b] & (f0_arr[a:b] > 50.0)
            if np.any(v_seg):
                tgt = _COMB_BAP_TP.get((t, pos_id), _GLOBAL_COMB_BAP)
                ratio_hi = tgt[1:] / np.maximum(0.05, _V9_MED_BAP[1:])
                mapped_hi = np.clip(bap[a:b, 1:][v_seg] * ratio_hi[None, :], 0.04, 0.94)
                cal_hi = 0.60 * mapped_hi + 0.40 * tgt[1:][None, :]
                bap[a:b, 1:][v_seg] = (1.0 - b_w) * bap[a:b, 1:][v_seg] + b_w * cal_hi

    if seam_polish:
        _smooth_voiced_seams(cep, syl_meta)

    log_env = cep_to_log_env(cep, n_bins=N_BINS) + knowledge.ref_log_amp

    if syl_gain_db is not None and len(syl_gain_db) == n_s:
        for idx_s, s in enumerate(syls):
            a = max(0, min(n_fr, int(s.start_fr)))
            b = max(a, min(n_fr, int(s.end_fr)))
            if b > a:
                g_nep = float(np.clip(syl_gain_db[idx_s], -6.0, 6.0)) / NEPER_TO_DB
                log_env[a:b] += 0.55 * g_nep

    ctrl = LarynxControl(
        f0_hz=f0_arr,
        voiced=voi_arr,
        log_env=log_env,
        bap=np.clip(bap, 0.004, 0.995),
        n_samples=n_fr * HOP,
    )
    return ctrl, {"mode": "stage7", "hits": hit_counts, "chosen_fin_medoids": chosen_fin}
