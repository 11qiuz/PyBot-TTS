"""Stage 2 unblurred medoid acoustic knowledge loader & context-aware Viterbi trajectory builder."""
from __future__ import annotations

from dataclasses import dataclass, field
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
from exp_20_pybot_larynx.acoustics.knowledge import (
    _interp_phase_bins,
    normalize_of,
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

N_ONSET_BINS_S2 = 5
N_FINAL_BINS_S2 = 9
N_COARSE_CEP_S2 = 24
STAGE2_KNOWLEDGE_PATH = EXP_DIR / "acoustics" / "knowledge_stage2.npz"
STAGE2_SUMMARY_PATH = EXP_DIR / "acoustics" / "knowledge_stage2_summary.json"

_ONSET_CENTERS_S2 = (np.arange(N_ONSET_BINS_S2, dtype=np.float64) + 0.5) / float(N_ONSET_BINS_S2)
_FINAL_CENTERS_S2 = (np.arange(N_FINAL_BINS_S2, dtype=np.float64) + 0.5) / float(N_FINAL_BINS_S2)

# Tapered low-quefrency anchoring weights over cep[0:24] (preserving high-quefrency medoid detail)
_DEFAULT_TAPER_24 = np.array(
    [0.90] * 12 + [0.65] * 5 + [0.35] * 7,
    dtype=np.float64,
)
_BELL_Q_START = 10
_BELL_Q_END = 26
_BELL_COS_WEIGHT = 0.5 * (
    1.0 - np.cos(2.0 * np.pi * (np.arange(_BELL_Q_START, _BELL_Q_END, dtype=np.float64) - 9.0) / 17.0)
)

# Stage 4 voiced BAP targets aligned to Real 1-7 kHz band HNR (~14.5 / 9.5 / 7.2 dB)
_COMB_BAP_TP: dict[tuple[int, int], np.ndarray] = {
    (1, 0): np.array([0.025, 0.14, 0.30, 0.58, 0.76, 0.85], dtype=np.float64),
    (1, 1): np.array([0.025, 0.14, 0.30, 0.58, 0.76, 0.85], dtype=np.float64),
    (1, 2): np.array([0.025, 0.14, 0.30, 0.58, 0.76, 0.85], dtype=np.float64),
    (2, 0): np.array([0.050, 0.21, 0.42, 0.70, 0.85, 0.91], dtype=np.float64),
    (2, 1): np.array([0.050, 0.21, 0.42, 0.70, 0.85, 0.91], dtype=np.float64),
    (2, 2): np.array([0.050, 0.21, 0.42, 0.70, 0.85, 0.91], dtype=np.float64),
    (3, 0): np.array([0.080, 0.27, 0.48, 0.75, 0.87, 0.92], dtype=np.float64),
    (3, 1): np.array([0.080, 0.27, 0.48, 0.75, 0.87, 0.92], dtype=np.float64),
    (3, 2): np.array([0.080, 0.27, 0.48, 0.75, 0.87, 0.92], dtype=np.float64),
    (4, 0): np.array([0.032, 0.18, 0.36, 0.64, 0.80, 0.88], dtype=np.float64),
    (4, 1): np.array([0.032, 0.18, 0.36, 0.64, 0.80, 0.88], dtype=np.float64),
    (4, 2): np.array([0.032, 0.18, 0.36, 0.64, 0.80, 0.88], dtype=np.float64),
    (5, 0): np.array([0.060, 0.23, 0.44, 0.72, 0.86, 0.91], dtype=np.float64),
    (5, 1): np.array([0.060, 0.23, 0.44, 0.72, 0.86, 0.91], dtype=np.float64),
    (5, 2): np.array([0.060, 0.23, 0.44, 0.72, 0.86, 0.91], dtype=np.float64),
}
_V9_MED_BAP = np.array([0.0200, 0.4030, 0.7090, 0.8851, 0.9632, 0.9787], dtype=np.float64)
_GLOBAL_COMB_BAP = np.array([0.050, 0.21, 0.42, 0.70, 0.85, 0.91], dtype=np.float64)


def _smooth_5pt_inner(arr: np.ndarray) -> np.ndarray:
    if len(arr) < 5:
        return arr
    p = np.pad(arr, ((2, 2), (0, 0)), mode="edge")
    return 0.10 * p[:-4] + 0.20 * p[1:-3] + 0.40 * p[2:-2] + 0.20 * p[3:-1] + 0.10 * p[4:]

PREV_FG_VOCAB: tuple[str, ...] = (
    "SIL",
    "i_front",
    "u_round",
    "v_front_round",
    "a_open_central",
)
PREV_FG_TO_ID: dict[str, int] = {k: i for i, k in enumerate(PREV_FG_VOCAB)}

NEXT_PLACE_VOCAB: tuple[str, ...] = (
    "SIL",
    "zero",
    "bilabial",
    "labiodental",
    "alveolar",
    "dental",
    "retroflex",
    "palatal",
    "velar",
)
NEXT_PLACE_TO_ID: dict[str, int] = {k: i for i, k in enumerate(NEXT_PLACE_VOCAB)}


def encode_prev_fg(fg: str) -> int:
    return PREV_FG_TO_ID.get(fg, PREV_FG_TO_ID["a_open_central"])


def encode_next_place(place: str) -> int:
    return NEXT_PLACE_TO_ID.get(place, NEXT_PLACE_TO_ID["alveolar"])


@dataclass(frozen=True)
class MedoidGroup:
    """K <= 4 unblurred real-token medoid trajectories + context signatures for one phone key."""

    cep: np.ndarray          # (K, T_bins, N_CEP) float64
    bap: np.ndarray          # (K, T_bins, N_BAP) float64
    vp: np.ndarray           # (K, T_bins) float64
    ctx_prev: np.ndarray     # (K,) int8
    ctx_next: np.ndarray     # (K,) int8
    ctx_pos: np.ndarray      # (K,) int8 (0=initial, 1=medial, 2=final)
    ctx_dur_ms: np.ndarray   # (K,) float64
    ctx_f0_st: np.ndarray    # (K,) float64
    cluster_sz: np.ndarray   # (K,) int32
    total_tokens: int


def _load_medoid_table(d: Any, prefix: str) -> dict[str, MedoidGroup]:
    keys = d[f"{prefix}_keys"]
    offsets = d[f"{prefix}_offsets"]
    tok_cnt = d[f"{prefix}_tok_cnt"]
    cep = d[f"{prefix}_cep"]
    bap = d[f"{prefix}_bap"]
    vp = d[f"{prefix}_vp"]
    c_prev = d[f"{prefix}_ctx_prev"]
    c_next = d[f"{prefix}_ctx_next"]
    c_pos = d[f"{prefix}_ctx_pos"]
    c_dur = d[f"{prefix}_ctx_dur_ms"]
    c_f0 = d[f"{prefix}_ctx_f0_st"]
    cl_sz = d[f"{prefix}_cluster_sz"]

    out: dict[str, MedoidGroup] = {}
    for i in range(len(keys)):
        a = int(offsets[i])
        b = int(offsets[i + 1])
        if b <= a:
            continue
        out[str(keys[i])] = MedoidGroup(
            cep=cep[a:b].astype(np.float64),
            bap=bap[a:b].astype(np.float64),
            vp=vp[a:b].astype(np.float64),
            ctx_prev=c_prev[a:b].astype(np.int8),
            ctx_next=c_next[a:b].astype(np.int8),
            ctx_pos=c_pos[a:b].astype(np.int8),
            ctx_dur_ms=c_dur[a:b].astype(np.float64),
            ctx_f0_st=c_f0[a:b].astype(np.float64),
            cluster_sz=cl_sz[a:b].astype(np.int32),
            total_tokens=int(tok_cnt[i]),
        )
    return out


def _load_coarse_table(d: Any, prefix: str) -> dict[str, np.ndarray]:
    k_name = f"{prefix}_keys"
    if k_name not in d:
        return {}
    keys = d[k_name]
    vals = d[f"{prefix}_cep"]
    return {str(keys[i]): vals[i].astype(np.float64) for i in range(len(keys))}


@dataclass
class AcousticKnowledgeStage2:
    """Unblurred K<=4 medoid acoustic knowledge base at 5-bin onset / 9-bin final resolution."""

    on_grp: dict[str, MedoidGroup]
    on_only: dict[str, MedoidGroup]
    on_global: tuple[np.ndarray, np.ndarray, np.ndarray]

    fin_oft: dict[str, MedoidGroup]
    fin_pft: dict[str, MedoidGroup]
    fin_ft: dict[str, MedoidGroup]
    fin_f: dict[str, MedoidGroup]
    fin_global: tuple[np.ndarray, np.ndarray, np.ndarray]

    onset_frac: dict[str, float]
    onset_ms: dict[str, float]
    sil_cep64: np.ndarray
    sil_bap6: np.ndarray
    speaker_f0_med_hz: float
    ref_log_amp: float
    n_train_utts: int

    # Optional context-conditioned low-quefrency coarse anchors (cep[:, :N_COARSE_CEP_S2])
    on_ctx_p: dict[str, np.ndarray] = field(default_factory=dict)
    on_ctx_base: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_pn: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_np: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_pp: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_pos: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_pft_pos: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str = STAGE2_KNOWLEDGE_PATH) -> AcousticKnowledgeStage2:
        with np.load(Path(path), allow_pickle=False) as d:
            on_frac_map = {str(k): float(v) for k, v in zip(d["onset_frac_keys"], d["onset_frac_vals"])}
            on_ms_map = {str(k): float(v) for k, v in zip(d["onset_frac_keys"], d["onset_ms_vals"])}
            return cls(
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
                onset_frac=on_frac_map,
                onset_ms=on_ms_map,
                sil_cep64=d["sil_cep64"].astype(np.float64),
                sil_bap6=d["sil_bap6"].astype(np.float64),
                speaker_f0_med_hz=float(d["speaker_f0_med_hz"][0]),
                ref_log_amp=float(d["ref_log_amp"][0]),
                n_train_utts=int(d["n_train_utts"][0]),
                on_ctx_p=_load_coarse_table(d, "on_ctx_p"),
                on_ctx_base=_load_coarse_table(d, "on_ctx_base"),
                fin_ctx_pn=_load_coarse_table(d, "fin_ctx_pn"),
                fin_ctx_np=_load_coarse_table(d, "fin_ctx_np"),
                fin_ctx_pp=_load_coarse_table(d, "fin_ctx_pp"),
                fin_ctx_pos=_load_coarse_table(d, "fin_ctx_pos"),
                fin_ctx_pft_pos=_load_coarse_table(d, "fin_ctx_pft_pos"),
            )

    def lookup_onset_coarse(self, onset: str, final: str, prev_fg_id: int) -> np.ndarray | None:
        fg = classify_final_group(final)
        c = self.on_ctx_p.get(f"{onset}|{fg}|{prev_fg_id}")
        if c is not None:
            return c
        return self.on_ctx_base.get(f"{onset}|{fg}")

    def lookup_final_coarse(
        self,
        onset: str,
        final: str,
        tone: int,
        prev_fg_id: int,
        next_pl_id: int,
        pos_id: int,
    ) -> np.ndarray | None:
        k_oft = f"{onset}|{final}|{tone}"
        c = self.fin_ctx_pn.get(f"{k_oft}|{prev_fg_id}|{next_pl_id}")
        if c is not None:
            return c
        c = self.fin_ctx_np.get(f"{k_oft}|{next_pl_id}|{pos_id}")
        if c is not None:
            return c
        c = self.fin_ctx_pp.get(f"{k_oft}|{prev_fg_id}|{pos_id}")
        if c is not None:
            return c
        c = self.fin_ctx_pos.get(f"{k_oft}|{pos_id}")
        if c is not None:
            return c
        place = classify_onset_place(onset)
        return self.fin_ctx_pft_pos.get(f"{place}|{final}|{tone}|{pos_id}")

    def lookup_onset_candidates(self, onset: str, final: str) -> tuple[MedoidGroup | None, str]:
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


def _score_medoid_unary(
    grp: MedoidGroup,
    prev_fg_id: int,
    next_pl_id: int,
    pos_id: int,
    dur_ms: float,
    f0_st: float,
) -> np.ndarray:
    """Compute context mismatch + cluster-prior cost `(K,)` (lower is better)."""
    k = grp.cep.shape[0]
    costs = np.zeros(k, dtype=np.float64)

    # Context signature costs
    costs += np.where(grp.ctx_prev == prev_fg_id, 0.0, 0.45)
    costs += np.where(grp.ctx_next == next_pl_id, 0.0, 0.40)
    costs += np.where(grp.ctx_pos == pos_id, 0.0, 0.35)

    # Duration & F0 register similarity
    dur_ratio = np.abs(np.log(np.maximum(10.0, dur_ms) / np.maximum(10.0, grp.ctx_dur_ms)))
    costs += 0.65 * dur_ratio
    costs += 0.06 * np.abs(f0_st - grp.ctx_f0_st)

    # Mild prior favoring well-supported cluster medoids over tiny sub-clusters
    max_sz = float(max(1, int(np.max(grp.cluster_sz))))
    costs += 0.15 * (1.0 - grp.cluster_sz.astype(np.float64) / max_sz)
    return costs


def _select_best_onset(
    grp: MedoidGroup | None,
    fallback: tuple[np.ndarray, np.ndarray, np.ndarray],
    prev_fg_id: int,
    next_pl_id: int,
    pos_id: int,
    on_dur_ms: float,
    f0_st: float,
    fin_start_cep: np.ndarray,
    prev_end_cep: np.ndarray | None,
    is_obstruent: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Select single best onset medoid matching context + onset-to-final continuity."""
    if grp is None:
        return fallback[0], fallback[1]
    k = grp.cep.shape[0]
    if k == 1:
        return grp.cep[0], grp.bap[0]

    costs = _score_medoid_unary(grp, prev_fg_id, next_pl_id, pos_id, on_dur_ms, f0_st)
    # Match onset end into selected final start (critical for sonorants & CV transitions)
    w_fin = 0.25 if is_obstruent else 0.55
    d_fin = np.sqrt(np.mean((grp.cep[:, -1, 1:24] - fin_start_cep[None, 1:24]) ** 2, axis=1))
    costs += w_fin * d_fin

    if prev_end_cep is not None and not is_obstruent:
        d_prev = np.sqrt(np.mean((grp.cep[:, 0, 1:24] - prev_end_cep[None, 1:24]) ** 2, axis=1))
        costs += 0.45 * d_prev

    best = int(np.argmin(costs))
    return grp.cep[best], grp.bap[best]


def build_stage2_control(
    ctx: UtteranceContext,
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    knowledge: AcousticKnowledgeStage2,
    *,
    mode: str = "stage2",
    syl_gain_db: np.ndarray | None = None,
    coarse_cutoff: int = 24,
    coarse_alpha: float = 0.90,
    clean_bap_weight: float = 0.85,
    smooth_inner_cep: bool = False,
    contrast_bell_peak: float = 0.12,
) -> tuple[LarynxControl, dict[str, Any]]:
    """Build Stage 2 frame-level LarynxControl using unblurred medoids + Viterbi continuity."""
    if mode not in ("stage2", "dynamic"):
        raise ValueError(f"Unsupported Stage 2 mode: {mode!r}")

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
        "on_grp": 0,
        "on_only": 0,
        "on_fallback": 0,
    }

    syls = ctx.syllables
    n_s = len(syls)
    if n_s == 0:
        log_env = cep_to_log_env(cep, n_bins=N_BINS) + knowledge.ref_log_amp
        ctrl = LarynxControl(
            f0_hz=f0_arr,
            voiced=voi_arr,
            log_env=log_env,
            bap=np.clip(bap, 0.005, 0.995),
            n_samples=n_fr * HOP,
        )
        return ctrl, {"mode": "stage2", "hits": hit_counts}

    ref_f0 = max(50.0, float(knowledge.speaker_f0_med_hz))

    # Step 1: Collect context + candidate medoids per syllable
    fin_cands: list[MedoidGroup | None] = []
    on_cands: list[MedoidGroup | None] = []
    syl_meta: list[tuple[int, int, int, int, int, int, float, float, float, str, str, bool]] = []

    for i, s in enumerate(syls):
        a = max(0, min(n_fr, int(s.start_fr)))
        b = max(a, min(n_fr, int(s.end_fr)))
        sp = max(a, min(b, int(s.split_fr)))
        o, f = normalize_of(s.onset, s.final)
        t = int(s.tone)

        gap_prev = (a - int(syls[i - 1].end_fr)) if i > 0 else 999
        pause_prev = float(syls[i - 1].pause_after_ms) if i > 0 else 999.0
        if i == 0 or gap_prev >= 3 or pause_prev >= 25.0:
            prev_fg = "SIL"
        else:
            _, prev_f_norm = normalize_of(syls[i - 1].onset, syls[i - 1].final)
            prev_fg = classify_final_group(prev_f_norm)

        gap_next = (int(syls[i + 1].start_fr) - b) if i < n_s - 1 else 999
        pause_cur = float(s.pause_after_ms) if i < n_s - 1 else 999.0
        if i == n_s - 1 or gap_next >= 3 or pause_cur >= 25.0:
            next_pl = "SIL"
        else:
            next_o_norm, _ = normalize_of(syls[i + 1].onset, syls[i + 1].final)
            next_pl = classify_onset_place(next_o_norm)

        if prev_fg == "SIL":
            pos_id = 0
        elif next_pl == "SIL":
            pos_id = 2
        else:
            pos_id = 1

        dur_ms = float(max(1, b - a)) * 5.0
        on_ms = float(max(1, sp - a)) * 5.0
        v_sub = voi_arr[a:b] & (f0_arr[a:b] > 50.0)
        f0_st = float(12.0 * np.log2(np.median(f0_arr[a:b][v_sub]) / ref_f0)) if np.any(v_sub) else 0.0

        f_grp, f_tag = knowledge.lookup_final_candidates(o, f, t)
        if b > (sp if (o and sp > a) else a):
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

    # Step 2: Viterbi DP across syllables to pick globally context-matched & edge-continuous final medoids
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
            unary_costs.append(_score_medoid_unary(grp, prev_id, next_id, pos_id, dur_ms, f0_st))
            fin_cep_list.append(grp.cep)
            fin_bap_list.append(grp.bap)

    dp: list[np.ndarray] = [unary_costs[0].copy()]
    back: list[np.ndarray] = []

    for i in range(1, n_s):
        k_prev = len(unary_costs[i - 1])
        k_cur = len(unary_costs[i])
        _, _, _, prev_id, _, _, _, _, _, cur_o, _, cur_obs = syl_meta[i]
        prev_c_end = fin_cep_list[i - 1][:, -1, 1:24]  # (k_prev, 23)

        # Edge continuity weight: strong on zero/sonorant onsets across short gaps, zero across pauses/obstruents
        no_pause = (prev_id != PREV_FG_TO_ID["SIL"])
        if not no_pause or cur_obs:
            trans = np.zeros((k_prev, k_cur), dtype=np.float64)
        else:
            if cur_o and on_cands[i] is not None:
                # Approximate onset entry using best matching onset candidate
                o_c0 = on_cands[i].cep[:, 0, 1:24]  # (ko, 23)
                d_mat = np.sqrt(np.mean((prev_c_end[:, None, :] - o_c0[None, :, :]) ** 2, axis=2))
                min_d = np.min(d_mat, axis=1)[:, None]
                trans = 0.25 * np.repeat(min_d, k_cur, axis=1)
            else:
                cur_c_start = fin_cep_list[i][:, 0, 1:24]  # (k_cur, 23)
                d_mat = np.sqrt(np.mean((prev_c_end[:, None, :] - cur_c_start[None, :, :]) ** 2, axis=2))
                trans = 0.50 * d_mat

        total = dp[i - 1][:, None] + trans + unary_costs[i][None, :]
        bp = np.argmin(total, axis=0)
        dp.append(total[bp, np.arange(k_cur)])
        back.append(bp)

    chosen_fin = [0] * n_s
    chosen_fin[-1] = int(np.argmin(dp[-1]))
    for i in range(n_s - 2, -1, -1):
        chosen_fin[i] = int(back[i][chosen_fin[i + 1]])

    # Step 3: Render each syllable using selected unblurred medoid trajectories + tapered low-quefrency context anchor
    cut = max(0, min(N_COARSE_CEP_S2, int(coarse_cutoff)))
    alpha = float(np.clip(coarse_alpha, 0.0, 1.0))
    bap_w = float(np.clip(clean_bap_weight, 0.0, 1.0))
    bell_pk = float(np.clip(contrast_bell_peak, 0.0, 0.40))
    bell_vec = (1.0 + bell_pk * _BELL_COS_WEIGHT)[None, :] if bell_pk > 0.0 else None
    if cut > 0 and alpha > 0.0:
        alpha_vec = (_DEFAULT_TAPER_24[:cut] * (alpha / 0.90)).clip(0.0, 1.0)
    else:
        alpha_vec = None

    for i in range(n_s):
        a, sp, b, prev_id, next_id, pos_id, _, on_ms, f0_st, o, f, is_obs = syl_meta[i]
        if b <= a:
            continue
        f_idx = chosen_fin[i]
        f_cep = fin_cep_list[i][f_idx].copy()
        f_bap = fin_bap_list[i][f_idx]
        if alpha_vec is not None:
            f_coarse = knowledge.lookup_final_coarse(o, f, int(syls[i].tone), prev_id, next_id, pos_id)
            if f_coarse is not None:
                c_dim = min(cut, f_coarse.shape[1])
                av = alpha_vec[:c_dim][None, :]
                f_cep[:, :c_dim] = av * f_coarse[:, :c_dim] + (1.0 - av) * f_cep[:, :c_dim]

        if o and sp > a:
            prev_end = (
                fin_cep_list[i - 1][chosen_fin[i - 1], -1]
                if (i > 0 and prev_id != PREV_FG_TO_ID["SIL"])
                else None
            )
            o_cep, o_bap = _select_best_onset(
                on_cands[i],
                knowledge.on_global,
                prev_id,
                next_id,
                pos_id,
                on_ms,
                f0_st,
                f_cep[0],
                prev_end,
                is_obs,
            )
            o_cep = o_cep.copy()
            if alpha_vec is not None:
                o_coarse = knowledge.lookup_onset_coarse(o, f, prev_id)
                if o_coarse is not None:
                    c_dim = min(cut, o_coarse.shape[1])
                    av = alpha_vec[:c_dim][None, :]
                    o_cep[:, :c_dim] = av * o_coarse[:, :c_dim] + (1.0 - av) * o_cep[:, :c_dim]
            u_on = ctx.phase_u[a:sp]
            cep[a:sp] = _interp_phase_bins(o_cep, u_on, _ONSET_CENTERS_S2)
            bap[a:sp] = _interp_phase_bins(o_bap, u_on, _ONSET_CENTERS_S2)
            f_a = sp
        else:
            f_a = a

        if b > f_a:
            u_fi = ctx.phase_u[f_a:b]
            cep[f_a:b] = _interp_phase_bins(f_cep, u_fi, _FINAL_CENTERS_S2)
            bap[f_a:b] = _interp_phase_bins(f_bap, u_fi, _FINAL_CENTERS_S2)
            if smooth_inner_cep and (b - f_a) >= 5:
                cep[f_a:b, 1:18] = _smooth_5pt_inner(cep[f_a:b, 1:18])
            if bell_vec is not None:
                cep[f_a:b, _BELL_Q_START:_BELL_Q_END] *= bell_vec

        # Calibrate voiced BAP bands 1..5 toward Stage 4 1-7 kHz HNR targets (band 0 & unvoiced untouched)
        if bap_w > 0.0 and b > a:
            v_seg = voi_arr[a:b] & (f0_arr[a:b] > 50.0)
            if np.any(v_seg):
                tgt = _COMB_BAP_TP.get((int(syls[i].tone), pos_id), _GLOBAL_COMB_BAP)
                ratio_hi = tgt[1:] / np.maximum(0.05, _V9_MED_BAP[1:])
                mapped_hi = np.clip(bap[a:b, 1:][v_seg] * ratio_hi[None, :], 0.04, 0.94)
                cal_hi = 0.60 * mapped_hi + 0.40 * tgt[1:][None, :]
                bap[a:b, 1:][v_seg] = (1.0 - bap_w) * bap[a:b, 1:][v_seg] + bap_w * cal_hi

    log_env = cep_to_log_env(cep, n_bins=N_BINS) + knowledge.ref_log_amp

    if syl_gain_db is not None and len(syl_gain_db) == n_s:
        for idx_s, s in enumerate(syls):
            a = max(0, min(n_fr, int(s.start_fr)))
            b = max(a, min(n_fr, int(s.end_fr)))
            if b > a:
                g_nep = float(np.clip(syl_gain_db[idx_s], -6.0, 6.0)) / NEPER_TO_DB
                log_env[a:b] += 0.5 * g_nep

    ctrl = LarynxControl(
        f0_hz=f0_arr,
        voiced=voi_arr,
        log_env=log_env,
        bap=np.clip(bap, 0.005, 0.995),
        n_samples=n_fr * HOP,
    )
    return ctrl, {"mode": "stage2", "hits": hit_counts, "chosen_fin_medoids": chosen_fin}
