"""Stage 5 high-resolution medoid acoustic knowledge loader & variance-preserving trajectory builder."""
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
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import (
    _BELL_COS_WEIGHT,
    _BELL_Q_END,
    _BELL_Q_START,
    _COMB_BAP_TP,
    _GLOBAL_COMB_BAP,
    _V9_MED_BAP,
    N_COARSE_CEP_S2,
    N_FINAL_BINS_S2,
    N_ONSET_BINS_S2,
    PREV_FG_TO_ID,
    MedoidGroup,
    _load_coarse_table,
    _load_medoid_table,
    encode_next_place,
    encode_prev_fg,
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

N_COARSE_CEP_S5 = 32
STAGE5_KNOWLEDGE_PATH = EXP_DIR / "acoustics" / "knowledge_stage5.npz"
STAGE5_SUMMARY_PATH = EXP_DIR / "acoustics" / "knowledge_stage5_summary.json"

_ONSET_CENTERS_S5 = (np.arange(N_ONSET_BINS_S2, dtype=np.float64) + 0.5) / float(N_ONSET_BINS_S2)
_FINAL_CENTERS_S5 = (np.arange(N_FINAL_BINS_S2, dtype=np.float64) + 0.5) / float(N_FINAL_BINS_S2)

# Tapered MMSE context+prosody anchor over cep[0:32] extending Stage 4's [0.90]*12 + [0.65]*5 + [0.35]*7
_TAPER_32 = np.array([0.90] * 12 + [0.65] * 5 + [0.35] * 7 + [0.18] * 8, dtype=np.float64)
# Raised-cosine bell over cep[10:36] for bounded GV contrast restoration
_GV_Q_START = 10
_GV_Q_END = 36
_GV_BELL_W = 0.5 * (
    1.0 - np.cos(2.0 * np.pi * (np.arange(_GV_Q_START, _GV_Q_END, dtype=np.float64) - 9.0) / 27.0)
)


def bin_dur_ms(dur_ms: float) -> int:
    """Discretize syllable duration (ms) into {0: short <145, 1: mid, 2: long >215}."""
    if dur_ms < 145.0:
        return 0
    if dur_ms <= 215.0:
        return 1
    return 2


def bin_f0_st(f0_st: float) -> int:
    """Discretize relative F0 (st) into {0: low <-1.5, 1: mid, 2: high >+1.5}."""
    if f0_st < -1.5:
        return 0
    if f0_st <= 1.5:
        return 1
    return 2


def _interp_phase_bins_cubic(
    bin_vals: np.ndarray,
    u: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Vectorized clamped Catmull-Rom cubic interpolation across evenly spaced phase bin centers."""
    k = bin_vals.shape[0]
    if k <= 2 or len(u) == 0:
        return _interp_phase_bins(bin_vals, u, centers)

    c0 = float(centers[0])
    step = float(centers[1] - centers[0])
    pos = np.clip((u - c0) / max(1e-9, step), 0.0, float(k - 1))
    idx = np.minimum(k - 2, np.floor(pos).astype(np.intp))
    t = pos - idx

    # Linearly extrapolate virtual boundary knots so endpoints are C^1 and overshoot-free
    p_m1 = 2.0 * bin_vals[0] - bin_vals[1]
    p_pk = 2.0 * bin_vals[-1] - bin_vals[-2]
    padded = np.concatenate([p_m1[None, :], bin_vals, p_pk[None, :]], axis=0)

    j = idx + 1
    p0 = padded[j - 1]
    p1 = padded[j]
    p2 = padded[j + 1]
    p3 = padded[j + 2]
    t2 = (t * t)[:, None]
    t3 = t2 * t[:, None]
    t1 = t[:, None]

    cubic = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t1
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
    # Clamp strictly within local 2-bin envelope (+5% tolerance) to prevent spline overshoot
    lo = np.minimum(p1, p2) - 0.05 * np.abs(p2 - p1)
    hi = np.maximum(p1, p2) + 0.05 * np.abs(p2 - p1)
    return np.clip(cubic, lo, hi)


def _load_gv_table(d: Any, prefix: str) -> dict[str, np.ndarray]:
    k_name = f"{prefix}_keys"
    if k_name not in d:
        return {}
    keys = d[k_name]
    vals = d[f"{prefix}_std"]
    return {str(keys[i]): vals[i].astype(np.float64) for i in range(len(keys))}


@dataclass
class AcousticKnowledgeStage5:
    """Stage 5 high-res medoids + prosody/context coarse anchors + per-key GV targets."""

    on_of: dict[str, MedoidGroup]
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
    onset_uv_frac: dict[str, float]
    sil_cep64: np.ndarray
    sil_bap6: np.ndarray
    speaker_f0_med_hz: float
    ref_log_amp: float
    n_train_utts: int

    # Onset coarse anchors (shrunk toward parent, cep[:, :32])
    on_ctx_of_pp: dict[str, np.ndarray] = field(default_factory=dict)
    on_ctx_of_pos: dict[str, np.ndarray] = field(default_factory=dict)
    on_ctx_of_prev: dict[str, np.ndarray] = field(default_factory=dict)
    on_ctx_of_base: dict[str, np.ndarray] = field(default_factory=dict)
    on_ctx_p: dict[str, np.ndarray] = field(default_factory=dict)
    on_ctx_base: dict[str, np.ndarray] = field(default_factory=dict)

    # Final context & prosody coarse anchors (shrunk toward parent, cep[:, :32])
    fin_ctx_pnp: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_pn: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_np: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_pp: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_pos: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_pft_pos: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_base: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_pos_df: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_df: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_d: dict[str, np.ndarray] = field(default_factory=dict)
    fin_ctx_f0: dict[str, np.ndarray] = field(default_factory=dict)

    # Per-key real cepstral std targets (40,) for GV contrast restoration
    fin_gv_oft: dict[str, np.ndarray] = field(default_factory=dict)
    fin_gv_glob: np.ndarray = field(default_factory=lambda: np.ones(40, dtype=np.float64))

    @classmethod
    def load(cls, path: Path | str = STAGE5_KNOWLEDGE_PATH) -> AcousticKnowledgeStage5:
        with np.load(Path(path), allow_pickle=False) as d:
            on_frac_map = {str(k): float(v) for k, v in zip(d["onset_frac_keys"], d["onset_frac_vals"])}
            on_ms_map = {str(k): float(v) for k, v in zip(d["onset_frac_keys"], d["onset_ms_vals"])}
            on_uv_map = (
                {str(k): float(v) for k, v in zip(d["onset_uv_keys"], d["onset_uv_vals"])}
                if "onset_uv_keys" in d
                else {}
            )
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
                onset_frac=on_frac_map,
                onset_ms=on_ms_map,
                onset_uv_frac=on_uv_map,
                sil_cep64=d["sil_cep64"].astype(np.float64),
                sil_bap6=d["sil_bap6"].astype(np.float64),
                speaker_f0_med_hz=float(d["speaker_f0_med_hz"][0]),
                ref_log_amp=float(d["ref_log_amp"][0]),
                n_train_utts=int(d["n_train_utts"][0]),
                on_ctx_of_pp=_load_coarse_table(d, "on_ctx_of_pp"),
                on_ctx_of_pos=_load_coarse_table(d, "on_ctx_of_pos"),
                on_ctx_of_prev=_load_coarse_table(d, "on_ctx_of_prev"),
                on_ctx_of_base=_load_coarse_table(d, "on_ctx_of_base"),
                on_ctx_p=_load_coarse_table(d, "on_ctx_p"),
                on_ctx_base=_load_coarse_table(d, "on_ctx_base"),
                fin_ctx_pnp=_load_coarse_table(d, "fin_ctx_pnp"),
                fin_ctx_pn=_load_coarse_table(d, "fin_ctx_pn"),
                fin_ctx_np=_load_coarse_table(d, "fin_ctx_np"),
                fin_ctx_pp=_load_coarse_table(d, "fin_ctx_pp"),
                fin_ctx_pos=_load_coarse_table(d, "fin_ctx_pos"),
                fin_ctx_pft_pos=_load_coarse_table(d, "fin_ctx_pft_pos"),
                fin_ctx_base=_load_coarse_table(d, "fin_ctx_base"),
                fin_ctx_pos_df=_load_coarse_table(d, "fin_ctx_pos_df"),
                fin_ctx_df=_load_coarse_table(d, "fin_ctx_df"),
                fin_ctx_d=_load_coarse_table(d, "fin_ctx_d"),
                fin_ctx_f0=_load_coarse_table(d, "fin_ctx_f0"),
                fin_gv_oft=_load_gv_table(d, "fin_gv_oft"),
                fin_gv_glob=d["fin_gv_glob"].astype(np.float64) if "fin_gv_glob" in d else np.ones(40),
            )

    def lookup_onset_coarse(
        self, onset: str, final: str, prev_fg_id: int, pos_id: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return `(context_coarse, base_coarse)` for an onset."""
        k_of = f"{onset}|{final}"
        fg = classify_final_group(final)
        base = self.on_ctx_of_base.get(k_of)
        if base is None:
            base = self.on_ctx_base.get(f"{onset}|{fg}")
        ctx = None
        for tbl, key in (
            (self.on_ctx_of_pp, f"{k_of}|{prev_fg_id}|{pos_id}"),
            (self.on_ctx_of_pos, f"{k_of}|{pos_id}"),
            (self.on_ctx_of_prev, f"{k_of}|{prev_fg_id}"),
            (self.on_ctx_of_base, k_of),
            (self.on_ctx_p, f"{onset}|{fg}|{prev_fg_id}"),
            (self.on_ctx_base, f"{onset}|{fg}"),
        ):
            val = tbl.get(key)
            if val is not None:
                ctx = val
                break
        return ctx, base

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
        for tbl, key in (
            (self.fin_ctx_pnp, f"{k_oft}|{prev_fg_id}|{next_pl_id}|{pos_id}"),
            (self.fin_ctx_pn, f"{k_oft}|{prev_fg_id}|{next_pl_id}"),
            (self.fin_ctx_np, f"{k_oft}|{next_pl_id}|{pos_id}"),
            (self.fin_ctx_pp, f"{k_oft}|{prev_fg_id}|{pos_id}"),
            (self.fin_ctx_pos, f"{k_oft}|{pos_id}"),
        ):
            c = tbl.get(key)
            if c is not None:
                return c
        place = classify_onset_place(onset)
        c = self.fin_ctx_pft_pos.get(f"{place}|{final}|{tone}|{pos_id}")
        if c is not None:
            return c
        return self.fin_ctx_base.get(k_oft)

    def lookup_final_base(self, onset: str, final: str, tone: int) -> np.ndarray | None:
        return self.fin_ctx_base.get(f"{onset}|{final}|{tone}")

    def lookup_final_prosody(
        self, onset: str, final: str, tone: int, dur_b: int, f0_b: int
    ) -> np.ndarray | None:
        k_oft = f"{onset}|{final}|{tone}"
        for tbl, key in (
            (self.fin_ctx_df, f"{k_oft}|{dur_b}|{f0_b}"),
            (self.fin_ctx_d, f"{k_oft}|{dur_b}"),
            (self.fin_ctx_f0, f"{k_oft}|{f0_b}"),
        ):
            c = tbl.get(key)
            if c is not None:
                return c
        return None

    def lookup_final_prosody_delta(
        self, onset: str, final: str, tone: int, pos_id: int, dur_b: int, f0_b: int
    ) -> np.ndarray | None:
        """Return position-debiased `(T_fin, >=12)` prosody cepstral shift relative to matched baseline."""
        k_oft = f"{onset}|{final}|{tone}"
        f_pos_df = self.fin_ctx_pos_df.get(f"{k_oft}|{pos_id}|{dur_b}|{f0_b}")
        f_pos = self.fin_ctx_pos.get(f"{k_oft}|{pos_id}")
        if f_pos_df is not None and f_pos is not None:
            return f_pos_df[:, :12] - f_pos[:, :12]
        f_pros = self.lookup_final_prosody(onset, final, tone, dur_b, f0_b)
        f_base = self.lookup_final_base(onset, final, tone)
        if f_pros is not None and f_base is not None:
            return f_pros[:, :12] - f_base[:, :12]
        return None

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


def _score_medoid_unary_s5(
    grp: MedoidGroup,
    prev_fg_id: int,
    next_pl_id: int,
    pos_id: int,
    dur_ms: float,
    f0_st: float,
) -> np.ndarray:
    """Compute Stage 5 context + prosody mismatch cost `(K,)` (lower is better)."""
    k = grp.cep.shape[0]
    costs = np.zeros(k, dtype=np.float64)
    costs += np.where(grp.ctx_prev == prev_fg_id, 0.0, 0.50)
    costs += np.where(grp.ctx_next == next_pl_id, 0.0, 0.45)
    costs += np.where(grp.ctx_pos == pos_id, 0.0, 0.35)

    dur_ratio = np.abs(np.log(np.maximum(10.0, dur_ms) / np.maximum(10.0, grp.ctx_dur_ms)))
    costs += 0.72 * dur_ratio
    costs += 0.075 * np.abs(f0_st - grp.ctx_f0_st)

    max_sz = float(max(1, int(np.max(grp.cluster_sz))))
    costs += 0.12 * (1.0 - grp.cluster_sz.astype(np.float64) / max_sz)
    return costs


def _select_best_onset_s5(
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
    """Select best Stage 5 onset medoid with cep[0:24] boundary continuity."""
    if grp is None:
        return fallback[0], fallback[1]
    if grp.cep.shape[0] == 1:
        return grp.cep[0], grp.bap[0]

    costs = _score_medoid_unary_s5(grp, prev_fg_id, next_pl_id, pos_id, on_dur_ms, f0_st)
    w_fin = 0.25 if is_obstruent else 0.55
    d_fin = np.sqrt(np.mean((grp.cep[:, -1, 1:24] - fin_start_cep[None, 1:24]) ** 2, axis=1))
    costs += w_fin * d_fin
    if prev_end_cep is not None and not is_obstruent:
        d_prev = np.sqrt(np.mean((grp.cep[:, 0, 1:24] - prev_end_cep[None, 1:24]) ** 2, axis=1))
        costs += 0.45 * d_prev
    return grp.cep[int(np.argmin(costs))], grp.bap[int(np.argmin(costs))]


def build_stage5_control(
    ctx: UtteranceContext,
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    knowledge: AcousticKnowledgeStage5,
    *,
    mode: str = "stage5",
    syl_gain_db: np.ndarray | None = None,
    anchor_alpha: float = 0.90,
    prosody_alpha: float = 0.55,
    contrast_bell_peak: float = 0.17,
    gv_strength: float = 0.15,
    use_cubic: bool = True,
    clean_bap_weight: float = 0.85,
) -> tuple[LarynxControl, dict[str, Any]]:
    """Build Stage 5 frame-level LarynxControl with Stage-4 tapered MMSE + shrunk prosody shift."""
    del mode
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
        return ctrl, {"mode": "stage5", "hits": hit_counts}

    ref_f0 = max(50.0, float(knowledge.speaker_f0_med_hz))

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

    # Step 1: Viterbi DP over K<=8 final medoids with cep[0:24] boundary continuity
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
            unary_costs.append(_score_medoid_unary_s5(grp, prev_id, next_id, pos_id, dur_ms, f0_st))
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
                d_e = 0.22 * np.abs(prev_e_end - o_e0.T)
                min_d = np.min(d_mat + d_e, axis=1)[:, None]
                trans = 0.26 * np.repeat(min_d, k_cur, axis=1)
            else:
                cur_c_start = fin_cep_list[i][:, 0, 1:24]
                cur_e_start = fin_cep_list[i][:, 0, :1]
                d_mat = np.sqrt(np.mean((prev_c_end[:, None, :] - cur_c_start[None, :, :]) ** 2, axis=2))
                d_e = 0.22 * np.abs(prev_e_end - cur_e_start.T)
                trans = 0.52 * (d_mat + d_e)

        total = dp[i - 1][:, None] + trans + unary_costs[i][None, :]
        bp = np.argmin(total, axis=0)
        dp.append(total[bp, np.arange(k_cur)])
        back.append(bp)

    chosen_fin = [0] * n_s
    chosen_fin[-1] = int(np.argmin(dp[-1]))
    for i in range(n_s - 2, -1, -1):
        chosen_fin[i] = int(back[i][chosen_fin[i + 1]])

    # Step 2: Tapered MMSE context+prosody anchoring + phase interpolation + contrast bell
    taper_vec = np.clip(_TAPER_32 * (float(anchor_alpha) / 0.90), 0.0, 1.0)
    p_alpha = float(np.clip(prosody_alpha, 0.0, 1.2))
    bell_pk = float(np.clip(contrast_bell_peak, 0.0, 0.40))
    bell_vec = (1.0 + bell_pk * _BELL_COS_WEIGHT)[None, :] if bell_pk > 0.0 else None
    gv_s = float(np.clip(gv_strength, 0.0, 1.0))
    bap_w = float(np.clip(clean_bap_weight, 0.0, 1.0))
    interp_low = _interp_phase_bins_cubic if use_cubic else _interp_phase_bins

    for i in range(n_s):
        a, sp, b, prev_id, next_id, pos_id, dur_ms, on_ms, f0_st, o, f, is_obs = syl_meta[i]
        if b <= a:
            continue
        t = int(syls[i].tone)
        dur_b = bin_dur_ms(dur_ms)
        f0_b = bin_f0_st(f0_st)

        f_idx = chosen_fin[i]
        f_cep = fin_cep_list[i][f_idx].copy()
        f_bap = fin_bap_list[i][f_idx]

        f_ctx = knowledge.lookup_final_coarse(o, f, t, prev_id, next_id, pos_id)
        if f_ctx is not None:
            c_dim = min(N_COARSE_CEP_S5, f_ctx.shape[1])
            tgt = f_ctx[:, :c_dim].copy()
            if p_alpha > 0.0:
                raw_dp = knowledge.lookup_final_prosody_delta(o, f, t, pos_id, dur_b, f0_b)
                if raw_dp is not None:
                    d_c0 = np.clip(raw_dp[:, 0], -3.8, 3.8)
                    d_c11 = np.clip(raw_dp[:, 1:12], -1.6, 1.6)
                    tgt[:, 0] += p_alpha * d_c0
                    if c_dim > 1:
                        hi_p = min(c_dim, 12)
                        tgt[:, 1:hi_p] += p_alpha * d_c11[:, : hi_p - 1]
            av = taper_vec[:c_dim][None, :]
            f_cep[:, :c_dim] = av * tgt + (1.0 - av) * f_cep[:, :c_dim]

        if o and sp > a:
            prev_end = (
                fin_cep_list[i - 1][chosen_fin[i - 1], -1]
                if (i > 0 and prev_id != PREV_FG_TO_ID["SIL"])
                else None
            )
            o_cep, o_bap = _select_best_onset_s5(
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
            o_ctx, _ = knowledge.lookup_onset_coarse(o, f, prev_id, pos_id)
            if o_ctx is not None:
                c_dim = min(N_COARSE_CEP_S5, o_ctx.shape[1])
                av = taper_vec[:c_dim][None, :]
                o_cep[:, :c_dim] = av * o_ctx[:, :c_dim] + (1.0 - av) * o_cep[:, :c_dim]

            u_on = ctx.phase_u[a:sp]
            if use_cubic:
                cep[a:sp, :24] = interp_low(o_cep[:, :24], u_on, _ONSET_CENTERS_S5)
                cep[a:sp, 24:] = _interp_phase_bins(o_cep[:, 24:], u_on, _ONSET_CENTERS_S5)
            else:
                cep[a:sp] = _interp_phase_bins(o_cep, u_on, _ONSET_CENTERS_S5)
            bap[a:sp] = _interp_phase_bins(o_bap, u_on, _ONSET_CENTERS_S5)
            f_a = sp
        else:
            f_a = a

        if b > f_a:
            u_fi = ctx.phase_u[f_a:b]
            if use_cubic:
                cep[f_a:b, :24] = interp_low(f_cep[:, :24], u_fi, _FINAL_CENTERS_S5)
                cep[f_a:b, 24:] = _interp_phase_bins(f_cep[:, 24:], u_fi, _FINAL_CENTERS_S5)
            else:
                cep[f_a:b] = _interp_phase_bins(f_cep, u_fi, _FINAL_CENTERS_S5)
            bap[f_a:b] = _interp_phase_bins(f_bap, u_fi, _FINAL_CENTERS_S5)

            if bell_vec is not None:
                cep[f_a:b, _BELL_Q_START:_BELL_Q_END] *= bell_vec

            # Optional bounded per-quefrency GV contrast restoration over cep[10:36]
            if gv_s > 0.0 and (b - f_a) >= 4:
                seg_sub = cep[f_a:b, _GV_Q_START:_GV_Q_END]
                mu = np.mean(seg_sub, axis=0, keepdims=True)
                cur_std = np.std(seg_sub, axis=0)
                tgt_std = knowledge.lookup_final_gv(o, f, t)[_GV_Q_START:_GV_Q_END]
                ratio = np.clip(tgt_std / np.maximum(cur_std, 0.15), 0.90, 1.32)
                scale_q = 1.0 + gv_s * _GV_BELL_W * (ratio - 1.0)
                cep[f_a:b, _GV_Q_START:_GV_Q_END] = mu + (seg_sub - mu) * scale_q[None, :]

        # Calibrate voiced BAP bands 1..5 toward Stage 4 1-7 kHz HNR targets
        if bap_w > 0.0 and b > a:
            v_seg = voi_arr[a:b] & (f0_arr[a:b] > 50.0)
            if np.any(v_seg):
                tgt = _COMB_BAP_TP.get((t, pos_id), _GLOBAL_COMB_BAP)
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
                log_env[a:b] += 0.55 * g_nep

    ctrl = LarynxControl(
        f0_hz=f0_arr,
        voiced=voi_arr,
        log_env=log_env,
        bap=np.clip(bap, 0.005, 0.995),
        n_samples=n_fr * HOP,
    )
    return ctrl, {"mode": "stage5", "hits": hit_counts, "chosen_fin_medoids": chosen_fin}
