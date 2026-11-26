"""Stage 7 zero-ML-weight Two-Pass Continuous Ridge + Xu-Coarticulation Prosody Engine."""
from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any
import zlib

import numpy as np
from scipy.signal import lfilter

from exp_20_pybot_larynx import EXP_DIR
from exp_20_pybot_larynx.acoustics.knowledge import normalize_of
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import (
    DEFAULT_UV_LEVEL_BOOST_DB_S7,
    AcousticKnowledgeStage7,
    build_stage7_control,
)
from exp_20_pybot_larynx.arbiter import (
    OBSTRUENT_MANNERS,
    Arbiter,
    SyllableSpan,
    UtteranceContext,
    build_utterance_context,
    classify_final_group,
    classify_onset_manner,
)
from exp_20_pybot_larynx.frontend.prosody_rules import (
    _CF0_VOICED_ST,
    _CF0_VOICELESS_ST,
    _F0_AR1_SIGMA,
    _F0_U5,
    _F0_U7,
    _catmull_rom_5pt,
    _catmull_rom_7pt,
    predict_prosody_rules,
)
from exp_20_pybot_larynx.larynx.synth_v2 import synthesize_larynx_v2

PROSODY_S7_TABLES_PATH = EXP_DIR / "frontend" / "prosody_s7_tables.npz"
PROSODY_S7_SUMMARY_PATH = EXP_DIR / "frontend" / "prosody_s7_summary.json"

_MANNER_TO_ID: dict[str, int] = {
    "zero": 0,
    "stop_unasp": 1,
    "stop_asp": 2,
    "affr_unasp": 3,
    "affr_asp": 4,
    "fric": 5,
    "nasal": 6,
    "liquid": 7,
}
_FG_TO_ID: dict[str, int] = {
    "i_front": 0,
    "u_round": 1,
    "v_front_round": 2,
    "a_open_central": 3,
}
_WPOS_TO_ID: dict[str, int] = {
    "mono": 0,
    "w_init": 1,
    "w_mid": 2,
    "w_fin": 3,
}

DEFAULT_ONSETS_VOCAB: tuple[str, ...] = (
    "", "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s",
)
DEFAULT_FINALS_VOCAB: tuple[str, ...] = (
    "a", "ai", "an", "ang", "ao",
    "e", "ei", "en", "eng", "er",
    "i", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "ir", "iu", "iz",
    "o", "ong", "ou",
    "u", "ua", "uai", "uan", "uang", "ui", "un", "uo",
    "v", "van", "ve", "vn", "io",
)


def compute_unit_context_s7(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute hierarchical clause/phrase/word/phonetic prosodic context per unit."""
    n_u = len(units)
    if n_u == 0:
        return []

    clauses: list[list[int]] = []
    cur: list[int] = []
    for i, u in enumerate(units):
        cur.append(i)
        if str(u.get("punct_after", "")) or float(u.get("pause_after_ms", 0.0)) >= 25.0:
            clauses.append(cur)
            cur = []
    if cur:
        clauses.append(cur)

    n_cl = len(clauses)
    out: list[dict[str, Any]] = [{} for _ in range(n_u)]

    for cl_idx, cl in enumerate(clauses):
        cl_len = len(cl)
        phrases: list[list[int]] = []
        pcur: list[int] = []
        for ci, ui in enumerate(cl):
            pcur.append(ui)
            if ci == cl_len - 1 or int(units[ui].get("prosodic_break", 0)) >= 2:
                phrases.append(pcur)
                pcur = []
        ph_info: dict[int, tuple[float, int, int]] = {}
        for ph in phrases:
            lp = len(ph)
            for j, ui in enumerate(ph):
                p_pos = float(j) / float(max(1, lp - 1)) if lp > 1 else 0.5
                ph_info[ui] = (p_pos, lp, lp - 1 - j)

        # Pre-normalize onset/final inside clause
        norm_of_cl: list[tuple[str, str, str, bool, int]] = []
        for ui in cl:
            u = units[ui]
            o, f = normalize_of(str(u.get("onset", "")), str(u.get("final", "")))
            om = classify_onset_manner(o) if o else "zero"
            t = int(np.clip(int(u.get("tone") or 1), 1, 5))
            norm_of_cl.append((o, f, om, om in OBSTRUENT_MANNERS, t))

        for ci, ui in enumerate(cl):
            u = units[ui]
            o, f, om, cur_vl, t = norm_of_cl[ci]
            fg = classify_final_group(f)
            pt = norm_of_cl[ci - 1][4] if ci > 0 else 0
            p2t = norm_of_cl[ci - 2][4] if ci > 1 else 0
            nt = norm_of_cl[ci + 1][4] if ci + 1 < cl_len else 0
            n2t = norm_of_cl[ci + 2][4] if ci + 2 < cl_len else 0

            prev_vl = 1.0 if (ci > 0 and norm_of_cl[ci - 1][3]) else 0.0
            next_vl = 1.0 if (ci + 1 < cl_len and norm_of_cl[ci + 1][3]) else 0.0

            wl = int(u.get("word_len", 1))
            piw = int(u.get("pos_in_word", 0))
            wpos_id = 0 if wl <= 1 else (1 if piw == 0 else (3 if piw >= wl - 1 else 2))
            brk = int(np.clip(int(u.get("prosodic_break", 0)), 0, 4))
            is_wf = 1 if bool(u.get("is_word_final", True)) else 0
            is_func = 1.0 if bool(u.get("is_func_word", False)) else 0.0

            prev_brk_ge2 = 1.0 if (ci == 0 or int(units[cl[ci - 1]].get("prosodic_break", 0)) >= 2) else 0.0
            next_brk_ge2 = 1.0 if (ci == cl_len - 1 or brk >= 2) else 0.0
            pause_after = float(u.get("pause_after_ms", 0.0))
            pause_prev = float(units[ui - 1].get("pause_after_ms", 0.0)) if ui > 0 else 500.0
            if ui > 0 and str(units[ui - 1].get("punct_after", "")):
                pause_prev = max(pause_prev, 180.0)

            cl_pos = float(ci) / float(max(1, cl_len - 1)) if cl_len > 1 else 0.5
            ph_pos, ph_len, rem_ph = ph_info.get(ui, (0.5, 1, 0))
            is_linked_prev = 1.0 if (ci > 0 and pause_prev < 25.0) else 0.0
            is_linked_next = 1.0 if (ci + 1 < cl_len and pause_after < 25.0 and not str(u.get("punct_after", ""))) else 0.0

            out[ui] = {
                "t": t,
                "pt": pt,
                "p2t": p2t,
                "nt": nt,
                "n2t": n2t,
                "o": o,
                "f": f,
                "om": om,
                "om_id": _MANNER_TO_ID.get(om, 0),
                "fg": fg,
                "fg_id": _FG_TO_ID.get(fg, 3),
                "wpos_id": wpos_id,
                "wl": min(wl, 4),
                "brk": brk,
                "brk_ge2": 1 if brk >= 2 else 0,
                "prev_brk_ge2": prev_brk_ge2,
                "next_brk_ge2": next_brk_ge2,
                "is_wf": is_wf,
                "is_func": is_func,
                "cl_pos": cl_pos,
                "cl_len": cl_len,
                "rem_cl": cl_len - 1 - ci,
                "ph_pos": ph_pos,
                "ph_len": ph_len,
                "rem_ph": rem_ph,
                "cl_idx": cl_idx,
                "n_cl": n_cl,
                "is_cl_init": 1.0 if ci == 0 else 0.0,
                "is_cl_fin": 1.0 if ci == cl_len - 1 else 0.0,
                "is_fin_cl": 1.0 if cl_idx == n_cl - 1 else 0.0,
                "is_utt_init": 1.0 if ui == 0 else 0.0,
                "is_utt_fin": 1.0 if ui == n_u - 1 else 0.0,
                "prev_vl": prev_vl,
                "cur_vl": 1.0 if cur_vl else 0.0,
                "next_vl": next_vl,
                "pause_after_ms": pause_after,
                "pause_prev_ms": pause_prev,
                "is_linked_prev": is_linked_prev,
                "is_linked_next": is_linked_next,
                "cl_pos_b": min(3, int(4 * ci / max(1, cl_len))) if cl_len > 1 else 1,
                "ph_pos_b": min(2, int(3 * (ph_len - 1 - rem_ph) / max(1, ph_len))) if ph_len > 1 else 1,
            }
    return out


def extract_prosody_stage7_pass1_features(
    units: list[dict[str, Any]],
    *,
    ctxs: list[dict[str, Any]] | None = None,
    tritone_map: dict[tuple[int, int, int], int] | None = None,
    n_tritones: int = 0,
    onset_map: dict[str, int] | None = None,
    n_onsets: int = 0,
    final_map: dict[str, int] | None = None,
    n_finals: int = 0,
) -> np.ndarray:
    """Build deterministic Pass-1 feature matrix `(N_u, D1)` from rule-planned units."""
    c_list = ctxs if ctxs is not None else compute_unit_context_s7(units)
    n_u = len(units)
    o_map = onset_map if onset_map is not None else {k: i for i, k in enumerate(DEFAULT_ONSETS_VOCAB)}
    k_on = n_onsets if onset_map is not None else len(DEFAULT_ONSETS_VOCAB)
    f_map = final_map if final_map is not None else {k: i for i, k in enumerate(DEFAULT_FINALS_VOCAB)}
    k_fin = n_finals if final_map is not None else len(DEFAULT_FINALS_VOCAB)
    k_tri = int(n_tritones) if tritone_map is not None else 0

    # Offsets
    # 0..10: continuous + bias
    # 10..15: t(5), 15..21: pt(6), 21..27: nt(6), 27..33: p2t(6), 33..39: n2t(6)
    # 39..69: (pt, t) 6x5=30, 69..99: (t, nt) 5x6=30
    # 99..99+k_tri: tri-tones
    # +4 wpos +1 wl +5 brk +4 edge_flags +4 pos +4 len_inv +4 cl_utt +8 om +3 vl +4 fg +k_on +k_fin +25 interact
    off_tri = 99
    off_wpos = off_tri + k_tri
    off_brk = off_wpos + 5
    off_flags = off_brk + 5
    off_pos = off_flags + 4
    off_len = off_pos + 4
    off_edge = off_len + 4
    off_om = off_edge + 5
    off_vl = off_om + 8
    off_fg = off_vl + 3
    off_on = off_fg + 4
    off_fi = off_on + k_on
    off_int = off_fi + k_fin
    d_tot = off_int + 25

    X = np.zeros((n_u, d_tot), dtype=np.float64)
    for i, (u, c) in enumerate(zip(units, c_list)):
        t = c["t"]
        t0 = t - 1
        pt = c["pt"]
        nt = c["nt"]
        p2t = c["p2t"]
        n2t = c["n2t"]
        f3 = u["f0_3pt_st"]
        dur_ms = max(30.0, float(u["duration_ms"]))
        has_on = 1.0 if c["o"] else 0.0
        on_ms = max(5.0, float(u.get("onset_ms", 0.0))) if has_on > 0.0 else 0.0

        X[i, 0] = 1.0
        X[i, 1] = float(f3[0])
        X[i, 2] = float(f3[1])
        X[i, 3] = float(f3[2])
        X[i, 4] = math.log(dur_ms)
        X[i, 5] = math.log(on_ms) if on_ms > 0.0 else 0.0
        X[i, 6] = has_on
        X[i, 7] = float(u.get("gain_db", 0.0))
        X[i, 8] = math.log1p(max(0.0, c["pause_after_ms"]))
        X[i, 9] = math.log1p(max(0.0, c["pause_prev_ms"]))

        X[i, 10 + t0] = 1.0
        X[i, 15 + pt] = 1.0
        X[i, 21 + nt] = 1.0
        X[i, 27 + p2t] = 1.0
        X[i, 33 + n2t] = 1.0
        X[i, 39 + pt * 5 + t0] = 1.0
        X[i, 69 + t0 * 6 + nt] = 1.0
        if tritone_map is not None:
            ti = tritone_map.get((pt, t, nt))
            if ti is not None:
                X[i, off_tri + ti] = 1.0

        X[i, off_wpos + c["wpos_id"]] = 1.0
        X[i, off_wpos + 4] = float(c["wl"]) / 4.0
        X[i, off_brk + c["brk"]] = 1.0
        X[i, off_flags + 0] = c["prev_brk_ge2"]
        X[i, off_flags + 1] = c["next_brk_ge2"]
        X[i, off_flags + 2] = float(c["is_wf"])
        X[i, off_flags + 3] = c["is_func"]

        cp = c["cl_pos"]
        pp = c["ph_pos"]
        X[i, off_pos + 0] = cp
        X[i, off_pos + 1] = cp * cp
        X[i, off_pos + 2] = pp
        X[i, off_pos + 3] = pp * pp

        X[i, off_len + 0] = math.log(max(1, c["cl_len"]))
        X[i, off_len + 1] = math.log(max(1, c["ph_len"]))
        X[i, off_len + 2] = 1.0 / float(1 + c["rem_cl"])
        X[i, off_len + 3] = 1.0 / float(1 + c["rem_ph"])

        X[i, off_edge + 0] = c["is_cl_init"]
        X[i, off_edge + 1] = c["is_cl_fin"]
        X[i, off_edge + 2] = c["is_fin_cl"]
        X[i, off_edge + 3] = c["is_utt_init"]
        X[i, off_edge + 4] = c["is_utt_fin"]

        X[i, off_om + c["om_id"]] = 1.0
        X[i, off_vl + 0] = c["prev_vl"]
        X[i, off_vl + 1] = c["cur_vl"]
        X[i, off_vl + 2] = c["next_vl"]
        X[i, off_fg + c["fg_id"]] = 1.0

        oi = o_map.get(c["o"])
        if oi is not None:
            X[i, off_on + oi] = 1.0
        fi = f_map.get(c["f"])
        if fi is not None:
            X[i, off_fi + fi] = 1.0

        X[i, off_int + 0 + t0] = cp
        X[i, off_int + 5 + t0] = pp
        X[i, off_int + 10 + t0] = float(c["brk_ge2"])
        X[i, off_int + 15 + t0] = float(c["is_wf"])
        X[i, off_int + 20 + t0] = c["is_fin_cl"]
    return X


def extract_prosody_stage7_pass2_features(
    units: list[dict[str, Any]],
    p1_f0_3pt: np.ndarray,
    *,
    ctxs: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Build Xu carry-over & anticipatory coarticulation features `(N_u, 34)` from Pass-1 F0."""
    c_list = ctxs if ctxs is not None else compute_unit_context_s7(units)
    n_u = len(units)
    X2 = np.zeros((n_u, 34), dtype=np.float64)
    p1 = np.asarray(p1_f0_3pt, dtype=np.float64)

    for i, c in enumerate(c_list):
        t0 = c["t"] - 1
        lp = c["is_linked_prev"]
        ln = c["is_linked_next"]
        c0, c1, c2 = float(p1[i, 0]), float(p1[i, 1]), float(p1[i, 2])

        X2[i, 0] = 1.0
        X2[i, 1] = lp
        X2[i, 2] = ln
        X2[i, 3] = c0
        X2[i, 4] = c1
        X2[i, 5] = c2

        d_left = 0.0
        if lp > 0.5 and i > 0:
            p_end = float(p1[i - 1, 2])
            p_mid = float(p1[i - 1, 1])
            p_slp = p_end - float(p1[i - 1, 0])
            d_left = p_end - c0
            X2[i, 6] = p_end
            X2[i, 7] = p_mid
            X2[i, 8] = d_left
            X2[i, 9] = p_slp

        d_right = 0.0
        if ln > 0.5 and i + 1 < n_u:
            n_beg = float(p1[i + 1, 0])
            n_mid = float(p1[i + 1, 1])
            n_slp = float(p1[i + 1, 2]) - n_beg
            d_right = n_beg - c2
            X2[i, 10] = n_beg
            X2[i, 11] = n_mid
            X2[i, 12] = d_right
            X2[i, 13] = n_slp

        X2[i, 14 + t0] = d_left
        X2[i, 19 + t0] = d_right
        X2[i, 24 + t0] = lp
        X2[i, 29 + t0] = ln
    return X2


class ProsodyStage7Model:
    """Explicit closed-form Ridge + Xu-coarticulation prosody model stored in `allow_pickle=False` .npz."""

    def __init__(self, path: Path | str = PROSODY_S7_TABLES_PATH) -> None:
        self.loaded = False
        self.tritone_vocab: list[tuple[int, int, int]] = []
        self.tritone_map: dict[tuple[int, int, int], int] = {}
        self.onsets_vocab: list[str] = list(DEFAULT_ONSETS_VOCAB)
        self.onset_map: dict[str, int] = {k: i for i, k in enumerate(self.onsets_vocab)}
        self.finals_vocab: list[str] = list(DEFAULT_FINALS_VOCAB)
        self.final_map: dict[str, int] = {k: i for i, k in enumerate(self.finals_vocab)}

        self.x1_mean: np.ndarray = np.zeros(1, dtype=np.float64)
        self.x1_std: np.ndarray = np.ones(1, dtype=np.float64)
        self.w_f0_p1: np.ndarray = np.zeros((1, 3), dtype=np.float64)
        self.x2_mean: np.ndarray = np.zeros(34, dtype=np.float64)
        self.x2_std: np.ndarray = np.ones(34, dtype=np.float64)
        self.w_f0_p2: np.ndarray = np.zeros((34, 3), dtype=np.float64)

        self.w_dur: np.ndarray = np.zeros(1, dtype=np.float64)
        self.w_ondur: np.ndarray = np.zeros(1, dtype=np.float64)
        self.w_gain: np.ndarray = np.zeros(1, dtype=np.float64)
        self.dur_ctx: dict[tuple[int, int, int, int, int], float] = {}
        self.onset_uv_ofg: dict[str, float] = {}
        self.onset_uv_o: dict[str, float] = {}
        self.weights: dict[str, float] = {
            "w_f0_p1": 0.70,
            "w_f0_p2": 0.70,
            "w_dur": 0.65,
            "w_dur_ctx": 0.25,
            "w_ondur": 0.55,
            "w_gain": 0.45,
            "w_uv": 1.0,
        }

        p = Path(path)
        if not p.exists():
            return
        with np.load(p, allow_pickle=False) as d:
            if "tritone_vocab" in d:
                self.tritone_vocab = [tuple(int(x) for x in row) for row in d["tritone_vocab"]]
                self.tritone_map = {k: i for i, k in enumerate(self.tritone_vocab)}
            if "onsets_vocab" in d:
                self.onsets_vocab = [str(x) for x in d["onsets_vocab"]]
                self.onset_map = {k: i for i, k in enumerate(self.onsets_vocab)}
            if "finals_vocab" in d:
                self.finals_vocab = [str(x) for x in d["finals_vocab"]]
                self.final_map = {k: i for i, k in enumerate(self.finals_vocab)}

            self.x1_mean = d["x1_mean"].astype(np.float64)
            self.x1_std = np.maximum(1e-6, d["x1_std"].astype(np.float64))
            self.w_f0_p1 = d["w_f0_p1"].astype(np.float64)
            self.x2_mean = d["x2_mean"].astype(np.float64)
            self.x2_std = np.maximum(1e-6, d["x2_std"].astype(np.float64))
            self.w_f0_p2 = d["w_f0_p2"].astype(np.float64)
            self.w_dur = d["w_dur"].astype(np.float64)
            self.w_ondur = d["w_ondur"].astype(np.float64)
            self.w_gain = d["w_gain"].astype(np.float64)

            if "dur_ctx_keys" in d and "dur_ctx_vals" in d:
                for k, v in zip(d["dur_ctx_keys"], d["dur_ctx_vals"]):
                    self.dur_ctx[tuple(int(x) for x in k)] = float(v)
            if "onset_uv_ofg_keys" in d and "onset_uv_ofg_vals" in d:
                for k, v in zip(d["onset_uv_ofg_keys"], d["onset_uv_ofg_vals"]):
                    self.onset_uv_ofg[str(k)] = float(v)
            if "onset_uv_o_keys" in d and "onset_uv_o_vals" in d:
                for k, v in zip(d["onset_uv_o_keys"], d["onset_uv_o_vals"]):
                    self.onset_uv_o[str(k)] = float(v)
            if "weight_keys" in d and "weight_vals" in d:
                for k, v in zip(d["weight_keys"], d["weight_vals"]):
                    self.weights[str(k)] = float(v)
        self.loaded = True


_MODEL_CACHE: dict[str, ProsodyStage7Model] = {}


def get_stage7_prosody_model(path: Path | str = PROSODY_S7_TABLES_PATH) -> ProsodyStage7Model:
    key = str(Path(path).resolve())
    m = _MODEL_CACHE.get(key)
    if m is None:
        m = ProsodyStage7Model(path)
        _MODEL_CACHE[key] = m
    return m


def clear_stage7_prosody_cache() -> None:
    _MODEL_CACHE.clear()


def predict_prosody_stage7(
    text: str,
    knowledge: Any | None = None,
    *,
    style: str = "auto",
    continues_after: bool = False,
    tables_path: Path | str = PROSODY_S7_TABLES_PATH,
    model: ProsodyStage7Model | None = None,
) -> dict[str, Any]:
    """Predict Stage 7 prosody via Two-Pass Xu-Coarticulation Ridge over `predict_prosody_rules`."""
    pro = predict_prosody_rules(text, knowledge, style=style, continues_after=continues_after)
    mdl = model or get_stage7_prosody_model(tables_path)
    if not mdl.loaded:
        return pro

    units = pro["units"]
    n_u = len(units)
    if n_u == 0:
        return pro

    ctxs = compute_unit_context_s7(units)
    X1 = extract_prosody_stage7_pass1_features(
        units,
        ctxs=ctxs,
        tritone_map=mdl.tritone_map,
        n_tritones=len(mdl.tritone_vocab),
        onset_map=mdl.onset_map,
        n_onsets=len(mdl.onsets_vocab),
        final_map=mdl.final_map,
        n_finals=len(mdl.finals_vocab),
    )
    Z1 = (X1 - mdl.x1_mean[None, :]) / mdl.x1_std[None, :]
    Z1[:, 0] = 1.0

    w = mdl.weights
    w_p1 = float(w.get("w_f0_p1", 0.70))
    w_p2 = float(w.get("w_f0_p2", 0.70))
    w_dur = float(w.get("w_dur", 0.65))
    w_dctx = float(w.get("w_dur_ctx", 0.25))
    w_on = float(w.get("w_ondur", 0.55))
    w_gain = float(w.get("w_gain", 0.45))
    use_uv = float(w.get("w_uv", 1.0)) > 0.5

    pred_df1 = Z1 @ mdl.w_f0_p1  # (n_u, 3)
    base_f3 = np.array([u["f0_3pt_st"] for u in units], dtype=np.float64)
    p1_total = base_f3 + w_p1 * pred_df1

    X2 = extract_prosody_stage7_pass2_features(units, p1_total, ctxs=ctxs)
    Z2 = (X2 - mdl.x2_mean[None, :]) / mdl.x2_std[None, :]
    Z2[:, 0] = 1.0
    pred_df2 = Z2 @ mdl.w_f0_p2

    pred_dl_dur = Z1 @ mdl.w_dur
    pred_dl_on = Z1 @ mdl.w_ondur
    pred_dgain = Z1 @ mdl.w_gain

    u7_arr = np.asarray(_F0_U7, dtype=np.float64)
    u5_arr = np.asarray(_F0_U5, dtype=np.float64)

    for i, (u, c) in enumerate(zip(units, ctxs)):
        # 1) Duration & onset duration & onset UV fraction
        d_log = w_dur * float(pred_dl_dur[i])
        if w_dctx != 0.0 and mdl.dur_ctx:
            d_log += w_dctx * mdl.dur_ctx.get((c["om_id"], c["fg_id"], c["t"], c["wpos_id"], c["brk_ge2"]), 0.0)
        d_log = float(np.clip(d_log, -0.35, 0.35))
        if abs(d_log) > 1e-6:
            new_dur = float(np.clip(float(u["duration_ms"]) * math.exp(d_log), 60.0, 380.0))
            u["duration_ms"] = round(new_dur, 1)

        if c["o"]:
            on_log = float(np.clip(w_on * float(pred_dl_on[i]), -0.35, 0.35))
            if abs(on_log) > 1e-6:
                new_on = float(
                    np.clip(float(u["onset_ms"]) * math.exp(on_log), 12.0, min(200.0, float(u["duration_ms"]) - 20.0))
                )
                u["onset_ms"] = round(new_on, 1)
            if use_uv:
                uv_v = mdl.onset_uv_ofg.get(f"{c['o']}|{c['fg']}", mdl.onset_uv_o.get(c["o"]))
                if uv_v is not None:
                    u["onset_uv_frac"] = round(float(uv_v), 4)

        # 2) Two-pass F0 residual
        df3 = np.clip(w_p1 * pred_df1[i] + w_p2 * pred_df2[i], -6.0, 6.0)
        if np.any(np.abs(df3) > 1e-6):
            df7 = np.interp(u7_arr, (0.20, 0.50, 0.80), df3)
            pts7 = np.asarray(u.get("f0_7pt_st", np.interp(u7_arr, u5_arr, u["f0_5pt_st"])), dtype=np.float64) + df7
            pts5 = _catmull_rom_7pt(u5_arr, pts7)
            u["f0_3pt_st"] = [round(float(x), 3) for x in (base_f3[i] + df3)]
            u["f0_7pt_st"] = [round(float(x), 3) for x in pts7]
            u["f0_5pt_st"] = [round(float(x), 3) for x in pts5]
            u["pitch_st"] = list(u["f0_5pt_st"])

        # 3) Gain residual
        dg = float(np.clip(w_gain * float(pred_dgain[i]), -2.5, 2.5))
        if abs(dg) > 1e-6:
            u["gain_db"] = round(float(np.clip(float(u.get("gain_db", 0.0)) + dg, -4.8, 4.8)), 3)

    return pro


def render_stage7_utterance(
    text: str,
    utt_id: str,
    knowledge: AcousticKnowledgeStage7,
    *,
    arbiter: Arbiter,
    rules: list[Any],
    glottal_mode: str = "dynamic_coherent",
    coherent_phasor: np.ndarray | None = None,
    seed: int = 123,
    style: str = "auto",
    tables_path: Path | str = PROSODY_S7_TABLES_PATH,
    model: ProsodyStage7Model | None = None,
    uv_level_boost_db: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pure zero-ML Stage 7 end-to-end Mandarin TTS synthesis."""
    pro = predict_prosody_stage7(text, knowledge, style=style, tables_path=tables_path, model=model)
    raw_units = pro["units"]
    base_f0 = float(pro["speaker_f0_med_hz"])
    n_u = len(raw_units)

    spans: list[SyllableSpan] = []
    f0_chunks: list[np.ndarray] = []
    voi_chunks: list[np.ndarray] = []
    syl_gains: list[float] = []

    cur_fr = 0
    prev_voiced_end_hz: float | None = None

    for i, u in enumerate(raw_units):
        o, f = normalize_of(str(u.get("onset", "")), str(u.get("final", "")))
        tone = int(u.get("tone") or 1)
        dur_ms = max(60.0, float(u.get("duration_ms", 180.0)))
        n_syl = max(3, int(round(dur_ms / 5.0)))

        if o:
            n_on = max(1, min(n_syl - 2, int(round(float(u.get("onset_ms", 50.0)) / 5.0))))
        else:
            n_on = 0
        split_fr = cur_fr + n_on
        end_fr = cur_fr + n_syl

        manner = classify_onset_manner(o) if o else "zero"
        n_fi = max(1, n_syl - n_on)
        u_fi = np.linspace(0.0, 1.0, n_fi) if n_fi > 1 else np.array([0.5], dtype=np.float64)
        if "f0_7pt_st" in u:
            fi_st = _catmull_rom_7pt(u_fi, np.asarray(u["f0_7pt_st"], dtype=np.float64))
        else:
            fi_st = _catmull_rom_5pt(u_fi, np.asarray(u["f0_5pt_st"], dtype=np.float64))

        if o and n_fi >= 3:
            k_cf0 = min(5, n_fi)
            decay = np.exp(-np.arange(k_cf0, dtype=np.float64) * 5.0 / 12.0)
            if manner in ("stop_asp", "affr_asp", "fric"):
                fi_st[:k_cf0] += _CF0_VOICELESS_ST * decay
            elif manner in ("nasal", "liquid"):
                fi_st[:k_cf0] += _CF0_VOICED_ST * decay
            elif manner in ("stop_unasp", "affr_unasp"):
                fi_st[:k_cf0] += 0.25 * _CF0_VOICELESS_ST * decay

        fi_hz = np.clip(base_f0 * (2.0 ** (fi_st / 12.0)), 65.0, 440.0)
        f0_seg = np.empty(n_syl, dtype=np.float64)
        f0_seg[n_on:] = fi_hz
        if n_on > 0:
            if manner in ("nasal", "liquid") and prev_voiced_end_hz is not None:
                f0_seg[:n_on] = np.linspace(
                    0.65 * prev_voiced_end_hz + 0.35 * fi_hz[0],
                    fi_hz[0],
                    n_on,
                    endpoint=False,
                )
            else:
                f0_seg[:n_on] = fi_hz[0]

        voi_seg = np.ones(n_syl, dtype=bool)
        if o and manner in OBSTRUENT_MANNERS and n_on > 0:
            uv_f = float(u.get("onset_uv_frac", 0.78))
            uv_n = max(1, min(n_on, int(round(uv_f * n_on))))
            voi_seg[:uv_n] = False
            f0_seg[:uv_n] = 0.0

        brk = int(u.get("prosodic_break", 0))
        t_uv = 1 if brk == 2 else (2 if brk == 3 else (4 if brk == 4 else 0))
        if t_uv > 0 and n_fi >= 8:
            t_uv = min(t_uv, n_fi - 6)
            voi_seg[-t_uv:] = False
            f0_seg[-t_uv:] = 0.0

        if prev_voiced_end_hz is not None:
            if voi_seg[0] and n_on == 0 and f0_seg[0] > 50.0:
                k_sm = min(4, n_syl)
                w_sm = np.linspace(0.30, 0.0, k_sm)
                f0_seg[:k_sm] = (1.0 - w_sm) * f0_seg[:k_sm] + w_sm * prev_voiced_end_hz
            elif n_on < n_syl and manner in OBSTRUENT_MANNERS:
                k_sm = min(3, n_syl - n_on)
                w_sm = np.linspace(0.10, 0.0, k_sm)
                f0_seg[n_on : n_on + k_sm] = (1.0 - w_sm) * f0_seg[n_on : n_on + k_sm] + w_sm * prev_voiced_end_hz

        p_ms = float(u.get("pause_after_ms", 0.0)) if i < n_u - 1 else 0.0
        spans.append(SyllableSpan(
            index=i,
            hanzi=str(u.get("hanzi") or u.get("token") or ""),
            onset=o,
            final=f,
            tone=tone,
            start_fr=cur_fr,
            split_fr=split_fr,
            end_fr=end_fr,
            pause_after_ms=p_ms,
            prosodic_break=brk,
        ))
        f0_chunks.append(f0_seg)
        voi_chunks.append(voi_seg)
        syl_gains.append(float(u.get("gain_db", 0.0)))
        cur_fr = end_fr

        if i < n_u - 1 and p_ms >= 20.0:
            n_p = int(round(p_ms / 5.0))
            if n_p > 0:
                f0_chunks.append(np.zeros(n_p, dtype=np.float64))
                voi_chunks.append(np.zeros(n_p, dtype=bool))
                cur_fr += n_p
            prev_voiced_end_hz = None
        else:
            prev_voiced_end_hz = float(f0_seg[-1]) if voi_seg[-1] and f0_seg[-1] > 50.0 else None

    f0_all = np.concatenate(f0_chunks)
    voi_all = np.concatenate(voi_chunks)

    if _F0_AR1_SIGMA > 0.0 and len(f0_all) > 4:
        rng_u = np.random.default_rng(zlib.crc32(f"{utt_id}|{seed}".encode("utf-8")) & 0xFFFFFFFF)
        rho = 0.72
        eps = rng_u.normal(0.0, _F0_AR1_SIGMA * np.sqrt(1.0 - rho * rho), size=len(f0_all))
        ar1 = lfilter((1.0,), (1.0, -rho), eps)
        vmask = voi_all & (f0_all > 50.0)
        f0_all[vmask] = np.clip(f0_all[vmask] * (2.0 ** (np.clip(ar1[vmask], -0.45, 0.45) / 12.0)), 65.0, 440.0)

    ctx: UtteranceContext = build_utterance_context(utt_id, cur_fr, spans, speaker_f0_med_hz=base_f0)
    base_ctrl, kb_info = build_stage7_control(
        ctx,
        f0_all,
        voi_all,
        knowledge,
        syl_gain_db=np.asarray(syl_gains, dtype=np.float64),
    )
    fused_ctrl, arb_trace = arbiter.apply(ctx, base_ctrl, rules)
    boost = DEFAULT_UV_LEVEL_BOOST_DB_S7 if uv_level_boost_db is None else float(uv_level_boost_db)
    t_k0 = time.perf_counter()
    wav = synthesize_larynx_v2(
        fused_ctrl,
        glottal_mode=glottal_mode,
        coherent_phasor=coherent_phasor,
        uv_level_boost_db=boost,
        seed=seed,
    )
    k_elapsed = time.perf_counter() - t_k0
    return wav, {
        "kb_info": kb_info,
        "arbiter_trace": arb_trace,
        "kernel_s": k_elapsed,
        "prosody_plan": pro,
    }
