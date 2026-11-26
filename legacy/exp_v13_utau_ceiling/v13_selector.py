"""Phase 9 (v13) UTAU-Grade Mandarin Unit Selection (Phonetic Soft-Context + Fixed Join + Joint O+F)."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from exp_v13_utau_ceiling.zh_bank_v13 import V13ZhBank, V13ZhToken

# Bins covering [80 Hz, 7200 Hz] at 24 kHz / 1024-FFT (df = 23.4375 Hz)
VT_BIN_LO = 3
VT_BIN_HI = 308

# Right-edge articulation class of a Mandarin final (influences next onset coarticulation)
FINAL_CODA_CLASS: dict[str, str] = {
    "a": "open_a", "ia": "open_a", "ua": "open_a",
    "ai": "high_i", "ei": "high_i", "uai": "high_i", "ui": "high_i", "i": "high_i",
    "ao": "round_u", "iao": "round_u", "ou": "round_u", "iu": "round_u",
    "u": "round_u", "o": "round_u", "uo": "round_u", "v": "round_u",
    "e": "mid_e", "ie": "mid_e", "ve": "mid_e", "er": "retro",
    "an": "nasal_n", "ian": "nasal_n", "uan": "nasal_n", "van": "nasal_n",
    "en": "nasal_n", "in": "nasal_n", "un": "nasal_n", "vn": "nasal_n",
    "ang": "nasal_ng", "iang": "nasal_ng", "uang": "nasal_ng",
    "eng": "nasal_ng", "ing": "nasal_ng", "ong": "nasal_ng", "iong": "nasal_ng",
    "ir": "retro", "iz": "apical",
}

# Left-edge vowel onset class of a Mandarin final (influences preceding onset release F2/formants)
FINAL_HEAD_CLASS: dict[str, str] = {
    "a": "open_a", "ai": "open_a", "ao": "open_a", "an": "open_a", "ang": "open_a",
    "o": "round_u", "ou": "round_u", "ong": "round_u", "u": "round_u", "ua": "round_u",
    "uo": "round_u", "uai": "round_u", "ui": "round_u", "uan": "round_u", "un": "round_u", "uang": "round_u",
    "e": "mid_e", "ei": "mid_e", "en": "mid_e", "eng": "mid_e", "er": "mid_e",
    "i": "front_i", "ia": "front_i", "ie": "front_i", "iao": "front_i", "iu": "front_i",
    "ian": "front_i", "in": "front_i", "iang": "front_i", "ing": "front_i", "iong": "front_i",
    "v": "front_round", "ve": "front_round", "van": "front_round", "vn": "front_round",
    "ir": "retro", "iz": "apical",
}

# Articulation place/manner class of Mandarin onset
ONSET_PLACE_CLASS: dict[str, str] = {
    "b": "labial", "p": "labial", "m": "labial", "f": "labial",
    "d": "alveolar_stop", "t": "alveolar_stop", "n": "sonorant_alv", "l": "sonorant_alv",
    "z": "alveolar_sib", "c": "alveolar_sib", "s": "alveolar_sib",
    "zh": "retroflex", "ch": "retroflex", "sh": "retroflex", "r": "retroflex",
    "j": "palatal", "q": "palatal", "x": "palatal",
    "g": "velar", "k": "velar", "h": "velar",
    "": "zero", "SIL": "sil",
}


def _vt_shape_rms(env_a: np.ndarray, env_b: np.ndarray) -> float:
    """Mean-removed vocal-tract [80, 7200] Hz log-envelope RMS difference."""
    a = env_a[VT_BIN_LO:VT_BIN_HI].astype(np.float64)
    b = env_b[VT_BIN_LO:VT_BIN_HI].astype(np.float64)
    a_n = a - float(np.mean(a))
    b_n = b - float(np.mean(b))
    d = a_n - b_n
    return float(math.sqrt(float(np.mean(d * d))))


def _soft_prev_final_cost(cand_prev_f: str, target_prev_f: str) -> float:
    if cand_prev_f == target_prev_f:
        return 0.0
    if cand_prev_f == "SIL" or target_prev_f == "SIL":
        return 1.2
    c1 = FINAL_CODA_CLASS.get(cand_prev_f)
    c2 = FINAL_CODA_CLASS.get(target_prev_f)
    if c1 is not None and c1 == c2:
        return 0.45
    return 1.25


def _soft_next_onset_cost(cand_next_o: str, target_next_o: str) -> float:
    if cand_next_o == target_next_o:
        return 0.0
    if cand_next_o == "SIL" or target_next_o == "SIL":
        return 1.2
    c1 = ONSET_PLACE_CLASS.get(cand_next_o)
    c2 = ONSET_PLACE_CLASS.get(target_next_o)
    if c1 is not None and c1 == c2:
        return 0.45
    return 1.25


def _tone_phonation_cost(cand_tone: int, target_tone: int) -> float:
    if cand_tone == target_tone:
        return 0.0
    # T3 creak/dip vs T4 sharp fall, or T5 neutral reduction vs full tone differ most spectrally
    if (cand_tone == 5) != (target_tone == 5):
        return 3.2
    if {cand_tone, target_tone} == {3, 4}:
        return 2.9
    return 2.0


def score_zh_candidate_v13(
    cand: V13ZhToken,
    *,
    target_speaker: str,
    target_tone: int,
    prev_final: str,
    next_onset: str,
    target_dur_ms: float,
    target_f0_st: float | None = None,
    target_onset_ms: float | None = None,
    target_pos: str | None = None,
) -> float:
    """Compute target cost for a Mandarin syllable token (lower is better)."""
    tc = 0.0 if cand.speaker == target_speaker else 4.5
    tc += _tone_phonation_cost(int(cand.tone), int(target_tone))
    tc += _soft_prev_final_cost(cand.prev_final, prev_final)
    tc += _soft_next_onset_cost(cand.next_onset, next_onset)

    is_uv_zh = cand.onset in ("f", "s", "sh", "x", "h", "c", "ch", "q")
    if is_uv_zh and cand.uv_hf_lf < -4.0:
        tc += 2.6

    tc += 0.75 * abs(math.log(max(20.0, cand.dur_ms) / max(20.0, target_dur_ms)))

    if target_onset_ms is not None and cand.f_split > 0 and target_onset_ms > 5.0:
        cand_on_ms = cand.f_split * 5.0
        tc += 0.40 * abs(math.log(max(10.0, cand_on_ms) / max(10.0, target_onset_ms)))

    if target_f0_st is not None:
        tc += 0.22 * abs(float(cand.mean_f0_st) - float(target_f0_st))

    if target_pos is not None and cand.syllable_pos != target_pos:
        if cand.syllable_pos == "final" or target_pos == "final":
            tc += 0.45
        else:
            tc += 0.18
    return tc


def viterbi_zh_v13(
    slots: list[tuple[list[V13ZhToken], list[float], bool, bool]],
    beam_k: int = 12,
) -> list[V13ZhToken]:
    """UTAU-grade Mandarin Viterbi selection fixing obs_prev bug + adding consecutive chaining."""
    if not slots:
        return []
    pruned: list[tuple[list[V13ZhToken], list[float], bool, bool]] = []
    for cands, tcosts, p_bef, obs_cur in slots:
        if len(cands) > beam_k:
            order = sorted(range(len(cands)), key=lambda i: tcosts[i])[:beam_k]
            pruned.append(([cands[i] for i in order], [tcosts[i] for i in order], p_bef, obs_cur))
        else:
            pruned.append((cands, tcosts, p_bef, obs_cur))

    dp_prev = np.array(pruned[0][1], dtype=np.float64)
    bps: list[np.ndarray] = []
    for t in range(1, len(pruned)):
        c_prev = pruned[t - 1][0]
        c_cur, tc_cur, p_bef, obs_cur = pruned[t]
        # Never zero join weight solely because previous syllable had an obstruent onset!
        w_join = 0.0 if p_bef else (0.35 if obs_cur else 0.85)
        dp_cur = np.empty(len(c_cur), dtype=np.float64)
        bp_t = np.zeros(len(c_cur), dtype=np.int32)
        for j, cj in enumerate(c_cur):
            best_v, best_i = float("inf"), 0
            for i, ci in enumerate(c_prev):
                if ci.utterance_id == cj.utterance_id and cj.syl_idx == ci.syl_idx + 1:
                    # Natural consecutive recording run preserves authentic coarticulation!
                    jcost = -1.60
                elif w_join > 0.0:
                    spec_d = _vt_shape_rms(ci.edge_right_env, cj.edge_left_env)
                    db_d = abs(ci.edge_right_db - cj.edge_left_db)
                    f0_d = abs(ci.edge_f0_r - cj.edge_f0_l)
                    jcost = w_join * (1.10 * spec_d + 0.08 * db_d + 0.10 * f0_d)
                else:
                    jcost = 0.0
                v = dp_prev[i] + jcost + tc_cur[j]
                if v < best_v:
                    best_v, best_i = v, i
            dp_cur[j], bp_t[j] = best_v, best_i
        bps.append(bp_t)
        dp_prev = dp_cur

    idx = int(np.argmin(dp_prev))
    chosen = [pruned[-1][0][idx]]
    for t in range(len(bps) - 1, -1, -1):
        idx = int(bps[t][idx])
        chosen.append(pruned[t][0][idx])
    chosen.reverse()
    return chosen


def select_best_onset_final_pair(
    onset_cands: list[V13ZhToken],
    final_cands: list[V13ZhToken],
    *,
    onset: str,
    final: str,
    tone: int,
    prev_final: str,
    next_onset: str,
    target_speaker: str,
    target_dur_ms: float,
    top_k: int = 8,
) -> tuple[V13ZhToken | None, V13ZhToken | None]:
    """Jointly pick (onset, final) minimizing target cost + CV coarticulation + boundary spectral jump."""
    target_f_head = FINAL_HEAD_CLASS.get(final)
    target_o_place = ONSET_PLACE_CLASS.get(onset)
    is_uv_o = onset in ("f", "s", "sh", "x", "h", "c", "ch", "q")

    def _score_o(c: V13ZhToken) -> float:
        s = 0.0 if c.speaker == target_speaker else 4.0
        s += _soft_prev_final_cost(c.prev_final, prev_final)
        # c.next_onset stores the original final of the syllable this onset came from!
        if c.next_onset == final:
            s += 0.0
        elif target_f_head is not None and FINAL_HEAD_CLASS.get(c.next_onset) == target_f_head:
            s += 0.35
        else:
            s += 1.35
        if c.dur_ms < 25.0:
            s += 2.0
        if is_uv_o and c.uv_hf_lf < -4.0:
            s += 2.5
        return s

    def _score_f(c: V13ZhToken) -> float:
        s = 0.0 if c.speaker == target_speaker else 4.0
        s += _tone_phonation_cost(int(c.tone), int(tone))
        # c.prev_final stores the original onset (or prev_final) preceding this rime
        if c.prev_final == onset:
            s += 0.0
        elif target_o_place is not None and ONSET_PLACE_CLASS.get(c.prev_final) == target_o_place:
            s += 0.35
        else:
            s += 1.25
        s += _soft_next_onset_cost(c.next_onset, next_onset)
        s += 0.60 * abs(math.log(max(20.0, c.dur_ms) / max(20.0, 0.72 * target_dur_ms)))
        return s

    top_o = sorted(onset_cands, key=_score_o)[:top_k] if onset_cands else []
    top_f = sorted(final_cands, key=_score_f)[:top_k] if final_cands else []
    if not top_o and not top_f:
        return None, None
    if not top_o:
        return None, top_f[0]
    if not top_f:
        return top_o[0], None

    best_score = float("inf")
    best_pair = (top_o[0], top_f[0])
    for co in top_o:
        so = _score_o(co)
        for cf_tok in top_f:
            sf = _score_f(cf_tok)
            if co.utterance_id == cf_tok.utterance_id and co.syl_idx == cf_tok.syl_idx:
                j_cv = -1.0
            else:
                spec_d = _vt_shape_rms(co.edge_right_env, cf_tok.edge_left_env)
                db_d = abs(co.edge_right_db - cf_tok.edge_left_db)
                # Voiced onset/rime joins weight spectral continuity heavily; obstruents focus on F2 match
                w_cv = 0.25 if onset in ("b", "p", "d", "t", "g", "k", "f", "s", "sh", "x", "h", "z", "c", "zh", "ch", "j", "q") else 0.75
                j_cv = w_cv * (spec_d + 0.06 * db_d)
            tot = so + sf + j_cv
            if tot < best_score:
                best_score = tot
                best_pair = (co, cf_tok)
    return best_pair
