"""Fluent Multi-Sample Viterbi Selector with Unbroken Chaining & Tonal Coarticulation (`exp_v14_fluent_zh_tts`)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from exp_v8_clarity.utau_concat import OBSTRUENT_ONSETS
from exp_v13_utau_ceiling.v13_selector import (
    FINAL_CODA_CLASS,
    ONSET_PLACE_CLASS,
    select_best_onset_final_pair,
)
from exp_v14_fluent_zh_tts.bank_v14 import V14ZhBank, V14ZhToken


def _soft_prev_final_cost_v14(cand_prev_f: str, target_prev_f: str) -> float:
    if cand_prev_f == target_prev_f:
        return 0.0
    if cand_prev_f == "SIL" or target_prev_f == "SIL":
        return 1.1
    c1 = FINAL_CODA_CLASS.get(cand_prev_f)
    c2 = FINAL_CODA_CLASS.get(target_prev_f)
    if c1 is not None and c1 == c2:
        return 0.35
    return 1.1


def _soft_next_onset_cost_v14(cand_next_o: str, target_next_o: str) -> float:
    if cand_next_o == target_next_o:
        return 0.0
    if cand_next_o == "SIL" or target_next_o == "SIL":
        return 1.1
    c1 = ONSET_PLACE_CLASS.get(cand_next_o)
    c2 = ONSET_PLACE_CLASS.get(target_next_o)
    if c1 is not None and c1 == c2:
        return 0.35
    return 1.1


def _tone_cost_v14(cand_tone: int, target_tone: int) -> float:
    if cand_tone == target_tone:
        return 0.0
    if (cand_tone == 5) != (target_tone == 5):
        return 5.0
    return 3.8


def score_zh_candidate_v14(
    cand: V14ZhToken,
    *,
    target_tone: int,
    prev_final: str,
    next_onset: str,
    prev_tone: int,
    next_tone: int,
    target_dur_ms: float,
    target_f0_3pt: tuple[float, float, float] | list[float] | None = None,
    target_f0_mid_st: float | None = None,
    target_pos: str | None = None,
) -> float:
    """Compute target cost for candidate token (lower is better)."""
    tc = _tone_cost_v14(int(cand.tone), int(target_tone))
    tc += _soft_prev_final_cost_v14(cand.prev_final, prev_final)
    tc += _soft_next_onset_cost_v14(cand.next_onset, next_onset)

    # Tonal coarticulation rewards matching adjacent tones
    if prev_tone > 0 and int(cand.prev_tone) == int(prev_tone):
        tc -= 0.35
    if next_tone > 0 and int(cand.next_tone) == int(next_tone):
        tc -= 0.35

    is_uv_zh = cand.onset in ("f", "s", "sh", "x", "h", "c", "ch", "q")
    if is_uv_zh and cand.uv_hf_lf < -4.0:
        tc += 2.6

    tc += 1.15 * abs(math.log(max(20.0, cand.dur_ms) / max(20.0, target_dur_ms)))

    if target_f0_3pt is not None and len(target_f0_3pt) == 3:
        ts, tm, te = float(target_f0_3pt[0]), float(target_f0_3pt[1]), float(target_f0_3pt[2])
        tc += 0.28 * abs(cand.f0_mid_st - tm)
        tc += 0.15 * (abs(cand.f0_start_st - ts) + abs(cand.f0_end_st - te))
    elif target_f0_mid_st is not None:
        tc += 0.28 * abs(cand.f0_mid_st - float(target_f0_mid_st))

    if target_pos is not None and cand.syllable_pos != target_pos:
        tc += 0.45

    tc += float(cand.quality_penalty)
    return tc


def _viterbi_chain_v14(
    slots: list[tuple[list[V14ZhToken], list[float], bool, bool]],
    *,
    chain_bonus: float = -2.60,
    top_k: int = 32,
) -> list[V14ZhToken]:
    """Pruned-with-chain-preservation Viterbi DP across contiguous candidate slots.

    `slots` is a list of `(cands, target_costs, pause_before, cur_is_obstruent)`.
    Always retains direct next-syllable continuations from the previous slot's kept set
    so no genuine corpus bigram/trigram is ever pruned away before the `-2.60` chain bonus.
    """
    if not slots:
        return []
    n_slots = len(slots)

    # Step 1: prune each slot to top_k + all direct chains from previous kept candidates
    kept_cands: list[list[V14ZhToken]] = []
    kept_tcosts: list[np.ndarray] = []

    for i, (cands, tcosts, pause_bef, _is_obs) in enumerate(slots):
        order = sorted(range(len(cands)), key=lambda idx: tcosts[idx])
        chosen_idx: list[int] = list(order[:top_k])
        chosen_set = set(chosen_idx)

        if i > 0 and not pause_bef:
            chain_targets = {
                (pk.utterance_id, pk.syl_idx + 1)
                for pk in kept_cands[i - 1]
                if pk.pause_after_ms < 20.0
            }
            if chain_targets:
                for idx_c, c in enumerate(cands):
                    if idx_c not in chosen_set and (c.utterance_id, c.syl_idx) in chain_targets:
                        chosen_idx.append(idx_c)
                        chosen_set.add(idx_c)

        kept_cands.append([cands[idx_c] for idx_c in chosen_idx])
        kept_tcosts.append(np.array([tcosts[idx_c] for idx_c in chosen_idx], dtype=np.float64))

    if n_slots == 1:
        best_0 = int(np.argmin(kept_tcosts[0]))
        return [kept_cands[0][best_0]]

    # Step 2: fast vectorized Viterbi over pruned candidate sets
    dp = kept_tcosts[0].copy()
    back: list[np.ndarray] = []

    for i in range(1, n_slots):
        prev_list = kept_cands[i - 1]
        cur_list = kept_cands[i]
        pause_bef = slots[i][2]
        cur_is_obs = slots[i][3]

        n_prev = len(prev_list)
        n_cur = len(cur_list)
        join_mat = np.zeros((n_prev, n_cur), dtype=np.float64)

        if not pause_bef:
            E_prev = np.stack([t.edge_end_vt.astype(np.float64) for t in prev_list], axis=0)
            E_cur = np.stack([t.edge_start_vt.astype(np.float64) for t in cur_list], axis=0)
            # RMS difference across VT bins
            diff2 = (
                np.sum(E_prev * E_prev, axis=1, keepdims=True)
                + np.sum(E_cur * E_cur, axis=1, keepdims=True).T
                - 2.0 * (E_prev @ E_cur.T)
            ) / float(E_prev.shape[1])
            vt_dist = np.sqrt(np.maximum(diff2, 0.0))

            w_vt = 0.25 if cur_is_obs else 0.75
            join_mat += np.clip(w_vt * vt_dist, 0.0, 5.5)

            # Edge F0 & energy continuity (masked where either edge is unvoiced / silent)
            for ip, pt in enumerate(prev_list):
                p_voi_end = bool(np.any(pt.voiced[-2:])) if len(pt.voiced) >= 2 else False
                for ic, ct in enumerate(cur_list):
                    if (
                        pt.utterance_id == ct.utterance_id
                        and ct.syl_idx == pt.syl_idx + 1
                        and pt.pause_after_ms < 20.0
                    ):
                        # Contiguous utterance span -> zero seam penalty + chaining bonus!
                        join_mat[ip, ic] = chain_bonus
                        continue

                    c_voi_start = bool(np.any(ct.voiced[:2])) if len(ct.voiced) >= 2 else False
                    if (not cur_is_obs) and p_voi_end and c_voi_start:
                        join_mat[ip, ic] += 0.22 * abs(pt.edge_f0_r - ct.edge_f0_l)
                    join_mat[ip, ic] += 0.12 * abs(pt.parseval_p75_db - ct.parseval_p75_db)

        tot = dp[:, None] + join_mat
        bp = np.argmin(tot, axis=0)
        dp = tot[bp, np.arange(n_cur)] + kept_tcosts[i]
        back.append(bp)

    idx = int(np.argmin(dp))
    out: list[V14ZhToken] = [kept_cands[-1][idx]]
    for i in range(n_slots - 2, -1, -1):
        idx = int(back[i][idx])
        out.append(kept_cands[i][idx])
    out.reverse()
    return out


def select_fluent_zh_sequence(
    parsed_units: list[tuple[str, str, int]],
    unit_dur_ms: list[float],
    unit_pause_before: list[bool],
    unit_f0_3pt: list[tuple[float, float, float] | None],
    unit_f0_mid_st: list[float | None],
    unit_pos: list[str],
    bank: V14ZhBank,
) -> list[tuple[str, V14ZhToken | None]]:
    """Select whole-syllable Mandarin tokens via pruned chain-preserving Viterbi."""
    n_u = len(parsed_units)
    slots: list[tuple[list[V14ZhToken], list[float], bool, bool]] = []
    slot_indices: list[int] = []
    selected: dict[int, V14ZhToken | None] = {}

    for idx, (o, f, tone) in enumerate(parsed_units):
        prev_f = parsed_units[idx - 1][1] if (idx > 0 and not unit_pause_before[idx]) else "SIL"
        prev_t = parsed_units[idx - 1][2] if (idx > 0 and not unit_pause_before[idx]) else 0
        next_o = parsed_units[idx + 1][0] if idx + 1 < n_u else "SIL"
        next_t = parsed_units[idx + 1][2] if idx + 1 < n_u else 0

        cands = bank.zh_syllables.get((o, f), [])
        p_bef = unit_pause_before[idx]
        if cands:
            tcosts = [
                score_zh_candidate_v14(
                    c,
                    target_tone=tone,
                    prev_final=prev_f,
                    next_onset=next_o,
                    prev_tone=prev_t,
                    next_tone=next_t,
                    target_dur_ms=unit_dur_ms[idx],
                    target_f0_3pt=unit_f0_3pt[idx],
                    target_f0_mid_st=unit_f0_mid_st[idx],
                    target_pos=unit_pos[idx],
                )
                for c in cands
            ]
            slots.append((cands, tcosts, p_bef, o in OBSTRUENT_ONSETS))
            slot_indices.append(idx)
        else:
            if slots:
                chosen = _viterbi_chain_v14(slots)
                for si, tk in zip(slot_indices, chosen):
                    selected[si] = tk
                slots, slot_indices = [], []
            selected[idx] = None

    if slots:
        chosen = _viterbi_chain_v14(slots)
        for si, tk in zip(slot_indices, chosen):
            selected[si] = tk

    out: list[tuple[str, V14ZhToken | None]] = []
    for idx in range(n_u):
        tk = selected.get(idx)
        if tk is not None:
            out.append(("zh_target_syl", tk))
        else:
            out.append(("zh_concat_of", None))
    return out


def fallback_onset_final_pair_v14(
    bank: V14ZhBank,
    *,
    onset: str,
    final: str,
    tone: int,
    prev_final: str,
    next_onset: str,
    target_dur_ms: float,
) -> tuple[V14ZhToken | None, V14ZhToken | None]:
    """Select compatible `(onset, final)` donor pair when exact `(o, f)` syllable is absent."""
    oc = bank.zh_onsets.get(onset, []) if onset else []
    fc = bank.zh_finals.get(final, [])
    if onset and oc and fc:
        return select_best_onset_final_pair(
            oc,
            fc,
            onset=onset,
            final=final,
            tone=tone,
            prev_final=prev_final,
            next_onset=next_onset,
            target_speaker="SSB0502",
            target_dur_ms=target_dur_ms,
        )
    bo = oc[0] if oc else None
    bf = min(fc, key=lambda c: 0.0 if int(c.tone) == int(tone) else 2.5) if fc else None
    return bo, bf
