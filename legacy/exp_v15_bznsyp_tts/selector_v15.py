"""Fast Bigram-Preseeded Vectorized Viterbi Selector (`BZNSYP` ~100k Tokens, `< 15 ms/utt`)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from exp_v8_clarity.utau_concat import OBSTRUENT_ONSETS
from exp_v13_utau_ceiling.v13_selector import FINAL_CODA_CLASS, ONSET_PLACE_CLASS
from exp_v15_bznsyp_tts.index_bznsyp_v15 import V15BzToken

_HF_CLAMP_ONSETS = frozenset({"f", "s", "sh", "x", "h", "c", "ch", "q", "p", "t", "k"})


def _soft_prev_final_v15(cand_prev_f: str, target_prev_f: str) -> float:
    if cand_prev_f == target_prev_f:
        return 0.0
    if cand_prev_f == "SIL" or target_prev_f == "SIL":
        return 1.1
    c1 = FINAL_CODA_CLASS.get(cand_prev_f)
    c2 = FINAL_CODA_CLASS.get(target_prev_f)
    return 0.35 if (c1 is not None and c1 == c2) else 1.1


def _soft_next_onset_v15(cand_next_o: str, target_next_o: str) -> float:
    if cand_next_o == target_next_o:
        return 0.0
    if cand_next_o == "SIL" or target_next_o == "SIL":
        return 1.1
    c1 = ONSET_PLACE_CLASS.get(cand_next_o)
    c2 = ONSET_PLACE_CLASS.get(target_next_o)
    return 0.35 if (c1 is not None and c1 == c2) else 1.1


def score_bznsyp_candidate_v15(
    tk: V15BzToken,
    *,
    target_tone: int,
    prev_final: str,
    next_onset: str,
    prev_tone: int,
    next_tone: int,
    target_dur_ms: float,
    target_f0_3pt: tuple[float, float, float] | list[float] | None = None,
    target_f0_mid_st: float | None = None,
    target_break: int = 0,
    target_pos: str | None = None,
    use_hf_clarity_guard: bool = True,
) -> float:
    """Compute scalar target cost for a `V15BzToken` candidate (lower is better)."""
    if tk.tone == target_tone:
        tc = 0.0
    elif (tk.tone == 5) != (target_tone == 5):
        tc = 5.0
    else:
        tc = 3.8

    tc += _soft_prev_final_v15(tk.prev_final, prev_final)
    tc += _soft_next_onset_v15(tk.next_onset, next_onset)
    if prev_tone > 0 and tk.prev_tone == prev_tone:
        tc -= 0.35
    if next_tone > 0 and tk.next_tone == next_tone:
        tc -= 0.35

    if target_break >= 2 and tk.prosodic_break >= 2:
        tc -= 0.30
    elif target_break == 0 and tk.prosodic_break >= 3:
        tc += 0.45

    if use_hf_clarity_guard and tk.onset in _HF_CLAMP_ONSETS:
        if tk.uv_hf_lf < -3.5:
            tc += 3.0
        elif tk.uv_hf_lf > 0.0:
            tc -= min(0.4, 0.05 * tk.uv_hf_lf)

    tc += 1.10 * abs(math.log(max(20.0, tk.dur_ms) / max(20.0, target_dur_ms)))

    if target_f0_3pt is not None and len(target_f0_3pt) == 3:
        ts, tm, te = float(target_f0_3pt[0]), float(target_f0_3pt[1]), float(target_f0_3pt[2])
        tc += 0.28 * abs(tk.f0_mid_st - tm)
        tc += 0.15 * (abs(tk.f0_start_st - ts) + abs(tk.f0_end_st - te))
    elif target_f0_mid_st is not None:
        tc += 0.28 * abs(tk.f0_mid_st - float(target_f0_mid_st))

    if target_pos is not None and tk.syllable_pos != target_pos:
        tc += 0.45

    tc += tk.quality_penalty
    return tc


def select_bznsyp_sequence(
    parsed_units: list[tuple[str, str, int]],
    unit_dur_ms: list[float],
    unit_pause_before: list[bool],
    unit_f0_3pt: list[tuple[float, float, float] | None],
    unit_f0_mid_st: list[float | None],
    unit_pos: list[str],
    unit_break: list[int],
    index_doc: dict[str, Any],
    *,
    allowed_utt_ids: set[str] | frozenset[str] | None = None,
    use_ngram_preseed: bool = True,
    use_hf_clarity_guard: bool = True,
    top_k: int = 48,
    preseed_k: int = 24,
    chain_bonus: float = -3.00,
) -> list[tuple[str, V15BzToken | None]]:
    """Bigram-preseeded Viterbi selection over BZNSYP training tokens with scaling filter."""
    tokens: list[V15BzToken] = index_doc["tokens"]
    by_of: dict[tuple[str, str], list[int]] = index_doc["by_of"]
    by_oft: dict[tuple[str, str, int], list[int]] = index_doc["by_oft"]
    by_utt_syl: dict[tuple[str, int], int] = index_doc["by_utt_syl"]
    bigram_of: dict[tuple[tuple[str, str], tuple[str, str]], list[tuple[str, int]]] = index_doc["bigram_of"]
    bigram_oft: dict[tuple[tuple[str, str, int], tuple[str, str, int]], list[tuple[str, int]]] = index_doc["bigram_oft"]

    n_u = len(parsed_units)
    if n_u == 0:
        return []

    def _cost_for_tid(i: int, tid: int) -> float:
        o, f, tone = parsed_units[i]
        prev_f = parsed_units[i - 1][1] if (i > 0 and not unit_pause_before[i]) else "SIL"
        prev_t = parsed_units[i - 1][2] if (i > 0 and not unit_pause_before[i]) else 0
        next_o = parsed_units[i + 1][0] if i + 1 < n_u else "SIL"
        next_t = parsed_units[i + 1][2] if i + 1 < n_u else 0
        return score_bznsyp_candidate_v15(
            tokens[tid],
            target_tone=tone,
            prev_final=prev_f,
            next_onset=next_o,
            prev_tone=prev_t,
            next_tone=next_t,
            target_dur_ms=unit_dur_ms[i],
            target_f0_3pt=unit_f0_3pt[i],
            target_f0_mid_st=unit_f0_mid_st[i],
            target_break=unit_break[i],
            target_pos=unit_pos[i],
            use_hf_clarity_guard=use_hf_clarity_guard,
        )

    # Stage 1a: collect top_k per slot (exact tone first, falling back to (o, f) if < 8)
    kept_tids: list[list[int]] = []
    kept_sets: list[set[int]] = []
    tcost_cache: list[dict[int, float]] = [{} for _ in range(n_u)]

    for i, (o, f, tone) in enumerate(parsed_units):
        c_tids = by_oft.get((o, f, tone), [])
        if allowed_utt_ids is not None:
            c_tids = [tid for tid in c_tids if tokens[tid].uid in allowed_utt_ids]
        if len(c_tids) < 8:
            of_tids = by_of.get((o, f), [])
            if allowed_utt_ids is not None:
                of_tids = [tid for tid in of_tids if tokens[tid].uid in allowed_utt_ids]
            seen_s = set(c_tids)
            c_tids = list(c_tids) + [tid for tid in of_tids if tid not in seen_s]

        # Fast pre-cap if an (o,f,t) bucket has > 400 tokens (sort cleanly on all or stratified sample)
        scored = []
        for tid in c_tids:
            c = _cost_for_tid(i, tid)
            tcost_cache[i][tid] = c
            scored.append((c, tid))
        scored.sort(key=lambda x: x[0])
        top = [tid for _, tid in scored[:top_k]]
        kept_tids.append(top)
        kept_sets.append(set(top))

    # Stage 1b: O(1) bigram/multi-gram pre-seeding + chain forward retention
    if use_ngram_preseed:
        for i in range(n_u - 1):
            if unit_pause_before[i + 1]:
                continue
            k_a = parsed_units[i]
            k_b = parsed_units[i + 1]
            bg_hits = bigram_oft.get((k_a, k_b), [])
            if allowed_utt_ids is not None:
                bg_hits = [(u, s) for u, s in bg_hits if u in allowed_utt_ids]
            if len(bg_hits) < 12:
                bg_of = bigram_of.get(((k_a[0], k_a[1]), (k_b[0], k_b[1])), [])
                if allowed_utt_ids is not None:
                    bg_of = [(u, s) for u, s in bg_of if u in allowed_utt_ids]
                seen_us = set(bg_hits)
                bg_hits = list(bg_hits) + [p for p in bg_of if p not in seen_us]

            pair_scored: list[tuple[float, int, int]] = []
            for uid_p, sidx_p in bg_hits:
                ta = by_utt_syl.get((uid_p, sidx_p))
                tb = by_utt_syl.get((uid_p, sidx_p + 1))
                if ta is None or tb is None:
                    continue
                ca = tcost_cache[i].get(ta)
                if ca is None:
                    ca = _cost_for_tid(i, ta)
                    tcost_cache[i][ta] = ca
                cb = tcost_cache[i + 1].get(tb)
                if cb is None:
                    cb = _cost_for_tid(i + 1, tb)
                    tcost_cache[i + 1][tb] = cb
                pair_scored.append((ca + cb, ta, tb))
            pair_scored.sort(key=lambda x: x[0])
            for _, ta, tb in pair_scored[:preseed_k]:
                if ta not in kept_sets[i]:
                    kept_sets[i].add(ta)
                    kept_tids[i].append(ta)
                if tb not in kept_sets[i + 1]:
                    kept_sets[i + 1].add(tb)
                    kept_tids[i + 1].append(tb)

        # Retain forward chains (3-gram, 4-gram+) from kept[i-1] -> kept[i]
        for i in range(1, n_u):
            if unit_pause_before[i]:
                continue
            exp_o, exp_f, _ = parsed_units[i]
            for tid_prev in kept_tids[i - 1]:
                tk_p = tokens[tid_prev]
                if tk_p.pause_after_ms >= 25.0:
                    continue
                nxt_tid = by_utt_syl.get((tk_p.uid, tk_p.syl_idx + 1))
                if nxt_tid is None or nxt_tid in kept_sets[i]:
                    continue
                tk_n = tokens[nxt_tid]
                if tk_n.onset == exp_o and tk_n.final == exp_f:
                    if allowed_utt_ids is not None and tk_n.uid not in allowed_utt_ids:
                        continue
                    tcost_cache[i][nxt_tid] = _cost_for_tid(i, nxt_tid)
                    kept_sets[i].add(nxt_tid)
                    kept_tids[i].append(nxt_tid)

    # Stage 2: run Viterbi over contiguous non-missing spans
    chosen_tid: dict[int, int | None] = {}
    slots: list[tuple[int, list[int], list[float], bool, bool]] = []

    def _solve_span(span_slots: list[tuple[int, list[int], list[float], bool, bool]]) -> None:
        if not span_slots:
            return
        if len(span_slots) == 1:
            ui, tids_u, costs_u, _, _ = span_slots[0]
            chosen_tid[ui] = tids_u[int(np.argmin(costs_u))]
            return

        dp = np.asarray(span_slots[0][2], dtype=np.float64)
        back: list[np.ndarray] = []
        for s_idx in range(1, len(span_slots)):
            _, prev_tids, _, _, _ = span_slots[s_idx - 1]
            _, cur_tids, cur_costs, pause_bef, cur_is_obs = span_slots[s_idx]
            n_p, n_c = len(prev_tids), len(cur_tids)
            join = np.zeros((n_p, n_c), dtype=np.float64)

            if not pause_bef:
                E_p = np.stack([tokens[t].edge_end_mfcc16.astype(np.float64) for t in prev_tids], axis=0)
                E_c = np.stack([tokens[t].edge_start_mfcc16.astype(np.float64) for t in cur_tids], axis=0)
                diff2 = (
                    np.sum(E_p * E_p, axis=1, keepdims=True)
                    + np.sum(E_c * E_c, axis=1, keepdims=True).T
                    - 2.0 * (E_p @ E_c.T)
                ) / float(E_p.shape[1])
                mfcc_dist = np.sqrt(np.maximum(diff2, 0.0))
                w_m = 0.22 if cur_is_obs else 0.65
                join += np.clip(w_m * mfcc_dist, 0.0, 5.5)

                p_map = {(tokens[tid].uid, tokens[tid].syl_idx): idx_p for idx_p, tid in enumerate(prev_tids)}
                for ic, tc_id in enumerate(cur_tids):
                    tc_tok = tokens[tc_id]
                    ip_hit = p_map.get((tc_tok.uid, tc_tok.syl_idx - 1))
                    if ip_hit is not None and tokens[prev_tids[ip_hit]].pause_after_ms < 25.0:
                        join[ip_hit, ic] = chain_bonus

                for ip, tp_id in enumerate(prev_tids):
                    tp_tok = tokens[tp_id]
                    for ic, tc_id in enumerate(cur_tids):
                        if join[ip, ic] == chain_bonus:
                            continue
                        tc_tok = tokens[tc_id]
                        if not cur_is_obs:
                            join[ip, ic] += 0.22 * abs(tp_tok.f0_end_st - tc_tok.f0_start_st)
                        join[ip, ic] += 0.12 * abs(tp_tok.rel_db - tc_tok.rel_db)

            tot = dp[:, None] + join
            bp = np.argmin(tot, axis=0)
            dp = tot[bp, np.arange(n_c)] + np.asarray(cur_costs, dtype=np.float64)
            back.append(bp)

        cur_i = int(np.argmin(dp))
        chosen_tid[span_slots[-1][0]] = span_slots[-1][1][cur_i]
        for k in range(len(span_slots) - 2, -1, -1):
            cur_i = int(back[k][cur_i])
            chosen_tid[span_slots[k][0]] = span_slots[k][1][cur_i]

    for i in range(n_u):
        if kept_tids[i]:
            tids_u = kept_tids[i]
            costs_u = [tcost_cache[i][t] for t in tids_u]
            slots.append((i, tids_u, costs_u, unit_pause_before[i], parsed_units[i][0] in OBSTRUENT_ONSETS))
        else:
            _solve_span(slots)
            slots = []
            chosen_tid[i] = None
    _solve_span(slots)

    out: list[tuple[str, V15BzToken | None]] = []
    for i in range(n_u):
        tid = chosen_tid.get(i)
        if tid is not None:
            out.append(("zh_target_syl", tokens[tid]))
        else:
            out.append(("zh_concat_of", None))
    return out
