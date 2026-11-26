"""Mine Stage 5 prosody residual lookup tables strictly from TRAIN and validate on DEV_60."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx.acoustics.build_stage5_knowledge import DEFAULT_S5_CACHE_PATH  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge import get_splits, normalize_of  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import (  # noqa: E402
    STAGE5_KNOWLEDGE_PATH,
    AcousticKnowledgeStage5,
)
from exp_20_pybot_larynx.frontend.prosody_rules import predict_prosody_rules  # noqa: E402
from exp_20_pybot_larynx.frontend.prosody_stage5 import (  # noqa: E402
    PROSODY_S5_TABLES_PATH,
    ProsodyStage5Tables,
    clear_stage5_prosody_cache,
    compute_unit_context_ids,
    render_stage5_utterance,
)


def build_prosody_stage5_tables(
    cache_path: Path = DEFAULT_S5_CACHE_PATH,
    knowledge_path: Path = STAGE5_KNOWLEDGE_PATH,
    out_path: Path = PROSODY_S5_TABLES_PATH,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Mine empirical Bayes shrinkage residual tables strictly from cached TRAIN utterances."""
    t0 = time.perf_counter()
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    kb5 = AcousticKnowledgeStage5.load(knowledge_path)

    train_ids, dev_60, held_20 = get_splits()
    train_set = set(train_ids)
    assert not (train_set & set(dev_60)) and not (train_set & set(held_20))

    utt_recs = data["utt_prosody_records"]
    tri_acc: dict[tuple[int, int, int, int, int], list[np.ndarray]] = defaultdict(list)
    phr_acc: dict[tuple[int, int, int, int], list[np.ndarray]] = defaultdict(list)
    dur_acc: dict[tuple[int, int, int, int, int], list[float]] = defaultdict(list)
    ondur_acc: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    gain_acc: dict[tuple[int, int, int, int], list[float]] = defaultdict(list)
    all_res_f3: list[np.ndarray] = []
    f0_syl_records: list[tuple[tuple[int, int, int, int, int], tuple[int, int, int, int], np.ndarray]] = []

    matched_utts = 0
    matched_syls = 0
    uids = sorted(utt_recs.keys())
    if limit is not None:
        uids = uids[:limit]

    for idx_u, uid in enumerate(uids, 1):
        if uid not in train_set:
            continue
        rec = utt_recs[uid]
        syls = rec["syls"]
        try:
            pro = predict_prosody_rules(rec["text"], kb5, style="auto")
        except Exception:
            continue
        units = pro["units"]
        if len(units) != len(syls):
            continue

        matched_utts += 1
        ctxs = compute_unit_context_ids(units)

        pred_f3s = []
        real_f3s = []
        real_c0s = []
        for u, sr_row in zip(units, syls):
            p3 = np.asarray(u["f0_3pt_st"], dtype=np.float64)
            r3 = np.asarray(sr_row[9], dtype=np.float64)
            pred_f3s.append(p3)
            real_f3s.append(r3)
            real_c0s.append(float(sr_row[10]))

        real_c0_med = float(np.median(real_c0s)) if real_c0s else 0.0

        for u, c, sr_row, p3, r3 in zip(units, ctxs, syls, pred_f3s, real_f3s):
            _, _, _, _, _, real_dur, real_on, _, _, _, real_c0, _, _, _ = sr_row
            matched_syls += 1

            res_f3 = r3 - p3
            tri_key = (c["pt"], c["t"], c["nt"], c["is_wf"], c["brk_ge2"])
            phr_key = (c["t"], c["cl_pos_b"], c["ph_pos_b"], c["brk_ge2"])
            all_res_f3.append(res_f3)
            tri_acc[tri_key].append(res_f3)
            f0_syl_records.append((tri_key, phr_key, res_f3))

            pred_dur = max(60.0, float(u["duration_ms"]))
            d_log = math.log(max(30.0, float(real_dur)) / pred_dur)
            dur_acc[(c["om_id"], c["fg_id"], c["t"], c["wpos_id"], c["brk_ge2"])].append(d_log)

            if c["o"] and float(real_on) >= 5.0 and float(u["onset_ms"]) >= 5.0:
                on_log = math.log(float(real_on) / float(u["onset_ms"]))
                ondur_acc[(c["o"], c["fg"], c["brk_ge2"])].append(on_log)

            # Relative gain residual (1 dB ~ 0.115 * sqrt(513) ~ 2.6 cep0 units, or rel_db ~ 0.38 * d_cep0)
            real_rel_db = (float(real_c0) - real_c0_med) * 0.383
            dg = real_rel_db - float(u.get("gain_db", 0.0))
            gain_acc[(c["t"], c["wpos_id"], c["brk_ge2"], c["cl_pos_b"])].append(dg)

        if idx_u % 2500 == 0 or idx_u == len(uids):
            print(f"  mined prosody pass1 {idx_u}/{len(uids)} (matched={matched_utts}, {time.perf_counter() - t0:.1f}s)", flush=True)

    f0_global = (
        np.clip(np.mean(np.stack(all_res_f3, axis=0), axis=0), -1.2, 1.2).astype(np.float32)
        if all_res_f3
        else np.zeros(3, dtype=np.float32)
    )
    f0_global64 = f0_global.astype(np.float64)

    def _shrink_vec3(
        acc: dict[tuple[int, ...], list[np.ndarray]],
        k_sh: float,
        clip_v: float = 1.6,
        prior: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        ks = sorted(acc.keys())
        if not ks:
            return np.zeros((0, 0), dtype=np.int16), np.zeros((0, 3), dtype=np.float32)
        prior_vec = prior if prior is not None else np.zeros(3, dtype=np.float64)
        vals = []
        for k in ks:
            arr = np.stack(acc[k], axis=0)
            n = len(arr)
            w_s = n / (n + k_sh)
            vals.append(np.clip(w_s * np.mean(arr, axis=0) + (1.0 - w_s) * prior_vec, -clip_v, clip_v))
        return np.array(ks, dtype=np.int16), np.stack(vals, axis=0).astype(np.float32)

    # Pass 1 output: tone-tritone residual shrunk toward global F0 bias; Pass 2 mines remaining phrase residual
    f0_tri_k, f0_tri_v = _shrink_vec3(tri_acc, k_sh=20.0, clip_v=1.8, prior=f0_global64)
    f0_tri_map = {tuple(int(x) for x in k): v.astype(np.float64) for k, v in zip(f0_tri_k, f0_tri_v)}
    for tri_key, phr_key, res_f3 in f0_syl_records:
        phr_acc[phr_key].append(res_f3 - f0_tri_map.get(tri_key, f0_global64))

    def _shrink_scalar(acc: dict[tuple[Any, ...], list[float]], k_sh: float, clip_v: float) -> tuple[list[Any], np.ndarray]:
        ks = sorted(acc.keys())
        vals = []
        for k in ks:
            arr = acc[k]
            n = len(arr)
            w_s = n / (n + k_sh)
            vals.append(float(np.clip(w_s * float(np.mean(arr)), -clip_v, clip_v)))
        return ks, np.array(vals, dtype=np.float32)

    f0_phr_k, f0_phr_v = _shrink_vec3(phr_acc, k_sh=24.0, clip_v=1.6)
    dur_k, dur_v = _shrink_scalar(dur_acc, k_sh=20.0, clip_v=0.22)
    on_k, on_v = _shrink_scalar(ondur_acc, k_sh=18.0, clip_v=0.22)
    gain_k, gain_v = _shrink_scalar(gain_acc, k_sh=22.0, clip_v=2.2)

    on_k_str = np.array([f"{o}|{fg}|{b}" for (o, fg, b) in on_k], dtype="<U32") if on_k else np.array([], dtype="<U1")

    uv_list = data.get("onset_uv_list", {})
    uv_keys = sorted(uv_list.keys())
    uv_vals = np.array([float(np.clip(np.median(uv_list[k]), 0.25, 0.96)) for k in uv_keys], dtype=np.float32)

    default_weights = {
        "w_f0_tri": 0.65,
        "w_f0_phr": 0.65,
        "w_dur": 0.70,
        "w_ondur": 0.70,
        "w_gain": 0.65,
        "w_uv": 1.0,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        f0_global=f0_global,
        f0_tri_keys=f0_tri_k,
        f0_tri_vals=f0_tri_v,
        f0_phr_keys=f0_phr_k,
        f0_phr_vals=f0_phr_v,
        dur_ctx_keys=np.array(dur_k, dtype=np.int16) if dur_k else np.zeros((0, 5), dtype=np.int16),
        dur_ctx_vals=dur_v,
        on_dur_keys=on_k_str,
        on_dur_vals=on_v,
        gain_ctx_keys=np.array(gain_k, dtype=np.int16) if gain_k else np.zeros((0, 4), dtype=np.int16),
        gain_ctx_vals=gain_v,
        onset_uv_keys=np.array(uv_keys, dtype="<U8") if uv_keys else np.array([], dtype="<U1"),
        onset_uv_vals=uv_vals,
        weight_keys=np.array(list(default_weights.keys()), dtype="<U16"),
        weight_vals=np.array(list(default_weights.values()), dtype=np.float32),
        matched_utts=np.array([matched_utts], dtype=np.int32),
    )
    clear_stage5_prosody_cache()
    summary = {
        "matched_utts": matched_utts,
        "matched_syls": matched_syls,
        "f0_tri_keys": int(len(f0_tri_k)),
        "f0_phr_keys": int(len(f0_phr_k)),
        "dur_keys": int(len(dur_k)),
        "on_dur_keys": int(len(on_k)),
        "gain_keys": int(len(gain_k)),
        "onset_uv_keys": int(len(uv_keys)),
        "npz_kb": round(out_path.stat().st_size / 1024.0, 1),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    print("Built Stage 5 prosody tables:", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def tune_prosody_weights_dev60(tables_path: Path = PROSODY_S5_TABLES_PATH) -> dict[str, Any]:
    """Verify & tune individual prosody residual table weights strictly on DEV_60."""
    from exp_20_pybot_larynx.acoustics.knowledge import load_v15_index_cached
    from exp_20_pybot_larynx.acoustics.knowledge_stage2 import AcousticKnowledgeStage2
    from exp_20_pybot_larynx.arbiter import Arbiter
    from exp_20_pybot_larynx.frontend.prosody_rules import render_rule_auto_utterance
    from exp_20_pybot_larynx.judge import evaluate_utterance, summarize_condition
    from exp_20_pybot_larynx.rules import default_stage0_rules, default_stage5_rules
    from exp_20_pybot_larynx.stage0_probe import _prepare_utterance_oracle
    from exp_v15_bznsyp_tts.index_bznsyp_v15 import parse_bznsyp_corpus

    kb2 = AcousticKnowledgeStage2.load()
    kb5 = AcousticKnowledgeStage5.load()
    arbiter = Arbiter()
    rules4 = default_stage0_rules()
    rules5 = default_stage5_rules()

    cache_dev60 = Path(
        r"C:\Users\MECHREVO\.gemini\antigravity\brain\f67e0437-e98d-4cb1-bbfd-da72d02478db\scratch\dev60_oracles.pkl"
    )
    if cache_dev60.exists():
        with open(cache_dev60, "rb") as f:
            dev_oracles = pickle.load(f)
    else:
        _, dev_60_ids, _ = get_splits()
        idx_doc = load_v15_index_cached()
        corpus_by_id = {r["id"]: r for r in parse_bznsyp_corpus()[0]}
        dev_oracles = []
        for uid in dev_60_ids:
            dm = dict(idx_doc["train_utt_metas"][uid])
            dm["syllables"] = corpus_by_id[uid]["syllables"]
            dev_oracles.append(_prepare_utterance_oracle(uid, dm))

    def _eval_stage4_c4(items: list[dict[str, Any]]) -> dict[str, float]:
        rows = []
        for idx_d, item in enumerate(items):
            y, _ = render_rule_auto_utterance(
                item["text"], item["uid"], kb2, arbiter=arbiter, rules=rules4, seed=5000 + idx_d
            )
            rows.append(evaluate_utterance(y, item["ref_span"], rtf=0.03, ref_fast=item["ref_fast"]))
        return summarize_condition(rows)["medians"]

    def _eval_weights(w_dict: dict[str, float], items: list[dict[str, Any]]) -> dict[str, float]:
        tbl = ProsodyStage5Tables(tables_path)
        tbl.weights = dict(w_dict)
        rows = []
        for idx_d, item in enumerate(items):
            y, _ = render_stage5_utterance(
                item["text"],
                item["uid"],
                kb5,
                arbiter=arbiter,
                rules=rules5,
                seed=5000 + idx_d,
                tables=tbl,
            )
            rows.append(evaluate_utterance(y, item["ref_span"], rtf=0.03, ref_fast=item["ref_fast"]))
        return summarize_condition(rows)["medians"]

    s4_med = _eval_stage4_c4(dev_oracles)
    base_w = {"w_f0_tri": 0.0, "w_f0_phr": 0.0, "w_dur": 0.0, "w_ondur": 0.0, "w_gain": 0.0, "w_uv": 0.0}
    cur_med = _eval_weights(base_w, dev_oracles)
    base_med = dict(cur_med)
    print("DEV_60 Stage4_C4 baseline:", s4_med, flush=True)
    print("DEV_60 Stage5_C4 (no residual tables):", base_med, flush=True)

    # Coordinate screen on strided 20-utt DEV_60 subset, then confirm each gain on all 60 DEV_60 utts
    dev_sub = dev_oracles[::3]
    sub_cur = _eval_weights(base_w, dev_sub)
    chosen = dict(base_w)

    def _loss(m: dict[str, float]) -> float:
        return float(m["mel_lsd_db"]) + 0.14 * float(m["f0_mae_st"]) + 0.08 * float(m["mcd_db"])

    for k, cands in (
        ("w_dur", (0.45, 0.65, 0.85)),
        ("w_f0_tri", (0.30, 0.45)),
        ("w_uv", (1.0,)),
        ("w_f0_phr", (0.15, 0.30)),
        ("w_ondur", (0.25,)),
        ("w_gain", (0.20,)),
    ):
        sub_scored: list[tuple[float, float, dict[str, float]]] = []
        for c in cands:
            trial = dict(chosen)
            trial[k] = c
            m_sub = _eval_weights(trial, dev_sub)
            l_sub = _loss(m_sub)
            f0_improved = m_sub["f0_mae_st"] < sub_cur["f0_mae_st"] - 0.006 and m_sub["mel_lsd_db"] <= sub_cur["mel_lsd_db"] + 0.005
            lsd_improved = m_sub["mel_lsd_db"] < sub_cur["mel_lsd_db"] - 0.004 and m_sub["f0_mae_st"] <= sub_cur["f0_mae_st"] + 0.005
            if l_sub < _loss(sub_cur) - 0.002 or f0_improved or lsd_improved:
                sub_scored.append((l_sub, c, m_sub))

        accepted = False
        for _, cand_c, cand_sub_m in sorted(sub_scored, key=lambda x: x[0]):
            trial_full = dict(chosen)
            trial_full[k] = cand_c
            m_full = _eval_weights(trial_full, dev_oracles)
            accept_full = (
                (_loss(m_full) < _loss(cur_med) - 0.002)
                and (m_full["mel_lsd_db"] <= cur_med["mel_lsd_db"] + 0.004)
                and (m_full["f0_mae_st"] <= cur_med["f0_mae_st"] + 0.005)
            )
            if accept_full:
                chosen[k] = cand_c
                cur_med = m_full
                sub_cur = cand_sub_m
                accepted = True
                print(
                    f"  [ACCEPT {k}={cand_c}] -> lsd={cur_med['mel_lsd_db']} mcd={cur_med['mcd_db']} "
                    f"f0={cur_med['f0_mae_st']} uv={cur_med['uv_2_8k_vs_lo_db']}",
                    flush=True,
                )
                break
            print(
                f"  [REJECT {k}={cand_c} on full60] -> cand lsd={m_full['mel_lsd_db']} "
                f"mcd={m_full['mcd_db']} f0={m_full['f0_mae_st']}",
                flush=True,
            )
        if not accepted:
            print(f"  [KEEP {k}={chosen[k]}]", flush=True)

    # Write final tuned weights back into prosody_s5_tables.npz
    with np.load(tables_path, allow_pickle=False) as old_d:
        arrays = {key: old_d[key] for key in old_d.files}
    arrays["weight_keys"] = np.array(list(chosen.keys()), dtype="<U16")
    arrays["weight_vals"] = np.array(list(chosen.values()), dtype=np.float32)
    np.savez_compressed(tables_path, **arrays)
    clear_stage5_prosody_cache()
    out_summary = {"weights": chosen, "stage4_c4": s4_med, "base_medians": base_med, "tuned_medians": cur_med}
    print("Tuned Stage 5 prosody summary:", json.dumps(out_summary, ensure_ascii=False, indent=2), flush=True)
    return out_summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-path", type=Path, default=DEFAULT_S5_CACHE_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tune-dev60", action="store_true")
    args = ap.parse_args()
    if args.tune_dev60:
        tune_prosody_weights_dev60()
    else:
        build_prosody_stage5_tables(cache_path=args.cache_path, limit=args.limit)
