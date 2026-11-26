"""Build full-TRAIN zero-leakage cepstral/BAP & context timing knowledge bases."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import (  # noqa: E402
    EXP_DIR,
    N_BAP,
    N_CEP,
    N_FFT,
    N_FINAL_BINS,
    N_ONSET_BINS,
    SR,
)
from exp_20_pybot_larynx.acoustics.build_knowledge import (  # noqa: E402
    _AccumTable,
    _bin_segment,
)
from exp_20_pybot_larynx.acoustics.knowledge import (  # noqa: E402
    AcousticKnowledge,
    get_splits,
    load_v15_index_cached,
    normalize_of,
)
from exp_20_pybot_larynx.arbiter import (  # noqa: E402
    NEPER_TO_DB,
    classify_final_group,
    classify_onset_place,
)
from exp_20_pybot_larynx.larynx.synth import log_env_to_cep  # noqa: E402

FULL_KNOWLEDGE_PATH = EXP_DIR / "acoustics" / "knowledge_full.npz"
FULL_SUMMARY_PATH = EXP_DIR / "acoustics" / "knowledge_full_summary.json"
CTX_KNOWLEDGE_PATH = EXP_DIR / "acoustics" / "knowledge_ctx.npz"


def _worker_init() -> None:
    """Prevent BLAS/OpenMP thread oversubscription inside worker processes."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _extract_one_utt_full(
    task: tuple[str, str, list[tuple[str, str, int, int, int, int]]],
) -> dict[str, Any]:
    """Worker extracting cepstral + BAP phase profiles and context timing/energy stats."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from exp_v9_clarity_pro.hnm_v9 import analyze_v9
    from exp_v12_natural_vocoder.v12_bank import _patch_unsmoothed_uv_env
    from exp_v15_bznsyp_tts.index_bznsyp_v15 import read_bznsyp_wav_24k

    uid, wav_rel, syls = task
    x24 = read_bznsyp_wav_24k(REPO_ROOT / wav_rel)
    fv9, bgt8 = analyze_v9(x24, sr=SR)
    feats_p = _patch_unsmoothed_uv_env(x24, fv9, sr=SR, n_fft=N_FFT)
    bap6 = np.concatenate([feats_p.bap[:, :4], bgt8], axis=1)

    n_fr = len(feats_p.f0_hz)
    v_mask = feats_p.voiced & (feats_p.f0_hz > 50.0)
    if np.any(v_mask):
        utt_log_ref = float(np.median(np.mean(feats_p.log_env[v_mask, 1:257], axis=1)))
        utt_f0_med = float(np.median(feats_p.f0_hz[v_mask]))
    else:
        utt_log_ref = 0.0
        utt_f0_med = 247.0

    log_env_norm = feats_p.log_env - utt_log_ref
    cep64 = log_env_to_cep(log_env_norm, n_cep=N_CEP)
    v_f64 = feats_p.voiced.astype(np.float64)
    # Mean log-energy per frame (in nepers, relative to utt_log_ref) for context energy bias
    frame_rel_nep = np.mean(log_env_norm[:, 1:257], axis=1)

    active_mask = np.zeros(n_fr, dtype=bool)
    onset_entries: list[tuple[str, str, int, np.ndarray, np.ndarray, np.ndarray, float, float, float]] = []
    final_entries: list[tuple[str, str, str, int, str, np.ndarray, np.ndarray, np.ndarray, float, float]] = []

    n_syls = len(syls)
    for si, (raw_o, raw_f, tone, start_fr, split_abs_fr, end_fr) in enumerate(syls):
        o, f = normalize_of(raw_o, raw_f)
        t = int(tone)
        fg = classify_final_group(f)
        syl_pos = "initial" if si == 0 else ("final" if si == n_syls - 1 else "medial")
        a = max(0, min(n_fr - 1, int(start_fr)))
        b = max(a + 1, min(n_fr, int(end_fr)))
        sp = max(a, min(b - 1, int(split_abs_fr)))
        active_mask[a:b] = True

        if o and sp > a:
            on_c = _bin_segment(cep64[a:sp], N_ONSET_BINS)
            on_b = _bin_segment(bap6[a:sp], N_ONSET_BINS)
            on_v = _bin_segment(v_f64[a:sp], N_ONSET_BINS)
            o_frac = float(sp - a) / float(max(1, b - a))
            o_ms = float(sp - a) * 5.0
            o_uv_frac = float(np.mean(1.0 - v_f64[a:sp]))
            onset_entries.append((o, fg, t, on_c, on_b, on_v, o_frac, o_ms, o_uv_frac))
            f_a = sp
        else:
            f_a = a

        if b > f_a:
            fi_c = _bin_segment(cep64[f_a:b], N_FINAL_BINS)
            fi_b = _bin_segment(bap6[f_a:b], N_FINAL_BINS)
            fi_v = _bin_segment(v_f64[f_a:b], N_FINAL_BINS)
            f_ms = float(b - f_a) * 5.0
            f_rel_db = float(np.mean(frame_rel_nep[f_a:b])) * NEPER_TO_DB
            final_entries.append((o, classify_onset_place(o), f, t, syl_pos, fi_c, fi_b, fi_v, f_ms, f_rel_db))

    sil_mask = ~active_mask
    if np.any(sil_mask):
        sil_cep = np.mean(cep64[sil_mask], axis=0)
        sil_bap = np.mean(bap6[sil_mask], axis=0)
        sil_cnt = int(np.sum(sil_mask))
    else:
        sil_cep = np.zeros(N_CEP, dtype=np.float64)
        sil_bap = np.ones(N_BAP, dtype=np.float64) * 0.95
        sil_cnt = 0

    return {
        "uid": uid,
        "utt_log_ref": utt_log_ref,
        "utt_f0_med": utt_f0_med,
        "onset_entries": onset_entries,
        "final_entries": final_entries,
        "sil_cep": sil_cep,
        "sil_bap": sil_bap,
        "sil_cnt": sil_cnt,
    }


def _pack_onset_timing_table(
    prefix: str,
    ms_map: dict[str, list[float]],
    frac_map: dict[str, list[float]],
    uv_map: dict[str, list[float]],
) -> dict[str, np.ndarray]:
    keys = sorted(ms_map.keys())
    if not keys:
        return {
            f"{prefix}_k": np.array([], dtype="<U1"),
            f"{prefix}_ms_med": np.zeros((0,), dtype=np.float32),
            f"{prefix}_ms_p25": np.zeros((0,), dtype=np.float32),
            f"{prefix}_ms_p75": np.zeros((0,), dtype=np.float32),
            f"{prefix}_frac_med": np.zeros((0,), dtype=np.float32),
            f"{prefix}_uv_mean": np.zeros((0,), dtype=np.float32),
            f"{prefix}_cnt": np.zeros((0,), dtype=np.int32),
        }
    max_len = max(len(k) for k in keys)
    return {
        f"{prefix}_k": np.array(keys, dtype=f"<U{max_len}"),
        f"{prefix}_ms_med": np.array([float(np.median(ms_map[k])) for k in keys], dtype=np.float32),
        f"{prefix}_ms_p25": np.array([float(np.percentile(ms_map[k], 25)) for k in keys], dtype=np.float32),
        f"{prefix}_ms_p75": np.array([float(np.percentile(ms_map[k], 75)) for k in keys], dtype=np.float32),
        f"{prefix}_frac_med": np.array([float(np.median(frac_map[k])) for k in keys], dtype=np.float32),
        f"{prefix}_uv_mean": np.array([float(np.mean(uv_map[k])) for k in keys], dtype=np.float32),
        f"{prefix}_cnt": np.array([len(ms_map[k]) for k in keys], dtype=np.int32),
    }


def _pack_scalar_med_table(prefix: str, val_map: dict[str, list[float]]) -> dict[str, np.ndarray]:
    keys = sorted(val_map.keys())
    if not keys:
        return {
            f"{prefix}_k": np.array([], dtype="<U1"),
            f"{prefix}_med": np.zeros((0,), dtype=np.float32),
            f"{prefix}_cnt": np.zeros((0,), dtype=np.int32),
        }
    max_len = max(len(k) for k in keys)
    return {
        f"{prefix}_k": np.array(keys, dtype=f"<U{max_len}"),
        f"{prefix}_med": np.array([float(np.median(val_map[k])) for k in keys], dtype=np.float32),
        f"{prefix}_cnt": np.array([len(val_map[k]) for k in keys], dtype=np.int32),
    }


def build_full_knowledge(
    out_path: Path = FULL_KNOWLEDGE_PATH,
    summary_path: Path = FULL_SUMMARY_PATH,
    ctx_path: Path = CTX_KNOWLEDGE_PATH,
    max_workers: int | None = 20,
    limit: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    train_ids, dev_60, held_ids = get_splits()
    train_set = set(train_ids)
    dev_set = set(dev_60)
    held_set = set(held_ids)
    assert not (train_set & dev_set), "Leakage: TRAIN overlaps DEV_60"
    assert not (train_set & held_set), "Leakage: TRAIN overlaps HELDOUT_20"
    assert not (dev_set & held_set), "Leakage: DEV_60 overlaps HELDOUT_20"

    selected_ids = sorted(train_ids)[:limit] if limit is not None else sorted(train_ids)
    assert not (set(selected_ids) & dev_set), "Selected IDs overlap DEV_60"
    assert not (set(selected_ids) & held_set), "Selected IDs overlap HELDOUT_20"

    idx = load_v15_index_cached()
    metas = idx["train_utt_metas"]
    tokens = idx["tokens"]
    by_utt_syl = idx["by_utt_syl"]

    tasks: list[tuple[str, str, list[tuple[str, str, int, int, int, int]]]] = []
    for uid in selected_ids:
        m = metas[uid]
        syl_list: list[tuple[str, str, int, int, int, int]] = []
        for si in range(int(m["n_syllables"])):
            t_idx = by_utt_syl.get((uid, si))
            if t_idx is None:
                continue
            tok = tokens[t_idx]
            syl_list.append((
                str(tok.onset),
                str(tok.final),
                int(tok.tone),
                int(tok.start_fr),
                int(tok.start_fr + tok.split_fr),
                int(tok.end_fr),
            ))
        if syl_list:
            tasks.append((uid, m["wav_path"], syl_list))

    workers = max_workers if max_workers is not None else min(20, os.cpu_count() or 4)
    print(f"Building FULL acoustic knowledge from {len(tasks)} TRAIN utts (workers={workers})...", flush=True)

    on_grp = _AccumTable((N_ONSET_BINS,))
    on_only = _AccumTable((N_ONSET_BINS,))
    fin_oft = _AccumTable((N_FINAL_BINS,))
    fin_pft = _AccumTable((N_FINAL_BINS,))
    fin_ft = _AccumTable((N_FINAL_BINS,))
    fin_f = _AccumTable((N_FINAL_BINS,))

    st_on = _AccumTable(())
    st_ft = _AccumTable(())
    st_f = _AccumTable(())

    on_glob_c = np.zeros((N_ONSET_BINS, N_CEP), dtype=np.float64)
    on_glob_b = np.zeros((N_ONSET_BINS, N_BAP), dtype=np.float64)
    on_glob_v = np.zeros((N_ONSET_BINS,), dtype=np.float64)
    on_glob_n = 0

    fin_glob_c = np.zeros((N_FINAL_BINS, N_CEP), dtype=np.float64)
    fin_glob_b = np.zeros((N_FINAL_BINS, N_BAP), dtype=np.float64)
    fin_glob_v = np.zeros((N_FINAL_BINS,), dtype=np.float64)
    fin_glob_n = 0

    st_bap_sum = np.zeros(N_BAP, dtype=np.float64)
    st_bap_n = 0

    onset_frac_list: dict[str, list[float]] = defaultdict(list)
    onset_ms_list: dict[str, list[float]] = defaultdict(list)
    onset_uv_list: dict[str, list[float]] = defaultdict(list)

    # Context timing / voicing / energy distributions for knowledge_ctx.npz
    oft_ms: dict[str, list[float]] = defaultdict(list)
    oft_frac: dict[str, list[float]] = defaultdict(list)
    oft_uv: dict[str, list[float]] = defaultdict(list)

    ofg_ms: dict[str, list[float]] = defaultdict(list)
    ofg_frac: dict[str, list[float]] = defaultdict(list)
    ofg_uv: dict[str, list[float]] = defaultdict(list)

    ot_ms: dict[str, list[float]] = defaultdict(list)
    ot_frac: dict[str, list[float]] = defaultdict(list)
    ot_uv: dict[str, list[float]] = defaultdict(list)

    fin_dur_ft: dict[str, list[float]] = defaultdict(list)
    tone_pos_rel_db: dict[str, list[float]] = defaultdict(list)

    utt_log_refs: list[float] = []
    utt_f0_meds: list[float] = []
    sil_cep_sum = np.zeros(N_CEP, dtype=np.float64)
    sil_bap_sum = np.zeros(N_BAP, dtype=np.float64)
    sil_tot = 0

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        for idx_done, res in enumerate(ex.map(_extract_one_utt_full, tasks, chunksize=24), 1):
            utt_log_refs.append(res["utt_log_ref"])
            utt_f0_meds.append(res["utt_f0_med"])
            if res["sil_cnt"] > 0:
                sil_cep_sum += res["sil_cep"] * res["sil_cnt"]
                sil_bap_sum += res["sil_bap"] * res["sil_cnt"]
                sil_tot += res["sil_cnt"]

            for o, fg, tone, c, b, v, o_frac, o_ms, o_uv_frac in res["onset_entries"]:
                on_grp.add(f"{o}|{fg}", c, b, v)
                on_only.add(o, c, b, v)
                st_on.add(o, np.mean(c, axis=0), np.mean(b, axis=0), float(np.mean(v)))
                on_glob_c += c
                on_glob_b += b
                on_glob_v += v
                on_glob_n += 1
                st_bap_sum += np.mean(b, axis=0)
                st_bap_n += 1

                onset_frac_list[o].append(o_frac)
                onset_ms_list[o].append(o_ms)
                onset_uv_list[o].append(o_uv_frac)

                k_oft = f"{o}|{fg}|{tone}"
                oft_ms[k_oft].append(o_ms)
                oft_frac[k_oft].append(o_frac)
                oft_uv[k_oft].append(o_uv_frac)

                k_ofg = f"{o}|{fg}"
                ofg_ms[k_ofg].append(o_ms)
                ofg_frac[k_ofg].append(o_frac)
                ofg_uv[k_ofg].append(o_uv_frac)

                k_ot = f"{o}|{tone}"
                ot_ms[k_ot].append(o_ms)
                ot_frac[k_ot].append(o_frac)
                ot_uv[k_ot].append(o_uv_frac)

            for o, place, f, tone, syl_pos, c, b, v, f_ms, f_rel_db in res["final_entries"]:
                fin_oft.add(f"{o}|{f}|{tone}", c, b, v)
                fin_pft.add(f"{place}|{f}|{tone}", c, b, v)
                fin_ft.add(f"{f}|{tone}", c, b, v)
                fin_f.add(f, c, b, v)
                st_ft.add(f"{f}|{tone}", np.mean(c, axis=0), np.mean(b, axis=0), float(np.mean(v)))
                st_f.add(f, np.mean(c, axis=0), np.mean(b, axis=0), float(np.mean(v)))
                fin_glob_c += c
                fin_glob_b += b
                fin_glob_v += v
                fin_glob_n += 1
                st_bap_sum += np.mean(b, axis=0)
                st_bap_n += 1

                fin_dur_ft[f"{f}|{tone}"].append(f_ms)
                tone_pos_rel_db[f"{tone}|{syl_pos}"].append(f_rel_db)
                tone_pos_rel_db[f"{tone}|all"].append(f_rel_db)

            if idx_done % 1000 == 0 or idx_done == len(tasks):
                print(f"  processed {idx_done}/{len(tasks)} ({time.perf_counter() - t0:.1f}s)", flush=True)

    # Exact AcousticKnowledge.load() schema
    payload: dict[str, np.ndarray] = {}
    payload.update(on_grp.export("on_grp"))
    payload.update(on_only.export("on_only"))
    payload["on_glob_cep"] = (on_glob_c / max(1, on_glob_n)).astype(np.float32)
    payload["on_glob_bap"] = (on_glob_b / max(1, on_glob_n)).astype(np.float32)
    payload["on_glob_vp"] = (on_glob_v / max(1, on_glob_n)).astype(np.float32)

    payload.update(fin_oft.export("fin_oft"))
    payload.update(fin_pft.export("fin_pft"))
    payload.update(fin_ft.export("fin_ft"))
    payload.update(fin_f.export("fin_f"))
    payload["fin_glob_cep"] = (fin_glob_c / max(1, fin_glob_n)).astype(np.float32)
    payload["fin_glob_bap"] = (fin_glob_b / max(1, fin_glob_n)).astype(np.float32)
    payload["fin_glob_vp"] = (fin_glob_v / max(1, fin_glob_n)).astype(np.float32)

    payload.update(st_on.export("st_on"))
    payload.update(st_ft.export("st_ft"))
    payload.update(st_f.export("st_f"))
    payload["st_bap_global"] = (st_bap_sum / max(1, st_bap_n)).astype(np.float32)

    on_keys = sorted(onset_frac_list.keys())
    max_ok = max((len(k) for k in on_keys), default=1)
    payload["onset_frac_keys"] = np.array(on_keys, dtype=f"<U{max_ok}")
    payload["onset_frac_vals"] = np.array([float(np.median(onset_frac_list[k])) for k in on_keys], dtype=np.float32)
    payload["onset_ms_vals"] = np.array([float(np.median(onset_ms_list[k])) for k in on_keys], dtype=np.float32)

    payload["sil_cep64"] = (sil_cep_sum / max(1, sil_tot)).astype(np.float32)
    payload["sil_bap6"] = (sil_bap_sum / max(1, sil_tot)).astype(np.float32)
    payload["speaker_f0_med_hz"] = np.array([float(np.median(utt_f0_meds))], dtype=np.float32)
    payload["ref_log_amp"] = np.array([float(np.median(utt_log_refs))], dtype=np.float32)
    payload["n_train_utts"] = np.array([len(tasks)], dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    size_kb = out_path.stat().st_size / 1024.0

    # Verify schema compatibility immediately
    kb_loaded = AcousticKnowledge.load(out_path)
    assert kb_loaded.n_train_utts == len(tasks)

    # Save context knowledge base
    ctx_payload: dict[str, np.ndarray] = {}
    ctx_payload.update(_pack_onset_timing_table("on_oft", oft_ms, oft_frac, oft_uv))
    ctx_payload.update(_pack_onset_timing_table("on_ofg", ofg_ms, ofg_frac, ofg_uv))
    ctx_payload.update(_pack_onset_timing_table("on_ot", ot_ms, ot_frac, ot_uv))
    ctx_payload.update(_pack_onset_timing_table("on_o", onset_ms_list, onset_frac_list, onset_uv_list))
    ctx_payload.update(_pack_scalar_med_table("fin_dur_ft", fin_dur_ft))
    ctx_payload.update(_pack_scalar_med_table("tone_pos_rel_db", tone_pos_rel_db))
    ctx_payload["n_train_utts"] = np.array([len(tasks)], dtype=np.int32)

    ctx_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ctx_path, **ctx_payload)
    ctx_size_kb = ctx_path.stat().st_size / 1024.0

    summary = {
        "n_train_utts": len(tasks),
        "n_dev_60": len(dev_60),
        "n_heldout_20": len(held_ids),
        "build_time_s": round(time.perf_counter() - t0, 2),
        "npz_size_kb": round(size_kb, 1),
        "ctx_npz_size_kb": round(ctx_size_kb, 1),
        "speaker_f0_med_hz": round(float(payload["speaker_f0_med_hz"][0]), 2),
        "ref_log_amp": round(float(payload["ref_log_amp"][0]), 4),
        "n_onset_tokens": on_glob_n,
        "n_final_tokens": fin_glob_n,
        "table_sizes": {
            "on_grp": len(on_grp.cnt),
            "on_only": len(on_only.cnt),
            "fin_oft": len(fin_oft.cnt),
            "fin_pft": len(fin_pft.cnt),
            "fin_ft": len(fin_ft.cnt),
            "fin_f": len(fin_f.cnt),
            "st_on": len(st_on.cnt),
            "st_ft": len(st_ft.cnt),
        },
        "ctx_table_sizes": {
            "on_oft": len(oft_ms),
            "on_ofg": len(ofg_ms),
            "on_ot": len(ot_ms),
            "on_o": len(onset_ms_list),
            "fin_dur_ft": len(fin_dur_ft),
            "tone_pos_rel_db": len(tone_pos_rel_db),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path} ({size_kb:.1f} KB) and {ctx_path} ({ctx_size_kb:.1f} KB) in {summary['build_time_s']}s", flush=True)
    return summary


def benchmark_dev60(
    stage0_path: Path | None = None,
    full_path: Path = FULL_KNOWLEDGE_PATH,
) -> dict[str, Any]:
    """Benchmark C2_dynamic_of on DEV_60 ONLY comparing knowledge_stage0.npz vs knowledge_full.npz."""
    from exp_20_pybot_larynx import KNOWLEDGE_PATH
    from exp_20_pybot_larynx.acoustics.knowledge import build_base_control
    from exp_20_pybot_larynx.acoustics.knowledge_ctx import ContextKnowledge
    from exp_20_pybot_larynx.judge import evaluate_utterance, summarize_condition
    from exp_20_pybot_larynx.larynx.synth import synthesize_larynx
    from exp_20_pybot_larynx.stage0_probe import _aggregate_kb_hits, _prepare_utterance_oracle
    from exp_v15_bznsyp_tts.index_bznsyp_v15 import parse_bznsyp_corpus

    s0_path = stage0_path or KNOWLEDGE_PATH
    kb_s0 = AcousticKnowledge.load(s0_path)
    kb_full = AcousticKnowledge.load(full_path)
    ctx_kb = ContextKnowledge.load(CTX_KNOWLEDGE_PATH) if CTX_KNOWLEDGE_PATH.exists() else None

    _, dev_60_ids, _ = get_splits()
    idx_doc = load_v15_index_cached()
    corpus_records, _ = parse_bznsyp_corpus()
    corpus_by_id = {r["id"]: r for r in corpus_records}

    dev_oracles = []
    for uid in dev_60_ids:
        dev_meta = dict(idx_doc["train_utt_metas"][uid])
        dev_meta["syllables"] = corpus_by_id[uid]["syllables"]
        dev_oracles.append(_prepare_utterance_oracle(uid, dev_meta))

    results: dict[str, Any] = {
        "n_dev_60": len(dev_60_ids),
        "kb_stage0_n_utts": kb_s0.n_train_utts,
        "kb_full_n_utts": kb_full.n_train_utts,
        "ctx_loaded": ctx_kb is not None,
    }
    for label, kb in (("stage0", kb_s0), ("full", kb_full)):
        rows = []
        kb_infos = []
        for d_idx, item in enumerate(dev_oracles):
            ref_span = item["ref_span"]
            ref_fast = item["ref_fast"]
            feats_p = item["feats_p"]
            ctx = item["ctx"]
            seed_d = 5000 + d_idx

            ts = time.perf_counter()
            c2, kb_info = build_base_control(ctx, feats_p.f0_hz, feats_p.voiced, kb, mode="dynamic")
            c2.n_samples = len(ref_span)
            y2 = synthesize_larynx(c2, glottal_mode="parametric_lf", seed=seed_d)
            dt = time.perf_counter() - ts
            rows.append(evaluate_utterance(y2, ref_span, rtf=dt / max(1e-6, len(y2) / SR), ref_fast=ref_fast))
            kb_infos.append(kb_info)
        results[label] = {
            "summary": summarize_condition(rows),
            "kb_hits": _aggregate_kb_hits(kb_infos),
        }

    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on TRAIN utts for smoke testing")
    ap.add_argument("--benchmark-dev60", action="store_true", help="Benchmark C2_dynamic_of on DEV_60 only")
    args = ap.parse_args()
    if args.benchmark_dev60:
        benchmark_dev60()
    else:
        build_full_knowledge(max_workers=args.workers, limit=args.limit)
