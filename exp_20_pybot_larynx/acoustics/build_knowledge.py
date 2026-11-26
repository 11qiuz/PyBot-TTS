"""Build zero-leakage explicit cepstral/BAP knowledge base strictly from TRAIN."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import (  # noqa: E402
    DEV_60_SEED,
    HOP,
    KNOWLEDGE_PATH,
    KNOWLEDGE_SUMMARY_PATH,
    N_BAP,
    N_CEP,
    N_FFT,
    N_FINAL_BINS,
    N_ONSET_BINS,
    SR,
    TRAIN_STAGE0_N,
)
from exp_20_pybot_larynx.acoustics.knowledge import (  # noqa: E402
    get_splits,
    load_v15_index_cached,
    normalize_of,
)
from exp_20_pybot_larynx.arbiter import (  # noqa: E402
    classify_final_group,
    classify_onset_place,
)
from exp_20_pybot_larynx.larynx.synth import log_env_to_cep  # noqa: E402


def _bin_segment(arr: np.ndarray, k: int) -> np.ndarray:
    """Aggregate `(M, ...)` segment frames into `k` evenly spaced phase bins."""
    m = len(arr)
    if m == 0:
        raise ValueError("Cannot bin empty segment")
    if m == 1:
        return np.repeat(arr, k, axis=0)
    if m >= k:
        groups = np.array_split(np.arange(m), k)
        return np.stack([np.mean(arr[g], axis=0) for g in groups], axis=0)
    src_t = np.linspace(0.0, 1.0, m)
    dst_t = np.linspace(0.0, 1.0, k)
    if arr.ndim == 1:
        return np.interp(dst_t, src_t, arr)
    cols = [np.interp(dst_t, src_t, arr[:, j]) for j in range(arr.shape[1])]
    return np.stack(cols, axis=1)


def _extract_one_utt(
    task: tuple[str, str, list[tuple[str, str, int, int, int, int]]],
) -> dict[str, Any]:
    """Worker extracting cepstral + BAP phase profiles from one TRAIN utterance."""
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

    active_mask = np.zeros(n_fr, dtype=bool)
    onset_entries: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, float, float]] = []
    final_entries: list[tuple[str, str, str, int, np.ndarray, np.ndarray, np.ndarray]] = []

    for raw_o, raw_f, tone, start_fr, split_abs_fr, end_fr in syls:
        o, f = normalize_of(raw_o, raw_f)
        t = int(tone)
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
            onset_entries.append((o, classify_final_group(f), on_c, on_b, on_v, o_frac, o_ms))
            f_a = sp
        else:
            f_a = a

        if b > f_a:
            fi_c = _bin_segment(cep64[f_a:b], N_FINAL_BINS)
            fi_b = _bin_segment(bap6[f_a:b], N_FINAL_BINS)
            fi_v = _bin_segment(v_f64[f_a:b], N_FINAL_BINS)
            final_entries.append((o, classify_onset_place(o), f, t, fi_c, fi_b, fi_v))

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


class _AccumTable:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.cep: dict[str, np.ndarray] = {}
        self.bap: dict[str, np.ndarray] = {}
        self.vp: dict[str, np.ndarray] = {}
        self.cnt: dict[str, int] = defaultdict(int)

    def add(self, key: str, c: np.ndarray, b: np.ndarray, v: np.ndarray) -> None:
        if key not in self.cep:
            self.cep[key] = np.array(c, dtype=np.float64, copy=True)
            self.bap[key] = np.array(b, dtype=np.float64, copy=True)
            self.vp[key] = np.array(v, dtype=np.float64, copy=True)
        else:
            self.cep[key] += c
            self.bap[key] += b
            self.vp[key] += v
        self.cnt[key] += 1

    def export(self, prefix: str) -> dict[str, np.ndarray]:
        keys = sorted(self.cnt.keys())
        n = len(keys)
        if n == 0:
            return {
                f"{prefix}_k": np.array([], dtype="<U1"),
                f"{prefix}_cep": np.zeros((0, *self.shape, N_CEP), dtype=np.float32),
                f"{prefix}_bap": np.zeros((0, *self.shape, N_BAP), dtype=np.float32),
                f"{prefix}_vp": np.zeros((0, *self.shape), dtype=np.float32),
                f"{prefix}_cnt": np.zeros((0,), dtype=np.int32),
            }
        max_len = max(len(k) for k in keys)
        k_arr = np.array(keys, dtype=f"<U{max_len}")
        c_arr = np.stack([self.cep[k] / self.cnt[k] for k in keys], axis=0).astype(np.float32)
        b_arr = np.stack([self.bap[k] / self.cnt[k] for k in keys], axis=0).astype(np.float32)
        v_arr = np.stack([self.vp[k] / self.cnt[k] for k in keys], axis=0).astype(np.float32)
        n_arr = np.array([self.cnt[k] for k in keys], dtype=np.int32)
        return {
            f"{prefix}_k": k_arr,
            f"{prefix}_cep": c_arr,
            f"{prefix}_bap": b_arr,
            f"{prefix}_vp": v_arr,
            f"{prefix}_cnt": n_arr,
        }


def build_stage0_knowledge(
    n_train: int = TRAIN_STAGE0_N,
    seed: int = DEV_60_SEED,
    out_path: Path = KNOWLEDGE_PATH,
    summary_path: Path = KNOWLEDGE_SUMMARY_PATH,
    max_workers: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    train_ids, dev_60, held_ids = get_splits()
    idx = load_v15_index_cached()
    metas = idx["train_utt_metas"]

    chosen = sorted(random.Random(seed).sample(train_ids, min(n_train, len(train_ids))))
    assert not (set(chosen) & set(dev_60)), "Leakage into DEV_60"
    assert not (set(chosen) & set(held_ids)), "Leakage into HELDOUT_20"

    tokens = idx["tokens"]
    by_utt_syl = idx["by_utt_syl"]
    tasks = []
    for uid in chosen:
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

    workers = max_workers or min(12, os.cpu_count() or 4)
    print(f"Building Stage 0 acoustic knowledge from {len(tasks)} TRAIN utts (workers={workers})...", flush=True)

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
    utt_log_refs: list[float] = []
    utt_f0_meds: list[float] = []
    sil_cep_sum = np.zeros(N_CEP, dtype=np.float64)
    sil_bap_sum = np.zeros(N_BAP, dtype=np.float64)
    sil_tot = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx_done, res in enumerate(ex.map(_extract_one_utt, tasks, chunksize=16), 1):
            utt_log_refs.append(res["utt_log_ref"])
            utt_f0_meds.append(res["utt_f0_med"])
            if res["sil_cnt"] > 0:
                sil_cep_sum += res["sil_cep"] * res["sil_cnt"]
                sil_bap_sum += res["sil_bap"] * res["sil_cnt"]
                sil_tot += res["sil_cnt"]

            for o, fg, c, b, v, o_frac, o_ms in res["onset_entries"]:
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

            for o, place, f, tone, c, b, v in res["final_entries"]:
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

            if idx_done % 300 == 0 or idx_done == len(tasks):
                print(f"  processed {idx_done}/{len(tasks)} ({time.perf_counter() - t0:.1f}s)", flush=True)

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

    summary = {
        "n_train_utts": len(tasks),
        "n_dev_60": len(dev_60),
        "n_heldout_20": len(held_ids),
        "build_time_s": round(time.perf_counter() - t0, 2),
        "npz_size_kb": round(size_kb, 1),
        "speaker_f0_med_hz": round(float(payload["speaker_f0_med_hz"][0]), 2),
        "ref_log_amp": round(float(payload["ref_log_amp"][0]), 4),
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
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path} ({size_kb:.1f} KB) in {summary['build_time_s']}s", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=TRAIN_STAGE0_N)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    build_stage0_knowledge(n_train=args.n_train, max_workers=args.workers)
