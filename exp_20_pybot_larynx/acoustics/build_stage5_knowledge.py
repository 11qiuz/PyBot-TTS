"""Build Stage 5 high-resolution medoid + prosody-anchored acoustic knowledge base strictly from TRAIN."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import N_BAP, N_CEP, N_FFT, SR  # noqa: E402
from exp_20_pybot_larynx.acoustics.build_knowledge import _bin_segment  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge import (  # noqa: E402
    get_splits,
    load_v15_index_cached,
    normalize_of,
)
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import (  # noqa: E402
    N_FINAL_BINS_S2,
    N_ONSET_BINS_S2,
    encode_next_place,
    encode_prev_fg,
)
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import (  # noqa: E402
    N_COARSE_CEP_S5,
    STAGE5_KNOWLEDGE_PATH,
    STAGE5_SUMMARY_PATH,
    AcousticKnowledgeStage5,
    bin_dur_ms,
    bin_f0_st,
)
from exp_20_pybot_larynx.arbiter import (  # noqa: E402
    classify_final_group,
    classify_onset_place,
)
from exp_20_pybot_larynx.larynx.synth import log_env_to_cep  # noqa: E402

DEFAULT_S5_CACHE_PATH = Path(
    r"C:\Users\MECHREVO\.gemini\antigravity\brain\f67e0437-e98d-4cb1-bbfd-da72d02478db\scratch\train_tokens_s5.pkl"
)


def _worker_init() -> None:
    """Prevent BLAS/OpenMP thread oversubscription inside worker processes."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _extract_one_utt_stage5(
    task: tuple[str, str, str, list[tuple[str, str, str, int, int, int, int, float, int]]],
) -> dict[str, Any]:
    """Extract Stage 5 tokens + prosody ground-truth metadata from one TRAIN utterance."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from exp_v9_clarity_pro.hnm_v9 import analyze_v9
    from exp_v12_natural_vocoder.v12_bank import _patch_unsmoothed_uv_env
    from exp_v15_bznsyp_tts.index_bznsyp_v15 import read_bznsyp_wav_24k

    uid, text, wav_rel, syls = task
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
    cep64 = log_env_to_cep(log_env_norm, n_cep=N_CEP).astype(np.float16)
    bap6_f16 = bap6.astype(np.float16)
    v_f16 = feats_p.voiced.astype(np.float16)
    ref_f0 = max(50.0, utt_f0_med)

    active_mask = np.zeros(n_fr, dtype=bool)
    onset_tokens: list[tuple[Any, ...]] = []
    final_tokens: list[tuple[Any, ...]] = []
    syl_records: list[tuple[Any, ...]] = []

    norm_syls = []
    for hz, raw_o, raw_f, tone, start_fr, split_abs_fr, end_fr, pause_ms, brk in syls:
        o, f = normalize_of(raw_o, raw_f)
        norm_syls.append((hz, o, f, int(tone), int(start_fr), int(split_abs_fr), int(end_fr), float(pause_ms), int(brk)))

    n_s = len(norm_syls)
    for si, (hz, o, f, t, start_fr, split_abs_fr, end_fr, pause_ms, brk) in enumerate(norm_syls):
        a = max(0, min(n_fr - 1, start_fr))
        b = max(a + 1, min(n_fr, end_fr))
        sp = max(a, min(b - 1, split_abs_fr))
        active_mask[a:b] = True

        if si == 0 or (a - norm_syls[si - 1][6]) >= 3 or norm_syls[si - 1][7] >= 25.0:
            prev_fg = "SIL"
        else:
            prev_fg = classify_final_group(norm_syls[si - 1][2])

        if si == n_s - 1 or pause_ms >= 25.0 or (norm_syls[si + 1][4] - b) >= 3:
            next_pl = "SIL"
        else:
            next_pl = classify_onset_place(norm_syls[si + 1][1])

        if prev_fg == "SIL":
            pos_id = 0
        elif next_pl == "SIL":
            pos_id = 2
        else:
            pos_id = 1

        prev_id = encode_prev_fg(prev_fg)
        next_id = encode_next_place(next_pl)
        dur_ms = float(b - a) * 5.0
        o_ms = float(max(0, sp - a)) * 5.0

        v_sub = v_mask[a:b]
        f0_st = float(12.0 * np.log2(np.median(feats_p.f0_hz[a:b][v_sub]) / ref_f0)) if np.any(v_sub) else 0.0

        on_uv_frac = -1.0
        if o and sp > a:
            on_c = _bin_segment(cep64[a:sp].astype(np.float32), N_ONSET_BINS_S2).astype(np.float16)
            on_b = _bin_segment(bap6_f16[a:sp].astype(np.float32), N_ONSET_BINS_S2).astype(np.float16)
            on_v = _bin_segment(v_f16[a:sp].astype(np.float32), N_ONSET_BINS_S2).astype(np.float16)
            o_frac = float(sp - a) / float(max(1, b - a))
            on_uv_frac = float(np.mean(~feats_p.voiced[a:sp]))
            onset_tokens.append((
                o, f, classify_final_group(f),
                prev_id, next_id, pos_id, o_ms, o_frac, on_uv_frac, f0_st,
                on_c, on_b, on_v,
            ))
            f_a = sp
        else:
            f_a = a

        # 3-point real F0 on final + real cep[0] mean
        f3_st = (0.0, 0.0, 0.0)
        cep0_mean = 0.0
        if b > f_a:
            fi_c = _bin_segment(cep64[f_a:b].astype(np.float32), N_FINAL_BINS_S2).astype(np.float16)
            fi_b = _bin_segment(bap6_f16[f_a:b].astype(np.float32), N_FINAL_BINS_S2).astype(np.float16)
            fi_v = _bin_segment(v_f16[f_a:b].astype(np.float32), N_FINAL_BINS_S2).astype(np.float16)
            cep0_mean = float(np.mean(fi_c[:, 0].astype(np.float32)))
            final_tokens.append((
                o, classify_onset_place(o), f, t,
                prev_id, next_id, pos_id, dur_ms, f0_st,
                fi_c, fi_b, fi_v,
            ))
            # Extract 3-point F0 over final voiced frames at u=[0.20, 0.50, 0.80]
            vf_idx = np.where(v_mask[f_a:b])[0]
            if len(vf_idx) >= 2:
                u_v = vf_idx.astype(np.float64) / float(max(1, (b - f_a) - 1))
                st_v = 12.0 * np.log2(feats_p.f0_hz[f_a:b][vf_idx] / ref_f0)
                p3 = np.interp([0.20, 0.50, 0.80], u_v, st_v)
                f3_st = (round(float(p3[0]), 3), round(float(p3[1]), 3), round(float(p3[2]), 3))
            elif len(vf_idx) == 1:
                val = round(float(12.0 * np.log2(feats_p.f0_hz[f_a:b][vf_idx[0]] / ref_f0)), 3)
                f3_st = (val, val, val)
            else:
                f3_st = (round(f0_st, 3), round(f0_st, 3), round(f0_st, 3))

        syl_records.append((
            si, hz, o, f, t, dur_ms, o_ms, on_uv_frac, f0_st, f3_st, cep0_mean, pos_id, brk, pause_ms,
        ))

    sil_mask = ~active_mask
    if np.any(sil_mask):
        sil_cep = np.mean(cep64[sil_mask].astype(np.float32), axis=0, dtype=np.float64)
        sil_bap = np.mean(bap6_f16[sil_mask].astype(np.float32), axis=0, dtype=np.float64)
        sil_cnt = int(np.sum(sil_mask))
    else:
        sil_cep = np.zeros(N_CEP, dtype=np.float64)
        sil_bap = np.ones(N_BAP, dtype=np.float64) * 0.95
        sil_cnt = 0

    return {
        "uid": uid,
        "text": text,
        "utt_log_ref": utt_log_ref,
        "utt_f0_med": utt_f0_med,
        "onset_tokens": onset_tokens,
        "final_tokens": final_tokens,
        "syl_records": syl_records,
        "sil_cep": sil_cep,
        "sil_bap": sil_bap,
        "sil_cnt": sil_cnt,
    }


def _select_medoids_kmeans_s5(
    feat_x: np.ndarray,
    k_max: int = 8,
    seed: int = 2026,
) -> tuple[list[int], list[int]]:
    """Adaptive deterministic NumPy K-means++ medoid selection (up to K=8)."""
    n = feat_x.shape[0]
    if n == 1:
        return [0], [1]
    if n <= 3:
        center = np.mean(feat_x, axis=0)
        d2 = np.sum((feat_x - center[None, :]) ** 2, axis=1)
        return [int(np.argmin(d2))], [n]

    if n <= 11:
        k = min(k_max, 2)
    elif n <= 29:
        k = min(k_max, 4)
    elif n <= 79:
        k = min(k_max, 6)
    else:
        k = min(k_max, 8)

    rng = np.random.default_rng(seed + n)
    g_mean = np.mean(feat_x, axis=0)
    first_idx = int(np.argmin(np.sum((feat_x - g_mean[None, :]) ** 2, axis=1)))
    centers = [feat_x[first_idx].copy()]
    min_d2 = np.sum((feat_x - centers[0][None, :]) ** 2, axis=1)
    for _ in range(1, k):
        tot = float(np.sum(min_d2))
        if tot <= 1e-12:
            break
        probs = min_d2 / tot
        nxt = int(rng.choice(n, p=probs))
        centers.append(feat_x[nxt].copy())
        d2_new = np.sum((feat_x - centers[-1][None, :]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)

    c_arr = np.stack(centers, axis=0)
    k_actual = c_arr.shape[0]
    x_norm2 = np.sum(feat_x * feat_x, axis=1, keepdims=True)
    labels = np.zeros(n, dtype=np.int32)
    for it in range(12):
        c_norm2 = np.sum(c_arr * c_arr, axis=1)[None, :]
        dist2 = x_norm2 + c_norm2 - 2.0 * (feat_x @ c_arr.T)
        new_labels = np.argmin(dist2, axis=1)
        if np.array_equal(new_labels, labels) and it > 0:
            break
        labels = new_labels
        for ci in range(k_actual):
            members = np.where(labels == ci)[0]
            if len(members) > 0:
                c_arr[ci] = np.mean(feat_x[members], axis=0)

    min_sz = 2 if n >= 8 else 1
    valid_clusters: list[tuple[np.ndarray, np.ndarray]] = []
    for ci in range(k_actual):
        members = np.where(labels == ci)[0]
        if len(members) >= min_sz:
            valid_clusters.append((members, np.mean(feat_x[members], axis=0)))

    if not valid_clusters:
        center = np.mean(feat_x, axis=0)
        d2 = np.sum((feat_x - center[None, :]) ** 2, axis=1)
        return [int(np.argmin(d2))], [n]

    medoids: list[int] = []
    sizes: list[int] = []
    seen: set[int] = set()
    for members, centroid in valid_clusters:
        d2 = np.sum((feat_x[members] - centroid[None, :]) ** 2, axis=1)
        order = np.argsort(d2)
        chosen = int(members[order[0]])
        for pos_idx in order:
            cand = int(members[pos_idx])
            if cand not in seen:
                chosen = cand
                break
        seen.add(chosen)
        medoids.append(chosen)
        sizes.append(int(len(members)))
    return medoids, sizes


def _build_medoid_table_s5(
    prefix: str,
    groups: dict[str, list[int]],
    cep_arr: np.ndarray,
    bap_arr: np.ndarray,
    vp_arr: np.ndarray,
    ctx_prev: np.ndarray,
    ctx_next: np.ndarray,
    ctx_pos: np.ndarray,
    ctx_dur_ms: np.ndarray,
    ctx_f0_st: np.ndarray,
    t_bins: int,
    *,
    k_max: int = 8,
    cap_tokens: int = 2400,
    compute_gv: bool = False,
) -> tuple[dict[str, np.ndarray], int]:
    """Cluster each key's tokens into K<=k_max unblurred medoids and pack as float16 CSR arrays."""
    keys = sorted(groups.keys())
    offsets = [0]
    tok_cnts = []

    med_cep = []
    med_bap = []
    med_vp = []
    med_prev = []
    med_next = []
    med_pos = []
    med_dur = []
    med_f0 = []
    med_csz = []
    gv_stds = []

    rng = np.random.default_rng(2026)
    for k in keys:
        idxs = groups[k]
        total_n = len(idxs)
        tok_cnts.append(total_n)

        if total_n > cap_tokens:
            sub_pos = rng.choice(total_n, size=cap_tokens, replace=False)
            idx_arr = np.asarray([idxs[j] for j in np.sort(sub_pos)], dtype=np.int32)
        else:
            idx_arr = np.asarray(idxs, dtype=np.int32)

        sub_cep = cep_arr[idx_arr].astype(np.float64)  # (M, t_bins, 64)
        if compute_gv:
            # Target median within-syllable temporal std over cep[0:40]
            tok_std = np.std(sub_cep[:, :, :40], axis=1)  # (M, 40)
            gv_stds.append(np.median(tok_std, axis=0).astype(np.float16))

        # Combine spectral shape + energy + normalized prosody cues for medoid diversity
        sub_dur = np.log(np.maximum(10.0, ctx_dur_ms[idx_arr].astype(np.float64)))
        sub_dur = (sub_dur - np.mean(sub_dur)) / max(0.15, float(np.std(sub_dur)))
        sub_f0 = ctx_f0_st[idx_arr].astype(np.float64)
        sub_f0 = (sub_f0 - np.mean(sub_f0)) / max(0.50, float(np.std(sub_f0)))

        spec_feat = np.concatenate([0.35 * sub_cep[:, :, :1], sub_cep[:, :, 1:40]], axis=2).reshape(len(idx_arr), -1)
        feat = np.concatenate([spec_feat, 2.0 * sub_dur[:, None], 1.5 * sub_f0[:, None]], axis=1)
        med_local_ids, cl_sizes = _select_medoids_kmeans_s5(feat, k_max=k_max, seed=2026)

        scale_factor = float(total_n) / float(len(idx_arr))
        for m_loc, c_sz in zip(med_local_ids, cl_sizes):
            gi = int(idx_arr[m_loc])
            med_cep.append(cep_arr[gi])
            med_bap.append(bap_arr[gi])
            med_vp.append(vp_arr[gi])
            med_prev.append(ctx_prev[gi])
            med_next.append(ctx_next[gi])
            med_pos.append(ctx_pos[gi])
            med_dur.append(ctx_dur_ms[gi])
            med_f0.append(ctx_f0_st[gi])
            med_csz.append(max(1, int(round(c_sz * scale_factor))))
        offsets.append(len(med_cep))

    n_keys = len(keys)
    n_med = len(med_cep)
    max_len = max((len(k) for k in keys), default=1)
    out = {
        f"{prefix}_keys": np.array(keys, dtype=f"<U{max_len}"),
        f"{prefix}_offsets": np.array(offsets, dtype=np.int32),
        f"{prefix}_tok_cnt": np.array(tok_cnts, dtype=np.int32),
        f"{prefix}_cep": np.stack(med_cep, axis=0).astype(np.float16) if n_med else np.zeros((0, t_bins, N_CEP), dtype=np.float16),
        f"{prefix}_bap": np.stack(med_bap, axis=0).astype(np.float16) if n_med else np.zeros((0, t_bins, N_BAP), dtype=np.float16),
        f"{prefix}_vp": np.stack(med_vp, axis=0).astype(np.float16) if n_med else np.zeros((0, t_bins), dtype=np.float16),
        f"{prefix}_ctx_prev": np.array(med_prev, dtype=np.int8),
        f"{prefix}_ctx_next": np.array(med_next, dtype=np.int8),
        f"{prefix}_ctx_pos": np.array(med_pos, dtype=np.int8),
        f"{prefix}_ctx_dur_ms": np.array(med_dur, dtype=np.float16),
        f"{prefix}_ctx_f0_st": np.array(med_f0, dtype=np.float16),
        f"{prefix}_cluster_sz": np.array(med_csz, dtype=np.int32),
    }
    if compute_gv:
        out["fin_gv_oft_keys"] = np.array(keys, dtype=f"<U{max_len}")
        out["fin_gv_oft_std"] = np.stack(gv_stds, axis=0).astype(np.float16) if gv_stds else np.ones((0, 40), dtype=np.float16)
    return out, n_keys


def extract_or_load_train_tokens_s5(
    cache_path: Path = DEFAULT_S5_CACHE_PATH,
    *,
    max_workers: int | None = 20,
    limit: int | None = None,
) -> dict[str, Any]:
    """Extract all TRAIN Stage 5 token features in parallel once, or load cached pickle."""
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

    if cache_path.exists():
        print(f"Loading cached Stage 5 TRAIN tokens from {cache_path}...", flush=True)
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if int(cached["n_utts"]) == len(selected_ids):
            return cached
        print(f"Cache size mismatch ({cached['n_utts']} != {len(selected_ids)}), rebuilding...", flush=True)

    t0 = time.perf_counter()
    idx = load_v15_index_cached()
    metas = idx["train_utt_metas"]
    tokens = idx["tokens"]
    by_utt_syl = idx["by_utt_syl"]

    tasks: list[tuple[str, str, str, list[tuple[str, str, str, int, int, int, int, float, int]]]] = []
    for uid in selected_ids:
        m = metas[uid]
        syl_list: list[tuple[str, str, str, int, int, int, int, float, int]] = []
        for si in range(int(m["n_syllables"])):
            t_idx = by_utt_syl.get((uid, si))
            if t_idx is None:
                continue
            tok = tokens[t_idx]
            syl_list.append((
                str(tok.hanzi),
                str(tok.onset),
                str(tok.final),
                int(tok.tone),
                int(tok.start_fr),
                int(tok.start_fr + tok.split_fr),
                int(tok.end_fr),
                float(tok.pause_after_ms),
                int(tok.prosodic_break),
            ))
        if syl_list:
            tasks.append((uid, str(m["text"]), str(m["wav_path"]), syl_list))

    n_utts = len(tasks)
    workers = max_workers if max_workers is not None else min(20, os.cpu_count() or 4)
    print(f"Extracting Stage 5 TRAIN tokens from {n_utts} utts (workers={workers})...", flush=True)

    utt_log_refs: list[float] = []
    utt_f0_meds: list[float] = []
    sil_cep_sum = np.zeros(N_CEP, dtype=np.float64)
    sil_bap_sum = np.zeros(N_BAP, dtype=np.float64)
    sil_tot = 0

    onset_frac_list: dict[str, list[float]] = defaultdict(list)
    onset_ms_list: dict[str, list[float]] = defaultdict(list)
    onset_uv_list: dict[str, list[float]] = defaultdict(list)

    on_cep_l: list[np.ndarray] = []
    on_bap_l: list[np.ndarray] = []
    on_vp_l: list[np.ndarray] = []
    on_prev_l: list[int] = []
    on_next_l: list[int] = []
    on_pos_l: list[int] = []
    on_dur_l: list[float] = []
    on_f0_l: list[float] = []
    on_of_idx: dict[str, list[int]] = defaultdict(list)
    on_grp_idx: dict[str, list[int]] = defaultdict(list)
    on_only_idx: dict[str, list[int]] = defaultdict(list)

    fi_cep_l: list[np.ndarray] = []
    fi_bap_l: list[np.ndarray] = []
    fi_vp_l: list[np.ndarray] = []
    fi_prev_l: list[int] = []
    fi_next_l: list[int] = []
    fi_pos_l: list[int] = []
    fi_dur_l: list[float] = []
    fi_f0_l: list[float] = []
    fi_tone_l: list[int] = []
    fin_oft_idx: dict[str, list[int]] = defaultdict(list)
    fin_pft_idx: dict[str, list[int]] = defaultdict(list)
    fin_ft_idx: dict[str, list[int]] = defaultdict(list)
    fin_f_idx: dict[str, list[int]] = defaultdict(list)

    utt_prosody_records: dict[str, dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        for idx_done, res in enumerate(ex.map(_extract_one_utt_stage5, tasks, chunksize=24), 1):
            uid = res["uid"]
            utt_log_refs.append(res["utt_log_ref"])
            utt_f0_meds.append(res["utt_f0_med"])
            if res["sil_cnt"] > 0:
                sil_cep_sum += res["sil_cep"] * res["sil_cnt"]
                sil_bap_sum += res["sil_bap"] * res["sil_cnt"]
                sil_tot += res["sil_cnt"]

            utt_prosody_records[uid] = {
                "text": res["text"],
                "utt_f0_med": res["utt_f0_med"],
                "syls": res["syl_records"],
            }

            for o, f, fg, prev_id, next_id, pos_id, o_ms, o_frac, uv_frac, f0_st, oc, ob, ov in res["onset_tokens"]:
                oi = len(on_cep_l)
                on_cep_l.append(oc)
                on_bap_l.append(ob)
                on_vp_l.append(ov)
                on_prev_l.append(prev_id)
                on_next_l.append(next_id)
                on_pos_l.append(pos_id)
                on_dur_l.append(o_ms)
                on_f0_l.append(f0_st)
                on_of_idx[f"{o}|{f}"].append(oi)
                on_grp_idx[f"{o}|{fg}"].append(oi)
                on_only_idx[o].append(oi)
                onset_frac_list[o].append(o_frac)
                onset_ms_list[o].append(o_ms)
                if uv_frac >= 0.0:
                    onset_uv_list[o].append(uv_frac)

            for o, place, f, t, prev_id, next_id, pos_id, dur_ms, f0_st, fc, fb, fv in res["final_tokens"]:
                fi = len(fi_cep_l)
                fi_cep_l.append(fc)
                fi_bap_l.append(fb)
                fi_vp_l.append(fv)
                fi_prev_l.append(prev_id)
                fi_next_l.append(next_id)
                fi_pos_l.append(pos_id)
                fi_dur_l.append(dur_ms)
                fi_f0_l.append(f0_st)
                fi_tone_l.append(t)
                fin_oft_idx[f"{o}|{f}|{t}"].append(fi)
                fin_pft_idx[f"{place}|{f}|{t}"].append(fi)
                fin_ft_idx[f"{f}|{t}"].append(fi)
                fin_f_idx[f].append(fi)

            if idx_done % 1500 == 0 or idx_done == n_utts:
                print(f"  processed {idx_done}/{n_utts} ({time.perf_counter() - t0:.1f}s)", flush=True)

    payload = {
        "n_utts": n_utts,
        "utt_log_refs": utt_log_refs,
        "utt_f0_meds": utt_f0_meds,
        "sil_cep64": (sil_cep_sum / max(1, sil_tot)).astype(np.float32),
        "sil_bap6": (sil_bap_sum / max(1, sil_tot)).astype(np.float32),
        "onset_frac_list": dict(onset_frac_list),
        "onset_ms_list": dict(onset_ms_list),
        "onset_uv_list": dict(onset_uv_list),
        "on_cep": np.stack(on_cep_l, axis=0).astype(np.float16),
        "on_bap": np.stack(on_bap_l, axis=0).astype(np.float16),
        "on_vp": np.stack(on_vp_l, axis=0).astype(np.float16),
        "on_prev": np.array(on_prev_l, dtype=np.int8),
        "on_next": np.array(on_next_l, dtype=np.int8),
        "on_pos": np.array(on_pos_l, dtype=np.int8),
        "on_dur": np.array(on_dur_l, dtype=np.float32),
        "on_f0": np.array(on_f0_l, dtype=np.float32),
        "on_of_idx": dict(on_of_idx),
        "on_grp_idx": dict(on_grp_idx),
        "on_only_idx": dict(on_only_idx),
        "fi_cep": np.stack(fi_cep_l, axis=0).astype(np.float16),
        "fi_bap": np.stack(fi_bap_l, axis=0).astype(np.float16),
        "fi_vp": np.stack(fi_vp_l, axis=0).astype(np.float16),
        "fi_prev": np.array(fi_prev_l, dtype=np.int8),
        "fi_next": np.array(fi_next_l, dtype=np.int8),
        "fi_pos": np.array(fi_pos_l, dtype=np.int8),
        "fi_dur": np.array(fi_dur_l, dtype=np.float32),
        "fi_f0": np.array(fi_f0_l, dtype=np.float32),
        "fi_tone": np.array(fi_tone_l, dtype=np.int8),
        "fin_oft_idx": dict(fin_oft_idx),
        "fin_pft_idx": dict(fin_pft_idx),
        "fin_ft_idx": dict(fin_ft_idx),
        "fin_f_idx": dict(fin_f_idx),
        "utt_prosody_records": utt_prosody_records,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Cached Stage 5 TRAIN tokens to {cache_path} in {time.perf_counter() - t0:.1f}s", flush=True)
    return payload


def build_stage5_knowledge(
    out_path: Path = STAGE5_KNOWLEDGE_PATH,
    summary_path: Path = STAGE5_SUMMARY_PATH,
    cache_path: Path = DEFAULT_S5_CACHE_PATH,
    max_workers: int | None = 20,
    limit: int | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    data = extract_or_load_train_tokens_s5(cache_path, max_workers=max_workers, limit=limit)
    _, dev_60, held_ids = get_splits()

    n_utts = int(data["n_utts"])
    utt_log_refs = data["utt_log_refs"]
    utt_f0_meds = data["utt_f0_meds"]
    sil_cep64 = data["sil_cep64"]
    sil_bap6 = data["sil_bap6"]
    onset_frac_list = data["onset_frac_list"]
    onset_ms_list = data["onset_ms_list"]
    onset_uv_list = data["onset_uv_list"]

    on_cep = data["on_cep"]
    on_bap = data["on_bap"]
    on_vp = data["on_vp"]
    on_prev = data["on_prev"]
    on_next = data["on_next"]
    on_pos = data["on_pos"]
    on_dur = data["on_dur"]
    on_f0 = data["on_f0"]
    on_of_idx = data["on_of_idx"]
    on_grp_idx = data["on_grp_idx"]
    on_only_idx = data["on_only_idx"]

    fi_cep = data["fi_cep"]
    fi_bap = data["fi_bap"]
    fi_vp = data["fi_vp"]
    fi_prev = data["fi_prev"]
    fi_next = data["fi_next"]
    fi_pos = data["fi_pos"]
    fi_dur = data["fi_dur"]
    fi_f0 = data["fi_f0"]
    fin_oft_idx = data["fin_oft_idx"]
    fin_pft_idx = data["fin_pft_idx"]
    fin_ft_idx = data["fin_ft_idx"]
    fin_f_idx = data["fin_f_idx"]

    t_cl = time.perf_counter()
    print(f"Clustering {len(on_cep)} onset + {len(fi_cep)} final tokens into Stage 5 medoids...", flush=True)

    payload: dict[str, np.ndarray] = {}
    t_on_of, n_on_of = _build_medoid_table_s5(
        "on_of", on_of_idx, on_cep, on_bap, on_vp, on_prev, on_next, on_pos, on_dur, on_f0,
        N_ONSET_BINS_S2, k_max=6,
    )
    t_on_grp, n_on_grp = _build_medoid_table_s5(
        "on_grp", on_grp_idx, on_cep, on_bap, on_vp, on_prev, on_next, on_pos, on_dur, on_f0,
        N_ONSET_BINS_S2, k_max=6,
    )
    t_on_only, n_on_only = _build_medoid_table_s5(
        "on_only", on_only_idx, on_cep, on_bap, on_vp, on_prev, on_next, on_pos, on_dur, on_f0,
        N_ONSET_BINS_S2, k_max=6,
    )
    t_fin_oft, n_fin_oft = _build_medoid_table_s5(
        "fin_oft", fin_oft_idx, fi_cep, fi_bap, fi_vp, fi_prev, fi_next, fi_pos, fi_dur, fi_f0,
        N_FINAL_BINS_S2, k_max=8, compute_gv=True,
    )
    t_fin_pft, n_fin_pft = _build_medoid_table_s5(
        "fin_pft", fin_pft_idx, fi_cep, fi_bap, fi_vp, fi_prev, fi_next, fi_pos, fi_dur, fi_f0,
        N_FINAL_BINS_S2, k_max=6,
    )
    t_fin_ft, n_fin_ft = _build_medoid_table_s5(
        "fin_ft", fin_ft_idx, fi_cep, fi_bap, fi_vp, fi_prev, fi_next, fi_pos, fi_dur, fi_f0,
        N_FINAL_BINS_S2, k_max=6,
    )
    t_fin_f, n_fin_f = _build_medoid_table_s5(
        "fin_f", fin_f_idx, fi_cep, fi_bap, fi_vp, fi_prev, fi_next, fi_pos, fi_dur, fi_f0,
        N_FINAL_BINS_S2, k_max=4,
    )

    for tbl in (t_on_of, t_on_grp, t_on_only, t_fin_oft, t_fin_pft, t_fin_ft, t_fin_f):
        payload.update(tbl)

    payload["fin_gv_glob"] = np.median(t_fin_oft["fin_gv_oft_std"].astype(np.float64), axis=0).astype(np.float16)

    on_c32_s5 = on_cep[:, :, :N_COARSE_CEP_S5].astype(np.float32)
    fi_c32_s5 = fi_cep[:, :, :N_COARSE_CEP_S5].astype(np.float32)

    def _build_shrunk_ctx(
        prefix: str,
        key_to_idxs: dict[str, list[int]],
        cep_arr: np.ndarray,
        t_bins: int,
        parent_fn: Any = None,
        k_shrink: float = 4.0,
    ) -> dict[str, np.ndarray]:
        c_keys = sorted(key_to_idxs.keys())
        if not c_keys:
            payload[f"{prefix}_keys"] = np.array([], dtype="<U1")
            payload[f"{prefix}_cep"] = np.zeros((0, t_bins, N_COARSE_CEP_S5), dtype=np.float16)
            return {}
        out_map: dict[str, np.ndarray] = {}
        rows: list[np.ndarray] = []
        for k in c_keys:
            idxs = key_to_idxs[k]
            mu = np.mean(cep_arr[idxs], axis=0).astype(np.float64)
            if parent_fn is not None:
                p_mu = parent_fn(k)
                if p_mu is not None:
                    w = float(len(idxs)) / (float(len(idxs)) + k_shrink)
                    mu = w * mu + (1.0 - w) * p_mu
            out_map[k] = mu
            rows.append(mu)
        max_len = max(len(k) for k in c_keys)
        payload[f"{prefix}_keys"] = np.array(c_keys, dtype=f"<U{max_len}")
        payload[f"{prefix}_cep"] = np.stack(rows, axis=0).astype(np.float16)
        return out_map

    # Onset hierarchy with Empirical-Bayes shrinkage (o -> o|fg -> o|fg|prev / o|f -> o|f|pos / prev -> o|f|prev|pos)
    on_only_means = {k: np.mean(on_c32_s5[idxs], axis=0).astype(np.float64) for k, idxs in on_only_idx.items()}
    on_ctx_base_map = _build_shrunk_ctx(
        "on_ctx_base",
        on_grp_idx,
        on_c32_s5,
        N_ONSET_BINS_S2,
        parent_fn=lambda k: on_only_means.get(k.split("|", 1)[0]),
    )
    on_ctx_of_pp_idx: dict[str, list[int]] = defaultdict(list)
    on_ctx_of_pos_idx: dict[str, list[int]] = defaultdict(list)
    on_ctx_of_prev_idx: dict[str, list[int]] = defaultdict(list)
    for k, idxs in on_of_idx.items():
        for i in idxs:
            pv, ps = int(on_prev[i]), int(on_pos[i])
            on_ctx_of_pp_idx[f"{k}|{pv}|{ps}"].append(i)
            on_ctx_of_pos_idx[f"{k}|{ps}"].append(i)
            on_ctx_of_prev_idx[f"{k}|{pv}"].append(i)
    on_ctx_of_pp_idx = {k: v for k, v in on_ctx_of_pp_idx.items() if len(v) >= 3}
    on_ctx_of_pos_idx = {k: v for k, v in on_ctx_of_pos_idx.items() if len(v) >= 3}
    on_ctx_of_prev_idx = {k: v for k, v in on_ctx_of_prev_idx.items() if len(v) >= 3}
    on_ctx_of_base_idx = {k: v for k, v in on_of_idx.items() if len(v) >= 2}

    on_ctx_p_idx: dict[str, list[int]] = defaultdict(list)
    for k, idxs in on_grp_idx.items():
        for i in idxs:
            on_ctx_p_idx[f"{k}|{int(on_prev[i])}"].append(i)
    on_ctx_p_idx = {k: v for k, v in on_ctx_p_idx.items() if len(v) >= 3}

    def _on_of_parent(k: str) -> np.ndarray | None:
        o_p, f_p = k.split("|", 1)
        fg_p = classify_final_group(f_p)
        return on_ctx_base_map.get(f"{o_p}|{fg_p}", on_only_means.get(o_p))

    on_ctx_of_base_map = _build_shrunk_ctx("on_ctx_of_base", on_ctx_of_base_idx, on_c32_s5, N_ONSET_BINS_S2, parent_fn=_on_of_parent)
    on_ctx_p_map = _build_shrunk_ctx(
        "on_ctx_p", on_ctx_p_idx, on_c32_s5, N_ONSET_BINS_S2, parent_fn=lambda k: on_ctx_base_map.get(k.rsplit("|", 1)[0])
    )
    on_ctx_of_pos_map = _build_shrunk_ctx(
        "on_ctx_of_pos", on_ctx_of_pos_idx, on_c32_s5, N_ONSET_BINS_S2, parent_fn=lambda k: on_ctx_of_base_map.get(k.rsplit("|", 1)[0])
    )
    on_ctx_of_prev_map = _build_shrunk_ctx(
        "on_ctx_of_prev", on_ctx_of_prev_idx, on_c32_s5, N_ONSET_BINS_S2, parent_fn=lambda k: on_ctx_of_base_map.get(k.rsplit("|", 1)[0])
    )

    def _on_pp_parent(k: str) -> np.ndarray | None:
        o_p, f_p, pv_p, ps_p = k.split("|", 3)
        kof = f"{o_p}|{f_p}"
        return on_ctx_of_prev_map.get(f"{kof}|{pv_p}", on_ctx_of_pos_map.get(f"{kof}|{ps_p}", on_ctx_of_base_map.get(kof)))

    on_ctx_of_pp_map = _build_shrunk_ctx("on_ctx_of_pp", on_ctx_of_pp_idx, on_c32_s5, N_ONSET_BINS_S2, parent_fn=_on_pp_parent)

    # Final hierarchy with Empirical-Bayes shrinkage (f -> f|t -> place|f|t -> o|f|t -> pos -> 2-way -> 3-way)
    fin_f_means = {k: np.mean(fi_c32_s5[idxs], axis=0).astype(np.float64) for k, idxs in fin_f_idx.items()}
    fin_ft_means: dict[str, np.ndarray] = {}
    for k, idxs in fin_ft_idx.items():
        mu = np.mean(fi_c32_s5[idxs], axis=0).astype(np.float64)
        p_mu = fin_f_means.get(k.split("|", 1)[0])
        if p_mu is not None:
            w = float(len(idxs)) / (float(len(idxs)) + 4.0)
            mu = w * mu + (1.0 - w) * p_mu
        fin_ft_means[k] = mu
    fin_pft_means: dict[str, np.ndarray] = {}
    for k, idxs in fin_pft_idx.items():
        mu = np.mean(fi_c32_s5[idxs], axis=0).astype(np.float64)
        _, f_p, t_p = k.split("|", 2)
        p_mu = fin_ft_means.get(f"{f_p}|{t_p}", fin_f_means.get(f_p))
        if p_mu is not None:
            w = float(len(idxs)) / (float(len(idxs)) + 4.0)
            mu = w * mu + (1.0 - w) * p_mu
        fin_pft_means[k] = mu

    def _fin_oft_parent(k: str) -> np.ndarray | None:
        o_p, f_p, t_p = k.split("|", 2)
        pl_p = classify_onset_place(o_p)
        return fin_pft_means.get(f"{pl_p}|{f_p}|{t_p}", fin_ft_means.get(f"{f_p}|{t_p}", fin_f_means.get(f_p)))

    fin_ctx_base_map = _build_shrunk_ctx("fin_ctx_base", fin_oft_idx, fi_c32_s5, N_FINAL_BINS_S2, parent_fn=_fin_oft_parent)

    fin_ctx_pnp_idx: dict[str, list[int]] = defaultdict(list)
    fin_ctx_pn_idx: dict[str, list[int]] = defaultdict(list)
    fin_ctx_np_idx: dict[str, list[int]] = defaultdict(list)
    fin_ctx_pp_idx: dict[str, list[int]] = defaultdict(list)
    fin_ctx_pos_idx: dict[str, list[int]] = defaultdict(list)
    fin_ctx_pos_df_idx: dict[str, list[int]] = defaultdict(list)
    fin_ctx_df_idx: dict[str, list[int]] = defaultdict(list)
    fin_ctx_d_idx: dict[str, list[int]] = defaultdict(list)
    fin_ctx_f0_idx: dict[str, list[int]] = defaultdict(list)

    for k, idxs in fin_oft_idx.items():
        for i in idxs:
            pv, nx, ps = int(fi_prev[i]), int(fi_next[i]), int(fi_pos[i])
            db = bin_dur_ms(float(fi_dur[i]))
            fb = bin_f0_st(float(fi_f0[i]))
            fin_ctx_pnp_idx[f"{k}|{pv}|{nx}|{ps}"].append(i)
            fin_ctx_pn_idx[f"{k}|{pv}|{nx}"].append(i)
            fin_ctx_np_idx[f"{k}|{nx}|{ps}"].append(i)
            fin_ctx_pp_idx[f"{k}|{pv}|{ps}"].append(i)
            fin_ctx_pos_idx[f"{k}|{ps}"].append(i)
            fin_ctx_pos_df_idx[f"{k}|{ps}|{db}|{fb}"].append(i)
            fin_ctx_df_idx[f"{k}|{db}|{fb}"].append(i)
            fin_ctx_d_idx[f"{k}|{db}"].append(i)
            fin_ctx_f0_idx[f"{k}|{fb}"].append(i)

    fin_ctx_pnp_idx = {k: v for k, v in fin_ctx_pnp_idx.items() if len(v) >= 3}
    fin_ctx_pn_idx = {k: v for k, v in fin_ctx_pn_idx.items() if len(v) >= 3}
    fin_ctx_np_idx = {k: v for k, v in fin_ctx_np_idx.items() if len(v) >= 3}
    fin_ctx_pp_idx = {k: v for k, v in fin_ctx_pp_idx.items() if len(v) >= 3}
    fin_ctx_pos_idx = {k: v for k, v in fin_ctx_pos_idx.items() if len(v) >= 2}
    fin_ctx_pos_df_idx = {k: v for k, v in fin_ctx_pos_df_idx.items() if len(v) >= 4}
    fin_ctx_df_idx = {k: v for k, v in fin_ctx_df_idx.items() if len(v) >= 4}
    fin_ctx_d_idx = {k: v for k, v in fin_ctx_d_idx.items() if len(v) >= 3}
    fin_ctx_f0_idx = {k: v for k, v in fin_ctx_f0_idx.items() if len(v) >= 3}

    fin_ctx_pft_pos_idx: dict[str, list[int]] = defaultdict(list)
    for k, idxs in fin_pft_idx.items():
        for i in idxs:
            fin_ctx_pft_pos_idx[f"{k}|{int(fi_pos[i])}"].append(i)

    fin_ctx_pft_map = _build_shrunk_ctx(
        "fin_ctx_pft_pos", fin_ctx_pft_pos_idx, fi_c32_s5, N_FINAL_BINS_S2, parent_fn=lambda k: fin_pft_means.get(k.rsplit("|", 1)[0])
    )
    fin_ctx_pos_map = _build_shrunk_ctx(
        "fin_ctx_pos", fin_ctx_pos_idx, fi_c32_s5, N_FINAL_BINS_S2, parent_fn=lambda k: fin_ctx_base_map.get(k.rsplit("|", 1)[0])
    )

    def _fin_2way_parent(k: str, has_pos: bool) -> np.ndarray | None:
        o_p, f_p, t_p, _, second = k.split("|", 4)
        koft = f"{o_p}|{f_p}|{t_p}"
        if has_pos:
            return fin_ctx_pos_map.get(f"{koft}|{second}", fin_ctx_base_map.get(koft))
        return fin_ctx_base_map.get(koft)

    fin_ctx_pn_map = _build_shrunk_ctx(
        "fin_ctx_pn", fin_ctx_pn_idx, fi_c32_s5, N_FINAL_BINS_S2, parent_fn=lambda k: _fin_2way_parent(k, False)
    )
    fin_ctx_np_map = _build_shrunk_ctx(
        "fin_ctx_np", fin_ctx_np_idx, fi_c32_s5, N_FINAL_BINS_S2, parent_fn=lambda k: _fin_2way_parent(k, True)
    )
    fin_ctx_pp_map = _build_shrunk_ctx(
        "fin_ctx_pp", fin_ctx_pp_idx, fi_c32_s5, N_FINAL_BINS_S2, parent_fn=lambda k: _fin_2way_parent(k, True)
    )

    def _fin_pnp_parent(k: str) -> np.ndarray | None:
        o_p, f_p, t_p, pv_p, nx_p, ps_p = k.split("|", 5)
        koft = f"{o_p}|{f_p}|{t_p}"
        return fin_ctx_pn_map.get(
            f"{koft}|{pv_p}|{nx_p}",
            fin_ctx_np_map.get(
                f"{koft}|{nx_p}|{ps_p}",
                fin_ctx_pos_map.get(f"{koft}|{ps_p}", fin_ctx_base_map.get(koft)),
            ),
        )

    fin_ctx_pnp_map = _build_shrunk_ctx("fin_ctx_pnp", fin_ctx_pnp_idx, fi_c32_s5, N_FINAL_BINS_S2, parent_fn=_fin_pnp_parent)

    fin_ctx_d_map = _build_shrunk_ctx(
        "fin_ctx_d", fin_ctx_d_idx, fi_c32_s5, N_FINAL_BINS_S2, parent_fn=lambda k: fin_ctx_base_map.get(k.rsplit("|", 1)[0])
    )
    fin_ctx_f0_map = _build_shrunk_ctx(
        "fin_ctx_f0", fin_ctx_f0_idx, fi_c32_s5, N_FINAL_BINS_S2, parent_fn=lambda k: fin_ctx_base_map.get(k.rsplit("|", 1)[0])
    )

    def _fin_df_parent(k: str) -> np.ndarray | None:
        o_p, f_p, t_p, db_p, fb_p = k.split("|", 4)
        koft = f"{o_p}|{f_p}|{t_p}"
        md = fin_ctx_d_map.get(f"{koft}|{db_p}")
        mf = fin_ctx_f0_map.get(f"{koft}|{fb_p}")
        if md is not None and mf is not None:
            return 0.5 * (md + mf)
        return md if md is not None else (mf if mf is not None else fin_ctx_base_map.get(koft))

    fin_ctx_df_map = _build_shrunk_ctx("fin_ctx_df", fin_ctx_df_idx, fi_c32_s5, N_FINAL_BINS_S2, parent_fn=_fin_df_parent)

    def _fin_pos_df_parent(k: str) -> np.ndarray | None:
        o_p, f_p, t_p, ps_p, db_p, fb_p = k.split("|", 5)
        koft = f"{o_p}|{f_p}|{t_p}"
        return fin_ctx_df_map.get(f"{koft}|{db_p}|{fb_p}", fin_ctx_pos_map.get(f"{koft}|{ps_p}", fin_ctx_base_map.get(koft)))

    fin_ctx_pos_df_map = _build_shrunk_ctx(
        "fin_ctx_pos_df", fin_ctx_pos_df_idx, fi_c32_s5, N_FINAL_BINS_S2, parent_fn=_fin_pos_df_parent
    )

    print(
        f"Coarse anchors (N_COARSE={N_COARSE_CEP_S5}, shrunk): on_of=({len(on_ctx_of_pp_map)}/{len(on_ctx_of_pos_map)}/{len(on_ctx_of_prev_map)}/{len(on_ctx_of_base_map)}), "
        f"fin_ctx=({len(fin_ctx_pnp_map)}/{len(fin_ctx_pn_map)}/{len(fin_ctx_np_map)}/{len(fin_ctx_pp_map)}/{len(fin_ctx_pos_map)}/{len(fin_ctx_pft_map)}), "
        f"fin_pros=({len(fin_ctx_base_map)}/{len(fin_ctx_pos_df_map)}/{len(fin_ctx_df_map)}/{len(fin_ctx_d_map)}/{len(fin_ctx_f0_map)})",
        flush=True,
    )

    on_c32 = on_cep[:, :, 1:40].astype(np.float32)
    on_g_mean = np.mean(on_c32, axis=0)
    on_g_idx = int(np.argmin(np.sum((on_c32 - on_g_mean[None, :, :]) ** 2, axis=(1, 2))))
    payload["on_glob_cep"] = on_cep[on_g_idx].astype(np.float16)
    payload["on_glob_bap"] = on_bap[on_g_idx].astype(np.float16)
    payload["on_glob_vp"] = on_vp[on_g_idx].astype(np.float16)

    fi_c32 = fi_cep[:, :, 1:40].astype(np.float32)
    fi_g_mean = np.mean(fi_c32, axis=0)
    fi_g_idx = int(np.argmin(np.sum((fi_c32 - fi_g_mean[None, :, :]) ** 2, axis=(1, 2))))
    payload["fin_glob_cep"] = fi_cep[fi_g_idx].astype(np.float16)
    payload["fin_glob_bap"] = fi_bap[fi_g_idx].astype(np.float16)
    payload["fin_glob_vp"] = fi_vp[fi_g_idx].astype(np.float16)

    on_keys = sorted(onset_frac_list.keys())
    max_ok = max((len(k) for k in on_keys), default=1)
    payload["onset_frac_keys"] = np.array(on_keys, dtype=f"<U{max_ok}")
    payload["onset_frac_vals"] = np.array([float(np.median(onset_frac_list[k])) for k in on_keys], dtype=np.float32)
    payload["onset_ms_vals"] = np.array([float(np.median(onset_ms_list[k])) for k in on_keys], dtype=np.float32)

    uv_keys = sorted(onset_uv_list.keys())
    max_uvk = max((len(k) for k in uv_keys), default=1)
    payload["onset_uv_keys"] = np.array(uv_keys, dtype=f"<U{max_uvk}")
    payload["onset_uv_vals"] = np.array([float(np.median(onset_uv_list[k])) for k in uv_keys], dtype=np.float32)

    payload["sil_cep64"] = sil_cep64
    payload["sil_bap6"] = sil_bap6
    payload["speaker_f0_med_hz"] = np.array([float(np.median(utt_f0_meds))], dtype=np.float32)
    payload["ref_log_amp"] = np.array([float(np.median(utt_log_refs))], dtype=np.float32)
    payload["n_train_utts"] = np.array([n_utts], dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    size_kb = out_path.stat().st_size / 1024.0

    kb5 = AcousticKnowledgeStage5.load(out_path)
    assert kb5.n_train_utts == n_utts

    summary = {
        "n_train_utts": n_utts,
        "n_dev_60": len(dev_60),
        "n_heldout_20": len(held_ids),
        "n_onset_bins": N_ONSET_BINS_S2,
        "n_final_bins": N_FINAL_BINS_S2,
        "cluster_time_s": round(time.perf_counter() - t_cl, 2),
        "build_time_s": round(time.perf_counter() - t0, 2),
        "npz_size_kb": round(size_kb, 1),
        "speaker_f0_med_hz": round(float(payload["speaker_f0_med_hz"][0]), 2),
        "ref_log_amp": round(float(payload["ref_log_amp"][0]), 4),
        "n_onset_tokens": int(len(on_cep)),
        "n_final_tokens": int(len(fi_cep)),
        "table_sizes": {
            "on_of": n_on_of,
            "on_grp": n_on_grp,
            "on_only": n_on_only,
            "fin_oft": n_fin_oft,
            "fin_pft": n_fin_pft,
            "fin_ft": n_fin_ft,
            "fin_f": n_fin_f,
        },
        "medoid_counts": {
            "on_of": int(len(t_on_of["on_of_cep"])),
            "on_grp": int(len(t_on_grp["on_grp_cep"])),
            "on_only": int(len(t_on_only["on_only_cep"])),
            "fin_oft": int(len(t_fin_oft["fin_oft_cep"])),
            "fin_pft": int(len(t_fin_pft["fin_pft_cep"])),
            "fin_ft": int(len(t_fin_ft["fin_ft_cep"])),
            "fin_f": int(len(t_fin_f["fin_f_cep"])),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path} ({size_kb:.1f} KB) in {summary['build_time_s']}s", flush=True)
    return summary


def benchmark_dev60_stage5_vs_stage4(
    stage5_path: Path = STAGE5_KNOWLEDGE_PATH,
    *,
    anchor_alpha: float = 0.92,
    prosody_alpha: float = 0.72,
    gv_strength: float = 0.45,
) -> dict[str, Any]:
    """Compare Stage4_C2 vs Stage5_C2 strictly on DEV_60 (never HELDOUT_20)."""
    from exp_20_pybot_larynx.acoustics.build_stage2_knowledge import _compute_detail_stats
    from exp_20_pybot_larynx.acoustics.knowledge_stage2 import AcousticKnowledgeStage2, build_stage2_control
    from exp_20_pybot_larynx.acoustics.knowledge_stage5 import build_stage5_control
    from exp_20_pybot_larynx.judge import evaluate_utterance, summarize_condition
    from exp_20_pybot_larynx.larynx.synth import synthesize_larynx
    from exp_20_pybot_larynx.stage0_probe import _aggregate_kb_hits, _prepare_utterance_oracle
    from exp_v15_bznsyp_tts.index_bznsyp_v15 import parse_bznsyp_corpus

    kb2 = AcousticKnowledgeStage2.load()
    kb5 = AcousticKnowledgeStage5.load(stage5_path)
    _, dev_60_ids, _ = get_splits()
    idx_doc = load_v15_index_cached()
    corpus_by_id = {r["id"]: r for r in parse_bznsyp_corpus()[0]}

    dev_oracles = []
    for uid in dev_60_ids:
        dev_meta = dict(idx_doc["train_utt_metas"][uid])
        dev_meta["syllables"] = corpus_by_id[uid]["syllables"]
        dev_oracles.append(_prepare_utterance_oracle(uid, dev_meta))

    c0_details: list[dict[str, float]] = []
    cond_rows: dict[str, list[dict[str, Any]]] = {"Stage4_C2": [], "Stage5_C2": []}
    cond_details: dict[str, list[dict[str, float]]] = {"Stage4_C2": [], "Stage5_C2": []}
    cond_hits: dict[str, list[dict[str, Any]]] = {"Stage4_C2": [], "Stage5_C2": []}

    for d_idx, item in enumerate(dev_oracles):
        ref_span = item["ref_span"]
        ref_fast = item["ref_fast"]
        feats_p = item["feats_p"]
        ctx = item["ctx"]
        active = ctx.syl_idx_per_frame >= 0
        seed_d = 5000 + d_idx

        c0_details.append(_compute_detail_stats(feats_p.log_env, feats_p.voiced, active))

        ts = time.perf_counter()
        c4, info4 = build_stage2_control(ctx, feats_p.f0_hz, feats_p.voiced, kb2)
        c4.n_samples = len(ref_span)
        y4 = synthesize_larynx(c4, glottal_mode="parametric_lf", seed=seed_d)
        dt4 = time.perf_counter() - ts
        cond_rows["Stage4_C2"].append(evaluate_utterance(y4, ref_span, rtf=dt4 / max(1e-6, len(y4) / SR), ref_fast=ref_fast))
        cond_details["Stage4_C2"].append(_compute_detail_stats(c4.log_env, c4.voiced, active))
        cond_hits["Stage4_C2"].append(info4)

        ts = time.perf_counter()
        c5, info5 = build_stage5_control(
            ctx, feats_p.f0_hz, feats_p.voiced, kb5,
            anchor_alpha=anchor_alpha, prosody_alpha=prosody_alpha, gv_strength=gv_strength,
        )
        c5.n_samples = len(ref_span)
        y5 = synthesize_larynx(c5, glottal_mode="parametric_lf", seed=seed_d)
        dt5 = time.perf_counter() - ts
        cond_rows["Stage5_C2"].append(evaluate_utterance(y5, ref_span, rtf=dt5 / max(1e-6, len(y5) / SR), ref_fast=ref_fast))
        cond_details["Stage5_C2"].append(_compute_detail_stats(c5.log_env, c5.voiced, active))
        cond_hits["Stage5_C2"].append(info5)

    c0_agg = {k: float(np.median([d[k] for d in c0_details])) for k in c0_details[0]}
    out: dict[str, Any] = {"n_dev_60": len(dev_60_ids)}
    for cname in ("Stage4_C2", "Stage5_C2"):
        d_agg = {k: float(np.median([d[k] for d in cond_details[cname]])) for k in c0_details[0]}
        out[cname] = {
            "summary": summarize_condition(cond_rows[cname]),
            "kb_hits": _aggregate_kb_hits(cond_hits[cname]),
            "detail": {
                "v_p90_p10_1_25k_db": round(d_agg["v_p90_p10_1_25k_db"], 3),
                "v_p90_p10_25_5k_db": round(d_agg["v_p90_p10_25_5k_db"], 3),
                "cep12_40_std_vs_c0": round(d_agg["cep_std_12_40"] / max(1e-9, c0_agg["cep_std_12_40"]), 3),
            },
        }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache-path", type=Path, default=DEFAULT_S5_CACHE_PATH)
    ap.add_argument("--benchmark-dev60", action="store_true")
    ap.add_argument("--anchor-alpha", type=float, default=0.92)
    ap.add_argument("--prosody-alpha", type=float, default=0.72)
    ap.add_argument("--gv-strength", type=float, default=0.45)
    args = ap.parse_args()
    if args.benchmark_dev60:
        benchmark_dev60_stage5_vs_stage4(
            anchor_alpha=args.anchor_alpha,
            prosody_alpha=args.prosody_alpha,
            gv_strength=args.gv_strength,
        )
    else:
        build_stage5_knowledge(
            cache_path=args.cache_path,
            max_workers=args.workers,
            limit=args.limit,
        )
