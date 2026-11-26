"""Build Stage 2 unblurred K<=4 medoid acoustic knowledge base strictly from TRAIN."""
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
from scipy.fft import rfftfreq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import (  # noqa: E402
    N_BAP,
    N_CEP,
    N_FFT,
    SR,
)
from exp_20_pybot_larynx.acoustics.build_knowledge import _bin_segment  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge import (  # noqa: E402
    get_splits,
    load_v15_index_cached,
    normalize_of,
)
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import (  # noqa: E402
    N_COARSE_CEP_S2,
    N_FINAL_BINS_S2,
    N_ONSET_BINS_S2,
    STAGE2_KNOWLEDGE_PATH,
    STAGE2_SUMMARY_PATH,
    AcousticKnowledgeStage2,
    encode_next_place,
    encode_prev_fg,
)
from exp_20_pybot_larynx.arbiter import (  # noqa: E402
    NEPER_TO_DB,
    classify_final_group,
    classify_onset_place,
)
from exp_20_pybot_larynx.larynx.synth import log_env_to_cep  # noqa: E402


def _worker_init() -> None:
    """Prevent BLAS/OpenMP thread oversubscription inside worker processes."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _extract_one_utt_stage2(
    task: tuple[str, str, list[tuple[str, str, int, int, int, int, float]]],
) -> dict[str, Any]:
    """Extract 5-bin onset / 9-bin final cep64+bap6 tokens & context signatures from one TRAIN utt."""
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
    cep64 = log_env_to_cep(log_env_norm, n_cep=N_CEP).astype(np.float32)
    bap6_f32 = bap6.astype(np.float32)
    v_f32 = feats_p.voiced.astype(np.float32)
    ref_f0 = max(50.0, utt_f0_med)

    active_mask = np.zeros(n_fr, dtype=bool)
    onset_tokens: list[tuple[Any, ...]] = []
    final_tokens: list[tuple[Any, ...]] = []

    norm_syls = []
    for raw_o, raw_f, tone, start_fr, split_abs_fr, end_fr, pause_ms in syls:
        o, f = normalize_of(raw_o, raw_f)
        norm_syls.append((o, f, int(tone), int(start_fr), int(split_abs_fr), int(end_fr), float(pause_ms)))

    n_s = len(norm_syls)
    for si, (o, f, t, start_fr, split_abs_fr, end_fr, pause_ms) in enumerate(norm_syls):
        a = max(0, min(n_fr - 1, start_fr))
        b = max(a + 1, min(n_fr, end_fr))
        sp = max(a, min(b - 1, split_abs_fr))
        active_mask[a:b] = True

        if si == 0 or (a - norm_syls[si - 1][5]) >= 3 or norm_syls[si - 1][6] >= 25.0:
            prev_fg = "SIL"
        else:
            prev_fg = classify_final_group(norm_syls[si - 1][1])

        if si == n_s - 1 or pause_ms >= 25.0 or (norm_syls[si + 1][3] - b) >= 3:
            next_pl = "SIL"
        else:
            next_pl = classify_onset_place(norm_syls[si + 1][0])

        if prev_fg == "SIL":
            pos_id = 0
        elif next_pl == "SIL":
            pos_id = 2
        else:
            pos_id = 1

        prev_id = encode_prev_fg(prev_fg)
        next_id = encode_next_place(next_pl)
        dur_ms = float(b - a) * 5.0

        v_sub = v_mask[a:b]
        f0_st = float(12.0 * np.log2(np.median(feats_p.f0_hz[a:b][v_sub]) / ref_f0)) if np.any(v_sub) else 0.0

        if o and sp > a:
            on_c = _bin_segment(cep64[a:sp], N_ONSET_BINS_S2).astype(np.float32)
            on_b = _bin_segment(bap6_f32[a:sp], N_ONSET_BINS_S2).astype(np.float32)
            on_v = _bin_segment(v_f32[a:sp], N_ONSET_BINS_S2).astype(np.float32)
            o_frac = float(sp - a) / float(max(1, b - a))
            o_ms = float(sp - a) * 5.0
            onset_tokens.append((
                o, classify_final_group(f),
                prev_id, next_id, pos_id, o_ms, o_frac, f0_st,
                on_c, on_b, on_v,
            ))
            f_a = sp
        else:
            f_a = a

        if b > f_a:
            fi_c = _bin_segment(cep64[f_a:b], N_FINAL_BINS_S2).astype(np.float32)
            fi_b = _bin_segment(bap6_f32[f_a:b], N_FINAL_BINS_S2).astype(np.float32)
            fi_v = _bin_segment(v_f32[f_a:b], N_FINAL_BINS_S2).astype(np.float32)
            final_tokens.append((
                o, classify_onset_place(o), f, t,
                prev_id, next_id, pos_id, dur_ms, f0_st,
                fi_c, fi_b, fi_v,
            ))

    sil_mask = ~active_mask
    if np.any(sil_mask):
        sil_cep = np.mean(cep64[sil_mask], axis=0, dtype=np.float64)
        sil_bap = np.mean(bap6_f32[sil_mask], axis=0, dtype=np.float64)
        sil_cnt = int(np.sum(sil_mask))
    else:
        sil_cep = np.zeros(N_CEP, dtype=np.float64)
        sil_bap = np.ones(N_BAP, dtype=np.float64) * 0.95
        sil_cnt = 0

    return {
        "uid": uid,
        "utt_log_ref": utt_log_ref,
        "utt_f0_med": utt_f0_med,
        "onset_tokens": onset_tokens,
        "final_tokens": final_tokens,
        "sil_cep": sil_cep,
        "sil_bap": sil_bap,
        "sil_cnt": sil_cnt,
    }


def _select_medoids_kmeans(
    feat_x: np.ndarray,
    k_max: int = 4,
    seed: int = 2026,
) -> tuple[list[int], list[int]]:
    """Cluster `(N, D)` token trajectories into `K <= k_max` clusters and return `(medoid_indices, cluster_sizes)`.

    Pure deterministic NumPy K-means++ (0 ML/sklearn weights). Picks exact real-token
    medoids closest to each cluster centroid so spectral formants remain unblurred.
    """
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
        k = min(k_max, 3)
    else:
        k = min(k_max, 4)

    rng = np.random.default_rng(seed + n)
    # Deterministic K-means++ init starting from token closest to global mean
    g_mean = np.mean(feat_x, axis=0)
    first_idx = int(np.argmin(np.sum((feat_x - g_mean[None, :]) ** 2, axis=1)))
    centers = [feat_x[first_idx].copy()]
    min_d2 = np.sum((feat_x - centers[0][None, :]) ** 2, axis=1)
    for _ in range(1, k):
        tot = float(np.sum(min_d2))
        if tot <= 1e-12:
            break
        # Pick deterministic farthest-or-weighted point (use weighted choice with fixed rng)
        probs = min_d2 / tot
        nxt = int(rng.choice(n, p=probs))
        centers.append(feat_x[nxt].copy())
        d2_new = np.sum((feat_x - centers[-1][None, :]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)

    c_arr = np.stack(centers, axis=0)
    k_actual = c_arr.shape[0]
    # Run 12 Lloyd iterations
    x_norm2 = np.sum(feat_x * feat_x, axis=1, keepdims=True)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(12):
        c_norm2 = np.sum(c_arr * c_arr, axis=1)[None, :]
        dist2 = x_norm2 + c_norm2 - 2.0 * (feat_x @ c_arr.T)
        new_labels = np.argmin(dist2, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for ci in range(k_actual):
            members = np.where(labels == ci)[0]
            if len(members) > 0:
                c_arr[ci] = np.mean(feat_x[members], axis=0)

    # Drop empty or singleton outlier clusters when n >= 8
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


def _build_medoid_table(
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
    k_max: int = 4,
    cap_tokens: int = 2000,
) -> tuple[dict[str, np.ndarray], int]:
    """Cluster each key's tokens into K<=4 unblurred medoids and pack as CSR arrays."""
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

        sub_cep = cep_arr[idx_arr]  # (M, t_bins, 64)
        # Clustering feature: weighted cep[0] + formant/spectral shape cep[1:40]
        feat = np.concatenate([0.25 * sub_cep[:, :, :1], sub_cep[:, :, 1:40]], axis=2).reshape(len(idx_arr), -1).astype(np.float64)
        med_local_ids, cl_sizes = _select_medoids_kmeans(feat, k_max=k_max, seed=2026)

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
        f"{prefix}_cep": np.stack(med_cep, axis=0).astype(np.float32) if n_med else np.zeros((0, t_bins, N_CEP), dtype=np.float32),
        f"{prefix}_bap": np.stack(med_bap, axis=0).astype(np.float32) if n_med else np.zeros((0, t_bins, N_BAP), dtype=np.float32),
        f"{prefix}_vp": np.stack(med_vp, axis=0).astype(np.float32) if n_med else np.zeros((0, t_bins), dtype=np.float32),
        f"{prefix}_ctx_prev": np.array(med_prev, dtype=np.int8),
        f"{prefix}_ctx_next": np.array(med_next, dtype=np.int8),
        f"{prefix}_ctx_pos": np.array(med_pos, dtype=np.int8),
        f"{prefix}_ctx_dur_ms": np.array(med_dur, dtype=np.float32),
        f"{prefix}_ctx_f0_st": np.array(med_f0, dtype=np.float32),
        f"{prefix}_cluster_sz": np.array(med_csz, dtype=np.int32),
    }
    return out, n_keys


def build_stage2_knowledge(
    out_path: Path = STAGE2_KNOWLEDGE_PATH,
    summary_path: Path = STAGE2_SUMMARY_PATH,
    max_workers: int | None = 20,
    limit: int | None = None,
    cache_path: Path | None = None,
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

    if cache_path is not None and cache_path.exists():
        print(f"Loading cached Stage 2 TRAIN token features from {cache_path}...", flush=True)
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        assert int(cached["n_utts"]) == len(selected_ids), "Cache size mismatch"
        n_utts = int(cached["n_utts"])
        utt_log_refs = cached["utt_log_refs"]
        utt_f0_meds = cached["utt_f0_meds"]
        sil_cep64 = cached["sil_cep64"]
        sil_bap6 = cached["sil_bap6"]
        onset_frac_list = cached["onset_frac_list"]
        onset_ms_list = cached["onset_ms_list"]
        on_cep = cached["on_cep"]
        on_bap = cached["on_bap"]
        on_vp = cached["on_vp"]
        on_prev = cached["on_prev"]
        on_next = cached["on_next"]
        on_pos = cached["on_pos"]
        on_dur = cached["on_dur"]
        on_f0 = cached["on_f0"]
        on_grp_idx = cached["on_grp_idx"]
        on_only_idx = cached["on_only_idx"]
        fi_cep = cached["fi_cep"]
        fi_bap = cached["fi_bap"]
        fi_vp = cached["fi_vp"]
        fi_prev = cached["fi_prev"]
        fi_next = cached["fi_next"]
        fi_pos = cached["fi_pos"]
        fi_dur = cached["fi_dur"]
        fi_f0 = cached["fi_f0"]
        fin_oft_idx = cached["fin_oft_idx"]
        fin_pft_idx = cached["fin_pft_idx"]
        fin_ft_idx = cached["fin_ft_idx"]
        fin_f_idx = cached["fin_f_idx"]
    else:
        idx = load_v15_index_cached()
        metas = idx["train_utt_metas"]
        tokens = idx["tokens"]
        by_utt_syl = idx["by_utt_syl"]

        tasks: list[tuple[str, str, list[tuple[str, str, int, int, int, int, float]]]] = []
        for uid in selected_ids:
            m = metas[uid]
            syl_list: list[tuple[str, str, int, int, int, int, float]] = []
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
                    float(tok.pause_after_ms),
                ))
            if syl_list:
                tasks.append((uid, m["wav_path"], syl_list))

        n_utts = len(tasks)
        workers = max_workers if max_workers is not None else min(20, os.cpu_count() or 4)
        print(f"Extracting Stage 2 (5-bin onset / 9-bin final) from {n_utts} TRAIN utts (workers={workers})...", flush=True)

        utt_log_refs: list[float] = []
        utt_f0_meds: list[float] = []
        sil_cep_sum = np.zeros(N_CEP, dtype=np.float64)
        sil_bap_sum = np.zeros(N_BAP, dtype=np.float64)
        sil_tot = 0

        onset_frac_list: dict[str, list[float]] = defaultdict(list)
        onset_ms_list: dict[str, list[float]] = defaultdict(list)

        on_cep_l: list[np.ndarray] = []
        on_bap_l: list[np.ndarray] = []
        on_vp_l: list[np.ndarray] = []
        on_prev_l: list[int] = []
        on_next_l: list[int] = []
        on_pos_l: list[int] = []
        on_dur_l: list[float] = []
        on_f0_l: list[float] = []
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
        fin_oft_idx: dict[str, list[int]] = defaultdict(list)
        fin_pft_idx: dict[str, list[int]] = defaultdict(list)
        fin_ft_idx: dict[str, list[int]] = defaultdict(list)
        fin_f_idx: dict[str, list[int]] = defaultdict(list)

        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
            for idx_done, res in enumerate(ex.map(_extract_one_utt_stage2, tasks, chunksize=24), 1):
                utt_log_refs.append(res["utt_log_ref"])
                utt_f0_meds.append(res["utt_f0_med"])
                if res["sil_cnt"] > 0:
                    sil_cep_sum += res["sil_cep"] * res["sil_cnt"]
                    sil_bap_sum += res["sil_bap"] * res["sil_cnt"]
                    sil_tot += res["sil_cnt"]

                for o, fg, prev_id, next_id, pos_id, o_ms, o_frac, f0_st, oc, ob, ov in res["onset_tokens"]:
                    oi = len(on_cep_l)
                    on_cep_l.append(oc)
                    on_bap_l.append(ob)
                    on_vp_l.append(ov)
                    on_prev_l.append(prev_id)
                    on_next_l.append(next_id)
                    on_pos_l.append(pos_id)
                    on_dur_l.append(o_ms)
                    on_f0_l.append(f0_st)
                    on_grp_idx[f"{o}|{fg}"].append(oi)
                    on_only_idx[o].append(oi)
                    onset_frac_list[o].append(o_frac)
                    onset_ms_list[o].append(o_ms)

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
                    fin_oft_idx[f"{o}|{f}|{t}"].append(fi)
                    fin_pft_idx[f"{place}|{f}|{t}"].append(fi)
                    fin_ft_idx[f"{f}|{t}"].append(fi)
                    fin_f_idx[f].append(fi)

                if idx_done % 1000 == 0 or idx_done == n_utts:
                    print(f"  processed {idx_done}/{n_utts} ({time.perf_counter() - t0:.1f}s)", flush=True)

        sil_cep64 = (sil_cep_sum / max(1, sil_tot)).astype(np.float32)
        sil_bap6 = (sil_bap_sum / max(1, sil_tot)).astype(np.float32)
        on_cep = np.stack(on_cep_l, axis=0).astype(np.float32)
        on_bap = np.stack(on_bap_l, axis=0).astype(np.float32)
        on_vp = np.stack(on_vp_l, axis=0).astype(np.float32)
        on_prev = np.array(on_prev_l, dtype=np.int8)
        on_next = np.array(on_next_l, dtype=np.int8)
        on_pos = np.array(on_pos_l, dtype=np.int8)
        on_dur = np.array(on_dur_l, dtype=np.float32)
        on_f0 = np.array(on_f0_l, dtype=np.float32)

        fi_cep = np.stack(fi_cep_l, axis=0).astype(np.float32)
        fi_bap = np.stack(fi_bap_l, axis=0).astype(np.float32)
        fi_vp = np.stack(fi_vp_l, axis=0).astype(np.float32)
        fi_prev = np.array(fi_prev_l, dtype=np.int8)
        fi_next = np.array(fi_next_l, dtype=np.int8)
        fi_pos = np.array(fi_pos_l, dtype=np.int8)
        fi_dur = np.array(fi_dur_l, dtype=np.float32)
        fi_f0 = np.array(fi_f0_l, dtype=np.float32)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "n_utts": n_utts,
                    "utt_log_refs": utt_log_refs,
                    "utt_f0_meds": utt_f0_meds,
                    "sil_cep64": sil_cep64,
                    "sil_bap6": sil_bap6,
                    "onset_frac_list": dict(onset_frac_list),
                    "onset_ms_list": dict(onset_ms_list),
                    "on_cep": on_cep, "on_bap": on_bap, "on_vp": on_vp,
                    "on_prev": on_prev, "on_next": on_next, "on_pos": on_pos,
                    "on_dur": on_dur, "on_f0": on_f0,
                    "on_grp_idx": dict(on_grp_idx), "on_only_idx": dict(on_only_idx),
                    "fi_cep": fi_cep, "fi_bap": fi_bap, "fi_vp": fi_vp,
                    "fi_prev": fi_prev, "fi_next": fi_next, "fi_pos": fi_pos,
                    "fi_dur": fi_dur, "fi_f0": fi_f0,
                    "fin_oft_idx": dict(fin_oft_idx), "fin_pft_idx": dict(fin_pft_idx),
                    "fin_ft_idx": dict(fin_ft_idx), "fin_f_idx": dict(fin_f_idx),
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Cached extracted tokens to {cache_path}", flush=True)

    t_cl = time.perf_counter()
    print(f"Clustering {len(on_cep)} onset + {len(fi_cep)} final tokens into K<=4 medoids...", flush=True)

    payload: dict[str, np.ndarray] = {}
    t_on_grp, n_on_grp = _build_medoid_table(
        "on_grp", on_grp_idx, on_cep, on_bap, on_vp, on_prev, on_next, on_pos, on_dur, on_f0, N_ONSET_BINS_S2
    )
    t_on_only, n_on_only = _build_medoid_table(
        "on_only", on_only_idx, on_cep, on_bap, on_vp, on_prev, on_next, on_pos, on_dur, on_f0, N_ONSET_BINS_S2
    )
    t_fin_oft, n_fin_oft = _build_medoid_table(
        "fin_oft", fin_oft_idx, fi_cep, fi_bap, fi_vp, fi_prev, fi_next, fi_pos, fi_dur, fi_f0, N_FINAL_BINS_S2
    )
    t_fin_pft, n_fin_pft = _build_medoid_table(
        "fin_pft", fin_pft_idx, fi_cep, fi_bap, fi_vp, fi_prev, fi_next, fi_pos, fi_dur, fi_f0, N_FINAL_BINS_S2
    )
    t_fin_ft, n_fin_ft = _build_medoid_table(
        "fin_ft", fin_ft_idx, fi_cep, fi_bap, fi_vp, fi_prev, fi_next, fi_pos, fi_dur, fi_f0, N_FINAL_BINS_S2
    )
    t_fin_f, n_fin_f = _build_medoid_table(
        "fin_f", fin_f_idx, fi_cep, fi_bap, fi_vp, fi_prev, fi_next, fi_pos, fi_dur, fi_f0, N_FINAL_BINS_S2
    )

    payload.update(t_on_grp)
    payload.update(t_on_only)
    payload.update(t_fin_oft)
    payload.update(t_fin_pft)
    payload.update(t_fin_ft)
    payload.update(t_fin_f)

    # Compact context-conditioned low-quefrency coarse anchors (cep[:, :N_COARSE_CEP_S2], float16)
    def _build_coarse_ctx(prefix: str, key_to_idxs: dict[str, list[int]], cep_arr: np.ndarray, t_bins: int) -> int:
        c_keys = sorted(key_to_idxs.keys())
        if not c_keys:
            payload[f"{prefix}_keys"] = np.array([], dtype="<U1")
            payload[f"{prefix}_cep"] = np.zeros((0, t_bins, N_COARSE_CEP_S2), dtype=np.float16)
            return 0
        max_len = max(len(k) for k in c_keys)
        payload[f"{prefix}_keys"] = np.array(c_keys, dtype=f"<U{max_len}")
        payload[f"{prefix}_cep"] = np.stack(
            [np.mean(cep_arr[key_to_idxs[k], :, :N_COARSE_CEP_S2], axis=0, dtype=np.float64) for k in c_keys],
            axis=0,
        ).astype(np.float16)
        return len(c_keys)

    on_ctx_p_idx: dict[str, list[int]] = defaultdict(list)
    for k, idxs in on_grp_idx.items():
        for i in idxs:
            on_ctx_p_idx[f"{k}|{int(on_prev[i])}"].append(i)
    on_ctx_p_idx = {k: v for k, v in on_ctx_p_idx.items() if len(v) >= 5}

    fin_ctx_pn_idx: dict[str, list[int]] = defaultdict(list)
    fin_ctx_np_idx: dict[str, list[int]] = defaultdict(list)
    fin_ctx_pp_idx: dict[str, list[int]] = defaultdict(list)
    fin_ctx_pos_idx: dict[str, list[int]] = defaultdict(list)
    for k, idxs in fin_oft_idx.items():
        for i in idxs:
            pv, nx, ps = int(fi_prev[i]), int(fi_next[i]), int(fi_pos[i])
            fin_ctx_pn_idx[f"{k}|{pv}|{nx}"].append(i)
            fin_ctx_np_idx[f"{k}|{nx}|{ps}"].append(i)
            fin_ctx_pp_idx[f"{k}|{pv}|{ps}"].append(i)
            fin_ctx_pos_idx[f"{k}|{ps}"].append(i)
    fin_ctx_pn_idx = {k: v for k, v in fin_ctx_pn_idx.items() if len(v) >= 4}
    fin_ctx_np_idx = {k: v for k, v in fin_ctx_np_idx.items() if len(v) >= 4}
    fin_ctx_pp_idx = {k: v for k, v in fin_ctx_pp_idx.items() if len(v) >= 4}

    fin_ctx_pft_pos_idx: dict[str, list[int]] = defaultdict(list)
    for k, idxs in fin_pft_idx.items():
        for i in idxs:
            fin_ctx_pft_pos_idx[f"{k}|{int(fi_pos[i])}"].append(i)

    n_on_ctx_p = _build_coarse_ctx("on_ctx_p", on_ctx_p_idx, on_cep, N_ONSET_BINS_S2)
    n_on_ctx_base = _build_coarse_ctx("on_ctx_base", on_grp_idx, on_cep, N_ONSET_BINS_S2)
    n_fin_ctx_pn = _build_coarse_ctx("fin_ctx_pn", fin_ctx_pn_idx, fi_cep, N_FINAL_BINS_S2)
    n_fin_ctx_np = _build_coarse_ctx("fin_ctx_np", fin_ctx_np_idx, fi_cep, N_FINAL_BINS_S2)
    n_fin_ctx_pp = _build_coarse_ctx("fin_ctx_pp", fin_ctx_pp_idx, fi_cep, N_FINAL_BINS_S2)
    n_fin_ctx_pos = _build_coarse_ctx("fin_ctx_pos", fin_ctx_pos_idx, fi_cep, N_FINAL_BINS_S2)
    n_fin_ctx_pft = _build_coarse_ctx("fin_ctx_pft_pos", fin_ctx_pft_pos_idx, fi_cep, N_FINAL_BINS_S2)
    print(
        f"Built coarse ctx anchors: on={n_on_ctx_p}/{n_on_ctx_base}, "
        f"fin={n_fin_ctx_pn}/{n_fin_ctx_np}/{n_fin_ctx_pp}/{n_fin_ctx_pos}/{n_fin_ctx_pft}",
        flush=True,
    )

    # Global fallback medoid (closest single real token to global mean)
    on_g_mean = np.mean(on_cep[:, :, 1:40], axis=0)
    on_g_idx = int(np.argmin(np.sum((on_cep[:, :, 1:40] - on_g_mean[None, :, :]) ** 2, axis=(1, 2))))
    payload["on_glob_cep"] = on_cep[on_g_idx].astype(np.float32)
    payload["on_glob_bap"] = on_bap[on_g_idx].astype(np.float32)
    payload["on_glob_vp"] = on_vp[on_g_idx].astype(np.float32)

    fi_g_mean = np.mean(fi_cep[:, :, 1:40], axis=0)
    fi_g_idx = int(np.argmin(np.sum((fi_cep[:, :, 1:40] - fi_g_mean[None, :, :]) ** 2, axis=(1, 2))))
    payload["fin_glob_cep"] = fi_cep[fi_g_idx].astype(np.float32)
    payload["fin_glob_bap"] = fi_bap[fi_g_idx].astype(np.float32)
    payload["fin_glob_vp"] = fi_vp[fi_g_idx].astype(np.float32)

    on_keys = sorted(onset_frac_list.keys())
    max_ok = max((len(k) for k in on_keys), default=1)
    payload["onset_frac_keys"] = np.array(on_keys, dtype=f"<U{max_ok}")
    payload["onset_frac_vals"] = np.array([float(np.median(onset_frac_list[k])) for k in on_keys], dtype=np.float32)
    payload["onset_ms_vals"] = np.array([float(np.median(onset_ms_list[k])) for k in on_keys], dtype=np.float32)

    payload["sil_cep64"] = sil_cep64
    payload["sil_bap6"] = sil_bap6
    payload["speaker_f0_med_hz"] = np.array([float(np.median(utt_f0_meds))], dtype=np.float32)
    payload["ref_log_amp"] = np.array([float(np.median(utt_log_refs))], dtype=np.float32)
    payload["n_train_utts"] = np.array([n_utts], dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    size_kb = out_path.stat().st_size / 1024.0

    kb2 = AcousticKnowledgeStage2.load(out_path)
    assert kb2.n_train_utts == n_utts

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
            "on_grp": n_on_grp,
            "on_only": n_on_only,
            "fin_oft": n_fin_oft,
            "fin_pft": n_fin_pft,
            "fin_ft": n_fin_ft,
            "fin_f": n_fin_f,
        },
        "medoid_counts": {
            "on_grp": int(len(t_on_grp["on_grp_cep"])),
            "on_only": int(len(t_on_only["on_only_cep"])),
            "fin_oft": int(len(t_fin_oft["fin_oft_cep"])),
            "fin_pft": int(len(t_fin_pft["fin_pft_cep"])),
            "fin_ft": int(len(t_fin_ft["fin_ft_cep"])),
            "fin_f": int(len(t_fin_f["fin_f_cep"])),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out_path} ({size_kb:.1f} KB) in {summary['build_time_s']}s (cluster {summary['cluster_time_s']}s)", flush=True)
    return summary


_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_M_1_25 = (_FREQS >= 1000.0) & (_FREQS <= 2500.0)
_M_25_5 = (_FREQS >= 2500.0) & (_FREQS <= 5000.0)
_M_1_11 = (_FREQS >= 1000.0) & (_FREQS <= 11000.0)


def _compute_detail_stats(log_env: np.ndarray, voiced: np.ndarray, active: np.ndarray) -> dict[str, float]:
    env_db = log_env * NEPER_TO_DB
    cep = log_env_to_cep(log_env, n_cep=N_CEP)

    v_act = voiced & active
    u_act = (~voiced) & active

    v_1_25 = float(np.median(np.percentile(env_db[v_act][:, _M_1_25], 90, axis=1) - np.percentile(env_db[v_act][:, _M_1_25], 10, axis=1))) if np.any(v_act) else 0.0
    v_25_5 = float(np.median(np.percentile(env_db[v_act][:, _M_25_5], 90, axis=1) - np.percentile(env_db[v_act][:, _M_25_5], 10, axis=1))) if np.any(v_act) else 0.0
    uv_1_11 = float(np.median(np.percentile(env_db[u_act][:, _M_1_11], 90, axis=1) - np.percentile(env_db[u_act][:, _M_1_11], 10, axis=1))) if np.any(u_act) else 0.0

    c_act = cep[active]
    return {
        "v_p90_p10_1_25k_db": v_1_25,
        "v_p90_p10_25_5k_db": v_25_5,
        "uv_p90_p10_1_11k_db": uv_1_11,
        "cep_std_12_24": float(np.std(c_act[:, 12:24])),
        "cep_std_24_40": float(np.std(c_act[:, 24:40])),
        "cep_std_12_40": float(np.std(c_act[:, 12:40])),
        "cep_std_40_64": float(np.std(c_act[:, 40:64])),
    }


def benchmark_dev60_stage2(
    full_path: Path | None = None,
    stage2_path: Path = STAGE2_KNOWLEDGE_PATH,
) -> dict[str, Any]:
    """Benchmark C0, C2_full, and C2_stage2 on DEV_60 ONLY (never HELDOUT_20)."""
    from exp_20_pybot_larynx import EXP_DIR
    from exp_20_pybot_larynx.acoustics.knowledge import AcousticKnowledge, build_base_control
    from exp_20_pybot_larynx.acoustics.knowledge_stage2 import build_stage2_control
    from exp_20_pybot_larynx.judge import evaluate_utterance, summarize_condition
    from exp_20_pybot_larynx.larynx.synth import LarynxControl, synthesize_larynx
    from exp_20_pybot_larynx.stage0_probe import _aggregate_kb_hits, _prepare_utterance_oracle
    from exp_v15_bznsyp_tts.index_bznsyp_v15 import parse_bznsyp_corpus

    f_path = full_path or (EXP_DIR / "acoustics" / "knowledge_full.npz")
    kb_full = AcousticKnowledge.load(f_path)
    kb_s2 = AcousticKnowledgeStage2.load(stage2_path)

    _, dev_60_ids, _ = get_splits()
    idx_doc = load_v15_index_cached()
    corpus_records, _ = parse_bznsyp_corpus()
    corpus_by_id = {r["id"]: r for r in corpus_records}

    dev_oracles = []
    for uid in dev_60_ids:
        dev_meta = dict(idx_doc["train_utt_metas"][uid])
        dev_meta["syllables"] = corpus_by_id[uid]["syllables"]
        dev_oracles.append(_prepare_utterance_oracle(uid, dev_meta))

    c0_details: list[dict[str, float]] = []
    cond_rows: dict[str, list[dict[str, Any]]] = {"C2_full": [], "C2_stage2": []}
    cond_details: dict[str, list[dict[str, float]]] = {"C2_full": [], "C2_stage2": []}
    cond_hits: dict[str, list[dict[str, Any]]] = {"C2_full": [], "C2_stage2": []}

    for d_idx, item in enumerate(dev_oracles):
        ref_span = item["ref_span"]
        ref_fast = item["ref_fast"]
        feats_p = item["feats_p"]
        ctx = item["ctx"]
        active = ctx.syl_idx_per_frame >= 0
        seed_d = 5000 + d_idx

        c0_details.append(_compute_detail_stats(feats_p.log_env, feats_p.voiced, active))

        # C2_full
        ts = time.perf_counter()
        c2_f, info_f = build_base_control(ctx, feats_p.f0_hz, feats_p.voiced, kb_full, mode="dynamic")
        c2_f.n_samples = len(ref_span)
        y_f = synthesize_larynx(c2_f, glottal_mode="parametric_lf", seed=seed_d)
        dt_f = time.perf_counter() - ts
        cond_rows["C2_full"].append(evaluate_utterance(y_f, ref_span, rtf=dt_f / max(1e-6, len(y_f) / SR), ref_fast=ref_fast))
        cond_details["C2_full"].append(_compute_detail_stats(c2_f.log_env, c2_f.voiced, active))
        cond_hits["C2_full"].append(info_f)

        # C2_stage2
        ts = time.perf_counter()
        c2_s2, info_s2 = build_stage2_control(ctx, feats_p.f0_hz, feats_p.voiced, kb_s2)
        c2_s2.n_samples = len(ref_span)
        y_s2 = synthesize_larynx(c2_s2, glottal_mode="parametric_lf", seed=seed_d)
        dt_s2 = time.perf_counter() - ts
        cond_rows["C2_stage2"].append(evaluate_utterance(y_s2, ref_span, rtf=dt_s2 / max(1e-6, len(y_s2) / SR), ref_fast=ref_fast))
        cond_details["C2_stage2"].append(_compute_detail_stats(c2_s2.log_env, c2_s2.voiced, active))
        cond_hits["C2_stage2"].append(info_s2)

    c0_agg = {k: float(np.median([d[k] for d in c0_details])) for k in c0_details[0]}
    out: dict[str, Any] = {
        "n_dev_60": len(dev_60_ids),
        "c0_detail": {k: round(v, 4) for k, v in c0_agg.items()},
    }
    for cname in ("C2_full", "C2_stage2"):
        d_agg = {k: float(np.median([d[k] for d in cond_details[cname]])) for k in c0_details[0]}
        out[cname] = {
            "summary": summarize_condition(cond_rows[cname]),
            "kb_hits": _aggregate_kb_hits(cond_hits[cname]),
            "detail": {
                "v_p90_p10_1_25k_db": round(d_agg["v_p90_p10_1_25k_db"], 3),
                "v_p90_p10_25_5k_db": round(d_agg["v_p90_p10_25_5k_db"], 3),
                "uv_p90_p10_1_11k_db": round(d_agg["uv_p90_p10_1_11k_db"], 3),
                "cep12_24_std_vs_c0": round(d_agg["cep_std_12_24"] / max(1e-9, c0_agg["cep_std_12_24"]), 3),
                "cep24_40_std_vs_c0": round(d_agg["cep_std_24_40"] / max(1e-9, c0_agg["cep_std_24_40"]), 3),
                "cep12_40_std_vs_c0": round(d_agg["cep_std_12_40"] / max(1e-9, c0_agg["cep_std_12_40"]), 3),
                "cep40_64_std_vs_c0": round(d_agg["cep_std_40_64"] / max(1e-9, c0_agg["cep_std_40_64"]), 3),
            },
        }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache-path", type=Path, default=None)
    ap.add_argument("--benchmark-dev60", action="store_true")
    args = ap.parse_args()
    if args.benchmark_dev60:
        benchmark_dev60_stage2()
    else:
        build_stage2_knowledge(
            max_workers=args.workers,
            limit=args.limit,
            cache_path=args.cache_path,
        )
