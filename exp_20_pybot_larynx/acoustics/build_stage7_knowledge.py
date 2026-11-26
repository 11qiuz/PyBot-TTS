"""Build Stage 7 Hierarchical Bayesian Ridge + Unblurred Medoid knowledge base strictly from TRAIN."""
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
from exp_20_pybot_larynx.acoustics.build_stage5_knowledge import (  # noqa: E402
    DEFAULT_S5_CACHE_PATH,
    _build_medoid_table_s5,
    _worker_init,
    extract_or_load_train_tokens_s5,
)
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
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import (  # noqa: E402
    D_ACOUSTIC_FEAT_S7,
    N_COARSE_CEP_S7,
    STAGE7_KNOWLEDGE_PATH,
    STAGE7_SUMMARY_PATH,
    STAGE7_TRUEENV_PATH,
    STAGE7_TRUEENV_SUMMARY_PATH,
    AcousticKnowledgeStage7,
    extract_stage7_acoustic_features,
)
from exp_20_pybot_larynx.arbiter import (  # noqa: E402
    classify_final_group,
    classify_onset_place,
)
from exp_20_pybot_larynx.larynx.synth import log_env_to_cep  # noqa: E402

DEFAULT_S7_TRUEENV_CACHE = Path(
    r"C:\Users\MECHREVO\.gemini\antigravity\brain\f67e0437-e98d-4cb1-bbfd-da72d02478db\scratch\train_tokens_s7_trueenv.pkl"
)


def _extract_one_utt_trueenv(
    task: tuple[str, str, str, list[tuple[str, str, str, int, int, int, int, float, int]]],
) -> dict[str, Any]:
    """Extract Stage 7 TrueHarmEnv tokens strictly from one TRAIN utterance."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from exp_20_pybot_larynx.larynx.synth_v2 import analyze_true_harmonic_env
    from exp_v15_bznsyp_tts.index_bznsyp_v15 import read_bznsyp_wav_24k

    uid, text, wav_rel, syls = task
    x24 = read_bznsyp_wav_24k(REPO_ROOT / wav_rel)
    log_env, f0_hz, voiced, bap6 = analyze_true_harmonic_env(x24, sr=SR, bap_scale=0.50)

    n_fr = len(f0_hz)
    v_mask = voiced & (f0_hz > 50.0)
    if np.any(v_mask):
        utt_log_ref = float(np.median(np.mean(log_env[v_mask, 1:257], axis=1)))
        utt_f0_med = float(np.median(f0_hz[v_mask]))
    else:
        utt_log_ref = 0.0
        utt_f0_med = 247.0

    log_env_norm = log_env - utt_log_ref
    cep64 = log_env_to_cep(log_env_norm, n_cep=N_CEP).astype(np.float16)
    bap6_f16 = bap6.astype(np.float16)
    v_f16 = voiced.astype(np.float16)
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

        pos_id = 0 if prev_fg == "SIL" else (2 if next_pl == "SIL" else 1)
        prev_id = encode_prev_fg(prev_fg)
        next_id = encode_next_place(next_pl)
        dur_ms = float(b - a) * 5.0
        o_ms = float(max(0, sp - a)) * 5.0
        v_sub = v_mask[a:b]
        f0_st = float(12.0 * np.log2(np.median(f0_hz[a:b][v_sub]) / ref_f0)) if np.any(v_sub) else 0.0

        on_uv_frac = -1.0
        if o and sp > a:
            on_c = _bin_segment(cep64[a:sp].astype(np.float32), N_ONSET_BINS_S2).astype(np.float16)
            on_b = _bin_segment(bap6_f16[a:sp].astype(np.float32), N_ONSET_BINS_S2).astype(np.float16)
            on_v = _bin_segment(v_f16[a:sp].astype(np.float32), N_ONSET_BINS_S2).astype(np.float16)
            o_frac = float(sp - a) / float(max(1, b - a))
            on_uv_frac = float(np.mean(~voiced[a:sp]))
            onset_tokens.append((
                o, f, classify_final_group(f),
                prev_id, next_id, pos_id, o_ms, o_frac, on_uv_frac, f0_st,
                on_c, on_b, on_v,
            ))
            f_a = sp
        else:
            f_a = a

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
            vf_idx = np.where(v_mask[f_a:b])[0]
            if len(vf_idx) >= 2:
                u_v = vf_idx.astype(np.float64) / float(max(1, (b - f_a) - 1))
                st_v = 12.0 * np.log2(f0_hz[f_a:b][vf_idx] / ref_f0)
                p3 = np.interp([0.20, 0.50, 0.80], u_v, st_v)
                f3_st = (round(float(p3[0]), 3), round(float(p3[1]), 3), round(float(p3[2]), 3))
            elif len(vf_idx) == 1:
                val = round(float(12.0 * np.log2(f0_hz[f_a:b][vf_idx[0]] / ref_f0)), 3)
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


def extract_or_load_train_tokens_trueenv(
    cache_path: Path = DEFAULT_S7_TRUEENV_CACHE,
    *,
    max_workers: int | None = 20,
    limit: int | None = None,
) -> dict[str, Any]:
    """Extract all TRAIN Stage 7 TrueHarmEnv tokens in parallel once, or load cached pickle."""
    train_ids, dev_60, held_ids = get_splits()
    train_set = set(train_ids)
    dev_set = set(dev_60)
    held_set = set(held_ids)
    assert not (train_set & dev_set) and not (train_set & held_set) and not (dev_set & held_set)

    selected_ids = sorted(train_ids)[:limit] if limit is not None else sorted(train_ids)
    assert not (set(selected_ids) & dev_set) and not (set(selected_ids) & held_set)

    if cache_path.exists():
        print(f"Loading cached Stage 7 TrueEnv TRAIN tokens from {cache_path}...", flush=True)
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if int(cached["n_utts"]) == len(selected_ids):
            return cached

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
    print(f"Extracting TrueHarmEnv Stage 7 TRAIN tokens ({n_utts} utts, workers={max_workers})...", flush=True)

    utt_log_refs, utt_f0_meds = [], []
    sil_cep_sum = np.zeros(N_CEP, dtype=np.float64)
    sil_bap_sum = np.zeros(N_BAP, dtype=np.float64)
    sil_tot = 0

    onset_frac_list: dict[str, list[float]] = defaultdict(list)
    onset_ms_list: dict[str, list[float]] = defaultdict(list)
    onset_uv_list: dict[str, list[float]] = defaultdict(list)

    on_cep_l, on_bap_l, on_vp_l = [], [], []
    on_prev_l, on_next_l, on_pos_l, on_dur_l, on_f0_l = [], [], [], [], []
    on_of_idx: dict[str, list[int]] = defaultdict(list)
    on_grp_idx: dict[str, list[int]] = defaultdict(list)
    on_only_idx: dict[str, list[int]] = defaultdict(list)

    fi_cep_l, fi_bap_l, fi_vp_l = [], [], []
    fi_prev_l, fi_next_l, fi_pos_l, fi_dur_l, fi_f0_l, fi_tone_l = [], [], [], [], [], []
    fin_oft_idx: dict[str, list[int]] = defaultdict(list)
    fin_pft_idx: dict[str, list[int]] = defaultdict(list)
    fin_ft_idx: dict[str, list[int]] = defaultdict(list)
    fin_f_idx: dict[str, list[int]] = defaultdict(list)
    utt_prosody_records: dict[str, dict[str, Any]] = {}

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init) as ex:
        for idx_done, res in enumerate(ex.map(_extract_one_utt_trueenv, tasks, chunksize=32), 1):
            uid = res["uid"]
            utt_log_refs.append(res["utt_log_ref"])
            utt_f0_meds.append(res["utt_f0_med"])
            sc = int(res["sil_cnt"])
            if sc > 0:
                sil_cep_sum += sc * res["sil_cep"]
                sil_bap_sum += sc * res["sil_bap"]
                sil_tot += sc

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
                print(f"  TrueEnv extracted {idx_done}/{n_utts} ({time.perf_counter() - t0:.1f}s)", flush=True)

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
    print(f"Cached TrueEnv TRAIN tokens to {cache_path} in {time.perf_counter() - t0:.1f}s", flush=True)
    return payload


def _fit_ridge(
    X_mat: np.ndarray,
    Y_mat: np.ndarray,
    idxs: list[int],
    lam: float,
    prior_W: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(idxs, dtype=np.int64)
    X = X_mat[arr].astype(np.float64)
    Y = Y_mat[arr].astype(np.float64)
    reg = np.eye(X.shape[1], dtype=np.float64) * float(lam)
    reg[0, 0] = 0.0 if prior_W is None else 2.0
    XtX = X.T @ X
    XtY = X.T @ Y
    rhs = XtY if prior_W is None else (XtY + reg @ prior_W)
    return np.linalg.solve(XtX + reg, rhs)


def _pack_weights(prefix: str, w_map: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = sorted(w_map.keys())
    if not keys:
        return {f"{prefix}_keys": np.array([], dtype="<U1"), f"{prefix}_w": np.zeros((0,), dtype=np.float16)}
    max_len = max(len(k) for k in keys)
    arr = np.stack([w_map[k] for k in keys], axis=0).astype(np.float16)
    return {
        f"{prefix}_keys": np.array(keys, dtype=f"<U{max_len}"),
        f"{prefix}_w": arr,
    }


def _pack_protos(prefix: str, p_map: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = sorted(p_map.keys())
    if not keys:
        return {f"{prefix}_keys": np.array([], dtype="<U1"), f"{prefix}_vec": np.zeros((0, 8), dtype=np.float32)}
    max_len = max(len(k) for k in keys)
    arr = np.stack([p_map[k] for k in keys], axis=0).astype(np.float32)
    return {
        f"{prefix}_keys": np.array(keys, dtype=f"<U{max_len}"),
        f"{prefix}_vec": arr,
    }


def build_stage7_knowledge(
    out_path: Path = STAGE7_KNOWLEDGE_PATH,
    summary_path: Path = STAGE7_SUMMARY_PATH,
    cache_path: Path = DEFAULT_S5_CACHE_PATH,
    *,
    is_trueenv: bool = False,
    c_dim: int = N_COARSE_CEP_S7,
    min_tokens_oft_w: int = 45,
    min_tokens_pft_w: int = 30,
    min_tokens_on_of_w: int = 45,
    max_workers: int | None = 20,
    limit: int | None = None,
) -> dict[str, Any]:
    """Fit Hierarchical Bayesian Ridge + Unblurred Medoids strictly from TRAIN tokens."""
    t0 = time.perf_counter()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    if is_trueenv:
        data = extract_or_load_train_tokens_trueenv(cache_path, max_workers=max_workers, limit=limit)
    else:
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
    utt_recs = data["utt_prosody_records"]

    t_cl = time.perf_counter()
    print(f"Building Stage 7 medoids & Ridge models (is_trueenv={is_trueenv}, C_DIM={c_dim})...", flush=True)

    payload: dict[str, np.ndarray] = {}
    t_on_of, n_on_of = _build_medoid_table_s5(
        "on_of", on_of_idx, on_cep, on_bap, on_vp, on_prev, on_next, on_pos, on_dur, on_f0,
        N_ONSET_BINS_S2, k_max=8,
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
        N_FINAL_BINS_S2, k_max=10, compute_gv=True,
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

    # 1. Articulatory neighbor prototypes strictly from TRAIN
    f_end_acc: dict[str, list[np.ndarray]] = defaultdict(list)
    f_start_acc: dict[str, list[np.ndarray]] = defaultdict(list)
    o_start_acc: dict[str, list[np.ndarray]] = defaultdict(list)
    gi = 0
    oi = 0
    for uid, rec in utt_recs.items():
        for s in rec["syls"]:
            o, f = s[2], s[3]
            f_end_acc[f].append(fi_cep[gi, -1, 1:9].astype(np.float64))
            f_start_acc[f].append(fi_cep[gi, 0, 1:9].astype(np.float64))
            if o and s[6] > 0 and oi < len(on_cep):
                o_start_acc[o].append(on_cep[oi, 0, 1:9].astype(np.float64))
                oi += 1
            gi += 1

    assert gi == len(fi_cep), f"Final token count mismatch: {gi} vs {len(fi_cep)}"
    assert oi == len(on_cep), f"Onset token count mismatch: {oi} vs {len(on_cep)}"

    f_end_proto = {k: np.mean(v, axis=0) for k, v in f_end_acc.items()}
    f_start_proto = {k: np.mean(v, axis=0) for k, v in f_start_acc.items()}
    o_start_proto = {k: np.mean(v, axis=0) for k, v in o_start_acc.items()}
    payload.update(_pack_protos("f_end_proto", f_end_proto))
    payload.update(_pack_protos("f_start_proto", f_start_proto))
    payload.update(_pack_protos("o_start_proto", o_start_proto))

    # 2. Build aligned 84-dim feature matrices X_fin and X_on
    X_fin = np.empty((gi, D_ACOUSTIC_FEAT_S7), dtype=np.float32)
    X_on = np.empty((oi, D_ACOUSTIC_FEAT_S7), dtype=np.float32)
    by_oft: dict[str, list[int]] = defaultdict(list)
    by_pft: dict[str, list[int]] = defaultdict(list)
    by_ft: dict[str, list[int]] = defaultdict(list)
    on_by_of: dict[str, list[int]] = defaultdict(list)
    on_by_og: dict[str, list[int]] = defaultdict(list)
    on_by_o: dict[str, list[int]] = defaultdict(list)

    gi = 0
    oi = 0
    for uid, rec in utt_recs.items():
        syls = rec["syls"]
        ns = len(syls)
        for si, s in enumerate(syls):
            _, _, o, f, t, dur_ms, on_ms, _, _, f3_st, _, pos_id, brk, pause_ms = s
            is_p = (si == 0 or pos_id == 0)
            is_n = (si == ns - 1 or pos_id == 2)
            row = extract_stage7_acoustic_features(
                prev_final="SIL" if is_p else syls[si - 1][3],
                prev_tone=0 if is_p else int(syls[si - 1][4]),
                next_onset="SIL" if is_n else syls[si + 1][2],
                next_final="SIL" if is_n else syls[si + 1][3],
                next_tone=0 if is_n else int(syls[si + 1][4]),
                pos_id=pos_id,
                dur_ms=dur_ms,
                on_ms=on_ms,
                f3_st=f3_st,
                brk=brk,
                pause_prev_ms=999.0 if is_p else float(syls[si - 1][13]),
                pause_after_ms=999.0 if is_n else pause_ms,
                f_end_proto=f_end_proto,
                o_start_proto=o_start_proto,
                f_start_proto=f_start_proto,
            )
            X_fin[gi] = row
            by_oft[f"{o}|{f}|{t}"].append(gi)
            by_pft[f"{classify_onset_place(o)}|{f}|{t}"].append(gi)
            by_ft[f"{f}|{t}"].append(gi)
            if o and on_ms > 0 and oi < len(on_cep):
                X_on[oi] = row
                on_by_of[f"{o}|{f}"].append(oi)
                on_by_og[f"{o}|{classify_final_group(f)}"].append(oi)
                on_by_o[o].append(oi)
                oi += 1
            gi += 1

    # 3. Fit Hierarchical Bayesian Ridge for Finals (ft -> pft -> oft + rare residual d_oft)
    Y_fin = fi_cep[:, :, :c_dim].reshape(gi, N_FINAL_BINS_S2 * c_dim)
    w_ft = {k: _fit_ridge(X_fin, Y_fin, idxs, lam=32.0) for k, idxs in by_ft.items() if len(idxs) >= 1}

    w_pft: dict[str, np.ndarray] = {}
    for k, idxs in by_pft.items():
        if len(idxs) < min_tokens_pft_w:
            continue
        _, f_p, t_p = k.split("|", 2)
        w_pft[k] = _fit_ridge(X_fin, Y_fin, idxs, lam=24.0, prior_W=w_ft.get(f"{f_p}|{t_p}"))

    w_oft: dict[str, np.ndarray] = {}
    d_oft: dict[str, np.ndarray] = {}
    for k, idxs in by_oft.items():
        o_p, f_p, t_p = k.split("|", 2)
        pk = f"{classify_onset_place(o_p)}|{f_p}|{t_p}"
        parent = w_pft.get(pk, w_ft.get(f"{f_p}|{t_p}"))
        if len(idxs) >= min_tokens_oft_w:
            w_oft[k] = _fit_ridge(X_fin, Y_fin, idxs, lam=18.0, prior_W=parent)
        elif parent is not None and len(idxs) >= 1:
            arr = np.asarray(idxs, dtype=np.int64)
            res = Y_fin[arr].astype(np.float64) - (X_fin[arr].astype(np.float64) @ parent)
            w_sh = float(len(idxs)) / (float(len(idxs)) + 6.0)
            d_oft[k] = (w_sh * np.mean(res, axis=0)).reshape(N_FINAL_BINS_S2, c_dim)

    # 4. Fit Hierarchical Bayesian Ridge for Onsets (o -> o|fg -> o|f + rare residual d_on_of)
    Y_on = on_cep[:, :, :c_dim].reshape(oi, N_ONSET_BINS_S2 * c_dim)
    w_on_o = {k: _fit_ridge(X_on, Y_on, idxs, lam=32.0) for k, idxs in on_by_o.items() if len(idxs) >= 1}

    w_on_og: dict[str, np.ndarray] = {}
    for k, idxs in on_by_og.items():
        if len(idxs) < min_tokens_pft_w:
            continue
        o_p = k.split("|", 1)[0]
        w_on_og[k] = _fit_ridge(X_on, Y_on, idxs, lam=22.0, prior_W=w_on_o.get(o_p))

    w_on_of: dict[str, np.ndarray] = {}
    d_on_of: dict[str, np.ndarray] = {}
    for k, idxs in on_by_of.items():
        o_p, f_p = k.split("|", 1)
        gk = f"{o_p}|{classify_final_group(f_p)}"
        parent = w_on_og.get(gk, w_on_o.get(o_p))
        if len(idxs) >= min_tokens_on_of_w:
            w_on_of[k] = _fit_ridge(X_on, Y_on, idxs, lam=16.0, prior_W=parent)
        elif parent is not None and len(idxs) >= 1:
            arr = np.asarray(idxs, dtype=np.int64)
            res = Y_on[arr].astype(np.float64) - (X_on[arr].astype(np.float64) @ parent)
            w_sh = float(len(idxs)) / (float(len(idxs)) + 6.0)
            d_on_of[k] = (w_sh * np.mean(res, axis=0)).reshape(N_ONSET_BINS_S2, c_dim)

    # 5. Context-conditioned BAP Ridge per (f, t)
    Y_bap = fi_bap.reshape(gi, N_FINAL_BINS_S2 * N_BAP)
    w_bap_ft = {k: _fit_ridge(X_fin, Y_bap, idxs, lam=30.0) for k, idxs in by_ft.items() if len(idxs) >= 1}

    payload.update(_pack_weights("w_ft", w_ft))
    payload.update(_pack_weights("w_pft", w_pft))
    payload.update(_pack_weights("w_oft", w_oft))
    payload.update(_pack_weights("d_oft", d_oft))
    payload.update(_pack_weights("w_on_o", w_on_o))
    payload.update(_pack_weights("w_on_og", w_on_og))
    payload.update(_pack_weights("w_on_of", w_on_of))
    payload.update(_pack_weights("d_on_of", d_on_of))
    payload.update(_pack_weights("w_bap_ft", w_bap_ft))

    # Metadata & onset timing statistics
    on_g_c32 = on_cep[:, :, 1:40].astype(np.float32)
    on_g_idx = int(np.argmin(np.sum((on_g_c32 - np.mean(on_g_c32, axis=0)[None, :, :]) ** 2, axis=(1, 2))))
    payload["on_glob_cep"] = on_cep[on_g_idx].astype(np.float16)
    payload["on_glob_bap"] = on_bap[on_g_idx].astype(np.float16)
    payload["on_glob_vp"] = on_vp[on_g_idx].astype(np.float16)

    fi_g_c32 = fi_cep[:, :, 1:40].astype(np.float32)
    fi_g_idx = int(np.argmin(np.sum((fi_g_c32 - np.mean(fi_g_c32, axis=0)[None, :, :]) ** 2, axis=(1, 2))))
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
    payload["c_dim"] = np.array([c_dim], dtype=np.int32)
    payload["is_trueenv"] = np.array([1 if is_trueenv else 0], dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    size_kb = out_path.stat().st_size / 1024.0

    kb7 = AcousticKnowledgeStage7.load(out_path)
    assert kb7.n_train_utts == n_utts

    summary = {
        "n_train_utts": n_utts,
        "n_dev_60": len(dev_60),
        "n_heldout_20": len(held_ids),
        "is_trueenv": bool(is_trueenv),
        "c_dim": int(c_dim),
        "build_time_s": round(time.perf_counter() - t0, 2),
        "fit_time_s": round(time.perf_counter() - t_cl, 2),
        "npz_size_kb": round(size_kb, 1),
        "npz_size_mb": round(size_kb / 1024.0, 2),
        "speaker_f0_med_hz": round(float(payload["speaker_f0_med_hz"][0]), 2),
        "ref_log_amp": round(float(payload["ref_log_amp"][0]), 4),
        "ridge_counts": {
            "w_ft": len(w_ft),
            "w_pft": len(w_pft),
            "w_oft": len(w_oft),
            "d_oft": len(d_oft),
            "w_on_o": len(w_on_o),
            "w_on_og": len(w_on_og),
            "w_on_of": len(w_on_of),
            "d_on_of": len(d_on_of),
            "w_bap_ft": len(w_bap_ft),
        },
        "medoid_tables": {
            "on_of": n_on_of,
            "on_grp": n_on_grp,
            "on_only": n_on_only,
            "fin_oft": n_fin_oft,
            "fin_pft": n_fin_pft,
            "fin_ft": n_fin_ft,
            "fin_f": n_fin_f,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved Stage 7 knowledge {out_path} ({summary['npz_size_mb']} MB) in {summary['build_time_s']}s", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-trueenv", action="store_true", help="Extract TrueHarmEnv TRAIN cache & build TrueEnv NPZ")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--c-dim", type=int, default=N_COARSE_CEP_S7)
    args = ap.parse_args()

    if args.extract_trueenv:
        build_stage7_knowledge(
            out_path=STAGE7_TRUEENV_PATH,
            summary_path=STAGE7_TRUEENV_SUMMARY_PATH,
            cache_path=DEFAULT_S7_TRUEENV_CACHE,
            is_trueenv=True,
            c_dim=args.c_dim,
            max_workers=args.workers,
            limit=args.limit,
        )
    else:
        build_stage7_knowledge(
            out_path=STAGE7_KNOWLEDGE_PATH,
            summary_path=STAGE7_SUMMARY_PATH,
            cache_path=DEFAULT_S5_CACHE_PATH,
            is_trueenv=False,
            c_dim=args.c_dim,
            max_workers=args.workers,
            limit=args.limit,
        )
