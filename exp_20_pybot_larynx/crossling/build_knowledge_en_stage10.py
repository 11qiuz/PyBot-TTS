"""Build Stage 10 Unblurred Contextual English Medoid + Hierarchical Bayesian Ridge KB.

Reads strictly TRAIN tokens from `en_stage10_train_tokens.pkl` (zero leakage from
held-out utterances), extracts:
- K <= 4 unblurred 5-bin TrueHarmEnv 64-dim cepstral + 6-band BAP + voicing medoids
  conditioned on `(prev_cls, phone, next_cls, stress)` and backoff hierarchies.
- Hierarchical Bayesian Ridge matrices (`w_base -> w_phone -> w_ps -> w_pns`, plus
  rare residual `d_pns`, plus `w_bap_ps`) predicting `cep[0:24]`.
- Articulatory boundary prototypes (`p_start_proto`, `p_end_proto`) & per-key GV targets.
Saves compact (`<= 25 MB`, `allow_pickle=False`) `knowledge_en_stage10.npz` + summary JSON.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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

from exp_20_pybot_larynx import EXP_DIR, N_BAP, N_CEP  # noqa: E402
from exp_20_pybot_larynx.acoustics.build_stage5_knowledge import (  # noqa: E402
    _select_medoids_kmeans_s5,
)
from exp_20_pybot_larynx.acoustics.build_stage7_knowledge import (  # noqa: E402
    _fit_ridge,
    _pack_protos,
    _pack_weights,
)
from exp_20_pybot_larynx.crossling.en_articulation import (  # noqa: E402
    classify_articulatory_class,
    resolve_en_allophone,
)
from exp_20_pybot_larynx.crossling.knowledge_en_stage10 import (  # noqa: E402
    D_ACOUSTIC_FEAT_S10,
    EN_STAGE10_KNOWLEDGE_PATH,
    EN_STAGE10_SUMMARY_PATH,
    N_COARSE_CEP_S10,
    N_EN_BINS,
    AcousticKnowledgeEnStage10,
    clear_en_stage10_cache,
    extract_en_stage10_acoustic_features,
    normalize_stress,
)

DEFAULT_TRAIN_TOKENS_PATH = EXP_DIR / "crossling_data" / "en_stage10_train_tokens.pkl"


def _build_medoid_table_en10(
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
    *,
    k_max: int = 4,
    cap_tokens: int = 1600,
    compute_gv: bool = False,
) -> tuple[dict[str, np.ndarray], int]:
    """Cluster each key's tokens into K <= 4 unblurred 5-bin medoids (float16 CSR)."""
    keys = sorted(groups.keys())
    offsets = [0]
    tok_cnts: list[int] = []

    med_cep, med_bap, med_vp = [], [], []
    med_prev, med_next, med_pos = [], [], []
    med_dur, med_f0, med_csz = [], [], []
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

        sub_cep = cep_arr[idx_arr].astype(np.float64)
        if compute_gv:
            tok_std = np.std(sub_cep[:, :, :40], axis=1)
            gv_stds.append(np.median(tok_std, axis=0).astype(np.float16))

        sub_dur = np.log(np.maximum(10.0, ctx_dur_ms[idx_arr].astype(np.float64)))
        sub_dur = (sub_dur - np.mean(sub_dur)) / max(0.15, float(np.std(sub_dur)))
        sub_f0 = ctx_f0_st[idx_arr].astype(np.float64)
        sub_f0 = (sub_f0 - np.mean(sub_f0)) / max(0.50, float(np.std(sub_f0)))

        spec_feat = np.concatenate([0.35 * sub_cep[:, :, :1], sub_cep[:, :, 1:40]], axis=2).reshape(len(idx_arr), -1)
        feat = np.concatenate([spec_feat, 1.8 * sub_dur[:, None], 1.4 * sub_f0[:, None]], axis=1)
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

    n_med = len(med_cep)
    max_len = max((len(k) for k in keys), default=1)
    out = {
        f"{prefix}_keys": np.array(keys, dtype=f"<U{max_len}"),
        f"{prefix}_offsets": np.array(offsets, dtype=np.int32),
        f"{prefix}_tok_cnt": np.array(tok_cnts, dtype=np.int32),
        f"{prefix}_cep": np.stack(med_cep, axis=0).astype(np.float16) if n_med else np.zeros((0, N_EN_BINS, N_CEP), dtype=np.float16),
        f"{prefix}_bap": np.stack(med_bap, axis=0).astype(np.float16) if n_med else np.zeros((0, N_EN_BINS, N_BAP), dtype=np.float16),
        f"{prefix}_vp": np.stack(med_vp, axis=0).astype(np.float16) if n_med else np.zeros((0, N_EN_BINS), dtype=np.float16),
        f"{prefix}_ctx_prev": np.array(med_prev, dtype=np.int8),
        f"{prefix}_ctx_next": np.array(med_next, dtype=np.int8),
        f"{prefix}_ctx_pos": np.array(med_pos, dtype=np.int8),
        f"{prefix}_ctx_dur_ms": np.array(med_dur, dtype=np.float16),
        f"{prefix}_ctx_f0_st": np.array(med_f0, dtype=np.float16),
        f"{prefix}_cluster_sz": np.array(med_csz, dtype=np.int32),
    }
    if compute_gv:
        out["gv_ps_keys"] = np.array(keys, dtype=f"<U{max_len}")
        out["gv_ps_std"] = np.stack(gv_stds, axis=0).astype(np.float16) if gv_stds else np.ones((0, 40), dtype=np.float16)
    return out, len(keys)


def _group_tokens_by_utt(data: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], int]:
    """Group flat `data['tokens']` by `utt_id` ordered by `phone_idx`, verifying zero heldout leakage."""
    train_ids = set(data.get("train_ids") or [])
    held_ids = set(data.get("heldout_ids") or [])
    if train_ids and held_ids and (train_ids & held_ids):
        raise AssertionError(f"TRAIN/HELDOUT leakage detected: {train_ids & held_ids}")

    if "tokens" in data and isinstance(data["tokens"], list):
        by_utt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for tok in data["tokens"]:
            uid = str(tok.get("utt_id", ""))
            if uid in held_ids:
                raise AssertionError(f"Heldout token {uid} leaked into train tokens")
            by_utt[uid].append(tok)
        utt_lists = [sorted(v, key=lambda t: int(t.get("phone_idx", 0))) for v in by_utt.values()]
        n_utts = int(data.get("n_utts", len(utt_lists)))
        return utt_lists, n_utts

    if "utterances" in data:
        utt_lists = [list(u.get("tokens") or u.get("phones") or []) for u in data["utterances"]]
        return utt_lists, len(utt_lists)
    raise KeyError(f"Unsupported train tokens schema keys: {sorted(data.keys())}")


def _filter_strict_indices(
    groups: dict[str, list[int]],
    purity_arr: np.ndarray,
    *,
    min_strict: int = 3,
) -> dict[str, list[int]]:
    """Use strictly `purity_ok` tokens per key when >= `min_strict`, else retain all."""
    out: dict[str, list[int]] = {}
    for k, idxs in groups.items():
        strict = [i for i in idxs if bool(purity_arr[i])]
        out[k] = strict if len(strict) >= min_strict else idxs
    return out


def build_knowledge_en_stage10(
    tokens_path: Path | str = DEFAULT_TRAIN_TOKENS_PATH,
    out_path: Path | str = EN_STAGE10_KNOWLEDGE_PATH,
    summary_path: Path | str = EN_STAGE10_SUMMARY_PATH,
    *,
    c_dim: int = N_COARSE_CEP_S10,
    min_tokens_pns_w: int = 20,
    min_tokens_ps_w: int = 16,
    min_tokens_phone_w: int = 12,
    k_max: int = 6,
) -> dict[str, Any]:
    """Build Stage 10 EN acoustic `.npz` strictly from TRAIN aligned TrueHarmEnv tokens."""
    t0 = time.perf_counter()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    tok_p = Path(tokens_path)
    with open(tok_p, "rb") as f:
        data = pickle.load(f)

    utt_token_lists, n_utts = _group_tokens_by_utt(data)
    speaker_f0_med_hz = float(data.get("speaker_f0_med_hz", 174.41))
    ref_log_amp = float(data.get("ref_log_amp", -5.318))
    sil_cep64 = np.asarray(data.get("sil_cep64", np.zeros(N_CEP)), dtype=np.float32)
    sil_bap6 = np.asarray(data.get("sil_bap6", np.full(N_BAP, 0.95)), dtype=np.float32)

    cep_l, bap_l, vp_l = [], [], []
    prev_cls_l, next_cls_l, pos_l, dur_l, f0_l, purity_l = [], [], [], [], [], []
    pns_idx: dict[str, list[int]] = defaultdict(list)
    pn_idx: dict[str, list[int]] = defaultdict(list)
    ps_idx: dict[str, list[int]] = defaultdict(list)
    p_idx: dict[str, list[int]] = defaultdict(list)
    base_idx: dict[str, list[int]] = defaultdict(list)

    p_start_acc: dict[str, list[np.ndarray]] = defaultdict(list)
    p_end_acc: dict[str, list[np.ndarray]] = defaultdict(list)

    flat_tokens: list[dict[str, Any]] = []
    for toks in utt_token_lists:
        n_t = len(toks)
        for i, tok in enumerate(toks):
            raw_cep = tok.get("cep_5") if "cep_5" in tok else tok.get("cep")
            raw_bap = tok.get("bap_5") if "bap_5" in tok else tok.get("bap")
            raw_vp = tok.get("vp_5") if "vp_5" in tok else tok.get("vp")
            if raw_cep is None or raw_bap is None or raw_vp is None:
                continue
            cep = np.asarray(raw_cep, dtype=np.float16)
            bap = np.asarray(raw_bap, dtype=np.float16)
            vp = np.asarray(raw_vp, dtype=np.float16)
            if cep.shape != (N_EN_BINS, N_CEP) or not np.all(np.isfinite(cep)):
                continue

            base_ipa = str(tok.get("ipa", "ə")).strip()
            prev_tok = toks[i - 1] if i > 0 else None
            next_tok = toks[i + 1] if i + 1 < n_t else None

            pause_prev = float(prev_tok.get("pause_after_ms", 0.0)) if prev_tok is not None else 999.0
            pause_after = float(tok.get("pause_after_ms", 999.0 if i == n_t - 1 else 0.0))
            if i == n_t - 1:
                pause_after = max(pause_after, 999.0)

            same_w_prev = bool(prev_tok is not None and pause_prev < 20.0 and prev_tok.get("word_idx") == tok.get("word_idx"))
            same_w_next = bool(next_tok is not None and pause_after < 20.0 and next_tok.get("word_idx") == tok.get("word_idx"))
            next_ipa = tok.get("next_ipa") or (next_tok.get("ipa") if next_tok is not None else None)

            phone = str(tok.get("allophone") or tok.get("phone") or resolve_en_allophone(base_ipa, next_ipa, same_w_next))
            stress = normalize_stress(tok.get("stress"), base_ipa)

            prev_ph = "sil" if (prev_tok is None or pause_prev >= 25.0) else str(prev_tok.get("allophone") or prev_tok.get("ipa") or "sil")
            next_ph = "sil" if (next_tok is None or pause_after >= 25.0) else str(next_tok.get("allophone") or next_tok.get("ipa") or "sil")
            prev_c = 0 if prev_ph == "sil" else int(tok.get("prev_cls", classify_articulatory_class(prev_ph)))
            next_c = 0 if next_ph == "sil" else int(tok.get("next_cls", classify_articulatory_class(next_ph)))
            pos_id = int(tok.get("pos_id", 0 if prev_c == 0 else (2 if next_c == 0 else 1)))

            n_w_ph = int(tok.get("word_n_phones", 1))
            if n_w_ph <= 1:
                word_pos = 0
            elif tok.get("is_word_initial", False):
                word_pos = 1
            elif tok.get("is_word_final", False):
                word_pos = 3
            else:
                word_pos = 2

            dur_ms = float(tok.get("dur_ms", 65.0))
            f0_st = float(tok.get("f0_med_st", tok.get("f0_st", 0.0)))
            f0_5 = tok.get("f0_st_5")
            if f0_5 is not None and len(f0_5) >= 5:
                f3_st = (float(f0_5[0]), float(f0_5[2]), float(f0_5[4]))
            else:
                f3_st = (f0_st, f0_st, f0_st)

            is_pure = bool(tok.get("purity_ok", True))

            ti = len(cep_l)
            cep_l.append(cep)
            bap_l.append(bap)
            vp_l.append(vp)
            prev_cls_l.append(prev_c)
            next_cls_l.append(next_c)
            pos_l.append(pos_id)
            dur_l.append(dur_ms)
            f0_l.append(f0_st)
            purity_l.append(is_pure)

            pns_idx[f"{prev_c}|{phone}|{next_c}|{stress}"].append(ti)
            pn_idx[f"{phone}|{next_c}|{stress}"].append(ti)
            ps_idx[f"{phone}|{stress}"].append(ti)
            p_idx[phone].append(ti)
            base_idx[base_ipa].append(ti)

            if is_pure:
                p_start_acc[phone].append(cep[0, 1:9].astype(np.float64))
                p_end_acc[phone].append(cep[-1, 1:9].astype(np.float64))
                if base_ipa != phone:
                    p_start_acc[base_ipa].append(cep[0, 1:9].astype(np.float64))
                    p_end_acc[base_ipa].append(cep[-1, 1:9].astype(np.float64))

            flat_tokens.append({
                "base_ipa": base_ipa,
                "phone": phone,
                "stress": stress,
                "prev_cls": prev_c,
                "next_cls": next_c,
                "pos_id": pos_id,
                "word_pos": word_pos,
                "is_func_word": bool(tok.get("is_func", False)),
                "is_coda": phone.endswith("_coda"),
                "same_word_prev": same_w_prev,
                "same_word_next": same_w_next,
                "dur_ms": dur_ms,
                "f3_st": f3_st,
                "pause_prev_ms": pause_prev,
                "pause_after_ms": pause_after,
                "prev_phone": prev_ph,
                "next_phone": next_ph,
            })

    n_tok = len(cep_l)
    if n_tok == 0:
        raise ValueError(f"No valid phone tokens found in {tok_p}")

    cep_arr = np.stack(cep_l, axis=0).astype(np.float16)
    bap_arr = np.stack(bap_l, axis=0).astype(np.float16)
    vp_arr = np.stack(vp_l, axis=0).astype(np.float16)
    prev_arr = np.array(prev_cls_l, dtype=np.int8)
    next_arr = np.array(next_cls_l, dtype=np.int8)
    pos_arr = np.array(pos_l, dtype=np.int8)
    dur_arr = np.array(dur_l, dtype=np.float32)
    f0_arr = np.array(f0_l, dtype=np.float32)
    purity_arr = np.array(purity_l, dtype=bool)

    # Prefer strict purity_ok tokens for prototype & medoid & Ridge sharpness
    pns_clean = _filter_strict_indices(pns_idx, purity_arr, min_strict=3)
    pn_clean = _filter_strict_indices(pn_idx, purity_arr, min_strict=3)
    ps_clean = _filter_strict_indices(ps_idx, purity_arr, min_strict=3)
    p_clean = _filter_strict_indices(p_idx, purity_arr, min_strict=3)
    base_clean = _filter_strict_indices(base_idx, purity_arr, min_strict=3)

    # Fallback prototypes for any phone that only had non-strict tokens
    for k, idxs in p_idx.items():
        if k not in p_start_acc:
            p_start_acc[k] = [cep_arr[i, 0, 1:9].astype(np.float64) for i in idxs]
            p_end_acc[k] = [cep_arr[i, -1, 1:9].astype(np.float64) for i in idxs]
    p_start_proto = {k: np.mean(v, axis=0) for k, v in p_start_acc.items()}
    p_end_proto = {k: np.mean(v, axis=0) for k, v in p_end_acc.items()}

    payload: dict[str, np.ndarray] = {}
    t_pns, n_pns = _build_medoid_table_en10("ctx_pns", pns_clean, cep_arr, bap_arr, vp_arr, prev_arr, next_arr, pos_arr, dur_arr, f0_arr, k_max=k_max)
    t_pn, n_pn = _build_medoid_table_en10("ctx_pn", pn_clean, cep_arr, bap_arr, vp_arr, prev_arr, next_arr, pos_arr, dur_arr, f0_arr, k_max=k_max)
    t_ps, n_ps = _build_medoid_table_en10("ctx_ps", ps_clean, cep_arr, bap_arr, vp_arr, prev_arr, next_arr, pos_arr, dur_arr, f0_arr, k_max=k_max, compute_gv=True)
    t_p, n_p = _build_medoid_table_en10("ctx_p", p_clean, cep_arr, bap_arr, vp_arr, prev_arr, next_arr, pos_arr, dur_arr, f0_arr, k_max=k_max)
    t_base, n_base = _build_medoid_table_en10("ctx_base", base_clean, cep_arr, bap_arr, vp_arr, prev_arr, next_arr, pos_arr, dur_arr, f0_arr, k_max=k_max)

    for tbl in (t_pns, t_pn, t_ps, t_p, t_base):
        payload.update(tbl)
    payload["gv_glob"] = np.median(t_ps["gv_ps_std"].astype(np.float64), axis=0).astype(np.float16)
    payload.update(_pack_protos("p_start_proto", p_start_proto))
    payload.update(_pack_protos("p_end_proto", p_end_proto))

    # Build 61-dim feature matrix & fit Hierarchical Bayesian Ridge
    X = np.empty((n_tok, D_ACOUSTIC_FEAT_S10), dtype=np.float32)
    for ti, ft in enumerate(flat_tokens):
        X[ti] = extract_en_stage10_acoustic_features(
            prev_cls=ft["prev_cls"],
            next_cls=ft["next_cls"],
            stress=ft["stress"],
            pos_id=ft["pos_id"],
            word_pos=ft["word_pos"],
            is_func_word=ft["is_func_word"],
            is_coda=ft["is_coda"],
            same_word_prev=ft["same_word_prev"],
            same_word_next=ft["same_word_next"],
            dur_ms=ft["dur_ms"],
            f3_st=ft["f3_st"],
            pause_prev_ms=ft["pause_prev_ms"],
            pause_after_ms=ft["pause_after_ms"],
            prev_phone=ft["prev_phone"],
            next_phone=ft["next_phone"],
            p_end_proto=p_end_proto,
            p_start_proto=p_start_proto,
        )

    Y_cep = cep_arr[:, :, :c_dim].reshape(n_tok, N_EN_BINS * c_dim)
    Y_bap = bap_arr.reshape(n_tok, N_EN_BINS * N_BAP)

    w_base = {k: _fit_ridge(X, Y_cep, idxs, lam=28.0) for k, idxs in base_clean.items() if len(idxs) >= 1}

    w_phone: dict[str, np.ndarray] = {}
    for k, idxs in p_clean.items():
        if len(idxs) < min_tokens_phone_w:
            continue
        b_key = k.replace("_coda", "")
        w_phone[k] = _fit_ridge(X, Y_cep, idxs, lam=22.0, prior_W=w_base.get(b_key))

    w_ps: dict[str, np.ndarray] = {}
    for k, idxs in ps_clean.items():
        if len(idxs) < min_tokens_ps_w:
            continue
        ph_k = k.split("|", 1)[0]
        b_k = ph_k.replace("_coda", "")
        parent = w_phone.get(ph_k, w_base.get(b_k))
        w_ps[k] = _fit_ridge(X, Y_cep, idxs, lam=18.0, prior_W=parent)

    w_pns: dict[str, np.ndarray] = {}
    d_pns: dict[str, np.ndarray] = {}
    for k, idxs in pns_clean.items():
        _, ph_k, _, st_k = k.split("|", 3)
        b_k = ph_k.replace("_coda", "")
        parent = w_ps.get(f"{ph_k}|{st_k}", w_phone.get(ph_k, w_base.get(b_k)))
        if len(idxs) >= min_tokens_pns_w:
            w_pns[k] = _fit_ridge(X, Y_cep, idxs, lam=14.0, prior_W=parent)
        elif parent is not None and len(idxs) >= 1:
            arr = np.asarray(idxs, dtype=np.int64)
            res = Y_cep[arr].astype(np.float64) - (X[arr].astype(np.float64) @ parent)
            w_sh = float(len(idxs)) / (float(len(idxs)) + 6.0)
            d_pns[k] = (w_sh * np.mean(res, axis=0)).reshape(N_EN_BINS, c_dim)

    w_bap_ps = {k: _fit_ridge(X, Y_bap, idxs, lam=24.0) for k, idxs in ps_clean.items() if len(idxs) >= 4}

    payload.update(_pack_weights("w_base", w_base))
    payload.update(_pack_weights("w_phone", w_phone))
    payload.update(_pack_weights("w_ps", w_ps))
    payload.update(_pack_weights("w_pns", w_pns))
    payload.update(_pack_weights("d_pns", d_pns))
    payload.update(_pack_weights("w_bap_ps", w_bap_ps))

    g_c32 = cep_arr[:, :, 1:40].astype(np.float32)
    g_idx = int(np.argmin(np.sum((g_c32 - np.mean(g_c32, axis=0)[None, :, :]) ** 2, axis=(1, 2))))
    payload["en_glob_cep"] = cep_arr[g_idx].astype(np.float16)
    payload["en_glob_bap"] = bap_arr[g_idx].astype(np.float16)
    payload["en_glob_vp"] = vp_arr[g_idx].astype(np.float16)

    payload["sil_cep64"] = sil_cep64
    payload["sil_bap6"] = sil_bap6
    payload["speaker_f0_med_hz"] = np.array([speaker_f0_med_hz], dtype=np.float32)
    payload["ref_log_amp"] = np.array([ref_log_amp], dtype=np.float32)
    payload["n_train_utts"] = np.array([n_utts], dtype=np.int32)
    payload["c_dim"] = np.array([c_dim], dtype=np.int32)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_p, **payload)
    clear_en_stage10_cache()
    size_mb = out_p.stat().st_size / (1024.0 * 1024.0)

    kb_chk = AcousticKnowledgeEnStage10.load(out_p)
    assert kb_chk.n_train_utts == n_utts

    summary = {
        "n_train_utts": n_utts,
        "n_train_tokens": n_tok,
        "n_strict_tokens": int(np.sum(purity_arr)),
        "c_dim": int(c_dim),
        "k_max": int(k_max),
        "npz_size_mb": round(size_mb, 3),
        "build_time_s": round(time.perf_counter() - t0, 2),
        "speaker_f0_med_hz": round(speaker_f0_med_hz, 2),
        "ref_log_amp": round(ref_log_amp, 4),
        "medoid_counts": {
            "ctx_pns": n_pns,
            "ctx_pn": n_pn,
            "ctx_ps": n_ps,
            "ctx_p": n_p,
            "ctx_base": n_base,
        },
        "ridge_counts": {
            "w_base": len(w_base),
            "w_phone": len(w_phone),
            "w_ps": len(w_ps),
            "w_pns": len(w_pns),
            "d_pns": len(d_pns),
            "w_bap_ps": len(w_bap_ps),
        },
    }
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[Stage10 EN KB] Built {out_p} ({size_mb:.2f} MB, {n_tok} tokens) in {summary['build_time_s']}s", flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TRAIN_TOKENS_PATH)
    ap.add_argument("--out", type=Path, default=EN_STAGE10_KNOWLEDGE_PATH)
    ap.add_argument("--summary", type=Path, default=EN_STAGE10_SUMMARY_PATH)
    args = ap.parse_args()
    build_knowledge_en_stage10(args.tokens, args.out, args.summary)
