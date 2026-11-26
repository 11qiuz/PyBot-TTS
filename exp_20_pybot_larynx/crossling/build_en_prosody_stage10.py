"""Train Stage 10 Zero-ML English Ridge Duration, F0 & Relative Energy Model strictly from TRAIN."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np
from scipy.linalg import solve as spd_solve
from scipy.sparse import csr_matrix, diags

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import EXP_DIR  # noqa: E402
from exp_20_pybot_larynx.crossling.en_prosody_stage10 import (  # noqa: E402
    PROSODY_EN_STAGE10_PATH,
    PROSODY_EN_STAGE10_SUMMARY_PATH,
    ProsodyEnStage10Model,
    clear_en_prosody_stage10_cache,
    compute_phone_prosody_context,
    extract_duration_features,
    extract_prosody_features,
    predict_en_prosody_stage10,
)

TRAIN_PKL_PATH: Path = EXP_DIR / "crossling_data" / "en_stage10_train_tokens.pkl"
HELDOUT_PKL_PATH: Path = EXP_DIR / "crossling_data" / "en_stage10_heldout.pkl"


def _solve_ridge(XtX: np.ndarray, XtY: np.ndarray, lam: float) -> np.ndarray:
    d = XtX.shape[0]
    reg = np.eye(d, dtype=np.float64) * float(lam)
    reg[0, 0] = 0.0  # Do not shrink bias
    return np.linalg.solve(XtX + reg, XtY)


def _solve_irls_huber(
    X: np.ndarray,
    y: np.ndarray,
    lam_l2: float,
    *,
    lam_irls_scale: float = 0.03,
    delta: float = 5.0,
    n_iter: int = 10,
) -> np.ndarray:
    """Closed-form Ridge initialized & refined via sparse-accelerated Huber/L1 IRLS."""
    d = X.shape[1]
    X_sp = csr_matrix(X)
    Xt = X_sp.T.tocsr()
    diag_idx = np.arange(1, d)

    A0 = (Xt @ X_sp).toarray()
    A0[diag_idx, diag_idx] += float(lam_l2)
    A0[0, 0] += 1e-10
    w = spd_solve(A0, Xt.dot(y), assume_a="pos", check_finite=False)

    lam_i = float(lam_l2 * lam_irls_scale)
    for _ in range(n_iter):
        res = y - X_sp.dot(w)
        wt = 1.0 / np.maximum(float(delta), np.abs(res))
        Aw = (Xt @ (diags(wt) @ X_sp)).toarray()
        Aw[diag_idx, diag_idx] += lam_i
        Aw[0, 0] += 1e-10
        w = spd_solve(Aw, Xt.dot(wt * y), assume_a="pos", check_finite=False)
    return w


def build_en_prosody_stage10(
    train_pkl: Path = TRAIN_PKL_PATH,
    heldout_pkl: Path = HELDOUT_PKL_PATH,
    out_path: Path = PROSODY_EN_STAGE10_PATH,
    summary_path: Path = PROSODY_EN_STAGE10_SUMMARY_PATH,
    *,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Train Stage 10 EN Ridge prosody/duration/energy model strictly on TRAIN & evaluate on HELDOUT."""
    t0 = time.perf_counter()
    with open(train_pkl, "rb") as f:
        tr_data = pickle.load(f)

    train_ids = set(tr_data.get("train_ids") or [])
    heldout_ids_ref = set(tr_data.get("heldout_ids") or [])
    assert not (train_ids & heldout_ids_ref), "Train/heldout ID overlap detected!"

    utts = list(tr_data["utterances"])
    by_utt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tok in tr_data["tokens"]:
        by_utt[str(tok["utt_id"])].append(tok)

    # Build vocabs strictly from TRAIN
    allos_vocab = sorted({str(t["allophone"]) for t in tr_data["tokens"]})
    ipas_vocab = sorted({str(t["ipa"]) for t in tr_data["tokens"]})
    ctx_vocab = ["SIL"] + ipas_vocab
    allo_map = {k: i for i, k in enumerate(allos_vocab)}
    ipa_map = {k: i for i, k in enumerate(ipas_vocab)}
    ctx_map = {k: i for i, k in enumerate(ctx_vocab)}

    as_pairs = sorted({
        (str(t["allophone"]), 0 if t["stress"] == 0 else (1 if t["stress"] == 1 else (2 if t["stress"] == 2 else 3)))
        for t in tr_data["tokens"]
    })
    as_map = {k: i for i, k in enumerate(as_pairs)}

    fw_vocab = sorted({str(t["word"]).lower() for t in tr_data["tokens"] if bool(t["is_func"])})
    fw_map = {k: i for i, k in enumerate(fw_vocab)}

    prev_bg_cnt = Counter((str(t["prev_ipa"]), str(t["allophone"])) for t in tr_data["tokens"])
    next_bg_cnt = Counter((str(t["allophone"]), str(t["next_ipa"])) for t in tr_data["tokens"])
    tri_bg_cnt = Counter((str(t["prev_ipa"]), str(t["allophone"]), str(t["next_ipa"])) for t in tr_data["tokens"])

    prev_bg = sorted(k for k, v in prev_bg_cnt.items() if v >= 12)
    next_bg = sorted(k for k, v in next_bg_cnt.items() if v >= 12)
    tri_bg = sorted(k for k, v in tri_bg_cnt.items() if v >= 20)
    pbg_map = {k: i for i, k in enumerate(prev_bg)}
    nbg_map = {k: i for i, k in enumerate(next_bg)}
    tbg_map = {k: i for i, k in enumerate(tri_bg)}

    X_dur_list: list[np.ndarray] = []
    X_pros_list: list[np.ndarray] = []
    y_dur_list: list[np.ndarray] = []
    y_f0_list: list[np.ndarray] = []
    y_gain_list: list[np.ndarray] = []
    f0_mask_list: list[np.ndarray] = []
    f0_wt_list: list[np.ndarray] = []
    fold_list: list[np.ndarray] = []

    for ui, u in enumerate(utts):
        toks = by_utt.get(str(u["id"]), [])
        if not toks:
            continue
        ctxs = compute_phone_prosody_context(toks)
        X_dur_list.append(
            extract_duration_features(
                ctxs,
                allo_map=allo_map,
                ipa_map=ipa_map,
                ctx_map=ctx_map,
                as_map=as_map,
                fw_map=fw_map,
                pbg_map=pbg_map,
                nbg_map=nbg_map,
                tbg_map=tbg_map,
            )
        )
        X_pros_list.append(
            extract_prosody_features(
                ctxs,
                allo_map=allo_map,
                ipa_map=ipa_map,
                ctx_map=ctx_map,
                fw_map=fw_map,
            )
        )
        y_dur_list.append(np.array([float(t["dur_ms"]) for t in toks], dtype=np.float64))
        y_f0_list.append(np.array([t["f0_st_5"] for t in toks], dtype=np.float64))
        y_gain_list.append(np.array([float(t["gain_db"]) for t in toks], dtype=np.float64))
        f0_mask_list.append(np.array([bool(t["f0_has_voiced"]) for t in toks], dtype=bool))
        f0_wt_list.append(np.array([max(0.25, float(t["voiced_frac"])) for t in toks], dtype=np.float64))
        fold_list.append(np.full(len(toks), ui % n_folds, dtype=np.int32))

    X_dur = np.vstack(X_dur_list)
    X_pros = np.vstack(X_pros_list)
    y_dur = np.concatenate(y_dur_list)
    y_f0_5 = np.vstack(y_f0_list)
    y_gain = np.concatenate(y_gain_list)
    f0_mask = np.concatenate(f0_mask_list)
    f0_wt = np.concatenate(f0_wt_list)
    folds = np.concatenate(fold_list)

    print(
        f"[Stage10 Prosody] TRAIN tokens={len(y_dur)}, X_dur={X_dur.shape}, X_pros={X_pros.shape}",
        flush=True,
    )

    # 1. 5-fold CV inside TRAIN for Duration IRLS Ridge
    best_lam_dur = 3.0
    best_cv_dur_mae = 1e9
    y_dur_clip = np.clip(y_dur, 20.0, 310.0)
    for lam in (3.0, 8.0):
        maes = []
        for k in range(n_folds):
            trm = folds != k
            vam = folds == k
            wk = _solve_irls_huber(X_dur[trm], y_dur_clip[trm], lam, lam_irls_scale=0.03, delta=5.0, n_iter=7)
            pk = np.clip(X_dur[vam] @ wk, 20.0, 310.0)
            maes.append(float(np.mean(np.abs(pk - y_dur[vam]))))
        cv_mae = float(np.mean(maes))
        if cv_mae < best_cv_dur_mae:
            best_cv_dur_mae = cv_mae
            best_lam_dur = lam
    w_dur = _solve_irls_huber(X_dur, y_dur_clip, best_lam_dur, lam_irls_scale=0.03, delta=5.0, n_iter=10)

    # 2. 5-fold CV inside TRAIN for F0 5-bin Ridge
    best_lam_f0 = 25.0
    best_cv_f0_rmse = 1e9
    y_f0_clip = np.clip(y_f0_5, -10.0, 10.0)
    for lam in (10.0, 25.0, 50.0, 100.0):
        rmses = []
        for k in range(n_folds):
            trm = (folds != k) & f0_mask
            vam = (folds == k) & f0_mask
            Xw = X_pros[trm] * f0_wt[trm, None]
            wk = _solve_ridge(Xw.T @ X_pros[trm], Xw.T @ y_f0_clip[trm], lam)
            pk = np.clip(X_pros[vam] @ wk, -10.0, 10.0)
            rmses.append(float(np.sqrt(np.mean((pk - y_f0_5[vam]) ** 2))))
        cv_rmse = float(np.mean(rmses))
        if cv_rmse < best_cv_f0_rmse:
            best_cv_f0_rmse = cv_rmse
            best_lam_f0 = lam
    Xw_f0 = X_pros[f0_mask] * f0_wt[f0_mask, None]
    w_f0_5 = _solve_ridge(Xw_f0.T @ X_pros[f0_mask], Xw_f0.T @ y_f0_clip[f0_mask], best_lam_f0)

    # 3. 5-fold CV inside TRAIN for Relative Energy gain_db Ridge
    best_lam_gain = 25.0
    best_cv_gain_mae = 1e9
    y_gain_clip = np.clip(y_gain, -16.0, 9.0)
    for lam in (10.0, 25.0, 50.0, 100.0):
        g_maes = []
        for k in range(n_folds):
            trm = folds != k
            vam = folds == k
            wk = _solve_ridge(X_pros[trm].T @ X_pros[trm], X_pros[trm].T @ y_gain_clip[trm], lam)
            pk = X_pros[vam] @ wk
            g_maes.append(float(np.mean(np.abs(pk - y_gain[vam]))))
        cv_g = float(np.mean(g_maes))
        if cv_g < best_cv_gain_mae:
            best_cv_gain_mae = cv_g
            best_lam_gain = lam
    w_gain = _solve_ridge(X_pros.T @ X_pros, X_pros.T @ y_gain_clip, best_lam_gain)

    print(
        f"[Stage10 Prosody] 5-fold TRAIN CV: dur_MAE={best_cv_dur_mae:.2f}ms (lam={best_lam_dur}), "
        f"f0_RMSE={best_cv_f0_rmse:.3f}st (lam={best_lam_f0}), gain_MAE={best_cv_gain_mae:.2f}dB (lam={best_lam_gain})",
        flush=True,
    )

    params_dict = {
        "f0_ridge_w": 0.95,
        "f0_accent_w": 1.74,
        "f0_nuc_boost_st": 2.52,
        "f0_content_boost_st": 1.75,
        "f0_sec_boost_st": 0.45,
        "f0_func_trough_st": -1.55,
        "f0_decl_span_st": 1.55,
        "f0_q_rise_st": 2.90,
        "f0_s_fall_st": 1.60,
        "f0_qta_tau_ms": 3.0,
        "f0_cf0_vl_st": 1.25,
        "f0_ar1_sigma": 0.20,
        "f0_ar1_rho": 0.76,
        "gain_scale": 0.45,
        "gain_min_db": -5.5,
        "gain_max_db": 3.5,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        allos_vocab=np.asarray(allos_vocab, dtype=str),
        ipas_vocab=np.asarray(ipas_vocab, dtype=str),
        ctx_vocab=np.asarray(ctx_vocab, dtype=str),
        as_allos=np.asarray([a for a, _ in as_pairs], dtype=str),
        as_stress=np.asarray([s for _, s in as_pairs], dtype=np.int32),
        fw_vocab=np.asarray(fw_vocab, dtype=str),
        prev_bg=np.asarray(prev_bg, dtype=str),
        next_bg=np.asarray(next_bg, dtype=str),
        tri_bg=np.asarray(tri_bg, dtype=str),
        w_dur=w_dur.astype(np.float64),
        w_f0_5=w_f0_5.astype(np.float64),
        w_gain=w_gain.astype(np.float64),
        speaker_f0_med_hz=np.asarray([float(tr_data.get("speaker_f0_med_hz", 174.41))], dtype=np.float64),
        param_keys=np.asarray(list(params_dict.keys()), dtype=str),
        param_vals=np.asarray(list(params_dict.values()), dtype=np.float64),
    )
    clear_en_prosody_stage10_cache()
    size_kb = out_path.stat().st_size / 1024.0
    assert size_kb < 200.0, f"{out_path} exceeded 200 KB: {size_kb:.2f} KB"

    # Evaluate strictly on unseen HELDOUT (zero leakage)
    mdl = ProsodyEnStage10Model(out_path)
    heldout_metrics: dict[str, Any] = {}
    if heldout_pkl.exists():
        with open(heldout_pkl, "rb") as f:
            ho_data = pickle.load(f)
        ho_utts = list(ho_data["utterances"])
        ho_ids = {str(u["id"]) for u in ho_utts}
        assert not (ho_ids & train_ids), "Train/heldout overlap on evaluation!"

        dur_errs: list[float] = []
        f0_rmses: list[float] = []
        df0_p95s: list[float] = []
        iqrs: list[float] = []

        for idx_u, u in enumerate(ho_utts):
            ref_phones = list(u.get("phones") or [])
            ref_durs = [float(p["dur_ms"]) for p in ref_phones]
            pred = predict_en_prosody_stage10(
                str(u["text"]),
                base_f0_hz=float(tr_data.get("speaker_f0_med_hz", 174.41)),
                include_pauses=False,
                seed=2026 + idx_u,
                model=mdl,
            )
            p_durs = pred["phone_durs_ms"]
            n_m = min(len(ref_durs), len(p_durs))
            for k in range(n_m):
                dur_errs.append(abs(float(p_durs[k]) - float(ref_durs[k])))

            # Per-phone voiced F0 RMSE & frame smoothness/IQR diagnostics
            pred_f0_st = pred["f0_st"]
            sp_list = [s for s in pred["phone_spans"] if not s["is_pause"]]
            p_errs = []
            for k in range(min(len(ref_phones), len(sp_list))):
                rp = ref_phones[k]
                if not rp.get("f0_has_voiced"):
                    continue
                sa, sb = sp_list[k]["start_fr"], sp_list[k]["end_fr"]
                if sb <= sa:
                    continue
                p_med = float(np.median(pred_f0_st[sa:sb]))
                p_errs.append((p_med - float(rp["f0_med_st"])) ** 2)
            if p_errs:
                f0_rmses.append(math.sqrt(float(np.mean(p_errs))))

            # Contour stats on voiced-like phone spans
            v_fr = np.zeros(pred["n_frames"], dtype=bool)
            for k, sp_row in enumerate(sp_list):
                if k < len(ref_phones) and ref_phones[k].get("f0_has_voiced"):
                    v_fr[sp_row["start_fr"] : sp_row["end_fr"]] = True
            if int(np.sum(v_fr)) >= 8:
                iqrs.append(float(np.percentile(pred_f0_st[v_fr], 75) - np.percentile(pred_f0_st[v_fr], 25)))
                adj = v_fr[:-1] & v_fr[1:]
                if np.any(adj):
                    df0_p95s.append(float(np.percentile(np.abs(np.diff(pred_f0_st))[adj], 95)))

        heldout_metrics = {
            "n_heldout_utts": len(ho_utts),
            "n_heldout_phones": len(dur_errs),
            "heldout_dur_mae_ms": round(float(np.mean(dur_errs)), 2),
            "heldout_dur_median_ae_ms": round(float(np.median(dur_errs)), 2),
            "heldout_f0_rmse_st": round(float(np.median(f0_rmses)), 3),
            "heldout_f0_iqr_st": round(float(np.median(iqrs)), 3),
            "heldout_df0_p95_st": round(float(np.median(df0_p95s)), 4),
        }

    # Also test forensic sentences in BZNSYP anchor space (247.86 Hz)
    forensic_sents = [
        "Sixty seven friends test our cross lingual speech synthesis clearly.",
        "This is a smooth benchmark of bilingual speech synthesis in our studio.",
        "Turn it on and take an apple in the box to one of us today.",
        "Think about this very smooth zoo vision in the morning.",
    ]
    for_iqrs: list[float] = []
    for_df0s: list[float] = []
    for_gains: list[float] = []
    for idx_s, txt in enumerate(forensic_sents):
        pr = predict_en_prosody_stage10(txt, base_f0_hz=247.86, seed=2026 + idx_s, model=mdl)
        st = pr["f0_st"]
        mask = pr["f0_hz"] > 40.0
        for_iqrs.append(float(np.percentile(st[mask], 75) - np.percentile(st[mask], 25)))
        adj = mask[:-1] & mask[1:]
        for_df0s.append(float(np.percentile(np.abs(np.diff(st))[adj], 95)))
        for_gains.append(float(np.percentile(pr["gain_db_frames"][mask], 90) - np.percentile(pr["gain_db_frames"][mask], 10)))

    summary = {
        "schema": "prosody_en_stage10",
        "artifact_path": str(out_path.relative_to(REPO_ROOT)),
        "artifact_size_kb": round(size_kb, 2),
        "allow_pickle": False,
        "train_utts": len(utts),
        "train_tokens": len(y_dur),
        "dur_dim": int(X_dur.shape[1]),
        "pros_dim": int(X_pros.shape[1]),
        "cv_train_dur_mae_ms": round(best_cv_dur_mae, 2),
        "cv_train_f0_rmse_st": round(best_cv_f0_rmse, 3),
        "cv_train_gain_mae_db": round(best_cv_gain_mae, 2),
        "best_lam_dur": best_lam_dur,
        "best_lam_f0": best_lam_f0,
        "best_lam_gain": best_lam_gain,
        "heldout": heldout_metrics,
        "forensic_contour": {
            "median_f0_iqr_st": round(float(np.median(for_iqrs)), 3),
            "median_df0_p95_st": round(float(np.median(for_df0s)), 4),
            "median_gain_p90_10_db": round(float(np.median(for_gains)), 2),
        },
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    build_en_prosody_stage10()
