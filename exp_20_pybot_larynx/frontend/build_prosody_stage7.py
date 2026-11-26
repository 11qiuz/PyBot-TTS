"""Train Stage 7 Two-Pass Continuous Ridge + Xu-Coarticulation prosody model strictly from TRAIN."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
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

from exp_20_pybot_larynx import N_CEP  # noqa: E402
from exp_20_pybot_larynx.acoustics.build_stage5_knowledge import DEFAULT_S5_CACHE_PATH  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge import get_splits, normalize_of  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import STAGE5_KNOWLEDGE_PATH, AcousticKnowledgeStage5  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import STAGE7_KNOWLEDGE_PATH, AcousticKnowledgeStage7  # noqa: E402
from exp_20_pybot_larynx.frontend.prosody_rules import predict_prosody_rules  # noqa: E402
from exp_20_pybot_larynx.frontend.prosody_stage5 import predict_prosody_stage5  # noqa: E402
from exp_20_pybot_larynx.frontend.prosody_stage7 import (  # noqa: E402
    DEFAULT_FINALS_VOCAB,
    DEFAULT_ONSETS_VOCAB,
    PROSODY_S7_SUMMARY_PATH,
    PROSODY_S7_TABLES_PATH,
    ProsodyStage7Model,
    clear_stage7_prosody_cache,
    compute_unit_context_s7,
    extract_prosody_stage7_pass1_features,
    extract_prosody_stage7_pass2_features,
    predict_prosody_stage7,
)
from exp_20_pybot_larynx.larynx.synth import log_env_to_cep  # noqa: E402

DEV60_CACHE_PATH = Path(
    r"C:\Users\MECHREVO\.gemini\antigravity\brain\f67e0437-e98d-4cb1-bbfd-da72d02478db\scratch\dev60_oracles.pkl"
)


def _solve_ridge_stats(XtX: np.ndarray, XtY: np.ndarray, lam: float) -> np.ndarray:
    d = XtX.shape[0]
    reg = np.eye(d, dtype=np.float64) * float(lam)
    reg[0, 0] = 0.0  # Do not shrink bias
    return np.linalg.solve(XtX + reg, XtY)


def _collect_dev60_ground_truth(
    kb7: AcousticKnowledgeStage7,
    kb5: AcousticKnowledgeStage5,
) -> list[dict[str, Any]]:
    """Load cached DEV_60 oracles and extract rule/Stage-5/ground-truth prosody targets."""
    if not DEV60_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing DEV_60 oracle cache: {DEV60_CACHE_PATH}")
    with open(DEV60_CACHE_PATH, "rb") as f:
        dev_oracles = pickle.load(f)

    records: list[dict[str, Any]] = []
    for item in dev_oracles:
        uid = item["uid"]
        text = item["text"]
        fp = item["feats_p"]
        ctx = item["ctx"]
        syls = ctx.syllables

        pro_r = predict_prosody_rules(text, kb7, style="auto")
        pro_s5 = predict_prosody_stage5(text, kb5, style="auto")
        if len(pro_r["units"]) != len(syls) or len(pro_s5["units"]) != len(syls):
            continue

        v_m = fp.voiced & (fp.f0_hz > 50.0)
        ref_f0 = max(50.0, float(ctx.speaker_f0_med_hz))
        utt_ref_amp = float(np.median(np.mean(fp.log_env[v_m, 1:257], axis=1))) if np.any(v_m) else float(kb7.ref_log_amp)
        cep64 = log_env_to_cep(fp.log_env - utt_ref_amp, n_cep=N_CEP)

        real_dur = []
        real_on = []
        real_f3 = []
        real_c0 = []
        has_on = []
        for sp_s in syls:
            a, sp, b = int(sp_s.start_fr), int(sp_s.split_fr), int(sp_s.end_fr)
            real_dur.append(float(max(1, b - a)) * 5.0)
            o_norm, _ = normalize_of(sp_s.onset, sp_s.final)
            if o_norm and sp > a:
                real_on.append(float(sp - a) * 5.0)
                has_on.append(True)
                fa = sp
            else:
                real_on.append(0.0)
                has_on.append(False)
                fa = a
            vf = np.where(v_m[fa:b])[0] if b > fa else np.zeros(0, dtype=np.intp)
            if len(vf) >= 2:
                uv = vf.astype(np.float64) / float(max(1, (b - fa) - 1))
                stv = 12.0 * np.log2(fp.f0_hz[fa:b][vf] / ref_f0)
                f3 = np.interp([0.20, 0.50, 0.80], uv, stv)
            elif len(vf) == 1:
                v = float(12.0 * np.log2(fp.f0_hz[fa:b][vf[0]] / ref_f0))
                f3 = np.array([v, v, v], dtype=np.float64)
            else:
                vs = v_m[a:b]
                v = float(12.0 * np.log2(np.median(fp.f0_hz[a:b][vs]) / ref_f0)) if np.any(vs) else 0.0
                f3 = np.array([v, v, v], dtype=np.float64)
            real_f3.append(f3)
            real_c0.append(float(np.mean(cep64[fa:b, 0])) if b > fa else 0.0)

        c0_med = float(np.median(real_c0)) if real_c0 else 0.0
        real_rel_db = np.array([(c0 - c0_med) * 0.383 for c0 in real_c0], dtype=np.float64)

        records.append({
            "uid": uid,
            "text": text,
            "units_rule": pro_r["units"],
            "units_s5": pro_s5["units"],
            "ctxs": compute_unit_context_s7(pro_r["units"]),
            "real_dur": np.array(real_dur, dtype=np.float64),
            "real_on": np.array(real_on, dtype=np.float64),
            "has_on": np.array(has_on, dtype=bool),
            "real_f3": np.stack(real_f3, axis=0),
            "real_rel_db": real_rel_db,
        })
    return records


def build_prosody_stage7_tables(
    cache_path: Path = DEFAULT_S5_CACHE_PATH,
    knowledge_path: Path = STAGE7_KNOWLEDGE_PATH,
    out_path: Path = PROSODY_S7_TABLES_PATH,
    summary_path: Path = PROSODY_S7_SUMMARY_PATH,
    *,
    n_tritones: int = 125,
    n_folds: int = 5,
    limit: int | None = None,
) -> dict[str, Any]:
    """Train Two-Pass Xu Ridge + empirical-Bayes prosody model strictly on TRAIN & tune on DEV_60."""
    t0 = time.perf_counter()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    kb7 = AcousticKnowledgeStage7.load(knowledge_path)
    kb5 = AcousticKnowledgeStage5.load(STAGE5_KNOWLEDGE_PATH)

    train_ids, dev_60, held_20 = get_splits()
    train_set = set(train_ids)
    dev_set = set(dev_60)
    held_set = set(held_20)
    assert not (train_set & dev_set) and not (train_set & held_set) and not (dev_set & held_set)

    utt_recs = data["utt_prosody_records"]
    uids = sorted(u for u in utt_recs.keys() if u in train_set)
    if limit is not None:
        uids = uids[:limit]

    # Pass A: align TRAIN utterances & count tri-tone frequencies / vocabs
    aligned: list[tuple[int, str, list[dict[str, Any]], list[dict[str, Any]], list[tuple[Any, ...]]]] = []
    tritone_cnt: Counter[tuple[int, int, int]] = Counter()
    onset_set: set[str] = set(DEFAULT_ONSETS_VOCAB)
    final_set: set[str] = set(DEFAULT_FINALS_VOCAB)
    dur_acc: dict[tuple[int, int, int, int, int], list[float]] = defaultdict(list)
    uv_ofg_acc: dict[str, list[float]] = defaultdict(list)
    uv_o_acc: dict[str, list[float]] = defaultdict(list)

    for idx_u, uid in enumerate(uids, 1):
        rec = utt_recs[uid]
        syls = rec["syls"]
        try:
            pro = predict_prosody_rules(rec["text"], kb7, style="auto")
        except Exception:
            continue
        units = pro["units"]
        if len(units) != len(syls):
            continue
        ctxs = compute_unit_context_s7(units)
        fold_id = len(aligned) % n_folds
        aligned.append((fold_id, uid, units, ctxs, syls))

        for u, c, sr_row in zip(units, ctxs, syls):
            tritone_cnt[(c["pt"], c["t"], c["nt"])] += 1
            onset_set.add(c["o"])
            if c["f"]:
                final_set.add(c["f"])
            real_dur = float(sr_row[5])
            d_log = math.log(max(30.0, real_dur) / max(60.0, float(u["duration_ms"])))
            dur_acc[(c["om_id"], c["fg_id"], c["t"], c["wpos_id"], c["brk_ge2"])].append(d_log)
            uv_f = float(sr_row[7])
            if c["o"] and uv_f >= 0.0:
                uv_ofg_acc[f"{c['o']}|{c['fg']}"].append(uv_f)
                uv_o_acc[c["o"]].append(uv_f)

        if idx_u % 3000 == 0 or idx_u == len(uids):
            print(f"  aligned TRAIN {idx_u}/{len(uids)} (matched={len(aligned)}, {time.perf_counter() - t0:.1f}s)", flush=True)

    tritone_vocab = [k for k, _ in tritone_cnt.most_common(n_tritones)]
    tritone_map = {k: i for i, k in enumerate(tritone_vocab)}
    onsets_vocab = sorted(onset_set)
    onset_map = {k: i for i, k in enumerate(onsets_vocab)}
    finals_vocab = sorted(final_set)
    final_map = {k: i for i, k in enumerate(finals_vocab)}

    # Pass B: build dense Pass-1 TRAIN design matrix & targets
    n_tot = sum(len(x[2]) for x in aligned)
    d1 = extract_prosody_stage7_pass1_features(
        aligned[0][2],
        ctxs=aligned[0][3],
        tritone_map=tritone_map,
        n_tritones=len(tritone_vocab),
        onset_map=onset_map,
        n_onsets=len(onsets_vocab),
        final_map=final_map,
        n_finals=len(finals_vocab),
    ).shape[1]

    X1_all = np.empty((n_tot, d1), dtype=np.float64)
    Y_f3_res = np.empty((n_tot, 3), dtype=np.float64)
    Y_dur = np.empty(n_tot, dtype=np.float64)
    Y_ondur = np.empty(n_tot, dtype=np.float64)
    M_ondur = np.zeros(n_tot, dtype=bool)
    Y_gain = np.empty(n_tot, dtype=np.float64)
    Fold_ids = np.empty(n_tot, dtype=np.int32)
    Base_f3_all = np.empty((n_tot, 3), dtype=np.float64)
    Utt_slices: list[tuple[int, int, int, list[dict[str, Any]], list[dict[str, Any]]]] = []

    cursor = 0
    for fold_id, _uid, units, ctxs, syls in aligned:
        nu = len(units)
        X_u = extract_prosody_stage7_pass1_features(
            units,
            ctxs=ctxs,
            tritone_map=tritone_map,
            n_tritones=len(tritone_vocab),
            onset_map=onset_map,
            n_onsets=len(onsets_vocab),
            final_map=final_map,
            n_finals=len(finals_vocab),
        )
        X1_all[cursor : cursor + nu] = X_u
        Fold_ids[cursor : cursor + nu] = fold_id

        c0s = [float(row[10]) for row in syls]
        c0_med = float(np.median(c0s)) if c0s else 0.0
        for j, (u, row) in enumerate(zip(units, syls)):
            gi = cursor + j
            p3 = np.asarray(u["f0_3pt_st"], dtype=np.float64)
            r3 = np.asarray(row[9], dtype=np.float64)
            Base_f3_all[gi] = p3
            Y_f3_res[gi] = np.clip(r3 - p3, -9.0, 9.0)

            real_dur = max(30.0, float(row[5]))
            pred_dur = max(60.0, float(u["duration_ms"]))
            Y_dur[gi] = float(np.clip(math.log(real_dur / pred_dur), -0.85, 0.85))

            real_on = float(row[6])
            pred_on = float(u.get("onset_ms", 0.0))
            if ctxs[j]["o"] and real_on >= 5.0 and pred_on >= 5.0:
                Y_ondur[gi] = float(np.clip(math.log(real_on / pred_on), -0.85, 0.85))
                M_ondur[gi] = True
            else:
                Y_ondur[gi] = 0.0

            real_rel_db = (float(row[10]) - c0_med) * 0.383
            Y_gain[gi] = float(np.clip(real_rel_db - float(u.get("gain_db", 0.0)), -5.5, 5.5))

        Utt_slices.append((fold_id, cursor, cursor + nu, units, ctxs))
        cursor += nu

    assert cursor == n_tot
    print(f"TRAIN matrix built: X1={X1_all.shape}, M_ondur={int(np.sum(M_ondur))} ({time.perf_counter() - t0:.1f}s)", flush=True)

    # Standardize non-bias columns of X1
    x1_mean = np.mean(X1_all, axis=0)
    x1_std = np.std(X1_all, axis=0)
    x1_mean[0] = 0.0
    x1_std[0] = 1.0
    x1_std = np.where(x1_std < 1e-4, 1.0, x1_std)
    Z1_all = (X1_all - x1_mean[None, :]) / x1_std[None, :]
    Z1_all[:, 0] = 1.0

    # Precompute sufficient statistics for fast multi-lambda Ridge fits
    XtX_1 = Z1_all.T @ Z1_all
    XtY_f3 = Z1_all.T @ Y_f3_res
    XtY_dur = Z1_all.T @ Y_dur
    XtY_gain = Z1_all.T @ Y_gain
    Z1_on = Z1_all[M_ondur]
    XtX_on = Z1_on.T @ Z1_on
    XtY_on = Z1_on.T @ Y_ondur[M_ondur]

    # Fold stats for 5-fold OOF Pass-1 predictions (strictly leak-free Pass-2 input!)
    fold_XtX = []
    fold_XtY = []
    for k in range(n_folds):
        mk = Fold_ids == k
        Zk = Z1_all[mk]
        fold_XtX.append(Zk.T @ Zk)
        fold_XtY.append(Zk.T @ Y_f3_res[mk])

    # Candidate Ridge solutions across regularization grid
    w_f0_p1_grid: dict[float, np.ndarray] = {}
    oof_p1_by_lam: dict[float, np.ndarray] = {}
    for lam in (100.0, 300.0, 1000.0, 3000.0):
        w_f0_p1_grid[lam] = _solve_ridge_stats(XtX_1, XtY_f3, lam)
        oof = np.empty_like(Y_f3_res)
        for k in range(n_folds):
            mk = Fold_ids == k
            wk = _solve_ridge_stats(XtX_1 - fold_XtX[k], XtY_f3 - fold_XtY[k], lam)
            oof[mk] = Z1_all[mk] @ wk
        oof_p1_by_lam[lam] = oof

    w_dur_grid = {lam: _solve_ridge_stats(XtX_1, XtY_dur, lam) for lam in (300.0, 1000.0, 3000.0, 10000.0)}
    w_on_grid = {lam: _solve_ridge_stats(XtX_on, XtY_on, lam) for lam in (300.0, 1000.0, 3000.0)}
    w_gain_grid = {lam: _solve_ridge_stats(XtX_1, XtY_gain, lam) for lam in (300.0, 1000.0, 3000.0)}

    # Empirical Bayes categorical dur_ctx & onset_uv_ofg
    dur_k = sorted(dur_acc.keys())
    dur_v = []
    for k in dur_k:
        arr = dur_acc[k]
        n = len(arr)
        ws = n / (n + 24.0)
        dur_v.append(float(np.clip(ws * float(np.mean(arr)), -0.22, 0.22)))
    dur_ctx_map = {k: v for k, v in zip(dur_k, dur_v)}

    uv_o_med = {o: float(np.clip(np.median(vals), 0.25, 0.96)) for o, vals in uv_o_acc.items()}
    uv_ofg_map = {}
    for k, vals in uv_ofg_acc.items():
        o_p = k.split("|", 1)[0]
        prior = uv_o_med.get(o_p, 0.78)
        n = len(vals)
        ws = n / (n + 10.0)
        uv_ofg_map[k] = float(np.clip(ws * float(np.median(vals)) + (1.0 - ws) * prior, 0.25, 0.96))

    # Prepare DEV_60 matrices for ultra-fast hyperparameter tuning (strictly on DEV_60!)
    dev_recs = _collect_dev60_ground_truth(kb7, kb5)
    dev_items: list[dict[str, Any]] = []
    for r in dev_recs:
        units_r = r["units_rule"]
        ctxs_r = r["ctxs"]
        X1_d = extract_prosody_stage7_pass1_features(
            units_r,
            ctxs=ctxs_r,
            tritone_map=tritone_map,
            n_tritones=len(tritone_vocab),
            onset_map=onset_map,
            n_onsets=len(onsets_vocab),
            final_map=final_map,
            n_finals=len(finals_vocab),
        )
        Z1_d = (X1_d - x1_mean[None, :]) / x1_std[None, :]
        Z1_d[:, 0] = 1.0
        base_f3_d = np.array([u["f0_3pt_st"] for u in units_r], dtype=np.float64)
        base_dur_d = np.array([float(u["duration_ms"]) for u in units_r], dtype=np.float64)
        base_on_d = np.array([float(u.get("onset_ms", 0.0)) for u in units_r], dtype=np.float64)
        base_gain_d = np.array([float(u.get("gain_db", 0.0)) for u in units_r], dtype=np.float64)
        d_ctx_d = np.array(
            [dur_ctx_map.get((c["om_id"], c["fg_id"], c["t"], c["wpos_id"], c["brk_ge2"]), 0.0) for c in ctxs_r],
            dtype=np.float64,
        )

        s5_f3 = np.array([u["f0_3pt_st"] for u in r["units_s5"]], dtype=np.float64)
        s5_dur = np.array([float(u["duration_ms"]) for u in r["units_s5"]], dtype=np.float64)
        s5_on = np.array([float(u.get("onset_ms", 0.0)) for u in r["units_s5"]], dtype=np.float64)
        s5_gain = np.array([float(u.get("gain_db", 0.0)) for u in r["units_s5"]], dtype=np.float64)

        dev_items.append({
            **r,
            "Z1": Z1_d,
            "base_f3": base_f3_d,
            "base_dur": base_dur_d,
            "base_on": base_on_d,
            "base_gain": base_gain_d,
            "d_ctx": d_ctx_d,
            "s5_f3": s5_f3,
            "s5_dur": s5_dur,
            "s5_on": s5_on,
            "s5_gain": s5_gain,
        })

    def _s5_baselines() -> dict[str, float]:
        f3_errs, dur_errs, on_errs, g_errs, g_stds = [], [], [], [], []
        for it in dev_items:
            f3_errs.append(np.mean(np.abs(it["s5_f3"] - it["real_f3"])))
            dur_errs.append(np.mean(np.abs(it["s5_dur"] - it["real_dur"])))
            ho = it["has_on"] & (it["real_on"] >= 5.0)
            if np.any(ho):
                on_errs.append(np.mean(np.abs(it["s5_on"][ho] - it["real_on"][ho])))
            g_errs.append(np.mean(np.abs(it["s5_gain"] - it["real_rel_db"])))
            if len(it["s5_gain"]) >= 2:
                g_stds.append(float(np.std(it["s5_gain"])))
        return {
            "f0_3pt_mae": round(float(np.mean(f3_errs)), 4),
            "dur_mae_ms": round(float(np.mean(dur_errs)), 3),
            "ondur_mae_ms": round(float(np.mean(on_errs)), 3),
            "gain_mae_db": round(float(np.mean(g_errs)), 4),
            "gain_std_med": round(float(np.median(g_stds)), 4),
        }

    s5_metrics = _s5_baselines()
    print(f"DEV_60 ({len(dev_items)} aligned utts) Stage5 baseline: {s5_metrics}", flush=True)

    # Tune Pass-1 + Pass-2 F0 (fit Pass-2 on OOF Pass-1 residuals for each candidate (lam_p1, w_p1))
    best_f0 = None
    for lam_p1 in (100.0, 300.0, 1000.0, 3000.0):
        oof_df1 = oof_p1_by_lam[lam_p1]
        for w_p1_fit in (0.60, 0.75, 0.90):
            p1_tot_oof = Base_f3_all + w_p1_fit * oof_df1
            X2_all = np.empty((n_tot, 34), dtype=np.float64)
            for _fid, c0_i, c1_i, u_list, c_list in Utt_slices:
                X2_all[c0_i:c1_i] = extract_prosody_stage7_pass2_features(u_list, p1_tot_oof[c0_i:c1_i], ctxs=c_list)
            x2_mean = np.mean(X2_all, axis=0)
            x2_std = np.std(X2_all, axis=0)
            x2_mean[0] = 0.0
            x2_std[0] = 1.0
            x2_std = np.where(x2_std < 1e-4, 1.0, x2_std)
            Z2_all = (X2_all - x2_mean[None, :]) / x2_std[None, :]
            Z2_all[:, 0] = 1.0

            XtX_2 = Z2_all.T @ Z2_all
            Y2_res = Y_f3_res - w_p1_fit * oof_df1
            XtY_2 = Z2_all.T @ Y2_res

            # Precompute DEV_60 Pass-1 & Pass-2 predictions
            W1 = w_f0_p1_grid[lam_p1]
            dev_p1 = []
            dev_Z2 = []
            for it in dev_items:
                df1_d = it["Z1"] @ W1
                p1_d = it["base_f3"] + w_p1_fit * df1_d
                X2_d = extract_prosody_stage7_pass2_features(it["units_rule"], p1_d, ctxs=it["ctxs"])
                Z2_d = (X2_d - x2_mean[None, :]) / x2_std[None, :]
                Z2_d[:, 0] = 1.0
                dev_p1.append(df1_d)
                dev_Z2.append(Z2_d)

            for lam_p2 in (100.0, 500.0, 2000.0):
                W2 = _solve_ridge_stats(XtX_2, XtY_2, lam_p2)
                for w_p2 in (0.0, 0.45, 0.70, 0.90):
                    maes = []
                    for idx_d, it in enumerate(dev_items):
                        df = np.clip(w_p1_fit * dev_p1[idx_d] + w_p2 * (dev_Z2[idx_d] @ W2), -6.0, 6.0)
                        maes.append(float(np.mean(np.abs((it["base_f3"] + df) - it["real_f3"]))))
                    mae = float(np.mean(maes))
                    if best_f0 is None or mae < best_f0["mae"]:
                        best_f0 = {
                            "mae": round(mae, 4),
                            "lam_p1": lam_p1,
                            "w_p1": w_p1_fit,
                            "lam_p2": lam_p2,
                            "w_p2": w_p2,
                            "W1": W1,
                            "x2_mean": x2_mean,
                            "x2_std": x2_std,
                            "W2": W2,
                        }

    assert best_f0 is not None
    print(
        f"DEV_60 best Two-Pass F0: mae={best_f0['mae']} st (Stage5={s5_metrics['f0_3pt_mae']} st) "
        f"lam_p1={best_f0['lam_p1']} w_p1={best_f0['w_p1']} lam_p2={best_f0['lam_p2']} w_p2={best_f0['w_p2']}",
        flush=True,
    )

    # Tune Duration Ridge + dur_ctx blend on DEV_60
    best_dur = None
    for lam_d, W_d in w_dur_grid.items():
        preds_d = [it["Z1"] @ W_d for it in dev_items]
        for w_d in (0.35, 0.55, 0.75, 0.95):
            for w_dc in (0.0, 0.25, 0.45):
                errs = []
                for idx_d, it in enumerate(dev_items):
                    dl = np.clip(w_d * preds_d[idx_d] + w_dc * it["d_ctx"], -0.35, 0.35)
                    pdur = np.clip(it["base_dur"] * np.exp(dl), 60.0, 380.0)
                    errs.append(float(np.mean(np.abs(pdur - it["real_dur"]))))
                mae = float(np.mean(errs))
                if best_dur is None or mae < best_dur["mae"]:
                    best_dur = {"mae": round(mae, 3), "lam": lam_d, "w_dur": w_d, "w_dur_ctx": w_dc, "W": W_d}

    assert best_dur is not None
    print(
        f"DEV_60 best Duration: mae={best_dur['mae']} ms (Stage5={s5_metrics['dur_mae_ms']} ms) "
        f"lam={best_dur['lam']} w_dur={best_dur['w_dur']} w_dur_ctx={best_dur['w_dur_ctx']}",
        flush=True,
    )

    # Tune Onset Duration Ridge on DEV_60 (using predicted total dur bounds cleanly)
    best_on = None
    for lam_o, W_o in w_on_grid.items():
        preds_o = [it["Z1"] @ W_o for it in dev_items]
        for w_o in (0.25, 0.45, 0.65, 0.85):
            errs = []
            for idx_d, it in enumerate(dev_items):
                ho = it["has_on"] & (it["real_on"] >= 5.0)
                if not np.any(ho):
                    continue
                ol = np.clip(w_o * preds_o[idx_d][ho], -0.35, 0.35)
                pon = np.clip(it["base_on"][ho] * np.exp(ol), 12.0, np.minimum(200.0, it["real_dur"][ho] - 20.0))
                errs.append(float(np.mean(np.abs(pon - it["real_on"][ho]))))
            mae = float(np.mean(errs))
            if best_on is None or mae < best_on["mae"]:
                best_on = {"mae": round(mae, 3), "lam": lam_o, "w_ondur": w_o, "W": W_o}

    assert best_on is not None
    print(
        f"DEV_60 best OnsetDur: mae={best_on['mae']} ms (Stage5={s5_metrics['ondur_mae_ms']} ms) "
        f"lam={best_on['lam']} w_ondur={best_on['w_ondur']}",
        flush=True,
    )

    # Tune Gain Ridge on DEV_60 keeping Gain Std >= 1.65 dB
    best_gain = None
    for lam_g, W_g in w_gain_grid.items():
        preds_g = [it["Z1"] @ W_g for it in dev_items]
        for w_g in (0.25, 0.45, 0.65, 0.85):
            errs, stds = [], []
            for idx_d, it in enumerate(dev_items):
                dg = np.clip(w_g * preds_g[idx_d], -3.0, 3.0)
                pg = np.clip(it["base_gain"] + dg, -5.5, 5.5)
                errs.append(float(np.mean(np.abs(pg - it["real_rel_db"]))))
                if len(pg) >= 2:
                    stds.append(float(np.std(pg)))
            gstd = float(np.median(stds))
            if gstd < 1.65:
                continue
            mae = float(np.mean(errs))
            if best_gain is None or mae < best_gain["mae"]:
                best_gain = {"mae": round(mae, 4), "std": round(gstd, 4), "lam": lam_g, "w_gain": w_g, "W": W_g}

    if best_gain is None:
        best_gain = {"mae": s5_metrics["gain_mae_db"], "std": s5_metrics["gain_std_med"], "lam": 1000.0, "w_gain": 0.25, "W": w_gain_grid[1000.0]}
    print(
        f"DEV_60 best Gain: mae={best_gain['mae']} dB std={best_gain['std']} dB (Stage5={s5_metrics['gain_mae_db']}/{s5_metrics['gain_std_med']}) "
        f"lam={best_gain['lam']} w_gain={best_gain['w_gain']}",
        flush=True,
    )

    weights = {
        "w_f0_p1": float(best_f0["w_p1"]),
        "w_f0_p2": float(best_f0["w_p2"]),
        "w_dur": float(best_dur["w_dur"]),
        "w_dur_ctx": float(best_dur["w_dur_ctx"]),
        "w_ondur": float(best_on["w_ondur"]),
        "w_gain": float(best_gain["w_gain"]),
        "w_uv": 1.0,
    }

    uv_ofg_keys = sorted(uv_ofg_map.keys())
    uv_o_keys = sorted(uv_o_med.keys())
    max_on = max((len(x) for x in onsets_vocab), default=1)
    max_fi = max((len(x) for x in finals_vocab), default=1)
    max_ofg = max((len(x) for x in uv_ofg_keys), default=1)
    max_o = max((len(x) for x in uv_o_keys), default=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        tritone_vocab=np.array(tritone_vocab, dtype=np.int16) if tritone_vocab else np.zeros((0, 3), dtype=np.int16),
        onsets_vocab=np.array(onsets_vocab, dtype=f"<U{max_on}"),
        finals_vocab=np.array(finals_vocab, dtype=f"<U{max_fi}"),
        x1_mean=x1_mean.astype(np.float32),
        x1_std=x1_std.astype(np.float32),
        w_f0_p1=best_f0["W1"].astype(np.float32),
        x2_mean=best_f0["x2_mean"].astype(np.float32),
        x2_std=best_f0["x2_std"].astype(np.float32),
        w_f0_p2=best_f0["W2"].astype(np.float32),
        w_dur=best_dur["W"].astype(np.float32),
        w_ondur=best_on["W"].astype(np.float32),
        w_gain=best_gain["W"].astype(np.float32),
        dur_ctx_keys=np.array(dur_k, dtype=np.int16) if dur_k else np.zeros((0, 5), dtype=np.int16),
        dur_ctx_vals=np.array(dur_v, dtype=np.float32),
        onset_uv_ofg_keys=np.array(uv_ofg_keys, dtype=f"<U{max_ofg}") if uv_ofg_keys else np.array([], dtype="<U1"),
        onset_uv_ofg_vals=np.array([uv_ofg_map[k] for k in uv_ofg_keys], dtype=np.float32),
        onset_uv_o_keys=np.array(uv_o_keys, dtype=f"<U{max_o}") if uv_o_keys else np.array([], dtype="<U1"),
        onset_uv_o_vals=np.array([uv_o_med[k] for k in uv_o_keys], dtype=np.float32),
        weight_keys=np.array(list(weights.keys()), dtype="<U16"),
        weight_vals=np.array(list(weights.values()), dtype=np.float32),
        matched_utts=np.array([len(aligned)], dtype=np.int32),
    )
    clear_stage7_prosody_cache()

    # Verify roundtrip inference via ProsodyStage7Model matches expectations
    mdl = ProsodyStage7Model(out_path)
    assert mdl.loaded
    e2e_f3, e2e_dur, e2e_on, e2e_g, e2e_gstd = [], [], [], [], []
    for it in dev_items:
        p7 = predict_prosody_stage7(it["text"], kb7, model=mdl)["units"]
        f3_7 = np.array([u["f0_3pt_st"] for u in p7], dtype=np.float64)
        dur_7 = np.array([float(u["duration_ms"]) for u in p7], dtype=np.float64)
        on_7 = np.array([float(u.get("onset_ms", 0.0)) for u in p7], dtype=np.float64)
        g_7 = np.array([float(u.get("gain_db", 0.0)) for u in p7], dtype=np.float64)
        e2e_f3.append(float(np.mean(np.abs(f3_7 - it["real_f3"]))))
        e2e_dur.append(float(np.mean(np.abs(dur_7 - it["real_dur"]))))
        ho = it["has_on"] & (it["real_on"] >= 5.0)
        if np.any(ho):
            e2e_on.append(float(np.mean(np.abs(on_7[ho] - it["real_on"][ho]))))
        e2e_g.append(float(np.mean(np.abs(g_7 - it["real_rel_db"]))))
        if len(g_7) >= 2:
            e2e_gstd.append(float(np.std(g_7)))

    s7_metrics = {
        "f0_3pt_mae": round(float(np.mean(e2e_f3)), 4),
        "dur_mae_ms": round(float(np.mean(e2e_dur)), 3),
        "ondur_mae_ms": round(float(np.mean(e2e_on)), 3),
        "gain_mae_db": round(float(np.mean(e2e_g)), 4),
        "gain_std_med": round(float(np.median(e2e_gstd)), 4),
    }

    size_kb = out_path.stat().st_size / 1024.0
    summary = {
        "matched_train_utts": len(aligned),
        "matched_train_syls": n_tot,
        "d1": int(d1),
        "d2": 34,
        "n_tritones": len(tritone_vocab),
        "n_dur_ctx": len(dur_k),
        "n_onset_uv_ofg": len(uv_ofg_keys),
        "npz_kb": round(size_kb, 1),
        "npz_mb": round(size_kb / 1024.0, 3),
        "weights": weights,
        "lams": {
            "f0_p1": best_f0["lam_p1"],
            "f0_p2": best_f0["lam_p2"],
            "dur": best_dur["lam"],
            "ondur": best_on["lam"],
            "gain": best_gain["lam"],
        },
        "dev60_aligned_utts": len(dev_items),
        "dev60_stage5": s5_metrics,
        "dev60_stage7": s7_metrics,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Stage 7 prosody summary:", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-path", type=Path, default=DEFAULT_S5_CACHE_PATH)
    ap.add_argument("--knowledge-path", type=Path, default=STAGE7_KNOWLEDGE_PATH)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build_prosody_stage7_tables(
        cache_path=args.cache_path,
        knowledge_path=args.knowledge_path,
        limit=args.limit,
    )
