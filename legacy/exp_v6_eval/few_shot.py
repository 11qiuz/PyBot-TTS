"""Few-shot sample efficiency convergence benchmark (1, 2, 3, 8, 15 utterances)."""
from __future__ import annotations

import math
import pathlib
from typing import Sequence

import numpy as np
from scipy.fft import rfft, rfftfreq

from exp_v5_eval.metrics import (
    DEFAULT_SR,
    compute_f0_hnr_metrics,
    compute_ltas_metrics,
    extract_active_frames,
    read_wav,
)


def extract_21band_ltas(x: np.ndarray, sr: int = DEFAULT_SR) -> np.ndarray | None:
    """Extract 21-band log-spaced LTAS (80-8000 Hz, mean-centered dB) over active frames."""
    frame_len = max(64, int(round(0.030 * sr)))
    hop = max(32, int(round(0.010 * sr)))
    frames, keep, _ = extract_active_frames(x, frame_len=frame_len, hop=hop, db_thresh=30.0)
    if int(np.sum(keep)) < 4:
        return None
    kept = frames[keep]
    kept = kept - np.mean(kept, axis=1, keepdims=True)
    win = np.hamming(frame_len)[None, :]
    nfft = 2048 if sr <= 24000 else 4096
    spec = np.abs(rfft(kept * win, n=nfft, axis=1))
    mean_mag = np.mean(spec, axis=0)
    freqs = rfftfreq(nfft, 1.0 / sr)

    hi_limit = min(8000.0, 0.48 * sr)
    centers = np.geomspace(80.0, hi_limit, 21)
    half = (2.0 ** (1.0 / 3.0) - 1.0) * 0.5 * 1.2
    vals = []
    for fc in centers:
        lo, hi = fc * (1.0 - half), fc * (1.0 + half)
        m = (freqs >= lo) & (freqs <= hi)
        if int(np.sum(m)) < 2:
            order = np.argsort(np.abs(freqs - fc))[:2]
            vals.append(float(np.mean(mean_mag[order])))
        else:
            vals.append(float(np.mean(mean_mag[m])))
    db = 20.0 * np.log10(np.maximum(np.array(vals, dtype=np.float64), 1e-12))
    return db - np.mean(db)


def extract_speaker_utterance_features(
    x: np.ndarray,
    sr: int = DEFAULT_SR,
    *,
    fmin: float = 75.0,
    fmax: float = 450.0,
) -> dict:
    """Extract utterance-level speaker timbre summary (F0, IQR, LTAS tilt, 21-band LTAS, HNR)."""
    ltas = compute_ltas_metrics(x, sr=sr)
    f0h = compute_f0_hnr_metrics(x, sr=sr, fmin=fmin, fmax=fmax)
    lt21 = extract_21band_ltas(x, sr=sr)
    if ltas is None or f0h is None or lt21 is None or f0h["f0_median_hz"] is None:
        raise ValueError("Utterance lacks enough voiced active frames for speaker feature extraction")
    return {
        "duration_s": round(len(x) / float(sr), 4),
        "f0_median_hz": float(f0h["f0_median_hz"]),
        "f0_iqr_st": float(f0h["f0_iqr_st"]),
        "tilt_db_oct": float(ltas["tilt_db_oct"]),
        "ltas_21": lt21,
        "hnr_db_median": float(f0h["hnr_db_median"]),
    }


def load_extended_utterance_features(
    utterances: Sequence[dict],
    *,
    fmin: float = 90.0,
    fmax: float = 450.0,
) -> dict[str, list[dict]]:
    """Extract cached utterance features grouped by speaker from extended/index.json entries."""
    by_spk: dict[str, list[dict]] = {}
    for utt in utterances:
        spk = str(utt["speaker"])
        x, sr = read_wav(pathlib.Path(utt["path"]))
        a0 = int(utt.get("active_start", 0))
        a1 = int(utt.get("active_stop", len(x)))
        x_act = x[a0:a1]
        feats = extract_speaker_utterance_features(x_act, sr=sr, fmin=fmin, fmax=fmax)
        feats["id"] = str(utt["id"])
        feats["speaker"] = spk
        by_spk.setdefault(spk, []).append(feats)
    return by_spk


def _mean_profile(rows: Sequence[dict]) -> dict:
    return {
        "duration_s": float(np.sum([r["duration_s"] for r in rows])),
        "f0_median_hz": float(np.mean([r["f0_median_hz"] for r in rows])),
        "f0_iqr_st": float(np.mean([r["f0_iqr_st"] for r in rows])),
        "tilt_db_oct": float(np.mean([r["tilt_db_oct"] for r in rows])),
        "ltas_21": np.mean(np.stack([r["ltas_21"] for r in rows], axis=0), axis=0),
        "hnr_db_median": float(np.mean([r["hnr_db_median"] for r in rows])),
    }


def _profile_errors(est: dict, ref: dict) -> dict:
    f0_err_hz = abs(est["f0_median_hz"] - ref["f0_median_hz"])
    f0_err_st = abs(12.0 * math.log2(est["f0_median_hz"] / ref["f0_median_hz"]))
    iqr_err_st = abs(est["f0_iqr_st"] - ref["f0_iqr_st"])
    tilt_err = abs(est["tilt_db_oct"] - ref["tilt_db_oct"])
    lt21_rmse = float(np.sqrt(np.mean((est["ltas_21"] - ref["ltas_21"]) ** 2)))
    hnr_err = abs(est["hnr_db_median"] - ref["hnr_db_median"])
    return {
        "f0_err_hz": f0_err_hz,
        "f0_err_st": f0_err_st,
        "f0_iqr_err_st": iqr_err_st,
        "tilt_err_db_oct": tilt_err,
        "ltas_21_rmse_db": lt21_rmse,
        "hnr_err_db": hnr_err,
    }


def run_few_shot_convergence(
    by_speaker: dict[str, list[dict]],
    *,
    k_values: Sequence[int] = (1, 2, 3, 8, 15),
    n_trials: int = 60,
    seed: int = 42,
) -> dict:
    """Measure few-shot convergence vs full oracle AND zero-leakage LOO held-out prediction."""
    rng = np.random.default_rng(seed)
    speakers = sorted(by_speaker.keys())
    oracles = {spk: _mean_profile(rows) for spk, rows in by_speaker.items()}

    oracle_curve: dict[str, dict] = {}
    loo_curve: dict[str, dict] = {}

    for k in k_values:
        # 1. Convergence vs full-speaker oracle
        k_durs: list[float] = []
        oracle_errs: list[dict] = []
        for spk in speakers:
            rows = by_speaker[spk]
            n = len(rows)
            k_eff = min(k, n)
            if k_eff == 1:
                subsets = [[i] for i in range(n)]
            elif k_eff == n:
                subsets = [list(range(n))]
            elif k_eff == n - 1:
                subsets = [[j for j in range(n) if j != i] for i in range(n)]
            else:
                subsets = [rng.choice(n, size=k_eff, replace=False).tolist() for _ in range(n_trials)]

            for idxs in subsets:
                est = _mean_profile([rows[i] for i in idxs])
                k_durs.append(est["duration_s"])
                oracle_errs.append(_profile_errors(est, oracles[spk]))

        oracle_curve[str(k)] = {
            "k": int(k),
            "mean_enroll_s": round(float(np.mean(k_durs)), 2),
            "n_evals": len(oracle_errs),
            "f0_mae_hz": round(float(np.mean([e["f0_err_hz"] for e in oracle_errs])), 3),
            "f0_mae_st": round(float(np.mean([e["f0_err_st"] for e in oracle_errs])), 4),
            "f0_iqr_mae_st": round(float(np.mean([e["f0_iqr_err_st"] for e in oracle_errs])), 4),
            "tilt_mae_db_oct": round(float(np.mean([e["tilt_err_db_oct"] for e in oracle_errs])), 4),
            "ltas_21_rmse_db": round(float(np.mean([e["ltas_21_rmse_db"] for e in oracle_errs])), 4),
            "hnr_mae_db": round(float(np.mean([e["hnr_err_db"] for e in oracle_errs])), 4),
        }

        # 2. Zero-leakage LOO held-out utterance prediction + 2-speaker match accuracy
        loo_errs: list[dict] = []
        acc_f0: list[int] = []
        acc_tilt: list[int] = []
        acc_lt21: list[int] = []
        acc_hnr: list[int] = []
        acc_comb: list[int] = []

        # Normalization scales computed strictly per-fold below when >= 2 speakers
        for true_spk in speakers:
            rows_true = by_speaker[true_spk]
            for hold_idx, test_row in enumerate(rows_true):
                pool_true = [r for i, r in enumerate(rows_true) if i != hold_idx]
                if not pool_true:
                    continue
                k_eff = min(k, len(pool_true))
                if k_eff == len(pool_true):
                    n_draw = 1
                elif k_eff == 1:
                    n_draw = len(pool_true)
                else:
                    n_draw = min(20, n_trials)

                for d_i in range(n_draw):
                    if k_eff == 1:
                        enroll_true = [pool_true[d_i]]
                    elif k_eff == len(pool_true):
                        enroll_true = pool_true
                    else:
                        idx_t = rng.choice(len(pool_true), size=k_eff, replace=False)
                        enroll_true = [pool_true[int(j)] for j in idx_t]

                    prof_by_spk = {true_spk: _mean_profile(enroll_true)}
                    for other_spk in speakers:
                        if other_spk == true_spk:
                            continue
                        pool_o = by_speaker[other_spk]
                        ko = min(k, len(pool_o))
                        if ko == len(pool_o):
                            enroll_o = pool_o
                        else:
                            idx_o = rng.choice(len(pool_o), size=ko, replace=False)
                            enroll_o = [pool_o[int(j)] for j in idx_o]
                        prof_by_spk[other_spk] = _mean_profile(enroll_o)

                    loo_errs.append(_profile_errors(prof_by_spk[true_spk], test_row))

                    if len(speakers) >= 2:
                        pred_f0 = min(speakers, key=lambda s: abs(test_row["f0_median_hz"] - prof_by_spk[s]["f0_median_hz"]))
                        pred_tilt = min(speakers, key=lambda s: abs(test_row["tilt_db_oct"] - prof_by_spk[s]["tilt_db_oct"]))
                        pred_lt21 = min(
                            speakers,
                            key=lambda s: float(np.sqrt(np.mean((test_row["ltas_21"] - prof_by_spk[s]["ltas_21"]) ** 2))),
                        )
                        pred_hnr = min(speakers, key=lambda s: abs(test_row["hnr_db_median"] - prof_by_spk[s]["hnr_db_median"]))

                        # Scale-normalized 4-cue distance (using pooled fold std, zero test leakage)
                        all_fold_rows = enroll_true + [r for s in speakers if s != true_spk for r in by_speaker[s]]
                        sf0 = max(1.0, float(np.std([r["f0_median_hz"] for r in all_fold_rows])))
                        siqr = max(0.2, float(np.std([r["f0_iqr_st"] for r in all_fold_rows])))
                        stilt = max(0.2, float(np.std([r["tilt_db_oct"] for r in all_fold_rows])))
                        shnr = max(0.5, float(np.std([r["hnr_db_median"] for r in all_fold_rows])))

                        def _comb_d(s: str) -> float:
                            p = prof_by_spk[s]
                            return (
                                ((test_row["f0_median_hz"] - p["f0_median_hz"]) / sf0) ** 2
                                + ((test_row["f0_iqr_st"] - p["f0_iqr_st"]) / siqr) ** 2
                                + ((test_row["tilt_db_oct"] - p["tilt_db_oct"]) / stilt) ** 2
                                + ((test_row["hnr_db_median"] - p["hnr_db_median"]) / shnr) ** 2
                            )

                        pred_comb = min(speakers, key=_comb_d)
                        acc_f0.append(int(pred_f0 == true_spk))
                        acc_tilt.append(int(pred_tilt == true_spk))
                        acc_lt21.append(int(pred_lt21 == true_spk))
                        acc_hnr.append(int(pred_hnr == true_spk))
                        acc_comb.append(int(pred_comb == true_spk))

        loo_entry: dict = {
            "k": int(k),
            "n_evals": len(loo_errs),
            "heldout_f0_mae_hz": round(float(np.mean([e["f0_err_hz"] for e in loo_errs])), 3),
            "heldout_f0_mae_st": round(float(np.mean([e["f0_err_st"] for e in loo_errs])), 4),
            "heldout_f0_iqr_mae_st": round(float(np.mean([e["f0_iqr_err_st"] for e in loo_errs])), 4),
            "heldout_tilt_mae_db_oct": round(float(np.mean([e["tilt_err_db_oct"] for e in loo_errs])), 4),
            "heldout_ltas_21_rmse_db": round(float(np.mean([e["ltas_21_rmse_db"] for e in loo_errs])), 4),
            "heldout_hnr_mae_db": round(float(np.mean([e["hnr_err_db"] for e in loo_errs])), 4),
        }
        if acc_comb:
            loo_entry["id_acc"] = {
                "f0_only": round(float(np.mean(acc_f0)), 4),
                "tilt_only": round(float(np.mean(acc_tilt)), 4),
                "ltas_21band": round(float(np.mean(acc_lt21)), 4),
                "hnr_only": round(float(np.mean(acc_hnr)), 4),
                "combined_4cue": round(float(np.mean(acc_comb)), 4),
            }
        loo_curve[str(k)] = loo_entry

    return {
        "speakers": speakers,
        "oracles": {
            spk: {
                "n_utterances": len(by_speaker[spk]),
                "total_active_s": round(oracles[spk]["duration_s"], 2),
                "f0_median_hz": round(oracles[spk]["f0_median_hz"], 2),
                "f0_iqr_st": round(oracles[spk]["f0_iqr_st"], 3),
                "tilt_db_oct": round(oracles[spk]["tilt_db_oct"], 3),
                "hnr_db_median": round(oracles[spk]["hnr_db_median"], 3),
            }
            for spk in speakers
        },
        "oracle_convergence": oracle_curve,
        "zero_leakage_loo": loo_curve,
    }
