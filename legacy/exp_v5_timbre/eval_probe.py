"""Phase 1 LOO-CV evaluation probe, acoustic gap benchmark, and sample exporter."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pybot_tts_v2_recorded import engine  # noqa: E402
from pybot_tts_v4_params import speaker as spk_mod, vocal  # noqa: E402

from exp_v5_timbre import (  # noqa: E402
    EXP_DIR,
    EXTENDED_INDEX,
    PHONEME_PARAMS,
    SAMPLES_DIR,
    SPEAKER_PARAMS,
    SR,
)
from exp_v5_timbre.features import (  # noqa: E402
    compute_dtw_mcd,
    compute_dtw_mel_lsd,
    compute_formant_speed,
    extract_f0_hnr,
    extract_ltas,
    extract_mfcc,
    fit_fold_model,
    read_wav_mono,
    resample_trajectory,
    track_formant_trajectory,
    write_wav_mono,
)
from exp_v5_timbre.synth_v5 import build_units_from_utterance, synthesize_variant  # noqa: E402


VARIANTS = ("v4_baseline", "v5_tilt_f0", "v5_glottal_hnr", "v5_full")


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _cohens_d(a: list[float], b: list[float]) -> float:
    xa, xb = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    va = float(np.var(xa, ddof=1)) if len(xa) > 1 else 0.0
    vb = float(np.var(xb, ddof=1)) if len(xb) > 1 else 0.0
    pooled = math.sqrt(max(1e-12, 0.5 * (va + vb)))
    return float((np.mean(xa) - np.mean(xb)) / pooled)


def _nearest_centroid_loo(utts: list[dict], vecs: dict[str, np.ndarray]) -> dict:
    """Strict LOO-CV standardized nearest-centroid classification on real utterances."""
    correct = 0
    scores_502 = []
    labels = []
    preds = []
    for i, held in enumerate(utts):
        train = [u for j, u in enumerate(utts) if j != i]
        x_tr = np.stack([vecs[u["id"]] for u in train], axis=0)
        y_tr = [u["speaker"] for u in train]
        mu = np.mean(x_tr, axis=0)
        sd = np.maximum(np.std(x_tr, axis=0, ddof=1), 1e-6)

        c502 = np.mean([(vecs[u["id"]] - mu) / sd for u in train if u["speaker"] == "SSB0502"], axis=0)
        c1593 = np.mean([(vecs[u["id"]] - mu) / sd for u in train if u["speaker"] == "SSB1593"], axis=0)

        x_te = (vecs[held["id"]] - mu) / sd
        d502 = float(np.linalg.norm(x_te - c502))
        d1593 = float(np.linalg.norm(x_te - c1593))
        pred = "SSB0502" if d502 < d1593 else "SSB1593"
        preds.append(pred)
        labels.append(held["speaker"])
        if pred == held["speaker"]:
            correct += 1
        # Margin positive -> favors SSB0502
        scores_502.append((d1593 - d502, 1 if held["speaker"] == "SSB0502" else 0))

    # ROC-AUC from margin scores
    pos = [s for s, y in scores_502 if y == 1]
    neg = [s for s, y in scores_502 if y == 0]
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    auc = wins / max(1, len(pos) * len(neg))
    return {
        "accuracy": round(correct / len(utts), 4),
        "correct": int(correct),
        "total": int(len(utts)),
        "roc_auc": round(auc, 4),
    }


def _extract_oracle_prosody(seg_real: np.ndarray, total_active_samples: int) -> tuple[np.ndarray, dict]:
    """Extract smooth F0 and tracked F1/F2/F3/BW1/BW2 aligned to active speech samples."""
    f0h = extract_f0_hnr(seg_real, sr=SR)
    t_f0 = f0h["times_s"]
    f0_vals = f0h["f0_hz"]
    voiced = np.isfinite(f0_vals)
    t_dst = np.arange(total_active_samples, dtype=np.float64) / SR
    if np.sum(voiced) >= 2:
        oracle_f0 = np.interp(t_dst, t_f0[voiced], f0_vals[voiced])
    else:
        oracle_f0 = np.full(total_active_samples, 210.0, dtype=np.float64)

    traj = track_formant_trajectory(seg_real, sr=SR)
    oracle_traj = {}
    if traj is not None and len(traj["times_s"]) >= 2:
        for k in ("f1", "f2", "f3", "bw1", "bw2"):
            oracle_traj[k] = np.interp(t_dst, traj["times_s"], traj[k])
    return oracle_f0, oracle_traj


def _eval_pair_metrics(syn: np.ndarray, real_feats: dict) -> dict:
    """Compute acoustic distance metrics between a synthesized utterance and real reference."""
    lt_s = extract_ltas(syn, sr=SR)
    f0_s = extract_f0_hnr(syn, sr=SR)
    mf_s = extract_mfcc(syn, sr=SR)

    lt_r = real_feats["ltas"]
    f0_r = real_feats["f0_hnr"]
    mf_r = real_feats["mfcc"]

    tilt_err = abs(lt_s["tilt_db_per_oct"] - lt_r["tilt_db_per_oct"])
    ltas_env_rmse = float(np.sqrt(np.mean((lt_s["ltas_db_norm"] - lt_r["ltas_db_norm"]) ** 2)))
    if np.isfinite(f0_s["f0_median_hz"]) and np.isfinite(f0_r["f0_median_hz"]):
        f0_med_err_st = abs(12.0 * math.log2(f0_s["f0_median_hz"] / f0_r["f0_median_hz"]))
    else:
        f0_med_err_st = float("nan")
    if np.isfinite(f0_s["f0_iqr_st"]) and np.isfinite(f0_r["f0_iqr_st"]):
        f0_iqr_err_st = abs(f0_s["f0_iqr_st"] - f0_r["f0_iqr_st"])
    else:
        f0_iqr_err_st = float("nan")
    if np.isfinite(f0_s["hnr_median_db"]) and np.isfinite(f0_r["hnr_median_db"]):
        hnr_err_db = abs(f0_s["hnr_median_db"] - f0_r["hnr_median_db"])
    else:
        hnr_err_db = float("nan")

    mcd = compute_dtw_mcd(mf_r["mfcc"], mf_s["mfcc"])
    mel_lsd = compute_dtw_mel_lsd(mf_r["log_mel_db_norm"], mf_s["log_mel_db_norm"])
    mfcc_cos = _cosine_sim(mf_r["mfcc_mean"][1:13], mf_s["mfcc_mean"][1:13])

    return {
        "syn_ltas": lt_s,
        "syn_f0": f0_s,
        "syn_mfcc": mf_s,
        "tilt_err_db_oct": float(tilt_err),
        "ltas_env_rmse_db": float(ltas_env_rmse),
        "f0_med_err_st": float(f0_med_err_st),
        "f0_iqr_err_st": float(f0_iqr_err_st),
        "hnr_err_db": float(hnr_err_db),
        "dtw_mcd_db": float(mcd),
        "dtw_mel_lsd_db": float(mel_lsd),
        "mfcc_cos_sim": float(mfcc_cos),
    }


def _probe_diphthong_trajectories(fold_full: dict, pp_norm: dict, pp_502: dict) -> dict:
    """Compare /a/ and /ai/ formant trajectory spans in v4_baseline vs v5_full vs real."""
    probes = {}
    for final_key, pinyin in (("a", "ma1"), ("ai", "tai2"), ("ao", "hao3"), ("uo", "guo2")):
        o, f, t = engine.split_pinyin(pinyin)
        units = [{
            "hanzi": "",
            "onset": o,
            "final": f,
            "tone": t,
            "duration_ms": 340.0,
            "gain": 0.83,
            "pitch_st": list(engine.CURVE[t]),
            "pause_after_ms": 0.0,
        }]
        w_v4 = synthesize_variant("v4_baseline", units, pp_norm, pp_502, fold_full["speakers"]["SSB0502"], "SSB0502", seed=72)
        w_v5 = synthesize_variant("v5_full", units, pp_norm, pp_502, fold_full["speakers"]["SSB0502"], "SSB0502", seed=72)
        tr_v4 = track_formant_trajectory(w_v4, sr=SR)
        tr_v5 = track_formant_trajectory(w_v5, sr=SR)
        real_traj = fold_full["speakers"]["SSB0502"]["final_trajectories"].get(final_key)
        entry = {"real_5pt_f1": None, "real_5pt_f2": None, "v4_f2_span_hz": None, "v5_f2_span_hz": None, "v4_f1_med": None, "v5_f1_med": None}
        if real_traj is not None:
            arr = np.array(real_traj)
            entry["real_5pt_f1"] = [round(float(v), 1) for v in arr[:, 0]]
            entry["real_5pt_f2"] = [round(float(v), 1) for v in arr[:, 1]]
            entry["real_f2_span_hz"] = round(float(np.max(arr[:, 1]) - np.min(arr[:, 1])), 1)
        if tr_v4 is not None:
            entry["v4_f1_med"] = round(float(np.median(tr_v4["f1"])), 1)
            entry["v4_f2_med"] = round(float(np.median(tr_v4["f2"])), 1)
            entry["v4_f2_span_hz"] = round(float(np.max(tr_v4["f2"]) - np.min(tr_v4["f2"])), 1)
        if tr_v5 is not None:
            entry["v5_f1_med"] = round(float(np.median(tr_v5["f1"])), 1)
            entry["v5_f2_med"] = round(float(np.median(tr_v5["f2"])), 1)
            entry["v5_f2_span_hz"] = round(float(np.max(tr_v5["f2"]) - np.min(tr_v5["f2"])), 1)
        probes[final_key] = entry
    return probes


def run_experiment(export_samples: bool = True) -> dict:
    t0 = time.time()
    idx = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))
    utts = idx["utterances"]
    pp_raw = json.loads(PHONEME_PARAMS.read_text(encoding="utf-8"))
    pp_norm = vocal.load_params(PHONEME_PARAMS)
    sp_doc = json.loads(SPEAKER_PARAMS.read_text(encoding="utf-8"))
    pp_by_spk = {
        "SSB0502": spk_mod.apply_speaker(sp_doc, "SSB0502", base=json.loads(PHONEME_PARAMS.read_text(encoding="utf-8"))),
        "SSB1593": spk_mod.apply_speaker(sp_doc, "SSB1593", base=json.loads(PHONEME_PARAMS.read_text(encoding="utf-8"))),
    }

    # Pre-extract per-utterance single-file features & per-syllable LPC pole-tracks (no cross-utterance info)
    real_audio: dict[str, np.ndarray] = {}
    utt_feats: dict[str, dict] = {}
    syl_trajs: dict[str, list[tuple[str, np.ndarray]]] = {}
    for u in utts:
        uid = u["id"]
        x, sr = read_wav_mono(u["path"])
        if sr != SR:
            raise ValueError(f"Expected {SR} Hz, got {sr} for {uid}")
        seg = x[u["active_start"] : u["active_stop"]]
        real_audio[uid] = seg
        utt_feats[uid] = {
            "ltas": extract_ltas(seg, sr=SR),
            "f0_hnr": extract_f0_hnr(seg, sr=SR),
            "mfcc": extract_mfcc(seg, sr=SR),
        }
        s_list = []
        for syl in u["syllables"]:
            _, final_key, _ = engine.split_pinyin(syl["pinyin"])
            s_seg = x[syl["start_sample"] : syl["stop_sample"]]
            tr = track_formant_trajectory(s_seg, sr=SR)
            res = resample_trajectory(tr, n_pts=5)
            if res is not None:
                s_list.append((final_key, res))
        syl_trajs[uid] = s_list

    # 1. Real-vs-Real Speaker Separability (LOO-CV)
    real_vecs = {
        "tilt_only": {u["id"]: np.array([utt_feats[u["id"]]["ltas"]["tilt_db_per_oct"]]) for u in utts},
        "ltas_vec": {u["id"]: utt_feats[u["id"]]["ltas"]["ltas_db_norm"] for u in utts},
        "f0_stats": {
            u["id"]: np.array([
                utt_feats[u["id"]]["f0_hnr"]["f0_median_hz"],
                utt_feats[u["id"]]["f0_hnr"]["f0_iqr_st"],
            ])
            for u in utts
        },
        "hnr_only": {u["id"]: np.array([utt_feats[u["id"]]["f0_hnr"]["hnr_median_db"]]) for u in utts},
        "mfcc_mean": {u["id"]: utt_feats[u["id"]]["mfcc"]["mfcc_mean"][1:13] for u in utts},
        "combined_timbre": {
            u["id"]: np.array([
                utt_feats[u["id"]]["ltas"]["tilt_db_per_oct"],
                utt_feats[u["id"]]["f0_hnr"]["f0_median_hz"],
                utt_feats[u["id"]]["f0_hnr"]["f0_iqr_st"],
                utt_feats[u["id"]]["f0_hnr"]["hnr_median_db"],
            ])
            for u in utts
        },
        "all_features": {
            u["id"]: np.concatenate([
                [
                    utt_feats[u["id"]]["ltas"]["tilt_db_per_oct"],
                    utt_feats[u["id"]]["f0_hnr"]["f0_median_hz"],
                    utt_feats[u["id"]]["f0_hnr"]["f0_iqr_st"],
                    utt_feats[u["id"]]["f0_hnr"]["hnr_median_db"],
                ],
                utt_feats[u["id"]]["ltas"]["ltas_db_norm"],
                utt_feats[u["id"]]["mfcc"]["mfcc_mean"][1:13],
            ])
            for u in utts
        },
    }
    real_loo = {name: _nearest_centroid_loo(utts, vecs) for name, vecs in real_vecs.items()}

    # Real speaker summary stats
    real_summary = {}
    for spk in ("SSB0502", "SSB1593"):
        sub = [utt_feats[u["id"]] for u in utts if u["speaker"] == spk]
        real_summary[spk] = {
            "n_utterances": len(sub),
            "tilt_mean_db_oct": round(float(np.mean([f["ltas"]["tilt_db_per_oct"] for f in sub])), 3),
            "tilt_std_db_oct": round(float(np.std([f["ltas"]["tilt_db_per_oct"] for f in sub], ddof=1)), 3),
            "f0_median_mean_hz": round(float(np.mean([f["f0_hnr"]["f0_median_hz"] for f in sub])), 2),
            "f0_median_std_hz": round(float(np.std([f["f0_hnr"]["f0_median_hz"] for f in sub], ddof=1)), 2),
            "f0_iqr_mean_st": round(float(np.mean([f["f0_hnr"]["f0_iqr_st"] for f in sub])), 3),
            "hnr_mean_db": round(float(np.mean([f["f0_hnr"]["hnr_median_db"] for f in sub])), 3),
            "hnr_std_db": round(float(np.std([f["f0_hnr"]["hnr_median_db"] for f in sub], ddof=1)), 3),
        }
    real_effect_sizes = {
        "tilt_cohens_d": round(
            _cohens_d(
                [utt_feats[u["id"]]["ltas"]["tilt_db_per_oct"] for u in utts if u["speaker"] == "SSB0502"],
                [utt_feats[u["id"]]["ltas"]["tilt_db_per_oct"] for u in utts if u["speaker"] == "SSB1593"],
            ),
            3,
        ),
        "f0_cohens_d": round(
            _cohens_d(
                [utt_feats[u["id"]]["f0_hnr"]["f0_median_hz"] for u in utts if u["speaker"] == "SSB0502"],
                [utt_feats[u["id"]]["f0_hnr"]["f0_median_hz"] for u in utts if u["speaker"] == "SSB1593"],
            ),
            3,
        ),
        "f0_iqr_cohens_d": round(
            _cohens_d(
                [utt_feats[u["id"]]["f0_hnr"]["f0_iqr_st"] for u in utts if u["speaker"] == "SSB0502"],
                [utt_feats[u["id"]]["f0_hnr"]["f0_iqr_st"] for u in utts if u["speaker"] == "SSB1593"],
            ),
            3,
        ),
        "hnr_cohens_d": round(
            _cohens_d(
                [utt_feats[u["id"]]["f0_hnr"]["hnr_median_db"] for u in utts if u["speaker"] == "SSB0502"],
                [utt_feats[u["id"]]["f0_hnr"]["hnr_median_db"] for u in utts if u["speaker"] == "SSB1593"],
            ),
            3,
        ),
    }

    # 2. Strict 32-fold LOO-CV Synthesis Evaluation
    per_variant_records: dict[str, list[dict]] = {v: [] for v in (*VARIANTS, "v5_oracle_prosody")}
    sample_utterance_ids = {"SSB05020005", "SSB15930031"}

    for fold_i, held in enumerate(utts):
        uid = held["id"]
        true_spk = held["speaker"]
        other_spk = "SSB1593" if true_spk == "SSB0502" else "SSB0502"
        train = [u for j, u in enumerate(utts) if j != fold_i]
        fold = fit_fold_model(train, utt_feats, syl_trajs, pp_raw)
        assert uid not in fold["train_ids"], f"Leakage detected for {uid}"

        units = build_units_from_utterance(held, use_real_durations=True)
        seed = 72 + fold_i

        for var in VARIANTS:
            wav_502 = synthesize_variant(
                var, units, pp_norm, pp_by_spk["SSB0502"], fold["speakers"]["SSB0502"], "SSB0502", seed=seed
            )
            wav_1593 = synthesize_variant(
                var, units, pp_norm, pp_by_spk["SSB1593"], fold["speakers"]["SSB1593"], "SSB1593", seed=seed
            )
            m_502 = _eval_pair_metrics(wav_502, utt_feats[uid])
            m_1593 = _eval_pair_metrics(wav_1593, utt_feats[uid])
            m_true = m_502 if true_spk == "SSB0502" else m_1593
            m_other = m_1593 if true_spk == "SSB0502" else m_502

            # Inter-speaker separation on the exact same text
            n_min = min(len(wav_502), len(wav_1593))
            wave_max_diff = float(np.max(np.abs(wav_502[:n_min] - wav_1593[:n_min])))
            wave_rms_diff = float(np.sqrt(np.mean((wav_502[:n_min] - wav_1593[:n_min]) ** 2)))
            sep_tilt = abs(m_502["syn_ltas"]["tilt_db_per_oct"] - m_1593["syn_ltas"]["tilt_db_per_oct"])
            sep_ltas_l2 = float(
                np.sqrt(np.mean((m_502["syn_ltas"]["ltas_db_norm"] - m_1593["syn_ltas"]["ltas_db_norm"]) ** 2))
            )
            sep_f0_st = abs(12.0 * math.log2(m_502["syn_f0"]["f0_median_hz"] / m_1593["syn_f0"]["f0_median_hz"]))
            sep_iqr_st = abs(m_502["syn_f0"]["f0_iqr_st"] - m_1593["syn_f0"]["f0_iqr_st"])
            sep_hnr_db = abs(m_502["syn_f0"]["hnr_median_db"] - m_1593["syn_f0"]["hnr_median_db"])
            sep_mfcc_cos_dist = 1.0 - _cosine_sim(
                m_502["syn_mfcc"]["mfcc_mean"][1:13], m_1593["syn_mfcc"]["mfcc_mean"][1:13]
            )

            # Combined speaker matching score: lower distance wins
            # (normalized per-axis error: tilt / 0.6dB + f0_st / 1.0st + hnr / 1.5dB)
            score_true = (
                m_true["tilt_err_db_oct"] / 0.6
                + m_true["f0_med_err_st"] / 1.0
                + m_true["hnr_err_db"] / 1.5
            )
            score_other = (
                m_other["tilt_err_db_oct"] / 0.6
                + m_other["f0_med_err_st"] / 1.0
                + m_other["hnr_err_db"] / 1.5
            )

            per_variant_records[var].append({
                "id": uid,
                "true_speaker": true_spk,
                "tilt_err_db_oct": m_true["tilt_err_db_oct"],
                "ltas_env_rmse_db": m_true["ltas_env_rmse_db"],
                "f0_med_err_st": m_true["f0_med_err_st"],
                "f0_iqr_err_st": m_true["f0_iqr_err_st"],
                "hnr_err_db": m_true["hnr_err_db"],
                "dtw_mcd_db": m_true["dtw_mcd_db"],
                "dtw_mel_lsd_db": m_true["dtw_mel_lsd_db"],
                "mfcc_cos_sim": m_true["mfcc_cos_sim"],
                # Separation metrics
                "sep_wave_max_diff": wave_max_diff,
                "sep_wave_rms_diff": wave_rms_diff,
                "sep_tilt_db_oct": sep_tilt,
                "sep_ltas_l2_db": sep_ltas_l2,
                "sep_f0_st": sep_f0_st,
                "sep_iqr_st": sep_iqr_st,
                "sep_hnr_db": sep_hnr_db,
                "sep_mfcc_cos_dist": sep_mfcc_cos_dist,
                # 2-way synth-to-real speaker match
                "match_tilt": bool(m_true["tilt_err_db_oct"] + 1e-9 < m_other["tilt_err_db_oct"]),
                "match_ltas_env": bool(m_true["ltas_env_rmse_db"] + 1e-9 < m_other["ltas_env_rmse_db"]),
                "match_f0": bool(m_true["f0_med_err_st"] + 1e-9 < m_other["f0_med_err_st"]),
                "match_hnr": bool(m_true["hnr_err_db"] + 1e-9 < m_other["hnr_err_db"]),
                "match_mcd": bool(m_true["dtw_mcd_db"] + 1e-9 < m_other["dtw_mcd_db"]),
                "match_combined": bool(score_true + 1e-9 < score_other),
            })

            if export_samples and uid in sample_utterance_ids:
                SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
                wav_true = wav_502 if true_spk == "SSB0502" else wav_1593
                wav_swap = wav_1593 if true_spk == "SSB0502" else wav_502
                write_wav_mono(SAMPLES_DIR / f"{uid}_{var}.wav", wav_true, sr=SR)
                if var == "v5_full":
                    write_wav_mono(SAMPLES_DIR / f"{uid}_{var}_as_{other_spk}.wav", wav_swap, sr=SR)

        # Diagnostic upper-bound probe: v5_oracle_prosody
        active_len = len(real_audio[uid])
        oracle_f0, oracle_traj = _extract_oracle_prosody(real_audio[uid], active_len)
        wav_oracle = synthesize_variant(
            "v5_oracle_prosody",
            units,
            pp_norm,
            pp_by_spk[true_spk],
            fold["speakers"][true_spk],
            true_spk,
            seed=seed,
            oracle_f0=oracle_f0,
            oracle_traj=oracle_traj,
        )
        m_ora = _eval_pair_metrics(wav_oracle, utt_feats[uid])
        per_variant_records["v5_oracle_prosody"].append({
            "id": uid,
            "true_speaker": true_spk,
            "tilt_err_db_oct": m_ora["tilt_err_db_oct"],
            "ltas_env_rmse_db": m_ora["ltas_env_rmse_db"],
            "f0_med_err_st": m_ora["f0_med_err_st"],
            "f0_iqr_err_st": m_ora["f0_iqr_err_st"],
            "hnr_err_db": m_ora["hnr_err_db"],
            "dtw_mcd_db": m_ora["dtw_mcd_db"],
            "mfcc_cos_sim": m_ora["mfcc_cos_sim"],
        })
        if export_samples and uid in sample_utterance_ids:
            write_wav_mono(SAMPLES_DIR / f"{uid}_real.wav", real_audio[uid], sr=SR)
            write_wav_mono(SAMPLES_DIR / f"{uid}_v5_oracle_prosody.wav", wav_oracle, sr=SR)

        per_variant_records["v5_oracle_prosody"][-1]["dtw_mel_lsd_db"] = m_ora["dtw_mel_lsd_db"]

    # Measure formant transition speeds on sample utterances
    formant_speed_probe = {}
    for suid in sorted(sample_utterance_ids):
        held_s = next(u for u in utts if u["id"] == suid)
        sp_real = compute_formant_speed(real_audio[suid], sr=SR)
        units_s = build_units_from_utterance(held_s)
        train_s = [u for u in utts if u["id"] != suid]
        fold_s = fit_fold_model(train_s, utt_feats, syl_trajs, pp_raw)
        tsp = held_s["speaker"]
        sp_v4 = compute_formant_speed(
            synthesize_variant("v4_baseline", units_s, pp_norm, pp_by_spk[tsp], fold_s["speakers"][tsp], tsp, seed=72), sr=SR
        )
        sp_v5 = compute_formant_speed(
            synthesize_variant("v5_full", units_s, pp_norm, pp_by_spk[tsp], fold_s["speakers"][tsp], tsp, seed=72), sr=SR
        )
        formant_speed_probe[suid] = {
            "real_df1_df2_hz_s": [round(sp_real[0], 1), round(sp_real[1], 1)],
            "v4_df1_df2_hz_s": [round(sp_v4[0], 1), round(sp_v4[1], 1)],
            "v5_full_df1_df2_hz_s": [round(sp_v5[0], 1), round(sp_v5[1], 1)],
        }

    # Aggregate variant summaries
    variant_summary = {}
    for var, recs in per_variant_records.items():
        summary = {
            "n_folds": len(recs),
            "acoustic_gap_vs_real": {
                "tilt_mae_db_oct": round(float(np.mean([r["tilt_err_db_oct"] for r in recs])), 3),
                "ltas_env_rmse_db": round(float(np.mean([r["ltas_env_rmse_db"] for r in recs])), 3),
                "f0_med_err_st": round(float(np.nanmean([r["f0_med_err_st"] for r in recs])), 3),
                "f0_iqr_err_st": round(float(np.nanmean([r["f0_iqr_err_st"] for r in recs])), 3),
                "hnr_err_db": round(float(np.nanmean([r["hnr_err_db"] for r in recs])), 3),
                "dtw_mcd_db": round(float(np.mean([r["dtw_mcd_db"] for r in recs])), 3),
                "dtw_mel_lsd_db": round(float(np.mean([r["dtw_mel_lsd_db"] for r in recs])), 3),
                "mfcc_cos_sim": round(float(np.mean([r["mfcc_cos_sim"] for r in recs])), 4),
            },
        }
        if var != "v5_oracle_prosody":
            summary["inter_speaker_separation"] = {
                "wave_max_diff": round(float(np.mean([r["sep_wave_max_diff"] for r in recs])), 4),
                "wave_rms_diff": round(float(np.mean([r["sep_wave_rms_diff"] for r in recs])), 4),
                "tilt_gap_db_oct": round(float(np.mean([r["sep_tilt_db_oct"] for r in recs])), 3),
                "ltas_l2_db": round(float(np.mean([r["sep_ltas_l2_db"] for r in recs])), 3),
                "f0_gap_st": round(float(np.mean([r["sep_f0_st"] for r in recs])), 3),
                "iqr_gap_st": round(float(np.mean([r["sep_iqr_st"] for r in recs])), 3),
                "hnr_gap_db": round(float(np.mean([r["sep_hnr_db"] for r in recs])), 3),
                "mfcc_cos_dist": round(float(np.mean([r["sep_mfcc_cos_dist"] for r in recs])), 4),
            }
            summary["synth_to_real_match_acc"] = {
                "tilt": round(float(np.mean([r["match_tilt"] for r in recs])), 4),
                "ltas_env": round(float(np.mean([r["match_ltas_env"] for r in recs])), 4),
                "f0": round(float(np.mean([r["match_f0"] for r in recs])), 4),
                "hnr": round(float(np.mean([r["match_hnr"] for r in recs])), 4),
                "mcd": round(float(np.mean([r["match_mcd"] for r in recs])), 4),
                "combined_timbre": round(float(np.mean([r["match_combined"] for r in recs])), 4),
            }
        variant_summary[var] = summary

    # Full-corpus fit for "现在回家？" sample & diphthong trajectory probe
    fold_full = fit_fold_model(utts, utt_feats, syl_trajs, pp_raw)
    diphthong_probe = _probe_diphthong_trajectories(fold_full, pp_norm, pp_by_spk["SSB0502"])
    if export_samples:
        demo_units = engine.plan("现在回家？", "auto")["units"]
        w_demo_v4 = synthesize_variant(
            "v4_baseline", demo_units, pp_norm, pp_by_spk["SSB0502"], fold_full["speakers"]["SSB0502"], "SSB0502", seed=72
        )
        w_demo_v5_502 = synthesize_variant(
            "v5_full", demo_units, pp_norm, pp_by_spk["SSB0502"], fold_full["speakers"]["SSB0502"], "SSB0502", seed=72
        )
        w_demo_v5_1593 = synthesize_variant(
            "v5_full", demo_units, pp_norm, pp_by_spk["SSB1593"], fold_full["speakers"]["SSB1593"], "SSB1593", seed=72
        )
        write_wav_mono(SAMPLES_DIR / "xianzai_huijia_v4_SSB0502.wav", w_demo_v4, sr=SR)
        write_wav_mono(SAMPLES_DIR / "xianzai_huijia_v5_full_SSB0502.wav", w_demo_v5_502, sr=SR)
        write_wav_mono(SAMPLES_DIR / "xianzai_huijia_v5_full_SSB1593.wav", w_demo_v5_1593, sr=SR)

    out = {
        "schema": 1,
        "corpus": "pybot_tts_v2_recorded/extended (32 utterances: 16 SSB0502, 16 SSB1593)",
        "cv_protocol": "strict 32-fold leave-one-utterance-out (zero data leakage)",
        "elapsed_s": round(time.time() - t0, 2),
        "real_speaker_summary": real_summary,
        "real_effect_sizes": real_effect_sizes,
        "real_vs_real_loo_cv": real_loo,
        "variants": variant_summary,
        "diphthong_trajectory_probe": diphthong_probe,
        "formant_speed_probe": formant_speed_probe,
        "full_corpus_profiles": {
            spk: {k: v for k, v in prof.items() if k != "final_trajectories"}
            for spk, prof in fold_full["speakers"].items()
        },
        "n_tracked_finals": len(fold_full["pooled_trajectories"]),
        "per_utterance_folds": per_variant_records,
    }

    def _sanitize(obj):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    (EXP_DIR / "results.json").write_text(json.dumps(_sanitize(out), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run_experiment(export_samples=True)
    print(f"Completed 32-fold LOO-CV in {res['elapsed_s']}s")
    print("\n--- Real-vs-Real LOO-CV Speaker Classification ---")
    for k, v in res["real_vs_real_loo_cv"].items():
        print(f"  {k:16s}: acc={v['accuracy']*100:5.1f}% ({v['correct']}/{v['total']})  AUC={v['roc_auc']:.3f}")
    print("\n--- Variant Summary (LOO-CV) ---")
    for var, info in res["variants"].items():
        gap = info["acoustic_gap_vs_real"]
        print(f"  [{var}]")
        print(
            f"    Gap vs real: tilt_MAE={gap['tilt_mae_db_oct']:.2f} dB/oct | "
            f"LTAS_RMSE={gap['ltas_env_rmse_db']:.2f} dB | F0_err={gap['f0_med_err_st']:.2f} st | "
            f"IQR_err={gap['f0_iqr_err_st']:.2f} st | HNR_err={gap['hnr_err_db']:.2f} dB | "
            f"DTW_MCD={gap['dtw_mcd_db']:.2f} dB | Mel_LSD={gap['dtw_mel_lsd_db']:.2f} dB | MFCC_cos={gap['mfcc_cos_sim']:.3f}"
        )
        if "inter_speaker_separation" in info:
            sep = info["inter_speaker_separation"]
            mat = info["synth_to_real_match_acc"]
            print(
                f"    Separation:  tilt_gap={sep['tilt_gap_db_oct']:.2f} | LTAS_L2={sep['ltas_l2_db']:.2f} dB | "
                f"F0_gap={sep['f0_gap_st']:.2f} st | HNR_gap={sep['hnr_gap_db']:.2f} dB | "
                f"MFCC_cos_dist={sep['mfcc_cos_dist']:.4f} | wave_max={sep['wave_max_diff']:.4f}"
            )
            print(
                f"    Match Acc:   tilt={mat['tilt']*100:5.1f}% | ltas={mat['ltas_env']*100:5.1f}% | "
                f"f0={mat['f0']*100:5.1f}% | hnr={mat['hnr']*100:5.1f}% | mcd={mat['mcd']*100:5.1f}% | "
                f"combined={mat['combined_timbre']*100:5.1f}%"
            )
    print("\n--- Diphthong / Vowel Trajectory Probe ---")
    for fk, pv in res["diphthong_trajectory_probe"].items():
        print(f"  /{fk}/: real_F2_span={pv.get('real_f2_span_hz')} Hz | v4_F2_span={pv.get('v4_f2_span_hz')} Hz | v5_F2_span={pv.get('v5_f2_span_hz')} Hz")
