"""Gate 1 (Analysis-Resynthesis) & Gate 2 (Zero-Leakage Few-Shot Cloning + Male 115 Hz Stress Test)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_v5_eval import metrics as eval_m  # noqa: E402
from exp_v5_timbre.features import extract_ltas, extract_mfcc  # noqa: E402
from exp_v6_vocoder import EXP_DIR, EXTENDED_INDEX, SAMPLES_DIR, SR  # noqa: E402
from exp_v6_vocoder.cloner import (  # noqa: E402
    adapt_features,
    extract_fewshot_profile,
    make_synthetic_male_profile,
)
from exp_v6_vocoder.vocoder import VocoderFeatures, analyze, read_wav_mono, synthesize, write_wav_mono  # noqa: E402


def _extract_clf_vec(wav: np.ndarray, sr: int = SR) -> dict[str, np.ndarray]:
    lt = extract_ltas(wav, sr=sr)
    fh = eval_m.compute_f0_hnr_metrics(wav, sr=sr)
    mf = extract_mfcc(wav, sr=sr)
    f0_med = fh["f0_median_hz"] if math.isfinite(fh["f0_median_hz"]) else 200.0
    f0_iqr = fh["f0_iqr_st"] if math.isfinite(fh["f0_iqr_st"]) else 3.0
    hnr = fh["hnr_db_median"] if math.isfinite(fh["hnr_db_median"]) else 12.0
    return {
        "tilt": float(lt["tilt_db_per_oct"]),
        "ltas_vec": lt["ltas_db_norm"],
        "f0_med": float(f0_med),
        "f0_iqr": float(f0_iqr),
        "hnr": float(hnr),
        "mfcc_mean": mf["mfcc_mean"][1:13],
        "f0_stats": np.array([f0_med, f0_iqr], dtype=np.float64),
        "timbre_4d": np.array([lt["tilt_db_per_oct"], f0_med, f0_iqr, hnr], dtype=np.float64),
        "all_features": np.concatenate([
            [lt["tilt_db_per_oct"], f0_med, f0_iqr, hnr],
            lt["ltas_db_norm"],
            mf["mfcc_mean"][1:13],
        ]),
    }


def _classify_against_train(
    query_vec: dict[str, np.ndarray],
    train_utts: list[dict],
    real_vecs: dict[str, dict[str, np.ndarray]],
    key: str,
) -> tuple[str, float, float]:
    """Standardized nearest-centroid classifier using ONLY training utterances."""
    x_tr = np.stack([real_vecs[u["id"]][key] for u in train_utts], axis=0)
    mu = np.mean(x_tr, axis=0)
    sd = np.maximum(np.std(x_tr, axis=0, ddof=1), 1e-6)
    c502 = np.mean([(real_vecs[u["id"]][key] - mu) / sd for u in train_utts if u["speaker"] == "SSB0502"], axis=0)
    c1593 = np.mean([(real_vecs[u["id"]][key] - mu) / sd for u in train_utts if u["speaker"] == "SSB1593"], axis=0)
    xq = (query_vec[key] - mu) / sd
    d502 = float(np.linalg.norm(xq - c502))
    d1593 = float(np.linalg.norm(xq - c1593))
    return ("SSB0502" if d502 < d1593 else "SSB1593"), d502, d1593


def _f0_contour_corr(fa: VocoderFeatures, fb: VocoderFeatures) -> float:
    m = fa.voiced & fb.voiced & (fa.f0_hz > 0) & (fb.f0_hz > 0)
    if np.sum(m) < 5:
        return 0.0
    la = np.log(fa.f0_hz[m])
    lb = np.log(fb.f0_hz[m])
    if np.std(la) < 1e-6 or np.std(lb) < 1e-6:
        return 0.0
    return float(np.corrcoef(la, lb)[0, 1])


def run_eval(export_samples: bool = True) -> dict:
    t0 = time.time()
    idx = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))
    utts = idx["utterances"]

    real_wavs: dict[str, np.ndarray] = {}
    real_feats: dict[str, VocoderFeatures] = {}
    real_eval: dict[str, dict] = {}
    real_clf: dict[str, dict] = {}

    gate1_records = {"pd_on": [], "pd_off": []}

    # --- GATE 1: Analysis-Resynthesis Transparency (n=32) ---
    for i, u in enumerate(utts):
        uid = u["id"]
        x, sr = read_wav_mono(u["path"])
        if sr != SR:
            raise ValueError(f"Expected {SR} Hz, got {sr} for {uid}")
        x_act = x[u["active_start"] : u["active_stop"]]
        dur_s = len(x_act) / SR
        real_wavs[uid] = x_act

        ta0 = time.time()
        feats = analyze(x_act, sr=SR)
        ta1 = time.time()
        real_feats[uid] = feats
        real_eval[uid] = eval_m.evaluate_audio(x_act, sr=SR)
        real_clf[uid] = _extract_clf_vec(x_act, sr=SR)

        for pd_flag, bucket in ((True, "pd_on"), (False, "pd_off")):
            ts0 = time.time()
            y = synthesize(feats, seed=72 + i, phase_dispersion=pd_flag)
            ts1 = time.time()

            sanity = eval_m.check_wav_sanity(y, sr=SR)
            sd = eval_m.compute_spectral_distances(x_act, y, sr=SR)
            ev_y = eval_m.evaluate_audio(y, sr=SR)
            ev_r = real_eval[uid]
            f0_err_st = abs(12.0 * math.log2(ev_y["f0_hnr"]["f0_median_hz"] / ev_r["f0_hnr"]["f0_median_hz"]))
            gate1_records[bucket].append({
                "id": uid,
                "speaker": u["speaker"],
                "sanity_ok": bool(sanity["ok"]),
                "mel_lsd_db": float(sd["mel_lsd_db"]),
                "mcd_db": float(sd["mcd_db"]),
                "mcd_cmn_db": float(sd["mcd_cmn_db"]),
                "tilt_err_db_oct": abs(ev_y["ltas"]["tilt_db_oct"] - ev_r["ltas"]["tilt_db_oct"]),
                "f0_err_st": float(f0_err_st),
                "hnr_err_db": abs(ev_y["f0_hnr"]["hnr_db_median"] - ev_r["f0_hnr"]["hnr_db_median"]),
                "df1_ratio": ev_y["formants"]["df1_dt_median_hz_s"] / max(1.0, ev_r["formants"]["df1_dt_median_hz_s"]),
                "df2_ratio": ev_y["formants"]["df2_dt_median_hz_s"] / max(1.0, ev_r["formants"]["df2_dt_median_hz_s"]),
                "rtf_ana": (ta1 - ta0) / dur_s,
                "rtf_syn": (ts1 - ts0) / dur_s,
                "rtf_tot": ((ta1 - ta0) + (ts1 - ts0)) / dur_s,
            })

            if export_samples and uid in ("SSB05020005", "SSB15930031"):
                SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
                if pd_flag:
                    write_wav_mono(SAMPLES_DIR / f"{uid}_real.wav", x_act, sr=SR)
                    write_wav_mono(SAMPLES_DIR / f"{uid}_gate1_resyn.wav", y, sr=SR)
                else:
                    write_wav_mono(SAMPLES_DIR / f"{uid}_gate1_nopd.wav", y, sr=SR)

    gate1_summary = {}
    for bucket, recs in gate1_records.items():
        gate1_summary[bucket] = {
            "n_utterances": len(recs),
            "sanity_ok": sum(1 for r in recs if r["sanity_ok"]),
            "mel_lsd_db_mean": round(float(np.mean([r["mel_lsd_db"] for r in recs])), 3),
            "mel_lsd_db_median": round(float(np.median([r["mel_lsd_db"] for r in recs])), 3),
            "mel_lsd_db_max": round(float(np.max([r["mel_lsd_db"] for r in recs])), 3),
            "mcd_db_mean": round(float(np.mean([r["mcd_db"] for r in recs])), 3),
            "mcd_cmn_db_mean": round(float(np.mean([r["mcd_cmn_db"] for r in recs])), 3),
            "tilt_mae_db_oct": round(float(np.mean([r["tilt_err_db_oct"] for r in recs])), 3),
            "f0_err_st": round(float(np.mean([r["f0_err_st"] for r in recs])), 4),
            "hnr_err_db": round(float(np.mean([r["hnr_err_db"] for r in recs])), 3),
            "df1_speed_ratio": round(float(np.median([r["df1_ratio"] for r in recs])), 3),
            "df2_speed_ratio": round(float(np.median([r["df2_ratio"] for r in recs])), 3),
            "rtf_ana": round(float(np.mean([r["rtf_ana"] for r in recs])), 4),
            "rtf_syn": round(float(np.mean([r["rtf_syn"] for r in recs])), 4),
            "rtf_tot": round(float(np.mean([r["rtf_tot"] for r in recs])), 4),
        }

    # --- GATE 2: Zero-Leakage LOO-CV Few-Shot Voice Cloning (N=3 and N=1) ---
    by_spk = {
        "SSB0502": [u for u in utts if u["speaker"] == "SSB0502"],
        "SSB1593": [u for u in utts if u["speaker"] == "SSB1593"],
    }
    gate2_modes = {
        "clone_3shot_full": [],
        "clone_1shot_full": [],
        "clone_3shot_f0_only": [],
        "unconverted_donor": [],
    }

    for fold_i, held in enumerate(utts):
        uid = held["id"]
        tgt_spk = held["speaker"]
        src_spk = "SSB1593" if tgt_spk == "SSB0502" else "SSB0502"
        train_utts = [u for j, u in enumerate(utts) if j != fold_i]
        # Target speaker's remaining utterances (strictly excluding held!)
        tgt_pool = [u for u in by_spk[tgt_spk] if u["id"] != uid]
        assert all(u["id"] != uid for u in tgt_pool)

        # Deterministic selection of 3-shot (~9s) and 1-shot (~3s) from tgt_pool
        fs3_utts = [tgt_pool[(fold_i + k) % len(tgt_pool)] for k in range(3)]
        fs1_utts = [tgt_pool[fold_i % len(tgt_pool)]]
        # Source donor utterance from opposite speaker
        src_u = by_spk[src_spk][fold_i % len(by_spk[src_spk])]
        src_uid = src_u["id"]

        prof_src = extract_fewshot_profile([(real_wavs[src_uid], real_feats[src_uid])], sr=SR, name=src_spk)
        prof_tgt3 = extract_fewshot_profile([(real_wavs[u["id"]], real_feats[u["id"]]) for u in fs3_utts], sr=SR, name=tgt_spk)
        prof_tgt1 = extract_fewshot_profile([(real_wavs[u["id"]], real_feats[u["id"]]) for u in fs1_utts], sr=SR, name=tgt_spk)

        tgt_vec = real_clf[uid]
        src_vec = real_clf[src_uid]

        # Record unconverted donor baseline against held-out target & classifier
        for mode_name, prof_tgt, use_ltas, use_bap in (
            ("clone_3shot_full", prof_tgt3, True, True),
            ("clone_1shot_full", prof_tgt1, True, True),
            ("clone_3shot_f0_only", prof_tgt3, False, False),
        ):
            t_c0 = time.time()
            adapted = adapt_features(
                real_feats[src_uid],
                prof_src,
                prof_tgt,
                vtln_alpha=1.0,  # same-gender similar vocal tract; damped VTLN=1.0
                use_ltas_transfer=use_ltas,
                use_bap_transfer=use_bap,
            )
            y_conv = synthesize(adapted, seed=72 + fold_i, phase_dispersion=True)
            t_c1 = time.time()
            conv_vec = _extract_clf_vec(y_conv, sr=SR)

            # Content preservation vs source donor
            sd_content = eval_m.compute_spectral_distances(real_wavs[src_uid], y_conv, sr=SR)
            f0_corr = _f0_contour_corr(real_feats[src_uid], adapted)

            # LOO-CV speaker classifier decisions (trained ONLY on train_utts!)
            pred_timbre, _, _ = _classify_against_train(conv_vec, train_utts, real_clf, "timbre_4d")
            pred_ltas, _, _ = _classify_against_train(conv_vec, train_utts, real_clf, "ltas_vec")
            pred_f0, _, _ = _classify_against_train(conv_vec, train_utts, real_clf, "f0_stats")
            pred_all, _, _ = _classify_against_train(conv_vec, train_utts, real_clf, "all_features")

            # Acoustic gaps to held-out real target utterance `uid`
            tilt_err = abs(conv_vec["tilt"] - tgt_vec["tilt"])
            ltas_rmse = float(np.sqrt(np.mean((conv_vec["ltas_vec"] - tgt_vec["ltas_vec"]) ** 2)))
            f0_err_st = abs(12.0 * math.log2(conv_vec["f0_med"] / tgt_vec["f0_med"]))
            iqr_err_st = abs(conv_vec["f0_iqr"] - tgt_vec["f0_iqr"])
            hnr_err_db = abs(conv_vec["hnr"] - tgt_vec["hnr"])
            mfcc_cos = float(
                np.dot(conv_vec["mfcc_mean"], tgt_vec["mfcc_mean"])
                / max(1e-12, np.linalg.norm(conv_vec["mfcc_mean"]) * np.linalg.norm(tgt_vec["mfcc_mean"]))
            )

            # Compare converted vs unconverted donor distance to target `uid`
            src_tilt_err = abs(src_vec["tilt"] - tgt_vec["tilt"])
            src_ltas_rmse = float(np.sqrt(np.mean((src_vec["ltas_vec"] - tgt_vec["ltas_vec"]) ** 2)))
            src_f0_err = abs(12.0 * math.log2(src_vec["f0_med"] / tgt_vec["f0_med"]))

            gate2_modes[mode_name].append({
                "held_id": uid,
                "target_speaker": tgt_spk,
                "source_id": src_uid,
                "fewshot_ids": [u["id"] for u in (fs3_utts if prof_tgt is prof_tgt3 else fs1_utts)],
                "fewshot_voiced_s": round(prof_tgt.total_voiced_s, 2),
                "clf_target_timbre4d": bool(pred_timbre == tgt_spk),
                "clf_target_ltas": bool(pred_ltas == tgt_spk),
                "clf_target_f0": bool(pred_f0 == tgt_spk),
                "clf_target_all": bool(pred_all == tgt_spk),
                "closer_than_donor_ltas": bool(ltas_rmse < src_ltas_rmse),
                "closer_than_donor_f0": bool(f0_err_st < src_f0_err),
                "closer_than_donor_tilt": bool(tilt_err < src_tilt_err),
                "tilt_err_db_oct": float(tilt_err),
                "ltas_env_rmse_db": float(ltas_rmse),
                "f0_err_st": float(f0_err_st),
                "iqr_err_st": float(iqr_err_st),
                "hnr_err_db": float(hnr_err_db),
                "mfcc_cos_sim": float(mfcc_cos),
                "content_mcd_cmn_db": float(sd_content["mcd_cmn_db"]),
                "content_mel_lsd_db": float(sd_content["mel_lsd_db"]),
                "f0_contour_corr": float(f0_corr),
                "rtf_conv": (t_c1 - t_c0) / (len(real_wavs[src_uid]) / SR),
            })

            if export_samples and src_uid in ("SSB05020005", "SSB15930031"):
                if mode_name in ("clone_3shot_full", "clone_1shot_full"):
                    tag = "3shot" if mode_name == "clone_3shot_full" else "1shot"
                    write_wav_mono(SAMPLES_DIR / f"{src_uid}_to_{tgt_spk}_{tag}.wav", y_conv, sr=SR)

        # Unconverted donor baseline
        pred_t0, _, _ = _classify_against_train(src_vec, train_utts, real_clf, "timbre_4d")
        pred_l0, _, _ = _classify_against_train(src_vec, train_utts, real_clf, "ltas_vec")
        pred_f0_0, _, _ = _classify_against_train(src_vec, train_utts, real_clf, "f0_stats")
        pred_a0, _, _ = _classify_against_train(src_vec, train_utts, real_clf, "all_features")
        gate2_modes["unconverted_donor"].append({
            "held_id": uid,
            "target_speaker": tgt_spk,
            "clf_target_timbre4d": bool(pred_t0 == tgt_spk),
            "clf_target_ltas": bool(pred_l0 == tgt_spk),
            "clf_target_f0": bool(pred_f0_0 == tgt_spk),
            "clf_target_all": bool(pred_a0 == tgt_spk),
            "tilt_err_db_oct": abs(src_vec["tilt"] - tgt_vec["tilt"]),
            "ltas_env_rmse_db": float(np.sqrt(np.mean((src_vec["ltas_vec"] - tgt_vec["ltas_vec"]) ** 2))),
            "f0_err_st": abs(12.0 * math.log2(src_vec["f0_med"] / tgt_vec["f0_med"])),
            "iqr_err_st": abs(src_vec["f0_iqr"] - tgt_vec["f0_iqr"]),
            "hnr_err_db": abs(src_vec["hnr"] - tgt_vec["hnr"]),
            "mfcc_cos_sim": float(
                np.dot(src_vec["mfcc_mean"], tgt_vec["mfcc_mean"])
                / max(1e-12, np.linalg.norm(src_vec["mfcc_mean"]) * np.linalg.norm(tgt_vec["mfcc_mean"]))
            ),
        })

    gate2_summary = {}
    for mname, recs in gate2_modes.items():
        row = {
            "n_folds": len(recs),
            "target_clf_acc_timbre4d": round(float(np.mean([r["clf_target_timbre4d"] for r in recs])), 4),
            "target_clf_acc_ltas": round(float(np.mean([r["clf_target_ltas"] for r in recs])), 4),
            "target_clf_acc_f0": round(float(np.mean([r["clf_target_f0"] for r in recs])), 4),
            "target_clf_acc_all": round(float(np.mean([r["clf_target_all"] for r in recs])), 4),
            "tilt_err_db_oct": round(float(np.mean([r["tilt_err_db_oct"] for r in recs])), 3),
            "ltas_env_rmse_db": round(float(np.mean([r["ltas_env_rmse_db"] for r in recs])), 3),
            "f0_err_st": round(float(np.mean([r["f0_err_st"] for r in recs])), 3),
            "iqr_err_st": round(float(np.mean([r["iqr_err_st"] for r in recs])), 3),
            "hnr_err_db": round(float(np.mean([r["hnr_err_db"] for r in recs])), 3),
            "mfcc_cos_sim": round(float(np.mean([r["mfcc_cos_sim"] for r in recs])), 4),
        }
        if mname != "unconverted_donor":
            row["fewshot_voiced_s_mean"] = round(float(np.mean([r["fewshot_voiced_s"] for r in recs])), 2)
            row["closer_than_donor_ltas"] = round(float(np.mean([r["closer_than_donor_ltas"] for r in recs])), 4)
            row["closer_than_donor_f0"] = round(float(np.mean([r["closer_than_donor_f0"] for r in recs])), 4)
            row["closer_than_donor_tilt"] = round(float(np.mean([r["closer_than_donor_tilt"] for r in recs])), 4)
            row["content_mcd_cmn_db"] = round(float(np.mean([r["content_mcd_cmn_db"] for r in recs])), 3)
            row["content_mel_lsd_db"] = round(float(np.mean([r["content_mel_lsd_db"] for r in recs])), 3)
            row["f0_contour_corr"] = round(float(np.mean([r["f0_contour_corr"] for r in recs])), 4)
            row["rtf_conv"] = round(float(np.mean([r["rtf_conv"] for r in recs])), 4)
        gate2_summary[mname] = row

    # --- Synthetic Cross-Gender Male 115 Hz Stress Test ---
    male_stress = {}
    for suid in ("SSB05020005", "SSB15930031"):
        prof_s = extract_fewshot_profile([(real_wavs[suid], real_feats[suid])], sr=SR, name=suid)
        prof_m = make_synthetic_male_profile(prof_s, f0_hz=115.0, iqr_st=2.4, tilt_db_per_oct=-4.5, vtln_alpha=0.84)
        ev_src = real_eval[suid]
        for alpha_tag, alpha_val in (("vtln_084", 0.84), ("vtln_100", 1.00)):
            ad_m = adapt_features(real_feats[suid], prof_s, prof_m, vtln_alpha=alpha_val, use_ltas_transfer=True, use_bap_transfer=True)
            y_m = synthesize(ad_m, seed=72, phase_dispersion=True)
            san = eval_m.check_wav_sanity(y_m, sr=SR)
            ev_m = eval_m.evaluate_audio(y_m, sr=SR)
            sd_m = eval_m.compute_spectral_distances(real_wavs[suid], y_m, sr=SR)
            male_stress[f"{suid}_{alpha_tag}"] = {
                "sanity_ok": bool(san["ok"]),
                "src_f0_hz": round(ev_src["f0_hnr"]["f0_median_hz"], 1),
                "out_f0_hz": round(ev_m["f0_hnr"]["f0_median_hz"], 1),
                "out_f0_iqr_st": round(ev_m["f0_hnr"]["f0_iqr_st"], 2),
                "out_tilt_db_oct": round(ev_m["ltas"]["tilt_db_oct"], 2),
                "out_hnr_db": round(ev_m["f0_hnr"]["hnr_db_median"], 2),
                "src_f1_f2_hz": [round(ev_src["formants"]["f1_median_hz"], 1), round(ev_src["formants"]["f2_median_hz"], 1)],
                "out_f1_f2_hz": [round(ev_m["formants"]["f1_median_hz"], 1), round(ev_m["formants"]["f2_median_hz"], 1)],
                "f1_shift_pct": round(100.0 * (ev_m["formants"]["f1_median_hz"] / ev_src["formants"]["f1_median_hz"] - 1.0), 1),
                "f2_shift_pct": round(100.0 * (ev_m["formants"]["f2_median_hz"] / ev_src["formants"]["f2_median_hz"] - 1.0), 1),
                "mcd_cmn_vs_src_db": round(sd_m["mcd_cmn_db"], 2),
                "f0_contour_corr": round(_f0_contour_corr(real_feats[suid], ad_m), 4),
            }
            if export_samples and alpha_tag == "vtln_084":
                write_wav_mono(SAMPLES_DIR / f"{suid}_to_male115_vtln084.wav", y_m, sr=SR)

    out = {
        "schema": 1,
        "corpus": "pybot_tts_v2_recorded/extended (32 utterances: 16 SSB0502, 16 SSB1593)",
        "elapsed_s": round(time.time() - t0, 2),
        "gate1_resynthesis": gate1_summary,
        "gate2_fewshot_cloning_loocv": gate2_summary,
        "male_115hz_stress_test": male_stress,
        "gate1_per_utterance": gate1_records["pd_on"],
        "gate2_3shot_folds": gate2_modes["clone_3shot_full"],
    }
    (EXP_DIR / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run_eval(export_samples=True)
    print(f"Completed Phase 2 Vocoder Eval in {res['elapsed_s']}s")
    print("\n=== GATE 1: Analysis-Resynthesis Transparency (n=32) ===")
    for k, v in res["gate1_resynthesis"].items():
        print(f"  [{k}] sanity={v['sanity_ok']}/{v['n_utterances']} | Mel-LSD={v['mel_lsd_db_mean']:.3f} dB (med {v['mel_lsd_db_median']:.3f}, max {v['mel_lsd_db_max']:.3f}) | MCD={v['mcd_db_mean']:.2f} (CMVN {v['mcd_cmn_db_mean']:.2f}) | tilt_MAE={v['tilt_mae_db_oct']:.2f} | F0_err={v['f0_err_st']:.3f} st | HNR_err={v['hnr_err_db']:.2f} dB | dF1/dF2_ratio={v['df1_speed_ratio']}/{v['df2_speed_ratio']} | RTF_tot={v['rtf_tot']:.4f} (ana={v['rtf_ana']:.4f}, syn={v['rtf_syn']:.4f})")
    print("\n=== GATE 2: Few-Shot Cross-Speaker Cloning (32-fold LOO-CV, Zero Leakage) ===")
    for k, v in res["gate2_fewshot_cloning_loocv"].items():
        extra = ""
        if "content_mcd_cmn_db" in v:
            extra = f" | closer(ltas={v['closer_than_donor_ltas']*100:.1f}%, f0={v['closer_than_donor_f0']*100:.1f}%) | content_MCD_CMVN={v['content_mcd_cmn_db']:.2f}, F0_corr={v['f0_contour_corr']:.3f}, RTF_conv={v['rtf_conv']:.4f}"
        print(f"  [{k:20s}] clf_target(4D={v['target_clf_acc_timbre4d']*100:5.1f}%, LTAS={v['target_clf_acc_ltas']*100:5.1f}%, F0={v['target_clf_acc_f0']*100:5.1f}%, All={v['target_clf_acc_all']*100:5.1f}%) | F0_err={v['f0_err_st']:.2f} st | LTAS_RMSE={v['ltas_env_rmse_db']:.2f} dB | tilt_err={v['tilt_err_db_oct']:.2f} | HNR_err={v['hnr_err_db']:.2f}{extra}")
    print("\n=== Synthetic Cross-Gender Male 115 Hz Stress Test ===")
    for k, v in res["male_115hz_stress_test"].items():
        print(f"  [{k}] ok={v['sanity_ok']} | F0 {v['src_f0_hz']}->{v['out_f0_hz']} Hz (IQR {v['out_f0_iqr_st']} st) | tilt={v['out_tilt_db_oct']} | HNR={v['out_hnr_db']} | F1/F2 {v['src_f1_f2_hz']}->{v['out_f1_f2_hz']} ({v['f1_shift_pct']:+.1f}%/{v['f2_shift_pct']:+.1f}%) | F0_corr={v['f0_contour_corr']}")
