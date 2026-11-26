"""Zero-leakage 32-fold LOO-CV & cross-gender ablation for Phase-4 few-shot timbre cloning."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_eval import metrics as eval_m
from exp_v6_vocoder import cloner as v6_cloner, eval_vocoder as v6_eval, vocoder as v6_voc
from exp_v8_timbre_clone import EXTENDED_INDEX, SAMPLES_DIR, SR
from exp_v8_timbre_clone.clarity_metrics import compute_clarity_metrics
from exp_v8_timbre_clone.timbre_cloner_v2 import (
    TimbreProfileV2,
    adapt_features_ridge,
    adapt_features_v2,
    extract_v2_profile,
    make_synthetic_male_v2_profile,
)


def _peak_limit(y: np.ndarray) -> np.ndarray:
    pk = float(np.max(np.abs(y))) if y.size else 0.0
    return y * (0.90 / pk) if pk > 0.95 else y


def _eval_mode_fold(
    mode_name: str,
    y_conv: np.ndarray,
    src_wav: np.ndarray,
    src_feats: v6_voc.VocoderFeatures,
    adapted_feats: v6_voc.VocoderFeatures,
    tgt_spk: str,
    tgt_vec: dict,
    src_vec: dict,
    train_utts: list[dict],
    real_clf: dict,
    clarity_src: dict,
    conv_wall_s: float,
    extra: dict,
) -> dict:
    conv_vec = v6_eval._extract_clf_vec(y_conv, sr=SR)
    clarity_conv = compute_clarity_metrics(y_conv, sr=SR)
    sd_content = eval_m.compute_spectral_distances(src_wav, y_conv, sr=SR)
    f0_corr = v6_eval._f0_contour_corr(src_feats, adapted_feats)

    pred_timbre, _, _ = v6_eval._classify_against_train(conv_vec, train_utts, real_clf, "timbre_4d")
    pred_ltas, _, _ = v6_eval._classify_against_train(conv_vec, train_utts, real_clf, "ltas_vec")
    pred_all, _, _ = v6_eval._classify_against_train(conv_vec, train_utts, real_clf, "all_features")

    tilt_err = abs(conv_vec["tilt"] - tgt_vec["tilt"])
    ltas_rmse = float(np.sqrt(np.mean((conv_vec["ltas_vec"] - tgt_vec["ltas_vec"]) ** 2)))
    f0_err_st = abs(12.0 * math.log2(conv_vec["f0_med"] / tgt_vec["f0_med"]))
    hnr_err_db = abs(conv_vec["hnr"] - tgt_vec["hnr"])
    mfcc_cos = float(
        np.dot(conv_vec["mfcc_mean"], tgt_vec["mfcc_mean"])
        / max(1e-12, np.linalg.norm(conv_vec["mfcc_mean"]) * np.linalg.norm(tgt_vec["mfcc_mean"]))
    )

    return {
        "mode": mode_name,
        "clf_target_timbre4d": bool(pred_timbre == tgt_spk),
        "clf_target_ltas": bool(pred_ltas == tgt_spk),
        "clf_target_all": bool(pred_all == tgt_spk),
        "tilt_err_db_oct": float(tilt_err),
        "ltas_env_rmse_db": float(ltas_rmse),
        "f0_err_st": float(f0_err_st),
        "hnr_err_db": float(hnr_err_db),
        "mfcc_cos_sim": float(mfcc_cos),
        "content_mel_lsd_db": float(sd_content["mel_lsd_db"]),
        "content_mcd_cmn_db": float(sd_content["mcd_cmn_db"]),
        "f0_contour_corr": float(f0_corr),
        "formant_prom_delta_db": float(clarity_conv["formant_prominence_db"] - clarity_src["formant_prominence_db"]),
        "consonant_hf_delta_db": float(clarity_conv["consonant_hf_ratio_db"] - clarity_src["consonant_hf_ratio_db"]),
        "rtf_conv": conv_wall_s / (len(src_wav) / SR),
        **extra,
    }


def _summarize_rows(rows: list[dict]) -> dict:
    out = {
        "n_folds": len(rows),
        "target_clf_acc_timbre4d": round(float(np.mean([r["clf_target_timbre4d"] for r in rows])), 4),
        "target_clf_acc_ltas": round(float(np.mean([r["clf_target_ltas"] for r in rows])), 4),
        "target_clf_acc_all": round(float(np.mean([r["clf_target_all"] for r in rows])), 4),
        "tilt_err_db_oct": round(float(np.mean([r["tilt_err_db_oct"] for r in rows])), 3),
        "ltas_env_rmse_db": round(float(np.mean([r["ltas_env_rmse_db"] for r in rows])), 3),
        "f0_err_st": round(float(np.mean([r["f0_err_st"] for r in rows])), 3),
        "hnr_err_db": round(float(np.mean([r["hnr_err_db"] for r in rows])), 3),
        "mfcc_cos_sim": round(float(np.mean([r["mfcc_cos_sim"] for r in rows])), 4),
        "content_mel_lsd_db": round(float(np.mean([r["content_mel_lsd_db"] for r in rows])), 3),
        "content_mcd_cmn_db": round(float(np.mean([r["content_mcd_cmn_db"] for r in rows])), 3),
        "f0_contour_corr": round(float(np.mean([r["f0_contour_corr"] for r in rows])), 4),
        "formant_prom_delta_db": round(float(np.mean([r["formant_prom_delta_db"] for r in rows])), 3),
        "consonant_hf_delta_db": round(float(np.mean([r["consonant_hf_delta_db"] for r in rows])), 3),
        "rtf_conv_mean": round(float(np.mean([r["rtf_conv"] for r in rows])), 4),
    }
    if "vtln_alpha" in rows[0]:
        out["auto_vtln_mean"] = round(float(np.mean([r["vtln_alpha"] for r in rows])), 4)
        out["auto_vtln_non1_count"] = int(sum(1 for r in rows if abs(r["vtln_alpha"] - 1.0) > 1e-4))
    return out


def run_ablation(export_samples: bool = True) -> dict:
    idx = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))
    utts = idx["utterances"]

    real_wavs: dict[str, np.ndarray] = {}
    real_feats: dict[str, v6_voc.VocoderFeatures] = {}
    real_clf: dict[str, dict] = {}
    real_clarity: dict[str, dict] = {}
    by_spk: dict[str, list[dict]] = {"SSB0502": [], "SSB1593": []}

    for u in utts:
        uid = u["id"]
        x, sr = v6_voc.read_wav_mono(u["path"])
        if sr != SR:
            raise ValueError(f"Expected {SR} Hz, got {sr}")
        x_act = x[u["active_start"] : u["active_stop"]]
        real_wavs[uid] = x_act
        real_feats[uid] = v6_voc.analyze(x_act, sr=SR)
        real_clf[uid] = v6_eval._extract_clf_vec(x_act, sr=SR)
        real_clarity[uid] = compute_clarity_metrics(x_act, sr=SR)
        by_spk[u["speaker"]].append(u)

    modes_1shot = {m: [] for m in ("v6_coupled", "v8_decoupled", "v8_ridge_cepstral")}
    modes_3shot = {m: [] for m in ("v6_coupled", "v8_decoupled", "v8_ridge_cepstral")}
    unconverted_rows = []

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Strict 32-fold LOO-CV (zero leakage)
    for fold_i, held in enumerate(utts):
        uid = held["id"]
        tgt_spk = held["speaker"]
        src_spk = "SSB1593" if tgt_spk == "SSB0502" else "SSB0502"
        train_utts = [u for j, u in enumerate(utts) if j != fold_i]
        tgt_pool = [u for u in by_spk[tgt_spk] if u["id"] != uid]
        assert all(u["id"] != uid for u in tgt_pool)

        fs3_utts = [tgt_pool[(fold_i + k) % len(tgt_pool)] for k in range(3)]
        fs1_utts = [tgt_pool[fold_i % len(tgt_pool)]]
        src_u = by_spk[src_spk][fold_i % len(by_spk[src_spk])]
        src_uid = src_u["id"]

        prof_src_v2 = extract_v2_profile([(real_wavs[src_uid], real_feats[src_uid])], sr=SR, name=src_spk)
        prof_tgt3_v2 = extract_v2_profile([(real_wavs[u["id"]], real_feats[u["id"]]) for u in fs3_utts], sr=SR, name=tgt_spk)
        prof_tgt1_v2 = extract_v2_profile([(real_wavs[u["id"]], real_feats[u["id"]]) for u in fs1_utts], sr=SR, name=tgt_spk)

        prof_src_v6 = prof_src_v2.to_v6_profile()
        prof_tgt3_v6 = prof_tgt3_v2.to_v6_profile()
        prof_tgt1_v6 = prof_tgt1_v2.to_v6_profile()

        tgt_vec = real_clf[uid]
        src_vec = real_clf[src_uid]

        # Unconverted donor baseline
        pred_t0, _, _ = v6_eval._classify_against_train(src_vec, train_utts, real_clf, "timbre_4d")
        pred_l0, _, _ = v6_eval._classify_against_train(src_vec, train_utts, real_clf, "ltas_vec")
        pred_a0, _, _ = v6_eval._classify_against_train(src_vec, train_utts, real_clf, "all_features")
        unconverted_rows.append({
            "mode": "unconverted_donor",
            "clf_target_timbre4d": bool(pred_t0 == tgt_spk),
            "clf_target_ltas": bool(pred_l0 == tgt_spk),
            "clf_target_all": bool(pred_a0 == tgt_spk),
            "tilt_err_db_oct": abs(src_vec["tilt"] - tgt_vec["tilt"]),
            "ltas_env_rmse_db": float(np.sqrt(np.mean((src_vec["ltas_vec"] - tgt_vec["ltas_vec"]) ** 2))),
            "f0_err_st": abs(12.0 * math.log2(src_vec["f0_med"] / tgt_vec["f0_med"])),
            "hnr_err_db": abs(src_vec["hnr"] - tgt_vec["hnr"]),
            "mfcc_cos_sim": float(
                np.dot(src_vec["mfcc_mean"], tgt_vec["mfcc_mean"])
                / max(1e-12, np.linalg.norm(src_vec["mfcc_mean"]) * np.linalg.norm(tgt_vec["mfcc_mean"]))
            ),
            "content_mel_lsd_db": 0.0,
            "content_mcd_cmn_db": 0.0,
            "f0_contour_corr": 1.0,
            "formant_prom_delta_db": 0.0,
            "consonant_hf_delta_db": 0.0,
            "rtf_conv": 0.0,
        })

        for shot_name, bucket, pt_v2, pt_v6 in (
            ("1shot", modes_1shot, prof_tgt1_v2, prof_tgt1_v6),
            ("3shot", modes_3shot, prof_tgt3_v2, prof_tgt3_v6),
        ):
            # 1) v6_coupled (using v6's auto-VTLN)
            t0 = time.perf_counter()
            ad_v6 = v6_cloner.adapt_features(real_feats[src_uid], prof_src_v6, pt_v6, vtln_alpha=None)
            y_v6 = _peak_limit(v6_voc.synthesize(ad_v6, seed=72 + fold_i, phase_dispersion=True))
            dt_v6 = time.perf_counter() - t0
            bucket["v6_coupled"].append(
                _eval_mode_fold(
                    "v6_coupled", y_v6, real_wavs[src_uid], real_feats[src_uid], ad_v6,
                    tgt_spk, tgt_vec, src_vec, train_utts, real_clf, real_clarity[src_uid], dt_v6, {},
                )
            )

            # 2) v8_decoupled (V/UV decoupled + sub-fundamental ramp + data-driven auto-VTLN)
            t0 = time.perf_counter()
            ad_v2, meta_v2 = adapt_features_v2(real_feats[src_uid], prof_src_v2, pt_v2, vtln_alpha=None)
            y_v2 = _peak_limit(v6_voc.synthesize(ad_v2, seed=72 + fold_i, phase_dispersion=True))
            dt_v2 = time.perf_counter() - t0
            bucket["v8_decoupled"].append(
                _eval_mode_fold(
                    "v8_decoupled", y_v2, real_wavs[src_uid], real_feats[src_uid], ad_v2,
                    tgt_spk, tgt_vec, src_vec, train_utts, real_clf, real_clarity[src_uid], dt_v2,
                    {"vtln_alpha": meta_v2["alpha"]},
                )
            )

            # 3) v8_ridge_cepstral (sklearn Ridge affine map over cepstral quantiles)
            t0 = time.perf_counter()
            ad_rg, _ = adapt_features_ridge(real_feats[src_uid], prof_src_v2, pt_v2, alpha_reg=50.0, blend=0.65)
            y_rg = _peak_limit(v6_voc.synthesize(ad_rg, seed=72 + fold_i, phase_dispersion=True))
            dt_rg = time.perf_counter() - t0
            bucket["v8_ridge_cepstral"].append(
                _eval_mode_fold(
                    "v8_ridge_cepstral", y_rg, real_wavs[src_uid], real_feats[src_uid], ad_rg,
                    tgt_spk, tgt_vec, src_vec, train_utts, real_clf, real_clarity[src_uid], dt_rg, {},
                )
            )

            if export_samples and shot_name == "3shot" and src_uid in ("SSB05020005", "SSB15930031"):
                v6_voc.write_wav_mono(SAMPLES_DIR / f"{src_uid}_to_{tgt_spk}_v6_coupled.wav", y_v6, sr=SR)
                v6_voc.write_wav_mono(SAMPLES_DIR / f"{src_uid}_to_{tgt_spk}_v8_decoupled.wav", y_v2, sr=SR)
                v6_voc.write_wav_mono(SAMPLES_DIR / f"{src_uid}_to_{tgt_spk}_v8_ridge.wav", y_rg, sr=SR)

    # Cross-gender 115 Hz male stress test (using auto-VTLN on v8_decoupled vs hardcoded 0.84 on v6_coupled)
    male_stress = []
    for src_uid in ("SSB05020005", "SSB15930031"):
        src_spk = "SSB0502" if src_uid.startswith("SSB0502") else "SSB1593"
        pool = [u for u in by_spk[src_spk] if u["id"] != src_uid][:3]
        prof_src_v2 = extract_v2_profile([(real_wavs[u["id"]], real_feats[u["id"]]) for u in pool], sr=SR, name=src_spk)
        prof_male_v2 = make_synthetic_male_v2_profile(prof_src_v2, f0_hz=115.0, tilt_db_per_oct=-4.5, vtln_alpha=0.84)

        ad_v6 = v6_cloner.adapt_features(real_feats[src_uid], prof_src_v2.to_v6_profile(), prof_male_v2.to_v6_profile(), vtln_alpha=0.84)
        y_v6 = _peak_limit(v6_voc.synthesize(ad_v6, seed=72, phase_dispersion=True))

        ad_v2, meta_v2 = adapt_features_v2(real_feats[src_uid], prof_src_v2, prof_male_v2, vtln_alpha=None)
        y_v2 = _peak_limit(v6_voc.synthesize(ad_v2, seed=72, phase_dispersion=True))

        ad_rg, _ = adapt_features_ridge(real_feats[src_uid], prof_src_v2, prof_male_v2, alpha_reg=50.0, blend=0.65)
        y_rg = _peak_limit(v6_voc.synthesize(ad_rg, seed=72, phase_dispersion=True))

        if export_samples:
            v6_voc.write_wav_mono(SAMPLES_DIR / f"{src_uid}_to_male115_v6_coupled.wav", y_v6, sr=SR)
            v6_voc.write_wav_mono(SAMPLES_DIR / f"{src_uid}_to_male115_v8_decoupled.wav", y_v2, sr=SR)
            v6_voc.write_wav_mono(SAMPLES_DIR / f"{src_uid}_to_male115_v8_ridge.wav", y_rg, sr=SR)

        for tag, y_m in (("v6_coupled_hard084", y_v6), ("v8_decoupled_autovtln", y_v2), ("v8_ridge", y_rg)):
            ev_m = eval_m.evaluate_audio(y_m, sr=SR)
            cl_m = compute_clarity_metrics(y_m, sr=SR)
            sd_m = eval_m.compute_spectral_distances(real_wavs[src_uid], y_m, sr=SR)
            male_stress.append({
                "source_id": src_uid,
                "method": tag,
                "auto_vtln_alpha": meta_v2["alpha"] if tag == "v8_decoupled_autovtln" else (0.84 if "084" in tag else None),
                "auto_vtln_raw_ratio": meta_v2["vtln_info"]["raw_ratio"] if tag == "v8_decoupled_autovtln" else None,
                "f0_median_hz": round(ev_m["f0_hnr"]["f0_median_hz"], 2),
                "tilt_80_8k_db_oct": round(ev_m["ltas"]["tilt_db_oct"], 3),
                "target_tilt_err_db_oct": round(abs(ev_m["ltas"]["tilt_db_oct"] - (-4.5)), 3),
                "f1_median_hz": round(ev_m["formants"]["f1_median_hz"], 1),
                "f2_median_hz": round(ev_m["formants"]["f2_median_hz"], 1),
                "formant_prom_delta_db": round(cl_m["formant_prominence_db"] - real_clarity[src_uid]["formant_prominence_db"], 3),
                "consonant_hf_delta_db": round(cl_m["consonant_hf_ratio_db"] - real_clarity[src_uid]["consonant_hf_ratio_db"], 3),
                "content_mcd_cmn_db": round(sd_m["mcd_cmn_db"], 3),
                "sanity_ok": bool(ev_m["sanity"]["ok"]),
            })

    summary = {
        "unconverted_donor": _summarize_rows(unconverted_rows),
        "loo_cv_1shot": {k: _summarize_rows(v) for k, v in modes_1shot.items()},
        "loo_cv_3shot": {k: _summarize_rows(v) for k, v in modes_3shot.items()},
        "male_115hz_stress_test": male_stress,
    }
    (Path(__file__).resolve().parent / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def evaluate_corpus_manifest(manifest_path: str | Path, *, n_prompt: int = 3, max_eval_per_pair: int = 4) -> dict:
    """Corpus-agnostic evaluator ready for ``exp_v8_data/corpus/manifest.json`` (cross-gender & EN <-> ZH)."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = manifest["utterances"] if isinstance(manifest, dict) and "utterances" in manifest else manifest
    by_spk: dict[str, list[dict]] = {}
    for row in entries:
        by_spk.setdefault(row["speaker"], []).append(row)

    profiles: dict[str, TimbreProfileV2] = {}
    held_utts: dict[str, list[dict]] = {}
    for spk, rows in sorted(by_spk.items()):
        prompt_rows = rows[:n_prompt]
        held_utts[spk] = rows[n_prompt : n_prompt + max_eval_per_pair]
        wavs = [v6_voc.read_wav_mono(r["path"])[0] for r in prompt_rows]
        profiles[spk] = extract_v2_profile(wavs, sr=SR, name=spk)

    pair_results = []
    spks = sorted(profiles.keys())
    for src_spk in spks:
        for tgt_spk in spks:
            if src_spk == tgt_spk or not held_utts[src_spk]:
                continue
            src_row = held_utts[src_spk][0]
            x_src, _ = v6_voc.read_wav_mono(src_row["path"])
            feats_src = v6_voc.analyze(x_src, sr=SR)
            ad_v2, meta_v2 = adapt_features_v2(feats_src, profiles[src_spk], profiles[tgt_spk], vtln_alpha=None)
            y_v2 = v6_voc.synthesize(ad_v2, seed=72, phase_dispersion=True)
            ev_y = eval_m.evaluate_audio(y_v2, sr=SR)
            cl_src = compute_clarity_metrics(x_src, sr=SR)
            cl_y = compute_clarity_metrics(y_v2, sr=SR)
            pair_results.append({
                "src_spk": src_spk,
                "tgt_spk": tgt_spk,
                "auto_vtln_alpha": meta_v2["alpha"],
                "f0_err_st": round(abs(12.0 * math.log2(ev_y["f0_hnr"]["f0_median_hz"] / profiles[tgt_spk].f0_median_hz)), 3),
                "tilt_err_db_oct": round(abs(ev_y["ltas"]["tilt_db_oct"] - profiles[tgt_spk].tilt_db_per_oct), 3),
                "consonant_hf_delta_db": round(cl_y["consonant_hf_ratio_db"] - cl_src["consonant_hf_ratio_db"], 3),
                "sanity_ok": bool(ev_y["sanity"]["ok"]),
            })
    return {"speakers": {k: {"f0_hz": round(v.f0_median_hz, 1), "tilt": round(v.tilt_db_per_oct, 2)} for k, v in profiles.items()}, "pairs": pair_results}


if __name__ == "__main__":
    res = run_ablation(export_samples=True)
    print(json.dumps(res, ensure_ascii=False, indent=2))
