"""Real 4-speaker cross-gender & same-gender English cloning evaluation (CMU Arctic).

Strict zero-leakage split over ``exp_v8_data/corpus/manifest.json`` (``bdl``, ``rms``, ``clb``, ``slt``):
- ``a0001..a0003`` (3 utts/spk): 3-shot prompt pool for TimbreProfile extraction
- ``a0004..a0019`` (16 utts/spk): 4-way speaker classifier training set
- ``a0020..a0024`` (5 utts/spk): held-out source & target evaluation set (60 conversions per method)

Compares ``unconverted_donor`` vs ``v6_coupled`` (auto-VTLN) vs ``v8_decoupled`` (V/UV decoupled + F1/F2/F3 auto-VTLN).
"""
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
from exp_v8_timbre_clone import EXP_DIR, SR
from exp_v8_timbre_clone.clarity_metrics import compute_clarity_metrics
from exp_v8_timbre_clone.timbre_cloner_v2 import adapt_features_v2, extract_v2_profile

MANIFEST_PATH = ROOT / "exp_v8_data" / "corpus" / "manifest.json"
SAMPLES_EN_DIR = EXP_DIR / "samples_en"


def _peak_limit(y: np.ndarray) -> np.ndarray:
    pk = float(np.max(np.abs(y))) if y.size else 0.0
    return y * (0.90 / pk) if pk > 0.95 else y


def _classify_4way(
    query_vec: dict[str, np.ndarray],
    train_rows: list[tuple[str, dict[str, np.ndarray]]],
    key: str,
) -> str:
    """Standardized 4-way nearest-centroid classifier fitted strictly on train_rows."""
    x_tr = np.stack([v[key] for _, v in train_rows], axis=0)
    mu = np.mean(x_tr, axis=0)
    sd = np.maximum(np.std(x_tr, axis=0, ddof=1), 1e-6)
    xq = (query_vec[key] - mu) / sd
    best_spk = ""
    best_d = float("inf")
    for spk in sorted({s for s, _ in train_rows}):
        cent = np.mean([(v[key] - mu) / sd for s, v in train_rows if s == spk], axis=0)
        d = float(np.linalg.norm(xq - cent))
        if d < best_d:
            best_d = d
            best_spk = spk
    return best_spk


def _pair_kind(src_gender: str, tgt_gender: str) -> str:
    if src_gender == "male" and tgt_gender == "female":
        return "M->F"
    if src_gender == "female" and tgt_gender == "male":
        return "F->M"
    return "M->M" if src_gender == "male" else "F->F"


def run_english_eval(export_samples: bool = True) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    en_utts = manifest["english"]["utterances"]
    by_spk: dict[str, list[dict]] = {}
    for u in en_utts:
        by_spk.setdefault(u["speaker"], []).append(u)
    for spk in by_spk:
        by_spk[spk].sort(key=lambda r: r["id"])

    spks = sorted(by_spk.keys())  # ['bdl', 'clb', 'rms', 'slt']
    genders = {s: by_spk[s][0]["gender"] for s in spks}

    # Pre-analyze prompt (0..2), clf_train (3..18), and held_test (19..23)
    wavs: dict[str, np.ndarray] = {}
    feats: dict[str, v6_voc.VocoderFeatures] = {}
    clf_vecs: dict[str, dict[str, np.ndarray]] = {}
    clarity_ref: dict[str, dict] = {}

    prompt_rows = {s: by_spk[s][:3] for s in spks}
    clf_train_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    held_rows = {s: by_spk[s][19:24] for s in spks}  # 5 held-out utts per speaker

    for s in spks:
        for u in by_spk[s][:24]:
            uid = u["id"]
            x, sr = v6_voc.read_wav_mono(ROOT / u["path"])
            if sr != SR:
                raise ValueError(f"Expected {SR} Hz, got {sr} for {uid}")
            wavs[uid] = x
            feats[uid] = v6_voc.analyze(x, sr=SR)
            clf_vecs[uid] = v6_eval._extract_clf_vec(x, sr=SR)
            if u in held_rows[s]:
                clarity_ref[uid] = compute_clarity_metrics(x, sr=SR)
        for u in by_spk[s][3:19]:
            clf_train_rows.append((s, clf_vecs[u["id"]]))

    # Real-vs-real 4-way held-out oracle accuracy
    real_acc = {"timbre4d": 0, "ltas": 0, "f0": 0, "all": 0, "n": 0}
    for s in spks:
        for u in held_rows[s]:
            qv = clf_vecs[u["id"]]
            real_acc["n"] += 1
            real_acc["timbre4d"] += int(_classify_4way(qv, clf_train_rows, "timbre_4d") == s)
            real_acc["ltas"] += int(_classify_4way(qv, clf_train_rows, "ltas_vec") == s)
            real_acc["f0"] += int(_classify_4way(qv, clf_train_rows, "f0_stats") == s)
            real_acc["all"] += int(_classify_4way(qv, clf_train_rows, "all_features") == s)

    # Extract 3-shot speaker profiles from prompt_rows (a0001..a0003)
    profs_v2 = {
        s: extract_v2_profile([(wavs[u["id"]], feats[u["id"]]) for u in prompt_rows[s]], sr=SR, name=s)
        for s in spks
    }
    profs_v6 = {s: profs_v2[s].to_v6_profile() for s in spks}

    # Target speaker held-out centroid stats for error measurement
    tgt_oracle_vecs = {
        s: {
            "tilt": float(np.mean([clf_vecs[u["id"]]["tilt"] for u in held_rows[s]])),
            "ltas_vec": np.mean([clf_vecs[u["id"]]["ltas_vec"] for u in held_rows[s]], axis=0),
            "f0_med": float(np.median([clf_vecs[u["id"]]["f0_med"] for u in held_rows[s]])),
            "hnr": float(np.mean([clf_vecs[u["id"]]["hnr"] for u in held_rows[s]])),
            "mfcc_mean": np.mean([clf_vecs[u["id"]]["mfcc_mean"] for u in held_rows[s]], axis=0),
        }
        for s in spks
    }

    SAMPLES_EN_DIR.mkdir(parents=True, exist_ok=True)
    export_pairs = {("bdl", "slt"), ("slt", "bdl"), ("rms", "clb"), ("clb", "rms"), ("bdl", "rms"), ("slt", "clb")}

    rows_by_method: dict[str, list[dict]] = {"unconverted": [], "v6_coupled": [], "v8_decoupled": []}

    for src_spk in spks:
        for tgt_spk in spks:
            if src_spk == tgt_spk:
                continue
            pkind = _pair_kind(genders[src_spk], genders[tgt_spk])
            t_ref = tgt_oracle_vecs[tgt_spk]
            for idx_u, src_u in enumerate(held_rows[src_spk]):
                suid = src_u["id"]
                src_x = wavs[suid]
                src_f = feats[suid]
                s_vec = clf_vecs[suid]

                # Unconverted donor baseline
                rows_by_method["unconverted"].append({
                    "pair": f"{src_spk}->{tgt_spk}",
                    "pair_kind": pkind,
                    "clf_4way_timbre4d": bool(_classify_4way(s_vec, clf_train_rows, "timbre_4d") == tgt_spk),
                    "clf_4way_ltas": bool(_classify_4way(s_vec, clf_train_rows, "ltas_vec") == tgt_spk),
                    "clf_4way_f0": bool(_classify_4way(s_vec, clf_train_rows, "f0_stats") == tgt_spk),
                    "clf_4way_all": bool(_classify_4way(s_vec, clf_train_rows, "all_features") == tgt_spk),
                    "tilt_err_db_oct": abs(s_vec["tilt"] - t_ref["tilt"]),
                    "ltas_env_rmse_db": float(np.sqrt(np.mean((s_vec["ltas_vec"] - t_ref["ltas_vec"]) ** 2))),
                    "f0_err_st": abs(12.0 * math.log2(s_vec["f0_med"] / t_ref["f0_med"])),
                    "hnr_err_db": abs(s_vec["hnr"] - t_ref["hnr"]),
                    "content_mel_lsd_db": 0.0,
                    "content_mcd_cmn_db": 0.0,
                    "formant_prom_delta_db": 0.0,
                    "consonant_hf_delta_db": 0.0,
                    "vtln_alpha": 1.0,
                    "sanity_ok": True,
                })

                for mname in ("v6_coupled", "v8_decoupled"):
                    t0 = time.perf_counter()
                    if mname == "v6_coupled":
                        ad = v6_cloner.adapt_features(src_f, profs_v6[src_spk], profs_v6[tgt_spk], vtln_alpha=None)
                        raw_alpha = profs_v6[tgt_spk].vtln_anchor_f2_hz / max(100.0, profs_v6[src_spk].vtln_anchor_f2_hz)
                        used_alpha = 1.0 if abs(raw_alpha - 1.0) < 0.04 else float(np.clip(raw_alpha, 0.78, 1.22))
                    else:
                        ad, meta_v2 = adapt_features_v2(src_f, profs_v2[src_spk], profs_v2[tgt_spk], vtln_alpha=None)
                        used_alpha = meta_v2["alpha"]
                    y_c = _peak_limit(v6_voc.synthesize(ad, seed=72 + idx_u, phase_dispersion=True))
                    dt = time.perf_counter() - t0

                    san = eval_m.check_wav_sanity(y_c, SR)
                    cv = v6_eval._extract_clf_vec(y_c, sr=SR)
                    cl = compute_clarity_metrics(y_c, sr=SR)
                    sd = eval_m.compute_spectral_distances(src_x, y_c, sr=SR)

                    rows_by_method[mname].append({
                        "pair": f"{src_spk}->{tgt_spk}",
                        "pair_kind": pkind,
                        "src_id": suid,
                        "clf_4way_timbre4d": bool(_classify_4way(cv, clf_train_rows, "timbre_4d") == tgt_spk),
                        "clf_4way_ltas": bool(_classify_4way(cv, clf_train_rows, "ltas_vec") == tgt_spk),
                        "clf_4way_f0": bool(_classify_4way(cv, clf_train_rows, "f0_stats") == tgt_spk),
                        "clf_4way_all": bool(_classify_4way(cv, clf_train_rows, "all_features") == tgt_spk),
                        "tilt_err_db_oct": abs(cv["tilt"] - t_ref["tilt"]),
                        "ltas_env_rmse_db": float(np.sqrt(np.mean((cv["ltas_vec"] - t_ref["ltas_vec"]) ** 2))),
                        "f0_err_st": abs(12.0 * math.log2(cv["f0_med"] / t_ref["f0_med"])),
                        "hnr_err_db": abs(cv["hnr"] - t_ref["hnr"]),
                        "content_mel_lsd_db": float(sd["mel_lsd_db"]),
                        "content_mcd_cmn_db": float(sd["mcd_cmn_db"]),
                        "formant_prom_delta_db": float(cl["formant_prominence_db"] - clarity_ref[suid]["formant_prominence_db"]),
                        "consonant_hf_delta_db": float(cl["consonant_hf_ratio_db"] - clarity_ref[suid]["consonant_hf_ratio_db"]),
                        "vtln_alpha": float(used_alpha),
                        "rtf": dt / (len(src_x) / SR),
                        "sanity_ok": bool(san["ok"]),
                    })

                    if export_samples and idx_u == 0 and (src_spk, tgt_spk) in export_pairs:
                        v6_voc.write_wav_mono(SAMPLES_EN_DIR / f"{suid}_to_{tgt_spk}_{mname}.wav", y_c, sr=SR)
                        if mname == "v8_decoupled":
                            v6_voc.write_wav_mono(SAMPLES_EN_DIR / f"{suid}_real_src.wav", src_x, sr=SR)
                            v6_voc.write_wav_mono(SAMPLES_EN_DIR / f"{tgt_spk}_real_ref.wav", wavs[held_rows[tgt_spk][0]["id"]], sr=SR)

    def _agg(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "sanity_ok": sum(1 for r in rows if r["sanity_ok"]),
            "clf_4way_timbre4d": round(float(np.mean([r["clf_4way_timbre4d"] for r in rows])), 4),
            "clf_4way_ltas": round(float(np.mean([r["clf_4way_ltas"] for r in rows])), 4),
            "clf_4way_f0": round(float(np.mean([r["clf_4way_f0"] for r in rows])), 4),
            "clf_4way_all": round(float(np.mean([r["clf_4way_all"] for r in rows])), 4),
            "tilt_err_db_oct": round(float(np.mean([r["tilt_err_db_oct"] for r in rows])), 3),
            "ltas_env_rmse_db": round(float(np.mean([r["ltas_env_rmse_db"] for r in rows])), 3),
            "f0_err_st": round(float(np.mean([r["f0_err_st"] for r in rows])), 3),
            "hnr_err_db": round(float(np.mean([r["hnr_err_db"] for r in rows])), 3),
            "content_mel_lsd_db": round(float(np.mean([r["content_mel_lsd_db"] for r in rows])), 3),
            "content_mcd_cmn_db": round(float(np.mean([r["content_mcd_cmn_db"] for r in rows])), 3),
            "formant_prom_delta_db": round(float(np.mean([r["formant_prom_delta_db"] for r in rows])), 3),
            "consonant_hf_delta_db": round(float(np.mean([r["consonant_hf_delta_db"] for r in rows])), 3),
            "vtln_alpha_mean": round(float(np.mean([r["vtln_alpha"] for r in rows])), 4),
        }

    summary = {
        "real_4way_oracle_acc": {
            k: round(real_acc[k] / real_acc["n"], 4) for k in ("timbre4d", "ltas", "f0", "all")
        } | {"n": real_acc["n"]},
        "speaker_profiles_3shot": {
            s: {
                "gender": genders[s],
                "voiced_s": round(profs_v2[s].total_voiced_s, 2),
                "f0_hz": round(profs_v2[s].f0_median_hz, 2),
                "tilt_db_oct": round(profs_v2[s].tilt_db_per_oct, 3),
                "hnr_db": round(profs_v2[s].hnr_median_db, 2),
                "f1_f2_f3_hz": [round(profs_v2[s].f1_median_hz, 1), round(profs_v2[s].f2_median_hz, 1), round(profs_v2[s].f3_median_hz, 1)],
            }
            for s in spks
        },
        "overall": {m: _agg(rows) for m, rows in rows_by_method.items()},
        "by_pair_kind": {
            m: {
                pk: _agg([r for r in rows if r["pair_kind"] == pk])
                for pk in ("M->F", "F->M", "M->M", "F->F")
            }
            for m, rows in rows_by_method.items()
        },
        "by_pair_v8_vs_v6": {
            pair: {
                "v6_coupled": _agg([r for r in rows_by_method["v6_coupled"] if r["pair"] == pair]),
                "v8_decoupled": _agg([r for r in rows_by_method["v8_decoupled"] if r["pair"] == pair]),
            }
            for pair in sorted({r["pair"] for r in rows_by_method["v8_decoupled"]})
        },
    }
    (EXP_DIR / "results_en.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    res = run_english_eval(export_samples=True)
    print(json.dumps({k: v for k, v in res.items() if k != "by_pair_v8_vs_v6"}, ensure_ascii=False, indent=2))
