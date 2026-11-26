"""5-Speaker Mandarin (M+F) & Bidirectional Cross-Lingual (ZH <-> EN) Timbre Cloning Evaluation.

Uses strictly disjoint prompt / classifier-train / held-out-test splits over:
- Mandarin (5 speakers): SSB0073 (M), SSB0139 (M), SSB0005 (F), SSB0502 (F), SSB1593 (F)
- Cross-lingual (4 speakers): SSB0073 (ZH-M), SSB0502 (ZH-F), bdl (EN-M), slt (EN-F)
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
from exp_v8_timbre_clone import EXP_DIR, EXTENDED_INDEX, SR
from exp_v8_timbre_clone.clarity_metrics import compute_clarity_metrics
from exp_v8_timbre_clone.eval_english_crossgender import _classify_4way, _pair_kind, _peak_limit
from exp_v8_timbre_clone.timbre_cloner_v2 import adapt_features_v2, extract_v2_profile

MANIFEST_PATH = ROOT / "exp_v8_data" / "corpus" / "manifest.json"
SAMPLES_ML_DIR = EXP_DIR / "samples_multiling"


def _load_zh_5speakers(manifest: dict) -> dict[str, list[dict]]:
    """Collect 16+ utterances per speaker for all 5 Mandarin speakers with active slicing on extended/."""
    by_spk: dict[str, list[dict]] = {}
    for u in manifest["mandarin"]["utterances"]:
        by_spk.setdefault(u["speaker"], []).append({
            "id": u["id"],
            "speaker": u["speaker"],
            "gender": u["gender"],
            "path": str(ROOT / u["path"]),
            "active_start": None,
            "active_stop": None,
        })
    ext_idx = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))
    for u in ext_idx["utterances"]:
        by_spk.setdefault(u["speaker"], []).append({
            "id": u["id"],
            "speaker": u["speaker"],
            "gender": "female",
            "path": u["path"],
            "active_start": int(u["active_start"]),
            "active_stop": int(u["active_stop"]),
        })
    for spk in by_spk:
        by_spk[spk].sort(key=lambda r: r["id"])
    return by_spk


def _read_utt(row: dict) -> np.ndarray:
    x, sr = v6_voc.read_wav_mono(row["path"])
    if sr != SR:
        raise ValueError(f"Expected {SR} Hz, got {sr} for {row['id']}")
    if row.get("active_start") is not None and row.get("active_stop") is not None:
        return x[row["active_start"] : row["active_stop"]]
    # Trim leading/trailing near-silence on full AISHELL-3 / Arctic files cleanly (20 ms margin)
    rms = np.abs(x)
    pk = float(np.max(rms)) if rms.size else 0.0
    if pk <= 1e-6:
        return x
    idx = np.where(rms >= 0.02 * pk)[0]
    if idx.size < 480:
        return x
    lo = max(0, int(idx[0]) - 480)
    hi = min(len(x), int(idx[-1]) + 480)
    return x[lo:hi]


def _run_speaker_eval(
    speaker_rows: dict[str, list[dict]],
    pairs: list[tuple[str, str]],
    *,
    n_prompt: int = 3,
    n_clf_train: int = 9,
    n_held: int = 3,
    export_pairs: set[tuple[str, str]] = (),
    tag_prefix: str = "",
) -> dict:
    spks = sorted(speaker_rows.keys())
    genders = {s: speaker_rows[s][0]["gender"] for s in spks}

    wavs: dict[str, np.ndarray] = {}
    feats: dict[str, v6_voc.VocoderFeatures] = {}
    clf_vecs: dict[str, dict[str, np.ndarray]] = {}
    clarity_ref: dict[str, dict] = {}

    prompt_rows = {s: speaker_rows[s][:n_prompt] for s in spks}
    clf_train_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    held_rows = {s: speaker_rows[s][n_prompt + n_clf_train : n_prompt + n_clf_train + n_held] for s in spks}

    for s in spks:
        need = speaker_rows[s][: n_prompt + n_clf_train + n_held]
        for u in need:
            uid = u["id"]
            x = _read_utt(u)
            wavs[uid] = x
            feats[uid] = v6_voc.analyze(x, sr=SR)
            clf_vecs[uid] = v6_eval._extract_clf_vec(x, sr=SR)
            if u in held_rows[s]:
                clarity_ref[uid] = compute_clarity_metrics(x, sr=SR)
        for u in speaker_rows[s][n_prompt : n_prompt + n_clf_train]:
            clf_train_rows.append((s, clf_vecs[u["id"]]))

    real_acc = {"timbre4d": 0, "ltas": 0, "f0": 0, "all": 0, "n": 0}
    for s in spks:
        for u in held_rows[s]:
            qv = clf_vecs[u["id"]]
            real_acc["n"] += 1
            real_acc["timbre4d"] += int(_classify_4way(qv, clf_train_rows, "timbre_4d") == s)
            real_acc["ltas"] += int(_classify_4way(qv, clf_train_rows, "ltas_vec") == s)
            real_acc["f0"] += int(_classify_4way(qv, clf_train_rows, "f0_stats") == s)
            real_acc["all"] += int(_classify_4way(qv, clf_train_rows, "all_features") == s)

    profs_v2 = {
        s: extract_v2_profile([(wavs[u["id"]], feats[u["id"]]) for u in prompt_rows[s]], sr=SR, name=s)
        for s in spks
    }
    profs_v6 = {s: profs_v2[s].to_v6_profile() for s in spks}

    tgt_oracle_vecs = {
        s: {
            "tilt": float(np.mean([clf_vecs[u["id"]]["tilt"] for u in held_rows[s]])),
            "ltas_vec": np.mean([clf_vecs[u["id"]]["ltas_vec"] for u in held_rows[s]], axis=0),
            "f0_med": float(np.median([clf_vecs[u["id"]]["f0_med"] for u in held_rows[s]])),
            "hnr": float(np.mean([clf_vecs[u["id"]]["hnr"] for u in held_rows[s]])),
        }
        for s in spks
    }

    SAMPLES_ML_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_method: dict[str, list[dict]] = {"unconverted": [], "v6_coupled": [], "v8_decoupled": []}

    for src_spk, tgt_spk in pairs:
        pkind = _pair_kind(genders[src_spk], genders[tgt_spk])
        t_ref = tgt_oracle_vecs[tgt_spk]
        for idx_u, src_u in enumerate(held_rows[src_spk]):
            suid = src_u["id"]
            src_x = wavs[suid]
            src_f = feats[suid]
            s_vec = clf_vecs[suid]

            rows_by_method["unconverted"].append({
                "pair": f"{src_spk}->{tgt_spk}",
                "pair_kind": pkind,
                "clf_timbre4d": bool(_classify_4way(s_vec, clf_train_rows, "timbre_4d") == tgt_spk),
                "clf_ltas": bool(_classify_4way(s_vec, clf_train_rows, "ltas_vec") == tgt_spk),
                "clf_f0": bool(_classify_4way(s_vec, clf_train_rows, "f0_stats") == tgt_spk),
                "clf_all": bool(_classify_4way(s_vec, clf_train_rows, "all_features") == tgt_spk),
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
                if mname == "v6_coupled":
                    ad = v6_cloner.adapt_features(src_f, profs_v6[src_spk], profs_v6[tgt_spk], vtln_alpha=None)
                    raw_a = profs_v6[tgt_spk].vtln_anchor_f2_hz / max(100.0, profs_v6[src_spk].vtln_anchor_f2_hz)
                    used_a = 1.0 if abs(raw_a - 1.0) < 0.04 else float(np.clip(raw_a, 0.78, 1.22))
                else:
                    ad, meta_v2 = adapt_features_v2(src_f, profs_v2[src_spk], profs_v2[tgt_spk], vtln_alpha=None)
                    used_a = meta_v2["alpha"]
                y_c = _peak_limit(v6_voc.synthesize(ad, seed=72 + idx_u, phase_dispersion=True))

                san = eval_m.check_wav_sanity(y_c, SR)
                cv = v6_eval._extract_clf_vec(y_c, sr=SR)
                cl = compute_clarity_metrics(y_c, sr=SR)
                sd = eval_m.compute_spectral_distances(src_x, y_c, sr=SR)

                rows_by_method[mname].append({
                    "pair": f"{src_spk}->{tgt_spk}",
                    "pair_kind": pkind,
                    "clf_timbre4d": bool(_classify_4way(cv, clf_train_rows, "timbre_4d") == tgt_spk),
                    "clf_ltas": bool(_classify_4way(cv, clf_train_rows, "ltas_vec") == tgt_spk),
                    "clf_f0": bool(_classify_4way(cv, clf_train_rows, "f0_stats") == tgt_spk),
                    "clf_all": bool(_classify_4way(cv, clf_train_rows, "all_features") == tgt_spk),
                    "tilt_err_db_oct": abs(cv["tilt"] - t_ref["tilt"]),
                    "ltas_env_rmse_db": float(np.sqrt(np.mean((cv["ltas_vec"] - t_ref["ltas_vec"]) ** 2))),
                    "f0_err_st": abs(12.0 * math.log2(cv["f0_med"] / t_ref["f0_med"])),
                    "hnr_err_db": abs(cv["hnr"] - t_ref["hnr"]),
                    "content_mel_lsd_db": float(sd["mel_lsd_db"]),
                    "content_mcd_cmn_db": float(sd["mcd_cmn_db"]),
                    "formant_prom_delta_db": float(cl["formant_prominence_db"] - clarity_ref[suid]["formant_prominence_db"]),
                    "consonant_hf_delta_db": float(cl["consonant_hf_ratio_db"] - clarity_ref[suid]["consonant_hf_ratio_db"]),
                    "vtln_alpha": float(used_a),
                    "sanity_ok": bool(san["ok"]),
                })

                if idx_u == 0 and (src_spk, tgt_spk) in export_pairs:
                    v6_voc.write_wav_mono(SAMPLES_ML_DIR / f"{tag_prefix}{suid}_to_{tgt_spk}_{mname}.wav", y_c, sr=SR)
                    if mname == "v8_decoupled":
                        v6_voc.write_wav_mono(SAMPLES_ML_DIR / f"{tag_prefix}{suid}_real_src.wav", src_x, sr=SR)
                        v6_voc.write_wav_mono(SAMPLES_ML_DIR / f"{tag_prefix}{tgt_spk}_real_ref.wav", wavs[held_rows[tgt_spk][0]["id"]], sr=SR)

    def _agg(rows: list[dict]) -> dict:
        if not rows:
            return {}
        return {
            "n": len(rows),
            "sanity_ok": sum(1 for r in rows if r["sanity_ok"]),
            "clf_timbre4d": round(float(np.mean([r["clf_timbre4d"] for r in rows])), 4),
            "clf_ltas": round(float(np.mean([r["clf_ltas"] for r in rows])), 4),
            "clf_f0": round(float(np.mean([r["clf_f0"] for r in rows])), 4),
            "clf_all": round(float(np.mean([r["clf_all"] for r in rows])), 4),
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

    return {
        "n_speakers": len(spks),
        "real_oracle_acc": {k: round(real_acc[k] / real_acc["n"], 4) for k in ("timbre4d", "ltas", "f0", "all")} | {"n": real_acc["n"]},
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
            m: {pk: _agg([r for r in rows if r["pair_kind"] == pk]) for pk in ("M->F", "F->M", "M->M", "F->F")}
            for m, rows in rows_by_method.items()
        },
    }


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    zh_by_spk = _load_zh_5speakers(manifest)
    zh_spks = sorted(zh_by_spk.keys())
    zh_pairs = [(s, t) for s in zh_spks for t in zh_spks if s != t]
    # 5 speakers x (3 prompt + 9 train + 3 held) = 15 utts each (fits SSB0502/1593 16-utt budget cleanly!)
    zh_5way = _run_speaker_eval(
        zh_by_spk,
        zh_pairs,
        n_prompt=3,
        n_clf_train=9,
        n_held=3,
        export_pairs={("SSB0073", "SSB0502"), ("SSB0502", "SSB0073"), ("SSB0139", "SSB0005")},
        tag_prefix="zh_",
    )

    # Cross-lingual 4-way (ZH-M SSB0073, ZH-F SSB0502, EN-M bdl, EN-F slt)
    en_by_spk = {}
    for u in manifest["english"]["utterances"]:
        if u["speaker"] in ("bdl", "slt"):
            en_by_spk.setdefault(u["speaker"], []).append({
                "id": u["id"],
                "speaker": u["speaker"],
                "gender": u["gender"],
                "path": str(ROOT / u["path"]),
                "active_start": None,
                "active_stop": None,
            })
    for s in en_by_spk:
        en_by_spk[s].sort(key=lambda r: r["id"])

    cl_rows = {
        "SSB0073": zh_by_spk["SSB0073"],
        "SSB0502": zh_by_spk["SSB0502"],
        "bdl": en_by_spk["bdl"],
        "slt": en_by_spk["slt"],
    }
    cl_pairs = [
        ("SSB0073", "slt"), ("SSB0073", "bdl"),
        ("SSB0502", "bdl"), ("SSB0502", "slt"),
        ("bdl", "SSB0502"), ("bdl", "SSB0073"),
        ("slt", "SSB0073"), ("slt", "SSB0502"),
    ]
    cross_lingual = _run_speaker_eval(
        cl_rows,
        cl_pairs,
        n_prompt=3,
        n_clf_train=9,
        n_held=3,
        export_pairs={("bdl", "SSB0502"), ("slt", "SSB0073"), ("SSB0502", "bdl"), ("SSB0073", "slt")},
        tag_prefix="cl_",
    )

    out = {"mandarin_5way": zh_5way, "cross_lingual_zh_en": cross_lingual}
    (EXP_DIR / "results_multiling.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
