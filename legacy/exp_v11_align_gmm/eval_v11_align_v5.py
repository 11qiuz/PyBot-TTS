"""Phase-7 verification benchmark for `gmm_hmm_aligner` (EN 140 + ZH 32) & `TimbreClonerV5`."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_eval import metrics as eval_m
from exp_v6_vocoder import eval_vocoder as v6_eval, vocoder as v6_voc
from exp_v8_timbre_clone.eval_english_crossgender import _classify_4way, _peak_limit
from exp_v8_timbre_clone.eval_multilingual_corpus import _load_zh_5speakers, _read_utt
from exp_v10_g2p_and_align.eval_v10_phase6 import _vu_contrast_db
from exp_v10_g2p_and_align.timbre_cloner_v4 import adapt_features_v4, extract_timbre_v4
from exp_v11_align_gmm import EXP_DIR, MANIFEST_PATH, SR
from exp_v11_align_gmm.gmm_hmm_aligner import evaluate_gmm_vs_v10
from exp_v11_align_gmm.timbre_cloner_v5 import adapt_features_v5, extract_timbre_v5


def eval_v5_vs_v4() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    en_by_spk: dict[str, list[dict]] = {}
    for u in manifest["english"]["utterances"]:
        en_by_spk.setdefault(u["speaker"], []).append({"id": u["id"], "speaker": u["speaker"], "path": str(ROOT / u["path"])})
    for s in en_by_spk:
        en_by_spk[s].sort(key=lambda r: r["id"])
    zh_by_spk = _load_zh_5speakers(manifest)

    spks = {
        "bdl": ("arctic", en_by_spk["bdl"]),
        "slt": ("arctic", en_by_spk["slt"]),
        "rms": ("arctic", en_by_spk["rms"]),
        "SSB0073": ("aishell3", zh_by_spk["SSB0073"]),
        "SSB0502": ("aishell3", zh_by_spk["SSB0502"]),
    }

    p4, p5 = {}, {}
    clf_train = []
    held = {}
    tgt_tilt = {}
    for s, (corp, rows) in spks.items():
        items = [(_read_utt(r), v6_voc.analyze(_read_utt(r), SR)) for r in rows[:3]]
        p4[s] = extract_timbre_v4(items, sr=SR, name=s, corpus=corp)
        p5[s] = extract_timbre_v5(items, sr=SR, name=s, corpus=corp)
        if s in ("bdl", "slt", "SSB0073", "SSB0502"):
            for r in rows[3:12]:
                clf_train.append((s, v6_eval._extract_clf_vec(_read_utt(r), sr=SR)))
        held[s] = [(_read_utt(r), v6_voc.analyze(_read_utt(r), SR)) for r in rows[12:15]]
        tgt_tilt[s] = float(np.mean([eval_m.compute_ltas_metrics(w, SR)["tilt_db_oct"] for w, _ in held[s]]))

    pairs = [
        ("slt", "rms", "intra_en_F2M"),
        ("bdl", "slt", "intra_en_M2F"),
        ("bdl", "SSB0073", "cross_zh_en_M2M"),
        ("SSB0073", "bdl", "cross_zh_en_M2M"),
        ("bdl", "SSB0502", "cross_zh_en_M2F"),
        ("SSB0502", "slt", "cross_zh_en_F2F"),
        ("slt", "SSB0073", "cross_zh_en_F2M"),
    ]

    v4_rows, v5_rows = [], []
    for src, tgt, tag in pairs:
        for _w0, f0 in held[src]:
            src_gap = _vu_contrast_db(f0, sr=SR)
            f4, _ = adapt_features_v4(f0, p4[src], p4[tgt])
            w4 = _peak_limit(v6_voc.synthesize(f4, seed=72, phase_dispersion=True))
            g4 = _vu_contrast_db(f4, sr=SR)
            t4 = eval_m.compute_ltas_metrics(w4, SR)["tilt_db_oct"]

            f5, _ = adapt_features_v5(f0, p5[src], p5[tgt])
            w5 = _peak_limit(v6_voc.synthesize(f5, seed=72, phase_dispersion=True))
            g5 = _vu_contrast_db(f5, sr=SR)
            t5 = eval_m.compute_ltas_metrics(w5, SR)["tilt_db_oct"]

            h4, h5 = None, None
            if src in ("bdl", "slt", "SSB0073", "SSB0502") and tgt in ("bdl", "slt", "SSB0073", "SSB0502"):
                q4 = v6_eval._extract_clf_vec(w4, sr=SR)
                q5 = v6_eval._extract_clf_vec(w5, sr=SR)
                h4 = int(_classify_4way(q4, clf_train, "timbre_4d") == tgt)
                h5 = int(_classify_4way(q5, clf_train, "timbre_4d") == tgt)

            v4_rows.append({"tag": tag, "vu_drift": abs(g4 - src_gap), "tilt_err": abs(t4 - tgt_tilt[tgt]), "hit4d": h4, "ok": int(eval_m.check_wav_sanity(w4, SR)["ok"])})
            v5_rows.append({"tag": tag, "vu_drift": abs(g5 - src_gap), "tilt_err": abs(t5 - tgt_tilt[tgt]), "hit4d": h5, "ok": int(eval_m.check_wav_sanity(w5, SR)["ok"])})

    def _agg(rs: list[dict], sub: str | None = None) -> dict:
        sel = [r for r in rs if (sub is None or r["tag"].startswith(sub))]
        hits = [r["hit4d"] for r in sel if r["hit4d"] is not None]
        return {
            "n": len(sel),
            "sanity_ok": int(sum(r["ok"] for r in sel)),
            "vu_drift_mean_db": round(float(np.mean([r["vu_drift"] for r in sel])), 3),
            "vu_drift_max_db": round(float(np.max([r["vu_drift"] for r in sel])), 3),
            "tilt_err_mean_db_oct": round(float(np.mean([r["tilt_err"] for r in sel])), 3),
            "clf_4way_timbre4d": round(float(np.mean(hits)), 4) if hits else None,
        }

    return {
        "v4_overall": _agg(v4_rows),
        "v5_overall": _agg(v5_rows),
        "v4_intra_en_F2M": _agg(v4_rows, "intra_en_F2M"),
        "v5_intra_en_F2M": _agg(v5_rows, "intra_en_F2M"),
        "v4_cross_zh_en": _agg(v4_rows, "cross_zh_en"),
        "v5_cross_zh_en": _agg(v5_rows, "cross_zh_en"),
        "v4_bdl_ssb0073": _agg(v4_rows, "cross_zh_en_M2M"),
        "v5_bdl_ssb0073": _agg(v5_rows, "cross_zh_en_M2M"),
    }


def main() -> None:
    out = {
        "gmm_alignment": evaluate_gmm_vs_v10(),
        "cloner_v5_eval": eval_v5_vs_v4(),
    }
    p = EXP_DIR / "results_v11_align.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {p}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
