"""Phase-6 end-to-end audit & verification script (`exp_v10_g2p_and_align`).

Evaluates:
1. G2P coverage & OOV fixes across 160 CMU-ARCTIC sentences & 26 benchmark sentences.
2. Landmark Aligner + Acoustic Token Quarantine vs Phase-4/5 DP aligner across 140 EN & 32 ZH tokens.
3. TimbreClonerV4 V/U loudness calibration (measuring V-U gap drift vs v3) & adaptive routing
   across EN (intra-corpus), ZH (intra-corpus), and ZH<->EN (cross-corpus).
"""
from __future__ import annotations

import json
import math
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
from exp_v9_timbre_v3.eq_decoupler import estimate_corpus_baseline
from exp_v9_timbre_v3.timbre_cloner_v3 import adapt_features_v3, extract_timbre_v3
from exp_v10_g2p_and_align import EXP_DIR, MANIFEST_PATH, SR
from exp_v10_g2p_and_align.g2p_audit_and_fix import audit_g2p_coverage
from exp_v10_g2p_and_align.landmark_aligner import audit_aligner_before_after
from exp_v10_g2p_and_align.timbre_cloner_v4 import adapt_features_v4, extract_timbre_v4


def _vu_contrast_db(feats: v6_voc.VocoderFeatures, sr: int = SR) -> float:
    freqs = np.fft.rfftfreq(feats.n_fft, 1.0 / sr)
    m = (freqs >= 80.0) & (freqs <= 7500.0)
    v = feats.voiced & (feats.f0_hz > 0.0)
    u = ~v
    if not np.any(v) or not np.any(u):
        return 14.0
    pwr = np.exp(2.0 * feats.log_env[:, m])
    v_db = 10.0 * np.log10(np.maximum(1e-18, np.mean(pwr[v], axis=1)))
    u_db = 10.0 * np.log10(np.maximum(1e-18, np.mean(pwr[u], axis=1)))
    u_act = u_db[u_db >= np.percentile(u_db, 30)]
    return float(np.median(v_db) - (np.median(u_act) if u_act.size else np.median(u_db)))


def eval_cloner_v4() -> dict:
    """Compare V3 vs V4 on V-U loudness jump control & intra/cross-corpus cloning."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    en_by_spk: dict[str, list[dict]] = {}
    for u in manifest["english"]["utterances"]:
        en_by_spk.setdefault(u["speaker"], []).append({"id": u["id"], "speaker": u["speaker"], "path": str(ROOT / u["path"])})
    for s in en_by_spk:
        en_by_spk[s].sort(key=lambda r: r["id"])
    zh_by_spk = _load_zh_5speakers(manifest)

    spks = {"bdl": ("arctic", en_by_spk["bdl"]), "slt": ("arctic", en_by_spk["slt"]), "rms": ("arctic", en_by_spk["rms"]),
            "SSB0073": ("aishell3", zh_by_spk["SSB0073"]), "SSB0502": ("aishell3", zh_by_spk["SSB0502"])}

    p3, p4 = {}, {}
    clf_train = []
    held = {}
    tgt_tilt = {}
    for s, (corp, rows) in spks.items():
        items = [(_read_utt(r), v6_voc.analyze(_read_utt(r), SR)) for r in rows[:3]]
        p3[s] = extract_timbre_v3(items, sr=SR, name=s, corpus=corp)
        p4[s] = extract_timbre_v4(items, sr=SR, name=s, corpus=corp)
        if s in ("bdl", "slt", "SSB0073", "SSB0502"):
            for r in rows[3:12]:
                clf_train.append((s, v6_eval._extract_clf_vec(_read_utt(r), sr=SR)))
        held[s] = [(_read_utt(r), v6_voc.analyze(_read_utt(r), SR)) for r in rows[12:15]]
        tgt_tilt[s] = float(np.mean([eval_m.compute_ltas_metrics(w, SR)["tilt_db_oct"] for w, _ in held[s]]))

    # Pairs covering EN F->M (slt->rms), EN M->F (bdl->slt), ZH<->EN M->M (bdl<->SSB0073) & cross-gender
    pairs = [
        ("slt", "rms", "intra_en_F2M"),
        ("bdl", "slt", "intra_en_M2F"),
        ("bdl", "SSB0073", "cross_zh_en_M2M"),
        ("SSB0073", "bdl", "cross_zh_en_M2M"),
        ("bdl", "SSB0502", "cross_zh_en_M2F"),
        ("SSB0502", "slt", "cross_zh_en_F2F"),
        ("slt", "SSB0073", "cross_zh_en_F2M"),
    ]

    v3_rows, v4_rows = [], []
    for src, tgt, tag in pairs:
        for w0, f0 in held[src]:
            src_gap = _vu_contrast_db(f0, sr=SR)
            f3, _ = adapt_features_v3(f0, p3[src], p3[tgt])
            w3 = _peak_limit(v6_voc.synthesize(f3, seed=72, phase_dispersion=True))
            g3 = _vu_contrast_db(f3, sr=SR)
            t3 = eval_m.compute_ltas_metrics(w3, SR)["tilt_db_oct"]

            f4, info4 = adapt_features_v4(f0, p4[src], p4[tgt])
            w4 = _peak_limit(v6_voc.synthesize(f4, seed=72, phase_dispersion=True))
            g4 = _vu_contrast_db(f4, sr=SR)
            t4 = eval_m.compute_ltas_metrics(w4, SR)["tilt_db_oct"]

            cl_hit3, cl_hit4 = None, None
            if src in ("bdl", "slt", "SSB0073", "SSB0502") and tgt in ("bdl", "slt", "SSB0073", "SSB0502"):
                q3 = v6_eval._extract_clf_vec(w3, sr=SR)
                q4 = v6_eval._extract_clf_vec(w4, sr=SR)
                cl_hit3 = int(_classify_4way(q3, clf_train, "timbre_4d") == tgt)
                cl_hit4 = int(_classify_4way(q4, clf_train, "timbre_4d") == tgt)

            v3_rows.append({"tag": tag, "vu_drift_db": abs(g3 - src_gap), "tilt_err": abs(t3 - tgt_tilt[tgt]), "hit4d": cl_hit3, "ok": int(eval_m.check_wav_sanity(w3, SR)["ok"])})
            v4_rows.append({"tag": tag, "mode": info4["routing_mode"], "vu_drift_db": abs(g4 - src_gap), "tilt_err": abs(t4 - tgt_tilt[tgt]), "hit4d": cl_hit4, "ok": int(eval_m.check_wav_sanity(w4, SR)["ok"])})

    def _agg(rs: list[dict], sub: str | None = None) -> dict:
        sel = [r for r in rs if (sub is None or r["tag"].startswith(sub))]
        hits = [r["hit4d"] for r in sel if r["hit4d"] is not None]
        return {
            "n": len(sel),
            "sanity_ok": int(sum(r["ok"] for r in sel)),
            "vu_drift_mean_db": round(float(np.mean([r["vu_drift_db"] for r in sel])), 3),
            "vu_drift_max_db": round(float(np.max([r["vu_drift_db"] for r in sel])), 3),
            "tilt_err_mean_db_oct": round(float(np.mean([r["tilt_err"] for r in sel])), 3),
            "clf_4way_timbre4d": round(float(np.mean(hits)), 4) if hits else None,
        }

    return {
        "v3_overall": _agg(v3_rows),
        "v4_overall": _agg(v4_rows),
        "v3_intra_en_F2M": _agg(v3_rows, "intra_en_F2M"),
        "v4_intra_en_F2M": _agg(v4_rows, "intra_en_F2M"),
        "v3_cross_zh_en": _agg(v3_rows, "cross_zh_en"),
        "v4_cross_zh_en": _agg(v4_rows, "cross_zh_en"),
        "v3_bdl_ssb0073": _agg(v3_rows, "cross_zh_en_M2M"),
        "v4_bdl_ssb0073": _agg(v4_rows, "cross_zh_en_M2M"),
    }


def main() -> None:
    report = {
        "g2p_audit": audit_g2p_coverage(),
        "aligner_and_quarantine": audit_aligner_before_after(),
        "cloner_v4_eval": eval_cloner_v4(),
    }
    out_p = EXP_DIR / "results_phase6.json"
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_p}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
