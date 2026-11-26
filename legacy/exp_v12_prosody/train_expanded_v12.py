"""Train & evaluate Phase-8 GMM-HMM alignment & `ProsodyModelV12` on expanded `manifest_v12.json`.

Generates:
1. `exp_v12_prosody/gmm_alignments_v12.json`: zero-leakage 3-state GMM-HMM boundaries
   for all 780 EN training utterances (plus 20 held-out aligned via train-only model) and
   all 112 Mandarin utterances (80 new + 32 extended ref) for `@timbre-lab`'s `v12` bank.
2. `exp_v12_prosody/prosody_v12_model.pkl`: compact `< 500 KB` sklearn prosody predictor
   trained strictly on the expanded training set (with `slt[30:40]` & `bdl[30:40]` held out).
3. `exp_v12_prosody/results_v12_expanded.json`: before (`140` EN) vs after (`780` EN) comparison.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v11_align_gmm.gmm_hmm_aligner import EN_HELDOUT_DEFAULT
from exp_v12_prosody import DEFAULT_MODEL_PATH, EXP_DIR
from exp_v12_prosody.eval_v12_prosody import eval_applied_rhythm_on_heldout_texts, eval_cs_bridge_on_real_units
from exp_v12_prosody.prosody_model import fit_prosody_v12
from exp_v12_prosody.v12_corpus_align import MANIFEST_V12, align_en_v12_gmm, align_zh_v12_gmm


def _serialize_aligned_dict(entry: dict) -> dict:
    au = entry["aligned"]
    return {
        "utterance_id": au.utterance_id,
        "speaker": au.speaker,
        "language": au.language,
        "text": au.text,
        "path": entry.get("path", ""),
        "trim_start_sample": int(entry.get("trim_start_sample", 0)),
        "trim_stop_sample": int(entry.get("trim_stop_sample", 0)),
        "sr": au.sr,
        "hop_samples": au.hop_samples,
        "n_frames": au.n_frames,
        "phones": [asdict(p) for p in au.phones],
        "syllables": [
            {
                "index": s.index,
                "hanzi": s.hanzi,
                "pinyin": s.pinyin,
                "onset": s.onset,
                "final": s.final,
                "tone": s.tone,
                "syl_start": s.syl_start,
                "f_split": s.f_split,
                "syl_end": s.syl_end,
                "dur_ms": s.dur_ms,
            }
            for s in au.syllables
        ],
    }


def main() -> None:
    t0 = time.perf_counter()
    print("[1/3] Running GMM-HMM alignment over 800 EN utterances (780 train, 20 held-out)...", flush=True)
    en_res = align_en_v12_gmm(MANIFEST_V12, exclude_ids=EN_HELDOUT_DEFAULT, align_heldout=True, n_em=4)
    print(f"      EN aligned: train={len(en_res['utterances'])}, heldout={len(en_res['heldout_utterances'])} in {time.perf_counter()-t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    print("[2/3] Running GMM-HMM alignment over expanded Mandarin utterances (32 ref + 80 expanded)...", flush=True)
    zh_res = align_zh_v12_gmm(MANIFEST_V12, exclude_ids=None, n_em=4)
    print(f"      ZH aligned: {len(zh_res['utterances'])} utterances in {time.perf_counter()-t1:.1f}s", flush=True)

    # Export clean reusable JSON cache for @timbre-lab
    cache_payload = {
        "schema": 1,
        "excluded_en_heldout": sorted(EN_HELDOUT_DEFAULT),
        "en_train": {k: _serialize_aligned_dict(v) for k, v in en_res["utterances"].items()},
        "en_heldout": {k: _serialize_aligned_dict(v) for k, v in en_res["heldout_utterances"].items()},
        "zh_all": {k: _serialize_aligned_dict(v) for k, v in zh_res["utterances"].items()},
    }
    cache_path = EXP_DIR / "gmm_alignments_v12.json"
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False), encoding="utf-8")
    print(f"[3/3] Saved alignment cache {cache_path} ({cache_path.stat().st_size // 1024} KB)", flush=True)

    # Fit & compare Ridge vs HistGradientBoostingRegressor on expanded corpus
    _, hgbr_metrics = fit_prosody_v12(en_res=en_res, zh_res=zh_res, model_kind="hgbr", save_path=None)
    ridge_model, ridge_metrics = fit_prosody_v12(en_res=en_res, zh_res=zh_res, model_kind="ridge", save_path=DEFAULT_MODEL_PATH)

    out = {
        "en_train_utts": len(en_res["utterances"]),
        "en_heldout_utts": len(en_res["heldout_utterances"]),
        "zh_total_utts": len(zh_res["utterances"]),
        "expanded_ridge_heldout": ridge_metrics,
        "expanded_hgbr_heldout": hgbr_metrics,
        "expanded_plan_rhythm": eval_applied_rhythm_on_heldout_texts(en_res, ridge_model),
        "expanded_cs_bridge": eval_cs_bridge_on_real_units(zh_res, en_res),
    }
    res_path = EXP_DIR / "results_v12_expanded.json"
    res_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {res_path} in {time.perf_counter()-t0:.1f}s total")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
