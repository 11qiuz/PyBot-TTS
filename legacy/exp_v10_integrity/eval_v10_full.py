"""Fast Phase 6 Full Integrated Benchmark: plan_v10 + extract_verified_en_bank + TimbreClonerV4 + v10_utau_renderer."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v8_clarity.en_utau_bank import _trim_active_span  # noqa: E402
from exp_v9_clarity_pro.hnm_v9 import analyze_v9  # noqa: E402
from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10  # noqa: E402
from exp_v10_g2p_and_align.timbre_cloner_v4 import adapt_features_v4, extract_timbre_v4  # noqa: E402
from exp_v10_integrity import BENCHMARK_TEXTS, EN_MANIFEST, EXP_DIR, EXTENDED_INDEX, SAMPLES_DIR, SR  # noqa: E402
from exp_v10_integrity.eval_v10 import _eval_vs_ref, _loudness_norm, _summarize  # noqa: E402
from exp_v10_integrity.v10_utau_renderer import build_v10_full_bank, render_v10_utterance  # noqa: E402


def run_v10_full_eval(export_samples: bool = True) -> dict:
    """Evaluate integrated Phase-6 stack (plan_v10 + verified_en_bank + TimbreClonerV4 + v10_integrity)."""
    t0 = time.time()
    zh_utts = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
    man_all = json.loads(EN_MANIFEST.read_text(encoding="utf-8"))
    en_man = man_all["english"]["utterances"]
    zh_new_man = man_all.get("mandarin", {}).get("utterances", [])

    en_heldout = [u for u in en_man if u["speaker"] in ("slt", "bdl") and int(u["id"].split("_")[1]) >= 30]
    en_excl_ids = {u["id"] for u in en_heldout}

    print("Building integrated v10_full bank (cmudict G2P + landmark_aligner + v10 gate + V4)...", flush=True)
    b10_full, profs_v4, gate_stats = build_v10_full_bank(
        zh_utterances=zh_utts, zh_exclude_ids=(), en_exclude_ids=en_excl_ids, extract_v4_profiles=True,
    )

    def _fewshot_v4(rows: list[dict], spk: str, corpus: str):
        items = []
        for r in rows[:3]:
            x, _ = vocoder.read_wav_mono(EN_MANIFEST.parent.parent.parent / r["path"])
            s0, s1 = _trim_active_span(x, sr=SR)
            xa = x[s0:s1]
            f9, _ = analyze_v9(xa, sr=SR)
            items.append((xa, f9))
        return extract_timbre_v4(items, sr=SR, name=spk, corpus=corpus)

    v4_slt_3s = _fewshot_v4([u for u in en_man if u["speaker"] == "slt" and u["id"] not in en_excl_ids], "slt", "cmu_arctic")
    v4_bdl_3s = _fewshot_v4([u for u in en_man if u["speaker"] == "bdl" and u["id"] not in en_excl_ids], "bdl", "cmu_arctic")
    v4_0502 = profs_v4["SSB0502"]
    ssb0073_rows = [u for u in zh_new_man if u["speaker"] == "SSB0073"]
    v4_0073 = _fewshot_v4(ssb0073_rows, "SSB0073", "aishell3") if ssb0073_rows else None

    # 1. Held-out EN n=20
    print("Evaluating v10_full on EN n=20...", flush=True)
    en_recs = []
    ab_en = {}
    src_counts_en: dict[str, int] = {}
    for u in en_heldout:
        uid, spk = u["id"], u["speaker"]
        x, _ = vocoder.read_wav_mono(EN_MANIFEST.parent.parent.parent / u["path"])
        s0, s1 = _trim_active_span(x, sr=SR)
        xa = x[s0:s1]
        tgt = v4_slt_3s if spk == "slt" else v4_bdl_3s
        t_s = time.perf_counter()
        y10, m10 = render_v10_utterance(
            u["text"], b10_full, tgt, cloner_fn=adapt_features_v4, plan_fn=plan_v10, src_profiles=profs_v4, seed=72
        )
        en_recs.append(_eval_vs_ref(y10, xa, (time.perf_counter() - t_s) / max(1e-6, len(y10) / SR), m10["integrity"]["min_en_phone_ms"]))
        for k, v in m10["source_counts"].items():
            src_counts_en[k] = src_counts_en.get(k, 0) + v
        if export_samples and uid in ("slt_0030", "bdl_0030"):
            ab_en[uid] = (xa, y10)

    # 2. CS sentences from benchmark_texts.json
    btexts = json.loads(BENCHMARK_TEXTS.read_text(encoding="utf-8"))
    cs_texts = [t["text"] for t in btexts.get("sentences", []) if t.get("lang") == "zh_en"]
    cs_recs = []
    cs_first = None
    for idx_c, txt in enumerate(cs_texts):
        t_s = time.perf_counter()
        y_c, m_c = render_v10_utterance(
            txt, b10_full, v4_0502, cloner_fn=adapt_features_v4, plan_fn=plan_v10, src_profiles=profs_v4, seed=72
        )
        cs_recs.append(_eval_vs_ref(y_c, None, (time.perf_counter() - t_s) / max(1e-6, len(y_c) / SR), m_c["integrity"]["min_en_phone_ms"]))
        if idx_c == 0:
            cs_first = y_c

    # 3. Cross-lingual ZH -> EN (SSB0502 + SSB0073 Male Mandarin speaking EN)
    cross_recs = {}
    en_sample_5 = [u for u in en_heldout if u["speaker"] == "slt"][:5]
    for label, tgt in [("SSB0502_speaks_EN", v4_0502)] + ([("SSB0073_M133_speaks_EN", v4_0073)] if v4_0073 else []):
        rows = []
        for u in en_sample_5:
            t_s = time.perf_counter()
            yx, mx = render_v10_utterance(
                u["text"], b10_full, tgt, cloner_fn=adapt_features_v4, plan_fn=plan_v10, src_profiles=profs_v4, seed=72
            )
            rows.append(_eval_vs_ref(yx, None, (time.perf_counter() - t_s) / max(1e-6, len(yx) / SR), mx["integrity"]["min_en_phone_ms"]))
        cross_recs[label] = _summarize(rows)

    exported = []
    if export_samples:
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        gap = np.zeros(int(0.45 * SR), dtype=np.float64)
        for uid, (xa, y10) in ab_en.items():
            p = SAMPLES_DIR / f"AB_EN_{uid}_real_vs_v10full.wav"
            vocoder.write_wav_mono(p, np.concatenate([_loudness_norm(xa), gap, _loudness_norm(y10)]), sr=SR)
            exported.append(p.name)
        if cs_first is not None:
            p_cs = SAMPLES_DIR / "CS01_SSB0502_v10full.wav"
            vocoder.write_wav_mono(p_cs, _loudness_norm(cs_first), sr=SR)
            exported.append(p_cs.name)
        if v4_0073 is not None:
            y_cs73, _ = render_v10_utterance(
                cs_texts[0], b10_full, v4_0073, cloner_fn=adapt_features_v4, plan_fn=plan_v10, src_profiles=profs_v4, seed=72
            )
            p_cs73 = SAMPLES_DIR / "CS01_SSB0073_M133_v10full.wav"
            vocoder.write_wav_mono(p_cs73, _loudness_norm(y_cs73), sr=SR)
            exported.append(p_cs73.name)

    out = {
        "schema": 1,
        "elapsed_s": round(time.time() - t0, 2),
        "bank_gate_stats": gate_stats,
        "en_source_counts": src_counts_en,
        "en_heldout_n20_v10_full": _summarize(en_recs),
        "cs_eval_v10_full": _summarize(cs_recs),
        "cross_lingual_v10_full": cross_recs,
        "exported_samples": exported,
    }
    (EXP_DIR / "results_full.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run_v10_full_eval(export_samples=True)
    print(f"Completed Integrated v10_full Eval in {res['elapsed_s']}s")
    print("EN source counts:", res["en_source_counts"])
    for nm, info in [("en_heldout_n20_v10_full", res["en_heldout_n20_v10_full"]), ("cs_eval_v10_full", res["cs_eval_v10_full"])]:
        d, s = info["medians"], info["sums"]
        lsd = f" | Mel-LSD={d['mel_lsd_db']}" if "mel_lsd_db" in d else ""
        print(
            f"  [{nm:24s}] ok={info['sanity_ok']}/{info['n']}{lsd} | contrast={d['spectral_contrast_db']} | "
            f"uv_HF/LF={d['unvoiced_hf_lf_db']} | p99_ΔdB={d['p99_db']} max_ΔdB={d['max_db']} (>6={s['gt6db_total']}, >9={s['gt9db_total']}) | "
            f"min_ms={d['min_phone_ms']} | RTF={d['rtf']}"
        )
    print("Cross-lingual:", res["cross_lingual_v10_full"])
    print("Exported:", res["exported_samples"])
