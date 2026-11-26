"""Zero-leakage Phase 6 Evaluation: v9_combined vs v10_integrity vs Real (Loudness Jumps + Integrity + Clarity)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v5_eval import metrics as ev  # noqa: E402
from exp_v6_crossling_dsp import arbiter as v6_arb  # noqa: E402
from exp_v6_eval import fluency as fl  # noqa: E402
from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v8_clarity.clarity_metrics import compute_clarity_metrics  # noqa: E402
from exp_v8_clarity.en_utau_bank import _trim_active_span  # noqa: E402
from exp_v9_clarity_pro.hnm_v9 import analyze_v9  # noqa: E402
from exp_v9_clarity_pro.viterbi_utau import build_v9_dual_bank, render_v9_utterance  # noqa: E402
from exp_v9_timbre_v3.timbre_cloner_v3 import adapt_features_v3, extract_timbre_v3  # noqa: E402
from exp_v10_integrity import BENCHMARK_TEXTS, EN_MANIFEST, EXP_DIR, EXTENDED_INDEX, SAMPLES_DIR, SR  # noqa: E402
from exp_v10_integrity.integrity_gate import apply_integrity_gate_to_bank  # noqa: E402
from exp_v10_integrity.v10_utau_renderer import render_v10_utterance  # noqa: E402


def _loudness_norm(x: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    act = arr[np.abs(arr) > 0.01 * max(1e-6, float(np.max(np.abs(arr))))]
    r = float(np.sqrt(np.mean(act * act))) if act.size > 0 else float(np.sqrt(np.mean(arr * arr)))
    if r > 1e-8:
        arr = arr * (target_rms / r)
    pk = float(np.max(np.abs(arr))) if arr.size else 0.0
    if pk > 0.90:
        arr = arr * (0.90 / pk)
    return arr


def measure_loudness_jumps(x: np.ndarray, sr: int = SR) -> dict[str, float]:
    """Measure active 20 ms-step (40 ms window) and voiced-only RMS ΔdB jumps plus click spikes."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    fr = int(round(0.040 * sr))
    hop = int(round(0.020 * sr))
    n = 1 + max(0, len(arr) - fr) // hop
    if n < 3:
        return {"p95_db": 0.0, "p99_db": 0.0, "max_db": 0.0, "gt6db": 0, "gt9db": 0, "voiced_max_db": 0.0, "clicks_gt8": 0.0}
    rms = np.array([math.sqrt(float(np.mean(arr[i * hop : i * hop + fr] ** 2)) + 1e-12) for i in range(n)])
    db = 20.0 * np.log10(rms)
    act = db > (float(np.max(db)) - 26.0)
    pair = act[:-1] & act[1:]
    d_act = np.abs(np.diff(db))[pair] if np.any(pair) else np.array([0.0])

    tr = fl.compute_f0_track(arr, sr=sr, fmin=75.0, fmax=450.0)
    v = tr["voiced"]
    vi = np.array([bool(np.mean(v[min(len(v) - 1, i * 2) : min(len(v), i * 2 + 2)]) > 0.8) for i in range(n)])
    pv = pair & vi[:-1] & vi[1:]
    d_voi = np.abs(np.diff(db))[pv] if np.any(pv) else np.array([0.0])

    # Click outlier rate (>8x median 2nd diff)
    d2 = np.abs(np.diff(arr, n=2))
    med_d2 = max(1e-6, float(np.median(d2[np.abs(arr[1:-1]) > 0.01]))) if np.any(np.abs(arr[1:-1]) > 0.01) else 1e-5
    clicks = float(np.mean((d2 / med_d2) > 8.0)) if d2.size else 0.0

    return {
        "p95_db": float(np.percentile(d_act, 95)),
        "p99_db": float(np.percentile(d_act, 99)),
        "max_db": float(np.max(d_act)),
        "gt6db": int(np.sum(d_act > 6.0)),
        "gt9db": int(np.sum(d_act > 9.0)),
        "voiced_max_db": float(np.max(d_voi)),
        "clicks_gt8": clicks,
    }


def _eval_vs_ref(y: np.ndarray, ref: np.ndarray | None, rtf: float, min_phone_ms: float | None = None) -> dict:
    ev_y = ev.evaluate_audio(y, sr=SR)
    fl_y = fl.compute_fluency_metrics(y, sr=SR)
    cl_y = compute_clarity_metrics(y, sr=SR)
    lj = measure_loudness_jumps(y, sr=SR)
    out = {
        "sanity_ok": bool(ev_y["sanity"]["ok"]),
        "vuv_flux_db": float(fl_y["vuv_transitions"]["vuv_flux_median_db"]),
        "rtf": float(rtf),
        "min_phone_ms": min_phone_ms if min_phone_ms is not None else float("nan"),
        **cl_y,
        **lj,
    }
    if ref is not None:
        sd = ev.compute_spectral_distances(ref, y, sr=SR)
        out["mel_lsd_db"] = float(sd["mel_lsd_db"])
        out["mcd_db"] = float(sd["mcd_db"])
    return out


def _summarize(recs: list[dict]) -> dict:
    keys = [k for k in recs[0].keys() if k != "sanity_ok"]
    meds = {}
    for k in keys:
        vals = [r[k] for r in recs if r[k] == r[k]]
        meds[k] = round(float(np.median(vals)), 3) if vals else None
    return {
        "n": len(recs),
        "sanity_ok": sum(1 for r in recs if r["sanity_ok"]),
        "medians": meds,
        "sums": {
            "gt6db_total": sum(int(r["gt6db"]) for r in recs),
            "gt9db_total": sum(int(r["gt9db"]) for r in recs),
        },
    }


def run_v10_eval(export_samples: bool = True) -> dict:
    """Evaluate v9_combined vs v10_integrity across held-out EN (n=20), ZH (n=26), and CS sentences."""
    t0 = time.time()
    zh_utts = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
    man_all = json.loads(EN_MANIFEST.read_text(encoding="utf-8"))
    en_man = man_all["english"]["utterances"]
    zh_new_man = man_all.get("mandarin", {}).get("utterances", [])

    en_heldout = [u for u in en_man if u["speaker"] in ("slt", "bdl") and int(u["id"].split("_")[1]) >= 30]
    en_excl_ids = {u["id"] for u in en_heldout}

    print("Precomputing HD features (32 ZH + 140 EN)...", flush=True)
    zh_pre, zh_eval_cache = {}, {}
    for u in zh_utts:
        x, _ = vocoder.read_wav_mono(u["path"])
        xa = x[u["active_start"] : u["active_stop"]]
        f_v9, bg = analyze_v9(xa, sr=SR)
        zh_pre[u["id"]] = (xa, f_v9, bg)
        zh_eval_cache[u["id"]] = xa

    en_pre = {}
    en_train = [u for u in en_man if u["id"] not in en_excl_ids]
    for idx_e, u in enumerate(en_train, 1):
        p_wav = EN_MANIFEST.parent.parent.parent / u["path"]
        x, _ = vocoder.read_wav_mono(p_wav)
        s0, s1 = _trim_active_span(x, sr=SR)
        xa = x[s0:s1]
        f_v9, bg = analyze_v9(xa, sr=SR)
        en_pre[u["id"]] = (xa, f_v9, bg)
        if idx_e % 35 == 0 or idx_e == len(en_train):
            print(f"  Precomputed {idx_e}/{len(en_train)} EN train utts", flush=True)

    en_real_cache = {}
    for u in en_heldout:
        p_wav = EN_MANIFEST.parent.parent.parent / u["path"]
        x, _ = vocoder.read_wav_mono(p_wav)
        s0, s1 = _trim_active_span(x, sr=SR)
        en_real_cache[u["id"]] = x[s0:s1]

    # Build v9 full bank & v10 quarantined bank (zero leakage on EN held-out)
    b9_full = build_v9_dual_bank(
        zh_utterances=zh_utts, zh_exclude_ids=(), en_utterances=en_man, en_exclude_ids=en_excl_ids,
        zh_precomputed=zh_pre, en_precomputed=en_pre, extract_v3_profiles=True,
    )
    b10_full, gate_stats = apply_integrity_gate_to_bank(b9_full)
    print("Bank Quarantine Gate stats:", gate_stats, flush=True)

    # Extract strictly 3-shot TimbreProfileV3 for slt, bdl, SSB0502, SSB0073
    def _fewshot_v3(rows: list[dict], spk: str, corpus: str):
        items = []
        for r in rows[:3]:
            x, _ = vocoder.read_wav_mono(EN_MANIFEST.parent.parent.parent / r["path"])
            s0, s1 = _trim_active_span(x, sr=SR)
            f, _ = analyze_v9(x[s0:s1], sr=SR)
            items.append((x[s0:s1], f))
        return extract_timbre_v3(items, sr=SR, name=spk, corpus=corpus)

    v3_slt_3s = _fewshot_v3([u for u in en_man if u["speaker"] == "slt" and u["id"] not in en_excl_ids], "slt", "cmu_arctic")
    v3_bdl_3s = _fewshot_v3([u for u in en_man if u["speaker"] == "bdl" and u["id"] not in en_excl_ids], "bdl", "cmu_arctic")

    # 1. Held-out English n=20 (v9_combined vs v10_integrity vs Real)
    print("Running English n=20 (Real vs v9_v3 vs v10_v3)...", flush=True)
    en_recs = {k: [] for k in ("real_en", "v9_combined", "v10_integrity")}
    ab_en = {}
    for idx_h, u in enumerate(en_heldout, 1):
        uid, spk = u["id"], u["speaker"]
        xa = en_real_cache[uid]
        tgt_v3 = v3_slt_3s if spk == "slt" else v3_bdl_3s
        pl = cf.plan(u["text"], allow_oov_fallback=True)
        en_recs["real_en"].append(_eval_vs_ref(xa, xa, 0.0))

        t_s = time.perf_counter()
        y9, _ = render_v9_utterance(pl, b9_full, tgt_v3, cloner_fn=adapt_features_v3, seed=72)
        en_recs["v9_combined"].append(_eval_vs_ref(y9, xa, (time.perf_counter() - t_s) / max(1e-6, len(y9) / SR)))

        t_s = time.perf_counter()
        y10, m10 = render_v10_utterance(pl, b10_full, tgt_v3, cloner_fn=adapt_features_v3, seed=72)
        en_recs["v10_integrity"].append(
            _eval_vs_ref(y10, xa, (time.perf_counter() - t_s) / max(1e-6, len(y10) / SR), m10["integrity"]["min_en_phone_ms"])
        )

        if export_samples and uid in ("slt_0030", "bdl_0030"):
            ab_en[uid] = {"real": xa, "v9": y9, "v10": y10}
        if idx_h % 5 == 0 or idx_h == len(en_heldout):
            print(f"  EN {idx_h}/{len(en_heldout)} done", flush=True)

    # 2. Mandarin n=26 Held-out LOO-CV (zero leakage per fold)
    print("Running Mandarin n=26 LOO-CV (Real vs v9_v3 vs v10_v3)...", flush=True)
    from exp_v7_pybot_tts import unit_bank as ub
    from exp_v9_clarity_pro.viterbi_utau import V9DualBank

    zh_prompt_ids = {"SSB05020005", "SSB05020077", "SSB05020082", "SSB15930031", "SSB15930079", "SSB15930105"}
    zh_held_utts = [u for u in zh_utts if u["id"] not in zh_prompt_ids]
    zh_for_ub = {k: (v[0], v[1]) for k, v in zh_pre.items()}
    zh_recs = {k: [] for k in ("real_zh", "v9_combined", "v10_integrity")}
    ab_zh = {}

    for idx_z, u in enumerate(zh_held_utts, 1):
        uid, spk = u["id"], u["speaker"]
        xa = zh_eval_cache[uid]
        pl = v6_arb.plan_from_pinyin_syllables(u["syllables"], text=u["text"])
        zh_recs["real_zh"].append(_eval_vs_ref(xa, xa, 0.0))

        ub_fold = ub.build_unit_bank([x for x in zh_utts if x["id"] != uid], exclude_ids={uid}, precomputed=zh_for_ub, sr=SR)
        spk_items = [zh_for_ub[x["id"]] for x in zh_utts if x["speaker"] == spk and x["id"] != uid]
        t_v3 = extract_timbre_v3(spk_items, sr=SR, name=spk, corpus="aishell3")
        p3_fold = dict(b9_full.zh_profiles_v3)
        p3_fold[spk] = t_v3

        b9_f = V9DualBank(
            zh_syllables={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in b9_full.zh_syllables.items()) if v},
            zh_onsets={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in b9_full.zh_onsets.items()) if v},
            zh_finals={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in b9_full.zh_finals.items()) if v},
            en_phones={},
            zh_profiles_v2=b9_full.zh_profiles_v2,
            en_profiles_v2={},
            zh_profiles_v3=p3_fold,
            en_profiles_v3={},
            fallback_unit_bank=ub_fold,
            prosody_rules=b9_full.prosody_rules,
            zh_included_ids=[i for i in b9_full.zh_included_ids if i != uid],
            zh_excluded_ids=[uid],
        )
        b10_f = V9DualBank(
            zh_syllables={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in b10_full.zh_syllables.items()) if v},
            zh_onsets={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in b10_full.zh_onsets.items()) if v},
            zh_finals={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in b10_full.zh_finals.items()) if v},
            en_phones={},
            zh_profiles_v2=b10_full.zh_profiles_v2,
            en_profiles_v2={},
            zh_profiles_v3=p3_fold,
            en_profiles_v3={},
            fallback_unit_bank=ub_fold,
            prosody_rules=b10_full.prosody_rules,
            zh_included_ids=[i for i in b10_full.zh_included_ids if i != uid],
            zh_excluded_ids=[uid],
        )

        t_s = time.perf_counter()
        y9, _ = render_v9_utterance(pl, b9_f, t_v3, cloner_fn=adapt_features_v3, seed=72)
        zh_recs["v9_combined"].append(_eval_vs_ref(y9, xa, (time.perf_counter() - t_s) / max(1e-6, len(y9) / SR)))

        t_s = time.perf_counter()
        y10, _ = render_v10_utterance(pl, b10_f, t_v3, cloner_fn=adapt_features_v3, seed=72)
        zh_recs["v10_integrity"].append(_eval_vs_ref(y10, xa, (time.perf_counter() - t_s) / max(1e-6, len(y10) / SR)))

        if export_samples and uid in ("SSB05020122", "SSB15930193"):
            ab_zh[uid] = {"real": xa, "v9": y9, "v10": y10}
        if idx_z % 10 == 0 or idx_z == len(zh_held_utts):
            print(f"  ZH LOO {idx_z}/{len(zh_held_utts)} done", flush=True)

    # 3. Code-Switched Loudness & Integrity Check (benchmark_texts CS01..CS04)
    print("Running Code-Switched benchmark...", flush=True)
    btexts = json.loads(BENCHMARK_TEXTS.read_text(encoding="utf-8"))
    cs_texts = [t["text"] for t in btexts.get("sentences", []) if t.get("lang") == "zh_en"]
    if not cs_texts:
        cs_texts = ["今天 PyBotTTS 测试了 natural speech。"]
    cs_recs = {"v9_combined": [], "v10_integrity": []}
    for txt in cs_texts[:6]:
        pl_cs = cf.plan(txt, allow_oov_fallback=True)
        t_s = time.perf_counter()
        y9, _ = render_v9_utterance(pl_cs, b9_full, b9_full.zh_profiles_v3["SSB0502"], cloner_fn=adapt_features_v3, seed=72)
        cs_recs["v9_combined"].append(_eval_vs_ref(y9, None, (time.perf_counter() - t_s) / max(1e-6, len(y9) / SR)))

        t_s = time.perf_counter()
        y10, m10 = render_v10_utterance(pl_cs, b10_full, b10_full.zh_profiles_v3["SSB0502"], cloner_fn=adapt_features_v3, seed=72)
        cs_recs["v10_integrity"].append(
            _eval_vs_ref(y10, None, (time.perf_counter() - t_s) / max(1e-6, len(y10) / SR), m10["integrity"]["min_en_phone_ms"])
        )

    exported = []
    if export_samples:
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        gap = np.zeros(int(0.45 * SR), dtype=np.float64)
        for uid, c in ab_en.items():
            p = SAMPLES_DIR / f"AB_EN_{uid}_real_v9_v10.wav"
            vocoder.write_wav_mono(p, np.concatenate([_loudness_norm(c["real"]), gap, _loudness_norm(c["v9"]), gap, _loudness_norm(c["v10"])]), sr=SR)
            exported.append(p.name)
        for uid, c in ab_zh.items():
            p = SAMPLES_DIR / f"AB_ZH_{uid}_real_v9_v10.wav"
            vocoder.write_wav_mono(p, np.concatenate([_loudness_norm(c["real"]), gap, _loudness_norm(c["v9"]), gap, _loudness_norm(c["v10"])]), sr=SR)
            exported.append(p.name)

        cs_demo = cf.plan(cs_texts[0], allow_oov_fallback=True)
        y9_cs, _ = render_v9_utterance(cs_demo, b9_full, b9_full.zh_profiles_v3["SSB0502"], cloner_fn=adapt_features_v3, seed=72)
        y10_cs, _ = render_v10_utterance(cs_demo, b10_full, b10_full.zh_profiles_v3["SSB0502"], cloner_fn=adapt_features_v3, seed=72)
        p_cs1 = SAMPLES_DIR / "AB_CS_SSB0502_v9_vs_v10.wav"
        vocoder.write_wav_mono(p_cs1, np.concatenate([_loudness_norm(y9_cs), gap, _loudness_norm(y10_cs)]), sr=SR)
        exported.append(p_cs1.name)

        ssb0073_rows = [u for u in zh_new_man if u["speaker"] == "SSB0073"]
        if ssb0073_rows:
            v3_0073 = _fewshot_v3(ssb0073_rows, "SSB0073", "aishell3")
            y9_73, _ = render_v9_utterance(cs_demo, b9_full, v3_0073, cloner_fn=adapt_features_v3, seed=72)
            y10_73, _ = render_v10_utterance(cs_demo, b10_full, v3_0073, cloner_fn=adapt_features_v3, seed=72)
            p_cs2 = SAMPLES_DIR / "AB_CS_SSB0073_M133_v9_vs_v10.wav"
            vocoder.write_wav_mono(p_cs2, np.concatenate([_loudness_norm(y9_73), gap, _loudness_norm(y10_73)]), sr=SR)
            exported.append(p_cs2.name)

    out = {
        "schema": 1,
        "elapsed_s": round(time.time() - t0, 2),
        "quarantine_gate_stats": gate_stats,
        "en_heldout_n20": {m: _summarize(r) for m, r in en_recs.items()},
        "zh_loo_n26": {m: _summarize(r) for m, r in zh_recs.items()},
        "cs_eval": {m: _summarize(r) for m, r in cs_recs.items()},
        "exported_samples": exported,
    }
    (EXP_DIR / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run_v10_eval(export_samples=True)
    print(f"Completed v10 Integrity Eval in {res['elapsed_s']}s")
    print("Quarantine stats:", res["quarantine_gate_stats"])
    for sec in ("en_heldout_n20", "zh_loo_n26", "cs_eval"):
        print(f"\n=== {sec} ===")
        for m, info in res[sec].items():
            d = info["medians"]
            s = info["sums"]
            lsd = f" | Mel-LSD={d['mel_lsd_db']}" if "mel_lsd_db" in d else ""
            print(
                f"  [{m:15s}] ok={info['sanity_ok']}/{info['n']}{lsd} | contrast={d['spectral_contrast_db']} | uv_HF/LF={d['unvoiced_hf_lf_db']} | "
                f"p99_ΔdB={d['p99_db']} max_ΔdB={d['max_db']} (>6={s['gt6db_total']}, >9={s['gt9db_total']}) | voi_max={d['voiced_max_db']} | RTF={d['rtf']}"
            )
    print("Exported:", res["exported_samples"])
