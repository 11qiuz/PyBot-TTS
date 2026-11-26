"""Zero-leakage Phase 7 Evaluation: v10_integrity vs v11_diphone vs Real."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v6_crossling_dsp import arbiter as v6_arb  # noqa: E402
from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v7_pybot_tts import unit_bank as ub  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity  # noqa: E402
from exp_v8_clarity.en_utau_bank import _trim_active_span  # noqa: E402
from exp_v9_clarity_pro.hnm_v9 import analyze_v9  # noqa: E402
from exp_v9_clarity_pro.viterbi_utau import V9DualBank, build_v9_dual_bank  # noqa: E402
from exp_v9_timbre_v3.timbre_cloner_v3 import adapt_features_v3, extract_timbre_v3  # noqa: E402
from exp_v10_integrity.eval_v10 import _eval_vs_ref, _loudness_norm, _summarize  # noqa: E402
from exp_v10_integrity.integrity_gate import apply_integrity_gate_to_bank  # noqa: E402
from exp_v10_integrity.v10_utau_renderer import render_v10_utterance  # noqa: E402
from exp_v11_align_gmm.gmm_hmm_aligner import align_en_corpus_gmm, align_zh_corpus_gmm  # noqa: E402
from exp_v11_align_gmm.timbre_cloner_v5 import adapt_features_v5, extract_timbre_v5  # noqa: E402
from exp_v11_diphone_tts import BENCHMARK_TEXTS, EN_MANIFEST, EXP_DIR, EXTENDED_INDEX, SAMPLES_DIR, SR  # noqa: E402
from exp_v11_diphone_tts.diphone_bank import V11DiphoneDualBank, build_v11_dual_bank  # noqa: E402
from exp_v11_diphone_tts.v11_renderer import render_v11_utterance  # noqa: E402


def _seam_jump_from_info(info: dict) -> float:
    b_frames = info.get("boundary_frames") or []
    le = info.get("log_env")
    if le is None or not b_frames:
        return 0.0
    jumps = []
    for f in b_frames:
        fi = int(f)
        if 1 <= fi < len(le):
            diff = le[fi] - le[fi - 1]
            jumps.append(float(math.sqrt(float(np.mean(diff * diff)))))
    return float(np.mean(jumps)) if jumps else 0.0


def _slot_drop_stats(y: np.ndarray, info: dict) -> tuple[int, float]:
    slots = info.get("slot_timings_ms") or []
    if not slots:
        return 0, 0.0
    dropped = 0
    min_dur = 1e9
    for ms_s, ms_e in slots:
        dur = float(ms_e - ms_s)
        min_dur = min(min_dur, dur)
        s0 = max(0, int(round(ms_s * SR / 1000.0)))
        s1 = min(len(y), int(round(ms_e * SR / 1000.0)))
        if s1 <= s0:
            dropped += 1
            continue
        rms_db = 20.0 * float(np.log10(max(1e-8, float(np.sqrt(np.mean(y[s0:s1] ** 2))))))
        if dur < 50.0 or rms_db < -42.0:
            dropped += 1
    return dropped, (min_dur if min_dur < 1e8 else 0.0)


def _fold_v11_bank(bank: V11DiphoneDualBank, uid: str, ub_fold: ub.UnitBank, spk: str, v5_fold) -> V11DiphoneDualBank:
    profs5 = dict(bank.profiles_v5)
    profs5[spk] = v5_fold
    return V11DiphoneDualBank(
        en_diphones=bank.en_diphones,
        en_phones=bank.en_phones,
        zh_syllables={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in bank.zh_syllables.items()) if v},
        zh_onsets={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in bank.zh_onsets.items()) if v},
        zh_finals={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in bank.zh_finals.items()) if v},
        profiles_v2=bank.profiles_v2,
        profiles_v5=profs5,
        fallback_unit_bank=ub_fold,
        zh_included_ids=[i for i in bank.zh_included_ids if i != uid],
        zh_excluded_ids=sorted(set(bank.zh_excluded_ids) | {uid}),
        en_included_ids=list(bank.en_included_ids),
        en_excluded_ids=list(bank.en_excluded_ids),
        stats=dict(bank.stats),
    )


def run_v11_eval(export_samples: bool = True) -> dict:
    t0 = time.time()
    zh_utts = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
    man_all = json.loads(EN_MANIFEST.read_text(encoding="utf-8"))
    en_man = man_all["english"]["utterances"]
    zh_new_man = man_all.get("mandarin", {}).get("utterances", [])

    en_heldout = [u for u in en_man if u["speaker"] in ("slt", "bdl") and int(u["id"].split("_")[-1]) >= 30]
    en_excl_ids = {u["id"] for u in en_heldout}

    print("Running GMM-HMM alignments (140 EN zero-leakage + 32 ZH)...", flush=True)
    en_gmm = align_en_corpus_gmm(exclude_ids=en_excl_ids, align_heldout=False, n_em=4, sr=SR)
    zh_gmm = align_zh_corpus_gmm(exclude_ids=None, n_em=4, sr=SR)

    print("Precomputing v9-style acoustic cache (32 ZH + 140 EN)...", flush=True)
    zh_pre, zh_real_cache = {}, {}
    for u in zh_utts:
        x, _ = vocoder.read_wav_mono(u["path"])
        xa = x[u["active_start"] : u["active_stop"]]
        f_v9, bg = analyze_v9(xa, sr=SR)
        zh_pre[u["id"]] = (xa, f_v9, bg)
        zh_real_cache[u["id"]] = xa

    en_pre = {}
    en_train = [u for u in en_man if u["id"] not in en_excl_ids]
    for u in en_train:
        g_e = en_gmm["utterances"][u["id"]]
        xa = g_e["wav"]
        f_v9, bg = analyze_v9(xa, sr=SR)
        en_pre[u["id"]] = (xa, f_v9, bg)

    en_real_cache = {}
    for u in en_heldout:
        p_w = REPO_ROOT / u["path"]
        x, _ = vocoder.read_wav_mono(p_w)
        s0, s1 = _trim_active_span(x, sr=SR)
        en_real_cache[u["id"]] = x[s0:s1]

    # Build v10 baseline bank (also reusing raw trimmed EN for v10)
    en_pre_v10 = {}
    for u in en_train:
        x, _ = vocoder.read_wav_mono(REPO_ROOT / u["path"])
        s0, s1 = _trim_active_span(x, sr=SR)
        xa_raw = x[s0:s1]
        f9_r, bg_r = analyze_v9(xa_raw, sr=SR)
        en_pre_v10[u["id"]] = (xa_raw, f9_r, bg_r)

    b9_full = build_v9_dual_bank(
        zh_utterances=zh_utts, zh_exclude_ids=(), en_utterances=en_man, en_exclude_ids=en_excl_ids,
        zh_precomputed=zh_pre, en_precomputed=en_pre_v10, extract_v3_profiles=True,
    )
    b10_full, _ = apply_integrity_gate_to_bank(b9_full)

    # Build v11 Diphone + Whole-Syllable bank
    b11_full = build_v11_dual_bank(
        en_gmm_alignments=en_gmm,
        zh_gmm_alignments=zh_gmm,
        zh_utterances=zh_utts,
        zh_exclude_ids=(),
        en_utterances=en_man,
        en_exclude_ids=en_excl_ids,
        zh_precomputed=zh_pre,
        en_precomputed=en_pre,
        extract_v5_profiles=True,
    )
    print("v11 Dual Bank stats:", b11_full.stats, flush=True)

    # Strictly 3-shot profiles (slt, bdl from train[0:3]; SSB0073 from mandarin[0:3])
    def _fewshot(rows: list[dict], spk: str, corpus: str):
        items = []
        for r in rows[:3]:
            p = Path(r["path"])
            if not p.is_absolute():
                p = REPO_ROOT / p
            x, _ = vocoder.read_wav_mono(p)
            s0, s1 = _trim_active_span(x, sr=SR)
            f, _ = analyze_v9(x[s0:s1], sr=SR)
            items.append((x[s0:s1], f))
        return extract_timbre_v3(items, sr=SR, name=spk, corpus=corpus), extract_timbre_v5(items, sr=SR, name=spk, corpus=corpus)

    slt_rows = [u for u in en_man if u["speaker"] == "slt" and u["id"] not in en_excl_ids]
    bdl_rows = [u for u in en_man if u["speaker"] == "bdl" and u["id"] not in en_excl_ids]
    v3_slt_3s, v5_slt_3s = _fewshot(slt_rows, "slt", "arctic")
    v3_bdl_3s, v5_bdl_3s = _fewshot(bdl_rows, "bdl", "arctic")
    v3_slt_v10 = extract_timbre_v3([(en_pre_v10[r["id"]][0], en_pre_v10[r["id"]][1]) for r in slt_rows[:3]], sr=SR, name="slt", corpus="cmu_arctic")
    v3_bdl_v10 = extract_timbre_v3([(en_pre_v10[r["id"]][0], en_pre_v10[r["id"]][1]) for r in bdl_rows[:3]], sr=SR, name="bdl", corpus="cmu_arctic")

    # 1. Held-out English n=20 (Real vs v10 vs v11)
    print("Evaluating English held-out n=20 (Real vs v10 vs v11)...", flush=True)
    en_recs = {k: [] for k in ("real_en", "v10_integrity", "v11_diphone")}
    en_extra = {"diphone_exact_pct": [], "phone_fallback_cnt": [], "synth_fill_cnt": [], "seam_jump": []}
    ab_en = {}
    for u in en_heldout:
        uid, spk = u["id"], u["speaker"]
        xa = en_real_cache[uid]
        tgt_v10 = v3_slt_v10 if spk == "slt" else v3_bdl_v10
        tgt_v11 = v5_slt_3s if spk == "slt" else v5_bdl_3s
        pl = cf.plan(u["text"], allow_oov_fallback=True)
        en_recs["real_en"].append(_eval_vs_ref(xa, xa, 0.0))

        ts = time.perf_counter()
        y10, m10 = render_v10_utterance(pl, b10_full, tgt_v10, cloner_fn=adapt_features_v3, seed=72)
        en_recs["v10_integrity"].append(_eval_vs_ref(y10, xa, (time.perf_counter() - ts) / max(1e-6, len(y10) / SR), m10["integrity"]["min_en_phone_ms"]))

        ts = time.perf_counter()
        y11, m11 = render_v11_utterance(pl, b11_full, tgt_v11, cloner_fn=adapt_features_v5, seed=72)
        rec11 = _eval_vs_ref(y11, xa, (time.perf_counter() - ts) / max(1e-6, len(y11) / SR))
        rec11["seam_jump"] = round(_seam_jump_from_info(m11), 4)
        en_recs["v11_diphone"].append(rec11)
        en_extra["diphone_exact_pct"].append(float(m11["en_diphone_exact_pct"] or 0.0))
        en_extra["phone_fallback_cnt"].append(int(m11["source_counts"].get("en_phone_fallback", 0)))
        en_extra["synth_fill_cnt"].append(int(m11["source_counts"].get("en_synth_fill", 0)))
        en_extra["seam_jump"].append(rec11["seam_jump"])

        if uid in ("slt_0030", "bdl_0030"):
            ab_en[uid] = {"real": xa, "v10": y10, "v11": y11}

    # 2. Mandarin n=26 Held-out LOO-CV (zero leakage per fold; strict LOO GMM on SSB05020122 too!)
    print("Evaluating Mandarin LOO-CV n=26 (Real vs v10 vs v11)...", flush=True)
    zh_prompt_ids = {"SSB05020005", "SSB05020077", "SSB05020082", "SSB15930031", "SSB15930079", "SSB15930105"}
    zh_held_utts = [u for u in zh_utts if u["id"] not in zh_prompt_ids]
    zh_for_ub = {k: (v[0], v[1]) for k, v in zh_pre.items()}
    zh_recs = {k: [] for k in ("real_zh", "v10_integrity", "v11_diphone")}
    zh_v11_drops, zh_v11_whole_pcts = [], []
    ab_zh, ssb05020122_diag = {}, {}

    for u in zh_held_utts:
        uid, spk = u["id"], u["speaker"]
        xa = zh_real_cache[uid]
        pl = v6_arb.plan_from_pinyin_syllables(u["syllables"], text=u["text"])
        zh_recs["real_zh"].append(_eval_vs_ref(xa, xa, 0.0))

        ub_fold = ub.build_unit_bank([x for x in zh_utts if x["id"] != uid], exclude_ids={uid}, precomputed=zh_for_ub, sr=SR)
        spk_items = [zh_for_ub[x["id"]] for x in zh_utts if x["speaker"] == spk and x["id"] != uid]
        t_v3 = extract_timbre_v3(spk_items, sr=SR, name=spk, corpus="aishell3")
        t_v5 = extract_timbre_v5(spk_items, sr=SR, name=spk, corpus="aishell3")
        p3_fold = dict(b10_full.zh_profiles_v3)
        p3_fold[spk] = t_v3
        b10_f = V9DualBank(
            zh_syllables={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in b10_full.zh_syllables.items()) if v},
            zh_onsets={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in b10_full.zh_onsets.items()) if v},
            zh_finals={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in b10_full.zh_finals.items()) if v},
            en_phones={}, zh_profiles_v2=b10_full.zh_profiles_v2, en_profiles_v2={},
            zh_profiles_v3=p3_fold, en_profiles_v3={}, fallback_unit_bank=ub_fold,
            prosody_rules=b10_full.prosody_rules, zh_included_ids=[i for i in b10_full.zh_included_ids if i != uid], zh_excluded_ids=[uid],
        )

        if uid == "SSB05020122":
            # Strict full LOO rebuild excluding SSB05020122 even from GMM alignment
            b11_f = build_v11_dual_bank(
                en_gmm_alignments=en_gmm,
                zh_utterances=zh_utts,
                zh_exclude_ids=["SSB05020122"],
                en_utterances=en_man,
                en_exclude_ids=en_excl_ids,
                zh_precomputed=zh_pre,
                en_precomputed=en_pre,
                extract_v5_profiles=True,
            )
            b11_f.profiles_v5[spk] = t_v5
        else:
            b11_f = _fold_v11_bank(b11_full, uid, ub_fold, spk, t_v5)

        ts = time.perf_counter()
        y10, _ = render_v10_utterance(pl, b10_f, t_v3, cloner_fn=adapt_features_v3, seed=72)
        zh_recs["v10_integrity"].append(_eval_vs_ref(y10, xa, (time.perf_counter() - ts) / max(1e-6, len(y10) / SR)))

        ts = time.perf_counter()
        y11, m11 = render_v11_utterance(pl, b11_f, t_v5, cloner_fn=adapt_features_v5, seed=72)
        drops, min_d = _slot_drop_stats(y11, m11)
        zh_v11_drops.append(drops)
        sc = m11["source_counts"]
        n_zh_w = sc.get("zh_target_syl", 0) + sc.get("zh_donor_syl", 0)
        n_zh_tot = n_zh_w + sc.get("zh_concat_of", 0) + sc.get("zh_ipa_fill", 0) + sc.get("zh_synth_fill", 0)
        zh_v11_whole_pcts.append(100.0 * n_zh_w / max(1, n_zh_tot))

        rec11 = _eval_vs_ref(y11, xa, (time.perf_counter() - ts) / max(1e-6, len(y11) / SR), min_d)
        rec11["seam_jump"] = round(_seam_jump_from_info(m11), 4)
        zh_recs["v11_diphone"].append(rec11)

        if uid == "SSB05020122":
            ab_zh[uid] = {"real": xa, "v10": y10, "v11": y11}
            slot_rows = []
            for k_s, (ms_s, ms_e) in enumerate(m11["slot_timings_ms"]):
                s0 = max(0, int(round(ms_s * SR / 1000.0)))
                s1 = min(len(y11), int(round(ms_e * SR / 1000.0)))
                rms_db = 20.0 * float(np.log10(max(1e-8, float(np.sqrt(np.mean(y11[s0:s1] ** 2))))))
                slot_rows.append({"idx": k_s, "hanzi": u["syllables"][k_s]["hanzi"], "pinyin": u["syllables"][k_s]["pinyin"], "dur_ms": round(ms_e - ms_s, 1), "rms_db": round(rms_db, 1)})
            ssb05020122_diag = {"source_counts": sc, "dropped_slots": drops, "min_dur_ms": round(min_d, 1), "slots": slot_rows}

    # 3. Code-Switched & Cross-lingual (CS_01..CS_06 + SSB0073_M133)
    print("Evaluating Code-Switched CS01..CS06 & SSB0073_M133...", flush=True)
    btexts = json.loads(BENCHMARK_TEXTS.read_text(encoding="utf-8"))
    cs_texts = [t["text"] for t in btexts.get("sentences", []) if t.get("lang") == "zh_en"][:6]
    cs_recs = {"v10_integrity": [], "v11_diphone": []}
    cs_exact_pcts = []
    for txt in cs_texts:
        pl_cs = cf.plan(txt, allow_oov_fallback=True)
        ts = time.perf_counter()
        y10, m10 = render_v10_utterance(pl_cs, b10_full, b10_full.zh_profiles_v3["SSB0502"], cloner_fn=adapt_features_v3, seed=72)
        cs_recs["v10_integrity"].append(_eval_vs_ref(y10, None, (time.perf_counter() - ts) / max(1e-6, len(y10) / SR), m10["integrity"]["min_en_phone_ms"]))

        ts = time.perf_counter()
        y11, m11 = render_v11_utterance(pl_cs, b11_full, b11_full.profiles_v5["SSB0502"], cloner_fn=adapt_features_v5, seed=72)
        rec11 = _eval_vs_ref(y11, None, (time.perf_counter() - ts) / max(1e-6, len(y11) / SR))
        rec11["seam_jump"] = round(_seam_jump_from_info(m11), 4)
        cs_recs["v11_diphone"].append(rec11)
        if m11["en_diphone_exact_pct"] is not None:
            cs_exact_pcts.append(float(m11["en_diphone_exact_pct"]))

    # Export WAV samples and run sanity checks
    exported = {}
    if export_samples:
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        gap = np.zeros(int(0.45 * SR), dtype=np.float64)

        def _write_checked(fname: str, wav: np.ndarray) -> None:
            p = SAMPLES_DIR / fname
            vocoder.write_wav_mono(p, wav, sr=SR)
            exported[fname] = check_wav_sanity(wav, SR)

        for uid, c in ab_en.items():
            _write_checked(
                f"AB_EN_{uid}_real_v10_v11.wav",
                np.concatenate([_loudness_norm(c["real"]), gap, _loudness_norm(c["v10"]), gap, _loudness_norm(c["v11"])]),
            )
        for uid, c in ab_zh.items():
            _write_checked(
                f"AB_ZH_{uid}_real_v10_v11.wav",
                np.concatenate([_loudness_norm(c["real"]), gap, _loudness_norm(c["v10"]), gap, _loudness_norm(c["v11"])]),
            )

        cs0_pl = cf.plan(cs_texts[0], allow_oov_fallback=True)
        y10_cs0, _ = render_v10_utterance(cs0_pl, b10_full, b10_full.zh_profiles_v3["SSB0502"], cloner_fn=adapt_features_v3, seed=72)
        y11_cs0, _ = render_v11_utterance(cs0_pl, b11_full, b11_full.profiles_v5["SSB0502"], cloner_fn=adapt_features_v5, seed=72)
        _write_checked("CS01_SSB0502_v11.wav", _loudness_norm(y11_cs0))
        _write_checked("AB_CS_SSB0502_v10_vs_v11.wav", np.concatenate([_loudness_norm(y10_cs0), gap, _loudness_norm(y11_cs0)]))

        ssb0073_rows = [u for u in zh_new_man if u["speaker"] == "SSB0073"]
        if ssb0073_rows:
            v3_73, v5_73 = _fewshot(ssb0073_rows, "SSB0073", "aishell3")
            y10_73, _ = render_v10_utterance(cs0_pl, b10_full, v3_73, cloner_fn=adapt_features_v3, seed=72)
            y11_73, _ = render_v11_utterance(cs0_pl, b11_full, v5_73, cloner_fn=adapt_features_v5, seed=72)
            _write_checked("CS01_SSB0073_M133_v11.wav", _loudness_norm(y11_73))
            _write_checked("AB_CS_SSB0073_M133_v10_vs_v11.wav", np.concatenate([_loudness_norm(y10_73), gap, _loudness_norm(y11_73)]))

    out = {
        "schema": 1,
        "elapsed_s": round(time.time() - t0, 2),
        "bank_stats": b11_full.stats,
        "en_heldout_n20": {m: _summarize(r) for m, r in en_recs.items()},
        "en_v11_diphone_telemetry": {
            "diphone_exact_pct_mean": round(float(np.mean(en_extra["diphone_exact_pct"])), 2),
            "diphone_exact_pct_median": round(float(np.median(en_extra["diphone_exact_pct"])), 2),
            "phone_fallback_total": sum(en_extra["phone_fallback_cnt"]),
            "synth_fill_total": sum(en_extra["synth_fill_cnt"]),
            "seam_jump_median": round(float(np.median(en_extra["seam_jump"])), 4),
        },
        "zh_loo_n26": {m: _summarize(r) for m, r in zh_recs.items()},
        "zh_v11_telemetry": {
            "whole_syl_pct_mean": round(float(np.mean(zh_v11_whole_pcts)), 2),
            "dropped_slots_total": sum(zh_v11_drops),
        },
        "ssb05020122_loo_diag": ssb05020122_diag,
        "cs_eval": {m: _summarize(r) for m, r in cs_recs.items()},
        "cs_v11_diphone_exact_pct_mean": round(float(np.mean(cs_exact_pcts)), 2) if cs_exact_pcts else None,
        "exported_samples": exported,
    }
    (EXP_DIR / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run_v11_eval(export_samples=True)
    print(f"Completed v11 Diphone Eval in {res['elapsed_s']}s")
    print("EN diphone telemetry:", res["en_v11_diphone_telemetry"])
    print("ZH v11 telemetry:", res["zh_v11_telemetry"])
    print("SSB05020122 diag:", res["ssb05020122_loo_diag"].get("source_counts"), "drops=", res["ssb05020122_loo_diag"].get("dropped_slots"))
    for sec in ("en_heldout_n20", "zh_loo_n26", "cs_eval"):
        print(f"\n=== {sec} ===")
        for m, info in res[sec].items():
            d = info["medians"]
            s = info["sums"]
            lsd = f" | Mel-LSD={d['mel_lsd_db']}" if "mel_lsd_db" in d else ""
            print(
                f"  [{m:14s}] ok={info['sanity_ok']}/{info['n']}{lsd} | contrast={d['spectral_contrast_db']} | "
                f"p99_ΔdB={d['p99_db']} max_ΔdB={d['max_db']} (>6={s['gt6db_total']}, >9={s['gt9db_total']}) | RTF={d['rtf']}"
            )
    print("Exported samples sanity:", {k: v["ok"] for k, v in res["exported_samples"].items()})
