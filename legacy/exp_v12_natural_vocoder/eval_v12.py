"""Zero-leakage Phase 8 Evaluation: v11_diphone vs v12_integrated vs Real + A/B Audio Export."""
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
from exp_v5_eval import metrics as evm  # noqa: E402
from exp_v6_crossling_dsp import arbiter as v6_arb  # noqa: E402
from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v7_pybot_tts import unit_bank as ub  # noqa: E402
from exp_v8_clarity.en_utau_bank import _trim_active_span  # noqa: E402
from exp_v8_data.clarity_diag import compute_clarity_metrics as compute_clarity_diag  # noqa: E402
from exp_v9_clarity_pro.hnm_v9 import analyze_v9  # noqa: E402
from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10  # noqa: E402
from exp_v10_integrity.eval_v10 import _eval_vs_ref, _loudness_norm, _summarize  # noqa: E402
from exp_v11_align_gmm.gmm_hmm_aligner import align_en_corpus_gmm, align_zh_corpus_gmm  # noqa: E402
from exp_v11_align_gmm.timbre_cloner_v5 import adapt_features_v5, extract_timbre_v5  # noqa: E402
from exp_v11_diphone_tts.diphone_bank import V11DiphoneDualBank, build_v11_dual_bank  # noqa: E402
from exp_v11_diphone_tts.v11_renderer import EN_FUNCTION_WORDS, render_v11_utterance  # noqa: E402
from exp_v12_natural_vocoder import (  # noqa: E402
    BENCHMARK_TEXTS,
    EN_MANIFEST,
    EXP_DIR,
    EXTENDED_INDEX,
    SAMPLES_DIR,
    SR,
    V12_MANIFEST,
)
from exp_v12_natural_vocoder.v12_bank import build_v12_dual_bank, filter_v12_bank_excluding_utterance  # noqa: E402
from exp_v12_natural_vocoder.v12_renderer import render_v12_utterance  # noqa: E402
from exp_v12_natural_vocoder.vocoder_v12 import analyze_v12, build_glottal_pulse_bank  # noqa: E402
from exp_v12_prosody.prosody_model import DEFAULT_MODEL_PATH, load_prosody_v12  # noqa: E402


def _eval_v12_extra(y: np.ndarray, ref: np.ndarray | None, rtf: float, info: dict | None = None) -> dict:
    rec = _eval_vs_ref(y, ref, rtf)
    cd = compute_clarity_diag(y, sr=SR)
    rec["uv_2_8k_vs_lo_db"] = cd["unvoiced_2_8k_vs_lo_db"] if cd["unvoiced_2_8k_vs_lo_db"] is not None else float("nan")
    if info is not None and info.get("boundary_frames") and info.get("log_env") is not None:
        le = info["log_env"]
        jumps = []
        for fi in info["boundary_frames"]:
            f = int(fi)
            if 1 <= f < len(le):
                d = le[f] - le[f - 1]
                jumps.append(float(math.sqrt(float(np.mean(d * d)))))
        rec["seam_jump"] = round(float(np.mean(jumps)), 4) if jumps else 0.0
    return rec


def _en_rhythm_from_slots(y: np.ndarray, plan_doc: dict, info: dict) -> tuple[float | None, float | None]:
    units = plan_doc.get("units") or []
    slots = info.get("slot_timings_ms") or []
    if len(units) != len(slots):
        return None, None
    c_dur, f_dur, c_db, f_db = [], [], [], []
    for u, (ms0, ms1) in zip(units, slots):
        if u.get("lang", "zh") != "en":
            continue
        tok = str(u.get("token") or u.get("text") or "").strip().lower().strip(".,!?;:\"'")
        s0, s1 = max(0, int(round(ms0 * SR / 1000.0))), min(len(y), int(round(ms1 * SR / 1000.0)))
        if s1 <= s0:
            continue
        rms = 20.0 * float(np.log10(max(1e-8, float(np.sqrt(np.mean(y[s0:s1] ** 2))))))
        dur = float(ms1 - ms0)
        if tok in EN_FUNCTION_WORDS:
            f_dur.append(dur); f_db.append(rms)
        else:
            c_dur.append(dur); c_db.append(rms)
    ratio = (float(np.mean(f_dur)) / max(1e-3, float(np.mean(c_dur)))) if (f_dur and c_dur) else None
    diff_db = (float(np.mean(c_db)) - float(np.mean(f_db))) if (c_db and f_db) else None
    return ratio, diff_db


def _slot_drop_count(y: np.ndarray, info: dict) -> int:
    slots = info.get("slot_timings_ms") or []
    drops = 0
    for ms0, ms1 in slots:
        dur = float(ms1 - ms0)
        s0, s1 = max(0, int(round(ms0 * SR / 1000.0))), min(len(y), int(round(ms1 * SR / 1000.0)))
        if s1 <= s0 or dur < 50.0:
            drops += 1
            continue
        rms_db = 20.0 * float(np.log10(max(1e-8, float(np.sqrt(np.mean(y[s0:s1] ** 2))))))
        if rms_db < -42.0:
            drops += 1
    return drops


def run_v12_eval(export_samples: bool = True) -> dict:
    t0 = time.time()
    zh_utts = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
    man_v8 = json.loads(EN_MANIFEST.read_text(encoding="utf-8"))
    man_v12 = json.loads(V12_MANIFEST.read_text(encoding="utf-8"))
    en_man_v8 = man_v8["english"]["utterances"]
    en_man_v12 = man_v12["english"]["utterances"]
    zh_new_man = man_v8.get("mandarin", {}).get("utterances", [])

    en_heldout = [u for u in en_man_v12 if u["speaker"] in ("slt", "bdl") and 30 <= int(u["id"].split("_")[-1]) < 40]
    en_excl_ids = {u["id"] for u in en_heldout}

    print("Building v11 baseline bank (140 EN + 32 ZH)...", flush=True)
    en_gmm_v11 = align_en_corpus_gmm(exclude_ids=en_excl_ids, align_heldout=False, n_em=4, sr=SR)
    zh_gmm_v11 = align_zh_corpus_gmm(exclude_ids=None, n_em=4, sr=SR)

    zh_pre, zh_real_cache = {}, {}
    for u in zh_utts:
        x, _ = vocoder.read_wav_mono(u["path"])
        xa = x[u["active_start"] : u["active_stop"]]
        f_v9, bg = analyze_v9(xa, sr=SR)
        zh_pre[u["id"]] = (xa, f_v9, bg)
        zh_real_cache[u["id"]] = xa

    en_pre_v11 = {}
    for u in en_man_v8:
        if u["id"] in en_excl_ids:
            continue
        ge = en_gmm_v11["utterances"][u["id"]]
        f_v9, bg = analyze_v9(ge["wav"], sr=SR)
        en_pre_v11[u["id"]] = (ge["wav"], f_v9, bg)

    b11_full = build_v11_dual_bank(
        en_gmm_alignments=en_gmm_v11,
        zh_gmm_alignments=zh_gmm_v11,
        zh_utterances=zh_utts,
        en_utterances=en_man_v8,
        en_exclude_ids=en_excl_ids,
        zh_precomputed=zh_pre,
        en_precomputed=en_pre_v11,
        extract_v5_profiles=True,
    )

    print("Building v12 expanded bank (780 EN + 112 ZH + GlottalPulseBanks)...", flush=True)
    b12_full = build_v12_dual_bank(
        manifest_path=V12_MANIFEST,
        en_exclude_ids=en_excl_ids,
        extract_v5_profiles=True,
    )
    prosody_model = load_prosody_v12(DEFAULT_MODEL_PATH)
    print("v12 Bank stats:", b12_full.stats, flush=True)

    en_real_cache = {}
    for idx_e, u in enumerate(en_heldout, 1):
        x, _ = vocoder.read_wav_mono(REPO_ROOT / u["path"])
        s0, s1 = _trim_active_span(x, sr=SR)
        en_real_cache[u["id"]] = x[s0:s1]

    # Strictly 3-shot TimbreProfileV5 + GlottalPulseBank from training [0:3]
    def _fewshot_v5_and_gb(rows: list[dict], spk: str, corpus: str):
        items_v5, items_v12 = [], []
        for r in rows[:3]:
            p = Path(r["path"])
            if not p.is_absolute():
                p = REPO_ROOT / p
            x, _ = vocoder.read_wav_mono(p)
            s0, s1 = _trim_active_span(x, sr=SR)
            xa = x[s0:s1]
            fv = analyze_v12(xa, sr=SR)
            items_v5.append((xa, fv.feats))
            items_v12.append(fv)
        return extract_timbre_v5(items_v5, sr=SR, name=spk, corpus=corpus), build_glottal_pulse_bank(spk, items_v12, max_pulses=24)

    slt_rows = [u for u in en_man_v12 if u["speaker"] == "slt" and u["id"] not in en_excl_ids]
    bdl_rows = [u for u in en_man_v12 if u["speaker"] == "bdl" and u["id"] not in en_excl_ids]
    v5_slt_3s, gb_slt_3s = _fewshot_v5_and_gb(slt_rows, "slt", "arctic")
    v5_bdl_3s, gb_bdl_3s = _fewshot_v5_and_gb(bdl_rows, "bdl", "arctic")

    # 1. Held-out English n=20 (Real vs v11 vs v12)
    print("Evaluating English held-out n=20 (Real vs v11 vs v12)...", flush=True)
    en_recs = {k: [] for k in ("real_en", "v11_diphone", "v12_integrated")}
    en_tel = {
        "v11_exact_pct": [], "v12_exact_pct": [],
        "v11_fallback_tot": 0, "v12_fallback_tot": 0,
        "v11_func_dur_r": [], "v12_func_dur_r": [],
        "v11_c_minus_f_db": [], "v12_c_minus_f_db": [],
    }
    ab_en = {}
    for idx_e, u in enumerate(en_heldout, 1):
        uid, spk = u["id"], u["speaker"]
        xa = en_real_cache[uid]
        tgt_p = v5_slt_3s if spk == "slt" else v5_bdl_3s
        tgt_gb = gb_slt_3s if spk == "slt" else gb_bdl_3s
        pl = plan_v10(u["text"])
        en_recs["real_en"].append(_eval_v12_extra(xa, xa, 0.0))

        ts = time.perf_counter()
        y11, m11 = render_v11_utterance(pl, b11_full, tgt_p, cloner_fn=adapt_features_v5, seed=72)
        en_recs["v11_diphone"].append(_eval_v12_extra(y11, xa, (time.perf_counter() - ts) / max(1e-6, len(y11) / SR), m11))
        en_tel["v11_exact_pct"].append(float(m11["en_diphone_exact_pct"] or 0.0))
        en_tel["v11_fallback_tot"] += int(m11["source_counts"].get("en_phone_fallback", 0))
        r11, d11 = _en_rhythm_from_slots(y11, pl, m11)
        if r11 is not None: en_tel["v11_func_dur_r"].append(r11)
        if d11 is not None: en_tel["v11_c_minus_f_db"].append(d11)

        ts = time.perf_counter()
        y12, m12 = render_v12_utterance(
            pl, b12_full, tgt_p, prosody_model=prosody_model, glottal_bank=tgt_gb, cloner_fn=adapt_features_v5, seed=72
        )
        en_recs["v12_integrated"].append(_eval_v12_extra(y12, xa, (time.perf_counter() - ts) / max(1e-6, len(y12) / SR), m12))
        en_tel["v12_exact_pct"].append(float(m12["en_diphone_exact_pct"] or 0.0))
        en_tel["v12_fallback_tot"] += int(m12["source_counts"].get("en_phone_fallback", 0))
        r12, d12 = _en_rhythm_from_slots(y12, pl, m12)
        if r12 is not None: en_tel["v12_func_dur_r"].append(r12)
        if d12 is not None: en_tel["v12_c_minus_f_db"].append(d12)

        if uid in ("slt_0030", "bdl_0030"):
            ab_en[uid] = {"real": xa, "v11": y11, "v12": y12}
        if idx_e % 5 == 0 or idx_e == len(en_heldout):
            print(f"  EN {idx_e}/{len(en_heldout)} done", flush=True)

    # 2. Mandarin n=26 Held-out LOO-CV (strict zero leakage on each held-out fold)
    print("Evaluating Mandarin LOO-CV n=26 (Real vs v11 vs v12)...", flush=True)
    zh_prompt_ids = {"SSB05020005", "SSB05020077", "SSB05020082", "SSB15930031", "SSB15930079", "SSB15930105"}
    zh_held_utts = [u for u in zh_utts if u["id"] not in zh_prompt_ids]
    zh_for_ub = {k: (v[0], v[1]) for k, v in zh_pre.items()}
    zh_recs = {k: [] for k in ("real_zh", "v11_diphone", "v12_integrated")}
    zh_v11_whole, zh_v12_whole, zh_v12_drops = [], [], []
    ab_zh = {}

    for idx_z, u in enumerate(zh_held_utts, 1):
        uid, spk = u["id"], u["speaker"]
        xa = zh_real_cache[uid]
        pl = v6_arb.plan_from_pinyin_syllables(u["syllables"], text=u["text"])
        zh_recs["real_zh"].append(_eval_v12_extra(xa, xa, 0.0))

        ub_fold = ub.build_unit_bank([x for x in zh_utts if x["id"] != uid], exclude_ids={uid}, precomputed=zh_for_ub, sr=SR)
        spk_items = [zh_for_ub[x["id"]] for x in zh_utts if x["speaker"] == spk and x["id"] != uid]
        t_v5 = extract_timbre_v5(spk_items, sr=SR, name=spk, corpus="aishell3")
        p5_11 = dict(b11_full.profiles_v5); p5_11[spk] = t_v5
        b11_f = V11DiphoneDualBank(
            en_diphones=b11_full.en_diphones, en_phones=b11_full.en_phones,
            zh_syllables={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in b11_full.zh_syllables.items()) if v},
            zh_onsets={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in b11_full.zh_onsets.items()) if v},
            zh_finals={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in b11_full.zh_finals.items()) if v},
            profiles_v2=b11_full.profiles_v2, profiles_v5=p5_11, fallback_unit_bank=ub_fold,
            zh_included_ids=[i for i in b11_full.zh_included_ids if i != uid], zh_excluded_ids=[uid],
        )
        b12_f = filter_v12_bank_excluding_utterance(b12_full, uid, ub_fold=ub_fold, spk_override=(spk, t_v5))

        ts = time.perf_counter()
        y11, m11 = render_v11_utterance(pl, b11_f, t_v5, cloner_fn=adapt_features_v5, seed=72)
        zh_recs["v11_diphone"].append(_eval_v12_extra(y11, xa, (time.perf_counter() - ts) / max(1e-6, len(y11) / SR), m11))
        sc11 = m11["source_counts"]
        nw11 = sc11.get("zh_target_syl", 0) + sc11.get("zh_donor_syl", 0)
        zh_v11_whole.append(100.0 * nw11 / max(1, nw11 + sc11.get("zh_concat_of", 0) + sc11.get("zh_ipa_fill", 0)))

        ts = time.perf_counter()
        y12, m12 = render_v12_utterance(pl, b12_f, t_v5, prosody_model=prosody_model, cloner_fn=adapt_features_v5, seed=72)
        zh_recs["v12_integrated"].append(_eval_v12_extra(y12, xa, (time.perf_counter() - ts) / max(1e-6, len(y12) / SR), m12))
        zh_v12_drops.append(_slot_drop_count(y12, m12))
        sc12 = m12["source_counts"]
        nw12 = sc12.get("zh_target_syl", 0) + sc12.get("zh_donor_syl", 0)
        zh_v12_whole.append(100.0 * nw12 / max(1, nw12 + sc12.get("zh_concat_of", 0) + sc12.get("zh_ipa_fill", 0)))

        if uid in ("SSB05020122", "SSB15930193"):
            ab_zh[uid] = {"real": xa, "v11": y11, "v12": y12}
        if idx_z % 5 == 0 or idx_z == len(zh_held_utts):
            print(f"  ZH LOO {idx_z}/{len(zh_held_utts)} done", flush=True)

    # 3. Code-Switched CS_01..CS_06 (SSB0502 + SSB0073_M133)
    print("Evaluating Code-Switched CS_01..CS_06...", flush=True)
    btexts = json.loads(BENCHMARK_TEXTS.read_text(encoding="utf-8"))
    cs_texts = [t["text"] for t in btexts.get("sentences", []) if t.get("lang") == "zh_en"][:6]
    cs_recs = {"v11_diphone": [], "v12_integrated": []}
    cs_l2_before, cs_l2_after, cs_exact12 = [], [], []
    for txt in cs_texts:
        pl_cs = plan_v10(txt)
        ts = time.perf_counter()
        y11, m11 = render_v11_utterance(pl_cs, b11_full, b11_full.profiles_v5["SSB0502"], cloner_fn=adapt_features_v5, seed=72)
        cs_recs["v11_diphone"].append(_eval_v12_extra(y11, None, (time.perf_counter() - ts) / max(1e-6, len(y11) / SR), m11))

        ts = time.perf_counter()
        y12, m12 = render_v12_utterance(pl_cs, b12_full, b12_full.profiles_v5["SSB0502"], prosody_model=prosody_model, cloner_fn=adapt_features_v5, seed=72)
        cs_recs["v12_integrated"].append(_eval_v12_extra(y12, None, (time.perf_counter() - ts) / max(1e-6, len(y12) / SR), m12))
        if m12["en_diphone_exact_pct"] is not None:
            cs_exact12.append(float(m12["en_diphone_exact_pct"]))
        bi = m12.get("cs_bridge_info") or {}
        if bi.get("join_l2_before_db") is not None:
            cs_l2_before.append(float(bi["join_l2_before_db"]))
        if bi.get("join_l2_after_db") is not None:
            cs_l2_after.append(float(bi["join_l2_after_db"]))

    exported = {}
    if export_samples:
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        gap = np.zeros(int(0.42 * SR), dtype=np.float64)

        def _write_chk(fn: str, wav: np.ndarray) -> None:
            vocoder.write_wav_mono(SAMPLES_DIR / fn, wav, sr=SR)
            exported[fn] = evm.check_wav_sanity(wav, SR)

        for uid, c in ab_en.items():
            _write_chk(f"AB_EN_{uid}_real_v11_v12.wav", np.concatenate([_loudness_norm(c["real"]), gap, _loudness_norm(c["v11"]), gap, _loudness_norm(c["v12"])]))
        for uid, c in ab_zh.items():
            _write_chk(f"AB_ZH_{uid}_real_v11_v12.wav", np.concatenate([_loudness_norm(c["real"]), gap, _loudness_norm(c["v11"]), gap, _loudness_norm(c["v12"])]))

        cs0_pl = plan_v10(cs_texts[0])
        y11_cs0, _ = render_v11_utterance(cs0_pl, b11_full, b11_full.profiles_v5["SSB0502"], cloner_fn=adapt_features_v5, seed=72)
        y12_cs0, _ = render_v12_utterance(cs0_pl, b12_full, b12_full.profiles_v5["SSB0502"], prosody_model=prosody_model, cloner_fn=adapt_features_v5, seed=72)
        _write_chk("AB_CS01_SSB0502_v11_vs_v12.wav", np.concatenate([_loudness_norm(y11_cs0), gap, _loudness_norm(y12_cs0)]))

        ssb0073_rows = [u for u in zh_new_man if u["speaker"] == "SSB0073"]
        if ssb0073_rows:
            v5_73, gb_73 = _fewshot_v5_and_gb(ssb0073_rows, "SSB0073", "aishell3")
            y11_73, _ = render_v11_utterance(cs0_pl, b11_full, v5_73, cloner_fn=adapt_features_v5, seed=72)
            y12_73, _ = render_v12_utterance(cs0_pl, b12_full, v5_73, prosody_model=prosody_model, glottal_bank=gb_73, cloner_fn=adapt_features_v5, seed=72)
            _write_chk("AB_CS01_SSB0073_M133_v11_vs_v12.wav", np.concatenate([_loudness_norm(y11_73), gap, _loudness_norm(y12_73)]))

    out = {
        "schema": 1,
        "elapsed_s": round(time.time() - t0, 2),
        "v11_bank_stats": b11_full.stats,
        "v12_bank_stats": b12_full.stats,
        "en_heldout_n20": {m: _summarize(r) for m, r in en_recs.items()},
        "en_telemetry": {
            "v11_diphone_exact_pct_mean": round(float(np.mean(en_tel["v11_exact_pct"])), 2),
            "v12_diphone_exact_pct_mean": round(float(np.mean(en_tel["v12_exact_pct"])), 2),
            "v11_fallback_total": en_tel["v11_fallback_tot"],
            "v12_fallback_total": en_tel["v12_fallback_tot"],
            "v11_func_to_content_dur_ratio": round(float(np.median(en_tel["v11_func_dur_r"])), 3) if en_tel["v11_func_dur_r"] else None,
            "v12_func_to_content_dur_ratio": round(float(np.median(en_tel["v12_func_dur_r"])), 3) if en_tel["v12_func_dur_r"] else None,
            "v11_content_minus_func_db": round(float(np.median(en_tel["v11_c_minus_f_db"])), 3) if en_tel["v11_c_minus_f_db"] else None,
            "v12_content_minus_func_db": round(float(np.median(en_tel["v12_c_minus_f_db"])), 3) if en_tel["v12_c_minus_f_db"] else None,
        },
        "zh_loo_n26": {m: _summarize(r) for m, r in zh_recs.items()},
        "zh_telemetry": {
            "v11_whole_syl_pct_mean": round(float(np.mean(zh_v11_whole)), 2),
            "v12_whole_syl_pct_mean": round(float(np.mean(zh_v12_whole)), 2),
            "v12_dropped_slots_total": sum(zh_v12_drops),
        },
        "cs_eval": {m: _summarize(r) for m, r in cs_recs.items()},
        "cs_telemetry": {
            "v12_diphone_exact_pct_mean": round(float(np.mean(cs_exact12)), 2) if cs_exact12 else None,
            "cs_join_l2_before_db": round(float(np.median(cs_l2_before)), 3) if cs_l2_before else None,
            "cs_join_l2_after_db": round(float(np.median(cs_l2_after)), 3) if cs_l2_after else None,
        },
        "exported_samples": exported,
    }
    (EXP_DIR / "results_v12.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run_v12_eval(export_samples=True)
    print(f"Completed Phase 8 v12 Eval in {res['elapsed_s']}s")
    print("EN telemetry:", res["en_telemetry"])
    print("ZH telemetry:", res["zh_telemetry"])
    print("CS telemetry:", res["cs_telemetry"])
    for sec in ("en_heldout_n20", "zh_loo_n26", "cs_eval"):
        print(f"\n=== {sec} ===")
        for m, info in res[sec].items():
            d, s = info["medians"], info["sums"]
            lsd = f" | MelLSD={d['mel_lsd_db']} MCD={d['mcd_db']}" if "mel_lsd_db" in d else ""
            print(
                f"  [{m:15s}] ok={info['sanity_ok']}/{info['n']}{lsd} | uv2_8k={d['uv_2_8k_vs_lo_db']} | "
                f"contrast={d['spectral_contrast_db']} | p99={d['p99_db']} (>9={s['gt9db_total']}) | RTF={d['rtf']}"
            )
    print("Samples:", {k: v["ok"] for k, v in res["exported_samples"].items()})
