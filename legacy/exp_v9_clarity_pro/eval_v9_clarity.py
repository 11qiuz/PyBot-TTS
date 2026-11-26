"""Zero-leakage Phase 5 Evaluation & Ablation for exp_v9_clarity_pro (ZH n=26 + EN n=20 + Boundary Purity)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.fft import rfft, rfftfreq

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v5_eval import metrics as ev
from exp_v6_crossling_dsp import arbiter as v6_arb
from exp_v6_eval import fluency as fl
from exp_v6_vocoder import vocoder
from exp_v8_clarity.clarity_metrics import compute_clarity_metrics
from exp_v8_clarity.en_utau_bank import _trim_active_span, build_dual_donor_bank
from exp_v8_clarity.utau_concat import build_fullres_bank, render_utau_plan
from exp_v8_data import clarity_diag as cd
from exp_v8_timbre_clone import timbre_cloner_v2 as tc2
from exp_v9_clarity_pro import EN_MANIFEST, EXP_DIR, EXTENDED_INDEX, SAMPLES_DIR, SR
from exp_v9_clarity_pro.boundary_refiner import measure_token_boundary_purity
from exp_v9_clarity_pro.hnm_v9 import analyze_v9
from exp_v9_clarity_pro.viterbi_utau import (
    build_v9_dual_bank,
    filter_v9_bank_excluding_speakers,
    render_v9_utterance,
)


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


def _gt8k_unvoiced_ratio(x: np.ndarray, sr: int = SR) -> float:
    """Measure >8 kHz vs 80-1000 Hz energy ratio (dB) on active unvoiced frames."""
    frames, keep, _ = ev.extract_active_frames(x, frame_len=720, hop=240, db_thresh=30.0)
    tr = fl.compute_f0_track(x, sr=sr, fmin=75.0, fmax=450.0)
    nc = min(len(keep), len(tr["voiced"]))
    uv = keep[:nc] & (~tr["voiced"][:nc])
    if int(np.sum(uv)) < 2:
        return 0.0
    fr_uv = frames[:nc][uv]
    win = np.hamming(720)[None, :]
    pwr = np.abs(rfft((fr_uv - np.mean(fr_uv, axis=1, keepdims=True)) * win, n=2048, axis=1)) ** 2
    freqs = rfftfreq(2048, 1.0 / sr)
    elo = np.sum(pwr[:, (freqs >= 80.0) & (freqs < 1000.0)], axis=1) + 1e-18
    ehi = np.sum(pwr[:, (freqs >= 8000.0) & (freqs <= 11800.0)], axis=1) + 1e-18
    return float(np.median(10.0 * np.log10(ehi / elo)))


def _eval_row(y: np.ndarray, ref: np.ndarray, ev_ref: dict, rtf: float) -> dict:
    ev_y = ev.evaluate_audio(y, sr=SR)
    fl_y = fl.compute_fluency_metrics(y, sr=SR)
    cl_y = compute_clarity_metrics(y, sr=SR)
    sd = ev.compute_spectral_distances(ref, y, sr=SR)
    f0_y = ev_y["f0_hnr"]["f0_median_hz"]
    f0_r = ev_ref["f0_hnr"]["f0_median_hz"]
    f0_err = abs(12.0 * math.log2(f0_y / f0_r)) if (f0_y and f0_r and math.isfinite(f0_y) and math.isfinite(f0_r)) else float("nan")
    return {
        "sanity_ok": bool(ev_y["sanity"]["ok"]),
        "mel_lsd_db": float(sd["mel_lsd_db"]),
        "mcd_db": float(sd["mcd_db"]),
        "mcd_cmn_db": float(sd["mcd_cmn_db"]),
        "tilt_err_db_oct": abs(ev_y["ltas"]["tilt_db_oct"] - ev_ref["ltas"]["tilt_db_oct"]),
        "f0_err_st": float(f0_err),
        "hnr_err_db": abs(ev_y["f0_hnr"]["hnr_db_median"] - ev_ref["f0_hnr"]["hnr_db_median"]),
        "flux_p95_med": float(fl_y["spectral_flux"]["spectral_flux_p95_to_median"]),
        "vuv_flux_db": float(fl_y["vuv_transitions"]["vuv_flux_median_db"]),
        "gt8k_uv_lf_db": _gt8k_unvoiced_ratio(y, sr=SR),
        "rtf": float(rtf),
        **cl_y,
    }


def _summarize(recs: list[dict]) -> dict:
    keys = (
        "mel_lsd_db", "mcd_db", "mcd_cmn_db", "spectral_contrast_db",
        "unvoiced_hf_lf_db", "gt8k_uv_lf_db", "transient_hf_p95_db",
        "flux_p95_med", "vuv_flux_db", "f0_err_st", "tilt_err_db_oct", "hnr_err_db", "rtf",
    )
    return {
        "n": len(recs),
        "sanity_ok": sum(1 for r in recs if r["sanity_ok"]),
        "medians": {k: round(float(np.nanmedian([r[k] for r in recs])), 3) for k in keys},
    }


def run_v9_eval(export_samples: bool = True) -> dict:
    """Run full Phase 5 evaluation: boundary purity, n=26 ZH LOO-CV, n=20 EN held-out."""
    t0 = time.time()
    zh_utts = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
    man_all = json.loads(EN_MANIFEST.read_text(encoding="utf-8"))
    en_man = man_all["english"]["utterances"]
    zh_new_man = man_all.get("mandarin", {}).get("utterances", [])

    en_heldout = [u for u in en_man if u["speaker"] in ("slt", "bdl") and int(u["id"].split("_")[1]) >= 30]
    en_excl_ids = {u["id"] for u in en_heldout}

    # Precompute v9 HD features once for all ZH and EN train utterances (huge speedup for 26-fold LOO!)
    print("Precomputing v8 + v9 features (ZH + EN)...", flush=True)
    zh_pre_v9, zh_pre_v8, zh_eval_cache = {}, {}, {}
    for u in zh_utts:
        x, _ = vocoder.read_wav_mono(u["path"])
        xa = x[u["active_start"] : u["active_stop"]]
        f_v9, bg = analyze_v9(xa, sr=SR)
        f_v8 = vocoder.analyze(xa, sr=SR)
        zh_pre_v9[u["id"]] = (xa, f_v9, bg)
        zh_pre_v8[u["id"]] = (xa, f_v8)
        zh_eval_cache[u["id"]] = (xa, ev.evaluate_audio(xa, sr=SR))

    en_pre_v9, en_pre_v8 = {}, {}
    en_train_list = [u for u in en_man if u["id"] not in en_excl_ids]
    for idx_e, u in enumerate(en_train_list, 1):
        p_wav = EN_MANIFEST.parent.parent.parent / u["path"]
        x, _ = vocoder.read_wav_mono(p_wav)
        s0, s1 = _trim_active_span(x, sr=SR)
        xa = x[s0:s1]
        f_v9, bg = analyze_v9(xa, sr=SR)
        f_v8 = vocoder.analyze(xa, sr=SR)
        en_pre_v9[u["id"]] = (xa, f_v9, bg)
        en_pre_v8[u["id"]] = (xa, f_v8)
        if idx_e % 35 == 0 or idx_e == len(en_train_list):
            print(f"  Precomputed {idx_e}/{len(en_train_list)} EN train utts", flush=True)

    en_real_cache = {}
    for u in en_heldout:
        p_wav = EN_MANIFEST.parent.parent.parent / u["path"]
        x, _ = vocoder.read_wav_mono(p_wav)
        s0, s1 = _trim_active_span(x, sr=SR)
        xa = x[s0:s1]
        en_real_cache[u["id"]] = (xa, ev.evaluate_audio(xa, sr=SR))

    # 1. Token Boundary Purity Before (v8) vs After (v9 Landmark Refiner)
    print("Measuring before/after token boundary purity...", flush=True)
    v8_dual = build_dual_donor_bank(
        zh_utterances=zh_utts, zh_exclude_ids=(), en_utterances=en_man, en_exclude_ids=en_excl_ids,
        zh_precomputed=zh_pre_v8, en_precomputed=en_pre_v8,
    )
    v9_dual = build_v9_dual_bank(
        zh_utterances=zh_utts, zh_exclude_ids=(), en_utterances=en_man, en_exclude_ids=en_excl_ids,
        zh_precomputed=zh_pre_v9, en_precomputed=en_pre_v9, refine_boundaries=True,
    )
    purity_v8 = measure_token_boundary_purity(v8_dual.en_phones, v8_dual.zh_bank.syllables, sr=SR)
    purity_v9 = measure_token_boundary_purity(v9_dual.en_phones, v9_dual.zh_syllables, sr=SR)

    # 2. Mandarin Held-Out LOO-CV (n=26 utterances matching exp_v7/exp_v8 held-out set)
    print("Running n=26 Mandarin LOO-CV...", flush=True)
    zh_prompt_ids = {"SSB05020005", "SSB05020077", "SSB05020082", "SSB15930031", "SSB15930079", "SSB15930105"}
    zh_held_utts = [u for u in zh_utts if u["id"] not in zh_prompt_ids]
    zh_recs = {k: [] for k in ("real_zh", "v8_utau_hnm", "v9_no_res", "v9_full")}
    diag_1593, diag_0502 = {}, {}
    ab_zh_clips = {}

    from exp_v7_pybot_tts import unit_bank as ub
    from exp_v8_clarity.utau_concat import FullResBank
    from exp_v9_clarity_pro.viterbi_utau import V9DualBank

    # Build base banks once and filter per LOO fold (zero leakage + 10x faster than rebuilding 26x)
    v8_zh_base = build_fullres_bank(zh_utts, exclude_ids=(), precomputed=zh_pre_v8, sr=SR, include_fallback_bank=False)
    v9_zh_base = build_v9_dual_bank(
        zh_utterances=zh_utts, zh_exclude_ids=(), en_utterances=[], en_exclude_ids=(),
        zh_precomputed=zh_pre_v9, refine_boundaries=True, include_fallback_bank=False,
    )
    zh_for_ub_v8 = zh_pre_v8
    zh_for_ub_v9 = {k: (v[0], v[1]) for k, v in zh_pre_v9.items()}

    for idx_u, u in enumerate(zh_held_utts, 1):
        uid = u["id"]
        spk = u["speaker"]
        xa, ev_r = zh_eval_cache[uid]
        pl = v6_arb.plan_from_pinyin_syllables(u["syllables"], text=u["text"])
        zh_recs["real_zh"].append(_eval_row(xa, xa, ev_r, 0.0))

        # LOO fold banks strictly excluding uid from tokens, fallback unit_bank, and target speaker profile
        ub_v8_fold = ub.build_unit_bank([x for x in zh_utts if x["id"] != uid], exclude_ids={uid}, precomputed=zh_for_ub_v8, sr=SR)
        b_v8 = FullResBank(
            syllables={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in v8_zh_base.syllables.items()) if v},
            onsets={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in v8_zh_base.onsets.items()) if v},
            finals={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in v8_zh_base.finals.items()) if v},
            spk_profiles=v8_zh_base.spk_profiles,
            fallback_unit_bank=ub_v8_fold,
            included_ids=[i for i in v8_zh_base.included_ids if i != uid],
            excluded_ids=[uid],
        )
        t_s = time.perf_counter()
        y_v8, _ = render_utau_plan(pl, b_v8, b_v8.spk_profiles[spk], seed=72)
        zh_recs["v8_utau_hnm"].append(_eval_row(y_v8, xa, ev_r, (time.perf_counter() - t_s) / max(1e-6, len(y_v8) / SR)))

        ub_v9_fold = ub.build_unit_bank([x for x in zh_utts if x["id"] != uid], exclude_ids={uid}, precomputed=zh_for_ub_v9, sr=SR)
        t_p = tc2.extract_v2_profile([zh_for_ub_v9[x["id"]] for x in zh_utts if x["speaker"] == spk and x["id"] != uid], sr=SR, name=spk)
        profs_fold = dict(v9_zh_base.zh_profiles_v2)
        profs_fold[spk] = t_p
        b_v9 = V9DualBank(
            zh_syllables={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in v9_zh_base.zh_syllables.items()) if v},
            zh_onsets={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in v9_zh_base.zh_onsets.items()) if v},
            zh_finals={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in v9_zh_base.zh_finals.items()) if v},
            en_phones={},
            zh_profiles_v2=profs_fold,
            en_profiles_v2={},
            fallback_unit_bank=ub_v9_fold,
            prosody_rules=v9_zh_base.prosody_rules,
            zh_included_ids=[i for i in v9_zh_base.zh_included_ids if i != uid],
            zh_excluded_ids=[uid],
        )

        t_s = time.perf_counter()
        y_nres, _ = render_v9_utterance(pl, b_v9, t_p, use_viterbi=True, use_unvoiced_residual=False, seed=72)
        zh_recs["v9_no_res"].append(_eval_row(y_nres, xa, ev_r, (time.perf_counter() - t_s) / max(1e-6, len(y_nres) / SR)))

        t_s = time.perf_counter()
        y_full, _ = render_v9_utterance(pl, b_v9, t_p, use_viterbi=True, use_unvoiced_residual=True, seed=72)
        zh_recs["v9_full"].append(_eval_row(y_full, xa, ev_r, (time.perf_counter() - t_s) / max(1e-6, len(y_full) / SR)))
        if idx_u % 10 == 0 or idx_u == len(zh_held_utts):
            print(f"  ZH LOO {idx_u}/{len(zh_held_utts)} done", flush=True)

        if uid == "SSB15930193":
            diag_1593 = {
                "real": cd.compute_clarity_metrics(xa, sr=SR),
                "v8_utau_hnm": cd.compute_clarity_metrics(y_v8, sr=SR),
                "v9_no_res": cd.compute_clarity_metrics(y_nres, sr=SR),
                "v9_full": cd.compute_clarity_metrics(y_full, sr=SR),
            }
            ab_zh_clips[uid] = {"real": xa, "v8": y_v8, "v9_no_res": y_nres, "v9_full": y_full}
        elif uid == "SSB05020122":
            diag_0502 = {
                "real": cd.compute_clarity_metrics(xa, sr=SR),
                "v8_utau_hnm": cd.compute_clarity_metrics(y_v8, sr=SR),
                "v9_no_res": cd.compute_clarity_metrics(y_nres, sr=SR),
                "v9_full": cd.compute_clarity_metrics(y_full, sr=SR),
            }
            ab_zh_clips[uid] = {"real": xa, "v8": y_v8, "v9_no_res": y_nres, "v9_full": y_full}

    # 3. English Held-Out Evaluation (n=20 slt[30:40] & bdl[30:40], 3-shot prompt)
    print("Running n=20 English held-out evaluation...", flush=True)
    def _fewshot_v2(rows: list[dict], spk: str) -> tc2.TimbreProfileV2:
        items = []
        for r in rows[:3]:
            x, _ = vocoder.read_wav_mono(EN_MANIFEST.parent.parent.parent / r["path"])
            s0, s1 = _trim_active_span(x, sr=SR)
            f, _ = analyze_v9(x[s0:s1], sr=SR)
            items.append((x[s0:s1], f))
        return tc2.extract_v2_profile(items, sr=SR, name=spk)

    prof_slt_3s = _fewshot_v2([u for u in en_man if u["speaker"] == "slt" and u["id"] not in en_excl_ids], "slt")
    prof_bdl_3s = _fewshot_v2([u for u in en_man if u["speaker"] == "bdl" and u["id"] not in en_excl_ids], "bdl")

    v9_no_bdl = filter_v9_bank_excluding_speakers(v9_dual, {"bdl"})
    v9_no_slt = filter_v9_bank_excluding_speakers(v9_dual, {"slt"})

    from exp_v8_clarity.en_utau_bank import filter_dual_bank_excluding_speakers, render_dual_utau_plan

    en_recs = {k: [] for k in ("real_en", "v8_utau_en", "v9_en_no_res", "v9_en_full", "v9_en_cross_donor")}
    ab_en_clips = {}
    for idx_h, u in enumerate(en_heldout, 1):
        uid, spk = u["id"], u["speaker"]
        xa, ev_r = en_real_cache[uid]
        prof_tgt = prof_slt_3s if spk == "slt" else prof_bdl_3s
        pl = cf.plan(u["text"], allow_oov_fallback=True)
        en_recs["real_en"].append(_eval_row(xa, xa, ev_r, 0.0))

        t_s = time.perf_counter()
        y_v8, _ = render_dual_utau_plan(pl, v8_dual, prof_tgt, seed=72)
        en_recs["v8_utau_en"].append(_eval_row(y_v8, xa, ev_r, (time.perf_counter() - t_s) / max(1e-6, len(y_v8) / SR)))

        t_s = time.perf_counter()
        y_nr, _ = render_v9_utterance(pl, v9_dual, prof_tgt, use_viterbi=True, use_unvoiced_residual=False, seed=72)
        en_recs["v9_en_no_res"].append(_eval_row(y_nr, xa, ev_r, (time.perf_counter() - t_s) / max(1e-6, len(y_nr) / SR)))

        t_s = time.perf_counter()
        y_fl, _ = render_v9_utterance(pl, v9_dual, prof_tgt, use_viterbi=True, use_unvoiced_residual=True, seed=72)
        en_recs["v9_en_full"].append(_eval_row(y_fl, xa, ev_r, (time.perf_counter() - t_s) / max(1e-6, len(y_fl) / SR)))

        cd_bank = v9_no_bdl if spk == "bdl" else v9_no_slt
        t_s = time.perf_counter()
        y_cd, _ = render_v9_utterance(pl, cd_bank, prof_tgt, use_viterbi=True, use_unvoiced_residual=True, seed=72)
        en_recs["v9_en_cross_donor"].append(_eval_row(y_cd, xa, ev_r, (time.perf_counter() - t_s) / max(1e-6, len(y_cd) / SR)))

        if export_samples and uid in ("slt_0030", "bdl_0030"):
            ab_en_clips[uid] = {"real": xa, "v8": y_v8, "v9_no_res": y_nr, "v9_full": y_fl}
        if idx_h % 5 == 0 or idx_h == len(en_heldout):
            print(f"  EN held-out {idx_h}/{len(en_heldout)} done", flush=True)

    exported = []
    if export_samples:
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        gap = np.zeros(int(0.45 * SR), dtype=np.float64)
        for uid, c in ab_zh_clips.items():
            pack = np.concatenate([_loudness_norm(c["real"]), gap, _loudness_norm(c["v8"]), gap, _loudness_norm(c["v9_no_res"]), gap, _loudness_norm(c["v9_full"])])
            p = SAMPLES_DIR / f"AB_ZH_{uid}_real_v8_v9nores_v9full.wav"
            vocoder.write_wav_mono(p, pack, sr=SR)
            exported.append(p.name)
        for uid, c in ab_en_clips.items():
            pack = np.concatenate([_loudness_norm(c["real"]), gap, _loudness_norm(c["v8"]), gap, _loudness_norm(c["v9_no_res"]), gap, _loudness_norm(c["v9_full"])])
            p = SAMPLES_DIR / f"AB_EN_{uid}_real_v8_v9nores_v9full.wav"
            vocoder.write_wav_mono(p, pack, sr=SR)
            exported.append(p.name)

        # Code-switched v9 demos (SSB0502 female + SSB0073 Male Mandarin)
        cs_text = "今天 PyBotTTS 测试了 natural English speech。"
        y_cs_502, _ = render_v9_utterance(cs_text, v9_dual, v9_dual.zh_profiles_v2["SSB0502"], seed=72)
        p_cs1 = SAMPLES_DIR / "CS_demo_v9_SSB0502.wav"
        vocoder.write_wav_mono(p_cs1, _loudness_norm(y_cs_502), sr=SR)
        exported.append(p_cs1.name)

        ssb0073_rows = [u for u in zh_new_man if u["speaker"] == "SSB0073"]
        if ssb0073_rows:
            prof_0073 = _fewshot_v2(ssb0073_rows, "SSB0073")
            y_cs_0073, _ = render_v9_utterance(cs_text, v9_dual, prof_0073, seed=72)
            p_cs2 = SAMPLES_DIR / "CS_demo_v9_SSB0073_M133.wav"
            vocoder.write_wav_mono(p_cs2, _loudness_norm(y_cs_0073), sr=SR)
            exported.append(p_cs2.name)

    out = {
        "schema": 1,
        "elapsed_s": round(time.time() - t0, 2),
        "v9_bank_stats": {
            "zh_included": len(v9_dual.zh_included_ids),
            "en_included": len(v9_dual.en_included_ids),
            "en_excluded": len(v9_dual.en_excluded_ids),
            "en_unique_ipas": len(v9_dual.en_phones),
            "en_total_tokens": sum(len(v) for v in v9_dual.en_phones.values()),
            "zh_final_lengthen_rule": round(v9_dual.prosody_rules.zh_final_lengthen, 3),
        },
        "boundary_purity_before_after": {"v8_unrefined": purity_v8, "v9_landmark_refined": purity_v9},
        "zh_loo_n26": {m: _summarize(r) for m, r in zh_recs.items()},
        "en_heldout_n20": {m: _summarize(r) for m, r in en_recs.items()},
        "clarity_diag_SSB15930193": diag_1593,
        "clarity_diag_SSB05020122": diag_0502,
        "exported_samples": exported,
    }
    (EXP_DIR / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run_v9_eval(export_samples=True)
    print(f"Completed v9 Clarity Pro Evaluation in {res['elapsed_s']}s")
    print("Boundary Purity:", res["boundary_purity_before_after"])
    print("\n=== Mandarin n=26 Held-Out LOO-CV ===")
    for m, info in res["zh_loo_n26"].items():
        d = info["medians"]
        print(
            f"  [{m:12s}] ok={info['sanity_ok']}/{info['n']} | Mel-LSD={d['mel_lsd_db']} | contrast={d['spectral_contrast_db']} | "
            f"uv_HF/LF={d['unvoiced_hf_lf_db']} | >8k_uv={d['gt8k_uv_lf_db']} | vuv_flux={d['vuv_flux_db']} | RTF={d['rtf']}"
        )
    print("\n=== English n=20 Held-Out (slt[30:40], bdl[30:40]) ===")
    for m, info in res["en_heldout_n20"].items():
        d = info["medians"]
        print(
            f"  [{m:18s}] ok={info['sanity_ok']}/{info['n']} | Mel-LSD={d['mel_lsd_db']} | contrast={d['spectral_contrast_db']} | "
            f"uv_HF/LF={d['unvoiced_hf_lf_db']} | >8k_uv={d['gt8k_uv_lf_db']} | vuv_flux={d['vuv_flux_db']} | RTF={d['rtf']}"
        )
    print("\nSSB15930193 clarity_diag uv_2_8k_vs_lo_db:", {k: v["unvoiced_2_8k_vs_lo_db"] for k, v in res["clarity_diag_SSB15930193"].items()})
    print("SSB05020122 clarity_diag uv_2_8k_vs_lo_db:", {k: v["unvoiced_2_8k_vs_lo_db"] for k, v in res["clarity_diag_SSB05020122"].items()})
    print("Exported:", res["exported_samples"])
