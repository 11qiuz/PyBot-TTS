"""Phase 5 End-to-End v9 Benchmark (v9_clarity_pro + v9_timbre_v3) and v4->v9 Comparison Suite."""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity, compute_spectral_distances, evaluate_audio, read_wav  # noqa: E402
from exp_v6_crossling_dsp import arbiter as v6_arb  # noqa: E402
from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v7_pybot_tts import unit_bank as ub  # noqa: E402
from exp_v8_clarity.en_utau_bank import _trim_active_span  # noqa: E402
from exp_v8_timbre_clone import timbre_cloner_v2 as tc2  # noqa: E402
from exp_v9_clarity_pro.hnm_v9 import analyze_v9  # noqa: E402
from exp_v9_clarity_pro.viterbi_utau import (  # noqa: E402
    V9DualBank,
    build_v9_dual_bank,
    filter_v9_bank_excluding_speakers,
    render_v9_utterance,
)
from exp_v9_eval.v9_metrics import compute_v9_extended_metrics  # noqa: E402
from exp_v9_timbre_v3 import timbre_cloner_v3 as tc3  # noqa: E402

SR = 24000
EVAL_DIR = pathlib.Path(__file__).resolve().parent
LISTENING_DIR = EVAL_DIR / "listening_ab"


def _rms_norm(x: np.ndarray, target_dbfs: float = -20.0, max_peak: float = 0.89) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    act = arr[np.abs(arr) > 0.01 * max(1e-6, float(np.max(np.abs(arr))))]
    rms = float(np.sqrt(np.mean(act * act))) if act.size else float(np.sqrt(np.mean(arr * arr)))
    if rms <= 1e-12:
        return arr.copy()
    scaled = arr * ((10.0 ** (target_dbfs / 20.0)) / rms)
    pk = float(np.max(np.abs(scaled)))
    if pk > max_peak:
        scaled = scaled * (max_peak / pk)
    return scaled


def _write_pcm16_wav(path: pathlib.Path, x: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _concat_ab(segments: list[tuple[str, np.ndarray]], gap_s: float = 0.40) -> tuple[np.ndarray, list[dict]]:
    gap = np.zeros(int(round(gap_s * SR)), dtype=np.float64)
    chunks: list[np.ndarray] = []
    timeline: list[dict] = []
    cursor_s = 0.0
    for idx, (label, raw) in enumerate(segments):
        xn = _rms_norm(raw)
        dur = len(xn) / float(SR)
        timeline.append({"label": label, "start_s": round(cursor_s, 3), "stop_s": round(cursor_s + dur, 3)})
        chunks.append(xn)
        cursor_s += dur
        if idx + 1 < len(segments):
            chunks.append(gap)
            cursor_s += gap_s
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64), timeline


def make_v3_cloner_adapter(src_v3_profiles: dict[str, tc3.TimbreProfileV3], eq_mode: str = "blind_tilt_anchor"):
    """Bridge source tokens inside render_v9_utterance -> TimbreClonerV3 safely."""
    def _adapter(feats: vocoder.VocoderFeatures, src_prof, tgt_prof, vtln_alpha=None):
        src_name = getattr(src_prof, "name", "")
        if isinstance(src_prof, tc3.TimbreProfileV3) and isinstance(tgt_prof, tc3.TimbreProfileV3):
            return tc3.adapt_features_v3(
                feats, src_prof, tgt_prof, vtln_alpha=vtln_alpha, eq_mode=eq_mode
            )
        if src_name in src_v3_profiles and isinstance(tgt_prof, tc3.TimbreProfileV3):
            return tc3.adapt_features_v3(
                feats,
                src_v3_profiles[src_name],
                tgt_prof,
                vtln_alpha=vtln_alpha,
                eq_mode=eq_mode,
            )
        return tc2.adapt_features_v2(feats, src_prof, tgt_prof, vtln_alpha=vtln_alpha)
    return _adapter


def _score_against_real(y: np.ndarray, ref: np.ndarray, ev_ref: dict, rtf: float, *, fmin: float = 75.0) -> dict:
    ev_y = evaluate_audio(y, sr=SR, fmin=fmin, fmax=450.0)
    v9_y = compute_v9_extended_metrics(y, sr=SR, fmin=fmin, fmax=450.0)
    sd = compute_spectral_distances(ref, y, sr=SR)
    f0_y = ev_y["f0_hnr"]["f0_median_hz"] if ev_y["f0_hnr"] else None
    f0_r = ev_ref["f0_hnr"]["f0_median_hz"] if ev_ref["f0_hnr"] else None
    f0_err = abs(12.0 * math.log2(f0_y / f0_r)) if (f0_y and f0_r) else float("nan")
    hnr_y = ev_y["f0_hnr"]["hnr_db_median"] if ev_y["f0_hnr"] else None
    hnr_r = ev_ref["f0_hnr"]["hnr_db_median"] if ev_ref["f0_hnr"] else None
    tilt_y = ev_y["ltas"]["tilt_db_oct"] if ev_y["ltas"] else None
    tilt_r = ev_ref["ltas"]["tilt_db_oct"] if ev_ref["ltas"] else None

    return {
        "sanity_ok": bool(ev_y["sanity"]["ok"]),
        "mel_lsd_db": float(sd["mel_lsd_db"]) if sd else float("nan"),
        "mcd_db": float(sd["mcd_db"]) if sd else float("nan"),
        "mcd_cmn_db": float(sd["mcd_cmn_db"]) if sd else float("nan"),
        "f0_err_st": float(f0_err),
        "tilt_err_db_oct": abs(tilt_y - tilt_r) if (tilt_y is not None and tilt_r is not None) else float("nan"),
        "hnr_err_db": abs(hnr_y - hnr_r) if (hnr_y is not None and hnr_r is not None) else float("nan"),
        "df2_dt_hz_s": ev_y["formants"]["df2_dt_median_hz_s"] if ev_y["formants"] else float("nan"),
        "voiced_contrast_db": v9_y["clarity"]["voiced_spectral_contrast_db"],
        "uv_2_8k_vs_lo_db": v9_y["clarity"]["unvoiced_2_8k_vs_lo_db"],
        "cv_energy_contrast_db": v9_y["cv_energy_contrast_db"],
        "cv_spectral_dist_db": v9_y["cv_spectral_dist_db"],
        "uv_gt8k_vs_lo_db": v9_y["unvoiced_8_115k_vs_lo_db"],
        "f0_declination_corr": v9_y["f0_declination_corr"],
        "final_lengthening_ratio": v9_y["final_lengthening_ratio"],
        "flux_p95_med": v9_y["fluency"]["spectral_flux"]["spectral_flux_p95_to_median"],
        "vuv_flux_db": v9_y["fluency"]["vuv_transitions"]["vuv_flux_median_db"],
        "f0_glitch_rate": v9_y["fluency"]["f0_continuity"]["f0_glitch_rate_gt4st"],
        "click_max_d2_norm": v9_y["click_max_d2_norm"],
        "click_outlier_rate_gt8": v9_y["click_outlier_rate_gt8"],
        "rtf": float(rtf),
    }


def _summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    keys = [k for k in rows[0] if k != "sanity_ok"]
    meds = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None and isinstance(r[k], (int, float)) and math.isfinite(r[k])]
        if vals:
            meds[k] = round(float(np.median(vals)), 4)
    return {"n": len(rows), "sanity_ok": sum(1 for r in rows if r["sanity_ok"]), "medians": meds}


def main() -> int:
    t0 = time.time()
    zh_utts = json.loads((ROOT / "pybot_tts_v2_recorded/extended/index.json").read_text(encoding="utf-8"))["utterances"]
    man_all = json.loads((ROOT / "exp_v8_data/corpus/manifest.json").read_text(encoding="utf-8"))
    en_man = man_all["english"]["utterances"]
    zh_new_man = man_all.get("mandarin", {}).get("utterances", [])

    en_heldout = [u for u in en_man if u["speaker"] in ("slt", "bdl") and int(u["id"].split("_")[1]) >= 30]
    en_excl_ids = {u["id"] for u in en_heldout}

    print("[1/5] Precomputing v9 HD features (ZH + EN train)...", flush=True)
    zh_pre_v9, zh_eval_cache = {}, {}
    for u in zh_utts:
        x, _ = vocoder.read_wav_mono(u["path"])
        xa = x[u["active_start"]:u["active_stop"]]
        fv9, bg = analyze_v9(xa, sr=SR)
        zh_pre_v9[u["id"]] = (xa, fv9, bg)
        zh_eval_cache[u["id"]] = (xa, evaluate_audio(xa, sr=SR, fmin=90.0))

    en_pre_v9, en_real_cache = {}, {}
    for u in en_man:
        x, _ = vocoder.read_wav_mono(ROOT / u["path"])
        s0, s1 = _trim_active_span(x, sr=SR)
        xa = x[s0:s1]
        if u["id"] in en_excl_ids:
            fmin = 65.0 if u["speaker"] == "bdl" else 110.0
            en_real_cache[u["id"]] = (xa, evaluate_audio(xa, sr=SR, fmin=fmin))
        else:
            fv9, bg = analyze_v9(xa, sr=SR)
            en_pre_v9[u["id"]] = (xa, fv9, bg)

    # Extract TimbreProfileV3 for EN speakers (3-shot non-heldout) & new ZH speakers (3-shot)
    en_v3_profs = {}
    for spk in ("slt", "bdl", "clb", "rms"):
        rows = [u for u in en_man if u["speaker"] == spk and u["id"] not in en_excl_ids][:3]
        items = [(en_pre_v9[r["id"]][0], en_pre_v9[r["id"]][1]) for r in rows]
        en_v3_profs[spk] = tc3.extract_timbre_v3(items, sr=SR, name=spk, corpus="cmu-arctic")

    zh_new_v3_profs = {}
    for spk in ("SSB0073", "SSB0005", "SSB0139"):
        rows = [u for u in zh_new_man if u["speaker"] == spk][:3]
        items = []
        for r in rows:
            x, _ = read_wav(ROOT / r["path"])
            s0, s1 = _trim_active_span(x, sr=SR)
            xa = x[s0:s1]
            fv9, _ = analyze_v9(xa, sr=SR)
            items.append((xa, fv9))
        if items:
            zh_new_v3_profs[spk] = tc3.extract_timbre_v3(items, sr=SR, name=spk, corpus="aishell3")

    zh_v3_full = {}
    for spk in ("SSB0502", "SSB1593"):
        items = [(zh_pre_v9[u["id"]][0], zh_pre_v9[u["id"]][1]) for u in zh_utts if u["speaker"] == spk]
        zh_v3_full[spk] = tc3.extract_timbre_v3(items, sr=SR, name=spk, corpus="aishell3")

    # Build base dual bank (excluding en_heldout)
    v9_dual = build_v9_dual_bank(
        zh_utterances=zh_utts, zh_exclude_ids=(), en_utterances=en_man, en_exclude_ids=en_excl_ids,
        zh_precomputed=zh_pre_v9, en_precomputed=en_pre_v9, refine_boundaries=True,
    )

    # 2. Suite A: n=26 Held-Out ZH LOO-CV with v9_clarity + TimbreClonerV3
    print("[2/5] Running Suite A: n=26 ZH LOO-CV (v9_clarity + TimbreClonerV3)...", flush=True)
    zh_prompt_ids = {"SSB05020005", "SSB05020077", "SSB05020082", "SSB15930031", "SSB15930079", "SSB15930105"}
    zh_held_utts = [u for u in zh_utts if u["id"] not in zh_prompt_ids]
    zh_v9_base = build_v9_dual_bank(
        zh_utterances=zh_utts, zh_exclude_ids=(), en_utterances=[], en_exclude_ids=(),
        zh_precomputed=zh_pre_v9, refine_boundaries=True, include_fallback_bank=False,
    )
    zh_for_ub = {k: (v[0], v[1]) for k, v in zh_pre_v9.items()}

    zh_real_rows, zh_v9_comb_rows = [], []
    ab_zh_clips = {}
    for u in zh_held_utts:
        uid, spk = u["id"], u["speaker"]
        xa, ev_r = zh_eval_cache[uid]
        pl = v6_arb.plan_from_pinyin_syllables(u["syllables"], text=u["text"])
        zh_real_rows.append(_score_against_real(xa, xa, ev_r, 0.0, fmin=90.0))

        # Strictly exclude uid from bank AND speaker profiles
        ub_fold = ub.build_unit_bank([x for x in zh_utts if x["id"] != uid], exclude_ids={uid}, precomputed=zh_for_ub, sr=SR)
        profs_v2_fold = {}
        profs_v3_fold = {}
        for s_name in ("SSB0502", "SSB1593"):
            fold_items = [(zh_pre_v9[x["id"]][0], zh_pre_v9[x["id"]][1]) for x in zh_utts if x["speaker"] == s_name and x["id"] != uid]
            profs_v2_fold[s_name] = tc2.extract_v2_profile(fold_items, sr=SR, name=s_name)
            profs_v3_fold[s_name] = tc3.extract_timbre_v3(fold_items, sr=SR, name=s_name, corpus="aishell3")

        b_fold = V9DualBank(
            zh_syllables={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in zh_v9_base.zh_syllables.items()) if v},
            zh_onsets={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in zh_v9_base.zh_onsets.items()) if v},
            zh_finals={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in zh_v9_base.zh_finals.items()) if v},
            en_phones={},
            zh_profiles_v2=profs_v2_fold,
            en_profiles_v2={},
            fallback_unit_bank=ub_fold,
            prosody_rules=zh_v9_base.prosody_rules,
            zh_included_ids=[i for i in zh_v9_base.zh_included_ids if i != uid],
            zh_excluded_ids=[uid],
        )
        adapter_zh = make_v3_cloner_adapter(profs_v3_fold, eq_mode="blind_tilt_anchor")
        ts = time.perf_counter()
        y_v9, _ = render_v9_utterance(pl, b_fold, profs_v3_fold[spk], cloner_fn=adapter_zh, src_profiles=profs_v3_fold, seed=72)
        rtf = (time.perf_counter() - ts) / max(1e-6, len(y_v9) / SR)
        zh_v9_comb_rows.append(_score_against_real(y_v9, xa, ev_r, rtf, fmin=90.0))
        if uid in ("SSB05020122", "SSB15930193"):
            ab_zh_clips[uid] = y_v9

    # 3. Suite B: n=20 Held-Out EN (slt[30:40], bdl[30:40]) with TimbreClonerV3
    print("[3/5] Running Suite B: n=20 EN Held-Out (target-first & cross-donor)...", flush=True)
    all_v3_for_bank = {**zh_v3_full, **en_v3_profs}
    adapter_en = make_v3_cloner_adapter(all_v3_for_bank, eq_mode="blind_tilt_anchor")
    v9_no_bdl = filter_v9_bank_excluding_speakers(v9_dual, {"bdl"})
    v9_no_slt = filter_v9_bank_excluding_speakers(v9_dual, {"slt"})

    en_real_rows, en_tgt_rows, en_cd_rows = [], [], []
    ab_en_clips = {}
    for u in en_heldout:
        uid, spk = u["id"], u["speaker"]
        fmin = 65.0 if spk == "bdl" else 110.0
        xa, ev_r = en_real_cache[uid]
        pl = cf.plan(u["text"], allow_oov_fallback=True)
        en_real_rows.append(_score_against_real(xa, xa, ev_r, 0.0, fmin=fmin))

        ts = time.perf_counter()
        y_tf, _ = render_v9_utterance(pl, v9_dual, en_v3_profs[spk], cloner_fn=adapter_en, src_profiles=all_v3_for_bank, seed=72)
        rtf_tf = (time.perf_counter() - ts) / max(1e-6, len(y_tf) / SR)
        en_tgt_rows.append(_score_against_real(y_tf, xa, ev_r, rtf_tf, fmin=fmin))

        cd_bank = v9_no_bdl if spk == "bdl" else v9_no_slt
        ts = time.perf_counter()
        y_cd, _ = render_v9_utterance(pl, cd_bank, en_v3_profs[spk], cloner_fn=adapter_en, src_profiles=all_v3_for_bank, seed=72)
        rtf_cd = (time.perf_counter() - ts) / max(1e-6, len(y_cd) / SR)
        en_cd_rows.append(_score_against_real(y_cd, xa, ev_r, rtf_cd, fmin=fmin))
        if uid in ("slt_0030", "bdl_0030"):
            ab_en_clips[uid] = {"real": xa, "target_first": y_tf, "cross_donor": y_cd}

    # 4. Suite C: 26 Frozen Benchmark Sentences x 3 speakers (78 clips)
    print("[4/5] Running Suite C: 26 Frozen Benchmark Sentences x 3 speakers (78 clips)...", flush=True)
    bm_doc = json.loads((ROOT / "exp_v5_arch/benchmark_texts.json").read_text(encoding="utf-8"))
    bm_items = bm_doc["sentences"] if isinstance(bm_doc, dict) else bm_doc
    bench_spks = {
        "SSB0502_3shot": tc3.extract_timbre_v3([(zh_pre_v9[u["id"]][0], zh_pre_v9[u["id"]][1]) for u in zh_utts if u["speaker"] == "SSB0502"][:3], sr=SR, name="SSB0502", corpus="aishell3"),
        "SSB0073_3shot": zh_new_v3_profs["SSB0073"],
        "slt_3shot": en_v3_profs["slt"],
    }
    bench_rows = {s: [] for s in bench_spks}
    bench_by_lang = {"zh": [], "en": [], "zh_en": []}
    cs01_clips = {}
    for it in bm_items:
        bid = it["id"]
        lang = it.get("lang") or it.get("category")
        pl = cf.plan(it["text"], allow_oov_fallback=True)
        for s_lbl, prof in bench_spks.items():
            fmin = 65.0 if "SSB0073" in s_lbl else 100.0
            ts = time.perf_counter()
            y, _ = render_v9_utterance(pl, v9_dual, prof, cloner_fn=adapter_en, src_profiles=all_v3_for_bank, seed=72)
            rtf = (time.perf_counter() - ts) / max(1e-6, len(y) / SR)
            ev_y = evaluate_audio(y, sr=SR, fmin=fmin, fmax=450.0)
            v9_y = compute_v9_extended_metrics(y, sr=SR, fmin=fmin, fmax=450.0)
            row = {
                "sanity_ok": bool(ev_y["sanity"]["ok"]),
                "f0_med": ev_y["f0_hnr"]["f0_median_hz"],
                "f0_err_st": abs(12.0 * math.log2(ev_y["f0_hnr"]["f0_median_hz"] / prof.f0_median_hz)) if ev_y["f0_hnr"]["f0_median_hz"] else float("nan"),
                "tilt_err_db_oct": abs(ev_y["ltas"]["tilt_db_oct"] - prof.tilt_db_per_oct) if ev_y["ltas"] else float("nan"),
                "flux_p95_med": v9_y["fluency"]["spectral_flux"]["spectral_flux_p95_to_median"],
                "vuv_flux_db": v9_y["fluency"]["vuv_transitions"]["vuv_flux_median_db"],
                "f0_declination_corr": v9_y["f0_declination_corr"],
                "final_lengthening_ratio": v9_y["final_lengthening_ratio"],
                "click_outlier_rate_gt8": v9_y["click_outlier_rate_gt8"],
                "rtf": rtf,
            }
            bench_rows[s_lbl].append(row)
            if lang in bench_by_lang:
                bench_by_lang[lang].append(row)
            if bid == "CS_01":
                cs01_clips[s_lbl] = y

    # 5. Suite D: Cross-lingual ZH <-> EN 4-case check + 4-way target ID
    print("[5/5] Running Suite D: Cross-lingual ZH <-> EN cloning & building listening_ab...", flush=True)
    zh_pl0 = v6_arb.plan_from_pinyin_syllables(zh_held_utts[0]["syllables"], text=zh_held_utts[0]["text"])
    xl_specs = [
        ("SSB0502_speaks_EN", cf.plan(en_heldout[0]["text"], allow_oov_fallback=True), v9_no_slt, bench_spks["SSB0502_3shot"], 100.0),
        ("SSB0073_M133_speaks_EN", cf.plan(en_heldout[10]["text"], allow_oov_fallback=True), v9_no_bdl, bench_spks["SSB0073_3shot"], 65.0),
        ("slt_F188_speaks_ZH", zh_pl0, v9_dual, en_v3_profs["slt"], 100.0),
        ("bdl_M119_speaks_ZH", zh_pl0, v9_dual, en_v3_profs["bdl"], 65.0),
    ]
    xl_results = {}
    xl_clips = {}
    for lbl, pl, b_use, tgt_p, fmin in xl_specs:
        ts = time.perf_counter()
        y, info = render_v9_utterance(pl, b_use, tgt_p, cloner_fn=adapter_en, src_profiles=all_v3_for_bank, seed=72)
        rtf = (time.perf_counter() - ts) / max(1e-6, len(y) / SR)
        ev_y = evaluate_audio(y, sr=SR, fmin=fmin, fmax=450.0)
        v9_y = compute_v9_extended_metrics(y, sr=SR, fmin=fmin, fmax=450.0)
        f0_y = ev_y["f0_hnr"]["f0_median_hz"]
        xl_results[lbl] = {
            "sanity_ok": bool(ev_y["sanity"]["ok"]),
            "target_f0_hz": round(tgt_p.f0_median_hz, 2),
            "synth_f0_hz": round(f0_y, 2) if f0_y else None,
            "f0_err_st": round(abs(12.0 * math.log2(f0_y / tgt_p.f0_median_hz)), 3) if f0_y else None,
            "target_tilt": round(tgt_p.tilt_db_per_oct, 3),
            "synth_tilt": round(ev_y["ltas"]["tilt_db_oct"], 3),
            "tilt_err_db_oct": round(abs(ev_y["ltas"]["tilt_db_oct"] - tgt_p.tilt_db_per_oct), 3),
            "flux_p95_med": v9_y["fluency"]["spectral_flux"]["spectral_flux_p95_to_median"],
            "click_outlier_rate_gt8": v9_y["click_outlier_rate_gt8"],
            "rtf": round(rtf, 4),
            "source_counts": info.get("source_counts"),
        }
        xl_clips[lbl] = y

    # Build loudness-matched listening_ab/ WAVs
    LISTENING_DIR.mkdir(parents=True, exist_ok=True)
    ab_manifest = []
    packs = [
        (
            "01_zh_evolution_SSB05020122.wav",
            "SSB05020122: Real -> v6_crossling_dsp -> v7_pybot -> v9_combined (v9_clarity + TimbreClonerV3)",
            [
                ("real", zh_eval_cache["SSB05020122"][0]),
                ("v6_crossling_dsp", read_wav(ROOT / "exp_v6_crossling_dsp/samples/matched_heldout/SSB05020122_v6_same.wav")[0]),
                ("v7_pybot", read_wav(ROOT / "exp_v7_pybot_tts/samples/heldout/SSB05020122_v7_same.wav")[0]),
                ("v9_combined", ab_zh_clips["SSB05020122"]),
            ],
        ),
        (
            "02_zh_evolution_SSB15930193.wav",
            "SSB15930193: Real -> v6_crossling_dsp -> v7_pybot -> v9_combined (v9_clarity + TimbreClonerV3)",
            [
                ("real", zh_eval_cache["SSB15930193"][0]),
                ("v6_crossling_dsp", read_wav(ROOT / "exp_v6_crossling_dsp/samples/matched_heldout/SSB15930193_v6_same.wav")[0]),
                ("v7_pybot", read_wav(ROOT / "exp_v7_pybot_tts/samples/heldout/SSB15930193_v7_same.wav")[0]),
                ("v9_combined", ab_zh_clips["SSB15930193"]),
            ],
        ),
        (
            "03_en_v9_slt_0030.wav",
            "English slt_0030: Real_slt -> v9_combined_target_first -> v9_combined_cross_donor",
            [
                ("real_slt", ab_en_clips["slt_0030"]["real"]),
                ("v9_target_first", ab_en_clips["slt_0030"]["target_first"]),
                ("v9_cross_donor", ab_en_clips["slt_0030"]["cross_donor"]),
            ],
        ),
        (
            "04_en_v9_bdl_0030.wav",
            "English bdl_0030: Real_bdl -> v9_combined_target_first -> v9_combined_cross_donor",
            [
                ("real_bdl", ab_en_clips["bdl_0030"]["real"]),
                ("v9_target_first", ab_en_clips["bdl_0030"]["target_first"]),
                ("v9_cross_donor", ab_en_clips["bdl_0030"]["cross_donor"]),
            ],
        ),
        (
            "05_crossling_zh_en_v9_cloning.wav",
            "Cross-lingual v9_combined: SSB0502_EN -> SSB0073_M133_EN -> slt_F188_ZH -> bdl_M119_ZH",
            [
                ("SSB0502_speaks_EN", xl_clips["SSB0502_speaks_EN"]),
                ("SSB0073_M133_speaks_EN", xl_clips["SSB0073_M133_speaks_EN"]),
                ("slt_F188_speaks_ZH", xl_clips["slt_F188_speaks_ZH"]),
                ("bdl_M119_speaks_ZH", xl_clips["bdl_M119_speaks_ZH"]),
            ],
        ),
        (
            "06_codeswitch_v6_v7_v9_CS01.wav",
            "Code-switched CS_01: v6_dsp_0502 -> v7_pybot_0502 -> v9_combined_0502 -> v9_combined_SSB0073_M133",
            [
                ("v6_dsp_SSB0502", read_wav(ROOT / "exp_v6_crossling_dsp/samples/benchmark/CS_01_SSB0502_3shot.wav")[0]),
                ("v7_pybot_SSB0502", read_wav(ROOT / "exp_v7_pybot_tts/samples/benchmark/CS_01_SSB0502_3shot.wav")[0]),
                ("v9_combined_SSB0502", cs01_clips["SSB0502_3shot"]),
                ("v9_combined_SSB0073_M133", cs01_clips["SSB0073_3shot"]),
            ],
        ),
    ]
    for fname, desc, segs in packs:
        wav, tl = _concat_ab(segs)
        out_p = LISTENING_DIR / fname
        _write_pcm16_wav(out_p, wav, SR)
        san = check_wav_sanity(wav, SR)
        ab_manifest.append({"file": fname, "description": desc, "duration_s": san["duration_s"], "sanity_ok": san["ok"], "rms_dbfs": san["rms_dbfs"], "peak": san["peak"], "segments": tl})

    # Load prior versions for unified v4->v9 comparison matrix
    v5_res = json.loads((ROOT / "exp_v5_timbre/results.json").read_text(encoding="utf-8"))
    v7_sum = json.loads((ROOT / "exp_v7_pybot_tts/listening_ab/comparison_summary.json").read_text(encoding="utf-8"))["matrix"]
    v8_res = json.loads((ROOT / "exp_v8_clarity/results.json").read_text(encoding="utf-8"))
    v9_clar = json.loads((ROOT / "exp_v9_clarity_pro/results.json").read_text(encoding="utf-8"))

    zh_comb_sum = _summarize_rows(zh_v9_comb_rows)
    zh_real_sum = _summarize_rows(zh_real_rows)
    en_real_sum = _summarize_rows(en_real_rows)
    en_tf_sum = _summarize_rows(en_tgt_rows)
    en_cd_sum = _summarize_rows(en_cd_rows)

    progression = {
        "Real_ZH_26": {
            "mel_lsd_db": 0.0, "mcd_db": 0.0,
            "df2_dt_hz_s": zh_real_sum["medians"].get("df2_dt_hz_s"),
            "flux_p95_med": zh_real_sum["medians"].get("flux_p95_med"),
            "vuv_flux_db": zh_real_sum["medians"].get("vuv_flux_db"),
            "f0_declination_corr": zh_real_sum["medians"].get("f0_declination_corr"),
            "cv_energy_contrast_db": zh_real_sum["medians"].get("cv_energy_contrast_db"),
            "click_outlier_rate": zh_real_sum["medians"].get("click_outlier_rate_gt8"),
        },
        "v4_baseline": {
            "mel_lsd_db": v7_sum["v4_baseline"]["mel_lsd_db"],
            "mcd_db": v7_sum["v4_baseline"]["mcd_db"],
            "df2_dt_hz_s": v7_sum["v4_baseline"]["df2_dt_hz_s"],
            "flux_p95_med": v7_sum["v4_baseline"]["flux_p95_med"],
            "vuv_flux_db": v7_sum["v4_baseline"]["vuv_flux_db"],
            "cpu_rtf": v7_sum["v4_baseline"]["cpu_rtf"],
        },
        "v5_full": {
            "mel_lsd_db": v7_sum["v5_full"]["mel_lsd_db"],
            "mcd_db": v7_sum["v5_full"]["mcd_db"],
            "df2_dt_hz_s": v7_sum["v5_full"]["df2_dt_hz_s"],
            "flux_p95_med": v7_sum["v5_full"]["flux_p95_med"],
            "vuv_flux_db": v7_sum["v5_full"]["vuv_flux_db"],
            "cpu_rtf": v7_sum["v5_full"]["cpu_rtf"],
        },
        "v6_crossling_dsp": {
            "mel_lsd_db": v7_sum["v6_crossling_dsp"]["mel_lsd_db"],
            "mcd_db": v7_sum["v6_crossling_dsp"]["mcd_db"],
            "df2_dt_hz_s": v7_sum["v6_crossling_dsp"]["df2_dt_hz_s"],
            "flux_p95_med": v7_sum["v6_crossling_dsp"]["flux_p95_med"],
            "vuv_flux_db": v7_sum["v6_crossling_dsp"]["vuv_flux_db"],
            "cpu_rtf": v7_sum["v6_crossling_dsp"]["cpu_rtf"],
        },
        "v7_pybot_tts": {
            "mel_lsd_db": v7_sum["v7_pybot_tts_loo"]["mel_lsd_db"],
            "mcd_db": v7_sum["v7_pybot_tts_loo"]["mcd_db"],
            "df2_dt_hz_s": v7_sum["v7_pybot_tts_loo"]["df2_dt_hz_s"],
            "flux_p95_med": v7_sum["v7_pybot_tts_loo"]["flux_p95_med"],
            "vuv_flux_db": v7_sum["v7_pybot_tts_loo"]["vuv_flux_db"],
            "cpu_rtf": v7_sum["v7_pybot_tts_loo"]["cpu_rtf"],
        },
        "v8_utau_hnm": {
            "mel_lsd_db": v8_res["summary"]["v8_utau_hnm"]["medians"]["mel_lsd_db"],
            "mcd_db": v8_res["summary"]["v8_utau_hnm"]["medians"]["mcd_db"],
            "flux_p95_med": v8_res["summary"]["v8_utau_hnm"]["medians"]["flux_p95_med"],
            "vuv_flux_db": v8_res["summary"]["v8_utau_hnm"]["medians"]["vuv_flux_db"],
            "cpu_rtf": v8_res["summary"]["v8_utau_hnm"]["medians"]["rtf"],
        },
        "v9_clarity_pro_v2": v9_clar["zh_loo_n26"]["v9_full"]["medians"],
        "v9_combined_zh_n26": zh_comb_sum["medians"],
    }

    report = {
        "schema": 1,
        "elapsed_s": round(time.time() - t0, 2),
        "neural_weights_mb": 0.0,
        "progression_zh_v4_to_v9": progression,
        "suite_a_zh_n26": {"real": zh_real_sum, "v9_combined": zh_comb_sum},
        "suite_b_en_n20": {"real": en_real_sum, "v9_combined_target_first": en_tf_sum, "v9_combined_cross_donor": en_cd_sum},
        "suite_c_benchmark_78": {
            "by_speaker": {s: _summarize_rows(r) for s, r in bench_rows.items()},
            "by_lang": {l: _summarize_rows(r) for l, r in bench_by_lang.items()},
        },
        "suite_d_crosslingual_zh_en": xl_results,
        "listening_ab_manifest": ab_manifest,
    }
    out_p = EVAL_DIR / "v9_benchmark_report.json"
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("WROTE", out_p, f"in {report['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
