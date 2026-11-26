"""Phase 7 End-to-End Benchmark: v10_full vs v11 (GMM + Diphone + Smooth HNM) vs Real."""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time
import wave
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_eval.metrics import check_wav_sanity, compute_spectral_distances, evaluate_audio, read_wav  # noqa: E402
from exp_v6_crossling_dsp import arbiter as v6_arb  # noqa: E402
from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v7_pybot_tts import unit_bank as ub  # noqa: E402
from exp_v8_clarity.en_utau_bank import _trim_active_span  # noqa: E402
from exp_v9_clarity_pro.hnm_v9 import analyze_v9  # noqa: E402
from exp_v9_clarity_pro.viterbi_utau import V9DualBank  # noqa: E402
from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10  # noqa: E402
from exp_v10_g2p_and_align.timbre_cloner_v4 import adapt_features_v4, extract_timbre_v4  # noqa: E402
from exp_v10_integrity.v10_utau_renderer import build_v10_full_bank, render_v10_utterance  # noqa: E402
from exp_v11_align_gmm.gmm_hmm_aligner import align_en_corpus_gmm, align_zh_corpus_gmm  # noqa: E402
from exp_v11_align_gmm.timbre_cloner_v5 import adapt_features_v5, extract_timbre_v5  # noqa: E402
from exp_v11_diphone_tts.diphone_bank import V11DiphoneDualBank, build_v11_dual_bank  # noqa: E402
from exp_v11_diphone_tts.v11_renderer import render_v11_utterance  # noqa: E402
from exp_v11_eval.v11_metrics import evaluate_v11_utterance  # noqa: E402

SR = 24000
EVAL_DIR = pathlib.Path(__file__).resolve().parent
LISTENING_DIR = EVAL_DIR / "listening_ab"
REPORT_PATH = EVAL_DIR / "v11_benchmark_report.json"


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


def _concat_ab(segments: list[tuple[str, np.ndarray]], gap_s: float = 0.45) -> tuple[np.ndarray, list[dict]]:
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


def _fold_v10_bank_for_zh(bank: V9DualBank, uid: str, ub_fold: ub.UnitBank, spk: str, prof_fold) -> V9DualBank:
    profs3 = dict(bank.zh_profiles_v3)
    profs3[spk] = prof_fold
    return V9DualBank(
        zh_syllables={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in bank.zh_syllables.items()) if v},
        zh_onsets={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in bank.zh_onsets.items()) if v},
        zh_finals={k: v for k, v in ((k, [t for t in toks if t.utterance_id != uid]) for k, toks in bank.zh_finals.items()) if v},
        en_phones=bank.en_phones,
        zh_profiles_v2=bank.zh_profiles_v2,
        en_profiles_v2=bank.en_profiles_v2,
        zh_profiles_v3=profs3,
        en_profiles_v3=bank.en_profiles_v3,
        fallback_unit_bank=ub_fold,
        prosody_rules=bank.prosody_rules,
        zh_included_ids=[i for i in bank.zh_included_ids if i != uid],
        zh_excluded_ids=sorted(set(bank.zh_excluded_ids) | {uid}),
        en_included_ids=list(bank.en_included_ids),
        en_excluded_ids=list(bank.en_excluded_ids),
    )


def _fold_v11_bank_for_zh(bank: V11DiphoneDualBank, uid: str, ub_fold: ub.UnitBank, spk: str, prof5_fold) -> V11DiphoneDualBank:
    profs5 = dict(bank.profiles_v5)
    profs5[spk] = prof5_fold
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


def _filter_v11_excluding_en_speaker(bank: V11DiphoneDualBank, excl_spk: str) -> V11DiphoneDualBank:
    return V11DiphoneDualBank(
        en_diphones={k: v for k, v in ((k, [t for t in toks if t.speaker != excl_spk]) for k, toks in bank.en_diphones.items()) if v},
        en_phones={k: v for k, v in ((k, [t for t in toks if t.speaker != excl_spk]) for k, toks in bank.en_phones.items()) if v},
        zh_syllables=bank.zh_syllables,
        zh_onsets=bank.zh_onsets,
        zh_finals=bank.zh_finals,
        profiles_v2={k: v for k, v in bank.profiles_v2.items() if k != excl_spk},
        profiles_v5={k: v for k, v in bank.profiles_v5.items() if k != excl_spk},
        fallback_unit_bank=bank.fallback_unit_bank,
        zh_included_ids=list(bank.zh_included_ids),
        zh_excluded_ids=list(bank.zh_excluded_ids),
        en_included_ids=list(bank.en_included_ids),
        en_excluded_ids=sorted(set(bank.en_excluded_ids) | {excl_spk}),
        stats=dict(bank.stats),
    )


def _score_row(
    y: np.ndarray,
    ref: np.ndarray | None,
    ev_ref: dict | None,
    plan_doc: dict,
    rtf: float,
    *,
    info: dict | None = None,
    fmin: float = 75.0,
) -> dict[str, Any]:
    info = info or {}
    ev_y = evaluate_audio(y, sr=SR, fmin=fmin, fmax=450.0)
    v11_u = evaluate_v11_utterance(
        y,
        plan_doc,
        sr=SR,
        fmin=fmin,
        fmax=450.0,
        slot_timings_ms=info.get("slot_timings_ms"),
        unit_boundaries_ms=info.get("unit_boundaries_ms"),
        log_env=info.get("log_env"),
        boundary_frames=info.get("boundary_frames"),
    )
    sd = compute_spectral_distances(ref, y, sr=SR) if ref is not None else None

    f0_y = ev_y["f0_hnr"]["f0_median_hz"] if ev_y["f0_hnr"] else None
    f0_r = ev_ref["f0_hnr"]["f0_median_hz"] if (ev_ref and ev_ref["f0_hnr"]) else None
    f0_err = abs(12.0 * math.log2(f0_y / f0_r)) if (f0_y and f0_r) else float("nan")
    hnr_y = ev_y["f0_hnr"]["hnr_db_median"] if ev_y["f0_hnr"] else None
    hnr_r = ev_ref["f0_hnr"]["hnr_db_median"] if (ev_ref and ev_ref["f0_hnr"]) else None
    tilt_y = ev_y["ltas"]["tilt_db_oct"] if ev_y["ltas"] else None
    tilt_r = ev_ref["ltas"]["tilt_db_oct"] if (ev_ref and ev_ref["ltas"]) else None

    lj = v11_u["loudness_jumps"]
    nd = v11_u["nondrop"]
    bs = v11_u["boundary_smoothness"]
    rh = v11_u["en_rhythm"]
    v9 = v11_u["v9"]

    sc = info.get("source_counts") or {}
    n_zh_tot = sc.get("zh_target_syl", 0) + sc.get("zh_donor_syl", 0) + sc.get("zh_concat_of", 0) + sc.get("zh_ipa_fill", 0)
    n_zh_whole = sc.get("zh_target_syl", 0) + sc.get("zh_donor_syl", 0)

    return {
        "sanity_ok": bool(v11_u["sanity_ok"]),
        "mel_lsd_db": float(sd["mel_lsd_db"]) if sd else float("nan"),
        "mcd_db": float(sd["mcd_db"]) if sd else float("nan"),
        "mcd_cmn_db": float(sd["mcd_cmn_db"]) if sd else float("nan"),
        "f0_err_st": float(f0_err),
        "tilt_err_db_oct": abs(tilt_y - tilt_r) if (tilt_y is not None and tilt_r is not None) else float("nan"),
        "hnr_err_db": abs(hnr_y - hnr_r) if (hnr_y is not None and hnr_r is not None) else float("nan"),
        "voiced_contrast_db": v9["clarity"]["voiced_spectral_contrast_db"],
        "uv_2_8k_vs_lo_db": v9["clarity"]["unvoiced_2_8k_vs_lo_db"],
        "vuv_flux_db": v9["fluency"]["vuv_transitions"]["vuv_flux_median_db"],
        "f0_declination_corr": v9["f0_declination_corr"],
        "click_outlier_rate_gt8": v9["click_outlier_rate_gt8"],
        "loudness_p95_delta_db": lj["p95_abs_delta_db"],
        "loudness_p99_delta_db": lj["p99_abs_delta_db"],
        "loudness_max_delta_db": lj["max_abs_delta_db"],
        "loudness_gt6db_per_s": lj["jumps_gt6db_per_s"],
        "loudness_gt9db_per_s": lj["jumps_gt9db_per_s"],
        "n_gt6db": lj["n_gt6db"],
        "n_gt9db": lj["n_gt9db"],
        "n_dropped_slots": nd["n_dropped_slots"],
        "syllable_peak_recall": nd["syllable_peak_recall"],
        "min_slot_active_ms": nd["min_slot_active_ms"],
        "min_slot_rms_dbfs": nd["min_slot_rms_dbfs"],
        "join_mel_jump_median_db": bs["join_mel_jump_median_db"],
        "feature_join_log_env_l2_db": bs["feature_join_log_env_l2_db"],
        "frame5ms_mel_jump_p95_db": bs["frame5ms_mel_jump_p95_db"],
        "func_to_content_dur_ratio": rh["func_to_content_dur_ratio"],
        "content_minus_func_db": rh["content_minus_func_db"],
        "en_diphone_exact_pct": info.get("en_diphone_exact_pct"),
        "zh_whole_syl_pct": round(100.0 * n_zh_whole / n_zh_tot, 2) if n_zh_tot else None,
        "rtf": float(rtf),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    meds = {}
    keys = [k for k in rows[0] if k not in ("sanity_ok", "n_gt6db", "n_gt9db", "n_dropped_slots")]
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None and isinstance(r[k], (int, float)) and math.isfinite(r[k])]
        if vals:
            meds[k] = round(float(np.median(vals)), 4)
    return {
        "n": len(rows),
        "sanity_ok": sum(1 for r in rows if r["sanity_ok"]),
        "total_gt6db_jumps": sum(int(r["n_gt6db"]) for r in rows),
        "total_gt9db_jumps": sum(int(r["n_gt9db"]) for r in rows),
        "total_dropped_slots": sum(int(r["n_dropped_slots"]) for r in rows),
        "medians": meds,
    }


def main() -> int:
    t0 = time.time()
    zh_utts = json.loads((ROOT / "pybot_tts_v2_recorded/extended/index.json").read_text(encoding="utf-8"))["utterances"]
    man_all = json.loads((ROOT / "exp_v8_data/corpus/manifest.json").read_text(encoding="utf-8"))
    en_man = man_all["english"]["utterances"]
    zh_new_man = man_all.get("mandarin", {}).get("utterances", [])

    en_heldout = [u for u in en_man if u["speaker"] in ("slt", "bdl") and int(u["id"].split("_")[1]) >= 30]
    en_excl_ids = {u["id"] for u in en_heldout}

    print("[1/6] Running GMM-HMM alignments (140 EN train + 32 ZH) & building v10_full / v11 banks...", flush=True)
    en_gmm = align_en_corpus_gmm(exclude_ids=en_excl_ids, align_heldout=False, n_em=4, sr=SR)
    zh_gmm = align_zh_corpus_gmm(exclude_ids=None, n_em=4, sr=SR)

    zh_pre, zh_real_eval = {}, {}
    for u in zh_utts:
        x, _ = read_wav(ROOT / u["path"])
        xa = x[int(u["active_start"]) : int(u["active_stop"])]
        fv9, bg = analyze_v9(xa, sr=SR)
        zh_pre[u["id"]] = (xa, fv9, bg)
        zh_real_eval[u["id"]] = (xa, evaluate_audio(xa, sr=SR, fmin=90.0, fmax=450.0))

    en_pre_v11 = {}
    for u in en_man:
        if u["id"] in en_excl_ids:
            continue
        g_e = en_gmm["utterances"][u["id"]]
        xa = g_e["wav"]
        fv9, bg = analyze_v9(xa, sr=SR)
        en_pre_v11[u["id"]] = (xa, fv9, bg)

    b10_full, profs_v4, _ = build_v10_full_bank(
        zh_utterances=zh_utts, zh_exclude_ids=(), en_exclude_ids=en_excl_ids, extract_v4_profiles=True,
    )
    b11_full = build_v11_dual_bank(
        en_gmm_alignments=en_gmm,
        zh_gmm_alignments=zh_gmm,
        zh_utterances=zh_utts,
        zh_exclude_ids=(),
        en_utterances=en_man,
        en_exclude_ids=en_excl_ids,
        zh_precomputed=zh_pre,
        en_precomputed=en_pre_v11,
        extract_v5_profiles=True,
    )

    def _fewshot_profiles(rows: list[dict], spk: str, v4_corpus: str, v5_corpus: str):
        items = []
        for r in rows[:3]:
            x, _ = read_wav(ROOT / r["path"])
            s0, s1 = _trim_active_span(x, sr=SR)
            xa = x[s0:s1]
            fv9, _ = analyze_v9(xa, sr=SR)
            items.append((xa, fv9))
        return extract_timbre_v4(items, sr=SR, name=spk, corpus=v4_corpus), extract_timbre_v5(items, sr=SR, name=spk, corpus=v5_corpus)

    slt_train = [u for u in en_man if u["speaker"] == "slt" and u["id"] not in en_excl_ids]
    bdl_train = [u for u in en_man if u["speaker"] == "bdl" and u["id"] not in en_excl_ids]
    v4_slt_3s, v5_slt_3s = _fewshot_profiles(slt_train, "slt", "cmu_arctic", "arctic")
    v4_bdl_3s, v5_bdl_3s = _fewshot_profiles(bdl_train, "bdl", "cmu_arctic", "arctic")
    ssb0073_rows = [u for u in zh_new_man if u["speaker"] == "SSB0073"]
    v4_0073, v5_0073 = _fewshot_profiles(ssb0073_rows, "SSB0073", "aishell3", "aishell3")

    # 2. Suite A: ZH n=26 Held-Out LOO-CV (Real vs v10_full vs v11)
    print("[2/6] Suite A: ZH LOO-CV (n=26, Real vs v10_full vs v11)...", flush=True)
    zh_prompt_ids = {"SSB05020005", "SSB05020077", "SSB05020082", "SSB15930031", "SSB15930079", "SSB15930105"}
    zh_held_utts = [u for u in zh_utts if u["id"] not in zh_prompt_ids]
    zh_for_ub = {k: (v[0], v[1]) for k, v in zh_pre.items()}

    zh_rows = {"real": [], "v10_full": [], "v11": []}
    ab_zh = {}
    ssb05020122_detail = {}

    for u in zh_held_utts:
        uid, spk = u["id"], u["speaker"]
        xa, ev_r = zh_real_eval[uid]
        pl = v6_arb.plan_from_pinyin_syllables(u["syllables"], text=u["text"])
        zh_rows["real"].append(_score_row(xa, xa, ev_r, pl, 0.0, fmin=90.0))

        ub_fold = ub.build_unit_bank([x for x in zh_utts if x["id"] != uid], exclude_ids={uid}, precomputed=zh_for_ub, sr=SR)
        fold_items = [(zh_pre[x["id"]][0], zh_pre[x["id"]][1]) for x in zh_utts if x["speaker"] == spk and x["id"] != uid]
        p4_fold = extract_timbre_v4(fold_items, sr=SR, name=spk, corpus="aishell3")
        p5_fold = extract_timbre_v5(fold_items, sr=SR, name=spk, corpus="aishell3")
        p4_map = {**profs_v4, spk: p4_fold}
        p5_map = {**b11_full.profiles_v5, spk: p5_fold}

        b10_fold = _fold_v10_bank_for_zh(b10_full, uid, ub_fold, spk, p4_fold)
        b11_fold = _fold_v11_bank_for_zh(b11_full, uid, ub_fold, spk, p5_fold)

        ts = time.perf_counter()
        y10, m10 = render_v10_utterance(pl, b10_fold, p4_fold, cloner_fn=adapt_features_v4, src_profiles=p4_map, seed=72)
        r10 = (time.perf_counter() - ts) / max(1e-6, len(y10) / SR)
        row10 = _score_row(y10, xa, ev_r, pl, r10, info=m10, fmin=90.0)
        zh_rows["v10_full"].append(row10)

        ts = time.perf_counter()
        y11, m11 = render_v11_utterance(pl, b11_fold, p5_fold, cloner_fn=adapt_features_v5, src_profiles=p5_map, seed=72)
        r11 = (time.perf_counter() - ts) / max(1e-6, len(y11) / SR)
        row11 = _score_row(y11, xa, ev_r, pl, r11, info=m11, fmin=90.0)
        zh_rows["v11"].append(row11)

        if uid in ("SSB05020122", "SSB15930193"):
            ab_zh[uid] = {"real": xa, "v10_full": y10, "v11": y11}
        if uid == "SSB05020122":
            nd10 = evaluate_v11_utterance(y10, pl, sr=SR, fmin=90.0)["nondrop"]
            nd11 = evaluate_v11_utterance(y11, pl, sr=SR, fmin=90.0, slot_timings_ms=m11.get("slot_timings_ms"))["nondrop"]
            ssb05020122_detail = {
                "text": u["text"],
                "v10_source_counts": m10.get("source_counts"),
                "v10_nondrop": nd10,
                "v11_source_counts": m11.get("source_counts"),
                "v11_nondrop": nd11,
                "v10_metrics": row10,
                "v11_metrics": row11,
            }

    # 3. Suite B: EN Held-Out n=20 (Real vs v10_full vs v11 target-first vs v11 cross-donor)
    print("[3/6] Suite B: EN Held-out (n=20, Real vs v10_full vs v11 vs v11_cross_donor)...", flush=True)
    b11_no_slt = _filter_v11_excluding_en_speaker(b11_full, "slt")
    b11_no_bdl = _filter_v11_excluding_en_speaker(b11_full, "bdl")
    en_rows = {"real": [], "v10_full": [], "v11": [], "v11_cross_donor": []}
    ab_en = {}

    for u in en_heldout:
        uid, spk = u["id"], u["speaker"]
        fmin = 65.0 if spk == "bdl" else 110.0
        x, _ = read_wav(ROOT / u["path"])
        s0, s1 = _trim_active_span(x, sr=SR)
        xa = x[s0:s1]
        ev_r = evaluate_audio(xa, sr=SR, fmin=fmin, fmax=450.0)
        pl = plan_v10(u["text"])
        en_rows["real"].append(_score_row(xa, xa, ev_r, pl, 0.0, fmin=fmin))

        t4 = v4_slt_3s if spk == "slt" else v4_bdl_3s
        t5 = v5_slt_3s if spk == "slt" else v5_bdl_3s

        ts = time.perf_counter()
        y10, m10 = render_v10_utterance(pl, b10_full, t4, cloner_fn=adapt_features_v4, src_profiles=profs_v4, seed=72)
        en_rows["v10_full"].append(_score_row(y10, xa, ev_r, pl, (time.perf_counter() - ts) / max(1e-6, len(y10) / SR), info=m10, fmin=fmin))

        ts = time.perf_counter()
        y11, m11 = render_v11_utterance(pl, b11_full, t5, cloner_fn=adapt_features_v5, src_profiles=b11_full.profiles_v5, seed=72)
        en_rows["v11"].append(_score_row(y11, xa, ev_r, pl, (time.perf_counter() - ts) / max(1e-6, len(y11) / SR), info=m11, fmin=fmin))

        b_cd = b11_no_slt if spk == "slt" else b11_no_bdl
        ts = time.perf_counter()
        y11_cd, m11_cd = render_v11_utterance(pl, b_cd, t5, cloner_fn=adapt_features_v5, src_profiles=b11_full.profiles_v5, seed=72)
        en_rows["v11_cross_donor"].append(_score_row(y11_cd, xa, ev_r, pl, (time.perf_counter() - ts) / max(1e-6, len(y11_cd) / SR), info=m11_cd, fmin=fmin))

        if uid in ("slt_0030", "bdl_0030"):
            ab_en[uid] = {"real": xa, "v10_full": y10, "v11": y11, "v11_cross_donor": y11_cd}

    # 4. Suite C: Code-Switching CS_01..CS_06 (v10_full vs v11 on SSB0502 & SSB0073)
    print("[4/6] Suite C: Code-Switching CS_01..CS_06 (v10_full vs v11)...", flush=True)
    btexts = json.loads((ROOT / "exp_v5_arch/benchmark_texts.json").read_text(encoding="utf-8"))
    cs_items = [s for s in btexts.get("sentences", []) if s.get("lang") == "zh_en"]
    cs_rows = {"v10_full": [], "v11": []}
    ab_cs = {}
    for idx_c, item in enumerate(cs_items):
        pl = plan_v10(item["text"])
        for label_spk, t4, t5, fmin in [
            ("SSB0502", profs_v4["SSB0502"], b11_full.profiles_v5["SSB0502"], 100.0),
            ("SSB0073", v4_0073, v5_0073, 65.0),
        ]:
            ts = time.perf_counter()
            y10, m10 = render_v10_utterance(pl, b10_full, t4, cloner_fn=adapt_features_v4, src_profiles=profs_v4, seed=72)
            cs_rows["v10_full"].append(_score_row(y10, None, None, pl, (time.perf_counter() - ts) / max(1e-6, len(y10) / SR), info=m10, fmin=fmin))

            ts = time.perf_counter()
            y11, m11 = render_v11_utterance(pl, b11_full, t5, cloner_fn=adapt_features_v5, src_profiles=b11_full.profiles_v5, seed=72)
            cs_rows["v11"].append(_score_row(y11, None, None, pl, (time.perf_counter() - ts) / max(1e-6, len(y11) / SR), info=m11, fmin=fmin))
            if idx_c == 0:
                ab_cs[label_spk] = (y10, y11)

    # 5. Suite D: Cross-lingual ZH <-> EN (v10_full vs v11)
    print("[5/6] Suite D: Cross-lingual ZH <-> EN (v10_full vs v11)...", flush=True)
    xl_en_sample = [u for u in en_heldout if u["speaker"] == "slt"][:5]
    xl_zh_sample = [u for u in zh_held_utts if u["speaker"] == "SSB0502"][:5]
    xl_rows = {"zh_to_en_v10_full": [], "zh_to_en_v11": [], "en_to_zh_v10_full": [], "en_to_zh_v11": []}
    ab_xl = {}

    for u in xl_en_sample:
        pl = plan_v10(u["text"])
        for tag, t4, t5, fmin in [
            ("SSB0502_speaks_EN", profs_v4["SSB0502"], b11_full.profiles_v5["SSB0502"], 100.0),
            ("SSB0073_M133_speaks_EN", v4_0073, v5_0073, 65.0),
        ]:
            ts = time.perf_counter()
            y10, m10 = render_v10_utterance(pl, b10_full, t4, cloner_fn=adapt_features_v4, src_profiles=profs_v4, seed=72)
            xl_rows["zh_to_en_v10_full"].append(_score_row(y10, None, None, pl, (time.perf_counter() - ts) / max(1e-6, len(y10) / SR), info=m10, fmin=fmin))

            ts = time.perf_counter()
            y11, m11 = render_v11_utterance(pl, b11_full, t5, cloner_fn=adapt_features_v5, src_profiles=b11_full.profiles_v5, seed=72)
            xl_rows["zh_to_en_v11"].append(_score_row(y11, None, None, pl, (time.perf_counter() - ts) / max(1e-6, len(y11) / SR), info=m11, fmin=fmin))
            if u["id"] == "slt_0030":
                ab_xl[tag] = y11

    for u in xl_zh_sample:
        pl = v6_arb.plan_from_pinyin_syllables(u["syllables"], text=u["text"])
        for tag, t4, t5, fmin in [
            ("slt_F188_speaks_ZH", v4_slt_3s, v5_slt_3s, 110.0),
            ("bdl_M119_speaks_ZH", v4_bdl_3s, v5_bdl_3s, 65.0),
        ]:
            ts = time.perf_counter()
            y10, m10 = render_v10_utterance(pl, b10_full, t4, cloner_fn=adapt_features_v4, src_profiles=profs_v4, seed=72)
            xl_rows["en_to_zh_v10_full"].append(_score_row(y10, None, None, pl, (time.perf_counter() - ts) / max(1e-6, len(y10) / SR), info=m10, fmin=fmin))

            ts = time.perf_counter()
            y11, m11 = render_v11_utterance(pl, b11_full, t5, cloner_fn=adapt_features_v5, src_profiles=b11_full.profiles_v5, seed=72)
            xl_rows["en_to_zh_v11"].append(_score_row(y11, None, None, pl, (time.perf_counter() - ts) / max(1e-6, len(y11) / SR), info=m11, fmin=fmin))
            if u["id"] == xl_zh_sample[0]["id"]:
                ab_xl[tag] = y11

    # 6. Export loudness-matched listening_ab WAVs
    print("[6/6] Exporting loudness-matched listening_ab WAVs & saving report...", flush=True)
    ab_specs = [
        ("01_zh_SSB05020122_real_v10_v11.wav", "SSB05020122 ZH LOO: real -> v10_full -> v11", [
            ("real", ab_zh["SSB05020122"]["real"]),
            ("v10_full", ab_zh["SSB05020122"]["v10_full"]),
            ("v11", ab_zh["SSB05020122"]["v11"]),
        ]),
        ("02_zh_SSB15930193_real_v10_v11.wav", "SSB15930193 ZH LOO: real -> v10_full -> v11", [
            ("real", ab_zh["SSB15930193"]["real"]),
            ("v10_full", ab_zh["SSB15930193"]["v10_full"]),
            ("v11", ab_zh["SSB15930193"]["v11"]),
        ]),
        ("03_en_slt_0030_real_v10_v11.wav", "slt_0030 EN Held-out: real -> v10_full -> v11 -> v11_cross_donor", [
            ("real_slt", ab_en["slt_0030"]["real"]),
            ("v10_full", ab_en["slt_0030"]["v10_full"]),
            ("v11", ab_en["slt_0030"]["v11"]),
            ("v11_cross_donor", ab_en["slt_0030"]["v11_cross_donor"]),
        ]),
        ("04_en_bdl_0030_real_v10_v11.wav", "bdl_0030 EN Held-out: real -> v10_full -> v11 -> v11_cross_donor", [
            ("real_bdl", ab_en["bdl_0030"]["real"]),
            ("v10_full", ab_en["bdl_0030"]["v10_full"]),
            ("v11", ab_en["bdl_0030"]["v11"]),
            ("v11_cross_donor", ab_en["bdl_0030"]["v11_cross_donor"]),
        ]),
        ("05_crossling_zh_en_v11.wav", "Cross-lingual v11: SSB0502_EN -> SSB0073_EN -> slt_ZH -> bdl_ZH", [
            ("SSB0502_speaks_EN", ab_xl["SSB0502_speaks_EN"]),
            ("SSB0073_M133_speaks_EN", ab_xl["SSB0073_M133_speaks_EN"]),
            ("slt_F188_speaks_ZH", ab_xl["slt_F188_speaks_ZH"]),
            ("bdl_M119_speaks_ZH", ab_xl["bdl_M119_speaks_ZH"]),
        ]),
        ("06_codeswitch_v10_vs_v11_CS01.wav", "CS_01: v10_full_SSB0502 -> v11_SSB0502 -> v10_full_SSB0073 -> v11_SSB0073", [
            ("v10_full_SSB0502", ab_cs["SSB0502"][0]),
            ("v11_SSB0502", ab_cs["SSB0502"][1]),
            ("v10_full_SSB0073", ab_cs["SSB0073"][0]),
            ("v11_SSB0073", ab_cs["SSB0073"][1]),
        ]),
    ]
    manifest = []
    for fname, desc, segs in ab_specs:
        wav_arr, tl = _concat_ab(segs, gap_s=0.45)
        out_p = LISTENING_DIR / fname
        _write_pcm16_wav(out_p, wav_arr, sr=SR)
        san = check_wav_sanity(wav_arr, SR)
        manifest.append(
            {
                "file": fname,
                "description": desc,
                "duration_s": round(len(wav_arr) / float(SR), 3),
                "sanity_ok": bool(san["ok"]),
                "rms_dbfs": round(float(san["rms_dbfs"]), 2),
                "peak": round(float(san["peak"]), 4),
                "segments": tl,
            }
        )

    report = {
        "runtime_s": round(time.time() - t0, 2),
        "bank_stats_v11": b11_full.stats,
        "suite_a_zh_loo_n26": {k: _summarize(v) for k, v in zh_rows.items()},
        "ssb05020122_nondrop_spotlight": ssb05020122_detail,
        "suite_b_en_heldout_n20": {k: _summarize(v) for k, v in en_rows.items()},
        "suite_c_codeswitch_n12": {k: _summarize(v) for k, v in cs_rows.items()},
        "suite_d_crossling_zh_en": {k: _summarize(v) for k, v in xl_rows.items()},
        "listening_ab_manifest": manifest,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {REPORT_PATH} in {report['runtime_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
