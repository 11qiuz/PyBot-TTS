"""Phase 4 Clarity Root-Cause Ablation and UTAU-Style Full-Resolution Evaluation (Zero Leakage)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v5_eval import metrics as ev  # noqa: E402
from exp_v6_crossling_dsp import arbiter as v6_arb  # noqa: E402
from exp_v6_eval import fluency as fl  # noqa: E402
from exp_v6_vocoder import SR, cloner, vocoder  # noqa: E402
from exp_v7_pybot_tts import planner_bridge as v7_pb, unit_bank as v7_ub  # noqa: E402
from exp_v8_clarity import EXP_DIR, EXTENDED_INDEX, SAMPLES_DIR  # noqa: E402
from exp_v8_clarity.clarity_metrics import compute_clarity_metrics  # noqa: E402
from exp_v8_clarity.utau_concat import build_fullres_bank, render_utau_plan  # noqa: E402


PROMPT_IDS = frozenset({
    "SSB05020005", "SSB05020077", "SSB05020082",
    "SSB15930031", "SSB15930079", "SSB15930105",
})


def _loudness_norm(x: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    act = arr[np.abs(arr) > 0.01 * max(1e-6, float(np.max(np.abs(arr))))]
    r = float(np.sqrt(np.mean(act * act))) if act.size > 0 else float(np.sqrt(np.mean(arr * arr)))
    if r > 1e-8:
        arr = arr * (target_rms / r)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 0.90:
        arr = arr * (0.90 / peak)
    return arr


def _eval_waveform(y: np.ndarray, real_x: np.ndarray, ev_real: dict, sr: int = SR) -> dict:
    ev_y = ev.evaluate_audio(y, sr=sr)
    fl_y = fl.compute_fluency_metrics(y, sr=sr)
    cl_y = compute_clarity_metrics(y, sr=sr)
    sd = ev.compute_spectral_distances(real_x, y, sr=sr)
    f0_y = ev_y["f0_hnr"]["f0_median_hz"]
    f0_r = ev_real["f0_hnr"]["f0_median_hz"]
    f0_err = abs(12.0 * math.log2(f0_y / f0_r)) if (math.isfinite(f0_y) and math.isfinite(f0_r) and f0_y > 0 and f0_r > 0) else float("nan")
    return {
        "sanity_ok": bool(ev_y["sanity"]["ok"]),
        "mel_lsd_db": float(sd["mel_lsd_db"]),
        "mcd_db": float(sd["mcd_db"]),
        "mcd_cmn_db": float(sd["mcd_cmn_db"]),
        "tilt_err_db_oct": abs(ev_y["ltas"]["tilt_db_oct"] - ev_real["ltas"]["tilt_db_oct"]),
        "f0_err_st": float(f0_err),
        "hnr_err_db": abs(ev_y["f0_hnr"]["hnr_db_median"] - ev_real["f0_hnr"]["hnr_db_median"]),
        "df1_dt": float(ev_y["formants"]["df1_dt_median_hz_s"]),
        "df2_dt": float(ev_y["formants"]["df2_dt_median_hz_s"]),
        "flux_p95_med": float(fl_y["spectral_flux"]["spectral_flux_p95_to_median"]),
        "vuv_flux_db": float(fl_y["vuv_transitions"]["vuv_flux_median_db"]),
        "f0_glitch_rate": float(fl_y["f0_continuity"]["f0_glitch_rate_gt4st"]),
        **cl_y,
    }


def _agg(rows: list[dict], keys: tuple[str, ...]) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and r[k] is not None and math.isfinite(float(r[k]))]
        out[k] = round(float(np.median(vals)), 3) if vals else None
    return out


def run_ablation(export_samples: bool = True) -> dict:
    t0 = time.time()
    idx = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))
    utts = idx["utterances"]
    heldout = [u for u in utts if u["id"] not in PROMPT_IDS]
    all_heldout_ids = {u["id"] for u in heldout}

    # Precompute single-utterance HNM features (zero cross-utterance leakage)
    precomputed: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures]] = {}
    real_eval_cache: dict[str, tuple[np.ndarray, dict, dict, dict]] = {}
    for u in utts:
        uid = u["id"]
        x, sr = vocoder.read_wav_mono(u["path"])
        x_act = x[u["active_start"] : u["active_stop"]]
        feats = vocoder.analyze(x_act, sr=SR)
        precomputed[uid] = (x_act, feats)
        ev_r = ev.evaluate_audio(x_act, sr=SR)
        fl_r = fl.compute_fluency_metrics(x_act, sr=SR)
        cl_r = compute_clarity_metrics(x_act, sr=SR)
        real_eval_cache[uid] = (x_act, ev_r, fl_r, cl_r)

    # 3-shot prompt profiles (strictly disjoint from the 26 held-out utterances)
    p502 = [precomputed[u["id"]] for u in utts if u["id"] in PROMPT_IDS and u["speaker"] == "SSB0502"]
    p1593 = [precomputed[u["id"]] for u in utts if u["id"] in PROMPT_IDS and u["speaker"] == "SSB1593"]
    prof_0502 = cloner.extract_fewshot_profile(p502, sr=SR, name="SSB0502")
    prof_1593 = cloner.extract_fewshot_profile(p1593, sr=SR, name="SSB1593")

    # Also build prompt6-only FullResBank once for minimal-corpus evaluation
    bank_prompt6 = build_fullres_bank(utts, exclude_ids=all_heldout_ids, precomputed=precomputed)

    modes = (
        "real",
        "v6_vocoder_resyn",
        "v7_baseline",
        "ablate_fullres_25ms",
        "v8_utau_hnm",
        "v8_utau_nopd",
        "v8_utau_wav_onset",
        "v8_prompt6_utau",
    )
    records: dict[str, list[dict]] = {m: [] for m in modes}
    source_totals: dict[str, dict[str, int]] = {"v8_utau_hnm": {}, "v8_prompt6_utau": {}}
    ab_examples: dict[str, dict[str, np.ndarray]] = {}

    for utt in heldout:
        uid = utt["id"]
        spk = utt["speaker"]
        prof_same = prof_0502 if spk == "SSB0502" else prof_1593
        x_act, ev_r, fl_r, cl_r = real_eval_cache[uid]

        # Real row
        records["real"].append({
            "id": uid,
            "sanity_ok": True,
            "mel_lsd_db": 0.0,
            "mcd_db": 0.0,
            "mcd_cmn_db": 0.0,
            "tilt_err_db_oct": 0.0,
            "f0_err_st": 0.0,
            "hnr_err_db": 0.0,
            "df1_dt": float(ev_r["formants"]["df1_dt_median_hz_s"]),
            "df2_dt": float(ev_r["formants"]["df2_dt_median_hz_s"]),
            "flux_p95_med": float(fl_r["spectral_flux"]["spectral_flux_p95_to_median"]),
            "vuv_flux_db": float(fl_r["vuv_transitions"]["vuv_flux_median_db"]),
            "f0_glitch_rate": float(fl_r["f0_continuity"]["f0_glitch_rate_gt4st"]),
            **cl_r,
            "rtf": 0.0,
        })

        # v6 vocoder oracle resynthesis
        y_v6 = vocoder.synthesize(precomputed[uid][1], seed=72, phase_dispersion=True)
        m_v6 = _eval_waveform(y_v6, x_act, ev_r)
        m_v6["id"] = uid
        m_v6["rtf"] = 0.02
        records["v6_vocoder_resyn"].append(m_v6)

        # Zero-leakage LOO banks (exclude_ids={uid})
        v7_bank = v7_ub.build_unit_bank(utts, exclude_ids={uid}, precomputed=precomputed)
        v8_bank = build_fullres_bank(utts, exclude_ids={uid}, precomputed=precomputed)
        plan_zh = v6_arb.plan_from_pinyin_syllables(utt["syllables"], text=utt["text"])

        # v7 baseline
        t_s = time.perf_counter()
        y_v7, _ = v7_pb.synthesize_plan(plan_zh, v7_bank, prof_same, seed=72)
        dt_v7 = time.perf_counter() - t_s
        m_v7 = _eval_waveform(y_v7, x_act, ev_r)
        m_v7["id"] = uid
        m_v7["rtf"] = dt_v7 / max(1e-6, len(y_v7) / SR)
        records["v7_baseline"].append(m_v7)

        # Ablation 1: Full-res unaveraged + Target-First, but keeping v7's 25 ms splines & uniform stretch
        t_s = time.perf_counter()
        y_f25, _ = render_utau_plan(
            plan_zh, v8_bank, prof_same, seed=72,
            phase_dispersion=True, preserve_onset=False, crisp_seams=False, use_waveform_unvoiced=False,
        )
        dt_f25 = time.perf_counter() - t_s
        m_f25 = _eval_waveform(y_f25, x_act, ev_r)
        m_f25["id"] = uid
        m_f25["rtf"] = dt_f25 / max(1e-6, len(y_f25) / SR)
        records["ablate_fullres_25ms"].append(m_f25)

        # Ablation 2: Full UTAU HNM (onset-preserving stretch + crisp obstruent seams + pd=True)
        t_s = time.perf_counter()
        y_u, meta_u = render_utau_plan(
            plan_zh, v8_bank, prof_same, seed=72,
            phase_dispersion=True, preserve_onset=True, crisp_seams=True, use_waveform_unvoiced=False,
        )
        dt_u = time.perf_counter() - t_s
        for mk, mv in meta_u["source_counts"].items():
            source_totals["v8_utau_hnm"][mk] = source_totals["v8_utau_hnm"].get(mk, 0) + mv
        m_u = _eval_waveform(y_u, x_act, ev_r)
        m_u["id"] = uid
        m_u["rtf"] = dt_u / max(1e-6, len(y_u) / SR)
        records["v8_utau_hnm"].append(m_u)

        # Ablation 3: Full UTAU HNM with phase_dispersion=False
        t_s = time.perf_counter()
        y_npd, _ = render_utau_plan(
            plan_zh, v8_bank, prof_same, seed=72,
            phase_dispersion=False, preserve_onset=True, crisp_seams=True, use_waveform_unvoiced=False,
        )
        dt_npd = time.perf_counter() - t_s
        m_npd = _eval_waveform(y_npd, x_act, ev_r)
        m_npd["id"] = uid
        m_npd["rtf"] = dt_npd / max(1e-6, len(y_npd) / SR)
        records["v8_utau_nopd"].append(m_npd)

        # Ablation 4: Full UTAU HNM + direct unvoiced waveform onset splice
        t_s = time.perf_counter()
        y_w, _ = render_utau_plan(
            plan_zh, v8_bank, prof_same, seed=72,
            phase_dispersion=True, preserve_onset=True, crisp_seams=True, use_waveform_unvoiced=True,
        )
        dt_w = time.perf_counter() - t_s
        m_w = _eval_waveform(y_w, x_act, ev_r)
        m_w["id"] = uid
        m_w["rtf"] = dt_w / max(1e-6, len(y_w) / SR)
        records["v8_utau_wav_onset"].append(m_w)

        # Minimal 6-prompt-only UTAU HNM (no heldout utterances in bank)
        t_s = time.perf_counter()
        y_p6, meta_p6 = render_utau_plan(
            plan_zh, bank_prompt6, prof_same, seed=72,
            phase_dispersion=True, preserve_onset=True, crisp_seams=True, use_waveform_unvoiced=False,
        )
        dt_p6 = time.perf_counter() - t_s
        for mk, mv in meta_p6["source_counts"].items():
            source_totals["v8_prompt6_utau"][mk] = source_totals["v8_prompt6_utau"].get(mk, 0) + mv
        m_p6 = _eval_waveform(y_p6, x_act, ev_r)
        m_p6["id"] = uid
        m_p6["rtf"] = dt_p6 / max(1e-6, len(y_p6) / SR)
        records["v8_prompt6_utau"].append(m_p6)

        if export_samples and uid in ("SSB05020122", "SSB15930193"):
            ab_examples[uid] = {
                "real": x_act,
                "v7": y_v7,
                "v8_utau_hnm": y_u,
                "v8_utau_wav_onset": y_w,
            }

    metric_keys = (
        "mel_lsd_db", "mcd_db", "mcd_cmn_db",
        "spectral_contrast_db", "spectral_std_db",
        "transient_hf_p95_db", "transient_hf_max_db", "hf_energy_ratio_2_8k_db", "unvoiced_hf_lf_db",
        "df1_dt", "df2_dt", "flux_p95_med", "vuv_flux_db", "f0_glitch_rate",
        "f0_err_st", "tilt_err_db_oct", "hnr_err_db", "rtf",
    )
    summary = {}
    for m in modes:
        summary[m] = {
            "n_utterances": len(records[m]),
            "sanity_ok": sum(1 for r in records[m] if r["sanity_ok"]),
            "medians": _agg(records[m], metric_keys),
        }

    # Export individual WAVs + loudness-matched A/B comparison packs + EN/CS samples
    exported = []
    if export_samples:
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        gap = np.zeros(int(0.45 * SR), dtype=np.float64)
        for uid, clips in ab_examples.items():
            for cname, wav in clips.items():
                p = SAMPLES_DIR / f"{uid}_{cname}.wav"
                vocoder.write_wav_mono(p, _loudness_norm(wav), sr=SR)
                exported.append(p.name)
            pack = np.concatenate([
                _loudness_norm(clips["real"]), gap,
                _loudness_norm(clips["v7"]), gap,
                _loudness_norm(clips["v8_utau_hnm"]), gap,
                _loudness_norm(clips["v8_utau_wav_onset"]),
            ])
            p_pack = SAMPLES_DIR / f"AB_{uid}_real_v7_v8hnm_v8wav.wav"
            vocoder.write_wav_mono(p_pack, pack, sr=SR)
            exported.append(p_pack.name)

        # Also export 1 English & 1 Code-Switched sample using full-res bank (with IPA synth-fill)
        full_bank = build_fullres_bank(utts, exclude_ids=(), precomputed=precomputed)
        for tag, txt in (
            ("EN_demo", "Please keep the Python audio path clean and fast."),
            ("CS_demo", "我们用 PyBotTTS 合成了一段 Python speech。"),
        ):
            cplan = cf.plan(txt, allow_oov_fallback=True)
            y_cs, _ = render_utau_plan(cplan, full_bank, prof_0502, seed=72)
            p_cs = SAMPLES_DIR / f"{tag}_v8_SSB0502.wav"
            vocoder.write_wav_mono(p_cs, _loudness_norm(y_cs), sr=SR)
            exported.append(p_cs.name)

    out = {
        "schema": 1,
        "corpus": "pybot_tts_v2_recorded/extended (26 held-out Mandarin utts, 6 prompt utts disjoint)",
        "elapsed_s": round(time.time() - t0, 2),
        "summary": summary,
        "source_counts": source_totals,
        "exported_samples": exported,
        "per_utterance": {m: records[m] for m in modes},
    }
    (EXP_DIR / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run_ablation(export_samples=True)
    print(f"Completed Phase 4 Clarity Ablation in {res['elapsed_s']}s")
    print("\n=== Clarity Root-Cause Ablation (n=26 Held-Out Mandarin, Zero Leakage) ===")
    for m, info in res["summary"].items():
        d = info["medians"]
        print(
            f"  [{m:20s}] ok={info['sanity_ok']}/{info['n_utterances']} | "
            f"Mel-LSD={d['mel_lsd_db']} dB | MCD={d['mcd_db']} [CMVN {d['mcd_cmn_db']}] | "
            f"contrast={d['spectral_contrast_db']} dB | uv_HF/LF={d['unvoiced_hf_lf_db']} dB | HF_p95={d['transient_hf_p95_db']} dB | "
            f"dF2/dt={d['df2_dt']} | flux_p95/med={d['flux_p95_med']}x | vuv={d['vuv_flux_db']} | RTF={d['rtf']}"
        )
    print("\nSource counts:", res["source_counts"])
    print("Exported samples:", res["exported_samples"])
