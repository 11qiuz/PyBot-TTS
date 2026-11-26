"""Full Phase-3 evaluation for exp_v7_pybot_tts (Hybrid PyBot Unit Bank + v6 Vocoder).

Evaluates:
1. Canonical UnitBank disk footprint & build time (saved to unit_bank_32.npz/.json).
2. Zero-leakage Held-Out Mandarin (n=26, same 26 held-out utts as exp_v6_crossling_dsp):
   (a) 26-fold LOO bank (exclude_ids={held_id}) -> upper-bound few-shot bank coverage
   (b) 6-prompt-only bank (exclude_ids=all_26_heldout_ids) -> minimal-corpus generalization
3. 26-sentence ZH/EN/CS Benchmark (78 clips = 26 x {SSB0502_3shot, SSB1593_3shot, MALE_115})
   with exp_v5_eval acoustics + exp_v6_eval fluency + CPU RTF + tracemalloc RAM.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_arch import crossling_frontend as cf
from exp_v5_eval import metrics as ev
from exp_v6_crossling_dsp import arbiter as v6_arb
from exp_v6_eval import fluency as fl
from exp_v6_vocoder import SR, cloner, vocoder
from exp_v7_pybot_tts import crossling_warp as cw, planner_bridge as pb, unit_bank as ub


def _agg_median(rows: list[dict], key_path: tuple[str, ...]) -> float | None:
    vals = []
    for r in rows:
        cur = r
        for k in key_path:
            if not isinstance(cur, dict) or cur.get(k) is None:
                cur = None
                break
            cur = cur[k]
        if cur is not None:
            vals.append(float(cur))
    return round(float(np.median(vals)), 4) if vals else None


def _eval_heldout_mode(
    mode_name: str,
    utts: list[dict],
    heldout_utts: list[dict],
    all_heldout_ids: set[str],
    precomputed: dict,
    prof_0502: cloner.FewShotProfile,
    prof_1593: cloner.FewShotProfile,
    prof_male: cloner.FewShotProfile,
    save_dir: Path | None = None,
) -> tuple[dict, list[dict]]:
    rows = []
    cross_rows = []
    src_totals: dict[str, int] = {}
    prompt_bank = None
    if mode_name == "prompt6_only":
        prompt_bank = ub.build_unit_bank(utts, exclude_ids=all_heldout_ids, precomputed=precomputed)

    for utt in heldout_utts:
        uid = utt["id"]
        spk = utt["speaker"]
        bank = (
            prompt_bank
            if prompt_bank is not None
            else ub.build_unit_bank(utts, exclude_ids={uid}, precomputed=precomputed)
        )
        prof_same = prof_0502 if spk == "SSB0502" else prof_1593
        prof_other = prof_1593 if spk == "SSB0502" else prof_0502
        plan_zh = v6_arb.plan_from_pinyin_syllables(utt["syllables"], text=utt["text"])

        t0 = time.perf_counter()
        y_same, meta_same = pb.synthesize_plan(plan_zh, bank, prof_same, seed=72)
        wall_s = time.perf_counter() - t0
        for mk, mv in meta_same["source_counts"].items():
            src_totals[mk] = src_totals.get(mk, 0) + mv

        y_other, _ = pb.synthesize_plan(plan_zh, bank, prof_other, seed=72)
        y_male, _ = pb.synthesize_plan(plan_zh, bank, prof_male, vtln_alpha=0.84, seed=72)

        if save_dir is not None:
            vocoder.write_wav_mono(save_dir / f"{uid}_v7_same.wav", y_same, sr=SR)
            if len(cross_rows) < 4:
                vocoder.write_wav_mono(save_dir / f"{uid}_v7_as_{prof_other.name}.wav", y_other, sr=SR)
                vocoder.write_wav_mono(save_dir / f"{uid}_v7_as_MALE_115.wav", y_male, sr=SR)

        real_x, real_sr = ev.read_wav(utt["path"])
        ev_real = ev.evaluate_audio(real_x, sr=real_sr)
        ev_same = ev.evaluate_audio(y_same, sr=SR)
        fl_same = fl.compute_fluency_metrics(y_same, sr=SR)
        dist_real = ev.compute_spectral_distances(real_x, y_same, sr=SR)
        dist_female = ev.compute_spectral_distances(y_same, y_other, sr=SR)
        dist_male = ev.compute_spectral_distances(y_same, y_male, sr=SR)

        rows.append(
            {
                "id": uid,
                "speaker": spk,
                "rtf": round(wall_s / max(len(y_same) / SR, 1e-6), 4),
                "sanity_ok": bool(ev_same["sanity"]["ok"]),
                "source_counts": meta_same["source_counts"],
                "dist_vs_real": dist_real,
                "tilt_err_db_oct": abs(ev_same["ltas"]["tilt_db_oct"] - ev_real["ltas"]["tilt_db_oct"]),
                "f0_err_st": abs(12.0 * np.log2(ev_same["f0_hnr"]["f0_median_hz"] / ev_real["f0_hnr"]["f0_median_hz"])),
                "hnr_err_db": abs(ev_same["f0_hnr"]["hnr_db_median"] - ev_real["f0_hnr"]["hnr_db_median"]),
                "synth": ev_same,
                "real": ev_real,
                "fluency": fl_same,
            }
        )
        cross_rows.append(
            {
                "id": uid,
                "female_pair_mel_lsd_db": dist_female["mel_lsd_db"],
                "male_pair_mel_lsd_db": dist_male["mel_lsd_db"],
            }
        )

    summary = {
        "mode": mode_name,
        "n_utterances": len(rows),
        "sanity_ok": sum(1 for r in rows if r["sanity_ok"]),
        "mean_rtf": round(float(np.mean([r["rtf"] for r in rows])), 4),
        "source_counts": src_totals,
        "mel_lsd_db_median": _agg_median(rows, ("dist_vs_real", "mel_lsd_db")),
        "mcd_db_median": _agg_median(rows, ("dist_vs_real", "mcd_db")),
        "mcd_cmn_db_median": _agg_median(rows, ("dist_vs_real", "mcd_cmn_db")),
        "tilt_err_db_oct_median": _agg_median(rows, ("tilt_err_db_oct",)),
        "f0_err_st_median": _agg_median(rows, ("f0_err_st",)),
        "hnr_err_db_median": _agg_median(rows, ("hnr_err_db",)),
        "synth_df1_dt_hz_s": _agg_median(rows, ("synth", "formants", "df1_dt_median_hz_s")),
        "synth_df2_dt_hz_s": _agg_median(rows, ("synth", "formants", "df2_dt_median_hz_s")),
        "flux_median_db": _agg_median(rows, ("fluency", "spectral_flux", "spectral_flux_median_db")),
        "flux_p95_to_median": _agg_median(rows, ("fluency", "spectral_flux", "spectral_flux_p95_to_median")),
        "vuv_flux_median_db": _agg_median(rows, ("fluency", "vuv_transitions", "vuv_flux_median_db")),
        "f0_glitch_rate_gt4st": _agg_median(rows, ("fluency", "f0_continuity", "f0_glitch_rate_gt4st")),
        "cross_speaker_separation": {
            "ssb0502_vs_ssb1593_mel_lsd_db": _agg_median(cross_rows, ("female_pair_mel_lsd_db",)),
            "ssb0502_vs_male115_mel_lsd_db": _agg_median(cross_rows, ("male_pair_mel_lsd_db",)),
        },
    }
    return summary, rows


def main() -> None:
    tracemalloc.start()
    samples_dir = HERE / "samples"
    bench_dir = samples_dir / "benchmark"
    held_dir = samples_dir / "heldout"
    bench_dir.mkdir(parents=True, exist_ok=True)
    held_dir.mkdir(parents=True, exist_ok=True)

    ext_dir = ROOT / "pybot_tts_v2_recorded" / "extended"
    ext_index = json.loads((ext_dir / "index.json").read_text(encoding="utf-8"))
    utts = ext_index["utterances"]

    wavs_0502 = sorted(glob.glob(str(ext_dir / "SSB0502*.wav")))
    wavs_1593 = sorted(glob.glob(str(ext_dir / "SSB1593*.wav")))
    prompt_0502_ids = {Path(p).stem for p in wavs_0502[:3]}
    prompt_1593_ids = {Path(p).stem for p in wavs_1593[:3]}
    prompt_ids = prompt_0502_ids | prompt_1593_ids
    heldout_utts = [u for u in utts if u["id"] not in prompt_ids]
    all_heldout_ids = {u["id"] for u in heldout_utts}

    # Precompute single-file VocoderFeatures once per utterance (zero cross-utterance info)
    precomputed = {}
    for u in utts:
        x, file_sr = vocoder.read_wav_mono(u["path"])
        x_act = x[u["active_start"] : u["active_stop"]]
        precomputed[u["id"]] = (x_act, vocoder.analyze(x_act, sr=SR))

    prof_0502 = cloner.extract_fewshot_profile(
        [vocoder.read_wav_mono(p)[0] for p in wavs_0502[:3]], sr=SR, name="SSB0502_3shot"
    )
    prof_1593 = cloner.extract_fewshot_profile(
        [vocoder.read_wav_mono(p)[0] for p in wavs_1593[:3]], sr=SR, name="SSB1593_3shot"
    )
    prof_male = cloner.make_synthetic_male_profile(
        prof_0502, f0_hz=115.0, iqr_st=2.4, tilt_db_per_oct=-4.5, vtln_alpha=0.84, sr=SR
    )
    prof_male.name = "MALE_115"
    profiles = [(prof_0502, None), (prof_1593, None), (prof_male, 0.84)]

    # 1. Build full 32-utt canonical bank & measure save size
    t_bank0 = time.perf_counter()
    full_bank = ub.build_unit_bank(utts, precomputed=precomputed)
    bank_build_s = time.perf_counter() - t_bank0
    bank_sizes = ub.save_bank(full_bank, HERE / "unit_bank_32")

    # 2. Zero-leakage Held-Out Mandarin evaluations (both 26-fold LOO and 6-prompt-only)
    loo_summary, loo_rows = _eval_heldout_mode(
        "loo_31utts",
        utts,
        heldout_utts,
        all_heldout_ids,
        precomputed,
        prof_0502,
        prof_1593,
        prof_male,
        save_dir=held_dir,
    )
    p6_summary, _ = _eval_heldout_mode(
        "prompt6_only",
        utts,
        heldout_utts,
        all_heldout_ids,
        precomputed,
        prof_0502,
        prof_1593,
        prof_male,
        save_dir=None,
    )

    # 3. 26-sentence ZH/EN/CS Benchmark x 3 speakers = 78 clips
    bench_doc = json.loads((ROOT / "exp_v5_arch" / "benchmark_texts.json").read_text(encoding="utf-8"))
    bench_rows = []
    total_wall_s = 0.0
    total_audio_s = 0.0
    bench_sources: dict[str, dict[str, int]] = {"zh": {}, "en": {}, "zh_en": {}}

    for item in bench_doc["sentences"]:
        plan = cf.plan(item["text"], style="auto", allow_oov_fallback=False)
        for prof, vtln_alpha in profiles:
            t0 = time.perf_counter()
            wav, meta = pb.synthesize_plan(plan, full_bank, prof, vtln_alpha=vtln_alpha, seed=72)
            wall_s = time.perf_counter() - t0
            dur_s = len(wav) / SR
            total_wall_s += wall_s
            total_audio_s += dur_s

            if prof.name == "SSB0502_3shot":
                for mk, mv in meta["source_counts"].items():
                    bench_sources[item["lang"]][mk] = bench_sources[item["lang"]].get(mk, 0) + mv

            out_wav = bench_dir / f"{item['id']}_{prof.name}.wav"
            vocoder.write_wav_mono(out_wav, wav, sr=SR)
            ev_res = ev.evaluate_audio(wav, sr=SR)
            fl_res = fl.compute_fluency_metrics(wav, sr=SR)

            bench_rows.append(
                {
                    "id": item["id"],
                    "lang": item["lang"],
                    "speaker": prof.name,
                    "wav": str(out_wav.relative_to(HERE)),
                    "duration_s": round(dur_s, 3),
                    "wall_ms": round(wall_s * 1000.0, 2),
                    "rtf": round(wall_s / max(dur_s, 1e-6), 4),
                    "sanity_ok": bool(ev_res["sanity"]["ok"]),
                    "source_counts": meta["source_counts"],
                    "ltas": ev_res["ltas"],
                    "f0_hnr": ev_res["f0_hnr"],
                    "formants": ev_res["formants"],
                    "fluency": fl_res,
                }
            )

    _, peak_tracemalloc_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report = {
        "bank_stats": {
            "n_utterances": len(full_bank.included_ids),
            "n_syllables": len(full_bank.syllables),
            "n_onsets": len(full_bank.onsets),
            "n_finals": len(full_bank.finals),
            "n_ipa_phones": len(full_bank.ipa_phones),
            "build_time_s": round(bank_build_s, 3),
            "total_kb": round(bank_sizes["total_bytes"] / 1024.0, 2),
            "npz_kb": round(bank_sizes["npz_bytes"] / 1024.0, 2),
            "json_kb": round(bank_sizes["json_bytes"] / 1024.0, 2),
        },
        "overhead": {
            "neural_weights_mb": 0.0,
            "benchmark_clips": len(bench_rows),
            "total_audio_s": round(total_audio_s, 2),
            "total_cpu_wall_s": round(total_wall_s, 3),
            "mean_cpu_rtf": round(total_wall_s / max(total_audio_s, 1e-6), 4),
            "peak_tracemalloc_mb": round(peak_tracemalloc_bytes / (1024 * 1024), 2),
        },
        "heldout_mandarin_loo_31utts": loo_summary,
        "heldout_mandarin_prompt6_only": p6_summary,
        "benchmark_summary": {
            "sanity_ok_count": sum(1 for r in bench_rows if r["sanity_ok"]),
            "source_counts_by_lang": bench_sources,
            "by_speaker": {
                prof.name: {
                    "tilt_db_oct": _agg_median([r for r in bench_rows if r["speaker"] == prof.name], ("ltas", "tilt_db_oct")),
                    "f0_median_hz": _agg_median([r for r in bench_rows if r["speaker"] == prof.name], ("f0_hnr", "f0_median_hz")),
                    "f0_iqr_st": _agg_median([r for r in bench_rows if r["speaker"] == prof.name], ("f0_hnr", "f0_iqr_st")),
                    "hnr_db": _agg_median([r for r in bench_rows if r["speaker"] == prof.name], ("f0_hnr", "hnr_db_median")),
                    "df2_dt_hz_s": _agg_median([r for r in bench_rows if r["speaker"] == prof.name], ("formants", "df2_dt_median_hz_s")),
                    "flux_p95_to_median": _agg_median([r for r in bench_rows if r["speaker"] == prof.name], ("fluency", "spectral_flux", "spectral_flux_p95_to_median")),
                    "vuv_flux_median_db": _agg_median([r for r in bench_rows if r["speaker"] == prof.name], ("fluency", "vuv_transitions", "vuv_flux_median_db")),
                    "f0_glitch_rate_gt4st": _agg_median([r for r in bench_rows if r["speaker"] == prof.name], ("fluency", "f0_continuity", "f0_glitch_rate_gt4st")),
                }
                for prof, _ in profiles
            },
            "by_lang": {
                lang: {
                    "sanity_ok": sum(1 for r in bench_rows if r["lang"] == lang and r["sanity_ok"]),
                    "total": sum(1 for r in bench_rows if r["lang"] == lang),
                    "mean_rtf": round(float(np.mean([r["rtf"] for r in bench_rows if r["lang"] == lang])), 4),
                    "flux_median_db": _agg_median([r for r in bench_rows if r["lang"] == lang], ("fluency", "spectral_flux", "spectral_flux_median_db")),
                    "flux_p95_to_median": _agg_median([r for r in bench_rows if r["lang"] == lang], ("fluency", "spectral_flux", "spectral_flux_p95_to_median")),
                    "vuv_flux_median_db": _agg_median([r for r in bench_rows if r["lang"] == lang], ("fluency", "vuv_transitions", "vuv_flux_median_db")),
                    "f0_glitch_rate_gt4st": _agg_median([r for r in bench_rows if r["lang"] == lang], ("fluency", "f0_continuity", "f0_glitch_rate_gt4st")),
                }
                for lang in ("zh", "en", "zh_en")
            },
        },
        "benchmark_rows": bench_rows,
        "heldout_loo_rows": loo_rows,
    }
    (HERE / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("benchmark_rows", "heldout_loo_rows")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
