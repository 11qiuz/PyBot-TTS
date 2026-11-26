"""Full Phase-2 benchmark runner for exp_v6_crossling_dsp.

Synthesizes:
1. All 26 frozen benchmark_texts.json sentences (10 ZH, 10 EN, 6 ZH+EN) across 3 speakers:
   - SSB0502 (3-utterance ~7 s few-shot profile from extended/)
   - SSB1593 (3-utterance ~8 s few-shot profile from extended/)
   - MALE_115 (synthetic male baritone, F0=115 Hz, vtln_scale=0.84)
2. All 32 extended/ Mandarin sentences in strict disjoint held-out mode
   (synthesized with the 3-shot prompt profile of the target speaker, evaluated
   on the remaining 13 held-out utterances against real donor WAVs via exp_v5_eval).
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
from exp_v6_crossling_dsp import SR, arbiter, hnm_synth, speaker_profile as sp


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


def main() -> None:
    tracemalloc.start()
    samples_dir = HERE / "samples"
    bench_dir = samples_dir / "benchmark"
    matched_dir = samples_dir / "matched_heldout"
    bench_dir.mkdir(parents=True, exist_ok=True)
    matched_dir.mkdir(parents=True, exist_ok=True)

    ext_dir = ROOT / "pybot_tts_v2_recorded" / "extended"
    wavs_0502 = sorted(glob.glob(str(ext_dir / "SSB0502*.wav")))
    wavs_1593 = sorted(glob.glob(str(ext_dir / "SSB1593*.wav")))
    prompt_0502_ids = {Path(p).stem for p in wavs_0502[:3]}
    prompt_1593_ids = {Path(p).stem for p in wavs_1593[:3]}

    prof_0502 = sp.extract_profile_from_wavs(wavs_0502[:3], name="SSB0502_3shot")
    prof_1593 = sp.extract_profile_from_wavs(wavs_1593[:3], name="SSB1593_3shot")
    prof_male = sp.male_baritone_115()
    profiles = [prof_0502, prof_1593, prof_male]

    # 1. Synthesize all 26 benchmark sentences x 3 speakers = 78 WAVs
    bench_doc = json.loads((ROOT / "exp_v5_arch" / "benchmark_texts.json").read_text(encoding="utf-8"))
    bench_rows = []
    total_wall_s = 0.0
    total_audio_s = 0.0

    for item in bench_doc["sentences"]:
        plan = cf.plan(item["text"], style="auto", allow_oov_fallback=False)
        for prof in profiles:
            t0 = time.perf_counter()
            tr = arbiter.build_frame_track(plan, prof)
            y = hnm_synth.render_track(tr, prof, seed=72)
            wall_s = time.perf_counter() - t0
            dur_s = len(y) / SR
            total_wall_s += wall_s
            total_audio_s += dur_s

            out_wav = bench_dir / f"{item['id']}_{prof.name}.wav"
            hnm_synth.write_wav(out_wav, y, sr=SR)
            eval_res = ev.evaluate_audio(y, sr=SR)

            # Check intra-voiced F0 continuity (glitch = frame step > 4 st)
            voiced = tr.f0_hz > 0.0
            glitches = 0
            for k in range(len(tr.f0_hz) - 1):
                if voiced[k] and voiced[k + 1]:
                    if abs(12.0 * np.log2(tr.f0_hz[k + 1] / tr.f0_hz[k])) > 4.0:
                        glitches += 1

            bench_rows.append(
                {
                    "id": item["id"],
                    "lang": item["lang"],
                    "speaker": prof.name,
                    "wav": str(out_wav.relative_to(HERE)),
                    "duration_s": round(dur_s, 3),
                    "wall_ms": round(wall_s * 1000.0, 2),
                    "rtf": round(wall_s / max(dur_s, 1e-6), 4),
                    "f0_glitches_gt4st": glitches,
                    "sanity_ok": bool(eval_res["sanity"]["ok"]),
                    "ltas": eval_res["ltas"],
                    "f0_hnr": eval_res["f0_hnr"],
                    "formants": eval_res["formants"],
                }
            )

    # 2. Evaluate held-out Mandarin utterances (26 held-out = 13 per speaker) vs real WAVs
    ext_index = json.loads((ext_dir / "index.json").read_text(encoding="utf-8"))
    matched_rows = []
    cross_spk_rows = []
    for utt in ext_index["utterances"]:
        uid = utt["id"]
        spk = utt["speaker"]
        if uid in prompt_0502_ids or uid in prompt_1593_ids:
            continue
        prof_same = prof_0502 if spk == "SSB0502" else prof_1593
        prof_other = prof_1593 if spk == "SSB0502" else prof_0502
        plan_zh = arbiter.plan_from_pinyin_syllables(utt["syllables"], text=utt["text"])

        y_same = hnm_synth.render_track(arbiter.build_frame_track(plan_zh, prof_same), prof_same, seed=72)
        y_other = hnm_synth.render_track(arbiter.build_frame_track(plan_zh, prof_other), prof_other, seed=72)
        y_male = hnm_synth.render_track(arbiter.build_frame_track(plan_zh, prof_male), prof_male, seed=72)

        out_same = matched_dir / f"{uid}_v6_same.wav"
        hnm_synth.write_wav(out_same, y_same, sr=SR)
        if len(cross_spk_rows) < 4:
            hnm_synth.write_wav(matched_dir / f"{uid}_v6_as_{prof_other.name}.wav", y_other, sr=SR)
            hnm_synth.write_wav(matched_dir / f"{uid}_v6_as_MALE_115.wav", y_male, sr=SR)

        real_x, real_sr = ev.read_wav(utt["path"])
        ev_real = ev.evaluate_audio(real_x, sr=real_sr)
        ev_same = ev.evaluate_audio(y_same, sr=SR)
        dist_same = ev.compute_spectral_distances(real_x, y_same, sr=SR)
        dist_spk_pair = ev.compute_spectral_distances(y_same, y_other, sr=SR)
        dist_male_pair = ev.compute_spectral_distances(y_same, y_male, sr=SR)

        matched_rows.append(
            {
                "id": uid,
                "speaker": spk,
                "real": ev_real,
                "synth": ev_same,
                "dist_vs_real": dist_same,
                "tilt_err_db_oct": abs(ev_same["ltas"]["tilt_db_oct"] - ev_real["ltas"]["tilt_db_oct"]),
                "f0_err_st": abs(12.0 * np.log2(ev_same["f0_hnr"]["f0_median_hz"] / ev_real["f0_hnr"]["f0_median_hz"])),
                "hnr_err_db": abs(ev_same["f0_hnr"]["hnr_db_median"] - ev_real["f0_hnr"]["hnr_db_median"]),
            }
        )
        cross_spk_rows.append(
            {
                "id": uid,
                "female_pair_mel_lsd_db": dist_spk_pair["mel_lsd_db"],
                "female_pair_wave_max": round(float(np.max(np.abs(y_same[: min(len(y_same), len(y_other))] - y_other[: min(len(y_same), len(y_other))]))), 4),
                "male_pair_mel_lsd_db": dist_male_pair["mel_lsd_db"],
            }
        )

    _, peak_tracemalloc_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary = {
        "profiles": {p.name: p.to_dict() for p in profiles},
        "profile_json_bytes": {p.name: len(p.to_json().encode("utf-8")) for p in profiles},
        "overhead": {
            "benchmark_clips": len(bench_rows),
            "total_audio_s": round(total_audio_s, 2),
            "total_cpu_wall_s": round(total_wall_s, 3),
            "mean_cpu_rtf": round(total_wall_s / max(total_audio_s, 1e-6), 4),
            "peak_tracemalloc_mb": round(peak_tracemalloc_bytes / (1024 * 1024), 2),
            "neural_weights_mb": 0.0,
        },
        "benchmark_summary": {
            "sanity_ok_count": sum(1 for r in bench_rows if r["sanity_ok"]),
            "total_f0_glitches_gt4st": sum(r["f0_glitches_gt4st"] for r in bench_rows),
            "by_speaker": {
                p.name: {
                    "tilt_db_oct": _agg_median([r for r in bench_rows if r["speaker"] == p.name], ("ltas", "tilt_db_oct")),
                    "hi_lo_1_8k_db": _agg_median([r for r in bench_rows if r["speaker"] == p.name], ("ltas", "hi_lo_db_1_8k")),
                    "vhi_lo_4_8k_db": _agg_median([r for r in bench_rows if r["speaker"] == p.name], ("ltas", "vhi_lo_db_4_8k")),
                    "f0_median_hz": _agg_median([r for r in bench_rows if r["speaker"] == p.name], ("f0_hnr", "f0_median_hz")),
                    "f0_iqr_st": _agg_median([r for r in bench_rows if r["speaker"] == p.name], ("f0_hnr", "f0_iqr_st")),
                    "hnr_db": _agg_median([r for r in bench_rows if r["speaker"] == p.name], ("f0_hnr", "hnr_db_median")),
                    "df2_dt_hz_s": _agg_median([r for r in bench_rows if r["speaker"] == p.name], ("formants", "df2_dt_median_hz_s")),
                }
                for p in profiles
            },
            "by_lang": {
                lang: {
                    "sanity_ok": sum(1 for r in bench_rows if r["lang"] == lang and r["sanity_ok"]),
                    "total": sum(1 for r in bench_rows if r["lang"] == lang),
                    "mean_rtf": round(float(np.mean([r["rtf"] for r in bench_rows if r["lang"] == lang])), 4),
                }
                for lang in ("zh", "en", "zh_en")
            },
        },
        "heldout_mandarin_vs_real": {
            "n_heldout_utterances": len(matched_rows),
            "mel_lsd_db_median": _agg_median(matched_rows, ("dist_vs_real", "mel_lsd_db")),
            "mcd_db_median": _agg_median(matched_rows, ("dist_vs_real", "mcd_db")),
            "mcd_cmn_db_median": _agg_median(matched_rows, ("dist_vs_real", "mcd_cmn_db")),
            "tilt_err_db_oct_median": _agg_median(matched_rows, ("tilt_err_db_oct",)),
            "f0_err_st_median": _agg_median(matched_rows, ("f0_err_st",)),
            "hnr_err_db_median": _agg_median(matched_rows, ("hnr_err_db",)),
            "synth_df1_dt_hz_s": _agg_median(matched_rows, ("synth", "formants", "df1_dt_median_hz_s")),
            "synth_df2_dt_hz_s": _agg_median(matched_rows, ("synth", "formants", "df2_dt_median_hz_s")),
            "real_df1_dt_hz_s": _agg_median(matched_rows, ("real", "formants", "df1_dt_median_hz_s")),
            "real_df2_dt_hz_s": _agg_median(matched_rows, ("real", "formants", "df2_dt_median_hz_s")),
            "cross_speaker_separation": {
                "ssb0502_vs_ssb1593_mel_lsd_db": _agg_median(cross_spk_rows, ("female_pair_mel_lsd_db",)),
                "ssb0502_vs_male115_mel_lsd_db": _agg_median(cross_spk_rows, ("male_pair_mel_lsd_db",)),
            },
        },
        "benchmark_rows": bench_rows,
        "matched_rows": matched_rows,
    }

    (HERE / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("benchmark_rows", "matched_rows")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
