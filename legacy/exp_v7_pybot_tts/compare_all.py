"""Cross-generation comparison (v4 -> v5 -> v6 -> v7 vs Real) and loudness-matched A/B listening packs."""
from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import sys
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity, compute_spectral_distances, evaluate_wav, read_wav  # noqa: E402
from exp_v6_eval.fluency import evaluate_fluency_wav  # noqa: E402
from exp_v6_eval.overhead import profile_bytes, profile_cold_start, profile_synthesis  # noqa: E402
from exp_v6_vocoder import cloner  # noqa: E402
from exp_v7_pybot_tts import planner_bridge, unit_bank  # noqa: E402

SR = 24000
LISTENING_DIR = pathlib.Path(__file__).resolve().parent / "listening_ab"


def _rms_normalize(x: np.ndarray, target_dbfs: float = -20.0, max_peak: float = 0.89) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    rms = float(np.sqrt(np.mean(arr * arr)))
    if rms <= 1e-12:
        return arr.copy()
    target_rms = 10.0 ** (target_dbfs / 20.0)
    scaled = arr * (target_rms / rms)
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


def _concat_ab(segments: list[tuple[str, pathlib.Path]], gap_s: float = 0.35) -> tuple[np.ndarray, list[dict]]:
    gap = np.zeros(int(round(gap_s * SR)), dtype=np.float64)
    chunks: list[np.ndarray] = []
    timeline: list[dict] = []
    cursor_s = 0.0
    for idx, (label, wav_path) in enumerate(segments):
        x, sr = read_wav(wav_path)
        if sr != SR:
            raise ValueError(f"Expected {SR} Hz in {wav_path}, got {sr}")
        x_norm = _rms_normalize(x)
        dur_s = len(x_norm) / float(SR)
        timeline.append({
            "label": label,
            "source": str(wav_path.relative_to(ROOT)).replace("\\", "/"),
            "start_s": round(cursor_s, 3),
            "stop_s": round(cursor_s + dur_s, 3),
        })
        chunks.append(x_norm)
        cursor_s += dur_s
        if idx + 1 < len(segments):
            chunks.append(gap)
            cursor_s += gap_s
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64), timeline


def profile_v7_overhead() -> dict:
    """Measure fresh-subprocess cold-start, e2e RTF per language, and artifact sizes for v7."""
    cold = profile_cold_start(
        "from exp_v7_pybot_tts import unit_bank, planner_bridge",
        "b = unit_bank.load_bank(r'exp_v7_pybot_tts/unit_bank_32.npz')",
        cwd=ROOT,
    )
    bank = unit_bank.load_bank(ROOT / "exp_v7_pybot_tts" / "unit_bank_32.npz")
    ext_idx = json.loads((ROOT / "pybot_tts_v2_recorded" / "extended" / "index.json").read_text(encoding="utf-8"))
    wavs_0502 = [read_wav(u["path"])[0][u["active_start"]:u["active_stop"]] for u in ext_idx["utterances"] if u["speaker"] == "SSB0502"][:3]
    prof_0502 = cloner.extract_fewshot_profile(wavs_0502, sr=SR, name="SSB0502_3shot")

    bm = json.loads((ROOT / "exp_v5_arch" / "benchmark_texts.json").read_text(encoding="utf-8"))
    items = bm["sentences"] if isinstance(bm, dict) else bm
    sample_by_lang = {}
    for it in items:
        lang = it.get("lang") or it.get("category")
        if lang not in sample_by_lang:
            sample_by_lang[lang] = it["text"]

    e2e = {}
    for lang, text in sample_by_lang.items():
        def _run(t=text):
            plan = cf.plan(t, allow_oov_fallback=True)
            wav, _ = planner_bridge.synthesize_plan(plan, bank, prof_0502, seed=72)
            return wav
        e2e[lang] = profile_synthesis(_run, sr=SR, repeats=3)

    npz_bytes = (ROOT / "exp_v7_pybot_tts" / "unit_bank_32.npz").stat().st_size
    json_bytes = (ROOT / "exp_v7_pybot_tts" / "unit_bank_32.json").stat().st_size
    return {
        "cold_start": cold,
        "e2e_by_lang": e2e,
        "bank_disk_bytes": {"npz": npz_bytes, "json": json_bytes, "total": npz_bytes + json_bytes},
        "profile_bytes": profile_bytes(dataclasses.asdict(prof_0502)),
    }


def evaluate_v7_independent() -> dict:
    """Independently run exp_v5_eval + exp_v6_eval across v7 samples/benchmark and samples/heldout."""
    bdir = ROOT / "exp_v7_pybot_tts" / "samples" / "benchmark"
    hdir = ROOT / "exp_v7_pybot_tts" / "samples" / "heldout"
    ext_idx = json.loads((ROOT / "pybot_tts_v2_recorded" / "extended" / "index.json").read_text(encoding="utf-8"))
    utt_map = {u["id"]: u for u in ext_idx["utterances"]}

    bench_ac, bench_fl = [], []
    for p in sorted(bdir.glob("*.wav")):
        fmin = 75.0 if "MALE_115" in p.name else 90.0
        bench_ac.append(evaluate_wav(p, fmin=fmin, fmax=450.0))
        bench_fl.append(evaluate_fluency_wav(p, fmin=fmin, fmax=450.0))

    held_same_ac, held_same_fl, held_same_dists = [], [], []
    for p in sorted(hdir.glob("*_v7_same.wav")):
        uid = p.name.split("_")[0]
        ac = evaluate_wav(p, fmin=90.0, fmax=450.0)
        fl = evaluate_fluency_wav(p, fmin=90.0, fmax=450.0)
        u = utt_map[uid]
        xr, _ = read_wav(u["path"])
        xr = xr[u["active_start"]:u["active_stop"]]
        xh, _ = read_wav(p)
        dist = compute_spectral_distances(xr, xh, sr=SR)
        held_same_ac.append(ac)
        held_same_fl.append(fl)
        if dist:
            held_same_dists.append(dist)

    def _med(items, sec, key):
        vals = [it[sec][key] for it in items if isinstance(it.get(sec), dict) and isinstance(it[sec].get(key), (int, float))]
        return round(float(np.median(vals)), 4) if vals else None

    def _mean(items, sec, key):
        vals = [it[sec][key] for it in items if isinstance(it.get(sec), dict) and isinstance(it[sec].get(key), (int, float))]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "benchmark_78": {
            "n": len(bench_ac),
            "sanity_ok": sum(1 for it in bench_ac if it["sanity"]["ok"]),
            "df1_dt_med": _med(bench_ac, "formants", "df1_dt_median_hz_s"),
            "df2_dt_med": _med(bench_ac, "formants", "df2_dt_median_hz_s"),
            "flux_med": _med(bench_fl, "spectral_flux", "spectral_flux_median_db"),
            "flux_p95_med": _med(bench_fl, "spectral_flux", "spectral_flux_p95_to_median"),
            "vuv_flux_med": _med(bench_fl, "vuv_transitions", "vuv_flux_median_db"),
            "f0_glitch_rate_mean": _mean(bench_fl, "f0_continuity", "f0_glitch_rate_gt4st"),
        },
        "heldout_same_26": {
            "n": len(held_same_ac),
            "sanity_ok": sum(1 for it in held_same_ac if it["sanity"]["ok"]),
            "mel_lsd_med": round(float(np.median([d["mel_lsd_db"] for d in held_same_dists])), 4),
            "mcd_med": round(float(np.median([d["mcd_db"] for d in held_same_dists])), 4),
            "mcd_cmn_med": round(float(np.median([d["mcd_cmn_db"] for d in held_same_dists])), 4),
            "df1_dt_med": _med(held_same_ac, "formants", "df1_dt_median_hz_s"),
            "df2_dt_med": _med(held_same_ac, "formants", "df2_dt_median_hz_s"),
            "flux_med": _med(held_same_fl, "spectral_flux", "spectral_flux_median_db"),
            "flux_p95_med": _med(held_same_fl, "spectral_flux", "spectral_flux_p95_to_median"),
            "vuv_flux_med": _med(held_same_fl, "vuv_transitions", "vuv_flux_median_db"),
            "f0_glitch_rate_mean": _mean(held_same_fl, "f0_continuity", "f0_glitch_rate_gt4st"),
        },
    }


def build_listening_ab() -> list[dict]:
    """Export loudness-matched A/B/C/D WAV comparisons in exp_v7_pybot_tts/listening_ab/."""
    LISTENING_DIR.mkdir(parents=True, exist_ok=True)
    ext_dir = ROOT / "pybot_tts_v2_recorded" / "extended"
    v5_dir = ROOT / "exp_v5_timbre" / "samples"
    v6_voc_dir = ROOT / "exp_v6_vocoder" / "samples"
    v6_dsp_b = ROOT / "exp_v6_crossling_dsp" / "samples" / "benchmark"
    v6_dsp_h = ROOT / "exp_v6_crossling_dsp" / "samples" / "matched_heldout"
    v7_b = ROOT / "exp_v7_pybot_tts" / "samples" / "benchmark"
    v7_h = ROOT / "exp_v7_pybot_tts" / "samples" / "heldout"

    packs = [
        (
            "01_zh_evolution_SSB05020122.wav",
            "Mandarin held-out SSB05020122: Real -> v6_crossling_dsp -> v7_pybot_tts (LOO)",
            [
                ("real", ext_dir / "SSB05020122.wav"),
                ("v6_crossling_dsp", v6_dsp_h / "SSB05020122_v6_same.wav"),
                ("v7_pybot_loo", v7_h / "SSB05020122_v7_same.wav"),
            ],
        ),
        (
            "02_zh_evolution_SSB15930193.wav",
            "Mandarin held-out SSB15930193: Real -> v6_crossling_dsp -> v7_pybot_tts (LOO)",
            [
                ("real", ext_dir / "SSB15930193.wav"),
                ("v6_crossling_dsp", v6_dsp_h / "SSB15930193_v6_same.wav"),
                ("v7_pybot_loo", v7_h / "SSB15930193_v7_same.wav"),
            ],
        ),
        (
            "03_zh_speaker_swap_SSB05020122.wav",
            "Zero-shot cloning on SSB05020122: Real_0502 -> v7_same_0502 -> v7_as_1593 -> v7_as_MALE115 -> Real_1593_ref",
            [
                ("real_SSB0502", ext_dir / "SSB05020122.wav"),
                ("v7_SSB0502", v7_h / "SSB05020122_v7_same.wav"),
                ("v7_as_SSB1593_3shot", v7_h / "SSB05020122_v7_as_SSB1593_3shot.wav"),
                ("v7_as_MALE115", v7_h / "SSB05020122_v7_as_MALE_115.wav"),
                ("real_SSB1593_ref", ext_dir / "SSB15930031.wav"),
            ],
        ),
        (
            "04_en_dsp_vs_v7_EN01.wav",
            "English EN_01: v6_dsp_0502 -> v7_0502 -> v6_dsp_MALE115 -> v7_MALE115",
            [
                ("v6_dsp_SSB0502", v6_dsp_b / "EN_01_SSB0502_3shot.wav"),
                ("v7_SSB0502", v7_b / "EN_01_SSB0502_3shot.wav"),
                ("v6_dsp_MALE115", v6_dsp_b / "EN_01_MALE_115.wav"),
                ("v7_MALE115", v7_b / "EN_01_MALE_115.wav"),
            ],
        ),
        (
            "05_cs_dsp_vs_v7_CS01.wav",
            "Code-switched ZH+EN CS_01: v6_dsp_0502 -> v7_0502 -> v6_dsp_1593 -> v7_1593",
            [
                ("v6_dsp_SSB0502", v6_dsp_b / "CS_01_SSB0502_3shot.wav"),
                ("v7_SSB0502", v7_b / "CS_01_SSB0502_3shot.wav"),
                ("v6_dsp_SSB1593", v6_dsp_b / "CS_01_SSB1593_3shot.wav"),
                ("v7_SSB1593", v7_b / "CS_01_SSB1593_3shot.wav"),
            ],
        ),
        (
            "06_vocoder_vs_v5_SSB05020005.wav",
            "SSB05020005 trajectory vs vocoder: Real -> v4_base -> v5_full -> v6_vocoder_resyn -> v6_vocoder_to1593",
            [
                ("real", v5_dir / "SSB05020005_real.wav"),
                ("v4_baseline", v5_dir / "SSB05020005_v4_baseline.wav"),
                ("v5_full", v5_dir / "SSB05020005_v5_full.wav"),
                ("v6_vocoder_resyn", v6_voc_dir / "SSB05020005_gate1_resyn.wav"),
                ("v6_vocoder_to1593", v6_voc_dir / "SSB05020005_to_SSB1593_3shot.wav"),
            ],
        ),
        (
            "07_vocoder_vs_v5_SSB15930031.wav",
            "SSB15930031 trajectory vs vocoder: Real -> v4_base -> v5_full -> v6_vocoder_resyn -> v6_vocoder_to0502",
            [
                ("real", v5_dir / "SSB15930031_real.wav"),
                ("v4_baseline", v5_dir / "SSB15930031_v4_baseline.wav"),
                ("v5_full", v5_dir / "SSB15930031_v5_full.wav"),
                ("v6_vocoder_resyn", v6_voc_dir / "SSB15930031_gate1_resyn.wav"),
                ("v6_vocoder_to0502", v6_voc_dir / "SSB15930031_to_SSB0502_3shot.wav"),
            ],
        ),
    ]

    manifest = []
    for fname, desc, segs in packs:
        wav, timeline = _concat_ab(segs)
        out_p = LISTENING_DIR / fname
        _write_pcm16_wav(out_p, wav, SR)
        sanity = check_wav_sanity(wav, SR)
        manifest.append({
            "file": fname,
            "description": desc,
            "duration_s": sanity["duration_s"],
            "sanity_ok": sanity["ok"],
            "rms_dbfs": sanity["rms_dbfs"],
            "peak": sanity["peak"],
            "segments": timeline,
        })
    return manifest


def main() -> int:
    v7_overhead = profile_v7_overhead()
    v7_eval = evaluate_v7_independent()
    manifest = build_listening_ab()

    # Load prior verified baselines for unified matrix
    v5_res = json.loads((ROOT / "exp_v5_timbre" / "results.json").read_text(encoding="utf-8"))
    v6_base = json.loads((ROOT / "exp_v6_eval" / "baseline_report.json").read_text(encoding="utf-8"))["part1"]
    v6_voc = json.loads((ROOT / "exp_v6_vocoder" / "results.json").read_text(encoding="utf-8"))
    v6_dsp = json.loads((ROOT / "exp_v6_eval" / "crossling_eval_report.json").read_text(encoding="utf-8"))
    v7_res = json.loads((ROOT / "exp_v7_pybot_tts" / "results.json").read_text(encoding="utf-8"))

    matrix = {
        "real_extended_32": {
            "neural_weights_mb": 0.0,
            "cpu_rtf": None,
            "mel_lsd_db": 0.0,
            "mcd_db": 0.0,
            "df2_dt_hz_s": 4910.0,
            "flux_p95_med": v6_base["fluency_real_extended_32"]["spectral_flux"]["spectral_flux_p95_to_median"]["median"],
            "vuv_flux_db": v6_base["fluency_real_extended_32"]["vuv_transitions"]["vuv_flux_median_db"]["median"],
            "cross_lingual": "ZH only",
        },
        "v4_baseline": {
            "neural_weights_mb": 0.0,
            "cpu_rtf": v6_base["overhead"]["v4_vocal"]["synthesis_xianzai"]["cpu_rtf_median"],
            "mel_lsd_db": v5_res["variants"]["v4_baseline"]["acoustic_gap_vs_real"]["dtw_mel_lsd_db"],
            "mcd_db": v5_res["variants"]["v4_baseline"]["acoustic_gap_vs_real"]["dtw_mcd_db"],
            "df2_dt_hz_s": 869.0,
            "flux_p95_med": 4.30,
            "vuv_flux_db": 9.38,
            "cross_lingual": "ZH only (closed lexicon)",
        },
        "v5_full": {
            "neural_weights_mb": 0.0,
            "cpu_rtf": v6_base["overhead"]["v5_timbre"]["synthesis_utt0"]["cpu_rtf_median"],
            "mel_lsd_db": v5_res["variants"]["v5_full"]["acoustic_gap_vs_real"]["dtw_mel_lsd_db"],
            "mcd_db": v5_res["variants"]["v5_full"]["acoustic_gap_vs_real"]["dtw_mcd_db"],
            "df2_dt_hz_s": 2879.0,
            "flux_p95_med": 3.30,
            "vuv_flux_db": 8.52,
            "cross_lingual": "ZH only",
        },
        "v6_vocoder_gate1_resyn": {
            "neural_weights_mb": 0.0,
            "cpu_rtf": v6_voc["gate1_resynthesis"]["pd_on"]["rtf_syn"],
            "mel_lsd_db": v6_voc["gate1_resynthesis"]["pd_on"]["mel_lsd_db_mean"],
            "mcd_db": v6_voc["gate1_resynthesis"]["pd_on"]["mcd_db_mean"],
            "df2_dt_hz_s": 5155.0,
            "flux_p95_med": 1.57,
            "vuv_flux_db": 4.00,
            "cross_lingual": "Donor conversion only (no TTS)",
        },
        "v6_crossling_dsp": {
            "neural_weights_mb": 0.0,
            "cpu_rtf": v6_dsp["overhead"]["e2e_by_lang"]["e2e_zh"]["cpu_rtf_median"],
            "mel_lsd_db": v6_dsp["matched_heldout"]["same_speaker_26"]["mel_lsd_med"],
            "mcd_db": v6_dsp["matched_heldout"]["same_speaker_26"]["mcd_med"],
            "df2_dt_hz_s": v6_dsp["matched_heldout"]["same_speaker_26"]["df2_dt_med"],
            "flux_p95_med": v6_dsp["matched_heldout"]["same_speaker_26"]["flux_p95_med_ratio"],
            "vuv_flux_db": v6_dsp["matched_heldout"]["same_speaker_26"]["vuv_flux_med"],
            "cross_lingual": "ZH + EN + CS (rule-only)",
        },
        "v7_pybot_tts_loo": {
            "neural_weights_mb": 0.0,
            "cpu_rtf": v7_res["overhead"]["mean_cpu_rtf"],
            "mel_lsd_db": v7_eval["heldout_same_26"]["mel_lsd_med"],
            "mcd_db": v7_eval["heldout_same_26"]["mcd_med"],
            "df2_dt_hz_s": v7_eval["heldout_same_26"]["df2_dt_med"],
            "flux_p95_med": v7_eval["heldout_same_26"]["flux_p95_med"],
            "vuv_flux_db": v7_eval["heldout_same_26"]["vuv_flux_med"],
            "cross_lingual": "ZH + EN + CS (hybrid unit-bank + vocoder)",
        },
    }

    summary = {
        "matrix": matrix,
        "v7_overhead_independent": v7_overhead,
        "v7_eval_independent": v7_eval,
        "listening_ab_manifest": manifest,
    }
    (LISTENING_DIR / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("WROTE", LISTENING_DIR / "comparison_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
