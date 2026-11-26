"""CLI benchmark runner for exp_v6_eval (overhead, fluency, few-shot convergence, exp_v6 WAVs)."""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

from exp_v5_eval.metrics import compute_spectral_distances, evaluate_wav, read_wav

from .few_shot import load_extended_utterance_features, run_few_shot_convergence
from .fluency import evaluate_fluency_wav
from .overhead import profile_bytes, profile_cold_start, profile_synthesis

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _summarize_fluency(items: list[dict]) -> dict:
    out: dict = {"n_files": len(items)}
    for sec in ("f0_continuity", "spectral_flux", "vuv_transitions", "rhythm"):
        keys = set()
        for it in items:
            if isinstance(it.get(sec), dict):
                keys.update(it[sec].keys())
        sec_sum = {}
        for k in sorted(keys):
            vals = [
                it[sec][k]
                for it in items
                if isinstance(it.get(sec), dict) and isinstance(it[sec].get(k), (int, float))
            ]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                sec_sum[k] = {
                    "n": int(arr.size),
                    "median": round(float(np.median(arr)), 4),
                    "mean": round(float(np.mean(arr)), 4),
                }
        out[sec] = sec_sum
    return out


def run_part1_baseline_benchmark() -> dict:
    """Measure v4/v5 overhead, real-vs-v4-vs-v5 fluency, and extended/ few-shot convergence."""
    ext_idx = json.loads((ROOT / "pybot_tts_v2_recorded" / "extended" / "index.json").read_text(encoding="utf-8"))

    # 1. Overhead profiler: v4 vocal vs v5 synth_v5 vs v5 crossling_frontend
    from pybot_tts_v2_recorded import engine
    from pybot_tts_v4_params import speaker as sp_mod, vocal
    from exp_v5_timbre import synth_v5
    from exp_v5_arch import crossling_frontend as clf

    v4_cold = profile_cold_start(
        "from pybot_tts_v2_recorded import engine; from pybot_tts_v4_params import speaker, vocal",
        "p = vocal.normalize(speaker.load_phoneme_params())",
        cwd=ROOT,
    )
    v5_cold = profile_cold_start(
        "from exp_v5_timbre import features, synth_v5",
        "from pybot_tts_v4_params import vocal; vocal.load_params(r'pybot_tts_v4_params/phoneme_params.json')",
        cwd=ROOT,
    )
    clf_cold = profile_cold_start(
        "from exp_v5_arch import crossling_frontend",
        "crossling_frontend.plan('PyBotTTS 现在回家。', allow_oov_fallback=True)",
        cwd=ROOT,
    )

    plan_units = engine.plan("现在回家。", "auto")["units"]
    ph_raw = sp_mod.load_phoneme_params()
    v4_params = vocal.normalize(ph_raw)
    v4_synth = profile_synthesis(lambda: vocal.render(plan_units, v4_params, seed=72), sr=24000, repeats=3)

    sp_doc = json.loads((ROOT / "pybot_tts_v4_params" / "speaker_params.json").read_text(encoding="utf-8"))
    pp_502 = vocal.normalize(sp_mod.apply_speaker(sp_doc, "SSB0502", base=ph_raw))
    prof_502 = {
        "f0_median_hz": 225.0,
        "f0_iqr_st": 2.12,
        "tilt_db_per_oct": -3.0,
        "hnr_median_db": 12.9,
        "final_trajectories": {},
    }
    v5_units = synth_v5.build_units_from_utterance(ext_idx["utterances"][0], use_real_durations=True)
    v5_synth = profile_synthesis(
        lambda: synth_v5.synthesize_variant("v5_full", v5_units, v4_params, pp_502, prof_502, "SSB0502"),
        sr=24000,
        repeats=3,
    )

    overhead = {
        "v4_vocal": {
            "cold_start": v4_cold,
            "synthesis_xianzai": v4_synth,
            "speaker_table_bytes": profile_bytes(sp_doc["speakers"]["SSB0502"]),
        },
        "v5_timbre": {
            "cold_start": v5_cold,
            "synthesis_utt0": v5_synth,
            "speaker_profile_bytes": profile_bytes(prof_502),
        },
        "v5_crossling_frontend": {
            "cold_start": clf_cold,
            "lexicon_bytes": profile_bytes({"zh_chars": len(engine.CHARS), "en_lex": len(clf.EN_LEXICON)}),
        },
    }

    # 2. Fluency comparison: real extended (n=32) vs v5 samples on disk
    real_fluency = []
    for u in ext_idx["utterances"]:
        real_fluency.append(
            evaluate_fluency_wav(u["path"], start_sample=u["active_start"], stop_sample=u["active_stop"], fmin=90.0, fmax=450.0)
        )

    v5_sdir = ROOT / "exp_v5_timbre" / "samples"
    sample_fluency = {}
    for p in sorted(v5_sdir.glob("*.wav")):
        sample_fluency[p.name] = evaluate_fluency_wav(p, fmin=75.0, fmax=450.0)

    # 3. Few-shot convergence curve on extended/ (32 utts)
    by_spk = load_extended_utterance_features(ext_idx["utterances"], fmin=90.0, fmax=450.0)
    few_shot = run_few_shot_convergence(by_spk, k_values=(1, 2, 3, 8, 15), n_trials=60, seed=42)

    return {
        "overhead": overhead,
        "fluency_real_extended_32": _summarize_fluency(real_fluency),
        "fluency_v5_samples": sample_fluency,
        "few_shot_convergence": few_shot,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PyBotTTS Phase-2 overhead, fluency & few-shot benchmark")
    parser.add_argument("--part1", action="store_true", help="Run baseline overhead, fluency, and few-shot convergence")
    parser.add_argument("--wavs", nargs="+", help="Evaluate fluency + acoustics on WAV files/globs")
    parser.add_argument("--pair", nargs=2, metavar=("REF_WAV", "HYP_WAV"), help="Compare two WAVs on fluency + spectral distance")
    parser.add_argument("--json", dest="json_out", help="Optional path to write JSON report")
    args = parser.parse_args(argv)

    if not (args.part1 or args.wavs or args.pair):
        parser.error("Specify --part1, --wavs ..., or --pair REF HYP")

    out: dict = {}
    if args.part1:
        out["part1"] = run_part1_baseline_benchmark()
    if args.wavs:
        paths: list[pathlib.Path] = []
        for pat in args.wavs:
            matched = sorted(glob.glob(pat))
            paths.extend(pathlib.Path(m) for m in matched) if matched else paths.append(pathlib.Path(pat))
        fl_items = [evaluate_fluency_wav(p) for p in paths]
        ac_items = [evaluate_wav(p) for p in paths]
        out["wavs_fluency"] = fl_items
        out["wavs_fluency_summary"] = _summarize_fluency(fl_items)
        out["wavs_acoustics"] = ac_items
    if args.pair:
        ref_p, hyp_p = args.pair
        x_r, sr_r = read_wav(ref_p)
        x_h, sr_h = read_wav(hyp_p)
        out["pair"] = {
            "ref_fluency": evaluate_fluency_wav(ref_p),
            "hyp_fluency": evaluate_fluency_wav(hyp_p),
            "ref_acoustics": evaluate_wav(ref_p),
            "hyp_acoustics": evaluate_wav(hyp_p),
            "spectral_distance": compute_spectral_distances(x_r, x_h, sr=sr_r) if sr_r == sr_h else None,
        }

    payload = json.dumps(out, indent=2, ensure_ascii=False)
    if args.json_out:
        p = pathlib.Path(args.json_out)
        p.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {p}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
