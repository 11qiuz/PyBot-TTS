"""CLI runner for exp_v5_eval objective acoustic evaluation and WAV sanity checks."""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

from .metrics import (
    DEFAULT_SR,
    compute_spectral_distances,
    evaluate_wav,
    read_wav,
)


def _round_nested(obj):
    if isinstance(obj, float):
        return round(obj, 4)
    if isinstance(obj, dict):
        return {k: _round_nested(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_nested(v) for v in obj]
    return obj


def _summarize_group(items: list[dict]) -> dict:
    sections = ("ltas", "f0_hnr", "formants")
    summary: dict = {"n_files": len(items), "sanity_ok": sum(1 for it in items if it["sanity"]["ok"])}
    for sec in sections:
        keys = set()
        for it in items:
            if isinstance(it.get(sec), dict):
                keys.update(it[sec].keys())
        sec_out = {}
        for k in sorted(keys):
            vals = [
                it[sec][k]
                for it in items
                if isinstance(it.get(sec), dict) and isinstance(it[sec].get(k), (int, float))
            ]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                sec_out[k] = {
                    "n": int(arr.size),
                    "median": round(float(np.median(arr)), 3),
                    "mean": round(float(np.mean(arr)), 3),
                    "std": round(float(np.std(arr)), 3),
                }
        summary[sec] = sec_out
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PyBotTTS v5 objective acoustic evaluation CLI")
    parser.add_argument("--wavs", nargs="+", help="One or more WAV paths or glob patterns to evaluate")
    parser.add_argument("--pair", nargs=2, metavar=("REF_WAV", "HYP_WAV"), help="Compare two WAV files")
    parser.add_argument("--expected-sr", type=int, default=DEFAULT_SR, help="Expected sample rate (default: 24000)")
    parser.add_argument("--fmin", type=float, default=75.0, help="Min F0 Hz for ACF search (default: 75)")
    parser.add_argument("--fmax", type=float, default=450.0, help="Max F0 Hz for ACF search (default: 450)")
    parser.add_argument("--json", dest="json_out", help="Optional path to write JSON report")
    args = parser.parse_args(argv)

    if not args.wavs and not args.pair:
        parser.error("Specify --wavs ... or --pair REF_WAV HYP_WAV")

    result: dict = {}
    if args.wavs:
        paths: list[pathlib.Path] = []
        for pat in args.wavs:
            matched = sorted(glob.glob(pat))
            if matched:
                paths.extend(pathlib.Path(m) for m in matched)
            else:
                paths.append(pathlib.Path(pat))
        per_file = [
            evaluate_wav(p, expected_sr=args.expected_sr, fmin=args.fmin, fmax=args.fmax)
            for p in paths
        ]
        result["files"] = _round_nested(per_file)
        if len(per_file) > 1:
            result["summary"] = _summarize_group(per_file)

    if args.pair:
        ref_p, hyp_p = args.pair
        ref_res = evaluate_wav(ref_p, expected_sr=args.expected_sr, fmin=args.fmin, fmax=args.fmax)
        hyp_res = evaluate_wav(hyp_p, expected_sr=args.expected_sr, fmin=args.fmin, fmax=args.fmax)
        x_ref, sr_ref = read_wav(ref_p)
        x_hyp, sr_hyp = read_wav(hyp_p)
        dist = None
        if sr_ref == sr_hyp:
            dist = compute_spectral_distances(x_ref, x_hyp, sr=sr_ref)
        result["pair"] = _round_nested({
            "ref": ref_res,
            "hyp": hyp_res,
            "spectral_distance": dist,
        })

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.json_out:
        out_p = pathlib.Path(args.json_out)
        out_p.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {out_p}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
