"""Apples-to-apples 40 ms / 20 ms hop loudness jump audit on exp_v10_integrity/samples/*.wav."""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_eval.metrics import read_wav  # noqa: E402
from exp_v10_audit.loudness_audit import (  # noqa: E402
    SR,
    _aggregate_loudness_rows,
    _extract_pair_deltas,
    measure_windowed_loudness,
)

AUDIT_DIR = pathlib.Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "exp_v10_integrity" / "samples"
OUT_PATH = AUDIT_DIR / "v10_loudness_comparison.json"

SAMPLE_LABELS: dict[str, tuple[str, list[str]]] = {
    "AB_EN_bdl_0030_real_v9_v10.wav": ("EN", ["real", "v9", "v10"]),
    "AB_EN_slt_0030_real_v9_v10.wav": ("EN", ["real", "v9", "v10"]),
    "AB_EN_bdl_0030_real_vs_v10full.wav": ("EN", ["real", "v10full"]),
    "AB_EN_slt_0030_real_vs_v10full.wav": ("EN", ["real", "v10full"]),
    "AB_ZH_SSB05020122_real_v9_v10.wav": ("ZH", ["real", "v9", "v10"]),
    "AB_ZH_SSB15930193_real_v9_v10.wav": ("ZH", ["real", "v9", "v10"]),
    "AB_CS_SSB0502_v9_vs_v10.wav": ("CS", ["v9", "v10"]),
    "AB_CS_SSB0073_M133_v9_vs_v10.wav": ("CS", ["v9", "v10"]),
    "CS01_SSB0502_v10full.wav": ("CS", ["v10full"]),
    "CS01_SSB0073_M133_v10full.wav": ("CS", ["v10full"]),
}


def split_on_digital_zero_gaps(
    x: np.ndarray,
    sr: int = SR,
    min_gap_s: float = 0.25,
    zero_eps: float = 1e-6,
) -> list[tuple[int, int]]:
    """Return (start_sample, stop_sample) spans separated by contiguous ~zero gaps >= min_gap_s."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return []
    min_len = max(1, int(round(min_gap_s * sr)))
    is_z = np.abs(arr) <= zero_eps
    # Find runs of True in is_z via padded diff
    padded = np.concatenate([[False], is_z, [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    stops = np.where(diffs == -1)[0]

    gaps = [(int(s), int(e)) for s, e in zip(starts, stops) if (e - s) >= min_len and s > 0 and e < len(arr)]
    spans: list[tuple[int, int]] = []
    cur = 0
    for gs, ge in gaps:
        if gs > cur:
            spans.append((cur, gs))
        cur = ge
    if cur < len(arr):
        spans.append((cur, len(arr)))
    return spans


def audit_v10_samples(samples_dir: pathlib.Path = SAMPLES_DIR) -> dict[str, Any]:
    per_file: dict[str, Any] = {}
    groups: dict[str, tuple[list[dict], list[np.ndarray]]] = {}

    for fname, (lang, labels) in SAMPLE_LABELS.items():
        fpath = samples_dir / fname
        if not fpath.exists():
            continue
        x, sr = read_wav(fpath)
        if len(labels) == 1:
            spans = [(0, len(x))]
        else:
            spans = split_on_digital_zero_gaps(x, sr=sr, min_gap_s=0.30, zero_eps=1e-5)
        if len(spans) != len(labels):
            raise RuntimeError(f"{fname}: expected {len(labels)} segments {labels}, found {len(spans)}: {spans}")

        seg_rows = []
        for lbl, (s0, s1) in zip(labels, spans):
            clip = x[s0:s1]
            m = measure_windowed_loudness(clip, sr=sr)
            dl = _extract_pair_deltas(clip, sr=sr)
            seg_rows.append(
                {
                    "label": lbl,
                    "start_s": round(s0 / float(sr), 3),
                    "stop_s": round(s1 / float(sr), 3),
                    **m,
                }
            )
            gkey = f"{lang}_{lbl}"
            gr, gd = groups.setdefault(gkey, ([], []))
            gr.append(m)
            gd.append(dl)

            ak = f"ALL_{lbl}"
            ar, ad = groups.setdefault(ak, ([], []))
            ar.append(m)
            ad.append(dl)

        per_file[fname] = seg_rows

    return {
        "window_params": {"win_ms": 40.0, "hop_ms": 20.0, "active_gate_dbfs_after_neg20_norm": -48.0},
        "per_file": per_file,
        "grouped_summary": {k: _aggregate_loudness_rows(rows, dls) for k, (rows, dls) in sorted(groups.items())},
    }


def main() -> int:
    rep = audit_v10_samples(SAMPLES_DIR)
    OUT_PATH.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
