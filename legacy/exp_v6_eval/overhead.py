"""Overhead profiler: cold-start import/init ms, CPU RTF, peak RSS MB, profile byte size."""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import subprocess
import sys
import time
import tracemalloc
from typing import Any, Callable

import numpy as np
import psutil


def _deep_sizeof(obj: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)
    total = sys.getsizeof(obj)
    if isinstance(obj, np.ndarray):
        total += int(obj.nbytes)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            total += _deep_sizeof(k, seen) + _deep_sizeof(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for it in obj:
            total += _deep_sizeof(it, seen)
    return total


def profile_bytes(obj: Any) -> dict:
    """Measure serialized JSON/gzip footprint and recursive in-memory bytes."""
    def _to_jsonable(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, dict):
            return {str(k): _to_jsonable(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_to_jsonable(it) for it in v]
        return v

    raw = json.dumps(_to_jsonable(obj), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    gz = gzip.compress(raw)
    return {
        "json_bytes": len(raw),
        "gzip_bytes": len(gz),
        "in_memory_bytes": _deep_sizeof(obj),
    }


def profile_cold_start(
    import_stmt: str,
    init_stmt: str = "",
    *,
    cwd: str | pathlib.Path | None = None,
    python_exe: str = sys.executable,
    timeout_s: float = 30.0,
) -> dict:
    """Measure fresh-subprocess import time, init time, and RSS/peak RSS MB."""
    script = (
        "import json, time, psutil\n"
        "proc = psutil.Process()\n"
        "t0 = time.perf_counter()\n"
        f"{import_stmt}\n"
        "t1 = time.perf_counter()\n"
        f"{init_stmt}\n"
        "t2 = time.perf_counter()\n"
        "mi = proc.memory_info()\n"
        "peak = getattr(mi, 'peak_wset', mi.rss)\n"
        "print(json.dumps({\n"
        "  'import_ms': (t1 - t0) * 1000.0,\n"
        "  'init_ms': (t2 - t1) * 1000.0,\n"
        "  'total_ms': (t2 - t0) * 1000.0,\n"
        "  'rss_mb': mi.rss / (1024 * 1024),\n"
        "  'peak_rss_mb': peak / (1024 * 1024),\n"
        "}))\n"
    )
    run_cwd = str(cwd) if cwd is not None else str(pathlib.Path(__file__).resolve().parent.parent)
    res = subprocess.run(
        [python_exe, "-B", "-c", script],
        cwd=run_cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout_s,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Cold-start probe failed (rc={res.returncode}): {res.stderr.strip()}")
    data = json.loads(res.stdout.strip().splitlines()[-1])
    return {k: round(float(v), 3) for k, v in data.items()}


def profile_synthesis(
    synth_fn: Callable[[], Any],
    *,
    sr: int = 24000,
    repeats: int = 3,
) -> dict:
    """Profile CPU RTF (wall / audio_dur), peak traced memory, and RSS delta."""
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss / (1024 * 1024)
    times_s: list[float] = []
    n_samples = 0

    tracemalloc.start()
    try:
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            out = synth_fn()
            t1 = time.perf_counter()
            times_s.append(t1 - t0)
            n_samples = len(out)
        _, peak_traced_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    rss_after = proc.memory_info().rss / (1024 * 1024)
    audio_dur_s = n_samples / float(sr) if sr > 0 else 0.0
    med_s = float(np.median(times_s))
    first_s = float(times_s[0])

    return {
        "repeats": int(repeats),
        "n_samples": int(n_samples),
        "audio_dur_s": round(audio_dur_s, 4),
        "first_wall_ms": round(first_s * 1000.0, 3),
        "median_wall_ms": round(med_s * 1000.0, 3),
        "cpu_rtf_first": round(first_s / audio_dur_s, 5) if audio_dur_s > 0 else None,
        "cpu_rtf_median": round(med_s / audio_dur_s, 5) if audio_dur_s > 0 else None,
        "rss_before_mb": round(rss_before, 2),
        "rss_after_mb": round(rss_after, 2),
        "rss_delta_mb": round(rss_after - rss_before, 3),
        "peak_traced_kb": round(peak_traced_bytes / 1024.0, 2),
    }
