"""Download and extract the DataBaker `BZNSYP` (CSMSC) studio Mandarin corpus (`exp_v15_bznsyp_tts`).

Usage:
  # 1. Download BZNSYP.tar.bz2 (needs network / BypassSandbox: true):
  D:\\Anaconda\\python.exe -B -m exp_v15_bznsyp_tts.fetch_bznsyp --download-only

  # 2. Extract and inspect on-disk format (runs inside sandbox):
  D:\\Anaconda\\python.exe -B -m exp_v15_bznsyp_tts.fetch_bznsyp --extract
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_v15_bznsyp_tts import BZNSYP_DIR, RAW_CACHE_DIR  # noqa: E402

ARCHIVE_PATH = RAW_CACHE_DIR / "BZNSYP.tar.bz2"
HF_MIRROR_URL = "https://hf-mirror.com/openspeech/BZNSYP/resolve/main/BZNSYP.tar.bz2"
HF_PRIMARY_URL = "https://huggingface.co/openspeech/BZNSYP/resolve/main/BZNSYP.tar.bz2"


def _read_text_graceful(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def download_bznsyp(out_path: Path = ARCHIVE_PATH) -> Path:
    """Download `BZNSYP.tar.bz2` via native Windows `curl.exe` (SChannel TLS, resume & stall timeout)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_file() and out_path.stat().st_size > 1_000_000_000:
        print(f"Archive already present: {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)", flush=True)
        return out_path

    curl_bin = r"C:\Windows\System32\curl.exe"
    tmp_path = out_path.with_suffix(".tar.bz2.part")

    for url in (HF_MIRROR_URL, HF_PRIMARY_URL):
        if tmp_path.is_file() and tmp_path.stat().st_size < 100_000:
            tmp_path.unlink(missing_ok=True)
        cmd = [
            curl_bin, "-f", "-L", "-C", "-", "--http1.1",
            "--retry", "4", "--retry-delay", "3",
            "--connect-timeout", "20", "--speed-time", "30", "--speed-limit", "20000",
            "-o", str(tmp_path), url,
        ]
        print(f"Downloading via curl.exe: {url}", flush=True)
        res = subprocess.run(cmd, check=False)
        sz = tmp_path.stat().st_size if tmp_path.is_file() else 0
        print(f"curl exit={res.returncode}, part_size={sz} bytes ({sz / 1e9:.2f} GB)", flush=True)
        if res.returncode == 0 and sz > 1_000_000_000:
            tmp_path.replace(out_path)
            print(f"Completed download: {out_path} ({sz / 1e9:.2f} GB)", flush=True)
            return out_path
        if tmp_path.is_file() and sz < 100_000:
            tmp_path.unlink(missing_ok=True)

    raise RuntimeError("curl.exe failed to download BZNSYP.tar.bz2 from both hf-mirror.com and huggingface.co.")


def extract_and_inspect_bznsyp(
    archive_path: Path = ARCHIVE_PATH,
    out_dir: Path = BZNSYP_DIR,
) -> dict:
    """Extract `BZNSYP.tar.bz2` cleanly and inspect exact `.interval` and `ProsodyLabeling` formats."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_wavs = list(out_dir.rglob("*.wav"))
    existing_intervals = list(out_dir.rglob("*.interval"))
    if len(existing_wavs) < 9900 or len(existing_intervals) < 9900:
        print(f"Extracting {archive_path} into {out_dir}...", flush=True)
        t0 = time.perf_counter()
        ok_tar = False
        if shutil.which("tar"):
            try:
                res = subprocess.run(
                    ["tar", "-xjf", str(archive_path), "-C", str(out_dir)],
                    capture_output=True,
                    text=True,
                )
                ok_tar = (res.returncode == 0)
                if not ok_tar:
                    print(f"System tar returned {res.returncode}: {res.stderr[:300]}; falling back to tarfile.", flush=True)
            except Exception as exc:
                print(f"System tar error ({exc}); falling back to tarfile.", flush=True)
        if not ok_tar:
            with tarfile.open(archive_path, "r:bz2") as tf:
                tf.extractall(out_dir)
        print(f"Extracted in {time.perf_counter() - t0:.1f}s.", flush=True)
    else:
        print(f"Already extracted: {len(existing_wavs)} wavs, {len(existing_intervals)} intervals.", flush=True)

    wavs = sorted(out_dir.rglob("*.wav"))
    intervals = sorted(out_dir.rglob("*.interval"))
    txts = sorted(out_dir.rglob("*.txt"))

    print(f"Counts -> .wav: {len(wavs)}, .interval: {len(intervals)}, .txt: {len(txts)}", flush=True)
    if wavs:
        print(f"Sample wav path: {wavs[0]}", flush=True)
    for txt_p in txts[:3]:
        lines = _read_text_graceful(txt_p).splitlines()[:10]
        print(f"\n--- Sample {txt_p.name} (first 10 lines) ---", flush=True)
        for ln in lines:
            print(" ", ln, flush=True)
    for iv_p in intervals[:2]:
        lines = _read_text_graceful(iv_p).splitlines()[:30]
        print(f"\n--- Sample {iv_p.name} (first 30 lines) ---", flush=True)
        for ln in lines:
            print(" ", ln, flush=True)

    return {
        "n_wavs": len(wavs),
        "n_intervals": len(intervals),
        "n_txts": len(txts),
        "sample_wav": str(wavs[0]) if wavs else None,
        "sample_interval": str(intervals[0]) if intervals else None,
        "sample_txts": [str(p) for p in txts],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch and extract BZNSYP (CSMSC) corpus.")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()

    if args.download_only:
        download_bznsyp()
        return 0
    if args.extract:
        info = extract_and_inspect_bznsyp()
        return 0 if (info["n_wavs"] >= 9900 and info["n_intervals"] >= 9900) else 1
    download_bznsyp()
    info = extract_and_inspect_bznsyp()
    return 0 if (info["n_wavs"] >= 9900 and info["n_intervals"] >= 9900) else 1


if __name__ == "__main__":
    raise SystemExit(main())
