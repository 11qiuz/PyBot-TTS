"""Stage 11 Full-Band Studio LJSpeech-1.1 Corpus Extractor & Zero-Leakage Curator.

1. Extracts `LJSpeech-1.1.tar.bz2` (22.05 kHz studio female audiobook corpus, 13,100 utts)
   into `exp_20_pybot_larynx/crossling_data/ljspeech_raw/LJSpeech-1.1/`.
2. Scans `metadata.csv` and WAVs in parallel to enforce:
   - Clean normalized English text (`4 <= n_words <= 24`), zero G2P failures under `plan_en_v2`.
   - Clean studio active span `2.0 <= dur_s <= 7.5`, no clipping, healthy RMS.
   - `max_internal_sil_ms <= 220.0` (prevents long mid-sentence pauses from corrupting
     pause-free GMM-HMM forced alignment).
3. Enforces a strict chapter-isolated zero-leakage split:
   - `LJ050-*` reserved strictly for **HELDOUT** (`30` diverse utterances).
   - `LJ001-*` .. `LJ049-*` curated for **TRAIN** (`3,500` utterances selected via greedy
     rare phone-context / diphone / phoneme-class coverage followed by diverse length fill).
4. Resamples cleanly `22,050 Hz -> 24,000 Hz` via `scipy.signal.resample_poly(x, 160, 147)`
   and saves PCM16 mono WAVs + `manifest.json` compatible with `align_en_stage10.py`.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path
import re
import sys
import tarfile
import time
import wave
from typing import Any

import numpy as np
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import EXP_DIR, SR  # noqa: E402
from exp_20_pybot_larynx.crossling.en_frontend_v2 import plan_en_v2  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity  # noqa: E402
from exp_v8_data.fetch_corpus import _write_pcm16_wav  # noqa: E402

CROSSLING_DATA_DIR = EXP_DIR / "crossling_data"
RAW_DIR = CROSSLING_DATA_DIR / "ljspeech_raw"
TAR_PATH = RAW_DIR / "LJSpeech-1.1.tar.bz2"
EXTRACTED_DIR = RAW_DIR / "LJSpeech-1.1"
OUT_DIR = CROSSLING_DATA_DIR / "ljspeech_s11"
WAVS_OUT_DIR = OUT_DIR / "wavs"
MANIFEST_OUT_PATH = OUT_DIR / "manifest.json"

LJ_SR = 22050
_CLEAN_TEXT_RE = re.compile(r"^[A-Za-z0-9\s\.,!\?;:'\"\-]+$")


def _worker_init() -> None:
    for _ev in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_ev] = "1"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def ensure_extracted(tar_path: Path = TAR_PATH, extracted_dir: Path = EXTRACTED_DIR) -> Path:
    """Extract `LJSpeech-1.1.tar.bz2` into `extracted_dir` if not already extracted."""
    meta = extracted_dir / "metadata.csv"
    wavs_dir = extracted_dir / "wavs"
    if meta.is_file() and wavs_dir.is_dir() and any(wavs_dir.glob("LJ*.wav")):
        return extracted_dir
    if not tar_path.is_file():
        raise FileNotFoundError(f"Missing LJSpeech archive at {tar_path}")
    print(f"[LJSpeech] Extracting {tar_path} -> {extracted_dir.parent} ...", flush=True)
    t0 = time.perf_counter()
    with tarfile.open(tar_path, "r:bz2") as tf:
        tf.extractall(extracted_dir.parent)
    if not meta.is_file():
        raise FileNotFoundError(f"Extraction succeeded but {meta} missing")
    print(f"[LJSpeech] Extraction completed in {time.perf_counter() - t0:.1f}s", flush=True)
    return extracted_dir


def _read_wav_float64(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"Unsupported sample width {sw} in {path}")
    arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if n_ch > 1:
        arr = arr.reshape(-1, n_ch).mean(axis=1)
    return arr, sr


def _analyze_candidate(task: tuple[str, str, str]) -> dict[str, Any] | None:
    """Screen one LJSpeech item for text cleanliness, internal pause purity, and diphones."""
    _worker_init()
    uid, wav_str, text = task
    text = text.strip()
    if not text or not _CLEAN_TEXT_RE.fullmatch(text):
        return None
    words_raw = [w for w in re.findall(r"[A-Za-z']+", text) if w]
    if not (4 <= len(words_raw) <= 24):
        return None
    if any(ch.isdigit() for ch in text):
        return None

    wav_path = Path(wav_str)
    if not wav_path.is_file():
        return None

    try:
        x, sr = _read_wav_float64(wav_path)
    except Exception:
        return None
    if len(x) < int(1.8 * sr) or len(x) > int(9.0 * sr):
        return None

    pk = float(np.max(np.abs(x)))
    if pk <= 0.02 or pk >= 0.998:
        return None

    # Measure trimmed active span and internal silence runs (5 ms hop, 20 ms frame)
    hop = max(1, int(round(0.005 * sr)))
    win = max(hop * 4, int(round(0.020 * sr)))
    if len(x) < win * 4:
        return None
    n_fr = (len(x) - win) // hop + 1
    idx = np.arange(win)[None, :] + (np.arange(n_fr)[:, None] * hop)
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms)
    speech_lvl = float(np.percentile(rms_db, 85))
    thr_active = max(-52.0, speech_lvl - 26.0)
    act = np.where(rms_db > thr_active)[0]
    if len(act) < 60:
        return None

    a0, a1 = int(act[0]), int(act[-1])
    dur_s = float((a1 - a0 + 1) * 0.005)
    if not (2.0 <= dur_s <= 7.5):
        return None

    # Internal silence gap check strictly inside active speech span
    inner = rms_db[a0 : a1 + 1] <= max(-54.0, speech_lvl - 28.0)
    max_sil_fr = 0
    cur = 0
    for flag in inner:
        if flag:
            cur += 1
            if cur > max_sil_fr:
                max_sil_fr = cur
        else:
            cur = 0
    max_sil_ms = max_sil_fr * 5.0
    if max_sil_ms > 220.0:
        return None

    try:
        plan = plan_en_v2(text)
    except Exception:
        return None
    phones = plan.get("phones") or []
    if len(phones) < 12 or len(phones) > 130:
        return None

    ipas = [str(p["ipa"]) for p in phones]
    allos = [str(p["allophone"]) for p in phones]
    diphones = list({(ipas[i], ipas[i + 1]) for i in range(len(ipas) - 1)})
    triphones = list({(ipas[i - 1] if i > 0 else "SIL", allos[i], ipas[i + 1] if i + 1 < len(ipas) else "SIL") for i in range(len(ipas))})

    return {
        "id": uid,
        "chapter": uid.split("-", 1)[0],
        "wav_src": wav_str,
        "text": text,
        "dur_s": round(dur_s, 3),
        "n_phones": len(phones),
        "max_sil_ms": round(max_sil_ms, 1),
        "is_question": bool(plan.get("is_question", False)),
        "diphones": diphones,
        "triphones": triphones,
        "phones_set": list(set(allos)),
    }


def _select_greedy_diverse(cands: list[dict[str, Any]], target_n: int) -> list[dict[str, Any]]:
    """Select `target_n` TRAIN utterances greedily maximizing rare phone/diphone coverage + length diversity."""
    if len(cands) <= target_n:
        return list(cands)

    di_freq: Counter[tuple[str, str]] = Counter()
    ph_freq: Counter[str] = Counter()
    for c in cands:
        for d in c["diphones"]:
            di_freq[d] += 1
        for p in c["phones_set"]:
            ph_freq[p] += 1

    # Stage 1: greedily cover up to 75% via marginal rare-unit gain
    phase1_n = min(len(cands), int(0.75 * target_n))
    cov_di: Counter[tuple[str, str]] = Counter()
    cov_ph: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    used: set[int] = set()

    # Precompute rarity weights
    cand_scores_static: list[float] = []
    for c in cands:
        sc = sum(1.0 / math.sqrt(max(1, ph_freq[p])) for p in c["phones_set"])
        sc += 0.35 * sum(1.0 / math.sqrt(max(1, di_freq[d])) for d in c["diphones"])
        # Slight preference for crisp low-silence utterances
        sc += max(0.0, (180.0 - c["max_sil_ms"]) / 180.0) * 0.4
        cand_scores_static.append(sc)

    order_init = sorted(range(len(cands)), key=lambda i: (-cand_scores_static[i], cands[i]["id"]))
    pool = order_init[: min(len(order_init), target_n * 3)]

    while len(selected) < phase1_n:
        best_i = -1
        best_gain = -1.0
        for idx in pool:
            if idx in used:
                continue
            c = cands[idx]
            g = 0.0
            for p in c["phones_set"]:
                if cov_ph[p] < 12:
                    g += 3.0 / (1.0 + cov_ph[p])
            for d in c["diphones"]:
                if cov_di[d] < 4:
                    g += 1.2 / (1.0 + cov_di[d])
            g += 0.05 * cand_scores_static[idx]
            if g > best_gain:
                best_gain = g
                best_i = idx
        if best_i < 0:
            break
        used.add(best_i)
        chosen = cands[best_i]
        selected.append(chosen)
        for p in chosen["phones_set"]:
            cov_ph[p] += 1
        for d in chosen["diphones"]:
            cov_di[d] += 1

    # Stage 2: fill remainder evenly across duration quartiles & chapters
    rem = [i for i in range(len(cands)) if i not in used]
    rem.sort(key=lambda i: (cands[i]["max_sil_ms"], -len(cands[i]["triphones"]), cands[i]["id"]))
    need = target_n - len(selected)
    if need > 0 and rem:
        step = max(1, len(rem) // need)
        picked_rem = rem[::step][:need]
        used.update(picked_rem)
        selected.extend(cands[i] for i in picked_rem)
        for i in rem:
            if len(selected) >= target_n:
                break
            if i not in used:
                used.add(i)
                selected.append(cands[i])

    return selected[:target_n]


def _select_heldout(cands: list[dict[str, Any]], n_heldout: int = 30) -> list[dict[str, Any]]:
    """Select `n_heldout` diverse utterances strictly from `LJ050-*`."""
    if len(cands) < n_heldout:
        raise RuntimeError(f"Only {len(cands)} clean LJ050 heldout candidates (< {n_heldout})")
    ordered = sorted(cands, key=lambda c: (not c["is_question"], c["dur_s"], c["id"]))
    step = max(1, len(ordered) // n_heldout)
    picked = ordered[::step][:n_heldout]
    seen = {c["id"] for c in picked}
    for c in ordered:
        if len(picked) >= n_heldout:
            break
        if c["id"] not in seen:
            picked.append(c)
            seen.add(c["id"])
    return sorted(picked, key=lambda c: c["id"])


def _worker_resample(task: tuple[str, str, str, str, str]) -> dict[str, Any]:
    """Resample one 22.05 kHz WAV cleanly to 24 kHz mono PCM16 and verify sanity."""
    _worker_init()
    uid, split, src_str, dst_str, text = task
    x, sr = _read_wav_float64(Path(src_str))
    g = math.gcd(SR, sr)
    y24 = resample_poly(x, SR // g, sr // g)
    pk = float(np.max(np.abs(y24))) if y24.size else 0.0
    if pk > 0.96:
        y24 = y24 * (0.96 / pk)
    _write_pcm16_wav(Path(dst_str), y24, sr=SR)
    san = check_wav_sanity(y24, SR, expected_sr=SR, min_duration_s=0.5)
    rel_path = Path(dst_str).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    return {
        "id": uid,
        "speaker": "ljspeech",
        "split": split,
        "path": rel_path,
        "wav_path": rel_path,
        "text": text,
        "dur_s": round(len(y24) / float(SR), 3),
        "sanity_ok": bool(san.get("ok", False)),
    }


def fetch_and_curate_ljspeech(
    *,
    n_train: int = 3500,
    n_heldout: int = 30,
    workers: int = 12,
    force: bool = False,
) -> dict[str, Any]:
    """Extract, filter, curate, and resample 24 kHz LJSpeech-1.1 TRAIN + HELDOUT manifest."""
    t0 = time.perf_counter()
    if MANIFEST_OUT_PATH.is_file() and not force:
        cur = json.loads(MANIFEST_OUT_PATH.read_text(encoding="utf-8"))
        if int(cur.get("n_train", 0)) == n_train and int(cur.get("n_heldout", 0)) == n_heldout:
            print(f"[LJSpeech] Reusing existing {MANIFEST_OUT_PATH} ({n_train}+{n_heldout})", flush=True)
            return cur

    raw_root = ensure_extracted()
    meta_path = raw_root / "metadata.csv"
    wav_raw_dir = raw_root / "wavs"

    tasks: list[tuple[str, str, str]] = []
    with meta_path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) < 3:
                continue
            uid = row[0].strip()
            norm_txt = row[2].strip() or row[1].strip()
            tasks.append((uid, str(wav_raw_dir / f"{uid}.wav"), norm_txt))

    print(f"[LJSpeech] Screening {len(tasks)} metadata rows (workers={workers})...", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        screened = [r for r in ex.map(_analyze_candidate, tasks, chunksize=64) if r is not None]

    train_cands = [r for r in screened if r["chapter"] != "LJ050"]
    held_cands = [r for r in screened if r["chapter"] == "LJ050"]
    print(f"[LJSpeech] Screened clean: {len(train_cands)} TRAIN candidates (LJ001-LJ049), {len(held_cands)} HELDOUT candidates (LJ050)", flush=True)

    held_sel = _select_heldout(held_cands, n_heldout=n_heldout)
    train_sel = _select_greedy_diverse(train_cands, target_n=n_train)

    train_ids = [c["id"] for c in train_sel]
    held_ids = [c["id"] for c in held_sel]
    assert not (set(train_ids) & set(held_ids)), "Zero-leakage split violated!"

    WAVS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    resamp_tasks = [
        (c["id"], "train", c["wav_src"], str(WAVS_OUT_DIR / f"{c['id']}.wav"), c["text"])
        for c in train_sel
    ] + [
        (c["id"], "heldout", c["wav_src"], str(WAVS_OUT_DIR / f"{c['id']}.wav"), c["text"])
        for c in held_sel
    ]
    print(f"[LJSpeech] Resampling {len(resamp_tasks)} WAVs (22,050 Hz -> 24,000 Hz)...", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        utts = list(ex.map(_worker_resample, resamp_tasks, chunksize=32))

    n_failed = sum(1 for u in utts if not u["sanity_ok"])
    total_dur_s = round(sum(float(u["dur_s"]) for u in utts), 2)
    cov_phones = len({p for c in train_sel for p in c["phones_set"]})
    cov_diphones = len({d for c in train_sel for d in c["diphones"]})

    manifest = {
        "schema": 1,
        "corpus": "ljspeech_1_1",
        "source": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        "sr": SR,
        "format": "PCM16 mono",
        "speaker": "ljspeech",
        "total_utterances": len(utts),
        "n_train": len(train_ids),
        "n_heldout": len(held_ids),
        "heldout_chapter": "LJ050",
        "total_duration_s": total_dur_s,
        "n_sanity_failed": n_failed,
        "covered_allophones": cov_phones,
        "covered_diphones": cov_diphones,
        "heldout_ids": held_ids,
        "train_ids": train_ids,
        "utterances": utts,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    MANIFEST_OUT_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[LJSpeech] Wrote {MANIFEST_OUT_PATH}: {len(train_ids)} TRAIN + {len(held_ids)} HELDOUT "
        f"({total_dur_s / 3600.0:.2f} h, allophones={cov_phones}, diphones={cov_diphones}, failed={n_failed})",
        flush=True,
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 11 LJSpeech-1.1 24kHz Corpus Curator")
    ap.add_argument("--n-train", type=int, default=3500)
    ap.add_argument("--n-heldout", type=int, default=30)
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 8))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    fetch_and_curate_ljspeech(n_train=args.n_train, n_heldout=args.n_heldout, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
