"""Fetch and validate the full 476-utterance `SSB0502` Mandarin corpus (`exp_v14_fluent_zh_tts`).

Usage:
  # 1. Inspect local WAV hits (sandboxed dry run):
  D:\\Anaconda\\python.exe -B -m exp_v14_fluent_zh_tts.fetch_ssb0502 --status

  # 2. Download missing MP3s to `_mp3_cache/` (needs network / BypassSandbox: true):
  D:\\Anaconda\\python.exe -B -m exp_v14_fluent_zh_tts.fetch_ssb0502 --download-only

  # 3. Decode MP3s -> 24 kHz WAVs, validate sanity, and build manifest (sandboxed):
  D:\\Anaconda\\python.exe -B -m exp_v14_fluent_zh_tts.fetch_ssb0502 --finalize
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any
import urllib.request

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_v5_eval.metrics import check_wav_sanity, read_wav  # noqa: E402
from exp_v8_data.fetch_corpus import (  # noqa: E402
    ZH_COMMIT,
    ZH_REPO,
    _write_pcm16_wav,
    decode_audio_bytes_to_24k,
)
from exp_v12_data.expand_corpus_v12 import _resolve_ffmpeg, _safe_split_pinyin  # noqa: E402
from exp_v14_fluent_zh_tts import CORPUS_DIR, EXP_DIR, MANIFEST_V14, SR  # noqa: E402

TRANSCRIPTS_SSB0502 = REPO_ROOT.parent / "_transcripts" / "SSB0502.json"
EXTENDED_INDEX = REPO_ROOT / "pybot_tts_v2_recorded" / "extended" / "index.json"
V12_ZH_SSB0502_DIR = REPO_ROOT / "exp_v12_data" / "corpus" / "zh" / "SSB0502"
MP3_CACHE_DIR = EXP_DIR / "_mp3_cache"


def load_valid_ssb0502_transcripts(
    transcript_path: Path = TRANSCRIPTS_SSB0502,
) -> list[dict[str, Any]]:
    """Filter valid `SSB0502` rows where len(text) == len(pinyin) and all syllables parse."""
    raw_rows = json.loads(transcript_path.read_text(encoding="utf-8"))
    valid: list[dict[str, Any]] = []
    for row in raw_rows:
        uid = str(row.get("id", "")).strip()
        text = str(row.get("text", "")).strip()
        pys = str(row.get("pinyin", "")).strip().split()
        if not uid or not text or len(text) != len(pys):
            continue
        parsed = [_safe_split_pinyin(py) for py in pys]
        if any(p is None for p in parsed):
            continue
        hsk_tokens = (
            row.get("grades", {})
            .get("hsk3_2025", {})
            .get("tokens", [])
        )
        hsk_words = [
            str(t.get("word", "")).strip()
            for t in hsk_tokens
            if isinstance(t, dict) and str(t.get("word", "")).strip()
        ]
        valid.append(
            {
                "id": uid,
                "speaker": "SSB0502",
                "text": text,
                "pinyin": pys,
                "parsed": [(str(o), str(f), int(t)) for o, f, t in parsed if True],
                "audio": str(row.get("audio", f"audio/aishell3/SSB0502/{uid}.mp3")),
                "split": str(row.get("split", "train")),
                "hsk_words": hsk_words,
            }
        )
    return valid


def _find_existing_local_wavs() -> dict[str, Path]:
    """Locate existing 24 kHz SSB0502 WAVs from frozen v2/v12/v14 directories."""
    existing: dict[str, Path] = {}
    if EXTENDED_INDEX.is_file():
        ext_doc = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))
        for u in ext_doc.get("utterances", []):
            if u.get("speaker") == "SSB0502":
                p = Path(u["path"])
                if not p.is_absolute():
                    p = REPO_ROOT / p
                if p.is_file():
                    existing[u["id"]] = p
    if V12_ZH_SSB0502_DIR.is_dir():
        for wav_p in V12_ZH_SSB0502_DIR.glob("SSB0502*.wav"):
            existing.setdefault(wav_p.stem, wav_p)
    if CORPUS_DIR.is_dir():
        for wav_p in CORPUS_DIR.glob("SSB0502*.wav"):
            existing[wav_p.stem] = wav_p
    return existing


def _download_one_mp3(item: dict[str, Any], raw_base: str, mp3_dir: Path, retries: int = 2) -> tuple[str, str]:
    uid = item["id"]
    mp3_path = mp3_dir / f"{uid}.mp3"
    if mp3_path.is_file() and mp3_path.stat().st_size > 512:
        return uid, "cached"
    url = f"{raw_base}/{item['audio']}"
    last_err = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PyBotTTS-v14-Fetcher/1.0"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = resp.read()
            if len(data) <= 512:
                raise RuntimeError(f"short response ({len(data)} B)")
            mp3_path.parent.mkdir(parents=True, exist_ok=True)
            mp3_path.write_bytes(data)
            return uid, "downloaded"
        except Exception as exc:
            last_err = str(exc)
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1))
    return uid, f"error:{last_err}"


def download_missing_mp3s(max_workers: int = 8) -> dict[str, Any]:
    """Download missing `SSB0502` MP3s from pinned AISHELL-3 GitHub mirror."""
    rows = load_valid_ssb0502_transcripts()
    local_wavs = _find_existing_local_wavs()
    raw_base = f"https://raw.githubusercontent.com/{ZH_REPO}/{ZH_COMMIT}"
    MP3_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    todo = [r for r in rows if r["id"] not in local_wavs]
    stats = {"total_valid": len(rows), "already_local_wav": len(rows) - len(todo), "to_fetch": len(todo)}
    downloaded = 0
    cached = 0
    errors: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_download_one_mp3, item, raw_base, MP3_CACHE_DIR) for item in todo]
        for fut in as_completed(futs):
            uid, status = fut.result()
            if status == "downloaded":
                downloaded += 1
            elif status == "cached":
                cached += 1
            else:
                errors.append({"id": uid, "error": status})

    stats.update({"downloaded": downloaded, "mp3_cached": cached, "errors": errors})
    return stats


def _materialize_one_wav(
    item: dict[str, Any],
    local_src: Path | None,
    mp3_dir: Path,
    out_dir: Path,
    ffmpeg_bin: str,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    uid = item["id"]
    dst_wav = out_dir / f"{uid}.wav"
    if not dst_wav.is_file():
        if local_src is not None and local_src.is_file():
            dst_wav.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_src, dst_wav)
        else:
            mp3_p = mp3_dir / f"{uid}.mp3"
            if not mp3_p.is_file():
                return None, {"id": uid, "error": "missing_wav_and_mp3"}
            try:
                x, sr = decode_audio_bytes_to_24k(mp3_p.read_bytes(), ffmpeg_bin=ffmpeg_bin)
                _write_pcm16_wav(dst_wav, x, sr)
            except Exception as exc:
                return None, {"id": uid, "error": f"decode_fail:{exc}"}

    try:
        x, sr = read_wav(dst_wav)
    except Exception as exc:
        dst_wav.unlink(missing_ok=True)
        return None, {"id": uid, "error": f"read_fail:{exc}"}

    sanity = check_wav_sanity(x, sr, expected_sr=SR, min_duration_s=0.3)
    if not sanity["ok"]:
        dst_wav.unlink(missing_ok=True)
        return None, {"id": uid, "error": f"sanity_fail:{sanity}"}

    rec = {
        "id": uid,
        "speaker": "SSB0502",
        "gender": "female",
        "lang": "zh",
        "source": f"github:{ZH_REPO}@{ZH_COMMIT[:8]}",
        "split": item["split"],
        "text": item["text"],
        "pinyin": item["pinyin"],
        "parsed_syllables": [
            {"onset": o, "final": f, "tone": t} for o, f, t in item["parsed"]
        ],
        "hsk_words": item["hsk_words"],
        "path": str(dst_wav.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sr": SR,
        "duration_s": sanity["duration_s"],
        "samples": int(len(x)),
        "wav_sha256": hashlib.sha256(dst_wav.read_bytes()).hexdigest(),
    }
    return rec, None


def finalize_manifest_v14(ffmpeg_path: str | None = None, *, clean_mp3_cache: bool = True) -> dict[str, Any]:
    """Decode cached MP3s + copy existing WAVs into `CORPUS_DIR` and write `MANIFEST_V14`."""
    rows = load_valid_ssb0502_transcripts()
    local_wavs = _find_existing_local_wavs()
    ffmpeg_bin = _resolve_ffmpeg(ffmpeg_path)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    recs_by_id: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [
            pool.submit(_materialize_one_wav, item, local_wavs.get(item["id"]), MP3_CACHE_DIR, CORPUS_DIR, ffmpeg_bin)
            for item in rows
        ]
        for fut in as_completed(futs):
            rec, err = fut.result()
            if rec is not None:
                recs_by_id[rec["id"]] = rec
            if err is not None:
                errors.append(err)

    records = [recs_by_id[r["id"]] for r in rows if r["id"] in recs_by_id]
    total_syls = sum(len(r["pinyin"]) for r in records)
    unique_of = len({(s["onset"], s["final"]) for r in records for s in r["parsed_syllables"]})
    unique_oft = len({(s["onset"], s["final"], s["tone"]) for r in records for s in r["parsed_syllables"]})

    # Preserve clean_10_ids if already populated in an existing manifest
    prev_clean_10: list[str] = []
    if MANIFEST_V14.is_file():
        try:
            prev = json.loads(MANIFEST_V14.read_text(encoding="utf-8"))
            prev_clean_10 = list(prev.get("clean_10_ids", []))
        except Exception:
            prev_clean_10 = []

    manifest = {
        "schema": 1,
        "speaker": "SSB0502",
        "sr": SR,
        "source": f"github:{ZH_REPO}@{ZH_COMMIT[:8]}",
        "total_utterances": len(records),
        "total_syllables": total_syls,
        "unique_of": unique_of,
        "unique_oft": unique_oft,
        "clean_10_ids": prev_clean_10,
        "errors": errors,
        "utterances": records,
    }
    MANIFEST_V14.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_V14.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if clean_mp3_cache and not errors and MP3_CACHE_DIR.is_dir():
        shutil.rmtree(MP3_CACHE_DIR, ignore_errors=True)

    return {
        "total_utterances": len(records),
        "total_syllables": total_syls,
        "unique_of": unique_of,
        "unique_oft": unique_oft,
        "errors": len(errors),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch & materialize full SSB0502 Mandarin corpus for v14.")
    ap.add_argument("--status", action="store_true", help="Print local availability counts without network/write")
    ap.add_argument("--download-only", action="store_true", help="Only download missing MP3s into _mp3_cache/")
    ap.add_argument("--finalize", action="store_true", help="Only decode MP3s/copy local WAVs and write manifest")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.status:
        rows = load_valid_ssb0502_transcripts()
        local_wavs = _find_existing_local_wavs()
        mp3s = {p.stem for p in MP3_CACHE_DIR.glob("*.mp3")} if MP3_CACHE_DIR.is_dir() else set()
        missing = [r["id"] for r in rows if r["id"] not in local_wavs and r["id"] not in mp3s]
        print(json.dumps({
            "valid_transcripts": len(rows),
            "existing_local_wavs": sum(1 for r in rows if r["id"] in local_wavs),
            "cached_mp3s": sum(1 for r in rows if r["id"] in mp3s),
            "missing_to_download": len(missing),
        }, indent=2))
        return 0

    if args.download_only:
        res = download_missing_mp3s(max_workers=args.workers)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if not res["errors"] else 1

    if args.finalize:
        res = finalize_manifest_v14()
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["total_utterances"] >= 465 else 1

    dl = download_missing_mp3s(max_workers=args.workers)
    fin = finalize_manifest_v14()
    print(json.dumps({"download": dl, "finalize": fin}, indent=2, ensure_ascii=False))
    return 0 if fin["total_utterances"] >= 465 else 1


if __name__ == "__main__":
    raise SystemExit(main())
