"""Idempotent multi-speaker EN (CMU-ARCTIC) + ZH (AISHELL-3) 24 kHz corpus fetcher.

Sources:
- English: HuggingFace dataset ``MikhailT/cmu-arctic`` (parquet shards via
  ``huggingface_hub`` + ``pyarrow.parquet``, decoded with stdlib ``wave`` and
  resampled 16 kHz -> 24 kHz via ``scipy.signal.resample_poly``; avoids
  ``torchcodec`` / ``soundfile`` dependencies).
- Mandarin: GitHub ``fbngrm/mandarin-sentence-audio`` at pinned commit
  ``431c2d31d96db79999366f3d2ae82603c7b80005`` (uses non-recursive GitHub
  ``contents`` API with ``git`` sparse-tree fallback to avoid HTTP 500 on
  large recursive tree endpoints; MP3s decoded to 24 kHz mono 16-bit PCM WAV
  via ``ffmpeg.exe``).

Usage:
  D:\\Anaconda\\python.exe -B -m exp_v8_data.fetch_corpus --finalize-local
  D:\\Anaconda\\python.exe -B -m exp_v8_data.fetch_corpus --dry-run
  D:\\Anaconda\\python.exe -B -m exp_v8_data.fetch_corpus --english-only
  D:\\Anaconda\\python.exe -B -m exp_v8_data.fetch_corpus --mandarin-only
  D:\\Anaconda\\python.exe -B -m exp_v8_data.fetch_corpus
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import urllib.request
import wave
from collections import Counter

import numpy as np
from scipy.signal import resample_poly

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity, compute_f0_hnr_metrics, read_wav  # noqa: E402
from pybot_tts_v2_recorded import decode_donors  # noqa: E402

SR = 24000
CORPUS_DIR = pathlib.Path(__file__).resolve().parent / "corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"

EN_DATASET = "MikhailT/cmu-arctic"
EN_SPEAKERS = {
    "slt": {"gender": "female", "accent": "US", "target_n": 40},
    "clb": {"gender": "female", "accent": "US", "target_n": 40},
    "bdl": {"gender": "male", "accent": "US", "target_n": 40},
    "rms": {"gender": "male", "accent": "US", "target_n": 40},
}

ZH_REPO = "fbngrm/mandarin-sentence-audio"
ZH_COMMIT = "431c2d31d96db79999366f3d2ae82603c7b80005"
ZH_TARGET_MALE_N = 2
ZH_TARGET_FEMALE_N = 1
ZH_UTTS_PER_SPK = 20
EXISTING_ZH_SPEAKERS = {"SSB0502", "SSB1593"}

DEFAULT_FFMPEG = (
    r"C:\Users\MECHREVO\Documents\deepseek-harness\default-workspace"
    r"\.tools\ffmpeg\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe"
)


def _math_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _write_pcm16_wav(path: pathlib.Path, samples: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(samples, dtype=np.float64).ravel()
    pk = float(np.max(np.abs(arr))) if arr.size else 0.0
    if pk > 0.99:
        arr = arr * (0.98 / pk)
    pcm = np.round(np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def decode_audio_bytes_to_24k(raw_bytes: bytes, ffmpeg_bin: str | None = None) -> tuple[np.ndarray, int]:
    """Decode WAV bytes in-memory (or fallback via ffmpeg) and resample to 24 kHz mono."""
    if raw_bytes[:4] == b"RIFF":
        with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
            src_sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if sw == 2:
            arr = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
        elif sw == 4:
            arr = np.frombuffer(frames, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sampwidth={sw}")
        if nch > 1:
            arr = arr.reshape(-1, nch).mean(axis=1)
        if src_sr != SR:
            g = _math_gcd(SR, src_sr)
            arr = resample_poly(arr, SR // g, src_sr // g)
        return arr, SR

    if ffmpeg_bin is None:
        ffmpeg_bin = decode_donors.find_ffmpeg(DEFAULT_FFMPEG)
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", "-f", "wav", "pipe:1",
    ]
    res = subprocess.run(cmd, input=raw_bytes, capture_output=True, timeout=30)
    if res.returncode != 0 or len(res.stdout) <= 44:
        raise RuntimeError(f"ffmpeg pipe decode failed: {res.stderr.decode('utf-8', 'replace')}")
    return decode_audio_bytes_to_24k(res.stdout, ffmpeg_bin=ffmpeg_bin)


def extract_english_ipa_coverage(texts: list[str]) -> dict:
    """Compute ARPAbet/IPA phone coverage over English texts using crossling_frontend."""
    ipa_counts: Counter[str] = Counter()
    arpa_counts: Counter[str] = Counter()
    oov_words: Counter[str] = Counter()
    for text in texts:
        if not text:
            continue
        plan = cf.plan(text, allow_oov_fallback=True)
        for u in plan["units"]:
            if u.get("lang") != "en":
                continue
            for p in u.get("ipa_phones", []):
                ipa_counts[p] += 1
            for a in u.get("arpa_phones", []):
                arpa_counts[a] += 1
            if u.get("g2p_source") == "en_rule_fallback":
                oov_words[u.get("token", "")] += 1
    return {
        "n_unique_ipa": len(ipa_counts),
        "n_unique_arpa": len(arpa_counts),
        "ipa_counts": dict(sorted(ipa_counts.items())),
        "oov_rule_words": len(oov_words),
    }


def _cached_en_texts_by_speaker() -> dict[str, list[dict]]:
    """Read text/file metadata from local HF cmu-arctic parquet cache (no network)."""
    import pyarrow.parquet as pq

    cache_base = pathlib.Path(os.environ.get("USERPROFILE", "")) / ".cache" / "huggingface" / "hub" / "datasets--MikhailT--cmu-arctic"
    out: dict[str, list[dict]] = {}
    if not cache_base.is_dir():
        return out
    for spk in EN_SPEAKERS:
        pqs = sorted(cache_base.rglob(f"{spk}-*.parquet"))
        if not pqs:
            continue
        t = pq.read_table(pqs[0], columns=[c for c in ("speaker", "file", "text") if c in pq.read_schema(pqs[0]).names])
        out[spk] = t.to_pylist()
    return out


def finalize_local_manifest(out_dir: pathlib.Path = CORPUS_DIR) -> dict:
    """Scan local corpus/en and corpus/zh WAVs, validate sanity, recover EN texts, write manifest.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cached_meta = _cached_en_texts_by_speaker()
    old_doc = {}
    mpath = out_dir / "manifest.json"
    if mpath.is_file():
        try:
            old_doc = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            old_doc = {}
    old_zh_by_id = {r["id"]: r for r in old_doc.get("mandarin", {}).get("utterances", []) if "id" in r}
    old_en_by_id = {r["id"]: r for r in old_doc.get("english", {}).get("utterances", []) if "id" in r}

    # English
    en_records: list[dict] = []
    en_counts: dict[str, int] = {}
    en_dir = out_dir / "en"
    for spk in sorted(EN_SPEAKERS):
        spk_dir = en_dir / spk
        if not spk_dir.is_dir():
            en_counts[spk] = 0
            continue
        wavs = sorted(spk_dir.glob("*.wav"))
        meta_rows = cached_meta.get(spk, [])
        cnt = 0
        for idx, wav_path in enumerate(wavs):
            x, sr = read_wav(wav_path)
            sanity = check_wav_sanity(x, sr, expected_sr=SR, min_duration_s=0.3)
            if not sanity["ok"]:
                continue
            uid = wav_path.stem
            text = old_en_by_id.get(uid, {}).get("text", "")
            arctic_file = ""
            if not text and idx < len(meta_rows):
                text = str(meta_rows[idx].get("text", "")).strip()
                arctic_file = str(meta_rows[idx].get("file", ""))
            cnt += 1
            en_records.append({
                "id": uid,
                "speaker": spk,
                "gender": EN_SPEAKERS[spk]["gender"],
                "accent": EN_SPEAKERS[spk]["accent"],
                "lang": "en",
                "source": f"hf:{EN_DATASET}",
                "arctic_file": arctic_file,
                "text": text,
                "path": str(wav_path.relative_to(ROOT)).replace("\\", "/"),
                "sr": SR,
                "duration_s": sanity["duration_s"],
                "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            })
        en_counts[spk] = cnt

    # Mandarin (if present on disk)
    zh_records: list[dict] = []
    zh_counts: dict[str, int] = {}
    zh_selected: dict[str, str] = dict(old_doc.get("mandarin", {}).get("selected_speakers", {}))
    zh_dir = out_dir / "zh"
    if zh_dir.is_dir():
        for spk_dir in sorted(zh_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            spk = spk_dir.name
            cnt = 0
            f0s = []
            for wav_path in sorted(spk_dir.glob("*.wav")):
                x, sr = read_wav(wav_path)
                sanity = check_wav_sanity(x, sr, expected_sr=SR, min_duration_s=0.3)
                if not sanity["ok"]:
                    continue
                f0h = compute_f0_hnr_metrics(x, sr, fmin=70.0, fmax=400.0)
                if f0h and f0h["f0_median_hz"] is not None:
                    f0s.append(float(f0h["f0_median_hz"]))
                uid = wav_path.stem
                old_r = old_zh_by_id.get(uid, {})
                gender = zh_selected.get(spk) or old_r.get("gender") or "unknown"
                cnt += 1
                zh_records.append({
                    "id": uid,
                    "speaker": spk,
                    "gender": gender,
                    "lang": "zh",
                    "source": old_r.get("source", f"github:{ZH_REPO}@{ZH_COMMIT[:8]}"),
                    "text": old_r.get("text", ""),
                    "pinyin": old_r.get("pinyin", ""),
                    "path": str(wav_path.relative_to(ROOT)).replace("\\", "/"),
                    "sr": SR,
                    "duration_s": sanity["duration_s"],
                    "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                })
            if cnt > 0:
                zh_counts[spk] = cnt
                if spk not in zh_selected and f0s:
                    zh_selected[spk] = "male" if float(np.median(f0s)) < 160.0 else "female"
                    for r in zh_records:
                        if r["speaker"] == spk and r["gender"] == "unknown":
                            r["gender"] = zh_selected[spk]

    en_cov = extract_english_ipa_coverage([r["text"] for r in en_records])
    manifest = {
        "schema": 1,
        "sr": SR,
        "english": {
            "counts": en_counts,
            "utterances": en_records,
            "ipa_coverage": en_cov,
        },
        "mandarin": {
            "selected_speakers": zh_selected,
            "counts": zh_counts,
            "utterances": zh_records,
        } if zh_records else old_doc.get("mandarin", {}),
        "existing_aishell3_ref": "pybot_tts_v2_recorded/extended/index.json (SSB0502, SSB1593)",
        "total_utterances": len(en_records) + len(zh_records),
    }
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _save_manifest(out_dir: pathlib.Path, english_doc: dict, mandarin_doc: dict) -> pathlib.Path:
    all_utts = list(english_doc.get("utterances", [])) + list(mandarin_doc.get("utterances", []))
    manifest = {
        "schema": 1,
        "sr": SR,
        "english": english_doc,
        "mandarin": mandarin_doc,
        "existing_aishell3_ref": "pybot_tts_v2_recorded/extended/index.json (SSB0502, SSB1593)",
        "total_utterances": len(all_utts),
    }
    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return mpath


def fetch_english_cmu_arctic(
    out_dir: pathlib.Path,
    *,
    per_speaker: int = 40,
    dry_run: bool = False,
) -> dict:
    """Fetch 40 utts/speaker for slt, clb, bdl, rms from MikhailT/cmu-arctic."""
    en_dir = out_dir / "en"
    en_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run and all(len(list((en_dir / spk).glob("*.wav"))) >= per_speaker for spk in EN_SPEAKERS):
        return finalize_local_manifest(out_dir)["english"]

    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    api = HfApi()
    files = sorted(f for f in api.list_repo_files(EN_DATASET, repo_type="dataset") if f.endswith(".parquet"))
    if dry_run:
        return {
            "dataset": EN_DATASET,
            "parquet_files": len(files),
            "sample_files": files[:6],
            "target_speakers": EN_SPEAKERS,
            "per_speaker": per_speaker,
        }

    counts = {spk: 0 for spk in EN_SPEAKERS}
    records: list[dict] = []
    for pq_rel in files:
        if all(counts[s] >= per_speaker for s in EN_SPEAKERS):
            break
        lower_rel = pq_rel.lower()
        matching_spks = [s for s in EN_SPEAKERS if s in lower_rel]
        if matching_spks and all(counts[s] >= per_speaker for s in matching_spks):
            continue

        local_pq = hf_hub_download(repo_id=EN_DATASET, filename=pq_rel, repo_type="dataset")
        table = pq.read_table(local_pq)
        cols = set(table.column_names)
        spk_col = "speaker" if "speaker" in cols else ("speaker_id" if "speaker_id" in cols else None)
        txt_col = "text" if "text" in cols else ("sentence" if "sentence" in cols else None)
        id_col = "id" if "id" in cols else ("utterance_id" if "utterance_id" in cols else None)

        for row in table.to_pylist():
            spk = str(row.get(spk_col, "")).lower() if spk_col else ""
            if spk not in EN_SPEAKERS:
                aud = row.get("audio") or {}
                apath = str(aud.get("path", "")).lower()
                for cand in EN_SPEAKERS:
                    if cand in apath:
                        spk = cand
                        break
            if spk not in EN_SPEAKERS or counts[spk] >= per_speaker:
                continue

            uid = str(row.get(id_col) or f"{spk}_{counts[spk]:04d}")
            safe_uid = "".join(c if c.isalnum() or c in "-_" else "_" for c in uid)
            wav_path = en_dir / spk / f"{safe_uid}.wav"
            text = str(row.get(txt_col, "")).strip()

            if wav_path.is_file():
                x, sr = read_wav(wav_path)
            else:
                aud = row.get("audio") or {}
                raw_bytes = aud.get("bytes")
                if not raw_bytes:
                    continue
                x, sr = decode_audio_bytes_to_24k(raw_bytes)
                _write_pcm16_wav(wav_path, x, sr)

            sanity = check_wav_sanity(x, sr, expected_sr=SR, min_duration_s=0.3)
            if not sanity["ok"]:
                wav_path.unlink(missing_ok=True)
                continue

            counts[spk] += 1
            records.append({
                "id": safe_uid,
                "speaker": spk,
                "gender": EN_SPEAKERS[spk]["gender"],
                "accent": EN_SPEAKERS[spk]["accent"],
                "lang": "en",
                "source": f"hf:{EN_DATASET}",
                "arctic_file": str(row.get("file", "")),
                "text": text,
                "path": str(wav_path.relative_to(ROOT)).replace("\\", "/"),
                "sr": SR,
                "duration_s": sanity["duration_s"],
                "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            })

    coverage = extract_english_ipa_coverage([r["text"] for r in records])
    return {"counts": counts, "utterances": records, "ipa_coverage": coverage}


def _http_get_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "PyBotTTS-Fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "PyBotTTS-Fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _list_aishell3_speakers_nonrecursive() -> tuple[list[str], list[str]]:
    """List speaker dir names under audio/aishell3 and top-level metadata files without recursive trees."""
    root_url = f"https://api.github.com/repos/{ZH_REPO}/contents?ref={ZH_COMMIT}"
    aishell_url = f"https://api.github.com/repos/{ZH_REPO}/contents/audio/aishell3?ref={ZH_COMMIT}"
    root_items = _http_get_json(root_url)
    meta_files = [
        it["path"] for it in root_items
        if isinstance(it, dict) and it.get("type") == "file" and it.get("name", "").endswith((".json", ".tsv", ".csv", ".txt"))
    ]
    spk_items = _http_get_json(aishell_url)
    speakers = sorted(
        it["name"] for it in spk_items
        if isinstance(it, dict) and it.get("type") == "dir" and it.get("name") not in EXISTING_ZH_SPEAKERS
    )
    return speakers, meta_files


def _list_speaker_mp3s(spk: str) -> list[str]:
    url = f"https://api.github.com/repos/{ZH_REPO}/contents/audio/aishell3/{spk}?ref={ZH_COMMIT}"
    items = _http_get_json(url)
    return sorted(
        it["path"] for it in items
        if isinstance(it, dict) and it.get("type") == "file" and it.get("name", "").endswith(".mp3")
    )


def fetch_mandarin_aishell3(
    out_dir: pathlib.Path,
    *,
    utts_per_speaker: int = ZH_UTTS_PER_SPK,
    ffmpeg_path: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Fetch 2 Male + 1 contrasting Female AISHELL-3 speakers using non-recursive contents API."""
    speakers, meta_files = _list_aishell3_speakers_nonrecursive()
    if dry_run:
        return {
            "repo": ZH_REPO,
            "commit": ZH_COMMIT,
            "candidate_speakers_count": len(speakers),
            "sample_speakers": speakers[:20],
            "meta_files": meta_files[:20],
        }

    ffmpeg_bin = decode_donors.find_ffmpeg(ffmpeg_path or DEFAULT_FFMPEG)
    raw_base = f"https://raw.githubusercontent.com/{ZH_REPO}/{ZH_COMMIT}"

    transcripts_map: dict[str, dict] = {}
    for mf in meta_files:
        if any(k in mf.lower() for k in ("aishell", "index", "manifest", "transcript", "meta")):
            try:
                raw_txt = _http_get_bytes(f"{raw_base}/{mf}").decode("utf-8", "replace")
                if mf.endswith(".json"):
                    doc = json.loads(raw_txt)
                    rows = doc if isinstance(doc, list) else (doc.get("utterances") or doc.get("records") or list(doc.values()))
                    if isinstance(rows, list):
                        for r in rows:
                            if isinstance(r, dict) and ("id" in r or "audio" in r):
                                uid = str(r.get("id") or pathlib.Path(str(r.get("audio", ""))).stem)
                                transcripts_map[uid] = r
            except Exception:
                continue

    tmp_dir = out_dir / "_tmp_probe"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, str] = {}
    selected_mp3s: dict[str, list[str]] = {}
    male_cnt = 0
    female_cnt = 0

    for spk in speakers:
        if male_cnt >= ZH_TARGET_MALE_N and female_cnt >= ZH_TARGET_FEMALE_N:
            break
        try:
            mp3_list = _list_speaker_mp3s(spk)
        except Exception:
            continue
        if len(mp3_list) < utts_per_speaker:
            continue

        probe_mp3 = tmp_dir / f"{spk}_probe.mp3"
        probe_wav = tmp_dir / f"{spk}_probe.wav"
        if not probe_wav.is_file():
            probe_mp3.write_bytes(_http_get_bytes(f"{raw_base}/{mp3_list[0]}"))
            decode_donors.decode(probe_mp3, probe_wav, ffmpeg_bin, sr=SR)
        x_p, sr_p = read_wav(probe_wav)
        f0h = compute_f0_hnr_metrics(x_p, sr_p, fmin=70.0, fmax=400.0)
        if not f0h or f0h["f0_median_hz"] is None:
            continue
        f0_med = float(f0h["f0_median_hz"])
        if f0_med < 155.0 and male_cnt < ZH_TARGET_MALE_N:
            selected[spk] = "male"
            selected_mp3s[spk] = mp3_list
            male_cnt += 1
        elif f0_med > 180.0 and female_cnt < ZH_TARGET_FEMALE_N:
            selected[spk] = "female"
            selected_mp3s[spk] = mp3_list
            female_cnt += 1

    for f in tmp_dir.glob("*"):
        f.unlink(missing_ok=True)
    tmp_dir.rmdir()

    zh_dir = out_dir / "zh"
    zh_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    counts: dict[str, int] = {}

    for spk, gender in selected.items():
        spk_dir = zh_dir / spk
        spk_dir.mkdir(parents=True, exist_ok=True)
        counts[spk] = 0
        for mp3_rel in selected_mp3s[spk]:
            if counts[spk] >= utts_per_speaker:
                break
            uid = pathlib.Path(mp3_rel).stem
            wav_path = spk_dir / f"{uid}.wav"
            mp3_path = spk_dir / f"{uid}.mp3"
            if not wav_path.is_file():
                if not mp3_path.is_file():
                    mp3_path.write_bytes(_http_get_bytes(f"{raw_base}/{mp3_rel}"))
                decode_donors.decode(mp3_path, wav_path, ffmpeg_bin, sr=SR)
                mp3_path.unlink(missing_ok=True)
            x, sr = read_wav(wav_path)
            sanity = check_wav_sanity(x, sr, expected_sr=SR, min_duration_s=0.4)
            if not sanity["ok"]:
                wav_path.unlink(missing_ok=True)
                continue
            meta_r = transcripts_map.get(uid, {})
            counts[spk] += 1
            records.append({
                "id": uid,
                "speaker": spk,
                "gender": gender,
                "lang": "zh",
                "source": f"github:{ZH_REPO}@{ZH_COMMIT[:8]}",
                "text": meta_r.get("text", ""),
                "pinyin": meta_r.get("pinyin", ""),
                "path": str(wav_path.relative_to(ROOT)).replace("\\", "/"),
                "sr": SR,
                "duration_s": sanity["duration_s"],
                "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            })

    return {"selected_speakers": selected, "counts": counts, "utterances": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch multi-speaker EN+ZH 24 kHz corpus into exp_v8_data/corpus/")
    parser.add_argument("--finalize-local", action="store_true", help="Build/update manifest.json from local corpus WAVs + HF cache")
    parser.add_argument("--dry-run", action="store_true", help="Inspect remote indexes without writing WAVs")
    parser.add_argument("--english-only", action="store_true", help="Only fetch/inspect English CMU-ARCTIC")
    parser.add_argument("--mandarin-only", action="store_true", help="Only fetch/inspect Mandarin AISHELL-3")
    parser.add_argument("--en-per-speaker", type=int, default=40)
    parser.add_argument("--zh-per-speaker", type=int, default=ZH_UTTS_PER_SPK)
    parser.add_argument("--ffmpeg", default=DEFAULT_FFMPEG)
    parser.add_argument("--out-dir", type=pathlib.Path, default=CORPUS_DIR)
    args = parser.parse_args(argv)

    if args.finalize_local:
        manifest = finalize_local_manifest(args.out_dir)
        print(json.dumps({
            "manifest": str(args.out_dir / "manifest.json"),
            "total_utterances": manifest["total_utterances"],
            "en_counts": manifest["english"]["counts"],
            "en_ipa_unique": manifest["english"]["ipa_coverage"]["n_unique_ipa"],
            "zh_counts": manifest.get("mandarin", {}).get("counts", {}),
        }, indent=2, ensure_ascii=False))
        return 0

    do_en = not args.mandarin_only
    do_zh = not args.english_only

    if args.dry_run:
        summary = {"dry_run": True, "out_dir": str(args.out_dir)}
        if do_en:
            try:
                summary["english"] = fetch_english_cmu_arctic(args.out_dir, per_speaker=args.en_per_speaker, dry_run=True)
            except Exception as exc:
                summary["english_error"] = repr(exc)
        if do_zh:
            try:
                summary["mandarin"] = fetch_mandarin_aishell3(args.out_dir, utts_per_speaker=args.zh_per_speaker, ffmpeg_path=args.ffmpeg, dry_run=True)
            except Exception as exc:
                summary["mandarin_error"] = repr(exc)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest = {}
    if (args.out_dir / "manifest.json").is_file():
        existing_manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))

    english_doc = existing_manifest.get("english", {})
    mandarin_doc = existing_manifest.get("mandarin", {})

    if do_en:
        english_doc = fetch_english_cmu_arctic(args.out_dir, per_speaker=args.en_per_speaker, dry_run=False)
        mpath = _save_manifest(args.out_dir, english_doc, mandarin_doc)
        print(f"Saved incremental manifest (after English) -> {mpath}")
    if do_zh:
        mandarin_doc = fetch_mandarin_aishell3(
            args.out_dir, utts_per_speaker=args.zh_per_speaker, ffmpeg_path=args.ffmpeg, dry_run=False
        )
        mpath = _save_manifest(args.out_dir, english_doc, mandarin_doc)
        print(f"Saved manifest (after Mandarin) -> {mpath}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
