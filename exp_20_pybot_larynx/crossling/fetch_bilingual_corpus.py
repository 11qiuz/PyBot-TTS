"""Stage 10 Bilingual & Studio English Corpus Fetcher (`exp_20_pybot_larynx`).

1. Verifies/fetches plain-text `exp_v10_g2p_and_align/lexicons/cmudict.dict`.
2. Extracts all 1,132 studio female CMU-ARCTIC `slt` utterances from the local HF
   parquet cache into `exp_20_pybot_larynx/crossling_data/slt_full/` at 24 kHz
   mono PCM16, with an explicit zero-leakage 30-utterance HELDOUT split
   (`slt_0030`..`slt_0059`).
3. Processes `CAiRE/ASCEND` Mandarin-English code-switching data into
   `exp_20_pybot_larynx/crossling_data/bilingual_ref/` (8 clean 24 kHz bilingual WAVs
   + transcripts) and writes `exp_20_pybot_larynx/crossling_data/bilingual_ref_stats.json`
   with measured human ZH<->EN metrics.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import io
import json
import math
from pathlib import Path
import re
import sys
import urllib.request
import wave
from typing import Any

import numpy as np
from scipy.fft import rfftfreq
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import EXP_DIR, N_FFT, SR  # noqa: E402
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB  # noqa: E402
from exp_20_pybot_larynx.clone.profile import _estimate_syllable_rate_hz  # noqa: E402
from exp_20_pybot_larynx.frontend.prosody_rules import _safe_plan_v13  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity, compute_f0_hnr_metrics  # noqa: E402
from exp_v6_vocoder import vocoder as v6_voc  # noqa: E402
from exp_v8_data.fetch_corpus import _write_pcm16_wav, decode_audio_bytes_to_24k  # noqa: E402
from exp_v9_clarity_pro.hnm_v9 import analyze_v9  # noqa: E402
from exp_v10_g2p_and_align.landmark_aligner import align_ipa_sequence_v10  # noqa: E402

CMUDICT_URL = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
CMUDICT_PATH = REPO_ROOT / "legacy" / "exp_v10_g2p_and_align" / "lexicons" / "cmudict.dict"

CROSSLING_DATA_DIR = EXP_DIR / "crossling_data"
SLT_FULL_DIR = CROSSLING_DATA_DIR / "slt_full"
SLT_WAVS_DIR = SLT_FULL_DIR / "wavs"
SLT_MANIFEST_PATH = SLT_FULL_DIR / "manifest.json"

BILINGUAL_REF_DIR = CROSSLING_DATA_DIR / "bilingual_ref"
BILINGUAL_REF_STATS_PATH = CROSSLING_DATA_DIR / "bilingual_ref_stats.json"

HF_HUB_DIR = Path(os.environ.get("USERPROFILE", r"C:\Users\MECHREVO")) / ".cache" / "huggingface" / "hub"
CMU_ARCTIC_CACHE = HF_HUB_DIR / "datasets--MikhailT--cmu-arctic"
ASCEND_CACHE = HF_HUB_DIR / "datasets--CAiRE--ASCEND"

HELDOUT_IDS: tuple[str, ...] = tuple(f"slt_{i:04d}" for i in range(30, 60))
HELDOUT_SET: frozenset[str] = frozenset(HELDOUT_IDS)


def ensure_cmudict(path: Path = CMUDICT_PATH) -> dict[str, Any]:
    """Verify `cmudict.dict` exists locally, downloading plain-text CMUdict if missing."""
    if path.is_file() and path.stat().st_size > 1_000_000:
        n_lines = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
        return {"path": str(path), "downloaded": False, "size_bytes": path.stat().st_size, "n_lines": n_lines}
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(CMUDICT_URL, path)
    n_lines = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
    return {"path": str(path), "downloaded": True, "size_bytes": path.stat().st_size, "n_lines": n_lines}


def _find_slt_parquet() -> Path:
    matches = sorted(CMU_ARCTIC_CACHE.rglob("slt-*.parquet"))
    if not matches:
        raise FileNotFoundError(f"Could not find slt-*.parquet under {CMU_ARCTIC_CACHE}")
    return matches[0]


def _find_ascend_parquet() -> Path:
    matches = sorted(ASCEND_CACHE.rglob("test-*.parquet"))
    if not matches:
        matches = sorted(ASCEND_CACHE.rglob("validation-*.parquet"))
    if not matches:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download("CAiRE/ASCEND", "main/test-00000-of-00001.parquet", repo_type="dataset"))
    return matches[0]


def _decode_audio_field_to_24k(audio_obj: Any) -> tuple[np.ndarray, int]:
    """Decode HF parquet audio dict (`bytes` or `array`+`sampling_rate`) to 24 kHz mono."""
    if isinstance(audio_obj, dict):
        if audio_obj.get("bytes"):
            return decode_audio_bytes_to_24k(audio_obj["bytes"])
        if audio_obj.get("array") is not None and audio_obj.get("sampling_rate"):
            arr = np.asarray(audio_obj["array"], dtype=np.float64).ravel()
            src_sr = int(audio_obj["sampling_rate"])
            if src_sr != SR:
                g = math.gcd(SR, src_sr)
                arr = resample_poly(arr, SR // g, src_sr // g)
            return arr, SR
    raise ValueError(f"Unsupported HF audio entry: {type(audio_obj)}")


def _active_rms_db(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).ravel()
    pk = float(np.max(np.abs(arr))) if arr.size else 0.0
    if pk <= 1e-9:
        return -120.0
    act = arr[np.abs(arr) > 0.03 * pk]
    rms = float(np.sqrt(np.mean(act * act))) if act.size else float(np.sqrt(np.mean(arr * arr)))
    return 20.0 * math.log10(max(1e-12, rms))


def extract_slt_full(
    out_dir: Path = SLT_FULL_DIR,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Extract all 1,132 CMU-ARCTIC `slt` utterances as 24 kHz mono PCM16 WAVs."""
    import pyarrow.parquet as pq

    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    pq_path = _find_slt_parquet()
    table = pq.read_table(pq_path)
    rows = table.to_pylist()

    utterances: list[dict[str, Any]] = []
    n_failed = 0
    total_dur_s = 0.0

    for idx, row in enumerate(rows):
        uid = f"slt_{idx:04d}"
        text = str(row.get("text", "")).strip()
        arctic_file = str(row.get("file", f"slt-{idx:04d}.wav"))
        wav_path = wav_dir / f"{uid}.wav"
        split = "heldout" if uid in HELDOUT_SET else "train"

        if force or not wav_path.is_file():
            x24, sr = _decode_audio_field_to_24k(row["audio"])
            assert sr == SR
            _write_pcm16_wav(wav_path, x24, sr=SR)
        else:
            with wave.open(str(wav_path), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            x24 = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0

        san = check_wav_sanity(x24, SR, expected_sr=SR, min_duration_s=0.25)
        if not san.get("ok", False):
            n_failed += 1
        dur_s = float(len(x24) / float(SR))
        total_dur_s += dur_s

        utterances.append({
            "id": uid,
            "speaker": "slt",
            "gender": "female",
            "accent": "US",
            "lang": "en",
            "split": split,
            "arctic_file": arctic_file,
            "text": text,
            "path": str(wav_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sr": SR,
            "duration_s": round(dur_s, 4),
            "sanity_ok": bool(san.get("ok", False)),
            "rms_dbfs": san.get("rms_dbfs"),
        })

    train_ids = [u["id"] for u in utterances if u["split"] == "train"]
    heldout_ids = [u["id"] for u in utterances if u["split"] == "heldout"]
    assert len(heldout_ids) == 30, f"Expected 30 heldout utterances, got {len(heldout_ids)}"
    assert not (set(train_ids) & set(heldout_ids)), "Zero-leakage split violated!"

    manifest = {
        "schema": 1,
        "corpus": "CMU-ARCTIC slt (full studio female)",
        "source": "hf:MikhailT/cmu-arctic",
        "sr": SR,
        "format": "PCM16 mono",
        "speaker": "slt",
        "total_utterances": len(utterances),
        "n_train": len(train_ids),
        "n_heldout": len(heldout_ids),
        "total_duration_s": round(total_dur_s, 2),
        "n_sanity_failed": n_failed,
        "heldout_ids": heldout_ids,
        "train_ids": train_ids,
        "utterances": utterances,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest_path": str(manifest_path),
        "total_utterances": len(utterances),
        "n_train": len(train_ids),
        "n_heldout": len(heldout_ids),
        "total_duration_s": round(total_dur_s, 2),
        "n_sanity_failed": n_failed,
    }


def build_bilingual_reference(
    out_dir: Path = BILINGUAL_REF_DIR,
    stats_path: Path = BILINGUAL_REF_STATS_PATH,
    *,
    speaker_id: int = 17,
    max_lang_clips: int = 80,
    max_mixed_clips: int = 60,
    n_export_refs: int = 8,
) -> dict[str, Any]:
    """Extract clean bilingual reference WAVs + measured ZH<->EN stats from `CAiRE/ASCEND`."""
    import pyarrow.parquet as pq

    pq_path = _find_ascend_parquet()
    rows = pq.read_table(pq_path).to_pylist()

    freqs = rfftfreq(N_FFT, 1.0 / SR)
    m_shared = (freqs >= 80.0) & (freqs <= 7600.0)
    m_norm = (freqs >= 300.0) & (freqs <= 3000.0)

    lang_metrics: dict[str, dict[str, Any]] = {}
    for lang in ("zh", "en"):
        f0s: list[float] = []
        iqrs: list[float] = []
        hnrs: list[float] = []
        rmss: list[float] = []
        wavs: list[np.ndarray] = []
        envs: list[np.ndarray] = []

        for r in rows:
            if int(r["original_speaker_id"]) != speaker_id or r["language"] != lang:
                continue
            if not (1.2 <= float(r["duration"]) <= 8.0):
                continue
            x24, _ = _decode_audio_field_to_24k(r["audio"])
            san = check_wav_sanity(x24, SR, expected_sr=SR, min_duration_s=1.0)
            if not san.get("ok", False):
                continue
            m = compute_f0_hnr_metrics(x24, sr=SR)
            if not m or m.get("hnr_db_median") is None or m.get("f0_median_hz") is None:
                continue
            if float(m["hnr_db_median"]) < 9.5:
                continue
            fv9, _ = analyze_v9(x24, sr=SR, hop_ms=5.0, n_fft=N_FFT)
            vm = fv9.voiced & (fv9.f0_hz > 45.0)
            if int(np.sum(vm)) < 20:
                continue

            f0s.append(float(m["f0_median_hz"]))
            if m.get("f0_iqr_st") is not None:
                iqrs.append(float(m["f0_iqr_st"]))
            hnrs.append(float(m["hnr_db_median"]))
            rmss.append(_active_rms_db(x24))
            wavs.append(x24)
            v_mean = np.mean(fv9.log_env[vm], axis=0)
            envs.append(v_mean - float(np.mean(v_mean[m_norm])))
            if len(wavs) >= max_lang_clips:
                break

        lang_metrics[lang] = {
            "n_clips": len(wavs),
            "f0_median_hz": float(np.median(f0s)),
            "f0_iqr_st_median": float(np.median(iqrs)) if iqrs else 0.0,
            "hnr_median_db": float(np.median(hnrs)),
            "rms_median_dbfs": float(np.median(rmss)),
            "syllable_rate_hz": float(_estimate_syllable_rate_hz(wavs, sr=SR)),
            "ltas_norm": np.mean(np.stack(envs, axis=0), axis=0),
        }

    # Measure code-switching pause durations and rank cleanest mixed utterances for audition export
    sw_pauses_ms: list[float] = []
    mixed_cands: list[tuple[float, dict[str, Any], np.ndarray]] = []
    n_mixed_measured = 0

    for r in rows:
        if int(r["original_speaker_id"]) != speaker_id or r["language"] != "mixed":
            continue
        if not (1.5 <= float(r["duration"]) <= 7.0):
            continue
        txt = str(r["transcription"]).strip()
        if not (re.search(r"[\u4e00-\u9fff]", txt) and re.search(r"[A-Za-z]{2,}", txt)):
            continue
        if any(tok in txt.lower() for tok in ("[", "uh", "um", "呃", "嗯")):
            continue

        x24, _ = _decode_audio_field_to_24k(r["audio"])
        san = check_wav_sanity(x24, SR, expected_sr=SR, min_duration_s=1.2)
        if not san.get("ok", False):
            continue
        m = compute_f0_hnr_metrics(x24, sr=SR)
        if not m or m.get("hnr_db_median") is None or float(m["hnr_db_median"]) < 9.5:
            continue

        hnr_val = float(m["hnr_db_median"])
        mixed_cands.append((hnr_val, r, x24))

        if n_mixed_measured < max_mixed_clips:
            try:
                pl = _safe_plan_v13(txt)
            except Exception:
                continue
            units = [u for u in pl["units"] if u.get("lang") in ("zh", "en") and u.get("ipa_phones")]
            ipas: list[str] = []
            sw_idx: list[int] = []
            for ui, u in enumerate(units):
                if ui > 0 and u["lang"] != units[ui - 1]["lang"]:
                    sw_idx.append(len(ipas))
                ipas.extend(u["ipa_phones"])
            if not sw_idx:
                continue

            feats = v6_voc.analyze(x24, sr=SR)
            bnds, info = align_ipa_sequence_v10(ipas, feats, sr=SR)
            if not info.get("dp_ok", False) or len(bnds) != len(ipas):
                continue

            pwr = np.exp(2.0 * feats.log_env)
            tot_db = 10.0 * np.log10(np.maximum(np.mean(pwr, axis=1), 1e-14))
            pk = float(np.percentile(tot_db, 95))
            rel = tot_db - pk
            low = (~feats.voiced) & (rel < -20.0)

            for si in sw_idx:
                bf = bnds[si][0]
                lo = max(0, bf - 25)
                hi = min(len(rel), bf + 25)
                best_len = 0
                cur = 0
                for t in range(lo, hi):
                    if low[t]:
                        cur += 1
                    else:
                        if cur > 0 and t >= bf - 10 and (t - cur) <= bf + 10:
                            best_len = max(best_len, cur)
                        cur = 0
                if cur > 0 and hi >= bf - 10 and (hi - cur) <= bf + 10:
                    best_len = max(best_len, cur)
                sw_pauses_ms.append(best_len * 5.0)
            n_mixed_measured += 1

    # Export top clean 24 kHz bilingual WAVs + transcripts
    out_dir.mkdir(parents=True, exist_ok=True)
    mixed_cands.sort(key=lambda t: (-t[0], str(t[1]["id"])))
    exported: list[dict[str, Any]] = []
    for idx_e, (hnr_val, r, x24) in enumerate(mixed_cands[:n_export_refs]):
        rid = str(r["id"])
        fname = f"ascend_spk{speaker_id}_{rid}.wav"
        wpath = out_dir / fname
        _write_pcm16_wav(wpath, x24, sr=SR)
        exported.append({
            "ref_idx": idx_e,
            "ascend_id": rid,
            "speaker_id": speaker_id,
            "language": "mixed",
            "text": str(r["transcription"]).strip(),
            "duration_s": round(len(x24) / float(SR), 3),
            "hnr_db": round(hnr_val, 2),
            "wav_path": str(wpath.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sr": SR,
        })

    (out_dir / "manifest.json").write_text(
        json.dumps({"schema": 1, "source": "hf:CAiRE/ASCEND", "speaker_id": speaker_id, "clips": exported}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    zh_m = lang_metrics["zh"]
    en_m = lang_metrics["en"]
    f0_diff_st = 12.0 * math.log2(en_m["f0_median_hz"] / max(40.0, zh_m["f0_median_hz"]))
    ltas_rmse_db = float(np.sqrt(np.mean(((en_m["ltas_norm"] - zh_m["ltas_norm"])[m_shared] * NEPER_TO_DB) ** 2)))
    rms_diff_db = en_m["rms_median_dbfs"] - zh_m["rms_median_dbfs"]
    rate_ratio = en_m["syllable_rate_hz"] / max(0.5, zh_m["syllable_rate_hz"])
    sw_arr = np.asarray(sw_pauses_ms, dtype=np.float64) if sw_pauses_ms else np.zeros(1, dtype=np.float64)

    stats_doc = {
        "schema": 1,
        "source": "hf:CAiRE/ASCEND (main/test)",
        "cleanest_bilingual_speaker_id": speaker_id,
        "speaker_gender_hint": "female",
        "en_zh_f0_diff_st": round(f0_diff_st, 3),
        "en_zh_ltas_rmse_db": round(ltas_rmse_db, 3),
        "en_zh_rms_diff_db": round(rms_diff_db, 3),
        "en_zh_rate_ratio": round(rate_ratio, 4),
        "switch_pause_median_ms": round(float(np.median(sw_arr)), 2),
        "switch_pause_p25_ms": round(float(np.percentile(sw_arr, 25)), 2),
        "switch_pause_p75_ms": round(float(np.percentile(sw_arr, 75)), 2),
        "switch_zero_pause_frac": round(float(np.mean(sw_arr == 0.0)), 4),
        "n_switches_measured": int(len(sw_arr)),
        "n_mixed_utts_measured": n_mixed_measured,
        "zh_stats": {
            "n_clips": zh_m["n_clips"],
            "f0_median_hz": round(zh_m["f0_median_hz"], 2),
            "f0_iqr_st_median": round(zh_m["f0_iqr_st_median"], 2),
            "hnr_median_db": round(zh_m["hnr_median_db"], 2),
            "rms_median_dbfs": round(zh_m["rms_median_dbfs"], 2),
            "syllable_rate_hz": round(zh_m["syllable_rate_hz"], 3),
        },
        "en_stats": {
            "n_clips": en_m["n_clips"],
            "f0_median_hz": round(en_m["f0_median_hz"], 2),
            "f0_iqr_st_median": round(en_m["f0_iqr_st_median"], 2),
            "hnr_median_db": round(en_m["hnr_median_db"], 2),
            "rms_median_dbfs": round(en_m["rms_median_dbfs"], 2),
            "syllable_rate_hz": round(en_m["syllable_rate_hz"], 3),
        },
        "bilingual_ref_dir": str(out_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "bilingual_ref_clips": exported,
    }
    stats_path.write_text(json.dumps(stats_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats_doc


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 10 corpus & lexicon fetcher")
    parser.add_argument("--slt-only", action="store_true", help="Only verify cmudict and extract slt_full")
    parser.add_argument("--ascend-only", action="store_true", help="Only build CAiRE/ASCEND bilingual ref & stats")
    parser.add_argument("--force", action="store_true", help="Re-write existing WAVs")
    args = parser.parse_args()

    if not args.ascend_only:
        cmu = ensure_cmudict()
        print("[cmudict]", json.dumps(cmu))
        slt_info = extract_slt_full(force=args.force)
        print("[slt_full]", json.dumps(slt_info))
        if args.slt_only:
            return

    b_stats = build_bilingual_reference()
    print("[bilingual_ref]", json.dumps({k: v for k, v in b_stats.items() if k != "bilingual_ref_clips"}, ensure_ascii=False, indent=2))
    print(f"[bilingual_ref] Exported {len(b_stats['bilingual_ref_clips'])} clean 24 kHz bilingual WAVs to {BILINGUAL_REF_DIR}")


if __name__ == "__main__":
    main()
