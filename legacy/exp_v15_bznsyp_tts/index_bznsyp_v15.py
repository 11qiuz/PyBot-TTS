"""Gold `.interval` + `#1/#2/#3/#4` Parser & 16-Worker Lightweight BZNSYP Indexer (`exp_v15_bznsyp_tts`)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import time
from typing import Any
import wave

import numpy as np
from scipy.fft import dct, irfft, rfft, rfftfreq
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v7_pybot_tts import unit_bank as ub  # noqa: E402
from exp_v8_clarity.utau_concat import OBSTRUENT_ONSETS  # noqa: E402
from exp_v8_timbre_clone import timbre_cloner_v2 as tc2  # noqa: E402
from exp_v10_integrity.gain_calibration import parseval_frame_db  # noqa: E402
from exp_v11_align_gmm.gmm_hmm_aligner import _MEL_FB  # noqa: E402
from exp_v11_align_gmm.timbre_cloner_v5 import extract_timbre_v5  # noqa: E402
from exp_v12_natural_vocoder.v12_bank import _patch_unsmoothed_uv_env  # noqa: E402
from exp_v12_natural_vocoder.vocoder_v12 import (  # noqa: E402
    GlottalPulseBank,
    VocoderFeaturesV12,
    analyze_v12,
    build_glottal_pulse_bank,
)
from exp_v15_bznsyp_tts import (  # noqa: E402
    BZNSYP_DIR,
    BZNSYP_HELDOUT_20,
    HOP,
    INDEX_V15,
    N_FFT,
    SPEAKER_ASSETS_V15,
    SR,
)
from exp_v15_bznsyp_tts.fetch_bznsyp import _read_text_graceful  # noqa: E402
from pybot_tts_v2_recorded import engine  # noqa: E402

_HANZI_RE = re.compile(r"[\u3400-\u9fff]")
_PUNCT_KEEP = frozenset("，、。？！；：,?!;:")
_FREQS_CACHE = rfftfreq(N_FFT, 1.0 / SR)
_LF_MASK = (_FREQS_CACHE >= 80.0) & (_FREQS_CACHE <= 1000.0)
_HF_MASK = (_FREQS_CACHE >= 2000.0) & (_FREQS_CACHE <= 8000.0)


@dataclass(slots=True)
class V15BzToken:
    """Compact lightweight acoustic descriptor (~350 B) for 100k-syllable BZNSYP indexing."""

    uid: str
    syl_idx: int
    n_syls: int
    hanzi: str
    onset: str
    final: str
    tone: int
    prev_final: str
    next_onset: str
    prev_tone: int
    next_tone: int
    start_fr: int
    split_fr: int
    end_fr: int
    dur_ms: float
    onset_ms: float
    pause_after_ms: float
    prosodic_break: int
    punct_after: str
    syllable_pos: str
    f0_start_st: float
    f0_mid_st: float
    f0_end_st: float
    rel_db: float
    parseval_p75_db: float
    uv_hf_lf: float
    voiced_frac: float
    quality_penalty: float
    edge_start_mfcc16: np.ndarray  # float16 (16,)
    edge_end_mfcc16: np.ndarray    # float16 (16,)


V15BzToken.__module__ = "exp_v15_bznsyp_tts.index_bznsyp_v15"
if __name__ == "__main__":
    sys.modules.setdefault("exp_v15_bznsyp_tts.index_bznsyp_v15", sys.modules[__name__])


def _normalize_bz_pinyin(py: str) -> tuple[str, str, int] | None:
    p = py.strip().lower().replace("lve", "lüe").replace("nve", "nüe")
    if p.startswith("u:"):
        p = "ü" + p[2:]
    p = p.replace("v", "ü")
    try:
        o, f, t = engine.split_pinyin(p)
        if o in ("y", "w"):
            o = ""
        if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
            f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"
        return o, f, int(t)
    except Exception:
        return None


def _parse_prosody_text(ptext: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract clean text and per-Hanzi `{ch, break, punct}` entries from `#1..#4` markup."""
    entries: list[dict[str, Any]] = []
    clean_chars: list[str] = []
    i = 0
    n = len(ptext)
    while i < n:
        ch = ptext[i]
        if ch == "#" and i + 1 < n and ptext[i + 1].isdigit():
            if entries:
                entries[-1]["break"] = max(entries[-1]["break"], int(ptext[i + 1]))
            i += 2
            continue
        if _HANZI_RE.fullmatch(ch):
            entries.append({"ch": ch, "break": 0, "punct": ""})
            clean_chars.append(ch)
        elif ch in _PUNCT_KEEP:
            if entries:
                entries[-1]["punct"] += ch
            clean_chars.append(ch)
        i += 1
    return "".join(clean_chars), entries


def _parse_interval_file(iv_path: Path) -> list[dict[str, Any]] | None:
    """Parse Praat short TextGrid `.interval` file into syllable timing dicts."""
    lines = _read_text_graceful(iv_path).splitlines()
    # Locate `"IntervalTier"`, skip name, xmin, xmax, count -> start triples
    start_idx = None
    for idx, ln in enumerate(lines[:20]):
        if "IntervalTier" in ln:
            start_idx = idx + 5
            break
    if start_idx is None:
        return None

    raw_ivs: list[tuple[float, float, str]] = []
    k = start_idx
    while k + 2 < len(lines):
        try:
            t0 = float(lines[k].strip())
            t1 = float(lines[k + 1].strip())
        except ValueError:
            k += 1
            continue
        lbl = lines[k + 2].strip().strip('"').strip()
        raw_ivs.append((t0, t1, lbl))
        k += 3

    syls: list[dict[str, Any]] = []
    buf: list[tuple[float, float, str]] = []
    for t0, t1, lbl in raw_ivs:
        if lbl in ("sil", "sp", "sp1", ""):
            if lbl == "sp1" and syls and not buf:
                syls[-1]["pause_after_ms"] += max(0.0, (t1 - t0) * 1000.0)
            continue
        if lbl[-1].isdigit():
            if len(buf) == 0:
                s0, sp, s1 = t0, t0, t1
            elif len(buf) == 1:
                s0, sp, s1 = buf[0][0], t0, t1
            else:
                return None
            if s1 <= s0:
                return None
            syls.append({
                "start_s": float(s0),
                "split_s": float(sp),
                "end_s": float(s1),
                "pause_after_ms": 0.0,
            })
            buf = []
        else:
            buf.append((t0, t1, lbl))
    if buf:
        return None
    return syls


def parse_bznsyp_corpus(bznsyp_dir: Path = BZNSYP_DIR) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Parse all 10,000 BZNSYP utterances enforcing 1-to-1 text/pinyin/interval correspondence."""
    base = bznsyp_dir / "BZNSYP" if (bznsyp_dir / "BZNSYP").is_dir() else bznsyp_dir
    if not (base / "Wave").is_dir():
        wav_cand = next(bznsyp_dir.rglob("Wave"), None)
        if wav_cand is None:
            raise FileNotFoundError(f"Could not find BZNSYP Wave/ inside {bznsyp_dir}")
        base = wav_cand.parent

    txt_files = sorted((base / "ProsodyLabeling").glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No ProsodyLabeling/*.txt inside {base}")
    raw_lines = [ln for ln in _read_text_graceful(txt_files[0]).splitlines() if ln.strip()]

    records: list[dict[str, Any]] = []
    stats = {"total": 0, "kept": 0, "skip_len_or_pinyin": 0, "skip_interval": 0, "skip_missing_wav": 0}

    for idx in range(0, len(raw_lines) - 1, 2):
        stats["total"] += 1
        uid, ptext = raw_lines[idx].split("\t", 1)
        uid = uid.strip()
        pys = raw_lines[idx + 1].strip().split()
        clean_text, h_entries = _parse_prosody_text(ptext)

        if len(h_entries) != len(pys) or len(pys) == 0:
            stats["skip_len_or_pinyin"] += 1
            continue
        parsed_pys = [_normalize_bz_pinyin(py) for py in pys]
        if any(p is None for p in parsed_pys):
            stats["skip_len_or_pinyin"] += 1
            continue

        iv_p = base / "PhoneLabeling" / f"{uid}.interval"
        wav_p = base / "Wave" / f"{uid}.wav"
        if not wav_p.is_file():
            stats["skip_missing_wav"] += 1
            continue
        iv_syls = _parse_interval_file(iv_p) if iv_p.is_file() else None
        if iv_syls is None or len(iv_syls) != len(pys):
            stats["skip_interval"] += 1
            continue

        syls = []
        for si, (he, py, (o, f, t), iv) in enumerate(zip(h_entries, pys, parsed_pys, iv_syls)):  # type: ignore[arg-type]
            syls.append({
                "index": si,
                "hanzi": he["ch"],
                "pinyin": py.strip().lower(),
                "onset": o,
                "final": f,
                "tone": int(t),
                "break": int(he["break"]),
                "punct_after": he["punct"],
                "start_s": iv["start_s"],
                "split_s": iv["split_s"],
                "end_s": iv["end_s"],
                "pause_after_ms": round(iv["pause_after_ms"], 2),
            })

        records.append({
            "id": uid,
            "text": clean_text,
            "prosody_text": ptext.strip(),
            "wav_path": str(wav_p.relative_to(REPO_ROOT)).replace("\\", "/"),
            "syllables": syls,
        })
        stats["kept"] += 1

    return records, stats


def select_heldout_20_ids(records: list[dict[str, Any]]) -> tuple[str, ...]:
    """Deterministically pick 20 diverse held-out utterances (12 statements, 4 ?, 4 !/multi-clause)."""
    def _clean_durs(r: dict[str, Any]) -> bool:
        durs = [(s["end_s"] - s["start_s"]) * 1000.0 for s in r["syllables"]]
        return all(60.0 <= d <= 420.0 for d in durs)

    q_cands = [r["id"] for r in records if ("？" in r["text"] or "?" in r["text"]) and 8 <= len(r["syllables"]) <= 18 and _clean_durs(r)]
    ex_cands = [r["id"] for r in records if ("！" in r["text"] or "!" in r["text"]) and 8 <= len(r["syllables"]) <= 20 and _clean_durs(r)]
    mc_cands = [r["id"] for r in records if ("，" in r["text"] and "！" not in r["text"] and "？" not in r["text"]) and 11 <= len(r["syllables"]) <= 20 and _clean_durs(r)]
    st_cands = [r["id"] for r in records if ("？" not in r["text"] and "！" not in r["text"] and "，" not in r["text"]) and 10 <= len(r["syllables"]) <= 16 and _clean_durs(r)]

    def _pick_even(pool: list[str], k: int, excl: set[str]) -> list[str]:
        filt = [x for x in pool if x not in excl]
        if len(filt) < k:
            return filt
        idxs = np.linspace(len(filt) // 10, len(filt) - 1 - len(filt) // 10, k).astype(int)
        return [filt[i] for i in idxs]

    picked: list[str] = []
    used: set[str] = set()
    for pool, cnt in ((q_cands, 4), (ex_cands, 2), (mc_cands, 2), (st_cands, 12)):
        got = _pick_even(pool, cnt, used)
        picked.extend(got)
        used.update(got)

    if len(picked) < 20:
        extra = _pick_even(mc_cands + st_cands, 20 - len(picked), used)
        picked.extend(extra)
    return tuple(picked[:20])


def read_bznsyp_wav_24k(wav_path: Path) -> np.ndarray:
    """Read 48 kHz mono PCM16 WAV cleanly and 2:1 downsample to 24 kHz float64."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    else:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1)
    if sr == 48000:
        return resample_poly(arr, 1, 2)
    if sr != SR:
        g = math.gcd(SR, sr)
        return resample_poly(arr, SR // g, sr // g)
    return arr


def _extract_f0_padded_fft(
    xp: np.ndarray,
    n_frames: int,
    pad_half: int,
    sr: int = SR,
    hop: int = 120,
    win_len: int = 960,
    fmin: float = 70.0,
    fmax: float = 420.0,
    min_acf: float = 0.54,
    min_rms: float = 0.008,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact FFT-accelerated normalized-ACF pitch tracker matching `vocoder._extract_f0_padded`."""
    idx = np.arange(win_len)[None, :] + np.arange(n_frames)[:, None] * hop
    frames = xp[idx]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    peak_rms = float(np.max(rms)) if rms.size else 0.0
    rms_floor = max(min_rms, 0.06 * peak_rms)

    lag_min = max(2, int(sr / fmax))
    lag_max = min(win_len // 2, int(sr / fmin))

    z = frames - np.mean(frames, axis=1, keepdims=True)
    sq = z * z
    csum = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(sq, axis=1)], axis=1)
    e0 = csum[:, win_len]

    lags = np.arange(lag_min - 1, lag_max + 2)
    n_fft_acf = 1 << (2 * win_len - 1).bit_length()
    z_spec = rfft(z, n=n_fft_acf, axis=1)
    acf_num = irfft(z_spec.real * z_spec.real + z_spec.imag * z_spec.imag, n=n_fft_acf, axis=1)[:, lags]
    den = np.sqrt(np.maximum(csum[:, win_len - lags] * (csum[:, win_len : win_len + 1] - csum[:, lags]), 1e-24))
    acf_mat = acf_num / den

    f0_raw = np.zeros(n_frames, dtype=np.float64)
    voiced = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        if rms[i] < rms_floor or e0[i] <= 1e-12:
            continue
        row = acf_mat[i]
        is_peak = (row[1:-1] >= row[:-2]) & (row[1:-1] >= row[2:])
        p_idxs = np.where(is_peak)[0] + 1
        if p_idxs.size == 0:
            continue
        top = float(np.max(row[p_idxs]))
        if top < min_acf:
            continue
        thresh = max(min_acf, top - 0.06)
        valid = p_idxs[row[p_idxs] >= thresh]
        k_best = int(valid[0])
        y0, y1, y2 = row[k_best - 1], row[k_best], row[k_best + 1]
        denom = 2.0 * (y0 - 2.0 * y1 + y2)
        delta = max(-0.5, min(0.5, (y0 - y2) / denom)) if abs(denom) > 1e-12 else 0.0
        f0_raw[i] = sr / (float(lags[k_best]) + delta)
        voiced[i] = True

    f0_smooth = f0_raw.copy()
    for i in range(1, n_frames - 1):
        if voiced[i - 1] and voiced[i] and voiced[i + 1]:
            f0_smooth[i] = float(np.median([f0_raw[i - 1], f0_raw[i], f0_raw[i + 1]]))
    for i in range(1, n_frames - 1):
        if voiced[i] and not voiced[i - 1] and not voiced[i + 1]:
            voiced[i] = False
            f0_smooth[i] = 0.0
    return f0_smooth, voiced


def _fast_mfcc_and_f0_24k(x24: np.ndarray) -> dict[str, Any]:
    """Fast vectorized 5 ms MFCC-16, HF/LF, Parseval dB, and FFT-NACF F0 extraction."""
    n = len(x24)
    n_fr = 1 + max(0, n - 1) // HOP
    pad_half = 480
    xp = np.pad(x24, (pad_half, pad_half))
    f0_hz, voiced = _extract_f0_padded_fft(xp, n_fr, pad_half, sr=SR, hop=HOP, win_len=2 * pad_half)

    win_len = 480
    half = win_len // 2
    x_pad_s = np.pad(x24, (half, half + win_len))
    starts = np.arange(n_fr) * HOP
    idx = starts[:, None] + np.arange(win_len)[None, :]
    win = np.hanning(win_len)
    win = win / math.sqrt(float(np.sum(win * win)))
    frames = x_pad_s[idx]
    frames = (frames - np.mean(frames, axis=1, keepdims=True)) * win[None, :]

    spec = np.abs(rfft(frames, n=N_FFT, axis=1))
    pwr = spec * spec
    log_env = 0.5 * np.log(np.maximum(pwr, 1e-18))

    mel_e = np.maximum(pwr @ _MEL_FB.T, 1e-12)
    mfcc16 = dct(np.log(mel_e), type=2, axis=1, norm="ortho")[:, :16]

    lf_db = 10.0 * np.log10(np.maximum(np.mean(pwr[:, _LF_MASK], axis=1), 1e-16))
    hf_db = 10.0 * np.log10(np.maximum(np.mean(pwr[:, _HF_MASK], axis=1), 1e-16))
    hf_lf = hf_db - lf_db
    p_db = parseval_frame_db(log_env, n_fft=N_FFT)

    return {
        "f0_hz": f0_hz,
        "voiced": voiced,
        "mfcc16": mfcc16,
        "hf_lf": hf_lf,
        "p_db": p_db,
        "n_fr": n_fr,
    }


def _index_one_bznsyp_utt(rec: dict[str, Any]) -> tuple[str, dict[str, Any], list[V15BzToken], int]:
    """Worker-safe feature extraction returning utterance summary + syllable tokens."""
    uid = rec["id"]
    wav_p = REPO_ROOT / rec["wav_path"]
    x24 = read_bznsyp_wav_24k(wav_p)
    ac = _fast_mfcc_and_f0_24k(x24)
    n_fr = ac["n_fr"]
    f0_hz = ac["f0_hz"]
    voiced = ac["voiced"]
    mfcc16 = ac["mfcc16"]
    hf_lf = ac["hf_lf"]
    p_db = ac["p_db"]

    syls = rec["syllables"]
    n_syls = len(syls)
    # Speech active mask
    s_first = max(0, int(round(syls[0]["start_s"] * SR / HOP)))
    s_last = min(n_fr, int(round(syls[-1]["end_s"] * SR / HOP)))
    act_v = voiced[s_first:s_last] & (f0_hz[s_first:s_last] > 50.0)
    utt_f0_vals = f0_hz[s_first:s_last][act_v]
    utt_f0_med = float(np.median(utt_f0_vals)) if utt_f0_vals.size else 220.0
    utt_db_vals = p_db[s_first:s_last][voiced[s_first:s_last]]
    utt_db_med = float(np.median(utt_db_vals)) if utt_db_vals.size else -20.0

    # Utterance-normalized MFCC (remove utterance-mean shape drift, center c0 by utt_db)
    mfcc_n = mfcc16.copy()
    v_all = np.zeros(n_fr, dtype=bool)
    v_all[s_first:s_last] = voiced[s_first:s_last]
    if np.any(v_all):
        mfcc_n[:, 0] -= float(np.median(mfcc_n[v_all, 0]))

    tokens: list[V15BzToken] = []
    dropped = 0
    for i, s in enumerate(syls):
        f0_fr = max(0, min(n_fr - 2, int(round(s["start_s"] * SR / HOP))))
        f_sp_fr = max(f0_fr, min(n_fr - 1, int(round(s["split_s"] * SR / HOP))))
        f1_fr = max(f0_fr + 2, min(n_fr, int(round(s["end_s"] * SR / HOP))))
        dur_ms = (f1_fr - f0_fr) * HOP * 1000.0 / SR
        if dur_ms < 40.0 or dur_ms > 480.0:
            dropped += 1
            continue

        o = s["onset"]
        f = s["final"]
        f_sp_rel = max(0, min(f1_fr - f0_fr - 2, f_sp_fr - f0_fr)) if o else 0
        voi_sl = voiced[f0_fr:f1_fr].copy()
        if o in OBSTRUENT_ONSETS and f_sp_rel > 0:
            for kr in range(f_sp_rel):
                if hf_lf[f0_fr + kr] > -4.0 or not voi_sl[kr]:
                    voi_sl[kr] = False

        rime_voi = voi_sl[f_sp_rel:] if f_sp_rel < len(voi_sl) else voi_sl
        v_frac = float(np.mean(voi_sl)) if len(voi_sl) else 0.0
        rime_vf = float(np.mean(rime_voi)) if len(rime_voi) else 0.0
        p75_db = float(np.percentile(p_db[f0_fr:f1_fr], 75))
        rel_db = float(np.clip(p75_db - utt_db_med, -18.0, 8.0))

        if rime_vf < 0.20 or float(np.percentile(p_db[f0_fr:f1_fr], 80)) < -40.0:
            dropped += 1
            continue

        # Soft quality penalty
        q_pen = 0.0
        if dur_ms < 65.0:
            q_pen += 0.05 * (65.0 - dur_ms)
        elif dur_ms > 350.0:
            q_pen += 0.03 * (dur_ms - 350.0)
        if rime_vf < 0.55:
            q_pen += 1.0 * (0.55 - rime_vf)

        r_idx = np.where(voi_sl[f_sp_rel:] & (f0_hz[f0_fr + f_sp_rel : f1_fr] > 50.0))[0] + f0_fr + f_sp_rel
        if r_idx.size < 2:
            r_idx = np.where(voi_sl & (f0_hz[f0_fr:f1_fr] > 50.0))[0] + f0_fr
        if r_idx.size >= 2:
            sts = np.clip(12.0 * np.log2(f0_hz[r_idx] / utt_f0_med), -10.0, 10.0)
            q3 = np.interp([0.2, 0.5, 0.8], np.linspace(0.0, 1.0, len(sts)), sts)
            f0_s, f0_m, f0_e = float(q3[0]), float(q3[1]), float(q3[2])
        elif r_idx.size == 1:
            st0 = float(np.clip(12.0 * np.log2(f0_hz[r_idx[0]] / utt_f0_med), -10.0, 10.0))
            f0_s = f0_m = f0_e = st0
        else:
            f0_s = f0_m = f0_e = 0.0

        uv_hflf = float(np.mean(hf_lf[f0_fr : f0_fr + f_sp_rel])) if f_sp_rel > 0 else 0.0
        es = np.mean(mfcc_n[f0_fr : min(f1_fr, f0_fr + 2)], axis=0).astype(np.float16)
        ee = np.mean(mfcc_n[max(f0_fr, f1_fr - 2) : f1_fr], axis=0).astype(np.float16)

        prev_f = syls[i - 1]["final"] if i > 0 else "SIL"
        prev_t = int(syls[i - 1]["tone"]) if i > 0 else 0
        next_o = syls[i + 1]["onset"] if i + 1 < n_syls else "SIL"
        next_t = int(syls[i + 1]["tone"]) if i + 1 < n_syls else 0
        syl_pos = "initial" if i == 0 else ("final" if i == n_syls - 1 else "medial")

        tokens.append(
            V15BzToken(
                uid=uid,
                syl_idx=i,
                n_syls=n_syls,
                hanzi=s["hanzi"],
                onset=o,
                final=f,
                tone=int(s["tone"]),
                prev_final=prev_f,
                next_onset=next_o,
                prev_tone=prev_t,
                next_tone=next_t,
                start_fr=f0_fr,
                split_fr=f_sp_rel,
                end_fr=f1_fr,
                dur_ms=round(dur_ms, 2),
                onset_ms=round(f_sp_rel * HOP * 1000.0 / SR, 2),
                pause_after_ms=float(s["pause_after_ms"]),
                prosodic_break=int(s["break"]),
                punct_after=str(s["punct_after"]),
                syllable_pos=syl_pos,
                f0_start_st=round(f0_s, 3),
                f0_mid_st=round(f0_m, 3),
                f0_end_st=round(f0_e, 3),
                rel_db=round(rel_db, 3),
                parseval_p75_db=round(p75_db, 3),
                uv_hf_lf=round(uv_hflf, 3),
                voiced_frac=round(v_frac, 4),
                quality_penalty=round(q_pen, 4),
                edge_start_mfcc16=es,
                edge_end_mfcc16=ee,
            )
        )

    utt_meta = {
        "id": uid,
        "text": rec["text"],
        "prosody_text": rec["prosody_text"],
        "wav_path": rec["wav_path"],
        "utt_f0_med_hz": round(utt_f0_med, 2),
        "utt_db_med": round(utt_db_med, 2),
        "n_frames": n_fr,
        "syllables": syls,
    }
    return uid, utt_meta, tokens, dropped


def _build_speaker_assets_v15(
    clean_train_metas: list[dict[str, Any]],
    *,
    max_profile_utts: int = 24,
    max_fallback_utts: int = 48,
) -> dict[str, Any]:
    """Build BZNSYP GlottalPulseBank, TimbreV5/V2 profiles, and UnitBank fallback strictly from train utts."""
    items: list[tuple[np.ndarray, vocoder.VocoderFeatures]] = []
    fv12_list: list[VocoderFeaturesV12] = []
    ub_utts: list[dict[str, Any]] = []
    ub_pre: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures]] = {}

    for meta in clean_train_metas[:max_fallback_utts]:
        uid = meta["id"]
        x24 = read_bznsyp_wav_24k(REPO_ROOT / meta["wav_path"])
        if len(items) < max_profile_utts:
            fv12 = analyze_v12(x24, sr=SR)
            feats = _patch_unsmoothed_uv_env(x24, fv12.feats, sr=SR, n_fft=N_FFT)
            fv12_list.append(
                VocoderFeaturesV12(
                    feats=feats,
                    bap_gt8k=fv12.bap_gt8k,
                    lp_pulses=fv12.lp_pulses,
                    coherent_phasor=fv12.coherent_phasor,
                )
            )
        else:
            feats = vocoder.analyze(x24, sr=SR)
        items.append((x24, feats))
        ub_pre[uid] = (x24, feats)
        ub_utts.append({
            "id": uid,
            "speaker": "BZNSYP",
            "active_start": 0,
            "active_stop": len(x24),
            "syllables": [
                {
                    "index": int(s["index"]),
                    "hanzi": s["hanzi"],
                    "pinyin": s["pinyin"],
                    "start_sample": int(round(s["start_s"] * SR)),
                    "stop_sample": int(round(s["end_s"] * SR)),
                    "ms": round((s["end_s"] - s["start_s"]) * 1000.0, 2),
                }
                for s in meta["syllables"]
            ],
        })

    prof_v2 = tc2.extract_v2_profile(items[:max_profile_utts], sr=SR, name="BZNSYP")
    prof_v5 = extract_timbre_v5(items[:max_profile_utts], sr=SR, name="BZNSYP", corpus="csmsc")
    gbank = build_glottal_pulse_bank("BZNSYP", fv12_list, max_pulses=32)
    fallback_ub = ub.build_unit_bank(ub_utts, exclude_ids=(), precomputed=ub_pre, sr=SR)
    return {
        "profile_v2": prof_v2,
        "profile_v5": prof_v5,
        "glottal_bank": gbank,
        "fallback_unit_bank": fallback_ub,
        "asset_train_ids": [m["id"] for m in clean_train_metas[:max_fallback_utts]],
    }


def build_bznsyp_index_v15(
    *,
    bznsyp_dir: Path = BZNSYP_DIR,
    index_path: Path = INDEX_V15,
    assets_path: Path = SPEAKER_ASSETS_V15,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Parse 10k BZNSYP, hold out 20 test utterances, and parallel-build compact acoustic + n-gram index."""
    t0 = time.perf_counter()
    records, parse_stats = parse_bznsyp_corpus(bznsyp_dir)
    print(f"Parsed BZNSYP: {parse_stats}", flush=True)
    if len(records) < 9000:
        raise RuntimeError(f"Expected >= 9000 valid BZNSYP utterances, got {len(records)}")

    held20 = BZNSYP_HELDOUT_20 if len(BZNSYP_HELDOUT_20) == 20 else select_heldout_20_ids(records)
    held_set = set(held20)
    print(f"Held-out 20 IDs: {list(held20)}", flush=True)

    workers = max_workers or min(16, max(4, (os.cpu_count() or 8) // 2))
    print(f"Extracting compact 5 ms descriptors across {len(records)} utts using {workers} processes...", flush=True)
    t_ext = time.perf_counter()

    tokens: list[V15BzToken] = []
    by_of: dict[tuple[str, str], list[int]] = {}
    by_oft: dict[tuple[str, str, int], list[int]] = {}
    by_utt_syl: dict[tuple[str, int], int] = {}
    bigram_of: dict[tuple[tuple[str, str], tuple[str, str]], list[tuple[str, int]]] = {}
    bigram_oft: dict[tuple[tuple[str, str, int], tuple[str, str, int]], list[tuple[str, int]]] = {}
    train_utt_metas: dict[str, dict[str, Any]] = {}
    held_utt_metas: dict[str, dict[str, Any]] = {}
    held_tokens_by_uid: dict[str, list[V15BzToken]] = {}
    dropped_tot = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done_i, (uid, u_meta, u_toks, drp) in enumerate(
            pool.map(_index_one_bznsyp_utt, records, chunksize=64), 1
        ):
            if uid in held_set:
                held_utt_metas[uid] = u_meta
                held_tokens_by_uid[uid] = u_toks
            else:
                dropped_tot += drp
                train_utt_metas[uid] = {
                    "id": u_meta["id"],
                    "text": u_meta["text"],
                    "prosody_text": u_meta["prosody_text"],
                    "wav_path": u_meta["wav_path"],
                    "utt_f0_med_hz": u_meta["utt_f0_med_hz"],
                    "utt_db_med": u_meta["utt_db_med"],
                    "n_frames": u_meta["n_frames"],
                    "n_syllables": len(u_meta["syllables"]),
                }
                tok_idx_by_syl: dict[int, int] = {}
                for tk in u_toks:
                    tid = len(tokens)
                    tokens.append(tk)
                    by_of.setdefault((tk.onset, tk.final), []).append(tid)
                    by_oft.setdefault((tk.onset, tk.final, tk.tone), []).append(tid)
                    by_utt_syl[(tk.uid, tk.syl_idx)] = tid
                    tok_idx_by_syl[tk.syl_idx] = tid
                # Build contiguous O(1) bigram indexes ONLY on non-paused consecutive syllables
                for tk in u_toks[:-1]:
                    if tk.pause_after_ms < 25.0 and (tk.syl_idx + 1) in tok_idx_by_syl:
                        nxt = tokens[tok_idx_by_syl[tk.syl_idx + 1]]
                        bigram_of.setdefault(((tk.onset, tk.final), (nxt.onset, nxt.final)), []).append((tk.uid, tk.syl_idx))
                        bigram_oft.setdefault(
                            ((tk.onset, tk.final, tk.tone), (nxt.onset, nxt.final, nxt.tone)), []
                        ).append((tk.uid, tk.syl_idx))
            if done_i % 2000 == 0 or done_i == len(records):
                print(f"  Indexed {done_i}/{len(records)} in {time.perf_counter() - t_ext:.1f}s (tokens={len(tokens)})", flush=True)

    # Pick 48 clean medium-duration training utterances for speaker assets (zero leakage!)
    clean_train = [
        m for m in train_utt_metas.values()
        if 8 <= len(m["syllables"]) <= 16 and all(70.0 <= (s["end_s"] - s["start_s"]) * 1000.0 <= 360.0 for s in m["syllables"])
    ]
    assets = _build_speaker_assets_v15(clean_train)
    assets_path.parent.mkdir(parents=True, exist_ok=True)
    assets_path.write_bytes(pickle.dumps(assets, protocol=pickle.HIGHEST_PROTOCOL))

    index_doc = {
        "version": 1,
        "heldout_20_ids": tuple(held20),
        "parse_stats": parse_stats,
        "n_train_utts": len(train_utt_metas),
        "n_train_tokens": len(tokens),
        "dropped_tokens": dropped_tot,
        "unique_of": len(by_of),
        "unique_oft": len(by_oft),
        "unique_bigram_of": len(bigram_of),
        "unique_bigram_oft": len(bigram_oft),
        "tokens": tokens,
        "by_of": by_of,
        "by_oft": by_oft,
        "by_utt_syl": by_utt_syl,
        "bigram_of": bigram_of,
        "bigram_oft": bigram_oft,
        "train_utt_metas": train_utt_metas,
        "held_utt_metas": held_utt_metas,
        "held_tokens_by_uid": held_tokens_by_uid,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    raw_idx = pickle.dumps(index_doc, protocol=pickle.HIGHEST_PROTOCOL)
    index_path.write_bytes(raw_idx)
    print(
        f"Saved BZNSYP v15 index: {index_path} ({len(raw_idx)/1e6:.2f} MB) | "
        f"train_utts={len(train_utt_metas)} tokens={len(tokens)} unique_oft={len(by_oft)} bigrams={len(bigram_of)} "
        f"elapsed={index_doc['elapsed_s']}s",
        flush=True,
    )
    return index_doc


def load_bznsyp_index_v15(index_path: Path = INDEX_V15) -> dict[str, Any]:
    raw = Path(index_path).read_bytes()

    class _CompatIndexUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if module == "__main__" and name == "V15BzToken":
                return V15BzToken
            return super().find_class(module, name)

    return _CompatIndexUnpickler(io.BytesIO(raw)).load()


def load_speaker_assets_v15(assets_path: Path = SPEAKER_ASSETS_V15) -> dict[str, Any]:
    return pickle.loads(Path(assets_path).read_bytes())


def main() -> int:
    ap = argparse.ArgumentParser(description="Build BZNSYP v15 lightweight multi-sample index.")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    doc = build_bznsyp_index_v15(max_workers=args.workers)
    size_mb = INDEX_V15.stat().st_size / 1e6
    ok = (doc["n_train_utts"] >= 9000 and doc["n_train_tokens"] >= 95000 and size_mb < 80.0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
