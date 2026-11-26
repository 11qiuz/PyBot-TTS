"""Stage 12 Zero-Leakage Extraction, 24 kHz Resampling & Joint ZH+EN Forced Alignment for SF_bilingual.

1. Extracts `SF_bilingual.zip` (Chunghwa Telecom single-female studio bilingual ZH+EN corpus,
   2,740 mixed utterances, 22.05 kHz -> resampled 24 kHz mono PCM16 into `crossling_data/sf_bilingual/`).
2. Parses exact Mandarin tonal pinyin + English ARPABET `<SEP>` sequences from
   `train_filelist.txt` / `eval_filelist.txt` + `text_SF.txt`.
3. Splits strictly zero-leakage:
   - HELDOUT (40 utts): 4 official `eval_filelist.txt` IDs (`com_SF_ce996..999`)
     + 36 held-out IDs (`com_SF_ce1000..1035`).
   - TRAIN (2,700 utts): all remaining utterances.
4. Extracts Stage 7 `TrueHarmEnv` (`analyze_true_harmonic_env(x24, sr=24000, bap_scale=0.50)`)
   in parallel, runs `_train_and_align_gmm(..., n_em=4)` jointly on TRAIN ZH+EN phone sequences,
   and aligns HELDOUT separately via `_align_with_trained_gmm`.
5. Saves:
   - `exp_20_pybot_larynx/crossling_data/sf_bilingual/manifest.json`
   - `exp_20_pybot_larynx/crossling_data/sf_bilingual_train_tokens.pkl`
   - `exp_20_pybot_larynx/crossling_data/sf_bilingual_heldout.pkl`
   - `exp_20_pybot_larynx/crossling_data/sf_bilingual_align_summary.json`
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import io
import json
import math
from pathlib import Path
import pickle
import re
import sys
import time
from typing import Any
import wave
import zipfile

import numpy as np
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import EXP_DIR, HOP, N_BAP, N_CEP, N_FFT, SR  # noqa: E402
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB  # noqa: E402
from exp_20_pybot_larynx.crossling.en_articulation import (  # noqa: E402
    ART_CLASSES,
    VOICELESS_OBSTRUENTS,
    VOWEL_IPAS,
    classify_articulatory_class,
    resolve_en_allophone,
)
from exp_20_pybot_larynx.crossling.en_frontend_v2 import EN_FUNCTION_WORDS  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import log_env_to_cep  # noqa: E402
from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v6_vocoder import vocoder as v6_voc  # noqa: E402
from exp_v8_clarity import en_utau_bank as v8_en  # noqa: E402
from exp_v10_g2p_and_align.landmark_aligner import score_token_purity  # noqa: E402
from exp_v11_align_gmm.gmm_hmm_aligner import (  # noqa: E402
    _align_with_trained_gmm,
    _train_and_align_gmm,
    _trim_utterance_active,
)

DEFAULT_SF_ZIP = Path(
    r"C:\Users\MECHREVO\.gemini\antigravity\brain\f67e0437-e98d-4cb1-bbfd-da72d02478db\browser_sandbox\downloads\SF_bilingual.zip"
)
CROSSLING_DATA_DIR = EXP_DIR / "crossling_data"
SF_DIR = CROSSLING_DATA_DIR / "sf_bilingual"
SF_WAV24_DIR = SF_DIR / "wav24k"
SF_MANIFEST_PATH = SF_DIR / "manifest.json"
SF_TRAIN_TOKENS_PATH = CROSSLING_DATA_DIR / "sf_bilingual_train_tokens.pkl"
SF_HELDOUT_PATH = CROSSLING_DATA_DIR / "sf_bilingual_heldout.pkl"
SF_ALIGN_SUMMARY_PATH = CROSSLING_DATA_DIR / "sf_bilingual_align_summary.json"

N_BINS = 5
_HELDOUT_IDS: frozenset[str] = frozenset(f"com_SF_ce{i}" for i in range(996, 1036))

_PUNCT_STRIP = "，。！？、；：,.!?;:「」『』（）()[]《》“”\"'—…·~"
_MAJOR_PAUSE_CHARS = frozenset("，,。！？；：.!?;:")
_MINOR_PAUSE_CHARS = frozenset("、")
_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_ARPA_RE = re.compile(r"^([A-Z]+)([012])?$")
_PY_RE = re.compile(r"^([a-z]+)([0-5])$")


def _worker_init() -> None:
    for _ev in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_ev] = "1"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _bin_segment(arr: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    t_len = len(arr)
    if t_len == 0:
        return np.zeros((n_bins, *arr.shape[1:]), dtype=np.float64)
    out = np.empty((n_bins, *arr.shape[1:]), dtype=np.float64)
    for bi in range(n_bins):
        lo = int(math.floor(bi * t_len / n_bins))
        hi = int(math.floor((bi + 1) * t_len / n_bins))
        if hi <= lo:
            lo = min(t_len - 1, bi * t_len // n_bins)
            hi = lo + 1
        out[bi] = np.mean(arr[lo:hi], axis=0)
    return out


def _clean_text_id(uid: str) -> str:
    return uid.replace("\ufeff", "").replace("com_DL_ce", "com_SF_ce").strip()


def _parse_trailing_pause(raw_tok: str) -> float:
    for ch in reversed(raw_tok):
        if ch not in _PUNCT_STRIP:
            break
        if ch in _MAJOR_PAUSE_CHARS:
            return 160.0
        if ch in _MINOR_PAUSE_CHARS:
            return 85.0
    return 0.0


def _count_en_syllables_arpa(arpas: list[str]) -> int:
    c = sum(1 for a in arpas if re.search(r"[012]$", a) or cf.ARPABET_TO_IPA.get(re.sub(r"[012]$", "", a), "") in VOWEL_IPAS)
    return max(1, c)


def _parse_bilingual_phones(seq_str: str, raw_text: str) -> tuple[list[dict[str, Any]], int, int]:
    """Parse SF_bilingual `left` sequence into unified bilingual phone dicts (`lang` in {'zh', 'en'})."""
    # Group sequence into linguistic items: EN words (between <SEP>/ZH/punct) & ZH syllables
    items: list[dict[str, Any]] = []
    cur_en: list[str] = []
    cur_en_pause = 0.0

    def _flush_en(pause_ms: float = 0.0) -> None:
        nonlocal cur_en, cur_en_pause
        if cur_en:
            items.append({"lang": "en", "arpas": cur_en, "pause_after_ms": max(cur_en_pause, pause_ms)})
            cur_en = []
            cur_en_pause = 0.0

    for raw_tok in seq_str.strip().split():
        if raw_tok == "<SEP>":
            _flush_en(0.0)
            continue
        p_ms = _parse_trailing_pause(raw_tok)
        tok = raw_tok.strip(_PUNCT_STRIP)
        if not tok:
            if p_ms > 0.0 and items and not cur_en:
                items[-1]["pause_after_ms"] = max(float(items[-1]["pause_after_ms"]), p_ms)
            _flush_en(p_ms)
            continue

        m_en = _ARPA_RE.fullmatch(tok)
        if m_en and m_en.group(1) in cf.ARPABET_TO_IPA:
            cur_en.append(tok)
            if p_ms > 0.0:
                _flush_en(p_ms)
            continue

        m_py = _PY_RE.fullmatch(tok)
        if m_py:
            _flush_en(0.0)
            stem, tone = m_py.group(1), m_py.group(2)
            py = stem + ("5" if tone == "0" else tone)
            items.append({"lang": "zh", "pinyin": py, "pause_after_ms": p_ms})
            continue

        raise ValueError(f"Unrecognized SF token {raw_tok!r} in {seq_str!r}")
    _flush_en(0.0)

    # Align EN surface word spellings from raw_text when counts match
    surface_en_words = _EN_WORD_RE.findall(raw_text)
    en_items = [it for it in items if it["lang"] == "en"]
    use_surface = len(surface_en_words) == len(en_items)
    if not use_surface:
        # Handle letter sequences written spaced like "N T H U" or "I D C" in text_SF.txt
        grouped_letters = re.findall(r"(?:[A-Za-z]\s+)+[A-Za-z]|[A-Za-z]+(?:['’][A-Za-z]+)?", raw_text)
        grouped_clean = [re.sub(r"\s+", "", g) for g in grouped_letters]
        if len(grouped_clean) == len(en_items):
            surface_en_words = grouped_clean
            use_surface = True

    en_cursor = 0
    zh_cursor = 0
    phones: list[dict[str, Any]] = []
    p_global_idx = 0

    for idx_it, it in enumerate(items):
        p_ms = float(it["pause_after_ms"])
        if idx_it == len(items) - 1:
            p_ms = 999.0

        if it["lang"] == "zh":
            py = it["pinyin"]
            onset, final, tone, syl_ipas = cf.pinyin_to_ipa(py)
            n_sp = len(syl_ipas)
            for k, ipa in enumerate(syl_ipas):
                phones.append({
                    "phone_idx": p_global_idx,
                    "lang": "zh",
                    "ipa": ipa,
                    "allophone": ipa,
                    "arpabet": None,
                    "stress": int(tone),
                    "word_stress": int(tone),
                    "is_func": False,
                    "is_focused": False,
                    "is_vowel": (k > 0 or not onset),
                    "word": py,
                    "word_idx": zh_cursor,
                    "phone_in_word_idx": k,
                    "word_n_phones": n_sp,
                    "word_n_syls": 1,
                    "is_word_initial": (k == 0),
                    "is_word_final": (k == n_sp - 1),
                    "is_phrase_final": (p_ms >= 45.0 and k == n_sp - 1),
                    "is_in_phrase_final_word": (p_ms >= 45.0),
                    "cluster_len": 1,
                    "pause_after_ms": p_ms if k == n_sp - 1 else 0.0,
                    "tone": int(tone),
                    "syllable_idx": zh_cursor,
                    "pinyin": py,
                })
                p_global_idx += 1
            zh_cursor += 1
        else:
            arpas: list[str] = it["arpas"]
            if use_surface:
                w_surf = surface_en_words[en_cursor]
            else:
                w_surf = "_".join(arpas)
            w_lower = w_surf.replace("’", "'").lower()
            is_func = (w_lower in EN_FUNCTION_WORDS) and (len(arpas) <= 5)
            n_syls = _count_en_syllables_arpa(arpas)
            stresses = []
            ipas = []
            for a in arpas:
                m_a = _ARPA_RE.fullmatch(a)
                assert m_a is not None
                base_a = m_a.group(1)
                st_str = m_a.group(2)
                st = int(st_str) if st_str is not None else 3
                ipa = cf.ARPABET_TO_IPA[base_a]
                if is_func and ipa in VOWEL_IPAS and st == 1 and p_ms < 45.0:
                    st = 0
                ipas.append(ipa)
                stresses.append(st)
            word_st = 1 if 1 in stresses else (2 if 2 in stresses else 0)
            n_ep = len(ipas)

            for k, (a, ipa, st) in enumerate(zip(arpas, ipas, stresses)):
                next_i = ipas[k + 1] if k + 1 < n_ep else None
                same_w_next = k + 1 < n_ep
                allo = resolve_en_allophone(ipa, next_i, same_w_next)
                is_v = ipa in VOWEL_IPAS
                cl = 1
                if not is_v:
                    cl_l = k
                    while cl_l > 0 and ipas[cl_l - 1] not in VOWEL_IPAS:
                        cl_l -= 1
                    cl_r = k
                    while cl_r + 1 < n_ep and ipas[cl_r + 1] not in VOWEL_IPAS:
                        cl_r += 1
                    cl = cl_r - cl_l + 1

                phones.append({
                    "phone_idx": p_global_idx,
                    "lang": "en",
                    "ipa": ipa,
                    "allophone": allo,
                    "arpabet": a,
                    "stress": int(st),
                    "word_stress": int(word_st),
                    "is_func": bool(is_func),
                    "is_focused": False,
                    "is_vowel": bool(is_v),
                    "word": w_surf,
                    "word_idx": 10000 + en_cursor,
                    "phone_in_word_idx": k,
                    "word_n_phones": n_ep,
                    "word_n_syls": int(n_syls),
                    "is_word_initial": (k == 0),
                    "is_word_final": (k == n_ep - 1),
                    "is_phrase_final": (p_ms >= 45.0 and k == n_ep - 1),
                    "is_in_phrase_final_word": (p_ms >= 45.0),
                    "cluster_len": int(cl),
                    "pause_after_ms": p_ms if k == n_ep - 1 else 0.0,
                })
                p_global_idx += 1
            en_cursor += 1

    return phones, zh_cursor, en_cursor


def _extract_sf_zip_and_manifest(sf_zip: Path) -> dict[str, Any]:
    SF_DIR.mkdir(parents=True, exist_ok=True)
    SF_WAV24_DIR.mkdir(parents=True, exist_ok=True)

    text_map: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(sf_zip, "r") as z:
        for ln in z.read("SF_bilingual/text_SF.txt").decode("utf-8-sig").splitlines():
            parts = ln.strip().split(None, 1)
            if len(parts) == 2:
                text_map[_clean_text_id(parts[0])] = parts[1].strip()

        seen: set[str] = set()
        for src_name, is_eval in (("SF_bilingual/eval_filelist.txt", True), ("SF_bilingual/train_filelist.txt", False)):
            for ln in z.read(src_name).decode("utf-8-sig").splitlines():
                if "|||||" not in ln:
                    continue
                left, right = ln.split("|||||", 1)
                wav_name = right.strip().split("|")[1].split("/")[-1]
                uid = wav_name[:-4] if wav_name.endswith(".wav") else wav_name
                if uid in seen:
                    continue
                seen.add(uid)
                wav24_path = SF_WAV24_DIR / f"{uid}.wav"
                if not wav24_path.is_file():
                    raw_bytes = z.read("SF_bilingual/wavs/" + wav_name)
                    with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
                        assert wf.getframerate() == 22050 and wf.getnchannels() == 1 and wf.getsampwidth() == 2
                        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float64)
                    y24 = np.clip(np.round(resample_poly(pcm, 160, 147)), -32768.0, 32767.0).astype(np.int16)
                    with wave.open(str(wav24_path), "wb") as wout:
                        wout.setnchannels(1)
                        wout.setsampwidth(2)
                        wout.setframerate(SR)
                        wout.writeframes(y24.tobytes())
                raw_txt = text_map.get(uid, "")
                records.append({
                    "id": uid,
                    "speaker": "sf_female",
                    "split": "heldout" if (is_eval or uid in _HELDOUT_IDS) else "train",
                    "wav_path": str(wav24_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "text": raw_txt,
                    "seq_str": left.strip(),
                })

    records.sort(key=lambda r: (0 if r["split"] == "train" else 1, r["id"]))
    train_ids = [r["id"] for r in records if r["split"] == "train"]
    heldout_ids = [r["id"] for r in records if r["split"] == "heldout"]
    assert len(heldout_ids) >= 40 and not (set(train_ids) & set(heldout_ids))

    manifest = {
        "corpus": "SF_bilingual_zh_en",
        "speaker": "sf_female",
        "sample_rate_hz": SR,
        "n_total": len(records),
        "n_train": len(train_ids),
        "n_heldout": len(heldout_ids),
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "utterances": records,
    }
    SF_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _worker_analyze_sf_utt(task: dict[str, Any]) -> dict[str, Any]:
    _worker_init()
    from exp_20_pybot_larynx.larynx.synth_v2 import analyze_true_harmonic_env

    uid = str(task["id"])
    wav_rel = str(task["wav_path"])
    raw_txt = str(task["text"])
    seq_str = str(task["seq_str"])

    x, file_sr = v6_voc.read_wav_mono(REPO_ROOT / wav_rel)
    if file_sr != SR:
        raise ValueError(f"Expected {SR} Hz, got {file_sr} for {uid}")
    s0, s1 = v8_en._trim_active_span(x, sr=SR)
    xa = x[s0:s1]

    log_env_full, f0_full, voiced_full, bap6_full = analyze_true_harmonic_env(
        xa, sr=SR, hop=HOP, n_fft=N_FFT, bap_scale=0.50
    )
    feats_pre = v6_voc.VocoderFeatures(
        f0_hz=f0_full,
        voiced=voiced_full,
        log_env=log_env_full,
        bap=bap6_full[:, :5],
        sr=SR,
        hop_samples=HOP,
        n_samples=len(xa),
        n_fft=N_FFT,
    )
    cs, ce, cf0, cf1 = _trim_utterance_active(xa, feats_pre, sr=SR, hop=HOP)
    xa_t = xa[cs:ce]
    log_env = log_env_full[cf0:cf1]
    f0_hz = f0_full[cf0:cf1]
    voiced = voiced_full[cf0:cf1]
    bap6 = bap6_full[cf0:cf1]

    feats = v6_voc.VocoderFeatures(
        f0_hz=f0_hz,
        voiced=voiced,
        log_env=log_env,
        bap=bap6[:, :5],
        sr=SR,
        hop_samples=HOP,
        n_samples=len(xa_t),
        n_fft=N_FFT,
    )

    phones, n_zh_syl, n_en_w = _parse_bilingual_phones(seq_str, raw_txt)
    ipas = [str(p["ipa"]) for p in phones]

    v_mask = voiced & (f0_hz > 50.0)
    if np.any(v_mask):
        utt_log_ref = float(np.median(np.mean(log_env[v_mask, 1:257], axis=1)))
        utt_f0_med = float(np.median(f0_hz[v_mask]))
        utt_voiced_db = float(np.mean(np.mean(log_env[v_mask], axis=1) * NEPER_TO_DB))
    else:
        utt_log_ref = 0.0
        utt_f0_med = 215.0
        utt_voiced_db = -30.0

    return {
        "id": uid,
        "speaker": "sf_female",
        "wav_path": wav_rel,
        "text": raw_txt,
        "seq_str": seq_str,
        "n_zh_syl": n_zh_syl,
        "n_en_words": n_en_w,
        "trim_s0": int(s0 + cs),
        "trim_s1": int(s0 + ce),
        "wav": xa_t.astype(np.float32),
        "feats": feats,
        "bap6": bap6.astype(np.float16),
        "utt_log_ref": utt_log_ref,
        "utt_f0_med": utt_f0_med,
        "utt_voiced_db": utt_voiced_db,
        "ipas": ipas,
        "plan_phones": phones,
    }


def _extract_sf_utt_tokens(
    rec: dict[str, Any],
    bounds: list[tuple[int, int, int]],
    *,
    include_full_frames: bool,
) -> dict[str, Any]:
    uid = rec["id"]
    wav = rec["wav"].astype(np.float64)
    feats: v6_voc.VocoderFeatures = rec["feats"]
    log_env = feats.log_env
    f0_hz = feats.f0_hz
    voiced = feats.voiced
    bap6 = rec["bap6"].astype(np.float64)
    n_fr = len(log_env)

    utt_log_ref = float(rec["utt_log_ref"])
    utt_f0_med = max(50.0, float(rec["utt_f0_med"]))
    utt_voiced_db = float(rec["utt_voiced_db"])

    log_env_norm = log_env - utt_log_ref
    cep64 = log_env_to_cep(log_env_norm, n_cep=N_CEP).astype(np.float16)
    frame_db = np.mean(log_env, axis=1) * NEPER_TO_DB
    frame_pwr_db = 10.0 * np.log10(np.mean(np.exp(2.0 * log_env), axis=1) + 1e-18)

    phones_plan = rec["plan_phones"]
    n_p = len(phones_plan)
    assert len(bounds) == n_p

    active_mask = np.zeros(n_fr, dtype=bool)
    zh_fr_mask = np.zeros(n_fr, dtype=bool)
    en_fr_mask = np.zeros(n_fr, dtype=bool)

    all_phones: list[dict[str, Any]] = []
    en_tokens: list[dict[str, Any]] = []
    n_en_strict = 0
    n_zh_strict = 0

    for i, (ph, (ps, pm, pe)) in enumerate(zip(phones_plan, bounds)):
        ps_i = max(0, min(n_fr - 1, int(ps)))
        pe_i = max(ps_i + 1, min(n_fr, int(pe)))
        pm_i = max(ps_i, min(pe_i - 1, int(pm)))
        active_mask[ps_i:pe_i] = True
        lang = str(ph["lang"])
        if lang == "zh":
            zh_fr_mask[ps_i:pe_i] = True
        else:
            en_fr_mask[ps_i:pe_i] = True

        ipa = str(ph["ipa"])
        allo = str(ph["allophone"])
        seg_w = wav[ps_i * HOP : min(len(wav), pe_i * HOP)]
        dur_ms = float((pe_i - ps_i) * HOP * 1000.0 / SR)
        purity_ok, reason, p_met = score_token_purity(
            ipa, log_env[ps_i:pe_i], voiced[ps_i:pe_i], seg_w, sr=SR, hop=HOP
        )
        if lang == "en" and purity_ok:
            n_en_strict += 1
        elif lang == "zh" and purity_ok:
            n_zh_strict += 1

        prev_ph = phones_plan[i - 1] if i > 0 else None
        next_ph = phones_plan[i + 1] if i + 1 < n_p else None
        prev_pause = float(prev_ph["pause_after_ms"]) if prev_ph is not None else 999.0
        cur_pause = float(ph["pause_after_ms"])
        prev_ipa = str(prev_ph["ipa"]) if (prev_ph is not None and prev_pause < 20.0) else "SIL"
        next_ipa = str(next_ph["ipa"]) if (next_ph is not None and cur_pause < 20.0) else "SIL"
        prev_cls = classify_articulatory_class(None if prev_ipa == "SIL" else prev_ipa)
        next_cls = classify_articulatory_class(None if next_ipa == "SIL" else next_ipa)
        pos_id = 0 if prev_ipa == "SIL" else (2 if next_ipa == "SIL" else 1)

        f0_seg = f0_hz[ps_i:pe_i]
        vm_seg = voiced[ps_i:pe_i] & (f0_seg > 45.0)
        if np.any(vm_seg):
            st_raw = 12.0 * np.log2(np.where(vm_seg, f0_seg, utt_f0_med) / utt_f0_med)
            if not np.all(vm_seg):
                idx_v = np.where(vm_seg)[0]
                st_raw = np.interp(np.arange(len(f0_seg)), idx_v, st_raw[idx_v])
            f0_st_5 = _bin_segment(st_raw, N_BINS).astype(np.float32)
            f0_med_st = float(np.median(12.0 * np.log2(f0_seg[vm_seg] / utt_f0_med)))
            f0_has_voiced = True
        else:
            f0_st_5 = np.zeros(N_BINS, dtype=np.float32)
            f0_med_st = 0.0
            f0_has_voiced = False

        rel_db5 = (_bin_segment(frame_db[ps_i:pe_i], N_BINS) - utt_voiced_db).astype(np.float32)
        gain_db = float(np.mean(rel_db5))

        row = {
            "utt_id": uid,
            "speaker": "sf_female",
            "lang": lang,
            "phone_idx": int(ph["phone_idx"]),
            "p_start": ps_i,
            "p_mid": pm_i,
            "p_end": pe_i,
            "dur_ms": round(dur_ms, 2),
            "ipa": ipa,
            "allophone": allo,
            "arpabet": ph["arpabet"],
            "stress": ph["stress"],
            "word_stress": int(ph["word_stress"]),
            "is_func": bool(ph["is_func"]),
            "is_focused": False,
            "is_vowel": bool(ph["is_vowel"]),
            "word": str(ph["word"]),
            "word_idx": int(ph["word_idx"]),
            "phone_in_word_idx": int(ph["phone_in_word_idx"]),
            "word_n_phones": int(ph["word_n_phones"]),
            "word_n_syls": int(ph.get("word_n_syls", 1)),
            "is_word_initial": bool(ph["is_word_initial"]),
            "is_word_final": bool(ph["is_word_final"]),
            "is_phrase_final": bool(ph["is_phrase_final"]),
            "is_utterance_final": (i == n_p - 1),
            "is_in_phrase_final_word": bool(ph["is_in_phrase_final_word"]),
            "cluster_len": int(ph["cluster_len"]),
            "pause_prev_ms": prev_pause,
            "pause_after_ms": float(ph["pause_after_ms"]),
            "prev_lang": str(prev_ph["lang"]) if (prev_ph is not None and prev_pause < 20.0) else "SIL",
            "next_lang": str(next_ph["lang"]) if (next_ph is not None and cur_pause < 20.0) else "SIL",
            "prev_ipa": prev_ipa,
            "next_ipa": next_ipa,
            "prev_cls": int(prev_cls),
            "next_cls": int(next_cls),
            "pos_id": int(pos_id),
            "is_question": str(rec["text"]).rstrip().endswith(("？", "?")),
            "voiced_frac": round(float(p_met["voiced_frac"]), 3),
            "rms_dbfs": float(p_met["rms_dbfs"]),
            "purity_ok": bool(purity_ok),
            "quarantine_reason": str(reason),
            "gain_db": round(gain_db, 3),
            "f0_st_5": f0_st_5,
            "f0_med_st": round(f0_med_st, 4),
            "f0_has_voiced": f0_has_voiced,
        }

        if lang == "en":
            same_w_prev = bool(
                prev_ph is not None
                and prev_ph["lang"] == "en"
                and prev_pause < 20.0
                and int(prev_ph["word_idx"]) == int(ph["word_idx"])
            )
            same_w_next = bool(
                next_ph is not None
                and next_ph["lang"] == "en"
                and cur_pause < 20.0
                and int(next_ph["word_idx"]) == int(ph["word_idx"])
            )
            row["same_word_prev"] = same_w_prev
            row["same_word_next"] = same_w_next
            row["pre_fortis"] = bool(ph["is_vowel"] and same_w_next and next_ipa in VOICELESS_OBSTRUENTS)
            row["cep_5"] = _bin_segment(cep64[ps_i:pe_i].astype(np.float64), N_BINS).astype(np.float16)
            row["bap_5"] = _bin_segment(bap6[ps_i:pe_i], N_BINS).astype(np.float16)
            row["vp_5"] = _bin_segment(voiced[ps_i:pe_i].astype(np.float64), N_BINS).astype(np.float16)
            row["rel_db_5"] = rel_db5
            en_tokens.append(row)
        all_phones.append(row)

    # Extract contiguous English runs & switch events (`ZH->EN`, `EN->ZH`)
    en_runs: list[dict[str, Any]] = []
    cur_run: list[dict[str, Any]] = []
    run_idx = 0
    for row in all_phones:
        if row["lang"] == "en":
            cur_run.append(row)
        else:
            if cur_run:
                run_id = f"{uid}__r{run_idx}"
                for r in cur_run:
                    r["run_id"] = run_id
                en_runs.append({
                    "run_id": run_id,
                    "utt_id": uid,
                    "start_fr": cur_run[0]["p_start"],
                    "end_fr": cur_run[-1]["p_end"],
                    "n_phones": len(cur_run),
                    "prev_lang": cur_run[0]["prev_lang"],
                    "next_lang": cur_run[-1]["next_lang"],
                })
                run_idx += 1
                cur_run = []
    if cur_run:
        run_id = f"{uid}__r{run_idx}"
        for r in cur_run:
            r["run_id"] = run_id
        en_runs.append({
            "run_id": run_id,
            "utt_id": uid,
            "start_fr": cur_run[0]["p_start"],
            "end_fr": cur_run[-1]["p_end"],
            "n_phones": len(cur_run),
            "prev_lang": cur_run[0]["prev_lang"],
            "next_lang": cur_run[-1]["next_lang"],
        })

    # Switch boundary acoustic stats across adjacent (ZH -> EN) and (EN -> ZH) phones
    switch_events: list[dict[str, Any]] = []
    for k in range(len(all_phones) - 1):
        a_ph = all_phones[k]
        b_ph = all_phones[k + 1]
        if a_ph["lang"] == b_ph["lang"]:
            continue
        direction = f"{a_ph['lang']}_to_{b_ph['lang']}"
        va = voiced[a_ph["p_start"] : a_ph["p_end"]] & (f0_hz[a_ph["p_start"] : a_ph["p_end"]] > 50.0)
        vb = voiced[b_ph["p_start"] : b_ph["p_end"]] & (f0_hz[b_ph["p_start"] : b_ph["p_end"]] > 50.0)
        # Measure acoustic dip between phones as a proxy for actual pause
        gap_frames = max(0, int(b_ph["p_start"]) - int(a_ph["p_end"]))
        lv_a = float(np.median(frame_pwr_db[a_ph["p_start"] : a_ph["p_end"]]))
        lv_b = float(np.median(frame_pwr_db[b_ph["p_start"] : b_ph["p_end"]]))
        level_diff_db = abs(lv_b - lv_a)
        f0_diff_st = None
        if np.any(va) and np.any(vb):
            fa = float(np.median(f0_hz[a_ph["p_start"] : a_ph["p_end"]][va]))
            fb = float(np.median(f0_hz[b_ph["p_start"] : b_ph["p_end"]][vb]))
            f0_diff_st = round(abs(12.0 * math.log2(fb / fa)), 3)
        sa = np.mean(log_env[a_ph["p_start"] : a_ph["p_end"]], axis=0)
        sb = np.mean(log_env[b_ph["p_start"] : b_ph["p_end"]], axis=0)
        sa = sa - float(np.mean(sa))
        sb = sb - float(np.mean(sb))
        spec_rmse_db = float(np.sqrt(np.mean(((sb - sa) * NEPER_TO_DB) ** 2)))
        switch_events.append({
            "utt_id": uid,
            "direction": direction,
            "punct_pause_ms": float(a_ph["pause_after_ms"]),
            "gap_frames": gap_frames,
            "level_diff_db": round(level_diff_db, 3),
            "f0_diff_st": f0_diff_st,
            "spec_rmse_db": round(spec_rmse_db, 3),
            "left_ipa": a_ph["ipa"],
            "right_ipa": b_ph["ipa"],
        })

    # Per-utterance paired ZH vs EN language summaries (strictly within the same utterance!)
    zh_vm = zh_fr_mask & voiced & (f0_hz > 50.0)
    en_vm = en_fr_mask & voiced & (f0_hz > 50.0)
    zh_um = zh_fr_mask & (~voiced) & (np.mean(log_env[:, 64:], axis=1) > -10.5)
    en_um = en_fr_mask & (~voiced) & (np.mean(log_env[:, 64:], axis=1) > -10.5)

    def _seg_f0_stats(mask: np.ndarray) -> tuple[float | None, float | None]:
        if int(np.sum(mask)) < 6:
            return None, None
        med = float(np.median(f0_hz[mask]))
        st = 12.0 * np.log2(f0_hz[mask] / med)
        return med, float(np.percentile(st, 75) - np.percentile(st, 25))

    zh_f0_med, zh_iqr = _seg_f0_stats(zh_vm)
    en_f0_med, en_iqr = _seg_f0_stats(en_vm)

    zh_rms = float(np.median(frame_pwr_db[zh_vm])) if int(np.sum(zh_vm)) >= 6 else None
    en_rms = float(np.median(frame_pwr_db[en_vm])) if int(np.sum(en_vm)) >= 6 else None

    sil_mask = ~active_mask
    if np.any(sil_mask):
        sil_cep = np.mean(cep64[sil_mask].astype(np.float64), axis=0)
        sil_bap = np.mean(bap6[sil_mask], axis=0)
        sil_cnt = int(np.sum(sil_mask))
    else:
        sil_cep = np.zeros(N_CEP, dtype=np.float64)
        sil_bap = np.ones(N_BAP, dtype=np.float64) * 0.95
        sil_cnt = 0

    utt_doc: dict[str, Any] = {
        "id": uid,
        "speaker": "sf_female",
        "wav_path": rec["wav_path"],
        "text": rec["text"],
        "seq_str": rec["seq_str"],
        "n_frames": n_fr,
        "trim_s0": int(rec["trim_s0"]),
        "trim_s1": int(rec["trim_s1"]),
        "utt_log_ref": utt_log_ref,
        "utt_f0_med": utt_f0_med,
        "utt_voiced_db": utt_voiced_db,
        "zh_f0_med_hz": zh_f0_med,
        "en_f0_med_hz": en_f0_med,
        "zh_f0_iqr_st": zh_iqr,
        "en_f0_iqr_st": en_iqr,
        "zh_voiced_pwr_db": zh_rms,
        "en_voiced_pwr_db": en_rms,
        "n_zh_v": int(np.sum(zh_vm)),
        "n_en_v": int(np.sum(en_vm)),
        "n_zh_u": int(np.sum(zh_um)),
        "n_en_u": int(np.sum(en_um)),
        "zh_v_ltas": np.mean(log_env[zh_vm], axis=0).astype(np.float16) if np.any(zh_vm) else None,
        "en_v_ltas": np.mean(log_env[en_vm], axis=0).astype(np.float16) if np.any(en_vm) else None,
        "zh_u_ltas": np.mean(log_env[zh_um], axis=0).astype(np.float16) if np.any(zh_um) else None,
        "en_u_ltas": np.mean(log_env[en_um], axis=0).astype(np.float16) if np.any(en_um) else None,
        "zh_bap6": np.median(bap6[zh_vm], axis=0).astype(np.float16) if np.any(zh_vm) else None,
        "en_bap6": np.median(bap6[en_vm], axis=0).astype(np.float16) if np.any(en_vm) else None,
        "en_runs": en_runs,
        "en_phones": en_tokens,
        "phones": all_phones,
        "sil_cep": sil_cep.astype(np.float32),
        "sil_bap": sil_bap.astype(np.float32),
        "sil_cnt": sil_cnt,
        "n_en_phones": len(en_tokens),
        "n_en_strict_ok": n_en_strict,
        "n_zh_phones": len(all_phones) - len(en_tokens),
        "n_zh_strict_ok": n_zh_strict,
    }
    if include_full_frames:
        utt_doc["wav"] = rec["wav"]
        utt_doc["log_env"] = log_env.astype(np.float16)
        utt_doc["bap6"] = rec["bap6"]
        utt_doc["f0_hz"] = f0_hz.astype(np.float32)
        utt_doc["voiced"] = voiced.astype(bool)
    return utt_doc, en_tokens, switch_events


def prepare_and_align_sf_bilingual(
    sf_zip: Path = DEFAULT_SF_ZIP,
    *,
    n_em: int = 4,
    workers: int = 12,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    print(f"[SF-Stage12] Extracting & 24 kHz resampling {sf_zip}...", flush=True)
    manifest = _extract_sf_zip_and_manifest(sf_zip)
    held_set = set(manifest["heldout_ids"])
    train_set = set(manifest["train_ids"])
    assert not (train_set & held_set) and len(held_set) >= 40

    train_tasks = [u for u in manifest["utterances"] if u["split"] == "train"]
    held_tasks = [u for u in manifest["utterances"] if u["split"] == "heldout"]
    print(
        f"[SF-Stage12] Extracting TrueHarmEnv (bap_scale=0.50): {len(train_tasks)} TRAIN, "
        f"{len(held_tasks)} HELDOUT (workers={workers})...",
        flush=True,
    )

    train_recs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        for i, rec in enumerate(ex.map(_worker_analyze_sf_utt, train_tasks, chunksize=16), start=1):
            train_recs.append(rec)
            if i % 500 == 0 or i == len(train_tasks):
                print(f"[SF-Stage12]   extracted {i}/{len(train_tasks)} TRAIN ({time.perf_counter() - t0:.1f}s)", flush=True)
        held_recs = list(ex.map(_worker_analyze_sf_utt, held_tasks, chunksize=8))

    t_feat = time.perf_counter()
    print(f"[SF-Stage12] Features extracted in {t_feat - t0:.1f}s. Training joint ZH+EN GMM-HMM (n_em={n_em})...", flush=True)
    gmm, train_bounds = _train_and_align_gmm(train_recs, n_em=n_em, sr=SR, hop=HOP)
    held_bounds = {
        rec["id"]: _align_with_trained_gmm(gmm, rec["wav"], rec["feats"], rec["ipas"], sr=SR, hop=HOP)
        for rec in held_recs
    }
    print(f"[SF-Stage12] GMM-HMM alignment finished in {time.perf_counter() - t_feat:.1f}s.", flush=True)

    train_utts: list[dict[str, Any]] = []
    flat_en_train: list[dict[str, Any]] = []
    all_switches_train: list[dict[str, Any]] = []
    sil_cep_sum = np.zeros(N_CEP, dtype=np.float64)
    sil_bap_sum = np.zeros(N_BAP, dtype=np.float64)
    sil_cnt_tot = 0
    en_ok_tr = 0
    en_tot_tr = 0
    zh_ok_tr = 0
    zh_tot_tr = 0
    reason_en: Counter[str] = Counter()

    for rec in train_recs:
        uid = rec["id"]
        assert uid not in held_set
        udoc, en_toks, sw = _extract_sf_utt_tokens(rec, train_bounds[uid], include_full_frames=False)
        train_utts.append(udoc)
        flat_en_train.extend(en_toks)
        all_switches_train.extend(sw)
        en_ok_tr += udoc["n_en_strict_ok"]
        en_tot_tr += udoc["n_en_phones"]
        zh_ok_tr += udoc["n_zh_strict_ok"]
        zh_tot_tr += udoc["n_zh_phones"]
        for tok in en_toks:
            reason_en[tok["quarantine_reason"]] += 1
        sc = int(udoc["sil_cnt"])
        if sc > 0:
            sil_cep_sum += sc * udoc["sil_cep"].astype(np.float64)
            sil_bap_sum += sc * udoc["sil_bap"].astype(np.float64)
            sil_cnt_tot += sc

    held_utts: list[dict[str, Any]] = []
    flat_en_held: list[dict[str, Any]] = []
    held_switches: list[dict[str, Any]] = []
    en_ok_ho = 0
    en_tot_ho = 0
    for rec in held_recs:
        uid = rec["id"]
        assert uid in held_set
        udoc, en_toks, sw = _extract_sf_utt_tokens(rec, held_bounds[uid], include_full_frames=True)
        held_utts.append(udoc)
        flat_en_held.extend(en_toks)
        held_switches.extend(sw)
        en_ok_ho += udoc["n_en_strict_ok"]
        en_tot_ho += udoc["n_en_phones"]

    en_acc_tr = en_ok_tr / max(1, en_tot_tr)
    zh_acc_tr = zh_ok_tr / max(1, zh_tot_tr)
    en_acc_ho = en_ok_ho / max(1, en_tot_ho)
    f0_med_global = float(np.median([u["utt_f0_med"] for u in train_utts]))
    zh_f0_vals = [u["zh_f0_med_hz"] for u in train_utts if u["zh_f0_med_hz"] is not None]
    en_f0_vals = [u["en_f0_med_hz"] for u in train_utts if u["en_f0_med_hz"] is not None]
    f0_med_zh = float(np.median(zh_f0_vals))
    f0_med_en = float(np.median(en_f0_vals))
    log_ref_med = float(np.median([u["utt_log_ref"] for u in train_utts]))
    voiced_db_med = float(np.median([u["utt_voiced_db"] for u in train_utts]))

    train_payload = {
        "schema": "sf_bilingual_train_v1",
        "corpus": "SF_bilingual_zh_en",
        "speaker": "sf_female",
        "sr": SR,
        "hop": HOP,
        "n_fft": N_FFT,
        "n_cep": N_CEP,
        "n_bap": N_BAP,
        "n_bins": N_BINS,
        "art_classes": ART_CLASSES,
        "n_utts": len(train_utts),
        "train_ids": [u["id"] for u in train_utts],
        "heldout_ids": sorted(held_set),
        "n_en_tokens": en_tot_tr,
        "n_en_strict_ok": en_ok_tr,
        "en_strict_accept_rate": round(en_acc_tr, 4),
        "zh_strict_accept_rate": round(zh_acc_tr, 4),
        "speaker_f0_med_hz": round(f0_med_global, 2),
        "zh_f0_med_hz": round(f0_med_zh, 2),
        "en_f0_med_hz": round(f0_med_en, 2),
        "ref_log_amp": round(log_ref_med, 5),
        "ref_voiced_db": round(voiced_db_med, 3),
        "sil_cep64": (sil_cep_sum / max(1, sil_cnt_tot)).astype(np.float32),
        "sil_bap6": (sil_bap_sum / max(1, sil_cnt_tot)).astype(np.float32),
        "utterances": train_utts,
        "tokens": flat_en_train,
        "switch_events": all_switches_train,
    }
    held_payload = {
        "schema": "sf_bilingual_heldout_v1",
        "corpus": "SF_bilingual_zh_en",
        "speaker": "sf_female",
        "sr": SR,
        "n_utts": len(held_utts),
        "heldout_ids": [u["id"] for u in held_utts],
        "utterances": held_utts,
        "tokens": flat_en_held,
        "switch_events": held_switches,
    }

    with open(SF_TRAIN_TOKENS_PATH, "wb") as f:
        pickle.dump(train_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(SF_HELDOUT_PATH, "wb") as f:
        pickle.dump(held_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "corpus": "SF_bilingual_zh_en",
        "speaker": "sf_female",
        "train_utterances": len(train_utts),
        "heldout_utterances": len(held_utts),
        "zero_leakage_verified": True,
        "n_train_en_tokens": en_tot_tr,
        "n_train_en_strict_ok": en_ok_tr,
        "en_strict_accept_rate": round(en_acc_tr, 4),
        "zh_strict_accept_rate": round(zh_acc_tr, 4),
        "heldout_en_accept_rate": round(en_acc_ho, 4),
        "n_switch_events": len(all_switches_train),
        "en_quarantine_reasons": dict(reason_en),
        "speaker_f0_med_hz": round(f0_med_global, 2),
        "zh_f0_med_hz": round(f0_med_zh, 2),
        "en_f0_med_hz": round(f0_med_en, 2),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    SF_ALIGN_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 12 SF_bilingual 24k extraction & joint ZH+EN forced alignment")
    parser.add_argument("--zip", type=Path, default=DEFAULT_SF_ZIP)
    parser.add_argument("--n-em", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 8))
    args = parser.parse_args()
    prepare_and_align_sf_bilingual(args.zip, n_em=args.n_em, workers=args.workers)
