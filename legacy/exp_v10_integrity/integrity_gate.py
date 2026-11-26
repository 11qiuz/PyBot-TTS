"""Acoustic Integrity Gate, Token Quarantine Filter & Perceptual Duration/Energy Floors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.fft import rfftfreq

from exp_v5_arch.crossling_frontend import PINYIN_ONSET_TO_IPA
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS
from exp_v10_integrity import HOP, N_FFT, SR
from exp_v10_integrity.gain_calibration import classify_ipa_manner, parseval_frame_db


_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_LF_MASK = (_FREQS >= 80.0) & (_FREQS <= 1000.0)
_HF_MASK = (_FREQS >= 2000.0) & (_FREQS <= 8000.0)

PERCEPTUAL_MIN_FRAMES: dict[str, int] = {
    "vowel_stressed": 9,       # 45 ms
    "vowel_unstressed": 7,     # 35 ms
    "nasal": 6,                # 30 ms
    "liquid": 6,               # 30 ms
    "glide": 6,                # 30 ms
    "voiced_fricative": 6,     # 30 ms
    "voiced_affricate": 7,     # 35 ms
    "voiced_stop": 5,          # 25 ms
    "voiceless_fricative": 7,  # 35 ms
    "voiceless_affricate": 7,  # 35 ms
    "voiceless_stop": 5,       # 25 ms
    "zh_syllable": 12,         # 60 ms
}


@dataclass
class TokenAuditResult:
    ok: bool
    reason: str
    dur_ms: float
    p90_db: float
    voiced_frac: float
    hf_lf_db: float


def _token_hf_lf_db(log_env: np.ndarray) -> float:
    pwr = np.exp(2.0 * np.asarray(log_env, dtype=np.float64))
    lf = float(np.mean(pwr[:, _LF_MASK])) + 1e-16
    hf = float(np.mean(pwr[:, _HF_MASK])) + 1e-16
    return 10.0 * float(np.log10(hf / lf))


def audit_en_token(tok: Any, ref_vowel_db: float = -14.5) -> TokenAuditResult:
    """Verify an English EnPhoneToken / V9EnToken satisfies duration, audibility, and manner integrity."""
    ipa = str(getattr(tok, "ipa", "ə"))
    dur_ms = float(getattr(tok, "dur_ms", 0.0))
    log_env = np.asarray(tok.log_env, dtype=np.float64)
    voiced = np.asarray(tok.voiced, dtype=bool)
    n_fr = len(log_env)
    if n_fr < 2 or dur_ms < 15.0:
        return TokenAuditResult(False, "too_short", dur_ms, -80.0, 0.0, 0.0)

    pdb = parseval_frame_db(log_env, n_fft=2 * (log_env.shape[1] - 1))
    p90_db = float(np.percentile(pdb, 90))
    v_frac = float(np.mean(voiced)) if voiced.size else 0.0
    hflf = _token_hf_lf_db(log_env)

    sp = IPA_SPECS.get(ipa)
    man = sp.manner if sp else "vowel"
    is_v_exp = bool(sp.voiced) if sp else True

    # 1. Silent/vanished check relative to utterance vowel anchor
    sil_thresh = (ref_vowel_db - 24.0) if man in ("stop",) else (ref_vowel_db - 19.0)
    if p90_db < sil_thresh:
        return TokenAuditResult(False, "near_silent", dur_ms, p90_db, v_frac, hflf)

    # 2. Manner consistency checks
    if man in ("vowel", "diphthong"):
        if dur_ms < 20.0:
            return TokenAuditResult(False, "too_short", dur_ms, p90_db, v_frac, hflf)
        if v_frac < 0.45:
            return TokenAuditResult(False, "manner_unvoiced_vowel", dur_ms, p90_db, v_frac, hflf)
    elif man in ("nasal", "liquid", "glide"):
        if v_frac < 0.40:
            return TokenAuditResult(False, "manner_unvoiced_sonorant", dur_ms, p90_db, v_frac, hflf)
    elif not is_v_exp and man in ("fricative", "affricate"):
        if dur_ms < 20.0:
            return TokenAuditResult(False, "too_short", dur_ms, p90_db, v_frac, hflf)
        if hflf < -4.5:
            return TokenAuditResult(False, "manner_lf_vowel_bleed", dur_ms, p90_db, v_frac, hflf)
    elif not is_v_exp and man == "stop":
        # Stop must have measurable burst > -38 dB and not be purely steady voiced vowel
        if float(np.max(pdb)) < ref_vowel_db - 21.0:
            return TokenAuditResult(False, "near_silent", dur_ms, p90_db, v_frac, hflf)

    return TokenAuditResult(True, "ok", dur_ms, p90_db, v_frac, hflf)


def min_zh_onset_frames(onset: str) -> int:
    spec = IPA_SPECS[PINYIN_ONSET_TO_IPA[onset][0]]
    return 4 if spec.manner in ("fricative", "affricate") or spec.ipa.endswith("ʰ") else 2


def _audit_zh_onset(tok: Any, end: int, ref_vowel_db: float) -> TokenAuditResult:
    onset = tok.onset
    ipa = PINYIN_ONSET_TO_IPA[onset][0]
    spec = IPA_SPECS[ipa]
    env = tok.log_env[:end]
    dur_ms = len(env) * HOP * 1000.0 / SR
    if len(env) < min_zh_onset_frames(onset):
        return TokenAuditResult(False, "onset_too_short", dur_ms, -80.0, 0.0, 0.0)

    v_frac = float(np.mean(tok.voiced[:end]))
    hflf = _token_hf_lf_db(env)
    p90_db = float(np.percentile(parseval_frame_db(env), 90))
    reason = "ok"
    if p90_db < ref_vowel_db - 23.0:
        reason = "onset_near_silent"
    elif spec.voiced and v_frac < 0.40:
        reason = "onset_unvoiced_sonorant"
    elif not spec.voiced and v_frac > 0.40:
        reason = "onset_vowel_bleed"
    elif spec.manner in ("fricative", "affricate") and ipa != "x" and hflf < -5.5:
        reason = "onset_weak_frication"
    return TokenAuditResult(reason == "ok", reason, dur_ms, p90_db, v_frac, hflf)


def audit_zh_token(tok: Any, ref_vowel_db: float = -14.5, is_onset_only: bool = False) -> TokenAuditResult:
    """Check the vowel nucleus and consonant of a full donor independently."""
    dur_ms = float(tok.dur_ms)
    log_env = np.asarray(tok.log_env, dtype=np.float64)
    if is_onset_only:
        return _audit_zh_onset(tok, len(log_env), ref_vowel_db)

    split = int(np.clip(tok.f_split, 0, len(log_env)))
    rhyme = log_env[split:]
    if len(rhyme) < 6 or dur_ms < 30.0:
        return TokenAuditResult(False, "rhyme_too_short", dur_ms, -80.0, 0.0, 0.0)
    p90_db = float(np.percentile(parseval_frame_db(rhyme), 90))
    rhyme_vf = float(np.mean(tok.voiced[split:]))
    hflf = _token_hf_lf_db(rhyme)
    if p90_db < ref_vowel_db - 18.0:
        return TokenAuditResult(False, "near_silent", dur_ms, p90_db, rhyme_vf, hflf)
    if rhyme_vf < 0.40:
        return TokenAuditResult(False, "manner_unvoiced_rhyme", dur_ms, p90_db, rhyme_vf, hflf)
    if tok.onset:
        onset_result = _audit_zh_onset(tok, split, ref_vowel_db)
        if not onset_result.ok:
            return onset_result
    return TokenAuditResult(True, "ok", dur_ms, p90_db, rhyme_vf, hflf)


def compute_perceptual_phone_frames(
    pairs: list[tuple[str, int | None]],
    raw_durs_ms: list[float],
    target_total_frames: int,
    hop_ms: float = 5.0,
) -> list[int]:
    """Allocate 5 ms frames across an English word's phonemes respecting strict perceptual minimum floors.

    Prevents unstressed vowels, liquids, or word-final codas from being crushed to <20 ms (dropped phone).
    """
    n_p = len(pairs)
    if n_p == 0:
        return []
    mins = [PERCEPTUAL_MIN_FRAMES.get(classify_ipa_manner(ip, st), 6) for ip, st in pairs]
    sum_min = sum(mins)
    total_fr = max(target_total_frames, sum_min)

    extra = total_fr - sum_min
    tot_raw = max(1e-4, sum(raw_durs_ms))
    alloc = list(mins)
    if extra > 0:
        shares = [extra * (max(1.0, d) / tot_raw) for d in raw_durs_ms]
        floors = [int(math.floor(s)) for s in shares]
        rem = extra - sum(floors)
        for k in range(n_p):
            alloc[k] += floors[k]
        # Distribute remainder to highest fractional shares (preferring vowels)
        order = sorted(range(n_p), key=lambda k: (shares[k] - floors[k], 1 if "vowel" in classify_ipa_manner(*pairs[k]) else 0), reverse=True)
        for k in order[:rem]:
            alloc[k] += 1
    return alloc


import math  # noqa: E402


def apply_integrity_gate_to_bank(bank: Any, ref_vowel_db: float | None = None) -> tuple[Any, dict]:
    """Filter a V9DualBank (or compatible EN+ZH bank) removing corrupted/silent/mislabeled tokens.

    Computes adaptive EN vowel & ZH syllable Parseval reference anchors so pre-calibrated
    corpus scaling differences never cause false rejections.
    """
    from exp_v9_clarity_pro.viterbi_utau import V9DualBank

    # Adaptive EN vowel anchor
    if ref_vowel_db is not None:
        en_ref_db = float(ref_vowel_db)
        zh_ref_db = float(ref_vowel_db)
    else:
        en_lvls = []
        for ipa, toks in bank.en_phones.items():
            sp = IPA_SPECS.get(ipa)
            if sp and sp.manner in ("vowel", "diphthong"):
                for t in toks:
                    arr = np.asarray(t.log_env, dtype=np.float64)
                    if len(arr) >= 2:
                        en_lvls.append(float(np.percentile(parseval_frame_db(arr, n_fft=2 * (arr.shape[1] - 1)), 75)))
        en_ref_db = float(np.median(en_lvls)) if en_lvls else -16.5

        zh_lvls = []
        for toks in bank.zh_syllables.values():
            for t in toks:
                arr = np.asarray(t.log_env, dtype=np.float64)
                if len(arr) >= 2:
                    zh_lvls.append(float(np.percentile(parseval_frame_db(arr, n_fft=2 * (arr.shape[1] - 1)), 75)))
        zh_ref_db = float(np.median(zh_lvls)) if zh_lvls else -25.0

    reasons_en: dict[str, int] = {}
    kept_en: dict[str, list] = {}
    tot_en, pass_en = 0, 0
    for ipa, toks in bank.en_phones.items():
        clean = []
        for t in toks:
            tot_en += 1
            res = audit_en_token(t, ref_vowel_db=en_ref_db)
            if res.ok:
                clean.append(t)
                pass_en += 1
            else:
                reasons_en[res.reason] = reasons_en.get(res.reason, 0) + 1
        if clean:
            kept_en[ipa] = clean

    reasons_zh: dict[str, int] = {}
    tot_zh, pass_zh = 0, 0
    kept_zh_syls = {}
    for k, toks in bank.zh_syllables.items():
        clean = []
        for t in toks:
            tot_zh += 1
            res = audit_zh_token(t, ref_vowel_db=zh_ref_db, is_onset_only=False)
            if res.ok:
                clean.append(t)
                pass_zh += 1
            else:
                reasons_zh[res.reason] = reasons_zh.get(res.reason, 0) + 1
        if clean:
            kept_zh_syls[k] = clean

    kept_zh_ons = {}
    for k, toks in bank.zh_onsets.items():
        clean = [t for t in toks if audit_zh_token(t, ref_vowel_db=zh_ref_db, is_onset_only=True).ok]
        if clean:
            kept_zh_ons[k] = clean

    kept_zh_fins = {}
    for k, toks in bank.zh_finals.items():
        clean = [t for t in toks if audit_zh_token(t, ref_vowel_db=zh_ref_db, is_onset_only=False).ok]
        if clean:
            kept_zh_fins[k] = clean

    purified = V9DualBank(
        zh_syllables=kept_zh_syls,
        zh_onsets=kept_zh_ons,
        zh_finals=kept_zh_fins,
        en_phones=kept_en,
        zh_profiles_v2=dict(bank.zh_profiles_v2),
        en_profiles_v2=dict(bank.en_profiles_v2),
        zh_profiles_v3=dict(getattr(bank, "zh_profiles_v3", {})),
        en_profiles_v3=dict(getattr(bank, "en_profiles_v3", {})),
        fallback_unit_bank=bank.fallback_unit_bank,
        prosody_rules=bank.prosody_rules,
        zh_included_ids=list(bank.zh_included_ids),
        zh_excluded_ids=list(bank.zh_excluded_ids),
        en_included_ids=list(bank.en_included_ids),
        en_excluded_ids=list(bank.en_excluded_ids),
    )
    stats = {
        "en_total": tot_en,
        "en_kept": pass_en,
        "en_quarantined": tot_en - pass_en,
        "en_reasons": reasons_en,
        "en_ipas_covered": len(kept_en),
        "zh_syl_total": tot_zh,
        "zh_syl_kept": pass_zh,
        "zh_syl_quarantined": tot_zh - pass_zh,
        "zh_reasons": reasons_zh,
    }
    return purified, stats
