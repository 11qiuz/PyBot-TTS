"""Phase-6 Manner-Consistent Forced Aligner & Acoustic Token Quarantine (`exp_v10_g2p_and_align`).

Upgrades ``exp_v8_clarity.en_utau_bank.align_ipa_sequence`` by:
1. Using Phase-6 G2P (``g2p_audit_and_fix.plan_v10``) so 100% of CMU-ARCTIC words
   align against true General American phoneme targets instead of broken rule OOVs.
2. Adding 5 ms formant-region F1/F2, nasal-murmur (120-450 vs 500-2k Hz), and HF stop-burst
   derivative emissions, plus strict per-manner minimum durations (no 5 ms starved tokens).
3. Quarantining acoustically contradicted, near-silent (< -42 dBFS), or sub-20 ms tokens
   before they enter the donor concatenation bank.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.fft import rfftfreq

ROOT_IMPORT = Path(__file__).resolve().parent.parent
if str(ROOT_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_IMPORT))

from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS
from exp_v6_vocoder import vocoder as v6_voc
from exp_v8_clarity import en_utau_bank as v8_en, utau_concat as v8_uc
from exp_v10_g2p_and_align import EXTENDED_INDEX, HOP, MANIFEST_PATH, ROOT_DIR, SR
from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10

UNVOICED_FRIC_AFFR = {"s", "ʃ", "f", "θ", "h", "tʃ", "x", "ɕ", "ʂ", "ts", "tɕ", "tʂ"}
UNVOICED_STOPS = {"p", "t", "k"}
VOICED_SONORANTS = {"m", "n", "ŋ", "l", "ɹ", "ɻ", "j", "w"}


@dataclass
class VerifiedToken:
    ipa: str
    speaker: str
    utterance_id: str
    prev_ipa: str
    next_ipa: str
    log_env: np.ndarray        # (T, 513) float32
    bap: np.ndarray            # (T, 5) float32
    voiced: np.ndarray         # (T,) bool
    f0_rel_st: np.ndarray      # (T,) float32
    dur_ms: float
    rms_dbfs: float
    voiced_frac: float
    uv_hf_lf_db: float
    manner: str
    purity_ok: bool
    quarantine_reason: str
    raw_wav: np.ndarray | None = None


def _compute_acoustic_tracks(feats: v6_voc.VocoderFeatures, sr: int = SR) -> dict[str, np.ndarray]:
    freqs = rfftfreq(feats.n_fft, 1.0 / sr)
    env_pwr = np.exp(2.0 * feats.log_env)
    m_lf = (freqs >= 80.0) & (freqs <= 1000.0)
    m_nas_lo = (freqs >= 120.0) & (freqs <= 450.0)
    m_mid = (freqs >= 500.0) & (freqs <= 2000.0)
    m_f1_lo = (freqs >= 200.0) & (freqs <= 500.0)
    m_f1_hi = (freqs >= 550.0) & (freqs <= 1200.0)
    m_hf = (freqs >= 2000.0) & (freqs <= 8000.0)

    def _band_db(m: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(np.mean(env_pwr[:, m], axis=1), 1e-14))

    lf_db = _band_db(m_lf)
    hf_db = _band_db(m_hf)
    nas_lo_db = _band_db(m_nas_lo)
    mid_db = _band_db(m_mid)
    f1_lo_db = _band_db(m_f1_lo)
    f1_hi_db = _band_db(m_f1_hi)
    tot_db = 10.0 * np.log10(np.maximum(np.mean(env_pwr, axis=1), 1e-14))
    tot_peak = float(np.percentile(tot_db, 95)) if tot_db.size else 0.0

    burst_pos = np.maximum(0.0, np.diff(hf_db, prepend=hf_db[0]))
    return {
        "voi": feats.voiced.astype(np.float64),
        "hf_lf_db": hf_db - lf_db,
        "mid_lf_db": mid_db - nas_lo_db,
        "f1_open_db": f1_hi_db - f1_lo_db,
        "rel_db": tot_db - tot_peak,
        "burst_pos": burst_pos,
    }


def _trim_silence_edges(feats: v6_voc.VocoderFeatures, tracks: dict[str, np.ndarray]) -> tuple[int, int]:
    """Tightly trim non-voiced dead silence (< -34 dB) from edges without clipping voiced starts."""
    act = np.where((tracks["voi"] > 0.5) | (tracks["rel_db"] > -30.0))[0]
    if act.size == 0:
        return 0, len(feats.log_env)
    lo = max(0, int(act[0]) - 2)
    hi = min(len(feats.log_env), int(act[-1]) + 3)
    return lo, max(lo + 4, hi)


def _manner_min_frames(ipa: str, manner: str, is_voiced: bool) -> int:
    if ipa == "ɾ" or (manner == "stop" and is_voiced):
        return 4  # 20 ms
    if manner == "stop":
        return 4  # 20 ms (closure + burst)
    if manner in ("fricative", "affricate"):
        return 5 if is_voiced else 7  # 25 ms voiced, 35 ms unvoiced
    if manner in ("vowel", "diphthong"):
        return 8  # 40 ms
    return 5  # 25 ms nasal / liquid / glide


def align_ipa_sequence_v10(
    ipas: list[str],
    feats: v6_voc.VocoderFeatures,
    sr: int = SR,
) -> tuple[list[tuple[int, int]], dict]:
    """Manner-consistent monotonic DP aligner with F1/nasal/burst cues and min durations."""
    n_tot = len(feats.log_env)
    m_phones = len(ipas)
    if m_phones == 0:
        return [], {"dp_ok": True}

    tracks_full = _compute_acoustic_tracks(feats, sr=sr)
    tr_s, tr_e = _trim_silence_edges(feats, tracks_full)
    n_fr = tr_e - tr_s
    if n_fr < m_phones * 3:
        tr_s, tr_e = 0, n_tot
        n_fr = n_tot

    voi = tracks_full["voi"][tr_s:tr_e]
    hf_lf = tracks_full["hf_lf_db"][tr_s:tr_e]
    mid_lf = tracks_full["mid_lf_db"][tr_s:tr_e]
    f1_open = tracks_full["f1_open_db"][tr_s:tr_e]
    rel_db = tracks_full["rel_db"][tr_s:tr_e]
    burst = tracks_full["burst_pos"][tr_s:tr_e]

    base_durs: list[float] = []
    manners: list[str] = []
    is_voi: list[bool] = []
    hard_min_f: list[int] = []
    for p in ipas:
        sp = IPA_SPECS.get(p)
        bd = sp.base_dur_ms if sp else 75.0
        mn = sp.manner if sp else "vowel"
        iv = bool(sp.voiced) if sp else True
        base_durs.append(bd)
        manners.append(mn)
        is_voi.append(iv)
        hard_min_f.append(_manner_min_frames(p, mn, iv))

    scale = (n_fr * 5.0) / max(1.0, sum(base_durs))
    exp_f = [max(float(hm), bd * scale / 5.0) for hm, bd in zip(hard_min_f, base_durs)]

    # Keep strict minimum durations unless utterance physically has fewer frames
    min_f = [max(hm, int(math.floor(0.45 * ef))) for hm, ef in zip(hard_min_f, exp_f)]
    if sum(min_f) > n_fr:
        # First drop down to hard_min_f, then proportional above 3 frames (15 ms)
        min_f = list(hard_min_f)
        while sum(min_f) > n_fr:
            changed = False
            for idx_r in np.argsort(min_f)[::-1]:
                if min_f[idx_r] > 3:
                    min_f[idx_r] -= 1
                    changed = True
                    if sum(min_f) <= n_fr:
                        break
            if not changed:
                break
        if sum(min_f) > n_fr:
            edges = np.linspace(tr_s, tr_e, m_phones + 1).astype(int)
            return [(int(edges[i]), int(max(edges[i] + 1, edges[i + 1]))) for i in range(m_phones)], {"dp_ok": False}

    max_f = [max(mn + 3, int(math.ceil(2.4 * ef))) for mn, ef in zip(min_f, exp_f)]
    if sum(max_f) < n_fr:
        extra_each = (n_fr - sum(max_f)) // m_phones + 3
        max_f = [mx + extra_each for mx in max_f]

    # Emission score matrix (m_phones, n_fr)
    score = np.zeros((m_phones, n_fr), dtype=np.float64)
    for i, p in enumerate(ipas):
        man = manners[i]
        v_exp = is_voi[i]
        if not v_exp and man in ("fricative", "affricate"):
            # Unvoiced fricative/affricate: strong penalty for voiced vowels, reward HF/LF + burst + audible energy
            score[i] = (
                2.1 * (1.0 - voi)
                - 1.4 * voi
                + 0.22 * np.clip(hf_lf, -12.0, 18.0)
                + (0.12 * np.clip(burst, 0.0, 10.0) if man == "affricate" else 0.0)
                - 0.25 * np.maximum(0.0, -rel_db - 28.0)
            )
        elif not v_exp and man == "stop":
            # Unvoiced stop: unvoiced closure or burst onset (quiet closure permitted)
            score[i] = (
                1.9 * (1.0 - voi)
                - 1.2 * voi
                + 0.14 * np.clip(burst, 0.0, 12.0)
                + 0.06 * np.clip(hf_lf, -10.0, 14.0)
                - 0.12 * np.maximum(0.0, -rel_db - 36.0)
            )
        elif v_exp and man in ("fricative", "affricate", "stop"):
            # Voiced obstruent: reward moderate HF/LF or burst while tolerating voicing
            score[i] = (
                0.7 * voi
                + 0.08 * np.clip(hf_lf, -10.0, 12.0)
                + 0.10 * np.clip(burst, 0.0, 10.0)
                - 0.15 * np.maximum(0.0, -rel_db - 28.0)
            )
        elif man == "nasal":
            # Nasal murmur: voiced + low mid_lf_db (strong 120-450 Hz vs 500-2k anti-resonance)
            score[i] = (
                1.6 * voi
                - 1.8 * (1.0 - voi)
                - 0.16 * np.clip(mid_lf, -12.0, 12.0)
                - 0.12 * np.maximum(0.0, hf_lf - 0.0)
                - 0.18 * np.maximum(0.0, -rel_db - 24.0)
            )
        elif man in ("liquid", "glide"):
            score[i] = (
                1.6 * voi
                - 1.8 * (1.0 - voi)
                - 0.08 * np.maximum(0.0, hf_lf - 3.0)
                - 0.18 * np.maximum(0.0, -rel_db - 22.0)
            )
        else:
            # Vowel / diphthong: strong voicing + audible RMS + low unvoiced HF dominance
            f1_bonus = 0.06 * np.clip(f1_open, -6.0, 10.0) if p in ("ɑ", "æ", "a", "ɔ", "aɪ", "aʊ") else 0.0
            score[i] = (
                2.2 * voi
                - 2.4 * (1.0 - voi)
                + f1_bonus
                - 0.14 * np.maximum(0.0, hf_lf - 2.0)
                - 0.25 * np.maximum(0.0, -rel_db - 18.0)
            )

    pref = np.concatenate([np.zeros((m_phones, 1)), np.cumsum(score, axis=1)], axis=1)
    suf_min = [0] * (m_phones + 1)
    suf_max = [0] * (m_phones + 1)
    for i in range(m_phones - 1, -1, -1):
        suf_min[i] = suf_min[i + 1] + min_f[i]
        suf_max[i] = suf_max[i + 1] + max_f[i]

    dp = np.full((m_phones + 1, n_fr + 1), -1e12, dtype=np.float64)
    bp = np.zeros((m_phones + 1, n_fr + 1), dtype=np.int32)
    dp[0, 0] = 0.0

    for i in range(m_phones):
        ef = exp_f[i]
        mn, mx = min_f[i], max_f[i]
        rem_min = suf_min[i + 1]
        rem_max = suf_max[i + 1]
        j_lo = max(1, n_fr - rem_max)
        j_hi = n_fr - rem_min
        p_row = pref[i]
        for j in range(j_lo, j_hi + 1):
            k_lo = max(0, j - mx)
            k_hi = min(j - mn, n_fr - suf_min[i])
            if k_hi < k_lo:
                continue
            ks = np.arange(k_lo, k_hi + 1)
            durs = j - ks
            dur_pen = -0.30 * ((durs - ef) / max(2.0, ef)) ** 2
            vals = dp[i, ks] + (p_row[j] - p_row[ks]) + dur_pen
            best_idx = int(np.argmax(vals))
            dp[i + 1, j] = vals[best_idx]
            bp[i + 1, j] = ks[best_idx]

    if dp[m_phones, n_fr] <= -1e11:
        edges = np.linspace(tr_s, tr_e, m_phones + 1).astype(int)
        return [(int(edges[i]), int(max(edges[i] + 1, edges[i + 1]))) for i in range(m_phones)], {"dp_ok": False}

    bounds = []
    cur = n_fr
    for i in range(m_phones, 0, -1):
        prev = int(bp[i, cur])
        bounds.append((tr_s + prev, tr_s + cur))
        cur = prev
    bounds.reverse()
    return bounds, {"dp_ok": True, "trimmed_span": [int(tr_s), int(tr_e)]}


def score_token_purity(
    ipa: str,
    log_env: np.ndarray,
    voiced: np.ndarray,
    raw_wav: np.ndarray | None,
    sr: int = SR,
    hop: int = HOP,
) -> tuple[bool, str, dict]:
    """Quarantine check returning ``(purity_ok, reason, metrics_dict)`` for a donor phone/syllable token."""
    n_fr = len(log_env)
    dur_ms = float(n_fr * hop * 1000.0 / sr)
    sp = IPA_SPECS.get(ipa)
    manner = sp.manner if sp else "vowel"
    v_exp = bool(sp.voiced) if sp else True
    voiced_frac = float(np.mean(voiced)) if voiced.size else 0.0

    if raw_wav is not None and raw_wav.size > 0:
        rms = float(np.sqrt(np.mean(raw_wav.astype(np.float64) ** 2)))
    else:
        # Envelope RMS approximation (scaled relative to normalized vocoder level)
        rms = float(np.sqrt(np.mean(np.exp(2.0 * log_env)))) * 0.08
    rms_dbfs = 20.0 * math.log10(max(1e-7, rms))

    freqs = rfftfreq((log_env.shape[1] - 1) * 2, 1.0 / sr)
    m_lo = (freqs >= 80.0) & (freqs <= 1000.0)
    m_hi = (freqs >= 2000.0) & (freqs <= 8000.0)
    pwr = np.exp(2.0 * log_env)
    uv_hf_lf_db = float(
        10.0 * math.log10(max(1e-14, float(np.mean(pwr[:, m_hi]))))
        - 10.0 * math.log10(max(1e-14, float(np.mean(pwr[:, m_lo]))))
    )

    info = {
        "dur_ms": round(dur_ms, 1),
        "rms_dbfs": round(rms_dbfs, 2),
        "voiced_frac": round(voiced_frac, 3),
        "uv_hf_lf_db": round(uv_hf_lf_db, 2),
        "manner": manner,
    }

    min_ms = 20.0 if manner == "stop" or ipa == "ɾ" else (28.0 if manner in ("fricative", "affricate") else 32.0)
    if dur_ms < min_ms:
        return False, "too_short", info
    if rms_dbfs < -45.0:
        return False, "near_silent", info

    # Strict manner-vs-acoustic consistency checks
    if manner in ("vowel", "diphthong") or ipa in VOICED_SONORANTS:
        if voiced_frac < 0.40:
            return False, "voiced_contradiction", info
    elif ipa in UNVOICED_FRIC_AFFR:
        if voiced_frac > 0.65:
            return False, "unvoiced_fric_too_voiced", info
        if uv_hf_lf_db < -5.0:
            return False, "low_hf_fricative", info
    elif ipa in UNVOICED_STOPS:
        if voiced_frac > 0.70:
            return False, "unvoiced_stop_too_voiced", info

    return True, "ok", info


def audit_aligner_before_after(max_en_utts: int = 140) -> dict:
    """Compare Phase-4/5 DP aligner vs Phase-6 landmark_aligner + quarantine on 140 EN train utts."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    excl_ids = {f"{spk}_00{i}" for spk in ("slt", "bdl") for i in range(30, 40)}
    en_utts = [u for u in manifest["english"]["utterances"] if u["id"] not in excl_ids][:max_en_utts]

    v9_stats = {"total": 0, "short_lt20ms": 0, "silent_lt45": 0, "manner_bad": 0, "passed_all": 0}
    v10_stats = {
        "total": 0,
        "short_lt20ms": 0,
        "silent_lt45": 0,
        "manner_bad": 0,
        "quarantined": 0,
        "accepted": 0,
        "dp_ok_utts": 0,
        "reasons": {},
    }

    for u in en_utts:
        x, sr = v6_voc.read_wav_mono(ROOT_DIR / u["path"])
        s0, s1 = v8_en._trim_active_span(x, sr=sr)
        xa = x[s0:s1]
        feats = v6_voc.analyze(xa, sr=sr)
        text = u["text"]

        # v9 baseline: cf.plan + v8_en.align_ipa_sequence
        ir9 = v8_en.cf.plan(text, allow_oov_fallback=True)
        ipas9 = [p for unit in ir9["units"] for p in unit.get("ipa_phones", [])]
        bounds9 = v8_en.align_ipa_sequence(ipas9, feats, sr=sr)
        for ipa, (f0_i, f1_i) in zip(ipas9, bounds9):
            w_seg = xa[f0_i * HOP : min(len(xa), f1_i * HOP)]
            ok, reason, met = score_token_purity(ipa, feats.log_env[f0_i:f1_i], feats.voiced[f0_i:f1_i], w_seg, sr=sr)
            v9_stats["total"] += 1
            if met["dur_ms"] < 20.0:
                v9_stats["short_lt20ms"] += 1
            if met["rms_dbfs"] < -45.0:
                v9_stats["silent_lt45"] += 1
            if reason in ("voiced_contradiction", "unvoiced_fric_too_voiced", "low_hf_fricative", "unvoiced_stop_too_voiced"):
                v9_stats["manner_bad"] += 1
            if ok:
                v9_stats["passed_all"] += 1

        # v10 upgraded: plan_v10 + align_ipa_sequence_v10 + quarantine
        ir10 = plan_v10(text)
        ipas10 = [p for unit in ir10["units"] for p in unit.get("ipa_phones", [])]
        bounds10, diag = align_ipa_sequence_v10(ipas10, feats, sr=sr)
        v10_stats["dp_ok_utts"] += int(diag["dp_ok"])
        for ipa, (f0_i, f1_i) in zip(ipas10, bounds10):
            w_seg = xa[f0_i * HOP : min(len(xa), f1_i * HOP)]
            ok, reason, met = score_token_purity(ipa, feats.log_env[f0_i:f1_i], feats.voiced[f0_i:f1_i], w_seg, sr=sr)
            v10_stats["total"] += 1
            if met["dur_ms"] < 20.0:
                v10_stats["short_lt20ms"] += 1
            if met["rms_dbfs"] < -45.0:
                v10_stats["silent_lt45"] += 1
            if reason in ("voiced_contradiction", "unvoiced_fric_too_voiced", "low_hf_fricative", "unvoiced_stop_too_voiced"):
                v10_stats["manner_bad"] += 1
            if ok:
                v10_stats["accepted"] += 1
            else:
                v10_stats["quarantined"] += 1
                v10_stats["reasons"][reason] = v10_stats["reasons"].get(reason, 0) + 1

    # Also audit Mandarin extended/ syllable + onset integrity before/after quarantine
    zh_bank = v8_uc.build_fullres_bank(include_fallback_bank=False)
    zh_tot = 0
    zh_ok = 0
    zh_onsets_tot = 0
    zh_onsets_lt20ms = 0
    zh_onsets_bad_manner = 0
    zh_reasons: dict[str, int] = {}
    for toks in zh_bank.syllables.values():
        for tk in toks:
            zh_tot += 1
            onset_ipas = v8_en.cf.PINYIN_ONSET_TO_IPA.get(tk.onset, []) if tk.onset else []
            onset_ipa = onset_ipas[0] if onset_ipas else ""
            sp = IPA_SPECS.get(onset_ipa) if onset_ipa else None
            v_fr = float(np.mean(tk.voiced)) if tk.voiced.size else 0.0
            bad = None
            if tk.f_split > 0:
                zh_onsets_tot += 1
                on_ms = tk.f_split * HOP * 1000.0 / SR
                if on_ms < 20.0:
                    zh_onsets_lt20ms += 1
                if onset_ipa:
                    _, r_on_raw, _ = score_token_purity(
                        onset_ipa,
                        tk.log_env[: tk.f_split],
                        tk.voiced[: tk.f_split],
                        tk.onset_wav,
                        sr=SR,
                    )
                    if r_on_raw in ("unvoiced_fric_too_voiced", "low_hf_fricative", "unvoiced_stop_too_voiced", "voiced_contradiction"):
                        zh_onsets_bad_manner += 1
            if tk.dur_ms < 55.0:
                bad = "short_syllable"
            elif v_fr < 0.30:
                bad = "low_voiced_syllable"
            elif sp and sp.manner in ("fricative", "affricate") and not sp.voiced and tk.f_split > 0:
                _, r_on, _ = score_token_purity(
                    onset_ipa,
                    tk.log_env[: tk.f_split],
                    tk.voiced[: tk.f_split],
                    tk.onset_wav,
                    sr=SR,
                )
                if r_on != "ok":
                    bad = f"onset_{r_on}"
            if bad is None:
                zh_ok += 1
            else:
                zh_reasons[bad] = zh_reasons.get(bad, 0) + 1

    return {
        "n_en_utts": len(en_utts),
        "en_v9_aligner": v9_stats,
        "en_v10_aligner_before_quarantine": {k: v for k, v in v10_stats.items() if k != "reasons"},
        "en_v10_after_quarantine": {
            "accepted": v10_stats["accepted"],
            "quarantined": v10_stats["quarantined"],
            "accept_rate": round(v10_stats["accepted"] / max(1, v10_stats["total"]), 4),
            "short_lt20ms": 0,
            "silent_lt45": 0,
            "manner_bad": 0,
            "quarantine_reasons": v10_stats["reasons"],
        },
        "zh_extended_bank_audit": {
            "total_tokens": zh_tot,
            "onsets_total": zh_onsets_tot,
            "onsets_lt20ms": zh_onsets_lt20ms,
            "onsets_bad_manner": zh_onsets_bad_manner,
            "verified_ok": zh_ok,
            "quarantined": zh_tot - zh_ok,
            "reasons": zh_reasons,
        },
    }


def extract_verified_en_bank(exclude_ids: set[str] | None = None) -> dict[str, list[VerifiedToken]]:
    """Build purified EN donor IPA token bank (quarantining contradicted/silent/starved units)."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    excl = exclude_ids or set()
    by_ipa: dict[str, list[VerifiedToken]] = {}
    for u in manifest["english"]["utterances"]:
        if u["id"] in excl:
            continue
        x, sr = v6_voc.read_wav_mono(ROOT_DIR / u["path"])
        s0, s1 = v8_en._trim_active_span(x, sr=sr)
        xa = x[s0:s1]
        feats = v6_voc.analyze(xa, sr=sr)
        ir = plan_v10(u["text"])
        ipas = [p for unit in ir["units"] for p in unit.get("ipa_phones", [])]
        bounds, _ = align_ipa_sequence_v10(ipas, feats, sr=sr)

        # Normalize utterance voiced level using [250, 7200] Hz Parseval RMS (-3.05 np)
        # matching AISHELL-3 ZH bank loudness (~ -26.5 dB full-band) without 8-12 kHz bias.
        v_mask = feats.voiced & (feats.f0_hz > 0.0)
        env_norm = feats.log_env.copy()
        if np.any(v_mask):
            freqs = rfftfreq(feats.n_fft, 1.0 / sr)
            m_sb = (freqs >= 250.0) & (freqs <= 7200.0)
            sb_rms_np = 0.5 * np.log(np.maximum(1e-18, np.mean(np.exp(2.0 * env_norm[v_mask][:, m_sb]), axis=1)))
            env_norm = env_norm + (-3.05 - float(np.median(sb_rms_np)))
        f0_rel = np.zeros(len(env_norm), dtype=np.float64)
        if np.any(v_mask):
            med_f0 = float(np.median(feats.f0_hz[v_mask]))
            f0_rel[v_mask] = 12.0 * np.log2(np.maximum(50.0, feats.f0_hz[v_mask]) / max(50.0, med_f0))

        for k, (ipa, (f0_i, f1_i)) in enumerate(zip(ipas, bounds)):
            w_seg = xa[f0_i * HOP : min(len(xa), f1_i * HOP)]
            ok, reason, met = score_token_purity(ipa, feats.log_env[f0_i:f1_i], feats.voiced[f0_i:f1_i], w_seg, sr=sr)
            if not ok:
                continue
            sp = IPA_SPECS.get(ipa)
            tok = VerifiedToken(
                ipa=ipa,
                speaker=u["speaker"],
                utterance_id=u["id"],
                prev_ipa=ipas[k - 1] if k > 0 else "sil",
                next_ipa=ipas[k + 1] if k + 1 < len(ipas) else "sil",
                log_env=env_norm[f0_i:f1_i].astype(np.float32),
                bap=feats.bap[f0_i:f1_i].astype(np.float32),
                voiced=feats.voiced[f0_i:f1_i].astype(bool),
                f0_rel_st=f0_rel[f0_i:f1_i].astype(np.float32),
                dur_ms=met["dur_ms"],
                rms_dbfs=met["rms_dbfs"],
                voiced_frac=met["voiced_frac"],
                uv_hf_lf_db=met["uv_hf_lf_db"],
                manner=sp.manner if sp else "vowel",
                purity_ok=True,
                quarantine_reason=reason,
                raw_wav=w_seg.astype(np.float32),
            )
            by_ipa.setdefault(ipa, []).append(tok)
    return by_ipa


if __name__ == "__main__":
    print(json.dumps(audit_aligner_before_after(), indent=2))
