"""Dual-Donor (Mandarin + English) UTAU Full-Res HNM Bank + Cross-Lingual Decoupled Cloner Evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np
from scipy.fft import rfft, rfftfreq

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v5_eval import metrics as ev  # noqa: E402
from exp_v6_crossling_dsp import arbiter as v6_arb  # noqa: E402
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS  # noqa: E402
from exp_v6_eval import fluency as fl  # noqa: E402
from exp_v6_vocoder import SR, vocoder  # noqa: E402
from exp_v7_pybot_tts import crossling_warp as cw, planner_bridge as v7_pb, unit_bank as v7_ub  # noqa: E402
from exp_v8_clarity import EXP_DIR, EXTENDED_INDEX, HOP, N_FFT, SAMPLES_DIR  # noqa: E402
from exp_v8_clarity.clarity_metrics import compute_clarity_metrics  # noqa: E402
from exp_v8_clarity.utau_concat import (  # noqa: E402
    N_BINS,
    OBSTRUENT_IPAS,
    OBSTRUENT_ONSETS,
    SILENCE_LOG_ENV,
    FullResBank,
    _blend_seam,
    _interp_list,
    _resample_1d,
    _resample_2d,
    _score_candidate,
    _stretch_preserving_onset,
    build_fullres_bank,
)
from exp_v8_timbre_clone import timbre_cloner_v2 as tc2  # noqa: E402


EN_MANIFEST = EXP_DIR.parent / "exp_v8_data" / "corpus" / "manifest.json"


@dataclass
class EnPhoneToken:
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
    raw_wav: np.ndarray | None = None


@dataclass
class DualDonorBank:
    zh_bank: FullResBank
    en_phones: dict[str, list[EnPhoneToken]] = field(default_factory=dict)
    en_profiles_v2: dict[str, tc2.TimbreProfileV2] = field(default_factory=dict)
    zh_profiles_v2: dict[str, tc2.TimbreProfileV2] = field(default_factory=dict)
    en_included_ids: list[str] = field(default_factory=list)
    en_excluded_ids: list[str] = field(default_factory=list)


def _trim_active_span(x: np.ndarray, sr: int = SR, pad_ms: float = 15.0) -> tuple[int, int]:
    """Return [start_sample, stop_sample) covering active speech with short padding."""
    hop = HOP
    win = 480
    if len(x) <= win:
        return 0, len(x)
    n_fr = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + np.arange(n_fr)[:, None] * hop
    rms = np.sqrt(np.mean(x[idx] ** 2, axis=1))
    pk = float(np.max(rms)) if rms.size else 0.0
    act = np.where(rms >= max(1e-4, 0.035 * pk))[0]
    if act.size == 0:
        return 0, len(x)
    pad = int(round(pad_ms * sr / 1000.0))
    s0 = max(0, int(act[0]) * hop - pad)
    s1 = min(len(x), int(act[-1]) * hop + win + pad)
    return s0, max(s0 + 480, s1)


def align_ipa_sequence(
    ipas: list[str],
    feats: vocoder.VocoderFeatures,
    sr: int = SR,
) -> list[tuple[int, int]]:
    """Zero-neural monotonic DP forced aligner over 5 ms (voiced, HF/LF, RMS) acoustic features."""
    n_fr = len(feats.log_env)
    m_phones = len(ipas)
    if m_phones == 0 or n_fr < m_phones * 2:
        # Uniform fallback if clip is too short
        edges = np.linspace(0, n_fr, m_phones + 1).astype(int)
        return [(int(edges[i]), int(max(edges[i] + 1, edges[i + 1]))) for i in range(m_phones)]

    freqs = rfftfreq(feats.n_fft, 1.0 / sr)
    lf_m = (freqs >= 80.0) & (freqs <= 1000.0)
    hf_m = (freqs >= 2000.0) & (freqs <= 8000.0)
    env_pwr = np.exp(2.0 * feats.log_env)
    lf_db = 10.0 * np.log10(np.maximum(np.mean(env_pwr[:, lf_m], axis=1), 1e-14))
    hf_db = 10.0 * np.log10(np.maximum(np.mean(env_pwr[:, hf_m], axis=1), 1e-14))
    hf_lf = hf_db - lf_db
    tot_db = 10.0 * np.log10(np.maximum(np.mean(env_pwr, axis=1), 1e-14))
    tot_peak = float(np.percentile(tot_db, 95))
    rel_db = tot_db - tot_peak
    voi = feats.voiced.astype(np.float64)

    # Prior duration in frames scaled to match total utterance frames
    base_durs = []
    manners = []
    is_voi = []
    for p in ipas:
        sp = IPA_SPECS.get(p)
        base_durs.append(sp.base_dur_ms if sp else 75.0)
        manners.append(sp.manner if sp else "vowel")
        is_voi.append(bool(sp.voiced) if sp else True)
    scale = (n_fr * 5.0) / max(1.0, sum(base_durs))

    # Frame emission log-likelihood matrix (m_phones, n_fr)
    score = np.zeros((m_phones, n_fr), dtype=np.float64)
    for i in range(m_phones):
        man = manners[i]
        v_exp = is_voi[i]
        if not v_exp and man in ("fricative", "affricate"):
            # Unvoiced fricative: unvoiced + positive HF/LF + audible energy
            score[i] = 1.6 * (1.0 - voi) + 0.18 * np.clip(hf_lf, -12.0, 18.0) - 0.15 * np.maximum(0.0, -rel_db - 28.0)
        elif not v_exp and man == "stop":
            # Unvoiced stop: unvoiced + low-to-burst energy
            score[i] = 1.6 * (1.0 - voi) + 0.08 * np.clip(hf_lf, -10.0, 15.0)
        elif v_exp and man in ("fricative", "affricate", "stop"):
            score[i] = 0.6 * voi + 0.06 * hf_lf - 0.10 * np.maximum(0.0, -rel_db - 26.0)
        elif man in ("vowel", "diphthong"):
            score[i] = 1.8 * voi - 0.10 * np.maximum(0.0, hf_lf - 2.0) - 0.18 * np.maximum(0.0, -rel_db - 18.0)
        else:  # nasal, liquid, glide
            score[i] = 1.3 * voi - 0.06 * np.maximum(0.0, hf_lf - 4.0) - 0.12 * np.maximum(0.0, -rel_db - 22.0)

    # Prefix sums for O(1) segment score lookup
    pref = np.concatenate([np.zeros((m_phones, 1)), np.cumsum(score, axis=1)], axis=1)

    exp_f = [max(2.0, bd * scale / 5.0) for bd in base_durs]
    min_f = [max(2, int(math.floor(0.42 * ef))) for ef in exp_f]
    max_f = [max(mn + 2, int(math.ceil(2.2 * ef))) for mn, ef in zip(min_f, exp_f)]
    # Ensure feasible sum of min_f <= n_fr
    while sum(min_f) > n_fr:
        for idx_r in range(m_phones):
            if min_f[idx_r] > 1:
                min_f[idx_r] -= 1
                if sum(min_f) <= n_fr:
                    break

    # Suffix sum of min_f and max_f to bound DP search cleanly
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
            dur_pen = -0.35 * ((durs - ef) / max(2.0, ef)) ** 2
            vals = dp[i, ks] + (p_row[j] - p_row[ks]) + dur_pen
            best_idx = int(np.argmax(vals))
            dp[i + 1, j] = vals[best_idx]
            bp[i + 1, j] = ks[best_idx]

    # If exact DP hit infeasible bounds (rare edge case), widen to uniform split
    if dp[m_phones, n_fr] <= -1e11:
        edges = np.linspace(0, n_fr, m_phones + 1).astype(int)
        return [(int(edges[i]), int(max(edges[i] + 1, edges[i + 1]))) for i in range(m_phones)]

    bounds = []
    cur = n_fr
    for i in range(m_phones, 0, -1):
        prev = int(bp[i, cur])
        bounds.append((prev, cur))
        cur = prev
    bounds.reverse()
    return bounds


def _utterance_ipas_from_text(text: str) -> list[str]:
    plan = cf.plan(text, allow_oov_fallback=True)
    out = []
    for u in plan["units"]:
        out.extend(u.get("ipa_phones") or [])
    return out


def build_dual_donor_bank(
    zh_utterances: list[dict] | None = None,
    zh_exclude_ids: Iterable[str] = (),
    en_utterances: list[dict] | None = None,
    en_exclude_ids: Iterable[str] = (),
    zh_precomputed: dict | None = None,
    en_precomputed: dict | None = None,
    sr: int = SR,
) -> DualDonorBank:
    """Build combined Mandarin FullResBank + English Full-Res DP-Aligned Phone Bank."""
    zh_bank = build_fullres_bank(
        zh_utterances, exclude_ids=zh_exclude_ids, precomputed=zh_precomputed, sr=sr, include_fallback_bank=True
    )

    if en_utterances is None:
        man = json.loads(EN_MANIFEST.read_text(encoding="utf-8"))
        en_utterances = man["english"]["utterances"]

    en_excl = set(en_exclude_ids)
    en_train = [u for u in en_utterances if u["id"] not in en_excl]
    en_data: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures]] = {}
    en_by_spk: dict[str, list[tuple[np.ndarray, vocoder.VocoderFeatures]]] = {}

    for u in en_train:
        uid = u["id"]
        if en_precomputed is not None and uid in en_precomputed:
            x_act, feats = en_precomputed[uid]
        else:
            p_wav = Path(u["path"])
            if not p_wav.is_absolute():
                p_wav = EXP_DIR.parent / p_wav
            x, fsr = vocoder.read_wav_mono(p_wav)
            if fsr != sr:
                raise ValueError(f"Expected {sr} Hz for {uid}, got {fsr}")
            s0, s1 = _trim_active_span(x, sr=sr)
            x_act = x[s0:s1]
            feats = vocoder.analyze(x_act, sr=sr)
        en_data[uid] = (x_act, feats)
        en_by_spk.setdefault(u["speaker"], []).append((x_act, feats))

    en_profs_v2 = {
        spk: tc2.extract_v2_profile(items, sr=sr, name=spk)
        for spk, items in sorted(en_by_spk.items())
    }

    # Build V2 profiles for ZH speakers in zh_bank too (for unified V/UV-decoupled adaptation!)
    if zh_utterances is None:
        zh_utterances = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
    zh_excl = set(zh_exclude_ids)
    zh_by_spk: dict[str, list] = {}
    for u in zh_utterances:
        if u["id"] in zh_excl:
            continue
        if zh_precomputed is not None and u["id"] in zh_precomputed:
            zh_by_spk.setdefault(u["speaker"], []).append(zh_precomputed[u["id"]])
        else:
            x, _ = vocoder.read_wav_mono(u["path"])
            xa = x[u["active_start"] : u["active_stop"]]
            zh_by_spk.setdefault(u["speaker"], []).append((xa, vocoder.analyze(xa, sr=sr)))
    zh_profs_v2 = {
        spk: tc2.extract_v2_profile(items, sr=sr, name=spk)
        for spk, items in sorted(zh_by_spk.items())
    }

    en_phones: dict[str, list[EnPhoneToken]] = {}
    for u in en_train:
        uid = u["id"]
        spk = u["speaker"]
        x_act, feats = en_data[uid]
        ipas = _utterance_ipas_from_text(u["text"])
        if not ipas:
            continue

        v_mask = feats.voiced & (feats.f0_hz > 0.0)
        env_norm = feats.log_env.copy()
        wav_scale = 1.0
        if np.any(v_mask):
            med_lvl = float(np.median(np.mean(env_norm[v_mask], axis=1)))
            shift = -6.9 - med_lvl
            env_norm = env_norm + shift
            wav_scale = math.exp(shift)
            f0_med = float(np.median(feats.f0_hz[v_mask]))
            f0_rel = np.where(v_mask, 12.0 * np.log2(np.maximum(50.0, feats.f0_hz) / max(50.0, f0_med)), 0.0)
        else:
            f0_rel = np.zeros(len(env_norm), dtype=np.float64)

        bounds = align_ipa_sequence(ipas, feats, sr=sr)
        for idx_p, (ipa, (f0_i, f1_i)) in enumerate(zip(ipas, bounds)):
            if f1_i <= f0_i:
                continue
            prev_ip = ipas[idx_p - 1] if idx_p > 0 else "SIL"
            next_ip = ipas[idx_p + 1] if idx_p + 1 < len(ipas) else "SIL"
            voi_sl = feats.voiced[f0_i:f1_i].copy()
            sp_info = IPA_SPECS.get(ipa)
            if sp_info is not None and not sp_info.voiced and sp_info.manner in ("fricative", "affricate", "stop"):
                voi_sl[:] = False
            s0 = max(0, f0_i * HOP)
            s1 = min(len(x_act), f1_i * HOP)
            raw_w = (x_act[s0:s1] * wav_scale).astype(np.float32) if (s1 > s0 and not np.any(voi_sl)) else None
            tok = EnPhoneToken(
                ipa=ipa,
                speaker=spk,
                utterance_id=uid,
                prev_ipa=prev_ip,
                next_ipa=next_ip,
                log_env=env_norm[f0_i:f1_i].astype(np.float32),
                bap=feats.bap[f0_i:f1_i].astype(np.float32),
                voiced=voi_sl.astype(bool),
                f0_rel_st=f0_rel[f0_i:f1_i].astype(np.float32),
                dur_ms=float((f1_i - f0_i) * HOP / sr * 1000.0),
                raw_wav=raw_w,
            )
            en_phones.setdefault(ipa, []).append(tok)

    return DualDonorBank(
        zh_bank=zh_bank,
        en_phones=en_phones,
        en_profiles_v2=en_profs_v2,
        zh_profiles_v2=zh_profs_v2,
        en_included_ids=sorted(en_data.keys()),
        en_excluded_ids=sorted(en_excl),
    )


def filter_dual_bank_excluding_speakers(
    bank: DualDonorBank,
    excluded_speakers: Iterable[str],
) -> DualDonorBank:
    """Return a zero-copy view of DualDonorBank excluding specified English donor speakers."""
    excl_spk = set(excluded_speakers)
    filt_phones: dict[str, list[EnPhoneToken]] = {}
    kept_ids: set[str] = set()
    removed_ids: set[str] = set(bank.en_excluded_ids)
    for ipa, toks in bank.en_phones.items():
        kept = [t for t in toks if t.speaker not in excl_spk]
        if kept:
            filt_phones[ipa] = kept
            for t in kept:
                kept_ids.add(t.utterance_id)
        for t in toks:
            if t.speaker in excl_spk:
                removed_ids.add(t.utterance_id)
    return DualDonorBank(
        zh_bank=bank.zh_bank,
        en_phones=filt_phones,
        en_profiles_v2={k: v for k, v in bank.en_profiles_v2.items() if k not in excl_spk},
        zh_profiles_v2=dict(bank.zh_profiles_v2),
        en_included_ids=sorted(kept_ids),
        en_excluded_ids=sorted(removed_ids),
    )


def _get_src_v2_profile(bank: DualDonorBank, spk: str) -> tc2.TimbreProfileV2 | None:
    if spk in bank.en_profiles_v2:
        return bank.en_profiles_v2[spk]
    if spk in bank.zh_profiles_v2:
        return bank.zh_profiles_v2[spk]
    return None


def _adapt_with_v2(
    env: np.ndarray,
    bap: np.ndarray,
    voi: np.ndarray,
    f0_rel_st: np.ndarray,
    src_prof: tc2.TimbreProfileV2 | None,
    tgt_prof: tc2.TimbreProfileV2,
    vtln_alpha: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if src_prof is None or src_prof.name == tgt_prof.name:
        return env.astype(np.float64), bap.astype(np.float64), voi.copy()
    dummy_f0 = np.where(voi, src_prof.f0_median_hz * (2.0 ** (f0_rel_st / 12.0)), 0.0)
    feats = vocoder.VocoderFeatures(
        f0_hz=dummy_f0,
        voiced=voi.copy(),
        log_env=env.astype(np.float64),
        bap=bap.astype(np.float64),
        sr=SR,
        hop_samples=HOP,
        n_samples=len(env) * HOP,
        n_fft=N_FFT,
    )
    ad, _ = tc2.adapt_features_v2(feats, src_prof, tgt_prof, vtln_alpha=vtln_alpha)
    return ad.log_env, ad.bap, ad.voiced


def render_dual_utau_plan(
    plan_doc: dict,
    bank: DualDonorBank,
    target_profile: tc2.TimbreProfileV2,
    *,
    vtln_alpha: float | None = None,
    seed: int = 72,
    phase_dispersion: bool = True,
    use_waveform_unvoiced: bool = False,
) -> tuple[np.ndarray, dict]:
    """Render ZH / EN / Code-Switched text using Dual-Donor UTAU Concat + V2 Decoupled Cloning."""
    units = plan_doc.get("units") or []
    if not units:
        raise ValueError("plan_doc has no units")

    envs, baps, vois, sts = [], [], [], []
    is_obs_head, is_pause_blk = [], []
    wav_overlays: list[tuple[int, int, np.ndarray]] = []
    source_counts: dict[str, int] = {}
    iqr_scale = max(0.55, min(1.75, float(target_profile.f0_iqr_st) / 3.2))

    # Build flat ZH context info & full EN IPA list for context matching
    parsed = []
    for u in units:
        lang = u.get("lang", "zh")
        if lang == "zh":
            o = str(u.get("onset", ""))
            f = str(u.get("final", ""))
            if o in ("y", "w"):
                o = ""
            if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
                f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"
            parsed.append(("zh", o, f, int(u.get("tone") or 1)))
        else:
            parsed.append((lang, "", "", int(u.get("tone") or 1)))

    # 20 ms lead silence
    lead_fr = 4
    envs.append(np.full((lead_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((lead_fr, 5), dtype=np.float64))
    vois.append(np.zeros(lead_fr, dtype=bool))
    sts.append(np.zeros(lead_fr, dtype=np.float64))
    is_obs_head.append(False)
    is_pause_blk.append(True)
    cur_fr = lead_fr

    zh_b = bank.zh_bank
    for u_idx, u in enumerate(units):
        lang, o, f, tone = parsed[u_idx]
        dur_ms = max(35.0, float(u.get("duration_ms", 200.0)))
        nf = max(2, int(round(dur_ms / 5.0)))
        gain = float(u.get("gain", 0.85))
        pitch_5 = [float(x) for x in u.get("pitch_st", [0.0] * 5)]
        pause_ms = float(u.get("pause_after_ms", 0.0))

        if lang == "zh":
            prev_f = parsed[u_idx - 1][2] if (u_idx > 0 and parsed[u_idx - 1][0] == "zh") else "SIL"
            next_o = parsed[u_idx + 1][1] if (u_idx + 1 < len(units) and parsed[u_idx + 1][0] == "zh") else "SIL"
            cands = zh_b.syllables.get((o, f), [])
            if cands:
                best = max(cands, key=lambda c: _score_candidate(c, target_profile.name, tone, prev_f, next_o, dur_ms))
                e_raw, b_raw, v_raw = _adapt_with_v2(
                    best.log_env, best.bap, best.voiced, best.f0_rel_st,
                    _get_src_v2_profile(bank, best.speaker), target_profile, vtln_alpha,
                )
                mode = "zh_target_syl" if best.speaker == target_profile.name else "zh_donor_syl"
                source_counts[mode] = source_counts.get(mode, 0) + 1
                e_st, b_st, v_st, f_sp_new = _stretch_preserving_onset(
                    e_raw, b_raw, v_raw, best.f_split, nf, preserve_onset=True
                )
                if use_waveform_unvoiced and best.onset_wav is not None and f_sp_new > 0:
                    wav_overlays.append((cur_fr, cur_fr + f_sp_new, best.onset_wav))
            else:
                # Fallback to ZH onset+final concat or IPA synth-fill
                parts_e, parts_b, parts_v = [], [], []
                split_pt = 0
                if o:
                    oc = zh_b.onsets.get(o, [])
                    if oc:
                        bo = max(oc, key=lambda c: _score_candidate(c, target_profile.name, tone, prev_f, f, dur_ms * 0.28))
                        eo, bo_b, vo = _adapt_with_v2(
                            bo.log_env, bo.bap, bo.voiced, bo.f0_rel_st,
                            _get_src_v2_profile(bank, bo.speaker), target_profile, vtln_alpha,
                        )
                        parts_e.append(eo); parts_b.append(bo_b); parts_v.append(vo)
                        split_pt = len(eo)
                fc = zh_b.finals.get(f, [])
                if fc:
                    bf = max(fc, key=lambda c: _score_candidate(c, target_profile.name, tone, o or prev_f, next_o, dur_ms * 0.72))
                    ef, bf_b, vf = _adapt_with_v2(
                        bf.log_env, bf.bap, bf.voiced, bf.f0_rel_st,
                        _get_src_v2_profile(bank, bf.speaker), target_profile, vtln_alpha,
                    )
                    parts_e.append(ef); parts_b.append(bf_b); parts_v.append(vf)
                    source_counts["zh_concat_of"] = source_counts.get("zh_concat_of", 0) + 1
                else:
                    for fi in cf.PINYIN_FINAL_TO_IPA.get(f, ("ə",)):
                        d = cw.lookup_ipa_template(zh_b.fallback_unit_bank, fi, is_onset=False)
                        ed, bd, vd = _adapt_with_v2(
                            d["log_env"] - 3.1, d["bap"], d["voiced"], np.zeros(len(d["log_env"])),
                            tc2.extract_v2_profile([], sr=SR) if False else _get_src_v2_profile(bank, "SSB0502"),
                            target_profile, vtln_alpha,
                        )
                        parts_e.append(ed); parts_b.append(bd); parts_v.append(vd)
                    source_counts["zh_ipa_fill"] = source_counts.get("zh_ipa_fill", 0) + 1
                e_st, b_st, v_st, _ = _stretch_preserving_onset(
                    np.concatenate(parts_e, axis=0),
                    np.concatenate(parts_b, axis=0),
                    np.concatenate(parts_v, axis=0),
                    split_pt, nf, preserve_onset=True,
                )
            if abs(gain - 0.85) > 0.05:
                e_st = e_st + math.log(max(0.25, min(1.8, gain / 0.85)))
            t_grid = np.linspace(0.0, 1.0, len(e_st))
            st_u = np.array([_interp_list(pitch_5, float(t)) for t in t_grid], dtype=np.float64) * iqr_scale
            envs.append(e_st); baps.append(b_st); vois.append(v_st); sts.append(st_u)
            is_obs_head.append(o in OBSTRUENT_ONSETS)
            is_pause_blk.append(False)
            cur_fr += len(e_st)
        else:
            # English unit: concatenate REAL English phone tokens from DualDonorBank (Target-First!), synth-fill only if missing!
            ipas = list(u.get("ipa_phones") or ["ə"])
            arpa = u.get("arpabet") or []
            pairs = v6_arb._split_en_arpabet_to_ipa_with_stress(arpa) if arpa else [(p, None) for p in ipas]
            if not pairs:
                pairs = [(p, None) for p in ipas]
            n_p = len(pairs)
            is_vow = [st is not None for _, st in pairs]

            chosen = []
            raw_durs = []
            for idx_p, (ip, st) in enumerate(pairs):
                prev_ip = pairs[idx_p - 1][0] if idx_p > 0 else "SIL"
                next_ip = pairs[idx_p + 1][0] if idx_p + 1 < n_p else "SIL"
                cands = bank.en_phones.get(ip, [])
                if cands:
                    def _en_score(c: EnPhoneToken) -> float:
                        sc = 0.0
                        if c.speaker == target_profile.name:
                            sc += 5.0
                        if c.prev_ipa == prev_ip:
                            sc += 1.5
                        if c.next_ipa == next_ip:
                            sc += 1.5
                        return sc
                    best = max(cands, key=_en_score)
                    e_r, b_r, v_r = _adapt_with_v2(
                        best.log_env, best.bap, best.voiced, best.f0_rel_st,
                        _get_src_v2_profile(bank, best.speaker), target_profile, vtln_alpha,
                    )
                    mode = "en_target_phone" if best.speaker == target_profile.name else "en_donor_phone"
                    source_counts[mode] = source_counts.get(mode, 0) + 1
                    d_ms = best.dur_ms
                    w_raw = best.raw_wav
                else:
                    # Strictly fill missing IPA phonemes using warped/synth template
                    d = cw.lookup_ipa_template(zh_b.fallback_unit_bank, ip, is_onset=(idx_p == 0))
                    src_p = _get_src_v2_profile(bank, "SSB0502") or next(iter(bank.en_profiles_v2.values()))
                    e_r, b_r, v_r = _adapt_with_v2(
                        d["log_env"] - 3.1, d["bap"], d["voiced"], np.zeros(len(d["log_env"])),
                        src_p, target_profile, vtln_alpha,
                    )
                    source_counts["en_synth_fill"] = source_counts.get("en_synth_fill", 0) + 1
                    d_ms = d["dur_ms"]
                    w_raw = None
                if st == 1:
                    d_ms *= 1.15
                elif st == 0:
                    d_ms = max(35.0, d_ms * 0.78)
                chosen.append((ip, e_r, b_r, v_r, w_raw))
                raw_durs.append(max(20.0, d_ms))

            # Render each phone as its own sub-slot with manner-aware seams (0 ms on obstruents, 10 ms on vowels)
            tot_d = max(1.0, sum(raw_durs))
            acc_f = 0
            for idx_p, ((ip, e_r, b_r, v_r, w_raw), d_ms) in enumerate(zip(chosen, raw_durs)):
                kp = max(2, nf - acc_f) if (idx_p + 1 == n_p) else max(2, min(nf - acc_f - 2 * (n_p - idx_p - 1), int(round(nf * d_ms / tot_d))))
                acc_f += kp
                ep = _resample_2d(e_r, kp)
                bp = np.clip(_resample_2d(b_r, kp), 0.005, 0.995)
                vp = _resample_1d(v_r.astype(np.float64), kp) >= 0.45
                if abs(gain - 0.85) > 0.05:
                    ep = ep + math.log(max(0.25, min(1.8, gain / 0.85)))
                sub_p = [_interp_list(pitch_5, (idx_p + m / 4.0) / n_p) for m in range(5)]
                tp = np.linspace(0.0, 1.0, kp)
                st_p = np.array([_interp_list(sub_p, float(t)) for t in tp], dtype=np.float64) * iqr_scale
                if use_waveform_unvoiced and w_raw is not None and not np.any(vp):
                    wav_overlays.append((cur_fr, cur_fr + kp, w_raw))
                envs.append(ep); baps.append(bp); vois.append(vp); sts.append(st_p)
                is_obs_head.append(ip in OBSTRUENT_IPAS)
                is_pause_blk.append(False)
                cur_fr += kp

        if pause_ms > 15.0:
            p_fr = max(2, int(round(pause_ms / 5.0)))
            envs.append(np.full((p_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
            baps.append(np.ones((p_fr, 5), dtype=np.float64))
            vois.append(np.zeros(p_fr, dtype=bool))
            sts.append(np.zeros(p_fr, dtype=np.float64))
            is_obs_head.append(False)
            is_pause_blk.append(True)
            cur_fr += p_fr

    # 30 ms tail silence
    tail_fr = 6
    envs.append(np.full((tail_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((tail_fr, 5), dtype=np.float64))
    vois.append(np.zeros(tail_fr, dtype=bool))
    sts.append(np.zeros(tail_fr, dtype=np.float64))
    is_obs_head.append(False)
    is_pause_blk.append(True)

    # Manner-aware seams: 0 ms on obstruents/V-UV edges, 10 ms on voiced vowel/sonorant joins
    for i in range(len(envs) - 1):
        if is_pause_blk[i] or is_pause_blk[i + 1]:
            continue
        if not is_obs_head[i + 1] and np.any(vois[i]) and np.any(vois[i + 1]):
            _blend_seam(envs[i], envs[i + 1], baps[i], baps[i + 1], blend_frames=2)

    env_all = np.concatenate(envs, axis=0)
    bap_all = np.clip(np.concatenate(baps, axis=0), 0.005, 0.995)
    voi_all = np.concatenate(vois, axis=0)
    st_all = np.concatenate(sts, axis=0)

    v_idx = np.where(voi_all)[0]
    if v_idx.size > 0:
        st_centered = st_all[v_idx] - float(np.median(st_all[v_idx]))
        prog = np.linspace(0.5, -0.5, len(st_all))[v_idx]
        st_v = st_centered + prog
        if len(st_v) >= 3:
            sm = st_v.copy()
            sm[1:-1] = 0.25 * st_v[:-2] + 0.50 * st_v[1:-1] + 0.25 * st_v[2:]
            st_v = sm
        f0_all = np.zeros(len(st_all), dtype=np.float64)
        f0_all[v_idx] = np.clip(target_profile.f0_median_hz * (2.0 ** (st_v / 12.0)), 65.0, 440.0)
    else:
        f0_all = np.zeros(len(st_all), dtype=np.float64)

    feats = vocoder.VocoderFeatures(
        f0_hz=f0_all,
        voiced=voi_all,
        log_env=env_all,
        bap=bap_all,
        sr=SR,
        hop_samples=HOP,
        n_samples=len(env_all) * HOP,
        n_fft=N_FFT,
    )
    y = vocoder.synthesize(feats, seed=seed, phase_dispersion=phase_dispersion)
    if use_waveform_unvoiced and wav_overlays:
        fade_n = 48
        for fr_s, fr_e, on_wav in wav_overlays:
            s0, s1 = fr_s * HOP, min(len(y), fr_e * HOP)
            seg_len = s1 - s0
            if seg_len < 60 or len(on_wav) < 60:
                continue
            src = on_wav.astype(np.float64)
            patch = src[-seg_len:].copy() if len(src) >= seg_len else np.interp(np.linspace(0, 1, seg_len), np.linspace(0, 1, len(src)), src)
            r_syn = float(np.sqrt(np.mean(y[s0:s1] ** 2)))
            r_pat = float(np.sqrt(np.mean(patch ** 2)))
            if r_syn > 1e-5 and r_pat > 1e-5:
                patch *= r_syn / r_pat
            fn = min(fade_n, seg_len // 3)
            if fn > 0:
                w_in = np.linspace(0.0, 1.0, fn)
                y[s0 : s0 + fn] = (1.0 - w_in) * y[s0 : s0 + fn] + w_in * patch[:fn]
                y[s0 + fn : s1 - fn] = patch[fn : seg_len - fn]
                w_out = np.linspace(1.0, 0.0, fn)
                y[s1 - fn : s1] = (1.0 - w_out) * y[s1 - fn : s1] + w_out * patch[seg_len - fn :]

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.89:
        y *= 0.89 / peak
    return y, {"source_counts": source_counts, "n_frames": len(env_all)}


def _loudness_norm(x: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    act = arr[np.abs(arr) > 0.01 * max(1e-6, float(np.max(np.abs(arr))))]
    r = float(np.sqrt(np.mean(act * act))) if act.size > 0 else float(np.sqrt(np.mean(arr * arr)))
    if r > 1e-8:
        arr = arr * (target_rms / r)
    pk = float(np.max(np.abs(arr))) if arr.size else 0.0
    if pk > 0.90:
        arr = arr * (0.90 / pk)
    return arr


def _eval_against_ref(y: np.ndarray, ref: np.ndarray, ev_ref: dict, rtf: float) -> dict:
    ev_y = ev.evaluate_audio(y, sr=SR)
    fl_y = fl.compute_fluency_metrics(y, sr=SR)
    cl_y = compute_clarity_metrics(y, sr=SR)
    sd = ev.compute_spectral_distances(ref, y, sr=SR)
    f0_y = ev_y["f0_hnr"]["f0_median_hz"]
    f0_r = ev_ref["f0_hnr"]["f0_median_hz"]
    f0_err = abs(12.0 * math.log2(f0_y / f0_r)) if (f0_y and f0_r and math.isfinite(f0_y) and math.isfinite(f0_r)) else float("nan")
    return {
        "sanity_ok": bool(ev_y["sanity"]["ok"]),
        "mel_lsd_db": float(sd["mel_lsd_db"]),
        "mcd_db": float(sd["mcd_db"]),
        "mcd_cmn_db": float(sd["mcd_cmn_db"]),
        "tilt_err_db_oct": abs(ev_y["ltas"]["tilt_db_oct"] - ev_ref["ltas"]["tilt_db_oct"]),
        "f0_err_st": float(f0_err),
        "hnr_err_db": abs(ev_y["f0_hnr"]["hnr_db_median"] - ev_ref["f0_hnr"]["hnr_db_median"]),
        "f0_hz": float(f0_y),
        "tilt_db_oct": float(ev_y["ltas"]["tilt_db_oct"]),
        "hnr_db": float(ev_y["f0_hnr"]["hnr_db_median"]),
        "flux_p95_med": float(fl_y["spectral_flux"]["spectral_flux_p95_to_median"]),
        "vuv_flux_db": float(fl_y["vuv_transitions"]["vuv_flux_median_db"]),
        "rtf": float(rtf),
        **cl_y,
    }


def run_dual_eval(export_samples: bool = True) -> dict:
    """Evaluate Intra-lingual EN->EN (held-out slt[30:40], bdl[30:40]) & Cross-lingual ZH<->EN."""
    t0 = time.time()
    zh_utts = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
    man_all = json.loads(EN_MANIFEST.read_text(encoding="utf-8"))
    en_man = man_all["english"]["utterances"]
    zh_new_man = man_all.get("mandarin", {}).get("utterances", [])

    # Strict Zero-Leakage Split: train English on [0:30] per speaker, hold out [30:40] of slt & bdl!
    en_heldout = [u for u in en_man if u["speaker"] in ("slt", "bdl") and int(u["id"].split("_")[1]) >= 30]
    en_excl_ids = {u["id"] for u in en_heldout}
    zh_held_ids = {"SSB05020122", "SSB05020137", "SSB05020159", "SSB15930193", "SSB15930204"}

    bank = build_dual_donor_bank(
        zh_utterances=zh_utts,
        zh_exclude_ids=zh_held_ids,
        en_utterances=en_man,
        en_exclude_ids=en_excl_ids,
    )
    # Also build v7 Mandarin-only bank for apples-to-apples comparison on held-out English (warped_ipa baseline)
    v7_zh_bank = v7_ub.build_unit_bank(zh_utts, exclude_ids=zh_held_ids)

    # 3-shot few-shot V2 profiles (strictly from training/prompt utterances!)
    def _fewshot_v2(rows: list[dict], spk: str) -> tc2.TimbreProfileV2:
        items = []
        for r in rows[:3]:
            x, _ = vocoder.read_wav_mono(EXP_DIR.parent / r["path"])
            s0, s1 = _trim_active_span(x, sr=SR)
            items.append((x[s0:s1], vocoder.analyze(x[s0:s1], sr=SR)))
        return tc2.extract_v2_profile(items, sr=SR, name=spk)

    prof_slt_3s = _fewshot_v2([u for u in en_man if u["speaker"] == "slt" and u["id"] not in en_excl_ids], "slt")
    prof_bdl_3s = _fewshot_v2([u for u in en_man if u["speaker"] == "bdl" and u["id"] not in en_excl_ids], "bdl")
    prof_0502_3s = bank.zh_profiles_v2["SSB0502"]
    prof_1593_3s = bank.zh_profiles_v2["SSB1593"]
    ssb0073_rows = [u for u in zh_new_man if u["speaker"] == "SSB0073"]
    prof_0073_3s = _fewshot_v2(ssb0073_rows, "SSB0073") if ssb0073_rows else None

    # Pre-cache held-out English real waveforms & metrics
    en_real_cache = {}
    for u in en_heldout:
        x, _ = vocoder.read_wav_mono(EXP_DIR.parent / u["path"])
        s0, s1 = _trim_active_span(x, sr=SR)
        xa = x[s0:s1]
        en_real_cache[u["id"]] = (xa, ev.evaluate_audio(xa, sr=SR))

    # 1. Intra-lingual EN -> EN on held-out slt[30:40] & bdl[30:40]
    en_modes = ("real_en", "v7_warped_en", "v8_utau_en_target_first", "v8_utau_en_cross_donor")
    en_records: dict[str, list[dict]] = {m: [] for m in en_modes}
    en_source_counts: dict[str, dict[str, int]] = {"v8_utau_en_target_first": {}, "v8_utau_en_cross_donor": {}}

    # Cross-donor views excluding the target speaker completely to test cross-gender EN->EN without re-analyzing WAVs
    bank_no_bdl = filter_dual_bank_excluding_speakers(bank, {"bdl"})
    bank_no_slt = filter_dual_bank_excluding_speakers(bank, {"slt"})

    ab_en_clips = {}
    for u in en_heldout:
        uid = u["id"]
        spk = u["speaker"]
        xa, ev_r = en_real_cache[uid]
        prof_tgt = prof_slt_3s if spk == "slt" else prof_bdl_3s
        plan_en = cf.plan(u["text"], allow_oov_fallback=True)
        # Scale unit durations slightly to align smoothly while keeping natural rhythm
        en_records["real_en"].append(_eval_against_ref(xa, xa, ev_r, 0.0))

        # v7 Mandarin-warped EN
        t_s = time.perf_counter()
        y_v7, _ = v7_pb.synthesize_plan(plan_en, v7_zh_bank, prof_tgt.to_v6_profile(), vtln_alpha=(0.86 if spk == "bdl" else 1.0), seed=72)
        en_records["v7_warped_en"].append(_eval_against_ref(y_v7, xa, ev_r, (time.perf_counter() - t_s) / max(1e-6, len(y_v7) / SR)))

        # v8 UTAU EN (Target-First, full bank excluding held-out)
        t_s = time.perf_counter()
        y_v8, meta_v8 = render_dual_utau_plan(plan_en, bank, prof_tgt, seed=72)
        en_records["v8_utau_en_target_first"].append(_eval_against_ref(y_v8, xa, ev_r, (time.perf_counter() - t_s) / max(1e-6, len(y_v8) / SR)))
        for k, v in meta_v8["source_counts"].items():
            en_source_counts["v8_utau_en_target_first"][k] = en_source_counts["v8_utau_en_target_first"].get(k, 0) + v

        # v8 UTAU EN Cross-Donor (e.g. bdl cloned from slt/clb/rms, slt cloned from bdl/clb/rms via V2 auto_vtln)
        cd_bank = bank_no_bdl if spk == "bdl" else bank_no_slt
        t_s = time.perf_counter()
        y_cd, meta_cd = render_dual_utau_plan(plan_en, cd_bank, prof_tgt, seed=72)
        en_records["v8_utau_en_cross_donor"].append(_eval_against_ref(y_cd, xa, ev_r, (time.perf_counter() - t_s) / max(1e-6, len(y_cd) / SR)))
        for k, v in meta_cd["source_counts"].items():
            en_source_counts["v8_utau_en_cross_donor"][k] = en_source_counts["v8_utau_en_cross_donor"].get(k, 0) + v

        if export_samples and uid in ("slt_0030", "bdl_0030"):
            ab_en_clips[uid] = {"real": xa, "v7_warped": y_v7, "v8_utau": y_v8, "v8_cross_donor": y_cd}

    # 2. Cross-Lingual ZH -> EN (SSB0502, SSB1593, and Male SSB0073 speaking EN) and EN/ZH-Male -> ZH
    zh_held_sample = [u for u in zh_utts if u["id"] in zh_held_ids]
    en_sample_5 = [u for u in en_heldout if u["speaker"] == "slt"][:5]

    cross_cases = [
        ("SSB0502_speaks_EN", prof_0502_3s, en_sample_5, "en"),
        ("SSB1593_speaks_EN", prof_1593_3s, en_sample_5, "en"),
        ("slt_F189_speaks_ZH", prof_slt_3s, zh_held_sample, "zh"),
        ("bdl_M122_speaks_ZH", prof_bdl_3s, zh_held_sample, "zh"),
    ]
    if prof_0073_3s is not None:
        cross_cases.extend([
            ("SSB0073_M133_speaks_EN", prof_0073_3s, en_sample_5, "en"),
            ("SSB0073_M133_speaks_ZH", prof_0073_3s, zh_held_sample, "zh"),
        ])

    cross_summary = {}
    for label, tgt_prof, texts_src, kind in cross_cases:
        rows_c = []
        src_c: dict[str, int] = {}
        for item in texts_src:
            if kind == "en":
                pl = cf.plan(item["text"], allow_oov_fallback=True)
            else:
                pl = v6_arb.plan_from_pinyin_syllables(item["syllables"], text=item["text"])
            t_s = time.perf_counter()
            y_c, meta_c = render_dual_utau_plan(pl, bank, tgt_prof, seed=72)
            rtf_c = (time.perf_counter() - t_s) / max(1e-6, len(y_c) / SR)
            for k, v in meta_c["source_counts"].items():
                src_c[k] = src_c.get(k, 0) + v
            ev_c = ev.evaluate_audio(y_c, sr=SR)
            fl_c = fl.compute_fluency_metrics(y_c, sr=SR)
            cl_c = compute_clarity_metrics(y_c, sr=SR)
            f0_c = ev_c["f0_hnr"]["f0_median_hz"]
            rows_c.append({
                "sanity_ok": bool(ev_c["sanity"]["ok"]),
                "f0_err_st": abs(12.0 * math.log2(f0_c / tgt_prof.f0_median_hz)) if (f0_c and math.isfinite(f0_c)) else float("nan"),
                "tilt_err_db_oct": abs(ev_c["ltas"]["tilt_db_oct"] - tgt_prof.tilt_db_per_oct),
                "hnr_err_db": abs(ev_c["f0_hnr"]["hnr_db_median"] - tgt_prof.hnr_median_db),
                "f0_hz": float(f0_c),
                "tilt_db_oct": float(ev_c["ltas"]["tilt_db_oct"]),
                "flux_p95_med": float(fl_c["spectral_flux"]["spectral_flux_p95_to_median"]),
                "vuv_flux_db": float(fl_c["vuv_transitions"]["vuv_flux_median_db"]),
                "spectral_contrast_db": cl_c["spectral_contrast_db"],
                "uv_hf_lf_db": cl_c["unvoiced_hf_lf_db"],
                "rtf": rtf_c,
                "_wav": y_c,
            })
        cross_summary[label] = {
            "n_utts": len(rows_c),
            "sanity_ok": sum(1 for r in rows_c if r["sanity_ok"]),
            "target_f0_hz": round(tgt_prof.f0_median_hz, 1),
            "target_tilt_db_oct": round(tgt_prof.tilt_db_per_oct, 2),
            "out_f0_hz_med": round(float(np.nanmedian([r["f0_hz"] for r in rows_c])), 1),
            "f0_err_st_med": round(float(np.nanmedian([r["f0_err_st"] for r in rows_c])), 3),
            "out_tilt_db_oct_med": round(float(np.median([r["tilt_db_oct"] for r in rows_c])), 2),
            "tilt_err_db_oct_med": round(float(np.median([r["tilt_err_db_oct"] for r in rows_c])), 2),
            "hnr_err_db_med": round(float(np.median([r["hnr_err_db"] for r in rows_c])), 2),
            "spectral_contrast_db": round(float(np.median([r["spectral_contrast_db"] for r in rows_c])), 2),
            "uv_hf_lf_db": round(float(np.median([r["uv_hf_lf_db"] for r in rows_c])), 2),
            "flux_p95_med": round(float(np.median([r["flux_p95_med"] for r in rows_c])), 3),
            "vuv_flux_db": round(float(np.median([r["vuv_flux_db"] for r in rows_c])), 2),
            "rtf_mean": round(float(np.mean([r["rtf"] for r in rows_c])), 4),
            "source_counts": src_c,
            "_first_wav": rows_c[0]["_wav"],
        }

    # Export A/B listening WAVs
    exported = []
    if export_samples:
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        gap = np.zeros(int(0.45 * SR), dtype=np.float64)
        for uid, clips in ab_en_clips.items():
            pack = np.concatenate([
                _loudness_norm(clips["real"]), gap,
                _loudness_norm(clips["v7_warped"]), gap,
                _loudness_norm(clips["v8_utau"]), gap,
                _loudness_norm(clips["v8_cross_donor"]),
            ])
            p_pack = SAMPLES_DIR / f"AB_EN_{uid}_real_v7warped_v8utau_v8xdom.wav"
            vocoder.write_wav_mono(p_pack, pack, sr=SR)
            exported.append(p_pack.name)

        # Cross-lingual packs
        x_0502_ref, _ = vocoder.read_wav_mono([u for u in zh_utts if u["id"] == "SSB05020005"][0]["path"])
        p_c1 = SAMPLES_DIR / "AB_CROSS_SSB0502_zhRef_speaks_EN.wav"
        vocoder.write_wav_mono(p_c1, np.concatenate([_loudness_norm(x_0502_ref[:3*SR]), gap, _loudness_norm(cross_summary["SSB0502_speaks_EN"].pop("_first_wav"))]), sr=SR)
        exported.append(p_c1.name)

        x_bdl_ref, _ = vocoder.read_wav_mono(EXP_DIR.parent / [u for u in en_man if u["id"] == "bdl_0000"][0]["path"])
        p_c2 = SAMPLES_DIR / "AB_CROSS_bdl_M122_enRef_speaks_ZH.wav"
        vocoder.write_wav_mono(p_c2, np.concatenate([_loudness_norm(x_bdl_ref[:3*SR]), gap, _loudness_norm(cross_summary["bdl_M122_speaks_ZH"].pop("_first_wav"))]), sr=SR)
        exported.append(p_c2.name)
        if ssb0073_rows:
            x_0073_ref, _ = vocoder.read_wav_mono(EXP_DIR.parent / ssb0073_rows[0]["path"])
            p_c3 = SAMPLES_DIR / "AB_CROSS_SSB0073_M133_zhRef_speaks_EN.wav"
            vocoder.write_wav_mono(p_c3, np.concatenate([_loudness_norm(x_0073_ref[:3*SR]), gap, _loudness_norm(cross_summary["SSB0073_M133_speaks_EN"].pop("_first_wav"))]), sr=SR)
            exported.append(p_c3.name)
            p_c4 = SAMPLES_DIR / "AB_CROSS_SSB0073_M133_zhRef_speaks_ZH.wav"
            vocoder.write_wav_mono(p_c4, np.concatenate([_loudness_norm(x_0073_ref[:3*SR]), gap, _loudness_norm(cross_summary["SSB0073_M133_speaks_ZH"].pop("_first_wav"))]), sr=SR)
            exported.append(p_c4.name)
        cross_summary["SSB1593_speaks_EN"].pop("_first_wav", None)
        cross_summary["slt_F189_speaks_ZH"].pop("_first_wav", None)
    else:
        for v in cross_summary.values():
            v.pop("_first_wav", None)

    keys = ("mel_lsd_db", "mcd_db", "mcd_cmn_db", "spectral_contrast_db", "unvoiced_hf_lf_db",
            "transient_hf_p95_db", "flux_p95_med", "vuv_flux_db", "f0_err_st", "tilt_err_db_oct", "hnr_err_db", "rtf")
    en_summary = {}
    for m, recs in en_records.items():
        en_summary[m] = {
            "n_utterances": len(recs),
            "sanity_ok": sum(1 for r in recs if r["sanity_ok"]),
            "medians": {k: round(float(np.nanmedian([r[k] for r in recs])), 3) for k in keys},
        }

    out = {
        "schema": 1,
        "elapsed_s": round(time.time() - t0, 2),
        "en_bank_stats": {
            "en_included_utts": len(bank.en_included_ids),
            "en_excluded_utts": len(bank.en_excluded_ids),
            "unique_en_ipas": len(bank.en_phones),
            "total_en_phone_tokens": sum(len(v) for v in bank.en_phones.values()),
        },
        "intra_en_eval": en_summary,
        "en_source_counts": en_source_counts,
        "cross_lingual_eval": cross_summary,
        "exported_samples": exported,
    }
    (EXP_DIR / "dual_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run_dual_eval(export_samples=True)
    print(f"Completed Dual-Donor EN+ZH Evaluation in {res['elapsed_s']}s")
    print("EN bank:", res["en_bank_stats"])
    print("\n=== Intra-Lingual EN->EN (n=20 Held-Out slt[30:40] & bdl[30:40], Zero Leakage) ===")
    for m, info in res["intra_en_eval"].items():
        d = info["medians"]
        print(
            f"  [{m:24s}] ok={info['sanity_ok']}/{info['n_utterances']} | Mel-LSD={d['mel_lsd_db']} dB | "
            f"MCD={d['mcd_db']} [CMVN {d['mcd_cmn_db']}] | contrast={d['spectral_contrast_db']} dB | "
            f"uv_HF/LF={d['unvoiced_hf_lf_db']} dB | F0_err={d['f0_err_st']} st | tilt_err={d['tilt_err_db_oct']} | RTF={d['rtf']}"
        )
    print("EN source counts:", res["en_source_counts"])
    print("\n=== Cross-Lingual ZH->EN & EN->ZH ===")
    for k, v in res["cross_lingual_eval"].items():
        print(
            f"  [{k:20s}] ok={v['sanity_ok']}/{v['n_utts']} | F0 {v['target_f0_hz']}->{v['out_f0_hz_med']} Hz (err {v['f0_err_st_med']} st) | "
            f"tilt {v['target_tilt_db_oct']}->{v['out_tilt_db_oct_med']} (err {v['tilt_err_db_oct_med']}) | "
            f"HNR_err={v['hnr_err_db_med']} dB | contrast={v['spectral_contrast_db']} dB | uv_HF/LF={v['uv_hf_lf_db']} dB | "
            f"flux_p95/med={v['flux_p95_med']}x | srcs={v['source_counts']}"
        )
    print("Exported:", res["exported_samples"])
