"""Viterbi Context-Aware UTAU Concat + Natural Prosody + HD HNM Renderer (100% Non-Neural)."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Callable, Iterable

import numpy as np

from pybot_tts_v2_recorded import engine
from pybot_tts_v4_params import vocal
from exp_v5_arch import crossling_frontend as cf
from exp_v6_crossling_dsp import arbiter as v6_arb
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS
from exp_v6_vocoder import vocoder
from exp_v7_pybot_tts import crossling_warp as cw, unit_bank as ub
from exp_v8_clarity.en_utau_bank import _trim_active_span, _utterance_ipas_from_text
from exp_v8_clarity.utau_concat import (
    N_BINS,
    OBSTRUENT_IPAS,
    OBSTRUENT_ONSETS,
    SILENCE_LOG_ENV,
    _blend_seam,
    _interp_list,
    _resample_1d,
    _resample_2d,
    _stretch_preserving_onset,
)
from exp_v8_timbre_clone import timbre_cloner_v2 as tc2
from exp_v9_clarity_pro import EN_MANIFEST, EXTENDED_INDEX, HOP, N_FFT, SR
from exp_v9_clarity_pro.boundary_refiner import (
    align_en_phones_with_landmarks,
    compute_landmark_tracks,
    refine_zh_syllable_bounds,
)
from exp_v9_clarity_pro.hnm_v9 import analyze_v9, synthesize_v9


ClonerFn = Callable[..., tuple[vocoder.VocoderFeatures, dict]]


@dataclass
class V9ZhToken:
    onset: str
    final: str
    tone: int
    speaker: str
    utterance_id: str
    prev_final: str
    next_onset: str
    log_env: np.ndarray        # (T, 513) float32
    bap: np.ndarray            # (T, 5) float32
    bap_gt8k: np.ndarray       # (T, 2) float32
    voiced: np.ndarray         # (T,) bool
    f0_rel_st: np.ndarray      # (T,) float32
    f_split: int
    dur_ms: float
    cmvn_left: np.ndarray      # (513,) float32 mean-normed entry envelope
    cmvn_right: np.ndarray     # (513,) float32 mean-normed exit envelope
    f0_edge_l: float
    f0_edge_r: float
    uv_hf_lf: float = 0.0
    onset_wav: np.ndarray | None = None


@dataclass
class V9EnToken:
    ipa: str
    speaker: str
    utterance_id: str
    prev_ipa: str
    next_ipa: str
    stress: int | None
    log_env: np.ndarray        # (T, 513) float32
    bap: np.ndarray            # (T, 5) float32
    bap_gt8k: np.ndarray       # (T, 2) float32
    voiced: np.ndarray         # (T,) bool
    f0_rel_st: np.ndarray      # (T,) float32
    dur_ms: float
    cmvn_left: np.ndarray      # (513,) float32
    cmvn_right: np.ndarray     # (513,) float32
    f0_edge_l: float
    f0_edge_r: float
    uv_hf_lf: float = 0.0
    raw_wav: np.ndarray | None = None


@dataclass
class ProsodyRules:
    zh_final_lengthen: float = 1.22
    en_final_lengthen: float = 1.18
    declination_st: float = 1.5
    donor_f0_blend: float = 0.32


@dataclass
class V9DualBank:
    zh_syllables: dict[tuple[str, str], list[V9ZhToken]] = field(default_factory=dict)
    zh_onsets: dict[str, list[V9ZhToken]] = field(default_factory=dict)
    zh_finals: dict[str, list[V9ZhToken]] = field(default_factory=dict)
    en_phones: dict[str, list[V9EnToken]] = field(default_factory=dict)
    zh_profiles_v2: dict[str, tc2.TimbreProfileV2] = field(default_factory=dict)
    en_profiles_v2: dict[str, tc2.TimbreProfileV2] = field(default_factory=dict)
    zh_profiles_v3: dict[str, Any] = field(default_factory=dict)
    en_profiles_v3: dict[str, Any] = field(default_factory=dict)
    fallback_unit_bank: ub.UnitBank | None = None
    prosody_rules: ProsodyRules = field(default_factory=ProsodyRules)
    zh_included_ids: list[str] = field(default_factory=list)
    zh_excluded_ids: list[str] = field(default_factory=list)
    en_included_ids: list[str] = field(default_factory=list)
    en_excluded_ids: list[str] = field(default_factory=list)


def _edge_cmvn(log_env: np.ndarray, left: bool) -> np.ndarray:
    sl = log_env[: min(2, len(log_env))] if left else log_env[-min(2, len(log_env)) :]
    v = np.mean(sl, axis=0)
    return (v - float(np.mean(v))).astype(np.float32)


def _edge_f0(f0_rel: np.ndarray, voiced: np.ndarray, left: bool) -> float:
    sl = f0_rel[: min(3, len(f0_rel))] if left else f0_rel[-min(3, len(f0_rel)) :]
    vm = voiced[: min(3, len(voiced))] if left else voiced[-min(3, len(voiced)) :]
    return float(np.mean(sl[vm])) if np.any(vm) else 0.0


def build_v9_dual_bank(
    zh_utterances: list[dict] | None = None,
    zh_exclude_ids: Iterable[str] = (),
    en_utterances: list[dict] | None = None,
    en_exclude_ids: Iterable[str] = (),
    zh_precomputed: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures, np.ndarray]] | None = None,
    en_precomputed: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures, np.ndarray]] | None = None,
    sr: int = SR,
    refine_boundaries: bool = True,
    include_fallback_bank: bool = True,
    extract_v3_profiles: bool = False,
) -> V9DualBank:
    """Build v9 Dual-Donor EN+ZH Bank with acoustic landmark boundary refinement & HD unvoiced feats."""
    if zh_utterances is None:
        zh_utterances = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
    if en_utterances is None:
        en_utterances = json.loads(EN_MANIFEST.read_text(encoding="utf-8"))["english"]["utterances"]

    zh_excl = set(zh_exclude_ids)
    en_excl = set(en_exclude_ids)
    zh_train = [u for u in zh_utterances if u["id"] not in zh_excl]
    en_train = [u for u in en_utterances if u["id"] not in en_excl]
    if not zh_train:
        raise ValueError("No Mandarin training utterances remain after zh_exclude_ids")

    zh_data: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures, np.ndarray]] = {}
    zh_by_spk: dict[str, list[tuple[np.ndarray, vocoder.VocoderFeatures]]] = {}
    zh_for_ub: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures]] = {}
    for u in zh_train:
        uid = u["id"]
        if zh_precomputed is not None and uid in zh_precomputed:
            x_act, feats, b_gt8 = zh_precomputed[uid]
        else:
            x, fsr = vocoder.read_wav_mono(u["path"])
            if fsr != sr:
                raise ValueError(f"Expected {sr} Hz for {uid}, got {fsr}")
            x_act = x[u["active_start"] : u["active_stop"]]
            feats, b_gt8 = analyze_v9(x_act, sr=sr)
        zh_data[uid] = (x_act, feats, b_gt8)
        zh_for_ub[uid] = (x_act, feats)
        zh_by_spk.setdefault(u["speaker"], []).append((x_act, feats))

    zh_profs_v2 = {
        spk: tc2.extract_v2_profile(items, sr=sr, name=spk)
        for spk, items in sorted(zh_by_spk.items())
    }

    fallback_ub = (
        ub.build_unit_bank(zh_train, exclude_ids=zh_excl, precomputed=zh_for_ub, sr=sr)
        if include_fallback_bank
        else None
    )

    zh_syllables: dict[tuple[str, str], list[V9ZhToken]] = {}
    zh_onsets: dict[str, list[V9ZhToken]] = {}
    zh_finals: dict[str, list[V9ZhToken]] = {}
    zh_fin_ratios = []

    for u in zh_train:
        uid = u["id"]
        spk = u["speaker"]
        x_act, feats, b_gt8 = zh_data[uid]
        act_start = int(u["active_start"])
        tracks = compute_landmark_tracks(x_act, feats, sr=sr) if refine_boundaries else None

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

        syls = u["syllables"]
        parsed = [engine.split_pinyin(s["pinyin"]) for s in syls]
        n_tot = len(env_norm)
        if len(syls) >= 3:
            med_ms = float(np.median([float(s["ms"]) for s in syls[:-1]]))
            if med_ms > 20.0:
                zh_fin_ratios.append(float(syls[-1]["ms"]) / med_ms)

        for idx_s, (syl, (o, f, tone)) in enumerate(zip(syls, parsed)):
            f_lo = max(0, min(n_tot - 2, int(round((int(syl["start_sample"]) - act_start) / HOP))))
            f_hi = max(f_lo + 2, min(n_tot, int(round((int(syl["stop_sample"]) - act_start) / HOP))))
            dur_ms = float(syl["ms"])
            _, c_dur = vocal._split_duration(o, max(0.07, dur_ms / 1000.0)) if o else (0.0, 0.0)
            prop_fr = max(2, min((f_hi - f_lo) - 2, int(round((c_dur / max(1e-4, dur_ms / 1000.0)) * (f_hi - f_lo))))) if o else 0

            if refine_boundaries and tracks is not None:
                f_lo_r, f_hi_r, f_sp = refine_zh_syllable_bounds(tracks, f_lo, f_hi, o, f, prop_fr)
            else:
                f_lo_r, f_hi_r = f_lo, f_hi
                f_sp = f_lo + prop_fr
                if o in OBSTRUENT_ONSETS:
                    for idx in range(f_lo + 1, min(f_hi - 2, f_lo + max(prop_fr + 3, int(round(0.48 * (f_hi - f_lo)))))):
                        if feats.voiced[idx] and feats.voiced[idx + 1]:
                            f_sp = idx
                            break

            f_sp_rel = max(0, min(f_hi_r - f_lo_r, f_sp - f_lo_r))
            prev_f = parsed[idx_s - 1][1] if idx_s > 0 else "SIL"
            next_o = parsed[idx_s + 1][0] if idx_s + 1 < len(parsed) else "SIL"

            voi_sl = feats.voiced[f_lo_r:f_hi_r].copy()
            onset_hflf = (
                float(np.mean(tracks["hf_lf"][f_lo_r:f_sp]))
                if (tracks is not None and f_sp > f_lo_r)
                else 0.0
            )
            if refine_boundaries and tracks is not None and o in OBSTRUENT_ONSETS and f_sp_rel > 0:
                for k_rel in range(f_sp_rel):
                    if tracks["voi"][f_lo_r + k_rel] < 0.5 or tracks["hf_lf"][f_lo_r + k_rel] > -4.0:
                        voi_sl[k_rel] = False
            elif o in OBSTRUENT_ONSETS and f_sp_rel > 0:
                voi_sl[:f_sp_rel] = False

            on_wav = None
            if o in OBSTRUENT_ONSETS and f_sp_rel > 0 and not np.any(voi_sl[:f_sp_rel]):
                if not refine_boundaries or onset_hflf > -2.0:
                    s0 = max(0, f_lo_r * HOP)
                    s1 = min(len(x_act), f_sp * HOP)
                    if s1 > s0:
                        on_wav = (x_act[s0:s1] * wav_scale).astype(np.float32)

            le_sl = env_norm[f_lo_r:f_hi_r].astype(np.float32)
            ba_sl = feats.bap[f_lo_r:f_hi_r].astype(np.float32)
            bg_sl = b_gt8[f_lo_r:f_hi_r].astype(np.float32)
            f0_sl = f0_rel[f_lo_r:f_hi_r].astype(np.float32)
            tok = V9ZhToken(
                onset=o,
                final=f,
                tone=tone,
                speaker=spk,
                utterance_id=uid,
                prev_final=prev_f,
                next_onset=next_o,
                log_env=le_sl,
                bap=ba_sl,
                bap_gt8k=bg_sl,
                voiced=voi_sl.astype(bool),
                f0_rel_st=f0_sl,
                f_split=int(f_sp_rel),
                dur_ms=float((f_hi_r - f_lo_r) * HOP / sr * 1000.0),
                cmvn_left=_edge_cmvn(le_sl, True),
                cmvn_right=_edge_cmvn(le_sl, False),
                f0_edge_l=_edge_f0(f0_sl, voi_sl, True),
                f0_edge_r=_edge_f0(f0_sl, voi_sl, False),
                uv_hf_lf=onset_hflf,
                onset_wav=on_wav,
            )
            zh_syllables.setdefault((o, f), []).append(tok)

            if o and f_sp_rel > 0:
                le_o = le_sl[:f_sp_rel]
                v_o = voi_sl[:f_sp_rel]
                f0_o = f0_sl[:f_sp_rel]
                o_tok = V9ZhToken(
                    onset=o, final="", tone=tone, speaker=spk, utterance_id=uid,
                    prev_final=prev_f, next_onset=f,
                    log_env=le_o, bap=ba_sl[:f_sp_rel], bap_gt8k=bg_sl[:f_sp_rel],
                    voiced=v_o, f0_rel_st=f0_o, f_split=int(f_sp_rel),
                    dur_ms=float(f_sp_rel * HOP / sr * 1000.0),
                    cmvn_left=_edge_cmvn(le_o, True), cmvn_right=_edge_cmvn(le_o, False),
                    f0_edge_l=_edge_f0(f0_o, v_o, True), f0_edge_r=_edge_f0(f0_o, v_o, False),
                    uv_hf_lf=onset_hflf,
                    onset_wav=on_wav,
                )
                zh_onsets.setdefault(o, []).append(o_tok)

            if f_hi_r > f_sp:
                le_f = le_sl[f_sp_rel:]
                v_f = voi_sl[f_sp_rel:]
                f0_f = f0_sl[f_sp_rel:]
                f_tok = V9ZhToken(
                    onset="", final=f, tone=tone, speaker=spk, utterance_id=uid,
                    prev_final=o or prev_f, next_onset=next_o,
                    log_env=le_f, bap=ba_sl[f_sp_rel:], bap_gt8k=bg_sl[f_sp_rel:],
                    voiced=v_f, f0_rel_st=f0_f, f_split=0,
                    dur_ms=float(len(le_f) * HOP / sr * 1000.0),
                    cmvn_left=_edge_cmvn(le_f, True), cmvn_right=_edge_cmvn(le_f, False),
                    f0_edge_l=_edge_f0(f0_f, v_f, True), f0_edge_r=_edge_f0(f0_f, v_f, False),
                    onset_wav=None,
                )
                zh_finals.setdefault(f, []).append(f_tok)

    # English bank extraction with acoustic-landmark DP aligner
    en_data: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures, np.ndarray]] = {}
    en_by_spk: dict[str, list[tuple[np.ndarray, vocoder.VocoderFeatures]]] = {}
    for u in en_train:
        uid = u["id"]
        if en_precomputed is not None and uid in en_precomputed:
            x_act, feats, b_gt8 = en_precomputed[uid]
        else:
            p_wav = (EN_MANIFEST.parent.parent.parent / u["path"]) if not u["path"].startswith(("D:", "C:", "/")) else u["path"]
            x, fsr = vocoder.read_wav_mono(p_wav)
            if fsr != sr:
                raise ValueError(f"Expected {sr} Hz for {uid}, got {fsr}")
            s0, s1 = _trim_active_span(x, sr=sr)
            x_act = x[s0:s1]
            feats, b_gt8 = analyze_v9(x_act, sr=sr)
        en_data[uid] = (x_act, feats, b_gt8)
        en_by_spk.setdefault(u["speaker"], []).append((x_act, feats))

    en_profs_v2 = {
        spk: tc2.extract_v2_profile(items, sr=sr, name=spk)
        for spk, items in sorted(en_by_spk.items())
    }

    en_phones: dict[str, list[V9EnToken]] = {}
    for u in en_train:
        uid = u["id"]
        spk = u["speaker"]
        x_act, feats, b_gt8 = en_data[uid]
        pl = cf.plan(u["text"], allow_oov_fallback=True)
        pairs: list[tuple[str, int | None]] = []
        for unit in pl["units"]:
            arpa = unit.get("arpabet") or []
            if arpa:
                pairs.extend(v6_arb._split_en_arpabet_to_ipa_with_stress(arpa))
            else:
                for p in (unit.get("ipa_phones") or []):
                    pairs.append((p, None))
        ipas = [p for p, _ in pairs]
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

        tracks = compute_landmark_tracks(x_act, feats, sr=sr)
        if refine_boundaries:
            bounds = align_en_phones_with_landmarks(ipas, feats, x_act, sr=sr, tracks=tracks)
        else:
            from exp_v8_clarity.en_utau_bank import align_ipa_sequence
            bounds = align_ipa_sequence(ipas, feats, sr=sr)

        for idx_p, ((ipa, stress), (f0_i, f1_i)) in enumerate(zip(pairs, bounds)):
            if f1_i <= f0_i:
                continue
            prev_ip = ipas[idx_p - 1] if idx_p > 0 else "SIL"
            next_ip = ipas[idx_p + 1] if idx_p + 1 < len(ipas) else "SIL"
            voi_sl = feats.voiced[f0_i:f1_i].copy()
            sp_info = IPA_SPECS.get(ipa)
            is_uv_obs = sp_info is not None and not sp_info.voiced and sp_info.manner in ("fricative", "affricate", "stop")
            if is_uv_obs:
                if refine_boundaries:
                    hf_sl = tracks["hf_lf"][f0_i:f1_i]
                    f0_trim, f1_trim = f0_i, f1_i
                    if (f1_trim - f0_trim) >= 4 and hf_sl[0] < -4.0 and tracks["voi"][f0_trim] > 0.5:
                        f0_trim += 1
                    if (f1_trim - f0_trim) >= 4 and hf_sl[-1] < -4.0 and tracks["voi"][f1_trim - 1] > 0.5:
                        f1_trim -= 1
                    f0_i, f1_i = f0_trim, f1_trim
                voi_sl = feats.voiced[f0_i:f1_i].copy()
                voi_sl[:] = False

            tok_hflf = float(np.mean(tracks["hf_lf"][f0_i:f1_i])) if f1_i > f0_i else 0.0
            s0 = max(0, f0_i * HOP)
            s1 = min(len(x_act), f1_i * HOP)
            raw_w = (x_act[s0:s1] * wav_scale).astype(np.float32) if (s1 > s0 and not np.any(voi_sl)) else None
            le_sl = env_norm[f0_i:f1_i].astype(np.float32)
            ba_sl = feats.bap[f0_i:f1_i].astype(np.float32)
            bg_sl = b_gt8[f0_i:f1_i].astype(np.float32)
            f0_sl = f0_rel[f0_i:f1_i].astype(np.float32)
            tok = V9EnToken(
                ipa=ipa,
                speaker=spk,
                utterance_id=uid,
                prev_ipa=prev_ip,
                next_ipa=next_ip,
                stress=stress,
                log_env=le_sl,
                bap=ba_sl,
                bap_gt8k=bg_sl,
                voiced=voi_sl.astype(bool),
                f0_rel_st=f0_sl,
                dur_ms=float((f1_i - f0_i) * HOP / sr * 1000.0),
                cmvn_left=_edge_cmvn(le_sl, True),
                cmvn_right=_edge_cmvn(le_sl, False),
                f0_edge_l=_edge_f0(f0_sl, voi_sl, True),
                f0_edge_r=_edge_f0(f0_sl, voi_sl, False),
                uv_hf_lf=tok_hflf,
                raw_wav=raw_w,
            )
            en_phones.setdefault(ipa, []).append(tok)

    zh_profs_v3: dict[str, Any] = {}
    en_profs_v3: dict[str, Any] = {}
    if extract_v3_profiles:
        from exp_v9_timbre_v3.timbre_cloner_v3 import extract_timbre_v3
        zh_profs_v3 = {
            spk: extract_timbre_v3(items[:12], sr=sr, name=spk, corpus="aishell3")
            for spk, items in sorted(zh_by_spk.items())
        }
        en_profs_v3 = {
            spk: extract_timbre_v3(items[:12], sr=sr, name=spk, corpus="cmu_arctic")
            for spk, items in sorted(en_by_spk.items())
        }

    zh_fin_len = float(np.clip(np.median(zh_fin_ratios), 1.12, 1.32)) if zh_fin_ratios else 1.22
    return V9DualBank(
        zh_syllables=zh_syllables,
        zh_onsets=zh_onsets,
        zh_finals=zh_finals,
        en_phones=en_phones,
        zh_profiles_v2=zh_profs_v2,
        en_profiles_v2=en_profs_v2,
        zh_profiles_v3=zh_profs_v3,
        en_profiles_v3=en_profs_v3,
        fallback_unit_bank=fallback_ub,
        prosody_rules=ProsodyRules(zh_final_lengthen=zh_fin_len),
        zh_included_ids=sorted(zh_data.keys()),
        zh_excluded_ids=sorted(zh_excl),
        en_included_ids=sorted(en_data.keys()),
        en_excluded_ids=sorted(en_excl),
    )


def filter_v9_bank_excluding_speakers(
    bank: V9DualBank,
    excluded_speakers: Iterable[str],
) -> V9DualBank:
    """Zero-copy view of V9DualBank excluding named donor speakers (for zero-leakage cross-donor tests)."""
    excl_spk = set(excluded_speakers)
    en_phones = {}
    kept_en_ids, rem_en_ids = set(), set(bank.en_excluded_ids)
    for ipa, toks in bank.en_phones.items():
        kept = [t for t in toks if t.speaker not in excl_spk]
        if kept:
            en_phones[ipa] = kept
            kept_en_ids.update(t.utterance_id for t in kept)
        for t in toks:
            if t.speaker in excl_spk:
                rem_en_ids.add(t.utterance_id)

    zh_syls = {k: v for k, v in ((k, [t for t in toks if t.speaker not in excl_spk]) for k, toks in bank.zh_syllables.items()) if v}
    zh_ons = {k: v for k, v in ((k, [t for t in toks if t.speaker not in excl_spk]) for k, toks in bank.zh_onsets.items()) if v}
    zh_fins = {k: v for k, v in ((k, [t for t in toks if t.speaker not in excl_spk]) for k, toks in bank.zh_finals.items()) if v}
    return V9DualBank(
        zh_syllables=zh_syls,
        zh_onsets=zh_ons,
        zh_finals=zh_fins,
        en_phones=en_phones,
        zh_profiles_v2={k: v for k, v in bank.zh_profiles_v2.items() if k not in excl_spk},
        en_profiles_v2={k: v for k, v in bank.en_profiles_v2.items() if k not in excl_spk},
        zh_profiles_v3={k: v for k, v in bank.zh_profiles_v3.items() if k not in excl_spk},
        en_profiles_v3={k: v for k, v in bank.en_profiles_v3.items() if k not in excl_spk},
        fallback_unit_bank=bank.fallback_unit_bank,
        prosody_rules=bank.prosody_rules,
        zh_included_ids=list(bank.zh_included_ids),
        zh_excluded_ids=list(bank.zh_excluded_ids),
        en_included_ids=sorted(kept_en_ids),
        en_excluded_ids=sorted(rem_en_ids),
    )


def _viterbi_select(
    slots: list[tuple[list[Any], list[float], bool, bool]],
    beam_k: int = 8,
) -> list[Any]:
    """Run exact Viterbi unit selection over `(candidates, target_costs, paused_before, is_obs_edge)`."""
    if not slots:
        return []
    pruned: list[tuple[list[Any], list[float], bool, bool]] = []
    for cands, tcosts, p_bef, obs_e in slots:
        if len(cands) > beam_k:
            order = sorted(range(len(cands)), key=lambda i: tcosts[i])[:beam_k]
            pruned.append(([cands[i] for i in order], [tcosts[i] for i in order], p_bef, obs_e))
        else:
            pruned.append((cands, tcosts, p_bef, obs_e))

    dp_prev = np.array(pruned[0][1], dtype=np.float64)
    bps: list[np.ndarray] = []
    for t in range(1, len(pruned)):
        c_prev = pruned[t - 1][0]
        obs_prev = pruned[t - 1][3]
        c_cur, tc_cur, p_bef, obs_cur = pruned[t]
        n_p, n_c = len(c_prev), len(c_cur)
        dp_cur = np.empty(n_c, dtype=np.float64)
        bp_t = np.zeros(n_c, dtype=np.int32)
        # Never smooth spectral joins across pauses or obstruent boundaries (either left OR right!)
        w_join = 0.0 if (p_bef or obs_prev or obs_cur) else 0.70
        for j in range(n_c):
            cj = c_cur[j]
            best_v = float("inf")
            best_i = 0
            for i in range(n_p):
                ci = c_prev[i]
                if w_join > 0.0:
                    diff = ci.cmvn_right - cj.cmvn_left
                    spec_j = float(math.sqrt(float(np.mean(diff * diff))))
                    f0_j = abs(ci.f0_edge_r - cj.f0_edge_l)
                    jcost = w_join * (spec_j + 0.12 * f0_j)
                else:
                    jcost = 0.0
                v = dp_prev[i] + jcost + tc_cur[j]
                if v < best_v:
                    best_v = v
                    best_i = i
            dp_cur[j] = best_v
            bp_t[j] = best_i
        bps.append(bp_t)
        dp_prev = dp_cur

    idx = int(np.argmin(dp_prev))
    chosen = [pruned[-1][0][idx]]
    for t in range(len(bps) - 1, -1, -1):
        idx = int(bps[t][idx])
        chosen.append(pruned[t][0][idx])
    chosen.reverse()
    return chosen


def _get_src_profile(
    bank: V9DualBank,
    spk: str,
    target_profile: Any = None,
    src_profiles: dict[str, Any] | None = None,
) -> Any:
    if src_profiles is not None and spk in src_profiles:
        return src_profiles[spk]
    if target_profile is not None and hasattr(target_profile, "ope_h1_h2_db"):
        if spk in bank.en_profiles_v3:
            return bank.en_profiles_v3[spk]
        if spk in bank.zh_profiles_v3:
            return bank.zh_profiles_v3[spk]
    if spk in bank.en_profiles_v2:
        return bank.en_profiles_v2[spk]
    if spk in bank.zh_profiles_v2:
        return bank.zh_profiles_v2[spk]
    return None


def _adapt_token_with_cloner(
    log_env: np.ndarray,
    bap: np.ndarray,
    bap_gt8k: np.ndarray,
    voiced: np.ndarray,
    f0_rel_st: np.ndarray,
    src_prof: tc2.TimbreProfileV2 | None,
    tgt_prof: Any,
    cloner_fn: ClonerFn | None,
    vtln_alpha: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tgt_name = getattr(tgt_prof, "name", "")
    if src_prof is None or src_prof.name == tgt_name:
        return log_env.astype(np.float64), bap.astype(np.float64), bap_gt8k.astype(np.float64), voiced.copy()

    dummy_f0 = np.where(voiced, src_prof.f0_median_hz * (2.0 ** (f0_rel_st / 12.0)), 0.0)
    feats = vocoder.VocoderFeatures(
        f0_hz=dummy_f0,
        voiced=voiced.copy(),
        log_env=log_env.astype(np.float64),
        bap=bap.astype(np.float64),
        sr=SR,
        hop_samples=HOP,
        n_samples=len(log_env) * HOP,
        n_fft=N_FFT,
    )
    fn = cloner_fn if cloner_fn is not None else tc2.adapt_features_v2
    try:
        res = fn(feats, src_prof, tgt_prof, vtln_alpha=vtln_alpha) if vtln_alpha is not None else fn(feats, src_prof, tgt_prof)
    except TypeError:
        res = fn(feats, src_prof, tgt_prof)
    ad = res[0] if isinstance(res, tuple) else res

    # Scale >8 kHz fine BAP smoothly by the same ratio the cloner applied to band 4
    ratio_b4 = np.clip(ad.bap[:, 4:5] / np.maximum(bap[:, 4:5], 1e-3), 0.25, 2.5)
    bg_ad = np.clip(bap_gt8k.astype(np.float64) * ratio_b4, 0.005, 0.995)
    return ad.log_env, ad.bap, bg_ad, ad.voiced


def render_v9_utterance(
    text_or_plan: str | dict,
    bank: V9DualBank,
    target_profile: Any,
    *,
    cloner_fn: ClonerFn | None = None,
    src_profiles: dict[str, Any] | None = None,
    vtln_alpha: float | None = None,
    use_viterbi: bool = True,
    use_unvoiced_residual: bool = True,
    prosody_mode: str = "natural",
    seed: int = 72,
    phase_dispersion: bool = True,
) -> tuple[np.ndarray, dict]:
    """Render ZH / EN / Code-Switched text or plan with Landmark Boundaries + Viterbi + Natural Prosody + HD HNM."""
    if isinstance(text_or_plan, str):
        plan_doc = cf.plan(text_or_plan, allow_oov_fallback=True)
    else:
        plan_doc = text_or_plan

    units = plan_doc.get("units") or []
    if not units:
        raise ValueError("plan_doc has no units")

    tgt_name = getattr(target_profile, "name", "")
    tgt_f0 = float(getattr(target_profile, "f0_median_hz", 180.0))
    tgt_iqr = float(getattr(target_profile, "f0_iqr_st", 3.2))
    iqr_scale = max(0.55, min(1.75, tgt_iqr / 3.2))
    rules = bank.prosody_rules

    # Pre-parse units & identify breath-group structure for natural declination & phrase-final lengthening
    n_u = len(units)
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

    # Identify breath-group indices per unit for phrase declination and final lengthening
    group_id = [0] * n_u
    cur_g = 0
    for i, u in enumerate(units):
        group_id[i] = cur_g
        if float(u.get("pause_after_ms", 0.0)) > 55.0:
            cur_g += 1
    group_spans: dict[int, tuple[int, int]] = {}
    for i, g in enumerate(group_id):
        if g not in group_spans:
            group_spans[g] = [i, i]  # type: ignore[assignment]
        else:
            group_spans[g][1] = i  # type: ignore[index]

    unit_dur_ms = []
    unit_decl_st = []
    unit_gain_db = []
    for i, u in enumerate(units):
        base_d = max(35.0, float(u.get("duration_ms", 200.0)))
        g0, g1 = group_spans[group_id[i]]
        g_len = max(1, g1 - g0)
        rel_pos = (i - g0) / max(1, g_len)
        if prosody_mode == "natural":
            if i == g1:
                base_d *= rules.zh_final_lengthen if parsed[i][0] == "zh" else rules.en_final_lengthen
            elif g_len >= 4 and i > g0 and i < g1:
                base_d *= 0.96
            decl = 0.5 * rules.declination_st - rules.declination_st * rel_pos
            gdb = 0.4 - 2.2 * (rel_pos ** 2) if i == g1 else 0.3 * (1.0 - rel_pos)
        else:
            decl = 0.5 - 1.0 * (i / max(1, n_u - 1))
            gdb = 0.0
        unit_dur_ms.append(base_d)
        unit_decl_st.append(decl)
        unit_gain_db.append(gdb)

    # First pass: select Mandarin tokens via Viterbi over contiguous ZH spans, and EN phones via Viterbi per EN word/span
    selected_zh: dict[int, tuple[str, Any]] = {}
    i = 0
    while i < n_u:
        if parsed[i][0] != "zh":
            i += 1
            continue
        j = i
        while j < n_u and parsed[j][0] == "zh":
            j += 1
        # Build Viterbi slots for [i, j) where exact syllable matches exist; split on fallback slots
        slots: list[tuple[list[V9ZhToken], list[float], bool, bool]] = []
        slot_indices: list[int] = []
        for idx in range(i, j):
            _, o, f, tone = parsed[idx]
            prev_f = parsed[idx - 1][2] if (idx > 0 and parsed[idx - 1][0] == "zh") else "SIL"
            next_o = parsed[idx + 1][1] if (idx + 1 < n_u and parsed[idx + 1][0] == "zh") else "SIL"
            cands = bank.zh_syllables.get((o, f), [])
            p_bef = idx > 0 and float(units[idx - 1].get("pause_after_ms", 0.0)) > 15.0
            if cands:
                is_uv_zh = o in ("f", "s", "sh", "x", "h", "c", "ch", "q")
                tcosts = []
                for c in cands:
                    tc = 0.0
                    if c.speaker != tgt_name:
                        tc += 4.5
                    if c.tone != tone:
                        tc += 2.2
                    if c.prev_final != prev_f:
                        tc += 1.4
                    if c.next_onset != next_o:
                        tc += 1.4
                    if is_uv_zh and c.uv_hf_lf < -6.0:
                        tc += 6.0  # Broken/clipped consonant onset: prefer clean donor or concat_of!
                    elif is_uv_zh and c.uv_hf_lf < -2.0:
                        tc += 2.0
                    tc += 0.65 * abs(math.log(max(20.0, c.dur_ms) / max(20.0, unit_dur_ms[idx])))
                    tcosts.append(tc)
                slots.append((cands, tcosts, p_bef, o in OBSTRUENT_ONSETS))
                slot_indices.append(idx)
            else:
                if slots:
                    chosen = _viterbi_select(slots) if use_viterbi else [s[0][int(np.argmin(s[1]))] for s in slots]
                    for s_i, tok in zip(slot_indices, chosen):
                        selected_zh[s_i] = ("syl", tok)
                    slots, slot_indices = [], []
                selected_zh[idx] = ("fallback", None)
        if slots:
            chosen = _viterbi_select(slots) if use_viterbi else [s[0][int(np.argmin(s[1]))] for s in slots]
            for s_i, tok in zip(slot_indices, chosen):
                selected_zh[s_i] = ("syl", tok)
        i = j

    # Build flat English phone list across contiguous EN units so Viterbi optimizes across word joins too!
    en_slot_meta: list[tuple[int, int, int, str, int | None, float, bool]] = []
    for idx in range(n_u):
        if parsed[idx][0] == "zh":
            continue
        u = units[idx]
        ipas = list(u.get("ipa_phones") or ["ə"])
        arpa = u.get("arpabet") or []
        pairs = v6_arb._split_en_arpabet_to_ipa_with_stress(arpa) if arpa else [(p, None) for p in ipas]
        if not pairs:
            pairs = [(p, None) for p in ipas]
        for idx_p, (ip, st) in enumerate(pairs):
            p_bef = (idx_p == 0) and (idx > 0 and (float(units[idx - 1].get("pause_after_ms", 0.0)) > 15.0 or parsed[idx - 1][0] == "zh"))
            en_slot_meta.append((idx, idx_p, len(pairs), ip, st, unit_dur_ms[idx] / len(pairs), p_bef))

    selected_en_map: dict[tuple[int, int], V9EnToken | None] = {}
    if en_slot_meta:
        en_slots: list[tuple[list[V9EnToken], list[float], bool, bool]] = []
        en_keys: list[tuple[int, int]] = []
        for s_k, (u_i, p_i, n_p, ip, st, est_d, p_bef) in enumerate(en_slot_meta):
            prev_ip = en_slot_meta[s_k - 1][3] if s_k > 0 else "SIL"
            next_ip = en_slot_meta[s_k + 1][3] if s_k + 1 < len(en_slot_meta) else "SIL"
            cands = bank.en_phones.get(ip, [])
            if cands:
                sp_ip = IPA_SPECS.get(ip)
                is_uv_en = sp_ip is not None and not sp_ip.voiced and sp_ip.manner in ("fricative", "affricate")
                tcosts = []
                for c in cands:
                    tc = 0.0
                    if c.speaker != tgt_name:
                        tc += 5.2
                    if c.prev_ipa != prev_ip:
                        tc += 1.6
                    if c.next_ipa != next_ip:
                        tc += 1.6
                    if st is not None and c.stress is not None and st != c.stress:
                        tc += 0.7
                    if is_uv_en:
                        tc -= 0.15 * max(-10.0, min(10.0, c.uv_hf_lf))
                    tc += 0.55 * abs(math.log(max(20.0, c.dur_ms) / max(20.0, est_d)))
                    tcosts.append(tc)
                en_slots.append((cands, tcosts, p_bef, ip in OBSTRUENT_IPAS))
                en_keys.append((u_i, p_i))
            else:
                if en_slots:
                    chosen = _viterbi_select(en_slots) if use_viterbi else [s[0][int(np.argmin(s[1]))] for s in en_slots]
                    for k_up, tok in zip(en_keys, chosen):
                        selected_en_map[k_up] = tok
                    en_slots, en_keys = [], []
                selected_en_map[(u_i, p_i)] = None
        if en_slots:
            chosen = _viterbi_select(en_slots) if use_viterbi else [s[0][int(np.argmin(s[1]))] for s in en_slots]
            for k_up, tok in zip(en_keys, chosen):
                selected_en_map[k_up] = tok

    # Assemble 5 ms feature sequence
    envs, baps, bgt8s, vois, sts = [], [], [], [], []
    is_obs_head, is_pause_blk = [], []
    residual_spans: list[tuple[int, int, np.ndarray]] = []
    source_counts: dict[str, int] = {}

    lead_fr = 4
    envs.append(np.full((lead_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((lead_fr, 5), dtype=np.float64))
    bgt8s.append(np.ones((lead_fr, 2), dtype=np.float64))
    vois.append(np.zeros(lead_fr, dtype=bool))
    sts.append(np.zeros(lead_fr, dtype=np.float64))
    is_obs_head.append(False)
    is_pause_blk.append(True)
    cur_fr = lead_fr

    zh_fb_prof = _get_src_profile(bank, "SSB0502", target_profile, src_profiles) or next(iter(bank.zh_profiles_v2.values()))
    fb_default_prof = zh_fb_prof

    for u_idx, u in enumerate(units):
        lang, o, f, tone = parsed[u_idx]
        dur_ms = unit_dur_ms[u_idx]
        nf = max(2, int(round(dur_ms / 5.0)))
        gain = float(u.get("gain", 0.85))
        pitch_5 = [float(x) for x in u.get("pitch_st", [0.0] * 5)]
        pause_ms = float(u.get("pause_after_ms", 0.0))
        decl_u = unit_decl_st[u_idx]
        gdb_u = unit_gain_db[u_idx]

        if lang == "zh":
            kind_sel, tok_sel = selected_zh[u_idx]
            prev_f = parsed[u_idx - 1][2] if (u_idx > 0 and parsed[u_idx - 1][0] == "zh") else "SIL"
            next_o = parsed[u_idx + 1][1] if (u_idx + 1 < n_u and parsed[u_idx + 1][0] == "zh") else "SIL"
            if kind_sel == "syl" and tok_sel is not None:
                best: V9ZhToken = tok_sel
                e_raw, b_raw, bg_raw, v_raw = _adapt_token_with_cloner(
                    best.log_env, best.bap, best.bap_gt8k, best.voiced, best.f0_rel_st,
                    _get_src_profile(bank, best.speaker, target_profile, src_profiles), target_profile, cloner_fn, vtln_alpha,
                )
                mode = "zh_target_syl" if best.speaker == tgt_name else "zh_donor_syl"
                source_counts[mode] = source_counts.get(mode, 0) + 1
                e_st, b_st, v_st, f_sp_new = _stretch_preserving_onset(
                    e_raw, b_raw, v_raw, best.f_split, nf, preserve_onset=True
                )
                bg_st = np.clip(_resample_2d(bg_raw, nf), 0.005, 0.995)
                f0_tok = _resample_1d(best.f0_rel_st.astype(np.float64), nf)
                if use_unvoiced_residual and best.onset_wav is not None and f_sp_new > 0:
                    residual_spans.append((cur_fr, cur_fr + f_sp_new, best.onset_wav))
            else:
                parts_e, parts_b, parts_bg, parts_v, parts_f0 = [], [], [], [], []
                split_pt = 0
                bo_sel = None
                if o:
                    oc = bank.zh_onsets.get(o, [])
                    if oc:
                        is_uv_o = o in ("f", "s", "sh", "x", "h", "c", "ch", "q")
                        bo = min(
                            oc,
                            key=lambda c: (0 if c.speaker == tgt_name else 4.5)
                            + (0 if c.prev_final == prev_f else 1.4)
                            + (0 if c.next_onset == f else 1.0)
                            + (6.0 if (is_uv_o and c.uv_hf_lf < -6.0) else (2.0 if (is_uv_o and c.uv_hf_lf < -2.0) else 0.0)),
                        )
                        bo_sel = bo
                        eo, bo_b, bgo, vo = _adapt_token_with_cloner(
                            bo.log_env, bo.bap, bo.bap_gt8k, bo.voiced, bo.f0_rel_st,
                            _get_src_profile(bank, bo.speaker, target_profile, src_profiles), target_profile, cloner_fn, vtln_alpha,
                        )
                        parts_e.append(eo); parts_b.append(bo_b); parts_bg.append(bgo); parts_v.append(vo); parts_f0.append(bo.f0_rel_st)
                        split_pt = len(eo)
                fc = bank.zh_finals.get(f, [])
                if fc:
                    bf = min(
                        fc,
                        key=lambda c: (0 if c.speaker == tgt_name else 4.5)
                        + (0 if c.tone == tone else 2.2)
                        + (0 if c.prev_final == (o or prev_f) else 1.0)
                        + (0 if c.next_onset == next_o else 1.4),
                    )
                    ef, bf_b, bgf, vf = _adapt_token_with_cloner(
                        bf.log_env, bf.bap, bf.bap_gt8k, bf.voiced, bf.f0_rel_st,
                        _get_src_profile(bank, bf.speaker, target_profile, src_profiles), target_profile, cloner_fn, vtln_alpha,
                    )
                    parts_e.append(ef); parts_b.append(bf_b); parts_bg.append(bgf); parts_v.append(vf); parts_f0.append(bf.f0_rel_st)
                    source_counts["zh_concat_of"] = source_counts.get("zh_concat_of", 0) + 1
                elif bank.fallback_unit_bank is not None:
                    for fi in cf.PINYIN_FINAL_TO_IPA.get(f, ("ə",)):
                        d = cw.lookup_ipa_template(bank.fallback_unit_bank, fi, is_onset=False)
                        bg_d = np.repeat(d["bap"][:, 4:5], 2, axis=1)
                        ed, bd, bgd, vd = _adapt_token_with_cloner(
                            d["log_env"] - 3.1, d["bap"], bg_d, d["voiced"], np.zeros(len(d["log_env"])),
                            zh_fb_prof, target_profile, cloner_fn, vtln_alpha,
                        )
                        parts_e.append(ed); parts_b.append(bd); parts_bg.append(bgd); parts_v.append(vd); parts_f0.append(np.zeros(len(ed)))
                    source_counts["zh_ipa_fill"] = source_counts.get("zh_ipa_fill", 0) + 1
                e_st, b_st, v_st, f_sp_cf = _stretch_preserving_onset(
                    np.concatenate(parts_e, axis=0),
                    np.concatenate(parts_b, axis=0),
                    np.concatenate(parts_v, axis=0),
                    split_pt, nf, preserve_onset=True,
                )
                if use_unvoiced_residual and bo_sel is not None and bo_sel.onset_wav is not None and f_sp_cf > 0:
                    residual_spans.append((cur_fr, cur_fr + f_sp_cf, bo_sel.onset_wav))
                bg_st = np.clip(_resample_2d(np.concatenate(parts_bg, axis=0), nf), 0.005, 0.995)
                f0_tok = _resample_1d(np.concatenate(parts_f0), nf)

            gain_shift = gdb_u * math.log(10.0) / 20.0
            if abs(gain - 0.85) > 0.05:
                gain_shift += math.log(max(0.25, min(1.8, gain / 0.85)))
            e_st = e_st + gain_shift

            t_grid = np.linspace(0.0, 1.0, len(e_st))
            p5_arr = np.array([_interp_list(pitch_5, float(t)) for t in t_grid], dtype=np.float64) * iqr_scale
            if prosody_mode == "natural":
                f0_micro = np.clip(f0_tok - float(np.median(f0_tok)), -3.0, 3.0)
                st_u = (1.0 - rules.donor_f0_blend) * p5_arr + rules.donor_f0_blend * f0_micro + decl_u
            else:
                st_u = p5_arr + decl_u

            envs.append(e_st); baps.append(b_st); bgt8s.append(bg_st); vois.append(v_st); sts.append(st_u)
            is_obs_head.append(o in OBSTRUENT_ONSETS)
            is_pause_blk.append(False)
            cur_fr += len(e_st)
        else:
            ipas = list(u.get("ipa_phones") or ["ə"])
            arpa = u.get("arpabet") or []
            pairs = v6_arb._split_en_arpabet_to_ipa_with_stress(arpa) if arpa else [(p, None) for p in ipas]
            if not pairs:
                pairs = [(p, None) for p in ipas]
            n_p = len(pairs)
            chosen_p = []
            raw_durs = []
            for idx_p, (ip, st) in enumerate(pairs):
                tok_e = selected_en_map.get((u_idx, idx_p))
                if tok_e is not None:
                    e_r, b_r, bg_r, v_r = _adapt_token_with_cloner(
                        tok_e.log_env, tok_e.bap, tok_e.bap_gt8k, tok_e.voiced, tok_e.f0_rel_st,
                        _get_src_profile(bank, tok_e.speaker, target_profile, src_profiles), target_profile, cloner_fn, vtln_alpha,
                    )
                    mode = "en_target_phone" if tok_e.speaker == tgt_name else "en_donor_phone"
                    source_counts[mode] = source_counts.get(mode, 0) + 1
                    d_ms = tok_e.dur_ms
                    w_raw = tok_e.raw_wav
                    f0_tk = tok_e.f0_rel_st.astype(np.float64)
                elif bank.fallback_unit_bank is not None:
                    d = cw.lookup_ipa_template(bank.fallback_unit_bank, ip, is_onset=(idx_p == 0))
                    bg_d = np.repeat(d["bap"][:, 4:5], 2, axis=1)
                    e_r, b_r, bg_r, v_r = _adapt_token_with_cloner(
                        d["log_env"] - 3.1, d["bap"], bg_d, d["voiced"], np.zeros(len(d["log_env"])),
                        fb_default_prof, target_profile, cloner_fn, vtln_alpha,
                    )
                    source_counts["en_synth_fill"] = source_counts.get("en_synth_fill", 0) + 1
                    d_ms = d["dur_ms"]
                    w_raw = None
                    f0_tk = np.zeros(len(e_r), dtype=np.float64)
                else:
                    raise ValueError(f"Missing EN phone {ip} and fallback_unit_bank=None")
                if st == 1:
                    d_ms *= 1.16
                elif st == 0:
                    d_ms = max(35.0, d_ms * 0.76)
                chosen_p.append((ip, e_r, b_r, bg_r, v_r, w_raw, f0_tk))
                raw_durs.append(max(20.0, d_ms))

            tot_d = max(1.0, sum(raw_durs))
            acc_f = 0
            for idx_p, ((ip, e_r, b_r, bg_r, v_r, w_raw, f0_tk), d_ms) in enumerate(zip(chosen_p, raw_durs)):
                kp = (
                    max(2, nf - acc_f)
                    if (idx_p + 1 == n_p)
                    else max(2, min(nf - acc_f - 2 * (n_p - idx_p - 1), int(round(nf * d_ms / tot_d))))
                )
                acc_f += kp
                ep = _resample_2d(e_r, kp)
                bp = np.clip(_resample_2d(b_r, kp), 0.005, 0.995)
                bgp = np.clip(_resample_2d(bg_r, kp), 0.005, 0.995)
                vp = _resample_1d(v_r.astype(np.float64), kp) >= 0.45
                gain_shift = gdb_u * math.log(10.0) / 20.0
                if abs(gain - 0.85) > 0.05:
                    gain_shift += math.log(max(0.25, min(1.8, gain / 0.85)))
                ep = ep + gain_shift

                sub_p = [_interp_list(pitch_5, (idx_p + m / 4.0) / n_p) for m in range(5)]
                tp = np.linspace(0.0, 1.0, kp)
                p_arr = np.array([_interp_list(sub_p, float(t)) for t in tp], dtype=np.float64) * iqr_scale
                if prosody_mode == "natural":
                    f0_tr = _resample_1d(f0_tk, kp)
                    f0_mic = np.clip(f0_tr - float(np.median(f0_tr)), -2.5, 2.5)
                    st_p = (1.0 - 0.25) * p_arr + 0.25 * f0_mic + decl_u
                else:
                    st_p = p_arr + decl_u

                if use_unvoiced_residual and w_raw is not None and not np.any(vp):
                    residual_spans.append((cur_fr, cur_fr + kp, w_raw))
                envs.append(ep); baps.append(bp); bgt8s.append(bgp); vois.append(vp); sts.append(st_p)
                is_obs_head.append(ip in OBSTRUENT_IPAS)
                is_pause_blk.append(False)
                cur_fr += kp

        if pause_ms > 15.0:
            p_fr = max(2, int(round(pause_ms / 5.0)))
            envs.append(np.full((p_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
            baps.append(np.ones((p_fr, 5), dtype=np.float64))
            bgt8s.append(np.ones((p_fr, 2), dtype=np.float64))
            vois.append(np.zeros(p_fr, dtype=bool))
            sts.append(np.zeros(p_fr, dtype=np.float64))
            is_obs_head.append(False)
            is_pause_blk.append(True)
            cur_fr += p_fr

    tail_fr = 6
    envs.append(np.full((tail_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((tail_fr, 5), dtype=np.float64))
    bgt8s.append(np.ones((tail_fr, 2), dtype=np.float64))
    vois.append(np.zeros(tail_fr, dtype=bool))
    sts.append(np.zeros(tail_fr, dtype=np.float64))
    is_obs_head.append(False)
    is_pause_blk.append(True)

    # Manner-aware seams: 0 ms on obstruents/V-UV edges, 10 ms on voiced vowel/sonorant joins
    for idx_b in range(len(envs) - 1):
        if is_pause_blk[idx_b] or is_pause_blk[idx_b + 1]:
            continue
        if not is_obs_head[idx_b + 1] and np.any(vois[idx_b]) and np.any(vois[idx_b + 1]):
            _blend_seam(envs[idx_b], envs[idx_b + 1], baps[idx_b], baps[idx_b + 1], blend_frames=2)

    env_all = np.concatenate(envs, axis=0)
    bap_all = np.clip(np.concatenate(baps, axis=0), 0.005, 0.995)
    bgt8_all = np.clip(np.concatenate(bgt8s, axis=0), 0.005, 0.995)
    voi_all = np.concatenate(vois, axis=0)
    st_all = np.concatenate(sts, axis=0)

    v_idx = np.where(voi_all)[0]
    if v_idx.size > 0:
        st_centered = st_all[v_idx] - float(np.median(st_all[v_idx]))
        if len(st_centered) >= 5:
            sm = st_centered.copy()
            sm[1:-1] = 0.25 * st_centered[:-2] + 0.50 * st_centered[1:-1] + 0.25 * st_centered[2:]
            st_centered = sm
        f0_all = np.zeros(len(st_all), dtype=np.float64)
        f0_all[v_idx] = np.clip(tgt_f0 * (2.0 ** (st_centered / 12.0)), 65.0, 440.0)
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
    y = synthesize_v9(
        feats,
        seed=seed,
        phase_dispersion=phase_dispersion,
        bap_gt8k=bgt8_all,
        unvoiced_residual_spans=residual_spans if use_unvoiced_residual else None,
    )
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.89:
        y *= 0.89 / peak
    return y, {"source_counts": source_counts, "n_frames": len(env_all), "n_residual_spans": len(residual_spans)}
