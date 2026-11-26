"""Multi-Sample `SSB0502` Voice Bank with Unbroken Utterance Feature Views (`exp_v14_fluent_zh_tts`)."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v7_pybot_tts import unit_bank as ub  # noqa: E402
from exp_v8_clarity.utau_concat import OBSTRUENT_ONSETS  # noqa: E402
from exp_v8_timbre_clone import timbre_cloner_v2 as tc2  # noqa: E402
from exp_v9_clarity_pro.boundary_refiner import compute_landmark_tracks  # noqa: E402
from exp_v10_integrity.gain_calibration import parseval_frame_db  # noqa: E402
from exp_v11_align_gmm.timbre_cloner_v5 import extract_timbre_v5  # noqa: E402
from exp_v11_diphone_tts.diphone_bank import _edge_stats, _normalize_shared_band  # noqa: E402
from exp_v12_natural_vocoder.v12_bank import _patch_unsmoothed_uv_env  # noqa: E402
from exp_v12_natural_vocoder.vocoder_v12 import (  # noqa: E402
    GlottalPulseBank,
    VocoderFeaturesV12,
    analyze_v12,
    build_glottal_pulse_bank,
)
from exp_v13_utau_ceiling.v13_selector import VT_BIN_HI, VT_BIN_LO  # noqa: E402
from exp_v13_utau_ceiling.zh_bank_v13 import V13ZhToken  # noqa: E402
from exp_v14_fluent_zh_tts import (  # noqa: E402
    ALIGNMENTS_V14,
    FEATS_CACHE_V14,
    HOP,
    MANIFEST_V14,
    N_FFT,
    SR,
)
from exp_v14_fluent_zh_tts.align_v14 import (  # noqa: E402
    _to_vocoder_features,
    extract_or_load_ssb0502_features,
)


def _mean_removed_vt_vec(env_slice: np.ndarray, *, from_end: bool) -> np.ndarray:
    """Mean-removed [80, 7200] Hz VT spectral envelope vector averaged over <= 2 edge frames."""
    if len(env_slice) == 0:
        return np.zeros(VT_BIN_HI - VT_BIN_LO, dtype=np.float32)
    sl = env_slice[-2:] if from_end else env_slice[:2]
    v = np.mean(sl[:, VT_BIN_LO:VT_BIN_HI].astype(np.float64), axis=0)
    return (v - float(np.mean(v))).astype(np.float32)


@dataclass
class V14ZhToken(V13ZhToken):
    """Mandarin syllable/sub-syllable token enriched with utterance span, 3-pt F0, and quality metrics."""

    syl_start: int = 0
    syl_end: int = 0
    prev_tone: int = 0
    next_tone: int = 0
    pause_after_ms: float = 0.0
    f0_start_st: float = 0.0
    f0_mid_st: float = 0.0
    f0_end_st: float = 0.0
    voiced_frac: float = 1.0
    quality_penalty: float = 0.0
    edge_start_vt: np.ndarray = field(default_factory=lambda: np.zeros(VT_BIN_HI - VT_BIN_LO, dtype=np.float32))
    edge_end_vt: np.ndarray = field(default_factory=lambda: np.zeros(VT_BIN_HI - VT_BIN_LO, dtype=np.float32))


@dataclass
class V14ZhBank:
    """Multi-sample `SSB0502` voice bank storing both indexed tokens and unbroken utterance tracks."""

    zh_syllables: dict[tuple[str, str], list[V14ZhToken]] = field(default_factory=dict)
    zh_onsets: dict[str, list[V14ZhToken]] = field(default_factory=dict)
    zh_finals: dict[str, list[V14ZhToken]] = field(default_factory=dict)
    utt_feats: dict[str, dict[str, Any]] = field(default_factory=dict)
    utt_tokens_by_id: dict[str, list[V14ZhToken]] = field(default_factory=dict)
    en_diphones: dict[tuple[str, str], list[Any]] = field(default_factory=dict)
    en_phones: dict[str, list[Any]] = field(default_factory=dict)
    profiles_v2: dict[str, Any] = field(default_factory=dict)
    profiles_v5: dict[str, Any] = field(default_factory=dict)
    fallback_unit_bank: ub.UnitBank | None = None
    glottal_banks: dict[str, GlottalPulseBank] = field(default_factory=dict)
    zh_included_ids: list[str] = field(default_factory=list)
    zh_excluded_ids: list[str] = field(default_factory=list)
    en_included_ids: list[str] = field(default_factory=list)
    en_excluded_ids: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _compute_token_quality_penalty(
    o: str,
    f: str,
    dur_ms: float,
    v_frac: float,
    rime_v_frac: float,
    p80_db: float,
) -> tuple[bool, float]:
    """Return `(should_drop, soft_quality_penalty)` catching starved/clipped/whispered outliers."""
    if dur_ms < 45.0 or dur_ms > 430.0 or p80_db < -38.0:
        return True, 5.0
    if rime_v_frac < 0.25:
        return True, 5.0

    pen = 0.0
    if dur_ms < 70.0:
        pen += 0.05 * (70.0 - dur_ms)
    elif dur_ms > 340.0:
        pen += 0.03 * (dur_ms - 340.0)
    if (not o or o in ("m", "n", "l", "r")) and v_frac < 0.72:
        pen += 1.2 * (0.72 - v_frac)
    if rime_v_frac < 0.55:
        pen += 1.0 * (0.55 - rime_v_frac)
    return False, round(float(pen), 4)


def _extract_utt_tokens_v14(
    u: dict[str, Any],
    fe: dict[str, Any],
    al_entry: dict[str, Any],
    *,
    sil_aware: bool = True,
    quarantine_outliers: bool = True,
    sr: int = SR,
) -> tuple[dict[str, Any], list[V14ZhToken], list[V14ZhToken], list[V14ZhToken], int]:
    """Extract normalized continuous utterance track + indexed whole/onset/final `V14ZhToken`s."""
    uid = u["id"]
    spk = u.get("speaker", "SSB0502")
    xa64 = np.asarray(fe["xa"], dtype=np.float64)
    feats = _to_vocoder_features(fe)
    b_gt8 = np.asarray(fe["bap_gt8k"], dtype=np.float32)

    env_norm, wav_scale, f0_rel = _normalize_shared_band(feats, sr=sr)
    tracks = compute_landmark_tracks(xa64, feats, sr=sr)
    n_tot = len(env_norm)
    env_f32 = env_norm.astype(np.float32)
    bap_f32 = feats.bap.astype(np.float32)
    f0_rel_f32 = f0_rel.astype(np.float32)
    voi_raw_all = feats.voiced.astype(bool)

    utt_entry = {
        "env_norm": env_f32,
        "bap": bap_f32,
        "bap_gt8k": b_gt8,
        "voiced": voi_raw_all,
        "f0_rel_st": f0_rel_f32,
        "xa": fe["xa"],
        "wav_scale": float(wav_scale),
        "n_frames": int(n_tot),
    }

    raw_syls = al_entry["syllables"]
    n_syls = len(raw_syls)
    # Build effective frame spans (when `sil_aware=False`, reconstruct pre-SIL merged pauses for A2 ablation)
    spans: list[tuple[int, int, int, float]] = []
    for idx_s, s in enumerate(raw_syls):
        s_lo = max(0, min(n_tot - 3, int(s["syl_start"])))
        s_hi = max(s_lo + 3, min(n_tot, int(s["syl_end"])))
        p_ms = float(s.get("pause_after_ms", 0.0))
        if not sil_aware:
            if idx_s + 1 < n_syls:
                nxt_lo = max(s_hi, min(n_tot, int(raw_syls[idx_s + 1]["syl_start"])))
                s_hi = nxt_lo
            else:
                s_hi = n_tot
            p_ms = 0.0
        f_sp_rel = max(0, min(s_hi - s_lo - 2, int(s.get("f_split", 0))))
        spans.append((s_lo, f_sp_rel, s_hi, p_ms))

    syl_tokens: list[V14ZhToken] = []
    onset_tokens: list[V14ZhToken] = []
    final_tokens: list[V14ZhToken] = []
    ordered_by_syl: list[V14ZhToken] = []
    dropped = 0

    for idx_s, (s, (f_lo, f_sp_rel, f_hi, p_after_ms)) in enumerate(zip(raw_syls, spans)):
        o, f, tone = cf.pinyin_to_ipa(s["pinyin"])[:3]
        if o in ("y", "w"):
            o = ""
        if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
            f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"
        f_sp = f_lo + f_sp_rel
        dur_ms = float((f_hi - f_lo) * HOP * 1000.0 / sr)

        voi_sl = voi_raw_all[f_lo:f_hi].copy()
        le_sl = env_f32[f_lo:f_hi]
        onset_hflf = float(np.mean(tracks["hf_lf"][f_lo:f_sp])) if f_sp > f_lo else 0.0
        if o in OBSTRUENT_ONSETS and f_sp_rel > 0:
            for k_rel in range(f_sp_rel):
                if tracks["voi"][f_lo + k_rel] < 0.5 or tracks["hf_lf"][f_lo + k_rel] > -4.0:
                    voi_sl[k_rel] = False

        rime_voi = voi_sl[f_sp_rel:] if f_sp_rel < len(voi_sl) else voi_sl
        v_frac = float(np.mean(voi_sl)) if len(voi_sl) else 0.0
        rime_v_frac = float(np.mean(rime_voi)) if len(rime_voi) else 0.0
        p80_db = float(np.percentile(parseval_frame_db(le_sl, n_fft=N_FFT), 80))

        should_drop, q_pen = _compute_token_quality_penalty(o, f, dur_ms, v_frac, rime_v_frac, p80_db)
        if quarantine_outliers and (should_drop or int(np.sum(voi_sl)) < 4):
            dropped += 1
            continue

        prev_f = raw_syls[idx_s - 1]["final"] if idx_s > 0 else "SIL"
        prev_t = int(raw_syls[idx_s - 1]["tone"]) if idx_s > 0 else 0
        next_o = raw_syls[idx_s + 1]["onset"] if idx_s + 1 < n_syls else "SIL"
        next_t = int(raw_syls[idx_s + 1]["tone"]) if idx_s + 1 < n_syls else 0

        on_wav = None
        if o in OBSTRUENT_ONSETS and f_sp_rel > 0 and not np.any(voi_sl[:f_sp_rel]) and onset_hflf > -2.0:
            w0, w1 = max(0, f_lo * HOP), min(len(xa64), f_sp * HOP)
            if w1 > w0:
                on_wav = (xa64[w0:w1] * wav_scale).astype(np.float32)

        ba_sl = bap_f32[f_lo:f_hi]
        bg_sl = b_gt8[f_lo:f_hi]
        f0_sl = f0_rel_f32[f_lo:f_hi]
        el, er, dbl, dbr, fl, fr_e = _edge_stats(le_sl, f0_sl, voi_sl)

        rime_v_idx = np.where(voi_sl[f_sp_rel:])[0] + f_sp_rel
        if rime_v_idx.size < 2:
            rime_v_idx = np.where(voi_sl)[0]
        if rime_v_idx.size >= 2:
            v_sts = f0_sl[rime_v_idx].astype(np.float64)
            f0_3 = np.interp([0.2, 0.5, 0.8], np.linspace(0.0, 1.0, len(v_sts)), v_sts)
            f0_s_st, f0_m_st, f0_e_st = float(f0_3[0]), float(f0_3[1]), float(f0_3[2])
        elif rime_v_idx.size == 1:
            f0_s_st = f0_m_st = f0_e_st = float(f0_sl[rime_v_idx[0]])
        else:
            f0_s_st = f0_m_st = f0_e_st = 0.0

        p75_db = float(np.percentile(parseval_frame_db(le_sl, n_fft=N_FFT), 75))
        syl_pos = "initial" if idx_s == 0 else ("final" if idx_s == n_syls - 1 else "medial")
        vt_s = _mean_removed_vt_vec(le_sl, from_end=False)
        vt_e = _mean_removed_vt_vec(le_sl, from_end=True)

        tok = V14ZhToken(
            onset=o,
            final=f,
            tone=int(tone),
            speaker=spk,
            utterance_id=uid,
            prev_final=prev_f,
            next_onset=next_o,
            log_env=le_sl,
            bap=ba_sl,
            bap_gt8k=bg_sl,
            voiced=voi_sl,
            f0_rel_st=f0_sl,
            f_split=int(f_sp_rel),
            dur_ms=dur_ms,
            edge_left_env=el,
            edge_right_env=er,
            edge_left_db=dbl,
            edge_right_db=dbr,
            edge_f0_l=fl,
            edge_f0_r=fr_e,
            uv_hf_lf=onset_hflf,
            onset_wav=on_wav,
            syl_idx=idx_s,
            n_syls_in_utt=n_syls,
            mean_f0_st=f0_m_st,
            parseval_p75_db=p75_db,
            syllable_pos=syl_pos,
            syl_start=int(f_lo),
            syl_end=int(f_hi),
            prev_tone=prev_t,
            next_tone=next_t,
            pause_after_ms=p_after_ms,
            f0_start_st=f0_s_st,
            f0_mid_st=f0_m_st,
            f0_end_st=f0_e_st,
            voiced_frac=round(v_frac, 4),
            quality_penalty=q_pen,
            edge_start_vt=vt_s,
            edge_end_vt=vt_e,
        )
        syl_tokens.append(tok)
        ordered_by_syl.append(tok)

        if o and f_sp_rel >= 4:
            is_uv_o = o in ("f", "s", "sh", "x", "h", "c", "ch", "q")
            if (not quarantine_outliers) or (not is_uv_o) or onset_hflf > -5.5:
                le_o, v_o, f0_o = le_sl[:f_sp_rel], voi_sl[:f_sp_rel], f0_sl[:f_sp_rel]
                el_o, er_o, dbl_o, dbr_o, fl_o, fr_o = _edge_stats(le_o, f0_o, v_o)
                onset_tokens.append(
                    V14ZhToken(
                        onset=o, final="", tone=int(tone), speaker=spk, utterance_id=uid,
                        prev_final=prev_f, next_onset=f, log_env=le_o, bap=ba_sl[:f_sp_rel],
                        bap_gt8k=bg_sl[:f_sp_rel], voiced=v_o, f0_rel_st=f0_o, f_split=int(f_sp_rel),
                        dur_ms=float(f_sp_rel * HOP * 1000.0 / sr),
                        edge_left_env=el_o, edge_right_env=er_o, edge_left_db=dbl_o, edge_right_db=dbr_o,
                        edge_f0_l=fl_o, edge_f0_r=fr_o, uv_hf_lf=onset_hflf, onset_wav=on_wav,
                        syl_idx=idx_s, n_syls_in_utt=n_syls, mean_f0_st=f0_m_st,
                        parseval_p75_db=float(np.percentile(parseval_frame_db(le_o, n_fft=N_FFT), 75)),
                        syllable_pos=syl_pos, syl_start=int(f_lo), syl_end=int(f_sp),
                    )
                )

        if f_hi - f_sp >= 6:
            le_f, v_f, f0_f = le_sl[f_sp_rel:], voi_sl[f_sp_rel:], f0_sl[f_sp_rel:]
            if (not quarantine_outliers) or (
                int(np.sum(v_f)) >= 6
                and float(np.mean(v_f)) >= 0.30
                and float(np.percentile(parseval_frame_db(le_f, n_fft=N_FFT), 80)) >= -38.0
            ):
                el_f, er_f, dbl_f, dbr_f, fl_f, fr_f = _edge_stats(le_f, f0_f, v_f)
                v_f0_r = f0_f[v_f] if np.any(v_f) else f0_f
                mf0 = float(np.median(v_f0_r)) if v_f0_r.size else 0.0
                final_tokens.append(
                    V14ZhToken(
                        onset="", final=f, tone=int(tone), speaker=spk, utterance_id=uid,
                        prev_final=o or prev_f, next_onset=next_o, log_env=le_f, bap=ba_sl[f_sp_rel:],
                        bap_gt8k=bg_sl[f_sp_rel:], voiced=v_f, f0_rel_st=f0_f, f_split=0,
                        dur_ms=float(len(le_f) * HOP * 1000.0 / sr),
                        edge_left_env=el_f, edge_right_env=er_f, edge_left_db=dbl_f, edge_right_db=dbr_f,
                        edge_f0_l=fl_f, edge_f0_r=fr_f, uv_hf_lf=0.0, onset_wav=None,
                        syl_idx=idx_s, n_syls_in_utt=n_syls, mean_f0_st=mf0,
                        parseval_p75_db=float(np.percentile(parseval_frame_db(le_f, n_fft=N_FFT), 75)),
                        syllable_pos=syl_pos, syl_start=int(f_sp), syl_end=int(f_hi),
                        f0_mid_st=mf0,
                    )
                )

    return utt_entry, syl_tokens, onset_tokens, final_tokens, dropped


def _build_spk_profiles_and_glottal(
    train_utts: list[dict[str, Any]],
    feats_map: dict[str, dict[str, Any]],
    *,
    sr: int = SR,
    max_profile_utts: int = 24,
    max_glottal_utts: int = 16,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, GlottalPulseBank]]:
    items: list[tuple[np.ndarray, vocoder.VocoderFeatures]] = []
    fv12_list: list[VocoderFeaturesV12] = []
    for u in train_utts[:max_profile_utts]:
        fe = feats_map[u["id"]]
        xa = np.asarray(fe["xa"], dtype=np.float64)
        vf = _to_vocoder_features(fe)
        items.append((xa, vf))
        if len(fv12_list) < max_glottal_utts:
            fv12 = analyze_v12(xa, sr=sr)
            fv12_patched = VocoderFeaturesV12(
                feats=_patch_unsmoothed_uv_env(xa, fv12.feats, sr=sr, n_fft=N_FFT),
                bap_gt8k=fv12.bap_gt8k,
                uv_raw_log_env=None,
                lp_pulses=fv12.lp_pulses,
                coherent_phasor=fv12.coherent_phasor,
            )
            fv12_list.append(fv12_patched)

    profs_v2 = {"SSB0502": tc2.extract_v2_profile(items, sr=sr, name="SSB0502")} if items else {}
    profs_v5 = {"SSB0502": extract_timbre_v5(items, sr=sr, name="SSB0502", corpus="aishell3")} if items else {}
    gbanks = {"SSB0502": build_glottal_pulse_bank("SSB0502", fv12_list, max_pulses=32)} if fv12_list else {}
    return profs_v2, profs_v5, gbanks


def build_v14_zh_bank(
    *,
    manifest_path: Path | str = MANIFEST_V14,
    align_path: Path | str = ALIGNMENTS_V14,
    cache_path: Path | str = FEATS_CACHE_V14,
    exclude_ids: Iterable[str] = (),
    max_utts: int | None = None,
    seed: int = 42,
    sil_aware: bool = True,
    quarantine_outliers: bool = True,
    build_fallback_ub: bool = True,
    sr: int = SR,
) -> V14ZhBank:
    """Build multi-sample `SSB0502` bank excluding `exclude_ids` from every component (zero leakage)."""
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    al_doc = json.loads(Path(align_path).read_text(encoding="utf-8"))
    excl = set(exclude_ids)
    feats_map = extract_or_load_ssb0502_features(man["utterances"], cache_path=Path(cache_path))

    train_utts = [u for u in man["utterances"] if u["id"] not in excl and u["id"] in al_doc["zh_all"] and u["id"] in feats_map]
    if max_utts is not None and max_utts < len(train_utts):
        rng = random.Random(seed)
        chosen_ids = set(rng.sample([u["id"] for u in train_utts], max_utts))
        train_utts = [u for u in train_utts if u["id"] in chosen_ids]

    zh_syllables: dict[tuple[str, str], list[V14ZhToken]] = {}
    zh_onsets: dict[str, list[V14ZhToken]] = {}
    zh_finals: dict[str, list[V14ZhToken]] = {}
    utt_feats: dict[str, dict[str, Any]] = {}
    utt_tokens_by_id: dict[str, list[V14ZhToken]] = {}
    kept_syls, dropped_syls = 0, 0

    for u in train_utts:
        uid = u["id"]
        ue, syls, ons, fins, drp = _extract_utt_tokens_v14(
            u,
            feats_map[uid],
            al_doc["zh_all"][uid],
            sil_aware=sil_aware,
            quarantine_outliers=quarantine_outliers,
            sr=sr,
        )
        utt_feats[uid] = ue
        utt_tokens_by_id[uid] = syls
        kept_syls += len(syls)
        dropped_syls += drp
        for t in syls:
            zh_syllables.setdefault((t.onset, t.final), []).append(t)
        for t in ons:
            zh_onsets.setdefault(t.onset, []).append(t)
        for t in fins:
            zh_finals.setdefault(t.final, []).append(t)

    profs_v2, profs_v5, gbanks = _build_spk_profiles_and_glottal(train_utts, feats_map, sr=sr)

    fallback_ub = None
    if build_fallback_ub and train_utts:
        ub_utts = []
        ub_pre: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures]] = {}
        for u in train_utts[:36]:
            uid = u["id"]
            fe = feats_map[uid]
            al = al_doc["zh_all"][uid]
            act_s = int(fe["trim_start_sample"])
            ub_pre[uid] = (np.asarray(fe["xa"], dtype=np.float64), _to_vocoder_features(fe))
            ub_utts.append({
                "id": uid,
                "speaker": "SSB0502",
                "active_start": act_s,
                "active_stop": int(fe["trim_stop_sample"]),
                "syllables": [
                    {
                        "index": int(s["index"]),
                        "hanzi": s["hanzi"],
                        "pinyin": s["pinyin"],
                        "start_sample": act_s + int(s["syl_start"]) * HOP,
                        "stop_sample": act_s + int(s["syl_end"]) * HOP,
                        "ms": float(s["dur_ms"]),
                    }
                    for s in al["syllables"]
                ],
            })
        fallback_ub = ub.build_unit_bank(ub_utts, exclude_ids=excl, precomputed=ub_pre, sr=sr)

    return V14ZhBank(
        zh_syllables=zh_syllables,
        zh_onsets=zh_onsets,
        zh_finals=zh_finals,
        utt_feats=utt_feats,
        utt_tokens_by_id=utt_tokens_by_id,
        profiles_v2=profs_v2,
        profiles_v5=profs_v5,
        fallback_unit_bank=fallback_ub,
        glottal_banks=gbanks,
        zh_included_ids=[u["id"] for u in train_utts],
        zh_excluded_ids=sorted(excl),
        stats={
            "n_utts": len(train_utts),
            "zh_syl_keys": len(zh_syllables),
            "zh_syl_tokens": kept_syls,
            "zh_syl_dropped": dropped_syls,
            "zh_onsets": len(zh_onsets),
            "zh_onset_tokens": sum(len(v) for v in zh_onsets.values()),
            "zh_finals": len(zh_finals),
            "zh_final_tokens": sum(len(v) for v in zh_finals.values()),
            "sil_aware": sil_aware,
            "quarantine_outliers": quarantine_outliers,
        },
    )


def filter_v14_bank_excluding(
    bank: V14ZhBank,
    exclude_ids: Iterable[str],
) -> V14ZhBank:
    """Return a zero-leakage view of `bank` excluding all tokens and tracks from `exclude_ids`."""
    excl = set(exclude_ids)
    kept_ids = [uid for uid in bank.zh_included_ids if uid not in excl]
    return _slice_bank_by_ids(bank, kept_ids, new_excl=set(bank.zh_excluded_ids) | excl)


def subsample_v14_bank(
    bank: V14ZhBank,
    *,
    max_utts: int,
    seed: int = 42,
) -> V14ZhBank:
    """Deterministically subsample `bank` to `max_utts` included utterances for scaling studies."""
    if max_utts >= len(bank.zh_included_ids):
        return bank
    rng = random.Random(seed)
    chosen = set(rng.sample(bank.zh_included_ids, max_utts))
    kept_ids = [uid for uid in bank.zh_included_ids if uid in chosen]
    dropped = [uid for uid in bank.zh_included_ids if uid not in chosen]
    return _slice_bank_by_ids(bank, kept_ids, new_excl=set(bank.zh_excluded_ids) | set(dropped))


def _slice_bank_by_ids(bank: V14ZhBank, kept_ids: list[str], *, new_excl: set[str]) -> V14ZhBank:
    kept_set = set(kept_ids)
    sylls = {
        k: v
        for k, v in ((k, [t for t in toks if t.utterance_id in kept_set]) for k, toks in bank.zh_syllables.items())
        if v
    }
    onsets = {
        k: v
        for k, v in ((k, [t for t in toks if t.utterance_id in kept_set]) for k, toks in bank.zh_onsets.items())
        if v
    }
    finals = {
        k: v
        for k, v in ((k, [t for t in toks if t.utterance_id in kept_set]) for k, toks in bank.zh_finals.items())
        if v
    }
    utt_feats = {uid: bank.utt_feats[uid] for uid in kept_ids if uid in bank.utt_feats}
    utt_toks = {uid: bank.utt_tokens_by_id[uid] for uid in kept_ids if uid in bank.utt_tokens_by_id}

    # Rebuild profiles/glottal if any of the first utterances used earlier were dropped
    profs_v2, profs_v5, gbanks = bank.profiles_v2, bank.profiles_v5, bank.glottal_banks
    if not kept_set.issuperset(bank.zh_included_ids[: min(24, len(bank.zh_included_ids))]):
        items = []
        fv12_list = []
        for uid in kept_ids[:24]:
            ue = utt_feats[uid]
            xa = np.asarray(ue["xa"], dtype=np.float64)
            vf = vocoder.VocoderFeatures(
                f0_hz=np.zeros(ue["n_frames"], dtype=np.float64),  # unused by profile extraction when recomputed
                voiced=ue["voiced"],
                log_env=ue["env_norm"].astype(np.float64),
                bap=ue["bap"].astype(np.float64),
                sr=SR,
                hop_samples=HOP,
                n_samples=len(xa),
                n_fft=N_FFT,
            )
            if len(fv12_list) < 12:
                fv12 = analyze_v12(xa, sr=SR)
                items.append((xa, fv12.feats))
                fv12_list.append(
                    VocoderFeaturesV12(
                        feats=_patch_unsmoothed_uv_env(xa, fv12.feats, sr=SR, n_fft=N_FFT),
                        bap_gt8k=fv12.bap_gt8k,
                        lp_pulses=fv12.lp_pulses,
                        coherent_phasor=fv12.coherent_phasor,
                    )
                )
            else:
                items.append((xa, vf))
        if items:
            profs_v2 = {"SSB0502": tc2.extract_v2_profile(items, sr=SR, name="SSB0502")}
            profs_v5 = {"SSB0502": extract_timbre_v5(items, sr=SR, name="SSB0502", corpus="aishell3")}
        if fv12_list:
            gbanks = {"SSB0502": build_glottal_pulse_bank("SSB0502", fv12_list, max_pulses=32)}

    stats = dict(bank.stats)
    stats.update({
        "n_utts": len(kept_ids),
        "zh_syl_keys": len(sylls),
        "zh_syl_tokens": sum(len(v) for v in sylls.values()),
        "zh_onsets": len(onsets),
        "zh_onset_tokens": sum(len(v) for v in onsets.values()),
        "zh_finals": len(finals),
        "zh_final_tokens": sum(len(v) for v in finals.values()),
    })

    return V14ZhBank(
        zh_syllables=sylls,
        zh_onsets=onsets,
        zh_finals=finals,
        utt_feats=utt_feats,
        utt_tokens_by_id=utt_toks,
        profiles_v2=profs_v2,
        profiles_v5=profs_v5,
        fallback_unit_bank=bank.fallback_unit_bank,
        glottal_banks=gbanks,
        zh_included_ids=kept_ids,
        zh_excluded_ids=sorted(new_excl),
        stats=stats,
    )
