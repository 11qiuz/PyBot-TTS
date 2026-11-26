"""Phase 8 (v12) Expanded Dual Bank: 780 EN Diphones + 112 ZH Whole-Syllables + Speaker Glottal Banks."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.fft import rfft

from exp_v5_arch import crossling_frontend as cf
from exp_v6_crossling_dsp import arbiter as v6_arb
from exp_v6_vocoder import vocoder
from exp_v7_pybot_tts import unit_bank as ub
from exp_v8_clarity.en_utau_bank import _trim_active_span
from exp_v8_timbre_clone import timbre_cloner_v2 as tc2
from exp_v9_clarity_pro.boundary_refiner import compute_landmark_tracks, refine_zh_syllable_bounds
from exp_v9_clarity_pro.hnm_v9 import analyze_v9
from exp_v10_integrity.gain_calibration import parseval_frame_db
from exp_v11_diphone_tts.diphone_bank import (
    EnDiphoneToken,
    EnHalfOrPhoneToken,
    V11DiphoneDualBank,
    V11ZhToken,
    _edge_stats,
    _normalize_shared_band,
    _parse_gmm_en_entry,
    _parse_gmm_zh_syllables,
)
from exp_v12_natural_vocoder import EXTENDED_INDEX, HOP, N_FFT, REPO_ROOT, SR, V12_MANIFEST
from exp_v12_natural_vocoder.vocoder_v12 import GlottalPulseBank, analyze_v12, build_glottal_pulse_bank

GMM_ALIGNMENTS_V12 = REPO_ROOT / "exp_v12_prosody" / "gmm_alignments_v12.json"
OBSTRUENT_ONSETS = frozenset({"b", "p", "d", "t", "g", "k", "j", "q", "x", "zh", "ch", "sh", "z", "c", "s", "f", "h"})


@dataclass
class V12DualBank(V11DiphoneDualBank):
    """Expanded v12 bank adding per-speaker coherent glottal excitation banks & corpus metadata."""
    glottal_banks: dict[str, GlottalPulseBank] = field(default_factory=dict)


def _patch_unsmoothed_uv_env(xa: np.ndarray, feats: vocoder.VocoderFeatures, sr: int = SR, n_fft: int = N_FFT) -> vocoder.VocoderFeatures:
    """Replace cepstrally-smoothed unvoiced log_env frames with razor-sharp 10 ms 513-bin unsmoothed spectra."""
    if np.all(feats.voiced):
        return feats
    n_fr = len(feats.voiced)
    hop = feats.hop_samples
    win_len = min(240, max(128, int(round(0.010 * sr))))
    win = np.hanning(win_len)
    win /= math.sqrt(float(np.sum(win * win)))
    pad = win_len // 2
    x_pad = np.pad(np.asarray(xa, dtype=np.float64).ravel(), (pad, pad))

    log_env = feats.log_env.copy()
    for i in range(n_fr):
        if feats.voiced[i]:
            continue
        seg = x_pad[i * hop : i * hop + win_len]
        if len(seg) < win_len:
            continue
        xw = (seg - np.mean(seg)) * win
        pwr = np.abs(rfft(xw, n=n_fft)) ** 2
        log_e = 0.5 * np.log(np.maximum(pwr, 1e-18))
        fp = float(np.sum(xw * xw))
        el = np.exp(log_e)
        ep = float(el[0] ** 2 + 2.0 * np.sum(el[1:-1] ** 2) + el[-1] ** 2) / n_fft
        if ep > 1e-18 and fp > 1e-18:
            log_e += 0.5 * math.log(fp / ep)
        log_env[i] = log_e

    return vocoder.VocoderFeatures(
        f0_hz=feats.f0_hz,
        voiced=feats.voiced,
        log_env=log_env,
        bap=feats.bap,
        sr=feats.sr,
        hop_samples=feats.hop_samples,
        n_samples=feats.n_samples,
        n_fft=feats.n_fft,
    )


def build_v12_dual_bank(
    *,
    manifest_path: Path | str = V12_MANIFEST,
    gmm_cache_path: Path | str | None = GMM_ALIGNMENTS_V12,
    zh_exclude_ids: Iterable[str] = (),
    en_exclude_ids: Iterable[str] = (),
    zh_precomputed: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures, np.ndarray]] | None = None,
    en_precomputed: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures, np.ndarray]] | None = None,
    extract_v5_profiles: bool = True,
    max_glottal_utts_per_spk: int = 10,
    sr: int = SR,
) -> V12DualBank:
    """Build v12 expanded bank over 800 EN + 112 ZH utterances with crisp unvoiced envelopes & glottal banks."""
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    en_utterances = man["english"]["utterances"]

    # Merge extended_ref (32) + mandarin_expanded (80) preserving deduplication by id
    zh_raw = list(man.get("mandarin_extended_ref", {}).get("utterances", [])) + list(
        man.get("mandarin_expanded", {}).get("utterances", [])
    )
    if not zh_raw:
        zh_raw = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
    zh_seen: set[str] = set()
    zh_utterances: list[dict] = []
    for u in zh_raw:
        if u["id"] not in zh_seen:
            zh_seen.add(u["id"])
            zh_utterances.append(u)

    zh_excl = set(zh_exclude_ids)
    en_excl = set(en_exclude_ids)
    zh_train = [u for u in zh_utterances if u["id"] not in zh_excl]
    en_train = [u for u in en_utterances if u["id"] not in en_excl]

    gmm_en_map: dict[str, Any] = {}
    gmm_zh_map: dict[str, Any] = {}
    if gmm_cache_path is not None and Path(gmm_cache_path).exists():
        gcache = json.loads(Path(gmm_cache_path).read_text(encoding="utf-8"))
        gmm_en_map = gcache.get("en_train", {})
        gmm_zh_map = gcache.get("zh_all", {})

    # 1. Analyze / load Mandarin utterances
    zh_data, zh_for_ub, zh_by_spk, zh_v12_by_spk = {}, {}, {}, {}
    for u in zh_train:
        uid, spk = u["id"], u["speaker"]
        if zh_precomputed is not None and uid in zh_precomputed:
            xa, f9, bg = zh_precomputed[uid]
        else:
            p_w = Path(u["path"])
            if not p_w.is_absolute():
                p_w = REPO_ROOT / p_w
            x, fsr = vocoder.read_wav_mono(p_w)
            if fsr != sr:
                raise ValueError(f"Expected {sr} Hz for {uid}")
            ge = gmm_zh_map.get(uid)
            if ge and "trim_stop_sample" in ge and ge["trim_stop_sample"] > ge["trim_start_sample"]:
                xa = x[int(ge["trim_start_sample"]) : int(ge["trim_stop_sample"])]
            else:
                s0 = int(u.get("active_start", 0))
                s1 = int(u.get("active_stop", len(x)))
                xa = x[s0:s1]
            if len(zh_v12_by_spk.get(spk, ())) < max_glottal_utts_per_spk:
                fv12 = analyze_v12(xa, sr=sr)
                zh_v12_by_spk.setdefault(spk, []).append(fv12)
                f9, bg = _patch_unsmoothed_uv_env(xa, fv12.feats, sr=sr), fv12.bap_gt8k
            else:
                f9_raw, bg = analyze_v9(xa, sr=sr)
                f9 = _patch_unsmoothed_uv_env(xa, f9_raw, sr=sr)
        zh_data[uid] = (xa, f9, bg)
        zh_for_ub[uid] = (xa, f9)
        zh_by_spk.setdefault(spk, []).append((xa, f9))

    fallback_ub = ub.build_unit_bank(zh_train, exclude_ids=zh_excl, precomputed=zh_for_ub, sr=sr) if zh_train else None

    zh_syllables: dict[tuple[str, str], list[V11ZhToken]] = {}
    zh_onsets: dict[str, list[V11ZhToken]] = {}
    zh_finals: dict[str, list[V11ZhToken]] = {}
    zh_syl_kept, zh_syl_dropped = 0, 0

    for u in zh_train:
        uid, spk = u["id"], u["speaker"]
        xa, feats, b_gt8 = zh_data[uid]
        ge = gmm_zh_map.get(uid)
        act_start = int(ge["trim_start_sample"]) if (ge and "trim_start_sample" in ge) else int(u.get("active_start", 0))
        env_norm, wav_scale, f0_rel = _normalize_shared_band(feats, sr=sr)
        tracks = compute_landmark_tracks(xa, feats, sr=sr)
        n_tot = len(env_norm)
        syls = u["syllables"]
        parsed = [cf.pinyin_to_ipa(s["pinyin"])[:3] for s in syls]
        zh_gmm_syls = _parse_gmm_zh_syllables(ge)

        for idx_s, (syl, (o, f, tone)) in enumerate(zip(syls, parsed)):
            # Match canonical v6 arbiter final mapping
            if o in ("y", "w"):
                o = ""
            if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
                f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"

            if zh_gmm_syls and idx_s < len(zh_gmm_syls):
                gs_s, gs_sp, gs_e = zh_gmm_syls[idx_s]
                f_lo_r = max(0, min(n_tot - 3, int(gs_s)))
                f_hi_r = max(f_lo_r + 3, min(n_tot, int(gs_e)))
                f_sp = max(f_lo_r, min(f_hi_r - 1, int(gs_sp)))
            else:
                f_lo = max(0, min(n_tot - 2, int(round((int(syl["start_sample"]) - act_start) / HOP))))
                f_hi = max(f_lo + 2, min(n_tot, int(round((int(syl["stop_sample"]) - act_start) / HOP))))
                dur_s = max(0.07, float(syl.get("ms", 180.0)) / 1000.0)
                prop_fr = max(2, min((f_hi - f_lo) - 2, int(round(0.28 * (f_hi - f_lo))))) if o else 0
                f_lo_r, f_hi_r, f_sp = refine_zh_syllable_bounds(tracks, f_lo, f_hi, o, f, prop_fr)

            f_sp_rel = max(0, min(f_hi_r - f_lo_r, f_sp - f_lo_r))
            dur_ms = float((f_hi_r - f_lo_r) * HOP * 1000.0 / sr)
            if dur_ms < 50.0:
                zh_syl_dropped += 1
                continue

            voi_sl = feats.voiced[f_lo_r:f_hi_r].copy()
            le_sl = env_norm[f_lo_r:f_hi_r].astype(np.float32)
            rime_voi = voi_sl[f_sp_rel:] if f_sp_rel < len(voi_sl) else voi_sl
            p80_syl = float(np.percentile(parseval_frame_db(le_sl, n_fft=N_FFT), 80))
            if (
                int(np.sum(voi_sl)) < 4
                or (len(rime_voi) > 0 and float(np.mean(rime_voi)) < 0.25)
                or p80_syl < -38.0
            ):
                zh_syl_dropped += 1
                continue

            onset_hflf = float(np.mean(tracks["hf_lf"][f_lo_r:f_sp])) if f_sp > f_lo_r else 0.0
            if o in OBSTRUENT_ONSETS and f_sp_rel > 0:
                for k_rel in range(f_sp_rel):
                    if tracks["voi"][f_lo_r + k_rel] < 0.5 or tracks["hf_lf"][f_lo_r + k_rel] > -4.0:
                        voi_sl[k_rel] = False

            prev_f = parsed[idx_s - 1][1] if idx_s > 0 else "SIL"
            next_o = parsed[idx_s + 1][0] if idx_s + 1 < len(parsed) else "SIL"
            on_wav = None
            if o in OBSTRUENT_ONSETS and f_sp_rel > 0 and not np.any(voi_sl[:f_sp_rel]) and onset_hflf > -2.0:
                s0, s1 = max(0, f_lo_r * HOP), min(len(xa), f_sp * HOP)
                if s1 > s0:
                    on_wav = (xa[s0:s1] * wav_scale).astype(np.float32)

            ba_sl = feats.bap[f_lo_r:f_hi_r].astype(np.float32)
            bg_sl = b_gt8[f_lo_r:f_hi_r].astype(np.float32)
            f0_sl = f0_rel[f_lo_r:f_hi_r].astype(np.float32)
            el, er, dbl, dbr, fl, fr_e = _edge_stats(le_sl, f0_sl, voi_sl)

            tok = V11ZhToken(
                onset=o, final=f, tone=tone, speaker=spk, utterance_id=uid,
                prev_final=prev_f, next_onset=next_o,
                log_env=le_sl, bap=ba_sl, bap_gt8k=bg_sl, voiced=voi_sl.astype(bool), f0_rel_st=f0_sl,
                f_split=int(f_sp_rel), dur_ms=dur_ms,
                edge_left_env=el, edge_right_env=er, edge_left_db=dbl, edge_right_db=dbr,
                edge_f0_l=fl, edge_f0_r=fr_e, uv_hf_lf=onset_hflf, onset_wav=on_wav,
            )
            zh_syllables.setdefault((o, f), []).append(tok)
            zh_syl_kept += 1

            if o and f_sp_rel >= 4:
                is_uv_o = o in ("f", "s", "sh", "x", "h", "c", "ch", "q")
                if (not is_uv_o) or onset_hflf > -5.5:
                    le_o, v_o, f0_o = le_sl[:f_sp_rel], voi_sl[:f_sp_rel], f0_sl[:f_sp_rel]
                    el_o, er_o, dbl_o, dbr_o, fl_o, fr_o = _edge_stats(le_o, f0_o, v_o)
                    zh_onsets.setdefault(o, []).append(V11ZhToken(
                        onset=o, final="", tone=tone, speaker=spk, utterance_id=uid,
                        prev_final=prev_f, next_onset=f,
                        log_env=le_o, bap=ba_sl[:f_sp_rel], bap_gt8k=bg_sl[:f_sp_rel], voiced=v_o, f0_rel_st=f0_o,
                        f_split=int(f_sp_rel), dur_ms=float(f_sp_rel * HOP * 1000.0 / sr),
                        edge_left_env=el_o, edge_right_env=er_o, edge_left_db=dbl_o, edge_right_db=dbr_o,
                        edge_f0_l=fl_o, edge_f0_r=fr_o, uv_hf_lf=onset_hflf, onset_wav=on_wav,
                    ))

            if f_hi_r - f_sp >= 6:
                le_f, v_f, f0_f = le_sl[f_sp_rel:], voi_sl[f_sp_rel:], f0_sl[f_sp_rel:]
                if int(np.sum(v_f)) >= 6 and float(np.mean(v_f)) >= 0.30 and float(np.percentile(parseval_frame_db(le_f, n_fft=N_FFT), 80)) >= -38.0:
                    el_f, er_f, dbl_f, dbr_f, fl_f, fr_f = _edge_stats(le_f, f0_f, v_f)
                    zh_finals.setdefault(f, []).append(V11ZhToken(
                        onset="", final=f, tone=tone, speaker=spk, utterance_id=uid,
                        prev_final=o or prev_f, next_onset=next_o,
                        log_env=le_f, bap=ba_sl[f_sp_rel:], bap_gt8k=bg_sl[f_sp_rel:], voiced=v_f, f0_rel_st=f0_f,
                        f_split=0, dur_ms=float(len(le_f) * HOP * 1000.0 / sr),
                        edge_left_env=el_f, edge_right_env=er_f, edge_left_db=dbl_f, edge_right_db=dbr_f,
                        edge_f0_l=fl_f, edge_f0_r=fr_f, uv_hf_lf=0.0, onset_wav=None,
                    ))

    # 2. Analyze / load English utterances (up to 780 train) using GMM cache boundaries
    en_data, en_by_spk, en_v12_by_spk = {}, {}, {}
    for idx_e, u in enumerate(en_train, 1):
        uid, spk = u["id"], u["speaker"]
        if en_precomputed is not None and uid in en_precomputed:
            xa, f9, bg = en_precomputed[uid]
        else:
            p_w = Path(u["path"])
            if not p_w.is_absolute():
                p_w = REPO_ROOT / p_w
            x, fsr = vocoder.read_wav_mono(p_w)
            if fsr != sr:
                raise ValueError(f"Expected {sr} Hz for {uid}")
            ge = gmm_en_map.get(uid)
            if ge and 0 <= int(ge.get("trim_start_sample", 0)) < int(ge.get("trim_stop_sample", 0)) <= len(x):
                xa = x[int(ge["trim_start_sample"]) : int(ge["trim_stop_sample"])]
            else:
                s0, s1 = _trim_active_span(x, sr=sr)
                xa = x[s0:s1]
            if len(en_v12_by_spk.get(spk, ())) < max_glottal_utts_per_spk:
                fv12 = analyze_v12(xa, sr=sr)
                en_v12_by_spk.setdefault(spk, []).append(fv12)
                f9, bg = _patch_unsmoothed_uv_env(xa, fv12.feats, sr=sr), fv12.bap_gt8k
            else:
                f9_raw, bg = analyze_v9(xa, sr=sr)
                f9 = _patch_unsmoothed_uv_env(xa, f9_raw, sr=sr)
        en_data[uid] = (xa, f9, bg)
        en_by_spk.setdefault(spk, []).append((xa, f9))
        if idx_e % 200 == 0 or idx_e == len(en_train):
            print(f"  v12_bank analyzed {idx_e}/{len(en_train)} EN utts", flush=True)

    en_diphones: dict[tuple[str, str], list[EnDiphoneToken]] = {}
    en_phones: dict[str, list[EnHalfOrPhoneToken]] = {}
    diph_kept, diph_dropped = 0, 0

    for u in en_train:
        uid, spk = u["id"], u["speaker"]
        xa, feats, b_gt8 = en_data[uid]
        env_norm, wav_scale, f0_rel = _normalize_shared_band(feats, sr=sr)
        n_tot = len(env_norm)
        ge = gmm_en_map.get(uid)
        if ge is not None:
            phone_spans = _parse_gmm_en_entry(ge)
        else:
            from exp_v9_clarity_pro.boundary_refiner import align_en_phones_with_landmarks
            from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10
            pl_u = plan_v10(u["text"])
            pairs = []
            for unit in pl_u["units"]:
                arpa = unit.get("arpabet") or []
                if arpa:
                    pairs.extend(v6_arb._split_en_arpabet_to_ipa_with_stress(arpa))
                else:
                    for p in (unit.get("ipa_phones") or []):
                        pairs.append((p, None))
            ipas = [p for p, _ in pairs]
            if not ipas:
                continue
            bnds = align_en_phones_with_landmarks(ipas, feats, xa, sr=sr)
            phone_spans = [(ip, f0, (f0 + f1) // 2, f1, st) for (ip, st), (f0, f1) in zip(pairs, bnds)]

        if not phone_spans:
            continue

        for idx_p, (ipa, fs, _fm, fe, _st) in enumerate(phone_spans):
            fs_c, fe_c = max(0, fs), min(n_tot, fe)
            if fe_c - fs_c < 3:
                continue
            le = env_norm[fs_c:fe_c].astype(np.float32)
            ba = feats.bap[fs_c:fe_c].astype(np.float32)
            bg = b_gt8[fs_c:fe_c].astype(np.float32)
            vo = feats.voiced[fs_c:fe_c].astype(bool)
            f0 = f0_rel[fs_c:fe_c].astype(np.float32)
            el, er, dbl, dbr, fl_e, fr_e = _edge_stats(le, f0, vo)
            prev_ip = phone_spans[idx_p - 1][0] if idx_p > 0 else "SIL"
            next_ip = phone_spans[idx_p + 1][0] if idx_p + 1 < len(phone_spans) else "SIL"
            raw_w = (xa[fs_c * HOP : min(len(xa), fe_c * HOP)] * wav_scale).astype(np.float32) if not np.any(vo) else None
            en_phones.setdefault(ipa, []).append(EnHalfOrPhoneToken(
                ipa=ipa, speaker=spk, utterance_id=uid, prev_ipa=prev_ip, next_ipa=next_ip,
                log_env=le, bap=ba, bap_gt8k=bg, voiced=vo, f0_rel_st=f0,
                dur_ms=float((fe_c - fs_c) * HOP * 1000.0 / sr),
                edge_left_env=el, edge_right_env=er, edge_left_db=dbl, edge_right_db=dbr,
                edge_f0_l=fl_e, edge_f0_r=fr_e, raw_wav=raw_w,
            ))

        seq_nodes: list[tuple[str, int, int, int, int | None]] = (
            [("SIL", phone_spans[0][1], phone_spans[0][1], phone_spans[0][1], None)]
            + phone_spans
            + [("SIL", phone_spans[-1][3], phone_spans[-1][3], phone_spans[-1][3], None)]
        )
        for k in range(len(seq_nodes) - 1):
            p1, _s1, m1, e1, st1 = seq_nodes[k]
            p2, s2, m2, _e2, st2 = seq_nodes[k + 1]
            f_start = max(0, min(n_tot - 2, m1))
            f_end = max(f_start + 2, min(n_tot, m2))
            min_len = 2 if (p1 == "SIL" or p2 == "SIL") else 4
            if f_end - f_start < min_len:
                diph_dropped += 1
                continue
            le = env_norm[f_start:f_end].astype(np.float32)
            if float(np.percentile(parseval_frame_db(le, n_fft=N_FFT), 90)) < -46.0:
                diph_dropped += 1
                continue
            ba = feats.bap[f_start:f_end].astype(np.float32)
            bg = b_gt8[f_start:f_end].astype(np.float32)
            vo = feats.voiced[f_start:f_end].astype(bool)
            f0 = f0_rel[f_start:f_end].astype(np.float32)
            el, er, dbl, dbr, fl_e, fr_e = _edge_stats(le, f0, vo)
            mid_rel = max(1, min(len(le) - 1, ((e1 + s2) // 2) - f_start))
            raw_w = (xa[f_start * HOP : min(len(xa), f_end * HOP)] * wav_scale).astype(np.float32) if not np.any(vo) else None

            dtok = EnDiphoneToken(
                p1=p1, p2=p2, speaker=spk, utterance_id=uid, seq_idx=k, stress1=st1, stress2=st2,
                log_env=le, bap=ba, bap_gt8k=bg, voiced=vo, f0_rel_st=f0, f_mid_split=mid_rel,
                dur_ms=float((f_end - f_start) * HOP * 1000.0 / sr),
                edge_left_env=el, edge_right_env=er, edge_left_db=dbl, edge_right_db=dbr,
                edge_f0_l=fl_e, edge_f0_r=fr_e, raw_wav=raw_w,
            )
            en_diphones.setdefault((p1, p2), []).append(dtok)
            diph_kept += 1

    # 3. Speaker profiles (V2 + V5) and GlottalPulseBanks
    profs_v2, profs_v5, gbanks = {}, {}, {}
    for spk, items in sorted(zh_by_spk.items()):
        profs_v2[spk] = tc2.extract_v2_profile(items[:12], sr=sr, name=spk)
    for spk, items in sorted(en_by_spk.items()):
        profs_v2[spk] = tc2.extract_v2_profile(items[:12], sr=sr, name=spk)

    if extract_v5_profiles:
        from exp_v11_align_gmm.timbre_cloner_v5 import extract_timbre_v5
        for spk, items in sorted(zh_by_spk.items()):
            profs_v5[spk] = extract_timbre_v5(items[:12], sr=sr, name=spk, corpus="aishell3")
        for spk, items in sorted(en_by_spk.items()):
            profs_v5[spk] = extract_timbre_v5(items[:10], sr=sr, name=spk, corpus="arctic")

    for spk, items in sorted(zh_v12_by_spk.items()):
        gbanks[spk] = build_glottal_pulse_bank(spk, items, max_pulses=32)
    for spk, items in sorted(en_v12_by_spk.items()):
        gbanks[spk] = build_glottal_pulse_bank(spk, items, max_pulses=32)

    return V12DualBank(
        en_diphones=en_diphones,
        en_phones=en_phones,
        zh_syllables=zh_syllables,
        zh_onsets=zh_onsets,
        zh_finals=zh_finals,
        profiles_v2=profs_v2,
        profiles_v5=profs_v5,
        fallback_unit_bank=fallback_ub,
        zh_included_ids=sorted(zh_data.keys()),
        zh_excluded_ids=sorted(zh_excl),
        en_included_ids=sorted(en_data.keys()),
        en_excluded_ids=sorted(en_excl),
        glottal_banks=gbanks,
        stats={
            "unique_diphones": len(en_diphones),
            "total_diphone_tokens": diph_kept,
            "dropped_diphone_tokens": diph_dropped,
            "unique_en_phones": len(en_phones),
            "zh_syl_keys": len(zh_syllables),
            "zh_syl_tokens": zh_syl_kept,
            "zh_syl_dropped": zh_syl_dropped,
            "zh_onsets": len(zh_onsets),
            "zh_finals": len(zh_finals),
            "glottal_banks": sorted(gbanks.keys()),
            "used_gmm_cache": bool(gmm_en_map or gmm_zh_map),
        },
    )


def filter_v12_bank_excluding_utterance(
    bank: V12DualBank,
    exclude_uid: str,
    *,
    ub_fold: ub.UnitBank | None = None,
    spk_override: tuple[str, Any] | None = None,
) -> V12DualBank:
    """Fast zero-leakage LOO fold filtering removing `exclude_uid` from all token pools."""
    profs5 = dict(bank.profiles_v5)
    if spk_override is not None:
        profs5[spk_override[0]] = spk_override[1]
    return V12DualBank(
        en_diphones={k: v for k, v in ((k, [t for t in toks if t.utterance_id != exclude_uid]) for k, toks in bank.en_diphones.items()) if v},
        en_phones={k: v for k, v in ((k, [t for t in toks if t.utterance_id != exclude_uid]) for k, toks in bank.en_phones.items()) if v},
        zh_syllables={k: v for k, v in ((k, [t for t in toks if t.utterance_id != exclude_uid]) for k, toks in bank.zh_syllables.items()) if v},
        zh_onsets={k: v for k, v in ((k, [t for t in toks if t.utterance_id != exclude_uid]) for k, toks in bank.zh_onsets.items()) if v},
        zh_finals={k: v for k, v in ((k, [t for t in toks if t.utterance_id != exclude_uid]) for k, toks in bank.zh_finals.items()) if v},
        profiles_v2=bank.profiles_v2,
        profiles_v5=profs5,
        fallback_unit_bank=ub_fold if ub_fold is not None else bank.fallback_unit_bank,
        zh_included_ids=[i for i in bank.zh_included_ids if i != exclude_uid],
        zh_excluded_ids=sorted(set(bank.zh_excluded_ids) | {exclude_uid}),
        en_included_ids=[i for i in bank.en_included_ids if i != exclude_uid],
        en_excluded_ids=sorted(set(bank.en_excluded_ids) | {exclude_uid}),
        glottal_banks=bank.glottal_banks,
        stats=dict(bank.stats),
    )


def filter_v12_bank_excluding_speakers(bank: V12DualBank, excluded_speakers: Iterable[str]) -> V12DualBank:
    excl = set(excluded_speakers)
    en_d = {k: v for k, v in ((k, [t for t in toks if t.speaker not in excl]) for k, toks in bank.en_diphones.items()) if v}
    en_p = {k: v for k, v in ((k, [t for t in toks if t.speaker not in excl]) for k, toks in bank.en_phones.items()) if v}
    zh_s = {k: v for k, v in ((k, [t for t in toks if t.speaker not in excl]) for k, toks in bank.zh_syllables.items()) if v}
    zh_o = {k: v for k, v in ((k, [t for t in toks if t.speaker not in excl]) for k, toks in bank.zh_onsets.items()) if v}
    zh_f = {k: v for k, v in ((k, [t for t in toks if t.speaker not in excl]) for k, toks in bank.zh_finals.items()) if v}
    kept_en_ids = sorted({t.utterance_id for toks in en_d.values() for t in toks})
    return V12DualBank(
        en_diphones=en_d,
        en_phones=en_p,
        zh_syllables=zh_s,
        zh_onsets=zh_o,
        zh_finals=zh_f,
        profiles_v2={k: v for k, v in bank.profiles_v2.items() if k not in excl},
        profiles_v5={k: v for k, v in bank.profiles_v5.items() if k not in excl},
        fallback_unit_bank=bank.fallback_unit_bank,
        zh_included_ids=list(bank.zh_included_ids),
        zh_excluded_ids=list(bank.zh_excluded_ids),
        en_included_ids=kept_en_ids,
        en_excluded_ids=list(bank.en_excluded_ids),
        glottal_banks={k: v for k, v in bank.glottal_banks.items() if k not in excl},
        stats=dict(bank.stats),
    )
