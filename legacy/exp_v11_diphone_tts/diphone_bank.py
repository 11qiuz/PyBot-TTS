"""Phase 7 (v11) English Diphone Bank (P1_mid -> P2_mid) & Mandarin Whole-Syllable Rescue Bank."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.fft import rfftfreq

from pybot_tts_v2_recorded import engine
from pybot_tts_v4_params import vocal
from exp_v5_arch import crossling_frontend as cf
from exp_v6_crossling_dsp import arbiter as v6_arb
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS
from exp_v6_vocoder import vocoder
from exp_v7_pybot_tts import unit_bank as ub
from exp_v8_clarity.en_utau_bank import _trim_active_span
from exp_v8_clarity.utau_concat import OBSTRUENT_ONSETS
from exp_v8_timbre_clone import timbre_cloner_v2 as tc2
from exp_v9_clarity_pro.boundary_refiner import compute_landmark_tracks, refine_zh_syllable_bounds
from exp_v9_clarity_pro.hnm_v9 import analyze_v9
from exp_v10_integrity.gain_calibration import parseval_frame_db
from exp_v11_diphone_tts import EN_MANIFEST, EXTENDED_INDEX, HOP, N_FFT, REPO_ROOT, SR


@dataclass
class EnDiphoneToken:
    p1: str
    p2: str
    speaker: str
    utterance_id: str
    seq_idx: int                 # index inside utterance (enables free consecutive run chaining!)
    stress1: int | None
    stress2: int | None
    log_env: np.ndarray          # (T, 513) float32
    bap: np.ndarray              # (T, 5) float32
    bap_gt8k: np.ndarray         # (T, 2) float32
    voiced: np.ndarray           # (T,) bool
    f0_rel_st: np.ndarray        # (T,) float32
    f_mid_split: int             # frame index inside diphone separating p1_right from p2_left
    dur_ms: float
    edge_left_env: np.ndarray    # (513,) float32 mean-normed entry spectrum
    edge_right_env: np.ndarray   # (513,) float32 mean-normed exit spectrum
    edge_left_db: float
    edge_right_db: float
    edge_f0_l: float
    edge_f0_r: float
    raw_wav: np.ndarray | None = None


@dataclass
class EnHalfOrPhoneToken:
    ipa: str
    speaker: str
    utterance_id: str
    prev_ipa: str
    next_ipa: str
    log_env: np.ndarray
    bap: np.ndarray
    bap_gt8k: np.ndarray
    voiced: np.ndarray
    f0_rel_st: np.ndarray
    dur_ms: float
    edge_left_env: np.ndarray
    edge_right_env: np.ndarray
    edge_left_db: float
    edge_right_db: float
    edge_f0_l: float
    edge_f0_r: float
    raw_wav: np.ndarray | None = None


@dataclass
class V11ZhToken:
    onset: str
    final: str
    tone: int
    speaker: str
    utterance_id: str
    prev_final: str
    next_onset: str
    log_env: np.ndarray
    bap: np.ndarray
    bap_gt8k: np.ndarray
    voiced: np.ndarray
    f0_rel_st: np.ndarray
    f_split: int
    dur_ms: float
    edge_left_env: np.ndarray
    edge_right_env: np.ndarray
    edge_left_db: float
    edge_right_db: float
    edge_f0_l: float
    edge_f0_r: float
    uv_hf_lf: float = 0.0
    onset_wav: np.ndarray | None = None


@dataclass
class V11DiphoneDualBank:
    en_diphones: dict[tuple[str, str], list[EnDiphoneToken]] = field(default_factory=dict)
    en_phones: dict[str, list[EnHalfOrPhoneToken]] = field(default_factory=dict)
    zh_syllables: dict[tuple[str, str], list[V11ZhToken]] = field(default_factory=dict)
    zh_onsets: dict[str, list[V11ZhToken]] = field(default_factory=dict)
    zh_finals: dict[str, list[V11ZhToken]] = field(default_factory=dict)
    profiles_v2: dict[str, Any] = field(default_factory=dict)
    profiles_v5: dict[str, Any] = field(default_factory=dict)
    fallback_unit_bank: ub.UnitBank | None = None
    zh_included_ids: list[str] = field(default_factory=list)
    zh_excluded_ids: list[str] = field(default_factory=list)
    en_included_ids: list[str] = field(default_factory=list)
    en_excluded_ids: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _edge_stats(log_env: np.ndarray, f0_rel: np.ndarray, voiced: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    sl_l = log_env[: min(2, len(log_env))]
    sl_r = log_env[-min(2, len(log_env)) :]
    ml = np.mean(sl_l, axis=0)
    mr = np.mean(sl_r, axis=0)
    cmvn_l = (ml - float(np.mean(ml))).astype(np.float32)
    cmvn_r = (mr - float(np.mean(mr))).astype(np.float32)
    db_l = float(np.mean(parseval_frame_db(sl_l, n_fft=2 * (log_env.shape[1] - 1))))
    db_r = float(np.mean(parseval_frame_db(sl_r, n_fft=2 * (log_env.shape[1] - 1))))
    vm_l = voiced[: min(2, len(voiced))]
    vm_r = voiced[-min(2, len(voiced)) :]
    f0_l = float(np.mean(f0_rel[: len(vm_l)][vm_l])) if np.any(vm_l) else 0.0
    f0_r = float(np.mean(f0_rel[-len(vm_r) :][vm_r])) if np.any(vm_r) else 0.0
    return cmvn_l, cmvn_r, db_l, db_r, f0_l, f0_r


def _normalize_shared_band(feats: vocoder.VocoderFeatures, sr: int = SR) -> tuple[np.ndarray, float, np.ndarray]:
    """Normalize utterance voiced energy on shared [250, 7200] Hz Parseval RMS (-3.05 np) for EN+ZH balance."""
    v_mask = feats.voiced & (feats.f0_hz > 0.0)
    env_norm = feats.log_env.copy()
    wav_scale = 1.0
    if np.any(v_mask):
        freqs = rfftfreq(feats.n_fft, 1.0 / sr)
        m_sb = (freqs >= 250.0) & (freqs <= 7200.0)
        sb_rms_np = 0.5 * np.log(np.maximum(1e-18, np.mean(np.exp(2.0 * env_norm[v_mask][:, m_sb]), axis=1)))
        shift = -3.05 - float(np.median(sb_rms_np))
        env_norm = env_norm + shift
        wav_scale = math.exp(shift)
        f0_med = float(np.median(feats.f0_hz[v_mask]))
        f0_rel = np.where(v_mask, 12.0 * np.log2(np.maximum(50.0, feats.f0_hz) / max(50.0, f0_med)), 0.0)
    else:
        f0_rel = np.zeros(len(env_norm), dtype=np.float64)
    return env_norm, wav_scale, f0_rel


def _parse_gmm_en_entry(entry: Any) -> list[tuple[str, int, int, int, int | None]]:
    """Normalize GMM aligner entry (AlignedUtterance, dict, or tuple list) into `[(ipa, f_start, f_mid, f_end, stress)]`."""
    aligned = entry.get("aligned", entry) if isinstance(entry, dict) else entry
    phones = getattr(aligned, "phones", None)
    if phones is None:
        phones = aligned.get("phones", aligned) if isinstance(aligned, dict) else aligned
    out = []
    for item in phones:
        if hasattr(item, "ipa"):
            ipa = str(item.ipa)
            s = int(getattr(item, "p_start", getattr(item, "f_start", 0)))
            e = int(getattr(item, "p_end", getattr(item, "f_end", s + 2)))
            m = int(getattr(item, "p_mid", getattr(item, "f_mid", (s + e) // 2)))
            st = getattr(item, "stress_or_tone", getattr(item, "stress", None))
        elif isinstance(item, dict):
            ipa = str(item["ipa"])
            s = int(item.get("p_start", item.get("f_start", item.get("start_frame", 0))))
            e = int(item.get("p_end", item.get("f_end", item.get("end_frame", s + 2))))
            m = int(item.get("p_mid", item.get("f_mid", item.get("mid_frame", (s + e) // 2))))
            st = item.get("stress_or_tone", item.get("stress", None))
        else:
            # Tuple/list: (ipa, f_start, f_mid, f_end) or (ipa, f_start, f_end)
            ipa = str(item[0])
            s = int(item[1])
            if len(item) >= 4:
                m, e = int(item[2]), int(item[3])
            else:
                e = int(item[2])
                m = (s + e) // 2
            st = item[4] if len(item) >= 5 else None
        e = max(s + 1, e)
        m = max(s, min(e, m))
        out.append((ipa, s, m, e, st))
    return out


def _parse_gmm_zh_syllables(entry: Any) -> list[tuple[int, int, int]] | None:
    """Normalize ZH GMM aligner entry into `[(syl_start, f_split, syl_end)]`."""
    if entry is None:
        return None
    aligned = entry.get("aligned", entry) if isinstance(entry, dict) else entry
    syls = getattr(aligned, "syllables", None)
    if syls is None:
        syls = aligned.get("syllables", aligned) if isinstance(aligned, dict) else aligned
    if not syls:
        return None
    out = []
    for gs in syls:
        if hasattr(gs, "syl_start"):
            s = int(gs.syl_start)
            e = int(gs.syl_end)
            sp = int(getattr(gs, "f_split", (s + e) // 2))
        elif isinstance(gs, dict):
            s = int(gs.get("syl_start", gs.get("f_start", 0)))
            e = int(gs.get("syl_end", gs.get("f_end", s + 3)))
            sp = int(gs.get("f_split", (s + e) // 2))
        else:
            s, sp, e = int(gs[0]), int(gs[1]), int(gs[2])
        out.append((s, sp, e))
    return out


def build_v11_dual_bank(
    *,
    en_gmm_alignments: dict[str, Any] | None = None,
    zh_gmm_alignments: dict[str, Any] | None = None,
    zh_utterances: list[dict] | None = None,
    zh_exclude_ids: Iterable[str] = (),
    en_utterances: list[dict] | None = None,
    en_exclude_ids: Iterable[str] = (),
    zh_precomputed: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures, np.ndarray]] | None = None,
    en_precomputed: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures, np.ndarray]] | None = None,
    extract_v5_profiles: bool = True,
    sr: int = SR,
) -> V11DiphoneDualBank:
    """Build v11 English Diphone Bank (`P1_mid -> P2_mid`) + Mandarin Whole-Syllable Bank.

    Accepts external GMM-HMM alignments (`en_gmm_alignments`, `zh_gmm_alignments`) from
    `exp_v11_align_gmm.gmm_hmm_aligner`, or falls back cleanly to acoustic landmark DP.
    """
    if zh_utterances is None:
        zh_utterances = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
    if en_utterances is None:
        en_utterances = json.loads(EN_MANIFEST.read_text(encoding="utf-8"))["english"]["utterances"]

    zh_excl = set(zh_exclude_ids)
    en_excl = set(en_exclude_ids)
    zh_train = [u for u in zh_utterances if u["id"] not in zh_excl]
    en_train = [u for u in en_utterances if u["id"] not in en_excl]

    # Auto-load exp_v11_align_gmm if installed and caller omitted explicit dicts
    if en_gmm_alignments is None and en_train:
        try:
            from exp_v11_align_gmm.gmm_hmm_aligner import align_en_corpus_gmm
            en_gmm_alignments = align_en_corpus_gmm(exclude_ids=en_excl)
        except Exception:
            en_gmm_alignments = None
    if zh_gmm_alignments is None and zh_train:
        try:
            from exp_v11_align_gmm.gmm_hmm_aligner import align_zh_corpus_gmm
            zh_gmm_alignments = align_zh_corpus_gmm(exclude_ids=zh_excl)
        except Exception:
            zh_gmm_alignments = None

    en_gmm_map = (
        en_gmm_alignments.get("utterances", en_gmm_alignments)
        if isinstance(en_gmm_alignments, dict)
        else en_gmm_alignments
    )
    zh_gmm_map = (
        zh_gmm_alignments.get("utterances", zh_gmm_alignments)
        if isinstance(zh_gmm_alignments, dict)
        else zh_gmm_alignments
    )

    # 1. Analyze / load Mandarin utterances
    zh_data, zh_for_ub, zh_by_spk = {}, {}, {}
    for u in zh_train:
        uid = u["id"]
        if zh_precomputed is not None and uid in zh_precomputed:
            xa, f9, bg = zh_precomputed[uid]
        elif zh_gmm_map and uid in zh_gmm_map and isinstance(zh_gmm_map[uid], dict) and "feats" in zh_gmm_map[uid]:
            g_e = zh_gmm_map[uid]
            xa = g_e["wav"]
            f9 = g_e["feats"]
            bg = np.repeat(f9.bap[:, 4:5], 2, axis=1).astype(np.float32)
        else:
            x, fsr = vocoder.read_wav_mono(u["path"])
            if fsr != sr:
                raise ValueError(f"Expected {sr} Hz for {uid}")
            xa = x[u["active_start"] : u["active_stop"]]
            f9, bg = analyze_v9(xa, sr=sr)
        zh_data[uid] = (xa, f9, bg)
        zh_for_ub[uid] = (xa, f9)
        zh_by_spk.setdefault(u["speaker"], []).append((xa, f9))

    fallback_ub = ub.build_unit_bank(zh_train, exclude_ids=zh_excl, precomputed=zh_for_ub, sr=sr) if zh_train else None

    zh_syllables: dict[tuple[str, str], list[V11ZhToken]] = {}
    zh_onsets: dict[str, list[V11ZhToken]] = {}
    zh_finals: dict[str, list[V11ZhToken]] = {}
    zh_syl_kept, zh_syl_dropped = 0, 0

    for u in zh_train:
        uid, spk = u["id"], u["speaker"]
        xa, feats, b_gt8 = zh_data[uid]
        act_start = int(u["active_start"])
        env_norm, wav_scale, f0_rel = _normalize_shared_band(feats, sr=sr)
        tracks = compute_landmark_tracks(xa, feats, sr=sr)
        n_tot = len(env_norm)
        syls = u["syllables"]
        parsed = [engine.split_pinyin(s["pinyin"]) for s in syls]

        zh_gmm_entry = zh_gmm_map.get(uid) if zh_gmm_map else None
        zh_gmm_syls = _parse_gmm_zh_syllables(zh_gmm_entry)

        for idx_s, (syl, (o, f, tone)) in enumerate(zip(syls, parsed)):
            if zh_gmm_syls and idx_s < len(zh_gmm_syls):
                gs_s, gs_sp, gs_e = zh_gmm_syls[idx_s]
                f_lo_r = max(0, min(n_tot - 3, int(gs_s)))
                f_hi_r = max(f_lo_r + 3, min(n_tot, int(gs_e)))
                f_sp = max(f_lo_r, min(f_hi_r - 1, int(gs_sp)))
            else:
                f_lo = max(0, min(n_tot - 2, int(round((int(syl["start_sample"]) - act_start) / HOP))))
                f_hi = max(f_lo + 2, min(n_tot, int(round((int(syl["stop_sample"]) - act_start) / HOP))))
                dur_s = max(0.07, float(syl["ms"]) / 1000.0)
                _, c_dur = vocal._split_duration(o, dur_s) if o else (0.0, 0.0)
                prop_fr = max(2, min((f_hi - f_lo) - 2, int(round((c_dur / dur_s) * (f_hi - f_lo))))) if o else 0
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

            # Only store clean >= 20 ms onset slices (eliminating 5 ms clipped stumps!)
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

    # 2. Analyze / load English utterances & extract Diphones (P1_mid -> P2_mid) + fallback phones
    en_data, en_by_spk = {}, {}
    for u in en_train:
        uid = u["id"]
        if en_precomputed is not None and uid in en_precomputed:
            xa, f9, bg = en_precomputed[uid]
        elif en_gmm_map and uid in en_gmm_map and isinstance(en_gmm_map[uid], dict) and "feats" in en_gmm_map[uid]:
            g_e = en_gmm_map[uid]
            xa = g_e["wav"]
            f9 = g_e["feats"]
            bg = np.repeat(f9.bap[:, 4:5], 2, axis=1).astype(np.float32)
        else:
            p_w = Path(u["path"])
            if not p_w.is_absolute():
                p_w = REPO_ROOT / p_w
            x, fsr = vocoder.read_wav_mono(p_w)
            if fsr != sr:
                raise ValueError(f"Expected {sr} Hz for {uid}")
            s0, s1 = _trim_active_span(x, sr=sr)
            xa = x[s0:s1]
            f9, bg = analyze_v9(xa, sr=sr)
        en_data[uid] = (xa, f9, bg)
        en_by_spk.setdefault(u["speaker"], []).append((xa, f9))

    en_diphones: dict[tuple[str, str], list[EnDiphoneToken]] = {}
    en_phones: dict[str, list[EnHalfOrPhoneToken]] = {}
    diph_kept, diph_dropped = 0, 0

    for u in en_train:
        uid, spk = u["id"], u["speaker"]
        xa, feats, b_gt8 = en_data[uid]
        env_norm, wav_scale, f0_rel = _normalize_shared_band(feats, sr=sr)
        n_tot = len(env_norm)

        if en_gmm_map is not None and uid in en_gmm_map:
            phone_spans = _parse_gmm_en_entry(en_gmm_map[uid])
        else:
            # Fallback acoustic-landmark DP alignment with midpoint split
            from exp_v9_clarity_pro.boundary_refiner import align_en_phones_with_landmarks
            try:
                from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10
                pl_u = plan_v10(u["text"])
            except Exception:
                pl_u = cf.plan(u["text"], allow_oov_fallback=True)
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

        # Single-phone tokens for fallback when a rare diphone is unseen
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

        # Diphone tokens: ("SIL", p0) -> (p0, p1) -> ... -> (p_{M-1}, "SIL") from midpoints!
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
            # Reject completely dead silence diphones (unless both are SIL)
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

    # 3. Extract unified speaker profiles (V2 + V5 when available)
    profs_v2, profs_v5 = {}, {}
    for spk, items in sorted(zh_by_spk.items()):
        profs_v2[spk] = tc2.extract_v2_profile(items[:12], sr=sr, name=spk)
    for spk, items in sorted(en_by_spk.items()):
        profs_v2[spk] = tc2.extract_v2_profile(items[:12], sr=sr, name=spk)

    if extract_v5_profiles:
        try:
            from exp_v11_align_gmm.timbre_cloner_v5 import extract_timbre_v5
            for spk, items in sorted(zh_by_spk.items()):
                profs_v5[spk] = extract_timbre_v5(items[:12], sr=sr, name=spk, corpus="aishell3")
            for spk, items in sorted(en_by_spk.items()):
                profs_v5[spk] = extract_timbre_v5(items[:10], sr=sr, name=spk, corpus="arctic")
        except Exception:
            try:
                from exp_v10_g2p_and_align.timbre_cloner_v4 import extract_timbre_v4
                for spk, items in sorted(zh_by_spk.items()):
                    profs_v5[spk] = extract_timbre_v4(items[:12], sr=sr, name=spk, corpus="aishell3")
                for spk, items in sorted(en_by_spk.items()):
                    profs_v5[spk] = extract_timbre_v4(items[:10], sr=sr, name=spk, corpus="arctic")
            except Exception:
                profs_v5 = {}

    return V11DiphoneDualBank(
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
            "used_en_gmm": en_gmm_alignments is not None,
            "used_zh_gmm": zh_gmm_alignments is not None,
        },
    )


def filter_v11_bank_excluding_speakers(bank: V11DiphoneDualBank, excluded_speakers: Iterable[str]) -> V11DiphoneDualBank:
    excl = set(excluded_speakers)
    en_d = {k: v for k, v in ((k, [t for t in toks if t.speaker not in excl]) for k, toks in bank.en_diphones.items()) if v}
    en_p = {k: v for k, v in ((k, [t for t in toks if t.speaker not in excl]) for k, toks in bank.en_phones.items()) if v}
    zh_s = {k: v for k, v in ((k, [t for t in toks if t.speaker not in excl]) for k, toks in bank.zh_syllables.items()) if v}
    zh_o = {k: v for k, v in ((k, [t for t in toks if t.speaker not in excl]) for k, toks in bank.zh_onsets.items()) if v}
    zh_f = {k: v for k, v in ((k, [t for t in toks if t.speaker not in excl]) for k, toks in bank.zh_finals.items()) if v}
    kept_en_ids = sorted({t.utterance_id for toks in en_d.values() for t in toks})
    return V11DiphoneDualBank(
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
        stats=dict(bank.stats),
    )
