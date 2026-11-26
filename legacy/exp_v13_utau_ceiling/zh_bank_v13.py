"""Fast Mandarin-only v13 voice bank with per-utterance feature caching and UTAU metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from exp_v5_arch import crossling_frontend as cf
from exp_v6_vocoder import vocoder
from exp_v7_pybot_tts import unit_bank as ub
from exp_v8_clarity.utau_concat import OBSTRUENT_ONSETS
from exp_v8_timbre_clone import timbre_cloner_v2 as tc2
from exp_v9_clarity_pro.boundary_refiner import compute_landmark_tracks, refine_zh_syllable_bounds
from exp_v9_clarity_pro.hnm_v9 import analyze_v9
from exp_v10_integrity.gain_calibration import parseval_frame_db
from exp_v11_align_gmm.timbre_cloner_v5 import extract_timbre_v5
from exp_v11_diphone_tts.diphone_bank import (
    V11ZhToken,
    _edge_stats,
    _normalize_shared_band,
    _parse_gmm_zh_syllables,
)
from exp_v12_natural_vocoder.v12_bank import _patch_unsmoothed_uv_env
from exp_v12_natural_vocoder.vocoder_v12 import (
    GlottalPulseBank,
    VocoderFeaturesV12,
    analyze_v12,
    build_glottal_pulse_bank,
)
from exp_v13_utau_ceiling import (
    EXTENDED_INDEX,
    GMM_ALIGNMENTS_V12,
    HOP,
    N_FFT,
    REPO_ROOT,
    SR,
    V12_MANIFEST,
    ZH_FEATS_CACHE,
)


@dataclass
class V13ZhToken(V11ZhToken):
    """Mandarin syllable/sub-syllable token enriched with UTAU selection metadata."""

    syl_idx: int = 0
    n_syls_in_utt: int = 1
    mean_f0_st: float = 0.0
    parseval_p75_db: float = -24.0
    syllable_pos: str = "medial"


@dataclass
class V13ZhBank:
    """Fast Mandarin-only voice bank compatible with v12_renderer and v13_renderer."""

    zh_syllables: dict[tuple[str, str], list[V13ZhToken]] = field(default_factory=dict)
    zh_onsets: dict[str, list[V13ZhToken]] = field(default_factory=dict)
    zh_finals: dict[str, list[V13ZhToken]] = field(default_factory=dict)
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


def load_zh_manifest_utterances(manifest_path: Path | str = V12_MANIFEST) -> list[dict[str, Any]]:
    """Load deduplicated Mandarin utterances from manifest_v12.json (32 extended + 80 expanded)."""
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    zh_raw = list(man.get("mandarin_extended_ref", {}).get("utterances", [])) + list(
        man.get("mandarin_expanded", {}).get("utterances", [])
    )
    if not zh_raw:
        zh_raw = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for u in zh_raw:
        if u["id"] not in seen:
            seen.add(u["id"])
            out.append(u)
    return out


def load_gmm_zh_map(gmm_cache_path: Path | str | None = GMM_ALIGNMENTS_V12) -> dict[str, Any]:
    if gmm_cache_path is None or not Path(gmm_cache_path).exists():
        return {}
    gcache = json.loads(Path(gmm_cache_path).read_text(encoding="utf-8"))
    return gcache.get("zh_all", {})


def _analyze_or_load_zh_features(
    utterances: list[dict[str, Any]],
    gmm_zh_map: dict[str, Any],
    *,
    sr: int = SR,
    use_cache: bool = True,
    cache_path: Path = ZH_FEATS_CACHE,
) -> dict[str, dict[str, Any]]:
    """Load or compute per-utterance signal features (strictly independent per utterance ID)."""
    cache: dict[str, dict[str, Any]] = {}
    if use_cache and cache_path.exists():
        try:
            loaded = pickle.loads(cache_path.read_bytes())
            if isinstance(loaded, dict) and loaded.get("version") == 1:
                cache = dict(loaded.get("items", {}))
        except Exception:
            cache = {}

    dirty = False
    out: dict[str, dict[str, Any]] = {}
    for u in utterances:
        uid = u["id"]
        entry = cache.get(uid)
        if entry is None:
            p_w = Path(u["path"])
            if not p_w.is_absolute():
                p_w = REPO_ROOT / p_w
            x, fsr = vocoder.read_wav_mono(p_w)
            if fsr != sr:
                raise ValueError(f"Expected {sr} Hz for {uid}, got {fsr}")
            ge = gmm_zh_map.get(uid)
            if ge and "trim_stop_sample" in ge and ge["trim_stop_sample"] > ge["trim_start_sample"]:
                act_start = int(ge["trim_start_sample"])
                act_stop = int(ge["trim_stop_sample"])
            else:
                act_start = int(u.get("active_start", 0))
                act_stop = int(u.get("active_stop", len(x)))
            xa = x[act_start:act_stop]
            fv12 = analyze_v12(xa, sr=sr)
            f9_patched = _patch_unsmoothed_uv_env(xa, fv12.feats, sr=sr)
            entry = {
                "xa": xa.astype(np.float32),
                "feats": f9_patched,
                "bap_gt8k": fv12.bap_gt8k.astype(np.float32),
                "coherent_phasor": fv12.coherent_phasor,
                "lp_pulses": fv12.lp_pulses,
                "act_start": act_start,
            }
            cache[uid] = entry
            dirty = True
        out[uid] = entry

    if use_cache and dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps({"version": 1, "items": cache}, protocol=pickle.HIGHEST_PROTOCOL))
    return out


def _refine_mandarin_f_split(
    tracks: dict[str, np.ndarray],
    f_lo_r: int,
    f_hi_r: int,
    onset: str,
    gs_sp: int,
) -> int:
    """Refine syllable-relative onset/rime boundary `f_sp_rel` using periodicity cliff + manner caps.

    Prevents oversized GMM onset boundaries (which sometimes capture 60-85% of the syllable into
    the onset) from corrupting `zh_onsets`, `_stretch_preserving_onset`, and voicing masks.
    """
    n_cur = f_hi_r - f_lo_r
    if not onset or n_cur < 4:
        return 0

    if 0 < int(gs_sp) < n_cur:
        raw_sp = int(gs_sp)
    elif f_lo_r < int(gs_sp) < f_hi_r:
        raw_sp = int(gs_sp) - f_lo_r
    else:
        raw_sp = max(2, min(n_cur - 2, int(round(0.28 * n_cur))))

    voi = tracks["voi"]
    hf_lf = tracks["hf_lf"]

    if onset in ("b", "d", "g"):
        max_sp = min(n_cur - 3, max(4, min(7, int(round(0.22 * n_cur)))))
        min_sp = 2
    elif onset in ("p", "t", "k"):
        max_sp = min(n_cur - 3, max(6, min(20, int(round(0.45 * n_cur)))))
        min_sp = 3
    elif onset in OBSTRUENT_ONSETS:
        max_sp = min(n_cur - 3, max(6, min(24, int(round(0.50 * n_cur)))))
        min_sp = 3
    else:
        # Sonorants (m, n, l, r): purely voiced nasals/liquids
        return max(2, min(raw_sp, min(12, max(4, int(round(0.32 * n_cur))))))

    # Locate the periodicity cliff (first sustained voiced vowel onset) inside [min_sp, max_sp]
    cliff_sp: int | None = None
    for rel in range(min_sp, max_sp + 1):
        abs_i = f_lo_r + rel
        if abs_i + 1 < f_hi_r and voi[abs_i] > 0.5 and voi[abs_i + 1] > 0.5 and hf_lf[abs_i] < 5.0:
            cliff_sp = rel
            break

    if cliff_sp is not None:
        return int(cliff_sp)
    return int(max(min_sp, min(raw_sp, max_sp)))


def extract_utterance_zh_tokens(
    u: dict[str, Any],
    xa: np.ndarray,
    feats: vocoder.VocoderFeatures,
    b_gt8: np.ndarray,
    ge: dict[str, Any] | None,
    *,
    act_start: int | None = None,
    sr: int = SR,
    fix_f_split: bool = True,
    apply_quality_filter: bool = True,
) -> tuple[list[V13ZhToken], list[V13ZhToken], list[V13ZhToken], int]:
    """Extract whole-syllable, onset, and final V13ZhToken items from one Mandarin utterance."""
    uid, spk = u["id"], u["speaker"]
    xa64 = np.asarray(xa, dtype=np.float64)
    if act_start is None:
        act_start = int(ge["trim_start_sample"]) if (ge and "trim_start_sample" in ge) else int(u.get("active_start", 0))

    env_norm, wav_scale, f0_rel = _normalize_shared_band(feats, sr=sr)
    tracks = compute_landmark_tracks(xa64, feats, sr=sr)
    n_tot = len(env_norm)
    syls = u["syllables"]
    n_syls = len(syls)
    parsed = [cf.pinyin_to_ipa(s["pinyin"])[:3] for s in syls]
    zh_gmm_syls = _parse_gmm_zh_syllables(ge)

    syl_tokens: list[V13ZhToken] = []
    onset_tokens: list[V13ZhToken] = []
    final_tokens: list[V13ZhToken] = []
    dropped = 0

    for idx_s, (syl, (o, f, tone)) in enumerate(zip(syls, parsed)):
        if o in ("y", "w"):
            o = ""
        if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
            f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"

        if zh_gmm_syls and idx_s < len(zh_gmm_syls):
            gs_s, gs_sp, gs_e = zh_gmm_syls[idx_s]
            f_lo_r = max(0, min(n_tot - 3, int(gs_s)))
            f_hi_r = max(f_lo_r + 3, min(n_tot, int(gs_e)))
            if fix_f_split:
                f_sp_rel = _refine_mandarin_f_split(tracks, f_lo_r, f_hi_r, o, int(gs_sp))
                f_sp = f_lo_r + f_sp_rel
            else:
                # Exact v12_bank bug reproduction for controlled diagnostic comparison
                f_sp = max(f_lo_r, min(f_hi_r - 1, int(gs_sp)))
                f_sp_rel = max(0, min(f_hi_r - f_lo_r, f_sp - f_lo_r))
        else:
            f_lo = max(0, min(n_tot - 2, int(round((int(syl["start_sample"]) - act_start) / HOP))))
            f_hi = max(f_lo + 2, min(n_tot, int(round((int(syl["stop_sample"]) - act_start) / HOP))))
            prop_fr = max(2, min((f_hi - f_lo) - 2, int(round(0.28 * (f_hi - f_lo))))) if o else 0
            f_lo_r, f_hi_r, f_sp = refine_zh_syllable_bounds(tracks, f_lo, f_hi, o, f, prop_fr)
            f_sp_rel = max(0, min(f_hi_r - f_lo_r, f_sp - f_lo_r))

        dur_ms = float((f_hi_r - f_lo_r) * HOP * 1000.0 / sr)
        if apply_quality_filter and dur_ms < 50.0:
            dropped += 1
            continue

        voi_sl = feats.voiced[f_lo_r:f_hi_r].copy()
        le_sl = env_norm[f_lo_r:f_hi_r].astype(np.float32)
        rime_voi = voi_sl[f_sp_rel:] if f_sp_rel < len(voi_sl) else voi_sl
        p80_syl = float(np.percentile(parseval_frame_db(le_sl, n_fft=N_FFT), 80))
        if apply_quality_filter and (
            int(np.sum(voi_sl)) < 4
            or (len(rime_voi) > 0 and float(np.mean(rime_voi)) < 0.25)
            or p80_syl < -38.0
        ):
            dropped += 1
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
            s0, s1 = max(0, f_lo_r * HOP), min(len(xa64), f_sp * HOP)
            if s1 > s0:
                on_wav = (xa64[s0:s1] * wav_scale).astype(np.float32)

        ba_sl = feats.bap[f_lo_r:f_hi_r].astype(np.float32)
        bg_sl = b_gt8[f_lo_r:f_hi_r].astype(np.float32)
        f0_sl = f0_rel[f_lo_r:f_hi_r].astype(np.float32)
        el, er, dbl, dbr, fl, fr_e = _edge_stats(le_sl, f0_sl, voi_sl)

        v_f0 = f0_sl[voi_sl] if np.any(voi_sl) else f0_sl
        mean_f0_st = float(np.median(v_f0)) if v_f0.size else 0.0
        p75_db = float(np.percentile(parseval_frame_db(le_sl, n_fft=N_FFT), 75))
        syl_pos = "initial" if idx_s == 0 else ("final" if idx_s == n_syls - 1 else "medial")

        tok = V13ZhToken(
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
            mean_f0_st=mean_f0_st,
            parseval_p75_db=p75_db,
            syllable_pos=syl_pos,
        )
        syl_tokens.append(tok)

        if o and f_sp_rel >= 4:
            is_uv_o = o in ("f", "s", "sh", "x", "h", "c", "ch", "q")
            if (not apply_quality_filter) or (not is_uv_o) or onset_hflf > -5.5:
                le_o, v_o, f0_o = le_sl[:f_sp_rel], voi_sl[:f_sp_rel], f0_sl[:f_sp_rel]
                el_o, er_o, dbl_o, dbr_o, fl_o, fr_o = _edge_stats(le_o, f0_o, v_o)
                onset_tokens.append(
                    V13ZhToken(
                        onset=o,
                        final="",
                        tone=tone,
                        speaker=spk,
                        utterance_id=uid,
                        prev_final=prev_f,
                        next_onset=f,
                        log_env=le_o,
                        bap=ba_sl[:f_sp_rel],
                        bap_gt8k=bg_sl[:f_sp_rel],
                        voiced=v_o,
                        f0_rel_st=f0_o,
                        f_split=int(f_sp_rel),
                        dur_ms=float(f_sp_rel * HOP * 1000.0 / sr),
                        edge_left_env=el_o,
                        edge_right_env=er_o,
                        edge_left_db=dbl_o,
                        edge_right_db=dbr_o,
                        edge_f0_l=fl_o,
                        edge_f0_r=fr_o,
                        uv_hf_lf=onset_hflf,
                        onset_wav=on_wav,
                        syl_idx=idx_s,
                        n_syls_in_utt=n_syls,
                        mean_f0_st=mean_f0_st,
                        parseval_p75_db=float(np.percentile(parseval_frame_db(le_o, n_fft=N_FFT), 75)),
                        syllable_pos=syl_pos,
                    )
                )

        if f_hi_r - f_sp >= 6:
            le_f, v_f, f0_f = le_sl[f_sp_rel:], voi_sl[f_sp_rel:], f0_sl[f_sp_rel:]
            if (not apply_quality_filter) or (
                int(np.sum(v_f)) >= 6
                and float(np.mean(v_f)) >= 0.30
                and float(np.percentile(parseval_frame_db(le_f, n_fft=N_FFT), 80)) >= -38.0
            ):
                el_f, er_f, dbl_f, dbr_f, fl_f, fr_f = _edge_stats(le_f, f0_f, v_f)
                v_f0_r = f0_f[v_f] if np.any(v_f) else f0_f
                final_tokens.append(
                    V13ZhToken(
                        onset="",
                        final=f,
                        tone=tone,
                        speaker=spk,
                        utterance_id=uid,
                        prev_final=o or prev_f,
                        next_onset=next_o,
                        log_env=le_f,
                        bap=ba_sl[f_sp_rel:],
                        bap_gt8k=bg_sl[f_sp_rel:],
                        voiced=v_f,
                        f0_rel_st=f0_f,
                        f_split=0,
                        dur_ms=float(len(le_f) * HOP * 1000.0 / sr),
                        edge_left_env=el_f,
                        edge_right_env=er_f,
                        edge_left_db=dbl_f,
                        edge_right_db=dbr_f,
                        edge_f0_l=fl_f,
                        edge_f0_r=fr_f,
                        uv_hf_lf=0.0,
                        onset_wav=None,
                        syl_idx=idx_s,
                        n_syls_in_utt=n_syls,
                        mean_f0_st=float(np.median(v_f0_r)) if v_f0_r.size else 0.0,
                        parseval_p75_db=float(np.percentile(parseval_frame_db(le_f, n_fft=N_FFT), 75)),
                        syllable_pos=syl_pos,
                    )
                )

    return syl_tokens, onset_tokens, final_tokens, dropped


def build_v13_zh_bank(
    manifest_path: Path | str = V12_MANIFEST,
    *,
    gmm_cache_path: Path | str | None = GMM_ALIGNMENTS_V12,
    exclude_ids: Iterable[str] = (),
    sr: int = SR,
    use_cache: bool = True,
    fix_f_split: bool = True,
    max_glottal_utts_per_spk: int = 8,
) -> V13ZhBank:
    """Build fast Mandarin-only V13ZhBank excluding `exclude_ids` from all pools, profiles, and glottal banks."""
    zh_utterances = load_zh_manifest_utterances(manifest_path)
    gmm_zh_map = load_gmm_zh_map(gmm_cache_path)
    feats_cache = _analyze_or_load_zh_features(zh_utterances, gmm_zh_map, sr=sr, use_cache=use_cache)

    excl = set(exclude_ids)
    zh_train = [u for u in zh_utterances if u["id"] not in excl]

    zh_for_ub: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures]] = {}
    zh_by_spk: dict[str, list[tuple[np.ndarray, vocoder.VocoderFeatures]]] = {}
    zh_v12_by_spk: dict[str, list[VocoderFeaturesV12]] = {}

    zh_syllables: dict[tuple[str, str], list[V13ZhToken]] = {}
    zh_onsets: dict[str, list[V13ZhToken]] = {}
    zh_finals: dict[str, list[V13ZhToken]] = {}
    zh_syl_kept, zh_syl_dropped = 0, 0

    for u in zh_train:
        uid, spk = u["id"], u["speaker"]
        rec = feats_cache[uid]
        xa = np.asarray(rec["xa"], dtype=np.float64)
        feats = rec["feats"]
        b_gt8 = np.asarray(rec["bap_gt8k"], dtype=np.float64)
        zh_for_ub[uid] = (xa, feats)
        zh_by_spk.setdefault(spk, []).append((xa, feats))
        if len(zh_v12_by_spk.get(spk, ())) < max_glottal_utts_per_spk:
            fv12 = VocoderFeaturesV12(
                feats=feats,
                bap_gt8k=b_gt8,
                uv_raw_log_env=None,
                lp_pulses=rec["lp_pulses"],
                coherent_phasor=rec["coherent_phasor"],
            )
            zh_v12_by_spk.setdefault(spk, []).append(fv12)

        syls, ons, fins, drp = extract_utterance_zh_tokens(
            u,
            xa,
            feats,
            b_gt8,
            gmm_zh_map.get(uid),
            act_start=rec["act_start"],
            sr=sr,
            fix_f_split=fix_f_split,
        )
        zh_syl_dropped += drp
        zh_syl_kept += len(syls)
        for t in syls:
            zh_syllables.setdefault((t.onset, t.final), []).append(t)
        for t in ons:
            zh_onsets.setdefault(t.onset, []).append(t)
        for t in fins:
            zh_finals.setdefault(t.final, []).append(t)

    fallback_ub = ub.build_unit_bank(zh_train, exclude_ids=excl, precomputed=zh_for_ub, sr=sr) if zh_train else None

    profs_v2: dict[str, Any] = {}
    profs_v5: dict[str, Any] = {}
    gbanks: dict[str, GlottalPulseBank] = {}
    for spk, items in sorted(zh_by_spk.items()):
        profs_v2[spk] = tc2.extract_v2_profile(items[:12], sr=sr, name=spk)
        profs_v5[spk] = extract_timbre_v5(items[:12], sr=sr, name=spk, corpus="aishell3")
    for spk, items in sorted(zh_v12_by_spk.items()):
        gbanks[spk] = build_glottal_pulse_bank(spk, items, max_pulses=32)

    return V13ZhBank(
        zh_syllables=zh_syllables,
        zh_onsets=zh_onsets,
        zh_finals=zh_finals,
        profiles_v2=profs_v2,
        profiles_v5=profs_v5,
        fallback_unit_bank=fallback_ub,
        glottal_banks=gbanks,
        zh_included_ids=[u["id"] for u in zh_train],
        zh_excluded_ids=sorted(excl),
        stats={
            "zh_syl_keys": len(zh_syllables),
            "zh_syl_tokens": zh_syl_kept,
            "zh_syl_dropped": zh_syl_dropped,
            "zh_onsets": len(zh_onsets),
            "zh_onset_tokens": sum(len(v) for v in zh_onsets.values()),
            "zh_finals": len(zh_finals),
            "zh_final_tokens": sum(len(v) for v in zh_finals.values()),
            "glottal_banks": sorted(gbanks.keys()),
            "fix_f_split": fix_f_split,
        },
    )


def filter_v13_bank_excluding(
    bank: V13ZhBank,
    exclude_ids: Iterable[str],
    *,
    ub_fold: ub.UnitBank | None = None,
    spk_override: tuple[str, Any] | None = None,
    glottal_override: tuple[str, GlottalPulseBank] | None = None,
) -> V13ZhBank:
    """Zero-leakage fold filtering removing all tokens belonging to `exclude_ids`."""
    excl = set(exclude_ids)
    profs5 = dict(bank.profiles_v5)
    if spk_override is not None:
        profs5[spk_override[0]] = spk_override[1]
    gbanks = dict(bank.glottal_banks)
    if glottal_override is not None:
        gbanks[glottal_override[0]] = glottal_override[1]

    return V13ZhBank(
        zh_syllables={
            k: v
            for k, v in ((k, [t for t in toks if t.utterance_id not in excl]) for k, toks in bank.zh_syllables.items())
            if v
        },
        zh_onsets={
            k: v
            for k, v in ((k, [t for t in toks if t.utterance_id not in excl]) for k, toks in bank.zh_onsets.items())
            if v
        },
        zh_finals={
            k: v
            for k, v in ((k, [t for t in toks if t.utterance_id not in excl]) for k, toks in bank.zh_finals.items())
            if v
        },
        profiles_v2=bank.profiles_v2,
        profiles_v5=profs5,
        fallback_unit_bank=ub_fold if ub_fold is not None else bank.fallback_unit_bank,
        glottal_banks=gbanks,
        zh_included_ids=[i for i in bank.zh_included_ids if i not in excl],
        zh_excluded_ids=sorted(set(bank.zh_excluded_ids) | excl),
        stats=dict(bank.stats),
    )
