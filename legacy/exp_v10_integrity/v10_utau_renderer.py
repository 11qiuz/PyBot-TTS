"""Phase 6 Integrity-Gated, Manner-Gain-Calibrated UTAU+HNM Renderer (100% Non-Neural)."""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import numpy as np

from exp_v5_arch import crossling_frontend as cf
from exp_v6_crossling_dsp import arbiter as v6_arb
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS
from exp_v6_vocoder import vocoder
from exp_v7_pybot_tts import crossling_warp as cw
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
from exp_v9_clarity_pro.hnm_v9 import synthesize_v9
from exp_v9_clarity_pro.viterbi_utau import (
    V9DualBank,
    V9EnToken,
    V9ZhToken,
    _edge_cmvn,
    _edge_f0,
    _get_src_profile,
    _viterbi_select,
    build_v9_dual_bank,
    filter_v9_bank_excluding_speakers,
)
from exp_v10_integrity import HOP, N_FFT, SR
from exp_v10_integrity.gain_calibration import (
    calibrate_token_parseval_gain,
    classify_ipa_manner,
    smooth_utterance_gain_contour,
)
from exp_v10_integrity.integrity_gate import (
    PERCEPTUAL_MIN_FRAMES,
    apply_integrity_gate_to_bank,
    compute_perceptual_phone_frames,
    min_zh_onset_frames,
)
from exp_v10_integrity.phone_edits import FRAME_MS, PhoneSpan, apply_phone_edits, parse_phone_edits
from exp_v10_integrity.zh_onsets import onset_template


ClonerFn = Callable[..., Any]
REF_VOWEL_PARSEVAL_DB = -17.5


def build_v10_bank(
    zh_utterances: list[dict] | None = None,
    zh_exclude_ids: Iterable[str] = (),
    en_utterances: list[dict] | None = None,
    en_exclude_ids: Iterable[str] = (),
    zh_precomputed: dict | None = None,
    en_precomputed: dict | None = None,
    sr: int = SR,
    refine_boundaries: bool = True,
    include_fallback_bank: bool = True,
    extract_v3_profiles: bool = False,
    ref_vowel_db: float | None = None,
) -> tuple[V9DualBank, dict]:
    """Build v9 Dual-Donor Bank and pass it through the strict v10 Acoustic Integrity Gate."""
    raw_bank = build_v9_dual_bank(
        zh_utterances=zh_utterances,
        zh_exclude_ids=zh_exclude_ids,
        en_utterances=en_utterances,
        en_exclude_ids=en_exclude_ids,
        zh_precomputed=zh_precomputed,
        en_precomputed=en_precomputed,
        sr=sr,
        refine_boundaries=refine_boundaries,
        include_fallback_bank=include_fallback_bank,
        extract_v3_profiles=extract_v3_profiles,
    )
    return apply_integrity_gate_to_bank(raw_bank, ref_vowel_db=ref_vowel_db)


def filter_v10_bank_excluding_speakers(bank: V9DualBank, excluded_speakers: Iterable[str]) -> V9DualBank:
    return filter_v9_bank_excluding_speakers(bank, excluded_speakers)


def build_v10_full_bank(
    zh_utterances: list[dict] | None = None,
    zh_exclude_ids: Iterable[str] = (),
    en_exclude_ids: Iterable[str] = (),
    zh_precomputed: dict | None = None,
    sr: int = SR,
    extract_v4_profiles: bool = True,
) -> tuple[V9DualBank, dict[str, Any], dict]:
    """Build combined Phase-6 bank using cmudict+landmark_aligner EN tokens, ZH v10 gate, and TimbreProfileV4."""
    import json
    from exp_v8_clarity.en_utau_bank import _trim_active_span
    from exp_v9_clarity_pro.hnm_v9 import analyze_v9
    from exp_v10_g2p_and_align.landmark_aligner import extract_verified_en_bank
    from exp_v10_g2p_and_align.timbre_cloner_v4 import extract_timbre_v4
    from exp_v10_integrity import EN_MANIFEST, EXTENDED_INDEX

    if zh_utterances is None:
        rows = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
        # The bundled index recorded absolute paths from its original machine.
        zh_utterances = [dict(u, path=str(EXTENDED_INDEX.parent / f"{u['id']}.wav")) for u in rows]
    zh_base = build_v9_dual_bank(
        zh_utterances=zh_utterances,
        zh_exclude_ids=zh_exclude_ids,
        en_utterances=[],
        en_exclude_ids=(),
        zh_precomputed=zh_precomputed,
        sr=sr,
        refine_boundaries=True,
        include_fallback_bank=True,
        extract_v3_profiles=False,
    )
    en_excl = set(en_exclude_ids)
    verified_en = extract_verified_en_bank(exclude_ids=en_excl)
    en_phones: dict[str, list[V9EnToken]] = {}
    en_incl_ids: set[str] = set()
    for ipa, toks in verified_en.items():
        conv = []
        for t in toks:
            en_incl_ids.add(t.utterance_id)
            le = t.log_env
            bg = np.repeat(t.bap[:, 4:5], 2, axis=1)
            conv.append(
                V9EnToken(
                    ipa=t.ipa,
                    speaker=t.speaker,
                    utterance_id=t.utterance_id,
                    prev_ipa=t.prev_ipa,
                    next_ipa=t.next_ipa,
                    stress=None,
                    log_env=le,
                    bap=t.bap,
                    bap_gt8k=bg.astype(np.float32),
                    voiced=t.voiced,
                    f0_rel_st=t.f0_rel_st,
                    dur_ms=t.dur_ms,
                    cmvn_left=_edge_cmvn(le, True),
                    cmvn_right=_edge_cmvn(le, False),
                    f0_edge_l=_edge_f0(t.f0_rel_st, t.voiced, True),
                    f0_edge_r=_edge_f0(t.f0_rel_st, t.voiced, False),
                    uv_hf_lf=t.uv_hf_lf_db,
                    raw_wav=t.raw_wav if not np.any(t.voiced) else None,
                )
            )
        if conv:
            en_phones[ipa] = conv

    zh_base.en_phones = en_phones
    zh_base.en_included_ids = sorted(en_incl_ids)
    zh_base.en_excluded_ids = sorted(en_excl)
    purified, gate_stats = apply_integrity_gate_to_bank(zh_base)

    profs_v4: dict[str, Any] = {}
    if extract_v4_profiles:
        zh_excl = set(zh_exclude_ids)
        by_zh: dict[str, list] = {}
        for u in zh_utterances:
            if u["id"] in zh_excl:
                continue
            if zh_precomputed is not None and u["id"] in zh_precomputed:
                xa, f9, _ = zh_precomputed[u["id"]]
            else:
                x, _ = vocoder.read_wav_mono(u["path"])
                xa = x[u["active_start"] : u["active_stop"]]
                f9, _ = analyze_v9(xa, sr=sr)
            by_zh.setdefault(u["speaker"], []).append((xa, f9))
        for spk, items in sorted(by_zh.items()):
            p4 = extract_timbre_v4(items[:12], sr=sr, name=spk, corpus="aishell3")
            profs_v4[spk] = p4
            purified.zh_profiles_v2[spk] = p4.v3.to_v2_stub()
            purified.zh_profiles_v3[spk] = p4.v3

        en_man = json.loads(EN_MANIFEST.read_text(encoding="utf-8"))["english"]["utterances"]
        by_en: dict[str, list] = {}
        for u in en_man:
            if u["id"] in en_excl:
                continue
            if len(by_en.get(u["speaker"], [])) >= 8:
                continue
            p_w = EN_MANIFEST.parent.parent.parent / u["path"]
            x, _ = vocoder.read_wav_mono(p_w)
            s0, s1 = _trim_active_span(x, sr=sr)
            xa = x[s0:s1]
            f9, _ = analyze_v9(xa, sr=sr)
            by_en.setdefault(u["speaker"], []).append((xa, f9))
        for spk, items in sorted(by_en.items()):
            p4 = extract_timbre_v4(items, sr=sr, name=spk, corpus="cmu_arctic")
            profs_v4[spk] = p4
            purified.en_profiles_v2[spk] = p4.v3.to_v2_stub()
            purified.en_profiles_v3[spk] = p4.v3

    return purified, profs_v4, gate_stats


def _resolve_f0_iqr(prof: Any) -> tuple[float, float]:
    base = getattr(prof, "v3", prof)
    return float(getattr(base, "f0_median_hz", 180.0)), float(getattr(base, "f0_iqr_st", 3.2))


def _adapt_token_with_cloner(
    log_env: np.ndarray,
    bap: np.ndarray,
    bap_gt8k: np.ndarray,
    voiced: np.ndarray,
    f0_rel_st: np.ndarray,
    src_prof: Any,
    tgt_prof: Any,
    cloner_fn: ClonerFn | None,
    vtln_alpha: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tgt_name = getattr(tgt_prof, "name", "")
    if src_prof is None or getattr(src_prof, "name", "") == tgt_name:
        return log_env.astype(np.float64), bap.astype(np.float64), bap_gt8k.astype(np.float64), voiced.copy()

    src_f0, _ = _resolve_f0_iqr(src_prof)
    dummy_f0 = np.where(voiced, src_f0 * (2.0 ** (f0_rel_st / 12.0)), 0.0)
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
    ratio_b4 = np.clip(ad.bap[:, 4:5] / np.maximum(bap[:, 4:5], 1e-3), 0.25, 2.5)
    bg_ad = np.clip(bap_gt8k.astype(np.float64) * ratio_b4, 0.005, 0.995)
    return ad.log_env, ad.bap, bg_ad, ad.voiced


def render_v10_utterance(
    text_or_plan: str | dict,
    bank: V9DualBank,
    target_profile: Any,
    *,
    cloner_fn: ClonerFn | None = None,
    plan_fn: Callable[[str], dict] | None = None,
    src_profiles: dict[str, Any] | None = None,
    vtln_alpha: float | None = None,
    use_viterbi: bool = True,
    use_unvoiced_residual: bool = True,
    prosody_mode: str = "natural",
    ref_vowel_db: float = REF_VOWEL_PARSEVAL_DB,
    seed: int = 72,
    phase_dispersion: bool = True,
    phone_edits: dict | None = None,
    include_timeline: bool = False,
    onset_repair_fn: Callable[[np.ndarray, list[Any], np.ndarray], tuple[np.ndarray, dict]] | None = None,
) -> tuple[np.ndarray, dict]:
    """Render speech and optionally edit or report individual acoustic segments."""
    edits = parse_phone_edits(phone_edits) if phone_edits is not None else {}
    collect_spans = include_timeline or bool(edits)
    if isinstance(text_or_plan, str):
        plan_doc = plan_fn(text_or_plan) if plan_fn is not None else cf.plan(text_or_plan, allow_oov_fallback=True)
    else:
        plan_doc = text_or_plan

    units = plan_doc.get("units") or []
    if not units:
        raise ValueError("plan_doc has no units")

    tgt_name = getattr(target_profile, "name", "")
    tgt_f0, tgt_iqr = _resolve_f0_iqr(target_profile)
    iqr_scale = max(0.55, min(1.75, tgt_iqr / 3.2))
    rules = bank.prosody_rules

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

    # Breath-group boundaries for declination & phrase-final lengthening
    group_id = [0] * n_u
    cur_g = 0
    for i, u in enumerate(units):
        group_id[i] = cur_g
        if float(u.get("pause_after_ms", 0.0)) > 55.0:
            cur_g += 1
    group_spans: dict[int, list[int]] = {}
    for i, g in enumerate(group_id):
        if g not in group_spans:
            group_spans[g] = [i, i]
        else:
            group_spans[g][1] = i

    unit_dur_ms, unit_decl_st, unit_gain_db = [], [], []
    for i, u in enumerate(units):
        base_d = max(60.0 if parsed[i][0] == "zh" else 45.0, float(u.get("duration_ms", 200.0)))
        g0, g1 = group_spans[group_id[i]]
        g_len = max(1, g1 - g0)
        rel_pos = (i - g0) / max(1, g_len)
        if prosody_mode == "natural":
            if i == g1:
                base_d *= rules.zh_final_lengthen if parsed[i][0] == "zh" else rules.en_final_lengthen
            elif g_len >= 4 and g0 < i < g1:
                base_d *= 0.96
            decl = 0.5 * rules.declination_st - rules.declination_st * rel_pos
            # Gentle phrase-end taper (-1.5 dB instead of -2.2 dB so final syllable never fades too quietly)
            gdb = 0.3 - 1.5 * (rel_pos ** 2) if i == g1 else 0.25 * (1.0 - rel_pos)
        else:
            decl = 0.5 - 1.0 * (i / max(1, n_u - 1))
            gdb = 0.0
        unit_dur_ms.append(base_d)
        unit_decl_st.append(decl)
        unit_gain_db.append(gdb)

    # 1. Select Mandarin syllables via obstruent-safe Viterbi
    selected_zh: dict[int, tuple[str, Any]] = {}
    i = 0
    while i < n_u:
        if parsed[i][0] != "zh":
            i += 1
            continue
        j = i
        while j < n_u and parsed[j][0] == "zh":
            j += 1
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
                    if is_uv_zh and getattr(c, "uv_hf_lf", 0.0) < -4.0:
                        tc += 4.0
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

    # 2. Select English phonemes via obstruent-safe Viterbi
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
                    if st is not None and getattr(c, "stress", None) is not None and st != c.stress:
                        tc += 0.7
                    if is_uv_en:
                        tc -= 0.15 * max(-10.0, min(10.0, getattr(c, "uv_hf_lf", 0.0)))
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

    # 3. Assemble 5 ms feature track with Perceptual Minimum Floors & Manner-Calibrated Loudness
    envs, baps, bgt8s, vois, sts = [], [], [], [], []
    is_obs_blk, is_pause_blk = [], []
    residual_spans: list[tuple[int, int, np.ndarray]] = []
    protected_closures: list[tuple[int, int]] = []
    phone_spans: list[PhoneSpan] = []
    source_counts: dict[str, int] = {}
    gain_corrs_db: list[float] = []
    min_phone_frames = 999

    lead_fr = 4
    envs.append(np.full((lead_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((lead_fr, 5), dtype=np.float64))
    bgt8s.append(np.ones((lead_fr, 2), dtype=np.float64))
    vois.append(np.zeros(lead_fr, dtype=bool))
    sts.append(np.zeros(lead_fr, dtype=np.float64))
    is_obs_blk.append(False)
    is_pause_blk.append(True)
    cur_fr = lead_fr

    zh_fb_prof = _get_src_profile(bank, "SSB0502", target_profile, src_profiles) or (
        next(iter(bank.zh_profiles_v2.values())) if bank.zh_profiles_v2 else target_profile
    )

    for u_idx, u in enumerate(units):
        lang, o, f, tone = parsed[u_idx]
        dur_ms = unit_dur_ms[u_idx]
        nf = max(PERCEPTUAL_MIN_FRAMES["zh_syllable"] if lang == "zh" else 4, int(round(dur_ms / 5.0)))
        gain = float(u.get("gain", 0.85))
        pitch_5 = [float(x) for x in u.get("pitch_st", [0.0] * 5)]
        pause_ms = float(u.get("pause_after_ms", 0.0))
        decl_u = unit_decl_st[u_idx]
        gdb_u = unit_gain_db[u_idx]

        if lang == "zh":
            kind_sel, tok_sel = selected_zh[u_idx]
            prev_f = parsed[u_idx - 1][2] if (u_idx > 0 and parsed[u_idx - 1][0] == "zh") else "SIL"
            next_o = parsed[u_idx + 1][1] if (u_idx + 1 < n_u and parsed[u_idx + 1][0] == "zh") else "SIL"
            min_rhyme_frames = PERCEPTUAL_MIN_FRAMES["vowel_unstressed"]
            if kind_sel == "syl" and tok_sel is not None:
                best: V9ZhToken = tok_sel
                e_raw, b_raw, bg_raw, v_raw = _adapt_token_with_cloner(
                    best.log_env, best.bap, best.bap_gt8k, best.voiced, best.f0_rel_st,
                    _get_src_profile(bank, best.speaker, target_profile, src_profiles), target_profile, cloner_fn, vtln_alpha,
                )
                e_raw, g_corr = calibrate_token_parseval_gain(e_raw, "zh_syllable", ref_vowel_db, max_corr_db=10.5)
                gain_corrs_db.append(abs(g_corr))
                mode = "zh_target_syl" if best.speaker == tgt_name else "zh_donor_syl"
                source_counts[mode] = source_counts.get(mode, 0) + 1
                nf = max(nf, best.f_split + min_rhyme_frames)
                e_st, b_st, v_st, split_frame = _stretch_preserving_onset(
                    e_raw, b_raw, v_raw, best.f_split, nf, preserve_onset=True
                )
                bg_st = np.clip(_resample_2d(bg_raw, nf), 0.005, 0.995)
                f0_tok = _resample_1d(best.f0_rel_st.astype(np.float64), nf)
                if use_unvoiced_residual and best.onset_wav is not None and split_frame > 0:
                    residual_spans.append((cur_fr, cur_fr + split_frame, best.onset_wav))
            else:
                parts_e, parts_b, parts_bg, parts_v, parts_f0 = [], [], [], [], []
                split_pt = 0
                bo_sel = None
                if o:
                    onset_ipas = cf.PINYIN_ONSET_TO_IPA.get(o)
                    if not onset_ipas:
                        raise ValueError(f"Unsupported Mandarin onset {o!r} in unit {u_idx + 1}")
                    oc = bank.zh_onsets.get(o, [])
                    if oc:
                        bo = min(
                            oc,
                            key=lambda c: (0 if c.speaker == tgt_name else 4.5)
                            + (0 if c.prev_final == prev_f else 1.4)
                            + (0 if c.next_onset == f else 1.0),
                        )
                        bo_sel = bo
                        eo, bo_b, bgo, vo = _adapt_token_with_cloner(
                            bo.log_env, bo.bap, bo.bap_gt8k, bo.voiced, bo.f0_rel_st,
                            _get_src_profile(bank, bo.speaker, target_profile, src_profiles), target_profile, cloner_fn, vtln_alpha,
                        )
                        onset_f0 = bo.f0_rel_st
                        onset_source = "zh_recorded_onset"
                    else:
                        env, bp, voiced, closure = onset_template(o)
                        onset_f0 = np.zeros(len(env))
                        eo, bo_b, bgo, vo = _adapt_token_with_cloner(
                            env, bp, np.repeat(bp[:, 4:5], 2, axis=1), voiced, onset_f0,
                            zh_fb_prof, target_profile, cloner_fn, vtln_alpha,
                        )
                        onset_source = "zh_synth_onset"
                        if closure:
                            protected_closures.append((cur_fr, cur_fr + closure))
                    source_counts[onset_source] = source_counts.get(onset_source, 0) + 1
                    eo, gc_o = calibrate_token_parseval_gain(eo, classify_ipa_manner(onset_ipas[0]), ref_vowel_db)
                    gain_corrs_db.append(abs(gc_o))
                    parts_e.append(eo); parts_b.append(bo_b); parts_bg.append(bgo); parts_v.append(vo); parts_f0.append(onset_f0)
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
                    ef, gc_f = calibrate_token_parseval_gain(ef, "zh_syllable", ref_vowel_db, max_corr_db=10.5)
                    gain_corrs_db.append(abs(gc_f))
                    parts_e.append(ef); parts_b.append(bf_b); parts_bg.append(bgf); parts_v.append(vf); parts_f0.append(bf.f0_rel_st)
                    source_counts["zh_concat_of"] = source_counts.get("zh_concat_of", 0) + 1
                elif bank.fallback_unit_bank is not None and f in cf.PINYIN_FINAL_TO_IPA:
                    for fi in cf.PINYIN_FINAL_TO_IPA[f]:
                        d = cw.lookup_ipa_template(bank.fallback_unit_bank, fi, is_onset=False)
                        bg_d = np.repeat(d["bap"][:, 4:5], 2, axis=1)
                        ed, bd, bgd, vd = _adapt_token_with_cloner(
                            d["log_env"] - 3.1, d["bap"], bg_d, d["voiced"], np.zeros(len(d["log_env"])),
                            zh_fb_prof, target_profile, cloner_fn, vtln_alpha,
                        )
                        ed, gc_f = calibrate_token_parseval_gain(ed, "zh_syllable", ref_vowel_db)
                        gain_corrs_db.append(abs(gc_f))
                        parts_e.append(ed); parts_b.append(bd); parts_bg.append(bgd); parts_v.append(vd); parts_f0.append(np.zeros(len(ed)))
                    source_counts["zh_ipa_fill"] = source_counts.get("zh_ipa_fill", 0) + 1
                else:
                    raise ValueError(f"Missing Mandarin rhyme {f!r} in unit {u_idx + 1}")
                nf = max(nf, split_pt + min_rhyme_frames)
                e_st, b_st, v_st, split_frame = _stretch_preserving_onset(
                    np.concatenate(parts_e, axis=0),
                    np.concatenate(parts_b, axis=0),
                    np.concatenate(parts_v, axis=0),
                    split_pt, nf, preserve_onset=True,
                )
                if use_unvoiced_residual and bo_sel is not None and bo_sel.onset_wav is not None and split_frame > 0:
                    residual_spans.append((cur_fr, cur_fr + split_frame, bo_sel.onset_wav))
                bg_st = np.clip(_resample_2d(np.concatenate(parts_bg, axis=0), nf), 0.005, 0.995)
                f0_tok = _resample_1d(np.concatenate(parts_f0), nf)

            # Apply bounded prosodic gain (clamped +-2.5 dB so no jarring inter-unit volume swing)
            raw_g_shift = gdb_u + 20.0 * math.log10(max(0.55, min(1.35, gain / 0.85)))
            e_st = e_st + float(np.clip(raw_g_shift, -2.5, 2.5)) * math.log(10.0) / 20.0

            t_grid = np.linspace(0.0, 1.0, len(e_st))
            p5_arr = np.array([_interp_list(pitch_5, float(t)) for t in t_grid], dtype=np.float64) * iqr_scale
            if prosody_mode == "natural":
                f0_micro = np.clip(f0_tok - float(np.median(f0_tok)), -3.0, 3.0)
                st_u = (1.0 - rules.donor_f0_blend) * p5_arr + rules.donor_f0_blend * f0_micro + decl_u
            else:
                st_u = p5_arr + decl_u

            envs.append(e_st); baps.append(b_st); bgt8s.append(bg_st); vois.append(v_st); sts.append(st_u)
            is_obs_blk.append(o in OBSTRUENT_ONSETS)
            is_pause_blk.append(False)
            if collect_spans:
                if split_frame > 0:
                    onset_min = min_zh_onset_frames(o)
                    phone_spans.append(PhoneSpan(
                        u_idx, "onset", o, cur_fr, cur_fr + split_frame, min(split_frame, onset_min),
                    ))
                rhyme_min = min(len(e_st) - split_frame, min_rhyme_frames)
                phone_spans.append(PhoneSpan(
                    u_idx, "rhyme", f, cur_fr + split_frame, cur_fr + len(e_st), rhyme_min,
                ))
            cur_fr += len(e_st)
        else:
            ipas = list(u.get("ipa_phones") or ["ə"])
            arpa = u.get("arpabet") or []
            pairs = v6_arb._split_en_arpabet_to_ipa_with_stress(arpa) if arpa else [(p, None) for p in ipas]
            if not pairs:
                pairs = [(p, None) for p in ipas]
            n_p = len(pairs)
            chosen_p, raw_durs = [], []
            for idx_p, (ip, st) in enumerate(pairs):
                tok_e = selected_en_map.get((u_idx, idx_p))
                man_k = classify_ipa_manner(ip, st)
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
                        zh_fb_prof, target_profile, cloner_fn, vtln_alpha,
                    )
                    source_counts["en_synth_fill"] = source_counts.get("en_synth_fill", 0) + 1
                    d_ms = d["dur_ms"]
                    w_raw = None
                    f0_tk = np.zeros(len(e_r), dtype=np.float64)
                else:
                    raise ValueError(f"Missing EN phone {ip} and fallback_unit_bank=None")

                e_r, gc_p = calibrate_token_parseval_gain(e_r, man_k, ref_vowel_db)
                gain_corrs_db.append(abs(gc_p))
                if st == 1:
                    d_ms *= 1.16
                elif st == 0:
                    d_ms = max(35.0, d_ms * 0.82)
                chosen_p.append((ip, e_r, b_r, bg_r, v_r, w_raw, f0_tk))
                raw_durs.append(max(25.0, d_ms))

            # Perceptual duration floor allocation (never crushes phones below 25-45 ms!)
            frame_alloc = compute_perceptual_phone_frames(pairs, raw_durs, nf)
            raw_g_shift = gdb_u + 20.0 * math.log10(max(0.55, min(1.35, gain / 0.85)))
            clamped_g_np = float(np.clip(raw_g_shift, -2.5, 2.5)) * math.log(10.0) / 20.0

            for idx_p, ((ip, e_r, b_r, bg_r, v_r, w_raw, f0_tk), kp) in enumerate(zip(chosen_p, frame_alloc)):
                min_phone_frames = min(min_phone_frames, kp)
                ep = _resample_2d(e_r, kp) + clamped_g_np
                bp = np.clip(_resample_2d(b_r, kp), 0.005, 0.995)
                bgp = np.clip(_resample_2d(bg_r, kp), 0.005, 0.995)
                vp = _resample_1d(v_r.astype(np.float64), kp) >= 0.45

                sub_p = [_interp_list(pitch_5, (idx_p + m / 4.0) / n_p) for m in range(5)]
                tp = np.linspace(0.0, 1.0, kp)
                p_arr = np.array([_interp_list(sub_p, float(t)) for t in tp], dtype=np.float64) * iqr_scale
                if prosody_mode == "natural":
                    f0_tr = _resample_1d(f0_tk, kp)
                    f0_mic = np.clip(f0_tr - float(np.median(f0_tr)), -2.5, 2.5)
                    st_p = 0.75 * p_arr + 0.25 * f0_mic + decl_u
                else:
                    st_p = p_arr + decl_u

                if use_unvoiced_residual and w_raw is not None and not np.any(vp):
                    residual_spans.append((cur_fr, cur_fr + kp, w_raw))
                envs.append(ep); baps.append(bp); bgt8s.append(bgp); vois.append(vp); sts.append(st_p)
                is_obs_blk.append(ip in OBSTRUENT_IPAS)
                is_pause_blk.append(False)
                if collect_spans:
                    floor = PERCEPTUAL_MIN_FRAMES[classify_ipa_manner(*pairs[idx_p])]
                    phone_spans.append(PhoneSpan(u_idx, idx_p + 1, ip, cur_fr, cur_fr + kp, floor))
                cur_fr += kp

        if pause_ms > 15.0:
            p_fr = max(2, int(round(pause_ms / 5.0)))
            envs.append(np.full((p_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
            baps.append(np.ones((p_fr, 5), dtype=np.float64))
            bgt8s.append(np.ones((p_fr, 2), dtype=np.float64))
            vois.append(np.zeros(p_fr, dtype=bool))
            sts.append(np.zeros(p_fr, dtype=np.float64))
            is_obs_blk.append(False)
            is_pause_blk.append(True)
            cur_fr += p_fr

    tail_fr = 6
    envs.append(np.full((tail_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((tail_fr, 5), dtype=np.float64))
    bgt8s.append(np.ones((tail_fr, 2), dtype=np.float64))
    vois.append(np.zeros(tail_fr, dtype=bool))
    sts.append(np.zeros(tail_fr, dtype=np.float64))
    is_obs_blk.append(False)
    is_pause_blk.append(True)

    # Manner-aware seams: 0 ms on voiceless obstruents, 5 ms on voiced obstruents, 10 ms on vowel/sonorant joins
    for idx_b in range(len(envs) - 1):
        if is_pause_blk[idx_b] or is_pause_blk[idx_b + 1]:
            continue
        if np.any(vois[idx_b]) and np.any(vois[idx_b + 1]):
            bf = 1 if is_obs_blk[idx_b + 1] else 2
            _blend_seam(envs[idx_b], envs[idx_b + 1], baps[idx_b], baps[idx_b + 1], blend_frames=bf)

    # Build frame-level masks for utterance-wide Parseval gain smoothing
    obs_fr_flags = np.concatenate([np.full(len(e), flag, dtype=bool) for e, flag in zip(envs, is_obs_blk)])
    for start, stop in protected_closures:
        obs_fr_flags[start:stop] = False
    pause_fr_flags = np.concatenate([np.full(len(e), flag, dtype=bool) for e, flag in zip(envs, is_pause_blk)])
    env_all = np.concatenate(envs, axis=0)
    bap_all = np.clip(np.concatenate(baps, axis=0), 0.005, 0.995)
    bgt8_all = np.clip(np.concatenate(bgt8s, axis=0), 0.005, 0.995)
    voi_all = np.concatenate(vois, axis=0)
    st_all = np.concatenate(sts, axis=0)

    # Smooth sudden >2 dB/5ms voiced jumps & clamp unvoiced ceiling/floor!
    env_all = smooth_utterance_gain_contour(
        env_all, voi_all, obs_fr_flags, pause_fr_flags, ref_vowel_db=ref_vowel_db
    )

    extra_trace: dict[str, Any] = {}
    if onset_repair_fn is not None:
        # Optional hook (used by v10b): repair onset frames on the exact envelope that is
        # about to be synthesised, i.e. after seam blending and gain smoothing. Default None
        # keeps v10 bit-exact.
        env_all, repair_stats = onset_repair_fn(env_all, phone_spans, voi_all)
        extra_trace["onset_repair"] = repair_stats

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
    base_spans = phone_spans
    if edits:
        feats, bgt8_all, residual_spans, phone_spans = apply_phone_edits(
            feats, bgt8_all, residual_spans, phone_spans, edits,
        )
        min_phone_frames = min((s.end - s.start for s in phone_spans if isinstance(s.part, int)), default=999)
    y = synthesize_v9(
        feats,
        seed=seed,
        phase_dispersion=phase_dispersion,
        bap_gt8k=bgt8_all,
        unvoiced_residual_spans=residual_spans if use_unvoiced_residual else None,
    )
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    peak_scale = 0.89 / peak if peak > 0.89 else 1.0
    if peak_scale < 1.0:
        y *= peak_scale

    trace = {
        "source_counts": source_counts,
        "n_frames": len(feats.f0_hz),
        "n_residual_spans": len(residual_spans),
        "integrity": {
            "min_en_phone_ms": (min_phone_frames * 5.0) if min_phone_frames < 999 else None,
            "mean_abs_gain_corr_db": round(float(np.mean(gain_corrs_db)), 3) if gain_corrs_db else 0.0,
            "max_abs_gain_corr_db": round(float(np.max(gain_corrs_db)), 3) if gain_corrs_db else 0.0,
        },
        **extra_trace,
    }
    if collect_spans:
        trace["peak_limit_db"] = round(20.0 * math.log10(peak_scale), 4)
        trace["phones"] = []
        for base, span in zip(base_spans, phone_spans):
            u = units[span.unit_index]
            f0 = feats.f0_hz[span.start:span.end]
            voiced_f0 = f0[f0 > 0]
            trace["phones"].append({
                "id": span.id,
                "lang": u.get("lang", "zh"),
                "token": u.get("token", u.get("hanzi")),
                "label": span.label,
                "text_span": u.get("text_span"),
                "start_ms": span.start * FRAME_MS,
                "end_ms": span.end * FRAME_MS,
                "duration_ms": (span.end - span.start) * FRAME_MS,
                "base_duration_ms": (base.end - base.start) * FRAME_MS,
                "min_duration_ms": span.min_frames * FRAME_MS,
                "median_f0_hz": round(float(np.median(voiced_f0)), 3) if voiced_f0.size else None,
            })
    return y, trace
