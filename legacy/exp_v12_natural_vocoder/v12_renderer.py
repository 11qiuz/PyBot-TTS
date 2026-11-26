"""Phase 8 (v12) Integrated Renderer: Expanded Diphone/Syllable Bank + Learned Prosody + CS Bridge + v12 Vocoder."""
from __future__ import annotations

import math
from typing import Any, Callable

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
from exp_v10_integrity.gain_calibration import (
    calibrate_token_parseval_gain,
    classify_ipa_manner,
    smooth_utterance_gain_contour,
)
from exp_v10_integrity.integrity_gate import compute_perceptual_phone_frames
from exp_v11_diphone_tts.v11_renderer import (
    EN_FUNCTION_WORDS,
    REF_VOWEL_PARSEVAL_DB,
    _adapt_v11,
    _default_plan,
    _get_src_prof,
    _make_pseudo_diphone,
    _resolve_profile_attrs,
    _viterbi_diphones,
    _viterbi_zh_syllables,
)
from exp_v12_natural_vocoder import HOP, N_FFT, SR
from exp_v12_natural_vocoder.v12_bank import V12DualBank
from exp_v12_natural_vocoder.vocoder_v12 import GlottalPulseBank, synthesize_v12
from exp_v12_prosody.cs_bridge import bridge_cs_boundary
from exp_v12_prosody.prosody_model import ProsodyModelV12, apply_prosody_v12

ClonerFn = Callable[..., Any]


def _pick_glottal_bank(bank: V12DualBank, tgt_name: str, tgt_f0: float, override: GlottalPulseBank | None) -> GlottalPulseBank | None:
    if override is not None:
        return override
    if tgt_name in bank.glottal_banks:
        return bank.glottal_banks[tgt_name]
    if not bank.glottal_banks:
        return None
    # Nearest speaker by median F0
    best_gb = None
    best_dist = 1e9
    for spk, gb in bank.glottal_banks.items():
        d = abs(float(np.median(gb.f0_hz)) - tgt_f0) if gb.f0_hz.size else 100.0
        if d < best_dist:
            best_dist = d
            best_gb = gb
    return best_gb


def render_v12_utterance(
    text_or_plan: str | dict,
    bank: V12DualBank,
    target_profile: Any,
    *,
    prosody_model: ProsodyModelV12 | None = None,
    use_learned_prosody: bool = True,
    use_cs_bridge: bool = True,
    vocoder_mode: str = "hybrid_lp_hnm",
    crisp_unvoiced: bool = True,
    glottal_bank: GlottalPulseBank | None = None,
    cloner_fn: ClonerFn | None = None,
    plan_fn: Callable[[str], dict] | None = None,
    src_profiles: dict[str, Any] | None = None,
    vtln_alpha: float | None = None,
    use_unvoiced_residual: bool = True,
    ref_vowel_db: float = REF_VOWEL_PARSEVAL_DB,
    seed: int = 72,
    phase_dispersion: bool = True,
) -> tuple[np.ndarray, dict]:
    """Render ZH / EN / Code-Switched text using expanded v12 bank + Ridge prosody + CS bridge + hybrid LP-HNM."""
    if isinstance(text_or_plan, str):
        plan_doc = plan_fn(text_or_plan) if plan_fn is not None else _default_plan(text_or_plan)
    else:
        plan_doc = text_or_plan

    prosody_applied = False
    if use_learned_prosody:
        try:
            plan_doc = apply_prosody_v12(plan_doc, model=prosody_model)
            prosody_applied = True
        except Exception:
            prosody_applied = False

    units = plan_doc.get("units") or []
    if not units:
        raise ValueError("plan_doc has no units")

    tgt_name = getattr(target_profile, "name", "")
    tgt_f0, tgt_iqr = _resolve_profile_attrs(target_profile)
    iqr_scale = max(0.55, min(1.75, tgt_iqr / 3.2))

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

    # Breath-group segmentation + prosody (respecting learned Ridge duration/gain_db/pitch_st when present)
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

    unit_dur_ms, unit_decl_st, unit_gain_db, unit_is_func = [], [], [], []
    for i, u in enumerate(units):
        tok_raw = str(u.get("token") or u.get("text") or u.get("hanzi") or "").strip().lower().strip(".,!?;:\"'")
        is_func = (parsed[i][0] != "zh") and (tok_raw in EN_FUNCTION_WORDS)
        g0, g1 = group_spans[group_id[i]]
        g_len = max(1, g1 - g0)
        rel_pos = (i - g0) / max(1, g_len)

        if "gain_db" in u and "duration_ms" in u:
            base_d = max(55.0 if parsed[i][0] == "zh" else 45.0, float(u["duration_ms"]))
            gdb = float(u["gain_db"])
            decl = 0.25 - 0.50 * rel_pos  # phrase declination already anchored in u["pitch_st"]
        else:
            base_d = max(60.0 if parsed[i][0] == "zh" else 45.0, float(u.get("duration_ms", 200.0)))
            if parsed[i][0] != "zh":
                if is_func:
                    base_d *= 0.76
                    gdb = -1.35
                else:
                    base_d *= 1.12
                    gdb = 0.75
            else:
                gdb = 0.25 * (1.0 - rel_pos)
            if i == g1:
                base_d *= 1.16 if parsed[i][0] == "zh" else 1.18
                gdb -= 0.8
            elif g_len >= 4 and g0 < i < g1:
                base_d *= 0.96
            decl = 0.75 - 1.50 * rel_pos

        unit_dur_ms.append(base_d)
        unit_decl_st.append(decl)
        unit_gain_db.append(gdb)
        unit_is_func.append(is_func)

    envs, baps, bgt8s, vois, sts = [], [], [], [], []
    is_obs_blk, is_pause_blk = [], []
    residual_spans: list[tuple[int, int, np.ndarray]] = []
    source_counts: dict[str, int] = {}
    slot_timings_ms: list[list[float]] = []
    unit_boundaries_ms: list[float] = []
    boundary_frames: list[int] = []
    cs_switch_frames: list[int] = []

    lead_fr = 4
    envs.append(np.full((lead_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((lead_fr, 5), dtype=np.float64))
    bgt8s.append(np.ones((lead_fr, 2), dtype=np.float64))
    vois.append(np.zeros(lead_fr, dtype=bool))
    sts.append(np.zeros(lead_fr, dtype=np.float64))
    is_obs_blk.append(False)
    is_pause_blk.append(True)
    cur_fr = lead_fr

    zh_fb_prof = _get_src_prof(bank, "SSB0502", target_profile, src_profiles) or (
        next(iter(bank.profiles_v2.values())) if bank.profiles_v2 else target_profile
    )

    prev_span_lang: str | None = None
    u_idx = 0
    while u_idx < n_u:
        cur_lang = "zh" if parsed[u_idx][0] == "zh" else "en"
        if prev_span_lang is not None and prev_span_lang != cur_lang:
            cs_switch_frames.append(cur_fr)

        if cur_lang == "zh":
            j = u_idx
            while j < n_u and parsed[j][0] == "zh":
                j += 1
            slots = []
            slot_indices = []
            selected_zh = {}
            for idx in range(u_idx, j):
                _, o, f, tone = parsed[idx]
                prev_f = parsed[idx - 1][2] if (idx > 0 and parsed[idx - 1][0] == "zh") else "SIL"
                next_o = parsed[idx + 1][1] if (idx + 1 < n_u and parsed[idx + 1][0] == "zh") else "SIL"
                cands = bank.zh_syllables.get((o, f), [])
                p_bef = idx > 0 and float(units[idx - 1].get("pause_after_ms", 0.0)) > 15.0
                if cands:
                    is_uv_zh = o in ("f", "s", "sh", "x", "h", "c", "ch", "q")
                    tcosts = []
                    for c in cands:
                        tc = 0.0 if c.speaker == tgt_name else 3.5
                        if c.tone != tone:
                            tc += 2.2
                        if c.prev_final != prev_f:
                            tc += 1.3
                        if c.next_onset != next_o:
                            tc += 1.3
                        if is_uv_zh and c.uv_hf_lf < -4.0:
                            tc += 2.5
                        tc += 0.65 * abs(math.log(max(20.0, c.dur_ms) / max(20.0, unit_dur_ms[idx])))
                        tcosts.append(tc)
                    slots.append((cands, tcosts, p_bef, o in OBSTRUENT_ONSETS))
                    slot_indices.append(idx)
                else:
                    if slots:
                        for si, tk in zip(slot_indices, _viterbi_zh_syllables(slots)):
                            selected_zh[si] = tk
                        slots, slot_indices = [], []
                    selected_zh[idx] = None
            if slots:
                for si, tk in zip(slot_indices, _viterbi_zh_syllables(slots)):
                    selected_zh[si] = tk

            for idx in range(u_idx, j):
                _, o, f, tone = parsed[idx]
                u = units[idx]
                nf = max(12, int(round(unit_dur_ms[idx] / 5.0)))
                pitch_5 = [float(x) for x in u.get("pitch_st", [0.0] * 5)]
                pause_ms = float(u.get("pause_after_ms", 0.0))
                prev_f = parsed[idx - 1][2] if (idx > 0 and parsed[idx - 1][0] == "zh") else "SIL"
                next_o = parsed[idx + 1][1] if (idx + 1 < n_u and parsed[idx + 1][0] == "zh") else "SIL"

                best = selected_zh.get(idx)
                if best is not None:
                    e_raw, b_raw, bg_raw, v_raw = _adapt_v11(
                        best.log_env, best.bap, best.bap_gt8k, best.voiced, best.f0_rel_st,
                        _get_src_prof(bank, best.speaker, target_profile, src_profiles), target_profile, cloner_fn, vtln_alpha,
                    )
                    e_raw, _ = calibrate_token_parseval_gain(e_raw, "zh_syllable", ref_vowel_db, max_corr_db=10.5)
                    mode = "zh_target_syl" if best.speaker == tgt_name else "zh_donor_syl"
                    source_counts[mode] = source_counts.get(mode, 0) + 1
                    e_st, b_st, v_st, f_sp_new = _stretch_preserving_onset(e_raw, b_raw, v_raw, best.f_split, nf, preserve_onset=True)
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
                                key=lambda c: (0.0 if c.speaker == tgt_name else 3.5)
                                + (0.0 if c.prev_final == prev_f else 1.3)
                                + (0.0 if c.next_onset == f else 1.0)
                                + (0.0 if c.dur_ms >= 25.0 else 2.5)
                                + (2.5 if (is_uv_o and c.uv_hf_lf < -4.0) else 0.0),
                            )
                            bo_sel = bo
                            eo, bo_b, bgo, vo = _adapt_v11(
                                bo.log_env, bo.bap, bo.bap_gt8k, bo.voiced, bo.f0_rel_st,
                                _get_src_prof(bank, bo.speaker, target_profile, src_profiles), target_profile, cloner_fn, vtln_alpha,
                            )
                            eo, _ = calibrate_token_parseval_gain(eo, "voiceless_fricative" if is_uv_o else "nasal", ref_vowel_db, max_corr_db=8.5)
                            parts_e.append(eo); parts_b.append(bo_b); parts_bg.append(bgo); parts_v.append(vo); parts_f0.append(bo.f0_rel_st)
                            split_pt = len(eo)
                        elif bank.fallback_unit_bank is not None:
                            for oi in cf.PINYIN_ONSET_TO_IPA.get(o, ()):
                                d = cw.lookup_ipa_template(bank.fallback_unit_bank, oi, is_onset=True)
                                bg_d = np.repeat(d["bap"][:, 4:5], 2, axis=1)
                                ed, bd, bgd, vd = _adapt_v11(
                                    d["log_env"] - 3.1, d["bap"], bg_d, d["voiced"], np.zeros(len(d["log_env"])),
                                    zh_fb_prof, target_profile, cloner_fn, vtln_alpha,
                                )
                                ed, _ = calibrate_token_parseval_gain(ed, classify_ipa_manner(oi, None), ref_vowel_db, max_corr_db=8.5)
                                parts_e.append(ed); parts_b.append(bd); parts_bg.append(bgd); parts_v.append(vd); parts_f0.append(np.zeros(len(ed)))
                            split_pt = sum(len(x) for x in parts_e)

                    fc = bank.zh_finals.get(f, [])
                    if fc:
                        bf = min(
                            fc,
                            key=lambda c: (0.0 if c.speaker == tgt_name else 3.5)
                            + (0.0 if c.tone == tone else (4.0 if (tone != 5 and c.tone == 5) else 1.8))
                            + (0.0 if c.prev_final in (o, "h", "g", "k", "") else 1.5)
                            + (0.0 if c.next_onset == next_o else 1.2),
                        )
                        ef, bf_b, bgf, vf = _adapt_v11(
                            bf.log_env, bf.bap, bf.bap_gt8k, bf.voiced, bf.f0_rel_st,
                            _get_src_prof(bank, bf.speaker, target_profile, src_profiles), target_profile, cloner_fn, vtln_alpha,
                        )
                        ef, _ = calibrate_token_parseval_gain(ef, "zh_syllable", ref_vowel_db, max_corr_db=10.5)
                        parts_e.append(ef); parts_b.append(bf_b); parts_bg.append(bgf); parts_v.append(vf); parts_f0.append(bf.f0_rel_st)
                        source_counts["zh_concat_of"] = source_counts.get("zh_concat_of", 0) + 1
                    elif bank.fallback_unit_bank is not None:
                        for fi in cf.PINYIN_FINAL_TO_IPA.get(f, ("ə",)):
                            d = cw.lookup_ipa_template(bank.fallback_unit_bank, fi, is_onset=False)
                            bg_d = np.repeat(d["bap"][:, 4:5], 2, axis=1)
                            ed, bd, bgd, vd = _adapt_v11(
                                d["log_env"] - 3.1, d["bap"], bg_d, d["voiced"], np.zeros(len(d["log_env"])),
                                zh_fb_prof, target_profile, cloner_fn, vtln_alpha,
                            )
                            ed, _ = calibrate_token_parseval_gain(ed, "zh_syllable", ref_vowel_db, max_corr_db=10.5)
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

                raw_g = unit_gain_db[idx]
                e_st = e_st + float(np.clip(raw_g, -3.0, 3.0)) * math.log(10.0) / 20.0
                t_grid = np.linspace(0.0, 1.0, len(e_st))
                p5_arr = np.array([_interp_list(pitch_5, float(t)) for t in t_grid], dtype=np.float64) * iqr_scale
                f0_mic = np.clip(f0_tok - float(np.median(f0_tok)), -3.0, 3.0)
                st_u = 0.68 * p5_arr + 0.32 * f0_mic + unit_decl_st[idx]

                fr_start_u = cur_fr
                envs.append(e_st); baps.append(b_st); bgt8s.append(bg_st); vois.append(v_st); sts.append(st_u)
                is_obs_blk.append(o in OBSTRUENT_ONSETS)
                is_pause_blk.append(False)
                cur_fr += len(e_st)
                slot_timings_ms.append([fr_start_u * 5.0, cur_fr * 5.0])
                unit_boundaries_ms.append(cur_fr * 5.0)
                boundary_frames.append(cur_fr)

                if pause_ms > 15.0:
                    p_fr = max(2, int(round(pause_ms / 5.0)))
                    envs.append(np.full((p_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
                    baps.append(np.ones((p_fr, 5), dtype=np.float64))
                    bgt8s.append(np.ones((p_fr, 2), dtype=np.float64))
                    vois.append(np.zeros(p_fr, dtype=bool))
                    sts.append(np.zeros(p_fr, dtype=np.float64))
                    is_obs_blk.append(False); is_pause_blk.append(True)
                    cur_fr += p_fr
            u_idx = j
            prev_span_lang = "zh"
        else:
            j = u_idx
            while j < n_u and parsed[j][0] != "zh":
                j += 1

            en_phones_flat: list[tuple[int, str, int | None, bool, float, float, float]] = []
            word_phone_ranges: dict[int, tuple[int, int]] = {}
            for idx in range(u_idx, j):
                u = units[idx]
                ipas = list(u.get("ipa_phones") or ["ə"])
                arpa = u.get("arpabet") or []
                pairs = v6_arb._split_en_arpabet_to_ipa_with_stress(arpa) if arpa else [(p, None) for p in ipas]
                if not pairs:
                    pairs = [(p, None) for p in ipas]

                learned_pdurs = u.get("phone_durations_ms")
                if isinstance(learned_pdurs, list) and len(learned_pdurs) == len(pairs):
                    alloc = [max(4, int(round(float(ms) / 5.0))) for ms in learned_pdurs]
                else:
                    raw_d = [
                        (IPA_SPECS[ip].base_dur_ms if ip in IPA_SPECS else 70.0) * (1.18 if st == 1 else (0.82 if st == 0 else 1.0))
                        for ip, st in pairs
                    ]
                    nf_w = max(4, int(round(unit_dur_ms[idx] / 5.0)))
                    alloc = compute_perceptual_phone_frames(pairs, raw_d, nf_w)

                # Base pitch from unit pitch_st median if prosody_model applied
                p_st_list = u.get("pitch_st")
                u_p_base = float(np.median(p_st_list)) if p_st_list else 0.0

                w_start = len(en_phones_flat)
                for idx_p, ((ip, st), kp) in enumerate(zip(pairs, alloc)):
                    accent = 0.9 if (st == 1 and not unit_is_func[idx] and "gain_db" not in u) else 0.0
                    en_phones_flat.append((idx, ip, st, unit_is_func[idx], float(kp), u_p_base + unit_decl_st[idx] + accent, unit_gain_db[idx]))
                word_phone_ranges[idx] = (w_start, len(en_phones_flat))

            M = len(en_phones_flat)
            nodes = [("SIL", None, 4.0)] + [(p[1], p[2], p[4]) for p in en_phones_flat] + [("SIL", None, 4.0)]

            diphone_slots = []
            diphone_kinds: list[str] = []
            for k in range(len(nodes) - 1):
                p1, st1, d1_fr = nodes[k]
                p2, st2, d2_fr = nodes[k + 1]
                cands = bank.en_diphones.get((p1, p2), [])
                if cands:
                    target_tot_ms = (0.5 * d1_fr + 0.5 * d2_fr) * 5.0
                    tcosts = []
                    for c in cands:
                        tc = 0.0 if c.speaker == tgt_name else 3.2
                        if st1 is not None and c.stress1 is not None and c.stress1 != st1:
                            tc += 0.9
                        if st2 is not None and c.stress2 is not None and c.stress2 != st2:
                            tc += 0.9
                        tc += 0.55 * abs(math.log(max(20.0, c.dur_ms) / max(20.0, target_tot_ms)))
                        tcosts.append(tc)
                    diphone_slots.append((cands, tcosts))
                    diphone_kinds.append("en_diphone_exact")
                else:
                    pseudo, pkind = _make_pseudo_diphone(p1, p2, bank, tgt_name)
                    diphone_slots.append(([pseudo], [0.0]))
                    diphone_kinds.append(pkind)

            chosen_diphones = _viterbi_diphones(diphone_slots)
            for dk in diphone_kinds:
                source_counts[dk] = source_counts.get(dk, 0) + 1

            diphone_start_fr: list[int] = []
            diphone_end_fr: list[int] = []
            for k, dtok in enumerate(chosen_diphones):
                p1, _, d1_fr = nodes[k]
                p2, _, d2_fr = nodes[k + 1]
                f1_target = 2 if p1 == "SIL" else max(2, int(round(0.5 * d1_fr)))
                f2_target = 2 if p2 == "SIL" else max(2, int(round(0.5 * d2_fr)))

                e_ad, b_ad, bg_ad, v_ad = _adapt_v11(
                    dtok.log_env, dtok.bap, dtok.bap_gt8k, dtok.voiced, dtok.f0_rel_st,
                    _get_src_prof(bank, dtok.speaker, target_profile, src_profiles), target_profile, cloner_fn, vtln_alpha,
                )
                sp_split = max(1, min(len(e_ad) - 1, dtok.f_mid_split))
                e_l, e_r = e_ad[:sp_split], e_ad[sp_split:]
                b_l, b_r = b_ad[:sp_split], b_ad[sp_split:]
                bg_l, bg_r = bg_ad[:sp_split], bg_ad[sp_split:]
                v_l, v_r = v_ad[:sp_split], v_ad[sp_split:]
                f0_l, f0_r = dtok.f0_rel_st[:sp_split].astype(np.float64), dtok.f0_rel_st[sp_split:].astype(np.float64)

                if p1 != "SIL":
                    e_l, _ = calibrate_token_parseval_gain(e_l, classify_ipa_manner(p1, nodes[k][1]), ref_vowel_db, max_corr_db=6.5)
                if p2 != "SIL":
                    e_r, _ = calibrate_token_parseval_gain(e_r, classify_ipa_manner(p2, nodes[k + 1][1]), ref_vowel_db, max_corr_db=6.5)

                el_rs = _resample_2d(e_l, f1_target)
                er_rs = _resample_2d(e_r, f2_target)
                bl_rs = np.clip(_resample_2d(b_l, f1_target), 0.005, 0.995)
                br_rs = np.clip(_resample_2d(b_r, f2_target), 0.005, 0.995)
                bgl_rs = np.clip(_resample_2d(bg_l, f1_target), 0.005, 0.995)
                bgr_rs = np.clip(_resample_2d(bg_r, f2_target), 0.005, 0.995)
                vl_rs = _resample_1d(v_l.astype(np.float64), f1_target) >= 0.45
                vr_rs = _resample_1d(v_r.astype(np.float64), f2_target) >= 0.45
                f0l_rs = _resample_1d(f0_l, f1_target)
                f0r_rs = _resample_1d(f0_r, f2_target)

                if k > 0:
                    g1_db = en_phones_flat[k - 1][6]
                    p1_st = en_phones_flat[k - 1][5]
                    el_rs = el_rs + float(np.clip(g1_db, -3.0, 3.0)) * math.log(10.0) / 20.0
                else:
                    el_rs = np.full((f1_target, N_BINS), SILENCE_LOG_ENV, dtype=np.float64)
                    vl_rs[:] = False
                    p1_st = en_phones_flat[0][5]
                if k < M:
                    g2_db = en_phones_flat[k][6]
                    p2_st = en_phones_flat[k][5]
                    er_rs = er_rs + float(np.clip(g2_db, -3.0, 3.0)) * math.log(10.0) / 20.0
                else:
                    er_rs = np.full((f2_target, N_BINS), SILENCE_LOG_ENV, dtype=np.float64)
                    vr_rs[:] = False
                    p2_st = en_phones_flat[-1][5]

                e_dp = np.concatenate([el_rs, er_rs], axis=0)
                b_dp = np.concatenate([bl_rs, br_rs], axis=0)
                bg_dp = np.concatenate([bgl_rs, bgr_rs], axis=0)
                v_dp = np.concatenate([vl_rs, vr_rs], axis=0)
                f0_dp = np.concatenate([
                    np.clip(f0l_rs - float(np.median(f0l_rs)), -2.5, 2.5) * 0.35 + p1_st,
                    np.clip(f0r_rs - float(np.median(f0r_rs)), -2.5, 2.5) * 0.35 + p2_st,
                ])

                fr_s_dp = cur_fr
                if use_unvoiced_residual and dtok.raw_wav is not None and not np.any(v_dp):
                    residual_spans.append((cur_fr, cur_fr + len(e_dp), dtok.raw_wav))
                envs.append(e_dp); baps.append(b_dp); bgt8s.append(bg_dp); vois.append(v_dp); sts.append(f0_dp)
                is_obs_blk.append(p1 in OBSTRUENT_IPAS)
                is_pause_blk.append(False)
                cur_fr += len(e_dp)
                diphone_start_fr.append(fr_s_dp)
                diphone_end_fr.append(cur_fr)

            for idx in range(u_idx, j):
                w_p0, w_p1 = word_phone_ranges[idx]
                ms_s = diphone_start_fr[w_p0] * 5.0
                ms_e = diphone_end_fr[w_p1] * 5.0
                slot_timings_ms.append([ms_s, ms_e])
                unit_boundaries_ms.append(ms_e)
                boundary_frames.append(diphone_end_fr[w_p1])

            last_pause = float(units[j - 1].get("pause_after_ms", 0.0))
            if last_pause > 15.0:
                p_fr = max(2, int(round(last_pause / 5.0)))
                envs.append(np.full((p_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
                baps.append(np.ones((p_fr, 5), dtype=np.float64))
                bgt8s.append(np.ones((p_fr, 2), dtype=np.float64))
                vois.append(np.zeros(p_fr, dtype=bool))
                sts.append(np.zeros(p_fr, dtype=np.float64))
                is_obs_blk.append(False); is_pause_blk.append(True)
                cur_fr += p_fr
            u_idx = j
            prev_span_lang = "en"

    tail_fr = 6
    envs.append(np.full((tail_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((tail_fr, 5), dtype=np.float64))
    bgt8s.append(np.ones((tail_fr, 2), dtype=np.float64))
    vois.append(np.zeros(tail_fr, dtype=bool))
    sts.append(np.zeros(tail_fr, dtype=np.float64))
    is_obs_blk.append(False); is_pause_blk.append(True)

    # Steady-state midpoint seam blending
    for idx_b in range(len(envs) - 1):
        if is_pause_blk[idx_b] or is_pause_blk[idx_b + 1]:
            continue
        if np.any(vois[idx_b]) and np.any(vois[idx_b + 1]):
            bf = 1 if is_obs_blk[idx_b + 1] else min(3, len(envs[idx_b]), len(envs[idx_b + 1]))
            _blend_seam(envs[idx_b], envs[idx_b + 1], baps[idx_b], baps[idx_b + 1], blend_frames=bf)

    obs_fr_flags = np.concatenate([np.full(len(e), flag, dtype=bool) for e, flag in zip(envs, is_obs_blk)])
    pause_fr_flags = np.concatenate([np.full(len(e), flag, dtype=bool) for e, flag in zip(envs, is_pause_blk)])
    env_all = np.concatenate(envs, axis=0)
    bap_all = np.clip(np.concatenate(baps, axis=0), 0.005, 0.995)
    bgt8_all = np.clip(np.concatenate(bgt8s, axis=0), 0.005, 0.995)
    voi_all = np.concatenate(vois, axis=0)
    st_all = np.concatenate(sts, axis=0)

    # Cross-lingual ZH <-> EN 30 ms spectral-tilt + formant ramp bridge (preserves unvoiced bursts)
    cs_bridge_info = {}
    if use_cs_bridge and cs_switch_frames:
        env_all, cs_bridge_info = bridge_cs_boundary(env_all, voi_all, cs_switch_frames, sr=SR)

    env_all = smooth_utterance_gain_contour(
        env_all, voi_all, obs_fr_flags, pause_fr_flags, ref_vowel_db=ref_vowel_db
    )

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
    gbank_use = _pick_glottal_bank(bank, tgt_name, tgt_f0, glottal_bank)
    y = synthesize_v12(
        feats,
        mode=vocoder_mode,
        glottal_bank=gbank_use,
        crisp_unvoiced=crisp_unvoiced,
        seed=seed,
        phase_dispersion=phase_dispersion,
        bap_gt8k=bgt8_all,
        unvoiced_residual_spans=residual_spans if use_unvoiced_residual else None,
    )
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.89:
        y *= 0.89 / peak

    n_exact = source_counts.get("en_diphone_exact", 0)
    n_en_tot = n_exact + source_counts.get("en_phone_fallback", 0) + source_counts.get("en_synth_fill", 0)
    return y, {
        "source_counts": source_counts,
        "en_diphone_exact_pct": round(100.0 * n_exact / max(1, n_en_tot), 2) if n_en_tot else None,
        "n_frames": len(env_all),
        "slot_timings_ms": slot_timings_ms,
        "unit_boundaries_ms": unit_boundaries_ms[:-1] if len(unit_boundaries_ms) > 1 else [],
        "boundary_frames": boundary_frames[:-1] if len(boundary_frames) > 1 else [],
        "cs_switch_frames": cs_switch_frames,
        "cs_bridge_info": cs_bridge_info,
        "prosody_applied": prosody_applied,
        "log_env": env_all.astype(np.float32),
    }
