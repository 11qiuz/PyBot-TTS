"""Phase 9 (v13) Integrated UTAU Mandarin Speech, Guided Control & Virtual Singer Renderer."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from exp_v5_arch import crossling_frontend as cf
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS
from exp_v6_vocoder import vocoder
from exp_v7_pybot_tts import crossling_warp as cw
from exp_v8_clarity.utau_concat import (
    N_BINS,
    OBSTRUENT_ONSETS,
    SILENCE_LOG_ENV,
    _blend_seam,
    _interp_list,
    _resample_1d,
    _resample_2d,
    _stretch_preserving_onset,
)
from exp_v10_integrity.gain_calibration import (
    calibrate_token_parseval_gain,
    classify_ipa_manner,
    smooth_utterance_gain_contour,
)
from exp_v11_align_gmm.timbre_cloner_v5 import adapt_features_v5
from exp_v11_diphone_tts.v11_renderer import (
    REF_VOWEL_PARSEVAL_DB,
    _adapt_v11,
    _get_src_prof,
    _resolve_profile_attrs,
    _viterbi_zh_syllables,
)
from exp_v12_natural_vocoder.v12_renderer import _pick_glottal_bank
from exp_v12_natural_vocoder.vocoder_v12 import GlottalPulseBank, synthesize_v12
from exp_v12_prosody.prosody_model import ProsodyModelV12, apply_prosody_v12
from exp_v13_utau_ceiling import HOP, N_FFT, SR
from exp_v13_utau_ceiling.sandhi_v13 import apply_t3_sandhi_v13, plan_v13
from exp_v13_utau_ceiling.v13_deform import (
    calibrate_gain_gentle,
    compensate_f0_shift_tilt,
    join_seam_v13,
    smooth_utterance_gain_gentle,
    stretch_utau_steady_nucleus,
)
from exp_v13_utau_ceiling.v13_selector import (
    score_zh_candidate_v13,
    select_best_onset_final_pair,
    viterbi_zh_v13,
)
from exp_v13_utau_ceiling.zh_bank_v13 import V13ZhBank, V13ZhToken

ClonerFn = Callable[..., Any]


@dataclass
class V13Config:
    """Feature toggles enabling clean single-factor ablations over the v12 Mandarin baseline."""

    fix_viterbi_join: bool = True
    soft_context_and_f0_sel: bool = True
    joint_of_concat: bool = True
    gentle_gain: bool = True
    steady_nucleus_stretch: bool = True
    steady_stretch_min_ratio: float = 1.0
    allow_pingpong: bool = True
    spectral_seam_join: bool = True
    f0_tilt_comp: bool = False
    use_sandhi_v13: bool = True
    use_learned_prosody: bool = True
    singing_mode: bool = False


V12_BASELINE_CFG = V13Config(
    fix_viterbi_join=False,
    soft_context_and_f0_sel=False,
    joint_of_concat=False,
    gentle_gain=False,
    steady_nucleus_stretch=False,
    allow_pingpong=False,
    spectral_seam_join=False,
    f0_tilt_comp=False,
    use_sandhi_v13=False,
    use_learned_prosody=True,
)


def _interpolate_f0_hz_onto_mask(
    f0_hz_src: np.ndarray,
    tok_voiced: np.ndarray,
    fallback_f0: float,
    voiced_hint: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate a target F0 contour (Hz) cleanly onto the token's voiced frames."""
    n = len(tok_voiced)
    out = np.zeros(n, dtype=np.float64)
    src = np.asarray(f0_hz_src, dtype=np.float64).ravel()
    if src.size == 0:
        out[tok_voiced] = fallback_f0
        return out
    if voiced_hint is not None and len(voiced_hint) == len(src):
        v_mask = np.asarray(voiced_hint, dtype=bool) & (src > 40.0)
    else:
        v_mask = src > 40.0
    v_idx = np.where(v_mask)[0]
    if v_idx.size == 0:
        out[tok_voiced] = fallback_f0
        return out
    t_src = np.linspace(0.0, 1.0, len(src))[v_idx]
    log_f0 = np.log2(src[v_idx])
    t_dst = np.linspace(0.0, 1.0, n)
    interp_hz = 2.0 ** np.interp(t_dst, t_src, log_f0)
    out[tok_voiced] = np.clip(interp_hz[tok_voiced], 60.0, 460.0)
    return out


def _select_zh_tokens_v13(
    parsed_units: list[tuple[str, str, int]],
    unit_dur_ms: list[float],
    unit_pause_before: list[bool],
    unit_target_f0_st: list[float | None],
    unit_target_onset_ms: list[float | None],
    unit_pos: list[str],
    bank: V13ZhBank,
    tgt_name: str,
    cfg: V13Config,
) -> list[tuple[str, V13ZhToken | None]]:
    """Select whole-syllable Mandarin tokens via Viterbi (either v12-style or v13-enhanced)."""
    n_u = len(parsed_units)
    slots: list[tuple[list[Any], list[float], bool, bool]] = []
    slot_indices: list[int] = []
    selected: dict[int, V13ZhToken | None] = {}

    for idx, (o, f, tone) in enumerate(parsed_units):
        prev_f = parsed_units[idx - 1][1] if idx > 0 else "SIL"
        next_o = parsed_units[idx + 1][0] if idx + 1 < n_u else "SIL"
        cands = bank.zh_syllables.get((o, f), [])
        p_bef = unit_pause_before[idx]
        if cands:
            is_uv_zh = o in ("f", "s", "sh", "x", "h", "c", "ch", "q")
            tcosts: list[float] = []
            for c in cands:
                if cfg.soft_context_and_f0_sel:
                    tc = score_zh_candidate_v13(
                        c,
                        target_speaker=tgt_name,
                        target_tone=tone,
                        prev_final=prev_f,
                        next_onset=next_o,
                        target_dur_ms=unit_dur_ms[idx],
                        target_f0_st=unit_target_f0_st[idx],
                        target_onset_ms=unit_target_onset_ms[idx],
                        target_pos=unit_pos[idx],
                    )
                    if cfg.singing_mode and int(c.tone) == 5 and unit_dur_ms[idx] > 180.0:
                        tc += 2.5
                else:
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
                chosen = viterbi_zh_v13(slots) if cfg.fix_viterbi_join else _viterbi_zh_syllables(slots)
                for si, tk in zip(slot_indices, chosen):
                    selected[si] = tk
                slots, slot_indices = [], []
            selected[idx] = None

    if slots:
        chosen = viterbi_zh_v13(slots) if cfg.fix_viterbi_join else _viterbi_zh_syllables(slots)
        for si, tk in zip(slot_indices, chosen):
            selected[si] = tk

    out: list[tuple[str, V13ZhToken | None]] = []
    for idx in range(n_u):
        tk = selected.get(idx)
        if tk is not None:
            mode = "zh_target_syl" if tk.speaker == tgt_name else "zh_donor_syl"
            out.append((mode, tk))
        else:
            out.append(("zh_concat_of", None))
    return out


def render_v13_utterance(
    text_or_plan: str | dict[str, Any],
    bank: V13ZhBank,
    target_profile: Any,
    *,
    guided_controls: list[dict[str, Any]] | None = None,
    cfg: V13Config | None = None,
    prosody_model: ProsodyModelV12 | None = None,
    cloner_fn: ClonerFn | None = None,
    src_profiles: dict[str, Any] | None = None,
    vtln_alpha: float | None = None,
    glottal_bank: GlottalPulseBank | None = None,
    use_unvoiced_residual: bool = True,
    ref_vowel_db: float = REF_VOWEL_PARSEVAL_DB,
    seed: int = 72,
    phase_dispersion: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render Mandarin utterance using configurable v13 UTAU selection, deformation & exact control."""
    if cfg is None:
        cfg = V13Config()
    if cloner_fn is None:
        cloner_fn = adapt_features_v5

    if isinstance(text_or_plan, str):
        if cfg.use_sandhi_v13:
            plan_doc = plan_v13(text_or_plan)
        else:
            from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10

            plan_doc = plan_v10(text_or_plan)
    else:
        plan_doc = apply_t3_sandhi_v13(text_or_plan) if (cfg.use_sandhi_v13 and guided_controls is None) else text_or_plan

    prosody_applied = False
    if cfg.use_learned_prosody and guided_controls is None:
        try:
            plan_doc = apply_prosody_v12(plan_doc, model=prosody_model)
            prosody_applied = True
        except Exception:
            prosody_applied = False

    units = plan_doc.get("units") or []
    if not units:
        raise ValueError("plan_doc has no units")
    if guided_controls is not None and len(guided_controls) != len(units):
        raise ValueError(f"guided_controls length ({len(guided_controls)}) != units ({len(units)})")

    tgt_name = getattr(target_profile, "name", "")
    tgt_f0, tgt_iqr = _resolve_profile_attrs(target_profile)
    iqr_scale = max(0.55, min(1.75, tgt_iqr / 3.2))
    n_u = len(units)

    parsed: list[tuple[str, str, int]] = []
    for idx, u in enumerate(units):
        if u.get("lang", "zh") != "zh":
            raise ValueError("render_v13_utterance currently targets Mandarin units (use v12 for EN/CS)")
        gc = guided_controls[idx] if guided_controls is not None else {}
        o = str(gc.get("onset", u.get("onset", "")))
        f = str(gc.get("final", u.get("final", "")))
        if o in ("y", "w"):
            o = ""
        if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
            f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"
        tone = int(gc.get("tone", u.get("tone") or 1))
        parsed.append((o, f, tone))

    # Breath-group segmentation
    group_id = [0] * n_u
    cur_g = 0
    for i, u in enumerate(units):
        group_id[i] = cur_g
        p_ms = float(guided_controls[i].get("pause_after_ms", u.get("pause_after_ms", 0.0))) if guided_controls else float(u.get("pause_after_ms", 0.0))
        if p_ms > 55.0:
            cur_g += 1
    group_spans: dict[int, list[int]] = {}
    for i, g in enumerate(group_id):
        if g not in group_spans:
            group_spans[g] = [i, i]
        else:
            group_spans[g][1] = i

    unit_dur_ms: list[float] = []
    unit_decl_st: list[float] = []
    unit_gain_db: list[float] = []
    unit_pause_before: list[bool] = []
    unit_target_f0_st: list[float | None] = []
    unit_target_onset_ms: list[float | None] = []
    unit_pos: list[str] = []

    for i, u in enumerate(units):
        gc = guided_controls[i] if guided_controls is not None else {}
        g0, g1 = group_spans[group_id[i]]
        g_len = max(1, g1 - g0)
        rel_pos = (i - g0) / max(1, g_len)

        if "dur_ms" in gc:
            base_d = max(15.0, float(gc["dur_ms"]))
        elif "gain_db" in u and "duration_ms" in u:
            base_d = max(55.0, float(u["duration_ms"]))
        else:
            base_d = max(60.0, float(u.get("duration_ms", 200.0)))

        if "gain_db" in gc:
            gdb = float(gc["gain_db"])
        elif "gain_db" in u:
            gdb = float(u["gain_db"])
        else:
            gdb = 0.25 * (1.0 - rel_pos)
            if i == g1:
                base_d *= 1.16 if "dur_ms" not in gc else 1.0
                gdb -= 0.8
            elif g_len >= 4 and g0 < i < g1 and "dur_ms" not in gc:
                base_d *= 0.96

        decl = (0.25 - 0.50 * rel_pos) if ("gain_db" in u and "duration_ms" in u) else (0.75 - 1.50 * rel_pos)
        prev_p = (
            float(guided_controls[i - 1].get("pause_after_ms", units[i - 1].get("pause_after_ms", 0.0)))
            if (i > 0 and guided_controls is not None)
            else (float(units[i - 1].get("pause_after_ms", 0.0)) if i > 0 else 0.0)
        )

        # Estimate target F0 in semitones rel tgt_f0 for better unit selection
        tf0_st: float | None = None
        if "f0_hz" in gc:
            arr_f = np.asarray(gc["f0_hz"], dtype=np.float64).ravel()
            v_f = arr_f[arr_f > 40.0]
            if v_f.size > 0:
                tf0_st = float(12.0 * math.log2(float(np.median(v_f)) / max(50.0, tgt_f0)))
        elif "f0_st_contour" in gc:
            tf0_st = float(np.median(np.asarray(gc["f0_st_contour"], dtype=np.float64)))
        elif u.get("pitch_st"):
            tf0_st = float(np.median(u["pitch_st"])) * iqr_scale + decl

        unit_dur_ms.append(base_d)
        unit_decl_st.append(decl)
        unit_gain_db.append(gdb)
        unit_pause_before.append(i > 0 and prev_p > 15.0)
        unit_target_f0_st.append(tf0_st)
        unit_target_onset_ms.append(float(gc["onset_ms"]) if "onset_ms" in gc else None)
        unit_pos.append("initial" if i == g0 else ("final" if i == g1 else "medial"))

    chosen_syls = _select_zh_tokens_v13(
        parsed,
        unit_dur_ms,
        unit_pause_before,
        unit_target_f0_st,
        unit_target_onset_ms,
        unit_pos,
        bank,
        tgt_name,
        cfg,
    )

    envs, baps, bgt8s, vois = [], [], [], []
    is_obs_blk, is_pause_blk = [], []
    explicit_f0_hz_blocks: list[np.ndarray | None] = []
    sts_rel_blocks: list[np.ndarray] = []
    residual_spans: list[tuple[int, int, np.ndarray]] = []
    source_counts: dict[str, int] = {}
    slot_timings_ms: list[list[float]] = []
    unit_boundaries_ms: list[float] = []
    boundary_frames: list[int] = []
    selected_trace: list[dict[str, Any]] = []

    lead_fr = 4
    envs.append(np.full((lead_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((lead_fr, 5), dtype=np.float64))
    bgt8s.append(np.ones((lead_fr, 2), dtype=np.float64))
    vois.append(np.zeros(lead_fr, dtype=bool))
    explicit_f0_hz_blocks.append(np.zeros(lead_fr, dtype=np.float64))
    sts_rel_blocks.append(np.zeros(lead_fr, dtype=np.float64))
    is_obs_blk.append(False)
    is_pause_blk.append(True)
    cur_fr = lead_fr

    zh_fb_prof = _get_src_prof(bank, "SSB0502", target_profile, src_profiles) or (
        next(iter(bank.profiles_v2.values())) if bank.profiles_v2 else target_profile
    )

    same_utt_chain_hits = 0
    same_utt_chain_possible = 0
    pingpong_count = 0
    prev_tok: V13ZhToken | None = None

    for idx in range(n_u):
        o, f, tone = parsed[idx]
        u = units[idx]
        gc = guided_controls[idx] if guided_controls is not None else {}
        nf = max(3 if guided_controls is not None else 12, int(round(unit_dur_ms[idx] / 5.0)))
        pitch_5 = [float(x) for x in u.get("pitch_st", [0.0] * 5)]
        pause_ms = float(gc.get("pause_after_ms", u.get("pause_after_ms", 0.0)))
        prev_f = parsed[idx - 1][1] if idx > 0 else "SIL"
        next_o = parsed[idx + 1][0] if idx + 1 < n_u else "SIL"
        on_target_fr = max(1, int(round(float(gc["onset_ms"]) / 5.0))) if "onset_ms" in gc else None

        mode, best = chosen_syls[idx]
        if idx > 0 and not unit_pause_before[idx]:
            same_utt_chain_possible += 1
            if best is not None and prev_tok is not None and best.utterance_id == prev_tok.utterance_id and best.syl_idx == prev_tok.syl_idx + 1:
                same_utt_chain_hits += 1
        prev_tok = best

        if best is not None:
            e_raw, b_raw, bg_raw, v_raw = _adapt_v11(
                best.log_env,
                best.bap,
                best.bap_gt8k,
                best.voiced,
                best.f0_rel_st,
                _get_src_prof(bank, best.speaker, target_profile, src_profiles),
                target_profile,
                cloner_fn,
                vtln_alpha,
            )
            if cfg.gentle_gain:
                e_raw, _ = calibrate_gain_gentle(e_raw, "zh_syllable", ref_vowel_db)
            else:
                e_raw, _ = calibrate_token_parseval_gain(e_raw, "zh_syllable", ref_vowel_db, max_corr_db=10.5)

            source_counts[mode] = source_counts.get(mode, 0) + 1
            use_steady_tok = cfg.steady_nucleus_stretch and (
                cfg.singing_mode or (nf / float(max(1, len(e_raw))) >= cfg.steady_stretch_min_ratio)
            )
            if use_steady_tok:
                e_st, b_st, bg_st, v_st, f0_tok, f_sp_new, used_pp = stretch_utau_steady_nucleus(
                    e_raw,
                    b_raw,
                    bg_raw,
                    v_raw,
                    best.f0_rel_st.astype(np.float64),
                    best.f_split,
                    nf,
                    target_onset_frames=on_target_fr,
                    allow_pingpong=cfg.allow_pingpong,
                )
                if used_pp:
                    pingpong_count += 1
            else:
                e_st, b_st, v_st, f_sp_new = _stretch_preserving_onset(e_raw, b_raw, v_raw, best.f_split, nf, preserve_onset=True)
                bg_st = np.clip(_resample_2d(bg_raw, nf), 0.005, 0.995)
                f0_tok = _resample_1d(best.f0_rel_st.astype(np.float64), nf)

            if use_unvoiced_residual and best.onset_wav is not None and f_sp_new > 0:
                residual_spans.append((cur_fr, cur_fr + f_sp_new, best.onset_wav))
            cand_mean_f0_st = best.mean_f0_st
            selected_trace.append({
                "idx": idx, "hanzi": u.get("hanzi", ""), "pinyin": f"{o}{f}{tone}",
                "mode": mode, "src_utt": best.utterance_id, "src_spk": best.speaker,
                "src_tone": best.tone, "src_syl_idx": best.syl_idx,
            })
        else:
            parts_e, parts_b, parts_bg, parts_v, parts_f0 = [], [], [], [], []
            split_pt = 0
            bo_sel = None
            oc = bank.zh_onsets.get(o, []) if o else []
            fc = bank.zh_finals.get(f, [])
            if cfg.joint_of_concat and o and oc and fc:
                bo_joint, bf_joint = select_best_onset_final_pair(
                    oc, fc,
                    onset=o, final=f, tone=tone, prev_final=prev_f, next_onset=next_o,
                    target_speaker=tgt_name, target_dur_ms=unit_dur_ms[idx],
                )
            else:
                bo_joint, bf_joint = None, None

            if o:
                if oc:
                    is_uv_o = o in ("f", "s", "sh", "x", "h", "c", "ch", "q")
                    bo = bo_joint if bo_joint is not None else min(
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
                    if cfg.gentle_gain:
                        eo, _ = calibrate_gain_gentle(eo, "voiceless_fricative" if is_uv_o else "nasal", ref_vowel_db, max_corr_db=5.0)
                    else:
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

            if fc:
                bf = bf_joint if bf_joint is not None else min(
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
                if cfg.gentle_gain:
                    ef, _ = calibrate_gain_gentle(ef, "zh_syllable", ref_vowel_db)
                else:
                    ef, _ = calibrate_token_parseval_gain(ef, "zh_syllable", ref_vowel_db, max_corr_db=10.5)
                parts_e.append(ef); parts_b.append(bf_b); parts_bg.append(bgf); parts_v.append(vf); parts_f0.append(bf.f0_rel_st)
                source_counts["zh_concat_of"] = source_counts.get("zh_concat_of", 0) + 1
                cand_mean_f0_st = bf.mean_f0_st
                selected_trace.append({
                    "idx": idx, "hanzi": u.get("hanzi", ""), "pinyin": f"{o}{f}{tone}",
                    "mode": "zh_concat_of", "onset_utt": bo_sel.utterance_id if bo_sel else None,
                    "final_utt": bf.utterance_id, "final_spk": bf.speaker,
                })
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
                cand_mean_f0_st = 0.0
                selected_trace.append({"idx": idx, "hanzi": u.get("hanzi", ""), "pinyin": f"{o}{f}{tone}", "mode": "zh_ipa_fill"})

            cat_e = np.concatenate(parts_e, axis=0)
            cat_b = np.concatenate(parts_b, axis=0)
            cat_bg = np.concatenate(parts_bg, axis=0)
            cat_v = np.concatenate(parts_v, axis=0)
            cat_f0 = np.concatenate(parts_f0)
            use_steady_cat = cfg.steady_nucleus_stretch and (
                cfg.singing_mode or (nf / float(max(1, len(cat_e))) >= cfg.steady_stretch_min_ratio)
            )
            if use_steady_cat:
                e_st, b_st, bg_st, v_st, f0_tok, f_sp_new, used_pp = stretch_utau_steady_nucleus(
                    cat_e, cat_b, cat_bg, cat_v, cat_f0, split_pt, nf,
                    target_onset_frames=on_target_fr, allow_pingpong=cfg.allow_pingpong,
                )
                if used_pp:
                    pingpong_count += 1
            else:
                e_st, b_st, v_st, f_sp_new = _stretch_preserving_onset(cat_e, cat_b, cat_v, split_pt, nf, preserve_onset=True)
                bg_st = np.clip(_resample_2d(cat_bg, nf), 0.005, 0.995)
                f0_tok = _resample_1d(cat_f0, nf)
            if use_unvoiced_residual and bo_sel is not None and bo_sel.onset_wav is not None and f_sp_new > 0:
                residual_spans.append((cur_fr, cur_fr + f_sp_new, bo_sel.onset_wav))

        # Gain adjustment
        raw_g = unit_gain_db[idx]
        if "gain_db" in gc:
            e_st = e_st + float(np.clip(raw_g, -12.0, 12.0)) * math.log(10.0) / 20.0
        else:
            e_st = e_st + float(np.clip(raw_g, -3.0, 3.0)) * math.log(10.0) / 20.0

        # Pitch target (explicit guided/singing F0 vs standard Mandarin contour)
        if "f0_hz" in gc:
            f0_hz_blk = _interpolate_f0_hz_onto_mask(
                np.asarray(gc["f0_hz"], dtype=np.float64),
                v_st,
                tgt_f0,
                voiced_hint=gc.get("voiced_hint"),
            )
            explicit_f0_hz_blocks.append(f0_hz_blk)
            v_hz = f0_hz_blk[v_st]
            tgt_mean_st = float(np.median(12.0 * np.log2(v_hz / max(50.0, tgt_f0)))) if v_hz.size else 0.0
            sts_rel_blocks.append(np.zeros(len(e_st), dtype=np.float64))
        elif "f0_st_contour" in gc:
            st_c = _resample_1d(np.asarray(gc["f0_st_contour"], dtype=np.float64), len(e_st))
            f0_hz_blk = np.zeros(len(e_st), dtype=np.float64)
            f0_hz_blk[v_st] = np.clip(tgt_f0 * (2.0 ** (st_c[v_st] / 12.0)), 60.0, 460.0)
            explicit_f0_hz_blocks.append(f0_hz_blk)
            tgt_mean_st = float(np.median(st_c[v_st])) if np.any(v_st) else 0.0
            sts_rel_blocks.append(np.zeros(len(e_st), dtype=np.float64))
        else:
            t_grid = np.linspace(0.0, 1.0, len(e_st))
            p5_arr = np.array([_interp_list(pitch_5, float(t)) for t in t_grid], dtype=np.float64) * iqr_scale
            f0_mic = np.clip(f0_tok - float(np.median(f0_tok)), -3.0, 3.0)
            st_u = 0.68 * p5_arr + 0.32 * f0_mic + unit_decl_st[idx]
            explicit_f0_hz_blocks.append(None)
            tgt_mean_st = float(np.median(st_u))
            sts_rel_blocks.append(st_u)

        if cfg.f0_tilt_comp:
            delta_st = tgt_mean_st - float(cand_mean_f0_st)
            e_st = compensate_f0_shift_tilt(e_st, v_st, delta_st, slope_db_oct_per_12st=1.0, sr=SR, n_fft=N_FFT)

        fr_start_u = cur_fr
        envs.append(e_st); baps.append(b_st); bgt8s.append(bg_st); vois.append(v_st)
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
            explicit_f0_hz_blocks.append(np.zeros(p_fr, dtype=np.float64))
            sts_rel_blocks.append(np.zeros(p_fr, dtype=np.float64))
            is_obs_blk.append(False); is_pause_blk.append(True)
            cur_fr += p_fr

    tail_fr = 6
    envs.append(np.full((tail_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((tail_fr, 5), dtype=np.float64))
    bgt8s.append(np.ones((tail_fr, 2), dtype=np.float64))
    vois.append(np.zeros(tail_fr, dtype=bool))
    explicit_f0_hz_blocks.append(np.zeros(tail_fr, dtype=np.float64))
    sts_rel_blocks.append(np.zeros(tail_fr, dtype=np.float64))
    is_obs_blk.append(False); is_pause_blk.append(True)

    # Seam blending
    for idx_b in range(len(envs) - 1):
        if is_pause_blk[idx_b] or is_pause_blk[idx_b + 1]:
            continue
        if cfg.spectral_seam_join:
            join_seam_v13(
                envs[idx_b], envs[idx_b + 1],
                baps[idx_b], baps[idx_b + 1],
                vois[idx_b], vois[idx_b + 1],
                obs_cur=is_obs_blk[idx_b + 1],
            )
        elif np.any(vois[idx_b]) and np.any(vois[idx_b + 1]):
            bf = 1 if is_obs_blk[idx_b + 1] else min(3, len(envs[idx_b]), len(envs[idx_b + 1]))
            _blend_seam(envs[idx_b], envs[idx_b + 1], baps[idx_b], baps[idx_b + 1], blend_frames=bf)

    obs_fr_flags = np.concatenate([np.full(len(e), flag, dtype=bool) for e, flag in zip(envs, is_obs_blk)])
    pause_fr_flags = np.concatenate([np.full(len(e), flag, dtype=bool) for e, flag in zip(envs, is_pause_blk)])
    env_all = np.concatenate(envs, axis=0)
    bap_all = np.clip(np.concatenate(baps, axis=0), 0.005, 0.995)
    bgt8_all = np.clip(np.concatenate(bgt8s, axis=0), 0.005, 0.995)
    voi_all = np.concatenate(vois, axis=0)

    if cfg.gentle_gain:
        env_all = smooth_utterance_gain_gentle(env_all, voi_all, obs_fr_flags, pause_fr_flags, ref_vowel_db=ref_vowel_db)
    else:
        env_all = smooth_utterance_gain_contour(env_all, voi_all, obs_fr_flags, pause_fr_flags, ref_vowel_db=ref_vowel_db)

    # Assemble final F0 track (exact guided Hz when supplied; standard median-centered relative contour otherwise)
    if any(b is not None for b in explicit_f0_hz_blocks):
        f0_all = np.zeros(len(env_all), dtype=np.float64)
        cursor = 0
        for e_blk, v_blk, hz_blk, st_blk in zip(envs, vois, explicit_f0_hz_blocks, sts_rel_blocks):
            L = len(e_blk)
            if hz_blk is not None:
                f0_all[cursor : cursor + L] = hz_blk
            else:
                f0_all[cursor : cursor + L][v_blk] = np.clip(tgt_f0 * (2.0 ** (st_blk[v_blk] / 12.0)), 65.0, 440.0)
            cursor += L
    else:
        st_all = np.concatenate(sts_rel_blocks, axis=0)
        f0_all = np.zeros(len(st_all), dtype=np.float64)
        v_idx = np.where(voi_all)[0]
        if v_idx.size > 0:
            st_centered = st_all[v_idx] - float(np.median(st_all[v_idx]))
            if len(st_centered) >= 5:
                sm = st_centered.copy()
                sm[1:-1] = 0.25 * st_centered[:-2] + 0.50 * st_centered[1:-1] + 0.25 * st_centered[2:]
                st_centered = sm
            f0_all[v_idx] = np.clip(tgt_f0 * (2.0 ** (st_centered / 12.0)), 65.0, 440.0)

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
        mode="hybrid_lp_hnm",
        glottal_bank=gbank_use,
        crisp_unvoiced=True,
        seed=seed,
        phase_dispersion=phase_dispersion,
        bap_gt8k=bgt8_all,
        unvoiced_residual_spans=residual_spans if use_unvoiced_residual else None,
    )
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.89:
        y *= 0.89 / peak

    return y, {
        "source_counts": source_counts,
        "n_frames": len(env_all),
        "slot_timings_ms": slot_timings_ms,
        "unit_boundaries_ms": unit_boundaries_ms[:-1] if len(unit_boundaries_ms) > 1 else [],
        "boundary_frames": boundary_frames[:-1] if len(boundary_frames) > 1 else [],
        "prosody_applied": prosody_applied,
        "same_utt_chain_hits": same_utt_chain_hits,
        "same_utt_chain_rate": round(same_utt_chain_hits / max(1, same_utt_chain_possible), 4),
        "pingpong_count": pingpong_count,
        "selected_trace": selected_trace,
        "f0_hz_all": f0_all,
        "log_env": env_all.astype(np.float32),
    }


def _midi_to_hz(midi_val: float) -> float:
    return 440.0 * (2.0 ** ((float(midi_val) - 69.0) / 12.0))


def render_singing_score_v13(
    notes: list[dict[str, Any]],
    bank: V13ZhBank,
    target_profile: Any,
    *,
    cfg: V13Config | None = None,
    cloner_fn: ClonerFn | None = None,
    glottal_bank: GlottalPulseBank | None = None,
    seed: int = 72,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render a Mandarin virtual-singer score with MIDI notes, rests, portamento & vibrato.

    Note schema:
      - Rest: `{"rest_ms": 160.0}`
      - Note: `{"hanzi": "明", "pinyin": "ming2" (optional), "midi_note": 60.0, "dur_ms": 480.0,
                "gain_db": 0.0, "portamento_ms": 35.0, "vibrato_hz": 5.4,
                "vibrato_depth_st": 0.24, "vibrato_delay_ms": 180.0}`
    """
    if cfg is None:
        cfg = V13Config(singing_mode=True, use_learned_prosody=False, use_sandhi_v13=False)

    # Filter out standalone rests while attaching rest duration as pause_after_ms on previous note
    sung_notes: list[dict[str, Any]] = []
    for item in notes:
        if "rest_ms" in item and "midi_note" not in item:
            if sung_notes:
                sung_notes[-1]["_pause_after_ms"] = sung_notes[-1].get("_pause_after_ms", 0.0) + float(item["rest_ms"])
            continue
        sung_notes.append(dict(item))

    if not sung_notes:
        raise ValueError("Singing score must contain at least one sung note")

    units: list[dict[str, Any]] = []
    guided_controls: list[dict[str, Any]] = []
    score_note_targets: list[dict[str, Any]] = []

    prev_midi: float | None = None
    prev_paused = True

    for i, note in enumerate(sung_notes):
        dur_ms = max(60.0, float(note["dur_ms"]))
        nf = max(12, int(round(dur_ms / 5.0)))
        midi_val = float(note["midi_note"])
        pause_after = float(note.get("_pause_after_ms", note.get("pause_after_ms", 0.0)))

        if "pinyin" in note and note["pinyin"]:
            o, f, tone, ipas = cf.pinyin_to_ipa(str(note["pinyin"]))
            hanzi = str(note.get("hanzi") or note["pinyin"])
        else:
            hanzi = str(note["hanzi"])
            pl_one = plan_v13(hanzi)
            u0 = pl_one["units"][0]
            o, f, tone, ipas = str(u0["onset"]), str(u0["final"]), int(u0["tone"]), list(u0["ipa_phones"])

        if o in ("y", "w"):
            o = ""
        if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
            f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"

        # Build continuous 5 ms frame MIDI curve with portamento + delayed vibrato
        t_ms = np.arange(nf, dtype=np.float64) * 5.0
        midi_track = np.full(nf, midi_val, dtype=np.float64)
        porta_ms = float(note.get("portamento_ms", 35.0))
        if (not prev_paused) and (prev_midi is not None) and porta_ms > 5.0 and abs(midi_val - prev_midi) > 0.05:
            p_end_ms = min(0.45 * dur_ms, porta_ms)
            mask_p = t_ms < p_end_ms
            u_p = np.clip(t_ms[mask_p] / max(1e-3, p_end_ms), 0.0, 1.0)
            w_p = 0.5 * (1.0 - np.cos(math.pi * u_p))
            midi_track[mask_p] = (1.0 - w_p) * prev_midi + w_p * midi_val

        vib_delay = float(note.get("vibrato_delay_ms", 180.0))
        vib_depth = float(note.get("vibrato_depth_st", 0.22))
        vib_hz = float(note.get("vibrato_hz", 5.4))
        if dur_ms > vib_delay + 60.0 and vib_depth > 0.0:
            fade_ms = 80.0
            env_v = np.clip((t_ms - vib_delay) / fade_ms, 0.0, 1.0)
            env_v = 0.5 * (1.0 - np.cos(math.pi * env_v))
            vib_st = env_v * vib_depth * np.sin(math.tau * vib_hz * np.maximum(0.0, t_ms - vib_delay) / 1000.0)
            midi_track += vib_st

        f0_hz_track = 440.0 * (2.0 ** ((midi_track - 69.0) / 12.0))

        units.append({
            "lang": "zh",
            "hanzi": hanzi,
            "token": hanzi,
            "onset": o,
            "final": f,
            "tone": tone,
            "lexical_tone": tone,
            "pinyin": f"{o}{f}{tone}",
            "ipa_phones": ipas,
            "stress_or_tone": f"T{tone}",
            "duration_ms": nf * 5.0,
            "pause_after_ms": pause_after,
            "pitch_st": [0.0] * 5,
        })
        guided_controls.append({
            "onset": o,
            "final": f,
            "tone": tone,
            "dur_ms": nf * 5.0,
            "f0_hz": f0_hz_track,
            "gain_db": float(note.get("gain_db", 0.0)),
            "pause_after_ms": pause_after,
        })
        score_note_targets.append({
            "idx": i,
            "hanzi": hanzi,
            "pinyin": f"{o}{f}{tone}",
            "midi_note": midi_val,
            "target_hz": round(_midi_to_hz(midi_val), 2),
            "dur_ms": nf * 5.0,
            "f0_hz_track": f0_hz_track,
        })
        prev_midi = midi_val
        prev_paused = pause_after > 15.0

    plan_doc = {"schema": 2, "text": "".join(u["hanzi"] for u in units), "style": "singing", "languages": ["zh"], "units": units}
    y, trace = render_v13_utterance(
        plan_doc,
        bank,
        target_profile,
        guided_controls=guided_controls,
        cfg=cfg,
        cloner_fn=cloner_fn,
        glottal_bank=glottal_bank,
        seed=seed,
    )
    trace["score_notes"] = score_note_targets
    return y, trace
