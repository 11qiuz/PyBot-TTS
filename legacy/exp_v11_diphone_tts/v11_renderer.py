"""Phase 7 (v11) English Diphone Viterbi + Mandarin Whole-Syllable Rescue + Smooth HNM Renderer."""
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
from exp_v11_diphone_tts import HOP, N_FFT, SR
from exp_v11_diphone_tts.diphone_bank import (
    EnDiphoneToken,
    V11DiphoneDualBank,
    V11ZhToken,
    _edge_stats,
)
from exp_v11_diphone_tts.hnm_v11 import synthesize_v11


ClonerFn = Callable[..., Any]
REF_VOWEL_PARSEVAL_DB = -17.5

EN_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "from", "by", "with", "as",
    "is", "was", "are", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "they", "them", "their", "he", "him", "his", "she", "her", "it", "its", "we", "us", "our",
    "you", "your", "i", "me", "my", "this", "that", "these", "those", "and", "or", "but", "not",
    "so", "if", "would", "could", "should", "can", "will", "were", "there", "here", "than",
})


def _default_plan(text: str) -> dict:
    try:
        from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10
        return plan_v10(text)
    except Exception:
        return cf.plan(text, allow_oov_fallback=True)


def _resolve_profile_attrs(prof: Any) -> tuple[float, float]:
    base = getattr(prof, "v3", prof)
    return float(getattr(base, "f0_median_hz", 180.0)), float(getattr(base, "f0_iqr_st", 3.2))


def _get_src_prof(bank: V11DiphoneDualBank, spk: str, tgt_prof: Any, src_profiles: dict[str, Any] | None) -> Any:
    if src_profiles is not None and spk in src_profiles:
        return src_profiles[spk]
    if (hasattr(tgt_prof, "vu_rms_gap_db") or hasattr(tgt_prof, "v3")) and spk in bank.profiles_v5:
        return bank.profiles_v5[spk]
    if spk in bank.profiles_v2:
        return bank.profiles_v2[spk]
    if spk in bank.profiles_v5:
        return bank.profiles_v5[spk]
    return None


def _adapt_v11(
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

    src_f0, _ = _resolve_profile_attrs(src_prof)
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
    if cloner_fn is not None:
        fn = cloner_fn
    elif hasattr(tgt_prof, "vu_rms_gap_db") or hasattr(tgt_prof, "v3"):
        try:
            from exp_v11_align_gmm.timbre_cloner_v5 import adapt_features_v5
            fn = adapt_features_v5
        except Exception:
            from exp_v10_g2p_and_align.timbre_cloner_v4 import adapt_features_v4
            fn = adapt_features_v4
    else:
        fn = tc2.adapt_features_v2

    try:
        res = fn(feats, src_prof, tgt_prof, vtln_alpha=vtln_alpha) if vtln_alpha is not None else fn(feats, src_prof, tgt_prof)
    except TypeError:
        res = fn(feats, src_prof, tgt_prof)
    ad = res[0] if isinstance(res, tuple) else res
    ratio_b4 = np.clip(ad.bap[:, 4:5] / np.maximum(bap[:, 4:5], 1e-3), 0.25, 2.5)
    bg_ad = np.clip(bap_gt8k.astype(np.float64) * ratio_b4, 0.005, 0.995)
    return ad.log_env, ad.bap, bg_ad, ad.voiced


def _make_pseudo_diphone(
    p1: str,
    p2: str,
    bank: V11DiphoneDualBank,
    tgt_name: str,
) -> tuple[EnDiphoneToken, str]:
    """Synthesize a smooth fallback diphone (p1_right -> p2_left) from verified single phones or IPA templates."""
    def _half_tok(ip: str, take_right: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
        if ip == "SIL":
            return (
                np.full((2, N_BINS), SILENCE_LOG_ENV, dtype=np.float32),
                np.ones((2, 5), dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
                np.zeros(2, dtype=bool),
                np.zeros(2, dtype=np.float32),
                tgt_name,
                "sil",
            )
        cands = bank.en_phones.get(ip, [])
        if cands:
            c = min(cands, key=lambda x: (0.0 if x.speaker == tgt_name else 4.0) + (0.0 if len(x.log_env) >= 5 else 2.0))
            m = max(1, len(c.log_env) // 2)
            sl = slice(m, None) if take_right else slice(0, m)
            if len(c.log_env[sl]) == 0:
                sl = slice(None)
            return (
                c.log_env[sl], c.bap[sl], c.bap_gt8k[sl], c.voiced[sl], c.f0_rel_st[sl], c.speaker, "phone"
            )
        if bank.fallback_unit_bank is not None:
            d = cw.lookup_ipa_template(bank.fallback_unit_bank, ip, is_onset=not take_right)
            le = (d["log_env"] - 3.1).astype(np.float32)
            ba = d["bap"].astype(np.float32)
            bg = np.repeat(ba[:, 4:5], 2, axis=1)
            vo = d["voiced"].astype(bool)
            f0 = np.zeros(len(le), dtype=np.float32)
            m = max(1, len(le) // 2)
            sl = slice(m, None) if take_right else slice(0, m)
            if len(le[sl]) == 0:
                sl = slice(None)
            return le[sl], ba[sl], bg[sl], vo[sl], f0[sl], "SSB0502", "synth"
        raise ValueError(f"Missing EN phone {ip} and fallback_unit_bank=None")

    e1, b1, bg1, v1, f1, spk1, src1 = _half_tok(p1, take_right=True)
    e2, b2, bg2, v2, f2, spk2, src2 = _half_tok(p2, take_right=False)
    e1c, e2c = e1.copy(), e2.copy()
    b1c, b2c = b1.copy(), b2.copy()
    # Smooth 15 ms transition across missing diphone edge when both halves are voiced
    if np.any(v1) and np.any(v2) and len(e1c) >= 2 and len(e2c) >= 2:
        k = min(2, len(e1c), len(e2c))
        mid = 0.5 * (e1c[-1] + e2c[0])
        for i in range(k):
            w = (i + 1.0) / (k + 1.0)
            e1c[-k + i] = (1.0 - w) * e1c[-k + i] + w * mid
            e2c[i] = (1.0 - w) * mid + w * e2c[i]

    le = np.concatenate([e1c, e2c], axis=0)
    ba = np.concatenate([b1c, b2c], axis=0)
    bg = np.concatenate([bg1, bg2], axis=0)
    vo = np.concatenate([v1, v2], axis=0)
    f0 = np.concatenate([f1, f2], axis=0)
    el, er, dbl, dbr, fl, fr = _edge_stats(le, f0, vo)
    spk = spk1 if spk1 != tgt_name else spk2
    if src1 == "synth" or src2 == "synth":
        kind = "en_synth_fill"
    else:
        kind = "en_phone_fallback"
    return EnDiphoneToken(
        p1=p1, p2=p2, speaker=spk, utterance_id="__fallback__", seq_idx=-1, stress1=None, stress2=None,
        log_env=le, bap=ba, bap_gt8k=bg, voiced=vo, f0_rel_st=f0, f_mid_split=len(e1c),
        dur_ms=float(len(le) * HOP * 1000.0 / SR),
        edge_left_env=el, edge_right_env=er, edge_left_db=dbl, edge_right_db=dbr,
        edge_f0_l=fl, edge_f0_r=fr, raw_wav=None,
    ), kind


def _viterbi_diphones(
    slots: list[tuple[list[EnDiphoneToken], list[float]]],
    beam_k: int = 8,
) -> list[EnDiphoneToken]:
    """Viterbi diphone unit selection minimizing target cost + midpoint steady-state join cost."""
    if not slots:
        return []
    pruned: list[tuple[list[EnDiphoneToken], list[float]]] = []
    for cands, tcosts in slots:
        if len(cands) > beam_k:
            order = sorted(range(len(cands)), key=lambda i: tcosts[i])[:beam_k]
            pruned.append(([cands[i] for i in order], [tcosts[i] for i in order]))
        else:
            pruned.append((cands, tcosts))

    dp_prev = np.array(pruned[0][1], dtype=np.float64)
    bps: list[np.ndarray] = []
    for t in range(1, len(pruned)):
        c_prev = pruned[t - 1][0]
        c_cur, tc_cur = pruned[t]
        n_p, n_c = len(c_prev), len(c_cur)
        dp_cur = np.empty(n_c, dtype=np.float64)
        bp_t = np.zeros(n_c, dtype=np.int32)
        for j in range(n_c):
            cj = c_cur[j]
            best_v = float("inf")
            best_i = 0
            for i in range(n_p):
                ci = c_prev[i]
                if ci.utterance_id == cj.utterance_id and ci.utterance_id != "__fallback__" and cj.seq_idx == ci.seq_idx + 1:
                    # Free consecutive natural transition from the exact same donor recording!
                    jcost = -1.25
                else:
                    diff = ci.edge_right_env - cj.edge_left_env
                    spec_j = float(math.sqrt(float(np.mean(diff * diff))))
                    db_j = abs(ci.edge_right_db - cj.edge_left_db)
                    f0_j = abs(ci.edge_f0_r - cj.edge_f0_l)
                    jcost = 0.95 * spec_j + 0.18 * db_j + 0.10 * f0_j
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


def _viterbi_zh_syllables(
    slots: list[tuple[list[V11ZhToken], list[float], bool, bool]],
    beam_k: int = 8,
) -> list[V11ZhToken]:
    if not slots:
        return []
    pruned = []
    for cands, tcosts, pb, oe in slots:
        if len(cands) > beam_k:
            order = sorted(range(len(cands)), key=lambda i: tcosts[i])[:beam_k]
            pruned.append(([cands[i] for i in order], [tcosts[i] for i in order], pb, oe))
        else:
            pruned.append((cands, tcosts, pb, oe))
    dp_prev = np.array(pruned[0][1], dtype=np.float64)
    bps = []
    for t in range(1, len(pruned)):
        c_prev, obs_prev = pruned[t - 1][0], pruned[t - 1][3]
        c_cur, tc_cur, p_bef, obs_cur = pruned[t]
        w_join = 0.0 if (p_bef or obs_prev or obs_cur) else 0.70
        dp_cur = np.empty(len(c_cur), dtype=np.float64)
        bp_t = np.zeros(len(c_cur), dtype=np.int32)
        for j, cj in enumerate(c_cur):
            best_v, best_i = float("inf"), 0
            for i, ci in enumerate(c_prev):
                if w_join > 0.0:
                    diff = ci.edge_right_env - cj.edge_left_env
                    jcost = w_join * (float(math.sqrt(float(np.mean(diff * diff)))) + 0.12 * abs(ci.edge_f0_r - cj.edge_f0_l))
                else:
                    jcost = 0.0
                v = dp_prev[i] + jcost + tc_cur[j]
                if v < best_v:
                    best_v, best_i = v, i
            dp_cur[j], bp_t[j] = best_v, best_i
        bps.append(bp_t)
        dp_prev = dp_cur
    idx = int(np.argmin(dp_prev))
    chosen = [pruned[-1][0][idx]]
    for t in range(len(bps) - 1, -1, -1):
        idx = int(bps[t][idx])
        chosen.append(pruned[t][0][idx])
    chosen.reverse()
    return chosen


def render_v11_utterance(
    text_or_plan: str | dict,
    bank: V11DiphoneDualBank,
    target_profile: Any,
    *,
    cloner_fn: ClonerFn | None = None,
    plan_fn: Callable[[str], dict] | None = None,
    src_profiles: dict[str, Any] | None = None,
    vtln_alpha: float | None = None,
    use_unvoiced_residual: bool = True,
    ref_vowel_db: float = REF_VOWEL_PARSEVAL_DB,
    seed: int = 72,
    phase_dispersion: bool = True,
) -> tuple[np.ndarray, dict]:
    """Render ZH / EN / Code-Switched text using English Diphone Viterbi + Mandarin Whole-Syllable Rescue + Smooth HNM."""
    if isinstance(text_or_plan, str):
        plan_doc = plan_fn(text_or_plan) if plan_fn is not None else _default_plan(text_or_plan)
    else:
        plan_doc = text_or_plan

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

    # Breath-group segmentation + natural English content-vs-function word prosody
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
        base_d = max(60.0 if parsed[i][0] == "zh" else 45.0, float(u.get("duration_ms", 200.0)))
        w_low = str(u.get("text", "")).strip().lower().strip(".,!?;:\"'")
        is_func = (parsed[i][0] != "zh") and (w_low in EN_FUNCTION_WORDS)
        g0, g1 = group_spans[group_id[i]]
        g_len = max(1, g1 - g0)
        rel_pos = (i - g0) / max(1, g_len)

        if parsed[i][0] != "zh":
            if is_func:
                base_d *= 0.78
                gdb = -1.5
            else:
                base_d *= 1.10
                gdb = 0.55
        else:
            gdb = 0.25 * (1.0 - rel_pos)

        if i == g1:
            base_d *= 1.16 if parsed[i][0] == "zh" else 1.18
            gdb -= 1.2
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

    # Walk spans of contiguous ZH units vs contiguous EN units
    u_idx = 0
    while u_idx < n_u:
        if parsed[u_idx][0] == "zh":
            j = u_idx
            while j < n_u and parsed[j][0] == "zh":
                j += 1
            slots: list[tuple[list[V11ZhToken], list[float], bool, bool]] = []
            slot_indices: list[int] = []
            selected_zh: dict[int, V11ZhToken | None] = {}
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
                        # Priority 1: Target whole syllable (0.0), Priority 2: Other same-corpus clean whole syllable (+3.5)
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
                gain = float(u.get("gain", 0.85))
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
                    # Priority 3: Clean onset (>=20 ms) + full-tone final concat (penalize tone=5 schwas!)
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

                raw_g = unit_gain_db[idx] + 20.0 * math.log10(max(0.55, min(1.35, gain / 0.85)))
                e_st = e_st + float(np.clip(raw_g, -2.5, 2.5)) * math.log(10.0) / 20.0
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
        else:
            # Contiguous English span: Diphone Viterbi path over ("SIL", p_0, ..., p_{M-1}, "SIL")!
            j = u_idx
            while j < n_u and parsed[j][0] != "zh":
                j += 1

            en_phones_flat: list[tuple[int, str, int | None, bool, float, float, float]] = []
            # Per-word metadata + perceptual frame budget
            word_phone_ranges: dict[int, tuple[int, int]] = {}
            for idx in range(u_idx, j):
                u = units[idx]
                ipas = list(u.get("ipa_phones") or ["ə"])
                arpa = u.get("arpabet") or []
                pairs = v6_arb._split_en_arpabet_to_ipa_with_stress(arpa) if arpa else [(p, None) for p in ipas]
                if not pairs:
                    pairs = [(p, None) for p in ipas]
                raw_d = [
                    (IPA_SPECS[ip].base_dur_ms if ip in IPA_SPECS else 70.0) * (1.18 if st == 1 else (0.82 if st == 0 else 1.0))
                    for ip, st in pairs
                ]
                nf_w = max(4, int(round(unit_dur_ms[idx] / 5.0)))
                alloc = compute_perceptual_phone_frames(pairs, raw_d, nf_w)
                w_start = len(en_phones_flat)
                for idx_p, ((ip, st), kp) in enumerate(zip(pairs, alloc)):
                    # Pitch accent boost (+1.3 st) on primary stressed vowel of content words
                    accent = (1.3 if (st == 1 and not unit_is_func[idx]) else 0.0)
                    en_phones_flat.append((idx, ip, st, unit_is_func[idx], float(kp), unit_decl_st[idx] + accent, unit_gain_db[idx]))
                word_phone_ranges[idx] = (w_start, len(en_phones_flat))

            M = len(en_phones_flat)
            nodes = [("SIL", None, 4.0)] + [(x[1], x[2], x[4]) for x in en_phones_flat] + [("SIL", None, 4.0)]
            diphone_slots: list[tuple[list[EnDiphoneToken], list[float]]] = []
            diphone_kinds: list[str] = []

            for k in range(M + 1):
                p1, st1, d1_fr = nodes[k]
                p2, st2, d2_fr = nodes[k + 1]
                tgt_dur_ms = 0.5 * (d1_fr + d2_fr) * 5.0
                cands = bank.en_diphones.get((p1, p2), [])
                if cands:
                    diphone_kinds.append("en_diphone_exact")
                    tcosts = []
                    for c in cands:
                        tc = 0.0 if c.speaker == tgt_name else 4.5
                        if st1 is not None and c.stress1 is not None and st1 != c.stress1:
                            tc += 0.6
                        if st2 is not None and c.stress2 is not None and st2 != c.stress2:
                            tc += 0.6
                        tc += 0.45 * abs(math.log(max(20.0, c.dur_ms) / max(20.0, tgt_dur_ms)))
                        tcosts.append(tc)
                    diphone_slots.append((cands, tcosts))
                else:
                    fb_tok, fb_kind = _make_pseudo_diphone(p1, p2, bank, tgt_name)
                    diphone_kinds.append(fb_kind)
                    diphone_slots.append(([fb_tok], [0.0]))

            chosen_diphones = _viterbi_diphones(diphone_slots)
            for dk in diphone_kinds:
                source_counts[dk] = source_counts.get(dk, 0) + 1

            # Render chosen diphones with left/right phone-half duration targets + steady-state midpoint blend
            diphone_start_fr: list[int] = []
            diphone_end_fr: list[int] = []
            for k, dtok in enumerate(chosen_diphones):
                p1, _, d1_fr = nodes[k]
                p2, _, d2_fr = nodes[k + 1]
                # Number of frames for each half inside this diphone
                f1_target = 2 if p1 == "SIL" else max(2, int(round(0.5 * d1_fr)))
                f2_target = 2 if p2 == "SIL" else max(2, int(round(0.5 * d2_fr)))
                tot_fr = max(4, f1_target + f2_target)

                e_ad, b_ad, bg_ad, v_ad = _adapt_v11(
                    dtok.log_env, dtok.bap, dtok.bap_gt8k, dtok.voiced, dtok.f0_rel_st,
                    _get_src_prof(bank, dtok.speaker, target_profile, src_profiles), target_profile, cloner_fn, vtln_alpha,
                )
                # Calibrate left half & right half separately to their respective phone manners!
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

                # Resample each half smoothly to f1_target / f2_target
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

                # Apply word prosody (gain + pitch accent + declination) per phone half
                if k > 0:
                    g1_db = en_phones_flat[k - 1][6]
                    p1_st = en_phones_flat[k - 1][5]
                    el_rs = el_rs + float(np.clip(g1_db, -2.5, 2.5)) * math.log(10.0) / 20.0
                else:
                    el_rs = np.full((f1_target, N_BINS), SILENCE_LOG_ENV, dtype=np.float64)
                    vl_rs[:] = False
                    p1_st = en_phones_flat[0][5]
                if k < M:
                    g2_db = en_phones_flat[k][6]
                    p2_st = en_phones_flat[k][5]
                    er_rs = er_rs + float(np.clip(g2_db, -2.5, 2.5)) * math.log(10.0) / 20.0
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
                # Steady-state join check: p1 (left edge of diphone k) determines if join with diphone k-1 is obstruent!
                is_obs_blk.append(p1 in OBSTRUENT_IPAS)
                is_pause_blk.append(False)
                cur_fr += len(e_dp)
                diphone_start_fr.append(fr_s_dp)
                diphone_end_fr.append(cur_fr)

            # Map word slot timings cleanly back from phone-midpoint diphones
            for idx in range(u_idx, j):
                w_p0, w_p1 = word_phone_ranges[idx]
                ms_s = diphone_start_fr[w_p0] * 5.0
                ms_e = diphone_end_fr[w_p1] * 5.0
                slot_timings_ms.append([ms_s, ms_e])
                unit_boundaries_ms.append(ms_e)
                boundary_frames.append(diphone_end_fr[w_p1])

            # Respect trailing pause after the EN span
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

    tail_fr = 6
    envs.append(np.full((tail_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((tail_fr, 5), dtype=np.float64))
    bgt8s.append(np.ones((tail_fr, 2), dtype=np.float64))
    vois.append(np.zeros(tail_fr, dtype=bool))
    sts.append(np.zeros(tail_fr, dtype=np.float64))
    is_obs_blk.append(False); is_pause_blk.append(True)

    # Steady-state midpoint seam blending: 15 ms (3 frames) on voiced midpoints, 5 ms on voiced obstruents, 0 ms on unvoiced!
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
    y = synthesize_v11(
        feats,
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
        "log_env": env_all.astype(np.float32),
    }
