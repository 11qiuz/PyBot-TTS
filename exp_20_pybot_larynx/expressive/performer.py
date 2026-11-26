"""Coupled prosodic + frame-level physical realization of Stage 8 emotional performances."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.fft import rfftfreq

from exp_20_pybot_larynx import HOP, N_BAP, N_BINS, N_FFT, SR
from exp_20_pybot_larynx.acoustics.knowledge import normalize_of
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import (
    DEFAULT_UV_LEVEL_BOOST_DB_S7,
    AcousticKnowledgeStage7,
    build_stage7_control,
)
from exp_20_pybot_larynx.arbiter import (
    OBSTRUENT_MANNERS,
    Arbiter,
    SyllableSpan,
    build_utterance_context,
    classify_onset_manner,
)
from exp_20_pybot_larynx.expressive.emotion_parser import (
    EMOTION_PROFILES,
    ExpressiveSegment,
    parse_expressive_text,
)
from exp_20_pybot_larynx.expressive.vocalizations import (
    build_breath_control,
    build_laugh_control,
    build_pause_control,
    build_sigh_control,
)
from exp_20_pybot_larynx.frontend.prosody_rules import (
    _CF0_VOICED_ST,
    _CF0_VOICELESS_ST,
    _F0_AR1_SIGMA,
    _F0_U5,
    _F0_U7,
    _catmull_rom_5pt,
    _catmull_rom_7pt,
)
from exp_20_pybot_larynx.frontend.prosody_stage7 import predict_prosody_stage7
from exp_20_pybot_larynx.larynx.synth import LarynxControl
from exp_20_pybot_larynx.larynx.synth_v2 import synthesize_larynx_v2
from exp_20_pybot_larynx.larynx.timbre_synth import TimbreSynthParams, apply_timbre_synth

_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_NEPER_TO_DB = 20.0 / math.log(10.0)
_DB_TO_NEPER = 1.0 / _NEPER_TO_DB
_BREATH_BAND_WEIGHTS = np.array([0.55, 0.85, 1.00, 1.00, 0.90, 0.80], dtype=np.float64)


def _map_focus_to_units(units: list[dict[str, Any]], seg: ExpressiveSegment) -> list[int]:
    """Map character-level FocusSpans onto linguistic unit indices."""
    n_u = len(units)
    levels = [0] * n_u
    if not seg.focus_spans or n_u == 0:
        return levels

    # Approximate character spans if `text_span` is missing
    cursor = 0
    for idx, u in enumerate(units):
        ts = u.get("text_span")
        if isinstance(ts, (list, tuple)) and len(ts) == 2:
            s0, s1 = int(ts[0]), int(ts[1])
        else:
            tok = str(u.get("hanzi") or u.get("token") or "")
            pos = seg.text.find(tok, cursor) if tok else -1
            if pos >= 0:
                s0, s1 = pos, pos + max(1, len(tok))
                cursor = s1
            else:
                s0, s1 = cursor, cursor + 1
                cursor = s1
        for fs in seg.focus_spans:
            if not (s1 <= fs.start_char or s0 >= fs.end_char):
                levels[idx] = max(levels[idx], int(fs.level))
    return levels


def apply_emotion_to_zh_prosody(
    pro: dict[str, Any],
    seg: ExpressiveSegment,
    *,
    stability: float = 0.55,
    seed: int = 123,
) -> tuple[dict[str, Any], list[int]]:
    """Transform Stage 7 predicted prosody according to emotion, focus, punctuation & rubato."""
    units = pro["units"]
    n_u = len(units)
    if n_u == 0:
        return pro, []

    focus_lvls = _map_focus_to_units(units, seg)
    prof = EMOTION_PROFILES.get(seg.emotion, EMOTION_PROFILES["neutral"])
    scale = max(0.0, min(1.5, float(seg.intensity)))
    stab = max(0.0, min(1.0, float(stability)))

    if (scale < 1e-4 or seg.emotion == "neutral") and not any(focus_lvls) and not seg.punct_style:
        return pro, focus_lvls

    # Pre-compute mean baseline F0 of this clause so range scaling pivots naturally around speaker/clause median
    f3_all = np.array([u["f0_3pt_st"] for u in units], dtype=np.float64)
    clause_f0_pivot = float(np.median(f3_all))

    rng = np.random.default_rng(int(seed) + len(seg.text) * 131 + 17)
    jitter_var = max(0.0, 1.0 - stab)

    f0_shift = float(prof["f0_shift_st"]) * scale
    f0_range = 1.0 + (float(prof["f0_range_scale"]) - 1.0) * scale
    f0_decl = float(prof["f0_declination_st"]) * scale
    speed_sc = max(0.70, min(1.38, 1.0 + (float(prof["speed_scale"]) - 1.0) * scale))
    pause_sc = max(0.65, min(1.60, 1.0 + (float(prof["pause_scale"]) - 1.0) * scale))
    gain_off = float(prof["gain_offset_db"]) * scale
    rubato = float(prof["rubato_strength"]) * scale

    first_focus = next((i for i, lv in enumerate(focus_lvls) if lv > 0), None)
    last_focus = max((i for i, lv in enumerate(focus_lvls) if lv > 0), default=-1)
    u7_arr = np.asarray(_F0_U7, dtype=np.float64)
    u5_arr = np.asarray(_F0_U5, dtype=np.float64)

    for i, u in enumerate(units):
        pos_u = float(i) / float(max(1, n_u - 1)) if n_u > 1 else 0.5
        fl = focus_lvls[i]
        is_post_focus = (first_focus is not None and i > last_focus and fl == 0)

        # 1. Pitch transformation: pivot + shift + range + clause declination + punctuation + focus/PFC
        pts7 = np.asarray(u.get("f0_7pt_st", np.interp(u7_arr, u5_arr, u["f0_5pt_st"])), dtype=np.float64)
        p_shift_u = f0_shift + f0_decl * (pos_u - 0.35)
        p_scale_u = f0_range
        g_add = gain_off
        dur_mul = 1.0 / speed_sc

        # Punctuation terminal phrasing contours
        if i >= max(0, n_u - 2):
            tail_w = 1.0 if i == n_u - 1 else 0.45
            if seg.punct_style == "?!":
                p_shift_u += 3.8 * tail_w * max(0.65, scale)
                p_scale_u *= 1.15
                g_add += 1.2 * tail_w
            elif seg.punct_style == "?":
                p_shift_u += 2.6 * tail_w * max(0.65, scale)
            elif seg.punct_style == "!":
                p_shift_u += 0.9 * tail_w * max(0.50, scale)
                p_scale_u *= 1.10
            elif seg.punct_style == "...":
                p_shift_u -= 1.6 * tail_w
                dur_mul *= 1.0 + 0.34 * tail_w
                g_add -= 1.4 * tail_w
            elif seg.punct_style == "~":
                # Playful scoop up on phrase-final vowel
                if i == n_u - 1:
                    pts7 = pts7 + np.linspace(-0.4, 2.4, len(pts7))
                dur_mul *= 1.0 + 0.30 * tail_w

        # Narrow focus boost & post-focus compression (PFC)
        if fl >= 2:
            p_shift_u += 3.2
            p_scale_u *= 1.22
            dur_mul *= 1.22
            g_add += 2.4
        elif fl == 1:
            p_shift_u += 2.1
            p_scale_u *= 1.12
            dur_mul *= 1.14
            g_add += 1.6
        elif is_post_focus:
            p_scale_u *= 0.72
            p_shift_u -= 0.7
            g_add -= 0.9

        # Rhythmic rubato (expressive contrast between content vs weak function words + final stretch)
        is_func = bool(u.get("is_func_word", False))
        if rubato > 1e-4:
            if is_func:
                dur_mul *= 1.0 - 0.14 * rubato
            elif i == n_u - 1:
                dur_mul *= 1.0 + 0.18 * rubato
        if jitter_var > 0.05 and scale > 0.25:
            dur_mul *= 1.0 + float(rng.normal(0.0, 0.045 * jitter_var))
            p_shift_u += float(rng.normal(0.0, 0.28 * jitter_var))

        pts7_new = np.clip(clause_f0_pivot + p_shift_u + p_scale_u * (pts7 - clause_f0_pivot), -14.0, 14.0)
        pts5_new = _catmull_rom_7pt(u5_arr, pts7_new)
        u["f0_7pt_st"] = [round(float(x), 3) for x in pts7_new]
        u["f0_5pt_st"] = [round(float(x), 3) for x in pts5_new]
        u["pitch_st"] = list(u["f0_5pt_st"])
        u["f0_3pt_st"] = [round(float(pts7_new[1]), 3), round(float(pts7_new[3]), 3), round(float(pts7_new[5]), 3)]

        # 2. Duration & onset duration & gain
        old_dur = max(60.0, float(u.get("duration_ms", 180.0)))
        new_dur = float(np.clip(old_dur * dur_mul, 55.0, 440.0))
        u["duration_ms"] = round(new_dur, 1)

        if str(u.get("onset", "")):
            old_on = max(12.0, float(u.get("onset_ms", 45.0)))
            on_mul = 0.5 * (1.0 + dur_mul)
            u["onset_ms"] = round(float(np.clip(old_on * on_mul, 12.0, min(200.0, new_dur - 20.0))), 1)

        u["gain_db"] = round(float(np.clip(float(u.get("gain_db", 0.0)) + g_add, -4.8, 4.6)), 3)

        raw_p = float(u.get("pause_after_ms", 0.0))
        if raw_p > 0.0:
            u["pause_after_ms"] = round(float(np.clip(raw_p * pause_sc, 0.0, 900.0)), 1)

    return pro, focus_lvls


def apply_emotion_to_control(
    ctrl: LarynxControl,
    seg: ExpressiveSegment,
    *,
    focus_frame_mask: np.ndarray | None = None,
    stability: float = 0.55,
    seed: int = 123,
) -> tuple[LarynxControl, float]:
    """Apply glottal, breathiness, tilt, tremor, articulatory EQ, and fry physics for an emotion."""
    out = ctrl.copy()
    n_fr = len(out.f0_hz)
    if n_fr == 0:
        return out, 0.0

    prof = EMOTION_PROFILES.get(seg.emotion, EMOTION_PROFILES["neutral"])
    scale = max(0.0, min(1.5, float(seg.intensity)))
    stab = max(0.0, min(1.0, float(stability)))

    if (scale < 1e-4 or seg.emotion == "neutral") and (focus_frame_mask is None or not np.any(focus_frame_mask)):
        return out, 0.0

    v_mask = out.voiced & (out.f0_hz > 40.0)
    if out.rd is None:
        out.rd = np.ones(n_fr, dtype=np.float64)
    if out.glottal_tilt_db_per_oct is None:
        out.glottal_tilt_db_per_oct = np.zeros(n_fr, dtype=np.float64)
    if out.jitter_frac is None:
        out.jitter_frac = np.full(n_fr, 0.0055, dtype=np.float64)
    if out.shimmer_frac is None:
        out.shimmer_frac = np.full(n_fr, 0.025, dtype=np.float64)

    # 1. Glottal rd, BAP, tilt, jitter & shimmer
    rd_off = float(prof["rd_offset"]) * scale
    if abs(rd_off) > 1e-4:
        out.rd = np.clip(out.rd + rd_off, 0.50, 2.20)

    bap_off = float(prof["bap_offset"]) * scale
    if abs(bap_off) > 1e-4 and np.any(v_mask):
        out.bap[v_mask] = np.clip(out.bap[v_mask] + bap_off * _BREATH_BAND_WEIGHTS[None, :], 0.004, 0.995)

    tilt_off = float(prof["tilt_db_oct"]) * scale
    if abs(tilt_off) > 1e-4:
        out.glottal_tilt_db_per_oct = np.clip(out.glottal_tilt_db_per_oct + tilt_off, -7.5, 7.5)

    out.jitter_frac = np.clip(out.jitter_frac + float(prof["jitter_add"]) * scale, 0.0, 0.035)
    out.shimmer_frac = np.clip(out.shimmer_frac + float(prof["shimmer_add"]) * scale, 0.0, 0.14)

    # 2. Laryngeal emotional tremor (fearful/sad)
    trem = float(prof["tremor_depth_st"]) * scale
    if trem > 1e-4 and np.any(v_mask):
        rng = np.random.default_rng(int(seed) + 911)
        phi0 = float(rng.uniform(0.0, math.tau))
        t_fr = np.arange(n_fr, dtype=np.float64) * (HOP / float(SR))
        rate = float(prof["tremor_rate_hz"])
        vib = trem * np.sin(math.tau * rate * t_fr + phi0)
        out.f0_hz[v_mask] = np.clip(out.f0_hz[v_mask] * (2.0 ** (vib[v_mask] / 12.0)), 55.0, 520.0)

    # 3. Focus frame boost (tighter vocal effort & subtle presence lift on focused words)
    if focus_frame_mask is not None and len(focus_frame_mask) == n_fr and np.any(focus_frame_mask):
        fm = np.asarray(focus_frame_mask, dtype=bool) & v_mask
        if np.any(fm):
            out.rd[fm] = np.clip(out.rd[fm] - 0.14, 0.50, 2.20)
            out.glottal_tilt_db_per_oct[fm] = np.clip(out.glottal_tilt_db_per_oct[fm] + 0.9, -7.5, 7.5)

    # 4. Expressive articulatory spectral timbre via Stage 8 synthesizer primitives
    emo_tp = TimbreSynthParams(
        patch_id=f"emo_{seg.emotion}",
        chest_db=float(prof["chest_db"]) * scale,
        warmth_db=float(prof["warmth_db"]) * scale,
        presence_db=float(prof["presence_db"]) * scale,
        air_db=float(prof["air_db"]) * scale,
        formant_q=float(prof["formant_q"]) * scale,
        vowel_openness=float(prof["vowel_openness"]) * scale,
    )
    if not emo_tp.is_neutral():
        out, _ = apply_timbre_synth(out, emo_tp, seed=seed)

    # 5. Trailing breathy release on ellipsis / sad / gentle
    if (seg.punct_style == "..." or seg.emotion in ("sad", "gentle")) and scale >= 0.40 and n_fr >= 12:
        k_tail = min(max(6, n_fr // 5), 24)
        fade = np.linspace(0.0, 1.0, k_tail)
        out.rd[-k_tail:] = np.clip(out.rd[-k_tail:] + 0.45 * fade * min(1.0, scale), 0.50, 2.20)
        v_tail = v_mask[-k_tail:]
        if np.any(v_tail):
            out.bap[-k_tail:][v_tail] = np.clip(
                out.bap[-k_tail:][v_tail] + (0.22 * fade[v_tail, None] * min(1.0, scale)),
                0.004,
                0.995,
            )

    subh = float(prof["subharm_base"]) * min(1.2, scale)
    if stab < 0.60 and seg.emotion in ("sad", "gentle", "whisper", "serious"):
        subh += 0.05 * (0.60 - stab) / 0.60
    subh = float(np.clip(subh, 0.0, 0.24))
    return out, subh


def render_expressive_utterance(
    text: str,
    utt_id: str,
    knowledge: AcousticKnowledgeStage7,
    *,
    arbiter: Arbiter,
    rules: list[Any],
    emotion_mode: str = "auto",
    emotion_intensity: float = 0.75,
    stability: float = 0.55,
    auto_breath: bool = True,
    synth_params: TimbreSynthParams | None = None,
    glottal_mode: str = "dynamic_coherent",
    coherent_phasor: np.ndarray | None = None,
    seed: int = 123,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Standalone zero-ML Stage 8 expressive Mandarin synthesis pipeline."""
    segs = parse_expressive_text(
        text,
        emotion_mode=emotion_mode,
        emotion_intensity=emotion_intensity,
        stability=stability,
        auto_breath=auto_breath,
    )
    if not segs:
        raise ValueError(f"Empty expressive text: {text!r}")

    base_f0 = float(knowledge.speaker_f0_med_hz)
    ctrl_parts: list[LarynxControl] = []
    subharm_max = 0.0
    detected_emotions: list[str] = []
    all_traces: list[dict[str, Any]] = []

    for s_idx, seg in enumerate(segs):
        if seg.kind == "pause":
            ctrl_parts.append(build_pause_control(seg.duration_ms))
            continue
        if seg.kind == "breath":
            ctrl_parts.append(build_breath_control(base_f0, duration_ms=seg.duration_ms, intensity=seg.intensity, is_gasp=False))
            continue
        if seg.kind == "gasp":
            ctrl_parts.append(build_breath_control(base_f0, duration_ms=seg.duration_ms, intensity=seg.intensity, is_gasp=True))
            continue
        if seg.kind == "sigh":
            ctrl_parts.append(build_sigh_control(base_f0, duration_ms=seg.duration_ms, intensity=seg.intensity))
            continue
        if seg.kind == "laugh":
            ctrl_parts.append(build_laugh_control(base_f0, duration_ms=seg.duration_ms, intensity=seg.intensity, soft=False))
            continue
        if seg.kind == "chuckle":
            ctrl_parts.append(build_laugh_control(base_f0, duration_ms=seg.duration_ms, intensity=seg.intensity, soft=True))
            continue

        if seg.auto_breath_before:
            ctrl_parts.append(build_breath_control(base_f0, duration_ms=175.0, intensity=seg.intensity))

        detected_emotions.append(seg.emotion)
        pro = predict_prosody_stage7(seg.text, knowledge, style="auto")
        pro, focus_lvls = apply_emotion_to_zh_prosody(pro, seg, stability=stability, seed=seed + s_idx)
        raw_units = pro["units"]
        n_u = len(raw_units)

        spans: list[SyllableSpan] = []
        f0_chunks: list[np.ndarray] = []
        voi_chunks: list[np.ndarray] = []
        focus_chunks: list[np.ndarray] = []
        syl_gains: list[float] = []
        cur_fr = 0
        prev_v_end: float | None = None

        for i, u in enumerate(raw_units):
            o, f = normalize_of(str(u.get("onset", "")), str(u.get("final", "")))
            tone = int(u.get("tone") or 1)
            dur_ms = max(55.0, float(u.get("duration_ms", 180.0)))
            n_syl = max(3, int(round(dur_ms / 5.0)))
            n_on = max(1, min(n_syl - 2, int(round(float(u.get("onset_ms", 45.0)) / 5.0)))) if o else 0
            split_fr = cur_fr + n_on
            end_fr = cur_fr + n_syl

            manner = classify_onset_manner(o) if o else "zero"
            n_fi = max(1, n_syl - n_on)
            u_fi = np.linspace(0.0, 1.0, n_fi) if n_fi > 1 else np.array([0.5], dtype=np.float64)
            fi_st = _catmull_rom_7pt(u_fi, np.asarray(u["f0_7pt_st"], dtype=np.float64)) if "f0_7pt_st" in u else _catmull_rom_5pt(u_fi, np.asarray(u["f0_5pt_st"], dtype=np.float64))

            if o and n_fi >= 3:
                k_cf0 = min(5, n_fi)
                decay = np.exp(-np.arange(k_cf0, dtype=np.float64) * 5.0 / 12.0)
                if manner in ("stop_asp", "affr_asp", "fric"):
                    fi_st[:k_cf0] += _CF0_VOICELESS_ST * decay
                elif manner in ("nasal", "liquid"):
                    fi_st[:k_cf0] += _CF0_VOICED_ST * decay
                elif manner in ("stop_unasp", "affr_unasp"):
                    fi_st[:k_cf0] += 0.25 * _CF0_VOICELESS_ST * decay

            fi_hz = np.clip(base_f0 * (2.0 ** (fi_st / 12.0)), 60.0, 480.0)
            f0_seg = np.empty(n_syl, dtype=np.float64)
            f0_seg[n_on:] = fi_hz
            if n_on > 0:
                if manner in ("nasal", "liquid") and prev_v_end is not None:
                    f0_seg[:n_on] = np.linspace(0.65 * prev_v_end + 0.35 * fi_hz[0], fi_hz[0], n_on, endpoint=False)
                else:
                    f0_seg[:n_on] = fi_hz[0]

            voi_seg = np.ones(n_syl, dtype=bool)
            if o and manner in OBSTRUENT_MANNERS and n_on > 0:
                uv_f = float(u.get("onset_uv_frac", 0.78))
                uv_n = max(1, min(n_on, int(round(uv_f * n_on))))
                voi_seg[:uv_n] = False
                f0_seg[:uv_n] = 0.0

            brk = int(u.get("prosodic_break", 0))
            t_uv = 1 if brk == 2 else (2 if brk == 3 else (4 if brk == 4 else 0))
            if t_uv > 0 and n_fi >= 8:
                t_uv = min(t_uv, n_fi - 6)
                voi_seg[-t_uv:] = False
                f0_seg[-t_uv:] = 0.0

            p_ms = float(u.get("pause_after_ms", 0.0)) if i < n_u - 1 else float(seg.pause_after_ms)
            spans.append(SyllableSpan(i, str(u.get("hanzi") or ""), o, f, tone, cur_fr, split_fr, end_fr, p_ms, brk))
            f0_chunks.append(f0_seg)
            voi_chunks.append(voi_seg)
            focus_chunks.append(np.full(n_syl, focus_lvls[i] > 0, dtype=bool))
            syl_gains.append(float(u.get("gain_db", 0.0)))
            cur_fr = end_fr

            if i < n_u - 1 and p_ms >= 20.0:
                n_p = int(round(p_ms / 5.0))
                if n_p > 0:
                    f0_chunks.append(np.zeros(n_p, dtype=np.float64))
                    voi_chunks.append(np.zeros(n_p, dtype=bool))
                    focus_chunks.append(np.zeros(n_p, dtype=bool))
                    cur_fr += n_p
                prev_v_end = None
            else:
                prev_v_end = float(f0_seg[-1]) if voi_seg[-1] and f0_seg[-1] > 50.0 else None

        f0_all = np.concatenate(f0_chunks)
        voi_all = np.concatenate(voi_chunks)
        foc_all = np.concatenate(focus_chunks)
        ctx = build_utterance_context(f"{utt_id}_s{s_idx}", cur_fr, spans, speaker_f0_med_hz=base_f0)
        base_ctrl, _ = build_stage7_control(ctx, f0_all, voi_all, knowledge, syl_gain_db=np.asarray(syl_gains, dtype=np.float64))
        fused_ctrl, arb_tr = arbiter.apply(ctx, base_ctrl, rules)
        all_traces.append(arb_tr)
        emo_ctrl, subh = apply_emotion_to_control(fused_ctrl, seg, focus_frame_mask=foc_all, stability=stability, seed=seed + s_idx)
        subharm_max = max(subharm_max, subh)
        ctrl_parts.append(emo_ctrl)

        if seg.pause_after_ms >= 20.0 and s_idx < len(segs) - 1:
            ctrl_parts.append(build_pause_control(seg.pause_after_ms))

    # Concatenate parts
    if len(ctrl_parts) == 1:
        full_ctrl = ctrl_parts[0]
    else:
        f0_c = np.concatenate([c.f0_hz for c in ctrl_parts])
        v_c = np.concatenate([c.voiced for c in ctrl_parts])
        e_c = np.concatenate([c.log_env for c in ctrl_parts], axis=0)
        b_c = np.concatenate([c.bap for c in ctrl_parts], axis=0)
        t_c = np.concatenate([c.glottal_tilt_db_per_oct if c.glottal_tilt_db_per_oct is not None else np.zeros(len(c.f0_hz)) for c in ctrl_parts])
        j_c = np.concatenate([c.jitter_frac if c.jitter_frac is not None else np.full(len(c.f0_hz), 0.0055) for c in ctrl_parts])
        s_c = np.concatenate([c.shimmer_frac if c.shimmer_frac is not None else np.full(len(c.f0_hz), 0.025) for c in ctrl_parts])
        r_c = np.concatenate([c.rd if c.rd is not None else np.ones(len(c.f0_hz)) for c in ctrl_parts])
        full_ctrl = LarynxControl(
            f0_hz=f0_c, voiced=v_c, log_env=e_c, bap=b_c,
            glottal_tilt_db_per_oct=t_c, jitter_frac=j_c, shimmer_frac=s_c,
            n_samples=len(f0_c) * HOP, rd=r_c,
        )

    if synth_params is not None and not synth_params.is_neutral():
        full_ctrl, s_info = apply_timbre_synth(full_ctrl, synth_params, seed=seed)
        subharm_max = max(subharm_max, float(s_info.get("subharm_strength", 0.0)))

    wav = synthesize_larynx_v2(
        full_ctrl,
        glottal_mode=glottal_mode,
        coherent_phasor=coherent_phasor,
        subharm_strength=subharm_max,
        uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7,
        seed=seed,
    )
    return wav, {
        "ctrl": full_ctrl,
        "detected_emotions": detected_emotions or ["neutral"],
        "subharm_strength": subharm_max,
        "arbiter_traces": all_traces,
        "segments": segs,
    }
