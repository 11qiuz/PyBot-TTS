"""Unified ZH + EN cross-lingual trajectory bridge for PyBot-Larynx (`exp_20`)."""
from __future__ import annotations

import math
import re
from typing import Any
import zlib

import numpy as np
from scipy.fft import rfftfreq
from scipy.signal import lfilter

from exp_20_pybot_larynx import HOP, N_BAP, N_BINS, N_FFT, SR
from exp_20_pybot_larynx.acoustics.knowledge import normalize_of
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import (
    AcousticKnowledgeStage2,
    build_stage2_control,
)
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import (
    AcousticKnowledgeStage5,
    build_stage5_control,
)
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import (
    AcousticKnowledgeStage7,
    build_stage7_control,
)
from exp_20_pybot_larynx.arbiter import (
    NEPER_TO_DB,
    OBSTRUENT_MANNERS,
    Arbiter,
    SyllableSpan,
    UtteranceContext,
    build_utterance_context,
    classify_onset_manner,
)
from exp_20_pybot_larynx.expressive import (
    EMOTION_PROFILES,
    ExpressiveSegment,
    apply_emotion_to_control,
    apply_emotion_to_zh_prosody,
    build_breath_control,
    build_laugh_control,
    build_pause_control,
    build_sigh_control,
    parse_expressive_text,
)
from exp_20_pybot_larynx.crossling import (
    build_speaker_anchored_en_control,
    classify_articulatory_class,
    get_speaker_donor_bank,
)
from exp_20_pybot_larynx.crossling.en_stage10_renderer import (
    build_stage10_en_control,
    is_stage10_en_ready,
    load_sf_bilingual_consistency,
)
from exp_20_pybot_larynx.frontend.prosody_rules import (
    _CF0_VOICED_ST,
    _CF0_VOICELESS_ST,
    _F0_AR1_SIGMA,
    _catmull_rom_5pt,
    _catmull_rom_7pt,
    _safe_plan_v13,
    predict_prosody_rules,
)
from exp_20_pybot_larynx.frontend.prosody_stage5 import predict_prosody_stage5
from exp_20_pybot_larynx.frontend.prosody_stage7 import predict_prosody_stage7
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env, log_env_to_cep
from exp_v6_crossling_dsp.arbiter import build_frame_track
from exp_v6_crossling_dsp.hnm_synth import _fricative_db, _resonator_db
from exp_v6_crossling_dsp.speaker_profile import EQ_BANDS_HZ, SpeakerProfile, neutral_female_reference
from exp_v5_arch import crossling_frontend as cf
from pybot_webui.presets import VoiceParams
from pybot_webui.splitter import normalize_text_for_larynx

_DB_TO_NEPER = 1.0 / NEPER_TO_DB
_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_FREQS_SAFE = np.maximum(_FREQS, 80.0)
_LOG2_FREQS = np.log2(_FREQS_SAFE)
_LOG2_EQ = np.log2(np.asarray(EQ_BANDS_HZ, dtype=np.float64))

# Calibrated offsets aligning exp_v6 IPA formant/fricative envelopes to Stage 4 BZNSYP log_env scale.
_EN_VOICED_OFFSET_DB = -24.5
_EN_UNVOICED_OFFSET_DB = -19.5

_STAGE4_EN_VOICED_BAP = np.array([0.045, 0.22, 0.42, 0.68, 0.84, 0.90], dtype=np.float64)
_STAGE4_VOICED_FRIC_BAP = np.array([0.16, 0.40, 0.64, 0.84, 0.92, 0.96], dtype=np.float64)

_PUNCT_AFTER_RE = re.compile(r"^[，,、。\.！!？\?；;：:…—\"'\u201d\u2019\u300d\u300f\uff09\u3011\]) ]+")


def _silence_control(n_frames: int) -> LarynxControl:
    """Create n_frames of clean silence LarynxControl."""
    n_fr = max(0, int(n_frames))
    return LarynxControl(
        f0_hz=np.zeros(n_fr, dtype=np.float64),
        voiced=np.zeros(n_fr, dtype=bool),
        log_env=np.full((n_fr, N_BINS), -16.0, dtype=np.float64),
        bap=np.full((n_fr, N_BAP), 0.99, dtype=np.float64),
        glottal_tilt_db_per_oct=np.zeros(n_fr, dtype=np.float64),
        jitter_frac=np.full(n_fr, 0.0055, dtype=np.float64),
        shimmer_frac=np.full(n_fr, 0.025, dtype=np.float64),
        n_samples=n_fr * HOP,
        rd=np.ones(n_fr, dtype=np.float64),
    )


def _vocal_tract_db_stage4(
    freqs_hz: np.ndarray,
    formants: np.ndarray,
    bandwidths: np.ndarray,
    zero_f: float,
    zero_bw: float,
    zero_depth: float,
    profile: SpeakerProfile,
) -> np.ndarray:
    """Pole-zero vocal tract transfer tuned for Stage 4 full-band 24 kHz spectral tilt."""
    f1, f2, f3 = float(formants[0]), float(formants[1]), float(formants[2])
    bw1, bw2, bw3 = float(bandwidths[0]), float(bandwidths[1]), float(bandwidths[2])
    f4 = 3850.0 * profile.vtln_scale

    env = (
        _resonator_db(freqs_hz, f1, max(95.0, bw1))
        + _resonator_db(freqs_hz, f2, max(135.0, bw2))
        + _resonator_db(freqs_hz, f3, max(200.0, bw3))
        + 0.55 * _resonator_db(freqs_hz, f4, 320.0)
    )
    log2f = np.log2(np.maximum(freqs_hz, 80.0))
    # Gentler higher-pole compensation above 1.4 kHz matches BZNSYP 4-8 kHz brightness without steep cliff.
    fant_comp = 8.4 * np.log2(1.0 + freqs_hz / 1400.0)

    if zero_f > 0.0 and zero_depth > 0.0:
        env = env - zero_depth / (1.0 + ((freqs_hz - zero_f) / max(75.0, zero_bw)) ** 2)

    tilt_term = profile.ltas_tilt_db_per_oct * (log2f - math.log2(500.0))
    eq_term = 0.5 * np.interp(log2f, _LOG2_EQ, np.asarray(profile.residual_eq_db, dtype=np.float64))
    return np.clip(env + fant_comp + tilt_term + eq_term, -56.0, 28.0)


def build_zh_run_control(
    zh_text: str,
    utt_id: str,
    kb: AcousticKnowledgeStage2 | AcousticKnowledgeStage5 | AcousticKnowledgeStage7,
    arbiter: Arbiter,
    rules: list[Any],
    params: VoiceParams,
    *,
    trailing_pause: bool = False,
    expressive_seg: ExpressiveSegment | None = None,
    continues_after: bool = False,
) -> tuple[LarynxControl, dict[str, Any]]:
    """Build a Stage 7 (or Stage 5 / Stage 4 fallback) expressive Mandarin LarynxControl."""
    use_s7 = isinstance(kb, AcousticKnowledgeStage7)
    use_s5 = isinstance(kb, AcousticKnowledgeStage5)
    if use_s7:
        pro = predict_prosody_stage7(zh_text, kb, style="auto", continues_after=continues_after)
    elif use_s5:
        pro = predict_prosody_stage5(zh_text, kb, style="auto", continues_after=continues_after)
    else:
        pro = predict_prosody_rules(zh_text, kb, style="auto", continues_after=continues_after)

    focus_lvls = [0] * len(pro["units"])
    if expressive_seg is not None:
        pro, focus_lvls = apply_emotion_to_zh_prosody(
            pro,
            expressive_seg,
            stability=params.stability,
            seed=params.seed,
        )

    raw_units = pro["units"]
    base_f0 = float(pro["speaker_f0_med_hz"])
    n_u = len(raw_units)
    speed = max(0.60, min(1.60, float(params.speed)))
    pause_scale = max(0.40, min(2.00, float(params.pause_scale)))

    spans: list[SyllableSpan] = []
    f0_chunks: list[np.ndarray] = []
    voi_chunks: list[np.ndarray] = []
    focus_chunks: list[np.ndarray] = []
    syl_gains: list[float] = []

    cur_fr = 0
    prev_voiced_end_hz: float | None = None

    for i, u in enumerate(raw_units):
        o, f = normalize_of(str(u.get("onset", "")), str(u.get("final", "")))
        tone = int(u.get("tone") or 1)
        dur_ms = max(60.0, float(u.get("duration_ms", 180.0)) / speed)
        n_syl = max(3, int(round(dur_ms / 5.0)))

        if o:
            raw_on_ms = max(15.0, float(u.get("onset_ms", 50.0)) / speed)
            n_on = max(1, min(n_syl - 2, int(round(raw_on_ms / 5.0))))
        else:
            n_on = 0
        split_fr = cur_fr + n_on
        end_fr = cur_fr + n_syl

        manner = classify_onset_manner(o) if o else "zero"
        n_fi = max(1, n_syl - n_on)
        u_fi = np.linspace(0.0, 1.0, n_fi) if n_fi > 1 else np.array([0.5], dtype=np.float64)
        if "f0_7pt_st" in u:
            fi_st = _catmull_rom_7pt(u_fi, np.asarray(u["f0_7pt_st"], dtype=np.float64))
        else:
            fi_st = _catmull_rom_5pt(u_fi, np.asarray(u["f0_5pt_st"], dtype=np.float64))

        if o and n_fi >= 3:
            k_cf0 = min(5, n_fi)
            decay = np.exp(-np.arange(k_cf0, dtype=np.float64) * 5.0 / 12.0)
            if manner in ("stop_asp", "affr_asp", "fric"):
                fi_st[:k_cf0] += _CF0_VOICELESS_ST * decay
            elif manner in ("nasal", "liquid"):
                fi_st[:k_cf0] += _CF0_VOICED_ST * decay
            elif manner in ("stop_unasp", "affr_unasp"):
                fi_st[:k_cf0] += 0.25 * _CF0_VOICELESS_ST * decay

        fi_hz = np.clip(base_f0 * (2.0 ** (fi_st / 12.0)), 65.0, 440.0)

        f0_seg = np.empty(n_syl, dtype=np.float64)
        f0_seg[n_on:] = fi_hz
        if n_on > 0:
            if manner in ("nasal", "liquid") and prev_voiced_end_hz is not None:
                f0_seg[:n_on] = np.linspace(
                    0.65 * prev_voiced_end_hz + 0.35 * fi_hz[0],
                    fi_hz[0],
                    n_on,
                    endpoint=False,
                )
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

        if prev_voiced_end_hz is not None:
            if voi_seg[0] and n_on == 0 and f0_seg[0] > 50.0:
                k_sm = min(4, n_syl)
                w = np.linspace(0.30, 0.0, k_sm)
                f0_seg[:k_sm] = (1.0 - w) * f0_seg[:k_sm] + w * prev_voiced_end_hz
            elif n_on < n_syl and manner in OBSTRUENT_MANNERS:
                k_sm = min(3, n_syl - n_on)
                w = np.linspace(0.10, 0.0, k_sm)
                f0_seg[n_on : n_on + k_sm] = (1.0 - w) * f0_seg[n_on : n_on + k_sm] + w * prev_voiced_end_hz

        raw_p = float(u.get("pause_after_ms", 0.0))
        p_ms = (raw_p * pause_scale) if (i < n_u - 1 or trailing_pause) else 0.0
        spans.append(
            SyllableSpan(
                index=i,
                hanzi=str(u.get("hanzi") or u.get("token") or ""),
                onset=o,
                final=f,
                tone=tone,
                start_fr=cur_fr,
                split_fr=split_fr,
                end_fr=end_fr,
                pause_after_ms=p_ms,
                prosodic_break=brk,
            )
        )
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
            prev_voiced_end_hz = None
        else:
            prev_voiced_end_hz = float(f0_seg[-1]) if voi_seg[-1] and f0_seg[-1] > 50.0 else None

    f0_all = np.concatenate(f0_chunks)
    voi_all = np.concatenate(voi_chunks)
    focus_all = np.concatenate(focus_chunks)

    if _F0_AR1_SIGMA > 0.0 and len(f0_all) > 4:
        rng_u = np.random.default_rng(zlib.crc32(f"{utt_id}|{int(params.seed)}".encode("utf-8")) & 0xFFFFFFFF)
        rho = 0.72
        eps = rng_u.normal(0.0, _F0_AR1_SIGMA * np.sqrt(1.0 - rho * rho), size=len(f0_all))
        ar1 = lfilter((1.0,), (1.0, -rho), eps)
        vmask = voi_all & (f0_all > 50.0)
        f0_all[vmask] = np.clip(f0_all[vmask] * (2.0 ** (np.clip(ar1[vmask], -0.45, 0.45) / 12.0)), 65.0, 440.0)

    ctx: UtteranceContext = build_utterance_context(
        utt_id, cur_fr, spans, speaker_f0_med_hz=base_f0,
        continues_after=continues_after and spans[-1].prosodic_break < 2,
    )
    if use_s7:
        assert isinstance(kb, AcousticKnowledgeStage7)
        base_ctrl, kb_info = build_stage7_control(
            ctx,
            f0_all,
            voi_all,
            kb,
            syl_gain_db=np.asarray(syl_gains, dtype=np.float64),
        )
    elif use_s5:
        assert isinstance(kb, AcousticKnowledgeStage5)
        base_ctrl, kb_info = build_stage5_control(
            ctx,
            f0_all,
            voi_all,
            kb,
            mode="stage5",
            syl_gain_db=np.asarray(syl_gains, dtype=np.float64),
        )
    else:
        assert isinstance(kb, AcousticKnowledgeStage2)
        base_ctrl, kb_info = build_stage2_control(
            ctx,
            f0_all,
            voi_all,
            kb,
            mode="stage2",
            syl_gain_db=np.asarray(syl_gains, dtype=np.float64),
        )
    fused_ctrl, arb_trace = arbiter.apply(ctx, base_ctrl, rules)
    subharm_strength = 0.0
    if expressive_seg is not None:
        fused_ctrl, subharm_strength = apply_emotion_to_control(
            fused_ctrl,
            expressive_seg,
            focus_frame_mask=focus_all,
            stability=params.stability,
            seed=params.seed,
        )
    return fused_ctrl, {
        "kb_info": kb_info,
        "arbiter_trace": arb_trace,
        "prosody_plan": pro,
        "subharm_strength": subharm_strength,
    }


def build_en_run_control(
    en_units: list[dict[str, Any]],
    base_f0_hz: float,
    params: VoiceParams,
    *,
    expressive_seg: ExpressiveSegment | None = None,
    kb: AcousticKnowledgeStage2 | AcousticKnowledgeStage5 | AcousticKnowledgeStage7 | None = None,
    clause_pos: tuple[float, float] = (0.0, 1.0),
    is_clause_final: bool = True,
    is_question: bool = False,
    use_stage10_en: bool = True,
    expand_projected_pitch: bool = True,
    continues_after: bool = False,
    embedded_in_zh: bool = False,
    next_boundary_cls: int | None = None,
) -> tuple[LarynxControl, float]:
    """Convert contiguous English units into an expressive 513-bin LarynxControl."""
    if kb is not None:
        if use_stage10_en and is_stage10_en_ready():
            return build_stage10_en_control(
                en_units,
                base_f0_hz,
                params,
                zh_kb=kb,
                expressive_seg=expressive_seg,
                clause_pos=clause_pos,
                is_clause_final=is_clause_final,
                is_question=is_question,
                project_to_zh_speaker=True,
                expand_projected_pitch=expand_projected_pitch,
                continues_after=continues_after,
                embedded_in_zh=embedded_in_zh,
                next_boundary_cls=next_boundary_cls,
            )
        return build_speaker_anchored_en_control(
            en_units,
            base_f0_hz,
            params,
            kb,
            expressive_seg=expressive_seg,
            clause_pos=clause_pos,
            is_clause_final=is_clause_final,
            is_question=is_question,
        )

    base_prof = neutral_female_reference()
    f0_eff = float(base_f0_hz)
    range_sc = 1.0
    speed_eff = max(0.60, min(1.60, float(params.speed)))
    if expressive_seg is not None and expressive_seg.intensity > 1e-4:
        e_prof = EMOTION_PROFILES.get(expressive_seg.emotion, EMOTION_PROFILES["neutral"])
        sc = float(expressive_seg.intensity)
        f0_eff = float(np.clip(f0_eff * (2.0 ** ((e_prof["f0_shift_st"] * sc) / 12.0)), 65.0, 460.0))
        range_sc = 1.0 + (float(e_prof["f0_range_scale"]) - 1.0) * sc
        speed_eff = float(np.clip(speed_eff * (1.0 + (float(e_prof["speed_scale"]) - 1.0) * sc), 0.60, 1.60))

    prof = SpeakerProfile(
        name="EN_STAGE4_BRIDGE",
        f0_median_hz=f0_eff,
        f0_iqr_st=base_prof.f0_iqr_st * range_sc,
        f0_range_st=base_prof.f0_range_st * range_sc,
        ltas_tilt_db_per_oct=base_prof.ltas_tilt_db_per_oct,
        residual_eq_db=base_prof.residual_eq_db,
        hnr_db=14.0,
        vtln_scale=1.0,
        speech_rate_scale=speed_eff,
    )
    pause_scale = max(0.40, min(2.00, float(params.pause_scale)))
    copied_units: list[dict[str, Any]] = []
    for idx, u in enumerate(en_units):
        uc = dict(u)
        if idx < len(en_units) - 1:
            uc["pause_after_ms"] = float(u.get("pause_after_ms", 0.0)) * pause_scale
        else:
            uc["pause_after_ms"] = 0.0
        copied_units.append(uc)

    track = build_frame_track(
        {"units": copied_units},
        prof,
        lead_in_ms=5.0,
        lead_out_ms=5.0,
    )
    # Strip the 1-frame lead_in and 1-frame lead_out inserted by build_frame_track
    s_fr = 1
    e_fr = max(s_fr, track.n_frames - 1)
    n_fr = e_fr - s_fr
    if n_fr <= 0:
        return _silence_control(0), 0.0

    f0_hz = np.zeros(n_fr, dtype=np.float64)
    voiced = np.zeros(n_fr, dtype=bool)
    log_env = np.full((n_fr, N_BINS), -16.0, dtype=np.float64)
    bap = np.full((n_fr, N_BAP), 0.99, dtype=np.float64)

    for k in range(n_fr):
        ti = s_fr + k
        f0 = float(track.f0_hz[ti])
        vg = float(track.voiced_gain[ti])
        ng = float(track.noise_gain[ti])
        vbar = float(track.voicing_bar[ti])

        has_voiced = vg > 1e-6 and f0 > 40.0
        has_noise = ng > 1e-6

        if has_voiced:
            voiced[k] = True
            f0_hz[k] = f0
            vt_db = _vocal_tract_db_stage4(
                _FREQS_SAFE,
                track.formants[ti],
                track.bandwidths[ti],
                float(track.zero_freq[ti]),
                float(track.zero_bw[ti]),
                float(track.zero_depth_db[ti]),
                prof,
            )
            gain_db = 20.0 * math.log10(max(0.20, vg))
            vt_db = vt_db + gain_db + _EN_VOICED_OFFSET_DB
            if vbar > 0.0:
                # Voiced obstruents keep low-frequency voicing bar while rolling off excess harmonics
                lp_db = -10.0 * np.log10(1.0 + (_FREQS_SAFE / 900.0) ** 4)
                vt_db = vt_db + lp_db

            if has_noise:
                f_db = _fricative_db(_FREQS_SAFE, track.fric_p1[ti], track.fric_p2[ti], prof)
                f_db = f_db + 20.0 * math.log10(max(0.15, ng)) + _EN_UNVOICED_OFFSET_DB
                comb_db = 10.0 * np.log10(10.0 ** (vt_db / 10.0) + 10.0 ** (f_db / 10.0) + 1e-18)
                log_env[k] = np.clip(comb_db * _DB_TO_NEPER, -16.0, 6.0)
                bap[k] = _STAGE4_VOICED_FRIC_BAP
            else:
                log_env[k] = np.clip(vt_db * _DB_TO_NEPER, -16.0, 6.0)
                bap[k] = _STAGE4_EN_VOICED_BAP
        elif has_noise:
            f_db = _fricative_db(_FREQS_SAFE, track.fric_p1[ti], track.fric_p2[ti], prof)
            burst_boost = 3.0 if float(track.burst_env[ti]) > 0.5 else 0.0
            f_db = f_db + 20.0 * math.log10(max(0.15, ng)) + _EN_UNVOICED_OFFSET_DB + burst_boost
            log_env[k] = np.clip(f_db * _DB_TO_NEPER, -16.0, 6.0)
            bap[k] = 0.99

    en_ctrl = LarynxControl(
        f0_hz=f0_hz,
        voiced=voiced,
        log_env=log_env,
        bap=bap,
        glottal_tilt_db_per_oct=np.zeros(n_fr, dtype=np.float64),
        jitter_frac=np.full(n_fr, 0.0055, dtype=np.float64),
        shimmer_frac=np.full(n_fr, 0.025, dtype=np.float64),
        n_samples=n_fr * HOP,
        rd=np.full(n_fr, 1.05, dtype=np.float64),
    )
    subharm_strength = 0.0
    if expressive_seg is not None:
        en_ctrl, subharm_strength = apply_emotion_to_control(
            en_ctrl,
            expressive_seg,
            stability=params.stability,
            seed=params.seed,
        )
    return en_ctrl, subharm_strength


def _match_mixed_voice_source(
    zh_controls: list[LarynxControl], en_controls: list[LarynxControl]
) -> None:
    """Bring the low-band aperiodicity, vocal body warmth, tilt & loudness of EN toward this clause's Mandarin voice."""
    if not zh_controls or not en_controls:
        return
    zh_bap = np.concatenate([ctrl.bap[ctrl.voiced, :2] for ctrl in zh_controls])
    en_bap = np.concatenate([ctrl.bap[ctrl.voiced, :2] for ctrl in en_controls])
    if len(zh_bap) >= 8 and len(en_bap) >= 8:
        # A bounded shared offset preserves variation among English sounds and leaves unvoiced
        # consonants and higher bands untouched.
        limit = np.array([0.06, 0.16])
        delta = np.clip(0.8 * (np.median(zh_bap, axis=0) - np.median(en_bap, axis=0)), -limit, limit)
        for ctrl in en_controls:
            ctrl.bap[ctrl.voiced, :2] = np.clip(ctrl.bap[ctrl.voiced, :2] + delta, 0.002, 0.985)

    # Same-clause vocal body (80-320 Hz chest warmth + 360-850 Hz low-mid balance), macro tilt & loudness
    zh_v_parts = [ctrl.log_env[ctrl.voiced & (ctrl.f0_hz > 40.0)] for ctrl in zh_controls if np.any(ctrl.voiced & (ctrl.f0_hz > 40.0))]
    en_v_parts = [ctrl.log_env[ctrl.voiced & (ctrl.f0_hz > 40.0)] for ctrl in en_controls if np.any(ctrl.voiced & (ctrl.f0_hz > 40.0))]
    if not zh_v_parts or not en_v_parts:
        return
    zh_v_env = np.concatenate(zh_v_parts, axis=0)
    en_v_env = np.concatenate(en_v_parts, axis=0)
    if len(zh_v_env) < 12 or len(en_v_env) < 8:
        return

    zh_ltas = np.mean(zh_v_env, axis=0)
    en_ltas = np.mean(en_v_env, axis=0)
    sf_cons = load_sf_bilingual_consistency()
    dv_nep = np.asarray(sf_cons.get("delta_voiced_ltas_en_minus_zh_nep") or np.zeros(N_BINS), dtype=np.float64)
    if dv_nep.shape != (N_BINS,):
        dv_nep = np.zeros(N_BINS, dtype=np.float64)
    zh_target_ltas = zh_ltas + 0.70 * dv_nep
    target_drms_nep = float(sf_cons.get("delta_rms_db_en_minus_zh", 0.0)) * _DB_TO_NEPER if sf_cons else 0.0

    w_nyq = 0.5 * (1.0 + np.cos(math.pi * np.clip((_FREQS - 11200.0) / 800.0, 0.0, 1.0)))
    m_lo = (_FREQS >= 80.0) & (_FREQS <= 320.0)
    m_lm = (_FREQS >= 360.0) & (_FREQS <= 850.0)
    w_chest = 0.5 * (1.0 + np.cos(math.pi * np.clip((_FREQS - 120.0) / 340.0, 0.0, 1.0))) * (_FREQS >= 60.0)
    w_lowmid = np.exp(-0.5 * ((_FREQS - 540.0) / 190.0) ** 2)
    w_en_len = float(np.clip(len(en_v_env) / 140.0, 0.35, 1.0))

    d_lo = float(np.clip(0.85 * (float(np.mean(zh_target_ltas[m_lo])) - float(np.mean(en_ltas[m_lo]))), -0.45, 0.65))
    d_lm = float(np.clip(0.65 * (float(np.mean(zh_target_ltas[m_lm])) - float(np.mean(en_ltas[m_lm]))), -0.55, 0.45)) * w_en_len
    res = zh_target_ltas - (en_ltas + d_lo * w_chest + d_lm * w_lowmid)
    r_cep = log_env_to_cep(res[None, :], n_cep=5)[0]
    r_cep[0] = 0.0
    macro_slope = np.clip(cep_to_log_env(r_cep[None, :], n_bins=N_BINS)[0], -0.45, 0.45) * (0.55 * w_en_len)
    eq_mix = np.clip(d_lo * w_chest + d_lm * w_lowmid + macro_slope, -0.65, 0.70) * w_nyq

    en_after_parts: list[np.ndarray] = []
    for ctrl in en_controls:
        vm = ctrl.voiced & (ctrl.f0_hz > 40.0)
        if np.any(vm):
            ctrl.log_env[vm] = np.clip(ctrl.log_env[vm] + eq_mix[None, :], -16.0, 2.5)
            en_after_parts.append(ctrl.log_env[vm])

    if en_after_parts:
        en_v_after = np.concatenate(en_after_parts, axis=0)
        zh_pwr = float(np.median(0.5 * np.log(np.mean(np.exp(2.0 * zh_v_env), axis=1) + 1e-18))) + target_drms_nep
        en_pwr = float(np.median(0.5 * np.log(np.mean(np.exp(2.0 * en_v_after), axis=1) + 1e-18)))
        pwr_shift = float(np.clip(0.85 * (zh_pwr - en_pwr), -0.45, 0.55))
        if abs(pwr_shift) > 0.02:
            for ctrl in en_controls:
                act = np.max(ctrl.log_env, axis=1) > -13.5
                if np.any(act):
                    ctrl.log_env[act] = np.clip(ctrl.log_env[act] + pwr_shift, -16.0, 2.5)


def _anchor_english_onset_pitch(mandarin: LarynxControl, english: LarynxControl) -> None:
    """Carry the nearby Mandarin register into the first voiced English frames."""
    left = np.flatnonzero(mandarin.voiced[-16:] & (mandarin.f0_hz[-16:] > 40.0))
    right = np.flatnonzero(english.voiced[:32] & (english.f0_hz[:32] > 40.0))
    if len(left) < 4 or len(right) < 4:
        return
    left_pitch = mandarin.f0_hz[-16:][left]
    gap_frames = len(mandarin.f0_hz[-16:]) - 1 - left[-1] + right[0]
    if gap_frames > 15:
        return
    # Apply an additive pitch offset, which keeps the local intonation contour.
    # Fade over real time so any intervening consonants retain their timing and voicing.
    reference = np.median(np.log2(left_pitch[-12:]))
    onset = np.median(np.log2(english.f0_hz[right[:12]]))
    offset_st = float(np.clip(0.8 * 12.0 * (reference - onset), -6.0, 6.0))
    start = int(right[0])
    elapsed = np.arange(min(36, len(english.f0_hz) - start))
    weight = 0.5 + 0.5 * np.cos(math.pi * np.maximum(0, elapsed - 7) / 28)
    active = english.voiced[start + elapsed] & (english.f0_hz[start + elapsed] > 40.0)
    frames = start + elapsed[active]
    english.f0_hz[frames] = np.clip(
        english.f0_hz[frames] * np.exp2(offset_st * weight[active] / 12.0), 62.0, 460.0
    )


def soften_mixed_onset_levels(
    samples: np.ndarray, control: LarynxControl, onsets: list[tuple[int, int, int]],
) -> None:
    """Taper excessive English onset level using the rendered speech as the reference."""
    if not onsets:
        return
    hop = control.hop
    n_frames = min(len(control.voiced), len(samples) // hop)
    power = np.mean(samples[:n_frames * hop].reshape(-1, hop) ** 2, axis=1)
    voiced = control.voiced[:n_frames] & (control.f0_hz[:n_frames] > 40.0)
    for left_start, join, right_stop in onsets:
        tail_start = max(left_start, join - 32)
        left = np.flatnonzero(voiced[tail_start:join])[-16:] + tail_start
        right = np.flatnonzero(voiced[join:min(right_stop, join + 32)])[:16] + join
        if len(left) < 4 or len(right) < 4 or right[0] - left[-1] > 16:
            continue
        left_power = float(np.median(power[left]))
        right_power = float(np.median(power[right]))
        jump_db = 10.0 * math.log10((right_power + 1e-18) / (left_power + 1e-18))
        if jump_db <= 2.0 or left_power < 1e-10:
            continue

        attenuation_db = min(3.5, 0.7 * (jump_db - 2.0))
        first = int(right[0])
        stop = min(first + 26, right_stop, n_frames)
        start_sample = first * hop
        end_sample = stop * hop
        elapsed = np.arange(end_sample - start_sample) / hop
        fade_in = 0.5 - 0.5 * np.cos(math.pi * np.clip(elapsed, 0.0, 1.0))
        fade_out = 0.5 + 0.5 * np.cos(math.pi * np.clip((elapsed - 8.0) / 18.0, 0.0, 1.0))
        # Use a short ramp at a nearby consonant or the end of a short switch.
        gate = np.repeat(voiced[first:stop], hop).astype(np.float64)
        kernel = np.hanning(hop + 1)
        gate *= np.convolve(gate, kernel / np.sum(kernel), mode="same")
        tail = np.clip((len(elapsed) - 1 - np.arange(len(elapsed))) / hop, 0.0, 1.0)
        gain_db = attenuation_db * fade_in * fade_out * gate * tail
        samples[start_sample:end_sample] *= np.exp(-_DB_TO_NEPER * gain_db)


def _bridge_en_to_zh_boundary(english: LarynxControl, mandarin: LarynxControl) -> None:
    """Bridge pause-free EN -> ZH tail loudness & trailing F0 smoothly into the Mandarin onset."""
    tail_n = min(32, len(english.voiced))
    head_n = min(32, len(mandarin.voiced))
    if tail_n < 4 or head_n < 4:
        return
    left = np.flatnonzero(english.voiced[-tail_n:] & (english.f0_hz[-tail_n:] > 40.0))
    right = np.flatnonzero(mandarin.voiced[:head_n] & (mandarin.f0_hz[:head_n] > 40.0))
    if len(left) < 4 or len(right) < 4:
        return
    gap_frames = (tail_n - 1 - int(left[-1])) + int(right[0])
    if gap_frames > 15:
        return

    en_start = len(english.log_env) - tail_n
    en_idx = en_start + left[-16:]
    zh_idx = right[:16]

    # Compare median log-amplitude over perceptually dominant speech bins [2..85] (~100..4000 Hz)
    en_pwr = float(np.median(0.5 * np.log(np.mean(np.exp(2.0 * english.log_env[en_idx, 2:86]), axis=1) + 1e-18)))
    zh_pwr = float(np.median(0.5 * np.log(np.mean(np.exp(2.0 * mandarin.log_env[zh_idx, 2:86]), axis=1) + 1e-18)))
    delta_nep = zh_pwr - en_pwr
    if abs(delta_nep) > 0.14:
        shift_nep = float(np.clip(0.80 * delta_nep, -0.55, 1.05))
        ramp_len = min(24, len(english.log_env))
        r0 = len(english.log_env) - ramp_len
        u = np.linspace(0.0, 1.0, ramp_len, dtype=np.float64)
        w = 0.5 * (1.0 - np.cos(math.pi * u))
        # Apply smoothly over voiced frames in tail, plus faint bridge on final unvoiced transitional frames
        v_mask = english.voiced[r0:]
        w_eff = np.where(v_mask, w, 0.35 * w)
        english.log_env[r0:] = np.clip(english.log_env[r0:] + (shift_nep * w_eff)[:, None], -16.0, 2.5)

    # Smooth trailing F0 toward Mandarin onset register when boundary diff > 0.8 st
    left_f0 = english.f0_hz[en_idx[-10:]]
    right_f0 = mandarin.f0_hz[zh_idx[:10]]
    en_st = float(np.median(np.log2(left_f0)))
    zh_st = float(np.median(np.log2(right_f0)))
    diff_st = 12.0 * (zh_st - en_st)
    if 0.8 < abs(diff_st) <= 7.0:
        f_ramp = min(32, len(english.f0_hz))
        f0_s = len(english.f0_hz) - f_ramp
        u_f = np.linspace(0.0, 1.0, f_ramp, dtype=np.float64)
        w_f = 0.5 * (1.0 - np.cos(math.pi * u_f))
        vm = english.voiced[f0_s:] & (english.f0_hz[f0_s:] > 40.0)
        adj_st = float(np.clip(0.78 * diff_st, -5.0, 5.0))
        english.f0_hz[f0_s:][vm] = np.clip(
            english.f0_hz[f0_s:][vm] * np.exp2((adj_st * w_f[vm]) / 12.0),
            62.0,
            460.0,
        )


def _concat_controls(chunks: list[tuple[LarynxControl, float]]) -> LarynxControl:
    """Stitch (control, pause_after_ms) chunks smoothly with silence or cepstral/F0 junction bridge."""
    if not chunks:
        return _silence_control(0)

    parts: list[LarynxControl] = []
    prev_pause_ms = 999.0
    for idx, (ctrl, pause_ms) in enumerate(chunks):
        c = ctrl.copy()
        if parts and prev_pause_ms < 20.0 and len(parts[-1].f0_hz) >= 5 and len(c.f0_hz) >= 5:
            # Smooth 25 ms (5-frame) voiced F0 glide + 20 ms cep[1:18] bridge across pause-free ZH <-> EN seams
            prev = parts[-1]
            if prev.voiced[-1] and prev.f0_hz[-1] > 40.0:
                k_f0 = 5
                first_v = int(np.argmax(c.voiced[:k_f0])) if np.any(c.voiced[:k_f0]) else -1
                if 0 <= first_v < k_f0 and c.f0_hz[first_v] > 40.0:
                    df_st = abs(12.0 * math.log2(c.f0_hz[first_v] / prev.f0_hz[-1]))
                    if df_st <= 6.0:
                        mid_f0 = math.sqrt(prev.f0_hz[-1] * c.f0_hz[first_v])
                        # Pull tail 3 frames of prev gently toward mid_f0
                        for j_p, w_p in enumerate((0.20, 0.45, 0.72)):
                            pi = len(prev.f0_hz) - 3 + j_p
                            if pi >= 0 and prev.voiced[pi] and prev.f0_hz[pi] > 40.0:
                                prev.f0_hz[pi] = (1.0 - 0.55 * w_p) * prev.f0_hz[pi] + 0.55 * w_p * mid_f0
                        # Pull leading voiced frames of cur smoothly from mid_f0
                        for j in range(first_v, k_f0):
                            if c.voiced[j] and c.f0_hz[j] > 40.0:
                                w_c = 1.0 - (j - first_v) / float(max(1, k_f0 - first_v))
                                c.f0_hz[j] = (1.0 - 0.55 * w_c) * c.f0_hz[j] + 0.55 * w_c * mid_f0

            if prev.voiced[-1] and c.voiced[0]:
                c_prev = log_env_to_cep(prev.log_env[-3:])
                c_cur = log_env_to_cep(c.log_env[:4])
                edge_l = c_prev[-1, 1:18].copy()
                edge_r = c_cur[2, 1:18].copy()
                mid_c = 0.5 * (edge_l + edge_r)
                c_prev[-2, 1:18] = 0.65 * c_prev[-2, 1:18] + 0.35 * mid_c
                c_prev[-1, 1:18] = 0.45 * c_prev[-1, 1:18] + 0.55 * mid_c
                c_cur[0, 1:18] = 0.45 * c_cur[0, 1:18] + 0.55 * mid_c
                c_cur[1, 1:18] = 0.70 * c_cur[1, 1:18] + 0.30 * mid_c
                prev.log_env[-3:] = np.clip(cep_to_log_env(c_prev, n_bins=N_BINS), -16.0, 6.0)
                c.log_env[:4] = np.clip(cep_to_log_env(c_cur, n_bins=N_BINS), -16.0, 6.0)

        parts.append(c)

        if idx < len(chunks) - 1 and pause_ms >= 20.0:
            n_gap = max(1, int(round(pause_ms / 5.0)))
            parts.append(_silence_control(n_gap))
            prev_pause_ms = float(pause_ms)
        else:
            prev_pause_ms = 0.0 if idx < len(chunks) - 1 else 999.0

    f0_hz = np.concatenate([p.f0_hz for p in parts])
    voiced = np.concatenate([p.voiced for p in parts])
    log_env = np.concatenate([p.log_env for p in parts], axis=0)
    bap = np.concatenate([p.bap for p in parts], axis=0)
    tilt = np.concatenate([p.glottal_tilt_db_per_oct for p in parts])
    jit = np.concatenate([p.jitter_frac for p in parts])
    shim = np.concatenate([p.shimmer_frac for p in parts])
    rd = np.concatenate([p.rd for p in parts])

    return LarynxControl(
        f0_hz=f0_hz,
        voiced=voiced,
        log_env=log_env,
        bap=bap,
        glottal_tilt_db_per_oct=tilt,
        jitter_frac=jit,
        shimmer_frac=shim,
        n_samples=len(f0_hz) * HOP,
        rd=rd,
    )


def _extract_zh_subtext(norm_text: str, run_units: list[dict[str, Any]]) -> str:
    """Reconstruct Chinese subtext for a contiguous ZH unit run, keeping trailing clause punctuation."""
    spans = [u.get("text_span") for u in run_units if isinstance(u.get("text_span"), list) and len(u["text_span"]) == 2]
    if spans and len(spans) == len(run_units):
        s0 = max(0, int(spans[0][0]))
        s1 = min(len(norm_text), int(spans[-1][1]))
        tail = norm_text[s1:]
        m = _PUNCT_AFTER_RE.match(tail)
        extra = m.group(0).strip() if m else ""
        sub = (norm_text[s0:s1] + extra).strip()
        if sub:
            return sub
    return "".join(str(u.get("hanzi") or u.get("token") or "") for u in run_units)


def build_utterance_control(
    text: str,
    utt_id: str,
    kb: AcousticKnowledgeStage2 | AcousticKnowledgeStage5 | AcousticKnowledgeStage7,
    arbiter: Arbiter,
    rules: list[Any],
    params: VoiceParams,
    *,
    use_stage10_en: bool = True,
) -> tuple[LarynxControl, dict[str, Any]]:
    """Build full expressive utterance LarynxControl supporting ZH, EN, ZH+EN & non-verbal tags."""
    allow_en = params.crossling_mode != "zh_only"
    segs = parse_expressive_text(
        text,
        emotion_mode=getattr(params, "emotion_mode", "auto"),
        emotion_intensity=getattr(params, "emotion_intensity", 0.75),
        stability=getattr(params, "stability", 0.55),
        auto_breath=getattr(params, "auto_breath", True),
    )
    if not segs:
        raise ValueError("输入文本为空，请输入中文或英文文本。")

    base_f0 = float(kb.speaker_f0_med_hz)
    pause_scale = max(0.40, min(2.00, float(params.pause_scale)))
    chunks: list[tuple[LarynxControl, float]] = [(_silence_control(8), 0.0)]
    arbiter_traces: list[dict[str, Any]] = []
    detected_emotions: list[str] = []
    norm_pieces: list[str] = []
    n_zh_tot = 0
    n_en_tot = 0
    english_onset_chunks: set[int] = set()
    n_voc_tot = 0
    subharm_max = 0.0

    for s_idx, seg in enumerate(segs):
        if seg.kind == "pause":
            chunks.append((build_pause_control(seg.duration_ms * pause_scale), 0.0))
            continue
        if seg.kind in ("breath", "gasp"):
            n_voc_tot += 1
            chunks.append((build_breath_control(base_f0, duration_ms=seg.duration_ms, intensity=seg.intensity, is_gasp=(seg.kind == "gasp")), 0.0))
            continue
        if seg.kind == "sigh":
            n_voc_tot += 1
            chunks.append((build_sigh_control(base_f0, duration_ms=seg.duration_ms, intensity=seg.intensity), 0.0))
            continue
        if seg.kind in ("laugh", "chuckle"):
            n_voc_tot += 1
            chunks.append((build_laugh_control(base_f0, duration_ms=seg.duration_ms, intensity=seg.intensity, soft=(seg.kind == "chuckle")), 0.0))
            continue

        norm_seg = normalize_text_for_larynx(seg.text, allow_english=allow_en)
        if not norm_seg:
            continue
        norm_pieces.append(norm_seg)
        ir = _safe_plan_v13(norm_seg)
        all_units = list(ir.get("units") or [])
        if not all_units:
            continue

        zh_units = [u for u in all_units if u.get("lang", "zh") == "zh"]
        en_units = [u for u in all_units if u.get("lang") == "en"]
        if not allow_en and not zh_units:
            continue

        if seg.auto_breath_before:
            n_voc_tot += 1
            chunks.append((build_breath_control(base_f0, duration_ms=170.0, intensity=seg.intensity), 0.0))

        detected_emotions.append(seg.emotion)
        seg_pause = seg.pause_after_ms * pause_scale if s_idx < len(segs) - 1 else 0.0

        if not allow_en or not en_units:
            n_zh_tot += len(zh_units)
            zh_ctrl, zh_meta = build_zh_run_control(
                norm_seg,
                f"{utt_id}_s{s_idx}",
                kb,
                arbiter,
                rules,
                params,
                trailing_pause=False,
                expressive_seg=seg,
            )
            subharm_max = max(subharm_max, float(zh_meta.get("subharm_strength", 0.0)))
            if zh_meta.get("arbiter_trace"):
                arbiter_traces.append(zh_meta["arbiter_trace"])
            chunks.append((zh_ctrl, seg_pause))
            continue

        # Partition speech clause into contiguous ZH / EN runs
        runs: list[tuple[str, list[dict[str, Any]]]] = []
        cur_lang: str | None = None
        cur_list: list[dict[str, Any]] = []
        for u in all_units:
            lang = "en" if u.get("lang") == "en" else "zh"
            if cur_lang is None or lang == cur_lang:
                cur_lang = lang
                cur_list.append(u)
            else:
                runs.append((cur_lang, cur_list))
                cur_lang = lang
                cur_list = [u]
        if cur_list and cur_lang is not None:
            runs.append((cur_lang, cur_list))

        n_zh_tot += len(zh_units)
        n_en_tot += len(en_units)
        stage10_enabled = use_stage10_en and is_stage10_en_ready()
        zh_controls: list[LarynxControl] = []
        en_controls: list[LarynxControl] = []
        total_u = max(1, len(all_units))
        unit_cursor = 0
        is_q = norm_seg.rstrip().endswith(("?", "？"))
        for r_idx, (lang, r_units) in enumerate(runs):
            is_last_run = r_idx == len(runs) - 1
            p0 = unit_cursor / float(total_u)
            unit_cursor += len(r_units)
            p1 = unit_cursor / float(total_u)
            last_pause = seg_pause if is_last_run else (float(r_units[-1].get("pause_after_ms", 0.0)) * pause_scale)
            if lang == "zh":
                sub_zh = _extract_zh_subtext(norm_seg, r_units)
                continues_after = stage10_enabled and not is_last_run and last_pause < 20.0
                zh_ctrl, zh_meta = build_zh_run_control(
                    sub_zh,
                    f"{utt_id}_s{s_idx}_zh{r_idx}",
                    kb,
                    arbiter,
                    rules,
                    params,
                    trailing_pause=False,
                    expressive_seg=seg,
                    continues_after=continues_after,
                )
                if stage10_enabled and r_idx > 0 and runs[r_idx - 1][0] == "en" and chunks[-1][1] < 20.0:
                    _bridge_en_to_zh_boundary(chunks[-1][0], zh_ctrl)
                subharm_max = max(subharm_max, float(zh_meta.get("subharm_strength", 0.0)))
                if zh_meta.get("arbiter_trace"):
                    arbiter_traces.append(zh_meta["arbiter_trace"])
                chunks.append((zh_ctrl, last_pause))
                zh_controls.append(zh_ctrl)
            else:
                connected_to_mandarin = stage10_enabled and r_idx > 0 and chunks[-1][1] < 20.0
                en_continues_after = stage10_enabled and not is_last_run and last_pause < 20.0
                next_b_cls: int | None = None
                if en_continues_after and r_idx + 1 < len(runs) and runs[r_idx + 1][1]:
                    nxt_zh_u = runs[r_idx + 1][1][0]
                    nxt_onset = str(nxt_zh_u.get("onset") or "").strip()
                    if nxt_onset:
                        nxt_ipa = (cf.PINYIN_ONSET_TO_IPA.get(nxt_onset) or ["ə"])[0]
                        next_b_cls = classify_articulatory_class(nxt_ipa)
                    else:
                        next_b_cls = 1
                run_base_f0 = base_f0
                if connected_to_mandarin:
                    prev_zh = chunks[-1][0]
                    pv_mask = prev_zh.voiced & (prev_zh.f0_hz > 40.0)
                    if int(np.sum(pv_mask)) >= 8:
                        prev_med_f0 = float(np.median(prev_zh.f0_hz[pv_mask][-60:]))
                        run_base_f0 = float(np.clip(0.65 * prev_med_f0 + 0.35 * base_f0, 0.78 * base_f0, 1.28 * base_f0))
                if stage10_enabled and r_idx > 0:
                    prior_units = runs[r_idx - 1][1][-8:]
                    context = "".join(str(unit.get("hanzi") or unit.get("token") or "") for unit in prior_units)
                    r_units = [{**r_units[0], "preceding_mandarin": context}, *r_units[1:]]
                en_ctrl, en_subh = build_en_run_control(
                    r_units,
                    run_base_f0,
                    params,
                    expressive_seg=seg,
                    kb=kb,
                    clause_pos=(p0, p1),
                    is_clause_final=is_last_run or (stage10_enabled and last_pause >= 20.0),
                    is_question=(is_last_run and is_q),
                    use_stage10_en=stage10_enabled,
                    expand_projected_pitch=not zh_units,
                    continues_after=en_continues_after,
                    embedded_in_zh=bool(zh_units),
                    next_boundary_cls=next_b_cls,
                )
                if connected_to_mandarin:
                    _anchor_english_onset_pitch(chunks[-1][0], en_ctrl)
                    english_onset_chunks.add(len(chunks))
                subharm_max = max(subharm_max, float(en_subh))
                chunks.append((en_ctrl, last_pause))
                en_controls.append(en_ctrl)

        if stage10_enabled:
            _match_mixed_voice_source(zh_controls, en_controls)

    if len(chunks) <= 1:
        raise ValueError(f"无法解析文本发音单元: {text!r}")

    chunks.append((_silence_control(14), 0.0))
    stitched = _concat_controls(chunks)
    level_onsets: list[tuple[int, int, int]] = []
    if english_onset_chunks:
        start = 0
        for index, (control, pause_ms) in enumerate(chunks):
            end = start + len(control.voiced)
            if index in english_onset_chunks:
                level_onsets.append((start - len(chunks[index - 1][0].voiced), start, end))
            start = end + (int(round(pause_ms / 5.0)) if pause_ms >= 20.0 else 0)
    uniq_emos = list(dict.fromkeys(detected_emotions)) or ["neutral"]
    return stitched, {
        "n_zh": n_zh_tot,
        "n_en": n_en_tot,
        "level_onsets": level_onsets,
        "n_vocalizations": n_voc_tot,
        "detected_emotions": uniq_emos,
        "subharm_strength": subharm_max,
        "normalized_text": " ".join(norm_pieces) if norm_pieces else text.strip(),
        "arbiter_traces": arbiter_traces,
    }
