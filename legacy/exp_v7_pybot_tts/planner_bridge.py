"""Phase-3 PyBot Planner Bridge: Cross-Lingual IPA/Prosody -> Canonical Unit Bank -> VocoderFeatures.

Bridges ``exp_v5_arch.crossling_frontend`` (ZH/EN/ZH+EN IPA + tone/stress rules)
with ``exp_v7_pybot_tts.unit_bank`` / ``crossling_warp`` and renders via
``exp_v6_vocoder.cloner.adapt_features`` + ``exp_v6_vocoder.vocoder.synthesize``.

Key fixes over ``exp_v6_crossling_dsp``:
1. Smooth 20-30 ms raised-cosine coarticulation splines on ``log_env`` and ``bap``
   across unit and V/UV boundaries (eliminates 5.05x flux_p95/med & 11 dB V/UV spikes).
2. Median-centered F0 track + max-slope clamping (eliminates +1.0..+1.6 st upward
   F0 bias and Tone-4 >4 st frame glitches).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_arch import crossling_frontend as cf
from exp_v6_crossling_dsp import arbiter as v6_arbiter
from exp_v6_vocoder import SR, cloner, vocoder
from exp_v7_pybot_tts import crossling_warp

HOP_SAMPLES = 120
HOP_MS = 5.0
N_FFT = 1024
N_BINS = N_FFT // 2 + 1
SILENCE_LOG_ENV = -9.5


@dataclass
class UnitSlot:
    label: str
    lang: str
    is_onset: bool
    dur_ms: float
    pitch_st_5: list[float]
    gain: float
    log_env: np.ndarray    # (K, 513)
    bap: np.ndarray        # (K, 5)
    voiced: np.ndarray     # (K,) float/bool in [0, 1]
    source_mode: str
    pause_after_ms: float = 0.0
    is_pause: bool = False


def _resample_2d(arr: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly resample ``(K, D)`` trajectory along axis 0 to ``n_out`` frames."""
    k = arr.shape[0]
    if k == n_out:
        return arr.astype(np.float64, copy=True)
    if k == 1:
        return np.repeat(arr.astype(np.float64), n_out, axis=0)
    t_in = np.linspace(0.0, 1.0, k)
    t_out = np.linspace(0.0, 1.0, n_out)
    out = np.empty((n_out, arr.shape[1]), dtype=np.float64)
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(t_out, t_in, arr[:, d])
    return out


def _resample_1d(arr: np.ndarray, n_out: int) -> np.ndarray:
    k = arr.shape[0]
    if k == n_out:
        return arr.astype(np.float64, copy=True)
    if k == 1:
        return np.repeat(arr.astype(np.float64), n_out)
    return np.interp(np.linspace(0.0, 1.0, n_out), np.linspace(0.0, 1.0, k), arr.astype(np.float64))


def _interp_list(vals: list[float], t: float) -> float:
    p = max(0.0, min(1.0, t)) * (len(vals) - 1)
    i = min(int(p), len(vals) - 1)
    j = min(i + 1, len(vals) - 1)
    return vals[i] + (vals[j] - vals[i]) * (p - i)


def _slice_pitch_5(pitch_5: list[float], a: float, b: float) -> list[float]:
    return [_interp_list(pitch_5, a + (b - a) * (m / 4.0)) for m in range(5)]


def _smooth_spline_boundary(
    env_a: np.ndarray,
    env_b: np.ndarray,
    bap_a: np.ndarray,
    bap_b: np.ndarray,
    v_a: np.ndarray,
    v_b: np.ndarray,
    blend_frames: int = 5,
) -> None:
    """Apply symmetric raised-cosine coarticulation across adjacent unit tails/heads in-place."""
    w = min(blend_frames, len(env_a) // 2, len(env_b) // 2)
    if w < 1:
        return
    mid_env = 0.5 * (env_a[-1] + env_b[0])
    mid_bap = 0.5 * (bap_a[-1] + bap_b[0])

    # Tail of unit A glides from its own contour toward mid_env at the boundary
    u_a = 0.5 * (1.0 - np.cos(math.pi * np.linspace(0.0, 0.5, w)))[:, None]
    env_a[-w:] = (1.0 - u_a) * env_a[-w:] + u_a * mid_env[None, :]
    bap_a[-w:] = (1.0 - u_a) * bap_a[-w:] + u_a * mid_bap[None, :]

    # Head of unit B glides from mid_env into its own contour
    u_b = 0.5 * (1.0 - np.cos(math.pi * np.linspace(0.5, 1.0, w)))[:, None]
    env_b[:w] = (1.0 - u_b) * mid_env[None, :] + u_b * env_b[:w]
    bap_b[:w] = (1.0 - u_b) * mid_bap[None, :] + u_b * bap_b[:w]

    # Soften abrupt V/UV bap cliffs without forcing unvoiced frames to oscillate
    if bool(np.any(v_a > 0.5)) != bool(np.any(v_b > 0.5)):
        v_edge = min(3, w)
        ramp = np.linspace(0.25, 0.75, v_edge)[:, None]
        if np.any(v_a > 0.5) and not np.any(v_b > 0.5):
            # Voiced -> Unvoiced: gradually raise BAP in last 3 voiced frames
            bap_a[-v_edge:] = np.maximum(bap_a[-v_edge:], ramp)
        elif not np.any(v_a > 0.5) and np.any(v_b > 0.5):
            # Unvoiced -> Voiced: soften first 3 voiced frames
            bap_b[:v_edge] = np.maximum(bap_b[:v_edge], 1.0 - ramp)


def _apply_f0_continuity_and_centering(
    slots: list[UnitSlot],
    *,
    canonical_f0_hz: float = 210.0,
    iqr_scale: float = 1.0,
    max_step_st: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble continuous (f0_hz, voiced, log_env, bap) across all slots."""
    rendered_envs: list[np.ndarray] = []
    rendered_baps: list[np.ndarray] = []
    rendered_voiced: list[np.ndarray] = []
    rendered_st: list[np.ndarray] = []

    for sl in slots:
        nf = max(2, int(round(sl.dur_ms / HOP_MS)))
        if sl.is_pause:
            rendered_envs.append(np.full((nf, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
            rendered_baps.append(np.ones((nf, 5), dtype=np.float64))
            rendered_voiced.append(np.zeros(nf, dtype=bool))
            rendered_st.append(np.zeros(nf, dtype=np.float64))
            continue

        env = _resample_2d(sl.log_env, nf)
        # Apply rule gain adjustment cleanly in natural-log amplitude domain
        if sl.gain > 0.0 and abs(sl.gain - 1.0) > 1e-4:
            env = env + math.log(max(0.15, min(2.0, sl.gain)))
        bap = np.clip(_resample_2d(sl.bap, nf), 0.005, 0.995)
        v_prob = _resample_1d(sl.voiced, nf)
        v_mask = v_prob >= 0.45

        t_grid = np.linspace(0.0, 1.0, nf)
        st = np.array([_interp_list(sl.pitch_st_5, float(t)) for t in t_grid], dtype=np.float64) * iqr_scale
        rendered_envs.append(env)
        rendered_baps.append(bap)
        rendered_voiced.append(v_mask)
        rendered_st.append(st)

    # Cross-boundary raised-cosine coarticulation spline between adjacent speakable slots
    for i in range(len(slots) - 1):
        if slots[i].is_pause or slots[i + 1].is_pause:
            # Gentle 2-frame fade at pause boundary to prevent hard digital step
            if not slots[i].is_pause and len(rendered_envs[i]) >= 2:
                rendered_envs[i][-2:] = 0.65 * rendered_envs[i][-2:] + 0.35 * SILENCE_LOG_ENV
            if not slots[i + 1].is_pause and len(rendered_envs[i + 1]) >= 2:
                rendered_envs[i + 1][:2] = 0.65 * rendered_envs[i + 1][:2] + 0.35 * SILENCE_LOG_ENV
            continue
        _smooth_spline_boundary(
            rendered_envs[i],
            rendered_envs[i + 1],
            rendered_baps[i],
            rendered_baps[i + 1],
            rendered_voiced[i].astype(np.float64),
            rendered_voiced[i + 1].astype(np.float64),
            blend_frames=5,
        )

    log_env = np.vstack(rendered_envs)
    bap = np.vstack(rendered_baps)
    voiced = np.concatenate(rendered_voiced)
    st_all = np.concatenate(rendered_st)

    # Center voiced F0 semitones on median 0.0 so speaker median F0 tracks target without +1..+1.6 st drift.
    # For very short single-syllable utterances (< 40 voiced frames), keep uncentered inflection.
    if int(np.sum(voiced)) >= 40:
        med_st = float(np.median(st_all[voiced]))
        st_all[voiced] = st_all[voiced] - float(np.clip(med_st, -1.8, 1.8))

    # Clamp frame-to-frame F0 derivative inside contiguous voiced regions (eliminates Tone-4 >4 st glitches)
    for idx in range(len(st_all) - 1):
        if voiced[idx] and voiced[idx + 1]:
            delta = st_all[idx + 1] - st_all[idx]
            if abs(delta) > max_step_st:
                st_all[idx + 1] = st_all[idx] + math.copysign(max_step_st, delta)

    f0_hz = np.where(voiced, canonical_f0_hz * (2.0 ** (st_all / 12.0)), 0.0)
    f0_hz = np.where(voiced, np.clip(f0_hz, 65.0, 440.0), 0.0)
    return f0_hz, voiced, log_env, bap


def plan_to_slots(
    plan_doc: dict,
    bank: Any,
    lookup_syllable_fn: Callable[[Any, str, str], dict] | None = None,
    lookup_ipa_fn: Callable[[Any, str, bool], dict] | None = None,
    *,
    lead_in_ms: float = 60.0,
    lead_out_ms: float = 85.0,
    speech_rate: float = 1.0,
) -> tuple[list[UnitSlot], dict]:
    """Expand a frontend plan into acoustic :class:`UnitSlot`s using canonical unit templates."""
    if lookup_syllable_fn is None:
        lookup_syllable_fn = crossling_warp.lookup_syllable_template
    if lookup_ipa_fn is None:
        lookup_ipa_fn = crossling_warp.lookup_ipa_template
    slots: list[UnitSlot] = [
        UnitSlot(
            label="sil", lang="sil", is_onset=False, dur_ms=lead_in_ms,
            pitch_st_5=[0.0] * 5, gain=0.0,
            log_env=np.full((2, N_BINS), SILENCE_LOG_ENV),
            bap=np.ones((2, 5)), voiced=np.zeros(2),
            source_mode="silence", is_pause=True,
        )
    ]
    source_counts: dict[str, int] = {}

    units = plan_doc["units"]
    n_units = len(units)
    for uidx, u in enumerate(units):
        lang = u["lang"]
        pitch_5 = [float(x) for x in u["pitch_st"]]
        # Apply smooth clause declination across the sentence
        decl = -0.60 * (uidx / max(1, n_units - 1)) + 0.30
        pitch_5 = [p + decl for p in pitch_5]
        gain = float(u.get("gain", 0.85))
        pause_ms = float(u.get("pause_after_ms", 0.0))

        if lang == "zh":
            onset = u.get("onset", "")
            final = u.get("final", "")
            tpl = lookup_syllable_fn(bank, onset, final)
            mode = tpl.get("source_mode", "unknown")
            source_counts[mode] = source_counts.get(mode, 0) + 1
            dur = float(u.get("duration_ms", tpl.get("dur_ms", 230.0))) / max(0.6, speech_rate)
            # Slightly normalize extreme gain differences between tones
            tone = u.get("tone")
            t_gain = 0.88 if tone == 5 else 1.0
            slots.append(
                UnitSlot(
                    label=f"{onset}+{final}" if onset else final,
                    lang="zh",
                    is_onset=False,
                    dur_ms=max(65.0, dur),
                    pitch_st_5=pitch_5,
                    gain=gain * t_gain,
                    log_env=np.asarray(tpl["log_env"], dtype=np.float64),
                    bap=np.asarray(tpl["bap"], dtype=np.float64),
                    voiced=np.asarray(tpl["voiced"], dtype=np.float64),
                    source_mode=mode,
                )
            )
        else:
            # English word unit: expand ARPAbet -> IPA phones, apply cluster & lexical stress rules
            arpa = u.get("arpabet") or []
            pairs = v6_arbiter._split_en_arpabet_to_ipa_with_stress(arpa)
            if not pairs:
                pairs = [(p, None) for p in u["ipa_phones"]]
            n_p = len(pairs)
            # Identify consonant cluster spans for compression
            is_vowel = [st is not None for _, st in pairs]
            raw_tpls = []
            raw_durs = []
            for idx_p, (ipa, st) in enumerate(pairs):
                is_on = (idx_p == 0) and (not is_vowel[idx_p])
                tpl = lookup_ipa_fn(bank, ipa, is_on)
                mode = tpl.get("source_mode", "unknown")
                source_counts[mode] = source_counts.get(mode, 0) + 1
                base_d = float(tpl.get("dur_ms", 85.0))
                if st == 1:
                    base_d *= 1.18
                elif st == 0:
                    base_d = max(42.0, base_d * 0.74)
                raw_tpls.append((ipa, st, tpl, is_on, mode))
                raw_durs.append(base_d)

            # Cluster compression on adjacent non-vowel English phones
            k_i = 0
            while k_i < n_p:
                if is_vowel[k_i]:
                    k_i += 1
                    continue
                k_j = k_i
                while k_j < n_p and not is_vowel[k_j]:
                    k_j += 1
                clen = k_j - k_i
                if clen >= 2:
                    c_scale = 0.72 if clen == 2 else 0.60
                    for m in range(k_i, k_j):
                        raw_durs[m] = max(25.0, raw_durs[m] * c_scale)
                k_i = k_j

            for idx_p, ((ipa, st, tpl, is_on, mode), d_ms) in enumerate(zip(raw_tpls, raw_durs)):
                sub_p = _slice_pitch_5(pitch_5, idx_p / n_p, (idx_p + 1) / n_p)
                if st == 1:
                    sub_p = [v + 0.55 for v in sub_p]
                    p_gain = gain * 1.06
                elif st == 0:
                    sub_p = [v - 0.35 for v in sub_p]
                    p_gain = gain * 0.88
                else:
                    p_gain = gain
                slots.append(
                    UnitSlot(
                        label=ipa,
                        lang="en",
                        is_onset=is_on,
                        dur_ms=max(25.0, d_ms / max(0.6, speech_rate)),
                        pitch_st_5=sub_p,
                        gain=p_gain,
                        log_env=np.asarray(tpl["log_env"], dtype=np.float64),
                        bap=np.asarray(tpl["bap"], dtype=np.float64),
                        voiced=np.asarray(tpl["voiced"], dtype=np.float64),
                        source_mode=mode,
                    )
                )

        if pause_ms > 0.0:
            slots.append(
                UnitSlot(
                    label="sil", lang="sil", is_onset=False, dur_ms=pause_ms,
                    pitch_st_5=[0.0] * 5, gain=0.0,
                    log_env=np.full((2, N_BINS), SILENCE_LOG_ENV),
                    bap=np.ones((2, 5)), voiced=np.zeros(2),
                    source_mode="silence", is_pause=True,
                )
            )

    slots.append(
        UnitSlot(
            label="sil", lang="sil", is_onset=False, dur_ms=lead_out_ms,
            pitch_st_5=[0.0] * 5, gain=0.0,
            log_env=np.full((2, N_BINS), SILENCE_LOG_ENV),
            bap=np.ones((2, 5)), voiced=np.zeros(2),
            source_mode="silence", is_pause=True,
        )
    )
    return slots, {"source_counts": source_counts, "n_slots": len(slots)}


def synthesize_plan(
    plan_doc: dict,
    bank: Any,
    tgt_profile: cloner.FewShotProfile,
    *,
    lookup_syllable_fn: Callable[[Any, str, str], dict] | None = None,
    lookup_ipa_fn: Callable[[Any, str, bool], dict] | None = None,
    canonical_profile: cloner.FewShotProfile | None = None,
    vtln_alpha: float | None = None,
    phase_dispersion: bool = True,
    seed: int = 72,
    speech_rate: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Synthesize a frontend plan through canonical unit bank + ``exp_v6_vocoder``."""
    if canonical_profile is None:
        canonical_profile = getattr(bank, "canonical_profile", None)
    if canonical_profile is None:
        canonical_profile = cloner.FewShotProfile(
            name="canonical",
            n_utterances=1,
            total_voiced_s=1.0,
            f0_median_hz=float(getattr(bank, "canonical_f0_hz", 210.0)),
            f0_log_std=float(getattr(bank, "canonical_f0_log_std", 0.13)),
            f0_iqr_st=float(getattr(bank, "canonical_f0_iqr_st", 2.8)),
            tilt_db_per_oct=float(getattr(bank, "canonical_tilt_db_per_oct", -3.5)),
            hnr_median_db=float(getattr(bank, "canonical_hnr_db", 12.0)),
            voiced_env_mean=np.asarray(
                getattr(bank, "canonical_voiced_env_mean", np.zeros(N_BINS)),
                dtype=np.float64,
            ),
            bap_voiced_median=np.asarray(
                getattr(bank, "canonical_bap_median", np.array([0.18, 0.25, 0.35, 0.55, 0.78])),
                dtype=np.float64,
            ),
            vtln_anchor_f2_hz=float(getattr(bank, "canonical_vtln_anchor_f2_hz", 1750.0)),
        )

    slots, slot_meta = plan_to_slots(
        plan_doc,
        bank,
        lookup_syllable_fn=lookup_syllable_fn,
        lookup_ipa_fn=lookup_ipa_fn,
        speech_rate=speech_rate,
    )
    f0_hz, voiced, log_env, bap = _apply_f0_continuity_and_centering(
        slots,
        canonical_f0_hz=canonical_profile.f0_median_hz,
        iqr_scale=1.0,  # cloner.adapt_features applies target f0_log_std ratio
    )
    n_samples = len(f0_hz) * HOP_SAMPLES
    src_feats = vocoder.VocoderFeatures(
        f0_hz=f0_hz,
        voiced=voiced,
        log_env=log_env,
        bap=bap,
        sr=SR,
        hop_samples=HOP_SAMPLES,
        n_samples=n_samples,
        n_fft=N_FFT,
    )
    adapted = cloner.adapt_features(
        src_feats,
        canonical_profile,
        tgt_profile,
        vtln_alpha=vtln_alpha,
        use_ltas_transfer=True,
        use_bap_transfer=True,
    )
    wav = vocoder.synthesize(adapted, seed=seed, phase_dispersion=phase_dispersion)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0.90:
        wav = wav * (0.90 / peak)
    meta = {
        **slot_meta,
        "n_frames": len(f0_hz),
        "n_samples": len(wav),
        "duration_s": round(len(wav) / SR, 4),
    }
    return wav, meta
