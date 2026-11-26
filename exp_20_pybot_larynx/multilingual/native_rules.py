"""Universal IPA physical & articulatory rules for Stage 13 Native Multilingual TTS.

Four language-agnostic rules operating purely on `PhoneEvent` + `IPAArticulatoryInfo`:
  1. IPAGlottalAntiBuzzRule: f0-coupled glottal Rd, smooth anti-buzz BAP floor, VOT & creak taper.
  2. IPANasalAntiresonanceRule: place-specific nasal zeros (m/n/ŋ/ɲ) + pre-nasal vowel coarticulation.
  3. IPAObstruentCrispnessRule: unvoiced obstruent rumble suppression, voiced obstruent noise/voicing mix.
  4. IPACoarticulationAndSwitchRule: adaptive cep[0:18] S-curve seam bridging (intra- & cross-lingual).
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.fft import dct, rfftfreq

from exp_20_pybot_larynx import N_BAP, N_BINS, N_FFT, SR
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB
from exp_20_pybot_larynx.larynx.synth import LarynxControl
from exp_20_pybot_larynx.multilingual.schema import (
    IPAArticulatoryInfo,
    PhoneEvent,
    lookup_ipa_info,
)

_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_DB_TO_NEPER = 1.0 / NEPER_TO_DB
_CEP_BRIDGE_K = 18
_CEP_DCT18 = dct(np.eye(N_BINS, dtype=np.float64), type=2, norm="ortho", axis=1)[:, :_CEP_BRIDGE_K]

# Physical glottal breathiness bounds preventing metallic buzz (floor) AND TV-static hiss (ceil)
_SONORANT_BAP_FLOOR = np.array([0.045, 0.19, 0.36, 0.58, 0.74, 0.85], dtype=np.float64)
_SONORANT_BAP_CEIL = np.array([0.14, 0.34, 0.54, 0.78, 0.88, 0.94], dtype=np.float64)
_VOICED_OBST_BAP_FLOOR = np.array([0.08, 0.24, 0.44, 0.68, 0.80, 0.88], dtype=np.float64)
_VOICED_OBST_BAP_CEIL = np.array([0.24, 0.48, 0.68, 0.86, 0.92, 0.96], dtype=np.float64)
_VOT_BAP_CEIL = np.array([0.22, 0.44, 0.64, 0.82, 0.90, 0.95], dtype=np.float64)

_SONORANT_MANNERS = frozenset({"vowel", "diphthong", "nasal", "liquid", "glide", "approximant", "tap"})
_OBSTRUENT_MANNERS = frozenset({"stop", "affricate", "fricative"})
_DEVOICED_HF_ROLLOFF = np.clip((_FREQS - 3500.0) / 2500.0, 0.0, 1.0) * (-4.0 * _DB_TO_NEPER)
_VOICED_STOP_HI_MASK = np.clip((_FREQS - 450.0) / 300.0, 0.0, 1.0)


def _gauss(center_hz: float, sigma_hz: float) -> np.ndarray:
    return np.exp(-0.5 * ((_FREQS - center_hz) / sigma_hz) ** 2)


# Place-specific nasal side-branch antiresonance templates (nepers)
_NASAL_TEMPLATES: dict[str, np.ndarray] = {
    "bilabial": _DB_TO_NEPER * (
        -0.75 + 0.45 * _gauss(240.0, 170.0) - 1.35 * _gauss(1350.0, 320.0) - 0.35 * _gauss(3000.0, 700.0)
    ),
    "alveolar": _DB_TO_NEPER * (
        -0.75 + 0.45 * _gauss(230.0, 170.0) - 0.55 * _gauss(900.0, 360.0) - 0.70 * _gauss(2300.0, 620.0)
    ),
    "palatal": _DB_TO_NEPER * (
        -0.75 + 0.45 * _gauss(230.0, 170.0) - 0.90 * _gauss(2100.0, 520.0)
    ),
    "velar": _DB_TO_NEPER * (
        -0.70 + 0.50 * _gauss(230.0, 190.0) - 0.95 * _gauss(1650.0, 560.0) - 0.30 * _gauss(3700.0, 720.0)
    ),
}

# Sub-380 Hz low-rumble suppression for voiceless obstruents + crisp high-band lift
_VOICELESS_RUMBLE_CUT = 1.0 / (1.0 + (_FREQS / 380.0) ** 3)
_VOICED_OBST_LO_WARM = _DB_TO_NEPER * 1.4 * _gauss(210.0, 140.0)


def _clipped_event_span(ev: PhoneEvent, n_fr: int) -> tuple[int, int]:
    a = max(0, min(n_fr, int(ev.start_fr)))
    b = max(a, min(n_fr, int(ev.end_fr)))
    return a, b


class IPAGlottalAntiBuzzRule:
    """F0/energy-coupled Rd, anti-buzz BAP floor, VOT aspiration ramp & tail stabilization."""

    name = "ipa_glottal_anti_buzz"

    def apply(
        self,
        events: Sequence[PhoneEvent],
        ctrl: LarynxControl,
        speaker_f0_med_hz: float,
        *,
        expressive_drive: float = 0.0,
    ) -> dict[str, Any]:
        n_fr = len(ctrl.f0_hz)
        if n_fr == 0:
            return {"rule": self.name, "frames": 0}

        ref_f0 = max(65.0, float(speaker_f0_med_hz))
        voiced = np.asarray(ctrl.voiced, dtype=bool)
        f0 = np.asarray(ctrl.f0_hz, dtype=np.float64)
        e_drive = float(np.clip(expressive_drive, 0.0, 1.5))

        # 1. Glottal Rd & tilt modulation by local F0 register + energy
        vm = voiced & (f0 > 40.0)
        n_v = int(np.sum(vm))
        if n_v > 0:
            st = 12.0 * np.log2(np.maximum(f0[vm], 50.0) / ref_f0)
            frame_db = np.max(ctrl.log_env[vm], axis=1) * NEPER_TO_DB
            e_rel_db = float(np.median(frame_db))
            e_dev = np.clip((frame_db - e_rel_db) / 8.0, -1.0, 1.0)

            rd_v = 1.18 + 0.032 * st - 0.08 * e_dev + 0.06 * e_drive * np.clip(-st / 6.0, 0.0, 1.0)
            ctrl.rd[vm] = np.clip(0.55 * ctrl.rd[vm] + 0.45 * np.clip(rd_v, 0.90, 1.72), 0.85, 1.78)
            ctrl.glottal_tilt_db_per_oct[vm] = np.clip(
                ctrl.glottal_tilt_db_per_oct[vm] - 0.14 * st + 0.35 * e_dev,
                -4.5,
                2.5,
            )
            ctrl.jitter_frac[vm] = np.clip(np.maximum(ctrl.jitter_frac[vm], 0.0062), 0.004, 0.025)
            ctrl.shimmer_frac[vm] = np.clip(np.maximum(ctrl.shimmer_frac[vm], 0.038), 0.020, 0.120)

        # 2. Event-level physical BAP floor/ceiling clamp, VOT ramp, tail relaxation & t_uv attenuation
        touched = 0
        e_bump = (0.02 * min(1.0, e_drive)) * np.array([0.3, 0.5, 0.6, 0.4, 0.2, 0.1])
        for idx, ev in enumerate(events):
            if ev.is_pause:
                continue
            a, b = _clipped_event_span(ev, n_fr)
            if b <= a:
                continue
            info = lookup_ipa_info(ev.ipa)

            if info.voiced:
                is_v_obst = info.manner in _OBSTRUENT_MANNERS
                flr = (_VOICED_OBST_BAP_FLOOR if is_v_obst else _SONORANT_BAP_FLOOR) + e_bump
                cei = (_VOICED_OBST_BAP_CEIL if is_v_obst else _SONORANT_BAP_CEIL) + e_bump
                vm_ev = ctrl.voiced[a:b]
                if np.any(vm_ev):
                    ctrl.bap[a:b][vm_ev] = np.clip(ctrl.bap[a:b][vm_ev], flr[None, :], cei[None, :])

            # Suppress loud whispered static on devoiced sonorant tail frames (e.g. phrase-final t_uv)
            if info.manner in _SONORANT_MANNERS:
                uv_m = ~ctrl.voiced[a:b]
                n_uv = int(np.sum(uv_m))
                if n_uv > 0:
                    atten_nep = np.linspace(-10.0, -22.0, n_uv) * _DB_TO_NEPER
                    ctrl.log_env[a:b][uv_m] = np.clip(
                        ctrl.log_env[a:b][uv_m] + atten_nep[:, None] + _DEVOICED_HF_ROLLOFF[None, :],
                        -16.0,
                        2.5,
                    )

            # Soften first 2 voiced frames right after an unvoiced obstruent (VOT breathiness ramp)
            if idx > 0 and info.voiced:
                prev_ev = events[idx - 1]
                prev_info = lookup_ipa_info(prev_ev.ipa)
                if (not prev_ev.is_pause) and prev_info.voiceless_obstruent and (a - int(prev_ev.end_fr)) <= 1:
                    k_vot = min(2, b - a)
                    for step in range(k_vot):
                        fr = a + step
                        if ctrl.voiced[fr]:
                            w = 1.0 - step / float(max(1, k_vot))
                            vot_flr = np.maximum(_SONORANT_BAP_FLOOR, ctrl.bap[fr] + 0.05 * w)
                            ctrl.bap[fr] = np.clip(vot_flr, _SONORANT_BAP_FLOOR, _VOT_BAP_CEIL)
                            ctrl.glottal_tilt_db_per_oct[fr] = np.clip(
                                ctrl.glottal_tilt_db_per_oct[fr] - 1.0 * w, -4.5, 2.5
                            )
                            touched += 1

            # Phrase-final vowel relaxation (last few frames): slightly higher Rd & softer high harmonics
            if ev.is_phrase_final and ev.is_vowel and (b - a) >= 6:
                tail_k = min(5, max(2, (b - a) // 3))
                t0 = b - tail_k
                u = np.linspace(0.2, 1.0, tail_k)
                vm_t = ctrl.voiced[t0:b]
                if np.any(vm_t):
                    ctrl.rd[t0:b][vm_t] = np.clip(ctrl.rd[t0:b][vm_t] + 0.16 * u[vm_t], 0.85, 1.80)
                    ctrl.bap[t0:b, :3][vm_t] = np.clip(
                        ctrl.bap[t0:b, :3][vm_t] + 0.02 * u[vm_t, None],
                        _SONORANT_BAP_FLOOR[:3][None, :],
                        _SONORANT_BAP_CEIL[:3][None, :],
                    )
                    touched += int(np.sum(vm_t))

        return {"rule": self.name, "voiced_frames": n_v, "edge_frames": touched}


class IPANasalAntiresonanceRule:
    """Language-agnostic oral-cavity nasal zero shaping and regressive vowel nasalization."""

    name = "ipa_nasal_antiresonance"

    def apply(self, events: Sequence[PhoneEvent], ctrl: LarynxControl) -> dict[str, Any]:
        n_fr = len(ctrl.f0_hz)
        if n_fr == 0:
            return {"rule": self.name, "nasal_phones": 0, "coartic_vowels": 0}

        n_nasals = 0
        n_coart = 0
        for idx, ev in enumerate(events):
            if ev.is_pause:
                continue
            a, b = _clipped_event_span(ev, n_fr)
            if b <= a:
                continue
            info: IPAArticulatoryInfo = lookup_ipa_info(ev.ipa)

            if info.manner == "nasal":
                tmpl = _NASAL_TEMPLATES.get(info.place, _NASAL_TEMPLATES["alveolar"])
                # Scale gently so native speaker recordings aren't over-attenuated while zeroes stay crisp
                ctrl.log_env[a:b] = np.clip(ctrl.log_env[a:b] + 0.55 * tmpl[None, :], -16.0, 2.5)
                vm = ctrl.voiced[a:b]
                if np.any(vm):
                    ctrl.bap[a:b, 1:4][vm] = np.clip(ctrl.bap[a:b, 1:4][vm] - 0.012, 0.008, 0.985)
                n_nasals += 1
                continue

            # Regressive VN coarticulation: any vowel right before a nasal consonant in any language
            if (ev.is_vowel or info.manner in ("vowel", "diphthong")) and idx + 1 < len(events):
                nxt = events[idx + 1]
                if (not nxt.is_pause) and int(nxt.start_fr) - b <= 1 and float(nxt.pause_prev_ms) < 20.0:
                    nxt_info = lookup_ipa_info(nxt.ipa)
                    if nxt_info.manner == "nasal":
                        tmpl = _NASAL_TEMPLATES.get(nxt_info.place, _NASAL_TEMPLATES["alveolar"])
                        length = b - a
                        u = np.linspace(0.0, 1.0, length) if length > 1 else np.array([1.0])
                        w = 0.42 * np.clip((u - 0.68) / 0.32, 0.0, 1.0)
                        nz = w > 0.0
                        if np.any(nz):
                            ctrl.log_env[a:b][nz] = np.clip(
                                ctrl.log_env[a:b][nz] + w[nz, None] * tmpl[None, :],
                                -16.0,
                                2.5,
                            )
                            n_coart += 1

        return {"rule": self.name, "nasal_phones": n_nasals, "coartic_vowels": n_coart}


class IPAObstruentCrispnessRule:
    """Universal voiceless/voiced obstruent clarity, low-rumble gating & burst preservation."""

    name = "ipa_obstruent_crispness"

    def apply(self, events: Sequence[PhoneEvent], ctrl: LarynxControl) -> dict[str, Any]:
        n_fr = len(ctrl.f0_hz)
        if n_fr == 0:
            return {"rule": self.name, "unvoiced_obstruents": 0, "voiced_obstruents": 0}

        n_uv = 0
        n_vo = 0
        vm_all = np.asarray(ctrl.voiced, dtype=bool) & (np.asarray(ctrl.f0_hz, dtype=np.float64) > 40.0)
        if np.any(vm_all):
            v_rms = 10.0 * np.log10(np.mean(np.exp(2.0 * ctrl.log_env[vm_all]), axis=1) + 1e-18)
            v_med_db = float(np.median(v_rms))
        else:
            v_med_db = -21.0
        uv_cap_db = v_med_db - 2.5

        for ev in events:
            if ev.is_pause:
                continue
            a, b = _clipped_event_span(ev, n_fr)
            if b <= a:
                continue
            info = lookup_ipa_info(ev.ipa)
            if info.manner not in _OBSTRUENT_MANNERS:
                continue

            if not info.voiced:
                # Enforce clean unvoiced excitation so pitch leakage cannot buzz inside stops/fricatives
                ctrl.voiced[a:b] = False
                ctrl.f0_hz[a:b] = 0.0
                ctrl.bap[a:b] = np.maximum(ctrl.bap[a:b], 0.96)
                # Suppress sub-380 Hz low rumble in unvoiced noise envelopes (-3.2 dB * cutoff)
                ctrl.log_env[a:b] = np.clip(
                    ctrl.log_env[a:b] - (3.2 * _DB_TO_NEPER) * _VOICELESS_RUMBLE_CUT[None, :],
                    -16.0,
                    2.5,
                )
                length = b - a
                if info.manner == "stop" and length >= 4:
                    u = np.linspace(0.0, 1.0, length)
                    # Clean physical stop closure attenuation before burst release
                    ramp = np.clip((u - 0.40) / 0.14, 0.0, 1.0)
                    clos_w = 0.5 * (1.0 + np.cos(math.pi * ramp))
                    ctrl.log_env[a:b] = np.clip(
                        ctrl.log_env[a:b] - (11.0 * _DB_TO_NEPER) * clos_w[:, None],
                        -16.0,
                        2.5,
                    )
                    burst = np.exp(-0.5 * ((u - 0.64) / 0.16) ** 2)
                    hf = (_FREQS >= 1400.0).astype(np.float64)
                    ctrl.log_env[a:b] = np.clip(
                        ctrl.log_env[a:b] + (1.6 * _DB_TO_NEPER) * burst[:, None] * hf[None, :],
                        -16.0,
                        2.5,
                    )
                elif info.manner == "affricate" and length >= 4:
                    u = np.linspace(0.0, 1.0, length)
                    burst = np.exp(-0.5 * ((u - 0.62) / 0.18) ** 2)
                    hf = (_FREQS >= 1400.0).astype(np.float64)
                    ctrl.log_env[a:b] = np.clip(
                        ctrl.log_env[a:b] + (1.1 * _DB_TO_NEPER) * burst[:, None] * hf[None, :],
                        -16.0,
                        2.5,
                    )

                # Soft per-frame peak guard preventing loud static hiss bursts on fricatives/aspirates
                frm_db = 10.0 * np.log10(np.mean(np.exp(2.0 * ctrl.log_env[a:b]), axis=1) + 1e-18)
                over = frm_db - uv_cap_db
                hi_m = over > 0.0
                if np.any(hi_m):
                    ctrl.log_env[a:b][hi_m] = np.clip(
                        ctrl.log_env[a:b][hi_m] - (over[hi_m, None] * _DB_TO_NEPER),
                        -16.0,
                        2.5,
                    )
                n_uv += 1
            else:
                # Voiced obstruent: soft upper-band closure on voiced stops + clean voicing bar + BAP ceiling
                length = b - a
                if info.manner == "stop" and length >= 4:
                    u = np.linspace(0.0, 1.0, length)
                    ramp = np.clip((u - 0.42) / 0.14, 0.0, 1.0)
                    clos_w = 0.5 * (1.0 + np.cos(math.pi * ramp))
                    ctrl.log_env[a:b] = np.clip(
                        ctrl.log_env[a:b] - (7.0 * _DB_TO_NEPER) * clos_w[:, None] * _VOICED_STOP_HI_MASK[None, :],
                        -16.0,
                        2.5,
                    )
                vm = ctrl.voiced[a:b]
                if np.any(vm):
                    ctrl.bap[a:b][vm] = np.clip(
                        ctrl.bap[a:b][vm],
                        _VOICED_OBST_BAP_FLOOR[None, :],
                        _VOICED_OBST_BAP_CEIL[None, :],
                    )
                    ctrl.log_env[a:b][vm] = np.clip(
                        ctrl.log_env[a:b][vm] + 0.50 * _VOICED_OBST_LO_WARM[None, :],
                        -16.0,
                        2.5,
                    )
                n_vo += 1

        return {"rule": self.name, "unvoiced_obstruents": n_uv, "voiced_obstruents": n_vo}


class IPACoarticulationAndSwitchRule:
    """Adaptive cep[0:18] S-curve spectral bridging across intra-lingual and code-switched seams."""

    name = "ipa_coarticulation_switch"

    def apply(
        self,
        events: Sequence[PhoneEvent],
        ctrl: LarynxControl,
        *,
        jump_thresh_nep: float = 0.35,
    ) -> dict[str, Any]:
        n_fr = len(ctrl.f0_hz)
        if n_fr < 4 or len(events) < 2:
            return {"rule": self.name, "seams_bridged": 0, "switches_bridged": 0}

        # Copy immutable base log_env so multi-seam bridges compose cleanly without smearing
        base_env = np.asarray(ctrl.log_env, dtype=np.float64).copy()
        d_cep18 = np.zeros((n_fr, _CEP_BRIDGE_K), dtype=np.float64)
        mask = np.zeros(n_fr, dtype=bool)
        seams = 0
        switches = 0

        for idx in range(len(events) - 1):
            ev = events[idx]
            nxt = events[idx + 1]
            if ev.is_pause or nxt.is_pause:
                continue
            a, b = _clipped_event_span(ev, n_fr)
            na, nb = _clipped_event_span(nxt, n_fr)
            if (b - a) < 3 or (nb - na) < 3:
                continue
            gap_fr = na - b
            if gap_fr < 0 or gap_fr > 2 or float(ev.pause_after_ms) >= 25.0 or float(nxt.pause_prev_ms) >= 25.0:
                continue

            cur_info = lookup_ipa_info(ev.ipa)
            nxt_info = lookup_ipa_info(nxt.ipa)
            if cur_info.voiceless_obstruent and nxt_info.voiceless_obstruent:
                continue

            is_switch = ev.lang != nxt.lang
            both_sonorant = (cur_info.manner in _SONORANT_MANNERS) and (nxt_info.manner in _SONORANT_MANNERS)
            has_vl_stop_affr = (
                (cur_info.voiceless_obstruent and cur_info.manner in ("stop", "affricate"))
                or (nxt_info.voiceless_obstruent and nxt_info.manner in ("stop", "affricate"))
            )
            if not both_sonorant and not is_switch:
                if ev.same_word_next and not has_vl_stop_affr:
                    half_w = 2
                    blend_max = 0.45
                else:
                    # Consonant <-> vowel transitions stay crisp (±1 frame micro-bridge max)
                    half_w = 1
                    blend_max = 0.32
            elif is_switch and both_sonorant:
                half_w = 4
                blend_max = 0.62
            elif is_switch:
                half_w = 2
                blend_max = 0.42
            else:
                half_w = 3
                blend_max = 0.52

            bnd = na
            i0 = max(a + 1, bnd - half_w)
            i1 = min(nb - 1, bnd + half_w)
            if i1 <= i0 + 1:
                continue

            left_idx = max(0, i0 - 1)
            right_idx = min(n_fr - 1, i1)
            span_env = np.vstack([
                base_env[left_idx : left_idx + 1],
                base_env[i0:i1],
                base_env[right_idx : right_idx + 1],
            ])
            span_cep = span_env @ _CEP_DCT18
            jump = float(np.sqrt(np.sum((span_cep[-1] - span_cep[0]) ** 2) / N_BINS))
            if jump < jump_thresh_nep:
                continue

            m = i1 - i0
            u_s = np.linspace(1.0 / (m + 1), m / (m + 1), m)
            alpha = 0.5 * (1.0 - np.cos(math.pi * u_s))[:, None]
            target_cep = (1.0 - alpha) * span_cep[0:1] + alpha * span_cep[-1:]
            blend = float(np.clip(0.32 + (blend_max - 0.32) * (jump - jump_thresh_nep) / 0.45, 0.32, blend_max))
            d_cep18[i0:i1] += blend * (target_cep - span_cep[1:-1])
            mask[i0:i1] = True
            seams += 1
            if is_switch:
                switches += 1

        if np.any(mask):
            ctrl.log_env[mask] = np.clip(ctrl.log_env[mask] + d_cep18[mask] @ _CEP_DCT18.T, -16.0, 2.5)

        return {"rule": self.name, "seams_bridged": seams, "switches_bridged": switches}


def apply_multilingual_native_rules(
    events: Sequence[PhoneEvent],
    ctrl: LarynxControl,
    speaker_f0_med_hz: float,
    *,
    expressive_drive: float = 0.0,
) -> tuple[LarynxControl, dict[str, Any]]:
    """Run the 4-rule universal IPA physical stack on a native multilingual LarynxControl."""
    glottal = IPAGlottalAntiBuzzRule()
    nasal = IPANasalAntiresonanceRule()
    obst = IPAObstruentCrispnessRule()
    coart = IPACoarticulationAndSwitchRule()

    s1 = glottal.apply(events, ctrl, speaker_f0_med_hz, expressive_drive=expressive_drive)
    s2 = nasal.apply(events, ctrl)
    s3 = obst.apply(events, ctrl)
    s4 = coart.apply(events, ctrl)

    return ctrl, {
        "rules": [s1, s2, s3, s4],
        "n_events": len(events),
    }
