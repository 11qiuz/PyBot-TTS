"""Rule 6 (Stage 5): Diphthong glide contrast, apical/rhotic sharpening, and neutral-tone reduction."""
from __future__ import annotations

import numpy as np

from exp_20_pybot_larynx import N_BINS
from exp_20_pybot_larynx.arbiter import (
    RuleProposal,
    UtteranceContext,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env, log_env_to_cep

DIPHTHONG_TRIPHTHONG_FINALS = frozenset({
    "ai", "ei", "ao", "ou",
    "ia", "ie", "iao", "iu", "iou",
    "ua", "uo", "uai", "ui", "uei",
    "ve", "üe", "ue",
    "ian", "uan", "van", "üan",
    "iang", "uang",
})

APICAL_RHOTIC_FINALS = frozenset({
    "ir", "iz", "ii", "ih", "i_z", "i_zh", "er",
})

_GLIDE_Q0 = 8
_GLIDE_Q1 = 30
_GLIDE_BELL = 0.5 * (
    1.0 - np.cos(2.0 * np.pi * (np.arange(_GLIDE_Q0, _GLIDE_Q1, dtype=np.float64) - (_GLIDE_Q0 - 1)) / (_GLIDE_Q1 - _GLIDE_Q0 + 1))
)

_APICAL_Q0 = 10
_APICAL_Q1 = 34
_APICAL_BELL = 0.5 * (
    1.0 - np.cos(2.0 * np.pi * (np.arange(_APICAL_Q0, _APICAL_Q1, dtype=np.float64) - (_APICAL_Q0 - 1)) / (_APICAL_Q1 - _APICAL_Q0 + 1))
)


class RuleFormantDynamics:
    """Enhances diphthong glide speed, apical/rhotic resonance clarity, and reduced-vowel damping."""

    name = "formant_dynamics"

    def __init__(
        self,
        *,
        glide_strength: float = 0.06,
        apical_strength: float = 0.10,
        reduction_damp: float = 0.05,
    ) -> None:
        self.glide_strength = float(glide_strength)
        self.apical_strength = float(apical_strength)
        self.reduction_damp = float(reduction_damp)

    def propose(self, ctx: UtteranceContext, base: LarynxControl) -> RuleProposal | None:
        n_fr = len(base.f0_hz)
        if n_fr == 0 or not ctx.syllables:
            return None

        voiced = np.asarray(base.voiced, dtype=bool)
        f0 = np.asarray(base.f0_hz, dtype=np.float64)
        v_mask = voiced & (f0 > 50.0)
        u_arr = np.asarray(ctx.phase_u, dtype=np.float64)

        mask = np.zeros(n_fr, dtype=bool)
        d_cep = np.zeros((n_fr, 36), dtype=np.float64)
        d_tilt = np.zeros(n_fr, dtype=np.float64)

        # Compute base cepstrum once over active voiced frames only (fast batch DCT)
        active_slices: list[tuple[int, int, str, int, float]] = []
        need_cep_mask = np.zeros(n_fr, dtype=bool)

        for s in ctx.syllables:
            a = max(0, min(n_fr, int(s.start_fr)))
            b = max(a, min(n_fr, int(s.end_fr)))
            sp = max(a, min(b, int(s.split_fr)))
            f_a = sp if (s.onset and sp > a) else a
            if b - f_a < 4:
                continue
            f_str = str(s.final).strip()
            t = int(s.tone)
            dur_ms = float(b - a) * 5.0
            is_diph = f_str in DIPHTHONG_TRIPHTHONG_FINALS
            is_apic = f_str in APICAL_RHOTIC_FINALS
            is_red = (t == 5) or (dur_ms < 110.0)
            if is_diph or is_apic or is_red:
                active_slices.append((f_a, b, f_str, t, dur_ms))
                need_cep_mask[f_a:b] = v_mask[f_a:b]

        if not np.any(need_cep_mask):
            return None

        cep_sub = np.zeros((n_fr, 36), dtype=np.float64)
        cep_sub[need_cep_mask] = log_env_to_cep(base.log_env[need_cep_mask], n_cep=36)

        for f_a, b, f_str, t, dur_ms in active_slices:
            seg_v = v_mask[f_a:b]
            if int(np.sum(seg_v)) < 3:
                continue
            idx_v = np.arange(f_a, b)[seg_v]
            u_v = u_arr[idx_v]

            if f_str in DIPHTHONG_TRIPHTHONG_FINALS and self.glide_strength > 0.0:
                w_g = np.sin(np.pi * np.clip((u_v - 0.12) / 0.76, 0.0, 1.0)) ** 1.15
                c_seg = cep_sub[idx_v, _GLIDE_Q0:_GLIDE_Q1]
                mu = np.mean(c_seg, axis=0, keepdims=True)
                delta = (c_seg - mu) * ((self.glide_strength * w_g)[:, None] * _GLIDE_BELL[None, :])
                d_cep[idx_v, _GLIDE_Q0:_GLIDE_Q1] += delta
                mask[idx_v] = True

            if f_str in APICAL_RHOTIC_FINALS and self.apical_strength > 0.0:
                w_a = np.sin(np.pi * np.clip((u_v - 0.10) / 0.80, 0.0, 1.0)) ** 0.85
                c_seg = cep_sub[idx_v, _APICAL_Q0:_APICAL_Q1]
                d_cep[idx_v, _APICAL_Q0:_APICAL_Q1] += c_seg * ((self.apical_strength * w_a)[:, None] * _APICAL_BELL[None, :])
                mask[idx_v] = True

            if (t == 5 or dur_ms < 110.0) and self.reduction_damp > 0.0:
                red_scale = 1.0 if t == 5 else 0.55
                w_r = np.sin(np.pi * np.clip(u_v, 0.05, 0.95)) ** 0.6
                d_cep[idx_v, 10:28] -= (self.reduction_damp * red_scale * w_r)[:, None] * cep_sub[idx_v, 10:28]
                d_tilt[idx_v] -= (0.35 if t == 5 else 0.18) * w_r
                mask[idx_v] = True

        if not np.any(mask):
            return None

        # Ensure zero cep[0] modification so energy balance stays strictly preserved
        d_cep[:, 0] = 0.0
        d_env = np.zeros((n_fr, N_BINS), dtype=np.float64)
        d_env[mask] = cep_to_log_env(d_cep[mask], n_bins=N_BINS)

        return RuleProposal(
            module_name=self.name,
            mask=mask,
            delta_log_env=d_env,
            delta_tilt=d_tilt,
        )
