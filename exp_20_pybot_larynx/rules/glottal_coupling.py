"""Rule 1: Stage 4 anti-buzz glottal coupling (band-matched BAP, energy-coupled BAP, softer rd, REVERSE jitter/shimmer)."""
from __future__ import annotations

import numpy as np

from exp_20_pybot_larynx import N_BAP
from exp_20_pybot_larynx.arbiter import (
    NEPER_TO_DB,
    OBSTRUENT_MANNERS,
    RuleProposal,
    UtteranceContext,
    classify_onset_manner,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl

# Band-matched target voiced BAP across [500, 1500, 3000, 6000, 9000, 10900] Hz
# calibrated on Stage 4 hybrid low-crest kernel to match Real 1-7 kHz HNR (~14.5 / 9.5 / 7.2 dB).
_PHYS_BAP_TARGET = np.asarray([0.044, 0.25, 0.46, 0.74, 0.88, 0.93], dtype=np.float64)
_VOT_EDGE_FLOOR = np.asarray([0.65, 0.75, 0.82, 0.88, 0.92, 0.95], dtype=np.float64)


class RuleGlottalF0Coupling:
    """Couples per-band BAP, syllable-energy dynamics, softer LF rd, VOT ramping, tilt, and jitter/shimmer."""

    name = "glottal_f0_coupling"

    def propose(self, ctx: UtteranceContext, base: LarynxControl) -> RuleProposal | None:
        n_fr = len(base.f0_hz)
        if n_fr == 0:
            return None
        f0 = np.asarray(base.f0_hz, dtype=np.float64)
        voiced = np.asarray(base.voiced, dtype=bool)
        mask = voiced & (f0 > 50.0)
        if not np.any(mask):
            return None

        tone_pf = np.zeros(n_fr, dtype=np.int32)
        pos_pf = np.zeros(n_fr, dtype=np.int32)   # 0=medial, 1=initial, 2=utterance_final
        pfin_pf = np.zeros(n_fr, dtype=bool)      # phrase-final syllable
        syl_len_pf = np.zeros(n_fr, dtype=np.int32)
        obs_edge_w = np.zeros(n_fr, dtype=np.float64)

        n_syl = len(ctx.syllables)
        for idx_s, s in enumerate(ctx.syllables):
            a = max(0, min(n_fr, int(s.start_fr)))
            b = max(a, min(n_fr, int(s.end_fr)))
            sp = max(a, min(b, int(s.split_fr)))
            if b <= a:
                continue
            tone_pf[a:b] = int(s.tone)
            syl_len_pf[a:b] = b - a
            is_last = idx_s == n_syl - 1 and not ctx.continues_after
            if idx_s == 0:
                pos_pf[a:b] = 1
            elif is_last:
                pos_pf[a:b] = 2
            if is_last or s.prosodic_break >= 2 or s.pause_after_ms >= 50.0:
                pfin_pf[a:b] = True

            # Unvoiced -> voiced VOT release ramping over first 2-3 voiced frames
            if classify_onset_manner(s.onset) in OBSTRUENT_MANNERS:
                v_rel = np.where(mask[a:b])[0]
                if v_rel.size > 0:
                    v0 = a + int(v_rel[0])
                    if v0 > a or idx_s == 0:
                        n_fin = max(1, b - sp)
                        max_ramp_end = min(b, max(sp, v0) + max(2, min(3, int(round(0.15 * n_fin)) + 2)))
                        for k, rw in enumerate((1.0, 0.55, 0.22)):
                            fk = v0 + k
                            if fk < max_ramp_end and mask[fk]:
                                obs_edge_w[fk] = max(obs_edge_w[fk], rw)

        # Generic unvoiced -> voiced edge transition smoothing across syllable boundaries
        uv_to_v = np.where(mask & np.concatenate([[False], ~mask[:-1]]))[0]
        for v0 in uv_to_v:
            if obs_edge_w[v0] == 0.0:
                for k, rw in enumerate((0.65, 0.30)):
                    fk = v0 + k
                    if fk < n_fr and mask[fk]:
                        obs_edge_w[fk] = max(obs_edge_w[fk], rw)

        ref_f0 = max(50.0, float(ctx.speaker_f0_med_hz))
        rel_st = np.zeros(n_fr, dtype=np.float64)
        rel_st[mask] = 12.0 * np.log2(f0[mask] / ref_f0)

        rst_m = rel_st[mask]
        t_m = tone_pf[mask]
        pos_m = pos_pf[mask]
        pfin_m = pfin_pf[mask]
        u_m = np.asarray(ctx.phase_u, dtype=np.float64)[mask]
        in_fin = ~np.asarray(ctx.is_onset, dtype=bool)[mask]
        len_m = syl_len_pf[mask]
        edge_m = obs_edge_w[mask]

        # 1. Per-band physical BAP modulation conditioned on F0, tone, position, energy, phase, and length
        reg_mul = np.clip(1.0 - 0.045 * np.clip(rst_m, -5.0, 6.0), 0.72, 1.25)

        tone_mul = np.ones_like(rst_m)
        tone_mul[t_m == 1] = 0.85
        tone_mul[t_m == 4] = 0.92
        tone_mul[t_m == 5] = 1.08
        tone_mul[t_m == 3] = 1.12

        pos_mul = np.ones_like(rst_m)
        pos_mul[pos_m == 1] *= 0.88
        pos_mul[pos_m == 2] *= 1.08
        nuc_m = in_fin & (u_m >= 0.20) & (u_m <= 0.65)
        pos_mul[nuc_m] *= 0.90

        tail_w = np.clip((u_m - 0.75) / 0.25, 0.0, 1.0)
        pfin_tail = in_fin & pfin_m & (tail_w > 0.0)
        pos_mul[pfin_tail] *= 1.0 + 0.30 * tail_w[pfin_tail]

        # Dynamic syllable/frame energy-coupled BAP: quiet tails breathier, bright vowel cores solid
        log_env = np.asarray(base.log_env, dtype=np.float64)
        frame_e = np.mean(log_env[:, 8:170], axis=1)
        med_e = float(np.median(frame_e[mask]))
        rel_e_db = (frame_e[mask] - med_e) * NEPER_TO_DB
        e_mul = np.clip(1.0 - 0.025 * np.clip(rel_e_db, -8.0, 6.0), 0.85, 1.25)
        pos_mul *= e_mul

        # Long-vowel naturalness (syllables >= 35 frames / 175 ms): gentle mid/late breathiness swell
        long_m = in_fin & (len_m >= 35)
        long_breath = np.zeros_like(rst_m)
        if np.any(long_m):
            ul = u_m[long_m]
            long_breath[long_m] = (
                0.10 * np.sin(np.pi * np.clip((ul - 0.20) / 0.75, 0.0, 1.0))
                + 0.08 * np.clip((ul - 0.65) / 0.35, 0.0, 1.0)
            )
            pos_mul[long_m] *= 1.0 + long_breath[long_m]

        mod_band = np.stack(
            [
                reg_mul * tone_mul * pos_mul,
                reg_mul * tone_mul * pos_mul,
                (0.65 + 0.35 * reg_mul) * (0.65 + 0.35 * tone_mul) * (0.60 + 0.40 * pos_mul),
                (0.75 + 0.25 * reg_mul) * (0.75 + 0.25 * pos_mul),
                0.88 + 0.12 * pos_mul,
                0.92 + 0.08 * pos_mul,
            ],
            axis=1,
        )

        phys_target = np.clip(_PHYS_BAP_TARGET[None, :] * mod_band, 0.008, 0.96)
        base_bap_m = np.asarray(base.bap, dtype=np.float64)[mask]

        # Self-normalizing blend: converges cleanly toward band-matched target from any KB
        alpha = np.where(base_bap_m > phys_target, 0.85, 0.65)
        target_bap = np.clip(alpha * phys_target + (1.0 - alpha) * base_bap_m, 0.008, 0.96)

        if np.any(edge_m > 0.0):
            ew = edge_m[:, None]
            target_bap = np.maximum(target_bap, (1.0 - ew) * target_bap + ew * _VOT_EDGE_FLOOR[None, :])

        d_bap = np.zeros((n_fr, N_BAP), dtype=np.float64)
        d_bap[mask] = target_bap - base_bap_m

        # 2. Gentle register/position spectral tilt + VOT release tilt ramp (dB/oct)
        d_tilt = np.zeros(n_fr, dtype=np.float64)
        tilt_val = 0.02 * np.clip(rst_m, -5.0, 5.0)
        tilt_val[pos_m == 1] += 0.04
        tilt_val[pos_m == 2] -= 0.06
        tilt_val[pfin_tail] -= 0.06 * tail_w[pfin_tail]
        if np.any(edge_m > 0.0):
            tilt_val -= 0.80 * edge_m
        d_tilt[mask] = tilt_val - base.glottal_tilt_db_per_oct[mask]

        # 3. REVERSE_REPORT-aligned jitter & shimmer (~0.0075 / 0.048 base) so electric spikes vanish
        d_jit = np.zeros(n_fr, dtype=np.float64)
        t_jit = np.clip(0.0075 - 0.00035 * rst_m, 0.0035, 0.0110)
        t_jit[t_m == 3] = np.minimum(0.0125, t_jit[t_m == 3] + 0.0018)
        t_jit[pfin_tail] = np.minimum(0.0135, t_jit[pfin_tail] + 0.0022 * tail_w[pfin_tail])
        if np.any(long_m):
            t_jit[long_m] = np.minimum(0.0135, t_jit[long_m] + 0.0040 * long_breath[long_m])
        d_jit[mask] = t_jit - base.jitter_frac[mask]

        d_shim = np.zeros(n_fr, dtype=np.float64)
        t_shim = np.clip(0.0480 - 0.0020 * rst_m, 0.0240, 0.0720)
        t_shim[t_m == 3] = np.minimum(0.0800, t_shim[t_m == 3] + 0.0100)
        t_shim[pfin_tail] = np.minimum(0.0880, t_shim[pfin_tail] + 0.0140 * tail_w[pfin_tail])
        if np.any(long_m):
            t_shim[long_m] = np.minimum(0.0880, t_shim[long_m] + 0.0240 * long_breath[long_m])
        d_shim[mask] = t_shim - base.shimmer_frac[mask]

        # 4. Softer dynamic LF waveshape rd in [0.92, 1.75] (avoids spiky pressed closure < 0.92)
        base_rd = np.ones(n_fr, dtype=np.float64) if base.rd is None else np.asarray(base.rd, dtype=np.float64)
        target_rd = 1.08 - 0.025 * np.clip(rst_m, -5.0, 6.0)
        target_rd[t_m == 1] -= 0.08
        target_rd[t_m == 4] -= 0.05
        target_rd[t_m == 3] += 0.18
        target_rd[t_m == 5] += 0.12
        target_rd[pos_m == 1] -= 0.06
        target_rd[pos_m == 2] += 0.10
        target_rd[nuc_m] -= 0.04
        target_rd[pfin_tail] += 0.32 * tail_w[pfin_tail]
        if np.any(long_m):
            target_rd[long_m] += 1.00 * long_breath[long_m]
        target_rd = np.clip(target_rd, 0.92, 1.75)

        d_rd = np.zeros(n_fr, dtype=np.float64)
        d_rd[mask] = target_rd - base_rd[mask]

        return RuleProposal(
            module_name=self.name,
            mask=mask,
            delta_bap=d_bap,
            delta_tilt=d_tilt,
            delta_jitter=d_jit,
            delta_shimmer=d_shim,
            delta_rd=d_rd,
        )
