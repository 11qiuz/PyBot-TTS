"""Unit tests for Stage 3 PyBot-Larynx kernel (`larynx/synth.py` and `arbiter.py`)."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import unittest

import numpy as np
from scipy.fft import rfft, rfftfreq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import HOP, N_BAP, N_BINS, N_FFT, SR  # noqa: E402
from exp_20_pybot_larynx.arbiter import (  # noqa: E402
    Arbiter,
    RuleProposal,
    SyllableSpan,
    UtteranceContext,
    build_utterance_context,
)
from exp_20_pybot_larynx.judge import compute_voiced_hnr_db  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import (  # noqa: E402
    LarynxControl,
    analytic_lf_phasor,
    synthesize_larynx,
)
from exp_v9_eval.v9_metrics import compute_v9_extended_metrics  # noqa: E402


def _make_vowel_env(freqs: np.ndarray, f1: float = 650.0, f2: float = 1450.0, boost_sub: float = 0.0) -> np.ndarray:
    env = (
        -2.0
        - 0.90 * np.log1p((freqs / 1100.0) ** 2)
        + 1.6 * np.exp(-0.5 * ((freqs - f1) / 140.0) ** 2)
        + 1.1 * np.exp(-0.5 * ((freqs - f2) / 190.0) ** 2)
    )
    if boost_sub > 0.0:
        env += boost_sub * np.exp(-0.5 * (freqs / 80.0) ** 2)
    return env


class _DynamicRdRule:
    name = "dynamic_rd_test"

    def __init__(self, delta_vals: np.ndarray, weight: float = 1.0, use_masked: bool = False) -> None:
        self._delta_vals = delta_vals
        self._weight = weight
        self._use_masked = use_masked

    def propose(self, ctx: UtteranceContext, base: LarynxControl) -> RuleProposal | None:
        mask = base.voiced.copy()
        d = self._delta_vals[mask] if self._use_masked else self._delta_vals
        return RuleProposal(module_name=self.name, mask=mask, delta_rd=d, weight=self._weight)


class TestLarynxKernelStage3(unittest.TestCase):
    def test_backward_compatible_defaults_and_copy(self) -> None:
        n_fr = 60
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        env = np.repeat(_make_vowel_env(freqs)[None, :], n_fr, axis=0)
        f0 = np.full(n_fr, 220.0, dtype=np.float64)
        voi = np.ones(n_fr, dtype=bool)
        bap = np.full((n_fr, N_BAP), 0.1, dtype=np.float64)

        ctrl = LarynxControl(f0_hz=f0, voiced=voi, log_env=env, bap=bap, n_samples=n_fr * HOP)
        self.assertIsNotNone(ctrl.rd)
        self.assertEqual(ctrl.rd.shape, (n_fr,))
        self.assertTrue(np.allclose(ctrl.rd, 1.0))

        ctrl_copy = ctrl.copy()
        ctrl_copy.rd[0] = 1.8
        self.assertAlmostEqual(float(ctrl.rd[0]), 1.0)

        with self.assertRaises(ValueError):
            LarynxControl(f0_hz=f0, voiced=voi, log_env=env, bap=bap, rd=np.ones(n_fr + 1))

        # All glottal modes & scalar rd fallback work cleanly
        coh = analytic_lf_phasor(n_fft=N_FFT, sr=SR, rd=1.15)
        for gmode in ("parametric_lf", "speaker_coherent", "zero_phase"):
            y = synthesize_larynx(
                ctrl,
                glottal_mode=gmode,
                coherent_phasor=coh,
                phase_dispersion=(gmode != "zero_phase"),
                rd=1.4,
                seed=11,
            )
            self.assertEqual(len(y), n_fr * HOP)
            self.assertTrue(np.all(np.isfinite(y)))

    def test_dc_and_sub_f0_removal_on_voiced_pulses(self) -> None:
        n_fr = 120
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        # Inject strong DC/sub-F0 energy (+3.5 neper ~ +30 dB) into log_env
        env = np.repeat(_make_vowel_env(freqs, boost_sub=3.5)[None, :], n_fr, axis=0)
        f0 = np.full(n_fr, 240.0, dtype=np.float64)
        voi = np.ones(n_fr, dtype=bool)
        bap = np.zeros((n_fr, N_BAP), dtype=np.float64)

        ctrl = LarynxControl(
            f0_hz=f0,
            voiced=voi,
            log_env=env,
            bap=bap,
            jitter_frac=np.zeros(n_fr),
            shimmer_frac=np.zeros(n_fr),
            n_samples=n_fr * HOP,
        )
        y = synthesize_larynx(ctrl, glottal_mode="parametric_lf", phase_dispersion=False, seed=7)
        # DC offset must be virtually zero
        rms = float(np.sqrt(np.mean(y ** 2)))
        self.assertGreater(rms, 1e-4)
        self.assertLess(abs(float(np.mean(y))) / rms, 1e-3)

        # Sub-F0 power (< 0.65 * 240 = 156 Hz) must be deeply suppressed vs F0 band (200..280 Hz)
        seg = y[20 * HOP : 100 * HOP]
        spec_p = np.abs(rfft(seg * np.hanning(len(seg)))) ** 2
        f_seg = rfftfreq(len(seg), 1.0 / SR)
        p_sub = float(np.sum(spec_p[f_seg < 150.0]))
        p_f0 = float(np.sum(spec_p[(f_seg >= 200.0) & (f_seg <= 280.0)]))
        ratio_db = 10.0 * math.log10((p_f0 + 1e-18) / (p_sub + 1e-18))
        self.assertGreater(ratio_db, 28.0)

        hnr = compute_voiced_hnr_db(y, sr=SR, f0_hz=f0, voiced=voi)
        self.assertIsNotNone(hnr)
        self.assertGreater(float(hnr), 21.0)

        # At higher F0 (320 Hz) where 0.35*F0 exceeds the 480-pt Hann mainlobe, HNR must exceed 28 dB
        ctrl_hi = LarynxControl(
            f0_hz=np.full(n_fr, 320.0, dtype=np.float64),
            voiced=voi,
            log_env=env,
            bap=bap,
            jitter_frac=np.zeros(n_fr),
            shimmer_frac=np.zeros(n_fr),
            n_samples=n_fr * HOP,
        )
        y_hi = synthesize_larynx(ctrl_hi, glottal_mode="parametric_lf", phase_dispersion=False, seed=7)
        hnr_hi = compute_voiced_hnr_db(y_hi, sr=SR, f0_hz=ctrl_hi.f0_hz, voiced=voi)
        self.assertIsNotNone(hnr_hi)
        self.assertGreater(float(hnr_hi), 28.0)

    def test_no_forced_hf_floor_when_bap_is_clean(self) -> None:
        n_fr = 120
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        env = np.repeat(_make_vowel_env(freqs)[None, :], n_fr, axis=0)
        f0 = np.full(n_fr, 320.0, dtype=np.float64)
        voi = np.ones(n_fr, dtype=bool)
        bap = np.zeros((n_fr, N_BAP), dtype=np.float64)

        ctrl = LarynxControl(
            f0_hz=f0,
            voiced=voi,
            log_env=env,
            bap=bap,
            jitter_frac=np.zeros(n_fr),
            shimmer_frac=np.zeros(n_fr),
            n_samples=n_fr * HOP,
        )
        y = synthesize_larynx(ctrl, glottal_mode="parametric_lf", phase_dispersion=True, seed=13)
        # Check HF (>3 kHz) inter-harmonic cleanliness directly on a long Hann window
        seg = y[20 * HOP : 100 * HOP]
        spec_p = np.abs(rfft(seg * np.hanning(len(seg)))) ** 2
        f_seg = rfftfreq(len(seg), 1.0 / SR)
        hf_m = (f_seg >= 3000.0) & (f_seg <= 9000.0)
        harm_dist = np.abs(f_seg[hf_m] - np.round(f_seg[hf_m] / 320.0) * 320.0)
        e_h = float(np.sum(spec_p[hf_m][harm_dist <= 30.0]))
        e_i = float(np.sum(spec_p[hf_m][harm_dist >= 60.0]))
        hf_hnr_db = 10.0 * math.log10((e_h + 1e-18) / (e_i + 1e-18))
        self.assertGreater(hf_hnr_db, 28.0)

    def test_y_ap_480_cola_unvoiced_fidelity_and_rms_balance(self) -> None:
        n_fr = 240  # 1.2s: first half voiced, second half unvoiced with identical flat spectrum
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        flat_env = np.full((n_fr, N_BINS), -2.5, dtype=np.float64)
        # Add a shared 4.2 kHz fricative formant on the unvoiced half and verify spectral recovery
        fr_peak = 2.0 * np.exp(-0.5 * ((freqs - 4200.0) / 280.0) ** 2)
        flat_env[140:220] += fr_peak[None, :]

        f0 = np.zeros(n_fr, dtype=np.float64)
        voi = np.zeros(n_fr, dtype=bool)
        voi[:120] = True
        f0[:120] = 220.0
        bap = np.zeros((n_fr, N_BAP), dtype=np.float64)
        bap[120:] = 1.0

        ctrl = LarynxControl(f0_hz=f0, voiced=voi, log_env=flat_env, bap=bap, n_samples=n_fr * HOP)
        y = synthesize_larynx(ctrl, glottal_mode="parametric_lf", seed=99)
        self.assertTrue(np.all(np.isfinite(y)))

        # Boundary continuity: tail frames must not drop to zero
        uv_end = y[(n_fr - 2) * HOP :]
        self.assertGreater(float(np.sqrt(np.mean(uv_end ** 2))), 1e-3)

        # Unvoiced spectral selectivity: 4.2 kHz peak vs 1.5 kHz trough
        uv_mid = y[150 * HOP : 210 * HOP]
        p_uv = np.abs(rfft(uv_mid * np.hanning(len(uv_mid)))) ** 2
        f_uv = rfftfreq(len(uv_mid), 1.0 / SR)
        e_peak = float(np.mean(p_uv[(f_uv >= 4050.0) & (f_uv <= 4350.0)]))
        e_trough = float(np.mean(p_uv[(f_uv >= 1200.0) & (f_uv <= 1800.0)]))
        self.assertGreater(10.0 * math.log10((e_peak + 1e-18) / (e_trough + 1e-18)), 15.5)

        # Global RMS balance between voiced flat span and unvoiced flat span (120..135)
        ctrl_bal = LarynxControl(
            f0_hz=f0,
            voiced=voi,
            log_env=np.full((n_fr, N_BINS), -2.5, dtype=np.float64),
            bap=bap,
            n_samples=n_fr * HOP,
        )
        y_bal = synthesize_larynx(ctrl_bal, glottal_mode="parametric_lf", seed=101)
        rms_v = float(np.sqrt(np.mean(y_bal[20 * HOP : 100 * HOP] ** 2)))
        rms_uv = float(np.sqrt(np.mean(y_bal[140 * HOP : 220 * HOP] ** 2)))
        bal_db = 20.0 * math.log10((rms_uv + 1e-12) / (rms_v + 1e-12))
        self.assertLess(abs(bal_db), 1.5)

    def test_arbiter_delta_rd_clamping_and_dynamic_rd_lut_speed(self) -> None:
        n_fr = 400  # 2.0s
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        env = np.repeat(_make_vowel_env(freqs)[None, :], n_fr, axis=0)
        f0 = np.zeros(n_fr, dtype=np.float64)
        voi = np.zeros(n_fr, dtype=bool)
        voi[20:380] = True
        f0[20:380] = np.linspace(260.0, 190.0, 360)
        bap = np.full((n_fr, N_BAP), 0.05, dtype=np.float64)

        spans = [SyllableSpan(0, "好", "h", "ao", 3, 0, 20, 400)]
        ctx = build_utterance_context("rd_test", n_fr, spans, speaker_f0_med_hz=230.0)
        base = LarynxControl(f0_hz=f0, voiced=voi, log_env=env, bap=bap, n_samples=n_fr * HOP)

        # Test extreme delta_rd clamping [0.50, 2.20], NaN handling, and masked array support
        d_raw = np.linspace(-1.0, 2.5, n_fr)
        d_raw[30] = float("nan")
        arb = Arbiter()
        fused, trace = arb.apply(ctx, base, [_DynamicRdRule(d_raw, weight=1.0, use_masked=True)])
        self.assertIsNone(trace["modules"][0]["error"])
        self.assertGreater(trace["modules"][0]["mean_abs_rd_delta"], 0.0)
        self.assertGreaterEqual(float(np.min(fused.rd)), 0.50)
        self.assertLessEqual(float(np.max(fused.rd)), 2.20)
        self.assertAlmostEqual(float(fused.rd[30]), 1.0)  # NaN sanitized to 0 delta
        self.assertAlmostEqual(float(fused.rd[0]), 1.0)   # unmasked untouched

        # Dynamic rd synthesis vs static rd=1.0 (warm up LUT/FFT once, take min of 3 runs for CPU-load stability)
        y_static = synthesize_larynx(base, glottal_mode="parametric_lf", seed=55)
        synthesize_larynx(fused, glottal_mode="parametric_lf", seed=55)
        best_dt = float("inf")
        y_dyn = y_static
        for _ in range(3):
            t0 = time.perf_counter()
            y_dyn = synthesize_larynx(fused, glottal_mode="parametric_lf", seed=55)
            best_dt = min(best_dt, time.perf_counter() - t0)
        rtf = best_dt / (len(y_dyn) / SR)

        self.assertEqual(len(y_dyn), n_fr * HOP)
        self.assertTrue(np.all(np.isfinite(y_dyn)))
        self.assertGreater(float(np.sqrt(np.mean((y_dyn - y_static) ** 2))), 1e-4)
        self.assertLess(rtf, 0.065)

        v9 = compute_v9_extended_metrics(y_dyn, sr=SR)
        self.assertLessEqual(float(v9["click_outlier_rate_gt8"]), 0.0006)

    def test_stage4_hybrid_low_crest_and_coh_mix(self) -> None:
        n_fr = 120
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        env = np.repeat(_make_vowel_env(freqs)[None, :], n_fr, axis=0)
        f0 = np.full(n_fr, 235.0, dtype=np.float64)
        voi = np.ones(n_fr, dtype=bool)
        bap = np.full((n_fr, N_BAP), 0.08, dtype=np.float64)
        ctrl = LarynxControl(f0_hz=f0, voiced=voi, log_env=env, bap=bap, n_samples=n_fr * HOP)

        coh = analytic_lf_phasor(n_fft=N_FFT, sr=SR, rd=0.75)
        y_nodisp = synthesize_larynx(ctrl, glottal_mode="parametric_lf", phase_dispersion=False, seed=21)
        y_lf = synthesize_larynx(ctrl, glottal_mode="parametric_lf", disp_strength=4.8, seed=21)
        y_coh = synthesize_larynx(ctrl, glottal_mode="speaker_coherent", coherent_phasor=coh, coh_mix=0.55, disp_strength=4.8, seed=21)
        y_coh0 = synthesize_larynx(ctrl, glottal_mode="speaker_coherent", coherent_phasor=coh, coh_mix=0.0, disp_strength=4.8, seed=21)

        def _crest(sig: np.ndarray) -> float:
            seg = sig[20 * HOP : 100 * HOP]
            return float(np.max(np.abs(seg)) / max(1e-12, float(np.sqrt(np.mean(seg ** 2)))))

        self.assertLess(_crest(y_lf), _crest(y_nodisp))
        self.assertLess(_crest(y_lf), 2.85)
        self.assertLess(_crest(y_coh), 2.95)
        self.assertGreater(float(np.sqrt(np.mean((y_coh - y_coh0) ** 2))), 1e-4)

    def test_stage4_ap_glottal_mod_energy_and_ar1_shimmer_stability(self) -> None:
        n_fr = 200
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        env = np.repeat(_make_vowel_env(freqs)[None, :], n_fr, axis=0)
        f0 = np.zeros(n_fr, dtype=np.float64)
        voi = np.zeros(n_fr, dtype=bool)
        voi[:100] = True
        f0[:100] = 220.0
        bap = np.full((n_fr, N_BAP), 0.55, dtype=np.float64)
        bap[100:] = 1.0

        # Extreme jitter/shimmer clamping & stability check
        ctrl = LarynxControl(
            f0_hz=f0,
            voiced=voi,
            log_env=env,
            bap=bap,
            jitter_frac=np.full(n_fr, 0.015),
            shimmer_frac=np.full(n_fr, 0.12),
            n_samples=n_fr * HOP,
        )
        y_mod = synthesize_larynx(ctrl, ap_glottal_mod=0.55, seed=77)
        y_unmod = synthesize_larynx(ctrl, ap_glottal_mod=0.0, seed=77)

        self.assertTrue(np.all(np.isfinite(y_mod)))
        # Unvoiced half must be identical regardless of ap_glottal_mod
        self.assertTrue(np.allclose(y_mod[115 * HOP :], y_unmod[115 * HOP :]))
        # Voiced half must differ (cycle modulation active) yet preserve RMS energy within 0.5 dB
        v_mod = y_mod[10 * HOP : 90 * HOP]
        v_unmod = y_unmod[10 * HOP : 90 * HOP]
        self.assertGreater(float(np.sqrt(np.mean((v_mod - v_unmod) ** 2))), 1e-4)
        rms_diff_db = 20.0 * math.log10(
            (float(np.sqrt(np.mean(v_mod ** 2))) + 1e-12) / (float(np.sqrt(np.mean(v_unmod ** 2))) + 1e-12)
        )
        self.assertLess(abs(rms_diff_db), 0.5)


if __name__ == "__main__":
    unittest.main()
