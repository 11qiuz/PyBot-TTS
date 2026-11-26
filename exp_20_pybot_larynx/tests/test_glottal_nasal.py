"""Unit tests for RuleGlottalF0Coupling (Stage 4 anti-buzz) and RuleNasalAntiresonance."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.fft import rfftfreq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import HOP, N_BAP, N_BINS, N_FFT, SR  # noqa: E402
from exp_20_pybot_larynx.arbiter import Arbiter, SyllableSpan, build_utterance_context  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import LarynxControl, synthesize_larynx  # noqa: E402
from exp_20_pybot_larynx.rules.glottal_coupling import RuleGlottalF0Coupling  # noqa: E402
from exp_20_pybot_larynx.rules.nasals import RuleNasalAntiresonance, classify_nasal_coda  # noqa: E402


def _make_base_ctrl(n_fr: int, f0_val: float = 240.0, bap_val: float | list[float] = 0.45) -> LarynxControl:
    f0 = np.full(n_fr, f0_val, dtype=np.float64)
    voi = np.ones(n_fr, dtype=bool)
    log_env = np.full((n_fr, N_BINS), -2.0, dtype=np.float64)
    if isinstance(bap_val, (list, tuple, np.ndarray)):
        bap = np.repeat(np.asarray(bap_val, dtype=np.float64)[None, :], n_fr, axis=0)
    else:
        bap = np.full((n_fr, N_BAP), float(bap_val), dtype=np.float64)
    return LarynxControl(f0_hz=f0, voiced=voi, log_env=log_env, bap=bap, n_samples=n_fr * HOP)


class TestGlottalCoupling(unittest.TestCase):
    def test_empty_and_unvoiced_returns_none(self) -> None:
        rule = RuleGlottalF0Coupling()
        ctx_empty = build_utterance_context("empty", 0, [])
        base_empty = LarynxControl(
            f0_hz=np.zeros(0),
            voiced=np.zeros(0, dtype=bool),
            log_env=np.zeros((0, N_BINS)),
            bap=np.zeros((0, N_BAP)),
        )
        self.assertIsNone(rule.propose(ctx_empty, base_empty))

        n_fr = 20
        ctx = build_utterance_context("uv", n_fr, [SyllableSpan(0, "嘶", "s", "iz", 1, 0, 10, 20)])
        base_uv = _make_base_ctrl(n_fr)
        base_uv.voiced[:] = False
        base_uv.f0_hz[:] = 0.0
        self.assertIsNone(rule.propose(ctx, base_uv))

    def test_register_tone_position_phase_and_rd_conditioning(self) -> None:
        rule = RuleGlottalF0Coupling()
        arb = Arbiter()
        spans = [
            SyllableSpan(0, "妈", "m", "a", 1, 0, 5, 30, prosodic_break=0),
            SyllableSpan(1, "马", "m", "a", 3, 30, 35, 60, prosodic_break=0),
            SyllableSpan(2, "买", "m", "ai", 3, 60, 65, 90, prosodic_break=3),
        ]
        ctx = build_utterance_context("cond", 90, spans, speaker_f0_med_hz=240.0)
        base = _make_base_ctrl(90, f0_val=240.0, bap_val=0.45)
        base.f0_hz[0:30] = 310.0   # high register (+4.4 st)
        base.f0_hz[30:60] = 175.0  # low register (-5.5 st)
        base.f0_hz[60:90] = 240.0  # neutral register

        fused, trace = arb.apply(ctx, base, [rule])
        self.assertIsNone(trace["modules"][0]["error"])
        self.assertEqual(trace["modules"][0]["frames_touched"], 90)
        self.assertGreater(trace["modules"][0]["mean_abs_rd_delta"], 0.04)

        # Initial high-F0 T1 must be cleaner & softer-pressed than medial low-F0 T3
        self.assertLess(float(np.mean(fused.bap[10:25, :3])), float(np.mean(fused.bap[40:55, :3])))
        self.assertLess(float(np.mean(fused.jitter_frac[10:25])), float(np.mean(fused.jitter_frac[40:55])))
        self.assertLess(float(np.mean(fused.shimmer_frac[10:25])), float(np.mean(fused.shimmer_frac[40:55])))
        self.assertGreater(float(np.mean(fused.glottal_tilt_db_per_oct[10:25])), float(np.mean(fused.glottal_tilt_db_per_oct[40:55])))
        assert fused.rd is not None
        self.assertTrue(np.all((fused.rd >= 0.92) & (fused.rd <= 1.75)))
        self.assertLess(float(np.mean(fused.rd[10:25])), float(np.mean(fused.rd[40:55])))

        # Phrase-final tail (frames 85..90, u>0.8) must be breathier & laxer than nucleus (70..78)
        self.assertGreater(float(np.mean(fused.bap[85:90, :3])), float(np.mean(fused.bap[70:78, :3])))
        self.assertGreater(float(np.mean(fused.jitter_frac[85:90])), float(np.mean(fused.jitter_frac[70:78])))
        self.assertGreater(float(np.mean(fused.rd[85:90])), float(np.mean(fused.rd[70:78])))

    def test_energy_coupled_bap_quiet_vs_bright(self) -> None:
        rule = RuleGlottalF0Coupling()
        arb = Arbiter()
        spans = [
            SyllableSpan(0, "巴", "b", "a", 1, 0, 5, 30),
            SyllableSpan(1, "巴", "b", "a", 1, 30, 35, 60),
        ]
        ctx = build_utterance_context("energy_bap", 60, spans, speaker_f0_med_hz=240.0)
        base = _make_base_ctrl(60, f0_val=240.0, bap_val=0.45)
        base.log_env[0:30, 8:170] = -1.2   # bright vowel core
        base.log_env[30:60, 8:170] = -2.4  # quiet syllable (-10.4 dB)

        fused, _ = arb.apply(ctx, base, [rule])
        self.assertLess(float(np.mean(fused.bap[10:25, :3])), float(np.mean(fused.bap[40:55, :3])))

    def test_vot_edge_ramping_and_long_vowel_microvariation(self) -> None:
        rule = RuleGlottalF0Coupling()
        arb = Arbiter()
        spans = [
            SyllableSpan(0, "他", "t", "a", 1, 0, 12, 55, prosodic_break=0),
            SyllableSpan(1, "好", "h", "ao", 3, 55, 65, 80, prosodic_break=3),
        ]
        ctx = build_utterance_context("vot_long", 80, spans, speaker_f0_med_hz=240.0)
        base = _make_base_ctrl(80, f0_val=240.0, bap_val=[0.16, 0.45, 0.68, 0.80, 0.90, 0.92])
        base.voiced[0:12] = False
        base.f0_hz[0:12] = 0.0
        base.voiced[55:65] = False
        base.f0_hz[55:65] = 0.0

        fused, trace = arb.apply(ctx, base, [rule])
        self.assertIsNone(trace["modules"][0]["error"])

        self.assertGreater(float(fused.bap[12, 0]), 0.55)
        self.assertGreater(float(fused.bap[12, 0]), float(fused.bap[13, 0]))
        self.assertGreater(float(fused.bap[13, 0]), float(fused.bap[14, 0]))
        self.assertGreater(float(fused.bap[14, 0]), float(fused.bap[20, 0]))
        self.assertLess(float(fused.glottal_tilt_db_per_oct[12]), float(fused.glottal_tilt_db_per_oct[20]))

        # Long-vowel mid-late region (frame 38) should show breathiness/jitter/rd lift over early nucleus (frame 22)
        self.assertGreater(float(fused.jitter_frac[38]), float(fused.jitter_frac[22]))
        self.assertGreater(float(fused.shimmer_frac[38]), float(fused.shimmer_frac[22]))
        assert fused.rd is not None
        self.assertGreater(float(fused.rd[38]), float(fused.rd[22]))

    def test_self_normalizing_bap_for_blurred_and_clean_kb(self) -> None:
        rule = RuleGlottalF0Coupling()
        arb = Arbiter()
        spans = [SyllableSpan(0, "巴", "b", "a", 1, 0, 5, 25)]
        ctx = build_utterance_context("self_norm", 25, spans, speaker_f0_med_hz=240.0)
        base_blurred = _make_base_ctrl(25, f0_val=240.0, bap_val=[0.18, 0.50, 0.72, 0.85, 0.92, 0.95])
        base_clean = _make_base_ctrl(25, f0_val=240.0, bap_val=[0.025, 0.35, 0.65, 0.85, 0.94, 0.96])

        f_blur, _ = arb.apply(ctx, base_blurred, [rule])
        f_clean, _ = arb.apply(ctx, base_clean, [rule])

        self.assertGreater(float(f_clean.bap[12, 0]), 0.020)
        self.assertLess(float(f_blur.bap[12, 0]), 0.080)
        self.assertLess(abs(float(f_blur.bap[12, 0]) - float(f_clean.bap[12, 0])), 0.04)


class TestNasalAntiresonance(unittest.TestCase):
    def test_no_nasals_returns_none(self) -> None:
        rule = RuleNasalAntiresonance()
        spans = [SyllableSpan(0, "八", "b", "a", 1, 0, 5, 25)]
        ctx = build_utterance_context("oral", 25, spans)
        base = _make_base_ctrl(25)
        self.assertIsNone(rule.propose(ctx, base))

    def test_classify_nasal_coda(self) -> None:
        self.assertEqual(classify_nasal_coda("an"), "coda_n")
        self.assertEqual(classify_nasal_coda("vn"), "coda_n")
        self.assertEqual(classify_nasal_coda("ang"), "coda_ng")
        self.assertEqual(classify_nasal_coda("ing"), "coda_ng")
        self.assertEqual(classify_nasal_coda("ong"), "coda_ong")
        self.assertEqual(classify_nasal_coda("iong"), "coda_ong")
        self.assertEqual(classify_nasal_coda("ao"), "none")

    def test_m_vs_n_onset_and_coda_differentiation(self) -> None:
        rule = RuleNasalAntiresonance()
        spans = [
            SyllableSpan(0, "妈", "m", "a", 1, 0, 10, 30),
            SyllableSpan(1, "拿", "n", "a", 2, 30, 40, 60),
            SyllableSpan(2, "安", "", "an", 1, 60, 60, 90),
            SyllableSpan(3, "昂", "", "ang", 2, 90, 90, 120),
            SyllableSpan(4, "东", "d", "ong", 1, 120, 125, 155),
        ]
        ctx = build_utterance_context("nasals", 155, spans)
        base = _make_base_ctrl(155)
        prop = rule.propose(ctx, base)
        self.assertIsNotNone(prop)
        assert prop is not None
        assert prop.delta_log_env is not None

        freqs = rfftfreq(N_FFT, 1.0 / SR)
        lo_band = (freqs >= 100.0) & (freqs < 400.0)
        m_zero_band = (freqs >= 1100.0) & (freqs < 1600.0)
        alv_band = (freqs >= 2400.0) & (freqs < 3400.0)

        self.assertTrue(np.allclose(prop.delta_log_env[12:28], 0.0))

        m_prof = prop.delta_log_env[4]
        n_prof = prop.delta_log_env[34]
        self.assertFalse(np.allclose(m_prof, n_prof))
        self.assertGreater(float(np.mean(m_prof[lo_band])), float(np.mean(n_prof[lo_band])))
        self.assertLess(float(np.mean(m_prof[m_zero_band])), float(np.mean(n_prof[m_zero_band])))

        an_tail = prop.delta_log_env[88]
        ang_tail = prop.delta_log_env[118]
        ong_tail = prop.delta_log_env[153]
        self.assertLess(float(np.mean(an_tail[alv_band])), float(np.mean(ang_tail[alv_band])))
        self.assertLess(float(np.mean(ang_tail[m_zero_band])), float(np.mean(an_tail[m_zero_band])))
        self.assertLess(float(np.mean(ong_tail[m_zero_band])), float(np.mean(an_tail[m_zero_band])))

    def test_combined_arbiter_and_larynx_synthesis(self) -> None:
        arb = Arbiter()
        mods = [RuleGlottalF0Coupling(), RuleNasalAntiresonance()]
        spans = [
            SyllableSpan(0, "明", "m", "ing", 2, 0, 8, 35),
            SyllableSpan(1, "天", "t", "ian", 1, 35, 45, 75, prosodic_break=3),
        ]
        ctx = build_utterance_context("combo", 75, spans, speaker_f0_med_hz=235.0)
        base = _make_base_ctrl(75, f0_val=235.0, bap_val=0.42)
        base.voiced[35:43] = False
        base.f0_hz[35:43] = 0.0

        fused, trace = arb.apply(ctx, base, mods)
        self.assertEqual(len(trace["modules"]), 2)
        self.assertIsNone(trace["modules"][0]["error"])
        self.assertIsNone(trace["modules"][1]["error"])
        self.assertTrue(np.all((fused.bap >= 0.005) & (fused.bap <= 0.995)))

        wav = synthesize_larynx(fused, glottal_mode="parametric_lf", seed=17)
        self.assertEqual(len(wav), 75 * HOP)
        self.assertTrue(np.all(np.isfinite(wav)))
        self.assertGreater(float(np.sqrt(np.mean(wav ** 2))), 1e-4)


if __name__ == "__main__":
    unittest.main()
