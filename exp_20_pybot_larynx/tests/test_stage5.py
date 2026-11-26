"""Unit tests for Stage 5 expert rules, cubic phase interpolation, and prosody refiner."""
from __future__ import annotations

import unittest

import numpy as np

from exp_20_pybot_larynx import N_BAP, N_BINS
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import (
    STAGE5_KNOWLEDGE_PATH,
    AcousticKnowledgeStage5,
    _interp_phase_bins_cubic,
    bin_dur_ms,
    bin_f0_st,
    build_stage5_control,
)
from exp_20_pybot_larynx.arbiter import (
    Arbiter,
    SyllableSpan,
    build_utterance_context,
)
from exp_20_pybot_larynx.frontend.prosody_stage5 import (
    compute_unit_context_ids,
    predict_prosody_stage5,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl, log_env_to_cep, synthesize_larynx
from exp_20_pybot_larynx.rules import default_stage5_rules
from exp_20_pybot_larynx.rules.formant_dynamics import RuleFormantDynamics
from exp_20_pybot_larynx.rules.prosodic_energy import RuleProsodicEnergyTilt


def _make_dummy_ctx_and_ctrl() -> tuple[object, LarynxControl]:
    # 3 syllables: b+ai (T3 low), m+a (T5 neutral), sh+ir (T4 apical)
    spans = [
        SyllableSpan(0, "摆", "b", "ai", 3, 0, 6, 40, prosodic_break=0),
        SyllableSpan(1, "吗", "m", "a", 5, 40, 50, 70, prosodic_break=0),
        SyllableSpan(2, "是", "sh", "ir", 4, 70, 88, 124, prosodic_break=4),
    ]
    n_fr = 124
    ctx = build_utterance_context("test_s5", n_fr, spans, speaker_f0_med_hz=240.0)
    f0 = np.full(n_fr, 235.0, dtype=np.float64)
    voi = np.ones(n_fr, dtype=bool)
    f0[:4] = 0.0
    voi[:4] = False
    f0[70:86] = 0.0
    voi[70:86] = False
    # Dip Tone-3 syllable below -2.2 st to trigger glottal softening
    f0[12:34] = 195.0

    # Non-flat spectral envelope with realistic formant peaks
    bins = np.linspace(0.0, 1.0, N_BINS)
    env_row = -3.5 - 2.2 * bins + 0.8 * np.sin(14.0 * bins)
    log_env = np.repeat(env_row[None, :], n_fr, axis=0)
    log_env[6:40] += 0.35 * np.sin(np.linspace(0, np.pi, 34))[:, None] * np.cos(10.0 * bins)[None, :]
    bap = np.full((n_fr, N_BAP), 0.25, dtype=np.float64)
    ctrl = LarynxControl(f0_hz=f0, voiced=voi, log_env=log_env, bap=bap, n_samples=n_fr * 120)
    return ctx, ctrl


class TestStage5RulesAndHelpers(unittest.TestCase):
    def test_cubic_phase_interpolation_bounds_and_endpoints(self) -> None:
        knots = np.array([[1.0, 10.0], [3.0, 6.0], [5.0, 2.0], [4.0, 4.0], [2.0, 8.0]], dtype=np.float64)
        centers = (np.arange(5, dtype=np.float64) + 0.5) / 5.0
        u = np.linspace(0.0, 1.0, 31)
        out = _interp_phase_bins_cubic(knots, u, centers)
        self.assertEqual(out.shape, (31, 2))
        self.assertTrue(np.all(np.isfinite(out)))
        # Should hit middle knot exactly at u=0.5
        mid = _interp_phase_bins_cubic(knots, np.array([0.5]), centers)
        np.testing.assert_allclose(mid[0], knots[2], atol=1e-6)

    def test_binning_helpers(self) -> None:
        self.assertEqual(bin_dur_ms(120.0), 0)
        self.assertEqual(bin_dur_ms(180.0), 1)
        self.assertEqual(bin_dur_ms(260.0), 2)
        self.assertEqual(bin_f0_st(-2.5), 0)
        self.assertEqual(bin_f0_st(0.0), 1)
        self.assertEqual(bin_f0_st(2.5), 2)

    def test_rule_formant_dynamics_preserves_cep0_and_shapes_diphthongs(self) -> None:
        ctx, ctrl = _make_dummy_ctx_and_ctrl()
        rule = RuleFormantDynamics()
        prop = rule.propose(ctx, ctrl)
        self.assertIsNotNone(prop)
        assert prop is not None
        self.assertEqual(prop.module_name, "formant_dynamics")
        self.assertEqual(prop.mask.shape, (124,))
        self.assertIsNotNone(prop.delta_log_env)
        assert prop.delta_log_env is not None

        # Must preserve cep[0] (DC energy) strictly across all masked frames
        d_cep = log_env_to_cep(prop.delta_log_env[prop.mask], n_cep=16)
        self.assertLess(float(np.max(np.abs(d_cep[:, 0]))), 1e-9)
        # Must modify diphthong / apical / neutral-tone frames
        self.assertGreater(float(np.max(np.abs(prop.delta_log_env[6:40]))), 1e-4)

    def test_rule_prosodic_energy_t3_relaxation_and_arbiter_clean(self) -> None:
        ctx, ctrl = _make_dummy_ctx_and_ctrl()
        r5 = RuleProsodicEnergyTilt()
        prop = r5.propose(ctx, ctrl)
        self.assertIsNotNone(prop)
        assert prop is not None
        self.assertEqual(prop.module_name, "prosodic_energy_tilt")
        # Tone-3 low-F0 region should propose positive rd relaxation
        self.assertIsNotNone(prop.delta_rd)
        assert prop.delta_rd is not None
        self.assertGreater(float(np.max(prop.delta_rd[12:34])), 0.05)

        arb = Arbiter()
        rules = default_stage5_rules()
        self.assertEqual(len(rules), 6)
        fused, trace = arb.apply(ctx, ctrl, rules)
        self.assertEqual(len(trace["modules"]), 6)
        for mod_tr in trace["modules"]:
            self.assertIsNone(mod_tr["error"])
        wav = synthesize_larynx(fused, glottal_mode="parametric_lf", seed=42)
        self.assertTrue(np.all(np.isfinite(wav)))
        self.assertGreater(float(np.sqrt(np.mean(wav * wav))), 1e-4)

    def test_compute_unit_context_ids(self) -> None:
        units = [
            {"onset": "n", "final": "i", "tone": 2, "word_len": 2, "pos_in_word": 0, "is_word_final": False, "prosodic_break": 0, "punct_after": ""},
            {"onset": "x", "final": "iang", "tone": 3, "word_len": 2, "pos_in_word": 1, "is_word_final": True, "prosodic_break": 4, "punct_after": "。"},
        ]
        ctxs = compute_unit_context_ids(units)
        self.assertEqual(len(ctxs), 2)
        self.assertEqual(ctxs[0]["pt"], 0)
        self.assertEqual(ctxs[0]["nt"], 3)
        self.assertEqual(ctxs[1]["pt"], 2)
        self.assertEqual(ctxs[1]["nt"], 0)
        self.assertEqual(ctxs[1]["brk_ge2"], 1)

    @unittest.skipUnless(STAGE5_KNOWLEDGE_PATH.exists(), "knowledge_stage5.npz not built yet")
    def test_stage5_knowledge_and_e2e_smoke(self) -> None:
        kb5 = AcousticKnowledgeStage5.load(STAGE5_KNOWLEDGE_PATH)
        pro = predict_prosody_stage5("今天测试五阶段电子声带。", kb5)
        self.assertTrue(pro.get("zero_ml_weights"))
        ctx, ctrl = _make_dummy_ctx_and_ctrl()
        c5, info5 = build_stage5_control(ctx, ctrl.f0_hz, ctrl.voiced, kb5)
        self.assertEqual(info5["mode"], "stage5")
        self.assertEqual(c5.log_env.shape, (124, N_BINS))
        self.assertTrue(np.all(np.isfinite(c5.log_env)))


if __name__ == "__main__":
    unittest.main()
