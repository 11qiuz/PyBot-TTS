"""Unit and integration contract tests for exp_20_pybot_larynx."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import unittest

import numpy as np
from scipy.fft import rfftfreq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import HOP, KNOWLEDGE_PATH, N_BAP, N_BINS, N_CEP, N_FFT, SR  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge import (  # noqa: E402
    AcousticKnowledge,
    build_base_control,
    get_splits,
)
from exp_20_pybot_larynx.arbiter import (  # noqa: E402
    NEPER_TO_DB,
    Arbiter,
    RuleProposal,
    SyllableSpan,
    UtteranceContext,
    build_utterance_context,
)
from exp_20_pybot_larynx.judge import evaluate_utterance  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import (  # noqa: E402
    LarynxControl,
    cep_to_log_env,
    log_env_to_cep,
    synthesize_larynx,
)
from exp_20_pybot_larynx.rules import default_stage0_rules  # noqa: E402
from exp_20_pybot_larynx.stage0_probe import render_c4_auto_utterance  # noqa: E402
from exp_v15_bznsyp_tts.prosody_v15 import ProsodyModelV15  # noqa: E402


class _CrashingRule:
    name = "crashing_rule"

    def propose(self, ctx: UtteranceContext, base: LarynxControl) -> RuleProposal | None:
        raise RuntimeError("intentional fault injection")


class _ValidTiltRule:
    name = "valid_tilt_rule"

    def propose(self, ctx: UtteranceContext, base: LarynxControl) -> RuleProposal | None:
        mask = base.voiced.copy()
        d_tilt = np.full(len(base.f0_hz), 2.5, dtype=np.float64)
        return RuleProposal(module_name=self.name, mask=mask, delta_tilt=d_tilt)


class TestExp20PyBotLarynx(unittest.TestCase):
    def test_zero_leakage_splits(self) -> None:
        train_ids, dev_60, held_20 = get_splits()
        self.assertEqual(len(dev_60), 60)
        self.assertEqual(len(held_20), 20)
        s_tr, s_dev, s_he = set(train_ids), set(dev_60), set(held_20)
        self.assertEqual(len(s_tr & s_dev), 0)
        self.assertEqual(len(s_tr & s_he), 0)
        self.assertEqual(len(s_dev & s_he), 0)
        self.assertGreater(len(train_ids), 9000)

    def test_cepstral_roundtrip(self) -> None:
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        # Smooth realistic 3-formant spectral envelope in natural log amplitude
        env1 = (
            -1.0
            - 0.35 * np.log1p((freqs / 700.0) ** 2)
            + 1.6 * np.exp(-0.5 * ((freqs - 750.0) / 140.0) ** 2)
            + 1.2 * np.exp(-0.5 * ((freqs - 1350.0) / 180.0) ** 2)
            + 0.8 * np.exp(-0.5 * ((freqs - 2650.0) / 240.0) ** 2)
        )
        env2 = (
            -1.5
            - 0.40 * np.log1p((freqs / 900.0) ** 2)
            + 1.5 * np.exp(-0.5 * ((freqs - 320.0) / 110.0) ** 2)
            + 1.3 * np.exp(-0.5 * ((freqs - 2200.0) / 200.0) ** 2)
        )
        envs = np.stack([env1, env2], axis=0)
        cep = log_env_to_cep(envs, n_cep=N_CEP)
        self.assertEqual(cep.shape, (2, N_CEP))
        rec = cep_to_log_env(cep, n_bins=N_BINS)
        self.assertEqual(rec.shape, (2, N_BINS))
        rms_db = float(np.sqrt(np.mean((rec - envs) ** 2)) * NEPER_TO_DB)
        self.assertLess(rms_db, 0.20)

    def test_larynx_synth_contract_and_speed(self) -> None:
        n_fr = 400  # 2.0 seconds @ 5 ms hop
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        base_env = -2.2 - 0.3 * np.log1p((freqs / 1200.0) ** 2) + 1.4 * np.exp(-0.5 * ((freqs - 650.0) / 160.0) ** 2)
        log_env = np.repeat(base_env[None, :], n_fr, axis=0)
        f0_hz = np.zeros(n_fr, dtype=np.float64)
        voiced = np.zeros(n_fr, dtype=bool)
        # Voiced span [60:260], unvoiced fricative span [260:360]
        voiced[60:260] = True
        f0_hz[60:260] = np.linspace(260.0, 210.0, 200)
        bap = np.full((n_fr, N_BAP), 0.12, dtype=np.float64)
        bap[260:360] = 0.98
        log_env[260:360] += 0.6 * np.exp(-0.5 * ((freqs - 4500.0) / 1000.0) ** 2)[None, :]

        ctrl = LarynxControl(f0_hz=f0_hz, voiced=voiced, log_env=log_env, bap=bap, n_samples=n_fr * HOP)
        t0 = time.perf_counter()
        y = synthesize_larynx(ctrl, glottal_mode="parametric_lf", seed=42)
        dt = time.perf_counter() - t0
        rtf = dt / (len(y) / SR)

        self.assertEqual(len(y), n_fr * HOP)
        self.assertTrue(np.all(np.isfinite(y)))
        rms = float(np.sqrt(np.mean(y ** 2)))
        self.assertGreater(rms, 1e-4)
        self.assertLess(rtf, 0.08)

        # Unvoiced span should be genuinely non-periodic (no strong low-freq voiced comb ringing)
        uv_seg = y[280 * HOP : 350 * HOP]
        uv_rms = float(np.sqrt(np.mean(uv_seg ** 2)))
        self.assertGreater(uv_rms, 1e-4)

    def test_arbiter_fault_isolation_and_attribution(self) -> None:
        n_fr = 40
        spans = [SyllableSpan(index=0, hanzi="好", onset="h", final="ao", tone=3, start_fr=0, split_fr=10, end_fr=40)]
        ctx = build_utterance_context("unit_test", n_fr, spans, speaker_f0_med_hz=240.0)
        f0 = np.zeros(n_fr, dtype=np.float64)
        voi = np.zeros(n_fr, dtype=bool)
        voi[10:] = True
        f0[10:] = 210.0
        base = LarynxControl(
            f0_hz=f0,
            voiced=voi,
            log_env=np.zeros((n_fr, N_BINS), dtype=np.float64),
            bap=np.full((n_fr, N_BAP), 0.2, dtype=np.float64),
        )
        arb = Arbiter()
        fused, trace = arb.apply(ctx, base, [_CrashingRule(), _ValidTiltRule()])
        self.assertEqual(len(trace["modules"]), 2)
        self.assertIsNotNone(trace["modules"][0]["error"])
        self.assertIsNone(trace["modules"][1]["error"])
        self.assertEqual(trace["modules"][1]["frames_touched"], 30)
        self.assertTrue(np.allclose(fused.glottal_tilt_db_per_oct[10:], 2.5))
        self.assertTrue(np.allclose(fused.glottal_tilt_db_per_oct[:10], 0.0))

    def test_end_to_end_rules_and_knowledge(self) -> None:
        if not KNOWLEDGE_PATH.exists():
            self.skipTest(f"{KNOWLEDGE_PATH} not built yet")
        kb = AcousticKnowledge.load(KNOWLEDGE_PATH)
        pm = ProsodyModelV15.load()
        arb = Arbiter()
        rules = default_stage0_rules()

        # C3-style context + dynamic base + rules
        spans = [
            SyllableSpan(index=0, hanzi="你", onset="n", final="i", tone=3, start_fr=0, split_fr=8, end_fr=30),
            SyllableSpan(index=1, hanzi="好", onset="h", final="ao", tone=3, start_fr=30, split_fr=42, end_fr=70),
        ]
        ctx = build_utterance_context("smoke_c3", 70, spans, speaker_f0_med_hz=kb.speaker_f0_med_hz)
        f0 = np.full(70, 230.0, dtype=np.float64)
        voi = np.ones(70, dtype=bool)
        voi[30:40] = False
        f0[30:40] = 0.0
        base, _ = build_base_control(ctx, f0, voi, kb, mode="dynamic")
        fused, trace = arb.apply(ctx, base, rules)
        y_c3 = synthesize_larynx(fused, glottal_mode="parametric_lf", seed=7)
        self.assertEqual(len(y_c3), 70 * HOP)
        self.assertTrue(np.all(np.isfinite(y_c3)))
        self.assertGreater(sum(m["frames_touched"] for m in trace["modules"]), 0)

        # C4 auto text-to-speech sanity (warm up lazy frontend/jieba/pypinyin caches once first)
        render_c4_auto_utterance("请测试电子声带。", "warmup", kb, pm, arbiter=arb, rules=rules, seed=1)
        t0 = time.perf_counter()
        y_c4, _ = render_c4_auto_utterance("你好，请测试电子声带。", "smoke_c4", kb, pm, arbiter=arb, rules=rules, seed=9)
        dt = time.perf_counter() - t0
        rtf = dt / max(1e-6, len(y_c4) / SR)
        ev = evaluate_utterance(y_c4, y_c4, rtf=rtf)
        self.assertTrue(ev["sanity_ok"])
        self.assertLess(rtf, 0.15)


if __name__ == "__main__":
    unittest.main()
