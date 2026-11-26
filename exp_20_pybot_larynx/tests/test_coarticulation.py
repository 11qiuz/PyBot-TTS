"""Unit and integration tests for Stage 3 dual-regime RuleBoundaryCoarticulation."""
from __future__ import annotations

from pathlib import Path
import sys
import time
import unittest

import numpy as np
from scipy.fft import rfftfreq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import HOP, KNOWLEDGE_PATH, N_BAP, N_BINS, N_FFT, SR  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge import AcousticKnowledge, build_base_control  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import (  # noqa: E402
    STAGE2_KNOWLEDGE_PATH,
    AcousticKnowledgeStage2,
    build_stage2_control,
)
from exp_20_pybot_larynx.arbiter import Arbiter, SyllableSpan, build_utterance_context  # noqa: E402
from exp_20_pybot_larynx.judge import evaluate_utterance  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env, log_env_to_cep, synthesize_larynx  # noqa: E402
from exp_20_pybot_larynx.rules.coarticulation import (  # noqa: E402
    _BAND_BASIS,
    _CEP_DCT18,
    _CEP_DCT24_64,
    _MEDIAL_PROFILES,
    _RIGHT_FIN_PROFILES,
    _RIGHT_PLC_S2_SHAPED,
    _TONE_PAIR_PROFILES,
    _TONE_PAIR_S2_PROFILES,
    RuleBoundaryCoarticulation,
    _is_unblurred_base,
    classify_medial_glide,
)


def _make_ctrl(syls: list[SyllableSpan], n_fr: int, *, unblurred: bool = False) -> tuple:
    ctx = build_utterance_context("test_coartic", n_fr, syls, speaker_f0_med_hz=240.0)
    f0 = np.full(n_fr, 230.0, dtype=np.float64)
    voi = np.ones(n_fr, dtype=bool)
    freqs = rfftfreq(N_FFT, 1.0 / SR)
    base_env = -4.0 - 0.3 * np.log1p((freqs / 1000.0) ** 2) + 1.2 * np.exp(-0.5 * ((freqs - 700.0) / 180.0) ** 2)
    env = np.repeat(base_env[None, :], n_fr, axis=0)
    if unblurred and n_fr > 0:
        rng = np.random.default_rng(77)
        cep_hi = np.zeros((n_fr, 64), dtype=np.float64)
        cep_hi[:, 24:64] = rng.normal(0.0, 0.70, size=(n_fr, 40))
        env = env + cep_to_log_env(cep_hi, n_bins=N_BINS)
    base = LarynxControl(
        f0_hz=f0,
        voiced=voi,
        log_env=env,
        bap=np.full((n_fr, N_BAP), 0.15, dtype=np.float64),
        n_samples=n_fr * HOP,
    )
    return ctx, base


class TestRuleBoundaryCoarticulation(unittest.TestCase):
    def test_bases_and_tables(self) -> None:
        self.assertEqual(_BAND_BASIS.shape, (10, N_BINS))
        self.assertTrue(np.all(_BAND_BASIS >= 0.0))
        self.assertTrue(np.allclose(_BAND_BASIS.sum(axis=0), 1.0, atol=1e-9))
        self.assertEqual(_CEP_DCT18.shape, (N_BINS, 18))
        self.assertEqual(_CEP_DCT24_64.shape, (N_BINS, 40))
        self.assertTrue(np.allclose(_CEP_DCT18.T @ _CEP_DCT18, np.eye(18), atol=1e-9))
        for tbl in (_RIGHT_FIN_PROFILES, _RIGHT_PLC_S2_SHAPED, _TONE_PAIR_PROFILES, _TONE_PAIR_S2_PROFILES):
            for k, v in tbl.items():
                self.assertEqual(v.shape, (10,), msg=f"Bad shape for {k}")
                self.assertTrue(np.all(np.isfinite(v)))
        for k, v in _MEDIAL_PROFILES.items():
            self.assertEqual(v.shape, (3, 10), msg=f"Bad shape for {k}")
            self.assertTrue(np.all(np.isfinite(v)))

    def test_blur_detector_and_empty_short_returns_none(self) -> None:
        rule = RuleBoundaryCoarticulation()
        ctx0, base0 = _make_ctrl([], 0)
        self.assertIsNone(rule.propose(ctx0, base0))
        self.assertFalse(_is_unblurred_base(base0))

        syl1 = [SyllableSpan(index=0, hanzi="巴", onset="b", final="a", tone=1, start_fr=0, split_fr=6, end_fr=22)]
        ctx1, base1 = _make_ctrl(syl1, 22, unblurred=False)
        self.assertFalse(_is_unblurred_base(base1))
        self.assertIsNone(rule.propose(ctx1, base1))

        _, base_s2 = _make_ctrl(syl1, 22, unblurred=True)
        self.assertTrue(_is_unblurred_base(base_s2))

        syl_pause = [
            SyllableSpan(index=0, hanzi="巴", onset="b", final="a", tone=1, start_fr=0, split_fr=6, end_fr=22, pause_after_ms=40.0),
            SyllableSpan(index=1, hanzi="大", onset="d", final="a", tone=4, start_fr=30, split_fr=36, end_fr=52),
        ]
        ctx_p, base_p = _make_ctrl(syl_pause, 52)
        self.assertIsNone(rule.propose(ctx_p, base_p))

    def test_right_anticipatory_place_and_obstruent_protection(self) -> None:
        rule = RuleBoundaryCoarticulation()
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        hi_m = freqs >= 3000.0
        retro_m = (freqs >= 2000.0) & (freqs <= 3600.0)

        syl_bp = [
            SyllableSpan(0, "啊", "", "a", 1, 0, 0, 18),
            SyllableSpan(1, "怕", "p", "a", 4, 18, 28, 44),
        ]
        syl_jq = [
            SyllableSpan(0, "啊", "", "a", 1, 0, 0, 18),
            SyllableSpan(1, "去", "q", "v", 4, 18, 28, 44),
        ]
        syl_zh = [
            SyllableSpan(0, "啊", "", "a", 1, 0, 0, 18),
            SyllableSpan(1, "是", "sh", "ir", 4, 18, 28, 44),
        ]
        ctx_bp, base_bp = _make_ctrl(syl_bp, 44)
        ctx_jq, base_jq = _make_ctrl(syl_jq, 44)
        ctx_zh, base_zh = _make_ctrl(syl_zh, 44)

        prop_bp = rule.propose(ctx_bp, base_bp)
        prop_jq = rule.propose(ctx_jq, base_jq)
        prop_zh = rule.propose(ctx_zh, base_zh)
        self.assertIsNotNone(prop_bp)
        self.assertIsNotNone(prop_jq)
        self.assertIsNotNone(prop_zh)

        self.assertLess(float(np.mean(prop_bp.delta_log_env[16, hi_m])), float(np.mean(prop_jq.delta_log_env[16, hi_m])))
        self.assertGreater(float(np.mean(prop_zh.delta_log_env[16, retro_m])), float(np.mean(prop_bp.delta_log_env[16, retro_m])))

        # Strictly protect obstruent onset closure/burst frames [18:28]
        self.assertFalse(np.any(prop_bp.mask[18:28]))
        self.assertFalse(np.any(prop_jq.mask[18:28]))
        self.assertFalse(np.any(prop_zh.mask[18:28]))
        self.assertTrue(np.allclose(prop_bp.delta_log_env[18:28], 0.0))

    def test_cep18_seam_bridging_preserves_high_quefrency_detail(self) -> None:
        rule = RuleBoundaryCoarticulation(
            half_win=3,
            jump_thresh_nep=0.16,
            right_gain=0.0,
            right_s2_gain=0.0,
            left_on_gain=0.0,
            left_fin_gain=0.0,
            tone_pair_gain=0.0,
            tone_pair_s2_gain=0.0,
            medial_gain=0.0,
            neutral_cent_gain=0.0,
            micro_amp_nep=0.0,
            micro_tilt_db=0.0,
        )
        syls = [
            SyllableSpan(0, "啊", "", "a", 1, 0, 0, 15),
            SyllableSpan(1, "妈", "m", "a", 1, 15, 22, 35),
        ]
        ctx, base = _make_ctrl(syls, 35, unblurred=True)
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        base.log_env[:15] += 1.6 * np.exp(-0.5 * ((freqs - 750.0) / 140.0) ** 2)[None, :]
        base.log_env[15:] += 1.6 * np.exp(-0.5 * ((freqs - 1800.0) / 160.0) ** 2)[None, :]

        prop = rule.propose(ctx, base)
        self.assertIsNotNone(prop)
        self.assertTrue(np.all(prop.mask[12:18]))
        self.assertFalse(np.any(prop.mask[:12]))
        self.assertFalse(np.any(prop.mask[18:]))

        after = base.log_env + prop.delta_log_env
        jump_before = float(np.sqrt(np.mean((base.log_env[15] - base.log_env[14]) ** 2)))
        jump_after = float(np.sqrt(np.mean((after[15] - after[14]) ** 2)))
        self.assertLess(jump_after, jump_before)

        d_cep = log_env_to_cep(prop.delta_log_env[12:18], n_cep=40)
        self.assertGreater(float(np.max(np.abs(d_cep[:, :18]))), 0.30)
        # cep[18:40] stays 100% untouched (no formant detail damping)
        self.assertTrue(np.allclose(d_cep[:, 18:], 0.0, atol=1e-7))

    def test_left_tone_pair_medial_neutral_and_micro_dynamics(self) -> None:
        rule = RuleBoundaryCoarticulation()
        self.assertEqual(classify_medial_glide("ia"), "i_medial")
        self.assertEqual(classify_medial_glide("uan"), "u_medial")
        self.assertEqual(classify_medial_glide("ve"), "v_medial")
        self.assertEqual(classify_medial_glide("ao"), "none")

        syl_32 = [
            SyllableSpan(0, "好", "h", "ao", 3, 0, 8, 22),
            SyllableSpan(1, "人", "r", "en", 2, 22, 28, 46),
        ]
        syl_12 = [
            SyllableSpan(0, "高", "g", "ao", 1, 0, 8, 22),
            SyllableSpan(1, "人", "r", "en", 2, 22, 28, 46),
        ]
        for unb in (False, True):
            ctx32, base32 = _make_ctrl(syl_32, 46, unblurred=unb)
            ctx12, base12 = _make_ctrl(syl_12, 46, unblurred=unb)
            p32 = rule.propose(ctx32, base32)
            p12 = rule.propose(ctx12, base12)
            self.assertIsNotNone(p32)
            self.assertIsNotNone(p12)
            self.assertLess(float(np.mean(p32.delta_log_env[29])), float(np.mean(p12.delta_log_env[29])))

        syl_med = [SyllableSpan(0, "家", "j", "ia", 1, 0, 8, 26)]
        ctx_m, base_m = _make_ctrl(syl_med, 26)
        pm = rule.propose(ctx_m, base_m)
        self.assertIsNotNone(pm)
        self.assertGreater(float(np.mean(pm.delta_log_env[17])), float(np.mean(pm.delta_log_env[9])))

        syl_t5 = [SyllableSpan(0, "的", "d", "e", 5, 0, 6, 24)]
        ctx_t5, base_t5 = _make_ctrl(syl_t5, 24)
        p_t5 = rule.propose(ctx_t5, base_t5)
        self.assertIsNotNone(p_t5)
        self.assertLess(float(np.std(base_t5.log_env[21] + p_t5.delta_log_env[21])), float(np.std(base_t5.log_env[21])))

        syl_sus = [SyllableSpan(0, "啊", "", "a", 1, 0, 0, 36)]
        ctx_sus, base_sus = _make_ctrl(syl_sus, 36)
        p_sus = rule.propose(ctx_sus, base_sus)
        self.assertIsNotNone(p_sus)
        self.assertIsNotNone(p_sus.delta_tilt)
        self.assertFalse(np.any(p_sus.mask[:8]))
        self.assertFalse(np.any(p_sus.mask[29:]))
        self.assertTrue(np.any(p_sus.mask[10:26]))
        self.assertGreater(float(np.std(p_sus.delta_log_env[10:26])), 0.002)
        self.assertGreater(float(np.max(np.abs(p_sus.delta_tilt[10:26]))), 0.01)

    def test_arbiter_and_larynx_integration_both_kbs(self) -> None:
        if not KNOWLEDGE_PATH.exists():
            self.skipTest(f"{KNOWLEDGE_PATH} not built yet")
        arb = Arbiter()
        rule = RuleBoundaryCoarticulation()
        syls = [
            SyllableSpan(0, "我", "", "uo", 3, 0, 0, 26),
            SyllableSpan(1, "们", "m", "en", 5, 26, 33, 52),
            SyllableSpan(2, "去", "q", "v", 4, 52, 64, 88),
            SyllableSpan(3, "北", "b", "ei", 3, 88, 94, 120),
            SyllableSpan(4, "京", "j", "ing", 1, 120, 132, 160),
        ]
        n_fr = 160
        f0 = np.full(n_fr, 235.0, dtype=np.float64)
        voi = np.ones(n_fr, dtype=bool)
        voi[52:61] = False
        voi[88:92] = False
        voi[120:128] = False
        f0[~voi] = 0.0

        kb = AcousticKnowledge.load(KNOWLEDGE_PATH)
        ctx = build_utterance_context("coart_e2e_s0", n_fr, syls, speaker_f0_med_hz=kb.speaker_f0_med_hz)
        base, _ = build_base_control(ctx, f0, voi, kb, mode="dynamic")
        base.n_samples = n_fr * HOP

        t0 = time.perf_counter()
        fused, trace = arb.apply(ctx, base, [rule])
        self.assertIsNone(trace["modules"][0]["error"])
        self.assertGreater(trace["modules"][0]["frames_touched"], 0)
        self.assertLess((time.perf_counter() - t0) * 1000.0, 50.0)

        t1 = time.perf_counter()
        y = synthesize_larynx(fused, glottal_mode="parametric_lf", seed=29)
        ev = evaluate_utterance(y, y, rtf=(time.perf_counter() - t1) / max(1e-6, len(y) / SR))
        self.assertTrue(ev["sanity_ok"])
        self.assertLess(ev["clicks_gt8"], 0.01)

        if STAGE2_KNOWLEDGE_PATH.exists():
            kb2 = AcousticKnowledgeStage2.load(STAGE2_KNOWLEDGE_PATH)
            ctx2 = build_utterance_context("coart_e2e_s2", n_fr, syls, speaker_f0_med_hz=kb2.speaker_f0_med_hz)
            base2, _ = build_stage2_control(ctx2, f0, voi, kb2)
            base2.n_samples = n_fr * HOP
            fused2, trace2 = arb.apply(ctx2, base2, [rule])
            self.assertIsNone(trace2["modules"][0]["error"])
            self.assertGreater(trace2["modules"][0]["frames_touched"], 0)
            y2 = synthesize_larynx(fused2, glottal_mode="parametric_lf", seed=29)
            ev2 = evaluate_utterance(y2, y2, rtf=0.02)
            self.assertTrue(ev2["sanity_ok"])
            self.assertLess(ev2["clicks_gt8"], 0.01)


if __name__ == "__main__":
    unittest.main()
