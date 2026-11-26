"""Unit tests for Stage 7 spectral decomposition, synth_v2, Hierarchical Ridge KB, and prosody engine."""
from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from exp_20_pybot_larynx import HOP, N_BAP, N_BINS, N_CEP, N_FFT, SR
from exp_20_pybot_larynx.acoustics.knowledge import get_splits
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import (
    DEFAULT_UV_LEVEL_BOOST_DB_S7,
    STAGE7_KNOWLEDGE_PATH,
    AcousticKnowledgeStage7,
    build_stage7_control,
    classify_offglide,
    extract_stage7_acoustic_features,
)
from exp_20_pybot_larynx.arbiter import Arbiter, SyllableSpan, build_utterance_context
from exp_20_pybot_larynx.frontend.prosody_stage7 import (
    PROSODY_S7_TABLES_PATH,
    ProsodyStage7Model,
    compute_unit_context_s7,
    predict_prosody_stage7,
    render_stage7_utterance,
)
from exp_20_pybot_larynx.larynx.spectral_decomp import (
    compute_pole_bandwidth_stats,
    decompose_frame_source_filter,
    log_env_to_lpc,
    lpc_to_log_env,
    lpc_to_lsf,
    lsf_to_formants,
    lsf_to_lpc,
    reconstruct_frame_source_filter,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl, log_env_to_cep
from exp_20_pybot_larynx.larynx.synth_v2 import analyze_true_harmonic_env, synthesize_larynx_v2
from exp_20_pybot_larynx.rules import default_stage5_rules


def _make_vowel_env() -> np.ndarray:
    """Create a realistic 513-bin log-magnitude vowel envelope with 3 formants and -4 dB/oct tilt."""
    freqs = np.linspace(0.0, SR * 0.5, N_BINS)
    env = -3.0 - 0.5 * np.log2(np.maximum(freqs, 120.0) / 1000.0)
    for f_c, bw, amp in ((520.0, 90.0, 1.8), (1550.0, 130.0, 1.3), (2750.0, 190.0, 0.9)):
        env += amp / (1.0 + ((freqs - f_c) / bw) ** 2)
    return env


def _make_dummy_ctrl(n_fr: int = 120) -> tuple[object, LarynxControl]:
    spans = [
        SyllableSpan(0, "大", "d", "a", 4, 0, 5, 38, prosodic_break=0),
        SyllableSpan(1, "家", "j", "ia", 1, 38, 48, 80, prosodic_break=0),
        SyllableSpan(2, "好", "h", "ao", 3, 80, 92, n_fr, prosodic_break=4),
    ]
    ctx = build_utterance_context("test_s7", n_fr, spans, speaker_f0_med_hz=235.0)
    f0 = np.full(n_fr, 230.0, dtype=np.float64)
    voi = np.ones(n_fr, dtype=bool)
    f0[:4] = 0.0
    voi[:4] = False
    f0[38:44] = 0.0
    voi[38:44] = False
    f0[80:88] = 0.0
    voi[80:88] = False
    env = _make_vowel_env()
    log_env = np.repeat(env[None, :], n_fr, axis=0)
    bap = np.full((n_fr, N_BAP), 0.18, dtype=np.float64)
    bap[~voi] = 0.92
    # Give final 22 frames low silence floor so active-frame UV boost detection mirrors real utterances
    n_sil = min(22, n_fr // 4)
    log_env[-n_sil:] = -14.0
    f0[-n_sil:] = 0.0
    voi[-n_sil:] = False
    ctrl = LarynxControl(f0_hz=f0, voiced=voi, log_env=log_env, bap=bap, n_samples=n_fr * HOP)
    return ctx, ctrl


class TestSpectralDecompAndSynthV2(unittest.TestCase):
    def test_lsf_roundtrip_formants_and_pole_bw(self) -> None:
        env = _make_vowel_env()
        a, gain = log_env_to_lpc(env, order=22, n_fft=N_FFT)
        self.assertEqual(a.shape, (22,))
        self.assertGreater(gain, 0.0)

        lsf = lpc_to_lsf(a)
        self.assertEqual(lsf.shape, (22,))
        self.assertTrue(np.all(np.diff(lsf) > 0.0))
        self.assertGreater(float(lsf[0]), 0.0)
        self.assertLess(float(lsf[-1]), np.pi)

        a_back = lsf_to_lpc(lsf)
        self.assertLess(float(np.max(np.abs(a - a_back))), 1e-8)

        hz, bw = lsf_to_formants(lsf, sr=SR)
        self.assertGreaterEqual(len(hz), 3)
        self.assertTrue(np.all(np.diff(hz) > 0.0))
        self.assertTrue(np.all(bw > 0.0))

        # Source-filter decomposition: cep64 projection + zero-residual exact roundtrip
        sf = decompose_frame_source_filter(env, order=22)
        recon = reconstruct_frame_source_filter(sf)
        c_in = log_env_to_cep(env, n_cep=N_CEP)
        c_out = log_env_to_cep(recon, n_cep=N_CEP)
        self.assertLess(float(np.max(np.abs(c_in - c_out))), 1e-10)

        sf_pure = sf
        sf_pure_env = lpc_to_log_env(lsf_to_lpc(sf_pure.lsf), sf_pure.lpc_gain)
        lsf2 = lpc_to_lsf(log_env_to_lpc(sf_pure_env, order=22)[0])
        self.assertLess(float(np.max(np.abs(sf_pure.lsf - lsf2))), 1e-6)

        stats = compute_pole_bandwidth_stats(np.repeat(env[None, :], 12, axis=0), np.ones(12, dtype=bool))
        self.assertGreater(stats["eff_bw_hz"], 10.0)
        self.assertGreater(stats["pole_radius_median"], 0.5)

    def test_analyze_true_harmonic_env_and_synth_v2_modes(self) -> None:
        _, ctrl = _make_dummy_ctrl(100)
        phasor = np.exp(1j * np.linspace(0.0, 2.0, N_BINS))

        wavs: dict[str, np.ndarray] = {}
        for mode in ("parametric_lf", "speaker_coherent", "dynamic_coherent"):
            y = synthesize_larynx_v2(
                ctrl,
                glottal_mode=mode,
                coherent_phasor=phasor,
                uv_level_boost_db=0.9,
                seed=77,
            )
            self.assertEqual(len(y), 100 * HOP)
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertGreater(float(np.sqrt(np.mean(y * y))), 1e-4)
            wavs[mode] = y

        # dynamic_coherent must respond smoothly to rd modulation
        ctrl_lo = ctrl.copy()
        ctrl_hi = ctrl.copy()
        ctrl_lo.rd = np.full(100, 0.70, dtype=np.float64)
        ctrl_hi.rd = np.full(100, 1.70, dtype=np.float64)
        y_lo = synthesize_larynx_v2(ctrl_lo, glottal_mode="dynamic_coherent", coherent_phasor=phasor, seed=77)
        y_hi = synthesize_larynx_v2(ctrl_hi, glottal_mode="dynamic_coherent", coherent_phasor=phasor, seed=77)
        self.assertGreater(float(np.max(np.abs(y_lo - y_hi))), 1e-3)

        # subharm_strength and uv_level_boost_db behavior
        y_sub = synthesize_larynx_v2(
            ctrl, glottal_mode="dynamic_coherent", coherent_phasor=phasor, subharm_strength=0.15, seed=77
        )
        self.assertTrue(np.all(np.isfinite(y_sub)))
        self.assertEqual(len(y_sub), len(wavs["dynamic_coherent"]))

        y_uv0 = synthesize_larynx_v2(ctrl, glottal_mode="dynamic_coherent", coherent_phasor=phasor, uv_level_boost_db=0.0, seed=77)
        y_uv12 = synthesize_larynx_v2(ctrl, glottal_mode="dynamic_coherent", coherent_phasor=phasor, uv_level_boost_db=1.2, seed=77)
        # Unvoiced prefix frames [0:4] must gain amplitude while voiced frames stay identical
        self.assertGreater(float(np.max(np.abs(y_uv12[: 3 * HOP]))), float(np.max(np.abs(y_uv0[: 3 * HOP]))))

        # Analyze synthesized voiced+unvoiced speech
        log_env, f0_hz, voiced, bap6 = analyze_true_harmonic_env(wavs["dynamic_coherent"])
        self.assertEqual(log_env.shape[1], N_BINS)
        self.assertEqual(bap6.shape[1], N_BAP)
        self.assertEqual(len(f0_hz), len(voiced))
        self.assertTrue(np.all(np.isfinite(log_env)))
        self.assertTrue(np.any(voiced))


class TestStage7KnowledgeAndProsody(unittest.TestCase):
    def test_offglide_and_acoustic_features(self) -> None:
        self.assertEqual(classify_offglide("ang"), "ng")
        self.assertEqual(classify_offglide("ian"), "n")
        self.assertEqual(classify_offglide("ir"), "ir")
        self.assertEqual(classify_offglide("ai"), "i")
        feat = extract_stage7_acoustic_features(
            prev_final="an",
            prev_tone=1,
            next_onset="zh",
            next_final="uo",
            next_tone=4,
            pos_id=1,
            dur_ms=190.0,
            on_ms=35.0,
            f3_st=(0.5, 0.0, -0.5),
            brk=1,
            pause_prev_ms=0.0,
            pause_after_ms=0.0,
            f_end_proto={},
            o_start_proto={},
            f_start_proto={},
        )
        self.assertEqual(feat.shape, (84,))
        self.assertEqual(feat[0], 1.0)
        self.assertTrue(np.all(np.isfinite(feat)))

    @unittest.skipUnless(STAGE7_KNOWLEDGE_PATH.exists(), "knowledge_stage7.npz not built yet")
    def test_stage7_knowledge_load_splits_and_determinism(self) -> None:
        size_mb = STAGE7_KNOWLEDGE_PATH.stat().st_size / (1024.0 * 1024.0)
        self.assertLessEqual(size_mb, 85.0, f"knowledge_stage7.npz too large: {size_mb:.2f} MB")

        kb7 = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
        self.assertEqual(kb7.n_train_utts, 9666)
        train_ids, dev_ids, held_ids = get_splits()
        self.assertEqual(len(train_ids), 9666)
        self.assertFalse(set(train_ids) & set(dev_ids))
        self.assertFalse(set(train_ids) & set(held_ids))

        ctx, ctrl = _make_dummy_ctrl(120)
        c1, info1 = build_stage7_control(ctx, ctrl.f0_hz, ctrl.voiced, kb7)
        c2, _ = build_stage7_control(ctx, ctrl.f0_hz, ctrl.voiced, kb7)
        np.testing.assert_array_equal(c1.log_env, c2.log_env)
        np.testing.assert_array_equal(c1.bap, c2.bap)
        self.assertEqual(info1["mode"], "stage7")
        self.assertEqual(c1.log_env.shape, (120, N_BINS))
        self.assertTrue(np.all(np.isfinite(c1.log_env)))

    @unittest.skipUnless(
        STAGE7_KNOWLEDGE_PATH.exists() and PROSODY_S7_TABLES_PATH.exists(),
        "Stage 7 artifacts not built yet",
    )
    def test_stage7_prosody_and_e2e_render(self) -> None:
        p_mb = PROSODY_S7_TABLES_PATH.stat().st_size / (1024.0 * 1024.0)
        self.assertLessEqual(p_mb, 3.0, f"prosody_s7_tables.npz too large: {p_mb:.2f} MB")

        mdl = ProsodyStage7Model(PROSODY_S7_TABLES_PATH)
        self.assertTrue(mdl.loaded)
        kb7 = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
        pro = predict_prosody_stage7("今天测试七阶段电子声带，表现非常稳定。", kb7, model=mdl)
        self.assertGreater(len(pro["units"]), 6)
        ctxs = compute_unit_context_s7(pro["units"])
        self.assertEqual(len(ctxs), len(pro["units"]))

        arb = Arbiter()
        rules5 = default_stage5_rules()
        w1, info1 = render_stage7_utterance(
            "今天测试七阶段电子声带。",
            "u_s7_1",
            kb7,
            arbiter=arb,
            rules=rules5,
            uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7,
            seed=123,
            model=mdl,
        )
        w2, _ = render_stage7_utterance(
            "今天测试七阶段电子声带。",
            "u_s7_1",
            kb7,
            arbiter=arb,
            rules=rules5,
            uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7,
            seed=123,
            model=mdl,
        )
        np.testing.assert_array_equal(w1, w2)
        self.assertTrue(np.all(np.isfinite(w1)))
        self.assertGreater(float(np.sqrt(np.mean(w1 * w1))), 1e-4)
        for m in info1["arbiter_trace"]["modules"]:
            self.assertIsNone(m["error"])


if __name__ == "__main__":
    unittest.main()
