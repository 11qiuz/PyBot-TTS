"""Unit tests for Stage 8 Physical Spectrum Synthesizer and Expressive Emotion Engine."""
from __future__ import annotations

import unittest

import numpy as np

from exp_20_pybot_larynx import HOP, N_BAP, N_BINS, SR
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import (
    STAGE7_KNOWLEDGE_PATH,
    AcousticKnowledgeStage7,
)
from exp_20_pybot_larynx.arbiter import Arbiter
from exp_20_pybot_larynx.expressive import (
    EMOTION_PROFILES,
    apply_emotion_to_control,
    apply_emotion_to_zh_prosody,
    build_breath_control,
    build_laugh_control,
    build_pause_control,
    build_sigh_control,
    parse_expressive_text,
    render_expressive_utterance,
    strip_performance_tags,
)
from exp_20_pybot_larynx.frontend.prosody_stage7 import PROSODY_S7_TABLES_PATH, predict_prosody_stage7
from exp_20_pybot_larynx.larynx.synth import LarynxControl
from exp_20_pybot_larynx.larynx.synth_v2 import synthesize_larynx_v2
from exp_20_pybot_larynx.larynx.timbre_synth import (
    TIMBRE_PATCHES,
    TimbreSynthParams,
    apply_timbre_synth,
    extract_spectral_profile_preview,
)
from exp_20_pybot_larynx.rules import default_stage5_rules
from exp_v5_eval.metrics import check_wav_sanity


def _dummy_voiced_ctrl(n_fr: int = 120) -> LarynxControl:
    freqs = np.linspace(0.0, SR * 0.5, N_BINS)
    env = -3.8 - 0.55 * np.log2(np.maximum(freqs, 120.0) / 1000.0)
    for fc, bw, amp in ((520.0, 95.0, 1.7), (1560.0, 135.0, 1.2), (2750.0, 195.0, 0.8)):
        env += amp / (1.0 + ((freqs - fc) / bw) ** 2)
    f0 = np.full(n_fr, 225.0, dtype=np.float64)
    voi = np.ones(n_fr, dtype=bool)
    f0[:6] = 0.0
    voi[:6] = False
    bap = np.full((n_fr, N_BAP), 0.16, dtype=np.float64)
    bap[~voi] = 0.96
    return LarynxControl(
        f0_hz=f0,
        voiced=voi,
        log_env=np.repeat(env[None, :], n_fr, axis=0),
        bap=bap,
        n_samples=n_fr * HOP,
    )


class TestTimbreSynthesizer(unittest.TestCase):
    def test_neutral_identity_and_all_patches_bounded(self) -> None:
        ctrl = _dummy_voiced_ctrl(120)
        p_neu = TimbreSynthParams.from_dict({"timbre_patch": "neutral"})
        self.assertTrue(p_neu.is_neutral())
        out_neu, info_neu = apply_timbre_synth(ctrl, p_neu)
        np.testing.assert_array_equal(out_neu.log_env, ctrl.log_env)
        np.testing.assert_array_equal(out_neu.f0_hz, ctrl.f0_hz)
        self.assertEqual(info_neu["subharm_strength"], 0.0)

        prev_by_patch: dict[str, dict] = {}
        for pat in TIMBRE_PATCHES:
            tp = TimbreSynthParams.from_dict(pat["params"])
            mod, _ = apply_timbre_synth(ctrl, tp, seed=42)
            self.assertEqual(mod.log_env.shape, ctrl.log_env.shape)
            self.assertTrue(np.all(np.isfinite(mod.log_env)))
            self.assertTrue(np.all(np.isfinite(mod.f0_hz)))
            prev_by_patch[pat["id"]] = extract_spectral_profile_preview(mod)
            self.assertGreaterEqual(len(prev_by_patch[pat["id"]]["formants_hz"]), 3)
            self.assertEqual(len(prev_by_patch[pat["id"]]["curve_hz"]), 48)

        # Radio magnetic boosts chest vs neutral; vintage attenuates air; anime_sweet opens F1/F2
        self.assertGreater(
            prev_by_patch["radio_magnetic"]["band_levels_db"]["chest"],
            prev_by_patch["neutral"]["band_levels_db"]["chest"] + 1.5,
        )
        self.assertLess(
            prev_by_patch["vintage_phone"]["band_levels_db"]["air"],
            prev_by_patch["neutral"]["band_levels_db"]["air"] - 8.0,
        )
        self.assertGreater(
            prev_by_patch["anime_sweet"]["formants_hz"][0],
            prev_by_patch["cinematic_epic"]["formants_hz"][0] + 40.0,
        )

    def test_formant_q_vibrato_and_vocal_fry(self) -> None:
        ctrl = _dummy_voiced_ctrl(120)
        tp_vib = TimbreSynthParams(vibrato_depth_st=0.35, vibrato_rate_hz=5.5, vocal_fry=0.5, formant_q=0.45)
        mod, info = apply_timbre_synth(ctrl, tp_vib, seed=99)
        self.assertAlmostEqual(info["subharm_strength"], 0.14, places=3)
        # Vibrato modulates F0 inside voiced segment after onset ramp
        self.assertGreater(float(np.std(mod.f0_hz[20:100])), 1.5)
        y = synthesize_larynx_v2(
            mod, glottal_mode="dynamic_coherent", subharm_strength=info["subharm_strength"], seed=99
        )
        san = check_wav_sanity(y, SR, expected_sr=SR)
        self.assertTrue(san.get("ok"), f"Sanity failed: {san}")


class TestExpressiveParserAndVocalizations(unittest.TestCase):
    def test_tag_parsing_emphasis_and_auto_emotion(self) -> None:
        self.assertEqual(strip_performance_tags("[吸气] 今天 **真的** 太棒了！"), "今天 真的 太棒了！")

        neutral_segs = parse_expressive_text("今天测试电子声带的日常稳定性。", emotion_mode="auto", auto_breath=True)
        self.assertEqual(len(neutral_segs), 1)
        self.assertEqual(neutral_segs[0].emotion, "neutral")
        self.assertEqual(neutral_segs[0].intensity, 0.0)
        self.assertFalse(neutral_segs[0].auto_breath_before)

        story = "[吸气] *天呐*，你真的不记得了吗？！ [叹气] 我等了整整三个小时…… [开心] [笑] 没关系，NOW IS GOOD!"
        segs = parse_expressive_text(story, emotion_mode="auto", emotion_intensity=0.80, stability=0.50)
        kinds = [s.kind for s in segs]
        self.assertIn("breath", kinds)
        self.assertIn("sigh", kinds)
        self.assertIn("laugh", kinds)
        speech_segs = [s for s in segs if s.kind == "speech"]
        self.assertGreaterEqual(len(speech_segs), 3)
        self.assertGreaterEqual(len(speech_segs[0].focus_spans), 1)
        self.assertEqual(speech_segs[-1].emotion, "happy")
        self.assertGreaterEqual(len(speech_segs[-1].focus_spans), 1)  # ALL CAPS focus

    def test_physical_non_verbal_vocalizations(self) -> None:
        for fn, name in (
            (lambda: build_breath_control(240.0, duration_ms=190.0, is_gasp=False), "breath"),
            (lambda: build_breath_control(240.0, duration_ms=230.0, is_gasp=True), "gasp"),
            (lambda: build_sigh_control(240.0, duration_ms=440.0), "sigh"),
            (lambda: build_laugh_control(240.0, duration_ms=500.0, soft=False), "laugh"),
            (lambda: build_pause_control(300.0), "pause"),
        ):
            ctrl = fn()
            y = synthesize_larynx_v2(ctrl, glottal_mode="parametric_lf", seed=7)
            self.assertEqual(len(y), ctrl.n_samples, name)
            self.assertTrue(np.all(np.isfinite(y)), name)
            if name != "pause":
                self.assertGreater(float(np.sqrt(np.mean(y * y))), 1e-4, name)


class TestStage8PerformerE2E(unittest.TestCase):
    @unittest.skipUnless(
        STAGE7_KNOWLEDGE_PATH.exists() and PROSODY_S7_TABLES_PATH.exists(),
        "Stage 7 artifacts missing",
    )
    def test_emotion_physical_separation_and_sanity(self) -> None:
        kb = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
        arb = Arbiter()
        rules = default_stage5_rules()
        text = "我们今天终于完成了这项重要的突破。"

        metrics: dict[str, dict[str, float]] = {}
        for emo in ("neutral", "excited", "sad", "angry", "gentle", "whisper"):
            wav, info = render_expressive_utterance(
                text,
                f"t_{emo}",
                kb,
                arbiter=arb,
                rules=rules,
                emotion_mode=emo,
                emotion_intensity=0.85,
                stability=0.55,
                auto_breath=False,
                seed=123,
            )
            san = check_wav_sanity(wav, SR, expected_sr=SR)
            self.assertTrue(san.get("ok"), f"{emo} failed sanity: {san}")
            c: LarynxControl = info["ctrl"]
            vm = c.voiced & (c.f0_hz > 40.0)
            metrics[emo] = {
                "f0_med": float(np.median(c.f0_hz[vm])),
                "dur_s": len(wav) / float(SR),
                "tilt": float(np.median(c.glottal_tilt_db_per_oct[vm])) if c.glottal_tilt_db_per_oct is not None else 0.0,
                "bap": float(np.median(c.bap[vm])),
            }

        f0_diff_st = 12.0 * np.log2(metrics["excited"]["f0_med"] / metrics["sad"]["f0_med"])
        self.assertGreaterEqual(f0_diff_st, 3.0)
        self.assertGreater(metrics["sad"]["dur_s"], metrics["neutral"]["dur_s"])
        self.assertGreater(metrics["neutral"]["dur_s"], metrics["excited"]["dur_s"])
        self.assertGreater(metrics["angry"]["tilt"], metrics["sad"]["tilt"] + 1.5)
        self.assertGreater(metrics["whisper"]["bap"], metrics["neutral"]["bap"] + 0.30)


if __name__ == "__main__":
    unittest.main()
