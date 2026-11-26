"""Unit and integration tests for zero-ML Mandarin prosody rules (`frontend/prosody_rules.py`)."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import time
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import EXP_DIR, KNOWLEDGE_PATH, SR  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge import AcousticKnowledge, build_base_control  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import AcousticKnowledgeStage2, STAGE2_KNOWLEDGE_PATH  # noqa: E402
from exp_20_pybot_larynx.arbiter import Arbiter  # noqa: E402
from exp_20_pybot_larynx.frontend import (  # noqa: E402
    predict_prosody_rules,
    render_rule_auto_utterance,
    segment_prosodic_words,
)
from exp_20_pybot_larynx.frontend import prosody_rules as pr_mod  # noqa: E402
from exp_20_pybot_larynx.judge import evaluate_utterance  # noqa: E402
from exp_20_pybot_larynx.rules import default_stage0_rules  # noqa: E402


class TestProsodyRules(unittest.TestCase):
    def test_zero_ml_weights_and_clean_module(self) -> None:
        src = Path(inspect.getfile(pr_mod)).read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for forbidden in ("sklearn", "torch", "tensorflow", "onnx", "onnxruntime", "pickle", "joblib"):
            self.assertNotIn(forbidden, imported)
        res = predict_prosody_rules("你好，测试电子声带。")
        self.assertTrue(res["zero_ml_weights"])
        self.assertGreaterEqual(len(res["units"]), 8)

    def test_word_segmentation_and_prosodic_breaks(self) -> None:
        text = "中国人民银行今天宣布了一项新的政策。"
        winfo = segment_prosodic_words(text)
        # "人民银行" or "银行" / "今天" / "宣布" should form multi-char prosodic words
        self.assertGreater(winfo[text.index("今")][0], 1)
        res = predict_prosody_rules(text)
        units = res["units"]
        self.assertEqual(units[-1]["prosodic_break"], 4)
        self.assertEqual(units[-1]["pause_after_ms"], 0.0)
        # Long clause should contain at least one #2 prosodic break
        self.assertTrue(any(u["prosodic_break"] == 2 for u in units[:-1]))

    def test_onset_duration_and_voicing_rules(self) -> None:
        # Compare aspirated affricate 'q' vs unaspirated stop 'd' vs nasal 'm' vs zero onset 'a'
        res = predict_prosody_rules("七大地妈啊。")
        units = res["units"]
        u_q, u_d, u_m, u_a = units[0], units[2], units[3], units[4]
        self.assertGreater(u_q["onset_ms"], u_d["onset_ms"])
        self.assertGreater(u_q["onset_uv_frac"], 0.80)
        self.assertLess(u_d["onset_uv_frac"], 0.65)
        self.assertLess(u_m["onset_uv_frac"], 0.05)
        self.assertEqual(u_a["onset_ms"], 0.0)
        self.assertEqual(u_a["onset_uv_frac"], 0.0)

    def test_neutral_tone_inheritance_and_sandhi(self) -> None:
        # T1 + T5 ("妈妈") vs T3 + T5 ("好了") vs T4 + T5 ("去了")
        r_t1 = predict_prosody_rules("妈妈。")["units"]
        r_t3 = predict_prosody_rules("好了。")["units"]
        r_t4 = predict_prosody_rules("去了。")["units"]
        self.assertEqual(r_t1[1]["tone"], 5)
        self.assertEqual(r_t3[1]["tone"], 5)
        self.assertEqual(r_t4[1]["tone"], 5)
        # T5 after T1 starts higher than T5 after T3 or T4
        self.assertGreater(r_t1[1]["f0_5pt_st"][0], r_t3[1]["f0_5pt_st"][0])
        self.assertGreater(r_t1[1]["f0_5pt_st"][0], r_t4[1]["f0_5pt_st"][0])
        # T5 medial duration is shortened relative to full-tone syllables (sentence-final T5 lengthens)
        r_med = predict_prosody_rules("妈妈来了。")["units"]
        self.assertEqual(r_med[1]["tone"], 5)
        self.assertLess(r_med[1]["duration_ms"], r_med[0]["duration_ms"])

        # T3 + T3 sandhi: "你好" -> T2 + T3
        r_nh = predict_prosody_rules("你好。")["units"]
        self.assertEqual(r_nh[0]["tone"], 2)
        self.assertEqual(r_nh[1]["tone"], 3)

    def test_question_vs_statement_and_declination(self) -> None:
        r_stmt = predict_prosody_rules("他去了吗。")["units"]
        r_ques = predict_prosody_rules("他去了吗？")["units"]
        # Question-final particle '吗' ends higher than statement-final
        self.assertGreater(r_ques[-1]["f0_5pt_st"][-1], r_stmt[-1]["f0_5pt_st"][-1])

        # Catmull-Rom 5-pt interpolation exact knot match at [0.1, 0.3, 0.5, 0.7, 0.9]
        pts5 = np.array([-1.0, -2.5, -2.0, 0.5, 2.5], dtype=np.float64)
        knots = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        vals = pr_mod._catmull_rom_5pt(knots, pts5)
        self.assertTrue(np.allclose(vals, pts5, atol=1e-6))

        # Gain dB bounded and centered
        r_long = predict_prosody_rules("今天天气真不错，我们一起去公园散步吧。")["units"]
        gains = [u["gain_db"] for u in r_long]
        self.assertTrue(all(-4.5 <= g <= 4.5 for g in gains))
        self.assertLess(abs(float(np.median(gains))), 0.25)
        self.assertGreaterEqual(float(np.std(gains)), 1.1)

    def test_stage4_dynamic_pauses_and_speaking_style(self) -> None:
        r_short = predict_prosody_rules("好了，走吧。")["units"]
        r_long = predict_prosody_rules(
            "今天上午我们在校园里认真讨论了这个课题，明天下午还要继续汇报最新的研究进展。"
        )["units"]
        p_short = [u["pause_after_ms"] for u in r_short if u["punct_after"] == "，"][0]
        p_long = [u["pause_after_ms"] for u in r_long if u["punct_after"] == "，"][0]
        self.assertGreater(p_long, p_short + 60.0)
        self.assertGreaterEqual(float(np.std([u["gain_db"] for u in r_long])), 1.2)

    def test_interjection_edge_case_and_pause_insertion(self) -> None:
        res = predict_prosody_rules("也可以给本委员反映哟，真的吗？")
        units = res["units"]
        self.assertTrue(any(u["hanzi"] == "哟" for u in units))
        comma_u = next(u for u in units if u.get("punct_after") == "，")
        self.assertGreaterEqual(comma_u["pause_after_ms"], 100.0)
        self.assertEqual(units[-1]["pause_after_ms"], 0.0)

    def test_render_rule_auto_utterance_e2e(self) -> None:
        kb_path = EXP_DIR / "acoustics" / "knowledge_full.npz"
        if not kb_path.exists():
            kb_path = KNOWLEDGE_PATH
        if not kb_path.exists():
            self.skipTest("No AcousticKnowledge npz available")
        kb = AcousticKnowledge.load(kb_path)
        arb = Arbiter()
        rules = default_stage0_rules()

        # Warmup once
        render_rule_auto_utterance("测试热身。", "warm", kb, arbiter=arb, rules=rules, seed=1)
        t0 = time.perf_counter()
        wav, info = render_rule_auto_utterance(
            "你好，请测试电子声带韵律规则。",
            "test_c4_rules",
            kb,
            arbiter=arb,
            rules=rules,
            seed=42,
        )
        dt = time.perf_counter() - t0
        rtf = dt / max(1e-6, len(wav) / SR)
        ev = evaluate_utterance(wav, wav, rtf=rtf)
        self.assertTrue(ev["sanity_ok"])
        self.assertLess(rtf, 0.12)
        self.assertIn("prosody_plan", info)
        self.assertGreater(len(info["arbiter_trace"]["modules"]), 0)

    def test_stage2_lexicon_tone_overrides_and_intrinsic_f0(self) -> None:
        self.assertGreater(len(pr_mod._WORD_LEXICON), 5000)
        # Polyphone/neutral-tone disambiguation overrides from TRAIN
        r_wei = predict_prosody_rules("认为可以。")["units"]
        self.assertEqual(r_wei[1]["tone"], 2)
        r_yf = predict_prosody_rules("衣服很干净。")["units"]
        self.assertEqual(r_yf[1]["tone"], 5)
        # Intrinsic vowel F0: high front /i/ ('衣') sits higher than open low /a/ ('啊') in matched T1 frame
        r_i = predict_prosody_rules("衣。")["units"][0]
        r_a = predict_prosody_rules("啊。")["units"][0]
        self.assertGreater(r_i["f0_3pt_st"][1], r_a["f0_3pt_st"][1])

    def test_render_stage2_kb_and_custom_base_builder(self) -> None:
        kb_path = EXP_DIR / "acoustics" / "knowledge_full.npz"
        if not kb_path.exists():
            kb_path = KNOWLEDGE_PATH
        if not kb_path.exists():
            self.skipTest("No AcousticKnowledge npz available")
        kb = AcousticKnowledge.load(kb_path)
        arb = Arbiter()
        rules = default_stage0_rules()
        called: dict[str, object] = {}

        def _spy_builder(ctx, f0_hz, voiced, k_obj, *, mode="dynamic", syl_gain_db=None):
            called["mode"] = mode
            called["n_syl"] = len(ctx.syllables)
            return build_base_control(ctx, f0_hz, voiced, k_obj, mode=mode, syl_gain_db=syl_gain_db)

        wav, info = render_rule_auto_utterance(
            "测试自定义构建器。",
            "test_custom_builder",
            kb,
            arbiter=arb,
            rules=rules,
            seed=7,
            base_builder_fn=_spy_builder,
            base_mode="dynamic",
        )
        self.assertEqual(called.get("mode"), "dynamic")
        self.assertGreaterEqual(int(called.get("n_syl", 0)), 5)
        self.assertTrue(evaluate_utterance(wav, wav, rtf=0.04)["sanity_ok"])

        if STAGE2_KNOWLEDGE_PATH.exists():
            kb2 = AcousticKnowledgeStage2.load(STAGE2_KNOWLEDGE_PATH)
            wav2, info2 = render_rule_auto_utterance(
                "测试二阶段知识库自动选择。",
                "test_kb_stage2_auto",
                kb2,
                arbiter=arb,
                rules=rules,
                seed=9,
            )
            self.assertEqual(info2["kb_info"].get("mode"), "stage2")
            self.assertTrue(evaluate_utterance(wav2, wav2, rtf=0.04)["sanity_ok"])

    def test_stage3_viterbi_brk2_and_7pt_contour(self) -> None:
        # 7-pt Catmull-Rom interpolation exact knot match at _F0_U7
        pts7 = np.array([1.2, 0.8, 0.2, -0.5, -1.2, -0.3, 1.5], dtype=np.float64)
        u7 = np.asarray(pr_mod._F0_U7, dtype=np.float64)
        vals7 = pr_mod._catmull_rom_7pt(u7, pts7)
        self.assertTrue(np.allclose(vals7, pts7, atol=1e-6))

        # Contextual tone disambiguation & 7-pt output
        res = predict_prosody_rules("我们的朋友喜欢这里，但是今天有点忙。")
        units = res["units"]
        self.assertIn("f0_7pt_st", units[0])
        self.assertEqual(len(units[0]["f0_7pt_st"]), 7)
        # Comma pause and enumeration mark pause differentiation
        enum_u = predict_prosody_rules("苹果、香蕉、橘子。")["units"][1]
        comma_u = next(u for u in units if u.get("punct_after") == "，")
        self.assertLess(enum_u["pause_after_ms"], comma_u["pause_after_ms"])


if __name__ == "__main__":
    unittest.main()
