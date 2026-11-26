"""Stage 13 tests: native bilingual SF KB, zero-mapping renderer, and ja/ko IPA extensibility."""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import numpy as np

from exp_20_pybot_larynx import EXP_DIR, SR
from exp_20_pybot_larynx.clone import load_base_speaker_profile
from exp_20_pybot_larynx.multilingual.knowledge_ipa_native import (
    MultilingualIPAKnowledge,
    SF_NATIVE_KB_PATH,
    SF_NATIVE_SUMMARY_PATH,
    get_sf_native_multilingual_kb,
)
from exp_20_pybot_larynx.multilingual.renderer import (
    build_multilingual_native_utterance,
    synthesize_multilingual_native,
)
from exp_20_pybot_larynx.multilingual.schema import (
    IPA_ARTICULATORY_TABLE,
    PhoneEvent,
    get_language_frontend,
    lookup_ipa_info,
    register_language_frontend,
)
from exp_v5_eval.metrics import check_wav_sanity


class TestStage13Multilingual(unittest.TestCase):
    def test_sf_native_kb_zero_ml_and_full_zh_en_coverage(self) -> None:
        self.assertTrue(SF_NATIVE_KB_PATH.is_file(), f"Missing {SF_NATIVE_KB_PATH}")
        size_mb = os.path.getsize(SF_NATIVE_KB_PATH) / (1024.0 * 1024.0)
        self.assertLess(size_mb, 30.0, f"Native KB size {size_mb:.2f} MB exceeded 30 MB cap")

        # Strict allow_pickle=False compliance
        with np.load(SF_NATIVE_KB_PATH, allow_pickle=False) as d:
            self.assertIn("ctx_pns_cep", d)
            self.assertIn("ipa_shared_cep", d)
            self.assertIn("speaker_f0_med_hz", d)

        if SF_NATIVE_SUMMARY_PATH.is_file():
            summary = json.loads(SF_NATIVE_SUMMARY_PATH.read_text(encoding="utf-8"))
            self.assertEqual(float(summary.get("ml_weights_mb", -1.0)), 0.0)
            self.assertGreaterEqual(int(summary.get("n_zh_tokens", 0)), 70000)
            self.assertGreaterEqual(int(summary.get("n_en_tokens", 0)), 18000)

        kb = get_sf_native_multilingual_kb(SF_NATIVE_KB_PATH)
        self.assertIsNotNone(kb)
        assert isinstance(kb, MultilingualIPAKnowledge)
        zh_keys = {k for k in kb.ctx_base if k.startswith("zh:")}
        en_keys = {k for k in kb.ctx_base if k.startswith("en:")}
        self.assertGreaterEqual(len(zh_keys), 35)
        self.assertGreaterEqual(len(en_keys), 41)
        self.assertIn("en:ə", en_keys)
        self.assertIn("en:ɚ", en_keys)

    def test_native_zero_mapping_bilingual_zh_en_and_single_exit_clone(self) -> None:
        kb = get_sf_native_multilingual_kb(SF_NATIVE_KB_PATH)
        self.assertIsNotNone(kb)

        # Native zero-mapping synthesis must make ZERO calls to apply_clone_to_control
        with patch("exp_20_pybot_larynx.multilingual.renderer.apply_clone_to_control") as mock_clone:
            y_mix, ctrl_mix, meta_mix = synthesize_multilingual_native(
                "今天用 PyBot 测试 this smooth bilingual voice。",
                knowledge=kb,
                seed=2026,
            )
            self.assertEqual(mock_clone.call_count, 0)

        self.assertEqual(meta_mix["projections"], 0)
        self.assertGreater(meta_mix["n_phones_by_lang"].get("zh", 0), 0)
        self.assertGreater(meta_mix["n_phones_by_lang"].get("en", 0), 0)
        self.assertTrue(check_wav_sanity(y_mix, SR, expected_sr=SR)["ok"])
        self.assertEqual(len(ctrl_mix.f0_hz), len(ctrl_mix.log_env))

        # Pure EN and Pure ZH native paths
        y_en, _ce, m_en = synthesize_multilingual_native("She walked carefully before sunrise.", knowledge=kb, seed=2027)
        y_zh, _cz, m_zh = synthesize_multilingual_native("今天天气真不错，适合出去走走。", knowledge=kb, seed=2028)
        self.assertEqual(m_en["projections"], 0)
        self.assertEqual(m_zh["projections"], 0)
        self.assertTrue(check_wav_sanity(y_en, SR, expected_sr=SR)["ok"])
        self.assertTrue(check_wav_sanity(y_zh, SR, expected_sr=SR)["ok"])

        # Single-exit cloning to BZNSYP invokes apply_clone_to_control exactly once
        bz_prof = load_base_speaker_profile()
        y_bz, _cb, meta_bz = synthesize_multilingual_native(
            "请帮我 check this status。",
            knowledge=kb,
            target_profile=bz_prof,
            seed=2029,
        )
        self.assertEqual(meta_bz["projections"], 1)
        self.assertTrue(check_wav_sanity(y_bz, SR, expected_sr=SR)["ok"])

    def test_universal_ipa_extensibility_ja_ko_plugins(self) -> None:
        # Verify universal table has Japanese (ɯ, ɕ, ɾ, ɸ) & Korean (k͈, tɕ͈, pʰ) articulatory entries
        for ipa in ("ɯ", "ɕ", "ɾ", "ɸ", "k͈", "tɕ͈", "ɲ"):
            self.assertIn(ipa, IPA_ARTICULATORY_TABLE)
            info = lookup_ipa_info(ipa)
            self.assertEqual(info.ipa, ipa)

        class MockJaFrontend:
            lang = "ja"

            def plan_run(
                self,
                text_or_units: object,
                *,
                base_f0_hz: float,
                speed: float = 1.0,
                pause_scale: float = 1.0,
                seed: int = 2026,
                clause_pos: tuple[float, float] = (0.0, 1.0),
                is_clause_final: bool = True,
                start_fr: int = 0,
            ) -> tuple[list[PhoneEvent], np.ndarray, np.ndarray]:
                # "さくら" -> [s a k ɯ ɾ a] mapped via universal IPA without JA recordings
                seq = [("s", False, False), ("a", True, True), ("k", False, False), ("ɯ", True, True), ("ɾ", True, False), ("a", True, True)]
                evs: list[PhoneEvent] = []
                f0s = []
                vois = []
                fr = int(start_fr)
                for idx, (ipa, voi, is_v) in enumerate(seq):
                    n = 12
                    evs.append(
                        PhoneEvent(
                            ipa=ipa,
                            allophone=ipa,
                            lang="ja",
                            phone_key=f"ja:{ipa}",
                            prosody_tag=1 if idx < 2 else 0,
                            start_fr=fr,
                            end_fr=fr + n,
                            word_id="ja_sakura",
                            word_pos=1 if idx == 0 else (3 if idx == len(seq) - 1 else 2),
                            is_vowel=is_v,
                            is_phrase_final=(idx == len(seq) - 1 and is_clause_final),
                            same_word_prev=(idx > 0),
                            same_word_next=(idx + 1 < len(seq)),
                        )
                    )
                    f0s.append(np.full(n, base_f0_hz * (1.06 if idx < 2 else 0.96) if voi else 0.0))
                    vois.append(np.full(n, voi, dtype=bool))
                    fr += n
                return evs, np.concatenate(f0s), np.concatenate(vois)

        class MockKoFrontend:
            lang = "ko"

            def plan_run(
                self,
                text_or_units: object,
                *,
                base_f0_hz: float,
                speed: float = 1.0,
                pause_scale: float = 1.0,
                seed: int = 2026,
                clause_pos: tuple[float, float] = (0.0, 1.0),
                is_clause_final: bool = True,
                start_fr: int = 0,
            ) -> tuple[list[PhoneEvent], np.ndarray, np.ndarray]:
                # "감사" -> [k a m s a]
                seq = [("k", False, False), ("a", True, True), ("m", True, False), ("s", False, False), ("a", True, True)]
                evs: list[PhoneEvent] = []
                f0s = []
                vois = []
                fr = int(start_fr)
                for idx, (ipa, voi, is_v) in enumerate(seq):
                    n = 12
                    evs.append(
                        PhoneEvent(
                            ipa=ipa,
                            allophone=ipa,
                            lang="ko",
                            phone_key=f"ko:{ipa}",
                            prosody_tag=1,
                            start_fr=fr,
                            end_fr=fr + n,
                            word_id="ko_gamsa",
                            word_pos=1 if idx == 0 else (3 if idx == len(seq) - 1 else 2),
                            is_vowel=is_v,
                            is_phrase_final=(idx == len(seq) - 1 and is_clause_final),
                            same_word_prev=(idx > 0),
                            same_word_next=(idx + 1 < len(seq)),
                        )
                    )
                    f0s.append(np.full(n, base_f0_hz if voi else 0.0))
                    vois.append(np.full(n, voi, dtype=bool))
                    fr += n
                return evs, np.concatenate(f0s), np.concatenate(vois)

        kb = get_sf_native_multilingual_kb(SF_NATIVE_KB_PATH)
        self.assertIsNotNone(kb)
        register_language_frontend("ja", MockJaFrontend())
        register_language_frontend("ko", MockKoFrontend())
        self.assertIsNotNone(get_language_frontend("ja"))
        self.assertIsNotNone(get_language_frontend("ko"))

        y_poly, _cp, meta_poly = synthesize_multilingual_native(
            "今天去看 [lang:ja]sakura[/lang]，再说一句 [lang:ko]gamsa[/lang] and thank you.",
            knowledge=kb,
            seed=2030,
        )
        self.assertEqual(meta_poly["projections"], 0)
        self.assertEqual(set(meta_poly["n_phones_by_lang"].keys()), {"zh", "ja", "ko", "en"})
        self.assertTrue(check_wav_sanity(y_poly, SR, expected_sr=SR)["ok"])

    def test_zero_tv_static_bursts_and_clean_pauses(self) -> None:
        kb = get_sf_native_multilingual_kb(SF_NATIVE_KB_PATH)
        self.assertIsNotNone(kb)
        probes = [
            ("en_studio", "This is a smooth benchmark of bilingual speech synthesis in our studio."),
            ("en_stops", "Pat took ten bright copper pots back to the deep dark cellar."),
            ("mix_multi", "先打一个 tag，then run the nightly build，最后把 log 发给我。"),
            ("zh_news", "今日早间消息，新一代国产开源语音引擎已经正式发布。"),
        ]
        for cid, txt in probes:
            y, ctrl, meta = synthesize_multilingual_native(txt, knowledge=kb, seed=2026)
            self.assertTrue(check_wav_sanity(y, SR, expected_sr=SR)["ok"], cid)
            vm = np.asarray(ctrl.voiced, dtype=bool) & (np.asarray(ctrl.f0_hz, dtype=np.float64) > 40.0)
            self.assertTrue(np.any(vm), cid)
            self.assertLessEqual(float(np.max(ctrl.bap[vm, 0])), 0.28, f"{cid} voiced bap0 spike")
            self.assertLessEqual(float(np.max(ctrl.bap[vm, 1])), 0.52, f"{cid} voiced bap1 spike")
            sil_m = (~np.asarray(ctrl.voiced, dtype=bool)) & (np.max(ctrl.log_env, axis=1) <= -14.0)
            if np.any(sil_m):
                sil_db = float(np.max(ctrl.log_env[sil_m]) * 8.685889638065035)
                self.assertLessEqual(sil_db, -80.0, f"{cid} pause noise floor too high: {sil_db:.1f} dB")
            if cid == "en_studio":
                n_shared = sum(int(d.get("hits", {}).get("ipa_shared", 0)) for d in meta.get("dp_stats", []))
                self.assertEqual(n_shared, 0, "en_studio should hit 100% en:* phones without ipa_shared fallback")


if __name__ == "__main__":
    unittest.main()
