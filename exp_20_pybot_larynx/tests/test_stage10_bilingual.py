"""Unit and forensic verification tests for Stage 10 Bilingual & Unblurred English TTS (`exp_20_pybot_larynx`)."""
from __future__ import annotations

import math
from pathlib import Path
import pickle
import sys
import time
import unittest
from unittest.mock import patch
from typing import Any

import numpy as np
from scipy.fft import rfftfreq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import EXP_DIR, N_FFT, SR  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import (  # noqa: E402
    DEFAULT_UV_LEVEL_BOOST_DB_S7,
    STAGE7_KNOWLEDGE_PATH,
    AcousticKnowledgeStage7,
)
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB, Arbiter  # noqa: E402
from exp_20_pybot_larynx.clone import (  # noqa: E402
    apply_clone_to_control,
    decode_wav_bytes_to_24k_mono,
    extract_clone_profile,
    load_base_speaker_profile,
)
from exp_20_pybot_larynx.crossling.en_frontend_v2 import (  # noqa: E402
    EN_FUNCTION_WORDS,
    plan_en_v2,
    resolve_en_word_v2,
)
from exp_20_pybot_larynx.frontend.prosody_rules import predict_prosody_rules  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import log_env_to_cep  # noqa: E402
from exp_20_pybot_larynx.larynx.synth_v2 import synthesize_larynx_v2  # noqa: E402
from exp_20_pybot_larynx.rules import default_stage5_rules  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity, compute_spectral_distances, evaluate_audio  # noqa: E402
from exp_v6_vocoder import vocoder as v6_voc  # noqa: E402
from exp_v15_bznsyp_tts.eval_v15 import _compute_dtw_f0_rmse_st  # noqa: E402
from exp_v15_bznsyp_tts.index_bznsyp_v15 import load_speaker_assets_v15  # noqa: E402
from pybot_webui.crossling import build_en_run_control, build_utterance_control, build_zh_run_control  # noqa: E402
from pybot_webui import crossling  # noqa: E402
from pybot_webui.presets import VoiceParams  # noqa: E402

STAGE10_KB_PATH = EXP_DIR / "crossling" / "knowledge_en_stage10.npz"
STAGE10_PROSODY_PATH = EXP_DIR / "crossling" / "prosody_en_stage10.npz"
STAGE10_HELDOUT_PKL = EXP_DIR / "crossling_data" / "en_stage10_heldout.pkl"
STAGE10_TRAIN_PKL = EXP_DIR / "crossling_data" / "en_stage10_train_tokens.pkl"
BILINGUAL_REF_STATS_PATH = EXP_DIR / "crossling_data" / "bilingual_ref_stats.json"

_FREQS = rfftfreq(N_FFT, 1.0 / SR)


def _stage10_artifacts_ready() -> bool:
    if not (STAGE7_KNOWLEDGE_PATH.exists() and STAGE10_KB_PATH.exists() and STAGE10_PROSODY_PATH.exists()):
        return False
    try:
        from exp_20_pybot_larynx.crossling import en_stage10_renderer  # noqa: F401
        return True
    except ImportError:
        return False


def _measure_forensic_metrics(y: np.ndarray, sr: int = SR) -> dict[str, Any]:
    """Measure anti-blur sharpness (q8_24), energy dynamics (dyn_p90_20), smooth |dF0| p95, F0 IQR, HNR."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    san = check_wav_sanity(arr, sr, expected_sr=sr)
    fv = v6_voc.analyze(arr, sr=sr)
    vm = fv.voiced & (fv.f0_hz > 40.0)
    cep = log_env_to_cep(fv.log_env, n_cep=64)
    q8_24 = float(np.mean(np.linalg.norm(cep[vm, 8:24], axis=1))) if np.any(vm) else 0.0

    env_rms_db = 10.0 * np.log10(np.mean(np.exp(2.0 * fv.log_env), axis=1) + 1e-18)
    act_mask = env_rms_db > (float(np.max(env_rms_db)) - 37.0)
    act_db = env_rms_db[act_mask]
    dyn_p90_20 = float(np.percentile(act_db, 90) - np.percentile(act_db, 20)) if act_db.size >= 5 else 0.0

    both_v = vm[:-1] & vm[1:]
    if np.any(both_v):
        df0_st = np.abs(12.0 * np.log2(fv.f0_hz[1:][both_v] / fv.f0_hz[:-1][both_v]))
        df0_p95_st = float(np.percentile(df0_st, 95))
    else:
        df0_p95_st = 0.0

    ev = evaluate_audio(arr, sr=sr)
    fh = ev.get("f0_hnr") or {}
    return {
        "sanity_ok": bool(san.get("ok", False)),
        "q8_24": q8_24,
        "dyn_p90_20_db": dyn_p90_20,
        "df0_p95_st": df0_p95_st,
        "f0_iqr_st": float(fh.get("f0_iqr_st") or 0.0),
        "hnr_db": float(fh.get("hnr_db_median") or 0.0),
        "f0_median_hz": float(fh.get("f0_median_hz") or 0.0),
    }


def _strip_stress(arp: str) -> str:
    return "".join(ch for ch in str(arp).upper() if not ch.isdigit())


class TestStage10FrontendV2(unittest.TestCase):
    def test_contextual_read_and_wind(self) -> None:
        cases = [
            ("Please read the note.", "read", ["R", "IY1", "D"]),
            ("I read the note yesterday.", "read", ["R", "EH1", "D"]),
            ("Wind the clock.", "wind", ["W", "AY1", "N", "D"]),
            ("The cold wind blew.", "wind", ["W", "IH1", "N", "D"]),
            ("Read the live audio stream.", "live", ["L", "AY1", "V"]),
            ("We live in a studio.", "live", ["L", "IH1", "V"]),
            ("Close the door carefully.", "close", ["K", "L", "OW1", "Z"]),
            ("Stand close to the mic.", "close", ["K", "L", "OW1", "S"]),
            ("Please record the demo.", "record", ["R", "IH0", "K", "AO1", "R", "D"]),
            ("This is a new record today.", "record", ["R", "EH1", "K", "ER0", "D"]),
            ("We will present the results.", "present", ["P", "R", "IH0", "Z", "EH1", "N", "T"]),
            ("At the present time.", "present", ["P", "R", "EH1", "Z", "AH0", "N", "T"]),
            ("You can use the tool.", "use", ["Y", "UW1", "Z"]),
            ("Heavy lead pipe broke.", "lead", ["L", "EH1", "D"]),
        ]
        for text, word, expected in cases:
            with self.subTest(text=text):
                reading = next(row for row in plan_en_v2(text)["words"] if row["lower"] == word)
                self.assertEqual(reading["arpabet"], expected)

    def test_the_prevocalic_and_preconsonantal(self) -> None:
        w_apple = next(r for r in plan_en_v2("Eat the apple now.")["words"] if r["lower"] == "the")
        w_hour = next(r for r in plan_en_v2("Wait the hour out.")["words"] if r["lower"] == "the")
        w_book = next(r for r in plan_en_v2("Read the book now.")["words"] if r["lower"] == "the")
        w_univ = next(r for r in plan_en_v2("Visit the university today.")["words"] if r["lower"] == "the")
        w_foc = next(r for r in plan_en_v2("Not *the* book.")["words"] if r["lower"] == "the")
        self.assertEqual(w_apple["arpabet"], ["DH", "IY0"])
        self.assertEqual(w_hour["arpabet"], ["DH", "IY0"])
        self.assertEqual(w_book["arpabet"], ["DH", "AH0"])
        self.assertEqual(w_univ["arpabet"], ["DH", "AH0"])
        self.assertEqual(w_foc["arpabet"], ["DH", "IY1"])

    """Gate 1: G2P dental fricatives (`this -> DH`) and function-word weak forms (`stress == 0`)."""

    def test_this_and_voiced_dental_fricatives(self) -> None:
        for w in ("this", "that", "the", "these", "those", "they", "them", "their", "there", "then", "than", "with", "smooth", "breathe", "other"):
            arpa, _src = resolve_en_word_v2(w)
            bases = [_strip_stress(a) for a in arpa]
            self.assertIn("DH", bases, f"Expected DH in {w!r}, got {arpa}")
            self.assertNotIn("TH", bases, f"Unexpected voiceless TH in {w!r}, got {arpa}")

        # Voiceless TH must stay TH
        arpa_think, _ = resolve_en_word_v2("think")
        self.assertEqual(_strip_stress(arpa_think[0]), "TH")

        ir = plan_en_v2("This is a smooth test of this system.")
        first_word = ir["words"][0]
        self.assertEqual(first_word["lower"], "this")
        self.assertEqual(_strip_stress(first_word["arpabet"][0]), "DH")
        self.assertEqual(first_word["ipa_phones"][0], "ð")

    def test_function_words_unstressed_weak_forms(self) -> None:
        sent = "Turn it on and take an apple in the box to one of us today."
        ir = plan_en_v2(sent)
        target_funcs = {"it", "on", "and", "an", "in", "the", "to", "of"}
        seen: set[str] = set()
        for w in ir["words"]:
            if w["lower"] in target_funcs:
                seen.add(w["lower"])
                self.assertTrue(w["is_func"])
                self.assertEqual(w["stress"], 0, f"Function word {w['lower']!r} not demoted: {w}")
        self.assertEqual(seen, target_funcs)

        for ph in ir["phones"]:
            if ph["word"].lower() in target_funcs and ph["is_vowel"]:
                self.assertEqual(ph["stress"], 0, f"Vowel in {ph['word']!r} has non-zero stress: {ph}")

        # Content words keep primary stress
        content_stresses = {w["lower"]: w["stress"] for w in ir["words"] if not w["is_func"]}
        self.assertEqual(content_stresses.get("turn"), 1)
        self.assertEqual(content_stresses.get("apple"), 1)
        self.assertEqual(content_stresses.get("today"), 1)

    def test_focused_or_phrase_final_retains_stress(self) -> None:
        ir_foc = plan_en_v2("Look at *this* right now.")
        w_this = next(w for w in ir_foc["words"] if w["lower"] == "this")
        self.assertTrue(w_this["is_focused"])
        self.assertEqual(w_this["stress"], 1)
        self.assertEqual(_strip_stress(w_this["arpabet"][0]), "DH")

        ir_fin = plan_en_v2("Who are you talking to?")
        w_to = ir_fin["words"][-1]
        self.assertEqual(w_to["lower"], "to")
        self.assertTrue(w_to["is_phrase_final"])
        self.assertEqual(w_to["stress"], 1)

    def test_tech_and_bilingual_vocabulary_coverage(self) -> None:
        self.assertGreaterEqual(len(EN_FUNCTION_WORDS), 60)
        for term in ("pybot", "studio", "benchmark", "lingual", "synthesis"):
            arpa, _ = resolve_en_word_v2(term)
            self.assertGreaterEqual(len(arpa), 4, f"Unexpected short G2P for {term}: {arpa}")


class TestStage11AcousticGuards(unittest.TestCase):
    def test_projected_treble_limit_preserves_unvoiced_and_low_bands(self) -> None:
        from exp_20_pybot_larynx.crossling import en_stage10_renderer

        native = crossling._silence_control(12)
        native.log_env[:] = -4.0
        native.voiced[2:9] = True
        projected = native.copy()
        projected.log_env[:, 260:] += 1.4
        original = projected.log_env.copy()
        en_stage10_renderer._limit_projected_treble(native, projected, projected.voiced)
        np.testing.assert_array_equal(projected.log_env[~projected.voiced], original[~projected.voiced])
        np.testing.assert_array_equal(projected.log_env[:, :120], original[:, :120])
        self.assertTrue(np.all(projected.log_env[projected.voiced, 350:] < original[projected.voiced, 350:]))

    def test_pitch_bridge_level_change_is_bounded_and_voiced_only(self) -> None:
        control = crossling._silence_control(80)
        control.voiced[:] = True
        control.voiced[40:42] = False
        control.voiced[50:53] = False
        control.f0_hz[control.voiced] = 210.0
        hop = control.hop
        samples = np.repeat(np.r_[np.full(40, 0.1), np.full(40, 0.4)], hop)
        original = samples.copy()
        crossling.soften_mixed_onset_levels(samples, control, [(0, 40, 80)])
        self.assertAlmostEqual(20 * np.log10(original[45 * hop] / samples[45 * hop]), 3.5, places=8)
        np.testing.assert_array_equal(samples[:42 * hop], original[:42 * hop])
        np.testing.assert_array_equal(samples[50 * hop:53 * hop], original[50 * hop:53 * hop])
        np.testing.assert_array_equal(samples[68 * hop:], original[68 * hop:])

        quiet = np.repeat(np.r_[np.full(40, 0.4), np.full(40, 0.1)], hop)
        same = quiet.copy()
        crossling.soften_mixed_onset_levels(quiet, control, [(0, 40, 80)])
        np.testing.assert_array_equal(quiet, same)


class TestStage10ArtifactsZeroML(unittest.TestCase):
    """Verify Stage 10 artifacts are strictly zero-ML, compact, and readable with allow_pickle=False."""

    @unittest.skipUnless(STAGE10_PROSODY_PATH.exists(), "prosody_en_stage10.npz not built yet")
    def test_prosody_artifact_compact_and_zero_pickle(self) -> None:
        size_kb = STAGE10_PROSODY_PATH.stat().st_size / 1024.0
        self.assertLessEqual(size_kb, 200.0, f"prosody_en_stage10.npz too large: {size_kb:.1f} KB")
        with np.load(STAGE10_PROSODY_PATH, allow_pickle=False) as d:
            self.assertGreater(len(d.files), 0)
            for k in d.files:
                arr = d[k]
                if np.issubdtype(arr.dtype, np.floating):
                    self.assertTrue(np.all(np.isfinite(arr)), f"Non-finite values in {k}")

    @unittest.skipUnless(STAGE10_KB_PATH.exists(), "knowledge_en_stage10.npz not built yet")
    def test_knowledge_artifact_compact_and_zero_pickle(self) -> None:
        size_mb = STAGE10_KB_PATH.stat().st_size / (1024.0 * 1024.0)
        self.assertLessEqual(size_mb, 25.0, f"knowledge_en_stage10.npz too large: {size_mb:.2f} MB")
        with np.load(STAGE10_KB_PATH, allow_pickle=False) as d:
            self.assertGreater(len(d.files), 0)
            for k in d.files:
                arr = d[k]
                if np.issubdtype(arr.dtype, np.floating):
                    self.assertTrue(np.all(np.isfinite(arr)), f"Non-finite values in {k}")


@unittest.skipUnless(_stage10_artifacts_ready(), "Stage 10 artifacts or renderer not built yet")
class TestStage10ForensicAndBilingualUnity(unittest.TestCase):
    kb: AcousticKnowledgeStage7
    arbiter: Arbiter
    rules: list
    phasor: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
        cls.arbiter = Arbiter()
        cls.rules = default_stage5_rules()
        assets = load_speaker_assets_v15()
        cls.phasor = np.asarray(assets["glottal_bank"].coherent_phasor, dtype=np.complex128)

    def test_mandarin_continuation_keeps_real_sentence_break(self) -> None:
        self.assertEqual(predict_prosody_rules("我说的是")["units"][-1]["prosodic_break"], 4)
        continued = predict_prosody_rules("我说的是", continues_after=True)
        self.assertEqual(continued["units"][-1]["prosodic_break"], 1)
        ended = predict_prosody_rules("我说的是。", continues_after=True)
        self.assertEqual(ended["units"][-1]["prosodic_break"], 4)

    def test_english_continuation_has_no_utterance_final_phone(self) -> None:
        continued = plan_en_v2("Let me know", is_clause_final=False)
        ended = plan_en_v2("Let me know.")
        self.assertFalse(continued["phones"][-1]["is_utterance_final"])
        self.assertTrue(ended["phones"][-1]["is_utterance_final"])

    def test_mixed_clause_avoids_extra_english_pitch_expansion(self) -> None:
        params = VoiceParams.from_dict({"preset_id": "female_std", "auto_breath": False})
        build = crossling.build_en_run_control
        with patch.object(crossling, "build_en_run_control", wraps=build) as spy:
            build_utterance_control(
                "你方便的话 let me know 我再安排。", "mixed_pitch",
                self.kb, self.arbiter, self.rules, params,
            )
            self.assertFalse(spy.call_args.kwargs["expand_projected_pitch"])
            build_utterance_control(
                "Let me know.", "english_pitch",
                self.kb, self.arbiter, self.rules, params,
            )
            self.assertTrue(spy.call_args.kwargs["expand_projected_pitch"])

    def test_connected_mixed_pitch_and_context(self) -> None:
        from exp_20_pybot_larynx.crossling import en_stage10_renderer

        params = VoiceParams.from_dict({"preset_id": "female_std", "auto_breath": False})
        builder = en_stage10_renderer.build_en_stage10_control
        readings = []

        def capture(phones, *args, **kwargs):
            readings.append([phone["ipa"] for phone in phones if not phone["is_pause"]][:3])
            return builder(phones, *args, **kwargs)

        join = crossling._anchor_english_onset_pitch
        with patch.object(crossling, "_anchor_english_onset_pitch", wraps=join) as bridge, patch.object(
            en_stage10_renderer, "build_en_stage10_control", side_effect=capture,
        ):
            build_utterance_control(
                "今天的任务是 read the note 然后归档。", "read_task",
                self.kb, self.arbiter, self.rules, params,
            )
            bridge.assert_called_once()
            build_utterance_control(
                "我昨天 read the note 然后归档了。", "read_past",
                self.kb, self.arbiter, self.rules, params,
            )
            bridge.reset_mock()
            build_utterance_control(
                "我先说完，[pause 0.5s] read the note。", "paused_pitch",
                self.kb, self.arbiter, self.rules, params,
            )
            bridge.assert_not_called()
        self.assertEqual(readings[:2], [["ɹ", "i", "d"], ["ɹ", "ɛ", "d"]])

    def test_mixed_clause_matches_low_band_bap_without_changing_consonants(self) -> None:
        params = VoiceParams.from_dict({"preset_id": "female_std", "auto_breath": False})
        match = crossling._match_mixed_voice_source

        def check_match(zh_controls, en_controls):
            before = [ctrl.bap.copy() for ctrl in en_controls]
            zh = np.concatenate([ctrl.bap[ctrl.voiced, :2] for ctrl in zh_controls])
            en_before = np.concatenate([arr[ctrl.voiced, :2] for ctrl, arr in zip(en_controls, before)])
            match(zh_controls, en_controls)
            en_after = np.concatenate([ctrl.bap[ctrl.voiced, :2] for ctrl in en_controls])
            target = np.median(zh, axis=0)
            gap_before = np.abs(np.median(en_before, axis=0) - target)
            gap_after = np.abs(np.median(en_after, axis=0) - target)
            self.assertTrue(np.all(gap_after < gap_before))
            for ctrl, arr in zip(en_controls, before):
                np.testing.assert_array_equal(ctrl.bap[~ctrl.voiced], arr[~ctrl.voiced])
                np.testing.assert_array_equal(ctrl.bap[:, 2:], arr[:, 2:])

        with patch.object(crossling, "_match_mixed_voice_source", side_effect=check_match) as match_spy:
            _, meta = build_utterance_control(
                "我说的是 ice cream 你听到了吗？", "mixed_voice",
                self.kb, self.arbiter, self.rules, params,
            )
        match_spy.assert_called_once()
        self.assertGreater(meta["n_en"], 0)
        self.assertGreater(meta["n_zh"], 0)

    def test_antiblur_and_dynamic_naturalness_forensics(self) -> None:
        vp = VoiceParams.from_dict({"preset_id": "female_std"})
        en_sentences = [
            "Sixty seven friends test our cross lingual speech synthesis clearly.",
            "This is a smooth benchmark of bilingual speech synthesis in our studio.",
            "Turn it on and take an apple to the station in the morning.",
        ]
        metrics_list: list[dict[str, Any]] = []
        rtfs: list[float] = []
        for idx, txt in enumerate(en_sentences):
            ir = plan_en_v2(txt)
            t0 = time.perf_counter()
            ctrl, subh = build_en_run_control(
                ir["units"],
                float(self.kb.speaker_f0_med_hz),
                vp,
                kb=self.kb,
            )
            y = synthesize_larynx_v2(
                ctrl,
                glottal_mode="dynamic_coherent",
                coherent_phasor=self.phasor,
                uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7,
                subharm_strength=float(subh),
                seed=vp.seed + idx,
            )
            rtf = (time.perf_counter() - t0) / max(0.05, len(y) / float(SR))
            rtfs.append(rtf)
            fm = _measure_forensic_metrics(y, sr=SR)
            self.assertTrue(fm["sanity_ok"], f"WAV sanity failed on {txt!r}")
            metrics_list.append(fm)

        med_q8_24 = float(np.median([m["q8_24"] for m in metrics_list]))
        med_dyn = float(np.median([m["dyn_p90_20_db"] for m in metrics_list]))
        med_df0_p95 = float(np.median([m["df0_p95_st"] for m in metrics_list]))
        med_iqr = float(np.median([m["f0_iqr_st"] for m in metrics_list]))
        med_hnr = float(np.median([m["hnr_db"] for m in metrics_list]))
        med_rtf = float(np.median(rtfs))

        self.assertGreaterEqual(med_q8_24, 10.8, f"Formant sharpness q8_24 blurred: {med_q8_24:.2f} < 10.8")
        self.assertTrue(10.5 <= med_dyn <= 19.0, f"Energy dynamic range out of [10.5, 19.0]: {med_dyn:.2f} dB")
        self.assertGreaterEqual(med_df0_p95, 0.40, f"F0 motion too stiff: |dF0| p95={med_df0_p95:.3f} < 0.40 st")
        self.assertGreaterEqual(med_iqr, 2.8, f"EN F0 IQR too flat: {med_iqr:.2f} < 2.8 st")
        self.assertGreaterEqual(med_hnr, 14.2, f"HNR too low: {med_hnr:.2f} < 14.2 dB")
        self.assertLessEqual(med_rtf, 0.06, f"Median RTF too slow: {med_rtf:.4f} > 0.06")

    def test_same_speaker_bilingual_unity_and_clone(self) -> None:
        vp = VoiceParams.from_dict({"preset_id": "female_std"})
        zh_txt = "今天四十三位朋友一起测试跨语种语音合成，声音清晰自然。"
        en_txt = "Sixty seven friends test our cross lingual speech synthesis clearly."

        c_zh, _ = build_zh_run_control(zh_txt, "s10_zh", self.kb, self.arbiter, self.rules, vp)
        en_u = plan_en_v2(en_txt)["units"]
        c_en, _ = build_en_run_control(en_u, float(self.kb.speaker_f0_med_hz), vp, kb=self.kb)

        y_zh = synthesize_larynx_v2(
            c_zh, glottal_mode="dynamic_coherent", coherent_phasor=self.phasor,
            uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7, seed=2026,
        )
        y_en = synthesize_larynx_v2(
            c_en, glottal_mode="dynamic_coherent", coherent_phasor=self.phasor,
            uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7, seed=2026,
        )

        def _active_rms_db(arr: np.ndarray) -> float:
            pk = float(np.max(np.abs(arr))) if arr.size else 0.0
            act = arr[np.abs(arr) > 0.02 * pk] if pk > 1e-9 else arr
            rms = float(np.sqrt(np.mean(act * act))) if act.size else 1e-12
            return 20.0 * math.log10(max(rms, 1e-12))

        raw_rms_diff = abs(_active_rms_db(y_zh) - _active_rms_db(y_en))
        self.assertLessEqual(raw_rms_diff, 2.0, f"Raw RMS diff too high: {raw_rms_diff:.2f} dB")

        vm_z = c_zh.voiced & (c_zh.f0_hz > 40.0)
        vm_e = c_en.voiced & (c_en.f0_hz > 40.0)
        m_v = (_FREQS >= 80.0) & (_FREQS <= 11500.0)
        voiced_ltas_rmse = float(
            np.sqrt(np.mean(((np.mean(c_zh.log_env[vm_z], 0) - np.mean(c_en.log_env[vm_e], 0))[m_v] * NEPER_TO_DB) ** 2))
        )
        self.assertLessEqual(voiced_ltas_rmse, 3.5, f"Voiced LTAS RMSE={voiced_ltas_rmse:.2f} > 3.5 dB")

        um_z = (~c_zh.voiced) & (np.mean(c_zh.log_env[:, _FREQS >= 1500.0], axis=1) > -9.5)
        um_e = (~c_en.voiced) & (np.mean(c_en.log_env[:, _FREQS >= 1500.0], axis=1) > -9.5)
        uv_ltas_rmse = float(
            np.sqrt(np.mean(((np.mean(c_zh.log_env[um_z], 0) - np.mean(c_en.log_env[um_e], 0))[_FREQS >= 800.0] * NEPER_TO_DB) ** 2))
        )
        self.assertLessEqual(uv_ltas_rmse, 5.5, f"UV LTAS RMSE={uv_ltas_rmse:.2f} > 5.5 dB")

        bap_mae = float(np.mean(np.abs(np.median(c_zh.bap[vm_z], 0) - np.median(c_en.bap[vm_e], 0))))
        self.assertLessEqual(bap_mae, 0.06, f"BAP6 MAE={bap_mae:.4f} > 0.06")

        f0_diff_st = abs(12.0 * math.log2(float(np.median(c_en.f0_hz[vm_e])) / float(np.median(c_zh.f0_hz[vm_z]))))
        self.assertLessEqual(f0_diff_st, 0.8, f"|dF0|={f0_diff_st:.2f} > 0.8 st")

        # Anti-buzz human glottal guard (Stage 11: no robotic zero-jitter / zero-BAP clamps)
        en_jit_med = float(np.median(c_en.jitter_frac[vm_e]))
        en_bap1_med = float(np.median(c_en.bap[vm_e, 1]))
        self.assertGreaterEqual(en_jit_med, 0.0040, f"Robotic jitter clamp detected: {en_jit_med:.5f} < 0.0040")
        self.assertGreaterEqual(en_bap1_med, 0.14, f"Robotic BAP1 clamp detected: {en_bap1_med:.4f} < 0.14")

        # Mixed code-switching + emotions
        for emo in ("auto", "excited", "whisper"):
            vp_e = VoiceParams.from_dict({"preset_id": "female_std", "emotion_mode": emo, "emotion_intensity": 0.9})
            ctrl_m, meta_m = build_utterance_control(
                "今天用 PyBot 测试 this smooth bilingual speech，效果非常自然！",
                f"s10_{emo}",
                self.kb,
                self.arbiter,
                self.rules,
                vp_e,
            )
            self.assertGreater(meta_m["n_zh"], 0)
            self.assertGreater(meta_m["n_en"], 0)
            ym = synthesize_larynx_v2(
                ctrl_m, glottal_mode="dynamic_coherent", coherent_phasor=self.phasor,
                uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7,
                subharm_strength=float(meta_m.get("subharm_strength", 0.0)),
                seed=vp_e.seed,
            )
            self.assertTrue(check_wav_sanity(ym, SR, expected_sr=SR)["ok"])

        # SSB0073 Male Voice Clone Bilingual Check
        sample_path = REPO_ROOT / "legacy" / "exp_v8_data" / "corpus" / "zh" / "SSB0073" / "SSB00730001.wav"
        if sample_path.exists():
            wav_24k, _ = decode_wav_bytes_to_24k_mono(sample_path.read_bytes())
            tgt_prof = extract_clone_profile([wav_24k], name="SSB0073_s10")
            base_prof = load_base_speaker_profile()
            c_czh, _, _ = apply_clone_to_control(c_zh, base_prof, tgt_prof, strength=0.90, clone_f0=True, base_coherent_phasor=self.phasor)
            c_cen, _, _ = apply_clone_to_control(c_en, base_prof, tgt_prof, strength=0.90, clone_f0=True, base_coherent_phasor=self.phasor)
            vm_cz = c_czh.voiced & (c_czh.f0_hz > 40.0)
            vm_ce = c_cen.voiced & (c_cen.f0_hz > 40.0)
            clone_f0_diff = abs(12.0 * math.log2(float(np.median(c_cen.f0_hz[vm_ce])) / float(np.median(c_czh.f0_hz[vm_cz]))))
            self.assertLessEqual(clone_f0_diff, 1.0, f"SSB0073 clone_f0_diff={clone_f0_diff:.2f} > 1.0 st")


@unittest.skipUnless(_stage10_artifacts_ready() and STAGE10_HELDOUT_PKL.exists(), "Stage 10 held-out artifacts not ready")
class TestStage10HeldOutZeroLeakage(unittest.TestCase):
    def test_heldout_mel_cd_duration_and_f0(self) -> None:
        from exp_20_pybot_larynx.clone.profile import VoiceCloneProfile
        from exp_20_pybot_larynx.crossling.en_larynx_builder import build_speaker_anchored_en_control
        from exp_20_pybot_larynx.crossling.en_prosody_stage10 import predict_en_prosody_stage10
        from exp_20_pybot_larynx.crossling.en_stage10_renderer import (
            EN_SOURCE_PROFILE_PATH,
            build_stage10_en_control,
            get_en_stage10_knowledge,
        )
        from exp_20_pybot_larynx.frontend.prosody_rules import _safe_plan_v13

        with open(STAGE10_HELDOUT_PKL, "rb") as f:
            held_data = pickle.load(f)
        utts = held_data.get("utterances", held_data) if isinstance(held_data, dict) else list(held_data)
        self.assertGreaterEqual(len(utts), 20)

        # Verify zero leakage against train IDs if train pkl exists
        if STAGE10_TRAIN_PKL.exists():
            with open(STAGE10_TRAIN_PKL, "rb") as f:
                tr_data = pickle.load(f)
            tr_ids = set(tr_data.get("train_ids") or tr_data.get("utterance_ids") or []) if isinstance(tr_data, dict) else {u["id"] for u in tr_data}
            ho_ids = {u["id"] for u in utts}
            self.assertEqual(len(ho_ids & tr_ids), 0, "Data leakage between train and heldout!")

        kb = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
        kb10 = get_en_stage10_knowledge()
        assert kb10 is not None
        assets = load_speaker_assets_v15()
        phasor = np.asarray(assets["glottal_bank"].coherent_phasor, dtype=np.complex128)
        src_phasor = (
            VoiceCloneProfile.from_json(EN_SOURCE_PROFILE_PATH.read_text(encoding="utf-8")).reconstruct_coherent_phasor()
            if EN_SOURCE_PROFILE_PATH.exists()
            else None
        )
        vp = VoiceParams.from_dict({"preset_id": "female_std"})

        mel_10_list: list[float] = []
        mel_9b_list: list[float] = []
        f0_rmse_list: list[float] = []
        dur_errs: list[float] = []

        for idx, item in enumerate(utts[:30]):
            text = str(item["text"])
            ref_wav = np.asarray(item["wav"], dtype=np.float64)

            # Native Stage 10 EN (project_to_zh_speaker=False)
            c10_nat, subh10 = build_stage10_en_control(
                text,
                float(kb10.speaker_f0_med_hz),
                vp,
                zh_kb=kb,
                en_kb=kb10,
                project_to_zh_speaker=False,
            )
            y10 = synthesize_larynx_v2(
                c10_nat,
                glottal_mode="dynamic_coherent" if src_phasor is not None else "parametric_lf",
                coherent_phasor=src_phasor,
                uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7,
                subharm_strength=float(subh10),
                seed=2026 + idx,
            )
            sd10 = compute_spectral_distances(ref_wav, y10, sr=SR)
            if sd10 and sd10.get("mel_lsd_db") is not None:
                mel_10_list.append(float(sd10["mel_lsd_db"]))
            f0_rmse_list.append(float(_compute_dtw_f0_rmse_st(y10, ref_wav)))

            # Stage 9B baseline
            en_u_9b = [u for u in _safe_plan_v13(text)["units"] if u.get("lang") == "en"]
            c9b, _ = build_speaker_anchored_en_control(en_u_9b, float(kb.speaker_f0_med_hz), vp, kb, use_en_articulation=True)
            y9b = synthesize_larynx_v2(
                c9b, glottal_mode="dynamic_coherent", coherent_phasor=phasor,
                uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7, seed=2026 + idx,
            )
            sd9b = compute_spectral_distances(ref_wav, y9b, sr=SR)
            if sd9b and sd9b.get("mel_lsd_db") is not None:
                mel_9b_list.append(float(sd9b["mel_lsd_db"]))

            # Phone duration MAE against aligned tokens
            ref_durs = [float(p["dur_ms"]) for p in item.get("phones", [])]
            if not ref_durs and item.get("bounds"):
                ref_durs = [(int(b[2]) - int(b[0])) * 5.0 for b in item["bounds"]]
            pro10 = predict_en_prosody_stage10(text, base_f0_hz=float(kb10.speaker_f0_med_hz), seed=2026 + idx)
            pred_durs = pro10.get("phone_durs_ms") or []
            if ref_durs and pred_durs:
                n_m = min(len(ref_durs), len(pred_durs))
                for k in range(n_m):
                    dur_errs.append(abs(float(pred_durs[k]) - float(ref_durs[k])))

        mel10 = float(np.median(mel_10_list))
        mel9b = float(np.median(mel_9b_list))
        f0_rmse = float(np.median(f0_rmse_list))
        self.assertLessEqual(mel10, 6.5, f"Held-out Native Mel-CD={mel10:.2f} > 6.5 dB")
        self.assertLess(mel10, mel9b, f"Stage 10 Mel-CD ({mel10:.2f}) did not improve over Stage 9B ({mel9b:.2f})")
        self.assertLessEqual(f0_rmse, 4.2, f"Held-out F0 RMSE={f0_rmse:.2f} > 4.2 st")
        if dur_errs:
            dur_mae = float(np.mean(dur_errs))
            self.assertLessEqual(dur_mae, 22.0, f"Held-out Phone Duration MAE={dur_mae:.2f} > 22.0 ms")


class TestStage12SFBilingualConsistency(unittest.TestCase):
    """Verify Stage 12 SF_bilingual same-speaker consistency decomposition & synthesis."""

    def test_stage12_sf_artifacts_zero_ml_and_bounds(self) -> None:
        import json

        from exp_20_pybot_larynx.crossling.en_prosody_stage10 import ProsodyEnStage10Model
        from exp_20_pybot_larynx.crossling.en_stage10_renderer import (
            EN_SF_STAGE12_KB_PATH,
            SF_CONSISTENCY_PATH,
            SF_ZH_PROFILE_PATH,
            is_stage12_sf_ready,
            load_sf_bilingual_consistency,
        )
        from exp_20_pybot_larynx.crossling.knowledge_en_stage10 import AcousticKnowledgeEnStage10

        self.assertTrue(is_stage12_sf_ready(), "Stage 12 SF bilingual artifacts missing.")
        self.assertTrue(SF_ZH_PROFILE_PATH.exists())
        prosody_sf_path = EN_SF_STAGE12_KB_PATH.parent / "prosody_en_sf_stage12.npz"
        self.assertTrue(prosody_sf_path.exists())

        with np.load(EN_SF_STAGE12_KB_PATH, allow_pickle=False) as d:
            self.assertGreater(len(d.files), 0)
            for k in d.files:
                if np.issubdtype(d[k].dtype, np.floating):
                    self.assertTrue(np.all(np.isfinite(d[k])), f"Non-finite weights in {k}")
        kb12 = AcousticKnowledgeEnStage10.load(EN_SF_STAGE12_KB_PATH)
        self.assertGreaterEqual(len(kb12.ctx_base), 39)

        with np.load(prosody_sf_path, allow_pickle=False) as d:
            self.assertGreater(len(d.files), 0)
            for k in d.files:
                if np.issubdtype(d[k].dtype, np.floating):
                    self.assertTrue(np.all(np.isfinite(d[k])), f"Non-finite prosody weights in {k}")
        pm = ProsodyEnStage10Model(prosody_sf_path)
        self.assertTrue(pm.loaded)

        cons = load_sf_bilingual_consistency()
        self.assertIsNotNone(cons)
        self.assertLessEqual(float(cons["delta_voiced_ltas_shape_rmse_db"]), 2.0)
        self.assertLessEqual(abs(float(cons["delta_rms_db_en_minus_zh"])), 1.5)
        self.assertGreaterEqual(float(cons["iqr_ratio_en_over_zh"]), 0.70)
        self.assertLessEqual(float(cons["iqr_ratio_en_over_zh"]), 1.30)
        deltas_bap = cons.get("delta_bap6_en_minus_zh") or []
        self.assertEqual(len(deltas_bap), 6)
        self.assertTrue(all(abs(float(x)) <= 0.15 for x in deltas_bap))

        # Verify zero leakage between SF train and heldout sets
        manifest_path = SF_CONSISTENCY_PATH.parent / "sf_bilingual" / "manifest.json"
        if manifest_path.exists():
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(set(man.get("train_ids", [])) & set(man.get("heldout_ids", []))), 0)

    def test_stage12_sf_pivot_mixed_synthesis_and_consistency(self) -> None:
        from exp_20_pybot_larynx.clone import VoiceCloneProfile, apply_clone_to_control, load_base_speaker_profile
        from exp_20_pybot_larynx.crossling.en_stage10_renderer import SF_ZH_PROFILE_PATH, is_stage12_sf_ready
        from exp_20_pybot_larynx.larynx.synth_v2 import synthesize_larynx_v2
        from exp_20_pybot_larynx.rules import default_stage5_rules
        from exp_20_pybot_larynx.verify_stage10_bilingual import SR, _synth_stage10_utt, compute_forensic_metrics
        from exp_v5_eval.metrics import check_wav_sanity
        from exp_v15_bznsyp_tts.index_bznsyp_v15 import load_speaker_assets_v15
        from pybot_webui.crossling import build_utterance_control, soften_mixed_onset_levels
        from pybot_webui.engine import apply_voice_params
        from pybot_webui.presets import VoiceParams

        self.assertTrue(is_stage12_sf_ready())
        kb = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
        arb = Arbiter()
        rules = default_stage5_rules()
        vp = VoiceParams.from_dict({"preset_id": "female_std"})
        phasor = np.asarray(load_speaker_assets_v15()["glottal_bank"].coherent_phasor, dtype=np.complex128)

        text = "今天用 PyBot 测试 this smooth cross lingual speech，表现非常自然。"
        _, y_bz, _rtf, meta_bz = _synth_stage10_utt(
            text, "test_s12_bz", kb, arb, rules, vp, phasor, seed=2026, use_stage10_en=True
        )
        self.assertGreater(int(meta_bz.get("n_zh", 0)), 0)
        self.assertGreater(int(meta_bz.get("n_en", 0)), 0)
        self.assertTrue(check_wav_sanity(y_bz, SR, expected_sr=SR)["ok"])
        m_bz = compute_forensic_metrics(y_bz, SR)
        self.assertGreaterEqual(float(m_bz["q8_24"]), 10.0)
        self.assertGreaterEqual(float(m_bz["hnr_db"]), 12.5)

        # Native SF clone path check
        base_prof = load_base_speaker_profile()
        sf_zh_prof = VoiceCloneProfile.from_json(SF_ZH_PROFILE_PATH.read_text(encoding="utf-8"))
        ctrl, meta = build_utterance_control(text, "test_s12_sf", kb, arb, rules, vp, use_stage10_en=True)
        ctrl_c, eff_ph, _ = apply_clone_to_control(
            ctrl, base_prof, sf_zh_prof, strength=0.92, clone_f0=True, clone_glottal=True, base_coherent_phasor=phasor
        )
        mod = apply_voice_params(ctrl_c, vp, 2026)
        y_sf = synthesize_larynx_v2(
            mod,
            glottal_mode="dynamic_coherent",
            coherent_phasor=eff_ph if eff_ph is not None else sf_zh_prof.reconstruct_coherent_phasor(),
            uv_level_boost_db=0.9,
            subharm_strength=float(meta.get("subharm_strength", 0.0)),
            seed=2026,
        )
        soften_mixed_onset_levels(y_sf, mod, meta["level_onsets"])
        self.assertTrue(check_wav_sanity(y_sf, SR, expected_sr=SR)["ok"])
        m_sf = compute_forensic_metrics(y_sf, SR)
        f0_diff_st = abs(12.0 * np.log2(max(40.0, float(m_sf["f0_median_hz"])) / max(40.0, float(sf_zh_prof.f0_median_hz))))
        self.assertLessEqual(f0_diff_st, 2.5, f"SF clone F0 drifted by {f0_diff_st:.2f} st")

    def test_pure_en_neutral_clarity_and_expressive_range(self) -> None:
        from unittest.mock import patch

        from exp_20_pybot_larynx.crossling import en_stage10_renderer
        from exp_20_pybot_larynx.crossling.en_frontend_v2 import plan_en_v2
        from exp_20_pybot_larynx.crossling.en_stage10_renderer import build_stage10_en_control
        from exp_20_pybot_larynx.expressive import parse_expressive_text
        from exp_20_pybot_larynx.larynx.synth_v2 import synthesize_larynx_v2
        from exp_20_pybot_larynx.verify_stage10_bilingual import SR, compute_forensic_metrics
        from exp_v5_eval.metrics import check_wav_sanity
        from exp_v14_fluent_zh_tts.eval_v14 import _rms_norm
        from exp_v15_bznsyp_tts.index_bznsyp_v15 import load_speaker_assets_v15
        from pybot_webui.presets import VoiceParams

        kb = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
        phasor = np.asarray(load_speaker_assets_v15()["glottal_bank"].coherent_phasor, dtype=np.complex128)
        vp_neu = VoiceParams.from_dict({"preset_id": "female_std", "emotion_mode": "neutral"})

        # 1. Neutral pure-EN clarity checks on en_studio & en_stops
        studio_txt = "This is a smooth benchmark of bilingual speech synthesis in our studio."
        stops_txt = "Pat took ten bright copper pots back to the deep dark cellar."
        for txt, min_hnr in ((studio_txt, 14.5), (stops_txt, 13.8)):
            ctrl, subh = build_stage10_en_control(
                txt, float(kb.speaker_f0_med_hz), vp_neu, zh_kb=kb, project_to_zh_speaker=True
            )
            vm = ctrl.voiced & (ctrl.f0_hz > 40.0)
            bap0_p90 = float(np.percentile(ctrl.bap[vm, 0], 90))
            rd50 = float(np.median(ctrl.rd[vm])) if ctrl.rd is not None else 1.0
            self.assertLessEqual(bap0_p90, 0.16, f"bap0 p90 too hoarse: {bap0_p90:.3f}")
            self.assertLessEqual(rd50, 1.15, f"rd50 too breathy: {rd50:.3f}")
            self.assertGreaterEqual(float(np.median(ctrl.shimmer_frac[vm])), 0.038)
            self.assertGreaterEqual(float(np.median(ctrl.jitter_frac[vm])), 0.0060)
            self.assertGreaterEqual(float(np.median(ctrl.bap[vm, 5])), 0.85)
            self.assertGreaterEqual(float(np.min(ctrl.rd[vm])), 0.90)

            y = _rms_norm(
                synthesize_larynx_v2(
                    ctrl,
                    glottal_mode="dynamic_coherent",
                    coherent_phasor=phasor,
                    uv_level_boost_db=0.9,
                    subharm_strength=float(subh),
                    seed=2026,
                ),
                -19.5,
            )
            self.assertTrue(check_wav_sanity(y, SR, expected_sr=SR)["ok"])
            m = compute_forensic_metrics(y, SR)
            self.assertGreaterEqual(float(m["hnr_db"]), min_hnr, f"HNR={m['hnr_db']} < {min_hnr} for {txt}")

        # Verify b/d voiced obstruents in en_stops stay voiced (>= 0.45 voiced fraction) using actual spans
        captured_plans: list[dict[str, Any]] = []
        orig_builder = en_stage10_renderer.build_en_stage10_control

        def capture_builder(plans, *args, **kwargs):
            captured_plans.extend(plans)
            return orig_builder(plans, *args, **kwargs)

        with patch.object(en_stage10_renderer, "build_en_stage10_control", side_effect=capture_builder):
            ctrl_st, _ = build_stage10_en_control(
                stops_txt, float(kb.speaker_f0_med_hz), vp_neu, zh_kb=kb, project_to_zh_speaker=True
            )
        bd_plans = [p for p in captured_plans if p.get("ipa") in ("b", "d")]
        self.assertGreaterEqual(len(bd_plans), 4)
        for p in bd_plans:
            a, b = int(p["start_fr"]), int(p["end_fr"])
            if b > a:
                vf = float(np.mean(ctrl_st.voiced[a:b]))
                self.assertGreaterEqual(vf, 0.45, f"Stop /{p['ipa']}/ devoiced: {vf:.2f}")

        # 2. Expressive/dramatic pure EN unlocks wider pitch dynamics than neutral
        plain_txt = "Finally it worked. This sounds much better."
        emo_txt = "[激动] *Finally* it worked! This sounds *so* much better!"
        vp_exc = VoiceParams.from_dict({"preset_id": "female_std", "emotion_mode": "excited", "emotion_intensity": 0.95})

        c_plain, subh_p = build_stage10_en_control(
            plain_txt, float(kb.speaker_f0_med_hz), vp_neu, zh_kb=kb, project_to_zh_speaker=True
        )
        segs = [
            s
            for s in parse_expressive_text(
                emo_txt, emotion_mode="excited", emotion_intensity=0.95, auto_breath=False
            )
            if s.text.strip()
        ]
        emo_seg = segs[0] if segs else None
        en_u = plan_en_v2(emo_seg.text if emo_seg else emo_txt, focus_spans=emo_seg.focus_spans if emo_seg else None)["units"]
        c_emo, subh_e = build_stage10_en_control(
            en_u, float(kb.speaker_f0_med_hz), vp_exc, zh_kb=kb, expressive_seg=emo_seg, project_to_zh_speaker=True
        )
        y_plain = synthesize_larynx_v2(
            c_plain, glottal_mode="dynamic_coherent", coherent_phasor=phasor, uv_level_boost_db=0.9, subharm_strength=float(subh_p), seed=2026
        )
        y_emo = synthesize_larynx_v2(
            c_emo, glottal_mode="dynamic_coherent", coherent_phasor=phasor, uv_level_boost_db=0.9, subharm_strength=float(subh_e), seed=2026
        )
        self.assertTrue(check_wav_sanity(y_plain, SR, expected_sr=SR)["ok"])
        self.assertTrue(check_wav_sanity(y_emo, SR, expected_sr=SR)["ok"])
        m_plain = compute_forensic_metrics(y_plain, SR)
        m_emo = compute_forensic_metrics(y_emo, SR)
        self.assertGreater(float(m_emo["f0_iqr_st"]), float(m_plain["f0_iqr_st"]))


if __name__ == "__main__":
    unittest.main()
