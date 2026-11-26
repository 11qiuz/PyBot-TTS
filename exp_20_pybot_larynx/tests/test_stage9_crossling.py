"""Unit and physical verification tests for Stage 9 Speaker-Anchored Cross-Lingual Engine."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import unittest

import numpy as np
from scipy.fft import rfftfreq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import N_FFT, SR  # noqa: E402
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
from exp_20_pybot_larynx.crossling import (  # noqa: E402
    SpeakerDonorBank,
    build_speaker_anchored_en_control,
    get_en_articulation_table,
    get_speaker_donor_bank,
)
from exp_20_pybot_larynx.crossling.en_articulation import VOWEL_IPAS  # noqa: E402
from exp_20_pybot_larynx.crossling.en_larynx_builder import _build_rhythm_track_stage9b  # noqa: E402
from exp_v6_crossling_dsp.speaker_profile import neutral_female_reference  # noqa: E402
from exp_20_pybot_larynx.frontend.prosody_rules import _safe_plan_v13  # noqa: E402
from exp_20_pybot_larynx.larynx.spectral_decomp import (  # noqa: E402
    decompose_frame_source_filter,
    lsf_to_formants,
)
from exp_20_pybot_larynx.larynx.synth_v2 import synthesize_larynx_v2  # noqa: E402
from exp_20_pybot_larynx.rules import default_stage5_rules  # noqa: E402
from exp_v15_bznsyp_tts.index_bznsyp_v15 import load_speaker_assets_v15  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity  # noqa: E402
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS  # noqa: E402
from pybot_webui.crossling import (  # noqa: E402
    build_en_run_control,
    build_utterance_control,
    build_zh_run_control,
)
from pybot_webui.presets import VoiceParams  # noqa: E402

_FREQS = rfftfreq(N_FFT, 1.0 / SR)


def _measure_word_nucleus_formants(word: str, kb: AcousticKnowledgeStage7, vp: VoiceParams) -> tuple[float, float, float]:
    ir = _safe_plan_v13(word)
    en_u = [u for u in ir["units"] if u.get("lang") == "en"]
    en_art = get_en_articulation_table()
    tr, _ = _build_rhythm_track_stage9b(en_u, neutral_female_reference(), en_art, use_en_articulation=(en_art is not None))
    ctrl, _ = build_en_run_control(en_u, float(kb.speaker_f0_med_hz), vp, kb=kb, use_stage10_en=False)
    mid = -1
    cur = 0
    for seg in tr.segments:
        nf = max(1, int(round(seg.dur_ms / 5.0)))
        if not seg.is_pause and seg.ipa in VOWEL_IPAS:
            mid = max(0, min(len(ctrl.f0_hz) - 1, cur - 1 + nf // 2))
            if seg.stress == 1:
                break
        cur += nf
    if mid < 0:
        vm = np.where(ctrl.voiced & (ctrl.f0_hz > 40.0))[0]
        mid = int(vm[len(vm) // 2])
    sf = decompose_frame_source_filter(ctrl.log_env[mid], order=22)
    hz, _ = lsf_to_formants(sf.lsf, sr=SR, f_min=200.0, f_max=4500.0)
    self_f = hz[(hz >= 230.0) & (hz <= 4250.0)]
    return float(self_f[0]), float(self_f[1]), float(self_f[2])


@unittest.skipUnless(STAGE7_KNOWLEDGE_PATH.exists(), "Stage 7 knowledge artifact missing")
class TestStage9Crossling(unittest.TestCase):
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

    def test_donor_bank_invariants_and_full_ipa_coverage(self) -> None:
        t0 = time.perf_counter()
        bank = SpeakerDonorBank(self.kb)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 800.0)
        self.assertIs(get_speaker_donor_bank(self.kb), bank.__class__ and get_speaker_donor_bank(self.kb))
        self.assertTrue(0.85 <= bank.scale_f1 <= 1.24)
        self.assertTrue(0.86 <= bank.scale_f2 <= 1.22)
        self.assertTrue(0.90 <= bank.scale_f3 <= 1.26)
        self.assertTrue(np.all(np.isfinite(bank.voiced_ltas_env)))
        self.assertTrue(np.all(np.isfinite(bank.unvoiced_ltas_env)))

        for ipa, spec in IPA_SPECS.items():
            if spec.manner in ("vowel", "diphthong", "glide", "nasal", "liquid"):
                don = bank.get_sonorant_donor(ipa)
                self.assertEqual(don.env.shape, (513,))
                self.assertEqual(don.formants.shape, (3,))
                self.assertTrue(don.formants[0] < don.formants[1] < don.formants[2])
            else:
                obs = bank.get_obstruent_donor(ipa)
                self.assertEqual(obs.env.shape, (513,))
                self.assertTrue(np.all(np.isfinite(obs.env)))

    def test_english_vowel_phonetic_distinctiveness(self) -> None:
        vp = VoiceParams.from_dict({"preset_id": "female_std"})
        f_beat = _measure_word_nucleus_formants("beat", self.kb, vp)     # /i/
        f_boot = _measure_word_nucleus_formants("boot", self.kb, vp)     # /u/
        f_father = _measure_word_nucleus_formants("father", self.kb, vp) # /ɑ/
        f_bird = _measure_word_nucleus_formants("bird", self.kb, vp)     # /ɝ/
        f_bad = _measure_word_nucleus_formants("bad", self.kb, vp)       # /æ/

        self.assertGreater(f_beat[1] - f_boot[1], 800.0)     # F2(/i/) > F2(/u/) + 800 Hz
        self.assertGreater(f_father[0] - f_beat[0], 250.0)   # F1(/ɑ/) > F1(/i/) + 250 Hz
        self.assertGreater(f_beat[2] - f_bird[2], 400.0)     # F3(/ɝ/) < F3(/i/) - 400 Hz
        self.assertGreater(f_bad[1] - f_father[1], 200.0)    # F2(/æ/) > F2(/ɑ/) + 200 Hz

    def test_same_speaker_physical_alignment_zh_vs_en(self) -> None:
        vp = VoiceParams.from_dict({"preset_id": "female_std"})
        zh_ctrl, _ = build_zh_run_control(
            "今天四十三位朋友一起测试语音合成系统，效果清晰自然。",
            "t9_zh",
            self.kb,
            self.arbiter,
            self.rules,
            vp,
        )
        en_text = "Sixty seven friends test our cross lingual speech system today clearly."
        en_units = [u for u in _safe_plan_v13(en_text)["units"] if u.get("lang") == "en"]
        en_ctrl, _ = build_speaker_anchored_en_control(en_units, float(self.kb.speaker_f0_med_hz), vp, self.kb)

        vm_zh = zh_ctrl.voiced & (zh_ctrl.f0_hz > 40.0)
        vm_en = en_ctrl.voiced & (en_ctrl.f0_hz > 40.0)

        m_v = (_FREQS >= 80.0) & (_FREQS <= 11500.0)
        zh_v_env = np.mean(zh_ctrl.log_env[vm_zh], axis=0)
        en_v_env = np.mean(en_ctrl.log_env[vm_en], axis=0)
        voiced_rmse_db = float(np.sqrt(np.mean(((zh_v_env - en_v_env)[m_v] * NEPER_TO_DB) ** 2)))
        self.assertLessEqual(voiced_rmse_db, 4.5, f"Voiced LTAS RMSE too high: {voiced_rmse_db:.2f} dB")

        um_zh = (~zh_ctrl.voiced) & (np.mean(zh_ctrl.log_env[:, _FREQS >= 1500.0], axis=1) > -9.5)
        um_en = (~en_ctrl.voiced) & (np.mean(en_ctrl.log_env[:, _FREQS >= 1500.0], axis=1) > -9.5)
        m_u = _FREQS >= 800.0
        zh_u_env = np.mean(zh_ctrl.log_env[um_zh], axis=0)
        en_u_env = np.mean(en_ctrl.log_env[um_en], axis=0)
        uv_rmse_db = float(np.sqrt(np.mean(((zh_u_env - en_u_env)[m_u] * NEPER_TO_DB) ** 2)))
        self.assertLessEqual(uv_rmse_db, 6.5, f"Unvoiced LTAS RMSE too high: {uv_rmse_db:.2f} dB")

        bap_mae = float(np.mean(np.abs(np.median(zh_ctrl.bap[vm_zh], axis=0) - np.median(en_ctrl.bap[vm_en], axis=0))))
        self.assertLessEqual(bap_mae, 0.08, f"BAP6 MAE too high: {bap_mae:.3f}")

        f0_diff_st = abs(12.0 * math.log2(float(np.median(en_ctrl.f0_hz[vm_en])) / float(np.median(zh_ctrl.f0_hz[vm_zh]))))
        self.assertLessEqual(f0_diff_st, 1.0, f"F0 median jump too high: {f0_diff_st:.2f} st")

        en_iqr_st = float(
            12.0
            * math.log2(
                float(np.percentile(en_ctrl.f0_hz[vm_en], 75)) / max(40.0, float(np.percentile(en_ctrl.f0_hz[vm_en], 25)))
            )
        )
        self.assertGreaterEqual(en_iqr_st, 2.4, f"EN F0 IQR too flat: {en_iqr_st:.2f} st")

    def test_mixed_code_switching_emotions_and_voice_clone(self) -> None:
        # 1. Mixed ZH+EN neutral + expressive emotions
        for emo in ("auto", "excited", "sad", "whisper"):
            vp = VoiceParams.from_dict({"preset_id": "female_std", "emotion_mode": emo, "emotion_intensity": 0.9})
            ctrl, meta = build_utterance_control(
                "今天用 PyBot 测试 cross lingual speech，效果非常自然！",
                f"t9_{emo}",
                self.kb,
                self.arbiter,
                self.rules,
                vp,
            )
            self.assertGreater(meta["n_zh"], 0)
            self.assertGreater(meta["n_en"], 0)
            y = synthesize_larynx_v2(
                ctrl,
                glottal_mode="dynamic_coherent",
                coherent_phasor=self.phasor,
                uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7,
                subharm_strength=float(meta.get("subharm_strength", 0.0)),
                seed=vp.seed,
            )
            san = check_wav_sanity(y, SR, expected_sr=SR)
            self.assertTrue(san["ok"], f"Sanity failed on {emo}: {san}")

        # 2. Voice Clone Cross-Lingual Male Identity Consistency (SSB0073)
        sample_path = REPO_ROOT / "legacy" / "exp_v8_data" / "corpus" / "zh" / "SSB0073" / "SSB00730001.wav"
        if sample_path.exists():
            wav_24k, _ = decode_wav_bytes_to_24k_mono(sample_path.read_bytes())
            tgt_prof = extract_clone_profile([wav_24k], name="SSB0073_test")
            base_prof = load_base_speaker_profile()
            vp = VoiceParams.from_dict({"preset_id": "female_std"})

            c_zh_base, _ = build_zh_run_control("欢迎使用跨语种语音合成。", "c_zh", self.kb, self.arbiter, self.rules, vp)
            en_u = [u for u in _safe_plan_v13("Welcome to cross lingual voice cloning.")["units"] if u.get("lang") == "en"]
            c_en_base, _ = build_speaker_anchored_en_control(en_u, float(self.kb.speaker_f0_med_hz), vp, self.kb)

            c_zh, _, _ = apply_clone_to_control(c_zh_base, base_prof, tgt_prof, strength=0.90, clone_f0=True, base_coherent_phasor=self.phasor)
            c_en, _, _ = apply_clone_to_control(c_en_base, base_prof, tgt_prof, strength=0.90, clone_f0=True, base_coherent_phasor=self.phasor)

            vm_z = c_zh.voiced & (c_zh.f0_hz > 40.0)
            vm_e = c_en.voiced & (c_en.f0_hz > 40.0)
            f0_z = float(np.median(c_zh.f0_hz[vm_z]))
            f0_e = float(np.median(c_en.f0_hz[vm_e]))
            self.assertLess(f0_z, 155.0)
            self.assertLess(f0_e, 155.0)
            self.assertLessEqual(abs(12.0 * math.log2(f0_e / f0_z)), 1.2)

            m_v = (_FREQS >= 80.0) & (_FREQS <= 11500.0)
            cloned_rmse_db = float(
                np.sqrt(np.mean(((np.mean(c_zh.log_env[vm_z], 0) - np.mean(c_en.log_env[vm_e], 0))[m_v] * NEPER_TO_DB) ** 2))
            )
            self.assertLessEqual(cloned_rmse_db, 5.0, f"Cloned ZH/EN LTAS RMSE too high: {cloned_rmse_db:.2f} dB")


if __name__ == "__main__":
    unittest.main()
