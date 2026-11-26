"""Unit and physical verification tests for Stage 9B English Articulation & Coarticulation Table."""
from __future__ import annotations

import math
from pathlib import Path
import sys
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
from exp_20_pybot_larynx.crossling.en_articulation import (  # noqa: E402
    EN_ARTICULATION_PATH,
    EnArticulationTable,
    get_en_articulation_table,
)
from exp_20_pybot_larynx.crossling.en_larynx_builder import (  # noqa: E402
    _build_rhythm_track_stage9b,
    build_speaker_anchored_en_control,
)
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
from exp_v6_crossling_dsp.speaker_profile import neutral_female_reference  # noqa: E402
from pybot_webui.crossling import build_utterance_control, build_zh_run_control  # noqa: E402
from pybot_webui.presets import VoiceParams  # noqa: E402

_FREQS = rfftfreq(N_FFT, 1.0 / SR)


def _extract_target_phone_frames(word: str, target_ipa: str, kb: AcousticKnowledgeStage7, vp: VoiceParams) -> tuple[np.ndarray, int, int]:
    """Return `(ctrl.log_env, start_fr, end_fr)` for the first matching phone in synthesized word."""
    ir = _safe_plan_v13(word)
    en_u = [u for u in ir["units"] if u.get("lang") == "en"]
    en_art = get_en_articulation_table()
    prof = neutral_female_reference()
    track, _meta = _build_rhythm_track_stage9b(en_u, prof, en_art, use_en_articulation=True)
    ctrl, _ = build_speaker_anchored_en_control(en_u, float(kb.speaker_f0_med_hz), vp, kb, use_en_articulation=True)

    cursor = 0
    for seg in track.segments:
        nf = max(1, int(round(seg.dur_ms / 5.0)))
        if not seg.is_pause and seg.ipa == target_ipa:
            s0 = max(0, cursor - 1)
            s1 = min(len(ctrl.f0_hz), cursor - 1 + nf)
            return ctrl.log_env, s0, s1
        cursor += nf
    raise AssertionError(f"Phone {target_ipa!r} not found in {word!r}")


@unittest.skipUnless(STAGE7_KNOWLEDGE_PATH.exists() and EN_ARTICULATION_PATH.exists(), "Stage 7 or Stage 9B artifact missing")
class TestStage9bArticulation(unittest.TestCase):
    kb: AcousticKnowledgeStage7
    arbiter: Arbiter
    rules: list
    phasor: np.ndarray
    art: EnArticulationTable

    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
        cls.arbiter = Arbiter()
        cls.rules = default_stage5_rules()
        assets = load_speaker_assets_v15()
        cls.phasor = np.asarray(assets["glottal_bank"].coherent_phasor, dtype=np.complex128)
        art = get_en_articulation_table()
        assert art is not None
        cls.art = art

    def test_artifact_size_zero_ml_and_ipa_coverage(self) -> None:
        size_bytes = EN_ARTICULATION_PATH.stat().st_size
        self.assertLessEqual(size_bytes, 1_048_576, f"Artifact exceeds 1 MB: {size_bytes}")
        for arr in (
            self.art.delta_cep,
            self.art.formants_5bin,
            self.art.voicing_prob_5bin,
            self.art.rel_level_db_5bin,
            self.art.left_coart_cep,
            self.art.right_coart_cep,
            self.art.left_coart_f123,
            self.art.right_coart_f123,
            self.art.dur_ms_by_stress,
            self.art.stress_f0_st_5bin,
            self.art.stress_gain_db,
        ):
            self.assertTrue(np.all(np.isfinite(arr)))

        for req in list(IPA_SPECS.keys()) + ["l_coda", "p_coda", "t_coda", "k_coda", "b_coda", "d_coda", "ɡ_coda"]:
            self.assertIn(req, self.art.key_to_id)

    def test_dark_l_vs_light_l_allophone(self) -> None:
        vp = VoiceParams.from_dict({"preset_id": "female_std"})
        env_light, a1, b1 = _extract_target_phone_frames("light", "l", self.kb, vp)
        env_milk, a2, b2 = _extract_target_phone_frames("milk", "l", self.kb, vp)

        sf_on = decompose_frame_source_filter(env_light[(a1 + b1) // 2], order=22)
        hz_on, _ = lsf_to_formants(sf_on.lsf, sr=SR, f_min=200.0, f_max=4500.0)
        f2_on = float(hz_on[hz_on >= 700.0][0])

        sf_coda = decompose_frame_source_filter(env_milk[(a2 + b2) // 2], order=22)
        hz_coda, _ = lsf_to_formants(sf_coda.lsf, sr=SR, f_min=200.0, f_max=4500.0)
        f2_coda = float(hz_coda[hz_coda >= 700.0][0])

        self.assertGreater(f2_on - f2_coda, 150.0, f"Expected F2(onset l)={f2_on:.0f} > F2(dark l)={f2_coda:.0f} + 150 Hz")

    def test_pre_fortis_vowel_clipping(self) -> None:
        vp = VoiceParams.from_dict({"preset_id": "female_std"})
        _, a_bat, b_bat = _extract_target_phone_frames("bat", "æ", self.kb, vp)
        _, a_bad, b_bad = _extract_target_phone_frames("bad", "æ", self.kb, vp)
        dur_bat = b_bat - a_bat
        dur_bad = b_bad - a_bad
        self.assertLessEqual(dur_bat, 0.85 * dur_bad, f"Pre-fortis clipping weak: bat={dur_bat*5}ms vs bad={dur_bad*5}ms")

    def test_english_obstruents_and_cross_word_linking(self) -> None:
        vp = VoiceParams.from_dict({"preset_id": "female_std"})
        for txt in ("think this very zoo vision", "turn it on an apple"):
            ctrl, meta = build_utterance_control(txt, "t9b_link", self.kb, self.arbiter, self.rules, vp)
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
            self.assertTrue(san["ok"], f"Sanity failed for {txt!r}: {san}")

        # Compare voiced (/v, z/) vs unvoiced (/f, s/) low-band BAP
        en_v = [u for u in _safe_plan_v13("very zoo")["units"] if u.get("lang") == "en"]
        c_v, _ = build_speaker_anchored_en_control(en_v, float(self.kb.speaker_f0_med_hz), vp, self.kb, use_en_articulation=True)
        en_f = [u for u in _safe_plan_v13("four see")["units"] if u.get("lang") == "en"]
        c_f, _ = build_speaker_anchored_en_control(en_f, float(self.kb.speaker_f0_med_hz), vp, self.kb, use_en_articulation=True)
        # First voiced obstruent frames in "very zoo" have low BAP < 0.40; unvoiced "f/s" onset frames have > 0.90
        self.assertLess(float(np.min(c_v.bap[:, 0])), 0.40)
        self.assertGreater(float(np.max(c_f.bap[:, 0])), 0.90)

    def test_same_speaker_identity_preserved_under_stage9b(self) -> None:
        vp = VoiceParams.from_dict({"preset_id": "female_std"})
        zh_ctrl, _ = build_zh_run_control(
            "今天四十三位朋友一起测试语音合成系统，效果清晰自然。",
            "t9b_zh",
            self.kb,
            self.arbiter,
            self.rules,
            vp,
        )
        en_text = "Sixty seven friends test our cross lingual speech system today clearly."
        en_units = [u for u in _safe_plan_v13(en_text)["units"] if u.get("lang") == "en"]
        en_ctrl, _ = build_speaker_anchored_en_control(
            en_units, float(self.kb.speaker_f0_med_hz), vp, self.kb, use_en_articulation=True
        )

        vm_z = zh_ctrl.voiced & (zh_ctrl.f0_hz > 40.0)
        vm_e = en_ctrl.voiced & (en_ctrl.f0_hz > 40.0)
        m_v = (_FREQS >= 80.0) & (_FREQS <= 11500.0)
        voiced_rmse = float(np.sqrt(np.mean(((np.mean(zh_ctrl.log_env[vm_z], 0) - np.mean(en_ctrl.log_env[vm_e], 0))[m_v] * NEPER_TO_DB) ** 2)))
        self.assertLessEqual(voiced_rmse, 4.5, f"Stage 9B Voiced LTAS RMSE too high: {voiced_rmse:.2f} dB")

        um_z = (~zh_ctrl.voiced) & (np.mean(zh_ctrl.log_env[:, _FREQS >= 1500.0], axis=1) > -9.5)
        um_e = (~en_ctrl.voiced) & (np.mean(en_ctrl.log_env[:, _FREQS >= 1500.0], axis=1) > -9.5)
        m_u = _FREQS >= 800.0
        uv_rmse = float(np.sqrt(np.mean(((np.mean(zh_ctrl.log_env[um_z], 0) - np.mean(en_ctrl.log_env[um_e], 0))[m_u] * NEPER_TO_DB) ** 2)))
        self.assertLessEqual(uv_rmse, 6.5, f"Stage 9B Unvoiced LTAS RMSE too high: {uv_rmse:.2f} dB")

        bap_mae = float(np.mean(np.abs(np.median(zh_ctrl.bap[vm_z], 0) - np.median(en_ctrl.bap[vm_e], 0))))
        self.assertLessEqual(bap_mae, 0.08)

        f0_diff_st = abs(12.0 * math.log2(float(np.median(en_ctrl.f0_hz[vm_e])) / float(np.median(zh_ctrl.f0_hz[vm_z]))))
        self.assertLessEqual(f0_diff_st, 1.0)

        en_iqr_st = float(
            12.0 * math.log2(float(np.percentile(en_ctrl.f0_hz[vm_e], 75)) / max(40.0, float(np.percentile(en_ctrl.f0_hz[vm_e], 25))))
        )
        self.assertGreaterEqual(en_iqr_st, 2.4)


if __name__ == "__main__":
    unittest.main()
