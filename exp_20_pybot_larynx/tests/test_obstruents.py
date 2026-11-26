"""Unit and integration tests for Stage 2 RuleObstruentCrispness and obstruent_hi_res.npz."""
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

from exp_20_pybot_larynx import HOP, KNOWLEDGE_PATH, N_BAP, N_BINS, N_CEP, N_FFT, SR  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge import AcousticKnowledge, build_base_control  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import (  # noqa: E402
    STAGE2_KNOWLEDGE_PATH,
    AcousticKnowledgeStage2,
    build_stage2_control,
)
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB, Arbiter, SyllableSpan, build_utterance_context  # noqa: E402
from exp_20_pybot_larynx.judge import evaluate_utterance  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env, log_env_to_cep, synthesize_larynx  # noqa: E402
from exp_20_pybot_larynx.rules.obstruents import (  # noqa: E402
    _BAND_BASIS,
    _NON_INIT_PROFILES,
    _STAGE2_SHARP_PROFILES,
    HI_RES_PATH,
    RuleObstruentCrispness,
)


def _make_single_onset_ctrl(onset: str, index: int = 1, n_on: int = 20, n_fin: int = 20, final: str = "a"):
    syls = []
    cur = 0
    if index > 0:
        syls.append(SyllableSpan(index=0, hanzi="啊", onset="", final="a", tone=1, start_fr=0, split_fr=0, end_fr=10))
        cur = 10
    syls.append(
        SyllableSpan(
            index=index,
            hanzi="测",
            onset=onset,
            final=final,
            tone=1,
            start_fr=cur,
            split_fr=cur + n_on,
            end_fr=cur + n_on + n_fin,
        )
    )
    n_fr = cur + n_on + n_fin
    ctx = build_utterance_context("test_obs", n_fr, syls, speaker_f0_med_hz=240.0)
    f0 = np.full(n_fr, 230.0, dtype=np.float64)
    voi = np.ones(n_fr, dtype=bool)
    base = LarynxControl(
        f0_hz=f0,
        voiced=voi,
        log_env=np.full((n_fr, N_BINS), -5.0, dtype=np.float64),
        bap=np.full((n_fr, N_BAP), 0.25, dtype=np.float64),
        n_samples=n_fr * HOP,
    )
    return ctx, base, cur, cur + n_on


class TestRuleObstruentCrispness(unittest.TestCase):
    def test_band_basis_partition_of_unity_and_profiles(self) -> None:
        self.assertEqual(_BAND_BASIS.shape, (5, N_BINS))
        self.assertTrue(np.all(_BAND_BASIS >= 0.0))
        self.assertTrue(np.allclose(_BAND_BASIS.sum(axis=0), 1.0, atol=1e-9))
        expected_obs = {
            "b", "p", "d", "t", "g", "k",
            "z", "c", "zh", "ch", "j", "q",
            "f", "s", "sh", "x", "h",
        }
        self.assertEqual(set(_NON_INIT_PROFILES.keys()), expected_obs)
        for o, prof in _NON_INIT_PROFILES.items():
            self.assertEqual(prof.shape, (5, 5), msg=f"Bad shape for {o}")
            self.assertTrue(np.all(np.isfinite(prof)))
        self.assertEqual(set(_STAGE2_SHARP_PROFILES.keys()), expected_obs)
        for o, prof in _STAGE2_SHARP_PROFILES.items():
            self.assertEqual(prof.shape, (5, 5), msg=f"Bad Stage 2 shape for {o}")
            self.assertTrue(np.all(np.isfinite(prof)))

    def test_hi_res_npz_schema_and_contrast(self) -> None:
        self.assertTrue(HI_RES_PATH.exists(), f"Missing {HI_RES_PATH}")
        expected_obs = {
            "b", "p", "d", "t", "g", "k",
            "z", "c", "zh", "ch", "j", "q",
            "f", "s", "sh", "x", "h",
        }
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        m_1_11 = (freqs >= 1000.0) & (freqs <= 11000.0)
        with np.load(HI_RES_PATH, allow_pickle=False) as d:
            self.assertEqual(set(str(k) for k in d["o_k"]), expected_obs)
            self.assertEqual(d["o_env"].shape, (17, 5, N_BINS))
            self.assertEqual(d["o_bap"].shape, (17, 5, N_BAP))
            self.assertGreater(len(d["ofg_k"]), 25)
            self.assertTrue(np.all(np.isfinite(d["o_env"])))
            self.assertTrue(np.all(np.isfinite(d["ofg_env"])))
            db = d["o_env"][:, 1:4, :].astype(np.float64) * NEPER_TO_DB
            p90_10 = np.median(np.percentile(db[:, :, m_1_11], 90, axis=2) - np.percentile(db[:, :, m_1_11], 10, axis=2))
            self.assertGreater(float(p90_10), 15.0)

    def test_empty_and_sonorant_return_none(self) -> None:
        rule = RuleObstruentCrispness()
        ctx0 = build_utterance_context("empty", 10, [])
        base0 = LarynxControl(
            f0_hz=np.zeros(10),
            voiced=np.zeros(10, dtype=bool),
            log_env=np.zeros((10, N_BINS)),
            bap=np.full((10, N_BAP), 0.5),
        )
        self.assertIsNone(rule.propose(ctx0, base0))

        son_syls = [
            SyllableSpan(index=0, hanzi="妈", onset="m", final="a", tone=1, start_fr=0, split_fr=6, end_fr=20),
            SyllableSpan(index=1, hanzi="来", onset="l", final="ai", tone=2, start_fr=20, split_fr=26, end_fr=40),
            SyllableSpan(index=2, hanzi="啊", onset="", final="a", tone=1, start_fr=40, split_fr=40, end_fr=55),
        ]
        ctx_son = build_utterance_context("son", 55, son_syls)
        base_son = LarynxControl(
            f0_hz=np.full(55, 220.0),
            voiced=np.ones(55, dtype=bool),
            log_env=np.zeros((55, N_BINS)),
            bap=np.full((55, N_BAP), 0.1),
        )
        self.assertIsNone(rule.propose(ctx_son, base_son))

    def test_hi_res_contrast_restoration_and_vot_taper(self) -> None:
        rule = RuleObstruentCrispness()
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        m_1_11 = (freqs >= 1000.0) & (freqs <= 11000.0)

        # Flat blurred sibilant 's' gains >9.5 dB contrast from 0 dB, and smooth s-slope reaches >15 dB
        ctx_s, base_s, a, sp = _make_single_onset_ctrl("s", index=1, n_on=20, final="a")
        prop_s = rule.propose(ctx_s, base_s)
        self.assertIsNotNone(prop_s)
        out_s = base_s.log_env + prop_s.delta_log_env
        core_fr = a + int(round(0.50 * (sp - a)))
        core_db = out_s[core_fr, m_1_11] * NEPER_TO_DB
        self.assertGreater(float(np.percentile(core_db, 90) - np.percentile(core_db, 10)), 9.5)

        smooth_s = -5.5 + 1.4 * np.clip((freqs - 3500.0) / 5000.0, 0.0, 1.0)
        base_s.log_env[a:sp] = smooth_s[None, :]
        prop_s2 = rule.propose(ctx_s, base_s)
        core_db2 = (base_s.log_env[core_fr, m_1_11] + prop_s2.delta_log_env[core_fr, m_1_11]) * NEPER_TO_DB
        self.assertGreater(float(np.percentile(core_db2, 90) - np.percentile(core_db2, 10)), 14.0)

        # Last VOT frame (u=1.0) receives zero hi-res medoid pull (only smooth coarse VOT taper)
        vot_fr = sp - 1
        vot_hi = out_s[vot_fr] - cep_to_log_env(log_env_to_cep(out_s[vot_fr : vot_fr + 1], n_cep=N_CEP), n_bins=N_BINS)[0]
        self.assertLess(float(np.max(np.abs(vot_hi))), 0.01)

    def test_sharp_base_preserves_cep64_while_adding_hi_res_detail(self) -> None:
        rule = RuleObstruentCrispness()
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        m_1_11 = (freqs >= 1000.0) & (freqs <= 11000.0)

        ctx_sh, base_sh, a, sp = _make_single_onset_ctrl("sh", index=1, n_on=20, final="uo")
        # Construct an already-sharp cep64 base (mimicking knowledge_stage2 medoid)
        sharp_profile = (
            -4.5
            + 1.8 * np.exp(-0.5 * ((freqs - 2800.0) / 350.0) ** 2)
            - 1.2 * np.exp(-0.5 * ((freqs - 4800.0) / 420.0) ** 2)
            + 1.5 * np.exp(-0.5 * ((freqs - 6800.0) / 450.0) ** 2)
        )
        sharp_cep64 = cep_to_log_env(log_env_to_cep(sharp_profile[None, :], n_cep=N_CEP), n_bins=N_BINS)[0]
        base_sh.log_env[a:sp] = sharp_cep64[None, :]
        prop_sh = rule.propose(ctx_sh, base_sh)
        self.assertIsNotNone(prop_sh)
        out_core = base_sh.log_env[a + 10] + prop_sh.delta_log_env[a + 10]
        # q >= 64 detail should be injected while broad cep[0:40] stays anchored
        hi_part = out_core - cep_to_log_env(log_env_to_cep(out_core[None, :], n_cep=N_CEP), n_bins=N_BINS)[0]
        self.assertGreater(float(np.std(hi_part[m_1_11])), 0.15)

    def test_place_and_manner_spectral_shaping(self) -> None:
        rule = RuleObstruentCrispness()
        freqs = rfftfreq(N_FFT, 1.0 / SR)
        lo_m = freqs < 400.0
        b48_m = (freqs >= 4600.0) & (freqs <= 7200.0)

        # Aspirated retroflex affricate 'ch': burst u~0.30 boosts 4.6-7.2 kHz strongly over <400 Hz
        ctx_ch, base_ch, a, sp = _make_single_onset_ctrl("ch", index=1, n_on=21)
        prop_ch = rule.propose(ctx_ch, base_ch)
        burst_fr = a + int(round(0.30 * (sp - a)))
        b48_delta = float(np.mean(prop_ch.delta_log_env[burst_fr, b48_m]))
        lo_delta = float(np.mean(prop_ch.delta_log_env[burst_fr, lo_m]))
        self.assertGreater(b48_delta, 0.50)
        self.assertGreater(b48_delta - lo_delta, 0.40)

        # Unaspirated stop 'd': closure u~0.67 dips deeply on both blurred and sharp Stage 2 bases
        ctx_d, base_d, ad, spd = _make_single_onset_ctrl("d", index=1, n_on=10)
        prop_d = rule.propose(ctx_d, base_d)
        clos_d = ad + int(round(0.60 * (spd - ad)))
        vot_d = spd - 1
        self.assertLess(float(np.mean(prop_d.delta_log_env[clos_d])), -0.50)
        self.assertGreater(float(np.mean(prop_d.delta_log_env[vot_d])), 0.10)

        sharp_d = -5.5 + 1.4 * np.exp(-0.5 * ((freqs - 2600.0) / 380.0) ** 2) - 1.1 * np.exp(-0.5 * ((freqs - 5200.0) / 420.0) ** 2)
        base_d.log_env[ad:spd] = cep_to_log_env(log_env_to_cep(sharp_d[None, :], n_cep=N_CEP), n_bins=N_BINS)[0][None, :]
        prop_d2 = rule.propose(ctx_d, base_d)
        self.assertLess(float(np.mean(prop_d2.delta_log_env[clos_d])), -0.75)

    def test_clean_voiced_unvoiced_vot_and_initial_ramp(self) -> None:
        rule = RuleObstruentCrispness()

        ctx_t, base_t, a, sp = _make_single_onset_ctrl("t", index=1, n_on=21)
        prop_t = rule.propose(ctx_t, base_t)
        u_t = ctx_t.phase_u[a:sp]
        ov_t = prop_t.override_voiced[a:sp]
        self.assertTrue(np.all(ov_t[(u_t >= 0.20) & (u_t <= 0.78)] == 0))
        self.assertTrue(np.all(ov_t[u_t > 0.80] == -1))
        self.assertTrue(np.all(ov_t[u_t < 0.18] == -1))

        ctx_d, base_d, ad, spd = _make_single_onset_ctrl("d", index=1, n_on=10)
        prop_d = rule.propose(ctx_d, base_d)
        u_d = ctx_d.phase_u[ad:spd]
        ov_d = prop_d.override_voiced[ad:spd]
        self.assertTrue(np.all(ov_d[u_d < 0.25] == -1))
        self.assertTrue(np.all(ov_d[u_d > 0.56] == -1))

        ctx_t0, base_t0, a0, sp0 = _make_single_onset_ctrl("t", index=0, n_on=20)
        prop_t0 = rule.propose(ctx_t0, base_t0)
        self.assertLess(float(np.mean(prop_t0.delta_log_env[a0])), -1.2)
        self.assertEqual(int(prop_t0.override_voiced[a0]), 0)
        self.assertEqual(int(prop_t0.override_voiced[sp0 - 1]), -1)

    def test_arbiter_and_larynx_integration_full_and_stage2(self) -> None:
        kb_path = REPO_ROOT / "exp_20_pybot_larynx" / "acoustics" / "knowledge_full.npz"
        if not kb_path.exists():
            kb_path = KNOWLEDGE_PATH
        if not kb_path.exists():
            self.skipTest(f"{kb_path} not built yet")
        kb = AcousticKnowledge.load(kb_path)
        arb = Arbiter()
        rule = RuleObstruentCrispness()

        syls = [
            SyllableSpan(index=0, hanzi="请", onset="q", final="ing", tone=3, start_fr=0, split_fr=18, end_fr=38),
            SyllableSpan(index=1, hanzi="说", onset="sh", final="uo", tone=1, start_fr=38, split_fr=54, end_fr=74),
            SyllableSpan(index=2, hanzi="大", onset="d", final="a", tone=4, start_fr=74, split_fr=81, end_fr=102),
            SyllableSpan(index=3, hanzi="家", onset="j", final="ia", tone=1, start_fr=102, split_fr=114, end_fr=134),
            SyllableSpan(index=4, hanzi="好", onset="h", final="ao", tone=3, start_fr=134, split_fr=148, end_fr=170),
        ]
        n_fr = 170
        ctx = build_utterance_context("obs_e2e", n_fr, syls, speaker_f0_med_hz=kb.speaker_f0_med_hz)
        f0 = np.full(n_fr, 230.0, dtype=np.float64)
        voi = np.ones(n_fr, dtype=bool)
        for s in syls:
            voi[s.start_fr : s.start_fr + max(1, int(0.7 * (s.split_fr - s.start_fr)))] = False
            f0[~voi] = 0.0

        base, _ = build_base_control(ctx, f0, voi, kb, mode="dynamic")
        base.n_samples = n_fr * HOP

        t0 = time.perf_counter()
        fused, trace = arb.apply(ctx, base, [rule])
        rule_ms = (time.perf_counter() - t0) * 1000.0
        self.assertIsNone(trace["modules"][0]["error"])
        self.assertGreater(trace["modules"][0]["frames_touched"], 0)
        self.assertLess(rule_ms, 50.0)

        t1 = time.perf_counter()
        y = synthesize_larynx(fused, glottal_mode="parametric_lf", seed=17)
        rtf = (time.perf_counter() - t1) / max(1e-6, len(y) / SR)
        self.assertEqual(len(y), n_fr * HOP)
        self.assertTrue(np.all(np.isfinite(y)))
        ev = evaluate_utterance(y, y, rtf=rtf)
        self.assertTrue(ev["sanity_ok"])
        self.assertLess(ev["clicks_gt8"], 0.01)

        if STAGE2_KNOWLEDGE_PATH.exists():
            kb2 = AcousticKnowledgeStage2.load(STAGE2_KNOWLEDGE_PATH)
            base2, _ = build_stage2_control(ctx, f0, voi, kb2)
            base2.n_samples = n_fr * HOP
            fused2, trace2 = arb.apply(ctx, base2, [rule])
            self.assertIsNone(trace2["modules"][0]["error"])
            y2 = synthesize_larynx(fused2, glottal_mode="parametric_lf", seed=17)
            self.assertEqual(len(y2), n_fr * HOP)
            self.assertTrue(np.all(np.isfinite(y2)))


if __name__ == "__main__":
    unittest.main()
