"""Unit and contract tests for Stage 2 unblurred medoid knowledge base."""
from __future__ import annotations

from pathlib import Path
import sys
import time
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import HOP, N_BAP, N_BINS, N_CEP, SR  # noqa: E402
from exp_20_pybot_larynx.acoustics.build_stage2_knowledge import _select_medoids_kmeans  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge import get_splits  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import (  # noqa: E402
    N_FINAL_BINS_S2,
    N_ONSET_BINS_S2,
    STAGE2_KNOWLEDGE_PATH,
    AcousticKnowledgeStage2,
    MedoidGroup,
    build_stage2_control,
    encode_next_place,
    encode_prev_fg,
)
from exp_20_pybot_larynx.arbiter import SyllableSpan, build_utterance_context  # noqa: E402
from exp_20_pybot_larynx.judge import evaluate_utterance  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import log_env_to_cep, synthesize_larynx  # noqa: E402


class TestKnowledgeStage2(unittest.TestCase):
    def test_zero_leakage_splits(self) -> None:
        tr, dv, he = get_splits()
        self.assertEqual(len(dv), 60)
        self.assertEqual(len(he), 20)
        self.assertGreater(len(tr), 9000)
        self.assertEqual(len(set(tr) & set(dv)), 0)
        self.assertEqual(len(set(tr) & set(he)), 0)
        self.assertEqual(len(set(dv) & set(he)), 0)

    def test_medoid_clustering_returns_exact_tokens(self) -> None:
        rng = np.random.default_rng(7)
        c1 = rng.normal(loc=2.0, scale=0.1, size=(20, 40))
        c2 = rng.normal(loc=-2.0, scale=0.1, size=(20, 40))
        x = np.vstack([c1, c2])
        meds, sizes = _select_medoids_kmeans(x, k_max=4, seed=2026)
        self.assertGreaterEqual(len(meds), 2)
        self.assertLessEqual(len(meds), 4)
        self.assertEqual(sum(sizes), len(x))
        self.assertEqual(len(set(meds)), len(meds))
        for m in meds:
            self.assertTrue(0 <= m < len(x))

    def test_viterbi_selects_context_and_edge_continuous_medoid(self) -> None:
        # Build synthetic MedoidGroup with 2 distinct medoids:
        # medoid 0: mismatched context & sharp edge jump
        # medoid 1: matching context & smooth edge from prev syllable
        cep_prev = np.zeros((N_FINAL_BINS_S2, N_CEP), dtype=np.float64)
        cep_prev[-1, 1:10] = 1.5

        cep_cand = np.zeros((2, N_FINAL_BINS_S2, N_CEP), dtype=np.float64)
        cep_cand[0, :, 1:10] = -2.0
        cep_cand[1, :, 1:10] = 1.45

        bap_cand = np.full((2, N_FINAL_BINS_S2, N_BAP), 0.1, dtype=np.float64)
        vp_cand = np.ones((2, N_FINAL_BINS_S2), dtype=np.float64)

        grp0 = MedoidGroup(
            cep=cep_prev[None, :, :],
            bap=np.full((1, N_FINAL_BINS_S2, N_BAP), 0.1),
            vp=np.ones((1, N_FINAL_BINS_S2)),
            ctx_prev=np.array([encode_prev_fg("SIL")], dtype=np.int8),
            ctx_next=np.array([encode_next_place("zero")], dtype=np.int8),
            ctx_pos=np.array([0], dtype=np.int8),
            ctx_dur_ms=np.array([150.0]),
            ctx_f0_st=np.array([0.0]),
            cluster_sz=np.array([10], dtype=np.int32),
            total_tokens=10,
        )
        grp1 = MedoidGroup(
            cep=cep_cand,
            bap=bap_cand,
            vp=vp_cand,
            ctx_prev=np.array([encode_prev_fg("u_round"), encode_prev_fg("a_open_central")], dtype=np.int8),
            ctx_next=np.array([encode_next_place("velar"), encode_next_place("SIL")], dtype=np.int8),
            ctx_pos=np.array([1, 2], dtype=np.int8),
            ctx_dur_ms=np.array([80.0, 150.0]),
            ctx_f0_st=np.array([4.0, 0.0]),
            cluster_sz=np.array([15, 15], dtype=np.int32),
            total_tokens=30,
        )

        kb = AcousticKnowledgeStage2(
            on_grp={},
            on_only={},
            on_global=(np.zeros((N_ONSET_BINS_S2, N_CEP)), np.full((N_ONSET_BINS_S2, N_BAP), 0.5), np.ones(N_ONSET_BINS_S2)),
            fin_oft={"|a|1": grp0, "|e|1": grp1},
            fin_pft={},
            fin_ft={},
            fin_f={},
            fin_global=(np.zeros((N_FINAL_BINS_S2, N_CEP)), np.full((N_FINAL_BINS_S2, N_BAP), 0.1), np.ones(N_FINAL_BINS_S2)),
            onset_frac={"b": 0.2},
            onset_ms={"b": 30.0},
            sil_cep64=np.zeros(N_CEP),
            sil_bap6=np.full(N_BAP, 0.95),
            speaker_f0_med_hz=247.0,
            ref_log_amp=-4.2,
            n_train_utts=100,
        )
        spans = [
            SyllableSpan(0, "啊", "", "a", 1, 0, 0, 30),
            SyllableSpan(1, "鹅", "", "e", 1, 30, 30, 60),
        ]
        ctx = build_utterance_context("syn", 60, spans, speaker_f0_med_hz=247.0)
        f0 = np.full(60, 247.0)
        voi = np.ones(60, dtype=bool)
        ctrl, info = build_stage2_control(ctx, f0, voi, kb)
        self.assertEqual(ctrl.log_env.shape, (60, N_BINS))
        self.assertEqual(info["chosen_fin_medoids"], [0, 1])

    def test_loaded_stage2_kb_contract_and_synth(self) -> None:
        if not STAGE2_KNOWLEDGE_PATH.exists():
            self.skipTest(f"{STAGE2_KNOWLEDGE_PATH} not built yet")
        kb2 = AcousticKnowledgeStage2.load(STAGE2_KNOWLEDGE_PATH)
        self.assertGreater(kb2.n_train_utts, 9000)
        self.assertGreater(len(kb2.fin_oft), 1400)
        self.assertIn("b", kb2.onset_ms)
        self.assertIn("b", kb2.onset_frac)

        # Verify stored medoid shapes (5-bin onset, 9-bin final)
        any_on = next(iter(kb2.on_grp.values()))
        self.assertEqual(any_on.cep.shape[1:], (N_ONSET_BINS_S2, N_CEP))
        self.assertLessEqual(any_on.cep.shape[0], 4)
        any_fin = next(iter(kb2.fin_oft.values()))
        self.assertEqual(any_fin.cep.shape[1:], (N_FINAL_BINS_S2, N_CEP))
        self.assertLessEqual(any_fin.cep.shape[0], 4)

        spans = [
            SyllableSpan(0, "你", "n", "i", 3, 0, 8, 32),
            SyllableSpan(1, "好", "h", "ao", 3, 32, 46, 76),
        ]
        ctx = build_utterance_context("smoke_s2", 76, spans, speaker_f0_med_hz=kb2.speaker_f0_med_hz)
        f0 = np.full(76, 230.0)
        voi = np.ones(76, dtype=bool)
        voi[32:44] = False
        f0[32:44] = 0.0

        t0 = time.perf_counter()
        ctrl, info = build_stage2_control(ctx, f0, voi, kb2)
        y = synthesize_larynx(ctrl, glottal_mode="parametric_lf", seed=7)
        dt = time.perf_counter() - t0
        rtf = dt / max(1e-6, len(y) / SR)

        self.assertEqual(len(y), 76 * HOP)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertLess(rtf, 0.08)
        ev = evaluate_utterance(y, y, rtf=rtf)
        self.assertTrue(ev["sanity_ok"])
        self.assertEqual(info["hits"]["fallback"], 0)
        cep = log_env_to_cep(ctrl.log_env, n_cep=N_CEP)
        self.assertGreater(float(np.std(cep[:, 12:40])), 0.05)

        # Verify coarse context anchors exist and high-quefrency detail (>=26) stays pure medoid
        c_anchor = kb2.lookup_final_coarse("h", "ao", 3, encode_prev_fg("i_front"), encode_next_place("SIL"), 2)
        self.assertIsNotNone(c_anchor)
        self.assertEqual(c_anchor.shape[1], 24)
        ctrl_pure, _ = build_stage2_control(
            ctx, f0, voi, kb2, coarse_cutoff=0, clean_bap_weight=0.0, smooth_inner_cep=False, contrast_bell_peak=0.0
        )
        cep_pure = log_env_to_cep(ctrl_pure.log_env, n_cep=N_CEP)
        self.assertTrue(np.allclose(cep[46:76, 26:], cep_pure[46:76, 26:], atol=1e-4))
        ctrl_nobell, _ = build_stage2_control(ctx, f0, voi, kb2, contrast_bell_peak=0.0)
        cep_nobell = log_env_to_cep(ctrl_nobell.log_env, n_cep=N_CEP)
        self.assertGreater(float(np.std(cep[46:76, 12:24])), float(np.std(cep_nobell[46:76, 12:24])))
        # Verify voiced BAP bands 1..5 calibrated cleanly below raw v9 while unvoiced frames remain turbulent
        self.assertLess(float(np.median(ctrl.bap[voi, 2])), 0.65)
        self.assertLess(float(np.median(ctrl.bap[voi, 2])), float(np.median(ctrl_pure.bap[voi, 2])) - 0.10)
        self.assertGreater(float(np.median(ctrl.bap[~voi, 2])), 0.80)


if __name__ == "__main__":
    unittest.main()
