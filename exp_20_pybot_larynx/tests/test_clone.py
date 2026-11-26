"""Unit tests for Stage 6 zero-ML reference sample imitation (`exp_20_pybot_larynx.clone`)."""
from __future__ import annotations

import io
from pathlib import Path
import time
import unittest
import wave

import numpy as np
from scipy.io import wavfile

from exp_20_pybot_larynx import HOP, N_BINS, SR
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import AcousticKnowledgeStage5, build_stage5_control
from exp_20_pybot_larynx.arbiter import Arbiter, SyllableSpan, build_utterance_context
from exp_20_pybot_larynx.clone import (
    BASE_SPEAKER_PROFILE_PATH,
    VoiceCloneProfile,
    apply_clone_to_control,
    build_base_speaker_profile,
    decode_wav_bytes_to_24k_mono,
    extract_clone_profile_from_wavs,
    load_base_speaker_profile,
)
from exp_20_pybot_larynx.frontend.prosody_stage5 import predict_prosody_stage5
from exp_20_pybot_larynx.larynx.synth import LarynxControl, synthesize_larynx
from exp_20_pybot_larynx.rules import default_stage5_rules
from exp_v5_eval.metrics import check_wav_sanity, compute_f0_hnr_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SSB0073_WAV1 = REPO_ROOT / "legacy" / "exp_v8_data" / "corpus" / "zh" / "SSB0073" / "SSB00730001.wav"


def _make_pcm16_wav_bytes(x: np.ndarray, sr: int, channels: int = 1) -> bytes:
    arr = np.asarray(x, dtype=np.float64)
    if channels > 1:
        arr = np.repeat(arr[:, None], channels, axis=1)
    pcm = np.clip(np.round(arr * 32767.0), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class TestVoiceCloneCore(unittest.TestCase):
    def test_audio_io_resample_and_guards(self) -> None:
        sr_in = 16000
        t = np.arange(int(1.8 * sr_in), dtype=np.float64) / float(sr_in)
        sig = 0.25 * np.sin(2.0 * np.pi * 160.0 * t) + 0.12 * np.sin(2.0 * np.pi * 320.0 * t)
        # Pad 250 ms silence at start and end
        pad = np.zeros(int(0.25 * sr_in), dtype=np.float64)
        padded = np.concatenate([pad, sig, pad])

        wav16_stereo = _make_pcm16_wav_bytes(padded, sr=sr_in, channels=2)
        y24, info = decode_wav_bytes_to_24k_mono(wav16_stereo)
        self.assertEqual(info["orig_sr"], 16000)
        self.assertEqual(info["orig_channels"], 2)
        self.assertTrue(info["was_resampled"])
        self.assertGreater(len(y24), int(1.4 * SR))
        self.assertLess(len(y24), int(2.2 * SR))
        self.assertTrue(np.all(np.isfinite(y24)))

        # IEEE float32 24 kHz WAV support
        t24 = np.arange(int(1.8 * SR), dtype=np.float64) / float(SR)
        sig24 = 0.25 * np.sin(2.0 * np.pi * 160.0 * t24) + 0.12 * np.sin(2.0 * np.pi * 320.0 * t24)
        buf = io.BytesIO()
        wavfile.write(buf, SR, sig24.astype(np.float32))
        y_f32, info_f = decode_wav_bytes_to_24k_mono(buf.getvalue())
        self.assertFalse(info_f["was_resampled"])
        self.assertGreater(len(y_f32), int(1.5 * SR))

        # Too-short and near-silent guards
        short_bytes = _make_pcm16_wav_bytes(sig[: int(0.3 * sr_in)], sr=sr_in)
        with self.assertRaises(ValueError):
            decode_wav_bytes_to_24k_mono(short_bytes)
        silent_bytes = _make_pcm16_wav_bytes(np.zeros(24000, dtype=np.float64), sr=24000)
        with self.assertRaises(ValueError):
            decode_wav_bytes_to_24k_mono(silent_bytes)

    @unittest.skipUnless(SSB0073_WAV1.exists(), "SSB0073 sample WAV missing")
    def test_profile_extract_json_roundtrip_and_size(self) -> None:
        t0 = time.perf_counter()
        prof = extract_clone_profile_from_wavs([SSB0073_WAV1], name="SSB0073_test", profile_id="ssb73_t")
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 8.0, f"Profile extraction took too long: {elapsed:.2f}s")

        raw_json = prof.to_json()
        self.assertLess(prof.serialized_size_bytes(), 8192)
        restored = VoiceCloneProfile.from_json(raw_json)
        self.assertEqual(restored.profile_id, "ssb73_t")
        self.assertEqual(restored.name, "SSB0073_test")
        self.assertAlmostEqual(restored.f0_median_hz, prof.f0_median_hz, places=2)
        self.assertLess(restored.f0_median_hz, 160.0)

        ph = restored.reconstruct_coherent_phasor()
        self.assertEqual(ph.shape, (N_BINS,))
        np.testing.assert_allclose(np.abs(ph), np.ones(N_BINS), atol=1e-6)

    def test_self_clone_identity(self) -> None:
        base = load_base_speaker_profile()
        n_fr = 80
        bins = np.linspace(0.0, 1.0, N_BINS)
        f0 = np.full(n_fr, base.f0_median_hz, dtype=np.float64)
        voi = np.ones(n_fr, dtype=bool)
        voi[:6] = False
        f0[:6] = 0.0
        log_env = np.repeat((-4.0 - 2.5 * bins)[None, :], n_fr, axis=0)
        bap = np.full((n_fr, 6), 0.20, dtype=np.float64)
        ctrl = LarynxControl(f0_hz=f0, voiced=voi, log_env=log_env, bap=bap, n_samples=n_fr * HOP)

        # strength=0.0 must be exact copy
        c0, _, info0 = apply_clone_to_control(ctrl, base, base, strength=0.0)
        np.testing.assert_array_equal(c0.f0_hz, ctrl.f0_hz)
        np.testing.assert_array_equal(c0.log_env, ctrl.log_env)
        self.assertEqual(info0["alpha"], 1.0)

        # self-clone at strength=1.0 must preserve F0 & envelope
        c1, _, info1 = apply_clone_to_control(ctrl, base, base, strength=1.0)
        self.assertEqual(info1["vtln_info"]["alpha"], 1.0)
        v_mask = voi & (f0 > 40.0)
        f0_diff_st = float(np.max(np.abs(12.0 * np.log2(c1.f0_hz[v_mask] / f0[v_mask]))))
        env_diff = float(np.max(np.abs(c1.log_env - ctrl.log_env)))
        self.assertLess(f0_diff_st, 0.1)
        self.assertLess(env_diff, 0.05)

    @unittest.skipUnless(SSB0073_WAV1.exists(), "SSB0073 sample WAV missing")
    def test_cross_gender_male_clone_synthesis(self) -> None:
        base = load_base_speaker_profile()
        male_prof = extract_clone_profile_from_wavs([SSB0073_WAV1], name="SSB0073_M", profile_id="ssb73_m")
        kb5 = AcousticKnowledgeStage5.load()
        arbiter = Arbiter()
        rules5 = default_stage5_rules()

        pro = predict_prosody_stage5("今天测试零权重样本模仿男声。", kb5)
        units = pro["units"]
        cur = 0
        spans: list[SyllableSpan] = []
        f0_l, voi_l = [], []
        for idx, u in enumerate(units):
            n_syl = max(4, int(round(float(u["duration_ms"]) / 5.0)))
            n_on = max(1, min(n_syl - 2, int(round(float(u.get("onset_ms", 45.0)) / 5.0)))) if u.get("onset") else 0
            spans.append(SyllableSpan(idx, str(u.get("hanzi", "")), str(u.get("onset", "")), str(u.get("final", "")), int(u.get("tone", 1)), cur, cur + n_on, cur + n_syl, 0.0, 0))
            f0_s = np.full(n_syl, 240.0, dtype=np.float64)
            v_s = np.ones(n_syl, dtype=bool)
            if n_on > 0:
                f0_s[:n_on] = 0.0
                v_s[:n_on] = False
            f0_l.append(f0_s)
            voi_l.append(v_s)
            cur += n_syl

        ctx = build_utterance_context("test_male_clone", cur, spans, speaker_f0_med_hz=kb5.speaker_f0_med_hz)
        c5, _ = build_stage5_control(ctx, np.concatenate(f0_l), np.concatenate(voi_l), kb5)
        c5, _ = arbiter.apply(ctx, c5, rules5)

        cloned, eff_ph, info = apply_clone_to_control(c5, base, male_prof, strength=0.90)
        self.assertLess(info["alpha_eff"], 0.96)
        wav = synthesize_larynx(cloned, glottal_mode="speaker_coherent", coherent_phasor=eff_ph, seed=123)
        san = check_wav_sanity(wav, SR, expected_sr=SR)
        self.assertTrue(san["ok"], san)

        fh = compute_f0_hnr_metrics(wav, sr=SR)
        self.assertIsNotNone(fh)
        assert fh is not None and fh["f0_median_hz"] is not None
        self.assertLess(float(fh["f0_median_hz"]), 155.0)


if __name__ == "__main__":
    unittest.main()
