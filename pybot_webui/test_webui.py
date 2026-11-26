"""Unit and integration tests for pybot_webui (splitter, presets, crossling, HTTP API)."""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
import wave

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pybot_webui import SR  # noqa: E402
from pybot_webui.engine import LarynxStudioEngine  # noqa: E402
from pybot_webui.presets import VOICE_PRESETS, VoiceParams  # noqa: E402
from pybot_webui.server import create_server  # noqa: E402
from pybot_webui.splitter import detect_lang_tag, normalize_text_for_larynx, split_text  # noqa: E402


def _active_rms_db(samples: np.ndarray) -> float:
    arr = np.asarray(samples, dtype=np.float64).ravel()
    pk = float(np.max(np.abs(arr))) if arr.size else 0.0
    if pk <= 1e-9:
        return -120.0
    act = arr[np.abs(arr) > 0.02 * pk]
    rms = float(np.sqrt(np.mean(act * act))) if act.size else float(np.sqrt(np.mean(arr * arr)))
    return 20.0 * math.log10(max(rms, 1e-12))


def _check_wav_bytes(wav_bytes: bytes) -> tuple[int, int, int]:
    assert wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE"
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    assert ch == 1
    assert sw == 2
    assert sr == SR
    assert len(raw) == nframes * 2
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    assert pcm.size > 100
    assert np.all(np.isfinite(pcm))
    assert float(np.max(np.abs(pcm))) <= 0.99
    assert float(np.sqrt(np.mean(pcm * pcm))) > 1e-4
    return ch, sw, sr


class TestSplitterAndNormalizer(unittest.TestCase):
    def test_sentence_line_none_modes(self) -> None:
        text = "第一句。“第二句！”第三句？\n第四句，有点长，在这里再切一刀吧；收尾。\n\n"
        sents = split_text(text, mode="sentence", max_chars=18)
        texts = [s["text"] for s in sents]
        self.assertEqual(texts[0], "第一句。")
        self.assertEqual(texts[1], "“第二句！”")
        self.assertEqual(texts[2], "第三句？")
        self.assertGreaterEqual(len(texts), 5)

        lines = split_text("  第一行 \n\n 第二行  \n", mode="line")
        self.assertEqual([x["text"] for x in lines], ["第一行", "第二行"])

        single = split_text("Hello world.\nNext line!", mode="none")
        self.assertEqual(len(single), 1)
        self.assertEqual(single[0]["lang_tag"], "en")

    def test_english_decimals_not_split_as_sentence(self) -> None:
        sents = split_text("The value is 3.14 today. Next sentence!", mode="sentence", max_chars=80)
        self.assertEqual([s["text"] for s in sents], ["The value is 3.14 today.", "Next sentence!"])

    def test_chinese_context_normalization(self) -> None:
        out = normalize_text_for_larynx("2026年增长12.5%，价格¥128，第3步是3.14倍 PyBot 测试。")
        self.assertIn("二零二六年", out)
        self.assertIn("百分之十二点五", out)
        self.assertIn("一百二十八元", out)
        self.assertIn("第三步", out)
        self.assertIn("三点一四倍", out)
        self.assertIn("PyBot", out)

        pure_en = "Hello world, 3.14% v4!"
        self.assertEqual(normalize_text_for_larynx(pure_en, allow_english=True), pure_en)

    def test_expressive_tags_preserved_across_split_and_normalize(self) -> None:
        zh_text = "[吸气] *终于*成功了！[停顿0.5s]"
        en_text = "[breath] *Finally* done! [pause 0.5s]"
        self.assertEqual(detect_lang_tag(zh_text), "zh")
        self.assertEqual(detect_lang_tag(en_text), "en")
        segs = split_text(f"{zh_text}\n{en_text}", mode="line", max_chars=80)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["lang_tag"], "zh")
        self.assertEqual(segs[1]["lang_tag"], "en")
        zh_norm = normalize_text_for_larynx(zh_text)
        en_norm = normalize_text_for_larynx(en_text, allow_english=True)
        self.assertIn("[吸气]", zh_norm)
        self.assertIn("[停顿0.5s]", zh_norm)
        self.assertIn("*终于*", zh_norm)
        self.assertIn("[pause 0.5s]", en_norm)


class TestLarynxEnginePresetsAndCrossling(unittest.TestCase):
    engine: LarynxStudioEngine

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = LarynxStudioEngine()
        cls.engine.ensure_ready()

    def test_all_nine_presets_and_pitch_order(self) -> None:
        f0_by_preset: dict[str, float] = {}
        for p in VOICE_PRESETS:
            vp = VoiceParams.from_dict(p["params"])
            rec = self.engine.synthesize_segment("今天天气真不错，适合出门散步。", vp)
            _check_wav_bytes(rec.wav_bytes)
            self.assertLess(rec.rtf, 0.20, f"Preset {p['id']} RTF too slow: {rec.rtf}")
            self.assertGreater(rec.duration_s, 0.8)
            f0_by_preset[p["id"]] = rec.f0_median_hz

        self.assertGreater(f0_by_preset["child"], f0_by_preset["girl_bright"])
        self.assertGreater(f0_by_preset["girl_bright"], f0_by_preset["female_std"])
        self.assertGreater(f0_by_preset["female_std"], f0_by_preset["male_young"])
        self.assertGreater(f0_by_preset["male_young"], f0_by_preset["male_mature"])

    def test_crossling_zh_en_mixed_and_balance(self) -> None:
        for preset_id in ("female_std", "male_young"):
            vp = VoiceParams.from_dict({"preset_id": preset_id})
            rec_zh = self.engine.synthesize_segment("今天天气真好。", vp)
            rec_en = self.engine.synthesize_segment("Hello world, welcome to PyBot.", vp)
            rec_mix = self.engine.synthesize_segment("今天用 PyBot 测试 hello world，2026年很棒！", vp)

            _check_wav_bytes(rec_zh.wav_bytes)
            _check_wav_bytes(rec_en.wav_bytes)
            _check_wav_bytes(rec_mix.wav_bytes)

            self.assertGreater(rec_en.n_en, 0)
            self.assertEqual(rec_en.n_zh, 0)
            self.assertGreater(rec_mix.n_zh, 0)
            self.assertGreater(rec_mix.n_en, 0)

            diff_db = abs(_active_rms_db(rec_zh.samples) - _active_rms_db(rec_en.samples))
            self.assertLessEqual(diff_db, 2.5, f"{preset_id} ZH vs EN RMS imbalance: {diff_db:.2f} dB")

    def test_zh_only_mode_and_rule_toggles(self) -> None:
        vp = VoiceParams.from_dict({
            "preset_id": "female_gentle",
            "crossling_mode": "zh_only",
            "rules_enabled": {
                "glottal_f0_coupling": False,
                "obstruent_crispness": True,
                "nasal_antiresonance": False,
                "boundary_coarticulation": True,
            },
        })
        rec = self.engine.synthesize_segment("你好 PyBot 测试。", vp)
        _check_wav_bytes(rec.wav_bytes)
        self.assertEqual(rec.n_en, 0)
        self.assertGreater(rec.n_zh, 0)

    def test_voice_clone_extract_and_crossling_synth(self) -> None:
        sample_path = REPO_ROOT / "legacy" / "exp_v8_data" / "corpus" / "zh" / "SSB0073" / "SSB00730001.wav"
        self.assertTrue(sample_path.exists(), f"Missing real speaker sample: {sample_path}")
        wav_bytes = sample_path.read_bytes()

        prof = self.engine.extract_and_save_clone_profile("测试男声", [wav_bytes], save_to_disk=False)
        self.assertLess(prof.f0_median_hz, 155.0)
        self.assertLess(prof.serialized_size_bytes(), 8192)

        try:
            rec_zh = self.engine.synthesize_segment(
                "今天天气真不错，测试样本模仿效果。",
                {"clone_profile_id": prof.profile_id, "imitation_strength": 0.90, "clone_f0": True},
            )
            rec_en = self.engine.synthesize_segment(
                "Hello world, testing voice cloning today.",
                {"clone_profile_id": prof.profile_id, "imitation_strength": 0.90, "clone_f0": True},
            )
            rec_mix = self.engine.synthesize_segment(
                "欢迎使用 PyBot 样本模仿，Hello 2026！",
                {"clone_profile_id": prof.profile_id, "imitation_strength": 0.90, "clone_f0": True},
            )

            for rec in (rec_zh, rec_en, rec_mix):
                _check_wav_bytes(rec.wav_bytes)
                self.assertEqual(rec.preset_name, "模仿:测试男声")
                self.assertLess(rec.f0_median_hz, 155.0)

            self.assertGreater(rec_en.n_en, 0)
            self.assertGreater(rec_mix.n_zh, 0)
            self.assertGreater(rec_mix.n_en, 0)

            rec_up = self.engine.synthesize_segment(
                "今天天气真不错，测试样本模仿效果。",
                {
                    "clone_profile_id": prof.profile_id,
                    "imitation_strength": 0.90,
                    "clone_f0": True,
                    "pitch_shift_st": 3.0,
                },
            )
            _check_wav_bytes(rec_up.wav_bytes)
            self.assertGreater(rec_up.f0_median_hz, rec_zh.f0_median_hz * 1.12)

            rec_rate = self.engine.synthesize_segment(
                "语速匹配测试。",
                {
                    "clone_profile_id": prof.profile_id,
                    "imitation_strength": 0.85,
                    "clone_f0": True,
                    "clone_rate": True,
                },
            )
            _check_wav_bytes(rec_rate.wav_bytes)
        finally:
            self.engine.delete_clone_profile(prof.profile_id)

    def test_expressive_emotion_tags_and_timbre_synth_patch(self) -> None:
        rec_exc = self.engine.synthesize_segment(
            "[吸气] *终于*成功了！[笑]",
            {
                "preset_id": "female_std",
                "emotion_mode": "excited",
                "emotion_intensity": 1.0,
                "timbre_patch": "radio_magnetic",
            },
        )
        _check_wav_bytes(rec_exc.wav_bytes)
        self.assertTrue(len(rec_exc.detected_emotions) > 0)
        self.assertTrue(len(rec_exc.spectral_preview.get("formants_hz", [])) >= 3)
        self.assertLess(rec_exc.rtf, 0.20)

        rec_sad = self.engine.synthesize_segment(
            "唉，这件事我也很难过。",
            {
                "preset_id": "female_std",
                "emotion_mode": "sad",
                "emotion_intensity": 1.0,
            },
        )
        _check_wav_bytes(rec_sad.wav_bytes)
        self.assertIn("sad", rec_sad.detected_emotions)
        self.assertLess(rec_sad.rtf, 0.20)


class TestWebUIHttpAPI(unittest.TestCase):
    engine: LarynxStudioEngine

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = LarynxStudioEngine()
        cls.engine.ensure_ready()

    def _http_json(self, base_url: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
        if payload is None:
            req = Request(f"{base_url}{path}", method="GET")
        else:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = Request(
                f"{base_url}{path}",
                data=raw,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
        with urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_end_to_end_http_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            srv = create_server("127.0.0.1", 0, engine=self.engine, quiet=True)
            port = srv.server_address[1]
            base = f"http://127.0.0.1:{port}"
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            try:
                # 1. GET /
                with urlopen(f"{base}/", timeout=10) as resp:
                    self.assertEqual(resp.status, 200)
                    html = resp.read().decode("utf-8")
                    self.assertIn("PyBot-Larynx", html)
                    self.assertIn("样本模仿", html)
                    self.assertIn("物理频谱合成器", html)
                    self.assertIn("/api/synthesize", html)
                    self.assertIn("/api/clone/extract", html)

                # 2. GET /api/status
                st_code, st = self._http_json(base, "/api/status")
                self.assertEqual(st_code, 200)
                self.assertTrue(st["ready"])
                self.assertEqual(st["engine_stage"], "stage8")
                self.assertEqual(st["crossling_engine"], "stage9_donor_transfer")
                self.assertTrue(st["en_articulation_loaded"])
                self.assertEqual(len(st["presets"]), 9)
                self.assertEqual(len(st["timbre_patches"]), 7)
                self.assertEqual(len(st["emotions"]), 10)

                # 3. POST /api/split
                _, sp = self._http_json(base, "/api/split", {
                    "text": "第一句测试。\nHello PyBot!",
                    "mode": "sentence",
                    "max_chars": 80,
                })
                segs = sp["segments"]
                self.assertEqual(len(segs), 2)

                # 4. POST /api/synthesize + GET audio + Range probe
                _, s1 = self._http_json(base, "/api/synthesize", {
                    "text": segs[0]["text"],
                    "params": {"preset_id": "female_std"},
                })
                _, s2 = self._http_json(base, "/api/synthesize", {
                    "text": segs[1]["text"],
                    "params": {"preset_id": "female_std"},
                })
                c1 = s1["clip"]
                c2 = s2["clip"]
                self.assertEqual(c1["version"], 1)
                self.assertIn("detected_emotions", c1)
                self.assertTrue(len(c1.get("spectral_preview", {}).get("formants_hz", [])) >= 3)

                with urlopen(f"{base}{c1['audio_url']}", timeout=10) as aresp:
                    self.assertEqual(aresp.status, 200)
                    wav1 = aresp.read()
                _check_wav_bytes(wav1)

                r_req = Request(f"{base}/api/audio/{c1['clip_id']}.wav", headers={"Range": "bytes=0-43"})
                with urlopen(r_req, timeout=10) as rresp:
                    self.assertEqual(rresp.status, 206)
                    hdr_bytes = rresp.read()
                    self.assertEqual(len(hdr_bytes), 44)
                    self.assertEqual(hdr_bytes[:4], b"RIFF")

                # 5. In-place regeneration (same clip_id, bumped version, new preset)
                _, r1 = self._http_json(base, "/api/synthesize", {
                    "text": c1["text"],
                    "clip_id": c1["clip_id"],
                    "params": {"preset_id": "male_mature"},
                })
                c1_regen = r1["clip"]
                self.assertEqual(c1_regen["clip_id"], c1["clip_id"])
                self.assertEqual(c1_regen["version"], 2)
                self.assertLess(c1_regen["f0_median_hz"], c1["f0_median_hz"] * 0.65)
                with urlopen(f"{base}{c1_regen['audio_url']}", timeout=10) as aresp2:
                    wav1_new = aresp2.read()
                _check_wav_bytes(wav1_new)
                self.assertNotEqual(wav1, wav1_new)

                # 6. Save individual + merged WAVs
                _, sv1 = self._http_json(base, "/api/save", {
                    "clip_ids": [c1_regen["clip_id"], c2["clip_id"]],
                    "save_dir": tmpdir,
                    "merge": False,
                })
                self.assertEqual(len(sv1["saved_files"]), 2)
                for item in sv1["saved_files"]:
                    p = Path(item["path"])
                    self.assertTrue(p.exists())
                    _check_wav_bytes(p.read_bytes())

                _, sv_merge = self._http_json(base, "/api/save", {
                    "clip_ids": [c1_regen["clip_id"], c2["clip_id"]],
                    "save_dir": tmpdir,
                    "merge": True,
                    "merge_gap_ms": 200,
                    "merge_name": "final_mix",
                })
                mp = Path(sv_merge["merged_path"])
                self.assertTrue(mp.exists())
                _check_wav_bytes(mp.read_bytes())

                # 7. Delete single + clear all
                _, d1 = self._http_json(base, "/api/delete_clip", {"clip_id": c1_regen["clip_id"]})
                self.assertTrue(d1["ok"])
                _, d_all = self._http_json(base, "/api/delete_clip", {"clear_all": True})
                self.assertTrue(d_all["ok"])

                # 8. Voice clone HTTP lifecycle (extract -> list -> synthesize -> delete)
                sample_path = REPO_ROOT / "legacy" / "exp_v8_data" / "corpus" / "zh" / "SSB0073" / "SSB00730001.wav"
                b64_wav = base64.b64encode(sample_path.read_bytes()).decode("ascii")
                ex_code, ex_res = self._http_json(
                    base,
                    "/api/clone/extract",
                    {"name": "HTTP克隆男声", "clips_b64": [b64_wav], "save": False},
                )
                self.assertEqual(ex_code, 200)
                prof = ex_res["profile"]
                pid = prof["profile_id"]
                self.assertLess(prof["f0_median_hz"], 155.0)

                _, lp_res = self._http_json(base, "/api/clone/profiles")
                self.assertIn(pid, [p["profile_id"] for p in lp_res["profiles"]])

                sc_code, sc_res = self._http_json(
                    base,
                    "/api/synthesize",
                    {
                        "text": "测试接口样本模仿 Hello PyBot。",
                        "params": {"clone_profile_id": pid, "imitation_strength": 0.90},
                    },
                )
                self.assertEqual(sc_code, 200)
                clone_clip = sc_res["clip"]
                self.assertEqual(clone_clip["preset_name"], "模仿:HTTP克隆男声")
                self.assertLess(clone_clip["f0_median_hz"], 155.0)
                with urlopen(f"{base}{clone_clip['audio_url']}", timeout=10) as cresp:
                    self.assertEqual(cresp.status, 200)
                    _check_wav_bytes(cresp.read())

                del_code, del_res = self._http_json(base, "/api/clone/delete", {"profile_id": pid})
                self.assertEqual(del_code, 200)
                self.assertTrue(del_res["ok"])
                self.assertNotIn(pid, [p["profile_id"] for p in del_res["profiles"]])
            finally:
                srv.shutdown()
                srv.server_close()


if __name__ == "__main__":
    unittest.main()
