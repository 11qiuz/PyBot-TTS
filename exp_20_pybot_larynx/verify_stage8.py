"""Stage 8 Verification & Expressive / Synthesizer Showcase (`exp_20_pybot_larynx`)."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import EXP_DIR, SAMPLES_DIR, SR  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import (  # noqa: E402
    STAGE7_KNOWLEDGE_PATH,
    AcousticKnowledgeStage7,
)
from exp_20_pybot_larynx.arbiter import Arbiter  # noqa: E402
from exp_20_pybot_larynx.expressive import render_expressive_utterance  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import LarynxControl  # noqa: E402
from exp_20_pybot_larynx.larynx.synth_v2 import synthesize_larynx_v2  # noqa: E402
from exp_20_pybot_larynx.larynx.timbre_synth import (  # noqa: E402
    TIMBRE_PATCHES,
    TimbreSynthParams,
    extract_spectral_profile_preview,
)
from exp_20_pybot_larynx.rules import default_stage5_rules  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity  # noqa: E402
from exp_v14_fluent_zh_tts.eval_v14 import _rms_norm, _write_pcm16_wav  # noqa: E402
from exp_v15_bznsyp_tts.index_bznsyp_v15 import load_speaker_assets_v15  # noqa: E402
from pybot_webui.crossling import build_utterance_control  # noqa: E402
from pybot_webui.engine import apply_voice_params  # noqa: E402
from pybot_webui.presets import VoiceParams  # noqa: E402
from exp_20_pybot_larynx.larynx.timbre_synth import apply_timbre_synth  # noqa: E402

STAGE8_RESULTS_PATH = EXP_DIR / "stage8_results.json"
STAGE8_REPORT_PATH = EXP_DIR / "STAGE8_REPORT.md"
LISTEN_STAGE8_PATH = EXP_DIR / "listen_stage8.html"


def _measure_ctrl_and_audio(y: np.ndarray, ctrl: LarynxControl, rtf: float) -> dict[str, Any]:
    san = check_wav_sanity(y, SR, expected_sr=SR)
    vm = ctrl.voiced & (ctrl.f0_hz > 40.0)
    f0_v = ctrl.f0_hz[vm]
    f0_med = float(np.median(f0_v)) if f0_v.size else 0.0
    f0_iqr_st = (
        float(12.0 * math.log2(float(np.percentile(f0_v, 75)) / max(40.0, float(np.percentile(f0_v, 25)))))
        if f0_v.size >= 4
        else 0.0
    )
    tilt_med = float(np.median(ctrl.glottal_tilt_db_per_oct[vm])) if (ctrl.glottal_tilt_db_per_oct is not None and np.any(vm)) else 0.0
    rd_med = float(np.median(ctrl.rd[vm])) if (ctrl.rd is not None and np.any(vm)) else 1.0
    bap_med = float(np.median(ctrl.bap[vm])) if np.any(vm) else 0.0
    spec_prev = extract_spectral_profile_preview(ctrl, max_frames=16)
    return {
        "sanity_ok": bool(san.get("ok", False)),
        "duration_s": round(len(y) / float(SR), 3),
        "rtf": round(float(rtf), 4),
        "f0_median_hz": round(f0_med, 1),
        "f0_iqr_st": round(f0_iqr_st, 2),
        "tilt_db_oct": round(tilt_med, 2),
        "rd_median": round(rd_med, 3),
        "bap_median": round(bap_med, 3),
        "formants_hz": spec_prev["formants_hz"],
        "band_levels_db": spec_prev["band_levels_db"],
    }


def _render_listen_stage8_html(
    emo_items: list[dict[str, Any]],
    story_items: list[dict[str, Any]],
    patch_items: list[dict[str, Any]],
    out_path: Path,
) -> None:
    payload = json.dumps({"emo": emo_items, "story": story_items, "patch": patch_items}, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-Larynx Stage 8 · 物理频谱合成器 & 情绪演播试听台</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#eef1f4;background:#11151c;max-width:1040px;margin:1.8rem auto;padding:0 1rem}}
h1{{font-size:1.5rem;margin-bottom:.25rem}}
h2{{font-size:1.15rem;margin:1.4rem 0 .5rem;color:#93c5fd}}
.sub{{color:#9aa6b8;margin-bottom:1.1rem}}
.card{{background:#19202b;border:1px solid #293345;padding:1rem 1.2rem;margin:.85rem 0;border-radius:10px}}
.row{{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;padding:.4rem 0;border-bottom:1px dashed #273142}}
.row:last-child{{border-bottom:none}}
audio{{width:min(100%,24rem);height:34px}}
.meta{{color:#9aa6b8;font-size:.86rem}}
</style>
</head>
<body>
<h1>PyBot-Larynx Stage 8 · 物理频谱合成器 (LSF-22) & ElevenLabs 风格情绪演播试听台</h1>
<p class="sub">0.0 MB 神经权重 · 所有呼吸/叹息/笑声/气泡音/颤音与共振峰塑形均为 100% 物理实时合成（-20 dBFS 统一响度）。</p>
<div id="root"></div>
<script>
const DATA = {payload};
function renderSec(title, arr) {{
  const root = document.getElementById('root');
  const h2 = document.createElement('h2'); h2.textContent = title; root.append(h2);
  const card = document.createElement('div'); card.className = 'card';
  arr.forEach(it => {{
    const r = document.createElement('div'); r.className = 'row';
    const b = document.createElement('b'); b.style.minWidth = '9rem'; b.textContent = it.label;
    const au = document.createElement('audio'); au.controls = true; au.preload = 'none'; au.src = it.file;
    const m = document.createElement('span'); m.className = 'meta'; m.textContent = it.meta;
    r.append(b, au, m); card.append(r);
  }});
  root.append(card);
}}
renderSec('一、同一句话 9 种情绪对比演绎', DATA.emo);
renderSec('二、ElevenLabs 风格长文本情绪演播（呼吸 / 叹气 / 轻笑 / 重读 / 中英混读）', DATA.story);
renderSec('三、物理频谱合成器 7 张音色卡对比（基于 LSF-22 / 共振峰雕刻 / 颤音）', DATA.patch);
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def run_stage8_verification() -> dict[str, Any]:
    t0 = time.perf_counter()
    kb = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
    coh = load_speaker_assets_v15()["glottal_bank"].coherent_phasor
    arb = Arbiter()
    rules = default_stage5_rules()

    # Warmup
    render_expressive_utterance("预热测试。", "warm", kb, arbiter=arb, rules=rules, coherent_phasor=coh, seed=1)

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Anchor sentence across 9 emotions
    anchor_text = "今天我们终于完成了这次非常重要的测试，真是让人难忘。"
    emotions = ("neutral", "happy", "excited", "sad", "angry", "gentle", "surprised", "fearful", "whisper")
    emo_results: dict[str, dict[str, Any]] = {}
    emo_listen: list[dict[str, Any]] = []
    arb_errors = 0

    for emo in emotions:
        ts = time.perf_counter()
        wav, info = render_expressive_utterance(
            anchor_text, f"s8_emo_{emo}", kb, arbiter=arb, rules=rules,
            emotion_mode=emo, emotion_intensity=0.85, stability=0.55,
            auto_breath=False, coherent_phasor=coh, seed=800,
        )
        dt = time.perf_counter() - ts
        rtf = dt / max(1e-6, len(wav) / SR)
        m = _measure_ctrl_and_audio(wav, info["ctrl"], rtf)
        emo_results[emo] = m
        for tr in info["arbiter_traces"]:
            arb_errors += sum(1 for mod in tr.get("modules", []) if mod.get("error"))
        fname = f"s8_emo_{emo}.wav"
        _write_pcm16_wav(SAMPLES_DIR / fname, _rms_norm(wav, target_dbfs=-20.0), sr=SR)
        emo_listen.append({
            "label": emo,
            "file": f"samples/{fname}",
            "meta": f"F0={m['f0_median_hz']}Hz IQR={m['f0_iqr_st']}st dur={m['duration_s']}s tilt={m['tilt_db_oct']} bap={m['bap_median']} RTF={m['rtf']}",
        })

    # 2. Multi-emotion ElevenLabs-style scripts (via full WebUI crossling pipeline)
    scripts = [
        ("story_drama_zh", "[吸气] *天呐*，你真的不记得昨晚发生什么了吗？！ [叹气] 我在门口等了整整三个小时…… [温柔] 算了，只要你没事就好。"),
        ("story_joy_mix", "[开心] 哈哈，[笑] 别总是板着脸啦！Today is going to be **AWESOME**, 2026年我们一起冲呀！"),
        ("story_angry_suspense", "[愤怒] **够了**！你凭什么随便做决定？！ [停顿0.4s] [害怕] 等等……你听外面是不是有脚步声……"),
        ("story_asmr_en", "[whisper] [breath] Listen closely... every formant and breath sound is shaped in real time, zero neural weights."),
    ]
    story_results: dict[str, dict[str, Any]] = {}
    story_listen: list[dict[str, Any]] = []
    for sid, stext in scripts:
        vp = VoiceParams.from_dict({"emotion_mode": "auto", "emotion_intensity": 0.85, "stability": 0.50, "auto_breath": True})
        ts = time.perf_counter()
        ctrl, meta = build_utterance_control(stext, sid, kb, arb, rules, vp)
        ctrl = apply_voice_params(ctrl, vp, vp.seed)
        wav = synthesize_larynx_v2(
            ctrl, glottal_mode="dynamic_coherent", coherent_phasor=coh,
            subharm_strength=float(meta.get("subharm_strength", 0.0)), seed=vp.seed,
        )
        dt = time.perf_counter() - ts
        rtf = dt / max(1e-6, len(wav) / SR)
        m = _measure_ctrl_and_audio(wav, ctrl, rtf)
        m["detected_emotions"] = meta.get("detected_emotions", [])
        m["n_vocalizations"] = int(meta.get("n_vocalizations", 0))
        story_results[sid] = m
        fname = f"s8_{sid}.wav"
        _write_pcm16_wav(SAMPLES_DIR / fname, _rms_norm(wav, target_dbfs=-20.0), sr=SR)
        story_listen.append({
            "label": sid,
            "file": f"samples/{fname}",
            "meta": f"{stext[:36]}... | 情绪={m['detected_emotions']} | 呼吸/声效={m['n_vocalizations']} | dur={m['duration_s']}s | RTF={m['rtf']}",
        })

    # 3. 7 Physical Spectrum Synthesizer patches
    patch_text = "夜深了，城市灯火渐暗，欢迎收听本期的物理合成器人声实验室。"
    patch_results: dict[str, dict[str, Any]] = {}
    patch_listen: list[dict[str, Any]] = []
    for pat in TIMBRE_PATCHES:
        pid = pat["id"]
        tp = TimbreSynthParams.from_dict(pat["params"])
        ts = time.perf_counter()
        wav, info = render_expressive_utterance(
            patch_text, f"s8_pat_{pid}", kb, arbiter=arb, rules=rules,
            emotion_mode="neutral", emotion_intensity=0.0, stability=0.70,
            auto_breath=False, synth_params=tp, coherent_phasor=coh, seed=888,
        )
        dt = time.perf_counter() - ts
        rtf = dt / max(1e-6, len(wav) / SR)
        m = _measure_ctrl_and_audio(wav, info["ctrl"], rtf)
        patch_results[pid] = m
        fname = f"s8_patch_{pid}.wav"
        _write_pcm16_wav(SAMPLES_DIR / fname, _rms_norm(wav, target_dbfs=-20.0), sr=SR)
        patch_listen.append({
            "label": f"{pat['name']} ({pid})",
            "file": f"samples/{fname}",
            "meta": f"F={m['formants_hz'][:3]}Hz | chest={m['band_levels_db']['chest']} pres={m['band_levels_db']['presence']} air={m['band_levels_db']['air']} dB | RTF={m['rtf']}",
        })

    _render_listen_stage8_html(emo_listen, story_listen, patch_listen, LISTEN_STAGE8_PATH)

    all_rows = list(emo_results.values()) + list(story_results.values()) + list(patch_results.values())
    all_rtfs = [r["rtf"] for r in all_rows]
    f0_exc_sad_st = 12.0 * math.log2(emo_results["excited"]["f0_median_hz"] / max(40.0, emo_results["sad"]["f0_median_hz"]))

    gates = {
        "all_sanity_ok": all(r["sanity_ok"] for r in all_rows),
        "zero_arbiter_errors": arb_errors == 0,
        "median_rtf_le_006": float(np.median(all_rtfs)) <= 0.06,
        "f0_excited_vs_sad_ge_3st": f0_exc_sad_st >= 3.0,
        "dur_sad_gt_neutral_gt_excited": emo_results["sad"]["duration_s"] > emo_results["neutral"]["duration_s"] > emo_results["excited"]["duration_s"],
        "tilt_angry_vs_sad_ge_1_5": (emo_results["angry"]["tilt_db_oct"] - emo_results["sad"]["tilt_db_oct"]) >= 1.5,
        "bap_whisper_vs_neutral_ge_0_3": (emo_results["whisper"]["bap_median"] - emo_results["neutral"]["bap_median"]) >= 0.30,
        "patch_radio_boosts_chest": patch_results["radio_magnetic"]["band_levels_db"]["chest"] > patch_results["neutral"]["band_levels_db"]["chest"] + 1.0,
        "patch_vintage_cuts_air": patch_results["vintage_phone"]["band_levels_db"]["air"] < patch_results["neutral"]["band_levels_db"]["air"] - 8.0,
        "patch_anime_raises_f1": patch_results["anime_sweet"]["formants_hz"][0] > patch_results["cinematic_epic"]["formants_hz"][0] + 30.0,
    }
    verdict = "GO" if all(gates.values()) else "NO_GO"

    results = {
        "verdict": verdict,
        "runtime_s": round(time.perf_counter() - t0, 2),
        "zero_ml_weights": True,
        "median_rtf": round(float(np.median(all_rtfs)), 4),
        "f0_excited_vs_sad_st": round(f0_exc_sad_st, 2),
        "gates": gates,
        "emotions": emo_results,
        "stories": story_results,
        "patches": patch_results,
    }
    STAGE8_RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# PyBot-Larynx Stage 8 物理频谱合成器 + 情绪演播引擎验证报告",
        "",
        f"- **判定结论 (`verdict`)**: **`{verdict}`** (`median_rtf={results['median_rtf']}`, `zero_ml_weights=True`, `all_sanity_ok={gates['all_sanity_ok']}`, `zero_arbiter_errors={gates['zero_arbiter_errors']}`)",
        f"- **情绪极差核心验证**: `excited` vs `sad` 基频分离 = **`+{results['f0_excited_vs_sad_st']} st`** (`>= 3.0 st`); 时长序 `sad ({emo_results['sad']['duration_s']}s) > neutral ({emo_results['neutral']['duration_s']}s) > excited ({emo_results['excited']['duration_s']}s)`; 倾斜度差 `angry - sad = {emo_results['angry']['tilt_db_oct'] - emo_results['sad']['tilt_db_oct']:+.2f} dB/oct`; 气声差 `whisper - neutral = {emo_results['whisper']['bap_median'] - emo_results['neutral']['bap_median']:+.3f}`",
        "",
        "## 1. 同一锚定句 9 种情绪物理维度实测",
        "",
        "| 情绪 | F0 中位 (Hz) | F0 IQR (st) | 时长 (s) | 倾斜度 (dB/oct) | 声门 Rd | 气声 BAP | 胸/明/空 (dB) | CPU RTF | Sanity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for emo in emotions:
        r = emo_results[emo]
        b = r["band_levels_db"]
        lines.append(
            f"| `{emo}` | {r['f0_median_hz']:.1f} | {r['f0_iqr_st']:.2f} | {r['duration_s']:.2f} | {r['tilt_db_oct']:+.2f} | "
            f"{r['rd_median']:.2f} | {r['bap_median']:.3f} | {b['chest']:+.1f}/{b['presence']:+.1f}/{b['air']:+.1f} | {r['rtf']:.4f} | {'OK' if r['sanity_ok'] else 'FAIL'} |"
        )

    lines += [
        "",
        "## 2. 物理频谱合成器 7 张音色卡实测（LSF-22 共振峰 + 频段均衡）",
        "",
        "| 音色卡 | F1 / F2 / F3 (Hz) | 胸腔 (dB) | 暖度 (dB) | 鼻腔聚焦 (dB) | 歌手峰 (dB) | 空气感 (dB) | CPU RTF | Sanity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pat in TIMBRE_PATCHES:
        r = patch_results[pat["id"]]
        b = r["band_levels_db"]
        fmts = "/".join(str(int(round(x))) for x in r["formants_hz"][:3])
        lines.append(
            f"| `{pat['id']}` ({pat['name']}) | {fmts} | {b['chest']:+.1f} | {b['warmth']:+.1f} | {b['nasal']:+.1f} | {b['presence']:+.1f} | {b['air']:+.1f} | {r['rtf']:.4f} | {'OK' if r['sanity_ok'] else 'FAIL'} |"
        )

    lines += [
        "",
        "## 3. ElevenLabs 风格复合剧本实测",
        "",
        "| 剧本 | 识别情绪序列 | 非言语呼吸/笑/叹 | 时长 (s) | CPU RTF | Sanity |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for sid, _ in scripts:
        r = story_results[sid]
        lines.append(f"| `{sid}` | `{r['detected_emotions']}` | {r['n_vocalizations']} | {r['duration_s']:.2f} | {r['rtf']:.4f} | {'OK' if r['sanity_ok'] else 'FAIL'} |")

    lines += [
        "",
        "## 4. 离线试听台",
        "- [listen_stage8.html](listen_stage8.html)（包含 9 种情绪对比 + 4 段多情绪呼吸/叹气/轻笑中英演播 + 7 张合成器音色卡 WAV）",
    ]
    STAGE8_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "median_rtf": results["median_rtf"], "f0_excited_vs_sad_st": results["f0_excited_vs_sad_st"], "gates": gates}, ensure_ascii=False, indent=2), flush=True)
    return results


if __name__ == "__main__":
    run_stage8_verification()
