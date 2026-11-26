"""Stage 11 Audition & Report Generator (Stage 10 old `slt` vs Stage 11 new `LJSpeech` + Real Human)."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import EXP_DIR, SR  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import STAGE7_KNOWLEDGE_PATH, AcousticKnowledgeStage7  # noqa: E402
from exp_20_pybot_larynx.arbiter import Arbiter  # noqa: E402
from exp_20_pybot_larynx.rules import default_stage5_rules  # noqa: E402
from exp_20_pybot_larynx.verify_stage10_bilingual import (  # noqa: E402
    _synth_stage10_utt,
    compute_forensic_metrics,
)
from exp_v14_fluent_zh_tts.eval_v14 import _rms_norm, _write_pcm16_wav  # noqa: E402
from exp_v15_bznsyp_tts.index_bznsyp_v15 import load_speaker_assets_v15  # noqa: E402
from pybot_webui.presets import VoiceParams  # noqa: E402

OUT_DIR = EXP_DIR / "samples_stage11_ceiling"
HTML_PATH = EXP_DIR / "listen_stage11_ceiling.html"
REPORT_PATH = EXP_DIR / "STAGE11_CEILING_REPORT.md"


def build_stage11_audition_and_report() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kb = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
    arb = Arbiter()
    rules = default_stage5_rules()
    vp = VoiceParams.from_dict({"preset_id": "female_std"})
    vp_emo = VoiceParams.from_dict({"preset_id": "female_std", "emotion_mode": "excited", "emotion_intensity": 0.90})
    phasor = load_speaker_assets_v15().get("coherent_phasor")

    cases = [
        ("en_studio", "This is a smooth benchmark of bilingual speech synthesis in our studio.", vp),
        ("en_story", "She walked carefully through the quiet autumn forest before sunrise.", vp),
        ("en_tech", "Read the live audio stream and close the record before we present it.", vp),
        ("en_stops", "Pat took ten bright copper pots back to the deep dark cellar.", vp),
        ("en_question", "Did you really think the apple in the silver box was fresh today?", vp),
        ("mix_en_zh_vowel", "Let me know 你今天几点出发。", vp),
        ("mix_en_zh_nasal", "我选了 green tea 按你的清单买的。", vp),
        ("mix_tech", "今天用 PyBot 测试 this smooth cross lingual speech，表现非常自然。", vp),
        ("mix_multi", "先打一个 tag，then run the nightly build，最后把 log 发给我。", vp),
        ("mix_excited", "[激动] *终于*成功了！This feels amazing![笑]", vp_emo),
    ]

    s10_rows_path = OUT_DIR / "s10_baseline_metrics.json"
    s10_map: dict[str, dict[str, Any]] = {}
    if s10_rows_path.exists():
        for r in json.loads(s10_rows_path.read_text(encoding="utf-8")):
            s10_map[str(r["id"])] = r

    s11_rows: list[dict[str, Any]] = []
    # Warm up cache once before timing
    _synth_stage10_utt("Warm up sentence.", "warmup", kb, arb, rules, vp, phasor, seed=1)

    for i, (cid, txt, vpi) in enumerate(cases):
        mod, y, rtf, _ = _synth_stage10_utt(txt, f"s11_{cid}", kb, arb, rules, vpi, phasor, seed=2026 + i)
        _write_pcm16_wav(OUT_DIR / f"s11_{cid}.wav", y, SR)
        fm = compute_forensic_metrics(y, SR)
        vm = mod.voiced & (mod.f0_hz > 40.0)
        fm["voiced_jitter_med"] = round(float(np.median(mod.jitter_frac[vm])), 5) if np.any(vm) else 0.0
        fm["voiced_shimmer_med"] = round(float(np.median(mod.shimmer_frac[vm])), 5) if np.any(vm) else 0.0
        fm["voiced_bap0_med"] = round(float(np.median(mod.bap[vm, 0])), 4) if np.any(vm) else 0.0
        fm["voiced_bap1_med"] = round(float(np.median(mod.bap[vm, 1])), 4) if np.any(vm) else 0.0
        fm["rtf"] = round(rtf, 4)

        b_old = s10_map.get(cid, {})
        meta_str = (
            f"S10(slt): q8={b_old.get('q8_24','?')} jit={b_old.get('voiced_jitter_med','?')} bap1={b_old.get('voiced_bap1_med','?')} HNR={b_old.get('hnr_db','?')}<br>"
            f"S11(LJ): q8={fm['q8_24']} jit={fm['voiced_jitter_med']} bap1={fm['voiced_bap1_med']} HNR={fm['hnr_db']} RTF={rtf:.4f}"
        )
        s11_rows.append({
            "id": cid,
            "text": txt,
            "metrics_s11": fm,
            "meta": meta_str,
            "wav_10": f"samples_stage11_ceiling/s10_{cid}.wav",
            "wav_11": f"samples_stage11_ceiling/s11_{cid}.wav",
        })

    (OUT_DIR / "s11_metrics.json").write_text(json.dumps(s11_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # Copy 4 LJSpeech heldout human references + synthesize Stage 11 native projection for direct human comparison
    heldout_pkl = EXP_DIR / "crossling_data" / "en_stage10_heldout.pkl"
    human_rows: list[dict[str, Any]] = []
    if heldout_pkl.exists():
        with open(heldout_pkl, "rb") as f:
            ho = pickle.load(f)
        utts = ho.get("utterances", [])[:4]
        for idx, u in enumerate(utts):
            uid = str(u["id"])
            txt = str(u["text"])
            href = OUT_DIR / f"human_{uid}.wav"
            _write_pcm16_wav(href, _rms_norm(np.asarray(u["wav"], dtype=np.float64), -19.5), SR)
            _, y11, _, _ = _synth_stage10_utt(txt, f"ho11_{uid}", kb, arb, rules, vp, phasor, seed=2026 + idx)
            h11 = OUT_DIR / f"s11_ho_{uid}.wav"
            _write_pcm16_wav(h11, y11, SR)
            human_rows.append({
                "id": uid,
                "text": txt,
                "wav_human": f"samples_stage11_ceiling/{href.name}",
                "wav_11": f"samples_stage11_ceiling/{h11.name}",
            })

    payload = json.dumps({"ab": s11_rows, "human": human_rows}, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-Larynx Stage 11 · LJSpeech 全频自然度 & Anti-Buzz 对比试听台</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#eef1f4;background:#0f141c;max-width:1180px;margin:1.8rem auto;padding:0 1rem}}
h1{{font-size:1.5rem;margin-bottom:.25rem}}
h2{{font-size:1.15rem;margin:1.3rem 0 .5rem;color:#7dd3fc}}
.sub{{color:#94a3b8;margin-bottom:1.1rem}}
.card{{background:#18202c;border:1px solid #293548;padding:1rem 1.2rem;margin:.85rem 0;border-radius:10px}}
.row{{display:grid;grid-template-columns:19rem 1fr 1fr;gap:.8rem;align-items:center;padding:.65rem 0;border-bottom:1px dashed #253042}}
.row:last-child{{border-bottom:none}}
audio{{width:100%;height:32px}}
.lbl{{font-weight:600;color:#e2e8f0}}
.meta{{font-size:.78rem;color:#93c5fd}}
.txt{{font-size:.82rem;color:#cbd5e1;margin-top:.2rem}}
@media(max-width:920px){{.row{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>PyBot-Larynx Stage 11 · 全频 LJSpeech + Anti-Buzz 清洗对比试听台</h1>
<p class="sub">22.05kHz->24kHz 全频自然库（去除 +48dB 伪高频 BWE） · 去除零 jitter/BAP 蜂鸣钳位 · 去除 +23dB 清音闭塞抽水 · EN->ZH 边界顺滑桥接</p>
<h2>1. Stage 10 旧 (CMU-slt + 合成钳位) vs Stage 11 新 (LJSpeech + 自然人声声门)</h2>
<div id="ab" class="card"></div>
<h2>2. 未见留出 LJSpeech 真实人声 vs Stage 11 合成本尊音色投影</h2>
<div id="human" class="card"></div>
<script>
const DATA = {payload};
document.getElementById("ab").innerHTML = DATA.ab.map(r => `
  <div class="row">
    <div><div class="lbl">${{r.id}}</div><div class="txt">${{r.text}}</div><div class="meta">${{r.meta}}</div></div>
    <div><div class="meta">Stage 10 (Old slt)</div><audio controls preload="metadata" src="${{r.wav_10}}"></audio></div>
    <div><div class="meta">Stage 11 (New LJSpeech)</div><audio controls preload="metadata" src="${{r.wav_11}}"></audio></div>
  </div>
`).join("");
document.getElementById("human").innerHTML = DATA.human.map(r => `
  <div class="row">
    <div><div class="lbl">${{r.id}}</div><div class="txt">${{r.text}}</div></div>
    <div><div class="meta">Real Human (LJSpeech LJ050)</div><audio controls preload="metadata" src="${{r.wav_human}}"></audio></div>
    <div><div class="meta">Stage 11 (BZNSYP Projected)</div><audio controls preload="metadata" src="${{r.wav_11}}"></audio></div>
  </div>
`).join("");
</script>
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")

    # Assemble Stage 11 Report using directly measured outputs
    res_s10 = json.loads((EXP_DIR / "stage10_bilingual_results.json").read_text(encoding="utf-8")) if (EXP_DIR / "stage10_bilingual_results.json").exists() else {}
    onsets_path = EXP_DIR / "stage10_onsets_results.json"
    if onsets_path.exists():
        raw_ons = onsets_path.read_text(encoding="utf-8-sig")
        res_onsets = json.loads(raw_ons)
        onsets_path.write_text(json.dumps(res_onsets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        res_onsets = {}
    src_old = json.loads((EXP_DIR / "crossling_data" / "stage10_slt_backup" / "en_source_profile.json").read_text(encoding="utf-8")) if (EXP_DIR / "crossling_data" / "stage10_slt_backup" / "en_source_profile.json").exists() else {}
    src_new = json.loads((EXP_DIR / "crossling_data" / "en_source_profile.json").read_text(encoding="utf-8")) if (EXP_DIR / "crossling_data" / "en_source_profile.json").exists() else {}
    kb_old = json.loads((EXP_DIR / "crossling_data" / "stage10_slt_backup" / "knowledge_en_stage10_summary.json").read_text(encoding="utf-8")) if (EXP_DIR / "crossling_data" / "stage10_slt_backup" / "knowledge_en_stage10_summary.json").exists() else {}
    kb_new = json.loads((EXP_DIR / "crossling" / "knowledge_en_stage10_summary.json").read_text(encoding="utf-8")) if (EXP_DIR / "crossling" / "knowledge_en_stage10_summary.json").exists() else {}

    ons_fem = res_onsets.get("results", {}).get("female_std_new", {}).get("connected_control_f0_before_after", {})
    enzh_v = ons_fem.get("enzh_vowel", {}).get("median_abs", [[], []])[-1]
    enzh_n = ons_fem.get("enzh_nasal", {}).get("median_abs", [[], []])[-1]
    enzh_all = ons_fem.get("enzh", {}).get("median_abs", [[], []])[-1]
    zhen_all = ons_fem.get("zhen", {}).get("median_abs", [[], []])[-1]

    md = [
        "# PyBot-Larynx Stage 11 全频自然度 & 跨语种天花板升级报告",
        "",
        f"- **Verdict**: **`{res_s10.get('verdict', 'UNKNOWN')}`** (`median_rtf={res_s10.get('median_rtf', '?')}`, `ml_weights=0.0 MB`)",
        f"- **语料源升级**: `CMU-ARCTIC slt` (`effective_bw={src_old.get('effective_bw_hz', '?')} Hz`, `F0={src_old.get('f0_median_hz', '?')} Hz`) -> **`LJSpeech-1.1`** (`effective_bw={src_new.get('effective_bw_hz', '?')} Hz`, `F0={src_new.get('f0_median_hz', '?')} Hz`, TRAIN=`{kb_new.get('n_train_utts', '?')}` utts / `{kb_new.get('n_train_tokens', '?')}` tokens)",
        f"- **上下文 Ridge (`w_pns`) 覆盖**: `{kb_old.get('ridge_counts', {}).get('w_pns', '?')}` -> **`{kb_new.get('ridge_counts', {}).get('w_pns', '?')}`**, `.npz` = **`{kb_new.get('npz_size_mb', '?')} MB`**",
        "",
        "## 1. Stage 11 反蜂鸣与物理真声核心门禁",
        f"- `src_effective_bw_hz` = `{res_s10.get('bilingual_unity', {}).get('src_effective_bw_hz', '?')}` Hz",
        f"- `en_voiced_jitter_med` = `{res_s10.get('bilingual_unity', {}).get('en_voiced_jitter_med', '?')}` (旧 Stage 10 被钳死在 `0.0012`)",
        f"- `en_voiced_bap1_med` = `{res_s10.get('bilingual_unity', {}).get('en_voiced_bap1_med', '?')}` (旧 Stage 10 被钳死在 `0.075`)",
        f"- `stop_closure_depth_db` = `{res_s10.get('bilingual_unity', {}).get('stop_closure_depth_db', '?')}` dB",
        f"- `voiced_ltas_rmse_db` = `{res_s10.get('bilingual_unity', {}).get('voiced_ltas_rmse_db', '?')}` dB, `raw_rms_diff_db` = `{res_s10.get('bilingual_unity', {}).get('raw_rms_diff_db', '?')}` dB",
        "",
        "## 2. 未见留出基准 (`en_stage10_heldout.pkl`, LJ050 零泄漏)",
        f"- `Mel-CD`: Stage 9B=`{res_s10.get('heldout_30', {}).get('stage9b_mel_cd_db', '?')}` dB vs Stage 11 Native=**`{res_s10.get('heldout_30', {}).get('stage10_native_mel_cd_db', '?')}` dB**",
        f"- `phone_dur_mae_ms` = **`{res_s10.get('heldout_30', {}).get('phone_dur_mae_ms', '?')}` ms**, `f0_rmse_st` = **`{res_s10.get('heldout_30', {}).get('f0_rmse_st', '?')}` st**",
        "",
        "## 3. 中英切换边界测量 (`verify_stage10_onsets`)",
        f"- `female_std` 无停顿 `EN -> ZH` 响度落差（Stage 10 报告曾高达 `8.79 / 9.95 dB` 元音、`9.79 / 10.96 dB` 鼻音）：现整体 = **`{enzh_all[0] if enzh_all else '?'} dB`**, 元音尾音 = **`{enzh_v[0] if enzh_v else '?'} dB`**, 鼻音尾音 = **`{enzh_n[0] if enzh_n else '?'} dB`**",
        f"- `female_std` 无停顿 `ZH -> EN` 衔接：响度差 = **`{zhen_all[0] if zhen_all else '?'} dB`**, 音高跳变 = **`{zhen_all[1] if len(zhen_all) > 1 else '?'} st`**",
        f"- JSON 原始测量见 [stage10_onsets_results.json](stage10_onsets_results.json)",
        "",
        "## 4. A/B 试听台 (Stage 10 旧 vs Stage 11 新 vs Real Human)",
        "- [listen_stage11_ceiling.html](listen_stage11_ceiling.html)",
    ]
    REPORT_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"html": str(HTML_PATH), "report": str(REPORT_PATH), "n_ab": len(s11_rows), "n_human": len(human_rows)}, ensure_ascii=False, indent=2))
    return {"ab": s11_rows, "human": human_rows}


if __name__ == "__main__":
    build_stage11_audition_and_report()
