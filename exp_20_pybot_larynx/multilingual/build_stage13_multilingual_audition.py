"""Stage 13 Native Zero-Mapping Multilingual Audition, Held-Out Benchmark, HTML & Report."""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import EXP_DIR, SR
from exp_20_pybot_larynx.clone import VoiceCloneProfile, load_base_speaker_profile
from exp_20_pybot_larynx.multilingual.knowledge_ipa_native import (
    SF_NATIVE_KB_PATH,
    SF_NATIVE_SUMMARY_PATH,
    get_sf_native_multilingual_kb,
)
from exp_20_pybot_larynx.multilingual.renderer import synthesize_multilingual_native
from exp_20_pybot_larynx.verify_stage10_bilingual import compute_forensic_metrics
from exp_v5_eval.metrics import check_wav_sanity, compute_spectral_distances
from exp_v14_fluent_zh_tts.eval_v14 import _rms_norm, _write_pcm16_wav
from exp_v15_bznsyp_tts.index_bznsyp_v15 import load_speaker_assets_v15
from pybot_webui.presets import VoiceParams

SAMPLES_S13_DIR = EXP_DIR / "samples_stage13_multilingual"
SAMPLES_S12_DIR = EXP_DIR / "samples_stage12_bilingual"
HELDOUT_PKL = EXP_DIR / "crossling_data" / "sf_bilingual_heldout.pkl"
S12_RESULTS_JSON = EXP_DIR / "stage12_sf_bilingual_results.json"

HTML_PATH = EXP_DIR / "listen_stage13_multilingual.html"
REPORT_PATH = EXP_DIR / "STAGE13_NATIVE_MULTILINGUAL_REPORT.md"
RESULTS_JSON_PATH = EXP_DIR / "stage13_multilingual_results.json"

TARGET_HELDOUT_IDS = (
    "com_SF_ce996",
    "com_SF_ce997",
    "com_SF_ce998",
    "com_SF_ce999",
    "com_SF_ce1000",
    "com_SF_ce1001",
)


def _crest_db(y: np.ndarray) -> float:
    arr = np.asarray(y, dtype=np.float64).ravel()
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    rms = float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0
    return round(20.0 * math.log10((peak + 1e-12) / (rms + 1e-12)), 3)


def build_stage13_audition_and_report() -> dict[str, Any]:
    kb = get_sf_native_multilingual_kb(SF_NATIVE_KB_PATH)
    assert kb is not None, f"Stage 13 KB not found at {SF_NATIVE_KB_PATH}"
    SAMPLES_S13_DIR.mkdir(parents=True, exist_ok=True)

    kb_summary = json.loads(SF_NATIVE_SUMMARY_PATH.read_text(encoding="utf-8")) if SF_NATIVE_SUMMARY_PATH.is_file() else {}
    s12_data = json.loads(S12_RESULTS_JSON.read_text(encoding="utf-8")) if S12_RESULTS_JSON.is_file() else {}
    s12_ho_map = {r["id"]: r for r in s12_data.get("heldout_rows", [])}
    s12_sc_map = {r["id"]: r for r in s12_data.get("showcase_rows", [])}

    assets = load_speaker_assets_v15()
    bz_phasor = np.asarray(assets["glottal_bank"].coherent_phasor, dtype=np.complex128)
    bz_prof = load_base_speaker_profile()

    vp_std = VoiceParams.from_dict({"preset_id": "female_std"})
    vp_exc = VoiceParams.from_dict({"preset_id": "female_std", "emotion_mode": "excited", "emotion_intensity": 0.90})
    vp_sad = VoiceParams.from_dict({"preset_id": "female_std", "emotion_mode": "sad", "emotion_intensity": 0.90})

    def _static_metrics(ctrl: Any, meta: dict[str, Any]) -> dict[str, Any]:
        vm = np.asarray(ctrl.voiced, dtype=bool) & (np.asarray(ctrl.f0_hz, dtype=np.float64) > 40.0)
        max_vb0 = round(float(np.max(ctrl.bap[vm, 0])), 3) if np.any(vm) else 0.0
        max_vb1 = round(float(np.max(ctrl.bap[vm, 1])), 3) if np.any(vm) else 0.0
        sil_m = (~np.asarray(ctrl.voiced, dtype=bool)) & (np.max(ctrl.log_env, axis=1) <= -14.0)
        max_sil_db = round(float(np.max(ctrl.log_env[sil_m]) * 8.685889638065035), 2) if np.any(sil_m) else -139.0
        n_sh = sum(int(d.get("hits", {}).get("ipa_shared", 0)) for d in meta.get("dp_stats", []))
        return {
            "max_voiced_bap0": max_vb0,
            "max_voiced_bap1": max_vb1,
            "max_pause_db": max_sil_db,
            "ipa_shared_fallbacks": n_sh,
        }

    # 1. Held-out unseen SF_bilingual benchmark (6 utterances)
    with open(HELDOUT_PKL, "rb") as f:
        held_doc = pickle.load(f)
    held_utts = held_doc.get("utterances", [])
    by_id = {str(u["id"]): u for u in held_utts}
    selected_ho = [by_id[uid] for uid in TARGET_HELDOUT_IDS if uid in by_id]

    heldout_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(selected_ho):
        uid = str(item["id"])
        txt = str(item["text"])
        ref_wav = _rms_norm(np.asarray(item["wav"], dtype=np.float64), -19.5)

        t0 = time.perf_counter()
        y_s13, _c13, meta13 = synthesize_multilingual_native(
            txt, knowledge=kb, params=vp_std, seed=2026 + idx
        )
        rtf_13 = (time.perf_counter() - t0) / max(0.05, len(y_s13) / float(SR))

        p_hum = SAMPLES_S13_DIR / f"human_{uid}.wav"
        p_s13 = SAMPLES_S13_DIR / f"s13_native_{uid}.wav"
        _write_pcm16_wav(p_hum, ref_wav, SR)
        _write_pcm16_wav(p_s13, y_s13, SR)

        sd_13 = compute_spectral_distances(ref_wav, y_s13, sr=SR) or {}
        m_hum = compute_forensic_metrics(ref_wav, SR)
        m_13 = compute_forensic_metrics(y_s13, SR)
        san_13 = bool(check_wav_sanity(y_s13, SR, expected_sr=SR).get("ok", False))

        s12_row = s12_ho_map.get(uid, {})
        s12_sf_rel = f"samples_stage12_bilingual/s12_sf_{uid}.wav"
        if not (SAMPLES_S12_DIR / f"s12_sf_{uid}.wav").is_file():
            s12_sf_rel = None

        heldout_rows.append({
            "id": uid,
            "text": txt,
            "human": {**m_hum, "crest_db": _crest_db(ref_wav)},
            "s12_sf": s12_row.get("s12_sf", {}),
            "s13_native": {
                **m_13,
                "crest_db": _crest_db(y_s13),
                "mel_cd_db": round(float(sd_13.get("mel_lsd_db") or 0.0), 3),
                "rtf": round(rtf_13, 4),
                "sanity_ok": san_13,
                "projections": meta13.get("projections", 0),
                "dp_stats": meta13.get("dp_stats", []),
            },
            "wav_human": f"samples_stage13_multilingual/human_{uid}.wav",
            "wav_s12_sf": s12_sf_rel,
            "wav_s13_native": f"samples_stage13_multilingual/s13_native_{uid}.wav",
        })

    # 2. Core code-switching (5) + Pure EN (3) + Pure ZH (2) + Emotional mixed (1)
    bench_cases: list[tuple[str, str, VoiceParams, str]] = [
        ("mix_en_zh_vowel", "Let me know 你今天几点出发。", vp_std, "mixed"),
        ("mix_en_zh_nasal", "我选了 green tea 按你的清单买的。", vp_std, "mixed"),
        ("mix_tech", "今天用 PyBot 测试 this smooth cross lingual speech，表现非常自然。", vp_std, "mixed"),
        ("mix_multi", "先打一个 tag，then run the nightly build，最后把 log 发给我。", vp_std, "mixed"),
        ("mix_excited", "[激动] *终于*成功了！This feels amazing![笑]", vp_exc, "mixed_emo"),
        ("en_studio", "This is a smooth benchmark of bilingual speech synthesis in our studio.", vp_std, "pure_en"),
        ("en_stops", "Pat took ten bright copper pots back to the deep dark cellar.", vp_std, "pure_en"),
        ("en_story", "She walked carefully through the quiet autumn forest before sunrise.", vp_std, "pure_en"),
        ("en_excited", "[激动] *Finally* it worked! This sounds *so* much better!", vp_exc, "pure_en_emo"),
        ("en_sad", "[悲伤] I waited until sunset... and nobody came back.", vp_sad, "pure_en_emo"),
        ("zh_news", "今日早间消息，新一代国产开源语音引擎已经正式发布。", vp_std, "pure_zh"),
        ("zh_chat", "我们待会儿一起去楼下喝杯热咖啡，顺便把方案对一下。", vp_std, "pure_zh"),
    ]

    showcase_rows: list[dict[str, Any]] = []
    for idx, (cid, txt, vpi, kind) in enumerate(bench_cases):
        t0 = time.perf_counter()
        y_13, c13, meta13 = synthesize_multilingual_native(
            txt, knowledge=kb, params=vpi, seed=2026 + idx
        )
        rtf_13 = (time.perf_counter() - t0) / max(0.05, len(y_13) / float(SR))
        _write_pcm16_wav(SAMPLES_S13_DIR / f"s13_native_{cid}.wav", y_13, SR)

        t1 = time.perf_counter()
        y_clone, _cc, meta_cl = synthesize_multilingual_native(
            txt,
            knowledge=kb,
            params=vpi,
            target_profile=bz_prof,
            target_coherent_phasor=bz_phasor,
            seed=2026 + idx,
        )
        rtf_cl = (time.perf_counter() - t1) / max(0.05, len(y_clone) / float(SR))
        _write_pcm16_wav(SAMPLES_S13_DIR / f"s13_clone_bz_{cid}.wav", y_clone, SR)

        m_13 = compute_forensic_metrics(y_13, SR)
        m_cl = compute_forensic_metrics(y_clone, SR)
        san_13 = bool(check_wav_sanity(y_13, SR, expected_sr=SR).get("ok", False))
        san_cl = bool(check_wav_sanity(y_clone, SR, expected_sr=SR).get("ok", False))

        s12_row = s12_sc_map.get(cid, {})
        wav_s12_sf = f"samples_stage12_bilingual/s12_sf_{cid}.wav" if (SAMPLES_S12_DIR / f"s12_sf_{cid}.wav").is_file() else None

        showcase_rows.append({
            "id": cid,
            "kind": kind,
            "text": txt,
            "s12_sf": s12_row.get("s12_sf", {}),
            "s13_native": {
                **m_13,
                **_static_metrics(c13, meta13),
                "crest_db": _crest_db(y_13),
                "rtf": round(rtf_13, 4),
                "sanity_ok": san_13,
                "projections": meta13.get("projections", 0),
                "dp_stats": meta13.get("dp_stats", []),
            },
            "s13_clone_bz": {
                **m_cl,
                "crest_db": _crest_db(y_clone),
                "rtf": round(rtf_cl, 4),
                "sanity_ok": san_cl,
                "projections": meta_cl.get("projections", 1),
            },
            "wav_s12_sf": wav_s12_sf,
            "wav_s13_native": f"samples_stage13_multilingual/s13_native_{cid}.wav",
            "wav_s13_clone_bz": f"samples_stage13_multilingual/s13_clone_bz_{cid}.wav",
        })

    # Aggregate statistics
    ho_mel_13 = round(float(np.median([r["s13_native"]["mel_cd_db"] for r in heldout_rows])), 3)
    ho_hnr_13 = round(float(np.median([r["s13_native"]["hnr_db"] for r in heldout_rows])), 3)
    ho_hnr_hum = round(float(np.median([r["human"]["hnr_db"] for r in heldout_rows])), 3)
    ho_s12_mels = [r["s12_sf"].get("mel_cd_db") for r in heldout_rows if isinstance(r.get("s12_sf"), dict) and r["s12_sf"].get("mel_cd_db") is not None]
    ho_mel_12 = round(float(np.median(ho_s12_mels)), 3) if ho_s12_mels else None

    mix_rows = [r for r in showcase_rows if r["kind"] in ("mixed", "mixed_emo")]
    en_rows = [r for r in showcase_rows if r["kind"] == "pure_en"]
    zh_rows = [r for r in showcase_rows if r["kind"] == "pure_zh"]
    all_san = all(r["s13_native"]["sanity_ok"] for r in heldout_rows) and all(
        r["s13_native"]["sanity_ok"] and r["s13_clone_bz"]["sanity_ok"] for r in showcase_rows
    )

    # DP tier hit breakdown across all synthesized runs
    tier_totals: dict[str, int] = {}
    for row in heldout_rows + showcase_rows:
        for dp_m in row["s13_native"].get("dp_stats", []):
            for tk, tv in dp_m.get("hits", {}).items():
                if tk in ("ctx_pns", "ctx_pn", "ctx_ps", "ctx_p", "ctx_base", "ipa_shared", "fallback"):
                    tier_totals[tk] = tier_totals.get(tk, 0) + int(tv)
    tot_tok = max(1, sum(tier_totals.values()))
    hit_rates = {k: round(100.0 * v / float(tot_tok), 2) for k, v in sorted(tier_totals.items())}

    en_emo_rows = [r for r in showcase_rows if r["kind"] == "pure_en_emo"]
    max_vb0_all = max(r["s13_native"].get("max_voiced_bap0", 0.0) for r in showcase_rows)
    max_vb1_all = max(r["s13_native"].get("max_voiced_bap1", 0.0) for r in showcase_rows)
    en_studio_sh = next((r["s13_native"].get("ipa_shared_fallbacks", -1) for r in showcase_rows if r["id"] == "en_studio"), -1)

    summary = {
        "stage": "stage13_native_multilingual_zero_mapping",
        "ml_weights_mb": 0.0,
        "all_sanity_ok": bool(all_san),
        "kb_summary": kb_summary,
        "tier_totals": tier_totals,
        "tier_rates_pct": hit_rates,
        "static_telemetry": {
            "max_voiced_bap0": max_vb0_all,
            "max_voiced_bap1": max_vb1_all,
            "en_studio_ipa_shared_fallbacks": en_studio_sh,
        },
        "heldout_summary": {
            "n_utterances": len(heldout_rows),
            "human_hnr_med_db": ho_hnr_hum,
            "s12_sf_mel_cd_med_db": ho_mel_12,
            "s13_native_mel_cd_med_db": ho_mel_13,
            "s13_native_hnr_med_db": ho_hnr_13,
        },
        "showcase_summary": {
            "mixed_hnr_med_db": round(float(np.median([r["s13_native"]["hnr_db"] for r in mix_rows])), 3),
            "mixed_q8_med": round(float(np.median([r["s13_native"]["q8_24"] for r in mix_rows])), 3),
            "mixed_dyn_med_db": round(float(np.median([r["s13_native"]["dyn_p90_20_db"] for r in mix_rows])), 3),
            "pure_en_hnr_med_db": round(float(np.median([r["s13_native"]["hnr_db"] for r in en_rows])), 3),
            "pure_en_q8_med": round(float(np.median([r["s13_native"]["q8_24"] for r in en_rows])), 3),
            "pure_en_f0_iqr_med_st": round(float(np.median([r["s13_native"]["f0_iqr_st"] for r in en_rows])), 3),
            "pure_en_emo_hnr_med_db": round(float(np.median([r["s13_native"]["hnr_db"] for r in en_emo_rows])), 3) if en_emo_rows else None,
            "pure_en_emo_f0_iqr_med_st": round(float(np.median([r["s13_native"]["f0_iqr_st"] for r in en_emo_rows])), 3) if en_emo_rows else None,
            "pure_zh_hnr_med_db": round(float(np.median([r["s13_native"]["hnr_db"] for r in zh_rows])), 3),
            "rtf_med": round(float(np.median([r["s13_native"]["rtf"] for r in showcase_rows])), 4),
        },
        "heldout_rows": heldout_rows,
        "showcase_rows": showcase_rows,
    }
    RESULTS_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Render HTML bench
    payload_json = json.dumps({"heldout": heldout_rows, "showcase": showcase_rows}, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-Larynx Stage 13 · 原生跨语种零映射架构 (SF Zero-Mapping) 试听台</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#eef1f4;background:#0f141c;max-width:1280px;margin:1.8rem auto;padding:0 1rem}}
h1{{font-size:1.45rem;margin-bottom:.25rem}}
h2{{font-size:1.12rem;margin:1.3rem 0 .5rem;color:#7dd3fc}}
.sub{{color:#94a3b8;margin-bottom:1rem}}
.card{{background:#18202c;border:1px solid #293548;padding:1rem 1.2rem;margin:.8rem 0;border-radius:10px}}
.row3{{display:grid;grid-template-columns:19rem 1fr 1fr 1fr;gap:.8rem;align-items:center;padding:.65rem 0;border-bottom:1px dashed #253042}}
audio{{width:100%;height:32px}}
audio.missing{{opacity:.35}}
.lbl{{font-weight:600;color:#e2e8f0}}
.tag{{display:inline-block;font-size:.72rem;padding:.05rem .4rem;border-radius:4px;background:#1e3a5f;color:#93c5fd;margin-left:.35rem}}
.meta{{font-size:.77rem;color:#93c5fd;margin-top:.15rem}}
.txt{{font-size:.84rem;color:#cbd5e1;margin-top:.2rem}}
@media(max-width:980px){{.row3{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>PyBot-Larynx Stage 13 · 统一 IPA 原生双语声学库 (零映射直出)</h1>
<p class="sub">单人统一声学库 <code>knowledge_sf_native_stage13.npz</code> ({kb_summary.get("npz_size_mb", "?")} MB, 0.0 MB ML)：中英统一于同一发声物理轨迹，跨语种零投影，留出真声 Mel-CD 中位数 <b>{ho_mel_13} dB</b>。</p>

<h2>1. 未见留出真人混合句 (com_SF_ce996..ce1001): 真人原声 vs Stage 12 SF映射版 vs Stage 13 原生零映射</h2>
<div id="heldout" class="card"></div>

<h2>2. 核心场景对比 (Stage 12 SF vs Stage 13 原生本尊音色 vs Stage 13 单出口克隆到标贝)</h2>
<div id="showcase" class="card"></div>

<script>
const DATA = {payload_json};
function slot(title, src) {{
  if (!src) return `<div><div class="meta">${{title}} (无旧样本)</div><audio class="missing" controls></audio></div>`;
  return `<div><div class="meta">${{title}}</div><audio controls preload="metadata" src="${{src}}"></audio></div>`;
}}
document.getElementById("heldout").innerHTML = DATA.heldout.map(r => `
  <div class="row3">
    <div>
      <div class="lbl">${{r.id}}</div>
      <div class="txt">${{r.text}}</div>
      <div class="meta">Mel-CD: S12=${{r.s12_sf && r.s12_sf.mel_cd_db || "?"}}dB | S13原生=${{r.s13_native.mel_cd_db}}dB<br>HNR: 人=${{r.human.hnr_db}} | S13=${{r.s13_native.hnr_db}} dB | 投影=${{r.s13_native.projections}}次</div>
    </div>
    ${{slot("Real Human (SF 留出真声)", r.wav_human)}}
    ${{slot("Stage 12 · SF(两次投影版)", r.wav_s12_sf)}}
    ${{slot("Stage 13 · 原生零映射(SF直出)", r.wav_s13_native)}}
  </div>
`).join("");
document.getElementById("showcase").innerHTML = DATA.showcase.map(r => `
  <div class="row3">
    <div>
      <div class="lbl">${{r.id}}<span class="tag">${{r.kind}}</span></div>
      <div class="txt">${{r.text}}</div>
      <div class="meta">S13原生: q8=${{r.s13_native.q8_24}} HNR=${{r.s13_native.hnr_db}} dyn=${{r.s13_native.dyn_p90_20_db}} RTF=${{r.s13_native.rtf}}<br>S13标贝克隆: q8=${{r.s13_clone_bz.q8_24}} HNR=${{r.s13_clone_bz.hnr_db}} (单出口投影=1次)</div>
    </div>
    ${{slot("Stage 12 · SF版", r.wav_s12_sf)}}
    ${{slot("Stage 13 · 原生 SF 直出 (0次投影)", r.wav_s13_native)}}
    ${{slot("Stage 13 · 单出口克隆至 BZNSYP", r.wav_s13_clone_bz)}}
  </div>
`).join("");
</script>
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")

    # Render Markdown Report
    md = [
        "# Stage 13 Native Multilingual Zero-Mapping Report",
        "",
        f"- **Artifact**: `knowledge_sf_native_stage13.npz` (`{kb_summary.get('npz_size_mb', '?')} MB`, `ml_weights_mb=0.0`, `allow_pickle=False`)",
        f"- **Inventory**: `{kb_summary.get('n_zh_tokens', '?')}` native ZH tokens (`39` IPAs) + `{kb_summary.get('n_en_tokens', '?')}` EN tokens (`+ {kb_summary.get('n_en_studio_warped_tokens', kb_summary.get('n_en_backstop_tokens', 0))}` studio EN->EN LJSpeech->SF_EN)",
        f"- **Static Telemetry**: `max_voiced_bap0={max_vb0_all}`, `max_voiced_bap1={max_vb1_all}`, `en_studio_ipa_shared_fallbacks={en_studio_sh}`",
        f"- **Held-out 6 utterances (`com_SF_ce996..1001`)**: S13 Native Mel-CD median = **{ho_mel_13} dB** (Stage 12 SF = {ho_mel_12} dB), HNR median = **{ho_hnr_13} dB** (Human = {ho_hnr_hum} dB)",
        f"- **Showcase medians**: Mixed HNR=`{summary['showcase_summary']['mixed_hnr_med_db']} dB`, Pure-EN HNR=`{summary['showcase_summary']['pure_en_hnr_med_db']} dB`, Pure-ZH HNR=`{summary['showcase_summary']['pure_zh_hnr_med_db']} dB`, RTF=`{summary['showcase_summary']['rtf_med']}`, All Sanity OK=`{all_san}`",
        f"- **DP lookup tier distribution**: `{hit_rates}`",
        "",
        "| Utterance | Stage 12 SF Mel-CD (dB) | Stage 13 Native Mel-CD (dB) | Human HNR (dB) | Stage 13 HNR (dB) | RTF |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in heldout_rows:
        s12m = r["s12_sf"].get("mel_cd_db", "?") if isinstance(r.get("s12_sf"), dict) else "?"
        md.append(
            f"| `{r['id']}` | {s12m} | {r['s13_native']['mel_cd_db']} | {r['human']['hnr_db']} | {r['s13_native']['hnr_db']} | {r['s13_native']['rtf']} |"
        )
    REPORT_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = build_stage13_audition_and_report()
    print(json.dumps({
        "all_sanity_ok": out["all_sanity_ok"],
        "heldout_summary": out["heldout_summary"],
        "showcase_summary": out["showcase_summary"],
        "tier_rates_pct": out["tier_rates_pct"],
    }, ensure_ascii=False, indent=2))
