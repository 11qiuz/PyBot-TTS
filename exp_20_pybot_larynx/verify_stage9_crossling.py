"""Stage 9 Comprehensive Cross-Lingual Verification & Audition Page Generator (`exp_20_pybot_larynx`)."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.fft import rfftfreq

REPO_ROOT = Path(__file__).resolve().parent.parent
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
from exp_20_pybot_larynx.crossling import (  # noqa: E402
    SpeakerDonorBank,
    build_speaker_anchored_en_control,
    get_en_articulation_table,
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
from exp_v14_fluent_zh_tts.eval_v14 import _rms_norm, _write_pcm16_wav  # noqa: E402
from exp_v15_bznsyp_tts.index_bznsyp_v15 import load_speaker_assets_v15  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity  # noqa: E402
from pybot_webui.crossling import build_utterance_control, build_zh_run_control  # noqa: E402
from pybot_webui.engine import apply_voice_params  # noqa: E402
from pybot_webui.presets import VoiceParams  # noqa: E402

SAMPLES_S9_DIR = EXP_DIR / "samples_stage9_crossling"
RESULTS_S9_PATH = EXP_DIR / "stage9_crossling_results.json"
REPORT_S9_PATH = EXP_DIR / "STAGE9_CROSSLING_REPORT.md"
LISTEN_S9_PATH = EXP_DIR / "listen_stage9_crossling.html"
_FREQS = rfftfreq(N_FFT, 1.0 / SR)


def _active_rms_db(y: np.ndarray) -> float:
    arr = np.asarray(y, dtype=np.float64).ravel()
    pk = float(np.max(np.abs(arr))) if arr.size else 0.0
    if pk <= 1e-9:
        return -120.0
    act = arr[np.abs(arr) > 0.02 * pk]
    rms = float(np.sqrt(np.mean(act * act))) if act.size else float(np.sqrt(np.mean(arr * arr)))
    return 20.0 * math.log10(max(rms, 1e-12))


def _ctrl_metrics(ctrl: Any, y: np.ndarray, rtf: float) -> dict[str, Any]:
    san = check_wav_sanity(y, SR, expected_sr=SR)
    vm = ctrl.voiced & (ctrl.f0_hz > 40.0)
    f0_v = ctrl.f0_hz[vm]
    f0_med = float(np.median(f0_v)) if f0_v.size else 0.0
    f0_iqr = (
        float(12.0 * math.log2(float(np.percentile(f0_v, 75)) / max(40.0, float(np.percentile(f0_v, 25)))))
        if f0_v.size >= 4
        else 0.0
    )
    return {
        "sanity_ok": bool(san.get("ok", False)),
        "duration_s": round(len(y) / float(SR), 3),
        "rtf": round(float(rtf), 4),
        "f0_median_hz": round(f0_med, 1),
        "f0_iqr_st": round(f0_iqr, 2),
        "active_rms_db": round(_active_rms_db(y), 2),
    }


def _synth_utt(
    text: str,
    uid: str,
    vp: VoiceParams,
    kb: AcousticKnowledgeStage7,
    arb: Arbiter,
    rules: list[Any],
    phasor: np.ndarray,
    *,
    base_prof: Any = None,
    tgt_prof: Any = None,
) -> tuple[Any, np.ndarray, float, dict[str, Any]]:
    t0 = time.perf_counter()
    ctrl, meta = build_utterance_control(text, uid, kb, arb, rules, vp)
    eff_phasor = phasor
    if tgt_prof is not None and base_prof is not None:
        ctrl, eff_phasor, _ = apply_clone_to_control(
            ctrl,
            base_prof,
            tgt_prof,
            strength=0.90,
            clone_f0=True,
            clone_glottal=True,
            base_coherent_phasor=phasor,
        )
    mod_ctrl = apply_voice_params(ctrl, vp, vp.seed)
    y = synthesize_larynx_v2(
        mod_ctrl,
        glottal_mode="dynamic_coherent",
        coherent_phasor=eff_phasor,
        uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7,
        subharm_strength=float(meta.get("subharm_strength", 0.0)),
        seed=vp.seed,
    )
    y_out = _rms_norm(y, -19.5)
    elapsed = time.perf_counter() - t0
    rtf = elapsed / max(0.05, len(y_out) / float(SR))
    return mod_ctrl, y_out, rtf, meta


def _render_listen_html(items: list[dict[str, Any]], out_path: Path) -> None:
    payload = json.dumps(items, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-Larynx Stage 9 · 跨语种同一说话人试听台</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#eef1f4;background:#0f141c;max-width:1040px;margin:1.8rem auto;padding:0 1rem}}
h1{{font-size:1.45rem;margin-bottom:.25rem}}
.sub{{color:#94a3b8;margin-bottom:1.1rem}}
.card{{background:#18202c;border:1px solid #293548;padding:1rem 1.2rem;margin:.8rem 0;border-radius:10px}}
.row{{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;padding:.45rem 0;border-bottom:1px dashed #253042}}
.row:last-child{{border-bottom:none}}
audio{{width:min(100%,24rem);height:34px}}
.lbl{{min-width:12rem;font-weight:600;color:#e2e8f0}}
.meta{{font-size:.82rem;color:#93c5fd}}
</style>
</head>
<body>
<h1>PyBot-Larynx Stage 9 · 说话人锚定跨语种试听台</h1>
<p class="sub">说话人声道供体传输 (DSFPT) + 统一跨语种韵律下倾与声门耦合 · Zero-ML</p>
<div id="app" class="card"></div>
<script>
const ITEMS = {payload};
const app = document.getElementById("app");
app.innerHTML = ITEMS.map(it => `
  <div class="row">
    <span class="lbl">${{it.label}}</span>
    <audio controls preload="metadata" src="${{it.wav}}"></audio>
    <span class="meta">${{it.meta}}</span>
  </div>
`).join("");
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    SAMPLES_S9_DIR.mkdir(parents=True, exist_ok=True)
    kb = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
    arb = Arbiter()
    rules = default_stage5_rules()
    assets = load_speaker_assets_v15()
    phasor = np.asarray(assets["glottal_bank"].coherent_phasor, dtype=np.complex128)
    bank = SpeakerDonorBank(kb)
    vp_std = VoiceParams.from_dict({"preset_id": "female_std"})

    # 1. Pure ZH vs Pure EN physical alignment
    zh_txt = "今天四十三位朋友一起测试跨语种语音合成，声音清晰自然。"
    en_txt = "Sixty seven friends test our cross lingual speech synthesis clearly."
    zh_ctrl, zh_y, zh_rtf, _ = _synth_utt(zh_txt, "s9_pure_zh", vp_std, kb, arb, rules, phasor)
    en_ctrl, en_y, en_rtf, _ = _synth_utt(en_txt, "s9_pure_en", vp_std, kb, arb, rules, phasor)

    # Also measure raw un-normalized RMS diff from build_*_run_control
    c_zh_raw, _ = build_zh_run_control(zh_txt, "raw_zh", kb, arb, rules, vp_std)
    en_u = [u for u in _safe_plan_v13(en_txt)["units"] if u.get("lang") == "en"]
    c_en_raw, _ = build_speaker_anchored_en_control(en_u, float(kb.speaker_f0_med_hz), vp_std, kb)
    y_zh_raw = synthesize_larynx_v2(c_zh_raw, glottal_mode="dynamic_coherent", coherent_phasor=phasor, uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7, seed=2026)
    y_en_raw = synthesize_larynx_v2(c_en_raw, glottal_mode="dynamic_coherent", coherent_phasor=phasor, uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7, seed=2026)
    raw_rms_diff_db = abs(_active_rms_db(y_zh_raw) - _active_rms_db(y_en_raw))

    vm_zh = c_zh_raw.voiced & (c_zh_raw.f0_hz > 40.0)
    vm_en = c_en_raw.voiced & (c_en_raw.f0_hz > 40.0)
    v_zh = np.mean(c_zh_raw.log_env[vm_zh], axis=0)
    v_en = np.mean(c_en_raw.log_env[vm_en], axis=0)
    m_all = (_FREQS >= 80.0) & (_FREQS <= 11500.0)
    m_hi = (_FREQS >= 4500.0) & (_FREQS <= 11500.0)
    voiced_ltas_rmse = float(np.sqrt(np.mean(((v_zh - v_en)[m_all] * NEPER_TO_DB) ** 2)))
    high_band_diff = abs(float(np.mean((v_zh - v_en)[m_hi] * NEPER_TO_DB)))

    um_zh = (~c_zh_raw.voiced) & (np.mean(c_zh_raw.log_env[:, _FREQS >= 1500.0], axis=1) > -9.5)
    um_en = (~c_en_raw.voiced) & (np.mean(c_en_raw.log_env[:, _FREQS >= 1500.0], axis=1) > -9.5)
    m_u = _FREQS >= 800.0
    uv_ltas_rmse = float(
        np.sqrt(np.mean(((np.mean(c_zh_raw.log_env[um_zh], 0) - np.mean(c_en_raw.log_env[um_en], 0))[m_u] * NEPER_TO_DB) ** 2))
    )
    bap_mae = float(np.mean(np.abs(np.median(c_zh_raw.bap[vm_zh], 0) - np.median(c_en_raw.bap[vm_en], 0))))
    f0_zh = float(np.median(c_zh_raw.f0_hz[vm_zh]))
    f0_en = float(np.median(c_en_raw.f0_hz[vm_en]))
    f0_diff_st = abs(12.0 * math.log2(f0_en / f0_zh))
    en_iqr_st = float(
        12.0
        * math.log2(
            float(np.percentile(c_en_raw.f0_hz[vm_en], 75)) / max(40.0, float(np.percentile(c_en_raw.f0_hz[vm_en], 25)))
        )
    )

    # 2. English vowel formant distinctiveness check (primary vowel nucleus)
    en_art = get_en_articulation_table()
    prof_ref = neutral_female_reference()
    vowel_fmts: dict[str, list[float]] = {}
    for w in ("beat", "boot", "father", "bird", "bad"):
        wu = [u for u in _safe_plan_v13(w)["units"] if u.get("lang") == "en"]
        tr_w, _ = _build_rhythm_track_stage9b(wu, prof_ref, en_art, use_en_articulation=(en_art is not None))
        wc, _ = build_speaker_anchored_en_control(wu, float(kb.speaker_f0_med_hz), vp_std, kb)
        mid = -1
        cur = 0
        for seg in tr_w.segments:
            nf = max(1, int(round(seg.dur_ms / 5.0)))
            if not seg.is_pause and seg.ipa in VOWEL_IPAS:
                mid = max(0, min(len(wc.f0_hz) - 1, cur - 1 + nf // 2))
                if seg.stress == 1:
                    break
            cur += nf
        if mid < 0:
            wvm = np.where(wc.voiced & (wc.f0_hz > 40.0))[0]
            mid = int(wvm[len(wvm) // 2])
        sf = decompose_frame_source_filter(wc.log_env[mid], order=22)
        hz, _ = lsf_to_formants(sf.lsf, sr=SR, f_min=200.0, f_max=4500.0)
        hz = hz[(hz >= 230.0) & (hz <= 4250.0)]
        vowel_fmts[w] = [round(float(hz[0]), 0), round(float(hz[1]), 0), round(float(hz[2]), 0)]

    # 3. Sample battery (Code-switched, presets, clone, expressive emotions)
    cases: list[tuple[str, str, VoiceParams, Any]] = [
        ("pure_zh", zh_txt, vp_std, None),
        ("pure_en", en_txt, vp_std, None),
        ("mix_tech", "今天用 PyBot 测试 cross lingual speech，表现非常自然。", vp_std, None),
        ("mix_chat", "我觉得这个 benchmark result 比之前自然很多，Really smooth!", vp_std, None),
        ("mix_question", "你确定这个 speaker identity 已经保持完全一致了吗?", vp_std, None),
        ("mix_num_year", "在2026年，PyBot 已经支持中文和 English 无缝切换。", vp_std, None),
        ("mix_short", "Hello 大家好，我是 PyBot!", vp_std, None),
        ("preset_male_young", "欢迎来到 Studio，Hello world and happy coding!", VoiceParams.from_dict({"preset_id": "male_young"}), None),
        ("preset_girl_bright", "早安！Today is a bright and lovely morning!", VoiceParams.from_dict({"preset_id": "girl_bright"}), None),
        ("emo_excited", "[激动] *终于*成功了！Today feels amazing![笑]", VoiceParams.from_dict({"preset_id": "female_std", "emotion_mode": "excited", "emotion_intensity": 0.95}), None),
        ("emo_whisper", "[耳语][吸气] 轻声说一句，Good night and sweet dreams.", VoiceParams.from_dict({"preset_id": "female_std", "emotion_mode": "whisper", "emotion_intensity": 0.95}), None),
    ]

    sample_clone = REPO_ROOT / "exp_v8_data" / "corpus" / "zh" / "SSB0073" / "SSB00730001.wav"
    base_prof = load_base_speaker_profile()
    tgt_prof = None
    if sample_clone.exists():
        w24, _ = decode_wav_bytes_to_24k_mono(sample_clone.read_bytes())
        tgt_prof = extract_clone_profile([w24], name="SSB0073")
        cases.extend([
            ("clone_zh", "欢迎使用跨语种样本模仿功能。", vp_std, tgt_prof),
            ("clone_en", "Welcome to cross lingual voice cloning today.", vp_std, tgt_prof),
            ("clone_mix", "你好朋友，This is my English voice too!", vp_std, tgt_prof),
        ])

    records: dict[str, Any] = {}
    listen_items: list[dict[str, Any]] = []
    rtfs: list[float] = []
    all_ok = True

    for cid, text, vp, clone_p in cases:
        ctrl, y_out, rtf, meta = _synth_utt(
            text, cid, vp, kb, arb, rules, phasor, base_prof=base_prof, tgt_prof=clone_p
        )
        m = _ctrl_metrics(ctrl, y_out, rtf)
        m["text"] = text
        m["n_zh"] = meta["n_zh"]
        m["n_en"] = meta["n_en"]
        records[cid] = m
        rtfs.append(rtf)
        if not m["sanity_ok"]:
            all_ok = False
        wav_rel = f"samples_stage9_crossling/{cid}.wav"
        _write_pcm16_wav(SAMPLES_S9_DIR / f"{cid}.wav", y_out, SR)
        listen_items.append({
            "label": f"{cid} · {vp.preset_id}",
            "wav": wav_rel,
            "meta": f"F0={m['f0_median_hz']}Hz IQR={m['f0_iqr_st']}st Dur={m['duration_s']}s RTF={m['rtf']} ZH={m['n_zh']}/EN={m['n_en']}",
        })

    # Clone ZH vs EN check
    clone_f0_diff_st = 0.0
    if tgt_prof is not None and "clone_zh" in records and "clone_en" in records:
        clone_f0_diff_st = abs(12.0 * math.log2(records["clone_en"]["f0_median_hz"] / max(40.0, records["clone_zh"]["f0_median_hz"])))

    median_rtf = round(float(np.median(rtfs)), 4) if rtfs else 1.0
    gates = {
        "all_sanity_ok": all_ok,
        "median_rtf_le_006": median_rtf <= 0.06,
        "voiced_ltas_rmse_le_4_5": voiced_ltas_rmse <= 4.5,
        "high_band_diff_le_4_5": high_band_diff <= 4.5,
        "uv_ltas_rmse_le_6_5": uv_ltas_rmse <= 6.5,
        "bap6_mae_le_0_08": bap_mae <= 0.08,
        "f0_diff_st_le_1_0": f0_diff_st <= 1.0,
        "en_f0_iqr_ge_2_4": en_iqr_st >= 2.4,
        "raw_rms_diff_le_2_5": raw_rms_diff_db <= 2.5,
        "vowel_f2_i_vs_u_ge_800": (vowel_fmts["beat"][1] - vowel_fmts["boot"][1]) >= 800.0,
        "vowel_f1_a_vs_i_ge_250": (vowel_fmts["father"][0] - vowel_fmts["beat"][0]) >= 250.0,
        "vowel_f3_er_vs_i_le_m400": (vowel_fmts["beat"][2] - vowel_fmts["bird"][2]) >= 400.0,
        "vowel_f2_ae_vs_a_ge_200": (vowel_fmts["bad"][1] - vowel_fmts["father"][1]) >= 200.0,
        "clone_zh_en_f0_diff_le_1_2": clone_f0_diff_st <= 1.2,
    }
    verdict = "GO" if all(gates.values()) else "NO_GO"

    summary = {
        "verdict": verdict,
        "median_rtf": median_rtf,
        "zero_ml_weights": True,
        "speaker_scales": {
            "f1": round(bank.scale_f1, 4),
            "f2": round(bank.scale_f2, 4),
            "f3": round(bank.scale_f3, 4),
        },
        "alignment": {
            "voiced_ltas_rmse_db": round(voiced_ltas_rmse, 2),
            "high_band_4500_11500_diff_db": round(high_band_diff, 2),
            "unvoiced_ltas_rmse_db": round(uv_ltas_rmse, 2),
            "bap6_mae": round(bap_mae, 4),
            "zh_f0_med_hz": round(f0_zh, 1),
            "en_f0_med_hz": round(f0_en, 1),
            "f0_diff_st": round(f0_diff_st, 2),
            "en_f0_iqr_st": round(en_iqr_st, 2),
            "raw_rms_diff_db": round(raw_rms_diff_db, 2),
            "clone_f0_diff_st": round(clone_f0_diff_st, 2),
        },
        "vowel_formants_hz": vowel_fmts,
        "gates": gates,
        "records": records,
    }
    RESULTS_S9_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_listen_html(listen_items, LISTEN_S9_PATH)

    md_lines = [
        "# PyBot-Larynx Stage 9 说话人锚定跨语种验证报告",
        "",
        f"- **判定结论 (`verdict`)**: **`{verdict}`** (`median_rtf={median_rtf}`, `zero_ml_weights=True`, `all_sanity_ok={all_ok}`)",
        f"- **说话人声道比例 (`SpeakerDonorBank`)**: `F1×{bank.scale_f1:.3f}`, `F2×{bank.scale_f2:.3f}`, `F3×{bank.scale_f3:.3f}`",
        "",
        "## 1. Stage 8 诊断基线 vs Stage 9 实测对照（中英同一说话人物理对齐）",
        "",
        "| 物理对齐指标 | Stage 8 旧桥接基线 | Stage 9 实测 | 门禁要求 |",
        "|---|---:|---:|---:|",
        f"| 浊音全频段 LTAS RMSE (80–11500 Hz) | 20.35 dB | **{voiced_ltas_rmse:.2f} dB** | <= 4.50 dB |",
        f"| 浊音高频亮暗差 (4500–11500 Hz) | > 20.0 dB | **{high_band_diff:.2f} dB** | <= 4.50 dB |",
        f"| 清音活跃擦音 LTAS RMSE (800–11500 Hz) | 23.19 dB | **{uv_ltas_rmse:.2f} dB** | <= 6.50 dB |",
        f"| 浊音 6 频段非周期度 BAP6 MAE | > 0.08 | **{bap_mae:.4f}** | <= 0.0800 |",
        f"| 中英文 F0 中位差 (`|ΔF0|`) | 1.81 st | **{f0_diff_st:.2f} st** (`{f0_zh:.1f}` vs `{f0_en:.1f}` Hz) | <= 1.00 st |",
        f"| 英文句内 F0 起伏 IQR | 1.12 st | **{en_iqr_st:.2f} st** | >= 2.40 st |",
        f"| 中英文原始活跃响度差 | - | **{raw_rms_diff_db:.2f} dB** | <= 2.50 dB |",
        f"| 声音模仿(SSB0073)跨语种 F0 漂移 | - | **{clone_f0_diff_st:.2f} st** | <= 1.20 st |",
        "",
        "## 2. 英语核心元音音质区分度实测 (`LSF-22` F1/F2/F3)",
        "",
        "| 测试词 / 元音 | F1 (Hz) | F2 (Hz) | F3 (Hz) |",
        "|---|---:|---:|---:|",
    ]
    for w, label in (("beat", "beat /i/"), ("boot", "boot /u/"), ("father", "father /ɑ/"), ("bird", "bird /ɝ/"), ("bad", "bad /æ/")):
        f = vowel_fmts[w]
        md_lines.append(f"| `{label}` | {f[0]:.0f} | {f[1]:.0f} | {f[2]:.0f} |")

    md_lines.extend([
        "",
        "## 3. 混合句、音色预设、情绪与克隆样本列表",
        "",
        "| 样本 ID | 音高 F0 (Hz) | F0 IQR (st) | 时长 (s) | CPU RTF | ZH/EN 词数 | Sanity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for cid, m in records.items():
        md_lines.append(
            f"| `{cid}` | {m['f0_median_hz']:.1f} | {m['f0_iqr_st']:.2f} | {m['duration_s']:.2f} | {m['rtf']:.4f} | {m['n_zh']}/{m['n_en']} | {'OK' if m['sanity_ok'] else 'FAIL'} |"
        )
    md_lines.extend([
        "",
        "## 4. 离线对比试听台",
        "- [listen_stage9_crossling.html](listen_stage9_crossling.html)（含纯中/纯英对照、5 句中英混读、预设男声/女声、情绪演播及 `SSB0073` 跨语种克隆样本 WAV）",
    ])
    REPORT_S9_PATH.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({"verdict": verdict, "median_rtf": median_rtf, "alignment": summary["alignment"], "gates": gates}, ensure_ascii=False, indent=2))
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
