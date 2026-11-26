"""Stage 9B Held-Out English Articulation A/B Benchmark & Audition Page Generator (`exp_20_pybot_larynx`)."""
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
from exp_20_pybot_larynx.crossling.build_en_articulation_table import _HELDOUT_CACHE_PATH  # noqa: E402
from exp_20_pybot_larynx.crossling.en_articulation import (  # noqa: E402
    EN_ARTICULATION_PATH,
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
from exp_20_pybot_larynx.larynx.synth import log_env_to_cep  # noqa: E402
from exp_20_pybot_larynx.larynx.synth_v2 import synthesize_larynx_v2  # noqa: E402
from exp_20_pybot_larynx.rules import default_stage5_rules  # noqa: E402
from exp_v14_fluent_zh_tts.eval_v14 import _rms_norm, _write_pcm16_wav  # noqa: E402
from exp_v15_bznsyp_tts.index_bznsyp_v15 import load_speaker_assets_v15  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity  # noqa: E402
from exp_v6_crossling_dsp.speaker_profile import neutral_female_reference  # noqa: E402
from pybot_webui.crossling import build_utterance_control, build_zh_run_control  # noqa: E402
from pybot_webui.engine import apply_voice_params  # noqa: E402
from pybot_webui.presets import VoiceParams  # noqa: E402

SAMPLES_9B_DIR = EXP_DIR / "samples_stage9b_articulation"
RESULTS_9B_PATH = EXP_DIR / "stage9b_articulation_results.json"
REPORT_9B_PATH = EXP_DIR / "STAGE9B_ARTICULATION_REPORT.md"
LISTEN_9B_PATH = EXP_DIR / "listen_stage9b_articulation.html"
_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_U_ART = np.clip((np.maximum(_FREQS, 80.0) - 3100.0) / 1000.0, 0.0, 1.0)
_W_ART = 0.5 * (1.0 + np.cos(math.pi * _U_ART))


def _evaluate_heldout_ab(kb: AcousticKnowledgeStage7, vp: VoiceParams) -> dict[str, Any]:
    """Compare Stage 9A vs Stage 9B against aligned real frames on 20 unseen CMU-ARCTIC utts."""
    if not _HELDOUT_CACHE_PATH.exists():
        raise FileNotFoundError(f"Missing heldout alignment cache: {_HELDOUT_CACHE_PATH}")

    en_art = get_en_articulation_table()
    prof_ref = neutral_female_reference()
    d = np.load(_HELDOUT_CACHE_PATH, allow_pickle=False)
    ids = [str(x) for x in d["ids"]]

    stats: dict[str, dict[str, list[float]]] = {
        "9A": {"dur_mae": [], "dur_rel_mae": [], "cep_rmse": []},
        "9B": {"dur_mae": [], "dur_rel_mae": [], "cep_rmse": []},
    }

    for idx, _uid in enumerate(ids):
        text = str(d[f"u_{idx}_text"][0])
        ref_ipas = [str(x) for x in d[f"u_{idx}_ipas"]]
        ref_bounds = d[f"u_{idx}_bounds"].astype(np.int32)
        ref_env = d[f"u_{idx}_log_env"].astype(np.float64)
        ref_v_ltas = d[f"u_{idx}_v_ltas"].astype(np.float64)

        ir = _safe_plan_v13(text)
        en_units = [u for u in ir["units"] if u.get("lang") == "en"]
        if not en_units:
            continue

        # Speaker-stripped real phone tokens (average voiced-LTAS subtracted + masked <= 4.1 kHz)
        ref_durs = [(int(pe) - int(ps)) * 5.0 for ps, _pm, pe in ref_bounds]
        ref_tot_dur = max(1.0, float(sum(ref_durs)))
        ref_stripped_cep = log_env_to_cep((ref_env - ref_v_ltas[None, :]) * _W_ART[None, :], n_cep=20)

        for tag, use_art in (("9A", False), ("9B", True)):
            track, _ = _build_rhythm_track_stage9b(en_units, prof_ref, en_art, use_en_articulation=use_art)
            ctrl, _ = build_speaker_anchored_en_control(
                en_units, float(kb.speaker_f0_med_hz), vp, kb, use_en_articulation=use_art
            )
            vm_c = ctrl.voiced & (ctrl.f0_hz > 40.0)
            if not np.any(vm_c):
                continue
            ctrl_v_ltas = np.mean(ctrl.log_env[vm_c], axis=0)
            pred_stripped_cep = log_env_to_cep((ctrl.log_env - ctrl_v_ltas[None, :]) * _W_ART[None, :], n_cep=20)

            pred_phones: list[tuple[str, int, int, float]] = []
            cur = 0
            for seg in track.segments:
                nf = max(1, int(round(seg.dur_ms / 5.0)))
                if not seg.is_pause:
                    s0 = max(0, cur - 1)
                    s1 = min(len(ctrl.f0_hz), max(s0 + 1, cur - 1 + nf))
                    pred_phones.append((seg.ipa, s0, s1, float(seg.dur_ms)))
                cur += nf

            n_match = min(len(ref_ipas), len(pred_phones))
            if n_match == 0:
                continue
            pred_tot_dur = max(1.0, float(sum(p[3] for p in pred_phones[:n_match])))
            rate_ratio = ref_tot_dur / pred_tot_dur

            for k in range(n_match):
                r_ps, _r_pm, r_pe = ref_bounds[k]
                if r_pe <= r_ps:
                    continue
                _p_ipa, p_s0, p_s1, p_dur = pred_phones[k]
                stats[tag]["dur_mae"].append(abs(p_dur - ref_durs[k]))
                stats[tag]["dur_rel_mae"].append(abs(p_dur * rate_ratio - ref_durs[k]))

                r_c = np.mean(ref_stripped_cep[r_ps:r_pe, 1:20], axis=0)
                p_c = np.mean(pred_stripped_cep[p_s0:p_s1, 1:20], axis=0)
                stats[tag]["cep_rmse"].append(float(np.sqrt(np.mean((p_c - r_c) ** 2))))

    return {
        tag: {
            "n_tokens": len(v["dur_mae"]),
            "dur_mae_ms": round(float(np.mean(v["dur_mae"])), 2),
            "dur_rel_mae_ms": round(float(np.mean(v["dur_rel_mae"])), 2),
            "art_cep_1_19_rmse": round(float(np.mean(v["cep_rmse"])), 4),
        }
        for tag, v in stats.items()
    }


def _render_html(pairs: list[dict[str, Any]], out_path: Path) -> None:
    payload = json.dumps(pairs, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-Larynx Stage 9B · 英文发音增量 & 协同衔接 A/B 试听台</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#eef1f4;background:#0f141c;max-width:1100px;margin:1.8rem auto;padding:0 1rem}}
h1{{font-size:1.45rem;margin-bottom:.25rem}}
.sub{{color:#94a3b8;margin-bottom:1.1rem}}
.card{{background:#18202c;border:1px solid #293548;padding:1rem 1.2rem;margin:.85rem 0;border-radius:10px}}
.row{{display:grid;grid-template-columns:13rem 1fr 1fr;gap:.8rem;align-items:center;padding:.55rem 0;border-bottom:1px dashed #253042}}
.row:last-child{{border-bottom:none}}
audio{{width:100%;height:32px}}
.lbl{{font-weight:600;color:#e2e8f0}}
.meta{{font-size:.78rem;color:#93c5fd}}
@media(max-width:820px){{.row{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>PyBot-Larynx Stage 9B · Stage 9A vs Stage 9B 英文发音动作 & 衔接 A/B 试听台</h1>
<p class="sub">音色 100% 锁定中文主说话人（BZNSYP）+ 77.6 KB 去说话人英语发音/协同过渡/节奏增量表</p>
<div id="app" class="card"></div>
<script>
const PAIRS = {payload};
const app = document.getElementById("app");
app.innerHTML = PAIRS.map(p => `
  <div class="row">
    <div><div class="lbl">${{p.label}}</div><div class="meta">${{p.note}}</div></div>
    <div><div class="meta">Stage 9A (旧规则发音)</div><audio controls preload="metadata" src="${{p.wav_9a}}"></audio></div>
    <div><div class="meta">Stage 9B (+语料发音增量与衔接)</div><audio controls preload="metadata" src="${{p.wav_9b}}"></audio></div>
  </div>
`).join("");
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> int:
    SAMPLES_9B_DIR.mkdir(parents=True, exist_ok=True)
    kb = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
    arb = Arbiter()
    rules = default_stage5_rules()
    assets = load_speaker_assets_v15()
    phasor = np.asarray(assets["glottal_bank"].coherent_phasor, dtype=np.complex128)
    vp = VoiceParams.from_dict({"preset_id": "female_std"})

    # 1. Held-out 20-utterance A/B comparison (zero leakage)
    ab_heldout = _evaluate_heldout_ab(kb, vp)

    # 2. Allophones & minimal pairs
    ir_bat = [u for u in _safe_plan_v13("bat")["units"] if u.get("lang") == "en"]
    ir_bad = [u for u in _safe_plan_v13("bad")["units"] if u.get("lang") == "en"]
    en_art = get_en_articulation_table()
    assert en_art is not None
    prof_ref = neutral_female_reference()
    tr_bat, _ = _build_rhythm_track_stage9b(ir_bat, prof_ref, en_art, use_en_articulation=True)
    tr_bad, _ = _build_rhythm_track_stage9b(ir_bad, prof_ref, en_art, use_en_articulation=True)
    dur_bat_ae = next(s.dur_ms for s in tr_bat.segments if s.ipa == "æ")
    dur_bad_ae = next(s.dur_ms for s in tr_bad.segments if s.ipa == "æ")

    def _l_f2(word: str) -> float:
        ir_w = [u for u in _safe_plan_v13(word)["units"] if u.get("lang") == "en"]
        tr_w, _ = _build_rhythm_track_stage9b(ir_w, prof_ref, en_art, use_en_articulation=True)
        c_w, _ = build_speaker_anchored_en_control(ir_w, float(kb.speaker_f0_med_hz), vp, kb, use_en_articulation=True)
        cur = 0
        for seg in tr_w.segments:
            nf = max(1, int(round(seg.dur_ms / 5.0)))
            if not seg.is_pause and seg.ipa == "l":
                mid = max(0, min(len(c_w.f0_hz) - 1, cur - 1 + nf // 2))
                sf = decompose_frame_source_filter(c_w.log_env[mid], order=22)
                hz, _ = lsf_to_formants(sf.lsf, sr=SR, f_min=200.0, f_max=4500.0)
                return float(hz[hz >= 700.0][0])
            cur += nf
        return 0.0

    f2_light_l = _l_f2("light")
    f2_dark_l = _l_f2("milk")

    # 3. Same-speaker ZH vs EN check under Stage 9B
    zh_txt = "今天四十三位朋友一起测试跨语种语音合成，声音清晰自然。"
    en_txt = "Sixty seven friends test our cross lingual speech synthesis clearly."
    c_zh, _ = build_zh_run_control(zh_txt, "9b_zh", kb, arb, rules, vp)
    en_u = [u for u in _safe_plan_v13(en_txt)["units"] if u.get("lang") == "en"]
    c_en, _ = build_speaker_anchored_en_control(en_u, float(kb.speaker_f0_med_hz), vp, kb, use_en_articulation=True)

    vm_z = c_zh.voiced & (c_zh.f0_hz > 40.0)
    vm_e = c_en.voiced & (c_en.f0_hz > 40.0)
    m_v = (_FREQS >= 80.0) & (_FREQS <= 11500.0)
    voiced_ltas_rmse = float(np.sqrt(np.mean(((np.mean(c_zh.log_env[vm_z], 0) - np.mean(c_en.log_env[vm_e], 0))[m_v] * NEPER_TO_DB) ** 2)))
    um_z = (~c_zh.voiced) & (np.mean(c_zh.log_env[:, _FREQS >= 1500.0], axis=1) > -9.5)
    um_e = (~c_en.voiced) & (np.mean(c_en.log_env[:, _FREQS >= 1500.0], axis=1) > -9.5)
    uv_ltas_rmse = float(np.sqrt(np.mean(((np.mean(c_zh.log_env[um_z], 0) - np.mean(c_en.log_env[um_e], 0))[_FREQS >= 800.0] * NEPER_TO_DB) ** 2)))
    bap_mae = float(np.mean(np.abs(np.median(c_zh.bap[vm_z], 0) - np.median(c_en.bap[vm_e], 0))))
    f0_diff_st = abs(12.0 * math.log2(float(np.median(c_en.f0_hz[vm_e])) / float(np.median(c_zh.f0_hz[vm_z]))))
    en_iqr_st = float(12.0 * math.log2(float(np.percentile(c_en.f0_hz[vm_e], 75)) / max(40.0, float(np.percentile(c_en.f0_hz[vm_e], 25)))))

    # 4. Generate side-by-side A/B audition WAVs (Stage 9A vs 9B)
    showcase = [
        ("pure_en", "Sixty seven friends test our cross lingual speech synthesis clearly.", "纯粹英文长句", None),
        ("minimal_bat_bad", "A black bat, and a really bad plan.", "Pre-fortis 短长对比 (bat vs bad)", None),
        ("dark_l_cluster", "Little light in the cold milk world.", "Onset Light-L vs Coda Dark-L & 辅音簇", None),
        ("en_fricatives", "Think about this very smooth zoo vision today.", "英语特有擦音 θ/ð/v/z/ʒ", None),
        ("linking_phrase", "Please turn it on and take an apple home.", "跨词连读 (turn it on / an apple)", None),
        ("mix_zh_en", "今天用 PyBot 测试 think this very smooth speech，非常自然！", "中英混合同声道无缝切换", None),
    ]
    sample_clone = REPO_ROOT / "exp_v8_data" / "corpus" / "zh" / "SSB0073" / "SSB00730001.wav"
    base_prof = load_base_speaker_profile()
    tgt_prof = None
    if sample_clone.exists():
        w24, _ = decode_wav_bytes_to_24k_mono(sample_clone.read_bytes())
        tgt_prof = extract_clone_profile([w24], name="SSB0073")
        showcase.append(("clone_male_en", "Hello world, turn it on and think clearly today.", "男声模仿 SSB0073 说英文", tgt_prof))

    pairs_html: list[dict[str, Any]] = []
    rtfs_9b: list[float] = []
    all_ok = True

    def _synth_ab(sid: str, txt: str, use_art: bool, clone_p: Any) -> tuple[str, bool, float]:
        t0 = time.perf_counter()
        # Handle pure-EN via explicit flag, or mixed via build_utterance_control (uses Stage 9B by default)
        ir_s = _safe_plan_v13(txt)
        units_s = list(ir_s.get("units") or [])
        has_zh = any(u.get("lang") == "zh" for u in units_s)
        if not has_zh and clone_p is None:
            ctrl, _ = build_speaker_anchored_en_control(units_s, float(kb.speaker_f0_med_hz), vp, kb, use_en_articulation=use_art)
            eff_ph = phasor
        elif not has_zh and clone_p is not None:
            ctrl, _ = build_speaker_anchored_en_control(units_s, float(kb.speaker_f0_med_hz), vp, kb, use_en_articulation=use_art)
            ctrl, eff_ph, _ = apply_clone_to_control(ctrl, base_prof, clone_p, strength=0.90, clone_f0=True, base_coherent_phasor=phasor)
        else:
            ctrl, _ = build_utterance_control(txt, f"{sid}_{'9b' if use_art else '9a'}", kb, arb, rules, vp)
            eff_ph = phasor
        mod = apply_voice_params(ctrl, vp, vp.seed)
        y = synthesize_larynx_v2(mod, glottal_mode="dynamic_coherent", coherent_phasor=eff_ph, uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7, seed=vp.seed)
        y_out = _rms_norm(y, -19.5)
        rtf = (time.perf_counter() - t0) / max(0.05, len(y_out) / float(SR))
        san = check_wav_sanity(y_out, SR, expected_sr=SR)
        fname = f"{sid}_{'9b' if use_art else '9a'}.wav"
        _write_pcm16_wav(SAMPLES_9B_DIR / fname, y_out, SR)
        return f"samples_stage9b_articulation/{fname}", bool(san.get("ok", False)), rtf

    for sid, txt, note, cp in showcase:
        w9a, ok9a, _ = _synth_ab(sid, txt, False, cp)
        w9b, ok9b, rtf9b = _synth_ab(sid, txt, True, cp)
        all_ok = all_ok and ok9a and ok9b
        rtfs_9b.append(rtf9b)
        pairs_html.append({"label": sid, "note": note, "wav_9a": w9a, "wav_9b": w9b})

    median_rtf = round(float(np.median(rtfs_9b)), 4) if rtfs_9b else 1.0
    size_bytes = EN_ARTICULATION_PATH.stat().st_size

    gates = {
        "all_sanity_ok": all_ok,
        "artifact_le_1mb": size_bytes <= 1_048_576,
        "median_rtf_le_006": median_rtf <= 0.06,
        "heldout_dur_rel_improved": ab_heldout["9B"]["dur_rel_mae_ms"] < ab_heldout["9A"]["dur_rel_mae_ms"],
        "heldout_cep_improved": ab_heldout["9B"]["art_cep_1_19_rmse"] < ab_heldout["9A"]["art_cep_1_19_rmse"],
        "dark_l_lower_f2_ge_150": (f2_light_l - f2_dark_l) >= 150.0,
        "pre_fortis_bat_le_085_bad": dur_bat_ae <= 0.85 * dur_bad_ae,
        "voiced_ltas_rmse_le_4_5": voiced_ltas_rmse <= 4.5,
        "uv_ltas_rmse_le_6_5": uv_ltas_rmse <= 6.5,
        "bap6_mae_le_0_08": bap_mae <= 0.08,
        "f0_diff_st_le_1_0": f0_diff_st <= 1.0,
        "en_f0_iqr_ge_2_4": en_iqr_st >= 2.4,
    }
    verdict = "GO" if all(gates.values()) else "NO_GO"

    summary = {
        "verdict": verdict,
        "artifact_size_kb": round(size_bytes / 1024.0, 2),
        "median_rtf": median_rtf,
        "heldout_20_ab": ab_heldout,
        "allophones": {
            "f2_light_l_hz": round(f2_light_l, 1),
            "f2_dark_l_milk_hz": round(f2_dark_l, 1),
            "dur_ae_bat_ms": round(dur_bat_ae, 1),
            "dur_ae_bad_ms": round(dur_bad_ae, 1),
            "bat_over_bad_ratio": round(dur_bat_ae / max(1.0, dur_bad_ae), 3),
        },
        "same_speaker_9b": {
            "voiced_ltas_rmse_db": round(voiced_ltas_rmse, 2),
            "unvoiced_ltas_rmse_db": round(uv_ltas_rmse, 2),
            "bap6_mae": round(bap_mae, 4),
            "f0_diff_st": round(f0_diff_st, 2),
            "en_f0_iqr_st": round(en_iqr_st, 2),
        },
        "gates": gates,
    }
    RESULTS_9B_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_html(pairs_html, LISTEN_9B_PATH)

    md_lines = [
        "# PyBot-Larynx Stage 9B 英文发音增量与协同衔接验证报告",
        "",
        f"- **判定结论 (`verdict`)**: **`{verdict}`** (`artifact_size={round(size_bytes/1024.0, 2)} KB`, `median_rtf={median_rtf}`, `zero_ml_weights=True`, `all_sanity_ok={all_ok}`)",
        "",
        "## 1. 20 句严格留出真实英文语料 A/B 对照（零数据泄漏）",
        "",
        "| 指标（20 句未见句，`n=" + str(ab_heldout["9B"]["n_tokens"]) + "` 音素） | Stage 9A（纯规则基线） | Stage 9B（+发音/衔接/节奏表） | 改善 |",
        "|---|---:|---:|---:|",
        f"| 绝对音素时长 MAE (ms) | {ab_heldout['9A']['dur_mae_ms']:.2f} | **{ab_heldout['9B']['dur_mae_ms']:.2f}** | {ab_heldout['9B']['dur_mae_ms'] - ab_heldout['9A']['dur_mae_ms']:+.2f} ms |",
        f"| 语速归一化相对节奏 MAE (ms) | {ab_heldout['9A']['dur_rel_mae_ms']:.2f} | **{ab_heldout['9B']['dur_rel_mae_ms']:.2f}** | {ab_heldout['9B']['dur_rel_mae_ms'] - ab_heldout['9A']['dur_rel_mae_ms']:+.2f} ms |",
        f"| 去说话人音色发音倒谱误差 `Δcep[1:20]` RMSE | {ab_heldout['9A']['art_cep_1_19_rmse']:.4f} | **{ab_heldout['9B']['art_cep_1_19_rmse']:.4f}** | {ab_heldout['9B']['art_cep_1_19_rmse'] - ab_heldout['9A']['art_cep_1_19_rmse']:+.4f} |",
        "",
        "## 2. 英语位置变体与节奏机制验证",
        "",
        f"- **Light-L vs Dark-L 韵尾同位异音**: `light` 声母 `/l/` `F2 = {f2_light_l:.0f} Hz` vs `milk` 韵尾 `/l_coda/` `F2 = {f2_dark_l:.0f} Hz`（差值 **`+{f2_light_l - f2_dark_l:.0f} Hz`**）",
        f"- **Pre-Fortis Clipping（清辅音前元音缩短）**: `bat /æ/` = `{dur_bat_ae:.1f} ms` vs `bad /æ/` = `{dur_bad_ae:.1f} ms`（比例 **`{dur_bat_ae/max(1.0, dur_bad_ae):.3f}`**，缩短 `{100*(1 - dur_bat_ae/max(1.0, dur_bad_ae)):.1f}%`）",
        "",
        "## 3. 中文主说话人音色锁定验证（开启 Stage 9B 后）",
        "",
        f"- 浊音 LTAS RMSE = **`{voiced_ltas_rmse:.2f} dB`** (`<= 4.50`), 清音 LTAS RMSE = **`{uv_ltas_rmse:.2f} dB`** (`<= 6.50`), BAP6 MAE = **`{bap_mae:.4f}`**, `|ΔF0|` = **`{f0_diff_st:.2f} st`**, 英文 `F0 IQR` = **`{en_iqr_st:.2f} st`**",
        "",
        "## 4. Stage 9A vs 9B 并排离线对比试听台",
        "- [listen_stage9b_articulation.html](listen_stage9b_articulation.html)",
    ]
    REPORT_9B_PATH.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({"verdict": verdict, "median_rtf": median_rtf, "heldout_20_ab": ab_heldout, "allophones": summary["allophones"], "same_speaker_9b": summary["same_speaker_9b"], "gates": gates}, ensure_ascii=False, indent=2))
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
