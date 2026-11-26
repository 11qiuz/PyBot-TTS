"""Stage 4 Integrated Verification & Final Readout for PyBot-Larynx (`exp_20`)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from scipy.fft import rfft, rfftfreq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import EXP_DIR, HOP, N_FFT, SAMPLES_DIR, SR  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge import (  # noqa: E402
    get_splits,
    load_v15_index_cached,
)
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import (  # noqa: E402
    STAGE2_KNOWLEDGE_PATH,
    STAGE2_SUMMARY_PATH,
    AcousticKnowledgeStage2,
    build_stage2_control,
)
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB, Arbiter  # noqa: E402
from exp_20_pybot_larynx.frontend.prosody_rules import render_rule_auto_utterance  # noqa: E402
from exp_20_pybot_larynx.judge import evaluate_utterance, summarize_condition  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import (  # noqa: E402
    LarynxControl,
    log_env_to_cep,
    synthesize_larynx,
)
from exp_20_pybot_larynx.rules import default_stage0_rules  # noqa: E402
from exp_20_pybot_larynx.stage0_probe import (  # noqa: E402
    V15_10_MAP,
    _aggregate_kb_hits,
    _prepare_utterance_oracle,
)
from exp_v5_eval.metrics import read_wav  # noqa: E402
from exp_v9_clarity_pro.hnm_v9 import analyze_v9  # noqa: E402
from exp_v12_natural_vocoder.v12_bank import _patch_unsmoothed_uv_env  # noqa: E402
from exp_v14_fluent_zh_tts.eval_v14 import _rms_norm, _write_pcm16_wav  # noqa: E402
from exp_v15_bznsyp_tts.index_bznsyp_v15 import (  # noqa: E402
    load_speaker_assets_v15,
    parse_bznsyp_corpus,
)

STAGE3_RESULTS_PATH = EXP_DIR / "stage3_results.json"
STAGE4_RESULTS_PATH = EXP_DIR / "stage4_results.json"
STAGE4_REPORT_PATH = EXP_DIR / "STAGE4_REPORT.md"
LISTEN_STAGE4_PATH = EXP_DIR / "listen_stage4.html"

LONG_DEV5_IDS: tuple[str, ...] = ("009880", "005860", "007118", "004365", "001738")

_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_M_V_1_25 = (_FREQS >= 1000.0) & (_FREQS <= 2500.0)
_M_V_25_5 = (_FREQS >= 2500.0) & (_FREQS <= 5000.0)
_M_UV_1_11 = (_FREQS >= 1000.0) & (_FREQS <= 11000.0)
_HANN = np.hanning(N_FFT)

_HNR_WIN = 480
_HNR_NFFT = 2048
_HNR_NBINS = _HNR_NFFT // 2 + 1
_HNR_HANN = np.hanning(_HNR_WIN)
_HNR_FREQS = rfftfreq(_HNR_NFFT, 1.0 / SR)
_B_1_25 = (_HNR_FREQS >= 1000.0) & (_HNR_FREQS <= 2500.0)
_B_25_45 = (_HNR_FREQS >= 2500.0) & (_HNR_FREQS <= 4500.0)
_B_45_7 = (_HNR_FREQS >= 4500.0) & (_HNR_FREQS <= 7000.0)


def _extract_stage4_detail(y: np.ndarray) -> dict[str, float | None]:
    """Extract Crest, 1-7 kHz Band HNR, F0 span, STFT/Env p90-p10, and cep[12:40] std."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    fv9, _ = analyze_v9(arr, sr=SR)
    fp = _patch_unsmoothed_uv_env(arr, fv9, sr=SR, n_fft=N_FFT)
    env_db = fp.log_env * NEPER_TO_DB
    cep = log_env_to_cep(fp.log_env, n_cep=64)

    nf = len(fp.f0_hz)
    idx = np.arange(nf)[:, None] * HOP + np.arange(-(N_FFT // 2), N_FFT // 2)[None, :]
    frames = arr[np.clip(idx, 0, len(arr) - 1)]
    stft_db = 20.0 * np.log10(np.abs(rfft(frames * _HANN[None, :], n=N_FFT, axis=1)) + 1e-9)

    active = np.max(env_db, axis=1) > (float(np.percentile(np.max(env_db, axis=1), 15)) + 3.0)
    v_mask = fp.voiced & (fp.f0_hz > 50.0)
    uv_mask = (~fp.voiced) & active

    idx_h = np.arange(nf)[:, None] * HOP + np.arange(-(_HNR_WIN // 2), _HNR_WIN // 2)[None, :]
    h_frames = arr[np.clip(idx_h, 0, len(arr) - 1)]
    v_idx = np.where(v_mask)[0]

    crest_v: float | None = None
    hnr_1_25: float | None = None
    hnr_25_45: float | None = None
    hnr_45_7: float | None = None
    f0_span_st: float | None = None

    if v_idx.size >= 3:
        vf = h_frames[v_idx]
        crest_v = float(np.median(np.max(np.abs(vf), axis=1) / np.sqrt(np.mean(vf * vf, axis=1) + 1e-18)))
        f0_v = fp.f0_hz[v_idx]
        f0_span_st = float(12.0 * math.log2(float(np.percentile(f0_v, 95)) / max(50.0, float(np.percentile(f0_v, 5)))))

        pwr = np.abs(rfft(vf * _HNR_HANN[None, :], n=_HNR_NFFT, axis=1)) ** 2
        v1, v2, v3 = [], [], []
        for k_i, t in enumerate(v_idx):
            f = float(fp.f0_hz[t])
            hm = np.zeros(_HNR_NBINS, dtype=bool)
            k = 1
            while k * f <= 7500.0:
                c = k * f * _HNR_NFFT / SR
                w = max(1, round(0.35 * f * _HNR_NFFT / SR))
                hm[max(0, int(round(c)) - w) : min(_HNR_NBINS, int(round(c)) + w + 1)] = True
                k += 1
            p_row = pwr[k_i]
            for band, acc in ((_B_1_25, v1), (_B_25_45, v2), (_B_45_7, v3)):
                mh = band & hm
                mi = band & (~hm)
                if np.any(mh) and np.any(mi):
                    acc.append(10.0 * math.log10((float(np.sum(p_row[mh])) + 1e-18) / (float(np.sum(p_row[mi])) + 1e-18)))
        hnr_1_25 = float(np.median(v1)) if v1 else None
        hnr_25_45 = float(np.median(v2)) if v2 else None
        hnr_45_7 = float(np.median(v3)) if v3 else None

    def _p90_10(sub: np.ndarray) -> float | None:
        if sub.shape[0] < 3:
            return None
        return float(np.median(np.percentile(sub, 90, axis=1) - np.percentile(sub, 10, axis=1)))

    return {
        "crest_voiced": crest_v,
        "hnr_1_25k_db": hnr_1_25,
        "hnr_25_45k_db": hnr_25_45,
        "hnr_45_7k_db": hnr_45_7,
        "f0_span_p95_p5_st": f0_span_st,
        "stft_v_p90_p10_1_25k_db": _p90_10(stft_db[v_mask][:, _M_V_1_25]) if np.any(v_mask) else None,
        "stft_v_p90_p10_25_5k_db": _p90_10(stft_db[v_mask][:, _M_V_25_5]) if np.any(v_mask) else None,
        "stft_uv_p90_p10_1_11k_db": _p90_10(stft_db[uv_mask][:, _M_UV_1_11]) if np.any(uv_mask) else None,
        "env_v_p90_p10_1_25k_db": _p90_10(env_db[v_mask][:, _M_V_1_25]) if np.any(v_mask) else None,
        "env_v_p90_p10_25_5k_db": _p90_10(env_db[v_mask][:, _M_V_25_5]) if np.any(v_mask) else None,
        "env_uv_p90_p10_1_11k_db": _p90_10(env_db[uv_mask][:, _M_UV_1_11]) if np.any(uv_mask) else None,
        "cep12_24_std": float(np.mean(np.std(cep[v_mask, 12:24], axis=0))) if np.sum(v_mask) >= 5 else None,
        "cep24_40_std": float(np.mean(np.std(cep[v_mask, 24:40], axis=0))) if np.sum(v_mask) >= 5 else None,
        "cep12_40_std": float(np.mean(np.std(cep[v_mask, 12:40], axis=0))) if np.sum(v_mask) >= 5 else None,
    }


def _summarize_detail(
    rows: list[dict[str, float | None]],
    c0_detail: dict[str, float] | None = None,
    gain_stds: list[float] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = (
        "crest_voiced",
        "hnr_1_25k_db",
        "hnr_25_45k_db",
        "hnr_45_7k_db",
        "f0_span_p95_p5_st",
        "stft_v_p90_p10_1_25k_db",
        "stft_v_p90_p10_25_5k_db",
        "stft_uv_p90_p10_1_11k_db",
        "env_v_p90_p10_1_25k_db",
        "env_v_p90_p10_25_5k_db",
        "env_uv_p90_p10_1_11k_db",
        "cep12_24_std",
        "cep24_40_std",
        "cep12_40_std",
    )
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
        if vals:
            med = float(np.median(vals))
            out[k] = round(med, 4)
            if c0_detail is not None and k.startswith("cep") and c0_detail.get(k, 0.0) > 1e-9:
                out[f"{k}_ratio_vs_c0"] = round(med / c0_detail[k], 4)
    if gain_stds:
        out["gain_db_std_median"] = round(float(np.median(gain_stds)), 4)
    return out


def _aggregate_arbiter(traces: list[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, dict[str, Any]] = {}
    for tr in traces:
        for m in tr.get("modules", []):
            name = m["module"]
            if name not in agg:
                agg[name] = {
                    "utts_fired": 0,
                    "frames_touched_total": 0,
                    "voiced_overrides_total": 0,
                    "env_deltas_db": [],
                    "bap_deltas": [],
                    "tilt_deltas": [],
                    "rd_deltas": [],
                    "errors": 0,
                }
            a = agg[name]
            if m.get("error"):
                a["errors"] += 1
            if m.get("frames_touched", 0) > 0:
                a["utts_fired"] += 1
                a["frames_touched_total"] += int(m["frames_touched"])
                a["voiced_overrides_total"] += int(m["voiced_overrides"])
                a["env_deltas_db"].append(float(m.get("mean_abs_env_delta_db", 0.0)))
                a["bap_deltas"].append(float(m.get("mean_abs_bap_delta", 0.0)))
                a["tilt_deltas"].append(float(m.get("mean_abs_tilt_delta", 0.0)))
                a["rd_deltas"].append(float(m.get("mean_abs_rd_delta", 0.0)))

    return {
        name: {
            "utts_fired": a["utts_fired"],
            "frames_touched_total": a["frames_touched_total"],
            "voiced_overrides_total": a["voiced_overrides_total"],
            "mean_abs_env_delta_db": round(float(np.mean(a["env_deltas_db"])), 4) if a["env_deltas_db"] else 0.0,
            "mean_abs_bap_delta": round(float(np.mean(a["bap_deltas"])), 4) if a["bap_deltas"] else 0.0,
            "mean_abs_tilt_delta": round(float(np.mean(a["tilt_deltas"])), 4) if a["tilt_deltas"] else 0.0,
            "mean_abs_rd_delta": round(float(np.mean(a["rd_deltas"])), 4) if a["rd_deltas"] else 0.0,
            "errors": a["errors"],
        }
        for name, a in agg.items()
    }


def _render_listen_stage4_html(groups: list[dict[str, Any]], out_path: Path) -> None:
    payload_json = json.dumps(groups, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-Larynx (exp_20 Stage 4) 去电音与自然说话方式盲听对照台</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#eef1f4;background:#11151c;max-width:1020px;margin:1.8rem auto;padding:0 1rem}}
h1{{font-size:1.5rem;margin-bottom:.25rem}}
h2{{font-size:1.15rem;margin:1.4rem 0 .4rem;color:#9edbc2}}
.sub{{color:#9aa6b8;margin-bottom:1.1rem}}
.card{{background:#19202b;border:1px solid #293345;padding:1rem 1.2rem;margin:1rem 0;border-radius:10px}}
.row{{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;padding:.45rem 0;border-bottom:1px dashed #273142}}
.row:last-child{{border-bottom:none}}
audio{{width:min(100%,25rem);height:34px}}
button.btn{{background:#2c5e4b;color:#fff;border:none;border-radius:6px;padding:.5rem .95rem;cursor:pointer;margin:.4rem .5rem .2rem 0}}
.ans{{display:none;margin-top:.55rem;padding:.55rem .8rem;background:#12211c;border-left:3px solid #449c7c;border-radius:4px;font-size:.92rem;color:#cfece0}}
.ans.show{{display:block}}
</style>
</head>
<body>
<h1>PyBot-Larynx (exp_20 Stage 4) · 去电音核 + 自然说话方式盲听对照台</h1>
<p class="sub">Stage 4 升级（混合低波峰因数色散相位 + AR(1) HF 相位/抖动/微颤 + 声门周期调制 y_ap + 1-7kHz 频段匹配 BAP + 25 因子说话方式 F0/响度/动态停顿）。包含 DEV_60 最长 5 句（L01–L05）与 HELDOUT_20 前 5 句（H01–H05），所有音频统一 -20 dBFS 并随机打乱。</p>
<div id="root"></div>
<div class="card">
  <button class="btn" onclick="toggleAns()">显示 / 隐藏答案映射</button>
</div>
<script>
const GROUPS = {payload_json};
let show = false;
function render() {{
  const root = document.getElementById('root');
  root.replaceChildren();
  let lastSec = '';
  GROUPS.forEach(g => {{
    if (g.section && g.section !== lastSec) {{
      const h2 = document.createElement('h2');
      h2.textContent = g.section;
      root.append(h2);
      lastSec = g.section;
    }}
    const card = document.createElement('div'); card.className = 'card';
    const h = document.createElement('h3'); h.textContent = g.title + ' — ' + g.text; card.append(h);
    const ansLines = [];
    g.items.forEach(it => {{
      const r = document.createElement('div'); r.className = 'row';
      const b = document.createElement('b'); b.textContent = it.label + ' 段';
      const au = document.createElement('audio'); au.controls = true; au.preload = 'none'; au.src = it.file;
      r.append(b, au); card.append(r);
      ansLines.push(it.label + ': ' + it.system_desc);
    }});
    const ans = document.createElement('div'); ans.className = 'ans'; ans.textContent = ansLines.join(' | ');
    card.append(ans);
    root.append(card);
  }});
}}
function toggleAns() {{
  show = !show;
  document.querySelectorAll('.ans').forEach(a => a.classList.toggle('show', show));
}}
render();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def _write_stage4_report(res: dict[str, Any], out_path: Path) -> None:
    ds = res["dev_60_summary"]
    hs = res["heldout_summary"]
    dd = res["dev_60_detail"]
    hd = res["heldout_detail"]
    s3_dev = res.get("stage3_dev_60_summary", {})
    s3_held = res.get("stage3_heldout_summary", {})
    kb = res["knowledge_summary"]
    hits_dev = res["dev_c3_kb_hits"]
    hits_held = res["held_c3_kb_hits"]
    attr = res["held_c3_arbiter_attribution"]
    g = res["gates"]

    def _fmt_row(cname: str, s: dict[str, Any], d: dict[str, float], s3: dict[str, Any] | None = None) -> str:
        m = s["medians"]
        uv_s = f"{m.get('uv_2_8k_vs_lo_db'):.3f}" if m.get("uv_2_8k_vs_lo_db") is not None else "—"
        hnr_s = f"{m.get('hnr_voiced_db'):.2f}" if m.get("hnr_voiced_db") is not None else "—"
        crest_s = f"{d.get('crest_voiced'):.3f}" if d.get("crest_voiced") is not None else "—"
        delta_s = ""
        if s3 and "medians" in s3:
            dm = m["mel_lsd_db"] - s3["medians"]["mel_lsd_db"]
            delta_s = f" ({dm:+.4f})"
        return (
            f"| `{cname}` | **{m['mel_lsd_db']:.4f}**{delta_s} | {m['mcd_db']:.4f} | {m['f0_mae_st']:.4f} | "
            f"{hnr_s} | {crest_s} | {uv_s} | {m['clicks_gt8']:.6f} | **{m['rtf']:.4f}** | {s['sanity_ok']}/{s['n']} |"
        )

    def _fmt_detail(cname: str, d: dict[str, float]) -> str:
        b1 = d.get("hnr_1_25k_db")
        b2 = d.get("hnr_25_45k_db")
        b3 = d.get("hnr_45_7k_db")
        bh = f"{b1:.2f} / {b2:.2f} / {b3:.2f}" if (b1 is not None and b2 is not None and b3 is not None) else "—"
        gstd = f"{d['gain_db_std_median']:.2f}" if d.get("gain_db_std_median") is not None else "—"
        return (
            f"| `{cname}` | {bh} | {d.get('f0_span_p95_p5_st', 0):.2f} | {gstd} | "
            f"{d.get('env_v_p90_p10_1_25k_db', 0):.2f} / {d.get('env_v_p90_p10_25_5k_db', 0):.2f} | "
            f"{d.get('stft_uv_p90_p10_1_11k_db', 0):.2f} / {d.get('env_uv_p90_p10_1_11k_db', 0):.2f} | "
            f"**{d.get('cep12_40_std_ratio_vs_c0', 1.0):.3f}x** |"
        )

    all_conds = ("Real", "C0_s4", "C0_s4_coh", "Stage4_C2", "Stage4_C3", "Stage4_C3_coh", "Stage4_C4", "Stage4_C4_coh")
    s3_dev_map = {
        "Real": None,
        "C0_s4": s3_dev.get("C0_s3"),
        "C0_s4_coh": None,
        "Stage4_C2": s3_dev.get("Stage3_C2"),
        "Stage4_C3": s3_dev.get("Stage3_C3"),
        "Stage4_C3_coh": None,
        "Stage4_C4": s3_dev.get("Stage3_C4"),
        "Stage4_C4_coh": None,
    }
    s3_held_map = {
        "Real": None,
        "C0_s4": s3_held.get("C0_s3"),
        "C0_s4_coh": None,
        "Stage4_C2": s3_held.get("Stage3_C2"),
        "Stage4_C3": s3_held.get("Stage3_C3"),
        "Stage4_C3_coh": s3_held.get("Stage3_C3_coh"),
        "Stage4_C4": s3_held.get("Stage3_C4"),
        "Stage4_C4_coh": None,
    }

    lines = [
        "# PyBot-Larynx (`exp_20_pybot_larynx`) Stage 4 去电音与自然说话方式集成验证报告",
        "",
        f"- **判定结论 (`verdict`)**: **`{res['verdict']}`** (`dev_pass={g['dev_pass']}`, `heldout_c3_pass={g['heldout_c3_pass']}`, `heldout_c4_beats_v15={g['heldout_c4_beats_v15']}`, `anti_buzz_and_style_gates={g['anti_buzz_and_style_gates']}`, `detail_gates_pass={g['detail_gates_pass']}`, `zero_arbiter_errors={g['zero_arbiter_errors']}`)",
        "- **零神经权重 / 零波形拼接**: `0.0 MB` neural / sklearn / pickle weights，`100%` 电子声带实时合成",
        f"- **Stage 4 知识库 (`knowledge_stage2.npz`)**: `{kb['n_train_utts']}` 条 `TRAIN` 句构建 (`{kb['npz_size_kb']} KB`)，保留无模糊 Medoid + 阶梯低频锚定 `[0.90]*12+[0.65]*5+[0.35]*7` + `cep[10:26]` 铃形对比度恢复 (`0.12`) + `1-7 kHz` 频段校准 BAP",
        f"- **知识库命中率**: `DEV_60` 精确 `(o,f,t)` **`{hits_dev['final_exact_oft_rate']*100:.2f}%`** (`place_or_better {hits_dev['final_place_or_better_rate']*100:.2f}%`); `HELDOUT_20` 精确 `(o,f,t)` **`{hits_held['final_exact_oft_rate']*100:.2f}%`** (`place_or_better {hits_held['final_place_or_better_rate']*100:.2f}%`)",
        "",
        "## 1. `DEV_60` 集成结果（中位数，括号内为相对 Stage 3 对应指标变化）",
        "",
        "| 条件 | Mel-LSD (dB) | MCD (dB) | F0 MAE (st) | HNR (dB) | Crest | 清音 2-8k/lo (dB) | clicks >8 | CPU RTF | Sanity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in all_conds:
        lines.append(_fmt_row(cname, ds[cname], dd[cname], s3_dev_map.get(cname)))

    lines += [
        "",
        "### `DEV_60` Stage 4 频段 HNR / 说话方式动态范围 / 共振峰清晰度",
        "",
        "| 条件 | 1-2.5k / 2.5-4.5k / 4.5-7k HNR (dB) | F0 span p95-p5 (st) | gain_db std (dB) | Env vp90-10 (1-2.5k / 2.5-5k dB) | UV p90-10 (STFT / Env 1-11k dB) | cep[12:40] vs C0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in all_conds:
        lines.append(_fmt_detail(cname, dd[cname]))

    lines += [
        "",
        "## 2. `HELDOUT_20` 最终集成盲测读出（中位数，仅作最终验证，未做调参）",
        "",
        "| 条件 | Mel-LSD (dB) | MCD (dB) | F0 MAE (st) | HNR (dB) | Crest | 清音 2-8k/lo (dB) | clicks >8 | CPU RTF | Sanity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| *参照：`v15_auto` (9,726句拼接)* | *6.8795* | *21.5875* | *3.7870* | *16.32 (10句)* | — | *8.838* | *0.000600* | *0.4761* | *20/20* |",
        "| *参照：`v15_guided` (参考节奏F0拼接)* | *5.7540* | *20.9761* | *1.3738* | — | — | — | — | — | *20/20* |",
    ]
    for cname in all_conds:
        lines.append(_fmt_row(cname, hs[cname], hd[cname], s3_held_map.get(cname)))

    lines += [
        "",
        "### `HELDOUT_20` Stage 4 频段 HNR / 说话方式动态范围 / 共振峰清晰度",
        "",
        "| 条件 | 1-2.5k / 2.5-4.5k / 4.5-7k HNR (dB) | F0 span p95-p5 (st) | gain_db std (dB) | Env vp90-10 (1-2.5k / 2.5-5k dB) | UV p90-10 (STFT / Env 1-11k dB) | cep[12:40] vs C0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in all_conds:
        lines.append(_fmt_detail(cname, hd[cname]))

    lines += [
        "",
        f"> 注：`Stage4_C4` 在 `HELDOUT_20` 端到端中位 RTF = **`{hs['Stage4_C4']['medians']['rtf']:.4f}`**（其中电子声带核 `synthesize_larynx` 占 `{res['held_c4_kernel_rtf_median']:.4f}`）；`DEV_60` 端到端中位 RTF = **`{ds['Stage4_C4']['medians']['rtf']:.4f}`**（核 `{res['dev_c4_kernel_rtf_median']:.4f}`）。",
        "",
        "## 3. Arbiter 规则模块归因（`Stage4_C3` on `HELDOUT_20`）",
        "",
        "| 规则模块 | 触发句数 | 命中总帧数 | 清浊覆盖帧 | 均值 `|Δenv|` (dB) | 均值 `|Δbap|` | 均值 `|Δtilt|` | 均值 `|Δrd|` | 异常 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mname, a in attr.items():
        lines.append(
            f"| `{mname}` | {a['utts_fired']}/20 | {a['frames_touched_total']} | {a['voiced_overrides_total']} | "
            f"{a['mean_abs_env_delta_db']:.4f} | {a['mean_abs_bap_delta']:.4f} | {a['mean_abs_tilt_delta']:.4f} | {a['mean_abs_rd_delta']:.4f} | {a['errors']} |"
        )

    lines += [
        "",
        "## 4. 试听交付物",
        "- 离线盲听对比台：[listen_stage4.html](listen_stage4.html)（含 `DEV_60` 最长 5 句 `009880 / 005860 / 007118 / 004365 / 001738` 与 `HELDOUT_20` 前 5 句：`Real` / `C0_s4` / `Stage4_C3` / `Stage4_C3_coh` / `Stage4_C4` / `Stage4_C4_coh` 或 `v15_auto`，全部统一 `-20 dBFS` 并随机打乱）",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _eval_split(
    split_name: str,
    oracles: list[dict[str, Any]],
    kb2: AcousticKnowledgeStage2,
    arbiter: Arbiter,
    rules: list[Any],
    coh_phasor: np.ndarray,
    *,
    seed_base: int,
    capture_uids: set[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[float], dict[str, dict[str, np.ndarray]]]:
    conds = ("Real", "C0_s4", "C0_s4_coh", "Stage4_C2", "Stage4_C3", "Stage4_C3_coh", "Stage4_C4", "Stage4_C4_coh")
    rows: dict[str, list[dict[str, Any]]] = {c: [] for c in conds}
    det: dict[str, list[dict[str, float | None]]] = {c: [] for c in conds}
    c3_traces: list[dict[str, Any]] = []
    c3_kb_infos: list[dict[str, Any]] = []
    c4_kernel_rtfs: list[float] = []
    c4_gain_stds: dict[str, list[float]] = {"Stage4_C4": [], "Stage4_C4_coh": []}
    captured: dict[str, dict[str, np.ndarray]] = {}

    for u_idx, item in enumerate(oracles):
        uid = item["uid"]
        ref_span = item["ref_span"]
        ref_fast = item["ref_fast"]
        feats_p = item["feats_p"]
        bap6 = item["bap6"]
        ctx = item["ctx"]
        seed_u = seed_base + u_idx
        want = uid in capture_uids

        # Real
        rows["Real"].append(evaluate_utterance(ref_span, ref_span, rtf=0.0, ref_fast=ref_fast))
        det["Real"].append(_extract_stage4_detail(ref_span))
        if want:
            captured.setdefault(uid, {})["Real"] = ref_span

        # C0_s4 & C0_s4_coh
        for cname, gmode in (("C0_s4", "parametric_lf"), ("C0_s4_coh", "speaker_coherent")):
            c0 = LarynxControl(f0_hz=feats_p.f0_hz, voiced=feats_p.voiced, log_env=feats_p.log_env, bap=bap6, n_samples=len(ref_span))
            ts = time.perf_counter()
            y0 = synthesize_larynx(c0, glottal_mode=gmode, coherent_phasor=coh_phasor, seed=seed_u)
            dt = time.perf_counter() - ts
            rows[cname].append(evaluate_utterance(y0, ref_span, rtf=dt / max(1e-6, len(y0) / SR), ref_fast=ref_fast))
            det[cname].append(_extract_stage4_detail(y0))
            if want and cname == "C0_s4":
                captured[uid]["C0_s4"] = y0

        # Stage4_C2
        ts = time.perf_counter()
        c2, _ = build_stage2_control(ctx, feats_p.f0_hz, feats_p.voiced, kb2)
        c2.n_samples = len(ref_span)
        y2 = synthesize_larynx(c2, glottal_mode="parametric_lf", seed=seed_u)
        dt = time.perf_counter() - ts
        rows["Stage4_C2"].append(evaluate_utterance(y2, ref_span, rtf=dt / max(1e-6, len(y2) / SR), ref_fast=ref_fast))
        det["Stage4_C2"].append(_extract_stage4_detail(y2))

        # Stage4_C3 & Stage4_C3_coh
        for cname, gmode in (("Stage4_C3", "parametric_lf"), ("Stage4_C3_coh", "speaker_coherent")):
            ts = time.perf_counter()
            c3b, kb_c3 = build_stage2_control(ctx, feats_p.f0_hz, feats_p.voiced, kb2)
            c3b.n_samples = len(ref_span)
            c3f, tr_c3 = arbiter.apply(ctx, c3b, rules)
            y3 = synthesize_larynx(c3f, glottal_mode=gmode, coherent_phasor=coh_phasor, seed=seed_u)
            dt = time.perf_counter() - ts
            rows[cname].append(evaluate_utterance(y3, ref_span, rtf=dt / max(1e-6, len(y3) / SR), ref_fast=ref_fast))
            det[cname].append(_extract_stage4_detail(y3))
            if cname == "Stage4_C3":
                c3_traces.append(tr_c3)
                c3_kb_infos.append(kb_c3)
            if want:
                captured[uid][cname] = y3

        # Stage4_C4 & Stage4_C4_coh
        for cname, gmode in (("Stage4_C4", "parametric_lf"), ("Stage4_C4_coh", "speaker_coherent")):
            ts = time.perf_counter()
            y4, info4 = render_rule_auto_utterance(
                item["text"],
                uid,
                kb2,
                arbiter=arbiter,
                rules=rules,
                glottal_mode=gmode,
                coherent_phasor=coh_phasor,
                seed=seed_u,
            )
            dt = time.perf_counter() - ts
            dur4 = max(1e-6, len(y4) / SR)
            assert info4["prosody_plan"].get("zero_ml_weights") is True
            assert info4["kb_info"].get("mode") == "stage2"
            if cname == "Stage4_C4":
                c4_kernel_rtfs.append(info4["kernel_s"] / dur4)
            gains = [float(u.get("gain_db", 0.0)) for u in info4["prosody_plan"]["units"]]
            if len(gains) >= 2:
                c4_gain_stds[cname].append(float(np.std(gains)))
            rows[cname].append(evaluate_utterance(y4, ref_span, rtf=dt / dur4, ref_fast=ref_fast))
            det[cname].append(_extract_stage4_detail(y4))
            if want:
                captured[uid][cname] = y4

        if (u_idx + 1) % 10 == 0 or (u_idx + 1) == len(oracles):
            print(f"[{split_name}] {u_idx + 1}/{len(oracles)} done", flush=True)

    summary = {c: summarize_condition(rows[c]) for c in conds}
    c0_det = _summarize_detail(det["C0_s4"])
    detail = {
        c: _summarize_detail(
            det[c],
            None if c == "C0_s4" else c0_det,
            c4_gain_stds.get(c),
        )
        for c in conds
    }
    return summary, detail, c3_traces, c3_kb_infos, c4_kernel_rtfs, captured


def run_stage4_verification() -> dict[str, Any]:
    t0 = time.perf_counter()
    if not STAGE2_KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(f"Missing {STAGE2_KNOWLEDGE_PATH}")

    kb_summary = json.loads(STAGE2_SUMMARY_PATH.read_text(encoding="utf-8"))
    s3_res = json.loads(STAGE3_RESULTS_PATH.read_text(encoding="utf-8")) if STAGE3_RESULTS_PATH.exists() else {}

    kb2 = AcousticKnowledgeStage2.load(STAGE2_KNOWLEDGE_PATH)
    idx_doc = load_v15_index_cached()
    assets = load_speaker_assets_v15()
    coh_phasor = assets["glottal_bank"].coherent_phasor
    arbiter = Arbiter()
    rules = default_stage0_rules()

    train_ids, dev_60_ids, held_20_ids = get_splits()
    assert len(set(train_ids) & set(dev_60_ids)) == 0
    assert len(set(train_ids) & set(held_20_ids)) == 0
    assert len(set(dev_60_ids) & set(held_20_ids)) == 0
    for uid in LONG_DEV5_IDS:
        assert uid in set(dev_60_ids), f"{uid} not in DEV_60"

    # Warmup frontend/jieba/pypinyin/LUT once before timing
    render_rule_auto_utterance("请测试四阶段电子声带。", "warmup_s4", kb2, arbiter=arbiter, rules=rules, seed=1)

    corpus_records, _ = parse_bznsyp_corpus()
    corpus_by_id = {r["id"]: r for r in corpus_records}

    print(f"Preparing DEV_60 ({len(dev_60_ids)} utts)...", flush=True)
    dev_oracles: list[dict[str, Any]] = []
    for uid in dev_60_ids:
        dev_meta = dict(idx_doc["train_utt_metas"][uid])
        dev_meta["syllables"] = corpus_by_id[uid]["syllables"]
        dev_oracles.append(_prepare_utterance_oracle(uid, dev_meta))

    dev_summary, dev_detail, dev_c3_traces, dev_c3_kb_infos, dev_c4_k_rtfs, dev_captured = _eval_split(
        "DEV_60",
        dev_oracles,
        kb2,
        arbiter,
        rules,
        coh_phasor,
        seed_base=5000,
        capture_uids=set(LONG_DEV5_IDS),
    )

    print(f"Preparing HELDOUT_20 ({len(held_20_ids)} utts)...", flush=True)
    held_oracles = [_prepare_utterance_oracle(u, idx_doc["held_utt_metas"][u]) for u in held_20_ids]
    held_5_uids = [it["uid"] for it in held_oracles[:5]]
    held_summary, held_detail, held_c3_traces, held_c3_kb_infos, held_c4_k_rtfs, held_captured = _eval_split(
        "HELDOUT_20",
        held_oracles,
        kb2,
        arbiter,
        rules,
        coh_phasor,
        seed_base=2000,
        capture_uids=set(held_5_uids),
    )

    # Export -20 dBFS WAVs and listen_stage4.html:
    # 5 longest DEV_60 sentences (009880, 005860, 007118, 004365, 001738) + first 5 HELDOUT_20 sentences
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    listen_groups: list[dict[str, Any]] = []
    dev_by_uid = {it["uid"]: it for it in dev_oracles}
    for idx_l, uid in enumerate(LONG_DEV5_IDS):
        item = dev_by_uid[uid]
        wmap = dev_captured[uid]
        sys_entries = [
            ("Real", "Real 真人原声", wmap["Real"]),
            ("C0_s4", "C0_s4 Stage4 电子声带原包络重合成 (混合低Crest + AR(1) + 周期调制y_ap)", wmap["C0_s4"]),
            ("Stage4_C3", "Stage4 C3 (Oracle节奏F0 + 铃形恢复Medoid + 1-7kHz匹配规则, analytic LF)", wmap["Stage4_C3"]),
            ("Stage4_C3_coh", "Stage4 C3_coh (Oracle节奏F0 + 低中胸声混合相位)", wmap["Stage4_C3_coh"]),
            ("Stage4_C4", "Stage4 C4 端到端零ML说话方式合成 (25因子F0/响度/动态停顿, analytic LF)", wmap["Stage4_C4"]),
            ("Stage4_C4_coh", "Stage4 C4_coh 端到端零ML说话方式合成 (低中胸声混合相位)", wmap["Stage4_C4_coh"]),
        ]
        random.Random(202640 + idx_l).shuffle(sys_entries)
        its = []
        for lbl, (scode, sdesc, w_s) in zip(("A", "B", "C", "D", "E", "F"), sys_entries):
            fname = f"s4_dev_{uid}_{lbl}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fname, _rms_norm(w_s, target_dbfs=-20.0), sr=SR)
            its.append({"label": lbl, "file": f"samples/{fname}", "system_code": scode, "system_desc": sdesc})
        listen_groups.append({
            "section": "第一部分：DEV_60 最长 5 句长难句盲听（重点听 C4 自然度与长句去电音）",
            "id": f"L{idx_l+1:02d}",
            "uid": uid,
            "title": f"L{idx_l+1:02d} ({uid})",
            "text": item["text"],
            "items": its,
        })

    v15_samples_dir = REPO_ROOT / "exp_v15_bznsyp_tts" / "samples"
    for idx_h, item in enumerate(held_oracles[:5]):
        uid = item["uid"]
        wmap = held_captured[uid]
        v15_lbl = V15_10_MAP[uid]
        v15_w, _ = read_wav(v15_samples_dir / f"bz10_{uid}_{v15_lbl}.wav")
        sys_entries = [
            ("Real", "Real 真人原声", wmap["Real"]),
            ("C0_s4", "C0_s4 Stage4 电子声带原包络重合成 (混合低Crest + AR(1) + 周期调制y_ap)", wmap["C0_s4"]),
            ("Stage4_C3", "Stage4 C3 (Oracle节奏F0 + 铃形恢复Medoid + 1-7kHz匹配规则, analytic LF)", wmap["Stage4_C3"]),
            ("Stage4_C3_coh", "Stage4 C3_coh (Oracle节奏F0 + 低中胸声混合相位)", wmap["Stage4_C3_coh"]),
            ("Stage4_C4", "Stage4 C4 端到端零ML说话方式合成 (25因子F0/响度/动态停顿)", wmap["Stage4_C4"]),
            ("v15_auto", "v15 Auto (9,726句拼接基线)", v15_w),
        ]
        random.Random(202641 + idx_h).shuffle(sys_entries)
        its = []
        for lbl, (scode, sdesc, w_s) in zip(("A", "B", "C", "D", "E", "F"), sys_entries):
            fname = f"s4_held_{uid}_{lbl}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fname, _rms_norm(w_s, target_dbfs=-20.0), sr=SR)
            its.append({"label": lbl, "file": f"samples/{fname}", "system_code": scode, "system_desc": sdesc})
        listen_groups.append({
            "section": "第二部分：HELDOUT_20 盲测前 5 句对照（含 v15_auto 拼接基线）",
            "id": f"H{idx_h+1:02d}",
            "uid": uid,
            "title": f"H{idx_h+1:02d} ({uid})",
            "text": item["text"],
            "items": its,
        })

    _render_listen_stage4_html(listen_groups, LISTEN_STAGE4_PATH)

    # Gates & verdict
    dev_attr = _aggregate_arbiter(dev_c3_traces)
    held_attr = _aggregate_arbiter(held_c3_traces)
    zero_arbiter_errors = (
        all(v["errors"] == 0 for v in dev_attr.values())
        and all(v["errors"] == 0 for v in held_attr.values())
    )

    dc2 = dev_summary["Stage4_C2"]["medians"]
    dc3 = dev_summary["Stage4_C3"]["medians"]
    dc4 = dev_summary["Stage4_C4"]["medians"]
    hc3 = held_summary["Stage4_C3"]["medians"]
    hc4 = held_summary["Stage4_C4"]["medians"]

    dc0_det = dev_detail["C0_s4"]
    dc3_det = dev_detail["Stage4_C3"]
    dc4_det = dev_detail["Stage4_C4"]
    hc3_det = held_detail["Stage4_C3"]
    hc4_det = held_detail["Stage4_C4"]

    dev_pass = (
        dev_summary["Stage4_C2"]["sanity_ok"] == 60
        and dev_summary["Stage4_C3"]["sanity_ok"] == 60
        and dev_summary["Stage4_C3_coh"]["sanity_ok"] == 60
        and dev_summary["Stage4_C4"]["sanity_ok"] == 60
        and dev_summary["Stage4_C4_coh"]["sanity_ok"] == 60
        and dc3["mel_lsd_db"] < dc2["mel_lsd_db"]
        and dc3["rtf"] <= 0.05
        and dc4["rtf"] <= 0.05
        and dc3["clicks_gt8"] <= 0.0006
        and dc4["clicks_gt8"] <= 0.0006
    )
    heldout_c3_pass = (
        held_summary["Stage4_C3"]["sanity_ok"] == 20
        and held_summary["Stage4_C3_coh"]["sanity_ok"] == 20
        and hc3["mel_lsd_db"] <= 5.20
        and hc3["rtf"] <= 0.05
        and hc3["clicks_gt8"] <= 0.0006
    )
    heldout_c4_beats_v15 = (
        held_summary["Stage4_C4"]["sanity_ok"] == 20
        and held_summary["Stage4_C4_coh"]["sanity_ok"] == 20
        and hc4["mel_lsd_db"] < 6.8795
        and hc4["rtf"] <= 0.05
        and hc4["clicks_gt8"] <= 0.0006
    )
    anti_buzz_and_style_gates = (
        2.40 <= dc0_det.get("crest_voiced", 0.0) <= 2.75
        and 2.40 <= dc3_det.get("crest_voiced", 0.0) <= 2.75
        and 2.40 <= dc4_det.get("crest_voiced", 0.0) <= 2.75
        and 19.0 <= dev_summary["C0_s4"]["medians"]["hnr_voiced_db"] <= 23.0
        and 19.0 <= dc3["hnr_voiced_db"] <= 23.0
        and 19.0 <= hc3["hnr_voiced_db"] <= 23.0
        and dc4_det.get("f0_span_p95_p5_st", 0.0) >= 12.0
        and dc4_det.get("gain_db_std_median", 0.0) >= 1.70
        and hc4_det.get("gain_db_std_median", 0.0) >= 1.70
    )
    detail_gates_pass = (
        dc3_det.get("cep12_40_std_ratio_vs_c0", 0.0) >= 0.80
        and hc3_det.get("cep12_40_std_ratio_vs_c0", 0.0) >= 0.80
        and dc3_det.get("stft_uv_p90_p10_1_11k_db", 0.0) >= 24.5
        and dc3_det.get("stft_v_p90_p10_1_25k_db", 0.0) >= 24.0
    )

    verdict = (
        "GO"
        if (
            dev_pass
            and heldout_c3_pass
            and heldout_c4_beats_v15
            and anti_buzz_and_style_gates
            and detail_gates_pass
            and zero_arbiter_errors
        )
        else "NO_GO"
    )

    results = {
        "runtime_s": round(time.perf_counter() - t0, 2),
        "verdict": verdict,
        "zero_ml_weights": True,
        "gates": {
            "dev_pass": dev_pass,
            "heldout_c3_pass": heldout_c3_pass,
            "heldout_c4_beats_v15": heldout_c4_beats_v15,
            "anti_buzz_and_style_gates": anti_buzz_and_style_gates,
            "detail_gates_pass": detail_gates_pass,
            "zero_arbiter_errors": zero_arbiter_errors,
        },
        "knowledge_summary": kb_summary,
        "dev_c3_kb_hits": _aggregate_kb_hits(dev_c3_kb_infos),
        "held_c3_kb_hits": _aggregate_kb_hits(held_c3_kb_infos),
        "dev_c3_arbiter_attribution": dev_attr,
        "held_c3_arbiter_attribution": held_attr,
        "dev_c4_kernel_rtf_median": round(float(np.median(dev_c4_k_rtfs)), 4),
        "held_c4_kernel_rtf_median": round(float(np.median(held_c4_k_rtfs)), 4),
        "dev_60_summary": dev_summary,
        "dev_60_detail": dev_detail,
        "heldout_summary": held_summary,
        "heldout_detail": held_detail,
        "stage3_dev_60_summary": s3_res.get("dev_60_summary", {}),
        "stage3_heldout_summary": s3_res.get("heldout_summary", {}),
    }

    STAGE4_RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_stage4_report(results, STAGE4_REPORT_PATH)
    print(json.dumps({
        "verdict": verdict,
        "runtime_s": results["runtime_s"],
        "gates": results["gates"],
        "dev_C0_mel": dev_summary["C0_s4"]["medians"]["mel_lsd_db"],
        "dev_C0_crest": dc0_det.get("crest_voiced"),
        "dev_C3_mel": dc3["mel_lsd_db"],
        "dev_C3_crest": dc3_det.get("crest_voiced"),
        "dev_C4_mel": dc4["mel_lsd_db"],
        "dev_C4_f0_span": dc4_det.get("f0_span_p95_p5_st"),
        "dev_C4_gain_std": dc4_det.get("gain_db_std_median"),
        "held_C0_mel": held_summary["C0_s4"]["medians"]["mel_lsd_db"],
        "held_C3_mel": hc3["mel_lsd_db"],
        "held_C3_coh_mel": held_summary["Stage4_C3_coh"]["medians"]["mel_lsd_db"],
        "held_C4_mel": hc4["mel_lsd_db"],
        "held_C4_coh_mel": held_summary["Stage4_C4_coh"]["medians"]["mel_lsd_db"],
        "held_C3_hnr_bands": [hc3_det.get("hnr_1_25k_db"), hc3_det.get("hnr_25_45k_db"), hc3_det.get("hnr_45_7k_db")],
        "held_C3_cep12_40_ratio": hc3_det.get("cep12_40_std_ratio_vs_c0"),
        "held_C3_rtf": hc3["rtf"],
        "held_C4_rtf": hc4["rtf"],
    }, ensure_ascii=False, indent=2), flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    run_stage4_verification()
