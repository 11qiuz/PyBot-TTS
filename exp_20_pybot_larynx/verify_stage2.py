"""Stage 2 Integrated Verification & Final Readout for PyBot-Larynx (`exp_20`)."""
from __future__ import annotations

import argparse
import json
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
    _aggregate_arbiter,
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

STAGE1_RESULTS_PATH = EXP_DIR / "stage1_results.json"
STAGE2_RESULTS_PATH = EXP_DIR / "stage2_results.json"
STAGE2_REPORT_PATH = EXP_DIR / "STAGE2_REPORT.md"
LISTEN_STAGE2_PATH = EXP_DIR / "listen_stage2.html"

_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_M_V_1_25 = (_FREQS >= 1000.0) & (_FREQS <= 2500.0)
_M_V_25_5 = (_FREQS >= 2500.0) & (_FREQS <= 5000.0)
_M_UV_1_11 = (_FREQS >= 1000.0) & (_FREQS <= 11000.0)
_HANN = np.hanning(N_FFT)


def _extract_spectral_detail(y: np.ndarray) -> dict[str, float | None]:
    """Extract STFT & envelope p90-p10 (dB) and cep[12:40] std from a synthesized waveform."""
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

    def _p90_10(sub: np.ndarray) -> float | None:
        if sub.shape[0] < 3:
            return None
        return float(np.median(np.percentile(sub, 90, axis=1) - np.percentile(sub, 10, axis=1)))

    return {
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
) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = (
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
    return out


def _render_listen_stage2_html(groups: list[dict[str, Any]], out_path: Path) -> None:
    payload_json = json.dumps(groups, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-Larynx (exp_20 Stage 2) 细节还原盲听对照台</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#eef1f4;background:#11151c;max-width:980px;margin:1.8rem auto;padding:0 1rem}}
h1{{font-size:1.5rem;margin-bottom:.25rem}}
.sub{{color:#9aa6b8;margin-bottom:1.1rem}}
.card{{background:#19202b;border:1px solid #293345;padding:1rem 1.2rem;margin:1rem 0;border-radius:10px}}
.row{{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;padding:.45rem 0;border-bottom:1px dashed #273142}}
.row:last-child{{border-bottom:none}}
audio{{width:min(100%,24rem);height:34px}}
button.btn{{background:#2c5e4b;color:#fff;border:none;border-radius:6px;padding:.5rem .95rem;cursor:pointer;margin:.4rem .5rem .2rem 0}}
.ans{{display:none;margin-top:.55rem;padding:.55rem .8rem;background:#12211c;border-left:3px solid #449c7c;border-radius:4px;font-size:.92rem;color:#cfece0}}
.ans.show{{display:block}}
</style>
</head>
<body>
<h1>PyBot-Larynx (exp_20 Stage 2) · 细节还原盲听对照台</h1>
<p class="sub">Stage 2 无模糊 Medoid 知识库 + 513-bin 清音谱 + VOT 平滑与自适应 BAP + 零 ML 韵律规则端到端，对比 Real、C0 与 v15_auto。所有音频统一 -20 dBFS。</p>
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
  GROUPS.forEach(g => {{
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


def _write_stage2_report(res: dict[str, Any], out_path: Path) -> None:
    ds = res["dev_60_summary"]
    hs = res["heldout_summary"]
    dd = res["dev_60_detail"]
    hd = res["heldout_detail"]
    s1_dev = res.get("stage1_dev_60_summary", {})
    s1_held = res.get("stage1_heldout_summary", {})
    kb = res["knowledge_summary"]
    hits_dev = res["dev_c3_kb_hits"]
    hits_held = res["held_c3_kb_hits"]
    attr = res["held_c3_arbiter_attribution"]
    g = res["gates"]

    def _fmt_row(cname: str, s: dict[str, Any], s1: dict[str, Any] | None = None) -> str:
        m = s["medians"]
        uv_s = f"{m.get('uv_2_8k_vs_lo_db'):.3f}" if m.get("uv_2_8k_vs_lo_db") is not None else "—"
        hnr_s = f"{m.get('hnr_voiced_db'):.2f}" if m.get("hnr_voiced_db") is not None else "—"
        delta_s = ""
        if s1 and "medians" in s1:
            dm = m["mel_lsd_db"] - s1["medians"]["mel_lsd_db"]
            delta_s = f" ({dm:+.4f})"
        return (
            f"| `{cname}` | **{m['mel_lsd_db']:.4f}**{delta_s} | {m['mcd_db']:.4f} | {m['f0_mae_st']:.4f} | "
            f"{hnr_s} | {uv_s} | {m['clicks_gt8']:.6f} | **{m['rtf']:.4f}** | {s['sanity_ok']}/{s['n']} |"
        )

    def _fmt_detail(cname: str, d: dict[str, float]) -> str:
        return (
            f"| `{cname}` | {d.get('stft_v_p90_p10_1_25k_db', 0):.2f} / {d.get('stft_v_p90_p10_25_5k_db', 0):.2f} | "
            f"{d.get('env_v_p90_p10_1_25k_db', 0):.2f} / {d.get('env_v_p90_p10_25_5k_db', 0):.2f} | "
            f"{d.get('stft_uv_p90_p10_1_11k_db', 0):.2f} / {d.get('env_uv_p90_p10_1_11k_db', 0):.2f} | "
            f"{d.get('cep12_24_std_ratio_vs_c0', 1.0):.3f}x | "
            f"{d.get('cep24_40_std_ratio_vs_c0', 1.0):.3f}x | "
            f"**{d.get('cep12_40_std_ratio_vs_c0', 1.0):.3f}x** |"
        )

    lines = [
        "# PyBot-Larynx (`exp_20_pybot_larynx`) Stage 2 细节还原集成验证报告",
        "",
        f"- **判定结论 (`verdict`)**: **`{res['verdict']}`** (`dev_pass={g['dev_pass']}`, `heldout_c3_pass={g['heldout_c3_pass']}`, `heldout_c4_beats_v15={g['heldout_c4_beats_v15']}`, `detail_gates_pass={g['detail_gates_pass']}`, `zero_arbiter_errors={g['zero_arbiter_errors']}`)",
        "- **零神经权重 / 零波形拼接**: `0.0 MB` neural / sklearn / pickle weights，`100%` 电子声带实时合成",
        f"- **Stage 2 知识库 (`knowledge_stage2.npz`)**: `{kb['n_train_utts']}` 条 `TRAIN` 句构建，5-bin onset / 9-bin final `K<=4` 无模糊 Medoid + Viterbi 连续选型 + 上下文锚定 (`{kb['npz_size_kb']} KB`)",
        f"- **知识库命中率**: `DEV_60` 精确 `(o,f,t)` **`{hits_dev['final_exact_oft_rate']*100:.2f}%`** (`place_or_better {hits_dev['final_place_or_better_rate']*100:.2f}%`); `HELDOUT_20` 精确 `(o,f,t)` **`{hits_held['final_exact_oft_rate']*100:.2f}%`** (`place_or_better {hits_held['final_place_or_better_rate']*100:.2f}%`)",
        "",
        "## 1. `DEV_60` 集成结果（中位数，括号内为相对 Stage 1 对应指标变化）",
        "",
        "| 条件 | Mel-LSD (dB) | MCD (dB) | F0 MAE (st) | HNR (dB) | 清音 2-8k/lo (dB) | clicks >8 | CPU RTF | Sanity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    dev_map = {
        "C0_copy_lf": None,
        "Stage2_C2": s1_dev.get("C2_full"),
        "Stage2_C3": s1_dev.get("C3_full"),
        "Stage2_C4": s1_dev.get("C4_rules_e2e"),
    }
    for cname in ("C0_copy_lf", "Stage2_C2", "Stage2_C3", "Stage2_C4"):
        lines.append(_fmt_row(cname, ds[cname], dev_map.get(cname)))

    lines += [
        "",
        "### `DEV_60` 频谱细节门控对比（Stage 2 vs `C0_copy_lf`）",
        "",
        "| 条件 | STFT vp90-10 (1-2.5k / 2.5-5k dB) | Env vp90-10 (1-2.5k / 2.5-5k dB) | UV p90-10 (STFT / Env 1-11k dB) | cep[12:24] vs C0 | cep[24:40] vs C0 | cep[12:40] vs C0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in ("C0_copy_lf", "Stage2_C2", "Stage2_C3", "Stage2_C4"):
        lines.append(_fmt_detail(cname, dd[cname]))

    lines += [
        "",
        "## 2. `HELDOUT_20` 最终集成盲测读出（中位数，仅作最终验证，未做调参）",
        "",
        "| 条件 | Mel-LSD (dB) | MCD (dB) | F0 MAE (st) | HNR (dB) | 清音 2-8k/lo (dB) | clicks >8 | CPU RTF | Sanity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| *参照：`v15_auto` (9,726句拼接)* | *6.8795* | *21.5875* | *3.7870* | *16.32 (10句)* | *8.838* | *0.000600* | *0.4761* | *20/20* |",
        "| *参照：`v15_guided` (参考节奏F0拼接)* | *5.7540* | *20.9761* | *1.3738* | — | — | — | — | *20/20* |",
    ]
    held_map = {
        "C0_copy_lf": s1_held.get("C0_copy_lf"),
        "Stage2_C2": s1_held.get("C2_full"),
        "Stage2_C3": s1_held.get("C3_full"),
        "Stage2_C3_coh": s1_held.get("C3_full_coh"),
        "Stage2_C4": s1_held.get("C4_rules_e2e"),
    }
    for cname in ("C0_copy_lf", "Stage2_C2", "Stage2_C3", "Stage2_C3_coh", "Stage2_C4"):
        lines.append(_fmt_row(cname, hs[cname], held_map.get(cname)))

    lines += [
        "",
        "### `HELDOUT_20` 频谱细节读出",
        "",
        "| 条件 | STFT vp90-10 (1-2.5k / 2.5-5k dB) | Env vp90-10 (1-2.5k / 2.5-5k dB) | UV p90-10 (STFT / Env 1-11k dB) | cep[12:24] vs C0 | cep[24:40] vs C0 | cep[12:40] vs C0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in ("C0_copy_lf", "Stage2_C2", "Stage2_C3", "Stage2_C3_coh", "Stage2_C4"):
        lines.append(_fmt_detail(cname, hd[cname]))

    lines += [
        "",
        f"> 注：`Stage2_C4` 在 `HELDOUT_20` 端到端中位 RTF = **`{hs['Stage2_C4']['medians']['rtf']:.4f}`**（其中电子声带核 `synthesize_larynx` 占 `{res['held_c4_kernel_rtf_median']:.4f}`）；`DEV_60` 端到端中位 RTF = **`{ds['Stage2_C4']['medians']['rtf']:.4f}`**。",
        "",
        "## 3. Arbiter 规则模块归因（`Stage2_C3` on `HELDOUT_20`）",
        "",
        "| 规则模块 | 触发句数 | 命中总帧数 | 清浊覆盖帧 | 均值 `|Δenv|` (dB) | 均值 `|Δbap|` | 均值 `|Δtilt|` | 异常 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mname, a in attr.items():
        lines.append(
            f"| `{mname}` | {a['utts_fired']}/20 | {a['frames_touched_total']} | {a['voiced_overrides_total']} | "
            f"{a['mean_abs_env_delta_db']:.4f} | {a['mean_abs_bap_delta']:.4f} | {a['mean_abs_tilt_delta']:.4f} | {a['errors']} |"
        )

    lines += [
        "",
        "## 4. 试听交付物",
        "- 离线盲听对比台：[listen_stage2.html](listen_stage2.html)（含 `HELDOUT_20` 前 10 句：`Real` / `C0_copy_lf` / `Stage2_C3` / `Stage2_C4` / `v15_auto`，全部统一 `-20 dBFS`）",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage2_verification() -> dict[str, Any]:
    t0 = time.perf_counter()
    if not STAGE2_KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(f"Missing {STAGE2_KNOWLEDGE_PATH}")

    kb_summary = json.loads(STAGE2_SUMMARY_PATH.read_text(encoding="utf-8"))
    s1_res = json.loads(STAGE1_RESULTS_PATH.read_text(encoding="utf-8")) if STAGE1_RESULTS_PATH.exists() else {}

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

    # Warmup frontend/jieba/pypinyin once before timing
    render_rule_auto_utterance("请测试二阶段电子声带。", "warmup_s2", kb2, arbiter=arbiter, rules=rules, seed=1)

    # 1. DEV_60 integrated evaluation
    print(f"Running Stage 2 DEV_60 evaluation ({len(dev_60_ids)} utts)...", flush=True)
    corpus_records, _ = parse_bznsyp_corpus()
    corpus_by_id = {r["id"]: r for r in corpus_records}
    dev_conds = ("C0_copy_lf", "Stage2_C2", "Stage2_C3", "Stage2_C4")
    dev_rows: dict[str, list[dict[str, Any]]] = {c: [] for c in dev_conds}
    dev_det: dict[str, list[dict[str, float | None]]] = {c: [] for c in dev_conds}
    dev_c3_traces: list[dict[str, Any]] = []
    dev_c3_kb_infos: list[dict[str, Any]] = []
    dev_c4_kernel_rtfs: list[float] = []

    for d_idx, uid in enumerate(dev_60_ids):
        dev_meta = dict(idx_doc["train_utt_metas"][uid])
        dev_meta["syllables"] = corpus_by_id[uid]["syllables"]
        item = _prepare_utterance_oracle(uid, dev_meta)
        ref_span = item["ref_span"]
        ref_fast = item["ref_fast"]
        feats_p = item["feats_p"]
        bap6 = item["bap6"]
        ctx = item["ctx"]
        seed_d = 5000 + d_idx

        # C0_copy_lf
        c0 = LarynxControl(f0_hz=feats_p.f0_hz, voiced=feats_p.voiced, log_env=feats_p.log_env, bap=bap6, n_samples=len(ref_span))
        ts = time.perf_counter()
        y0 = synthesize_larynx(c0, glottal_mode="parametric_lf", seed=seed_d)
        dt = time.perf_counter() - ts
        dev_rows["C0_copy_lf"].append(evaluate_utterance(y0, ref_span, rtf=dt / max(1e-6, len(y0) / SR), ref_fast=ref_fast))
        dev_det["C0_copy_lf"].append(_extract_spectral_detail(y0))

        # Stage2_C2
        ts = time.perf_counter()
        c2, _ = build_stage2_control(ctx, feats_p.f0_hz, feats_p.voiced, kb2)
        c2.n_samples = len(ref_span)
        y2 = synthesize_larynx(c2, glottal_mode="parametric_lf", seed=seed_d)
        dt = time.perf_counter() - ts
        dev_rows["Stage2_C2"].append(evaluate_utterance(y2, ref_span, rtf=dt / max(1e-6, len(y2) / SR), ref_fast=ref_fast))
        dev_det["Stage2_C2"].append(_extract_spectral_detail(y2))

        # Stage2_C3
        ts = time.perf_counter()
        c3b, kb_c3 = build_stage2_control(ctx, feats_p.f0_hz, feats_p.voiced, kb2)
        c3b.n_samples = len(ref_span)
        c3f, tr_c3 = arbiter.apply(ctx, c3b, rules)
        y3 = synthesize_larynx(c3f, glottal_mode="parametric_lf", seed=seed_d)
        dt = time.perf_counter() - ts
        dev_rows["Stage2_C3"].append(evaluate_utterance(y3, ref_span, rtf=dt / max(1e-6, len(y3) / SR), ref_fast=ref_fast))
        dev_det["Stage2_C3"].append(_extract_spectral_detail(y3))
        dev_c3_traces.append(tr_c3)
        dev_c3_kb_infos.append(kb_c3)

        # Stage2_C4
        ts = time.perf_counter()
        y4, info4 = render_rule_auto_utterance(item["text"], uid, kb2, arbiter=arbiter, rules=rules, seed=seed_d)
        dt = time.perf_counter() - ts
        dur4 = max(1e-6, len(y4) / SR)
        assert info4["prosody_plan"].get("zero_ml_weights") is True
        assert info4["kb_info"].get("mode") == "stage2"
        dev_c4_kernel_rtfs.append(info4["kernel_s"] / dur4)
        dev_rows["Stage2_C4"].append(evaluate_utterance(y4, ref_span, rtf=dt / dur4, ref_fast=ref_fast))
        dev_det["Stage2_C4"].append(_extract_spectral_detail(y4))

    dev_summary = {c: summarize_condition(dev_rows[c]) for c in dev_conds}
    dev_c0_detail = _summarize_detail(dev_det["C0_copy_lf"])
    dev_detail = {c: _summarize_detail(dev_det[c], None if c == "C0_copy_lf" else dev_c0_detail) for c in dev_conds}

    # 2. Final HELDOUT_20 integrated readout
    print(f"Running Stage 2 HELDOUT_20 final readout ({len(held_20_ids)} utts)...", flush=True)
    held_oracles = [_prepare_utterance_oracle(u, idx_doc["held_utt_metas"][u]) for u in held_20_ids]
    held_conds = ("C0_copy_lf", "Stage2_C2", "Stage2_C3", "Stage2_C3_coh", "Stage2_C4")
    held_rows: dict[str, list[dict[str, Any]]] = {c: [] for c in held_conds}
    held_det: dict[str, list[dict[str, float | None]]] = {c: [] for c in held_conds}
    held_c3_traces: list[dict[str, Any]] = []
    held_c3_kb_infos: list[dict[str, Any]] = []
    held_c4_kernel_rtfs: list[float] = []
    listen_wavs: dict[str, dict[str, np.ndarray]] = {}

    for u_idx, item in enumerate(held_oracles):
        uid = item["uid"]
        ref_span = item["ref_span"]
        ref_fast = item["ref_fast"]
        feats_p = item["feats_p"]
        bap6 = item["bap6"]
        ctx = item["ctx"]
        seed_u = 2000 + u_idx

        # C0_copy_lf
        c0 = LarynxControl(f0_hz=feats_p.f0_hz, voiced=feats_p.voiced, log_env=feats_p.log_env, bap=bap6, n_samples=len(ref_span))
        ts = time.perf_counter()
        y0 = synthesize_larynx(c0, glottal_mode="parametric_lf", seed=seed_u)
        dt = time.perf_counter() - ts
        held_rows["C0_copy_lf"].append(evaluate_utterance(y0, ref_span, rtf=dt / max(1e-6, len(y0) / SR), ref_fast=ref_fast))
        held_det["C0_copy_lf"].append(_extract_spectral_detail(y0))
        if u_idx < 10:
            listen_wavs.setdefault(uid, {})["Real"] = ref_span
            listen_wavs[uid]["C0_copy_lf"] = y0

        # Stage2_C2
        ts = time.perf_counter()
        c2, _ = build_stage2_control(ctx, feats_p.f0_hz, feats_p.voiced, kb2)
        c2.n_samples = len(ref_span)
        y2 = synthesize_larynx(c2, glottal_mode="parametric_lf", seed=seed_u)
        dt = time.perf_counter() - ts
        held_rows["Stage2_C2"].append(evaluate_utterance(y2, ref_span, rtf=dt / max(1e-6, len(y2) / SR), ref_fast=ref_fast))
        held_det["Stage2_C2"].append(_extract_spectral_detail(y2))

        # Stage2_C3 & Stage2_C3_coh
        for cname, gmode in (("Stage2_C3", "parametric_lf"), ("Stage2_C3_coh", "speaker_coherent")):
            ts = time.perf_counter()
            c3b, kb_c3 = build_stage2_control(ctx, feats_p.f0_hz, feats_p.voiced, kb2)
            c3b.n_samples = len(ref_span)
            c3f, tr_c3 = arbiter.apply(ctx, c3b, rules)
            y3 = synthesize_larynx(c3f, glottal_mode=gmode, coherent_phasor=coh_phasor, seed=seed_u)
            dt = time.perf_counter() - ts
            held_rows[cname].append(evaluate_utterance(y3, ref_span, rtf=dt / max(1e-6, len(y3) / SR), ref_fast=ref_fast))
            held_det[cname].append(_extract_spectral_detail(y3))
            if cname == "Stage2_C3":
                held_c3_traces.append(tr_c3)
                held_c3_kb_infos.append(kb_c3)
                if u_idx < 10:
                    listen_wavs[uid]["Stage2_C3"] = y3

        # Stage2_C4
        ts = time.perf_counter()
        y4, info4 = render_rule_auto_utterance(item["text"], uid, kb2, arbiter=arbiter, rules=rules, seed=seed_u)
        dt = time.perf_counter() - ts
        dur4 = max(1e-6, len(y4) / SR)
        assert info4["prosody_plan"].get("zero_ml_weights") is True
        held_c4_kernel_rtfs.append(info4["kernel_s"] / dur4)
        held_rows["Stage2_C4"].append(evaluate_utterance(y4, ref_span, rtf=dt / dur4, ref_fast=ref_fast))
        held_det["Stage2_C4"].append(_extract_spectral_detail(y4))
        if u_idx < 10:
            listen_wavs[uid]["Stage2_C4"] = y4

    held_summary = {c: summarize_condition(held_rows[c]) for c in held_conds}
    held_c0_detail = _summarize_detail(held_det["C0_copy_lf"])
    held_detail = {c: _summarize_detail(held_det[c], None if c == "C0_copy_lf" else held_c0_detail) for c in held_conds}

    # 3. Export -20 dBFS WAVs and listen_stage2.html for HELDOUT_20[:10]
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    listen_groups: list[dict[str, Any]] = []
    v15_samples_dir = REPO_ROOT / "exp_v15_bznsyp_tts" / "samples"
    for idx_l, item in enumerate(held_oracles[:10]):
        uid = item["uid"]
        v15_lbl = V15_10_MAP[uid]
        v15_w, _ = read_wav(v15_samples_dir / f"bz10_{uid}_{v15_lbl}.wav")
        sys_entries = [
            ("Real", "Real 真人原声", listen_wavs[uid]["Real"]),
            ("C0_copy_lf", "C0 电子声带原包络重合成 (LF)", listen_wavs[uid]["C0_copy_lf"]),
            ("Stage2_C3", "Stage2 C3 电子声带 (Oracle节奏F0 + Medoid知识库 + Stage2规则)", listen_wavs[uid]["Stage2_C3"]),
            ("Stage2_C4", "Stage2 C4 端到端零ML规则合成 (文本->语音)", listen_wavs[uid]["Stage2_C4"]),
            ("v15_auto", "v15 Auto (9,726句拼接基线)", v15_w),
        ]
        random.Random(20262 + idx_l).shuffle(sys_entries)
        its = []
        for lbl, (scode, sdesc, w_s) in zip(("A", "B", "C", "D", "E"), sys_entries):
            fname = f"s2_{uid}_{lbl}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fname, _rms_norm(w_s, target_dbfs=-20.0), sr=SR)
            its.append({"label": lbl, "file": f"samples/{fname}", "system_code": scode, "system_desc": sdesc})
        listen_groups.append({"id": f"H{idx_l+1:02d}", "uid": uid, "title": f"H{idx_l+1:02d} ({uid})", "text": item["text"], "items": its})

    _render_listen_stage2_html(listen_groups, LISTEN_STAGE2_PATH)

    # 4. Gates & verdict
    dev_attr = _aggregate_arbiter(dev_c3_traces)
    held_attr = _aggregate_arbiter(held_c3_traces)
    zero_arbiter_errors = (
        all(v["errors"] == 0 for v in dev_attr.values())
        and all(v["errors"] == 0 for v in held_attr.values())
    )

    dc2 = dev_summary["Stage2_C2"]["medians"]
    dc3 = dev_summary["Stage2_C3"]["medians"]
    dc4 = dev_summary["Stage2_C4"]["medians"]
    hc3 = held_summary["Stage2_C3"]["medians"]
    hc4 = held_summary["Stage2_C4"]["medians"]

    s1_dc3_mel = s1_res.get("dev_60_summary", {}).get("C3_full", {}).get("medians", {}).get("mel_lsd_db", 5.0797)
    s1_hc3_mel = s1_res.get("heldout_summary", {}).get("C3_full", {}).get("medians", {}).get("mel_lsd_db", 5.0143)

    dc3_det = dev_detail["Stage2_C3"]
    hc3_det = held_detail["Stage2_C3"]
    detail_gates_pass = (
        dc3_det.get("cep12_40_std_ratio_vs_c0", 0.0) >= 0.70
        and hc3_det.get("cep12_40_std_ratio_vs_c0", 0.0) >= 0.70
        and dc3_det.get("stft_uv_p90_p10_1_11k_db", 0.0) >= 20.0
        and dc3_det.get("stft_v_p90_p10_1_25k_db", 0.0) >= 22.0
    )

    dev_pass = (
        dev_summary["Stage2_C2"]["sanity_ok"] == 60
        and dev_summary["Stage2_C3"]["sanity_ok"] == 60
        and dev_summary["Stage2_C4"]["sanity_ok"] == 60
        and dc3["mel_lsd_db"] < dc2["mel_lsd_db"]
        and dc3["mel_lsd_db"] < s1_dc3_mel
        and dc3["rtf"] <= 0.08
        and dc4["rtf"] <= 0.08
        and dc3["clicks_gt8"] <= 0.0006
    )
    heldout_c3_pass = (
        held_summary["Stage2_C3"]["sanity_ok"] == 20
        and hc3["mel_lsd_db"] <= 5.8
        and hc3["mel_lsd_db"] < s1_hc3_mel
        and hc3["rtf"] <= 0.08
    )
    heldout_c4_beats_v15 = (
        held_summary["Stage2_C4"]["sanity_ok"] == 20
        and hc4["mel_lsd_db"] < 6.8795
        and hc4["rtf"] <= 0.08
    )

    verdict = (
        "GO"
        if (dev_pass and heldout_c3_pass and heldout_c4_beats_v15 and detail_gates_pass and zero_arbiter_errors)
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
            "detail_gates_pass": detail_gates_pass,
            "zero_arbiter_errors": zero_arbiter_errors,
        },
        "knowledge_summary": kb_summary,
        "dev_c3_kb_hits": _aggregate_kb_hits(dev_c3_kb_infos),
        "held_c3_kb_hits": _aggregate_kb_hits(held_c3_kb_infos),
        "dev_c3_arbiter_attribution": dev_attr,
        "held_c3_arbiter_attribution": held_attr,
        "dev_c4_kernel_rtf_median": round(float(np.median(dev_c4_kernel_rtfs)), 4),
        "held_c4_kernel_rtf_median": round(float(np.median(held_c4_kernel_rtfs)), 4),
        "dev_60_summary": dev_summary,
        "dev_60_detail": dev_detail,
        "heldout_summary": held_summary,
        "heldout_detail": held_detail,
        "stage1_dev_60_summary": s1_res.get("dev_60_summary", {}),
        "stage1_heldout_summary": s1_res.get("heldout_summary", {}),
    }

    STAGE2_RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_stage2_report(results, STAGE2_REPORT_PATH)
    print(json.dumps({
        "verdict": verdict,
        "runtime_s": results["runtime_s"],
        "gates": results["gates"],
        "dev_C2_mel": dc2["mel_lsd_db"],
        "dev_C3_mel": dc3["mel_lsd_db"],
        "dev_C4_mel": dc4["mel_lsd_db"],
        "dev_C3_cep12_40_ratio": dc3_det.get("cep12_40_std_ratio_vs_c0"),
        "held_C2_mel": held_summary["Stage2_C2"]["medians"]["mel_lsd_db"],
        "held_C3_mel": hc3["mel_lsd_db"],
        "held_C3_coh_mel": held_summary["Stage2_C3_coh"]["medians"]["mel_lsd_db"],
        "held_C4_mel": hc4["mel_lsd_db"],
        "held_C3_cep12_40_ratio": hc3_det.get("cep12_40_std_ratio_vs_c0"),
        "held_C3_hnr": hc3["hnr_voiced_db"],
        "held_C3_uv2_8k": hc3["uv_2_8k_vs_lo_db"],
        "held_C3_rtf": hc3["rtf"],
        "held_C4_rtf": hc4["rtf"],
    }, ensure_ascii=False, indent=2), flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    run_stage2_verification()
