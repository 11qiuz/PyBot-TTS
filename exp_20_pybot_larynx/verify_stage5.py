"""Stage 5 Integrated Verification & Blind Readout for PyBot-Larynx (`exp_20`)."""
from __future__ import annotations

import argparse
import json
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
from exp_20_pybot_larynx.acoustics.knowledge import (  # noqa: E402
    get_splits,
    load_v15_index_cached,
)
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import (  # noqa: E402
    AcousticKnowledgeStage2,
)
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import (  # noqa: E402
    STAGE5_KNOWLEDGE_PATH,
    STAGE5_SUMMARY_PATH,
    AcousticKnowledgeStage5,
    build_stage5_control,
)
from exp_20_pybot_larynx.arbiter import Arbiter  # noqa: E402
from exp_20_pybot_larynx.frontend.prosody_rules import render_rule_auto_utterance  # noqa: E402
from exp_20_pybot_larynx.frontend.prosody_stage5 import render_stage5_utterance  # noqa: E402
from exp_20_pybot_larynx.judge import evaluate_utterance, summarize_condition  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import LarynxControl, synthesize_larynx  # noqa: E402
from exp_20_pybot_larynx.rules import default_stage0_rules, default_stage5_rules  # noqa: E402
from exp_20_pybot_larynx.stage0_probe import (  # noqa: E402
    V15_10_MAP,
    _aggregate_kb_hits,
    _prepare_utterance_oracle,
)
from exp_20_pybot_larynx.verify_stage4 import (  # noqa: E402
    LONG_DEV5_IDS,
    _aggregate_arbiter,
    _extract_stage4_detail,
    _summarize_detail,
)
from exp_v5_eval.metrics import read_wav  # noqa: E402
from exp_v14_fluent_zh_tts.eval_v14 import _rms_norm, _write_pcm16_wav  # noqa: E402
from exp_v15_bznsyp_tts.index_bznsyp_v15 import (  # noqa: E402
    load_speaker_assets_v15,
    parse_bznsyp_corpus,
)

STAGE4_RESULTS_PATH = EXP_DIR / "stage4_results.json"
STAGE5_RESULTS_PATH = EXP_DIR / "stage5_results.json"
STAGE5_REPORT_PATH = EXP_DIR / "STAGE5_REPORT.md"
LISTEN_STAGE5_PATH = EXP_DIR / "listen_stage5.html"


def _aggregate_s5_hits(kb_infos: list[dict[str, Any]]) -> dict[str, Any]:
    base = _aggregate_kb_hits(kb_infos)
    on_of = sum(int(k.get("hits", {}).get("on_of", 0)) for k in kb_infos)
    on_grp = sum(int(k.get("hits", {}).get("on_grp", 0)) for k in kb_infos)
    on_only = sum(int(k.get("hits", {}).get("on_only", 0)) for k in kb_infos)
    on_fb = sum(int(k.get("hits", {}).get("on_fallback", 0)) for k in kb_infos)
    on_tot = max(1, on_of + on_grp + on_only + on_fb)
    base["onset_of_rate"] = round(on_of / on_tot, 4)
    return base


def _render_listen_stage5_html(groups: list[dict[str, Any]], out_path: Path) -> None:
    payload_json = json.dumps(groups, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-Larynx (exp_20 Stage 5) 追上限盲听对照台</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#eef1f4;background:#11151c;max-width:1040px;margin:1.8rem auto;padding:0 1rem}}
h1{{font-size:1.5rem;margin-bottom:.25rem}}
h2{{font-size:1.15rem;margin:1.4rem 0 .4rem;color:#93c5fd}}
.sub{{color:#9aa6b8;margin-bottom:1.1rem}}
.card{{background:#19202b;border:1px solid #293345;padding:1rem 1.2rem;margin:1rem 0;border-radius:10px}}
.row{{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;padding:.45rem 0;border-bottom:1px dashed #273142}}
.row:last-child{{border-bottom:none}}
audio{{width:min(100%,25rem);height:34px}}
button.btn{{background:#2563eb;color:#fff;border:none;border-radius:6px;padding:.5rem .95rem;cursor:pointer;margin:.4rem .5rem .2rem 0}}
.ans{{display:none;margin-top:.55rem;padding:.55rem .8rem;background:#122033;border-left:3px solid #60a5fa;border-radius:4px;font-size:.92rem;color:#dbeafe}}
.ans.show{{display:block}}
</style>
</head>
<body>
<h1>PyBot-Larynx (exp_20 Stage 5) · 追上限高保真电子声带盲听对照台</h1>
<p class="sub">Stage 5 升级（(o,f) 精化声母库 + K&lt;=8 韵母库 + 方差保留语境/时长/音高迁移 + 三次相位插值 + 6 条专家规则 + 收缩韵律残差）。包含 DEV_60 最长 5 句（L01–L05）与 HELDOUT_20 前 5 句（H01–H05），统一 -20 dBFS 随机打乱。</p>
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


def _write_stage5_report(res: dict[str, Any], out_path: Path) -> None:
    ds = res["dev_60_summary"]
    hs = res["heldout_summary"]
    dd = res["dev_60_detail"]
    hd = res["heldout_detail"]
    kb = res["knowledge_summary"]
    hits_dev = res["dev_c3_kb_hits"]
    hits_held = res["held_c3_kb_hits"]
    attr = res["held_c3_arbiter_attribution"]
    g = res["gates"]

    s4_dev_c3 = ds["Stage4_C3"]["medians"]["mel_lsd_db"]
    s4_dev_c4 = ds["Stage4_C4"]["medians"]["mel_lsd_db"]
    s4_held_c3 = hs["Stage4_C3"]["medians"]["mel_lsd_db"]
    s4_held_c4 = hs["Stage4_C4"]["medians"]["mel_lsd_db"]

    def _fmt_row(cname: str, s: dict[str, Any], d: dict[str, float], ref_mel: float | None = None) -> str:
        m = s["medians"]
        uv_s = f"{m.get('uv_2_8k_vs_lo_db'):.3f}" if m.get("uv_2_8k_vs_lo_db") is not None else "—"
        hnr_s = f"{m.get('hnr_voiced_db'):.2f}" if m.get("hnr_voiced_db") is not None else "—"
        crest_s = f"{d.get('crest_voiced'):.3f}" if d.get("crest_voiced") is not None else "—"
        delta_s = f" ({m['mel_lsd_db'] - ref_mel:+.4f})" if ref_mel is not None else ""
        return (
            f"| `{cname}` | **{m['mel_lsd_db']:.4f}**{delta_s} | {m['mcd_db']:.4f} | {m['f0_mae_st']:.4f} | "
            f"{hnr_s} | {crest_s} | {uv_s} | {m['clicks_gt8']:.6f} | **{m['rtf']:.4f}** | {s['sanity_ok']}/{s['n']} |"
        )

    def _fmt_detail(cname: str, d: dict[str, float]) -> str:
        b1, b2, b3 = d.get("hnr_1_25k_db"), d.get("hnr_25_45k_db"), d.get("hnr_45_7k_db")
        bh = f"{b1:.2f} / {b2:.2f} / {b3:.2f}" if (b1 is not None and b2 is not None and b3 is not None) else "—"
        gstd = f"{d['gain_db_std_median']:.2f}" if d.get("gain_db_std_median") is not None else "—"
        return (
            f"| `{cname}` | {bh} | {d.get('f0_span_p95_p5_st', 0):.2f} | {gstd} | "
            f"{d.get('env_v_p90_p10_1_25k_db', 0):.2f} / {d.get('env_v_p90_p10_25_5k_db', 0):.2f} | "
            f"{d.get('stft_uv_p90_p10_1_11k_db', 0):.2f} / {d.get('env_uv_p90_p10_1_11k_db', 0):.2f} | "
            f"**{d.get('cep12_40_std_ratio_vs_c0', 1.0):.3f}x** |"
        )

    conds = ("Real", "C0_s4", "Stage4_C3", "Stage4_C4", "Stage5_C2", "Stage5_C3", "Stage5_C3_coh", "Stage5_C4", "Stage5_C4_coh")
    dev_ref_map = {
        "Stage5_C2": s4_dev_c3,
        "Stage5_C3": s4_dev_c3,
        "Stage5_C3_coh": s4_dev_c3,
        "Stage5_C4": s4_dev_c4,
        "Stage5_C4_coh": s4_dev_c4,
    }
    held_ref_map = {
        "Stage5_C2": s4_held_c3,
        "Stage5_C3": s4_held_c3,
        "Stage5_C3_coh": s4_held_c3,
        "Stage5_C4": s4_held_c4,
        "Stage5_C4_coh": s4_held_c4,
    }

    lines = [
        "# PyBot-Larynx (`exp_20_pybot_larynx`) Stage 5 追上限集成验证报告",
        "",
        f"- **判定结论 (`verdict`)**: **`{res['verdict']}`** (`dev_pass={g['dev_pass']}`, `heldout_c3_improves={g['heldout_c3_improves']}`, `heldout_c4_improves={g['heldout_c4_improves']}`, `anti_buzz_and_style_gates={g['anti_buzz_and_style_gates']}`, `zero_arbiter_errors={g['zero_arbiter_errors']}`)",
        "- **零神经权重 / 零波形拼接**: `0.0 MB` neural / sklearn / pickle weights，`100%` 电子声带实时合成",
        f"- **Stage 5 知识库 (`knowledge_stage5.npz`)**: `{kb['n_train_utts']}` 条 `TRAIN` 句构建 (`{kb['npz_size_kb']} KB`)，精化 `(o,f)` 声母库 + `K<=8` 韵母库 + 方差保留语境/时长/音高迁移 + 局部 GV 恢复 + 6 条专家规则 + 收缩韵律残差",
        f"- **知识库命中率**: `DEV_60` 精确 `(o,f,t)` **`{hits_dev['final_exact_oft_rate']*100:.2f}%`** (精确声母 `on_of` **`{hits_dev.get('onset_of_rate', 0)*100:.2f}%`**); `HELDOUT_20` 精确 `(o,f,t)` **`{hits_held['final_exact_oft_rate']*100:.2f}%`** (精确声母 `on_of` **`{hits_held.get('onset_of_rate', 0)*100:.2f}%`**)",
        "",
        "## 1. `DEV_60` 集成结果（中位数，括号内为相对 Stage 4 对应 `C3`/`C4` 的 Mel-LSD 变化）",
        "",
        "| 条件 | Mel-LSD (dB) | MCD (dB) | F0 MAE (st) | HNR (dB) | Crest | 清音 2-8k/lo (dB) | clicks >8 | CPU RTF | Sanity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in conds:
        lines.append(_fmt_row(cname, ds[cname], dd[cname], dev_ref_map.get(cname)))

    lines += [
        "",
        "### `DEV_60` Stage 5 频段 HNR / 说话方式动态范围 / 共振峰清晰度",
        "",
        "| 条件 | 1-2.5k / 2.5-4.5k / 4.5-7k HNR (dB) | F0 span p95-p5 (st) | gain_db std (dB) | Env vp90-10 (1-2.5k / 2.5-5k dB) | UV p90-10 (STFT / Env 1-11k dB) | cep[12:40] vs C0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in conds:
        lines.append(_fmt_detail(cname, dd[cname]))

    lines += [
        "",
        "## 2. `HELDOUT_20` 最终集成盲测读出（中位数，严格单次盲测，未做调参）",
        "",
        "| 条件 | Mel-LSD (dB) | MCD (dB) | F0 MAE (st) | HNR (dB) | Crest | 清音 2-8k/lo (dB) | clicks >8 | CPU RTF | Sanity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| *参照：`v15_auto` (9,726句拼接)* | *6.8795* | *21.5875* | *3.7870* | *16.32 (10句)* | — | *8.838* | *0.000600* | *0.4761* | *20/20* |",
    ]
    for cname in conds:
        lines.append(_fmt_row(cname, hs[cname], hd[cname], held_ref_map.get(cname)))

    lines += [
        "",
        "### `HELDOUT_20` Stage 5 频段 HNR / 说话方式动态范围 / 共振峰清晰度",
        "",
        "| 条件 | 1-2.5k / 2.5-4.5k / 4.5-7k HNR (dB) | F0 span p95-p5 (st) | gain_db std (dB) | Env vp90-10 (1-2.5k / 2.5-5k dB) | UV p90-10 (STFT / Env 1-11k dB) | cep[12:40] vs C0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in conds:
        lines.append(_fmt_detail(cname, hd[cname]))

    lines += [
        "",
        f"> 注：`Stage5_C4` 在 `HELDOUT_20` 端到端中位 RTF = **`{hs['Stage5_C4']['medians']['rtf']:.4f}`**（其中电子声带核占 `{res['held_c4_kernel_rtf_median']:.4f}`）；`DEV_60` 端到端中位 RTF = **`{ds['Stage5_C4']['medians']['rtf']:.4f}`**。",
        "",
        "## 3. Arbiter 6 模块归因（`Stage5_C3` on `HELDOUT_20`）",
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
        "- 离线盲听对比台：[listen_stage5.html](listen_stage5.html)（含 `DEV_60` 最长 5 句与 `HELDOUT_20` 前 5 句：`Real` / `Stage4_C4` / `Stage5_C3` / `Stage5_C4` / `Stage5_C4_coh` / `v15_auto`）",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _eval_split_stage5(
    split_name: str,
    oracles: list[dict[str, Any]],
    kb2: AcousticKnowledgeStage2,
    kb5: AcousticKnowledgeStage5,
    arbiter: Arbiter,
    rules4: list[Any],
    rules5: list[Any],
    coh_phasor: np.ndarray,
    *,
    seed_base: int,
    capture_uids: set[str],
    s5_params: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[float], dict[str, dict[str, np.ndarray]]]:
    conds = ("Real", "C0_s4", "Stage4_C3", "Stage4_C4", "Stage5_C2", "Stage5_C3", "Stage5_C3_coh", "Stage5_C4", "Stage5_C4_coh")
    rows: dict[str, list[dict[str, Any]]] = {c: [] for c in conds}
    det: dict[str, list[dict[str, float | None]]] = {c: [] for c in conds}
    c3_traces: list[dict[str, Any]] = []
    c3_kb_infos: list[dict[str, Any]] = []
    c4_kernel_rtfs: list[float] = []
    c4_gain_stds: dict[str, list[float]] = {"Stage4_C4": [], "Stage5_C4": [], "Stage5_C4_coh": []}
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

        rows["Real"].append(evaluate_utterance(ref_span, ref_span, rtf=0.0, ref_fast=ref_fast))
        det["Real"].append(_extract_stage4_detail(ref_span))
        if want:
            captured.setdefault(uid, {})["Real"] = ref_span

        c0 = LarynxControl(f0_hz=feats_p.f0_hz, voiced=feats_p.voiced, log_env=feats_p.log_env, bap=bap6, n_samples=len(ref_span))
        ts = time.perf_counter()
        y0 = synthesize_larynx(c0, glottal_mode="parametric_lf", coherent_phasor=coh_phasor, seed=seed_u)
        dt = time.perf_counter() - ts
        rows["C0_s4"].append(evaluate_utterance(y0, ref_span, rtf=dt / max(1e-6, len(y0) / SR), ref_fast=ref_fast))
        det["C0_s4"].append(_extract_stage4_detail(y0))

        # Stage4_C3 & Stage4_C4 baselines on exact same oracle & seed
        from exp_20_pybot_larynx.acoustics.knowledge_stage2 import build_stage2_control
        ts = time.perf_counter()
        c4_b, _ = build_stage2_control(ctx, feats_p.f0_hz, feats_p.voiced, kb2)
        c4_b.n_samples = len(ref_span)
        c4_f, _ = arbiter.apply(ctx, c4_b, rules4)
        y43 = synthesize_larynx(c4_f, glottal_mode="parametric_lf", seed=seed_u)
        dt43 = time.perf_counter() - ts
        rows["Stage4_C3"].append(evaluate_utterance(y43, ref_span, rtf=dt43 / max(1e-6, len(y43) / SR), ref_fast=ref_fast))
        det["Stage4_C3"].append(_extract_stage4_detail(y43))

        ts = time.perf_counter()
        y44, info44 = render_rule_auto_utterance(
            item["text"], uid, kb2, arbiter=arbiter, rules=rules4, glottal_mode="parametric_lf", coherent_phasor=coh_phasor, seed=seed_u
        )
        dt44 = time.perf_counter() - ts
        rows["Stage4_C4"].append(evaluate_utterance(y44, ref_span, rtf=dt44 / max(1e-6, len(y44) / SR), ref_fast=ref_fast))
        det["Stage4_C4"].append(_extract_stage4_detail(y44))
        g44 = [float(u.get("gain_db", 0.0)) for u in info44["prosody_plan"]["units"]]
        if len(g44) >= 2:
            c4_gain_stds["Stage4_C4"].append(float(np.std(g44)))
        if want:
            captured[uid]["Stage4_C4"] = y44

        # Stage5_C2
        ts = time.perf_counter()
        c5_2, _ = build_stage5_control(ctx, feats_p.f0_hz, feats_p.voiced, kb5, **s5_params)
        c5_2.n_samples = len(ref_span)
        y52 = synthesize_larynx(c5_2, glottal_mode="parametric_lf", seed=seed_u)
        dt52 = time.perf_counter() - ts
        rows["Stage5_C2"].append(evaluate_utterance(y52, ref_span, rtf=dt52 / max(1e-6, len(y52) / SR), ref_fast=ref_fast))
        det["Stage5_C2"].append(_extract_stage4_detail(y52))

        # Stage5_C3 & Stage5_C3_coh
        for cname, gmode in (("Stage5_C3", "parametric_lf"), ("Stage5_C3_coh", "speaker_coherent")):
            ts = time.perf_counter()
            c5_b, kb_c3 = build_stage5_control(ctx, feats_p.f0_hz, feats_p.voiced, kb5, **s5_params)
            c5_b.n_samples = len(ref_span)
            c5_f, tr_c3 = arbiter.apply(ctx, c5_b, rules5)
            y53 = synthesize_larynx(c5_f, glottal_mode=gmode, coherent_phasor=coh_phasor, seed=seed_u)
            dt53 = time.perf_counter() - ts
            rows[cname].append(evaluate_utterance(y53, ref_span, rtf=dt53 / max(1e-6, len(y53) / SR), ref_fast=ref_fast))
            det[cname].append(_extract_stage4_detail(y53))
            if cname == "Stage5_C3":
                c3_traces.append(tr_c3)
                c3_kb_infos.append(kb_c3)
            if want:
                captured[uid][cname] = y53

        # Stage5_C4 & Stage5_C4_coh
        for cname, gmode in (("Stage5_C4", "parametric_lf"), ("Stage5_C4_coh", "speaker_coherent")):
            ts = time.perf_counter()
            y54, info54 = render_stage5_utterance(
                item["text"],
                uid,
                kb5,
                arbiter=arbiter,
                rules=rules5,
                glottal_mode=gmode,
                coherent_phasor=coh_phasor,
                seed=seed_u,
            )
            dt54 = time.perf_counter() - ts
            dur54 = max(1e-6, len(y54) / SR)
            if cname == "Stage5_C4":
                c4_kernel_rtfs.append(info54["kernel_s"] / dur54)
            g54 = [float(u.get("gain_db", 0.0)) for u in info54["prosody_plan"]["units"]]
            if len(g54) >= 2:
                c4_gain_stds[cname].append(float(np.std(g54)))
            rows[cname].append(evaluate_utterance(y54, ref_span, rtf=dt54 / dur54, ref_fast=ref_fast))
            det[cname].append(_extract_stage4_detail(y54))
            if want:
                captured[uid][cname] = y54

        if (u_idx + 1) % 10 == 0 or (u_idx + 1) == len(oracles):
            print(f"[{split_name}] {u_idx + 1}/{len(oracles)} done", flush=True)

    summary = {c: summarize_condition(rows[c]) for c in conds}
    c0_det = _summarize_detail(det["C0_s4"])
    detail = {
        c: _summarize_detail(det[c], None if c == "C0_s4" else c0_det, c4_gain_stds.get(c))
        for c in conds
    }
    return summary, detail, c3_traces, c3_kb_infos, c4_kernel_rtfs, captured


def run_stage5_verification() -> dict[str, Any]:
    t0 = time.perf_counter()
    if not STAGE5_KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(f"Missing {STAGE5_KNOWLEDGE_PATH}")

    kb_summary = json.loads(STAGE5_SUMMARY_PATH.read_text(encoding="utf-8"))
    kb2 = AcousticKnowledgeStage2.load()
    kb5 = AcousticKnowledgeStage5.load(STAGE5_KNOWLEDGE_PATH)
    idx_doc = load_v15_index_cached()
    assets = load_speaker_assets_v15()
    coh_phasor = assets["glottal_bank"].coherent_phasor
    arbiter = Arbiter()
    rules4 = default_stage0_rules()
    rules5 = default_stage5_rules()

    train_ids, dev_60_ids, held_20_ids = get_splits()
    assert not (set(train_ids) & set(dev_60_ids))
    assert not (set(train_ids) & set(held_20_ids))
    assert not (set(dev_60_ids) & set(held_20_ids))

    # Warmup frontend + Stage 5 LUTs once before timing
    render_stage5_utterance("请测试五阶段电子声带。", "warmup_s5", kb5, arbiter=arbiter, rules=rules5, seed=1)

    corpus_records, _ = parse_bznsyp_corpus()
    corpus_by_id = {r["id"]: r for r in corpus_records}

    s5_params: dict[str, float] = {}

    print(f"Preparing DEV_60 ({len(dev_60_ids)} utts)...", flush=True)
    dev_oracles = []
    for uid in dev_60_ids:
        dm = dict(idx_doc["train_utt_metas"][uid])
        dm["syllables"] = corpus_by_id[uid]["syllables"]
        dev_oracles.append(_prepare_utterance_oracle(uid, dm))

    dev_summary, dev_detail, dev_c3_traces, dev_c3_kb_infos, dev_c4_k_rtfs, dev_captured = _eval_split_stage5(
        "DEV_60",
        dev_oracles,
        kb2,
        kb5,
        arbiter,
        rules4,
        rules5,
        coh_phasor,
        seed_base=5000,
        capture_uids=set(LONG_DEV5_IDS),
        s5_params=s5_params,
    )

    print(f"Preparing HELDOUT_20 ({len(held_20_ids)} utts)...", flush=True)
    held_oracles = [_prepare_utterance_oracle(u, idx_doc["held_utt_metas"][u]) for u in held_20_ids]
    held_5_uids = [it["uid"] for it in held_oracles[:5]]
    held_summary, held_detail, held_c3_traces, held_c3_kb_infos, held_c4_k_rtfs, held_captured = _eval_split_stage5(
        "HELDOUT_20",
        held_oracles,
        kb2,
        kb5,
        arbiter,
        rules4,
        rules5,
        coh_phasor,
        seed_base=2000,
        capture_uids=set(held_5_uids),
        s5_params=s5_params,
    )

    # Export -20 dBFS blind listening WAVs and listen_stage5.html
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    listen_groups: list[dict[str, Any]] = []
    dev_by_uid = {it["uid"]: it for it in dev_oracles}
    for idx_l, uid in enumerate(LONG_DEV5_IDS):
        item = dev_by_uid[uid]
        wmap = dev_captured[uid]
        sys_entries = [
            ("Real", "Real 真人原声", wmap["Real"]),
            ("Stage4_C4", "Stage4 C4 端到端基线", wmap["Stage4_C4"]),
            ("Stage5_C3", "Stage5 C3 (Oracle节奏F0 + K<=8精化声库 + 6规则)", wmap["Stage5_C3"]),
            ("Stage5_C4", "Stage5 C4 端到端追上限 (analytic LF)", wmap["Stage5_C4"]),
            ("Stage5_C4_coh", "Stage5 C4_coh 端到端追上限 (相干胸声相位)", wmap["Stage5_C4_coh"]),
        ]
        random.Random(202650 + idx_l).shuffle(sys_entries)
        its = []
        for lbl, (scode, sdesc, w_s) in zip(("A", "B", "C", "D", "E"), sys_entries):
            fname = f"s5_dev_{uid}_{lbl}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fname, _rms_norm(w_s, target_dbfs=-20.0), sr=SR)
            its.append({"label": lbl, "file": f"samples/{fname}", "system_code": scode, "system_desc": sdesc})
        listen_groups.append({
            "section": "第一部分：DEV_60 最长 5 句盲听对照（重点听 Stage5 vs Stage4 清晰度与韵律提升）",
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
            ("Stage4_C4", "Stage4 C4 端到端基线", wmap["Stage4_C4"]),
            ("Stage5_C3", "Stage5 C3 (Oracle节奏F0 + Stage5声库 + 6规则)", wmap["Stage5_C3"]),
            ("Stage5_C4", "Stage5 C4 端到端追上限 (analytic LF)", wmap["Stage5_C4"]),
            ("Stage5_C4_coh", "Stage5 C4_coh 端到端追上限 (相干胸声相位)", wmap["Stage5_C4_coh"]),
            ("v15_auto", "v15 Auto (9,726句拼接基线)", v15_w),
        ]
        random.Random(202651 + idx_h).shuffle(sys_entries)
        its = []
        for lbl, (scode, sdesc, w_s) in zip(("A", "B", "C", "D", "E", "F"), sys_entries):
            fname = f"s5_held_{uid}_{lbl}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fname, _rms_norm(w_s, target_dbfs=-20.0), sr=SR)
            its.append({"label": lbl, "file": f"samples/{fname}", "system_code": scode, "system_desc": sdesc})
        listen_groups.append({
            "section": "第二部分：HELDOUT_20 盲测前 5 句对照（含 Stage4_C4 与 v15_auto）",
            "id": f"H{idx_h+1:02d}",
            "uid": uid,
            "title": f"H{idx_h+1:02d} ({uid})",
            "text": item["text"],
            "items": its,
        })

    _render_listen_stage5_html(listen_groups, LISTEN_STAGE5_PATH)

    dev_attr = _aggregate_arbiter(dev_c3_traces)
    held_attr = _aggregate_arbiter(held_c3_traces)
    zero_arbiter_errors = (
        all(v["errors"] == 0 for v in dev_attr.values())
        and all(v["errors"] == 0 for v in held_attr.values())
    )

    d43 = dev_summary["Stage4_C3"]["medians"]
    d44 = dev_summary["Stage4_C4"]["medians"]
    d53 = dev_summary["Stage5_C3"]["medians"]
    d54 = dev_summary["Stage5_C4"]["medians"]

    h43 = held_summary["Stage4_C3"]["medians"]
    h44 = held_summary["Stage4_C4"]["medians"]
    h53 = held_summary["Stage5_C3"]["medians"]
    h54 = held_summary["Stage5_C4"]["medians"]

    d53_det = dev_detail["Stage5_C3"]
    d54_det = dev_detail["Stage5_C4"]
    h53_det = held_detail["Stage5_C3"]

    dev_pass = (
        dev_summary["Stage5_C2"]["sanity_ok"] == 60
        and dev_summary["Stage5_C3"]["sanity_ok"] == 60
        and dev_summary["Stage5_C3_coh"]["sanity_ok"] == 60
        and dev_summary["Stage5_C4"]["sanity_ok"] == 60
        and dev_summary["Stage5_C4_coh"]["sanity_ok"] == 60
        and d53["mel_lsd_db"] < d43["mel_lsd_db"]
        and d53["mcd_db"] < d43["mcd_db"]
        and d54["mel_lsd_db"] < d44["mel_lsd_db"]
        and d54["f0_mae_st"] < d44["f0_mae_st"]
        and d53["rtf"] <= 0.055
        and d54["rtf"] <= 0.055
        and d53["clicks_gt8"] <= 0.0007
        and d54["clicks_gt8"] <= 0.0007
    )
    heldout_c3_improves = (
        held_summary["Stage5_C3"]["sanity_ok"] == 20
        and held_summary["Stage5_C3_coh"]["sanity_ok"] == 20
        and h53["mel_lsd_db"] < h43["mel_lsd_db"]
        and h53["mcd_db"] < h43["mcd_db"]
        and h53["rtf"] <= 0.055
        and h53["clicks_gt8"] <= 0.0007
    )
    heldout_c4_improves = (
        held_summary["Stage5_C4"]["sanity_ok"] == 20
        and held_summary["Stage5_C4_coh"]["sanity_ok"] == 20
        and h54["mel_lsd_db"] < h44["mel_lsd_db"]
        and h54["mcd_db"] < h44["mcd_db"]
        and h54["rtf"] <= 0.055
        and h54["clicks_gt8"] <= 0.0007
    )
    anti_buzz_and_style_gates = (
        2.35 <= d53_det.get("crest_voiced", 0.0) <= 2.80
        and 2.35 <= d54_det.get("crest_voiced", 0.0) <= 2.80
        and 18.5 <= d53["hnr_voiced_db"] <= 23.0
        and 18.5 <= h53["hnr_voiced_db"] <= 23.0
        and d53_det.get("cep12_40_std_ratio_vs_c0", 0.0) >= 0.85
        and h53_det.get("cep12_40_std_ratio_vs_c0", 0.0) >= 0.85
    )

    verdict = (
        "GO"
        if (dev_pass and heldout_c3_improves and heldout_c4_improves and anti_buzz_and_style_gates and zero_arbiter_errors)
        else "NO_GO"
    )

    results = {
        "runtime_s": round(time.perf_counter() - t0, 2),
        "verdict": verdict,
        "zero_ml_weights": True,
        "gates": {
            "dev_pass": dev_pass,
            "heldout_c3_improves": heldout_c3_improves,
            "heldout_c4_improves": heldout_c4_improves,
            "anti_buzz_and_style_gates": anti_buzz_and_style_gates,
            "zero_arbiter_errors": zero_arbiter_errors,
        },
        "knowledge_summary": kb_summary,
        "dev_c3_kb_hits": _aggregate_s5_hits(dev_c3_kb_infos),
        "held_c3_kb_hits": _aggregate_s5_hits(held_c3_kb_infos),
        "dev_c3_arbiter_attribution": dev_attr,
        "held_c3_arbiter_attribution": held_attr,
        "dev_c4_kernel_rtf_median": round(float(np.median(dev_c4_k_rtfs)), 4),
        "held_c4_kernel_rtf_median": round(float(np.median(held_c4_k_rtfs)), 4),
        "dev_60_summary": dev_summary,
        "dev_60_detail": dev_detail,
        "heldout_summary": held_summary,
        "heldout_detail": held_detail,
    }

    STAGE5_RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_stage5_report(results, STAGE5_REPORT_PATH)
    print(json.dumps({
        "verdict": verdict,
        "runtime_s": results["runtime_s"],
        "gates": results["gates"],
        "dev_s4_c3_mel": d43["mel_lsd_db"],
        "dev_s5_c3_mel": d53["mel_lsd_db"],
        "dev_s4_c4_mel": d44["mel_lsd_db"],
        "dev_s5_c4_mel": d54["mel_lsd_db"],
        "dev_s4_c4_f0": d44["f0_mae_st"],
        "dev_s5_c4_f0": d54["f0_mae_st"],
        "held_s4_c3_mel": h43["mel_lsd_db"],
        "held_s5_c3_mel": h53["mel_lsd_db"],
        "held_s4_c4_mel": h44["mel_lsd_db"],
        "held_s5_c4_mel": h54["mel_lsd_db"],
        "held_s5_c3_cep_ratio": h53_det.get("cep12_40_std_ratio_vs_c0"),
        "held_s5_c4_rtf": h54["rtf"],
    }, ensure_ascii=False, indent=2), flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    run_stage5_verification()
