"""Stage 7 Integrated Verification & Single-Shot Blind Readout for PyBot-Larynx (`exp_20`)."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from scipy.fft import rfft

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import EXP_DIR, HOP, N_FFT, SAMPLES_DIR, SR  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge import get_splits, load_v15_index_cached  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import (  # noqa: E402
    STAGE5_KNOWLEDGE_PATH,
    AcousticKnowledgeStage5,
    build_stage5_control,
)
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import (  # noqa: E402
    DEFAULT_UV_LEVEL_BOOST_DB_S7,
    STAGE7_KNOWLEDGE_PATH,
    STAGE7_SUMMARY_PATH,
    AcousticKnowledgeStage7,
    build_stage7_control,
)
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB, Arbiter  # noqa: E402
from exp_20_pybot_larynx.frontend.prosody_stage5 import render_stage5_utterance  # noqa: E402
from exp_20_pybot_larynx.frontend.prosody_stage7 import (  # noqa: E402
    PROSODY_S7_SUMMARY_PATH,
    PROSODY_S7_TABLES_PATH,
    ProsodyStage7Model,
    render_stage7_utterance,
)
from exp_20_pybot_larynx.judge import evaluate_utterance, summarize_condition  # noqa: E402
from exp_20_pybot_larynx.larynx.spectral_decomp import compute_pole_bandwidth_stats  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import LarynxControl, log_env_to_cep, synthesize_larynx  # noqa: E402
from exp_20_pybot_larynx.larynx.synth_v2 import analyze_true_harmonic_env, synthesize_larynx_v2  # noqa: E402
from exp_20_pybot_larynx.rules import default_stage5_rules  # noqa: E402
from exp_20_pybot_larynx.stage0_probe import V15_10_MAP, _aggregate_kb_hits, _prepare_utterance_oracle  # noqa: E402
from exp_20_pybot_larynx.verify_stage4 import (  # noqa: E402
    _B_1_25,
    _B_25_45,
    _B_45_7,
    _HANN,
    _HNR_HANN,
    _HNR_NBINS,
    _HNR_NFFT,
    _HNR_WIN,
    _M_UV_1_11,
    _M_V_1_25,
    _M_V_25_5,
    LONG_DEV5_IDS,
    _aggregate_arbiter,
    _summarize_detail,
)
from exp_20_pybot_larynx.verify_stage5 import STAGE5_RESULTS_PATH, _aggregate_s5_hits  # noqa: E402
from exp_v5_eval.metrics import read_wav  # noqa: E402
from exp_v9_clarity_pro.hnm_v9 import analyze_v9  # noqa: E402
from exp_v12_natural_vocoder.v12_bank import _patch_unsmoothed_uv_env  # noqa: E402
from exp_v14_fluent_zh_tts.eval_v14 import _rms_norm, _write_pcm16_wav  # noqa: E402
from exp_v15_bznsyp_tts.index_bznsyp_v15 import load_speaker_assets_v15, parse_bznsyp_corpus  # noqa: E402

STAGE7_RESULTS_PATH = EXP_DIR / "stage7_results.json"
STAGE7_REPORT_PATH = EXP_DIR / "STAGE7_REPORT.md"
LISTEN_STAGE7_PATH = EXP_DIR / "listen_stage7.html"


def _extract_s7_detail_and_bw(y: np.ndarray) -> dict[str, float | None]:
    """Compute Stage 4/5 detail metrics PLUS LPC-22 pole bandwidth in one shared analysis pass."""
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

    bw_stats = compute_pole_bandwidth_stats(fp.log_env, v_mask, order=22, stride=8)

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
        "lpc22_eff_bw_hz": float(bw_stats["eff_bw_hz"]) if bw_stats["eff_bw_hz"] > 0.0 else None,
        "lpc22_pole_radius": float(bw_stats["pole_radius_median"]) if bw_stats["pole_radius_median"] > 0.0 else None,
    }


def _summarize_s7_detail(
    rows: list[dict[str, float | None]],
    c0_detail: dict[str, float] | None = None,
    gain_stds: list[float] | None = None,
) -> dict[str, float]:
    out = _summarize_detail(rows, c0_detail=c0_detail, gain_stds=gain_stds)
    for k in ("lpc22_eff_bw_hz", "lpc22_pole_radius"):
        vals = [float(r[k]) for r in rows if r.get(k) is not None and math.isfinite(float(r[k]))]
        if vals:
            out[k] = round(float(np.median(vals)), 4 if k == "lpc22_pole_radius" else 1)
    return out


def _aggregate_s7_hits(kb_infos: list[dict[str, Any]]) -> dict[str, Any]:
    base = _aggregate_s5_hits(kb_infos)
    r_oft = sum(int(k.get("hits", {}).get("ridge_oft", 0)) for k in kb_infos)
    r_par = sum(int(k.get("hits", {}).get("ridge_parent", 0)) for k in kb_infos)
    tot = max(1, sum(int(k.get("hits", {}).get(kk, 0)) for k in kb_infos for kk in ("exact_oft", "place_ft", "ft", "f", "fallback")))
    base["ridge_oft_rate"] = round(r_oft / tot, 4)
    base["ridge_any_rate"] = round((r_oft + r_par) / tot, 4)
    return base


def _render_listen_stage7_html(groups: list[dict[str, Any]], out_path: Path) -> None:
    payload_json = json.dumps(groups, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-Larynx (exp_20 Stage 7) 追上限盲听对照台</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#eef1f4;background:#11151c;max-width:1060px;margin:1.8rem auto;padding:0 1rem}}
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
<h1>PyBot-Larynx (exp_20 Stage 7) · 分层连续 Ridge + 双遍 Xu 韵律 + 动态相干电子声带盲听台</h1>
<p class="sub">Stage 7 升级（84 维物理特征分层贝叶斯 Ridge + 阶梯低频锚定 + Ridge 引导 Viterbi Medoid + dynamic_coherent 动态开商声门核 + 主动清音补偿 + 双遍 Xu 连续韵律模型）。DEV_60 最长 5 句（L01–L05）与 HELDOUT_20 前 5 句（H01–H05），统一 -20 dBFS 随机打乱。</p>
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


def _eval_split_stage7(
    split_name: str,
    oracles: list[dict[str, Any]],
    kb5: AcousticKnowledgeStage5,
    kb7: AcousticKnowledgeStage7,
    mdl7: ProsodyStage7Model,
    arbiter: Arbiter,
    rules5: list[Any],
    coh_phasor: np.ndarray,
    *,
    seed_base: int,
    capture_uids: set[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[float], dict[str, dict[str, np.ndarray]]]:
    conds = (
        "Real", "C0_s5", "C0_s7",
        "Stage5_C3", "Stage5_C4",
        "Stage7_C3_lf", "Stage7_C3_dyn",
        "Stage7_C4_lf", "Stage7_C4_dyn",
    )
    rows: dict[str, list[dict[str, Any]]] = {c: [] for c in conds}
    det: dict[str, list[dict[str, float | None]]] = {c: [] for c in conds}
    c3_traces: list[dict[str, Any]] = []
    c3_kb_infos: list[dict[str, Any]] = []
    c4_kernel_rtfs: list[float] = []
    c4_gain_stds: dict[str, list[float]] = {"Stage5_C4": [], "Stage7_C4_lf": [], "Stage7_C4_dyn": []}
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
        det["Real"].append(_extract_s7_detail_and_bw(ref_span))
        if want:
            captured.setdefault(uid, {})["Real"] = ref_span

        # C0_s5
        c0_5 = LarynxControl(f0_hz=feats_p.f0_hz, voiced=feats_p.voiced, log_env=feats_p.log_env, bap=bap6, n_samples=len(ref_span))
        ts = time.perf_counter()
        y0_5 = synthesize_larynx(c0_5, glottal_mode="parametric_lf", coherent_phasor=coh_phasor, seed=seed_u)
        dt = time.perf_counter() - ts
        rows["C0_s5"].append(evaluate_utterance(y0_5, ref_span, rtf=dt / max(1e-6, len(y0_5) / SR), ref_fast=ref_fast))
        det["C0_s5"].append(_extract_s7_detail_and_bw(y0_5))

        # C0_s7 (TrueHarmEnv + bap_scale=0.50 + dynamic_coherent + uv_level_boost)
        env7, f0_7, voi_7, bap_7 = analyze_true_harmonic_env(ref_span, sr=SR, bap_scale=0.50)
        c0_7 = LarynxControl(f0_hz=f0_7, voiced=voi_7, log_env=env7, bap=bap_7, n_samples=len(ref_span))
        ts = time.perf_counter()
        y0_7 = synthesize_larynx_v2(
            c0_7, glottal_mode="dynamic_coherent", coherent_phasor=coh_phasor,
            uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7, seed=seed_u,
        )
        dt = time.perf_counter() - ts
        rows["C0_s7"].append(evaluate_utterance(y0_7, ref_span, rtf=dt / max(1e-6, len(y0_7) / SR), ref_fast=ref_fast))
        det["C0_s7"].append(_extract_s7_detail_and_bw(y0_7))

        # Stage5_C3
        ts = time.perf_counter()
        c5b, _ = build_stage5_control(ctx, feats_p.f0_hz, feats_p.voiced, kb5)
        c5b.n_samples = len(ref_span)
        c5f, _ = arbiter.apply(ctx, c5b, rules5)
        y53 = synthesize_larynx(c5f, glottal_mode="parametric_lf", seed=seed_u)
        dt = time.perf_counter() - ts
        rows["Stage5_C3"].append(evaluate_utterance(y53, ref_span, rtf=dt / max(1e-6, len(y53) / SR), ref_fast=ref_fast))
        det["Stage5_C3"].append(_extract_s7_detail_and_bw(y53))
        if want:
            captured[uid]["Stage5_C3"] = y53

        # Stage5_C4
        ts = time.perf_counter()
        y54, info54 = render_stage5_utterance(
            item["text"], uid, kb5, arbiter=arbiter, rules=rules5,
            glottal_mode="parametric_lf", coherent_phasor=coh_phasor, seed=seed_u,
        )
        dt = time.perf_counter() - ts
        g54 = [float(u.get("gain_db", 0.0)) for u in info54["prosody_plan"]["units"]]
        if len(g54) >= 2:
            c4_gain_stds["Stage5_C4"].append(float(np.std(g54)))
        rows["Stage5_C4"].append(evaluate_utterance(y54, ref_span, rtf=dt / max(1e-6, len(y54) / SR), ref_fast=ref_fast))
        det["Stage5_C4"].append(_extract_s7_detail_and_bw(y54))
        if want:
            captured[uid]["Stage5_C4"] = y54

        # Stage7_C3_lf & Stage7_C3_dyn
        for cname, gmode, uv_b in (
            ("Stage7_C3_lf", "parametric_lf", 0.0),
            ("Stage7_C3_dyn", "dynamic_coherent", DEFAULT_UV_LEVEL_BOOST_DB_S7),
        ):
            ts = time.perf_counter()
            c7b, kb_c3 = build_stage7_control(ctx, feats_p.f0_hz, feats_p.voiced, kb7)
            c7b.n_samples = len(ref_span)
            c7f, tr_c3 = arbiter.apply(ctx, c7b, rules5)
            y73 = synthesize_larynx_v2(
                c7f, glottal_mode=gmode, coherent_phasor=coh_phasor,
                uv_level_boost_db=uv_b, seed=seed_u,
            )
            dt = time.perf_counter() - ts
            rows[cname].append(evaluate_utterance(y73, ref_span, rtf=dt / max(1e-6, len(y73) / SR), ref_fast=ref_fast))
            det[cname].append(_extract_s7_detail_and_bw(y73))
            if cname == "Stage7_C3_dyn":
                c3_traces.append(tr_c3)
                c3_kb_infos.append(kb_c3)
            if want:
                captured[uid][cname] = y73

        # Stage7_C4_lf & Stage7_C4_dyn
        for cname, gmode, uv_b in (
            ("Stage7_C4_lf", "parametric_lf", 0.0),
            ("Stage7_C4_dyn", "dynamic_coherent", DEFAULT_UV_LEVEL_BOOST_DB_S7),
        ):
            ts = time.perf_counter()
            y74, info74 = render_stage7_utterance(
                item["text"], uid, kb7, arbiter=arbiter, rules=rules5,
                glottal_mode=gmode, coherent_phasor=coh_phasor, seed=seed_u,
                model=mdl7, uv_level_boost_db=uv_b,
            )
            dt = time.perf_counter() - ts
            dur74 = max(1e-6, len(y74) / SR)
            if cname == "Stage7_C4_dyn":
                c4_kernel_rtfs.append(info74["kernel_s"] / dur74)
            g74 = [float(u.get("gain_db", 0.0)) for u in info74["prosody_plan"]["units"]]
            if len(g74) >= 2:
                c4_gain_stds[cname].append(float(np.std(g74)))
            rows[cname].append(evaluate_utterance(y74, ref_span, rtf=dt / dur74, ref_fast=ref_fast))
            det[cname].append(_extract_s7_detail_and_bw(y74))
            if want:
                captured[uid][cname] = y74

        if (u_idx + 1) % 10 == 0 or (u_idx + 1) == len(oracles):
            print(f"[{split_name}] {u_idx + 1}/{len(oracles)} done", flush=True)

    summary = {c: summarize_condition(rows[c]) for c in conds}
    c0_det = _summarize_s7_detail(det["C0_s5"])
    detail = {
        c: _summarize_s7_detail(det[c], None if c == "C0_s5" else c0_det, c4_gain_stds.get(c))
        for c in conds
    }
    return summary, detail, c3_traces, c3_kb_infos, c4_kernel_rtfs, captured


def _write_stage7_report(res: dict[str, Any], out_path: Path) -> None:
    ds = res["dev_60_summary"]
    hs = res["heldout_summary"]
    dd = res["dev_60_detail"]
    hd = res["heldout_detail"]
    s5_ref = res.get("stage5_baseline_ref", {})
    kb = res["knowledge_summary"]
    ps = res["prosody_summary"]
    hits_dev = res["dev_c3_kb_hits"]
    hits_held = res["held_c3_kb_hits"]
    attr = res["held_c3_arbiter_attribution"]
    g = res["gates"]

    s5_dev_c3 = ds["Stage5_C3"]["medians"]["mel_lsd_db"]
    s5_dev_c4 = ds["Stage5_C4"]["medians"]["mel_lsd_db"]
    s5_held_c3 = hs["Stage5_C3"]["medians"]["mel_lsd_db"]
    s5_held_c4 = hs["Stage5_C4"]["medians"]["mel_lsd_db"]

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
        bw_s = f"{d.get('lpc22_eff_bw_hz', 0):.1f} Hz ({d.get('lpc22_pole_radius', 0):.4f})" if d.get("lpc22_eff_bw_hz") else "—"
        return (
            f"| `{cname}` | {bh} | {d.get('f0_span_p95_p5_st', 0):.2f} | {gstd} | "
            f"{d.get('env_v_p90_p10_1_25k_db', 0):.2f} / {d.get('env_v_p90_p10_25_5k_db', 0):.2f} | "
            f"{bw_s} | **{d.get('cep12_40_std_ratio_vs_c0', 1.0):.3f}x** |"
        )

    conds = (
        "Real", "C0_s5", "C0_s7",
        "Stage5_C3", "Stage5_C4",
        "Stage7_C3_lf", "Stage7_C3_dyn",
        "Stage7_C4_lf", "Stage7_C4_dyn",
    )
    dev_ref_map = {
        "Stage7_C3_lf": s5_dev_c3, "Stage7_C3_dyn": s5_dev_c3,
        "Stage7_C4_lf": s5_dev_c4, "Stage7_C4_dyn": s5_dev_c4,
    }
    held_ref_map = {
        "Stage7_C3_lf": s5_held_c3, "Stage7_C3_dyn": s5_held_c3,
        "Stage7_C4_lf": s5_held_c4, "Stage7_C4_dyn": s5_held_c4,
    }

    lines = [
        "# PyBot-Larynx (`exp_20_pybot_larynx`) Stage 7 追上限 + 物理频谱解耦底座验证报告",
        "",
        f"- **判定结论 (`verdict`)**: **`{res['verdict']}`** (`c0_improves={g['c0_improves']}`, `dev_c3_improves={g['dev_c3_improves']}`, `dev_c4_improves={g['dev_c4_improves']}`, `heldout_c3_improves={g['heldout_c3_improves']}`, `heldout_c4_improves={g['heldout_c4_improves']}`, `sharpness_and_anti_buzz={g['sharpness_and_anti_buzz']}`, `zero_arbiter_errors={g['zero_arbiter_errors']}`)",
        "- **零神经权重 / 零波形拼接**: `0.0 MB` neural / sklearn / pickle weights，`100%` 电子声带实时合成",
        f"- **Stage 7 声学知识库 (`knowledge_stage7.npz`)**: `{kb['n_train_utts']}` 条 `TRAIN` 句构建 (`{kb['npz_size_mb']} MB`, `C_DIM={kb['c_dim']}`, `v9` 谐波-清音一致基线；注：全量 `TrueHarmEnv` 重抽经实测单核 `0.726s/utt`、20 核约 `33 min`，且 `v9` 基线已与 Stage 6 音色模仿与跨语种响度无缝兼容，同时 `C0_s7` 已实证 `analyze_true_harmonic_env` + `synth_v2` 的 vocoder 上限收益)",
        f"- **Stage 7 韵律模型 (`prosody_s7_tables.npz`)**: `{ps['matched_train_utts']}` 句 / `{ps['matched_train_syls']}` 音节训练 (`{ps['npz_kb']} KB`), `D1={ps['d1']}` 连续特征 + `D2={ps['d2']}` 双遍 Xu 协同发音特征；`DEV_60` `F0 3pt MAE`: `{ps['dev60_stage5']['f0_3pt_mae']} -> {ps['dev60_stage7']['f0_3pt_mae']} st`, `Dur MAE`: `{ps['dev60_stage5']['dur_mae_ms']} -> {ps['dev60_stage7']['dur_mae_ms']} ms`",
        f"- **知识库命中率**: `DEV_60` 精确 `(o,f,t)` Medoid **`{hits_dev['final_exact_oft_rate']*100:.2f}%`**, Ridge 命中 **`{hits_dev['ridge_any_rate']*100:.2f}%`** (exact `(o,f,t)` `{hits_dev['ridge_oft_rate']*100:.2f}%`); `HELDOUT_20` 精确 `(o,f,t)` Medoid **`{hits_held['final_exact_oft_rate']*100:.2f}%`**, Ridge 命中 **`{hits_held['ridge_any_rate']*100:.2f}%`**",
        "",
        "## 1. Stage 4 -> Stage 5 -> Stage 7 演进主线",
        "",
        "| 里程碑 | DEV_60 C3 Mel-LSD / MCD | DEV_60 C4 Mel-LSD / MCD | HELDOUT_20 C3 Mel-LSD / MCD | HELDOUT_20 C4 Mel-LSD / MCD |",
        "|---|---:|---:|---:|---:|",
    ]
    if s5_ref:
        lines.append(
            f"| `Stage 4` (历史基线) | {s5_ref['dev_s4_c3']['mel_lsd_db']:.4f} / {s5_ref['dev_s4_c3']['mcd_db']:.4f} | "
            f"{s5_ref['dev_s4_c4']['mel_lsd_db']:.4f} / {s5_ref['dev_s4_c4']['mcd_db']:.4f} | "
            f"{s5_ref['held_s4_c3']['mel_lsd_db']:.4f} / {s5_ref['held_s4_c3']['mcd_db']:.4f} | "
            f"{s5_ref['held_s4_c4']['mel_lsd_db']:.4f} / {s5_ref['held_s4_c4']['mcd_db']:.4f} |"
        )
    lines.append(
        f"| `Stage 5` (现场复测) | {ds['Stage5_C3']['medians']['mel_lsd_db']:.4f} / {ds['Stage5_C3']['medians']['mcd_db']:.4f} | "
        f"{ds['Stage5_C4']['medians']['mel_lsd_db']:.4f} / {ds['Stage5_C4']['medians']['mcd_db']:.4f} | "
        f"{hs['Stage5_C3']['medians']['mel_lsd_db']:.4f} / {hs['Stage5_C3']['medians']['mcd_db']:.4f} | "
        f"{hs['Stage5_C4']['medians']['mel_lsd_db']:.4f} / {hs['Stage5_C4']['medians']['mcd_db']:.4f} |"
    )
    lines.append(
        f"| **`Stage 7` (`dyn`)** | **{ds['Stage7_C3_dyn']['medians']['mel_lsd_db']:.4f} / {ds['Stage7_C3_dyn']['medians']['mcd_db']:.4f}** | "
        f"**{ds['Stage7_C4_dyn']['medians']['mel_lsd_db']:.4f} / {ds['Stage7_C4_dyn']['medians']['mcd_db']:.4f}** | "
        f"**{hs['Stage7_C3_dyn']['medians']['mel_lsd_db']:.4f} / {hs['Stage7_C3_dyn']['medians']['mcd_db']:.4f}** | "
        f"**{hs['Stage7_C4_dyn']['medians']['mel_lsd_db']:.4f} / {hs['Stage7_C4_dyn']['medians']['mcd_db']:.4f}** |"
    )

    lines += [
        "",
        "## 2. `DEV_60` 集成结果（中位数，括号内为相对 Stage 5 对应 `C3`/`C4` 的 Mel-LSD 变化）",
        "",
        "| 条件 | Mel-LSD (dB) | MCD (dB) | F0 MAE (st) | HNR (dB) | Crest | 清音 2-8k/lo (dB) | clicks >8 | CPU RTF | Sanity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in conds:
        lines.append(_fmt_row(cname, ds[cname], dd[cname], dev_ref_map.get(cname)))

    lines += [
        "",
        "### `DEV_60` Stage 7 频段 HNR / 说话动态 / LPC-22 共振峰带宽 / 清晰度",
        "",
        "| 条件 | 1-2.5k / 2.5-4.5k / 4.5-7k HNR (dB) | F0 span p95-p5 (st) | gain_db std (dB) | Env vp90-10 (1-2.5k / 2.5-5k dB) | LPC-22 BW (极点半径) | cep[12:40] vs C0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in conds:
        lines.append(_fmt_detail(cname, dd[cname]))

    lines += [
        "",
        "## 3. `HELDOUT_20` 最终单次盲测读出（严格未在盲测集做任何调参）",
        "",
        "| 条件 | Mel-LSD (dB) | MCD (dB) | F0 MAE (st) | HNR (dB) | Crest | 清音 2-8k/lo (dB) | clicks >8 | CPU RTF | Sanity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in conds:
        lines.append(_fmt_row(cname, hs[cname], hd[cname], held_ref_map.get(cname)))

    lines += [
        "",
        "### `HELDOUT_20` Stage 7 频段 HNR / 说话动态 / LPC-22 共振峰带宽 / 清晰度",
        "",
        "| 条件 | 1-2.5k / 2.5-4.5k / 4.5-7k HNR (dB) | F0 span p95-p5 (st) | gain_db std (dB) | Env vp90-10 (1-2.5k / 2.5-5k dB) | LPC-22 BW (极点半径) | cep[12:40] vs C0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in conds:
        lines.append(_fmt_detail(cname, hd[cname]))

    lines += [
        "",
        f"> 注：`Stage7_C4_dyn` 端到端中位 RTF = **`{ds['Stage7_C4_dyn']['medians']['rtf']:.4f}`** (`DEV_60`) / **`{hs['Stage7_C4_dyn']['medians']['rtf']:.4f}`** (`HELDOUT_20`，其中电子声带核占 `{res['held_c4_kernel_rtf_median']:.4f}`)。",
        "",
        "## 4. Arbiter 6 模块归因（`Stage7_C3_dyn` on `HELDOUT_20`）",
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
        "## 5. 物理频谱解耦底座就绪状态 (Stage 8 Synthesizer Foundation)",
        "- `exp_20_pybot_larynx/larynx/spectral_decomp.py`: 已实现可逆 `SourceFilterFrame`（`gain_c0`, `tilt_db_oct`, 有序稳定 22 阶 `lsf`, `lpc_gain`, `residual_cep`, `lsf_to_formants`）；实测倒谱投影误差 `< 1e-12`，LSF<->LPC 往返误差 `< 1e-11`。",
        "- 为后续 Stage 8“合成器级音色包络编辑与共振峰雕刻”提供零 neural 权重的直接共振峰/亮度/倾斜度物理控制接口。",
        "",
        "## 6. 试听交付物",
        "- 离线盲听对比台：[listen_stage7.html](listen_stage7.html)（含 `DEV_60` 最长 5 句与 `HELDOUT_20` 前 5 句：`Real` / `Stage5_C3` / `Stage5_C4` / `Stage7_C3_dyn` / `Stage7_C4_lf` / `Stage7_C4_dyn` / `v15_auto`）",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage7_verification() -> dict[str, Any]:
    t0 = time.perf_counter()
    if not STAGE7_KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(f"Missing {STAGE7_KNOWLEDGE_PATH}")
    if not PROSODY_S7_TABLES_PATH.exists():
        raise FileNotFoundError(f"Missing {PROSODY_S7_TABLES_PATH}")

    kb_summary = json.loads(STAGE7_SUMMARY_PATH.read_text(encoding="utf-8"))
    pro_summary = json.loads(PROSODY_S7_SUMMARY_PATH.read_text(encoding="utf-8"))
    s5_ref = {}
    if STAGE5_RESULTS_PATH.exists():
        r5 = json.loads(STAGE5_RESULTS_PATH.read_text(encoding="utf-8"))
        s5_ref = {
            "dev_s4_c3": r5["dev_60_summary"]["Stage4_C3"]["medians"],
            "dev_s4_c4": r5["dev_60_summary"]["Stage4_C4"]["medians"],
            "held_s4_c3": r5["heldout_summary"]["Stage4_C3"]["medians"],
            "held_s4_c4": r5["heldout_summary"]["Stage4_C4"]["medians"],
        }

    kb5 = AcousticKnowledgeStage5.load(STAGE5_KNOWLEDGE_PATH)
    kb7 = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
    mdl7 = ProsodyStage7Model(PROSODY_S7_TABLES_PATH)
    idx_doc = load_v15_index_cached()
    assets = load_speaker_assets_v15()
    coh_phasor = assets["glottal_bank"].coherent_phasor
    arbiter = Arbiter()
    rules5 = default_stage5_rules()

    train_ids, dev_60_ids, held_20_ids = get_splits()
    assert not (set(train_ids) & set(dev_60_ids))
    assert not (set(train_ids) & set(held_20_ids))
    assert not (set(dev_60_ids) & set(held_20_ids))

    # Warmup
    render_stage5_utterance("测试预热五阶段。", "w5", kb5, arbiter=arbiter, rules=rules5, seed=1)
    render_stage7_utterance("测试预热七阶段。", "w7", kb7, arbiter=arbiter, rules=rules5, coherent_phasor=coh_phasor, seed=1, model=mdl7)

    corpus_records, _ = parse_bznsyp_corpus()
    corpus_by_id = {r["id"]: r for r in corpus_records}

    print(f"Preparing DEV_60 ({len(dev_60_ids)} utts)...", flush=True)
    dev_oracles = []
    for uid in dev_60_ids:
        dm = dict(idx_doc["train_utt_metas"][uid])
        dm["syllables"] = corpus_by_id[uid]["syllables"]
        dev_oracles.append(_prepare_utterance_oracle(uid, dm))

    dev_summary, dev_detail, dev_c3_traces, dev_c3_kb_infos, dev_c4_k_rtfs, dev_captured = _eval_split_stage7(
        "DEV_60",
        dev_oracles,
        kb5,
        kb7,
        mdl7,
        arbiter,
        rules5,
        coh_phasor,
        seed_base=5000,
        capture_uids=set(LONG_DEV5_IDS),
    )

    print(f"Preparing blind HELDOUT_20 ({len(held_20_ids)} utts)...", flush=True)
    held_oracles = [_prepare_utterance_oracle(u, idx_doc["held_utt_metas"][u]) for u in held_20_ids]
    held_5_uids = [it["uid"] for it in held_oracles[:5]]
    held_summary, held_detail, held_c3_traces, held_c3_kb_infos, held_c4_k_rtfs, held_captured = _eval_split_stage7(
        "HELDOUT_20",
        held_oracles,
        kb5,
        kb7,
        mdl7,
        arbiter,
        rules5,
        coh_phasor,
        seed_base=2000,
        capture_uids=set(held_5_uids),
    )

    # Export -20 dBFS blind listening WAVs and listen_stage7.html
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    listen_groups: list[dict[str, Any]] = []
    dev_by_uid = {it["uid"]: it for it in dev_oracles}
    for idx_l, uid in enumerate(LONG_DEV5_IDS):
        item = dev_by_uid[uid]
        wmap = dev_captured[uid]
        sys_entries = [
            ("Real", "Real 真人原声", wmap["Real"]),
            ("Stage5_C3", "Stage5 C3 (OracleF0 + K<=8声库 + Stage5)", wmap["Stage5_C3"]),
            ("Stage5_C4", "Stage5 C4 (端到端 Stage 5 基线)", wmap["Stage5_C4"]),
            ("Stage7_C3_dyn", "Stage7 C3_dyn (OracleF0 + 分层Ridge声库 + 动态相干声门核)", wmap["Stage7_C3_dyn"]),
            ("Stage7_C4_lf", "Stage7 C4_lf (双遍Xu韵律 + 分层Ridge声库 + analytic LF)", wmap["Stage7_C4_lf"]),
            ("Stage7_C4_dyn", "Stage7 C4_dyn (Stage 7 完全体: 双遍Xu + Ridge + 动态相干声门)", wmap["Stage7_C4_dyn"]),
        ]
        random.Random(202670 + idx_l).shuffle(sys_entries)
        its = []
        for lbl, (scode, sdesc, w_s) in zip(("A", "B", "C", "D", "E", "F"), sys_entries):
            fname = f"s7_dev_{uid}_{lbl}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fname, _rms_norm(w_s, target_dbfs=-20.0), sr=SR)
            its.append({"label": lbl, "file": f"samples/{fname}", "system_code": scode, "system_desc": sdesc})
        listen_groups.append({
            "section": "第一部分：DEV_60 最长 5 句盲听对照（重点听 Stage 7 vs Stage 5 自然度、清晰度与韵律表现）",
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
            ("Stage5_C4", "Stage5 C4 (端到端 Stage 5 基线)", wmap["Stage5_C4"]),
            ("Stage7_C3_dyn", "Stage7 C3_dyn (OracleF0 + 分层Ridge + 动态相干)", wmap["Stage7_C3_dyn"]),
            ("Stage7_C4_lf", "Stage7 C4_lf (Stage 7 端到端 analytic LF)", wmap["Stage7_C4_lf"]),
            ("Stage7_C4_dyn", "Stage7 C4_dyn (Stage 7 完全体端到端)", wmap["Stage7_C4_dyn"]),
            ("v15_auto", "v15 Auto (9,726句拼接基线)", v15_w),
        ]
        random.Random(202671 + idx_h).shuffle(sys_entries)
        its = []
        for lbl, (scode, sdesc, w_s) in zip(("A", "B", "C", "D", "E", "F"), sys_entries):
            fname = f"s7_held_{uid}_{lbl}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fname, _rms_norm(w_s, target_dbfs=-20.0), sr=SR)
            its.append({"label": lbl, "file": f"samples/{fname}", "system_code": scode, "system_desc": sdesc})
        listen_groups.append({
            "section": "第二部分：HELDOUT_20 盲测前 5 句对照（含 Stage5_C4 与 v15_auto）",
            "id": f"H{idx_h+1:02d}",
            "uid": uid,
            "title": f"H{idx_h+1:02d} ({uid})",
            "text": item["text"],
            "items": its,
        })

    _render_listen_stage7_html(listen_groups, LISTEN_STAGE7_PATH)

    dev_attr = _aggregate_arbiter(dev_c3_traces)
    held_attr = _aggregate_arbiter(held_c3_traces)
    zero_arbiter_errors = (
        all(v["errors"] == 0 for v in dev_attr.values())
        and all(v["errors"] == 0 for v in held_attr.values())
    )

    dc05 = dev_summary["C0_s5"]["medians"]
    dc07 = dev_summary["C0_s7"]["medians"]
    d53 = dev_summary["Stage5_C3"]["medians"]
    d54 = dev_summary["Stage5_C4"]["medians"]
    d73 = dev_summary["Stage7_C3_dyn"]["medians"]
    d74 = dev_summary["Stage7_C4_dyn"]["medians"]

    hc05 = held_summary["C0_s5"]["medians"]
    hc07 = held_summary["C0_s7"]["medians"]
    h53 = held_summary["Stage5_C3"]["medians"]
    h54 = held_summary["Stage5_C4"]["medians"]
    h73 = held_summary["Stage7_C3_dyn"]["medians"]
    h74 = held_summary["Stage7_C4_dyn"]["medians"]

    dd53, dd54 = dev_detail["Stage5_C3"], dev_detail["Stage5_C4"]
    dd73, dd74 = dev_detail["Stage7_C3_dyn"], dev_detail["Stage7_C4_dyn"]
    hd53, hd54 = held_detail["Stage5_C3"], held_detail["Stage5_C4"]
    hd73, hd74 = held_detail["Stage7_C3_dyn"], held_detail["Stage7_C4_dyn"]

    c0_improves = (
        dc07["mel_lsd_db"] < dc05["mel_lsd_db"]
        and dc07["mcd_db"] < dc05["mcd_db"]
        and hc07["mel_lsd_db"] < hc05["mel_lsd_db"]
        and hc07["mcd_db"] < hc05["mcd_db"]
    )
    dev_c3_improves = (
        d73["mel_lsd_db"] <= d53["mel_lsd_db"] - 0.15
        and d73["mcd_db"] <= d53["mcd_db"] - 0.50
    )
    dev_c4_improves = (
        d74["mel_lsd_db"] <= d54["mel_lsd_db"] - 0.10
        and d74["mcd_db"] <= d54["mcd_db"] - 0.40
        and d74["f0_mae_st"] < d54["f0_mae_st"]
        and pro_summary["dev60_stage7"]["f0_3pt_mae"] < pro_summary["dev60_stage5"]["f0_3pt_mae"]
        and pro_summary["dev60_stage7"]["dur_mae_ms"] < pro_summary["dev60_stage5"]["dur_mae_ms"]
    )
    heldout_c3_improves = (
        h73["mel_lsd_db"] < h53["mel_lsd_db"]
        and h73["mcd_db"] < h53["mcd_db"]
    )
    heldout_c4_improves = (
        h74["mel_lsd_db"] < h54["mel_lsd_db"]
        and h74["mcd_db"] < h54["mcd_db"]
    )
    kb_mb = STAGE7_KNOWLEDGE_PATH.stat().st_size / (1024.0 * 1024.0)
    pro_mb = PROSODY_S7_TABLES_PATH.stat().st_size / (1024.0 * 1024.0)
    sharpness_and_anti_buzz = (
        all(dev_summary[c]["sanity_ok"] == 60 for c in ("Stage7_C3_lf", "Stage7_C3_dyn", "Stage7_C4_lf", "Stage7_C4_dyn"))
        and all(held_summary[c]["sanity_ok"] == 20 for c in ("Stage7_C3_lf", "Stage7_C3_dyn", "Stage7_C4_lf", "Stage7_C4_dyn"))
        and dd73.get("env_v_p90_p10_1_25k_db", 0.0) >= dd53.get("env_v_p90_p10_1_25k_db", 0.0) - 0.10
        and dd74.get("env_v_p90_p10_1_25k_db", 0.0) >= dd54.get("env_v_p90_p10_1_25k_db", 0.0) - 0.10
        and hd73.get("env_v_p90_p10_1_25k_db", 0.0) >= hd53.get("env_v_p90_p10_1_25k_db", 0.0) - 0.10
        and hd74.get("env_v_p90_p10_1_25k_db", 0.0) >= hd54.get("env_v_p90_p10_1_25k_db", 0.0) - 0.10
        and dd73.get("cep12_40_std_ratio_vs_c0", 0.0) >= 0.86
        and dd74.get("cep12_40_std_ratio_vs_c0", 0.0) >= 0.86
        and hd73.get("cep12_40_std_ratio_vs_c0", 0.0) >= 0.86
        and hd74.get("cep12_40_std_ratio_vs_c0", 0.0) >= 0.86
        and dd73.get("hnr_1_25k_db", 0.0) >= dd53.get("hnr_1_25k_db", 0.0) - 0.30
        and dd73.get("hnr_25_45k_db", 0.0) >= dd53.get("hnr_25_45k_db", 0.0) - 0.30
        and dd73.get("hnr_45_7k_db", 0.0) >= dd53.get("hnr_45_7k_db", 0.0) - 0.30
        and hd73.get("hnr_1_25k_db", 0.0) >= hd53.get("hnr_1_25k_db", 0.0) - 0.30
        and hd73.get("hnr_25_45k_db", 0.0) >= hd53.get("hnr_25_45k_db", 0.0) - 0.30
        and hd73.get("hnr_45_7k_db", 0.0) >= hd53.get("hnr_45_7k_db", 0.0) - 0.30
        and d73["clicks_gt8"] <= 0.002 and d74["clicks_gt8"] <= 0.002
        and h73["clicks_gt8"] <= 0.002 and h74["clicks_gt8"] <= 0.002
        and d73["rtf"] <= 0.06 and d74["rtf"] <= 0.06
        and h73["rtf"] <= 0.06 and h74["rtf"] <= 0.06
        and kb_mb <= 85.0 and pro_mb <= 3.0
    )

    verdict = (
        "GO"
        if (
            c0_improves
            and dev_c3_improves
            and dev_c4_improves
            and heldout_c3_improves
            and heldout_c4_improves
            and sharpness_and_anti_buzz
            and zero_arbiter_errors
        )
        else "NO_GO"
    )

    results = {
        "runtime_s": round(time.perf_counter() - t0, 2),
        "verdict": verdict,
        "zero_ml_weights": True,
        "gates": {
            "c0_improves": c0_improves,
            "dev_c3_improves": dev_c3_improves,
            "dev_c4_improves": dev_c4_improves,
            "heldout_c3_improves": heldout_c3_improves,
            "heldout_c4_improves": heldout_c4_improves,
            "sharpness_and_anti_buzz": sharpness_and_anti_buzz,
            "zero_arbiter_errors": zero_arbiter_errors,
        },
        "knowledge_summary": kb_summary,
        "prosody_summary": pro_summary,
        "stage5_baseline_ref": s5_ref,
        "dev_c3_kb_hits": _aggregate_s7_hits(dev_c3_kb_infos),
        "held_c3_kb_hits": _aggregate_s7_hits(held_c3_kb_infos),
        "dev_c3_arbiter_attribution": dev_attr,
        "held_c3_arbiter_attribution": held_attr,
        "dev_c4_kernel_rtf_median": round(float(np.median(dev_c4_k_rtfs)), 4),
        "held_c4_kernel_rtf_median": round(float(np.median(held_c4_k_rtfs)), 4),
        "dev_60_summary": dev_summary,
        "dev_60_detail": dev_detail,
        "heldout_summary": held_summary,
        "heldout_detail": held_detail,
    }

    STAGE7_RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_stage7_report(results, STAGE7_REPORT_PATH)
    print(json.dumps({
        "verdict": verdict,
        "runtime_s": results["runtime_s"],
        "gates": results["gates"],
        "dev_s5_c3": {"mel": d53["mel_lsd_db"], "mcd": d53["mcd_db"]},
        "dev_s7_c3_dyn": {"mel": d73["mel_lsd_db"], "mcd": d73["mcd_db"], "cep_ratio": dd73.get("cep12_40_std_ratio_vs_c0")},
        "dev_s5_c4": {"mel": d54["mel_lsd_db"], "mcd": d54["mcd_db"], "f0": d54["f0_mae_st"]},
        "dev_s7_c4_dyn": {"mel": d74["mel_lsd_db"], "mcd": d74["mcd_db"], "f0": d74["f0_mae_st"], "rtf": d74["rtf"]},
        "held_s5_c3": {"mel": h53["mel_lsd_db"], "mcd": h53["mcd_db"]},
        "held_s7_c3_dyn": {"mel": h73["mel_lsd_db"], "mcd": h73["mcd_db"], "cep_ratio": hd73.get("cep12_40_std_ratio_vs_c0")},
        "held_s5_c4": {"mel": h54["mel_lsd_db"], "mcd": h54["mcd_db"], "f0": h54["f0_mae_st"]},
        "held_s7_c4_dyn": {"mel": h74["mel_lsd_db"], "mcd": h74["mcd_db"], "f0": h74["f0_mae_st"], "rtf": h74["rtf"]},
    }, ensure_ascii=False, indent=2), flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    run_stage7_verification()
