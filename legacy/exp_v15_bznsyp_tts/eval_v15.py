"""Full BZNSYP v15 Evaluation: Ablations (B0..B3, Guided), Scaling Curve, Open-Domain & USER_6 vs v14, Listen Page."""
from __future__ import annotations

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")

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

from exp_v12_natural_vocoder.v12_bank import _patch_unsmoothed_uv_env  # noqa: E402
from exp_v12_natural_vocoder.vocoder_v12 import (  # noqa: E402
    VocoderFeaturesV12,
    analyze_v12,
    synthesize_v12,
)
from exp_v13_utau_ceiling.sandhi_v13 import plan_v13  # noqa: E402
from exp_v14_fluent_zh_tts import (  # noqa: E402
    CLEAN_10_IDS,
    PROSODY_MODEL_V14,
    USER_6_IDS,
)
from exp_v14_fluent_zh_tts.bank_v14 import build_v14_zh_bank  # noqa: E402
from exp_v14_fluent_zh_tts.eval_v14 import (  # noqa: E402
    _rms_norm,
    _score_utterance_v14,
    _summarize_metrics,
    _write_pcm16_wav,
)
from exp_v14_fluent_zh_tts.prosody_v14 import ProsodyModelV14  # noqa: E402
from exp_v14_fluent_zh_tts.renderer_v14 import V14Config, render_v14_utterance  # noqa: E402
from exp_v15_bznsyp_tts import (  # noqa: E402
    EXP_DIR,
    HOP,
    INDEX_V15,
    N_FFT,
    PROSODY_MODEL_V15,
    SAMPLES_DIR,
    SPEAKER_ASSETS_V15,
    SR,
)
from exp_v15_bznsyp_tts.index_bznsyp_v15 import (  # noqa: E402
    _fast_mfcc_and_f0_24k,
    load_bznsyp_index_v15,
    load_speaker_assets_v15,
    read_bznsyp_wav_24k,
)
from exp_v15_bznsyp_tts.prosody_v15 import ProsodyModelV15  # noqa: E402
from exp_v15_bznsyp_tts.renderer_v15 import (  # noqa: E402
    LazyUttFeatureCache,
    V15Config,
    render_v15_utterance,
)

USER_6_SPECS = [
    {"id": "Q01", "uid": "SSB05020298", "text": "市区所有私人机动车禁行"},
    {"id": "Q02", "uid": "SSB05020282", "text": "特别是一双舒适的跑鞋"},
    {"id": "Q03", "uid": "SSB05020319", "text": "自去年七月房企再融资重启以来"},
    {"id": "Q04", "uid": "SSB05020184", "text": "话语权是政府政策给你的"},
    {"id": "Q05", "uid": "SSB05020400", "text": "但这时不能说亏损就不做苹果订单"},
    {"id": "Q06", "uid": "SSB05020005", "text": "但同时也有利好政策出台"},
]


def _dtw_align_path(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute DTW alignment index pairs `(ia, ib)` between two spectral track matrices."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    a_c = a - np.mean(a, axis=1, keepdims=True)
    b_c = b - np.mean(b, axis=1, keepdims=True)
    diff2 = (
        np.sum(a_c * a_c, axis=1, keepdims=True)
        + np.sum(b_c * b_c, axis=1, keepdims=True).T
        - 2.0 * (a_c @ b_c.T)
    ) / float(max(1, a.shape[1]))
    cost = np.sqrt(np.maximum(diff2, 0.0))
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    acc[0, 0] = 0.0
    band = max(48, abs(n - m) + 48)
    for i in range(1, n + 1):
        row = cost[i - 1]
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            acc[i, j] = row[j - 1] + min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
    ia, ib = [], []
    i, j = n, m
    while i > 0 and j > 0:
        ia.append(i - 1)
        ib.append(j - 1)
        d, u, l = acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]
        if d <= u and d <= l:
            i -= 1
            j -= 1
        elif u <= l:
            i -= 1
        else:
            j -= 1
    ia.reverse()
    ib.reverse()
    return np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)


def _compute_dtw_f0_rmse_st(
    y: np.ndarray,
    xa: np.ndarray,
    *,
    fx_fast: dict[str, Any] | None = None,
    lead_fr: int = 0,
) -> float:
    """Compute voiced F0 RMSE (semitones) between DTW-aligned synthesis and reference."""
    y_act = y[lead_fr * HOP : lead_fr * HOP + len(xa) + 12 * HOP] if lead_fr > 0 else y
    if len(y_act) < 240 or len(xa) < 240:
        return 0.0
    fy = _fast_mfcc_and_f0_24k(y_act)
    if fx_fast is None:
        fx_fast = _fast_mfcc_and_f0_24k(xa)
    if len(fy["mfcc16"]) == 0 or len(fx_fast["mfcc16"]) == 0:
        return 0.0
    iy, ix = _dtw_align_path(fy["mfcc16"], fx_fast["mfcc16"])
    vy = fy["voiced"][iy] & (fy["f0_hz"][iy] > 50.0)
    vx = fx_fast["voiced"][ix] & (fx_fast["f0_hz"][ix] > 50.0)
    both = vy & vx
    if int(np.sum(both)) < 3:
        return 0.0
    errs = 12.0 * np.log2(fy["f0_hz"][iy][both] / fx_fast["f0_hz"][ix][both])
    return round(float(math.sqrt(float(np.mean(errs * errs)))), 4)


def _safe_score_v15(
    y: np.ndarray,
    xa: np.ndarray,
    plan_doc: dict[str, Any],
    oracle_durs_ms: list[float],
    *,
    fx_fast: dict[str, Any] | None = None,
    info: dict[str, Any] | None = None,
    lead_fr: int = 6,
    rtf: float = 0.0,
    compute_f0_rmse: bool = True,
) -> dict[str, Any]:
    """Run `_score_utterance_v14` safely while enriching `avg_chunk_len`, `dur_mae_ms`, and `f0_rmse_st`."""
    n_sp_fr = max(1, len(xa) // HOP)
    # Guard edge case where an all-sonorant utterance returns uv_2_8k_vs_lo_db = None inside eval_v14
    try:
        res = _score_utterance_v14(
            y, xa, plan_doc, oracle_durs_ms, n_sp_fr, info=info, lead_fr=lead_fr, rtf=rtf,
        )
    except TypeError:
        # Add 2 quiet high-band samples at extreme tail? Fallback direct calculation cleanly:
        y_pad = np.concatenate([y, np.random.default_rng(7).normal(0.0, 1e-4, 240)])
        res = _score_utterance_v14(
            y_pad, xa, plan_doc, oracle_durs_ms, n_sp_fr, info=info, lead_fr=lead_fr, rtf=rtf,
        )

    if info is not None:
        res["avg_chunk_len"] = info.get("avg_chunk_len")
        tgt_d = info.get("target_durs_ms")
        if tgt_d and len(tgt_d) == len(oracle_durs_ms):
            res["dur_mae_ms"] = round(float(np.mean(np.abs(np.asarray(tgt_d) - np.asarray(oracle_durs_ms)))), 2)
    else:
        res["avg_chunk_len"] = None

    if compute_f0_rmse:
        res["f0_rmse_st"] = 0.0 if (lead_fr == 0 and len(y) == len(xa) and np.allclose(y, xa)) else _compute_dtw_f0_rmse_st(y, xa, fx_fast=fx_fast, lead_fr=lead_fr)
    return res


def _summarize_v15(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = _summarize_metrics(rows)
    for k in ("avg_chunk_len", "dur_mae_ms", "f0_rmse_st"):
        vals = [float(r[k]) for r in rows if r.get(k) is not None and not math.isnan(float(r[k]))]
        if vals:
            base["medians"][k] = round(float(np.median(vals)), 4)
            base["means"][k] = round(float(np.mean(vals)), 4)
    return base


def _prepare_heldout_refs(index_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load real audio slices and reference acoustic contours for the 20 held-out BZNSYP utterances."""
    refs: dict[str, dict[str, Any]] = {}
    for uid in index_doc["heldout_20_ids"]:
        meta = index_doc["held_utt_metas"][uid]
        x24 = read_bznsyp_wav_24k(REPO_ROOT / meta["wav_path"])
        syls = meta["syllables"]
        s0_fr = max(0, int(round(syls[0]["start_s"] * SR / HOP)))
        s1_fr = max(s0_fr + 2, int(round(syls[-1]["end_s"] * SR / HOP)))
        xa = x24[s0_fr * HOP : min(len(x24), s1_fr * HOP)]
        fv12_xa = analyze_v12(xa, sr=SR)
        feats_xa_p = _patch_unsmoothed_uv_env(xa, fv12_xa.feats, sr=SR, n_fft=N_FFT)
        v_xa = feats_xa_p.voiced & (feats_xa_p.f0_hz > 50.0)
        utt_f0_med = float(np.median(feats_xa_p.f0_hz[v_xa])) if np.any(v_xa) else float(meta["utt_f0_med_hz"])

        guided_units: list[dict[str, Any]] = []
        oracle_durs_ms: list[float] = []
        for s in syls:
            dur_ms = (float(s["end_s"]) - float(s["start_s"])) * 1000.0
            oracle_durs_ms.append(dur_ms)
            r0 = max(0, int(round(float(s["start_s"]) * SR / HOP)) - s0_fr)
            r1 = min(len(feats_xa_p.f0_hz), max(r0 + 2, int(round(float(s["end_s"]) * SR / HOP)) - s0_fr))
            f0_sl = np.where(feats_xa_p.voiced[r0:r1] & (feats_xa_p.f0_hz[r0:r1] > 50.0), feats_xa_p.f0_hz[r0:r1], 0.0)
            vf0 = f0_sl[f0_sl > 50.0]
            if len(vf0) >= 2:
                st = 12.0 * np.log2(vf0 / utt_f0_med)
                q3 = np.interp([0.2, 0.5, 0.8], np.linspace(0.0, 1.0, len(st)), st)
                f3 = [float(q3[0]), float(q3[1]), float(q3[2])]
            elif len(vf0) == 1:
                st0 = float(12.0 * np.log2(vf0[0] / utt_f0_med))
                f3 = [st0, st0, st0]
            else:
                f3 = [0.0, 0.0, 0.0]

            guided_units.append({
                "lang": "zh",
                "hanzi": s["hanzi"],
                "pinyin": s["pinyin"],
                "onset": s["onset"],
                "final": s["final"],
                "tone": int(s["tone"]),
                "duration_ms": dur_ms,
                "f0_3pt_st": f3,
                "guided_f0_hz": f0_sl,
                "pause_after_ms": float(s["pause_after_ms"]),
                "break": int(s["break"]),
            })

        p_ir = plan_v13(meta["text"])
        refs[uid] = {
            "id": uid,
            "meta": meta,
            "xa": xa,
            "fv12_xa": fv12_xa,
            "feats_xa_p": feats_xa_p,
            "fx_fast": _fast_mfcc_and_f0_24k(xa),
            "utt_f0_med_hz": utt_f0_med,
            "guided_units": guided_units,
            "oracle_durs_ms": oracle_durs_ms,
            "plan_doc": p_ir,
        }
    return refs


def _check_v14_nonfinal_t3(y: np.ndarray, info14: dict[str, Any], text: str) -> list[float]:
    """Measure non-final tone-3 medians (st vs utt median) from synthesized v14 speech."""
    ir = plan_v13(text)
    zh_units = [u for u in (ir.get("units") or []) if u.get("lang", "zh") == "zh"]
    raw_b = info14.get("slot_timings_ms")
    bounds = [] if raw_b is None else list(raw_b)
    raw_f0 = info14.get("f0_hz_all")
    f0_hz = np.asarray([] if raw_f0 is None else raw_f0, dtype=np.float64)
    if len(bounds) != len(zh_units) or len(f0_hz) == 0:
        return []
    vf = f0_hz > 50.0
    if not np.any(vf):
        return []
    utt_med = float(np.median(f0_hz[vf]))
    out: list[float] = []
    n_u = len(zh_units)
    for i, u in enumerate(zh_units):
        if i < n_u - 1 and int(u.get("tone") or 1) == 3:
            b = bounds[i]
            s_ms = float(b["start_ms"] if isinstance(b, dict) else b[0])
            e_ms = float(b["end_ms"] if isinstance(b, dict) else b[1])
            r0 = max(0, int(round(s_ms * SR / (1000.0 * HOP))))
            r1 = min(len(f0_hz), max(r0 + 1, int(round(e_ms * SR / (1000.0 * HOP)))))
            sl = f0_hz[r0:r1]
            vsl = sl > 50.0
            if np.any(vsl):
                out.append(round(float(np.median(12.0 * np.log2(sl[vsl] / utt_med))), 3))
    return out


def _render_listen_html(
    bznsyp10_groups: list[dict[str, Any]],
    open_groups: list[dict[str, Any]],
    user6_groups: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Write standalone offline 100%-TTS UTF-8 listening UI exporting `pybot-v15-listening.json`."""
    payload = {
        "bznsyp10": bznsyp10_groups,
        "open_domain": open_groups,
        "user6": user6_groups,
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-TTS v15 · BZNSYP 万句非神经中文盲听评测台</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#eef1f4;background:#11151c;max-width:980px;margin:1.8rem auto;padding:0 1rem}}
h1{{font-size:1.55rem;margin-bottom:.25rem}}
.sub{{color:#9aa6b8;margin-bottom:1.1rem}}
.tabs{{display:flex;gap:.6rem;margin-bottom:1.1rem;flex-wrap:wrap}}
.tab{{background:#1c2330;color:#c9d3e0;border:1px solid #2e394d;padding:.55rem 1rem;border-radius:8px;cursor:pointer}}
.tab.active{{background:#2f6f58;color:#fff;border-color:#449c7c}}
.panel{{display:none}}.panel.active{{display:block}}
.card{{background:#19202b;border:1px solid #293345;padding:1rem 1.2rem;margin:1rem 0;border-radius:10px}}
.row{{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;padding:.45rem 0;border-bottom:1px dashed #273142}}
.row:last-child{{border-bottom:none}}
audio{{width:min(100%,24rem);height:34px}}
select,input[type=text]{{background:#10151e;color:#eef1f4;border:1px solid #35435b;border-radius:6px;padding:.35rem .55rem;font:inherit}}
input[type=text]{{width:100%}}
button.btn{{background:#2c5e4b;color:#fff;border:none;border-radius:6px;padding:.5rem .95rem;cursor:pointer;margin:.4rem .5rem .2rem 0}}
button.btn.sec{{background:#2b364a}}
.ans{{display:none;margin-top:.55rem;padding:.55rem .8rem;background:#12211c;border-left:3px solid #449c7c;border-radius:4px;font-size:.92rem;color:#cfece0}}
.ans.show{{display:block}}
.meta{{color:#9aa6b8;font-size:.9rem}}
</style>
</head>
<body>
<h1>PyBot-TTS v15 · BZNSYP 万句非神经中文盲听对照台</h1>
<p class="sub">100% 纯 TTS 离线盲听（无唱歌功能）。所有片段均统一响度为 -20 dBFS 并按句随机打乱。</p>

<div class="tabs">
  <button class="tab active" onclick="showTab('bznsyp10', this)">1. BZNSYP 留出句四路盲测 (10句)</button>
  <button class="tab" onclick="showTab('open', this)">2. 开放域 ZH_01–ZH_10 A/B 盲测 (v14 vs v15)</button>
  <button class="tab" onclick="showTab('user6', this)">3. USER_6 Q01–Q06 A/B 盲测 (v14 vs v15)</button>
</div>

<div id="tab-bznsyp10" class="panel active"></div>
<div id="tab-open" class="panel"></div>
<div id="tab-user6" class="panel"></div>

<div class="card">
  <button class="btn" onclick="toggleAnswers()">显示 / 隐藏答案与系统映射</button>
  <button class="btn sec" onclick="exportScores()">导出盲听评分 (pybot-v15-listening.json)</button>
  <span id="status" class="meta"></span>
</div>

<script>
const DATA = {data_json};
const STATE = {{
  bznsyp10: {{dictation:{{}}, naturalness:{{}}, favorite:{{}}, notes:{{}}}},
  open_domain: {{choice:{{}}, naturalness:{{}}, notes:{{}}}},
  user6: {{choice:{{}}, naturalness:{{}}, notes:{{}}}}
}};
let answersVisible = false;

function showTab(id, el) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  el.classList.add('active');
}}

function el(tag, cls, txt) {{
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt !== undefined) e.textContent = txt;
  return e;
}}

function renderFourWayTab() {{
  const root = document.getElementById('tab-bznsyp10');
  root.replaceChildren();
  DATA.bznsyp10.forEach(g => {{
    const card = el('div', 'card');
    card.append(el('h3', '', g.title));
    const dict = el('input');
    dict.type = 'text';
    dict.placeholder = '先只听音频写下听到的句子（检查咬字、声调、疑问语气）…';
    dict.oninput = () => STATE.bznsyp10.dictation[g.id] = dict.value;
    card.append(dict);
    g.items.forEach(it => {{
      const r = el('div', 'row');
      const au = el('audio'); au.controls = true; au.preload = 'none'; au.src = it.file;
      const sel = el('select');
      ['未评分','1 (极生硬)','2','3 (一般)','4','5 (真人级)'].forEach((lab, idx) => sel.add(new Option(lab, idx===0?'':String(idx))));
      sel.onchange = () => STATE.bznsyp10.naturalness[g.id + '-' + it.label] = sel.value;
      r.append(el('b', '', it.label + ' 段'), au, sel);
      card.append(r);
    }});
    const favRow = el('div', 'row');
    favRow.append(el('span', 'meta', '最自然流畅：'));
    const fav = el('select');
    ['未选择', ...g.items.map(x => x.label), '难分伯仲'].forEach(x => fav.add(new Option(x, x==='未选择'?'':x)));
    fav.onchange = () => STATE.bznsyp10.favorite[g.id] = fav.value;
    const note = el('input'); note.type='text'; note.placeholder='备注（语调/半上声/顿挫/音色）'; note.style.flex='1';
    note.oninput = () => STATE.bznsyp10.notes[g.id] = note.value;
    favRow.append(fav, note);
    card.append(favRow);
    const ans = el('div', 'ans');
    ans.textContent = `原文：${{g.text}} ｜ 映射：` + g.items.map(x => `${{x.label}}=${{x.system_desc}}`).join('；');
    card.append(ans);
    root.append(card);
  }});
}}

function renderABTab(rootId, groups, bucket) {{
  const root = document.getElementById(rootId);
  root.replaceChildren();
  groups.forEach(g => {{
    const card = el('div', 'card');
    card.append(el('h3', '', g.title + ' · ' + g.text));
    g.items.forEach(it => {{
      const r = el('div', 'row');
      const au = el('audio'); au.controls = true; au.preload = 'none'; au.src = it.file;
      const sel = el('select');
      ['自然度','1 (差)','2','3 (中)','4','5 (佳)'].forEach((lab, idx) => sel.add(new Option(lab, idx===0?'':String(idx))));
      sel.onchange = () => STATE[bucket].naturalness[g.id + '-' + it.label] = sel.value;
      r.append(el('b', '', it.label + ' 段'), au, sel);
      card.append(r);
    }});
    const chRow = el('div', 'row');
    chRow.append(el('span', 'meta', '更好听更像真人说话：'));
    const ch = el('select');
    ['未选择','A','B','难分伯仲'].forEach(x => ch.add(new Option(x, x==='未选择'?'':x)));
    ch.onchange = () => STATE[bucket].choice[g.id] = ch.value;
    const note = el('input'); note.type='text'; note.placeholder='备注（特别留意 Q02「跑」声调、停连、咬字）'; note.style.flex='1';
    note.oninput = () => STATE[bucket].notes[g.id] = note.value;
    chRow.append(ch, note);
    card.append(chRow);
    const ans = el('div', 'ans');
    ans.textContent = '映射：' + g.items.map(x => `${{x.label}}=${{x.system_desc}}`).join('；');
    card.append(ans);
    root.append(card);
  }});
}}

function toggleAnswers() {{
  answersVisible = !answersVisible;
  document.querySelectorAll('.ans').forEach(a => a.classList.toggle('show', answersVisible));
}}

function exportScores() {{
  const blob = new Blob([JSON.stringify(STATE, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'pybot-v15-listening.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  document.getElementById('status').textContent = ' 已导出 pybot-v15-listening.json';
}}

renderFourWayTab();
renderABTab('tab-open', DATA.open_domain, 'open_domain');
renderABTab('tab-user6', DATA.user6, 'user6');
</script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def run_eval_v15() -> dict[str, Any]:
    t0 = time.perf_counter()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    index_doc = load_bznsyp_index_v15(INDEX_V15)
    assets = load_speaker_assets_v15(SPEAKER_ASSETS_V15)
    pm_v15 = ProsodyModelV15.load(PROSODY_MODEL_V15)
    pm_v14 = ProsodyModelV14.load(PROSODY_MODEL_V14)
    cache = LazyUttFeatureCache(index_doc["train_utt_metas"], max_utts=2048)

    refs = _prepare_heldout_refs(index_doc)
    held_ids = list(index_doc["heldout_20_ids"])

    # 1. Held-out 20 ablations
    print("Running BZNSYP_HELDOUT_20 real & copy_synth...", flush=True)
    ablation: dict[str, Any] = {}
    held_wavs_for_listen: dict[str, dict[str, np.ndarray]] = {uid: {} for uid in held_ids[:10]}

    real_rows: list[dict[str, Any]] = []
    cs_rows: list[dict[str, Any]] = []
    for uid in held_ids:
        rf = refs[uid]
        xa = rf["xa"]
        real_rows.append(_safe_score_v15(xa, xa, rf["plan_doc"], rf["oracle_durs_ms"], fx_fast=rf["fx_fast"], lead_fr=0, compute_f0_rmse=True))
        t_c0 = time.perf_counter()
        v12_in = VocoderFeaturesV12(
            feats=rf["feats_xa_p"],
            bap_gt8k=rf["fv12_xa"].bap_gt8k,
            lp_pulses=rf["fv12_xa"].lp_pulses,
            coherent_phasor=rf["fv12_xa"].coherent_phasor,
        )
        y_cs = synthesize_v12(
            v12_in,
            mode="hybrid_lp_hnm",
            glottal_bank=assets["glottal_bank"],
            crisp_unvoiced=True,
            seed=101,
            phase_dispersion=True,
            bap_gt8k=rf["fv12_xa"].bap_gt8k,
        )
        rtf_cs = (time.perf_counter() - t_c0) / max(0.05, len(y_cs) / SR)
        cs_rows.append(_safe_score_v15(y_cs, xa, rf["plan_doc"], rf["oracle_durs_ms"], fx_fast=rf["fx_fast"], lead_fr=0, rtf=rtf_cs, compute_f0_rmse=True))
        if uid in held_wavs_for_listen:
            held_wavs_for_listen[uid]["real"] = xa
            held_wavs_for_listen[uid]["copy_synth"] = y_cs

    ablation["real"] = _summarize_v15(real_rows)
    ablation["copy_synth"] = _summarize_v15(cs_rows)

    ablation_cfgs = [
        ("B0_v14_style", V15Config(prosody_mode="v14_hgbr", ngram_preseed=False, t3_register_clamp=False, hf_clarity_guard=False, contiguous_chunk_extract=True)),
        ("B1_plus_v15_prosody", V15Config(prosody_mode="v15_hgbr", ngram_preseed=False, t3_register_clamp=False, hf_clarity_guard=False, contiguous_chunk_extract=True)),
        ("B2_plus_ngram_preseed", V15Config(prosody_mode="v15_hgbr", ngram_preseed=True, t3_register_clamp=False, hf_clarity_guard=False, contiguous_chunk_extract=True)),
        ("B3_v15_auto_9726utt", V15Config(prosody_mode="v15_hgbr", ngram_preseed=True, t3_register_clamp=True, hf_clarity_guard=True, contiguous_chunk_extract=True)),
    ]

    for cname, cfg in ablation_cfgs:
        print(f"Evaluating ablation {cname} on 20 held-out utts...", flush=True)
        rows = []
        for idx_u, uid in enumerate(held_ids):
            rf = refs[uid]
            t_s = time.perf_counter()
            y, info = render_v15_utterance(
                rf["meta"]["text"],
                index_doc,
                assets,
                prosody_model_v15=pm_v15,
                prosody_model_v14=pm_v14,
                cfg=cfg,
                cache=cache,
                seed=200 + idx_u,
            )
            rtf = (time.perf_counter() - t_s) / max(0.05, len(y) / SR)
            rows.append(_safe_score_v15(y, rf["xa"], rf["plan_doc"], rf["oracle_durs_ms"], fx_fast=rf["fx_fast"], info=info, lead_fr=6, rtf=rtf))
            if cname == "B3_v15_auto_9726utt" and uid in held_wavs_for_listen:
                held_wavs_for_listen[uid]["v15_auto"] = y
        ablation[cname] = _summarize_v15(rows)
        print(f"  {cname}: mel_lsd={ablation[cname]['medians']['mel_lsd_db']} f0_mae={ablation[cname]['medians']['f0_frame_mae_st']} chain={ablation[cname]['means']['chain_rate']} avg_chunk={ablation[cname]['means'].get('avg_chunk_len')}", flush=True)

    # Guided 9726
    print("Evaluating G_v15_guided_9726utt on 20 held-out utts...", flush=True)
    cfg_g = V15Config(prosody_mode="v15_hgbr", ngram_preseed=True, t3_register_clamp=False, hf_clarity_guard=True, contiguous_chunk_extract=True)
    g_rows = []
    for idx_u, uid in enumerate(held_ids):
        rf = refs[uid]
        t_s = time.perf_counter()
        y_g, info_g = render_v15_utterance(
            rf["meta"]["text"],
            index_doc,
            assets,
            cfg=cfg_g,
            cache=cache,
            guided_units=rf["guided_units"],
            exact_guided_timing=True,
            target_f0_hz=rf["utt_f0_med_hz"],
            seed=300 + idx_u,
        )
        rtf = (time.perf_counter() - t_s) / max(0.05, len(y_g) / SR)
        g_rows.append(_safe_score_v15(y_g, rf["xa"], rf["plan_doc"], rf["oracle_durs_ms"], fx_fast=rf["fx_fast"], info=info_g, lead_fr=4, rtf=rtf))
        if uid in held_wavs_for_listen:
            held_wavs_for_listen[uid]["v15_guided"] = y_g
    ablation["G_v15_guided_9726utt"] = _summarize_v15(g_rows)
    print(f"  G_v15_guided_9726utt: mel_lsd={ablation['G_v15_guided_9726utt']['medians']['mel_lsd_db']} f0_mae={ablation['G_v15_guided_9726utt']['medians']['f0_frame_mae_st']} dur_mae={ablation['G_v15_guided_9726utt']['medians'].get('dur_mae_ms')}", flush=True)

    # 2. Nested corpus scaling curve: 500 -> 2000 -> 5000 -> 9726 (both v15_auto & v15_guided)
    all_train_ids = sorted(index_doc["train_utt_metas"].keys())
    rng_sub = random.Random(42)
    rng_sub.shuffle(all_train_ids)
    scales = [500, 2000, 5000, len(all_train_ids)]
    scaling: dict[str, Any] = {"v15_auto": {}, "v15_guided": {}}
    cfg_auto = ablation_cfgs[-1][1]

    for k_sz in scales:
        if k_sz >= len(all_train_ids):
            scaling["v15_auto"][str(k_sz)] = ablation["B3_v15_auto_9726utt"]
            scaling["v15_guided"][str(k_sz)] = ablation["G_v15_guided_9726utt"]
            continue
        print(f"Running scaling subset k={k_sz}...", flush=True)
        sub_set = frozenset(all_train_ids[:k_sz])
        rows_a, rows_g = [], []
        for idx_u, uid in enumerate(held_ids):
            rf = refs[uid]
            ya, ia = render_v15_utterance(
                rf["meta"]["text"], index_doc, assets, prosody_model_v15=pm_v15,
                cfg=cfg_auto, cache=cache, allowed_utt_ids=sub_set, seed=400 + idx_u,
            )
            rows_a.append(_safe_score_v15(ya, rf["xa"], rf["plan_doc"], rf["oracle_durs_ms"], fx_fast=rf["fx_fast"], info=ia, lead_fr=6))
            yg, ig = render_v15_utterance(
                rf["meta"]["text"], index_doc, assets,
                cfg=cfg_g, cache=cache, allowed_utt_ids=sub_set,
                guided_units=rf["guided_units"], exact_guided_timing=True,
                target_f0_hz=rf["utt_f0_med_hz"], seed=500 + idx_u,
            )
            rows_g.append(_safe_score_v15(yg, rf["xa"], rf["plan_doc"], rf["oracle_durs_ms"], fx_fast=rf["fx_fast"], info=ig, lead_fr=4))
        scaling["v15_auto"][str(k_sz)] = _summarize_v15(rows_a)
        scaling["v15_guided"][str(k_sz)] = _summarize_v15(rows_g)
        print(
            f"  k={k_sz}: auto_mel={scaling['v15_auto'][str(k_sz)]['medians']['mel_lsd_db']} "
            f"auto_chain={scaling['v15_auto'][str(k_sz)]['means']['chain_rate']} "
            f"guided_mel={scaling['v15_guided'][str(k_sz)]['medians']['mel_lsd_db']}",
            flush=True,
        )

    # 3. Open-domain 10 sentences (ZH_01..ZH_10) + USER_6 vs v14_auto (SSB0502 458 utts)
    print("Evaluating Open-Domain ZH_01..ZH_10 & USER_6 (v14_auto vs v15_auto)...", flush=True)
    bench_all = json.loads((REPO_ROOT / "exp_v5_arch" / "benchmark_texts.json").read_text(encoding="utf-8"))
    zh_10 = [x for x in bench_all["sentences"] if str(x.get("id", "")).startswith("ZH_")]
    zh_10.sort(key=lambda x: x["id"])

    bank_v14 = build_v14_zh_bank(exclude_ids=list(USER_6_IDS) + list(CLEAN_10_IDS))
    prof_v14 = bank_v14.profiles_v2["SSB0502"]
    gbank_v14 = bank_v14.glottal_banks.get("SSB0502")
    cfg_v14 = V14Config()

    def _run_ab_suite(items: list[dict[str, str]], prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = []
        for idx_it, item in enumerate(items):
            sid = item["id"]
            txt = item["text"]
            w14, i14 = render_v14_utterance(
                txt, bank_v14, prof_v14, cfg=cfg_v14, prosody_model_v14=pm_v14,
                src_profiles=bank_v14.profiles_v5, glottal_bank=gbank_v14, seed=600 + idx_it,
            )
            t3_v14 = _check_v14_nonfinal_t3(w14, i14, txt)
            w15, i15 = render_v15_utterance(
                txt, index_doc, assets, prosody_model_v15=pm_v15,
                cfg=cfg_auto, cache=cache, seed=700 + idx_it,
            )
            def _extra(w: np.ndarray, info_d: dict[str, Any], t3_list: list[float]) -> dict[str, Any]:
                sc = _safe_score_v15(w, w, plan_v13(txt), [180.0], info=info_d, lead_fr=0, compute_f0_rmse=False)
                return {
                    "sanity_ok": bool(sc.get("sanity_ok", False)),
                    "duration_s": round(len(w) / SR, 2),
                    "chain_rate": round(float(info_d.get("chain_rate") or 0.0), 4),
                    "avg_chunk_len": round(float(info_d.get("avg_chunk_len") or 1.0), 3),
                    "exact_tone_hit_rate": round(float(info_d.get("exact_tone_hit_rate") or 0.0), 4),
                    "fallback_rate": round(float(info_d.get("fallback_rate") or 0.0), 4),
                    "t3_nonfinal_medians_st": t3_list,
                    "n_gt9db": int(sc.get("n_gt9db", 0)),
                    "clicks_gt8": round(float(sc.get("clicks_gt8", 0.0)), 6),
                    "voiced_contrast_db": round(float(sc.get("voiced_contrast_db", 0.0)), 3),
                }

            # Compute avg_chunk_len for v14 accurately from selected_trace if present
            if "avg_chunk_len" not in i14:
                n_u14 = len(i14.get("slot_timings_ms") or [])
                n_c14 = max(1, int(i14.get("n_chunks") or n_u14))
                i14["avg_chunk_len"] = round(float(n_u14) / float(n_c14), 3) if n_u14 > 0 else 1.0

            rec = {
                "id": sid,
                "text": txt,
                "v14_auto_458utt": _extra(w14, i14, t3_v14),
                "v15_auto_9726utt": _extra(w15, i15, i15.get("t3_nonfinal_medians_st", [])),
            }
            rows.append(rec)

            pair = [
                ("v14_auto_458utt", "v14 Auto (SSB0502 458句)", w14),
                ("v15_auto_9726utt", "v15 Auto (BZNSYP 9,726句 + #1-#3 + 半上声钳制)", w15),
            ]
            random.Random(3000 + idx_it + len(prefix) * 100).shuffle(pair)
            ab_items = []
            for lbl, (scode, sdesc, w_s) in zip(("A", "B"), pair):
                fname = f"{prefix}_{sid}_{lbl}.wav"
                _write_pcm16_wav(SAMPLES_DIR / fname, _rms_norm(w_s), sr=SR)
                ab_items.append({"label": lbl, "file": f"samples/{fname}", "system_code": scode, "system_desc": sdesc})
            groups.append({"id": sid, "title": item.get("uid", sid), "text": txt, "items": ab_items})
        return rows, groups

    open_rows, open_groups = _run_ab_suite(zh_10, "open")
    user6_rows, user6_groups = _run_ab_suite(USER_6_SPECS, "user6")

    # 4. Write 10 diverse BZNSYP_HELDOUT_20 4-way WAVs
    bz_groups: list[dict[str, Any]] = []
    sys_labels = [
        ("real", "Real 真人原录音"),
        ("copy_synth", "CopySynth v12原声学重合成"),
        ("v15_guided", "v15 Guided (参考节奏音高)"),
        ("v15_auto", "v15 Auto (9,726句全自动)"),
    ]
    for idx_b, uid in enumerate(held_ids[:10]):
        txt = refs[uid]["meta"]["text"]
        perm = list(sys_labels)
        random.Random(2026 + idx_b).shuffle(perm)
        its = []
        for lbl, (scode, sdesc) in zip(("A", "B", "C", "D"), perm):
            fname = f"bz10_{uid}_{lbl}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fname, _rms_norm(held_wavs_for_listen[uid][scode]), sr=SR)
            its.append({"label": lbl, "file": f"samples/{fname}", "system_code": scode, "system_desc": sdesc})
        kind = "疑问" if ("？" in txt or "?" in txt) else ("感叹" if ("！" in txt or "!" in txt) else ("长句" if "，" in txt else "陈述"))
        bz_groups.append({"id": f"BZ{idx_b+1:02d}", "uid": uid, "title": f"BZ{idx_b+1:02d} [{kind}] ({uid})", "text": txt, "items": its})

    listen_path = EXP_DIR / "listen.html"
    _render_listen_html(bz_groups, open_groups, user6_groups, listen_path)

    # Compute concise Open-Domain & USER_6 summaries
    def _ab_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        return {
            "n": len(rows),
            "mean_chain_rate": round(float(np.mean([r[key]["chain_rate"] for r in rows])), 4),
            "mean_avg_chunk_len": round(float(np.mean([r[key]["avg_chunk_len"] for r in rows])), 3),
            "mean_exact_tone_hit_rate": round(float(np.mean([r[key]["exact_tone_hit_rate"] for r in rows])), 4),
            "mean_fallback_rate": round(float(np.mean([r[key]["fallback_rate"] for r in rows])), 4),
            "total_gt9db_jumps": int(sum(r[key]["n_gt9db"] for r in rows)),
            "mean_clicks_gt8": round(float(np.mean([r[key]["clicks_gt8"] for r in rows])), 6),
        }

    results = {
        "runtime_s": round(time.perf_counter() - t0, 2),
        "heldout_20_ids": held_ids,
        "prosody_model_v15_stats": pm_v15.train_stats,
        "ablation": ablation,
        "scaling": scaling,
        "open_domain_summary": {
            "v14_auto_458utt": _ab_summary(open_rows, "v14_auto_458utt"),
            "v15_auto_9726utt": _ab_summary(open_rows, "v15_auto_9726utt"),
        },
        "open_domain": open_rows,
        "user6_summary": {
            "v14_auto_458utt": _ab_summary(user6_rows, "v14_auto_458utt"),
            "v15_auto_9726utt": _ab_summary(user6_rows, "v15_auto_9726utt"),
        },
        "user6": user6_rows,
        "cache_disk_reads": cache.disk_reads,
        "listen_html": str(listen_path.relative_to(REPO_ROOT)).replace("\\", "/"),
    }
    res_path = EXP_DIR / "results_v15.json"
    res_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "runtime_s": results["runtime_s"],
        "b3_auto_mel_lsd_db": ablation["B3_v15_auto_9726utt"]["medians"]["mel_lsd_db"],
        "guided_mel_lsd_db": ablation["G_v15_guided_9726utt"]["medians"]["mel_lsd_db"],
        "open_domain_summary": results["open_domain_summary"],
        "user6_q02_t3": next(r for r in user6_rows if r["id"] == "Q02"),
        "cache_disk_reads": cache.disk_reads,
    }, ensure_ascii=False, indent=2), flush=True)
    return results


if __name__ == "__main__":
    run_eval_v15()
