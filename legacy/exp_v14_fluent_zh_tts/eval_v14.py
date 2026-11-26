"""Phase 10 (v14) Single-Factor Ablations (`A0..A6`), Data Scaling (`50->458`), Open-Domain Eval & Blind TTS UI."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable
import wave

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity, compute_spectral_distances  # noqa: E402
from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10  # noqa: E402
from exp_v11_eval.v11_metrics import evaluate_v11_utterance  # noqa: E402
from exp_v12_natural_vocoder.vocoder_v12 import synthesize_v12  # noqa: E402
from exp_v12_prosody.prosody_model import DEFAULT_MODEL_PATH, load_prosody_v12  # noqa: E402
from exp_v13_utau_ceiling.diagnose_v13 import _compute_f0_mae_st, _frame_aligned_mel_lsd  # noqa: E402
from exp_v13_utau_ceiling.sandhi_v13 import plan_v13  # noqa: E402
from exp_v14_fluent_zh_tts import (  # noqa: E402
    ALIGNMENTS_V14,
    CLEAN_10_IDS,
    EXP_DIR,
    FEATS_CACHE_V14,
    HOP,
    MANIFEST_V14,
    PROSODY_MODEL_V14,
    SAMPLES_DIR,
    SR,
    USER_6_IDS,
)
from exp_v14_fluent_zh_tts.align_v14 import (  # noqa: E402
    _to_vocoder_features,
    extract_or_load_ssb0502_features,
)
from exp_v14_fluent_zh_tts.bank_v14 import (  # noqa: E402
    V14ZhBank,
    build_v14_zh_bank,
    subsample_v14_bank,
)
from exp_v14_fluent_zh_tts.prosody_v14 import ProsodyModelV14  # noqa: E402
from exp_v14_fluent_zh_tts.renderer_v14 import V14Config, render_v14_utterance  # noqa: E402

LEAD_SIL_FRAMES = 4
OPEN_DOMAIN_IDS = ("ZH_01", "ZH_02", "ZH_04", "ZH_05", "ZH_08", "ZH_10")


def _rms_norm(x: np.ndarray, target_dbfs: float = -20.0, max_peak: float = 0.89) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    act = arr[np.abs(arr) > 0.01 * max(1e-6, float(np.max(np.abs(arr))))]
    rms = float(np.sqrt(np.mean(act * act))) if act.size else float(np.sqrt(np.mean(arr * arr)))
    if rms <= 1e-12:
        return arr.copy()
    scaled = arr * ((10.0 ** (target_dbfs / 20.0)) / rms)
    pk = float(np.max(np.abs(scaled)))
    if pk > max_peak:
        scaled = scaled * (max_peak / pk)
    return scaled


def _write_pcm16_wav(path: Path, x: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _build_guided_controls_v14(
    u: dict[str, Any],
    feats_real: vocoder.VocoderFeatures,
    al_entry: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float], int]:
    units: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    oracle_durs: list[float] = []
    syls = al_entry["syllables"]
    for s in syls:
        o, f, tone, ipas = cf.pinyin_to_ipa(s["pinyin"])
        if o in ("y", "w"):
            o = ""
        if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
            f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"
        s0, s1 = int(s["syl_start"]), int(s["syl_end"])
        nf = max(3, s1 - s0)
        dur_ms = nf * 5.0
        p_ms = float(s.get("pause_after_ms", 0.0))
        oracle_durs.append(dur_ms)
        units.append({
            "lang": "zh",
            "hanzi": s["hanzi"],
            "token": s["hanzi"],
            "onset": o,
            "final": f,
            "tone": int(tone),
            "lexical_tone": int(tone),
            "pinyin": s["pinyin"],
            "ipa_phones": ipas,
            "stress_or_tone": f"T{tone}",
            "duration_ms": dur_ms,
            "pause_after_ms": p_ms,
            "pitch_st": [0.0] * 5,
        })
        controls.append({
            "onset": o,
            "final": f,
            "tone": int(tone),
            "dur_ms": dur_ms,
            "f0_hz": feats_real.f0_hz[s0:s1],
            "voiced_hint": feats_real.voiced[s0:s1],
            "pause_after_ms": p_ms,
        })
    plan_doc = {"schema": 2, "text": u["text"], "style": "guided", "languages": ["zh"], "units": units}
    n_sp_fr = int(syls[-1]["syl_end"]) if syls else len(feats_real.f0_hz)
    return plan_doc, controls, oracle_durs, n_sp_fr


def _score_utterance_v14(
    y: np.ndarray,
    xa: np.ndarray,
    plan_doc: dict[str, Any],
    oracle_durs_ms: list[float],
    n_speech_frames: int,
    *,
    info: dict[str, Any] | None = None,
    lead_fr: int = LEAD_SIL_FRAMES,
    rtf: float = 0.0,
) -> dict[str, Any]:
    info = info or {}
    sd = compute_spectral_distances(xa, y, sr=SR)
    fa = _frame_aligned_mel_lsd(xa, y, syn_lead_frames=lead_fr, n_speech_frames=n_speech_frames)
    f0_fr_mae, f0_med_err = _compute_f0_mae_st(xa, y, sr=SR, syn_lead_frames=lead_fr)
    v11_u = evaluate_v11_utterance(
        y,
        plan_doc,
        sr=SR,
        fmin=75.0,
        fmax=450.0,
        slot_timings_ms=info.get("slot_timings_ms"),
        unit_boundaries_ms=info.get("unit_boundaries_ms"),
        log_env=info.get("log_env"),
        boundary_frames=info.get("boundary_frames"),
    )
    v9 = v11_u["v9"]
    lj = v11_u["loudness_jumps"]
    nd = v11_u["nondrop"]

    slots = info.get("slot_timings_ms") or []
    if len(slots) == len(oracle_durs_ms):
        pred_durs = [float(e - s) for s, e in slots]
        dur_mae_ms = float(np.mean(np.abs(np.asarray(pred_durs) - np.asarray(oracle_durs_ms))))
    else:
        dur_mae_ms = float("nan")

    return {
        "sanity_ok": bool(v11_u["sanity_ok"]),
        "mel_lsd_db": round(float(sd["mel_lsd_db"]), 4) if sd else float("nan"),
        "mcd_db": round(float(sd["mcd_db"]), 4) if sd else float("nan"),
        "mcd_cmn_db": round(float(sd["mcd_cmn_db"]), 4) if sd else float("nan"),
        "nodtw_shape_mel_lsd_db": round(float(fa["nodtw_shape_mel_lsd_db"]), 4),
        "f0_frame_mae_st": round(float(f0_fr_mae), 4) if math.isfinite(f0_fr_mae) else float("nan"),
        "f0_med_err_st": round(float(f0_med_err), 4) if math.isfinite(f0_med_err) else float("nan"),
        "dur_mae_ms": round(float(dur_mae_ms), 2) if math.isfinite(dur_mae_ms) else float("nan"),
        "voiced_contrast_db": round(float(v9["clarity"]["voiced_spectral_contrast_db"]), 4),
        "uv_2_8k_vs_lo_db": round(float(v9["clarity"]["unvoiced_2_8k_vs_lo_db"]), 4),
        "clicks_gt8": round(float(v9["click_outlier_rate_gt8"]), 5),
        "n_gt9db": int(lj["n_gt9db"]),
        "n_dropped_slots": int(nd["n_dropped_slots"]),
        "chain_rate": info.get("chain_rate"),
        "exact_tone_hit_rate": info.get("exact_tone_hit_rate"),
        "fallback_rate": info.get("fallback_rate"),
        "n_internal_seams_saved": int(info.get("n_internal_seams_saved", 0)),
        "f0_natural_corr": info.get("f0_natural_corr"),
        "rtf": round(float(rtf), 4),
    }


def _summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    med_keys = [
        "mel_lsd_db", "mcd_db", "mcd_cmn_db", "nodtw_shape_mel_lsd_db",
        "f0_frame_mae_st", "dur_mae_ms", "voiced_contrast_db", "uv_2_8k_vs_lo_db",
        "clicks_gt8", "chain_rate", "exact_tone_hit_rate", "fallback_rate",
        "f0_natural_corr", "rtf",
    ]
    meds: dict[str, float] = {}
    means: dict[str, float] = {}
    for k in med_keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None and math.isfinite(float(r[k]))]
        if vals:
            meds[k] = round(float(np.median(vals)), 4)
            means[k] = round(float(np.mean(vals)), 4)
    return {
        "n": len(rows),
        "sanity_ok": sum(1 for r in rows if r["sanity_ok"]),
        "total_gt9db_jumps": sum(int(r["n_gt9db"]) for r in rows),
        "total_dropped_slots": sum(int(r["n_dropped_slots"]) for r in rows),
        "total_seams_saved": sum(int(r["n_internal_seams_saved"]) for r in rows),
        "medians": meds,
        "means": means,
    }


def build_listen_html_v14(
    user6_blind: list[dict[str, Any]],
    clean6_blind: list[dict[str, Any]],
    open_ab: list[dict[str, Any]],
) -> Path:
    """Write standalone offline 100%-TTS blind listening evaluation page (`listen.html`)."""
    data_json = json.dumps(
        {
            "user6_blind": user6_blind,
            "clean6_blind": clean6_blind,
            "open_domain_ab": open_ab,
        },
        ensure_ascii=False,
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-TTS v14 · 多样本中文流畅说话盲听台</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#eef1f4;background:#11151c;max-width:960px;margin:1.8rem auto;padding:0 1rem}}
h1{{font-size:1.6rem;margin-bottom:.25rem}}
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
<h1>PyBot-TTS v14 · 多样本中文流畅说话盲听对照台</h1>
<p class="sub">完全离线运行（100% 纯 TTS 说话测试）。所有音频均已统一响度（-20 dBFS）且按句随机打乱。</p>

<div class="tabs">
  <button class="tab active" onclick="showTab('user6', this)">1. USER_6 盲测 (Q01–Q06 四路对照)</button>
  <button class="tab" onclick="showTab('clean6', this)">2. CLEAN_6 流畅句盲测 (三路对照)</button>
  <button class="tab" onclick="showTab('open', this)">3. 开放域自由文本盲测 (v12 vs v14 Auto)</button>
</div>

<div id="tab-user6" class="panel active"></div>
<div id="tab-clean6" class="panel"></div>
<div id="tab-open" class="panel"></div>

<div class="card">
  <button class="btn" onclick="toggleAnswers()">显示 / 隐藏答案与映射</button>
  <button class="btn sec" onclick="exportScores()">导出盲听记录 (pybot-v14-listening.json)</button>
  <span id="status" class="meta"></span>
</div>

<script>
const DATA = {data_json};
const STATE = {{ user6: {{dictation:{{}}, naturalness:{{}}, favorite:{{}}, notes:{{}}}}, clean6: {{naturalness:{{}}, favorite:{{}}, notes:{{}}}}, open_domain: {{choice:{{}}, notes:{{}}}} }};
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

function renderBlindTab(rootId, groups, bucket, withDictation) {{
  const root = document.getElementById(rootId);
  root.replaceChildren();
  groups.forEach(g => {{
    const card = el('div', 'card');
    card.append(el('h3', '', g.title));
    if (withDictation) {{
      const dict = el('input');
      dict.type = 'text';
      dict.placeholder = '先只听音频，写下你听到的句子（核对咬字/声调）…';
      dict.oninput = () => STATE[bucket].dictation[g.id] = dict.value;
      card.append(dict);
    }} else {{
      card.append(el('p', 'meta', '文本：' + g.text));
    }}
    g.items.forEach(it => {{
      const r = el('div', 'row');
      const au = el('audio'); au.controls = true; au.preload = 'none'; au.src = it.file;
      const sel = el('select');
      ['未评分','1 (极生硬)','2','3 (一般)','4','5 (真人级)'].forEach((lab, idx) => sel.add(new Option(lab, idx===0?'':String(idx))));
      sel.onchange = () => STATE[bucket].naturalness[g.id + '-' + it.label] = sel.value;
      r.append(el('b', '', it.label + ' 段'), au, sel);
      card.append(r);
    }});
    const favRow = el('div', 'row');
    favRow.append(el('span', 'meta', '最自然流畅：'));
    const fav = el('select');
    const opts = ['未选择', ...g.items.map(x => x.label), '难分伯仲'];
    opts.forEach(x => fav.add(new Option(x, x==='未选择'?'':x)));
    fav.onchange = () => STATE[bucket].favorite[g.id] = fav.value;
    const note = el('input'); note.type='text'; note.placeholder='备注（停顿/连贯度/拖音/咬字）'; note.style.flex='1';
    note.oninput = () => STATE[bucket].notes[g.id] = note.value;
    favRow.append(fav, note);
    card.append(favRow);

    const ans = el('div', 'ans');
    ans.textContent = `原文：${{g.text}} (${{g.pinyin || ''}}) ｜ 映射：` + g.items.map(x => `${{x.label}}=${{x.system_desc}}`).join('；');
    card.append(ans);
    root.append(card);
  }});
}}

function renderOpenTab() {{
  const root = document.getElementById('tab-open');
  root.replaceChildren();
  DATA.open_domain_ab.forEach(g => {{
    const card = el('div', 'card');
    card.append(el('h3', '', g.title + ' · ' + g.text));
    g.items.forEach(it => {{
      const r = el('div', 'row');
      const au = el('audio'); au.controls = true; au.preload = 'none'; au.src = it.file;
      r.append(el('b', '', it.label + ' 段'), au);
      card.append(r);
    }});
    const chRow = el('div', 'row');
    chRow.append(el('span', 'meta', '更好听更连贯：'));
    const ch = el('select');
    ['未选择','A','B','难分伯仲'].forEach(x => ch.add(new Option(x, x==='未选择'?'':x)));
    ch.onchange = () => STATE.open_domain.choice[g.id] = ch.value;
    const note = el('input'); note.type='text'; note.placeholder='备注（重音/停顿/清晰度）'; note.style.flex='1';
    note.oninput = () => STATE.open_domain.notes[g.id] = note.value;
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
  a.download = 'pybot-v14-listening.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  document.getElementById('status').textContent = ' 已导出 pybot-v14-listening.json';
}}

renderBlindTab('tab-user6', DATA.user6_blind, 'user6', true);
renderBlindTab('tab-clean6', DATA.clean6_blind, 'clean6', false);
renderOpenTab();
</script>
</body>
</html>
"""
    out = EXP_DIR / "listen.html"
    out.write_text(html, encoding="utf-8")
    return out


def run_eval_v14() -> dict[str, Any]:
    """Run complete zero-leakage v14 evaluation: A0..A6 ablations, 50->458 scaling, open-domain & listen.html."""
    t_start = time.perf_counter()
    man = json.loads(MANIFEST_V14.read_text(encoding="utf-8"))
    al_doc = json.loads(ALIGNMENTS_V14.read_text(encoding="utf-8"))
    feats_map = extract_or_load_ssb0502_features(man["utterances"], cache_path=FEATS_CACHE_V14)
    utt_map = {u["id"]: u for u in man["utterances"]}

    clean_10 = tuple(CLEAN_10_IDS) if CLEAN_10_IDS else tuple(man.get("clean_10_ids") or ())
    eval_ids = list(USER_6_IDS) + list(clean_10)
    excl_all = set(eval_ids)

    print(f"Building zero-leakage SSB0502 banks (heldout={len(excl_all)})...", flush=True)
    bank_458_nosil = build_v14_zh_bank(exclude_ids=excl_all, sil_aware=False, quarantine_outliers=False)
    bank_458_sil = build_v14_zh_bank(exclude_ids=excl_all, sil_aware=True, quarantine_outliers=True)
    bank_50_nosil = subsample_v14_bank(bank_458_nosil, max_utts=50, seed=42)
    bank_50_sil = subsample_v14_bank(bank_458_sil, max_utts=50, seed=42)
    bank_150_sil = subsample_v14_bank(bank_458_sil, max_utts=150, seed=42)
    bank_300_sil = subsample_v14_bank(bank_458_sil, max_utts=300, seed=42)

    tgt_prof = bank_458_sil.profiles_v5["SSB0502"]
    gbank_held = bank_458_sil.glottal_banks["SSB0502"]
    pm_v12 = load_prosody_v12(DEFAULT_MODEL_PATH)
    pm_v14 = ProsodyModelV14.load(PROSODY_MODEL_V14)

    # 1. Prebuild held-out oracle ground-truth references (16 utterances = USER_6 + CLEAN_10)
    ref_data: dict[str, dict[str, Any]] = {}
    real_rows: dict[str, list[dict[str, Any]]] = {"all_16": [], "user_6": [], "clean_10": []}
    cs_rows: dict[str, list[dict[str, Any]]] = {"all_16": [], "user_6": [], "clean_10": []}

    for uid in eval_ids:
        u = utt_map[uid]
        fe = feats_map[uid]
        xa = np.asarray(fe["xa"], dtype=np.float64)
        feats_real = _to_vocoder_features(fe)
        b_gt8 = np.asarray(fe["bap_gt8k"], dtype=np.float64)
        al_e = al_doc["zh_all"][uid]
        plan_g, ctrls_g, oracle_durs, n_sp_fr = _build_guided_controls_v14(u, feats_real, al_e)
        y_cs = synthesize_v12(
            feats_real,
            mode="hybrid_lp_hnm",
            glottal_bank=gbank_held,
            crisp_unvoiced=True,
            seed=72,
            phase_dispersion=True,
            bap_gt8k=b_gt8,
        )
        split_tag = "user_6" if uid in USER_6_IDS else "clean_10"
        r_score = _score_utterance_v14(xa, xa, plan_g, oracle_durs, n_sp_fr, lead_fr=0, rtf=0.0)
        c_score = _score_utterance_v14(y_cs, xa, plan_g, oracle_durs, n_sp_fr, lead_fr=0, rtf=0.03)
        real_rows["all_16"].append(r_score); real_rows[split_tag].append(r_score)
        cs_rows["all_16"].append(c_score); cs_rows[split_tag].append(c_score)
        ref_data[uid] = {
            "u": u,
            "xa": xa,
            "plan_g": plan_g,
            "ctrls_g": ctrls_g,
            "oracle_durs": oracle_durs,
            "n_sp_fr": n_sp_fr,
            "y_cs": y_cs,
            "split": split_tag,
        }

    # 2. Part A: Single-Factor Ablation Ladder (Auto A0..A6 + Guided baselines)
    ablation_specs: list[tuple[str, V14ZhBank, V14Config, bool]] = [
        ("A0_v12_auto_50utt", bank_50_nosil, V14Config(
            use_sandhi_v13=False, prosody_mode="v12_ridge", selector_mode="v12",
            contiguous_chunk_extract=False, preserve_natural_f0=False,
            chunk_dur_damp=1.0, syl_dur_damp=1.0, gentle_gain=False,
        ), False),
        ("A1_v13_auto_50utt", bank_50_nosil, V14Config(
            use_sandhi_v13=True, prosody_mode="v12_ridge", selector_mode="v13",
            contiguous_chunk_extract=False, preserve_natural_f0=False,
            chunk_dur_damp=1.0, syl_dur_damp=1.0, gentle_gain=True,
        ), False),
        ("A2_v13_auto_458utt", bank_458_nosil, V14Config(
            use_sandhi_v13=True, prosody_mode="v12_ridge", selector_mode="v13",
            contiguous_chunk_extract=False, preserve_natural_f0=False,
            chunk_dur_damp=1.0, syl_dur_damp=1.0, gentle_gain=True,
        ), False),
        ("A3_+sil_align_bank", bank_458_sil, V14Config(
            use_sandhi_v13=True, prosody_mode="v12_ridge", selector_mode="v13",
            contiguous_chunk_extract=False, preserve_natural_f0=False,
            chunk_dur_damp=1.0, syl_dur_damp=1.0, gentle_gain=True,
        ), False),
        ("A4_+hgbr_prosody_v14", bank_458_sil, V14Config(
            use_sandhi_v13=True, prosody_mode="v14_hgbr", selector_mode="v14",
            contiguous_chunk_extract=False, preserve_natural_f0=False,
            chunk_dur_damp=1.0, syl_dur_damp=1.0, gentle_gain=True,
        ), False),
        ("A5_+contiguous_chunks", bank_458_sil, V14Config(
            use_sandhi_v13=True, prosody_mode="v14_hgbr", selector_mode="v14",
            contiguous_chunk_extract=True, preserve_natural_f0=False,
            chunk_dur_damp=1.0, syl_dur_damp=1.0, gentle_gain=True,
        ), False),
        ("A6_+natural_f0_damped_dur", bank_458_sil, V14Config(
            use_sandhi_v13=True, prosody_mode="v14_hgbr", selector_mode="v14",
            contiguous_chunk_extract=True, preserve_natural_f0=True,
            chunk_dur_damp=0.45, syl_dur_damp=0.55, gentle_gain=True,
        ), False),
        ("G0_v12_guided_50utt", bank_50_nosil, V14Config(
            use_sandhi_v13=False, prosody_mode="v12_ridge", selector_mode="v12",
            contiguous_chunk_extract=False, preserve_natural_f0=False,
            chunk_dur_damp=1.0, syl_dur_damp=1.0, gentle_gain=False,
        ), True),
        ("G1_v14_guided_458utt", bank_458_sil, V14Config(
            use_sandhi_v13=True, prosody_mode="v14_hgbr", selector_mode="v14",
            contiguous_chunk_extract=True, preserve_natural_f0=False,
            chunk_dur_damp=1.0, syl_dur_damp=1.0, gentle_gain=True,
        ), True),
    ]

    ablation_res: dict[str, Any] = {
        "real": {k: _summarize_metrics(v) for k, v in real_rows.items()},
        "copy_synth": {k: _summarize_metrics(v) for k, v in cs_rows.items()},
    }
    audio_out_cache: dict[str, dict[str, np.ndarray]] = {uid: {"real": ref_data[uid]["xa"]} for uid in eval_ids}

    print("\nRunning Part A (Single-Factor Ablations A0..A6 + Guided)...", flush=True)
    for name, bnk, cfg_i, is_guided in ablation_specs:
        by_split: dict[str, list[dict[str, Any]]] = {"all_16": [], "user_6": [], "clean_10": []}
        for uid in eval_ids:
            rd = ref_data[uid]
            pl_in = rd["plan_g"] if is_guided else (plan_v13(rd["u"]["text"]) if cfg_i.use_sandhi_v13 else plan_v10(rd["u"]["text"]))
            ts = time.perf_counter()
            y_syn, info = render_v14_utterance(
                pl_in,
                bnk,
                tgt_prof,
                guided_controls=rd["ctrls_g"] if is_guided else None,
                cfg=cfg_i,
                prosody_model_v14=pm_v14,
                prosody_model_v12=pm_v12,
                glottal_bank=gbank_held,
                seed=72,
            )
            rtf = (time.perf_counter() - ts) / max(1e-6, len(y_syn) / SR)
            pl_eval = rd["plan_g"] if is_guided else pl_in
            sc = _score_utterance_v14(
                y_syn, rd["xa"], pl_eval, rd["oracle_durs"], rd["n_sp_fr"],
                info=info, lead_fr=LEAD_SIL_FRAMES, rtf=rtf,
            )
            by_split["all_16"].append(sc)
            by_split[rd["split"]].append(sc)
            audio_out_cache[uid][name] = y_syn
        ablation_res[name] = {k: _summarize_metrics(v) for k, v in by_split.items()}
        m_all = ablation_res[name]["all_16"]["medians"]
        print(
            f"  {name:26s} | Mel={m_all['mel_lsd_db']:.3f} MCD={m_all['mcd_db']:.2f} NoDTW={m_all['nodtw_shape_mel_lsd_db']:.3f} "
            f"F0={m_all['f0_frame_mae_st']:.2f}st Dur={m_all['dur_mae_ms']:.1f}ms Chain={100*m_all.get('chain_rate',0):.1f}% "
            f"Tone%={100*m_all.get('exact_tone_hit_rate',0):.1f}% gt9={ablation_res[name]['all_16']['total_gt9db_jumps']}",
            flush=True,
        )

    # 3. Part B: Scaling Law (50 -> 150 -> 300 -> 458 training utterances)
    print("\nRunning Part B (Data Scaling Curve 50 -> 150 -> 300 -> 458)...", flush=True)
    scale_banks = [("50", bank_50_sil), ("150", bank_150_sil), ("300", bank_300_sil), ("458", bank_458_sil)]
    v14_auto_cfg = ablation_specs[6][2]
    v14_guided_cfg = ablation_specs[8][2]
    scaling_res: dict[str, Any] = {}

    for label_n, bnk in scale_banks:
        scaling_res[label_n] = {"bank_stats": bnk.stats}
        for mode_name, cfg_m, is_g in [("v14_auto", v14_auto_cfg, False), ("v14_guided", v14_guided_cfg, True)]:
            by_split_s: dict[str, list[dict[str, Any]]] = {"all_16": [], "user_6": [], "clean_10": []}
            for uid in eval_ids:
                rd = ref_data[uid]
                pl_in = rd["plan_g"] if is_g else plan_v13(rd["u"]["text"])
                ts = time.perf_counter()
                y_syn, info = render_v14_utterance(
                    pl_in, bnk, tgt_prof,
                    guided_controls=rd["ctrls_g"] if is_g else None,
                    cfg=cfg_m, prosody_model_v14=pm_v14, glottal_bank=gbank_held, seed=72,
                )
                rtf = (time.perf_counter() - ts) / max(1e-6, len(y_syn) / SR)
                sc = _score_utterance_v14(
                    y_syn, rd["xa"], pl_in, rd["oracle_durs"], rd["n_sp_fr"],
                    info=info, lead_fr=LEAD_SIL_FRAMES, rtf=rtf,
                )
                by_split_s["all_16"].append(sc)
                by_split_s[rd["split"]].append(sc)
            scaling_res[label_n][mode_name] = {k: _summarize_metrics(v) for k, v in by_split_s.items()}
            ms = scaling_res[label_n][mode_name]["all_16"]["medians"]
            print(
                f"  N={label_n:>3s} {mode_name:10s} | Mel={ms['mel_lsd_db']:.3f} MCD={ms['mcd_db']:.2f} "
                f"Chain={100*ms.get('chain_rate',0):.1f}% Tone%={100*ms.get('exact_tone_hit_rate',0):.1f}% "
                f"Fallback%={100*ms.get('fallback_rate',0):.1f}% gt9={scaling_res[label_n][mode_name]['all_16']['total_gt9db_jumps']}",
                flush=True,
            )

    # 4. Part C: Open-Domain Free-Text Benchmark (6 unseen Mandarin sentences)
    print("\nRunning Part C (Open-Domain Benchmark Texts)...", flush=True)
    bdoc = json.loads((REPO_ROOT / "exp_v5_arch" / "benchmark_texts.json").read_text(encoding="utf-8"))
    bench_sents = [s for s in bdoc.get("sentences", []) if s.get("id") in OPEN_DOMAIN_IDS]
    cfg_v12_a = ablation_specs[0][2]
    open_domain_res: list[dict[str, Any]] = []
    open_ab_items: list[dict[str, Any]] = []
    rng = random.Random(20261123)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for s_item in bench_sents:
        sid = s_item["id"]
        txt = s_item["text"]
        pl10 = plan_v10(txt)
        pl13 = plan_v13(txt)
        y12, info12 = render_v14_utterance(pl10, bank_50_nosil, tgt_prof, cfg=cfg_v12_a, prosody_model_v12=pm_v12, glottal_bank=gbank_held, seed=72)
        y14, info14 = render_v14_utterance(pl13, bank_458_sil, tgt_prof, cfg=v14_auto_cfg, prosody_model_v14=pm_v14, glottal_bank=gbank_held, seed=72)

        def _open_metrics(y: np.ndarray, pl: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
            sanity = check_wav_sanity(y, SR)
            u_eval = evaluate_v11_utterance(
                y, pl, sr=SR, fmin=75.0, fmax=450.0,
                slot_timings_ms=info.get("slot_timings_ms"),
                unit_boundaries_ms=info.get("unit_boundaries_ms"),
                log_env=info.get("log_env"),
                boundary_frames=info.get("boundary_frames"),
            )
            return {
                "sanity_ok": bool(sanity["ok"] and u_eval["sanity_ok"]),
                "duration_s": sanity["duration_s"],
                "chain_rate": info["chain_rate"],
                "exact_tone_hit_rate": info["exact_tone_hit_rate"],
                "fallback_rate": info["fallback_rate"],
                "n_gt9db": int(u_eval["loudness_jumps"]["n_gt9db"]),
                "clicks_gt8": round(float(u_eval["v9"]["click_outlier_rate_gt8"]), 5),
                "uv_2_8k_db": round(float(u_eval["v9"]["clarity"]["unvoiced_2_8k_vs_lo_db"]), 4),
            }

        m12 = _open_metrics(y12, pl10, info12)
        m14 = _open_metrics(y14, pl13, info14)
        open_domain_res.append({"id": sid, "text": txt, "v12_auto_50utt": m12, "v14_auto_458utt": m14})

        pair = [("v12 Auto (50句)", y12, "v12_auto_50utt"), ("v14 Auto (458句+连拼+自然F0)", y14, "v14_auto_458utt")]
        rng.shuffle(pair)
        ab_items = []
        for ltr, (desc, wav_arr, code) in zip(["A", "B"], pair):
            fn = f"open_{sid}_{ltr}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fn, _rms_norm(wav_arr))
            ab_items.append({"label": ltr, "file": f"samples/{fn}", "system_code": code, "system_desc": desc})
        open_ab_items.append({"id": sid, "title": f"开放域 {sid}", "text": txt, "items": ab_items})

    # 5. Part D: Export Normalized Blind WAVs & build 100%-TTS listen.html
    user6_blind: list[dict[str, Any]] = []
    for idx_u, uid in enumerate(USER_6_IDS, 1):
        u = utt_map[uid]
        c_dict = audio_out_cache[uid]
        tracks_4 = [
            ("Real 真人原录音", c_dict["real"], "real"),
            ("v12 Auto (50句)", c_dict["A0_v12_auto_50utt"], "v12_auto_50utt"),
            ("v14 Auto (458句全自动)", c_dict["A6_+natural_f0_damped_dur"], "v14_auto_458utt"),
            ("v14 Guided (458句参考节奏音高)", c_dict["G1_v14_guided_458utt"], "v14_guided_458utt"),
        ]
        rng.shuffle(tracks_4)
        items_4 = []
        for ltr, (desc, wav_arr, code) in zip(["A", "B", "C", "D"], tracks_4):
            fn = f"user6_Q0{idx_u}_{uid}_{ltr}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fn, _rms_norm(wav_arr))
            items_4.append({"label": ltr, "file": f"samples/{fn}", "system_code": code, "system_desc": desc})
        user6_blind.append({
            "id": f"Q0{idx_u}",
            "uid": uid,
            "title": f"USER_6 Q0{idx_u} ({uid})",
            "text": u["text"],
            "pinyin": " ".join(u["pinyin"]),
            "items": items_4,
        })

    clean6_blind: list[dict[str, Any]] = []
    for idx_c, uid in enumerate(clean_10[:6], 1):
        u = utt_map[uid]
        c_dict = audio_out_cache[uid]
        tracks_3 = [
            ("Real 真人原录音", c_dict["real"], "real"),
            ("v13 Auto (50句)", c_dict["A1_v13_auto_50utt"], "v13_auto_50utt"),
            ("v14 Auto (458句全自动)", c_dict["A6_+natural_f0_damped_dur"], "v14_auto_458utt"),
        ]
        rng.shuffle(tracks_3)
        items_3 = []
        for ltr, (desc, wav_arr, code) in zip(["A", "B", "C"], tracks_3):
            fn = f"clean6_C0{idx_c}_{uid}_{ltr}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fn, _rms_norm(wav_arr))
            items_3.append({"label": ltr, "file": f"samples/{fn}", "system_code": code, "system_desc": desc})
        clean6_blind.append({
            "id": f"C0{idx_c}",
            "uid": uid,
            "title": f"CLEAN_6 C0{idx_c} ({uid})",
            "text": u["text"],
            "pinyin": " ".join(u["pinyin"]),
            "items": items_3,
        })

    html_path = build_listen_html_v14(user6_blind, clean6_blind, open_ab_items)

    results = {
        "runtime_s": round(time.perf_counter() - t_start, 2),
        "heldout_user6": list(USER_6_IDS),
        "heldout_clean10": list(clean_10),
        "ablation": ablation_res,
        "scaling": scaling_res,
        "open_domain": open_domain_res,
        "blind_mapping": {
            "user6": user6_blind,
            "clean6": clean6_blind,
            "open_domain": open_ab_items,
        },
        "listen_html": str(html_path),
    }
    (EXP_DIR / "results_v14.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nDone in {results['runtime_s']}s. Listening UI: {html_path}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate v14 multi-sample fluent Mandarin TTS.")
    ap.parse_args()
    run_eval_v14()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
