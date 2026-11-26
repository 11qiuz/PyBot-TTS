"""Phase 9 (v13) Single-Factor Ablation, 26-Utterance ZH LOO-CV Benchmark & Blind Listening Export."""
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
from exp_v5_eval.metrics import (  # noqa: E402
    check_wav_sanity,
    compute_spectral_distances,
)
from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v7_pybot_tts import unit_bank as ub  # noqa: E402
from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10  # noqa: E402
from exp_v11_align_gmm.timbre_cloner_v5 import extract_timbre_v5  # noqa: E402
from exp_v11_eval.v11_metrics import evaluate_v11_utterance  # noqa: E402
from exp_v12_natural_vocoder.vocoder_v12 import (  # noqa: E402
    VocoderFeaturesV12,
    build_glottal_pulse_bank,
    synthesize_v12,
)
from exp_v12_prosody.prosody_model import DEFAULT_MODEL_PATH, load_prosody_v12  # noqa: E402
from exp_v13_utau_ceiling import (  # noqa: E402
    EXP_DIR,
    EXTENDED_INDEX,
    SAMPLES_DIR,
    SPOTLIGHT_6_IDS,
    SR,
    USER_6_IDS,
)
from exp_v13_utau_ceiling.diagnose_v13 import _compute_f0_mae_st, _frame_aligned_mel_lsd  # noqa: E402
from exp_v13_utau_ceiling.sandhi_v13 import plan_v13  # noqa: E402
from exp_v13_utau_ceiling.v13_renderer import V13Config, render_v13_utterance  # noqa: E402
from exp_v13_utau_ceiling.zh_bank_v13 import (  # noqa: E402
    V13ZhBank,
    _analyze_or_load_zh_features,
    build_v13_zh_bank,
    filter_v13_bank_excluding,
    load_gmm_zh_map,
    load_zh_manifest_utterances,
)

LEAD_SIL_FRAMES = 4
ZH_LOO_EXCL_PROMPTS = {"SSB05020005", "SSB05020077", "SSB05020082", "SSB15930031", "SSB15930079", "SSB15930105"}


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


def _build_oracle_plan_and_controls(
    u: dict[str, Any],
    feats_real: vocoder.VocoderFeatures,
    ge: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float], int]:
    """Build exact pronunciation, duration, and F0 guided controls from ground truth."""
    units: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    oracle_durs_ms: list[float] = []
    gs_list = ge["syllables"]
    for syl, gs in zip(u["syllables"], gs_list):
        o, f, tone, ipas = cf.pinyin_to_ipa(syl["pinyin"])
        if o in ("y", "w"):
            o = ""
        if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
            f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"
        gs_s, gs_e = int(gs["syl_start"]), int(gs["syl_end"])
        nf = max(3, gs_e - gs_s)
        dur_ms = nf * 5.0
        oracle_durs_ms.append(dur_ms)
        units.append({
            "lang": "zh",
            "hanzi": syl["hanzi"],
            "token": syl["hanzi"],
            "onset": o,
            "final": f,
            "tone": tone,
            "lexical_tone": tone,
            "pinyin": syl["pinyin"],
            "ipa_phones": ipas,
            "stress_or_tone": f"T{tone}",
            "duration_ms": dur_ms,
            "pause_after_ms": 0.0,
            "pitch_st": [0.0] * 5,
        })
        controls.append({
            "onset": o,
            "final": f,
            "tone": tone,
            "dur_ms": dur_ms,
            "f0_hz": feats_real.f0_hz[gs_s:gs_e],
            "voiced_hint": feats_real.voiced[gs_s:gs_e],
            "pause_after_ms": 0.0,
        })
    plan_doc = {"schema": 2, "text": u["text"], "style": "guided", "languages": ["zh"], "units": units}
    n_speech_frames = int(gs_list[-1]["syl_end"]) if gs_list else len(feats_real.f0_hz)
    return plan_doc, controls, oracle_durs_ms, n_speech_frames


def _score_candidate(
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

    sc = info.get("source_counts") or {}
    n_zh_whole = sc.get("zh_target_syl", 0) + sc.get("zh_donor_syl", 0)
    n_zh_tot = n_zh_whole + sc.get("zh_concat_of", 0) + sc.get("zh_ipa_fill", 0)

    return {
        "sanity_ok": bool(v11_u["sanity_ok"]),
        "mel_lsd_db": round(float(sd["mel_lsd_db"]), 4) if sd else float("nan"),
        "mcd_db": round(float(sd["mcd_db"]), 4) if sd else float("nan"),
        "mcd_cmn_db": round(float(sd["mcd_cmn_db"]), 4) if sd else float("nan"),
        "nodtw_shape_mel_lsd_db": round(float(fa["nodtw_shape_mel_lsd_db"]), 4),
        "nodtw_raw_mel_lsd_db": round(float(fa["nodtw_raw_mel_lsd_db"]), 4),
        "f0_frame_mae_st": round(float(f0_fr_mae), 4) if math.isfinite(f0_fr_mae) else float("nan"),
        "f0_med_err_st": round(float(f0_med_err), 4) if math.isfinite(f0_med_err) else float("nan"),
        "dur_mae_ms": round(float(dur_mae_ms), 2) if math.isfinite(dur_mae_ms) else float("nan"),
        "voiced_contrast_db": round(float(v9["clarity"]["voiced_spectral_contrast_db"]), 4),
        "uv_2_8k_vs_lo_db": round(float(v9["clarity"]["unvoiced_2_8k_vs_lo_db"]), 4),
        "clicks_gt8": round(float(v9["click_outlier_rate_gt8"]), 5),
        "n_gt9db": int(lj["n_gt9db"]),
        "n_dropped_slots": int(nd["n_dropped_slots"]),
        "zh_whole_syl_pct": round(100.0 * n_zh_whole / max(1, n_zh_tot), 2) if n_zh_tot else None,
        "same_utt_chain_rate": info.get("same_utt_chain_rate"),
        "rtf": round(float(rtf), 4),
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    med_keys = [
        "mel_lsd_db", "mcd_db", "mcd_cmn_db",
        "nodtw_shape_mel_lsd_db", "nodtw_raw_mel_lsd_db",
        "f0_frame_mae_st", "f0_med_err_st", "dur_mae_ms",
        "voiced_contrast_db", "uv_2_8k_vs_lo_db", "clicks_gt8",
        "zh_whole_syl_pct", "same_utt_chain_rate", "rtf",
    ]
    meds: dict[str, float] = {}
    for k in med_keys:
        vals = [r[k] for r in rows if r.get(k) is not None and math.isfinite(float(r[k]))]
        if vals:
            meds[k] = round(float(np.median(vals)), 4)
    return {
        "n": len(rows),
        "sanity_ok": sum(1 for r in rows if r["sanity_ok"]),
        "total_gt9db_jumps": sum(int(r["n_gt9db"]) for r in rows),
        "total_dropped_slots": sum(int(r["n_dropped_slots"]) for r in rows),
        "medians": meds,
    }


def run_ablation_on_split(
    eval_ids: Iterable[str],
    split_label: str,
) -> tuple[dict[str, Any], V13Config, dict[str, dict[str, np.ndarray]]]:
    """Run controlled single-factor + combination ablations on a 6-utterance zero-leakage split."""
    id_tuple = tuple(eval_ids)
    zh_utts = load_zh_manifest_utterances()
    gmm_zh_map = load_gmm_zh_map()
    feats_cache = _analyze_or_load_zh_features(zh_utts, gmm_zh_map, use_cache=True)
    utt_map = {u["id"]: u for u in zh_utts}

    bank_v12bug = build_v13_zh_bank(exclude_ids=id_tuple, use_cache=True, fix_f_split=False)
    bank_v13 = build_v13_zh_bank(exclude_ids=id_tuple, use_cache=True, fix_f_split=True)
    tgt_prof = bank_v13.profiles_v5["SSB0502"]

    variants: list[tuple[str, V13ZhBank, V13Config]] = [
        ("0_v12_guided", bank_v12bug, V13Config(
            fix_viterbi_join=False, soft_context_and_f0_sel=False, joint_of_concat=False,
            gentle_gain=False, steady_nucleus_stretch=False, allow_pingpong=False,
            spectral_seam_join=False, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=False,
        )),
        ("1_+fix_f_split", bank_v13, V13Config(
            fix_viterbi_join=False, soft_context_and_f0_sel=False, joint_of_concat=False,
            gentle_gain=False, steady_nucleus_stretch=False, allow_pingpong=False,
            spectral_seam_join=False, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=False,
        )),
        ("2_+fix_viterbi_join", bank_v13, V13Config(
            fix_viterbi_join=True, soft_context_and_f0_sel=False, joint_of_concat=False,
            gentle_gain=False, steady_nucleus_stretch=False, allow_pingpong=False,
            spectral_seam_join=False, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=False,
        )),
        ("3_+soft_context_f0", bank_v13, V13Config(
            fix_viterbi_join=False, soft_context_and_f0_sel=True, joint_of_concat=False,
            gentle_gain=False, steady_nucleus_stretch=False, allow_pingpong=False,
            spectral_seam_join=False, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=False,
        )),
        ("4_+joint_of_concat", bank_v13, V13Config(
            fix_viterbi_join=False, soft_context_and_f0_sel=False, joint_of_concat=True,
            gentle_gain=False, steady_nucleus_stretch=False, allow_pingpong=False,
            spectral_seam_join=False, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=False,
        )),
        ("5_+gentle_gain", bank_v13, V13Config(
            fix_viterbi_join=False, soft_context_and_f0_sel=False, joint_of_concat=False,
            gentle_gain=True, steady_nucleus_stretch=False, allow_pingpong=False,
            spectral_seam_join=False, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=False,
        )),
        ("6_+steady_nucleus", bank_v13, V13Config(
            fix_viterbi_join=False, soft_context_and_f0_sel=False, joint_of_concat=False,
            gentle_gain=False, steady_nucleus_stretch=True, allow_pingpong=True,
            spectral_seam_join=False, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=False,
        )),
        ("7_+spectral_seam", bank_v13, V13Config(
            fix_viterbi_join=False, soft_context_and_f0_sel=False, joint_of_concat=False,
            gentle_gain=False, steady_nucleus_stretch=False, allow_pingpong=False,
            spectral_seam_join=True, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=False,
        )),
        ("8_+f0_tilt_comp", bank_v13, V13Config(
            fix_viterbi_join=False, soft_context_and_f0_sel=False, joint_of_concat=False,
            gentle_gain=False, steady_nucleus_stretch=False, allow_pingpong=False,
            spectral_seam_join=False, f0_tilt_comp=True, use_sandhi_v13=False, use_learned_prosody=False,
        )),
        ("9_v13_all_no_tilt", bank_v13, V13Config(
            fix_viterbi_join=True, soft_context_and_f0_sel=True, joint_of_concat=True,
            gentle_gain=True, steady_nucleus_stretch=True, allow_pingpong=True,
            spectral_seam_join=True, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=False,
        )),
        ("10_v13_sel_core", bank_v13, V13Config(
            fix_viterbi_join=True, soft_context_and_f0_sel=True, joint_of_concat=True,
            gentle_gain=False, steady_nucleus_stretch=False, allow_pingpong=False,
            spectral_seam_join=False, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=False,
        )),
        ("11_v13_sel_gentle", bank_v13, V13Config(
            fix_viterbi_join=True, soft_context_and_f0_sel=True, joint_of_concat=True,
            gentle_gain=True, steady_nucleus_stretch=False, allow_pingpong=False,
            spectral_seam_join=False, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=False,
        )),
        ("12_v13_sel_sust_nuc", bank_v13, V13Config(
            fix_viterbi_join=True, soft_context_and_f0_sel=True, joint_of_concat=True,
            gentle_gain=True, steady_nucleus_stretch=True, steady_stretch_min_ratio=1.35, allow_pingpong=True,
            spectral_seam_join=False, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=False,
        )),
    ]

    rows_by_var: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in variants}
    audio_cache: dict[str, dict[str, np.ndarray]] = {uid: {} for uid in id_tuple}

    for uid in id_tuple:
        u = utt_map[uid]
        rec = feats_cache[uid]
        xa = np.asarray(rec["xa"], dtype=np.float64)
        feats_real = rec["feats"]
        b_gt8 = np.asarray(rec["bap_gt8k"], dtype=np.float64)
        ge = gmm_zh_map[uid]
        plan_doc, controls, oracle_durs, n_sp_fr = _build_oracle_plan_and_controls(u, feats_real, ge)

        audio_cache[uid]["real"] = xa
        y_cs = synthesize_v12(
            feats_real,
            mode="hybrid_lp_hnm",
            glottal_bank=bank_v13.glottal_banks["SSB0502"],
            crisp_unvoiced=True,
            seed=72,
            phase_dispersion=True,
            bap_gt8k=b_gt8,
        )
        audio_cache[uid]["copy_synth"] = y_cs

        for name, b_use, cfg in variants:
            ts = time.perf_counter()
            y, info = render_v13_utterance(
                plan_doc, b_use, tgt_prof, guided_controls=controls, cfg=cfg, seed=72,
            )
            rtf = (time.perf_counter() - ts) / max(1e-6, len(y) / SR)
            row = _score_candidate(y, xa, plan_doc, oracle_durs, n_sp_fr, info=info, lead_fr=LEAD_SIL_FRAMES, rtf=rtf)
            row["uid"] = uid
            rows_by_var[name].append(row)
            audio_cache[uid][name] = y

    summary = {name: _summarize_rows(rows_by_var[name]) for name, _, _ in variants}
    base_med = summary["0_v12_guided"]["medians"]
    base_gt9 = summary["0_v12_guided"]["total_gt9db_jumps"]

    def _is_clean_gain(cand_sum: dict[str, Any]) -> bool:
        m = cand_sum["medians"]
        return (
            cand_sum["sanity_ok"] == summary["0_v12_guided"]["sanity_ok"]
            and cand_sum["total_dropped_slots"] <= summary["0_v12_guided"]["total_dropped_slots"]
            and cand_sum["total_gt9db_jumps"] <= base_gt9 + 2
            and m["clicks_gt8"] <= base_med["clicks_gt8"] + 0.002
        )

    best_name, best_cfg = "11_v13_sel_gentle", variants[11][2]
    best_score = float("inf")
    for cname, _, ccfg in variants[9:]:
        cs = summary[cname]
        if not _is_clean_gain(cs):
            continue
        m = cs["medians"]
        sc_val = m["mel_lsd_db"] + 0.08 * m["mcd_db"] + 0.20 * m["f0_frame_mae_st"]
        if sc_val < best_score:
            best_score = sc_val
            best_name, best_cfg = cname, ccfg

    print(f"\n=== ABLATIONS ON {split_label} (n={len(id_tuple)}, ZERO LEAKAGE) ===")
    print(f"{'Variant':22s} | {'Mel-LSD':>7s} | {'MCD':>7s} | {'NoDTW':>6s} | {'F0-MAE':>6s} | {'Chain':>5s} | {'gt9':>3s} | {'Clk':>7s}")
    print("-" * 86)
    for name, _, _ in variants:
        s = summary[name]
        m = s["medians"]
        print(
            f"{name:22s} | {m['mel_lsd_db']:7.4f} | {m['mcd_db']:7.4f} | {m['nodtw_shape_mel_lsd_db']:6.3f} | "
            f"{m['f0_frame_mae_st']:6.3f} | {str(m.get('same_utt_chain_rate')):>5s} | {s['total_gt9db_jumps']:3d} | {m['clicks_gt8']:7.5f}"
        )
    print(f"Selected best on {split_label}: {best_name} -> {asdict(best_cfg)}")

    return {
        "split_label": split_label,
        "ids": list(id_tuple),
        "summary": summary,
        "rows": rows_by_var,
        "selected_name": best_name,
        "selected_cfg": asdict(best_cfg),
    }, best_cfg, audio_cache


def build_listen_html(
    blind_groups: list[dict[str, Any]],
    auto_ab_groups: list[dict[str, Any]],
    singing_demos: dict[str, Any],
    out_path: Path | None = None,
) -> Path:
    """Generate standalone offline listen.html (zero fetch(), relative samples/*.wav paths)."""
    if out_path is None:
        out_path = EXP_DIR / "listen.html"

    payload = json.dumps(
        {
            "blind_groups": blind_groups,
            "auto_ab_groups": auto_ab_groups,
            "singing_demos": singing_demos,
        },
        ensure_ascii=False,
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-TTS v13 · 盲听与虚拟歌姬对比台</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#eef1f4;background:#11151c;max-width:960px;margin:1.8rem auto;padding:0 1rem}}
h1{{font-size:1.65rem;margin-bottom:.3rem}}
.sub{{color:#9aa6b8;margin-bottom:1.2rem}}
.tabs{{display:flex;gap:.6rem;margin-bottom:1.2rem;flex-wrap:wrap}}
.tab{{background:#1c2330;color:#c9d3e0;border:1px solid #2e394d;padding:.55rem 1rem;border-radius:8px;cursor:pointer}}
.tab.active{{background:#2f6f58;color:#fff;border-color:#449c7c}}
.panel{{display:none}}.panel.active{{display:block}}
.card{{background:#19202b;border:1px solid #293345;padding:1rem 1.2rem;margin:1rem 0;border-radius:10px}}
.row{{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;padding:.4rem 0;border-bottom:1px dashed #273142}}
.row:last-child{{border-bottom:none}}
audio{{width:min(100%,24rem);height:34px}}
select,input[type=text],textarea{{background:#10151e;color:#eef1f4;border:1px solid #35435b;border-radius:6px;padding:.35rem .55rem;font:inherit}}
input[type=text]{{width:100%}}
button.btn{{background:#2c5e4b;color:#fff;border:none;border-radius:6px;padding:.5rem .95rem;cursor:pointer;margin:.4rem .5rem .2rem 0}}
button.btn.sec{{background:#2b364a}}
.ans{{display:none;margin-top:.55rem;padding:.55rem .8rem;background:#12211c;border-left:3px solid #449c7c;border-radius:4px;font-size:.92rem;color:#cfece0}}
.ans.show{{display:block}}
.meta{{color:#9aa6b8;font-size:.9rem}}
.pill{{display:inline-block;padding:.12rem .5rem;border-radius:999px;background:#273449;font-size:.82rem;color:#c8d4e6;margin-right:.35rem}}
</style>
</head>
<body>
<h1>PyBot-TTS v13 · 选音拼接修复 &amp; 虚拟歌姬试听台</h1>
<p class="sub">完全离线运行（双击即可播放）。所有音频均已统一响度（-20 dBFS）且按句随机打乱顺序。</p>

<div class="tabs">
  <button class="tab active" onclick="showTab('blind', this)">1. 盲听四路对照 (Q01–Q06)</button>
  <button class="tab" onclick="showTab('auto', this)">2. 自动合成对比 (v12 vs v13)</button>
  <button class="tab" onclick="showTab('sing', this)">3. 虚拟歌姬 (UTAU Ping-Pong vs 线性拉伸)</button>
</div>

<div id="tab-blind" class="panel active"></div>
<div id="tab-auto" class="panel"></div>
<div id="tab-sing" class="panel"></div>

<div class="card">
  <button class="btn" onclick="toggleAnswers()">显示 / 隐藏答案与原文</button>
  <button class="btn sec" onclick="exportScores()">导出我的盲听记录 (JSON)</button>
  <span id="status" class="meta"></span>
</div>

<script>
const DATA = {payload};
const STATE = {{ dictation: {{}}, naturalness: {{}}, favorite: {{}}, notes: {{}} }};
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

function renderBlind() {{
  const root = document.getElementById('tab-blind');
  root.replaceChildren();
  DATA.blind_groups.forEach(g => {{
    const card = el('div', 'card');
    card.append(el('h3', '', g.title));
    const dict = el('input');
    dict.type = 'text';
    dict.placeholder = '先只听 A 段或任意一段写下你听到的句子（避免先看答案）…';
    dict.oninput = () => STATE.dictation[g.id] = dict.value;
    card.append(dict);

    g.items.forEach(it => {{
      const r = el('div', 'row');
      const b = el('b', '', it.label + ' 段');
      const au = el('audio'); au.controls = true; au.preload = 'none'; au.src = it.file;
      const sel = el('select');
      ['未评分','1 (极生硬)','2','3 (一般)','4','5 (真人级)'].forEach((lab, idx) => sel.add(new Option(lab, idx===0?'':String(idx))));
      sel.onchange = () => STATE.naturalness[g.id + '-' + it.label] = sel.value;
      r.append(b, au, sel);
      card.append(r);
    }});

    const favRow = el('div', 'row');
    favRow.append(el('span', 'meta', '最自然：'));
    const fav = el('select');
    ['未选择','A','B','C','D','难分伯仲'].forEach(x => fav.add(new Option(x, x==='未选择'?'':x)));
    fav.onchange = () => STATE.favorite[g.id] = fav.value;
    const note = el('input'); note.type='text'; note.placeholder='备注（拖音/爆破/声调/接缝感）'; note.style.flex='1';
    note.oninput = () => STATE.notes[g.id] = note.value;
    favRow.append(fav, note);
    card.append(favRow);

    const ans = el('div', 'ans');
    ans.textContent = `原文：${{g.text}} (${{g.pinyin}}) ｜ 映射：` + g.items.map(x => `${{x.label}}=${{x.system_desc}}`).join('；');
    card.append(ans);
    root.append(card);
  }});
}}

function renderAuto() {{
  const root = document.getElementById('tab-auto');
  root.replaceChildren();
  DATA.auto_ab_groups.forEach(g => {{
    const card = el('div', 'card');
    card.append(el('h3', '', g.title + ' · ' + g.text));
    g.items.forEach(it => {{
      const r = el('div', 'row');
      const au = el('audio'); au.controls = true; au.preload = 'none'; au.src = it.file;
      r.append(el('b', '', it.label + ' 段'), au);
      card.append(r);
    }});
    const ans = el('div', 'ans');
    ans.textContent = '映射：' + g.items.map(x => `${{x.label}}=${{x.system_desc}}`).join('；');
    card.append(ans);
    root.append(card);
  }});
}}

function renderSing() {{
  const root = document.getElementById('tab-sing');
  root.replaceChildren();
  Object.entries(DATA.singing_demos).forEach(([k, d]) => {{
    const card = el('div', 'card');
    card.append(el('h3', '', d.title));
    card.append(el('p', 'meta', d.description));
    const r1 = el('div', 'row');
    const a1 = el('audio'); a1.controls=true; a1.preload='none'; a1.src=d.v12_linear.file;
    r1.append(el('span', 'pill', 'v12 线性拉伸'), a1, el('span', 'meta', `F0-MAE=${{d.v12_linear.metrics.f0_steady_mae_st}}st`));
    const r2 = el('div', 'row');
    const a2 = el('audio'); a2.controls=true; a2.preload='none'; a2.src=d.v13_utau.file;
    r2.append(el('span', 'pill', 'v13 稳态循环+滑音颤音'), a2, el('span', 'meta', `F0-MAE=${{d.v13_utau.metrics.f0_steady_mae_st}}st | PingPong=${{d.v13_utau.metrics.pingpong_count}}`));
    const r3 = el('div', 'row');
    const a3 = el('audio'); a3.controls=true; a3.preload='none'; a3.src=d.ab_file;
    r3.append(el('span', 'pill', 'AB 连续对照 (v12 -> v13)'), a3);
    card.append(r1, r2, r3);
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
  a.download = 'pybot-v13-listening.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  document.getElementById('status').textContent = ' 已导出 pybot-v13-listening.json';
}}

renderBlind();
renderAuto();
renderSing();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


def run_26utt_loo_and_export(best_guided_cfg: V13Config | None = None) -> dict[str, Any]:
    """Run ablations on both splits, 26-utt ZH LOO-CV, singing demos, and export listen.html."""
    t0 = time.perf_counter()
    ab_spot, best_spot_cfg, _ = run_ablation_on_split(SPOTLIGHT_6_IDS, "SPOTLIGHT_6")
    ab_user, best_user_cfg, user_audio = run_ablation_on_split(USER_6_IDS, "USER_6 (Q01..Q06)")

    if best_guided_cfg is None:
        best_guided_cfg = best_user_cfg

    zh_utts = load_zh_manifest_utterances()
    gmm_zh_map = load_gmm_zh_map()
    feats_cache = _analyze_or_load_zh_features(zh_utts, gmm_zh_map, use_cache=True)

    ext_utts = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
    eval_utts = [u for u in ext_utts if u["id"] not in ZH_LOO_EXCL_PROMPTS]
    utt_by_id = {u["id"]: u for u in zh_utts}

    print("\nBuilding full v12bug and v13 ZH banks for fast zero-leakage LOO folding...", flush=True)
    bank_v12_full = build_v13_zh_bank(use_cache=True, fix_f_split=False)
    bank_v13_full = build_v13_zh_bank(use_cache=True, fix_f_split=True)
    prosody_model = load_prosody_v12(DEFAULT_MODEL_PATH)
    zh_for_ub = {u["id"]: (np.asarray(feats_cache[u["id"]]["xa"], dtype=np.float64), feats_cache[u["id"]]["feats"]) for u in zh_utts}

    v12_guided_cfg = V13Config(
        fix_viterbi_join=False, soft_context_and_f0_sel=False, joint_of_concat=False,
        gentle_gain=False, steady_nucleus_stretch=False, allow_pingpong=False,
        spectral_seam_join=False, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=False,
    )
    v12_auto_cfg = V13Config(
        fix_viterbi_join=False, soft_context_and_f0_sel=False, joint_of_concat=False,
        gentle_gain=False, steady_nucleus_stretch=False, allow_pingpong=False,
        spectral_seam_join=False, f0_tilt_comp=False, use_sandhi_v13=False, use_learned_prosody=True,
    )
    v13_auto_cfg = V13Config(
        **{**asdict(best_guided_cfg), "use_sandhi_v13": True, "use_learned_prosody": True, "singing_mode": False}
    )

    tracks = ["real", "copy_synth", "v12_guided", "v13_guided", "v12_auto", "v13_auto"]
    loo_rows: dict[str, list[dict[str, Any]]] = {k: [] for k in tracks}

    print(f"Evaluating 26-utterance zero-leakage ZH LOO-CV across {len(eval_utts)} utterances...", flush=True)
    for idx_z, u_ext in enumerate(eval_utts, 1):
        uid, spk = u_ext["id"], u_ext["speaker"]
        u = utt_by_id[uid]
        rec = feats_cache[uid]
        xa = np.asarray(rec["xa"], dtype=np.float64)
        feats_real = rec["feats"]
        b_gt8 = np.asarray(rec["bap_gt8k"], dtype=np.float64)
        ge = gmm_zh_map[uid]
        plan_g, controls_g, oracle_durs, n_sp_fr = _build_oracle_plan_and_controls(u, feats_real, ge)

        train_meta_other = [x for x in zh_utts if x["id"] != uid]
        ub_fold = ub.build_unit_bank(train_meta_other, exclude_ids={uid}, precomputed=zh_for_ub, sr=SR)
        spk_items = [zh_for_ub[x["id"]] for x in train_meta_other if x["speaker"] == spk][:12]
        t_v5 = extract_timbre_v5(spk_items, sr=SR, name=spk, corpus="aishell3")
        gb_items = [
            VocoderFeaturesV12(
                feats=feats_cache[x["id"]]["feats"],
                bap_gt8k=np.asarray(feats_cache[x["id"]]["bap_gt8k"], dtype=np.float64),
                lp_pulses=feats_cache[x["id"]]["lp_pulses"],
                coherent_phasor=feats_cache[x["id"]]["coherent_phasor"],
            )
            for x in train_meta_other if x["speaker"] == spk
        ][:8]
        gb_fold = build_glottal_pulse_bank(spk, gb_items, max_pulses=32)

        b12_f = filter_v13_bank_excluding(bank_v12_full, {uid}, ub_fold=ub_fold, spk_override=(spk, t_v5), glottal_override=(spk, gb_fold))
        b13_f = filter_v13_bank_excluding(bank_v13_full, {uid}, ub_fold=ub_fold, spk_override=(spk, t_v5), glottal_override=(spk, gb_fold))

        loo_rows["real"].append(_score_candidate(xa, xa, plan_g, oracle_durs, n_sp_fr, lead_fr=0, rtf=0.0))

        y_cs = synthesize_v12(feats_real, mode="hybrid_lp_hnm", glottal_bank=gb_fold, crisp_unvoiced=True, seed=72, phase_dispersion=True, bap_gt8k=b_gt8)
        loo_rows["copy_synth"].append(_score_candidate(y_cs, xa, plan_g, oracle_durs, n_sp_fr, lead_fr=0, rtf=0.03))

        ts = time.perf_counter()
        y12g, m12g = render_v13_utterance(plan_g, b12_f, t_v5, guided_controls=controls_g, cfg=v12_guided_cfg, glottal_bank=gb_fold, seed=72)
        loo_rows["v12_guided"].append(_score_candidate(y12g, xa, plan_g, oracle_durs, n_sp_fr, info=m12g, lead_fr=LEAD_SIL_FRAMES, rtf=(time.perf_counter() - ts) / max(1e-6, len(y12g) / SR)))

        ts = time.perf_counter()
        y13g, m13g = render_v13_utterance(plan_g, b13_f, t_v5, guided_controls=controls_g, cfg=best_guided_cfg, glottal_bank=gb_fold, seed=72)
        loo_rows["v13_guided"].append(_score_candidate(y13g, xa, plan_g, oracle_durs, n_sp_fr, info=m13g, lead_fr=LEAD_SIL_FRAMES, rtf=(time.perf_counter() - ts) / max(1e-6, len(y13g) / SR)))

        pl10 = plan_v10(u["text"])
        ts = time.perf_counter()
        y12a, m12a = render_v13_utterance(pl10, b12_f, t_v5, cfg=v12_auto_cfg, prosody_model=prosody_model, glottal_bank=gb_fold, seed=72)
        loo_rows["v12_auto"].append(_score_candidate(y12a, xa, pl10, oracle_durs, n_sp_fr, info=m12a, lead_fr=LEAD_SIL_FRAMES, rtf=(time.perf_counter() - ts) / max(1e-6, len(y12a) / SR)))

        pl13 = plan_v13(u["text"])
        ts = time.perf_counter()
        y13a, m13a = render_v13_utterance(pl13, b13_f, t_v5, cfg=v13_auto_cfg, prosody_model=prosody_model, glottal_bank=gb_fold, seed=72)
        loo_rows["v13_auto"].append(_score_candidate(y13a, xa, pl13, oracle_durs, n_sp_fr, info=m13a, lead_fr=LEAD_SIL_FRAMES, rtf=(time.perf_counter() - ts) / max(1e-6, len(y13a) / SR)))

        if idx_z % 5 == 0 or idx_z == len(eval_utts):
            print(f"  LOO {idx_z}/{len(eval_utts)} done", flush=True)

    loo_summary = {k: _summarize_rows(loo_rows[k]) for k in tracks}

    # Export equal-loudness blind listening WAVs on USER_6_IDS (Q01..Q06)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    bank_user_12 = filter_v13_bank_excluding(bank_v12_full, USER_6_IDS)
    bank_user_13 = filter_v13_bank_excluding(bank_v13_full, USER_6_IDS)
    user_prof = bank_user_13.profiles_v5["SSB0502"]

    rng = random.Random(20261123)
    blind_groups: list[dict[str, Any]] = []
    auto_ab_groups: list[dict[str, Any]] = []
    best_user_key = ab_user["selected_name"]

    for idx_u, uid in enumerate(USER_6_IDS, 1):
        u = utt_by_id[uid]
        c_dict = user_audio[uid]
        tracks_4 = [
            ("Real 真人原录音", c_dict["real"], "real"),
            ("Copy-Synth 原声学参数重合成", c_dict["copy_synth"], "copy_synth"),
            ("v12 Guided 指导拼接", c_dict["0_v12_guided"], "v12_guided"),
            (f"v13 Guided ({best_user_key})", c_dict[best_user_key], "v13_guided"),
        ]
        rng.shuffle(tracks_4)
        items_4 = []
        for ltr, (desc, wav_arr, code) in zip(["A", "B", "C", "D"], tracks_4):
            fn = f"Q0{idx_u}_{uid}_{ltr}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fn, _rms_norm(wav_arr))
            items_4.append({"label": ltr, "file": f"samples/{fn}", "system_code": code, "system_desc": desc})
        blind_groups.append({
            "id": f"Q0{idx_u}",
            "uid": uid,
            "title": f"Q0{idx_u} ({uid})",
            "text": u["text"],
            "pinyin": " ".join(s["pinyin"] for s in u["syllables"]),
            "items": items_4,
        })

        y_a12, _ = render_v13_utterance(plan_v10(u["text"]), bank_user_12, user_prof, cfg=v12_auto_cfg, prosody_model=prosody_model, seed=72)
        y_a13, _ = render_v13_utterance(plan_v13(u["text"]), bank_user_13, user_prof, cfg=v13_auto_cfg, prosody_model=prosody_model, seed=72)
        pair = [("v12 Auto", y_a12, "v12_auto"), ("v13 Auto (跨词三声+fix_f_split+选音)", y_a13, "v13_auto")]
        rng.shuffle(pair)
        ab_items = []
        for ltr, (desc, wav_arr, code) in zip(["A", "B"], pair):
            fn = f"auto_Q0{idx_u}_{uid}_{ltr}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fn, _rms_norm(wav_arr))
            ab_items.append({"label": ltr, "file": f"samples/{fn}", "system_code": code, "system_desc": desc})
        auto_ab_groups.append({
            "id": f"Q0{idx_u}_auto",
            "title": f"自动合成 Q0{idx_u} ({uid})",
            "text": u["text"],
            "items": ab_items,
        })

    from exp_v13_utau_ceiling.sing_demo_v13 import run_singing_demos

    sing_res = run_singing_demos(bank=bank_v13_full, v13_cfg=best_guided_cfg)
    html_path = build_listen_html(blind_groups, auto_ab_groups, sing_res)

    results = {
        "runtime_s": round(time.perf_counter() - t0, 2),
        "spotlight_ablation": ab_spot,
        "user6_ablation": ab_user,
        "loo_26_summary": loo_summary,
        "loo_26_per_utterance": loo_rows,
        "blind_groups": blind_groups,
        "auto_ab_groups": auto_ab_groups,
        "singing_demos": sing_res,
        "listen_html": str(html_path),
    }
    out_path = EXP_DIR / "results_v13.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 26-UTTERANCE MANDARIN LOO-CV SUMMARY (ZERO LEAKAGE) ===")
    print(f"{'System':14s} | {'Mel-LSD':>7s} | {'MCD':>7s} | {'NoDTW':>6s} | {'F0-MAE':>6s} | {'DurMAE':>6s} | {'Whole%':>6s} | {'gt9':>3s} | {'Clk':>7s}")
    print("-" * 92)
    for k in tracks:
        s = loo_summary[k]
        m = s["medians"]
        print(
            f"{k:14s} | {m.get('mel_lsd_db', 0.0):7.4f} | {m.get('mcd_db', 0.0):7.4f} | {m.get('nodtw_shape_mel_lsd_db', 0.0):6.3f} | "
            f"{m.get('f0_frame_mae_st', 0.0):6.3f} | {str(m.get('dur_mae_ms', '-')):>6s} | {str(m.get('zh_whole_syl_pct')):>6s} | {s['total_gt9db_jumps']:3d} | {m.get('clicks_gt8', 0.0):7.5f}"
        )
    print(f"Generated blind listening UI: {html_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablate-only", action="store_true")
    args = parser.parse_args()
    if args.ablate_only:
        ab_spot, _, _ = run_ablation_on_split(SPOTLIGHT_6_IDS, "SPOTLIGHT_6")
        ab_user, _, _ = run_ablation_on_split(USER_6_IDS, "USER_6 (Q01..Q06)")
        (EXP_DIR / "ablation_spotlight.json").write_text(
            json.dumps({"spotlight_6": ab_spot, "user_6": ab_user}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        run_26utt_loo_and_export()


if __name__ == "__main__":
    main()
