"""Stage 0 Feasibility Probe & DEV_60 Swarm Baseline for PyBot-Larynx (`exp_20`)."""
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

from exp_20_pybot_larynx import (  # noqa: E402
    EXP_DIR,
    HOP,
    KNOWLEDGE_PATH,
    N_FFT,
    SAMPLES_DIR,
    SR,
    TRAIN_STAGE0_N,
)
from exp_20_pybot_larynx.acoustics.build_knowledge import build_stage0_knowledge  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge import (  # noqa: E402
    AcousticKnowledge,
    build_base_control,
    get_splits,
    load_v15_index_cached,
    normalize_of,
)
from exp_20_pybot_larynx.arbiter import (  # noqa: E402
    OBSTRUENT_MANNERS,
    Arbiter,
    SyllableSpan,
    UtteranceContext,
    build_utterance_context,
    classify_onset_manner,
)
from exp_20_pybot_larynx.judge import evaluate_utterance, summarize_condition  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import LarynxControl, synthesize_larynx  # noqa: E402
from exp_20_pybot_larynx.rules import default_stage0_rules  # noqa: E402
from exp_v9_clarity_pro.hnm_v9 import analyze_v9  # noqa: E402
from exp_v12_natural_vocoder.v12_bank import _patch_unsmoothed_uv_env  # noqa: E402
from exp_v13_utau_ceiling.sandhi_v13 import plan_v13  # noqa: E402
from exp_v14_fluent_zh_tts.eval_v14 import _rms_norm, _write_pcm16_wav  # noqa: E402
from exp_v14_fluent_zh_tts.renderer_v14 import _template_contour_for_unit  # noqa: E402
from exp_v15_bznsyp_tts.index_bznsyp_v15 import (  # noqa: E402
    _fast_mfcc_and_f0_24k,
    load_speaker_assets_v15,
    parse_bznsyp_corpus,
    read_bznsyp_wav_24k,
)
from exp_v15_bznsyp_tts.prosody_v15 import ProsodyModelV15, apply_prosody_v15  # noqa: E402

import importlib.util
_RZ_SPEC = importlib.util.spec_from_file_location("rz_read", REPO_ROOT / "exp_19b_speech_reverse" / "rz.py")
assert _RZ_SPEC and _RZ_SPEC.loader
_RZ_MOD = importlib.util.module_from_spec(_RZ_SPEC)
_RZ_SPEC.loader.exec_module(_RZ_MOD)
V15_10_MAP: dict[str, str] = dict(_RZ_MOD.V15_10)


def _prepare_utterance_oracle(uid: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Load `rz.span(u)` real waveform + aligned oracle frame features & syllable spans."""
    x24 = read_bznsyp_wav_24k(REPO_ROOT / meta["wav_path"])
    syls_raw = meta["syllables"]
    s0_fr = max(0, int(round(float(syls_raw[0]["start_s"]) * SR / HOP)))
    s1_fr = max(s0_fr + 2, int(round(float(syls_raw[-1]["end_s"]) * SR / HOP)))
    ref_span = x24[s0_fr * HOP : min(len(x24), s1_fr * HOP)]

    fv9, bgt8 = analyze_v9(ref_span, sr=SR)
    feats_p = _patch_unsmoothed_uv_env(ref_span, fv9, sr=SR, n_fft=N_FFT)
    bap6 = np.concatenate([feats_p.bap[:, :4], bgt8], axis=1)
    n_fr = len(feats_p.f0_hz)

    v_m = feats_p.voiced & (feats_p.f0_hz > 50.0)
    utt_f0_med = float(np.median(feats_p.f0_hz[v_m])) if np.any(v_m) else float(meta.get("utt_f0_med_hz", 247.0))

    spans: list[SyllableSpan] = []
    for idx_s, s in enumerate(syls_raw):
        o, f = normalize_of(str(s["onset"]), str(s["final"]))
        a = max(0, min(n_fr - 1, int(round(float(s["start_s"]) * SR / HOP)) - s0_fr))
        b = max(a + 1, min(n_fr, int(round(float(s["end_s"]) * SR / HOP)) - s0_fr))
        sp = max(a, min(b - 1, int(round(float(s["split_s"]) * SR / HOP)) - s0_fr))
        spans.append(SyllableSpan(
            index=idx_s,
            hanzi=str(s["hanzi"]),
            onset=o,
            final=f,
            tone=int(s["tone"]),
            start_fr=a,
            split_fr=sp,
            end_fr=b,
            pause_after_ms=float(s.get("pause_after_ms", 0.0)),
            prosodic_break=int(s.get("break", 0)),
        ))

    ctx = build_utterance_context(uid, n_fr, spans, speaker_f0_med_hz=utt_f0_med)
    return {
        "uid": uid,
        "text": str(meta["text"]),
        "ref_span": ref_span,
        "ref_fast": _fast_mfcc_and_f0_24k(ref_span),
        "feats_p": feats_p,
        "bap6": bap6,
        "ctx": ctx,
    }


def render_c4_auto_utterance(
    text: str,
    utt_id: str,
    knowledge: AcousticKnowledge,
    prosody_model: ProsodyModelV15,
    *,
    arbiter: Arbiter,
    rules: list[Any],
    glottal_mode: str = "parametric_lf",
    coherent_phasor: np.ndarray | None = None,
    seed: int = 123,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pure text-to-speech pipeline using PyBot acoustic knowledge, rules, and electronic larynx."""
    ir = plan_v13(text)
    pro_ir = apply_prosody_v15(ir, prosody_model)
    raw_units = [u for u in (pro_ir.get("units") or []) if u.get("lang", "zh") == "zh"]
    if not raw_units:
        raise ValueError(f"No Mandarin units planned for text: {text!r}")

    base_f0 = float(knowledge.speaker_f0_med_hz)
    n_u = len(raw_units)
    spans: list[SyllableSpan] = []
    f0_chunks: list[np.ndarray] = []
    voi_chunks: list[np.ndarray] = []
    syl_gains: list[float] = []

    cur_fr = 0
    for i, u in enumerate(raw_units):
        o, f = normalize_of(str(u.get("onset", "")), str(u.get("final", "")))
        tone = int(u.get("tone") or 1)
        dur_ms = max(60.0, float(u.get("duration_ms", 180.0)))
        n_syl = max(3, int(round(dur_ms / 5.0)))

        if o:
            pred_on_ms = knowledge.onset_ms.get(o, knowledge.onset_frac.get(o, 0.25) * dur_ms)
            n_on = max(1, min(n_syl - 2, int(round(pred_on_ms / 5.0))))
        else:
            n_on = 0
        split_fr = cur_fr + n_on
        end_fr = cur_fr + n_syl

        f3 = u.get("f0_3pt_st")
        if f3 is not None and len(f3) == 3:
            f3_clean = [float(f3[0]), float(f3[1]), float(f3[2])]
        else:
            tmpl = _template_contour_for_unit({"tone": tone, "f0_3pt_st": None}, 9, 1.0, 0.0)
            f3_clean = [float(tmpl[1]), float(tmpl[4]), float(tmpl[7])]

        if (i < n_u - 1) and (tone == 3):
            ceil = -1.15
            f3_clean = [min(f3_clean[0], ceil + 0.3), min(f3_clean[1], ceil), min(f3_clean[2], ceil)]

        contour_st = _template_contour_for_unit({"tone": tone, "f0_3pt_st": f3_clean}, n_syl, 1.0, 0.0)
        f0_seg = np.clip(base_f0 * (2.0 ** (contour_st / 12.0)), 65.0, 440.0)
        voi_seg = np.ones(n_syl, dtype=bool)
        if o and classify_onset_manner(o) in OBSTRUENT_MANNERS and n_on > 0:
            uv_n = max(1, min(n_on, int(round(0.78 * n_on))))
            voi_seg[:uv_n] = False
            f0_seg[:uv_n] = 0.0

        p_ms = float(u.get("pause_after_ms", 0.0)) if i < n_u - 1 else 0.0
        spans.append(SyllableSpan(
            index=i,
            hanzi=str(u.get("hanzi") or u.get("token") or ""),
            onset=o,
            final=f,
            tone=tone,
            start_fr=cur_fr,
            split_fr=split_fr,
            end_fr=end_fr,
            pause_after_ms=p_ms,
            prosodic_break=int(u.get("prosodic_break", u.get("break", 0))),
        ))
        f0_chunks.append(f0_seg)
        voi_chunks.append(voi_seg)
        syl_gains.append(float(u.get("gain_db", 0.0)))
        cur_fr = end_fr

        if i < n_u - 1 and p_ms >= 20.0:
            n_p = int(round(p_ms / 5.0))
            if n_p > 0:
                f0_chunks.append(np.zeros(n_p, dtype=np.float64))
                voi_chunks.append(np.zeros(n_p, dtype=bool))
                cur_fr += n_p

    f0_all = np.concatenate(f0_chunks)
    voi_all = np.concatenate(voi_chunks)
    ctx = build_utterance_context(utt_id, cur_fr, spans, speaker_f0_med_hz=base_f0)
    base_ctrl, kb_info = build_base_control(
        ctx, f0_all, voi_all, knowledge, mode="dynamic", syl_gain_db=np.asarray(syl_gains, dtype=np.float64)
    )
    fused_ctrl, arb_trace = arbiter.apply(ctx, base_ctrl, rules)
    t_k0 = time.perf_counter()
    wav = synthesize_larynx(
        fused_ctrl,
        glottal_mode=glottal_mode,
        coherent_phasor=coherent_phasor,
        seed=seed,
    )
    k_elapsed = time.perf_counter() - t_k0
    return wav, {
        "kb_info": kb_info,
        "arbiter_trace": arb_trace,
        "kernel_s": k_elapsed,
    }


def _render_listen_html(groups: list[dict[str, Any]], out_path: Path) -> None:
    payload_json = json.dumps(groups, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-Larynx (exp_20 Stage 0) 电子声带听感对照台</title>
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
<h1>PyBot-Larynx (exp_20 Stage 0) · 电子声带盲听对照台</h1>
<p class="sub">100% 电子声带合成（零神经权重、零波形拼接）对比 Real 原声与 v15_auto。所有音频已统一至 -20 dBFS。</p>
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
                    "errors": 0,
                }
            a = agg[name]
            if m.get("error"):
                a["errors"] += 1
            if m.get("frames_touched", 0) > 0:
                a["utts_fired"] += 1
                a["frames_touched_total"] += int(m["frames_touched"])
                a["voiced_overrides_total"] += int(m["voiced_overrides"])
                a["env_deltas_db"].append(float(m["mean_abs_env_delta_db"]))
                a["bap_deltas"].append(float(m["mean_abs_bap_delta"]))
                a["tilt_deltas"].append(float(m["mean_abs_tilt_delta"]))

    out: dict[str, Any] = {}
    for name, a in agg.items():
        out[name] = {
            "utts_fired": a["utts_fired"],
            "frames_touched_total": a["frames_touched_total"],
            "voiced_overrides_total": a["voiced_overrides_total"],
            "mean_abs_env_delta_db": round(float(np.mean(a["env_deltas_db"])), 4) if a["env_deltas_db"] else 0.0,
            "mean_abs_bap_delta": round(float(np.mean(a["bap_deltas"])), 4) if a["bap_deltas"] else 0.0,
            "mean_abs_tilt_delta": round(float(np.mean(a["tilt_deltas"])), 4) if a["tilt_deltas"] else 0.0,
            "errors": a["errors"],
        }
    return out


def _aggregate_kb_hits(kb_infos: list[dict[str, Any]]) -> dict[str, Any]:
    tot: dict[str, int] = {}
    for info in kb_infos:
        for k, v in info.get("hits", {}).items():
            tot[k] = tot.get(k, 0) + int(v)
    fin_tot = max(1, sum(tot.get(k, 0) for k in ("exact_oft", "place_ft", "ft", "f", "fallback")))
    return {
        "counts": tot,
        "final_exact_oft_rate": round(tot.get("exact_oft", 0) / fin_tot, 4),
        "final_place_or_better_rate": round((tot.get("exact_oft", 0) + tot.get("place_ft", 0)) / fin_tot, 4),
        "final_fallback_rate": round(tot.get("fallback", 0) / fin_tot, 4),
    }


def run_stage0(rebuild_knowledge: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    if rebuild_knowledge or not KNOWLEDGE_PATH.exists():
        kb_summary = build_stage0_knowledge(n_train=TRAIN_STAGE0_N)
    else:
        kb_summary = json.loads((EXP_DIR / "acoustics" / "knowledge_summary.json").read_text(encoding="utf-8"))

    knowledge = AcousticKnowledge.load(KNOWLEDGE_PATH)
    idx_doc = load_v15_index_cached()
    assets = load_speaker_assets_v15()
    coh_phasor = assets["glottal_bank"].coherent_phasor
    pm_v15 = ProsodyModelV15.load()
    arbiter = Arbiter()
    rules = default_stage0_rules()

    _, dev_60_ids, held_20_ids = get_splits()

    print(f"Preparing HELDOUT_20 ({len(held_20_ids)} utts)...", flush=True)
    held_oracles = [_prepare_utterance_oracle(u, idx_doc["held_utt_metas"][u]) for u in held_20_ids]

    cond_names = (
        "C0_copy_lf",
        "C0_copy_coh",
        "C1_static_stft",
        "C2_dynamic_of",
        "C3_pybot_rules_lf",
        "C3_pybot_rules_coh",
        "C4_auto_e2e",
    )
    held_rows: dict[str, list[dict[str, Any]]] = {c: [] for c in cond_names}
    c3_traces: list[dict[str, Any]] = []
    c3_kb_infos: list[dict[str, Any]] = []
    c4_kernel_rtfs: list[float] = []
    listen_wavs: dict[str, dict[str, np.ndarray]] = {}

    for u_idx, item in enumerate(held_oracles):
        uid = item["uid"]
        ref_span = item["ref_span"]
        ref_fast = item["ref_fast"]
        feats_p = item["feats_p"]
        bap6 = item["bap6"]
        ctx = item["ctx"]
        seed_u = 2000 + u_idx

        # C0_copy_lf & C0_copy_coh
        ctrl_c0 = LarynxControl(
            f0_hz=feats_p.f0_hz,
            voiced=feats_p.voiced,
            log_env=feats_p.log_env,
            bap=bap6,
            n_samples=len(ref_span),
        )
        for cname, gmode in (("C0_copy_lf", "parametric_lf"), ("C0_copy_coh", "speaker_coherent")):
            ts = time.perf_counter()
            y = synthesize_larynx(ctrl_c0, glottal_mode=gmode, coherent_phasor=coh_phasor, seed=seed_u)
            dt = time.perf_counter() - ts
            rtf = dt / max(1e-6, len(y) / SR)
            held_rows[cname].append(evaluate_utterance(y, ref_span, rtf=rtf, ref_fast=ref_fast))
            if u_idx < 10 and cname == "C0_copy_lf":
                listen_wavs.setdefault(uid, {})["C0_copy_lf"] = y

        # C1_static_stft
        ts = time.perf_counter()
        ctrl_c1, _ = build_base_control(ctx, feats_p.f0_hz, feats_p.voiced, knowledge, mode="static")
        ctrl_c1.n_samples = len(ref_span)
        y_c1 = synthesize_larynx(ctrl_c1, glottal_mode="parametric_lf", seed=seed_u)
        dt = time.perf_counter() - ts
        held_rows["C1_static_stft"].append(
            evaluate_utterance(y_c1, ref_span, rtf=dt / max(1e-6, len(y_c1) / SR), ref_fast=ref_fast)
        )

        # C2_dynamic_of
        ts = time.perf_counter()
        ctrl_c2, kb_c2 = build_base_control(ctx, feats_p.f0_hz, feats_p.voiced, knowledge, mode="dynamic")
        ctrl_c2.n_samples = len(ref_span)
        y_c2 = synthesize_larynx(ctrl_c2, glottal_mode="parametric_lf", seed=seed_u)
        dt = time.perf_counter() - ts
        held_rows["C2_dynamic_of"].append(
            evaluate_utterance(y_c2, ref_span, rtf=dt / max(1e-6, len(y_c2) / SR), ref_fast=ref_fast)
        )

        # C3_pybot_rules_lf & C3_pybot_rules_coh
        for cname, gmode in (("C3_pybot_rules_lf", "parametric_lf"), ("C3_pybot_rules_coh", "speaker_coherent")):
            ts = time.perf_counter()
            ctrl_b, kb_b = build_base_control(ctx, feats_p.f0_hz, feats_p.voiced, knowledge, mode="dynamic")
            ctrl_b.n_samples = len(ref_span)
            ctrl_fused, trace = arbiter.apply(ctx, ctrl_b, rules)
            y_c3 = synthesize_larynx(ctrl_fused, glottal_mode=gmode, coherent_phasor=coh_phasor, seed=seed_u)
            dt = time.perf_counter() - ts
            rtf = dt / max(1e-6, len(y_c3) / SR)
            held_rows[cname].append(evaluate_utterance(y_c3, ref_span, rtf=rtf, ref_fast=ref_fast))
            if cname == "C3_pybot_rules_lf":
                c3_traces.append(trace)
                c3_kb_infos.append(kb_b)
                if u_idx < 10:
                    listen_wavs[uid]["C3_pybot_rules_lf"] = y_c3

        # C4_auto_e2e
        ts = time.perf_counter()
        y_c4, c4_info = render_c4_auto_utterance(
            item["text"], uid, knowledge, pm_v15, arbiter=arbiter, rules=rules, seed=seed_u
        )
        dt = time.perf_counter() - ts
        dur_c4 = max(1e-6, len(y_c4) / SR)
        c4_kernel_rtfs.append(c4_info["kernel_s"] / dur_c4)
        held_rows["C4_auto_e2e"].append(
            evaluate_utterance(y_c4, ref_span, rtf=dt / dur_c4, ref_fast=ref_fast)
        )
        if u_idx < 10:
            listen_wavs[uid]["Real"] = ref_span
            listen_wavs[uid]["C4_auto_e2e"] = y_c4

    held_summary = {c: summarize_condition(held_rows[c]) for c in cond_names}

    # Evaluate locked DEV_60 swarm baseline
    print(f"Running DEV_60 swarm baseline ({len(dev_60_ids)} utts)...", flush=True)
    corpus_records, _ = parse_bznsyp_corpus()
    corpus_by_id = {r["id"]: r for r in corpus_records}
    dev_conds = ("C2_dynamic_of", "C3_pybot_rules_lf", "C4_auto_e2e")
    dev_rows: dict[str, list[dict[str, Any]]] = {c: [] for c in dev_conds}
    for d_idx, uid in enumerate(dev_60_ids):
        dev_meta = dict(idx_doc["train_utt_metas"][uid])
        dev_meta["syllables"] = corpus_by_id[uid]["syllables"]
        item = _prepare_utterance_oracle(uid, dev_meta)
        ref_span = item["ref_span"]
        ref_fast = item["ref_fast"]
        feats_p = item["feats_p"]
        ctx = item["ctx"]
        seed_d = 5000 + d_idx

        ts = time.perf_counter()
        c2, _ = build_base_control(ctx, feats_p.f0_hz, feats_p.voiced, knowledge, mode="dynamic")
        c2.n_samples = len(ref_span)
        y2 = synthesize_larynx(c2, glottal_mode="parametric_lf", seed=seed_d)
        dt = time.perf_counter() - ts
        dev_rows["C2_dynamic_of"].append(evaluate_utterance(y2, ref_span, rtf=dt / max(1e-6, len(y2) / SR), ref_fast=ref_fast))

        ts = time.perf_counter()
        c3b, _ = build_base_control(ctx, feats_p.f0_hz, feats_p.voiced, knowledge, mode="dynamic")
        c3b.n_samples = len(ref_span)
        c3f, _ = arbiter.apply(ctx, c3b, rules)
        y3 = synthesize_larynx(c3f, glottal_mode="parametric_lf", seed=seed_d)
        dt = time.perf_counter() - ts
        dev_rows["C3_pybot_rules_lf"].append(evaluate_utterance(y3, ref_span, rtf=dt / max(1e-6, len(y3) / SR), ref_fast=ref_fast))

        ts = time.perf_counter()
        y4, _ = render_c4_auto_utterance(item["text"], uid, knowledge, pm_v15, arbiter=arbiter, rules=rules, seed=seed_d)
        dt = time.perf_counter() - ts
        dev_rows["C4_auto_e2e"].append(evaluate_utterance(y4, ref_span, rtf=dt / max(1e-6, len(y4) / SR), ref_fast=ref_fast))

    dev_summary = {c: summarize_condition(dev_rows[c]) for c in dev_conds}

    # Export listening pack for HELDOUT_20[:10]
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    listen_groups: list[dict[str, Any]] = []
    v15_samples_dir = REPO_ROOT / "exp_v15_bznsyp_tts" / "samples"
    for idx_l, item in enumerate(held_oracles[:10]):
        uid = item["uid"]
        v15_lbl = V15_10_MAP[uid]
        v15_src = v15_samples_dir / f"bz10_{uid}_{v15_lbl}.wav"
        sys_entries = [
            ("Real", "Real 真人原声", listen_wavs[uid]["Real"]),
            ("C0_copy_lf", "C0 电子声带原包络重合成 (LF)", listen_wavs[uid]["C0_copy_lf"]),
            ("C3_pybot_rules_lf", "C3 电子声带 (Oracle节奏F0 + PyBot 4规则)", listen_wavs[uid]["C3_pybot_rules_lf"]),
            ("C4_auto_e2e", "C4 电子声带全自动端到端 (文本->语音)", listen_wavs[uid]["C4_auto_e2e"]),
        ]
        # Include v15_auto wav copied as normalized PCM16
        from exp_v5_eval.metrics import read_wav
        v15_w, _ = read_wav(v15_src)
        sys_entries.append(("v15_auto", "v15 Auto (9,726句拼接基线)", v15_w))

        random.Random(20260 + idx_l).shuffle(sys_entries)
        its = []
        for lbl, (scode, sdesc, w_s) in zip(("A", "B", "C", "D", "E"), sys_entries):
            fname = f"s0_{uid}_{lbl}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fname, _rms_norm(w_s), sr=SR)
            its.append({"label": lbl, "file": f"samples/{fname}", "system_code": scode, "system_desc": sdesc})
        listen_groups.append({"id": f"H{idx_l+1:02d}", "uid": uid, "title": f"H{idx_l+1:02d} ({uid})", "text": item["text"], "items": its})

    listen_path = EXP_DIR / "listen.html"
    _render_listen_html(listen_groups, listen_path)

    # Gate checks
    c0_m = held_summary["C0_copy_lf"]["medians"]
    c3_m = held_summary["C3_pybot_rules_lf"]["medians"]
    c0_pass = (
        held_summary["C0_copy_lf"]["sanity_ok"] == 20
        and c0_m["mel_lsd_db"] <= 3.0
        and c0_m["rtf"] <= 0.05
    )
    c3_pass = (
        held_summary["C3_pybot_rules_lf"]["sanity_ok"] == 20
        and c3_m["mel_lsd_db"] <= 5.8
        and c3_m["rtf"] <= 0.08
    )
    verdict = "GO" if (c0_pass and c3_pass) else ("CONDITIONAL_GO" if (c0_pass and c3_m["mel_lsd_db"] <= 6.8) else "NO_GO")

    results = {
        "runtime_s": round(time.perf_counter() - t0, 2),
        "verdict": verdict,
        "gates": {"c0_pass": c0_pass, "c3_pass": c3_pass},
        "knowledge_summary": kb_summary,
        "c3_kb_hits": _aggregate_kb_hits(c3_kb_infos),
        "c3_arbiter_attribution": _aggregate_arbiter(c3_traces),
        "c4_kernel_rtf_median": round(float(np.median(c4_kernel_rtfs)), 4),
        "heldout_20_ids": held_20_ids,
        "heldout_summary": held_summary,
        "dev_60_ids": dev_60_ids,
        "dev_60_summary": dev_summary,
    }
    (EXP_DIR / "stage0_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_stage0_report(results, EXP_DIR / "STAGE0_REPORT.md")
    print(json.dumps({
        "verdict": verdict,
        "runtime_s": results["runtime_s"],
        "heldout_C0_mel": c0_m["mel_lsd_db"],
        "heldout_C1_mel": held_summary["C1_static_stft"]["medians"]["mel_lsd_db"],
        "heldout_C2_mel": held_summary["C2_dynamic_of"]["medians"]["mel_lsd_db"],
        "heldout_C3_lf_mel": c3_m["mel_lsd_db"],
        "heldout_C3_coh_mel": held_summary["C3_pybot_rules_coh"]["medians"]["mel_lsd_db"],
        "heldout_C4_mel": held_summary["C4_auto_e2e"]["medians"]["mel_lsd_db"],
        "heldout_C3_rtf": c3_m["rtf"],
        "heldout_C4_rtf": held_summary["C4_auto_e2e"]["medians"]["rtf"],
    }, ensure_ascii=False, indent=2), flush=True)
    return results


def _write_stage0_report(res: dict[str, Any], out_path: Path) -> None:
    hs = res["heldout_summary"]
    ds = res["dev_60_summary"]
    kb = res["knowledge_summary"]
    hits = res["c3_kb_hits"]
    attr = res["c3_arbiter_attribution"]

    lines = [
        "# PyBot-Larynx (`exp_20_pybot_larynx`) Stage 0 可行性探针报告",
        "",
        f"- **判定结论 (`verdict`)**: **`{res['verdict']}`** (`c0_pass={res['gates']['c0_pass']}`, `c3_pass={res['gates']['c3_pass']}`)",
        f"- **零神经权重 / 零波形拼接**: `0.0 MB` neural weights，`100%` 电子声带实时合成",
        f"- **知识库 (`knowledge_stage0.npz`)**: 仅用 `{kb['n_train_utts']}` 条 `TRAIN` 句构建，体积 **`{kb['npz_size_kb']} KB`**，`HELDOUT_20` 与 `DEV_60` 严格零泄漏隔离",
        "",
        "## 1. `HELDOUT_20` 核心定量结果（中位数）",
        "",
        "| 条件 | Mel-LSD (dB) | MCD (dB) | F0 MAE (st) | HNR (dB) | 清音 2-8k/lo (dB) | clicks >8 | CPU RTF | Sanity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| *参照：`19b aPN3` (LPC残差全参数)* | *9.2432* | *34.2689* | — | — | — | — | — | — |",
        "| *参照：`v5_oracle_prosody` (Klatt)* | *8.26* | *33.10* | — | — | — | — | — | — |",
        "| *参照：`v15_auto` (9,726句拼接)* | *6.8795* | *21.5875* | *3.7870* | *16.32 (10句)* | *8.838* | *0.0006* | *0.4761* | *20/20* |",
        "| *参照：`v15_guided` (参考节奏F0拼接)* | *5.7540* | *20.9761* | *1.3738* | — | — | — | — | *20/20* |",
    ]
    for cname in (
        "C0_copy_lf",
        "C0_copy_coh",
        "C1_static_stft",
        "C2_dynamic_of",
        "C3_pybot_rules_lf",
        "C3_pybot_rules_coh",
        "C4_auto_e2e",
    ):
        m = hs[cname]["medians"]
        uv_s = f"{m.get('uv_2_8k_vs_lo_db'):.3f}" if m.get("uv_2_8k_vs_lo_db") is not None else "—"
        hnr_s = f"{m.get('hnr_voiced_db'):.2f}" if m.get("hnr_voiced_db") is not None else "—"
        lines.append(
            f"| `{cname}` | **{m['mel_lsd_db']:.4f}** | {m['mcd_db']:.4f} | {m['f0_mae_st']:.4f} | "
            f"{hnr_s} | {uv_s} | {m['clicks_gt8']:.6f} | **{m['rtf']:.4f}** | {hs[cname]['sanity_ok']}/{hs[cname]['n']} |"
        )

    lines += [
        "",
        f"> 注：`C4_auto_e2e` 端到端中位 RTF = `{hs['C4_auto_e2e']['medians']['rtf']:.4f}`（其中电子声带核 `synthesize_larynx` 仅占 `{res['c4_kernel_rtf_median']:.4f}`）。",
        "",
        "## 2. `DEV_60` 蜂群演进锁定基线（中位数）",
        "",
        "| 条件 | Mel-LSD (dB) | MCD (dB) | F0 MAE (st) | HNR (dB) | 清音 2-8k/lo (dB) | CPU RTF | Sanity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cname in ("C2_dynamic_of", "C3_pybot_rules_lf", "C4_auto_e2e"):
        m = ds[cname]["medians"]
        uv_s = f"{m.get('uv_2_8k_vs_lo_db'):.3f}" if m.get("uv_2_8k_vs_lo_db") is not None else "—"
        hnr_s = f"{m.get('hnr_voiced_db'):.2f}" if m.get("hnr_voiced_db") is not None else "—"
        lines.append(
            f"| `{cname}` | {m['mel_lsd_db']:.4f} | {m['mcd_db']:.4f} | {m['f0_mae_st']:.4f} | "
            f"{hnr_s} | {uv_s} | {m['rtf']:.4f} | {ds[cname]['sanity_ok']}/{ds[cname]['n']} |"
        )

    lines += [
        "",
        "## 3. 知识库回退率与 Arbiter 模块归因（`C3_pybot_rules_lf` on `HELDOUT_20`）",
        "",
        f"- 韵母精确 `(onset, final, tone)` 命中率：**`{hits['final_exact_oft_rate']*100:.1f}%`**，`place_ft` 及以上命中率：**`{hits['final_place_or_better_rate']*100:.1f}%`**，兜底率：**`{hits['final_fallback_rate']*100:.1f}%`**。",
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
        "- 离线盲听对比台：[listen.html](listen.html)（含 `HELDOUT_20` 前 10 句：`Real` / `C0_copy_lf` / `C3_pybot_rules_lf` / `C4_auto_e2e` / `v15_auto`，全部统一 `-20 dBFS`）",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-knowledge", action="store_true")
    args = ap.parse_args()
    run_stage0(rebuild_knowledge=args.rebuild_knowledge)
