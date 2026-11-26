"""Zero-leakage multi-speaker & cross-lingual verification benchmark for Stage 6 Voice Cloning."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import EXP_DIR, SAMPLES_DIR, SR  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import AcousticKnowledgeStage5  # noqa: E402
from exp_20_pybot_larynx.arbiter import Arbiter  # noqa: E402
from exp_20_pybot_larynx.clone import (  # noqa: E402
    VoiceCloneProfile,
    apply_clone_to_control,
    extract_clone_profile_from_wavs,
    load_base_speaker_profile,
)
from exp_20_pybot_larynx.larynx.synth import synthesize_larynx  # noqa: E402
from exp_20_pybot_larynx.rules import default_stage5_rules  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity, compute_f0_hnr_metrics, compute_ltas_metrics  # noqa: E402
from exp_v6_vocoder import eval_vocoder as v6_eval  # noqa: E402
from exp_v8_timbre_clone.eval_english_crossgender import _classify_4way  # noqa: E402
from exp_v8_timbre_clone.eval_multilingual_corpus import _load_zh_5speakers, _read_utt  # noqa: E402
from exp_v9_eval.v9_metrics import compute_v9_extended_metrics  # noqa: E402
from exp_v14_fluent_zh_tts.eval_v14 import _rms_norm, _write_pcm16_wav  # noqa: E402
from exp_v15_bznsyp_tts.index_bznsyp_v15 import load_speaker_assets_v15  # noqa: E402
from pybot_webui.crossling import build_utterance_control  # noqa: E402
from pybot_webui.presets import VoiceParams  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "legacy" / "exp_v8_data" / "corpus" / "manifest.json"
CLONE_RESULTS_PATH = EXP_DIR / "clone" / "clone_results.json"
CLONE_REPORT_PATH = EXP_DIR / "clone" / "CLONE_REPORT.md"
LISTEN_CLONE_PATH = EXP_DIR / "listen_clone.html"

TEST_ZH_TEXTS: tuple[str, ...] = (
    "今天天气真好，微风吹过树梢，适合出门散步。",
    "二零二六年，我们用纯电子声带实现了零权重音色模仿。",
    "无论高音女声还是低沉男声，都可以灵活调控与平滑切换。",
    "科技不只是冷冰冰的算法，也能带来自然温暖的声音表达。",
)

CROSS_LINGUAL_TEXTS: tuple[tuple[str, str], ...] = (
    ("zh", "今天用全新电子声带模仿这段参考说话人的音色。"),
    ("en", "Hello world, this is a zero neural voice clone test."),
    ("zh+en", "今天用 PyBot 测试 cross-lingual voice cloning，效果很自然。"),
)


def _synth_with_clone(
    text: str,
    utt_id: str,
    kb5: AcousticKnowledgeStage5,
    arbiter: Arbiter,
    rules5: list[Any],
    base_prof: VoiceCloneProfile,
    tgt_prof: VoiceCloneProfile,
    coh_phasor: np.ndarray,
    *,
    strength: float,
    seed: int = 123,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    vp = VoiceParams(glottal_mode="speaker_coherent", crossling_mode="auto", seed=seed)
    t0 = time.perf_counter()
    base_ctrl, _ = build_utterance_control(text, utt_id, kb5, arbiter, rules5, vp)
    cloned_ctrl, eff_ph, diag = apply_clone_to_control(
        base_ctrl,
        base_prof,
        tgt_prof,
        strength=strength,
        clone_f0=True,
        clone_glottal=True,
        base_coherent_phasor=coh_phasor,
    )
    wav = synthesize_larynx(cloned_ctrl, glottal_mode="speaker_coherent", coherent_phasor=eff_ph, seed=seed)
    dt = time.perf_counter() - t0
    rtf = dt / max(1e-6, len(wav) / float(SR))
    wav_n = _rms_norm(wav, target_dbfs=-20.0)
    return wav_n, rtf, diag


def _render_listen_clone_html(groups: list[dict[str, Any]], out_path: Path) -> None:
    payload = json.dumps(groups, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBot-Larynx Stage 6 零样本声音模仿盲听对照台</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#eef1f4;background:#11151c;max-width:1040px;margin:1.8rem auto;padding:0 1rem}}
.card{{background:#19202b;border:1px solid #293345;padding:1rem 1.2rem;margin:1rem 0;border-radius:10px}}
.row{{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;padding:.45rem 0;border-bottom:1px dashed #273142}}
.row:last-child{{border-bottom:none}}
audio{{width:min(100%,25rem);height:34px}}
button.btn{{background:#2563eb;color:#fff;border:none;border-radius:6px;padding:.5rem .95rem;cursor:pointer}}
.ans{{display:none;margin-top:.55rem;padding:.55rem .8rem;background:#122033;border-left:3px solid #60a5fa;border-radius:4px;font-size:.92rem;color:#dbeafe}}
.ans.show{{display:block}}
</style>
</head>
<body>
<h1>PyBot-Larynx (Stage 6) · 0 MB 神经权重样本模仿试听台</h1>
<p>对比目标说话人真实参考音频（Ref）、未克隆 Stage 5 基底女声、1-shot 克隆与 3-shot 克隆（含跨语种中英混读），全部统一 -20 dBFS。</p>
<div id="root"></div>
<div class="card"><button class="btn" onclick="toggleAns()">显示 / 隐藏答案映射</button></div>
<script>
const GROUPS = {payload};
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
      ansLines.push(it.label + ': ' + it.desc);
    }});
    const ans = document.createElement('div'); ans.className = 'ans'; ans.textContent = ansLines.join(' | ');
    card.append(ans); root.append(card);
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


def run_clone_verification() -> dict[str, Any]:
    t0 = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    zh_5spk = _load_zh_5speakers(manifest)
    spks_zh = sorted(zh_5spk.keys())  # ['SSB0005', 'SSB0073', 'SSB0139', 'SSB0502', 'SSB1593']

    en_by_spk: dict[str, list[dict[str, Any]]] = {}
    for u in manifest["english"]["utterances"]:
        if u["speaker"] in ("bdl", "slt"):
            en_by_spk.setdefault(u["speaker"], []).append({
                "id": u["id"],
                "speaker": u["speaker"],
                "gender": u["gender"],
                "path": str(REPO_ROOT / u["path"]),
            })
    for s in en_by_spk:
        en_by_spk[s].sort(key=lambda r: r["id"])

    kb5 = AcousticKnowledgeStage5.load()
    assets = load_speaker_assets_v15()
    coh_phasor = np.asarray(assets["glottal_bank"].coherent_phasor, dtype=np.complex128)
    arbiter = Arbiter()
    rules5 = default_stage5_rules()
    base_prof = load_base_speaker_profile()

    # Warmup synthesis once
    _synth_with_clone("测试克隆预热。", "warm", kb5, arbiter, rules5, base_prof, base_prof, coh_phasor, strength=0.0, seed=1)

    # 1. Extract strictly prompt-only 1-shot & 3-shot profiles, and held-out centroids ([3:15])
    profiles_1s: dict[str, VoiceCloneProfile] = {}
    profiles_3s: dict[str, VoiceCloneProfile] = {}
    clf_train_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    target_refs: dict[str, dict[str, float]] = {}

    for spk in spks_zh:
        rows = zh_5spk[spk]
        p1_paths = [r["path"] for r in rows[:1]]
        p3_paths = [r["path"] for r in rows[:3]]
        profiles_1s[spk] = extract_clone_profile_from_wavs(p1_paths, name=f"{spk}_1s", profile_id=f"{spk}_1s")
        profiles_3s[spk] = extract_clone_profile_from_wavs(p3_paths, name=f"{spk}_3s", profile_id=f"{spk}_3s")

        f0s, tilts, hnrs = [], [], []
        for r in rows[3:]:
            x = _read_utt(r)
            clf_train_rows.append((spk, v6_eval._extract_clf_vec(x, sr=SR)))
            fh = compute_f0_hnr_metrics(x, sr=SR)
            lt = compute_ltas_metrics(x, sr=SR)
            if fh and fh.get("f0_median_hz") is not None:
                f0s.append(float(fh["f0_median_hz"]))
                if fh.get("hnr_db_median") is not None:
                    hnrs.append(float(fh["hnr_db_median"]))
            if lt and lt.get("tilt_db_oct") is not None:
                tilts.append(float(lt["tilt_db_oct"]))
        target_refs[spk] = {
            "gender": rows[0]["gender"],
            "f0_hz": float(np.median(f0s)),
            "tilt_db_oct": float(np.median(tilts)),
            "hnr_db": float(np.median(hnrs)),
        }

    # 2. Evaluate 5-way Mandarin cloning: unconverted vs 1-shot vs 3-shot (4 unseen sentences x 5 speakers = 20 samples/cond)
    cond_rows: dict[str, list[dict[str, Any]]] = {"unconverted": [], "clone_1shot": [], "clone_3shot": []}
    captured_samples: dict[str, dict[str, np.ndarray]] = {}

    for spk in spks_zh:
        ref = target_refs[spk]
        # Save one held-out real reference clip for listening comparison
        captured_samples[spk] = {"Real_Ref": _rms_norm(_read_utt(zh_5spk[spk][3]), target_dbfs=-20.0)}
        for mname, prof, str_v in (
            ("unconverted", base_prof, 0.0),
            ("clone_1shot", profiles_1s[spk], 0.90),
            ("clone_3shot", profiles_3s[spk], 0.90),
        ):
            for t_idx, txt in enumerate(TEST_ZH_TEXTS):
                y, rtf, diag = _synth_with_clone(
                    txt, f"zh5_{mname}_{spk}_{t_idx}", kb5, arbiter, rules5, base_prof, prof, coh_phasor,
                    strength=str_v, seed=4000 + t_idx,
                )
                san = check_wav_sanity(y, SR, expected_sr=SR)
                qv = v6_eval._extract_clf_vec(y, sr=SR)
                fh = compute_f0_hnr_metrics(y, sr=SR)
                lt = compute_ltas_metrics(y, sr=SR)
                v9 = compute_v9_extended_metrics(y, sr=SR, fmin=75.0, fmax=450.0)
                pred_4d = _classify_4way(qv, clf_train_rows, "timbre_4d")
                pred_ltas = _classify_4way(qv, clf_train_rows, "ltas_vec")
                pred_all = _classify_4way(qv, clf_train_rows, "all_features")

                f0_val = float(fh["f0_median_hz"]) if (fh and fh.get("f0_median_hz") is not None) else 200.0
                tilt_val = float(lt["tilt_db_oct"]) if lt else -3.6
                hnr_val = float(fh["hnr_db_median"]) if (fh and fh.get("hnr_db_median") is not None) else 12.0

                cond_rows[mname].append({
                    "speaker": spk,
                    "gender": ref["gender"],
                    "sanity_ok": bool(san["ok"]),
                    "hit_4d": pred_4d == spk,
                    "hit_ltas": pred_ltas == spk,
                    "hit_all": pred_all == spk,
                    "f0_err_st": abs(12.0 * math.log2(max(50.0, f0_val) / max(50.0, ref["f0_hz"]))),
                    "tilt_err": abs(tilt_val - ref["tilt_db_oct"]),
                    "hnr_err": abs(hnr_val - ref["hnr_db"]),
                    "clicks_gt8": float(v9.get("click_outlier_rate_gt8", 0.0)),
                    "rtf": rtf,
                    "alpha_eff": diag.get("alpha_eff", 1.0),
                })
                if t_idx == 0:
                    captured_samples[spk][mname] = y

    def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = max(1, len(rows))
        fm_rows = [r for r in rows if r["gender"] == "male"]
        return {
            "n": len(rows),
            "sanity_ok": sum(1 for r in rows if r["sanity_ok"]),
            "acc_timbre4d": round(sum(1 for r in rows if r["hit_4d"]) / n, 4),
            "acc_ltas": round(sum(1 for r in rows if r["hit_ltas"]) / n, 4),
            "acc_all": round(sum(1 for r in rows if r["hit_all"]) / n, 4),
            "f0_err_st_med": round(float(np.median([r["f0_err_st"] for r in rows])), 3),
            "tilt_err_db_oct_med": round(float(np.median([r["tilt_err"] for r in rows])), 3),
            "fm_tilt_err_med": round(float(np.median([r["tilt_err"] for r in fm_rows])), 3) if fm_rows else 0.0,
            "hnr_err_db_med": round(float(np.median([r["hnr_err"] for r in rows])), 3),
            "clicks_gt8_med": round(float(np.median([r["clicks_gt8"] for r in rows])), 6),
            "rtf_med": round(float(np.median([r["rtf"] for r in rows])), 4),
        }

    summary = {k: _agg(v) for k, v in cond_rows.items()}

    # 3. Cross-lingual cloning check (EN bdl/slt -> ZH & ZH+EN; ZH SSB0073 -> EN & ZH+EN; self-clone)
    crossling_rows: list[dict[str, Any]] = []
    cross_targets = {
        "bdl_en_m": extract_clone_profile_from_wavs([r["path"] for r in en_by_spk["bdl"][:3]], name="bdl_M", profile_id="bdl_3s"),
        "slt_en_f": extract_clone_profile_from_wavs([r["path"] for r in en_by_spk["slt"][:3]], name="slt_F", profile_id="slt_3s"),
        "ssb0073_zh_m": profiles_3s["SSB0073"],
        "base_self": base_prof,
    }
    for tname, tprof in cross_targets.items():
        for lang_tag, txt in CROSS_LINGUAL_TEXTS:
            y, rtf, _ = _synth_with_clone(
                txt, f"cl_{tname}_{lang_tag}", kb5, arbiter, rules5, base_prof, tprof, coh_phasor, strength=0.90, seed=555
            )
            san = check_wav_sanity(y, SR, expected_sr=SR)
            fh = compute_f0_hnr_metrics(y, sr=SR)
            f0_out = float(fh["f0_median_hz"]) if (fh and fh.get("f0_median_hz") is not None) else 0.0
            crossling_rows.append({
                "target": tname,
                "lang": lang_tag,
                "sanity_ok": bool(san["ok"]),
                "tgt_f0_hz": round(tprof.f0_median_hz, 1),
                "out_f0_hz": round(f0_out, 1),
                "f0_err_st": round(abs(12.0 * math.log2(max(50.0, f0_out) / max(50.0, tprof.f0_median_hz))), 3),
                "rtf": round(rtf, 4),
            })
            if tname in ("bdl_en_m", "slt_en_f") and lang_tag == "zh+en":
                captured_samples.setdefault(tname, {})["clone_zh_en"] = y
                ref_en = _read_utt(en_by_spk["bdl" if tname.startswith("bdl") else "slt"][3])
                captured_samples[tname]["Real_EN_Ref"] = _rms_norm(ref_en, target_dbfs=-20.0)

    # Export blind listening samples
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    listen_groups: list[dict[str, Any]] = []
    for spk in spks_zh:
        wmap = captured_samples[spk]
        entries = [
            ("Real_Ref", f"{spk} ({target_refs[spk]['gender']}) 真人参考音频", wmap["Real_Ref"]),
            ("unconverted", "未克隆 Stage 5 基底音色", wmap["unconverted"]),
            ("clone_1shot", f"1-shot 样本模仿 ({spk})", wmap["clone_1shot"]),
            ("clone_3shot", f"3-shot 样本模仿 ({spk})", wmap["clone_3shot"]),
        ]
        random.Random(202660 + hash(spk) % 1000).shuffle(entries)
        its = []
        for lbl, (code, desc, wav_s) in zip(("A", "B", "C", "D"), entries):
            fname = f"clone_{spk}_{lbl}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fname, wav_s, sr=SR)
            its.append({"label": lbl, "file": f"samples/{fname}", "desc": f"{code}: {desc}"})
        listen_groups.append({
            "title": f"普通话 5 选 1 模仿对照 · 目标 {spk} ({target_refs[spk]['gender']}, {target_refs[spk]['f0_hz']:.0f} Hz)",
            "text": TEST_ZH_TEXTS[0],
            "items": its,
        })

    for tname in ("bdl_en_m", "slt_en_f"):
        wmap = captured_samples[tname]
        its = []
        for lbl, key, desc in (
            ("Ref", "Real_EN_Ref", f"{tname} 英文真人参考"),
            ("Clone", "clone_zh_en", f"{tname} 3-shot 跨语种中英混读克隆"),
        ):
            fname = f"clone_{tname}_{lbl}.wav"
            _write_pcm16_wav(SAMPLES_DIR / fname, wmap[key], sr=SR)
            its.append({"label": lbl, "file": f"samples/{fname}", "desc": desc})
        listen_groups.append({
            "title": f"跨语种模仿对照 · {tname} ({cross_targets[tname].f0_median_hz:.0f} Hz)",
            "text": CROSS_LINGUAL_TEXTS[2][1],
            "items": its,
        })

    _render_listen_clone_html(listen_groups, LISTEN_CLONE_PATH)

    out_doc = {
        "runtime_s": round(time.perf_counter() - t0, 2),
        "zero_ml_weights": True,
        "base_profile_bytes": base_prof.serialized_size_bytes(),
        "mandarin_5way_summary": summary,
        "target_speakers": target_refs,
        "crossling_checks": crossling_rows,
        "gates": {
            "all_sanity_ok": (
                summary["clone_1shot"]["sanity_ok"] == 20
                and summary["clone_3shot"]["sanity_ok"] == 20
                and all(r["sanity_ok"] for r in crossling_rows)
            ),
            "acc_all_ge_75": summary["clone_3shot"]["acc_all"] >= 0.75,
            "f0_err_le_1_2": summary["clone_3shot"]["f0_err_st_med"] <= 1.2,
            "tilt_err_le_1_2": summary["clone_3shot"]["tilt_err_db_oct_med"] <= 1.2,
            "fm_tilt_err_le_1_2": summary["clone_3shot"]["fm_tilt_err_med"] <= 1.2,
            "rtf_le_0_06": summary["clone_3shot"]["rtf_med"] <= 0.06,
        },
    }
    CLONE_RESULTS_PATH.write_text(json.dumps(out_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    s0, s1, s3 = summary["unconverted"], summary["clone_1shot"], summary["clone_3shot"]
    lines = [
        "# PyBot-Larynx Stage 6 零样本声音模仿（样本模仿）验证报告",
        "",
        f"- **零神经权重**: `0.0 MB` ML 权重，音色档案仅 JSON（基底档案 `{base_prof.serialized_size_bytes()} B`）",
        f"- **门禁汇总**: `{out_doc['gates']}`",
        "",
        "## 1. 5 说话人零样本普通话模仿（严格零泄漏：`0..2` 作 Prompt，`3+` 建离心集/参考，`4` 句未见文本，共 `20` 次/条件）",
        "",
        "| 方法 | timbre4d 准确率 | LTAS 准确率 | All 准确率 | F0 Err (st) | Tilt Err (dB/oct) | F->M Tilt Err | HNR Err (dB) | clicks >8 | CPU RTF | Sanity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| `unconverted (Stage 5 基底)` | {s0['acc_timbre4d']*100:.1f}% | {s0['acc_ltas']*100:.1f}% | {s0['acc_all']*100:.1f}% | {s0['f0_err_st_med']:.3f} | {s0['tilt_err_db_oct_med']:.3f} | {s0['fm_tilt_err_med']:.3f} | {s0['hnr_err_db_med']:.3f} | {s0['clicks_gt8_med']:.6f} | {s0['rtf_med']:.4f} | {s0['sanity_ok']}/{s0['n']} |",
        f"| `clone_1shot` | {s1['acc_timbre4d']*100:.1f}% | {s1['acc_ltas']*100:.1f}% | {s1['acc_all']*100:.1f}% | {s1['f0_err_st_med']:.3f} | {s1['tilt_err_db_oct_med']:.3f} | {s1['fm_tilt_err_med']:.3f} | {s1['hnr_err_db_med']:.3f} | {s1['clicks_gt8_med']:.6f} | {s1['rtf_med']:.4f} | {s1['sanity_ok']}/{s1['n']} |",
        f"| `clone_3shot` | **{s3['acc_timbre4d']*100:.1f}%** | **{s3['acc_ltas']*100:.1f}%** | **{s3['acc_all']*100:.1f}%** | **{s3['f0_err_st_med']:.3f}** | **{s3['tilt_err_db_oct_med']:.3f}** | **{s3['fm_tilt_err_med']:.3f}** | **{s3['hnr_err_db_med']:.3f}** | {s3['clicks_gt8_med']:.6f} | **{s3['rtf_med']:.4f}** | {s3['sanity_ok']}/{s3['n']} |",
        "",
        "## 2. 跨语种样本模仿与自克隆（`bdl`, `slt`, `SSB0073`, `base_self`）",
        "",
        "| 目标音色 | 测试语言 | 目标 F0 (Hz) | 输出 F0 (Hz) | F0 Err (st) | CPU RTF | Sanity |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in crossling_rows:
        lines.append(
            f"| `{r['target']}` | `{r['lang']}` | {r['tgt_f0_hz']:.1f} | {r['out_f0_hz']:.1f} | {r['f0_err_st']:.3f} | {r['rtf']:.4f} | {r['sanity_ok']} |"
        )
    lines += [
        "",
        "## 3. 试听页面",
        "- 离线盲听对比台：[listen_clone.html](../listen_clone.html)",
    ]
    CLONE_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out_doc, ensure_ascii=False, indent=2), flush=True)
    return out_doc


if __name__ == "__main__":
    run_clone_verification()
