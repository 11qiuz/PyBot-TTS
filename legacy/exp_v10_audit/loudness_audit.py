"""Forensic Loudness Jump & Root-Cause Audit for PyBotTTS v9.

Measures 40 ms windowed RMS (20 ms hop) |ΔdB| distributions across:
1. All 6 exp_v9_eval/listening_ab/*.wav files (segment-by-segment, excluding 0.40s gaps).
2. Real human speech baselines (extended/ ZH + CMU-ARCTIC EN).
3. Fresh v9_combined renders and exact code-culprit ablations:
   (a) Utterance-level -6.9 neper normalization + level-blind _edge_cmvn joins + fallback -3.1 offset.
   (b) TimbreClonerV3 v_mask per-frame RMS preservation vs ~v_mask uncompensated gain.
   (c) hnm_v9 unvoiced_residual_spans (>1.6 kHz SOS high-pass patch) gain/RMS jumps.
   (d) 0 ms hard seams (w_join = 0.0 & skipped _blend_seam) at obstruents/V-UV edges.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import wave
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v5_eval.metrics import extract_active_frames, read_wav  # noqa: E402
from exp_v6_crossling_dsp import arbiter as v6_arb  # noqa: E402
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS  # noqa: E402
from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v8_clarity.en_utau_bank import _trim_active_span  # noqa: E402
from exp_v9_clarity_pro.hnm_v9 import _hp_sos, analyze_v9, synthesize_v9  # noqa: E402
from exp_v9_clarity_pro.viterbi_utau import (  # noqa: E402
    V9DualBank,
    build_v9_dual_bank,
    filter_v9_bank_excluding_speakers,
    render_v9_utterance,
)
from exp_v9_eval.run_v9_benchmark import make_v3_cloner_adapter  # noqa: E402
from exp_v9_timbre_v3 import timbre_cloner_v3 as tc3  # noqa: E402
from scipy.signal import sosfiltfilt  # noqa: E402

SR = 24000
HOP = 120
AUDIT_DIR = pathlib.Path(__file__).resolve().parent
REPORT_PATH = AUDIT_DIR / "loudness_report.json"


def _normalize_active_rms(x: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    pk = float(np.max(np.abs(arr)))
    if pk <= 1e-12:
        return arr.copy()
    act = arr[np.abs(arr) > 0.01 * pk]
    rms = float(np.sqrt(np.mean(act * act))) if act.size else float(np.sqrt(np.mean(arr * arr)))
    if rms <= 1e-12:
        return arr.copy()
    return arr * ((10.0 ** (target_dbfs / 20.0)) / rms)


def measure_windowed_loudness(
    x: np.ndarray,
    sr: int = SR,
    win_ms: float = 40.0,
    hop_ms: float = 20.0,
    active_gate_dbfs: float = -48.0,
) -> dict[str, Any]:
    """Compute 40 ms RMS (20 ms hop) and adjacent active |ΔdB| statistics.

    Input is normalized to -20 dBFS active RMS before gating at `active_gate_dbfs`
    (28 dB below active speech mean), so inter-recording volume differences do not
    bias the gate while pauses and trailing silence are excluded.
    """
    arr = _normalize_active_rms(x, target_dbfs=-20.0)
    win = max(24, int(round(win_ms * 0.001 * sr)))
    hop = max(12, int(round(hop_ms * 0.001 * sr)))
    if len(arr) < win:
        return {
            "n_windows": 0,
            "n_active_pairs": 0,
            "active_dur_s": 0.0,
            "p50_abs_delta_db": 0.0,
            "p95_abs_delta_db": 0.0,
            "p99_abs_delta_db": 0.0,
            "max_abs_delta_db": 0.0,
            "n_gt6db": 0,
            "n_gt9db": 0,
            "jumps_gt6db_per_s": 0.0,
            "jumps_gt9db_per_s": 0.0,
            "rms_db_iqr": 0.0,
            "top_jumps": [],
        }

    n_w = 1 + (len(arr) - win) // hop
    idx = np.arange(win)[None, :] + np.arange(n_w)[:, None] * hop
    frames = arr[idx]
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))

    active = rms_db >= active_gate_dbfs
    pair_mask = active[:-1] & active[1:]
    deltas = np.abs(np.diff(rms_db))
    pair_deltas = deltas[pair_mask]
    active_dur_s = float(np.sum(active)) * (hop / float(sr))

    if pair_deltas.size == 0:
        return {
            "n_windows": int(n_w),
            "n_active_pairs": 0,
            "active_dur_s": round(active_dur_s, 4),
            "p50_abs_delta_db": 0.0,
            "p95_abs_delta_db": 0.0,
            "p99_abs_delta_db": 0.0,
            "max_abs_delta_db": 0.0,
            "n_gt6db": 0,
            "n_gt9db": 0,
            "jumps_gt6db_per_s": 0.0,
            "jumps_gt9db_per_s": 0.0,
            "rms_db_iqr": 0.0,
            "top_jumps": [],
        }

    pair_idx = np.where(pair_mask)[0]
    order = np.argsort(pair_deltas)[::-1][:5]
    top_jumps = [
        {
            "t_s": round(float((pair_idx[k] + 1) * hop / sr), 3),
            "prev_db": round(float(rms_db[pair_idx[k]]), 2),
            "cur_db": round(float(rms_db[pair_idx[k] + 1]), 2),
            "abs_delta_db": round(float(pair_deltas[k]), 2),
        }
        for k in order
    ]
    n6 = int(np.sum(pair_deltas > 6.0))
    n9 = int(np.sum(pair_deltas > 9.0))
    act_vals = rms_db[active]
    iqr = float(np.percentile(act_vals, 75) - np.percentile(act_vals, 25)) if act_vals.size >= 4 else 0.0
    denom_s = max(1e-6, active_dur_s)
    return {
        "n_windows": int(n_w),
        "n_active_pairs": int(pair_deltas.size),
        "active_dur_s": round(active_dur_s, 4),
        "p50_abs_delta_db": round(float(np.percentile(pair_deltas, 50)), 3),
        "p95_abs_delta_db": round(float(np.percentile(pair_deltas, 95)), 3),
        "p99_abs_delta_db": round(float(np.percentile(pair_deltas, 99)), 3),
        "max_abs_delta_db": round(float(np.max(pair_deltas)), 3),
        "n_gt6db": n6,
        "n_gt9db": n9,
        "jumps_gt6db_per_s": round(n6 / denom_s, 3),
        "jumps_gt9db_per_s": round(n9 / denom_s, 3),
        "rms_db_iqr": round(iqr, 3),
        "top_jumps": top_jumps,
    }


def _aggregate_loudness_rows(rows: list[dict[str, Any]], raw_deltas: list[np.ndarray]) -> dict[str, Any]:
    if not rows:
        return {"n_segments": 0}
    tot_dur = sum(r["active_dur_s"] for r in rows)
    tot_n6 = sum(r["n_gt6db"] for r in rows)
    tot_n9 = sum(r["n_gt9db"] for r in rows)
    cat = np.concatenate([d for d in raw_deltas if d.size > 0]) if any(d.size > 0 for d in raw_deltas) else np.zeros(0)
    return {
        "n_segments": len(rows),
        "total_active_dur_s": round(tot_dur, 3),
        "n_active_pairs": int(cat.size),
        "p50_abs_delta_db": round(float(np.percentile(cat, 50)) if cat.size else 0.0, 3),
        "p95_abs_delta_db": round(float(np.percentile(cat, 95)) if cat.size else 0.0, 3),
        "p99_abs_delta_db": round(float(np.percentile(cat, 99)) if cat.size else 0.0, 3),
        "max_abs_delta_db": round(float(np.max(cat)) if cat.size else 0.0, 3),
        "n_gt6db": tot_n6,
        "n_gt9db": tot_n9,
        "jumps_gt6db_per_s": round(tot_n6 / max(1e-6, tot_dur), 3),
        "jumps_gt9db_per_s": round(tot_n9 / max(1e-6, tot_dur), 3),
    }


def _extract_pair_deltas(x: np.ndarray, sr: int = SR, win_ms: float = 40.0, hop_ms: float = 20.0, gate: float = -48.0) -> np.ndarray:
    arr = _normalize_active_rms(x, target_dbfs=-20.0)
    win = max(24, int(round(win_ms * 0.001 * sr)))
    hop = max(12, int(round(hop_ms * 0.001 * sr)))
    if len(arr) < win:
        return np.zeros(0, dtype=np.float64)
    n_w = 1 + (len(arr) - win) // hop
    idx = np.arange(win)[None, :] + np.arange(n_w)[:, None] * hop
    rms = np.sqrt(np.mean(arr[idx] ** 2, axis=1))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    act = rms_db >= gate
    return np.abs(np.diff(rms_db))[act[:-1] & act[1:]]


def audit_listening_ab(report_path: pathlib.Path, listening_dir: pathlib.Path) -> dict[str, Any]:
    """Audit every non-gap segment inside exp_v9_eval/listening_ab/*.wav."""
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = rep["listening_ab_manifest"]
    per_file: dict[str, Any] = {}
    groups: dict[str, tuple[list[dict], list[np.ndarray]]] = {}

    for item in manifest:
        fname = item["file"]
        segs = item["segments"]
        wpath = listening_dir / fname
        x, sr = read_wav(wpath)
        seg_rows = []
        for seg in segs:
            s0 = max(0, int(round(float(seg["start_s"]) * sr)))
            s1 = min(len(x), int(round(float(seg["stop_s"]) * sr)))
            clip = x[s0:s1]
            m = measure_windowed_loudness(clip, sr=sr)
            deltas = _extract_pair_deltas(clip, sr=sr)
            lbl = str(seg["label"])
            entry = {"label": lbl, "start_s": seg["start_s"], "stop_s": seg["stop_s"], **m}
            seg_rows.append(entry)

            # Categorize by system/condition
            if lbl == "real" or lbl.startswith("real_"):
                gkey = "Real (listening_ab)"
            elif lbl.startswith("v6_"):
                gkey = "v6_crossling_dsp"
            elif lbl.startswith("v7_"):
                gkey = "v7_pybot"
            elif fname.startswith("05_crossling"):
                gkey = "v9_combined_ZH_EN_crossling"
            elif "CS" in fname and lbl.startswith("v9_"):
                gkey = "v9_combined_CS"
            elif lbl in ("v9_combined", "v9_target_first", "v9_cross_donor"):
                lang_tag = "ZH" if fname.startswith(("01_", "02_")) else "EN"
                gkey = f"v9_{lang_tag}_{lbl.replace('v9_', '')}"
            else:
                gkey = f"other_{lbl}"
            g_rows, g_d = groups.setdefault(gkey, ([], []))
            g_rows.append(m)
            g_d.append(deltas)

            if lbl.startswith("v9_") or lbl in ("SSB0502_speaks_EN", "SSB0073_M133_speaks_EN", "slt_F188_speaks_ZH", "bdl_M119_speaks_ZH"):
                va_rows, va_d = groups.setdefault("v9_ALL_segments", ([], []))
                va_rows.append(m)
                va_d.append(deltas)

        per_file[fname] = seg_rows

    grouped_summary = {k: _aggregate_loudness_rows(rows, dls) for k, (rows, dls) in sorted(groups.items())}
    return {"per_file": per_file, "grouped_summary": grouped_summary}


def audit_real_corpora() -> dict[str, Any]:
    """Measure real human loudness jump distributions on extended/ ZH and CMU-ARCTIC EN."""
    zh_utts = json.loads((ROOT / "pybot_tts_v2_recorded/extended/index.json").read_text(encoding="utf-8"))["utterances"]
    en_man = json.loads((ROOT / "exp_v8_data/corpus/manifest.json").read_text(encoding="utf-8"))["english"]["utterances"]
    en_held = [u for u in en_man if u["speaker"] in ("slt", "bdl") and int(u["id"].split("_")[1]) >= 30]

    out = {}
    for name, utts in [("real_zh_extended_32", zh_utts), ("real_en_cmu_heldout_20", en_held), ("real_en_cmu_all_160", en_man)]:
        rows, dls = [], []
        for u in utts:
            x, sr = read_wav(ROOT / u["path"])
            if "active_start" in u and "active_stop" in u:
                xa = x[int(u["active_start"]) : int(u["active_stop"])]
            else:
                s0, s1 = _trim_active_span(x, sr=sr)
                xa = x[s0:s1]
            rows.append(measure_windowed_loudness(xa, sr=sr))
            dls.append(_extract_pair_deltas(xa, sr=sr))
        out[name] = _aggregate_loudness_rows(rows, dls)
    return out


def _build_audit_bank_and_profiles():
    zh_utts = json.loads((ROOT / "pybot_tts_v2_recorded/extended/index.json").read_text(encoding="utf-8"))["utterances"]
    en_man = json.loads((ROOT / "exp_v8_data/corpus/manifest.json").read_text(encoding="utf-8"))["english"]["utterances"]
    en_held = [u for u in en_man if u["speaker"] in ("slt", "bdl") and int(u["id"].split("_")[1]) >= 30]
    en_excl = {u["id"] for u in en_held}

    zh_pre, en_pre = {}, {}
    zh_by_spk, en_by_spk = {}, {}
    for u in zh_utts:
        x, _ = read_wav(ROOT / u["path"])
        xa = x[int(u["active_start"]) : int(u["active_stop"])]
        fv9, bg = analyze_v9(xa, sr=SR)
        zh_pre[u["id"]] = (xa, fv9, bg)
        zh_by_spk.setdefault(u["speaker"], []).append((xa, fv9))
    for u in en_man:
        if u["id"] in en_excl:
            continue
        x, _ = read_wav(ROOT / u["path"])
        s0, s1 = _trim_active_span(x, sr=SR)
        xa = x[s0:s1]
        fv9, bg = analyze_v9(xa, sr=SR)
        en_pre[u["id"]] = (xa, fv9, bg)
        en_by_spk.setdefault(u["speaker"], []).append((xa, fv9))

    v3_profs = {}
    for spk, items in zh_by_spk.items():
        v3_profs[spk] = tc3.extract_timbre_v3(items, sr=SR, name=spk, corpus="aishell3")
    for spk, items in en_by_spk.items():
        v3_profs[spk] = tc3.extract_timbre_v3(items, sr=SR, name=spk, corpus="cmu_arctic")

    bank = build_v9_dual_bank(
        zh_utterances=zh_utts,
        zh_exclude_ids=(),
        en_utterances=en_man,
        en_exclude_ids=en_excl,
        zh_precomputed=zh_pre,
        en_precomputed=en_pre,
        refine_boundaries=True,
    )
    return zh_utts, en_held, zh_pre, en_pre, v3_profs, bank


def audit_code_culprits(
    zh_utts: list[dict],
    en_held: list[dict],
    zh_pre: dict,
    en_pre: dict,
    v3_profs: dict[str, tc3.TimbreProfileV3],
    bank: V9DualBank,
) -> dict[str, Any]:
    """Quantify each exact code culprit driving v9 loudness discontinuities."""
    # Culprit A: -6.9 nepers voiced-median normalization + per-token loudness spread & utterance shift variance
    en_utt_shifts = []
    for u in json.loads((ROOT / "exp_v8_data/corpus/manifest.json").read_text(encoding="utf-8"))["english"]["utterances"]:
        if u["id"] not in en_pre:
            continue
        _, fv9, _ = en_pre[u["id"]]
        vm = fv9.voiced & (fv9.f0_hz > 0.0)
        if np.any(vm):
            med = float(np.median(np.mean(fv9.log_env[vm], axis=1)))
            en_utt_shifts.append((-6.9 - med) * 20.0 / math.log(10.0))
    zh_utt_shifts = []
    for u in zh_utts:
        _, fv9, _ = zh_pre[u["id"]]
        vm = fv9.voiced & (fv9.f0_hz > 0.0)
        if np.any(vm):
            med = float(np.median(np.mean(fv9.log_env[vm], axis=1)))
            zh_utt_shifts.append((-6.9 - med) * 20.0 / math.log(10.0))

    tok_lvl_by_manner: dict[str, list[float]] = {"vowel": [], "sonorant": [], "uv_fricative": [], "uv_stop": [], "v_obstruent": []}
    for ipa, toks in bank.en_phones.items():
        sp = IPA_SPECS.get(ipa)
        if sp is None:
            continue
        if sp.manner == "vowel":
            mkey = "vowel"
        elif sp.manner in ("nasal", "approximant"):
            mkey = "sonorant"
        elif not sp.voiced and sp.manner in ("fricative", "affricate"):
            mkey = "uv_fricative"
        elif not sp.voiced and sp.manner == "stop":
            mkey = "uv_stop"
        else:
            mkey = "v_obstruent"
        for t in toks:
            # Frame power in dB: 10*log10(mean(exp(2*log_env)))
            pwr_db = 10.0 * math.log10(max(1e-16, float(np.mean(np.exp(2.0 * t.log_env.astype(np.float64))))))
            tok_lvl_by_manner[mkey].append(pwr_db)

    tok_manner_summary = {}
    for k, vals in tok_lvl_by_manner.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size:
            tok_manner_summary[k] = {
                "count": int(arr.size),
                "p10_db": round(float(np.percentile(arr, 10)), 2),
                "median_db": round(float(np.percentile(arr, 50)), 2),
                "p90_db": round(float(np.percentile(arr, 90)), 2),
                "spread_p90_p10_db": round(float(np.percentile(arr, 90) - np.percentile(arr, 10)), 2),
            }

    # Culprit B: TimbreClonerV3 v_mask power preservation vs ~v_mask unpreserved shift
    cloner_pairs = [
        ("bdl", "SSB0502"),
        ("slt", "SSB0502"),
        ("clb", "SSB0502"),
        ("rms", "SSB0502"),
        ("SSB0502", "slt"),
        ("SSB0502", "bdl"),
        ("SSB1593", "SSB0502"),
    ]
    cloner_v_vs_uv = []
    for src_s, tgt_s in cloner_pairs:
        if src_s in en_pre or src_s == "bdl":
            pass
        sample_feats = None
        if src_s in ("SSB0502", "SSB1593"):
            for u in zh_utts:
                if u["speaker"] == src_s:
                    sample_feats = zh_pre[u["id"]][1]
                    break
        else:
            for uid, (_, fv9, _) in en_pre.items():
                if uid.startswith(src_s):
                    sample_feats = fv9
                    break
        if sample_feats is None or src_s not in v3_profs or tgt_s not in v3_profs:
            continue
        ad, rep = tc3.adapt_features_v3(sample_feats, v3_profs[src_s], v3_profs[tgt_s], eq_mode="blind_tilt_anchor")
        pre_pwr = np.mean(np.exp(2.0 * sample_feats.log_env), axis=1)
        post_pwr = np.mean(np.exp(2.0 * ad.log_env), axis=1)
        vm = sample_feats.voiced
        um = ~vm
        v_delta_db = 10.0 * np.log10(np.maximum(post_pwr[vm], 1e-16) / np.maximum(pre_pwr[vm], 1e-16)) if np.any(vm) else np.zeros(1)
        u_delta_db = 10.0 * np.log10(np.maximum(post_pwr[um], 1e-16) / np.maximum(pre_pwr[um], 1e-16)) if np.any(um) else np.zeros(1)
        cloner_v_vs_uv.append(
            {
                "pair": f"{src_s}->{tgt_s}",
                "v_mask_median_delta_db": round(float(np.median(v_delta_db)), 3),
                "u_mask_median_delta_db": round(float(np.median(u_delta_db)), 3),
                "u_mask_max_abs_delta_db": round(float(np.max(np.abs(u_delta_db))), 3),
                "v_minus_u_step_db": round(float(np.median(v_delta_db) - np.median(u_delta_db)), 3),
            }
        )

    # Culprit C & D & E: Controlled ablation on fresh renders (ZH, EN target-first, EN cross-donor, Crossling ZH<-EN, CS)
    adapter = make_v3_cloner_adapter(v3_profs, eq_mode="blind_tilt_anchor")
    en_held_by_spk = {
        "slt": [u for u in en_held if u["speaker"] == "slt"][:3],
        "bdl": [u for u in en_held if u["speaker"] == "bdl"][:3],
    }
    bank_no_slt = filter_v9_bank_excluding_speakers(bank, {"slt"})
    bank_no_bdl = filter_v9_bank_excluding_speakers(bank, {"bdl"})
    zh_sample_utts = [u for u in zh_utts if u["id"] in ("SSB05020122", "SSB05020110", "SSB15930193", "SSB15930140")]
    bench_data = json.loads((ROOT / "exp_v5_arch/benchmark_texts.json").read_text(encoding="utf-8"))
    cs_texts = [
        s for s in bench_data.get("sentences", [])
        if str(s.get("id", "")).startswith("CS") or str(s.get("lang", "")) in ("cs", "codeswitch", "mixed")
    ][:4]

    ablation_configs = [
        ("full_v9_combined", True, True, "natural"),
        ("no_unvoiced_residual", False, True, "natural"),
        ("no_cloner_v3", True, False, "natural"),
        ("no_res_no_cloner", False, False, "natural"),
        ("no_res_flat_prosody", False, True, "flat"),
    ]
    ablation_results: dict[str, Any] = {}
    for cfg_name, use_res, use_cloner, pmode in ablation_configs:
        cl_fn = adapter if use_cloner else None
        all_rows, all_dls = [], []
        by_mode: dict[str, tuple[list[dict], list[np.ndarray]]] = {}

        # EN target-first + EN cross-donor + ZH->EN crossling
        for spk in ("slt", "bdl"):
            for u in en_held_by_spk[spk]:
                pl = cf.plan(u["text"], allow_oov_fallback=True)
                y_tf, _ = render_v9_utterance(
                    pl, bank, v3_profs[spk], cloner_fn=cl_fn, src_profiles=v3_profs,
                    use_unvoiced_residual=use_res, prosody_mode=pmode, seed=72,
                )
                b_cd = bank_no_slt if spk == "slt" else bank_no_bdl
                y_cd, _ = render_v9_utterance(
                    pl, b_cd, v3_profs[spk], cloner_fn=cl_fn, src_profiles=v3_profs,
                    use_unvoiced_residual=use_res, prosody_mode=pmode, seed=72,
                )
                for mk, yy in [("en_target_first", y_tf), ("en_cross_donor", y_cd)]:
                    m = measure_windowed_loudness(yy, sr=SR)
                    dl = _extract_pair_deltas(yy, sr=SR)
                    all_rows.append(m); all_dls.append(dl)
                    br, bd = by_mode.setdefault(mk, ([], []))
                    br.append(m); bd.append(dl)

        # Crossling SSB0502 speaks EN + slt speaks ZH
        for u in en_held_by_spk["slt"][:2]:
            pl = cf.plan(u["text"], allow_oov_fallback=True)
            y_xl, _ = render_v9_utterance(
                pl, bank, v3_profs["SSB0502"], cloner_fn=cl_fn, src_profiles=v3_profs,
                use_unvoiced_residual=use_res, prosody_mode=pmode, seed=72,
            )
            m = measure_windowed_loudness(y_xl, sr=SR)
            dl = _extract_pair_deltas(y_xl, sr=SR)
            all_rows.append(m); all_dls.append(dl)
            br, bd = by_mode.setdefault("crossling_zh_spk_en", ([], []))
            br.append(m); bd.append(dl)

        for u in zh_sample_utts:
            pl = v6_arb.plan_from_pinyin_syllables(u["syllables"], text=u["text"])
            y_zh, _ = render_v9_utterance(
                pl, bank, v3_profs[u["speaker"]], cloner_fn=cl_fn, src_profiles=v3_profs,
                use_unvoiced_residual=use_res, prosody_mode=pmode, seed=72,
            )
            m = measure_windowed_loudness(y_zh, sr=SR)
            dl = _extract_pair_deltas(y_zh, sr=SR)
            all_rows.append(m); all_dls.append(dl)
            br, bd = by_mode.setdefault("zh_loo", ([], []))
            br.append(m); bd.append(dl)

        for item in cs_texts:
            txt = item["text"] if isinstance(item, dict) else str(item)
            pl = cf.plan(txt, allow_oov_fallback=True)
            y_cs, _ = render_v9_utterance(
                pl, bank, v3_profs["SSB0502"], cloner_fn=cl_fn, src_profiles=v3_profs,
                use_unvoiced_residual=use_res, prosody_mode=pmode, seed=72,
            )
            m = measure_windowed_loudness(y_cs, sr=SR)
            dl = _extract_pair_deltas(y_cs, sr=SR)
            all_rows.append(m); all_dls.append(dl)
            br, bd = by_mode.setdefault("codeswitch", ([], []))
            br.append(m); bd.append(dl)

        ablation_results[cfg_name] = {
            "overall": _aggregate_loudness_rows(all_rows, all_dls),
            "by_mode": {mk: _aggregate_loudness_rows(br, bd) for mk, (br, bd) in sorted(by_mode.items())},
        }

    # Measure direct seam jumps (5 ms feature env & 40 ms RMS) on obstruent 0 ms vs sonorant 10 ms joins
    seam_diag = _measure_seam_and_residual_diagnostics(en_held_by_spk["slt"][:3], v3_profs, bank, adapter)

    return {
        "culprit_a_utterance_shift_db": {
            "en_shift_std_db": round(float(np.std(en_utt_shifts)), 3) if en_utt_shifts else 0.0,
            "en_shift_range_db": [round(float(min(en_utt_shifts)), 2), round(float(max(en_utt_shifts)), 2)] if en_utt_shifts else [0.0, 0.0],
            "zh_shift_std_db": round(float(np.std(zh_utt_shifts)), 3) if zh_utt_shifts else 0.0,
            "zh_shift_range_db": [round(float(min(zh_utt_shifts)), 2), round(float(max(zh_utt_shifts)), 2)] if zh_utt_shifts else [0.0, 0.0],
            "en_token_log_env_level_by_manner_db": tok_manner_summary,
        },
        "culprit_b_cloner_v3_v_vs_uv_db": cloner_v_vs_uv,
        "culprit_cde_render_ablation": ablation_results,
        "culprit_d_seam_and_residual_micro": seam_diag,
    }


def _measure_seam_and_residual_diagnostics(
    en_utts: list[dict],
    v3_profs: dict[str, tc3.TimbreProfileV3],
    bank: V9DualBank,
    adapter,
) -> dict[str, Any]:
    """Inspect frame-level energy steps specifically at 0 ms obstruent vs 10 ms voiced seams, plus residual patch gain."""
    obs_seam_40ms = []
    son_seam_40ms = []
    obs_cmvn_delta = []
    obs_actual_lvl_delta_db = []
    son_actual_lvl_delta_db = []
    res_gain_db = []
    res_fullband_delta_db = []

    hp_sos = _hp_sos(SR, 1600.0)
    win = 960  # 40 ms
    for u in en_utts:
        pl = cf.plan(u["text"], allow_oov_fallback=True)
        y_res, info = render_v9_utterance(
            pl, bank, v3_profs["SSB0502"], cloner_fn=adapter, src_profiles=v3_profs,
            use_unvoiced_residual=True, seed=72,
        )
        y_no, _ = render_v9_utterance(
            pl, bank, v3_profs["SSB0502"], cloner_fn=adapter, src_profiles=v3_profs,
            use_unvoiced_residual=False, seed=72,
        )
        # Inspect consecutive phone energy jumps using bank tokens directly along the planned pairs
        prev_tok = None
        for unit in pl["units"]:
            if unit["lang"] != "en":
                prev_tok = None
                continue
            arpa = unit.get("arpabet") or []
            pairs = v6_arb._split_en_arpabet_to_ipa_with_stress(arpa) if arpa else [(p, None) for p in (unit.get("ipa_phones") or [])]
            for ip, _ in pairs:
                cands = bank.en_phones.get(ip, [])
                if not cands:
                    prev_tok = None
                    continue
                tok = max(cands, key=lambda c: -len(c.log_env) * 0.0 + (1.0 if c.prev_ipa == (prev_tok.ipa if prev_tok else "SIL") else 0.0))
                if prev_tok is not None:
                    l_prev = 10.0 * math.log10(max(1e-16, float(np.mean(np.exp(2.0 * prev_tok.log_env[-1].astype(np.float64))))))
                    l_cur = 10.0 * math.log10(max(1e-16, float(np.mean(np.exp(2.0 * tok.log_env[0].astype(np.float64))))))
                    diff_lvl = abs(l_cur - l_prev)
                    sp_c = IPA_SPECS.get(ip)
                    sp_p = IPA_SPECS.get(prev_tok.ipa)
                    is_obs = (sp_c is not None and sp_c.manner in ("fricative", "affricate", "stop")) or (
                        sp_p is not None and sp_p.manner in ("fricative", "affricate", "stop")
                    )
                    if is_obs:
                        obs_actual_lvl_delta_db.append(diff_lvl)
                        obs_cmvn_delta.append(float(math.sqrt(float(np.mean((prev_tok.cmvn_right - tok.cmvn_left) ** 2)))))
                    else:
                        son_actual_lvl_delta_db.append(diff_lvl)
                prev_tok = tok

        # Measure full-band RMS difference where unvoiced residual patched audio vs y_no
        n_min = min(len(y_res), len(y_no))
        for s in range(0, n_min - win, 480):
            d_rms = abs(
                20.0 * math.log10(max(1e-9, float(np.sqrt(np.mean(y_res[s : s + win] ** 2)))))
                - 20.0 * math.log10(max(1e-9, float(np.sqrt(np.mean(y_no[s : s + win] ** 2)))))
            )
            if d_rms > 0.25:
                res_fullband_delta_db.append(d_rms)

    # Direct measurement of r_syn / r_pat inside synthesize_v9 residual spans on 1 sample EN utterance
    for u in en_utts[:1]:
        pl = cf.plan(u["text"], allow_oov_fallback=True)
        feats_dummy = en_utts
        # Inspect unvoiced tokens raw_wav HP RMS vs log_env >1.6k expected RMS
        for tok_list in list(bank.en_phones.values())[:20]:
            for tok in tok_list[:2]:
                if tok.raw_wav is not None and len(tok.raw_wav) >= 72:
                    src_hp = sosfiltfilt(hp_sos, tok.raw_wav.astype(np.float64))
                    r_pat = float(np.sqrt(np.mean(src_hp ** 2)))
                    r_full = float(np.sqrt(np.mean(tok.raw_wav.astype(np.float64) ** 2)))
                    if r_pat > 1e-8 and r_full > 1e-8:
                        res_gain_db.append(20.0 * math.log10(r_pat / r_full))

    return {
        "obstruent_join_boundary_lvl_jump_db": {
            "n": len(obs_actual_lvl_delta_db),
            "median_db": round(float(np.median(obs_actual_lvl_delta_db)) if obs_actual_lvl_delta_db else 0.0, 2),
            "p95_db": round(float(np.percentile(obs_actual_lvl_delta_db, 95)) if obs_actual_lvl_delta_db else 0.0, 2),
            "max_db": round(float(np.max(obs_actual_lvl_delta_db)) if obs_actual_lvl_delta_db else 0.0, 2),
        },
        "sonorant_join_boundary_lvl_jump_db": {
            "n": len(son_actual_lvl_delta_db),
            "median_db": round(float(np.median(son_actual_lvl_delta_db)) if son_actual_lvl_delta_db else 0.0, 2),
            "p95_db": round(float(np.percentile(son_actual_lvl_delta_db, 95)) if son_actual_lvl_delta_db else 0.0, 2),
            "max_db": round(float(np.max(son_actual_lvl_delta_db)) if son_actual_lvl_delta_db else 0.0, 2),
        },
        "residual_patch_40ms_fullband_diff_db": {
            "n_changed_windows": len(res_fullband_delta_db),
            "median_db": round(float(np.median(res_fullband_delta_db)) if res_fullband_delta_db else 0.0, 3),
            "max_db": round(float(np.max(res_fullband_delta_db)) if res_fullband_delta_db else 0.0, 3),
        },
    }


def main() -> int:
    print("[1/4] Auditing exp_v9_eval/listening_ab/*.wav segment-by-segment...", flush=True)
    listening_rep = audit_listening_ab(
        ROOT / "exp_v9_eval/v9_benchmark_report.json",
        ROOT / "exp_v9_eval/listening_ab",
    )
    print("[2/4] Auditing real human speech baselines (ZH extended + EN CMU-ARCTIC)...", flush=True)
    real_rep = audit_real_corpora()
    print("[3/4] Building v9 bank & running culprit quantification + ablations...", flush=True)
    zh_utts, en_held, zh_pre, en_pre, v3_profs, bank = _build_audit_bank_and_profiles()
    culprits = audit_code_culprits(zh_utts, en_held, zh_pre, en_pre, v3_profs, bank)

    out = {
        "window_params": {"win_ms": 40.0, "hop_ms": 20.0, "active_gate_dbfs_after_neg20_norm": -48.0},
        "real_corpora": real_rep,
        "listening_ab": listening_rep,
        "code_culprits": culprits,
    }
    REPORT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[4/4] Saved {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
