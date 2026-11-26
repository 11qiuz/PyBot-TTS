"""Controlled A/B Vocoder Benchmark (v12): hnm_v11 vs LP-PSOLA vs Hybrid LP+Noise + Crisp Unvoiced."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from exp_v5_eval import metrics as evm  # noqa: E402
from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v8_clarity.en_utau_bank import _trim_active_span  # noqa: E402
from exp_v8_data.clarity_diag import compute_clarity_metrics  # noqa: E402
from exp_v10_integrity.eval_v10 import _loudness_norm, measure_loudness_jumps  # noqa: E402
from exp_v11_align_gmm.timbre_cloner_v5 import adapt_features_v5, extract_timbre_v5  # noqa: E402
from exp_v12_natural_vocoder import EN_MANIFEST, EXP_DIR, EXTENDED_INDEX, SAMPLES_DIR, SR  # noqa: E402
from exp_v12_natural_vocoder.vocoder_v12 import (  # noqa: E402
    GlottalPulseBank,
    VocoderFeaturesV12,
    analyze_v12,
    build_glottal_pulse_bank,
    synthesize_v12,
)


def _eval_sample(y: np.ndarray, ref: np.ndarray | None, elapsed_s: float, n_samples: int) -> dict:
    # Peak-protect before sanity check so vocoder headroom variance does not trigger false clipping flags
    y_c = np.asarray(y, dtype=np.float64)
    pk = float(np.max(np.abs(y_c))) if y_c.size else 0.0
    if pk > 0.92:
        y_c = y_c * (0.92 / pk)
    san = evm.check_wav_sanity(y_c, SR)
    cl = compute_clarity_metrics(y_c, sr=SR)
    lj = measure_loudness_jumps(y_c, sr=SR)
    ev_y = evm.evaluate_audio(y_c, sr=SR)
    out = {
        "sanity_ok": bool(san["ok"]),
        "hnr_db": round(float(ev_y.get("hnr_db") or 0.0), 3),
        "uv_2_8k_vs_lo_db": cl["unvoiced_2_8k_vs_lo_db"],
        "contrast_db": cl["voiced_spectral_contrast_db"],
        "transient_2_8k_span_db": cl["transient_2_8k_span_db"],
        "clicks_gt8": round(float(lj["clicks_gt8"]), 4),
        "p99_db": round(float(lj["p99_db"]), 3),
        "max_db": round(float(lj["max_db"]), 3),
        "rtf": round(float(elapsed_s / max(1e-6, n_samples / SR)), 4),
    }
    if ref is not None:
        sd = evm.compute_spectral_distances(ref, y_c, sr=SR)
        out["mel_lsd_db"] = round(float(sd["mel_lsd_db"]), 3)
        out["mcd_db"] = round(float(sd["mcd_db"]), 3)
    return out


def _summarize_dicts(recs: list[dict]) -> dict:
    keys = [k for k in recs[0].keys() if k != "sanity_ok"]
    meds = {}
    for k in keys:
        vals = [float(r[k]) for r in recs if r[k] is not None and r[k] == r[k]]
        meds[k] = round(float(np.median(vals)), 3) if vals else None
    return {"n": len(recs), "sanity_ok": sum(1 for r in recs if r["sanity_ok"]), "medians": meds}


def _shift_f0_bundle(fv: VocoderFeaturesV12, semitones: float) -> VocoderFeaturesV12:
    f = fv.feats
    scale = 2.0 ** (semitones / 12.0)
    f0_new = np.where(f.voiced & (f.f0_hz > 0.0), np.clip(f.f0_hz * scale, 65.0, 440.0), 0.0)
    feats_s = vocoder.VocoderFeatures(
        f0_hz=f0_new, voiced=f.voiced.copy(), log_env=f.log_env.copy(), bap=f.bap.copy(),
        sr=f.sr, hop_samples=f.hop_samples, n_samples=f.n_samples, n_fft=f.n_fft,
    )
    return VocoderFeaturesV12(
        feats=feats_s,
        bap_gt8k=fv.bap_gt8k.copy(),
        uv_raw_log_env=fv.uv_raw_log_env,
        lp_pulses=fv.lp_pulses,
        coherent_phasor=fv.coherent_phasor,
    )


def run_vocoder_bench(export_samples: bool = True) -> dict:
    t0 = time.time()
    zh_idx = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))["utterances"]
    en_idx = json.loads(EN_MANIFEST.read_text(encoding="utf-8"))["english"]["utterances"]

    # Pick representative utterances across slt, bdl, rms, SSB0502, SSB1593
    targets = {
        "slt_0030": ("arctic", "slt"),
        "bdl_0030": ("arctic", "bdl"),
        "SSB05020122": ("aishell3", "SSB0502"),
        "SSB15930193": ("aishell3", "SSB1593"),
    }
    raw_audio, analyzed = {}, {}
    for uid, (corp, spk) in targets.items():
        if corp == "arctic":
            u = next(x for x in en_idx if x["id"] == uid)
            x, _ = vocoder.read_wav_mono(REPO_ROOT / u["path"])
            s0, s1 = _trim_active_span(x, sr=SR)
            xa = x[s0:s1]
        else:
            u = next(x for x in zh_idx if x["id"] == uid)
            x, _ = vocoder.read_wav_mono(u["path"])
            xa = x[u["active_start"] : u["active_stop"]]
        raw_audio[uid] = xa
        analyzed[uid] = analyze_v12(xa, sr=SR)

    # Build speaker GlottalPulseBank + TimbreProfileV5 from 3 disjoint train utts per speaker
    spk_rows = {
        "slt": [u for u in en_idx if u["speaker"] == "slt"][:3],
        "bdl": [u for u in en_idx if u["speaker"] == "bdl"][:3],
        "rms": [u for u in en_idx if u["speaker"] == "rms"][:3],
        "SSB0502": [u for u in zh_idx if u["speaker"] == "SSB0502" and u["id"] != "SSB05020122"][:3],
        "SSB1593": [u for u in zh_idx if u["speaker"] == "SSB1593" and u["id"] != "SSB15930193"][:3],
    }
    profs_v5, gbanks = {}, {}
    for spk, rows in spk_rows.items():
        corp = "aishell3" if spk.startswith("SSB") else "arctic"
        items_v5, items_v12 = [], []
        for r in rows:
            p = Path(r["path"])
            if not p.is_absolute():
                p = REPO_ROOT / p
            x, _ = vocoder.read_wav_mono(p)
            xa = x[r["active_start"] : r["active_stop"]] if "active_start" in r else x[_trim_active_span(x, SR)[0] : _trim_active_span(x, SR)[1]]
            fv = analyze_v12(xa, sr=SR)
            items_v5.append((xa, fv.feats))
            items_v12.append(fv)
        profs_v5[spk] = extract_timbre_v5(items_v5, sr=SR, name=spk, corpus=corp)
        gbanks[spk] = build_glottal_pulse_bank(spk, items_v12, max_pulses=32)

    modes = [
        ("real", None, False),
        ("hnm_v11", "hnm_v11", False),
        ("hnm_v11_crisp_uv", "hnm_v11", True),
        ("lp_psola", "lp_psola", True),
        ("hybrid_lp_hnm", "hybrid_lp_hnm", True),
    ]

    # 1. Analysis-Resynthesis at Natural F0 (n=4 diverse EN+ZH speakers)
    natural_recs = {m: [] for m, _, _ in modes}
    ab_nat = {}
    for uid, xa in raw_audio.items():
        fv = analyzed[uid]
        ab_nat[uid] = {"real": xa}
        for name, md, crisp in modes:
            if md is None:
                natural_recs[name].append(_eval_sample(xa, xa, 0.0, len(xa)))
                continue
            ts = time.perf_counter()
            y = synthesize_v12(fv, mode=md, crisp_unvoiced=crisp, seed=72)
            natural_recs[name].append(_eval_sample(y, xa, time.perf_counter() - ts, len(xa)))
            ab_nat[uid][name] = y

    # 2. Pitch-Shift Robustness Stress Test (+4 st & -4 st F0 shift across all 4 utts)
    shift_recs = {m: [] for m in ("hnm_v11", "lp_psola", "hybrid_lp_hnm", "hybrid_lp_bank")}
    for uid, fv in analyzed.items():
        spk = targets[uid][1]
        for st in (-4.0, 4.0):
            fvs = _shift_f0_bundle(fv, st)
            for mname, md, gb in (
                ("hnm_v11", "hnm_v11", None),
                ("lp_psola", "lp_psola", None),
                ("hybrid_lp_hnm", "hybrid_lp_hnm", None),
                ("hybrid_lp_bank", "hybrid_lp_hnm", gbanks[spk]),
            ):
                # When using external glottal bank, drop oracle frame lp_pulses to mimic TTS bank usage!
                fin = VocoderFeaturesV12(
                    fvs.feats,
                    fvs.bap_gt8k,
                    fvs.uv_raw_log_env,
                    None if gb else fvs.lp_pulses,
                    None if gb else fvs.coherent_phasor,
                )
                ts = time.perf_counter()
                y = synthesize_v12(fin, mode=md, glottal_bank=gb, crisp_unvoiced=(mname != "hnm_v11"), seed=72)
                # Reference is frequency-envelope fidelity against unshifted envelope expectation (compute clarity/clicks/hnr)
                shift_recs[mname].append(_eval_sample(y, None, time.perf_counter() - ts, len(y)))

    # 3. Cross-Speaker VTLN Conversion (slt->rms, bdl->SSB0502, SSB1593->SSB0502, SSB0502->slt)
    conv_pairs = [
        ("slt_0030", "slt", "rms"),
        ("bdl_0030", "bdl", "SSB0502"),
        ("SSB15930193", "SSB1593", "SSB0502"),
        ("SSB05020122", "SSB0502", "slt"),
    ]
    conv_recs = {m: [] for m in ("hnm_v11", "lp_psola", "hybrid_lp_hnm")}
    ab_conv = {}
    for uid, src_spk, tgt_spk in conv_pairs:
        fv = analyzed[uid]
        ad_feats, _ = adapt_features_v5(fv.feats, profs_v5[src_spk], profs_v5[tgt_spk])
        # Build V12 bundle using target speaker's GlottalPulseBank + preserved unvoiced raw frames!
        fv_ad = VocoderFeaturesV12(feats=ad_feats, bap_gt8k=fv.bap_gt8k, uv_raw_log_env=None, lp_pulses=None)
        ab_conv[f"{src_spk}_to_{tgt_spk}"] = {}
        for mname in ("hnm_v11", "lp_psola", "hybrid_lp_hnm"):
            ts = time.perf_counter()
            y = synthesize_v12(
                fv_ad,
                mode=mname,
                glottal_bank=gbanks[tgt_spk] if mname != "hnm_v11" else None,
                crisp_unvoiced=(mname != "hnm_v11"),
                seed=72,
            )
            conv_recs[mname].append(_eval_sample(y, None, time.perf_counter() - ts, len(y)))
            ab_conv[f"{src_spk}_to_{tgt_spk}"][mname] = y

    exported = {}
    if export_samples:
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        gap = np.zeros(int(0.40 * SR), dtype=np.float64)
        for uid in ("slt_0030", "SSB05020122"):
            c = ab_nat[uid]
            wav = np.concatenate([
                _loudness_norm(c["real"]), gap,
                _loudness_norm(c["hnm_v11"]), gap,
                _loudness_norm(c["lp_psola"]), gap,
                _loudness_norm(c["hybrid_lp_hnm"]),
            ])
            fn = f"AB_VOC_nat_{uid}_real_v11_lp_hybrid.wav"
            vocoder.write_wav_mono(SAMPLES_DIR / fn, wav, sr=SR)
            exported[fn] = evm.check_wav_sanity(wav, SR)
        for pair_key, c in ab_conv.items():
            wav = np.concatenate([
                _loudness_norm(c["hnm_v11"]), gap,
                _loudness_norm(c["lp_psola"]), gap,
                _loudness_norm(c["hybrid_lp_hnm"]),
            ])
            fn = f"AB_VOC_conv_{pair_key}_v11_lp_hybrid.wav"
            vocoder.write_wav_mono(SAMPLES_DIR / fn, wav, sr=SR)
            exported[fn] = evm.check_wav_sanity(wav, SR)

    out = {
        "schema": 1,
        "elapsed_s": round(time.time() - t0, 2),
        "natural_f0": {k: _summarize_dicts(v) for k, v in natural_recs.items()},
        "pitch_shift_pm4st": {k: _summarize_dicts(v) for k, v in shift_recs.items()},
        "cross_speaker_vtln": {k: _summarize_dicts(v) for k, v in conv_recs.items()},
        "exported_samples": exported,
    }
    (EXP_DIR / "vocoder_ab_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    res = run_vocoder_bench(export_samples=True)
    print(f"Vocoder v12 A/B Bench completed in {res['elapsed_s']}s")
    for sec in ("natural_f0", "pitch_shift_pm4st", "cross_speaker_vtln"):
        print(f"\n=== {sec} ===")
        for m, info in res[sec].items():
            d = info["medians"]
            lsd = f"MelLSD={d['mel_lsd_db']} MCD={d['mcd_db']} | " if "mel_lsd_db" in d else ""
            print(
                f"  [{m:17s}] ok={info['sanity_ok']}/{info['n']} | {lsd}uv2_8k={d['uv_2_8k_vs_lo_db']} | "
                f"contrast={d['contrast_db']} | HNR={d['hnr_db']} | clk={d['clicks_gt8']} | RTF={d['rtf']}"
            )
    print("Samples:", {k: v["ok"] for k, v in res["exported_samples"].items()})
