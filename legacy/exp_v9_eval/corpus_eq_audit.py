"""Corpus EQ & Speaker-Separability Analyzer across CMU-ARCTIC (EN) and AISHELL-3 (ZH)."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.fft import rfft, rfftfreq

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_eval.metrics import DEFAULT_SR, extract_active_frames, read_wav  # noqa: E402
from exp_v6_eval.fluency import compute_f0_track  # noqa: E402

BAND_SUMMARY_HZ = (
    ("80_250", 80.0, 250.0),
    ("250_1k", 250.0, 1000.0),
    ("1k_2k", 1000.0, 2000.0),
    ("2k_4k", 2000.0, 4000.0),
    ("4k_8k", 4000.0, 8000.0),
    ("8k_11.5k", 8000.0, 11500.0),
)

LTAS_24_GRID = np.geomspace(80.0, 11500.0, 24)


def compute_speaker_eq_signature(
    wav_specs: list[tuple[pathlib.Path, int, int | None]],
    *,
    fmin: float = 70.0,
    fmax: float = 420.0,
) -> dict:
    """Compute mean-centered 24-band (80-11500 Hz) voiced/unvoiced/all LTAS & band powers."""
    nfft = 2048
    frame_len = 720
    hop = 240
    freqs = rfftfreq(nfft, 1.0 / DEFAULT_SR)
    win = np.hamming(frame_len)[None, :]

    voiced_specs = []
    unvoiced_specs = []
    all_specs = []

    for path, a0, a1 in wav_specs:
        x, sr = read_wav(path)
        if sr != DEFAULT_SR:
            continue
        if a1 is not None or a0 > 0:
            x = x[max(0, a0):a1]
        frames, keep, _ = extract_active_frames(x, frame_len=frame_len, hop=hop, db_thresh=30.0)
        track = compute_f0_track(x, sr=sr, fmin=fmin, fmax=fmax)
        n_c = min(len(keep), len(track["voiced"]))
        if n_c < 4:
            continue
        frames = frames[:n_c]
        keep = keep[:n_c]
        voiced = track["voiced"][:n_c]

        spec_pow = np.abs(rfft((frames - np.mean(frames, axis=1, keepdims=True)) * win, n=nfft, axis=1)) ** 2
        if int(np.sum(keep)) >= 2:
            all_specs.append(np.mean(spec_pow[keep], axis=0))
        if int(np.sum(keep & voiced)) >= 2:
            voiced_specs.append(np.mean(spec_pow[keep & voiced], axis=0))
        if int(np.sum(keep & (~voiced))) >= 1:
            unvoiced_specs.append(np.mean(spec_pow[keep & (~voiced)], axis=0))

    def _to_24band_db(pow_list: list[np.ndarray]) -> np.ndarray | None:
        if not pow_list:
            return None
        mean_p = np.mean(np.stack(pow_list, axis=0), axis=0)
        half = (2.0 ** (1.0 / 3.0) - 1.0) * 0.5 * 1.2
        vals = []
        for fc in LTAS_24_GRID:
            lo, hi = fc * (1.0 - half), fc * (1.0 + half)
            m = (freqs >= lo) & (freqs <= hi)
            if int(np.sum(m)) < 2:
                idx = np.argsort(np.abs(freqs - fc))[:2]
                vals.append(float(np.mean(mean_p[idx])))
            else:
                vals.append(float(np.mean(mean_p[m])))
        db = 10.0 * np.log10(np.maximum(np.array(vals, dtype=np.float64), 1e-18))
        # Center over speech core 80-8000 Hz so >8k Nyquist cliff doesn't skew baseline
        core = LTAS_24_GRID <= 8000.0
        return db - float(np.mean(db[core]))

    def _band_summary(pow_list: list[np.ndarray]) -> dict:
        if not pow_list:
            return {}
        mean_p = np.mean(np.stack(pow_list, axis=0), axis=0)
        ref_m = (freqs >= 250.0) & (freqs <= 2000.0)
        ref_db = 10.0 * np.log10(float(np.mean(mean_p[ref_m])) + 1e-18)
        out = {}
        for name, lo, hi in BAND_SUMMARY_HZ:
            m = (freqs >= lo) & (freqs <= hi)
            b_db = 10.0 * np.log10(float(np.mean(mean_p[m])) + 1e-18)
            out[name] = round(b_db - ref_db, 3)
        return out

    v_24 = _to_24band_db(voiced_specs)
    a_24 = _to_24band_db(all_specs)
    u_24 = _to_24band_db(unvoiced_specs)
    return {
        "n_utts": len(wav_specs),
        "voiced_24_db": v_24.round(3).tolist() if v_24 is not None else None,
        "all_24_db": a_24.round(3).tolist() if a_24 is not None else None,
        "unvoiced_24_db": u_24.round(3).tolist() if u_24 is not None else None,
        "voiced_bands_rel_250_2k_db": _band_summary(voiced_specs),
        "all_bands_rel_250_2k_db": _band_summary(all_specs),
    }


def run_corpus_eq_audit() -> dict:
    """Measure exact per-speaker and per-corpus EQ transfer curves (CMU-ARCTIC vs AISHELL-3)."""
    manifest = json.loads((ROOT / "exp_v8_data/corpus/manifest.json").read_text(encoding="utf-8"))
    ext_idx = json.loads((ROOT / "pybot_tts_v2_recorded/extended/index.json").read_text(encoding="utf-8"))

    by_spk: dict[str, dict] = {}
    for u in manifest["english"]["utterances"]:
        spk = u["speaker"]
        by_spk.setdefault(spk, {"corpus": "CMU-ARCTIC", "lang": "en", "gender": u["gender"], "items": []})
        by_spk[spk]["items"].append((ROOT / u["path"], 0, None))
    for u in manifest["mandarin"]["utterances"]:
        spk = u["speaker"]
        by_spk.setdefault(spk, {"corpus": "AISHELL-3_new", "lang": "zh", "gender": u["gender"], "items": []})
        by_spk[spk]["items"].append((ROOT / u["path"], 0, None))
    for u in ext_idx["utterances"]:
        spk = u["speaker"]
        by_spk.setdefault(spk, {"corpus": "AISHELL-3_ext", "lang": "zh", "gender": "female", "items": []})
        by_spk[spk]["items"].append((pathlib.Path(u["path"]), int(u["active_start"]), int(u["active_stop"])))

    speaker_sigs = {}
    for spk, info in sorted(by_spk.items()):
        fmin = 65.0 if info["gender"] == "male" else 110.0
        sig = compute_speaker_eq_signature(info["items"], fmin=fmin, fmax=420.0)
        speaker_sigs[spk] = {
            "corpus": info["corpus"],
            "lang": info["lang"],
            "gender": info["gender"],
            **sig,
        }

    # Corpus-level mean profiles (equal weight per speaker)
    cmu_spks = [s for s, d in speaker_sigs.items() if d["corpus"] == "CMU-ARCTIC"]
    ais_spks = [s for s, d in speaker_sigs.items() if d["corpus"].startswith("AISHELL-3")]

    cmu_v = np.mean(np.stack([speaker_sigs[s]["voiced_24_db"] for s in cmu_spks], axis=0), axis=0)
    ais_v = np.mean(np.stack([speaker_sigs[s]["voiced_24_db"] for s in ais_spks], axis=0), axis=0)
    diff_cmu_minus_ais = cmu_v - ais_v

    # Also inspect bdl vs SSB0073 specifically
    bdl_v = np.array(speaker_sigs["bdl"]["voiced_24_db"], dtype=np.float64)
    s73_v = np.array(speaker_sigs["SSB0073"]["voiced_24_db"], dtype=np.float64)
    diff_bdl_minus_73 = bdl_v - s73_v
    diff_bdl_minus_73_deeq = (bdl_v - cmu_v) - (s73_v - ais_v)

    core_mask = LTAS_24_GRID <= 8000.0
    x_log2 = np.log2(LTAS_24_GRID[core_mask])
    slope_cmu_ais, _ = np.polyfit(x_log2, diff_cmu_minus_ais[core_mask], 1)
    slope_bdl_73_raw, _ = np.polyfit(x_log2, diff_bdl_minus_73[core_mask], 1)
    slope_bdl_73_deeq, _ = np.polyfit(x_log2, diff_bdl_minus_73_deeq[core_mask], 1)

    # Peak ridge in 2-4 kHz
    m_2_4 = (LTAS_24_GRID >= 2000.0) & (LTAS_24_GRID <= 4000.0)
    m_gt8 = LTAS_24_GRID > 8000.0

    transfer = {
        "grid_hz": [round(float(f), 1) for f in LTAS_24_GRID],
        "cmu_arctic_mean_voiced_db": cmu_v.round(3).tolist(),
        "aishell3_mean_voiced_db": ais_v.round(3).tolist(),
        "cmu_minus_aishell_voiced_db": diff_cmu_minus_ais.round(3).tolist(),
        "summary": {
            "cmu_minus_aishell_core_rmse_db": round(float(np.sqrt(np.mean(diff_cmu_minus_ais[core_mask] ** 2))), 3),
            "cmu_minus_aishell_tilt_db_oct": round(float(slope_cmu_ais), 3),
            "cmu_minus_aishell_2_4k_mean_db": round(float(np.mean(diff_cmu_minus_ais[m_2_4])), 3),
            "cmu_minus_aishell_gt8k_mean_db": round(float(np.mean(diff_cmu_minus_ais[m_gt8])), 3),
            "bdl_minus_SSB0073_raw_tilt_db_oct": round(float(slope_bdl_73_raw), 3),
            "bdl_minus_SSB0073_raw_2_4k_db": round(float(np.mean(diff_bdl_minus_73[m_2_4])), 3),
            "bdl_minus_SSB0073_corpus_deeq_tilt_db_oct": round(float(slope_bdl_73_deeq), 3),
            "bdl_minus_SSB0073_corpus_deeq_2_4k_db": round(float(np.mean(diff_bdl_minus_73_deeq[m_2_4])), 3),
            "bdl_minus_SSB0073_raw_rmse_db": round(float(np.sqrt(np.mean(diff_bdl_minus_73[core_mask] ** 2))), 3),
            "bdl_minus_SSB0073_deeq_rmse_db": round(float(np.sqrt(np.mean(diff_bdl_minus_73_deeq[core_mask] ** 2))), 3),
        },
    }

    return {"speakers": speaker_sigs, "corpus_transfer": transfer}


def main() -> int:
    report = run_corpus_eq_audit()
    out_p = pathlib.Path(__file__).resolve().parent / "corpus_eq_report.json"
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("WROTE", out_p)
    print("TRANSFER SUMMARY:", json.dumps(report["corpus_transfer"]["summary"], indent=2))
    for spk, d in report["speakers"].items():
        print(f"  {spk:7s} ({d['corpus']:13s}, {d['gender']:6s}): {d['voiced_bands_rel_250_2k_db']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
