"""Spectral sharpness, consonant transient, and cepstral/temporal blur diagnostics."""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_eval.metrics import DEFAULT_SR, _mel_filterbank, extract_active_frames, read_wav  # noqa: E402
from exp_v6_eval.fluency import compute_f0_track  # noqa: E402
from exp_v6_vocoder import vocoder  # noqa: E402

SUBBANDS_HZ = ((200.0, 500.0), (500.0, 1200.0), (1200.0, 2400.0), (2400.0, 4000.0))


def _spectral_contrast_db(mag_spec: np.ndarray, freqs: np.ndarray) -> float:
    """Mean octave-subband peak-to-valley contrast (90th - 10th pct in dB) across frames."""
    vals = []
    for lo, hi in SUBBANDS_HZ:
        m = (freqs >= lo) & (freqs <= hi)
        if int(np.sum(m)) < 4:
            continue
        band_db = 20.0 * np.log10(np.maximum(mag_spec[:, m], 1e-10))
        p90 = np.percentile(band_db, 90, axis=1)
        p10 = np.percentile(band_db, 10, axis=1)
        vals.append(float(np.mean(p90 - p10)))
    return float(np.mean(vals)) if vals else 0.0


def compute_clarity_metrics(x: np.ndarray, sr: int = DEFAULT_SR, *, fmin: float = 75.0, fmax: float = 450.0) -> dict:
    """Measure formant prominence, consonant 2-8 kHz transient energy, and temporal modulation."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    frame_len = max(128, int(round(0.030 * sr)))
    hop = max(32, int(round(0.010 * sr)))
    nfft = 2048 if sr <= 24000 else 4096

    frames, keep, _ = extract_active_frames(arr, frame_len=frame_len, hop=hop, db_thresh=30.0)
    track = compute_f0_track(arr, sr=sr, fmin=fmin, fmax=fmax)
    n_common = min(len(keep), len(track["voiced"]))
    keep = keep[:n_common]
    voiced = track["voiced"][:n_common]
    frames = frames[:n_common]

    win = np.hamming(frame_len)[None, :]
    spec = np.abs(rfft((frames - np.mean(frames, axis=1, keepdims=True)) * win, n=nfft, axis=1))
    pow_spec = spec ** 2
    freqs = rfftfreq(nfft, 1.0 / sr)

    # 1. Voiced formant peak-to-valley prominence (200-4000 Hz)
    v_mask = keep & voiced
    u_mask = keep & (~voiced)
    voiced_contrast_db = _spectral_contrast_db(spec[v_mask], freqs) if int(np.sum(v_mask)) >= 3 else None

    # 2. 2-8 kHz consonant transient energy & onset sharpness
    m_lo = (freqs >= 80.0) & (freqs < 1000.0)
    m_2_8 = (freqs >= 2000.0) & (freqs <= min(8000.0, 0.48 * sr))
    e_lo = np.sum(pow_spec[:, m_lo], axis=1) + 1e-18
    e_2_8 = np.sum(pow_spec[:, m_2_8], axis=1) + 1e-18
    e_2_8_db = 10.0 * np.log10(e_2_8)

    act_2_8_db = e_2_8_db[keep]
    transient_span_db = float(np.percentile(act_2_8_db, 95) - np.median(act_2_8_db)) if act_2_8_db.size >= 4 else None
    d_2_8 = np.diff(e_2_8_db)
    act_pair = keep[:-1] & keep[1:]
    pos_bursts = d_2_8[act_pair & (d_2_8 > 0)]
    transient_burst_p95_db = float(np.percentile(pos_bursts, 95)) if pos_bursts.size >= 2 else 0.0

    unvoiced_frac = float(np.mean(~voiced[keep])) if int(np.sum(keep)) > 0 else 0.0
    if int(np.sum(u_mask)) >= 2:
        unvoiced_hf_ratio_db = float(np.median(10.0 * np.log10(e_2_8[u_mask] / e_lo[u_mask])))
    else:
        unvoiced_hf_ratio_db = None

    # 3. Temporal modulation crispness across 40 Mel bands (dynamic range over active frames)
    fb = _mel_filterbank(sr=sr, nfft=nfft, n_mels=40)
    mel_db = 10.0 * np.log10(np.maximum(pow_spec[keep] @ fb.T, 1e-12))
    mel_norm = mel_db - np.mean(mel_db, axis=1, keepdims=True)
    if mel_norm.shape[0] >= 4:
        temporal_mod_std_db = float(np.mean(np.std(mel_norm, axis=0)))
        temporal_mod_iqr_db = float(np.mean(np.percentile(mel_norm, 90, axis=0) - np.percentile(mel_norm, 10, axis=0)))
    else:
        temporal_mod_std_db = None
        temporal_mod_iqr_db = None

    return {
        "voiced_spectral_contrast_db": round(voiced_contrast_db, 4) if voiced_contrast_db is not None else None,
        "unvoiced_frac": round(unvoiced_frac, 4),
        "unvoiced_2_8k_vs_lo_db": round(unvoiced_hf_ratio_db, 4) if unvoiced_hf_ratio_db is not None else None,
        "transient_2_8k_span_db": round(transient_span_db, 4) if transient_span_db is not None else None,
        "transient_2_8k_burst_p95_db": round(transient_burst_p95_db, 4),
        "mel_temporal_std_db": round(temporal_mod_std_db, 4) if temporal_mod_std_db is not None else None,
        "mel_temporal_p10_90_db": round(temporal_mod_iqr_db, 4) if temporal_mod_iqr_db is not None else None,
    }


def evaluate_clarity_wav(path: str | pathlib.Path, *, start_sample: int = 0, stop_sample: int | None = None, fmin: float = 75.0) -> dict:
    x, sr = read_wav(path)
    if stop_sample is not None or start_sample > 0:
        x = x[max(0, start_sample):stop_sample]
    res = compute_clarity_metrics(x, sr=sr, fmin=fmin)
    res["path"] = str(path)
    return res


def diagnose_v7_blur_causes(wav_path: str | pathlib.Path, start_sample: int = 0, stop_sample: int | None = None) -> dict:
    """Controlled ablation on a real utterance: measure contrast & temporal loss from cepstral-128 & K=14."""
    x, sr = read_wav(wav_path)
    if stop_sample is not None or start_sample > 0:
        x = x[max(0, start_sample):stop_sample]
    feats = vocoder.analyze(x, sr=sr)
    freqs = rfftfreq(feats.n_fft, 1.0 / sr)
    mag_full = np.exp(feats.log_env)

    # Cepstral-128 liftering (as in exp_v7_pybot_tts/unit_bank.py)
    ceps = irfft(feats.log_env, axis=1)
    ceps[:, 128:-127] = 0.0
    log_c128 = np.real(rfft(ceps, n=feats.n_fft, axis=1))
    mag_c128 = np.exp(log_c128)

    # K=14 block temporal downsampling (simulating syllable-level K=14 template compression per ~250ms syllable)
    block = 4  # ~20 ms -> ~14 frames per 280ms syllable
    n_blk = max(1, feats.log_env.shape[0] // block)
    trimmed = log_c128[: n_blk * block].reshape(n_blk, block, -1).mean(axis=1)
    log_c128_k14 = np.repeat(trimmed, block, axis=0)
    mag_c128_k14 = np.exp(log_c128_k14)

    v_mask = feats.voiced[: len(mag_full)]
    return {
        "full_513_contrast_db": round(_spectral_contrast_db(mag_full[v_mask], freqs), 4),
        "cep128_contrast_db": round(_spectral_contrast_db(mag_c128[v_mask], freqs), 4),
        "cep128_k14_contrast_db": round(_spectral_contrast_db(mag_c128_k14[v_mask[: len(mag_c128_k14)]], freqs), 4),
        "full_513_temporal_std_db": round(float(np.mean(np.std(20.0 * np.log10(np.maximum(mag_full, 1e-10)), axis=0))), 4),
        "cep128_k14_temporal_std_db": round(float(np.mean(np.std(20.0 * np.log10(np.maximum(mag_c128_k14, 1e-10)), axis=0))), 4),
    }


def main() -> int:
    ext_idx = json.loads((ROOT / "pybot_tts_v2_recorded/extended/index.json").read_text(encoding="utf-8"))
    utt_map = {u["id"]: u for u in ext_idx["utterances"]}
    held_ids = sorted({p.name.split("_")[0] for p in (ROOT / "exp_v7_pybot_tts/samples/heldout").glob("*_v7_same.wav")})

    groups = {"real": [], "v6_crossling_dsp": [], "v7_pybot_tts": []}
    for uid in held_ids:
        u = utt_map[uid]
        groups["real"].append(evaluate_clarity_wav(u["path"], start_sample=u["active_start"], stop_sample=u["active_stop"], fmin=90.0))
        groups["v6_crossling_dsp"].append(evaluate_clarity_wav(ROOT / f"exp_v6_crossling_dsp/samples/matched_heldout/{uid}_v6_same.wav", fmin=90.0))
        groups["v7_pybot_tts"].append(evaluate_clarity_wav(ROOT / f"exp_v7_pybot_tts/samples/heldout/{uid}_v7_same.wav", fmin=90.0))

    # Also check v6_vocoder gate1_resyn on the 2 reference clips
    v6_voc_pair = {
        "SSB05020005_real": evaluate_clarity_wav(ROOT / "exp_v6_vocoder/samples/SSB05020005_real.wav", fmin=90.0),
        "SSB05020005_gate1_resyn": evaluate_clarity_wav(ROOT / "exp_v6_vocoder/samples/SSB05020005_gate1_resyn.wav", fmin=90.0),
        "SSB15930031_real": evaluate_clarity_wav(ROOT / "exp_v6_vocoder/samples/SSB15930031_real.wav", fmin=90.0),
        "SSB15930031_gate1_resyn": evaluate_clarity_wav(ROOT / "exp_v6_vocoder/samples/SSB15930031_gate1_resyn.wav", fmin=90.0),
    }

    blur_ablation = diagnose_v7_blur_causes(
        utt_map["SSB05020122"]["path"],
        start_sample=utt_map["SSB05020122"]["active_start"],
        stop_sample=utt_map["SSB05020122"]["active_stop"],
    )

    summary = {}
    for gname, rows in groups.items():
        keys = [k for k in rows[0] if k != "path"]
        summary[gname] = {
            k: round(float(np.median([r[k] for r in rows if r.get(k) is not None])), 4)
            for k in keys
            if any(r.get(k) is not None for r in rows)
        }

    report = {
        "heldout_26_clarity_medians": summary,
        "v6_vocoder_pair_clarity": v6_voc_pair,
        "controlled_blur_ablation_SSB05020122": blur_ablation,
    }
    out_p = pathlib.Path(__file__).resolve().parent / "clarity_report.json"
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("WROTE", out_p)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
