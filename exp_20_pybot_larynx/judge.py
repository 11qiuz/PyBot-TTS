"""Multi-metric objective judge for PyBot-Larynx (matching 19b & v15 criteria)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.fft import rfft

from exp_20_pybot_larynx import HOP, SR
from exp_v5_eval.metrics import check_wav_sanity, compute_spectral_distances
from exp_v9_eval.v9_metrics import compute_v9_extended_metrics
from exp_v15_bznsyp_tts.eval_v15 import _dtw_align_path
from exp_v15_bznsyp_tts.index_bznsyp_v15 import _fast_mfcc_and_f0_24k

_HNR_WIN = 480
_HNR_NFFT = 2048
_HNR_NBINS = _HNR_NFFT // 2 + 1
_HNR_HANN = np.hanning(_HNR_WIN)


def compute_voiced_hnr_db(
    x: np.ndarray,
    *,
    sr: int = SR,
    f0_hz: np.ndarray | None = None,
    voiced: np.ndarray | None = None,
) -> float | None:
    """Compute 19b-style harmonic-band HNR (dB) over voiced frames (F0 > 50 Hz)."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if len(arr) < _HNR_WIN:
        return None
    if f0_hz is None or voiced is None:
        ac = _fast_mfcc_and_f0_24k(arr)
        f0_hz = ac["f0_hz"]
        voiced = ac["voiced"]

    nf = len(f0_hz)
    idx = np.arange(nf)[:, None] * HOP + np.arange(-(_HNR_WIN // 2), _HNR_WIN // 2)[None, :]
    frames = arr[np.clip(idx, 0, len(arr) - 1)]
    pwr = np.abs(rfft(frames * _HNR_HANN[None, :], n=_HNR_NFFT, axis=1)) ** 2

    vals: list[float] = []
    for t in np.where(voiced & (f0_hz > 50.0))[0]:
        f = float(f0_hz[t])
        hm = np.zeros(_HNR_NBINS, dtype=bool)
        k = 1
        while k * f <= 11000.0:
            c = k * f * _HNR_NFFT / sr
            w = max(1, round(0.35 * f * _HNR_NFFT / sr))
            a = max(0, int(round(c)) - w)
            b = min(_HNR_NBINS - 1, int(round(c)) + w)
            hm[a : b + 1] = True
            k += 1
        eh = float(np.sum(pwr[t, hm]))
        ei = float(np.sum(pwr[t, ~hm]))
        vals.append(10.0 * math.log10((eh + 1e-18) / (ei + 1e-18)))

    return float(np.median(vals)) if vals else None


def evaluate_utterance(
    y: np.ndarray,
    ref_span: np.ndarray,
    *,
    rtf: float,
    sr: int = SR,
    ref_fast: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate synthesized waveform against reference span across spectrum, F0, HNR, clarity, clicks, speed."""
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    r_arr = np.asarray(ref_span, dtype=np.float64).ravel()

    sanity = check_wav_sanity(y_arr, sr, expected_sr=sr)
    sd = compute_spectral_distances(r_arr, y_arr, sr=sr, n_mels=40, n_ceps=13) or {}

    fy = _fast_mfcc_and_f0_24k(y_arr)
    fx = ref_fast if ref_fast is not None else _fast_mfcc_and_f0_24k(r_arr)
    f0_mae_st: float | None = None
    if len(fy["mfcc16"]) > 0 and len(fx["mfcc16"]) > 0:
        iy, ix = _dtw_align_path(fy["mfcc16"], fx["mfcc16"])
        both = fy["voiced"][iy] & fx["voiced"][ix] & (fy["f0_hz"][iy] > 50.0) & (fx["f0_hz"][ix] > 50.0)
        if int(np.sum(both)) > 2:
            f0_mae_st = float(np.mean(np.abs(12.0 * np.log2(fy["f0_hz"][iy][both] / fx["f0_hz"][ix][both]))))

    hnr_db = compute_voiced_hnr_db(y_arr, sr=sr, f0_hz=fy["f0_hz"], voiced=fy["voiced"])
    v9 = compute_v9_extended_metrics(y_arr, sr=sr, fmin=75.0, fmax=450.0)
    clarity = v9.get("clarity", {})

    def _r4(v: Any) -> float | None:
        return round(float(v), 4) if v is not None and math.isfinite(float(v)) else None

    return {
        "sanity_ok": bool(sanity.get("ok", False)),
        "mel_lsd_db": _r4(sd.get("mel_lsd_db")),
        "mcd_db": _r4(sd.get("mcd_db")),
        "mcd_cmn_db": _r4(sd.get("mcd_cmn_db")),
        "f0_mae_st": _r4(f0_mae_st),
        "hnr_voiced_db": _r4(hnr_db),
        "uv_2_8k_vs_lo_db": _r4(clarity.get("unvoiced_2_8k_vs_lo_db")),
        "voiced_contrast_db": _r4(clarity.get("voiced_spectral_contrast_db")),
        "clicks_gt8": round(float(v9.get("click_outlier_rate_gt8", 0.0)), 6),
        "rtf": round(float(rtf), 4),
    }


def summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute medians, means, and sanity pass counts over evaluated utterances."""
    keys = (
        "mel_lsd_db",
        "mcd_db",
        "mcd_cmn_db",
        "f0_mae_st",
        "hnr_voiced_db",
        "uv_2_8k_vs_lo_db",
        "voiced_contrast_db",
        "clicks_gt8",
        "rtf",
    )
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None and math.isfinite(float(r[k]))]
        if vals:
            medians[k] = round(float(np.median(vals)), 4 if k != "clicks_gt8" else 6)
            means[k] = round(float(np.mean(vals)), 4 if k != "clicks_gt8" else 6)

    return {
        "n": len(rows),
        "sanity_ok": sum(1 for r in rows if r.get("sanity_ok")),
        "medians": medians,
        "means": means,
    }
