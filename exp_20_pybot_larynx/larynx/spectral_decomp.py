"""Invertible Source-Filter LSF-22 spectral decomposition for PyBot-Larynx.

Decomposes a 513-bin natural-log vocal-tract+glottal envelope into explicit,
synthesizer-compatible physical controls:
  - frame gain (`gain_c0` / `lpc_gain`)
  - glottal spectral tilt (`tilt_db_oct` over 250..4500 Hz)
  - vocal-tract resonance structure as ordered 22-order Line Spectral Frequencies (`lsf`)
  - compact 64-dim cepstral texture residual (`residual_cep`)
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from exp_20_pybot_larynx import N_BINS, N_CEP, N_FFT, SR
from exp_20_pybot_larynx.larynx.synth import cep_to_log_env, log_env_to_cep

NEPER_TO_DB = 20.0 / math.log(10.0)
_FREQS_513 = rfftfreq(N_FFT, 1.0 / SR)
_TILT_MASK = (_FREQS_513 >= 250.0) & (_FREQS_513 <= 4500.0)
_OCT_FROM_1K = np.log2(np.clip(_FREQS_513, 120.0, 11500.0) / 1000.0)
_HI_TAPER = np.where(
    _FREQS_513 > 8000.0,
    np.maximum(0.0, 1.0 - (_FREQS_513 - 8000.0) / max(1.0, SR * 0.5 - 8000.0)),
    1.0,
)
_TILT_BASIS = _OCT_FROM_1K * _HI_TAPER
_TILT_X = np.log2(_FREQS_513[_TILT_MASK] / 1000.0)
_TILT_X_MEAN = float(np.mean(_TILT_X))
_TILT_X_DEN = float(np.sum((_TILT_X - _TILT_X_MEAN) ** 2))


@dataclass
class SourceFilterFrame:
    """Interpretable physical source-filter decomposition of a single spectral frame."""

    gain_c0: float
    tilt_db_oct: float
    lsf: np.ndarray            # (order,) strictly increasing radians in (0, pi)
    lpc_gain: float
    residual_cep: np.ndarray   # (n_cep,)
    order: int = 22
    n_fft: int = N_FFT
    sr: int = SR


def _estimate_tilt_db_oct(log_env_1d: np.ndarray, freqs: np.ndarray = _FREQS_513) -> float:
    """Linear spectral tilt (dB/oct) over 250..4500 Hz."""
    if len(freqs) == N_BINS:
        y = np.asarray(log_env_1d, dtype=np.float64)[_TILT_MASK] * NEPER_TO_DB
        num = float(np.sum((_TILT_X - _TILT_X_MEAN) * (y - float(np.mean(y)))))
        return num / max(1e-12, _TILT_X_DEN)
    mask = (freqs >= 250.0) & (freqs <= 4500.0)
    x = np.log2(freqs[mask] / 1000.0)
    y = np.asarray(log_env_1d, dtype=np.float64)[mask] * NEPER_TO_DB
    xm = float(np.mean(x))
    den = float(np.sum((x - xm) ** 2))
    return float(np.sum((x - xm) * (y - float(np.mean(y))))) / max(1e-12, den)


def _tilt_curve(tilt_db_oct: float, freqs: np.ndarray = _FREQS_513, sr: int = SR) -> np.ndarray:
    if len(freqs) == N_BINS and int(sr) == SR:
        return (float(tilt_db_oct) / NEPER_TO_DB) * _TILT_BASIS
    oct_1k = np.log2(np.clip(freqs, 120.0, 11500.0) / 1000.0)
    hi_taper = np.where(freqs > 8000.0, np.maximum(0.0, 1.0 - (freqs - 8000.0) / max(1.0, sr * 0.5 - 8000.0)), 1.0)
    return (float(tilt_db_oct) / NEPER_TO_DB) * oct_1k * hi_taper


def log_env_to_lpc(
    log_env_1d: np.ndarray,
    order: int = 22,
    *,
    n_fft: int = N_FFT,
) -> tuple[np.ndarray, float]:
    """Fit all-pole LPC filter `A(z) = 1 + sum a[k] z^{-(k+1)}` via spectral autocorrelation."""
    p = int(order)
    env = np.clip(np.asarray(log_env_1d, dtype=np.float64).ravel(), -25.0, 25.0)
    pwr = np.exp(2.0 * env)
    ac = irfft(pwr, n=n_fft)[: p + 1]
    if ac[0] <= 1e-18 or not np.isfinite(ac[0]):
        return np.zeros(p, dtype=np.float64), 1e-6

    # Build symmetric Toeplitz autocorrelation matrix with tiny stabilization jitter
    idx = np.arange(p)
    R = ac[np.abs(idx[:, None] - idx[None, :])]
    R = R.copy()
    R[idx, idx] += 1e-7 * ac[0]
    a = np.linalg.solve(R, -ac[1 : p + 1])
    err_p = max(1e-18, float(ac[0] + np.dot(a, ac[1 : p + 1])))
    return a.astype(np.float64), math.sqrt(err_p)


def lpc_to_log_env(
    a: np.ndarray,
    gain: float,
    *,
    n_fft: int = N_FFT,
) -> np.ndarray:
    """Evaluate 513-bin natural-log magnitude spectrum of `gain / A(z)`."""
    a_full = np.concatenate(([1.0], np.asarray(a, dtype=np.float64).ravel()))
    H = rfft(a_full, n=n_fft)
    return np.log(max(1e-9, float(gain)) / np.maximum(np.abs(H), 1e-9))


def lpc_to_lsf(a: np.ndarray) -> np.ndarray:
    """Convert even-order LPC coefficients `a` to strictly ordered Line Spectral Frequencies in `(0, pi)`."""
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    p = len(a_arr)
    if p % 2 != 0:
        raise ValueError(f"lpc_to_lsf expects even order, got {p}")
    half = p // 2
    a1 = np.concatenate(([1.0], a_arr, [0.0]))
    a2 = a1[::-1]
    P = a1 + a2
    Q = a1 - a2
    P_red, _ = np.polydiv(P, np.array([1.0, 1.0], dtype=np.float64))
    Q_red, _ = np.polydiv(Q, np.array([1.0, -1.0], dtype=np.float64))

    rp = np.roots(P_red)
    rq = np.roots(Q_red)
    ang_p = np.sort(np.angle(rp[np.imag(rp) > 1e-9]))
    ang_q = np.sort(np.angle(rq[np.imag(rq) > 1e-9]))

    if len(ang_p) != half or len(ang_q) != half:
        # Clean monotone fallback if filter had degenerate pole collapse
        return np.linspace(0.05, math.pi - 0.05, p, dtype=np.float64)

    lsf = np.empty(p, dtype=np.float64)
    lsf[0::2] = ang_p
    lsf[1::2] = ang_q
    lsf = np.sort(np.clip(lsf, 1e-4, math.pi - 1e-4))
    for i in range(1, p):
        if lsf[i] <= lsf[i - 1] + 1e-5:
            lsf[i] = min(math.pi - 1e-4, lsf[i - 1] + 1e-5)
    return lsf


def lsf_to_lpc(lsf: np.ndarray) -> np.ndarray:
    """Reconstruct guaranteed-stable even-order LPC coefficients from ordered LSF angles."""
    w = np.sort(np.clip(np.asarray(lsf, dtype=np.float64).ravel(), 1e-4, math.pi - 1e-4))
    p = len(w)
    if p % 2 != 0:
        raise ValueError(f"lsf_to_lpc expects even order, got {p}")

    P = np.array([1.0, 1.0], dtype=np.float64)
    for wp in w[0::2]:
        P = np.convolve(P, np.array([1.0, -2.0 * math.cos(float(wp)), 1.0], dtype=np.float64))

    Q = np.array([1.0, -1.0], dtype=np.float64)
    for wq in w[1::2]:
        Q = np.convolve(Q, np.array([1.0, -2.0 * math.cos(float(wq)), 1.0], dtype=np.float64))

    A = 0.5 * (P + Q)
    return A[1 : p + 1].astype(np.float64)


def lsf_to_formants(
    lsf: np.ndarray,
    *,
    sr: int = SR,
    f_min: float = 180.0,
    f_max: float = 5500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sorted vocal-tract formant frequencies (Hz) and 3-dB bandwidths (Hz) from LSF."""
    a = lsf_to_lpc(lsf)
    roots = np.roots(np.concatenate(([1.0], a)))
    imag_pos = np.imag(roots) > 1e-7
    if not np.any(imag_pos):
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    r_pos = roots[imag_pos]
    hz = np.angle(r_pos) * (sr / math.tau)
    mag = np.clip(np.abs(r_pos), 1e-6, 0.9995)
    bw = -(sr / math.pi) * np.log(mag)
    keep = (hz >= f_min) & (hz <= f_max) & (bw < 900.0)
    if not np.any(keep):
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    order_idx = np.argsort(hz[keep])
    return hz[keep][order_idx], bw[keep][order_idx]


def decompose_frame_source_filter(
    log_env_1d: np.ndarray,
    order: int = 22,
    *,
    sr: int = SR,
    n_fft: int = N_FFT,
    n_cep: int = N_CEP,
) -> SourceFilterFrame:
    """Decompose 1 spectral envelope frame into Source (gain, tilt) + Filter (LSF) + Texture residual."""
    env = np.asarray(log_env_1d, dtype=np.float64).ravel()
    freqs = _FREQS_513 if (n_fft == N_FFT and sr == SR) else rfftfreq(n_fft, 1.0 / sr)
    tilt_db_oct = _estimate_tilt_db_oct(env, freqs=freqs)
    tilt_c = _tilt_curve(tilt_db_oct, freqs=freqs, sr=sr)

    # Fit vocal-tract LSF on tilt-normalized envelope so glottal roll-off and resonances decouple
    flat_env = env - tilt_c
    a, lpc_gain = log_env_to_lpc(flat_env, order=order, n_fft=n_fft)
    lsf = lpc_to_lsf(a)
    a_clean = lsf_to_lpc(lsf)
    sf_env = lpc_to_log_env(a_clean, lpc_gain, n_fft=n_fft) + tilt_c

    res_cep = log_env_to_cep(env - sf_env, n_cep=n_cep)
    gain_c0 = float(log_env_to_cep(env, n_cep=1)[0])
    return SourceFilterFrame(
        gain_c0=gain_c0,
        tilt_db_oct=float(tilt_db_oct),
        lsf=lsf,
        lpc_gain=float(lpc_gain),
        residual_cep=res_cep,
        order=int(order),
        n_fft=int(n_fft),
        sr=int(sr),
    )


def reconstruct_frame_source_filter(frame: SourceFilterFrame) -> np.ndarray:
    """Invert `decompose_frame_source_filter` back to a 513-bin natural-log amplitude envelope."""
    freqs = _FREQS_513 if (frame.n_fft == N_FFT and frame.sr == SR) else rfftfreq(frame.n_fft, 1.0 / frame.sr)
    a = lsf_to_lpc(frame.lsf)
    sf_env = lpc_to_log_env(a, frame.lpc_gain, n_fft=frame.n_fft) + _tilt_curve(
        frame.tilt_db_oct, freqs=freqs, sr=frame.sr
    )
    res_env = cep_to_log_env(frame.residual_cep, n_bins=frame.n_fft // 2 + 1)
    return sf_env + res_env


def compute_pole_bandwidth_stats(
    log_env: np.ndarray,
    voiced_mask: np.ndarray,
    *,
    order: int = 22,
    stride: int = 4,
    sr: int = SR,
    n_fft: int = N_FFT,
) -> dict[str, float]:
    """Measure median top-4 LPC pole radius and effective formant bandwidth (Hz) in 200..4500 Hz."""
    rads: list[float] = []
    idx = np.where(np.asarray(voiced_mask, dtype=bool))[0][:: max(1, int(stride))]
    for t in idx:
        try:
            a, _ = log_env_to_lpc(log_env[t], order=order, n_fft=n_fft)
            roots = np.roots(np.concatenate(([1.0], a)))
            hz = np.angle(roots) * (sr / math.tau)
            sel = (hz > 200.0) & (hz < 4500.0)
            if np.any(sel):
                r_top = np.sort(np.abs(roots[sel]))[::-1][:4]
                rads.extend(r_top.tolist())
        except Exception:
            continue
    if not rads:
        return {"pole_radius_median": 0.0, "eff_bw_hz": 0.0}
    med_r = float(np.median(rads))
    bw = -(sr / math.pi) * math.log(max(1e-6, min(0.9999, med_r)))
    return {
        "pole_radius_median": round(med_r, 5),
        "eff_bw_hz": round(bw, 1),
    }
