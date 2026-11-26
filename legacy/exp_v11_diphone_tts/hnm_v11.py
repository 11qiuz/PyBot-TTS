"""Smooth Anti-Metallic HNM Vocoder Synthesis (v11) with Micro-Jitter & Crossover Smoothing."""
from __future__ import annotations

import math

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import sosfiltfilt

from exp_v6_vocoder import vocoder
from exp_v9_clarity_pro.hnm_v9 import BAND_CENTERS_V9, _hp_sos, analyze_v9  # noqa: F401


def _smooth_bap_curve(bap_row: np.ndarray, freqs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Interpolate band aperiodicity smoothly across frequency (cosine-smoothed log-freq crossover)."""
    log_c = np.log2(np.maximum(60.0, centers))
    log_f = np.log2(np.maximum(60.0, freqs))
    ap = np.interp(log_f, log_c, bap_row)
    # 5-bin light spectral smoothing to remove piecewise-linear kinks across band crossovers
    pad = np.pad(ap, (2, 2), mode="edge")
    return np.clip(0.15 * pad[:-4] + 0.20 * pad[1:-3] + 0.30 * pad[2:-2] + 0.20 * pad[3:-1] + 0.15 * pad[4:], 0.0, 1.0)


def synthesize_v11(
    feats: vocoder.VocoderFeatures,
    *,
    seed: int = 72,
    phase_dispersion: bool = True,
    f0_jitter_frac: float = 0.0035,
    amp_shimmer_frac: float = 0.012,
    bap_gt8k: np.ndarray | None = None,
    unvoiced_residual_spans: list[tuple[int, int, np.ndarray]] | None = None,
    residual_highpass_hz: float = 1600.0,
) -> np.ndarray:
    """Synthesize smooth, natural non-metallic HNM waveform from VocoderFeatures.

    Improvements over vocoder.py / hnm_v9.py:
    1. Fractional frame-time interpolation of spectral magnitude & BAP at every pitch pulse (removes 5 ms staircase).
    2. Log-frequency smoothed BAP crossover + smooth >2.2 kHz quadratic phase dispersion + subtle pulse phase randomization
       proportional to high-band aperiodicity (eliminates metallic comb buzz).
    3. Subtle correlated natural glottal micro-jitter (~0.35%) & amplitude shimmer (~1.2%).
    4. Optional click-free >1.6 kHz unvoiced residual overlay strictly on unvoiced spans.
    """
    sr = feats.sr
    hop = feats.hop_samples
    n = feats.n_samples
    n_fft = feats.n_fft
    n_fr = len(feats.f0_hz)
    freqs = rfftfreq(n_fft, 1.0 / sr)
    rng = np.random.default_rng(seed)

    # Smooth causal minimum-phase spectral envelope reconstruction
    ceps = irfft(feats.log_env, n=n_fft, axis=1)
    win_mp = np.zeros(n_fft, dtype=np.float64)
    win_mp[0] = 1.0
    win_mp[1 : n_fft // 2] = 2.0
    win_mp[n_fft // 2] = 1.0
    H_min = np.exp(rfft(ceps * win_mp[None, :], n=n_fft, axis=1))

    if phase_dispersion:
        f_disp = np.maximum(0.0, freqs - 2000.0) / max(1.0, (sr * 0.5 - 2000.0))
        phi_disp = np.exp(-1j * (2.6 * (f_disp ** 1.85) * math.pi))
        H_harm_base = H_min * phi_disp[None, :]
    else:
        H_harm_base = H_min

    # Smooth log-freq BAP interpolation + >2.5 kHz gentle high-frequency harmonic roll-off floor
    ap_full = np.empty((n_fr, len(freqs)), dtype=np.float64)
    sub70 = freqs < 70.0
    sub70_gain = (freqs[sub70] / 70.0) ** 2
    hf_soft_mask = np.clip((freqs - 3000.0) / 7000.0, 0.0, 1.0)
    if bap_gt8k is not None and len(bap_gt8k) == n_fr:
        baps_use = np.concatenate([feats.bap[:, :4], bap_gt8k], axis=1)
        centers_use = BAND_CENTERS_V9
    else:
        baps_use = feats.bap
        centers_use = vocoder.BAND_CENTERS

    for i in range(n_fr):
        ap_row = _smooth_bap_curve(baps_use[i], freqs, centers_use)
        ap_row[sub70] *= sub70_gain
        if feats.voiced[i] and feats.f0_hz[i] > 0.0:
            f_cut = 0.85 * float(feats.f0_hz[i])
            sub_mask = freqs < f_cut
            ap_row[sub_mask] *= np.clip((freqs[sub_mask] / max(1.0, f_cut)) ** 3, 0.0, 1.0)
            # Subtle natural high-frequency aspiration floor above 3.2 kHz prevents sterile laser harmonics
            ap_row = np.maximum(ap_row, 0.10 * (hf_soft_mask ** 1.5))
        ap_full[i] = np.clip(ap_row, 0.0, 1.0)
    harm_gain = np.sqrt(np.maximum(0.0, 1.0 - ap_full * ap_full))

    # Pre-split complex harmonic spectrum with interpolated frame step
    y_harm = np.zeros(n + n_fft, dtype=np.float64)
    w_ang = math.tau * freqs / sr
    pulse_offset = n_fft // 4
    hf_rand_scale = np.clip((freqs - 2800.0) / 5000.0, 0.0, 1.0) * 0.45

    t_cur = 0.0
    jit_state = 0.0
    shim_state = 0.0
    while t_cur < n:
        f_pos = min(float(n_fr - 1), max(0.0, t_cur / hop))
        f0_i = int(math.floor(f_pos))
        f1_i = min(n_fr - 1, f0_i + 1)
        w1 = f_pos - f0_i
        w0 = 1.0 - w1

        fi_round = min(n_fr - 1, max(0, int(round(f_pos))))
        f0 = float(feats.f0_hz[fi_round])
        if feats.voiced[fi_round] and f0 > 0.0:
            # Interpolate smoothly across frames only when neighbor is also voiced
            if feats.voiced[f0_i] and feats.voiced[f1_i] and feats.f0_hz[f0_i] > 0.0 and feats.f0_hz[f1_i] > 0.0:
                f0 = w0 * float(feats.f0_hz[f0_i]) + w1 * float(feats.f0_hz[f1_i])
                spec_env = w0 * H_harm_base[f0_i] + w1 * H_harm_base[f1_i]
                hg = w0 * harm_gain[f0_i] + w1 * harm_gain[f1_i]
            else:
                spec_env = H_harm_base[fi_round]
                hg = harm_gain[fi_round]

            # Correlated AR(1) glottal micro-jitter & shimmer (breaks robotic periodicity cleanly)
            jit_state = 0.35 * jit_state + 0.65 * float(rng.standard_normal()) * f0_jitter_frac
            shim_state = 0.45 * shim_state + 0.55 * float(rng.standard_normal()) * amp_shimmer_frac
            T0_nom = sr / f0
            T0 = max(sr / 460.0, min(sr / 60.0, T0_nom * (1.0 + jit_state)))
            amp_mul = max(0.92, min(1.08, 1.0 + shim_state))

            idx_int = int(math.floor(t_cur))
            frac = t_cur - idx_int
            delay_phasor = np.exp(-1j * w_ang * frac)
            # Subtle per-pulse high-frequency phase micro-dither proportional to high-band aperiodicity
            if phase_dispersion:
                phi_jit = np.exp(1j * (rng.uniform(-math.pi, math.pi, size=len(freqs)) * hf_rand_scale * ap_full[fi_round]))
                spec_h = spec_env * hg * delay_phasor * phi_jit
            else:
                spec_h = spec_env * hg * delay_phasor

            pulse = np.roll(irfft(spec_h, n=n_fft), pulse_offset) * (math.sqrt(T0_nom) * amp_mul)
            start = idx_int - pulse_offset
            if start >= 0 and start + n_fft <= len(y_harm):
                y_harm[start : start + n_fft] += pulse
            elif start < 0 and start + n_fft > 0:
                y_harm[0 : start + n_fft] += pulse[-start:]
            t_cur += T0
        else:
            jit_state = 0.0
            shim_state = 0.0
            t_cur += hop

    # Smooth 5 ms COLA aperiodic noise excitation
    y_ap = np.zeros(n + n_fft, dtype=np.float64)
    norm_ola = np.zeros(n + n_fft, dtype=np.float64)
    win_ola = np.hanning(2 * hop)
    noise_all = rng.standard_normal((n_fr, n_fft))
    noise_specs = rfft(noise_all, axis=1)
    ap_frames = irfft(H_min * ap_full * noise_specs, n=n_fft, axis=1)[:, : 2 * hop]
    for i in range(n_fr):
        s = i * hop - hop
        if s < 0 or s + 2 * hop > len(y_ap):
            continue
        y_ap[s : s + 2 * hop] += ap_frames[i] * win_ola
        norm_ola[s : s + 2 * hop] += win_ola * win_ola

    y = (y_harm + y_ap / np.maximum(np.sqrt(norm_ola), 1e-6))[:n]

    # Optional click-free >1.6 kHz burst/sibilant residual strictly on unvoiced spans
    if unvoiced_residual_spans:
        hp_sos = _hp_sos(sr, residual_highpass_hz)
        fade_n = max(24, min(72, int(round(0.003 * sr))))
        for fr_s, fr_e, raw_w in unvoiced_residual_spans:
            fr_s_c = max(0, min(n_fr, int(fr_s)))
            fr_e_c = max(fr_s_c, min(n_fr, int(fr_e)))
            if fr_e_c <= fr_s_c or np.any(feats.voiced[fr_s_c:fr_e_c]):
                continue
            s0 = fr_s_c * hop
            s1 = min(n, fr_e_c * hop)
            seg_len = s1 - s0
            if seg_len < 72 or len(raw_w) < 72:
                continue
            src = np.asarray(raw_w, dtype=np.float64)
            src_hp = sosfiltfilt(hp_sos, src) if len(src) > 32 else src.copy()
            patch = (
                src_hp[-seg_len:].copy()
                if len(src_hp) >= seg_len
                else np.interp(np.linspace(0.0, 1.0, seg_len), np.linspace(0.0, 1.0, len(src_hp)), src_hp)
            )
            y_seg_hp = sosfiltfilt(hp_sos, y[s0:s1]) if seg_len > 32 else y[s0:s1]
            r_syn = float(np.sqrt(np.mean(y_seg_hp ** 2)))
            r_pat = float(np.sqrt(np.mean(patch ** 2)))
            if r_syn > 1e-6 and r_pat > 1e-6:
                patch *= r_syn / r_pat
            else:
                continue
            y_seg_lf = y[s0:s1] - y_seg_hp
            hd_seg = y_seg_lf + patch
            fn = min(fade_n, seg_len // 3)
            if fn > 0:
                w_in = 0.5 * (1.0 - np.cos(math.pi * np.linspace(0.0, 1.0, fn)))
                w_out = 0.5 * (1.0 + np.cos(math.pi * np.linspace(0.0, 1.0, fn)))
                y[s0 : s0 + fn] = (1.0 - w_in) * y[s0 : s0 + fn] + w_in * hd_seg[:fn]
                y[s0 + fn : s1 - fn] = hd_seg[fn : seg_len - fn]
                y[s1 - fn : s1] = (1.0 - w_out) * hd_seg[seg_len - fn :] + w_out * y[s1 - fn : s1]

    return y
