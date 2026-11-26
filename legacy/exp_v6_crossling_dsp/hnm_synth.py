"""Zero-weight Harmonic-Plus-Noise (HNM) & Pole-Zero Spectral Envelope Synthesizer.

Synthesizes 24 kHz mono waveforms from :class:`arbiter.FrameTrack` and
:class:`speaker_profile.SpeakerProfile` using pure numpy/scipy:
- Full-band harmonics up to 11.5 kHz (eliminates v4's 26-harmonic 4-8 kHz cliff)
- Bounded log-dB pole-zero vocal tract envelope (prevents v5's all-pole cascade
  spectral contrast overshoot and includes nasal/lateral/rhotic anti-resonance zeros)
- Direct post-envelope HNR scaling (prevents v5's +3 dB cascade ringing HNR overshoot)
- Multi-pole FFT-shaped turbulence for fricatives, affricates, and stop bursts
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import wave

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from exp_v6_crossling_dsp import SR
from exp_v6_crossling_dsp.arbiter import FrameTrack, HOP_SAMPLES
from exp_v6_crossling_dsp.speaker_profile import EQ_BANDS_HZ, SpeakerProfile

WIN_SAMPLES = HOP_SAMPLES * 2   # 240 samples (10 ms) for 50% Hann OLA
N_FFT = 256
_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_FREQS_SAFE = np.maximum(_FREQS, 80.0)
_LOG2_FREQS = np.log2(_FREQS_SAFE)
_LOG2_EQ = np.log2(np.asarray(EQ_BANDS_HZ, dtype=np.float64))
_HANN_OLA = np.hanning(WIN_SAMPLES + 1)[:-1]


def _resonator_db(freqs_hz: np.ndarray, f_c: float, bw: float) -> np.ndarray:
    """2nd-order vocal-tract resonator log-magnitude (0 dB at DC, bounded peak)."""
    f2 = freqs_hz * freqs_hz
    fc2 = f_c * f_c
    denom = (fc2 - f2) ** 2 + (freqs_hz * max(bw, 55.0)) ** 2
    mag2 = (fc2 * fc2) / np.maximum(denom, 1e-12)
    return np.clip(10.0 * np.log10(np.maximum(mag2, 1e-6)), -22.0, 18.0)


def _vocal_tract_db(
    freqs_hz: np.ndarray,
    formants: np.ndarray,
    bandwidths: np.ndarray,
    zero_f: float,
    zero_bw: float,
    zero_depth: float,
    profile: SpeakerProfile,
) -> np.ndarray:
    """Pole-zero vocal tract transfer (3 resonators + F4 + nasal/liquid zero + tilt)."""
    f1, f2, f3 = float(formants[0]), float(formants[1]), float(formants[2])
    bw1, bw2, bw3 = float(bandwidths[0]), float(bandwidths[1]), float(bandwidths[2])
    f4 = 3850.0 * profile.vtln_scale

    # Sum 2nd-order resonator transfer functions in dB with Fant higher-pole compensation
    # (+6 dB/oct lip radiation + higher-pole lift offsets intrinsic cascade roll-off above F3)
    env = (
        _resonator_db(freqs_hz, f1, max(95.0, bw1))
        + _resonator_db(freqs_hz, f2, max(135.0, bw2))
        + _resonator_db(freqs_hz, f3, max(200.0, bw3))
        + 0.55 * _resonator_db(freqs_hz, f4, 320.0)
    )
    # Higher-pole + lip-radiation compensation only lifts above ~1.8 kHz where F2/F3 roll-off stacks.
    log2f = np.log2(np.maximum(freqs_hz, 80.0))
    fant_comp = 5.5 * np.log2(1.0 + freqs_hz / 1800.0)

    if zero_f > 0.0 and zero_depth > 0.0:
        env = env - zero_depth / (1.0 + ((freqs_hz - zero_f) / max(75.0, zero_bw)) ** 2)

    # Speaker LTAS tilt + 8-band residual EQ
    tilt_term = profile.ltas_tilt_db_per_oct * (log2f - math.log2(500.0))
    eq_term = 0.5 * np.interp(log2f, _LOG2_EQ, np.asarray(profile.residual_eq_db, dtype=np.float64))
    return np.clip(env + fant_comp + tilt_term + eq_term, -40.0, 28.0)


def _fricative_db(
    freqs_hz: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    profile: SpeakerProfile,
) -> np.ndarray:
    """Multi-band turbulence spectral envelope (dB)."""
    env = np.full_like(freqs_hz, -14.0, dtype=np.float64)
    if p1[0] > 0.0:
        env = env + p1[2] / (1.0 + ((freqs_hz - p1[0]) / max(120.0, 0.75 * p1[1])) ** 2)
    if p2[0] > 0.0:
        env = env + p2[2] / (1.0 + ((freqs_hz - p2[0]) / max(150.0, 0.75 * p2[1])) ** 2)
    # Gentle low-frequency roll-off below 500 Hz for clean sibilants/obstruents
    env = env - 12.0 / (1.0 + (freqs_hz / 420.0) ** 2)
    log2f = np.log2(np.maximum(freqs_hz, 80.0))
    env = env + 0.35 * (profile.ltas_tilt_db_per_oct + 3.5) * (log2f - math.log2(2500.0))
    return env


def render_track(track: FrameTrack, profile: SpeakerProfile, *, seed: int = 72) -> np.ndarray:
    """Render a :class:`FrameTrack` to a 24 kHz float64 waveform in ``[-1, 1]``."""
    rng = np.random.default_rng(seed)
    n_frames = track.n_frames
    n_samples = (n_frames - 1) * HOP_SAMPLES + WIN_SAMPLES
    out_harm = np.zeros(n_samples, dtype=np.float64)
    out_noise = np.zeros(n_samples, dtype=np.float64)

    # Smoothly interpolate F0 across voiced frame centers (avoids staircase pitch jumps)
    voiced_mask = track.f0_hz > 0.0
    if np.any(voiced_mask):
        centers = np.arange(n_frames, dtype=np.float64) * HOP_SAMPLES + 0.5 * WIN_SAMPLES
        f0_filled = np.interp(centers, centers[voiced_mask], track.f0_hz[voiced_mask])
        sample_idx = np.arange(n_samples, dtype=np.float64)
        f0_inst = np.interp(sample_idx, centers, f0_filled)
    else:
        f0_inst = np.zeros(n_samples, dtype=np.float64)

    # Subtle physiological micro-flutter (~0.25%) prevents static buzz on long vowels
    t_sec = np.arange(n_samples, dtype=np.float64) / SR
    f0_jittered = f0_inst * (1.0 + 0.0025 * np.sin(2.0 * math.pi * 4.7 * t_sec))
    dphi = np.where(f0_inst > 0.0, 2.0 * math.pi * f0_jittered / SR, 0.0)
    phase_cont = np.cumsum(dphi)

    # Target post-envelope aspiration-to-harmonic RMS ratio from speaker HNR.
    asp_ratio = 0.55 * (10.0 ** (-profile.hnr_db / 20.0))
    asp_ratio = float(max(0.025, min(0.45, asp_ratio)))

    # Pre-generate deterministic white noise pool for fast FFT spectral shaping
    noise_white = rng.standard_normal((n_frames, WIN_SAMPLES))
    noise_fft_in = np.fft.rfft(noise_white * _HANN_OLA[None, :], n=N_FFT, axis=1)

    for idx in range(n_frames):
        s0 = idx * HOP_SAMPLES
        s1 = s0 + WIN_SAMPLES
        f0 = float(track.f0_hz[idx])
        v_gain = float(track.voiced_gain[idx])
        n_gain = float(track.noise_gain[idx])
        vbar = float(track.voicing_bar[idx])

        if v_gain <= 1e-6 and n_gain <= 1e-6:
            continue

        # 1. Harmonic + tract-shaped aspiration noise
        if v_gain > 1e-6 and f0 > 40.0:
            max_h = max(4, min(140, int(11500.0 / f0)))
            hs = np.arange(1, max_h + 1, dtype=np.float64)
            h_freqs = hs * f0
            h_db = _vocal_tract_db(
                h_freqs,
                track.formants[idx],
                track.bandwidths[idx],
                float(track.zero_freq[idx]),
                float(track.zero_bw[idx]),
                float(track.zero_depth_db[idx]),
                profile,
            )
            if vbar > 0.0:
                # Voiced obstruent voicing bar: strong low-pass attenuation above 750 Hz
                h_amps = (10.0 ** (h_db / 20.0)) / (1.0 + (h_freqs / 750.0) ** 4)
            else:
                h_amps = 10.0 ** (h_db / 20.0)

            # Normalize harmonic frame energy before applying rule gain
            norm_h = math.sqrt(float(np.sum(h_amps * h_amps)) + 1e-12)
            h_amps = h_amps / norm_h

            ph_block = phase_cont[s0:s1]
            # Smooth phase relation preserves periodicity across overlapping OLA frames
            disp = 0.25 * hs
            # Vectorized harmonic sum: (max_h, WIN_SAMPLES)
            arg = hs[:, None] * ph_block[None, :] + disp[:, None]
            y_h = np.dot(h_amps, np.sin(arg)) * math.sqrt(2.0)

            # Scale by OLA Hann window (50% overlap Hann sums to 1.0)
            out_harm[s0:s1] += v_gain * 0.22 * y_h * _HANN_OLA

            # Tract-shaped aspiration noise (only for voiced sonorants, not stop voicing bar)
            if vbar <= 0.0:
                asp_db = _vocal_tract_db(
                    _FREQS_SAFE,
                    track.formants[idx],
                    track.bandwidths[idx],
                    float(track.zero_freq[idx]),
                    float(track.zero_bw[idx]),
                    float(track.zero_depth_db[idx]),
                    profile,
                )
                asp_mag = 10.0 ** (asp_db / 20.0)
                asp_mag[0] = 0.0
                asp_time = irfft(noise_fft_in[idx] * asp_mag, n=N_FFT)[:WIN_SAMPLES]
                asp_rms = float(np.sqrt(np.mean(asp_time * asp_time)) + 1e-12)
                y_h_rms = float(np.sqrt(np.mean(y_h * y_h)) + 1e-12)
                asp_scaled = asp_time * (y_h_rms * asp_ratio / asp_rms)
                out_noise[s0:s1] += v_gain * 0.22 * asp_scaled * _HANN_OLA

        # 2. Unvoiced / obstruent shaped turbulence
        if n_gain > 1e-6:
            f_db = _fricative_db(_FREQS_SAFE, track.fric_p1[idx], track.fric_p2[idx], profile)
            f_mag = 10.0 ** (f_db / 20.0)
            f_mag[0] = 0.0
            turb_time = irfft(noise_fft_in[idx] * f_mag, n=N_FFT)[:WIN_SAMPLES]
            turb_rms = float(np.sqrt(np.mean(turb_time * turb_time)) + 1e-12)
            # Match realistic obstruent-to-vowel RMS ratio (~12-16 dB below vowel peaks)
            turb_scaled = turb_time / turb_rms
            out_noise[s0:s1] += n_gain * 0.065 * turb_scaled * _HANN_OLA

    y = out_harm + out_noise
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.90:
        y = y * (0.90 / peak)
    return y


def write_wav(path: str | Path, samples: np.ndarray, sr: int = SR, peak_norm: float = 0.90) -> Path:
    """Write 24 kHz 16-bit mono PCM WAV."""
    arr = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    factor = min(peak_norm / max(peak, 1e-8), 1.0) if peak_norm else 1.0
    pcm = np.round(np.clip(arr * factor, -0.999, 0.999) * 32767.0).astype("<i2")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as wf:
        wf.setparams((1, 2, int(sr), 0, "NONE", "not compressed"))
        wf.writeframes(pcm.tobytes())
    return p
