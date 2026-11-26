"""Pure physical non-verbal vocalization LarynxControl generators (0.0 MB audio samples)."""
from __future__ import annotations

import math

import numpy as np
from scipy.fft import rfftfreq

from exp_20_pybot_larynx import HOP, N_BAP, N_BINS, N_FFT, SR
from exp_20_pybot_larynx.larynx.synth import LarynxControl

_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_NEPER_TO_DB = 20.0 / math.log(10.0)
_DB_TO_NEPER = 1.0 / _NEPER_TO_DB


def _formant_tract_db(freqs_hz: np.ndarray, formants: tuple[float, float, float], bws: tuple[float, float, float]) -> np.ndarray:
    """Smooth all-pole formant vocal-tract curve in dB matching Stage 7 BZNSYP scale."""
    env = np.zeros_like(freqs_hz, dtype=np.float64)
    for fc, bw in zip(formants, bws):
        env += 10.0 * np.log10(1.0 / ((1.0 - (freqs_hz / max(80.0, fc)) ** 2) ** 2 + (freqs_hz * bw / (max(80.0, fc) ** 2)) ** 2 + 1e-6))
    tilt = -4.2 * np.log2(np.maximum(freqs_hz, 120.0) / 800.0)
    return env + tilt


def build_pause_control(duration_ms: float) -> LarynxControl:
    """Create a clean silence LarynxControl of requested duration."""
    n_fr = max(1, int(round(max(20.0, float(duration_ms)) / 5.0)))
    return LarynxControl(
        f0_hz=np.zeros(n_fr, dtype=np.float64),
        voiced=np.zeros(n_fr, dtype=bool),
        log_env=np.full((n_fr, N_BINS), -16.0, dtype=np.float64),
        bap=np.full((n_fr, N_BAP), 0.99, dtype=np.float64),
        glottal_tilt_db_per_oct=np.zeros(n_fr, dtype=np.float64),
        jitter_frac=np.zeros(n_fr, dtype=np.float64),
        shimmer_frac=np.zeros(n_fr, dtype=np.float64),
        n_samples=n_fr * HOP,
        rd=np.ones(n_fr, dtype=np.float64),
    )


def build_breath_control(
    base_f0_hz: float = 240.0,
    *,
    duration_ms: float = 190.0,
    intensity: float = 0.75,
    is_gasp: bool = False,
) -> LarynxControl:
    """Synthesize pure physical turbulent inhalation / gasp noise shaped by open vocal tract."""
    n_fr = max(8, int(round(max(80.0, min(600.0, float(duration_ms))) / 5.0)))
    inten = max(0.25, min(1.3, float(intensity)))
    t_norm = np.linspace(0.0, 1.0, n_fr)

    if is_gasp:
        # Sharp gasp attack peaking quickly around t=0.25 with brighter upper-tract formants
        amp_curve = np.where(t_norm < 0.25, t_norm / 0.25, np.exp(-2.8 * (t_norm - 0.25)))
        tract_db = _formant_tract_db(_FREQS, (720.0, 1450.0, 2950.0), (160.0, 220.0, 320.0))
        level_offset_db = -31.0 + 3.2 * inten
    else:
        amp_curve = np.sin(math.pi * t_norm) ** 1.15
        tract_db = _formant_tract_db(_FREQS, (640.0, 1380.0, 2750.0), (210.0, 260.0, 380.0))
        level_offset_db = -35.5 + 2.8 * inten

    # Gentle high-pass so inhalation sounds airy rather than sub-bass rumble
    hp_db = -14.0 * 0.5 * (1.0 - np.tanh((_FREQS - 420.0) / 140.0))
    frame_gain_db = 20.0 * np.log10(np.maximum(0.04, amp_curve)) + level_offset_db
    log_env = np.clip((tract_db + hp_db)[None, :] + frame_gain_db[:, None], -65.0, 20.0) * _DB_TO_NEPER
    log_env = np.clip(log_env, -16.0, 4.0)

    return LarynxControl(
        f0_hz=np.zeros(n_fr, dtype=np.float64),
        voiced=np.zeros(n_fr, dtype=bool),
        log_env=log_env,
        bap=np.full((n_fr, N_BAP), 0.99, dtype=np.float64),
        glottal_tilt_db_per_oct=np.zeros(n_fr, dtype=np.float64),
        jitter_frac=np.zeros(n_fr, dtype=np.float64),
        shimmer_frac=np.zeros(n_fr, dtype=np.float64),
        n_samples=n_fr * HOP,
        rd=np.full(n_fr, 1.8, dtype=np.float64),
    )


def build_sigh_control(
    base_f0_hz: float = 240.0,
    *,
    duration_ms: float = 450.0,
    intensity: float = 0.80,
) -> LarynxControl:
    """Synthesize an expressive falling breathy-voiced onset dissolving into warm exhalation."""
    n_fr = max(24, int(round(max(200.0, min(900.0, float(duration_ms))) / 5.0)))
    inten = max(0.3, min(1.3, float(intensity)))
    f0_ref = max(75.0, min(420.0, float(base_f0_hz)))
    t_norm = np.linspace(0.0, 1.0, n_fr)

    voiced_end = int(round(0.38 * n_fr))
    voiced = np.zeros(n_fr, dtype=bool)
    voiced[2:voiced_end] = True
    f0_hz = np.zeros(n_fr, dtype=np.float64)
    if voiced_end > 2:
        st_drop = np.linspace(0.6, -4.6, voiced_end - 2)
        f0_hz[2:voiced_end] = np.clip(f0_ref * (2.0 ** (st_drop / 12.0)), 60.0, 420.0)

    tract_db = _formant_tract_db(_FREQS, (620.0, 1280.0, 2600.0), (180.0, 240.0, 360.0))
    amp_env = np.sin(math.pi * np.clip(t_norm / 0.95, 0.0, 1.0)) ** 0.85 * np.exp(-1.1 * t_norm)
    level_db = -30.5 + 2.4 * inten + 20.0 * np.log10(np.maximum(0.06, amp_env))
    log_env = np.clip((tract_db[None, :] + level_db[:, None]) * _DB_TO_NEPER, -16.0, 4.0)

    bap = np.full((n_fr, N_BAP), 0.96, dtype=np.float64)
    bap[voiced] = np.linspace(0.68, 0.94, int(np.sum(voiced)))[:, None]

    return LarynxControl(
        f0_hz=f0_hz,
        voiced=voiced,
        log_env=log_env,
        bap=np.clip(bap, 0.05, 0.995),
        glottal_tilt_db_per_oct=np.full(n_fr, -2.2, dtype=np.float64),
        jitter_frac=np.full(n_fr, 0.012, dtype=np.float64),
        shimmer_frac=np.full(n_fr, 0.045, dtype=np.float64),
        n_samples=n_fr * HOP,
        rd=np.full(n_fr, 1.95, dtype=np.float64),
    )


def build_laugh_control(
    base_f0_hz: float = 240.0,
    *,
    duration_ms: float = 500.0,
    intensity: float = 0.80,
    soft: bool = False,
) -> LarynxControl:
    """Synthesize multi-pulse rhythmic voiced-breathy laughter / chuckle cycles."""
    n_pulses = 3 if soft else 4
    pulse_ms = max(95.0, float(duration_ms) / n_pulses)
    fr_per_p = max(16, int(round(pulse_ms / 5.0)))
    n_fr = n_pulses * fr_per_p
    inten = max(0.3, min(1.3, float(intensity)))
    f0_ref = max(80.0, min(420.0, float(base_f0_hz)))

    f0_hz = np.zeros(n_fr, dtype=np.float64)
    voiced = np.zeros(n_fr, dtype=bool)
    log_env = np.full((n_fr, N_BINS), -16.0, dtype=np.float64)
    bap = np.full((n_fr, N_BAP), 0.96, dtype=np.float64)
    rd = np.full(n_fr, 1.35 if soft else 1.10, dtype=np.float64)

    # Open smiling vowel tract for "ha/he" pulse body & airy "h" burst
    open_db = _formant_tract_db(_FREQS, (740.0, 1520.0, 2720.0), (140.0, 180.0, 280.0))
    breath_db = _formant_tract_db(_FREQS, (680.0, 1450.0, 2800.0), (220.0, 280.0, 380.0))

    base_st = 2.8 if not soft else 1.4
    base_lvl = (-28.5 if not soft else -31.5) + 2.2 * inten

    for p_i in range(n_pulses):
        a = p_i * fr_per_p
        b = a + fr_per_p
        n_h = max(3, int(round(0.30 * fr_per_p)))
        n_v = max(5, int(round(0.45 * fr_per_p)))
        v0 = a + n_h
        v1 = min(b, v0 + n_v)

        decay_db = -1.6 * p_i
        st_pulse = base_st - 1.1 * p_i
        for t in range(a, b):
            if t < v0:
                u = (t - a + 0.5) / max(1, n_h)
                g_db = base_lvl - 6.0 + 3.0 * math.sin(math.pi * u) + decay_db
                log_env[t] = np.clip((breath_db + g_db) * _DB_TO_NEPER, -16.0, 4.0)
                bap[t] = 0.97
            elif t < v1:
                u = (t - v0) / max(1, (v1 - v0) - 1)
                voiced[t] = True
                f0_hz[t] = np.clip(f0_ref * (2.0 ** ((st_pulse - 1.4 * u) / 12.0)), 65.0, 480.0)
                g_db = base_lvl + 2.5 * math.sin(math.pi * (0.15 + 0.70 * u)) + decay_db
                log_env[t] = np.clip((open_db + g_db) * _DB_TO_NEPER, -16.0, 4.0)
                bap[t] = np.array([0.20, 0.35, 0.55, 0.75, 0.88, 0.92]) + (0.14 if soft else 0.0)
            else:
                u = (t - v1 + 0.5) / max(1, b - v1)
                g_db = base_lvl - 8.0 - 10.0 * u + decay_db
                log_env[t] = np.clip((breath_db + g_db) * _DB_TO_NEPER, -16.0, 4.0)
                bap[t] = 0.98

    return LarynxControl(
        f0_hz=f0_hz,
        voiced=voiced,
        log_env=log_env,
        bap=np.clip(bap, 0.05, 0.995),
        glottal_tilt_db_per_oct=np.full(n_fr, 0.5 if not soft else -0.6, dtype=np.float64),
        jitter_frac=np.full(n_fr, 0.008, dtype=np.float64),
        shimmer_frac=np.full(n_fr, 0.035, dtype=np.float64),
        n_samples=n_fr * HOP,
        rd=rd,
    )
