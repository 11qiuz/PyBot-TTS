"""Frozen vectorized STFT/HNM electronic vocal cord synthesis kernel.

100% deterministic electronic synthesis (zero recorded waveform concatenation,
zero neural weights). Follows the unified STFT/HNM coupled domain proven in
exp_12 (coherent glottal phase) and exp_19b (STFT harmonic/noise separation).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.fft import dct, fftshift, idct, irfft, rfft, rfftfreq
from scipy.signal import lfilter

from exp_20_pybot_larynx import HOP, N_BAP, N_BINS, N_CEP, N_FFT, SR
from exp_v9_clarity_pro.hnm_v9 import BAND_CENTERS_V9

BAND_CENTERS_6 = np.asarray(BAND_CENTERS_V9, dtype=np.float64)


@dataclass
class LarynxControl:
    """Frame-synchronous electronic vocal cord + vocal tract control trajectory."""

    f0_hz: np.ndarray                     # (T,) 0.0 on unvoiced/silence frames
    voiced: np.ndarray                    # (T,) bool
    log_env: np.ndarray                   # (T, 513) natural-log amplitude spectrum
    bap: np.ndarray                       # (T, 6) band aperiodicity in [0, 1]
    glottal_tilt_db_per_oct: np.ndarray | None = None  # (T,) relative spectral tilt
    jitter_frac: np.ndarray | None = None              # (T,) cycle-to-cycle T0 jitter std
    shimmer_frac: np.ndarray | None = None             # (T,) cycle-to-cycle amp shimmer std
    sr: int = SR
    hop: int = HOP
    n_fft: int = N_FFT
    n_samples: int | None = None
    rd: np.ndarray | None = None                        # (T,) LF waveshape parameter in [0.5, 2.2]

    def __post_init__(self) -> None:
        self.f0_hz = np.asarray(self.f0_hz, dtype=np.float64).ravel()
        self.voiced = np.asarray(self.voiced, dtype=bool).ravel()
        self.log_env = np.asarray(self.log_env, dtype=np.float64)
        self.bap = np.asarray(self.bap, dtype=np.float64)
        n_fr = len(self.f0_hz)
        if self.voiced.shape != (n_fr,):
            raise ValueError(f"voiced shape {self.voiced.shape} != ({n_fr},)")
        if self.log_env.shape != (n_fr, self.n_fft // 2 + 1):
            raise ValueError(f"log_env shape {self.log_env.shape} != ({n_fr}, {self.n_fft // 2 + 1})")
        if self.bap.shape != (n_fr, N_BAP):
            raise ValueError(f"bap shape {self.bap.shape} != ({n_fr}, {N_BAP})")

        if self.glottal_tilt_db_per_oct is None:
            self.glottal_tilt_db_per_oct = np.zeros(n_fr, dtype=np.float64)
        else:
            self.glottal_tilt_db_per_oct = np.asarray(self.glottal_tilt_db_per_oct, dtype=np.float64).ravel()

        if self.jitter_frac is None:
            self.jitter_frac = np.full(n_fr, 0.0055, dtype=np.float64)
        else:
            self.jitter_frac = np.asarray(self.jitter_frac, dtype=np.float64).ravel()

        if self.shimmer_frac is None:
            self.shimmer_frac = np.full(n_fr, 0.025, dtype=np.float64)
        else:
            self.shimmer_frac = np.asarray(self.shimmer_frac, dtype=np.float64).ravel()

        if self.rd is None:
            self.rd = np.ones(n_fr, dtype=np.float64)
        else:
            self.rd = np.asarray(self.rd, dtype=np.float64).ravel()
            if self.rd.shape != (n_fr,):
                raise ValueError(f"rd shape {self.rd.shape} != ({n_fr},)")

    def copy(self) -> LarynxControl:
        """Return an independent deep copy of all trajectory arrays."""
        return LarynxControl(
            f0_hz=self.f0_hz.copy(),
            voiced=self.voiced.copy(),
            log_env=self.log_env.copy(),
            bap=self.bap.copy(),
            glottal_tilt_db_per_oct=self.glottal_tilt_db_per_oct.copy(),
            jitter_frac=self.jitter_frac.copy(),
            shimmer_frac=self.shimmer_frac.copy(),
            sr=self.sr,
            hop=self.hop,
            n_fft=self.n_fft,
            n_samples=self.n_samples,
            rd=self.rd.copy() if self.rd is not None else None,
        )


def log_env_to_cep(log_env: np.ndarray, n_cep: int = N_CEP) -> np.ndarray:
    """Compress (..., 513) natural-log amplitude envelope to (..., n_cep) ortho DCT."""
    arr = np.asarray(log_env, dtype=np.float64)
    full_cep = dct(arr, type=2, norm="ortho", axis=-1)
    return full_cep[..., :n_cep]


def cep_to_log_env(cep: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Reconstruct (..., n_bins) natural-log amplitude envelope from (..., n_cep) ortho DCT."""
    c = np.asarray(cep, dtype=np.float64)
    n_c = c.shape[-1]
    if n_c >= n_bins:
        padded = c[..., :n_bins]
    else:
        pad_shape = (*c.shape[:-1], n_bins - n_c)
        padded = np.concatenate([c, np.zeros(pad_shape, dtype=np.float64)], axis=-1)
    return idct(padded, type=2, norm="ortho", axis=-1)


def analytic_lf_phasor(
    n_fft: int = N_FFT,
    sr: int = SR,
    rd: float = 1.0,
) -> np.ndarray:
    """Synthesize unit-magnitude all-pass phase spectrum of a GCI-aligned LF glottal pulse.

    Purely mathematical (zero recorded audio). Positive peak at sample 0 matches
    the GCI alignment convention of `vocoder_v12._whiten_cycle_preserve_phase`.
    """
    rd_c = max(0.5, min(2.2, float(rd)))
    oq = min(0.85, max(0.42, 0.48 + 0.14 * rd_c))
    tp = 0.68 * oq
    sharp = 1.0 / (0.45 + 0.55 * rd_c)
    tau_r = max(0.015, 0.035 * rd_c)

    t0_ref = max(64, int(round(sr / 240.0)))
    phase = np.linspace(0.0, 1.0, t0_ref, endpoint=False)
    g_p = np.zeros_like(phase)

    m1 = phase < tp
    g_p[m1] = np.sin(math.pi * (phase[m1] / tp))

    m2 = (phase >= tp) & (phase < oq)
    u2 = (phase[m2] - tp) / max(1e-4, oq - tp)
    g_p[m2] = -sharp * 1.4 * (np.sin(0.5 * math.pi * u2) ** 1.2)

    m3 = phase >= oq
    g_p[m3] = -sharp * 0.45 * np.exp(-(phase[m3] - oq) / tau_r)

    g_p -= np.mean(g_p)
    # Flip sign so GCI closure excitation is a strong positive impulse (matching vocoder_v12)
    pulse = -g_p
    gci_idx = int(np.argmax(pulse))
    # Center GCI inside a ~1.15*T0 Hann window, then circularly shift GCI to index 0
    cyc = np.roll(pulse, (t0_ref // 2) - gci_idx) * np.hanning(t0_ref)
    half = t0_ref // 2
    c_pad = np.zeros(n_fft, dtype=np.float64)
    c_pad[: t0_ref - half] = cyc[half:]
    if half > 0:
        c_pad[-half:] = cyc[:half]

    spec = rfft(c_pad)
    mag = np.maximum(np.abs(spec), 1e-8)
    return spec / mag


_LF_RD_MIN = 0.50
_LF_RD_MAX = 2.20
_LF_RD_STEP = 0.05
_LF_RD_N = int(round((_LF_RD_MAX - _LF_RD_MIN) / _LF_RD_STEP)) + 1
_LF_RD_GRID = tuple(round(_LF_RD_MIN + i * _LF_RD_STEP, 4) for i in range(_LF_RD_N))


@lru_cache(maxsize=8)
def _lf_phasor_table(n_fft: int, sr: int) -> np.ndarray:
    """Precomputed (N_rd, N_bins) unit-magnitude LF glottal phasor lookup table."""
    table = np.stack([analytic_lf_phasor(n_fft=n_fft, sr=sr, rd=r) for r in _LF_RD_GRID], axis=0)
    table.setflags(write=False)
    return table


def _smooth_bap_batch(bap: np.ndarray, freqs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Vectorized log-frequency interpolation + 5-bin smoothing for (T, 6) BAP."""
    log_c = np.log2(np.maximum(60.0, centers))
    log_f = np.log2(np.maximum(60.0, freqs))
    # Piecewise linear interpolation along log_f across all frames simultaneously
    idx_hi = np.searchsorted(log_c, log_f, side="right")
    idx_hi = np.clip(idx_hi, 1, len(log_c) - 1)
    idx_lo = idx_hi - 1
    denom = np.maximum(1e-9, log_c[idx_hi] - log_c[idx_lo])
    alpha = np.clip((log_f - log_c[idx_lo]) / denom, 0.0, 1.0)
    ap = (1.0 - alpha[None, :]) * bap[:, idx_lo] + alpha[None, :] * bap[:, idx_hi]

    pad = np.pad(ap, ((0, 0), (2, 2)), mode="edge")
    sm = (
        0.15 * pad[:, :-4]
        + 0.20 * pad[:, 1:-3]
        + 0.30 * pad[:, 2:-2]
        + 0.20 * pad[:, 3:-1]
        + 0.15 * pad[:, 4:]
    )
    return np.clip(sm, 0.0, 1.0)


def synthesize_larynx(
    ctrl: LarynxControl,
    *,
    glottal_mode: str = "parametric_lf",
    coherent_phasor: np.ndarray | None = None,
    phase_dispersion: bool = True,
    lp_crossover_hz: float = 3500.0,
    rd: float = 1.0,
    seed: int = 123,
    disp_strength: float = 4.8,
    ap_glottal_mod: float = 0.55,
    coh_mix: float = 0.55,
) -> np.ndarray:
    """Synthesize 24 kHz mono speech from LarynxControl without any audio concatenation."""
    sr = int(ctrl.sr)
    hop = int(ctrl.hop)
    n_fft = int(ctrl.n_fft)
    n_fr = len(ctrl.f0_hz)
    if n_fr == 0:
        return np.zeros(0, dtype=np.float64)

    n = int(ctrl.n_samples) if ctrl.n_samples is not None else n_fr * hop
    freqs = rfftfreq(n_fft, 1.0 / sr)
    n_bins = len(freqs)
    rng = np.random.default_rng(seed)

    # 1. Apply dynamic glottal spectral tilt (pivoted at 1 kHz, tapered >8 kHz)
    env_use = np.asarray(ctrl.log_env, dtype=np.float64).copy()
    tilt = ctrl.glottal_tilt_db_per_oct
    if tilt is not None and np.any(np.abs(tilt) > 1e-6):
        oct_from_1k = np.log2(np.clip(freqs, 120.0, 11500.0) / 1000.0)
        hi_taper = np.where(freqs > 8000.0, np.maximum(0.0, 1.0 - (freqs - 8000.0) / max(1.0, sr * 0.5 - 8000.0)), 1.0)
        # 1 neper = 8.685889638 dB
        delta_ln = (tilt[:, None] / 8.685889638065035) * (oct_from_1k * hi_taper)[None, :]
        # Only modulate voiced frames so unvoiced consonant brightness stays pure
        env_use[ctrl.voiced] += delta_ln[ctrl.voiced]

    # 2. Vectorized minimum-phase causal vocal tract filter H_min (T, 513)
    ceps = irfft(env_use, n=n_fft, axis=1)
    win_mp = np.zeros(n_fft, dtype=np.float64)
    win_mp[0] = 1.0
    win_mp[1 : n_fft // 2] = 2.0
    win_mp[n_fft // 2] = 1.0
    H_min = np.exp(rfft(ceps * win_mp[None, :], n=n_fft, axis=1))

    # 2b. Smoothly remove DC and sub-F0 leakage from voiced harmonic envelopes
    h_env_all = H_min.copy()
    v_h_idx = np.where(ctrl.voiced & (ctrl.f0_hz > 40.0))[0]
    if v_h_idx.size > 0:
        f_cut_h = 0.78 * ctrl.f0_hz[v_h_idx][:, None]
        k_sub_h = min(n_bins, int(np.searchsorted(freqs, float(np.max(f_cut_h)), side="left")) + 1)
        if k_sub_h > 0:
            f_sub_h = freqs[None, :k_sub_h]
            h_atten = np.clip((f_sub_h / np.maximum(1.0, f_cut_h)) ** 4, 0.0, 1.0)
            h_env_all[v_h_idx, :k_sub_h] *= np.where(f_sub_h < f_cut_h, h_atten, 1.0)
        h_env_all[v_h_idx, 0] = 0.0

    # 3. Full-band aperiodicity ap_full (T, 513)
    ap_full = _smooth_bap_batch(np.clip(ctrl.bap, 0.0, 1.0), freqs, BAND_CENTERS_6)
    uv_mask = ~ctrl.voiced
    if np.any(uv_mask):
        ap_full[uv_mask] = 1.0

    sub70 = freqs < 70.0
    sub70_gain = (freqs[sub70] / 70.0) ** 2
    ap_full[:, sub70] *= sub70_gain[None, :]

    v_idx = np.where(ctrl.voiced & (ctrl.f0_hz > 0.0))[0]
    if v_idx.size > 0:
        f_cuts = 0.85 * ctrl.f0_hz[v_idx][:, None]
        k_sub_v = min(n_bins, int(np.searchsorted(freqs, float(np.max(f_cuts)), side="left")) + 1)
        if k_sub_v > 0:
            f_sub_v = freqs[None, :k_sub_v]
            atten = np.clip((f_sub_v / np.maximum(1.0, f_cuts)) ** 3, 0.0, 1.0)
            ap_full[v_idx, :k_sub_v] *= np.where(f_sub_v < f_cuts, atten, 1.0)

    ap_full = np.clip(ap_full, 0.0, 1.0)
    harm_gain = np.sqrt(np.maximum(0.0, 1.0 - ap_full * ap_full))

    # 4. Coherent glottal phase crossover (unified low-crest hybrid LF / speaker-coherent + chirp dispersion)
    if phase_dispersion:
        f_u = np.clip((freqs - 350.0) / 3650.0, 0.0, 3.0)
        phi_disp = np.exp(-1j * (float(disp_strength) * (f_u ** 1.75) * 0.45 * math.pi))
    else:
        phi_disp = np.ones(n_bins, dtype=np.complex128)

    lo_x = max(1200.0, lp_crossover_hz - 700.0)
    hi_x = lp_crossover_hz + 700.0
    u_x = np.clip((freqs - lo_x) / (hi_x - lo_x), 0.0, 1.0)
    w_hf = 0.5 * (1.0 - np.cos(math.pi * u_x))

    use_rd_lut = False
    hybrid_table: np.ndarray | None = None
    rd_arr: np.ndarray | None = None
    hybrid_phasor_static: np.ndarray | None = None

    if glottal_mode == "speaker_coherent" and coherent_phasor is not None and len(coherent_phasor) == n_bins:
        phi_coh = np.asarray(coherent_phasor, dtype=np.complex128)
        phi_coh = phi_coh / np.maximum(np.abs(phi_coh), 1e-8)

        # Low-mid chest phase (<2.2 kHz, coh_mix blend with LF) + smooth chirp dispersion above 1.5 kHz
        mix_c = min(1.0, max(0.0, float(coh_mix)))
        u_chest = np.clip((freqs - 1200.0) / 1000.0, 0.0, 1.0)
        w_chest = mix_c * 0.5 * (1.0 + np.cos(math.pi * u_chest))
        lf_table = _lf_phasor_table(n_fft, sr)
        rd_ref_idx = min(_LF_RD_N - 1, max(0, int(round((1.0 - _LF_RD_MIN) / _LF_RD_STEP))))
        phi_lf_ref = lf_table[rd_ref_idx]
        phi_chest = (1.0 - w_chest) * phi_lf_ref + w_chest * phi_coh
        phi_chest /= np.maximum(np.abs(phi_chest), 1e-8)

        if phase_dispersion:
            u_d_coh = np.clip((freqs - 1000.0) / 1200.0, 0.0, 1.0)
            w_d_coh = 0.5 * (1.0 - np.cos(math.pi * u_d_coh))
            phi_disp_coh = np.exp(-1j * (float(disp_strength) * (f_u ** 1.75) * 0.45 * math.pi * w_d_coh))
        else:
            phi_disp_coh = 1.0

        hybrid_phasor_static = ((1.0 - w_hf) * phi_chest + w_hf) * phi_disp_coh
        hybrid_phasor_static /= np.maximum(np.abs(hybrid_phasor_static), 1e-8)
    elif glottal_mode == "parametric_lf":
        if ctrl.rd is None:
            rd_arr = np.full(n_fr, float(rd), dtype=np.float64)
        else:
            rd_arr = np.asarray(ctrl.rd, dtype=np.float64).ravel()
            if abs(float(rd) - 1.0) > 1e-6 and np.allclose(rd_arr, 1.0):
                rd_arr = np.full(n_fr, float(rd), dtype=np.float64)
        rd_arr = np.clip(np.nan_to_num(rd_arr, nan=1.0, posinf=_LF_RD_MAX, neginf=_LF_RD_MIN), _LF_RD_MIN, _LF_RD_MAX)

        lf_table = _lf_phasor_table(n_fft, sr)
        hybrid_table = ((1.0 - w_hf[None, :]) * lf_table + w_hf[None, :]) * phi_disp[None, :]
        hybrid_table /= np.maximum(np.abs(hybrid_table), 1e-8)

        if float(np.ptp(rd_arr)) < 1e-6:
            rd_pos = (float(rd_arr[0]) - _LF_RD_MIN) / _LF_RD_STEP
            r_lo = min(_LF_RD_N - 2, max(0, int(math.floor(rd_pos))))
            r_alpha = min(1.0, max(0.0, rd_pos - r_lo))
            if r_alpha <= 1e-6:
                hybrid_phasor_static = hybrid_table[r_lo]
            elif r_alpha >= 1.0 - 1e-6:
                hybrid_phasor_static = hybrid_table[r_lo + 1]
            else:
                p_mix = (1.0 - r_alpha) * hybrid_table[r_lo] + r_alpha * hybrid_table[r_lo + 1]
                hybrid_phasor_static = p_mix / np.maximum(np.abs(p_mix), 1e-8)
        else:
            use_rd_lut = True
    elif glottal_mode == "zero_phase":
        hybrid_phasor_static = phi_disp
    else:
        raise ValueError(f"Unsupported glottal_mode: {glottal_mode!r}")

    # 5. Voiced sample gate (5 ms cosine taper prevents vowel tails ringing into consonants)
    v_smp = np.repeat(ctrl.voiced.astype(np.float64), hop)
    gate_len = n + n_fft
    if len(v_smp) < gate_len:
        v_smp = np.pad(v_smp, (0, gate_len - len(v_smp)), constant_values=0.0)
    else:
        v_smp = v_smp[:gate_len]
    k_sm = max(48, min(144, hop))
    kernel = np.hanning(k_sm + 2)[1:-1]
    kernel /= np.sum(kernel)
    voiced_sample_gate = np.convolve(v_smp, kernel, mode="same")

    # 6. Pitch-synchronous coherent glottal pulse overlap-add (chunked batch IRFFT)
    y_harm = np.zeros(n + n_fft, dtype=np.float64)
    pulse_offset = n_fft // 4
    w_ang = math.tau * freqs / sr
    roll_phasor = np.exp(-1j * (math.tau * pulse_offset / n_fft) * np.arange(n_bins, dtype=np.float64))
    if use_rd_lut and hybrid_table is not None:
        hybrid_table_rolled = hybrid_table * roll_phasor[None, :]
    else:
        hybrid_table_rolled = None
    hybrid_static_rolled = (hybrid_phasor_static * roll_phasor)[None, :] if hybrid_phasor_static is not None else None

    hf_start = int(np.searchsorted(freqs, 1800.0, side="right"))
    has_hf = bool(phase_dispersion and hf_start < n_bins)
    hf_rand_scale = (np.clip((freqs[hf_start:] - 1800.0) / 5000.0, 0.0, 1.0) * 0.42) if has_hf else None

    jit_arr = ctrl.jitter_frac if ctrl.jitter_frac is not None else np.full(n_fr, 0.0055)
    shim_arr = ctrl.shimmer_frac if ctrl.shimmer_frac is not None else np.full(n_fr, 0.025)

    p_f0_i: list[int] = []
    p_f1_i: list[int] = []
    p_w1: list[float] = []
    p_fi_round: list[int] = []
    p_both_v: list[bool] = []
    p_idx_int: list[int] = []
    p_frac: list[float] = []
    p_scale: list[float] = []
    p_t_cur: list[float] = []
    p_t0: list[float] = []
    p_seg_start: list[bool] = []

    ar_j = 0.52
    ar_j_in = math.sqrt(1.0 - ar_j * ar_j)
    ar_s = 0.82
    ar_s_in = math.sqrt(1.0 - ar_s * ar_s)

    t_cur = 0.0
    jit_state = 0.0
    shim_state = 0.0
    in_voiced_run = False
    t0_min = sr / 460.0
    t0_max = sr / 60.0
    while t_cur < n:
        f_pos = min(float(n_fr - 1), max(0.0, t_cur / hop))
        f0_i = int(math.floor(f_pos))
        f1_i = min(n_fr - 1, f0_i + 1)
        w1 = f_pos - f0_i
        fi_round = min(n_fr - 1, max(0, int(round(f_pos))))
        f0 = float(ctrl.f0_hz[fi_round])
        if ctrl.voiced[fi_round] and f0 > 0.0:
            both_v = bool(ctrl.voiced[f0_i] and ctrl.voiced[f1_i] and ctrl.f0_hz[f0_i] > 0.0 and ctrl.f0_hz[f1_i] > 0.0)
            if both_v:
                f0 = (1.0 - w1) * float(ctrl.f0_hz[f0_i]) + w1 * float(ctrl.f0_hz[f1_i])

            j_scale = float(jit_arr[fi_round])
            s_scale = float(shim_arr[fi_round])
            jit_state = ar_j * jit_state + ar_j_in * float(rng.standard_normal()) * j_scale
            shim_state = ar_s * shim_state + ar_s_in * float(rng.standard_normal()) * s_scale
            t0_nom = sr / f0
            t0 = max(t0_min, min(t0_max, t0_nom * (1.0 + jit_state)))
            amp_mul = max(0.80, min(1.22, 1.0 + shim_state))

            idx_int = int(math.floor(t_cur))
            p_f0_i.append(f0_i)
            p_f1_i.append(f1_i)
            p_w1.append(w1)
            p_fi_round.append(fi_round)
            p_both_v.append(both_v)
            p_idx_int.append(idx_int)
            p_frac.append(t_cur - idx_int)
            p_scale.append(math.sqrt(t0_nom) * amp_mul)
            p_t_cur.append(t_cur)
            p_t0.append(t0)
            p_seg_start.append(not in_voiced_run)
            in_voiced_run = True
            t_cur += t0
        else:
            jit_state = 0.0
            shim_state = 0.0
            in_voiced_run = False
            t_cur += hop

    n_pulses = len(p_idx_int)
    if n_pulses > 0:
        arr_f0_i = np.asarray(p_f0_i, dtype=np.intp)
        arr_f1_i = np.asarray(p_f1_i, dtype=np.intp)
        arr_fi_r = np.asarray(p_fi_round, dtype=np.intp)
        arr_w1 = np.asarray(p_w1, dtype=np.float64)
        arr_both = np.asarray(p_both_v, dtype=bool)
        arr_frac = np.asarray(p_frac, dtype=np.float64)
        arr_scale = np.asarray(p_scale, dtype=np.float64)
        y_len = len(y_harm)
        n_hf = (n_bins - hf_start) if has_hf else 0

        # Precompute AR(1) (rho=0.82) smooth HF phase trajectory across contiguous voiced runs
        ar_hf_phases: np.ndarray | None = None
        if has_hf and hf_rand_scale is not None and n_hf > 0:
            ar_p = 0.82
            ar_in = math.sqrt(1.0 - ar_p * ar_p)
            eps_hf = rng.uniform(-math.pi, math.pi, size=(n_pulses, n_hf))
            ar_hf_phases = np.empty_like(eps_hf)
            seg_starts = np.flatnonzero(p_seg_start)
            for s_idx_i, r0 in enumerate(seg_starts):
                r1 = int(seg_starts[s_idx_i + 1]) if s_idx_i + 1 < len(seg_starts) else n_pulses
                ar_hf_phases[r0:r1] = lfilter([ar_in], [1.0, -ar_p], eps_hf[r0:r1], axis=0)

        chunk_sz = 256
        for c0 in range(0, n_pulses, chunk_sz):
            c1 = min(n_pulses, c0 + chunk_sz)
            c_f0 = arr_f0_i[c0:c1]
            c_f1 = arr_f1_i[c0:c1]
            c_fi = arr_fi_r[c0:c1]
            c_w1 = arr_w1[c0:c1, None]
            c_w0 = 1.0 - c_w1
            nb_idx = np.flatnonzero(~arr_both[c0:c1])

            spec_b = c_w0 * h_env_all[c_f0] + c_w1 * h_env_all[c_f1]
            hg_b = c_w0 * harm_gain[c_f0] + c_w1 * harm_gain[c_f1]
            if nb_idx.size > 0:
                spec_b[nb_idx] = h_env_all[c_fi[nb_idx]]
                hg_b[nb_idx] = harm_gain[c_fi[nb_idx]]
            spec_b *= hg_b

            if use_rd_lut and hybrid_table_rolled is not None and rd_arr is not None:
                rd_p = c_w0[:, 0] * rd_arr[c_f0] + c_w1[:, 0] * rd_arr[c_f1]
                if nb_idx.size > 0:
                    rd_p[nb_idx] = rd_arr[c_fi[nb_idx]]
                rd_pos = (rd_p - _LF_RD_MIN) / _LF_RD_STEP
                r_lo = np.clip(np.floor(rd_pos).astype(np.intp), 0, _LF_RD_N - 2)
                r_alpha = np.clip(rd_pos - r_lo, 0.0, 1.0)[:, None]
                hybrid_b = (1.0 - r_alpha) * hybrid_table_rolled[r_lo] + r_alpha * hybrid_table_rolled[r_lo + 1]
                hybrid_b /= np.maximum(np.abs(hybrid_b), 1e-8)
                spec_b *= hybrid_b
            else:
                spec_b *= hybrid_static_rolled

            spec_b *= np.exp(-1j * (arr_frac[c0:c1, None] * w_ang[None, :]))

            if ar_hf_phases is not None and hf_rand_scale is not None:
                jit_phases = ar_hf_phases[c0:c1] * (hf_rand_scale[None, :] * ap_full[c_fi, hf_start:])
                spec_b[:, hf_start:] *= np.exp(1j * jit_phases)

            pulses = irfft(spec_b, n=n_fft, axis=1)
            pulses -= np.mean(pulses, axis=1, keepdims=True)
            pulses *= arr_scale[c0:c1, None]

            for k in range(c1 - c0):
                start = p_idx_int[c0 + k] - pulse_offset
                if start >= 0 and start + n_fft <= y_len:
                    y_harm[start : start + n_fft] += pulses[k]
                elif start < 0 and start + n_fft > 0:
                    y_harm[0 : start + n_fft] += pulses[k, -start:]

    y_harm *= voiced_sample_gate

    # 7. Vectorized 20 ms (480-sample) centered Hann COLA STFT noise synthesis + glottal-cycle modulation
    win_len = min(n_fft, 4 * hop)
    if win_len % 2 == 1:
        win_len -= 1
    half_w = win_len // 2
    y_ap = np.zeros(n + n_fft, dtype=np.float64)
    norm_ola = np.zeros(n + n_fft, dtype=np.float64)
    win_ola = np.hanning(win_len)
    w2 = win_ola * win_ola
    noise_all = rng.standard_normal((n_fr, n_fft))
    noise_specs = rfft(noise_all, axis=1)
    ap_time = fftshift(irfft(H_min * ap_full * noise_specs, n=n_fft, axis=1), axes=1)
    mid = n_fft // 2
    ap_frames = ap_time[:, mid - half_w : mid + half_w] * win_ola[None, :]
    for i in range(n_fr):
        s = i * hop - half_w
        e = s + win_len
        if e <= 0 or s >= len(y_ap):
            continue
        src_s = 0 if s >= 0 else -s
        dst_s = max(0, s)
        dst_e = min(len(y_ap), e)
        take = dst_e - dst_s
        y_ap[dst_s:dst_e] += ap_frames[i, src_s : src_s + take]
        norm_ola[dst_s:dst_e] += w2[src_s : src_s + take]

    y_ap_out = (y_ap / np.maximum(np.sqrt(norm_ola), 1e-6))[:n]
    mod_depth = min(0.95, max(0.0, float(ap_glottal_mod)))
    if n_pulses > 0 and mod_depth > 1e-6:
        cyc_mod = np.ones(n, dtype=np.float64)
        for tc, t0_p in zip(p_t_cur, p_t0):
            i0 = max(0, int(math.floor(tc)))
            i1 = min(n, int(math.floor(tc + t0_p)))
            if i1 <= i0 + 2:
                continue
            phi = (np.arange(i0, i1, dtype=np.float64) - tc) / t0_p
            m_cyc = np.maximum(0.05, 1.0 - mod_depth * np.cos(math.tau * phi))
            rms_m = float(np.sqrt(np.mean(m_cyc * m_cyc)))
            if rms_m > 1e-9:
                m_cyc /= rms_m
            cyc_mod[i0:i1] = m_cyc
        v_gate = voiced_sample_gate[:n]
        y_ap_out *= (1.0 - v_gate) + v_gate * cyc_mod

    return y_harm[:n] + y_ap_out
