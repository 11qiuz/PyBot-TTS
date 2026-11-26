"""Phase 8 (v12): LP-Residual Pitch-Synchronous Excitation, Hybrid LP+Noise & Crisp Unvoiced Bursts."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter, sosfiltfilt

from exp_v6_vocoder import vocoder
from exp_v9_clarity_pro.hnm_v9 import BAND_CENTERS_V9, _hp_sos, analyze_v9
from exp_v11_diphone_tts.hnm_v11 import _smooth_bap_curve, synthesize_v11
from exp_v12_natural_vocoder import HOP, N_FFT, SR


@dataclass
class GlottalPulseBank:
    """Speaker-specific library of whitened GCI-aligned real human LP-residual cycles (0.0 MB neural weights)."""
    speaker: str
    pulses: np.ndarray        # (K, L_pulse) unit-energy zero-lag-centered whitened cycles
    f0_hz: np.ndarray         # (K,) source F0 of each cycle
    lpc_order: int
    sr: int = SR
    coherent_phasor: np.ndarray | None = None   # (513,) sub-sample GCI-aligned mean speaker glottal phase


@dataclass
class VocoderFeaturesV12:
    """Extended v12 vocoder bundle preserving 100% VocoderFeatures compatibility for TimbreClonerV5."""
    feats: vocoder.VocoderFeatures
    bap_gt8k: np.ndarray               # (T, 2)
    uv_raw_log_env: np.ndarray | None = None   # (T, 513) unsmoothed 513-bin magnitude for voiced==False
    lp_pulses: np.ndarray | None = None        # (T, N_FFT) unit-energy whitened real glottal residual cycle per frame
    coherent_phasor: np.ndarray | None = None  # (513,) sub-sample GCI-aligned coherent utterance glottal phase


def _compute_lpc_and_residual(x: np.ndarray, order: int = 26, frame_len: int = 600, hop: int = HOP) -> np.ndarray:
    """Extract frame-wise LPC inverse-filter residual r[n] = A(z) x[n] across full signal x."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    n = len(arr)
    if n < order + 8:
        return arr.copy()
    # Slight pre-emphasis before LPC helps capture sharp vocal-tract formants cleanly
    xp = np.empty(n, dtype=np.float64)
    xp[0] = arr[0]
    xp[1:] = arr[1:] - 0.92 * arr[:-1]

    res = np.zeros(n, dtype=np.float64)
    norm_w = np.zeros(n, dtype=np.float64)
    half = frame_len // 2
    win = np.hanning(frame_len)
    pad = half
    x_pad = np.pad(xp, (pad, pad))

    for center in range(0, n, hop):
        seg = x_pad[center : center + frame_len]
        seg_w = (seg - np.mean(seg)) * win
        e0 = float(np.dot(seg_w, seg_w))
        if e0 <= 1e-12:
            continue
        # Autocorrelation r[0..order]
        ac = np.correlate(seg_w, seg_w, mode="full")[frame_len - 1 : frame_len + order]
        ac[0] = max(ac[0] * 1.001, 1e-12)  # mild white-noise floor prevents ill-conditioned poles
        try:
            a_tail = solve_toeplitz((ac[:-1], ac[:-1]), -ac[1:])
            a = np.concatenate([[1.0], a_tail])
        except Exception:
            a = np.array([1.0], dtype=np.float64)

        # Inverse filter the padded window with context so boundary transient vanishes inside win
        s_ctx = max(0, center - order)
        s_end = min(n + 2 * pad, center + frame_len)
        r_blk = lfilter(a, [1.0], x_pad[s_ctx:s_end])
        r_seg = r_blk[center - s_ctx :]
        if len(r_seg) == frame_len:
            s_out = center - half
            o0 = max(0, s_out)
            o1 = min(n, s_out + frame_len)
            if o1 > o0:
                w_sl = win[o0 - s_out : o1 - s_out]
                res[o0:o1] += r_seg[o0 - s_out : o1 - s_out] * w_sl
                norm_w[o0:o1] += w_sl

    good = norm_w > 1e-6
    res[good] /= norm_w[good]
    return res


def _whiten_cycle_preserve_phase(
    cycle: np.ndarray,
    n_fft: int = N_FFT,
    q_flat: int = 8,
    mag_exponent: float = 0.0,
) -> np.ndarray:
    """Flatten smooth/comb magnitude of a GCI-aligned residual grain while preserving true glottal phase.

    When `mag_exponent == 0.0`, outputs a strict unit-magnitude all-pass glottal impulse (zero comb distortion,
    preserving exact real human glottal open/close phase dispersion). When `mag_exponent > 0`, retains mild
    cepstrally-flattened residual magnitude variation.
    """
    c_pad = np.zeros(n_fft, dtype=np.float64)
    L = len(cycle)
    half = L // 2
    c_pad[: L - half] = cycle[half:]
    if half > 0:
        c_pad[-half:] = cycle[:half]
    spec = rfft(c_pad)
    mag = np.maximum(np.abs(spec), 1e-8)
    phase = spec / mag
    if mag_exponent <= 0.0:
        spec_wh = phase
    else:
        log_m = np.log(mag)
        cep = irfft(log_m, n=n_fft)
        cep_sm = np.zeros_like(cep)
        cep_sm[0] = cep[0]
        cep_sm[1:q_flat] = cep[1:q_flat]
        cep_sm[-q_flat + 1 :] = cep[-q_flat + 1 :]
        smooth_mag = np.exp(rfft(cep_sm, n=n_fft).real)
        flat_mag = np.clip((mag / np.maximum(smooth_mag, 1e-6)) ** mag_exponent, 0.72, 1.38)
        spec_wh = flat_mag * phase
    grain = irfft(spec_wh, n=n_fft)
    e = float(np.sum(grain * grain))
    if e > 1e-12:
        grain *= 1.0 / math.sqrt(e)
    return grain


def analyze_v12(
    x: np.ndarray,
    sr: int = SR,
    hop_ms: float = 5.0,
    n_fft: int = N_FFT,
    lpc_order: int = 26,
    pulse_mag_exponent: float = 0.0,
) -> VocoderFeaturesV12:
    """Analyze speech into VocoderFeatures + unsmoothed 513-bin unvoiced envelopes + GCI-aligned LP residual pulses."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    feats, b_gt8 = analyze_v9(arr, sr=sr, hop_ms=hop_ms, n_fft=n_fft)
    n = len(arr)
    hop = feats.hop_samples
    n_fr = len(feats.f0_hz)

    # 1. Unsmoothed high-def 513-bin unvoiced envelope (no cepstral smoothing, shorter 10 ms Hann window)
    uv_raw_env = feats.log_env.copy()
    win_uv_len = min(240, max(128, int(round(0.010 * sr))))
    win_uv = np.hanning(win_uv_len)
    win_uv /= math.sqrt(float(np.sum(win_uv * win_uv)))
    pad_uv = win_uv_len // 2
    x_uv_pad = np.pad(arr, (pad_uv, pad_uv))
    for i in range(n_fr):
        if feats.voiced[i]:
            continue
        seg = x_uv_pad[i * hop : i * hop + win_uv_len]
        if len(seg) < win_uv_len:
            continue
        xw = (seg - np.mean(seg)) * win_uv
        pwr = np.abs(rfft(xw, n=n_fft)) ** 2
        # Very tiny 3-bin spectral variance reduction only on floor; keep sharp fricative peaks intact
        log_e = 0.5 * np.log(np.maximum(pwr, 1e-18))
        fp = float(np.sum(xw * xw))
        el = np.exp(log_e)
        ep = float(el[0] ** 2 + 2.0 * np.sum(el[1:-1] ** 2) + el[-1] ** 2) / n_fft
        if ep > 1e-18 and fp > 1e-18:
            log_e += 0.5 * math.log(fp / ep)
        uv_raw_env[i] = log_e

    # 2. Extract GCI-aligned LP-residual pulses per voiced frame + coherent sub-sample glottal phase
    res = _compute_lpc_and_residual(arr, order=lpc_order, frame_len=600, hop=hop)
    pulses = np.zeros((n_fr, n_fft), dtype=np.float32)
    coh_phasors: list[np.ndarray] = []
    bins_norm = np.arange(n_fft // 2 + 1, dtype=np.float64) / float(n_fft)

    for i in range(n_fr):
        f0 = float(feats.f0_hz[i])
        if not (feats.voiced[i] and f0 > 0.0):
            continue
        T0 = int(np.clip(round(sr / f0), 48, 360))
        c = i * hop
        s_search0 = max(0, c - T0 // 2)
        s_search1 = min(n, c + T0 // 2 + 1)
        if s_search1 <= s_search0 + 4:
            continue
        win_search = res[s_search0:s_search1]
        # Consistent polarity GCI lock: strongest magnitude impulse, forcing positive impulse sign
        pk_idx = int(np.argmax(np.abs(win_search)))
        gci = s_search0 + pk_idx
        sign = 1.0 if res[gci] >= 0.0 else -1.0

        # Extract single-cycle (1.15*T0) Hann grain centered at GCI so neighbor pulses do not cause double-echo
        L = min(n_fft - 4, max(32, int(round(1.15 * T0))))
        if L % 2 == 1:
            L += 1
        half = L // 2
        g0 = gci - half
        g1 = gci + half
        cyc = np.zeros(L, dtype=np.float64)
        o0 = max(0, g0)
        o1 = min(n, g1)
        if o1 > o0:
            cyc[o0 - g0 : o1 - g0] = res[o0:o1] * sign
        cyc *= np.hanning(L)
        pulses[i] = _whiten_cycle_preserve_phase(cyc, n_fft=n_fft, q_flat=8, mag_exponent=pulse_mag_exponent).astype(np.float32)

        # Sub-sample peak alignment for coherent glottal phase averaging on clean periodic frames
        if 1 < i < n_fr - 2 and feats.voiced[i - 1] and feats.voiced[i + 1] and float(np.mean(feats.bap[i, :2])) < 0.30:
            if 1 <= gci < n - 1:
                ym, y0, yp = res[gci - 1] * sign, res[gci] * sign, res[gci + 1] * sign
                den = ym - 2.0 * y0 + yp
                delta = float(np.clip(0.5 * (ym - yp) / den, -0.5, 0.5)) if abs(den) > 1e-12 else 0.0
            else:
                delta = 0.0
            c_pad = np.zeros(n_fft, dtype=np.float64)
            c_pad[: L - half] = cyc[half:]
            if half > 0:
                c_pad[-half:] = cyc[:half]
            sp = rfft(c_pad)
            sp_al = sp * np.exp(1j * math.tau * bins_norm * delta)
            coh_phasors.append(sp_al / np.maximum(np.abs(sp_al), 1e-8))

    if coh_phasors:
        mean_ph = np.mean(np.stack(coh_phasors, axis=0), axis=0)
        mean_ph = (mean_ph / np.maximum(np.abs(mean_ph), 1e-8)).astype(np.complex64)
    else:
        mean_ph = np.ones(n_fft // 2 + 1, dtype=np.complex64)

    return VocoderFeaturesV12(
        feats=feats,
        bap_gt8k=b_gt8,
        uv_raw_log_env=uv_raw_env,
        lp_pulses=pulses,
        coherent_phasor=mean_ph,
    )


def build_glottal_pulse_bank(
    speaker: str,
    analyzed_items: list[VocoderFeaturesV12],
    max_pulses: int = 32,
) -> GlottalPulseBank:
    """Build compact speaker glottal LP-residual cycle codebook + coherent glottal phasor from clean voiced frames."""
    cands: list[tuple[float, np.ndarray, float]] = []
    ph_list: list[np.ndarray] = []
    for item in analyzed_items:
        if item.coherent_phasor is not None:
            ph_list.append(item.coherent_phasor.astype(np.complex128))
        if item.lp_pulses is None:
            continue
        f = item.feats
        n_fr = len(f.voiced)
        for i in range(2, n_fr - 2):
            if not (f.voiced[i - 1] and f.voiced[i] and f.voiced[i + 1]):
                continue
            if f.f0_hz[i] <= 0.0:
                continue
            if float(np.mean(f.bap[i, :2])) > 0.35:
                continue
            p = item.lp_pulses[i]
            if float(np.sum(p * p)) < 0.5:
                continue
            # Prefer mid-energy clear periodic pulses
            pwr = float(np.mean(f.log_env[i, 5:60]))
            cands.append((pwr, p.astype(np.float32), float(f.f0_hz[i])))

    if ph_list:
        spk_ph = np.mean(np.stack(ph_list, axis=0), axis=0)
        spk_ph = (spk_ph / np.maximum(np.abs(spk_ph), 1e-8)).astype(np.complex64)
    else:
        spk_ph = np.ones(N_FFT // 2 + 1, dtype=np.complex64)

    if not cands:
        # Fallback neutral Kronecker-like unit impulse centered at 0
        imp = np.zeros((1, N_FFT), dtype=np.float32)
        imp[0, 0] = 1.0
        return GlottalPulseBank(
            speaker=speaker, pulses=imp, f0_hz=np.array([150.0], dtype=np.float32),
            lpc_order=26, sr=SR, coherent_phasor=spk_ph,
        )

    # Sort by energy & spread uniformly across dynamic range to get diverse representative pulses
    cands.sort(key=lambda t: t[0])
    idx = np.linspace(len(cands) // 5, max(len(cands) // 5, len(cands) - 1), min(max_pulses, len(cands)), dtype=int)
    sel_p = np.stack([cands[k][1] for k in idx], axis=0)
    sel_f0 = np.array([cands[k][2] for k in idx], dtype=np.float32)
    return GlottalPulseBank(speaker=speaker, pulses=sel_p, f0_hz=sel_f0, lpc_order=26, sr=SR, coherent_phasor=spk_ph)


def _minimum_phase_filters(log_env: np.ndarray, n_fft: int) -> np.ndarray:
    ceps = irfft(log_env, n=n_fft, axis=1)
    win_mp = np.zeros(n_fft, dtype=np.float64)
    win_mp[0] = 1.0
    win_mp[1 : n_fft // 2] = 2.0
    win_mp[n_fft // 2] = 1.0
    return np.exp(rfft(ceps * win_mp[None, :], n=n_fft, axis=1))


def synthesize_v12(
    feats: vocoder.VocoderFeatures | VocoderFeaturesV12,
    *,
    mode: str = "hybrid_lp_hnm",   # "hnm_v11" | "lp_psola" | "hybrid_lp_hnm"
    glottal_bank: GlottalPulseBank | None = None,
    seed: int = 72,
    lp_crossover_hz: float = 3500.0,
    phase_dispersion: bool = True,
    f0_jitter_frac: float = 0.003,
    amp_shimmer_frac: float = 0.010,
    bap_gt8k: np.ndarray | None = None,
    uv_raw_log_env: np.ndarray | None = None,
    unvoiced_residual_spans: list[tuple[int, int, np.ndarray]] | None = None,
    residual_highpass_hz: float = 1600.0,
    crisp_unvoiced: bool = True,
) -> np.ndarray:
    """Synthesize natural speech with LP-Residual excitation + crisp unsmoothed unvoiced consonants.

    Modes:
    - `hnm_v11`: delegates cleanly to v11 baseline (optionally with crisp unvoiced fix if requested).
    - `lp_psola`: pitch-synchronous overlap-add of real human LP-residual cycles across full band,
      filtered through minimum-phase H_min (no synthetic sine harmonics).
    - `hybrid_lp_hnm`: true LP-residual cycle excitation below `lp_crossover_hz` (natural glottal waveform,
      no pitch-shift phasiness at >3.5 kHz) + BAP-shaped HNM / phase-dispersed high band above `lp_crossover_hz`.
    In `lp_psola` & `hybrid_lp_hnm` (when `crisp_unvoiced=True`), unvoiced frames bypass smoothing/jitter and use
    unsmoothed 513-bin spectral bursts + voiced-tail isolation so vowel pulses never bleed into sibilants.
    """
    if isinstance(feats, VocoderFeaturesV12):
        lp_pulses = feats.lp_pulses
        coh_phasor_feat = feats.coherent_phasor
        uv_env = uv_raw_log_env if uv_raw_log_env is not None else feats.uv_raw_log_env
        bg8 = bap_gt8k if bap_gt8k is not None else feats.bap_gt8k
        base_feats = feats.feats
    else:
        lp_pulses = None
        coh_phasor_feat = None
        uv_env = uv_raw_log_env
        bg8 = bap_gt8k
        base_feats = feats

    if mode == "hnm_v11" and not crisp_unvoiced:
        return synthesize_v11(
            base_feats,
            seed=seed,
            phase_dispersion=phase_dispersion,
            bap_gt8k=bg8,
            unvoiced_residual_spans=unvoiced_residual_spans,
            residual_highpass_hz=residual_highpass_hz,
        )

    sr = base_feats.sr
    hop = base_feats.hop_samples
    n = base_feats.n_samples
    n_fft = base_feats.n_fft
    n_fr = len(base_feats.f0_hz)
    freqs = rfftfreq(n_fft, 1.0 / sr)
    rng = np.random.default_rng(seed)

    # Combine voiced log_env with unsmoothed raw 513-bin unvoiced bursts on voiced==False
    env_use = base_feats.log_env.astype(np.float64).copy()
    if crisp_unvoiced and uv_env is not None and len(uv_env) == n_fr:
        uv_m = ~base_feats.voiced
        env_use[uv_m] = uv_env[uv_m].astype(np.float64)

    H_min = _minimum_phase_filters(env_use, n_fft=n_fft)

    if phase_dispersion:
        f_disp = np.maximum(0.0, freqs - 2200.0) / max(1.0, (sr * 0.5 - 2200.0))
        phi_disp = np.exp(-1j * (2.4 * (f_disp ** 1.85) * math.pi))
        H_harm_base = H_min * phi_disp[None, :]
    else:
        H_harm_base = H_min

    # Band aperiodicity: bypass smoothing & set ap=1.0 cleanly above 65 Hz for pure unvoiced consonant frames
    ap_full = np.empty((n_fr, len(freqs)), dtype=np.float64)
    sub70 = freqs < 70.0
    sub70_gain = (freqs[sub70] / 70.0) ** 2
    hf_soft_mask = np.clip((freqs - 3000.0) / 7000.0, 0.0, 1.0)

    if bg8 is not None and len(bg8) == n_fr:
        baps_use = np.concatenate([base_feats.bap[:, :4], bg8], axis=1)
        centers_use = BAND_CENTERS_V9
    else:
        baps_use = base_feats.bap
        centers_use = vocoder.BAND_CENTERS

    for i in range(n_fr):
        if crisp_unvoiced and not base_feats.voiced[i]:
            ap_row = np.ones(len(freqs), dtype=np.float64)
            ap_row[sub70] *= sub70_gain
        else:
            ap_row = _smooth_bap_curve(baps_use[i], freqs, centers_use)
            ap_row[sub70] *= sub70_gain
            if base_feats.voiced[i] and base_feats.f0_hz[i] > 0.0:
                f_cut = 0.85 * float(base_feats.f0_hz[i])
                sub_mask = freqs < f_cut
                ap_row[sub_mask] *= np.clip((freqs[sub_mask] / max(1.0, f_cut)) ** 3, 0.0, 1.0)
                ap_row = np.maximum(ap_row, 0.08 * (hf_soft_mask ** 1.5))
        ap_full[i] = np.clip(ap_row, 0.0, 1.0)
    harm_gain = np.sqrt(np.maximum(0.0, 1.0 - ap_full * ap_full))

    # Hybrid crossover: sub-sample GCI-aligned coherent human glottal phase below lp_crossover_hz + dispersion above
    phi_disp = np.exp(-1j * (2.4 * ((np.maximum(0.0, freqs - 2200.0) / max(1.0, sr * 0.5 - 2200.0)) ** 1.85) * math.pi)) if phase_dispersion else np.ones(len(freqs), dtype=np.complex128)
    phi_glottal = None
    if mode == "hybrid_lp_hnm":
        if coh_phasor_feat is not None and len(coh_phasor_feat) == len(freqs):
            phi_glottal = coh_phasor_feat.astype(np.complex128)
        elif glottal_bank is not None and glottal_bank.coherent_phasor is not None and len(glottal_bank.coherent_phasor) == len(freqs):
            phi_glottal = glottal_bank.coherent_phasor.astype(np.complex128)
    if phi_glottal is not None:
        lo_x = max(1200.0, lp_crossover_hz - 700.0)
        hi_x = lp_crossover_hz + 700.0
        u_x = np.clip((freqs - lo_x) / (hi_x - lo_x), 0.0, 1.0)
        w_hf = 0.5 * (1.0 - np.cos(math.pi * u_x))
        hybrid_phasor = (1.0 - w_hf) * phi_glottal + w_hf * phi_disp
        hybrid_phasor /= np.maximum(np.abs(hybrid_phasor), 1e-8)
    else:
        hybrid_phasor = phi_disp

    # Precompute FFTs of LP-residual grains strictly for per-frame lp_psola ablation
    lp_specs = None
    bank_specs = None
    if mode == "lp_psola":
        if lp_pulses is not None and len(lp_pulses) == n_fr:
            lp_specs = rfft(lp_pulses.astype(np.float64), n=n_fft, axis=1)
        elif glottal_bank is not None and len(glottal_bank.pulses) > 0:
            bank_specs = rfft(glottal_bank.pulses.astype(np.float64), n=n_fft, axis=1)
        w_lp = np.ones(len(freqs), dtype=np.float64)
        w_lp[freqs < 65.0] = (freqs[freqs < 65.0] / 65.0) ** 2
    else:
        w_lp = None

    # Precompute voiced sample gate mask so harmonic tails NEVER ring 25 ms into unvoiced consonants!
    voiced_sample_gate = np.ones(n + n_fft, dtype=np.float64)
    if crisp_unvoiced:
        # Smoothly damp voiced excitation outside voiced spans (5 ms release taper at voiced->unvoiced edges)
        v_smp = np.repeat(base_feats.voiced.astype(np.float64), hop)
        if len(v_smp) < len(voiced_sample_gate):
            v_smp = np.pad(v_smp, (0, len(voiced_sample_gate) - len(v_smp)), constant_values=0.0)
        else:
            v_smp = v_smp[: len(voiced_sample_gate)]
        # Light 5 ms cosine edge smoothing (120 samples) so no click at V<->UV transitions while blocking 30 ms LF tail!
        k_sm = max(48, min(144, hop))
        kernel = np.hanning(k_sm + 2)[1:-1]
        kernel /= np.sum(kernel)
        voiced_sample_gate = np.convolve(v_smp, kernel, mode="same")

    y_harm = np.zeros(n + n_fft, dtype=np.float64)
    w_ang = math.tau * freqs / sr
    pulse_offset = n_fft // 4
    hf_rand_scale = np.clip((freqs - 2800.0) / 5000.0, 0.0, 1.0) * 0.45

    t_cur = 0.0
    jit_state = 0.0
    shim_state = 0.0
    pulse_idx = 0

    while t_cur < n:
        f_pos = min(float(n_fr - 1), max(0.0, t_cur / hop))
        f0_i = int(math.floor(f_pos))
        f1_i = min(n_fr - 1, f0_i + 1)
        w1 = f_pos - f0_i
        w0 = 1.0 - w1

        fi_round = min(n_fr - 1, max(0, int(round(f_pos))))
        f0 = float(base_feats.f0_hz[fi_round])
        if base_feats.voiced[fi_round] and f0 > 0.0:
            if base_feats.voiced[f0_i] and base_feats.voiced[f1_i] and base_feats.f0_hz[f0_i] > 0.0 and base_feats.f0_hz[f1_i] > 0.0:
                f0 = w0 * float(base_feats.f0_hz[f0_i]) + w1 * float(base_feats.f0_hz[f1_i])
                h_env_f = w0 * H_min[f0_i] + w1 * H_min[f1_i]
                h_disp_f = w0 * H_harm_base[f0_i] + w1 * H_harm_base[f1_i]
                hg = w0 * harm_gain[f0_i] + w1 * harm_gain[f1_i]
            else:
                h_env_f = H_min[fi_round]
                h_disp_f = H_harm_base[fi_round]
                hg = harm_gain[fi_round]

            # Reduced synthetic jitter when real LP-residual already provides natural human cycle micro-variation
            j_scale = f0_jitter_frac * (0.45 if (lp_specs is not None or bank_specs is not None) and mode != "hnm_v11" else 1.0)
            s_scale = amp_shimmer_frac * (0.45 if (lp_specs is not None or bank_specs is not None) and mode != "hnm_v11" else 1.0)
            jit_state = 0.35 * jit_state + 0.65 * float(rng.standard_normal()) * j_scale
            shim_state = 0.45 * shim_state + 0.55 * float(rng.standard_normal()) * s_scale
            T0_nom = sr / f0
            T0 = max(sr / 460.0, min(sr / 60.0, T0_nom * (1.0 + jit_state)))
            amp_mul = max(0.92, min(1.08, 1.0 + shim_state))

            idx_int = int(math.floor(t_cur))
            frac = t_cur - idx_int
            delay_phasor = np.exp(-1j * w_ang * frac)

            if mode == "hnm_v11":
                phi_jit = np.exp(1j * (rng.uniform(-math.pi, math.pi, size=len(freqs)) * hf_rand_scale * ap_full[fi_round])) if phase_dispersion else 1.0
                spec_h = h_disp_f * hg * delay_phasor * phi_jit
            elif mode == "hybrid_lp_hnm":
                phi_jit = np.exp(1j * (rng.uniform(-math.pi, math.pi, size=len(freqs)) * hf_rand_scale * ap_full[fi_round])) if phase_dispersion else 1.0
                spec_h = h_env_f * hybrid_phasor * hg * delay_phasor * phi_jit
            else:  # lp_psola per-frame OLA ablation
                if lp_specs is not None:
                    e_spec = lp_specs[fi_round]
                    if float(np.sum(np.abs(e_spec) ** 2)) < 1e-6:
                        e_spec = np.ones(len(freqs), dtype=np.complex128)
                elif bank_specs is not None:
                    k_b = (pulse_idx + int(rng.integers(0, min(3, len(bank_specs))))) % len(bank_specs)
                    e_spec = bank_specs[k_b]
                else:
                    e_spec = np.ones(len(freqs), dtype=np.complex128)
                spec_h = h_env_f * e_spec * w_lp * hg * delay_phasor

            pulse = np.roll(irfft(spec_h, n=n_fft), pulse_offset) * (math.sqrt(T0_nom) * amp_mul)
            start = idx_int - pulse_offset
            if start >= 0 and start + n_fft <= len(y_harm):
                y_harm[start : start + n_fft] += pulse
            elif start < 0 and start + n_fft > 0:
                y_harm[0 : start + n_fft] += pulse[-start:]
            t_cur += T0
            pulse_idx += 1
        else:
            jit_state = 0.0
            shim_state = 0.0
            t_cur += hop

    if crisp_unvoiced:
        y_harm *= voiced_sample_gate

    # Smooth 5 ms COLA noise excitation (with exact unsmoothed H_min on unvoiced frames!)
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

    if unvoiced_residual_spans:
        hp_sos = _hp_sos(sr, residual_highpass_hz)
        fade_n = max(24, min(72, int(round(0.003 * sr))))
        for fr_s, fr_e, raw_w in unvoiced_residual_spans:
            fr_s_c = max(0, min(n_fr, int(fr_s)))
            fr_e_c = max(fr_s_c, min(n_fr, int(fr_e)))
            if fr_e_c <= fr_s_c or np.any(base_feats.voiced[fr_s_c:fr_e_c]):
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
