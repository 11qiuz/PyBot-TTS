"""Experimental v5 source-filter synthesizer with LTAS tilt, LF glottal pulse, HNR, dynamic F0, and pole-tracked trajectories."""
from __future__ import annotations

import math
from pathlib import Path
import random
import sys

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter

from exp_v5_timbre import SR
from exp_v5_timbre.features import extract_ltas

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pybot_tts_v2_recorded import engine, synth as _synth  # noqa: E402
from pybot_tts_v4_params import vocal  # noqa: E402


# Reference IQR of engine.CURVE tone contours (~3.2 semitones)
REF_CURVE_IQR_ST = 3.2


def build_units_from_utterance(utt: dict, use_real_durations: bool = True) -> list[dict]:
    """Convert an extended/index.json utterance record into vocal-compatible units."""
    units = []
    syls = utt["syllables"]
    for idx, syl in enumerate(syls):
        o, f, t = engine.split_pinyin(syl["pinyin"])
        dur_ms = float(syl["ms"]) if use_real_durations else 260.0
        dur_ms = max(70.0, min(1500.0, dur_ms))
        pause_ms = 0.0
        if idx + 1 < len(syls):
            gap = (syls[idx + 1]["start_sample"] - syl["stop_sample"]) / SR * 1000.0
            if gap > 5.0:
                pause_ms = min(2000.0, gap)
        units.append({
            "hanzi": syl.get("hanzi", ""),
            "onset": o,
            "final": f,
            "tone": t,
            "duration_ms": dur_ms,
            "gain": 0.83,
            "pitch_st": list(engine.CURVE[t]),
            "pause_after_ms": pause_ms,
        })
    return units


def apply_tilt_filter(
    sig: np.ndarray,
    target_tilt_db_per_oct: float,
    sr: int = SR,
    f_low: float = 80.0,
    f_high: float = 8000.0,
    f_pivot: float = 1000.0,
) -> np.ndarray:
    """Shape signal spectrum in dB/oct so its active LTAS tilt matches target_tilt_db_per_oct."""
    x = np.asarray(sig, dtype=np.float64)
    if x.size < 64:
        return x.copy()
    lt = extract_ltas(x, sr=sr, low_hz=f_low, high_hz=f_high)
    current_tilt = lt["tilt_db_per_oct"]
    delta_slope = float(target_tilt_db_per_oct) - float(current_tilt)
    if abs(delta_slope) < 0.02:
        return x.copy()

    n_fft = 1 << int(math.ceil(math.log2(max(2048, len(x)))))
    freqs = rfftfreq(n_fft, 1.0 / sr)
    f_clipped = np.clip(freqs, f_low, f_high)
    # delta_db = delta_slope * log2(f / f_pivot)
    delta_db = delta_slope * np.log2(f_clipped / f_pivot)
    # Taper gently above f_high so 8-12 kHz doesn't boost noise excessively
    hi_mask = freqs > f_high
    if np.any(hi_mask):
        taper = np.maximum(0.0, 1.0 - (freqs[hi_mask] - f_high) / max(1.0, (sr * 0.5 - f_high)))
        delta_db[hi_mask] *= taper
    gain_mag = np.power(10.0, delta_db / 20.0)
    gain_mag[0] = 1.0

    spec = rfft(x, n=n_fft)
    y = irfft(spec * gain_mag, n=n_fft)[: len(x)]
    # Preserve original RMS on active region
    in_rms = float(np.sqrt(np.mean(x * x)))
    out_rms = float(np.sqrt(np.mean(y * y)))
    if out_rms > 1e-12 and in_rms > 1e-12:
        y = y * (in_rms / out_rms)
    return y


def _biquad_resonator(f_hz: float, bw_hz: float, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """2nd-order digital resonator (Klatt-style all-pole with unity DC gain)."""
    f = max(40.0, min(sr * 0.48, float(f_hz)))
    bw = max(20.0, min(2000.0, float(bw_hz)))
    r = math.exp(-math.pi * bw / sr)
    theta = math.tau * f / sr
    a1 = -2.0 * r * math.cos(theta)
    a2 = r * r
    b0 = 1.0 + a1 + a2  # unity gain at DC
    return np.array([b0, 0.0, 0.0], dtype=np.float64), np.array([1.0, a1, a2], dtype=np.float64)


def _rosenberg_lf_derivative(phase: np.ndarray, rd: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Parameterized Rosenberg/LF-like glottal flow derivative and open-phase mask.

    ``rd`` (Fant 1995): ~0.7 pressed/bright, ~1.0 modal, ~1.5 breathy/steep tilt.
    Returns ``(g_prime, open_gate)`` normalized to unit RMS per voiced sample.
    """
    rd = max(0.5, min(2.2, float(rd)))
    oq = min(0.85, max(0.42, 0.48 + 0.14 * rd))
    tp = 0.68 * oq
    # Rosenberg C flow g(phi) -> analytic derivative with controllable return steepness
    p = np.mod(phase, 1.0)
    g_p = np.zeros_like(p)
    open_mask = p < oq

    # Rising/opening phase [0, tp]: smooth half-sine-like polynomial
    m1 = p < tp
    u1 = p[m1] / tp
    g_p[m1] = np.sin(math.pi * u1)

    # Closing phase [tp, oq]: sharper negative closure impulse controlled by 1/rd
    m2 = (p >= tp) & (p < oq)
    u2 = (p[m2] - tp) / max(1e-4, oq - tp)
    sharp = 1.0 / (0.45 + 0.55 * rd)
    g_p[m2] = -sharp * 1.4 * np.sin(0.5 * math.pi * u2) ** 1.2

    # Return phase [oq, 1): exponential tail toward 0
    m3 = p >= oq
    tau_r = max(0.015, 0.035 * rd)
    g_p[m3] = -sharp * 0.45 * np.exp(-(p[m3] - oq) / tau_r)

    # Remove DC offset
    g_p = g_p - np.mean(g_p)
    rms = float(np.sqrt(np.mean(g_p * g_p)))
    if rms > 1e-12:
        g_p = g_p / rms
    open_gate = np.where(open_mask, 1.0, 0.25)
    return g_p, open_gate


def _cascade_filter_time_varying(
    exc: np.ndarray,
    f1_track: np.ndarray,
    f2_track: np.ndarray,
    f3_track: np.ndarray,
    bw1_track: np.ndarray,
    bw2_track: np.ndarray,
    nasal_track: np.ndarray,
    sr: int = SR,
    block_samples: int = 48,
) -> np.ndarray:
    """Pass excitation through 5-formant cascade biquad filterbank in 2ms blocks with state continuity."""
    n = len(exc)
    y = exc.copy()
    f4, bw4 = 3750.0, 500.0
    f5, bw5 = 4850.0, 700.0

    # Fixed high formants F4, F5 across entire signal
    b4, a4 = _biquad_resonator(f4, bw4, sr)
    b5, a5 = _biquad_resonator(f5, bw5, sr)
    y = lfilter(b4, a4, y)
    y = lfilter(b5, a5, y)

    # Time-varying F1, F2, F3 + mild nasal pole-zero
    zi1 = np.zeros(2, dtype=np.float64)
    zi2 = np.zeros(2, dtype=np.float64)
    zi3 = np.zeros(2, dtype=np.float64)
    zin = np.zeros(2, dtype=np.float64)

    out = np.empty(n, dtype=np.float64)
    for s in range(0, n, block_samples):
        e = min(n, s + block_samples)
        mid = (s + e) // 2
        b1, a1 = _biquad_resonator(f1_track[mid], bw1_track[mid], sr)
        b2, a2 = _biquad_resonator(f2_track[mid], bw2_track[mid], sr)
        b3, a3 = _biquad_resonator(f3_track[mid], 220.0, sr)

        blk = y[s:e]
        blk, zi1 = lfilter(b1, a1, blk, zi=zi1)
        blk, zi2 = lfilter(b2, a2, blk, zi=zi2)
        blk, zi3 = lfilter(b3, a3, blk, zi=zi3)

        nas = float(nasal_track[mid])
        if nas > 0.02:
            # Nasalization: low-frequency pole ~280 Hz + mild attenuation above 700 Hz
            bn, an = _biquad_resonator(280.0, 95.0, sr)
            blk_n, zin = lfilter(bn, an, blk, zi=zin)
            blk = (1.0 - 0.55 * nas) * blk + (0.55 * nas) * blk_n
        else:
            # Keep zin decayed cleanly
            _, zin = lfilter([1.0], [1.0, 0.0, 0.0], blk * 0.0, zi=zin)
        out[s:e] = blk
    return out


def _render_consonant_v5(
    onset: str,
    c_dur: float,
    rng: np.random.Generator,
    gain: float,
    sr: int = SR,
    brightness_override: float | None = None,
) -> np.ndarray:
    """Render voiceless consonant (fricative/stop) with shaped band noise."""
    n = int(round(c_dur * sr))
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    noise = rng.standard_normal(n)
    # 1-pole LP and 1-diff HP
    lp = lfilter([0.30], [1.0, -0.70], noise)
    hp = np.empty_like(noise)
    hp[0] = noise[0]
    hp[1:] = noise[1:] - noise[:-1]

    if onset in _synth.FRIC:
        amp, bright = _synth.FRIC[onset]
        if brightness_override is not None:
            bright = brightness_override
        env = np.minimum(1.0, np.minimum(t * 5.0, (1.0 - t) * 4.0)) * amp
    else:
        _, amp = _synth.STOP.get(onset, (0.04, 0.15))
        bright = 0.47 if onset in ("z", "c", "zh", "ch", "j", "q", "t", "d") else 0.21
        if brightness_override is not None:
            bright = brightness_override
        env = np.where(t < 0.27, 0.0, amp * np.maximum(0.0, 1.0 - t) ** 0.40)
        burst = (t >= 0.27) & (t <= 0.36)
        env[burst] = np.minimum(0.75, env[burst] + 0.20)

    return gain * env * (bright * hp + (1.0 - bright) * lp) * 0.38


def _plan_unit_f12(unit: dict, phoneme_params: dict, profile: dict | None, use_trajectories: bool) -> np.ndarray:
    """Return (5, 4) trajectory [F1, F2, BW1, BW2] for this unit's final."""
    final = unit["final"]
    if use_trajectories and profile is not None:
        traj = profile.get("final_trajectories", {}).get(final)
        if traj is not None:
            return np.array(traj, dtype=np.float64)

    # Fallback: build 5-point trajectory from phoneme_params / synth.V segments
    segs = _synth.FINALS[final].split()
    pts = []
    for sk in segs:
        entry = vocal._measured_f12(phoneme_params, sk, sk)
        if entry is not None and entry.get("f12"):
            f1, f2 = entry["f12"]
            bws = entry.get("formant_bw12") or [100.0, 140.0]
            pts.append([float(f1), float(f2), float(bws[0]), float(bws[1])])
        else:
            f1, f2, _f3 = _synth.V[sk]
            pts.append([float(f1), float(f2), 100.0, 140.0])
    pts_arr = np.array(pts, dtype=np.float64)
    u_src = np.linspace(0.0, 1.0, len(pts_arr)) if len(pts_arr) > 1 else np.array([0.0])
    u_dst = np.linspace(0.0, 1.0, 5)
    if len(pts_arr) == 1:
        return np.repeat(pts_arr, 5, axis=0)
    return np.column_stack([np.interp(u_dst, u_src, pts_arr[:, c]) for c in range(4)])


def _synthesize_tilt_f0_v4_style(
    units: list[dict],
    phoneme_params_spk: dict,
    profile: dict,
    seed: int,
    speaker: str | None,
) -> np.ndarray:
    """Variant B (v5_tilt_f0): v4 harmonic synthesis with speaker F0 median/range + LTAS tilt EQ."""
    # Re-use vocal.py's phoneme resolution, but drive F0 with speaker median & IQR scale
    params = vocal._coerce_params(phoneme_params_spk)
    scale = 1.0
    if speaker and speaker in params["speakers"]:
        scale = params["speakers"][speaker]["scale"]

    f0_base = float(profile["f0_median_hz"])
    range_scale = max(0.55, min(1.8, float(profile["f0_iqr_st"]) / REF_CURVE_IQR_ST))
    all_p = [float(x) for u in units for x in u["pitch_st"]]
    utt_mu_st = float(np.mean(all_p)) if all_p else 0.0
    rng = random.Random(seed)
    out = [0.0] * int(0.075 * SR)

    for u_idx, u in enumerate(units):
        onset = u["onset"]
        final = u["final"]
        pitch = [utt_mu_st + (float(x) - utt_mu_st) * range_scale for x in u["pitch_st"]]
        duration = float(u["duration_ms"]) / 1000.0
        gain = float(u["gain"])
        pause = float(u.get("pause_after_ms", 0.0)) / 1000.0
        if onset in ("y", "w"):
            onset = ""
        if final == "i" and onset in ("zh", "ch", "sh", "r", "z", "c", "s"):
            final = "ir" if onset in ("zh", "ch", "sh", "r") else "iz"

        plan, track_fn = vocal.plan_ctd(params, final)
        final_dur, c_dur = vocal._split_duration(onset, duration)
        brightness = vocal._brightness_for(params, onset, plan)
        track = vocal._scaled_track(track_fn, scale)
        bws = [s["bandwidths"] for s in plan["segments"] if s["bandwidths"]]
        bandwidths = [bws[0][0], bws[0][1], vocal.BASELINE_BW[2]] if len(bws) == len(plan["segments"]) and bws else None

        c = vocal._consonant(onset, c_dur, pitch, rng, gain, brightness=brightness, plan_on=True)

        # Render vowel with speaker F0 base + range-scaled pitch
        rime_dur = final_dur - c_dur
        n = max(1, int(rime_dur * SR))
        keys = tuple(_synth.FINALS[final].split())
        phase = 0.0
        weights = ()
        v = []
        decl = 0.6 - 1.2 * (u_idx / max(1, len(units) - 1)) if len(units) > 1 else 0.0
        for i in range(n):
            t = i / max(1, n - 1)
            st = _synth.interp(pitch, t) + decl + 0.12 * math.sin(t * math.tau * 1.7)
            f0 = f0_base * (2.0 ** (st / 12.0))
            if i % 72 == 0:
                nasality = max(0.0, (t - 0.62) / 0.38) if keys[-1] == "nasal" else 0.0
                fs = [track(t, k) for k in range(3)]
                weights = vocal._weights_full(f0, fs, bandwidths, nasality)
            phase = (phase + math.tau * f0 / SR) % math.tau
            pressure = sum(w * math.sin(h * phase + 0.39 * h) for h, w in enumerate(weights, 1))
            env = min(1.0, i / (SR * 0.017), (n - 1 - i) / (SR * 0.032))
            flutter = 0.95 + 0.045 * math.sin(t * math.tau)
            v.append(gain * env * flutter * (0.61 * pressure + 0.014 * rng.uniform(-1, 1)))

        overlap = min(len(c), len(v), int(0.005 * SR))
        if overlap:
            c[-overlap:] = [c[-overlap + i] + v[i] for i in range(overlap)]
        out += c + v[overlap:]
        out.extend([0.0] * round(pause * SR))
    out.extend([0.0] * int(0.100 * SR))

    arr = np.array(out, dtype=np.float64)
    return apply_tilt_filter(arr, profile["tilt_db_per_oct"], sr=SR)


def _synthesize_cascade_v5(
    units: list[dict],
    phoneme_params_norm: dict,
    profile: dict,
    seed: int,
    use_trajectories: bool,
    use_dynamic_f0: bool,
    oracle_f0: np.ndarray | None = None,
    oracle_traj: dict | None = None,
) -> np.ndarray:
    """Render Variants C (v5_glottal_hnr), D (v5_full), and E (v5_oracle_prosody)."""
    rng = np.random.default_rng(seed)

    # First lay out exact sample timeline matching v4 length convention
    lead_n = int(0.075 * SR)
    tail_n = int(0.100 * SR)
    unit_spans = []
    cursor = lead_n
    for u in units:
        onset = u["onset"]
        final = u["final"]
        if onset in ("y", "w"):
            onset = ""
        if final == "i" and onset in ("zh", "ch", "sh", "r", "z", "c", "s"):
            final = "ir" if onset in ("zh", "ch", "sh", "r") else "iz"
        dur = float(u["duration_ms"]) / 1000.0
        pause = float(u.get("pause_after_ms", 0.0)) / 1000.0
        final_dur, c_dur = vocal._split_duration(onset, dur)
        c_n = int(c_dur * SR)
        v_n = max(1, int((final_dur - c_dur) * SR))
        overlap = min(c_n, v_n, int(0.005 * SR))
        p_n = round(pause * SR)
        total_u = c_n + v_n - overlap
        unit_spans.append({
            "onset": onset,
            "final": final,
            "tone": int(u["tone"]),
            "pitch_st": [float(x) for x in u["pitch_st"]],
            "gain": float(u["gain"]),
            "c_n": c_n,
            "v_n": v_n,
            "overlap": overlap,
            "p_n": p_n,
            "start": cursor,
            "v_start": cursor + c_n - overlap,
            "end": cursor + total_u,
        })
        cursor += total_u + p_n
    total_n = cursor + tail_n

    # Build global sample-level tracks for voiced source-filter synthesis
    f0_track = np.full(total_n, float(profile["f0_median_hz"]), dtype=np.float64)
    f1_track = np.full(total_n, 500.0, dtype=np.float64)
    f2_track = np.full(total_n, 1500.0, dtype=np.float64)
    f3_track = np.full(total_n, 2650.0, dtype=np.float64)
    bw1_track = np.full(total_n, 100.0, dtype=np.float64)
    bw2_track = np.full(total_n, 140.0, dtype=np.float64)
    nasal_track = np.zeros(total_n, dtype=np.float64)
    voiced_env = np.zeros(total_n, dtype=np.float64)
    unvoiced_audio = np.zeros(total_n, dtype=np.float64)

    f0_base = float(profile["f0_median_hz"])
    range_scale = max(0.55, min(1.8, float(profile["f0_iqr_st"]) / REF_CURVE_IQR_ST))
    all_p = [float(x) for sp in unit_spans for x in sp["pitch_st"]]
    utt_mu_st = float(np.mean(all_p)) if all_p else 0.0
    u_grid = np.linspace(0.0, 1.0, 5)

    for idx, sp in enumerate(unit_spans):
        onset = sp["onset"]
        final = sp["final"]
        s0, vs, e0 = sp["start"], sp["v_start"], sp["end"]
        c_n, v_n = sp["c_n"], sp["v_n"]

        # Voiceless consonant onset vs voiced sonorant onset
        if onset in _synth.NASAL_SOUND:
            # Voiced pre-onset sonorant fills [s0, vs + overlap]
            son_end = vs + sp["overlap"]
            if son_end > s0:
                voiced_env[s0:son_end] = sp["gain"] * 0.45
                nasal_track[s0:son_end] = 0.8 if onset in _synth.NASALS else 0.1
                f1_track[s0:son_end] = 300.0
                f2_track[s0:son_end] = 1200.0
                f3_track[s0:son_end] = 2400.0
                bw1_track[s0:son_end] = 95.0
                bw2_track[s0:son_end] = 140.0
                pitch_s = utt_mu_st + (np.array(sp["pitch_st"]) - utt_mu_st) * range_scale
                f0_track[s0:son_end] = f0_base * (2.0 ** (pitch_s[0] / 12.0))
        elif c_n > 0:
            c_wav = _render_consonant_v5(onset, c_n / SR, rng, sp["gain"], sr=SR)
            unvoiced_audio[s0 : s0 + len(c_wav)] += c_wav

        # Vowel / rime trajectory
        traj_5 = _plan_unit_f12({"final": final}, phoneme_params_norm, profile, use_trajectories)
        t_v = np.linspace(0.0, 1.0, v_n)
        f1_v = np.interp(t_v, u_grid, traj_5[:, 0])
        f2_v = np.interp(t_v, u_grid, traj_5[:, 1])
        bw1_v = np.clip(np.interp(t_v, u_grid, traj_5[:, 2]), 40.0, 450.0)
        bw2_v = np.clip(np.interp(t_v, u_grid, traj_5[:, 3]), 50.0, 550.0)
        f3_v = np.clip(2650.0 + 0.45 * (f2_v - 1500.0), 2100.0, 3600.0)

        f1_track[vs:e0] = f1_v
        f2_track[vs:e0] = f2_v
        f3_track[vs:e0] = f3_v
        bw1_track[vs:e0] = bw1_v
        bw2_track[vs:e0] = bw2_v

        keys = _synth.FINALS[final].split()
        if keys[-1] == "nasal":
            nasal_track[vs:e0] = np.maximum(0.0, (t_v - 0.55) / 0.45)

        # Amplitude envelope with smooth rise/fall
        rise = np.minimum(1.0, np.arange(v_n) / (SR * 0.015))
        fall = np.minimum(1.0, (v_n - 1 - np.arange(v_n)) / (SR * 0.025))
        flutter = 0.96 + 0.04 * np.sin(t_v * math.tau)
        env_v = sp["gain"] * np.minimum(rise, fall) * flutter
        voiced_env[vs:e0] = np.maximum(voiced_env[vs:e0], env_v)

        # F0 contour for rime
        decl = (0.75 - 1.5 * (idx / max(1, len(unit_spans) - 1))) if (use_dynamic_f0 and len(unit_spans) > 1) else 0.0
        p_st = utt_mu_st + (np.array(sp["pitch_st"], dtype=np.float64) - utt_mu_st) * range_scale + decl
        p_grid = np.linspace(0.0, 1.0, len(p_st))
        st_v = np.interp(t_v, p_grid, p_st) + 0.10 * np.sin(t_v * math.tau * 1.7)
        f0_track[vs:e0] = f0_base * (2.0 ** (st_v / 12.0))

    # Smooth F0 & formant tracks across voiced transitions when dynamic F0 is enabled
    if use_dynamic_f0:
        # Smooth F0 in log-Hz space across voiced transitions (sigma ~ 25 ms = 600 samples)
        log_f0 = np.log2(np.maximum(60.0, f0_track))
        log_f0 = gaussian_filter1d(log_f0, sigma=480.0)
        f0_track = 2.0 ** log_f0
        # Subtle slow prosodic micro-jitter (~0.4% RMS)
        drift = gaussian_filter1d(rng.standard_normal(total_n), sigma=300.0)
        drift = drift / max(1e-6, float(np.std(drift))) * 0.004
        f0_track = f0_track * (1.0 + drift)

    # Diagnostic oracle prosody override (pure source-filter driven by real extracted F0 + F1/F2)
    if oracle_f0 is not None:
        n_copy = min(total_n - lead_n - tail_n, len(oracle_f0))
        if n_copy > 0:
            f0_track[lead_n : lead_n + n_copy] = np.clip(oracle_f0[:n_copy], 75.0, 400.0)
    if oracle_traj is not None:
        for k, arr in (("f1", f1_track), ("f2", f2_track), ("f3", f3_track), ("bw1", bw1_track), ("bw2", bw2_track)):
            if k in oracle_traj and len(oracle_traj[k]) > 0:
                n_copy = min(total_n - lead_n - tail_n, len(oracle_traj[k]))
                arr[lead_n : lead_n + n_copy] = oracle_traj[k][:n_copy]

    # Smooth formant tracks slightly to avoid abrupt block jumps at syllable boundaries
    f1_track = gaussian_filter1d(f1_track, sigma=180.0)
    f2_track = gaussian_filter1d(f2_track, sigma=180.0)
    f3_track = gaussian_filter1d(f3_track, sigma=180.0)

    # Generate continuous glottal phase and LF pulse train
    phase = np.cumsum(f0_track / SR)
    # Rd calibrated from speaker tilt/HNR: brighter/higher HNR -> lower Rd (~0.85), breathier -> higher Rd (~1.45)
    tilt_val = float(profile.get("tilt_db_per_oct", -3.5))
    hnr_val = float(profile.get("hnr_median_db", 12.0))
    rd = np.clip(1.0 + 0.18 * (-3.5 - tilt_val) + 0.08 * (12.0 - hnr_val), 0.65, 1.85)
    glottal, open_gate = _rosenberg_lf_derivative(phase, rd=float(rd))

    # Aspiration noise calibrated to speaker HNR
    raw_asp = rng.standard_normal(total_n)
    # Gentle breathiness shaping (mild high-pass emphasis)
    asp = np.empty_like(raw_asp)
    asp[0] = raw_asp[0]
    asp[1:] = raw_asp[1:] - 0.35 * raw_asp[:-1]
    asp = asp / max(1e-8, float(np.std(asp))) * open_gate
    # Mix ratio from HNR (with resonator-compensation offset because all-pole ringing boosts ACF corr)
    hnr_eff_db = max(2.0, min(22.0, hnr_val - 5.5))
    noise_amp = 10.0 ** (-hnr_eff_db / 20.0)
    exc = (glottal + noise_amp * asp) * voiced_env

    voiced_out = _cascade_filter_time_varying(
        exc, f1_track, f2_track, f3_track, bw1_track, bw2_track, nasal_track, sr=SR
    )
    # Balance voiced RMS relative to unvoiced consonant energy
    v_active = voiced_out[voiced_env > 0.05]
    if v_active.size > 0:
        v_rms = float(np.sqrt(np.mean(v_active * v_active)))
        if v_rms > 1e-12:
            voiced_out = voiced_out * (0.18 / v_rms)

    mix = voiced_out + unvoiced_audio
    return apply_tilt_filter(mix, profile["tilt_db_per_oct"], sr=SR)


def synthesize_variant(
    variant: str,
    units: list[dict],
    phoneme_params_norm: dict,
    phoneme_params_spk: dict,
    profile: dict,
    speaker: str,
    seed: int = 72,
    oracle_f0: np.ndarray | None = None,
    oracle_traj: dict | None = None,
) -> np.ndarray:
    """Dispatch synthesis across the Phase 1 ablation ladder."""
    if variant == "v4_baseline":
        wav = vocal.render(units, params=phoneme_params_spk, seed=seed, speaker=speaker)
        return np.array(wav, dtype=np.float64)
    if variant == "v5_tilt_f0":
        return _synthesize_tilt_f0_v4_style(units, phoneme_params_spk, profile, seed=seed, speaker=speaker)
    if variant == "v5_glottal_hnr":
        return _synthesize_cascade_v5(
            units, phoneme_params_norm, profile, seed=seed, use_trajectories=False, use_dynamic_f0=False
        )
    if variant == "v5_full":
        return _synthesize_cascade_v5(
            units, phoneme_params_norm, profile, seed=seed, use_trajectories=True, use_dynamic_f0=True
        )
    if variant == "v5_oracle_prosody":
        return _synthesize_cascade_v5(
            units,
            phoneme_params_norm,
            profile,
            seed=seed,
            use_trajectories=True,
            use_dynamic_f0=True,
            oracle_f0=oracle_f0,
            oracle_traj=oracle_traj,
        )
    raise ValueError(f"Unknown variant: {variant!r}")
