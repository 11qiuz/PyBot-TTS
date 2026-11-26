"""Acoustic Landmark Boundary Refiner for Mandarin and English 5 ms UTAU Token Banks."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.fft import rfft, rfftfreq

from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS
from exp_v6_vocoder import vocoder
from exp_v8_clarity.utau_concat import OBSTRUENT_IPAS, OBSTRUENT_ONSETS
from exp_v9_clarity_pro import HOP, N_FFT, SR


def compute_landmark_tracks(
    x_act: np.ndarray,
    feats: vocoder.VocoderFeatures,
    sr: int = SR,
) -> dict[str, np.ndarray]:
    """Compute 5 ms acoustic landmark cues: HF/LF, >8k ratio, 2-8 kHz flux onset, and periodicity cliff."""
    n_fr = len(feats.log_env)
    n_fft = feats.n_fft
    freqs = rfftfreq(n_fft, 1.0 / sr)

    lf_m = (freqs >= 80.0) & (freqs <= 1000.0)
    hf_m = (freqs >= 2000.0) & (freqs <= 8000.0)
    sib_m = (freqs >= 3500.0) & (freqs <= 11500.0)

    env_pwr = np.exp(2.0 * feats.log_env)
    lf_db = 10.0 * np.log10(np.maximum(np.mean(env_pwr[:, lf_m], axis=1), 1e-16))
    hf_db = 10.0 * np.log10(np.maximum(np.mean(env_pwr[:, hf_m], axis=1), 1e-16))
    sib_db = 10.0 * np.log10(np.maximum(np.mean(env_pwr[:, sib_m], axis=1), 1e-16))
    tot_db = 10.0 * np.log10(np.maximum(np.mean(env_pwr, axis=1), 1e-16))
    tot_peak = float(np.percentile(tot_db, 95)) if tot_db.size else 0.0

    # Fine 2.5 ms half-wave rectified 2-8 kHz waveform spectral flux downsampled to 5 ms max
    x = np.asarray(x_act, dtype=np.float64)
    win_f = 192  # 8 ms short window for sharp burst localization
    hop_f = 60   # 2.5 ms sub-hop
    n_sub = max(1, 1 + max(0, len(x) - win_f) // hop_f)
    pad_len = (n_sub - 1) * hop_f + win_f
    xp = np.pad(x, (0, max(0, pad_len - len(x))))
    idx = np.arange(win_f)[None, :] + np.arange(n_sub)[:, None] * hop_f
    w_han = np.hanning(win_f)[None, :]
    spec_sub = np.abs(rfft(xp[idx] * w_han, n=512, axis=1))
    f_sub = rfftfreq(512, 1.0 / sr)
    hf_sub_m = (f_sub >= 2000.0) & (f_sub <= 8000.0)
    log_hf_sub = np.log1p(spec_sub[:, hf_sub_m] * 50.0)
    if len(log_hf_sub) > 1:
        diff_hf = np.maximum(0.0, np.diff(log_hf_sub, axis=0, prepend=log_hf_sub[:1]))
        flux_sub = np.mean(diff_hf, axis=1)
    else:
        flux_sub = np.zeros(len(log_hf_sub), dtype=np.float64)

    hf_flux = np.zeros(n_fr, dtype=np.float64)
    for t in range(n_fr):
        s_i = min(len(flux_sub) - 1, t * 2)
        e_i = min(len(flux_sub), s_i + 2)
        hf_flux[t] = float(np.max(flux_sub[s_i:e_i])) if e_i > s_i else float(flux_sub[s_i])

    # Normalize flux by 90th percentile for stable DP weighting across speaker volumes
    f_scale = float(np.percentile(hf_flux, 90)) if hf_flux.size else 1.0
    hf_flux_norm = hf_flux / max(1e-4, f_scale)

    voi = feats.voiced.astype(np.float64)
    # Periodicity cliff magnitude at frame t: |voi[t] - voi[t-1]| + weighted |delta HF/LF|
    d_voi = np.abs(np.diff(voi, prepend=voi[:1]))
    hf_lf = hf_db - lf_db
    d_hflf = np.abs(np.diff(hf_lf, prepend=hf_lf[:1]))
    cliff = 1.5 * d_voi + 0.12 * np.clip(d_hflf, 0.0, 20.0) + 0.6 * np.clip(hf_flux_norm, 0.0, 3.0)

    return {
        "lf_db": lf_db,
        "hf_db": hf_db,
        "hf_lf": hf_lf,
        "sib_lf": sib_db - lf_db,
        "rel_db": tot_db - tot_peak,
        "hf_flux": hf_flux_norm,
        "voi": voi,
        "cliff": cliff,
    }


def refine_zh_syllable_bounds(
    tracks: dict[str, np.ndarray],
    f_lo: int,
    f_hi: int,
    onset: str,
    final: str,
    prop_split_fr: int,
) -> tuple[int, int, int]:
    """Refine Mandarin syllable [f_lo, f_hi) and onset/final split f_sp using acoustic landmarks.

    Fixes two root defects of energy-dip segmentation (`extended/index.json`):
    1) Preceding-vowel tail bleeding into the first 10-30 ms of obstruent onsets.
    2) Trailing silent/low-frequency room fade frames dragged into unvoiced edges.
    """
    n_tot = len(tracks["voi"])
    f_lo = max(0, min(n_tot - 2, int(f_lo)))
    f_hi = max(f_lo + 2, min(n_tot, int(f_hi)))
    voi = tracks["voi"]
    hf_lf = tracks["hf_lf"]
    rel_db = tracks["rel_db"]
    flux = tracks["hf_flux"]

    lo_new = f_lo
    hi_new = f_hi

    if onset in OBSTRUENT_ONSETS and (hi_new - lo_new) >= 6:
        # Advance past leading preceding-vowel tail bleed ONLY if an actual unvoiced/HF obstruent follows
        scan_lim = min(hi_new - 4, lo_new + 6)
        obs_start = None
        for idx in range(lo_new, min(hi_new - 3, lo_new + 8)):
            if voi[idx] < 0.5 or hf_lf[idx] > -2.5:
                obs_start = idx
                break
        if obs_start is not None and obs_start > lo_new:
            if all(voi[k] > 0.5 and hf_lf[k] < -3.0 for k in range(lo_new, obs_start)):
                lo_new = min(scan_lim, obs_start)
        # Trim leading dead silence/rumble (< -30 dB & no burst) leaving 2 frames closure if stop/affricate
        is_stop_or_aff = onset in ("b", "p", "d", "t", "g", "k", "z", "c", "zh", "ch", "j", "q")
        keep_pre = 2 if is_stop_or_aff else 0
        while (hi_new - lo_new) >= 5 and lo_new < scan_lim:
            if rel_db[lo_new] < -30.0 and flux[lo_new] < 0.6 and hf_lf[lo_new] < -4.0:
                if is_stop_or_aff and (obs_start is None or obs_start - lo_new <= keep_pre):
                    break
                lo_new += 1
            else:
                break

    # Trim trailing dead/unvoiced fade-out frames at syllable end that inject LF rumble
    while (hi_new - lo_new) >= 5:
        k = hi_new - 1
        if (voi[k] < 0.5 and hf_lf[k] < -5.0 and rel_db[k] < -20.0) or rel_db[k] < -34.0:
            hi_new -= 1
        else:
            break

    n_cur = hi_new - lo_new
    if not onset or n_cur < 3:
        return lo_new, hi_new, lo_new

    if onset in OBSTRUENT_ONSETS:
        # If quiet_dip_split already missed the consonant (syllable begins deeply voiced & LF-dominated), cap at 2 frames
        if voi[lo_new] > 0.5 and hf_lf[lo_new] < -4.0:
            return lo_new, hi_new, min(hi_new - 2, lo_new + min(2, max(1, prop_split_fr)))

        # Find periodicity cliff: first sustained voiced vowel frame after the unvoiced obstruent onset
        search_lo = lo_new + 1
        search_hi = min(hi_new - 2, lo_new + max(prop_split_fr + 3, int(round(0.45 * n_cur))))
        best_sp = None
        for idx in range(search_lo, search_hi):
            if voi[idx] > 0.5 and (idx + 1 < hi_new and voi[idx + 1] > 0.5) and hf_lf[idx] < 6.0:
                best_sp = idx
                break
        if best_sp is None:
            best_sp = min(search_hi, lo_new + max(1, prop_split_fr))
        return lo_new, hi_new, best_sp

    # Sonorant / glide onset: proportional split clamped safely
    sp_new = max(lo_new + 1, min(hi_new - 2, lo_new + prop_split_fr))
    return lo_new, hi_new, sp_new


def align_en_phones_with_landmarks(
    ipas: list[str],
    feats: vocoder.VocoderFeatures,
    x_act: np.ndarray,
    sr: int = SR,
    tracks: dict[str, np.ndarray] | None = None,
) -> list[tuple[int, int]]:
    """Landmark-guided monotonic DP forced aligner + +-2 frame periodicity-cliff snapping."""
    n_fr = len(feats.log_env)
    m_phones = len(ipas)
    if m_phones == 0 or n_fr < m_phones * 2:
        edges = np.linspace(0, n_fr, m_phones + 1).astype(int)
        return [(int(edges[i]), int(max(edges[i] + 1, edges[i + 1]))) for i in range(m_phones)]

    if tracks is None:
        tracks = compute_landmark_tracks(x_act, feats, sr=sr)

    voi = tracks["voi"]
    hf_lf = tracks["hf_lf"]
    sib_lf = tracks["sib_lf"]
    rel_db = tracks["rel_db"]
    flux = tracks["hf_flux"]
    cliff = tracks["cliff"]

    base_durs, manners, is_voi = [], [], []
    for p in ipas:
        sp = IPA_SPECS.get(p)
        base_durs.append(sp.base_dur_ms if sp else 75.0)
        manners.append(sp.manner if sp else "vowel")
        is_voi.append(bool(sp.voiced) if sp else True)
    scale = (n_fr * 5.0) / max(1.0, sum(base_durs))

    # Richer emission log-likelihoods separating sibilants, stops, voiced fricatives, sonorants, vowels
    score = np.zeros((m_phones, n_fr), dtype=np.float64)
    for i, p in enumerate(ipas):
        man = manners[i]
        v_exp = is_voi[i]
        if not v_exp and man in ("fricative", "affricate"):
            sib_bonus = 0.14 * np.clip(sib_lf, -10.0, 20.0) if p in ("s", "ʃ", "tʃ") else 0.08 * np.clip(hf_lf, -10.0, 16.0)
            burst = 0.28 * np.clip(flux, 0.0, 2.5) if man == "affricate" else 0.0
            score[i] = 2.1 * (1.0 - voi) + 0.20 * np.clip(hf_lf, -12.0, 18.0) + sib_bonus + burst - 0.14 * np.maximum(0.0, -rel_db - 28.0)
        elif not v_exp and man == "stop":
            score[i] = 1.9 * (1.0 - voi) + 0.10 * np.clip(hf_lf, -10.0, 16.0) + 0.35 * np.clip(flux, 0.0, 2.5)
        elif v_exp and man in ("fricative", "affricate", "stop"):
            score[i] = 0.8 * voi + 0.08 * np.clip(hf_lf, -12.0, 14.0) + 0.22 * np.clip(flux, 0.0, 2.0) - 0.10 * np.maximum(0.0, -rel_db - 28.0)
        elif man in ("vowel", "diphthong"):
            score[i] = 2.0 * voi - 0.14 * np.maximum(0.0, hf_lf - 1.0) - 0.20 * np.maximum(0.0, -rel_db - 19.0) - 0.15 * np.maximum(0.0, flux - 1.2)
        else:
            score[i] = 1.4 * voi - 0.08 * np.maximum(0.0, hf_lf - 3.0) - 0.12 * np.maximum(0.0, -rel_db - 23.0)

    pref = np.concatenate([np.zeros((m_phones, 1)), np.cumsum(score, axis=1)], axis=1)

    exp_f = [max(2.0, bd * scale / 5.0) for bd in base_durs]
    min_f = [max(2, int(math.floor(0.40 * ef))) for ef in exp_f]
    max_f = [max(mn + 2, int(math.ceil(2.3 * ef))) for mn, ef in zip(min_f, exp_f)]
    while sum(min_f) > n_fr:
        changed = False
        for idx_r in range(m_phones):
            if min_f[idx_r] > 1:
                min_f[idx_r] -= 1
                changed = True
                if sum(min_f) <= n_fr:
                    break
        if not changed:
            break

    suf_min = [0] * (m_phones + 1)
    suf_max = [0] * (m_phones + 1)
    for i in range(m_phones - 1, -1, -1):
        suf_min[i] = suf_min[i + 1] + min_f[i]
        suf_max[i] = suf_max[i + 1] + max_f[i]

    # Boundary-change flags reward snapping V<->UV transitions to acoustic landmark cliffs
    b_is_transition = [
        (i > 0) and ((is_voi[i - 1] != is_voi[i]) or ((not is_voi[i]) != (not is_voi[i - 1])))
        for i in range(m_phones)
    ]

    dp = np.full((m_phones + 1, n_fr + 1), -1e12, dtype=np.float64)
    bp = np.zeros((m_phones + 1, n_fr + 1), dtype=np.int32)
    dp[0, 0] = 0.0

    for i in range(m_phones):
        ef = exp_f[i]
        mn, mx = min_f[i], max_f[i]
        rem_min = suf_min[i + 1]
        rem_max = suf_max[i + 1]
        j_lo = max(1, n_fr - rem_max)
        j_hi = n_fr - rem_min
        p_row = pref[i]
        tr_bonus = 0.45 * cliff if b_is_transition[i] else np.zeros(n_fr)
        for j in range(j_lo, j_hi + 1):
            k_lo = max(0, j - mx)
            k_hi = min(j - mn, n_fr - suf_min[i])
            if k_hi < k_lo:
                continue
            ks = np.arange(k_lo, k_hi + 1)
            durs = j - ks
            dur_pen = -0.32 * ((durs - ef) / max(2.0, ef)) ** 2
            vals = dp[i, ks] + (p_row[j] - p_row[ks]) + dur_pen + tr_bonus[ks]
            best_idx = int(np.argmax(vals))
            dp[i + 1, j] = vals[best_idx]
            bp[i + 1, j] = ks[best_idx]

    if dp[m_phones, n_fr] <= -1e11:
        edges = np.linspace(0, n_fr, m_phones + 1).astype(int)
        return [(int(edges[i]), int(max(edges[i] + 1, edges[i + 1]))) for i in range(m_phones)]

    bounds: list[tuple[int, int]] = []
    cur = n_fr
    for i in range(m_phones, 0, -1):
        prev = int(bp[i, cur])
        bounds.append((prev, cur))
        cur = prev
    bounds.reverse()

    # Post-DP +-2 frame landmark snapping at V<->UV obstruent edges so vowel tails don't leak into fricatives
    refined = [list(b) for b in bounds]
    for i in range(m_phones - 1):
        p_cur, p_nxt = ipas[i], ipas[i + 1]
        v_cur, v_nxt = is_voi[i], is_voi[i + 1]
        if v_cur == v_nxt and (p_cur in OBSTRUENT_IPAS) == (p_nxt in OBSTRUENT_IPAS):
            continue
        b_cur = refined[i][1]
        c_lo = max(refined[i][0] + 2, b_cur - 2)
        c_hi = min(refined[i + 1][1] - 2, b_cur + 2)
        if c_hi > c_lo:
            best_b = c_lo + int(np.argmax(cliff[c_lo : c_hi + 1]))
            refined[i][1] = best_b
            refined[i + 1][0] = best_b

    return [(b[0], b[1]) for b in refined]


def measure_token_boundary_purity(
    en_tokens: dict[str, list],
    zh_tokens: dict[tuple[str, str], list] | None = None,
    sr: int = SR,
) -> dict[str, float]:
    """Measure unvoiced obstruent token purity (HF/LF dB & voiced-frame contamination)."""
    freqs = rfftfreq(N_FFT, 1.0 / sr)
    lf_m = (freqs >= 80.0) & (freqs <= 1000.0)
    hf_m = (freqs >= 2000.0) & (freqs <= 8000.0)
    sib_m = (freqs >= 8000.0) & (freqs <= 11500.0)

    en_ratios, en_sib_ratios = [], []
    en_leaks = 0
    en_uv_tot = 0
    for ipa, toks in en_tokens.items():
        sp = IPA_SPECS.get(ipa)
        if sp is None or sp.voiced or sp.manner not in ("fricative", "affricate"):
            continue
        for tok in toks:
            pwr = np.exp(2.0 * tok.log_env.astype(np.float64))
            lf = float(np.mean(pwr[:, lf_m]))
            hf = float(np.mean(pwr[:, hf_m]))
            sb = float(np.mean(pwr[:, sib_m]))
            en_ratios.append(10.0 * math.log10(max(1e-16, hf) / max(1e-16, lf)))
            en_sib_ratios.append(10.0 * math.log10(max(1e-16, sb) / max(1e-16, lf)))
            en_uv_tot += len(tok.voiced)
            # Flag edge frames with strong LF energy (> 6 dB above HF)
            fr_lf = np.mean(pwr[:, lf_m], axis=1)
            fr_hf = np.mean(pwr[:, hf_m], axis=1)
            en_leaks += int(np.sum(10.0 * np.log10(np.maximum(1e-16, fr_hf) / np.maximum(1e-16, fr_lf)) < -3.0))

    zh_ratios = []
    zh_leaks = 0
    zh_uv_tot = 0
    if zh_tokens:
        for (o, _f), toks in zh_tokens.items():
            if o not in ("f", "s", "sh", "x", "h", "c", "ch", "q"):
                continue
            for tok in toks:
                fs = max(1, min(len(tok.log_env), int(tok.f_split)))
                pwr = np.exp(2.0 * tok.log_env[:fs].astype(np.float64))
                lf = float(np.mean(pwr[:, lf_m]))
                hf = float(np.mean(pwr[:, hf_m]))
                zh_ratios.append(10.0 * math.log10(max(1e-16, hf) / max(1e-16, lf)))
                zh_uv_tot += fs
                fr_lf = np.mean(pwr[:, lf_m], axis=1)
                fr_hf = np.mean(pwr[:, hf_m], axis=1)
                zh_leaks += int(np.sum(10.0 * np.log10(np.maximum(1e-16, fr_hf) / np.maximum(1e-16, fr_lf)) < -3.0))

    return {
        "en_unvoiced_fric_hf_lf_db": round(float(np.median(en_ratios)) if en_ratios else 0.0, 3),
        "en_unvoiced_fric_gt8k_lf_db": round(float(np.median(en_sib_ratios)) if en_sib_ratios else 0.0, 3),
        "en_lf_leak_frac": round(en_leaks / max(1, en_uv_tot), 4),
        "zh_unvoiced_onset_hf_lf_db": round(float(np.median(zh_ratios)) if zh_ratios else 0.0, 3),
        "zh_lf_leak_frac": round(zh_leaks / max(1, zh_uv_tot), 4),
    }
