"""Stage 9 + 9B Speaker-Anchored English LarynxControl Builder.

Combines:
- Stage 9A: BZNSYP `SpeakerDonorBank` + Differential Source-Filter Prototype Transfer (DSFPT)
  + unified prosody & glottal coupling (keeps speaker identity 100% locked across ZH & EN).
- Stage 9B: Zero-ML speaker-independent English articulation, coarticulation & rhythm table
  (`EnArticulationTable`) providing authentic English vowels/obstruents/dark-L/coda stops,
  CV/VC context bends, pre-fortis clipping, consonant cluster timing, and cross-word linking.
"""
from __future__ import annotations

import math
from typing import Any
import zlib

import numpy as np
from scipy.fft import rfftfreq
from scipy.signal import lfilter

from exp_20_pybot_larynx import HOP, N_BAP, N_BINS, N_FFT, SR
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB
from exp_20_pybot_larynx.crossling.en_articulation import (
    NEW_EN_OBSTRUENTS,
    VOWEL_IPAS,
    EnArticulationTable,
    classify_articulatory_class,
    get_en_articulation_table,
    resolve_en_allophone,
)
from exp_20_pybot_larynx.crossling.speaker_donor_bank import (
    SpeakerDonorBank,
    get_speaker_donor_bank,
)
from exp_20_pybot_larynx.expressive import (
    EMOTION_PROFILES,
    ExpressiveSegment,
    apply_emotion_to_control,
)
from exp_20_pybot_larynx.frontend.prosody_rules import _F0_AR1_SIGMA
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env, log_env_to_cep
from exp_v6_crossling_dsp.arbiter import (
    HOP_S,
    FrameTrack,
    PhoneSegment,
    _cos_blend,
    _interp_list,
    rule_apply_stress_and_tone,
    rule_crossling_arbiter,
    rule_expand_unit,
)
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS
from exp_v6_crossling_dsp.speaker_profile import SpeakerProfile

_DB_TO_NEPER = 1.0 / NEPER_TO_DB
_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_FREQS_SAFE = np.maximum(_FREQS, 80.0)
_FREQS_SQ = _FREQS_SAFE * _FREQS_SAFE

# Raised-cosine articulatory sculpting mask: 1.0 <= 3.1 kHz, cosine taper to 0.0 >= 4.1 kHz
_U_ART = np.clip((_FREQS_SAFE - 3100.0) / 1000.0, 0.0, 1.0)
_W_ART = 0.5 * (1.0 + np.cos(math.pi * _U_ART))

_VOT_EDGE_FLOOR = np.asarray([0.65, 0.75, 0.82, 0.88, 0.92, 0.95], dtype=np.float64)
_VOICED_FRIC_BAP = np.asarray([0.14, 0.36, 0.68, 0.88, 0.94, 0.97], dtype=np.float64)
_ANCHOR_STRICT_PHONES = frozenset({"u", "ʊ", "ɝ", "ɚ", "ɻ̩"})


def _resonator_db_fast(f_c: float, bw: float) -> np.ndarray:
    """2nd-order vocal-tract resonator transfer in dB across `_FREQS_SAFE`."""
    fc2 = float(f_c) * float(f_c)
    bw_s = max(float(bw), 55.0)
    denom = (fc2 - _FREQS_SQ) ** 2 + (_FREQS_SAFE * bw_s) ** 2
    mag2 = (fc2 * fc2) / np.maximum(denom, 1e-12)
    return np.clip(10.0 * np.log10(np.maximum(mag2, 1e-6)), -22.0, 18.0)


def _build_rhythm_track_stage9b(
    copied_units: list[dict[str, Any]],
    profile: SpeakerProfile,
    en_art: EnArticulationTable | None,
    *,
    use_en_articulation: bool,
) -> tuple[FrameTrack, list[dict[str, Any]]]:
    """Expand EN units, apply corpus-driven rhythm/allophone rules, & build 5 ms FrameTrack."""
    segs: list[PhoneSegment] = [
        PhoneSegment(
            ipa="sil", lang="sil", spec=IPA_SPECS["ə"],
            dur_ms=5.0, pitch_st_5=[0.0] * 5, gain=0.0, unit_idx=-1, is_pause=True,
        )
    ]
    for uidx, u in enumerate(copied_units):
        segs.extend(rule_expand_unit(u, uidx, profile))
        pause_ms = float(u.get("pause_after_ms", 0.0))
        if pause_ms > 0.0:
            segs.append(
                PhoneSegment(
                    ipa="sil", lang="sil", spec=IPA_SPECS["ə"],
                    dur_ms=pause_ms, pitch_st_5=[0.0] * 5, gain=0.0, unit_idx=uidx, is_pause=True,
                )
            )
    segs.append(
        PhoneSegment(
            ipa="sil", lang="sil", spec=IPA_SPECS["ə"],
            dur_ms=5.0, pitch_st_5=[0.0] * 5, gain=0.0, unit_idx=-1, is_pause=True,
        )
    )

    rule_apply_stress_and_tone(segs, profile)
    rule_crossling_arbiter(segs)

    n_s = len(segs)
    # Compute consonant cluster lengths within non-pause runs
    cluster_lens = [1] * n_s
    ii = 0
    while ii < n_s:
        if segs[ii].is_pause or segs[ii].ipa in VOWEL_IPAS:
            ii += 1
            continue
        jj = ii + 1
        while jj < n_s and not segs[jj].is_pause and segs[jj].ipa not in VOWEL_IPAS and segs[jj].unit_idx == segs[jj - 1].unit_idx:
            jj += 1
        cl = jj - ii
        for kk in range(ii, jj):
            cluster_lens[kk] = cl
        ii = jj

    seg_meta: list[dict[str, Any]] = []
    rate = max(0.60, min(1.60, float(profile.speech_rate_scale)))
    last_nonpause_idx = max((idx for idx, s in enumerate(segs) if not s.is_pause), default=-1)

    for s_idx, seg in enumerate(segs):
        if seg.is_pause:
            seg_meta.append({"key": "sil", "prev_cls": 0, "next_cls": 0, "coda_stop": False})
            continue

        prev_seg = segs[s_idx - 1] if (s_idx > 0 and not segs[s_idx - 1].is_pause) else None
        next_seg = segs[s_idx + 1] if (s_idx + 1 < n_s and not segs[s_idx + 1].is_pause) else None
        prev_ipa = prev_seg.ipa if prev_seg is not None else None
        next_ipa = next_seg.ipa if next_seg is not None else None
        same_word_next = bool(next_seg is not None and next_seg.unit_idx == seg.unit_idx)
        key = resolve_en_allophone(seg.ipa, next_ipa, same_word_next)
        p_cls = classify_articulatory_class(prev_ipa)
        n_cls = classify_articulatory_class(next_ipa)
        is_pfin = (s_idx == last_nonpause_idx) or (
            s_idx + 1 < n_s and segs[s_idx + 1].is_pause and segs[s_idx + 1].dur_ms >= 45.0
        )

        if use_en_articulation and en_art is not None:
            seg.dur_ms = en_art.predict_duration_ms(
                key,
                seg.ipa,
                seg.stress,
                next_ipa=next_ipa,
                same_word_next=same_word_next,
                cluster_len=cluster_lens[s_idx],
                is_phrase_final=is_pfin,
                rate=rate,
            )

        coda_stop = key.endswith("_coda") and key != "l_coda" and (next_ipa is None or next_ipa not in VOWEL_IPAS)
        seg_meta.append({"key": key, "prev_cls": p_cls, "next_cls": n_cls, "coda_stop": coda_stop})

    frame_counts = [max(1, int(round(s.dur_ms / (HOP_S * 1000.0)))) for s in segs]
    n_frames = sum(frame_counts)

    f0_hz = np.zeros(n_frames, dtype=np.float64)
    voiced_gain = np.zeros(n_frames, dtype=np.float64)
    noise_gain = np.zeros(n_frames, dtype=np.float64)
    formants = np.zeros((n_frames, 3), dtype=np.float64)
    bandwidths = np.zeros((n_frames, 3), dtype=np.float64)
    zero_freq = np.zeros(n_frames, dtype=np.float64)
    zero_bw = np.full(n_frames, 240.0, dtype=np.float64)
    zero_depth_db = np.zeros(n_frames, dtype=np.float64)
    fric_p1 = np.zeros((n_frames, 3), dtype=np.float64)
    fric_p2 = np.zeros((n_frames, 3), dtype=np.float64)
    voicing_bar = np.zeros(n_frames, dtype=np.float64)
    burst_env = np.zeros(n_frames, dtype=np.float64)

    vtln = max(0.75, min(1.20, profile.vtln_scale))
    f0_ref = profile.f0_median_hz

    cursor = 0
    for s_idx, (seg, nf) in enumerate(zip(segs, frame_counts)):
        if seg.is_pause:
            formants[cursor : cursor + nf] = np.array([500.0, 1500.0, 2700.0]) * vtln
            bandwidths[cursor : cursor + nf] = [120.0, 180.0, 240.0]
            cursor += nf
            continue

        sp = seg.spec
        prev_seg = segs[s_idx - 1] if (s_idx > 0 and not segs[s_idx - 1].is_pause) else None
        next_seg = segs[s_idx + 1] if (s_idx + 1 < n_s and not segs[s_idx + 1].is_pause) else None

        f_start = np.array(sp.formants, dtype=np.float64) * vtln
        f_end = np.array(sp.end_formants if sp.end_formants else sp.formants, dtype=np.float64) * vtln
        if use_en_articulation and seg_meta[s_idx]["key"] == "l_coda":
            # Velarized English dark-L target
            f_start = np.array([465.0, 1020.0, 2780.0], dtype=np.float64) * vtln
            f_end = f_start.copy()

        if seg.lang == "en" and seg.stress == 0 and sp.manner == "vowel":
            schwa = np.array(IPA_SPECS["ə"].formants, dtype=np.float64) * vtln
            f_start = 0.78 * f_start + 0.22 * schwa
            f_end = 0.78 * f_end + 0.22 * schwa

        f_cv = f_start.copy()
        if prev_seg is not None and prev_seg.spec.locus_f2_hz is not None and sp.manner in ("vowel", "diphthong", "glide", "liquid"):
            w = prev_seg.spec.locus_weight
            f_cv[1] = w * (prev_seg.spec.locus_f2_hz * vtln) + (1.0 - w) * f_start[1]
            f_cv[0] = 0.45 * (prev_seg.spec.formants[0] * vtln) + 0.55 * f_start[0]

        f_vc = f_end.copy()
        if next_seg is not None and next_seg.spec.locus_f2_hz is not None and sp.manner in ("vowel", "diphthong", "glide"):
            w = next_seg.spec.locus_weight * 0.85
            if use_en_articulation and next_seg.ipa == "l" and seg_meta[s_idx + 1]["key"] == "l_coda":
                f_vc[1] = w * (1020.0 * vtln) + (1.0 - w) * f_end[1]
            else:
                f_vc[1] = w * (next_seg.spec.locus_f2_hz * vtln) + (1.0 - w) * f_end[1]

        closure_frac = sp.closure_ms / max(1.0, sp.closure_ms + sp.vot_ms) if sp.manner in ("stop", "affricate") else 0.0

        for k in range(nf):
            t = k / max(1, nf - 1)
            f_idx = cursor + k
            p_st = _interp_list(seg.pitch_st_5, t)
            f0_cur = f0_ref * (2.0 ** (p_st / 12.0))

            if t < 0.32 and prev_seg is not None:
                u = _cos_blend(t / 0.32)
                f_cur = (1.0 - u) * f_cv + u * f_start
            elif t > 0.72 and next_seg is not None:
                u = _cos_blend((t - 0.72) / 0.28)
                f_cur = (1.0 - u) * f_end + u * f_vc
            else:
                u = max(0.0, min(1.0, (t - 0.22) / 0.56)) if sp.end_formants else 0.0
                f_cur = (1.0 - u) * f_start + u * f_end

            formants[f_idx] = f_cur
            bandwidths[f_idx] = sp.bandwidths

            if sp.anti_zero is not None:
                zero_freq[f_idx] = sp.anti_zero[0] * vtln
                zero_bw[f_idx] = sp.anti_zero[1]
                zero_depth_db[f_idx] = sp.anti_zero[2]
            elif next_seg is not None and next_seg.spec.manner == "nasal" and t > 0.55 and sp.manner in ("vowel", "diphthong"):
                z_u = (t - 0.55) / 0.45
                zero_freq[f_idx] = next_seg.spec.anti_zero[0] * vtln
                zero_bw[f_idx] = next_seg.spec.anti_zero[1]
                zero_depth_db[f_idx] = next_seg.spec.anti_zero[2] * 0.65 * z_u

            if sp.manner in ("vowel", "diphthong", "glide", "nasal", "liquid"):
                f0_hz[f_idx] = f0_cur
                env = min(1.0, (k + 0.5) / 2.5, (nf - k + 0.5) / 2.5)
                manner_gain = 1.0 if sp.manner in ("vowel", "diphthong") else (0.76 if sp.manner in ("glide", "liquid") else 0.62)
                voiced_gain[f_idx] = seg.gain * manner_gain * (0.85 + 0.15 * env)
                noise_gain[f_idx] = 0.0
            elif sp.manner == "fricative":
                env = math.sin(math.pi * max(0.05, min(0.95, t))) ** 0.55
                noise_gain[f_idx] = seg.gain * 0.55 * env
                if sp.voiced:
                    f0_hz[f_idx] = f0_cur
                    voiced_gain[f_idx] = seg.gain * sp.voicing_bar * 0.65
                    voicing_bar[f_idx] = sp.voicing_bar
                if sp.fric_poles:
                    p1 = sp.fric_poles[0]
                    fric_p1[f_idx] = (p1[0] * (0.92 + 0.08 * vtln), p1[1], p1[2])
                    if len(sp.fric_poles) > 1:
                        p2 = sp.fric_poles[1]
                        fric_p2[f_idx] = (p2[0] * (0.92 + 0.08 * vtln), p2[1], p2[2])
            else:
                if t < closure_frac:
                    if sp.voiced:
                        f0_hz[f_idx] = f0_cur
                        voiced_gain[f_idx] = seg.gain * sp.voicing_bar * 0.35
                        voicing_bar[f_idx] = sp.voicing_bar
                    noise_gain[f_idx] = 0.0
                else:
                    rel_t = (t - closure_frac) / max(1e-6, 1.0 - closure_frac)
                    is_burst = rel_t < 0.22
                    if sp.manner == "affricate":
                        env = (1.0 - 0.35 * rel_t) * (1.25 if is_burst else 0.95)
                    else:
                        env = (1.35 if is_burst else max(0.15, (1.0 - rel_t) ** 0.85)) * (0.75 if sp.voiced else 1.0)
                    noise_gain[f_idx] = seg.gain * 0.52 * env
                    burst_env[f_idx] = 1.0 if is_burst else 0.0
                    if sp.voiced:
                        f0_hz[f_idx] = f0_cur
                        voiced_gain[f_idx] = seg.gain * sp.voicing_bar * (0.45 + 0.35 * rel_t)
                        voicing_bar[f_idx] = sp.voicing_bar
                    if sp.fric_poles:
                        p1 = sp.fric_poles[0]
                        fric_p1[f_idx] = (p1[0] * (0.92 + 0.08 * vtln), p1[1], p1[2])
                        if len(sp.fric_poles) > 1:
                            p2 = sp.fric_poles[1]
                            fric_p2[f_idx] = (p2[0] * (0.92 + 0.08 * vtln), p2[1], p2[2])
        cursor += nf

    track = FrameTrack(
        n_frames=n_frames,
        f0_hz=f0_hz,
        voiced_gain=voiced_gain,
        noise_gain=noise_gain,
        formants=formants,
        bandwidths=bandwidths,
        zero_freq=zero_freq,
        zero_bw=zero_bw,
        zero_depth_db=zero_depth_db,
        fric_p1=fric_p1,
        fric_p2=fric_p2,
        voicing_bar=voicing_bar,
        burst_env=burst_env,
        segments=segs,
    )
    return track, seg_meta


def _build_segment_frame_map(
    track: FrameTrack,
    s_fr: int,
    e_fr: int,
) -> tuple[list[int], list[float], list[tuple[int, int, PhoneSegment]]]:
    """Map stripped output frame indices `[0..e_fr-s_fr-1]` to `(seg_idx, local_t)` and segment spans."""
    n_tot = track.n_frames
    seg_per_ti = np.zeros(n_tot, dtype=np.int32)
    t_per_ti = np.zeros(n_tot, dtype=np.float64)
    spans_out: list[tuple[int, int, PhoneSegment]] = []

    cursor = 0
    for seg_i, seg in enumerate(track.segments):
        nf = max(1, int(round(seg.dur_ms / 5.0)))
        end = min(n_tot, cursor + nf)
        if end > cursor:
            seg_per_ti[cursor:end] = seg_i
            if nf > 1:
                t_per_ti[cursor:end] = np.arange(end - cursor, dtype=np.float64) / float(nf - 1)
            else:
                t_per_ti[cursor:end] = 0.5
        oa = max(0, cursor - s_fr)
        ob = min(e_fr - s_fr, end - s_fr)
        if ob > oa:
            spans_out.append((oa, ob, seg))
        cursor = end

    seg_idx_out = [int(seg_per_ti[ti]) for ti in range(s_fr, e_fr)]
    t_out = [float(t_per_ti[ti]) for ti in range(s_fr, e_fr)]
    return seg_idx_out, t_out, spans_out


def _polish_cepstral_seams_en(
    log_env: np.ndarray,
    voiced: np.ndarray,
    seg_spans: list[tuple[int, int, PhoneSegment]],
) -> np.ndarray:
    """Bridge low-quefrency `cep[1:18]` over 2-frame boundaries across adjacent voiced phones & C-V links."""
    n_fr = log_env.shape[0]
    if n_fr < 6 or len(seg_spans) < 2:
        return log_env
    cep = log_env_to_cep(log_env)
    modified = False
    for idx in range(1, len(seg_spans)):
        prev_a, prev_b, prev_seg = seg_spans[idx - 1]
        cur_a, cur_b, cur_seg = seg_spans[idx]
        if prev_seg.is_pause or cur_seg.is_pause or cur_a != prev_b:
            continue
        # Protect unvoiced obstruent attacks, while smoothly linking voiced C->V word junctions too
        if prev_seg.spec.manner in ("fricative", "stop", "affricate") and not prev_seg.spec.voiced:
            continue
        if cur_seg.spec.manner in ("fricative", "stop", "affricate"):
            continue
        if cur_a < 2 or cur_a + 1 >= n_fr:
            continue
        if not (voiced[cur_a - 2] and voiced[cur_a - 1] and voiced[cur_a] and voiced[cur_a + 1]):
            continue
        c_left = cep[cur_a - 2, 1:18].copy()
        c_right = cep[cur_a + 1, 1:18].copy()
        if float(np.sqrt(np.mean((c_right - c_left) ** 2))) < 0.05:
            continue
        cep[cur_a - 1, 1:18] = 0.65 * cep[cur_a - 1, 1:18] + 0.35 * (0.67 * c_left + 0.33 * c_right)
        cep[cur_a, 1:18] = 0.65 * cep[cur_a, 1:18] + 0.35 * (0.33 * c_left + 0.67 * c_right)
        modified = True
    if not modified:
        return log_env
    return np.clip(cep_to_log_env(cep, n_bins=N_BINS), -16.0, 6.0)


def _shape_en_f0_contour(
    raw_f0: np.ndarray,
    voiced: np.ndarray,
    seg_spans: list[tuple[int, int, PhoneSegment]],
    f0_eff: float,
    range_sc: float,
    *,
    clause_pos: tuple[float, float],
    is_clause_final: bool,
    is_question: bool,
    seed: int,
    n_units: int,
    stress_f0_st_5bin: np.ndarray | None = None,
) -> np.ndarray:
    """De-bias EN frontend DC F0 shift, reshape lexical/nuclear stress, clause declination, & AR(1) flutter."""
    n_fr = len(raw_f0)
    out_f0 = np.zeros(n_fr, dtype=np.float64)
    vm = voiced & (raw_f0 > 40.0)
    if not np.any(vm):
        return out_f0

    st = np.zeros(n_fr, dtype=np.float64)
    st[vm] = 12.0 * np.log2(raw_f0[vm] / max(50.0, f0_eff))
    st[vm] -= float(np.median(st[vm]))

    stressed_idxs = [
        idx
        for idx, (oa, ob, seg) in enumerate(seg_spans)
        if not seg.is_pause and seg.stress == 1 and seg.spec.manner in ("vowel", "diphthong") and ob > oa
    ]
    nuclear_idx = stressed_idxs[-1] if stressed_idxs else -1

    stress_offset = np.zeros(n_fr, dtype=np.float64)
    n_sp = len(seg_spans)
    for idx, (oa, ob, seg) in enumerate(seg_spans):
        if seg.is_pause or not seg.spec.voiced or ob <= oa:
            continue
        nf = ob - oa
        u = np.linspace(0.0, 1.0, nf, dtype=np.float64)
        hat = 0.5 - 0.5 * np.cos(math.tau * u)

        # Blend corpus-measured intra-vowel slope gently when Stage 9B table is active
        art_curve = np.zeros(nf, dtype=np.float64)
        if stress_f0_st_5bin is not None and seg.stress in (0, 1, 2):
            xp = np.linspace(0.0, 1.0, stress_f0_st_5bin.shape[1], dtype=np.float64)
            art_curve = np.interp(u, xp, stress_f0_st_5bin[int(seg.stress)])

        if seg.stress == 1:
            amp = (2.55 if idx == nuclear_idx else 2.05) * range_sc
            stress_offset[oa:ob] = 0.45 * amp + 0.55 * amp * hat + 0.35 * range_sc * art_curve
        elif seg.stress == 2:
            amp = 0.55 * range_sc
            stress_offset[oa:ob] = amp * (0.4 + 0.6 * hat) + 0.30 * range_sc * art_curve
        elif seg.stress == 0:
            stress_offset[oa:ob] = -1.55 * range_sc + 0.25 * range_sc * art_curve
        else:
            nb = 0.0
            if idx + 1 < n_sp and seg_spans[idx + 1][2].stress == 1:
                nb += 0.65 * range_sc
            elif idx + 1 < n_sp and seg_spans[idx + 1][2].stress == 0:
                nb -= 0.55 * range_sc
            elif idx - 1 >= 0 and seg_spans[idx - 1][2].stress == 1:
                nb += 0.45 * range_sc
            elif idx - 1 >= 0 and seg_spans[idx - 1][2].stress == 0:
                nb -= 0.55 * range_sc
            stress_offset[oa:ob] = nb

    st[vm] = 0.65 * st[vm] + stress_offset[vm]

    p0, p1 = float(clause_pos[0]), float(clause_pos[1])
    u_cl = np.linspace(p0, p1, n_fr, dtype=np.float64)
    decl_st = 0.55 - 1.40 * u_cl
    st[vm] += decl_st[vm]

    decl_mid = 0.55 - 1.40 * 0.5 * (p0 + p1)
    st[vm] += decl_mid - float(np.median(st[vm]))

    if is_clause_final:
        v_idx = np.where(vm)[0]
        if v_idx.size >= 8:
            tail_n = min(v_idx.size, 32 if is_question else 30)
            tail_pos = v_idx[-tail_n:]
            u_tail = np.linspace(0.0, 1.0, tail_n, dtype=np.float64)
            if is_question:
                st[tail_pos] += (2.6 * range_sc) * (u_tail ** 1.4)
            else:
                st[tail_pos] -= (1.55 * range_sc) * (u_tail ** 1.3)

    if _F0_AR1_SIGMA > 0.0 and n_fr > 4:
        rng = np.random.default_rng(zlib.crc32(f"en_s9|{int(seed)}|{int(n_units)}".encode("utf-8")) & 0xFFFFFFFF)
        rho = 0.72
        eps = rng.normal(0.0, _F0_AR1_SIGMA * math.sqrt(1.0 - rho * rho), size=n_fr)
        ar1 = lfilter((1.0,), (1.0, -rho), eps)
        st[vm] += np.clip(ar1[vm], -0.45, 0.45)

    st_sm = st.copy()
    for k in range(1, n_fr - 1):
        if vm[k - 1] and vm[k] and vm[k + 1]:
            st_sm[k] = 0.20 * st[k - 1] + 0.60 * st[k] + 0.20 * st[k + 1]

    out_f0[vm] = np.clip(f0_eff * (2.0 ** (st_sm[vm] / 12.0)), 65.0, 460.0)
    return out_f0


def build_speaker_anchored_en_control(
    en_units: list[dict[str, Any]],
    base_f0_hz: float,
    params: Any,
    kb: Any,
    *,
    expressive_seg: ExpressiveSegment | None = None,
    clause_pos: tuple[float, float] = (0.0, 1.0),
    is_clause_final: bool = True,
    is_question: bool = False,
    use_en_articulation: bool = True,
) -> tuple[LarynxControl, float]:
    """Synthesize English units anchored to `kb`'s speaker donors + optional Stage 9B articulation."""
    bank: SpeakerDonorBank = get_speaker_donor_bank(kb)
    en_art: EnArticulationTable | None = get_en_articulation_table() if use_en_articulation else None
    art_active = en_art is not None and bool(use_en_articulation)

    f0_eff = float(base_f0_hz)
    range_sc = 1.0
    speed_eff = max(0.60, min(1.60, float(getattr(params, "speed", 1.0))))
    if expressive_seg is not None and expressive_seg.intensity > 1e-4:
        e_prof = EMOTION_PROFILES.get(expressive_seg.emotion, EMOTION_PROFILES["neutral"])
        sc = float(expressive_seg.intensity)
        f0_eff = float(np.clip(f0_eff * (2.0 ** ((float(e_prof["f0_shift_st"]) * sc) / 12.0)), 65.0, 460.0))
        range_sc = 1.0 + (float(e_prof["f0_range_scale"]) - 1.0) * sc
        speed_eff = float(np.clip(speed_eff * (1.0 + (float(e_prof["speed_scale"]) - 1.0) * sc), 0.60, 1.60))

    prof = SpeakerProfile(
        name="EN_STAGE9B_ART" if art_active else "EN_STAGE9_DONOR",
        f0_median_hz=f0_eff,
        f0_iqr_st=max(3.2, 4.1 * range_sc),
        f0_range_st=max(6.5, 8.0 * range_sc),
        ltas_tilt_db_per_oct=-2.5,
        residual_eq_db=(0.0,) * 8,
        hnr_db=14.5,
        vtln_scale=1.0,
        speech_rate_scale=speed_eff,
    )

    pause_scale = max(0.40, min(2.00, float(getattr(params, "pause_scale", 1.0))))
    copied_units: list[dict[str, Any]] = []
    for idx, u in enumerate(en_units):
        uc = dict(u)
        uc["pause_after_ms"] = float(u.get("pause_after_ms", 0.0)) * pause_scale if idx < len(en_units) - 1 else 0.0
        copied_units.append(uc)

    track, seg_meta = _build_rhythm_track_stage9b(
        copied_units,
        prof,
        en_art,
        use_en_articulation=art_active,
    )
    s_fr = 1
    e_fr = max(s_fr, track.n_frames - 1)
    n_fr = e_fr - s_fr
    if n_fr <= 0:
        return LarynxControl(
            f0_hz=np.zeros(0, dtype=np.float64),
            voiced=np.zeros(0, dtype=bool),
            log_env=np.zeros((0, N_BINS), dtype=np.float64),
            bap=np.zeros((0, N_BAP), dtype=np.float64),
            n_samples=0,
        ), 0.0

    seg_idx_arr, local_t_arr, seg_spans = _build_segment_frame_map(track, s_fr, e_fr)

    raw_f0 = track.f0_hz[s_fr:e_fr].astype(np.float64)
    voiced = (track.voiced_gain[s_fr:e_fr] > 1e-6) & (raw_f0 > 40.0)
    f0_hz = _shape_en_f0_contour(
        raw_f0,
        voiced,
        seg_spans,
        f0_eff,
        range_sc,
        clause_pos=clause_pos,
        is_clause_final=is_clause_final,
        is_question=is_question,
        seed=int(getattr(params, "seed", 2026)),
        n_units=len(en_units),
        stress_f0_st_5bin=en_art.stress_f0_st_5bin if art_active else None,
    )

    log_env = np.full((n_fr, N_BINS), -14.5, dtype=np.float64)
    bap = np.full((n_fr, N_BAP), 0.99, dtype=np.float64)
    tilt = np.zeros(n_fr, dtype=np.float64)
    jit = np.full(n_fr, 0.0055, dtype=np.float64)
    shim = np.full(n_fr, 0.025, dtype=np.float64)
    rd = np.ones(n_fr, dtype=np.float64)

    v_indices = np.where(voiced)[0]
    pfin_mask = np.zeros(n_fr, dtype=bool)
    pfin_u = np.zeros(n_fr, dtype=np.float64)
    if is_clause_final and v_indices.size >= 4:
        tail_len = min(v_indices.size, 28)
        t_pos = v_indices[-tail_len:]
        pfin_mask[t_pos] = True
        pfin_u[t_pos] = np.linspace(0.0, 1.0, tail_len, dtype=np.float64)

    max_obs_delta_np = 8.5 * _DB_TO_NEPER
    max_coart_np = 4.5 * _DB_TO_NEPER

    for k in range(n_fr):
        ti = s_fr + k
        seg_i = seg_idx_arr[k]
        seg = track.segments[seg_i]
        meta_s = seg_meta[seg_i]
        loc_t = local_t_arr[k]
        vg = float(track.voiced_gain[ti])
        ng = float(track.noise_gain[ti])
        vbar = float(track.voicing_bar[ti])
        has_voiced = bool(voiced[k])
        has_noise = ng > 1e-6

        if seg.is_pause or (not has_voiced and not has_noise):
            log_env[k] = -14.5
            bap[k] = 0.99
            continue

        key = meta_s["key"]
        p_cls = meta_s["prev_cls"]
        n_cls = meta_s["next_cls"]

        if has_voiced and not has_noise and vbar <= 0.0:
            # --- Pure Sonorant via BZNSYP Donor + DSFPT + Stage 9B Corpus Coarticulation ---
            don = bank.get_sonorant_donor(seg.ipa)
            if don.traj_env is not None and don.traj_formants is not None and don.traj_bap6 is not None:
                n_b = don.traj_env.shape[0]
                u_bin = float(np.clip((loc_t - 0.10) / 0.80, 0.0, 1.0)) * float(n_b - 1)
                i0 = min(n_b - 1, max(0, int(math.floor(u_bin))))
                i1 = min(n_b - 1, i0 + 1)
                w = u_bin - float(i0)
                e_don = (1.0 - w) * don.traj_env[i0] + w * don.traj_env[i1]
                f_don = (1.0 - w) * don.traj_formants[i0] + w * don.traj_formants[i1]
                bap_don = (1.0 - w) * don.traj_bap6[i0] + w * don.traj_bap6[i1]
            else:
                e_don = don.env
                f_don = don.formants
                bap_don = don.bap6

            rule_fmt = np.array(
                [
                    track.formants[ti, 0],
                    track.formants[ti, 1],
                    track.formants[ti, 2],
                ],
                dtype=np.float64,
            )

            coart_term = np.zeros(N_BINS, dtype=np.float64)
            if art_active:
                f_art, _d_art_env, coart_env, _rel_db = en_art.sample_sonorant_articulation(
                    key, loc_t, p_cls, n_cls, ipa_fallback=seg.ipa
                )
                # Tight phonemic anchor on extreme rounded / rhotic vowels where corpus LSF drifts
                if seg.ipa in _ANCHOR_STRICT_PHONES:
                    w_art = 0.22
                elif key == "l_coda":
                    w_art = 0.55
                else:
                    w_art = 0.55
                f_canon = w_art * f_art + (1.0 - w_art) * rule_fmt
                coart_term = np.clip(0.45 * coart_env * _W_ART, -max_coart_np, max_coart_np)
            else:
                f_canon = rule_fmt

            f_tgt = np.array(
                [
                    f_canon[0] * bank.scale_f1,
                    f_canon[1] * bank.scale_f2,
                    f_canon[2] * bank.scale_f3,
                ],
                dtype=np.float64,
            )
            f_tgt[0] = float(np.clip(f_tgt[0], 240.0, 1200.0))
            f_tgt[1] = float(np.clip(f_tgt[1], max(f_tgt[0] + 220.0, 740.0), 3050.0))
            f_tgt[2] = float(np.clip(f_tgt[2], max(f_tgt[1] + 280.0, 1800.0), 4200.0))

            bw1 = max(95.0, float(track.bandwidths[ti, 0]))
            bw2 = max(135.0, float(track.bandwidths[ti, 1]))
            bw3 = max(200.0, float(track.bandwidths[ti, 2]))

            delta_db = _W_ART * (
                (_resonator_db_fast(f_tgt[0], bw1) - _resonator_db_fast(f_don[0], bw1))
                + (_resonator_db_fast(f_tgt[1], bw2) - _resonator_db_fast(f_don[1], bw2))
                + (_resonator_db_fast(f_tgt[2], bw3) - _resonator_db_fast(f_don[2], bw3))
            )

            z_d = float(track.zero_depth_db[ti])
            if z_d > 0.0 and seg.spec.anti_zero is None:
                z_f = float(track.zero_freq[ti]) * bank.scale_f2
                z_bw = max(85.0, float(track.zero_bw[ti]))
                if z_f > 200.0:
                    delta_db = delta_db - _W_ART * (z_d / (1.0 + ((_FREQS_SAFE - z_f) / z_bw) ** 2))

            delta_db = np.clip(delta_db, -11.0, 11.0)
            env_raw = e_don + delta_db * _DB_TO_NEPER + coart_term

            pre_pwr = 0.72 * don.power + 0.28 * bank.voiced_ref_power
            post_pwr = float(np.sum(np.exp(2.0 * np.clip(env_raw, -16.0, 6.0))))
            gain_pwr = 0.5 * math.log(max(pre_pwr, 1e-18) / max(post_pwr, 1e-18))

            dyn_db = float(np.clip(20.0 * math.log10(max(0.25, vg) / 0.88), -3.2, 2.2))
            if art_active and seg.stress in (0, 1, 2):
                dyn_db += float(np.clip(0.35 * en_art.stress_gain_db[int(seg.stress)], -0.9, 0.9))
            if pfin_mask[k]:
                dyn_db -= 1.8 * pfin_u[k]

            log_env[k] = np.clip(env_raw + gain_pwr + dyn_db * _DB_TO_NEPER, -16.0, 6.0)
            bap[k] = bap_don

        elif has_voiced:
            # --- Voiced Obstruent (v, ð, z, ʒ, b, d, ɡ, dʒ + coda stops) ---
            obs = bank.get_obstruent_donor(seg.ipa)
            if art_active and seg.ipa in NEW_EN_OBSTRUENTS:
                d_obs, v_prob_c, _obs_db = en_art.sample_obstruent_articulation(
                    key, loc_t, p_cls, n_cls, ipa_fallback=seg.ipa
                )
                v_eff = max(0.30, 0.55 * float(vbar) + 0.45 * v_prob_c)
                e_vbar = bank.voicing_bar_env + float(np.clip(20.0 * math.log10(max(0.20, vg) * (v_eff / max(0.1, vbar)) / 0.50), -6.0, 2.5)) * _DB_TO_NEPER
                e_art_obs = bank.unvoiced_ltas_env + np.clip(d_obs, -max_obs_delta_np, max_obs_delta_np)
                obs_mix = 0.55 * (obs.burst_env if float(track.burst_env[ti]) > 0.5 else obs.env) + 0.45 * e_art_obs
            else:
                e_vbar = bank.voicing_bar_env + float(np.clip(20.0 * math.log10(max(0.20, vg) / 0.50), -6.0, 2.0)) * _DB_TO_NEPER
                obs_mix = obs.burst_env if float(track.burst_env[ti]) > 0.5 else obs.env

            if has_noise:
                n_mod_db = float(np.clip(20.0 * math.log10(max(0.15, ng) / 0.42), -8.0, 3.0)) - 3.5
                if meta_s["coda_stop"]:
                    n_mod_db -= 3.0
                hf_taper_db = -4.5 * np.clip((_FREQS_SAFE - 7800.0) / 3000.0, 0.0, 1.0)
                e_fric = obs_mix + (n_mod_db + hf_taper_db) * _DB_TO_NEPER
                log_env[k] = np.clip(
                    0.5 * np.log(np.exp(2.0 * e_vbar) + np.exp(2.0 * e_fric) + 1e-18),
                    -16.0,
                    6.0,
                )
            else:
                log_env[k] = np.clip(e_vbar, -16.0, 6.0)
            bap[k] = _VOICED_FRIC_BAP

        else:
            # --- Voiceless Obstruent (f, θ, s, ʃ, h, p, t, k, tʃ + coda stops) ---
            obs = bank.get_obstruent_donor(seg.ipa)
            base_obs = obs.burst_env if float(track.burst_env[ti]) > 0.5 else obs.env
            if art_active and seg.ipa in NEW_EN_OBSTRUENTS:
                d_obs, _vp, obs_db = en_art.sample_obstruent_articulation(
                    key, loc_t, p_cls, n_cls, ipa_fallback=seg.ipa
                )
                e_art_obs = bank.unvoiced_ltas_env + np.clip(d_obs, -max_obs_delta_np, max_obs_delta_np)
                env_u = 0.55 * (0.78 * base_obs + 0.22 * bank.unvoiced_ltas_env) + 0.45 * e_art_obs
                rel_shape = float(np.clip(0.25 * (obs_db + 8.0), -2.0, 1.5))
            elif art_active:
                d_obs, _vp, obs_db = en_art.sample_obstruent_articulation(
                    key, loc_t, p_cls, n_cls, ipa_fallback=seg.ipa
                )
                # Keep BZNSYP's real obstruent spectrum + subtle entry/exit coarticulation only
                env_u = 0.78 * base_obs + 0.22 * bank.unvoiced_ltas_env + 0.22 * np.clip(d_obs, -max_coart_np, max_coart_np)
                rel_shape = float(np.clip(0.20 * (obs_db + 8.0), -1.5, 1.0))
            else:
                env_u = 0.78 * base_obs + 0.22 * bank.unvoiced_ltas_env
                rel_shape = 0.0

            n_mod_db = float(np.clip(20.0 * math.log10(max(0.15, ng) / 0.48), -8.5, 1.5)) + rel_shape
            if meta_s["coda_stop"]:
                n_mod_db -= 3.0
            hf_taper_db = -6.0 * np.clip((_FREQS_SAFE - 7600.0) / 3000.0, 0.0, 1.0)
            log_env[k] = np.clip(env_u + (n_mod_db + hf_taper_db) * _DB_TO_NEPER, -16.0, 6.0)
            bap[k] = 0.99

    log_env = _polish_cepstral_seams_en(log_env, voiced, seg_spans)

    if np.any(voiced):
        rel_st = np.zeros(n_fr, dtype=np.float64)
        rel_st[voiced] = 12.0 * np.log2(f0_hz[voiced] / max(50.0, f0_eff))

        rd_v = 1.08 - 0.028 * rel_st[voiced] + 0.18 * pfin_u[voiced]
        rd[voiced] = np.clip(rd_v, 0.72, 1.55)

        tilt_v = 0.18 * rel_st[voiced] - 0.55 * pfin_u[voiced]
        tilt[voiced] = np.clip(tilt_v, -1.35, 1.35)

        jit_v = 0.0055 + 0.0020 * pfin_u[voiced] + 0.0005 * np.maximum(0.0, -rel_st[voiced] / 5.0)
        jit[voiced] = np.clip(jit_v, 0.0045, 0.012)
        shim_v = 0.025 + 0.011 * pfin_u[voiced] + 0.003 * np.maximum(0.0, -rel_st[voiced] / 5.0)
        shim[voiced] = np.clip(shim_v, 0.020, 0.055)

        uv_to_v = np.where(voiced & np.concatenate([[True], ~voiced[:-1]]))[0]
        for v0 in uv_to_v:
            seg0 = track.segments[seg_idx_arr[v0]]
            is_obs_prev = v0 > 0 and track.segments[seg_idx_arr[v0 - 1]].spec.manner in ("stop", "affricate", "fricative")
            weights = (1.0, 0.55, 0.22) if (is_obs_prev or seg0.spec.manner in ("stop", "affricate", "fricative")) else (0.55, 0.25)
            for dk, rw in enumerate(weights):
                fk = v0 + dk
                if fk < n_fr and voiced[fk]:
                    bap[fk] = np.maximum(bap[fk], (1.0 - rw) * bap[fk] + rw * _VOT_EDGE_FLOOR)

    en_ctrl = LarynxControl(
        f0_hz=f0_hz,
        voiced=voiced,
        log_env=np.clip(log_env, -16.0, 6.0),
        bap=np.clip(bap, 0.005, 0.99),
        glottal_tilt_db_per_oct=tilt,
        jitter_frac=jit,
        shimmer_frac=shim,
        n_samples=n_fr * HOP,
        rd=rd,
    )

    subharm_strength = 0.0
    if expressive_seg is not None:
        en_ctrl, subharm_strength = apply_emotion_to_control(
            en_ctrl,
            expressive_seg,
            stability=float(getattr(params, "stability", 0.55)),
            seed=int(getattr(params, "seed", 2026)),
        )
    return en_ctrl, float(subharm_strength)
