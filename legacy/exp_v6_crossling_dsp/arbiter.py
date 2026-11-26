"""Modular PyBot-style IPA-to-Acoustic-Trajectory Rule Engine & Arbiter.

Transforms ``exp_v5_arch.crossling_frontend`` plans (or Pinyin syllable lists)
plus a :class:`SpeakerProfile` into 5 ms acoustic control frames via explicit,
inspectable rule modules:
1. Segment & cluster expansion (ZH onset/rime, EN CC-/CCC- clusters, stops)
2. English lexical stress timing & unstressed vowel centralization
3. Mandarin tone + sandhi F0 realization scaled to speaker F0 median & IQR
4. Cross-lingual prosody continuity arbiter across ZH <-> EN switches
5. CV/VC locus coarticulation trajectories (boosting |dF2/dt| toward real speech)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_arch import crossling_frontend as cf
from exp_v6_crossling_dsp import SR
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS, PhoneSpec, get_phone_spec
from exp_v6_crossling_dsp.speaker_profile import SpeakerProfile

HOP_SAMPLES = 120          # 5 ms control rate at 24 kHz
HOP_S = HOP_SAMPLES / SR
_ARPA_RE = cf._ARPA_TOKEN_RE


@dataclass
class PhoneSegment:
    ipa: str
    lang: str
    spec: PhoneSpec
    dur_ms: float
    pitch_st_5: list[float]
    gain: float
    unit_idx: int
    stress: int | None = None
    is_pause: bool = False


@dataclass
class FrameTrack:
    n_frames: int
    f0_hz: np.ndarray              # (T,)
    voiced_gain: np.ndarray        # (T,)
    noise_gain: np.ndarray         # (T,)
    formants: np.ndarray           # (T, 3) Hz (F1, F2, F3)
    bandwidths: np.ndarray         # (T, 3) Hz
    zero_freq: np.ndarray          # (T,) Hz (0 if inactive)
    zero_bw: np.ndarray            # (T,) Hz
    zero_depth_db: np.ndarray      # (T,) dB
    fric_p1: np.ndarray            # (T, 3) (freq, bw, gain_db)
    fric_p2: np.ndarray            # (T, 3)
    voicing_bar: np.ndarray        # (T,) 0..1
    burst_env: np.ndarray          # (T,) transient stop-burst boost
    segments: list[PhoneSegment] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule 1: Segment & Cluster Expansion
# ---------------------------------------------------------------------------
def _split_en_arpabet_to_ipa_with_stress(arpa: list[str]) -> list[tuple[str, int | None]]:
    out: list[tuple[str, int | None]] = []
    for tok in arpa:
        m = _ARPA_RE.fullmatch(tok.strip().upper())
        if not m:
            continue
        base, st = m.group(1), m.group(2)
        ipa = cf.arpabet_phone_to_ipa(tok, include_stress=False)
        stress = int(st) if (st is not None and base in cf.ARPABET_VOWEL_BASE) else None
        out.append((ipa, stress))
    return out


def _apply_cluster_compression(segs: list[PhoneSegment]) -> None:
    """Rule 1b: Compress adjacent English consonant clusters (CC-, -CCC)."""
    i = 0
    while i < len(segs):
        if segs[i].spec.manner in ("vowel", "diphthong"):
            i += 1
            continue
        j = i
        while j < len(segs) and segs[j].spec.manner not in ("vowel", "diphthong"):
            j += 1
        clen = j - i
        if clen >= 2:
            scale = 0.74 if clen == 2 else 0.62
            for k in range(i, j):
                segs[k].dur_ms = max(24.0, segs[k].dur_ms * scale)
        i = j


def rule_expand_unit(unit: dict, unit_idx: int, profile: SpeakerProfile) -> list[PhoneSegment]:
    """Expand one ZH syllable or EN word unit into acoustic :class:`PhoneSegment`s."""
    lang = unit["lang"]
    rate = max(0.6, min(1.6, profile.speech_rate_scale))
    pitch_5 = [float(x) for x in unit["pitch_st"]]
    gain = float(unit["gain"])

    if lang == "zh":
        ipa_list = list(unit["ipa_phones"])
        specs = [get_phone_spec(p) for p in ipa_list]
        raw_durs = [s.base_dur_ms for s in specs]
        target_dur = float(unit["duration_ms"]) / rate
        # Keep stop/affricate/fricative onset capped cleanly (max 42% of syllable),
        # while scaling vowel/coda phones to hit target_dur.
        total_raw = max(1.0, sum(raw_durs))
        scale = target_dur / total_raw
        durs = [max(24.0, d * scale) for d in raw_durs]
        segs = []
        n = len(ipa_list)
        for idx, (ipa, sp, d) in enumerate(zip(ipa_list, specs, durs)):
            # Slice 5-point syllable contour across sub-segments proportionally
            a = idx / n
            b = (idx + 1) / n
            sub_p = [_interp_list(pitch_5, a + (b - a) * (m / 4.0)) for m in range(5)]
            segs.append(
                PhoneSegment(
                    ipa=ipa,
                    lang="zh",
                    spec=sp,
                    dur_ms=d,
                    pitch_st_5=sub_p,
                    gain=gain,
                    unit_idx=unit_idx,
                    stress=unit.get("tone"),
                )
            )
        return segs

    # English word unit
    arpa = unit.get("arpabet") or []
    pairs = _split_en_arpabet_to_ipa_with_stress(arpa)
    if not pairs:
        pairs = [(p, None) for p in unit["ipa_phones"]]
    segs = []
    n = len(pairs)
    for idx, (ipa, st) in enumerate(pairs):
        sp = get_phone_spec(ipa)
        d = sp.base_dur_ms / rate
        a = idx / n
        b = (idx + 1) / n
        sub_p = [_interp_list(pitch_5, a + (b - a) * (m / 4.0)) for m in range(5)]
        segs.append(
            PhoneSegment(
                ipa=ipa,
                lang="en",
                spec=sp,
                dur_ms=d,
                pitch_st_5=sub_p,
                gain=gain,
                unit_idx=unit_idx,
                stress=st,
            )
        )
    _apply_cluster_compression(segs)
    return segs


# ---------------------------------------------------------------------------
# Rule 2 & 3: English Stress/Reduction & Mandarin Tone Scaling
# ---------------------------------------------------------------------------
def rule_apply_stress_and_tone(segs: list[PhoneSegment], profile: SpeakerProfile) -> None:
    """Scale F0 excursion to speaker IQR and apply English stress timing/gain."""
    iqr_scale = max(0.55, min(1.85, profile.f0_iqr_st / 3.0))
    for seg in segs:
        if seg.is_pause:
            continue
        if seg.lang == "zh":
            # engine.CURVE averages ~ -0.80 st across Mandarin tones 1..4; offset +0.80 st
            # so the speaker's median F0 tracks profile.f0_median_hz.
            seg.pitch_st_5 = [(v + 0.80) * iqr_scale for v in seg.pitch_st_5]
            if seg.stress == 5:  # neutral tone softer
                seg.gain *= 0.88
        else:
            if seg.stress == 1:
                seg.dur_ms *= 1.18
                seg.gain *= 1.08
                seg.pitch_st_5 = [v * iqr_scale + 0.7 for v in seg.pitch_st_5]
            elif seg.stress == 2:
                seg.dur_ms *= 1.04
                seg.pitch_st_5 = [v * iqr_scale + 0.2 for v in seg.pitch_st_5]
            elif seg.stress == 0:
                seg.dur_ms = max(40.0, seg.dur_ms * 0.74)
                seg.gain *= 0.86
                seg.pitch_st_5 = [v * iqr_scale - 0.5 for v in seg.pitch_st_5]
            else:
                seg.pitch_st_5 = [v * iqr_scale for v in seg.pitch_st_5]


# ---------------------------------------------------------------------------
# Rule 4: Cross-Lingual Prosody & F0 Continuity Arbiter
# ---------------------------------------------------------------------------
def rule_crossling_arbiter(segs: list[PhoneSegment]) -> None:
    """Ensure smooth F0 continuity across ZH <-> EN switches & clause declination."""
    voiced_indices = [i for i, s in enumerate(segs) if not s.is_pause and s.spec.voiced]
    if not voiced_indices:
        return
    # Gentle clause-wide declination (-0.7 st start->end) so EN and ZH share one breath baseline
    total = max(1, len(segs) - 1)
    for i, seg in enumerate(segs):
        if seg.is_pause:
            continue
        decl = -0.70 * (i / total) + 0.35
        seg.pitch_st_5 = [max(-15.0, min(15.0, v + decl)) for v in seg.pitch_st_5]

    # Clamp abrupt boundary F0 jumps (> 2.2 st) between adjacent voiced segments
    for prev_i, cur_i in zip(voiced_indices[:-1], voiced_indices[1:]):
        if cur_i - prev_i > 2:
            continue  # separated by long unvoiced cluster or pause
        p_end = segs[prev_i].pitch_st_5[-1]
        c_start = segs[cur_i].pitch_st_5[0]
        diff = c_start - p_end
        if abs(diff) > 2.2:
            mid = 0.5 * (p_end + c_start)
            segs[prev_i].pitch_st_5[-1] = 0.5 * (p_end + mid)
            segs[cur_i].pitch_st_5[0] = 0.5 * (c_start + mid)


# ---------------------------------------------------------------------------
# Rule 5: Coarticulation Locus & Frame Trajectory Builder
# ---------------------------------------------------------------------------
def _interp_list(vals: list[float], t: float) -> float:
    p = max(0.0, min(1.0, t)) * (len(vals) - 1)
    i = min(int(p), len(vals) - 1)
    j = min(i + 1, len(vals) - 1)
    return vals[i] + (vals[j] - vals[i]) * (p - i)


def _cos_blend(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return 0.5 - 0.5 * math.cos(math.pi * u)


def build_frame_track(
    plan_doc: dict,
    profile: SpeakerProfile,
    *,
    lead_in_ms: float = 65.0,
    lead_out_ms: float = 95.0,
) -> FrameTrack:
    """Run all PyBot rules and produce 5 ms acoustic trajectory frames."""
    segs: list[PhoneSegment] = [
        PhoneSegment(
            ipa="sil", lang="sil", spec=IPA_SPECS["ə"],
            dur_ms=lead_in_ms, pitch_st_5=[0.0] * 5, gain=0.0, unit_idx=-1, is_pause=True,
        )
    ]
    for uidx, u in enumerate(plan_doc["units"]):
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
            dur_ms=lead_out_ms, pitch_st_5=[0.0] * 5, gain=0.0, unit_idx=-1, is_pause=True,
        )
    )

    rule_apply_stress_and_tone(segs, profile)
    rule_crossling_arbiter(segs)

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
        prev_seg = segs[s_idx - 1] if s_idx > 0 and not segs[s_idx - 1].is_pause else None
        next_seg = segs[s_idx + 1] if s_idx + 1 < len(segs) and not segs[s_idx + 1].is_pause else None

        f_start = np.array(sp.formants, dtype=np.float64) * vtln
        f_end = np.array(sp.end_formants if sp.end_formants else sp.formants, dtype=np.float64) * vtln

        # Unstressed English vowel centralization (rule-based vowel reduction)
        if seg.lang == "en" and seg.stress == 0 and sp.manner == "vowel":
            schwa = np.array(IPA_SPECS["ə"].formants, dtype=np.float64) * vtln
            f_start = 0.80 * f_start + 0.20 * schwa
            f_end = 0.80 * f_end + 0.20 * schwa

        # CV & VC locus edges for rapid, natural |dF2/dt| transitions
        f_cv = f_start.copy()
        if prev_seg is not None and prev_seg.spec.locus_f2_hz is not None and sp.manner in ("vowel", "diphthong", "glide", "liquid"):
            w = prev_seg.spec.locus_weight
            f_cv[1] = w * (prev_seg.spec.locus_f2_hz * vtln) + (1.0 - w) * f_start[1]
            f_cv[0] = 0.45 * (prev_seg.spec.formants[0] * vtln) + 0.55 * f_start[0]

        f_vc = f_end.copy()
        if next_seg is not None and next_seg.spec.locus_f2_hz is not None and sp.manner in ("vowel", "diphthong", "glide"):
            w = next_seg.spec.locus_weight * 0.85
            f_vc[1] = w * (next_seg.spec.locus_f2_hz * vtln) + (1.0 - w) * f_end[1]

        closure_frac = sp.closure_ms / max(1.0, sp.closure_ms + sp.vot_ms) if sp.manner in ("stop", "affricate") else 0.0

        for k in range(nf):
            t = k / max(1, nf - 1)
            f_idx = cursor + k
            p_st = _interp_list(seg.pitch_st_5, t)
            f0_cur = f0_ref * (2.0 ** (p_st / 12.0))

            # Formant trajectory with CV onset slide (first 32%) and VC coda slide (last 28%)
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

            # Anti-resonance zero (nasals/liquids + coarticulatory vowel nasalization before nasal)
            if sp.anti_zero is not None:
                zero_freq[f_idx] = sp.anti_zero[0] * vtln
                zero_bw[f_idx] = sp.anti_zero[1]
                zero_depth_db[f_idx] = sp.anti_zero[2]
            elif next_seg is not None and next_seg.spec.manner == "nasal" and t > 0.55 and sp.manner in ("vowel", "diphthong"):
                z_u = (t - 0.55) / 0.45
                zero_freq[f_idx] = next_seg.spec.anti_zero[0] * vtln
                zero_bw[f_idx] = next_seg.spec.anti_zero[1]
                zero_depth_db[f_idx] = next_seg.spec.anti_zero[2] * 0.65 * z_u

            # Excitation: voiced sonorants vs fricatives vs stop closure/burst/VOT
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
                # Stop or affricate: closure -> transient burst -> VOT/frication
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

    return FrameTrack(
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


def plan_from_pinyin_syllables(syllables: list[dict], *, text: str = "") -> dict:
    """Convert ``extended/index.json`` syllable records into a frontend-compatible plan."""
    units = []
    n = len(syllables)
    for i, row in enumerate(syllables):
        onset, final, tone, ipa = cf.pinyin_to_ipa(row["pinyin"])
        t_curve = list(cf.engine.CURVE[tone])
        if tone == 3 and i < n - 1:
            t_curve = [-1.4, -2.8, -4.0, -4.15, -3.8]
        decl = -0.32 * (i / max(1, n - 1))
        pitch_st = [round(v + decl, 2) for v in t_curve]
        units.append(
            {
                "lang": "zh",
                "hanzi": row.get("hanzi", "字"),
                "token": row.get("hanzi", "字"),
                "onset": onset,
                "final": final,
                "lexical_tone": tone,
                "tone": tone,
                "pinyin": row["pinyin"],
                "arpabet": None,
                "ipa_phones": ipa,
                "stress_or_tone": f"T{tone}",
                "duration_ms": float(row.get("ms", 230.0)),
                "gain": 0.83 if tone != 5 else 0.69,
                "pitch_st": pitch_st,
                "pause_after_ms": 0.0,
                "focus": False,
            }
        )
    return {"schema": 2, "text": text, "style": "auto", "languages": ["zh"], "units": units, "clauses": []}
