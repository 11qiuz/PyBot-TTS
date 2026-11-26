"""Zero-ML compact speaker voice clone profile extraction and base Stage 5 calibration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Iterable
import uuid

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt

from exp_20_pybot_larynx import EXP_DIR, HOP, N_BAP, N_FFT, SR
from exp_20_pybot_larynx.clone.audio_io import load_wav_file_to_24k_mono
from exp_v5_eval import metrics as eval_m
from exp_v9_clarity_pro.hnm_v9 import analyze_v9
from exp_v9_timbre_v3.eq_decoupler import DB_PER_NP, normalize_env_shared_band
from exp_v9_timbre_v3.timbre_cloner_v3 import (
    MANNER_NAMES,
    Q_ENV_STORE,
    Q_MANNER_STORE,
    _ceps_decode,
    _ceps_encode,
    _measure_bri_stats,
    _measure_h1_h2_db,
    classify_acoustic_manner,
)
from exp_v12_natural_vocoder.vocoder_v12 import _compute_lpc_and_residual

BASE_SPEAKER_PROFILE_PATH = EXP_DIR / "clone" / "base_speaker_profile.json"
_N_PHASE_CTRL = 32
_PHASE_CTRL_HZ = 3000.0


@lru_cache(maxsize=2)
def _syl_band_sos(sr: int = SR) -> np.ndarray:
    return butter(4, [300.0, min(3000.0, sr * 0.45)], btype="band", fs=float(sr), output="sos")


def _estimate_syllable_rate_hz(waves_24k: list[np.ndarray], sr: int = SR) -> float:
    """Estimate active syllable rate (Hz) via smoothed 300-3000 Hz envelope local peaks."""
    sos = _syl_band_sos(sr)
    step = max(1, int(round(0.010 * sr)))  # 10 ms step
    win = max(step, int(round(0.020 * sr)))  # 20 ms RMS window
    k_sm = np.hanning(6)
    k_sm /= np.sum(k_sm)

    tot_peaks = 0
    tot_active_s = 0.0
    for wav in waves_24k:
        arr = np.asarray(wav, dtype=np.float64).ravel()
        if len(arr) < win * 4:
            continue
        bp = sosfiltfilt(sos, arr)
        n_fr = 1 + (len(bp) - win) // step
        if n_fr < 12:
            continue
        idx = np.arange(n_fr)[:, None] * step + np.arange(win)[None, :]
        rms = np.sqrt(np.mean(bp[idx] ** 2, axis=1) + 1e-18)
        sm = np.convolve(rms, k_sm, mode="same")
        p95 = float(np.percentile(sm, 95))
        if p95 <= 1e-6:
            continue
        tot_active_s += float(np.sum(sm > 0.10 * p95)) * (step / float(sr))

        min_val = 0.18 * p95
        last_pk = -100
        for j in range(1, len(sm) - 1):
            if sm[j] >= min_val and sm[j] > sm[j - 1] and sm[j] >= sm[j + 1]:
                if j - last_pk >= 10:  # >= 100 ms spacing
                    tot_peaks += 1
                    last_pk = j

    if tot_active_s < 0.8 or tot_peaks < 2:
        return 4.3
    return round(float(np.clip(tot_peaks / tot_active_s, 2.2, 7.0)), 3)


def _extract_chest_glottal_phasor_32(
    wav: np.ndarray,
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    bap6: np.ndarray,
    *,
    sr: int = SR,
    hop: int = HOP,
    n_fft: int = N_FFT,
    max_frames: int = 80,
) -> tuple[np.ndarray, int]:
    """Extract normalized 32-point complex glottal phase over 0..3000 Hz from clean periodic cycles."""
    res = _compute_lpc_and_residual(wav, order=26, frame_len=600, hop=hop)
    n = len(wav)
    n_fr = len(f0_hz)
    freqs = rfftfreq(n_fft, 1.0 / sr)
    ctrl_f = np.linspace(0.0, _PHASE_CTRL_HZ, _N_PHASE_CTRL)
    bins_norm = np.arange(n_fft // 2 + 1, dtype=np.float64) / float(n_fft)

    valid = []
    for i in range(2, max(2, n_fr - 2)):
        if (
            voiced[i - 1]
            and voiced[i]
            and voiced[i + 1]
            and f0_hz[i] > 50.0
            and float(np.mean(bap6[i, :2])) < 0.34
        ):
            valid.append(i)

    if not valid:
        return np.ones(_N_PHASE_CTRL, dtype=np.complex128), 0

    if len(valid) > max_frames:
        sel = np.linspace(0, len(valid) - 1, max_frames, dtype=int)
        valid = [valid[k] for k in sel]

    phasors: list[np.ndarray] = []
    for i in valid:
        f0 = float(f0_hz[i])
        t0 = int(np.clip(round(sr / f0), 48, 360))
        c = i * hop
        s0 = max(0, c - t0 // 2)
        s1 = min(n, c + t0 // 2 + 1)
        if s1 <= s0 + 4:
            continue
        w_search = res[s0:s1]
        gci = s0 + int(np.argmax(np.abs(w_search)))
        sign = 1.0 if res[gci] >= 0.0 else -1.0

        L = min(n_fft - 4, max(32, int(round(1.15 * t0))))
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
        sp_al = sp_al / np.maximum(np.abs(sp_al), 1e-8)

        re_32 = np.interp(ctrl_f, freqs, sp_al.real)
        im_32 = np.interp(ctrl_f, freqs, sp_al.imag)
        ph_32 = re_32 + 1j * im_32
        ph_32 /= np.maximum(np.abs(ph_32), 1e-8)
        phasors.append(ph_32)

    if not phasors:
        return np.ones(_N_PHASE_CTRL, dtype=np.complex128), 0

    mean_ph = np.mean(np.stack(phasors, axis=0), axis=0)
    mean_ph /= np.maximum(np.abs(mean_ph), 1e-8)
    return mean_ph, len(phasors)


@dataclass
class VoiceCloneProfile:
    """Compact (~4 KB JSON) physically grounded zero-ML voice clone profile."""

    profile_id: str
    name: str
    created_at: str
    n_clips: int
    total_voiced_s: float
    effective_bw_hz: float
    f0_median_hz: float
    f0_log_std: float
    f0_iqr_st: float
    syllable_rate_hz: float
    f1_median_hz: float
    f2_median_hz: float
    f3_median_hz: float
    n_formant_frames: int
    tilt_db_per_oct: float
    hnr_median_db: float
    ope_h1_h2_db: float
    bri_formant_sharp_db: float
    bap6_voiced_median: list[float]
    voiced_ceps_q32: list[float]
    manner_deltas_q14: dict[str, list[float]]
    manner_counts: dict[str, int]
    unvoiced_tilt_slope: float
    unvoiced_rel_level_np: float
    glottal_phase_re: list[float]
    glottal_phase_im: list[float]
    glottal_phase_frames: int

    @property
    def voiced_env_mean(self) -> np.ndarray:
        return _ceps_decode(self.voiced_ceps_q32, n_fft=N_FFT)

    def reconstruct_coherent_phasor(self, n_fft: int = N_FFT, sr: int = SR) -> np.ndarray:
        """Reconstruct a unit-magnitude `(n_fft//2+1,)` chest-band glottal phasor."""
        n_bins = n_fft // 2 + 1
        if (
            self.glottal_phase_frames < 1
            or len(self.glottal_phase_re) != _N_PHASE_CTRL
            or len(self.glottal_phase_im) != _N_PHASE_CTRL
        ):
            return np.ones(n_bins, dtype=np.complex128)

        freqs = rfftfreq(n_fft, 1.0 / sr)
        ctrl_f = np.linspace(0.0, _PHASE_CTRL_HZ, _N_PHASE_CTRL)
        re = np.interp(freqs, ctrl_f, np.asarray(self.glottal_phase_re, dtype=np.float64), left=1.0, right=1.0)
        im = np.interp(freqs, ctrl_f, np.asarray(self.glottal_phase_im, dtype=np.float64), left=0.0, right=0.0)
        ph = re + 1j * im

        # Smooth cosine roll to neutral (1+0j) over 2400..3000 Hz so HF dispersion stays crisp
        u = np.clip((freqs - 2400.0) / 600.0, 0.0, 1.0)
        w = 0.5 * (1.0 + np.cos(math.pi * u))
        ph = w * ph + (1.0 - w) * 1.0
        return ph / np.maximum(np.abs(ph), 1e-8)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> VoiceCloneProfile:
        return cls(
            profile_id=str(doc["profile_id"]),
            name=str(doc["name"]),
            created_at=str(doc.get("created_at", "")),
            n_clips=int(doc["n_clips"]),
            total_voiced_s=float(doc["total_voiced_s"]),
            effective_bw_hz=float(doc.get("effective_bw_hz", 11000.0)),
            f0_median_hz=float(doc["f0_median_hz"]),
            f0_log_std=float(doc["f0_log_std"]),
            f0_iqr_st=float(doc["f0_iqr_st"]),
            syllable_rate_hz=float(doc.get("syllable_rate_hz", 4.3)),
            f1_median_hz=float(doc["f1_median_hz"]),
            f2_median_hz=float(doc["f2_median_hz"]),
            f3_median_hz=float(doc["f3_median_hz"]),
            n_formant_frames=int(doc["n_formant_frames"]),
            tilt_db_per_oct=float(doc["tilt_db_per_oct"]),
            hnr_median_db=float(doc["hnr_median_db"]),
            ope_h1_h2_db=float(doc["ope_h1_h2_db"]),
            bri_formant_sharp_db=float(doc["bri_formant_sharp_db"]),
            bap6_voiced_median=[float(x) for x in doc["bap6_voiced_median"]],
            voiced_ceps_q32=[float(x) for x in doc["voiced_ceps_q32"]],
            manner_deltas_q14={str(k): [float(x) for x in v] for k, v in doc["manner_deltas_q14"].items()},
            manner_counts={str(k): int(v) for k, v in doc["manner_counts"].items()},
            unvoiced_tilt_slope=float(doc["unvoiced_tilt_slope"]),
            unvoiced_rel_level_np=float(doc["unvoiced_rel_level_np"]),
            glottal_phase_re=[float(x) for x in doc["glottal_phase_re"]],
            glottal_phase_im=[float(x) for x in doc["glottal_phase_im"]],
            glottal_phase_frames=int(doc.get("glottal_phase_frames", 0)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> VoiceCloneProfile:
        return cls.from_dict(json.loads(raw))

    def serialized_size_bytes(self) -> int:
        return len(self.to_json().encode("utf-8"))

    def summary_dict(self) -> dict[str, Any]:
        if self.f0_median_hz < 155.0:
            hint = f"偏低男声 ({self.f0_median_hz:.0f} Hz)"
        elif self.f0_median_hz < 195.0:
            hint = f"中性/柔润嗓 ({self.f0_median_hz:.0f} Hz)"
        elif self.f0_median_hz < 265.0:
            hint = f"清亮女声 ({self.f0_median_hz:.0f} Hz)"
        else:
            hint = f"高音/童声 ({self.f0_median_hz:.0f} Hz)"
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "created_at": self.created_at,
            "n_clips": self.n_clips,
            "total_voiced_s": round(self.total_voiced_s, 2),
            "f0_median_hz": round(self.f0_median_hz, 1),
            "hnr_median_db": round(self.hnr_median_db, 2),
            "tilt_db_per_oct": round(self.tilt_db_per_oct, 2),
            "syllable_rate_hz": round(self.syllable_rate_hz, 2),
            "effective_bw_hz": round(self.effective_bw_hz, 0),
            "size_bytes": self.serialized_size_bytes(),
            "voice_hint": hint,
        }


def extract_clone_profile(
    waves_24k: Iterable[np.ndarray],
    *,
    name: str = "自定义音色",
    profile_id: str | None = None,
    orig_sr_hint: int = SR,
) -> VoiceCloneProfile:
    """Extract a compact (< 8 KB JSON) Stage-5-compatible voice clone profile from 1+ 24 kHz clips."""
    wave_list = [np.asarray(w, dtype=np.float64).ravel() for w in waves_24k if np.asarray(w).size > 0]
    if not wave_list:
        raise ValueError("没有可用于提取音色的有效音频片段。")

    freqs = rfftfreq(N_FFT, 1.0 / SR)
    f0_list: list[np.ndarray] = []
    env_list: list[np.ndarray] = []
    bap6_list: list[np.ndarray] = []
    manner_env_bins: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    tilts: list[float] = []
    hnrs: list[float] = []
    f1s: list[tuple[float, int]] = []
    f2s: list[tuple[float, int]] = []
    f3s: list[tuple[float, int]] = []
    h1_h2s: list[float] = []
    sharps: list[float] = []
    u_slopes: list[float] = []
    u_rels: list[float] = []
    phase_vecs: list[np.ndarray] = []
    phase_counts: list[int] = []
    n_fmt_total = 0

    max_ph_per_clip = max(24, 80 // max(1, len(wave_list)))
    for wav in wave_list:
        feats, b_gt8 = analyze_v9(wav, sr=SR, hop_ms=5.0, n_fft=N_FFT)
        bap6 = np.clip(np.concatenate([feats.bap[:, :4], b_gt8], axis=1), 0.005, 0.99)
        assert bap6.shape[1] == N_BAP

        v_mask = feats.voiced & (feats.f0_hz > 40.0)
        u_mask = ~v_mask
        if np.any(v_mask):
            f0_list.append(feats.f0_hz[v_mask])
            ve_norm = normalize_env_shared_band(feats.log_env[v_mask], freqs)
            env_list.append(ve_norm)
            bap6_list.append(bap6[v_mask])
            h1_h2s.append(_measure_h1_h2_db(wav, feats.f0_hz, v_mask, sr=SR, hop_samples=HOP))
            sh_db, _ = _measure_bri_stats(feats.log_env, v_mask, N_FFT, freqs)
            sharps.append(sh_db)

            m_labels = classify_acoustic_manner(feats.log_env, feats.voiced, feats.f0_hz, freqs)
            full_norm = normalize_env_shared_band(feats.log_env, freqs)
            for cls_id in (1, 2, 3):
                c_mask = m_labels == cls_id
                if np.any(c_mask):
                    manner_env_bins[cls_id].append(full_norm[c_mask])

            ph32, n_ph = _extract_chest_glottal_phasor_32(
                wav, feats.f0_hz, feats.voiced, bap6, sr=SR, hop=HOP, n_fft=N_FFT, max_frames=max_ph_per_clip
            )
            if n_ph > 0:
                phase_vecs.append(ph32 * float(n_ph))
                phase_counts.append(n_ph)

        if int(np.sum(u_mask)) >= 3 and int(np.sum(v_mask)) >= 3:
            u_mean = np.mean(feats.log_env[u_mask], axis=0)
            m_hi = (freqs >= 1000.0) & (freqs <= 7200.0)
            slope_np, _ = np.polyfit(np.log2(freqs[m_hi]), u_mean[m_hi], 1)
            m_uv_lvl = (freqs >= 250.0) & (freqs <= 7200.0)
            v_lvl = float(np.median(np.mean(feats.log_env[v_mask][:, m_uv_lvl], axis=1)))
            u_lvl = float(np.median(np.mean(feats.log_env[u_mask][:, m_uv_lvl], axis=1)))
            u_slopes.append(float(slope_np))
            u_rels.append(float(np.clip(u_lvl - v_lvl, -3.5, 0.5)))

        lt = eval_m.compute_ltas_metrics(wav, sr=SR)
        if lt and lt.get("tilt_db_oct") is not None and math.isfinite(float(lt["tilt_db_oct"])):
            tilts.append(float(lt["tilt_db_oct"]))
        f0_m = eval_m.compute_f0_hnr_metrics(wav, sr=SR)
        if f0_m and f0_m.get("hnr_db_median") is not None and math.isfinite(float(f0_m["hnr_db_median"])):
            hnrs.append(float(f0_m["hnr_db_median"]))
        fm = eval_m.compute_formant_metrics(wav, sr=SR)
        nf = int(fm.get("n_formant_frames", 0)) if fm else 0
        if fm and nf > 0 and fm.get("f1_median_hz") is not None and fm.get("f2_median_hz") is not None:
            f1s.append((float(fm["f1_median_hz"]), nf))
            f2s.append((float(fm["f2_median_hz"]), nf))
            if fm.get("f3_median_hz") is not None:
                f3s.append((float(fm["f3_median_hz"]), nf))
            n_fmt_total += nf

    f0_all = np.concatenate(f0_list) if f0_list else np.zeros(0, dtype=np.float64)
    total_voiced_s = float(len(f0_all) * HOP / float(SR))
    if total_voiced_s < 0.5:
        raise ValueError("参考音频中有效人声不足，请提供至少 1 秒清晰语音。")

    log_f0 = np.log(f0_all)
    f0_med = float(np.median(f0_all))
    f0_std = float(np.clip(np.std(log_f0), 0.05, 0.45)) if len(log_f0) > 2 else 0.14
    iqr_st = (
        float(12.0 * (np.percentile(log_f0, 75) - np.percentile(log_f0, 25)) / math.log(2.0))
        if len(log_f0) > 4
        else 3.0
    )

    env_all = np.concatenate(env_list, axis=0)
    voiced_env_raw = normalize_env_shared_band(np.mean(env_all, axis=0), freqs)
    voiced_ceps_q32 = _ceps_encode(voiced_env_raw, n_fft=N_FFT, q_max=Q_ENV_STORE)
    voiced_env_smooth = _ceps_decode(voiced_ceps_q32, n_fft=N_FFT)

    m_mid = (freqs >= 2000.0) & (freqs <= 5000.0)
    m_uhi = (freqs >= 7800.0) & (freqs <= 11000.0)
    bw_drop = float(np.mean(voiced_env_raw[m_uhi]) - np.mean(voiced_env_raw[m_mid]))
    effective_bw_hz = 7200.0 if (int(orig_sr_hint) <= 16500 or bw_drop < -2.6) else 11000.0

    bap6_all = np.concatenate(bap6_list, axis=0)
    bap6_med = [round(float(x), 4) for x in np.clip(np.median(bap6_all, axis=0), 0.005, 0.985)]

    manner_deltas_q14: dict[str, list[float]] = {}
    manner_counts: dict[str, int] = {}
    for cls_id, cls_name in zip((1, 2, 3), MANNER_NAMES):
        chunks = manner_env_bins[cls_id]
        if chunks:
            arr = np.concatenate(chunks, axis=0)
            manner_counts[cls_name] = int(len(arr))
            cls_diff = normalize_env_shared_band(np.mean(arr, axis=0) - voiced_env_smooth, freqs)
            manner_deltas_q14[cls_name] = _ceps_encode(cls_diff, n_fft=N_FFT, q_max=Q_MANNER_STORE)
        else:
            manner_counts[cls_name] = 0
            manner_deltas_q14[cls_name] = [0.0] * (Q_MANNER_STORE + 1)

    tot_ph_frames = int(sum(phase_counts))
    if tot_ph_frames > 0:
        ph_sum = np.sum(np.stack(phase_vecs, axis=0), axis=0)
        ph_mean = ph_sum / np.maximum(np.abs(ph_sum), 1e-8)
    else:
        ph_mean = np.ones(_N_PHASE_CTRL, dtype=np.complex128)

    pid = profile_id or uuid.uuid4().hex[:10]
    syl_rate = _estimate_syllable_rate_hz(wave_list, sr=SR)

    return VoiceCloneProfile(
        profile_id=str(pid),
        name=str(name).strip() or "自定义音色",
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_clips=len(wave_list),
        total_voiced_s=round(total_voiced_s, 3),
        effective_bw_hz=float(effective_bw_hz),
        f0_median_hz=round(f0_med, 2),
        f0_log_std=round(f0_std, 4),
        f0_iqr_st=round(float(np.clip(iqr_st, 1.0, 9.0)), 3),
        syllable_rate_hz=syl_rate,
        f1_median_hz=round(float(np.average([v for v, _ in f1s], weights=[w for _, w in f1s])) if f1s else 500.0, 1),
        f2_median_hz=round(float(np.average([v for v, _ in f2s], weights=[w for _, w in f2s])) if f2s else 1750.0, 1),
        f3_median_hz=round(float(np.average([v for v, _ in f3s], weights=[w for _, w in f3s])) if f3s else 2900.0, 1),
        n_formant_frames=int(n_fmt_total),
        tilt_db_per_oct=round(float(np.median(tilts)) if tilts else -3.6, 3),
        hnr_median_db=round(float(np.median(hnrs)) if hnrs else 14.0, 2),
        ope_h1_h2_db=round(float(np.median(h1_h2s)) if h1_h2s else 4.0, 3),
        bri_formant_sharp_db=round(float(np.median(sharps)) if sharps else 4.0, 3),
        bap6_voiced_median=bap6_med,
        voiced_ceps_q32=[round(float(x), 4) for x in voiced_ceps_q32],
        manner_deltas_q14={k: [round(float(x), 4) for x in v] for k, v in manner_deltas_q14.items()},
        manner_counts=manner_counts,
        unvoiced_tilt_slope=round(float(np.median(u_slopes)) if u_slopes else 0.0, 4),
        unvoiced_rel_level_np=round(float(np.median(u_rels)) if u_rels else -1.2, 4),
        glottal_phase_re=[round(float(x), 5) for x in ph_mean.real],
        glottal_phase_im=[round(float(x), 5) for x in ph_mean.imag],
        glottal_phase_frames=tot_ph_frames,
    )


def extract_clone_profile_from_wavs(
    wav_paths: Iterable[Path | str],
    *,
    name: str = "自定义音色",
    profile_id: str | None = None,
) -> VoiceCloneProfile:
    """Load WAV files from disk, convert to 24 kHz mono, and extract `VoiceCloneProfile`."""
    waves: list[np.ndarray] = []
    sr_hints: list[int] = []
    for p in wav_paths:
        w, info = load_wav_file_to_24k_mono(p)
        waves.append(w)
        sr_hints.append(int(info["orig_sr"]))
    return extract_clone_profile(
        waves,
        name=name,
        profile_id=profile_id,
        orig_sr_hint=min(sr_hints) if sr_hints else SR,
    )


def build_base_speaker_profile(
    out_path: Path = BASE_SPEAKER_PROFILE_PATH,
    *,
    n_utts: int = 12,
) -> VoiceCloneProfile:
    """Build analyzer-matched Stage 5 base speaker profile strictly from `TRAIN_IDS`."""
    from exp_20_pybot_larynx.acoustics.knowledge import get_splits, load_v15_index_cached
    from exp_20_pybot_larynx.acoustics.knowledge_stage5 import AcousticKnowledgeStage5
    from exp_20_pybot_larynx.arbiter import Arbiter
    from exp_20_pybot_larynx.frontend.prosody_stage5 import render_stage5_utterance
    from exp_20_pybot_larynx.rules import default_stage5_rules
    from exp_v14_fluent_zh_tts.eval_v14 import _rms_norm

    train_ids, dev_60, held_20 = get_splits()
    assert not (set(train_ids) & set(dev_60)) and not (set(train_ids) & set(held_20))
    idx_doc = load_v15_index_cached()
    metas = idx_doc["train_utt_metas"]

    picked: list[str] = []
    step = max(1, min(len(train_ids), 1500) // max(1, n_utts))
    for uid in train_ids[::step]:
        m = metas.get(uid)
        if m and m.get("text"):
            picked.append(uid)
        if len(picked) >= n_utts:
            break

    kb5 = AcousticKnowledgeStage5.load()
    arbiter = Arbiter()
    rules5 = default_stage5_rules()

    waves: list[np.ndarray] = []
    for idx_u, uid in enumerate(picked):
        txt = str(metas[uid]["text"])
        wav, _ = render_stage5_utterance(
            txt,
            f"base_prof_{uid}",
            kb5,
            arbiter=arbiter,
            rules=rules5,
            glottal_mode="parametric_lf",
            seed=7000 + idx_u,
        )
        waves.append(_rms_norm(wav, target_dbfs=-20.0))

    prof = extract_clone_profile(waves, name="BZNSYP_STAGE5_BASE", profile_id="base_stage5", orig_sr_hint=SR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prof.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_base_speaker_profile.cache_clear()
    return prof


@lru_cache(maxsize=2)
def load_base_speaker_profile(path: Path | str = BASE_SPEAKER_PROFILE_PATH) -> VoiceCloneProfile:
    """Load the cached analyzer-matched Stage 5 base speaker profile (building if absent)."""
    p = Path(path)
    if not p.exists():
        return build_base_speaker_profile(p)
    return VoiceCloneProfile.from_json(p.read_text(encoding="utf-8"))
