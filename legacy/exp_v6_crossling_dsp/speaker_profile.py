"""Compact few-shot speaker timbre profile (extractable from 3-10 s of speech).

Designed to stay well under 1 KB JSON per speaker while capturing the macro
timbre axes proven separable in v4/v5 audits:
- F0 median (Hz), IQR (st), 5-95% range (st)
- LTAS tilt (dB/oct over 80-8000 Hz) + 8-band residual EQ (dB)
- Boersma-style HNR (dB)
- Bounded VTLN formant scale (female ~1.00, male baritone ~0.84)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from scipy.fft import rfft, rfftfreq

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_eval import metrics as eval_metrics

EQ_BANDS_HZ: tuple[float, ...] = (200.0, 400.0, 800.0, 1400.0, 2400.0, 3800.0, 5600.0, 7600.0)


@dataclass(frozen=True)
class SpeakerProfile:
    name: str
    f0_median_hz: float
    f0_iqr_st: float
    f0_range_st: float
    ltas_tilt_db_per_oct: float
    residual_eq_db: tuple[float, ...]  # 8 points at EQ_BANDS_HZ
    hnr_db: float
    vtln_scale: float
    speech_rate_scale: float = 1.0
    n_utterances: int = 0
    total_active_s: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["residual_eq_db"] = [round(float(x), 3) for x in self.residual_eq_db]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, doc: dict) -> "SpeakerProfile":
        return cls(
            name=str(doc["name"]),
            f0_median_hz=float(doc["f0_median_hz"]),
            f0_iqr_st=float(doc["f0_iqr_st"]),
            f0_range_st=float(doc["f0_range_st"]),
            ltas_tilt_db_per_oct=float(doc["ltas_tilt_db_per_oct"]),
            residual_eq_db=tuple(float(x) for x in doc["residual_eq_db"]),
            hnr_db=float(doc["hnr_db"]),
            vtln_scale=float(doc["vtln_scale"]),
            speech_rate_scale=float(doc.get("speech_rate_scale", 1.0)),
            n_utterances=int(doc.get("n_utterances", 0)),
            total_active_s=float(doc.get("total_active_s", 0.0)),
        )


def neutral_female_reference() -> SpeakerProfile:
    """Neutral female reference matching v6 base IPA formants."""
    return SpeakerProfile(
        name="NEUTRAL_FEMALE",
        f0_median_hz=215.0,
        f0_iqr_st=3.0,
        f0_range_st=7.0,
        ltas_tilt_db_per_oct=-3.6,
        residual_eq_db=(0.0,) * len(EQ_BANDS_HZ),
        hnr_db=13.0,
        vtln_scale=1.0,
        speech_rate_scale=1.0,
    )


def male_baritone_115() -> SpeakerProfile:
    """Synthetic male baritone preset (F0=115 Hz, scaled male vocal tract)."""
    return SpeakerProfile(
        name="MALE_115",
        f0_median_hz=115.0,
        f0_iqr_st=2.5,
        f0_range_st=6.2,
        ltas_tilt_db_per_oct=-4.6,
        residual_eq_db=(1.2, 0.8, 0.0, -0.8, -1.2, -1.6, -2.0, -2.4),
        hnr_db=11.5,
        vtln_scale=0.84,
        speech_rate_scale=1.0,
    )


def _estimate_8band_residual(active_frames: np.ndarray, sr: int, tilt_db_oct: float) -> tuple[float, ...]:
    """Compute zero-mean 8-band LTAS deviation (dB) relative to the linear tilt line."""
    if active_frames.shape[0] == 0:
        return (0.0,) * len(EQ_BANDS_HZ)
    win = np.hamming(active_frames.shape[1])
    centered = (active_frames - np.mean(active_frames, axis=1, keepdims=True)) * win[None, :]
    n_fft = 2048
    mag = np.abs(rfft(centered, n=n_fft, axis=1))
    power_mean = np.sqrt(np.mean(mag * mag, axis=0))
    freqs = rfftfreq(n_fft, 1.0 / sr)

    band_db = []
    for fc in EQ_BANDS_HZ:
        lo, hi = fc * (2.0 ** (-1.0 / 3.0)), fc * (2.0 ** (1.0 / 3.0))
        mask = (freqs >= lo) & (freqs <= hi)
        val = float(np.mean(power_mean[mask])) if np.any(mask) else float(power_mean[int(np.argmin(np.abs(freqs - fc)))])
        band_db.append(20.0 * math.log10(max(val, 1e-12)))

    band_arr = np.asarray(band_db, dtype=np.float64)
    log2f = np.log2(np.asarray(EQ_BANDS_HZ, dtype=np.float64))
    slope, intercept = np.polyfit(log2f, band_arr, 1)
    # Subtract the fitted linear slope+intercept so residual_eq_db captures curvature only.
    res = band_arr - (slope * log2f + intercept)
    res = res - np.mean(res)
    # Clamp conservatively to avoid fitting utterance-specific phoneme bias on 3-10 s prompts.
    res = np.clip(res, -4.5, 4.5)
    return tuple(round(float(v), 3) for v in res)


def _estimate_vtln_scale(f0_med: float, f2_med: float | None) -> float:
    """Conservative VTLN scale bounded against noisy few-shot formant scatter."""
    if f0_med < 165.0:
        # Smooth male-to-female vocal tract length ramp [0.83 .. 0.94]
        u = max(0.0, min(1.0, (f0_med - 100.0) / 65.0))
        return round(0.83 + 0.11 * u, 4)
    if f2_med is not None and math.isfinite(f2_med) and 1100.0 <= f2_med <= 2600.0:
        raw = f2_med / 1810.0
        # Heavy shrinkage toward 1.0 because v4 proved F1/F2 scatter dominates within gender.
        shrunk = 1.0 + 0.15 * (raw - 1.0)
        return round(float(max(0.96, min(1.04, shrunk))), 4)
    return 1.0


def extract_profile_from_arrays(
    waves: Iterable[tuple[np.ndarray, int]],
    *,
    name: str = "SPEAKER",
) -> SpeakerProfile:
    """Extract a compact :class:`SpeakerProfile` from 1+ decoded waveforms."""
    f0_meds: list[float] = []
    f0_iqrs: list[float] = []
    f0_rngs: list[float] = []
    tilts: list[float] = []
    hnrs: list[float] = []
    f2s: list[float] = []
    active_chunks: list[np.ndarray] = []
    total_active_s = 0.0
    n_utts = 0

    for x, sr in waves:
        if sr != 24000:
            raise ValueError(f"Expected 24000 Hz audio, got {sr}")
        arr = np.asarray(x, dtype=np.float64).ravel()
        frames, keep, _ = eval_metrics.extract_active_frames(arr, frame_len=720, hop=240, db_thresh=30.0)
        kept = frames[keep]
        if kept.shape[0] < 4:
            continue
        n_utts += 1
        total_active_s += kept.shape[0] * 0.010  # 10 ms hop
        active_chunks.append(kept)

        ltas = eval_metrics.compute_ltas_metrics(arr, sr)
        if ltas and ltas.get("tilt_db_oct") is not None:
            tilts.append(float(ltas["tilt_db_oct"]))

        f0_hnr = eval_metrics.compute_f0_hnr_metrics(arr, sr)
        if f0_hnr:
            if f0_hnr.get("hnr_db_median") is not None:
                hnrs.append(float(f0_hnr["hnr_db_median"]))
            if f0_hnr.get("f0_median_hz") is not None:
                f0_meds.append(float(f0_hnr["f0_median_hz"]))
                f0_iqrs.append(float(f0_hnr["f0_iqr_st"]))
                f0_rngs.append(float(f0_hnr["f0_range_5_95_st"]))

        fmt = eval_metrics.compute_formant_metrics(arr, sr)
        if fmt and fmt.get("f2_median_hz") is not None:
            f2s.append(float(fmt["f2_median_hz"]))

    if n_utts == 0 or not f0_meds:
        raise ValueError(f"Insufficient voiced speech to build profile for {name!r}")

    f0_med = float(np.median(f0_meds))
    f0_iqr = float(np.median(f0_iqrs))
    f0_rng = float(np.median(f0_rngs))
    tilt_med = float(np.median(tilts)) if tilts else -3.6
    hnr_med = float(np.median(hnrs)) if hnrs else 12.0
    f2_med = float(np.median(f2s)) if f2s else None
    merged_frames = np.vstack(active_chunks)
    res_eq = _estimate_8band_residual(merged_frames, 24000, tilt_med)
    vtln = _estimate_vtln_scale(f0_med, f2_med)

    return SpeakerProfile(
        name=name,
        f0_median_hz=round(f0_med, 2),
        f0_iqr_st=round(max(1.0, min(6.5, f0_iqr)), 3),
        f0_range_st=round(max(2.5, min(14.0, f0_rng)), 3),
        ltas_tilt_db_per_oct=round(tilt_med, 3),
        residual_eq_db=res_eq,
        hnr_db=round(max(4.0, min(22.0, hnr_med)), 2),
        vtln_scale=vtln,
        speech_rate_scale=1.0,
        n_utterances=n_utts,
        total_active_s=round(total_active_s, 3),
    )


def extract_profile_from_wavs(wav_paths: Iterable[str | Path], *, name: str) -> SpeakerProfile:
    """Load WAV files via ``exp_v5_eval.metrics.read_wav`` and extract profile."""
    waves = [eval_metrics.read_wav(p) for p in wav_paths]
    return extract_profile_from_arrays(waves, name=name)
