"""Apply local timing, pitch and gain edits to the v10 vocoder tracks."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from exp_v6_vocoder.vocoder import VocoderFeatures
from exp_v8_clarity.utau_concat import _resample_1d, _resample_2d
from exp_v10_integrity import HOP, SR
from exp_v10_integrity.integrity_gate import PERCEPTUAL_MIN_FRAMES


FRAME_MS = HOP * 1000 // SR


@dataclass(frozen=True)
class PhoneSpan:
    unit_index: int  # zero-based index into the plan
    part: int | str  # one-based English phone number, or Mandarin onset/rhyme
    label: str
    start: int
    end: int
    min_frames: int

    @property
    def id(self) -> str:
        return f"{self.unit_index + 1}:{self.part}"


@dataclass(frozen=True)
class PhoneEdit:
    duration_ms: float | None
    pitch_st: tuple[float, ...]
    gain_db: float


def _bounded(value: object, lower: float, upper: float, field: str) -> float:
    if type(value) not in (int, float) or not lower <= value <= upper:
        raise ValueError(f"{field} must be a number in [{lower}, {upper}]")
    return float(value)


def parse_phone_edits(value: object) -> dict[str, PhoneEdit]:
    """Validate the editable JSON fields before assembling any acoustic tracks."""
    if not isinstance(value, dict):
        raise ValueError("Phone edits must be an object keyed by phone ID")
    edits = {}
    fields = {"duration_ms", "pitch_st", "gain_db"}
    for phone_id, row in value.items():
        if not isinstance(phone_id, str) or not phone_id:
            raise ValueError("Phone edit IDs must be non-empty strings")
        if not isinstance(row, dict) or not row or row.keys() - fields:
            raise ValueError(f"Phone {phone_id}: supply duration_ms, pitch_st and/or gain_db")
        pitch = row.get("pitch_st", 0.0)
        if isinstance(pitch, list):
            if len(pitch) != 5:
                raise ValueError(f"{phone_id}.pitch_st requires five offsets")
        else:
            pitch = [pitch]
        edits[phone_id] = PhoneEdit(
            duration_ms=(
                _bounded(row["duration_ms"], FRAME_MS, 2000, f"{phone_id}.duration_ms")
                if "duration_ms" in row else None
            ),
            pitch_st=tuple(_bounded(p, -18, 18, f"{phone_id}.pitch_st") for p in pitch),
            gain_db=_bounded(row.get("gain_db", 0.0), -12, 12, f"{phone_id}.gain_db"),
        )
    return edits


def apply_phone_edits(
    feats: VocoderFeatures,
    bap_gt8k: np.ndarray,
    residuals: list[tuple[int, int, np.ndarray]],
    spans: list[PhoneSpan],
    edits: dict[str, PhoneEdit],
) -> tuple[VocoderFeatures, np.ndarray, list[tuple[int, int, np.ndarray]], list[PhoneSpan]]:
    """Keep the baseline donor selection and stretch only the addressed frame span."""
    missing = edits.keys() - {span.id for span in spans}
    if missing:
        raise ValueError(f"Unknown or unaligned phone ID(s): {', '.join(sorted(missing))}. See --trace.")

    tracks = (feats.log_env, feats.bap, bap_gt8k, feats.voiced, feats.f0_hz)
    chunks = []
    updated_spans = []
    last = offset = 0
    zh_totals: dict[int, int] = {}
    for span in spans:
        length = span.end - span.start
        count = length
        edit = edits.get(span.id)
        if edit is not None and edit.duration_ms is not None:
            floor_ms = span.min_frames * FRAME_MS
            if edit.duration_ms < floor_ms:
                raise ValueError(f"Phone {span.id}: duration_ms must be at least {floor_ms} ms")
            count = round(edit.duration_ms / FRAME_MS)

        if isinstance(span.part, str):
            zh_totals[span.unit_index] = zh_totals.get(span.unit_index, 0) + count

        if last < span.start:
            chunks.append(tuple(track[last:span.start] for track in tracks))
        segment = tuple(track[span.start:span.end] for track in tracks)
        if edit is not None:
            segment = _edit_tracks(segment, count, edit, span.id)
        chunks.append(segment)

        start = span.start + offset
        updated_spans.append(replace(span, start=start, end=start + count))
        last = span.end
        offset += count - length

    for unit, length in zh_totals.items():
        if length < PERCEPTUAL_MIN_FRAMES["zh_syllable"]:
            floor_ms = PERCEPTUAL_MIN_FRAMES["zh_syllable"] * FRAME_MS
            raise ValueError(f"Mandarin unit {unit + 1}: onset and rhyme must total at least {floor_ms} ms")

    chunks.append(tuple(track[last:] for track in tracks))
    env, bap, bg, voiced, f0 = (np.concatenate(col, axis=0) for col in zip(*chunks))
    # The renderer only adds a residual for a complete onset or English phone.
    new_bounds = {(old.start, old.end): (new.start, new.end) for old, new in zip(spans, updated_spans)}
    shifted_residuals = [(*new_bounds[(start, end)], wav) for start, end, wav in residuals]
    updated_feats = replace(feats, log_env=env, bap=bap, voiced=voiced, f0_hz=f0, n_samples=len(f0) * HOP)
    return updated_feats, bg, shifted_residuals, updated_spans


def _edit_tracks(segment: tuple, count: int, edit: PhoneEdit, phone_id: str) -> tuple:
    env, bap, bg, voiced, f0 = segment
    if count != len(f0):
        env, bap, bg = (_resample_2d(track, count) for track in (env, bap, bg))
        source_voiced = voiced
        voiced = _resample_1d(source_voiced.astype(np.float64), count) >= 0.45
        voiced_indices = np.flatnonzero(source_voiced)
        f0 = (
            np.where(voiced, np.interp(np.linspace(0, len(f0) - 1, count), voiced_indices, f0[source_voiced]), 0.0)
            if voiced_indices.size else np.zeros(count, dtype=np.float64)
        )

    if not edit.gain_db and not any(edit.pitch_st):
        return env, bap, bg, voiced, f0

    # Taper the offset within the phone so neighboring track values stay untouched.
    fade_frames = min(2, (count - 1) // 2)
    taper = np.ones(count)
    if fade_frames:
        ramp = np.arange(fade_frames) / fade_frames
        taper[:fade_frames] = ramp
        taper[-fade_frames:] = ramp[::-1]
    if edit.gain_db:
        env = env + (taper * (edit.gain_db * math.log(10) / 20))[:, None]
    if any(edit.pitch_st):
        if not np.any(voiced):
            raise ValueError(f"Phone {phone_id}: pitch_st requires voiced frames")
        offsets = np.interp(np.linspace(0, 1, count), np.linspace(0, 1, len(edit.pitch_st)), edit.pitch_st)
        f0 = np.where(voiced, np.clip(f0 * np.exp2(taper * offsets / 12), 65.0, 440.0), 0.0)
    return env, bap, bg, voiced, f0
