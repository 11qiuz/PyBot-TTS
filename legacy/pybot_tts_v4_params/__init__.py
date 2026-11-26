"""Parametric (source-filter) TTS scaffolding: phoneme alignment first, then LPC.

This is **not** speech recognition. Nothing here reads text, guesses characters or
claims to know what was said. Given *known* pinyin syllables (from
``engine.split_pinyin``) it only proposes where, in seconds, the onset/final and
the intra-final transitions of an already-cut recording sit, and it labels every
proposal with an honest confidence so downstream LPC analysis can refuse to trust
a shaky boundary instead of silently averaging across it.
"""
from __future__ import annotations

from pybot_tts_v4_params.align import (
    SR,
    Segment,
    boundaries,
    confidence_histogram,
    features,
    frames,
    refine_syllable,
    segment_word,
    syllable_segments,
)

__all__ = [
    "SR",
    "Segment",
    "boundaries",
    "confidence_histogram",
    "features",
    "frames",
    "refine_syllable",
    "segment_word",
    "syllable_segments",
]
