"""Phase 11 (v15): Large-Scale 10,000-Utterance Studio Mandarin Non-Neural TTS (`BZNSYP` / CSMSC)."""
from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
RAW_CACHE_DIR = EXP_DIR / "raw_cache"
BZNSYP_DIR = EXP_DIR / "bznsyp_data"
INDEX_V15 = EXP_DIR / "bznsyp_index_v15.pkl"
PROSODY_MODEL_V15 = EXP_DIR / "prosody_v15_bznsyp.pkl"
SPEAKER_ASSETS_V15 = EXP_DIR / "bznsyp_speaker_assets.pkl"
SAMPLES_DIR = EXP_DIR / "samples"

SR = 24000
HOP = 120
N_FFT = 1024

# Pinned deterministically from verified clean alignments (12 statements, 4 questions ?, 4 !/multi-clause)
BZNSYP_HELDOUT_20: tuple[str, ...] = (
    "001395",
    "004343",
    "006484",
    "009118",
    "001309",
    "009210",
    "001222",
    "009086",
    "000547",
    "000988",
    "001526",
    "002100",
    "002724",
    "003366",
    "004112",
    "004885",
    "005750",
    "006801",
    "007631",
    "008651",
)
