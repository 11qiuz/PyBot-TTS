"""Phase 5 Clarity Pro: Acoustic Landmark Boundary Refiner, HD HNM Unvoiced/Sibilant Excitation, and Viterbi UTAU + Natural Prosody."""
from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
EXTENDED_INDEX = REPO_ROOT / "pybot_tts_v2_recorded" / "extended" / "index.json"
EN_MANIFEST = REPO_ROOT / "exp_v8_data" / "corpus" / "manifest.json"
SAMPLES_DIR = EXP_DIR / "samples"

SR = 24000
HOP = 120       # 5 ms @ 24 kHz
N_FFT = 1024    # 513 frequency bins
