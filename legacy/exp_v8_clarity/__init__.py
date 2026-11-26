"""Phase 4 Clarity Root-Cause Ablation & UTAU-Style Full-Resolution Concatenative-HNM Renderer."""
from __future__ import annotations

from pathlib import Path

SR = 24000
HOP = 120
N_FFT = 1024
EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
EXTENDED_INDEX = REPO_ROOT / "pybot_tts_v2_recorded" / "extended" / "index.json"
SAMPLES_DIR = EXP_DIR / "samples"
