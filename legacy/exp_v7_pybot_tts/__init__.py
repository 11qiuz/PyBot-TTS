"""Phase 3 Hybrid PyBot Unit-Envelope Bank + HNM Vocoder + Cross-Lingual IPA Rules."""
from __future__ import annotations

from pathlib import Path

SR = 24000
HOP = 120
N_FFT = 1024
EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
EXTENDED_INDEX = REPO_ROOT / "pybot_tts_v2_recorded" / "extended" / "index.json"
