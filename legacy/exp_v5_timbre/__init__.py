"""Experimental v5 timbre, glottal source, and formant trajectory probes."""
from __future__ import annotations

from pathlib import Path

SR = 24000
EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
EXTENDED_INDEX = REPO_ROOT / "pybot_tts_v2_recorded" / "extended" / "index.json"
PHONEME_PARAMS = REPO_ROOT / "pybot_tts_v4_params" / "phoneme_params.json"
SPEAKER_PARAMS = REPO_ROOT / "pybot_tts_v4_params" / "speaker_params.json"
SAMPLES_DIR = EXP_DIR / "samples"
