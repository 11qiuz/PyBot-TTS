"""PyBot-Larynx (exp_20): Zero-weight, zero-concatenation electronic vocal cord TTS."""
from __future__ import annotations

from pathlib import Path
import sys

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
_LEGACY_DIR = REPO_ROOT / "legacy"
if _LEGACY_DIR.is_dir() and str(_LEGACY_DIR) not in sys.path:
    sys.path.append(str(_LEGACY_DIR))

SR = 24000
HOP = 120
N_FFT = 1024
N_BINS = N_FFT // 2 + 1
N_CEP = 64
N_BAP = 6
N_ONSET_BINS = 3
N_FINAL_BINS = 5

DEV_60_SEED = 2026
TRAIN_STAGE0_N = 1200
KNOWLEDGE_PATH = EXP_DIR / "acoustics" / "knowledge_stage0.npz"
KNOWLEDGE_SUMMARY_PATH = EXP_DIR / "acoustics" / "knowledge_summary.json"
SAMPLES_DIR = EXP_DIR / "samples"
