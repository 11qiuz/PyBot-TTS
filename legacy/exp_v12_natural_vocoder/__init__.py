"""Phase 8 (v12): Natural LP-Residual Excitation, Crisp Unvoiced Bursts & Integrated Renderer."""
from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
EXTENDED_INDEX = REPO_ROOT / "pybot_tts_v2_recorded" / "extended" / "index.json"
EN_MANIFEST = REPO_ROOT / "exp_v8_data" / "corpus" / "manifest.json"
V12_MANIFEST = REPO_ROOT / "exp_v12_data" / "corpus" / "manifest_v12.json"
BENCHMARK_TEXTS = REPO_ROOT / "exp_v5_arch" / "benchmark_texts.json"
SAMPLES_DIR = EXP_DIR / "samples"

SR = 24000
HOP = 120
N_FFT = 1024
