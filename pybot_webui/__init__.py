"""PyBot-Larynx local WebUI voice studio package."""
from __future__ import annotations

from pathlib import Path
import sys

WEBUI_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBUI_DIR.parent
_LEGACY_DIR = REPO_ROOT / "legacy"
if _LEGACY_DIR.is_dir() and str(_LEGACY_DIR) not in sys.path:
    sys.path.append(str(_LEGACY_DIR))
STATIC_DIR = WEBUI_DIR / "static"
DEFAULT_SAVE_DIR = REPO_ROOT / "webui_out"

SR = 24000
HOP = 120
N_FFT = 1024
N_BINS = N_FFT // 2 + 1
N_BAP = 6
