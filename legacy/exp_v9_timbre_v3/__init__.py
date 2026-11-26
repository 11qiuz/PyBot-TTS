"""Phase-5 non-neural UTAU-inspired timbre cloning package (`exp_v9_timbre_v3`)."""
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXP_DIR.parent
SR = 24000
N_FFT = 1024
HOP_SAMPLES = 120
EXTENDED_INDEX = ROOT_DIR / "pybot_tts_v2_recorded" / "extended" / "index.json"
MANIFEST_PATH = ROOT_DIR / "exp_v8_data" / "corpus" / "manifest.json"
