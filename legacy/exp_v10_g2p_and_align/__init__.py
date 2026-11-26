"""Phase-6 G2P integrity, manner-consistent forced alignment, and V/U-calibrated timbre cloning."""
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXP_DIR.parent
LEXICON_DIR = EXP_DIR / "lexicons"
EXTENDED_INDEX = ROOT_DIR / "pybot_tts_v2_recorded" / "extended" / "index.json"
MANIFEST_PATH = ROOT_DIR / "exp_v8_data" / "corpus" / "manifest.json"
SR = 24000
HOP = 120
N_FFT = 1024
