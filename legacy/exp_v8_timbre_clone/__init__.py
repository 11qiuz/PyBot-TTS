"""Phase-4 Non-Neural Few-Shot Timbre Cloning v2 (`exp_v8_timbre_clone`)."""
from pathlib import Path

SR = 24000
EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parent
EXTENDED_INDEX = ROOT / "pybot_tts_v2_recorded" / "extended" / "index.json"
SAMPLES_DIR = EXP_DIR / "samples"
