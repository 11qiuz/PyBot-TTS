"""Phase-7 bootstrapped non-neural GMM-HMM forced aligner & TimbreClonerV5 (`exp_v11_align_gmm`)."""
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXP_DIR.parent
EXTENDED_INDEX = ROOT_DIR / "pybot_tts_v2_recorded" / "extended" / "index.json"
MANIFEST_PATH = ROOT_DIR / "exp_v8_data" / "corpus" / "manifest.json"
SR = 24000
HOP = 120
N_FFT = 1024
