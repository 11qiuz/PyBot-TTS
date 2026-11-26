"""Phase-8 non-neural learned prosody predictor & cross-lingual seam bridge (`exp_v12_prosody`)."""
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXP_DIR.parent
DEFAULT_MODEL_PATH = EXP_DIR / "prosody_v12_model.pkl"
SR = 24000
HOP = 120
N_FFT = 1024
