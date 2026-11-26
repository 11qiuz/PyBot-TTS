"""Phase 9 (v13): Diagnosing the 4.03 dB Concatenation Gap & Non-Neural UTAU / Virtual-Singer Ceiling."""
from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
SAMPLES_DIR = EXP_DIR / "samples"
V12_MANIFEST = REPO_ROOT / "exp_v12_data" / "corpus" / "manifest_v12.json"
EXTENDED_INDEX = REPO_ROOT / "pybot_tts_v2_recorded" / "extended" / "index.json"
GMM_ALIGNMENTS_V12 = REPO_ROOT / "exp_v12_prosody" / "gmm_alignments_v12.json"
ZH_FEATS_CACHE = EXP_DIR / "_zh_feats_cache.pkl"

SR = 24000
HOP = 120
N_FFT = 1024

SPOTLIGHT_6_IDS = (
    "SSB05020005",
    "SSB05020082",
    "SSB05020122",
    "SSB05020137",
    "SSB05020159",
    "SSB05020222",
)

# Exact 6 utterances from the user's Q01..Q06 blind four-way listening test
USER_6_IDS = (
    "SSB05020298",  # Q01: 市区所有私人机动车禁行
    "SSB05020282",  # Q02: 特别是一双舒适的跑鞋
    "SSB05020319",  # Q03: 自去年七月房企再融资重启以来
    "SSB05020184",  # Q04: 话语权是政府政策给你的
    "SSB05020400",  # Q05: 但这时不能说亏损就不做苹果订单
    "SSB05020005",  # Q06: 但同时也有利好政策出台
)
