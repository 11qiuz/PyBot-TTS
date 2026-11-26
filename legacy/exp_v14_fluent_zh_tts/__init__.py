"""Phase 10 (v14): Multi-Sample Fluent Mandarin Non-Neural TTS (`SSB0502` Full Corpus)."""
from __future__ import annotations

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent
CORPUS_DIR = EXP_DIR / "corpus" / "SSB0502"
MANIFEST_V14 = EXP_DIR / "manifest_ssb0502_v14.json"
ALIGNMENTS_V14 = EXP_DIR / "alignments_v14.json"
FEATS_CACHE_V14 = EXP_DIR / "_ssb0502_feats_cache.pkl"
PROSODY_MODEL_V14 = EXP_DIR / "prosody_v14_zh.pkl"
SAMPLES_DIR = EXP_DIR / "samples"

SR = 24000
HOP = 120
N_FFT = 1024

# Exact 6 utterances from the user's v12/v13 4-way blind listening test (Q01..Q06)
USER_6_IDS: tuple[str, ...] = (
    "SSB05020298",  # Q01: 市区所有私人机动车禁行
    "SSB05020282",  # Q02: 特别是一双舒适的跑鞋
    "SSB05020319",  # Q03: 自去年七月房企再融资重启以来
    "SSB05020184",  # Q04: 话语权是政府政策给你的
    "SSB05020400",  # Q05: 但这时不能说亏损就不做苹果订单
    "SSB05020005",  # Q06: 但同时也有利好政策出台
)

CLEAN_10_IDS: tuple[str, ...] = (
    "SSB05020018",
    "SSB05020031",
    "SSB05020068",
    "SSB05020102",
    "SSB05020126",
    "SSB05020198",
    "SSB05020232",
    "SSB05020262",
    "SSB05020305",
    "SSB05020459",
)
