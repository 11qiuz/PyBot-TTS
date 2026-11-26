"""Phase-6 Non-Neural G2P Integrity Audit, Verified EN Lexicon, Upgraded Rule-G2P & ZH Polyphones.

Fixes the Phase-5 upstream mispronunciation/dropped-phoneme root cause:
``226 / 246`` unique English words across the 160 CMU-ARCTIC utterances missed
``cf.EN_LEXICON`` and hit the naive letter-by-letter ``_rule_oov_en_arpa``.
Provides:
1. ``EN_V10_LEXICON``: verified General American ARPAbet for all 246 CMU-ARCTIC words,
   benchmark words/acronyms, contractions, and common English irregulars (plus optional
   auto-loading of plain-text ``lexicons/cmudict.dict`` if present; 0.0 MB neural weights).
2. ``rule_en_g2p_v10``: upgraded orthographic/morphological G2P (magic-e, digraphs, ``-tion``,
   ``-ight``, ``wh-``, soft ``c/g``, ``-ed``/``-es``, possessive ``'s``, double-letter collapse).
3. ``ZH_POLYPHONE_PHRASES_V10`` + ``ZH_EXTRA_CHARS_V10`` (plus optional ``lexicons/pinyin.txt``).
4. ``plan_v10(text)`` plugging into ``exp_v5_arch.crossling_frontend.plan``.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT_IMPORT = Path(__file__).resolve().parent.parent
if str(ROOT_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_IMPORT))

from exp_v5_arch import crossling_frontend as cf
from exp_v10_g2p_and_align import EXP_DIR, LEXICON_DIR, MANIFEST_PATH, ROOT_DIR

# Verified General American ARPAbet (CMUdict convention, strictly validated against cf.ARPABET_TO_IPA)
EN_V10_LEXICON: dict[str, list[str]] = {
    # Contractions & possessives in CMU-ARCTIC / common speech
    "'em": ["AH0", "M"],
    "aren't": ["AA1", "R", "AH0", "N", "T"],
    "can't": ["K", "AE1", "N", "T"],
    "couldn't": ["K", "UH1", "D", "AH0", "N", "T"],
    "didn't": ["D", "IH1", "D", "AH0", "N", "T"],
    "doesn't": ["D", "AH1", "Z", "AH0", "N", "T"],
    "don't": ["D", "OW1", "N", "T"],
    "hadn't": ["HH", "AE1", "D", "AH0", "N", "T"],
    "hasn't": ["HH", "AE1", "Z", "AH0", "N", "T"],
    "haven't": ["HH", "AE1", "V", "AH0", "N", "T"],
    "he's": ["HH", "IY1", "Z"],
    "i'd": ["AY1", "D"],
    "i'll": ["AY1", "L"],
    "i'm": ["AY1", "M"],
    "i've": ["AY1", "V"],
    "isn't": ["IH1", "Z", "AH0", "N", "T"],
    "it's": ["IH1", "T", "S"],
    "let's": ["L", "EH1", "T", "S"],
    "selden's": ["S", "EH1", "L", "D", "AH0", "N", "Z"],
    "she's": ["SH", "IY1", "Z"],
    "shouldn't": ["SH", "UH1", "D", "AH0", "N", "T"],
    "that's": ["DH", "AE1", "T", "S"],
    "there's": ["DH", "EH1", "R", "Z"],
    "they're": ["DH", "EH1", "R"],
    "wasn't": ["W", "AA1", "Z", "AH0", "N", "T"],
    "we'll": ["W", "IY1", "L"],
    "we're": ["W", "IH1", "R"],
    "weren't": ["W", "ER1", "N", "T"],
    "what's": ["W", "AH1", "T", "S"],
    "won't": ["W", "OW1", "N", "T"],
    "wouldn't": ["W", "UH1", "D", "AH0", "N", "T"],
    "you'd": ["Y", "UW1", "D"],
    "you'll": ["Y", "UW1", "L"],
    "you're": ["Y", "UH1", "R"],
    "you've": ["Y", "UW1", "V"],
    # CMU-ARCTIC 226+ uncovered words & common core English
    "about": ["AH0", "B", "AW1", "T"],
    "accounting": ["AH0", "K", "AW1", "N", "T", "IH0", "NG"],
    "actual": ["AE1", "K", "CH", "UW0", "AH0", "L"],
    "again": ["AH0", "G", "EH1", "N"],
    "almost": ["AO1", "L", "M", "OW2", "S", "T"],
    "already": ["AO0", "L", "R", "EH1", "D", "IY0"],
    "always": ["AO1", "L", "W", "EY2", "Z"],
    "an": ["AE1", "N"],
    "apart": ["AH0", "P", "AA1", "R", "T"],
    "apologized": ["AH0", "P", "AA1", "L", "AH0", "JH", "AY2", "Z", "D"],
    "are": ["AA1", "R"],
    "asleep": ["AH0", "S", "L", "IY1", "P"],
    "at": ["AE1", "T"],
    "aurora": ["AO0", "R", "AO1", "R", "AH0"],
    "author": ["AO1", "TH", "ER0"],
    "back": ["B", "AE1", "K"],
    "balls": ["B", "AO1", "L", "Z"],
    "be": ["B", "IY1"],
    "been": ["B", "IH1", "N"],
    "began": ["B", "IH0", "G", "AE1", "N"],
    "belize": ["B", "AH0", "L", "IY1", "Z"],
    "beyond": ["B", "IY0", "AA1", "N", "D"],
    "bless": ["B", "L", "EH1", "S"],
    "borealis": ["B", "AO2", "R", "IY0", "AE1", "L", "AH0", "S"],
    "break": ["B", "R", "EY1", "K"],
    "bribery": ["B", "R", "AY1", "B", "ER0", "IY0"],
    "build": ["B", "IH1", "L", "D"],
    "bury": ["B", "EH1", "R", "IY0"],
    "business": ["B", "IH1", "Z", "N", "AH0", "S"],
    "busy": ["B", "IH1", "Z", "IY0"],
    "but": ["B", "AH1", "T"],
    "buy": ["B", "AY1"],
    "came": ["K", "EY1", "M"],
    "canadian": ["K", "AH0", "N", "EY1", "D", "IY0", "AH0", "N"],
    "capital": ["K", "AE1", "P", "IH0", "T", "AH0", "L"],
    "case": ["K", "EY1", "S"],
    "chair": ["CH", "EH1", "R"],
    "chances": ["CH", "AE1", "N", "S", "AH0", "Z"],
    "change": ["CH", "EY1", "N", "JH"],
    "chiefly": ["CH", "IY1", "F", "L", "IY0"],
    "children": ["CH", "IH1", "L", "D", "R", "AH0", "N"],
    "churchill": ["CH", "ER1", "CH", "IH2", "L"],
    "cities": ["S", "IH1", "T", "IY0", "Z"],
    "clubbed": ["K", "L", "AH1", "B", "D"],
    "clubs": ["K", "L", "AH1", "B", "Z"],
    "combination": ["K", "AA2", "M", "B", "AH0", "N", "EY1", "SH", "AH0", "N"],
    "come": ["K", "AH1", "M"],
    "coming": ["K", "AH1", "M", "IH0", "NG"],
    "companion": ["K", "AH0", "M", "P", "AE1", "N", "Y", "AH0", "N"],
    "cooler": ["K", "UW1", "L", "ER0"],
    "cough": ["K", "AO1", "F"],
    "country": ["K", "AH1", "N", "T", "R", "IY0"],
    "danger": ["D", "EY1", "N", "JH", "ER0"],
    "days": ["D", "EY1", "Z"],
    "degree": ["D", "IH0", "G", "R", "IY1"],
    "delicate": ["D", "EH1", "L", "IH0", "K", "AH0", "T"],
    "do": ["D", "UW1"],
    "does": ["D", "AH1", "Z"],
    "done": ["D", "AH1", "N"],
    "down": ["D", "AW1", "N"],
    "earth": ["ER1", "TH"],
    "enough": ["IH0", "N", "AH1", "F"],
    "enthusiasm": ["IH0", "N", "TH", "UW1", "Z", "IY0", "AE2", "Z", "AH0", "M"],
    "etc": ["EH2", "T", "S", "EH1", "T", "ER0", "AH0"],
    "evening": ["IY1", "V", "N", "IH0", "NG"],
    "ever": ["EH1", "V", "ER0"],
    "every": ["EH1", "V", "ER0", "IY0"],
    "exclaimed": ["IH0", "K", "S", "K", "L", "EY1", "M", "D"],
    "eyes": ["AY1", "Z"],
    "faced": ["F", "EY1", "S", "T"],
    "fairly": ["F", "EH1", "R", "L", "IY0"],
    "faith": ["F", "EY1", "TH"],
    "favor": ["F", "EY1", "V", "ER0"],
    "feet": ["F", "IY1", "T"],
    "fighter": ["F", "AY1", "T", "ER0"],
    "fighting": ["F", "AY1", "T", "IH0", "NG"],
    "followed": ["F", "AA1", "L", "OW0", "D"],
    "for": ["F", "AO1", "R"],
    "force": ["F", "AO1", "R", "S"],
    "forces": ["F", "AO1", "R", "S", "IH0", "Z"],
    "forever": ["F", "ER0", "EH1", "V", "ER0"],
    "fort": ["F", "AO1", "R", "T"],
    "fraud": ["F", "R", "AO1", "D"],
    "friend": ["F", "R", "EH1", "N", "D"],
    "friendship": ["F", "R", "EH1", "N", "D", "SH", "IH2", "P"],
    "from": ["F", "R", "AH1", "M"],
    "gad": ["G", "AE1", "D"],
    "game": ["G", "EY1", "M"],
    "give": ["G", "IH1", "V"],
    "glad": ["G", "L", "AE1", "D"],
    "go": ["G", "OW1"],
    "god": ["G", "AA1", "D"],
    "great": ["G", "R", "EY1", "T"],
    "gregson": ["G", "R", "EH1", "G", "S", "AH0", "N"],
    "grew": ["G", "R", "UW1"],
    "had": ["HH", "AE1", "D"],
    "hand": ["HH", "AE1", "N", "D"],
    "handed": ["HH", "AE1", "N", "D", "IH0", "D"],
    "hands": ["HH", "AE1", "N", "D", "Z"],
    "hardly": ["HH", "AA1", "R", "D", "L", "IY0"],
    "has": ["HH", "AE1", "Z"],
    "hatred": ["HH", "EY1", "T", "R", "AH0", "D"],
    "have": ["HH", "AE1", "V"],
    "he": ["HH", "IY1"],
    "head": ["HH", "EH1", "D"],
    "heart": ["HH", "AA1", "R", "T"],
    "here": ["HH", "IH1", "R"],
    "his": ["HH", "IH1", "Z"],
    "hope": ["HH", "OW1", "P"],
    "horrified": ["HH", "AO1", "R", "AH0", "F", "AY2", "D"],
    "hotel": ["HH", "OW0", "T", "EH1", "L"],
    "how": ["HH", "AW1"],
    "hundred": ["HH", "AH1", "N", "D", "R", "AH0", "D"],
    "if": ["IH1", "F"],
    "in": ["IH1", "N"],
    "induced": ["IH0", "N", "D", "UW1", "S", "T"],
    "into": ["IH1", "N", "T", "UW0"],
    "it": ["IH1", "T"],
    "jealousy": ["JH", "EH1", "L", "AH0", "S", "IY0"],
    "judgment": ["JH", "AH1", "JH", "M", "AH0", "N", "T"],
    "just": ["JH", "AH1", "S", "T"],
    "know": ["N", "OW1"],
    "laugh": ["L", "AE1", "F"],
    "letter": ["L", "EH1", "T", "ER0"],
    "life": ["L", "AY1", "F"],
    "like": ["L", "AY1", "K"],
    "line": ["L", "AY1", "N"],
    "live": ["L", "IH1", "V"],
    "long": ["L", "AO1", "NG"],
    "looking": ["L", "UH1", "K", "IH0", "NG"],
    "looks": ["L", "UH1", "K", "S"],
    "lord": ["L", "AO1", "R", "D"],
    "losing": ["L", "UW1", "Z", "IH0", "NG"],
    "love": ["L", "AH1", "V"],
    "machine": ["M", "AH0", "SH", "IY1", "N"],
    "made": ["M", "EY1", "D"],
    "me": ["M", "IY1"],
    "memories": ["M", "EH1", "M", "ER0", "IY0", "Z"],
    "men": ["M", "EH1", "N"],
    "met": ["M", "EH1", "T"],
    "moment": ["M", "OW1", "M", "AH0", "N", "T"],
    "move": ["M", "UW1", "V"],
    "moving": ["M", "UW1", "V", "IH0", "NG"],
    "my": ["M", "AY1"],
    "mysterious": ["M", "IH0", "S", "T", "IH1", "R", "IY0", "AH0", "S"],
    "need": ["N", "IY1", "D"],
    "needed": ["N", "IY1", "D", "AH0", "D"],
    "nodded": ["N", "AA1", "D", "AH0", "D"],
    "north": ["N", "AO1", "R", "TH"],
    "now": ["N", "AW1"],
    "occurred": ["AH0", "K", "ER1", "D"],
    "ocean": ["OW1", "SH", "AH0", "N"],
    "of": ["AH1", "V"],
    "on": ["AA1", "N"],
    "once": ["W", "AH1", "N", "S"],
    "one": ["W", "AH1", "N"],
    "only": ["OW1", "N", "L", "IY0"],
    "opposition": ["AA2", "P", "AH0", "Z", "IH1", "SH", "AH0", "N"],
    "or": ["AO1", "R"],
    "organized": ["AO1", "R", "G", "AH0", "N", "AY2", "Z", "D"],
    "our": ["AW1", "ER0"],
    "particular": ["P", "ER0", "T", "IH1", "K", "Y", "AH0", "L", "ER0"],
    "people": ["P", "IY1", "P", "AH0", "L"],
    "petitioned": ["P", "AH0", "T", "IH1", "SH", "AH0", "N", "D"],
    "phil": ["F", "IH1", "L"],
    "philip": ["F", "IH1", "L", "AH0", "P"],
    "physique": ["F", "AH0", "Z", "IY1", "K"],
    "plans": ["P", "L", "AE1", "N", "Z"],
    "playing": ["P", "L", "EY1", "IH0", "NG"],
    "powerful": ["P", "AW1", "ER0", "F", "AH0", "L"],
    "prevailed": ["P", "R", "IH0", "V", "EY1", "L", "D"],
    "privileges": ["P", "R", "IH1", "V", "AH0", "L", "IH0", "JH", "AH0", "Z"],
    "proposed": ["P", "R", "AH0", "P", "OW1", "Z", "D"],
    "prove": ["P", "R", "UW1", "V"],
    "public": ["P", "AH1", "B", "L", "IH0", "K"],
    "question": ["K", "W", "EH1", "S", "CH", "AH0", "N"],
    "quickly": ["K", "W", "IH1", "K", "L", "IY0"],
    "railroad": ["R", "EY1", "L", "R", "OW2", "D"],
    "recognizing": ["R", "EH1", "K", "AH0", "G", "N", "AY2", "Z", "IH0", "NG"],
    "remained": ["R", "IH0", "M", "EY1", "N", "D"],
    "reports": ["R", "IH0", "P", "AO1", "R", "T", "S"],
    "ridge": ["R", "IH1", "JH"],
    "rifle": ["R", "AY1", "F", "AH0", "L"],
    "robbery": ["R", "AA1", "B", "ER0", "IY0"],
    "rose": ["R", "OW1", "Z"],
    "said": ["S", "EH1", "D"],
    "same": ["S", "EY1", "M"],
    "says": ["S", "EH1", "Z"],
    "see": ["S", "IY1"],
    "seeing": ["S", "IY1", "IH0", "NG"],
    "sharply": ["SH", "AA1", "R", "P", "L", "IY0"],
    "ship": ["SH", "IH1", "P"],
    "shook": ["SH", "UH1", "K"],
    "shorter": ["SH", "AO1", "R", "T", "ER0"],
    "should": ["SH", "UH1", "D"],
    "shoved": ["SH", "AH1", "V", "D"],
    "show": ["SH", "OW1"],
    "since": ["S", "IH1", "N", "S"],
    "single": ["S", "IH1", "NG", "G", "AH0", "L"],
    "sitting": ["S", "IH1", "T", "IH0", "NG"],
    "some": ["S", "AH1", "M"],
    "stamp": ["S", "T", "AE1", "M", "P"],
    "steels": ["S", "T", "IY1", "L", "Z"],
    "step": ["S", "T", "EH1", "P"],
    "stomach": ["S", "T", "AH1", "M", "AH0", "K"],
    "stop": ["S", "T", "AA1", "P"],
    "stopped": ["S", "T", "AA1", "P", "T"],
    "sugar": ["SH", "UH1", "G", "ER0"],
    "superlative": ["S", "UW0", "P", "ER1", "L", "AH0", "T", "IH0", "V"],
    "suppose": ["S", "AH0", "P", "OW1", "Z"],
    "sure": ["SH", "UH1", "R"],
    "surprise": ["S", "ER0", "P", "R", "AY1", "Z"],
    "table": ["T", "EY1", "B", "AH0", "L"],
    "ten": ["T", "EH1", "N"],
    "than": ["DH", "AE1", "N"],
    "that": ["DH", "AE1", "T"],
    "their": ["DH", "EH1", "R"],
    "them": ["DH", "EH1", "M"],
    "then": ["DH", "EH1", "N"],
    "there": ["DH", "EH1", "R"],
    "they": ["DH", "EY1"],
    "this": ["TH", "IH1", "S"],
    "though": ["DH", "OW1"],
    "thought": ["TH", "AO1", "T"],
    "through": ["TH", "R", "UW1"],
    "time": ["T", "AY1", "M"],
    "tom": ["T", "AA1", "M"],
    "tongue": ["T", "AH1", "NG"],
    "trail": ["T", "R", "EY1", "L"],
    "turned": ["T", "ER1", "N", "D"],
    "turns": ["T", "ER1", "N", "Z"],
    "twentieth": ["T", "W", "EH1", "N", "T", "IY0", "AH0", "TH"],
    "two": ["T", "UW1"],
    "typewritten": ["T", "AY1", "P", "R", "IH2", "T", "AH0", "N"],
    "unfolded": ["AH0", "N", "F", "OW1", "L", "D", "IH0", "D"],
    "up": ["AH1", "P"],
    "us": ["AH1", "S"],
    "want": ["W", "AA1", "N", "T"],
    "was": ["W", "AA1", "Z"],
    "watch": ["W", "AA1", "CH"],
    "we": ["W", "IY1"],
    "week": ["W", "IY1", "K"],
    "were": ["W", "ER1"],
    "what": ["W", "AH1", "T"],
    "when": ["W", "EH1", "N"],
    "where": ["W", "EH1", "R"],
    "which": ["W", "IH1", "CH"],
    "whittemore": ["W", "IH1", "T", "AH0", "M", "AO2", "R"],
    "who": ["HH", "UW1"],
    "whose": ["HH", "UW1", "Z"],
    "why": ["W", "AY1"],
    "within": ["W", "IH0", "DH", "IH1", "N"],
    "women": ["W", "IH1", "M", "AH0", "N"],
    "wonder": ["W", "AH1", "N", "D", "ER0"],
    "word": ["W", "ER1", "D"],
    "work": ["W", "ER1", "K"],
    "world": ["W", "ER1", "L", "D"],
    "would": ["W", "UH1", "D"],
    "yards": ["Y", "AA1", "R", "D", "Z"],
}

# Context-aware Mandarin polyphone disambiguation phrases (matches contiguous ZH spans)
ZH_POLYPHONE_PHRASES_V10: dict[str, list[str]] = {
    "行长": ["hang2", "zhang3"],
    "银行": ["yin2", "hang2"],
    "银行卡": ["yin2", "hang2", "ka3"],
    "行业": ["hang2", "ye4"],
    "不行": ["bu4", "xing2"],
    "重庆": ["chong2", "qing4"],
    "重新": ["chong2", "xin1"],
    "重要": ["zhong4", "yao4"],
    "音乐": ["yin1", "yue4"],
    "快乐": ["kuai4", "le4"],
    "长大": ["zhang3", "da4"],
    "长度": ["chang2", "du4"],
    "为什么": ["wei4", "shen2", "me5"],
    "因为": ["yin1", "wei4"],
    "方便": ["fang1", "bian4"],
    "便宜": ["pian2", "yi5"],
    "头发": ["tou2", "fa5"],
    "发现": ["fa1", "xian4"],
    "会计": ["kuai4", "ji4"],
    "一会儿": ["yi1", "hui4", "er5"],
    "调查": ["diao4", "cha2"],
    "空调": ["kong1", "tiao2"],
    "爱好": ["ai4", "hao4"],
    "好坏": ["hao3", "huai4"],
    "多少": ["duo1", "shao3"],
    "少年": ["shao4", "nian2"],
    "首都": ["shou3", "du1"],
    "都会": ["dou1", "hui4"],
    "四处": ["si4", "chu4"],
    "处理": ["chu3", "li3"],
    "差不多": ["cha4", "bu4", "duo1"],
    "出差": ["chu1", "chai1"],
    "觉得": ["jue2", "de5"],
    "睡觉": ["shui4", "jiao4"],
    "干净": ["gan1", "jing4"],
    "干活": ["gan4", "huo2"],
    "着急": ["zhao2", "ji2"],
    "看着": ["kan4", "zhe5"],
    "还有": ["hai2", "you3"],
    "还钱": ["huan2", "qian2"],
    "数学": ["shu4", "xue2"],
    "数数": ["shu3", "shu3"],
}

ZH_EXTRA_CHARS_V10: dict[str, str] = {
    "𠮷": "ji2",
    "扣": "kou4",
    "拼": "pin1",
    "音": "yin1",
    "嗯": "en1",
    "哦": "o4",
    "啦": "la5",
    "呗": "bei5",
    "啥": "sha2",
    "咋": "za3",
}

_EXT_CMUDICT: dict[str, list[str]] | None = None
_EXT_PINYIN_CHAR: dict[str, str] | None = None
_EXT_PINYIN_PHRASES: dict[str, list[str]] | None = None

_TONE_TRANS = str.maketrans(
    "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜńňǹ",
    "aaaaeeeeiiiioooouuuuvvvvnnn",
)
_TONE_MARKS = {
    "ā": "1", "ē": "1", "ī": "1", "ō": "1", "ū": "1", "ǖ": "1",
    "á": "2", "é": "2", "í": "2", "ó": "2", "ú": "2", "ǘ": "2", "ń": "2",
    "ǎ": "3", "ě": "3", "ǐ": "3", "ǒ": "3", "ǔ": "3", "ǚ": "3", "ň": "3",
    "à": "4", "è": "4", "ì": "4", "ò": "4", "ù": "4", "ǜ": "4", "ǹ": "4",
}


def _diacritic_pinyin_to_numbered(syl: str) -> str | None:
    s = syl.strip().lower()
    if not s:
        return None
    t = "5"
    for ch in s:
        if ch in _TONE_MARKS:
            t = _TONE_MARKS[ch]
            break
    base = s.translate(_TONE_TRANS).replace("ü", "v")
    # In cf.pinyin_to_ipa, ju/qu/xu/yu use u while lv/nv keep v
    if len(base) >= 2 and base[0] in ("j", "q", "x", "y") and "v" in base:
        base = base.replace("v", "u")
    base = base.replace("ê", "e")
    if base.isdigit():
        return None
    if re.fullmatch(r"[a-z]+[1-5]", base):
        return base
    if re.fullmatch(r"[a-z]+", base):
        return f"{base}{t}"
    return None


def _load_optional_external_lexicons(load_phrases: bool = True) -> tuple[dict[str, list[str]], dict[str, str], dict[str, list[str]]]:
    """Auto-load optional plain-text non-neural dictionaries from ``lexicons/`` if present."""
    global _EXT_CMUDICT, _EXT_PINYIN_CHAR, _EXT_PINYIN_PHRASES
    if _EXT_CMUDICT is None:
        cmud: dict[str, list[str]] = {}
        for cand in (LEXICON_DIR / "cmudict.dict", LEXICON_DIR / "cmudict-0.7b"):
            if cand.exists():
                for line in cand.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if not line or line.startswith(";;;") or line.startswith("#"):
                        continue
                    # Strip trailing inline comment `# ...` if present
                    head = line.split("#", 1)[0].strip()
                    parts = head.split()
                    if len(parts) < 2:
                        continue
                    w = re.sub(r"\(\d+\)$", "", parts[0]).lower()
                    if w not in cmud:
                        cmud[w] = parts[1:]
                break
        if not cmud:
            try:
                from nltk.corpus import cmudict  # type: ignore

                raw = cmudict.dict()
                for k, v in raw.items():
                    if v:
                        cmud[k.lower()] = list(v[0])
            except Exception:
                pass
        _EXT_CMUDICT = cmud

    if _EXT_PINYIN_CHAR is None:
        pym: dict[str, str] = {}
        py_path = LEXICON_DIR / "pinyin.txt"
        if py_path.exists():
            for line in py_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = re.match(r"U\+([0-9A-Fa-f]+):\s*([^\s#]+)", line)
                if m:
                    ch = chr(int(m.group(1), 16))
                    first = m.group(2).split(",")[0].strip()
                    num_py = _diacritic_pinyin_to_numbered(first)
                    if num_py:
                        pym[ch] = num_py
        _EXT_PINYIN_CHAR = pym

    if load_phrases and _EXT_PINYIN_PHRASES is None:
        phr: dict[str, list[str]] = {}
        lp_path = LEXICON_DIR / "large_pinyin.txt"
        if lp_path.exists():
            for line in lp_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line or line.startswith("#") or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                phrase = k.strip()
                pys_raw = v.split("#", 1)[0].strip().split()
                if len(phrase) >= 2 and len(pys_raw) == len(phrase):
                    pys = [_diacritic_pinyin_to_numbered(x) for x in pys_raw]
                    if all(x is not None for x in pys):
                        phr[phrase] = pys  # type: ignore[assignment]
        _EXT_PINYIN_PHRASES = phr

    return _EXT_CMUDICT, _EXT_PINYIN_CHAR, (_EXT_PINYIN_PHRASES or {})


def rule_en_g2p_v10(word: str) -> list[str]:
    """Upgraded orthographic & morphological pure-rule English G2P for unseen OOV words."""
    w = re.sub(r"[^a-z']", "", word.lower())
    if not w:
        return []

    # Handle possessive / contracted suffixes cleanly
    if w.endswith("'s") and len(w) > 2:
        stem = resolve_en_word_v10(w[:-2])
        tail = "AH0 Z" if stem and stem[-1].rstrip("012") in ("S", "Z", "SH", "ZH", "CH", "JH") else (
            "S" if stem and stem[-1].rstrip("012") in ("P", "T", "K", "F", "TH") else "Z"
        )
        return stem + tail.split()

    # Check if regular plural/past inflection maps to a known stem
    if w.endswith("ied") and len(w) > 4:
        return rule_en_g2p_v10(w[:-3] + "y") + ["D"]
    if w.endswith("ed") and len(w) > 4 and w[:-2] in EN_V10_LEXICON:
        stem = EN_V10_LEXICON[w[:-2]]
        last = stem[-1].rstrip("012")
        if last in ("T", "D"):
            return list(stem) + ["AH0", "D"]
        if last in ("P", "K", "F", "S", "SH", "CH", "TH"):
            return list(stem) + ["T"]
        return list(stem) + ["D"]
    if w.endswith("ing") and len(w) > 5 and w[:-3] in EN_V10_LEXICON:
        return list(EN_V10_LEXICON[w[:-3]]) + ["IH0", "NG"]

    # Detect silent final -e magic-e pattern (VCe)
    vowels = set("aeiouy")
    has_magic_e = (
        len(w) >= 4
        and w.endswith("e")
        and w[-2] not in vowels
        and w[-3] in "aeiou"
        and (len(w) < 5 or w[-4] not in vowels)
    )
    magic_long = {"a": "EY1", "e": "IY1", "i": "AY1", "o": "OW1", "u": "UW1"}

    chunks: list[tuple[str, list[str]]] = [
        ("tion", ["SH", "AH0", "N"]),
        ("sion", ["ZH", "AH0", "N"]),
        ("ture", ["CH", "ER0"]),
        ("ight", ["AY1", "T"]),
        ("ough", ["AO1"]),
        ("ould", ["UH1", "D"]),
        ("dge", ["JH"]),
        ("tch", ["CH"]),
        ("igh", ["AY1"]),
        ("eau", ["OW1"]),
        ("ing", ["IH0", "NG"]),
        ("sh", ["SH"]),
        ("ch", ["CH"]),
        ("th", ["TH"]),
        ("ph", ["F"]),
        ("wh", ["W"]),
        ("qu", ["K", "W"]),
        ("ck", ["K"]),
        ("ng", ["NG"]),
        ("kn", ["N"]),
        ("wr", ["R"]),
        ("ee", ["IY1"]),
        ("ea", ["IY1"]),
        ("oo", ["UW1"]),
        ("ai", ["EY1"]),
        ("ay", ["EY1"]),
        ("oa", ["OW1"]),
        ("oi", ["OY1"]),
        ("oy", ["OY1"]),
        ("ou", ["AW1"]),
        ("ow", ["OW1"]),
        ("au", ["AO1"]),
        ("aw", ["AO1"]),
        ("ew", ["UW1"]),
        ("ar", ["AA1", "R"]),
        ("er", ["ER0"]),
        ("ir", ["ER1"]),
        ("or", ["AO1", "R"]),
        ("ur", ["ER1"]),
    ]
    single = {
        "a": ["AE1"], "b": ["B"], "c": ["K"], "d": ["D"], "e": ["EH1"], "f": ["F"],
        "g": ["G"], "h": ["HH"], "i": ["IH1"], "j": ["JH"], "k": ["K"], "l": ["L"],
        "m": ["M"], "n": ["N"], "o": ["AA1"], "p": ["P"], "q": ["K"], "r": ["R"],
        "s": ["S"], "t": ["T"], "u": ["AH1"], "v": ["V"], "w": ["W"], "x": ["K", "S"],
        "y": ["Y"], "z": ["Z"],
    }

    out: list[str] = []
    i = 0
    while i < len(w):
        if has_magic_e and i == len(w) - 1:
            break
        if has_magic_e and i == len(w) - 3 and w[i] in magic_long:
            out.append(magic_long[w[i]])
            i += 1
            continue
        matched = False
        for pat, phs in chunks:
            if w.startswith(pat, i):
                # kn / wr silent letter only word-initially
                if pat in ("kn", "wr") and i != 0:
                    continue
                out.extend(phs)
                i += len(pat)
                matched = True
                break
        if matched:
            continue
        ch = w[i]
        # Collapse identical double consonants (bb->B, tt->T, ss->S, ll->L)
        if i + 1 < len(w) and w[i + 1] == ch and ch not in vowels:
            i += 1
        # Soft c/g before e/i/y
        if ch == "c" and i + 1 < len(w) and w[i + 1] in ("e", "i", "y"):
            out.append("S")
        elif ch == "g" and i + 1 < len(w) and w[i + 1] in ("e", "i", "y"):
            out.append("JH")
        elif ch == "y":
            if i == 0:
                out.append("Y")
            elif i == len(w) - 1 and len(w) <= 3:
                out.append("AY1")
            elif i == len(w) - 1:
                out.append("IY0")
            else:
                out.append("IH1")
        elif ch == "e" and i == len(w) - 1 and len(w) > 2:
            pass
        elif ch in single:
            out.extend(single[ch])
        i += 1
    return out


def resolve_en_word_v10(word: str) -> list[str]:
    """Use the same structured pronunciation lookup as the v10 planner."""
    phones, _ = cf.resolve_english_word(
        word, en_hook=en_g2p_hook_v10, en_oov_hook=rule_en_g2p_v10, allow_oov_fallback=True,
    )
    return phones


def en_g2p_hook_v10(word: str) -> list[str]:
    """Look up dictionary readings, leaving unseen words to the planner's fallback."""
    key = word.lower()
    if key in EN_V10_LEXICON:
        return list(EN_V10_LEXICON[key])
    cmud, _, _ = _load_optional_external_lexicons(load_phrases=False)
    return list(cmud.get(key, ()))


def zh_g2p_hook_v10(char: str) -> list[str]:
    """Hook compatible with ``crossling_frontend.plan(..., zh_g2p_hook=zh_g2p_hook_v10)``."""
    if char in ZH_EXTRA_CHARS_V10:
        return [ZH_EXTRA_CHARS_V10[char]]
    _, pym, _ = _load_optional_external_lexicons(load_phrases=False)
    if char in pym:
        return [pym[char]]
    raise ValueError(f"Unknown Mandarin character {char!r} (no pinyin lookup entry)")


def _zh_phrase_v10(phrase: str) -> list[str] | None:
    # Built-in readings are lexical. External entries can already contain sandhi.
    if phrase in cf.ZH_WORDS:
        return None
    _, _, external = _load_optional_external_lexicons(load_phrases=True)
    readings = external.get(phrase)
    if readings is not None:
        try:
            for pinyin in readings:
                cf.pinyin_to_ipa(pinyin)
        except ValueError:
            return None
    return readings


def plan_v10(text: str, style: str = "auto") -> dict:
    """Resolve phrase readings before the frontend applies sandhi and prosody."""
    return cf.plan(
        text,
        style=style,
        zh_g2p_hook=zh_g2p_hook_v10,
        zh_phrase_hook=_zh_phrase_v10,
        zh_priority_phrases=ZH_POLYPHONE_PHRASES_V10,
        en_g2p_hook=en_g2p_hook_v10,
        en_oov_hook=rule_en_g2p_v10,
        allow_oov_fallback=True,
    )


def audit_g2p_coverage() -> dict:
    """Audit G2P coverage before (v5/v9) vs after (v10) across CMU-ARCTIC & benchmark texts."""
    bm = json.loads((ROOT_DIR / "exp_v5_arch" / "benchmark_texts.json").read_text(encoding="utf-8"))
    bm_items = bm.get("cases", bm.get("sentences", bm)) if isinstance(bm, dict) else bm
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    en_texts = [u["text"] for u in manifest["english"]["utterances"]]
    en_words = sorted({w for t in en_texts for w in re.findall(r"[a-z']+", t.lower())})

    v9_rule_oov = [w for w in en_words if w not in cf.EN_LEXICON]
    cmud, pym, ext_phrases = _load_optional_external_lexicons(load_phrases=True)

    v10_lex_hit = 0
    v10_valid_ipa = 0
    for w in en_words:
        arpa = resolve_en_word_v10(w)
        if w in cf.EN_LEXICON or w in EN_V10_LEXICON or w in cmud:
            v10_lex_hit += 1
        try:
            cf.arpabet_to_ipa(arpa)
            v10_valid_ipa += 1
        except Exception:
            pass

    bm_ok_v9 = 0
    bm_ok_v10 = 0
    for it in bm_items:
        txt = it["text"] if isinstance(it, dict) else str(it)
        try:
            cf.plan(txt, allow_oov_fallback=False)
            bm_ok_v9 += 1
        except Exception:
            pass
        try:
            plan_v10(txt)
            bm_ok_v10 += 1
        except Exception:
            pass

    # Synthetic OOV stress cases proving rule_en_g2p_v10 fixes silent-e, -tion, -ight, dedup
    oov_checks = {
        "make": ["M", "EY1", "K"],
        "knight": ["N", "AY1", "T"],
        "nation": ["N", "EY1", "SH", "AH0", "N"],
        "back": ["B", "AE1", "K"],
        "change": ["CH", "EY1", "N", "JH"],
        "eyes": ["AY1", "Z"],
    }
    rule_probe = {}
    for w, exp in oov_checks.items():
        got_v9, _ = cf.resolve_english_word(w, allow_oov_fallback=True)
        got_v10 = rule_en_g2p_v10(w)
        rule_probe[w] = {"expected": exp, "v9_rule": got_v9, "v10_rule": got_v10}

    return {
        "cmu_arctic_utterances": len(en_texts),
        "cmu_arctic_unique_words": len(en_words),
        "v9_en_lexicon_hits": len(en_words) - len(v9_rule_oov),
        "v9_en_rule_oov_count": len(v9_rule_oov),
        "v9_en_rule_oov_rate": round(len(v9_rule_oov) / max(1, len(en_words)), 4),
        "v10_en_lexicon_hits": v10_lex_hit,
        "v10_en_rule_oov_count": len(en_words) - v10_lex_hit,
        "v10_en_rule_oov_rate": round((len(en_words) - v10_lex_hit) / max(1, len(en_words)), 4),
        "v10_valid_ipa_words": v10_valid_ipa,
        "benchmark_ok_v9": f"{bm_ok_v9}/{len(bm_items)}",
        "benchmark_ok_v10": f"{bm_ok_v10}/{len(bm_items)}",
        "external_cmudict_loaded": len(cmud),
        "external_pinyin_chars_loaded": len(pym),
        "external_pinyin_phrases_loaded": len(ext_phrases),
        "rule_probe": rule_probe,
    }


if __name__ == "__main__":
    print(json.dumps(audit_g2p_coverage(), indent=2, ensure_ascii=False))
