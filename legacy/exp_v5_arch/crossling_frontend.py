"""Cross-lingual (ZH <-> EN) tokenizer, G2P, IPA mapper, and IR planner.

Designed for PyBotTTS Phase-2 Milestone 1 inside ``exp_v5_arch/``.
Pure stdlib by default (reuses ``pybot_tts_v2_recorded.engine`` and
``auto_rules`` for Mandarin sandhi/clause prosody without modifying them) and
provides clean pluggable hooks for external G2P libraries (``pypinyin``,
``g2p_en``) once installed in the Phase-2 ML environment.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
import sys
from typing import Callable, Iterable
from unicodedata import decimal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pybot_tts_v2_recorded import auto_rules, engine, synth


# ---------------------------------------------------------------------------
# 1. Deterministic Pinyin -> IPA inventory (49 synth.FINALS + 21 onsets)
# ---------------------------------------------------------------------------
PINYIN_ONSET_TO_IPA: dict[str, tuple[str, ...]] = {
    "": (),
    "b": ("p",),
    "p": ("pʰ",),
    "m": ("m",),
    "f": ("f",),
    "d": ("t",),
    "t": ("tʰ",),
    "n": ("n",),
    "l": ("l",),
    "g": ("k",),
    "k": ("kʰ",),
    "h": ("x",),
    "j": ("tɕ",),
    "q": ("tɕʰ",),
    "x": ("ɕ",),
    "zh": ("ʈʂ",),
    "ch": ("ʈʂʰ",),
    "sh": ("ʂ",),
    "r": ("ɻ",),
    "z": ("ts",),
    "c": ("tsʰ",),
    "s": ("s",),
}

# Covers every key in pybot_tts_v2_recorded.synth.FINALS (49 keys) plus 'ue'.
PINYIN_FINAL_TO_IPA: dict[str, tuple[str, ...]] = {
    "a": ("a",),
    "o": ("o",),
    "e": ("ɤ",),
    "i": ("i",),
    "u": ("u",),
    "v": ("y",),
    "ü": ("y",),
    "er": ("ɚ",),
    "ir": ("ɻ̩",),
    "ih": ("ɻ̩",),
    "i_zh": ("ɻ̩",),
    "iz": ("ɹ̩",),
    "ii": ("ɹ̩",),
    "i_z": ("ɹ̩",),
    "ai": ("a", "i"),
    "ei": ("e", "i"),
    "ao": ("ɑ", "u"),
    "ou": ("o", "u"),
    "an": ("a", "n"),
    "en": ("ə", "n"),
    "ang": ("ɑ", "ŋ"),
    "eng": ("ə", "ŋ"),
    "ong": ("ʊ", "ŋ"),
    "ia": ("j", "a"),
    "ie": ("j", "ɛ"),
    "iao": ("j", "ɑ", "u"),
    "iu": ("j", "o", "u"),
    "iou": ("j", "o", "u"),
    "ian": ("j", "ɛ", "n"),
    "in": ("i", "n"),
    "iang": ("j", "ɑ", "ŋ"),
    "ing": ("i", "ŋ"),
    "iong": ("j", "ʊ", "ŋ"),
    "ua": ("w", "a"),
    "uo": ("w", "o"),
    "uai": ("w", "a", "i"),
    "ui": ("w", "e", "i"),
    "uei": ("w", "e", "i"),
    "uan": ("w", "a", "n"),
    "un": ("w", "ə", "n"),
    "uen": ("w", "ə", "n"),
    "uang": ("w", "ɑ", "ŋ"),
    "ueng": ("w", "ə", "ŋ"),
    "ve": ("ɥ", "ɛ"),
    "üe": ("ɥ", "ɛ"),
    "ue": ("ɥ", "ɛ"),
    "van": ("ɥ", "ɛ", "n"),
    "üan": ("ɥ", "ɛ", "n"),
    "vn": ("y", "n"),
    "ün": ("y", "n"),
}


# ---------------------------------------------------------------------------
# 2. Deterministic ARPAbet -> IPA inventory (39 core + common variants)
# ---------------------------------------------------------------------------
ARPABET_VOWEL_BASE: dict[str, str] = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "IH": "ɪ",
    "IY": "i",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "u",
    "AX": "ə",
    "IX": "ɨ",
    "UX": "ʉ",
}

ARPABET_CONSONANT_TO_IPA: dict[str, str] = {
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "DX": "ɾ",
    "EL": "l̩",
    "EM": "m̩",
    "EN": "n̩",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "V": "v",
    "W": "w",
    "WH": "ʍ",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

ARPABET_TO_IPA: dict[str, str] = {**ARPABET_VOWEL_BASE, **ARPABET_CONSONANT_TO_IPA}
_ARPA_TOKEN_RE = re.compile(r"^([A-Z]+)([012])?$")


def arpabet_phone_to_ipa(phone: str, *, include_stress: bool = False) -> str:
    """Convert one ARPAbet symbol (e.g. ``'AE1'``, ``'AH0'``, ``'TH'``) to IPA."""
    m = _ARPA_TOKEN_RE.fullmatch(phone.strip().upper())
    if not m:
        raise ValueError(f"Invalid ARPAbet phone: {phone!r}")
    base, stress = m.group(1), m.group(2)
    if base not in ARPABET_TO_IPA:
        raise ValueError(f"Unsupported ARPAbet phone: {phone!r}")
    if base == "AH" and stress == "0":
        ipa = "ə"
    elif base == "ER" and stress == "0":
        ipa = "ɚ"
    else:
        ipa = ARPABET_TO_IPA[base]
    if include_stress and stress is not None and base in ARPABET_VOWEL_BASE:
        return f"{ipa}{stress}"
    return ipa


def arpabet_to_ipa(
    phones: Iterable[str], *, include_stress: bool = False
) -> tuple[list[str], int | None]:
    """Convert an ARPAbet sequence to ``(ipa_phones, primary_stress)``."""
    ipa_list: list[str] = []
    stresses: list[int] = []
    for p in phones:
        m = _ARPA_TOKEN_RE.fullmatch(p.strip().upper())
        if not m:
            raise ValueError(f"Invalid ARPAbet phone: {p!r}")
        if m.group(2) is not None and m.group(1) in ARPABET_VOWEL_BASE:
            stresses.append(int(m.group(2)))
        ipa_list.append(arpabet_phone_to_ipa(p, include_stress=include_stress))
    primary = 1 if 1 in stresses else (2 if 2 in stresses else (0 if 0 in stresses else None))
    return ipa_list, primary


def pinyin_to_ipa(pinyin: str) -> tuple[str, str, int, list[str]]:
    """Map marked-suffix Pinyin (e.g. ``'ni3'``, ``'chi1'``) to ``(onset, final, tone, ipa_phones)``."""
    onset, final, tone = engine.split_pinyin(pinyin)
    if onset not in PINYIN_ONSET_TO_IPA:
        raise ValueError(f"Unmapped Pinyin onset {onset!r} from {pinyin!r}")
    if final not in PINYIN_FINAL_TO_IPA:
        raise ValueError(f"Unmapped Pinyin final {final!r} from {pinyin!r}")
    ipa = list(PINYIN_ONSET_TO_IPA[onset]) + list(PINYIN_FINAL_TO_IPA[final])
    return onset, final, tone, ipa


# ---------------------------------------------------------------------------
# 3. Built-in ZH + EN Golden Lexicon (reuses engine.WORDS / engine.CHARS)
# ---------------------------------------------------------------------------
ZH_EXT_WORDS: dict[str, str] = {
    "狮子": "shi1 zi5",
    "四处": "si4 chu4",
    "二月": "er4 yue4",
    "二十": "er4 shi2",
    "二十四": "er4 shi2 si4",
    "西安": "xi1 an1",
    "早饭": "zao3 fan4",
    "开心": "kai1 xin1",
    "大家": "da4 jia1",
    "为什么": "wei4 shen2 me5",
    "居然": "ju1 ran2",
    "晴天": "qing2 tian1",
    "而是": "er2 shi4",
    "下雨": "xia4 yu3",
    "下雨天": "xia4 yu3 tian1",
    "目标": "mu4 biao1",
    "实现": "shi2 xian4",
    "这段": "zhe4 duan4",
    "语音": "yu3 yin1",
    "合成": "he2 cheng2",
    "准确": "zhun3 que4",
    "控制": "kong4 zhi4",
    "银行卡": "yin2 hang2 ka3",
    "测试": "ce4 shi4",
    "模型": "mo2 xing2",
    "跑了": "pao3 le5",
    "好听": "hao3 ting1",
    "需要": "xu1 yao4",
}

ZH_EXT_CHARS: dict[str, str] = dict(
    x.split(":")
    for x in (
        "她:ta1 和:he2 十:shi2 月:yue4 狮:shi1 处:chu4 晴:qing2 雨:yu3"
        " 下:xia4 目:mu4 标:biao1 实:shi2 段:duan4 放:fang4 成:cheng2 准:zhun3"
        " 控:kong4 制:zhi4 卡:ka3 测:ce4 模:mo2 型:xing2 跑:pao3"
    ).split()
)

# Combined Mandarin lookup: engine.WORDS / CHARS take precedence where defined.
ZH_WORDS: dict[str, str] = {**ZH_EXT_WORDS, **engine.WORDS}
ZH_CHARS: dict[str, str] = {**ZH_EXT_CHARS, **engine.CHARS}
ZH_MAX_CHUNK = max(map(len, ZH_WORDS))

ZH_DIGITS: dict[str, str] = {
    "0": "ling2", "1": "yi1", "2": "er4", "3": "san1", "4": "si4",
    "5": "wu3", "6": "liu4", "7": "qi1", "8": "ba1", "9": "jiu3",
}

EN_DIGIT_ARPA: dict[str, list[str]] = {
    "0": ["Z", "IH1", "R", "OW0"],
    "1": ["W", "AH1", "N"],
    "2": ["T", "UW1"],
    "3": ["TH", "R", "IY1"],
    "4": ["F", "AO1", "R"],
    "5": ["F", "AY1", "V"],
    "6": ["S", "IH1", "K", "S"],
    "7": ["S", "EH1", "V", "AH0", "N"],
    "8": ["EY1", "T"],
    "9": ["N", "AY1", "N"],
}

EN_LETTER_ARPA: dict[str, list[str]] = {
    "A": ["EY1"], "B": ["B", "IY1"], "C": ["S", "IY1"], "D": ["D", "IY1"],
    "E": ["IY1"], "F": ["EH1", "F"], "G": ["JH", "IY1"], "H": ["EY1", "CH"],
    "I": ["AY1"], "J": ["JH", "EY1"], "K": ["K", "EY1"], "L": ["EH1", "L"],
    "M": ["EH1", "M"], "N": ["EH1", "N"], "O": ["OW1"], "P": ["P", "IY1"],
    "Q": ["K", "Y", "UW1"], "R": ["AA1", "R"], "S": ["EH1", "S"], "T": ["T", "IY1"],
    "U": ["Y", "UW1"], "V": ["V", "IY1"], "W": ["D", "AH1", "B", "AH0", "L", "Y", "UW0"],
    "X": ["EH1", "K", "S"], "Y": ["W", "AY1"], "Z": ["Z", "IY1"],
}

# Case-sensitive overrides for lexical stress minimal pairs & mixed-case brand/acronyms.
EN_CASE_SENSITIVE_LEXICON: dict[str, list[str]] = {
    "REcord": ["R", "EH1", "K", "ER0", "D"],
    "reCORD": ["R", "IH0", "K", "AO1", "R", "D"],
    "PREsent": ["P", "R", "EH1", "Z", "AH0", "N", "T"],
    "preSENT": ["P", "R", "IH0", "Z", "EH1", "N", "T"],
    "CONduct": ["K", "AA1", "N", "D", "AH0", "K", "T"],
    "conDUCT": ["K", "AH0", "N", "D", "AH1", "K", "T"],
    "PyBotTTS": ["P", "AY1", "B", "AA2", "T", "T", "IY1", "T", "IY1", "EH1", "S"],
    "PyTorch": ["P", "AY1", "T", "AO2", "R", "CH"],
}

EN_ACRONYM_LEXICON: dict[str, list[str]] = {
    "AI": ["EY1", "AY1"],
    "TTS": ["T", "IY1", "T", "IY1", "EH1", "S"],
    "VC": ["V", "IY1", "S", "IY1"],
    "GPU": ["JH", "IY1", "P", "IY1", "Y", "UW1"],
    "CPU": ["S", "IY1", "P", "IY1", "Y", "UW1"],
    "RTX": ["AA1", "R", "T", "IY1", "EH1", "K", "S"],
    "G2P": ["JH", "IY1", "T", "UW1", "P", "IY1"],
    "IPA": ["AY1", "P", "IY1", "EY1"],
    "CUDA": ["K", "UW1", "D", "AH0"],
}

# Built-in golden CMUdict-compatible English lexicon covering benchmark & domain terms.
EN_LEXICON: dict[str, list[str]] = {
    "a": ["AH0"],
    "across": ["AH0", "K", "R", "AO1", "S"],
    "and": ["AE1", "N", "D"],
    "audio": ["AO1", "D", "IY0", "OW2"],
    "azure": ["AE1", "ZH", "ER0"],
    "balance": ["B", "AE1", "L", "AH0", "N", "S"],
    "baseline": ["B", "EY1", "S", "L", "AY2", "N"],
    "before": ["B", "IH0", "F", "AO1", "R"],
    "bring": ["B", "R", "IH1", "NG"],
    "brown": ["B", "R", "AW1", "N"],
    "by": ["B", "AY1"],
    "carefully": ["K", "EH1", "R", "F", "AH0", "L", "IY0"],
    "charger": ["CH", "AA1", "R", "JH", "ER0"],
    "check": ["CH", "EH1", "K"],
    "clean": ["K", "L", "IY1", "N"],
    "clear": ["K", "L", "IH1", "R"],
    "cloning": ["K", "L", "OW1", "N", "IH0", "NG"],
    "could": ["K", "UH1", "D"],
    "cross": ["K", "R", "AO1", "S"],
    "do": ["D", "UW1"],
    "dog": ["D", "AO1", "G"],
    "forget": ["F", "ER0", "G", "EH1", "T"],
    "fox": ["F", "AA1", "K", "S"],
    "garage": ["G", "ER0", "AA1", "ZH"],
    "headphones": ["HH", "EH1", "D", "F", "OW2", "N", "Z"],
    "i": ["AY1"],
    "inference": ["IH1", "N", "F", "ER0", "AH0", "N", "S"],
    "is": ["IH1", "Z"],
    "jumps": ["JH", "AH1", "M", "P", "S"],
    "lazy": ["L", "EY1", "Z", "IY0"],
    "lingual": ["L", "IH1", "NG", "G", "W", "AH0", "L"],
    "measure": ["M", "EH1", "ZH", "ER0"],
    "natural": ["N", "AE1", "CH", "ER0", "AH0", "L"],
    "near": ["N", "IH1", "R"],
    "not": ["N", "AA1", "T"],
    "over": ["OW1", "V", "ER0"],
    "please": ["P", "L", "IY1", "Z"],
    "prosody": ["P", "R", "AA1", "S", "AH0", "D", "IY0"],
    "python": ["P", "AY1", "TH", "AA2", "N"],
    "quick": ["K", "W", "IH1", "K"],
    "record": ["R", "IH0", "K", "AO1", "R", "D"],
    "reference": ["R", "EH1", "F", "ER0", "AH0", "N", "S"],
    "requires": ["R", "IH0", "K", "W", "AY1", "ER0", "Z"],
    "seashore": ["S", "IY1", "SH", "AO2", "R"],
    "seashells": ["S", "IY1", "SH", "EH2", "L", "Z"],
    "sells": ["S", "EH1", "L", "Z"],
    "she": ["SH", "IY1"],
    "shot": ["SH", "AA1", "T"],
    "street": ["S", "T", "R", "IY1", "T"],
    "strengths": ["S", "T", "R", "EH1", "NG", "K", "TH", "S"],
    "stretch": ["S", "T", "R", "EH1", "CH"],
    "strong": ["S", "T", "R", "AO1", "NG"],
    "studio": ["S", "T", "UW1", "D", "IY0", "OW2"],
    "surprisingly": ["S", "ER0", "P", "R", "AY1", "Z", "IH0", "NG", "L", "IY0"],
    "synthetic": ["S", "IH0", "N", "TH", "EH1", "T", "IH0", "K"],
    "the": ["DH", "AH0"],
    "themes": ["TH", "IY1", "M", "Z"],
    "these": ["DH", "IY1", "Z"],
    "they": ["DH", "EY1"],
    "think": ["TH", "IH1", "NG", "K"],
    "three": ["TH", "R", "IY1"],
    "through": ["TH", "R", "UW1"],
    "to": ["T", "UW1"],
    "today": ["T", "AH0", "D", "EY1"],
    "treasure": ["T", "R", "EH1", "ZH", "ER0"],
    "tuesday": ["T", "UW1", "Z", "D", "EY2"],
    "voice": ["V", "OY1", "S"],
    "weather": ["W", "EH1", "DH", "ER0"],
    "what": ["W", "AH1", "T"],
    "whether": ["W", "EH1", "DH", "ER0"],
    "will": ["W", "IH1", "L"],
    "world": ["W", "ER1", "L", "D"],
    "you": ["Y", "UW1"],
    "your": ["Y", "AO1", "R"],
    "zero": ["Z", "IH1", "R", "OW0"],
}


# ---------------------------------------------------------------------------
# 4. Pluggable External G2P Hook Factories (pypinyin / g2p_en)
# ---------------------------------------------------------------------------
ZhG2PHook = Callable[[str], list[str]]
ZhPhraseHook = Callable[[str], list[str] | None]
EnG2PHook = Callable[[str], list[str]]


def make_pypinyin_hook() -> ZhG2PHook | None:
    """Return a ``pypinyin``-backed ZH G2P hook if ``pypinyin`` is installed, else ``None``."""
    try:
        from pypinyin import Style, pinyin as _pypinyin  # type: ignore
    except ImportError:
        return None

    def hook(text: str) -> list[str]:
        rows = _pypinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
        return [r[0] for r in rows]

    return hook


def make_g2p_en_hook() -> EnG2PHook | None:
    """Return a ``g2p_en``-backed EN G2P hook if ``g2p_en`` is installed, else ``None``."""
    try:
        from g2p_en import G2p  # type: ignore
    except ImportError:
        return None
    model = G2p()

    def hook(word: str) -> list[str]:
        return [p for p in model(word) if p.strip() and p not in (" ", "'", "-")]

    return hook


# ---------------------------------------------------------------------------
# 5. Mixed-Language Tokenizer (zh / en / num / punct)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(
    r"(?P<zh>[\u3007\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\U00030000-\U000323af]+)"
    r"|(?P<en>[A-Za-z][A-Za-z0-9]*(?:['’-][A-Za-z0-9]+)*(?:(?<=[0-9])\.[0-9]+)*)"
    r"|(?P<num>\d+(?:[.．]\d+)?)"
    r"|(?P<space>\s+)"
    r"|(?P<punct>.)"
)


def tokenize(text: str) -> list[dict]:
    """Segment mixed ZH/EN text into ``zh``, ``en``, ``num``, and ``punct`` tokens.

    Preserves exact Unicode ``span=[start, end]`` offsets and whitespace breaks.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    tokens: list[dict] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind == "space":
            if tokens:
                tokens[-1]["space_after"] = True
            continue
        assert kind is not None
        tokens.append(
            {
                "kind": kind,
                "text": m.group(0),
                "span": [m.start(), m.end()],
                "space_after": False,
            }
        )
    return tokens


# ---------------------------------------------------------------------------
# 6. Mandarin Segmenter & Sandhi (bridges engine.py + auto_rules.py)
# ---------------------------------------------------------------------------
def _segment_zh_run(
    run_text: str,
    base_offset: int,
    *,
    zh_hook: ZhG2PHook | None = None,
    zh_phrase_hook: ZhPhraseHook | None = None,
    zh_priority_phrases: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Segment a contiguous Hanzi run into word groups with Pinyin readings."""
    groups: list[dict] = []
    i = 0
    n_run = len(run_text)
    max_chunk = max(ZH_MAX_CHUNK, 6) if zh_phrase_hook is not None else ZH_MAX_CHUNK
    protected = []
    if zh_priority_phrases:
        width = max(map(len, zh_priority_phrases))
        start = 0
        while start < n_run:
            for size in range(min(width, n_run - start), 1, -1):
                phrase = run_text[start : start + size]
                if phrase in zh_priority_phrases:
                    protected.append((start, phrase, zh_priority_phrases[phrase]))
                    start += size
                    break
            else:
                start += 1

    next_protected = iter(protected)
    boundary, protected_word, protected_readings = next(next_protected, (n_run, "", None))
    while i < n_run:
        word = run_text[i]
        phrase_sounds = None
        if i == boundary:
            word, phrase_sounds = protected_word, protected_readings
            boundary, protected_word, protected_readings = next(next_protected, (n_run, "", None))
        else:
            for size in range(min(max_chunk, boundary - i), 1, -1):
                phrase = run_text[i : i + size]
                phrase_sounds = zh_phrase_hook(phrase) if zh_phrase_hook is not None else None
                if phrase_sounds is not None or phrase in ZH_WORDS:
                    word = phrase
                    break
        if phrase_sounds is not None:
            sounds = phrase_sounds
            source = "v10_polyphone"
        elif word in engine.WORDS:
            sounds = engine.WORDS[word].split()
            source = "engine_words"
        elif word in ZH_WORDS:
            sounds = ZH_WORDS[word].split()
            source = "zh_ext"
        elif run_text[i] in engine.CHARS:
            word = run_text[i]
            sounds = [engine.CHARS[word]]
            source = "engine_chars"
        elif run_text[i] in ZH_CHARS:
            word = run_text[i]
            sounds = [ZH_CHARS[word]]
            source = "zh_ext"
        elif zh_hook is not None:
            word = run_text[i]
            sounds = list(zh_hook(word))
            source = "zh_hook"
        else:
            raise ValueError(
                f"Unsupported Mandarin character {run_text[i]!r} at offset {base_offset + i}; "
                "add to ZH_WORDS/ZH_CHARS or supply zh_g2p_hook."
            )
        if len(sounds) != len(word):
            raise ValueError(f"Mandarin lexicon syllable mismatch for {word!r}: {sounds!r}")
        units = []
        for j, (ch, py) in enumerate(zip(word, sounds)):
            onset, final, tone, ipa = pinyin_to_ipa(py)
            units.append(
                {
                    "hanzi": ch,
                    "token": ch,
                    "onset": onset,
                    "final": final,
                    "lexical_tone": tone,
                    "tone": tone,
                    "pinyin": f"{py[:-1]}{tone}",
                    "ipa_phones": ipa,
                    "text_span": [base_offset + i + j, base_offset + i + j + 1],
                    "syllable_source": source,
                }
            )
        groups.append({"word": word, "units": units, "boundary": False})
        i += len(word)
    return groups


def _apply_zh_sandhi(groups: list[dict]) -> None:
    """Apply Mandarin tone sandhi matching ``engine._sandhi`` across word groups."""
    for gi, g in enumerate(groups):
        us = g["units"]
        next_first = (
            groups[gi + 1]["units"][0]
            if gi + 1 < len(groups) and not g["boundary"]
            else None
        )
        for k, u in enumerate(us):
            nxt = us[k + 1] if k + 1 < len(us) else next_first
            if nxt is None:
                continue
            t = nxt["lexical_tone"]
            c = u["hanzi"]
            if c == "不" and u["lexical_tone"] == t == 4:
                u["tone"] = 2
            elif c == "一" and u["lexical_tone"] == 1 and not (
                k > 0 and us[k - 1]["hanzi"] == "第"
            ):
                if t in (1, 2, 3, 4):
                    u["tone"] = 2 if t == 4 else 4
        for k in range(len(us) - 1):
            if us[k]["tone"] == us[k + 1]["tone"] == 3:
                us[k]["tone"] = 2
        for u in us:
            u["pinyin"] = f"{u['pinyin'][:-1]}{u['tone']}"


# ---------------------------------------------------------------------------
# 7. English G2P (Lexicon + Acronym/CamelCase + Hook + Optional Rule Fallback)
# ---------------------------------------------------------------------------
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")
_VERSION_RE = re.compile(r"([A-Za-z]+)([0-9]+(?:\.[0-9]+)*)")
_NUMBER_RE = re.compile(r"\d+(?:[.．]\d+)*")
_POINT = ["P", "OY1", "N", "T"]


def _en_number_arpabet(text: str) -> list[str]:
    return [phone for ch in text for phone in (_POINT if ch in ".．" else EN_DIGIT_ARPA[str(decimal(ch))])]


def _spell_alnum_arpa(token: str) -> list[str]:
    phones: list[str] = []
    for ch in token.upper():
        if ch in EN_LETTER_ARPA:
            phones.extend(EN_LETTER_ARPA[ch])
        elif ch.isdecimal() or ch in ".．":
            phones.extend(_en_number_arpabet(ch))
    return phones


def _rule_oov_en_arpa(word: str) -> list[str]:
    """Conservative letter-to-ARPAbet fallback for unseen English words when enabled."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return []
    digraphs = {
        "tion": ["SH", "AH0", "N"], "sh": ["SH"], "ch": ["CH"], "th": ["TH"],
        "ph": ["F"], "ng": ["NG"], "qu": ["K", "W"], "ee": ["IY1"], "oo": ["UW1"],
        "ai": ["EY1"], "ay": ["EY1"], "ou": ["AW1"], "ow": ["OW1"], "ar": ["AA1", "R"],
        "er": ["ER0"], "or": ["AO1", "R"],
    }
    letters = {
        "a": ["AE1"], "b": ["B"], "c": ["K"], "d": ["D"], "e": ["EH1"], "f": ["F"],
        "g": ["G"], "h": ["HH"], "i": ["IH1"], "j": ["JH"], "k": ["K"], "l": ["L"],
        "m": ["M"], "n": ["N"], "o": ["AA1"], "p": ["P"], "q": ["K"], "r": ["R"],
        "s": ["S"], "t": ["T"], "u": ["AH1"], "v": ["V"], "w": ["W"], "x": ["K", "S"],
        "y": ["Y"], "z": ["Z"],
    }
    out: list[str] = []
    i = 0
    while i < len(w):
        matched = False
        for size in (4, 2):
            sub = w[i : i + size]
            if sub in digraphs:
                out.extend(digraphs[sub])
                i += size
                matched = True
                break
        if not matched:
            if w[i] == "e" and i == len(w) - 1 and len(w) > 2:
                i += 1
            else:
                out.extend(letters[w[i]])
                i += 1
    return out


def resolve_english_word(
    word: str,
    *,
    en_hook: EnG2PHook | None = None,
    en_oov_hook: EnG2PHook | None = None,
    allow_oov_fallback: bool = False,
) -> tuple[list[str], str]:
    """Resolve an English word/acronym/hyphenated compound to ``(arpabet, source)``."""
    word = word.replace("’", "'")
    if word in EN_CASE_SENSITIVE_LEXICON:
        return list(EN_CASE_SENSITIVE_LEXICON[word]), "en_lexicon"
    upper = word.upper()
    if upper in EN_ACRONYM_LEXICON:
        return list(EN_ACRONYM_LEXICON[upper]), "en_acronym"
    lower = word.lower()
    if lower in EN_LEXICON:
        return list(EN_LEXICON[lower]), "en_lexicon"
    if _NUMBER_RE.fullmatch(word):
        return _en_number_arpabet(word), "num_verbalized"

    # Hyphenated compound (e.g., "cross-lingual", "Zero-shot")
    if "-" in word:
        combined: list[str] = []
        sources: list[str] = []
        for part in word.split("-"):
            if not part:
                continue
            arpa, src = resolve_english_word(
                part, en_hook=en_hook, en_oov_hook=en_oov_hook, allow_oov_fallback=allow_oov_fallback
            )
            combined.extend(arpa)
            sources.append(src)
        if combined:
            return combined, sources[0] if len(set(sources)) == 1 else "en_compound"

    # Keep numeric suffixes out of the word pronunciation lookup.
    version = _VERSION_RE.fullmatch(word)
    if version:
        stem, number = version.groups()
        if len(stem) == 1:
            stem_phones = _spell_alnum_arpa(stem)
        else:
            try:
                stem_phones, _ = resolve_english_word(
                    stem, en_hook=en_hook, en_oov_hook=en_oov_hook, allow_oov_fallback=allow_oov_fallback,
                )
            except ValueError:
                stem_phones = _spell_alnum_arpa(stem)
        return stem_phones + _en_number_arpabet(number), "en_compound"

    if any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
        return _spell_alnum_arpa(word), "en_acronym_spelled"

    if en_hook is not None:
        hook_phones = [p.strip().upper() for p in en_hook(word) if p.strip()]
        if hook_phones:
            return hook_phones, "en_hook"

    if word.isupper() and 2 <= len(word) <= 6:
        return _spell_alnum_arpa(word), "en_acronym_spelled"

    # CamelCase can combine dictionary words and acronyms (e.g. "OpenAI").
    parts = _CAMEL_RE.findall(word)
    if len(parts) > 1 and "".join(parts) == word:
        try:
            combined = [phone for part in parts for phone in resolve_english_word(part, en_hook=en_hook)[0]]
        except ValueError:
            pass
        else:
            return combined, "en_camelcase"

    if allow_oov_fallback:
        fallback = en_oov_hook if en_oov_hook is not None else _rule_oov_en_arpa
        phones = fallback(word)
        if phones:
            return phones, "en_rule_oov"
    raise ValueError(
        f"Unsupported English word {word!r}; add to EN_LEXICON, pass en_g2p_hook, "
        "or enable allow_oov_fallback=True."
    )


def _en_prosody(arpa: list[str], clause_type: str, is_tail: bool, fraction: float) -> tuple[int, list[float], float]:
    """Compute heuristic duration (ms), 5-point pitch_st, and gain for an English token."""
    vowels = [p for p in arpa if _ARPA_TOKEN_RE.fullmatch(p) and _ARPA_TOKEN_RE.fullmatch(p).group(1) in ARPABET_VOWEL_BASE]
    n_syl = max(1, len(vowels))
    n_cons = max(0, len(arpa) - len(vowels))
    _, primary = arpabet_to_ipa(arpa)
    duration = 135 * n_syl + 24 * n_cons + (35 if primary == 1 else 0)
    if is_tail:
        duration *= 1.15
    duration = max(90.0, min(1450.0, duration))

    accent = 1.4 if primary == 1 else (0.6 if primary == 2 else -0.4)
    base_shift = 0.4 - 0.8 * fraction
    if is_tail and "question" in clause_type:
        tail_curve = [0.0, 0.3, 0.8, 1.6, 2.5]
    elif is_tail and clause_type == "exclamation":
        tail_curve = [0.8, 1.1, 0.7, 0.1, -0.5]
    elif is_tail:
        tail_curve = [0.2, 0.0, -0.4, -1.0, -1.5]
    else:
        tail_curve = [0.0, 0.35, 0.5, 0.25, 0.0]
    pitch_st = [round(max(-18.0, min(18.0, base_shift + accent * (1.0 - abs(j - 2) * 0.35) + tail_curve[j])), 2) for j in range(5)]
    gain = 0.88 if primary == 1 else 0.80
    return round(duration), pitch_st, round(gain, 3)


# ---------------------------------------------------------------------------
# 8. Unified Cross-Lingual Plan Builder & Validator
# ---------------------------------------------------------------------------
def _clause_kind_en(clause_text: str, marks: str) -> tuple[str, int]:
    lower = clause_text.strip().lower()
    if any(x in marks for x in "?？"):
        if lower.startswith(("who ", "what ", "where ", "when ", "why ", "how ", "which ")):
            return "wh_question", 250
        return "polar_question", 250
    if any(x in marks for x in "!！"):
        return "exclamation", 250
    if any(x in marks for x in "，,、：:"):
        return "continuation", 132
    if any(x in marks for x in "。.…"):
        return "statement", 235
    if any(x in marks for x in ";；\n"):
        return "statement", 195
    return "statement", 0


def plan(
    text: str,
    style: str = "auto",
    *,
    zh_g2p_hook: ZhG2PHook | None = None,
    zh_phrase_hook: ZhPhraseHook | None = None,
    zh_priority_phrases: dict[str, list[str]] | None = None,
    en_g2p_hook: EnG2PHook | None = None,
    en_oov_hook: EnG2PHook | None = None,
    allow_oov_fallback: bool = False,
) -> dict:
    """Plan mixed ZH/EN speech into an extended ``units`` IR with IPA and tone/stress."""
    if style not in engine.MODES:
        raise ValueError(f"style must be one of {engine.MODES}")
    tokens = tokenize(text)
    if not any(t["kind"] in ("zh", "en", "num") for t in tokens):
        raise ValueError("text must contain at least one speakable ZH/EN/numeric token")

    # Resolve tokens into raw proto-units + punctuation pause bindings.
    proto_items: list[dict] = []
    quote_inner = {
        idx
        for m in auto_rules.QUOTE.finditer(text)
        for idx in range(m.start() + 1, m.end() - 1)
    }

    for tidx, tok in enumerate(tokens):
        kind = tok["kind"]
        ttext = tok["text"]
        s, e = tok["span"]
        if kind == "punct":
            if ttext in auto_rules.QUOTE_MARKS:
                continue
            if ttext in engine.PAUSE and proto_items:
                prev = proto_items[-1]
                prev["pause"] = max(prev["pause"], engine.PAUSE[ttext])
                prev["boundary"] = True
                if s not in quote_inner:
                    prev["terminal"] |= ttext in engine.TERMINAL
                    prev["pmarks"] += ttext
                    prev["punct_end"] = e
            continue

        if kind == "zh":
            groups = _segment_zh_run(
                ttext, s, zh_hook=zh_g2p_hook, zh_phrase_hook=zh_phrase_hook,
                zh_priority_phrases=zh_priority_phrases,
            )
            # Mark last group boundary if token ends with whitespace before EN/num
            if tok["space_after"] and groups:
                groups[-1]["boundary"] = True
            for g in groups:
                proto_items.append(
                    {
                        "lang": "zh",
                        "group": g,
                        "pause": 0,
                        "boundary": g["boundary"],
                        "terminal": False,
                        "pmarks": "",
                        "punct_end": g["units"][-1]["text_span"][1],
                    }
                )
        elif kind == "en":
            arpa, src = resolve_english_word(
                ttext, en_hook=en_g2p_hook, en_oov_hook=en_oov_hook, allow_oov_fallback=allow_oov_fallback
            )
            ipa_phones, primary = arpabet_to_ipa(arpa, include_stress=False)
            stress_label = f"S{primary}" if primary is not None else "S0"
            proto_items.append(
                {
                    "lang": "en",
                    "unit": {
                        "lang": "en",
                        "hanzi": ttext,
                        "token": ttext,
                        "onset": "",
                        "final": "",
                        "lexical_tone": None,
                        "tone": None,
                        "pinyin": None,
                        "arpabet": arpa,
                        "ipa_phones": ipa_phones,
                        "ipa": " ".join(ipa_phones) + (f" [{stress_label}]" if primary is not None else ""),
                        "stress_or_tone": stress_label,
                        "text_span": [s, e],
                        "syllable_source": src,
                    },
                    "pause": 0,
                    "boundary": tok["space_after"],
                    "terminal": False,
                    "pmarks": "",
                    "punct_end": e,
                }
            )
        elif kind == "num":
            # Contextual language selection: ZH if adjacent non-punct token is ZH, else EN.
            prev_lang = next((p["lang"] for p in reversed(proto_items)), None)
            next_kind = next(
                (tokens[k]["kind"] for k in range(tidx + 1, len(tokens)) if tokens[k]["kind"] in ("zh", "en")),
                None,
            )
            num_lang = "zh" if (prev_lang == "zh" and next_kind != "en") or (prev_lang is None and next_kind == "zh") else "en"
            if num_lang == "zh":
                units = []
                for dch in ttext:
                    py = "dian3" if dch in ".．" else ZH_DIGITS[str(decimal(dch))]
                    onset, final, tone, ipa = pinyin_to_ipa(py)
                    units.append(
                        {
                            "hanzi": dch,
                            "token": dch,
                            "onset": onset,
                            "final": final,
                            "lexical_tone": tone,
                            "tone": tone,
                            "pinyin": py,
                            "ipa_phones": ipa,
                            "text_span": [s, e],
                            "syllable_source": "num_verbalized",
                        }
                    )
                proto_items.append(
                    {
                        "lang": "zh",
                        "group": {"word": ttext, "units": units, "boundary": tok["space_after"]},
                        "pause": 0,
                        "boundary": tok["space_after"],
                        "terminal": False,
                        "pmarks": "",
                        "punct_end": e,
                    }
                )
            else:
                arpa = _en_number_arpabet(ttext)
                ipa_phones, primary = arpabet_to_ipa(arpa)
                stress_label = f"S{primary}" if primary is not None else "S0"
                proto_items.append(
                    {
                        "lang": "en",
                        "unit": {
                            "lang": "en",
                            "hanzi": ttext,
                            "token": ttext,
                            "onset": "",
                            "final": "",
                            "lexical_tone": None,
                            "tone": None,
                            "pinyin": None,
                            "arpabet": arpa,
                            "ipa_phones": ipa_phones,
                            "ipa": " ".join(ipa_phones) + f" [{stress_label}]",
                            "stress_or_tone": stress_label,
                            "text_span": [s, e],
                            "syllable_source": "num_verbalized",
                        },
                        "pause": 0,
                        "boundary": tok["space_after"],
                        "terminal": False,
                        "pmarks": "",
                        "punct_end": e,
                    }
                )

    # Apply Mandarin sandhi on contiguous ZH group runs.
    zh_run: list[dict] = []
    for item in proto_items:
        if item["lang"] == "zh":
            item["group"]["boundary"] = item["boundary"]
            zh_run.append(item["group"])
            if item["boundary"]:
                _apply_zh_sandhi(zh_run)
                zh_run = []
        else:
            if zh_run:
                _apply_zh_sandhi(zh_run)
                zh_run = []
    if zh_run:
        _apply_zh_sandhi(zh_run)

    # Flatten into unit records with group metadata.
    flat: list[tuple[dict, bool, dict]] = []
    for item in proto_items:
        if item["lang"] == "zh":
            us = item["group"]["units"]
            for k, u in enumerate(us):
                t = u["tone"]
                u["lang"] = "zh"
                u["arpabet"] = None
                u["stress_or_tone"] = f"T{t}"
                u["ipa"] = " ".join(u["ipa_phones"]) + f" T{t}"
                flat.append((u, k == len(us) - 1, item))
        else:
            flat.append((item["unit"], True, item))

    # Clause segmentation & prosody assignment.
    ends = sorted(
        set([i for i, (_, last, meta) in enumerate(flat) if last and meta["terminal"]] + [len(flat) - 1])
    )
    units_out: list[dict] = []
    clauses: list[dict] = []
    clause_start = 0
    text_start = 0
    for end in ends:
        meta_end = flat[end][2]
        text_end = meta_end["punct_end"] if end < len(flat) - 1 else len(text)
        chars = text[text_start:text_end]
        marks = meta_end["pmarks"]
        clause_langs = {flat[k][0]["lang"] for k in range(clause_start, end + 1)}
        focus = None

        if "zh" in clause_langs:
            info = auto_rules.analyze_clause(chars, marks) if style == "auto" else None
            if info:
                focus = [text_start + x for x in info["focus_span"]] if info["focus_span"] is not None else None
                if info["break_at"] is not None:
                    info["break_at"] += text_start
                info["focus_span"] = focus
                clauses.append(
                    dict(
                        info,
                        lang="zh" if clause_langs == {"zh"} else "zh_en",
                        text_span=[text_start, text_end],
                        text=chars,
                        focus_text=text[focus[0] : focus[1]] if focus else None,
                    )
                )
        else:
            focus = None
            ckind, cpause = _clause_kind_en(chars, marks)
            info = {
                "clause_type": ckind,
                "focus_span": None,
                "focus_reason": None,
                "boundary_pitch": 0.0,
                "pause_ms": cpause,
                "break_at": None,
                "base_pitch": 0.0,
                "final_duration": 1.12,
                "gain": 1.0,
            }
            clauses.append(
                dict(
                    info,
                    lang="en",
                    text_span=[text_start, text_end],
                    text=chars,
                    focus_text=None,
                )
            )

        for i in range(clause_start, end + 1):
            u, last, meta = flat[i]
            fraction = (i - clause_start) / max(1, end - clause_start)
            tail = i == end
            pause = meta["pause"] if last else 0
            focus_on = bool(focus and focus[0] <= u["text_span"][0] < focus[1])

            if u["lang"] == "zh":
                t = u["tone"]
                duration = 185 if t == 5 else 270
                duration += 18 if u["onset"] in {"p", "t", "k", "q", "ch", "c", "sh", "x", "s"} else 0
                duration += 20 if len(u["final"]) > 2 else 0
                duration += 20 if t == 3 else 0
                if style == "auto":
                    assert info is not None
                    shift = info["base_pitch"] - 0.32 * fraction + (1.02 if focus_on else 0.0)
                    slope = info["boundary_pitch"] if tail else (0.32 if pause else 0.0)
                    spread = 1.05 if focus_on else 1.0
                    gain = (0.83 if t != 5 else 0.69) * info["gain"] * (1.11 if focus_on else 1.0)
                    duration *= 1.20 if focus_on else 1.0
                    if tail:
                        duration *= info["final_duration"]
                        pause = info["pause_ms"] if marks else pause
                    elif pause and last:
                        if not meta["terminal"] and any(c in meta["pmarks"] for c in "，,、：:"):
                            pause = 132
                        duration *= 1.07
                    if info["break_at"] is not None and u["text_span"][1] == info["break_at"] and not pause:
                        pause = 88
                        duration *= 1.05
                elif style == "confirmation":
                    shift = 0.6 - 0.9 * fraction
                    slope = -1.0 if tail else 0.0
                    spread = 1.0
                    gain = (0.83 if t != 5 else 0.69) * (0.96 if tail else 1.0)
                    duration *= 1.15 if tail else 1.0
                else:
                    shift = 1.15 + 1.15 * fraction
                    slope = 4.0 if tail else 0.7
                    spread = 1.18
                    gain = (0.92 if t != 5 else 0.77) * (1.06 if tail else 1.0)
                    duration *= 1.29 if tail else 0.94

                lexical = engine.CURVE[t]
                if t == 3 and not (tail or pause):
                    lexical = [-1.4, -2.8, -4.0, -4.15, -3.8]
                tone_curve = [round(v * spread, 2) for v in lexical]
                style_curve = [round(shift + slope * (j / 4) ** 2, 2) for j in range(5)]
                u.update(
                    duration_ms=round(duration),
                    gain=round(gain, 3),
                    pitch_st=[round(max(-18.0, min(18.0, a + b)), 2) for a, b in zip(tone_curve, style_curve)],
                    pitch_lexical_st=tone_curve,
                    pitch_prosody_st=style_curve,
                    pause_after_ms=pause,
                    focus=focus_on,
                )
            else:
                ckind = info["clause_type"] if info else "statement"
                dur_ms, pitch_st, gain = _en_prosody(u["arpabet"], ckind, tail, fraction)
                if tail and info and marks:
                    pause = info["pause_ms"]
                u.update(
                    duration_ms=dur_ms,
                    gain=gain,
                    pitch_st=pitch_st,
                    pitch_lexical_st=[0.0] * 5,
                    pitch_prosody_st=pitch_st,
                    pause_after_ms=pause,
                    focus=focus_on,
                )
            units_out.append(u)
        clause_start = end + 1
        text_start = text_end

    langs = sorted({u["lang"] for u in units_out})
    return {
        "schema": 2,
        "text": text,
        "style": style,
        "languages": langs,
        "units": units_out,
        "clauses": clauses,
    }


def validate_crossling_plan(value: dict) -> dict:
    """Validate an extended cross-lingual JSON plan without altering values."""
    def bounded(num, lo, hi):
        return type(num) in (int, float) and lo <= num <= hi and math.isfinite(num)

    if not isinstance(value, dict) or not isinstance(value.get("units"), list) or not (1 <= len(value["units"]) <= 400):
        raise ValueError("plan needs a units list of 1-400 units")

    for idx, u in enumerate(value["units"], 1):
        if not isinstance(u, dict):
            raise ValueError(f"unit {idx}: expected dict")
        lang = u.get("lang")
        if lang not in ("zh", "en"):
            raise ValueError(f"unit {idx}: invalid lang {lang!r}")
        if not isinstance(u.get("token"), str) or not u["token"]:
            raise ValueError(f"unit {idx}: missing token")
        if not isinstance(u.get("ipa_phones"), list) or not u["ipa_phones"] or not all(isinstance(p, str) and p for p in u["ipa_phones"]):
            raise ValueError(f"unit {idx}: invalid ipa_phones")
        if not isinstance(u.get("stress_or_tone"), str) or not u["stress_or_tone"]:
            raise ValueError(f"unit {idx}: missing stress_or_tone")
        if lang == "zh":
            if (u.get("onset"), u.get("final")) not in {
                (o, f) for o in PINYIN_ONSET_TO_IPA for f in PINYIN_FINAL_TO_IPA
            }:
                raise ValueError(f"unit {idx}: unsupported Mandarin onset/final")
            if type(u.get("tone")) is not int or not 1 <= u["tone"] <= 5:
                raise ValueError(f"unit {idx}: invalid Mandarin tone")
        else:
            if not isinstance(u.get("arpabet"), list) or not u["arpabet"]:
                raise ValueError(f"unit {idx}: English unit missing arpabet")
        if not bounded(u.get("duration_ms"), 60, 2000):
            raise ValueError(f"unit {idx}: invalid duration_ms")
        if not bounded(u.get("pause_after_ms"), 0, 2500):
            raise ValueError(f"unit {idx}: invalid pause_after_ms")
        if not bounded(u.get("gain"), 0, 2):
            raise ValueError(f"unit {idx}: invalid gain")
        pitch = u.get("pitch_st")
        if not isinstance(pitch, list) or len(pitch) != 5 or not all(bounded(v, -18, 18) for v in pitch):
            raise ValueError(f"unit {idx}: invalid pitch_st")
    return value
