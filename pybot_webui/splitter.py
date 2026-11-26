"""Text splitting (sentence/line/none) and Chinese-context text normalization."""
from __future__ import annotations

import re
from typing import Any

_HANZI_RE = re.compile(r"[\u3400-\u9fff]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_TAG_MASK_RE = re.compile(
    r"\[(?:breath|inhale|吸气|换气|gasp|倒吸气|惊呼|sigh|叹气|叹息|laugh|笑|大笑|哈哈|chuckle|轻笑|"
    r"neutral|calm|平静|正常|自然|happy|cheerful|开心|高兴|喜悦|愉快|excited|激动|兴奋|热情|"
    r"sad|sorrow|悲伤|难过|伤心|低落|angry|furious|愤怒|生气|怒斥|gentle|soft|tender|温柔|轻柔|安慰|"
    r"surprised|shocked|惊讶|震惊|惊奇|fearful|scared|afraid|害怕|紧张|恐惧|whisper|whispers|耳语|悄悄话|气声|"
    r"serious|严肃|playful|调皮|(?:pause|停顿)\s*(?:\d+(?:\.\d+)?\s*(?:ms|s|秒)?)?)\]",
    re.IGNORECASE,
)

# Sentence terminators keeping trailing closing quotes/brackets attached.
_SENT_END_RE = re.compile(
    r"(?:[。！？!?；;]|…+|(?<!\d)\.(?!\d)(?=\s|$|[\"'\u201d\u2019\u300d\u300f\uff09\u3011\])]))"
    r"[\"'\u201d\u2019\u300d\u300f\uff09\u3011\])]*\s*"
)
_CLAUSE_SEP_RE = re.compile(
    r"([，,、：:][\"'\u201d\u2019\u300d\u300f\uff09\u3011\])]*\s*)"
)

_ZH_DIGITS = ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九")
_ZH_SMALL_UNITS = ("", "十", "百", "千")
_ZH_BIG_UNITS = ("", "万", "亿", "兆")

_CURRENCY_RE = re.compile(r"[¥￥]\s*(\d+(?:\.\d+)?)")
_PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
_YEAR_RE = re.compile(r"(?<![A-Za-z0-9.])((?:19|20)\d{2})年")
_TEMP_C_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*[℃°C]")
_STANDALONE_NUM_RE = re.compile(r"(?<![A-Za-z0-9_.])(-?\d+(?:\.\d+)?)(?![A-Za-z0-9_.%])")


def detect_lang_tag(text: str) -> str:
    """Detect whether a segment contains Mandarin ('zh'), English ('en'), or both ('zh+en')."""
    cleaned = _TAG_MASK_RE.sub(" ", str(text or "")).replace("*", "")
    has_zh = bool(_HANZI_RE.search(cleaned))
    has_en = bool(_ASCII_LETTER_RE.search(cleaned))
    if has_zh and has_en:
        return "zh+en"
    if has_en:
        return "en"
    return "zh"


def _digits_literal_to_zh(s: str) -> str:
    return "".join(_ZH_DIGITS[int(ch)] if ch.isdigit() else ch for ch in s)


def _section_to_zh(num: int) -> str:
    """Convert 0..9999 integer to Mandarin reading."""
    if num == 0:
        return "零"
    parts: list[str] = []
    zero_pending = False
    for unit_idx in range(3, -1, -1):
        base = 10 ** unit_idx
        digit = (num // base) % 10
        if digit == 0:
            if parts:
                zero_pending = True
        else:
            if zero_pending:
                parts.append("零")
                zero_pending = False
            if not (digit == 1 and unit_idx == 1 and not parts):
                parts.append(_ZH_DIGITS[digit])
            parts.append(_ZH_SMALL_UNITS[unit_idx])
    return "".join(parts)


def int_to_zh(num: int) -> str:
    """Convert an arbitrary integer to standard Mandarin reading."""
    if num < 0:
        return "负" + int_to_zh(-num)
    if num == 0:
        return "零"
    sections: list[int] = []
    n = num
    while n > 0:
        sections.append(n % 10000)
        n //= 10000

    out: list[str] = []
    need_zero = False
    for idx in range(len(sections) - 1, -1, -1):
        sec = sections[idx]
        if sec == 0:
            if out:
                need_zero = True
            continue
        if need_zero or (out and sec < 1000):
            out.append("零")
        sec_str = _section_to_zh(sec)
        # Avoid dropping leading "一" in "十一万..." when inside higher section
        if sec == 10 and out:
            sec_str = "一十"
        elif 10 < sec < 20 and out:
            sec_str = "一" + sec_str
        out.append(sec_str + _ZH_BIG_UNITS[idx])
        need_zero = False
    return "".join(out)


def number_str_to_zh(num_str: str) -> str:
    """Convert numeric string (integer or decimal, optional leading '-') to Mandarin."""
    s = num_str.strip()
    if not s:
        return s
    neg = False
    if s.startswith("-"):
        neg = True
        s = s[1:]
    if "." in s:
        int_part, frac_part = s.split(".", 1)
        int_val = int(int_part) if int_part else 0
        res = f"{int_to_zh(int_val)}点{_digits_literal_to_zh(frac_part)}"
    else:
        if len(s) > 12 or (len(s) > 1 and s.startswith("0")):
            res = _digits_literal_to_zh(s)
        else:
            res = int_to_zh(int(s))
    return ("负" + res) if neg else res


def normalize_text_for_larynx(text: str, *, allow_english: bool = True) -> str:
    """Normalize Chinese-context numbers, years, percentages, and currency into readable Mandarin.

    Pure English text (without Hanzi) is returned intact when ``allow_english=True`` so English
    acronyms, numbers, and words are handled natively by the English G2P/prosody path.
    """
    src = text.strip()
    if not src:
        return ""

    stripped = _TAG_MASK_RE.sub(" ", src).replace("*", "")
    has_zh = bool(_HANZI_RE.search(stripped))
    if not has_zh and allow_english:
        return src

    # Temporarily mask performance tags so `[pause 0.5s]` / `[停顿0.5s]` are not rewritten
    saved_tags: list[str] = []

    def _stash_tag(m: re.Match[str]) -> str:
        saved_tags.append(m.group(0))
        return f"@@PYBOTTAG{len(saved_tags) - 1}@@"

    out = _TAG_MASK_RE.sub(_stash_tag, src)
    out = _CURRENCY_RE.sub(lambda m: f"{number_str_to_zh(m.group(1))}元", out)
    out = _PERCENT_RE.sub(lambda m: f"百分之{number_str_to_zh(m.group(1))}", out)
    out = _TEMP_C_RE.sub(lambda m: f"{number_str_to_zh(m.group(1))}摄氏度", out)
    out = _YEAR_RE.sub(lambda m: f"{_digits_literal_to_zh(m.group(1))}年", out)
    out = _STANDALONE_NUM_RE.sub(lambda m: number_str_to_zh(m.group(1)), out)
    for idx_t, tag_str in enumerate(saved_tags):
        out = out.replace(f"@@PYBOTTAG{idx_t}@@", tag_str)
    return out


def _soft_split_clause(sentence: str, max_chars: int) -> list[str]:
    """Soft-split an over-long sentence along clause punctuation when max_chars > 0."""
    if max_chars <= 0 or len(sentence) <= max_chars:
        return [sentence]

    parts = _CLAUSE_SEP_RE.split(sentence)
    clauses: list[str] = []
    i = 0
    while i < len(parts):
        chunk = parts[i]
        if i + 1 < len(parts):
            chunk += parts[i + 1]
            i += 2
        else:
            i += 1
        chunk = chunk.strip()
        if chunk:
            clauses.append(chunk)

    if len(clauses) <= 1:
        return [sentence]

    merged: list[str] = []
    cur = ""
    for cl in clauses:
        if not cur:
            cur = cl
        elif len(cur) + len(cl) <= max_chars:
            joiner = " " if (_ASCII_LETTER_RE.match(cl[0]) and _ASCII_LETTER_RE.match(cur[-1])) else ""
            cur = cur + joiner + cl
        else:
            merged.append(cur)
            cur = cl
    if cur:
        merged.append(cur)
    return merged


def _split_paragraph_sentences(para: str, max_chars: int) -> list[str]:
    """Split one non-empty paragraph on sentence-ending punctuation."""
    pieces: list[str] = []
    start = 0
    for m in _SENT_END_RE.finditer(para):
        seg = para[start : m.end()].strip()
        start = m.end()
        if seg:
            pieces.extend(_soft_split_clause(seg, max_chars))
    tail = para[start:].strip()
    if tail:
        pieces.extend(_soft_split_clause(tail, max_chars))
    return pieces


def split_text(text: str, mode: str = "sentence", max_chars: int = 80) -> list[dict[str, Any]]:
    """Split input text into editable synthesis segments.

    Modes:
    - ``"sentence"``: split by paragraphs then sentence punctuation (with soft clause wrap at ``max_chars``)
    - ``"line"``: split strictly by non-empty lines
    - ``"none"``: keep as a single normalized segment
    """
    raw = str(text or "")
    if not raw.strip():
        return []

    norm_mode = mode if mode in ("sentence", "line", "none") else "sentence"
    try:
        limit = max(0, int(max_chars))
    except (TypeError, ValueError):
        limit = 80

    raw_segments: list[str] = []
    if norm_mode == "none":
        collapsed = " ".join(line.strip() for line in raw.splitlines() if line.strip())
        if collapsed:
            raw_segments.append(collapsed)
    elif norm_mode == "line":
        for line in raw.splitlines():
            s = line.strip()
            if s:
                raw_segments.append(s)
    else:
        for line in raw.splitlines():
            para = line.strip()
            if para:
                raw_segments.extend(_split_paragraph_sentences(para, limit))

    segments: list[dict[str, Any]] = []
    for idx, seg_txt in enumerate(raw_segments, start=1):
        segments.append(
            {
                "index": idx,
                "text": seg_txt,
                "char_count": len(seg_txt),
                "lang_tag": detect_lang_tag(seg_txt),
            }
        )
    return segments
