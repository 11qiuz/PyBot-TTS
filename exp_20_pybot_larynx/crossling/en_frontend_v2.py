"""Stage 10 English G2P, CMUdict Loader, Function-Word Weak-Form & Prosodic-Feature Frontend (`en_frontend_v2`).

Zero-ML deterministic English linguistic frontend:
1. Loads plain-text `exp_v10_g2p_and_align/lexicons/cmudict.dict` if present (with
   thread-safe lazy/auto-reload if downloaded after import), merged over
   `EN_V10_LEXICON`, plus verified `EN_V2_OVERRIDES` fixing all forensic G2P bugs
   (`this -> DH IH1 S`, voiced dental fricative `DH` words, tech/bilingual terms).
2. Applies an English function-word weak-form & de-stressing engine (~65 function
   words): unless phrase-final or explicitly focused (`*word*` / `**word**`),
   demotes primary stress `1 -> 0` and substitutes natural weak-form reductions.
3. Exports `plan_en_v2(text_or_units)` accepting either raw English text or a list
   of unit dicts (e.g., from `crossling_frontend.plan`), returning structured
   `words`, `phones`, and backward-compatible `units`.
"""
from __future__ import annotations

from pathlib import Path
import re
from threading import Lock
from typing import Any, Sequence

from exp_20_pybot_larynx import REPO_ROOT
from exp_20_pybot_larynx.crossling.en_articulation import VOWEL_IPAS, resolve_en_allophone
from exp_v5_arch import crossling_frontend as cf
from exp_v10_g2p_and_align.g2p_audit_and_fix import EN_V10_LEXICON, rule_en_g2p_v10

CMUDICT_PATH: Path = REPO_ROOT / "legacy" / "exp_v10_g2p_and_align" / "lexicons" / "cmudict.dict"
CMUDICT_ALT_PATH: Path = REPO_ROOT / "legacy" / "exp_v10_g2p_and_align" / "lexicons" / "cmudict-0.7b"

_ARPA_RE = re.compile(r"^([A-Z]+)([012])?$")
_EMPH_RE = re.compile(r"\*\*([^\*\n]+?)\*\*|\*([^\*\n]+?)\*")
_VARIANT_SUFFIX_RE = re.compile(r"\(\d+\)$")

# Verified overrides fixing all forensic G2P bugs + domain/bilingual terms.
# Merged ON TOP of CMUdict and EN_V10_LEXICON so these pronunciations are guaranteed.
EN_V2_OVERRIDES: dict[str, list[str]] = {
    # Voiced dental fricative (DH) forensic fixes (specifically `this` was TH in EN_V10_LEXICON)
    "this": ["DH", "IH1", "S"],
    "that": ["DH", "AE1", "T"],
    "the": ["DH", "AH0"],
    "these": ["DH", "IY1", "Z"],
    "those": ["DH", "OW1", "Z"],
    "they": ["DH", "EY1"],
    "them": ["DH", "EH1", "M"],
    "their": ["DH", "EH1", "R"],
    "theirs": ["DH", "EH1", "R", "Z"],
    "there": ["DH", "EH1", "R"],
    "then": ["DH", "EH1", "N"],
    "than": ["DH", "AE1", "N"],
    "with": ["W", "IH1", "DH"],
    "within": ["W", "IH0", "DH", "IH1", "N"],
    "without": ["W", "IH0", "DH", "AW1", "T"],
    "smooth": ["S", "M", "UW1", "DH"],
    "smoothed": ["S", "M", "UW1", "DH", "D"],
    "smoothly": ["S", "M", "UW1", "DH", "L", "IY0"],
    "breathe": ["B", "R", "IY1", "DH"],
    "breathes": ["B", "R", "IY1", "DH", "Z"],
    "breathing": ["B", "R", "IY1", "DH", "IH0", "NG"],
    "other": ["AH1", "DH", "ER0"],
    "others": ["AH1", "DH", "ER0", "Z"],
    "mother": ["M", "AH1", "DH", "ER0"],
    "father": ["F", "AA1", "DH", "ER0"],
    "brother": ["B", "R", "AH1", "DH", "ER0"],
    "though": ["DH", "OW1"],
    "although": ["AO2", "L", "DH", "OW1"],
    "thus": ["DH", "AH1", "S"],
    "weather": ["W", "EH1", "DH", "ER0"],
    "whether": ["W", "EH1", "DH", "ER0"],
    "either": ["IY1", "DH", "ER0"],
    "neither": ["N", "IY1", "DH", "ER0"],
    "gather": ["G", "AE1", "DH", "ER0"],
    "rather": ["R", "AE1", "DH", "ER0"],
    # Tech, speech & bilingual domain terms
    "pybot": ["P", "AY1", "B", "AA2", "T"],
    "pybottts": ["P", "AY1", "B", "AA2", "T", "T", "IY1", "T", "IY1", "EH1", "S"],
    "pytorch": ["P", "AY1", "T", "AO2", "R", "CH"],
    "studio": ["S", "T", "UW1", "D", "IY0", "OW2"],
    "benchmark": ["B", "EH1", "N", "CH", "M", "AA2", "R", "K"],
    "benchmarks": ["B", "EH1", "N", "CH", "M", "AA2", "R", "K", "S"],
    "lingual": ["L", "IH1", "NG", "G", "W", "AH0", "L"],
    "bilingual": ["B", "AY0", "L", "IH1", "NG", "G", "W", "AH0", "L"],
    "crossling": ["K", "R", "AO1", "S", "L", "IH2", "NG"],
    "multilingual": ["M", "AH2", "L", "T", "IY0", "L", "IH1", "NG", "G", "W", "AH0", "L"],
    "synthesis": ["S", "IH1", "N", "TH", "AH0", "S", "AH0", "S"],
    "synthesize": ["S", "IH1", "N", "TH", "AH0", "S", "AY2", "Z"],
    "synthesizer": ["S", "IH1", "N", "TH", "AH0", "S", "AY2", "Z", "ER0"],
    "frontend": ["F", "R", "AH1", "N", "T", "EH2", "N", "D"],
    "prosody": ["P", "R", "AA1", "S", "AH0", "D", "IY0"],
    "prosodic": ["P", "R", "AH0", "S", "AA1", "D", "IH0", "K"],
    "larynx": ["L", "EH1", "R", "IH0", "NG", "K", "S"],
    "allophone": ["AE1", "L", "AH0", "F", "OW2", "N"],
    "allophones": ["AE1", "L", "AH0", "F", "OW2", "N", "Z"],
    "phoneme": ["F", "OW1", "N", "IY2", "M"],
    "phonemes": ["F", "OW1", "N", "IY2", "M", "Z"],
    "formant": ["F", "AO1", "R", "M", "AH0", "N", "T"],
    "formants": ["F", "AO1", "R", "M", "AH0", "N", "T", "S"],
    "cepstral": ["K", "EH1", "P", "S", "T", "R", "AH0", "L"],
    "vocoder": ["V", "OW0", "K", "OW1", "D", "ER0"],
    "webui": ["W", "EH1", "B", "Y", "UW1", "AY1"],
    "timbre": ["T", "AE1", "M", "B", "ER0"],
    "qwen": ["K", "W", "EH1", "N"],
    "deepseek": ["D", "IY1", "P", "S", "IY1", "K"],
    "openai": ["OW1", "P", "AH0", "N", "EY1", "AY1"],
    "chatgpt": ["CH", "AE1", "T", "JH", "IY1", "P", "IY1", "T", "IY1"],
    "nvidia": ["EH0", "N", "V", "IH1", "D", "IY0", "AH0"],
}

# ~65 English function words subject to weak-form reduction & stress demotion (1 -> 0)
# when non-phrase-final and non-focused.
EN_FUNCTION_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "nor", "yet", "so",
    "of", "to", "in", "on", "at", "for", "from", "with", "by", "as",
    "into", "onto", "upon", "than", "then", "about", "per", "via",
    "it", "its", "is", "are", "was", "were", "be", "been", "am",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should", "must", "may", "might",
    "that", "this",
    "i", "me", "my", "he", "him", "his", "she", "her",
    "we", "us", "our", "you", "your", "they", "them", "their", "there",
    "some", "such",
})

# Explicit natural weak-form ARPAbet reductions (used when is_func and not phrase-final and not focused).
EN_WEAK_FORMS: dict[str, list[str]] = {
    "the": ["DH", "AH0"],
    "a": ["AH0"],
    "an": ["AH0", "N"],
    "and": ["AH0", "N", "D"],
    "or": ["AO0", "R"],
    "but": ["B", "AH0", "T"],
    "nor": ["N", "AO0", "R"],
    "of": ["AH0", "V"],
    "to": ["T", "AH0"],
    "in": ["IH0", "N"],
    "on": ["AA0", "N"],
    "at": ["AH0", "T"],
    "for": ["F", "ER0"],
    "from": ["F", "R", "AH0", "M"],
    "with": ["W", "IH0", "DH"],
    "by": ["B", "AY0"],
    "as": ["AH0", "Z"],
    "into": ["IH0", "N", "T", "UW0"],
    "onto": ["AA0", "N", "T", "UW0"],
    "upon": ["AH0", "P", "AA0", "N"],
    "than": ["DH", "AH0", "N"],
    "then": ["DH", "EH0", "N"],
    "about": ["AH0", "B", "AW0", "T"],
    "it": ["IH0", "T"],
    "its": ["IH0", "T", "S"],
    "is": ["IH0", "Z"],
    "are": ["AA0", "R"],
    "was": ["W", "AH0", "Z"],
    "were": ["W", "ER0"],
    "be": ["B", "IY0"],
    "been": ["B", "IH0", "N"],
    "am": ["AH0", "M"],
    "have": ["HH", "AH0", "V"],
    "has": ["HH", "AH0", "Z"],
    "had": ["HH", "AH0", "D"],
    "do": ["D", "UW0"],
    "does": ["D", "AH0", "Z"],
    "did": ["D", "IH0", "D"],
    "can": ["K", "AH0", "N"],
    "could": ["K", "UH0", "D"],
    "will": ["W", "IH0", "L"],
    "would": ["W", "UH0", "D"],
    "shall": ["SH", "AH0", "L"],
    "should": ["SH", "UH0", "D"],
    "that": ["DH", "AH0", "T"],
    "this": ["DH", "IH0", "S"],
    "he": ["HH", "IY0"],
    "him": ["HH", "IH0", "M"],
    "his": ["HH", "IH0", "Z"],
    "she": ["SH", "IY0"],
    "her": ["HH", "ER0"],
    "we": ["W", "IY0"],
    "us": ["AH0", "S"],
    "our": ["AW0", "ER0"],
    "you": ["Y", "UW0"],
    "your": ["Y", "AO0", "R"],
    "they": ["DH", "EY0"],
    "them": ["DH", "AH0", "M"],
    "their": ["DH", "EH0", "R"],
    "there": ["DH", "EH0", "R"],
    "some": ["S", "AH0", "M"],
    "such": ["S", "AH0", "CH"],
}

# Focused strong-form overrides for words whose default dictionary form is unstressed.
_EN_STRONG_FOCUS_FORMS: dict[str, list[str]] = {
    "the": ["DH", "IY1"],
    "a": ["EY1"],
}

# Merge base fallback lexicon: cf.EN_LEXICON < EN_V10_LEXICON (minus known forensic bugs)
_BASE_FALLBACK_LEXICON: dict[str, list[str]] = {
    **cf.EN_LEXICON,
    **{k: v for k, v in EN_V10_LEXICON.items() if k not in EN_V2_OVERRIDES},
}

_CMU_LOCK = Lock()
_CMU_DICT: dict[str, list[str]] = {}
_CMU_LOADED_PATH: Path | None = None
_CMU_LOADED_MTIME: float | None = None


def _find_cmudict_path() -> Path | None:
    if CMUDICT_PATH.exists():
        return CMUDICT_PATH
    if CMUDICT_ALT_PATH.exists():
        return CMUDICT_ALT_PATH
    return None


def _parse_cmudict_file(path: Path) -> dict[str, list[str]]:
    """Parse plain-text CMUdict into `{word_lower: [ARPA, ...]}` keeping first valid reading."""
    parsed: dict[str, list[str]] = {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if not line or line.startswith(";;;") or line.startswith("#"):
            continue
        head = line.split("#", 1)[0].strip()
        if not head:
            continue
        parts = head.split()
        if len(parts) < 2:
            continue
        raw_word = parts[0]
        if _VARIANT_SUFFIX_RE.search(raw_word):
            word = _VARIANT_SUFFIX_RE.sub("", raw_word).lower()
            if word in parsed:
                continue
        else:
            word = raw_word.lower()
        phones = [p.strip().upper() for p in parts[1:] if p.strip()]
        if not phones:
            continue
        valid = True
        for p in phones:
            m = _ARPA_RE.fullmatch(p)
            if not m or m.group(1) not in cf.ARPABET_TO_IPA:
                valid = False
                break
        if valid and word not in parsed:
            parsed[word] = phones
    return parsed


def load_cmudict(*, force: bool = False) -> dict[str, list[str]]:
    """Load plain-text CMUdict if available, auto-reloading if downloaded/updated later."""
    global _CMU_DICT, _CMU_LOADED_PATH, _CMU_LOADED_MTIME
    cand = _find_cmudict_path()
    if cand is None:
        with _CMU_LOCK:
            _CMU_DICT = {}
            _CMU_LOADED_PATH = None
            _CMU_LOADED_MTIME = None
        return {}
    mtime = cand.stat().st_mtime
    with _CMU_LOCK:
        if (
            not force
            and _CMU_DICT
            and _CMU_LOADED_PATH == cand
            and _CMU_LOADED_MTIME == mtime
        ):
            return _CMU_DICT
        loaded = _parse_cmudict_file(cand)
        _CMU_DICT = loaded
        _CMU_LOADED_PATH = cand
        _CMU_LOADED_MTIME = mtime
        return _CMU_DICT


def reload_cmudict() -> int:
    """Explicitly reload CMUdict from disk and return number of entries loaded."""
    return len(load_cmudict(force=True))


def en_g2p_hook_v2(word: str) -> list[str]:
    """Lookup hook for `crossling_frontend.resolve_english_word` (V2 overrides > CMUdict > V10)."""
    key = word.replace("’", "'").lower()
    if key in EN_V2_OVERRIDES:
        return list(EN_V2_OVERRIDES[key])
    cmud = load_cmudict()
    if key in cmud:
        return list(cmud[key])
    if key in _BASE_FALLBACK_LEXICON:
        return list(_BASE_FALLBACK_LEXICON[key])
    return []


def rule_en_g2p_v2(word: str) -> list[str]:
    """Rule-based OOV fallback preserving initial voiced `DH` for th-function stems."""
    w = re.sub(r"[^a-z']", "", word.lower())
    if w in EN_V2_OVERRIDES:
        return list(EN_V2_OVERRIDES[w])
    phones = rule_en_g2p_v10(word)
    if w.startswith("th") and w in EN_FUNCTION_WORDS and phones and phones[0] == "TH":
        phones = ["DH"] + phones[1:]
    return phones


def resolve_en_word_v2(word: str) -> tuple[list[str], str]:
    """Resolve an English word/compound/version/acronym into `(arpabet_list, source)`."""
    norm = word.replace("’", "'").strip()
    if not norm:
        return [], "empty"
    if norm in cf.EN_CASE_SENSITIVE_LEXICON:
        return list(cf.EN_CASE_SENSITIVE_LEXICON[norm]), "en_case_lexicon"
    upper = norm.upper()
    if upper in cf.EN_ACRONYM_LEXICON:
        return list(cf.EN_ACRONYM_LEXICON[upper]), "en_acronym"
    lower = norm.lower()
    if lower in EN_V2_OVERRIDES:
        return list(EN_V2_OVERRIDES[lower]), "en_v2_override"
    cmud = load_cmudict()
    if lower in cmud:
        return list(cmud[lower]), "cmudict"
    if lower in _BASE_FALLBACK_LEXICON:
        return list(_BASE_FALLBACK_LEXICON[lower]), "en_v10_lexicon"

    phones, src = cf.resolve_english_word(
        norm,
        en_hook=en_g2p_hook_v2,
        en_oov_hook=rule_en_g2p_v2,
        allow_oov_fallback=True,
    )
    return list(phones), src


def _contextual_pronunciation(
    word: str, words: list[dict[str, Any]], index: int
) -> list[str] | None:
    prev = words[index - 1] if index else None
    previous = (
        prev["token"].lower().replace("’", "'")
        if prev is not None and prev["pause_after_ms"] < 45.0 else ""
    )
    has_pause_after = words[index]["pause_after_ms"] >= 45.0
    following = (
        [row["token"].lower().replace("’", "'") for row in words[index + 1:index + 3]]
        if not has_pause_after else []
    )
    nxt = following[0] if following else ""
    unit = words[index]["orig_unit"] or {}
    preceding_mandarin = str(unit.get("preceding_mandarin") or "")

    # Prevocalic `the`: /ði/ before phonetically vowel-initial non-paused next word
    if word == "the" and not has_pause_after and index + 1 < len(words):
        nxt_tok = words[index + 1]["token"]
        try:
            nxt_arpa, _ = resolve_en_word_v2(nxt_tok)
            if nxt_arpa:
                m0 = _ARPA_RE.fullmatch(nxt_arpa[0].strip().upper())
                if m0 and m0.group(1) in cf.ARPABET_VOWEL_BASE:
                    return ["DH", "IY0"]
        except Exception:
            pass
        return None

    _MODAL_OR_AUX = {
        "to", "please", "can", "could", "will", "would", "should", "must", "may", "might",
        "shall", "do", "does", "did", "don't", "doesn't", "didn't", "can't", "won't", "let's",
    }
    _SUBJ_PRON = {"i", "we", "you", "they"}
    _DET_ADJ = {"a", "an", "the", "this", "that", "my", "your", "his", "her", "our", "their", "new", "world", "track", "heavy", "pure"}

    if word == "wind":
        if previous in {"a", "the", "cold", "strong", "high", "gusty", "prevailing"} or (
            nxt in {"blew", "blows", "speed", "direction", "turbine", "chill"}
        ):
            return ["W", "IH1", "N", "D"]
        if len(following) == 2 and nxt in {"a", "the", "this", "your", "my"}:
            if following[1] in {"clock", "watch", "timer", "spring", "mechanism"}:
                return ["W", "AY1", "N", "D"]
        return None

    if word == "read":
        if previous in _MODAL_OR_AUX - {"let's"}:
            return ["R", "IY1", "D"]
        if index == 0 and preceding_mandarin.endswith(("任务是", "步骤是", "要求是", "需要", "请")):
            return ["R", "IY1", "D"]
        if index == 0 and not preceding_mandarin and words[index]["token"].istitle():
            if nxt in {"the", "a", "this", "that", "it"}:
                return ["R", "IY1", "D"]
        return None

    if word == "live":
        if previous in (_MODAL_OR_AUX | _SUBJ_PRON | {"still", "never", "always"}) or nxt in {"in", "at", "on", "with", "near", "here", "there"}:
            return ["L", "IH1", "V"]
        if previous in {"a", "the", "go", "is", "are", "was"} or nxt in {
            "stream", "streaming", "demo", "audio", "video", "show", "broadcast", "music", "session", "feed", "event", "performance", "version", "mode"
        }:
            return ["L", "AY1", "V"]
        return ["L", "AY1", "V"] if previous in _DET_ADJ else ["L", "IH1", "V"]

    if word == "lead":
        if previous in {"of", "pure", "heavy", "solid", "toxic"} or nxt in {"pipe", "pipes", "paint", "metal", "acid", "battery", "bullet", "weight"}:
            return ["L", "EH1", "D"]
        return ["L", "IY1", "D"]

    if word == "close":
        if previous in (_MODAL_OR_AUX | {"please"}) or nxt in {"the", "this", "that", "it", "them", "your", "my", "our", "all"}:
            return ["K", "L", "OW1", "Z"]
        if previous in {"very", "too", "so", "pretty", "quite", "stay", "keep", "get", "stand", "came", "is", "are", "was"} or nxt in {
            "to", "by", "enough", "friend", "friends", "look", "match", "call", "range", "up", "at"
        }:
            return ["K", "L", "OW1", "S"]
        return None

    if word == "record":
        if previous in _MODAL_OR_AUX or (index == 0 and words[index]["token"].istitle() and nxt in {"the", "a", "this", "my", "your", "our", "it"}):
            return ["R", "IH0", "K", "AO1", "R", "D"]
        if previous in (_DET_ADJ | {"in", "on", "off", "best", "full", "public", "medical", "audio"}):
            return ["R", "EH1", "K", "ER0", "D"]
        return None

    if word == "present":
        if previous in (_MODAL_OR_AUX | _SUBJ_PRON) or nxt in {"the", "this", "that", "a", "an", "my", "our", "it", "you", "results"}:
            return ["P", "R", "IH0", "Z", "EH1", "N", "T"]
        if previous in (_DET_ADJ | {"at", "in", "for", "past", "perfect", "birthday"}) or has_pause_after:
            return ["P", "R", "EH1", "Z", "AH0", "N", "T"]
        return None

    if word == "use":
        if previous in (_MODAL_OR_AUX | _SUBJ_PRON | {"always", "often", "can"}) or nxt in {"the", "this", "that", "a", "an", "it", "them", "our", "your", "my"}:
            return ["Y", "UW1", "Z"]
        if previous in (_DET_ADJ | {"in", "of", "no", "good", "great", "full", "daily", "easy"}):
            return ["Y", "UW1", "S"]
        return None

    return None


def _demote_stress_arpa(arpa: list[str]) -> list[str]:
    """Demote primary/secondary vowel stress (`1`/`2` -> `0`) while preserving phone identities."""
    out: list[str] = []
    for p in arpa:
        m = _ARPA_RE.fullmatch(p.strip().upper())
        if not m:
            out.append(p)
            continue
        base, st = m.group(1), m.group(2)
        if base in cf.ARPABET_VOWEL_BASE and st is not None:
            out.append(f"{base}0")
        else:
            out.append(p.strip().upper())
    return out


def _apply_weak_form(
    token: str,
    arpa: list[str],
    *,
    is_func: bool,
    is_phrase_final: bool,
    is_focused: bool,
    contextual_override: bool = False,
) -> list[str]:
    """Apply function-word weak-form reduction & de-stressing unless phrase-final or focused."""
    lower = token.replace("’", "'").lower()
    if is_focused and lower in _EN_STRONG_FOCUS_FORMS:
        return list(_EN_STRONG_FOCUS_FORMS[lower])
    if contextual_override or not is_func or is_focused:
        return list(arpa)
    # `the` and `a` stay weak unless explicitly focused
    if is_phrase_final and lower not in ("the", "a"):
        return list(arpa)
    if lower in EN_WEAK_FORMS:
        return list(EN_WEAK_FORMS[lower])
    return _demote_stress_arpa(arpa)


def _extract_focus_and_clean_text(raw_text: str) -> tuple[str, list[tuple[int, int, int]]]:
    """Strip `*word*` / `**word**` markers and return `(clean_text, [(start, end, level), ...])`."""
    out_parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    raw_cursor = 0
    clean_cursor = 0

    for m in _EMPH_RE.finditer(raw_text):
        if m.start() > raw_cursor:
            prefix = raw_text[raw_cursor : m.start()]
            out_parts.append(prefix)
            clean_cursor += len(prefix)
        if m.group(1) is not None:
            inner = m.group(1)
            lvl = 2
        else:
            inner = m.group(2) or ""
            lvl = 1
        out_parts.append(inner)
        if inner:
            spans.append((clean_cursor, clean_cursor + len(inner), lvl))
            clean_cursor += len(inner)
        raw_cursor = m.end()

    if raw_cursor < len(raw_text):
        out_parts.append(raw_text[raw_cursor:])

    return "".join(out_parts), spans


def _punct_pause_ms(marks: str) -> float:
    if not marks:
        return 0.0
    if any(ch in marks for ch in "?？!！"):
        return 250.0
    if any(ch in marks for ch in ".。…"):
        return 235.0
    if any(ch in marks for ch in ";；\n"):
        return 195.0
    if any(ch in marks for ch in "—-"):
        return 150.0
    if any(ch in marks for ch in ",，、:："):
        return 132.0
    return 0.0


def _arpa_to_phone_pairs(arpa: list[str]) -> list[tuple[str, str, int | None, bool]]:
    """Convert ARPAbet tokens to `(arpabet, ipa, vowel_stress_or_none, is_vowel)`."""
    pairs: list[tuple[str, str, int | None, bool]] = []
    for tok in arpa:
        norm = tok.strip().upper()
        m = _ARPA_RE.fullmatch(norm)
        if not m:
            continue
        base, st = m.group(1), m.group(2)
        if base not in cf.ARPABET_TO_IPA:
            continue
        ipa = cf.arpabet_phone_to_ipa(norm, include_stress=False)
        is_v = base in cf.ARPABET_VOWEL_BASE
        stress = int(st) if (is_v and st is not None) else (0 if is_v else None)
        pairs.append((norm, ipa, stress, is_v))
    return pairs


def _word_primary_stress(pairs: list[tuple[str, str, int | None, bool]]) -> int:
    stresses = [st for _, _, st, is_v in pairs if is_v and st is not None]
    if 1 in stresses:
        return 1
    if 2 in stresses:
        return 2
    return 0


def plan_en_v2(
    text_or_units: str | Sequence[dict[str, Any]],
    *,
    style: str = "auto",
    focus_spans: Sequence[Any] | None = None,
    is_clause_final: bool = True,
    is_question: bool | None = None,
) -> dict[str, Any]:
    """Plan English text or pre-segmented unit dicts into words, phones, and units.

    Parameters
    ----------
    text_or_units:
        Either an English text string (supporting `*word*` / `**word**` focus
        markers and punctuation pauses) OR a sequence of unit dicts (e.g., an EN
        run from `crossling_frontend.plan`).
    style:
        Preserved for caller parity (`"auto"` default).
    focus_spans:
        Optional sequence of `(start_char, end_char, level)` tuples or objects with
        `.start_char`, `.end_char`, `.level` (such as `ExpressiveSegment.focus_spans`).
    is_clause_final:
        Whether the end of this input closes a phrase/clause (default `True`).
    is_question:
        Optional explicit question flag; inferred from trailing `?`/`？` when `None`.

    Returns
    -------
    dict with keys:
        - `schema`: `"en_v2"`
        - `text`: str
        - `is_question`: bool
        - `words`: list[dict]
        - `phones`: list[dict] (each with `ipa`, `allophone`, `arpabet`, `stress`,
          `is_func`, `word_idx`, `is_word_initial`, `is_word_final`,
          `is_phrase_final`, `cluster_len`, `pause_after_ms`, plus `word`,
          `is_vowel`, `is_focused`, `word_stress`, `is_in_phrase_final_word`, `phone_idx`)
        - `units`: list[dict] (drop-in unit dicts with updated `arpabet`, `ipa_phones`,
          `allophones`, `stress_or_tone`, `is_func`, `is_focused`, `is_phrase_final`,
          `pause_after_ms`, etc.)
    """
    raw_words: list[dict[str, Any]] = []
    clean_text = ""
    inferred_q = False

    if isinstance(text_or_units, str):
        clean_text, inline_spans = _extract_focus_and_clean_text(text_or_units)
        inferred_q = clean_text.rstrip().endswith(("?", "？"))
        all_focus: list[tuple[int, int, int]] = list(inline_spans)
        if focus_spans:
            for fs in focus_spans:
                if hasattr(fs, "start_char"):
                    all_focus.append((int(fs.start_char), int(fs.end_char), int(getattr(fs, "level", 1))))
                elif isinstance(fs, (tuple, list)) and len(fs) >= 2:
                    all_focus.append((int(fs[0]), int(fs[1]), int(fs[2]) if len(fs) >= 3 else 1))

        if clean_text.strip():
            tokens = cf.tokenize(clean_text)
            for tok in tokens:
                kind = tok["kind"]
                ttext = tok["text"]
                s0, s1 = int(tok["span"][0]), int(tok["span"][1])
                if kind == "punct":
                    if raw_words and ttext not in ('"', "'", "“", "”", "‘", "’", "(", ")", "[", "]"):
                        raw_words[-1]["punct_after"] += ttext
                        p_ms = _punct_pause_ms(ttext)
                        if p_ms > raw_words[-1]["pause_after_ms"]:
                            raw_words[-1]["pause_after_ms"] = p_ms
                    continue
                if kind not in ("en", "num"):
                    continue
                f_lvl = 0
                for fs0, fs1, fl in all_focus:
                    if not (s1 <= fs0 or s0 >= fs1):
                        f_lvl = max(f_lvl, fl)
                raw_words.append({
                    "token": ttext,
                    "text_span": [s0, s1],
                    "focus_level": f_lvl,
                    "is_focused": f_lvl > 0,
                    "pause_after_ms": 0.0,
                    "punct_after": "",
                    "orig_unit": None,
                })
    else:
        units_in = list(text_or_units)
        tok_strs: list[str] = []
        for u in units_in:
            tok_txt = str(u.get("token") or u.get("hanzi") or "").strip()
            if not tok_txt:
                continue
            tok_strs.append(tok_txt)
            ts = u.get("text_span")
            f_lvl = int(u.get("focus_level", 1 if u.get("focus") or u.get("is_focused") else 0))
            if focus_spans and isinstance(ts, (list, tuple)) and len(ts) == 2:
                s0, s1 = int(ts[0]), int(ts[1])
                for fs in focus_spans:
                    if hasattr(fs, "start_char"):
                        fs0, fs1, fl = int(fs.start_char), int(fs.end_char), int(getattr(fs, "level", 1))
                    elif isinstance(fs, (tuple, list)) and len(fs) >= 2:
                        fs0, fs1, fl = int(fs[0]), int(fs[1]), int(fs[2]) if len(fs) >= 3 else 1
                    else:
                        continue
                    if not (s1 <= fs0 or s0 >= fs1):
                        f_lvl = max(f_lvl, fl)
            raw_words.append({
                "token": tok_txt,
                "text_span": list(ts) if isinstance(ts, (list, tuple)) and len(ts) == 2 else None,
                "focus_level": f_lvl,
                "is_focused": f_lvl > 0,
                "pause_after_ms": float(u.get("pause_after_ms", 0.0)),
                "punct_after": str(u.get("punct_after", "")),
                "orig_unit": u,
            })
        clean_text = " ".join(tok_strs)
        inferred_q = bool(raw_words and any(x in raw_words[-1]["punct_after"] for x in ("?", "？")))

    q_flag = inferred_q if is_question is None else bool(is_question)
    n_w = len(raw_words)
    if n_w == 0:
        return {
            "schema": "en_v2",
            "text": clean_text,
            "style": style,
            "is_question": q_flag,
            "words": [],
            "phones": [],
            "units": [],
        }

    # Segment into phrases by punctuation / pauses >= 45 ms
    phrases: list[list[int]] = []
    cur_ph: list[int] = []
    for wi, rw in enumerate(raw_words):
        cur_ph.append(wi)
        is_brk = (rw["pause_after_ms"] >= 45.0) or (wi == n_w - 1)
        if is_brk:
            phrases.append(cur_ph)
            cur_ph = []

    phrase_meta: dict[int, tuple[int, int, int, bool]] = {}
    for p_idx, ph in enumerate(phrases):
        plen = len(ph)
        for pos, wi in enumerate(ph):
            is_last_in_ph = (pos == plen - 1)
            is_pfin_word = is_last_in_ph and (
                p_idx < len(phrases) - 1 or is_clause_final or raw_words[wi]["pause_after_ms"] >= 45.0
            )
            phrase_meta[wi] = (p_idx, pos, plen, is_pfin_word)

    words_out: list[dict[str, Any]] = []
    units_out: list[dict[str, Any]] = []
    flat_phone_tuples: list[dict[str, Any]] = []

    for wi, rw in enumerate(raw_words):
        tok = rw["token"]
        lower = tok.replace("’", "'").lower()
        is_func = lower in EN_FUNCTION_WORDS
        p_idx, pos_in_ph, ph_len, is_pfin_word = phrase_meta[wi]
        is_focused = bool(rw["is_focused"])

        try:
            arpa_base, src = resolve_en_word_v2(tok)
        except Exception:
            orig_u = rw["orig_unit"]
            if orig_u and orig_u.get("arpabet"):
                arpa_base = [str(p).upper() for p in orig_u["arpabet"]]
                src = str(orig_u.get("syllable_source", "unit_arpabet"))
            else:
                raise

        contextual = _contextual_pronunciation(lower, raw_words, wi)
        if contextual is not None:
            arpa_base, src = contextual, "en_context"

        arpa_eff = _apply_weak_form(
            tok,
            arpa_base,
            is_func=is_func,
            is_phrase_final=is_pfin_word,
            is_focused=is_focused,
            contextual_override=(contextual is not None),
        )
        pairs = _arpa_to_phone_pairs(arpa_eff)
        if not pairs:
            continue
        arpa_clean = [p[0] for p in pairs]
        ipa_phones = [p[1] for p in pairs]
        w_stress = _word_primary_stress(pairs)
        stress_label = f"S{w_stress}"

        # Resolve word-internal allophones first (next_ipa across word boundary refined below)
        w_n_p = len(pairs)
        for pi, (arp_s, ipa_s, st_i, is_v) in enumerate(pairs):
            flat_phone_tuples.append({
                "arpabet": arp_s,
                "ipa": ipa_s,
                "stress": st_i,
                "is_vowel": is_v,
                "is_func": is_func,
                "is_focused": is_focused,
                "focus_level": int(rw["focus_level"]),
                "word": tok,
                "word_idx": wi,
                "word_stress": w_stress,
                "phone_in_word_idx": pi,
                "word_n_phones": w_n_p,
                "is_word_initial": pi == 0,
                "is_word_final": pi == w_n_p - 1,
                "is_in_phrase_final_word": is_pfin_word,
                "pause_after_ms": float(rw["pause_after_ms"]) if pi == w_n_p - 1 else 0.0,
            })

        w_rec = {
            "word_idx": wi,
            "token": tok,
            "lower": lower,
            "arpabet": arpa_clean,
            "ipa_phones": ipa_phones,
            "allophones": [],  # populated after cross-word allophone resolution
            "stress": w_stress,
            "is_func": is_func,
            "is_focused": is_focused,
            "focus_level": int(rw["focus_level"]),
            "is_phrase_final": is_pfin_word,
            "is_utterance_final": wi == n_w - 1 and is_clause_final,
            "phrase_idx": p_idx,
            "word_in_phrase_idx": pos_in_ph,
            "phrase_len": ph_len,
            "pause_after_ms": float(rw["pause_after_ms"]),
            "punct_after": rw["punct_after"],
            "source": src,
            "text_span": rw["text_span"],
        }
        words_out.append(w_rec)

        orig_u = dict(rw["orig_unit"]) if rw["orig_unit"] is not None else {}
        orig_u.update({
            "lang": "en",
            "hanzi": tok,
            "token": tok,
            "onset": "",
            "final": "",
            "lexical_tone": None,
            "tone": None,
            "pinyin": None,
            "arpabet": arpa_clean,
            "ipa_phones": ipa_phones,
            "ipa": " ".join(ipa_phones) + f" [{stress_label}]",
            "stress_or_tone": stress_label,
            "stress": w_stress,
            "is_func": is_func,
            "is_focused": is_focused,
            "focus": is_focused,
            "focus_level": int(rw["focus_level"]),
            "is_phrase_final": is_pfin_word,
            "pause_after_ms": float(rw["pause_after_ms"]),
            "punct_after": rw["punct_after"],
            "syllable_source": src,
        })
        if rw["text_span"] is not None:
            orig_u["text_span"] = rw["text_span"]
        if "pitch_st" not in orig_u:
            orig_u["pitch_st"] = [0.0] * 5
        if "gain" not in orig_u:
            orig_u["gain"] = 0.88 if w_stress == 1 else 0.78
        if "duration_ms" not in orig_u:
            orig_u["duration_ms"] = max(90.0, 125.0 * sum(1 for p in pairs if p[3]) + 24.0 * sum(1 for p in pairs if not p[3]))
        units_out.append(orig_u)

    # Second pass over flat phones: allophones, cluster_len, is_phrase_final
    n_p = len(flat_phone_tuples)
    cluster_lens = [1] * n_p
    ii = 0
    while ii < n_p:
        if flat_phone_tuples[ii]["ipa"] in VOWEL_IPAS:
            ii += 1
            continue
        jj = ii + 1
        while (
            jj < n_p
            and flat_phone_tuples[jj]["ipa"] not in VOWEL_IPAS
            and flat_phone_tuples[jj]["word_idx"] == flat_phone_tuples[jj - 1]["word_idx"]
            and flat_phone_tuples[jj - 1]["pause_after_ms"] < 20.0
        ):
            jj += 1
        cl = jj - ii
        for kk in range(ii, jj):
            cluster_lens[kk] = cl
        ii = jj

    phones_out: list[dict[str, Any]] = []
    word_allos: dict[int, list[str]] = {w["word_idx"]: [] for w in words_out}

    for pi, ph in enumerate(flat_phone_tuples):
        nxt = flat_phone_tuples[pi + 1] if pi + 1 < n_p else None
        has_pause_after = ph["pause_after_ms"] >= 20.0
        same_word_next = bool(
            nxt is not None
            and not has_pause_after
            and nxt["word_idx"] == ph["word_idx"]
        )
        next_ipa = nxt["ipa"] if (nxt is not None and not has_pause_after) else None
        allo = resolve_en_allophone(ph["ipa"], next_ipa, same_word_next)
        is_pfin_phone = (
            (pi == n_p - 1 and is_clause_final)
            or (ph["pause_after_ms"] >= 45.0)
        )
        rec = {
            "phone_idx": pi,
            "ipa": ph["ipa"],
            "allophone": allo,
            "arpabet": ph["arpabet"],
            "stress": ph["stress"],
            "is_func": ph["is_func"],
            "word_idx": ph["word_idx"],
            "is_word_initial": ph["is_word_initial"],
            "is_word_final": ph["is_word_final"],
            "is_phrase_final": is_pfin_phone,
            "is_utterance_final": pi == n_p - 1 and is_clause_final,
            "cluster_len": cluster_lens[pi],
            "pause_after_ms": ph["pause_after_ms"],
            # Extra rich fields for Stage 10 prosody/acoustics
            "word": ph["word"],
            "is_vowel": ph["is_vowel"],
            "is_focused": ph["is_focused"],
            "focus_level": ph["focus_level"],
            "word_stress": ph["word_stress"],
            "phone_in_word_idx": ph["phone_in_word_idx"],
            "word_n_phones": ph["word_n_phones"],
            "is_in_phrase_final_word": ph["is_in_phrase_final_word"],
        }
        phones_out.append(rec)
        word_allos[ph["word_idx"]].append(allo)

    for w_rec, u_rec in zip(words_out, units_out):
        allos = word_allos.get(w_rec["word_idx"], [])
        w_rec["allophones"] = allos
        u_rec["allophones"] = list(allos)

    return {
        "schema": "en_v2",
        "text": clean_text,
        "style": style,
        "is_question": q_flag,
        "words": words_out,
        "phones": phones_out,
        "units": units_out,
    }


__all__ = [
    "CMUDICT_PATH",
    "EN_FUNCTION_WORDS",
    "EN_V2_OVERRIDES",
    "EN_WEAK_FORMS",
    "en_g2p_hook_v2",
    "load_cmudict",
    "plan_en_v2",
    "reload_cmudict",
    "resolve_en_word_v2",
    "rule_en_g2p_v2",
]
