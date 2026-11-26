"""Universal IPA articulatory inventory, utterance `PhoneEvent` schema, and frontend registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

ART_CLASS_NAMES: tuple[str, ...] = (
    "SIL",          # 0: Silence / pause / utterance edge
    "V_FRONT_HI",   # 1: Front / high vowels (i, y, ɪ, e, ɛ, eɪ, æ, ɨ, ɹ̩, ɻ̩)
    "V_BACK_LO",    # 2: Central / back / open vowels (a, ɑ, o, u, ʊ, ɯ, ʌ, ɤ, ə, ɚ, ɝ, ɔ, aɪ, aʊ, oʊ, ɔɪ)
    "GLIDE_LIQ",    # 3: Approximants & liquids (j, w, ɥ, l, ɹ)
    "NASAL",        # 4: Nasals (m, n, ŋ, ɲ, ɴ)
    "STOP_VOICED",  # 5: Voiced stops & affricates (b, d, ɡ, dʒ, dʑ)
    "STOP_VL",      # 6: Voiceless stops & aspirated/tense variants (p, pʰ, p͈, t, tʰ, t͈, k, kʰ, k͈)
    "FRIC_VOICED",  # 7: Voiced fricatives & rhotic/tap (v, ð, z, ʒ, ʑ, ɻ, ɾ)
    "FRIC_VL_AFFR", # 8: Voiceless fricatives & affricates (f, θ, s, s͈, ʃ, ɕ, x, h, ç, ɸ, ts, tsʰ, ʈʂ, ʈʂʰ, tɕ, tɕʰ, tɕ͈, tʃ)
    "RHOTIC_TAP",   # 9: Explicit tap / rhotic slot for future languages if distinct
)


@dataclass(frozen=True)
class IPAArticulatoryInfo:
    """Universal articulatory specification for any phoneme regardless of language."""

    ipa: str
    manner: str          # vowel, diphthong, glide, liquid, nasal, stop, affricate, fricative, tap
    place: str           # front, central, back, bilabial, labiodental, dental, alveolar, retroflex, palatal, velar, glottal, uvular
    voiced: bool
    art_class: int       # index in ART_CLASS_NAMES
    nasal_zero_hz: float | None = None
    voiceless_obstruent: bool = False


def _v(ipa: str, place: str, art_cls: int) -> tuple[str, IPAArticulatoryInfo]:
    return ipa, IPAArticulatoryInfo(ipa=ipa, manner="vowel", place=place, voiced=True, art_class=art_cls)


def _d(ipa: str, place: str, art_cls: int) -> tuple[str, IPAArticulatoryInfo]:
    return ipa, IPAArticulatoryInfo(ipa=ipa, manner="diphthong", place=place, voiced=True, art_class=art_cls)


def _g(ipa: str, place: str, manner: str = "glide") -> tuple[str, IPAArticulatoryInfo]:
    return ipa, IPAArticulatoryInfo(ipa=ipa, manner=manner, place=place, voiced=True, art_class=3)


def _n(ipa: str, place: str, zero_hz: float) -> tuple[str, IPAArticulatoryInfo]:
    return ipa, IPAArticulatoryInfo(
        ipa=ipa, manner="nasal", place=place, voiced=True, art_class=4, nasal_zero_hz=zero_hz
    )


def _st(ipa: str, place: str, voiced: bool) -> tuple[str, IPAArticulatoryInfo]:
    return ipa, IPAArticulatoryInfo(
        ipa=ipa,
        manner="stop",
        place=place,
        voiced=voiced,
        art_class=5 if voiced else 6,
        voiceless_obstruent=not voiced,
    )


def _fr(ipa: str, place: str, voiced: bool, manner: str = "fricative") -> tuple[str, IPAArticulatoryInfo]:
    return ipa, IPAArticulatoryInfo(
        ipa=ipa,
        manner=manner,
        place=place,
        voiced=voiced,
        art_class=7 if voiced else 8,
        voiceless_obstruent=not voiced,
    )


def _af(ipa: str, place: str, voiced: bool) -> tuple[str, IPAArticulatoryInfo]:
    return ipa, IPAArticulatoryInfo(
        ipa=ipa,
        manner="affricate",
        place=place,
        voiced=voiced,
        art_class=5 if voiced else 8,
        voiceless_obstruent=not voiced,
    )


IPA_ARTICULATORY_TABLE: dict[str, IPAArticulatoryInfo] = dict([
    # --- Shared + Mandarin + English Vowels & Diphthongs ---
    _v("i", "front", 1),
    _v("y", "front", 1),
    _v("ɪ", "front", 1),
    _v("e", "front", 1),
    _v("ɛ", "front", 1),
    _v("æ", "front", 1),
    _v("ɹ̩", "front", 1),
    _v("ɻ̩", "front", 1),
    _v("ɨ", "front", 1),
    _d("eɪ", "front", 1),

    _v("a", "central", 2),
    _v("ɑ", "back", 2),
    _v("o", "back", 2),
    _v("u", "back", 2),
    _v("ʊ", "back", 2),
    _v("ɤ", "back", 2),
    _v("ɔ", "back", 2),
    _v("ʌ", "back", 2),
    _v("ə", "central", 2),
    _v("ɚ", "central", 2),
    _v("ɝ", "central", 2),
    _v("ɯ", "back", 2),   # Japanese & Korean close back unrounded vowel
    _d("aɪ", "central", 2),
    _d("aʊ", "central", 2),
    _d("oʊ", "back", 2),
    _d("ɔɪ", "back", 2),

    # --- Glides & Liquids ---
    _g("j", "palatal", "glide"),
    _g("w", "bilabial", "glide"),
    _g("ɥ", "palatal", "glide"),
    _g("l", "alveolar", "liquid"),
    _g("ɹ", "retroflex", "liquid"),
    _g("ɾ", "alveolar", "tap"),  # Japanese & Korean alveolar tap

    # --- Nasals ---
    _n("m", "bilabial", 1050.0),
    _n("n", "alveolar", 1550.0),
    _n("ŋ", "velar", 2300.0),
    _n("ɲ", "palatal", 1850.0),  # Japanese / Korean palatal nasal
    _n("ɴ", "uvular", 2350.0),   # Japanese moraic nasal

    # --- Stops (ZH, EN, JA, KO lax / aspirated / tense) ---
    _st("p", "bilabial", False),
    _st("pʰ", "bilabial", False),
    _st("p͈", "bilabial", False),
    _st("b", "bilabial", True),
    _st("t", "alveolar", False),
    _st("tʰ", "alveolar", False),
    _st("t͈", "alveolar", False),
    _st("d", "alveolar", True),
    _st("k", "velar", False),
    _st("kʰ", "velar", False),
    _st("k͈", "velar", False),
    _st("ɡ", "velar", True),
    _st("g", "velar", True),

    # --- Affricates ---
    _af("ts", "dental", False),
    _af("tsʰ", "dental", False),
    _af("dz", "dental", True),
    _af("ʈʂ", "retroflex", False),
    _af("ʈʂʰ", "retroflex", False),
    _af("tɕ", "palatal", False),
    _af("tɕʰ", "palatal", False),
    _af("tɕ͈", "palatal", False),
    _af("dʑ", "palatal", True),
    _af("tʃ", "palatal", False),
    _af("dʒ", "palatal", True),

    # --- Fricatives ---
    _fr("f", "labiodental", False),
    _fr("v", "labiodental", True),
    _fr("ɸ", "bilabial", False),   # Japanese fu
    _fr("θ", "dental", False),
    _fr("ð", "dental", True),
    _fr("s", "alveolar", False),
    _fr("s͈", "alveolar", False),   # Korean tense ss
    _fr("z", "alveolar", True),
    _fr("ʂ", "retroflex", False),
    _fr("ɻ", "retroflex", True),   # Mandarin voiced retroflex continuant/fricative
    _fr("ɕ", "palatal", False),
    _fr("ʑ", "palatal", True),
    _fr("ç", "palatal", False),    # Japanese hi
    _fr("ʃ", "palatal", False),
    _fr("ʒ", "palatal", True),
    _fr("x", "velar", False),
    _fr("h", "glottal", False),
])

UNIVERSAL_VOICELESS_OBSTRUENTS: frozenset[str] = frozenset(
    k for k, v in IPA_ARTICULATORY_TABLE.items() if v.voiceless_obstruent
)
UNIVERSAL_VOWEL_IPAS: frozenset[str] = frozenset(
    k for k, v in IPA_ARTICULATORY_TABLE.items() if v.manner in ("vowel", "diphthong")
)
UNIVERSAL_NASAL_IPAS: frozenset[str] = frozenset(
    k for k, v in IPA_ARTICULATORY_TABLE.items() if v.manner == "nasal"
)


def strip_allophone_suffix(phone: str | None) -> str:
    """Strip positional tags such as `_coda` or `lang:` prefix to get base IPA."""
    s = str(phone or "").strip()
    if ":" in s:
        s = s.split(":", 1)[1]
    if s.endswith("_coda"):
        s = s[:-5]
    return s


def lookup_ipa_info(ipa_or_phone: str | None) -> IPAArticulatoryInfo:
    """Lookup universal articulatory spec for an IPA or positional allophone."""
    base = strip_allophone_suffix(ipa_or_phone)
    spec = IPA_ARTICULATORY_TABLE.get(base)
    if spec is not None:
        return spec
    return IPAArticulatoryInfo(ipa=base or "ə", manner="vowel", place="central", voiced=True, art_class=2)


def classify_universal_art_class(ipa_or_phone: str | None) -> int:
    """Return universal articulatory class index `0..9` (`0` = silence)."""
    s = str(ipa_or_phone or "").strip().lower()
    if not s or s in ("sil", "sp", "pau"):
        return 0
    return lookup_ipa_info(ipa_or_phone).art_class


def normalize_prosody_tag(lang: str, prosody_tag: int | None, ipa: str) -> int:
    """Normalize language-specific prosody tag into unified discrete bucket `0..5`.

    - Consonants -> `0` (shared articulatory consonant prosody slot unless toned/coda)
    - Mandarin (`zh`) vowels -> tones `1..5`
    - English (`en`) / Japanese (`ja`) / Korean (`ko`) vowels -> `0, 1, 2`
    """
    info = lookup_ipa_info(ipa)
    if info.manner not in ("vowel", "diphthong", "nasal"):
        return 0
    if info.manner == "nasal" and lang != "zh":
        return 0
    val = int(prosody_tag) if prosody_tag is not None else 0
    if lang == "zh":
        return max(1, min(5, val if 1 <= val <= 5 else 1))
    return max(0, min(2, val))


@dataclass
class PhoneEvent:
    """Unified phoneme-level timeline event shared across all languages (`zh`, `en`, `ja`, `ko`, ...)."""

    ipa: str
    allophone: str
    lang: str
    phone_key: str        # f"{lang}:{allophone}"
    prosody_tag: int      # Normalized discrete tag `0..5`
    start_fr: int
    end_fr: int
    word_id: str = ""
    word_pos: int = 0     # 0=single, 1=initial, 2=medial, 3=final
    is_vowel: bool = False
    is_coda: bool = False
    is_func_word: bool = False
    is_phrase_final: bool = False
    is_pause: bool = False
    pause_prev_ms: float = 0.0
    pause_after_ms: float = 0.0
    same_word_prev: bool = False
    same_word_next: bool = False
    gain_db: float = 0.0


class LanguageFrontend(Protocol):
    """Pluggable frontend contract for adding a language (e.g. `ja`, `ko`) without changing core synthesis."""

    lang: str

    def plan_run(
        self,
        text_or_units: Any,
        *,
        base_f0_hz: float,
        speed: float,
        pause_scale: float,
        seed: int,
        clause_pos: tuple[float, float] = (0.0, 1.0),
        is_clause_final: bool = True,
        start_fr: int = 0,
    ) -> tuple[list[PhoneEvent], np.ndarray, np.ndarray]:
        ...


_FRONTEND_REGISTRY: dict[str, LanguageFrontend] = {}


def register_language_frontend(lang: str, frontend: LanguageFrontend) -> None:
    """Register a pluggable language frontend (`zh`, `en`, `ja`, `ko`, ...)."""
    _FRONTEND_REGISTRY[str(lang).strip().lower()] = frontend


def get_language_frontend(lang: str) -> LanguageFrontend | None:
    """Fetch a registered pluggable language frontend by ISO code."""
    return _FRONTEND_REGISTRY.get(str(lang).strip().lower())


def list_registered_languages() -> list[str]:
    """Return sorted ISO codes of registered language frontends."""
    return sorted(_FRONTEND_REGISTRY.keys())
