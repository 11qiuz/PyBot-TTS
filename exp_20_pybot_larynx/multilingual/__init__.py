"""Stage 13 Zero-Mapping Multilingual Native IPA Acoustic & Rule Package."""
from __future__ import annotations

from exp_20_pybot_larynx.multilingual.knowledge_ipa_native import (
    MultilingualIPAKnowledge,
    SF_NATIVE_KB_PATH,
    SF_NATIVE_SUMMARY_PATH,
    build_multilingual_native_control,
    clear_sf_native_multilingual_cache,
    get_sf_native_multilingual_kb,
)
from exp_20_pybot_larynx.multilingual.native_rules import (
    IPACoarticulationAndSwitchRule,
    IPAGlottalAntiBuzzRule,
    IPANasalAntiresonanceRule,
    IPAObstruentCrispnessRule,
    apply_multilingual_native_rules,
)
from exp_20_pybot_larynx.multilingual.renderer import (
    EnglishFrontend,
    MandarinFrontend,
    build_multilingual_native_utterance,
    get_sf_coherent_phasor,
    register_default_frontends,
    synthesize_multilingual_native,
)
from exp_20_pybot_larynx.multilingual.schema import (
    ART_CLASS_NAMES,
    IPA_ARTICULATORY_TABLE,
    IPAArticulatoryInfo,
    LanguageFrontend,
    PhoneEvent,
    classify_universal_art_class,
    get_language_frontend,
    list_registered_languages,
    lookup_ipa_info,
    normalize_prosody_tag,
    register_language_frontend,
    strip_allophone_suffix,
)

__all__ = [
    "ART_CLASS_NAMES",
    "EnglishFrontend",
    "IPAArticulatoryInfo",
    "IPACoarticulationAndSwitchRule",
    "IPAGlottalAntiBuzzRule",
    "IPANasalAntiresonanceRule",
    "IPAObstruentCrispnessRule",
    "IPA_ARTICULATORY_TABLE",
    "LanguageFrontend",
    "MandarinFrontend",
    "MultilingualIPAKnowledge",
    "PhoneEvent",
    "SF_NATIVE_KB_PATH",
    "SF_NATIVE_SUMMARY_PATH",
    "apply_multilingual_native_rules",
    "build_multilingual_native_control",
    "build_multilingual_native_utterance",
    "classify_universal_art_class",
    "clear_sf_native_multilingual_cache",
    "get_language_frontend",
    "get_sf_coherent_phasor",
    "get_sf_native_multilingual_kb",
    "list_registered_languages",
    "lookup_ipa_info",
    "normalize_prosody_tag",
    "register_default_frontends",
    "register_language_frontend",
    "strip_allophone_suffix",
    "synthesize_multilingual_native",
]
