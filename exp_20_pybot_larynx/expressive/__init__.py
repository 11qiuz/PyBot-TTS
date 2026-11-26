"""Stage 8 Expressive Emotion, Focus & Non-Verbal Performance Engine for PyBot-Larynx."""
from exp_20_pybot_larynx.expressive.emotion_parser import (
    EMOTION_METADATA,
    EMOTION_MODE_NAMES,
    EMOTION_PROFILES,
    INLINE_TAG_CHIPS,
    ExpressiveSegment,
    FocusSpan,
    parse_expressive_text,
    strip_performance_tags,
)
from exp_20_pybot_larynx.expressive.performer import (
    apply_emotion_to_control,
    apply_emotion_to_zh_prosody,
    render_expressive_utterance,
)
from exp_20_pybot_larynx.expressive.vocalizations import (
    build_breath_control,
    build_laugh_control,
    build_pause_control,
    build_sigh_control,
)

__all__ = [
    "EMOTION_METADATA",
    "EMOTION_MODE_NAMES",
    "EMOTION_PROFILES",
    "ExpressiveSegment",
    "FocusSpan",
    "INLINE_TAG_CHIPS",
    "apply_emotion_to_control",
    "apply_emotion_to_zh_prosody",
    "build_breath_control",
    "build_laugh_control",
    "build_pause_control",
    "build_sigh_control",
    "parse_expressive_text",
    "render_expressive_utterance",
    "strip_performance_tags",
]
