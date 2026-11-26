"""Expressive performance markup parser, emotion lexicons, and ElevenLabs-style timeline planner."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

EMOTION_PROFILES: dict[str, dict[str, float]] = {
    "neutral": {
        "f0_shift_st": 0.0,
        "f0_range_scale": 1.0,
        "f0_declination_st": 0.0,
        "speed_scale": 1.0,
        "pause_scale": 1.0,
        "gain_offset_db": 0.0,
        "rd_offset": 0.0,
        "bap_offset": 0.0,
        "tilt_db_oct": 0.0,
        "jitter_add": 0.0,
        "shimmer_add": 0.0,
        "tremor_depth_st": 0.0,
        "tremor_rate_hz": 5.0,
        "vowel_openness": 0.0,
        "formant_q": 0.0,
        "chest_db": 0.0,
        "warmth_db": 0.0,
        "presence_db": 0.0,
        "air_db": 0.0,
        "subharm_base": 0.0,
        "rubato_strength": 0.0,
    },
    "happy": {
        "f0_shift_st": 1.6,
        "f0_range_scale": 1.28,
        "f0_declination_st": 0.4,
        "speed_scale": 1.08,
        "pause_scale": 0.92,
        "gain_offset_db": 0.6,
        "rd_offset": -0.10,
        "bap_offset": 0.02,
        "tilt_db_oct": 1.3,
        "jitter_add": 0.001,
        "shimmer_add": 0.004,
        "tremor_depth_st": 0.0,
        "tremor_rate_hz": 5.5,
        "vowel_openness": 0.34,
        "formant_q": 0.12,
        "chest_db": -0.4,
        "warmth_db": 0.4,
        "presence_db": 1.6,
        "air_db": 1.5,
        "subharm_base": 0.0,
        "rubato_strength": 0.25,
    },
    "excited": {
        "f0_shift_st": 2.6,
        "f0_range_scale": 1.42,
        "f0_declination_st": 0.8,
        "speed_scale": 1.15,
        "pause_scale": 0.82,
        "gain_offset_db": 1.2,
        "rd_offset": -0.22,
        "bap_offset": -0.05,
        "tilt_db_oct": 1.9,
        "jitter_add": 0.001,
        "shimmer_add": 0.006,
        "tremor_depth_st": 0.0,
        "tremor_rate_hz": 5.8,
        "vowel_openness": 0.28,
        "formant_q": 0.18,
        "chest_db": 0.6,
        "warmth_db": 0.0,
        "presence_db": 2.4,
        "air_db": 1.8,
        "subharm_base": 0.0,
        "rubato_strength": 0.35,
    },
    "sad": {
        "f0_shift_st": -2.1,
        "f0_range_scale": 0.66,
        "f0_declination_st": -1.7,
        "speed_scale": 0.83,
        "pause_scale": 1.28,
        "gain_offset_db": -1.2,
        "rd_offset": 0.40,
        "bap_offset": 0.16,
        "tilt_db_oct": -1.7,
        "jitter_add": 0.004,
        "shimmer_add": 0.018,
        "tremor_depth_st": 0.18,
        "tremor_rate_hz": 4.6,
        "vowel_openness": -0.24,
        "formant_q": -0.14,
        "chest_db": 1.2,
        "warmth_db": 1.8,
        "presence_db": -1.5,
        "air_db": 0.4,
        "subharm_base": 0.08,
        "rubato_strength": 0.30,
    },
    "angry": {
        "f0_shift_st": 1.0,
        "f0_range_scale": 1.34,
        "f0_declination_st": -0.7,
        "speed_scale": 1.13,
        "pause_scale": 0.80,
        "gain_offset_db": 1.6,
        "rd_offset": -0.34,
        "bap_offset": -0.06,
        "tilt_db_oct": 2.5,
        "jitter_add": 0.003,
        "shimmer_add": 0.014,
        "tremor_depth_st": 0.0,
        "tremor_rate_hz": 6.0,
        "vowel_openness": 0.14,
        "formant_q": 0.28,
        "chest_db": 2.2,
        "warmth_db": 0.4,
        "presence_db": 3.4,
        "air_db": 0.8,
        "subharm_base": 0.03,
        "rubato_strength": 0.20,
    },
    "gentle": {
        "f0_shift_st": -0.9,
        "f0_range_scale": 0.85,
        "f0_declination_st": -0.6,
        "speed_scale": 0.89,
        "pause_scale": 1.15,
        "gain_offset_db": -0.6,
        "rd_offset": 0.32,
        "bap_offset": 0.18,
        "tilt_db_oct": -1.1,
        "jitter_add": 0.001,
        "shimmer_add": 0.005,
        "tremor_depth_st": 0.0,
        "tremor_rate_hz": 4.8,
        "vowel_openness": -0.06,
        "formant_q": -0.06,
        "chest_db": 0.8,
        "warmth_db": 1.6,
        "presence_db": -0.8,
        "air_db": 1.8,
        "subharm_base": 0.04,
        "rubato_strength": 0.22,
    },
    "surprised": {
        "f0_shift_st": 2.9,
        "f0_range_scale": 1.46,
        "f0_declination_st": 2.2,
        "speed_scale": 1.05,
        "pause_scale": 0.90,
        "gain_offset_db": 0.9,
        "rd_offset": -0.14,
        "bap_offset": 0.04,
        "tilt_db_oct": 1.6,
        "jitter_add": 0.002,
        "shimmer_add": 0.006,
        "tremor_depth_st": 0.0,
        "tremor_rate_hz": 5.6,
        "vowel_openness": 0.30,
        "formant_q": 0.16,
        "chest_db": -0.5,
        "warmth_db": 0.0,
        "presence_db": 2.0,
        "air_db": 1.6,
        "subharm_base": 0.0,
        "rubato_strength": 0.25,
    },
    "fearful": {
        "f0_shift_st": 1.4,
        "f0_range_scale": 1.12,
        "f0_declination_st": 0.4,
        "speed_scale": 1.09,
        "pause_scale": 1.12,
        "gain_offset_db": -0.7,
        "rd_offset": 0.26,
        "bap_offset": 0.24,
        "tilt_db_oct": -0.3,
        "jitter_add": 0.008,
        "shimmer_add": 0.032,
        "tremor_depth_st": 0.34,
        "tremor_rate_hz": 6.2,
        "vowel_openness": -0.08,
        "formant_q": 0.08,
        "chest_db": -1.0,
        "warmth_db": -0.5,
        "presence_db": 0.6,
        "air_db": 1.4,
        "subharm_base": 0.06,
        "rubato_strength": 0.30,
    },
    "whisper": {
        "f0_shift_st": -1.2,
        "f0_range_scale": 0.65,
        "f0_declination_st": -0.8,
        "speed_scale": 0.88,
        "pause_scale": 1.16,
        "gain_offset_db": -1.4,
        "rd_offset": 0.72,
        "bap_offset": 0.52,
        "tilt_db_oct": -0.6,
        "jitter_add": 0.004,
        "shimmer_add": 0.015,
        "tremor_depth_st": 0.0,
        "tremor_rate_hz": 5.0,
        "vowel_openness": -0.05,
        "formant_q": -0.10,
        "chest_db": 0.8,
        "warmth_db": 1.2,
        "presence_db": -0.8,
        "air_db": 2.8,
        "subharm_base": 0.04,
        "rubato_strength": 0.18,
    },
    "serious": {
        "f0_shift_st": -1.1,
        "f0_range_scale": 0.90,
        "f0_declination_st": -1.0,
        "speed_scale": 0.94,
        "pause_scale": 1.12,
        "gain_offset_db": 0.4,
        "rd_offset": -0.18,
        "bap_offset": -0.04,
        "tilt_db_oct": 0.6,
        "jitter_add": 0.0,
        "shimmer_add": 0.0,
        "tremor_depth_st": 0.0,
        "tremor_rate_hz": 5.0,
        "vowel_openness": -0.14,
        "formant_q": 0.18,
        "chest_db": 2.0,
        "warmth_db": 0.8,
        "presence_db": 1.6,
        "air_db": 0.0,
        "subharm_base": 0.04,
        "rubato_strength": 0.15,
    },
    "playful": {
        "f0_shift_st": 2.0,
        "f0_range_scale": 1.38,
        "f0_declination_st": 1.1,
        "speed_scale": 1.07,
        "pause_scale": 0.92,
        "gain_offset_db": 0.5,
        "rd_offset": -0.06,
        "bap_offset": 0.04,
        "tilt_db_oct": 1.1,
        "jitter_add": 0.001,
        "shimmer_add": 0.004,
        "tremor_depth_st": 0.0,
        "tremor_rate_hz": 5.6,
        "vowel_openness": 0.36,
        "formant_q": 0.12,
        "chest_db": -0.4,
        "warmth_db": 0.4,
        "presence_db": 1.4,
        "air_db": 1.6,
        "subharm_base": 0.0,
        "rubato_strength": 0.35,
    },
}

EMOTION_MODE_NAMES: tuple[str, ...] = (
    "auto",
    "neutral",
    "happy",
    "excited",
    "sad",
    "angry",
    "gentle",
    "surprised",
    "fearful",
    "whisper",
)

EMOTION_METADATA: list[dict[str, str]] = [
    {"id": "auto", "label": "✨ 自动感应", "desc": "根据文本情绪词、标点起伏与重音自动演绎"},
    {"id": "neutral", "label": "😐 平静直出", "desc": "保持 Stage 7 纯净播音原声"},
    {"id": "happy", "label": "😄 开心愉悦", "desc": "音调明亮上扬、微笑开阔口腔与轻快节奏"},
    {"id": "excited", "label": "🔥 激动高昂", "desc": "高音域大跨度起伏、紧凑声门与有力爆发"},
    {"id": "sad", "label": "😢 悲伤低落", "desc": "低沉慢速拖长、微颤气声与叹息感尾音"},
    {"id": "angry", "label": "😠 愤怒紧迫", "desc": "高声门紧实度、明亮高频压迫感与强重音"},
    {"id": "gentle", "label": "🌸 温柔安抚", "desc": "柔和宽松声门、温暖低中频与舒缓停顿"},
    {"id": "surprised", "label": "😲 惊讶愕然", "desc": "突升起调与剧烈尾音扬起"},
    {"id": "fearful", "label": "😨 紧张害怕", "desc": "气声夹杂高频声带微颤与微抖呼吸"},
    {"id": "whisper", "label": "🤫 轻声耳语", "desc": "极松弛高气声与近场柔和倾诉感"},
]

INLINE_TAG_CHIPS: list[dict[str, str]] = [
    {"tag": "[吸气]", "label": "吸气", "desc": "句前换气吸气声"},
    {"tag": "[叹气]", "label": "叹气", "desc": "带声带松弛的叹息声"},
    {"tag": "[笑]", "label": "轻笑", "desc": "笑声笑语节奏脉冲"},
    {"tag": "[停顿0.5s]", "label": "停顿", "desc": "显式停顿 0.5 秒"},
    {"tag": "[开心]", "label": "开心", "desc": "切换后续语句为开心情绪"},
    {"tag": "[激动]", "label": "激动", "desc": "切换为激动情绪"},
    {"tag": "[悲伤]", "label": "悲伤", "desc": "切换为悲伤情绪"},
    {"tag": "[愤怒]", "label": "愤怒", "desc": "切换为愤怒情绪"},
    {"tag": "[温柔]", "label": "温柔", "desc": "切换为温柔情绪"},
    {"tag": "[惊讶]", "label": "惊讶", "desc": "切换为惊讶情绪"},
    {"tag": "[耳语]", "label": "耳语", "desc": "切换为耳语气声"},
    {"tag": "*重点*", "label": "*重读*", "desc": "用 *...* 或 **...** 标注焦点重读"},
]

_TAG_TOKEN_RE = re.compile(r"\[([^\[\]\n]{1,28})\]")
_EMPH_DOUBLE_RE = re.compile(r"\*\*([^\*\n]+?)\*\*")
_EMPH_SINGLE_RE = re.compile(r"\*([^\*\n]+?)\*")
_ALL_CAPS_RE = re.compile(r"\b([A-Z]{3,})\b")
_CLAUSE_SPLIT_RE = re.compile(
    r"([^。！？!?;；\n……—~]*?(?:[?!？！]{1,3}|……+|…+|——|[。！？!?;；~\n]|$))"
)

_COMMON_ACRONYMS = frozenset({
    "TTS", "WAV", "PCM", "API", "HTTP", "HTTPS", "JSON", "CPU", "GPU", "DSP",
    "Larynx", "HTML", "CSS", "RMS", "DBFS", "FFT", "STFT", "LPC", "LSF",
    "AI", "USB", "RGB", "ASMR", "MFCC", "HNR", "MCD", "RTF", "UI", "ZH", "EN",
})

_VOCALIZATION_ALIASES: dict[str, tuple[str, float]] = {
    "breath": ("breath", 190.0),
    "inhale": ("breath", 190.0),
    "吸气": ("breath", 190.0),
    "换气": ("breath", 180.0),
    "gasp": ("gasp", 240.0),
    "倒吸气": ("gasp", 240.0),
    "惊呼": ("gasp", 240.0),
    "sigh": ("sigh", 450.0),
    "叹气": ("sigh", 450.0),
    "叹息": ("sigh", 450.0),
    "laugh": ("laugh", 520.0),
    "笑": ("laugh", 480.0),
    "大笑": ("laugh", 580.0),
    "哈哈": ("laugh", 520.0),
    "chuckle": ("chuckle", 360.0),
    "轻笑": ("chuckle", 360.0),
}

_EMOTION_TAG_ALIASES: dict[str, str] = {
    "neutral": "neutral",
    "calm": "neutral",
    "平静": "neutral",
    "正常": "neutral",
    "自然": "neutral",
    "happy": "happy",
    "cheerful": "happy",
    "开心": "happy",
    "高兴": "happy",
    "喜悦": "happy",
    "愉快": "happy",
    "excited": "excited",
    "激动": "excited",
    "兴奋": "excited",
    "热情": "excited",
    "sad": "sad",
    "sorrow": "sad",
    "悲伤": "sad",
    "难过": "sad",
    "伤心": "sad",
    "低落": "sad",
    "angry": "angry",
    "furious": "angry",
    "愤怒": "angry",
    "生气": "angry",
    "怒斥": "angry",
    "gentle": "gentle",
    "soft": "gentle",
    "tender": "gentle",
    "温柔": "gentle",
    "轻柔": "gentle",
    "安慰": "gentle",
    "surprised": "surprised",
    "shocked": "surprised",
    "惊讶": "surprised",
    "震惊": "surprised",
    "惊奇": "surprised",
    "fearful": "fearful",
    "scared": "fearful",
    "afraid": "fearful",
    "害怕": "fearful",
    "紧张": "fearful",
    "恐惧": "fearful",
    "whisper": "whisper",
    "whispers": "whisper",
    "耳语": "whisper",
    "悄悄话": "whisper",
    "气声": "whisper",
    "serious": "serious",
    "严肃": "serious",
    "playful": "playful",
    "调皮": "playful",
}

_PAUSE_TAG_RE = re.compile(r"^(?:pause|停顿)\s*(?:(\d+(?:\.\d+)?)\s*(ms|s|秒)?)?$", re.IGNORECASE)

_ZH_LEXICON: dict[str, tuple[str, ...]] = {
    "happy": ("开心", "高兴", "快乐", "太好了", "哈哈", "喜悦", "幸福", "真棒", "恭喜", "愉快", "欣喜", "真不错"),
    "excited": ("太震撼", "激动", "太棒了", "哇塞", "难以置信", "终于成功", "万岁", "天呐", "不得了", "太牛", "太赞"),
    "sad": ("难过", "伤心", "悲伤", "遗憾", "失落", "泪", "哭", "心痛", "绝望", "可惜", "孤单", "离开", "告别", "对不起", "叹气"),
    "angry": ("气死", "愤怒", "生气", "混蛋", "岂有此理", "荒谬", "可恶", "闭嘴", "受够了", "凭什么", "怎么敢", "滚"),
    "gentle": ("温柔", "轻轻", "晚安", "别怕", "没事的", "抱抱", "安静", "慢慢来", "放心", "陪着你", "亲爱的"),
    "surprised": ("什么", "居然", "竟然", "真的吗", "天啊", "不可能", "怎么会", "吓一跳", "啊？", "咦"),
    "fearful": ("害怕", "恐惧", "发抖", "别过来", "救命", "慌张", "吓死", "糟糕", "怎么办", "大事不好"),
    "whisper": ("小声", "嘘", "悄悄", "秘密", "别告诉", "耳边"),
}

_EN_LEXICON: dict[str, tuple[str, ...]] = {
    "happy": ("happy", "glad", "wonderful", "great", "awesome", "yay", "haha", "love", "delighted", "cheerful"),
    "excited": ("amazing", "incredible", "unbelievable", "omg", "woohoo", "finally", "fantastic", "brilliant"),
    "sad": ("sad", "sorry", "miss", "cry", "tears", "lonely", "heartbroken", "unfortunately", "goodbye", "sigh"),
    "angry": ("angry", "furious", "ridiculous", "unacceptable", "stop", "enough", "hate", "damn", "shut"),
    "gentle": ("gently", "softly", "sweetheart", "goodnight", "calm", "peaceful", "safe", "breathe"),
    "surprised": ("what", "really", "seriously", "whoa", "no way", "suddenly"),
    "fearful": ("scared", "afraid", "terrified", "help", "panic", "trembling", "careful"),
    "whisper": ("whisper", "secret", "quietly", "shh"),
}


@dataclass
class FocusSpan:
    """Character span of an emphasized word/phrase within a cleaned segment string."""

    start_char: int
    end_char: int
    level: int = 1  # 1 = normal focus (*word*), 2 = strong focus (**word** or ALL CAPS)


@dataclass
class ExpressiveSegment:
    """Parsed timeline unit for ElevenLabs-style emotional & non-verbal synthesis."""

    kind: str  # "speech", "breath", "gasp", "sigh", "laugh", "chuckle", "pause"
    text: str = ""
    emotion: str = "neutral"
    intensity: float = 0.0
    focus_spans: list[FocusSpan] = field(default_factory=list)
    punct_style: str = ""
    pause_after_ms: float = 0.0
    duration_ms: float = 0.0
    auto_breath_before: bool = False


def strip_performance_tags(text: str) -> str:
    """Remove inline `[tag]` markers and unwrap `*emphasis*` stars cleanly."""
    s = str(text or "")

    def _repl(m: re.Match[str]) -> str:
        inner = m.group(1).strip()
        key = inner.lower()
        if key in _VOCALIZATION_ALIASES or key in _EMOTION_TAG_ALIASES or _PAUSE_TAG_RE.match(inner):
            return " "
        return m.group(0)

    s = _TAG_TOKEN_RE.sub(_repl, s)
    s = _EMPH_DOUBLE_RE.sub(r"\1", s)
    s = _EMPH_SINGLE_RE.sub(r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_focus_and_clean(clause_raw: str) -> tuple[str, list[FocusSpan]]:
    """Extract `*word*`, `**word**`, and English ALL-CAPS focus spans and return clean text."""
    spans: list[FocusSpan] = []
    out_chars: list[str] = []
    i = 0
    n = len(clause_raw)
    while i < n:
        if clause_raw.startswith("**", i):
            end = clause_raw.find("**", i + 2)
            if end != -1 and end > i + 2 and "\n" not in clause_raw[i + 2 : end]:
                inner = clause_raw[i + 2 : end]
                st = len("".join(out_chars))
                out_chars.append(inner)
                en = len("".join(out_chars))
                if inner.strip():
                    spans.append(FocusSpan(st, en, level=2))
                i = end + 2
                continue
        if clause_raw[i] == "*":
            end = clause_raw.find("*", i + 1)
            if end != -1 and end > i + 1 and "\n" not in clause_raw[i + 1 : end]:
                inner = clause_raw[i + 1 : end]
                st = len("".join(out_chars))
                out_chars.append(inner)
                en = len("".join(out_chars))
                if inner.strip():
                    spans.append(FocusSpan(st, en, level=1))
                i = end + 1
                continue
        out_chars.append(clause_raw[i])
        i += 1

    clean = "".join(out_chars).strip()
    if clean:
        for m in _ALL_CAPS_RE.finditer(clean):
            tok = m.group(1)
            if tok in _COMMON_ACRONYMS:
                continue
            overlap = any(not (m.end() <= fs.start_char or m.start() >= fs.end_char) for fs in spans)
            if not overlap:
                spans.append(FocusSpan(m.start(), m.end(), level=2))
    spans.sort(key=lambda sp: sp.start_char)
    return clean, spans


def _detect_punct_style(clause_raw: str) -> str:
    t = clause_raw.rstrip()
    if t.endswith(("？！", "!?", "?!", "！？")):
        return "?!"
    if t.endswith(("……", "...")):
        return "..."
    if t.endswith(("~", "～")):
        return "~"
    if t.endswith(("——", "--")):
        return "--"
    if t.endswith(("?", "？")):
        return "?"
    if t.endswith(("!", "！")):
        return "!"
    return ""


def _auto_score_emotion(
    clean_text: str,
    punct_style: str,
    focus_spans: list[FocusSpan],
    prev_emotion: str,
) -> tuple[str, float]:
    """Rule + lexicon + punctuation emotion detector for `emotion_mode == 'auto'`."""
    scores: dict[str, float] = {k: 0.0 for k in EMOTION_PROFILES if k != "neutral"}
    low = clean_text.lower()

    for emo, words in _ZH_LEXICON.items():
        for w in words:
            if w in clean_text:
                scores[emo] += 1.4
    for emo, words in _EN_LEXICON.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", low):
                scores[emo] += 1.3

    if punct_style == "?!":
        scores["surprised"] += 1.2
        scores["angry"] += 0.6
    elif punct_style == "!":
        scores["excited"] += 0.75
        scores["happy"] += 0.45
        scores["angry"] += 0.35
    elif punct_style == "?":
        scores["surprised"] += 0.55
    elif punct_style == "...":
        scores["sad"] += 0.70
        scores["gentle"] += 0.45
    elif punct_style == "~":
        scores["happy"] += 0.85
        scores["gentle"] += 0.45

    if len(focus_spans) >= 1 and punct_style in ("!", "?!"):
        scores["excited"] += 0.35

    best_emo = max(scores, key=lambda k: scores[k])
    best_sc = scores[best_emo]
    if best_sc < 0.70:
        # Plain declarative sentence with no cues stays strictly neutral (0.0 intensity)
        if prev_emotion not in ("neutral", "") and punct_style in ("...", "!"):
            return prev_emotion, 0.35
        return "neutral", 0.0

    # Gentle inertia if previous clause already established a compatible emotion
    if prev_emotion in scores and scores[prev_emotion] >= 0.6 * best_sc and prev_emotion != best_emo:
        scores[prev_emotion] += 0.30
        best_emo = max(scores, key=lambda k: scores[k])
        best_sc = scores[best_emo]

    conf = min(1.0, 0.48 + 0.28 * best_sc)
    return best_emo, round(conf, 3)


def parse_expressive_text(
    text: str,
    *,
    emotion_mode: str = "auto",
    emotion_intensity: float = 0.75,
    stability: float = 0.55,
    auto_breath: bool = True,
) -> list[ExpressiveSegment]:
    """Parse raw text + `[tags]` + `*emphasis*` + punctuation into ExpressiveSegments."""
    src = str(text or "").strip()
    if not src:
        return []

    mode = str(emotion_mode or "auto").strip().lower()
    if mode not in ("auto",) and mode not in EMOTION_PROFILES:
        mode = "auto"
    base_int = max(0.0, min(1.5, float(emotion_intensity)))
    stab = max(0.0, min(1.0, float(stability)))

    # Tokenize inline tags while preserving literal bracketed non-performance text.
    tokens: list[tuple[str, Any]] = []
    pos = 0
    for m in _TAG_TOKEN_RE.finditer(src):
        if m.start() > pos:
            tokens.append(("text", src[pos : m.start()]))
        inner = m.group(1).strip()
        key = inner.lower()
        pm = _PAUSE_TAG_RE.match(inner)
        if key in _VOCALIZATION_ALIASES:
            vkind, vdur = _VOCALIZATION_ALIASES[key]
            tokens.append(("voc", (vkind, vdur)))
        elif pm:
            val_s = pm.group(1)
            unit = (pm.group(2) or "s").lower()
            if val_s is None:
                dur_ms = 420.0
            else:
                v_num = float(val_s)
                dur_ms = v_num if unit == "ms" else v_num * 1000.0
            tokens.append(("pause", max(80.0, min(4000.0, dur_ms))))
        elif key in _EMOTION_TAG_ALIASES:
            tokens.append(("emo", _EMOTION_TAG_ALIASES[key]))
        else:
            tokens.append(("text", m.group(0)))
        pos = m.end()
    if pos < len(src):
        tokens.append(("text", src[pos:]))

    segments: list[ExpressiveSegment] = []
    cur_explicit_emo: str | None = None
    last_auto_emo = "neutral"

    for tkind, tval in tokens:
        if tkind == "emo":
            cur_explicit_emo = str(tval)
            continue
        if tkind == "pause":
            segments.append(ExpressiveSegment(kind="pause", duration_ms=float(tval)))
            continue
        if tkind == "voc":
            vkind, vdur = tval
            segments.append(
                ExpressiveSegment(
                    kind=str(vkind),
                    emotion=cur_explicit_emo or (mode if mode != "auto" else last_auto_emo),
                    intensity=max(0.35, min(1.2, base_int)),
                    duration_ms=float(vdur),
                )
            )
            continue

        # Split raw text run into clauses for per-clause punctuation/focus/emotion dynamics
        raw_clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.findall(str(tval)) if c and c.strip()]
        if not raw_clauses:
            continue

        for cl_raw in raw_clauses:
            p_style = _detect_punct_style(cl_raw)
            clean, spans = _extract_focus_and_clean(cl_raw)
            if not clean:
                continue

            if cur_explicit_emo is not None:
                emo = cur_explicit_emo
                eff_int = base_int if emo != "neutral" else 0.0
            elif mode != "auto":
                emo = mode
                eff_int = base_int if emo != "neutral" else 0.0
            else:
                emo, auto_conf = _auto_score_emotion(clean, p_style, spans, last_auto_emo)
                eff_int = base_int * auto_conf
                last_auto_emo = emo

            # Add extra pause hint on ellipsis or dramatic dash
            p_after = 0.0
            if p_style == "...":
                p_after = 220.0 * (1.15 - 0.3 * stab)
            elif p_style == "--":
                p_after = 160.0

            need_breath = False
            if auto_breath and stab < 0.78 and eff_int >= 0.45:
                high_arousal = emo in ("sad", "excited", "angry", "fearful", "surprised")
                already_voc = len(segments) > 0 and segments[-1].kind in ("breath", "gasp", "sigh", "pause")
                if not already_voc and (
                    (high_arousal and len(clean) >= 8 and len(segments) == 0)
                    or (len(clean) >= 18)
                ):
                    need_breath = True

            segments.append(
                ExpressiveSegment(
                    kind="speech",
                    text=clean,
                    emotion=emo,
                    intensity=round(eff_int, 3),
                    focus_spans=spans,
                    punct_style=p_style,
                    pause_after_ms=p_after,
                    auto_breath_before=need_breath,
                )
            )

    return segments
