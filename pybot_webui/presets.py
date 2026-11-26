"""Voice parameter schema, validation, and curated presets (男女老少 + whisper)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from exp_20_pybot_larynx.expressive.emotion_parser import (
    EMOTION_METADATA,
    EMOTION_MODE_NAMES,
    INLINE_TAG_CHIPS,
)
from exp_20_pybot_larynx.larynx.timbre_synth import (
    TIMBRE_BOUNDS,
    TIMBRE_PATCH_MAP,
    TIMBRE_PATCHES,
)

RULE_NAMES: tuple[str, ...] = (
    "glottal_f0_coupling",
    "obstruent_crispness",
    "nasal_antiresonance",
    "boundary_coarticulation",
    "prosodic_energy_tilt",
    "formant_dynamics",
)

RULE_METADATA: list[dict[str, str]] = [
    {
        "id": "glottal_f0_coupling",
        "label": "声门F0耦合",
        "desc": "音高-气声-能量动态耦合、VOT起始过渡与轻微声带波动",
    },
    {
        "id": "obstruent_crispness",
        "label": "阻塞音清晰度",
        "desc": "擦音/塞擦音/塞音高分辨频谱塑形与干净的清浊切分",
    },
    {
        "id": "nasal_antiresonance",
        "label": "鼻音反共振",
        "desc": "鼻声母与鼻韵尾的低频反共振谷深度调节",
    },
    {
        "id": "boundary_coarticulation",
        "label": "边界协同发音",
        "desc": "音节间共鸣过渡、倒谱接缝桥接与长元音微动态",
    },
    {
        "id": "prosodic_energy_tilt",
        "label": "韵律能量与声门放松",
        "desc": "语境/韵律位置能量残差迁移及低音域上声声门自然放松",
    },
    {
        "id": "formant_dynamics",
        "label": "共振峰动态塑形",
        "desc": "复合韵母滑音张力、舌尖音清晰度与轻声温和弱化",
    },
]

BOUNDS: dict[str, tuple[float, float]] = {
    "pitch_shift_st": (-14.0, 10.0),
    "vtln_scale": (0.78, 1.25),
    "speed": (0.60, 1.60),
    "pitch_range": (0.50, 1.60),
    "rd_offset": (-0.45, 0.85),
    "breathiness": (-0.25, 0.65),
    "tilt_db_per_oct": (-3.5, 3.5),
    "hoarseness": (0.0, 1.0),
    "pause_scale": (0.40, 2.00),
    "target_dbfs": (-24.0, -12.0),
    "imitation_strength": (0.0, 1.0),
    "emotion_intensity": (0.0, 1.5),
    "stability": (0.0, 1.0),
    **TIMBRE_BOUNDS,
}


def _clamp(val: Any, key: str, default: float) -> float:
    lo, hi = BOUNDS[key]
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN guard
        return default
    return max(lo, min(hi, v))


def _default_rules_enabled() -> dict[str, bool]:
    return {r: True for r in RULE_NAMES}


@dataclass
class VoiceParams:
    """Physical electronic larynx & prosody control parameters."""

    preset_id: str = "female_std"
    pitch_shift_st: float = 0.0
    vtln_scale: float = 1.0
    speed: float = 1.0
    pitch_range: float = 1.0
    rd_offset: float = 0.0
    breathiness: float = 0.0
    tilt_db_per_oct: float = 0.0
    hoarseness: float = 0.0
    pause_scale: float = 1.0
    glottal_mode: str = "speaker_coherent"
    crossling_mode: str = "auto"
    rules_enabled: dict[str, bool] = field(default_factory=_default_rules_enabled)
    seed: int = 123
    target_dbfs: float = -18.0
    clone_profile_id: str | None = None
    imitation_strength: float = 0.85
    clone_f0: bool = True
    clone_rate: bool = False
    emotion_mode: str = "auto"
    emotion_intensity: float = 0.75
    stability: float = 0.55
    auto_breath: bool = True
    timbre_patch: str = "neutral"
    chest_db: float = 0.0
    warmth_db: float = 0.0
    nasal_twang_db: float = 0.0
    presence_db: float = 0.0
    air_db: float = 0.0
    formant_q: float = 0.0
    vowel_openness: float = 0.0
    vocal_fry: float = 0.0
    vibrato_depth_st: float = 0.0
    vibrato_rate_hz: float = 5.4
    exciter_drive: float = 0.0
    bandpass_vintage: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> VoiceParams:
        """Build clamped VoiceParams from a JSON/dict payload."""
        d = data or {}
        preset_id = str(d.get("preset_id") or "female_std")
        base_dict = get_preset_params_dict(preset_id) if preset_id in PRESET_MAP else {}
        patch_id = str(d.get("timbre_patch") or "neutral")
        if patch_id not in TIMBRE_PATCH_MAP:
            patch_id = "neutral"
        patch_dict = dict(TIMBRE_PATCH_MAP[patch_id]["params"]) if patch_id in TIMBRE_PATCH_MAP else {}
        merged = {**base_dict, **patch_dict, **d}

        glottal = str(merged.get("glottal_mode") or "speaker_coherent")
        if glottal not in ("speaker_coherent", "dynamic_coherent", "parametric_lf"):
            glottal = "speaker_coherent"

        crossling = str(merged.get("crossling_mode") or "auto")
        if crossling not in ("auto", "zh_only"):
            crossling = "auto"

        emo_mode = str(merged.get("emotion_mode") or "auto").strip().lower()
        if emo_mode not in EMOTION_MODE_NAMES:
            emo_mode = "auto"

        raw_rules = merged.get("rules_enabled")
        rules = _default_rules_enabled()
        if isinstance(raw_rules, dict):
            for r in RULE_NAMES:
                if r in raw_rules:
                    rules[r] = bool(raw_rules[r])

        try:
            seed = int(merged.get("seed", 123))
        except (TypeError, ValueError):
            seed = 123
        seed = max(0, min(9999999, seed))

        raw_cid = d.get("clone_profile_id")
        if raw_cid is None or str(raw_cid).strip().lower() in ("", "none", "null"):
            clone_profile_id = None
        else:
            clone_profile_id = str(raw_cid).strip()

        return cls(
            preset_id=preset_id,
            pitch_shift_st=_clamp(merged.get("pitch_shift_st", 0.0), "pitch_shift_st", 0.0),
            vtln_scale=_clamp(merged.get("vtln_scale", 1.0), "vtln_scale", 1.0),
            speed=_clamp(merged.get("speed", 1.0), "speed", 1.0),
            pitch_range=_clamp(merged.get("pitch_range", 1.0), "pitch_range", 1.0),
            rd_offset=_clamp(merged.get("rd_offset", 0.0), "rd_offset", 0.0),
            breathiness=_clamp(merged.get("breathiness", 0.0), "breathiness", 0.0),
            tilt_db_per_oct=_clamp(merged.get("tilt_db_per_oct", 0.0), "tilt_db_per_oct", 0.0),
            hoarseness=_clamp(merged.get("hoarseness", 0.0), "hoarseness", 0.0),
            pause_scale=_clamp(merged.get("pause_scale", 1.0), "pause_scale", 1.0),
            glottal_mode=glottal,
            crossling_mode=crossling,
            rules_enabled=rules,
            seed=seed,
            target_dbfs=_clamp(merged.get("target_dbfs", -18.0), "target_dbfs", -18.0),
            clone_profile_id=clone_profile_id,
            imitation_strength=_clamp(merged.get("imitation_strength", 0.85), "imitation_strength", 0.85),
            clone_f0=bool(merged.get("clone_f0", True)),
            clone_rate=bool(merged.get("clone_rate", False)),
            emotion_mode=emo_mode,
            emotion_intensity=_clamp(merged.get("emotion_intensity", 0.75), "emotion_intensity", 0.75),
            stability=_clamp(merged.get("stability", 0.55), "stability", 0.55),
            auto_breath=bool(merged.get("auto_breath", True)),
            timbre_patch=patch_id,
            chest_db=_clamp(merged.get("chest_db", 0.0), "chest_db", 0.0),
            warmth_db=_clamp(merged.get("warmth_db", 0.0), "warmth_db", 0.0),
            nasal_twang_db=_clamp(merged.get("nasal_twang_db", 0.0), "nasal_twang_db", 0.0),
            presence_db=_clamp(merged.get("presence_db", 0.0), "presence_db", 0.0),
            air_db=_clamp(merged.get("air_db", 0.0), "air_db", 0.0),
            formant_q=_clamp(merged.get("formant_q", 0.0), "formant_q", 0.0),
            vowel_openness=_clamp(merged.get("vowel_openness", 0.0), "vowel_openness", 0.0),
            vocal_fry=_clamp(merged.get("vocal_fry", 0.0), "vocal_fry", 0.0),
            vibrato_depth_st=_clamp(merged.get("vibrato_depth_st", 0.0), "vibrato_depth_st", 0.0),
            vibrato_rate_hz=_clamp(merged.get("vibrato_rate_hz", 5.4), "vibrato_rate_hz", 5.4),
            exciter_drive=_clamp(merged.get("exciter_drive", 0.0), "exciter_drive", 0.0),
            bandpass_vintage=_clamp(merged.get("bandpass_vintage", 0.0), "bandpass_vintage", 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize parameters to JSON-friendly dict."""
        return asdict(self)


VOICE_PRESETS: list[dict[str, Any]] = [
    {
        "id": "female_std",
        "name": "青年女声（标准）",
        "group": "女声",
        "desc": "Stage 4 原声线（248 Hz），清晰自然均衡",
        "params": {
            "preset_id": "female_std",
            "pitch_shift_st": 0.0,
            "vtln_scale": 1.00,
            "speed": 1.00,
            "pitch_range": 1.00,
            "rd_offset": 0.0,
            "breathiness": 0.0,
            "tilt_db_per_oct": 0.0,
            "hoarseness": 0.0,
            "pause_scale": 1.00,
            "glottal_mode": "speaker_coherent",
        },
    },
    {
        "id": "female_gentle",
        "name": "温柔女声（柔和）",
        "group": "女声",
        "desc": "音高略降、声门略松带轻柔气声、语速舒缓",
        "params": {
            "preset_id": "female_gentle",
            "pitch_shift_st": -1.5,
            "vtln_scale": 0.98,
            "speed": 0.92,
            "pitch_range": 0.92,
            "rd_offset": 0.28,
            "breathiness": 0.14,
            "tilt_db_per_oct": -0.8,
            "hoarseness": 0.05,
            "pause_scale": 1.08,
            "glottal_mode": "parametric_lf",
        },
    },
    {
        "id": "girl_bright",
        "name": "元气少女（清甜）",
        "group": "女声",
        "desc": "音高更明亮、声道略短、抑扬幅度更大",
        "params": {
            "preset_id": "girl_bright",
            "pitch_shift_st": 2.8,
            "vtln_scale": 1.08,
            "speed": 1.05,
            "pitch_range": 1.14,
            "rd_offset": -0.08,
            "breathiness": 0.03,
            "tilt_db_per_oct": 0.5,
            "hoarseness": 0.0,
            "pause_scale": 0.95,
            "glottal_mode": "parametric_lf",
        },
    },
    {
        "id": "child",
        "name": "清脆童声（少儿）",
        "group": "少儿",
        "desc": "短声道高频共振峰 + 高音区活泼语调",
        "params": {
            "preset_id": "child",
            "pitch_shift_st": 5.5,
            "vtln_scale": 1.16,
            "speed": 1.08,
            "pitch_range": 1.20,
            "rd_offset": -0.10,
            "breathiness": 0.05,
            "tilt_db_per_oct": 0.8,
            "hoarseness": 0.0,
            "pause_scale": 0.95,
            "glottal_mode": "parametric_lf",
        },
    },
    {
        "id": "male_young",
        "name": "青年男声（清朗）",
        "group": "男声",
        "desc": "约 143 Hz 男中音区 + 0.87 声道拉伸",
        "params": {
            "preset_id": "male_young",
            "pitch_shift_st": -9.5,
            "vtln_scale": 0.87,
            "speed": 0.98,
            "pitch_range": 0.94,
            "rd_offset": -0.15,
            "breathiness": -0.02,
            "tilt_db_per_oct": 0.7,
            "hoarseness": 0.06,
            "pause_scale": 1.00,
            "glottal_mode": "parametric_lf",
        },
    },
    {
        "id": "male_mature",
        "name": "沉稳大叔（低沉）",
        "group": "男声",
        "desc": "约 120 Hz 厚重低音 + 长声道浑厚胸腔感",
        "params": {
            "preset_id": "male_mature",
            "pitch_shift_st": -12.5,
            "vtln_scale": 0.82,
            "speed": 0.90,
            "pitch_range": 0.86,
            "rd_offset": -0.18,
            "breathiness": 0.02,
            "tilt_db_per_oct": 1.0,
            "hoarseness": 0.18,
            "pause_scale": 1.08,
            "glottal_mode": "parametric_lf",
        },
    },
    {
        "id": "elder_female",
        "name": "慈祥老奶奶（老年女）",
        "group": "老年",
        "desc": "语速放慢、适度声门震颤与岁月沙哑感",
        "params": {
            "preset_id": "elder_female",
            "pitch_shift_st": -3.5,
            "vtln_scale": 0.95,
            "speed": 0.82,
            "pitch_range": 0.88,
            "rd_offset": 0.32,
            "breathiness": 0.16,
            "tilt_db_per_oct": -0.9,
            "hoarseness": 0.55,
            "pause_scale": 1.25,
            "glottal_mode": "parametric_lf",
        },
    },
    {
        "id": "elder_male",
        "name": "沧桑老爷爷（老年男）",
        "group": "老年",
        "desc": "低沉男声 + 明显声带微颤/沙哑与宽停顿",
        "params": {
            "preset_id": "elder_male",
            "pitch_shift_st": -11.5,
            "vtln_scale": 0.83,
            "speed": 0.80,
            "pitch_range": 0.84,
            "rd_offset": 0.25,
            "breathiness": 0.20,
            "tilt_db_per_oct": -0.2,
            "hoarseness": 0.65,
            "pause_scale": 1.30,
            "glottal_mode": "parametric_lf",
        },
    },
    {
        "id": "whisper",
        "name": "气声耳语（特色）",
        "group": "特色",
        "desc": "高非周期气声占比 + 极松声门波形（近耳语）",
        "params": {
            "preset_id": "whisper",
            "pitch_shift_st": -2.0,
            "vtln_scale": 1.00,
            "speed": 0.88,
            "pitch_range": 0.70,
            "rd_offset": 0.75,
            "breathiness": 0.58,
            "tilt_db_per_oct": -0.4,
            "hoarseness": 0.15,
            "pause_scale": 1.12,
            "glottal_mode": "parametric_lf",
        },
    },
]

PRESET_MAP: dict[str, dict[str, Any]] = {p["id"]: p for p in VOICE_PRESETS}


def get_preset_params_dict(preset_id: str) -> dict[str, Any]:
    """Return a copy of the parameter dictionary for a preset ID."""
    p = PRESET_MAP.get(preset_id, VOICE_PRESETS[0])
    return dict(p["params"])


KNOB_SCHEMA: list[dict[str, Any]] = [
    {
        "key": "pitch_shift_st",
        "label": "整体变调",
        "min": -14.0,
        "max": 10.0,
        "step": 0.5,
        "default": 0.0,
        "unit": "st",
        "group": "声带音高与声道",
        "tooltip": "以半音为单位平移基频 F0，负数偏男声/低沉，正数偏少女/童声",
    },
    {
        "key": "vtln_scale",
        "label": "声道长度 (VTLN)",
        "min": 0.78,
        "max": 1.25,
        "step": 0.01,
        "default": 1.00,
        "unit": "×",
        "group": "声带音高与声道",
        "tooltip": "低于 1.0 模拟更长声道（男声共振峰下移），高于 1.0 模拟更短声道（少儿共振峰上移）",
    },
    {
        "key": "speed",
        "label": "语速倍率",
        "min": 0.60,
        "max": 1.60,
        "step": 0.02,
        "default": 1.00,
        "unit": "×",
        "group": "韵律与节奏",
        "tooltip": "大于 1.0 语速更快，小于 1.0 更舒缓",
    },
    {
        "key": "pitch_range",
        "label": "语调起伏",
        "min": 0.50,
        "max": 1.60,
        "step": 0.02,
        "default": 1.00,
        "unit": "×",
        "group": "韵律与节奏",
        "tooltip": "缩放音高围绕均值波动的幅度，越大抑扬顿挫越强",
    },
    {
        "key": "pause_scale",
        "label": "停顿倍率",
        "min": 0.40,
        "max": 2.00,
        "step": 0.05,
        "default": 1.00,
        "unit": "×",
        "group": "韵律与节奏",
        "tooltip": "缩放标点与韵律边界处的静音停顿时长",
    },
    {
        "key": "rd_offset",
        "label": "声门松紧 (Rd)",
        "min": -0.45,
        "max": 0.85,
        "step": 0.02,
        "default": 0.00,
        "unit": "",
        "group": "声门质感",
        "tooltip": "调节 LF 声门脉冲波形参数 Rd：负值更紧实有力，正值更松弛柔和",
    },
    {
        "key": "breathiness",
        "label": "气声程度 (BAP)",
        "min": -0.25,
        "max": 0.65,
        "step": 0.02,
        "default": 0.00,
        "unit": "",
        "group": "声门质感",
        "tooltip": "增减有声段非周期湍流噪声比例，偏大呈温柔气声/耳语感",
    },
    {
        "key": "tilt_db_per_oct",
        "label": "音色明暗 (Tilt)",
        "min": -3.5,
        "max": 3.5,
        "step": 0.1,
        "default": 0.0,
        "unit": "dB/oct",
        "group": "声门质感",
        "tooltip": "高频频谱倾斜调节：正值更清亮靠前，负值更温暖厚暗",
    },
    {
        "key": "hoarseness",
        "label": "老化/沙哑",
        "min": 0.00,
        "max": 1.00,
        "step": 0.02,
        "default": 0.00,
        "unit": "",
        "group": "声门质感",
        "tooltip": "增加声带周期抖动(Jitter)、振幅微颤(Shimmer)与生理慢颤音",
    },
    {
        "key": "target_dbfs",
        "label": "输出响度",
        "min": -24.0,
        "max": -12.0,
        "step": 0.5,
        "default": -18.0,
        "unit": "dBFS",
        "group": "声门质感",
        "tooltip": "输出 WAV 的有效音频 RMS 响度归一化目标",
    },
]

SYNTH_KNOB_SCHEMA: list[dict[str, Any]] = [
    {"key": "chest_db", "label": "胸腔低频 (240Hz)", "min": -8.0, "max": 8.0, "step": 0.2, "default": 0.0, "unit": "dB", "group": "频谱共鸣均衡", "tooltip": "增强/减弱胸腔低频共鸣厚重感"},
    {"key": "warmth_db", "label": "喉体暖度 (700Hz)", "min": -8.0, "max": 8.0, "step": 0.2, "default": 0.0, "unit": "dB", "group": "频谱共鸣均衡", "tooltip": "调节中低频元音温暖丰满度"},
    {"key": "nasal_twang_db", "label": "鼻腔聚焦 (1.6kHz)", "min": -8.0, "max": 8.0, "step": 0.2, "default": 0.0, "unit": "dB", "group": "频谱共鸣均衡", "tooltip": "增强则更靠前犀利有金属感，负值更圆润暗哑"},
    {"key": "presence_db", "label": "歌手共振峰 (3.3kHz)", "min": -8.0, "max": 8.0, "step": 0.2, "default": 0.0, "unit": "dB", "group": "频谱共鸣均衡", "tooltip": "人声临场感与咬字穿透力频段"},
    {"key": "air_db", "label": "高音空气感 (>6.2kHz)", "min": -8.0, "max": 8.0, "step": 0.2, "default": 0.0, "unit": "dB", "group": "频谱共鸣均衡", "tooltip": "高频呼吸丝滑亮度与近场 ASMR 空气感"},
    {"key": "formant_q", "label": "共振峰锐度 Q", "min": -0.60, "max": 0.80, "step": 0.02, "default": 0.0, "unit": "", "group": "共振峰与质感雕刻", "tooltip": "通过倒谱滤波收窄或放宽共振峰带宽"},
    {"key": "vowel_openness", "label": "微笑开口度 (F1/F2)", "min": -1.0, "max": 1.0, "step": 0.05, "default": 0.0, "unit": "", "group": "共振峰与质感雕刻", "tooltip": "正值模拟微笑开口音色(F1/F2上移)，负值偏含蓄暗沉"},
    {"key": "vocal_fry", "label": "气泡音质感 (Fry)", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.0, "unit": "", "group": "共振峰与质感雕刻", "tooltip": "声门次谐波颗粒感，常用于磁性/ASMR/低沉倾诉"},
    {"key": "vibrato_depth_st", "label": "自然颤音幅度", "min": 0.0, "max": 0.60, "step": 0.02, "default": 0.0, "unit": "st", "group": "共振峰与质感雕刻", "tooltip": "长音节延迟进入的平滑基频颤音幅度"},
    {"key": "vibrato_rate_hz", "label": "颤音速率", "min": 4.0, "max": 7.5, "step": 0.1, "default": 5.4, "unit": "Hz", "group": "共振峰与质感雕刻", "tooltip": "自然抒情颤音频率 (常见 5.0–6.0 Hz)"},
    {"key": "exciter_drive", "label": "谐波激励 (Exciter)", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.0, "unit": "", "group": "共振峰与质感雕刻", "tooltip": "收紧声门脉冲并点亮高频泛音明亮度"},
    {"key": "bandpass_vintage", "label": "复古电话带通", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.0, "unit": "", "group": "共振峰与质感雕刻", "tooltip": "切掉 360 Hz 以下与 3.5 kHz 以上模拟老式对讲/留声机"},
]
