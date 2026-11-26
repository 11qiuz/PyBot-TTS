"""Physical voice parameter transforms, synthesis engine, and WAV export manager."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime
import io
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid
import wave

import numpy as np
from scipy.fft import rfftfreq

from exp_20_pybot_larynx.acoustics.knowledge_stage2 import AcousticKnowledgeStage2
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import (
    STAGE5_KNOWLEDGE_PATH,
    AcousticKnowledgeStage5,
)
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import (
    DEFAULT_UV_LEVEL_BOOST_DB_S7,
    STAGE7_KNOWLEDGE_PATH,
    AcousticKnowledgeStage7,
)
from exp_20_pybot_larynx.arbiter import Arbiter
from exp_20_pybot_larynx.crossling import get_en_articulation_table, get_speaker_donor_bank
from exp_20_pybot_larynx.crossling.en_frontend_v2 import load_cmudict
from exp_20_pybot_larynx.crossling.en_stage10_renderer import (
    PROSODY_EN_STAGE10_PATH,
    is_stage10_en_ready,
)
from exp_20_pybot_larynx.crossling.knowledge_en_stage10 import get_en_stage10_knowledge
from exp_20_pybot_larynx.clone import (
    VoiceCloneProfile,
    apply_clone_to_control,
    decode_wav_bytes_to_24k_mono,
    extract_clone_profile,
    load_base_speaker_profile,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl, synthesize_larynx
from exp_20_pybot_larynx.larynx.synth_v2 import synthesize_larynx_v2
from exp_20_pybot_larynx.larynx.timbre_synth import (
    TimbreSynthParams,
    apply_timbre_synth,
    extract_spectral_profile_preview,
)
from exp_20_pybot_larynx.rules import default_stage0_rules, default_stage5_rules
from exp_v15_bznsyp_tts.index_bznsyp_v15 import load_speaker_assets_v15
from pybot_webui import DEFAULT_SAVE_DIR, HOP, N_BINS, N_FFT, SR
from pybot_webui.crossling import build_utterance_control, soften_mixed_onset_levels
from pybot_webui.presets import (
    EMOTION_METADATA,
    INLINE_TAG_CHIPS,
    KNOB_SCHEMA,
    PRESET_MAP,
    RULE_METADATA,
    RULE_NAMES,
    SYNTH_KNOB_SCHEMA,
    TIMBRE_PATCHES,
    VOICE_PRESETS,
    VoiceParams,
)
from pybot_webui.splitter import detect_lang_tag

_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_NYQ = 0.5 * SR
_VTLN_PIVOT_HZ = 7500.0
_BREATH_BAND_WEIGHTS = np.array([0.55, 0.85, 1.00, 1.00, 0.90, 0.80], dtype=np.float64)
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f\s]+')
DEFAULT_CLONE_DIR = Path(__file__).resolve().parent / "clone_profiles"
_SAFE_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _warp_log_env(log_env: np.ndarray, mask: np.ndarray, vtln_scale: float) -> None:
    """Apply vectorized piecewise-linear VTLN spectral envelope warping in-place."""
    if not np.any(mask) or abs(vtln_scale - 1.0) < 1e-4:
        return
    scale = max(0.75, min(1.30, float(vtln_scale)))
    if scale <= 1.0:
        f_out_mid = _VTLN_PIVOT_HZ * scale
        f_src_mid = _VTLN_PIVOT_HZ
    else:
        f_out_mid = _VTLN_PIVOT_HZ
        f_src_mid = _VTLN_PIVOT_HZ / scale

    f_src = np.interp(
        _FREQS,
        (0.0, f_out_mid, _NYQ),
        (0.0, f_src_mid, _NYQ),
    )
    bin_pos = np.clip(f_src * (N_FFT / SR), 0.0, float(N_BINS - 1))
    i0 = np.floor(bin_pos).astype(np.intp)
    i1 = np.minimum(N_BINS - 1, i0 + 1)
    alpha = (bin_pos - i0)[None, :]
    sub = log_env[mask]
    log_env[mask] = (1.0 - alpha) * sub[:, i0] + alpha * sub[:, i1]


def apply_voice_params(ctrl: LarynxControl, params: VoiceParams, seed: int) -> LarynxControl:
    """Apply physical vocal-tract, F0, glottal LF, breathiness, tilt, and aging transforms."""
    out = ctrl.copy()
    n_fr = len(out.f0_hz)
    if n_fr == 0:
        return out

    v_mask = out.voiced & (out.f0_hz > 40.0)
    if np.any(v_mask):
        log_f0 = np.log2(out.f0_hz[v_mask])
        med_log = float(np.median(log_f0))
        shifted_log = (
            med_log
            + float(params.pitch_range) * (log_f0 - med_log)
            + (float(params.pitch_shift_st) / 12.0)
        )
        out.f0_hz[v_mask] = np.clip(2.0 ** shifted_log, 55.0, 520.0)

    # VTLN vocal-tract warping (softened on unvoiced frames to keep sibilants crisp)
    vtln = float(params.vtln_scale)
    if abs(vtln - 1.0) >= 1e-4:
        _warp_log_env(out.log_env, v_mask, vtln)
        uv_scale = 1.0 + 0.45 * (vtln - 1.0)
        _warp_log_env(out.log_env, ~v_mask, uv_scale)

    # Glottal LF waveshape parameter rd
    if out.rd is not None and abs(float(params.rd_offset)) >= 1e-4:
        out.rd = np.clip(out.rd + float(params.rd_offset), 0.50, 2.20)

    # Voiced breathiness (BAP shift)
    breath = float(params.breathiness)
    if abs(breath) >= 1e-4 and np.any(v_mask):
        out.bap[v_mask] = np.clip(
            out.bap[v_mask] + breath * _BREATH_BAND_WEIGHTS[None, :],
            0.005,
            0.995,
        )

    # Spectral tilt modulation
    tilt = float(params.tilt_db_per_oct)
    if abs(tilt) >= 1e-4 and out.glottal_tilt_db_per_oct is not None:
        out.glottal_tilt_db_per_oct = np.clip(
            out.glottal_tilt_db_per_oct + tilt,
            -8.0,
            8.0,
        )

    # Aging / hoarseness (jitter, shimmer, subtle 4.8 Hz tremor, low-band roughness)
    hoarse = max(0.0, min(1.0, float(params.hoarseness)))
    if hoarse >= 1e-4:
        if out.jitter_frac is not None:
            out.jitter_frac = np.clip(out.jitter_frac + 0.018 * hoarse, 0.0, 0.03)
        if out.shimmer_frac is not None:
            out.shimmer_frac = np.clip(out.shimmer_frac + 0.070 * hoarse, 0.0, 0.12)
        if hoarse > 0.15 and np.any(v_mask):
            rng = np.random.default_rng(int(seed) + 7919)
            phi0 = float(rng.uniform(0.0, math.tau))
            t_fr = np.arange(n_fr, dtype=np.float64) * (HOP / SR)
            tremor_st = (0.26 * hoarse) * np.sin(math.tau * 4.8 * t_fr + phi0)
            out.f0_hz[v_mask] = np.clip(
                out.f0_hz[v_mask] * (2.0 ** (tremor_st[v_mask] / 12.0)),
                55.0,
                520.0,
            )
            out.bap[v_mask, :3] = np.clip(out.bap[v_mask, :3] + 0.08 * hoarse, 0.005, 0.995)

    return out


def normalize_audio(y: np.ndarray, target_dbfs: float = -18.0, max_peak: float = 0.95) -> np.ndarray:
    """Normalize active speech RMS to target_dbfs with soft peak cap."""
    arr = np.asarray(y, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    peak = float(np.max(np.abs(arr)))
    if peak <= 1e-9:
        return np.zeros_like(arr)
    act = arr[np.abs(arr) > 0.01 * peak]
    rms = float(np.sqrt(np.mean(act * act))) if act.size else float(np.sqrt(np.mean(arr * arr)))
    if rms <= 1e-12:
        return arr.copy()
    gain = (10.0 ** (float(target_dbfs) / 20.0)) / rms
    scaled = arr * gain
    new_peak = float(np.max(np.abs(scaled)))
    if new_peak > max_peak:
        scaled = scaled * (max_peak / new_peak)
    return scaled


def encode_wav_bytes(samples: np.ndarray, sr: int = SR) -> bytes:
    """Encode float64 [-1, 1] mono audio into 16-bit PCM WAV bytes."""
    arr = np.asarray(samples, dtype=np.float64).ravel()
    pcm = np.round(np.clip(arr, -0.9999, 0.9999) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def sanitize_filename_stem(text: str, max_len: int = 24) -> str:
    """Build a Windows-safe filename stem from utterance text."""
    cleaned = _UNSAFE_FILENAME_RE.sub("_", text.strip()).strip("._")
    if not cleaned:
        cleaned = "clip"
    return cleaned[:max_len].rstrip("._") or "clip"


def _unique_wav_path(directory: Path, base_stem: str) -> Path:
    candidate = directory / f"{base_stem}.wav"
    if not candidate.exists():
        return candidate
    idx = 2
    while True:
        cand = directory / f"{base_stem}_{idx}.wav"
        if not cand.exists():
            return cand
        idx += 1


@dataclass
class ClipRecord:
    """Synthesized clip metadata + audio payload."""

    clip_id: str
    text: str
    normalized_text: str
    lang_tag: str
    params: dict[str, Any]
    preset_name: str
    samples: np.ndarray
    wav_bytes: bytes
    duration_s: float
    rtf: float
    f0_median_hz: float
    n_zh: int
    n_en: int
    version: int = 1
    saved_path: str | None = None
    created_at: str = ""
    detected_emotions: list[str] = field(default_factory=list)
    spectral_preview: dict[str, Any] = field(default_factory=dict)

    def to_meta_dict(self) -> dict[str, Any]:
        """Return JSON-serializable metadata (excluding raw audio buffers)."""
        return {
            "clip_id": self.clip_id,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "lang_tag": self.lang_tag,
            "params": self.params,
            "preset_name": self.preset_name,
            "duration_s": round(self.duration_s, 3),
            "rtf": round(self.rtf, 4),
            "f0_median_hz": round(self.f0_median_hz, 1),
            "n_zh": self.n_zh,
            "n_en": self.n_en,
            "version": self.version,
            "audio_url": f"/api/audio/{self.clip_id}.wav?v={self.version}",
            "saved_path": self.saved_path,
            "created_at": self.created_at,
            "detected_emotions": self.detected_emotions,
            "spectral_preview": self.spectral_preview,
        }


class LarynxStudioEngine:
    """Thread-safe PyBot-Larynx Stage 4 synthesis engine with in-memory clip management."""

    def __init__(
        self,
        *,
        max_clips: int = 500,
        default_save_dir: Path | str | None = None,
        clone_dir: Path | str | None = None,
    ) -> None:
        self.max_clips = max(20, int(max_clips))
        self.default_save_dir = Path(default_save_dir) if default_save_dir else DEFAULT_SAVE_DIR
        self._clone_dir = Path(clone_dir) if clone_dir else DEFAULT_CLONE_DIR

        self._load_lock = threading.Lock()
        self._synth_lock = threading.Lock()
        self._clips_lock = threading.Lock()
        self._clone_lock = threading.Lock()
        self._clips: OrderedDict[str, ClipRecord] = OrderedDict()
        self._clone_profiles: OrderedDict[str, VoiceCloneProfile] = OrderedDict()

        self._ready = False
        self._loading = False
        self._loading_error: str | None = None
        self._kb: AcousticKnowledgeStage2 | AcousticKnowledgeStage5 | AcousticKnowledgeStage7 | None = None
        self._coh_phasor: np.ndarray | None = None
        self._arbiter: Arbiter | None = None
        self._all_rules: dict[str, Any] = {}
        self._base_profile: VoiceCloneProfile | None = None

    def start_loading_async(self) -> None:
        """Start loading knowledge base and assets in a background thread."""
        with self._load_lock:
            if self._ready or self._loading:
                return
            self._loading = True

        def _worker() -> None:
            try:
                self.ensure_ready()
            except Exception:
                pass

        t = threading.Thread(target=_worker, name="pybot-larynx-loader", daemon=True)
        t.start()

    def ensure_ready(self) -> None:
        """Synchronously load Stage 7 (or Stage 5 / Stage 4 fallback) acoustic knowledge, phasor, and rules."""
        if self._ready:
            return
        with self._load_lock:
            if self._ready:
                return
            self._loading = True
            self._loading_error = None
            try:
                if STAGE7_KNOWLEDGE_PATH.exists():
                    kb: AcousticKnowledgeStage2 | AcousticKnowledgeStage5 | AcousticKnowledgeStage7 = AcousticKnowledgeStage7.load(STAGE7_KNOWLEDGE_PATH)
                    rules_list = default_stage5_rules()
                elif STAGE5_KNOWLEDGE_PATH.exists():
                    kb = AcousticKnowledgeStage5.load(STAGE5_KNOWLEDGE_PATH)
                    rules_list = default_stage5_rules()
                else:
                    kb = AcousticKnowledgeStage2.load()
                    rules_list = default_stage0_rules()
                assets = load_speaker_assets_v15()
                coh = np.asarray(assets["glottal_bank"].coherent_phasor, dtype=np.complex128)
                arb = Arbiter()
                rules_map = {getattr(r, "name", r.__class__.__name__): r for r in rules_list}
                get_speaker_donor_bank(kb)
                get_en_articulation_table()
                load_cmudict()
                get_en_stage10_knowledge()
                if PROSODY_EN_STAGE10_PATH.exists():
                    try:
                        from exp_20_pybot_larynx.crossling.en_prosody_stage10 import get_en_prosody_stage10_model

                        get_en_prosody_stage10_model()
                    except Exception:
                        pass

                # Warm up the full ZH+EN frontend (jieba/lexicons) + acoustic + larynx pipeline once
                warm_vp = VoiceParams()
                warm_ctrl, _ = build_utterance_control(
                    "今天测试一下电子声带和 Hello World，二零二六年一切顺利。",
                    "warmup",
                    kb,
                    arb,
                    rules_list,
                    warm_vp,
                )
                warm_mod = apply_voice_params(warm_ctrl, warm_vp, warm_vp.seed)
                if isinstance(kb, AcousticKnowledgeStage7):
                    synthesize_larynx_v2(
                        warm_mod,
                        glottal_mode="dynamic_coherent",
                        coherent_phasor=coh,
                        uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7,
                        seed=warm_vp.seed,
                    )
                else:
                    synthesize_larynx(
                        warm_mod,
                        glottal_mode="parametric_lf",
                        coherent_phasor=coh,
                        seed=warm_vp.seed,
                    )

                self._kb = kb
                self._coh_phasor = coh
                self._arbiter = arb
                self._all_rules = rules_map
                base_prof = load_base_speaker_profile()
                self._base_profile = base_prof
                self._load_saved_clone_profiles()
                self._ready = True
            except Exception as exc:
                self._loading_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self._loading = False

    def _load_saved_clone_profiles(self) -> None:
        """Load persisted JSON voice clone profiles from disk (best-effort)."""
        if not self._clone_dir.exists():
            return
        loaded: list[VoiceCloneProfile] = []
        for path in sorted(self._clone_dir.glob("*.json")):
            try:
                raw = path.read_text(encoding="utf-8")
                prof = VoiceCloneProfile.from_json(raw)
                loaded.append(prof)
            except Exception:
                continue
        with self._clone_lock:
            for prof in loaded:
                self._clone_profiles[prof.profile_id] = prof

    def list_clone_profiles(self) -> list[dict[str, Any]]:
        """Return summary dicts for all loaded voice clone profiles."""
        with self._clone_lock:
            return [p.summary_dict() for p in self._clone_profiles.values()]

    def get_clone_profile(self, profile_id: str | None) -> VoiceCloneProfile | None:
        """Fetch a loaded voice clone profile by ID."""
        if not profile_id:
            return None
        with self._clone_lock:
            return self._clone_profiles.get(str(profile_id))

    def extract_and_save_clone_profile(
        self,
        name: str,
        wav_bytes_list: list[bytes],
        *,
        save_to_disk: bool = True,
        profile_id: str | None = None,
    ) -> VoiceCloneProfile:
        """Extract a voice clone profile from WAV payloads and optionally persist to JSON."""
        self.ensure_ready()
        if not wav_bytes_list:
            raise ValueError("请提供至少一段参考 WAV 音频。")
        if profile_id is not None:
            pid_clean = str(profile_id).strip()
            if not _SAFE_PROFILE_ID_RE.match(pid_clean):
                raise ValueError(f"非法的 profile_id: {profile_id}")
            profile_id = pid_clean

        waves: list[np.ndarray] = []
        orig_srs: list[int] = []
        for blob in wav_bytes_list:
            w, info = decode_wav_bytes_to_24k_mono(blob)
            waves.append(w)
            orig_srs.append(int(info.get("orig_sr", SR)))

        clean_name = str(name or "自定义音色").strip() or "自定义音色"
        prof = extract_clone_profile(
            waves,
            name=clean_name,
            profile_id=profile_id,
            orig_sr_hint=min(orig_srs) if orig_srs else SR,
        )

        if save_to_disk:
            if not _SAFE_PROFILE_ID_RE.match(prof.profile_id):
                raise ValueError(f"生成的 profile_id 不安全: {prof.profile_id}")
            self._clone_dir.mkdir(parents=True, exist_ok=True)
            out_path = (self._clone_dir / f"{prof.profile_id}.json").resolve()
            if out_path.parent != self._clone_dir.resolve():
                raise ValueError("非法的保存路径")
            out_path.write_text(prof.to_json(), encoding="utf-8")

        with self._clone_lock:
            self._clone_profiles[prof.profile_id] = prof
            self._clone_profiles.move_to_end(prof.profile_id)

        return prof

    def delete_clone_profile(self, profile_id: str) -> bool:
        """Delete a voice clone profile from memory and disk."""
        pid = str(profile_id or "").strip()
        if not pid:
            return False
        with self._clone_lock:
            removed = self._clone_profiles.pop(pid, None) is not None
        if _SAFE_PROFILE_ID_RE.match(pid) and self._clone_dir.exists():
            cand = (self._clone_dir / f"{pid}.json").resolve()
            if cand.parent == self._clone_dir.resolve() and cand.exists():
                cand.unlink()
                removed = True
        return removed

    def status_dict(self) -> dict[str, Any]:
        """Return engine readiness, presets, knob schema, emotion & synth metadata."""
        return {
            "ready": self._ready,
            "loading": self._loading,
            "loading_error": self._loading_error,
            "engine_stage": "stage8" if isinstance(self._kb, AcousticKnowledgeStage7) else ("stage5" if isinstance(self._kb, AcousticKnowledgeStage5) else "stage4"),
            "crossling_engine": "stage9_donor_transfer",
            "en_articulation_loaded": bool(get_en_articulation_table() is not None),
            "en_stage10_loaded": bool(is_stage10_en_ready()),
            "default_save_dir": str(self.default_save_dir),
            "presets": VOICE_PRESETS,
            "knobs": KNOB_SCHEMA,
            "synth_knobs": SYNTH_KNOB_SCHEMA,
            "timbre_patches": TIMBRE_PATCHES,
            "emotions": EMOTION_METADATA,
            "inline_tags": INLINE_TAG_CHIPS,
            "rules": RULE_METADATA,
            "clone_profiles": self.list_clone_profiles(),
            "clip_count": len(self._clips),
        }

    def synthesize_segment(
        self,
        text: str,
        params: VoiceParams | dict[str, Any] | None = None,
        *,
        clip_id: str | None = None,
    ) -> ClipRecord:
        """Synthesize one text segment and store/update it in the in-memory clip store."""
        self.ensure_ready()
        assert self._kb is not None and self._arbiter is not None

        vp = params if isinstance(params, VoiceParams) else VoiceParams.from_dict(params)
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("合成文本不能为空")

        tgt_prof: VoiceCloneProfile | None = None
        if vp.clone_profile_id:
            tgt_prof = self.get_clone_profile(vp.clone_profile_id)
            if tgt_prof is None:
                raise ValueError(f"找不到音色模仿档案: {vp.clone_profile_id}")

        vp_for_ctrl = vp
        if tgt_prof is not None and vp.clone_rate and self._base_profile is not None:
            rate_ratio = float(
                np.clip(
                    tgt_prof.syllable_rate_hz / max(1.0, self._base_profile.syllable_rate_hz),
                    0.80,
                    1.25,
                )
            )
            eff_speed = float(np.clip(vp.speed * rate_ratio, 0.60, 1.60))
            vp_for_ctrl = replace(vp, speed=eff_speed)

        active_rules = [
            self._all_rules[name]
            for name in RULE_NAMES
            if vp.rules_enabled.get(name, True) and name in self._all_rules
        ]

        cid = str(clip_id).strip() if clip_id else uuid.uuid4().hex[:12]
        t0 = time.perf_counter()

        with self._synth_lock:
            base_ctrl, meta = build_utterance_control(
                clean_text,
                cid,
                self._kb,
                self._arbiter,
                active_rules,
                vp_for_ctrl,
            )
            eff_phasor = self._coh_phasor
            if tgt_prof is not None and self._base_profile is not None:
                cloned_ctrl, eff_phasor, _ = apply_clone_to_control(
                    base_ctrl,
                    self._base_profile,
                    tgt_prof,
                    strength=vp.imitation_strength,
                    clone_f0=vp.clone_f0,
                    clone_glottal=True,
                    base_coherent_phasor=self._coh_phasor,
                )
            else:
                cloned_ctrl = base_ctrl
            mod_ctrl = apply_voice_params(cloned_ctrl, vp, vp.seed)
            synth_params = TimbreSynthParams.from_dict(vp.to_dict())
            mod_ctrl, synth_info = apply_timbre_synth(mod_ctrl, synth_params, seed=vp.seed)
            spectral_preview = extract_spectral_profile_preview(mod_ctrl, max_frames=16)

            hoarse_fry = min(0.15, 0.18 * max(0.0, float(vp.hoarseness) - 0.15)) if float(vp.hoarseness) > 0.15 else 0.0
            subh = float(
                np.clip(
                    max(
                        hoarse_fry,
                        float(meta.get("subharm_strength", 0.0)),
                        float(synth_info.get("subharm_strength", 0.0)),
                    ),
                    0.0,
                    0.30,
                )
            )
            if isinstance(self._kb, AcousticKnowledgeStage7):
                eff_gmode = "dynamic_coherent" if vp.glottal_mode == "speaker_coherent" else vp.glottal_mode
                wav_raw = synthesize_larynx_v2(
                    mod_ctrl,
                    glottal_mode=eff_gmode,
                    coherent_phasor=eff_phasor,
                    subharm_strength=subh,
                    uv_level_boost_db=DEFAULT_UV_LEVEL_BOOST_DB_S7,
                    seed=vp.seed,
                )
            else:
                wav_raw = synthesize_larynx(
                    mod_ctrl,
                    glottal_mode=vp.glottal_mode,
                    coherent_phasor=eff_phasor,
                    seed=vp.seed,
                )

            soften_mixed_onset_levels(wav_raw, mod_ctrl, meta["level_onsets"])

        elapsed = time.perf_counter() - t0
        wav_norm = normalize_audio(wav_raw, target_dbfs=vp.target_dbfs, max_peak=0.95)
        wav_bytes = encode_wav_bytes(wav_norm, sr=SR)
        dur_s = len(wav_norm) / float(SR)
        rtf = elapsed / max(1e-6, dur_s)

        v_mask = mod_ctrl.voiced & (mod_ctrl.f0_hz > 40.0)
        f0_med = float(np.median(mod_ctrl.f0_hz[v_mask])) if np.any(v_mask) else 0.0

        if tgt_prof is not None:
            preset_name = f"模仿:{tgt_prof.name}"
        else:
            preset_info = PRESET_MAP.get(vp.preset_id)
            preset_name = preset_info["name"] if preset_info else vp.preset_id

        with self._clips_lock:
            old = self._clips.get(cid)
            version = (old.version + 1) if old is not None else 1
            rec = ClipRecord(
                clip_id=cid,
                text=clean_text,
                normalized_text=str(meta.get("normalized_text", clean_text)),
                lang_tag=detect_lang_tag(clean_text),
                params=vp.to_dict(),
                preset_name=preset_name,
                samples=wav_norm,
                wav_bytes=wav_bytes,
                duration_s=dur_s,
                rtf=rtf,
                f0_median_hz=f0_med,
                n_zh=int(meta.get("n_zh", 0)),
                n_en=int(meta.get("n_en", 0)),
                version=version,
                saved_path=None,
                created_at=datetime.now().strftime("%H:%M:%S"),
                detected_emotions=list(meta.get("detected_emotions", ["neutral"])),
                spectral_preview=spectral_preview,
            )
            if cid in self._clips:
                self._clips.move_to_end(cid)
            self._clips[cid] = rec
            while len(self._clips) > self.max_clips:
                self._clips.popitem(last=False)

        return rec

    def get_clip(self, clip_id: str) -> ClipRecord | None:
        with self._clips_lock:
            return self._clips.get(clip_id)

    def delete_clip(self, clip_id: str) -> bool:
        with self._clips_lock:
            return self._clips.pop(clip_id, None) is not None

    def clear_clips(self) -> int:
        with self._clips_lock:
            n = len(self._clips)
            self._clips.clear()
            return n

    def save_clips(
        self,
        clip_ids: list[str],
        save_dir: str | Path,
        *,
        merge: bool = False,
        merge_gap_ms: float = 280.0,
        merge_name: str = "",
    ) -> dict[str, Any]:
        """Save selected clips individually and/or merged into one WAV file."""
        if not clip_ids:
            raise ValueError("请先勾选至少一条要保存的音频")

        out_dir = Path(save_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        records: list[ClipRecord] = []
        with self._clips_lock:
            for cid in clip_ids:
                rec = self._clips.get(cid)
                if rec is None:
                    raise KeyError(f"音频片段不存在或已过期: {cid}")
                records.append(rec)

        saved_files: list[dict[str, str]] = []
        if not merge:
            for idx, rec in enumerate(records, start=1):
                stem = f"{idx:03d}_{sanitize_filename_stem(rec.text)}"
                target = _unique_wav_path(out_dir, stem)
                target.write_bytes(rec.wav_bytes)
                rec.saved_path = str(target)
                saved_files.append({"clip_id": rec.clip_id, "path": str(target)})
            return {"save_dir": str(out_dir), "saved_files": saved_files, "merged_path": None}

        gap_n = max(0, int(round(max(0.0, float(merge_gap_ms)) * SR / 1000.0)))
        gap_arr = np.zeros(gap_n, dtype=np.float64)
        chunks: list[np.ndarray] = []
        for idx, rec in enumerate(records):
            chunks.append(rec.samples)
            if idx < len(records) - 1 and gap_n > 0:
                chunks.append(gap_arr)
        merged_audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)
        merged_bytes = encode_wav_bytes(merged_audio, sr=SR)

        raw_name = str(merge_name or "").strip()
        if raw_name:
            if raw_name.lower().endswith(".wav"):
                raw_name = raw_name[:-4]
            base_stem = sanitize_filename_stem(raw_name, max_len=48)
        else:
            base_stem = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        merged_path = _unique_wav_path(out_dir, base_stem)
        merged_path.write_bytes(merged_bytes)
        return {
            "save_dir": str(out_dir),
            "saved_files": [{"clip_id": "merged", "path": str(merged_path)}],
            "merged_path": str(merged_path),
        }
