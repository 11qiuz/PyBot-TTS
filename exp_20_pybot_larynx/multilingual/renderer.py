"""Stage 13 Native Zero-Mapping Multilingual Renderer (`zh` + `en` + pluggable languages).

Core design:
  - Pluggable `LanguageFrontend` modules (`zh`, `en`, and user-registered `ja`, `ko`, ...)
    emit unified `PhoneEvent` + F0 + voiced trajectories into ONE shared utterance timeline.
  - One cross-lingual Viterbi DP (`build_multilingual_native_control`) selects native tokens
    directly from the speaker's unified multilingual IPA bank (`knowledge_sf_native_stage13.npz`)
    with ZERO cross-speaker projection or multi-bank spectral patching.
  - One universal 4-rule IPA physical stack (`apply_multilingual_native_rules`) shapes glottal
    anti-buzz excitation, nasal zeros + VN coarticulation, obstruent crispness, and cep[0:18]
    intra/cross-lingual seam coarticulation.
  - Optional single-exit `apply_clone_to_control` when targeting an external voice (`BZNSYP`).
"""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, Sequence
import zlib

import numpy as np
from scipy.signal import lfilter

from exp_20_pybot_larynx import HOP, N_BAP, N_BINS, SR
from exp_20_pybot_larynx.acoustics.knowledge import normalize_of
from exp_20_pybot_larynx.arbiter import (
    OBSTRUENT_MANNERS,
    classify_onset_manner,
)
from exp_20_pybot_larynx.clone import (
    VoiceCloneProfile,
    apply_clone_to_control,
    load_base_speaker_profile,
)
from exp_20_pybot_larynx.crossling.en_prosody_stage10 import (
    PROSODY_EN_STAGE10_PATH,
    ProsodyEnStage10Model,
    get_en_prosody_stage10_model,
    predict_en_prosody_stage10,
)
from exp_20_pybot_larynx.crossling.en_stage10_renderer import (
    SF_ZH_PROFILE_PATH,
    _compute_expressive_drive,
    load_sf_bilingual_consistency,
)
from exp_20_pybot_larynx.expressive import (
    EMOTION_PROFILES,
    ExpressiveSegment,
    apply_emotion_to_control,
    apply_emotion_to_zh_prosody,
    build_breath_control,
    build_laugh_control,
    build_pause_control,
    build_sigh_control,
    parse_expressive_text,
)
from exp_20_pybot_larynx.frontend.prosody_rules import (
    _CF0_VOICED_ST,
    _CF0_VOICELESS_ST,
    _F0_AR1_SIGMA,
    _catmull_rom_5pt,
    _catmull_rom_7pt,
    _safe_plan_v13,
)
from exp_20_pybot_larynx.frontend.prosody_stage7 import predict_prosody_stage7
from exp_20_pybot_larynx.larynx.synth import LarynxControl
from exp_20_pybot_larynx.larynx.synth_v2 import synthesize_larynx_v2
from exp_20_pybot_larynx.multilingual.knowledge_ipa_native import (
    MultilingualIPAKnowledge,
    SF_NATIVE_KB_PATH,
    build_multilingual_native_control,
    get_sf_native_multilingual_kb,
)
from exp_20_pybot_larynx.multilingual.native_rules import apply_multilingual_native_rules
from exp_20_pybot_larynx.multilingual.schema import (
    PhoneEvent,
    get_language_frontend,
    lookup_ipa_info,
    normalize_prosody_tag,
    register_language_frontend,
)
from exp_v5_arch import crossling_frontend as cf

_PUNCT_AFTER_RE = re.compile(r"^[，,、。\.！!？\?；;：:…—\"'\u201d\u2019\u300d\u300f\uff09\u3011\]) ]+")
_LANG_TAG_RE = re.compile(r"\[lang:([a-zA-Z]{2,4})\](.*?)\[/lang\]", re.DOTALL)
_GLIDES_ZH = frozenset({"j", "w", "ɥ"})
_NASALS_ZH = frozenset({"n", "ŋ"})
_SF_PHASOR_CACHE: np.ndarray | None = None
_SF_PHASOR_LOADED: bool = False


def get_sf_coherent_phasor() -> np.ndarray | None:
    """Load SF speaker harmonic glottal phase spectrum once."""
    global _SF_PHASOR_CACHE, _SF_PHASOR_LOADED
    if not _SF_PHASOR_LOADED:
        _SF_PHASOR_LOADED = True
        if SF_ZH_PROFILE_PATH.is_file():
            prof = VoiceCloneProfile.from_json(SF_ZH_PROFILE_PATH.read_text(encoding="utf-8"))
            _SF_PHASOR_CACHE = prof.reconstruct_coherent_phasor()
    return _SF_PHASOR_CACHE


def _silence_control(n_frames: int) -> LarynxControl:
    n_fr = max(0, int(n_frames))
    return LarynxControl(
        f0_hz=np.zeros(n_fr, dtype=np.float64),
        voiced=np.zeros(n_fr, dtype=bool),
        log_env=np.full((n_fr, N_BINS), -16.0, dtype=np.float64),
        bap=np.full((n_fr, N_BAP), 0.99, dtype=np.float64),
        glottal_tilt_db_per_oct=np.zeros(n_fr, dtype=np.float64),
        jitter_frac=np.full(n_fr, 0.0055, dtype=np.float64),
        shimmer_frac=np.full(n_fr, 0.025, dtype=np.float64),
        n_samples=n_fr * HOP,
        rd=np.ones(n_fr, dtype=np.float64),
    )


def _extract_zh_subtext(norm_text: str, run_units: list[dict[str, Any]]) -> str:
    spans = [u.get("text_span") for u in run_units if isinstance(u.get("text_span"), list) and len(u["text_span"]) == 2]
    if spans and len(spans) == len(run_units):
        s0 = max(0, int(spans[0][0]))
        s1 = min(len(norm_text), int(spans[-1][1]))
        m = _PUNCT_AFTER_RE.match(norm_text[s1:])
        extra = m.group(0).strip() if m else ""
        sub = (norm_text[s0:s1] + extra).strip()
        if sub:
            return sub
    return "".join(str(u.get("hanzi") or u.get("token") or "") for u in run_units)


def _split_final_frames(f_ipas: tuple[str, ...], n_fi: int) -> list[int]:
    """Allocate frame counts across 1, 2, or 3 IPA segments inside one Mandarin final."""
    if len(f_ipas) <= 1 or n_fi <= len(f_ipas):
        out = [1] * max(1, len(f_ipas))
        diff = n_fi - sum(out)
        if out:
            out[0] = max(1, out[0] + diff)
        return out

    if len(f_ipas) == 2:
        p0, p1 = f_ipas
        if p0 in _GLIDES_ZH:
            g = max(2, min(n_fi - 2, int(round(0.24 * n_fi))))
            return [g, max(1, n_fi - g)]
        if p1 in _NASALS_ZH:
            c = max(2, min(n_fi - 2, int(round(0.38 * n_fi))))
            return [max(1, n_fi - c), c]
        off = max(2, min(n_fi - 2, int(round(0.42 * n_fi))))
        return [max(1, n_fi - off), off]

    # 3-phone final: medial glide + nucleus vowel + nasal/offglide
    g = max(1, int(round(0.20 * n_fi)))
    c = max(1, int(round(0.32 * n_fi)))
    while g + c > n_fi - 2 and (g > 1 or c > 1):
        if c > g and c > 1:
            c -= 1
        elif g > 1:
            g -= 1
        else:
            break
    return [g, max(1, n_fi - g - c), c]


class MandarinFrontend:
    """Mandarin (`zh`) LanguageFrontend plugin producing IPA PhoneEvents."""

    lang: str = "zh"

    def plan_run(
        self,
        text_or_units: Any,
        *,
        base_f0_hz: float,
        speed: float = 1.0,
        pause_scale: float = 1.0,
        seed: int = 2026,
        clause_pos: tuple[float, float] = (0.0, 1.0),
        is_clause_final: bool = True,
        start_fr: int = 0,
        expressive_seg: ExpressiveSegment | None = None,
        stability: float = 0.55,
        continues_after: bool = False,
    ) -> tuple[list[PhoneEvent], np.ndarray, np.ndarray]:
        if isinstance(text_or_units, str):
            zh_text = text_or_units
        elif isinstance(text_or_units, tuple) and len(text_or_units) == 2:
            zh_text = _extract_zh_subtext(str(text_or_units[0]), list(text_or_units[1]))
        else:
            zh_text = "".join(str(u.get("hanzi") or u.get("token") or "") for u in text_or_units)

        class _F0Holder:
            def __init__(self, f0: float) -> None:
                self.speaker_f0_med_hz = f0

        pro = predict_prosody_stage7(zh_text, _F0Holder(base_f0_hz), style="auto", continues_after=continues_after)
        focus_lvls = [0] * len(pro["units"])
        if expressive_seg is not None:
            pro, focus_lvls = apply_emotion_to_zh_prosody(pro, expressive_seg, stability=stability, seed=seed)

        raw_units = pro["units"]
        n_u = len(raw_units)
        eff_speed = max(0.60, min(1.60, float(speed)))
        eff_pause = max(0.40, min(2.00, float(pause_scale)))

        events: list[PhoneEvent] = []
        f0_chunks: list[np.ndarray] = []
        voi_chunks: list[np.ndarray] = []
        cur_fr = int(start_fr)
        prev_voiced_end_hz: float | None = None
        prev_pause_ms = 0.0

        for idx, u in enumerate(raw_units):
            o, f = normalize_of(str(u.get("onset", "")), str(u.get("final", "")))
            tone = int(u.get("tone") or 1)
            dur_ms = max(60.0, float(u.get("duration_ms", 180.0)) / eff_speed)
            n_syl = max(3, int(round(dur_ms / 5.0)))
            o_ipas = cf.PINYIN_ONSET_TO_IPA.get(o, ()) if o else ()
            f_ipas = cf.PINYIN_FINAL_TO_IPA.get(f, ("ə",))

            if o_ipas:
                raw_on_ms = max(15.0, float(u.get("onset_ms", 50.0)) / eff_speed)
                n_on = max(1, min(n_syl - len(f_ipas), int(round(raw_on_ms / 5.0))))
            else:
                n_on = 0

            n_fi = max(len(f_ipas), n_syl - n_on)
            n_syl = n_on + n_fi
            manner = classify_onset_manner(o) if o else "zero"

            u_fi = np.linspace(0.0, 1.0, n_fi) if n_fi > 1 else np.array([0.5], dtype=np.float64)
            if "f0_7pt_st" in u:
                fi_st = _catmull_rom_7pt(u_fi, np.asarray(u["f0_7pt_st"], dtype=np.float64))
            else:
                fi_st = _catmull_rom_5pt(u_fi, np.asarray(u["f0_5pt_st"], dtype=np.float64))

            if o and n_fi >= 3:
                k_cf0 = min(5, n_fi)
                decay = np.exp(-np.arange(k_cf0, dtype=np.float64) * 5.0 / 12.0)
                if manner in ("stop_asp", "affr_asp", "fric"):
                    fi_st[:k_cf0] += _CF0_VOICELESS_ST * decay
                elif manner in ("nasal", "liquid"):
                    fi_st[:k_cf0] += _CF0_VOICED_ST * decay
                elif manner in ("stop_unasp", "affr_unasp"):
                    fi_st[:k_cf0] += 0.25 * _CF0_VOICELESS_ST * decay

            fi_hz = np.clip(base_f0_hz * (2.0 ** (fi_st / 12.0)), 65.0, 440.0)
            f0_seg = np.empty(n_syl, dtype=np.float64)
            f0_seg[n_on:] = fi_hz
            if n_on > 0:
                if manner in ("nasal", "liquid") and prev_voiced_end_hz is not None:
                    f0_seg[:n_on] = np.linspace(0.65 * prev_voiced_end_hz + 0.35 * fi_hz[0], fi_hz[0], n_on, endpoint=False)
                else:
                    f0_seg[:n_on] = fi_hz[0]

            voi_seg = np.ones(n_syl, dtype=bool)
            if o and manner in OBSTRUENT_MANNERS and n_on > 0:
                uv_f = float(u.get("onset_uv_frac", 0.78))
                uv_n = max(1, min(n_on, int(round(uv_f * n_on))))
                voi_seg[:uv_n] = False
                f0_seg[:uv_n] = 0.0

            brk = int(u.get("prosodic_break", 0))
            t_uv = 0 if brk < 3 else (1 if brk == 3 else 2)
            if t_uv > 0 and n_fi >= 8:
                t_uv = min(t_uv, n_fi - 6)
                voi_seg[-t_uv:] = False
                f0_seg[-t_uv:] = 0.0

            if prev_voiced_end_hz is not None:
                if voi_seg[0] and n_on == 0 and f0_seg[0] > 50.0:
                    k_sm = min(4, n_syl)
                    w = np.linspace(0.30, 0.0, k_sm)
                    f0_seg[:k_sm] = (1.0 - w) * f0_seg[:k_sm] + w * prev_voiced_end_hz
                elif n_on < n_syl and manner in OBSTRUENT_MANNERS:
                    k_sm = min(3, n_syl - n_on)
                    w = np.linspace(0.10, 0.0, k_sm)
                    f0_seg[n_on : n_on + k_sm] = (1.0 - w) * f0_seg[n_on : n_on + k_sm] + w * prev_voiced_end_hz

            raw_p = float(u.get("pause_after_ms", 0.0))
            p_ms = (raw_p * eff_pause) if idx < n_u - 1 else 0.0
            gain_db = float(u.get("gain_db", 0.0)) + (1.2 if focus_lvls[idx] > 0 else 0.0)

            syl_phones: list[tuple[str, int, int, bool]] = []
            offset = 0
            if o_ipas:
                syl_phones.append((o_ipas[0], cur_fr, cur_fr + n_on, False))
                offset = n_on
            fi_alloc = _split_final_frames(f_ipas, n_fi)
            f_cursor = cur_fr + offset
            for fi_idx, (f_ipa, f_len) in enumerate(zip(f_ipas, fi_alloc)):
                is_v = f_ipa not in _GLIDES_ZH and f_ipa not in _NASALS_ZH
                syl_phones.append((f_ipa, f_cursor, f_cursor + f_len, is_v))
                f_cursor += f_len

            word_id = f"zh_{idx}_{o}_{f}"
            n_sp = len(syl_phones)
            for p_k, (ipa_s, sa, sb, is_v) in enumerate(syl_phones):
                wpos = 0 if n_sp == 1 else (1 if p_k == 0 else (3 if p_k == n_sp - 1 else 2))
                is_last_ph = p_k == n_sp - 1
                ptag = normalize_prosody_tag("zh", tone, ipa_s)
                events.append(
                    PhoneEvent(
                        ipa=ipa_s,
                        allophone=ipa_s,
                        lang="zh",
                        phone_key=f"zh:{ipa_s}",
                        prosody_tag=ptag,
                        start_fr=sa,
                        end_fr=sb,
                        word_id=word_id,
                        word_pos=wpos,
                        is_vowel=is_v,
                        is_coda=False,
                        is_func_word=False,
                        is_phrase_final=(is_last_ph and (brk >= 2 or p_ms >= 20.0 or (is_clause_final and idx == n_u - 1))),
                        is_pause=False,
                        pause_prev_ms=prev_pause_ms if p_k == 0 else 0.0,
                        pause_after_ms=p_ms if is_last_ph else 0.0,
                        same_word_prev=p_k > 0,
                        same_word_next=p_k + 1 < n_sp,
                        gain_db=gain_db,
                    )
                )

            f0_chunks.append(f0_seg)
            voi_chunks.append(voi_seg)
            cur_fr += n_syl

            if idx < n_u - 1 and p_ms >= 20.0:
                n_p = max(1, int(round(p_ms / 5.0)))
                events.append(
                    PhoneEvent(
                        ipa="sil", allophone="sil", lang="zh", phone_key="sil", prosody_tag=0,
                        start_fr=cur_fr, end_fr=cur_fr + n_p, is_pause=True,
                        pause_prev_ms=p_ms, pause_after_ms=p_ms,
                    )
                )
                f0_chunks.append(np.zeros(n_p, dtype=np.float64))
                voi_chunks.append(np.zeros(n_p, dtype=bool))
                cur_fr += n_p
                prev_voiced_end_hz = None
                prev_pause_ms = p_ms
            else:
                prev_voiced_end_hz = float(f0_seg[-1]) if voi_seg[-1] and f0_seg[-1] > 50.0 else None
                prev_pause_ms = 0.0

        f0_all = np.concatenate(f0_chunks) if f0_chunks else np.zeros(0, dtype=np.float64)
        voi_all = np.concatenate(voi_chunks) if voi_chunks else np.zeros(0, dtype=bool)
        if _F0_AR1_SIGMA > 0.0 and len(f0_all) > 4:
            rng_u = np.random.default_rng(zlib.crc32(f"s13_zh|{int(seed)}|{n_u}".encode("utf-8")) & 0xFFFFFFFF)
            rho = 0.72
            eps = rng_u.normal(0.0, _F0_AR1_SIGMA * np.sqrt(1.0 - rho * rho), size=len(f0_all))
            ar1 = lfilter((1.0,), (1.0, -rho), eps)
            vm = voi_all & (f0_all > 50.0)
            f0_all[vm] = np.clip(f0_all[vm] * (2.0 ** (np.clip(ar1[vm], -0.45, 0.45) / 12.0)), 65.0, 440.0)
        return events, f0_all, voi_all


class EnglishFrontend:
    """English (`en`) LanguageFrontend plugin producing IPA PhoneEvents via conversational prosody."""

    lang: str = "en"

    def __init__(self, prosody_model: ProsodyEnStage10Model | None = None) -> None:
        self._prosody_model = prosody_model

    def plan_run(
        self,
        text_or_units: Any,
        *,
        base_f0_hz: float,
        speed: float = 1.0,
        pause_scale: float = 1.0,
        seed: int = 2026,
        clause_pos: tuple[float, float] = (0.0, 1.0),
        is_clause_final: bool = True,
        start_fr: int = 0,
        is_question: bool | None = None,
        embedded_in_zh: bool = False,
        expressive_drive: float | None = None,
        range_scale: float = 1.0,
    ) -> tuple[list[PhoneEvent], np.ndarray, np.ndarray]:
        mdl = self._prosody_model or get_en_prosody_stage10_model(PROSODY_EN_STAGE10_PATH)
        # Anchor English F0 gently relative to speaker Mandarin baseline (~+2% natural EN lift)
        eff_base = float(np.clip(base_f0_hz * (1.02 if embedded_in_zh else 1.03), 65.0, 440.0))
        # Slightly calibrate embedded conversational EN rate naturally to fit SF speaking rhythm
        eff_rate = float(np.clip(speed * (1.00 if embedded_in_zh else 1.02), 0.60, 1.60))
        eff_range_scale = range_scale * (1.05 if not embedded_in_zh else 1.0)

        pro = predict_en_prosody_stage10(
            text_or_units,
            base_f0_hz=eff_base,
            rate=eff_rate,
            pause_scale=pause_scale,
            range_scale=eff_range_scale,
            clause_pos=clause_pos,
            is_clause_final=is_clause_final,
            is_question=is_question,
            include_pauses=True,
            seed=seed,
            model=mdl,
            embedded_in_zh=embedded_in_zh,
            expressive_drive=expressive_drive,
        )
        n_fr = int(pro["n_frames"])
        if n_fr <= 0:
            return [], np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool)

        f0_hz = np.asarray(pro["f0_hz"], dtype=np.float64).copy()
        voiced = f0_hz > 40.0
        phones = pro["phones"]
        spans = pro["phone_spans"]
        events: list[PhoneEvent] = []
        offset = int(start_fr)
        prev_pause_ms = 0.0

        for idx_s, sp in enumerate(spans):
            a = int(sp["start_fr"])
            b = int(sp["end_fr"])
            if sp.get("is_pause", False) or int(sp.get("phone_idx", -1)) < 0:
                events.append(
                    PhoneEvent(
                        ipa="sil", allophone="sil", lang="en", phone_key="sil", prosody_tag=0,
                        start_fr=offset + a, end_fr=offset + b, is_pause=True,
                        pause_prev_ms=(b - a) * 5.0, pause_after_ms=(b - a) * 5.0,
                    )
                )
                voiced[a:b] = False
                f0_hz[a:b] = 0.0
                prev_pause_ms = (b - a) * 5.0
                continue

            pi = int(sp["phone_idx"])
            ph = phones[pi]
            ipa = str(ph.get("ipa", "ə"))
            allo_raw = str(ph.get("allophone") or ipa)
            is_coda = allo_raw.endswith("_coda")
            info = lookup_ipa_info(ipa)
            if info.voiceless_obstruent:
                voiced[a:b] = False
                f0_hz[a:b] = 0.0

            n_w_ph = int(ph.get("word_n_phones", 1))
            if n_w_ph <= 1:
                wpos = 0
            elif ph.get("is_word_initial", False):
                wpos = 1
            elif ph.get("is_word_final", False):
                wpos = 3
            else:
                wpos = 2

            p_after = float(ph.get("pause_after_ms", 0.0))
            ptag = normalize_prosody_tag("en", ph.get("stress", 0), ipa)
            prev_same = pi > 0 and phones[pi - 1].get("word_idx") == ph.get("word_idx") and prev_pause_ms < 20.0
            next_same = pi + 1 < len(phones) and phones[pi + 1].get("word_idx") == ph.get("word_idx") and p_after < 20.0

            events.append(
                PhoneEvent(
                    ipa=ipa,
                    allophone=allo_raw,
                    lang="en",
                    phone_key=f"en:{allo_raw}",
                    prosody_tag=ptag,
                    start_fr=offset + a,
                    end_fr=offset + b,
                    word_id=f"en_{ph.get('word_idx', pi)}_{ph.get('word', '')}",
                    word_pos=wpos,
                    is_vowel=bool(ph.get("is_vowel", info.manner in ("vowel", "diphthong"))),
                    is_coda=is_coda,
                    is_func_word=bool(ph.get("is_func", False)),
                    is_phrase_final=bool(ph.get("is_phrase_final", False)),
                    is_pause=False,
                    pause_prev_ms=prev_pause_ms,
                    pause_after_ms=p_after,
                    same_word_prev=prev_same,
                    same_word_next=next_same,
                    gain_db=float(ph.get("gain_db", 0.0)),
                )
            )
            prev_pause_ms = 0.0

        return events, f0_hz, voiced


def register_default_frontends() -> None:
    """Ensure built-in `zh` and `en` frontends are registered without clobbering user overrides."""
    if get_language_frontend("zh") is None:
        register_language_frontend("zh", MandarinFrontend())
    if get_language_frontend("en") is None:
        register_language_frontend("en", EnglishFrontend())


register_default_frontends()


def _parse_language_runs(text: str, allow_english: bool = True) -> tuple[str, list[tuple[str, Any]]]:
    """Split a clean text clause into ordered `(lang, payload)` runs supporting `[lang:xx]...[/lang]` tags."""
    # Check explicit language plugin tags (enables registered ja/ko/... plugins directly from text!)
    tagged_matches = list(_LANG_TAG_RE.finditer(text))
    if tagged_matches:
        runs: list[tuple[str, Any]] = []
        cursor = 0
        for m in tagged_matches:
            if m.start() > cursor:
                prefix = text[cursor : m.start()].strip()
                if prefix:
                    _norm, sub_runs = _parse_language_runs(prefix, allow_english=allow_english)
                    runs.extend(sub_runs)
            lang_code = m.group(1).strip().lower()
            inner = m.group(2).strip()
            if inner:
                runs.append((lang_code, inner))
            cursor = m.end()
        if cursor < len(text):
            tail = text[cursor:].strip()
            if tail:
                _norm, sub_runs = _parse_language_runs(tail, allow_english=allow_english)
                runs.extend(sub_runs)
        return text, runs

    try:
        from pybot_webui.splitter import normalize_text_for_larynx
        norm_seg = normalize_text_for_larynx(text, allow_english=allow_english)
    except Exception:
        norm_seg = text.strip()

    if not norm_seg:
        return "", []

    ir = _safe_plan_v13(norm_seg)
    all_units = list(ir.get("units") or [])
    if not all_units:
        # Pure ASCII / fallback English if _safe_plan_v13 returned empty
        if allow_english and re.search(r"[A-Za-z]", norm_seg):
            return norm_seg, [("en", norm_seg)]
        return norm_seg, []

    runs: list[tuple[str, Any]] = []
    cur_lang: str | None = None
    cur_list: list[dict[str, Any]] = []
    for u in all_units:
        lang = "en" if u.get("lang") == "en" else "zh"
        if cur_lang is None or lang == cur_lang:
            cur_lang = lang
            cur_list.append(u)
        else:
            runs.append((cur_lang, (norm_seg, cur_list) if cur_lang == "zh" else cur_list))
            cur_lang = lang
            cur_list = [u]
    if cur_list and cur_lang is not None:
        runs.append((cur_lang, (norm_seg, cur_list) if cur_lang == "zh" else cur_list))
    return norm_seg, runs


def _concat_controls(chunks: list[tuple[LarynxControl, float]]) -> LarynxControl:
    parts: list[LarynxControl] = []
    prev_pause_ms = 999.0
    for idx, (ctrl, pause_ms) in enumerate(chunks):
        c = LarynxControl(
            f0_hz=ctrl.f0_hz.copy(),
            voiced=ctrl.voiced.copy(),
            log_env=ctrl.log_env.copy(),
            bap=ctrl.bap.copy(),
            glottal_tilt_db_per_oct=ctrl.glottal_tilt_db_per_oct.copy(),
            jitter_frac=ctrl.jitter_frac.copy(),
            shimmer_frac=ctrl.shimmer_frac.copy(),
            n_samples=len(ctrl.voiced) * HOP,
            rd=ctrl.rd.copy(),
        )
        n = len(c.voiced)
        if n > 0 and prev_pause_ms >= 20.0:
            fade = np.linspace(0.0, 1.0, min(4, n), endpoint=True)
            ramp = -8.0 * (1.0 - fade) / 8.685889638065035
            c.log_env[: len(fade)] = np.clip(c.log_env[: len(fade)] + ramp[:, None], -16.0, 2.5)
        if n > 0 and pause_ms >= 20.0:
            fade = np.linspace(1.0, 0.0, min(5, n), endpoint=True)
            ramp = -9.0 * (1.0 - fade) / 8.685889638065035
            c.log_env[-len(fade) :] = np.clip(c.log_env[-len(fade) :] + ramp[:, None], -16.0, 2.5)
        parts.append(c)
        if pause_ms >= 20.0:
            n_p = max(1, int(round(pause_ms / 5.0)))
            parts.append(_silence_control(n_p))
            prev_pause_ms = pause_ms
        else:
            prev_pause_ms = 0.0 if idx < len(chunks) - 1 else 999.0

    f0_hz = np.concatenate([p.f0_hz for p in parts])
    voiced = np.concatenate([p.voiced for p in parts])
    log_env = np.concatenate([p.log_env for p in parts], axis=0)
    bap = np.concatenate([p.bap for p in parts], axis=0)
    tilt = np.concatenate([p.glottal_tilt_db_per_oct for p in parts])
    jit = np.concatenate([p.jitter_frac for p in parts])
    shim = np.concatenate([p.shimmer_frac for p in parts])
    rd = np.concatenate([p.rd for p in parts])
    return LarynxControl(
        f0_hz=f0_hz,
        voiced=voiced,
        log_env=log_env,
        bap=bap,
        glottal_tilt_db_per_oct=tilt,
        jitter_frac=jit,
        shimmer_frac=shim,
        n_samples=len(f0_hz) * HOP,
        rd=rd,
    )


def _anchor_f0_seam(f0_hz: np.ndarray, voiced: np.ndarray, events: Sequence[PhoneEvent]) -> None:
    """Smooth pitch register across zero-pause cross-lingual switches."""
    n_fr = len(f0_hz)
    for i in range(len(events) - 1):
        e0, e1 = events[i], events[i + 1]
        if e0.is_pause or e1.is_pause or e0.lang == e1.lang:
            continue
        if float(e1.pause_prev_ms) >= 20.0 or int(e1.start_fr) - int(e0.end_fr) > 3:
            continue
        left_a = max(0, int(e0.end_fr) - 18)
        right_b = min(n_fr, int(e1.start_fr) + 32)
        lv = np.flatnonzero(voiced[left_a : int(e0.end_fr)] & (f0_hz[left_a : int(e0.end_fr)] > 40.0))
        rv = np.flatnonzero(voiced[int(e1.start_fr) : right_b] & (f0_hz[int(e1.start_fr) : right_b] > 40.0))
        if len(lv) < 3 or len(rv) < 3:
            continue
        ref = float(np.median(np.log2(f0_hz[left_a + lv[-min(10, len(lv)) :]])))
        ons = float(np.median(np.log2(f0_hz[int(e1.start_fr) + rv[: min(10, len(rv))]])))
        off_st = float(np.clip(0.75 * 12.0 * (ref - ons), -5.0, 5.0))
        start = int(int(e1.start_fr) + rv[0])
        span = min(32, n_fr - start)
        if span <= 1:
            continue
        el = np.arange(span, dtype=np.float64)
        w = 0.5 + 0.5 * np.cos(math.pi * np.maximum(0.0, el - 6.0) / 26.0)
        act = voiced[start : start + span] & (f0_hz[start : start + span] > 40.0)
        frames = start + np.flatnonzero(act)
        f0_hz[frames] = np.clip(f0_hz[frames] * np.exp2(off_st * w[act] / 12.0), 62.0, 460.0)


def _apply_voiced_en_switch_softness(ctrl: LarynxControl, events: Sequence[PhoneEvent], sf_cons: dict[str, Any] | None) -> None:
    """Carry tiny measured SF speaker-intrinsic EN/ZH LTAS nuance only when safe (<= 0.35 np)."""
    if not sf_cons:
        return
    dv = sf_cons.get("delta_voiced_ltas_en_minus_zh_nep")
    if dv is None:
        return
    dv_arr = np.clip(np.asarray(dv, dtype=np.float64), -0.35, 0.35)
    if dv_arr.shape != (N_BINS,):
        return
    n_fr = len(ctrl.f0_hz)
    has_zh = any((not e.is_pause) and e.lang == "zh" for e in events)
    has_en = any((not e.is_pause) and e.lang == "en" for e in events)
    if not (has_zh and has_en):
        return
    for e in events:
        if e.is_pause or e.lang != "en":
            continue
        a = max(0, min(n_fr, int(e.start_fr)))
        b = max(a, min(n_fr, int(e.end_fr)))
        vm = ctrl.voiced[a:b] & (ctrl.f0_hz[a:b] > 40.0)
        if np.any(vm):
            ctrl.log_env[a:b][vm] = np.clip(ctrl.log_env[a:b][vm] + 0.25 * dv_arr[None, :], -16.0, 2.5)


def build_multilingual_native_utterance(
    text: str,
    *,
    knowledge: MultilingualIPAKnowledge | None = None,
    kb_path: Path | str = SF_NATIVE_KB_PATH,
    runs_override: Sequence[tuple[str, Any]] | None = None,
    params: Any | None = None,
    speed: float = 1.0,
    pause_scale: float = 1.0,
    pitch_range: float = 1.0,
    emotion_mode: str = "auto",
    emotion_intensity: float = 0.75,
    stability: float = 0.55,
    auto_breath: bool = True,
    seed: int = 2026,
) -> tuple[LarynxControl, dict[str, Any]]:
    """Build one unified Stage 13 Native LarynxControl via a single cross-lingual Viterbi pass."""
    register_default_frontends()
    kb = knowledge or get_sf_native_multilingual_kb(kb_path)
    if kb is None:
        raise FileNotFoundError(f"Native multilingual KB not found at {kb_path}")

    if params is not None:
        speed = float(getattr(params, "speed", speed))
        pause_scale = float(getattr(params, "pause_scale", pause_scale))
        pitch_range = float(getattr(params, "pitch_range", pitch_range))
        emotion_mode = str(getattr(params, "emotion_mode", emotion_mode))
        emotion_intensity = float(getattr(params, "emotion_intensity", emotion_intensity))
        stability = float(getattr(params, "stability", stability))
        auto_breath = bool(getattr(params, "auto_breath", auto_breath))
        seed = int(getattr(params, "seed", seed))

    base_f0 = float(kb.speaker_f0_med_hz)
    p_scale = max(0.40, min(2.00, float(pause_scale)))
    sf_cons = load_sf_bilingual_consistency()
    if params is not None:
        vp_drive_proxy = params
    else:
        class _DriveProxy:
            def __init__(self, em: str, ei: float, st: float) -> None:
                self.emotion_mode = em
                self.emotion_intensity = ei
                self.stability = st
        vp_drive_proxy = _DriveProxy(emotion_mode, emotion_intensity, stability)

    # Preserve [lang:xx]...[/lang] blocks across parse_expressive_text (which tokenizes [brackets] individually)
    lang_placeholders: dict[str, str] = {}
    masked_text = text
    if runs_override is None and "[lang:" in masked_text:
        def _mask_lang(m: re.Match[str]) -> str:
            key = f"LANGTAG{len(lang_placeholders)}END"
            lang_placeholders[key] = m.group(0)
            return key
        masked_text = _LANG_TAG_RE.sub(_mask_lang, masked_text)

    # Direct pre-tokenized runs path (used for ja/ko plugins or unit tests)
    if runs_override is not None:
        segs = [ExpressiveSegment(kind="speech", text=text or "", emotion="neutral", intensity=0.0, pause_after_ms=0.0, auto_breath_before=False)]
    else:
        segs = parse_expressive_text(
            masked_text,
            emotion_mode=emotion_mode,
            emotion_intensity=emotion_intensity,
            stability=stability,
            auto_breath=auto_breath,
        )
        if lang_placeholders:
            for seg in segs:
                if seg.kind == "speech" and seg.text:
                    for k_tok, orig in lang_placeholders.items():
                        seg.text = seg.text.replace(k_tok, orig)
    if not segs:
        raise ValueError("Input text is empty.")

    chunks: list[tuple[LarynxControl, float]] = [(_silence_control(8), 0.0)]
    all_events_log: list[PhoneEvent] = []
    subharm_max = 0.0
    n_by_lang: dict[str, int] = {}
    fallback_stats: list[dict[str, Any]] = []
    rule_stats: list[dict[str, Any]] = []
    detected_emotions: list[str] = []

    for s_idx, seg in enumerate(segs):
        if seg.kind == "pause":
            chunks.append((build_pause_control(seg.duration_ms * p_scale), 0.0))
            continue
        if seg.kind in ("breath", "gasp"):
            chunks.append((build_breath_control(base_f0, duration_ms=seg.duration_ms, intensity=seg.intensity, is_gasp=(seg.kind == "gasp")), 0.0))
            continue
        if seg.kind == "sigh":
            chunks.append((build_sigh_control(base_f0, duration_ms=seg.duration_ms, intensity=seg.intensity), 0.0))
            continue
        if seg.kind in ("laugh", "chuckle"):
            chunks.append((build_laugh_control(base_f0, duration_ms=seg.duration_ms, intensity=seg.intensity, soft=(seg.kind == "chuckle")), 0.0))
            continue

        if seg.auto_breath_before:
            chunks.append((build_breath_control(base_f0, duration_ms=170.0, intensity=seg.intensity), 0.0))

        detected_emotions.append(seg.emotion)
        if runs_override is not None:
            runs = list(runs_override)
        else:
            norm_seg, runs = _parse_language_runs(seg.text, allow_english=True)
        if not runs:
            continue

        seg_pause = seg.pause_after_ms * p_scale if s_idx < len(segs) - 1 else 0.0
        has_zh = any(lang == "zh" for lang, _ in runs)
        seg_q = str(seg.text).rstrip().endswith(("?", "？"))
        e_drive = float(_compute_expressive_drive(seg, vp_drive_proxy, seg_q, seg.text or ""))
        seg_events: list[PhoneEvent] = []
        seg_f0_parts: list[np.ndarray] = []
        seg_voi_parts: list[np.ndarray] = []
        cur_fr = 0
        total_r = max(1, len(runs))

        for r_idx, (lang, payload) in enumerate(runs):
            fe = get_language_frontend(lang)
            if fe is None:
                raise KeyError(f"Language frontend {lang!r} is not registered.")
            is_last_r = r_idx == len(runs) - 1
            c_pos = (r_idx / float(total_r), (r_idx + 1) / float(total_r))

            extra_kw: dict[str, Any] = {}
            if lang == "zh":
                extra_kw["expressive_seg"] = seg if seg.intensity > 1e-4 else None
                extra_kw["stability"] = stability
                extra_kw["continues_after"] = not is_last_r
            elif lang == "en":
                extra_kw["is_question"] = (is_last_r and seg_q)
                extra_kw["embedded_in_zh"] = has_zh
                extra_kw["expressive_drive"] = e_drive
                extra_kw["range_scale"] = pitch_range

            r_evs, r_f0, r_voi = fe.plan_run(
                payload,
                base_f0_hz=base_f0,
                speed=speed,
                pause_scale=p_scale,
                seed=seed + s_idx * 31 + r_idx,
                clause_pos=c_pos,
                is_clause_final=is_last_r,
                start_fr=cur_fr,
                **extra_kw,
            )
            seg_events.extend(r_evs)
            seg_f0_parts.append(r_f0)
            seg_voi_parts.append(r_voi)
            cur_fr += len(r_f0)
            n_by_lang[lang] = n_by_lang.get(lang, 0) + sum(1 for e in r_evs if not e.is_pause)

        if not seg_f0_parts or cur_fr == 0:
            continue

        seg_f0 = np.concatenate(seg_f0_parts)
        seg_voi = np.concatenate(seg_voi_parts)
        _anchor_f0_seam(seg_f0, seg_voi, seg_events)

        ctrl, dp_meta = build_multilingual_native_control(seg_events, seg_f0, seg_voi, kb)
        _apply_voiced_en_switch_softness(ctrl, seg_events, sf_cons)
        ctrl, r_meta = apply_multilingual_native_rules(
            seg_events,
            ctrl,
            base_f0,
            expressive_drive=e_drive if seg.intensity > 1e-4 else 0.0,
        )

        if seg.intensity > 1e-4 and seg.emotion != "neutral":
            ctrl, subh = apply_emotion_to_control(ctrl, seg, stability=stability, seed=seed + s_idx)
            subharm_max = max(subharm_max, float(subh))

        all_events_log.extend(seg_events)
        fallback_stats.append(dp_meta)
        rule_stats.append(r_meta)
        chunks.append((ctrl, seg_pause))

    if len(chunks) <= 1:
        raise ValueError(f"Could not plan any speech events for {text!r}")

    chunks.append((_silence_control(14), 0.0))
    stitched = _concat_controls(chunks)
    return stitched, {
        "mode": "stage13_native_zero_mapping",
        "projections": 0,
        "speaker_f0_med_hz": base_f0,
        "n_phones_by_lang": n_by_lang,
        "n_events": len(all_events_log),
        "detected_emotions": list(dict.fromkeys(detected_emotions)) or ["neutral"],
        "subharm_strength": subharm_max,
        "dp_stats": fallback_stats,
        "rule_stats": rule_stats,
    }


def synthesize_multilingual_native(
    text: str,
    *,
    knowledge: MultilingualIPAKnowledge | None = None,
    kb_path: Path | str = SF_NATIVE_KB_PATH,
    runs_override: Sequence[tuple[str, Any]] | None = None,
    params: Any | None = None,
    target_profile: VoiceCloneProfile | None = None,
    target_coherent_phasor: np.ndarray | None = None,
    clone_strength: float = 0.90,
    seed: int = 2026,
    rms_target_db: float = -19.5,
) -> tuple[np.ndarray, LarynxControl, dict[str, Any]]:
    """Synthesize native bilingual/multilingual speech (0 projections by default; optional single-exit clone)."""
    kb = knowledge or get_sf_native_multilingual_kb(kb_path)
    if kb is None:
        raise FileNotFoundError(f"Native multilingual KB not found at {kb_path}")

    ctrl, meta = build_multilingual_native_utterance(
        text,
        knowledge=kb,
        runs_override=runs_override,
        params=params,
        seed=seed,
    )

    eff_phasor = get_sf_coherent_phasor()
    if target_profile is not None:
        # Optional single-exit cloning to an external speaker (e.g. BZNSYP)
        if SF_ZH_PROFILE_PATH.is_file():
            src_prof = VoiceCloneProfile.from_json(SF_ZH_PROFILE_PATH.read_text(encoding="utf-8"))
        else:
            src_prof = load_base_speaker_profile()
        ctrl, eff_c_ph, clone_meta = apply_clone_to_control(
            ctrl,
            src_prof,
            target_profile,
            strength=float(clone_strength),
            clone_f0=True,
            clone_glottal=True,
            base_coherent_phasor=eff_phasor,
        )
        eff_phasor = eff_c_ph if eff_c_ph is not None else target_coherent_phasor
        meta["projections"] = 1
        meta["clone"] = clone_meta

    if params is not None:
        try:
            from pybot_webui.engine import apply_voice_params
            ctrl = apply_voice_params(ctrl, params, seed)
        except Exception:
            pass

    y = synthesize_larynx_v2(
        ctrl,
        glottal_mode="dynamic_coherent",
        coherent_phasor=eff_phasor,
        uv_level_boost_db=0.35,
        subharm_strength=float(meta.get("subharm_strength", 0.0)),
        seed=seed,
    )
    arr = np.asarray(y, dtype=np.float64)
    rms = float(np.sqrt(np.mean(arr * arr) + 1e-18))
    if rms > 1e-7:
        arr = arr * ((10.0 ** (rms_target_db / 20.0)) / rms)
        peak = float(np.max(np.abs(arr)))
        if peak > 0.96:
            arr = arr * (0.96 / peak)
    return arr, ctrl, meta
