"""Lazy-Feature Contiguous-Chunk Renderer (`BZNSYP` 9,726 utts, half-T3 clamp, 0.0 MB neural)."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v8_clarity.utau_concat import OBSTRUENT_ONSETS, _resample_1d, _resample_2d  # noqa: E402
from exp_v9_clarity_pro.boundary_refiner import compute_landmark_tracks  # noqa: E402
from exp_v10_integrity.gain_calibration import parseval_frame_db  # noqa: E402
from exp_v11_diphone_tts.diphone_bank import _normalize_shared_band  # noqa: E402
from exp_v12_natural_vocoder.v12_bank import _patch_unsmoothed_uv_env  # noqa: E402
from exp_v12_natural_vocoder.vocoder_v12 import analyze_v12, synthesize_v12  # noqa: E402
from exp_v13_utau_ceiling.sandhi_v13 import plan_v13  # noqa: E402
from exp_v13_utau_ceiling.v13_deform import (  # noqa: E402
    _pingpong_indices,
    _sample_by_float_indices_1d,
    _sample_by_float_indices_2d,
    calibrate_gain_gentle,
    find_steady_vowel_nucleus,
    join_seam_v13,
    smooth_utterance_gain_gentle,
)
from exp_v14_fluent_zh_tts.prosody_v14 import ProsodyModelV14, apply_prosody_v14, _normalize_of  # noqa: E402
from exp_v14_fluent_zh_tts.renderer_v14 import _interpolate_f0_hz_onto_mask, _template_contour_for_unit  # noqa: E402
from exp_v15_bznsyp_tts import HOP, N_FFT, PROSODY_MODEL_V15, SR  # noqa: E402
from exp_v15_bznsyp_tts.index_bznsyp_v15 import (  # noqa: E402
    V15BzToken,
    read_bznsyp_wav_24k,
)
from exp_v15_bznsyp_tts.prosody_v15 import ProsodyModelV15, apply_prosody_v15  # noqa: E402
from exp_v15_bznsyp_tts.selector_v15 import select_bznsyp_sequence  # noqa: E402


@dataclass
class V15Config:
    """Configuration & single-factor ablation flags for BZNSYP v15 synthesis."""

    prosody_mode: str = "v15_hgbr"     # "v15_hgbr" or "v14_hgbr"
    ngram_preseed: bool = True
    hf_clarity_guard: bool = True
    contiguous_chunk_extract: bool = True
    preserve_natural_f0: bool = True
    t3_register_clamp: bool = True
    t3_nonfinal_ceiling_st: float = -0.80
    damped_dur_alpha: float = 0.65
    top_k: int = 48
    preseed_k: int = 24
    chain_bonus: float = -3.00
    cache_max_utts: int = 256
    pause_floor_ms: float = 0.0


class LazyUttFeatureCache:
    """On-demand LRU cache running full `analyze_v12` ONLY for utterances actually selected."""

    def __init__(
        self,
        utt_metas: dict[str, dict[str, Any]],
        *,
        max_utts: int = 256,
    ) -> None:
        self.utt_metas = utt_metas
        self.max_utts = max(8, int(max_utts))
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.disk_reads = 0

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, uid: str) -> bool:
        return uid in self._cache

    def get(self, uid: str) -> dict[str, Any]:
        if uid in self._cache:
            self._cache.move_to_end(uid)
            return self._cache[uid]
        meta = self.utt_metas[uid]
        x24 = read_bznsyp_wav_24k(REPO_ROOT / meta["wav_path"])
        fv12 = analyze_v12(x24, sr=SR)
        feats_p = _patch_unsmoothed_uv_env(x24, fv12.feats, sr=SR, n_fft=N_FFT)
        env_norm, wav_scale, f0_rel_st = _normalize_shared_band(feats_p, sr=SR)
        xa = x24 * float(wav_scale)
        landmarks = compute_landmark_tracks(xa, feats_p, sr=SR)

        entry = {
            "id": uid,
            "xa": xa,
            "env_norm": env_norm.astype(np.float32),
            "bap": feats_p.bap.astype(np.float32),
            "bap_gt8k": fv12.bap_gt8k.astype(np.float32),
            "voiced": np.asarray(feats_p.voiced, dtype=bool),
            "f0_hz": np.asarray(feats_p.f0_hz, dtype=np.float32),
            "f0_rel_st": np.asarray(f0_rel_st, dtype=np.float32),
            "hf_lf": np.asarray(landmarks.get("hf_lf", np.zeros(len(env_norm))), dtype=np.float32),
            "n_frames": len(env_norm),
            "wav_scale": float(wav_scale),
        }
        self._cache[uid] = entry
        self.disk_reads += 1
        while len(self._cache) > self.max_utts:
            self._cache.popitem(last=False)
        return entry


def group_contiguous_chunks_v15(
    chosen: list[tuple[str, V15BzToken | None]],
    unit_pause_before: list[bool],
    *,
    contiguous_chunk_extract: bool = True,
) -> list[list[int]]:
    """Group consecutive tokens recorded adjacently (same uid, syl_idx+1, no pause) into unbroken chunks."""
    n = len(chosen)
    if n == 0:
        return []
    groups: list[list[int]] = [[0]]
    for i in range(1, n):
        mode_prev, tk_prev = chosen[i - 1]
        mode_cur, tk_cur = chosen[i]
        can_chain = (
            contiguous_chunk_extract
            and tk_prev is not None
            and tk_cur is not None
            and mode_prev == "zh_target_syl"
            and mode_cur == "zh_target_syl"
            and not unit_pause_before[i]
            and tk_cur.uid == tk_prev.uid
            and tk_cur.syl_idx == tk_prev.syl_idx + 1
            and tk_prev.pause_after_ms < 25.0
        )
        if can_chain:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _smooth_interp_f0_st(f0_rel: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """Fill unvoiced gaps smoothly with linear interpolation across voiced frames."""
    n = len(f0_rel)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    v_idx = np.where(voiced)[0]
    if v_idx.size == 0:
        return np.zeros(n, dtype=np.float64)
    if v_idx.size == 1:
        return np.full(n, float(f0_rel[v_idx[0]]), dtype=np.float64)
    return np.interp(np.arange(n, dtype=np.float64), v_idx.astype(np.float64), f0_rel[v_idx].astype(np.float64))


def _fallback_unit_features(
    fb_bank: Any,
    raw_o: str,
    raw_f: str,
    norm_o: str,
    norm_f: str,
    tone: int,
    n_target: int,
    f0_3pt: list[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize frame tracks for an out-of-subset syllable from `fallback_unit_bank`."""
    n_target = max(4, int(n_target))
    tmpl = fb_bank.syllables.get((raw_o, raw_f)) or fb_bank.syllables.get((norm_o, norm_f))
    if tmpl is not None:
        env = _resample_2d(tmpl.log_env.astype(np.float64), n_target)
        bap = np.clip(_resample_2d(tmpl.bap.astype(np.float64), n_target), 0.005, 0.995)
        voi = _resample_1d(tmpl.voiced_ratio.astype(np.float64), n_target) >= 0.45
    else:
        tm_o = fb_bank.onsets.get(raw_o) or fb_bank.onsets.get(norm_o)
        tm_f = fb_bank.finals.get(raw_f) or fb_bank.finals.get(norm_f) or fb_bank.finals.get("a")
        if tm_o is not None and tm_f is not None:
            n_o = max(2, min(n_target - 3, int(round(0.32 * n_target))))
            n_f = n_target - n_o
            env = np.concatenate(
                [_resample_2d(tm_o.log_env.astype(np.float64), n_o), _resample_2d(tm_f.log_env.astype(np.float64), n_f)],
                axis=0,
            )
            bap = np.clip(
                np.concatenate(
                    [_resample_2d(tm_o.bap.astype(np.float64), n_o), _resample_2d(tm_f.bap.astype(np.float64), n_f)],
                    axis=0,
                ),
                0.005,
                0.995,
            )
            voi = np.concatenate(
                [
                    _resample_1d(tm_o.voiced_ratio.astype(np.float64), n_o) >= 0.45,
                    _resample_1d(tm_f.voiced_ratio.astype(np.float64), n_f) >= 0.45,
                ]
            )
        elif tm_f is not None:
            env = _resample_2d(tm_f.log_env.astype(np.float64), n_target)
            bap = np.clip(_resample_2d(tm_f.bap.astype(np.float64), n_target), 0.005, 0.995)
            voi = _resample_1d(tm_f.voiced_ratio.astype(np.float64), n_target) >= 0.45
        else:
            base_env = fb_bank.canonical_profile.voiced_env_mean.astype(np.float64)
            base_bap = fb_bank.canonical_profile.bap_voiced_median.astype(np.float64)
            env = np.repeat(base_env[None, :], n_target, axis=0)
            bap = np.repeat(np.clip(base_bap[None, :], 0.01, 0.95), n_target, axis=0)
            voi = np.ones(n_target, dtype=bool)

    if norm_o in OBSTRUENT_ONSETS:
        uv_cut = min(n_target - 2, max(1, int(round(0.28 * n_target))))
        voi[:uv_cut] = False
    bgt8_col = np.clip(np.mean(bap[:, -1:], axis=1, keepdims=True) * 1.15, 0.04, 0.99)
    bgt8 = np.repeat(bgt8_col, 2, axis=1)
    f0_tmpl = _template_contour_for_unit({"tone": tone, "f0_3pt_st": f0_3pt}, n_target, 1.0, 0.0)
    return env, bap, bgt8, voi, f0_tmpl


def _piecewise_pos_for_syllable(
    src_s: int,
    src_split: int,
    src_e: int,
    n_tgt: int,
    voiced_slice: np.ndarray,
    env_slice: np.ndarray,
    onset: str,
    *,
    allow_pingpong: bool = True,
) -> np.ndarray:
    """Build monotonic float source indices within `[src_s, src_e)` respecting onset & vowel nucleus."""
    n_src = max(1, src_e - src_s)
    n_tgt = max(3, int(n_tgt))
    if n_src == 1:
        return np.full(n_tgt, float(src_s), dtype=np.float64)
    if abs(n_tgt - n_src) <= 1:
        return np.linspace(float(src_s), float(src_e - 1), n_tgt)

    f_sp = max(0, min(n_src - 2, int(src_split)))
    if f_sp > 0:
        if onset in OBSTRUENT_ONSETS:
            n_on = int(np.clip(round(f_sp * min(1.30, max(0.70, n_tgt / float(n_src)))), 1, max(1, n_tgt - 2)))
        else:
            n_on = max(1, min(n_tgt - 2, int(round(f_sp * n_tgt / float(n_src)))))
        pos_on = np.linspace(float(src_s), float(src_s + max(0, f_sp - 1)), n_on) if n_on > 0 else np.empty(0)
        start_ri = float(src_s + f_sp)
    else:
        n_on = 0
        pos_on = np.empty(0, dtype=np.float64)
        start_ri = float(src_s)

    n_ri_tgt = n_tgt - n_on
    n_ri_src = src_e - int(start_ri)
    if n_ri_src <= 2:
        pos_ri = np.linspace(start_ri, float(src_e - 1), n_ri_tgt)
    else:
        ratio = n_ri_tgt / float(n_ri_src)
        if allow_pingpong and ratio > 1.75:
            nuc_s, nuc_e = find_steady_vowel_nucleus(env_slice[f_sp:], voiced_slice[f_sp:])
            n_pre = nuc_s
            n_nuc_src = nuc_e - nuc_s
            n_post = n_ri_src - nuc_e
            n_nuc_tgt = n_ri_tgt - n_pre - n_post
            if n_nuc_src >= 4 and n_nuc_tgt >= n_nuc_src + 2:
                pos_pre = np.arange(start_ri, start_ri + n_pre, dtype=np.float64) if n_pre > 0 else np.empty(0)
                pp = start_ri + n_pre + _pingpong_indices(n_nuc_src, n_nuc_tgt)
                pos_post = np.arange(start_ri + nuc_e, src_e, dtype=np.float64) if n_post > 0 else np.empty(0)
                pos_ri = np.concatenate([pos_pre, pp, pos_post])[:n_ri_tgt]
                if len(pos_ri) < n_ri_tgt:
                    pos_ri = np.pad(pos_ri, (0, n_ri_tgt - len(pos_ri)), mode="edge")
            else:
                pos_ri = np.linspace(start_ri, float(src_e - 1), n_ri_tgt)
        else:
            pos_ri = np.linspace(start_ri, float(src_e - 1), n_ri_tgt)

    pos = np.concatenate([pos_on, pos_ri]) if n_on > 0 else pos_ri
    return np.clip(pos, float(src_s), float(src_e - 1))


def _target_f0_segment(up: dict[str, Any], n_tgt: int, base_f0_hz: float) -> np.ndarray:
    """Compute desired F0 contour (in semitones vs `base_f0_hz`) for a syllable slot."""
    ghz = up.get("guided_f0_hz")
    if ghz is not None and len(ghz) >= 2:
        arr = np.asarray(ghz, dtype=np.float64)
        v_mask = arr > 50.0
        if np.any(v_mask):
            idx_v = np.where(v_mask)[0].astype(np.float64)
            st_v = 12.0 * np.log2(arr[v_mask] / max(50.0, base_f0_hz))
            dense = np.interp(np.arange(len(arr), dtype=np.float64), idx_v, st_v)
            return np.clip(_resample_1d(dense, n_tgt), -10.0, 10.0)
    return _template_contour_for_unit({"tone": up["tone"], "f0_3pt_st": up["f0_3pt_st"]}, n_tgt, 1.0, 0.0)


def _extract_unit_plans(
    text: str,
    *,
    guided_units: list[dict[str, Any]] | None,
    prosody_mode: str,
    prosody_model_v15: ProsodyModelV15 | None,
    prosody_model_v14: ProsodyModelV14 | None,
    t3_register_clamp: bool,
    t3_nonfinal_ceiling_st: float,
    pause_floor_ms: float,
) -> list[dict[str, Any]]:
    """Build ordered Mandarin syllable targets from either `guided_units` or `plan_v13 + HGBR`."""
    plans: list[dict[str, Any]] = []
    if guided_units is not None:
        raw_list = guided_units
    else:
        ir = plan_v13(text)
        if prosody_mode == "v14_hgbr":
            pro_ir = apply_prosody_v14(ir, prosody_model_v14)
        else:
            pro_ir = apply_prosody_v15(ir, prosody_model_v15)
        raw_list = [u for u in (pro_ir.get("units") or []) if u.get("lang", "zh") == "zh"]

    n_u = len(raw_list)
    for i, u in enumerate(raw_list):
        raw_o = str(u.get("onset", ""))
        raw_f = str(u.get("final", ""))
        norm_o, norm_f = _normalize_of(raw_o, raw_f)
        tone = int(u.get("tone") or 1)
        dur_ms = float(u.get("duration_ms", u.get("dur_ms", 180.0)))
        f3 = u.get("f0_3pt_st")
        if f3 is not None and len(f3) == 3:
            f3_clean = [float(f3[0]), float(f3[1]), float(f3[2])]
        else:
            tmpl = _template_contour_for_unit({"tone": tone, "f0_3pt_st": None}, 9, 1.0, 0.0)
            f3_clean = [float(tmpl[1]), float(tmpl[4]), float(tmpl[7])]

        is_nf_t3 = (i < n_u - 1) and (tone == 3)
        if t3_register_clamp and is_nf_t3 and guided_units is None:
            ceil = float(t3_nonfinal_ceiling_st) - 0.35
            f3_clean = [min(f3_clean[0], ceil + 0.3), min(f3_clean[1], ceil), min(f3_clean[2], ceil)]

        p_after = max(pause_floor_ms, float(u.get("pause_after_ms", 0.0)))
        prev_p = plans[-1]["pause_after_ms"] if plans else 0.0
        pos_s = "initial" if i == 0 else ("final" if i == n_u - 1 else "medial")
        plans.append({
            "hanzi": str(u.get("hanzi") or u.get("token") or ""),
            "raw_onset": raw_o,
            "raw_final": raw_f,
            "onset": norm_o,
            "final": norm_f,
            "tone": tone,
            "dur_ms": dur_ms,
            "f0_3pt_st": f3_clean,
            "f0_mid_st": f3_clean[1],
            "gain_db": float(u.get("gain_db", 0.0)),
            "prosodic_break": int(u.get("prosodic_break", u.get("break", 0))),
            "pause_after_ms": p_after,
            "pause_before": bool(i > 0 and prev_p >= 25.0),
            "pos": pos_s,
            "guided_f0_hz": u.get("guided_f0_hz"),
        })
    return plans


def render_v15_utterance(
    text: str,
    index_doc: dict[str, Any],
    assets: dict[str, Any],
    *,
    prosody_model_v15: ProsodyModelV15 | None = None,
    prosody_model_v14: ProsodyModelV14 | None = None,
    cfg: V15Config | None = None,
    cache: LazyUttFeatureCache | None = None,
    allowed_utt_ids: set[str] | frozenset[str] | None = None,
    guided_units: list[dict[str, Any]] | None = None,
    exact_guided_timing: bool = False,
    target_f0_hz: float | None = None,
    pitch_offsets_st: dict[int, tuple[float, float, float]] | None = None,
    seed: int = 123,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render fluent Mandarin speech from BZNSYP 9,726-utt index + lazy v12 cache."""
    if cfg is None:
        cfg = V15Config()
    if cache is None:
        cache = LazyUttFeatureCache(index_doc["train_utt_metas"], max_utts=cfg.cache_max_utts)
    if prosody_model_v15 is None and guided_units is None and cfg.prosody_mode == "v15_hgbr":
        if PROSODY_MODEL_V15.exists():
            prosody_model_v15 = ProsodyModelV15.load(PROSODY_MODEL_V15)

    unit_plans = _extract_unit_plans(
        text,
        guided_units=guided_units,
        prosody_mode=cfg.prosody_mode,
        prosody_model_v15=prosody_model_v15,
        prosody_model_v14=prosody_model_v14,
        t3_register_clamp=cfg.t3_register_clamp,
        t3_nonfinal_ceiling_st=cfg.t3_nonfinal_ceiling_st,
        pause_floor_ms=cfg.pause_floor_ms,
    )
    n_u = len(unit_plans)
    if n_u == 0:
        return np.zeros(int(0.25 * SR), dtype=np.float64), {
            "n_syllables": 0, "chain_rate": 0.0, "avg_chunk_len": 0.0,
            "exact_tone_hit_rate": 0.0, "fallback_rate": 0.0, "t3_nonfinal_medians_st": [],
        }

    chosen = select_bznsyp_sequence(
        [(u["onset"], u["final"], u["tone"]) for u in unit_plans],
        [u["dur_ms"] for u in unit_plans],
        [u["pause_before"] for u in unit_plans],
        [tuple(u["f0_3pt_st"]) for u in unit_plans],
        [u["f0_mid_st"] for u in unit_plans],
        [u["pos"] for u in unit_plans],
        [u["prosodic_break"] for u in unit_plans],
        index_doc,
        allowed_utt_ids=allowed_utt_ids,
        use_ngram_preseed=cfg.ngram_preseed,
        use_hf_clarity_guard=cfg.hf_clarity_guard,
        top_k=cfg.top_k,
        preseed_k=cfg.preseed_k,
        chain_bonus=cfg.chain_bonus,
    )

    unit_pause_before = [u["pause_before"] for u in unit_plans]
    groups = group_contiguous_chunks_v15(
        chosen, unit_pause_before, contiguous_chunk_extract=cfg.contiguous_chunk_extract
    )

    prof_v2 = assets["profile_v2"]
    gbank = assets["glottal_bank"]
    fb_bank = assets["fallback_unit_bank"]
    base_f0_hz = float(target_f0_hz) if target_f0_hz is not None else float(prof_v2.f0_median_hz)
    # Match BZNSYP natural studio voiced p75 level (-22.5 dB Parseval) and clean studio noise floor (-9.0 neper)
    ref_vowel_db = -22.5
    bg_env = np.clip(prof_v2.voiced_env_mean - 9.0, -13.0, -7.2)
    bg_bap = np.full(5, 0.92, dtype=np.float64)

    # Render each chunk / fallback block cleanly
    chunk_blocks: list[dict[str, Any]] = []
    unit_tgt_durs_ms = [0.0] * n_u
    n_tone_hits = 0
    n_fallbacks = 0
    for _, tk in chosen:
        if tk is None:
            n_fallbacks += 1
    for i, (_, tk) in enumerate(chosen):
        if tk is not None and tk.tone == unit_plans[i]["tone"]:
            n_tone_hits += 1

    for g in groups:
        if len(g) == 1 and chosen[g[0]][1] is None:
            idx = g[0]
            up = unit_plans[idx]
            dur_ms = max(30.0, float(up["dur_ms"]))
            unit_tgt_durs_ms[idx] = dur_ms
            n_tgt = max(4, int(round(dur_ms * SR / (1000.0 * HOP))))
            env_g, bap_g, bgt8_g, voi_g, f0_g = _fallback_unit_features(
                fb_bank,
                up["raw_onset"],
                up["raw_final"],
                up["onset"],
                up["final"],
                up["tone"],
                n_tgt,
                up["f0_3pt_st"],
            )
            env_g, _ = calibrate_gain_gentle(env_g, "zh_syllable", ref_vowel_db)
            env_g = env_g + float(np.clip(0.35 * up["gain_db"], -2.0, 2.0)) * math.log(10.0) / 20.0
            chunk_blocks.append({
                "g": g,
                "syl_spans": [(0, n_tgt, idx)],
                "env": env_g,
                "bap": bap_g,
                "bgt8": bgt8_g,
                "voi": voi_g,
                "f0_st": f0_g,
                "is_obs_first": up["onset"] in OBSTRUENT_ONSETS,
                "pause_after_ms": up["pause_after_ms"],
                "uv_spans": [],
            })
            continue

        # Contiguous chunk from a single BZNSYP training utterance
        first_tk = chosen[g[0]][1]
        assert first_tk is not None
        ue = cache.get(first_tk.uid)
        n_g = len(g)
        src_pos_parts: list[np.ndarray] = []
        syl_spans: list[tuple[int, int, int]] = []
        f0_parts: list[np.ndarray] = []
        uv_spans_local: list[tuple[int, int, np.ndarray]] = []
        fr_cursor = 0

        for j, ui in enumerate(g):
            tk = chosen[ui][1]
            assert tk is not None
            up = unit_plans[ui]
            nat_ms = max(25.0, float(tk.dur_ms))
            if exact_guided_timing:
                dur_ms = max(25.0, float(up["dur_ms"]))
            else:
                alpha = 0.55 if n_g >= 2 else cfg.damped_dur_alpha
                ratio_lo, ratio_hi = (0.65, 1.55) if n_g >= 2 else (0.55, 1.75)
                blended = alpha * float(up["dur_ms"]) + (1.0 - alpha) * nat_ms
                dur_ms = float(np.clip(blended, nat_ms * ratio_lo, nat_ms * ratio_hi))
            unit_tgt_durs_ms[ui] = dur_ms
            n_tgt = max(4, int(round(dur_ms * SR / (1000.0 * HOP))))

            src_s = int(tk.start_fr)
            src_e = int(chosen[g[j + 1]][1].start_fr) if (j + 1 < n_g and chosen[g[j + 1]][1] is not None) else int(tk.end_fr)
            src_e = max(src_s + 2, min(ue["n_frames"], max(src_e, int(tk.end_fr))))
            sl_pos = _piecewise_pos_for_syllable(
                src_s,
                int(tk.split_fr),
                src_e,
                n_tgt,
                ue["voiced"][src_s:src_e],
                ue["env_norm"][src_s:src_e],
                up["onset"],
                allow_pingpong=(not exact_guided_timing),
            )
            src_pos_parts.append(sl_pos)
            syl_spans.append((fr_cursor, fr_cursor + n_tgt, ui))

            # Collect crisp unvoiced residual from source obstruent onset
            if up["onset"] in OBSTRUENT_ONSETS and int(tk.split_fr) >= 2:
                uv_len_tgt = max(2, min(n_tgt - 2, int(round(int(tk.split_fr) * n_tgt / float(max(1, src_e - src_s))))))
                s0_sam = max(0, src_s * HOP)
                s1_sam = min(len(ue["xa"]), (src_s + int(tk.split_fr)) * HOP)
                if s1_sam - s0_sam >= 72:
                    uv_spans_local.append((fr_cursor, fr_cursor + uv_len_tgt, ue["xa"][s0_sam:s1_sam].copy()))

            # Construct syllable F0 contour
            tgt_c = _target_f0_segment(up, n_tgt, base_f0_hz)
            nat_seg = _sample_by_float_indices_1d(ue["f0_rel_st"], sl_pos)
            voi_seg = _sample_by_float_indices_1d(ue["voiced"].astype(np.float64), sl_pos) >= 0.45
            nat_sm = _smooth_interp_f0_st(nat_seg, voi_seg)
            if exact_guided_timing:
                w_tgt = 1.0 if up.get("guided_f0_hz") is not None else 0.88
                f0_seg = w_tgt * tgt_c + (1.0 - w_tgt) * nat_sm
            elif not cfg.preserve_natural_f0:
                f0_seg = tgt_c
            elif n_g >= 2 and tk.tone == up["tone"]:
                shift_m = float(np.clip(0.45 * (float(np.median(tgt_c)) - float(np.median(nat_sm))), -2.2, 2.2))
                f0_seg = nat_sm + shift_m
            elif tk.tone == up["tone"]:
                tgt_m = float(np.median(tgt_c))
                nat_m = float(np.median(nat_sm))
                f0_seg = 0.55 * tgt_c + 0.45 * (nat_sm - nat_m + tgt_m)
            else:
                nat_m = float(np.median(nat_sm))
                f0_seg = 0.88 * tgt_c + 0.12 * (nat_sm - nat_m)

            if cfg.t3_register_clamp and (ui < n_u - 1) and (up["tone"] == 3) and (not exact_guided_timing):
                f0_seg = np.minimum(f0_seg, cfg.t3_nonfinal_ceiling_st - 0.15)
            f0_parts.append(f0_seg)
            fr_cursor += n_tgt

        pos_chunk = np.concatenate(src_pos_parts)
        env_g = _sample_by_float_indices_2d(ue["env_norm"], pos_chunk)
        bap_g = np.clip(_sample_by_float_indices_2d(ue["bap"], pos_chunk), 0.005, 0.995)
        bgt8_g = np.clip(_sample_by_float_indices_2d(ue["bap_gt8k"], pos_chunk), 0.005, 0.995)
        voi_g = _sample_by_float_indices_1d(ue["voiced"].astype(np.float64), pos_chunk) >= 0.45
        hf_lf_g = _sample_by_float_indices_1d(ue["hf_lf"], pos_chunk)
        f0_g = np.concatenate(f0_parts)

        # Clean obstruent onset unvoicing inside chunk
        for fr_s, fr_e, ui in syl_spans:
            if unit_plans[ui]["onset"] in OBSTRUENT_ONSETS:
                tk_u = chosen[ui][1]
                assert tk_u is not None
                n_on_max = min(fr_e - fr_s - 2, max(1, int(round(tk_u.split_fr * 1.15))))
                for k in range(fr_s, fr_s + max(0, n_on_max)):
                    if hf_lf_g[k] > -4.0 or not voi_g[k]:
                        voi_g[k] = False

        env_g, _ = calibrate_gain_gentle(env_g, "zh_syllable", ref_vowel_db)
        if not exact_guided_timing:
            mean_g = float(np.mean([unit_plans[ui]["gain_db"] for ui in g]))
            env_g = env_g + float(np.clip(0.35 * mean_g, -2.0, 2.0)) * math.log(10.0) / 20.0

        # Smooth internal chunk F0 transitions across concatenated syllables inside chunk
        if len(f0_g) >= 5 and cfg.preserve_natural_f0 and not exact_guided_timing:
            for fr_s, _, _ in syl_spans[1:]:
                lo = max(0, fr_s - 2)
                hi = min(len(f0_g), fr_s + 2)
                if hi - lo >= 3:
                    sm = f0_g[lo:hi].copy()
                    sm[1:-1] = 0.25 * f0_g[lo : hi - 2] + 0.50 * f0_g[lo + 1 : hi - 1] + 0.25 * f0_g[lo + 2 : hi]
                    f0_g[lo:hi] = sm

        chunk_blocks.append({
            "g": g,
            "syl_spans": syl_spans,
            "env": env_g,
            "bap": bap_g,
            "bgt8": bgt8_g,
            "voi": voi_g,
            "f0_st": f0_g,
            "is_obs_first": unit_plans[g[0]]["onset"] in OBSTRUENT_ONSETS,
            "pause_after_ms": unit_plans[g[-1]]["pause_after_ms"],
            "uv_spans": uv_spans_local,
        })

    # Apply inter-chunk cosine seam smoothing (NEVER touch interior of contiguous chunks!)
    for bi in range(1, len(chunk_blocks)):
        prev_b = chunk_blocks[bi - 1]
        cur_b = chunk_blocks[bi]
        if prev_b["pause_after_ms"] < 25.0:
            join_seam_v13(
                prev_b["env"],
                cur_b["env"],
                prev_b["bap"],
                cur_b["bap"],
                prev_b["voi"],
                cur_b["voi"],
                obs_cur=bool(cur_b["is_obs_first"]),
            )

    # Assemble full utterance timeline + trailing/inter-clause pauses
    env_parts: list[np.ndarray] = []
    bap_parts: list[np.ndarray] = []
    bgt8_parts: list[np.ndarray] = []
    voi_parts: list[np.ndarray] = []
    f0_parts_all: list[np.ndarray] = []
    obs_masks: list[np.ndarray] = []
    pause_masks: list[np.ndarray] = []
    global_syl_spans: list[tuple[int, int, int]] = []
    residual_spans: list[tuple[int, int, np.ndarray]] = []
    cur_fr = 0

    bgt8_dim = next((b["bgt8"].shape[1] for b in chunk_blocks), 2)

    def _append_sil(n_p: int) -> None:
        nonlocal cur_fr
        if n_p <= 0:
            return
        env_parts.append(np.repeat(bg_env[None, :], n_p, axis=0))
        bap_parts.append(np.repeat(bg_bap[None, :], n_p, axis=0))
        bgt8_parts.append(np.full((n_p, bgt8_dim), 0.92, dtype=np.float64))
        voi_parts.append(np.zeros(n_p, dtype=bool))
        f0_parts_all.append(np.zeros(n_p, dtype=np.float64))
        obs_masks.append(np.zeros(n_p, dtype=bool))
        pause_masks.append(np.ones(n_p, dtype=bool))
        cur_fr += n_p

    lead_sil = 4 if exact_guided_timing else 6
    _append_sil(lead_sil)

    for bi, blk in enumerate(chunk_blocks):
        nf = len(blk["env"])
        env_parts.append(blk["env"])
        bap_parts.append(blk["bap"])
        bgt8_parts.append(blk["bgt8"])
        voi_parts.append(blk["voi"])
        f0_parts_all.append(blk["f0_st"])
        om = np.zeros(nf, dtype=bool)
        for s_l, e_l, ui in blk["syl_spans"]:
            global_syl_spans.append((cur_fr + s_l, cur_fr + e_l, ui))
            if unit_plans[ui]["onset"] in OBSTRUENT_ONSETS:
                om[s_l : min(e_l, s_l + max(2, (e_l - s_l) // 3))] = True
        obs_masks.append(om)
        pause_masks.append(np.zeros(nf, dtype=bool))
        for r0, r1, rw in blk["uv_spans"]:
            residual_spans.append((cur_fr + r0, cur_fr + r1, rw))
        cur_fr += nf

        p_ms = float(blk["pause_after_ms"])
        if p_ms >= 20.0 and bi < len(chunk_blocks) - 1:
            n_p = max(4, int(round(p_ms * SR / (1000.0 * HOP))))
            _append_sil(n_p)

    _append_sil(6 if exact_guided_timing else 8)

    env_all = np.concatenate(env_parts, axis=0)
    bap_all = np.concatenate(bap_parts, axis=0)
    bgt8_all = np.concatenate(bgt8_parts, axis=0)
    voi_all = np.concatenate(voi_parts, axis=0)
    f0_all_st = np.concatenate(f0_parts_all, axis=0)
    obs_all = np.concatenate(obs_masks, axis=0)
    pause_all = np.concatenate(pause_masks, axis=0)

    # Utterance-level centering (auto mode only) + post-centering non-final half-T3 clamp!
    if np.any(voi_all):
        if not exact_guided_timing:
            f0_all_st = f0_all_st - float(np.median(f0_all_st[voi_all]))
        if cfg.t3_register_clamp:
            for _ in range(2):
                utt_med = float(np.median(f0_all_st[voi_all]))
                ceil_abs = utt_med + float(cfg.t3_nonfinal_ceiling_st) - 0.05
                for f_s, f_e, ui in global_syl_spans:
                    if ui < n_u - 1 and int(unit_plans[ui]["tone"]) == 3:
                        v_sl = voi_all[f_s:f_e]
                        if np.any(v_sl):
                            med_sl = float(np.median(f0_all_st[f_s:f_e][v_sl]))
                            if med_sl > utt_med + float(cfg.t3_nonfinal_ceiling_st) or np.max(f0_all_st[f_s:f_e][v_sl]) > ceil_abs:
                                f0_all_st[f_s:f_e] = np.minimum(f0_all_st[f_s:f_e], ceil_abs)
        if not exact_guided_timing:
            f0_all_st[voi_all] = np.clip(f0_all_st[voi_all], -6.5, 7.5)

    f0_hz_all = np.zeros(len(f0_all_st), dtype=np.float64)
    if np.any(voi_all):
        if pitch_offsets_st:
            for start, end, ui in global_syl_spans:
                offsets = pitch_offsets_st.get(ui)
                voiced_indices = np.flatnonzero(voi_all[start:end]) + start
                if offsets is not None and voiced_indices.size:
                    phases = np.linspace(0.0, 1.0, len(voiced_indices)) if voiced_indices.size > 1 else [0.5]
                    f0_all_st[voiced_indices] += np.interp(phases, (0.0, 0.5, 1.0), offsets)
        f0_hz_all[voi_all] = np.clip(base_f0_hz * (2.0 ** (f0_all_st[voi_all] / 12.0)), 65.0, 440.0)
        f0_hz_all = _interpolate_f0_hz_onto_mask(f0_hz_all, voi_all, base_f0_hz)

    # Record actual final non-final T3 medians relative to utterance F0 median
    t3_nonfinal_medians: list[float] = []
    if np.any(voi_all & (f0_hz_all > 50.0)):
        utt_hz_med = float(np.median(f0_hz_all[voi_all & (f0_hz_all > 50.0)]))
        for f_s, f_e, ui in global_syl_spans:
            if ui < n_u - 1 and int(unit_plans[ui]["tone"]) == 3:
                vm = voi_all[f_s:f_e] & (f0_hz_all[f_s:f_e] > 50.0)
                if np.any(vm):
                    rel_st = 12.0 * np.log2(f0_hz_all[f_s:f_e][vm] / utt_hz_med)
                    t3_nonfinal_medians.append(round(float(np.median(rel_st)), 3))

    env_all = smooth_utterance_gain_gentle(env_all, voi_all, obs_all, pause_all, ref_vowel_db=ref_vowel_db)
    feats_out = vocoder.VocoderFeatures(
        f0_hz=f0_hz_all,
        voiced=voi_all,
        log_env=env_all,
        bap=bap_all,
        sr=SR,
        hop_samples=HOP,
        n_samples=len(env_all) * HOP,
        n_fft=N_FFT,
    )
    wav_out = synthesize_v12(
        feats_out,
        mode="hybrid_lp_hnm",
        glottal_bank=gbank,
        crisp_unvoiced=True,
        seed=seed,
        phase_dispersion=True,
        bap_gt8k=bgt8_all,
        unvoiced_residual_spans=residual_spans,
    )

    chunk_lens = [len(g) for g in groups if all(chosen[ui][1] is not None for ui in g)]
    n_chained = sum(max(0, len(g) - 1) for g in groups if all(chosen[ui][1] is not None for ui in g))
    chain_rate = float(n_chained) / float(max(1, n_u - 1))
    avg_chunk_len = float(np.mean(chunk_lens)) if chunk_lens else 0.0

    info = {
        "n_syllables": n_u,
        "n_chained": n_chained,
        "chain_rate": round(chain_rate, 4),
        "avg_chunk_len": round(avg_chunk_len, 3),
        "n_chunks": len(chunk_lens),
        "exact_tone_hit_rate": round(n_tone_hits / float(max(1, n_u)), 4),
        "fallback_rate": round(n_fallbacks / float(max(1, n_u)), 4),
        "t3_nonfinal_medians_st": t3_nonfinal_medians,
        "selected_uids": [chosen[i][1].uid if chosen[i][1] is not None else "FALLBACK" for i in range(n_u)],
        "target_durs_ms": [round(x, 1) for x in unit_tgt_durs_ms],
        "target_pauses_ms": [round(float(u["pause_after_ms"]), 1) for u in unit_plans],
        "slot_timings_ms": [(start * HOP * 1000 / SR, end * HOP * 1000 / SR)
                            for start, end, _ in global_syl_spans],
        "parsed_units": [(u["onset"], u["final"], u["tone"]) for u in unit_plans],
        "cache_hits_utts": len(cache),
    }
    return wav_out, info
