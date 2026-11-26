"""Build Unified Stage 13 Native Multilingual (`zh` + `en`) Acoustic KB from `SF_bilingual`."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import EXP_DIR, HOP, N_BAP, N_BINS, N_CEP, SR  # noqa: E402
from exp_20_pybot_larynx.acoustics.build_stage7_knowledge import _fit_ridge, _pack_protos, _pack_weights  # noqa: E402
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import MedoidGroup, _load_medoid_table  # noqa: E402
from exp_20_pybot_larynx.clone import VoiceCloneProfile, apply_clone_to_control  # noqa: E402
from exp_20_pybot_larynx.crossling.build_knowledge_en_stage10 import _build_medoid_table_en10, _filter_strict_indices  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env, log_env_to_cep  # noqa: E402
from exp_20_pybot_larynx.multilingual.knowledge_ipa_native import (  # noqa: E402
    D_ACOUSTIC_FEAT_M,
    MultilingualIPAKnowledge,
    N_BINS_M,
    N_COARSE_CEP_M,
    SF_NATIVE_KB_PATH,
    SF_NATIVE_SUMMARY_PATH,
    clear_sf_native_multilingual_cache,
    extract_multilingual_acoustic_features,
)
from exp_20_pybot_larynx.multilingual.schema import (  # noqa: E402
    classify_universal_art_class,
    lookup_ipa_info,
    normalize_prosody_tag,
    strip_allophone_suffix,
)
from exp_v6_vocoder import vocoder as v6_voc  # noqa: E402

SF_TRAIN_TOKENS_PATH = EXP_DIR / "crossling_data" / "sf_bilingual_train_tokens.pkl"
SF_EN_PROFILE_PATH = EXP_DIR / "crossling_data" / "sf_en_profile.json"
LJ_TRAIN_TOKENS_PATH = EXP_DIR / "crossling_data" / "en_stage10_train_tokens.pkl"
LJ_SOURCE_PROFILE_PATH = EXP_DIR / "crossling_data" / "en_source_profile.json"
_SONORANT_MANNERS_KB = frozenset({"vowel", "diphthong", "nasal", "liquid", "glide", "approximant", "tap"})


def _worker_init() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _bin_segment(arr: np.ndarray, n_bins: int = N_BINS_M) -> np.ndarray:
    n = len(arr)
    if n <= 0:
        raise ValueError("Empty segment")
    if n >= n_bins:
        edges = np.linspace(0, n, n_bins + 1, dtype=np.int32)
        out = [np.mean(arr[edges[b] : edges[b + 1]], axis=0) for b in range(n_bins)]
        return np.stack(out, axis=0)
    xp = np.linspace(0.0, 1.0, n, dtype=np.float64)
    xq = (np.arange(n_bins, dtype=np.float64) + 0.5) / float(n_bins)
    if arr.ndim == 1:
        return np.interp(xq, xp, arr)
    cols = [np.interp(xq, xp, arr[:, d]) for d in range(arr.shape[1])]
    return np.stack(cols, axis=1)


def _worker_slice_native_utt(utt: dict[str, Any]) -> list[dict[str, Any]]:
    """Populate 5-bin TrueHarmEnv acoustic tensors for ZH phones using existing alignments."""
    wav_path = REPO_ROOT / str(utt["wav_path"])
    x, sr = v6_voc.read_wav_mono(wav_path)
    if sr != SR:
        raise ValueError(f"Expected {SR} Hz, got {sr} for {utt['id']}")
    s0 = int(utt["trim_s0"])
    s1 = int(utt["trim_s1"])
    xa_t = x[s0:s1]

    from exp_20_pybot_larynx.larynx.synth_v2 import analyze_true_harmonic_env
    log_env, _f0, voiced, bap6 = analyze_true_harmonic_env(xa_t, sr=SR, hop=HOP, n_fft=1024, bap_scale=0.50)
    utt_log_ref = float(utt["utt_log_ref"])
    cep64 = log_env_to_cep(log_env - utt_log_ref, n_cep=N_CEP)
    n_fr = len(cep64)

    out_phones: list[dict[str, Any]] = []
    phones = utt.get("phones", [])
    n_p = len(phones)
    for i, ph in enumerate(phones):
        row = dict(ph)
        lang = str(row.get("lang", "zh"))
        if lang == "zh" and "cep_5" not in row:
            ps = max(0, min(n_fr - 1, int(row["p_start"])))
            pe = max(ps + 1, min(n_fr, int(row["p_end"])))
            row["cep_5"] = _bin_segment(cep64[ps:pe], N_BINS_M).astype(np.float16)
            row["bap_5"] = _bin_segment(bap6[ps:pe], N_BINS_M).astype(np.float16)
            row["vp_5"] = _bin_segment(voiced[ps:pe].astype(np.float64), N_BINS_M).astype(np.float16)
        row["same_word_prev"] = bool(
            row.get("same_word_prev", i > 0 and phones[i - 1].get("lang") == lang and float(row.get("pause_prev_ms", 0.0)) < 20.0 and phones[i - 1].get("word_idx") == row.get("word_idx"))
        )
        row["same_word_next"] = bool(
            row.get("same_word_next", i + 1 < n_p and phones[i + 1].get("lang") == lang and float(row.get("pause_after_ms", 0.0)) < 20.0 and phones[i + 1].get("word_idx") == row.get("word_idx"))
        )
        out_phones.append(row)
    return out_phones


def _sanitize_bap_bins(base_ipa: str, bap: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Clamp voiced alignment-bleed BAP spikes in stored acoustic bins."""
    out = np.asarray(bap, dtype=np.float64).copy()
    info = lookup_ipa_info(base_ipa)
    if not info.voiced:
        return np.clip(out, 0.004, 0.995)
    vm = np.asarray(vp, dtype=np.float64) >= 0.45
    if np.any(vm):
        if info.manner in _SONORANT_MANNERS_KB:
            out[vm, 0] = np.minimum(out[vm, 0], 0.18)
            out[vm, 1] = np.minimum(out[vm, 1], 0.38)
        else:
            out[vm, 0] = np.minimum(out[vm, 0], 0.28)
            out[vm, 1] = np.minimum(out[vm, 1], 0.50)
    return np.clip(out, 0.004, 0.995)


def _warp_studio_en_tokens_ljspeech_to_sf(
    sf_en_pns_counts: Counter[str],
    sf_en_base_counts: Counter[str],
    *,
    sf_en_prof: VoiceCloneProfile,
    sf_ref_log_amp: float,
    min_ctx_tokens: int = 12,
    max_per_ctx: int = 18,
    max_per_sparse_base: int = 180,
) -> list[dict[str, Any]]:
    """Pure EN->EN studio enrichment from LJSpeech -> SF_EN for missing (`ə`, `ɚ`, rare) and sparse triphones."""
    if not LJ_TRAIN_TOKENS_PATH.is_file() or not LJ_SOURCE_PROFILE_PATH.is_file():
        return []
    lj_prof = VoiceCloneProfile.from_json(LJ_SOURCE_PROFILE_PATH.read_text(encoding="utf-8"))
    with open(LJ_TRAIN_TOKENS_PATH, "rb") as f:
        lj_data = pickle.load(f)
    lj_ref_amp = float(lj_data.get("ref_log_amp", -4.9631))
    lj_f0 = float(lj_data.get("speaker_f0_med_hz", 202.26))

    missing_bases = {k for k in ("ə", "ɚ", "ð", "θ", "ʒ") if sf_en_base_counts.get(k, 0) < 40}
    base_caps = {"ə": 3500, "ɚ": 1500}
    picked: list[dict[str, Any]] = []
    ctx_picked: Counter[str] = Counter()
    base_picked: Counter[str] = Counter()

    # Pass 1: strict purity_ok studio tokens filling sparse triphones & missing weak vowels/rhotics
    for purity_req in (True, False):
        for raw_t in lj_data.get("tokens", []):
            if purity_req and not bool(raw_t.get("purity_ok", True)):
                continue
            base = str(raw_t.get("ipa", "")).strip()
            if not base:
                continue
            allo = str(raw_t.get("allophone") or base).strip()
            pk = f"en:{allo}"
            prev_c = int(raw_t.get("prev_cls", 0))
            next_c = int(raw_t.get("next_cls", 0))
            ptag = normalize_prosody_tag("en", raw_t.get("stress"), base)
            ctx_k = f"{prev_c}|{pk}|{next_c}|{ptag}"

            need_missing = base in missing_bases and base_picked[base] < base_caps.get(base, max_per_sparse_base)
            need_ctx = sf_en_pns_counts.get(ctx_k, 0) < min_ctx_tokens
            if not (need_missing or need_ctx):
                continue
            cap_ctx = max_per_ctx if base not in ("ə", "ɚ") else 24
            if ctx_picked[ctx_k] >= cap_ctx:
                continue
            if len(picked) >= 44000:
                break

            ctx_picked[ctx_k] += 1
            base_picked[base] += 1
            t_copy = dict(raw_t)
            t_copy["lang"] = "en"
            picked.append(t_copy)
        if len(picked) >= 44000 or all(base_picked[b] >= 20 for b in missing_bases):
            break

    if not picked:
        return []

    n_tok = len(picked)
    ceps = np.stack([np.asarray(t["cep_5"], dtype=np.float64) for t in picked], axis=0)
    baps = np.stack([np.asarray(t["bap_5"], dtype=np.float64) for t in picked], axis=0)
    vps = np.stack([np.asarray(t["vp_5"], dtype=np.float64) for t in picked], axis=0)
    flat_log_env = cep_to_log_env(ceps.reshape(-1, N_CEP), n_bins=N_BINS) + lj_ref_amp
    flat_vp = vps.reshape(-1) >= 0.45
    flat_f0 = np.where(flat_vp, lj_f0, 0.0)

    ctrl = LarynxControl(
        f0_hz=flat_f0,
        voiced=flat_vp,
        log_env=flat_log_env,
        bap=np.clip(baps.reshape(-1, N_BAP), 0.004, 0.995),
        glottal_tilt_db_per_oct=np.zeros(n_tok * N_BINS_M, dtype=np.float64),
        jitter_frac=np.full(n_tok * N_BINS_M, 0.0055, dtype=np.float64),
        shimmer_frac=np.full(n_tok * N_BINS_M, 0.025, dtype=np.float64),
        n_samples=n_tok * N_BINS_M * HOP,
        rd=np.ones(n_tok * N_BINS_M, dtype=np.float64),
    )
    warped_ctrl, _, _ = apply_clone_to_control(
        ctrl, lj_prof, sf_en_prof, strength=0.95, clone_f0=False, clone_glottal=True
    )
    warped_cep = log_env_to_cep(warped_ctrl.log_env - sf_ref_log_amp, n_cep=N_CEP).reshape(n_tok, N_BINS_M, N_CEP)
    warped_bap = warped_ctrl.bap.reshape(n_tok, N_BINS_M, N_BAP)
    for k, tok in enumerate(picked):
        clean_bap = _sanitize_bap_bins(str(tok.get("ipa", "ə")), warped_bap[k], vps[k])
        tok["cep_5"] = warped_cep[k].astype(np.float16)
        tok["bap_5"] = clean_bap.astype(np.float16)
        tok["speaker"] = "sf_en_studio"
        tok["utt_id"] = f"st_{tok['utt_id']}"
    return picked


def _pack_medoid_groups(
    prefix: str,
    groups: dict[str, MedoidGroup],
    *,
    gv_ps: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    """Pack `dict[str, MedoidGroup]` back into `_build_medoid_table_en10` CSR npz format."""
    keys = sorted(groups.keys())
    offsets = [0]
    tok_cnts = []
    med_cep, med_bap, med_vp = [], [], []
    med_prev, med_next, med_pos, med_dur, med_f0, med_csz = [], [], [], [], [], []
    gv_stds = []

    for k in keys:
        g = groups[k]
        tok_cnts.append(int(g.total_tokens))
        m_cnt = len(g.cep)
        base_ipa = strip_allophone_suffix(k.split("|")[1] if "|" in k else k)
        for m in range(m_cnt):
            med_cep.append(g.cep[m])
            med_bap.append(_sanitize_bap_bins(base_ipa, g.bap[m], g.vp[m]))
            med_vp.append(g.vp[m])
            med_prev.append(int(g.ctx_prev[m]))
            med_next.append(int(g.ctx_next[m]))
            med_pos.append(int(g.ctx_pos[m]))
            med_dur.append(float(g.ctx_dur_ms[m]))
            med_f0.append(float(g.ctx_f0_st[m]))
            med_csz.append(int(g.cluster_sz[m]))
        offsets.append(len(med_cep))
        if gv_ps is not None:
            gv_stds.append(gv_ps.get(k, np.ones(40, dtype=np.float64)).astype(np.float16))

    n_med = len(med_cep)
    max_len = max((len(k) for k in keys), default=1)
    out = {
        f"{prefix}_keys": np.array(keys, dtype=f"<U{max_len}"),
        f"{prefix}_offsets": np.array(offsets, dtype=np.int32),
        f"{prefix}_tok_cnt": np.array(tok_cnts, dtype=np.int32),
        f"{prefix}_cep": np.stack(med_cep, axis=0).astype(np.float16) if n_med else np.zeros((0, N_BINS_M, N_CEP), dtype=np.float16),
        f"{prefix}_bap": np.stack(med_bap, axis=0).astype(np.float16) if n_med else np.zeros((0, N_BINS_M, N_BAP), dtype=np.float16),
        f"{prefix}_vp": np.stack(med_vp, axis=0).astype(np.float16) if n_med else np.zeros((0, N_BINS_M), dtype=np.float16),
        f"{prefix}_ctx_prev": np.array(med_prev, dtype=np.int8),
        f"{prefix}_ctx_next": np.array(med_next, dtype=np.int8),
        f"{prefix}_ctx_pos": np.array(med_pos, dtype=np.int8),
        f"{prefix}_ctx_dur_ms": np.array(med_dur, dtype=np.float16),
        f"{prefix}_ctx_f0_st": np.array(med_f0, dtype=np.float16),
        f"{prefix}_cluster_sz": np.array(med_csz, dtype=np.int32),
    }
    if gv_ps is not None:
        out["gv_ps_keys"] = np.array(keys, dtype=f"<U{max_len}")
        out["gv_ps_std"] = np.stack(gv_stds, axis=0).astype(np.float16) if gv_stds else np.ones((0, 40), dtype=np.float16)
    return out, len(keys)


def _unpack_table_to_groups(tbl: dict[str, np.ndarray], prefix: str) -> dict[str, MedoidGroup]:
    return _load_medoid_table(tbl, prefix)


def build_sf_native_multilingual_kb(
    tokens_path: Path | str = SF_TRAIN_TOKENS_PATH,
    out_path: Path | str = SF_NATIVE_KB_PATH,
    summary_path: Path | str = SF_NATIVE_SUMMARY_PATH,
    *,
    workers: int = 12,
    c_dim: int = N_COARSE_CEP_M,
    k_max: int = 6,
    reuse_zh_from: Path | str | None = None,
) -> dict[str, Any]:
    """Build Stage 13 zero-mapping SF multilingual (`zh` + `en`) native acoustic `.npz`."""
    t0 = time.perf_counter()
    with open(Path(tokens_path), "rb") as f:
        tr = pickle.load(f)

    train_ids = set(tr.get("train_ids") or [])
    held_ids = set(tr.get("heldout_ids") or [])
    if train_ids & held_ids:
        raise AssertionError("Leakage between train_ids and heldout_ids")

    utterances = tr.get("utterances", [])
    for u in utterances:
        if str(u["id"]) in held_ids:
            raise AssertionError(f"Heldout utterance {u['id']} leaked into TRAIN")

    kb_old: MultilingualIPAKnowledge | None = None
    reuse_p = Path(reuse_zh_from) if reuse_zh_from is not None else None
    if reuse_p is not None and reuse_p.is_file():
        print(f"[SF-Stage13 Native] Reusing native ZH tables from {reuse_p} (fast EN studio enrichment mode)...", flush=True)
        kb_old = MultilingualIPAKnowledge.load(reuse_p)
        utt_phone_lists = []
        for u in utterances:
            phones = [dict(ph) for ph in u.get("phones", [])]
            n_p = len(phones)
            for i, row in enumerate(phones):
                lang = str(row.get("lang", "zh"))
                row["same_word_prev"] = bool(
                    row.get("same_word_prev", i > 0 and phones[i - 1].get("lang") == lang and float(row.get("pause_prev_ms", 0.0)) < 20.0 and phones[i - 1].get("word_idx") == row.get("word_idx"))
                )
                row["same_word_next"] = bool(
                    row.get("same_word_next", i + 1 < n_p and phones[i + 1].get("lang") == lang and float(row.get("pause_after_ms", 0.0)) < 20.0 and phones[i + 1].get("word_idx") == row.get("word_idx"))
                )
            utt_phone_lists.append(phones)
    else:
        print(f"[SF-Stage13 Native] Extracting native ZH TrueHarmEnv tokens across {len(utterances)} TRAIN utts (workers={workers})...", flush=True)
        utt_phone_lists = []
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
            for i, plist in enumerate(ex.map(_worker_slice_native_utt, utterances, chunksize=32), start=1):
                utt_phone_lists.append(plist)
                if i % 600 == 0 or i == len(utterances):
                    print(f"[SF-Stage13 Native]   sliced {i}/{len(utterances)} utts ({time.perf_counter() - t0:.1f}s)", flush=True)

    # Pure EN->EN studio enrichment from LJSpeech -> SF_EN (fills `ə`, `ɚ`, rare phones & sparse triphones)
    en_base_cnt: Counter[str] = Counter()
    en_pns_cnt: Counter[str] = Counter()
    for plist in utt_phone_lists:
        n_p = len(plist)
        for i, ph in enumerate(plist):
            if ph.get("lang") != "en":
                continue
            base = str(ph.get("ipa", "ə"))
            en_base_cnt[base] += 1
            if bool(ph.get("purity_ok", True)):
                allo = strip_allophone_suffix(str(ph.get("allophone") or base))
                is_c = str(ph.get("allophone") or "").endswith("_coda")
                pk = f"en:{allo}_coda" if is_c else f"en:{allo}"
                ptag = normalize_prosody_tag("en", ph.get("stress"), base)
                prev_c = int(ph.get("prev_cls", 0))
                next_c = int(ph.get("next_cls", 0))
                en_pns_cnt[f"{prev_c}|{pk}|{next_c}|{ptag}"] += 1
            # Sanitize outlier voiced BAP spikes on native EN tokens too
            if ph.get("bap_5") is not None and ph.get("vp_5") is not None:
                ph["bap_5"] = _sanitize_bap_bins(base, ph["bap_5"], ph["vp_5"]).astype(np.float16)

    speaker_f0_med_hz = float(tr.get("speaker_f0_med_hz", 223.73))
    ref_log_amp = float(tr.get("ref_log_amp", -4.1984))

    backstop_utts: list[list[dict[str, Any]]] = []
    n_studio_en = 0
    if SF_EN_PROFILE_PATH.is_file():
        sf_en_prof = VoiceCloneProfile.from_json(SF_EN_PROFILE_PATH.read_text(encoding="utf-8"))
        b_toks = _warp_studio_en_tokens_ljspeech_to_sf(
            en_pns_cnt,
            en_base_cnt,
            sf_en_prof=sf_en_prof,
            sf_ref_log_amp=ref_log_amp,
        )
        n_studio_en = len(b_toks)
        for bt in b_toks:
            backstop_utts.append([bt])
        print(f"[SF-Stage13 Native]   warped {n_studio_en} studio EN->EN tokens (LJSpeech -> SF_EN)", flush=True)

    all_lists = utt_phone_lists + backstop_utts
    sil_cep64 = np.zeros(N_CEP, dtype=np.float32)
    sil_cep64[0] = float((-16.0 - ref_log_amp) * math.sqrt(N_BINS))
    sil_bap6 = np.full(N_BAP, 0.99, dtype=np.float32)

    cep_l, bap_l, vp_l = [], [], []
    prev_cls_l, next_cls_l, pos_l, dur_l, f0_l, purity_l = [], [], [], [], [], []
    pns_idx: dict[str, list[int]] = defaultdict(list)
    pn_idx: dict[str, list[int]] = defaultdict(list)
    ps_idx: dict[str, list[int]] = defaultdict(list)
    p_idx: dict[str, list[int]] = defaultdict(list)
    base_idx: dict[str, list[int]] = defaultdict(list)
    shared_idx: dict[str, list[int]] = defaultdict(list)
    p_start_acc: dict[str, list[np.ndarray]] = defaultdict(list)
    p_end_acc: dict[str, list[np.ndarray]] = defaultdict(list)
    flat_tokens: list[dict[str, Any]] = []
    lang_counts: Counter[str] = Counter()

    for toks in all_lists:
        n_t = len(toks)
        for i, tok in enumerate(toks):
            cep = tok.get("cep_5")
            bap = tok.get("bap_5")
            vp = tok.get("vp_5")
            if cep is None or bap is None or vp is None:
                continue
            cep_a = np.asarray(cep, dtype=np.float16)
            bap_a = np.asarray(bap, dtype=np.float16)
            vp_a = np.asarray(vp, dtype=np.float16)
            if cep_a.shape != (N_BINS_M, N_CEP) or not np.all(np.isfinite(cep_a)):
                continue

            lang = str(tok.get("lang", "zh"))
            base_ipa = str(tok.get("ipa", "ə")).strip()
            allo = strip_allophone_suffix(str(tok.get("allophone") or base_ipa))
            is_coda = str(tok.get("allophone") or "").endswith("_coda")
            phone_str = f"{allo}_coda" if is_coda else allo
            phone_key = f"{lang}:{phone_str}"
            lang_base_key = f"{lang}:{base_ipa}"
            ptag = normalize_prosody_tag(lang, tok.get("stress"), base_ipa)

            prev_tok = toks[i - 1] if i > 0 else None
            next_tok = toks[i + 1] if i + 1 < n_t else None
            pause_prev = float(prev_tok.get("pause_after_ms", 0.0)) if prev_tok is not None else 999.0
            pause_after = float(tok.get("pause_after_ms", 999.0 if i == n_t - 1 else 0.0))
            if i == n_t - 1:
                pause_after = max(pause_after, 999.0)

            def _pk(t_obj: dict[str, Any] | None) -> str:
                if t_obj is None:
                    return "sil"
                tl = str(t_obj.get("lang", "zh"))
                ta = str(t_obj.get("allophone") or t_obj.get("ipa") or "sil")
                return f"{tl}:{ta}"

            prev_pk = "sil" if (prev_tok is None or pause_prev >= 25.0) else _pk(prev_tok)
            next_pk = "sil" if (next_tok is None or pause_after >= 25.0) else _pk(next_tok)
            prev_c = classify_universal_art_class(prev_pk)
            next_c = classify_universal_art_class(next_pk)
            pos_id = int(tok.get("pos_id", 0 if prev_c == 0 else (2 if next_c == 0 else 1)))

            n_w_ph = int(tok.get("word_n_phones", 1))
            if n_w_ph <= 1:
                word_pos = 0
            elif tok.get("is_word_initial", False):
                word_pos = 1
            elif tok.get("is_word_final", False):
                word_pos = 3
            else:
                word_pos = 2

            dur_ms = float(tok.get("dur_ms", 65.0))
            f0_st = float(tok.get("f0_med_st", 0.0))
            f0_5 = tok.get("f0_st_5")
            f3_st = (float(f0_5[0]), float(f0_5[2]), float(f0_5[4])) if (f0_5 is not None and len(f0_5) >= 5) else (f0_st, f0_st, f0_st)
            is_pure = bool(tok.get("purity_ok", True))

            ti = len(cep_l)
            cep_l.append(cep_a)
            bap_l.append(bap_a)
            vp_l.append(vp_a)
            prev_cls_l.append(prev_c)
            next_cls_l.append(next_c)
            pos_l.append(pos_id)
            dur_l.append(dur_ms)
            f0_l.append(f0_st)
            purity_l.append(is_pure)
            lang_counts[lang] += 1

            pns_idx[f"{prev_c}|{phone_key}|{next_c}|{ptag}"].append(ti)
            pn_idx[f"{phone_key}|{next_c}|{ptag}"].append(ti)
            ps_idx[f"{phone_key}|{ptag}"].append(ti)
            p_idx[phone_key].append(ti)
            base_idx[lang_base_key].append(ti)
            shared_idx[base_ipa].append(ti)

            if is_pure:
                p_start_acc[phone_key].append(cep_a[0, 1:9].astype(np.float64))
                p_end_acc[phone_key].append(cep_a[-1, 1:9].astype(np.float64))
                p_start_acc[base_ipa].append(cep_a[0, 1:9].astype(np.float64))
                p_end_acc[base_ipa].append(cep_a[-1, 1:9].astype(np.float64))

            flat_tokens.append({
                "base_ipa": base_ipa,
                "lang_base_key": lang_base_key,
                "phone_key": phone_key,
                "ptag": ptag,
                "prev_cls": prev_c,
                "next_cls": next_c,
                "pos_id": pos_id,
                "word_pos": word_pos,
                "is_func_word": bool(tok.get("is_func", False)),
                "is_coda": is_coda,
                "same_word_prev": bool(tok.get("same_word_prev", False)),
                "same_word_next": bool(tok.get("same_word_next", False)),
                "dur_ms": dur_ms,
                "f3_st": f3_st,
                "pause_prev_ms": pause_prev,
                "pause_after_ms": pause_after,
                "prev_pk": prev_pk,
                "next_pk": next_pk,
            })

    n_tok = len(cep_l)
    cep_arr = np.stack(cep_l, axis=0).astype(np.float16)
    bap_arr = np.stack(bap_l, axis=0).astype(np.float16)
    vp_arr = np.stack(vp_l, axis=0).astype(np.float16)
    prev_arr = np.array(prev_cls_l, dtype=np.int8)
    next_arr = np.array(next_cls_l, dtype=np.int8)
    pos_arr = np.array(pos_l, dtype=np.int8)
    dur_arr = np.array(dur_l, dtype=np.float32)
    f0_arr = np.array(f0_l, dtype=np.float32)
    purity_arr = np.array(purity_l, dtype=bool)

    pns_clean = _filter_strict_indices(pns_idx, purity_arr, min_strict=3)
    pn_clean = _filter_strict_indices(pn_idx, purity_arr, min_strict=3)
    ps_clean = _filter_strict_indices(ps_idx, purity_arr, min_strict=3)
    p_clean = _filter_strict_indices(p_idx, purity_arr, min_strict=3)
    base_clean = _filter_strict_indices(base_idx, purity_arr, min_strict=3)
    shared_clean = _filter_strict_indices(shared_idx, purity_arr, min_strict=3)

    for k, idxs in p_idx.items():
        if k not in p_start_acc:
            p_start_acc[k] = [cep_arr[i, 0, 1:9].astype(np.float64) for i in idxs]
            p_end_acc[k] = [cep_arr[i, -1, 1:9].astype(np.float64) for i in idxs]
    p_start_proto = {k: np.mean(v, axis=0) for k, v in p_start_acc.items()}
    p_end_proto = {k: np.mean(v, axis=0) for k, v in p_end_acc.items()}

    payload: dict[str, np.ndarray] = {}
    t_pns, n_pns = _build_medoid_table_en10("ctx_pns", pns_clean, cep_arr, bap_arr, vp_arr, prev_arr, next_arr, pos_arr, dur_arr, f0_arr, k_max=k_max)
    t_pn, n_pn = _build_medoid_table_en10("ctx_pn", pn_clean, cep_arr, bap_arr, vp_arr, prev_arr, next_arr, pos_arr, dur_arr, f0_arr, k_max=k_max)
    t_ps, n_ps = _build_medoid_table_en10("ctx_ps", ps_clean, cep_arr, bap_arr, vp_arr, prev_arr, next_arr, pos_arr, dur_arr, f0_arr, k_max=k_max, compute_gv=True)
    t_p, n_p = _build_medoid_table_en10("ctx_p", p_clean, cep_arr, bap_arr, vp_arr, prev_arr, next_arr, pos_arr, dur_arr, f0_arr, k_max=k_max)
    t_base, n_base = _build_medoid_table_en10("ctx_base", base_clean, cep_arr, bap_arr, vp_arr, prev_arr, next_arr, pos_arr, dur_arr, f0_arr, k_max=k_max)
    t_shared, n_shared = _build_medoid_table_en10("ipa_shared", shared_clean, cep_arr, bap_arr, vp_arr, prev_arr, next_arr, pos_arr, dur_arr, f0_arr, k_max=k_max)

    if kb_old is not None:
        def _is_zh_key(key: str) -> bool:
            return key.startswith("zh:") or "|zh:" in key

        p_start_proto = {**{k: v for k, v in kb_old.p_start_proto.items() if _is_zh_key(k) or ":" not in k}, **p_start_proto}
        p_end_proto = {**{k: v for k, v in kb_old.p_end_proto.items() if _is_zh_key(k) or ":" not in k}, **p_end_proto}

    X = np.empty((n_tok, D_ACOUSTIC_FEAT_M), dtype=np.float32)
    for ti, ft in enumerate(flat_tokens):
        X[ti] = extract_multilingual_acoustic_features(
            prev_cls=ft["prev_cls"],
            next_cls=ft["next_cls"],
            prosody_tag=ft["ptag"],
            pos_id=ft["pos_id"],
            word_pos=ft["word_pos"],
            is_func_word=ft["is_func_word"],
            is_coda=ft["is_coda"],
            same_word_prev=ft["same_word_prev"],
            same_word_next=ft["same_word_next"],
            dur_ms=ft["dur_ms"],
            f3_st=ft["f3_st"],
            pause_prev_ms=ft["pause_prev_ms"],
            pause_after_ms=ft["pause_after_ms"],
            prev_phone_key=ft["prev_pk"],
            next_phone_key=ft["next_pk"],
            p_end_proto=p_end_proto,
            p_start_proto=p_start_proto,
        )

    Y_cep = cep_arr[:, :, :c_dim].reshape(n_tok, N_BINS_M * c_dim)
    Y_bap = bap_arr.reshape(n_tok, N_BINS_M * N_BAP)

    w_shared = {k: _fit_ridge(X, Y_cep, idxs, lam=28.0) for k, idxs in shared_clean.items() if len(idxs) >= 1}
    w_base = {
        k: _fit_ridge(X, Y_cep, idxs, lam=24.0, prior_W=w_shared.get(k.split(":", 1)[1]))
        for k, idxs in base_clean.items()
        if len(idxs) >= 1
    }
    w_phone: dict[str, np.ndarray] = {}
    for k, idxs in p_clean.items():
        if len(idxs) < 12:
            continue
        lang_pfx, ph_nm = k.split(":", 1)
        b_key = f"{lang_pfx}:{strip_allophone_suffix(ph_nm)}"
        w_phone[k] = _fit_ridge(X, Y_cep, idxs, lam=20.0, prior_W=w_base.get(b_key, w_shared.get(strip_allophone_suffix(ph_nm))))

    w_ps: dict[str, np.ndarray] = {}
    for k, idxs in ps_clean.items():
        if len(idxs) < 16:
            continue
        ph_k = k.split("|", 1)[0]
        lang_pfx, ph_nm = ph_k.split(":", 1)
        b_key = f"{lang_pfx}:{strip_allophone_suffix(ph_nm)}"
        parent = w_phone.get(ph_k, w_base.get(b_key, w_shared.get(strip_allophone_suffix(ph_nm))))
        w_ps[k] = _fit_ridge(X, Y_cep, idxs, lam=16.0, prior_W=parent)

    w_pns: dict[str, np.ndarray] = {}
    d_pns: dict[str, np.ndarray] = {}
    for k, idxs in pns_clean.items():
        _, pk, _, pt_k = k.split("|", 3)
        lang_pfx, ph_nm = pk.split(":", 1)
        b_key = f"{lang_pfx}:{strip_allophone_suffix(ph_nm)}"
        parent = w_ps.get(f"{pk}|{pt_k}", w_phone.get(pk, w_base.get(b_key, w_shared.get(strip_allophone_suffix(ph_nm)))))
        if len(idxs) >= 24:
            w_pns[k] = _fit_ridge(X, Y_cep, idxs, lam=14.0, prior_W=parent)
        elif parent is not None and len(idxs) >= 1:
            arr = np.asarray(idxs, dtype=np.int64)
            res = Y_cep[arr].astype(np.float64) - (X[arr].astype(np.float64) @ parent)
            d_pns[k] = ((float(len(idxs)) / (float(len(idxs)) + 6.0)) * np.mean(res, axis=0)).reshape(N_BINS_M, c_dim)

    w_bap_ps = {k: _fit_ridge(X, Y_bap, idxs, lam=24.0) for k, idxs in ps_clean.items() if len(idxs) >= 4}
    for k, idxs in shared_clean.items():
        if len(idxs) >= 4:
            w_bap_ps[f"{k}|0"] = _fit_ridge(X, Y_bap, idxs, lam=26.0)

    if kb_old is not None:
        # Fast merge: combine reused native `zh:` tables (bap-sanitized) + newly built `en:` studio tables
        def _is_zh_key(key: str) -> bool:
            return key.startswith("zh:") or "|zh:" in key

        zh_pns = {k: v for k, v in kb_old.ctx_pns.items() if _is_zh_key(k)}
        zh_pn = {k: v for k, v in kb_old.ctx_pn.items() if _is_zh_key(k)}
        zh_ps = {k: v for k, v in kb_old.ctx_ps.items() if _is_zh_key(k)}
        zh_p = {k: v for k, v in kb_old.ctx_p.items() if _is_zh_key(k)}
        zh_base = {k: v for k, v in kb_old.ctx_base.items() if _is_zh_key(k)}

        merged_pns = {**zh_pns, **_unpack_table_to_groups(t_pns, "ctx_pns")}
        merged_pn = {**zh_pn, **_unpack_table_to_groups(t_pn, "ctx_pn")}
        en_ps_grp = _unpack_table_to_groups(t_ps, "ctx_ps")
        merged_ps = {**zh_ps, **en_ps_grp}
        merged_p = {**zh_p, **_unpack_table_to_groups(t_p, "ctx_p")}
        merged_base = {**zh_base, **_unpack_table_to_groups(t_base, "ctx_base")}
        merged_shared = {**kb_old.ipa_shared, **_unpack_table_to_groups(t_shared, "ipa_shared")}

        en_gv = {str(k): t_ps["gv_ps_std"][i].astype(np.float64) for i, k in enumerate(t_ps["gv_ps_keys"])}
        merged_gv = {**{k: v for k, v in kb_old.gv_ps.items() if _is_zh_key(k)}, **en_gv}

        t_pns, n_pns = _pack_medoid_groups("ctx_pns", merged_pns)
        t_pn, n_pn = _pack_medoid_groups("ctx_pn", merged_pn)
        t_ps, n_ps = _pack_medoid_groups("ctx_ps", merged_ps, gv_ps=merged_gv)
        t_p, n_p = _pack_medoid_groups("ctx_p", merged_p)
        t_base, n_base = _pack_medoid_groups("ctx_base", merged_base)
        t_shared, n_shared = _pack_medoid_groups("ipa_shared", merged_shared)

        p_start_proto = {**{k: v for k, v in kb_old.p_start_proto.items() if _is_zh_key(k) or ":" not in k}, **p_start_proto}
        p_end_proto = {**{k: v for k, v in kb_old.p_end_proto.items() if _is_zh_key(k) or ":" not in k}, **p_end_proto}

        w_shared = {**kb_old.w_shared, **w_shared}
        w_base = {**{k: v for k, v in kb_old.w_base.items() if _is_zh_key(k)}, **w_base}
        w_phone = {**{k: v for k, v in kb_old.w_phone.items() if _is_zh_key(k)}, **w_phone}
        w_ps = {**{k: v for k, v in kb_old.w_ps.items() if _is_zh_key(k)}, **w_ps}
        w_pns = {**{k: v for k, v in kb_old.w_pns.items() if _is_zh_key(k)}, **w_pns}
        d_pns = {**{k: v for k, v in kb_old.d_pns.items() if _is_zh_key(k)}, **d_pns}
        w_bap_ps = {**{k: v for k, v in kb_old.w_bap_ps.items() if _is_zh_key(k)}, **w_bap_ps}

    for tbl in (t_pns, t_pn, t_ps, t_p, t_base, t_shared):
        payload.update(tbl)
    payload["gv_glob"] = np.median(t_ps["gv_ps_std"].astype(np.float64), axis=0).astype(np.float16)
    payload.update(_pack_protos("p_start_proto", p_start_proto))
    payload.update(_pack_protos("p_end_proto", p_end_proto))

    payload.update(_pack_weights("w_shared", w_shared))
    payload.update(_pack_weights("w_base", w_base))
    payload.update(_pack_weights("w_phone", w_phone))
    payload.update(_pack_weights("w_ps", w_ps))
    payload.update(_pack_weights("w_pns", w_pns))
    payload.update(_pack_weights("d_pns", d_pns))
    payload.update(_pack_weights("w_bap_ps", w_bap_ps))

    g_c32 = cep_arr[:, :, 1:40].astype(np.float32)
    g_idx = int(np.argmin(np.sum((g_c32 - np.mean(g_c32, axis=0)[None, :, :]) ** 2, axis=(1, 2))))
    payload["glob_cep"] = cep_arr[g_idx].astype(np.float16)
    payload["glob_bap"] = bap_arr[g_idx].astype(np.float16)
    payload["glob_vp"] = vp_arr[g_idx].astype(np.float16)

    payload["sil_cep64"] = sil_cep64
    payload["sil_bap6"] = sil_bap6
    payload["speaker_f0_med_hz"] = np.array([speaker_f0_med_hz], dtype=np.float32)
    payload["ref_log_amp"] = np.array([ref_log_amp], dtype=np.float32)
    payload["n_train_utts"] = np.array([len(utterances)], dtype=np.int32)
    payload["c_dim"] = np.array([c_dim], dtype=np.int32)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_p, **payload)
    clear_sf_native_multilingual_cache()
    size_mb = out_p.stat().st_size / (1024.0 * 1024.0)

    kb_chk = MultilingualIPAKnowledge.load(out_p)
    assert kb_chk.n_train_utts == len(utterances)

    zh_to_en_lv = [abs(float(e["level_diff_db"])) for e in tr.get("switch_events", []) if e.get("direction") == "zh_to_en" and e.get("gap_frames", 1) == 0 and e.get("level_diff_db") is not None]
    en_to_zh_lv = [abs(float(e["level_diff_db"])) for e in tr.get("switch_events", []) if e.get("direction") == "en_to_zh" and e.get("gap_frames", 1) == 0 and e.get("level_diff_db") is not None]
    zh_to_en_f0 = [abs(float(e["f0_diff_st"])) for e in tr.get("switch_events", []) if e.get("direction") == "zh_to_en" and e.get("gap_frames", 1) == 0 and e.get("f0_diff_st") is not None]
    en_to_zh_f0 = [abs(float(e["f0_diff_st"])) for e in tr.get("switch_events", []) if e.get("direction") == "en_to_zh" and e.get("gap_frames", 1) == 0 and e.get("f0_diff_st") is not None]
    sw_stats: dict[str, dict[str, float]] = {
        "zero_gap": {
            "n_zh_to_en": len(zh_to_en_lv),
            "n_en_to_zh": len(en_to_zh_lv),
            "zh_to_en_med_level_diff_db": round(float(np.median(zh_to_en_lv)), 3) if zh_to_en_lv else 0.0,
            "en_to_zh_med_level_diff_db": round(float(np.median(en_to_zh_lv)), 3) if en_to_zh_lv else 0.0,
            "zh_to_en_med_f0_diff_st": round(float(np.median(zh_to_en_f0)), 3) if zh_to_en_f0 else 0.0,
            "en_to_zh_med_f0_diff_st": round(float(np.median(en_to_zh_f0)), 3) if en_to_zh_f0 else 0.0,
        }
    }

    n_zh_final = int(lang_counts["zh"]) if kb_old is None else int(tr.get("n_zh_phones", 88284))
    summary = {
        "schema": "sf_native_multilingual_stage13",
        "ml_weights_mb": 0.0,
        "allow_pickle": False,
        "n_train_utts": len(utterances),
        "n_train_tokens": n_zh_final + int(lang_counts["en"]),
        "n_zh_tokens": n_zh_final,
        "n_en_tokens": int(lang_counts["en"]),
        "n_en_studio_warped_tokens": n_studio_en,
        "n_strict_tokens": int(np.sum(purity_arr)),
        "npz_size_mb": round(size_mb, 3),
        "speaker_f0_med_hz": round(speaker_f0_med_hz, 2),
        "ref_log_amp": round(ref_log_amp, 4),
        "medoid_counts": {
            "ctx_pns": n_pns,
            "ctx_pn": n_pn,
            "ctx_ps": n_ps,
            "ctx_p": n_p,
            "ctx_base": n_base,
            "ipa_shared": n_shared,
        },
        "ridge_counts": {
            "w_shared": len(w_shared),
            "w_base": len(w_base),
            "w_phone": len(w_phone),
            "w_ps": len(w_ps),
            "w_pns": len(w_pns),
            "d_pns": len(d_pns),
            "w_bap_ps": len(w_bap_ps),
        },
        "switch_stats": sw_stats,
        "build_time_s": round(time.perf_counter() - t0, 2),
    }
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[SF-Stage13 Native] Built {out_p} ({size_mb:.2f} MB, ZH={n_zh_final}, EN={lang_counts['en']}, studio_EN={n_studio_en}, shared={n_shared}) in {summary['build_time_s']}s", flush=True)
    return summary


def lookup_ipa_info_safe(ipa: str | None) -> str:
    return strip_allophone_suffix(ipa) or "sil"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reuse-zh-from", type=str, default=None)
    args = ap.parse_args()
    build_sf_native_multilingual_kb(workers=args.workers, reuse_zh_from=args.reuse_zh_from)
