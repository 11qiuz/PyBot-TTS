"""Build Stage 12 Same-Speaker Bilingual Embedded-EN Acoustic KB & Prosody Model (`SF_bilingual`).

Zero-ML deterministic `.npz` artifacts (`allow_pickle=False`):
1. `knowledge_en_sf_stage12.npz`:
   - Built strictly from `sf_bilingual_train_tokens.pkl` (2,700 TRAIN mixed utterances, zero leakage),
     retaining true Mandarin/English neighbor boundary classes across code switches!
   - Pure same-language `EN -> EN` rare-phone backstop: any base IPA / allophone with `< 20` tokens
     in `SF_bilingual` (e.g. `ð`, `ʒ`, rare codas) is warped via pure EN->EN `apply_clone_to_control`
     from `en_source_profile.json (LJSpeech_EN)` -> `sf_en_profile.json (SF_EN)` so all 39 English
     phonemes live cleanly in `SF`'s exact vocal tract.
2. `prosody_en_sf_stage12.npz`:
   - Built strictly from `SF_bilingual` TRAIN contiguous English runs (5-fold CV Huber-IRLS duration +
     Ridge F0 + Ridge gain_db), preserving `ProsodyEnStage10Model` schema with natural embedded F0 tuning.
   - Evaluated strictly on unseen `sf_bilingual_heldout.pkl` (40 utterances, zero leakage).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

from collections import Counter, defaultdict
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
from exp_20_pybot_larynx.clone import VoiceCloneProfile, apply_clone_to_control  # noqa: E402
from exp_20_pybot_larynx.crossling.build_en_prosody_stage10 import _solve_irls_huber, _solve_ridge  # noqa: E402
from exp_20_pybot_larynx.crossling.build_knowledge_en_stage10 import (  # noqa: E402
    _build_medoid_table_en10,
    _filter_strict_indices,
)
from exp_20_pybot_larynx.crossling.en_articulation import resolve_en_allophone  # noqa: E402
from exp_20_pybot_larynx.crossling.en_prosody_stage10 import (  # noqa: E402
    ProsodyEnStage10Model,
    compute_phone_prosody_context,
    extract_duration_features,
    extract_prosody_features,
    predict_en_prosody_stage10,
)
from exp_20_pybot_larynx.crossling.knowledge_en_stage10 import (  # noqa: E402
    D_ACOUSTIC_FEAT_S10,
    N_COARSE_CEP_S10,
    N_EN_BINS,
    AcousticKnowledgeEnStage10,
    extract_en_stage10_acoustic_features,
    normalize_stress,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env, log_env_to_cep  # noqa: E402

CROSSLING_DATA_DIR = EXP_DIR / "crossling_data"
CROSSLING_DIR = EXP_DIR / "crossling"
SF_TRAIN_TOKENS_PATH = CROSSLING_DATA_DIR / "sf_bilingual_train_tokens.pkl"
SF_HELDOUT_PATH = CROSSLING_DATA_DIR / "sf_bilingual_heldout.pkl"
SF_EN_PROFILE_PATH = CROSSLING_DATA_DIR / "sf_en_profile.json"
LJ_TRAIN_TOKENS_PATH = CROSSLING_DATA_DIR / "en_stage10_train_tokens.pkl"
LJ_SOURCE_PROFILE_PATH = CROSSLING_DATA_DIR / "en_source_profile.json"

EN_SF_STAGE12_KB_PATH = CROSSLING_DIR / "knowledge_en_sf_stage12.npz"
EN_SF_STAGE12_KB_SUMMARY_PATH = CROSSLING_DIR / "knowledge_en_sf_stage12_summary.json"
EN_SF_STAGE12_PROSODY_PATH = CROSSLING_DIR / "prosody_en_sf_stage12.npz"
EN_SF_STAGE12_PROSODY_SUMMARY_PATH = CROSSLING_DIR / "prosody_en_sf_stage12_summary.json"


def _warp_rare_en_tokens_ljspeech_to_sf(
    sparse_bases: set[str],
    *,
    sf_en_prof: VoiceCloneProfile,
    sf_ref_log_amp: float,
    sf_f0_med_hz: float,
    max_per_base: int = 160,
) -> list[dict[str, Any]]:
    """Pure EN -> EN backstop: warp rare EN phoneme tokens from LJSpeech_EN into SF_EN space."""
    if not sparse_bases or not LJ_TRAIN_TOKENS_PATH.is_file() or not LJ_SOURCE_PROFILE_PATH.is_file():
        return []
    lj_prof = VoiceCloneProfile.from_json(LJ_SOURCE_PROFILE_PATH.read_text(encoding="utf-8"))
    with open(LJ_TRAIN_TOKENS_PATH, "rb") as f:
        lj_data = pickle.load(f)
    lj_ref_amp = float(lj_data.get("ref_log_amp", -4.96))
    lj_f0 = float(lj_data.get("speaker_f0_med_hz", 202.26))

    picked: list[dict[str, Any]] = []
    cnt: Counter[str] = Counter()
    # Deterministic evenly-spaced scan over LJSpeech TRAIN tokens
    all_lj = [t for t in lj_data["tokens"] if str(t.get("ipa")) in sparse_bases and bool(t.get("purity_ok", True))]
    if not all_lj:
        all_lj = [t for t in lj_data["tokens"] if str(t.get("ipa")) in sparse_bases]
    for tok in all_lj:
        base = str(tok["ipa"])
        if cnt[base] >= max_per_base:
            continue
        cnt[base] += 1
        picked.append(dict(tok))
    if not picked:
        return []

    # Stack 5-bin tokens & warp batch via `apply_clone_to_control(lj_prof -> sf_en_prof)`
    n_tok = len(picked)
    ceps = np.stack([np.asarray(t["cep_5"], dtype=np.float64) for t in picked], axis=0)  # (N, 5, 64)
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
        glottal_tilt_db_per_oct=np.zeros(n_tok * N_EN_BINS, dtype=np.float64),
        jitter_frac=np.full(n_tok * N_EN_BINS, 0.0055, dtype=np.float64),
        shimmer_frac=np.full(n_tok * N_EN_BINS, 0.025, dtype=np.float64),
        n_samples=n_tok * N_EN_BINS * HOP,
        rd=np.ones(n_tok * N_EN_BINS, dtype=np.float64),
    )
    warped_ctrl, _, _ = apply_clone_to_control(
        ctrl, lj_prof, sf_en_prof, strength=0.92, clone_f0=False, clone_glottal=False
    )
    warped_cep = log_env_to_cep(warped_ctrl.log_env - sf_ref_log_amp, n_cep=N_CEP).reshape(n_tok, N_EN_BINS, N_CEP)
    warped_bap = warped_ctrl.bap.reshape(n_tok, N_EN_BINS, N_BAP)
    for k, tok in enumerate(picked):
        tok["cep_5"] = warped_cep[k].astype(np.float16)
        tok["bap_5"] = warped_bap[k].astype(np.float16)
        tok["speaker"] = "sf_en_backstop"
        tok["utt_id"] = f"bs_{tok['utt_id']}"
    return picked


def build_sf_bilingual_kb(
    tokens_path: Path = SF_TRAIN_TOKENS_PATH,
    out_path: Path = EN_SF_STAGE12_KB_PATH,
    summary_path: Path = EN_SF_STAGE12_KB_SUMMARY_PATH,
    *,
    c_dim: int = N_COARSE_CEP_S10,
    min_tokens_pns_w: int = 16,
    min_tokens_ps_w: int = 12,
    min_tokens_phone_w: int = 8,
    k_max: int = 4,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    with open(tokens_path, "rb") as f:
        data = pickle.load(f)

    train_ids = set(data["train_ids"])
    held_ids = set(data["heldout_ids"])
    assert not (train_ids & held_ids), "Zero-leakage violated!"

    sf_tokens = []
    for raw_t in data["tokens"]:
        if str(raw_t["utt_id"]) in held_ids:
            continue
        t = dict(raw_t)
        arpa_s = str(t.get("arpabet") or "").strip().upper()
        if arpa_s == "AH0":
            t["ipa"] = "ə"
            t["allophone"] = resolve_en_allophone("ə", None if str(t.get("next_ipa")) == "SIL" else str(t.get("next_ipa")), bool(t.get("same_word_next", False)))
        elif arpa_s == "ER0":
            t["ipa"] = "ɚ"
            t["allophone"] = resolve_en_allophone("ɚ", None if str(t.get("next_ipa")) == "SIL" else str(t.get("next_ipa")), bool(t.get("same_word_next", False)))
        sf_tokens.append(t)
    speaker_f0_med_hz = float(data["en_f0_med_hz"])
    ref_log_amp = float(data["ref_log_amp"])
    sil_cep64 = np.asarray(data["sil_cep64"], dtype=np.float32)
    sil_bap6 = np.asarray(data["sil_bap6"], dtype=np.float32)

    # Identify sparse EN phonemes (< 20 tokens in SF_bilingual) and apply pure EN->EN SF backstop
    sf_base_counts = Counter(str(t["ipa"]) for t in sf_tokens)
    all_en_bases = {
        "i", "ɪ", "eɪ", "ɛ", "æ", "ɑ", "ɔ", "oʊ", "ʊ", "u", "ʌ", "ə", "ɚ", "ɝ", "aɪ", "aʊ", "ɔɪ",
        "p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
        "m", "n", "ŋ", "l", "ɹ", "w", "j",
    }
    sparse_bases = {b for b in all_en_bases if sf_base_counts[b] < 20}
    backstop_tokens: list[dict[str, Any]] = []
    if sparse_bases and SF_EN_PROFILE_PATH.is_file():
        sf_en_prof = VoiceCloneProfile.from_json(SF_EN_PROFILE_PATH.read_text(encoding="utf-8"))
        backstop_tokens = _warp_rare_en_tokens_ljspeech_to_sf(
            sparse_bases,
            sf_en_prof=sf_en_prof,
            sf_ref_log_amp=ref_log_amp,
            sf_f0_med_hz=speaker_f0_med_hz,
            max_per_base=160,
        )
    all_tokens = sf_tokens + backstop_tokens

    cep_l, bap_l, vp_l = [], [], []
    prev_cls_l, next_cls_l, pos_l, dur_l, f0_l, purity_l = [], [], [], [], [], []
    pns_idx: dict[str, list[int]] = defaultdict(list)
    pn_idx: dict[str, list[int]] = defaultdict(list)
    ps_idx: dict[str, list[int]] = defaultdict(list)
    p_idx: dict[str, list[int]] = defaultdict(list)
    base_idx: dict[str, list[int]] = defaultdict(list)
    p_start_acc: dict[str, list[np.ndarray]] = defaultdict(list)
    p_end_acc: dict[str, list[np.ndarray]] = defaultdict(list)
    flat_tokens: list[dict[str, Any]] = []

    for tok in all_tokens:
        cep = np.asarray(tok["cep_5"], dtype=np.float16)
        bap = np.asarray(tok["bap_5"], dtype=np.float16)
        vp = np.asarray(tok["vp_5"], dtype=np.float16)
        if cep.shape != (N_EN_BINS, N_CEP) or not np.all(np.isfinite(cep)):
            continue

        base_ipa = str(tok["ipa"]).strip()
        same_w_prev = bool(tok.get("same_word_prev", False))
        same_w_next = bool(tok.get("same_word_next", False))
        next_ipa = str(tok.get("next_ipa", "SIL"))
        prev_ipa = str(tok.get("prev_ipa", "SIL"))
        phone = str(tok.get("allophone") or resolve_en_allophone(base_ipa, None if next_ipa == "SIL" else next_ipa, same_w_next))
        stress = normalize_stress(tok.get("stress"), base_ipa)
        prev_c = int(tok.get("prev_cls", 0))
        next_c = int(tok.get("next_cls", 0))
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

        pause_prev = float(tok.get("pause_prev_ms", 0.0))
        pause_after = float(tok.get("pause_after_ms", 0.0))
        dur_ms = float(tok.get("dur_ms", 65.0))
        f0_st = float(tok.get("f0_med_st", 0.0))
        f0_5 = tok.get("f0_st_5")
        f3_st = (float(f0_5[0]), float(f0_5[2]), float(f0_5[4])) if (f0_5 is not None and len(f0_5) >= 5) else (f0_st, f0_st, f0_st)
        is_pure = bool(tok.get("purity_ok", True))

        ti = len(cep_l)
        cep_l.append(cep)
        bap_l.append(bap)
        vp_l.append(vp)
        prev_cls_l.append(prev_c)
        next_cls_l.append(next_c)
        pos_l.append(pos_id)
        dur_l.append(dur_ms)
        f0_l.append(f0_st)
        purity_l.append(is_pure)

        pns_idx[f"{prev_c}|{phone}|{next_c}|{stress}"].append(ti)
        pn_idx[f"{phone}|{next_c}|{stress}"].append(ti)
        ps_idx[f"{phone}|{stress}"].append(ti)
        p_idx[phone].append(ti)
        base_idx[base_ipa].append(ti)

        if is_pure:
            p_start_acc[phone].append(cep[0, 1:9].astype(np.float64))
            p_end_acc[phone].append(cep[-1, 1:9].astype(np.float64))
            if base_ipa != phone:
                p_start_acc[base_ipa].append(cep[0, 1:9].astype(np.float64))
                p_end_acc[base_ipa].append(cep[-1, 1:9].astype(np.float64))

        flat_tokens.append({
            "base_ipa": base_ipa,
            "phone": phone,
            "stress": stress,
            "prev_cls": prev_c,
            "next_cls": next_c,
            "pos_id": pos_id,
            "word_pos": word_pos,
            "is_func_word": bool(tok.get("is_func", False)),
            "is_coda": phone.endswith("_coda"),
            "same_word_prev": same_w_prev,
            "same_word_next": same_w_next,
            "dur_ms": dur_ms,
            "f3_st": f3_st,
            "pause_prev_ms": pause_prev,
            "pause_after_ms": pause_after,
            "prev_phone": "sil" if prev_ipa == "SIL" else prev_ipa,
            "next_phone": "sil" if next_ipa == "SIL" else next_ipa,
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

    for tbl in (t_pns, t_pn, t_ps, t_p, t_base):
        payload.update(tbl)
    payload["gv_glob"] = np.median(t_ps["gv_ps_std"].astype(np.float64), axis=0).astype(np.float16)
    payload.update(_pack_protos("p_start_proto", p_start_proto))
    payload.update(_pack_protos("p_end_proto", p_end_proto))

    X = np.empty((n_tok, D_ACOUSTIC_FEAT_S10), dtype=np.float32)
    for ti, ft in enumerate(flat_tokens):
        X[ti] = extract_en_stage10_acoustic_features(
            prev_cls=ft["prev_cls"],
            next_cls=ft["next_cls"],
            stress=ft["stress"],
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
            prev_phone=ft["prev_phone"],
            next_phone=ft["next_phone"],
            p_end_proto=p_end_proto,
            p_start_proto=p_start_proto,
        )

    Y_cep = cep_arr[:, :, :c_dim].reshape(n_tok, N_EN_BINS * c_dim)
    Y_bap = bap_arr.reshape(n_tok, N_EN_BINS * N_BAP)
    w_base = {k: _fit_ridge(X, Y_cep, idxs, lam=28.0) for k, idxs in base_clean.items() if len(idxs) >= 1}
    w_phone = {k: _fit_ridge(X, Y_cep, idxs, lam=22.0, prior_W=w_base.get(k.replace("_coda", ""))) for k, idxs in p_clean.items() if len(idxs) >= min_tokens_phone_w}
    w_ps: dict[str, np.ndarray] = {}
    for k, idxs in ps_clean.items():
        if len(idxs) < min_tokens_ps_w:
            continue
        ph_k = k.split("|", 1)[0]
        w_ps[k] = _fit_ridge(X, Y_cep, idxs, lam=18.0, prior_W=w_phone.get(ph_k, w_base.get(ph_k.replace("_coda", ""))))
    w_pns: dict[str, np.ndarray] = {}
    d_pns: dict[str, np.ndarray] = {}
    for k, idxs in pns_clean.items():
        _, ph_k, _, st_k = k.split("|", 3)
        parent = w_ps.get(f"{ph_k}|{st_k}", w_phone.get(ph_k, w_base.get(ph_k.replace("_coda", ""))))
        if len(idxs) >= min_tokens_pns_w:
            w_pns[k] = _fit_ridge(X, Y_cep, idxs, lam=14.0, prior_W=parent)
        elif parent is not None and len(idxs) >= 1:
            arr = np.asarray(idxs, dtype=np.int64)
            res = Y_cep[arr].astype(np.float64) - (X[arr].astype(np.float64) @ parent)
            w_sh = float(len(idxs)) / (float(len(idxs)) + 6.0)
            d_pns[k] = (w_sh * np.mean(res, axis=0)).reshape(N_EN_BINS, c_dim)
    w_bap_ps = {k: _fit_ridge(X, Y_bap, idxs, lam=24.0) for k, idxs in ps_clean.items() if len(idxs) >= 4}

    payload.update(_pack_weights("w_base", w_base))
    payload.update(_pack_weights("w_phone", w_phone))
    payload.update(_pack_weights("w_ps", w_ps))
    payload.update(_pack_weights("w_pns", w_pns))
    payload.update(_pack_weights("d_pns", d_pns))
    payload.update(_pack_weights("w_bap_ps", w_bap_ps))

    g_c32 = cep_arr[:, :, 1:40].astype(np.float32)
    g_idx = int(np.argmin(np.sum((g_c32 - np.mean(g_c32, axis=0)[None, :, :]) ** 2, axis=(1, 2))))
    payload["en_glob_cep"] = cep_arr[g_idx].astype(np.float16)
    payload["en_glob_bap"] = bap_arr[g_idx].astype(np.float16)
    payload["en_glob_vp"] = vp_arr[g_idx].astype(np.float16)
    payload["sil_cep64"] = sil_cep64
    payload["sil_bap6"] = sil_bap6
    payload["speaker_f0_med_hz"] = np.array([speaker_f0_med_hz], dtype=np.float32)
    payload["ref_log_amp"] = np.array([ref_log_amp], dtype=np.float32)
    payload["n_train_utts"] = np.array([int(data["n_utts"])], dtype=np.int32)
    payload["c_dim"] = np.array([c_dim], dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    size_mb = out_path.stat().st_size / (1024.0 * 1024.0)
    kb_chk = AcousticKnowledgeEnStage10.load(out_path)
    assert kb_chk.n_train_utts == int(data["n_utts"])

    summary = {
        "schema": "knowledge_en_sf_stage12",
        "allow_pickle": False,
        "n_train_utts": int(data["n_utts"]),
        "n_sf_tokens": len(sf_tokens),
        "n_backstop_tokens": len(backstop_tokens),
        "sparse_backstop_bases": sorted(sparse_bases),
        "n_total_tokens": n_tok,
        "npz_size_mb": round(size_mb, 3),
        "speaker_f0_med_hz": round(speaker_f0_med_hz, 2),
        "ref_log_amp": round(ref_log_amp, 4),
        "medoid_counts": {"ctx_pns": n_pns, "ctx_pn": n_pn, "ctx_ps": n_ps, "ctx_p": n_p, "ctx_base": n_base},
        "ridge_counts": {"w_base": len(w_base), "w_phone": len(w_phone), "w_ps": len(w_ps), "w_pns": len(w_pns), "d_pns": len(d_pns)},
        "build_time_s": round(time.perf_counter() - t0, 2),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_sf_bilingual_prosody(
    tokens_path: Path = SF_TRAIN_TOKENS_PATH,
    heldout_path: Path = SF_HELDOUT_PATH,
    out_path: Path = EN_SF_STAGE12_PROSODY_PATH,
    summary_path: Path = EN_SF_STAGE12_PROSODY_SUMMARY_PATH,
    *,
    n_folds: int = 5,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    with open(tokens_path, "rb") as f:
        tr = pickle.load(f)

    train_ids = set(tr["train_ids"])
    held_ids = set(tr["heldout_ids"])
    assert not (train_ids & held_ids)
    en_f0_med = float(tr["en_f0_med_hz"])

    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tok in tr["tokens"]:
        rid = str(tok.get("run_id") or tok["utt_id"])
        by_run[rid].append(tok)
    runs = sorted(by_run.items(), key=lambda kv: kv[0])

    allos_vocab = sorted({str(t["allophone"]) for t in tr["tokens"]})
    ipas_vocab = sorted({str(t["ipa"]) for t in tr["tokens"]})
    ctx_vocab = ["SIL"] + ipas_vocab
    allo_map = {k: i for i, k in enumerate(allos_vocab)}
    ipa_map = {k: i for i, k in enumerate(ipas_vocab)}
    ctx_map = {k: i for i, k in enumerate(ctx_vocab)}
    as_pairs = sorted({(str(t["allophone"]), 0 if t["stress"] == 0 else (1 if t["stress"] == 1 else (2 if t["stress"] == 2 else 3))) for t in tr["tokens"]})
    as_map = {k: i for i, k in enumerate(as_pairs)}
    fw_vocab = sorted({str(t["word"]).lower() for t in tr["tokens"] if bool(t["is_func"])})
    fw_map = {k: i for i, k in enumerate(fw_vocab)}

    prev_bg_cnt = Counter((str(t["prev_ipa"]), str(t["allophone"])) for t in tr["tokens"])
    next_bg_cnt = Counter((str(t["allophone"]), str(t["next_ipa"])) for t in tr["tokens"])
    tri_bg_cnt = Counter((str(t["prev_ipa"]), str(t["allophone"]), str(t["next_ipa"])) for t in tr["tokens"])
    prev_bg = sorted(k for k, v in prev_bg_cnt.items() if v >= 8)
    next_bg = sorted(k for k, v in next_bg_cnt.items() if v >= 8)
    tri_bg = sorted(k for k, v in tri_bg_cnt.items() if v >= 12)
    pbg_map = {k: i for i, k in enumerate(prev_bg)}
    nbg_map = {k: i for i, k in enumerate(next_bg)}
    tbg_map = {k: i for i, k in enumerate(tri_bg)}

    X_dur_l, X_pros_l, y_dur_l, y_f0_l, y_gain_l, f0_m_l, f0_w_l, fold_l = [], [], [], [], [], [], [], []
    for ri, (_rid, toks) in enumerate(runs):
        toks = sorted(toks, key=lambda t: int(t["phone_idx"]))
        is_cfin = bool(toks[-1]["is_phrase_final"] or toks[-1]["is_utterance_final"])
        ctxs = compute_phone_prosody_context(toks, is_clause_final=is_cfin)
        X_dur_l.append(extract_duration_features(ctxs, allo_map=allo_map, ipa_map=ipa_map, ctx_map=ctx_map, as_map=as_map, fw_map=fw_map, pbg_map=pbg_map, nbg_map=nbg_map, tbg_map=tbg_map))
        X_pros_l.append(extract_prosody_features(ctxs, allo_map=allo_map, ipa_map=ipa_map, ctx_map=ctx_map, fw_map=fw_map))
        y_dur_l.append(np.array([float(t["dur_ms"]) for t in toks], dtype=np.float64))
        y_f0_l.append(np.array([t["f0_st_5"] for t in toks], dtype=np.float64))
        y_gain_l.append(np.array([float(t["gain_db"]) for t in toks], dtype=np.float64))
        f0_m_l.append(np.array([bool(t["f0_has_voiced"]) for t in toks], dtype=bool))
        f0_w_l.append(np.array([max(0.25, float(t["voiced_frac"])) for t in toks], dtype=np.float64))
        fold_l.append(np.full(len(toks), ri % n_folds, dtype=np.int32))

    X_dur = np.vstack(X_dur_l)
    X_pros = np.vstack(X_pros_l)
    y_dur = np.concatenate(y_dur_l)
    y_f0_5 = np.vstack(y_f0_l)
    y_gain = np.concatenate(y_gain_l)
    f0_mask = np.concatenate(f0_m_l)
    f0_wt = np.concatenate(f0_w_l)
    folds = np.concatenate(fold_l)

    best_lam_dur, best_cv_dur = 3.0, 1e9
    y_dur_c = np.clip(y_dur, 20.0, 310.0)
    for lam in (3.0, 8.0):
        maes = []
        for k in range(n_folds):
            trm, vam = folds != k, folds == k
            wk = _solve_irls_huber(X_dur[trm], y_dur_c[trm], lam, lam_irls_scale=0.03, delta=5.0, n_iter=7)
            maes.append(float(np.mean(np.abs(np.clip(X_dur[vam] @ wk, 20.0, 310.0) - y_dur[vam]))))
        if float(np.mean(maes)) < best_cv_dur:
            best_cv_dur, best_lam_dur = float(np.mean(maes)), lam
    w_dur = _solve_irls_huber(X_dur, y_dur_c, best_lam_dur, lam_irls_scale=0.03, delta=5.0, n_iter=10)

    best_lam_f0, best_cv_f0 = 25.0, 1e9
    y_f0_c = np.clip(y_f0_5, -10.0, 10.0)
    for lam in (10.0, 25.0, 50.0, 100.0):
        rmses = []
        for k in range(n_folds):
            trm, vam = (folds != k) & f0_mask, (folds == k) & f0_mask
            Xw = X_pros[trm] * f0_wt[trm, None]
            wk = _solve_ridge(Xw.T @ X_pros[trm], Xw.T @ y_f0_c[trm], lam)
            rmses.append(float(np.sqrt(np.mean((np.clip(X_pros[vam] @ wk, -10.0, 10.0) - y_f0_5[vam]) ** 2))))
        if float(np.mean(rmses)) < best_cv_f0:
            best_cv_f0, best_lam_f0 = float(np.mean(rmses)), lam
    Xw_f0 = X_pros[f0_mask] * f0_wt[f0_mask, None]
    w_f0_5 = _solve_ridge(Xw_f0.T @ X_pros[f0_mask], Xw_f0.T @ y_f0_c[f0_mask], best_lam_f0)

    best_lam_g, best_cv_g = 25.0, 1e9
    y_g_c = np.clip(y_gain, -16.0, 9.0)
    for lam in (10.0, 25.0, 50.0):
        g_maes = []
        for k in range(n_folds):
            trm, vam = folds != k, folds == k
            wk = _solve_ridge(X_pros[trm].T @ X_pros[trm], X_pros[trm].T @ y_g_c[trm], lam)
            g_maes.append(float(np.mean(np.abs(X_pros[vam] @ wk - y_gain[vam]))))
        if float(np.mean(g_maes)) < best_cv_g:
            best_cv_g, best_lam_g = float(np.mean(g_maes)), lam
    w_gain = _solve_ridge(X_pros.T @ X_pros, X_pros.T @ y_g_c, best_lam_g)

    # Embedded conversational prosody tuned directly to SF_bilingual mixed F0 kinematics
    params_dict = {
        "f0_ridge_w": 0.90,
        "f0_accent_w": 0.88,
        "f0_nuc_boost_st": 1.45,
        "f0_content_boost_st": 1.05,
        "f0_sec_boost_st": 0.35,
        "f0_func_trough_st": -0.95,
        "f0_decl_span_st": 1.05,
        "f0_q_rise_st": 2.20,
        "f0_s_fall_st": 0.95,
        "f0_qta_tau_ms": 16.0,
        "f0_cf0_vl_st": 0.55,
        "f0_ar1_sigma": 0.14,
        "f0_ar1_rho": 0.76,
        "gain_scale": 0.35,
        "gain_min_db": -5.0,
        "gain_max_db": 3.5,
    }
    np.savez_compressed(
        out_path,
        allos_vocab=np.asarray(allos_vocab, dtype=str),
        ipas_vocab=np.asarray(ipas_vocab, dtype=str),
        ctx_vocab=np.asarray(ctx_vocab, dtype=str),
        as_allos=np.asarray([a for a, _ in as_pairs], dtype=str),
        as_stress=np.asarray([s for _, s in as_pairs], dtype=np.int32),
        fw_vocab=np.asarray(fw_vocab, dtype=str),
        prev_bg=np.asarray(prev_bg, dtype=str),
        next_bg=np.asarray(next_bg, dtype=str),
        tri_bg=np.asarray(tri_bg, dtype=str),
        w_dur=w_dur.astype(np.float64),
        w_f0_5=w_f0_5.astype(np.float64),
        w_gain=w_gain.astype(np.float64),
        speaker_f0_med_hz=np.asarray([en_f0_med], dtype=np.float64),
        param_keys=np.asarray(list(params_dict.keys()), dtype=str),
        param_vals=np.asarray(list(params_dict.values()), dtype=np.float64),
    )
    size_kb = out_path.stat().st_size / 1024.0
    mdl = ProsodyEnStage10Model(out_path)

    # Zero-leakage evaluation on SF_bilingual heldout runs
    ho_dur_errs, ho_f0_rmses, ho_iqrs = [], [], []
    if heldout_path.is_file():
        with open(heldout_path, "rb") as f:
            ho = pickle.load(f)
        for idx_u, u in enumerate(ho["utterances"]):
            assert str(u["id"]) not in train_ids
            # Group heldout EN tokens by run_id
            by_r: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for ep in u.get("en_phones") or []:
                by_r[str(ep.get("run_id") or u["id"])].append(ep)
            for _rid, ref_phones in sorted(by_r.items()):
                # Convert aligned ref tokens into minimal `en_units` list matching runtime
                units_in = []
                cur_w_idx = None
                cur_arpas = []
                cur_pause = 0.0
                for rp in ref_phones:
                    wi = int(rp["word_idx"])
                    if cur_w_idx is not None and wi != cur_w_idx:
                        units_in.append({"lang": "en", "token": "word", "arpabet": list(cur_arpas), "pause_after_ms": cur_pause})
                        cur_arpas = []
                    cur_w_idx = wi
                    if rp.get("arpabet"):
                        cur_arpas.append(str(rp["arpabet"]))
                    cur_pause = float(rp.get("pause_after_ms", 0.0))
                if cur_arpas:
                    units_in.append({"lang": "en", "token": "word", "arpabet": list(cur_arpas), "pause_after_ms": 0.0})
                if not units_in:
                    continue
                pred = predict_en_prosody_stage10(
                    units_in, base_f0_hz=en_f0_med, include_pauses=False, seed=2026 + idx_u, model=mdl
                )
                p_durs = pred["phone_durs_ms"]
                for k in range(min(len(ref_phones), len(p_durs))):
                    ho_dur_errs.append(abs(float(p_durs[k]) - float(ref_phones[k]["dur_ms"])))
                sp_list = [s for s in pred["phone_spans"] if not s["is_pause"]]
                p_errs = []
                v_fr = np.zeros(pred["n_frames"], dtype=bool)
                for k in range(min(len(ref_phones), len(sp_list))):
                    rp = ref_phones[k]
                    sa, sb = sp_list[k]["start_fr"], sp_list[k]["end_fr"]
                    if not rp.get("f0_has_voiced") or sb <= sa:
                        continue
                    v_fr[sa:sb] = True
                    p_errs.append((float(np.median(pred["f0_st"][sa:sb])) - float(rp["f0_med_st"])) ** 2)
                if p_errs:
                    ho_f0_rmses.append(math.sqrt(float(np.mean(p_errs))))
                if int(np.sum(v_fr)) >= 8:
                    ho_iqrs.append(float(np.percentile(pred["f0_st"][v_fr], 75) - np.percentile(pred["f0_st"][v_fr], 25)))

    summary = {
        "schema": "prosody_en_sf_stage12",
        "allow_pickle": False,
        "artifact_size_kb": round(size_kb, 2),
        "n_runs": len(runs),
        "n_tokens": len(y_dur),
        "cv_train_dur_mae_ms": round(best_cv_dur, 2),
        "cv_train_f0_rmse_st": round(best_cv_f0, 3),
        "cv_train_gain_mae_db": round(best_cv_g, 2),
        "heldout_dur_mae_ms": round(float(np.mean(ho_dur_errs)), 2) if ho_dur_errs else None,
        "heldout_f0_rmse_st": round(float(np.median(ho_f0_rmses)), 3) if ho_f0_rmses else None,
        "heldout_f0_iqr_st": round(float(np.median(ho_iqrs)), 3) if ho_iqrs else None,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    build_sf_bilingual_kb()
    build_sf_bilingual_prosody()
