"""Stage 10 Zero-Leakage Forced Alignment, TrueHarmEnv Token Extraction & Source Profile.

1. Loads `exp_20_pybot_larynx/crossling_data/slt_full/manifest.json` (1,102 TRAIN,
   30 HELDOUT: `slt_0030`..`slt_0059`).
2. Plans every utterance with `@en-frontend`'s `en_frontend_v2.plan_en_v2`
   (fixing `this -> DH`, full CMUdict, function-word weak forms).
3. Extracts Stage 7 `TrueHarmEnv` features (`analyze_true_harmonic_env(x24, sr=24000, bap_scale=0.50)`)
   in parallel via `ProcessPoolExecutor`.
4. Runs `exp_v11_align_gmm.gmm_hmm_aligner._train_and_align_gmm(..., n_em=4)` strictly on
   TRAIN utterances, then aligns HELDOUT separately with `_align_with_trained_gmm` (zero leakage).
5. Saves:
   - `exp_20_pybot_larynx/crossling_data/en_stage10_train_tokens.pkl`
   - `exp_20_pybot_larynx/crossling_data/en_stage10_heldout.pkl`
   - `exp_20_pybot_larynx/crossling_data/en_source_profile.json`
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
from collections import Counter
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

from exp_20_pybot_larynx import EXP_DIR, HOP, N_BAP, N_CEP, N_FFT, SR  # noqa: E402
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB  # noqa: E402
from exp_20_pybot_larynx.clone import extract_clone_profile  # noqa: E402
from exp_20_pybot_larynx.crossling.en_articulation import (  # noqa: E402
    ART_CLASSES,
    VOICELESS_OBSTRUENTS,
    VOWEL_IPAS,
    classify_articulatory_class,
)
from exp_20_pybot_larynx.crossling.en_frontend_v2 import plan_en_v2  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import log_env_to_cep  # noqa: E402
from exp_v6_vocoder import vocoder as v6_voc  # noqa: E402
from exp_v8_clarity import en_utau_bank as v8_en  # noqa: E402
from exp_v10_g2p_and_align.landmark_aligner import score_token_purity  # noqa: E402
from exp_v11_align_gmm.gmm_hmm_aligner import (  # noqa: E402
    _align_with_trained_gmm,
    _train_and_align_gmm,
    _trim_utterance_active,
)

CROSSLING_DATA_DIR = EXP_DIR / "crossling_data"
LJSPEECH_MANIFEST_PATH = CROSSLING_DATA_DIR / "ljspeech_s11" / "manifest.json"
SLT_MANIFEST_PATH = CROSSLING_DATA_DIR / "slt_full" / "manifest.json"
TRAIN_TOKENS_PATH = CROSSLING_DATA_DIR / "en_stage10_train_tokens.pkl"
HELDOUT_PATH = CROSSLING_DATA_DIR / "en_stage10_heldout.pkl"
SOURCE_PROFILE_PATH = CROSSLING_DATA_DIR / "en_source_profile.json"
ALIGN_SUMMARY_PATH = CROSSLING_DATA_DIR / "en_stage10_align_summary.json"

N_BINS = 5


def _worker_init() -> None:
    for _ev in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_ev] = "1"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _bin_segment(arr: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Average `(T, ...)` array slice uniformly into `n_bins` phase bins."""
    t_len = len(arr)
    if t_len == 0:
        return np.zeros((n_bins, *arr.shape[1:]), dtype=np.float64)
    out = np.empty((n_bins, *arr.shape[1:]), dtype=np.float64)
    for bi in range(n_bins):
        lo = int(math.floor(bi * t_len / n_bins))
        hi = int(math.floor((bi + 1) * t_len / n_bins))
        if hi <= lo:
            lo = min(t_len - 1, bi * t_len // n_bins)
            hi = lo + 1
        out[bi] = np.mean(arr[lo:hi], axis=0)
    return out


def _worker_analyze_utt(task: tuple[str, str, str, str]) -> dict[str, Any]:
    """Extract Stage 7 TrueHarmEnv features & plan_en_v2 phones for one utterance."""
    _worker_init()
    from exp_20_pybot_larynx.larynx.synth_v2 import analyze_true_harmonic_env

    uid, spk, wav_rel, text = task
    x, file_sr = v6_voc.read_wav_mono(REPO_ROOT / wav_rel)
    if file_sr != SR:
        raise ValueError(f"Expected {SR} Hz, got {file_sr} for {uid}")

    s0, s1 = v8_en._trim_active_span(x, sr=SR)
    xa = x[s0:s1]

    log_env_full, f0_full, voiced_full, bap6_full = analyze_true_harmonic_env(
        xa, sr=SR, hop=HOP, n_fft=N_FFT, bap_scale=0.50
    )
    feats_pre = v6_voc.VocoderFeatures(
        f0_hz=f0_full,
        voiced=voiced_full,
        log_env=log_env_full,
        bap=bap6_full[:, :5],
        sr=SR,
        hop_samples=HOP,
        n_samples=len(xa),
        n_fft=N_FFT,
    )
    cs, ce, cf0, cf1 = _trim_utterance_active(xa, feats_pre, sr=SR, hop=HOP)
    xa_t = xa[cs:ce]
    log_env = log_env_full[cf0:cf1]
    f0_hz = f0_full[cf0:cf1]
    voiced = voiced_full[cf0:cf1]
    bap6 = bap6_full[cf0:cf1]

    feats = v6_voc.VocoderFeatures(
        f0_hz=f0_hz,
        voiced=voiced,
        log_env=log_env,
        bap=bap6[:, :5],
        sr=SR,
        hop_samples=HOP,
        n_samples=len(xa_t),
        n_fft=N_FFT,
    )

    plan = plan_en_v2(text)
    phones_plan = list(plan.get("phones") or [])
    ipas = [str(p["ipa"]) for p in phones_plan]

    v_mask = voiced & (f0_hz > 50.0)
    if np.any(v_mask):
        utt_log_ref = float(np.median(np.mean(log_env[v_mask, 1:257], axis=1)))
        utt_f0_med = float(np.median(f0_hz[v_mask]))
        utt_voiced_db = float(np.mean(np.mean(log_env[v_mask], axis=1) * NEPER_TO_DB))
    else:
        utt_log_ref = 0.0
        utt_f0_med = 175.0
        utt_voiced_db = -35.0

    return {
        "id": uid,
        "speaker": spk,
        "wav_path": wav_rel,
        "text": text,
        "clean_text": str(plan.get("text", text)),
        "is_question": bool(plan.get("is_question", False)),
        "trim_s0": int(s0 + cs),
        "trim_s1": int(s0 + ce),
        "wav": xa_t.astype(np.float32),
        "feats": feats,
        "bap6": bap6.astype(np.float16),
        "utt_log_ref": utt_log_ref,
        "utt_f0_med": utt_f0_med,
        "utt_voiced_db": utt_voiced_db,
        "ipas": ipas,
        "plan_words": list(plan.get("words") or []),
        "plan_phones": phones_plan,
    }


def _extract_utt_tokens(
    rec: dict[str, Any],
    bounds: list[tuple[int, int, int]],
    *,
    include_full_frames: bool,
) -> dict[str, Any]:
    """Build structured Stage 10 phone tokens & utterance record from GMM-HMM bounds."""
    uid = rec["id"]
    wav = rec["wav"].astype(np.float64)
    feats: v6_voc.VocoderFeatures = rec["feats"]
    log_env = feats.log_env
    f0_hz = feats.f0_hz
    voiced = feats.voiced
    bap6 = rec["bap6"].astype(np.float64)
    n_fr = len(log_env)

    utt_log_ref = float(rec["utt_log_ref"])
    utt_f0_med = max(50.0, float(rec["utt_f0_med"]))
    utt_voiced_db = float(rec["utt_voiced_db"])

    log_env_norm = log_env - utt_log_ref
    cep64 = log_env_to_cep(log_env_norm, n_cep=N_CEP).astype(np.float16)
    frame_db = np.mean(log_env, axis=1) * NEPER_TO_DB

    phones_plan = rec["plan_phones"]
    n_p = len(phones_plan)
    assert len(bounds) == n_p, f"Alignment length mismatch for {uid}: {len(bounds)} != {n_p}"

    active_mask = np.zeros(n_fr, dtype=bool)
    tokens: list[dict[str, Any]] = []
    n_strict = 0
    reason_counts: Counter[str] = Counter()

    for i, (ph, (ps, pm, pe)) in enumerate(zip(phones_plan, bounds)):
        ps_i = max(0, min(n_fr - 1, int(ps)))
        pe_i = max(ps_i + 1, min(n_fr, int(pe)))
        pm_i = max(ps_i, min(pe_i - 1, int(pm)))
        active_mask[ps_i:pe_i] = True

        ipa = str(ph["ipa"])
        allo = str(ph["allophone"])
        seg_w = wav[ps_i * HOP : min(len(wav), pe_i * HOP)]
        dur_ms = float((pe_i - ps_i) * HOP * 1000.0 / SR)
        purity_ok, reason, p_met = score_token_purity(
            ipa, log_env[ps_i:pe_i], voiced[ps_i:pe_i], seg_w, sr=SR, hop=HOP
        )
        if purity_ok:
            n_strict += 1
        reason_counts[reason] += 1

        prev_ph = phones_plan[i - 1] if i > 0 else None
        next_ph = phones_plan[i + 1] if i + 1 < n_p else None
        prev_pause = float(prev_ph["pause_after_ms"]) if prev_ph is not None else 999.0
        cur_pause = float(ph["pause_after_ms"])

        prev_ipa = str(prev_ph["ipa"]) if (prev_ph is not None and prev_pause < 20.0) else "SIL"
        next_ipa = str(next_ph["ipa"]) if (next_ph is not None and cur_pause < 20.0) else "SIL"
        prev_cls = classify_articulatory_class(None if prev_ipa == "SIL" else prev_ipa)
        next_cls = classify_articulatory_class(None if next_ipa == "SIL" else next_ipa)
        pos_id = 0 if prev_ipa == "SIL" else (2 if next_ipa == "SIL" else 1)

        same_word_next = bool(
            next_ph is not None
            and cur_pause < 20.0
            and int(next_ph["word_idx"]) == int(ph["word_idx"])
        )
        pre_fortis = bool(
            bool(ph["is_vowel"])
            and same_word_next
            and next_ipa in VOICELESS_OBSTRUENTS
        )

        cep5 = _bin_segment(cep64[ps_i:pe_i].astype(np.float64), N_BINS).astype(np.float16)
        bap5 = _bin_segment(bap6[ps_i:pe_i], N_BINS).astype(np.float16)
        vp5 = _bin_segment(voiced[ps_i:pe_i].astype(np.float64), N_BINS).astype(np.float16)
        rel_db5 = (_bin_segment(frame_db[ps_i:pe_i], N_BINS) - utt_voiced_db).astype(np.float32)
        gain_db = float(np.mean(rel_db5))

        f0_seg = f0_hz[ps_i:pe_i]
        vm_seg = voiced[ps_i:pe_i] & (f0_seg > 45.0)
        if np.any(vm_seg):
            st_raw = 12.0 * np.log2(np.where(vm_seg, f0_seg, utt_f0_med) / utt_f0_med)
            if not np.all(vm_seg):
                idx_v = np.where(vm_seg)[0]
                st_raw = np.interp(np.arange(len(f0_seg)), idx_v, st_raw[idx_v])
            f0_st_5 = _bin_segment(st_raw, N_BINS).astype(np.float32)
            f0_med_st = float(np.median(12.0 * np.log2(f0_seg[vm_seg] / utt_f0_med)))
            f0_has_voiced = True
        else:
            f0_st_5 = np.zeros(N_BINS, dtype=np.float32)
            f0_med_st = 0.0
            f0_has_voiced = False

        tokens.append({
            "utt_id": uid,
            "speaker": rec["speaker"],
            "phone_idx": int(ph["phone_idx"]),
            "p_start": ps_i,
            "p_mid": pm_i,
            "p_end": pe_i,
            "dur_ms": round(dur_ms, 2),
            "ipa": ipa,
            "allophone": allo,
            "arpabet": ph["arpabet"],
            "stress": ph["stress"],
            "word_stress": int(ph["word_stress"]),
            "is_func": bool(ph["is_func"]),
            "is_focused": bool(ph["is_focused"]),
            "is_vowel": bool(ph["is_vowel"]),
            "word": str(ph["word"]),
            "word_idx": int(ph["word_idx"]),
            "phone_in_word_idx": int(ph["phone_in_word_idx"]),
            "word_n_phones": int(ph["word_n_phones"]),
            "is_word_initial": bool(ph["is_word_initial"]),
            "is_word_final": bool(ph["is_word_final"]),
            "is_phrase_final": bool(ph["is_phrase_final"]),
            "is_utterance_final": (i == n_p - 1),
            "is_in_phrase_final_word": bool(ph["is_in_phrase_final_word"]),
            "cluster_len": int(ph["cluster_len"]),
            "pause_after_ms": float(ph["pause_after_ms"]),
            "pre_fortis": pre_fortis,
            "prev_ipa": prev_ipa,
            "next_ipa": next_ipa,
            "prev_cls": int(prev_cls),
            "next_cls": int(next_cls),
            "pos_id": int(pos_id),
            "is_question": bool(rec["is_question"]),
            "voiced_frac": round(float(p_met["voiced_frac"]), 3),
            "rms_dbfs": float(p_met["rms_dbfs"]),
            "purity_ok": bool(purity_ok),
            "quarantine_reason": str(reason),
            "cep_5": cep5,
            "bap_5": bap5,
            "vp_5": vp5,
            "rel_db_5": rel_db5,
            "gain_db": round(gain_db, 3),
            "f0_st_5": f0_st_5,
            "f0_med_st": round(f0_med_st, 4),
            "f0_has_voiced": f0_has_voiced,
        })

    sil_mask = ~active_mask
    if np.any(sil_mask):
        sil_cep = np.mean(cep64[sil_mask].astype(np.float64), axis=0)
        sil_bap = np.mean(bap6[sil_mask], axis=0)
        sil_cnt = int(np.sum(sil_mask))
    else:
        sil_cep = np.zeros(N_CEP, dtype=np.float64)
        sil_bap = np.ones(N_BAP, dtype=np.float64) * 0.95
        sil_cnt = 0

    utt_doc: dict[str, Any] = {
        "id": uid,
        "speaker": rec["speaker"],
        "wav_path": rec["wav_path"],
        "text": rec["text"],
        "clean_text": rec["clean_text"],
        "is_question": bool(rec["is_question"]),
        "sr": SR,
        "hop": HOP,
        "n_fft": N_FFT,
        "n_frames": n_fr,
        "trim_s0": int(rec["trim_s0"]),
        "trim_s1": int(rec["trim_s1"]),
        "utt_log_ref": utt_log_ref,
        "utt_f0_med": utt_f0_med,
        "utt_voiced_db": utt_voiced_db,
        "words": rec["plan_words"],
        "bounds": [(int(a), int(m), int(b)) for a, m, b in bounds],
        "phones": tokens,
        "f0_hz": f0_hz.astype(np.float32),
        "voiced": voiced.astype(bool),
        "sil_cep": sil_cep.astype(np.float32),
        "sil_bap": sil_bap.astype(np.float32),
        "sil_cnt": sil_cnt,
        "n_phones": n_p,
        "n_strict_ok": n_strict,
    }
    if include_full_frames:
        utt_doc["wav"] = rec["wav"]
        utt_doc["log_env"] = log_env.astype(np.float16)
        utt_doc["cep64"] = cep64
        utt_doc["bap6"] = rec["bap6"]
    return utt_doc


def _select_clean_source_clips(train_recs: list[dict[str, Any]], n_clips: int = 25) -> list[dict[str, Any]]:
    """Select `n_clips` clean mid-length TRAIN clips for `en_source_profile.json`."""
    cands: list[tuple[float, dict[str, Any]]] = []
    for rec in train_recs:
        feats: v6_voc.VocoderFeatures = rec["feats"]
        dur_s = len(rec["wav"]) / float(SR)
        v_mask = feats.voiced & (feats.f0_hz > 50.0)
        v_frac = float(np.mean(v_mask)) if len(v_mask) else 0.0
        if not (1.8 <= dur_s <= 5.5 and v_frac >= 0.45):
            continue
        rms = float(np.sqrt(np.mean(rec["wav"].astype(np.float64) ** 2)))
        if rms < 0.02:
            continue
        score = v_frac * min(dur_s, 4.0)
        cands.append((score, rec))
    cands.sort(key=lambda t: (-t[0], t[1]["id"]))
    step = max(1, len(cands) // max(1, n_clips * 2))
    picked = [c[1] for c in cands[::step][:n_clips]]
    if len(picked) < min(n_clips, len(cands)):
        seen = {r["id"] for r in picked}
        for _, r in cands:
            if r["id"] not in seen:
                picked.append(r)
                seen.add(r["id"])
            if len(picked) >= n_clips:
                break
    return picked


def run_stage10_alignment(
    *,
    manifest_path: Path | None = None,
    n_em: int = 4,
    workers: int = 12,
) -> dict[str, Any]:
    """Run zero-leakage Stage 10/11 alignment, TrueHarmEnv extraction & source profile build."""
    t0 = time.perf_counter()
    if manifest_path is None:
        manifest_path = LJSPEECH_MANIFEST_PATH if LJSPEECH_MANIFEST_PATH.is_file() else SLT_MANIFEST_PATH
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    speaker_name = str(manifest.get("speaker") or "slt")
    corpus_name = str(manifest.get("corpus") or speaker_name)
    heldout_set = set(manifest["heldout_ids"])
    train_set = set(manifest["train_ids"])
    assert not (train_set & heldout_set), "Zero-leakage split violated!"
    assert len(heldout_set) >= 20, f"Expected >=20 heldout utterances, got {len(heldout_set)}"

    train_tasks: list[tuple[str, str, str, str]] = []
    held_tasks: list[tuple[str, str, str, str]] = []
    for u in manifest["utterances"]:
        rel_p = str(u.get("path") or u.get("wav_path"))
        item = (u["id"], u.get("speaker", speaker_name), rel_p, u["text"])
        if u["id"] in heldout_set:
            held_tasks.append(item)
        else:
            train_tasks.append(item)

    print(
        f"[Stage11-Align] Corpus={corpus_name} speaker={speaker_name}: Extracting TrueHarmEnv (bap_scale=0.50) + plan_en_v2: "
        f"{len(train_tasks)} TRAIN, {len(held_tasks)} HELDOUT (workers={workers})...",
        flush=True,
    )
    train_recs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        for i, rec in enumerate(ex.map(_worker_analyze_utt, train_tasks, chunksize=16), start=1):
            train_recs.append(rec)
            if i % 500 == 0 or i == len(train_tasks):
                print(f"[Stage11-Align]   extracted {i}/{len(train_tasks)} TRAIN ({time.perf_counter() - t0:.1f}s)", flush=True)
        held_recs = list(ex.map(_worker_analyze_utt, held_tasks, chunksize=8))

    t_feat = time.perf_counter()
    print(f"[Stage11-Align] Feature extraction done in {t_feat - t0:.1f}s. Training GMM-HMM ({n_em} EM iters) strictly on TRAIN...", flush=True)
    gmm, train_bounds = _train_and_align_gmm(train_recs, n_em=n_em, sr=SR, hop=HOP)

    held_bounds: dict[str, list[tuple[int, int, int]]] = {}
    for rec in held_recs:
        held_bounds[rec["id"]] = _align_with_trained_gmm(
            gmm, rec["wav"], rec["feats"], rec["ipas"], sr=SR, hop=HOP
        )
    t_align = time.perf_counter()
    print(f"[Stage11-Align] GMM-HMM alignment done in {t_align - t_feat:.1f}s.", flush=True)

    # Package TRAIN tokens (strictly excluding HELDOUT)
    train_utts: list[dict[str, Any]] = []
    flat_train_tokens: list[dict[str, Any]] = []
    sil_cep_sum = np.zeros(N_CEP, dtype=np.float64)
    sil_bap_sum = np.zeros(N_BAP, dtype=np.float64)
    sil_cnt_tot = 0
    n_tok_train = 0
    n_ok_train = 0
    reason_train: Counter[str] = Counter()

    for rec in train_recs:
        uid = rec["id"]
        assert uid not in heldout_set
        udoc = _extract_utt_tokens(rec, train_bounds[uid], include_full_frames=False)
        train_utts.append(udoc)
        for tok in udoc["phones"]:
            flat_train_tokens.append(tok)
            n_tok_train += 1
            if tok["purity_ok"]:
                n_ok_train += 1
            reason_train[tok["quarantine_reason"]] += 1
        sc = int(udoc["sil_cnt"])
        if sc > 0:
            sil_cep_sum += sc * udoc["sil_cep"].astype(np.float64)
            sil_bap_sum += sc * udoc["sil_bap"].astype(np.float64)
            sil_cnt_tot += sc

    # Package HELDOUT utterances + aligned tokens
    held_utts: list[dict[str, Any]] = []
    flat_held_tokens: list[dict[str, Any]] = []
    n_tok_held = 0
    n_ok_held = 0
    for rec in held_recs:
        uid = rec["id"]
        assert uid in heldout_set
        udoc = _extract_utt_tokens(rec, held_bounds[uid], include_full_frames=True)
        held_utts.append(udoc)
        for tok in udoc["phones"]:
            flat_held_tokens.append(tok)
            n_tok_held += 1
            if tok["purity_ok"]:
                n_ok_held += 1

    train_accept = n_ok_train / max(1, n_tok_train)
    held_accept = n_ok_held / max(1, n_tok_held)
    assert train_accept >= 0.75, f"TRAIN token acceptance {train_accept:.3f} < 0.75!"

    train_f0_med = float(np.median([u["utt_f0_med"] for u in train_utts]))
    train_log_ref = float(np.median([u["utt_log_ref"] for u in train_utts]))
    train_voiced_db = float(np.median([u["utt_voiced_db"] for u in train_utts]))

    train_payload = {
        "schema": "en_stage10_tokens_v1",
        "corpus": corpus_name,
        "speaker": speaker_name,
        "sr": SR,
        "hop": HOP,
        "n_fft": N_FFT,
        "n_cep": N_CEP,
        "n_bap": N_BAP,
        "n_bins": N_BINS,
        "bap_scale": 0.50,
        "art_classes": ART_CLASSES,
        "n_utts": len(train_utts),
        "train_ids": [u["id"] for u in train_utts],
        "heldout_ids": sorted(heldout_set),
        "n_tokens": n_tok_train,
        "n_strict_ok": n_ok_train,
        "strict_accept_rate": round(train_accept, 4),
        "quarantine_reasons": dict(reason_train),
        "speaker_f0_med_hz": round(train_f0_med, 2),
        "ref_log_amp": round(train_log_ref, 5),
        "ref_voiced_db": round(train_voiced_db, 3),
        "sil_cep64": (sil_cep_sum / max(1, sil_cnt_tot)).astype(np.float32),
        "sil_bap6": (sil_bap_sum / max(1, sil_cnt_tot)).astype(np.float32),
        "utterances": train_utts,
        "tokens": flat_train_tokens,
    }

    held_payload = {
        "schema": "en_stage10_heldout_v1",
        "corpus": corpus_name,
        "speaker": speaker_name,
        "sr": SR,
        "hop": HOP,
        "n_fft": N_FFT,
        "n_cep": N_CEP,
        "n_bap": N_BAP,
        "n_bins": N_BINS,
        "bap_scale": 0.50,
        "n_utts": len(held_utts),
        "heldout_ids": [u["id"] for u in held_utts],
        "n_tokens": n_tok_held,
        "n_strict_ok": n_ok_held,
        "strict_accept_rate": round(held_accept, 4),
        "utterances": held_utts,
        "tokens": flat_held_tokens,
    }

    TRAIN_TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_TOKENS_PATH, "wb") as f:
        pickle.dump(train_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(HELDOUT_PATH, "wb") as f:
        pickle.dump(held_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Extract `en_source_profile.json` strictly from clean TRAIN clips
    src_clips = _select_clean_source_clips(train_recs, n_clips=36 if len(train_recs) >= 1500 else 25)
    assert not ({r["id"] for r in src_clips} & heldout_set), "Leakage in en_source_profile!"
    src_waves = [r["wav"].astype(np.float64) for r in src_clips]
    prof_name = "LJSPEECH_1_1_TRAIN" if "ljspeech" in corpus_name.lower() else "CMU_ARCTIC_SLT_TRAIN"
    prof_id = "en_ljspeech_stage11_source" if "ljspeech" in corpus_name.lower() else "en_slt_stage10_source"
    prof = extract_clone_profile(
        src_waves,
        name=prof_name,
        profile_id=prof_id,
        orig_sr_hint=22050 if "ljspeech" in corpus_name.lower() else SR,
    )
    prof_doc = prof.to_dict()
    prof_doc["corpus"] = corpus_name
    prof_doc["n_train"] = len(train_utts)
    prof_doc["n_heldout"] = len(held_utts)
    prof_doc["source_clip_ids"] = [r["id"] for r in src_clips]
    SOURCE_PROFILE_PATH.write_text(
        json.dumps(prof_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "corpus": corpus_name,
        "speaker": speaker_name,
        "manifest_path": str(manifest_path),
        "train_utterances": len(train_utts),
        "heldout_utterances": len(held_utts),
        "zero_leakage_verified": True,
        "train_tokens": n_tok_train,
        "train_strict_ok": n_ok_train,
        "train_accept_rate": round(train_accept, 4),
        "heldout_tokens": n_tok_held,
        "heldout_strict_ok": n_ok_held,
        "heldout_accept_rate": round(held_accept, 4),
        "quarantine_reasons": dict(reason_train),
        "speaker_f0_med_hz": round(train_f0_med, 2),
        "source_profile": prof.summary_dict(),
        "paths": {
            "train_tokens": str(TRAIN_TOKENS_PATH),
            "heldout": str(HELDOUT_PATH),
            "source_profile": str(SOURCE_PROFILE_PATH),
        },
        "sizes_mb": {
            "train_tokens_mb": round(TRAIN_TOKENS_PATH.stat().st_size / (1024 * 1024), 2),
            "heldout_mb": round(HELDOUT_PATH.stat().st_size / (1024 * 1024), 2),
            "source_profile_kb": round(SOURCE_PROFILE_PATH.stat().st_size / 1024, 2),
        },
        "total_time_s": round(time.perf_counter() - t0, 2),
    }
    ALIGN_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 10/11 EN TrueHarmEnv Alignment & Source Profile")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--n-em", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 8))
    args = parser.parse_args()
    run_stage10_alignment(manifest_path=args.manifest, n_em=args.n_em, workers=args.workers)


if __name__ == "__main__":
    main()
