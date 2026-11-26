"""Parallel Feature Extractor, Acoustic Valley Cleaner Selection & SIL-Aware Mandarin GMM-HMM Aligner (`exp_v14_fluent_zh_tts`)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v9_clarity_pro.boundary_refiner import compute_landmark_tracks  # noqa: E402
from exp_v11_align_gmm.gmm_hmm_aligner import (  # noqa: E402
    _PhoneGMMModel,
    _build_phone_boundaries,
    _manner_group,
    _substate_count,
    _substate_logical_map,
    extract_29d_features,
)
from exp_v12_natural_vocoder.v12_bank import _patch_unsmoothed_uv_env  # noqa: E402
from exp_v12_natural_vocoder.vocoder_v12 import analyze_v12  # noqa: E402
from exp_v13_utau_ceiling.zh_bank_v13 import _refine_mandarin_f_split  # noqa: E402
from exp_v14_fluent_zh_tts import (  # noqa: E402
    ALIGNMENTS_V14,
    CLEAN_10_IDS,
    FEATS_CACHE_V14,
    HOP,
    MANIFEST_V14,
    N_FFT,
    SR,
    USER_6_IDS,
)
from pybot_tts_v2_recorded import build_extended_index as bei  # noqa: E402


def _fast_trim_bounds_db(x: np.ndarray, sr: int = SR, hop: int = HOP, db_thresh: float = -40.0) -> tuple[int, int]:
    """Conservative energy trim preserving weak unvoiced onsets and soft trailing breaths."""
    n = len(x)
    if n <= hop * 4:
        return 0, n
    win = 240
    half = win // 2
    n_fr = 1 + max(0, n - 1) // hop
    xp = np.pad(x.astype(np.float64), (half, half + win))
    starts = np.arange(n_fr) * hop
    idx = starts[:, None] + np.arange(win)[None, :]
    frames = xp[idx]
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-16)
    db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    pk = float(np.percentile(db, 95))
    act = np.where((db - pk) > db_thresh)[0]
    if act.size == 0:
        return 0, n
    f0 = max(0, int(act[0]) - 6)
    f1 = min(n_fr, int(act[-1]) + 7)
    return int(f0 * hop), int(min(n, f1 * hop))


def _extract_one_utt(entry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Worker-safe per-utterance DSP & 29-dim alignment feature extraction."""
    uid = entry["id"]
    wav_p = Path(entry["path"])
    if not wav_p.is_absolute():
        wav_p = REPO_ROOT / wav_p
    x, fsr = vocoder.read_wav_mono(wav_p)
    if fsr != SR:
        raise ValueError(f"Expected {SR} Hz for {uid}, got {fsr}")
    s0, s1 = _fast_trim_bounds_db(x, sr=SR, hop=HOP, db_thresh=-40.0)
    xa = np.asarray(x[s0:s1], dtype=np.float64)

    fv12 = analyze_v12(xa, sr=SR)
    feats = _patch_unsmoothed_uv_env(xa, fv12.feats, sr=SR, n_fft=N_FFT)
    X29 = extract_29d_features(xa, feats, sr=SR, hop=HOP)

    env_pwr = np.exp(2.0 * feats.log_env)
    tot_db = 10.0 * np.log10(np.maximum(np.mean(env_pwr, axis=1), 1e-16))
    tot_pk = float(np.percentile(tot_db, 95)) if tot_db.size else 0.0
    rel_db = (tot_db - tot_pk).astype(np.float32)

    return uid, {
        "xa": xa.astype(np.float32),
        "f0_hz": feats.f0_hz.astype(np.float32),
        "voiced": feats.voiced.astype(bool),
        "log_env_f16": feats.log_env.astype(np.float16),
        "bap": feats.bap.astype(np.float32),
        "bap_gt8k": fv12.bap_gt8k.astype(np.float32),
        "coherent_phasor": fv12.coherent_phasor,
        "X29_f16": X29.astype(np.float16),
        "rel_db": rel_db,
        "trim_start_sample": int(s0),
        "trim_stop_sample": int(s1),
    }


def extract_or_load_ssb0502_features(
    manifest_utts: list[dict[str, Any]],
    *,
    cache_path: Path = FEATS_CACHE_V14,
    max_workers: int | None = None,
    force_rebuild: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load or parallel-extract DSP + 29-dim GMM features for all `SSB0502` utterances."""
    cache: dict[str, dict[str, Any]] = {}
    if not force_rebuild and cache_path.is_file():
        try:
            loaded = pickle.loads(cache_path.read_bytes())
            if isinstance(loaded, dict) and loaded.get("version") == 1:
                cache = dict(loaded.get("items", {}))
        except Exception:
            cache = {}

    todo = [u for u in manifest_utts if u["id"] not in cache]
    if todo:
        workers = max_workers or min(12, max(2, (os.cpu_count() or 4) // 2))
        if workers > 1 and len(todo) >= 8:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_extract_one_utt, u) for u in todo]
                for fut in as_completed(futs):
                    uid, item = fut.result()
                    cache[uid] = item
        else:
            for u in todo:
                uid, item = _extract_one_utt(u)
                cache[uid] = item

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps({"version": 1, "items": cache}, protocol=pickle.HIGHEST_PROTOCOL))

    return {u["id"]: cache[u["id"]] for u in manifest_utts if u["id"] in cache}


def measure_max_internal_quiet_dip_ms(feat_entry: dict[str, Any]) -> float:
    """Measure longest unvoiced quiet valley strictly inside active speech (zero transcript leakage)."""
    rel_db = np.asarray(feat_entry["rel_db"], dtype=np.float64)
    voi = np.asarray(feat_entry["voiced"], dtype=bool)
    act = np.where(voi | (rel_db > -24.0))[0]
    if act.size < 4:
        return 999.0
    a0, a1 = int(act[0]), int(act[-1]) + 1
    quiet = (~voi[a0:a1]) & (rel_db[a0:a1] < -24.0)
    max_run = 0
    cur = 0
    for q in quiet:
        if q:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return round(max_run * HOP * 1000.0 / SR, 2)


def select_clean_10_ids(
    manifest_utts: list[dict[str, Any]],
    feats_map: dict[str, dict[str, Any]],
    *,
    exclude_ids: set[str] = set(USER_6_IDS),
    n_pick: int = 10,
) -> tuple[str, ...]:
    """Deterministically select 10 clean, fluent, 10-15 syllable `SSB0502` utterances from raw energy scan."""
    cands: list[tuple[float, float, str]] = []
    for u in manifest_utts:
        uid = u["id"]
        if uid in exclude_ids or uid not in feats_map:
            continue
        n_syl = len(u.get("pinyin", []))
        if not (10 <= n_syl <= 15):
            continue
        fe = feats_map[uid]
        dip_ms = measure_max_internal_quiet_dip_ms(fe)
        if dip_ms > 120.0:
            continue
        rel_db = np.asarray(fe["rel_db"], dtype=np.float64)
        voi = np.asarray(fe["voiced"], dtype=bool)
        act = np.where(voi | (rel_db > -30.0))[0]
        if act.size < 10:
            continue
        act_ms = float(act[-1] - act[0] + 1) * 5.0
        syl_rate_ms = act_ms / max(1, n_syl)
        if not (140.0 <= syl_rate_ms <= 265.0):
            continue
        cands.append((dip_ms, abs(syl_rate_ms - 195.0), uid))

    # Sort deterministically by uid among top fluent candidates (pick evenly spaced indices)
    cands.sort(key=lambda x: (round(x[0] / 25.0), x[1], x[2]))
    pool_ids = sorted(x[2] for x in cands[: max(n_pick * 3, 30)])
    if len(pool_ids) < n_pick:
        raise RuntimeError(f"Only found {len(pool_ids)} clean candidates for CLEAN_10_IDS")
    idxs = np.linspace(0, len(pool_ids) - 1, n_pick).astype(int)
    return tuple(pool_ids[i] for i in idxs)


def _to_vocoder_features(fe: dict[str, Any]) -> vocoder.VocoderFeatures:
    log_env = np.asarray(fe["log_env_f16"], dtype=np.float64)
    xa = fe["xa"]
    return vocoder.VocoderFeatures(
        f0_hz=np.asarray(fe["f0_hz"], dtype=np.float64),
        voiced=np.asarray(fe["voiced"], dtype=bool),
        log_env=log_env,
        bap=np.asarray(fe["bap"], dtype=np.float64),
        sr=SR,
        hop_samples=HOP,
        n_samples=len(xa),
        n_fft=N_FFT,
    )


def _seed_init_bounds_and_ipas(
    u: dict[str, Any],
    fe: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]], list[tuple[int, int, int]], list[dict[str, Any]]]:
    """Build initial syllable/phone spans from energy dips so GMM EM starts cleanly inside speech."""
    xa = np.asarray(fe["xa"], dtype=np.float64)
    n_fr = len(fe["log_env_f16"])
    rel_db = np.asarray(fe["rel_db"], dtype=np.float64)
    voi = np.asarray(fe["voiced"], dtype=bool)
    act = np.where(voi | (rel_db > -30.0))[0]
    a0 = int(act[0]) * HOP if act.size else 0
    a1 = min(len(xa), (int(act[-1]) + 1) * HOP) if act.size else len(xa)
    pys = u["pinyin"]
    spans, _ = bei.syllable_spans(xa.tolist(), (a0, a1), pys)

    ipas: list[str] = []
    meta_phones: list[dict[str, Any]] = []
    init_bounds: list[tuple[int, int, int]] = []
    syl_meta: list[dict[str, Any]] = []

    for idx_s, (py, (s0, s1)) in enumerate(zip(pys, spans)):
        onset, final, tone, syl_ipas = cf.pinyin_to_ipa(py)
        f_lo = max(0, min(n_fr - 1, int(s0 // HOP)))
        f_hi = max(f_lo + 1, min(n_fr, int(math.ceil(s1 / HOP))))
        edges = np.linspace(f_lo, f_hi, len(syl_ipas) + 1).astype(int)
        p_start_idx = len(ipas)
        for k, p in enumerate(syl_ipas):
            ipas.append(p)
            ps_k = int(edges[k])
            pe_k = int(max(ps_k + 1, edges[k + 1]))
            init_bounds.append((ps_k, (ps_k + pe_k) // 2, pe_k))
            meta_phones.append(
                {
                    "ipa": p,
                    "arpabet": None,
                    "word": u["text"][idx_s],
                    "syllable_idx": idx_s,
                    "stress_or_tone": f"T{tone}",
                    "is_onset": (k == 0 and bool(onset)),
                }
            )
        syl_meta.append(
            {
                "index": idx_s,
                "hanzi": u["text"][idx_s],
                "pinyin": py,
                "onset": onset,
                "final": final,
                "tone": int(tone),
                "p_slice": (p_start_idx, len(ipas)),
            }
        )
    return ipas, meta_phones, init_bounds, syl_meta


def _score_sil_emit(
    gmm: _PhoneGMMModel,
    X: np.ndarray,
    rel_db: np.ndarray,
    voiced: np.ndarray,
) -> np.ndarray:
    """Compute acoustic log-likelihood for the `SIL` state gated cleanly by quiet unvoiced energy."""
    if ("SIL", 0) in gmm.params:
        sil_scores = gmm.score_sequence(X, [("SIL", 0)])[0].copy()
    else:
        # Bootstrap prior: quiet unvoiced frames match SIL, loud/voiced frames strongly disfavor SIL
        sil_scores = np.where((~voiced) & (rel_db < -24.0), -18.0, -140.0)
    # Hard gate: voiced frames or > -22 dB active bursts cannot be swallowed by SIL
    disallow = voiced | (rel_db > -22.0)
    sil_scores[disallow] = -140.0
    # Extra mild bonus for deep silence so 70+ ms hesitations overcome min-SIL barrier
    deep = (~voiced) & (rel_db < -27.0)
    sil_scores[deep] += 4.5
    return sil_scores


def _viterbi_bakis_with_opt_sil(
    emit_phones: np.ndarray,
    emit_sil: np.ndarray,
    phone_subcounts: list[int],
    self_bonus: np.ndarray,
    syl_phone_ranges: list[tuple[int, int]],
    *,
    min_inter_sil: int = 12,     # 60 ms min inter-syllable pause (blocks stop closures!)
    min_edge_sil: int = 2,       # 10 ms min leading/trailing silence (trims trailing quiet fade!)
    inter_entry_pen: float = -12.0,
    edge_entry_pen: float = -3.0,
) -> tuple[list[tuple[int, int, int]], list[int]]:
    """Left-to-right Bakis Viterbi with skip-able edge & inter-syllable `SIL` chains.

    Returns:
      phone_bounds: [(p_start, p_mid, p_end), ...] matching speech phones
      inter_sil_frames: length-(len(syl_phone_ranges) + 1) list where entry 0 is leading SIL,
        1..S-1 are inter-syllable pauses after syllables 0..S-2, and S is trailing SIL frames.
    """
    m_sp, t_n = emit_phones.shape
    n_syl = len(syl_phone_ranges)
    if m_sp == 0 or t_n < m_sp:
        n_p = len(phone_subcounts)
        edges = np.linspace(0, t_n, n_p + 1).astype(int)
        out_b = [(int(edges[i]), int((edges[i] + max(edges[i] + 1, edges[i + 1])) // 2), int(max(edges[i] + 1, edges[i + 1]))) for i in range(n_p)]
        return out_b, [0] * (n_syl + 1)

    # Map each phone index -> speech backbone state range [s_lo, s_hi]
    phone_state_spans: list[tuple[int, int]] = []
    cur = 0
    for nc in phone_subcounts:
        phone_state_spans.append((cur, cur + nc - 1))
        cur += nc

    # Build augmented state table: interleave speech states and optional SIL chains
    # State descriptor: kind="sp"|"sil", emit_row (idx in emit_phones or -1 for sil), slot_idx (for sil)
    state_emits: list[np.ndarray] = []
    stay_pen_l: list[float] = []
    # transitions: predecessors list of (prev_state_idx, trans_pen)
    preds: list[list[tuple[int, float]]] = []
    sp_to_aug: dict[int, int] = {}
    sil_slot_aug_ids: list[list[int]] = []

    def _add_sil_chain(n_chain: int, slot_idx: int, is_edge: bool) -> tuple[int, int]:
        ids = []
        for c_i in range(n_chain):
            aug_id = len(state_emits)
            state_emits.append(emit_sil)
            # Only the final substate of a SIL chain can self-loop (guarantees min duration n_chain!)
            stay_pen_l.append(-0.15 if (c_i == n_chain - 1) else -1e9)
            preds.append([])
            if c_i == 0:
                pass  # entry preds attached by caller
            else:
                preds[aug_id].append((aug_id - 1, -0.05))
            if c_i == n_chain - 1:
                preds[aug_id].append((aug_id, -0.15))
            ids.append(aug_id)
        sil_slot_aug_ids.append(ids)
        return ids[0], ids[-1]

    # Leading SIL (slot 0)
    lead_s0, lead_s1 = _add_sil_chain(min_edge_sil, 0, is_edge=True)
    # First speech state can start directly at frame 0 (handled below) OR from lead_s1

    prev_syl_end_aug: int | None = None
    prev_sil_end_aug: int | None = lead_s1

    for s_idx, (pa, pb) in enumerate(syl_phone_ranges):
        sp_lo = phone_state_spans[pa][0]
        sp_hi = phone_state_spans[pb - 1][1]
        for sp_s in range(sp_lo, sp_hi + 1):
            aug_id = len(state_emits)
            sp_to_aug[sp_s] = aug_id
            state_emits.append(emit_phones[sp_s])
            st_p = -0.35 + float(self_bonus[sp_s])
            stay_pen_l.append(st_p)
            pr = [(aug_id, st_p)]  # self loop
            if sp_s == sp_lo:
                if prev_syl_end_aug is not None:
                    pr.append((prev_syl_end_aug, -0.85))  # fluent direct advance (skip inter-SIL)
                if prev_sil_end_aug is not None:
                    pr.append((prev_sil_end_aug, -0.85))  # resume after SIL
            else:
                pr.append((aug_id - 1, -0.85))
            preds.append(pr)

        cur_syl_end_aug = sp_to_aug[sp_hi]
        is_last_syl = (s_idx == n_syl - 1)
        sil_len = min_edge_sil if is_last_syl else min_inter_sil
        entry_p = edge_entry_pen if is_last_syl else inter_entry_pen
        sil_s0, sil_s1 = _add_sil_chain(sil_len, s_idx + 1, is_edge=is_last_syl)
        preds[sil_s0].append((cur_syl_end_aug, entry_p))
        prev_syl_end_aug = cur_syl_end_aug
        prev_sil_end_aug = sil_s1

    n_aug = len(state_emits)
    E = np.stack(state_emits, axis=0)  # (n_aug, T)

    # Compact sparse predecessor lists (max fan-in <= 3!) into vectorized lookup arrays
    max_deg = max(len(pr) for pr in preds)
    pred_idx = np.zeros((n_aug, max_deg), dtype=np.int32)
    pred_pen = np.full((n_aug, max_deg), -1e15, dtype=np.float64)
    for st_i, pr in enumerate(preds):
        for k, (ps, pen) in enumerate(pr):
            pred_idx[st_i, k] = ps
            pred_pen[st_i, k] = pen

    dp_prev = np.full(n_aug, -1e15, dtype=np.float64)
    # Initial states allowed at t=0: either lead_s0 (with edge_entry_pen) or first speech state sp_to_aug[0]
    dp_prev[lead_s0] = E[lead_s0, 0] + edge_entry_pen
    dp_prev[sp_to_aug[0]] = E[sp_to_aug[0], 0]

    bp = np.zeros((t_n, n_aug), dtype=np.int32)
    for t in range(1, t_n):
        cand_vals = dp_prev[pred_idx] + pred_pen  # (n_aug, max_deg)
        best_k = np.argmax(cand_vals, axis=1)
        best_val = cand_vals[np.arange(n_aug), best_k]
        dp_cur = best_val + E[:, t]
        bp[t] = pred_idx[np.arange(n_aug), best_k]
        dp_prev = dp_cur

    # Allowed terminal states at t_n - 1: last speech state or trailing SIL end
    term_cands = [prev_syl_end_aug, prev_sil_end_aug]
    best_term = max(term_cands, key=lambda s: float(dp_prev[s]) if s is not None else -1e15)

    path = np.empty(t_n, dtype=np.int32)
    cur_s = int(best_term)
    for t in range(t_n - 1, -1, -1):
        path[t] = cur_s
        if t > 0:
            cur_s = int(bp[t, cur_s])

    # Decode phone bounds & SIL counts
    aug_to_sil_slot: dict[int, int] = {}
    for sl_i, ids in enumerate(sil_slot_aug_ids):
        for aid in ids:
            aug_to_sil_slot[aid] = sl_i

    sil_counts = [0] * len(sil_slot_aug_ids)
    for aid in path:
        sl_i = aug_to_sil_slot.get(int(aid), -1)
        if sl_i >= 0:
            sil_counts[sl_i] += 1

    bounds: list[tuple[int, int, int]] = []
    for sp_lo, sp_hi in phone_state_spans:
        aug_lo = sp_to_aug[sp_lo]
        aug_hi = sp_to_aug[sp_hi]
        frames_p = np.where((path >= aug_lo) & (path <= aug_hi))[0]
        if frames_p.size == 0:
            prev_e = bounds[-1][2] if bounds else 0
            bounds.append((prev_e, prev_e, min(t_n, prev_e + 1)))
        else:
            p_start = int(frames_p[0])
            p_end = int(frames_p[-1]) + 1
            if sp_hi - sp_lo >= 2:
                core_frames = np.where((path >= sp_to_aug[sp_lo + 1]) & (path <= sp_to_aug[sp_hi - 1]))[0]
            else:
                core_frames = frames_p
            p_mid = int(round(float(np.median(core_frames)))) if core_frames.size else ((p_start + p_end) // 2)
            p_mid = max(p_start, min(p_end - 1, p_mid))
            bounds.append((p_start, p_mid, p_end))

    return bounds, sil_counts


def _extract_aligned_utterance_entry(
    rec: dict[str, Any],
    fe: dict[str, Any],
    bounds: list[tuple[int, int, int]],
    sil_counts: list[int],
) -> dict[str, Any]:
    """Refine onset/rime splits & zero-onset boundaries and serialize alignment dict."""
    feats = _to_vocoder_features(fe)
    xa64 = np.asarray(fe["xa"], dtype=np.float64)
    tracks = compute_landmark_tracks(xa64, feats, sr=SR)
    pbs = _build_phone_boundaries(rec["meta_phones"], bounds, xa64, feats, sr=SR, hop=HOP)
    n_tot = len(feats.log_env)

    _UV_OBSTRUENTS = {"p", "t", "k", "f", "s", "sh", "x", "h", "c", "ch", "q", "z", "zh", "j"}
    n_syls_rec = len(rec["syl_meta"])
    syl_rows: list[dict[str, Any]] = []
    for idx_s, sm in enumerate(rec["syl_meta"]):
        a, b = sm["p_slice"]
        spbs = pbs[a:b]
        s_start = int(spbs[0].p_start)
        s_end = int(spbs[-1].p_end)
        o = sm["onset"]

        # Fix A: Slide unvoiced-obstruent onset past leading voiced overhang captured from prior syllable
        if o in _UV_OBSTRUENTS and (s_end - s_start) >= 8 and tracks["voi"][s_start] > 0.5:
            max_k = min(11, s_end - s_start - 5)
            for k in range(1, max_k):
                if tracks["voi"][s_start + k] < 0.5 and tracks["voi"][s_start + k + 1] < 0.5:
                    s_start += k
                    if syl_rows and syl_rows[-1]["pause_after_ms"] == 0.0:
                        syl_rows[-1]["syl_end"] = s_start
                        prev_dur_fr = s_start - syl_rows[-1]["syl_start"]
                        syl_rows[-1]["dur_ms"] = round(prev_dur_fr * HOP * 1000.0 / SR, 2)
                        if syl_rows[-1]["f_split"] > 0 and syl_rows[-1]["f_split"] >= prev_dur_fr - 2:
                            syl_rows[-1]["f_split"] = max(2, prev_dur_fr - 3)
                    break

        # Fix B: Trim sentence-final weak trailing breath / room echo when followed by sustained drop
        if idx_s == n_syls_rec - 1 and (s_end - s_start) >= 48:
            strong_v = np.where((tracks["voi"][s_start:s_end] > 0.5) & (tracks["rel_db"][s_start:s_end] > -9.5))[0]
            if strong_v.size >= 4:
                last_strong = s_start + int(strong_v[-1])
                cand_end = last_strong + 1
                if s_end - cand_end >= 4:
                    post_seg = tracks["rel_db"][cand_end : min(s_end, cand_end + 8)]
                    if post_seg.size > 0 and float(np.mean(post_seg)) < -11.5:
                        s_end = cand_end

        # Raw GMM f_split relative to updated syllable start
        raw_sp = max(2, int(spbs[0].p_end) - s_start) if (o and len(spbs) >= 2) else 0
        f_sp_rel = _refine_mandarin_f_split(tracks, s_start, s_end, o, raw_sp) if (o and s_end - s_start >= 4) else 0

        # Tighten zero-onset syllable entry against unvoiced weak gap
        if not o and idx_s > 0 and syl_rows and syl_rows[-1]["pause_after_ms"] == 0.0 and (s_end - s_start) >= 8:
            core_v_db = float(np.median(tracks["lf_db"][min(s_end - 1, s_start + 3) : s_end]))
            while (s_end - s_start) >= 8 and s_start < s_end - 6 and tracks["voi"][s_start] < 0.5 and tracks["lf_db"][s_start] < core_v_db - 10.0:
                s_start += 1
                syl_rows[-1]["syl_end"] = s_start
                syl_rows[-1]["dur_ms"] = round((s_start - syl_rows[-1]["syl_start"]) * HOP * 1000.0 / SR, 2)

        pause_after_fr = int(sil_counts[idx_s + 1]) if (idx_s + 1) < (len(sil_counts) - 1) else 0
        syl_rows.append(
            {
                "index": int(sm["index"]),
                "hanzi": sm["hanzi"],
                "pinyin": sm["pinyin"],
                "onset": o,
                "final": sm["final"],
                "tone": int(sm["tone"]),
                "syl_start": max(0, s_start),
                "f_split": int(f_sp_rel),
                "syl_end": min(n_tot, max(s_start + 1, s_end)),
                "dur_ms": round(max(1, s_end - s_start) * HOP * 1000.0 / SR, 2),
                "pause_after_ms": round(pause_after_fr * HOP * 1000.0 / SR, 2),
            }
        )

    return {
        "id": rec["id"],
        "speaker": "SSB0502",
        "text": rec["text"],
        "n_frames": int(n_tot),
        "trim_start_sample": int(fe["trim_start_sample"]),
        "trim_stop_sample": int(fe["trim_stop_sample"]),
        "lead_sil_ms": round(int(sil_counts[0]) * HOP * 1000.0 / SR, 2),
        "tail_sil_ms": round(int(sil_counts[-1]) * HOP * 1000.0 / SR, 2),
        "syllables": syl_rows,
    }


def _recompute_X_and_sil_pools(
    train_recs: list[dict[str, Any]],
    feats_map: dict[str, dict[str, Any]],
    cur_bounds: dict[str, list[tuple[int, int, int]]],
    cur_sils: dict[str, list[int]],
) -> _PhoneGMMModel:
    gmm = _PhoneGMMModel(dim=29, var_floor=0.14)
    phone_pools: dict[tuple[str, int], list[np.ndarray]] = {}
    manner_pools: dict[tuple[str, int], list[np.ndarray]] = {}
    sil_vecs: list[np.ndarray] = []

    for rec in train_recs:
        uid = rec["id"]
        fe = feats_map[uid]
        X = np.asarray(fe["X29_f16"], dtype=np.float64)
        rel_db = np.asarray(fe["rel_db"], dtype=np.float64)
        voi = np.asarray(fe["voiced"], dtype=bool)
        q_mask = (~voi) & (rel_db < -24.0)
        if np.any(q_mask):
            sil_vecs.extend(X[q_mask])

        for ipa, (ps, _pm, pe) in zip(rec["ipas"], cur_bounds[uid]):
            if pe <= ps:
                continue
            mg = _manner_group(ipa)
            seg = X[ps:pe]
            t_seg = len(seg)
            if t_seg >= 3:
                k1 = max(1, t_seg // 4)
                k2 = max(k1 + 1, (3 * t_seg) // 4)
                chunks = (seg[:k1], seg[k1:k2], seg[k2:])
            else:
                chunks = (seg, seg, seg)
            for st_idx, ch in enumerate(chunks):
                if len(ch) > 0:
                    phone_pools.setdefault((ipa, st_idx), []).extend(ch)
                    manner_pools.setdefault((mg, st_idx), []).extend(ch)

    gmm.fit_from_pools(phone_pools, manner_pools)
    if len(sil_vecs) >= 24:
        gmm.params[("SIL", 0)] = gmm._fit_2gmm_raw(np.stack(sil_vecs, axis=0), prior=None, shrink_n=0.0)
    return gmm


def _align_rec_with_gmm(
    gmm: _PhoneGMMModel,
    rec: dict[str, Any],
    fe: dict[str, Any],
) -> tuple[list[tuple[int, int, int]], list[int]]:
    X = np.asarray(fe["X29_f16"], dtype=np.float64)
    rel_db = np.asarray(fe["rel_db"], dtype=np.float64)
    voi = np.asarray(fe["voiced"], dtype=bool)
    ipas = rec["ipas"]
    subcounts = [_substate_count(p) for p in ipas]
    if sum(subcounts) > len(X):
        subcounts = [3 if len(X) >= 3 * len(ipas) else 1 for _ in ipas]
    seq_keys: list[tuple[str, int]] = []
    self_b: list[float] = []
    for p, ns in zip(ipas, subcounts):
        for ls in _substate_logical_map(ns):
            seq_keys.append((p, ls))
            self_b.append(0.15 if ls == 1 else -0.05)

    emit_phones = gmm.score_sequence(X, seq_keys)
    emit_sil = _score_sil_emit(gmm, X, rel_db, voi)
    return _viterbi_bakis_with_opt_sil(
        emit_phones,
        emit_sil,
        subcounts,
        np.asarray(self_b, dtype=np.float64),
        rec["syl_ranges"],
    )


def train_and_align_ssb0502_v14(
    *,
    manifest_path: Path = MANIFEST_V14,
    cache_path: Path = FEATS_CACHE_V14,
    out_align_path: Path = ALIGNMENTS_V14,
    n_em: int = 5,
    extra_exclude_ids: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Strict zero-leakage SIL-aware GMM-HMM training (`train_ids`) + held-out alignment."""
    t0 = time.perf_counter()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    utts = manifest["utterances"]
    feats_map = extract_or_load_ssb0502_features(utts, cache_path=cache_path)
    t_feats = time.perf_counter() - t0

    clean_10 = tuple(CLEAN_10_IDS) if CLEAN_10_IDS else tuple(manifest.get("clean_10_ids") or ())
    if len(clean_10) != 10:
        clean_10 = select_clean_10_ids(utts, feats_map, exclude_ids=set(USER_6_IDS), n_pick=10)
        manifest["clean_10_ids"] = list(clean_10)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    heldout_set = set(USER_6_IDS) | set(clean_10) | set(extra_exclude_ids)
    train_recs: list[dict[str, Any]] = []
    held_recs: list[dict[str, Any]] = []
    cur_bounds: dict[str, list[tuple[int, int, int]]] = {}
    cur_sils: dict[str, list[int]] = {}

    for u in utts:
        uid = u["id"]
        fe = feats_map[uid]
        ipas, meta_phones, init_bounds, syl_meta = _seed_init_bounds_and_ipas(u, fe)
        syl_ranges = [sm["p_slice"] for sm in syl_meta]
        rec = {
            "id": uid,
            "text": u["text"],
            "ipas": ipas,
            "meta_phones": meta_phones,
            "init_bounds": init_bounds,
            "syl_meta": syl_meta,
            "syl_ranges": syl_ranges,
        }
        cur_bounds[uid] = init_bounds
        cur_sils[uid] = [0] * (len(syl_ranges) + 1)
        if uid in heldout_set:
            held_recs.append(rec)
        else:
            train_recs.append(rec)

    t1 = time.perf_counter()
    gmm: _PhoneGMMModel | None = None
    for it in range(n_em + 1):
        gmm = _recompute_X_and_sil_pools(train_recs, feats_map, cur_bounds, cur_sils)
        if it == n_em:
            break
        for rec in train_recs:
            b, sc = _align_rec_with_gmm(gmm, rec, feats_map[rec["id"]])
            cur_bounds[rec["id"]] = b
            cur_sils[rec["id"]] = sc

    assert gmm is not None
    for rec in held_recs:
        b, sc = _align_rec_with_gmm(gmm, rec, feats_map[rec["id"]])
        cur_bounds[rec["id"]] = b
        cur_sils[rec["id"]] = sc
    t_em = time.perf_counter() - t1

    zh_train: dict[str, Any] = {}
    zh_held: dict[str, Any] = {}
    all_durs: list[float] = []
    inter_pauses_n = 0
    inter_slots_n = 0

    for rec in train_recs:
        uid = rec["id"]
        entry = _extract_aligned_utterance_entry(rec, feats_map[uid], cur_bounds[uid], cur_sils[uid])
        zh_train[uid] = entry
        for s in entry["syllables"][:-1]:
            inter_slots_n += 1
            if s["pause_after_ms"] > 0.0:
                inter_pauses_n += 1
        all_durs.extend(s["dur_ms"] for s in entry["syllables"])

    for rec in held_recs:
        uid = rec["id"]
        zh_held[uid] = _extract_aligned_utterance_entry(rec, feats_map[uid], cur_bounds[uid], cur_sils[uid])

    stats = {
        "n_train_utts": len(zh_train),
        "n_heldout_utts": len(zh_held),
        "train_syl_dur_mean_ms": round(float(np.mean(all_durs)), 2),
        "train_syl_dur_p95_ms": round(float(np.percentile(all_durs, 95)), 2),
        "train_syl_dur_max_ms": round(float(np.max(all_durs)), 2),
        "inter_word_pause_rate": round(inter_pauses_n / max(1, inter_slots_n), 4),
        "inter_pauses_count": inter_pauses_n,
        "inter_slots_total": inter_slots_n,
        "feats_time_s": round(t_feats, 2),
        "gmm_em_time_s": round(t_em, 2),
    }
    doc = {
        "schema": 1,
        "n_em": n_em,
        "user_6_ids": list(USER_6_IDS),
        "clean_10_ids": list(clean_10),
        "excluded_from_training": sorted(heldout_set),
        "stats": stats,
        "zh_train": zh_train,
        "zh_heldout": zh_held,
        "zh_all": {**zh_train, **zh_held},
    }
    out_align_path.parent.mkdir(parents=True, exist_ok=True)
    out_align_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract SSB0502 features & train SIL-aware GMM-HMM aligner.")
    ap.add_argument("--n-em", type=int, default=5)
    args = ap.parse_args()
    doc = train_and_align_ssb0502_v14(n_em=args.n_em)
    print("Alignment stats:", json.dumps(doc["stats"], indent=2))
    print("clean_10_ids:", doc["clean_10_ids"])
    for uid in USER_6_IDS:
        row = doc["zh_heldout"][uid]
        s_str = " | ".join(
            f"{s['hanzi']}({s['dur_ms']:.0f}ms fsp={s['f_split']} p={s['pause_after_ms']:.0f})"
            for s in row["syllables"]
        )
        print(f"{uid}: lead={row['lead_sil_ms']:.0f} tail={row['tail_sil_ms']:.0f} | {s_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
