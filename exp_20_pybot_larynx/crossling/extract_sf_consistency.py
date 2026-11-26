"""Stage 12 Same-Speaker Bilingual Consistency Decomposition (`SF_ZH`, `SF_EN`, `Delta_EN_minus_ZH`).

Reads strictly TRAIN aligned `SF_bilingual` data (`sf_bilingual_train_tokens.pkl`, zero leakage):
1. Extracts `sf_zh_profile.json` from pure Mandarin intervals of ~120 TRAIN utterances.
2. Extracts `sf_en_profile.json` from pure English intervals of clean TRAIN utterances.
3. Decomposes true same-speaker language-intrinsic differences (`EN - ZH`) with ZERO
   speaker/microphone/studio confound into `sf_bilingual_consistency.json`:
   - paired per-utterance F0 shift & F0 IQR ratio
   - paired per-utterance voiced RMS loudness difference
   - paired 6-band BAP difference (`delta_bap6_en_minus_zh`)
   - smooth zero-DC voiced & unvoiced LTAS shape deltas (`delta_voiced_ltas_en_minus_zh_nep`)
   - aligned `ZH -> EN` and `EN -> ZH` switch boundary statistics
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

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

from exp_20_pybot_larynx import EXP_DIR, HOP, N_BINS, SR  # noqa: E402
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB  # noqa: E402
from exp_20_pybot_larynx.clone import extract_clone_profile  # noqa: E402
from exp_20_pybot_larynx.larynx.synth import cep_to_log_env, log_env_to_cep  # noqa: E402
from exp_v6_vocoder import vocoder as v6_voc  # noqa: E402

CROSSLING_DATA_DIR = EXP_DIR / "crossling_data"
SF_TRAIN_TOKENS_PATH = CROSSLING_DATA_DIR / "sf_bilingual_train_tokens.pkl"
SF_ZH_PROFILE_PATH = CROSSLING_DATA_DIR / "sf_zh_profile.json"
SF_EN_PROFILE_PATH = CROSSLING_DATA_DIR / "sf_en_profile.json"
SF_CONSISTENCY_PATH = CROSSLING_DATA_DIR / "sf_bilingual_consistency.json"


def _collect_language_clips(
    utts: list[dict[str, Any]],
    lang: str,
    *,
    target_clips: int = 100,
    min_clip_s: float = 1.15,
    max_clip_s: float = 4.5,
) -> tuple[list[np.ndarray], list[str]]:
    """Collect clean 24 kHz language-specific audio clips by concatenating aligned segments."""
    fade_n = max(4, int(round(0.008 * SR)))
    fade_in = np.linspace(0.0, 1.0, fade_n, dtype=np.float64)
    fade_out = 1.0 - fade_in

    scored: list[tuple[int, dict[str, Any]]] = []
    key_cnt = "n_zh_v" if lang == "zh" else "n_en_v"
    for u in utts:
        nv = int(u.get(key_cnt, 0))
        if nv >= (80 if lang == "zh" else 20):
            scored.append((nv, u))
    scored.sort(key=lambda t: (-t[0], t[1]["id"]))
    step = max(1, len(scored) // (target_clips * 4))
    pool = [u for _, u in scored[::step]]
    if len(pool) < min(len(scored), target_clips * 2):
        pool = [u for _, u in scored]

    clips: list[np.ndarray] = []
    used_ids: list[str] = []
    buf: list[np.ndarray] = []
    buf_len = 0

    for u in pool:
        wav_path = REPO_ROOT / str(u["wav_path"])
        x, file_sr = v6_voc.read_wav_mono(wav_path)
        if file_sr != SR:
            continue
        ts0 = int(u["trim_s0"])
        ts1 = int(u["trim_s1"])
        xa = x[ts0:ts1]

        # Extract contiguous runs of `lang` phones within utterance
        phones = u["phones"]
        segs: list[np.ndarray] = []
        i = 0
        while i < len(phones):
            if phones[i]["lang"] != lang:
                i += 1
                continue
            s_fr = int(phones[i]["p_start"])
            j = i
            while j + 1 < len(phones) and phones[j + 1]["lang"] == lang and int(phones[j + 1]["p_start"]) - int(phones[j]["p_end"]) <= 2:
                j += 1
            e_fr = int(phones[j]["p_end"])
            if e_fr - s_fr >= 6:
                s0 = max(0, s_fr * HOP)
                s1 = min(len(xa), e_fr * HOP)
                if s1 - s0 > fade_n * 3:
                    seg = xa[s0:s1].astype(np.float64).copy()
                    seg[:fade_n] *= fade_in
                    seg[-fade_n:] *= fade_out
                    segs.append(seg)
            i = j + 1

        if not segs:
            continue
        seg_cat = np.concatenate(segs)
        if float(np.sqrt(np.mean(seg_cat ** 2))) < 0.015:
            continue
        used_ids.append(str(u["id"]))
        buf.append(seg_cat)
        buf_len += len(seg_cat)
        if buf_len >= int(round(min_clip_s * SR)):
            clip = np.concatenate(buf)
            if len(clip) > int(round(max_clip_s * SR)):
                clip = clip[: int(round(max_clip_s * SR))]
            clips.append(clip)
            buf = []
            buf_len = 0
            if len(clips) >= target_clips:
                break

    if buf_len >= int(round(0.8 * SR)) and len(clips) < target_clips:
        clips.append(np.concatenate(buf))
    return clips, used_ids


def _summarize_switch_direction(events: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    sub = [e for e in events if e["direction"] == direction]
    if not sub:
        return {"n": 0}
    lv = [float(e["level_diff_db"]) for e in sub]
    sp = [float(e["spec_rmse_db"]) for e in sub]
    f0d = [float(e["f0_diff_st"]) for e in sub if e.get("f0_diff_st") is not None]
    zero_p = [e for e in sub if float(e.get("punct_pause_ms", 0.0)) < 20.0 and int(e.get("gap_frames", 0)) == 0]
    zp_lv = [float(e["level_diff_db"]) for e in zero_p] or lv
    zp_f0 = [float(e["f0_diff_st"]) for e in zero_p if e.get("f0_diff_st") is not None] or f0d
    return {
        "n": len(sub),
        "n_zero_pause": len(zero_p),
        "zero_pause_frac": round(len(zero_p) / max(1, len(sub)), 4),
        "level_diff_db_med": round(float(np.median(lv)), 3),
        "zero_pause_level_diff_db_med": round(float(np.median(zp_lv)), 3),
        "f0_diff_st_med": round(float(np.median(f0d)), 3) if f0d else 0.0,
        "zero_pause_f0_diff_st_med": round(float(np.median(zp_f0)), 3) if zp_f0 else 0.0,
        "spec_rmse_db_med": round(float(np.median(sp)), 3),
    }


def extract_sf_consistency(train_pkl: Path = SF_TRAIN_TOKENS_PATH) -> dict[str, Any]:
    t0 = time.perf_counter()
    with open(train_pkl, "rb") as f:
        tr = pickle.load(f)

    train_ids = set(tr["train_ids"])
    held_ids = set(tr["heldout_ids"])
    assert not (train_ids & held_ids), "Train/heldout overlap detected!"
    utts = list(tr["utterances"])
    switches = list(tr.get("switch_events") or [])

    # 1. Extract SF_ZH and SF_EN VoiceCloneProfiles strictly from aligned same-speaker segments
    zh_clips, zh_uids = _collect_language_clips(utts, "zh", target_clips=90, min_clip_s=1.4)
    en_clips, en_uids = _collect_language_clips(utts, "en", target_clips=90, min_clip_s=1.0)
    print(f"[SF-Consistency] Collected {len(zh_clips)} ZH clips & {len(en_clips)} EN clips from TRAIN.", flush=True)

    zh_prof = extract_clone_profile(
        zh_clips,
        name="SF_BILINGUAL_ZH_TRAIN",
        profile_id="sf_zh_stage12_source",
        orig_sr_hint=22050,
    )
    en_prof = extract_clone_profile(
        en_clips,
        name="SF_BILINGUAL_EN_TRAIN",
        profile_id="sf_en_stage12_source",
        orig_sr_hint=22050,
    )
    for prof, out_p, uids in (
        (zh_prof, SF_ZH_PROFILE_PATH, zh_uids),
        (en_prof, SF_EN_PROFILE_PATH, en_uids),
    ):
        doc = prof.to_dict()
        doc["corpus"] = "SF_bilingual_zh_en"
        doc["n_train_utts"] = len(utts)
        doc["source_clip_ids"] = uids[:50]
        out_p.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 2. Paired same-speaker within-utterance consistency decomposition (EN minus ZH)
    df0_st: list[float] = []
    zh_iqrs: list[float] = []
    en_iqrs: list[float] = []
    iqr_ratios: list[float] = []
    drms_db: list[float] = []
    zh_baps: list[np.ndarray] = []
    en_baps: list[np.ndarray] = []
    dbaps: list[np.ndarray] = []
    v_ltas_diffs: list[np.ndarray] = []
    v_weights: list[float] = []
    u_ltas_diffs: list[np.ndarray] = []
    u_weights: list[float] = []

    tot_zh_v = 0
    tot_en_v = 0
    for u in utts:
        nz = int(u.get("n_zh_v", 0))
        ne = int(u.get("n_en_v", 0))
        tot_zh_v += nz
        tot_en_v += ne

        zf0 = u.get("zh_f0_med_hz")
        ef0 = u.get("en_f0_med_hz")
        if zf0 is not None and ef0 is not None and nz >= 12 and ne >= 10:
            df0_st.append(12.0 * math.log2(float(ef0) / float(zf0)))

        ziqr = u.get("zh_f0_iqr_st")
        eiqr = u.get("en_f0_iqr_st")
        if ziqr is not None and nz >= 20:
            zh_iqrs.append(float(ziqr))
        if eiqr is not None and ne >= 15:
            en_iqrs.append(float(eiqr))
        if ziqr is not None and eiqr is not None and float(ziqr) >= 0.8 and nz >= 25 and ne >= 20:
            iqr_ratios.append(float(eiqr) / float(ziqr))

        zpwr = u.get("zh_voiced_pwr_db")
        epwr = u.get("en_voiced_pwr_db")
        if zpwr is not None and epwr is not None and nz >= 12 and ne >= 10:
            drms_db.append(float(epwr) - float(zpwr))

        zb = u.get("zh_bap6")
        eb = u.get("en_bap6")
        if zb is not None and nz >= 12:
            zh_baps.append(np.asarray(zb, dtype=np.float64))
        if eb is not None and ne >= 10:
            en_baps.append(np.asarray(eb, dtype=np.float64))
        if zb is not None and eb is not None and nz >= 12 and ne >= 10:
            dbaps.append(np.asarray(eb, dtype=np.float64) - np.asarray(zb, dtype=np.float64))

        zv = u.get("zh_v_ltas")
        ev = u.get("en_v_ltas")
        if zv is not None and ev is not None and nz >= 15 and ne >= 10:
            diff_v = np.asarray(ev, dtype=np.float64) - np.asarray(zv, dtype=np.float64)
            # Zero DC over 80..11000 Hz (broadband loudness offset is tracked separately in drms_db)
            diff_v -= float(np.mean(diff_v[2:235]))
            v_ltas_diffs.append(diff_v)
            v_weights.append(float(min(nz, ne)))

        nzu = int(u.get("n_zh_u", 0))
        neu = int(u.get("n_en_u", 0))
        zu = u.get("zh_u_ltas")
        eu = u.get("en_u_ltas")
        if zu is not None and eu is not None and nzu >= 6 and neu >= 6:
            diff_u = np.asarray(eu, dtype=np.float64) - np.asarray(zu, dtype=np.float64)
            diff_u -= float(np.mean(diff_u[16:235]))
            u_ltas_diffs.append(diff_u)
            u_weights.append(float(min(nzu, neu)))

    # Smooth deltas via low-quefrency DCT (q < 16 preserves language acoustic-body shift without overfitting noise)
    v_raw = np.average(np.stack(v_ltas_diffs, axis=0), axis=0, weights=np.asarray(v_weights))
    v_cep = log_env_to_cep(v_raw[None, :], n_cep=16)[0]
    v_cep[0] = 0.0
    v_smooth = cep_to_log_env(v_cep[None, :], n_bins=N_BINS)[0]
    v_smooth -= float(np.mean(v_smooth[2:235]))
    v_smooth = np.clip(v_smooth, -0.55, 0.55)

    if u_ltas_diffs:
        u_raw = np.average(np.stack(u_ltas_diffs, axis=0), axis=0, weights=np.asarray(u_weights))
        u_cep = log_env_to_cep(u_raw[None, :], n_cep=12)[0]
        u_cep[0] = 0.0
        u_smooth = np.clip(cep_to_log_env(u_cep[None, :], n_bins=N_BINS)[0], -0.65, 0.65)
    else:
        u_smooth = np.zeros(N_BINS, dtype=np.float64)

    zh_bap_med = np.median(np.stack(zh_baps, axis=0), axis=0)
    en_bap_med = np.median(np.stack(en_baps, axis=0), axis=0)
    dbap_med = np.median(np.stack(dbaps, axis=0), axis=0)

    consistency = {
        "schema": "sf_bilingual_consistency_v1",
        "corpus": "SF_bilingual_zh_en",
        "speaker": "sf_female",
        "n_train_utts": len(utts),
        "zero_leakage_verified": True,
        "n_zh_voiced_frames": tot_zh_v,
        "n_en_voiced_frames": tot_en_v,
        "sf_zh_profile": zh_prof.summary_dict(),
        "sf_en_profile": en_prof.summary_dict(),
        "sf_zh_f0_med_hz": float(tr["zh_f0_med_hz"]),
        "sf_en_f0_med_hz": float(tr["en_f0_med_hz"]),
        "delta_f0_st_en_minus_zh": round(float(np.median(df0_st)), 3),
        "sf_zh_f0_iqr_st": round(float(np.median(zh_iqrs)), 3),
        "sf_en_f0_iqr_st": round(float(np.median(en_iqrs)), 3),
        "iqr_ratio_en_over_zh": round(float(np.median(iqr_ratios)), 3) if iqr_ratios else 1.0,
        "delta_rms_db_en_minus_zh": round(float(np.median(drms_db)), 3),
        "sf_zh_bap6_med": [round(float(x), 4) for x in zh_bap_med],
        "sf_en_bap6_med": [round(float(x), 4) for x in en_bap_med],
        "delta_bap6_en_minus_zh": [round(float(x), 4) for x in dbap_med],
        "delta_voiced_ltas_shape_rmse_db": round(float(np.sqrt(np.mean((v_smooth[2:235] * NEPER_TO_DB) ** 2))), 3),
        "delta_voiced_ltas_en_minus_zh_nep": [round(float(x), 5) for x in v_smooth],
        "delta_unvoiced_ltas_en_minus_zh_nep": [round(float(x), 5) for x in u_smooth],
        "switch_stats": {
            "zh_to_en": _summarize_switch_direction(switches, "zh_to_en"),
            "en_to_zh": _summarize_switch_direction(switches, "en_to_zh"),
        },
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    SF_CONSISTENCY_PATH.write_text(json.dumps(consistency, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {k: v for k, v in consistency.items() if "ltas_en_minus_zh_nep" not in k}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return consistency


if __name__ == "__main__":
    extract_sf_consistency()
