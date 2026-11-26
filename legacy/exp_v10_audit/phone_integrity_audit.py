"""Forensic Token, Alignment, Manner & Frontend G2P Integrity Audit.

Audits:
1. All EN tokens (v8 & v9 dual-bank alignment on 140/160 CMU-ARCTIC utterances) and
   Mandarin extended/ tokens (32 utterances, syllables + onsets + finals):
   - Duration defects (< 20 ms = <4 frames, == 5 ms = 1 frame, <= 10 ms)
   - Near-silence (RMS < -45 dBFS on utterance-normalized & raw audio)
   - Manner contradictions BEFORE forced voi_sl=False clamp AND on stored tokens:
     * vowel or sonorant with voiced_frac < 0.40
     * unvoiced fricative/affricate/stop with voiced_frac > 0.65 or uv_HF/LF < -5.0 dB
   Broken down by IPA phone/onset and speaker.
2. exp_v5_arch/crossling_frontend.py across all 160 CMU-ARCTIC EN texts +
   26 exp_v5_arch/benchmark_texts.json texts (10 ZH, 10 EN, 6 CS):
   - Exact OOV / en_rule_fallback / en_acronym_spelled word lists
   - Polyphone & heteronym / morphological mispronunciation risk scan.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.fft import rfftfreq

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v5_eval.metrics import read_wav  # noqa: E402
from exp_v6_crossling_dsp import arbiter as v6_arb  # noqa: E402
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS  # noqa: E402
from exp_v8_clarity.en_utau_bank import _trim_active_span, align_ipa_sequence  # noqa: E402
from exp_v9_clarity_pro.boundary_refiner import (  # noqa: E402
    align_en_phones_with_landmarks,
    compute_landmark_tracks,
    refine_zh_syllable_bounds,
)
from exp_v9_clarity_pro.hnm_v9 import analyze_v9  # noqa: E402
from exp_v9_clarity_pro.viterbi_utau import N_FFT, HOP, SR  # noqa: E402
from pybot_tts_v2_recorded import engine  # noqa: E402
from pybot_tts_v4_params import vocal  # noqa: E402

AUDIT_DIR = pathlib.Path(__file__).resolve().parent
REPORT_PATH = AUDIT_DIR / "phone_integrity_report.json"

ZH_UV_OBSTRUENTS = {"p", "t", "k", "f", "s", "sh", "x", "h", "c", "ch", "q"}

EN_HETERONYMS = {
    "read", "lead", "live", "record", "present", "produce", "use", "close",
    "wind", "tear", "bow", "row", "content", "project", "object", "subject",
    "contract", "permit", "conflict", "desert", "minute", "resume", "refuse",
    "separate", "graduate", "wound", "bass", "dove", "polish", "appropriate",
    "house", "does", "tear", "sow", "invalid", "moderate", "alternate",
}

ZH_COMMON_POLYPHONES = {
    "重", "长", "行", "还", "乐", "调", "发", "为", "只", "着", "得", "地",
    "的", "了", "和", "数", "转", "传", "参", "差", "看", "没", "中", "间",
    "分", "好", "要", "少", "觉", "应", "都", "给", "把", "相", "当", "干",
    "教", "空", "难", "便", "更", "强", "兴", "处", "量", "省", "系", "好",
    "卡", "率", "模", "调", "载", "供", "角", "倒", "弹", "降", "盛", "累",
}


def classify_ipa_manner(ipa: str) -> str:
    sp = IPA_SPECS.get(ipa)
    if sp is None:
        return "unknown"
    if sp.manner == "vowel":
        return "vowel"
    if sp.manner in ("nasal", "approximant"):
        return "sonorant"
    if not sp.voiced and sp.manner in ("fricative", "affricate", "stop"):
        return "uv_obstruent"
    return "v_obstruent"


def check_token_integrity(
    *,
    label: str,
    manner: str,
    n_frames: int,
    raw_voiced_frac: float,
    uv_hf_lf_db: float,
    norm_rms_dbfs: float,
    raw_rms_dbfs: float,
) -> dict[str, bool]:
    """Classify a single aligned token span for duration, silence, and manner defects."""
    dur_ms = n_frames * (HOP / float(SR) * 1000.0)
    is_lt20ms = dur_ms < 20.0 - 1e-6  # < 4 frames at 5 ms hop
    is_5ms = n_frames <= 1
    is_le10ms = n_frames <= 2
    is_near_silence_norm = norm_rms_dbfs < -45.0
    is_near_silence_raw = raw_rms_dbfs < -45.0

    vowel_low_voi = (manner == "vowel") and (raw_voiced_frac < 0.40)
    sonorant_low_voi = (manner == "sonorant") and (raw_voiced_frac < 0.40)
    uv_high_voi = (manner == "uv_obstruent") and (raw_voiced_frac > 0.65)
    uv_low_hflf = (manner == "uv_obstruent") and (uv_hf_lf_db < -5.0)
    manner_contradiction = vowel_low_voi or sonorant_low_voi or uv_high_voi or uv_low_hflf
    return {
        "lt20ms": is_lt20ms,
        "eq5ms": is_5ms,
        "le10ms": is_le10ms,
        "near_silence_norm": is_near_silence_norm,
        "near_silence_raw": is_near_silence_raw,
        "vowel_low_voi": vowel_low_voi,
        "sonorant_low_voi": sonorant_low_voi,
        "uv_high_voi": uv_high_voi,
        "uv_low_hflf": uv_low_hflf,
        "manner_contradiction": manner_contradiction,
        "any_defect": is_lt20ms or is_near_silence_norm or manner_contradiction,
    }


def _utts_norm_scale(x_act: np.ndarray) -> float:
    pk = float(np.max(np.abs(x_act))) if x_act.size else 0.0
    if pk <= 1e-12:
        return 1.0
    act = x_act[np.abs(x_act) > 0.01 * pk]
    rms = float(np.sqrt(np.mean(act * act))) if act.size else float(np.sqrt(np.mean(x_act * x_act)))
    if rms <= 1e-12:
        return 1.0
    return (10.0 ** (-20.0 / 20.0)) / rms


def _summarize_records(records: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[str(r[group_key])].append(r)
    out = {}
    for k, rows in sorted(groups.items()):
        n = len(rows)
        out[k] = {
            "count": n,
            "lt20ms": sum(1 for r in rows if r["lt20ms"]),
            "eq5ms": sum(1 for r in rows if r["eq5ms"]),
            "le10ms": sum(1 for r in rows if r["le10ms"]),
            "near_silence_norm_lt_neg45db": sum(1 for r in rows if r["near_silence_norm"]),
            "near_silence_raw_lt_neg45db": sum(1 for r in rows if r["near_silence_raw"]),
            "manner_contradiction": sum(1 for r in rows if r["manner_contradiction"]),
            "vowel_low_voi": sum(1 for r in rows if r["vowel_low_voi"]),
            "sonorant_low_voi": sum(1 for r in rows if r["sonorant_low_voi"]),
            "uv_high_voi": sum(1 for r in rows if r["uv_high_voi"]),
            "uv_low_hflf": sum(1 for r in rows if r["uv_low_hflf"]),
            "defect_rate": round(sum(1 for r in rows if r["any_defect"]) / max(1, n), 4),
            "median_dur_ms": round(float(np.median([r["dur_ms"] for r in rows])), 2),
        }
    return out


def audit_en_tokens(en_utterances: list[dict], *, include_heldout: bool = False) -> dict[str, Any]:
    """Audit EN aligned tokens for both v9 (landmark DP) and v8 (proportional DP) aligners."""
    if not include_heldout:
        utts = [u for u in en_utterances if not (u["speaker"] in ("slt", "bdl") and int(u["id"].split("_")[1]) >= 30)]
    else:
        utts = list(en_utterances)

    recs_v9: list[dict[str, Any]] = []
    recs_v8: list[dict[str, Any]] = []

    for u in utts:
        uid = u["id"]
        spk = u["speaker"]
        x, _ = read_wav(ROOT / u["path"])
        s0, s1 = _trim_active_span(x, sr=SR)
        x_act = x[s0:s1]
        norm_sc = _utts_norm_scale(x_act)
        x_norm = x_act * norm_sc
        feats, _ = analyze_v9(x_act, sr=SR)
        tracks = compute_landmark_tracks(x_act, feats, sr=SR)

        pl = cf.plan(u["text"], allow_oov_fallback=True)
        pairs: list[tuple[str, int | None]] = []
        for unit in pl["units"]:
            arpa = unit.get("arpabet") or []
            if arpa:
                pairs.extend(v6_arb._split_en_arpabet_to_ipa_with_stress(arpa))
            else:
                for p in (unit.get("ipa_phones") or []):
                    pairs.append((p, None))
        ipas = [p for p, _ in pairs]
        if not ipas:
            continue

        b_v9 = align_en_phones_with_landmarks(ipas, feats, x_act, sr=SR, tracks=tracks)
        b_v8 = align_ipa_sequence(ipas, feats, sr=SR)

        for ver, bounds, target_list in [("v9", b_v9, recs_v9), ("v8", b_v8, recs_v8)]:
            for idx_p, (ipa, (f0_i, f1_i)) in enumerate(zip(ipas, bounds)):
                if f1_i <= f0_i:
                    continue
                manner = classify_ipa_manner(ipa)
                # Mirror v9 edge trim before scoring v9
                f0_u, f1_u = f0_i, f1_i
                if ver == "v9" and manner == "uv_obstruent":
                    hf_sl = tracks["hf_lf"][f0_u:f1_u]
                    if (f1_u - f0_u) >= 4 and hf_sl[0] < -4.0 and tracks["voi"][f0_u] > 0.5:
                        f0_u += 1
                    if (f1_u - f0_u) >= 4 and hf_sl[-1] < -4.0 and tracks["voi"][f1_u - 1] > 0.5:
                        f1_u -= 1

                nf = f1_u - f0_u
                dur_ms = nf * (HOP / float(SR) * 1000.0)
                raw_vf = float(np.mean(feats.voiced[f0_u:f1_u])) if nf > 0 else 0.0
                hflf = float(np.mean(tracks["hf_lf"][f0_u:f1_u])) if nf > 0 else 0.0
                ws0 = max(0, f0_u * HOP)
                ws1 = min(len(x_act), f1_u * HOP)
                if ws1 > ws0:
                    norm_db = 20.0 * math.log10(max(1e-10, float(np.sqrt(np.mean(x_norm[ws0:ws1] ** 2)))))
                    raw_db = 20.0 * math.log10(max(1e-10, float(np.sqrt(np.mean(x_act[ws0:ws1] ** 2)))))
                else:
                    norm_db = -120.0
                    raw_db = -120.0
                flags = check_token_integrity(
                    label=ipa,
                    manner=manner,
                    n_frames=nf,
                    raw_voiced_frac=raw_vf,
                    uv_hf_lf_db=hflf,
                    norm_rms_dbfs=norm_db,
                    raw_rms_dbfs=raw_db,
                )
                target_list.append(
                    {
                        "ipa": ipa,
                        "manner": manner,
                        "speaker": spk,
                        "utterance_id": uid,
                        "n_frames": nf,
                        "dur_ms": round(dur_ms, 2),
                        "raw_voiced_frac": round(raw_vf, 3),
                        "uv_hf_lf_db": round(hflf, 2),
                        "norm_rms_dbfs": round(norm_db, 2),
                        "raw_rms_dbfs": round(raw_db, 2),
                        **flags,
                    }
                )

    def _pack(recs: list[dict[str, Any]]) -> dict[str, Any]:
        by_ipa = _summarize_records(recs, "ipa")
        by_spk = _summarize_records(recs, "speaker")
        by_manner = _summarize_records(recs, "manner")
        n = len(recs)
        worst_ipas = sorted(
            by_ipa.items(),
            key=lambda kv: (kv[1]["manner_contradiction"] + kv[1]["lt20ms"] + kv[1]["near_silence_norm_lt_neg45db"]),
            reverse=True,
        )[:15]
        return {
            "total_tokens": n,
            "lt20ms_count": sum(1 for r in recs if r["lt20ms"]),
            "eq5ms_count": sum(1 for r in recs if r["eq5ms"]),
            "le10ms_count": sum(1 for r in recs if r["le10ms"]),
            "near_silence_norm_lt_neg45db": sum(1 for r in recs if r["near_silence_norm"]),
            "near_silence_raw_lt_neg45db": sum(1 for r in recs if r["near_silence_raw"]),
            "manner_contradiction_total": sum(1 for r in recs if r["manner_contradiction"]),
            "vowel_low_voi_lt040": sum(1 for r in recs if r["vowel_low_voi"]),
            "sonorant_low_voi_lt040": sum(1 for r in recs if r["sonorant_low_voi"]),
            "uv_high_voi_gt065": sum(1 for r in recs if r["uv_high_voi"]),
            "uv_low_hflf_lt_neg5db": sum(1 for r in recs if r["uv_low_hflf"]),
            "any_defect_count": sum(1 for r in recs if r["any_defect"]),
            "by_manner": by_manner,
            "by_speaker": by_spk,
            "top_15_offending_ipas": {k: v for k, v in worst_ipas},
            "by_ipa": by_ipa,
        }

    return {"n_utterances": len(utts), "v9_landmark_dp": _pack(recs_v9), "v8_prop_dp": _pack(recs_v8)}


def audit_zh_tokens(zh_utts: list[dict]) -> dict[str, Any]:
    """Audit Mandarin extended/ syllables, onsets, and finals with v9 landmark refinement."""
    syl_recs: list[dict[str, Any]] = []
    onset_recs: list[dict[str, Any]] = []
    final_recs: list[dict[str, Any]] = []

    for u in zh_utts:
        uid, spk = u["id"], u["speaker"]
        x, _ = read_wav(ROOT / u["path"])
        act_start = int(u["active_start"])
        x_act = x[act_start : int(u["active_stop"])]
        norm_sc = _utts_norm_scale(x_act)
        x_norm = x_act * norm_sc
        feats, _ = analyze_v9(x_act, sr=SR)
        tracks = compute_landmark_tracks(x_act, feats, sr=SR)
        n_tot = len(feats.log_env)

        for syl in u["syllables"]:
            o, f, _ = engine.split_pinyin(syl["pinyin"])
            f_lo = max(0, min(n_tot - 2, int(round((int(syl["start_sample"]) - act_start) / HOP))))
            f_hi = max(f_lo + 2, min(n_tot, int(round((int(syl["stop_sample"]) - act_start) / HOP))))
            dur_ms = float(syl["ms"])
            _, c_dur = vocal._split_duration(o, max(0.07, dur_ms / 1000.0)) if o else (0.0, 0.0)
            prop_fr = max(2, min((f_hi - f_lo) - 2, int(round((c_dur / max(1e-4, dur_ms / 1000.0)) * (f_hi - f_lo))))) if o else 0
            f_lo_r, f_hi_r, f_sp = refine_zh_syllable_bounds(tracks, f_lo, f_hi, o, f, prop_fr)

            nf_s = f_hi_r - f_lo_r
            ws0, ws1 = max(0, f_lo_r * HOP), min(len(x_act), f_hi_r * HOP)
            norm_db = 20.0 * math.log10(max(1e-10, float(np.sqrt(np.mean(x_norm[ws0:ws1] ** 2))))) if ws1 > ws0 else -120.0
            raw_db = 20.0 * math.log10(max(1e-10, float(np.sqrt(np.mean(x_act[ws0:ws1] ** 2))))) if ws1 > ws0 else -120.0
            syl_vf = float(np.mean(feats.voiced[f_lo_r:f_hi_r]))
            syl_flags = check_token_integrity(
                label=f"{o}+{f}",
                manner="vowel",  # syllable nucleus should have strong voicing
                n_frames=nf_s,
                raw_voiced_frac=syl_vf,
                uv_hf_lf_db=0.0,
                norm_rms_dbfs=norm_db,
                raw_rms_dbfs=raw_db,
            )
            syl_recs.append(
                {
                    "pinyin_of": f"{o}+{f}",
                    "speaker": spk,
                    "utterance_id": uid,
                    "n_frames": nf_s,
                    "dur_ms": round(nf_s * 5.0, 2),
                    "raw_voiced_frac": round(syl_vf, 3),
                    "norm_rms_dbfs": round(norm_db, 2),
                    **syl_flags,
                }
            )

            if o and f_sp > f_lo_r:
                nf_o = f_sp - f_lo_r
                os0, os1 = max(0, f_lo_r * HOP), min(len(x_act), f_sp * HOP)
                on_norm_db = 20.0 * math.log10(max(1e-10, float(np.sqrt(np.mean(x_norm[os0:os1] ** 2))))) if os1 > os0 else -120.0
                on_raw_db = 20.0 * math.log10(max(1e-10, float(np.sqrt(np.mean(x_act[os0:os1] ** 2))))) if os1 > os0 else -120.0
                on_vf = float(np.mean(feats.voiced[f_lo_r:f_sp]))
                on_hf = float(np.mean(tracks["hf_lf"][f_lo_r:f_sp]))
                man = "uv_obstruent" if o in ZH_UV_OBSTRUENTS else ("v_obstruent" if o in ("b", "d", "g", "z", "zh", "j") else "sonorant")
                on_flags = check_token_integrity(
                    label=o, manner=man, n_frames=nf_o, raw_voiced_frac=on_vf,
                    uv_hf_lf_db=on_hf, norm_rms_dbfs=on_norm_db, raw_rms_dbfs=on_raw_db,
                )
                onset_recs.append(
                    {
                        "onset": o,
                        "manner": man,
                        "speaker": spk,
                        "utterance_id": uid,
                        "n_frames": nf_o,
                        "dur_ms": round(nf_o * 5.0, 2),
                        "raw_voiced_frac": round(on_vf, 3),
                        "uv_hf_lf_db": round(on_hf, 2),
                        "norm_rms_dbfs": round(on_norm_db, 2),
                        **on_flags,
                    }
                )

            if f_hi_r > f_sp:
                nf_f = f_hi_r - f_sp
                fs0, fs1 = max(0, f_sp * HOP), min(len(x_act), f_hi_r * HOP)
                fn_norm_db = 20.0 * math.log10(max(1e-10, float(np.sqrt(np.mean(x_norm[fs0:fs1] ** 2))))) if fs1 > fs0 else -120.0
                fn_raw_db = 20.0 * math.log10(max(1e-10, float(np.sqrt(np.mean(x_act[fs0:fs1] ** 2))))) if fs1 > fs0 else -120.0
                fn_vf = float(np.mean(feats.voiced[f_sp:f_hi_r]))
                fn_flags = check_token_integrity(
                    label=f, manner="vowel", n_frames=nf_f, raw_voiced_frac=fn_vf,
                    uv_hf_lf_db=0.0, norm_rms_dbfs=fn_norm_db, raw_rms_dbfs=fn_raw_db,
                )
                final_recs.append(
                    {
                        "final": f,
                        "speaker": spk,
                        "utterance_id": uid,
                        "n_frames": nf_f,
                        "dur_ms": round(nf_f * 5.0, 2),
                        "raw_voiced_frac": round(fn_vf, 3),
                        "norm_rms_dbfs": round(fn_norm_db, 2),
                        **fn_flags,
                    }
                )

    return {
        "n_utterances": len(zh_utts),
        "syllables": {
            "total": len(syl_recs),
            "lt20ms": sum(1 for r in syl_recs if r["lt20ms"]),
            "eq5ms": sum(1 for r in syl_recs if r["eq5ms"]),
            "near_silence_norm_lt_neg45db": sum(1 for r in syl_recs if r["near_silence_norm"]),
            "low_voicing_lt040": sum(1 for r in syl_recs if r["vowel_low_voi"]),
            "by_speaker": _summarize_records(syl_recs, "speaker"),
        },
        "onsets": {
            "total": len(onset_recs),
            "lt20ms": sum(1 for r in onset_recs if r["lt20ms"]),
            "eq5ms": sum(1 for r in onset_recs if r["eq5ms"]),
            "near_silence_norm_lt_neg45db": sum(1 for r in onset_recs if r["near_silence_norm"]),
            "manner_contradiction": sum(1 for r in onset_recs if r["manner_contradiction"]),
            "uv_high_voi_gt065": sum(1 for r in onset_recs if r["uv_high_voi"]),
            "uv_low_hflf_lt_neg5db": sum(1 for r in onset_recs if r["uv_low_hflf"]),
            "sonorant_low_voi_lt040": sum(1 for r in onset_recs if r["sonorant_low_voi"]),
            "by_onset": _summarize_records(onset_recs, "onset"),
            "by_speaker": _summarize_records(onset_recs, "speaker"),
        },
        "finals": {
            "total": len(final_recs),
            "lt20ms": sum(1 for r in final_recs if r["lt20ms"]),
            "eq5ms": sum(1 for r in final_recs if r["eq5ms"]),
            "near_silence_norm_lt_neg45db": sum(1 for r in final_recs if r["near_silence_norm"]),
            "vowel_low_voi_lt040": sum(1 for r in final_recs if r["vowel_low_voi"]),
            "by_speaker": _summarize_records(final_recs, "speaker"),
        },
    }


def audit_frontend_g2p(
    en_utterances: list[dict],
    benchmark_texts: dict[str, Any],
) -> dict[str, Any]:
    """Audit crossling_frontend.py on all 160 CMU-ARCTIC sentences + 26 benchmark_texts."""
    all_items: list[tuple[str, str, str]] = []
    for u in en_utterances:
        all_items.append(("cmu_arctic_160", u["id"], u["text"]))
    bench_items: list[tuple[str, str, str]] = []
    if "sentences" in benchmark_texts:
        for idx, item in enumerate(benchmark_texts["sentences"]):
            cat = str(item.get("lang", "mix"))
            tid = str(item.get("id", f"{cat}_{idx}"))
            txt = str(item["text"])
            bench_items.append((f"bench_{cat}", tid, txt))
    else:
        for cat in ("zh", "en", "codeswitch"):
            for idx, item in enumerate(benchmark_texts.get(cat, [])):
                txt = item["text"] if isinstance(item, dict) else str(item)
                tid = item.get("id", f"{cat}_{idx}") if isinstance(item, dict) else f"{cat}_{idx}"
                bench_items.append((f"bench_{cat}", tid, txt))
    all_items.extend(bench_items)

    src_counts = Counter()
    oov_fallbacks: dict[str, dict[str, Any]] = {}
    spelled_acronyms: dict[str, dict[str, Any]] = {}
    en_heteronym_hits: dict[str, list[dict[str, str]]] = defaultdict(list)
    zh_polyphone_hits: list[dict[str, str]] = []
    missing_ipa_from_bank = []

    for split_name, tid, text in all_items:
        pl = cf.plan(text, allow_oov_fallback=True)
        for unit in pl["units"]:
            src = str(unit.get("syllable_source", "unknown"))
            src_counts[src] += 1
            lang = unit.get("lang")
            tok = str(unit.get("token") or unit.get("hanzi") or "")
            if lang == "en":
                if src == "en_rule_oov":
                    ent = oov_fallbacks.setdefault(
                        tok,
                        {
                            "token": tok,
                            "arpabet": unit.get("arpabet"),
                            "ipa_phones": unit.get("ipa_phones"),
                            "count": 0,
                            "sources": [],
                        },
                    )
                    ent["count"] += 1
                    if f"{split_name}:{tid}" not in ent["sources"]:
                        ent["sources"].append(f"{split_name}:{tid}")
                elif src == "en_acronym_spelled":
                    ent = spelled_acronyms.setdefault(
                        tok,
                        {"token": tok, "arpabet": unit.get("arpabet"), "count": 0, "sources": []},
                    )
                    ent["count"] += 1
                    ent["sources"].append(f"{split_name}:{tid}")

                if tok.lower() in EN_HETERONYMS:
                    en_heteronym_hits[tok.lower()].append(
                        {"id": f"{split_name}:{tid}", "token": tok, "arpabet": " ".join(unit.get("arpabet") or []), "text": text}
                    )
            else:
                hz = str(unit.get("hanzi") or "")
                if len(hz) == 1 and hz in ZH_COMMON_POLYPHONES:
                    zh_polyphone_hits.append(
                        {"id": f"{split_name}:{tid}", "hanzi": hz, "pinyin": str(unit.get("pinyin")), "source": src, "text": text}
                    )

    # Also inspect en_lexicon size and apostrophe/hyphen handling
    return {
        "n_texts_audited": len(all_items),
        "n_cmu_arctic": len(en_utterances),
        "n_benchmark_texts": len(bench_items),
        "unit_source_counts": dict(src_counts.most_common()),
        "oov_rule_fallback_unique_count": len(oov_fallbacks),
        "oov_rule_fallback_token_count": sum(v["count"] for v in oov_fallbacks.values()),
        "oov_rule_fallback_words": sorted(oov_fallbacks.values(), key=lambda x: (-x["count"], x["token"].lower())),
        "spelled_acronyms": list(spelled_acronyms.values()),
        "en_heteronym_occurrences": dict(en_heteronym_hits),
        "zh_polyphone_occurrences_in_benchmark": zh_polyphone_hits,
    }


def main() -> int:
    en_man = json.loads((ROOT / "exp_v8_data/corpus/manifest.json").read_text(encoding="utf-8"))["english"]["utterances"]
    zh_utts = json.loads((ROOT / "pybot_tts_v2_recorded/extended/index.json").read_text(encoding="utf-8"))["utterances"]
    bench = json.loads((ROOT / "exp_v5_arch/benchmark_texts.json").read_text(encoding="utf-8"))

    print("[1/3] Auditing English tokens (v9 landmark DP & v8 prop DP across 140 train utterances)...", flush=True)
    en_rep = audit_en_tokens(en_man, include_heldout=False)

    print("[2/3] Auditing Mandarin extended/ tokens (32 utterances)...", flush=True)
    zh_rep = audit_zh_tokens(zh_utts)

    print("[3/3] Auditing crossling_frontend G2P on 160 CMU-ARCTIC + 26 benchmark texts...", flush=True)
    g2p_rep = audit_frontend_g2p(en_man, bench)

    out = {
        "en_tokens": en_rep,
        "zh_tokens": zh_rep,
        "frontend_g2p": g2p_rep,
    }
    REPORT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
