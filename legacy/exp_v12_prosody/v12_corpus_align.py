"""Zero-leakage Phase-8 GMM-HMM corpus aligner over expanded `manifest_v12.json` (`exp_v12_prosody`).

Reuses `exp_v11_align_gmm.gmm_hmm_aligner`'s 3-state 2-comp diagonal GMM-HMM without
modifying any `exp_v11_*` files, scaling cleanly to the 800-EN + expanded-ZH corpus
while holding out `slt[30:40]` & `bdl[30:40]` and any LOO Mandarin target IDs.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT_IMPORT = Path(__file__).resolve().parent.parent
if str(ROOT_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_IMPORT))

from exp_v5_arch import crossling_frontend as cf
from exp_v6_vocoder import vocoder as v6_voc
from exp_v8_clarity import en_utau_bank as v8_en
from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10
from exp_v11_align_gmm.gmm_hmm_aligner import (
    EN_HELDOUT_DEFAULT,
    AlignedUtterance,
    SyllableBoundary,
    _align_with_trained_gmm,
    _build_phone_boundaries,
    _train_and_align_gmm,
    _trim_utterance_active,
)
from exp_v12_prosody import HOP, ROOT_DIR, SR

MANIFEST_V12 = ROOT_DIR / "exp_v12_data" / "corpus" / "manifest_v12.json"


def align_en_v12_gmm(
    manifest_path: Path | str = MANIFEST_V12,
    exclude_ids: set[str] | frozenset[str] = EN_HELDOUT_DEFAULT,
    *,
    align_heldout: bool = True,
    n_em: int = 4,
    max_utts: int | None = None,
    sr: int = SR,
) -> dict:
    """Train 3-state GMM-HMM on expanded EN training utterances (`manifest_v12.json`)."""
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    en_utts = m["english"]["utterances"]
    if max_utts is not None:
        en_utts = en_utts[:max_utts]
    excl = set(exclude_ids)

    train_recs: list[dict] = []
    held_recs: list[dict] = []
    for u in en_utts:
        uid = u["id"]
        x, fsr = v6_voc.read_wav_mono(ROOT_DIR / u["path"])
        if fsr != sr:
            raise ValueError(f"Expected {sr} Hz, got {fsr} for {uid}")
        s0, s1 = v8_en._trim_active_span(x, sr=sr)
        xa = x[s0:s1]
        ff = v6_voc.analyze(xa, sr=sr)
        cs, ce, cf0, cf1 = _trim_utterance_active(xa, ff, sr=sr)
        xa_t = xa[cs:ce]
        feats = v6_voc.VocoderFeatures(
            f0_hz=ff.f0_hz[cf0:cf1],
            voiced=ff.voiced[cf0:cf1],
            log_env=ff.log_env[cf0:cf1],
            bap=ff.bap[cf0:cf1],
            sr=sr,
            hop_samples=HOP,
            n_samples=len(xa_t),
            n_fft=ff.n_fft,
        )
        ir = plan_v10(u["text"])
        meta_phones, ipas = [], []
        for s_idx, unit in enumerate(ir["units"]):
            arps = unit.get("arpabet") or []
            for k, p in enumerate(unit.get("ipa_phones", [])):
                ipas.append(p)
                meta_phones.append({
                    "ipa": p,
                    "arpabet": arps[k] if k < len(arps) else None,
                    "word": unit.get("token", ""),
                    "syllable_idx": s_idx,
                    "stress_or_tone": unit.get("stress_or_tone", "S0"),
                })
        trim_s = int(s0 + cs)
        trim_e = int(s0 + ce)
        rec = {"id": uid, "speaker": u["speaker"], "text": u["text"], "path": u["path"], "trim_start_sample": trim_s, "trim_stop_sample": trim_e, "wav": xa_t, "feats": feats, "ipas": ipas, "meta_phones": meta_phones}
        if uid in excl:
            held_recs.append(rec)
        else:
            train_recs.append(rec)

    gmm, train_bounds = _train_and_align_gmm(train_recs, n_em=n_em, sr=sr)
    aligned = {}
    for rec in train_recs:
        uid = rec["id"]
        pbs = _build_phone_boundaries(rec["meta_phones"], train_bounds[uid], rec["wav"], rec["feats"], sr=sr)
        aligned[uid] = {
            "aligned": AlignedUtterance(uid, rec["speaker"], "en", rec["text"], sr, HOP, len(rec["feats"].log_env), pbs),
            "wav": rec["wav"],
            "feats": rec["feats"],
            "path": rec["path"],
            "trim_start_sample": rec["trim_start_sample"],
            "trim_stop_sample": rec["trim_stop_sample"],
        }
    held_aligned = {}
    if align_heldout and held_recs:
        for rec in held_recs:
            uid = rec["id"]
            b = _align_with_trained_gmm(gmm, rec["wav"], rec["feats"], rec["ipas"], sr=sr)
            pbs = _build_phone_boundaries(rec["meta_phones"], b, rec["wav"], rec["feats"], sr=sr)
            held_aligned[uid] = {
                "aligned": AlignedUtterance(uid, rec["speaker"], "en", rec["text"], sr, HOP, len(rec["feats"].log_env), pbs),
                "wav": rec["wav"],
                "feats": rec["feats"],
                "path": rec["path"],
                "trim_start_sample": rec["trim_start_sample"],
                "trim_stop_sample": rec["trim_stop_sample"],
            }
    return {"utterances": aligned, "heldout_utterances": held_aligned, "included_ids": sorted(aligned.keys()), "excluded_ids": sorted(excl)}


def align_zh_v12_gmm(
    manifest_path: Path | str = MANIFEST_V12,
    exclude_ids: set[str] | frozenset[str] | None = None,
    *,
    n_em: int = 4,
    sr: int = SR,
) -> dict:
    """Train & align all Mandarin utterances in `manifest_v12.json` (`extended_ref` + `mandarin_expanded`)."""
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    excl = set(exclude_ids) if exclude_ids else set()
    rows = list(m.get("mandarin_extended_ref", {}).get("utterances", [])) + list(m.get("mandarin_expanded", {}).get("utterances", []))

    seen = set()
    recs: list[dict] = []
    for u in rows:
        uid = u["id"]
        if uid in excl or uid in seen:
            continue
        seen.add(uid)
        wav_p = Path(u["path"])
        if not wav_p.is_absolute():
            wav_p = ROOT_DIR / wav_p
        if not wav_p.exists():
            continue
        x, fsr = v6_voc.read_wav_mono(wav_p)
        if fsr != sr:
            continue
        s0 = int(u.get("active_start", 0))
        s1 = int(u.get("active_stop", len(x)))
        xa = x[s0:s1]
        feats = v6_voc.analyze(xa, sr=sr)
        n_fr = len(feats.log_env)

        ipas, meta_phones, init_bounds, syl_meta = [], [], [], []
        for sy in u.get("syllables", []):
            py = sy["pinyin"]
            try:
                onset, final, tone, syl_ipas = cf.pinyin_to_ipa(py)
            except Exception:
                continue
            f_lo = max(0, min(n_fr - 1, (int(sy["start_sample"]) - s0) // HOP))
            f_hi = max(f_lo + 1, min(n_fr, (int(sy["stop_sample"]) - s0) // HOP))
            edges = np.linspace(f_lo, f_hi, len(syl_ipas) + 1).astype(int)
            p_start_idx = len(ipas)
            for k, p in enumerate(syl_ipas):
                ipas.append(p)
                ps_k = int(edges[k])
                pe_k = int(max(ps_k + 1, edges[k + 1]))
                init_bounds.append((ps_k, (ps_k + pe_k) // 2, pe_k))
                meta_phones.append({
                    "ipa": p,
                    "arpabet": None,
                    "word": sy["hanzi"],
                    "syllable_idx": int(sy["index"]),
                    "stress_or_tone": f"T{tone}",
                    "is_onset": (k == 0 and bool(onset)),
                })
            syl_meta.append({
                "index": int(sy["index"]),
                "hanzi": sy["hanzi"],
                "pinyin": py,
                "onset": onset,
                "final": final,
                "tone": int(tone),
                "p_slice": (p_start_idx, len(ipas)),
            })
        if ipas:
            recs.append({"id": uid, "speaker": u["speaker"], "text": u["text"], "path": u["path"], "trim_start_sample": s0, "trim_stop_sample": s1, "wav": xa, "feats": feats, "ipas": ipas, "meta_phones": meta_phones, "init_bounds": init_bounds, "syl_meta": syl_meta})

    _, final_bounds = _train_and_align_gmm(recs, n_em=n_em, sr=sr)
    aligned = {}
    for rec in recs:
        uid = rec["id"]
        pbs = _build_phone_boundaries(rec["meta_phones"], final_bounds[uid], rec["wav"], rec["feats"], sr=sr)
        syls = []
        for sm in rec["syl_meta"]:
            a, b = sm["p_slice"]
            spbs = pbs[a:b]
            s_start, s_end = spbs[0].p_start, spbs[-1].p_end
            f_split = max(1, min(s_end - s_start - 1, spbs[0].p_end - s_start)) if (sm["onset"] and len(spbs) >= 2) else 0
            syls.append(SyllableBoundary(sm["index"], sm["hanzi"], sm["pinyin"], sm["onset"], sm["final"], sm["tone"], s_start, f_split, s_end, round((s_end - s_start) * HOP * 1000.0 / sr, 2), spbs))
        aligned[uid] = {
            "aligned": AlignedUtterance(uid, rec["speaker"], "zh", rec["text"], sr, HOP, len(rec["feats"].log_env), pbs, syls),
            "wav": rec["wav"],
            "feats": rec["feats"],
            "path": rec["path"],
            "trim_start_sample": rec["trim_start_sample"],
            "trim_stop_sample": rec["trim_stop_sample"],
        }
    return {"utterances": aligned, "included_ids": sorted(aligned.keys()), "excluded_ids": sorted(excl)}
