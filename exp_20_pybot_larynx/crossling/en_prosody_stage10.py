"""Stage 10 Zero-ML English Ridge Prosody, Duration, F0 Target-Approximation & Energy Engine.

Pure numpy/scipy deterministic prosody engine trained strictly on `en_stage10_train_tokens.pkl`:
1. Closed-form IRLS/Ridge Phone Duration Model (allophone, context bigrams/trigrams,
   stress, function-word status, pre-fortis clipping, cluster compression, boundary lengthening).
2. Smooth Phrase-Level English F0 & Target-Approximation (qTA) Contour Model
   (content-word pitch accents, unstressed troughs, nuclear accent, continuous declination,
   question rise vs statement fall, CF0 microprosody, AR(1) micro-F0).
3. Smooth Phone Relative Energy (`gain_db`) Model with temporal smoothing.
"""
from __future__ import annotations

import math
from pathlib import Path
from threading import Lock
from typing import Any, Sequence
import zlib

import numpy as np
from scipy.signal import lfilter

from exp_20_pybot_larynx import EXP_DIR
from exp_20_pybot_larynx.crossling.en_articulation import (
    ART_CLASSES,
    VOICELESS_OBSTRUENTS,
    VOWEL_IPAS,
    classify_articulatory_class,
)
from exp_20_pybot_larynx.crossling.en_frontend_v2 import plan_en_v2
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS

PROSODY_EN_STAGE10_PATH: Path = EXP_DIR / "crossling" / "prosody_en_stage10.npz"
PROSODY_EN_STAGE10_SUMMARY_PATH: Path = EXP_DIR / "crossling" / "prosody_en_stage10_summary.json"

MANNERS_VOCAB: tuple[str, ...] = (
    "vowel", "diphthong", "glide", "nasal", "liquid", "fricative", "affricate", "stop",
)
_MANNER_TO_ID: dict[str, int] = {m: i for i, m in enumerate(MANNERS_VOCAB)}

_MODEL_LOCK = Lock()
_MODEL_CACHE: dict[str, "ProsodyEnStage10Model"] = {}


def _stress_bucket(stress: Any) -> int:
    if stress == 0:
        return 0
    if stress == 1:
        return 1
    if stress == 2:
        return 2
    return 3


def compute_phone_prosody_context(
    phones: Sequence[dict[str, Any]],
    *,
    is_clause_final: bool = True,
) -> list[dict[str, Any]]:
    """Derive utterance, phrase, word, nuclear-accent, and phonetic context per phone."""
    n = len(phones)
    if n == 0:
        return []

    phrases: list[list[int]] = []
    cur: list[int] = []
    word_syls: dict[int, int] = {}
    word_stress_map: dict[int, int] = {}
    word_indices_ordered: list[int] = []
    seen_words: set[int] = set()

    for i, ph in enumerate(phones):
        wi = int(ph.get("word_idx", i))
        if wi not in seen_words:
            seen_words.add(wi)
            word_indices_ordered.append(wi)
        word_stress_map[wi] = int(ph.get("word_stress", ph.get("stress") or 0))
        if bool(ph.get("is_vowel", ph.get("ipa") in VOWEL_IPAS)):
            word_syls[wi] = word_syls.get(wi, 0) + 1
        cur.append(i)
        if bool(ph.get("is_phrase_final", False)) or float(ph.get("pause_after_ms", 0.0)) >= 45.0 or i == n - 1:
            phrases.append(cur)
            cur = []

    n_words = max(1, len(seen_words))
    word_pos_map = {wi: idx for idx, wi in enumerate(word_indices_ordered)}

    ph_pos = [0.5] * n
    ph_rem = [0] * n
    ph_len = [1] * n
    nuc_widx: set[int] = set()
    phrase_id_arr = [0] * n

    for pid, ph_idxs in enumerate(phrases):
        lp = len(ph_idxs)
        is_nonfinal_tail_phrase = (not is_clause_final) and (pid == len(phrases) - 1)
        cands = [
            int(phones[idx].get("word_idx", idx))
            for idx in ph_idxs
            if not bool(phones[idx].get("is_func", False))
            and int(phones[idx].get("word_stress", 0)) == 1
        ]
        if not cands:
            cands = [
                int(phones[idx].get("word_idx", idx))
                for idx in ph_idxs
                if int(phones[idx].get("word_stress", 0)) == 1
            ]
        if not cands:
            cands = [
                int(phones[idx].get("word_idx", idx))
                for idx in ph_idxs
                if not bool(phones[idx].get("is_func", False))
            ]
        if cands:
            nuc_widx.add(cands[-1])
        pos_cap = 0.72 if is_nonfinal_tail_phrase else 1.0
        for j, idx in enumerate(ph_idxs):
            phrase_id_arr[idx] = pid
            raw_p = float(j) / float(max(1, lp - 1)) if lp > 1 else 0.5
            ph_pos[idx] = raw_p * pos_cap
            ph_rem[idx] = (lp - 1 - j) + (2 if is_nonfinal_tail_phrase else 0)
            ph_len[idx] = lp

    pos_scale_utt = 1.0 if is_clause_final else 0.72
    out: list[dict[str, Any]] = []
    for i, ph in enumerate(phones):
        ipa = str(ph.get("ipa", "ə"))
        allo = str(ph.get("allophone", ipa))
        wi = int(ph.get("word_idx", i))
        pause_after = float(ph.get("pause_after_ms", 0.0))
        pause_prev = float(phones[i - 1].get("pause_after_ms", 0.0)) if i > 0 else 200.0

        prev_ph = phones[i - 1] if i > 0 else None
        next_ph = phones[i + 1] if i + 1 < n else None
        prev_ipa = str(prev_ph.get("ipa", "SIL")) if (prev_ph is not None and pause_prev < 20.0) else "SIL"
        next_ipa = str(next_ph.get("ipa", "SIL")) if (next_ph is not None and pause_after < 20.0) else "SIL"
        if not is_clause_final and i == n - 1 and pause_after < 20.0:
            next_ipa = "ə"
        prev_cls = int(ph.get("prev_cls", classify_articulatory_class(None if prev_ipa == "SIL" else prev_ipa)))
        next_cls = int(ph.get("next_cls", classify_articulatory_class(None if next_ipa == "SIL" else next_ipa)))

        is_v = bool(ph.get("is_vowel", ipa in VOWEL_IPAS))
        same_w_next = bool(next_ph is not None and pause_after < 20.0 and int(next_ph.get("word_idx", -1)) == wi)
        same_w_prev = bool(prev_ph is not None and pause_prev < 20.0 and int(prev_ph.get("word_idx", -1)) == wi)
        pre_fortis = bool(ph.get("pre_fortis", is_v and same_w_next and next_ipa in VOICELESS_OBSTRUENTS))

        sp = IPA_SPECS.get(ipa)
        manner = sp.manner if sp is not None else "vowel"
        man_id = _MANNER_TO_ID.get(manner, 0)

        w_n = max(1, int(ph.get("word_n_phones", 1)))
        p_in_w = int(ph.get("phone_in_word_idx", 0))
        w_syl = max(1, int(word_syls.get(wi, 1)))
        w_ord = word_pos_map.get(wi, 0)
        prev_w_stress = word_stress_map.get(wi - 1, 0)
        next_w_stress = word_stress_map.get(wi + 1, 0)

        prev_v = bool(prev_ph is not None and bool(prev_ph.get("is_vowel", prev_ipa in VOWEL_IPAS)))
        next_v = bool(next_ph is not None and bool(next_ph.get("is_vowel", next_ipa in VOWEL_IPAS)))
        next_stress = next_ph.get("stress") if next_ph is not None else None
        is_flap_ctx = bool(ipa in ("t", "d") and prev_v and next_v and next_stress == 0)

        last_phrase_idx = len(phrases) - 1
        in_tail_nonfinal = (not is_clause_final) and (phrase_id_arr[i] == last_phrase_idx) and (pause_after < 45.0)
        is_pfin_eff = False if in_tail_nonfinal else bool(ph.get("is_phrase_final", i == n - 1 or pause_after >= 45.0))
        is_pfin_w_eff = False if in_tail_nonfinal else bool(ph.get("is_in_phrase_final_word", False))
        is_ufin_eff = False if not is_clause_final else bool(ph.get("is_utterance_final", i == n - 1))

        out.append({
            "ipa": ipa,
            "allophone": allo,
            "stress": ph.get("stress"),
            "stress_b": _stress_bucket(ph.get("stress")),
            "word_stress": min(2, max(0, int(ph.get("word_stress", 0)))),
            "prev_w_stress": min(2, max(0, int(prev_w_stress))),
            "next_w_stress": min(2, max(0, int(next_w_stress))),
            "word_lower": str(ph.get("word", "")).lower(),
            "word_idx": wi,
            "word_ord": w_ord,
            "n_words": n_words,
            "word_n_phones": w_n,
            "word_n_syls": w_syl,
            "phone_in_word_idx": p_in_w,
            "is_word_initial": bool(ph.get("is_word_initial", p_in_w == 0)),
            "is_word_final": bool(ph.get("is_word_final", p_in_w >= w_n - 1)),
            "is_phrase_final": is_pfin_eff,
            "is_in_phrase_final_word": is_pfin_w_eff,
            "is_utterance_final": is_ufin_eff,
            "is_vowel": is_v,
            "is_func": bool(ph.get("is_func", False)),
            "is_focused": bool(ph.get("is_focused", False)),
            "focus_level": int(ph.get("focus_level", 1 if ph.get("is_focused") else 0)),
            "is_nuclear": wi in nuc_widx,
            "is_question": bool(ph.get("is_question", False)),
            "cluster_len": max(1, int(ph.get("cluster_len", 1))),
            "pause_after_ms": pause_after,
            "pause_prev_ms": pause_prev,
            "pre_fortis": pre_fortis,
            "prev_ipa": prev_ipa,
            "next_ipa": next_ipa,
            "prev_cls": max(0, min(len(ART_CLASSES) - 1, prev_cls)),
            "next_cls": max(0, min(len(ART_CLASSES) - 1, next_cls)),
            "manner": manner,
            "man_id": man_id,
            "same_w_next": same_w_next,
            "same_w_prev": same_w_prev,
            "is_flap_ctx": is_flap_ctx,
            "utt_pos": (float(i) / float(max(1, n - 1)) if n > 1 else 0.5) * pos_scale_utt,
            "n_phones": n,
            "phrase_id": phrase_id_arr[i],
            "ph_pos": ph_pos[i],
            "ph_rem": ph_rem[i],
            "ph_len": ph_len[i],
        })
    return out


def extract_duration_features(
    ctxs: Sequence[dict[str, Any]],
    *,
    allo_map: dict[str, int],
    ipa_map: dict[str, int],
    ctx_map: dict[str, int],
    as_map: dict[tuple[str, int], int],
    fw_map: dict[str, int],
    pbg_map: dict[tuple[str, str], int],
    nbg_map: dict[tuple[str, str], int],
    tbg_map: dict[tuple[str, str, str], int],
) -> np.ndarray:
    """Build deterministic Ridge feature matrix `(N_phones, D_dur)` matching TRAIN exactly."""
    n = len(ctxs)
    na = len(allo_map)
    ni = len(ipa_map)
    nc = len(ctx_map)
    nas = len(as_map)
    nfw = len(fw_map)
    npb = len(pbg_map)
    nnb = len(nbg_map)
    ntb = len(tbg_map)
    nm = len(MANNERS_VOCAB)
    ncls = len(ART_CLASSES)
    n_flags = 46

    d = 1 + na * 6 + ni + nc * 2 + nas + nfw + npb + nnb + ntb + 4 + 3 + nm * 4 + ncls * 2 + n_flags
    X = np.zeros((n, d), dtype=np.float64)

    for i, c in enumerate(ctxs):
        col = 0
        X[i, col] = 1.0
        col += 1

        pfin = 1.0 if c["is_phrase_final"] else 0.0
        pfin_w = 1.0 if c["is_in_phrase_final_word"] else 0.0
        ufin = 1.0 if c["is_utterance_final"] else 0.0
        wi = 1.0 if c["is_word_initial"] else 0.0
        wf = 1.0 if c["is_word_final"] else 0.0

        ai = allo_map.get(c["allophone"])
        if ai is not None:
            X[i, col + ai] = 1.0
            X[i, col + na + ai] = pfin
            X[i, col + 2 * na + ai] = ufin
            X[i, col + 3 * na + ai] = wf
            X[i, col + 4 * na + ai] = wi
            X[i, col + 5 * na + ai] = pfin_w
        col += 6 * na

        ii = ipa_map.get(c["ipa"])
        if ii is not None:
            X[i, col + ii] = 1.0
        col += ni

        pi = ctx_map.get(c["prev_ipa"])
        if pi is not None:
            X[i, col + pi] = 1.0
        col += nc

        nxi = ctx_map.get(c["next_ipa"])
        if nxi is not None:
            X[i, col + nxi] = 1.0
        col += nc

        s_oh = c["stress_b"]
        asi = as_map.get((c["allophone"], s_oh))
        if asi is not None:
            X[i, col + asi] = 1.0
        col += nas

        if c["is_func"]:
            fwi = fw_map.get(c["word_lower"])
            if fwi is not None:
                X[i, col + fwi] = 1.0
        col += nfw

        pbi = pbg_map.get((c["prev_ipa"], c["allophone"]))
        if pbi is not None:
            X[i, col + pbi] = 1.0
        col += npb

        nbi = nbg_map.get((c["allophone"], c["next_ipa"]))
        if nbi is not None:
            X[i, col + nbi] = 1.0
        col += nnb

        tbi = tbg_map.get((c["prev_ipa"], c["allophone"], c["next_ipa"]))
        if tbi is not None:
            X[i, col + tbi] = 1.0
        col += ntb

        X[i, col + s_oh] = 1.0
        col += 4
        X[i, col + c["word_stress"]] = 1.0
        col += 3

        mi = c["man_id"]
        X[i, col + mi] = 1.0
        X[i, col + nm + mi] = pfin
        X[i, col + 2 * nm + mi] = ufin
        X[i, col + 3 * nm + mi] = pfin_w
        col += 4 * nm

        X[i, col + c["prev_cls"]] = 1.0
        col += ncls
        X[i, col + c["next_cls"]] = 1.0
        col += ncls

        is_v = 1.0 if c["is_vowel"] else 0.0
        is_f = 1.0 if c["is_func"] else 0.0
        pf = 1.0 if c["pre_fortis"] else 0.0
        cl = float(c["cluster_len"])
        upos = c["utt_pos"]
        n_ph = max(1, int(c["n_phones"]))
        n_w = max(1, int(c["n_words"]))
        w_n = float(c["word_n_phones"])
        w_syl = float(c["word_n_syls"])
        w_pos = float(c["phone_in_word_idx"]) / max(1.0, w_n - 1.0) if w_n > 1.0 else 0.5
        prev_vl = 1.0 if c["prev_ipa"] in VOICELESS_OBSTRUENTS else 0.0
        next_vl = 1.0 if c["next_ipa"] in VOICELESS_OBSTRUENTS else 0.0
        prev_sil = 1.0 if c["prev_ipa"] == "SIL" else 0.0
        next_sil = 1.0 if c["next_ipa"] == "SIL" else 0.0
        p_after = c["pause_after_ms"]
        p_prev = c["pause_prev_ms"]

        flags = (
            is_v,
            is_f,
            pf,
            pf * (1.0 if s_oh == 1 else 0.0),
            pfin,
            pfin * is_v,
            pfin_w,
            pfin_w * is_v,
            ufin,
            ufin * is_v,
            wi,
            wf,
            1.0 if w_n == 1.0 else 0.0,
            1.0 / w_n,
            1.0 / w_syl,
            math.log1p(w_syl),
            w_pos,
            1.0 if (not c["is_vowel"] and cl == 2.0) else 0.0,
            1.0 if (not c["is_vowel"] and cl >= 3.0) else 0.0,
            1.0 / max(1.0, cl),
            upos,
            upos * upos,
            math.log1p(n_ph),
            1.0 / math.sqrt(float(n_ph)),
            1.0 / float(n_w),
            c["ph_pos"],
            c["ph_pos"] ** 2,
            1.0 / (1.0 + c["ph_rem"]),
            1.0 if c["ph_rem"] == 1 else 0.0,
            1.0 if c["ph_rem"] == 2 else 0.0,
            math.log1p(c["ph_len"]),
            math.log1p(p_after),
            1.0 if p_after >= 45.0 else 0.0,
            1.0 if p_after >= 180.0 else 0.0,
            math.log1p(p_prev),
            1.0 if p_prev >= 45.0 else 0.0,
            1.0 if c["is_nuclear"] else 0.0,
            1.0 if (c["is_nuclear"] and c["is_vowel"]) else 0.0,
            1.0 if c["is_question"] else 0.0,
            prev_vl,
            next_vl,
            prev_sil,
            next_sil,
            1.0 if c["same_w_next"] else 0.0,
            1.0 if c["same_w_prev"] else 0.0,
            1.0 if c["is_flap_ctx"] else 0.0,
        )
        X[i, col : col + n_flags] = flags
    return X


def extract_prosody_features(
    ctxs: Sequence[dict[str, Any]],
    *,
    allo_map: dict[str, int],
    ipa_map: dict[str, int],
    ctx_map: dict[str, int],
    fw_map: dict[str, int],
) -> np.ndarray:
    """Build deterministic Ridge feature matrix `(N_phones, D_pros)` for F0 & relative energy."""
    n = len(ctxs)
    na = len(allo_map)
    ni = len(ipa_map)
    nc = len(ctx_map)
    nfw = len(fw_map)
    nm = len(MANNERS_VOCAB)
    ncls = len(ART_CLASSES)
    n_flags = 36

    d = 1 + na + ni + nc * 2 + nfw + 4 + 3 + 3 + 3 + nm * 2 + ncls * 2 + n_flags
    X = np.zeros((n, d), dtype=np.float64)

    for i, c in enumerate(ctxs):
        col = 0
        X[i, col] = 1.0
        col += 1

        ai = allo_map.get(c["allophone"])
        if ai is not None:
            X[i, col + ai] = 1.0
        col += na

        ii = ipa_map.get(c["ipa"])
        if ii is not None:
            X[i, col + ii] = 1.0
        col += ni

        pi = ctx_map.get(c["prev_ipa"])
        if pi is not None:
            X[i, col + pi] = 1.0
        col += nc

        nxi = ctx_map.get(c["next_ipa"])
        if nxi is not None:
            X[i, col + nxi] = 1.0
        col += nc

        if c["is_func"]:
            fwi = fw_map.get(c["word_lower"])
            if fwi is not None:
                X[i, col + fwi] = 1.0
        col += nfw

        s_oh = c["stress_b"]
        ws = c["word_stress"]
        X[i, col + s_oh] = 1.0
        col += 4
        X[i, col + ws] = 1.0
        col += 3
        X[i, col + c["prev_w_stress"]] = 1.0
        col += 3
        X[i, col + c["next_w_stress"]] = 1.0
        col += 3

        pfin = 1.0 if c["is_phrase_final"] else 0.0
        pfin_w = 1.0 if c["is_in_phrase_final_word"] else 0.0
        ufin = 1.0 if c["is_utterance_final"] else 0.0
        mi = c["man_id"]
        X[i, col + mi] = 1.0
        X[i, col + nm + mi] = pfin
        col += 2 * nm

        X[i, col + c["prev_cls"]] = 1.0
        col += ncls
        X[i, col + c["next_cls"]] = 1.0
        col += ncls

        is_v = 1.0 if c["is_vowel"] else 0.0
        is_f = 1.0 if c["is_func"] else 0.0
        is_nuc = 1.0 if c["is_nuclear"] else 0.0
        is_q = 1.0 if c["is_question"] else 0.0
        upos = c["utt_pos"]
        ph_p = c["ph_pos"]
        prev_vl = 1.0 if c["prev_ipa"] in VOICELESS_OBSTRUENTS else 0.0
        next_vl = 1.0 if c["next_ipa"] in VOICELESS_OBSTRUENTS else 0.0
        p_after = c["pause_after_ms"]

        flags = (
            is_v,
            is_f,
            pfin,
            pfin_w,
            ufin,
            1.0 if c["is_word_initial"] else 0.0,
            1.0 if c["is_word_final"] else 0.0,
            is_nuc,
            is_nuc * is_v,
            1.0 if (c["is_nuclear"] and s_oh == 1) else 0.0,
            is_q,
            is_q * pfin_w,
            is_q * ph_p,
            (1.0 - is_q) * pfin_w,
            upos,
            upos * upos,
            1.0 / max(1.0, float(c["n_phones"])),
            ph_p,
            ph_p * ph_p,
            1.0 / (1.0 + c["ph_rem"]),
            1.0 if ph_p < 0.25 else 0.0,
            math.log1p(c["ph_len"]),
            math.log1p(p_after),
            1.0 if p_after >= 45.0 else 0.0,
            prev_vl,
            next_vl,
            1.0 if c["word_ord"] == 0 else 0.0,
            1.0 if c["word_ord"] == c["n_words"] - 1 else 0.0,
            float(c["word_ord"]) / float(max(1, c["n_words"] - 1)),
            1.0 if (not c["is_func"] and ws == 1) else 0.0,
            1.0 if (c["is_func"] and ws == 0) else 0.0,
            is_v * (1.0 if s_oh == 1 else 0.0),
            is_v * (1.0 if s_oh == 0 else 0.0),
            (1.0 - is_f) * pfin_w,
            (1.0 - is_q) * ufin,
            is_q * ufin,
        )
        X[i, col : col + n_flags] = flags
    return X


class ProsodyEnStage10Model:
    """Zero-ML closed-form Ridge Stage 10 English Prosody Model loaded from `.npz` (`allow_pickle=False`)."""

    def __init__(self, path: Path | str = PROSODY_EN_STAGE10_PATH) -> None:
        self.loaded = False
        self.path = Path(path)

        self.allos_vocab: list[str] = []
        self.allo_map: dict[str, int] = {}
        self.ipas_vocab: list[str] = []
        self.ipa_map: dict[str, int] = {}
        self.ctx_vocab: list[str] = []
        self.ctx_map: dict[str, int] = {}
        self.as_pairs: list[tuple[str, int]] = []
        self.as_map: dict[tuple[str, int], int] = {}
        self.fw_vocab: list[str] = []
        self.fw_map: dict[str, int] = {}
        self.prev_bg: list[tuple[str, str]] = []
        self.pbg_map: dict[tuple[str, str], int] = {}
        self.next_bg: list[tuple[str, str]] = []
        self.nbg_map: dict[tuple[str, str], int] = {}
        self.tri_bg: list[tuple[str, str, str]] = []
        self.tbg_map: dict[tuple[str, str, str], int] = {}

        self.w_dur: np.ndarray = np.zeros(1, dtype=np.float64)
        self.w_f0_5: np.ndarray = np.zeros((1, 5), dtype=np.float64)
        self.w_gain: np.ndarray = np.zeros(1, dtype=np.float64)
        self.speaker_f0_med_hz: float = 174.41

        # Calibrated contour & dynamics parameters stored in .npz
        self.params: dict[str, float] = {
            "f0_ridge_w": 0.95,
            "f0_accent_w": 1.25,
            "f0_nuc_boost_st": 2.20,
            "f0_content_boost_st": 1.50,
            "f0_sec_boost_st": 0.45,
            "f0_func_trough_st": -1.40,
            "f0_decl_span_st": 1.55,
            "f0_q_rise_st": 2.90,
            "f0_s_fall_st": 1.60,
            "f0_qta_tau_ms": 5.5,
            "f0_cf0_vl_st": 0.65,
            "f0_ar1_sigma": 0.30,
            "f0_ar1_rho": 0.76,
            "gain_scale": 0.45,
            "gain_min_db": -5.5,
            "gain_max_db": 3.5,
        }

        if not self.path.exists():
            return

        with np.load(self.path, allow_pickle=False) as d:
            self.allos_vocab = [str(x) for x in d["allos_vocab"]]
            self.allo_map = {k: i for i, k in enumerate(self.allos_vocab)}
            self.ipas_vocab = [str(x) for x in d["ipas_vocab"]]
            self.ipa_map = {k: i for i, k in enumerate(self.ipas_vocab)}
            self.ctx_vocab = [str(x) for x in d["ctx_vocab"]]
            self.ctx_map = {k: i for i, k in enumerate(self.ctx_vocab)}
            self.as_pairs = [(str(a), int(s)) for a, s in zip(d["as_allos"], d["as_stress"])]
            self.as_map = {k: i for i, k in enumerate(self.as_pairs)}
            self.fw_vocab = [str(x) for x in d["fw_vocab"]]
            self.fw_map = {k: i for i, k in enumerate(self.fw_vocab)}
            self.prev_bg = [(str(a), str(b)) for a, b in d["prev_bg"]]
            self.pbg_map = {k: i for i, k in enumerate(self.prev_bg)}
            self.next_bg = [(str(a), str(b)) for a, b in d["next_bg"]]
            self.nbg_map = {k: i for i, k in enumerate(self.next_bg)}
            self.tri_bg = [(str(a), str(b), str(c)) for a, b, c in d["tri_bg"]]
            self.tbg_map = {k: i for i, k in enumerate(self.tri_bg)}

            self.w_dur = d["w_dur"].astype(np.float64)
            self.w_f0_5 = d["w_f0_5"].astype(np.float64)
            self.w_gain = d["w_gain"].astype(np.float64)
            if "speaker_f0_med_hz" in d:
                self.speaker_f0_med_hz = float(d["speaker_f0_med_hz"][0])
            if "param_keys" in d and "param_vals" in d:
                for k, v in zip(d["param_keys"], d["param_vals"]):
                    self.params[str(k)] = float(v)
        self.loaded = True


def get_en_prosody_stage10_model(path: Path | str = PROSODY_EN_STAGE10_PATH) -> ProsodyEnStage10Model:
    """Thread-safe cached loader for :class:`ProsodyEnStage10Model`."""
    p = Path(path)
    key = str(p.resolve()) if p.exists() else str(p)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None and cached.loaded:
            return cached
    mdl = ProsodyEnStage10Model(p)
    with _MODEL_LOCK:
        _MODEL_CACHE[key] = mdl
    return mdl


def clear_en_prosody_stage10_cache() -> None:
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()


def _qta_smooth_2nd_order(target: np.ndarray, tau_ms: float = 22.0, hop_ms: float = 5.0) -> np.ndarray:
    """Critically damped 2nd-order Target-Approximation smoothing across a continuous span."""
    n = len(target)
    if n <= 2:
        return target.copy()
    alpha = 1.0 - math.exp(-hop_ms / max(5.0, float(tau_ms)))
    # Symmetric 2-pass causal+anti-causal EMA yields C^1 smooth zero-lag target approximation
    z = np.empty(n, dtype=np.float64)
    z[0] = target[0]
    for k in range(1, n):
        z[k] = (1.0 - alpha) * z[k - 1] + alpha * target[k]
    y = np.empty(n, dtype=np.float64)
    y[-1] = z[-1]
    for k in range(n - 2, -1, -1):
        y[k] = (1.0 - alpha) * y[k + 1] + alpha * z[k]
    return y


def _fallback_phone_dur_ms(c: dict[str, Any]) -> float:
    ipa = c["ipa"]
    sp = IPA_SPECS.get(ipa)
    d = float(sp.base_dur_ms) if sp is not None else 75.0
    if c["is_vowel"]:
        if c["stress_b"] == 1:
            d *= 1.15
        elif c["stress_b"] == 0:
            d = max(35.0, d * 0.72)
    if c["pre_fortis"]:
        d *= 0.80
    if not c["is_vowel"] and c["cluster_len"] >= 2:
        d *= 0.76 if c["cluster_len"] == 2 else 0.64
    if c["is_phrase_final"]:
        d *= 1.35
    return float(np.clip(d, 20.0, 280.0))


def predict_en_prosody_stage10(
    text_or_units: str | Sequence[dict[str, Any]],
    *,
    base_f0_hz: float | None = None,
    rate: float = 1.0,
    pause_scale: float = 1.0,
    range_scale: float = 1.0,
    clause_pos: tuple[float, float] = (0.0, 1.0),
    is_clause_final: bool = True,
    is_question: bool | None = None,
    focus_spans: Sequence[Any] | None = None,
    include_pauses: bool = True,
    seed: int = 2026,
    model: ProsodyEnStage10Model | None = None,
    model_path: Path | str = PROSODY_EN_STAGE10_PATH,
    embedded_in_zh: bool = False,
    expressive_drive: float | None = None,
) -> dict[str, Any]:
    """Predict Stage 10 English phone durations, smooth F0 target-approximation contour, and smooth gain_db."""
    plan = plan_en_v2(
        text_or_units,
        focus_spans=focus_spans,
        is_clause_final=is_clause_final,
        is_question=is_question,
    )
    mdl = model if model is not None else get_en_prosody_stage10_model(model_path)
    eff_base_f0 = float(base_f0_hz) if base_f0_hz is not None else float(mdl.speaker_f0_med_hz)
    eff_base_f0 = float(np.clip(eff_base_f0, 60.0, 460.0))
    eff_rate = max(0.60, min(1.60, float(rate)))
    dual_mode = expressive_drive is not None and not embedded_in_zh
    drive = float(np.clip(expressive_drive, 0.0, 1.0)) if dual_mode else 1.0
    calm = 1.0 - drive
    if dual_mode:
        eff_rate = float(np.clip(eff_rate * (1.0 + 0.055 * calm), 0.60, 1.65))
    eff_pause = max(0.40, min(2.00, float(pause_scale)))
    eff_range = max(0.40, min(2.00, float(range_scale)))

    phones = [dict(p) for p in plan.get("phones", [])]
    n_p = len(phones)
    if n_p == 0:
        return {
            "plan": plan,
            "phones": [],
            "phone_durs_ms": [],
            "phone_gains_db": [],
            "n_frames": 0,
            "f0_hz": np.zeros(0, dtype=np.float64),
            "f0_st": np.zeros(0, dtype=np.float64),
            "gain_db_frames": np.zeros(0, dtype=np.float64),
            "phone_spans": [],
            "base_f0_hz": eff_base_f0,
            "is_question": bool(plan.get("is_question", False)),
        }

    for ph in phones:
        ph["is_question"] = bool(plan.get("is_question", False))
    ctxs = compute_phone_prosody_context(phones, is_clause_final=is_clause_final)

    if mdl.loaded:
        X_dur = extract_duration_features(
            ctxs,
            allo_map=mdl.allo_map,
            ipa_map=mdl.ipa_map,
            ctx_map=mdl.ctx_map,
            as_map=mdl.as_map,
            fw_map=mdl.fw_map,
            pbg_map=mdl.pbg_map,
            nbg_map=mdl.nbg_map,
            tbg_map=mdl.tbg_map,
        )
        X_pros = extract_prosody_features(
            ctxs,
            allo_map=mdl.allo_map,
            ipa_map=mdl.ipa_map,
            ctx_map=mdl.ctx_map,
            fw_map=mdl.fw_map,
        )
        raw_durs = np.clip(X_dur @ mdl.w_dur, 20.0, 310.0)
        f0_5_pred = np.clip(X_pros @ mdl.w_f0_5, -10.0, 10.0)
        raw_gains = X_pros @ mdl.w_gain
    else:
        raw_durs = np.array([_fallback_phone_dur_ms(c) for c in ctxs], dtype=np.float64)
        f0_5_pred = np.zeros((n_p, 5), dtype=np.float64)
        raw_gains = np.array([1.0 if c["is_vowel"] else -3.0 for c in ctxs], dtype=np.float64)

    prm = mdl.params
    g_scale = float(prm.get("gain_scale", 0.45))
    g_min = float(prm.get("gain_min_db", -5.5))
    g_max = float(prm.get("gain_max_db", 3.5))

    # Calibrate phone durations & build frame layout
    durs_ms: list[float] = []
    gains_db: list[float] = []
    spans: list[dict[str, Any]] = []
    cur_fr = 0

    for i, (ph, c) in enumerate(zip(phones, ctxs)):
        d_eff = float(np.clip(raw_durs[i] / eff_rate, 20.0, 320.0))
        if dual_mode and str(c["ipa"]) in ("s", "ʃ", "f", "θ", "h"):
            d_eff = min(d_eff, 125.0 + 45.0 * drive)
        if c["is_focused"] and c["is_vowel"]:
            d_eff = min(330.0, d_eff * (1.0 + 0.14 * min(2, c["focus_level"])))
        nf = max(1, int(round(d_eff / 5.0)))
        d_quant = float(nf * 5.0)
        durs_ms.append(d_quant)

        g_eff = float(np.clip(raw_gains[i] * g_scale, g_min, g_max))
        if c["is_focused"] and c["is_vowel"]:
            g_eff = min(g_max + 1.2, g_eff + 1.1 * min(2, c["focus_level"]))
        gains_db.append(round(g_eff, 3))

        ph["dur_ms"] = d_quant
        ph["n_frames"] = nf
        ph["start_fr"] = cur_fr
        ph["end_fr"] = cur_fr + nf
        ph["gain_db"] = round(g_eff, 3)
        ph["f0_st_5"] = [round(float(x), 3) for x in f0_5_pred[i]]

        spans.append({
            "phone_idx": i,
            "start_fr": cur_fr,
            "end_fr": cur_fr + nf,
            "is_pause": False,
            "phrase_id": c["phrase_id"],
        })
        cur_fr += nf

        p_ms = float(c["pause_after_ms"])
        if include_pauses and i < n_p - 1 and p_ms >= 45.0:
            n_sil = max(1, int(round((p_ms * eff_pause * 0.65) / 5.0)))
            spans.append({
                "phone_idx": -1,
                "start_fr": cur_fr,
                "end_fr": cur_fr + n_sil,
                "is_pause": True,
                "phrase_id": c["phrase_id"],
            })
            cur_fr += n_sil

    n_fr = cur_fr
    if n_fr <= 0:
        return {
            "plan": plan,
            "phones": phones,
            "phone_durs_ms": durs_ms,
            "phone_gains_db": gains_db,
            "n_frames": 0,
            "f0_hz": np.zeros(0, dtype=np.float64),
            "f0_st": np.zeros(0, dtype=np.float64),
            "gain_db_frames": np.zeros(0, dtype=np.float64),
            "phone_spans": spans,
            "base_f0_hz": eff_base_f0,
            "is_question": bool(plan.get("is_question", False)),
        }

    # Build frame-level F0 target contour & relative energy trajectory
    f0_ridge_fr = np.zeros(n_fr, dtype=np.float64)
    f0_acc_fr = np.zeros(n_fr, dtype=np.float64)
    gain_fr = np.zeros(n_fr, dtype=np.float64)
    speech_mask = np.zeros(n_fr, dtype=bool)
    xp5 = np.linspace(0.0, 1.0, 5, dtype=np.float64)

    w_ridge = float(prm.get("f0_ridge_w", 0.72))
    w_acc = float(prm.get("f0_accent_w", 0.88))
    if embedded_in_zh:
        w_acc *= 0.48
        if n_p <= 12:
            eff_range *= 0.80
    elif dual_mode:
        w_acc *= 1.0 + 0.15 * calm
    nuc_boost = float(prm.get("f0_nuc_boost_st", 1.65))
    cnt_boost = float(prm.get("f0_content_boost_st", 1.05))
    sec_boost = float(prm.get("f0_sec_boost_st", 0.30))
    func_trough = float(prm.get("f0_func_trough_st", -0.95))

    for sp_row in spans:
        a, b = sp_row["start_fr"], sp_row["end_fr"]
        if sp_row["is_pause"] or b <= a:
            continue
        pi = sp_row["phone_idx"]
        c = ctxs[pi]
        nf = b - a
        speech_mask[a:b] = True

        u = np.linspace(0.0, 1.0, nf, dtype=np.float64)
        f0_ridge_fr[a:b] = np.interp(u, xp5, f0_5_pred[pi])
        gain_fr[a:b] = gains_db[pi]

        # Phrase-level pitch-accent & trough target
        hat = 0.35 + 0.65 * (0.5 - 0.5 * np.cos(math.tau * u))
        if c["is_vowel"]:
            if c["is_focused"] and c["stress_b"] in (1, 2):
                f0_acc_fr[a:b] = (nuc_boost + 0.85 * min(2, c["focus_level"])) * hat
            elif c["is_nuclear"] and c["stress_b"] == 1:
                f0_acc_fr[a:b] = nuc_boost * hat
            elif not c["is_func"] and c["stress_b"] == 1:
                f0_acc_fr[a:b] = cnt_boost * hat
            elif c["stress_b"] == 2:
                f0_acc_fr[a:b] = sec_boost * hat
            elif c["is_func"] and c["stress_b"] == 0:
                f0_acc_fr[a:b] = func_trough
            elif c["stress_b"] == 0:
                f0_acc_fr[a:b] = -0.35
        else:
            # Smooth consonant anchor toward adjacent vowel stress context
            prev_s = ctxs[pi - 1]["stress_b"] if pi > 0 else -1
            next_s = ctxs[pi + 1]["stress_b"] if pi + 1 < n_p else -1
            if c["is_func"]:
                f0_acc_fr[a:b] = 0.70 * func_trough
            elif next_s == 1 or prev_s == 1:
                f0_acc_fr[a:b] = 0.45 * cnt_boost
            elif next_s == 0 and prev_s == 0:
                f0_acc_fr[a:b] = -0.35

    if dual_mode:
        eff_range = float(np.clip(eff_range * (0.92 + 0.08 * drive), 0.40, 2.00))
        f0_ridge_soft = 6.0 * np.tanh(f0_ridge_fr / 6.0)
        f0_ridge_fr = calm * f0_ridge_soft + drive * f0_ridge_fr

    target_st = w_ridge * f0_ridge_fr + w_acc * f0_acc_fr

    # Continuous clause declination & median preservation
    p0, p1 = float(clause_pos[0]), float(clause_pos[1])
    decl_span = float(prm.get("f0_decl_span_st", 1.35))
    u_cl = np.linspace(p0, p1, n_fr, dtype=np.float64)
    decl_st = 0.5 * decl_span - decl_span * u_cl
    decl_mid = 0.5 * decl_span - decl_span * 0.5 * (p0 + p1)

    if np.any(speech_mask):
        med_base = float(np.median(target_st[speech_mask]))
        target_st[speech_mask] = eff_range * (target_st[speech_mask] - med_base) + decl_st[speech_mask]
        target_st[speech_mask] += decl_mid - float(np.median(target_st[speech_mask]))

    # Boundary tone on clause-final speech tail (question rise vs statement fall)
    q_flag = bool(plan.get("is_question", False))
    if is_clause_final and np.any(speech_mask):
        sp_idx = np.where(speech_mask)[0]
        tail_n = min(sp_idx.size, 34 if q_flag else 30)
        if tail_n >= 4:
            tail_pos = sp_idx[-tail_n:]
            u_tail = np.linspace(0.0, 1.0, tail_n, dtype=np.float64)
            if q_flag:
                q_rise = float(prm.get("f0_q_rise_st", 2.80))
                target_st[tail_pos] += (q_rise * eff_range) * (u_tail ** 1.35)
            else:
                s_fall = float(prm.get("f0_s_fall_st", 1.50))
                target_st[tail_pos] -= (s_fall * eff_range) * (u_tail ** 1.30)

    # 2nd-order Target-Approximation (qTA) smoothing within each pause-separated phrase
    f0_st = np.zeros(n_fr, dtype=np.float64)
    tau_ms = float(prm.get("f0_qta_tau_ms", 22.0))
    if dual_mode:
        tau_ms = max(tau_ms, 3.0 + 0.2 * calm)
    if embedded_in_zh:
        tau_ms = max(tau_ms, 16.0)
    idx = 0
    while idx < n_fr:
        if not speech_mask[idx]:
            idx += 1
            continue
        end = idx + 1
        while end < n_fr and speech_mask[end]:
            end += 1
        f0_st[idx:end] = _qta_smooth_2nd_order(target_st[idx:end], tau_ms=tau_ms, hop_ms=5.0)
        idx = end

    # Consonant CF0 microprosody on vowel onset after voiceless obstruent
    cf0_vl = float(prm.get("f0_cf0_vl_st", 0.60)) * (0.45 if embedded_in_zh else 1.0)
    cf0_eff = cf0_vl + (1.05 * calm if dual_mode else 0.0)
    cf0_kmax = 7 if dual_mode else 4
    cf0_decay = 16.0 if dual_mode else 14.0
    for sp_row in spans:
        if sp_row["is_pause"]:
            continue
        pi = sp_row["phone_idx"]
        c = ctxs[pi]
        if c["is_vowel"] and c["prev_ipa"] in VOICELESS_OBSTRUENTS:
            a, b = sp_row["start_fr"], sp_row["end_fr"]
            k_len = min(cf0_kmax, b - a)
            if k_len > 0:
                f0_st[a : a + k_len] += cf0_eff * np.exp(-np.arange(k_len, dtype=np.float64) * 5.0 / cf0_decay)

    # AR(1) vocal-fold micro-F0 flutter
    ar1_sig = float(prm.get("f0_ar1_sigma", 0.145)) * (0.65 if embedded_in_zh else 1.0)
    ar1_rho = float(prm.get("f0_ar1_rho", 0.72))
    if ar1_sig > 0.0 and n_fr > 4 and np.any(speech_mask):
        rng = np.random.default_rng(zlib.crc32(f"en_s10|{int(seed)}|{n_p}".encode("utf-8")) & 0xFFFFFFFF)
        eps = rng.normal(0.0, ar1_sig * math.sqrt(max(1e-4, 1.0 - ar1_rho * ar1_rho)), size=n_fr)
        ar1 = lfilter((1.0,), (1.0, -ar1_rho), eps)
        f0_st[speech_mask] += np.clip(ar1[speech_mask], -0.50, 0.50)

    f0_hz = np.zeros(n_fr, dtype=np.float64)
    f0_hz[speech_mask] = np.clip(eff_base_f0 * (2.0 ** (f0_st[speech_mask] / 12.0)), 60.0, 480.0)

    # Smooth relative energy trajectory across speech spans (eliminates choppy frame jumps)
    gain_smooth = np.zeros(n_fr, dtype=np.float64)
    idx = 0
    while idx < n_fr:
        if not speech_mask[idx]:
            idx += 1
            continue
        end = idx + 1
        while end < n_fr and speech_mask[end]:
            end += 1
        gain_smooth[idx:end] = _qta_smooth_2nd_order(gain_fr[idx:end], tau_ms=14.0, hop_ms=5.0)
        idx = end

    return {
        "plan": plan,
        "phones": phones,
        "phone_durs_ms": durs_ms,
        "phone_gains_db": gains_db,
        "n_frames": n_fr,
        "f0_hz": f0_hz,
        "f0_st": f0_st,
        "gain_db_frames": gain_smooth,
        "phone_spans": spans,
        "base_f0_hz": eff_base_f0,
        "is_question": q_flag,
    }


__all__ = [
    "MANNERS_VOCAB",
    "PROSODY_EN_STAGE10_PATH",
    "PROSODY_EN_STAGE10_SUMMARY_PATH",
    "ProsodyEnStage10Model",
    "clear_en_prosody_stage10_cache",
    "compute_phone_prosody_context",
    "extract_duration_features",
    "extract_prosody_features",
    "get_en_prosody_stage10_model",
    "predict_en_prosody_stage10",
]
