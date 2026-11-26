"""Non-neural lightweight `sklearn` prosody predictor (`< 500 KB`, `0.0 MB` neural weights).

Predicts phone duration (ms), word/phone relative energy offset (dB), and phrase-level
F0 anchor (semitones rel speaker median) strictly from zero-leakage training alignments
(`exp_v11_align_gmm` or expanded `exp_v12_data`), fixing `v11`'s flat English phrasing
and `-0.445 dB` content-vs-function word energy inversion.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT_IMPORT = Path(__file__).resolve().parent.parent
if str(ROOT_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_IMPORT))

from exp_v11_align_gmm import SR
from exp_v11_align_gmm.gmm_hmm_aligner import (
    EN_HELDOUT_DEFAULT,
    AlignedUtterance,
    PhoneBoundary,
    align_en_corpus_gmm,
    align_zh_corpus_gmm,
)
from exp_v11_eval.v11_metrics import EN_FUNCTION_WORDS
from exp_v12_prosody import DEFAULT_MODEL_PATH, EXP_DIR

MANNER_LIST = ("vowel", "diphthong", "nasal", "liquid", "glide", "fricative", "affricate", "stop", "flap", "other")
MANNER_TO_ID = {m: i for i, m in enumerate(MANNER_LIST)}


def _manner_id(m: str) -> int:
    return MANNER_TO_ID.get(m, len(MANNER_LIST) - 1)


def _stress_or_tone_code(s: str) -> tuple[int, int]:
    """Return (is_zh, code_int) from ``S0/S1/S2`` or ``T1..T5``."""
    if not s:
        return 0, 0
    if s.startswith("T") and s[1:].isdigit():
        return 1, int(s[1:])
    if s.startswith("S") and s[1:].isdigit():
        return 0, int(s[1:])
    return 0, 0


def _encode_feature_row(
    *,
    ipa: str,
    manner: str,
    prev_manner: str,
    next_manner: str,
    stress_or_tone: str,
    is_func: bool,
    word_pos_norm: float,
    phrase_pos_norm: float,
    is_final_word: bool,
    is_final_phone_in_word: bool,
    utt_n_phones: int,
    n_phones_in_word: int,
    clause_kind: str,
    ipa_vocab: dict[str, int],
) -> np.ndarray:
    """Compact fixed-width numeric feature vector (no high-dim text embeddings, 0 neural weights)."""
    n_v = len(ipa_vocab)
    n_m = len(MANNER_LIST)
    # Compact one-hot for current IPA + 3x manner one-hots + 12 scalar prosodic/positional features
    vec = np.zeros(n_v + 3 * n_m + 12, dtype=np.float32)
    if ipa in ipa_vocab:
        vec[ipa_vocab[ipa]] = 1.0
    off = n_v
    vec[off + _manner_id(manner)] = 1.0
    vec[off + n_m + _manner_id(prev_manner)] = 1.0
    vec[off + 2 * n_m + _manner_id(next_manner)] = 1.0
    s_off = off + 3 * n_m

    is_zh, code = _stress_or_tone_code(stress_or_tone)
    vec[s_off + 0] = float(is_zh)
    vec[s_off + 1] = float(code if not is_zh else 0)          # EN stress (0,1,2)
    vec[s_off + 2] = float(code if is_zh else 0)              # ZH tone (1..5)
    vec[s_off + 3] = 1.0 if is_func else 0.0
    vec[s_off + 4] = 0.0 if is_func else 1.0
    vec[s_off + 5] = float(word_pos_norm)
    vec[s_off + 6] = float(phrase_pos_norm)
    vec[s_off + 7] = 1.0 if is_final_word else 0.0
    vec[s_off + 8] = 1.0 if is_final_phone_in_word else 0.0
    vec[s_off + 9] = min(2.0, utt_n_phones / 50.0)
    vec[s_off + 10] = min(2.0, n_phones_in_word / 6.0)
    vec[s_off + 11] = 1.0 if clause_kind in ("question", "exclamation") else 0.0
    return vec


def _extract_samples_from_aligned(
    aligned_items: list[dict],
    ipa_vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Convert AlignedUtterance records into (X, y_log_dur, y_rel_db, y_f0_st, meta_rows)."""
    X_rows: list[np.ndarray] = []
    y_dur: list[float] = []
    y_db: list[float] = []
    y_f0: list[float] = []
    metas: list[dict] = []

    for item in aligned_items:
        au: AlignedUtterance = item["aligned"]
        feats = item["feats"]
        pbs: list[PhoneBoundary] = au.phones
        if not pbs:
            continue

        v_f0 = feats.f0_hz[feats.voiced & (feats.f0_hz > 0.0)]
        utt_f0_med = float(np.median(v_f0)) if v_f0.size else 160.0
        # Utterance voiced vowel median RMS as 0 dB reference
        v_phone_dbs = [p.rms_dbfs for p in pbs if p.manner in ("vowel", "diphthong") and p.rms_dbfs > -50.0]
        utt_db_ref = float(np.median(v_phone_dbs)) if v_phone_dbs else float(np.median([p.rms_dbfs for p in pbs]))

        n_tot = len(pbs)
        max_syl = max(1, max(p.syllable_idx for p in pbs))
        txt = au.text or ""
        c_kind = "question" if ("?" in txt or "？" in txt) else ("exclamation" if ("!" in txt or "！" in txt) else "statement")

        # Count phones per syllable_idx/word
        counts_by_syl: dict[int, int] = {}
        last_idx_by_syl: dict[int, int] = {}
        for idx, pb in enumerate(pbs):
            counts_by_syl[pb.syllable_idx] = counts_by_syl.get(pb.syllable_idx, 0) + 1
            last_idx_by_syl[pb.syllable_idx] = idx

        for idx, pb in enumerate(pbs):
            prev_m = pbs[idx - 1].manner if idx > 0 else "other"
            next_m = pbs[idx + 1].manner if idx + 1 < n_tot else "other"
            w_clean = pb.word.strip().lower()
            is_func = (au.language == "en") and (w_clean in EN_FUNCTION_WORDS)
            w_pos = pb.syllable_idx / max(1, max_syl)
            is_fin_w = pb.syllable_idx == max_syl
            is_fin_p = idx == last_idx_by_syl.get(pb.syllable_idx, idx)

            x_row = _encode_feature_row(
                ipa=pb.ipa,
                manner=pb.manner,
                prev_manner=prev_m,
                next_manner=next_m,
                stress_or_tone=pb.stress_or_tone,
                is_func=is_func,
                word_pos_norm=w_pos,
                phrase_pos_norm=w_pos,
                is_final_word=is_fin_w,
                is_final_phone_in_word=is_fin_p,
                utt_n_phones=n_tot,
                n_phones_in_word=counts_by_syl.get(pb.syllable_idx, 1),
                clause_kind=c_kind,
                ipa_vocab=ipa_vocab,
            )

            # Voiced F0 target for this phone (interpolated fallback to utterance contour if unvoiced)
            seg_f0 = feats.f0_hz[pb.p_start : pb.p_end]
            seg_v = feats.voiced[pb.p_start : pb.p_end] & (seg_f0 > 0.0)
            if np.any(seg_v):
                f0_rel = float(np.median(12.0 * np.log2(seg_f0[seg_v] / utt_f0_med)))
            else:
                # Linear phrase declination fallback so F0 target stays smooth
                f0_rel = 1.2 - 2.4 * w_pos
            f0_rel = float(np.clip(f0_rel, -8.0, 8.0))

            rel_db = float(np.clip(pb.rms_dbfs - utt_db_ref, -24.0, 8.0))
            log_dur = math.log(max(20.0, min(400.0, pb.dur_ms)))

            X_rows.append(x_row)
            y_dur.append(log_dur)
            y_db.append(rel_db)
            y_f0.append(f0_rel)
            metas.append({
                "utterance_id": au.utterance_id,
                "language": au.language,
                "word": pb.word,
                "syllable_idx": pb.syllable_idx,
                "is_func": is_func,
                "manner": pb.manner,
                "dur_ms": pb.dur_ms,
                "rel_db": rel_db,
                "f0_rel": f0_rel,
            })

    return (
        np.stack(X_rows, axis=0) if X_rows else np.zeros((0, len(ipa_vocab) + 3 * len(MANNER_LIST) + 12), dtype=np.float32),
        np.asarray(y_dur, dtype=np.float64),
        np.asarray(y_db, dtype=np.float64),
        np.asarray(y_f0, dtype=np.float64),
        metas,
    )


@dataclass
class ProsodyModelV12:
    """Compact non-neural sklearn prosody predictor (< 500 KB, 0.0 MB neural weights)."""

    ipa_vocab: dict[str, int]
    dur_model: object
    db_model: object
    f0_model: object
    model_kind: str
    content_func_db_learned: float
    train_stats: dict

    def save(self, path: Path | str = DEFAULT_MODEL_PATH) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        p.write_bytes(raw)
        return len(raw)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_MODEL_PATH) -> ProsodyModelV12:
        return pickle.loads(Path(path).read_bytes())


def fit_prosody_v12(
    en_res: dict | None = None,
    zh_res: dict | None = None,
    *,
    model_kind: str = "hgbr",
    save_path: Path | str | None = DEFAULT_MODEL_PATH,
    exclude_en_ids: frozenset[str] | set[str] = EN_HELDOUT_DEFAULT,
    exclude_zh_ids: set[str] | None = None,
) -> tuple[ProsodyModelV12, dict]:
    """Train duration, relative energy, and F0 predictors strictly on training alignments."""
    if en_res is None:
        en_res = align_en_corpus_gmm(exclude_ids=exclude_en_ids, align_heldout=True, n_em=4)
    if zh_res is None:
        zh_res = align_zh_corpus_gmm(exclude_ids=exclude_zh_ids, n_em=4)

    train_items = list(en_res["utterances"].values()) + list(zh_res["utterances"].values())
    vocab_ipas = sorted({pb.ipa for item in train_items for pb in item["aligned"].phones})
    ipa_vocab = {p: i for i, p in enumerate(vocab_ipas)}

    X_tr, y_dur_tr, y_db_tr, y_f0_tr, metas_tr = _extract_samples_from_aligned(train_items, ipa_vocab)

    if model_kind == "ridge":
        dur_m = Ridge(alpha=8.0).fit(X_tr, y_dur_tr)
        db_m = Ridge(alpha=8.0).fit(X_tr, y_db_tr)
        f0_m = Ridge(alpha=8.0).fit(X_tr, y_f0_tr)
    else:
        # Compact HistGradientBoostingRegressor (< 300 KB total for all 3 models!)
        kw = dict(max_iter=35, max_depth=4, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=2.0, random_state=42)
        dur_m = HistGradientBoostingRegressor(**kw).fit(X_tr, y_dur_tr)
        db_m = HistGradientBoostingRegressor(**kw).fit(X_tr, y_db_tr)
        f0_m = HistGradientBoostingRegressor(**kw).fit(X_tr, y_f0_tr)

    # Compute true & predicted word-level EN content-vs-function contrast on training set
    pred_db_tr = db_m.predict(X_tr)
    en_func_dbs = [float(p) for p, mt in zip(pred_db_tr, metas_tr) if mt["language"] == "en" and mt["is_func"] and mt["manner"] in ("vowel", "diphthong")]
    en_cont_dbs = [float(p) for p, mt in zip(pred_db_tr, metas_tr) if mt["language"] == "en" and (not mt["is_func"]) and mt["manner"] in ("vowel", "diphthong")]
    learned_gap = float(np.median(en_cont_dbs) - np.median(en_func_dbs)) if (en_func_dbs and en_cont_dbs) else 1.5

    model = ProsodyModelV12(
        ipa_vocab=ipa_vocab,
        dur_model=dur_m,
        db_model=db_m,
        f0_model=f0_m,
        model_kind=model_kind,
        content_func_db_learned=round(learned_gap, 3),
        train_stats={"n_train_phones": int(len(X_tr)), "n_ipa_vocab": len(ipa_vocab)},
    )
    model_bytes = model.save(save_path) if save_path is not None else len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))

    # Held-out verification on EN slt[30:40] & bdl[30:40] if available
    held_metrics = {"model_bytes": int(model_bytes), "n_train_phones": int(len(X_tr))}
    held_items = list(en_res.get("heldout_utterances", {}).values())
    if held_items:
        X_ho, y_dur_ho, y_db_ho, y_f0_ho, metas_ho = _extract_samples_from_aligned(held_items, ipa_vocab)
        p_dur_ho = np.exp(dur_m.predict(X_ho))
        y_dur_ms_ho = np.exp(y_dur_ho)
        p_db_ho = db_m.predict(X_ho)
        p_f0_ho = f0_m.predict(X_ho)

        def _corr(a: np.ndarray, b: np.ndarray) -> float:
            if len(a) < 3 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
                return 0.0
            return round(float(np.corrcoef(a, b)[0, 1]), 4)

        # Group by utterance + word index via duration-weighted linear power (matching waveform RMS!)
        def _word_contrast(db_vals: np.ndarray, durs_ms: np.ndarray) -> tuple[float, float]:
            by_w: dict[tuple[str, int], tuple[bool, float, float]] = {}
            for val, dur, mt in zip(db_vals, durs_ms, metas_ho):
                key = (mt["utterance_id"], mt["syllable_idx"])
                is_f, num, den = by_w.get(key, (mt["is_func"], 0.0, 0.0))
                pwr = 10.0 ** (float(val) / 10.0)
                by_w[key] = (is_f, num + float(dur) * pwr, den + float(dur))
            f_vals = [10.0 * math.log10(max(1e-12, num / max(1e-3, den))) for is_f, num, den in by_w.values() if is_f]
            c_vals = [10.0 * math.log10(max(1e-12, num / max(1e-3, den))) for is_f, num, den in by_w.values() if not is_f]
            return (float(np.median(c_vals) - np.median(f_vals)) if (f_vals and c_vals) else 0.0), float(len(c_vals))

        # Word-level duration ratio (func / content)
        by_wdur_true: dict[tuple[str, int], tuple[bool, float]] = {}
        by_wdur_pred: dict[tuple[str, int], tuple[bool, float]] = {}
        for yt, yp, mt in zip(y_dur_ms_ho, p_dur_ho, metas_ho):
            key = (mt["utterance_id"], mt["syllable_idx"])
            by_wdur_true[key] = (mt["is_func"], by_wdur_true.get(key, (mt["is_func"], 0.0))[1] + float(yt))
            by_wdur_pred[key] = (mt["is_func"], by_wdur_pred.get(key, (mt["is_func"], 0.0))[1] + float(yp))

        def _dur_ratio(mp: dict) -> float:
            f_d = [v for is_f, v in mp.values() if is_f]
            c_d = [v for is_f, v in mp.values() if not is_f]
            return round(float(np.median(f_d) / max(1e-3, np.median(c_d))), 4) if (f_d and c_d) else 1.0

        true_db_gap, _ = _word_contrast(y_db_ho, y_dur_ms_ho)
        pred_db_gap, _ = _word_contrast(p_db_ho, p_dur_ho)
        held_metrics.update({
            "n_heldout_phones": int(len(X_ho)),
            "dur_corr_r": _corr(p_dur_ho, y_dur_ms_ho),
            "dur_rmse_ms": round(float(np.sqrt(np.mean((p_dur_ho - y_dur_ms_ho) ** 2))), 2),
            "rel_db_corr_r": _corr(p_db_ho, y_db_ho),
            "f0_corr_r": _corr(p_f0_ho, y_f0_ho),
            "real_func_to_content_dur_ratio": _dur_ratio(by_wdur_true),
            "pred_func_to_content_dur_ratio": _dur_ratio(by_wdur_pred),
            "real_content_minus_func_db": round(true_db_gap, 3),
            "pred_content_minus_func_db": round(pred_db_gap, 3),
        })

    return model, held_metrics


def load_prosody_v12(path: Path | str = DEFAULT_MODEL_PATH) -> ProsodyModelV12:
    return ProsodyModelV12.load(path)


def apply_prosody_v12(
    ir_plan: dict,
    model: ProsodyModelV12 | None = None,
    *,
    strength_dur: float = 1.0,
    strength_gain: float = 1.0,
    strength_f0: float = 1.0,
) -> dict:
    """Apply non-neural learned prosody predictions to a ``plan_v10`` / ``cf.plan`` dict."""
    if model is None:
        model = load_prosody_v12(DEFAULT_MODEL_PATH)

    out = json.loads(json.dumps(ir_plan))
    units = out.get("units", [])
    if not units:
        return out

    # Flatten phone entries across all units to compute contextual features cleanly
    phone_feats: list[np.ndarray] = []
    phone_unit_idx: list[int] = []
    phone_is_func: list[bool] = []
    phone_is_vowel: list[bool] = []
    phone_langs: list[str] = []

    flat_phones: list[dict] = []
    n_u = len(units)
    txt = str(out.get("text", ""))
    clause_kind = "question" if ("?" in txt or "？" in txt) else ("exclamation" if ("!" in txt or "！" in txt) else "statement")

    for ui, u in enumerate(units):
        lang = str(u.get("lang", "zh"))
        tok = str(u.get("token") or u.get("hanzi") or "").strip().lower()
        is_func = (lang == "en") and (tok in EN_FUNCTION_WORDS)
        ipas = u.get("ipa_phones") or []
        for pi, p in enumerate(ipas):
            flat_phones.append({
                "u_idx": ui,
                "lang": lang,
                "ipa": p,
                "manner": "vowel" if p in ("ɑ","æ","ə","ɚ","ɛ","eɪ","ɪ","i","oʊ","ɔ","ɔɪ","ʊ","u","ʌ","aɪ","aʊ","a","e","o","y","ɤ","ɨ","ʅ","ʮ","aɚ") else "stop",
                "stress_or_tone": str(u.get("stress_or_tone", "S0")),
                "is_func": is_func,
                "w_pos": ui / max(1, n_u - 1),
                "is_fin_w": (ui == n_u - 1),
                "is_fin_p": (pi == len(ipas) - 1),
                "n_w_p": len(ipas),
            })

    # Better manner lookup via IPA_SPECS
    from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS

    for fp in flat_phones:
        sp = IPA_SPECS.get(fp["ipa"])
        if sp:
            fp["manner"] = sp.manner

    n_tot = len(flat_phones)
    for idx, fp in enumerate(flat_phones):
        prev_m = flat_phones[idx - 1]["manner"] if idx > 0 else "other"
        next_m = flat_phones[idx + 1]["manner"] if idx + 1 < n_tot else "other"
        x_row = _encode_feature_row(
            ipa=fp["ipa"],
            manner=fp["manner"],
            prev_manner=prev_m,
            next_manner=next_m,
            stress_or_tone=fp["stress_or_tone"],
            is_func=fp["is_func"],
            word_pos_norm=fp["w_pos"],
            phrase_pos_norm=fp["w_pos"],
            is_final_word=fp["is_fin_w"],
            is_final_phone_in_word=fp["is_fin_p"],
            utt_n_phones=n_tot,
            n_phones_in_word=fp["n_w_p"],
            clause_kind=clause_kind,
            ipa_vocab=model.ipa_vocab,
        )
        phone_feats.append(x_row)
        phone_unit_idx.append(fp["u_idx"])
        phone_is_func.append(fp["is_func"])
        phone_is_vowel.append(fp["manner"] in ("vowel", "diphthong"))
        phone_langs.append(fp["lang"])

    if not phone_feats:
        return out

    X = np.stack(phone_feats, axis=0)
    pred_durs = np.clip(np.exp(model.dur_model.predict(X)), 22.0, 260.0)
    pred_dbs = model.db_model.predict(X)
    pred_f0s = np.clip(model.f0_model.predict(X), -6.0, 6.0)

    # Center predicted energy offsets around 0 dB across utterance so content > func cleanly
    en_vowel_dbs = [float(d) for d, l, iv in zip(pred_dbs, phone_langs, phone_is_vowel) if l == "en" and iv]
    db_center = float(np.mean(en_vowel_dbs)) if en_vowel_dbs else float(np.mean(pred_dbs))

    for ui, u in enumerate(units):
        idxs = [k for k, u_i in enumerate(phone_unit_idx) if u_i == ui]
        if not idxs:
            continue
        lang = str(u.get("lang", "zh"))
        tok = str(u.get("token") or u.get("hanzi") or "").strip().lower()
        is_func = (lang == "en") and (tok in EN_FUNCTION_WORDS)

        p_durs = [round(float(pred_durs[k]), 2) for k in idxs]
        dur_pred_ms = float(sum(p_durs))
        dur_orig_ms = float(u.get("duration_ms", dur_pred_ms))
        # Blend learned duration smoothly while clamping extreme fast/slow outliers
        dur_new = dur_orig_ms + strength_dur * (dur_pred_ms - dur_orig_ms)
        if lang == "en" and is_func:
            dur_new = max(55.0, min(dur_new, 0.85 * max(90.0, dur_orig_ms)))
        elif lang == "en":
            dur_new = max(110.0, dur_new)
        u["phone_durations_ms"] = p_durs
        u["duration_ms"] = round(float(np.clip(dur_new, 55.0, 650.0)), 1)

        # Unit gain_db: use vowel/sonorant predicted energy offset (avoid stop closure dragging gain down!)
        v_dbs = [float(pred_dbs[k]) - db_center for k in idxs if phone_is_vowel[k]]
        unit_db = float(np.mean(v_dbs)) if v_dbs else float(np.mean([pred_dbs[k] - db_center for k in idxs]))
        # Subtle explicit safeguard so content words always lead adjacent function words naturally
        if lang == "en" and is_func:
            unit_db = min(unit_db, -0.75)
        elif lang == "en":
            unit_db = max(unit_db, 0.45)
        unit_db = float(np.clip(strength_gain * unit_db, -3.0, 3.0))
        u["gain_db"] = round(unit_db, 3)
        u["gain"] = round(0.85 * (10.0 ** (unit_db / 20.0)), 4)

        # Phrase-level F0 anchor trajectory (5 control points in semitones rel speaker median)
        u_f0 = float(np.mean([pred_f0s[k] for k in idxs]))
        orig_st = list(u.get("pitch_st") or [0.0, 0.0, 0.0, 0.0, 0.0])
        if lang == "en":
            # Smooth declination + lexical stress prominence peak on middle vowel points
            stress_bump = 1.1 if u.get("stress_or_tone") == "S1" else (-0.8 if is_func else 0.3)
            shape = np.array([0.0, 0.45, 0.90, 0.55, 0.10]) * stress_bump
            new_st = (u_f0 + shape) * strength_f0 + (1.0 - strength_f0) * np.asarray(orig_st, dtype=np.float64)
            u["pitch_st"] = [round(float(x), 3) for x in np.clip(new_st, -7.0, 7.0)]
            u["pitch_prosody_st"] = [round(float(u_f0), 3)] * 5
        else:
            # Keep Mandarin lexical tone contour intact while adding gentle phrase-level F0 offset
            lex = np.asarray(u.get("pitch_lexical_st") or orig_st, dtype=np.float64)
            offset = 0.45 * strength_f0 * u_f0
            u["pitch_prosody_st"] = [round(float(offset), 3)] * 5
            u["pitch_st"] = [round(float(x + offset), 3) for x in lex]

    return out


if __name__ == "__main__":
    _m, metrics = fit_prosody_v12(model_kind="hgbr")
    print(json.dumps(metrics, indent=2))
