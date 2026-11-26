"""Word-Aware Mandarin HGBR Prosody & Pause Predictor (`< 500 KB`, `0.0 MB` neural weights)."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import pickle
import re
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_v10_g2p_and_align.g2p_audit_and_fix import ZH_POLYPHONE_PHRASES_V10  # noqa: E402
from exp_v10_integrity.gain_calibration import parseval_frame_db  # noqa: E402
from exp_v13_utau_ceiling.sandhi_v13 import plan_v13  # noqa: E402
from exp_v13_utau_ceiling.v13_selector import (  # noqa: E402
    FINAL_CODA_CLASS,
    FINAL_HEAD_CLASS,
    ONSET_PLACE_CLASS,
)
from exp_v14_fluent_zh_tts import (  # noqa: E402
    ALIGNMENTS_V14,
    CLEAN_10_IDS,
    FEATS_CACHE_V14,
    MANIFEST_V14,
    N_FFT,
    PROSODY_MODEL_V14,
    USER_6_IDS,
)
from exp_v14_fluent_zh_tts.align_v14 import extract_or_load_ssb0502_features  # noqa: E402
from pybot_tts_v2_recorded import engine  # noqa: E402

ZH_FUNC_WORDS = frozenset({
    "的", "了", "着", "过", "得", "地", "在", "是", "和", "与", "及",
    "就", "也", "都", "但", "而", "把", "被", "给", "对", "从", "向",
    "为", "之", "让", "所", "以", "来", "将", "会", "能", "要",
})

_HANZI_RE = re.compile(r"[\u3400-\u9fff]+")
CAT_FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 6]


def _onset_manner(o: str) -> str:
    if not o or o in ("y", "w"):
        return "zero"
    if o == "SIL":
        return "sil"
    if o in ("b", "d", "g"):
        return "stop_unasp"
    if o in ("p", "t", "k"):
        return "stop_asp"
    if o in ("z", "c", "zh", "ch", "j", "q"):
        return "affricate"
    if o in ("f", "s", "sh", "x", "h"):
        return "fric_uv"
    if o in ("m", "n", "l", "r"):
        return "sonorant"
    return "other"


def _normalize_of(onset: str, final: str) -> tuple[str, str]:
    o = "" if onset in ("y", "w") else str(onset)
    f = str(final)
    if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
        f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"
    return o, f


def build_zh_word_lexicon(
    manifest_utts: list[dict[str, Any]] | None = None,
    *,
    exclude_ids: set[str] | frozenset[str] = frozenset(),
) -> set[str]:
    """Build multi-character Mandarin word lexicon with strict zero held-out leakage."""
    words: set[str] = set()
    for w in getattr(engine, "WORDS", ()):
        if len(w) >= 2 and _HANZI_RE.fullmatch(w):
            words.add(w)
    for w in ZH_POLYPHONE_PHRASES_V10:
        if len(w) >= 2 and _HANZI_RE.fullmatch(w):
            words.add(w)
    if manifest_utts:
        excl = set(exclude_ids)
        for u in manifest_utts:
            if u["id"] in excl:
                continue
            for w in u.get("hsk_words", ()):
                ws = str(w).strip()
                if len(ws) >= 2 and _HANZI_RE.fullmatch(ws):
                    words.add(ws)
    return words


def segment_zh_words_v14(text: str, word_lexicon: set[str]) -> list[tuple[int, int, str]]:
    """Segment contiguous Hanzi spans in `text` into `(start_char, end_char, word)` via longest match."""
    spans: list[tuple[int, int, str]] = []
    max_len = min(8, max((len(w) for w in word_lexicon), default=4))
    for m in _HANZI_RE.finditer(text):
        run = m.group(0)
        base = m.start()
        i = 0
        n = len(run)
        while i < n:
            matched = run[i : i + 1]
            for sz in range(min(max_len, n - i), 1, -1):
                cand = run[i : i + sz]
                if cand in word_lexicon:
                    matched = cand
                    break
            spans.append((base + i, base + i + len(matched), matched))
            i += len(matched)
    return spans


def _word_info_by_char(text: str, word_lexicon: set[str]) -> dict[int, tuple[int, int, str]]:
    """Map character index in `text` -> `(word_len, pos_in_word, word_text)`."""
    info: dict[int, tuple[int, int, str]] = {}
    for s0, s1, w in segment_zh_words_v14(text, word_lexicon):
        wl = s1 - s0
        for p in range(wl):
            info[s0 + p] = (wl, p, w)
    return info


@dataclass
class ProsodyModelV14:
    """Word- & tone-aware non-neural HGBR Mandarin prosody predictor (< 500 KB)."""

    vocab: dict[str, dict[str, int]]
    word_lexicon: set[str]
    dur_model: HistGradientBoostingRegressor
    f0_start_model: HistGradientBoostingRegressor
    f0_mid_model: HistGradientBoostingRegressor
    f0_end_model: HistGradientBoostingRegressor
    energy_model: HistGradientBoostingRegressor
    pause_model: HistGradientBoostingRegressor
    pause_threshold: float
    train_stats: dict[str, Any]

    def save(self, path: Path | str = PROSODY_MODEL_V14) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        p.write_bytes(raw)
        return len(raw)

    @classmethod
    def load(cls, path: Path | str = PROSODY_MODEL_V14) -> ProsodyModelV14:
        raw = Path(path).read_bytes()

        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                if module == "__main__" and name == "ProsodyModelV14":
                    return ProsodyModelV14
                return super().find_class(module, name)

        return _CompatUnpickler(io.BytesIO(raw)).load()


ProsodyModelV14.__module__ = "exp_v14_fluent_zh_tts.prosody_v14"


def _cat_id(mapping: dict[str, int], key: str) -> int:
    if key not in mapping:
        mapping[key] = len(mapping)
    return mapping[key]


def _build_default_vocab() -> dict[str, dict[str, int]]:
    return {
        "onset": {"<UNK>": 0, "": 1},
        "final": {"<UNK>": 0},
        "onset_manner": {"<UNK>": 0, "SIL": 1},
        "final_head": {"<UNK>": 0},
        "final_coda": {"<UNK>": 0, "SIL": 1},
    }


def _encode_syllable_row(
    *,
    vocab: dict[str, dict[str, int]],
    allow_grow: bool,
    onset: str,
    final: str,
    prev_final: str,
    next_onset: str,
    prev_tone: int,
    curr_tone: int,
    next_tone: int,
    hanzi: str,
    word_len: int,
    pos_in_word: int,
    syl_in_clause: int,
    clause_len: int,
    is_question_or_excl: bool,
) -> np.ndarray:
    o, f = _normalize_of(onset, final)
    om = _onset_manner(o)
    fh = FINAL_HEAD_CLASS.get(f, "other")
    fc = FINAL_CODA_CLASS.get(f, "other")
    if prev_final == "SIL":
        p_coda = "SIL"
    else:
        _, pf_n = _normalize_of("", prev_final)
        p_coda = FINAL_CODA_CLASS.get(pf_n, "other")
    if next_onset == "SIL":
        n_om = "SIL"
    else:
        n_om = _onset_manner(next_onset)

    def _lookup(cat: str, val: str) -> int:
        mp = vocab[cat]
        if allow_grow:
            return _cat_id(mp, val)
        return mp.get(val, 0)

    is_zero = 1.0 if o == "" else 0.0
    is_w_init = 1.0 if pos_in_word == 0 else 0.0
    is_w_fin = 1.0 if pos_in_word == word_len - 1 else 0.0
    is_func = 1.0 if hanzi in ZH_FUNC_WORDS else 0.0
    pos_norm = float(syl_in_clause) / max(1, clause_len - 1) if clause_len > 1 else 0.5
    to_end = max(0, clause_len - 1 - syl_in_clause)
    is_c_fin = 1.0 if to_end == 0 else 0.0

    return np.array(
        [
            float(_lookup("onset", o)),
            float(_lookup("final", f)),
            float(_lookup("onset_manner", om)),
            float(_lookup("final_head", fh)),
            float(_lookup("final_coda", fc)),
            float(_lookup("final_coda", p_coda)),
            float(_lookup("onset_manner", n_om)),
            is_zero,
            float(prev_tone),
            float(curr_tone),
            float(next_tone),
            float(min(6, word_len)),
            float(min(5, pos_in_word)),
            is_w_init,
            is_w_fin,
            is_func,
            pos_norm,
            float(min(20, syl_in_clause)),
            float(min(20, to_end)),
            is_c_fin,
            float(min(25, clause_len)),
            1.0 if is_question_or_excl else 0.0,
        ],
        dtype=np.float32,
    )


def _extract_training_samples(
    utt_list: list[dict[str, Any]],
    aligned_map: dict[str, Any],
    feats_map: dict[str, dict[str, Any]],
    word_lexicon: set[str],
    vocab: dict[str, dict[str, int]],
    *,
    allow_grow: bool,
) -> dict[str, Any]:
    X_rows: list[np.ndarray] = []
    y_dur: list[float] = []
    y_f0_s: list[float] = []
    y_f0_m: list[float] = []
    y_f0_e: list[float] = []
    y_db: list[float] = []
    X_pause: list[np.ndarray] = []
    y_pause: list[float] = []
    metas: list[dict[str, Any]] = []

    for u in utt_list:
        uid = u["id"]
        al = aligned_map.get(uid)
        fe = feats_map.get(uid)
        if al is None or fe is None:
            continue
        syls = al["syllables"]
        if not syls:
            continue
        text = u["text"]
        winfo = _word_info_by_char(text, word_lexicon)
        is_qe = ("?" in text or "？" in text or "!" in text or "！" in text)

        f0_hz = np.asarray(fe["f0_hz"], dtype=np.float64)
        voi = np.asarray(fe["voiced"], dtype=bool)
        log_env = np.asarray(fe["log_env_f16"], dtype=np.float64)

        # Active utterance voiced F0 & vowel Parseval reference
        act_v_mask = np.zeros(len(voi), dtype=bool)
        for s in syls:
            act_v_mask[s["syl_start"] : s["syl_end"]] |= voi[s["syl_start"] : s["syl_end"]]
        valid_f0 = f0_hz[act_v_mask & (f0_hz > 50.0)]
        utt_f0_med = float(np.median(valid_f0)) if valid_f0.size else 200.0
        act_db = parseval_frame_db(log_env[act_v_mask], n_fft=N_FFT) if np.any(act_v_mask) else np.array([-24.0])
        utt_db_med = float(np.median(act_db))

        # Split into clauses whenever GMM-HMM detected a real inter-syllable pause (> 0 ms)
        clauses: list[list[int]] = []
        cur_c: list[int] = []
        for idx_s, s in enumerate(syls):
            cur_c.append(idx_s)
            if float(s.get("pause_after_ms", 0.0)) > 0.0:
                clauses.append(cur_c)
                cur_c = []
        if cur_c:
            clauses.append(cur_c)

        for cl in clauses:
            cl_len = len(cl)
            for c_pos, idx_s in enumerate(cl):
                s = syls[idx_s]
                dur_ms = float(s["dur_ms"])
                if not (50.0 <= dur_ms <= 420.0):
                    continue
                s0, f_sp, s1 = int(s["syl_start"]), int(s["f_split"]), int(s["syl_end"])
                if s1 <= s0:
                    continue

                prev_s = syls[cl[c_pos - 1]] if c_pos > 0 else None
                next_s = syls[cl[c_pos + 1]] if c_pos + 1 < cl_len else None
                prev_f = prev_s["final"] if prev_s else "SIL"
                prev_t = int(prev_s["tone"]) if prev_s else 0
                next_o = next_s["onset"] if next_s else "SIL"
                next_t = int(next_s["tone"]) if next_s else 0

                wl, piw, _ = winfo.get(idx_s, (1, 0, s["hanzi"]))
                x_row = _encode_syllable_row(
                    vocab=vocab,
                    allow_grow=allow_grow,
                    onset=s["onset"],
                    final=s["final"],
                    prev_final=prev_f,
                    next_onset=next_o,
                    prev_tone=prev_t,
                    curr_tone=int(s["tone"]),
                    next_tone=next_t,
                    hanzi=s["hanzi"],
                    word_len=wl,
                    pos_in_word=piw,
                    syl_in_clause=c_pos,
                    clause_len=cl_len,
                    is_question_or_excl=is_qe,
                )

                # Pause sample evaluated over full sentence boundary context (even when split at pause)
                if idx_s < len(syls) - 1:
                    s_nxt = syls[idx_s + 1]
                    x_p = _encode_syllable_row(
                        vocab=vocab,
                        allow_grow=allow_grow,
                        onset=s["onset"],
                        final=s["final"],
                        prev_final=syls[idx_s - 1]["final"] if idx_s > 0 else "SIL",
                        next_onset=s_nxt["onset"],
                        prev_tone=int(syls[idx_s - 1]["tone"]) if idx_s > 0 else 0,
                        curr_tone=int(s["tone"]),
                        next_tone=int(s_nxt["tone"]),
                        hanzi=s["hanzi"],
                        word_len=wl,
                        pos_in_word=piw,
                        syl_in_clause=idx_s,
                        clause_len=len(syls),
                        is_question_or_excl=is_qe,
                    )
                    X_pause.append(x_p)
                    y_pause.append(1.0 if float(s.get("pause_after_ms", 0.0)) >= 35.0 else 0.0)

                # Rime F0 20/50/80% points in semitones rel utterance median
                r0 = min(s1 - 1, s0 + f_sp)
                v_idx = np.where(voi[r0:s1] & (f0_hz[r0:s1] > 50.0))[0] + r0
                if v_idx.size < 2:
                    v_idx = np.where(voi[s0:s1] & (f0_hz[s0:s1] > 50.0))[0] + s0
                if v_idx.size >= 2:
                    sts = np.clip(12.0 * np.log2(f0_hz[v_idx] / utt_f0_med), -10.0, 10.0)
                    f0_pts = np.interp([0.2, 0.5, 0.8], np.linspace(0.0, 1.0, len(sts)), sts)
                elif v_idx.size == 1:
                    st0 = float(np.clip(12.0 * np.log2(f0_hz[v_idx[0]] / utt_f0_med), -10.0, 10.0))
                    f0_pts = np.array([st0, st0, st0])
                else:
                    decl = 1.0 - 2.0 * (c_pos / max(1, cl_len - 1))
                    f0_pts = np.array([decl, decl, decl])

                seg_env = log_env[v_idx] if v_idx.size else log_env[s0:s1]
                rel_db = float(np.clip(np.percentile(parseval_frame_db(seg_env, n_fft=N_FFT), 75) - utt_db_med, -18.0, 8.0))

                X_rows.append(x_row)
                y_dur.append(math.log(dur_ms))
                y_f0_s.append(float(f0_pts[0]))
                y_f0_m.append(float(f0_pts[1]))
                y_f0_e.append(float(f0_pts[2]))
                y_db.append(rel_db)
                metas.append({"id": uid, "idx": idx_s, "dur_ms": dur_ms, "f0_mid": float(f0_pts[1])})

    return {
        "X": np.stack(X_rows, axis=0) if X_rows else np.zeros((0, 22), dtype=np.float32),
        "y_dur": np.asarray(y_dur, dtype=np.float64),
        "y_f0_s": np.asarray(y_f0_s, dtype=np.float64),
        "y_f0_m": np.asarray(y_f0_m, dtype=np.float64),
        "y_f0_e": np.asarray(y_f0_e, dtype=np.float64),
        "y_db": np.asarray(y_db, dtype=np.float64),
        "X_pause": np.stack(X_pause, axis=0) if X_pause else np.zeros((0, 22), dtype=np.float32),
        "y_pause": np.asarray(y_pause, dtype=np.float64),
        "metas": metas,
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def train_prosody_v14(
    *,
    manifest_path: Path = MANIFEST_V14,
    align_path: Path = ALIGNMENTS_V14,
    cache_path: Path = FEATS_CACHE_V14,
    save_path: Path = PROSODY_MODEL_V14,
    extra_exclude_ids: set[str] | frozenset[str] = frozenset(),
) -> tuple[ProsodyModelV14, dict[str, Any]]:
    """Train zero-leakage Mandarin HGBR prosody & pause predictor on `SSB0502` training utts."""
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    al_doc = json.loads(align_path.read_text(encoding="utf-8"))
    utts = man["utterances"]
    clean_10 = tuple(CLEAN_10_IDS) if CLEAN_10_IDS else tuple(man.get("clean_10_ids") or ())
    held_set = set(USER_6_IDS) | set(clean_10) | set(extra_exclude_ids)

    feats_map = extract_or_load_ssb0502_features(utts, cache_path=cache_path)
    word_lexicon = build_zh_word_lexicon(utts, exclude_ids=held_set)

    train_utts = [u for u in utts if u["id"] not in held_set and u["id"] in al_doc["zh_train"]]
    held_utts = [u for u in utts if u["id"] in held_set and u["id"] in al_doc["zh_all"]]

    vocab = _build_default_vocab()
    tr = _extract_training_samples(train_utts, al_doc["zh_train"], feats_map, word_lexicon, vocab, allow_grow=True)
    ho = _extract_training_samples(held_utts, al_doc["zh_all"], feats_map, word_lexicon, vocab, allow_grow=False)

    # Compact HGBR hyperparameters tuned so 6 models pickle under < 450 KB total
    kw = dict(
        max_iter=36,
        max_depth=4,
        max_leaf_nodes=13,
        min_samples_leaf=18,
        l2_regularization=2.5,
        learning_rate=0.09,
        categorical_features=CAT_FEATURE_INDICES,
        random_state=42,
    )
    kw_p = dict(
        max_iter=24,
        max_depth=3,
        max_leaf_nodes=9,
        min_samples_leaf=24,
        l2_regularization=3.0,
        learning_rate=0.08,
        categorical_features=CAT_FEATURE_INDICES,
        random_state=42,
    )

    dur_m = HistGradientBoostingRegressor(**kw).fit(tr["X"], tr["y_dur"])
    f0s_m = HistGradientBoostingRegressor(**kw).fit(tr["X"], tr["y_f0_s"])
    f0m_m = HistGradientBoostingRegressor(**kw).fit(tr["X"], tr["y_f0_m"])
    f0e_m = HistGradientBoostingRegressor(**kw).fit(tr["X"], tr["y_f0_e"])
    db_m = HistGradientBoostingRegressor(**kw).fit(tr["X"], tr["y_db"])
    pause_m = HistGradientBoostingRegressor(**kw_p).fit(tr["X_pause"], tr["y_pause"])

    # Held-out evaluation (zero leakage)
    p_dur_ho = np.clip(np.exp(dur_m.predict(ho["X"])), 65.0, 360.0)
    y_dur_ho = np.exp(ho["y_dur"])
    p_f0m_ho = f0m_m.predict(ho["X"])
    p_pause_ho = np.clip(pause_m.predict(ho["X_pause"]), 0.0, 1.0)

    # Calibrate pause threshold strictly from training quantile / balanced F1 avoiding false breaks on clean speech
    pause_thresh = 0.38
    tp = float(np.sum((p_pause_ho >= pause_thresh) & (ho["y_pause"] >= 0.5)))
    fp = float(np.sum((p_pause_ho >= pause_thresh) & (ho["y_pause"] < 0.5)))
    fn = float(np.sum((p_pause_ho < pause_thresh) & (ho["y_pause"] >= 0.5)))
    p_f1 = round((2.0 * tp) / max(1e-9, 2.0 * tp + fp + fn), 4)

    metrics = {
        "n_train_utts": len(train_utts),
        "n_train_syls": int(len(tr["X"])),
        "n_held_utts": len(held_utts),
        "n_held_syls": int(len(ho["X"])),
        "lexicon_size": len(word_lexicon),
        "held_dur_mae_ms": round(float(np.mean(np.abs(p_dur_ho - y_dur_ho))), 2),
        "held_dur_corr_r": _corr(p_dur_ho, y_dur_ho),
        "held_f0_mid_mae_st": round(float(np.mean(np.abs(p_f0m_ho - ho["y_f0_m"]))), 3),
        "held_f0_mid_corr_r": _corr(p_f0m_ho, ho["y_f0_m"]),
        "held_pause_f1": p_f1,
        "train_pause_pos_rate": round(float(np.mean(tr["y_pause"])), 4),
    }
    model = ProsodyModelV14(
        vocab=vocab,
        word_lexicon=word_lexicon,
        dur_model=dur_m,
        f0_start_model=f0s_m,
        f0_mid_model=f0m_m,
        f0_end_model=f0e_m,
        energy_model=db_m,
        pause_model=pause_m,
        pause_threshold=pause_thresh,
        train_stats=metrics,
    )
    model_bytes = model.save(save_path) if save_path is not None else len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    metrics["model_bytes"] = int(model_bytes)
    model.train_stats["model_bytes"] = int(model_bytes)
    if save_path is not None:
        model.save(save_path)
    return model, metrics


def apply_prosody_v14(
    ir_plan: dict[str, Any],
    model: ProsodyModelV14 | None = None,
) -> dict[str, Any]:
    """Attach word- & tone-aware HGBR duration, 3-pt F0, gain_db, and pause predictions to `plan_v13`."""
    if model is None:
        model = ProsodyModelV14.load(PROSODY_MODEL_V14)

    out = json.loads(json.dumps(ir_plan))
    units = out.get("units") or []
    if not units:
        return out

    text = str(out.get("text", ""))
    winfo = _word_info_by_char(text, model.word_lexicon) if text else {}
    is_qe = ("?" in text or "？" in text or "!" in text or "！" in text)

    # Split ZH units into explicit punctuation-bounded clauses (never override punctuation pauses)
    clauses: list[list[int]] = []
    cur: list[int] = []
    for idx, u in enumerate(units):
        if u.get("lang", "zh") != "zh":
            if cur:
                clauses.append(cur)
                cur = []
            continue
        cur.append(idx)
        if float(u.get("pause_after_ms", 0.0)) > 15.0:
            clauses.append(cur)
            cur = []
    if cur:
        clauses.append(cur)

    rows_X: list[np.ndarray] = []
    row_unit_idx: list[int] = []
    row_meta: list[dict[str, Any]] = []

    for cl in clauses:
        cl_len = len(cl)
        for c_pos, ui in enumerate(cl):
            u = units[ui]
            prev_u = units[cl[c_pos - 1]] if c_pos > 0 else None
            next_u = units[cl[c_pos + 1]] if c_pos + 1 < cl_len else None
            ts = u.get("text_span")
            ch_pos = int(ts[0]) if isinstance(ts, list) and ts else -1
            hanzi = str(u.get("hanzi") or u.get("token") or "")
            wl, piw, _ = winfo.get(ch_pos, (1, 0, hanzi))

            xr = _encode_syllable_row(
                vocab=model.vocab,
                allow_grow=False,
                onset=str(u.get("onset", "")),
                final=str(u.get("final", "")),
                prev_final=str(prev_u.get("final", "")) if prev_u else "SIL",
                next_onset=str(next_u.get("onset", "")) if next_u else "SIL",
                prev_tone=int(prev_u.get("tone") or 1) if prev_u else 0,
                curr_tone=int(u.get("tone") or 1),
                next_tone=int(next_u.get("tone") or 1) if next_u else 0,
                hanzi=hanzi,
                word_len=wl,
                pos_in_word=piw,
                syl_in_clause=c_pos,
                clause_len=cl_len,
                is_question_or_excl=is_qe,
            )
            rows_X.append(xr)
            row_unit_idx.append(ui)
            row_meta.append({
                "word_len": wl,
                "pos_in_word": piw,
                "is_word_final": (piw == wl - 1),
                "is_clause_final": (c_pos == cl_len - 1),
                "clause_len": cl_len,
            })

    if not rows_X:
        return out

    X = np.stack(rows_X, axis=0)
    pred_durs = np.clip(np.exp(model.dur_model.predict(X)), 65.0, 360.0)
    pred_f0_s = np.clip(model.f0_start_model.predict(X), -7.5, 7.5)
    pred_f0_m = np.clip(model.f0_mid_model.predict(X), -7.5, 7.5)
    pred_f0_e = np.clip(model.f0_end_model.predict(X), -7.5, 7.5)
    pred_dbs = model.energy_model.predict(X)
    db_med = float(np.median(pred_dbs)) if len(pred_dbs) else 0.0
    pred_pause_p = np.clip(model.pause_model.predict(X), 0.0, 1.0)

    for k, ui in enumerate(row_unit_idx):
        u = units[ui]
        rm = row_meta[k]
        u["duration_ms"] = round(float(pred_durs[k]), 1)
        f3 = [round(float(pred_f0_s[k]), 3), round(float(pred_f0_m[k]), 3), round(float(pred_f0_e[k]), 3)]
        u["f0_3pt_st"] = f3
        u["pitch_st"] = [
            f3[0],
            round(0.5 * (f3[0] + f3[1]), 3),
            f3[1],
            round(0.5 * (f3[1] + f3[2]), 3),
            f3[2],
        ]
        gdb = float(np.clip(pred_dbs[k] - db_med, -2.5, 2.5))
        u["gain_db"] = round(gdb, 3)
        u["word_len"] = rm["word_len"]
        u["pos_in_word"] = rm["pos_in_word"]
        u["is_word_final"] = rm["is_word_final"]

        # Only insert prosodic break at non-clause-final word boundaries in longer clauses
        existing_p = float(u.get("pause_after_ms", 0.0))
        if (
            existing_p <= 15.0
            and rm["is_word_final"]
            and (not rm["is_clause_final"])
            and rm["clause_len"] >= 9
            and float(pred_pause_p[k]) >= model.pause_threshold
        ):
            u["pause_after_ms"] = round(float(np.clip(80.0 + 180.0 * (float(pred_pause_p[k]) - model.pause_threshold), 80.0, 180.0)), 1)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Train v14 Mandarin HGBR prosody predictor.")
    ap.parse_args()
    _, metrics = train_prosody_v14()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if metrics["model_bytes"] < 500_000 else 1


if __name__ == "__main__":
    raise SystemExit(main())
