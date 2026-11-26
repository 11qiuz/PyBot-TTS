"""Word-, `#1/#2/#3` Break- & Question-Aware HGBR Mandarin Prosody Predictor (`BZNSYP`, `< 500 KB`)."""
from __future__ import annotations

import argparse
from collections import Counter
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
from exp_v13_utau_ceiling.sandhi_v13 import plan_v13  # noqa: E402
from exp_v13_utau_ceiling.v13_selector import FINAL_CODA_CLASS, FINAL_HEAD_CLASS  # noqa: E402
from exp_v14_fluent_zh_tts.prosody_v14 import (  # noqa: E402
    ZH_FUNC_WORDS,
    _build_default_vocab,
    _cat_id,
    _corr,
    _normalize_of,
    _onset_manner,
    _word_info_by_char,
)
from exp_v15_bznsyp_tts import INDEX_V15, PROSODY_MODEL_V15  # noqa: E402
from exp_v15_bznsyp_tts.index_bznsyp_v15 import V15BzToken, load_bznsyp_index_v15  # noqa: E402
from pybot_tts_v2_recorded import engine  # noqa: E402

_HANZI_RUN_RE = re.compile(r"[\u3400-\u9fff]+")
_SPLIT_PROSODY_WORDS_RE = re.compile(r"#[0-9]+|[^\u3400-\u9fff]+")
_Q_PARTICLES = frozenset({"吗", "呢", "啊", "吧", "么", "嘛"})
CAT_FEATURE_INDICES_V15 = [0, 1, 2, 3, 4, 5, 6]


def build_bznsyp_word_lexicon(
    train_utt_metas: dict[str, dict[str, Any]],
    *,
    min_bz_freq: int = 2,
) -> set[str]:
    """Build multi-char prosodic word lexicon from `#1..#4` training lines (zero heldout leakage)."""
    words: set[str] = set()
    for w in getattr(engine, "WORDS", ()):
        if len(w) >= 2 and _HANZI_RUN_RE.fullmatch(w):
            words.add(w)
    for w in ZH_POLYPHONE_PHRASES_V10:
        if len(w) >= 2 and _HANZI_RUN_RE.fullmatch(w):
            words.add(w)

    freq: Counter[str] = Counter()
    for meta in train_utt_metas.values():
        ptext = str(meta.get("prosody_text", ""))
        for seg in _SPLIT_PROSODY_WORDS_RE.split(ptext):
            if 2 <= len(seg) <= 6 and _HANZI_RUN_RE.fullmatch(seg):
                freq[seg] += 1

    for w, cnt in freq.items():
        if cnt >= min_bz_freq:
            words.add(w)
    return words


def segment_zh_words_v15(text: str, word_lexicon: set[str]) -> dict[int, tuple[int, int, str]]:
    """Map character position -> `(word_len, pos_in_word, word_str)` via longest match."""
    return _word_info_by_char(text, word_lexicon)


@dataclass
class ProsodyModelV15:
    """Compact `< 500 KB` HGBR prosody + `#2/#3` prosodic break + question intonation model."""

    vocab: dict[str, dict[str, int]]
    word_lexicon: set[str]
    dur_model: HistGradientBoostingRegressor
    f0_start_model: HistGradientBoostingRegressor
    f0_mid_model: HistGradientBoostingRegressor
    f0_end_model: HistGradientBoostingRegressor
    energy_model: HistGradientBoostingRegressor
    break_model: HistGradientBoostingRegressor
    pause_dur_model: HistGradientBoostingRegressor
    break_threshold: float
    train_stats: dict[str, Any]

    def save(self, path: Path | str = PROSODY_MODEL_V15) -> int:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        p.write_bytes(raw)
        return len(raw)

    @classmethod
    def load(cls, path: Path | str = PROSODY_MODEL_V15) -> ProsodyModelV15:
        raw = Path(path).read_bytes()

        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                if module == "__main__" and name == "ProsodyModelV15":
                    return ProsodyModelV15
                return super().find_class(module, name)

        return _CompatUnpickler(io.BytesIO(raw)).load()


ProsodyModelV15.__module__ = "exp_v15_bznsyp_tts.prosody_v15"
if __name__ == "__main__":
    sys.modules.setdefault("exp_v15_bznsyp_tts.prosody_v15", sys.modules[__name__])


def _encode_row_v15(
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
    clause_idx: int,
    word_idx_in_clause: int,
    dist_from_prev_break: int,
    is_question: bool,
    is_exclamation: bool,
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
    n_om = "SIL" if next_onset == "SIL" else _onset_manner(next_onset)

    def _lu(cat: str, val: str) -> int:
        mp = vocab[cat]
        return _cat_id(mp, val) if allow_grow else mp.get(val, 0)

    is_zero = 1.0 if o == "" else 0.0
    is_w_init = 1.0 if pos_in_word == 0 else 0.0
    is_w_fin = 1.0 if pos_in_word == word_len - 1 else 0.0
    is_func = 1.0 if hanzi in ZH_FUNC_WORDS else 0.0
    pos_norm = float(syl_in_clause) / max(1, clause_len - 1) if clause_len > 1 else 0.5
    to_end = max(0, clause_len - 1 - syl_in_clause)
    is_c_fin = 1.0 if to_end == 0 else 0.0
    is_q_part = 1.0 if (is_c_fin > 0.5 and hanzi in _Q_PARTICLES) else 0.0

    return np.array(
        [
            float(_lu("onset", o)),
            float(_lu("final", f)),
            float(_lu("onset_manner", om)),
            float(_lu("final_head", fh)),
            float(_lu("final_coda", fc)),
            float(_lu("final_coda", p_coda)),
            float(_lu("onset_manner", n_om)),
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
            float(min(22, syl_in_clause)),
            float(min(22, to_end)),
            is_c_fin,
            float(min(28, clause_len)),
            1.0 if is_question else 0.0,
            1.0 if is_exclamation else 0.0,
            float(min(6, clause_idx)),
            float(min(15, word_idx_in_clause)),
            float(min(15, dist_from_prev_break)),
            is_q_part,
        ],
        dtype=np.float32,
    )


def _extract_bznsyp_samples(
    tokens_by_uid: dict[str, list[V15BzToken]],
    utt_metas: dict[str, dict[str, Any]],
    word_lexicon: set[str],
    vocab: dict[str, dict[str, int]],
    *,
    allow_grow: bool,
) -> dict[str, Any]:
    X_rows: list[np.ndarray] = []
    y_dur, y_f0s, y_f0m, y_f0e, y_db = [], [], [], [], []
    X_brk, y_brk = [], []
    X_pdur, y_pdur = [], []

    for uid in sorted(tokens_by_uid.keys()):
        toks = tokens_by_uid[uid]
        if not toks:
            continue
        meta = utt_metas[uid]
        text = meta["text"]
        winfo = segment_zh_words_v15(text, word_lexicon)
        has_q = ("？" in text or "?" in text)
        has_exc = ("！" in text or "!" in text)

        # Clause split strictly on real punctuation or >= 25ms sp1 pauses (as punctuation boundaries are known at TTS inference)
        clauses: list[list[int]] = []
        cur_c: list[int] = []
        for idx_t, tk in enumerate(toks):
            cur_c.append(idx_t)
            if tk.punct_after or tk.pause_after_ms >= 25.0:
                clauses.append(cur_c)
                cur_c = []
        if cur_c:
            clauses.append(cur_c)

        # Sentence-wide word index tracking for break prediction
        hanzi_offset = 0
        # Map token index -> character index in clean text
        char_idxs: list[int] = []
        pos_c = 0
        for ch in text:
            if _HANZI_RUN_RE.fullmatch(ch):
                char_idxs.append(pos_c)
            pos_c += 1

        last_brk_pos = 0
        word_cnt_cl = 0
        for cl_idx, cl in enumerate(clauses):
            cl_len = len(cl)
            word_cnt_cl = 0
            last_brk_pos = 0
            cl_last_punct = toks[cl[-1]].punct_after
            cl_q = has_q and (cl_idx == len(clauses) - 1 or "？" in cl_last_punct or "?" in cl_last_punct)
            cl_exc = has_exc and (cl_idx == len(clauses) - 1 or "！" in cl_last_punct or "!" in cl_last_punct)

            for c_pos, idx_t in enumerate(cl):
                tk = toks[idx_t]
                c_txt_idx = char_idxs[tk.syl_idx] if tk.syl_idx < len(char_idxs) else -1
                wl, piw, _ = winfo.get(c_txt_idx, (1, 0, tk.hanzi))
                if piw == 0 and c_pos > 0:
                    word_cnt_cl += 1

                prev_tk = toks[cl[c_pos - 1]] if c_pos > 0 else None
                next_tk = toks[cl[c_pos + 1]] if c_pos + 1 < cl_len else None

                xr = _encode_row_v15(
                    vocab=vocab,
                    allow_grow=allow_grow,
                    onset=tk.onset,
                    final=tk.final,
                    prev_final=prev_tk.final if prev_tk else "SIL",
                    next_onset=next_tk.onset if next_tk else "SIL",
                    prev_tone=prev_tk.tone if prev_tk else 0,
                    curr_tone=tk.tone,
                    next_tone=next_tk.tone if next_tk else 0,
                    hanzi=tk.hanzi,
                    word_len=wl,
                    pos_in_word=piw,
                    syl_in_clause=c_pos,
                    clause_len=cl_len,
                    clause_idx=cl_idx,
                    word_idx_in_clause=word_cnt_cl,
                    dist_from_prev_break=c_pos - last_brk_pos,
                    is_question=cl_q,
                    is_exclamation=cl_exc,
                )

                # Regression targets (dur, F0, energy)
                if 45.0 <= tk.dur_ms <= 420.0:
                    X_rows.append(xr)
                    y_dur.append(math.log(tk.dur_ms))
                    y_f0s.append(tk.f0_start_st)
                    y_f0m.append(tk.f0_mid_st)
                    y_f0e.append(tk.f0_end_st)
                    y_db.append(tk.rel_db)

                # Break prediction sample on word-final boundaries without explicit punctuation (not sentence end)
                if idx_t < len(toks) - 1 and (not tk.punct_after) and piw == wl - 1:
                    is_major_brk = 1.0 if tk.prosodic_break in (2, 3) else 0.0
                    X_brk.append(xr)
                    y_brk.append(is_major_brk)
                    if tk.prosodic_break in (2, 3) and tk.pause_after_ms >= 20.0:
                        X_pdur.append(xr)
                        y_pdur.append(float(np.clip(tk.pause_after_ms, 30.0, 260.0)))

                if tk.prosodic_break >= 2:
                    last_brk_pos = c_pos

    return {
        "X": np.stack(X_rows, axis=0),
        "y_dur": np.asarray(y_dur, dtype=np.float64),
        "y_f0s": np.asarray(y_f0s, dtype=np.float64),
        "y_f0m": np.asarray(y_f0m, dtype=np.float64),
        "y_f0e": np.asarray(y_f0e, dtype=np.float64),
        "y_db": np.asarray(y_db, dtype=np.float64),
        "X_brk": np.stack(X_brk, axis=0) if X_brk else np.zeros((0, 27), dtype=np.float32),
        "y_brk": np.asarray(y_brk, dtype=np.float64),
        "X_pdur": np.stack(X_pdur, axis=0) if X_pdur else np.zeros((0, 27), dtype=np.float32),
        "y_pdur": np.asarray(y_pdur, dtype=np.float64),
    }


def _group_tokens_by_uid(tokens: list[V15BzToken]) -> dict[str, list[V15BzToken]]:
    by_u: dict[str, list[V15BzToken]] = {}
    for tk in tokens:
        by_u.setdefault(tk.uid, []).append(tk)
    return by_u


def train_prosody_v15(
    *,
    index_path: Path = INDEX_V15,
    save_path: Path = PROSODY_MODEL_V15,
) -> tuple[ProsodyModelV15, dict[str, Any]]:
    """Train zero-leakage `#2/#3` break + question-aware Mandarin HGBR prosody model on BZNSYP."""
    idx_doc = load_bznsyp_index_v15(index_path)
    train_metas = idx_doc["train_utt_metas"]
    held_metas = idx_doc["held_utt_metas"]
    train_by_u = _group_tokens_by_uid(idx_doc["tokens"])
    held_by_u = idx_doc["held_tokens_by_uid"]

    lexicon = build_bznsyp_word_lexicon(train_metas, min_bz_freq=2)
    vocab = _build_default_vocab()

    tr = _extract_bznsyp_samples(train_by_u, train_metas, lexicon, vocab, allow_grow=True)
    ho = _extract_bznsyp_samples(held_by_u, held_metas, lexicon, vocab, allow_grow=False)

    # Compact HGBR settings keep 7 regressors + lexicon comfortably under 500 KB
    kw = dict(
        max_iter=30,
        max_depth=4,
        max_leaf_nodes=11,
        min_samples_leaf=30,
        l2_regularization=2.5,
        learning_rate=0.10,
        categorical_features=CAT_FEATURE_INDICES_V15,
        random_state=42,
    )
    kw_b = dict(
        max_iter=28,
        max_depth=4,
        max_leaf_nodes=11,
        min_samples_leaf=35,
        l2_regularization=3.0,
        learning_rate=0.09,
        categorical_features=CAT_FEATURE_INDICES_V15,
        random_state=42,
    )

    dur_m = HistGradientBoostingRegressor(**kw).fit(tr["X"], tr["y_dur"])
    f0s_m = HistGradientBoostingRegressor(**kw).fit(tr["X"], tr["y_f0s"])
    f0m_m = HistGradientBoostingRegressor(**kw).fit(tr["X"], tr["y_f0m"])
    f0e_m = HistGradientBoostingRegressor(**kw).fit(tr["X"], tr["y_f0e"])
    db_m = HistGradientBoostingRegressor(**kw).fit(tr["X"], tr["y_db"])
    brk_m = HistGradientBoostingRegressor(**kw_b).fit(tr["X_brk"], tr["y_brk"])
    pdur_m = HistGradientBoostingRegressor(**kw_b).fit(tr["X_pdur"], tr["y_pdur"])

    # Calibrate #2/#3 break threshold strictly on a 10% validation tail of training boundaries (zero heldout leakage!)
    n_brk = len(tr["X_brk"])
    val_off = int(0.90 * n_brk)
    p_val = np.clip(brk_m.predict(tr["X_brk"][val_off:]), 0.0, 1.0)
    y_val = tr["y_brk"][val_off:]
    best_thr, best_vf1 = 0.35, -1.0
    for thr in np.linspace(0.20, 0.55, 15):
        tp = float(np.sum((p_val >= thr) & (y_val >= 0.5)))
        fp = float(np.sum((p_val >= thr) & (y_val < 0.5)))
        fn = float(np.sum((p_val < thr) & (y_val >= 0.5)))
        f1 = (2.0 * tp) / max(1e-9, 2.0 * tp + fp + fn)
        if f1 > best_vf1:
            best_vf1, best_thr = f1, float(thr)

    # Evaluate strictly on heldout 20 utterances
    p_dur_ho = np.clip(np.exp(dur_m.predict(ho["X"])), 60.0, 380.0)
    y_dur_ho = np.exp(ho["y_dur"])
    p_f0m_ho = f0m_m.predict(ho["X"])
    p_brk_ho = np.clip(brk_m.predict(ho["X_brk"]), 0.0, 1.0)
    tp_h = float(np.sum((p_brk_ho >= best_thr) & (ho["y_brk"] >= 0.5)))
    fp_h = float(np.sum((p_brk_ho >= best_thr) & (ho["y_brk"] < 0.5)))
    fn_h = float(np.sum((p_brk_ho < best_thr) & (ho["y_brk"] >= 0.5)))
    held_brk_f1 = round((2.0 * tp_h) / max(1e-9, 2.0 * tp_h + fp_h + fn_h), 4)

    metrics = {
        "n_train_utts": len(train_metas),
        "n_train_syls": int(len(tr["X"])),
        "n_held_utts": len(held_metas),
        "n_held_syls": int(len(ho["X"])),
        "lexicon_size": len(lexicon),
        "held_dur_mae_ms": round(float(np.mean(np.abs(p_dur_ho - y_dur_ho))), 2),
        "held_dur_corr_r": _corr(p_dur_ho, y_dur_ho),
        "held_f0_mid_mae_st": round(float(np.mean(np.abs(p_f0m_ho - ho["y_f0m"]))), 3),
        "held_f0_mid_corr_r": _corr(p_f0m_ho, ho["y_f0m"]),
        "break_threshold": round(best_thr, 4),
        "train_break_pos_rate": round(float(np.mean(tr["y_brk"])), 4),
        "held_break_f1": held_brk_f1,
    }
    model = ProsodyModelV15(
        vocab=vocab,
        word_lexicon=lexicon,
        dur_model=dur_m,
        f0_start_model=f0s_m,
        f0_mid_model=f0m_m,
        f0_end_model=f0e_m,
        energy_model=db_m,
        break_model=brk_m,
        pause_dur_model=pdur_m,
        break_threshold=round(best_thr, 4),
        train_stats=metrics,
    )
    model_bytes = model.save(save_path) if save_path is not None else len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
    metrics["model_bytes"] = int(model_bytes)
    model.train_stats["model_bytes"] = int(model_bytes)
    if save_path is not None:
        model.save(save_path)
    return model, metrics


def apply_prosody_v15(
    ir_plan: dict[str, Any],
    model: ProsodyModelV15 | None = None,
) -> dict[str, Any]:
    """Attach `#2/#3` prosodic-break & question-aware HGBR duration, 3-pt F0, gain & breaks to `plan_v13`."""
    if model is None:
        model = ProsodyModelV15.load(PROSODY_MODEL_V15)

    out = json.loads(json.dumps(ir_plan))
    units = out.get("units") or []
    if not units:
        return out

    text = str(out.get("text", ""))
    winfo = segment_zh_words_v15(text, model.word_lexicon) if text else {}
    has_q = ("？" in text or "?" in text)
    has_exc = ("！" in text or "!" in text)

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
    row_ui: list[int] = []
    row_meta: list[dict[str, Any]] = []

    for cl_idx, cl in enumerate(clauses):
        cl_len = len(cl)
        word_cnt_cl = 0
        cl_q = has_q and (cl_idx == len(clauses) - 1)
        cl_exc = has_exc and (cl_idx == len(clauses) - 1)
        for c_pos, ui in enumerate(cl):
            u = units[ui]
            prev_u = units[cl[c_pos - 1]] if c_pos > 0 else None
            next_u = units[cl[c_pos + 1]] if c_pos + 1 < cl_len else None
            ts = u.get("text_span")
            ch_pos = int(ts[0]) if isinstance(ts, list) and ts else -1
            hanzi = str(u.get("hanzi") or u.get("token") or "")
            wl, piw, _ = winfo.get(ch_pos, (1, 0, hanzi))
            if piw == 0 and c_pos > 0:
                word_cnt_cl += 1

            xr = _encode_row_v15(
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
                clause_idx=cl_idx,
                word_idx_in_clause=word_cnt_cl,
                dist_from_prev_break=c_pos,
                is_question=cl_q,
                is_exclamation=cl_exc,
            )
            rows_X.append(xr)
            row_ui.append(ui)
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
    pred_brk = np.clip(model.break_model.predict(X), 0.0, 1.0)
    pred_pdur = np.clip(model.pause_dur_model.predict(X), 30.0, 200.0)

    for k, ui in enumerate(row_ui):
        u = units[ui]
        rm = row_meta[k]
        u["duration_ms"] = round(float(pred_durs[k]), 1)
        f3 = [round(float(pred_f0_s[k]), 3), round(float(pred_f0_m[k]), 3), round(float(pred_f0_e[k]), 3)]
        u["f0_3pt_st"] = f3
        u["pitch_st"] = [f3[0], round(0.5 * (f3[0] + f3[1]), 3), f3[1], round(0.5 * (f3[1] + f3[2]), 3), f3[2]]
        u["gain_db"] = round(float(np.clip(pred_dbs[k] - db_med, -2.5, 2.5)), 3)
        u["word_len"] = rm["word_len"]
        u["pos_in_word"] = rm["pos_in_word"]
        u["is_word_final"] = rm["is_word_final"]
        u["pred_break_score"] = round(float(pred_brk[k]), 3)
        u["prosodic_break"] = 2 if (rm["is_word_final"] and float(pred_brk[k]) >= model.break_threshold) else (1 if rm["is_word_final"] else 0)

        existing_p = float(u.get("pause_after_ms", 0.0))
        if (
            existing_p <= 15.0
            and rm["is_word_final"]
            and (not rm["is_clause_final"])
            and rm["clause_len"] >= 9
            and float(pred_brk[k]) >= model.break_threshold
        ):
            u["pause_after_ms"] = round(float(pred_pdur[k]), 1)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Train BZNSYP v15 prosody & break predictor.")
    ap.parse_args()
    _, metrics = train_prosody_v15()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if metrics["model_bytes"] < 500_000 else 1


if __name__ == "__main__":
    raise SystemExit(main())
