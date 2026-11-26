"""Unified Zero-ML Native Multilingual IPA Medoid + Hierarchical Ridge Acoustic Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

import numpy as np

from exp_20_pybot_larynx import EXP_DIR, HOP, N_BAP, N_BINS, N_CEP
from exp_20_pybot_larynx.acoustics.knowledge import _interp_phase_bins
from exp_20_pybot_larynx.acoustics.knowledge_stage2 import MedoidGroup, _load_medoid_table
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import _GV_BELL_W, _GV_Q_END, _GV_Q_START, _interp_phase_bins_cubic, _load_gv_table
from exp_20_pybot_larynx.acoustics.knowledge_stage7 import _load_proto_dict, _load_weight_dict
from exp_20_pybot_larynx.crossling.knowledge_en_stage10 import _score_en_medoid_ridge_guided
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env
from exp_20_pybot_larynx.multilingual.schema import (
    PhoneEvent,
    UNIVERSAL_VOICELESS_OBSTRUENTS,
    classify_universal_art_class,
    lookup_ipa_info,
    normalize_prosody_tag,
    strip_allophone_suffix,
)

MULTILINGUAL_DIR = Path(__file__).resolve().parent
SF_NATIVE_KB_PATH = MULTILINGUAL_DIR / "knowledge_sf_native_stage13.npz"
SF_NATIVE_SUMMARY_PATH = MULTILINGUAL_DIR / "knowledge_sf_native_stage13_summary.json"

N_BINS_M = 5
N_COARSE_CEP_M = 24
N_ART_CLS_M = 10   # 0..9 in ART_CLASS_NAMES
N_PROS_SLOTS = 6   # 0..5 normalized prosody_tag (ZH 1..5, EN/JA/KO 0..2)
D_ACOUSTIC_FEAT_M = 64
_BIN_CENTERS_M = (np.arange(N_BINS_M, dtype=np.float64) + 0.5) / float(N_BINS_M)
_ZERO_8 = np.zeros(8, dtype=np.float64)

DEFAULT_MULTILINGUAL_TAPER_24 = np.array(
    [0.84] * 6 + [0.44] * 2 + [0.22] * 4 + [0.10] * 4 + [0.04] * 8,
    dtype=np.float64,
)

_KB_LOCK = Lock()
_CACHED_KB_PATH: str | None = None
_CACHED_KB: "MultilingualIPAKnowledge | None" = None


def extract_multilingual_acoustic_features(
    *,
    prev_cls: int,
    next_cls: int,
    prosody_tag: int,
    pos_id: int,
    word_pos: int,
    is_func_word: bool,
    is_coda: bool,
    same_word_prev: bool,
    same_word_next: bool,
    dur_ms: float,
    f3_st: tuple[float, float, float] | np.ndarray,
    pause_prev_ms: float,
    pause_after_ms: float,
    prev_phone_key: str,
    next_phone_key: str,
    p_end_proto: dict[str, np.ndarray],
    p_start_proto: dict[str, np.ndarray],
) -> np.ndarray:
    """Build deterministic 64-dim universal phonetic context vector (`allow_pickle=False`)."""
    x = np.zeros(D_ACOUSTIC_FEAT_M, dtype=np.float64)
    c = 0
    x[c] = 1.0
    c += 1

    x[c + max(0, min(N_ART_CLS_M - 1, int(prev_cls)))] = 1.0
    c += N_ART_CLS_M  # -> 11

    x[c + max(0, min(N_ART_CLS_M - 1, int(next_cls)))] = 1.0
    c += N_ART_CLS_M  # -> 21

    x[c + max(0, min(N_PROS_SLOTS - 1, int(prosody_tag)))] = 1.0
    c += N_PROS_SLOTS  # -> 27

    x[c + max(0, min(2, int(pos_id)))] = 1.0
    c += 3  # -> 30

    x[c + max(0, min(3, int(word_pos)))] = 1.0
    c += 4  # -> 34

    x[c] = 1.0 if is_func_word else 0.0
    x[c + 1] = 1.0 if is_coda else 0.0
    x[c + 2] = 1.0 if same_word_prev else 0.0
    x[c + 3] = 1.0 if same_word_next else 0.0
    c += 4  # -> 38 (note: keep total exact!)

    ld = math.log(max(15.0, float(dur_ms)) / 75.0)
    p0 = float(f3_st[0]) / 5.0
    p1 = float(f3_st[1]) / 5.0
    p2 = float(f3_st[2]) / 5.0
    x[c : c + 10] = (
        ld,
        ld * ld,
        p0,
        p1,
        p2,
        p2 - p0,
        p0 + p2 - 2.0 * p1,
        math.log1p(max(0.0, float(pause_prev_ms)) / 50.0),
        math.log1p(max(0.0, float(pause_after_ms)) / 50.0),
        float(float(pause_prev_ms) >= 25.0),
    )
    c += 10  # -> 48

    def _get_proto(tbl: dict[str, np.ndarray], pk: str) -> np.ndarray:
        if not pk or pk == "sil":
            return _ZERO_8
        if pk in tbl:
            return tbl[pk] / 15.0
        base = strip_allophone_suffix(pk)
        return tbl.get(base, _ZERO_8) / 15.0

    x[c : c + 8] = _get_proto(p_end_proto, prev_phone_key)
    c += 8  # -> 56
    x[c : c + 8] = _get_proto(p_start_proto, next_phone_key)
    return x


@dataclass
class MultilingualIPAKnowledge:
    """Unified single-speaker multilingual IPA knowledge (`zh`, `en`, extensible to `ja`/`ko`)."""

    ctx_pns: dict[str, MedoidGroup]
    ctx_pn: dict[str, MedoidGroup]
    ctx_ps: dict[str, MedoidGroup]
    ctx_p: dict[str, MedoidGroup]
    ctx_base: dict[str, MedoidGroup]
    ipa_shared: dict[str, MedoidGroup]
    global_proto: tuple[np.ndarray, np.ndarray, np.ndarray]

    w_shared: dict[str, np.ndarray]
    w_base: dict[str, np.ndarray]
    w_phone: dict[str, np.ndarray]
    w_ps: dict[str, np.ndarray]
    w_pns: dict[str, np.ndarray]
    d_pns: dict[str, np.ndarray]
    w_bap_ps: dict[str, np.ndarray]

    p_start_proto: dict[str, np.ndarray]
    p_end_proto: dict[str, np.ndarray]

    sil_cep64: np.ndarray
    sil_bap6: np.ndarray
    speaker_f0_med_hz: float
    ref_log_amp: float
    n_train_utts: int
    c_dim: int = N_COARSE_CEP_M

    gv_ps: dict[str, np.ndarray] = field(default_factory=dict)
    gv_glob: np.ndarray = field(default_factory=lambda: np.ones(40, dtype=np.float64))

    @classmethod
    def load(cls, path: Path | str = SF_NATIVE_KB_PATH) -> "MultilingualIPAKnowledge":
        with np.load(Path(path), allow_pickle=False) as d:
            c_dim = int(d["c_dim"][0]) if "c_dim" in d else N_COARSE_CEP_M
            return cls(
                ctx_pns=_load_medoid_table(d, "ctx_pns"),
                ctx_pn=_load_medoid_table(d, "ctx_pn"),
                ctx_ps=_load_medoid_table(d, "ctx_ps"),
                ctx_p=_load_medoid_table(d, "ctx_p"),
                ctx_base=_load_medoid_table(d, "ctx_base"),
                ipa_shared=_load_medoid_table(d, "ipa_shared"),
                global_proto=(
                    d["glob_cep"].astype(np.float64),
                    d["glob_bap"].astype(np.float64),
                    d["glob_vp"].astype(np.float64),
                ),
                w_shared=_load_weight_dict(d, "w_shared"),
                w_base=_load_weight_dict(d, "w_base"),
                w_phone=_load_weight_dict(d, "w_phone"),
                w_ps=_load_weight_dict(d, "w_ps"),
                w_pns=_load_weight_dict(d, "w_pns"),
                d_pns=_load_weight_dict(d, "d_pns"),
                w_bap_ps=_load_weight_dict(d, "w_bap_ps"),
                p_start_proto=_load_proto_dict(d, "p_start_proto"),
                p_end_proto=_load_proto_dict(d, "p_end_proto"),
                sil_cep64=d["sil_cep64"].astype(np.float64),
                sil_bap6=d["sil_bap6"].astype(np.float64),
                speaker_f0_med_hz=float(d["speaker_f0_med_hz"][0]),
                ref_log_amp=float(d["ref_log_amp"][0]),
                n_train_utts=int(d["n_train_utts"][0]),
                c_dim=c_dim,
                gv_ps=_load_gv_table(d, "gv_ps"),
                gv_glob=d["gv_glob"].astype(np.float64) if "gv_glob" in d else np.ones(40),
            )

    def predict_ridge(
        self,
        prev_cls: int,
        phone_key: str,
        next_cls: int,
        prosody_tag: int,
        x_feat: np.ndarray,
    ) -> np.ndarray | None:
        base_ipa = strip_allophone_suffix(phone_key)
        lang_prefix = phone_key.split(":", 1)[0] if ":" in phone_key else "en"
        lang_base = f"{lang_prefix}:{base_ipa}"
        k_pns = f"{prev_cls}|{phone_key}|{next_cls}|{prosody_tag}"
        w_exact = self.w_pns.get(k_pns)
        if w_exact is not None:
            return (x_feat @ w_exact).reshape(N_BINS_M, self.c_dim)

        w_parent = self.w_ps.get(
            f"{phone_key}|{prosody_tag}",
            self.w_phone.get(phone_key, self.w_base.get(lang_base, self.w_shared.get(base_ipa))),
        )
        if w_parent is None:
            return None
        pred = (x_feat @ w_parent).reshape(N_BINS_M, self.c_dim)
        d_rare = self.d_pns.get(k_pns)
        if d_rare is not None:
            pred = pred + d_rare
        return pred

    def predict_bap_ridge(self, phone_key: str, prosody_tag: int, x_feat: np.ndarray) -> np.ndarray | None:
        base_ipa = strip_allophone_suffix(phone_key)
        lang_prefix = phone_key.split(":", 1)[0] if ":" in phone_key else "en"
        lang_base = f"{lang_prefix}:{base_ipa}"
        w = self.w_bap_ps.get(
            f"{phone_key}|{prosody_tag}",
            self.w_bap_ps.get(f"{lang_base}|{prosody_tag}", self.w_bap_ps.get(f"{base_ipa}|0")),
        )
        if w is None:
            return None
        return np.clip((x_feat @ w).reshape(N_BINS_M, N_BAP), 0.004, 0.995)

    def lookup_gv(self, phone_key: str, prosody_tag: int) -> np.ndarray:
        base_ipa = strip_allophone_suffix(phone_key)
        lang_prefix = phone_key.split(":", 1)[0] if ":" in phone_key else "en"
        lang_base = f"{lang_prefix}:{base_ipa}"
        return self.gv_ps.get(
            f"{phone_key}|{prosody_tag}",
            self.gv_ps.get(f"{lang_base}|{prosody_tag}", self.gv_glob),
        )

    def lookup_candidates(
        self,
        prev_cls: int,
        phone_key: str,
        next_cls: int,
        prosody_tag: int,
    ) -> tuple[MedoidGroup | None, str]:
        g = self.ctx_pns.get(f"{prev_cls}|{phone_key}|{next_cls}|{prosody_tag}")
        if g is not None and g.total_tokens >= 3:
            return g, "ctx_pns"
        g = self.ctx_pn.get(f"{phone_key}|{next_cls}|{prosody_tag}")
        if g is not None and g.total_tokens >= 3:
            return g, "ctx_pn"
        g = self.ctx_ps.get(f"{phone_key}|{prosody_tag}")
        if g is not None and g.total_tokens >= 2:
            return g, "ctx_ps"
        g = self.ctx_p.get(phone_key)
        if g is not None and g.total_tokens >= 1:
            return g, "ctx_p"

        base_ipa = strip_allophone_suffix(phone_key)
        lang_prefix = phone_key.split(":", 1)[0] if ":" in phone_key else "en"
        g = self.ctx_base.get(f"{lang_prefix}:{base_ipa}")
        if g is not None and g.total_tokens >= 1:
            return g, "ctx_base"

        # 6th-tier cross-lingual shared native IPA fallback (direct support for unseen languages e.g. ja/ko)
        g = self.ipa_shared.get(base_ipa)
        if g is not None and g.total_tokens >= 1:
            return g, "ipa_shared"
        return None, "fallback"


def get_sf_native_multilingual_kb(path: Path | str = SF_NATIVE_KB_PATH) -> MultilingualIPAKnowledge | None:
    """Thread-safe cached loader for unified SF native multilingual knowledge."""
    global _CACHED_KB_PATH, _CACHED_KB
    p = Path(path)
    p_str = str(p.resolve()) if p.exists() else str(p)
    with _KB_LOCK:
        if _CACHED_KB is not None and _CACHED_KB_PATH == p_str:
            return _CACHED_KB
    if not p.is_file():
        return None
    kb = MultilingualIPAKnowledge.load(p)
    with _KB_LOCK:
        _CACHED_KB = kb
        _CACHED_KB_PATH = p_str
    return kb


def clear_sf_native_multilingual_cache() -> None:
    """Reset cached multilingual KB."""
    global _CACHED_KB_PATH, _CACHED_KB
    with _KB_LOCK:
        _CACHED_KB_PATH = None
        _CACHED_KB = None


def build_multilingual_native_control(
    events: Sequence[PhoneEvent],
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    knowledge: MultilingualIPAKnowledge,
    *,
    taper_vec: np.ndarray | None = None,
    bap_ridge_w: float = 0.25,
    gv_strength: float = 0.22,
    gain_scale: float = 0.16,
) -> tuple[LarynxControl, dict[str, Any]]:
    """Build native single-speaker multilingual `LarynxControl` in one unified Viterbi DP pass."""
    n_fr = len(f0_hz)
    f0_arr = np.asarray(f0_hz, dtype=np.float64).copy()
    voi_arr = np.asarray(voiced, dtype=bool).copy()

    cep = np.repeat(knowledge.sil_cep64[None, :], n_fr, axis=0)
    bap = np.repeat(knowledge.sil_bap6[None, :], n_fr, axis=0)
    tilt = np.zeros(n_fr, dtype=np.float64)
    jit = np.full(n_fr, 0.0074, dtype=np.float64)
    shim = np.full(n_fr, 0.0460, dtype=np.float64)
    rd = np.ones(n_fr, dtype=np.float64)

    hit_counts: dict[str, int] = {
        "ctx_pns": 0,
        "ctx_pn": 0,
        "ctx_ps": 0,
        "ctx_p": 0,
        "ctx_base": 0,
        "ipa_shared": 0,
        "fallback": 0,
        "ridge_pns": 0,
        "ridge_parent": 0,
    }

    nonpause_indices = [i for i, ev in enumerate(events) if not ev.is_pause]
    if not nonpause_indices or n_fr == 0:
        log_env = np.clip(cep_to_log_env(cep, n_bins=N_BINS) + knowledge.ref_log_amp, -16.0, 6.0)
        return LarynxControl(
            f0_hz=f0_arr,
            voiced=voi_arr,
            log_env=log_env,
            bap=np.clip(bap, 0.004, 0.995),
            glottal_tilt_db_per_oct=tilt,
            jitter_frac=jit,
            shimmer_frac=shim,
            n_samples=n_fr * HOP,
            rd=rd,
        ), {"mode": "sf_native_multilingual", "hits": hit_counts}

    ref_f0 = max(50.0, float(knowledge.speaker_f0_med_hz))
    n_ev = len(events)

    cands: dict[int, MedoidGroup | None] = {}
    pred_ceps: dict[int, np.ndarray | None] = {}
    pred_baps: dict[int, np.ndarray | None] = {}
    ev_meta: dict[int, tuple[int, int, int, int, int, int, float, float, str, str, float]] = {}

    for idx in nonpause_indices:
        ev = events[idx]
        a = max(0, min(n_fr, int(ev.start_fr)))
        b = max(a, min(n_fr, int(ev.end_fr)))
        base_ipa = strip_allophone_suffix(ev.ipa)
        pk = ev.phone_key or f"{ev.lang}:{ev.allophone or base_ipa}"
        ptag = normalize_prosody_tag(ev.lang, ev.prosody_tag, base_ipa)

        prev_ev = events[idx - 1] if idx > 0 else None
        next_ev = events[idx + 1] if idx + 1 < n_ev else None

        pause_prev = float(ev.pause_prev_ms)
        if prev_ev is not None and not prev_ev.is_pause and (a - int(prev_ev.end_fr)) >= 2:
            pause_prev = max(pause_prev, float(a - int(prev_ev.end_fr)) * 5.0)
        elif prev_ev is None or prev_ev.is_pause:
            pause_prev = max(pause_prev, 999.0)

        pause_after = float(ev.pause_after_ms)
        if next_ev is None or next_ev.is_pause:
            pause_after = max(pause_after, 999.0 if next_ev is None else float(next_ev.end_fr - next_ev.start_fr) * 5.0)

        prev_pk = "sil" if (prev_ev is None or prev_ev.is_pause or pause_prev >= 25.0) else prev_ev.phone_key
        next_pk = "sil" if (next_ev is None or next_ev.is_pause or pause_after >= 25.0) else next_ev.phone_key
        prev_cls = classify_universal_art_class(prev_pk)
        next_cls = classify_universal_art_class(next_pk)
        pos_id = 0 if prev_cls == 0 else (2 if next_cls == 0 else 1)

        dur_ms = float(max(1, b - a)) * 5.0
        v_sub = voi_arr[a:b] & (f0_arr[a:b] > 40.0)
        f0_st = float(12.0 * np.log2(np.median(f0_arr[a:b][v_sub]) / ref_f0)) if np.any(v_sub) else 0.0
        vf = np.where(v_sub)[0] if b > a else np.zeros(0, dtype=np.intp)
        if len(vf) >= 2:
            uv = vf.astype(np.float64) / float(max(1, (b - a) - 1))
            st_v = 12.0 * np.log2(f0_arr[a:b][vf] / ref_f0)
            p3 = np.interp([0.20, 0.50, 0.80], uv, st_v)
            f3_st = (float(p3[0]), float(p3[1]), float(p3[2]))
        elif len(vf) == 1:
            val = float(12.0 * np.log2(f0_arr[a:b][vf[0]] / ref_f0))
            f3_st = (val, val, val)
        else:
            f3_st = (f0_st, f0_st, f0_st)

        x_feat = extract_multilingual_acoustic_features(
            prev_cls=prev_cls,
            next_cls=next_cls,
            prosody_tag=ptag,
            pos_id=pos_id,
            word_pos=ev.word_pos,
            is_func_word=ev.is_func_word,
            is_coda=ev.is_coda,
            same_word_prev=ev.same_word_prev,
            same_word_next=ev.same_word_next,
            dur_ms=dur_ms,
            f3_st=f3_st,
            pause_prev_ms=pause_prev,
            pause_after_ms=pause_after,
            prev_phone_key=prev_pk,
            next_phone_key=next_pk,
            p_end_proto=knowledge.p_end_proto,
            p_start_proto=knowledge.p_start_proto,
        )
        pr = knowledge.predict_ridge(prev_cls, pk, next_cls, ptag, x_feat)
        pred_ceps[idx] = pr
        if f"{prev_cls}|{pk}|{next_cls}|{ptag}" in knowledge.w_pns:
            hit_counts["ridge_pns"] += 1
        elif pr is not None:
            hit_counts["ridge_parent"] += 1
        pred_baps[idx] = knowledge.predict_bap_ridge(pk, ptag, x_feat) if bap_ridge_w > 0.0 else None

        grp, tag = knowledge.lookup_candidates(prev_cls, pk, next_cls, ptag)
        hit_counts[tag] += 1
        cands[idx] = grp
        ev_meta[idx] = (a, b, prev_cls, next_cls, ptag, pos_id, dur_ms, f0_st, base_ipa, pk, pause_prev)

    # Single unified Viterbi DP across the whole multilingual utterance (spans ZH, EN, and switches)
    unary_list: list[np.ndarray] = []
    cep_list: list[np.ndarray] = []
    bap_list: list[np.ndarray] = []
    vp_list: list[np.ndarray] = []
    for idx in nonpause_indices:
        _a, _b, prev_c, next_c, _pt, pos_id, dur_ms, f0_st, _bi, _pk, _pp = ev_meta[idx]
        grp = cands[idx]
        if grp is None:
            unary_list.append(np.zeros(1, dtype=np.float64))
            cep_list.append(knowledge.global_proto[0][None, :, :])
            bap_list.append(knowledge.global_proto[1][None, :, :])
            vp_list.append(knowledge.global_proto[2][None, :])
        else:
            unary_list.append(
                _score_en_medoid_ridge_guided(grp, pred_ceps[idx], prev_c, next_c, pos_id, dur_ms, f0_st)
            )
            cep_list.append(grp.cep)
            bap_list.append(grp.bap)
            vp_list.append(grp.vp)

    n_np = len(nonpause_indices)
    dp: list[np.ndarray] = [unary_list[0].copy()]
    back: list[np.ndarray] = []
    for m in range(1, n_np):
        idx_cur = nonpause_indices[m]
        idx_prv = nonpause_indices[m - 1]
        k_prev = len(unary_list[m - 1])
        k_cur = len(unary_list[m])
        cur_a, _cb, prev_c, _nc, _pt, _pid, _dur, _f0, cur_base, _pk, pause_prev = ev_meta[idx_cur]
        _pa, prev_b, _pc2, _nc2, _pt2, _pid2, _dur2, _f02, prev_base, _pk2, _pp2 = ev_meta[idx_prv]

        contiguous = (prev_c != 0) and (pause_prev < 20.0) and (abs(cur_a - prev_b) <= 1)
        prev_info = lookup_ipa_info(prev_base)
        cur_info = lookup_ipa_info(cur_base)
        if not contiguous or cur_info.manner in ("stop", "affricate") or prev_info.voiceless_obstruent:
            trans = np.zeros((k_prev, k_cur), dtype=np.float64)
        else:
            p_end = cep_list[m - 1][:, -1, 1:24]
            p_e0 = cep_list[m - 1][:, -1, :1]
            c_st = cep_list[m][:, 0, 1:24]
            c_e0 = cep_list[m][:, 0, :1]
            d_mat = np.sqrt(np.mean((p_end[:, None, :] - c_st[None, :, :]) ** 2, axis=2))
            d_e = 0.20 * np.abs(p_e0 - c_e0.T)
            trans = 0.42 * (d_mat + d_e)

        total = dp[m - 1][:, None] + trans + unary_list[m][None, :]
        bp = np.argmin(total, axis=0)
        dp.append(total[bp, np.arange(k_cur)])
        back.append(bp)

    best_k = [0] * n_np
    best_k[-1] = int(np.argmin(dp[-1]))
    for m in range(n_np - 2, -1, -1):
        best_k[m] = int(back[m][best_k[m + 1]])

    t_vec = DEFAULT_MULTILINGUAL_TAPER_24 if taper_vec is None else np.asarray(taper_vec, dtype=np.float64)
    c_dim = min(int(knowledge.c_dim), len(t_vec))
    gv_s = float(np.clip(gv_strength, 0.0, 1.0))
    br_w = float(np.clip(bap_ridge_w, 0.0, 1.0))
    covered = np.zeros(n_fr, dtype=bool)

    for m_pos, idx_ev in enumerate(nonpause_indices):
        ev = events[idx_ev]
        k_sel = best_k[m_pos]
        a, b, _pc, _nc, ptag, _pid, _dur, _f0, base_ipa, pk, _pp = ev_meta[idx_ev]
        if b <= a:
            continue

        m_cep = cep_list[m_pos][k_sel].copy()
        m_bap = bap_list[m_pos][k_sel].copy()
        m_vp = vp_list[m_pos][k_sel].copy()

        info = lookup_ipa_info(base_ipa)
        orig_m_cep = m_cep.copy()
        pr = pred_ceps[idx_ev]
        if pr is not None:
            av = t_vec[:c_dim][None, :]
            m_cep[:, :c_dim] = av * pr[:, :c_dim] + (1.0 - av) * m_cep[:, :c_dim]
            if info.manner in ("vowel", "diphthong", "glide", "liquid", "nasal", "tap"):
                orig_q = np.sqrt(np.sum(orig_m_cep[:, 8:24] ** 2, axis=1, keepdims=True) + 1e-12)
                cur_q = np.sqrt(np.sum(m_cep[:, 8:24] ** 2, axis=1, keepdims=True) + 1e-12)
                m_cep[:, 8:24] *= np.clip((1.04 * orig_q) / cur_q, 0.98, 1.18)

        pb = pred_baps[idx_ev]
        if pb is not None and br_w > 0.0:
            m_bap = br_w * pb + (1.0 - br_w) * m_bap

        nf = b - a
        u_ph = np.linspace(0.0, 1.0, nf, dtype=np.float64) if nf > 1 else np.array([0.5], dtype=np.float64)
        covered[a:b] = True

        cep[a:b, :24] = _interp_phase_bins_cubic(m_cep[:, :24], u_ph, _BIN_CENTERS_M)
        cep[a:b, 24:] = _interp_phase_bins(m_cep[:, 24:], u_ph, _BIN_CENTERS_M)
        bap[a:b] = _interp_phase_bins(m_bap, u_ph, _BIN_CENTERS_M)
        vp_fr = _interp_phase_bins(m_vp[:, None], u_ph, _BIN_CENTERS_M)[:, 0]

        if base_ipa in UNIVERSAL_VOICELESS_OBSTRUENTS or info.voiceless_obstruent:
            voi_arr[a:b] = False
            f0_arr[a:b] = 0.0
        elif info.manner in ("stop", "affricate", "fricative") and info.voiced:
            is_stop_or_affr = info.manner in ("stop", "affricate")
            thresh = 0.20 if is_stop_or_affr else 0.18
            v_gate = vp_fr >= thresh
            max_vp = float(np.max(vp_fr)) if nf > 0 else 0.0
            if float(np.mean(voi_arr[a:b])) >= 0.50 and float(np.mean(v_gate)) < 0.50:
                if is_stop_or_affr and (max_vp >= 0.10 or not ev.is_coda):
                    core_n = max(1, min(nf, int(round(0.65 * nf))))
                    core_mask = np.zeros(nf, dtype=bool)
                    core_mask[:core_n] = True
                    v_gate = v_gate | core_mask | ((vp_fr >= max(0.10, 0.55 * max_vp)) if max_vp >= 0.10 else False)
                elif (not is_stop_or_affr) and max_vp >= 0.08:
                    v_gate = v_gate | (vp_fr >= max(0.08, 0.45 * max_vp))
            voi_arr[a:b] = voi_arr[a:b] & v_gate
            f0_arr[a:b][~voi_arr[a:b]] = 0.0

        # Sanitize alignment-bleed BAP spikes on voiced frames before rule shaping
        if info.voiced:
            vm_ev = voi_arr[a:b]
            if np.any(vm_ev):
                if info.manner in ("stop", "affricate", "fricative"):
                    bap[a:b, 0][vm_ev] = np.minimum(bap[a:b, 0][vm_ev], 0.34)
                    bap[a:b, 1][vm_ev] = np.minimum(bap[a:b, 1][vm_ev], 0.56)
                else:
                    bap[a:b, 0][vm_ev] = np.minimum(bap[a:b, 0][vm_ev], 0.26)
                    bap[a:b, 1][vm_ev] = np.minimum(bap[a:b, 1][vm_ev], 0.48)

        if info.manner in ("vowel", "diphthong", "glide", "liquid", "nasal", "tap") and gv_s > 0.0 and nf >= 4:
            seg_sub = cep[a:b, _GV_Q_START:_GV_Q_END]
            mu = np.mean(seg_sub, axis=0, keepdims=True)
            cur_std = np.std(seg_sub, axis=0)
            tgt_std = knowledge.lookup_gv(pk, ptag)[_GV_Q_START:_GV_Q_END]
            ratio = np.clip(tgt_std / np.maximum(cur_std, 0.15), 0.90, 1.38)
            scale_q = 1.0 + gv_s * _GV_BELL_W * (ratio - 1.0)
            cep[a:b, _GV_Q_START:_GV_Q_END] = mu + (seg_sub - mu) * scale_q[None, :]

    for ev in events:
        if ev.is_pause:
            pa = max(0, min(n_fr, int(ev.start_fr)))
            pb = max(pa, min(n_fr, int(ev.end_fr)))
            if pb > pa:
                voi_arr[pa:pb] = False
                f0_arr[pa:pb] = 0.0

    log_env = cep_to_log_env(cep, n_bins=N_BINS) + knowledge.ref_log_amp

    g_sc = float(np.clip(gain_scale, 0.0, 1.0))
    if g_sc > 0.0:
        for ev in events:
            if ev.is_pause or abs(float(ev.gain_db)) < 1e-4:
                continue
            a = max(0, min(n_fr, int(ev.start_fr)))
            b = max(a, min(n_fr, int(ev.end_fr)))
            if b > a:
                g_nep = float(np.clip(float(ev.gain_db), -5.0, 5.0)) / 8.685889638065035
                log_env[a:b] += g_sc * g_nep

    # Dead-quiet floor for pauses & uncovered spans + smooth 2-frame speech<->pause fades
    sil_mask = ~covered
    if np.any(sil_mask):
        log_env[sil_mask] = -16.0
        bap[sil_mask] = 0.99
        voi_arr[sil_mask] = False
        f0_arr[sil_mask] = 0.0
        idx_s = 0
        while idx_s < n_fr:
            if not sil_mask[idx_s]:
                idx_s += 1
                continue
            end_s = idx_s + 1
            while end_s < n_fr and sil_mask[end_s]:
                end_s += 1
            if (end_s - idx_s) >= 3:
                if idx_s >= 1 and covered[idx_s - 1]:
                    log_env[idx_s - 1] = np.clip(log_env[idx_s - 1] - (13.0 / 8.685889638065035), -16.0, 6.0)
                    if idx_s >= 2 and covered[idx_s - 2]:
                        log_env[idx_s - 2] = np.clip(log_env[idx_s - 2] - (5.0 / 8.685889638065035), -16.0, 6.0)
                if end_s < n_fr and covered[end_s]:
                    log_env[end_s] = np.clip(log_env[end_s] - (13.0 / 8.685889638065035), -16.0, 6.0)
                    if end_s + 1 < n_fr and covered[end_s + 1]:
                        log_env[end_s + 1] = np.clip(log_env[end_s + 1] - (5.0 / 8.685889638065035), -16.0, 6.0)
            idx_s = end_s

    ctrl = LarynxControl(
        f0_hz=f0_arr,
        voiced=voi_arr,
        log_env=np.clip(log_env, -16.0, 6.0),
        bap=np.clip(bap, 0.004, 0.995),
        glottal_tilt_db_per_oct=tilt,
        jitter_frac=jit,
        shimmer_frac=shim,
        n_samples=n_fr * HOP,
        rd=rd,
    )
    return ctrl, {"mode": "sf_native_multilingual", "hits": hit_counts}
