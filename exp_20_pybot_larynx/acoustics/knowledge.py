"""Zero-leakage data splits and explicit cepstral/aperiodicity acoustic knowledge loader."""
from __future__ import annotations

from dataclasses import dataclass
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

from exp_20_pybot_larynx import (
    DEV_60_SEED,
    HOP,
    KNOWLEDGE_PATH,
    N_BAP,
    N_BINS,
    N_CEP,
    N_FINAL_BINS,
    N_ONSET_BINS,
    REPO_ROOT,
)
from exp_20_pybot_larynx.arbiter import (
    NEPER_TO_DB,
    UtteranceContext,
    classify_final_group,
    classify_onset_place,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env
from exp_v14_fluent_zh_tts.prosody_v14 import _normalize_of

INDEX_V15_PKL = (
    REPO_ROOT / "legacy" / "exp_v15_bznsyp_tts" / "bznsyp_index_v15.pkl"
    if (REPO_ROOT / "legacy" / "exp_v15_bznsyp_tts" / "bznsyp_index_v15.pkl").is_file()
    else REPO_ROOT / "exp_v15_bznsyp_tts" / "bznsyp_index_v15.pkl"
)
HELDOUT_20_IDS: tuple[str, ...] = (
    "001395", "004343", "006484", "009118", "001309",
    "009210", "001222", "009086", "000547", "000988",
    "001526", "002100", "002724", "003366", "004112",
    "004885", "005750", "006801", "007631", "008651",
)

_INDEX_CACHE: dict[str, Any] | None = None


def load_v15_index_cached() -> dict[str, Any]:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        with open(INDEX_V15_PKL, "rb") as f:
            _INDEX_CACHE = pickle.load(f)
    return _INDEX_CACHE


def normalize_of(onset: str, final: str) -> tuple[str, str]:
    o, f = _normalize_of(onset, final)
    f = f.replace("ü", "v")
    return o, f


def get_splits() -> tuple[list[str], list[str], list[str]]:
    """Return pairwise-disjoint `(TRAIN_IDS, DEV_60_IDS, HELDOUT_20_IDS)`."""
    idx = load_v15_index_cached()
    all_train = sorted(idx["train_utt_metas"].keys())
    dev_60 = sorted(random.Random(DEV_60_SEED).sample(all_train, 60))
    dev_set = set(dev_60)
    held_set = set(HELDOUT_20_IDS)
    train_ids = sorted(uid for uid in all_train if uid not in dev_set and uid not in held_set)
    held_ids = list(HELDOUT_20_IDS)

    if set(train_ids) & dev_set:
        raise AssertionError("TRAIN and DEV_60 overlap")
    if set(train_ids) & held_set:
        raise AssertionError("TRAIN and HELDOUT_20 overlap")
    if dev_set & held_set:
        raise AssertionError("DEV_60 and HELDOUT_20 overlap")
    return train_ids, dev_60, held_ids


def _table_to_dict(
    keys: np.ndarray,
    cep: np.ndarray,
    bap: np.ndarray,
    vprob: np.ndarray,
    counts: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for i in range(len(keys)):
        out[str(keys[i])] = (
            cep[i].astype(np.float64),
            bap[i].astype(np.float64),
            vprob[i].astype(np.float64),
            int(counts[i]),
        )
    return out


def _interp_phase_bins(bin_vals: np.ndarray, u: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Linearly interpolate `(K, D)` phase bin profiles at normalized positions `u in [0, 1]`."""
    k = bin_vals.shape[0]
    if k == 1:
        return np.repeat(bin_vals, len(u), axis=0)
    uc = np.clip(u, centers[0], centers[-1])
    hi = np.searchsorted(centers, uc, side="right")
    hi = np.clip(hi, 1, k - 1)
    lo = hi - 1
    denom = np.maximum(1e-9, centers[hi] - centers[lo])
    w = ((uc - centers[lo]) / denom)[:, None]
    return (1.0 - w) * bin_vals[lo] + w * bin_vals[hi]


@dataclass
class AcousticKnowledge:
    """Inspectable explicit cepstral + BAP + onset-timing knowledge base (zero weights)."""

    on_grp: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]
    on_only: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]
    on_global: tuple[np.ndarray, np.ndarray, np.ndarray]

    fin_oft: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]
    fin_pft: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]
    fin_ft: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]
    fin_f: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]
    fin_global: tuple[np.ndarray, np.ndarray, np.ndarray]

    st_on: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]
    st_ft: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]
    st_f: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]
    st_bap_global: np.ndarray

    onset_frac: dict[str, float]
    onset_ms: dict[str, float]
    sil_cep64: np.ndarray
    sil_bap6: np.ndarray
    speaker_f0_med_hz: float
    ref_log_amp: float
    n_train_utts: int

    @classmethod
    def load(cls, path: Path | str = KNOWLEDGE_PATH) -> AcousticKnowledge:
        with np.load(Path(path), allow_pickle=False) as d:
            on_frac_map = {str(k): float(v) for k, v in zip(d["onset_frac_keys"], d["onset_frac_vals"])}
            on_ms_map = {str(k): float(v) for k, v in zip(d["onset_frac_keys"], d["onset_ms_vals"])}
            return cls(
                on_grp=_table_to_dict(d["on_grp_k"], d["on_grp_cep"], d["on_grp_bap"], d["on_grp_vp"], d["on_grp_cnt"]),
                on_only=_table_to_dict(d["on_only_k"], d["on_only_cep"], d["on_only_bap"], d["on_only_vp"], d["on_only_cnt"]),
                on_global=(d["on_glob_cep"].astype(np.float64), d["on_glob_bap"].astype(np.float64), d["on_glob_vp"].astype(np.float64)),
                fin_oft=_table_to_dict(d["fin_oft_k"], d["fin_oft_cep"], d["fin_oft_bap"], d["fin_oft_vp"], d["fin_oft_cnt"]),
                fin_pft=_table_to_dict(d["fin_pft_k"], d["fin_pft_cep"], d["fin_pft_bap"], d["fin_pft_vp"], d["fin_pft_cnt"]),
                fin_ft=_table_to_dict(d["fin_ft_k"], d["fin_ft_cep"], d["fin_ft_bap"], d["fin_ft_vp"], d["fin_ft_cnt"]),
                fin_f=_table_to_dict(d["fin_f_k"], d["fin_f_cep"], d["fin_f_bap"], d["fin_f_vp"], d["fin_f_cnt"]),
                fin_global=(d["fin_glob_cep"].astype(np.float64), d["fin_glob_bap"].astype(np.float64), d["fin_glob_vp"].astype(np.float64)),
                st_on=_table_to_dict(d["st_on_k"], d["st_on_cep"], d["st_on_bap"], d["st_on_vp"], d["st_on_cnt"]),
                st_ft=_table_to_dict(d["st_ft_k"], d["st_ft_cep"], d["st_ft_bap"], d["st_ft_vp"], d["st_ft_cnt"]),
                st_f=_table_to_dict(d["st_f_k"], d["st_f_cep"], d["st_f_bap"], d["st_f_vp"], d["st_f_cnt"]),
                st_bap_global=d["st_bap_global"].astype(np.float64),
                onset_frac=on_frac_map,
                onset_ms=on_ms_map,
                sil_cep64=d["sil_cep64"].astype(np.float64),
                sil_bap6=d["sil_bap6"].astype(np.float64),
                speaker_f0_med_hz=float(d["speaker_f0_med_hz"][0]),
                ref_log_amp=float(d["ref_log_amp"][0]),
                n_train_utts=int(d["n_train_utts"][0]),
            )

    def lookup_onset_dynamic(self, onset: str, final: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        fg = classify_final_group(final)
        k_grp = f"{onset}|{fg}"
        e = self.on_grp.get(k_grp)
        if e is not None and e[3] >= 3:
            return e[0], e[1], e[2], "on_grp"
        e = self.on_only.get(onset)
        if e is not None and e[3] >= 1:
            return e[0], e[1], e[2], "on_only"
        return self.on_global[0], self.on_global[1], self.on_global[2], "on_fallback"

    def lookup_final_dynamic(self, onset: str, final: str, tone: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        k_oft = f"{onset}|{final}|{tone}"
        e = self.fin_oft.get(k_oft)
        if e is not None and e[3] >= 4:
            return e[0], e[1], e[2], "exact_oft"
        place = classify_onset_place(onset)
        k_pft = f"{place}|{final}|{tone}"
        e = self.fin_pft.get(k_pft)
        if e is not None and e[3] >= 3:
            return e[0], e[1], e[2], "place_ft"
        k_ft = f"{final}|{tone}"
        e = self.fin_ft.get(k_ft)
        if e is not None and e[3] >= 1:
            return e[0], e[1], e[2], "ft"
        e = self.fin_f.get(final)
        if e is not None and e[3] >= 1:
            return e[0], e[1], e[2], "f"
        return self.fin_global[0], self.fin_global[1], self.fin_global[2], "fallback"


_ONSET_CENTERS = (np.arange(N_ONSET_BINS, dtype=np.float64) + 0.5) / float(N_ONSET_BINS)
_FINAL_CENTERS = (np.arange(N_FINAL_BINS, dtype=np.float64) + 0.5) / float(N_FINAL_BINS)


def build_base_control(
    ctx: UtteranceContext,
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    knowledge: AcousticKnowledge,
    *,
    mode: str = "dynamic",
    syl_gain_db: np.ndarray | None = None,
) -> tuple[LarynxControl, dict[str, Any]]:
    """Build base frame-level LarynxControl from explicit acoustic knowledge."""
    n_fr = ctx.n_frames
    cep = np.repeat(knowledge.sil_cep64[None, :], n_fr, axis=0)
    bap = np.repeat(knowledge.sil_bap6[None, :], n_fr, axis=0)

    hit_counts: dict[str, int] = {
        "exact_oft": 0,
        "place_ft": 0,
        "ft": 0,
        "f": 0,
        "fallback": 0,
        "on_grp": 0,
        "on_only": 0,
        "on_fallback": 0,
    }

    for s in ctx.syllables:
        a = max(0, min(n_fr, int(s.start_fr)))
        b = max(a, min(n_fr, int(s.end_fr)))
        sp = max(a, min(b, int(s.split_fr)))
        if b <= a:
            continue

        o, f = normalize_of(s.onset, s.final)
        t = int(s.tone)

        if mode == "static":
            if o and sp > a:
                e_on = knowledge.st_on.get(o)
                if e_on is not None:
                    cep[a:sp] = e_on[0][None, :]
                    bap[a:sp] = e_on[1][None, :]
                else:
                    cep[a:sp] = np.mean(knowledge.on_global[0], axis=0, keepdims=True)
                    bap[a:sp] = knowledge.st_bap_global[None, :]
                f_a = sp
            else:
                f_a = a

            if b > f_a:
                e_fin = knowledge.st_ft.get(f"{f}|{t}") or knowledge.st_f.get(f)
                if e_fin is not None:
                    cep[f_a:b] = e_fin[0][None, :]
                    bap[f_a:b] = knowledge.st_bap_global[None, :]
                    hit_counts["ft" if f"{f}|{t}" in knowledge.st_ft else "f"] += 1
                else:
                    cep[f_a:b] = np.mean(knowledge.fin_global[0], axis=0, keepdims=True)
                    bap[f_a:b] = knowledge.st_bap_global[None, :]
                    hit_counts["fallback"] += 1
        elif mode == "dynamic":
            if o and sp > a:
                o_cep, o_bap, _, o_tag = knowledge.lookup_onset_dynamic(o, f)
                hit_counts[o_tag] += 1
                u_on = ctx.phase_u[a:sp]
                cep[a:sp] = _interp_phase_bins(o_cep, u_on, _ONSET_CENTERS)
                bap[a:sp] = _interp_phase_bins(o_bap, u_on, _ONSET_CENTERS)
                f_a = sp
            else:
                f_a = a

            if b > f_a:
                f_cep, f_bap, _, f_tag = knowledge.lookup_final_dynamic(o, f, t)
                hit_counts[f_tag] += 1
                u_fi = ctx.phase_u[f_a:b]
                cep[f_a:b] = _interp_phase_bins(f_cep, u_fi, _FINAL_CENTERS)
                bap[f_a:b] = _interp_phase_bins(f_bap, u_fi, _FINAL_CENTERS)
        else:
            raise ValueError(f"Unsupported base mode: {mode!r}")

    log_env = cep_to_log_env(cep, n_bins=N_BINS) + knowledge.ref_log_amp

    if syl_gain_db is not None and len(syl_gain_db) == len(ctx.syllables):
        for idx_s, s in enumerate(ctx.syllables):
            a = max(0, min(n_fr, int(s.start_fr)))
            b = max(a, min(n_fr, int(s.end_fr)))
            if b > a:
                g_nep = float(np.clip(syl_gain_db[idx_s], -6.0, 6.0)) / NEPER_TO_DB
                log_env[a:b] += 0.5 * g_nep

    ctrl = LarynxControl(
        f0_hz=np.asarray(f0_hz, dtype=np.float64),
        voiced=np.asarray(voiced, dtype=bool),
        log_env=log_env,
        bap=np.clip(bap, 0.005, 0.995),
        n_samples=n_fr * HOP,
    )
    return ctrl, {"mode": mode, "hits": hit_counts}
