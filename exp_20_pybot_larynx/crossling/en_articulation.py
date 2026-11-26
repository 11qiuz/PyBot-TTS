"""Stage 9B Speaker-Independent English Articulation, Coarticulation & Rhythm Knowledge (`exp_20_pybot_larynx`).

All tables hold strictly zero neural/sklearn/pickle weights: pure deterministic acoustic
statistics extracted from forced-aligned English utterances after stripping each speaker's
LTAS and VTLN vocal-tract proportions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from scipy.fft import idct

from exp_20_pybot_larynx import EXP_DIR, N_BINS
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS

EN_ARTICULATION_PATH = EXP_DIR / "crossling" / "en_articulation_table.npz"
EN_ARTICULATION_SUMMARY_PATH = EXP_DIR / "crossling" / "en_articulation_summary.json"

N_ART_BINS = 5
N_ART_CEP = 24
N_COART_CEP = 20

ART_CLASSES: tuple[str, ...] = (
    "SIL",
    "labial",
    "alveolar",
    "postalv_pal",
    "velar_glottal",
    "nasal",
    "liquid_rhotic",
    "front_vowel",
    "back_open_vowel",
)
_CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(ART_CLASSES)}

VOICELESS_OBSTRUENTS = frozenset({
    "p", "pʰ", "t", "tʰ", "k", "kʰ",
    "f", "θ", "s", "ʃ", "ʂ", "ɕ", "h", "x", "ʍ",
    "tʃ", "ts", "tsʰ", "tɕ", "tɕʰ", "ʈʂ", "ʈʂʰ",
})

VOWEL_IPAS = frozenset({
    "a", "ɑ", "æ", "ʌ", "ɔ", "o", "ɤ", "e", "ɛ", "ə", "ɨ", "ɪ", "i",
    "y", "u", "ʉ", "ʊ", "ɚ", "ɝ", "ɻ̩", "ɹ̩",
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ",
})

NEW_EN_OBSTRUENTS = frozenset({"θ", "ð", "v", "z", "ʒ", "b", "d", "ɡ", "dʒ"})

_LABIAL = frozenset({"p", "pʰ", "b", "f", "v", "w", "ʍ"})
_ALVEOLAR = frozenset({"t", "tʰ", "d", "s", "z", "θ", "ð", "ts", "tsʰ", "ɾ"})
_POSTALV_PAL = frozenset({"ʃ", "ʒ", "tʃ", "dʒ", "ɕ", "tɕ", "tɕʰ", "ʂ", "ʈʂ", "ʈʂʰ", "j", "ɥ"})
_VELAR_GLOTTAL = frozenset({"k", "kʰ", "ɡ", "h", "x"})
_NASAL = frozenset({"m", "m̩", "n", "n̩", "ŋ"})
_LIQUID_RHOTIC = frozenset({"l", "l̩", "ɹ", "ɻ", "ɚ", "ɝ", "ɻ̩", "ɹ̩"})
_FRONT_VOWEL = frozenset({"i", "ɪ", "e", "ɛ", "æ", "y", "ɨ", "eɪ", "aɪ"})
_BACK_OPEN_VOWEL = frozenset({"a", "ɑ", "ʌ", "ɔ", "o", "ɤ", "ə", "u", "ʉ", "ʊ", "aʊ", "oʊ", "ɔɪ"})

_CODA_STOP_IPAS = frozenset({"p", "t", "k", "b", "d", "ɡ"})
_DB_TO_NEPER = 1.0 / NEPER_TO_DB

# Precomputed orthogonal IDCT basis matrices matching `synth.cep_to_log_env` identically (~10x faster).
_CEP24_BASIS = idct(np.eye(N_ART_CEP, N_BINS, dtype=np.float64), type=2, norm="ortho", axis=-1)
_CEP20_BASIS = _CEP24_BASIS[:N_COART_CEP]

_TABLE_LOCK = Lock()
_CACHED_PATH: str | None = None
_CACHED_TABLE: "EnArticulationTable | None" = None


def classify_articulatory_class(ipa: str | None) -> int:
    """Map an IPA or allophone key to one of 9 place/manner articulatory classes."""
    if not ipa:
        return 0
    base = str(ipa).replace("_coda", "").strip()
    if not base or base.lower() == "sil":
        return 0
    if base in _NASAL:
        return 5
    if base in _LIQUID_RHOTIC:
        return 6
    if base in _LABIAL:
        return 1
    if base in _ALVEOLAR:
        return 2
    if base in _POSTALV_PAL:
        return 3
    if base in _VELAR_GLOTTAL:
        return 4
    if base in _FRONT_VOWEL:
        return 7
    if base in _BACK_OPEN_VOWEL:
        return 8
    return 0


def resolve_en_allophone(ipa: str, next_ipa: str | None, same_word_next: bool) -> str:
    """Resolve positional English allophone key (`l_coda` dark-L, unreleased `*_coda` stops)."""
    base = str(ipa).strip()
    if base == "l" and (not same_word_next or next_ipa not in VOWEL_IPAS):
        return "l_coda"
    if base in _CODA_STOP_IPAS and (not same_word_next or next_ipa not in VOWEL_IPAS):
        return f"{base}_coda"
    return base


@dataclass
class EnArticulationTable:
    """Compact (<1 MB), speaker-independent English articulation, coarticulation, and rhythm table."""

    keys: list[str]
    key_to_id: dict[str, int]
    phone_is_voiced: np.ndarray       # (K,) bool
    phone_counts: np.ndarray          # (K,) int32
    delta_cep: np.ndarray             # (K, 5, 24) float64
    formants_5bin: np.ndarray         # (K, 5, 3) float64
    voicing_prob_5bin: np.ndarray     # (K, 5) float64
    rel_level_db_5bin: np.ndarray     # (K, 5) float64
    left_coart_cep: np.ndarray        # (K, 9, 20) float64
    right_coart_cep: np.ndarray       # (K, 9, 20) float64
    left_coart_f123: np.ndarray       # (K, 9, 3) float64
    right_coart_f123: np.ndarray      # (K, 9, 3) float64
    dur_ms_by_stress: np.ndarray      # (K, 3) float64
    pre_fortis_scale: float
    cluster2_scale: float
    cluster3_scale: float
    phrase_final_scale: float
    stress_f0_st_5bin: np.ndarray     # (3, 5) float64
    stress_gain_db: np.ndarray        # (3,) float64

    @classmethod
    def load(cls, path: Path | str = EN_ARTICULATION_PATH) -> "EnArticulationTable":
        with np.load(Path(path), allow_pickle=False) as d:
            keys = [str(k) for k in d["keys"]]
            return cls(
                keys=keys,
                key_to_id={k: i for i, k in enumerate(keys)},
                phone_is_voiced=d["phone_is_voiced"].astype(bool),
                phone_counts=d["phone_counts"].astype(np.int32),
                delta_cep=d["delta_cep"].astype(np.float64),
                formants_5bin=d["formants_5bin"].astype(np.float64),
                voicing_prob_5bin=d["voicing_prob_5bin"].astype(np.float64),
                rel_level_db_5bin=d["rel_level_db_5bin"].astype(np.float64),
                left_coart_cep=d["left_coart_cep"].astype(np.float64),
                right_coart_cep=d["right_coart_cep"].astype(np.float64),
                left_coart_f123=d["left_coart_f123"].astype(np.float64),
                right_coart_f123=d["right_coart_f123"].astype(np.float64),
                dur_ms_by_stress=d["dur_ms_by_stress"].astype(np.float64),
                pre_fortis_scale=float(d["pre_fortis_scale"][0]),
                cluster2_scale=float(d["cluster2_scale"][0]),
                cluster3_scale=float(d["cluster3_scale"][0]),
                phrase_final_scale=float(d["phrase_final_scale"][0]),
                stress_f0_st_5bin=d["stress_f0_st_5bin"].astype(np.float64),
                stress_gain_db=d["stress_gain_db"].astype(np.float64),
            )

    def _resolve_id(self, key: str, ipa_fallback: str | None = None) -> int:
        if key in self.key_to_id:
            return self.key_to_id[key]
        base = ipa_fallback or str(key).replace("_coda", "")
        if base in self.key_to_id:
            return self.key_to_id[base]
        return self.key_to_id.get("ə", 0)

    def predict_duration_ms(
        self,
        key: str,
        ipa: str,
        stress: int | None,
        *,
        next_ipa: str | None,
        same_word_next: bool,
        cluster_len: int,
        is_phrase_final: bool,
        rate: float = 1.0,
    ) -> float:
        """Predict corpus-calibrated phone duration (ms) with pre-fortis clipping & rhythm."""
        kid = self._resolve_id(key, ipa)
        s_idx = 1 if stress == 1 else (2 if stress == 2 else 0)
        d_meas = float(self.dur_ms_by_stress[kid, s_idx])
        cnt = int(self.phone_counts[kid])

        base_spec = IPA_SPECS[ipa].base_dur_ms if ipa in IPA_SPECS else 75.0
        if ipa in VOWEL_IPAS:
            if stress == 1:
                base_rule = base_spec * 1.16
            elif stress == 2:
                base_rule = base_spec * 1.04
            elif stress == 0:
                base_rule = max(40.0, base_spec * 0.74)
            else:
                base_rule = base_spec
        else:
            base_rule = base_spec

        w = cnt / (cnt + 14.0) if cnt > 0 else 0.0
        dur = w * d_meas + (1.0 - w) * base_rule
        if key == "l_coda" and not is_phrase_final:
            dur = min(dur, 92.0)

        if ipa in VOWEL_IPAS and same_word_next and next_ipa in VOICELESS_OBSTRUENTS:
            pf_eff = min(self.pre_fortis_scale, 0.78) if stress == 1 else self.pre_fortis_scale
            dur *= pf_eff
        if ipa not in VOWEL_IPAS and cluster_len >= 2:
            dur *= self.cluster2_scale if cluster_len == 2 else self.cluster3_scale
        if is_phrase_final:
            dur *= self.phrase_final_scale

        eff_rate = max(0.60, min(1.60, float(rate)))
        return float(np.clip(dur / eff_rate, 24.0, 290.0))

    def _interp_bin(self, phase_t: float) -> tuple[int, int, float]:
        u = float(np.clip(phase_t, 0.0, 1.0)) * float(N_ART_BINS - 1)
        i0 = min(N_ART_BINS - 1, max(0, int(math.floor(u))))
        i1 = min(N_ART_BINS - 1, i0 + 1)
        return i0, i1, u - float(i0)

    def sample_sonorant_articulation(
        self,
        key: str,
        phase_t: float,
        prev_cls: int,
        next_cls: int,
        *,
        ipa_fallback: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return `(f123_canon, delta_env_513, coart_env_513, rel_db)` for a voiced frame."""
        kid = self._resolve_id(key, ipa_fallback)
        i0, i1, w = self._interp_bin(phase_t)
        t = float(np.clip(phase_t, 0.0, 1.0))

        p_cls = max(0, min(len(ART_CLASSES) - 1, int(prev_cls)))
        n_cls = max(0, min(len(ART_CLASSES) - 1, int(next_cls)))

        f_base = (1.0 - w) * self.formants_5bin[kid, i0] + w * self.formants_5bin[kid, i1]
        w_l = (1.0 - t) ** 2.0
        w_r = t ** 2.0
        f_canon = f_base + w_l * self.left_coart_f123[kid, p_cls] + w_r * self.right_coart_f123[kid, n_cls]
        f1 = float(np.clip(f_canon[0], 230.0, 1180.0))
        f2 = float(np.clip(f_canon[1], max(f1 + 220.0, 740.0), 3050.0))
        f3 = float(np.clip(f_canon[2], max(f2 + 260.0, 1780.0), 4200.0))
        f123 = np.array([f1, f2, f3], dtype=np.float64)

        d_cep = (1.0 - w) * self.delta_cep[kid, i0] + w * self.delta_cep[kid, i1]
        c_cep = ((1.0 - t) ** 2.2) * self.left_coart_cep[kid, p_cls] + (t ** 2.2) * self.right_coart_cep[kid, n_cls]

        delta_env = d_cep @ _CEP24_BASIS
        coart_env = c_cep @ _CEP20_BASIS
        rel_db = float((1.0 - w) * self.rel_level_db_5bin[kid, i0] + w * self.rel_level_db_5bin[kid, i1])
        return f123, delta_env, coart_env, rel_db

    def sample_obstruent_articulation(
        self,
        key: str,
        phase_t: float,
        prev_cls: int,
        next_cls: int,
        *,
        ipa_fallback: str | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Return `(delta_obs_env_513, voicing_prob, rel_level_db)` for an obstruent frame."""
        kid = self._resolve_id(key, ipa_fallback)
        i0, i1, w = self._interp_bin(phase_t)
        t = float(np.clip(phase_t, 0.0, 1.0))

        p_cls = max(0, min(len(ART_CLASSES) - 1, int(prev_cls)))
        n_cls = max(0, min(len(ART_CLASSES) - 1, int(next_cls)))

        d_cep = (1.0 - w) * self.delta_cep[kid, i0] + w * self.delta_cep[kid, i1]
        c_cep = ((1.0 - t) ** 2.2) * self.left_coart_cep[kid, p_cls] + (t ** 2.2) * self.right_coart_cep[kid, n_cls]
        obs_cep = d_cep.copy()
        obs_cep[:N_COART_CEP] += 0.35 * c_cep
        delta_env = obs_cep @ _CEP24_BASIS

        v_prob = float(np.clip((1.0 - w) * self.voicing_prob_5bin[kid, i0] + w * self.voicing_prob_5bin[kid, i1], 0.0, 1.0))
        rel_db = float((1.0 - w) * self.rel_level_db_5bin[kid, i0] + w * self.rel_level_db_5bin[kid, i1])
        return delta_env, v_prob, rel_db


def get_en_articulation_table(path: Path | str = EN_ARTICULATION_PATH) -> EnArticulationTable | None:
    """Thread-safe cached loader for :class:`EnArticulationTable` (None if artifact absent)."""
    global _CACHED_PATH, _CACHED_TABLE
    p = Path(path)
    p_str = str(p.resolve()) if p.exists() else str(p)
    with _TABLE_LOCK:
        if _CACHED_TABLE is not None and _CACHED_PATH == p_str:
            return _CACHED_TABLE
    if not p.exists():
        return None
    tbl = EnArticulationTable.load(p)
    with _TABLE_LOCK:
        _CACHED_TABLE = tbl
        _CACHED_PATH = p_str
    return tbl


def clear_en_articulation_cache() -> None:
    """Reset module cache (used after rebuilding the `.npz` table)."""
    global _CACHED_PATH, _CACHED_TABLE
    with _TABLE_LOCK:
        _CACHED_PATH = None
        _CACHED_TABLE = None
