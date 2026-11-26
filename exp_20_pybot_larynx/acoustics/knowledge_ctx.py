"""Optional context timing, voicing, and relative energy bias loader (zero neural weights)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from exp_20_pybot_larynx import EXP_DIR
from exp_20_pybot_larynx.acoustics.knowledge import normalize_of
from exp_20_pybot_larynx.arbiter import classify_final_group

KNOWLEDGE_CTX_PATH = EXP_DIR / "acoustics" / "knowledge_ctx.npz"


@dataclass(frozen=True)
class OnsetTimingStat:
    ms_med: float
    ms_p25: float
    ms_p75: float
    frac_med: float
    uv_mean: float
    count: int


def _load_onset_table(d: Any, prefix: str) -> dict[str, OnsetTimingStat]:
    out: dict[str, OnsetTimingStat] = {}
    keys = d[f"{prefix}_k"]
    ms_med = d[f"{prefix}_ms_med"]
    ms_p25 = d[f"{prefix}_ms_p25"]
    ms_p75 = d[f"{prefix}_ms_p75"]
    frac_med = d[f"{prefix}_frac_med"]
    uv_mean = d[f"{prefix}_uv_mean"]
    cnt = d[f"{prefix}_cnt"]
    for i in range(len(keys)):
        out[str(keys[i])] = OnsetTimingStat(
            ms_med=float(ms_med[i]),
            ms_p25=float(ms_p25[i]),
            ms_p75=float(ms_p75[i]),
            frac_med=float(frac_med[i]),
            uv_mean=float(uv_mean[i]),
            count=int(cnt[i]),
        )
    return out


def _load_scalar_table(d: Any, prefix: str) -> dict[str, tuple[float, int]]:
    out: dict[str, tuple[float, int]] = {}
    keys = d[f"{prefix}_k"]
    med = d[f"{prefix}_med"]
    cnt = d[f"{prefix}_cnt"]
    for i in range(len(keys)):
        out[str(keys[i])] = (float(med[i]), int(cnt[i]))
    return out


@dataclass
class ContextKnowledge:
    """Inspectable context-dependent onset timing, unvoiced ratio, and tone/position energy bias."""

    on_oft: dict[str, OnsetTimingStat]
    on_ofg: dict[str, OnsetTimingStat]
    on_ot: dict[str, OnsetTimingStat]
    on_o: dict[str, OnsetTimingStat]
    fin_dur_ft: dict[str, tuple[float, int]]
    tone_pos_rel_db: dict[str, tuple[float, int]]
    n_train_utts: int

    @classmethod
    def load(cls, path: Path | str = KNOWLEDGE_CTX_PATH) -> ContextKnowledge:
        with np.load(Path(path), allow_pickle=False) as d:
            return cls(
                on_oft=_load_onset_table(d, "on_oft"),
                on_ofg=_load_onset_table(d, "on_ofg"),
                on_ot=_load_onset_table(d, "on_ot"),
                on_o=_load_onset_table(d, "on_o"),
                fin_dur_ft=_load_scalar_table(d, "fin_dur_ft"),
                tone_pos_rel_db=_load_scalar_table(d, "tone_pos_rel_db"),
                n_train_utts=int(d["n_train_utts"][0]),
            )

    def lookup_onset_timing(
        self,
        onset: str,
        final: str,
        tone: int,
        *,
        min_oft_count: int = 8,
        min_sub_count: int = 5,
    ) -> tuple[float, float, float, str]:
        """Return `(onset_ms, onset_frac, onset_uv_frac, tier)` with hierarchical backoff."""
        o, f = normalize_of(onset, final)
        if not o:
            return 0.0, 0.0, 0.0, "zero_onset"
        fg = classify_final_group(f)
        t = int(tone)

        stat = self.on_oft.get(f"{o}|{fg}|{t}")
        if stat is not None and stat.count >= min_oft_count:
            return stat.ms_med, stat.frac_med, stat.uv_mean, "on_oft"

        stat = self.on_ofg.get(f"{o}|{fg}")
        if stat is not None and stat.count >= min_sub_count:
            return stat.ms_med, stat.frac_med, stat.uv_mean, "on_ofg"

        stat = self.on_ot.get(f"{o}|{t}")
        if stat is not None and stat.count >= min_sub_count:
            return stat.ms_med, stat.frac_med, stat.uv_mean, "on_ot"

        stat = self.on_o.get(o)
        if stat is not None and stat.count >= 1:
            return stat.ms_med, stat.frac_med, stat.uv_mean, "on_o"

        return 35.0, 0.25, 0.78, "fallback"

    def predict_onset_frames(
        self,
        onset: str,
        final: str,
        tone: int,
        syl_dur_ms: float,
    ) -> tuple[int, int, str]:
        """Predict `(n_onset_frames, n_unvoiced_frames, tier)` at 5 ms hop."""
        o, _ = normalize_of(onset, final)
        if not o:
            return 0, 0, "zero_onset"
        n_syl = max(3, int(round(float(syl_dur_ms) / 5.0)))
        ms_med, frac_med, uv_mean, tier = self.lookup_onset_timing(o, final, tone)
        # Blend absolute onset duration with proportional duration for extreme speech rates
        pred_ms = 0.70 * ms_med + 0.30 * (frac_med * float(syl_dur_ms))
        n_on = max(1, min(n_syl - 2, int(round(pred_ms / 5.0))))
        n_uv = max(0, min(n_on, int(round(uv_mean * n_on))))
        return n_on, n_uv, tier

    def lookup_final_dur_ms(self, final: str, tone: int, default_ms: float = 140.0) -> float:
        _, f = normalize_of("", final)
        e = self.fin_dur_ft.get(f"{f}|{int(tone)}")
        return e[0] if e is not None and e[1] >= 3 else float(default_ms)

    def lookup_tone_energy_bias_db(self, tone: int, syl_pos: str = "medial") -> float:
        """Return empirical relative energy bias in dB for `(tone, position)`."""
        t = int(tone)
        e = self.tone_pos_rel_db.get(f"{t}|{syl_pos}")
        if e is not None and e[1] >= 10:
            return e[0]
        e_all = self.tone_pos_rel_db.get(f"{t}|all")
        return e_all[0] if e_all is not None else 0.0
