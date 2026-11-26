"""Continuous layered PyBot Arbiter, phonetic taxonomy, and utterance context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from exp_20_pybot_larynx import N_BAP, N_BINS
from exp_20_pybot_larynx.larynx.synth import LarynxControl

NEPER_TO_DB = 8.685889638065035

ONSET_PLACE: dict[str, str] = {
    "": "zero",
    "b": "bilabial",
    "p": "bilabial",
    "m": "bilabial",
    "f": "labiodental",
    "d": "alveolar",
    "t": "alveolar",
    "n": "alveolar",
    "l": "alveolar",
    "z": "dental",
    "c": "dental",
    "s": "dental",
    "zh": "retroflex",
    "ch": "retroflex",
    "sh": "retroflex",
    "r": "retroflex",
    "j": "palatal",
    "q": "palatal",
    "x": "palatal",
    "g": "velar",
    "k": "velar",
    "h": "velar",
}

ONSET_MANNER: dict[str, str] = {
    "": "zero",
    "b": "stop_unasp",
    "d": "stop_unasp",
    "g": "stop_unasp",
    "p": "stop_asp",
    "t": "stop_asp",
    "k": "stop_asp",
    "z": "affr_unasp",
    "zh": "affr_unasp",
    "j": "affr_unasp",
    "c": "affr_asp",
    "ch": "affr_asp",
    "q": "affr_asp",
    "f": "fric",
    "s": "fric",
    "sh": "fric",
    "x": "fric",
    "h": "fric",
    "m": "nasal",
    "n": "nasal",
    "l": "liquid",
    "r": "liquid",
}

OBSTRUENT_MANNERS = frozenset({"stop_unasp", "stop_asp", "affr_unasp", "affr_asp", "fric"})

FINAL_GROUP: dict[str, str] = {
    "i": "i_front",
    "ia": "i_front",
    "ie": "i_front",
    "iao": "i_front",
    "iu": "i_front",
    "iou": "i_front",
    "ian": "i_front",
    "in": "i_front",
    "iang": "i_front",
    "ing": "i_front",
    "ei": "i_front",
    "iz": "i_front",
    "ir": "i_front",
    "ii": "i_front",
    "ih": "i_front",
    "i_z": "i_front",
    "i_zh": "i_front",
    "u": "u_round",
    "ua": "u_round",
    "uo": "u_round",
    "uai": "u_round",
    "ui": "u_round",
    "uei": "u_round",
    "uan": "u_round",
    "un": "u_round",
    "uen": "u_round",
    "uang": "u_round",
    "ueng": "u_round",
    "o": "u_round",
    "ou": "u_round",
    "ong": "u_round",
    "v": "v_front_round",
    "ü": "v_front_round",
    "ve": "v_front_round",
    "üe": "v_front_round",
    "van": "v_front_round",
    "üan": "v_front_round",
    "vn": "v_front_round",
    "ün": "v_front_round",
    "iong": "v_front_round",
    "a": "a_open_central",
    "ai": "a_open_central",
    "ao": "a_open_central",
    "an": "a_open_central",
    "ang": "a_open_central",
    "e": "a_open_central",
    "en": "a_open_central",
    "eng": "a_open_central",
    "er": "a_open_central",
}

FINAL_HAS_NASAL_CODA = frozenset({
    "an", "en", "ang", "eng", "ong",
    "ian", "in", "iang", "ing", "iong",
    "uan", "un", "uen", "uang", "ueng",
    "van", "vn", "üan", "ün", "on",
})


def classify_onset_place(onset: str) -> str:
    o = "" if onset in ("y", "w") else str(onset)
    return ONSET_PLACE.get(o, "zero" if not o else "alveolar")


def classify_onset_manner(onset: str) -> str:
    o = "" if onset in ("y", "w") else str(onset)
    return ONSET_MANNER.get(o, "zero" if not o else "fric")


def classify_final_group(final: str) -> str:
    f = str(final).replace("v", "ü") if final.startswith("v") else str(final)
    return FINAL_GROUP.get(final, FINAL_GROUP.get(f, "a_open_central"))


def has_nasal_coda(final: str) -> bool:
    return str(final) in FINAL_HAS_NASAL_CODA


@dataclass(slots=True)
class SyllableSpan:
    index: int
    hanzi: str
    onset: str
    final: str
    tone: int
    start_fr: int
    split_fr: int
    end_fr: int
    pause_after_ms: float = 0.0
    prosodic_break: int = 0


@dataclass
class UtteranceContext:
    utt_id: str
    n_frames: int
    syllables: list[SyllableSpan]
    syl_idx_per_frame: np.ndarray   # (T,) int32, -1 for silence
    is_onset: np.ndarray            # (T,) bool
    phase_u: np.ndarray             # (T,) float64 in [0, 1] inside onset or final
    speaker_f0_med_hz: float = 247.69
    continues_after: bool = False


def build_utterance_context(
    utt_id: str,
    n_frames: int,
    syllables: list[SyllableSpan],
    speaker_f0_med_hz: float = 247.69,
    *,
    continues_after: bool = False,
) -> UtteranceContext:
    """Populate frame-aligned syllable index, onset flag, and intra-phone phase progress."""
    syl_idx = np.full(n_frames, -1, dtype=np.int32)
    is_onset = np.zeros(n_frames, dtype=bool)
    phase_u = np.zeros(n_frames, dtype=np.float64)

    for s in syllables:
        a = max(0, min(n_frames, int(s.start_fr)))
        b = max(a, min(n_frames, int(s.end_fr)))
        sp = max(a, min(b, int(s.split_fr)))
        if b <= a:
            continue
        syl_idx[a:b] = int(s.index)
        if s.onset and sp > a:
            is_onset[a:sp] = True
            n_on = sp - a
            phase_u[a:sp] = np.linspace(0.0, 1.0, n_on) if n_on > 1 else 0.5
            n_fi = b - sp
            if n_fi > 0:
                phase_u[sp:b] = np.linspace(0.0, 1.0, n_fi) if n_fi > 1 else 0.5
        else:
            n_fi = b - a
            phase_u[a:b] = np.linspace(0.0, 1.0, n_fi) if n_fi > 1 else 0.5

    return UtteranceContext(
        utt_id=utt_id,
        n_frames=n_frames,
        syllables=syllables,
        syl_idx_per_frame=syl_idx,
        is_onset=is_onset,
        phase_u=phase_u,
        speaker_f0_med_hz=float(speaker_f0_med_hz),
        continues_after=continues_after,
    )


@dataclass
class RuleProposal:
    """Masked additive modification (and optional voicing override) proposed by a rule module."""

    module_name: str
    mask: np.ndarray                               # (T,) bool
    delta_log_env: np.ndarray | None = None        # (T, 513) or (N_mask, 513)
    delta_bap: np.ndarray | None = None            # (T, 6) or (N_mask, 6)
    delta_tilt: np.ndarray | None = None           # (T,) or (N_mask,)
    delta_jitter: np.ndarray | None = None         # (T,) or (N_mask,)
    delta_shimmer: np.ndarray | None = None        # (T,) or (N_mask,)
    override_voiced: np.ndarray | None = None      # (T,) int8: -1 keep, 0 unvoiced, 1 voiced
    weight: float = 1.0
    delta_rd: np.ndarray | None = None             # (T,) or (N_mask,) LF shape rd delta


class RuleModule(Protocol):
    name: str

    def propose(self, ctx: UtteranceContext, base: LarynxControl) -> RuleProposal | None:
        ...


def _extract_masked(arr: np.ndarray | None, mask: np.ndarray, n_frames: int) -> np.ndarray | None:
    if arr is None:
        return None
    val = np.asarray(arr, dtype=np.float64)
    if val.shape[0] == n_frames:
        return val[mask]
    n_m = int(np.sum(mask))
    if val.shape[0] == n_m:
        return val
    raise ValueError(f"Delta leading dim {val.shape[0]} matches neither T={n_frames} nor sum(mask)={n_m}")


@dataclass
class Arbiter:
    """Fuses independent rule proposals onto a base LarynxControl with isolation & attribution."""

    log_env_bounds: tuple[float, float] = (-18.0, 6.0)
    bap_bounds: tuple[float, float] = (0.005, 0.995)
    tilt_bounds: tuple[float, float] = (-8.0, 8.0)
    jitter_bounds: tuple[float, float] = (0.0, 0.03)
    shimmer_bounds: tuple[float, float] = (0.0, 0.12)
    rd_bounds: tuple[float, float] = (0.50, 2.20)

    def apply(
        self,
        ctx: UtteranceContext,
        base: LarynxControl,
        modules: list[RuleModule],
    ) -> tuple[LarynxControl, dict[str, Any]]:
        out = base.copy()
        n_fr = len(out.f0_hz)
        if out.rd is None:
            out.rd = np.ones(n_fr, dtype=np.float64)
        module_traces: list[dict[str, Any]] = []

        for mod in modules:
            m_name = getattr(mod, "name", mod.__class__.__name__)
            rec: dict[str, Any] = {
                "module": m_name,
                "frames_touched": 0,
                "mean_abs_env_delta_db": 0.0,
                "mean_abs_bap_delta": 0.0,
                "mean_abs_tilt_delta": 0.0,
                "mean_abs_rd_delta": 0.0,
                "voiced_overrides": 0,
                "error": None,
            }
            try:
                prop = mod.propose(ctx, base)
                if prop is None:
                    module_traces.append(rec)
                    continue
                mask = np.asarray(prop.mask, dtype=bool).ravel()
                if mask.shape != (n_fr,):
                    raise ValueError(f"Proposal mask shape {mask.shape} != ({n_fr},)")
                n_m = int(np.sum(mask))
                if n_m == 0:
                    module_traces.append(rec)
                    continue

                w = float(prop.weight)
                d_env = _extract_masked(prop.delta_log_env, mask, n_fr)
                d_bap = _extract_masked(prop.delta_bap, mask, n_fr)
                d_tilt = _extract_masked(prop.delta_tilt, mask, n_fr)
                d_jit = _extract_masked(prop.delta_jitter, mask, n_fr)
                d_shim = _extract_masked(prop.delta_shimmer, mask, n_fr)
                d_rd = _extract_masked(prop.delta_rd, mask, n_fr)

                if d_env is not None:
                    if d_env.shape != (n_m, N_BINS):
                        raise ValueError(f"delta_log_env shape {d_env.shape} != ({n_m}, {N_BINS})")
                    delta = np.nan_to_num(w * d_env, nan=0.0, posinf=0.0, neginf=0.0)
                    out.log_env[mask] += delta
                    rec["mean_abs_env_delta_db"] = round(float(np.mean(np.abs(delta)) * NEPER_TO_DB), 4)

                if d_bap is not None:
                    if d_bap.shape != (n_m, N_BAP):
                        raise ValueError(f"delta_bap shape {d_bap.shape} != ({n_m}, {N_BAP})")
                    delta_b = np.nan_to_num(w * d_bap, nan=0.0, posinf=0.0, neginf=0.0)
                    out.bap[mask] += delta_b
                    rec["mean_abs_bap_delta"] = round(float(np.mean(np.abs(delta_b))), 4)

                if d_tilt is not None:
                    delta_t = np.nan_to_num(w * d_tilt.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
                    out.glottal_tilt_db_per_oct[mask] += delta_t
                    rec["mean_abs_tilt_delta"] = round(float(np.mean(np.abs(delta_t))), 4)

                if d_jit is not None:
                    out.jitter_frac[mask] += np.nan_to_num(w * d_jit.ravel(), nan=0.0, posinf=0.0, neginf=0.0)

                if d_shim is not None:
                    out.shimmer_frac[mask] += np.nan_to_num(w * d_shim.ravel(), nan=0.0, posinf=0.0, neginf=0.0)

                if d_rd is not None:
                    d_rd_flat = d_rd.ravel()
                    if d_rd_flat.shape != (n_m,):
                        raise ValueError(f"delta_rd shape {d_rd.shape} != ({n_m},)")
                    delta_r = np.nan_to_num(w * d_rd_flat, nan=0.0, posinf=0.0, neginf=0.0)
                    out.rd[mask] += delta_r
                    rec["mean_abs_rd_delta"] = round(float(np.mean(np.abs(delta_r))), 4)

                if prop.override_voiced is not None:
                    ov = np.asarray(prop.override_voiced, dtype=np.int8).ravel()
                    if ov.shape != (n_fr,):
                        raise ValueError(f"override_voiced shape {ov.shape} != ({n_fr},)")
                    force_uv = mask & (ov == 0)
                    force_v = mask & (ov == 1)
                    n_ov = int(np.sum(force_uv | force_v))
                    if np.any(force_uv):
                        out.voiced[force_uv] = False
                        out.f0_hz[force_uv] = 0.0
                    if np.any(force_v):
                        out.voiced[force_v] = True
                        need_f0 = force_v & (out.f0_hz <= 40.0)
                        if np.any(need_f0):
                            out.f0_hz[need_f0] = ctx.speaker_f0_med_hz
                    rec["voiced_overrides"] = n_ov

                rec["frames_touched"] = n_m
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
            module_traces.append(rec)

        # Enforce physical stability invariants across the entire fused trajectory
        out.log_env = np.clip(np.nan_to_num(out.log_env, nan=-12.0, posinf=6.0, neginf=-18.0), *self.log_env_bounds)
        out.bap = np.clip(np.nan_to_num(out.bap, nan=0.5, posinf=0.995, neginf=0.005), *self.bap_bounds)
        out.glottal_tilt_db_per_oct = np.clip(
            np.nan_to_num(out.glottal_tilt_db_per_oct, nan=0.0, posinf=8.0, neginf=-8.0),
            *self.tilt_bounds,
        )
        out.jitter_frac = np.clip(np.nan_to_num(out.jitter_frac, nan=0.0055, posinf=0.03, neginf=0.0), *self.jitter_bounds)
        out.shimmer_frac = np.clip(np.nan_to_num(out.shimmer_frac, nan=0.025, posinf=0.12, neginf=0.0), *self.shimmer_bounds)
        out.rd = np.clip(np.nan_to_num(out.rd, nan=1.0, posinf=self.rd_bounds[1], neginf=self.rd_bounds[0]), *self.rd_bounds)
        out.f0_hz = np.where(out.voiced & (out.f0_hz > 40.0), out.f0_hz, 0.0)
        out.voiced = out.voiced & (out.f0_hz > 40.0)

        return out, {"utt_id": ctx.utt_id, "n_frames": n_fr, "modules": module_traces}
