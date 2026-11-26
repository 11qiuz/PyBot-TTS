"""Rule 5 (Stage 5): Prosodic energy/tilt trajectory & low-F0 Tone-3 / phrase-final glottal relaxation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import rfftfreq

from exp_20_pybot_larynx import EXP_DIR, N_BAP, N_BINS, N_FFT, SR
from exp_20_pybot_larynx.arbiter import (
    NEPER_TO_DB,
    RuleProposal,
    UtteranceContext,
)
from exp_20_pybot_larynx.larynx.synth import LarynxControl

ENERGY_TABLES_PATH = EXP_DIR / "rules" / "prosodic_energy_tables.npz"
_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_SQRT_NBINS = math.sqrt(float(N_BINS))
_DB_TO_NEPER = 1.0 / NEPER_TO_DB
_OCT_FROM_1K = np.log2(np.clip(_FREQS, 120.0, 11500.0) / 1000.0)
_HI_TAPER = np.where(
    _FREQS > 8000.0,
    np.maximum(0.0, 1.0 - (_FREQS - 8000.0) / max(1.0, SR * 0.5 - 8000.0)),
    1.0,
)
_SLOPE_VEC = _OCT_FROM_1K * _HI_TAPER


def _bin_dur(dur_ms: float) -> int:
    if dur_ms < 145.0:
        return 0
    if dur_ms <= 215.0:
        return 1
    return 2


def _bin_f0(f0_st: float) -> int:
    if f0_st < -1.5:
        return 0
    if f0_st <= 1.5:
        return 1
    return 2


def build_prosodic_energy_tables(
    cache_path: Path | None = None,
    knowledge_path: Path | None = None,
    out_path: Path = ENERGY_TABLES_PATH,
) -> dict[str, Any]:
    """Mine empirical syllable cep[0] residual tables strictly from cached TRAIN tokens."""
    from collections import defaultdict
    from exp_20_pybot_larynx.acoustics.build_stage5_knowledge import DEFAULT_S5_CACHE_PATH
    from exp_20_pybot_larynx.acoustics.knowledge import get_splits, normalize_of
    from exp_20_pybot_larynx.acoustics.knowledge_stage2 import encode_next_place, encode_prev_fg
    from exp_20_pybot_larynx.acoustics.knowledge_stage5 import STAGE5_KNOWLEDGE_PATH, AcousticKnowledgeStage5
    from exp_20_pybot_larynx.arbiter import classify_final_group, classify_onset_place
    import pickle

    c_path = cache_path or DEFAULT_S5_CACHE_PATH
    k_path = knowledge_path or STAGE5_KNOWLEDGE_PATH
    with open(c_path, "rb") as f:
        data = pickle.load(f)
    kb5 = AcousticKnowledgeStage5.load(k_path)

    train_ids, dev_60, held_20 = get_splits()
    train_set = set(train_ids)
    assert not (train_set & set(dev_60)) and not (train_set & set(held_20))

    utt_recs = data["utt_prosody_records"]
    full_acc: dict[tuple[int, int, int, int, int], list[float]] = defaultdict(list)
    back_acc: dict[tuple[int, int, int], list[float]] = defaultdict(list)

    for uid in sorted(utt_recs.keys()):
        if uid not in train_set:
            continue
        rec = utt_recs[uid]
        syls = rec["syls"]
        n_s = len(syls)
        if n_s == 0:
            continue
        for si, row in enumerate(syls):
            _, _, raw_o, raw_f, t, dur_ms, _, _, f0_st, _, cep0_mean, pos_id, _, pause_ms = row
            o, f = normalize_of(str(raw_o), str(raw_f))
            t = int(t)
            pos_id = int(pos_id)
            prev_fg = "SIL" if (si == 0 or pos_id == 0) else classify_final_group(normalize_of(syls[si - 1][2], syls[si - 1][3])[1])
            next_pl = "SIL" if (si == n_s - 1 or pos_id == 2 or float(pause_ms) >= 25.0) else classify_onset_place(normalize_of(syls[si + 1][2], syls[si + 1][3])[0])
            prev_id = encode_prev_fg(prev_fg)
            next_id = encode_next_place(next_pl)

            f_ctx = kb5.lookup_final_coarse(o, f, t, prev_id, next_id, pos_id)
            if f_ctx is None:
                continue
            pred_c0 = float(np.mean(f_ctx[:, 0]))
            raw_dp = kb5.lookup_final_prosody_delta(o, f, t, pos_id, _bin_dur(float(dur_ms)), _bin_f0(float(f0_st)))
            if raw_dp is not None:
                dp0 = float(np.clip(np.mean(raw_dp[:, 0]), -3.8, 3.8))
                pred_c0 += 0.55 * dp0

            res_c0 = float(cep0_mean) - pred_c0
            prog_b = min(3, int(4 * si / max(1, n_s - 1))) if n_s > 1 else 1
            db = _bin_dur(float(dur_ms))
            fb = _bin_f0(float(f0_st))
            full_acc[(t, pos_id, db, fb, prog_b)].append(res_c0)
            back_acc[(t, pos_id, prog_b)].append(res_c0)

    def _shrink_pack(acc: dict[tuple[int, ...], list[float]], k_shrink: float) -> tuple[np.ndarray, np.ndarray]:
        keys = sorted(acc.keys())
        if not keys:
            return np.zeros((0, 0), dtype=np.int16), np.zeros(0, dtype=np.float32)
        k_arr = np.array(keys, dtype=np.int16)
        vals = []
        for k in keys:
            arr = acc[k]
            n = len(arr)
            shrink = n / (n + k_shrink)
            vals.append(float(np.clip(shrink * float(np.mean(arr)), -3.0, 3.0)))
        return k_arr, np.array(vals, dtype=np.float32)

    fk, fv = _shrink_pack(full_acc, k_shrink=25.0)
    bk, bv = _shrink_pack(back_acc, k_shrink=18.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        full_keys=fk,
        full_vals=fv,
        back_keys=bk,
        back_vals=bv,
        n_train_utts=np.array([len(utt_recs)], dtype=np.int32),
    )
    return {"full_entries": int(len(fk)), "back_entries": int(len(bk)), "path": str(out_path)}


class RuleProsodicEnergyTilt:
    """Applies TRAIN-mined prosodic energy/tilt residuals + Tone-3/phrase-final glottal relaxation."""

    name = "prosodic_energy_tilt"

    def __init__(self, table_path: Path = ENERGY_TABLES_PATH, energy_weight: float = 0.30) -> None:
        self.energy_weight = float(energy_weight)
        self._full_map: dict[tuple[int, int, int, int, int], float] = {}
        self._back_map: dict[tuple[int, int, int], float] = {}
        if Path(table_path).exists():
            with np.load(Path(table_path), allow_pickle=False) as d:
                for row, val in zip(d["full_keys"], d["full_vals"]):
                    self._full_map[tuple(int(x) for x in row)] = float(val)
                for row, val in zip(d["back_keys"], d["back_vals"]):
                    self._back_map[tuple(int(x) for x in row)] = float(val)

    def propose(self, ctx: UtteranceContext, base: LarynxControl) -> RuleProposal | None:
        n_fr = len(base.f0_hz)
        if n_fr == 0 or not ctx.syllables:
            return None

        f0 = np.asarray(base.f0_hz, dtype=np.float64)
        voiced = np.asarray(base.voiced, dtype=bool)
        v_mask = voiced & (f0 > 50.0)
        ref_f0 = max(50.0, float(ctx.speaker_f0_med_hz))

        mask = np.zeros(n_fr, dtype=bool)
        d_env = np.zeros((n_fr, N_BINS), dtype=np.float64)
        d_bap = np.zeros((n_fr, N_BAP), dtype=np.float64)
        d_tilt = np.zeros(n_fr, dtype=np.float64)
        d_jit = np.zeros(n_fr, dtype=np.float64)
        d_shim = np.zeros(n_fr, dtype=np.float64)
        d_rd = np.zeros(n_fr, dtype=np.float64)

        n_s = len(ctx.syllables)
        u_arr = np.asarray(ctx.phase_u, dtype=np.float64)
        is_on = np.asarray(ctx.is_onset, dtype=bool)

        for si, s in enumerate(ctx.syllables):
            a = max(0, min(n_fr, int(s.start_fr)))
            b = max(a, min(n_fr, int(s.end_fr)))
            sp = max(a, min(b, int(s.split_fr)))
            f_a = sp if (s.onset and sp > a) else a
            if b <= f_a:
                continue

            dur_ms = float(b - a) * 5.0
            v_sub = v_mask[a:b]
            f0_st = float(12.0 * np.log2(np.median(f0[a:b][v_sub]) / ref_f0)) if np.any(v_sub) else 0.0
            pos_id = 0 if si == 0 else (2 if (si == n_s - 1 or float(s.pause_after_ms) >= 25.0) else 1)
            prog_b = min(3, int(4 * si / max(1, n_s - 1))) if n_s > 1 else 1

            res_c0 = self._full_map.get((int(s.tone), pos_id, _bin_dur(dur_ms), _bin_f0(f0_st), prog_b))
            if res_c0 is None:
                res_c0 = self._back_map.get((int(s.tone), pos_id, prog_b), 0.0)

            # Apply smooth DCT-0 energy residual across voiced final frames
            fi_v = v_mask[f_a:b] & (~is_on[f_a:b])
            if abs(res_c0) > 1e-4 and np.any(fi_v):
                idx_v = np.arange(f_a, b)[fi_v]
                w_smooth = 0.65 + 0.35 * (np.sin(np.pi * np.clip(u_arr[idx_v], 0.02, 0.98)) ** 0.45)
                delta_nep = self.energy_weight * (res_c0 / _SQRT_NBINS) * w_smooth
                d_env[idx_v] += delta_nep[:, None]
                mask[idx_v] = True

            # Low-F0 Tone-3 & low phrase-tail glottal softening (surgical, avoids degrading non-dip tails)
            if np.any(v_mask[f_a:b]):
                idx_fv = np.arange(f_a, b)[v_mask[f_a:b]]
                rel_st = 12.0 * np.log2(f0[idx_fv] / ref_f0)
                t = int(s.tone)
                is_utt_end = (si == n_s - 1) or (int(s.prosodic_break) >= 4)

                creak_w = np.zeros(len(idx_fv), dtype=np.float64)
                if t == 3:
                    creak_w = np.maximum(creak_w, np.clip((-2.0 - rel_st) / 2.6, 0.0, 1.0))
                if is_utt_end:
                    tail_u = np.clip((u_arr[idx_fv] - 0.78) / 0.22, 0.0, 1.0)
                    low_w = np.clip((-1.5 - rel_st) / 2.5, 0.0, 1.0)
                    creak_w = np.maximum(creak_w, 0.65 * tail_u * low_w)

                hit = creak_w > 0.02
                if np.any(hit):
                    hi_idx = idx_fv[hit]
                    cw = creak_w[hit]
                    mask[hi_idx] = True
                    d_rd[hi_idx] += 0.12 * cw
                    d_jit[hi_idx] += 0.0006 * cw
                    d_shim[hi_idx] += 0.0015 * cw

        if not np.any(mask):
            return None

        return RuleProposal(
            module_name=self.name,
            mask=mask,
            delta_log_env=d_env,
            delta_bap=d_bap,
            delta_tilt=d_tilt,
            delta_jitter=d_jit,
            delta_shimmer=d_shim,
            delta_rd=d_rd,
        )
