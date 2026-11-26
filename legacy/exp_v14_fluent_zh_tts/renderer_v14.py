"""Multi-Sample Fluent Mandarin Renderer: Unbroken Contiguous Chunks & Minimal-Deformation F0 (`exp_v14_fluent_zh_tts`)."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from exp_v5_arch import crossling_frontend as cf
from exp_v6_vocoder import vocoder
from exp_v7_pybot_tts import crossling_warp as cw
from exp_v8_clarity.utau_concat import (
    N_BINS,
    OBSTRUENT_ONSETS,
    SILENCE_LOG_ENV,
    _blend_seam,
    _interp_list,
    _resample_1d,
    _resample_2d,
    _stretch_preserving_onset,
)
from exp_v10_integrity.gain_calibration import (
    calibrate_token_parseval_gain,
    classify_ipa_manner,
)
from exp_v11_align_gmm.timbre_cloner_v5 import adapt_features_v5
from exp_v11_diphone_tts.v11_renderer import (
    REF_VOWEL_PARSEVAL_DB,
    _adapt_v11,
    _get_src_prof,
    _resolve_profile_attrs,
)
from exp_v12_natural_vocoder.v12_renderer import _pick_glottal_bank
from exp_v12_natural_vocoder.vocoder_v12 import GlottalPulseBank, synthesize_v12
from exp_v12_prosody.prosody_model import ProsodyModelV12, apply_prosody_v12
from exp_v13_utau_ceiling.sandhi_v13 import apply_t3_sandhi_v13, plan_v13
from exp_v13_utau_ceiling.v13_deform import (
    _sample_by_float_indices_1d,
    _sample_by_float_indices_2d,
    calibrate_gain_gentle,
    smooth_utterance_gain_gentle,
)
from exp_v13_utau_ceiling.v13_renderer import V12_BASELINE_CFG, V13Config, _select_zh_tokens_v13
from exp_v14_fluent_zh_tts import HOP, N_FFT, SR
from exp_v14_fluent_zh_tts.bank_v14 import V14ZhBank, V14ZhToken
from exp_v14_fluent_zh_tts.prosody_v14 import ProsodyModelV14, apply_prosody_v14
from exp_v14_fluent_zh_tts.selector_v14 import fallback_onset_final_pair_v14, select_fluent_zh_sequence

ClonerFn = Callable[..., Any]


@dataclass
class V14Config:
    """Single-factor toggles for fluent multi-sample Mandarin TTS (`A0..A6`)."""

    use_sandhi_v13: bool = True
    prosody_mode: str = "v14_hgbr"          # "v12_ridge" | "v14_hgbr"
    selector_mode: str = "v14"              # "v12" | "v13" | "v14"
    contiguous_chunk_extract: bool = True
    preserve_natural_f0: bool = True
    chunk_dur_damp: float = 0.45
    syl_dur_damp: float = 0.55
    gentle_gain: bool = True
    exact_guided_timing: bool = True
    cos_seam_f0_pull: bool = True


V14_BEST_CFG = V14Config()


def _interpolate_f0_hz_onto_mask(
    f0_hz_src: np.ndarray,
    tok_voiced: np.ndarray,
    fallback_f0: float,
    voiced_hint: np.ndarray | None = None,
) -> np.ndarray:
    n = len(tok_voiced)
    out = np.zeros(n, dtype=np.float64)
    src = np.asarray(f0_hz_src, dtype=np.float64).ravel()
    if src.size == 0:
        out[tok_voiced] = fallback_f0
        return out
    if voiced_hint is not None and len(voiced_hint) == len(src):
        v_mask = np.asarray(voiced_hint, dtype=bool) & (src > 40.0)
    else:
        v_mask = src > 40.0
    v_idx = np.where(v_mask)[0]
    if v_idx.size == 0:
        out[tok_voiced] = fallback_f0
        return out
    t_src = np.linspace(0.0, 1.0, len(src))[v_idx]
    t_dst = np.linspace(0.0, 1.0, n)
    interp_hz = 2.0 ** np.interp(t_dst, t_src, np.log2(src[v_idx]))
    out[tok_voiced] = np.clip(interp_hz[tok_voiced], 60.0, 460.0)
    return out


def group_contiguous_chunks(
    chosen_syls: list[tuple[str, V14ZhToken | None]],
    unit_pause_before: list[bool],
    *,
    contiguous_chunk_extract: bool = True,
) -> list[list[int]]:
    """Group maximal unbroken same-utterance consecutive token runs."""
    n_u = len(chosen_syls)
    groups: list[list[int]] = []
    i = 0
    while i < n_u:
        grp = [i]
        tok_prev = chosen_syls[i][1]
        j = i + 1
        while contiguous_chunk_extract and j < n_u and tok_prev is not None:
            tok_cur = chosen_syls[j][1]
            if (
                tok_cur is not None
                and (not unit_pause_before[j])
                and tok_cur.utterance_id == tok_prev.utterance_id
                and tok_cur.syl_idx == tok_prev.syl_idx + 1
                and tok_prev.pause_after_ms < 20.0
                and abs(tok_cur.syl_start - tok_prev.syl_end) <= 4
            ):
                grp.append(j)
                tok_prev = tok_cur
                j += 1
            else:
                break
        groups.append(grp)
        i = grp[-1] + 1
    return groups


def _template_contour_for_unit(
    u: dict[str, Any],
    n_fr: int,
    iqr_scale: float,
    decl: float,
) -> np.ndarray:
    f3 = u.get("f0_3pt_st")
    if isinstance(f3, (list, tuple)) and len(f3) == 3:
        t_dst = np.linspace(0.0, 1.0, max(1, n_fr))
        pts = np.interp(t_dst, [0.15, 0.50, 0.85], [float(f3[0]), float(f3[1]), float(f3[2])])
        return pts * min(1.35, max(0.85, iqr_scale))
    p5 = [float(x) for x in u.get("pitch_st", [0.0] * 5)]
    t_grid = np.linspace(0.0, 1.0, max(1, n_fr))
    return np.array([_interp_list(p5, float(t)) for t in t_grid], dtype=np.float64) * iqr_scale + decl


def _safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 4 or float(np.std(a)) < 1e-4 or float(np.std(b)) < 1e-4:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def render_v14_utterance(
    text_or_plan: str | dict[str, Any],
    bank: V14ZhBank,
    target_profile: Any,
    *,
    guided_controls: list[dict[str, Any]] | None = None,
    cfg: V14Config | None = None,
    prosody_model_v14: ProsodyModelV14 | None = None,
    prosody_model_v12: ProsodyModelV12 | None = None,
    cloner_fn: ClonerFn | None = None,
    src_profiles: dict[str, Any] | None = None,
    vtln_alpha: float | None = None,
    glottal_bank: GlottalPulseBank | None = None,
    use_unvoiced_residual: bool = True,
    ref_vowel_db: float = REF_VOWEL_PARSEVAL_DB,
    seed: int = 72,
    phase_dispersion: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render Mandarin utterance using multi-sample selection, contiguous chunks & minimal-deform F0."""
    if cfg is None:
        cfg = V14Config()
    if cloner_fn is None:
        cloner_fn = adapt_features_v5

    if isinstance(text_or_plan, str):
        if cfg.use_sandhi_v13:
            plan_doc = plan_v13(text_or_plan)
        else:
            from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10

            plan_doc = plan_v10(text_or_plan)
    else:
        plan_doc = (
            apply_t3_sandhi_v13(text_or_plan)
            if (cfg.use_sandhi_v13 and guided_controls is None)
            else text_or_plan
        )

    prosody_applied = False
    if guided_controls is None:
        if cfg.prosody_mode == "v14_hgbr":
            plan_doc = apply_prosody_v14(plan_doc, model=prosody_model_v14)
            prosody_applied = True
        else:
            try:
                plan_doc = apply_prosody_v12(plan_doc, model=prosody_model_v12)
                prosody_applied = True
            except Exception:
                prosody_applied = False

    units = plan_doc.get("units") or []
    if not units:
        raise ValueError("plan_doc has no units")
    if guided_controls is not None and len(guided_controls) != len(units):
        raise ValueError(f"guided_controls length ({len(guided_controls)}) != units ({len(units)})")

    tgt_name = getattr(target_profile, "name", "")
    tgt_f0, tgt_iqr = _resolve_profile_attrs(target_profile)
    iqr_scale = max(0.55, min(1.75, tgt_iqr / 3.2))
    n_u = len(units)

    parsed: list[tuple[str, str, int]] = []
    for idx, u in enumerate(units):
        if u.get("lang", "zh") != "zh":
            raise ValueError("render_v14_utterance targets Mandarin units")
        gc = guided_controls[idx] if guided_controls is not None else {}
        o = str(gc.get("onset", u.get("onset", "")))
        f = str(gc.get("final", u.get("final", "")))
        if o in ("y", "w"):
            o = ""
        if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
            f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"
        tone = int(gc.get("tone", u.get("tone") or 1))
        parsed.append((o, f, tone))

    group_id = [0] * n_u
    cur_g = 0
    for i, u in enumerate(units):
        group_id[i] = cur_g
        p_ms = float(
            guided_controls[i].get("pause_after_ms", u.get("pause_after_ms", 0.0))
            if guided_controls is not None
            else u.get("pause_after_ms", 0.0)
        )
        if p_ms > 55.0:
            cur_g += 1
    group_spans: dict[int, list[int]] = {}
    for i, g in enumerate(group_id):
        if g not in group_spans:
            group_spans[g] = [i, i]
        else:
            group_spans[g][1] = i

    unit_dur_ms: list[float] = []
    unit_decl_st: list[float] = []
    unit_gain_db: list[float] = []
    unit_pause_after: list[float] = []
    unit_pause_before: list[bool] = []
    unit_f0_3pt: list[tuple[float, float, float] | None] = []
    unit_f0_mid_st: list[float | None] = []
    unit_target_onset_ms: list[float | None] = []
    unit_pos: list[str] = []

    for i, u in enumerate(units):
        gc = guided_controls[i] if guided_controls is not None else {}
        g0, g1 = group_spans[group_id[i]]
        g_len = max(1, g1 - g0)
        rel_pos = (i - g0) / max(1, g_len)

        if "dur_ms" in gc:
            base_d = max(15.0, float(gc["dur_ms"]))
        elif "duration_ms" in u:
            base_d = max(55.0, float(u["duration_ms"]))
        else:
            base_d = 200.0

        if "gain_db" in gc:
            gdb = float(gc["gain_db"])
        elif "gain_db" in u:
            gdb = float(u["gain_db"])
        else:
            gdb = 0.25 * (1.0 - rel_pos) - (0.8 if i == g1 else 0.0)

        decl = 0.0 if ("f0_3pt_st" in u) else ((0.25 - 0.50 * rel_pos) if ("gain_db" in u and "duration_ms" in u) else (0.75 - 1.50 * rel_pos))
        p_after = float(gc.get("pause_after_ms", u.get("pause_after_ms", 0.0))) if guided_controls is not None else float(u.get("pause_after_ms", 0.0))
        prev_p = unit_pause_after[i - 1] if i > 0 else 0.0

        f3 = None
        f0_mid = None
        if "f0_hz" in gc:
            arr_f = np.asarray(gc["f0_hz"], dtype=np.float64).ravel()
            v_f = arr_f[arr_f > 40.0]
            if v_f.size > 0:
                sts_f = 12.0 * np.log2(v_f / max(50.0, tgt_f0))
                f0_mid = float(np.median(sts_f))
                q3 = np.interp([0.2, 0.5, 0.8], np.linspace(0.0, 1.0, len(sts_f)), sts_f)
                f3 = (float(q3[0]), float(q3[1]), float(q3[2]))
        elif isinstance(u.get("f0_3pt_st"), (list, tuple)) and len(u["f0_3pt_st"]) == 3:
            pts = [float(x) for x in u["f0_3pt_st"]]
            f3 = (pts[0], pts[1], pts[2])
            f0_mid = pts[1]
        elif u.get("pitch_st"):
            f0_mid = float(np.median(u["pitch_st"])) * iqr_scale + decl

        unit_dur_ms.append(base_d)
        unit_decl_st.append(decl)
        unit_gain_db.append(gdb)
        unit_pause_after.append(p_after)
        unit_pause_before.append(i > 0 and prev_p > 15.0)
        unit_f0_3pt.append(f3)
        unit_f0_mid_st.append(f0_mid)
        unit_target_onset_ms.append(float(gc["onset_ms"]) if "onset_ms" in gc else None)
        unit_pos.append("initial" if i == g0 else ("final" if i == g1 else "medial"))

    # Unit selection dispatch across v12 / v13 / v14
    if cfg.selector_mode == "v14":
        chosen_syls = select_fluent_zh_sequence(
            parsed,
            unit_dur_ms,
            unit_pause_before,
            unit_f0_3pt,
            unit_f0_mid_st,
            unit_pos,
            bank,
        )
    elif cfg.selector_mode == "v12":
        chosen_syls = _select_zh_tokens_v13(
            parsed,
            unit_dur_ms,
            unit_pause_before,
            unit_f0_mid_st,
            unit_target_onset_ms,
            unit_pos,
            bank,  # compatible V13ZhToken fields
            tgt_name or "SSB0502",
            V12_BASELINE_CFG,
        )
    else:
        v13_sel_cfg = V13Config(
            fix_viterbi_join=True,
            soft_context_and_f0_sel=True,
            joint_of_concat=True,
            gentle_gain=cfg.gentle_gain,
            steady_nucleus_stretch=False,
        )
        chosen_syls = _select_zh_tokens_v13(
            parsed,
            unit_dur_ms,
            unit_pause_before,
            unit_f0_mid_st,
            unit_target_onset_ms,
            unit_pos,
            bank,
            tgt_name or "SSB0502",
            v13_sel_cfg,
        )

    raw_nf = [max(3 if guided_controls is not None else 12, int(round(unit_dur_ms[i] / 5.0))) for i in range(n_u)]
    groups = group_contiguous_chunks(
        chosen_syls,
        unit_pause_before,
        contiguous_chunk_extract=cfg.contiguous_chunk_extract,
    )

    envs, baps, bgt8s, vois = [], [], [], []
    is_obs_blk, is_pause_blk = [], []
    explicit_f0_hz_blocks: list[np.ndarray | None] = []
    sts_rel_blocks: list[np.ndarray] = []
    residual_spans: list[tuple[int, int, np.ndarray]] = []
    source_counts: dict[str, int] = {}
    slot_timings_ms: list[list[float]] = [[] for _ in range(n_u)]
    unit_boundaries_ms: list[float] = []
    boundary_frames: list[int] = []
    selected_trace: list[dict[str, Any]] = [{} for _ in range(n_u)]
    f0_corrs: list[float] = []

    lead_fr = 4
    envs.append(np.full((lead_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((lead_fr, 5), dtype=np.float64))
    bgt8s.append(np.ones((lead_fr, 2), dtype=np.float64))
    vois.append(np.zeros(lead_fr, dtype=bool))
    explicit_f0_hz_blocks.append(np.zeros(lead_fr, dtype=np.float64))
    sts_rel_blocks.append(np.zeros(lead_fr, dtype=np.float64))
    is_obs_blk.append(False)
    is_pause_blk.append(True)
    cur_fr = lead_fr

    same_utt_chain_hits = 0
    same_utt_chain_possible = 0
    seam_blends_applied = 0
    prev_sel_tok: V14ZhToken | None = None
    for idx in range(n_u):
        _, tk = chosen_syls[idx]
        if idx > 0 and not unit_pause_before[idx]:
            same_utt_chain_possible += 1
            if (
                tk is not None
                and prev_sel_tok is not None
                and tk.utterance_id == prev_sel_tok.utterance_id
                and tk.syl_idx == prev_sel_tok.syl_idx + 1
                and prev_sel_tok.pause_after_ms < 20.0
            ):
                same_utt_chain_hits += 1
        prev_sel_tok = tk

    zh_fb_prof = _get_src_prof(bank, "SSB0502", target_profile, src_profiles) or (
        next(iter(bank.profiles_v2.values())) if bank.profiles_v2 else target_profile
    )

    for grp in groups:
        if len(grp) >= 2 and all(chosen_syls[i][1] is not None for i in grp):
            # --- CONTIGUOUS UNBROKEN MULTI-SYLLABLE CHUNK ---
            toks = [chosen_syls[i][1] for i in grp]
            tok0 = toks[0]
            assert tok0 is not None
            uid_src = tok0.utterance_id
            ue = bank.utt_feats[uid_src]
            s_lo = int(tok0.syl_start)
            s_hi = int(toks[-1].syl_end)

            e_raw = ue["env_norm"][s_lo:s_hi].astype(np.float64)
            b_raw = ue["bap"][s_lo:s_hi].astype(np.float64)
            bg_raw = ue["bap_gt8k"][s_lo:s_hi].astype(np.float64)
            v_raw = ue["voiced"][s_lo:s_hi].copy()
            f0_raw = ue["f0_rel_st"][s_lo:s_hi].astype(np.float64)
            for tk in toks:
                assert tk is not None
                rl = max(0, tk.syl_start - s_lo)
                rh = min(len(v_raw), tk.syl_end - s_lo)
                if rh > rl and len(tk.voiced) >= rh - rl:
                    v_raw[rl:rh] = tk.voiced[: rh - rl]

            src_p = _get_src_prof(bank, tok0.speaker, target_profile, src_profiles)
            e_raw, b_raw, bg_raw, v_raw = _adapt_v11(
                e_raw, b_raw, bg_raw, v_raw, f0_raw, src_p, target_profile, cloner_fn, vtln_alpha,
            )
            if cfg.gentle_gain:
                e_raw, _ = calibrate_gain_gentle(e_raw, "zh_syllable", ref_vowel_db)
            else:
                e_raw, _ = calibrate_token_parseval_gain(e_raw, "zh_syllable", ref_vowel_db, max_corr_db=10.5)

            orig_len = len(e_raw)
            syl_out_spans: list[tuple[int, int]] = []
            if guided_controls is not None and cfg.exact_guided_timing:
                # Continuous piecewise-linear warp inside unbroken chunk -> exact dur error = 0 ms, zero internal seams!
                pos_parts: list[np.ndarray] = []
                voi_parts: list[np.ndarray] = []
                out_cursor = 0
                for tk, idx_u in zip(toks, grp):
                    assert tk is not None
                    loc_s = max(0, tk.syl_start - s_lo)
                    loc_e = min(orig_len, max(loc_s + 1, tk.syl_end - s_lo))
                    nf_k = raw_nf[idx_u]
                    if loc_e - loc_s == 1:
                        p_k = np.full(nf_k, float(loc_s), dtype=np.float64)
                    else:
                        p_k = np.linspace(float(loc_s), float(loc_e - 1), nf_k)
                    pos_parts.append(p_k)
                    voi_parts.append(v_raw[np.clip(np.round(p_k).astype(int), 0, orig_len - 1)])
                    syl_out_spans.append((out_cursor, out_cursor + nf_k))
                    if use_unvoiced_residual and tk.onset_wav is not None and tk.f_split > 0:
                        fsp_k = max(1, min(nf_k - 1, int(round(tk.f_split * (nf_k / max(1, loc_e - loc_s))))))
                        residual_spans.append((cur_fr + out_cursor, cur_fr + out_cursor + fsp_k, tk.onset_wav))
                    out_cursor += nf_k
                pos_all = np.concatenate(pos_parts)
                e_st = _sample_by_float_indices_2d(e_raw, pos_all)
                b_st = np.clip(_sample_by_float_indices_2d(b_raw, pos_all), 0.005, 0.995)
                bg_st = np.clip(_sample_by_float_indices_2d(bg_raw, pos_all), 0.005, 0.995)
                v_st = np.concatenate(voi_parts).astype(bool)
                f0_tok = _sample_by_float_indices_1d(f0_raw, pos_all)
            else:
                tgt_sum = sum(raw_nf[i] for i in grp)
                raw_ratio = tgt_sum / max(1.0, float(orig_len))
                if cfg.chunk_dur_damp < 0.999:
                    eff_ratio = float(np.clip(1.0 + cfg.chunk_dur_damp * (raw_ratio - 1.0), 0.85, 1.18))
                else:
                    eff_ratio = float(np.clip(raw_ratio, 0.55, 2.0))
                nf_chunk = max(6, int(round(orig_len * eff_ratio)))
                e_st, b_st, v_st, f_sp0 = _stretch_preserving_onset(
                    e_raw, b_raw, v_raw, tok0.f_split, nf_chunk, preserve_onset=True,
                )
                bg_st = np.clip(_resample_2d(bg_raw, nf_chunk), 0.005, 0.995)
                f0_tok = _resample_1d(f0_raw, nf_chunk)

                # Coordinate map orig frame -> warped chunk frame preserving tok0's onset split
                def _map_fr(orig_fr: int) -> int:
                    if orig_fr <= tok0.f_split:
                        return int(round(orig_fr * (f_sp0 / max(1, tok0.f_split)))) if tok0.f_split > 0 else 0
                    rem_src = max(1, orig_len - tok0.f_split)
                    rem_dst = max(1, nf_chunk - f_sp0)
                    return min(nf_chunk, f_sp0 + int(round((orig_fr - tok0.f_split) * rem_dst / rem_src)))

                for pos_c, (tk, idx_u) in enumerate(zip(toks, grp)):
                    assert tk is not None
                    os0 = _map_fr(tk.syl_start - s_lo)
                    os1 = nf_chunk if pos_c == len(grp) - 1 else max(os0 + 1, _map_fr(tk.syl_end - s_lo))
                    syl_out_spans.append((os0, os1))
                    if use_unvoiced_residual and tk.onset_wav is not None and tk.f_split > 0:
                        ron0 = os0
                        ron1 = max(ron0 + 1, _map_fr((tk.syl_start - s_lo) + tk.f_split))
                        residual_spans.append((cur_fr + ron0, cur_fr + ron1, tk.onset_wav))

            # Chunk gain contour (preserving relative speech gain smoothly inside unbroken chunk)
            for idx_u, (os0, os1) in zip(grp, syl_out_spans):
                g_u = float(np.clip(unit_gain_db[idx_u], -12.0 if guided_controls is not None else -2.5, 12.0 if guided_controls is not None else 2.5))
                e_st[os0:os1] = e_st[os0:os1] + g_u * math.log(10.0) / 20.0

            # F0 track for contiguous chunk
            if guided_controls is not None:
                hz_chunk = np.zeros(len(e_st), dtype=np.float64)
                for idx_u, (os0, os1) in zip(grp, syl_out_spans):
                    gc = guided_controls[idx_u]
                    if "f0_hz" in gc:
                        hz_chunk[os0:os1] = _interpolate_f0_hz_onto_mask(
                            np.asarray(gc["f0_hz"], dtype=np.float64), v_st[os0:os1], tgt_f0, voiced_hint=gc.get("voiced_hint"),
                        )
                    elif "f0_st_contour" in gc:
                        stc = _resample_1d(np.asarray(gc["f0_st_contour"], dtype=np.float64), os1 - os0)
                        sub_h = np.zeros(os1 - os0, dtype=np.float64)
                        sub_h[v_st[os0:os1]] = np.clip(tgt_f0 * (2.0 ** (stc[v_st[os0:os1]] / 12.0)), 60.0, 460.0)
                        hz_chunk[os0:os1] = sub_h
                explicit_f0_hz_blocks.append(hz_chunk)
                sts_rel_blocks.append(np.zeros(len(e_st), dtype=np.float64))
            else:
                st_chunk = np.zeros(len(e_st), dtype=np.float64)
                tgt_parts = [
                    _template_contour_for_unit(units[idx_u], os1 - os0, iqr_scale, unit_decl_st[idx_u])
                    for idx_u, (os0, os1) in zip(grp, syl_out_spans)
                ]
                tgt_all_c = np.concatenate(tgt_parts)
                all_match = all(int(tk.tone) == int(parsed[idx_u][2]) for tk, idx_u in zip(toks, grp) if tk is not None)
                if cfg.preserve_natural_f0 and all_match:
                    mid_frames = []
                    deltas = []
                    for (os0, os1), tgt_k in zip(syl_out_spans, tgt_parts):
                        vm = v_st[os0:os1]
                        tok_m = float(np.median(f0_tok[os0:os1][vm])) if np.any(vm) else float(np.median(f0_tok[os0:os1]))
                        tgt_m = float(np.median(tgt_k[vm])) if np.any(vm) else float(np.median(tgt_k))
                        mid_frames.append(0.5 * (os0 + max(os0 + 1, os1 - 1)))
                        deltas.append(float(np.clip(0.70 * (tgt_m - tok_m), -3.2, 3.2)))
                    delta_curve = np.interp(np.arange(len(e_st), dtype=np.float64), mid_frames, deltas)
                    st_chunk = f0_tok + delta_curve
                    vm_all = v_st
                    if int(np.sum(vm_all)) >= 4:
                        cr = _safe_corr_1d(f0_tok[vm_all], st_chunk[vm_all])
                        if cr is not None:
                            f0_corrs.append(cr)
                elif cfg.preserve_natural_f0:
                    for tk, idx_u, (os0, os1), tgt_k in zip(toks, grp, syl_out_spans, tgt_parts):
                        vm = v_st[os0:os1]
                        sub_f0 = f0_tok[os0:os1]
                        tok_m = float(np.median(sub_f0[vm])) if np.any(vm) else float(np.median(sub_f0))
                        tgt_m = float(np.median(tgt_k[vm])) if np.any(vm) else float(np.median(tgt_k))
                        delta_k = float(np.clip(0.70 * (tgt_m - tok_m), -3.2, 3.2))
                        if tk is not None and int(tk.tone) == int(parsed[idx_u][2]):
                            st_chunk[os0:os1] = sub_f0 + delta_k
                            if int(np.sum(vm)) >= 4:
                                cr = _safe_corr_1d(sub_f0[vm], st_chunk[os0:os1][vm])
                                if cr is not None:
                                    f0_corrs.append(cr)
                        else:
                            st_chunk[os0:os1] = 0.78 * tgt_k + 0.22 * (sub_f0 - tok_m + tgt_m)
                else:
                    f0_mic = np.clip(f0_tok - float(np.median(f0_tok)), -3.0, 3.0)
                    st_chunk = 0.68 * tgt_all_c + 0.32 * f0_mic
                explicit_f0_hz_blocks.append(None)
                sts_rel_blocks.append(st_chunk)

            envs.append(e_st); baps.append(b_st); bgt8s.append(bg_st); vois.append(v_st)
            is_obs_blk.append(parsed[grp[0]][0] in OBSTRUENT_ONSETS)
            is_pause_blk.append(False)

            for pos_c, (tk, idx_u, (os0, os1)) in enumerate(zip(toks, grp, syl_out_spans)):
                assert tk is not None
                source_counts["zh_target_chunk"] = source_counts.get("zh_target_chunk", 0) + 1
                slot_timings_ms[idx_u] = [(cur_fr + os0) * 5.0, (cur_fr + os1) * 5.0]
                unit_boundaries_ms.append((cur_fr + os1) * 5.0)
                boundary_frames.append(cur_fr + os1)
                selected_trace[idx_u] = {
                    "idx": idx_u,
                    "hanzi": units[idx_u].get("hanzi", ""),
                    "pinyin": f"{parsed[idx_u][0]}{parsed[idx_u][1]}{parsed[idx_u][2]}",
                    "mode": "zh_target_chunk",
                    "chunk_len": len(grp),
                    "chunk_pos": pos_c,
                    "src_utt": tk.utterance_id,
                    "src_tone": int(tk.tone),
                    "src_syl_idx": int(tk.syl_idx),
                    "tgt_dur_ms": round(unit_dur_ms[idx_u], 1),
                    "actual_dur_ms": round((os1 - os0) * 5.0, 1),
                }
            cur_fr += len(e_st)

            # Emit clause pause if last unit in chunk carries punctuation/hesitation pause
            p_ms_tail = unit_pause_after[grp[-1]]
            if p_ms_tail > 15.0:
                p_fr = max(2, int(round(p_ms_tail / 5.0)))
                envs.append(np.full((p_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
                baps.append(np.ones((p_fr, 5), dtype=np.float64))
                bgt8s.append(np.ones((p_fr, 2), dtype=np.float64))
                vois.append(np.zeros(p_fr, dtype=bool))
                explicit_f0_hz_blocks.append(np.zeros(p_fr, dtype=np.float64))
                sts_rel_blocks.append(np.zeros(p_fr, dtype=np.float64))
                is_obs_blk.append(False); is_pause_blk.append(True)
                cur_fr += p_fr
        else:
            # --- STANDALONE SYLLABLE OR O+F FALLBACK ---
            idx = grp[0]
            o, f, tone = parsed[idx]
            u = units[idx]
            gc = guided_controls[idx] if guided_controls is not None else {}
            mode, best = chosen_syls[idx]
            prev_f = parsed[idx - 1][1] if idx > 0 else "SIL"
            next_o = parsed[idx + 1][0] if idx + 1 < n_u else "SIL"

            if best is not None:
                src_p = _get_src_prof(bank, best.speaker, target_profile, src_profiles)
                e_raw, b_raw, bg_raw, v_raw = _adapt_v11(
                    best.log_env.astype(np.float64),
                    best.bap.astype(np.float64),
                    best.bap_gt8k.astype(np.float64),
                    best.voiced.copy(),
                    best.f0_rel_st.astype(np.float64),
                    src_p,
                    target_profile,
                    cloner_fn,
                    vtln_alpha,
                )
                if cfg.gentle_gain:
                    e_raw, _ = calibrate_gain_gentle(e_raw, "zh_syllable", ref_vowel_db)
                else:
                    e_raw, _ = calibrate_token_parseval_gain(e_raw, "zh_syllable", ref_vowel_db, max_corr_db=10.5)

                if guided_controls is not None:
                    nf = raw_nf[idx]
                else:
                    orig_n = len(e_raw)
                    raw_r = raw_nf[idx] / max(1.0, float(orig_n))
                    if cfg.syl_dur_damp < 0.999:
                        eff_r = float(np.clip(1.0 + cfg.syl_dur_damp * (raw_r - 1.0), 0.78, 1.28))
                    else:
                        eff_r = float(np.clip(raw_r, 0.55, 2.0))
                    nf = max(12, int(round(orig_n * eff_r)))

                e_st, b_st, v_st, f_sp_new = _stretch_preserving_onset(
                    e_raw, b_raw, v_raw, best.f_split, nf, preserve_onset=True,
                )
                bg_st = np.clip(_resample_2d(bg_raw, nf), 0.005, 0.995)
                f0_tok = _resample_1d(best.f0_rel_st.astype(np.float64), nf)

                if use_unvoiced_residual and best.onset_wav is not None and f_sp_new > 0:
                    residual_spans.append((cur_fr, cur_fr + f_sp_new, best.onset_wav))
                source_counts[mode] = source_counts.get(mode, 0) + 1
                matched_tone = (int(best.tone) == int(tone))
                selected_trace[idx] = {
                    "idx": idx,
                    "hanzi": u.get("hanzi", ""),
                    "pinyin": f"{o}{f}{tone}",
                    "mode": mode,
                    "chunk_len": 1,
                    "src_utt": best.utterance_id,
                    "src_tone": int(best.tone),
                    "src_syl_idx": int(best.syl_idx),
                    "tgt_dur_ms": round(unit_dur_ms[idx], 1),
                    "actual_dur_ms": round(nf * 5.0, 1),
                }
            else:
                parts_e, parts_b, parts_bg, parts_v, parts_f0 = [], [], [], [], []
                split_pt = 0
                bo_joint, bf_joint = fallback_onset_final_pair_v14(
                    bank,
                    onset=o,
                    final=f,
                    tone=tone,
                    prev_final=prev_f,
                    next_onset=next_o,
                    target_dur_ms=unit_dur_ms[idx],
                )
                bo_sel = bo_joint
                if o:
                    if bo_sel is not None:
                        is_uv_o = o in ("f", "s", "sh", "x", "h", "c", "ch", "q")
                        eo, bo_b, bgo, vo = _adapt_v11(
                            bo_sel.log_env.astype(np.float64), bo_sel.bap.astype(np.float64),
                            bo_sel.bap_gt8k.astype(np.float64), bo_sel.voiced.copy(), bo_sel.f0_rel_st.astype(np.float64),
                            _get_src_prof(bank, bo_sel.speaker, target_profile, src_profiles), target_profile, cloner_fn, vtln_alpha,
                        )
                        if cfg.gentle_gain:
                            eo, _ = calibrate_gain_gentle(eo, "voiceless_fricative" if is_uv_o else "nasal", ref_vowel_db, max_corr_db=5.0)
                        else:
                            eo, _ = calibrate_token_parseval_gain(eo, "voiceless_fricative" if is_uv_o else "nasal", ref_vowel_db, max_corr_db=8.5)
                        parts_e.append(eo); parts_b.append(bo_b); parts_bg.append(bgo); parts_v.append(vo); parts_f0.append(bo_sel.f0_rel_st)
                        split_pt = len(eo)
                    elif bank.fallback_unit_bank is not None:
                        for oi in cf.PINYIN_ONSET_TO_IPA.get(o, ()):
                            d = cw.lookup_ipa_template(bank.fallback_unit_bank, oi, is_onset=True)
                            bg_d = np.repeat(d["bap"][:, 4:5], 2, axis=1)
                            ed, bd, bgd, vd = _adapt_v11(
                                d["log_env"] - 3.1, d["bap"], bg_d, d["voiced"], np.zeros(len(d["log_env"])),
                                zh_fb_prof, target_profile, cloner_fn, vtln_alpha,
                            )
                            ed, _ = calibrate_token_parseval_gain(ed, classify_ipa_manner(oi, None), ref_vowel_db, max_corr_db=8.5)
                            parts_e.append(ed); parts_b.append(bd); parts_bg.append(bgd); parts_v.append(vd); parts_f0.append(np.zeros(len(ed)))
                        split_pt = sum(len(x) for x in parts_e)

                if bf_joint is not None:
                    ef, bf_b, bgf, vf = _adapt_v11(
                        bf_joint.log_env.astype(np.float64), bf_joint.bap.astype(np.float64),
                        bf_joint.bap_gt8k.astype(np.float64), bf_joint.voiced.copy(), bf_joint.f0_rel_st.astype(np.float64),
                        _get_src_prof(bank, bf_joint.speaker, target_profile, src_profiles), target_profile, cloner_fn, vtln_alpha,
                    )
                    if cfg.gentle_gain:
                        ef, _ = calibrate_gain_gentle(ef, "zh_syllable", ref_vowel_db)
                    else:
                        ef, _ = calibrate_token_parseval_gain(ef, "zh_syllable", ref_vowel_db, max_corr_db=10.5)
                    parts_e.append(ef); parts_b.append(bf_b); parts_bg.append(bgf); parts_v.append(vf); parts_f0.append(bf_joint.f0_rel_st)
                    source_counts["zh_concat_of"] = source_counts.get("zh_concat_of", 0) + 1
                    selected_trace[idx] = {
                        "idx": idx,
                        "hanzi": u.get("hanzi", ""),
                        "pinyin": f"{o}{f}{tone}",
                        "mode": "zh_concat_of",
                        "chunk_len": 1,
                        "onset_utt": bo_sel.utterance_id if bo_sel else None,
                        "final_utt": bf_joint.utterance_id,
                    }
                elif bank.fallback_unit_bank is not None:
                    for fi in cf.PINYIN_FINAL_TO_IPA.get(f, ("ə",)):
                        d = cw.lookup_ipa_template(bank.fallback_unit_bank, fi, is_onset=False)
                        bg_d = np.repeat(d["bap"][:, 4:5], 2, axis=1)
                        ed, bd, bgd, vd = _adapt_v11(
                            d["log_env"] - 3.1, d["bap"], bg_d, d["voiced"], np.zeros(len(d["log_env"])),
                            zh_fb_prof, target_profile, cloner_fn, vtln_alpha,
                        )
                        ed, _ = calibrate_token_parseval_gain(ed, "zh_syllable", ref_vowel_db, max_corr_db=10.5)
                        parts_e.append(ed); parts_b.append(bd); parts_bg.append(bgd); parts_v.append(vd); parts_f0.append(np.zeros(len(ed)))
                    source_counts["zh_ipa_fill"] = source_counts.get("zh_ipa_fill", 0) + 1
                    selected_trace[idx] = {
                        "idx": idx, "hanzi": u.get("hanzi", ""), "pinyin": f"{o}{f}{tone}", "mode": "zh_ipa_fill", "chunk_len": 1,
                    }

                cat_e = np.concatenate(parts_e, axis=0)
                cat_b = np.concatenate(parts_b, axis=0)
                cat_bg = np.concatenate(parts_bg, axis=0)
                cat_v = np.concatenate(parts_v, axis=0)
                cat_f0 = np.concatenate(parts_f0)
                if guided_controls is not None:
                    nf = raw_nf[idx]
                else:
                    raw_r = raw_nf[idx] / max(1.0, float(len(cat_e)))
                    eff_r = float(np.clip(1.0 + cfg.syl_dur_damp * (raw_r - 1.0), 0.78, 1.28)) if cfg.syl_dur_damp < 0.999 else float(np.clip(raw_r, 0.55, 2.0))
                    nf = max(12, int(round(len(cat_e) * eff_r)))
                e_st, b_st, v_st, f_sp_new = _stretch_preserving_onset(cat_e, cat_b, cat_v, split_pt, nf, preserve_onset=True)
                bg_st = np.clip(_resample_2d(cat_bg, nf), 0.005, 0.995)
                f0_tok = _resample_1d(cat_f0.astype(np.float64), nf)
                if use_unvoiced_residual and bo_sel is not None and bo_sel.onset_wav is not None and f_sp_new > 0:
                    residual_spans.append((cur_fr, cur_fr + f_sp_new, bo_sel.onset_wav))
                matched_tone = False

            raw_g = unit_gain_db[idx]
            if guided_controls is not None:
                e_st = e_st + float(np.clip(raw_g, -12.0, 12.0)) * math.log(10.0) / 20.0
            else:
                e_st = e_st + float(np.clip(raw_g, -2.5, 2.5)) * math.log(10.0) / 20.0

            if "f0_hz" in gc:
                explicit_f0_hz_blocks.append(
                    _interpolate_f0_hz_onto_mask(
                        np.asarray(gc["f0_hz"], dtype=np.float64), v_st, tgt_f0, voiced_hint=gc.get("voiced_hint"),
                    )
                )
                sts_rel_blocks.append(np.zeros(len(e_st), dtype=np.float64))
            elif "f0_st_contour" in gc:
                st_c = _resample_1d(np.asarray(gc["f0_st_contour"], dtype=np.float64), len(e_st))
                hz_b = np.zeros(len(e_st), dtype=np.float64)
                hz_b[v_st] = np.clip(tgt_f0 * (2.0 ** (st_c[v_st] / 12.0)), 60.0, 460.0)
                explicit_f0_hz_blocks.append(hz_b)
                sts_rel_blocks.append(np.zeros(len(e_st), dtype=np.float64))
            else:
                tgt_c = _template_contour_for_unit(u, len(e_st), iqr_scale, unit_decl_st[idx])
                if cfg.preserve_natural_f0:
                    tok_m = float(np.median(f0_tok[v_st])) if np.any(v_st) else float(np.median(f0_tok))
                    tgt_m = float(np.median(tgt_c[v_st])) if np.any(v_st) else float(np.median(tgt_c))
                    delta_s = float(np.clip(0.70 * (tgt_m - tok_m), -3.2, 3.2))
                    if matched_tone:
                        st_u = f0_tok + delta_s
                        if int(np.sum(v_st)) >= 4:
                            cr = _safe_corr_1d(f0_tok[v_st], st_u[v_st])
                            if cr is not None:
                                f0_corrs.append(cr)
                    else:
                        st_u = 0.78 * tgt_c + 0.22 * (f0_tok - tok_m + tgt_m)
                else:
                    f0_mic = np.clip(f0_tok - float(np.median(f0_tok)), -3.0, 3.0)
                    st_u = 0.68 * tgt_c + 0.32 * f0_mic
                explicit_f0_hz_blocks.append(None)
                sts_rel_blocks.append(st_u)

            envs.append(e_st); baps.append(b_st); bgt8s.append(bg_st); vois.append(v_st)
            is_obs_blk.append(o in OBSTRUENT_ONSETS)
            is_pause_blk.append(False)

            slot_timings_ms[idx] = [cur_fr * 5.0, (cur_fr + len(e_st)) * 5.0]
            cur_fr += len(e_st)
            unit_boundaries_ms.append(cur_fr * 5.0)
            boundary_frames.append(cur_fr)

            p_ms = unit_pause_after[idx]
            if p_ms > 15.0:
                p_fr = max(2, int(round(p_ms / 5.0)))
                envs.append(np.full((p_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
                baps.append(np.ones((p_fr, 5), dtype=np.float64))
                bgt8s.append(np.ones((p_fr, 2), dtype=np.float64))
                vois.append(np.zeros(p_fr, dtype=bool))
                explicit_f0_hz_blocks.append(np.zeros(p_fr, dtype=np.float64))
                sts_rel_blocks.append(np.zeros(p_fr, dtype=np.float64))
                is_obs_blk.append(False); is_pause_blk.append(True)
                cur_fr += p_fr

    tail_fr = 6
    envs.append(np.full((tail_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((tail_fr, 5), dtype=np.float64))
    bgt8s.append(np.ones((tail_fr, 2), dtype=np.float64))
    vois.append(np.zeros(tail_fr, dtype=bool))
    explicit_f0_hz_blocks.append(np.zeros(tail_fr, dtype=np.float64))
    sts_rel_blocks.append(np.zeros(tail_fr, dtype=np.float64))
    is_obs_blk.append(False); is_pause_blk.append(True)

    # Seam blending & 3-frame cosine boundary F0 pull ONLY at real chunk boundaries
    cos_w = np.array([0.85, 0.50, 0.20], dtype=np.float64)
    for idx_b in range(len(envs) - 1):
        if is_pause_blk[idx_b] or is_pause_blk[idx_b + 1]:
            continue
        if np.any(vois[idx_b]) and np.any(vois[idx_b + 1]):
            bf = 1 if is_obs_blk[idx_b + 1] else min(3, len(envs[idx_b]), len(envs[idx_b + 1]))
            _blend_seam(envs[idx_b], envs[idx_b + 1], baps[idx_b], baps[idx_b + 1], blend_frames=bf)
            seam_blends_applied += 1

        if (
            cfg.cos_seam_f0_pull
            and cfg.preserve_natural_f0
            and guided_controls is None
            and (not is_obs_blk[idx_b + 1])
            and explicit_f0_hz_blocks[idx_b] is None
            and explicit_f0_hz_blocks[idx_b + 1] is None
            and len(vois[idx_b]) >= 2
            and len(vois[idx_b + 1]) >= 2
            and bool(np.all(vois[idx_b][-2:]))
            and bool(np.all(vois[idx_b + 1][:2]))
        ):
            sb_a = sts_rel_blocks[idx_b]
            sb_b = sts_rel_blocks[idx_b + 1]
            jump = float(sb_b[0] - sb_a[-1])
            if abs(jump) > 0.05:
                na = min(3, len(sb_a))
                nb = min(3, len(sb_b))
                sb_a[-na:] += 0.5 * jump * cos_w[:na][::-1]
                sb_b[:nb] -= 0.5 * jump * cos_w[:nb]

    obs_fr_flags = np.concatenate([np.full(len(e), flag, dtype=bool) for e, flag in zip(envs, is_obs_blk)])
    pause_fr_flags = np.concatenate([np.full(len(e), flag, dtype=bool) for e, flag in zip(envs, is_pause_blk)])
    env_all = np.concatenate(envs, axis=0)
    bap_all = np.clip(np.concatenate(baps, axis=0), 0.005, 0.995)
    bgt8_all = np.clip(np.concatenate(bgt8s, axis=0), 0.005, 0.995)
    voi_all = np.concatenate(vois, axis=0)

    if cfg.gentle_gain:
        env_all = smooth_utterance_gain_gentle(env_all, voi_all, obs_fr_flags, pause_fr_flags, ref_vowel_db=ref_vowel_db)
    else:
        from exp_v10_integrity.gain_calibration import smooth_utterance_gain_contour

        env_all = smooth_utterance_gain_contour(env_all, voi_all, obs_fr_flags, pause_fr_flags, ref_vowel_db=ref_vowel_db)

    if any(b is not None for b in explicit_f0_hz_blocks):
        f0_all = np.zeros(len(env_all), dtype=np.float64)
        cursor = 0
        for e_blk, v_blk, hz_blk, st_blk in zip(envs, vois, explicit_f0_hz_blocks, sts_rel_blocks):
            L = len(e_blk)
            if hz_blk is not None:
                f0_all[cursor : cursor + L] = hz_blk
            else:
                f0_all[cursor : cursor + L][v_blk] = np.clip(tgt_f0 * (2.0 ** (st_blk[v_blk] / 12.0)), 65.0, 440.0)
            cursor += L
    else:
        st_all = np.concatenate(sts_rel_blocks, axis=0)
        f0_all = np.zeros(len(st_all), dtype=np.float64)
        v_idx = np.where(voi_all)[0]
        if v_idx.size > 0:
            st_centered = st_all[v_idx] - float(np.median(st_all[v_idx]))
            if len(st_centered) >= 5:
                sm = st_centered.copy()
                sm[1:-1] = 0.25 * st_centered[:-2] + 0.50 * st_centered[1:-1] + 0.25 * st_centered[2:]
                st_centered = sm
            f0_all[v_idx] = np.clip(tgt_f0 * (2.0 ** (st_centered / 12.0)), 65.0, 440.0)

    feats = vocoder.VocoderFeatures(
        f0_hz=f0_all,
        voiced=voi_all,
        log_env=env_all,
        bap=bap_all,
        sr=SR,
        hop_samples=HOP,
        n_samples=len(env_all) * HOP,
        n_fft=N_FFT,
    )
    gbank_use = _pick_glottal_bank(bank, tgt_name or "SSB0502", tgt_f0, glottal_bank)
    y = synthesize_v12(
        feats,
        mode="hybrid_lp_hnm",
        glottal_bank=gbank_use,
        crisp_unvoiced=True,
        seed=seed,
        phase_dispersion=phase_dispersion,
        bap_gt8k=bgt8_all,
        unvoiced_residual_spans=residual_spans if use_unvoiced_residual else None,
    )
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.89:
        y *= 0.89 / peak

    n_chunks = sum(1 for blk in is_pause_blk if not blk)
    n_seams_saved = sum(len(g) - 1 for g in groups if len(g) >= 2 and all(chosen_syls[i][1] is not None for i in g))
    exact_tone_hits = sum(
        1 for i, (_m, tk) in enumerate(chosen_syls)
        if tk is not None and int(tk.tone) == int(parsed[i][2])
    )
    fallback_cnt = sum(1 for m, _tk in chosen_syls if m != "zh_target_syl")
    chain_rate = round(same_utt_chain_hits / max(1, same_utt_chain_possible), 4)

    return y, {
        "source_counts": source_counts,
        "n_frames": len(env_all),
        "slot_timings_ms": slot_timings_ms,
        "unit_boundaries_ms": unit_boundaries_ms[:-1] if len(unit_boundaries_ms) > 1 else [],
        "boundary_frames": boundary_frames[:-1] if len(boundary_frames) > 1 else [],
        "prosody_applied": prosody_applied,
        "chain_rate": chain_rate,
        "same_utt_chain_hits": same_utt_chain_hits,
        "same_utt_chain_rate": chain_rate,
        "n_chunks": n_chunks,
        "n_internal_seams_saved": n_seams_saved,
        "seam_blends_applied": seam_blends_applied,
        "exact_tone_hit_rate": round(exact_tone_hits / max(1, n_u), 4),
        "fallback_rate": round(fallback_cnt / max(1, n_u), 4),
        "f0_natural_corr": round(float(np.mean(f0_corrs)), 4) if f0_corrs else None,
        "selected_trace": selected_trace,
        "f0_hz_all": f0_all,
        "log_env": env_all.astype(np.float32),
    }
