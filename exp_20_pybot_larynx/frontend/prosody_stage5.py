"""Stage 5 zero-ML-weight Mandarin prosody, duration, and onset-voicing residual refiner."""
from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Callable
import zlib

import numpy as np
from scipy.signal import lfilter

from exp_20_pybot_larynx import EXP_DIR
from exp_20_pybot_larynx.acoustics.knowledge import normalize_of
from exp_20_pybot_larynx.acoustics.knowledge_stage5 import (
    AcousticKnowledgeStage5,
    build_stage5_control,
)
from exp_20_pybot_larynx.arbiter import (
    OBSTRUENT_MANNERS,
    Arbiter,
    SyllableSpan,
    UtteranceContext,
    build_utterance_context,
    classify_final_group,
    classify_onset_manner,
)
from exp_20_pybot_larynx.frontend.prosody_rules import (
    _CF0_VOICED_ST,
    _CF0_VOICELESS_ST,
    _F0_AR1_SIGMA,
    _F0_U5,
    _F0_U7,
    _catmull_rom_5pt,
    _catmull_rom_7pt,
    predict_prosody_rules,
)
from exp_20_pybot_larynx.larynx.synth import synthesize_larynx

PROSODY_S5_TABLES_PATH = EXP_DIR / "frontend" / "prosody_s5_tables.npz"

_MANNER_TO_ID: dict[str, int] = {
    "zero": 0,
    "stop_unasp": 1,
    "stop_asp": 2,
    "affr_unasp": 3,
    "affr_asp": 4,
    "fric": 5,
    "nasal": 6,
    "liquid": 7,
}
_FG_TO_ID: dict[str, int] = {
    "i_front": 0,
    "u_round": 1,
    "v_front_round": 2,
    "a_open_central": 3,
}
_WPOS_TO_ID: dict[str, int] = {
    "mono": 0,
    "w_init": 1,
    "w_mid": 2,
    "w_fin": 3,
}


def compute_unit_context_ids(units: list[dict[str, Any]]) -> list[dict[str, int]]:
    """Compute integer prosodic context keys per unit without re-segmenting text."""
    n_u = len(units)
    if n_u == 0:
        return []

    # Group into clauses via punct_after or pause_after_ms >= 25
    clauses: list[list[int]] = []
    cur: list[int] = []
    for i, u in enumerate(units):
        cur.append(i)
        if str(u.get("punct_after", "")) or float(u.get("pause_after_ms", 0.0)) >= 25.0:
            clauses.append(cur)
            cur = []
    if cur:
        clauses.append(cur)

    ctx_out: list[dict[str, int]] = [{} for _ in range(n_u)]
    for cl in clauses:
        cl_len = len(cl)
        # Partition clause into prosodic phrases (break >= 2)
        phrases: list[list[int]] = []
        pcur: list[int] = []
        for ci, ui in enumerate(cl):
            pcur.append(ui)
            if ci == cl_len - 1 or int(units[ui].get("prosodic_break", 0)) >= 2:
                phrases.append(pcur)
                pcur = []
        ph_pos_map: dict[int, int] = {}
        for ph in phrases:
            lp = len(ph)
            for j, ui in enumerate(ph):
                ph_pos_map[ui] = min(2, int(3 * j / max(1, lp))) if lp > 1 else 1

        for ci, ui in enumerate(cl):
            u = units[ui]
            wl = int(u.get("word_len", 1))
            piw = int(u.get("pos_in_word", 0))
            wpos_s = "mono" if wl <= 1 else ("w_init" if piw == 0 else ("w_fin" if piw >= wl - 1 else "w_mid"))
            t = int(u.get("tone") or 1)
            pt = int(units[cl[ci - 1]].get("tone") or 1) if ci > 0 else 0
            nt = int(units[cl[ci + 1]].get("tone") or 1) if ci + 1 < cl_len else 0
            brk_ge2 = 1 if int(u.get("prosodic_break", 0)) >= 2 else 0
            is_wf = 1 if bool(u.get("is_word_final", True)) else 0
            cl_pos_b = min(3, int(4 * ci / max(1, cl_len))) if cl_len > 1 else 1

            o, f = normalize_of(str(u.get("onset", "")), str(u.get("final", "")))
            om = classify_onset_manner(o) if o else "zero"
            fg = classify_final_group(f)

            ctx_out[ui] = {
                "t": t,
                "pt": pt,
                "nt": nt,
                "is_wf": is_wf,
                "brk_ge2": brk_ge2,
                "cl_pos_b": cl_pos_b,
                "ph_pos_b": ph_pos_map.get(ui, 1),
                "om_id": _MANNER_TO_ID.get(om, 0),
                "fg_id": _FG_TO_ID.get(fg, 3),
                "wpos_id": _WPOS_TO_ID.get(wpos_s, 0),
                "o": o,
                "fg": fg,
            }
    return ctx_out


class ProsodyStage5Tables:
    """Explicit lookup tables mined strictly from TRAIN for Stage 5 prosody refinement."""

    def __init__(self, path: Path | str = PROSODY_S5_TABLES_PATH) -> None:
        self.loaded = False
        self.f0_global: np.ndarray = np.zeros(3, dtype=np.float64)
        self.f0_tritone: dict[tuple[int, int, int, int, int], np.ndarray] = {}
        self.f0_phrase: dict[tuple[int, int, int, int], np.ndarray] = {}
        self.dur_ctx: dict[tuple[int, int, int, int, int], float] = {}
        self.on_dur_ctx: dict[tuple[str, str, int], float] = {}
        self.gain_ctx: dict[tuple[int, int, int, int], float] = {}
        self.onset_uv: dict[str, float] = {}
        self.weights: dict[str, float] = {
            "w_f0_tri": 0.65,
            "w_f0_phr": 0.65,
            "w_dur": 0.70,
            "w_ondur": 0.70,
            "w_gain": 0.65,
            "w_uv": 1.0,
        }

        p = Path(path)
        if not p.exists():
            return
        with np.load(p, allow_pickle=False) as d:
            if "f0_global" in d:
                self.f0_global = d["f0_global"].astype(np.float64)
            if "f0_tri_keys" in d:
                for k, v in zip(d["f0_tri_keys"], d["f0_tri_vals"]):
                    self.f0_tritone[tuple(int(x) for x in k)] = v.astype(np.float64)
            if "f0_phr_keys" in d:
                for k, v in zip(d["f0_phr_keys"], d["f0_phr_vals"]):
                    self.f0_phrase[tuple(int(x) for x in k)] = v.astype(np.float64)
            if "dur_ctx_keys" in d:
                for k, v in zip(d["dur_ctx_keys"], d["dur_ctx_vals"]):
                    self.dur_ctx[tuple(int(x) for x in k)] = float(v)
            if "on_dur_keys" in d:
                for k, v in zip(d["on_dur_keys"], d["on_dur_vals"]):
                    o_s, fg_s, brk_s = str(k).split("|")
                    self.on_dur_ctx[(o_s, fg_s, int(brk_s))] = float(v)
            if "gain_ctx_keys" in d:
                for k, v in zip(d["gain_ctx_keys"], d["gain_ctx_vals"]):
                    self.gain_ctx[tuple(int(x) for x in k)] = float(v)
            if "onset_uv_keys" in d:
                for k, v in zip(d["onset_uv_keys"], d["onset_uv_vals"]):
                    self.onset_uv[str(k)] = float(v)
            if "weight_keys" in d and "weight_vals" in d:
                for k, v in zip(d["weight_keys"], d["weight_vals"]):
                    self.weights[str(k)] = float(v)
        self.loaded = True


_TABLES_CACHE: dict[str, ProsodyStage5Tables] = {}


def get_stage5_prosody_tables(path: Path | str = PROSODY_S5_TABLES_PATH) -> ProsodyStage5Tables:
    key = str(Path(path).resolve())
    tbl = _TABLES_CACHE.get(key)
    if tbl is None:
        tbl = ProsodyStage5Tables(path)
        _TABLES_CACHE[key] = tbl
    return tbl


def clear_stage5_prosody_cache() -> None:
    _TABLES_CACHE.clear()


def predict_prosody_stage5(
    text: str,
    knowledge: Any | None = None,
    *,
    style: str = "auto",
    continues_after: bool = False,
    tables_path: Path | str = PROSODY_S5_TABLES_PATH,
    tables: ProsodyStage5Tables | None = None,
) -> dict[str, Any]:
    """Predict Stage 5 prosody by layering TRAIN-mined residual tables onto predict_prosody_rules."""
    pro = predict_prosody_rules(text, knowledge, style=style, continues_after=continues_after)
    tbl = tables or get_stage5_prosody_tables(tables_path)
    if not tbl.loaded:
        return pro

    units = pro["units"]
    ctxs = compute_unit_context_ids(units)
    w = tbl.weights
    w_tri = float(w.get("w_f0_tri", 0.65))
    w_phr = float(w.get("w_f0_phr", 0.65))
    w_dur = float(w.get("w_dur", 0.70))
    w_on = float(w.get("w_ondur", 0.70))
    w_gain = float(w.get("w_gain", 0.65))
    use_uv = float(w.get("w_uv", 1.0)) > 0.5

    u7_arr = np.asarray(_F0_U7, dtype=np.float64)
    u5_arr = np.asarray(_F0_U5, dtype=np.float64)

    for u, c in zip(units, ctxs):
        t = c["t"]
        # 1) Duration & onset split residual
        d_log = tbl.dur_ctx.get((c["om_id"], c["fg_id"], t, c["wpos_id"], c["brk_ge2"]), 0.0)
        if w_dur != 0.0 and abs(d_log) > 1e-6:
            new_dur = float(np.clip(float(u["duration_ms"]) * math.exp(w_dur * d_log), 60.0, 380.0))
            u["duration_ms"] = round(new_dur, 1)

        if c["o"]:
            on_log = tbl.on_dur_ctx.get((c["o"], c["fg"], c["brk_ge2"]), 0.0)
            if w_on != 0.0 and abs(on_log) > 1e-6:
                new_on = float(np.clip(float(u["onset_ms"]) * math.exp(w_on * on_log), 12.0, min(200.0, float(u["duration_ms"]) - 20.0)))
                u["onset_ms"] = round(new_on, 1)
            if use_uv and c["o"] in tbl.onset_uv:
                u["onset_uv_frac"] = round(float(tbl.onset_uv[c["o"]]), 4)

        # 2) F0 residual (3-point interpolated onto 7-point & 5-point contours)
        df_tri = tbl.f0_tritone.get((c["pt"], t, c["nt"], c["is_wf"], c["brk_ge2"]), tbl.f0_global)
        df_phr = tbl.f0_phrase.get((t, c["cl_pos_b"], c["ph_pos_b"], c["brk_ge2"]))
        df3 = w_tri * df_tri
        if df_phr is not None:
            df3 = df3 + w_phr * df_phr

        if np.any(np.abs(df3) > 1e-6):
            df7 = np.interp(u7_arr, (0.20, 0.50, 0.80), df3)
            pts7 = np.asarray(u.get("f0_7pt_st", np.interp(u7_arr, u5_arr, u["f0_5pt_st"])), dtype=np.float64) + df7
            pts5 = _catmull_rom_7pt(u5_arr, pts7)
            u["f0_3pt_st"] = [round(float(x), 3) for x in (np.asarray(u.get("f0_3pt_st", [pts7[1], pts7[3], pts7[5]])) + df3)]
            u["f0_7pt_st"] = [round(float(x), 3) for x in pts7]
            u["f0_5pt_st"] = [round(float(x), 3) for x in pts5]
            u["pitch_st"] = list(u["f0_5pt_st"])

        # 3) Gain residual
        dg = tbl.gain_ctx.get((t, c["wpos_id"], c["brk_ge2"], c["cl_pos_b"]), 0.0)
        if w_gain != 0.0 and abs(dg) > 1e-6:
            u["gain_db"] = round(float(np.clip(float(u.get("gain_db", 0.0)) + w_gain * dg, -5.0, 5.0)), 3)

    return pro


def render_stage5_utterance(
    text: str,
    utt_id: str,
    knowledge: AcousticKnowledgeStage5,
    *,
    arbiter: Arbiter,
    rules: list[Any],
    glottal_mode: str = "parametric_lf",
    coherent_phasor: np.ndarray | None = None,
    seed: int = 123,
    style: str = "auto",
    tables_path: Path | str = PROSODY_S5_TABLES_PATH,
    tables: ProsodyStage5Tables | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pure zero-ML Stage 5 end-to-end Mandarin TTS synthesis."""
    pro = predict_prosody_stage5(text, knowledge, style=style, tables_path=tables_path, tables=tables)
    raw_units = pro["units"]
    base_f0 = float(pro["speaker_f0_med_hz"])
    n_u = len(raw_units)

    spans: list[SyllableSpan] = []
    f0_chunks: list[np.ndarray] = []
    voi_chunks: list[np.ndarray] = []
    syl_gains: list[float] = []

    cur_fr = 0
    prev_voiced_end_hz: float | None = None

    for i, u in enumerate(raw_units):
        o, f = normalize_of(str(u.get("onset", "")), str(u.get("final", "")))
        tone = int(u.get("tone") or 1)
        dur_ms = max(60.0, float(u.get("duration_ms", 180.0)))
        n_syl = max(3, int(round(dur_ms / 5.0)))

        if o:
            n_on = max(1, min(n_syl - 2, int(round(float(u.get("onset_ms", 50.0)) / 5.0))))
        else:
            n_on = 0
        split_fr = cur_fr + n_on
        end_fr = cur_fr + n_syl

        manner = classify_onset_manner(o) if o else "zero"
        n_fi = max(1, n_syl - n_on)
        u_fi = np.linspace(0.0, 1.0, n_fi) if n_fi > 1 else np.array([0.5], dtype=np.float64)
        if "f0_7pt_st" in u:
            fi_st = _catmull_rom_7pt(u_fi, np.asarray(u["f0_7pt_st"], dtype=np.float64))
        else:
            fi_st = _catmull_rom_5pt(u_fi, np.asarray(u["f0_5pt_st"], dtype=np.float64))

        if o and n_fi >= 3:
            k_cf0 = min(5, n_fi)
            decay = np.exp(-np.arange(k_cf0, dtype=np.float64) * 5.0 / 12.0)
            if manner in ("stop_asp", "affr_asp", "fric"):
                fi_st[:k_cf0] += _CF0_VOICELESS_ST * decay
            elif manner in ("nasal", "liquid"):
                fi_st[:k_cf0] += _CF0_VOICED_ST * decay
            elif manner in ("stop_unasp", "affr_unasp"):
                fi_st[:k_cf0] += 0.25 * _CF0_VOICELESS_ST * decay

        fi_hz = np.clip(base_f0 * (2.0 ** (fi_st / 12.0)), 65.0, 440.0)
        f0_seg = np.empty(n_syl, dtype=np.float64)
        f0_seg[n_on:] = fi_hz
        if n_on > 0:
            if manner in ("nasal", "liquid") and prev_voiced_end_hz is not None:
                f0_seg[:n_on] = np.linspace(
                    0.65 * prev_voiced_end_hz + 0.35 * fi_hz[0],
                    fi_hz[0],
                    n_on,
                    endpoint=False,
                )
            else:
                f0_seg[:n_on] = fi_hz[0]

        voi_seg = np.ones(n_syl, dtype=bool)
        if o and manner in OBSTRUENT_MANNERS and n_on > 0:
            uv_f = float(u.get("onset_uv_frac", 0.78))
            uv_n = max(1, min(n_on, int(round(uv_f * n_on))))
            voi_seg[:uv_n] = False
            f0_seg[:uv_n] = 0.0

        brk = int(u.get("prosodic_break", 0))
        t_uv = 1 if brk == 2 else (2 if brk == 3 else (4 if brk == 4 else 0))
        if t_uv > 0 and n_fi >= 8:
            t_uv = min(t_uv, n_fi - 6)
            voi_seg[-t_uv:] = False
            f0_seg[-t_uv:] = 0.0

        if prev_voiced_end_hz is not None:
            if voi_seg[0] and n_on == 0 and f0_seg[0] > 50.0:
                k_sm = min(4, n_syl)
                w_sm = np.linspace(0.30, 0.0, k_sm)
                f0_seg[:k_sm] = (1.0 - w_sm) * f0_seg[:k_sm] + w_sm * prev_voiced_end_hz
            elif n_on < n_syl and manner in OBSTRUENT_MANNERS:
                k_sm = min(3, n_syl - n_on)
                w_sm = np.linspace(0.10, 0.0, k_sm)
                f0_seg[n_on : n_on + k_sm] = (1.0 - w_sm) * f0_seg[n_on : n_on + k_sm] + w_sm * prev_voiced_end_hz

        p_ms = float(u.get("pause_after_ms", 0.0)) if i < n_u - 1 else 0.0
        spans.append(SyllableSpan(
            index=i,
            hanzi=str(u.get("hanzi") or u.get("token") or ""),
            onset=o,
            final=f,
            tone=tone,
            start_fr=cur_fr,
            split_fr=split_fr,
            end_fr=end_fr,
            pause_after_ms=p_ms,
            prosodic_break=brk,
        ))
        f0_chunks.append(f0_seg)
        voi_chunks.append(voi_seg)
        syl_gains.append(float(u.get("gain_db", 0.0)))
        cur_fr = end_fr

        if i < n_u - 1 and p_ms >= 20.0:
            n_p = int(round(p_ms / 5.0))
            if n_p > 0:
                f0_chunks.append(np.zeros(n_p, dtype=np.float64))
                voi_chunks.append(np.zeros(n_p, dtype=bool))
                cur_fr += n_p
            prev_voiced_end_hz = None
        else:
            prev_voiced_end_hz = float(f0_seg[-1]) if voi_seg[-1] and f0_seg[-1] > 50.0 else None

    f0_all = np.concatenate(f0_chunks)
    voi_all = np.concatenate(voi_chunks)

    if _F0_AR1_SIGMA > 0.0 and len(f0_all) > 4:
        rng_u = np.random.default_rng(zlib.crc32(f"{utt_id}|{seed}".encode("utf-8")) & 0xFFFFFFFF)
        rho = 0.72
        eps = rng_u.normal(0.0, _F0_AR1_SIGMA * np.sqrt(1.0 - rho * rho), size=len(f0_all))
        ar1 = lfilter((1.0,), (1.0, -rho), eps)
        vmask = voi_all & (f0_all > 50.0)
        f0_all[vmask] = np.clip(f0_all[vmask] * (2.0 ** (np.clip(ar1[vmask], -0.45, 0.45) / 12.0)), 65.0, 440.0)

    ctx: UtteranceContext = build_utterance_context(utt_id, cur_fr, spans, speaker_f0_med_hz=base_f0)
    base_ctrl, kb_info = build_stage5_control(
        ctx,
        f0_all,
        voi_all,
        knowledge,
        syl_gain_db=np.asarray(syl_gains, dtype=np.float64),
    )
    fused_ctrl, arb_trace = arbiter.apply(ctx, base_ctrl, rules)
    t_k0 = time.perf_counter()
    wav = synthesize_larynx(
        fused_ctrl,
        glottal_mode=glottal_mode,
        coherent_phasor=coherent_phasor,
        seed=seed,
    )
    k_elapsed = time.perf_counter() - t_k0
    return wav, {
        "kb_info": kb_info,
        "arbiter_trace": arb_trace,
        "kernel_s": k_elapsed,
        "prosody_plan": pro,
    }
