"""Zero-leakage 5-rung diagnostic decomposing the 4.03 dB gap + Mandarin T3 sandhi audit."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.fft import rfft

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v5_eval.metrics import (  # noqa: E402
    _mel_filterbank,
    compute_f0_hnr_metrics,
    compute_spectral_distances,
)
from exp_v6_crossling_dsp import arbiter as v6_arb  # noqa: E402
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS  # noqa: E402
from exp_v6_eval.fluency import compute_f0_track  # noqa: E402
from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v7_pybot_tts import crossling_warp as cw  # noqa: E402
from exp_v8_clarity.utau_concat import (  # noqa: E402
    N_BINS,
    OBSTRUENT_ONSETS,
    SILENCE_LOG_ENV,
    _blend_seam,
    _stretch_preserving_onset,
    _resample_1d,
    _resample_2d,
)
from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10  # noqa: E402
from exp_v10_integrity.gain_calibration import (  # noqa: E402
    calibrate_token_parseval_gain,
    classify_ipa_manner,
    smooth_utterance_gain_contour,
)
from exp_v11_align_gmm.timbre_cloner_v5 import adapt_features_v5  # noqa: E402
from exp_v11_diphone_tts.v11_renderer import (  # noqa: E402
    REF_VOWEL_PARSEVAL_DB,
    _adapt_v11,
    _get_src_prof,
    _viterbi_zh_syllables,
)
from exp_v12_natural_vocoder.v12_renderer import render_v12_utterance  # noqa: E402
from exp_v12_natural_vocoder.vocoder_v12 import VocoderFeaturesV12, synthesize_v12  # noqa: E402
from exp_v12_prosody.prosody_model import DEFAULT_MODEL_PATH, load_prosody_v12  # noqa: E402
from exp_v13_utau_ceiling import EXP_DIR, HOP, N_FFT, SPOTLIGHT_6_IDS, SR  # noqa: E402
from exp_v13_utau_ceiling.zh_bank_v13 import (  # noqa: E402
    V13ZhBank,
    V13ZhToken,
    _analyze_or_load_zh_features,
    build_v13_zh_bank,
    extract_utterance_zh_tokens,
    load_gmm_zh_map,
    load_zh_manifest_utterances,
)

LEAD_SIL_FRAMES = 4
TAIL_SIL_FRAMES = 6


def _extract_mel_frames_aligned(x: np.ndarray, sr: int = SR, n_mels: int = 40) -> np.ndarray:
    """Extract 5 ms-aligned 40-band log-mel dB frames matching HOP=120 framing."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    win_len = int(round(0.025 * sr))
    n_fr = max(1, len(arr) // HOP)
    pad_l = win_len // 2
    xp = np.pad(arr, (pad_l, pad_l + win_len))
    win = np.hamming(win_len)
    fb = _mel_filterbank(sr=sr, nfft=2048, n_mels=n_mels)
    rows = np.empty((n_fr, win_len), dtype=np.float64)
    for i in range(n_fr):
        c = i * HOP + pad_l
        seg = xp[c - pad_l : c - pad_l + win_len]
        rows[i] = (seg - np.mean(seg)) * win
    pow_spec = np.abs(rfft(rows, n=2048, axis=1)) ** 2
    mel_pow = np.maximum(pow_spec @ fb.T, 1e-12)
    return (10.0 / math.log(10.0)) * np.log(mel_pow)


def _frame_aligned_mel_lsd(
    x_ref: np.ndarray,
    x_syn: np.ndarray,
    *,
    syn_lead_frames: int = 0,
    n_speech_frames: int | None = None,
) -> dict[str, float]:
    """Compute non-DTW frame-aligned Mel-LSD (both shape-only mean-normed and raw energy-included)."""
    mel_r = _extract_mel_frames_aligned(x_ref)
    mel_s = _extract_mel_frames_aligned(x_syn)
    if syn_lead_frames > 0 and len(mel_s) > syn_lead_frames:
        mel_s = mel_s[syn_lead_frames:]
    n = min(len(mel_r), len(mel_s))
    if n_speech_frames is not None:
        n = min(n, int(n_speech_frames))
    if n <= 0:
        return {"nodtw_shape_mel_lsd_db": float("nan"), "nodtw_raw_mel_lsd_db": float("nan")}
    mr = mel_r[:n]
    ms = mel_s[:n]
    # Align global RMS offset before computing raw Mel-LSD so overall playback scale doesn't skew
    global_off = float(np.mean(mr) - np.mean(ms))
    ms_g = ms + global_off
    raw_lsd = float(np.mean(np.sqrt(np.mean((mr - ms_g) ** 2, axis=1))))

    mr_n = mr - np.mean(mr, axis=1, keepdims=True)
    ms_n = ms - np.mean(ms, axis=1, keepdims=True)
    shape_lsd = float(np.mean(np.sqrt(np.mean((mr_n - ms_n) ** 2, axis=1))))
    return {
        "nodtw_shape_mel_lsd_db": shape_lsd,
        "nodtw_raw_mel_lsd_db": raw_lsd,
    }


def _compute_f0_mae_st(
    x_ref: np.ndarray,
    x_syn: np.ndarray,
    sr: int = SR,
    *,
    syn_lead_frames: int = 0,
) -> tuple[float, float]:
    """Return (framewise_f0_mae_st, utt_median_f0_err_st)."""
    x_syn_al = x_syn[syn_lead_frames * HOP :] if syn_lead_frames > 0 and len(x_syn) > syn_lead_frames * HOP else x_syn
    tr = compute_f0_track(x_ref, sr=sr, fmin=75.0, fmax=450.0)
    ts = compute_f0_track(x_syn_al, sr=sr, fmin=75.0, fmax=450.0)
    n = min(len(tr["f0_hz"]), len(ts["f0_hz"]))
    mask = tr["voiced"][:n] & ts["voiced"][:n] & np.isfinite(tr["f0_hz"][:n]) & np.isfinite(ts["f0_hz"][:n])
    if np.any(mask):
        fr_err = float(np.mean(np.abs(12.0 * np.log2(ts["f0_hz"][:n][mask] / tr["f0_hz"][:n][mask]))))
    else:
        fr_err = float("nan")

    hr = compute_f0_hnr_metrics(x_ref, sr=sr, fmin=75.0, fmax=450.0)
    hs = compute_f0_hnr_metrics(x_syn_al, sr=sr, fmin=75.0, fmax=450.0)
    if hr and hs and hr["f0_median_hz"] and hs["f0_median_hz"]:
        med_err = abs(12.0 * math.log2(hs["f0_median_hz"] / hr["f0_median_hz"]))
    else:
        med_err = float("nan")
    return fr_err, med_err


def _interpolate_oracle_f0_for_syllable(
    real_f0_seg: np.ndarray,
    real_voi_seg: np.ndarray,
    tok_voiced: np.ndarray,
    fallback_f0: float,
) -> np.ndarray:
    """Map real utterance syllable F0 onto token's voiced mask without zeroing unaligned edge frames."""
    n = len(tok_voiced)
    out = np.zeros(n, dtype=np.float64)
    v_idx = np.where(real_voi_seg & (real_f0_seg > 40.0))[0]
    if v_idx.size == 0:
        out[tok_voiced] = fallback_f0
        return out
    t_src = np.linspace(0.0, 1.0, len(real_f0_seg))[v_idx]
    log_f0 = np.log2(real_f0_seg[v_idx])
    t_dst = np.linspace(0.0, 1.0, n)
    interp_f0 = 2.0 ** np.interp(t_dst, t_src, log_f0)
    out[tok_voiced] = np.clip(interp_f0[tok_voiced], 65.0, 440.0)
    return out


def _select_v12_tokens_for_utterance(
    syl_specs: list[tuple[str, str, int, float]],
    bank: V13ZhBank,
    tgt_name: str,
) -> list[tuple[str, V13ZhToken | None]]:
    """Exact replica of v12_renderer Mandarin selection logic (whole-syllable Viterbi + greedy O+F)."""
    slots: list[tuple[list[V13ZhToken], list[float], bool, bool]] = []
    slot_indices: list[int] = []
    selected: dict[int, V13ZhToken | None] = {}
    n_u = len(syl_specs)

    for idx, (o, f, tone, target_dur_ms) in enumerate(syl_specs):
        prev_f = syl_specs[idx - 1][1] if idx > 0 else "SIL"
        next_o = syl_specs[idx + 1][0] if idx + 1 < n_u else "SIL"
        cands = bank.zh_syllables.get((o, f), [])
        if cands:
            is_uv_zh = o in ("f", "s", "sh", "x", "h", "c", "ch", "q")
            tcosts = []
            for c in cands:
                tc = 0.0 if c.speaker == tgt_name else 3.5
                if c.tone != tone:
                    tc += 2.2
                if c.prev_final != prev_f:
                    tc += 1.3
                if c.next_onset != next_o:
                    tc += 1.3
                if is_uv_zh and c.uv_hf_lf < -4.0:
                    tc += 2.5
                tc += 0.65 * abs(math.log(max(20.0, c.dur_ms) / max(20.0, target_dur_ms)))
                tcosts.append(tc)
            slots.append((cands, tcosts, False, o in OBSTRUENT_ONSETS))
            slot_indices.append(idx)
        else:
            if slots:
                for si, tk in zip(slot_indices, _viterbi_zh_syllables(slots)):
                    selected[si] = tk
                slots, slot_indices = [], []
            selected[idx] = None
    if slots:
        for si, tk in zip(slot_indices, _viterbi_zh_syllables(slots)):
            selected[si] = tk

    out: list[tuple[str, V13ZhToken | None]] = []
    for idx in range(n_u):
        tk = selected.get(idx)
        if tk is not None:
            bucket = "same_spk_same_tone" if (tk.speaker == tgt_name and tk.tone == syl_specs[idx][2]) else (
                "same_spk_diff_tone" if tk.speaker == tgt_name else "donor_spk"
            )
            out.append((bucket, tk))
        else:
            out.append(("zh_concat_of", None))
    return out


def _render_tokens_guided(
    syl_specs: list[tuple[str, str, int, int, np.ndarray, np.ndarray]],
    chosen: list[tuple[str, V13ZhToken | None]],
    bank: V13ZhBank,
    tgt_name: str,
    *,
    apply_gain_cal: bool = True,
    apply_seam_blend: bool = True,
    apply_smooth_contour: bool = True,
    use_unvoiced_residual: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render chosen ZH tokens through v12 pipeline under exact oracle durations and F0 contours."""
    target_profile = bank.profiles_v5[tgt_name]
    tgt_f0 = float(target_profile.v3.f0_median_hz)
    zh_fb_prof = _get_src_prof(bank, "SSB0502", target_profile, bank.profiles_v5) or target_profile

    envs, baps, bgt8s, vois, f0s = [], [], [], [], []
    is_obs_blk, is_pause_blk = [], []
    residual_spans: list[tuple[int, int, np.ndarray]] = []
    syl_frame_spans: list[tuple[int, int, str, int]] = []  # (start_fr, end_fr, bucket, f_split_new)

    envs.append(np.full((LEAD_SIL_FRAMES, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((LEAD_SIL_FRAMES, 5), dtype=np.float64))
    bgt8s.append(np.ones((LEAD_SIL_FRAMES, 2), dtype=np.float64))
    vois.append(np.zeros(LEAD_SIL_FRAMES, dtype=bool))
    f0s.append(np.zeros(LEAD_SIL_FRAMES, dtype=np.float64))
    is_obs_blk.append(False)
    is_pause_blk.append(True)
    cur_fr = LEAD_SIL_FRAMES
    n_u = len(syl_specs)

    for idx, ((o, f, tone, nf, real_f0_seg, real_voi_seg), (bucket, tok)) in enumerate(zip(syl_specs, chosen)):
        prev_f = syl_specs[idx - 1][1] if idx > 0 else "SIL"
        next_o = syl_specs[idx + 1][0] if idx + 1 < n_u else "SIL"
        if tok is not None:
            e_raw, b_raw, bg_raw, v_raw = _adapt_v11(
                tok.log_env,
                tok.bap,
                tok.bap_gt8k,
                tok.voiced,
                tok.f0_rel_st,
                _get_src_prof(bank, tok.speaker, target_profile, bank.profiles_v5),
                target_profile,
                adapt_features_v5,
                None,
            )
            if apply_gain_cal:
                e_raw, _ = calibrate_token_parseval_gain(e_raw, "zh_syllable", REF_VOWEL_PARSEVAL_DB, max_corr_db=10.5)
            e_st, b_st, v_st, f_sp_new = _stretch_preserving_onset(e_raw, b_raw, v_raw, tok.f_split, nf, preserve_onset=True)
            bg_st = np.clip(_resample_2d(bg_raw, nf), 0.005, 0.995)
            if use_unvoiced_residual and tok.onset_wav is not None and f_sp_new > 0:
                residual_spans.append((cur_fr, cur_fr + f_sp_new, tok.onset_wav))
        else:
            parts_e, parts_b, parts_bg, parts_v = [], [], [], []
            split_pt = 0
            bo_sel = None
            if o:
                oc = bank.zh_onsets.get(o, [])
                if oc:
                    is_uv_o = o in ("f", "s", "sh", "x", "h", "c", "ch", "q")
                    bo = min(
                        oc,
                        key=lambda c: (0.0 if c.speaker == tgt_name else 3.5)
                        + (0.0 if c.prev_final == prev_f else 1.3)
                        + (0.0 if c.next_onset == f else 1.0)
                        + (0.0 if c.dur_ms >= 25.0 else 2.5)
                        + (2.5 if (is_uv_o and c.uv_hf_lf < -4.0) else 0.0),
                    )
                    bo_sel = bo
                    eo, bo_b, bgo, vo = _adapt_v11(
                        bo.log_env,
                        bo.bap,
                        bo.bap_gt8k,
                        bo.voiced,
                        bo.f0_rel_st,
                        _get_src_prof(bank, bo.speaker, target_profile, bank.profiles_v5),
                        target_profile,
                        adapt_features_v5,
                        None,
                    )
                    if apply_gain_cal:
                        eo, _ = calibrate_token_parseval_gain(eo, "voiceless_fricative" if is_uv_o else "nasal", REF_VOWEL_PARSEVAL_DB, max_corr_db=8.5)
                    parts_e.append(eo); parts_b.append(bo_b); parts_bg.append(bgo); parts_v.append(vo)
                    split_pt = len(eo)
                elif bank.fallback_unit_bank is not None:
                    for oi in cf.PINYIN_ONSET_TO_IPA.get(o, ()):
                        d = cw.lookup_ipa_template(bank.fallback_unit_bank, oi, is_onset=True)
                        bg_d = np.repeat(d["bap"][:, 4:5], 2, axis=1)
                        ed, bd, bgd, vd = _adapt_v11(
                            d["log_env"] - 3.1, d["bap"], bg_d, d["voiced"], np.zeros(len(d["log_env"])),
                            zh_fb_prof, target_profile, adapt_features_v5, None,
                        )
                        if apply_gain_cal:
                            ed, _ = calibrate_token_parseval_gain(ed, classify_ipa_manner(oi, None), REF_VOWEL_PARSEVAL_DB, max_corr_db=8.5)
                        parts_e.append(ed); parts_b.append(bd); parts_bg.append(bgd); parts_v.append(vd)
                    split_pt = sum(len(x) for x in parts_e)

            fc = bank.zh_finals.get(f, [])
            if fc:
                bf = min(
                    fc,
                    key=lambda c: (0.0 if c.speaker == tgt_name else 3.5)
                    + (0.0 if c.tone == tone else (4.0 if (tone != 5 and c.tone == 5) else 1.8))
                    + (0.0 if c.prev_final in (o, "h", "g", "k", "") else 1.5)
                    + (0.0 if c.next_onset == next_o else 1.2),
                )
                ef, bf_b, bgf, vf = _adapt_v11(
                    bf.log_env, bf.bap, bf.bap_gt8k, bf.voiced, bf.f0_rel_st,
                    _get_src_prof(bank, bf.speaker, target_profile, bank.profiles_v5), target_profile, adapt_features_v5, None,
                )
                if apply_gain_cal:
                    ef, _ = calibrate_token_parseval_gain(ef, "zh_syllable", REF_VOWEL_PARSEVAL_DB, max_corr_db=10.5)
                parts_e.append(ef); parts_b.append(bf_b); parts_bg.append(bgf); parts_v.append(vf)
            elif bank.fallback_unit_bank is not None:
                bucket = "zh_ipa_fill"
                for fi in cf.PINYIN_FINAL_TO_IPA.get(f, ("ə",)):
                    d = cw.lookup_ipa_template(bank.fallback_unit_bank, fi, is_onset=False)
                    bg_d = np.repeat(d["bap"][:, 4:5], 2, axis=1)
                    ed, bd, bgd, vd = _adapt_v11(
                        d["log_env"] - 3.1, d["bap"], bg_d, d["voiced"], np.zeros(len(d["log_env"])),
                        zh_fb_prof, target_profile, adapt_features_v5, None,
                    )
                    if apply_gain_cal:
                        ed, _ = calibrate_token_parseval_gain(ed, "zh_syllable", REF_VOWEL_PARSEVAL_DB, max_corr_db=10.5)
                    parts_e.append(ed); parts_b.append(bd); parts_bg.append(bgd); parts_v.append(vd)

            e_st, b_st, v_st, f_sp_new = _stretch_preserving_onset(
                np.concatenate(parts_e, axis=0),
                np.concatenate(parts_b, axis=0),
                np.concatenate(parts_v, axis=0),
                split_pt,
                nf,
                preserve_onset=True,
            )
            if use_unvoiced_residual and bo_sel is not None and bo_sel.onset_wav is not None and f_sp_new > 0:
                residual_spans.append((cur_fr, cur_fr + f_sp_new, bo_sel.onset_wav))
            bg_st = np.clip(_resample_2d(np.concatenate(parts_bg, axis=0), nf), 0.005, 0.995)

        f0_u = _interpolate_oracle_f0_for_syllable(real_f0_seg, real_voi_seg, v_st, tgt_f0)
        envs.append(e_st); baps.append(b_st); bgt8s.append(bg_st); vois.append(v_st); f0s.append(f0_u)
        is_obs_blk.append(o in OBSTRUENT_ONSETS)
        is_pause_blk.append(False)
        syl_frame_spans.append((cur_fr, cur_fr + nf, bucket, f_sp_new))
        cur_fr += nf

    envs.append(np.full((TAIL_SIL_FRAMES, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((TAIL_SIL_FRAMES, 5), dtype=np.float64))
    bgt8s.append(np.ones((TAIL_SIL_FRAMES, 2), dtype=np.float64))
    vois.append(np.zeros(TAIL_SIL_FRAMES, dtype=bool))
    f0s.append(np.zeros(TAIL_SIL_FRAMES, dtype=np.float64))
    is_obs_blk.append(False)
    is_pause_blk.append(True)

    if apply_seam_blend:
        for idx_b in range(len(envs) - 1):
            if is_pause_blk[idx_b] or is_pause_blk[idx_b + 1]:
                continue
            if np.any(vois[idx_b]) and np.any(vois[idx_b + 1]):
                bf = 1 if is_obs_blk[idx_b + 1] else min(3, len(envs[idx_b]), len(envs[idx_b + 1]))
                _blend_seam(envs[idx_b], envs[idx_b + 1], baps[idx_b], baps[idx_b + 1], blend_frames=bf)

    obs_fr_flags = np.concatenate([np.full(len(e), flag, dtype=bool) for e, flag in zip(envs, is_obs_blk)])
    pause_fr_flags = np.concatenate([np.full(len(e), flag, dtype=bool) for e, flag in zip(envs, is_pause_blk)])
    env_all = np.concatenate(envs, axis=0)
    bap_all = np.clip(np.concatenate(baps, axis=0), 0.005, 0.995)
    bgt8_all = np.clip(np.concatenate(bgt8s, axis=0), 0.005, 0.995)
    voi_all = np.concatenate(vois, axis=0)
    f0_all = np.concatenate(f0s, axis=0)

    if apply_smooth_contour:
        env_all = smooth_utterance_gain_contour(
            env_all, voi_all, obs_fr_flags, pause_fr_flags, ref_vowel_db=REF_VOWEL_PARSEVAL_DB
        )

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
    gbank_use = bank.glottal_banks.get(tgt_name)
    y = synthesize_v12(
        feats,
        mode="hybrid_lp_hnm",
        glottal_bank=gbank_use,
        crisp_unvoiced=True,
        seed=72,
        phase_dispersion=True,
        bap_gt8k=bgt8_all,
        unvoiced_residual_spans=residual_spans if use_unvoiced_residual else None,
    )
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.89:
        y *= 0.89 / peak
    return y, {"syl_frame_spans": syl_frame_spans, "log_env": env_all}


def _oracle_select_for_utterance(
    u: dict[str, Any],
    xa: np.ndarray,
    feats_real: vocoder.VocoderFeatures,
    syl_bounds: list[tuple[int, int, int]],
    bank: V13ZhBank,
    tgt_name: str,
) -> list[tuple[str, V13ZhToken | None]]:
    """Pick the single candidate token per syllable minimizing Mel-LSD against the real syllable."""
    fb = _mel_filterbank(sr=SR, nfft=N_FFT, n_mels=40)
    target_profile = bank.profiles_v5[tgt_name]
    tgt_f0 = float(target_profile.v3.f0_median_hz)
    mel_real_all = _extract_mel_frames_aligned(xa)

    chosen: list[tuple[str, V13ZhToken | None]] = []
    for idx_s, (syl, (gs_s, gs_sp, gs_e)) in enumerate(zip(u["syllables"], syl_bounds)):
        o, f, tone = cf.pinyin_to_ipa(syl["pinyin"])[:3]
        if o in ("y", "w"):
            o = ""
        if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
            f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"

        nf = max(3, gs_e - gs_s)
        cands = bank.zh_syllables.get((o, f), [])
        if not cands:
            chosen.append(("zh_concat_of", None))
            continue

        # Prefilter candidates by shape-normed stretched Mel-filterbank distance, then pick exact waveform best in top-6
        ref_slice = mel_real_all[gs_s : min(len(mel_real_all), gs_e)]
        if len(ref_slice) < 2:
            chosen.append(("same_spk_same_tone", cands[0]))
            continue
        ref_norm = ref_slice - np.mean(ref_slice, axis=1, keepdims=True)

        scored: list[tuple[float, V13ZhToken]] = []
        for c in cands:
            e_ad, _, _, _ = _adapt_v11(
                c.log_env, c.bap, c.bap_gt8k, c.voiced, c.f0_rel_st,
                _get_src_prof(bank, c.speaker, target_profile, bank.profiles_v5), target_profile, adapt_features_v5, None,
            )
            e_st, _, _, _ = _stretch_preserving_onset(e_ad, c.bap, c.voiced, c.f_split, len(ref_slice), preserve_onset=True)
            pwr = np.exp(2.0 * e_st)
            mel_cand = (10.0 / math.log(10.0)) * np.log(np.maximum(pwr @ fb.T, 1e-12))
            cand_norm = mel_cand - np.mean(mel_cand, axis=1, keepdims=True)
            d = float(np.mean(np.sqrt(np.mean((ref_norm - cand_norm) ** 2, axis=1))))
            scored.append((d, c))
        scored.sort(key=lambda x: x[0])

        # Exact waveform synthesis refinement on top-6 candidates
        x_seg_real = xa[gs_s * HOP : min(len(xa), gs_e * HOP)]
        real_f0_seg = feats_real.f0_hz[gs_s:gs_e]
        real_voi_seg = feats_real.voiced[gs_s:gs_e]
        best_c = scored[0][1]
        best_lsd = float("inf")
        for _, c in scored[:6]:
            e_ad, b_ad, bg_ad, v_ad = _adapt_v11(
                c.log_env, c.bap, c.bap_gt8k, c.voiced, c.f0_rel_st,
                _get_src_prof(bank, c.speaker, target_profile, bank.profiles_v5), target_profile, adapt_features_v5, None,
            )
            e_st, b_st, v_st, _ = _stretch_preserving_onset(e_ad, b_ad, v_ad, c.f_split, nf, preserve_onset=True)
            bg_st = np.clip(_resample_2d(bg_ad, nf), 0.005, 0.995)
            f0_u = _interpolate_oracle_f0_for_syllable(real_f0_seg, real_voi_seg, v_st, tgt_f0)
            ft = vocoder.VocoderFeatures(
                f0_hz=f0_u, voiced=v_st, log_env=e_st, bap=b_st,
                sr=SR, hop_samples=HOP, n_samples=nf * HOP, n_fft=N_FFT,
            )
            ys = synthesize_v12(
                ft, mode="hybrid_lp_hnm", glottal_bank=bank.glottal_banks.get(tgt_name),
                crisp_unvoiced=True, seed=72, phase_dispersion=True, bap_gt8k=bg_st,
            )
            sd = compute_spectral_distances(x_seg_real, ys, sr=SR)
            lsd = sd["mel_lsd_db"] if sd is not None else 99.0
            if lsd < best_lsd:
                best_lsd = lsd
                best_c = c

        bucket = "same_spk_same_tone" if (best_c.speaker == tgt_name and best_c.tone == tone) else (
            "same_spk_diff_tone" if best_c.speaker == tgt_name else "donor_spk"
        )
        chosen.append((bucket, best_c))
    return chosen


def _compute_region_and_bucket_breakdown(
    xa: np.ndarray,
    y: np.ndarray,
    syl_bounds: list[tuple[int, int, int]],
    syl_frame_spans: list[tuple[int, int, str, int]],
) -> dict[str, Any]:
    """Decompose non-DTW frame-aligned Mel-LSD into onset / nucleus / seam ±20 ms and per-source buckets."""
    mel_r = _extract_mel_frames_aligned(xa)
    mel_s = _extract_mel_frames_aligned(y)
    mr_n = mel_r - np.mean(mel_r, axis=1, keepdims=True)
    ms_n = mel_s - np.mean(mel_s, axis=1, keepdims=True)

    onset_errs, nuc_errs, seam_errs = [], [], []
    bucket_errs: dict[str, list[float]] = {}

    for idx, ((gs_s, gs_sp, gs_e), (syn_s, syn_e, bucket, _)) in enumerate(zip(syl_bounds, syl_frame_spans)):
        nf = min(gs_e - gs_s, syn_e - syn_s, len(mr_n) - gs_s, len(ms_n) - syn_s)
        if nf <= 0:
            continue
        fr_d = np.sqrt(np.mean((mr_n[gs_s : gs_s + nf] - ms_n[syn_s : syn_s + nf]) ** 2, axis=1))
        bucket_errs.setdefault(bucket, []).extend(fr_d.tolist())
        sp = max(0, min(nf, gs_sp))
        if sp > 0:
            onset_errs.extend(fr_d[:sp].tolist())
        if nf > sp:
            nuc_errs.extend(fr_d[sp:].tolist())
        if idx > 0:
            seam_errs.extend(fr_d[: min(4, nf)].tolist())
        if idx + 1 < len(syl_bounds):
            seam_errs.extend(fr_d[max(0, nf - 4) : nf].tolist())

    return {
        "onset_mel_lsd_db": round(float(np.mean(onset_errs)), 4) if onset_errs else None,
        "nucleus_mel_lsd_db": round(float(np.mean(nuc_errs)), 4) if nuc_errs else None,
        "seam_20ms_mel_lsd_db": round(float(np.mean(seam_errs)), 4) if seam_errs else None,
        "buckets": {
            b: {"n_frames": len(v), "mean_mel_lsd_db": round(float(np.mean(v)), 4)}
            for b, v in sorted(bucket_errs.items())
        },
    }


def run_t3_sandhi_audit(
    utterances: list[dict[str, Any]],
    feats_cache: dict[str, dict[str, Any]],
    gmm_zh_map: dict[str, Any],
) -> dict[str, Any]:
    """Compare plan_v10 tones against manifest_v12.json across all 112 Mandarin utterances."""
    total_syls = 0
    mismatches: list[dict[str, Any]] = []
    t3_consecutive_acoustic_slopes: list[dict[str, Any]] = []

    for u in utterances:
        uid = u["id"]
        pl = plan_v10(u["text"])
        units = pl.get("units", [])
        syls = u["syllables"]
        if len(units) != len(syls):
            continue
        ge = gmm_zh_map.get(uid)
        feats = feats_cache[uid]["feats"]
        n_tot = len(feats.f0_hz)

        # Extract word groupings implied by text_span / word field
        for i, (pu, mu) in enumerate(zip(units, syls)):
            total_syls += 1
            _, _, man_tone = cf.pinyin_to_ipa(mu["pinyin"])[:3]
            pred_tone = int(pu.get("tone", 1))
            pred_lex = int(pu.get("lexical_tone", pred_tone))
            man_prev_t = cf.pinyin_to_ipa(syls[i - 1]["pinyin"])[2] if i > 0 else None
            man_next_t = cf.pinyin_to_ipa(syls[i + 1]["pinyin"])[2] if i + 1 < len(syls) else None

            # Record acoustic F0 trajectory for all consecutive T3 runs in manifest
            if man_tone == 3 and man_next_t == 3 and ge and i < len(ge.get("syllables", [])):
                gs = ge["syllables"][i]
                s0, s1 = max(0, int(gs["syl_start"])), min(n_tot, int(gs["syl_end"]))
                f_sl = feats.f0_hz[s0:s1]
                v_sl = feats.voiced[s0:s1] & (f_sl > 0)
                if np.sum(v_sl) >= 6:
                    vf = 12.0 * np.log2(f_sl[v_sl])
                    k3 = max(2, len(vf) // 3)
                    slope_st = float(np.median(vf[-k3:]) - np.median(vf[:k3]))
                    t3_consecutive_acoustic_slopes.append({
                        "uid": uid,
                        "idx": i,
                        "hanzi": mu["hanzi"],
                        "pred_tone": pred_tone,
                        "manifest_tone": man_tone,
                        "word": pu.get("word", mu["hanzi"]),
                        "next_hanzi": syls[i + 1]["hanzi"],
                        "next_word": units[i + 1].get("word", syls[i + 1]["hanzi"]),
                        "acoustic_slope_st": round(slope_st, 2),
                    })

            if pred_tone != man_tone:
                same_word_next = (i + 1 < len(units)) and (pu.get("text_span", [0, 0])[1] == units[i + 1].get("text_span", [-1, -1])[0]) and (pu.get("word") == units[i + 1].get("word"))
                if pred_lex == 3 and man_tone == 3 and pred_tone == 2:
                    cat = "t3_sandhi_applied_manifest_lexical"
                elif pred_lex == 3 and man_next_t == 3 and pred_tone == 3:
                    cat = "t3_sandhi_missed_cross_word" if not same_word_next else "t3_sandhi_missed_other"
                elif pu.get("hanzi") in ("一", "不"):
                    cat = "yi_bu_sandhi_vs_manifest"
                else:
                    cat = "polyphone_or_tone_mismatch"

                mismatches.append({
                    "uid": uid,
                    "idx": i,
                    "hanzi": mu["hanzi"],
                    "text": u["text"],
                    "pred_pinyin": pu.get("pinyin"),
                    "pred_tone": pred_tone,
                    "pred_lexical_tone": pred_lex,
                    "manifest_pinyin": mu["pinyin"],
                    "manifest_tone": man_tone,
                    "word": pu.get("word"),
                    "next_word": units[i + 1].get("word") if i + 1 < len(units) else None,
                    "prev_manifest_tone": man_prev_t,
                    "next_manifest_tone": man_next_t,
                    "category": cat,
                })

    cat_counts: dict[str, int] = {}
    for m in mismatches:
        cat_counts[m["category"]] = cat_counts.get(m["category"], 0) + 1

    # Check spotlight_6 specifically
    spot_mismatches = [m for m in mismatches if m["uid"] in SPOTLIGHT_6_IDS]
    rising_count = sum(1 for r in t3_consecutive_acoustic_slopes if r["acoustic_slope_st"] > 0.5)
    return {
        "total_utterances": len(utterances),
        "total_syllables": total_syls,
        "total_tone_mismatches": len(mismatches),
        "category_counts": cat_counts,
        "spotlight_6_mismatches": spot_mismatches,
        "t3_consecutive_pairs_n": len(t3_consecutive_acoustic_slopes),
        "t3_consecutive_rising_f0_count": rising_count,
        "t3_consecutive_samples": t3_consecutive_acoustic_slopes[:12],
        "mismatches_sample": mismatches[:25],
    }


def run_diagnostic() -> dict[str, Any]:
    t0 = time.perf_counter()
    zh_utts = load_zh_manifest_utterances()
    gmm_zh_map = load_gmm_zh_map()
    feats_cache = _analyze_or_load_zh_features(zh_utts, gmm_zh_map, use_cache=True)

    print("[1/3] Running T3 sandhi audit across 112 Mandarin utterances...", flush=True)
    sandhi_audit = run_t3_sandhi_audit(zh_utts, feats_cache, gmm_zh_map)

    print("[2/3] Building strict zero-leakage banks excluding 6 spotlight utterances...", flush=True)
    bank_v12bug_excl = build_v13_zh_bank(exclude_ids=SPOTLIGHT_6_IDS, use_cache=True, fix_f_split=False)
    bank_v13_excl = build_v13_zh_bank(exclude_ids=SPOTLIGHT_6_IDS, use_cache=True, fix_f_split=True)
    prosody_model = load_prosody_v12(DEFAULT_MODEL_PATH)
    tgt_name = "SSB0502"
    tgt_prof = bank_v13_excl.profiles_v5[tgt_name]

    utt_map = {u["id"]: u for u in zh_utts}
    rungs = [
        "rung1_copy_synth_self",
        "rung1b_copy_synth_excl_gb",
        "rung2_same_utt_v12_pipeline_bugfsplit",
        "rung2b_same_utt_v12_pipeline_fixfsplit",
        "rung2c_same_utt_no_gain_no_seam",
        "rung3_oracle_other_utt_v12pipe",
        "rung3b_oracle_other_utt_nogain_noseam",
        "rung4_v12_guided_bugfsplit",
        "rung4b_v12_guided_fixfsplit",
        "rung5_v12_auto",
    ]
    per_rung_rows: dict[str, list[dict[str, Any]]] = {r: [] for r in rungs}

    print("[3/3] Evaluating 5-rung ladder on 6 SSB0502 spotlight utterances...", flush=True)
    for uid in SPOTLIGHT_6_IDS:
        u = utt_map[uid]
        rec = feats_cache[uid]
        xa = np.asarray(rec["xa"], dtype=np.float64)
        feats_real: vocoder.VocoderFeatures = rec["feats"]
        b_gt8 = np.asarray(rec["bap_gt8k"], dtype=np.float64)
        ge = gmm_zh_map[uid]

        syl_bounds: list[tuple[int, int, int]] = []
        syl_specs_v12: list[tuple[str, str, int, float]] = []
        syl_specs_guided: list[tuple[str, str, int, int, np.ndarray, np.ndarray]] = []
        for idx_s, (syl, gs) in enumerate(zip(u["syllables"], ge["syllables"])):
            o, f, tone = cf.pinyin_to_ipa(syl["pinyin"])[:3]
            if o in ("y", "w"):
                o = ""
            if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
                f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"
            gs_s, gs_e, gs_sp = int(gs["syl_start"]), int(gs["syl_end"]), int(gs["f_split"])
            nf = max(3, gs_e - gs_s)
            syl_bounds.append((gs_s, gs_sp, gs_e))
            syl_specs_v12.append((o, f, tone, nf * 5.0))
            syl_specs_guided.append((o, f, tone, nf, feats_real.f0_hz[gs_s:gs_e], feats_real.voiced[gs_s:gs_e]))

        def _score_audio(rung_key: str, y_syn: np.ndarray, lead_fr: int = 0, spans=None, plan_durs: list[float] | None = None) -> None:
            sd = compute_spectral_distances(xa, y_syn, sr=SR)
            fa = _frame_aligned_mel_lsd(xa, y_syn, syn_lead_frames=lead_fr, n_speech_frames=syl_bounds[-1][2])
            f0_fr_err, f0_med_err = _compute_f0_mae_st(xa, y_syn, sr=SR, syn_lead_frames=lead_fr)
            dur_mae = 0.0
            if plan_durs is not None:
                oracle_durs = [(b[2] - b[0]) * 5.0 for b in syl_bounds]
                dur_mae = float(np.mean(np.abs(np.asarray(plan_durs) - np.asarray(oracle_durs))))
            row: dict[str, Any] = {
                "uid": uid,
                "mel_lsd_db": round(sd["mel_lsd_db"], 4) if sd else None,
                "mcd_db": round(sd["mcd_db"], 4) if sd else None,
                "mcd_cmn_db": round(sd["mcd_cmn_db"], 4) if sd else None,
                "nodtw_shape_mel_lsd_db": round(fa["nodtw_shape_mel_lsd_db"], 4),
                "nodtw_raw_mel_lsd_db": round(fa["nodtw_raw_mel_lsd_db"], 4),
                "f0_frame_mae_st": round(f0_fr_err, 4) if math.isfinite(f0_fr_err) else None,
                "f0_med_err_st": round(f0_med_err, 4) if math.isfinite(f0_med_err) else None,
                "dur_mae_ms": round(dur_mae, 2),
            }
            if spans is not None:
                row["breakdown"] = _compute_region_and_bucket_breakdown(xa, y_syn, syl_bounds, spans)
            per_rung_rows[rung_key].append(row)

        # Rung 1: Copy-Synth
        fv12_self = VocoderFeaturesV12(
            feats=feats_real, bap_gt8k=b_gt8, uv_raw_log_env=None,
            lp_pulses=rec["lp_pulses"], coherent_phasor=rec["coherent_phasor"],
        )
        y_r1 = synthesize_v12(fv12_self, mode="hybrid_lp_hnm", crisp_unvoiced=True, seed=72, phase_dispersion=True)
        _score_audio("rung1_copy_synth_self", y_r1, lead_fr=0, spans=[(gs_s, gs_e, "copy_synth", gs_sp) for gs_s, gs_sp, gs_e in syl_bounds])

        y_r1b = synthesize_v12(
            feats_real, mode="hybrid_lp_hnm", glottal_bank=bank_v13_excl.glottal_banks[tgt_name],
            crisp_unvoiced=True, seed=72, phase_dispersion=True, bap_gt8k=b_gt8,
        )
        _score_audio("rung1b_copy_synth_excl_gb", y_r1b, lead_fr=0, spans=[(gs_s, gs_e, "copy_synth", gs_sp) for gs_s, gs_sp, gs_e in syl_bounds])

        # Rung 2: Same-utterance tokens pushed through pipeline
        own_bug, _, _, _ = extract_utterance_zh_tokens(u, xa, feats_real, b_gt8, ge, fix_f_split=False, apply_quality_filter=False)
        own_fix, _, _, _ = extract_utterance_zh_tokens(u, xa, feats_real, b_gt8, ge, fix_f_split=True, apply_quality_filter=False)
        chosen_own_bug = [("same_utt", t) for t in own_bug]
        chosen_own_fix = [("same_utt", t) for t in own_fix]

        y_r2, m_r2 = _render_tokens_guided(syl_specs_guided, chosen_own_bug, bank_v12bug_excl, tgt_name, apply_gain_cal=True, apply_seam_blend=True, apply_smooth_contour=True)
        _score_audio("rung2_same_utt_v12_pipeline_bugfsplit", y_r2, lead_fr=LEAD_SIL_FRAMES, spans=m_r2["syl_frame_spans"])

        y_r2b, m_r2b = _render_tokens_guided(syl_specs_guided, chosen_own_fix, bank_v13_excl, tgt_name, apply_gain_cal=True, apply_seam_blend=True, apply_smooth_contour=True)
        _score_audio("rung2b_same_utt_v12_pipeline_fixfsplit", y_r2b, lead_fr=LEAD_SIL_FRAMES, spans=m_r2b["syl_frame_spans"])

        y_r2c, m_r2c = _render_tokens_guided(syl_specs_guided, chosen_own_fix, bank_v13_excl, tgt_name, apply_gain_cal=False, apply_seam_blend=False, apply_smooth_contour=False)
        _score_audio("rung2c_same_utt_no_gain_no_seam", y_r2c, lead_fr=LEAD_SIL_FRAMES, spans=m_r2c["syl_frame_spans"])

        # Rung 3: Oracle unit selection from OTHER utterances
        chosen_oracle = _oracle_select_for_utterance(u, xa, feats_real, syl_bounds, bank_v13_excl, tgt_name)
        y_r3, m_r3 = _render_tokens_guided(syl_specs_guided, chosen_oracle, bank_v13_excl, tgt_name, apply_gain_cal=True, apply_seam_blend=True, apply_smooth_contour=True)
        _score_audio("rung3_oracle_other_utt_v12pipe", y_r3, lead_fr=LEAD_SIL_FRAMES, spans=m_r3["syl_frame_spans"])

        y_r3b, m_r3b = _render_tokens_guided(syl_specs_guided, chosen_oracle, bank_v13_excl, tgt_name, apply_gain_cal=False, apply_seam_blend=False, apply_smooth_contour=False)
        _score_audio("rung3b_oracle_other_utt_nogain_noseam", y_r3b, lead_fr=LEAD_SIL_FRAMES, spans=m_r3b["syl_frame_spans"])

        # Rung 4: v12 guided selection + oracle dur + oracle F0
        ch_v12_bug = _select_v12_tokens_for_utterance(syl_specs_v12, bank_v12bug_excl, tgt_name)
        y_r4, m_r4 = _render_tokens_guided(syl_specs_guided, ch_v12_bug, bank_v12bug_excl, tgt_name, apply_gain_cal=True, apply_seam_blend=True, apply_smooth_contour=True)
        _score_audio("rung4_v12_guided_bugfsplit", y_r4, lead_fr=LEAD_SIL_FRAMES, spans=m_r4["syl_frame_spans"])

        ch_v12_fix = _select_v12_tokens_for_utterance(syl_specs_v12, bank_v13_excl, tgt_name)
        y_r4b, m_r4b = _render_tokens_guided(syl_specs_guided, ch_v12_fix, bank_v13_excl, tgt_name, apply_gain_cal=True, apply_seam_blend=True, apply_smooth_contour=True)
        _score_audio("rung4b_v12_guided_fixfsplit", y_r4b, lead_fr=LEAD_SIL_FRAMES, spans=m_r4b["syl_frame_spans"])

        # Rung 5: Unmodified v12 auto from text
        pl_auto = plan_v10(u["text"])
        y_r5, m_r5 = render_v12_utterance(
            pl_auto, bank_v12bug_excl, tgt_prof, prosody_model=prosody_model,
            use_learned_prosody=True, cloner_fn=adapt_features_v5, seed=72,
        )
        auto_durs = [float(e - s) for s, e in (m_r5.get("slot_timings_ms") or [])]
        _score_audio("rung5_v12_auto", y_r5, lead_fr=LEAD_SIL_FRAMES, spans=None, plan_durs=auto_durs if len(auto_durs) == len(syl_bounds) else None)

    # Aggregate medians and pooled region/bucket stats
    summary: dict[str, Any] = {}
    for rk in rungs:
        rows = per_rung_rows[rk]
        def _med(k: str) -> float | None:
            vals = [r[k] for r in rows if r.get(k) is not None and math.isfinite(r[k])]
            return round(float(np.median(vals)), 4) if vals else None

        reg_on = [r["breakdown"]["onset_mel_lsd_db"] for r in rows if r.get("breakdown") and r["breakdown"]["onset_mel_lsd_db"] is not None]
        reg_nu = [r["breakdown"]["nucleus_mel_lsd_db"] for r in rows if r.get("breakdown") and r["breakdown"]["nucleus_mel_lsd_db"] is not None]
        reg_se = [r["breakdown"]["seam_20ms_mel_lsd_db"] for r in rows if r.get("breakdown") and r["breakdown"]["seam_20ms_mel_lsd_db"] is not None]
        b_pool: dict[str, list[float]] = {}
        for r in rows:
            if not r.get("breakdown"):
                continue
            for b, info in r["breakdown"]["buckets"].items():
                b_pool.setdefault(b, []).extend([info["mean_mel_lsd_db"]] * info["n_frames"])
        summary[rk] = {
            "mel_lsd_db_median": _med("mel_lsd_db"),
            "mcd_db_median": _med("mcd_db"),
            "mcd_cmn_db_median": _med("mcd_cmn_db"),
            "nodtw_shape_mel_lsd_db_median": _med("nodtw_shape_mel_lsd_db"),
            "nodtw_raw_mel_lsd_db_median": _med("nodtw_raw_mel_lsd_db"),
            "f0_frame_mae_st_median": _med("f0_frame_mae_st"),
            "f0_med_err_st_median": _med("f0_med_err_st"),
            "dur_mae_ms_median": _med("dur_mae_ms"),
            "region_onset_mel_lsd_db": round(float(np.mean(reg_on)), 4) if reg_on else None,
            "region_nucleus_mel_lsd_db": round(float(np.mean(reg_nu)), 4) if reg_nu else None,
            "region_seam_20ms_mel_lsd_db": round(float(np.mean(reg_se)), 4) if reg_se else None,
            "bucket_summary": {
                b: {"n_frames": len(v), "mean_mel_lsd_db": round(float(np.mean(v)), 4)}
                for b, v in sorted(b_pool.items())
            },
        }

    report = {
        "runtime_s": round(time.perf_counter() - t0, 2),
        "spotlight_ids": list(SPOTLIGHT_6_IDS),
        "bank_v12bug_stats": bank_v12bug_excl.stats,
        "bank_v13_stats": bank_v13_excl.stats,
        "summary_5rung": summary,
        "per_utterance": per_rung_rows,
        "t3_sandhi_audit": sandhi_audit,
    }
    out_path = EXP_DIR / "diagnostic_5rung.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 5-RUNG DIAGNOSTIC SUMMARY (6 SSB0502 Spotlight Utterances, Zero Leakage) ===")
    print(f"{'Rung':42s} | {'Mel-LSD':>7s} | {'MCD':>7s} | {'NoDTW-Shp':>9s} | {'F0-FrMAE':>8s} | {'DurMAE':>7s}")
    print("-" * 96)
    for rk in rungs:
        s = summary[rk]
        print(
            f"{rk:42s} | {str(s['mel_lsd_db_median']):>7s} | {str(s['mcd_db_median']):>7s} | "
            f"{str(s['nodtw_shape_mel_lsd_db_median']):>9s} | {str(s['f0_frame_mae_st_median']):>8s} | {str(s['dur_mae_ms_median']):>7s}"
        )
    print("\n=== T3 SANDHI AUDIT (112 Mandarin Utterances) ===")
    print(json.dumps({k: v for k, v in sandhi_audit.items() if k not in ("t3_consecutive_samples", "mismatches_sample")}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run_diagnostic()
