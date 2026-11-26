"""UTAU-style full-resolution (5 ms) concatenative-HNM renderer with synth-fill for unseen phonemes."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pybot_tts_v2_recorded import engine  # noqa: E402
from pybot_tts_v4_params import vocal  # noqa: E402
from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v6_crossling_dsp import arbiter as v6_arb  # noqa: E402
from exp_v6_vocoder import cloner, vocoder  # noqa: E402
from exp_v7_pybot_tts import crossling_warp as cw, unit_bank as ub  # noqa: E402
from exp_v8_clarity import EXTENDED_INDEX, HOP, N_FFT, SR  # noqa: E402


OBSTRUENT_ONSETS = frozenset({
    "b", "p", "d", "t", "g", "k",
    "z", "c", "zh", "ch", "j", "q",
    "f", "s", "sh", "x", "h",
})
OBSTRUENT_IPAS = frozenset({
    "p", "pʰ", "b", "t", "tʰ", "d", "k", "kʰ", "ɡ",
    "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "ʂ", "ɕ", "x", "h", "ʍ",
    "ts", "tsʰ", "tɕ", "tɕʰ", "ʈʂ", "ʈʂʰ", "tʃ", "dʒ",
})
N_BINS = N_FFT // 2 + 1
SILENCE_LOG_ENV = -9.5


@dataclass
class FullResToken:
    onset: str
    final: str
    tone: int
    speaker: str
    utterance_id: str
    prev_final: str
    next_onset: str
    log_env: np.ndarray        # (T, 513) float32
    bap: np.ndarray            # (T, 5) float32
    voiced: np.ndarray         # (T,) bool
    f0_rel_st: np.ndarray      # (T,) float32 relative to speaker's utterance median F0
    f_split: int               # onset->final frame boundary inside [0, T]
    dur_ms: float
    onset_wav: np.ndarray | None = None  # raw unvoiced onset samples [0 : f_split*HOP]


@dataclass
class FullResBank:
    syllables: dict[tuple[str, str], list[FullResToken]] = field(default_factory=dict)
    onsets: dict[str, list[FullResToken]] = field(default_factory=dict)
    finals: dict[str, list[FullResToken]] = field(default_factory=dict)
    spk_profiles: dict[str, cloner.FewShotProfile] = field(default_factory=dict)
    fallback_unit_bank: ub.UnitBank | None = None
    included_ids: list[str] = field(default_factory=list)
    excluded_ids: list[str] = field(default_factory=list)


def _slice_syllable_bounds(
    onset: str,
    dur_s: float,
    feats: vocoder.VocoderFeatures,
    f_lo: int,
    f_hi: int,
) -> int:
    n_fr = f_hi - f_lo
    if not onset or n_fr < 3:
        return f_lo
    _, c_dur = vocal._split_duration(onset, max(0.07, dur_s))
    prop = max(2, min(n_fr - 2, int(round((c_dur / max(1e-4, dur_s)) * n_fr))))
    if onset in OBSTRUENT_ONSETS:
        search_end = min(f_hi - 2, f_lo + max(prop + 3, int(round(0.48 * n_fr))))
        for idx in range(f_lo + 1, search_end):
            if feats.voiced[idx] and (idx + 1 < f_hi and feats.voiced[idx + 1]):
                return idx
    return f_lo + prop


def build_fullres_bank(
    utterances: list[dict] | None = None,
    exclude_ids: Iterable[str] = (),
    precomputed: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures]] | None = None,
    sr: int = SR,
    include_fallback_bank: bool = True,
) -> FullResBank:
    """Build an unaveraged, full 5 ms resolution context-indexed HNM segment bank (zero leakage)."""
    if utterances is None:
        idx = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))
        utterances = idx["utterances"]

    excl = set(exclude_ids)
    train_utts = [u for u in utterances if u["id"] not in excl]
    if not train_utts:
        raise ValueError("No training utterances remain after exclude_ids")

    utt_data: dict[str, tuple[np.ndarray, vocoder.VocoderFeatures]] = {}
    by_spk: dict[str, list[tuple[np.ndarray, vocoder.VocoderFeatures]]] = {}
    for u in train_utts:
        uid = u["id"]
        if precomputed is not None and uid in precomputed:
            x_act, feats = precomputed[uid]
        else:
            x, file_sr = vocoder.read_wav_mono(u["path"])
            if file_sr != sr:
                raise ValueError(f"Expected {sr} Hz, got {file_sr} for {uid}")
            x_act = x[u["active_start"] : u["active_stop"]]
            feats = vocoder.analyze(x_act, sr=sr)
        utt_data[uid] = (x_act, feats)
        by_spk.setdefault(u["speaker"], []).append((x_act, feats))

    spk_profiles = {
        spk: cloner.extract_fewshot_profile(items, sr=sr, name=spk)
        for spk, items in sorted(by_spk.items())
    }

    bank = FullResBank(
        spk_profiles=spk_profiles,
        included_ids=sorted(utt_data.keys()),
        excluded_ids=sorted(excl),
    )
    if include_fallback_bank:
        bank.fallback_unit_bank = ub.build_unit_bank(
            train_utts, exclude_ids=excl, precomputed=utt_data, sr=sr
        )

    for u in train_utts:
        uid = u["id"]
        spk = u["speaker"]
        x_act, feats = utt_data[uid]
        act_start = int(u["active_start"])

        # Gentle utterance-level RMS normalization so concatenated segments match cleanly
        v_mask = feats.voiced & (feats.f0_hz > 0.0)
        env_norm = feats.log_env.copy()
        wav_scale = 1.0
        if np.any(v_mask):
            med_lvl = float(np.median(np.mean(env_norm[v_mask], axis=1)))
            shift = -6.9 - med_lvl
            env_norm = env_norm + shift
            wav_scale = math.exp(shift)
            f0_med = float(np.median(feats.f0_hz[v_mask]))
            f0_rel = np.where(v_mask, 12.0 * np.log2(np.maximum(50.0, feats.f0_hz) / max(50.0, f0_med)), 0.0)
        else:
            f0_rel = np.zeros(len(env_norm), dtype=np.float64)

        syls = u["syllables"]
        parsed = [engine.split_pinyin(s["pinyin"]) for s in syls]
        n_tot = len(env_norm)

        for idx_s, (syl, (o, f, tone)) in enumerate(zip(syls, parsed)):
            f_lo = max(0, min(n_tot - 2, int(round((int(syl["start_sample"]) - act_start) / HOP))))
            f_hi = max(f_lo + 2, min(n_tot, int(round((int(syl["stop_sample"]) - act_start) / HOP))))
            dur_ms = float(syl["ms"])
            f_sp = _slice_syllable_bounds(o, dur_ms / 1000.0, feats, f_lo, f_hi)
            f_split_rel = f_sp - f_lo

            prev_f = parsed[idx_s - 1][1] if idx_s > 0 else "SIL"
            next_o = parsed[idx_s + 1][0] if idx_s + 1 < len(parsed) else "SIL"

            voi_sl = feats.voiced[f_lo:f_hi].copy()
            if o in OBSTRUENT_ONSETS and f_split_rel > 0:
                voi_sl[:f_split_rel] = False

            on_wav = None
            if o in OBSTRUENT_ONSETS and f_split_rel > 0:
                s0 = max(0, f_lo * HOP)
                s1 = min(len(x_act), f_sp * HOP)
                if s1 > s0:
                    on_wav = (x_act[s0:s1] * wav_scale).astype(np.float32)

            tok = FullResToken(
                onset=o,
                final=f,
                tone=tone,
                speaker=spk,
                utterance_id=uid,
                prev_final=prev_f,
                next_onset=next_o,
                log_env=env_norm[f_lo:f_hi].astype(np.float32),
                bap=feats.bap[f_lo:f_hi].astype(np.float32),
                voiced=voi_sl.astype(bool),
                f0_rel_st=f0_rel[f_lo:f_hi].astype(np.float32),
                f_split=int(f_split_rel),
                dur_ms=dur_ms,
                onset_wav=on_wav,
            )
            bank.syllables.setdefault((o, f), []).append(tok)

            if o and f_split_rel > 0:
                o_tok = FullResToken(
                    onset=o, final="", tone=tone, speaker=spk, utterance_id=uid,
                    prev_final=prev_f, next_onset=f,
                    log_env=env_norm[f_lo:f_sp].astype(np.float32),
                    bap=feats.bap[f_lo:f_sp].astype(np.float32),
                    voiced=voi_sl[:f_split_rel].astype(bool),
                    f0_rel_st=f0_rel[f_lo:f_sp].astype(np.float32),
                    f_split=int(f_split_rel),
                    dur_ms=float(f_split_rel * HOP / sr * 1000.0),
                    onset_wav=on_wav,
                )
                bank.onsets.setdefault(o, []).append(o_tok)

            if f_hi > f_sp:
                r_dur = float((f_hi - f_sp) * HOP / sr * 1000.0)
                f_tok = FullResToken(
                    onset="", final=f, tone=tone, speaker=spk, utterance_id=uid,
                    prev_final=o or prev_f, next_onset=next_o,
                    log_env=env_norm[f_sp:f_hi].astype(np.float32),
                    bap=feats.bap[f_sp:f_hi].astype(np.float32),
                    voiced=voi_sl[f_split_rel:].astype(bool),
                    f0_rel_st=f0_rel[f_sp:f_hi].astype(np.float32),
                    f_split=0,
                    dur_ms=r_dur,
                    onset_wav=None,
                )
                bank.finals.setdefault(f, []).append(f_tok)
    return bank


def _score_candidate(
    cand: FullResToken,
    target_speaker: str,
    tone: int,
    prev_final: str,
    next_onset: str,
    target_dur_ms: float,
) -> float:
    s = 0.0
    if cand.speaker == target_speaker:
        s += 4.5  # Target-First priority
    if cand.tone == tone:
        s += 2.0
    if cand.prev_final == prev_final:
        s += 1.5
    if cand.next_onset == next_onset:
        s += 1.5
    s -= 0.6 * abs(math.log(max(20.0, cand.dur_ms) / max(20.0, target_dur_ms)))
    return s


def _adapt_token_if_needed(
    tok: FullResToken,
    bank: FullResBank,
    target_profile: cloner.FewShotProfile,
    vtln_alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Adapt donor token to target speaker ONLY if token does not already come from target speaker."""
    if tok.speaker == target_profile.name:
        return tok.log_env.astype(np.float64), tok.bap.astype(np.float64), tok.voiced.copy(), "target"

    src_prof = bank.spk_profiles.get(tok.speaker)
    if src_prof is None:
        return tok.log_env.astype(np.float64), tok.bap.astype(np.float64), tok.voiced.copy(), "donor_raw"

    t_len = len(tok.log_env)
    dummy_f0 = np.where(tok.voiced, src_prof.f0_median_hz * (2.0 ** (tok.f0_rel_st / 12.0)), 0.0)
    feats = vocoder.VocoderFeatures(
        f0_hz=dummy_f0,
        voiced=tok.voiced.copy(),
        log_env=tok.log_env.astype(np.float64),
        bap=tok.bap.astype(np.float64),
        sr=SR,
        hop_samples=HOP,
        n_samples=t_len * HOP,
        n_fft=N_FFT,
    )
    ad = cloner.adapt_features(
        feats,
        src_prof,
        target_profile,
        vtln_alpha=vtln_alpha,
        use_ltas_transfer=True,
        use_bap_transfer=True,
    )
    return ad.log_env, ad.bap, ad.voiced, "donor_adapted"


def _resample_2d(arr: np.ndarray, n_out: int) -> np.ndarray:
    k = len(arr)
    if k == n_out:
        return arr.astype(np.float64, copy=True)
    if k <= 1:
        return np.repeat(arr.astype(np.float64), n_out, axis=0)
    t_in = np.linspace(0.0, 1.0, k)
    t_out = np.linspace(0.0, 1.0, n_out)
    out = np.empty((n_out, arr.shape[1]), dtype=np.float64)
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(t_out, t_in, arr[:, d])
    return out


def _resample_1d(arr: np.ndarray, n_out: int) -> np.ndarray:
    k = len(arr)
    if k == n_out:
        return arr.astype(np.float64, copy=True)
    if k <= 1:
        return np.repeat(arr.astype(np.float64), n_out)
    return np.interp(np.linspace(0.0, 1.0, n_out), np.linspace(0.0, 1.0, k), arr.astype(np.float64))


def _stretch_preserving_onset(
    env: np.ndarray,
    bap: np.ndarray,
    voi: np.ndarray,
    f_split: int,
    n_target_frames: int,
    preserve_onset: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Keep onset burst & CV release frames unstretched; absorb duration scaling in steady nucleus."""
    n_src = len(env)
    n_target_frames = max(2, int(n_target_frames))
    if not preserve_onset or f_split <= 0 or f_split >= n_src - 2:
        env_r = _resample_2d(env, n_target_frames)
        bap_r = np.clip(_resample_2d(bap, n_target_frames), 0.005, 0.995)
        voi_r = _resample_1d(voi.astype(np.float64), n_target_frames) >= 0.45
        new_split = min(n_target_frames - 1, max(0, int(round(f_split * n_target_frames / max(1, n_src)))))
        return env_r, bap_r, voi_r, new_split

    # Keep onset [0 : f_split] + first 2 frames of CV release [f_split : f_split+2] unstretched if space allows
    keep_head = f_split + min(2, max(0, n_src - f_split - 2))
    if n_target_frames <= keep_head + 2:
        # Very short syllable: shrink onset gently while keeping >= 2 rime frames
        n_on = max(1, min(f_split, n_target_frames - 2))
        env_on = _resample_2d(env[:f_split], n_on)
        bap_on = _resample_2d(bap[:f_split], n_on)
        voi_on = _resample_1d(voi[:f_split].astype(np.float64), n_on) >= 0.45
        n_ri = n_target_frames - n_on
        env_ri = _resample_2d(env[f_split:], n_ri)
        bap_ri = _resample_2d(bap[f_split:], n_ri)
        voi_ri = _resample_1d(voi[f_split:].astype(np.float64), n_ri) >= 0.45
        return (
            np.concatenate([env_on, env_ri], axis=0),
            np.clip(np.concatenate([bap_on, bap_ri], axis=0), 0.005, 0.995),
            np.concatenate([voi_on, voi_ri], axis=0),
            n_on,
        )

    n_tail = n_target_frames - keep_head
    env_head = env[:keep_head].astype(np.float64)
    bap_head = bap[:keep_head].astype(np.float64)
    voi_head = voi[:keep_head].astype(bool)

    env_tail = _resample_2d(env[keep_head:], n_tail)
    bap_tail = _resample_2d(bap[keep_head:], n_tail)
    voi_tail = _resample_1d(voi[keep_head:].astype(np.float64), n_tail) >= 0.45
    return (
        np.concatenate([env_head, env_tail], axis=0),
        np.clip(np.concatenate([bap_head, bap_tail], axis=0), 0.005, 0.995),
        np.concatenate([voi_head, voi_tail], axis=0),
        f_split,
    )


def _blend_seam(
    env_a: np.ndarray,
    env_b: np.ndarray,
    bap_a: np.ndarray,
    bap_b: np.ndarray,
    blend_frames: int,
) -> None:
    """Gentle cosine seam blend ONLY for vowel/sonorant joins (0-1 frame on obstruents)."""
    w = min(blend_frames, len(env_a) // 2, len(env_b) // 2)
    if w < 1:
        return
    mid_env = 0.5 * (env_a[-1] + env_b[0])
    mid_bap = 0.5 * (bap_a[-1] + bap_b[0])
    u_a = 0.5 * (1.0 - np.cos(math.pi * np.linspace(0.0, 0.5, w)))[:, None]
    env_a[-w:] = (1.0 - u_a) * env_a[-w:] + u_a * mid_env[None, :]
    bap_a[-w:] = (1.0 - u_a) * bap_a[-w:] + u_a * mid_bap[None, :]
    u_b = 0.5 * (1.0 - np.cos(math.pi * np.linspace(0.5, 1.0, w)))[:, None]
    env_b[:w] = (1.0 - u_b) * mid_env[None, :] + u_b * env_b[:w]
    bap_b[:w] = (1.0 - u_b) * mid_bap[None, :] + u_b * bap_b[:w]


def _interp_list(vals: list[float], t: float) -> float:
    p = max(0.0, min(1.0, t)) * (len(vals) - 1)
    i = min(int(p), len(vals) - 1)
    j = min(i + 1, len(vals) - 1)
    return vals[i] + (vals[j] - vals[i]) * (p - i)


def render_utau_plan(
    plan_doc: dict,
    bank: FullResBank,
    target_profile: cloner.FewShotProfile,
    *,
    vtln_alpha: float | None = None,
    seed: int = 72,
    phase_dispersion: bool = True,
    preserve_onset: bool = True,
    crisp_seams: bool = True,
    use_waveform_unvoiced: bool = False,
    cep128_blur: bool = False,
) -> tuple[np.ndarray, dict]:
    """Synthesize a Mandarin/English/Code-Switched plan using UTAU full-res concatenation + synth-fill."""
    units = plan_doc.get("units") or []
    if not units:
        raise ValueError("plan_doc has no units")

    # Build unit slots with full-res real selection or IPA synth-fill
    envs: list[np.ndarray] = []
    baps: list[np.ndarray] = []
    vois: list[np.ndarray] = []
    sts: list[np.ndarray] = []
    is_obstruent_head: list[bool] = []
    is_pause_block: list[bool] = []
    onset_wav_overlays: list[tuple[int, int, np.ndarray]] = []  # (start_fr, end_fr, wav)
    source_counts: dict[str, int] = {}

    iqr_scale = max(0.55, min(1.75, float(target_profile.f0_iqr_st) / 3.2))

    # Extract syllable sequence for context matching
    n_u = len(units)
    parsed_units = []
    for u in units:
        lang = u.get("lang", "zh")
        if lang == "zh":
            o = str(u.get("onset", ""))
            f = str(u.get("final", ""))
            if o in ("y", "w"):
                o = ""
            if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
                f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"
            parsed_units.append(("zh", o, f, int(u.get("tone") or 1)))
        else:
            parsed_units.append((lang, "", "", int(u.get("tone") or 1)))

    cur_fr = 0
    # 20 ms lead silence
    lead_fr = 4
    envs.append(np.full((lead_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((lead_fr, 5), dtype=np.float64))
    vois.append(np.zeros(lead_fr, dtype=bool))
    sts.append(np.zeros(lead_fr, dtype=np.float64))
    is_obstruent_head.append(False)
    is_pause_block.append(True)
    cur_fr += lead_fr

    for u_idx, u in enumerate(units):
        lang, o, f, tone = parsed_units[u_idx]
        dur_ms = max(35.0, float(u.get("duration_ms", 200.0)))
        nf = max(2, int(round(dur_ms / 5.0)))
        gain = float(u.get("gain", 0.85))
        pitch_5 = [float(x) for x in u.get("pitch_st", [0.0] * 5)]
        pause_ms = float(u.get("pause_after_ms", 0.0))

        prev_f = parsed_units[u_idx - 1][2] if (u_idx > 0 and parsed_units[u_idx - 1][0] == "zh") else "SIL"
        next_o = parsed_units[u_idx + 1][1] if (u_idx + 1 < n_u and parsed_units[u_idx + 1][0] == "zh") else "SIL"

        if lang == "zh":
            cands = bank.syllables.get((o, f), [])
            if cands:
                best = max(
                    cands,
                    key=lambda c: _score_candidate(c, target_profile.name, tone, prev_f, next_o, dur_ms),
                )
                e_raw, b_raw, v_raw, tag = _adapt_token_if_needed(best, bank, target_profile, vtln_alpha=vtln_alpha)
                mode = f"{tag}_exact_syllable"
                source_counts[mode] = source_counts.get(mode, 0) + 1
                e_st, b_st, v_st, f_sp_new = _stretch_preserving_onset(
                    e_raw, b_raw, v_raw, best.f_split, nf, preserve_onset=preserve_onset
                )
                if use_waveform_unvoiced and best.onset_wav is not None and f_sp_new > 0:
                    onset_wav_overlays.append((cur_fr, cur_fr + f_sp_new, best.onset_wav))
                obs_head = o in OBSTRUENT_ONSETS
            else:
                # Concat onset + final independently (Target-First each), or IPA synth-fill
                parts_e, parts_b, parts_v = [], [], []
                split_pt = 0
                obs_head = o in OBSTRUENT_ONSETS
                if o:
                    o_cands = bank.onsets.get(o, [])
                    if o_cands:
                        bo = max(o_cands, key=lambda c: _score_candidate(c, target_profile.name, tone, prev_f, f, dur_ms * 0.28))
                        eo, bo_b, vo, _ = _adapt_token_if_needed(bo, bank, target_profile, vtln_alpha=vtln_alpha)
                        parts_e.append(eo); parts_b.append(bo_b); parts_v.append(vo)
                        split_pt = len(eo)
                        if use_waveform_unvoiced and bo.onset_wav is not None:
                            onset_wav_overlays.append((cur_fr, cur_fr + min(split_pt, nf - 2), bo.onset_wav))
                    else:
                        o_ipas = cf.PINYIN_ONSET_TO_IPA.get(o, ())
                        for oi in o_ipas:
                            d = cw.lookup_ipa_template(bank.fallback_unit_bank, oi, is_onset=True)
                            ed, bd, vd, _ = _adapt_synth_dict(d, bank.fallback_unit_bank.canonical_profile, target_profile, vtln_alpha)
                            parts_e.append(ed); parts_b.append(bd); parts_v.append(vd)
                            split_pt += len(ed)
                f_cands = bank.finals.get(f, [])
                if f_cands:
                    bf = max(f_cands, key=lambda c: _score_candidate(c, target_profile.name, tone, o or prev_f, next_o, dur_ms * 0.72))
                    ef, bf_b, vf, _ = _adapt_token_if_needed(bf, bank, target_profile, vtln_alpha=vtln_alpha)
                    parts_e.append(ef); parts_b.append(bf_b); parts_v.append(vf)
                    mode = "concat_onset_final"
                else:
                    for fi in cf.PINYIN_FINAL_TO_IPA.get(f, ("ə",)):
                        d = cw.lookup_ipa_template(bank.fallback_unit_bank, fi, is_onset=False)
                        ed, bd, vd, _ = _adapt_synth_dict(d, bank.fallback_unit_bank.canonical_profile, target_profile, vtln_alpha)
                        parts_e.append(ed); parts_b.append(bd); parts_v.append(vd)
                    mode = "ipa_synth_fill"
                source_counts[mode] = source_counts.get(mode, 0) + 1
                e_st, b_st, v_st, _ = _stretch_preserving_onset(
                    np.concatenate(parts_e, axis=0),
                    np.concatenate(parts_b, axis=0),
                    np.concatenate(parts_v, axis=0),
                    split_pt,
                    nf,
                    preserve_onset=preserve_onset,
                )
        else:
            # English / non-Mandarin IPA phones: synth-fill via crossling_warp adapted to target speaker
            ipas = list(u.get("ipa_phones") or ["ə"])
            parts_e, parts_b, parts_v, parts_dur = [], [], [], []
            for ip in ipas:
                d = cw.lookup_ipa_template(bank.fallback_unit_bank, ip, is_onset=False)
                ed, bd, vd, _ = _adapt_synth_dict(d, bank.fallback_unit_bank.canonical_profile, target_profile, vtln_alpha)
                parts_e.append(ed); parts_b.append(bd); parts_v.append(vd); parts_dur.append(max(20.0, d["dur_ms"]))
                source_counts[d["source_mode"]] = source_counts.get(d["source_mode"], 0) + 1
            tot_d = max(1.0, sum(parts_dur))
            res_e, res_b, res_v = [], [], []
            acc = 0
            for idx_p, (ed, bd, vd, pd) in enumerate(zip(parts_e, parts_b, parts_v, parts_dur)):
                if idx_p + 1 == len(parts_dur):
                    kp = max(2, nf - acc)
                else:
                    kp = max(2, min(nf - acc - 2 * (len(parts_dur) - idx_p - 1), int(round(nf * pd / tot_d))))
                res_e.append(_resample_2d(ed, kp))
                res_b.append(np.clip(_resample_2d(bd, kp), 0.005, 0.995))
                res_v.append(_resample_1d(vd.astype(np.float64), kp) >= 0.45)
                acc += kp
            e_st = np.concatenate(res_e, axis=0)
            b_st = np.concatenate(res_b, axis=0)
            v_st = np.concatenate(res_v, axis=0)
            obs_head = ipas[0] in OBSTRUENT_IPAS

        if abs(gain - 0.85) > 0.05:
            e_st = e_st + math.log(max(0.25, min(1.8, gain / 0.85)))

        if cep128_blur:
            from scipy.fft import irfft, rfft
            cep = irfft(e_st, n=N_FFT, axis=1)
            cep[:, 128 : N_FFT - 127] = 0.0
            e_st = rfft(cep, n=N_FFT, axis=1).real

        t_grid = np.linspace(0.0, 1.0, len(e_st))
        st_u = np.array([_interp_list(pitch_5, float(t)) for t in t_grid], dtype=np.float64) * iqr_scale

        envs.append(e_st)
        baps.append(b_st)
        vois.append(v_st)
        sts.append(st_u)
        is_obstruent_head.append(obs_head)
        is_pause_block.append(False)
        cur_fr += len(e_st)

        if pause_ms > 15.0:
            p_fr = max(2, int(round(pause_ms / 5.0)))
            envs.append(np.full((p_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
            baps.append(np.ones((p_fr, 5), dtype=np.float64))
            vois.append(np.zeros(p_fr, dtype=bool))
            sts.append(np.zeros(p_fr, dtype=np.float64))
            is_obstruent_head.append(False)
            is_pause_block.append(True)
            cur_fr += p_fr

    # 30 ms tail silence
    tail_fr = 6
    envs.append(np.full((tail_fr, N_BINS), SILENCE_LOG_ENV, dtype=np.float64))
    baps.append(np.ones((tail_fr, 5), dtype=np.float64))
    vois.append(np.zeros(tail_fr, dtype=bool))
    sts.append(np.zeros(tail_fr, dtype=np.float64))
    is_obstruent_head.append(False)
    is_pause_block.append(True)

    # Manner-aware seam blending: ZERO envelope blend across V/UV & obstruent boundaries;
    # 10 ms (2-frame) cosine blend ONLY across voiced vowel/sonorant joins.
    for i in range(len(envs) - 1):
        if is_pause_block[i] or is_pause_block[i + 1]:
            continue
        if crisp_seams:
            if not is_obstruent_head[i + 1]:
                _blend_seam(envs[i], envs[i + 1], baps[i], baps[i + 1], blend_frames=2)
        else:
            # v7-style heavy 25 ms symmetric spline everywhere (ablation comparison)
            _blend_seam(envs[i], envs[i + 1], baps[i], baps[i + 1], blend_frames=5)

    env_all = np.concatenate(envs, axis=0)
    bap_all = np.clip(np.concatenate(baps, axis=0), 0.005, 0.995)
    voi_all = np.concatenate(vois, axis=0)
    st_all = np.concatenate(sts, axis=0)

    # Center pitch contour on target speaker's f0_median_hz + gentle phrase declination
    v_idx = np.where(voi_all)[0]
    if v_idx.size > 0:
        st_centered = st_all[v_idx] - float(np.median(st_all[v_idx]))
        # Subtle clause declination (+0.5 -> -0.5 st)
        prog = np.linspace(0.5, -0.5, len(st_all))[v_idx]
        st_v = st_centered + prog
        # Smooth F0 over 3 frames (15 ms) and clamp abrupt jumps
        if len(st_v) >= 3:
            sm = st_v.copy()
            sm[1:-1] = 0.25 * st_v[:-2] + 0.50 * st_v[1:-1] + 0.25 * st_v[2:]
            st_v = sm
        f0_all = np.zeros(len(st_all), dtype=np.float64)
        f0_all[v_idx] = np.clip(target_profile.f0_median_hz * (2.0 ** (st_v / 12.0)), 65.0, 440.0)
    else:
        f0_all = np.zeros(len(st_all), dtype=np.float64)

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
    y = vocoder.synthesize(feats, seed=seed, phase_dispersion=phase_dispersion)

    # Optional hybrid direct waveform splice for unvoiced obstruent bursts/fricatives
    if use_waveform_unvoiced and onset_wav_overlays:
        fade_n = 48  # 2 ms crossfade
        for fr_s, fr_e, on_wav in onset_wav_overlays:
            s0 = fr_s * HOP
            s1 = min(len(y), fr_e * HOP)
            seg_len = s1 - s0
            if seg_len < 60 or len(on_wav) < 60:
                continue
            src = on_wav.astype(np.float64)
            if len(src) >= seg_len:
                # Take the tail (containing the release burst!) of src
                patch = src[-seg_len:].copy()
            else:
                patch = np.interp(np.linspace(0, 1, seg_len), np.linspace(0, 1, len(src)), src)
            # Match RMS to the synthesized onset window so burst level sits naturally
            rms_syn = float(np.sqrt(np.mean(y[s0:s1] ** 2)))
            rms_pat = float(np.sqrt(np.mean(patch ** 2)))
            if rms_syn > 1e-5 and rms_pat > 1e-5:
                patch = patch * (rms_syn / rms_pat)
            # 2 ms crossfade at head & tail
            fn = min(fade_n, seg_len // 3)
            if fn > 0:
                w_in = np.linspace(0.0, 1.0, fn)
                y[s0 : s0 + fn] = (1.0 - w_in) * y[s0 : s0 + fn] + w_in * patch[:fn]
                y[s0 + fn : s1 - fn] = patch[fn : seg_len - fn]
                w_out = np.linspace(1.0, 0.0, fn)
                y[s1 - fn : s1] = (1.0 - w_out) * y[s1 - fn : s1] + w_out * patch[seg_len - fn :]

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.89:
        y = y * (0.89 / peak)
    return y, {"source_counts": source_counts, "n_frames": len(env_all)}


def _adapt_synth_dict(
    d: dict,
    canon_profile: cloner.FewShotProfile,
    target_profile: cloner.FewShotProfile,
    vtln_alpha: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    # v7 unit_bank normalizes voiced level to -3.8 nepers vs real speech ~-6.9 nepers
    env = d["log_env"].astype(np.float64) - 3.1
    bap = d["bap"].astype(np.float64)
    voi = d["voiced"].astype(bool)
    f0 = np.where(voi, canon_profile.f0_median_hz, 0.0)
    feats = vocoder.VocoderFeatures(
        f0_hz=f0, voiced=voi, log_env=env, bap=bap,
        sr=SR, hop_samples=HOP, n_samples=len(env) * HOP, n_fft=N_FFT,
    )
    ad = cloner.adapt_features(feats, canon_profile, target_profile, vtln_alpha=vtln_alpha)
    return ad.log_env, ad.bap, ad.voiced, d["source_mode"]
