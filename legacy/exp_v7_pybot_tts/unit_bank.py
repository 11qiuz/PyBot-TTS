"""Speaker-normalized HNM (log_env, bap, voiced) unit template bank extracted from training speech."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
from typing import Iterable

from scipy.fft import irfft, rfft
import numpy as np

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pybot_tts_v2_recorded import engine  # noqa: E402
from pybot_tts_v4_params import vocal  # noqa: E402
from exp_v5_arch.crossling_frontend import PINYIN_FINAL_TO_IPA, PINYIN_ONSET_TO_IPA  # noqa: E402
from exp_v6_vocoder.cloner import FewShotProfile, _cepstral_smooth_1d, extract_fewshot_profile  # noqa: E402
from exp_v6_vocoder.vocoder import VocoderFeatures, analyze, read_wav_mono  # noqa: E402
from exp_v7_pybot_tts import EXTENDED_INDEX, HOP, N_FFT, SR  # noqa: E402


K_ONSET = 6
K_FINAL = 10
K_PHONE = 8
K_SYLLABLE = 14
_CEP_KEEP = 128  # compact cepstral representation (~1 MB total for full 32-utt bank, <0.01 neper error)


@dataclass
class UnitTemplate:
    key: str
    log_env: np.ndarray        # (K, 513) float32 speaker-normalized
    bap: np.ndarray            # (K, 5) float32
    voiced_ratio: np.ndarray   # (K,) float32 in [0, 1]
    dur_ms: float
    count: int


@dataclass
class UnitBank:
    syllables: dict[tuple[str, str], UnitTemplate] = field(default_factory=dict)
    onsets: dict[str, UnitTemplate] = field(default_factory=dict)
    finals: dict[str, UnitTemplate] = field(default_factory=dict)
    ipa_phones: dict[str, UnitTemplate] = field(default_factory=dict)
    canonical_profile: FewShotProfile | None = None
    included_ids: list[str] = field(default_factory=list)
    excluded_ids: list[str] = field(default_factory=list)
    sr: int = SR
    hop_samples: int = HOP
    n_fft: int = N_FFT


def _resample_frames(
    log_env: np.ndarray,
    bap: np.ndarray,
    voiced: np.ndarray,
    k_target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample (T, 513), (T, 5), (T,) onto k_target evenly spaced frames."""
    t_len = len(log_env)
    if t_len == 0:
        return (
            np.zeros((k_target, log_env.shape[1]), dtype=np.float32),
            np.ones((k_target, bap.shape[1]), dtype=np.float32),
            np.zeros(k_target, dtype=np.float32),
        )
    if t_len == 1:
        return (
            np.repeat(log_env.astype(np.float32), k_target, axis=0),
            np.repeat(bap.astype(np.float32), k_target, axis=0),
            np.repeat(voiced.astype(np.float32), k_target, axis=0),
        )
    u_src = np.linspace(0.0, 1.0, t_len)
    u_dst = np.linspace(0.0, 1.0, k_target)
    # Vectorized linear interpolation along time axis
    idx_hi = np.searchsorted(u_src, u_dst, side="right")
    idx_hi = np.clip(idx_hi, 1, t_len - 1)
    idx_lo = idx_hi - 1
    denom = np.maximum(1e-9, u_src[idx_hi] - u_src[idx_lo])
    w = ((u_dst - u_src[idx_lo]) / denom)[:, None]

    env_r = ((1.0 - w) * log_env[idx_lo] + w * log_env[idx_hi]).astype(np.float32)
    bap_r = ((1.0 - w) * bap[idx_lo] + w * bap[idx_hi]).astype(np.float32)
    voi_r = ((1.0 - w[:, 0]) * voiced[idx_lo].astype(np.float32) + w[:, 0] * voiced[idx_hi].astype(np.float32)).astype(np.float32)
    return env_r, bap_r, voi_r


def _aggregate_templates(
    key: str,
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
) -> UnitTemplate:
    """Aggregate multiple resampled occurrences using median + cepstral-128 smoothing."""
    envs = np.stack([s[0] for s in samples], axis=0)
    baps = np.stack([s[1] for s in samples], axis=0)
    vois = np.stack([s[2] for s in samples], axis=0)
    durs = [s[3] for s in samples]
    med_env = np.median(envs, axis=0).astype(np.float64)
    # Smooth piecewise-median corners at q < _CEP_KEEP so save/load roundtrip is lossless
    cep = irfft(med_env, n=N_FFT, axis=1)
    cep_lp = np.zeros_like(cep)
    cep_lp[:, :_CEP_KEEP] = cep[:, :_CEP_KEEP]
    cep_lp[:, N_FFT - _CEP_KEEP + 1 :] = cep[:, 1:_CEP_KEEP][:, ::-1]
    smooth_env = rfft(cep_lp, n=N_FFT, axis=1).real.astype(np.float32)
    return UnitTemplate(
        key=key,
        log_env=smooth_env,
        bap=np.clip(np.median(baps, axis=0), 0.005, 0.99).astype(np.float32),
        voiced_ratio=np.clip(np.mean(vois, axis=0), 0.0, 1.0).astype(np.float32),
        dur_ms=float(np.median(durs)),
        count=len(samples),
    )


def _split_syllable_frames(
    onset: str,
    final: str,
    dur_s: float,
    feats: VocoderFeatures,
    f_lo: int,
    f_hi: int,
) -> tuple[int, int, int]:
    """Return (f_lo, f_split, f_hi) splitting onset and final frames inside syllable."""
    n_fr = f_hi - f_lo
    if not onset or n_fr < 3:
        return f_lo, f_lo, f_hi

    _, c_dur = vocal._split_duration(onset, max(0.07, dur_s))
    prop_frames = max(2, min(n_fr - 2, int(round((c_dur / max(1e-4, dur_s)) * n_fr))))

    # For obstruent onsets, snap to the first sustained voiced frame if plausible
    if onset not in ("m", "n", "l", "r"):
        search_end = min(f_hi - 2, f_lo + max(prop_frames + 3, int(round(0.48 * n_fr))))
        for idx in range(f_lo + 1, search_end):
            if feats.voiced[idx] and (idx + 1 < f_hi and feats.voiced[idx + 1]):
                return f_lo, idx, f_hi
    return f_lo, f_lo + prop_frames, f_hi


def _canonical_profile(spk_profiles: dict[str, FewShotProfile]) -> FewShotProfile:
    """Average FewShotProfile across training speakers to define speaker-neutral canonical space."""
    profs = list(spk_profiles.values())
    if len(profs) == 1:
        p = profs[0]
        return FewShotProfile(
            name="canonical",
            n_utterances=p.n_utterances,
            total_voiced_s=p.total_voiced_s,
            f0_median_hz=p.f0_median_hz,
            f0_log_std=p.f0_log_std,
            f0_iqr_st=p.f0_iqr_st,
            tilt_db_per_oct=p.tilt_db_per_oct,
            hnr_median_db=p.hnr_median_db,
            voiced_env_mean=p.voiced_env_mean.copy(),
            bap_voiced_median=p.bap_voiced_median.copy(),
            vtln_anchor_f2_hz=p.vtln_anchor_f2_hz,
        )
    return FewShotProfile(
        name="canonical",
        n_utterances=sum(p.n_utterances for p in profs),
        total_voiced_s=sum(p.total_voiced_s for p in profs),
        f0_median_hz=float(np.mean([p.f0_median_hz for p in profs])),
        f0_log_std=float(np.mean([p.f0_log_std for p in profs])),
        f0_iqr_st=float(np.mean([p.f0_iqr_st for p in profs])),
        tilt_db_per_oct=float(np.mean([p.tilt_db_per_oct for p in profs])),
        hnr_median_db=float(np.mean([p.hnr_median_db for p in profs])),
        voiced_env_mean=np.mean([p.voiced_env_mean for p in profs], axis=0),
        bap_voiced_median=np.median([p.bap_voiced_median for p in profs], axis=0),
        vtln_anchor_f2_hz=float(np.mean([p.vtln_anchor_f2_hz for p in profs])),
    )


def build_unit_bank(
    utterances: list[dict] | None = None,
    exclude_ids: Iterable[str] = (),
    precomputed: dict[str, tuple[np.ndarray, VocoderFeatures]] | None = None,
    sr: int = SR,
) -> UnitBank:
    """Build a speaker-normalized HNM template bank strictly excluding ``exclude_ids``."""
    if utterances is None:
        idx = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))
        utterances = idx["utterances"]

    excl = set(exclude_ids)
    train_utts = [u for u in utterances if u["id"] not in excl]
    if not train_utts:
        raise ValueError("No training utterances remain after applying exclude_ids")

    # 1. Analyze training utterances (using precomputed single-file features if provided)
    utt_data: dict[str, tuple[np.ndarray, VocoderFeatures]] = {}
    by_spk: dict[str, list[tuple[np.ndarray, VocoderFeatures]]] = {}
    for u in train_utts:
        uid = u["id"]
        if precomputed is not None and uid in precomputed:
            x_act, feats = precomputed[uid]
        else:
            x, file_sr = read_wav_mono(u["path"])
            if file_sr != sr:
                raise ValueError(f"Expected {sr} Hz, got {file_sr} for {uid}")
            x_act = x[u["active_start"] : u["active_stop"]]
            feats = analyze(x_act, sr=sr)
        utt_data[uid] = (x_act, feats)
        by_spk.setdefault(u["speaker"], []).append((x_act, feats))

    # 2. Fit training-only speaker profiles & canonical neutral profile
    spk_profiles = {
        spk: extract_fewshot_profile(items, sr=sr, name=spk)
        for spk, items in sorted(by_spk.items())
    }
    canon = _canonical_profile(spk_profiles)

    # Smooth de-speakerizing spectral tilt delta per training speaker (voiced frames)
    spk_deltas: dict[str, np.ndarray] = {}
    for spk, prof in spk_profiles.items():
        raw_diff = canon.voiced_env_mean - prof.voiced_env_mean
        spk_deltas[spk] = np.clip(_cepstral_smooth_1d(raw_diff, n_fft=N_FFT, q_max=24), -1.0, 1.0)

    # 3. Extract speaker-normalized syllable, onset, final, and IPA phone templates
    syl_buckets: dict[tuple[str, str], list] = {}
    onset_buckets: dict[str, list] = {}
    final_buckets: dict[str, list] = {}
    ipa_buckets: dict[str, list] = {}

    for u in train_utts:
        uid = u["id"]
        spk = u["speaker"]
        _, feats = utt_data[uid]
        act_start = int(u["active_start"])

        # De-speakerize log_env on voiced frames + loudness-normalize utterance median voiced level
        env_norm = feats.log_env.copy()
        v_mask = feats.voiced & (feats.f0_hz > 0.0)
        if np.any(v_mask):
            delta = spk_deltas[spk]
            pre_pwr = np.sum(np.exp(2.0 * env_norm[v_mask]), axis=1)
            shifted = env_norm[v_mask] + delta[None, :]
            post_pwr = np.sum(np.exp(2.0 * shifted), axis=1)
            adj = 0.5 * np.log(np.maximum(pre_pwr, 1e-18) / np.maximum(post_pwr, 1e-18))
            env_norm[v_mask] = shifted + adj[:, None]

            # Normalize utterance RMS level so templates stitch at consistent loudness
            med_lvl = float(np.median(np.mean(env_norm[v_mask], axis=1)))
            target_lvl = -3.8  # canonical log-amplitude level matching typical active speech
            env_norm += (target_lvl - med_lvl)

        n_tot_fr = len(env_norm)
        for syl in u["syllables"]:
            o, f, _t = engine.split_pinyin(syl["pinyin"])
            f_lo = max(0, min(n_tot_fr - 2, int(round((int(syl["start_sample"]) - act_start) / HOP))))
            f_hi = max(f_lo + 2, min(n_tot_fr, int(round((int(syl["stop_sample"]) - act_start) / HOP))))
            dur_ms = float(syl["ms"])

            _, f_split, _ = _split_syllable_frames(o, f, dur_ms / 1000.0, feats, f_lo, f_hi)

            # Syllable prototype
            s_env, s_bap, s_voi = _resample_frames(
                env_norm[f_lo:f_hi], feats.bap[f_lo:f_hi], feats.voiced[f_lo:f_hi], K_SYLLABLE
            )
            syl_buckets.setdefault((o, f), []).append((s_env, s_bap, s_voi, dur_ms))

            # Onset prototype + IPA onset
            if o and f_split > f_lo:
                o_dur = max(12.0, (f_split - f_lo) * HOP / sr * 1000.0)
                o_voi_slice = feats.voiced[f_lo:f_split].copy()
                if o not in ("m", "n", "l", "r"):
                    # Mandarin stops, affricates, and fricatives are phonologically voiceless;
                    # any voiced flag inside the onset slice is ACF window overhang from neighbors.
                    o_voi_slice[:] = False
                o_env, o_bap, o_voi = _resample_frames(
                    env_norm[f_lo:f_split], feats.bap[f_lo:f_split], o_voi_slice, K_ONSET
                )
                onset_buckets.setdefault(o, []).append((o_env, o_bap, o_voi, o_dur))
                o_ipas = PINYIN_ONSET_TO_IPA.get(o, ())
                if len(o_ipas) == 1:
                    ip_env, ip_bap, ip_voi = _resample_frames(
                        env_norm[f_lo:f_split], feats.bap[f_lo:f_split], o_voi_slice, K_PHONE
                    )
                    ipa_buckets.setdefault(o_ipas[0], []).append((ip_env, ip_bap, ip_voi, o_dur))

            # Final prototype + constituent IPA phones
            if f_hi > f_split:
                fin_len = f_hi - f_split
                f_dur = fin_len * HOP / sr * 1000.0
                fn_env, fn_bap, fn_voi = _resample_frames(
                    env_norm[f_split:f_hi], feats.bap[f_split:f_hi], feats.voiced[f_split:f_hi], K_FINAL
                )
                final_buckets.setdefault(f, []).append((fn_env, fn_bap, fn_voi, f_dur))

                f_ipas = PINYIN_FINAL_TO_IPA.get(f, ())
                if len(f_ipas) == 1:
                    ip_env, ip_bap, ip_voi = _resample_frames(
                        env_norm[f_split:f_hi], feats.bap[f_split:f_hi], feats.voiced[f_split:f_hi], K_PHONE
                    )
                    ipa_buckets.setdefault(f_ipas[0], []).append((ip_env, ip_bap, ip_voi, f_dur))
                elif len(f_ipas) > 1 and fin_len >= len(f_ipas):
                    # Proportional split across glide / nucleus / nasal coda
                    if len(f_ipas) == 2:
                        weights = (0.28, 0.72) if f_ipas[0] in ("j", "w", "ɥ") else (
                            (0.68, 0.32) if f_ipas[1] in ("n", "ŋ") else (0.50, 0.50)
                        )
                    else:
                        weights = (0.22, 0.48, 0.30)
                    bounds = [f_split]
                    acc = 0.0
                    for w_seg in weights[:-1]:
                        acc += w_seg
                        bounds.append(min(f_hi - 1, max(bounds[-1] + 1, f_split + int(round(acc * fin_len)))))
                    bounds.append(f_hi)
                    for idx_p, ipa_sym in enumerate(f_ipas):
                        p0, p1 = bounds[idx_p], bounds[idx_p + 1]
                        if p1 <= p0:
                            continue
                        p_dur = max(12.0, (p1 - p0) * HOP / sr * 1000.0)
                        ip_env, ip_bap, ip_voi = _resample_frames(
                            env_norm[p0:p1], feats.bap[p0:p1], feats.voiced[p0:p1], K_PHONE
                        )
                        ipa_buckets.setdefault(ipa_sym, []).append((ip_env, ip_bap, ip_voi, p_dur))

    bank = UnitBank(
        canonical_profile=canon,
        included_ids=sorted(utt_data.keys()),
        excluded_ids=sorted(excl),
        sr=sr,
        hop_samples=HOP,
        n_fft=N_FFT,
    )
    for (o, f), rows in sorted(syl_buckets.items()):
        bank.syllables[(o, f)] = _aggregate_templates(f"{o}|{f}", rows)
    for o, rows in sorted(onset_buckets.items()):
        bank.onsets[o] = _aggregate_templates(o, rows)
    for f, rows in sorted(final_buckets.items()):
        bank.finals[f] = _aggregate_templates(f, rows)
    for ipa_sym, rows in sorted(ipa_buckets.items()):
        bank.ipa_phones[ipa_sym] = _aggregate_templates(ipa_sym, rows)
    return bank


def save_bank(bank: UnitBank, path_stem: str | Path) -> dict[str, int]:
    """Save bank to compressed ``<stem>.npz`` (float16 envelopes) + ``<stem>.json`` metadata."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    npz_path = stem.with_suffix(".npz")
    json_path = stem.with_suffix(".json")

    arrays = {}
    meta_groups = {"syllables": [], "onsets": [], "finals": [], "ipa_phones": []}

    def _pack(group: str, items: list[tuple[str, UnitTemplate]]):
        for idx, (k_str, tmpl) in enumerate(items):
            tag = f"{group}_{idx}"
            # Convert smooth (K, 513) log_env to compact (K, 96) real cepstrum (5.3x smaller, ~exact)
            cep = irfft(tmpl.log_env.astype(np.float64), n=bank.n_fft, axis=1)[:, :_CEP_KEEP]
            arrays[f"{tag}_cep"] = cep.astype(np.float32)
            arrays[f"{tag}_bap"] = np.clip(np.round(tmpl.bap * 1000.0), 0, 1000).astype(np.uint16)
            arrays[f"{tag}_voi"] = np.clip(np.round(tmpl.voiced_ratio * 1000.0), 0, 1000).astype(np.uint16)
            meta_groups[group].append({
                "key": k_str,
                "tag": tag,
                "dur_ms": round(tmpl.dur_ms, 2),
                "count": int(tmpl.count),
            })

    _pack("syllables", [(f"{o}|{f}", t) for (o, f), t in sorted(bank.syllables.items())])
    _pack("onsets", sorted(bank.onsets.items()))
    _pack("finals", sorted(bank.finals.items()))
    _pack("ipa_phones", sorted(bank.ipa_phones.items()))

    canon = bank.canonical_profile
    if canon is not None:
        arrays["canon_env"] = canon.voiced_env_mean.astype(np.float32)
        arrays["canon_bap"] = canon.bap_voiced_median.astype(np.float32)

    np.savez_compressed(npz_path, **arrays)
    meta = {
        "schema": 3,
        "cep_keep": _CEP_KEEP,
        "sr": bank.sr,
        "hop_samples": bank.hop_samples,
        "n_fft": bank.n_fft,
        "included_ids": bank.included_ids,
        "excluded_ids": bank.excluded_ids,
        "groups": meta_groups,
        "canonical_profile": None if canon is None else {
            "name": canon.name,
            "n_utterances": canon.n_utterances,
            "total_voiced_s": round(canon.total_voiced_s, 3),
            "f0_median_hz": round(canon.f0_median_hz, 3),
            "f0_log_std": round(canon.f0_log_std, 5),
            "f0_iqr_st": round(canon.f0_iqr_st, 4),
            "tilt_db_per_oct": round(canon.tilt_db_per_oct, 4),
            "hnr_median_db": round(canon.hnr_median_db, 4),
            "vtln_anchor_f2_hz": round(canon.vtln_anchor_f2_hz, 2),
        },
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    npz_bytes = npz_path.stat().st_size
    json_bytes = json_path.stat().st_size
    return {"npz_bytes": npz_bytes, "json_bytes": json_bytes, "total_bytes": npz_bytes + json_bytes}


def load_bank(path_stem: str | Path) -> UnitBank:
    """Load UnitBank from ``<stem>.npz`` + ``<stem>.json``."""
    stem = Path(path_stem)
    npz = np.load(stem.with_suffix(".npz"))
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))

    canon = None
    cp = meta.get("canonical_profile")
    if cp is not None:
        canon = FewShotProfile(
            name=cp["name"],
            n_utterances=cp["n_utterances"],
            total_voiced_s=cp["total_voiced_s"],
            f0_median_hz=cp["f0_median_hz"],
            f0_log_std=cp["f0_log_std"],
            f0_iqr_st=cp["f0_iqr_st"],
            tilt_db_per_oct=cp["tilt_db_per_oct"],
            hnr_median_db=cp["hnr_median_db"],
            voiced_env_mean=npz["canon_env"].astype(np.float64),
            bap_voiced_median=npz["canon_bap"].astype(np.float64),
            vtln_anchor_f2_hz=cp["vtln_anchor_f2_hz"],
        )

    bank = UnitBank(
        canonical_profile=canon,
        included_ids=meta["included_ids"],
        excluded_ids=meta["excluded_ids"],
        sr=meta["sr"],
        hop_samples=meta["hop_samples"],
        n_fft=meta["n_fft"],
    )

    cep_keep = int(meta.get("cep_keep", 0))
    n_fft_bank = int(meta["n_fft"])

    def _unpack(group: str) -> dict:
        out = {}
        for row in meta["groups"][group]:
            tag = row["tag"]
            if cep_keep > 0 and f"{tag}_cep" in npz:
                c_part = npz[f"{tag}_cep"].astype(np.float64)
                c_full = np.zeros((c_part.shape[0], n_fft_bank), dtype=np.float64)
                c_full[:, :cep_keep] = c_part
                c_full[:, n_fft_bank - cep_keep + 1 :] = c_part[:, 1:][:, ::-1]
                env_arr = rfft(c_full, n=n_fft_bank, axis=1).real.astype(np.float32)
            else:
                env_arr = npz[f"{tag}_env"].astype(np.float32)
            bap_arr = npz[f"{tag}_bap"].astype(np.float32) / 1000.0
            voi_arr = npz[f"{tag}_voi"].astype(np.float32) / 1000.0
            tmpl = UnitTemplate(
                key=row["key"],
                log_env=env_arr,
                bap=bap_arr,
                voiced_ratio=voi_arr,
                dur_ms=float(row["dur_ms"]),
                count=int(row["count"]),
            )
            if group == "syllables":
                o, f = row["key"].split("|", 1)
                out[(o, f)] = tmpl
            else:
                out[row["key"]] = tmpl
        return out

    bank.syllables = _unpack("syllables")
    bank.onsets = _unpack("onsets")
    bank.finals = _unpack("finals")
    bank.ipa_phones = _unpack("ipa_phones")
    return bank
