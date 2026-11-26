"""Phase-5 non-neural UTAU-parameterized & manner-aware timbre cloner (`TimbreClonerV3`).

Combines:
1. Low-dimensional, physically grounded UTAU voice-color descriptor (< 2 KB serialized):
   - ``GEN``: vocal-tract size / F1-F2-F3 geometric ``auto_vtln``
   - ``OPE``: glottal open quotient (harmonic H1-H2 & 100-800 vs 800-2500 Hz balance)
   - ``BRI``: formant-peak sharpness (cepstral q=8..28 prominence) & 2-6 kHz brightness
   - ``BRE``: voiced 5-band logit aperiodicity (breathiness) & HNR
2. Cross-corpus microphone/channel EQ decoupling + controlled global tilt anchor (``eq_decoupler.py``).
3. Utterance-normalized acoustic manner partitioning with shrinkage toward global delta.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from exp_v5_eval import metrics as eval_m
from exp_v6_vocoder import cloner as v6_cloner, vocoder as v6_voc
from exp_v8_timbre_clone.timbre_cloner_v2 import TimbreProfileV2, estimate_auto_vtln
from exp_v9_timbre_v3 import HOP_SAMPLES, N_FFT, SR
from exp_v9_timbre_v3.eq_decoupler import (
    DB_PER_NP,
    decompose_envelope_eq,
    linear_tilt_fit,
    normalize_env_shared_band,
)

Q_ENV_STORE = 32
Q_MANNER_STORE = 14
MANNER_NAMES = ("sonorant_nasal", "vowel_open", "vowel_front_high")


def _ceps_encode(vec: np.ndarray, n_fft: int = N_FFT, q_max: int = Q_ENV_STORE) -> list[float]:
    c = np.real(irfft(vec, n=n_fft))
    return [round(float(x), 5) for x in c[: q_max + 1]]


def _ceps_decode(coefs: list[float] | np.ndarray, n_fft: int = N_FFT) -> np.ndarray:
    c_arr = np.asarray(coefs, dtype=np.float64)
    q_max = len(c_arr) - 1
    spec = np.zeros(n_fft, dtype=np.float64)
    spec[0] = c_arr[0]
    if q_max > 0:
        spec[1 : q_max + 1] = c_arr[1:]
        spec[n_fft - q_max :] = c_arr[1:][::-1]
    return np.real(rfft(spec, n=n_fft))


def _measure_h1_h2_db(
    wav: np.ndarray,
    f0_hz: np.ndarray,
    v_mask: np.ndarray,
    sr: int = SR,
    hop_samples: int = HOP_SAMPLES,
) -> float:
    """Measure harmonic H1-H2 (dB) on steady voiced frames via narrowband 2048-pt STFT."""
    idx_v = np.where(v_mask & (f0_hz >= 65.0) & (f0_hz <= 360.0))[0]
    if idx_v.size < 4:
        return 4.0
    if idx_v.size > 48:
        step = max(1, idx_v.size // 48)
        idx_v = idx_v[::step][:48]
    win_len = 2048
    half = win_len // 2
    win = np.hanning(win_len)
    freqs_stft = rfftfreq(win_len, 1.0 / sr)
    df = sr / win_len
    diffs: list[float] = []
    for fi in idx_v:
        center = int(fi * hop_samples)
        lo = center - half
        hi = center + half
        if lo < 0 or hi > len(wav):
            continue
        seg = wav[lo:hi] * win
        mag = np.abs(rfft(seg)) + 1e-9
        f0 = float(f0_hz[fi])
        # Peak search around H1 (0.85..1.15 * f0) and H2 (1.80..2.20 * f0)
        m1 = (freqs_stft >= 0.85 * f0) & (freqs_stft <= 1.15 * f0)
        m2 = (freqs_stft >= 1.80 * f0) & (freqs_stft <= 2.20 * f0)
        if not np.any(m1) or not np.any(m2):
            continue
        # Ensure bin separation covers bandwidth cleanly
        if 0.85 * f0 < df:
            continue
        h1_db = 20.0 * math.log10(float(np.max(mag[m1])))
        h2_db = 20.0 * math.log10(float(np.max(mag[m2])))
        diffs.append(h1_db - h2_db)
    if not diffs:
        return 4.0
    return float(np.clip(np.median(diffs), -12.0, 22.0))


def _measure_bri_stats(log_env: np.ndarray, v_mask: np.ndarray, n_fft: int, freqs: np.ndarray) -> tuple[float, float]:
    """Return (formant_peak_sharpness_db, brightness_2_6k_vs_05_2k_db) on voiced frames."""
    if not np.any(v_mask):
        return 4.0, -8.0
    ve = log_env[v_mask]
    # Formant-band cepstral modulation (q=8..28) captures formant peak sharpness independent of tilt
    ceps = np.real(irfft(ve, n=n_fft, axis=1))
    band_spec = np.zeros_like(ceps)
    band_spec[:, 8:29] = ceps[:, 8:29]
    band_spec[:, n_fft - 28 : n_fft - 7] = ceps[:, n_fft - 28 : n_fft - 7]
    formant_mod = np.real(rfft(band_spec, n=n_fft, axis=1))
    m_fmt = (freqs >= 350.0) & (freqs <= 4000.0)
    sharp_db = float(np.median(np.std(formant_mod[:, m_fmt], axis=1)) * DB_PER_NP)

    m_lo = (freqs >= 500.0) & (freqs <= 2000.0)
    m_hi = (freqs >= 2000.0) & (freqs <= 6000.0)
    ve_mean = np.mean(ve, axis=0)
    bri_db = float((np.mean(ve_mean[m_hi]) - np.mean(ve_mean[m_lo])) * DB_PER_NP)
    return float(np.clip(sharp_db, 1.0, 14.0)), float(np.clip(bri_db, -30.0, 10.0))


def classify_acoustic_manner(
    log_env: np.ndarray,
    voiced: np.ndarray,
    f0_hz: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Speaker-normalized acoustic manner classification (zero text/corpus leakage).

    Labels:
    - ``0``: unvoiced / silence
    - ``1``: sonorant_nasal (depressed 1-4.5 kHz relative to speaker's own voiced mean)
    - ``2``: vowel_open (elevated 520-1150 Hz vs 200-480 Hz relative to speaker mean)
    - ``3``: vowel_front_high (remaining voiced frames)
    """
    t_len = len(f0_hz)
    labels = np.zeros(t_len, dtype=np.int32)
    v_mask = voiced & (f0_hz > 0.0)
    if int(np.sum(v_mask)) < 4:
        labels[v_mask] = 3
        return labels

    ve = normalize_env_shared_band(log_env[v_mask], freqs)
    ve_mean = np.mean(ve, axis=0)
    rel = ve - ve_mean[None, :]

    m_lf = (freqs >= 120.0) & (freqs <= 480.0)
    m_f1_lo = (freqs >= 200.0) & (freqs <= 480.0)
    m_f1_open = (freqs >= 520.0) & (freqs <= 1150.0)
    m_hf = (freqs >= 1000.0) & (freqs <= 4500.0)

    e_hf_vs_lf = np.mean(rel[:, m_hf], axis=1) - np.mean(rel[:, m_lf], axis=1)
    e_open_vs_close = np.mean(rel[:, m_f1_open], axis=1) - np.mean(rel[:, m_f1_lo], axis=1)

    sub = np.full(int(np.sum(v_mask)), 3, dtype=np.int32)
    is_nasal = e_hf_vs_lf < -0.50
    is_open = (~is_nasal) & (e_open_vs_close > 0.15)
    sub[is_nasal] = 1
    sub[is_open] = 2
    labels[v_mask] = sub
    return labels


@dataclass
class TimbreProfileV3:
    """Compact (< 2 KB serialized) physically grounded UTAU + manner-aware timbre profile."""

    name: str
    corpus: str
    sr: int
    n_fft: int
    total_voiced_s: float
    f0_median_hz: float
    f0_log_std: float
    f0_iqr_st: float
    tilt_db_per_oct: float
    hnr_median_db: float
    # UTAU GEN / OPE / BRI / BRE core descriptors
    f1_median_hz: float
    f2_median_hz: float
    f3_median_hz: float
    n_formant_frames: int
    ope_h1_h2_db: float
    ope_low_mid_db: float
    bri_formant_sharp_db: float
    bri_2_6k_db: float
    bap_voiced_median: list[float]
    # Compact global & manner-relative cepstral coefficients
    voiced_ceps_q32: list[float]
    manner_deltas_q14: dict[str, list[float]]
    manner_counts: dict[str, int]
    # Unvoiced consonant statistics (1-7.2 kHz safe)
    unvoiced_tilt_slope: float
    unvoiced_rel_level_np: float

    @property
    def voiced_env_mean(self) -> np.ndarray:
        return _ceps_decode(self.voiced_ceps_q32, n_fft=self.n_fft)

    def utau_params(self, ref_profile: TimbreProfileV3 | None = None) -> dict[str, float | list[float]]:
        """Readable UTAU-style voice color summary (absolute or relative to `ref_profile`)."""
        if ref_profile is None:
            return {
                "GEN_vtln_f3_hz": round(self.f3_median_hz, 1),
                "OPE_h1_h2_db": round(self.ope_h1_h2_db, 2),
                "BRI_sharp_db": round(self.bri_formant_sharp_db, 2),
                "BRI_2_6k_db": round(self.bri_2_6k_db, 2),
                "BRE_bap_mean": round(float(np.mean(self.bap_voiced_median)), 4),
            }
        alpha = estimate_auto_vtln(ref_profile.to_v2_stub(), self.to_v2_stub())["alpha"]
        return {
            "GEN_delta": round((1.0 - alpha) * 100.0, 2),
            "OPE_delta_db": round(self.ope_h1_h2_db - ref_profile.ope_h1_h2_db, 2),
            "BRI_sharp_ratio": round(self.bri_formant_sharp_db / max(1.0, ref_profile.bri_formant_sharp_db), 3),
            "BRI_2_6k_delta_db": round(self.bri_2_6k_db - ref_profile.bri_2_6k_db, 2),
            "BRE_bap_delta": round(float(np.mean(self.bap_voiced_median) - np.mean(ref_profile.bap_voiced_median)), 4),
        }

    def to_compact_dict(self) -> dict:
        return {
            "name": self.name,
            "corpus": self.corpus,
            "sr": self.sr,
            "n_fft": self.n_fft,
            "voiced_s": round(self.total_voiced_s, 3),
            "f0": [round(self.f0_median_hz, 2), round(self.f0_log_std, 4), round(self.f0_iqr_st, 2)],
            "tilt": round(self.tilt_db_per_oct, 3),
            "hnr": round(self.hnr_median_db, 2),
            "gen_fmt": [round(self.f1_median_hz, 1), round(self.f2_median_hz, 1), round(self.f3_median_hz, 1), int(self.n_formant_frames)],
            "ope": [round(self.ope_h1_h2_db, 3), round(self.ope_low_mid_db, 3)],
            "bri": [round(self.bri_formant_sharp_db, 3), round(self.bri_2_6k_db, 3)],
            "bre_bap": [round(float(x), 4) for x in self.bap_voiced_median],
            "uv": [round(self.unvoiced_tilt_slope, 4), round(self.unvoiced_rel_level_np, 4)],
            "c_v32": [round(float(x), 4) for x in self.voiced_ceps_q32],
            "c_m14": {k: [round(float(x), 4) for x in v] for k, v in self.manner_deltas_q14.items()},
            "m_n": {k: int(v) for k, v in self.manner_counts.items()},
        }

    def to_compact_json(self) -> str:
        return json.dumps(self.to_compact_dict(), separators=(",", ":"))

    def serialized_size_bytes(self) -> int:
        return len(self.to_compact_json().encode("utf-8"))

    @classmethod
    def from_compact_json(cls, payload: str) -> TimbreProfileV3:
        d = json.loads(payload)
        return cls(
            name=str(d["name"]),
            corpus=str(d["corpus"]),
            sr=int(d["sr"]),
            n_fft=int(d["n_fft"]),
            total_voiced_s=float(d["voiced_s"]),
            f0_median_hz=float(d["f0"][0]),
            f0_log_std=float(d["f0"][1]),
            f0_iqr_st=float(d["f0"][2]),
            tilt_db_per_oct=float(d["tilt"]),
            hnr_median_db=float(d["hnr"]),
            f1_median_hz=float(d["gen_fmt"][0]),
            f2_median_hz=float(d["gen_fmt"][1]),
            f3_median_hz=float(d["gen_fmt"][2]),
            n_formant_frames=int(d["gen_fmt"][3]),
            ope_h1_h2_db=float(d["ope"][0]),
            ope_low_mid_db=float(d["ope"][1]),
            bri_formant_sharp_db=float(d["bri"][0]),
            bri_2_6k_db=float(d["bri"][1]),
            bap_voiced_median=[float(x) for x in d["bre_bap"]],
            voiced_ceps_q32=[float(x) for x in d["c_v32"]],
            manner_deltas_q14={k: [float(x) for x in v] for k, v in d["c_m14"].items()},
            manner_counts={k: int(v) for k, v in d["m_n"].items()},
            unvoiced_tilt_slope=float(d["uv"][0]),
            unvoiced_rel_level_np=float(d["uv"][1]),
        )

    def to_v2_stub(self) -> TimbreProfileV2:
        """Lightweight bridge for ``estimate_auto_vtln`` or backward-compatible V2 comparison."""
        env = self.voiced_env_mean
        return TimbreProfileV2(
            name=self.name,
            n_utterances=3,
            total_voiced_s=self.total_voiced_s,
            f0_median_hz=self.f0_median_hz,
            f0_log_std=self.f0_log_std,
            f0_iqr_st=self.f0_iqr_st,
            tilt_db_per_oct=self.tilt_db_per_oct,
            voiced_tilt_hf_db_oct=self.tilt_db_per_oct,
            hnr_median_db=self.hnr_median_db,
            voiced_env_mean=env,
            bap_voiced_median=np.asarray(self.bap_voiced_median, dtype=np.float64),
            f1_median_hz=self.f1_median_hz,
            f2_median_hz=self.f2_median_hz,
            f3_median_hz=self.f3_median_hz,
            n_formant_frames=self.n_formant_frames,
            unvoiced_tilt_slope=self.unvoiced_tilt_slope,
            unvoiced_rel_level_np=self.unvoiced_rel_level_np,
            voiced_ceps_q24=np.zeros((1, 24), dtype=np.float64),
            vtln_anchor_f2_hz=self.f2_median_hz,
        )


def extract_timbre_v3(
    items: list[tuple[np.ndarray, v6_voc.VocoderFeatures]] | list[np.ndarray],
    sr: int = SR,
    name: str = "target_v3",
    corpus: str = "",
) -> TimbreProfileV3:
    """Extract compact (< 2 KB) UTAU-parameterized V3 timbre profile from 1+ utterances."""
    f0_list: list[np.ndarray] = []
    env_list: list[np.ndarray] = []
    bap_list: list[np.ndarray] = []
    manner_env_bins: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    tilts: list[float] = []
    hnrs: list[float] = []
    f1s: list[tuple[float, int]] = []
    f2s: list[tuple[float, int]] = []
    f3s: list[tuple[float, int]] = []
    h1_h2s: list[float] = []
    sharps: list[float] = []
    bris: list[float] = []
    u_slopes: list[float] = []
    u_rels: list[float] = []
    n_fmt_total = 0
    n_fft = N_FFT
    hop = HOP_SAMPLES
    freqs = rfftfreq(n_fft, 1.0 / sr)

    for item in items:
        if isinstance(item, tuple):
            wav, feats = item
        else:
            wav = np.asarray(item, dtype=np.float64)
            feats = v6_voc.analyze(wav, sr=sr)
        n_fft = feats.n_fft
        hop = feats.hop_samples
        freqs = rfftfreq(n_fft, 1.0 / sr)

        v_mask = feats.voiced & (feats.f0_hz > 0.0)
        u_mask = ~v_mask
        if np.any(v_mask):
            f0_list.append(feats.f0_hz[v_mask])
            ve_norm = normalize_env_shared_band(feats.log_env[v_mask], freqs)
            env_list.append(ve_norm)
            bap_list.append(feats.bap[v_mask])
            h1_h2s.append(_measure_h1_h2_db(wav, feats.f0_hz, v_mask, sr=sr, hop_samples=hop))
            sh_db, br_db = _measure_bri_stats(feats.log_env, v_mask, n_fft, freqs)
            sharps.append(sh_db)
            bris.append(br_db)

            m_labels = classify_acoustic_manner(feats.log_env, feats.voiced, feats.f0_hz, freqs)
            full_norm = normalize_env_shared_band(feats.log_env, freqs)
            for cls_id in (1, 2, 3):
                c_mask = m_labels == cls_id
                if np.any(c_mask):
                    manner_env_bins[cls_id].append(full_norm[c_mask])

        if int(np.sum(u_mask)) >= 3 and int(np.sum(v_mask)) >= 3:
            u_mean = np.mean(feats.log_env[u_mask], axis=0)
            m_hi = (freqs >= 1000.0) & (freqs <= 7200.0)
            slope_np, _ = np.polyfit(np.log2(freqs[m_hi]), u_mean[m_hi], 1)
            m_uv_lvl = (freqs >= 250.0) & (freqs <= 7200.0)
            v_lvl = float(np.median(np.mean(feats.log_env[v_mask][:, m_uv_lvl], axis=1)))
            u_lvl = float(np.median(np.mean(feats.log_env[u_mask][:, m_uv_lvl], axis=1)))
            u_slopes.append(float(slope_np))
            u_rels.append(float(np.clip(u_lvl - v_lvl, -3.5, 0.5)))

        lt = eval_m.compute_ltas_metrics(wav, sr=sr)
        if lt and math.isfinite(lt["tilt_db_oct"]):
            tilts.append(float(lt["tilt_db_oct"]))
        f0_m = eval_m.compute_f0_hnr_metrics(wav, sr=sr)
        if f0_m and f0_m.get("hnr_db_median") is not None and math.isfinite(f0_m["hnr_db_median"]):
            hnrs.append(float(f0_m["hnr_db_median"]))
        fm = eval_m.compute_formant_metrics(wav, sr=sr)
        nf = int(fm.get("n_formant_frames", 0)) if fm else 0
        if fm and nf > 0 and fm.get("f1_median_hz") is not None and fm.get("f2_median_hz") is not None:
            f1s.append((float(fm["f1_median_hz"]), nf))
            f2s.append((float(fm["f2_median_hz"]), nf))
            if fm.get("f3_median_hz") is not None:
                f3s.append((float(fm["f3_median_hz"]), nf))
            n_fmt_total += nf

    f0_all = np.concatenate(f0_list) if f0_list else np.array([160.0])
    log_f0 = np.log(f0_all)
    f0_med = float(np.median(f0_all))
    f0_std = float(np.clip(np.std(log_f0), 0.05, 0.45))
    iqr_st = float(12.0 * (np.percentile(log_f0, 75) - np.percentile(log_f0, 25)) / math.log(2.0))

    env_all = np.concatenate(env_list, axis=0) if env_list else np.zeros((1, len(freqs)))
    voiced_env_raw = normalize_env_shared_band(np.mean(env_all, axis=0), freqs)
    voiced_ceps_q32 = _ceps_encode(voiced_env_raw, n_fft=n_fft, q_max=Q_ENV_STORE)
    voiced_env_smooth = _ceps_decode(voiced_ceps_q32, n_fft=n_fft)

    m_lo_mid = (freqs >= 100.0) & (freqs <= 800.0)
    m_hi_mid = (freqs > 800.0) & (freqs <= 2500.0)
    ope_low_mid = float((np.mean(voiced_env_smooth[m_lo_mid]) - np.mean(voiced_env_smooth[m_hi_mid])) * DB_PER_NP)

    bap_all = np.concatenate(bap_list, axis=0) if bap_list else np.full((1, 5), 0.1)
    bap_med = [float(x) for x in np.clip(np.median(bap_all, axis=0), 0.01, 0.98)]

    manner_deltas_q14: dict[str, list[float]] = {}
    manner_counts: dict[str, int] = {}
    for cls_id, cls_name in zip((1, 2, 3), MANNER_NAMES):
        chunks = manner_env_bins[cls_id]
        if chunks:
            arr = np.concatenate(chunks, axis=0)
            manner_counts[cls_name] = int(len(arr))
            cls_diff = normalize_env_shared_band(np.mean(arr, axis=0) - voiced_env_smooth, freqs)
            manner_deltas_q14[cls_name] = _ceps_encode(cls_diff, n_fft=n_fft, q_max=Q_MANNER_STORE)
        else:
            manner_counts[cls_name] = 0
            manner_deltas_q14[cls_name] = [0.0] * (Q_MANNER_STORE + 1)

    return TimbreProfileV3(
        name=name,
        corpus=corpus,
        sr=sr,
        n_fft=n_fft,
        total_voiced_s=float(len(f0_all) * hop / sr),
        f0_median_hz=f0_med,
        f0_log_std=f0_std,
        f0_iqr_st=iqr_st,
        tilt_db_per_oct=float(np.mean(tilts)) if tilts else -3.5,
        hnr_median_db=float(np.median(hnrs)) if hnrs else 13.0,
        f1_median_hz=float(np.average([v for v, _ in f1s], weights=[w for _, w in f1s])) if f1s else 500.0,
        f2_median_hz=float(np.average([v for v, _ in f2s], weights=[w for _, w in f2s])) if f2s else 1750.0,
        f3_median_hz=float(np.average([v for v, _ in f3s], weights=[w for _, w in f3s])) if f3s else 2900.0,
        n_formant_frames=int(n_fmt_total),
        ope_h1_h2_db=float(np.median(h1_h2s)) if h1_h2s else 4.0,
        ope_low_mid_db=ope_low_mid,
        bri_formant_sharp_db=float(np.median(sharps)) if sharps else 4.0,
        bri_2_6k_db=float(np.median(bris)) if bris else -8.0,
        bap_voiced_median=bap_med,
        voiced_ceps_q32=voiced_ceps_q32,
        manner_deltas_q14=manner_deltas_q14,
        manner_counts=manner_counts,
        unvoiced_tilt_slope=float(np.median(u_slopes)) if u_slopes else 0.0,
        unvoiced_rel_level_np=float(np.median(u_rels)) if u_rels else -1.2,
    )


def _apply_bri_contrast(ve: np.ndarray, bri_ratio: float, n_fft: int = N_FFT) -> np.ndarray:
    """Scale mid-quefrency formant resonances (q=8..28) mildly without changing center frequencies or tilt."""
    delta_gamma = float(np.clip(bri_ratio - 1.0, -0.14, 0.14))
    if abs(delta_gamma) < 0.015:
        return ve
    ceps = np.real(irfft(ve, n=n_fft, axis=1))
    mod = np.zeros_like(ceps)
    mod[:, 8:29] = delta_gamma * ceps[:, 8:29]
    mod[:, n_fft - 28 : n_fft - 7] = delta_gamma * ceps[:, n_fft - 28 : n_fft - 7]
    return ve + np.real(rfft(mod, n=n_fft, axis=1))


def adapt_features_v3(
    src_feats: v6_voc.VocoderFeatures,
    src_profile: TimbreProfileV3,
    tgt_profile: TimbreProfileV3,
    *,
    vtln_alpha: float | None = None,
    eq_mode: str = "blind_tilt_anchor",
    tilt_anchor_weight: float = 0.85,
    src_corpus_env: np.ndarray | None = None,
    tgt_corpus_env: np.ndarray | None = None,
    corpus_eq_keep: float = 0.65,
    channel_ridge_damp: float = 0.65,
    use_manner_delta: bool = True,
    use_utau_colors: bool = True,
    k_manner: float = 350.0,
    q_timbre: int = 18,
    gen_override: float | None = None,
    ope_override_db: float | None = None,
    bri_override_ratio: float | None = None,
    bre_override_shift: float | None = None,
) -> tuple[v6_voc.VocoderFeatures, dict]:
    """Adapt HNM/WORLD features using V3 (< 2 KB) UTAU descriptor, EQ decoupler, and shrunk manner delta."""
    sr = src_feats.sr
    n_fft = src_feats.n_fft
    freqs = rfftfreq(n_fft, 1.0 / sr)
    v_mask = src_feats.voiced & (src_feats.f0_hz > 0.0)
    u_mask = ~v_mask

    # 1. F0 log-z-score transfer
    f0_out = np.zeros_like(src_feats.f0_hz)
    if np.any(v_mask):
        ratio_std = max(0.55, min(1.85, tgt_profile.f0_log_std / max(0.03, src_profile.f0_log_std)))
        log_f0_src = np.log(src_feats.f0_hz[v_mask])
        log_f0_tgt = math.log(tgt_profile.f0_median_hz) + ratio_std * (
            log_f0_src - math.log(src_profile.f0_median_hz)
        )
        f0_out[v_mask] = np.clip(np.exp(log_f0_tgt), 65.0, 440.0)

    # 2. UTAU GEN (data-driven F1/F2/F3 VTLN on voiced frames only)
    vtln_info = estimate_auto_vtln(src_profile.to_v2_stub(), tgt_profile.to_v2_stub())
    if vtln_alpha is not None:
        alpha = float(vtln_alpha)
    elif gen_override is not None:
        alpha = float(np.clip(1.0 - float(gen_override) / 100.0, 0.78, 1.22))
    else:
        alpha = float(vtln_info["alpha"])

    env_out = src_feats.log_env.copy()
    if abs(alpha - 1.0) >= 1e-4 and np.any(v_mask):
        env_out[v_mask] = v6_cloner._vtln_warp_2d(src_feats.log_env[v_mask], alpha, sr=sr, n_fft=n_fft)

    eq_diag = {}
    manner_weights = {}
    if np.any(v_mask):
        src_env_unwarped = src_profile.voiced_env_mean
        src_env_warped = src_env_unwarped
        warp_tilt_delta_db = 0.0
        if abs(alpha - 1.0) >= 1e-4:
            src_env_warped = v6_cloner._vtln_warp_2d(src_env_unwarped[None, :], alpha, sr=sr, n_fft=n_fft)[0]
            s_unw, _ = linear_tilt_fit(normalize_env_shared_band(src_env_unwarped, freqs), freqs)
            s_wrp, _ = linear_tilt_fit(normalize_env_shared_band(src_env_warped, freqs), freqs)
            warp_tilt_delta_db = (s_wrp - s_unw) * DB_PER_NP

        base_curve, eq_diag = decompose_envelope_eq(
            src_env=src_env_warped,
            tgt_env=tgt_profile.voiced_env_mean,
            freqs=freqs,
            src_tilt_db_oct=src_profile.tilt_db_per_oct + warp_tilt_delta_db,
            tgt_tilt_db_oct=tgt_profile.tilt_db_per_oct,
            n_fft=n_fft,
            q_timbre=q_timbre,
            tilt_anchor_weight=tilt_anchor_weight,
            eq_mode=eq_mode,
            src_corpus_env=src_corpus_env,
            tgt_corpus_env=tgt_corpus_env,
            corpus_eq_keep=corpus_eq_keep,
            channel_ridge_damp=channel_ridge_damp,
        )

        # UTAU OPE: low-mid glottal open-quotient harmonic shelf (100-1400 Hz)
        if use_utau_colors:
            d_ope_db = (
                float(ope_override_db)
                if ope_override_db is not None
                else float(tgt_profile.ope_h1_h2_db - src_profile.ope_h1_h2_db)
            )
            d_ope_db = float(np.clip(d_ope_db, -4.5, 4.5))
            # Smooth shelf: + below 450 Hz, smooth transition to 0 by 1600 Hz
            u_ope = np.clip((freqs - 400.0) / 1100.0, 0.0, 1.0)
            ope_shelf = (1.0 - 0.5 * (1.0 - np.cos(math.pi * u_ope))) * float(np.clip(0.20 * d_ope_db / DB_PER_NP, -0.18, 0.18))
            base_curve = np.clip(base_curve + ope_shelf, -1.20, 1.20)

        # Synthetic male_ sub-fundamental boom clamp (only if real sub-F0 audio absent)
        if tgt_profile.name.startswith("male_") and tgt_profile.f0_median_hz < 0.85 * src_profile.f0_median_hz:
            f_cut = max(250.0, 1.3 * tgt_profile.f0_median_hz)
            base_curve[freqs < f_cut] = np.minimum(base_curve[freqs < f_cut], 0.0)

        pre_pwr = np.sum(np.exp(2.0 * env_out[v_mask]), axis=1)
        env_shifted = env_out[v_mask] + base_curve[None, :]

        # Manner-aware shrunk delta per acoustic class
        if use_manner_delta:
            m_labels = classify_acoustic_manner(src_feats.log_env, src_feats.voiced, src_feats.f0_hz, freqs)
            v_indices = np.where(v_mask)[0]
            v_labels = m_labels[v_indices]
            for cls_id, cls_name in zip((1, 2, 3), MANNER_NAMES):
                cls_pos = np.where(v_labels == cls_id)[0]
                if cls_pos.size == 0:
                    manner_weights[cls_name] = 0.0
                    continue
                n_eff = min(src_profile.manner_counts.get(cls_name, 0), tgt_profile.manner_counts.get(cls_name, 0))
                w_c = float(n_eff / (n_eff + k_manner)) if n_eff > 0 else 0.0
                manner_weights[cls_name] = round(w_c, 4)
                if w_c < 0.03:
                    continue
                d_src_c = _ceps_decode(src_profile.manner_deltas_q14[cls_name], n_fft=n_fft)
                if abs(alpha - 1.0) >= 1e-4:
                    d_src_c = v6_cloner._vtln_warp_2d(d_src_c[None, :], alpha, sr=sr, n_fft=n_fft)[0]
                d_tgt_c = _ceps_decode(tgt_profile.manner_deltas_q14[cls_name], n_fft=n_fft)
                extra_c = w_c * np.clip(normalize_env_shared_band(d_tgt_c - d_src_c, freqs), -0.40, 0.40)
                env_shifted[cls_pos] = env_shifted[cls_pos] + extra_c[None, :]

        # UTAU BRI: formant peak sharpness scaling (q=8..28)
        if use_utau_colors:
            n_bri = min(src_profile.n_formant_frames, tgt_profile.n_formant_frames)
            w_bri = n_bri / (n_bri + 320.0) if n_bri > 0 else 0.0
            raw_bri = (
                float(bri_override_ratio)
                if bri_override_ratio is not None
                else tgt_profile.bri_formant_sharp_db / max(1.0, src_profile.bri_formant_sharp_db)
            )
            bri_ratio = 1.0 + w_bri * (raw_bri - 1.0)
            env_shifted = _apply_bri_contrast(env_shifted, bri_ratio=bri_ratio, n_fft=n_fft)

        post_pwr = np.sum(np.exp(2.0 * env_shifted), axis=1)
        gain_adj = 0.5 * np.log(np.maximum(pre_pwr, 1e-18) / np.maximum(post_pwr, 1e-18))
        env_out[v_mask] = env_shifted + gain_adj[:, None]

    # 3b. Unvoiced consonant preservation (safe [800, 7200] Hz alignment, never VTLN-warped)
    if np.any(u_mask):
        u_ramp = np.clip((freqs - 800.0) / 800.0, 0.0, 1.0)
        hi_taper = np.where(freqs > 7200.0, np.clip(1.0 - (freqs - 7200.0) / 1800.0, 0.0, 1.0), 1.0)
        d_u_slope = float(np.clip(tgt_profile.unvoiced_tilt_slope - src_profile.unvoiced_tilt_slope, -0.40, 0.40))
        u_shift = u_ramp * hi_taper * d_u_slope * np.log2(np.clip(freqs, 1000.0, 7200.0) / 3000.0)
        u_shift = np.clip(u_shift, -0.35, 0.35)
        d_u_lvl = float(np.clip(tgt_profile.unvoiced_rel_level_np - src_profile.unvoiced_rel_level_np, -0.25, 0.25))
        env_out[u_mask] = env_out[u_mask] + u_shift[None, :] + 0.5 * d_u_lvl

    # 4. UTAU BRE: band-wise logit breathiness transfer on voiced frames ONLY
    bap_out = src_feats.bap.copy()
    if np.any(v_mask):
        tgt_bap = np.asarray(tgt_profile.bap_voiced_median, dtype=np.float64)
        src_bap = np.asarray(src_profile.bap_voiced_median, dtype=np.float64)
        delta_logit = v6_cloner._logit(tgt_bap) - v6_cloner._logit(src_bap)
        if bre_override_shift is not None:
            delta_logit = delta_logit + float(bre_override_shift)
        bap_out[v_mask] = np.clip(
            v6_cloner._sigmoid(v6_cloner._logit(src_feats.bap[v_mask]) + delta_logit[None, :]),
            0.005,
            0.985,
        )

    out_feats = v6_voc.VocoderFeatures(
        f0_hz=f0_out,
        voiced=src_feats.voiced.copy(),
        log_env=env_out,
        bap=bap_out,
        sr=sr,
        hop_samples=src_feats.hop_samples,
        n_samples=src_feats.n_samples,
        n_fft=n_fft,
    )
    info = {
        "alpha": round(alpha, 4),
        "vtln_info": vtln_info,
        "eq_diag": eq_diag,
        "manner_weights": manner_weights,
        "utau": tgt_profile.utau_params(src_profile),
        "profile_bytes": tgt_profile.serialized_size_bytes(),
    }
    return out_feats, info


def adapt_features_v3_only(
    src_feats: v6_voc.VocoderFeatures,
    src_profile: TimbreProfileV3,
    tgt_profile: TimbreProfileV3,
) -> v6_voc.VocoderFeatures:
    """Thin wrapper returning only VocoderFeatures (for direct `v6_cloner.adapt_features`-compatible callers)."""
    feats, _ = adapt_features_v3(src_feats, src_profile, tgt_profile)
    return feats
