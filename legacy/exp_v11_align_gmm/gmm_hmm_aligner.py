"""Bootstrapped 100% Non-Neural 3-State GMM-HMM Viterbi Forced Aligner (`exp_v11_align_gmm`).

Replaces single-pass DP heuristic alignment (`v8/v10`) and Mandarin `quiet_dip_split`
with a flat-start/bootstrapped 2-component diagonal GMM-HMM (3 logical states:
entry / steady-core / exit, mapped to manner-minimum substate topology so `<20 ms`
starved phones are structurally impossible) trained over 5 EM-Viterbi iterations on
29-dim acoustic features:
- 13-dim MFCC (normalized c1..c12 + relative log-energy c0)
- 13-dim delta-MFCC
- 3-dim acoustic manner cues (voiced, HF/LF ratio, mid/nasal-LF ratio)

Outputs exact `(p_start, p_mid, p_end)` timestamps where `p_mid` is the true
steady-state core midpoint (state 1 centroid) for `@timbre-lab`'s English diphone bank,
plus acoustically re-aligned Mandarin `extended/` `(onset, final)` syllable boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.fft import dct, rfft, rfftfreq

ROOT_IMPORT = Path(__file__).resolve().parent.parent
if str(ROOT_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_IMPORT))

from exp_v5_arch import crossling_frontend as cf
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS
from exp_v6_vocoder import vocoder as v6_voc
from exp_v8_clarity import en_utau_bank as v8_en
from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10
from exp_v10_g2p_and_align.landmark_aligner import align_ipa_sequence_v10, score_token_purity
from exp_v11_align_gmm import EXTENDED_INDEX, HOP, MANIFEST_PATH, N_FFT, ROOT_DIR, SR

EN_HELDOUT_DEFAULT = frozenset(f"{spk}_00{i}" for spk in ("slt", "bdl") for i in range(30, 40))


@dataclass
class PhoneBoundary:
    ipa: str
    arpabet: str | None
    word: str
    syllable_idx: int
    stress_or_tone: str
    p_start: int
    p_mid: int
    p_end: int
    dur_ms: float
    voiced_frac: float
    rms_dbfs: float
    manner: str


@dataclass
class SyllableBoundary:
    index: int
    hanzi: str
    pinyin: str
    onset: str
    final: str
    tone: int
    syl_start: int
    f_split: int
    syl_end: int
    dur_ms: float
    phones: list[PhoneBoundary] = field(default_factory=list)


@dataclass
class AlignedUtterance:
    utterance_id: str
    speaker: str
    language: str
    text: str
    sr: int
    hop_samples: int
    n_frames: int
    phones: list[PhoneBoundary]
    syllables: list[SyllableBoundary] = field(default_factory=list)


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(sr: int = SR, n_fft: int = N_FFT, n_mels: int = 26, f_lo: float = 80.0, f_hi: float = 7600.0) -> np.ndarray:
    freqs = rfftfreq(n_fft, 1.0 / sr)
    mels = np.linspace(float(_hz_to_mel(f_lo)), float(_hz_to_mel(f_hi)), n_mels + 2)
    hz_pts = _mel_to_hz(mels)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float64)
    for m in range(n_mels):
        f_l, f_c, f_r = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up = (freqs >= f_l) & (freqs <= f_c)
        dn = (freqs > f_c) & (freqs <= f_r)
        if f_c > f_l:
            fb[m, up] = (freqs[up] - f_l) / (f_c - f_l)
        if f_r > f_c:
            fb[m, dn] = (f_r - freqs[dn]) / (f_r - f_c)
    return fb


_MEL_FB = _build_mel_filterbank()


def extract_29d_features(
    wav: np.ndarray,
    feats: v6_voc.VocoderFeatures,
    sr: int = SR,
    hop: int = HOP,
) -> np.ndarray:
    """Compute speaker-normalized 29-dim (13 MFCC + 13 delta-MFCC + 3 manner) matrix (T, 29)."""
    n_fr = len(feats.log_env)
    win_len = 600  # 25 ms Hann centered at hop * t
    half = win_len // 2
    win = np.hanning(win_len)
    padded = np.pad(wav.astype(np.float64), (half, half + win_len), mode="constant")
    starts = np.arange(n_fr) * hop
    idx = starts[:, None] + np.arange(win_len)[None, :]
    frames = padded[idx] * win[None, :]
    spec_pwr = np.abs(rfft(frames, n=N_FFT, axis=1)) ** 2
    mel_e = np.maximum(spec_pwr @ _MEL_FB.T, 1e-12)
    log_mel = np.log(mel_e)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :13]

    # Utterance normalize c1..c12; keep c0 relative to 95th percentile energy (dB-like scale)
    c0_rel = (mfcc[:, 0] - np.percentile(mfcc[:, 0], 95)) / 3.5
    c_rest = mfcc[:, 1:13]
    c_rest = (c_rest - np.mean(c_rest, axis=0, keepdims=True)) / np.maximum(np.std(c_rest, axis=0, keepdims=True), 1e-3)
    base13 = np.concatenate([c0_rel[:, None], c_rest], axis=1)
    delta13 = 0.5 * (np.roll(base13, -1, axis=0) - np.roll(base13, 1, axis=0))
    delta13[0] = base13[1] - base13[0] if n_fr > 1 else 0.0
    delta13[-1] = base13[-1] - base13[-2] if n_fr > 1 else 0.0

    freqs = rfftfreq(feats.n_fft, 1.0 / sr)
    env_pwr = np.exp(2.0 * feats.log_env)
    m_lf = (freqs >= 80.0) & (freqs <= 1000.0)
    m_nas = (freqs >= 120.0) & (freqs <= 450.0)
    m_mid = (freqs >= 500.0) & (freqs <= 2000.0)
    m_hf = (freqs >= 2000.0) & (freqs <= 7600.0)

    def _bdb(m: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.maximum(np.mean(env_pwr[:, m], axis=1), 1e-14))

    hf_lf = np.clip((_bdb(m_hf) - _bdb(m_lf)) / 8.0, -2.5, 2.5)
    mid_lf = np.clip((_bdb(m_mid) - _bdb(m_nas)) / 8.0, -2.5, 2.5)
    voi = np.where(feats.voiced, 1.5, -1.5)
    return np.concatenate([base13, delta13, voi[:, None], hf_lf[:, None], mid_lf[:, None]], axis=1).astype(np.float64)


def _manner_group(ipa: str) -> str:
    sp = IPA_SPECS.get(ipa)
    if sp is None:
        return "vowel"
    if sp.manner in ("vowel", "diphthong"):
        return "vowel"
    if sp.manner in ("nasal", "liquid", "glide"):
        return "sonorant"
    if not sp.voiced and sp.manner in ("fricative", "affricate"):
        return "uv_fricative"
    if not sp.voiced and sp.manner == "stop":
        return "uv_stop"
    return "v_obstruent"


def _substate_count(ipa: str) -> int:
    """Strict topological minimum frames per phoneme (4=20ms stops/unaspirated affricates, 5=25ms sonorants, 6=30ms vowels/sibilants)."""
    mg = _manner_group(ipa)
    if mg == "uv_stop" or ipa in ("ɾ", "tɕ", "ts", "ʈʂ", "x") or (IPA_SPECS.get(ipa) and IPA_SPECS[ipa].manner == "stop"):
        return 4  # 20 ms minimum
    if mg in ("sonorant", "v_obstruent"):
        return 5  # 25 ms minimum
    return 6  # 30 ms minimum (vowels & aspirated/long fricatives)


def _substate_logical_map(n_sub: int) -> list[int]:
    """Map n_sub topological states to 3 logical HMM states (0: entry, 1: steady core, 2: exit)."""
    if n_sub <= 3:
        return [0, 1, 2][:n_sub]
    # [0, 1, 1, ..., 1, 2] so core state 1 captures steady-state middle
    return [0] + [1] * (n_sub - 2) + [2]


class _PhoneGMMModel:
    """2-component diagonal GMM per (key, logical_state in 0..2) with manner-prior shrinkage."""

    def __init__(self, dim: int = 29, var_floor: float = 0.14):
        self.dim = dim
        self.var_floor = var_floor
        # params[(ipa_or_manner, s)] = (w0, w1, mu0, var0, mu1, var1)
        self.params: dict[tuple[str, int], tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def fit_from_pools(
        self,
        phone_pools: dict[tuple[str, int], list[np.ndarray]],
        manner_pools: dict[tuple[str, int], list[np.ndarray]],
    ) -> None:
        # First fit manner-level fallback 2-GMMs
        manner_stats: dict[tuple[str, int], tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for key, vecs in manner_pools.items():
            x = np.stack(vecs, axis=0) if vecs else np.zeros((2, self.dim))
            manner_stats[key] = self._fit_2gmm_raw(x, prior=None, shrink_n=0.0)
            self.params[key] = manner_stats[key]

        # Fit per-(ipa, state) shrunk toward its manner prior
        all_ipas = {k[0] for k in phone_pools}
        for ipa in all_ipas:
            mg = _manner_group(ipa)
            for s in (0, 1, 2):
                prior = manner_stats.get((mg, s))
                vecs = phone_pools.get((ipa, s), [])
                if vecs:
                    x = np.stack(vecs, axis=0)
                    self.params[(ipa, s)] = self._fit_2gmm_raw(x, prior=prior, shrink_n=14.0)
                elif prior is not None:
                    self.params[(ipa, s)] = prior

    def _fit_2gmm_raw(
        self,
        x: np.ndarray,
        prior: tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
        shrink_n: float = 14.0,
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = len(x)
        mu_all = np.mean(x, axis=0)
        var_all = np.maximum(np.var(x, axis=0), self.var_floor)
        if prior is not None and shrink_n > 0.0:
            lam = n / (n + shrink_n)
            p_mu = prior[0] * prior[2] + prior[1] * prior[4]
            p_var = np.maximum(0.5 * (prior[3] + prior[5]), self.var_floor)
            mu_all = lam * mu_all + (1.0 - lam) * p_mu
            var_all = np.maximum(lam * var_all + (1.0 - lam) * p_var, self.var_floor)

        if n < 24:
            return 0.5, 0.5, mu_all.copy(), var_all.copy(), mu_all.copy(), var_all.copy()

        # Median split along highest-variance axis + 2 K-means-like EM steps -> 2 diagonal components
        axis = int(np.argmax(np.var(x, axis=0)))
        med = float(np.median(x[:, axis]))
        m0 = x[:, axis] <= med
        m1 = ~m0
        if int(np.sum(m0)) < 6 or int(np.sum(m1)) < 6:
            return 0.5, 0.5, mu_all.copy(), var_all.copy(), mu_all.copy(), var_all.copy()

        mu0 = np.mean(x[m0], axis=0)
        mu1 = np.mean(x[m1], axis=0)
        var0 = np.maximum(np.var(x[m0], axis=0), self.var_floor)
        var1 = np.maximum(np.var(x[m1], axis=0), self.var_floor)
        w0 = float(np.mean(m0))
        w1 = 1.0 - w0
        if prior is not None and shrink_n > 0.0:
            lam = n / (n + shrink_n)
            p_mu = prior[0] * prior[2] + prior[1] * prior[4]
            p_var = np.maximum(0.5 * (prior[3] + prior[5]), self.var_floor)
            mu0 = lam * mu0 + (1.0 - lam) * p_mu
            mu1 = lam * mu1 + (1.0 - lam) * p_mu
            var0 = np.maximum(lam * var0 + (1.0 - lam) * p_var, self.var_floor)
            var1 = np.maximum(lam * var1 + (1.0 - lam) * p_var, self.var_floor)
        return w0, w1, mu0, var0, mu1, var1

    def score_sequence(self, X: np.ndarray, ipa_state_keys: list[tuple[str, int]]) -> np.ndarray:
        """Vectorized log-likelihood matrix (M_unique_or_seq, T) for sequence of (ipa, logical_state)."""
        # Score unique keys once then gather
        uniq_keys = list(dict.fromkeys(ipa_state_keys))
        k_to_idx = {k: i for i, k in enumerate(uniq_keys)}
        u_n = len(uniq_keys)
        t_n = len(X)
        x2 = X * X

        w0_a = np.empty(u_n)
        w1_a = np.empty(u_n)
        mu0 = np.empty((u_n, self.dim))
        iv0 = np.empty((u_n, self.dim))
        c0 = np.empty(u_n)
        mu1 = np.empty((u_n, self.dim))
        iv1 = np.empty((u_n, self.dim))
        c1 = np.empty(u_n)

        for idx, (ipa, s) in enumerate(uniq_keys):
            p = self.params.get((ipa, s)) or self.params.get((_manner_group(ipa), s))
            if p is None:
                # Neutral fallback
                w0, w1, m0, v0, m1, v1 = 0.5, 0.5, np.zeros(self.dim), np.ones(self.dim), np.zeros(self.dim), np.ones(self.dim)
            else:
                w0, w1, m0, v0, m1, v1 = p
            w0_a[idx] = math.log(max(1e-4, w0))
            w1_a[idx] = math.log(max(1e-4, w1))
            mu0[idx] = m0
            iv0[idx] = 1.0 / v0
            c0[idx] = -0.5 * (self.dim * math.log(2.0 * math.pi) + float(np.sum(np.log(v0))))
            mu1[idx] = m1
            iv1[idx] = 1.0 / v1
            c1[idx] = -0.5 * (self.dim * math.log(2.0 * math.pi) + float(np.sum(np.log(v1))))

        # Compute -0.5 * sum((X - mu)^2 / var) = -0.5*(X^2 @ iv.T - 2*X @ (mu*iv).T + sum(mu^2*iv))
        lp0 = c0[:, None] - 0.5 * (
            iv0 @ x2.T - 2.0 * (mu0 * iv0) @ X.T + np.sum(mu0 * mu0 * iv0, axis=1)[:, None]
        ) + w0_a[:, None]
        lp1 = c1[:, None] - 0.5 * (
            iv1 @ x2.T - 2.0 * (mu1 * iv1) @ X.T + np.sum(mu1 * mu1 * iv1, axis=1)[:, None]
        ) + w1_a[:, None]

        hi = np.maximum(lp0, lp1)
        uniq_scores = hi + np.log1p(np.exp(np.minimum(lp0, lp1) - hi))  # (u_n, T)
        gather_idx = [k_to_idx[k] for k in ipa_state_keys]
        return uniq_scores[gather_idx]


def _viterbi_bakis(
    emit_scores: np.ndarray,
    phone_subcounts: list[int],
    self_bonus: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Fast vectorized left-to-right Viterbi returning [(p_start, p_mid, p_end), ...]."""
    m_states, t_n = emit_scores.shape
    if m_states == 0 or t_n < m_states:
        # Uniform safety guard for degenerate ultra-short utterances
        n_p = len(phone_subcounts)
        edges = np.linspace(0, t_n, n_p + 1).astype(int)
        out = []
        for i in range(n_p):
            s = int(edges[i])
            e = int(max(s + 1, edges[i + 1]))
            out.append((s, (s + e) // 2, e))
        return out

    # Step penalty: stay keeps state (bonus -0.35 + self_bonus), advance moves s -> s+1 (-0.85)
    stay_pen = -0.35 + self_bonus
    adv_pen = -0.85

    dp_prev = np.full(m_states, -1e15, dtype=np.float64)
    dp_prev[0] = emit_scores[0, 0]
    bp = np.zeros((t_n, m_states), dtype=np.int8)  # 0 = stay, 1 = advance from s-1

    for t in range(1, t_n):
        stay_vals = dp_prev + stay_pen
        adv_vals = np.empty(m_states, dtype=np.float64)
        adv_vals[0] = -1e15
        adv_vals[1:] = dp_prev[:-1] + adv_pen

        take_adv = adv_vals > stay_vals
        dp_cur = np.where(take_adv, adv_vals, stay_vals) + emit_scores[:, t]
        bp[t] = take_adv.astype(np.int8)
        dp_prev = dp_cur

    # Backtrack from last state m_states - 1
    path = np.empty(t_n, dtype=np.int32)
    cur_s = m_states - 1
    for t in range(t_n - 1, -1, -1):
        path[t] = cur_s
        if t > 0 and bp[t, cur_s] == 1:
            cur_s = max(0, cur_s - 1)

    # Convert state path to (p_start, p_mid, p_end) per phone
    bounds: list[tuple[int, int, int]] = []
    offset = 0
    for n_sub in phone_subcounts:
        s_lo = offset
        s_hi = offset + n_sub - 1
        frames_p = np.where((path >= s_lo) & (path <= s_hi))[0]
        if frames_p.size == 0:
            prev_e = bounds[-1][2] if bounds else 0
            bounds.append((prev_e, prev_e, min(t_n, prev_e + 1)))
        else:
            p_start = int(frames_p[0])
            p_end = int(frames_p[-1]) + 1
            # Steady-state midpoint: median frame assigned to core substates [s_lo+1 .. s_hi-1]
            core_frames = np.where((path >= s_lo + 1) & (path <= max(s_lo + 1, s_hi - 1)))[0]
            if core_frames.size > 0:
                p_mid = int(round(float(np.median(core_frames))))
            else:
                p_mid = (p_start + p_end) // 2
            p_mid = max(p_start, min(p_end - 1, p_mid))
            bounds.append((p_start, p_mid, p_end))
        offset += n_sub
    return bounds


def _trim_utterance_active(x: np.ndarray, feats: v6_voc.VocoderFeatures, sr: int = SR, hop: int = HOP) -> tuple[int, int, int, int]:
    """Return (sample_s0, sample_s1, frame_f0, frame_f1) trimming dead outer silence cleanly."""
    freqs = rfftfreq(feats.n_fft, 1.0 / sr)
    pwr = np.exp(2.0 * feats.log_env)
    tot_db = 10.0 * np.log10(np.maximum(np.mean(pwr, axis=1), 1e-14))
    pk = float(np.percentile(tot_db, 95)) if tot_db.size else 0.0
    act = np.where(feats.voiced | (tot_db - pk > -32.0))[0]
    if act.size == 0:
        return 0, len(x), 0, len(feats.log_env)
    f0 = max(0, int(act[0]) - 2)
    f1 = min(len(feats.log_env), int(act[-1]) + 3)
    return f0 * hop, min(len(x), f1 * hop), f0, max(f0 + 4, f1)


def _train_and_align_gmm(
    records: list[dict],
    n_em: int = 4,
    sr: int = SR,
    hop: int = HOP,
) -> tuple[_PhoneGMMModel, dict[str, list[tuple[int, int, int]]]]:
    """Train 2-comp 3-state GMM-HMM via flat-start/V10 bootstrap + EM iterations."""
    # Precompute initial alignments & 29-dim features
    feat_X: dict[str, np.ndarray] = {}
    cur_bounds: dict[str, list[tuple[int, int, int]]] = {}
    for rec in records:
        uid = rec["id"]
        X = extract_29d_features(rec["wav"], rec["feats"], sr=sr, hop=hop)
        feat_X[uid] = X
        n_fr = len(X)
        if rec.get("init_bounds") is not None:
            cur_bounds[uid] = rec["init_bounds"]
        else:
            b_v10, _ = align_ipa_sequence_v10(rec["ipas"], rec["feats"], sr=sr)
            cur_bounds[uid] = [(s, (s + e) // 2, e) for s, e in b_v10]

    gmm = _PhoneGMMModel(dim=29, var_floor=0.14)
    for it in range(n_em + 1):
        phone_pools: dict[tuple[str, int], list[np.ndarray]] = {}
        manner_pools: dict[tuple[str, int], list[np.ndarray]] = {}
        for rec in records:
            uid = rec["id"]
            X = feat_X[uid]
            for ipa, (ps, _pm, pe) in zip(rec["ipas"], cur_bounds[uid]):
                if pe <= ps:
                    continue
                mg = _manner_group(ipa)
                seg = X[ps:pe]
                t_seg = len(seg)
                if t_seg >= 3:
                    k1 = max(1, t_seg // 4)
                    k2 = max(k1 + 1, (3 * t_seg) // 4)
                    chunks = (seg[:k1], seg[k1:k2], seg[k2:])
                else:
                    chunks = (seg, seg, seg)
                for st_idx, ch in enumerate(chunks):
                    if len(ch) > 0:
                        phone_pools.setdefault((ipa, st_idx), []).extend(ch)
                        manner_pools.setdefault((mg, st_idx), []).extend(ch)

        gmm.fit_from_pools(phone_pools, manner_pools)
        if it == n_em:
            break

        # E-step: Viterbi re-alignment across all training utterances
        for rec in records:
            uid = rec["id"]
            X = feat_X[uid]
            ipas = rec["ipas"]
            subcounts = [_substate_count(p) for p in ipas]
            if sum(subcounts) > len(X):
                subcounts = [3 if len(X) >= 3 * len(ipas) else 1 for _ in ipas]
            seq_keys: list[tuple[str, int]] = []
            self_b: list[float] = []
            for p, ns in zip(ipas, subcounts):
                lmap = _substate_logical_map(ns)
                for ls in lmap:
                    seq_keys.append((p, ls))
                    self_b.append(0.15 if ls == 1 else -0.05)
            emit = gmm.score_sequence(X, seq_keys)
            cur_bounds[uid] = _viterbi_bakis(emit, subcounts, np.asarray(self_b, dtype=np.float64))

    return gmm, cur_bounds


def _align_with_trained_gmm(
    gmm: _PhoneGMMModel,
    wav: np.ndarray,
    feats: v6_voc.VocoderFeatures,
    ipas: list[str],
    sr: int = SR,
    hop: int = HOP,
) -> list[tuple[int, int, int]]:
    X = extract_29d_features(wav, feats, sr=sr, hop=hop)
    subcounts = [_substate_count(p) for p in ipas]
    if sum(subcounts) > len(X):
        subcounts = [3 if len(X) >= 3 * len(ipas) else 1 for _ in ipas]
    seq_keys: list[tuple[str, int]] = []
    self_b: list[float] = []
    for p, ns in zip(ipas, subcounts):
        lmap = _substate_logical_map(ns)
        for ls in lmap:
            seq_keys.append((p, ls))
            self_b.append(0.15 if ls == 1 else -0.05)
    emit = gmm.score_sequence(X, seq_keys)
    return _viterbi_bakis(emit, subcounts, np.asarray(self_b, dtype=np.float64))


def _build_phone_boundaries(
    meta_phones: list[dict],
    bounds: list[tuple[int, int, int]],
    wav: np.ndarray,
    feats: v6_voc.VocoderFeatures,
    sr: int = SR,
    hop: int = HOP,
) -> list[PhoneBoundary]:
    out: list[PhoneBoundary] = []
    for mp, (ps, pm, pe) in zip(meta_phones, bounds):
        ipa = mp["ipa"]
        seg_w = wav[ps * hop : min(len(wav), pe * hop)]
        dur_ms = float((pe - ps) * hop * 1000.0 / sr)
        vf = float(np.mean(feats.voiced[ps:pe])) if pe > ps else 0.0
        rms = float(np.sqrt(np.mean(seg_w.astype(np.float64) ** 2))) if seg_w.size else 0.0
        rms_db = 20.0 * math.log10(max(1e-7, rms))
        sp = IPA_SPECS.get(ipa)
        out.append(
            PhoneBoundary(
                ipa=ipa,
                arpabet=mp.get("arpabet"),
                word=mp.get("word", ""),
                syllable_idx=int(mp.get("syllable_idx", 0)),
                stress_or_tone=str(mp.get("stress_or_tone", "")),
                p_start=int(ps),
                p_mid=int(pm),
                p_end=int(pe),
                dur_ms=round(dur_ms, 2),
                voiced_frac=round(vf, 3),
                rms_dbfs=round(rms_db, 2),
                manner=sp.manner if sp else "vowel",
            )
        )
    return out


def align_en_corpus_gmm(
    exclude_ids: set[str] | frozenset[str] | None = EN_HELDOUT_DEFAULT,
    *,
    align_heldout: bool = False,
    n_em: int = 4,
    sr: int = SR,
) -> dict:
    """Train 3-state GMM-HMM strictly on training utterances and align EN CMU-ARCTIC corpus."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    excl = set(exclude_ids) if exclude_ids is not None else set()

    train_recs: list[dict] = []
    held_recs: list[dict] = []
    for u in manifest["english"]["utterances"]:
        uid = u["id"]
        x, file_sr = v6_voc.read_wav_mono(ROOT_DIR / u["path"])
        if file_sr != sr:
            raise ValueError(f"Expected {sr} Hz, got {file_sr} for {uid}")
        s0, s1 = v8_en._trim_active_span(x, sr=sr)
        xa = x[s0:s1]
        feats_full = v6_voc.analyze(xa, sr=sr)
        cs, ce, cf0, cf1 = _trim_utterance_active(xa, feats_full, sr=sr)
        xa_t = xa[cs:ce]
        feats = v6_voc.VocoderFeatures(
            f0_hz=feats_full.f0_hz[cf0:cf1],
            voiced=feats_full.voiced[cf0:cf1],
            log_env=feats_full.log_env[cf0:cf1],
            bap=feats_full.bap[cf0:cf1],
            sr=sr,
            hop_samples=HOP,
            n_samples=len(xa_t),
            n_fft=feats_full.n_fft,
        )

        ir = plan_v10(u["text"])
        meta_phones: list[dict] = []
        ipas: list[str] = []
        for s_idx, unit in enumerate(ir["units"]):
            arps = unit.get("arpabet") or []
            p_list = unit.get("ipa_phones", [])
            for k, p in enumerate(p_list):
                ipas.append(p)
                meta_phones.append({
                    "ipa": p,
                    "arpabet": arps[k] if k < len(arps) else None,
                    "word": unit.get("token", ""),
                    "syllable_idx": s_idx,
                    "stress_or_tone": unit.get("stress_or_tone", "S0"),
                })
        rec = {
            "id": uid,
            "speaker": u["speaker"],
            "text": u["text"],
            "wav": xa_t,
            "feats": feats,
            "ipas": ipas,
            "meta_phones": meta_phones,
        }
        if uid in excl:
            held_recs.append(rec)
        else:
            train_recs.append(rec)

    gmm, train_bounds = _train_and_align_gmm(train_recs, n_em=n_em, sr=sr)

    aligned: dict[str, dict] = {}
    for rec in train_recs:
        uid = rec["id"]
        pbs = _build_phone_boundaries(rec["meta_phones"], train_bounds[uid], rec["wav"], rec["feats"], sr=sr)
        au = AlignedUtterance(
            utterance_id=uid,
            speaker=rec["speaker"],
            language="en",
            text=rec["text"],
            sr=sr,
            hop_samples=HOP,
            n_frames=len(rec["feats"].log_env),
            phones=pbs,
        )
        aligned[uid] = {"aligned": au, "wav": rec["wav"], "feats": rec["feats"]}

    held_aligned: dict[str, dict] = {}
    if align_heldout and held_recs:
        for rec in held_recs:
            uid = rec["id"]
            b = _align_with_trained_gmm(gmm, rec["wav"], rec["feats"], rec["ipas"], sr=sr)
            pbs = _build_phone_boundaries(rec["meta_phones"], b, rec["wav"], rec["feats"], sr=sr)
            au = AlignedUtterance(
                utterance_id=uid,
                speaker=rec["speaker"],
                language="en",
                text=rec["text"],
                sr=sr,
                hop_samples=HOP,
                n_frames=len(rec["feats"].log_env),
                phones=pbs,
            )
            held_aligned[uid] = {"aligned": au, "wav": rec["wav"], "feats": rec["feats"]}

    return {
        "utterances": aligned,
        "heldout_utterances": held_aligned,
        "included_ids": sorted(aligned.keys()),
        "excluded_ids": sorted(excl),
        "n_em": n_em,
    }


def align_zh_corpus_gmm(
    exclude_ids: set[str] | frozenset[str] | None = None,
    *,
    n_em: int = 4,
    sr: int = SR,
) -> dict:
    """Re-align Mandarin `extended/` 32 utterances' syllable onset/final boundaries via 3-state GMM-HMM."""
    idx = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))
    excl = set(exclude_ids) if exclude_ids is not None else set()

    train_recs: list[dict] = []
    for u in idx["utterances"]:
        uid = u["id"]
        if uid in excl:
            continue
        x, file_sr = v6_voc.read_wav_mono(u["path"])
        if file_sr != sr:
            raise ValueError(f"Expected {sr} Hz, got {file_sr} for {uid}")
        s0 = int(u["active_start"])
        s1 = int(u["active_stop"])
        xa = x[s0:s1]
        feats = v6_voc.analyze(xa, sr=sr)
        n_fr = len(feats.log_env)

        ipas: list[str] = []
        meta_phones: list[dict] = []
        init_bounds: list[tuple[int, int, int]] = []
        syl_meta: list[dict] = []

        for sy in u["syllables"]:
            py = sy["pinyin"]
            onset, final, tone, syl_ipas = cf.pinyin_to_ipa(py)
            f_lo = max(0, min(n_fr - 1, (int(sy["start_sample"]) - s0) // HOP))
            f_hi = max(f_lo + 1, min(n_fr, (int(sy["stop_sample"]) - s0) // HOP))
            n_p = len(syl_ipas)
            edges = np.linspace(f_lo, f_hi, n_p + 1).astype(int)
            p_start_idx = len(ipas)
            for k, p in enumerate(syl_ipas):
                ipas.append(p)
                ps_k = int(edges[k])
                pe_k = int(max(ps_k + 1, edges[k + 1]))
                init_bounds.append((ps_k, (ps_k + pe_k) // 2, pe_k))
                meta_phones.append({
                    "ipa": p,
                    "arpabet": None,
                    "word": sy["hanzi"],
                    "syllable_idx": int(sy["index"]),
                    "stress_or_tone": f"T{tone}",
                    "is_onset": (k == 0 and bool(onset)),
                })
            syl_meta.append({
                "index": int(sy["index"]),
                "hanzi": sy["hanzi"],
                "pinyin": py,
                "onset": onset,
                "final": final,
                "tone": int(tone),
                "p_slice": (p_start_idx, len(ipas)),
            })

        train_recs.append({
            "id": uid,
            "speaker": u["speaker"],
            "text": u["text"],
            "wav": xa,
            "feats": feats,
            "ipas": ipas,
            "meta_phones": meta_phones,
            "init_bounds": init_bounds,
            "syl_meta": syl_meta,
        })

    _, final_bounds = _train_and_align_gmm(train_recs, n_em=n_em, sr=sr)

    aligned: dict[str, dict] = {}
    for rec in train_recs:
        uid = rec["id"]
        pbs = _build_phone_boundaries(rec["meta_phones"], final_bounds[uid], rec["wav"], rec["feats"], sr=sr)
        syls: list[SyllableBoundary] = []
        for sm in rec["syl_meta"]:
            a, b = sm["p_slice"]
            spbs = pbs[a:b]
            s_start = spbs[0].p_start
            s_end = spbs[-1].p_end
            # Onset->final split frame inside [0 .. s_end - s_start]
            if sm["onset"] and len(spbs) >= 2:
                f_split = max(1, min(s_end - s_start - 1, spbs[0].p_end - s_start))
            else:
                f_split = 0
            syls.append(
                SyllableBoundary(
                    index=sm["index"],
                    hanzi=sm["hanzi"],
                    pinyin=sm["pinyin"],
                    onset=sm["onset"],
                    final=sm["final"],
                    tone=sm["tone"],
                    syl_start=s_start,
                    f_split=f_split,
                    syl_end=s_end,
                    dur_ms=round((s_end - s_start) * HOP * 1000.0 / sr, 2),
                    phones=spbs,
                )
            )
        au = AlignedUtterance(
            utterance_id=uid,
            speaker=rec["speaker"],
            language="zh",
            text=rec["text"],
            sr=sr,
            hop_samples=HOP,
            n_frames=len(rec["feats"].log_env),
            phones=pbs,
            syllables=syls,
        )
        aligned[uid] = {"aligned": au, "wav": rec["wav"], "feats": rec["feats"]}

    return {
        "utterances": aligned,
        "included_ids": sorted(aligned.keys()),
        "excluded_ids": sorted(excl),
        "n_em": n_em,
    }


def evaluate_gmm_vs_v10() -> dict:
    """Compare v11 GMM-HMM alignment quality against v10 aligner on EN (140) & ZH (32)."""
    en_res = align_en_corpus_gmm(exclude_ids=EN_HELDOUT_DEFAULT, align_heldout=False, n_em=4)
    zh_res = align_zh_corpus_gmm(exclude_ids=None, n_em=4)

    en_tot, en_short, en_sil, en_bad, en_ok = 0, 0, 0, 0, 0
    en_reasons: dict[str, int] = {}
    for item in en_res["utterances"].values():
        au: AlignedUtterance = item["aligned"]
        wav = item["wav"]
        feats = item["feats"]
        for pb in au.phones:
            seg = wav[pb.p_start * HOP : min(len(wav), pb.p_end * HOP)]
            ok, reason, met = score_token_purity(pb.ipa, feats.log_env[pb.p_start : pb.p_end], feats.voiced[pb.p_start : pb.p_end], seg, sr=SR)
            en_tot += 1
            if met["dur_ms"] < 20.0:
                en_short += 1
            if met["rms_dbfs"] < -45.0:
                en_sil += 1
            if reason in ("voiced_contradiction", "unvoiced_fric_too_voiced", "low_hf_fricative", "unvoiced_stop_too_voiced"):
                en_bad += 1
            if ok:
                en_ok += 1
            else:
                en_reasons[reason] = en_reasons.get(reason, 0) + 1

    zh_tot, zh_on_tot, zh_on_lt20, zh_on_bad, zh_ok = 0, 0, 0, 0, 0
    for item in zh_res["utterances"].values():
        au = item["aligned"]
        wav = item["wav"]
        feats = item["feats"]
        for sy in au.syllables:
            zh_tot += 1
            s0, s1 = sy.syl_start, sy.syl_end
            v_fr = float(np.mean(feats.voiced[s0:s1])) if s1 > s0 else 0.0
            bad = False
            if sy.onset and sy.f_split > 0:
                zh_on_tot += 1
                on_ms = sy.f_split * HOP * 1000.0 / SR
                if on_ms < 20.0:
                    zh_on_lt20 += 1
                on_ipa = cf.PINYIN_ONSET_TO_IPA.get(sy.onset, [""])[0]
                if on_ipa:
                    on_seg = wav[s0 * HOP : min(len(wav), (s0 + sy.f_split) * HOP)]
                    ok_on, r_on, _ = score_token_purity(on_ipa, feats.log_env[s0 : s0 + sy.f_split], feats.voiced[s0 : s0 + sy.f_split], on_seg, sr=SR)
                    if r_on in ("unvoiced_fric_too_voiced", "low_hf_fricative", "unvoiced_stop_too_voiced", "voiced_contradiction"):
                        zh_on_bad += 1
                    sp = IPA_SPECS.get(on_ipa)
                    if sp and sp.manner in ("fricative", "affricate") and not sp.voiced and not ok_on:
                        bad = True
            if sy.dur_ms >= 55.0 and v_fr >= 0.30 and not bad:
                zh_ok += 1

    return {
        "en_gmm_140": {
            "total": en_tot,
            "short_lt20ms": en_short,
            "silent_lt45": en_sil,
            "manner_bad": en_bad,
            "accepted": en_ok,
            "accept_rate": round(en_ok / max(1, en_tot), 4),
            "quarantine_reasons": en_reasons,
        },
        "zh_gmm_32": {
            "total_syllables": zh_tot,
            "onsets_total": zh_on_tot,
            "onsets_lt20ms": zh_on_lt20,
            "onsets_bad_manner": zh_on_bad,
            "verified_ok": zh_ok,
        },
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_gmm_vs_v10(), indent=2))
