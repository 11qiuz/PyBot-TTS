"""Extract speaker-anchored ZH donor acoustic inventories for zero-ML cross-lingual synthesis."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from threading import Lock
from typing import Any

import numpy as np
from scipy.fft import rfftfreq

from exp_20_pybot_larynx import N_BAP, N_BINS, N_FFT, SR
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB
from exp_20_pybot_larynx.larynx.spectral_decomp import (
    decompose_frame_source_filter,
    lsf_to_formants,
)
from exp_20_pybot_larynx.larynx.synth import cep_to_log_env
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS

_DB_TO_NEPER = 1.0 / NEPER_TO_DB
_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_FREQS_SAFE = np.maximum(_FREQS, 80.0)
_PHYS_BAP_TARGET = np.asarray([0.044, 0.25, 0.46, 0.74, 0.88, 0.93], dtype=np.float64)

_DONOR_CACHE_LOCK = Lock()
_DONOR_CACHE: OrderedDict[int, "SpeakerDonorBank"] = OrderedDict()
_MAX_DONOR_CACHE = 8


@dataclass(frozen=True)
class SonorantDonor:
    """Real-speaker 513-bin voiced donor envelope, BAP6, measured formants, and linear power."""

    ipa: str
    env: np.ndarray            # (513,) natural-log amplitude in KB scale
    bap6: np.ndarray           # (6,)
    formants: np.ndarray       # (3,) Hz [F1, F2, F3]
    power: float
    traj_env: np.ndarray | None = None      # Optional (T_bins, 513) for diphthongs
    traj_bap6: np.ndarray | None = None     # Optional (T_bins, 6)
    traj_formants: np.ndarray | None = None # Optional (T_bins, 3)


@dataclass(frozen=True)
class ObstruentDonor:
    """Real-speaker 513-bin unvoiced turbulence / burst donor envelope in KB scale."""

    ipa: str
    env: np.ndarray            # (513,) steady turbulence/release
    burst_env: np.ndarray      # (513,) transient burst (same or +2 dB shaped)
    bap6: np.ndarray           # (6,)


def _compute_log_power(log_env: np.ndarray) -> float:
    return float(np.sum(np.exp(2.0 * np.clip(log_env, -16.0, 6.0))))


def _mix_log_envs(items: list[tuple[np.ndarray, float]], *, shift_db: float = 0.0) -> np.ndarray:
    """Blend multiple 513-bin natural-log envelopes and optionally apply dB offset."""
    acc = np.zeros(N_BINS, dtype=np.float64)
    w_sum = 0.0
    for env, w in items:
        acc += float(w) * np.asarray(env, dtype=np.float64)
        w_sum += float(w)
    out = acc / max(1e-8, w_sum)
    if abs(shift_db) > 1e-6:
        out = out + shift_db * _DB_TO_NEPER
    return np.clip(out, -16.0, 6.0)


def _estimate_formants_safe(
    log_env: np.ndarray,
    fallback: tuple[float, float, float],
) -> np.ndarray:
    """Extract monotonic (F1, F2, F3) via LSF-22 with clean fallback per pole."""
    try:
        sf = decompose_frame_source_filter(log_env, order=22)
        hz, _ = lsf_to_formants(sf.lsf, sr=SR, f_min=200.0, f_max=4500.0)
    except Exception:
        hz = np.zeros(0, dtype=np.float64)

    f1_fb, f2_fb, f3_fb = float(fallback[0]), float(fallback[1]), float(fallback[2])
    c1 = hz[(hz >= 230.0) & (hz <= 1220.0)]
    f1 = float(c1[np.argmin(np.abs(c1 - f1_fb))]) if c1.size else f1_fb
    f1 = float(np.clip(f1, 240.0, 1200.0))

    c2 = hz[(hz >= max(f1 + 220.0, 740.0)) & (hz <= 3050.0)]
    f2 = float(c2[np.argmin(np.abs(c2 - f2_fb))]) if c2.size else max(f1 + 260.0, f2_fb)
    f2 = float(np.clip(f2, max(f1 + 220.0, 740.0), 3050.0))

    c3 = hz[(hz >= max(f2 + 280.0, 1800.0)) & (hz <= 4250.0)]
    f3 = float(c3[np.argmin(np.abs(c3 - f3_fb))]) if c3.size else max(f2 + 320.0, f3_fb)
    f3 = float(np.clip(f3, max(f2 + 280.0, 1800.0), 4250.0))

    return np.array([f1, f2, f3], dtype=np.float64)


class SpeakerDonorBank:
    """Speaker-specific ZH acoustic donor bank mapped onto the full EN+ZH IPA inventory."""

    def __init__(self, kb: Any) -> None:
        self.kb_id = id(kb)
        self.speaker_f0_med_hz = float(getattr(kb, "speaker_f0_med_hz", 248.0))
        self.ref_log_amp = float(getattr(kb, "ref_log_amp", 0.0))

        # 1. Extract helper closures anchored to kb.fin_f / kb.on_only with safe global fallbacks
        def fin_group(key: str) -> Any:
            grp = kb.fin_f.get(key)
            if grp is not None and len(grp.cep) > 0:
                return grp
            return None

        def on_group(key: str) -> Any:
            grp = kb.on_only.get(key)
            if grp is not None and len(grp.cep) > 0:
                return grp
            return None

        def fin_steady(key: str, bins: tuple[int, ...] = (3, 4, 5)) -> tuple[np.ndarray, np.ndarray]:
            grp = fin_group(key)
            if grp is not None:
                m_idx = int(np.argmax(grp.cluster_sz))
                valid_b = [min(max(0, b), grp.cep.shape[1] - 1) for b in bins]
                cep_m = np.mean(grp.cep[m_idx, valid_b], axis=0)
                bap_m = np.mean(grp.bap[m_idx, valid_b], axis=0)
            else:
                g_cep, g_bap, _ = kb.fin_global
                valid_b = [min(max(0, b), len(g_cep) - 1) for b in bins]
                cep_m = np.mean(g_cep[valid_b], axis=0)
                bap_m = np.mean(g_bap[valid_b], axis=0)
            env = cep_to_log_env(cep_m[None, :], n_bins=N_BINS)[0] + self.ref_log_amp
            return np.clip(env, -16.0, 6.0), np.clip(bap_m, 0.005, 0.99)

        def fin_traj(key: str, bins: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)) -> tuple[np.ndarray, np.ndarray]:
            grp = fin_group(key)
            if grp is not None:
                m_idx = int(np.argmax(grp.cluster_sz))
                valid_b = [min(max(0, b), grp.cep.shape[1] - 1) for b in bins]
                cep_t = grp.cep[m_idx, valid_b]
                bap_t = grp.bap[m_idx, valid_b]
            else:
                g_cep, g_bap, _ = kb.fin_global
                valid_b = [min(max(0, b), len(g_cep) - 1) for b in bins]
                cep_t = g_cep[valid_b]
                bap_t = g_bap[valid_b]
            envs = cep_to_log_env(cep_t, n_bins=N_BINS) + self.ref_log_amp
            return np.clip(envs, -16.0, 6.0), np.clip(bap_t, 0.005, 0.99)

        def on_steady(key: str, bins: tuple[int, ...] = (1, 2, 3)) -> tuple[np.ndarray, np.ndarray]:
            grp = on_group(key)
            if grp is not None:
                m_idx = int(np.argmax(grp.cluster_sz))
                valid_b = [min(max(0, b), grp.cep.shape[1] - 1) for b in bins]
                cep_m = np.mean(grp.cep[m_idx, valid_b], axis=0)
                bap_m = np.mean(grp.bap[m_idx, valid_b], axis=0)
            else:
                g_cep, g_bap, _ = kb.on_global
                valid_b = [min(max(0, b), len(g_cep) - 1) for b in bins]
                cep_m = np.mean(g_cep[valid_b], axis=0)
                bap_m = np.mean(g_bap[valid_b], axis=0)
            env = cep_to_log_env(cep_m[None, :], n_bins=N_BINS)[0] + self.ref_log_amp
            return np.clip(env, -16.0, 6.0), np.clip(bap_m, 0.005, 0.99)

        # 2. Extract core Mandarin steady anchors
        env_a, bap_a = fin_steady("a", (3, 4, 5))
        env_o, bap_o = fin_steady("o", (3, 4, 5))
        env_e, bap_e = fin_steady("e", (3, 4, 5))
        env_i, bap_i = fin_steady("i", (3, 4, 5))
        env_u, bap_u = fin_steady("u", (3, 4, 5))
        env_v, bap_v = fin_steady("v", (3, 4, 5))
        env_er, bap_er = fin_steady("er", (3, 4, 5))
        env_iz, bap_iz = fin_steady("iz", (3, 4, 5))
        env_ie, bap_ie = fin_steady("ie", (5, 6, 7))
        env_uo, bap_uo = fin_steady("uo", (5, 6, 7))
        env_eng_coda, bap_eng_coda = fin_steady("eng", (6, 7, 8))
        env_ang_coda, bap_ang_coda = fin_steady("ang", (6, 7, 8))

        env_i_on, bap_i_on = fin_steady("i", (1, 2, 3))
        env_u_on, bap_u_on = fin_steady("u", (1, 2, 3))

        env_m, bap_m = on_steady("m", (1, 2, 3))
        env_n, bap_n = on_steady("n", (1, 2, 3))
        env_l, bap_l = on_steady("l", (1, 2, 3))
        env_r, bap_r = on_steady("r", (1, 2, 3))

        # 3. Measure anatomical vocal-tract formant scaling (speaker vs base IPA_SPECS)
        anchor_pairs = [
            ("a", env_a, IPA_SPECS["a"].formants),
            ("o", env_o, IPA_SPECS["o"].formants),
            ("e", env_e, IPA_SPECS["ɤ"].formants),
            ("i", env_i, IPA_SPECS["i"].formants),
            ("u", env_u, IPA_SPECS["u"].formants),
        ]
        r1s: list[float] = []
        r2s: list[float] = []
        r3s: list[float] = []
        for _, a_env, fb in anchor_pairs:
            fm = _estimate_formants_safe(a_env, fb)
            r1s.append(float(fm[0] / max(150.0, fb[0])))
            r2s.append(float(fm[1] / max(500.0, fb[1])))
            r3s.append(float(fm[2] / max(1500.0, fb[2])))

        self.scale_f1 = float(np.clip(np.median(r1s), 0.85, 1.24))
        self.scale_f2 = float(np.clip(np.median(r2s), 0.86, 1.22))
        self.scale_f3 = float(np.clip(np.median(r3s), 0.90, 1.26))

        # 4. Reference LTAS, linear power, and post-Rule-1-aligned BAP6
        ltas_keys = ("a", "o", "e", "i", "u", "er", "ai", "ei", "ao", "ou")
        ltas_envs = []
        ltas_baps = []
        ltas_pwrs = []
        for k in ltas_keys:
            ke, kbap = fin_steady(k, (3, 4, 5))
            ltas_envs.append(ke)
            ltas_baps.append(kbap)
            ltas_pwrs.append(_compute_log_power(ke))

        self.voiced_ltas_env = np.mean(np.stack(ltas_envs, axis=0), axis=0)
        kb_bap_med = np.median(np.stack(ltas_baps, axis=0), axis=0)
        self.voiced_bap6_med = np.clip(0.25 * kb_bap_med + 0.75 * _PHYS_BAP_TARGET, 0.015, 0.985)
        self.voiced_ref_power = float(np.median(ltas_pwrs))

        # Low-pass chest voice bar for voiced obstruents anchored on speaker's /m/
        lp_roll = -12.0 * np.log10(1.0 + (_FREQS_SAFE / 750.0) ** 4) * _DB_TO_NEPER
        self.voicing_bar_env = np.clip(env_m + lp_roll, -16.0, 6.0)

        # 5. Build complete Sonorant / Vowel / Nasal / Liquid Donor Table
        self.sonorant_donors: dict[str, SonorantDonor] = {}

        def _reg_sonorant(ipa: str, env: np.ndarray, bap_raw: np.ndarray) -> None:
            fb_raw = IPA_SPECS[ipa].formants if ipa in IPA_SPECS else (500.0, 1500.0, 2800.0)
            fb = (fb_raw[0] * self.scale_f1, fb_raw[1] * self.scale_f2, fb_raw[2] * self.scale_f3)
            fm = _estimate_formants_safe(env, fb)
            bap_al = np.clip(0.25 * bap_raw + 0.75 * self.voiced_bap6_med, 0.015, 0.985)
            self.sonorant_donors[ipa] = SonorantDonor(
                ipa=ipa,
                env=env,
                bap6=bap_al,
                formants=fm,
                power=_compute_log_power(env),
            )

        _reg_sonorant("i", env_i, bap_i)
        _reg_sonorant("u", env_u, bap_u)
        _reg_sonorant("ʉ", env_u, bap_u)
        _reg_sonorant("a", env_a, bap_a)
        _reg_sonorant("ɑ", _mix_log_envs([(env_a, 0.65), (env_o, 0.35)]), 0.65 * bap_a + 0.35 * bap_o)
        _reg_sonorant("o", env_o, bap_o)
        _reg_sonorant("ɔ", _mix_log_envs([(env_o, 0.70), (env_a, 0.30)]), 0.70 * bap_o + 0.30 * bap_a)
        _reg_sonorant("e", env_ie, bap_ie)
        _reg_sonorant("ɛ", _mix_log_envs([(env_ie, 0.55), (env_a, 0.45)]), 0.55 * bap_ie + 0.45 * bap_a)
        _reg_sonorant("æ", _mix_log_envs([(env_a, 0.48), (env_ie, 0.52)]), 0.48 * bap_a + 0.52 * bap_ie)
        _reg_sonorant("ɤ", env_e, bap_e)
        _reg_sonorant("ʌ", _mix_log_envs([(env_e, 0.55), (env_a, 0.45)]), 0.55 * bap_e + 0.45 * bap_a)
        _reg_sonorant("ə", _mix_log_envs([(env_e, 0.65), (env_o, 0.35)]), 0.65 * bap_e + 0.35 * bap_o)
        _reg_sonorant("ɨ", _mix_log_envs([(env_e, 0.65), (env_o, 0.35)]), 0.65 * bap_e + 0.35 * bap_o)
        _reg_sonorant("ɪ", _mix_log_envs([(env_i, 0.58), (env_ie, 0.42)]), 0.58 * bap_i + 0.42 * bap_ie)
        _reg_sonorant("ʊ", _mix_log_envs([(env_u, 0.58), (env_o, 0.42)]), 0.58 * bap_u + 0.42 * bap_o)
        _reg_sonorant("y", env_v, bap_v)
        _reg_sonorant("ɚ", env_er, bap_er)
        _reg_sonorant("ɝ", env_er, bap_er)
        _reg_sonorant("ɻ̩", env_er, bap_er)
        _reg_sonorant("ɹ̩", env_iz, bap_iz)
        _reg_sonorant("j", env_i_on, bap_i_on)
        _reg_sonorant("ɥ", env_i_on, bap_i_on)
        _reg_sonorant("w", env_u_on, bap_u_on)
        _reg_sonorant("m", env_m, bap_m)
        _reg_sonorant("m̩", env_m, bap_m)
        _reg_sonorant("n", env_n, bap_n)
        _reg_sonorant("n̩", env_n, bap_n)
        _reg_sonorant(
            "ŋ",
            _mix_log_envs([(env_eng_coda, 0.50), (env_ang_coda, 0.50)]),
            0.50 * bap_eng_coda + 0.50 * bap_ang_coda,
        )
        _reg_sonorant("l", env_l, bap_l)
        _reg_sonorant("l̩", env_l, bap_l)
        _reg_sonorant("ɹ", _mix_log_envs([(env_r, 0.55), (env_er, 0.45)]), 0.55 * bap_r + 0.45 * bap_er)
        _reg_sonorant("ɻ", _mix_log_envs([(env_r, 0.55), (env_er, 0.45)]), 0.55 * bap_r + 0.45 * bap_er)
        _reg_sonorant("ɾ", _mix_log_envs([(env_l, 0.60), (env_e, 0.40)]), 0.60 * bap_l + 0.40 * bap_e)

        # 6. Diphthong Trajectory Donors
        self.diphthong_donors: dict[str, SonorantDonor] = {}

        def _reg_diph(ipa: str, traj_envs: np.ndarray, traj_baps: np.ndarray) -> None:
            sp = IPA_SPECS[ipa]
            f_start = np.array(sp.formants, dtype=np.float64) * np.array([self.scale_f1, self.scale_f2, self.scale_f3])
            f_end = np.array(sp.end_formants or sp.formants, dtype=np.float64) * np.array([self.scale_f1, self.scale_f2, self.scale_f3])
            n_b = traj_envs.shape[0]
            fm_traj = np.empty((n_b, 3), dtype=np.float64)
            bap_traj = np.empty((n_b, N_BAP), dtype=np.float64)
            for bi in range(n_b):
                u = bi / max(1, n_b - 1)
                fb = (1.0 - u) * f_start + u * f_end
                fm_traj[bi] = _estimate_formants_safe(traj_envs[bi], (float(fb[0]), float(fb[1]), float(fb[2])))
                bap_traj[bi] = np.clip(0.25 * traj_baps[bi] + 0.75 * self.voiced_bap6_med, 0.015, 0.985)
            mid = n_b // 2
            self.diphthong_donors[ipa] = SonorantDonor(
                ipa=ipa,
                env=traj_envs[mid],
                bap6=bap_traj[mid],
                formants=fm_traj[mid],
                power=_compute_log_power(traj_envs[mid]),
                traj_env=traj_envs,
                traj_bap6=bap_traj,
                traj_formants=fm_traj,
            )

        for d_ipa, kb_key in (("aɪ", "ai"), ("aʊ", "ao"), ("eɪ", "ei"), ("oʊ", "ou")):
            te, tb = fin_traj(kb_key, (1, 2, 3, 4, 5, 6, 7))
            _reg_diph(d_ipa, te, tb)

        # Morph /ɔɪ/ from real /ɔ/ donor to real /i/ donor
        don_aw = self.sonorant_donors["ɔ"]
        don_iy = self.sonorant_donors["i"]
        n_oi = 7
        oi_envs = np.stack(
            [_mix_log_envs([(don_aw.env, 1.0 - u), (don_iy.env, u)]) for u in np.linspace(0.0, 1.0, n_oi)],
            axis=0,
        )
        oi_baps = np.stack([(1.0 - u) * don_aw.bap6 + u * don_iy.bap6 for u in np.linspace(0.0, 1.0, n_oi)], axis=0)
        _reg_diph("ɔɪ", oi_envs, oi_baps)

        # 7. Real-speaker Unvoiced / Obstruent Donors
        self.obstruent_donors: dict[str, ObstruentDonor] = {}
        env_f, bap_f = on_steady("f", (1, 2, 3))
        env_s, bap_s = on_steady("s", (1, 2, 3))
        env_sh, bap_sh = on_steady("sh", (1, 2, 3))
        env_x, bap_x = on_steady("x", (1, 2, 3))
        env_h, bap_h = on_steady("h", (1, 2, 3))
        env_p, bap_p = on_steady("p", (2, 3, 4))
        env_t, bap_t = on_steady("t", (2, 3, 4))
        env_k, bap_k = on_steady("k", (2, 3, 4))
        env_ch, bap_ch = on_steady("ch", (2, 3, 4))
        env_q, bap_q = on_steady("q", (2, 3, 4))
        env_c, bap_c = on_steady("c", (2, 3, 4))

        def _reg_obs(ipa: str, env: np.ndarray, bap_raw: np.ndarray, burst_boost_db: float = 2.0) -> None:
            b_env = np.clip(env + burst_boost_db * _DB_TO_NEPER, -16.0, 6.0)
            self.obstruent_donors[ipa] = ObstruentDonor(
                ipa=ipa,
                env=env,
                burst_env=b_env,
                bap6=np.clip(bap_raw, 0.40, 0.995),
            )

        _reg_obs("f", env_f, bap_f, burst_boost_db=0.0)
        _reg_obs("v", env_f, bap_f, burst_boost_db=0.0)
        env_th = _mix_log_envs([(env_f, 0.68), (env_s, 0.32)], shift_db=-2.5)
        _reg_obs("θ", env_th, 0.68 * bap_f + 0.32 * bap_s, burst_boost_db=0.0)
        _reg_obs("ð", env_th, 0.68 * bap_f + 0.32 * bap_s, burst_boost_db=0.0)
        _reg_obs("s", env_s, bap_s, burst_boost_db=0.0)
        _reg_obs("z", env_s, bap_s, burst_boost_db=0.0)
        env_sh_en = _mix_log_envs([(env_sh, 0.62), (env_x, 0.38)])
        _reg_obs("ʃ", env_sh_en, 0.62 * bap_sh + 0.38 * bap_x, burst_boost_db=0.0)
        _reg_obs("ʒ", env_sh_en, 0.62 * bap_sh + 0.38 * bap_x, burst_boost_db=0.0)
        _reg_obs("ʂ", env_sh, bap_sh, burst_boost_db=0.0)
        _reg_obs("ɕ", env_x, bap_x, burst_boost_db=0.0)
        _reg_obs("h", env_h, bap_h, burst_boost_db=0.0)
        _reg_obs("x", env_h, bap_h, burst_boost_db=0.0)
        _reg_obs("ʍ", env_h, bap_h, burst_boost_db=0.0)

        _reg_obs("p", env_p, bap_p, burst_boost_db=2.0)
        _reg_obs("pʰ", env_p, bap_p, burst_boost_db=2.5)
        _reg_obs("b", _mix_log_envs([(env_p, 1.0)], shift_db=-4.0), bap_p, burst_boost_db=1.5)
        _reg_obs("t", env_t, bap_t, burst_boost_db=2.0)
        _reg_obs("tʰ", env_t, bap_t, burst_boost_db=2.5)
        _reg_obs("d", _mix_log_envs([(env_t, 1.0)], shift_db=-4.0), bap_t, burst_boost_db=1.5)
        _reg_obs("k", env_k, bap_k, burst_boost_db=2.0)
        _reg_obs("kʰ", env_k, bap_k, burst_boost_db=2.5)
        _reg_obs("ɡ", _mix_log_envs([(env_k, 1.0)], shift_db=-4.0), bap_k, burst_boost_db=1.5)

        env_tch = _mix_log_envs([(env_ch, 0.62), (env_q, 0.38)])
        _reg_obs("tʃ", env_tch, 0.62 * bap_ch + 0.38 * bap_q, burst_boost_db=2.0)
        _reg_obs("dʒ", _mix_log_envs([(env_tch, 1.0)], shift_db=-2.5), 0.62 * bap_ch + 0.38 * bap_q, burst_boost_db=1.5)
        _reg_obs("ts", env_c, bap_c, burst_boost_db=1.5)
        _reg_obs("tsʰ", env_c, bap_c, burst_boost_db=2.0)
        _reg_obs("tɕ", env_q, bap_q, burst_boost_db=1.5)
        _reg_obs("tɕʰ", env_q, bap_q, burst_boost_db=2.0)
        _reg_obs("ʈʂ", env_ch, bap_ch, burst_boost_db=1.5)
        _reg_obs("ʈʂʰ", env_ch, bap_ch, burst_boost_db=2.0)

        # Unvoiced speaker reference LTAS for diagnostics & normalization
        uv_ref_envs = [self.obstruent_donors[k].env for k in ("f", "s", "ʃ", "h", "p", "t", "k", "tʃ")]
        self.unvoiced_ltas_env = np.mean(np.stack(uv_ref_envs, axis=0), axis=0)
        self.unvoiced_ref_power = float(np.median([_compute_log_power(e) for e in uv_ref_envs]))

    def get_sonorant_donor(self, ipa: str) -> SonorantDonor:
        if ipa in self.diphthong_donors:
            return self.diphthong_donors[ipa]
        if ipa in self.sonorant_donors:
            return self.sonorant_donors[ipa]
        return self.sonorant_donors["ə"]

    def get_obstruent_donor(self, ipa: str) -> ObstruentDonor:
        if ipa in self.obstruent_donors:
            return self.obstruent_donors[ipa]
        return self.obstruent_donors["h"]


def get_speaker_donor_bank(kb: Any) -> SpeakerDonorBank:
    """Return cached :class:`SpeakerDonorBank` for ``kb`` (thread-safe, <= 8 entries)."""
    key = id(kb)
    with _DONOR_CACHE_LOCK:
        cached = _DONOR_CACHE.get(key)
        if cached is not None:
            _DONOR_CACHE.move_to_end(key)
            return cached
    bank = SpeakerDonorBank(kb)
    with _DONOR_CACHE_LOCK:
        _DONOR_CACHE[key] = bank
        _DONOR_CACHE.move_to_end(key)
        while len(_DONOR_CACHE) > _MAX_DONOR_CACHE:
            _DONOR_CACHE.popitem(last=False)
    return bank
