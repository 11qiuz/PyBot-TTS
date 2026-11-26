"""Cross-lingual IPA spectral-envelope warp and syllable template lookup."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
from scipy.fft import rfftfreq

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exp_v5_arch.crossling_frontend import PINYIN_FINAL_TO_IPA, PINYIN_ONSET_TO_IPA  # noqa: E402
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS, PhoneSpec  # noqa: E402
from exp_v7_pybot_tts import N_FFT, SR  # noqa: E402
from exp_v7_pybot_tts.unit_bank import K_PHONE, UnitBank, UnitTemplate, _resample_frames  # noqa: E402


# Nearest Mandarin / cross-lingual anchor candidates per IPA symbol (ordered by acoustic similarity)
_ANCHOR_CANDIDATES: dict[str, tuple[str, ...]] = {
    # Vowels
    "a": ("a", "ɑ", "ɛ", "ə"),
    "ɑ": ("ɑ", "a", "o", "ɤ"),
    "æ": ("a", "ɛ", "ɑ", "e"),
    "ʌ": ("a", "ɤ", "ɑ", "ə"),
    "ɔ": ("o", "ɑ", "u", "ʊ"),
    "o": ("o", "u", "ɑ", "ʊ"),
    "ɤ": ("ɤ", "ə", "a", "o"),
    "e": ("e", "ɛ", "i", "ə"),
    "ɛ": ("ɛ", "e", "a", "i"),
    "ə": ("ə", "ɤ", "ɛ", "a"),
    "ɨ": ("i", "ə", "ɤ", "ɹ̩"),
    "ɪ": ("i", "e", "ɛ", "ə"),
    "i": ("i", "e", "y", "ɪ"),
    "y": ("y", "i", "u", "ɥ"),
    "u": ("u", "o", "ʊ", "ɑ"),
    "ʉ": ("u", "y", "ə", "o"),
    "ʊ": ("u", "o", "ə", "ɤ"),
    "ɚ": ("ɚ", "ɻ̩", "ɤ", "ə"),
    "ɝ": ("ɚ", "ɻ̩", "ɤ", "ə"),
    "ɻ̩": ("ɻ̩", "ɚ", "ɹ̩", "ɤ"),
    "ɹ̩": ("ɹ̩", "ɻ̩", "i", "ə"),
    # Glides / nasals / liquids
    "j": ("j", "i", "e"),
    "w": ("w", "u", "o"),
    "ɥ": ("ɥ", "y", "j", "i"),
    "m": ("m", "n", "ŋ"),
    "m̩": ("m", "n", "ŋ"),
    "n": ("n", "m", "ŋ"),
    "n̩": ("n", "m", "ŋ"),
    "ŋ": ("ŋ", "n", "m"),
    "l": ("l", "n", "ɻ"),
    "l̩": ("l", "n", "ɻ"),
    "ɹ": ("ɻ", "ɚ", "w", "l"),
    "ɻ": ("ɻ", "ɚ", "l", "ʂ"),
    "ɾ": ("l", "t", "n"),
    # Fricatives
    "f": ("f", "x", "s", "ʂ"),
    "v": ("f", "w", "m", "x"),
    "θ": ("f", "s", "ɕ", "x"),
    "ð": ("f", "s", "l", "n"),
    "s": ("s", "ɕ", "ʂ", "f"),
    "z": ("s", "ts", "ɕ", "l"),
    "ʃ": ("ʂ", "ɕ", "s"),
    "ʒ": ("ʂ", "ɕ", "ɻ"),
    "ʂ": ("ʂ", "ɕ", "s"),
    "ɕ": ("ɕ", "ʂ", "s"),
    "x": ("x", "f", "ʂ"),
    "h": ("x", "f", "ə"),
    "ʍ": ("x", "f", "w"),
    # Affricates
    "ts": ("ts", "tsʰ", "s"),
    "tsʰ": ("tsʰ", "ts", "s"),
    "tɕ": ("tɕ", "tɕʰ", "ɕ"),
    "tɕʰ": ("tɕʰ", "tɕ", "ɕ"),
    "ʈʂ": ("ʈʂ", "ʈʂʰ", "ʂ"),
    "ʈʂʰ": ("ʈʂʰ", "ʈʂ", "ʂ"),
    "tʃ": ("ʈʂʰ", "tɕʰ", "ʂ"),
    "dʒ": ("ʈʂ", "tɕ", "ʂ"),
    # Stops
    "p": ("pʰ", "p", "t"),
    "pʰ": ("pʰ", "p", "tʰ"),
    "b": ("p", "m", "pʰ"),
    "t": ("tʰ", "t", "k"),
    "tʰ": ("tʰ", "t", "kʰ"),
    "d": ("t", "n", "tʰ"),
    "k": ("kʰ", "k", "t"),
    "kʰ": ("kʰ", "k", "tʰ"),
    "ɡ": ("k", "ŋ", "kʰ"),
}

_DIPHTHONG_SPLITS: dict[str, tuple[str, str]] = {
    "aɪ": ("ɑ", "i"),
    "aʊ": ("ɑ", "u"),
    "eɪ": ("e", "i"),
    "oʊ": ("o", "u"),
    "ɔɪ": ("ɔ", "i"),
}


def _lorentzian_log_mag(
    freqs: np.ndarray,
    formants: tuple[float, float, float],
    bws: tuple[float, float, float],
    anti_zero: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Smooth 3-formant log-magnitude response (plus optional anti-resonance notch) in nepers."""
    log_h = np.zeros_like(freqs)
    for f0, bw in zip(formants, bws):
        # All-pole resonance peak shape (normalized near DC, capped cleanly)
        denom = 1.0 + ((freqs - f0) / max(30.0, bw)) ** 2
        log_h += -0.5 * np.log(denom)
    if anti_zero is not None:
        fz, bwz, depth_db = anti_zero
        notch = 1.0 - (1.0 - 10.0 ** (-depth_db / 20.0)) / (1.0 + ((freqs - fz) / max(40.0, bwz)) ** 2)
        log_h += np.log(np.maximum(0.05, notch))
    return log_h - float(np.mean(log_h))


def _pick_donor(bank: UnitBank, ipa: str) -> tuple[str, UnitTemplate]:
    """Find best available donor template in bank.ipa_phones (or fallback to bank.onsets/finals)."""
    for cand in _ANCHOR_CANDIDATES.get(ipa, ()):
        if cand in bank.ipa_phones:
            return cand, bank.ipa_phones[cand]
    # Last resort: pick first available vowel or any template in bank
    for fallback in ("ə", "a", "i", "u", "s", "t"):
        if fallback in bank.ipa_phones:
            return fallback, bank.ipa_phones[fallback]
    if bank.ipa_phones:
        k = next(iter(bank.ipa_phones))
        return k, bank.ipa_phones[k]
    if bank.finals:
        k = next(iter(bank.finals))
        return k, bank.finals[k]
    raise ValueError("UnitBank has no templates to look up or warp")


def _template_to_dict(
    log_env: np.ndarray,
    bap: np.ndarray,
    voiced_ratio: np.ndarray,
    dur_ms: float,
    source_mode: str,
) -> dict:
    return {
        "log_env": np.asarray(log_env, dtype=np.float32),
        "bap": np.clip(bap, 0.005, 0.99).astype(np.float32),
        "voiced": (np.asarray(voiced_ratio) >= 0.5).astype(bool),
        "voiced_ratio": np.asarray(voiced_ratio, dtype=np.float32),
        "dur_ms": float(dur_ms),
        "source_mode": source_mode,
    }


def _warp_vowel_env(
    env_src: np.ndarray,
    spec_src: PhoneSpec,
    spec_tgt: PhoneSpec,
    sr: int = SR,
    n_fft: int = N_FFT,
) -> np.ndarray:
    """Apply formant-ratio differential shaping from spec_src to spec_tgt while preserving real fine envelope."""
    freqs = rfftfreq(n_fft, 1.0 / sr)
    h_src = _lorentzian_log_mag(freqs, spec_src.formants, spec_src.bandwidths, spec_src.anti_zero)
    h_tgt = _lorentzian_log_mag(freqs, spec_tgt.formants, spec_tgt.bandwidths, spec_tgt.anti_zero)
    delta = np.clip(h_tgt - h_src, -1.15, 1.15)  # cap at +/- 10 dB
    # Taper correction above 3.6 kHz so natural upper vocal-tract resonances remain intact
    hi_taper = np.clip(1.0 - np.maximum(0.0, freqs - 3600.0) / 3000.0, 0.0, 1.0)
    delta *= hi_taper
    # Preserve per-frame power
    pre_pwr = np.sum(np.exp(2.0 * env_src), axis=1)
    env_w = env_src + delta[None, :]
    post_pwr = np.sum(np.exp(2.0 * env_w), axis=1)
    gain = 0.5 * np.log(np.maximum(pre_pwr, 1e-18) / np.maximum(post_pwr, 1e-18))
    return (env_w + gain[:, None]).astype(np.float32)


def _apply_fric_poles_diff(
    env_src: np.ndarray,
    poles_src: tuple[tuple[float, float, float], ...],
    poles_tgt: tuple[tuple[float, float, float], ...],
    sr: int = SR,
    n_fft: int = N_FFT,
) -> np.ndarray:
    """Shift fricative spectral centroid/poles from donor fricative to target IPA fricative."""
    freqs = rfftfreq(n_fft, 1.0 / sr)

    def _fric_shape(poles):
        s = np.zeros_like(freqs)
        for fp, bw, g_db in poles:
            s += (g_db / 8.6858896) / (1.0 + ((freqs - fp) / max(150.0, bw)) ** 2)
        return s - float(np.mean(s))

    delta = np.clip(_fric_shape(poles_tgt) - _fric_shape(poles_src), -1.1, 1.1)
    pre_pwr = np.sum(np.exp(2.0 * env_src), axis=1)
    env_w = env_src + delta[None, :]
    post_pwr = np.sum(np.exp(2.0 * env_w), axis=1)
    gain = 0.5 * np.log(np.maximum(pre_pwr, 1e-18) / np.maximum(post_pwr, 1e-18))
    return (env_w + gain[:, None]).astype(np.float32)


def _add_voicing_bar(
    env_src: np.ndarray,
    bap_src: np.ndarray,
    bank: UnitBank,
    vbar: float = 0.5,
    sr: int = SR,
    n_fft: int = N_FFT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend low-frequency harmonic voicing bar (< 450 Hz) into voiceless obstruent donor."""
    freqs = rfftfreq(n_fft, 1.0 / sr)
    # Use a real nasal or schwa template as natural voicing-bar low-band source if available
    v_anchor_key = "m" if "m" in bank.ipa_phones else ("ə" if "ə" in bank.ipa_phones else None)
    env_out = env_src.copy()
    if v_anchor_key is not None:
        v_env = bank.ipa_phones[v_anchor_key].log_env
        v_mean = np.mean(v_env, axis=0)
        low_w = np.clip(1.0 - np.maximum(0.0, freqs - 220.0) / 380.0, 0.0, 1.0) * vbar
        env_out = (1.0 - low_w[None, :]) * env_out + low_w[None, :] * v_mean[None, :]
    else:
        low_boost = 0.65 * vbar * np.exp(-0.5 * ((freqs - 180.0) / 140.0) ** 2)
        env_out = env_out + low_boost[None, :]

    bap_out = bap_src.copy()
    # Band 0 (60-1k) becomes largely periodic; Band 1 (1-2k) partially voiced; >=2k stays turbulent
    bap_out[:, 0] = np.minimum(bap_out[:, 0], 0.22 + 0.15 * (1.0 - vbar))
    bap_out[:, 1] = np.minimum(bap_out[:, 1], 0.55 + 0.15 * (1.0 - vbar))
    voi_out = np.ones(len(env_out), dtype=np.float32)
    return env_out.astype(np.float32), bap_out.astype(np.float32), voi_out


def lookup_ipa_template(bank: UnitBank, ipa_phone: str, is_onset: bool = False) -> dict:
    """Look up exact corpus IPA template, or construct via PyBot cross-lingual warp rules."""
    if ipa_phone in bank.ipa_phones:
        t = bank.ipa_phones[ipa_phone]
        return _template_to_dict(t.log_env, t.bap, t.voiced_ratio, t.dur_ms, "exact_ipa")

    # English diphthongs: stitch & blend two component vowels
    if ipa_phone in _DIPHTHONG_SPLITS:
        a_sym, b_sym = _DIPHTHONG_SPLITS[ipa_phone]
        ta = lookup_ipa_template(bank, a_sym, is_onset=False)
        tb = lookup_ipa_template(bank, b_sym, is_onset=False)
        k1 = ta["log_env"].shape[0]
        k2 = tb["log_env"].shape[0]
        env_cat = np.concatenate([ta["log_env"], tb["log_env"]], axis=0)
        bap_cat = np.concatenate([ta["bap"], tb["bap"]], axis=0)
        voi_cat = np.concatenate([ta["voiced_ratio"], tb["voiced_ratio"]], axis=0)
        # Blend middle 2 frames across seam
        if k1 >= 2 and k2 >= 2:
            mid = k1
            env_cat[mid - 1] = 0.65 * env_cat[mid - 1] + 0.35 * env_cat[mid]
            env_cat[mid] = 0.35 * ta["log_env"][-1] + 0.65 * env_cat[mid]
        env_r, bap_r, voi_r = _resample_frames(env_cat, bap_cat, voi_cat, K_PHONE + 2)
        spec_d = IPA_SPECS.get(ipa_phone)
        dur = spec_d.base_dur_ms if spec_d else (ta["dur_ms"] + tb["dur_ms"]) * 0.72
        return _template_to_dict(env_r, bap_r, voi_r, dur, "warped_ipa")

    donor_sym, donor = _pick_donor(bank, ipa_phone)
    spec_tgt = IPA_SPECS.get(ipa_phone)
    spec_src = IPA_SPECS.get(donor_sym)

    env = donor.log_env.copy()
    bap = donor.bap.copy()
    voi = donor.voiced_ratio.copy()
    dur = spec_tgt.base_dur_ms if spec_tgt is not None else donor.dur_ms

    if spec_tgt is not None and spec_src is not None:
        if spec_tgt.manner in ("vowel", "glide", "nasal", "liquid"):
            env = _warp_vowel_env(env, spec_src, spec_tgt, sr=bank.sr, n_fft=bank.n_fft)
            if spec_tgt.manner == "vowel":
                voi = np.ones(len(env), dtype=np.float32)
                bap = np.minimum(bap, np.array([[0.22, 0.55, 0.72, 0.88, 0.92]], dtype=np.float32))
        elif spec_tgt.manner in ("fricative", "affricate", "stop"):
            if spec_tgt.fric_poles and spec_src.fric_poles:
                env = _apply_fric_poles_diff(
                    env, spec_src.fric_poles, spec_tgt.fric_poles, sr=bank.sr, n_fft=bank.n_fft
                )
            if spec_tgt.voiced:
                env, bap, voi = _add_voicing_bar(
                    env, bap, bank, vbar=max(0.4, spec_tgt.voicing_bar), sr=bank.sr, n_fft=bank.n_fft
                )
            else:
                voi = np.zeros(len(env), dtype=np.float32)
                bap = np.maximum(bap, 0.88)

    return _template_to_dict(env, bap, voi, dur, "warped_ipa")


def lookup_syllable_template(bank: UnitBank, onset: str, final: str) -> dict:
    """Look up full Mandarin syllable template, or concatenate onset + final (with IPA fallback)."""
    if (onset, final) in bank.syllables:
        t = bank.syllables[(onset, final)]
        return _template_to_dict(t.log_env, t.bap, t.voiced_ratio, t.dur_ms, "exact_syllable")

    parts_env, parts_bap, parts_voi = [], [], []
    dur_ms = 0.0
    modes = []

    if onset:
        if onset in bank.onsets:
            to = bank.onsets[onset]
            parts_env.append(to.log_env)
            parts_bap.append(to.bap)
            parts_voi.append(to.voiced_ratio)
            dur_ms += to.dur_ms
            modes.append("onset")
        else:
            o_ipas = PINYIN_ONSET_TO_IPA.get(onset, ())
            for oi in o_ipas:
                d = lookup_ipa_template(bank, oi, is_onset=True)
                parts_env.append(d["log_env"])
                parts_bap.append(d["bap"])
                parts_voi.append(d["voiced_ratio"])
                dur_ms += d["dur_ms"]
                modes.append(d["source_mode"])

    if final in bank.finals:
        tf = bank.finals[final]
        parts_env.append(tf.log_env)
        parts_bap.append(tf.bap)
        parts_voi.append(tf.voiced_ratio)
        dur_ms += tf.dur_ms
        modes.append("final")
    else:
        f_ipas = PINYIN_FINAL_TO_IPA.get(final, ("ə",))
        for fi in f_ipas:
            d = lookup_ipa_template(bank, fi, is_onset=False)
            parts_env.append(d["log_env"])
            parts_bap.append(d["bap"])
            parts_voi.append(d["voiced_ratio"])
            dur_ms += d["dur_ms"]
            modes.append(d["source_mode"])

    source_mode = "concat_onset_final" if set(modes) <= {"onset", "final"} else "warped_ipa"
    return _template_to_dict(
        np.concatenate(parts_env, axis=0),
        np.concatenate(parts_bap, axis=0),
        np.concatenate(parts_voi, axis=0),
        dur_ms,
        source_mode,
    )
