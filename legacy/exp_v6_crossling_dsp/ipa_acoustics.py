"""Unified ZH + EN IPA acoustic inventory (poles, anti-resonance zeros, fricative shapes, loci).

Covers all 69 IPA symbols emitted by ``exp_v5_arch.crossling_frontend``:
- 39 Mandarin IPA phones
- 48 English IPA phones (including diphthongs & syllabic consonants)

Formants are anchored on a neutral female-leaning vocal tract (vtln_scale=1.0),
compatible with AISHELL-3 female targets and scalable downward (~0.84) for male voices.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhoneSpec:
    ipa: str
    manner: str            # vowel, diphthong, glide, nasal, liquid, fricative, affricate, stop
    voiced: bool
    formants: tuple[float, float, float]       # (F1, F2, F3) Hz
    bandwidths: tuple[float, float, float]     # (BW1, BW2, BW3) Hz
    base_dur_ms: float
    locus_f2_hz: float | None = None
    locus_weight: float = 0.0
    anti_zero: tuple[float, float, float] | None = None  # (freq_hz, bw_hz, depth_db)
    fric_poles: tuple[tuple[float, float, float], ...] = ()  # ((freq_hz, bw_hz, gain_db), ...)
    voicing_bar: float = 0.0                   # voiced low-band fraction for voiced obstruents
    closure_ms: float = 0.0
    vot_ms: float = 0.0
    end_formants: tuple[float, float, float] | None = None  # for diphthongs


def _v(ipa: str, f1: float, f2: float, f3: float, dur: float = 110.0,
       bw: tuple[float, float, float] = (95.0, 145.0, 220.0),
       anti: tuple[float, float, float] | None = None) -> PhoneSpec:
    return PhoneSpec(
        ipa=ipa, manner="vowel", voiced=True,
        formants=(f1, f2, f3), bandwidths=bw, base_dur_ms=dur, anti_zero=anti,
    )


def _diph(ipa: str, f_start: tuple[float, float, float], f_end: tuple[float, float, float],
          dur: float = 155.0) -> PhoneSpec:
    return PhoneSpec(
        ipa=ipa, manner="diphthong", voiced=True,
        formants=f_start, end_formants=f_end,
        bandwidths=(100.0, 155.0, 230.0), base_dur_ms=dur,
    )


def _gl(ipa: str, f1: float, f2: float, f3: float, locus_f2: float, dur: float = 55.0) -> PhoneSpec:
    return PhoneSpec(
        ipa=ipa, manner="glide", voiced=True,
        formants=(f1, f2, f3), bandwidths=(90.0, 140.0, 210.0),
        base_dur_ms=dur, locus_f2_hz=locus_f2, locus_weight=0.55,
    )


def _nas(ipa: str, f1: float, f2: float, f3: float, zero_hz: float,
         locus_f2: float, dur: float = 68.0) -> PhoneSpec:
    return PhoneSpec(
        ipa=ipa, manner="nasal", voiced=True,
        formants=(f1, f2, f3), bandwidths=(120.0, 190.0, 260.0),
        base_dur_ms=dur, locus_f2_hz=locus_f2, locus_weight=0.50,
        anti_zero=(zero_hz, 240.0, 12.0),
    )


def _liq(ipa: str, f1: float, f2: float, f3: float, zero_hz: float,
         locus_f2: float, dur: float = 65.0) -> PhoneSpec:
    return PhoneSpec(
        ipa=ipa, manner="liquid", voiced=True,
        formants=(f1, f2, f3), bandwidths=(105.0, 165.0, 220.0),
        base_dur_ms=dur, locus_f2_hz=locus_f2, locus_weight=0.50,
        anti_zero=(zero_hz, 260.0, 9.5),
    )


def _fric(ipa: str, voiced: bool, poles: tuple[tuple[float, float, float], ...],
          locus_f2: float, f_place: tuple[float, float, float] = (420.0, 1750.0, 2850.0),
          dur: float = 92.0, vbar: float = 0.0) -> PhoneSpec:
    return PhoneSpec(
        ipa=ipa, manner="fricative", voiced=voiced,
        formants=f_place, bandwidths=(130.0, 190.0, 260.0),
        base_dur_ms=dur, locus_f2_hz=locus_f2, locus_weight=0.48,
        fric_poles=poles, voicing_bar=vbar,
    )


def _aff(ipa: str, voiced: bool, poles: tuple[tuple[float, float, float], ...],
         locus_f2: float, closure_ms: float = 38.0, vot_ms: float = 62.0,
         vbar: float = 0.0) -> PhoneSpec:
    return PhoneSpec(
        ipa=ipa, manner="affricate", voiced=voiced,
        formants=(380.0, locus_f2, 2850.0), bandwidths=(130.0, 190.0, 260.0),
        base_dur_ms=closure_ms + vot_ms, locus_f2_hz=locus_f2, locus_weight=0.55,
        fric_poles=poles, voicing_bar=vbar, closure_ms=closure_ms, vot_ms=vot_ms,
    )


def _stop(ipa: str, voiced: bool, poles: tuple[tuple[float, float, float], ...],
          locus_f2: float, closure_ms: float, vot_ms: float, vbar: float = 0.0) -> PhoneSpec:
    return PhoneSpec(
        ipa=ipa, manner="stop", voiced=voiced,
        formants=(360.0, locus_f2, 2750.0), bandwidths=(130.0, 190.0, 260.0),
        base_dur_ms=closure_ms + vot_ms, locus_f2_hz=locus_f2, locus_weight=0.58,
        fric_poles=poles, voicing_bar=vbar, closure_ms=closure_ms, vot_ms=vot_ms,
    )


IPA_SPECS: dict[str, PhoneSpec] = {
    # --- Monophthongs (ZH + EN) ---
    "a": _v("a", 890.0, 1540.0, 2850.0, dur=115.0),
    "ɑ": _v("ɑ", 840.0, 1260.0, 2750.0, dur=120.0),
    "æ": _v("æ", 760.0, 1880.0, 2850.0, dur=118.0),
    "ʌ": _v("ʌ", 680.0, 1380.0, 2750.0, dur=95.0),
    "ɔ": _v("ɔ", 610.0, 1020.0, 2700.0, dur=118.0),
    "o": _v("o", 510.0, 1040.0, 2750.0, dur=108.0),
    "ɤ": _v("ɤ", 540.0, 1520.0, 2900.0, dur=108.0),
    "e": _v("e", 460.0, 2120.0, 2950.0, dur=105.0),
    "ɛ": _v("ɛ", 595.0, 1950.0, 2880.0, dur=105.0),
    "ə": _v("ə", 515.0, 1540.0, 2820.0, dur=72.0),
    "ɨ": _v("ɨ", 410.0, 1780.0, 2880.0, dur=75.0),
    "ɪ": _v("ɪ", 425.0, 2140.0, 2980.0, dur=88.0),
    "i": _v("i", 340.0, 2480.0, 3220.0, dur=112.0),
    "y": _v("y", 365.0, 2240.0, 2820.0, dur=110.0),
    "u": _v("u", 390.0, 1010.0, 2650.0, dur=112.0),
    "ʉ": _v("ʉ", 390.0, 1480.0, 2680.0, dur=105.0),
    "ʊ": _v("ʊ", 465.0, 1180.0, 2680.0, dur=92.0),
    "ɚ": _v("ɚ", 500.0, 1480.0, 1980.0, dur=98.0, anti=(2450.0, 260.0, 8.0)),
    "ɝ": _v("ɝ", 515.0, 1460.0, 1940.0, dur=122.0, anti=(2450.0, 260.0, 8.5)),
    "ɻ̩": _v("ɻ̩", 420.0, 1620.0, 2250.0, dur=108.0, anti=(2550.0, 260.0, 6.5)),
    "ɹ̩": _v("ɹ̩", 385.0, 1690.0, 2720.0, dur=105.0),

    # --- English Diphthongs ---
    "aɪ": _diph("aɪ", (830.0, 1420.0, 2800.0), (380.0, 2320.0, 3100.0), dur=160.0),
    "aʊ": _diph("aʊ", (830.0, 1380.0, 2780.0), (430.0, 1020.0, 2650.0), dur=158.0),
    "eɪ": _diph("eɪ", (520.0, 2060.0, 2940.0), (365.0, 2380.0, 3120.0), dur=148.0),
    "oʊ": _diph("oʊ", (510.0, 1120.0, 2720.0), (395.0, 960.0, 2640.0), dur=148.0),
    "ɔɪ": _diph("ɔɪ", (590.0, 1010.0, 2700.0), (385.0, 2260.0, 3050.0), dur=162.0),

    # --- Glides ---
    "j": _gl("j", 345.0, 2450.0, 3180.0, locus_f2=2420.0, dur=55.0),
    "w": _gl("w", 375.0, 880.0, 2600.0, locus_f2=860.0, dur=56.0),
    "ɥ": _gl("ɥ", 360.0, 2180.0, 2800.0, locus_f2=2150.0, dur=56.0),

    # --- Nasals (with explicit anti-resonance zeros) ---
    "m": _nas("m", 320.0, 1180.0, 2550.0, zero_hz=1050.0, locus_f2=860.0, dur=66.0),
    "m̩": _nas("m̩", 325.0, 1180.0, 2550.0, zero_hz=1050.0, locus_f2=860.0, dur=92.0),
    "n": _nas("n", 330.0, 1650.0, 2720.0, zero_hz=1650.0, locus_f2=1780.0, dur=64.0),
    "n̩": _nas("n̩", 335.0, 1650.0, 2720.0, zero_hz=1650.0, locus_f2=1780.0, dur=90.0),
    "ŋ": _nas("ŋ", 340.0, 1420.0, 2650.0, zero_hz=2350.0, locus_f2=1420.0, dur=74.0),

    # --- Liquids & Taps (with lateral/rhotic zeros) ---
    "l": _liq("l", 390.0, 1480.0, 2780.0, zero_hz=2100.0, locus_f2=1680.0, dur=62.0),
    "l̩": _liq("l̩", 410.0, 1340.0, 2760.0, zero_hz=2050.0, locus_f2=1450.0, dur=95.0),
    "ɹ": _liq("ɹ", 440.0, 1340.0, 1860.0, zero_hz=2350.0, locus_f2=1380.0, dur=64.0),
    "ɻ": _liq("ɻ", 435.0, 1420.0, 1980.0, zero_hz=2400.0, locus_f2=1460.0, dur=66.0),
    "ɾ": _liq("ɾ", 390.0, 1740.0, 2750.0, zero_hz=2100.0, locus_f2=1780.0, dur=32.0),

    # --- Fricatives (multi-band spectral poles + voicing bars) ---
    "f": _fric("f", False, ((1600.0, 900.0, 7.0), (4400.0, 1600.0, 9.0)), locus_f2=950.0, dur=88.0),
    "v": _fric("v", True, ((1500.0, 850.0, 5.5), (4200.0, 1500.0, 7.0)), locus_f2=950.0, dur=72.0, vbar=0.48),
    "θ": _fric("θ", False, ((1900.0, 1100.0, 6.0), (5400.0, 1800.0, 7.5)), locus_f2=1680.0, dur=88.0),
    "ð": _fric("ð", True, ((1800.0, 1000.0, 4.5), (5000.0, 1600.0, 6.0)), locus_f2=1680.0, dur=62.0, vbar=0.50),
    "s": _fric("s", False, ((4800.0, 1100.0, 12.0), (7400.0, 1500.0, 15.0)), locus_f2=1780.0, dur=102.0),
    "z": _fric("z", True, ((4500.0, 1100.0, 9.5), (7000.0, 1500.0, 12.0)), locus_f2=1780.0, dur=82.0, vbar=0.45),
    "ʃ": _fric("ʃ", False, ((2650.0, 750.0, 14.0), (4900.0, 1300.0, 10.5)), locus_f2=2020.0, dur=104.0),
    "ʒ": _fric("ʒ", True, ((2550.0, 750.0, 11.5), (4700.0, 1300.0, 8.5)), locus_f2=2020.0, dur=82.0, vbar=0.45),
    "ʂ": _fric("ʂ", False, ((2250.0, 720.0, 13.5), (4200.0, 1250.0, 9.5)), locus_f2=1680.0, dur=96.0),
    "ɕ": _fric("ɕ", False, ((3650.0, 850.0, 13.5), (6200.0, 1400.0, 11.0)), locus_f2=2280.0, dur=96.0),
    "x": _fric("x", False, ((1350.0, 550.0, 11.5), (3200.0, 1100.0, 8.0)), locus_f2=1350.0, dur=86.0),
    "h": _fric("h", False, ((1100.0, 650.0, 8.0), (2600.0, 1200.0, 6.5)), locus_f2=1500.0, dur=74.0),
    "ʍ": _fric("ʍ", False, ((850.0, 500.0, 9.0), (2200.0, 1000.0, 6.0)), locus_f2=860.0, dur=78.0),

    # --- Affricates ---
    "ts": _aff("ts", False, ((4600.0, 1100.0, 12.0), (7200.0, 1400.0, 14.0)), locus_f2=1760.0, closure_ms=34.0, vot_ms=42.0),
    "tsʰ": _aff("tsʰ", False, ((4700.0, 1100.0, 12.5), (7300.0, 1400.0, 14.5)), locus_f2=1760.0, closure_ms=32.0, vot_ms=72.0),
    "tɕ": _aff("tɕ", False, ((3600.0, 850.0, 13.0), (6100.0, 1350.0, 11.0)), locus_f2=2280.0, closure_ms=34.0, vot_ms=44.0),
    "tɕʰ": _aff("tɕʰ", False, ((3650.0, 850.0, 13.5), (6200.0, 1350.0, 11.5)), locus_f2=2280.0, closure_ms=32.0, vot_ms=74.0),
    "ʈʂ": _aff("ʈʂ", False, ((2250.0, 720.0, 13.0), (4100.0, 1200.0, 9.5)), locus_f2=1660.0, closure_ms=36.0, vot_ms=46.0),
    "ʈʂʰ": _aff("ʈʂʰ", False, ((2300.0, 720.0, 13.5), (4200.0, 1200.0, 10.0)), locus_f2=1660.0, closure_ms=34.0, vot_ms=76.0),
    "tʃ": _aff("tʃ", False, ((2700.0, 750.0, 13.5), (4900.0, 1300.0, 10.5)), locus_f2=2020.0, closure_ms=36.0, vot_ms=54.0),
    "dʒ": _aff("dʒ", True, ((2550.0, 750.0, 11.0), (4700.0, 1300.0, 8.5)), locus_f2=2020.0, closure_ms=32.0, vot_ms=42.0, vbar=0.45),

    # --- Stops (closure + burst + VOT) ---
    "p": _stop("p", False, ((900.0, 600.0, 8.5), (2400.0, 1200.0, 6.0)), locus_f2=860.0, closure_ms=38.0, vot_ms=26.0),
    "pʰ": _stop("pʰ", False, ((950.0, 600.0, 9.0), (2500.0, 1200.0, 7.5)), locus_f2=860.0, closure_ms=34.0, vot_ms=56.0),
    "b": _stop("b", True, ((850.0, 550.0, 7.0), (2200.0, 1100.0, 5.0)), locus_f2=860.0, closure_ms=34.0, vot_ms=14.0, vbar=0.55),
    "t": _stop("t", False, ((3400.0, 1100.0, 11.0), (6200.0, 1600.0, 10.5)), locus_f2=1800.0, closure_ms=34.0, vot_ms=24.0),
    "tʰ": _stop("tʰ", False, ((3500.0, 1100.0, 11.5), (6300.0, 1600.0, 11.0)), locus_f2=1800.0, closure_ms=32.0, vot_ms=56.0),
    "d": _stop("d", True, ((3200.0, 1050.0, 9.0), (5800.0, 1500.0, 8.0)), locus_f2=1800.0, closure_ms=30.0, vot_ms=14.0, vbar=0.55),
    "k": _stop("k", False, ((1850.0, 750.0, 11.0), (3700.0, 1300.0, 7.5)), locus_f2=1550.0, closure_ms=36.0, vot_ms=28.0),
    "kʰ": _stop("kʰ", False, ((1900.0, 750.0, 11.5), (3800.0, 1300.0, 8.5)), locus_f2=1550.0, closure_ms=34.0, vot_ms=60.0),
    "ɡ": _stop("ɡ", True, ((1750.0, 700.0, 9.0), (3500.0, 1250.0, 6.0)), locus_f2=1550.0, closure_ms=32.0, vot_ms=16.0, vbar=0.55),
}


def get_phone_spec(ipa: str) -> PhoneSpec:
    """Return the :class:`PhoneSpec` for an IPA symbol, raising cleanly if unknown."""
    if ipa not in IPA_SPECS:
        raise ValueError(f"Unsupported IPA symbol in acoustic inventory: {ipa!r}")
    return IPA_SPECS[ipa]
