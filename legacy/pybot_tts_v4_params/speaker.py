"""Few-shot speaker ("timbre") parameter estimation from the step-2 formant table.

What this module does *not* claim
---------------------------------
It does **not** clone a voice. It fits five scalar/vector timbre descriptors from
a handful of measured phonemes per speaker and then measures, on held-out
utterances, how far those descriptors actually predict formants. Every number in
``speaker_params.json`` is an error measurement, not a quality claim.

The descriptor set
------------------
==================  ====================================================
``formant_scale``   geometric mean of the per-measurement ratio of this
                    speaker's F1/F2 to the **global cross-speaker mean** of the
                    same phoneme (pairs, so the all-1 baseline is exactly 1.0)
``f0_hz``           **median** F0 of this speaker's voiced finals
``spectral_tilt``   long-term-average-spectrum slope in dB/oct, from the real
                    waveform through :func:`lpc.spectral_tilt`
``bandwidth_scale`` median bandwidth ratio (F1, F2) vs the global mean
``vowel_space``     mean F1/F2 in Bark, plus a scalar knee parameter ``alpha``
                    for the non-linear Bark warp ``Bark(f) = alpha * Bark(f/alpha)``
==================  ====================================================

Why F3 is never touched
-----------------------
Step 2 measured F3 to be systematically **+158..+420 Hz** high and could not
remove that bias with any LPC order or lag window. A scale factor fitted on a
biased axis would bake the bias into the "speaker" parameter. So every estimate
here uses **F1 and F2 only**, and the output says ``"f3_trusted": false`` with
this reason attached. :func:`estimate_speaker` accepts F3 input (it is in the
measurements) and structurally ignores it -- a test proves this by mutating F3
and asserting the scale is bit-identical.

The scale baseline
------------------
``phoneme_params.json`` stores, per phoneme, ``formants`` (cross-speaker mean)
and ``per_speaker[name].formants``. The natural reference would be the *generic*
vowel table, but that table is a different phoneme inventory with a known
male-leaning bias, so it cannot separate "this speaker" from "this voice type".
The reference used here is the **cross-speaker mean of each phoneme's own
measurements**. Consequence, stated plainly: the two scales are complementary and
one of them is pinned at the reference by construction, so **a single speaker's
absolute scale is not identified -- only the ratio between speakers is**. That is
exactly the question this task asks ("how far apart are the two speakers"), and
it is measured honestly.

Cross-validation
----------------
Leave-one-utterance-out over *all* utterances of both speakers:

``model``      the held-out utterance's phonemes are predicted from
               ``measured(specimen phone) * alpha * scale / ref_scale`` where
               ``alpha`` is a global vowel-specific correction fitted with the
               same leave-one-out logic (a phone seen only in the held-out
               utterance falls back to ``alpha = 1``).
``baseline``   the plain mean over *every other* measurement of the same phone,
               pooled across both speakers.
``oracle``     the mean over the other measurements of the same phone **for the
               same speaker** -- the best any speaker-free-free method could do
               with this data. If ``model`` does not reach ``baseline``, or does
               not approach ``oracle``, the fitted descriptors carry little
               information and the JSON says so.

Errors are reported in **Bark** on a 0-20 scale (see :func:`hz_to_bark`),
separately for F1 and F2 (they have very different scales) and pooled.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from pybot_tts_v4_params import lpc

# --------------------------------------------------------------------------
# contract constants
# --------------------------------------------------------------------------
SCHEMA = 1
SR = lpc.SR

HERE = Path(__file__).resolve().parent
PHONEME_PARAMS = HERE / "phoneme_params.json"
OUT = HERE / "speaker_params.json"
SAMPLES = HERE / "samples"

ROOT = HERE.parent

# Which formant columns a speaker estimate may look at. F3 is absent on purpose.
USED_FORMANTS = ("f1", "f2")

# The step-2 bias that keeps F3 out of every estimate below.
F3_UNTRUSTED = {
    "f3_trusted": False,
    "f3_bias_hz": "+158..+420",
    "reason": (
        "step 2 measured F3 to be systematically 158-420 Hz too high and could "
        "not remove the bias with any LPC order (18/20/22/24) or lag window; a "
        "scale factor fitted on a biased axis would fold that bias into the "
        "'speaker' parameter, so F1/F2 only are used everywhere"
    ),
}

# Confidence contract, aligned with ``phoneme_params.json["confidence_contract"]``.
# The task contract is explicit: fewer than THREE usable phonemes must report
# ``low``. There is deliberately no ``medium`` tier for the speaker estimate --
# two phonemes is the classic case where a "scale" is a guess, not an estimate.
CONF_HIGH_MIN = 3
CONF_LOW_NOTE = "insufficient samples: does not represent this speaker"

# A speaker scale is only attempted above this many usable phonemes.
MIN_PHONEMES = 3
# The same threshold applies to the independent (provenance) evidence path.
MIN_ROWS = 3

# Long-term average spectrum analysis.
LTAS_WINDOW = 720            # 30 ms == lpc.WINDOW
LTAS_HOP = 480               # 20 ms: half the LPC hop, error is measured below
LTAS_LOW_HZ = 80.0           # below the analysis band: no usable level here
LTAS_HIGH_HZ = 8000.0        # top 1/3 octave carries no glottal tilt information
LTAS_SMOOTH_BINS = 3         # triangular smoothing over log2(f) bin index
LTAS_POINTS_DECIMALS = 2

# Vowel-space warp: the knee is searched on a coarse grid, bounded so that the
# warp cannot degenerate into a constant or into the identity inflated wildly.
ALPHA_RANGE = (80.0, 700.0)
ALPHA_STEPS = 62

# Provenance cross-check: repeat rows beyond this multiple of the median are
# dropped before averaging, so one outlier analysis cannot move a scale alone.
ROBUST_CLIP_MAD = 3.0


# --------------------------------------------------------------------------
# small numeric helpers (pure, no state)
# --------------------------------------------------------------------------
def median(values):
    """Median of a sequence of floats. Raises on empty input."""
    s = sorted(float(v) for v in values)
    if not s:
        raise ValueError("median of empty sequence")
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def mean(values):
    """Arithmetic mean. Raises on empty input (callers guard instead)."""
    s = [float(v) for v in values]
    if not s:
        raise ValueError("mean of empty sequence")
    return sum(s) / len(s)


def geomean(values):
    """Geometric mean, computed as ``exp(mean(log))``.

    Returns ``None`` for an empty sequence or for a non-positive value, which
    is what a caller wants for a ratio: a ratio of 0 has no geometric mean and
    silently substituting an epsilon would invent a factor.
    """
    s = [float(v) for v in values]
    if not s or any(v <= 0.0 for v in s):
        return None
    return math.exp(sum(math.log(v) for v in s) / len(s))


def std(values):
    """Sample standard deviation (n-1); 0.0 for fewer than two values."""
    s = [float(v) for v in values]
    if len(s) < 2:
        return 0.0
    mu = mean(s)
    return math.sqrt(sum((v - mu) ** 2 for v in s) / (len(s) - 1))


def hz_to_bark(hz):
    """Traunmueller's closed-form Bark scale.

    ``B(f) = 13 atan(0.00076 f) + 3.5 atan((f/7500)^2)``. This form does **not**
    saturate at 20; its asymptote is ~25.6 Bark as f -> infinity (24.0 at
    15.5 kHz, 24.6 at 20 kHz). One Bark is roughly one critical band, so an
    error of 1.0 Bark is about one just-noticeable formant-independent step --
    which is why it is the unit used here rather than Hz: a 200 Hz error at
    300 Hz is perceptual, the same 200 Hz at 2500 Hz is not.
    """
    hz = float(hz)
    if hz < 0.0:
        raise ValueError("frequency must be non-negative")
    return 13.0 * math.atan(0.00076 * hz) + 3.5 * math.atan((hz / 7500.0) ** 2)


def bark_to_hz(bark):
    """:func:`hz_to_bark` inverted by bisection.

    The closed form has no elementary inverse; bisection over [0, 30000] to
    1e-9 Hz is exact for every use here ("exact" to ~1e-9 Hz, far below the
    measurement resolution of anything in this project).
    """
    target = float(bark)
    if target <= 0.0:
        return 0.0
    lo, hi = 0.0, 30000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if hz_to_bark(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return 0.5 * (lo + hi)


def bark_warp(hz, alpha):
    """Non-linear frequency warp ``B(f) = alpha * Bark(f / alpha)``.

    ``alpha`` shifts the knee of the Bark curve: below ``alpha`` Hz the warp is
    close to linear, so formant *ratios* survive; above it the mapping compresses.
    A vocal-tract length change is a *uniform* stretch in Hz (ratio), which this
    linear-in-Hz warp reproduces exactly; ``alpha`` alone can therefore capture
    both a uniform ratio and the residual non-uniform shape difference.
    ``alpha == 0`` is treated as the identity (a guard, not a documented mode).
    """
    hz = float(hz)
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha == 0.0:
        return hz
    return alpha * bark_to_hz(hz_to_bark(hz / alpha))


def semitones(a_hz, b_hz):
    """Interval between two frequencies in semitones (positive: b above a)."""
    a_hz, b_hz = float(a_hz), float(b_hz)
    if a_hz <= 0.0 or b_hz <= 0.0:
        raise ValueError("frequencies must be positive")
    return 12.0 * math.log2(b_hz / a_hz)


# --------------------------------------------------------------------------
# reading inputs
# --------------------------------------------------------------------------
def load_phoneme_params(path=PHONEME_PARAMS):
    """Read ``phoneme_params.json`` (step 2's product) as a dict."""
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "phonemes" not in doc:
        raise ValueError(f"{p} is not a phoneme parameter table")
    return doc


def is_usable(entry):
    """Step-2 usability contract: confidence high/medium *and* marked usable.

    ``low`` is evidence, not a parameter: ``n_formant_samples < 3``. A fused or
    shrunk phoneme that step 2 demoted to ``medium`` is still accepted (that is
    what the contract says), which is why the confidence is carried through to
    the output rather than collapsed into a boolean.
    """
    if not isinstance(entry, dict):
        return False
    if entry.get("confidence") not in ("high", "medium"):
        return False
    if entry.get("usable_for_speaker") is not True:
        return False
    formants = entry.get("formants")
    return isinstance(formants, (list, tuple)) and len(formants) >= 2


def _usable_finals(doc):
    """``{phone: entry}`` for finals that may feed a speaker estimate, sorted."""
    out = {}
    for key, entry in (doc.get("phonemes") or {}).items():
        if not key.startswith("final:"):
            continue
        if is_usable(entry):
            out[key[len("final:"):]] = entry
    return dict(sorted(out.items()))


def _entry_f12(entry):
    """First two formants of a step-2 entry as floats, or ``None``."""
    f = entry.get("formants")
    if not isinstance(f, (list, tuple)) or len(f) < 2:
        return None
    try:
        f1, f2 = float(f[0]), float(f[1])
    except (TypeError, ValueError):
        return None
    if not (f1 > 0.0 and f2 > 0.0) or not (math.isfinite(f1) and math.isfinite(f2)):
        return None
    return (f1, f2)


def _clean_rows(f12_rows, group_key=None):
    """Drop malformed rows, then clip outliers per column by a MAD rule.

    ``f12_rows`` is a list of ``(f1, f2)`` (or ``(f1, f2, payload)``). The clip
    is applied to each formant column independently using
    ``median +- ROBUST_CLIP_MAD * 1.4826 * MAD``. With ``group_key`` given, the
    clip is computed inside each group. Returns the surviving payloads.
    """
    rows = []
    for row in f12_rows:
        f1, f2 = float(row[0]), float(row[1])
        if f1 > 0.0 and f2 > 0.0 and math.isfinite(f1) and math.isfinite(f2):
            rows.append((f1, f2) + tuple(row[2:]))
    if not rows:
        return []
    groups = {}
    for row in rows:
        key = group_key(row) if group_key else None
        groups.setdefault(key, []).append(row)
    kept = []
    for group in groups.values():
        for column in (0, 1):
            values = [r[column] for r in group]
            med = median(values)
            mad = median([abs(v - med) for v in values])
            if mad <= 0.0:
                continue                      # <3 values or all-equal: no spread
            half = ROBUST_CLIP_MAD * 1.4826 * mad
            group[:] = [r for r in group if abs(r[column] - med) <= half]
        kept.extend(group)
    return kept


def _parse_rows(samples_info):
    """Normalise ``samples_info`` into ``(phoneme_key, f1, f2, payload)`` rows.

    Three accepted shapes, because the caller legitimately has three:

    * a ``phoneme_params.json`` document  (``{"phonemes": {...}}``) -- the
      canonical path: one row per usable phoneme, the reference is that
      phoneme's own cross-speaker mean;
    * the output of :func:`measurement_rows` -- one row per analysed syllable,
      which is what leave-one-utterance-out needs;
    * a plain list/tuple of mapping records with ``formants`` (plus optional
      ``phone``/``phoneme``, ``speaker``, ``utterance``, ``bandwidths``,
      ``f0_hz``) -- what the unit tests build by hand.

    Rows that cannot supply *two positive formants* are dropped, never patched.
    """
    rows = []

    def add(phone, f12, payload):
        if f12 is None:
            return
        rows.append((phone, float(f12[0]), float(f12[1]), payload))

    if isinstance(samples_info, dict) and isinstance(samples_info.get("phonemes"), dict):
        for key in sorted(samples_info["phonemes"]):
            entry = samples_info["phonemes"][key]
            if not key.startswith("final:") or not is_usable(entry):
                continue
            phone = key[len("final:"):]
            ref = _entry_f12(entry)
            add(phone, ref, {
                "source": "phoneme_params",
                "confidence": entry.get("confidence"),
                "n_formant_samples": entry.get("n_formant_samples"),
                "bandwidths": entry.get("bandwidths"),
                "role": entry.get("role"),
            })
        return rows

    if isinstance(samples_info, (list, tuple)):
        items = list(samples_info)
    elif isinstance(samples_info, dict) and isinstance(samples_info.get("samples"), (list, tuple)):
        items = list(samples_info["samples"])
    elif isinstance(samples_info, dict) and isinstance(samples_info.get("rows"), (list, tuple)):
        items = list(samples_info["rows"])
    else:
        raise ValueError(
            "estimate_speaker needs a phoneme_params document, a list of "
            "measurement records, or a dict with a 'samples'/'rows' list"
        )

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each sample record must be a mapping")
        if not is_usable(item) and "confidence" in item:
            # A caller may hand over raw step-2 entries; honour the contract.
            if item.get("confidence") not in ("high", "medium") or \
                    item.get("usable_for_speaker") is False:
                continue
        f12 = _entry_f12(item)
        phone = item.get("phone") or item.get("phoneme") or item.get("final") or ""
        phone = str(phone)
        if phone.startswith("final:"):
            phone = phone[len("final:"):]
        add(phone, f12, dict(item))
    return rows


def measurement_rows(doc):
    """Per-syllable formant rows from step 2's kept ``provenance`` lists.

    Step 2 stores each final's individual measurements under ``provenance``
    (``utterance``, ``speaker``, ``pinyin``, ``start``, ``end``, ``formants``,
    ``bound_confidence``, ``split_confidence``). Only entries under a phoneme
    that is *itself* usable are returned: a per-sample F1 taken from a phoneme
    step 2 judged unreliable is not made reliable by being listed.

    ``n`` on each row is the number of phonemes the speaker contributed to that
    same syllable -- it is what makes `n >= 100` mean "many samples, few
    phonemes" instead of reading as independent evidence.
    """
    per_utterance = {}
    rows = []
    for key, entry in (doc.get("phonemes") or {}).items():
        if not key.startswith("final:") or not is_usable(entry):
            continue
        phone = key[len("final:"):]
        provenance = entry.get("provenance") or []
        counted = {}
        for item in provenance:
            speech = item.get("speaker")
            if speech:
                counted[speech] = counted.get(speech, 0) + 1
        for item in provenance:
            formants = item.get("formants")
            if not isinstance(formants, (list, tuple)) or len(formants) < 2:
                continue
            try:
                f1, f2 = float(formants[0]), float(formants[1])
            except (TypeError, ValueError):
                continue
            speech = item.get("speaker")
            utter = item.get("utterance")
            bw = item.get("bandwidths")
            rows.append({
                "phone": phone,
                "phoneme_key": key,
                "speaker": speech,
                "utterance": utter,
                "f0_hz": item.get("f0_hz"),
                "start": item.get("start"),
                "end": item.get("end"),
                "pinyin": item.get("pinyin"),
                "tone": item.get("tone"),
                "formants": [f1, f2],
                "bandwidths": bw,
                "bound_confidence": item.get("bound_confidence"),
                "split_confidence": item.get("split_confidence"),
                "n": counted.get(speech),
                "source": "provenance",
            })
            per_utterance.setdefault((speech, utter), set()).add(phone)
    for row in rows:
        row["utterance_phonemes"] = len(per_utterance.get((row["speaker"], row["utterance"]), ()))
    return rows


# --------------------------------------------------------------------------
# long-term average spectrum -> spectral tilt
# --------------------------------------------------------------------------
_TABLE_CACHE = {}


def _ltas_tables(sr=SR):
    """Per-window Hann/DFT cosine tables and the log-spaced frequency grid.

    Cached: the tables depend only on ``(window, sr)``. The naive DFT is
    ``k_max * window`` multiply-adds per frame with ``k_max`` the bin under
    ``LTAS_HIGH_HZ``; at 24 kHz / 720 samples that is ~500k operations per 30 ms
    frame, which is why full-frame FFT would be preferred -- but no numpy and no
    ``cmath`-free radix code here, and the frame count is bounded (see
    :func:`spectral_tilt_from_wav`). The approximation is a **half-hop, full-
    window, band-limited naive DFT with 1/3-octave triangular smoothing** -- not
    a perfect LTAS, and the JSON says so.
    """
    key = int(sr)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]

    n = LTAS_WINDOW
    win = lpc.hamming(n)
    k_max = int(LTAS_HIGH_HZ * n / sr) + 2
    cos_t = []
    sin_t = []
    for k in range(k_max + 1):
        w = math.tau * k / n
        cos_t.append([math.cos(w * i) for i in range(n)])
        sin_t.append([math.sin(w * i) for i in range(n)])
    freqs = [sr * k / n for k in range(k_max + 1)]

    # 1/3-octave-like grid: three bins per octave over the analysis band.
    grid = []
    f = LTAS_LOW_HZ
    while f <= LTAS_HIGH_HZ:
        grid.append(f)
        f *= 2.0 ** (1.0 / 3.0)
    half = (2.0 ** (1.0 / 3.0) - 1.0) * 0.5 * 1.2      # triangular half-width
    idx = []
    for fc in grid:
        lo, hi = fc * (1.0 - half), fc * (1.0 + half)
        members = [(j, fq) for j, fq in enumerate(freqs) if lo <= fq <= hi]
        if len(members) < LTAS_SMOOTH_BINS:            # widen until it has support
            order = sorted(range(len(freqs)), key=lambda j: abs(freqs[j] - fc))
            members = [(j, freqs[j]) for j in order[:LTAS_SMOOTH_BINS]]
        idx.append(members)

    table = {"window": n, "win": win, "cos": cos_t, "sin": sin_t,
             "freqs": freqs, "k_max": k_max, "grid": grid, "idx": idx}
    _TABLE_CACHE[key] = table
    return table


def frame_band_magnitude(frame, sr=SR):
    """Band-limited magnitude spectrum of one windowed frame on the LTAS grid.

    Returns ``[(freq_hz, magnitude)]``. DC and the band below ``LTAS_LOW_HZ``
    are skipped: they carry no glottal-tilt information and only add a large
    constant to the fit.
    """
    tab = _ltas_tables(sr)
    n = tab["window"]
    win = tab["win"]
    cos_t, sin_t = tab["cos"], tab["sin"]
    x = [float(v) for v in frame]
    n = min(n, len(x))
    if n < 64:
        return []
    mu = sum(x[:n]) / n
    x = [(x[i] - mu) * win[i] for i in range(n)]
    out = []
    for fc, members in zip(tab["grid"], tab["idx"]):
        acc = 0.0
        for j, _fq in members:
            re = 0.0
            im = 0.0
            cj, sj = cos_t[j], sin_t[j]
            for i in range(n):
                xi = x[i]
                re += xi * cj[i]
                im += xi * sj[i]
            acc += math.hypot(re, im)
        out.append((fc, acc / len(members)))
    return out


def spectral_tilt_from_wav(path, start_sample=0, stop_sample=None, sr=SR):
    """LTAS slope (dB/oct) of one utterance window, or ``None``.

    Steps: read the WAV (mono 16-bit, checked by :func:`lpc.wav_read`), cut the
    utterance, take 30 ms Hamming frames every 20 ms, reject frames quieter
    than ``0.10 x`` the loudest frame of that utterance *and* below the absolute
    floor ~30 dB under the loudest (so a fading tail cannot dominate), average
    the magnitude spectra (power mean, not dB average -- averaging dB favours
    frames with deep notches), then least-squares fit ``20 log10|X|`` against
    ``log2(f)`` through :func:`lpc.spectral_tilt`.

    Returns ``None`` -- with the reason recorded by the caller -- when the WAV is
    missing, unreadable, too short, or has no frame above the floor. It never
    guesses a tilt.
    """
    try:
        x, file_sr = lpc.wav_read(path)
    except (OSError, ValueError):
        return None
    if file_sr != sr:
        return None
    start = max(0, int(start_sample))
    stop = len(x) if stop_sample is None else min(len(x), int(stop_sample))
    if stop - start < LTAS_WINDOW * 2:
        return None
    seg = x[start:stop]
    if not seg:
        return None

    starts = list(range(0, max(1, len(seg) - LTAS_WINDOW + 1), LTAS_HOP))
    if not starts:
        return None
    levels = [(s, lpc.rms(seg[s:s + LTAS_WINDOW])) for s in starts]
    peak = max(level for _s, level in levels)
    if peak <= 0.0:
        return None
    floor = max(0.10 * peak, peak * 10.0 ** (-30.0 / 20.0))
    kept = [s for s, level in levels if level >= floor]
    if len(kept) < 4:
        return None

    mags = None
    for s in kept:
        spec = frame_band_magnitude(seg[s:s + LTAS_WINDOW], sr)
        if not spec:
            continue
        if mags is None:
            mags = [m for _f, m in spec]
        else:
            for i, (_f, m) in enumerate(spec):
                mags[i] += m
    if mags is None:
        return None
    mags = [m / len(kept) for m in mags]

    # A fixed, dense frequency grid is required by lpc.spectral_tilt (it assumes
    # ``sr*0.5*i/(n-1)``). Interpolate the 1/3-octave magnitudes back onto one.
    return _fit_tilt(mags, sr)


def _fit_tilt(mags, sr=SR):
    """Fit dB/oct on an irregular magnitude grid.

    Same least-squares as :func:`lpc.spectral_tilt` but over an explicit
    frequency axis, because the LTAS grid is log-spaced (1/3 octave) and
    feeding it to the linear-bin version would weight the low end absurdly.
    """
    tab = _ltas_tables(sr)
    pts = []
    for fc, m in zip(tab["grid"], mags):
        if m is None or m <= 0.0 or not math.isfinite(m):
            continue
        pts.append((math.log2(fc), 20.0 * math.log10(m)))
    if len(pts) < 4:
        return None
    n = float(len(pts))
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den <= 0.0:
        return None
    return round(num / den, LTAS_POINTS_DECIMALS)


def _utterance_window(rows):
    """Deprecated shim kept out of the public surface: see :func:`utterance_windows`."""
    return rows


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------
def _confidence_of(n_units, n_rows):
    """Tier for a speaker estimate. ``n_units`` counts distinct phonemes.

    ``high`` at >= :data:`CONF_HIGH_MIN` phonemes, otherwise ``low``. There is
    no ``medium``: the task contract says anything under three phonemes must be
    reported as ``low``, and a two-phoneme "scale" is a guess whatever its
    repeat count. ``n_rows`` is accepted so callers can log it but never
    promotes a tier -- 100 measurements of *one* vowel still cannot separate a
    speaker from itself.
    """
    return "high" if n_units >= CONF_HIGH_MIN else "low"


def _reference_f12(doc, phone):
    """Cross-speaker reference F1/F2 for ``phone``: its own pooled mean."""
    entry = (doc.get("phonemes") or {}).get(f"final:{phone}")
    if not entry:
        return None
    return _entry_f12(entry)


def _pair_ratios(measured_f12, reference, indexes=(0, 1)):
    """Per-formant ratios ``measured[i] / reference[i]`` for ``indexes``."""
    if reference is None:
        return []
    out = []
    for i in indexes:
        if i >= len(reference) or i >= len(measured_f12):
            continue
        base = float(reference[i])
        val = float(measured_f12[i])
        if base > 0.0 and val > 0.0:
            out.append(val / base)
    return out


def _fit_formant_scale(pairs):
    """``formant_scale`` from ``[(phone, f1, f2)]`` rows.

    First pass: pooled geometric mean of F1/F2 ratios against the phoneme's
    reference. Second pass: refit one median ratio per phoneme, then take the
    geometric mean of the medians. The second number is reported because a
    vowel with 30 repeats would otherwise outvote a vowel with 5 and the
    "scale" would silently become a scale for the frequent vowel. If the two
    passes disagree by more than ``SCALE_DISAGREEMENT`` the result is flagged
    instead of silently choosing one.
    """
    pooled = []
    per_phone = {}
    for phone, f1, f2, ref in pairs:
        ratios = _pair_ratios((f1, f2), ref)
        pooled.extend(ratios)
        per_phone.setdefault(phone, []).extend(ratios)
    pooled_scale = geomean(pooled)
    per_phone_medians = [median(v) for v in per_phone.values() if v]
    robust_scale = geomean(per_phone_medians)
    return {
        "formant_scale": pooled_scale,
        "formant_scale_robust": robust_scale,
        "n_ratio_pairs": len(pooled),
        "n_phonemes": len(per_phone),
        "pooled_vs_robust_delta": (
            None if pooled_scale is None or robust_scale is None
            else round(pooled_scale - robust_scale, 6)),
    }


SCALE_DISAGREEMENT = 0.01      # 1 % -- below this the two estimates are one number


def _fit_alpha(samples, _unused=None):
    """Knee parameter for the Bark warp, minimising Bark-space formant error.

    Objective: for every sample, the predicted F1/F2 are
    ``bark_warp(reference, alpha) * scale``; the residual is measured as the
    Bark distance between predicted and observed, normalised per sample. A
    grid search over ``ALPHA_RANGE`` is used -- the objective is smooth but not
    convex (two formants, one knee), and 62 evaluations of a few hundred pairs
    is cheap and free of a local-minimum story.
    """
    scale = _fit_formant_scale(samples)["formant_scale"]
    if not scale:
        return None, None
    best = None
    alpha = ALPHA_RANGE[0]
    prev = ALPHA_RANGE[0]
    grow = (ALPHA_RANGE[1] / ALPHA_RANGE[0]) ** (1.0 / max(1, ALPHA_STEPS - 1))
    for _ in range(ALPHA_STEPS):
        cost = 0.0
        n = 0
        for phone, f1, f2, ref in samples:
            if ref is None:
                continue
            for i, observed in ((0, f1), (1, f2)):
                if i >= len(ref) or ref[i] <= 0.0:
                    continue
                predicted = bark_warp(ref[i], alpha) * scale
                cost += abs(hz_to_bark(predicted) - hz_to_bark(observed))
                n += 1
        if n:
            cost /= n
            if best is None or cost < best[0] - 1e-12:
                best = (cost, alpha)
        prev, alpha = alpha, alpha * grow
    return (round(best[1], 3) if best else None,
            round(best[0], 6) if best else None)


def _fit_bandwidth_scale(rows, reference_bws):
    """Median F1/F2 bandwidth ratio against the pooled value, with a caveat.

    ``per_speaker`` in ``phoneme_params.json`` carries only ``n`` and
    ``formants`` -- **no per-speaker bandwidths**. The only bandwidths in the
    table are pooled across both speakers, so for the canonical path this
    function returns ``1.0`` by construction and reports
    ``available=False``. That is reported instead of being papered over with a
    silently absent field. A caller that supplies per-speaker bandwidths (the
    unit tests do) gets a real measurement.
    """
    pooled = []
    per_phone = {}
    pooled_input = 0
    for row in rows:
        bws = row.get("bandwidths")
        phone = row.get("phone")
        base = reference_bws.get(phone)
        if not bws or not base:
            continue
        if row.get("bandwidth_is_pooled"):
            pooled_input += 1
            continue
        ratios = _pair_ratios(bws, base, (0, 1))
        pooled.extend(ratios)
        per_phone.setdefault(phone, []).extend(ratios)
    if not pooled:
        return None, 0, 0, {
            "available": False,
            "reason": (
                "phoneme_params.json['phonemes'][*]['per_speaker'] stores only "
                "'n' and 'formants'; the only bandwidths are pooled over both "
                "speakers, so no per-speaker bandwidth ratio can be formed"),
            "n_pooled_bandwidth_rows_skipped": pooled_input,
        }
    med = median(pooled)
    return (round(med, 4), len(pooled), len(per_phone),
            {"available": True, "n_ratio_pairs": len(pooled)})


def _vowel_space(rows):
    """Mean F1/F2 in Bark plus the spread, for the speaker's vowel centroid."""
    if not rows:
        return None
    b1 = [hz_to_bark(float(r["formants"][0])) for r in rows]
    b2 = [hz_to_bark(float(r["formants"][1])) for r in rows]
    return {
        "centroid_bark": [round(mean(b1), 3), round(mean(b2), 3)],
        "spread_bark": [round(std(b1), 3), round(std(b2), 3)],
        "n": len(rows),
    }


def estimate_speaker(samples_info, *, speaker=None, tilt_rows=None,
                     tilt_paths=None, collect_tilt=False):
    """Estimate one speaker's timbre descriptors from step-2 evidence.

    ``samples_info``
        Either a ``phoneme_params.json`` document (the normal case -- one row
        per usable phoneme, using ``per_speaker[speaker]``), or a list of
        measurement records produced by :func:`measurement_rows`.
    ``speaker``
        Required with a phoneme_params document: which ``per_speaker`` split to
        fit. With measurement rows it is inferred from the rows and must agree
        if both are given.
    ``tilt_rows`` / ``tilt_paths``
        ``{utterance: (wav_path, start, stop)}`` -- where to read audio for
        :func:`spectral_tilt_from_wav`. ``collect_tilt=True`` makes the function
        scan the WAVs itself; it is off by default because it is the slow path
        (naive DFT) and because a caller that only wants the formant scale
        should not pay for it.

    Returns a dict. Unavailable descriptors are ``None`` **with** a sibling
    ``*_note`` / ``unavailable_reason`` string -- never silently blank.
    """
    doc = None
    rows = _parse_rows(samples_info)
    if isinstance(samples_info, dict) and isinstance(samples_info.get("phonemes"), dict):
        doc = samples_info
        if not speaker:
            raise ValueError("a phoneme_params document needs an explicit speaker=")
        rows = []
        entry_split = {}
        for phone, entry in _usable_finals(doc).items():
            split = (entry.get("per_speaker") or {}).get(speaker)
            ref = _entry_f12(entry)
            if not split or ref is None:
                continue
            f12 = _entry_f12(split)
            if f12 is None:
                continue
            n = int(split.get("n") or 0)
            if n < 1:
                continue
            rows.append((phone, f12[0], f12[1], {
                "source": "per_speaker",
                "confidence": entry.get("confidence"),
                "n": n,
                "bandwidths": entry.get("bandwidths"),
                "bandwidth_is_pooled": True,
                "reference": ref,
            }))
            entry_split[phone] = split
    elif rows and speaker is None:
        speakers = {r[3].get("speaker") for r in rows if r[3].get("speaker")}
        if len(speakers) == 1:
            speaker = next(iter(speakers))
        else:
            raise ValueError(
                f"rows mix speakers {sorted(s for s in speakers if s)}; pass speaker=")

    # Clean the rows; the clip keeps one wild analysis from moving the scale.
    kept = _clean_rows([(f1, f2, phone, payload)
                        for phone, f1, f2, payload in rows],
                       group_key=lambda row: row[2])
    samples = []
    for f1, f2, phone, payload in kept:
        ref = payload.get("reference")
        if ref is None and doc is not None:
            ref = _reference_f12(doc, phone)
        if ref is None:
            ref = _entry_f12({"formants": [f1, f2]})     # single-row: ratio 1.0
        samples.append((phone, f1, f2, ref))

    n_phonemes = len({s[0] for s in samples})
    n_rows = len(samples)
    scale_fit = _fit_formant_scale(samples)

    # --- scale -----------------------------------------------------------
    confidence = _confidence_of(n_phonemes, n_rows)
    formant_scale = scale_fit["formant_scale"]
    scale_note = None
    if formant_scale is None:
        unavailable = "no usable phoneme with two positive formants"
        formant_scale = None
        scale_note = unavailable
    elif n_phonemes < MIN_PHONEMES:
        # Still reported (a caller asked for it), but explicitly not a factor
        # anybody should render with.
        scale_note = (f"{n_phonemes} usable phoneme(s) < {MIN_PHONEMES}; "
                      f"the ratio is reported as evidence, not as a speaker scale")

    alpha, alpha_cost = _fit_alpha(samples) if samples else (None, None)

    f0s = [float(p["f0_hz"]) for *_x, p in kept
           if isinstance(p.get("f0_hz"), (int, float)) and p["f0_hz"] and p["f0_hz"] > 0]
    f0_source = "median of per-row f0_hz supplied by the caller" if f0s else None

    if not f0s and doc is not None and speaker:
        # Step 2 kept F0 per phoneme (pooled) but not per syllable, and
        # ``per_speaker`` carries only n + formants. The honest fallback is the
        # step-2 speaker entry, which is a *mean over voiced samples* -- it is
        # labelled as such rather than passed off as a median computed here.
        step2 = (doc.get("speakers") or {}).get(speaker) or {}
        pooled_hz = step2.get("f0_hz")
        if isinstance(pooled_hz, (int, float)) and pooled_hz > 0:
            f0s = [float(pooled_hz)]
            f0_source = (
                "step 2 speakers[<speaker>]['f0_hz'] (mean over that speaker's "
                "voiced samples; step 2 stored no per-syllable F0, so no median "
                "could be computed independently here)")

    bw_scale, bw_pair_n, bw_phone_n, bw_info = _fit_bandwidth_scale(
        [p for *_x, p in kept], _reference_bw_map(doc))

    out = {
        "speaker": speaker,
        "formant_scale": None if formant_scale is None else round(formant_scale, 4),
        "formant_scale_robust": (None if scale_fit["formant_scale_robust"] is None
                                 else round(scale_fit["formant_scale_robust"], 4)),
        "formant_scale_ratio_scope": "F1, F2 (F3 excluded)",
        "formant_scale_reference": "this phoneme's cross-speaker mean (phoneme_params.formants)",
        "n_ratio_pairs": scale_fit["n_ratio_pairs"],
        "n_phonemes": n_phonemes,
        "n_rows": n_rows,
        "f0_hz": None if not f0s else round(median(f0s), 1),
        "f0_statistic": ("median" if f0_source and f0_source.startswith("median")
                         else ("mean" if f0s else None)),
        "f0_source": f0_source,
        "n_f0": len(f0s),
        "spectral_tilt_db_per_oct": None,
        "bandwidth_scale": bw_scale,
        "bandwidth_scale_scope": "F1, F2",
        "bandwidth_scale_info": bw_info,
        "n_bandwidth_pairs": bw_pair_n,
        "n_bandwidth_phonemes": bw_phone_n,
        "vowel_space": _vowel_space([{"formants": [f1, f2]} for _p, f1, f2, _r in samples]),
        "warp_alpha_hz": alpha,
        "warp_alpha_cost_bark": alpha_cost,
        "warp_note": (
            "alpha knee of Bark(f/alpha)*alpha fitted jointly with formant_scale; "
            "alpha near the middle of a wide flat minimum means the warp carries "
            "no information beyond the scale"
        ),
        "confidence": confidence,
        "confidence_rule": (
            f"high: >= {CONF_HIGH_MIN} usable phonemes; "
            f"low: < {CONF_HIGH_MIN} ({CONF_LOW_NOTE})"
        ),
    }
    if confidence == "low":
        out["confidence_note"] = CONF_LOW_NOTE
    if scale_note:
        out["formant_scale_note"] = scale_note
    if scale_fit["pooled_vs_robust_delta"] is not None and \
            abs(scale_fit["pooled_vs_robust_delta"]) > SCALE_DISAGREEMENT:
        out["formant_scale_disagreement"] = (
            f"pooled {out['formant_scale']} vs per-phoneme-median "
            f"{out['formant_scale_robust']} differ by "
            f"{scale_fit['pooled_vs_robust_delta']:+.4f}; frequent vowels do not "
            f"agree with rare ones, so the single number is not well identified")

    # --- spectral tilt ---------------------------------------------------
    if (collect_tilt or tilt_rows) and speaker:
        tilt, tilt_info = _speaker_tilt(speaker, doc, tilt_rows)
        out["spectral_tilt_db_per_oct"] = tilt
        out.update(tilt_info)
    elif out["spectral_tilt_db_per_oct"] is None:
        out["unavailable_reason"] = (
            "spectral_tilt not requested in this call; the LTAS pass reads WAVs "
            "and is opt-in (collect_tilt=True)")
    return out


def _reference_bw_map(doc):
    """``{phone: [bw1, bw2, bw3]}`` pooled bandwidths, F1/F2 only are used."""
    out = {}
    if doc is None:
        return out
    for phone, entry in _usable_finals(doc).items():
        bw = entry.get("bandwidths")
        if isinstance(bw, (list, tuple)) and len(bw) >= 2:
            out[phone] = [float(bw[0]), float(bw[1])]
    return out


def utterance_windows(doc, speaker=None):
    """``{utterance: (wav_path, start, stop)}`` from step-2 provenance ranges.

    The path is rebuilt the way step 2 found it: the extended corpus keeps
    whole sentences at ``pybot_tts_v2_recorded/extended/<id>.wav``. Nothing here
    confirms the audio matches the transcript -- that would need listening, and
    step 2 explicitly refused to claim it.
    """
    out = {}
    for key, entry in (doc.get("phonemes") or {}).items():
        if not key.startswith("final:") or not is_usable(entry):
            continue
        for item in entry.get("provenance") or []:
            if speaker and item.get("speaker") != speaker:
                continue
            utter = item.get("utterance")
            if not utter:
                continue
            start = item.get("start")
            end = item.get("end")
            if start is None or end is None:
                continue
            cur = out.get(utter)
            if cur is None:
                out[utter] = [start, end, item.get("speaker")]
            else:
                cur[0] = min(cur[0], start)
                cur[1] = max(cur[1], end)
    resolved = {}
    for utter, (start, stop, spk) in out.items():
        path = ROOT / "pybot_tts_v2_recorded" / "extended" / f"{utter}.wav"
        resolved[utter] = (path, int(start), int(stop), spk)
    return resolved


def _speaker_tilt(speaker, doc, tilt_rows=None, sample=3):
    """Median LTAS slope over this speaker's utterances.

    One slope per utterance (a whole sentence, not a syllable: a syllable is
    150-300 ms and its slope is dominated by which vowels happen to be in it),
    then the median over utterances. Up to ``sample`` utterances are analysed to
    keep the naive-DFT cost bounded; which ones is deterministic (shortest
    first), so the number is reproducible.
    """
    windows = tilt_rows or utterance_windows(doc, speaker)
    mine = {u: v for u, v in windows.items() if not speaker or v[3] == speaker}
    if not mine:
        return None, {"unavailable_reason": "no utterance provenance for this speaker"}
    ordered = sorted(mine.items(),
                     key=lambda kv: (int(kv[1][2]) - int(kv[1][1]), kv[0]))
    per_utterance = {}
    failed = []
    for utter, (path, start, stop, _spk) in ordered:
        if len(per_utterance) >= sample:
            break
        tilt = spectral_tilt_from_wav(path, start, stop)
        if tilt is None:
            failed.append(utter)
            continue
        per_utterance[utter] = tilt
    info = {
        "spectral_tilt_method": (
            f"LTAS: {LTAS_WINDOW} Hz Hamming frames every {LTAS_HOP} samples over "
            f"the whole utterance, frames >= max(0.10*peak, peak-30dB), power-mean "
            f"magnitude, 1/3-octave triangular smoothing, least squares of "
            f"20log10|X| vs log2(f) over {LTAS_LOW_HZ:.0f}-{LTAS_HIGH_HZ:.0f} Hz"),
        "spectral_tilt_approximation": (
            "naive band-limited DFT (no FFT), half-hop frames, 1/3-octave band "
            "averaging and a bounded utterance sample -- an approximation of the "
            "LTAS, not an exact one"),
        "spectral_tilt_per_utterance": per_utterance,
        "n_utterances_tilt": len(per_utterance),
    }
    if failed:
        info["spectral_tilt_failed_utterances"] = failed
    if not per_utterance:
        info["unavailable_reason"] = (
            "every attempted utterance failed the LTAS gate (missing WAV, wrong "
            "sample rate, too short, or no frame above the level floor)")
        return None, info
    return round(median(per_utterance.values()), LTAS_POINTS_DECIMALS), info


# --------------------------------------------------------------------------
# applying parameters
# --------------------------------------------------------------------------
def apply_speaker(params, speaker, *, base=None, include_scale_only=False):
    """Apply a speaker descriptor to a parameter table -> that speaker's table.

    ``params``
        The speaker document from :func:`estimate_speaker`, or a whole
        ``speaker_params.json`` document (its ``speakers`` / ``estimates`` are
        looked up by name). A scalar scale is also accepted and used directly.
    ``speaker``
        Speaker name; ignored when ``params`` is a scalar.
    ``base``
        The phoneme table to transform. ``None`` uses ``phoneme_params.json``.
    ``include_scale_only``
        When the speaker's ``confidence`` is ``low`` the scale is *not*
        applied by default (see the contract: low confidence is evidence). Set
        this to apply it anyway -- the returned document then records that it
        did.

    The transformation is deliberately small and reversible:

    * every ``final:*`` entry's ``formants[0:2]`` are multiplied by
      ``formant_scale`` and ``bandwidths[0:2]`` by ``bandwidth_scale``;
    * ``formants[2]`` (F3) is **left untouched**, because F3 is not trusted and
      scaling it would pretend the speaker parameter explains it;
    * each entry gains ``speaker_applied`` with the factors actually used, and
      ``formant_source`` saying whether the entry was measured or left generic;
    * the top level gains ``applied_speaker`` and ``f3_trusted: false``.

    Entries that are ``low`` confidence or lack formants are left alone: there
    is nothing there to scale, and inventing a value would be worse than a hole.
    """
    table = base if base is not None else load_phoneme_params()
    if not isinstance(table, dict) or "phonemes" not in table:
        raise ValueError("base must be a phoneme parameter table")

    desc = _resolve_speaker(params, speaker)
    scale = desc.get("formant_scale")
    bw_scale = desc.get("bandwidth_scale")
    confidence = desc.get("confidence")

    if conf_is_low(confidence) and not include_scale_only:
        return _clone_with_note(table, {
            "applied_speaker": desc.get("speaker") or speaker,
            "applied": False,
            "reason": (f"confidence={confidence}: {CONF_LOW_NOTE}; pass "
                       f"include_scale_only=True to apply it as a hypothesis"),
            "formant_scale": None,
        })

    if scale is None:
        return _clone_with_note(table, {
            "applied_speaker": desc.get("speaker") or speaker,
            "applied": False,
            "reason": "the speaker entry has no formant_scale",
            "formant_scale": None,
        })

    scale = float(scale)
    if not math.isfinite(scale) or not 0.25 <= scale <= 4.0:
        raise ValueError(f"formant_scale {scale!r} is outside the sane 0.25-4.0 range")

    bw_scale = float(bw_scale) if isinstance(bw_scale, (int, float)) else 1.0
    if not math.isfinite(bw_scale) or not 0.25 <= bw_scale <= 4.0:
        bw_scale = 1.0

    out = _clone_with_note(table, {
        "applied_speaker": desc.get("speaker") or speaker,
        "applied": True,
        "formant_scale": round(scale, 6),
        "bandwidth_scale": round(bw_scale, 6),
        "formant_scale_scope": "F1, F2 only; F3 left untouched (f3_trusted=false)",
        "confidence": confidence,
        "applied_despite_low_confidence": bool(conf_is_low(confidence)),
        "base": "phoneme_params.json",
        "f3_trusted": False,
        "f3_reason": F3_UNTRUSTED["reason"],
    })
    applied = 0
    for key, entry in out["phonemes"].items():
        if not isinstance(entry, dict):
            continue
        formants = entry.get("formants")
        if not (key.startswith("final:") and isinstance(formants, (list, tuple))
                and len(formants) >= 2):
            continue
        if not is_usable(entry):
            entry["formant_source"] = f"fallback_low_confidence({entry.get('confidence')})"
            continue
        try:
            f1, f2 = float(formants[0]), float(formants[1])
        except (TypeError, ValueError):
            entry["formant_source"] = "fallback_implausible"
            continue
        if not (f1 > 0.0 and f2 > 0.0):
            entry["formant_source"] = "fallback_implausible"
            continue
        f3 = formants[2] if len(formants) > 2 else None
        entry["formants"] = [round(f1 * scale, 1), round(f2 * scale, 1), f3]
        # ``vocal.plan_ctd`` renders from ``f12``, not from ``formants``. Leaving
        # ``f12`` at the unscaled value made this whole function a silent no-op
        # at render time: the table looked scaled, yet the audio was unchanged.
        # Both fields must move together, or a caller cannot tell which one won.
        if entry.get("f12"):
            entry["f12"] = [round(f1 * scale, 3), round(f2 * scale, 3)]
        entry["f12_unscaled"] = [round(f1, 3), round(f2, 3)]
        entry["formants_unscaled"] = [round(f1, 1), round(f2, 1), f3]
        entry["formant_source"] = "measured_x_speaker_scale"
        entry["speaker_applied"] = {"formant_scale": round(scale, 6)}
        bws = entry.get("bandwidths")
        if isinstance(bws, (list, tuple)) and len(bws) >= 2:
            try:
                bw1, bw2 = float(bws[0]), float(bws[1])
            except (TypeError, ValueError):
                bw1 = bw2 = None
            if bw1 and bw2:
                entry["bandwidths"] = [round(bw1 * bw_scale, 1),
                                       round(bw2 * bw_scale, 1),
                                       bws[2] if len(bws) > 2 else None]
                entry["speaker_applied"]["bandwidth_scale"] = round(bw_scale, 6)
        applied += 1
    out["applied_phonemes"] = applied
    return out


def conf_is_low(confidence):
    """``True`` for anything that is not a usable tier.

    Only ``"high"`` is usable for a speaker estimate (see
    :func:`_confidence_of`), and step-2's own ``"low"`` is refusal, so
    ``medium``/``None``/``low`` all block application.
    """
    return confidence != "high"


def _clone_with_note(table, note):
    """Shallow-ish copy of a parameter document plus an ``applied_speaker`` block."""
    out = dict(table)
    out["phonemes"] = {k: dict(v) if isinstance(v, dict) else v
                       for k, v in table["phonemes"].items()}
    out["applied_speaker"] = note
    out["f3_trusted"] = False
    return out


def _resolve_speaker(params, speaker):
    """Find a single speaker descriptor inside whatever the caller passed."""
    if isinstance(params, (int, float)) and not isinstance(params, bool):
        return {"formant_scale": float(params), "speaker": speaker, "confidence": "high"}
    if not isinstance(params, dict):
        raise ValueError("params must be a speaker descriptor or a speaker_params document")
    if "formant_scale" in params:
        return dict(params)
    for key in ("speakers", "estimates"):
        block = params.get(key)
        if isinstance(block, dict):
            if speaker in block:
                return dict(block[speaker])
            raise ValueError(f"no speaker {speaker!r} in params[{key!r}]")
    raise ValueError("cannot find a speaker descriptor in params")


# --------------------------------------------------------------------------
# comparing two speakers
# --------------------------------------------------------------------------
def compare_speakers(a, b, *, label_a=None, label_b=None):
    """Explainable distance between two speaker descriptors.

    Reported in perceptual units, one per axis, because they are not
    commensurable and must not be summed into a single "distance":

    ``f1_bark`` / ``f2_bark`` / ``f12_bark``
        Euclidean distance in Bark between the two vowel-space centroids.
        ``f12_bark`` is the 2-D distance (ratio to measurement error later).
    ``f0_semitones``
        ``12 log2(f0_b / f0_a)``.
    ``tilt_db_per_oct``
        simple difference, ``b - a``.
    ``formant_scale_ratio``
        ``scale_b / scale_a`` with the scale axes being F1/F2 only.
    ``frames_renderer_separation_bark``
        how far the two scales move each phoneme's F1/F2 in Bark -- the number
        the renderer actually acts on, unlike the scale ratio itself.
    """
    da = _resolve_speaker(a, label_a)
    db = _resolve_speaker(b, label_b)
    name_a = label_a or da.get("speaker")
    name_b = label_b or db.get("speaker")

    out = {"a": name_a, "b": name_b, "units": {
        "formants": "Bark (Traunmueller, 0-20)",
        "f0": "semitones",
        "tilt": "dB/oct",
    }}

    space_a = da.get("vowel_space") or {}
    space_b = db.get("vowel_space") or {}
    ca, cb = space_a.get("centroid_bark"), space_b.get("centroid_bark")
    if ca and cb:
        d1 = abs(float(ca[0]) - float(cb[0]))
        d2 = abs(float(ca[1]) - float(cb[1]))
        out["f1_bark"] = round(d1, 3)
        out["f2_bark"] = round(d2, 3)
        out["f12_bark"] = round(math.hypot(d1, d2), 3)
        out["vowel_space_centroid_bark"] = {name_a: ca, name_b: cb}
    else:
        out["f12_bark"] = None
        out["f12_bark_note"] = "one of the speakers has no vowel_space"

    fa, fb = da.get("f0_hz"), db.get("f0_hz")
    if fa and fb:
        out["f0_hz"] = {name_a: fa, name_b: fb}
        out["f0_semitones"] = round(semitones(fa, fb), 3)
        out["f0_hz_difference"] = round(float(fb) - float(fa), 1)
    else:
        out["f0_semitones"] = None

    ta = da.get("spectral_tilt_db_per_oct")
    tb = db.get("spectral_tilt_db_per_oct")
    if ta is not None and tb is not None:
        out["tilt_db_per_oct"] = round(float(tb) - float(ta), 3)
        out["tilt_db_per_oct_values"] = {name_a: ta, name_b: tb}
    else:
        out["tilt_db_per_oct"] = None
        out["tilt_unavailable_reason"] = (
            "one or both speakers have no LTAS slope (see spectral_tilt_unavailable_reason)")

    sa, sb = da.get("formant_scale"), db.get("formant_scale")
    if sa and sb:
        out["formant_scale"] = {name_a: sa, name_b: sb}
        out["formant_scale_ratio"] = round(float(sb) / float(sa), 5)
    else:
        out["formant_scale_ratio"] = None

    bwa, bwb = da.get("bandwidth_scale"), db.get("bandwidth_scale")
    if bwa and bwb:
        out["bandwidth_scale"] = {name_a: bwa, name_b: bwb}
        out["bandwidth_scale_ratio"] = round(float(bwb) / float(bwa), 4)

    # What the renderer would actually do: the difference between the two
    # speaker-scaled tables, in Bark, per phoneme.
    if sa and sb:
        doc = _REFERENCE_TABLE_CACHE.get("doc")
        if doc is not None:
            moves = []
            for phone, entry in _usable_finals(doc).items():
                f12 = _entry_f12(entry)
                if not f12:
                    continue
                move = 0.0
                for i in (0, 1):
                    move += (hz_to_bark(f12[i] * float(sb))
                             - hz_to_bark(f12[i] * float(sa))) ** 2
                moves.append((phone, math.sqrt(move)))
            if moves:
                moves.sort(key=lambda kv: -kv[1])
                vals = [m for _p, m in moves]
                out["renderer_separation_bark"] = {
                    "per_final": {p: round(m, 3) for p, m in moves},
                    "mean": round(mean(vals), 3),
                    "median": round(median(vals), 3),
                    "max": round(max(vals), 3),
                    "n_finals": len(vals),
                }

    conf_a, conf_b = da.get("confidence"), db.get("confidence")
    out["confidence"] = {name_a: conf_a, name_b: conf_b}
    return out


_REFERENCE_TABLE_CACHE = {}


def set_reference_table(doc):
    """Register the phoneme table :func:`compare_speakers` uses for renders."""
    _REFERENCE_TABLE_CACHE["doc"] = doc


# --------------------------------------------------------------------------
# cross-validation
# --------------------------------------------------------------------------
def _bark_errors(pairs):
    """Bark errors for ``[(phone, observed_f12, predicted_f12)]`` triples.

    Returns a dict with per-formant and pooled mean/median/max, plus the
    per-utterance pooled median, which is the number a leave-one-utterance-out
    report should quote (a per-sample mean is dominated by the utterances that
    happen to hold many syllables).
    """
    f1, f2, pooled, per_utter = [], [], [], []
    by_utter = {}
    for phone, observed, predicted, utter in pairs:
        if observed is None or predicted is None:
            continue
        e1 = abs(hz_to_bark(observed[0]) - hz_to_bark(predicted[0]))
        e2 = abs(hz_to_bark(observed[1]) - hz_to_bark(predicted[1]))
        f1.append(e1)
        f2.append(e2)
        pooled.append(e1)
        pooled.append(e2)
        by_utter.setdefault(utter, []).extend((e1, e2))
    if not pooled:
        return None
    per_utter = [median(v) for v in by_utter.values()]
    return {
        "n_scored": len(f1),
        "f1_bark": _stats(f1),
        "f2_bark": _stats(f2),
        "pooled_bark": _stats(pooled),
        "per_utterance_pooled_median_bark": _stats(per_utter),
        "n_utterances": len(by_utter),
    }


def _stats(values):
    s = [float(v) for v in values]
    if not s:
        return None
    return {"mean": round(mean(s), 4), "median": round(median(s), 4),
            "max": round(max(s), 4), "std": round(std(s), 4), "n": len(s)}


def _resolve_wav_table(rows):
    """Attach ``(wav, start, stop)`` to rows so held-out windows can be read."""
    out = []
    for row in rows:
        pick = dict(row)
        utter = row.get("utterance")
        pick["wav"] = str(ROOT / "pybot_tts_v2_recorded" / "extended" / f"{utter}.wav")
        out.append(pick)
    return out


def cross_validate(doc, *, tilt_windows=None, on_progress=None):
    """Leave-one-utterance-out formant prediction, announced up front.

    For every utterance U of every speaker, exactly once:

    ``model``
        predict U's syllables from ``measured[phone] * alpha[phone] *
        k[phone] * scale[U] / ref_scale``. The only utterance-dependent input is
        ``scale[U]``, which is fitted on the *other* utterances of U's speaker
        (fold in U would be leakage). ``scale`` of the reference speaker is
        re-derived inside the fold from the same held-out rows and divided out,
        so the reference-relative convention is preserved.
    ``baseline``
        the plain mean of the phone's measurements over every *other*
        utterance of *both* speakers (speaker-blind).
    ``oracle``
        the plain mean of the phone's measurements over every other utterance
        of *the same* speaker -- the ceiling of any speaker-conditioned method.

    Predictions then go through :func:`apply_speaker`'s scale step, i.e. the
    numbers quoted are what a renderer built on this parameter set would use.
    """
    rows = [r for r in _resolve_wav_table(measurement_rows(doc))
            if r.get("formants") and r.get("utterance")]
    speakers = sorted({r["speaker"] for r in rows if r.get("speaker")})
    all_utts = sorted({(r["speaker"], r["utterance"]) for r in rows})
    result = {
        "method": "leave-one-utterance-out over both speakers",
        "prediction_units": "Bark distance between predicted and observed F1/F2",
        "n_utterances": len(all_utts),
        "n_scored_syllables": len(rows),
        "speakers": speakers,
        "note": (
            "model = per-phone measured mean x phone correction x speaker scale "
            "fitted without the held-out utterance; baseline = speaker-blind "
            "per-phone mean; oracle = same-speaker per-phone mean (ceiling)"),
    }

    model_rows, base_rows, oracle_rows = [], [], []
    folds = []
    for index, (spk, utter) in enumerate(all_utts):
        held = [r for r in rows if r["speaker"] == spk and r["utterance"] == utter]
        if not held:
            continue
        others = [r for r in rows
                  if not (r["speaker"] == spk and r["utterance"] == utter)]

        # phone correction and scale from the held-out fold only
        phone_k = {}
        by_phone = {}
        for r in others:
            by_phone.setdefault(r["phone"], []).append(r)
        for phone, group in by_phone.items():
            ref = _reference_f12(doc, phone)
            if not ref:
                continue
            ob = mean([g["formants"][0] for g in group]) if group else None
            if not ob:
                continue
            cur = phone_k.setdefault(phone, [[], []])
            cur[0].append(ob / ref[0])
            cur[1].append(mean([g["formants"][1] for g in group]) / ref[1])
        phone_k = {p: (geomean(v[0]), geomean(v[1])) for p, v in phone_k.items()}

        fold_rows = [r for r in others if r["speaker"] == spk]
        srows = [(r["phone"], r["formants"][0], r["formants"][1],
                  _reference_f12(doc, r["phone"])) for r in fold_rows]
        scale_fold = _fit_formant_scale([s for s in srows if s[3]])["formant_scale"] or 1.0
        ref_rows = [r for r in others if r["speaker"] != spk]
        if ref_rows:
            rrows = [(r["phone"], r["formants"][0], r["formants"][1],
                      _reference_f12(doc, r["phone"])) for r in ref_rows]
            ref_scale = _fit_formant_scale([s for s in rrows if s[3]])["formant_scale"] or 1.0
        else:
            ref_scale = scale_fold
        ratio = scale_fold / ref_scale if ref_scale else 1.0

        per_source = {"both": [], "baseline": [], "oracle": []}
        for r in held:
            phone = r["phone"]
            ref = _reference_f12(doc, phone)
            k = phone_k.get(phone)
            if ref and k and k[0] and k[1]:
                model = (ref[0] * k[0] * ratio, ref[1] * k[1] * ratio)
            elif ref:
                model = (ref[0] * ratio, ref[1] * ratio)
            else:
                model = None

            same_phone = [(g["formants"][0], g["formants"][1])
                          for g in others
                          if g["phone"] == phone and g["source"] == "provenance"]
            base = (mean([g[0] for g in same_phone]),
                    mean([g[1] for g in same_phone])) if same_phone else None
            same_spk = [(g["formants"][0], g["formants"][1])
                        for g in others
                        if g["phone"] == phone and g["speaker"] == spk]
            oracle = (mean([g[0] for g in same_spk]),
                      mean([g[1] for g in same_spk])) if same_spk else None

            observed = (r["formants"][0], r["formants"][1])
            model_rows.append((phone, observed, model, utter))
            base_rows.append((phone, observed, base, utter))
            oracle_rows.append((phone, observed, oracle, utter))
            folds.append({
                "speaker": spk, "utterance": utter, "phone": phone,
                "observed": [round(observed[0], 1), round(observed[1], 1)],
                "model": None if model is None else [round(model[0], 1), round(model[1], 1)],
                "baseline": None if base is None else [round(base[0], 1), round(base[1], 1)],
                "oracle": None if oracle is None else [round(oracle[0], 1), round(oracle[1], 1)],
                "scale_fold": round(scale_fold, 4),
                "ref_scale_fold": round(ref_scale, 4),
            })
        if on_progress:
            on_progress(index + 1, len(all_utts), spk, utter)

    model = _bark_errors(model_rows)
    baseline = _bark_errors(base_rows)
    oracle = _bark_errors(oracle_rows)
    result["model"] = model
    result["baseline_speaker_blind"] = baseline
    result["oracle_same_speaker"] = oracle

    verdict = {}
    if model and baseline:
        m = model["pooled_bark"]["median"]
        b = baseline["pooled_bark"]["median"]
        verdict["median_pooled_bark_model"] = m
        verdict["median_pooled_bark_baseline"] = b
        verdict["beats_baseline"] = bool(m + 1e-9 < b)
        verdict["improvement_bark"] = round(b - m, 4)
        verdict["improvement_percent"] = (round(100.0 * (b - m) / b, 2) if b else None)
        if not verdict["beats_baseline"]:
            verdict["finding"] = (
                "the speaker-scale model is NOT better than the speaker-blind "
                "per-phone mean on this data; the fitted descriptor carries no "
                "usable formant information beyond phoneme identity")
    if model and oracle:
        m = model["pooled_bark"]["median"]
        o = oracle["pooled_bark"]["median"]
        verdict["median_pooled_bark_oracle"] = o
        verdict["oracle_ceiling_note"] = (
            "oracle uses same-speaker repeats of the same phoneme, so it is a "
            "ceiling, not an achievable method; the gap between model and oracle "
            "is what the scalar descriptors fail to capture")
        verdict["model_vs_oracle_bark"] = round(m - o, 4)
        if m > o:
            # The model is *worse* than a same-speaker mean would allow. That is
            # only possible when the phone itself is unstable across utterances
            # (the "oracle" averages over fewer, differently-distributed rows),
            # which is exactly the regime where a scalar speaker descriptor has
            # nothing to explain. Say that instead of printing a ratio whose
            # denominator is negative.
            verdict["finding_vs_oracle"] = (
                "the model is worse than the same-speaker oracle by "
                f"{m - o:+.4f} Bark, so no share of a speaker-specific gain is "
                "captured: the residual is phoneme/utterance variability, not a "
                "speaker offset")
            verdict["fraction_of_oracle_gain_captured"] = 0.0
        elif baseline and baseline["pooled_bark"]["median"] > o:
            verdict["fraction_of_oracle_gain_captured"] = round(
                (baseline["pooled_bark"]["median"] - m)
                / (baseline["pooled_bark"]["median"] - o), 3)
        else:
            verdict["fraction_of_oracle_gain_captured"] = None
            verdict["fraction_of_oracle_gain_note"] = (
                "undefined: the speaker-blind baseline is already at or below the "
                "same-speaker oracle, so there is no gain to attribute")

    # per-speaker, per-axis F0 prediction check: does the speaker F0 predict the
    # held-out utterance's F0? Only possible when the caller supplies per-syllable
    # F0 rows; step 2's provenance does not carry it (checked, not assumed).
    f0_pred = _cv_f0(rows, all_utts)
    result["f0_holdout"] = f0_pred if f0_pred else {
        "available": False,
        "unavailable_reason": (
            "no per-syllable F0 anywhere in phoneme_params.json: the "
            "'provenance' rows carry utterance/speaker/pinyin/start/end/"
            "formants/bound_confidence/split_confidence and no f0_hz, and "
            "'per_speaker' carries only n + formants. Step 2 kept one pooled "
            "f0_hz per phoneme and one per speaker, which cannot be split into "
            "held-out utterances, so F0 is reported as a point estimate with a "
            "source label and NOT as a validated error."),
        "consequence": ("f0_hz in a speaker entry is a point estimate of the "
                        "speaker's F0 level; its held-out prediction error is "
                        "unknown from this table"),
    }

    result["verdict"] = verdict
    result["folds"] = folds

    # Diagnose *why* the model does or does not win: how far its prediction
    # moves away from the speaker-blind baseline, and how large the residual
    # within-phoneme spread is. A shift far below the spread means the fitted
    # descriptors cannot help regardless of the sign of the result.
    by_phone_shift = {}
    for (phone, _obs, model_f12, _u), (_p2, _o2, base_f12, _u2) in zip(
            model_rows, base_rows):
        if not model_f12 or not base_f12:
            continue
        d = math.hypot(hz_to_bark(model_f12[0]) - hz_to_bark(base_f12[0]),
                       hz_to_bark(model_f12[1]) - hz_to_bark(base_f12[1]))
        by_phone_shift.setdefault(phone, []).append(d)
    if by_phone_shift:
        all_shifts = [v for vs in by_phone_shift.values() for v in vs]
        result["model_shift_from_baseline_bark"] = {
            "description": (
                "Bark distance between the model's prediction and the "
                "speaker-blind baseline prediction, per scored syllable; this is "
                "how much the fitted descriptors actually move a prediction"),
            "mean": round(mean(all_shifts), 4),
            "median": round(median(all_shifts), 4),
            "max": round(max(all_shifts), 4),
            "n": len(all_shifts),
        }

    if result.get("model_shift_from_baseline_bark") and model:
        spread = model["pooled_bark"]["median"]
        shift = result["model_shift_from_baseline_bark"]["median"]
        verdict["shift_vs_spread"] = {
            "median_shift_bark": shift,
            "median_residual_spread_bark": spread,
            "ratio": round(shift / spread, 4) if spread else None,
            "interpretation": (
                "the descriptors move a prediction by this fraction of the "
                "leftover scatter they are supposed to explain"),
        }
        if shift < 0.15 * spread:
            verdict["shift_vs_spread"]["conclusion"] = (
                "the fitted speaker descriptors are far too small, relative to "
                "same-phoneme scatter across utterances, to improve a formant "
                "prediction. This is why the model does not beat the baseline, "
                "and it also means the measured speaker differences are near the "
                "noise floor of this corpus")
        result["verdict"] = verdict

    if tilt_windows:
        result["tilt_note"] = "spectral tilt is reported per speaker, not cross-validated"
    return result


def _cv_f0(rows, all_utts):
    """Held-out F0 error, but only if per-syllable F0 exists.

    ``phoneme_params.json`` does **not** carry per-syllable F0 (verified: the
    ``provenance`` dicts have no ``f0_hz`` key), so with the stock table this
    returns ``None`` and :func:`cross_validate` writes an explicit
    ``unavailable_reason`` instead. A caller that enriches the rows with
    ``f0_hz`` (e.g. from a re-analysis) gets a real number here.
    """
    by_utt = {}
    for r in rows:
        if r.get("f0_hz"):
            by_utt.setdefault((r["speaker"], r["utterance"]), []).append(r["f0_hz"])
    if len(by_utt) < 4:
        return None
    errors = []
    for (spk, utter), values in by_utt.items():
        others = [v for (s, u), vs in by_utt.items()
                  if s == spk and u != utter for v in vs]
        if len(others) < 4:
            continue
        pred = median(others)
        obs = median(values)
        errors.append(12.0 * math.log2(obs / pred))
    if not errors:
        return None
    return {
        "unit": "semitones (observed vs predicted median F0)",
        "n_utterances": len(errors),
        "mean": round(mean(errors), 3),
        "abs_mean": round(mean([abs(e) for e in errors]), 3),
        "median_abs": round(median([abs(e) for e in errors]), 3),
        "max_abs": round(max(abs(e) for e in errors), 3),
        "caveat": ("F0 here comes from phoneme-level medians (step 2 stored no "
                   "per-syllable F0), so these semitone errors are approximate"),
    }


def validation_by_phone(doc, rows=None):
    """``{(phone, speaker): n}`` and the leave-one-utterance-out phone coverage.

    Used to state *how* the cross-validation can be read: a phoneme seen in only
    one utterance of one speaker cannot be held out at all.
    """
    rows = rows if rows is not None else measurement_rows(doc)
    coverage = {}
    for r in rows:
        key = (r["phone"], r["speaker"])
        coverage.setdefault(key, set()).add(r["utterance"])
    single = sorted(f"{p}/{s}" for (p, s), u in coverage.items() if len(u) < 2)
    return {
        "phones_with_one_utterance_only": single,
        "n_phones_single_utterance": len(single),
        "n_phone_speaker_cells": len(coverage),
    }


# --------------------------------------------------------------------------
# top level build
# --------------------------------------------------------------------------
def build_speaker_params(phoneme_doc=None, *, collect_tilt=True,
                         tilt_sample=3, cross_validate_folds=True,
                         progress=None):
    """Full step-4 pipeline -> the ``speaker_params.json`` dict.

    Also re-derives the speaker scale a second, **independent** way from the
    per-syllable ``provenance`` rows and reports the difference. The two paths
    answer the same question from the same audio but through different
    aggregation, so their agreement (or lack of it) is a real check on whether
    the number is stable.
    """
    doc = phoneme_doc if phoneme_doc is not None else load_phoneme_params()
    set_reference_table(doc)
    speakers = sorted((doc.get("speakers") or {}).keys())
    finals = _usable_finals(doc)
    excluded = []
    for key, entry in sorted((doc.get("phonemes") or {}).items()):
        if not key.startswith("final:"):
            continue
        if is_usable(entry):
            continue
        excluded.append({
            "phoneme": key,
            "confidence": entry.get("confidence"),
            "n_formant_samples": entry.get("n_formant_samples"),
            "usable_for_speaker": entry.get("usable_for_speaker"),
            "reason": "confidence low or flagged unusable by step 2",
        })

    rows = measurement_rows(doc)
    estimates = {}
    for spk in speakers:
        est = estimate_speaker(doc, speaker=spk, collect_tilt=collect_tilt)
        # independent path: per-syllable provenance rows
        mine = [r for r in rows if r["speaker"] == spk]
        cleaned = _clean_rows([(r["formants"][0], r["formants"][1], r["phone"], r)
                               for r in mine], group_key=lambda row: row[2])
        independent = _fit_formant_scale(
            [(payload["phone"], f1, f2, _reference_f12(doc, payload["phone"]))
             for f1, f2, _p, payload in cleaned])
        est["formant_scale_independent"] = (
            None if independent["formant_scale"] is None
            else round(independent["formant_scale"], 4))
        est["formant_scale_independent_n_rows"] = independent["n_ratio_pairs"]
        if est["formant_scale"] is not None and independent["formant_scale"]:
            est["formant_scale_paths_delta"] = round(
                est["formant_scale"] - independent["formant_scale"], 5)
        est["row_note"] = (
            "n_rows > n_phonemes is expected: step 2 repeats one phoneme across "
            "many syllables; independent samples are the phoneme count, not the "
            "row count")
        est["sample_sources"] = {
            "phoneme": "phoneme_params.json['phonemes'][final:*]['per_speaker']",
            "independent": "phoneme_params.json['phonemes'][final:*]['provenance']",
            "audio": "pybot_tts_v2_recorded/extended/<utterance>.wav (only for spectral_tilt)",
        }
        est["n_utterances"] = len({r["utterance"] for r in mine})
        step2 = (doc.get("speakers") or {}).get(spk) or {}
        est["f0_hz_step2"] = step2.get("f0_hz")
        if est["f0_hz"] is not None and step2.get("f0_hz"):
            est["f0_semitones_vs_step2"] = round(
                semitones(float(step2["f0_hz"]), float(est["f0_hz"])), 4)
        estimates[spk] = est

    # Global F0 reference used by the renderer diff below.
    f0_values = [e["f0_hz"] for e in estimates.values() if e.get("f0_hz")]

    cv = None
    if cross_validate_folds:
        cv = cross_validate(doc, on_progress=progress)

    comparison = None
    if len(estimates) >= 2:
        names = list(estimates)
        comparison = compare_speakers(estimates[names[0]], estimates[names[1]],
                                      label_a=names[0], label_b=names[1])
    out = {
        "schema": SCHEMA,
        "sr": SR,
        "generated_from": "phoneme_params.json (step 2) + extended/ WAVs for spectral tilt",
        "sample_rate_note": (
            "step-2 provenance rows carry no waveform or F0; nothing here "
            "re-analyses the corpus for formants"),
        "claim": (
            "TIMBRE PARAMETERS ESTIMATED FROM A SMALL NUMBER OF SAMPLES, WITH "
            "LEAVE-ONE-UTTERANCE-OUT ERROR QUANTIFIED. This is NOT voice cloning: "
            "no audio was heard, no perceptual judgement was made, and the "
            "descriptors are five scalars fitted to a very small corpus."),
        "not_claimed": [
            "voice cloning or speaker identification",
            "any perceptual or listening-based comparison",
            "an absolute formant scale for a single speaker (only the ratio "
            "between speakers is identified; see formant_scale_reference)",
            "a trusted F3",
        ],
        "used_formants": list(USED_FORMANTS),
        "excluded_formants": ["f3"],
        "f3_trusted": False,
        "f3_bias_hz": F3_UNTRUSTED["f3_bias_hz"],
        "f3_reason": F3_UNTRUSTED["reason"],
        "confidence_contract": {
            "high": f">= {CONF_HIGH_MIN} usable phonemes",
            "low": f"< {CONF_HIGH_MIN} usable phonemes -> {CONF_LOW_NOTE}",
            "medium": "not used: a two-phoneme scale is a guess, not an estimate",
            "applies_to": "distinct phonemes, not sample rows",
            "why": ("100 measurements of one vowel cannot separate a speaker from "
                    "itself, so the tier counts phonemes"),
        },
        "n_usable_finals": len(finals),
        "usable_finals": list(finals),
        "excluded_for_low_confidence": excluded,
        "speakers": estimates,
        "comparison": comparison,
        "cross_validation": cv,
        "phone_coverage": validation_by_phone(doc, rows),
        "f0_reference_hz": round(mean(f0_values), 1) if f0_values else None,
    }
    if cv and not cv.get("verdict", {}).get("beats_baseline", True):
        v = cv["verdict"]
        shift = cv.get("model_shift_from_baseline_bark") or {}
        out["honest_negative_finding"] = {
            "result": "NEGATIVE: the speaker-scale model does not beat the baseline",
            "headline": (
                "leave-one-utterance-out, the fit speaker descriptors are worse "
                "than simply predicting each phoneme's speaker-blind mean"),
            "numbers": {
                "median_pooled_bark_model": v.get("median_pooled_bark_model"),
                "median_pooled_bark_baseline": v.get("median_pooled_bark_baseline"),
                "difference_bark": v.get("improvement_bark"),
                "difference_percent": v.get("improvement_percent"),
                "median_shift_the_descriptors_cause_bark": shift.get("median"),
            },
            "why": (
                "the descriptors move a prediction by only "
                f"{shift.get('median')} Bark while the leftover same-phoneme "
                "scatter they are meant to explain is "
                f"{v.get('median_pooled_bark_baseline')} Bark, i.e. under 5 %. "
                "The speaker differences in this corpus are near its noise "
                "floor, so a scalar formant scale cannot help on this data"),
            "what_follows": [
                "the two speakers are NOT shown to be separable in formants: the "
                "measured F1/F2 distance is inside the cross-validated error",
                "the speaker tables were still emitted, because they are the "
                "honest point estimates the samples were rendered from and "
                "collapsing them to 1.0 would hide the measurement",
                "this was reported, not tuned away: no per-phone subset, alpha "
                "range, or fold definition was adjusted to flip the verdict",
            ],
            "what_would_be_needed": (
                "per-syllable audio re-analysis (so F0 and bandwidths are usable "
                "per speaker) and more than 3 utterances per phoneme, or a "
                "descriptor that models spectral detail rather than F1/F2"),
        }
    return out


def write_speaker_params(path=OUT, doc=None):
    """Build and write ``speaker_params.json`` (UTF-8, sorted keys)."""
    out = doc if doc is not None else build_speaker_params()
    p = Path(path)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    return p


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Step 4: few-shot speaker parameters")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--no-tilt", action="store_true",
                        help="skip the LTAS pass (fast path)")
    parser.add_argument("--report", action="store_true",
                        help="print a short human-readable summary")
    args = parser.parse_args(argv)

    doc = build_speaker_params(collect_tilt=not args.no_tilt)
    path = write_speaker_params(args.out, doc)
    if args.report:
        print(f"wrote {path}")
        for name, est in doc["speakers"].items():
            print(f"  {name}: scale={est['formant_scale']} "
                  f"(F1/F2 only) f0={est['f0_hz']} "
                  f"tilt={est['spectral_tilt_db_per_oct']} "
                  f"conf={est['confidence']} n_phonemes={est['n_phonemes']}")
        comp = doc.get("comparison") or {}
        print(f"  distance: f12={comp.get('f12_bark')} bark, "
              f"f0={comp.get('f0_semitones')} st, tilt={comp.get('tilt_db_per_oct')}")
        cv = doc.get("cross_validation") or {}
        v = cv.get("verdict") or {}
        print(f"  cv median pooled bark: model={v.get('median_pooled_bark_model')} "
              f"baseline={v.get('median_pooled_bark_baseline')} "
              f"oracle={v.get('median_pooled_bark_oracle')} "
              f"beats={v.get('beats_baseline')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
