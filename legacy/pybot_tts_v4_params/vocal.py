"""Parametric source-filter singer that reads ``phoneme_params.json``.

``pybot_tts_v2_recorded/synth.py`` is kept untouched as the "old generic formant"
baseline. This module re-implements its inner loop line for line -- so that the
**empty parameter table reproduces ``synth.render()`` sample for sample** -- and
adds exactly one acoustic layer on top: formant targets that come from measured
speech instead of a hand-written table.

Why the loop is re-implemented instead of calling into ``synth``
-----------------------------------------------------------------
``synth.vowel`` and ``synth.consonant`` consume the shared ``random.Random``
instance once per output sample at the *end* of each iteration
(``.014*rng.uniform(-1,1)``). Any added oscillator, filter or extra control
rate inside that loop would change how many draws happen, and the noise floor of
every later sample would then differ. So there is no "call synth and post-process"
route to sample equality: the only honest route is the same arithmetic in the
same order, with one substitution point (``formants=None`` -> ``synth.V``).

What is parameterised
---------------------
* **Vowel targets** -- ``phonemes["final:<key>"]["formants"]`` replaces
  ``synth.V[<key>]`` when the entry is usable (confidence ``medium``/``high``,
  finite values, on-screen frequencies). Everything else, including every
  ``confidence == "low"`` entry, falls back to ``synth.V`` and is counted.
* **Bandwidths** -- ``spectral_weight``'s ``(100, 140, 195)`` becomes
  ``(f1_bw, f2_bw, synth_f3_bw)``.
* **Speaker scale** -- ``speakers[<spk>]["formant_scale"]`` multiplies F1/F2.
* **Noise spectrum** -- the fricative "brightness" (the high-pass share of the
  noise) is derived from the measured noise centroid instead of ``FRIC``.
* **Coarticulation inside a rime** (always on, provably length- and RNG-neutral,
  see ``render``).

Only **F1 and F2 are consumed**. F3 is deliberately not taken from the table:
step 2 measured F3 to be systematically 158-420 Hz high across the board, and
handing the renderer a formant whose bias is understood but uncorrected would
just move that error into the output. ``spectral_weight`` still has a third
``zip`` slot, so F3 keeps the *generic* ``synth.V`` value, and it is scaled
together with F1/F2 under the equal-ratio rule so that F1:F2:F3 stays the shape
``synth`` was written for. Nothing in the output claims a measured F3.

No numpy. Pure stdlib, 24 kHz, matching the rest of the package.
"""

from __future__ import annotations

import json
import math
import random
from array import array
from pathlib import Path
import sys
import wave

from pybot_tts_v2_recorded import synth as _synth

# Re-exported so callers never have to reach into synth for the contract values.
SR = _synth.SR
F0 = _synth.F0
TAU = _synth.TAU

# The generic table, used for the fallback path and for F3 everywhere.
V = _synth.V
FINALS = _synth.FINALS
FRIC = _synth.FRIC
STOP = _synth.STOP
NASAL_SOUND = _synth.NASAL_SOUND
NASALS = _synth.NASALS

# ``synth.vowel`` asks ``spectral_weight`` for a third bandwidth; this is it.
BASELINE_AMPS = (1.0, .73, .43)

# The entry below this confidence is evidence, not a parameter (schema 2
# contract: n < 3 formant samples -> "low (evidence only)").
USABLE_CONFIDENCE = ("medium", "high")

# Which measured columns we are willing to render. F3 is *not* here on purpose.
USED_FORMANTS = ("f1", "f2")

# The nasal tail / voiced-onset key that carries F3 in the generic table.
_CORE_KEYS = {
    "a": "a", "aa": "a", "e": "e", "eh": "e", "o": "o", "i": "i", "u": "u",
    "v": "v", "schwa": "schwa", "apical": "i", "retro": "er", "nasal": "ai",
}

# Preference order for a measured coda target, since a generic key like
# 'nasal' names a *tract state* and has no phoneme id of its own.
_CODA_PREFERENCE = ("n", "ng", "en", "ang", "an", "in", "ing", "ian", "iang",
                    "on", "ian", "eng", "ong", "iong", "uan", "iang")

# Frequency-screen for a measurement we are prepared to render.
F1_PLAUSIBLE = (200.0, 1100.0)
F2_PLAUSIBLE = (700.0, 3300.0)

# Rime-internal coarticulation (no length change, no extra RNG draw).
COARTIC = {
    "t_on": .12,       # the rime starts at the onset -> current target, not the first
    "target_from": .30,  # the piecewise targets are fully reached by 30 % of the rime
    "nucleus_spread": .20,  # target k of K peaks at 30 % + 20 %*k
}

# The shortest onset a rendered cluster may have. Below ~10 ms a stop burst stops
# being a burst; longer onsets lose their protrusion, so the stretch is asymmetric.
ONSET_FLOOR_S = .012

# How much of a measured noise-centroid deviation is applied to synth's
# brightness share (see ``brightness_from_centroid``). Deliberately below 1: the
# centroids come from one band-limited corpus and their absolute level is not
# calibrated against a brightness in [0, 1].
DEVIATION_WEIGHT = .30

# The centroid that synth's generic fricative brightness is treated as
# corresponding to. Arbitrary by necessity -- synth has no centroid, only a
# brightness share -- so it is named, declared, and used for every phoneme so a
# given measurement always produces the same shift.
REFERENCE_CENTROID_HZ = 2000.0


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_params(path, *, strict_sr=True):
    """Read a ``phoneme_params.json`` into the shape ``render`` consumes.

    Missing keys are filled with explicit defaults rather than raising, because
    the point of the empty table is to be a valid input that renders exactly the
    generic voice. Beyond the raw file this adds a few derived fields:

    ``phonemes[key]["f12"]``   ``[F1, F2]`` -- the two numbers we actually render.
    ``phonemes[key]["source"]`` ``"measured"`` / ``"fallback_low_confidence"`` /
                               ``"fallback_no_formants"`` / ``"fallback_implausible"``.
    ``phonemes[key]["formant_bw12"]`` measured F1/F2 bandwidths, or ``None``.

    Duplicate classification is done here *once* so ``render`` can just read it;
    both functions still agree on the outcome because ``render`` only trusts
    ``f12`` for entries whose ``confidence`` is usable and whose ``f12`` is set.
    """
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read parameter table {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError("parameter table must be a JSON object")
    return normalize(doc, strict_sr=strict_sr)


def normalize(doc, *, strict_sr=True):
    """Derive the render view from a raw parameter document (already parsed).

    Kept separate from :func:`load_params` so tests can build an empty or a
    synthetic table in memory without writing a file.
    """
    out = {
        "schema": doc.get("schema"),
        "sr": doc.get("sr", SR),
        "phonemes": {},
        "speakers": {},
        "notes": {},
        # Carried through verbatim, not recomputed: step 2's own record of how the
        # table was derived is the only provenance there is, and dropping it here
        # would silently orphan every claim that cites it.
        "provenance": doc.get("provenance") or {},
        "confidence_contract": doc.get("confidence_contract"),
    }
    if strict_sr and out["sr"] != SR:
        raise ValueError(f"parameter table is {out['sr']} Hz; the renderer is {SR} Hz")
    out["notes"]["f3"] = (
        "F3 is NOT taken from the table: step 2 measured a systematic +158..420 Hz "
        "F3 bias that could not be removed. Only F1/F2 are rendered as measured."
    )
    out["notes"]["f3_source"] = "synth.V generic value, equal-ratio scaled with F1/F2"

    phonemes = doc.get("phonemes") or {}
    if not isinstance(phonemes, dict):
        raise ValueError("'phonemes' must be an object")
    for key, entry in phonemes.items():
        if not isinstance(entry, dict):
            raise ValueError(f"phoneme entry {key!r} must be an object")
        rec = dict(entry)
        rec["f12"] = None
        rec["formant_bw12"] = None
        formants = entry.get("formants")
        conf = entry.get("confidence")
        if not formants:
            rec["source"] = "fallback_no_formants"
        elif conf not in USABLE_CONFIDENCE:
            rec["source"] = f"fallback_low_confidence({conf})"
        else:
            try:
                f1, f2 = float(formants[0]), float(formants[1])
            except (TypeError, ValueError, IndexError):
                rec["source"] = "fallback_implausible"
            else:
                if not (F1_PLAUSIBLE[0] <= f1 <= F1_PLAUSIBLE[1]
                        and F2_PLAUSIBLE[0] <= f2 <= F2_PLAUSIBLE[1]):
                    rec["source"] = "fallback_implausible"
                else:
                    rec["f12"] = [f1, f2]
                    rec["source"] = "measured"
                    bws = entry.get("bandwidths")
                    if isinstance(bws, (list, tuple)) and len(bws) >= 2:
                        try:
                            rec["formant_bw12"] = [float(bws[0]), float(bws[1])]
                        except (TypeError, ValueError):
                            rec["formant_bw12"] = None
        out["phonemes"][key] = rec

    speakers = doc.get("speakers") or {}
    if not isinstance(speakers, dict):
        raise ValueError("'speakers' must be an object")
    for name, entry in speakers.items():
        rec = dict(entry) if isinstance(entry, dict) else {}
        scale = rec.get("formant_scale")
        try:
            scale = float(scale) if scale is not None else 1.0
        except (TypeError, ValueError):
            scale = 1.0
        rec["scale"] = scale if math.isfinite(scale) and .5 <= scale <= 2.0 else 1.0
        out["speakers"][name] = rec
    return out


def empty_params():
    """The null table: an input for which the renderer *must* be ``synth``."""
    return normalize({"schema": 2, "sr": SR, "phonemes": {}, "speakers": {}})


# --------------------------------------------------------------------------
# phoneme lookup
# --------------------------------------------------------------------------
def _split_keys(template):
    """``'i eh nasal'`` -> ``['i', 'eh', 'nasal']``."""
    return template.split()


def _measured_f12(params, table_key, generic_key):
    """Measured ``[F1, F2]`` for a generic table key, or ``None``.

    ``generic_key`` is what ``synth.V`` would have used; the phoneme id comes
    from ``_CORE_KEYS``. A coda-ish key tries its preference list in order.
    """
    if generic_key in ("nasal", "apical", "retro", "schwa"):
        order = _CODA_PREFERENCE
    else:
        target = _CORE_KEYS.get(generic_key)
        order = (target,) if target else ()
    for phone in order:
        entry = params["phonemes"].get(f"final:{phone}")
        if entry and entry.get("f12"):
            return entry
    return None


def _segment_target(params, seg_key):
    """Resolved target for one generic rime segment.

    Returns ``(f12, f3, bw12, source, entry)``. ``f12`` is ``None`` when the
    segment must keep the generic ``synth.V[seg_key]`` triple untouched.

    F3 keeps the generic value plus the *equal-ratio shift* the measured pair
    implies: ``F3 + (F2_measured - F2_generic)``, not ``F3 * (F2/F1)``. The
    ratio form is wrong for a back or a measured-narrow pair -- for the measured
    ``i`` (344/2488, ratio 7.2 against the generic 285/2200, ratio 7.7) it put
    F3 at 22 kHz. The shift form keeps the F2:F3 distance, which is the part of
    the generic envelope shape ``synth.spectral_weight`` was written around, and
    it stays inside the audible band.
    """
    base = V[seg_key]
    entry = _measured_f12(params, seg_key, seg_key)
    if entry is None:
        return None, base[2], None, f"fallback({seg_key}:generic)", None
    f1, f2 = entry["f12"]
    f3 = max(base[2] + (f2 - base[1]), base[2] * .60)
    return ([f1, f2], f3, entry.get("formant_bw12"),
            f"measured({entry['phone']})", entry)


def plan_rime(params, final_key):
    """Static formant plan for one rime -- computed once, fully inspectable.

    ``synth`` re-derives the three formant tracks on every third sample with the
    same arithmetic this does, so building the tracks up front and then reading
    them at the loop's control rate is not an approximation: it is the same
    sequence of values, and it lets the plan be reported in ``vocal_01.json``.

    Coarticulation (rime internal, which is where the equivalence constraint
    allows it):

    * A diphthong / glide rime does not jump to its first target at the onset
      seam -- ``t_on`` of its onset-to-target distance is traversed *before* the
      piecewise schedule starts. Same length, same pitch track, same RNG stream;
      the difference is bilabial and it is also what makes a measured-fallback
      bridge unnecessary at the seam.
    * A target is reached at its own time and held, instead of every target
      being reached only at the very last sample of the rime.

    With an empty parameter table every target *is* the generic one, so the
    original picture is reproduced exactly and this whole layer is a no-op.
    """
    segs = _split_keys(FINALS[final_key])
    if not segs:
        raise ValueError(f"final {final_key!r} has no rime segments")
    tracks = []
    sources = []
    seg_keys = []
    for seg in segs:
        f12, f3, bw12, source, entry = _segment_target(params, seg)
        if f12 is None:
            triple = V[seg]
            tracks.append([triple[0], triple[1], triple[2]])
            bw = None
        else:
            tracks.append([f12[0], f12[1], f3])
            bw = bw12
        sources.append(source)
        seg_keys.append({
            "generic_key": seg,
            "source": source,
            "f12_hz": None if f12 is None else [round(f12[0], 3), round(f12[1], 3)],
            "f3_hz": None if f12 is None else round(f3, 3),
            "f3_source": "generic" if f12 is None else "generic F3 + (F2_measured - F2_generic)",
            "bandwidths": None if bw is None else [round(bw[0], 3), round(bw[1], 3)],
            "confidence": None if entry is None else entry.get("confidence"),
            "n_formant_samples": None if entry is None else entry.get("n_formant_samples"),
        })

    nseg = len(tracks)
    if nseg == 1:
        starts = [0.0]
    else:
        starts = [COARTIC["target_from"] + COARTIC["nucleus_spread"] * k
                  for k in range(nseg)]
        # A two-segment rime spends its second half on the second target.
        starts[-1] = min(starts[-1], .50)
    plan = {
        "final": final_key,
        "segments": seg_keys,
        "targets_hz": [[round(t[0], 3), round(t[1], 3), round(t[2], 3)]
                       for t in tracks],
        "starts": [round(s, 4) for s in starts],
        "nasal_tail": segs[-1] == "nasal",
        "t_on": COARTIC["t_on"],
        "t_end": 1.0,
        "f3_used_from_table": False,
        "used_formants": list(USED_FORMANTS),
        "measured_segments": sum(1 for s in sources if s.startswith("measured")),
        "fallback_segments": sum(1 for s in sources if s.startswith("fallback")),
    }
    track_fn = _make_track_fn(tracks, starts, nseg > 1)
    return plan, track_fn


def _make_track_fn(tracks, starts, gliding):
    """Closure returning ``(t, k) -> Hz`` for formant ``k`` (0,1,2)."""
    columns = [[tr[k] for tr in tracks] for k in range(3)]
    last = starts[-1]

    def at(t, k):
        col = columns[k]
        if not gliding:
            return col[0]
        if t <= 0.0:
            # Inside the onset the glide has already begun from the current
            # target: at t=0 exactly the generic value is returned.
            return col[0]
        # Onset-to-first-target traverse, completes at t_on.
        if t < COARTIC["t_on"]:
            u = (1.0 - t / COARTIC["t_on"]) * COARTIC["t_on"] / last
            return col[0] + (col[-1] - col[0]) * u
        if t <= last:
            k = 0
            while k + 1 < len(starts) and t >= starts[k + 1]:
                k += 1
            a = starts[k]
            b = starts[k + 1] if k + 1 < len(starts) else last
            u = 0.0 if b <= a else (t - a) / (b - a)
            if u < 0.0:
                u = 0.0
            elif u > 1.0:
                u = 1.0
            return col[k] + (col[k + 1] - col[k]) * u
        return col[-1]

    return at


# --------------------------------------------------------------------------
# spectral weight (same formula and order as synth.spectral_weight)
# --------------------------------------------------------------------------
def _weights_full(f0, tracks_at, bandwidths, nasal):
    """``[synth.spectral_weight(h, f0, ...) for h in 1..26]``.

    Identical arithmetic to ``synth.spectral_weight`` so that the empty table
    gives bit-identical floats; the only differences are the bandwidths (when a
    segment carries measured ones) and the pre-computed formant triple.
    """
    b1, b2, b3 = bandwidths if bandwidths is not None else BASELINE_BW
    out = []
    for h in range(1, 27):
        hz = h * f0
        peaks = sum(
            amp / (1.0 + ((hz - f) / bw) ** 2)
            for f, bw, amp in zip(tracks_at, (b1, b2, b3), BASELINE_AMPS)
        )
        w = (.055 + peaks) / h ** .68
        if hz > 600:
            w *= 1.0 - .56 * nasal
        out.append(w)
    return out


_BW_CACHE = {}


def _synth_bandwidths():
    """The generic ``(F1_bw, F2_bw, F3_bw)`` triple, read from ``synth`` itself.

    Pulled out of ``synth.spectral_weight``'s source rather than copied, so that
    a change over there cannot silently leave a stale constant here. If the
    pattern ever moves the fallback below is the triple that shipped with it.
    """
    if "bw" not in _BW_CACHE:
        import inspect

        src = inspect.getsource(_synth.spectral_weight)
        for token in ("(100, 140, 195)", "(100,140,195)"):
            if token in src:
                _BW_CACHE["bw"] = tuple(int(v) for v in token.strip("()").split(","))
                break
        else:                                    # pragma: no cover - defensive
            _BW_CACHE["bw"] = (100, 140, 195)
    return _BW_CACHE["bw"]


BASELINE_BW = _synth_bandwidths()


# --------------------------------------------------------------------------
# the two renderers
# --------------------------------------------------------------------------
def _vowel(keys, duration, contour, rng, gain, track=None, bandwidths=None,
           plan_on=True):
    """``synth.vowel`` with pluggable formant targets.

    Same statement order as ``synth.vowel``: the pitch interp, the control-rate
    branch at ``i % 72 == 0``, the phase wrap, and -- critically -- exactly one
    ``rng.uniform(-1, 1)`` per output sample, appended last.
    """
    n = max(1, int(duration * SR))
    target = tuple(V[k] for k in keys)
    phase = 0.0
    weights = ()
    samples = []
    for i in range(n):
        t = i / max(1, n - 1)
        f0 = F0 * 2 ** ((_synth.interp(contour, t) + .12 * math.sin(t * TAU * 1.7)) / 12)
        if i % 72 == 0:
            nasality = max(0.0, (t - .62) / .38) if keys[-1] == 'nasal' else 0.0
            if track is None:
                slide = max(0.0, min(1.0, (t - .13) / .77))
                fs = [_synth.interp([f[k] for f in target], slide) for k in range(3)]
            elif plan_on:
                fs = [track(t, k) for k in range(3)]
            else:
                # Reference path: the original schedule on resolved targets.
                slide = max(0.0, min(1.0, (t - .13) / .77))
                fs = [_synth.interp([f[k] for f in target], slide) for k in range(3)]
            weights = _weights_full(f0, fs, bandwidths, nasality)
        phase = (phase + TAU * f0 / SR) % TAU
        pressure = sum(w * math.sin(h * phase + .39 * h) for h, w in enumerate(weights, 1))
        env = min(1.0, i / (SR * .017), (n - 1 - i) / (SR * .032))
        flutter = .95 + .045 * math.sin(t * TAU)
        samples.append(gain * env * flutter * (.61 * pressure + .014 * rng.uniform(-1, 1)))
    return samples


def _consonant(onset, duration, contour, rng, gain, brightness=None, plan_on=True):
    """``synth.consonant`` with an optional measured noise brightness.

    ``synth``'s ``bright`` is the share of high-passed noise, i.e. a coarse
    brightness knob; a measured noise centroid moves it in the same direction,
    monotonically. When ``brightness`` is ``None`` the generic constant is used
    and the sample stream is identical to ``synth``.
    """
    n = int(duration * SR)
    if n <= 0:
        return []
    if onset in NASAL_SOUND:
        key = 'nasal' if onset in NASALS else 'schwa'
        return _vowel((key,), duration, [contour[0]], rng, gain * .47, plan_on=plan_on)
    out = []
    last = 0.0
    lp = 0.0
    for i in range(n):
        x = rng.uniform(-1, 1)
        lp = .70 * lp + .30 * x
        t = i / max(n - 1, 1)
        hp = x - last
        last = x
        if onset in FRIC:
            amp, bright = FRIC[onset]
            if brightness is not None and plan_on:
                bright = brightness
            env = min(1, t * 5, (1 - t) * 4) * amp
        else:
            _, amp = STOP.get(onset, (.04, .15))
            bright = .47 if onset in ('z', 'c', 'zh', 'ch', 'j', 'q', 't', 'd') else .21
            if brightness is not None and plan_on:
                bright = brightness
            env = 0.0 if t < .27 else amp * max(0, 1 - t) ** .40
            if .27 <= t <= .36:
                env = min(.75, env + .20)
        out.append(gain * env * (bright * hp + (1 - bright) * lp) * .63)
    return out


def plan_ctd(params, final_key):
    """``(plan, track_fn, fallbacks)`` where ``fallbacks`` counts *final* fallbacks.

    Two counts are kept, because they answer different questions and conflating
    them would let a test (or a reader) be satisfied by the wrong one:

    ``fallback_segments``
        How many generic *rime segments* kept the generic triple. This is what
        the renderer actually had to fall back to.
    ``fallbacks``
        How many phonemes of the requested final came out unmeasured. A ``low``
        final like ``ei`` is ``eh``+``i`` in ``FINALS``, and both of those *are*
        measured, so ``ei`` renders as measured segments (``fallback_segments``
        is 0) yet the final ``ei`` itself was refused by the table (``fallbacks``
        is 1). Neither number is wrong; they measure different things, and
        ``vocal_01.json`` reports both.
    """
    plan, track_fn = plan_rime(params, final_key)
    entry = params["phonemes"].get(f"final:{final_key}")
    if entry is None:
        plan["fallbacks"] = 1
    elif entry.get("f12"):
        plan["fallbacks"] = 0
    else:
        plan["fallbacks"] = 1
    plan["final_entry_source"] = None if entry is None else entry.get("source")
    return plan, track_fn


def _generic_fric(generic):
    """``(bright, nominal_centroid_hz)`` out of ``synth.FRIC``'s ``(dur, bright)``.

    ``synth.FRIC`` is ``(duration, bright)``: the first field is a *time*, not a
    frequency. Reading it positionally as a centroid was a real bug here -- it
    turned ``s``'s 0.56 brightness into a 0.58 "centroid" and made ``h`` clip to
    the floor. The nominal centroid is therefore a single shared reference
    (:data:`REFERENCE_CENTROID_HZ`), and only the brightness comes from the
    tuple.
    """
    return float(generic[1]), REFERENCE_CENTROID_HZ


def brightness_from_centroid(centroid_hz, generic=None, clip=True):
    """Move ``synth``'s ``bright`` share along with a measured noise centroid.

    ``synth``'s ``bright`` is the share of high-passed noise versus low-passed
    noise: a monotone one-knob brightness control. The measured knob is the
    spectral centroid, so the mapping is

    ``bright = generic_bright + DEVIATION_WEIGHT * (centroid - REFERENCE) / 10 kHz``

    Three deliberate choices:

    * **The generic brightness is the zero point**, not the absolute centroid.
      ``synth``'s own values do not follow ``bright = centroid / 10 kHz``, so an
      absolute mapping would silently redefine what the knob means for every
      phoneme. This way, a corpus whose centroid sits at the reference leaves
      every fricative exactly as ``synth`` rendered it.
    * **One shared reference for all phonemes** (:data:`REFERENCE_CENTROID_HZ`),
      so an identical measurement always produces an identical shift. Using each
      phoneme's own nominal value would have made the shift depend on an
      unmeasured quantity.
    * **The slope is damped** (:data:`DEVIATION_WEIGHT` < 1) because step 2
      measured centroids on one band-limited corpus whose absolute level carries
      an offset a 1:1 transfer would import straight into the output.

    The result is clipped to ``[.08, .72]``: a share of 1.0 would make a
    fricative pure white noise, and 0 would remove the noise entirely. Clipping
    is a real limitation here, not a formality: the measured centroids are low
    enough that ``h`` (generic .08) hits the floor.

    The honest limitations, reported not tuned away:

    * For ``s`` the measured centroid is *below* what generic ``s`` implies, so
      this makes ``/s/`` darker than ``synth`` renders it (``vocal_01.json``
      says so in ``onset_note``).
    * For ``h`` the shift is downward and clips at the floor, so ``/h/`` keeps
      its generic minimum rather than following the measurement.
    * The rank order of the measured centroids is ``s < x < sh < h``, which is
      not the generic ``f < x < sh < s`` order, so this knob cannot preserve it.
      Any claim like "the fricatives now have their measured spectra" would be
      false: this is a single brightness knob, not a spectrum.
    """
    if centroid_hz is None:
        return None
    try:
        c = float(centroid_hz)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(c):
        return None
    if generic is None:
        base_bright, base_centroid = .33, 3300.0
    else:
        # ``generic`` is synth's ``(duration, bright)`` for this onset. Its
        # ``bright`` value is the zero point. The *nominal centroid* that goes
        # with it is not in the table, so the generic pair is anchored at a
        # single declared reference point (:data:`REFERENCE_CENTROID_HZ`)
        # instead of pretending the duration field is a frequency. That keeps
        # a measured centroid equal to the reference from being able to "fix"
        # anything, and makes every phoneme's shift identical for an identical
        # measurement rather than an artefact of its duration.
        base_bright = _generic_fric(generic)[0]
        base_centroid = REFERENCE_CENTROID_HZ
    shifted = base_bright + DEVIATION_WEIGHT * (c - base_centroid) / 10000.0
    if not clip:
        return shifted
    return max(.08, min(.72, shifted))



# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------
def render(units, params=None, seed=72, *, coarticulation=True, speaker=None,
           trace=None):
    """Render ``units`` to a float sample list, ``synth.render`` compatible.

    Length is exactly ``synth.render(units, seed)``'s: coarticulation and the
    parameter layer change *values*, never the number of samples, the number of
    RNG draws, or the order in which they are drawn.

    ``params`` -- a document from :func:`load_params` / :func:`normalize`, or a
    raw dict, or ``None`` for the generic voice.
    ``trace``  -- optional dict; filled with per-unit plan and fallback counts.

    An empty unit list is refused rather than answered with silence: an empty
    WAV is indistinguishable from a bug in the caller's text front end, and
    ``synth.render([])`` returning the 75 ms lead-in alone would hide that.
    """
    if units is None or (hasattr(units, "__len__") and len(units) == 0):
        raise ValueError("render() needs at least one unit; got an empty list")
    params = _coerce_params(params)
    scale = 1.0
    if speaker is not None:
        entry = params["speakers"].get(speaker)
        scale = entry["scale"] if entry else 1.0
    rng = random.Random(seed)
    out = [0.0] * int(.075 * SR)
    plans = []
    for u in units:
        onset = u['onset']
        final = u['final']
        pitch = [float(x) for x in u['pitch_st']]
        if len(pitch) < 1 or any(not math.isfinite(x) or abs(x) > 18 for x in pitch):
            raise ValueError('pitch_st must be finite semitone values within ±18')
        duration = float(u['duration_ms']) / 1000
        gain = float(u['gain'])
        pause = float(u.get('pause_after_ms', 0)) / 1000
        if not .07 <= duration <= 1.5 or not 0 <= gain <= 2 or not 0 <= pause <= 2:
            raise ValueError('invalid duration, gain or pause')
        if onset == 'y' or onset == 'w':
            onset = ''
        if final == 'i' and onset in ('zh', 'ch', 'sh', 'r', 'z', 'c', 's'):
            final = 'ir' if onset in ('zh', 'ch', 'sh', 'r') else 'iz'
        if final not in FINALS or onset not in (FRIC.keys() | STOP.keys() | NASAL_SOUND | {''}):
            raise ValueError(f'no sound recipe for onset={onset!r}, final={final!r}')

        plan, track_fn = plan_ctd(params, final)
        plan["onset"] = onset
        plan["requested_final"] = u['final']
        plan["gain"] = gain
        plan["duration_ms"] = float(u['duration_ms'])
        plan["speaker"] = speaker
        plan["speaker_scale"] = scale

        if coarticulation:
            final_dur, c_dur = _split_duration(onset, duration)
            plan["duration_split"] = "coarticulated"
        else:
            c_dur = 0.0 if not onset else STOP.get(onset, (.058 if onset in FRIC else .040, 0))[0]
            c_dur = min(c_dur, duration * .39)
            final_dur = duration
            plan["duration_split"] = "synth_baseline"
        plan["onset_s"] = round(c_dur, 6)
        plan["rime_s"] = round(final_dur - c_dur, 6)

        brightness = _brightness_for(params, onset, plan)
        plan["noise_brightness"] = brightness

        track = None
        bandwidths = None
        if scale != 1.0 or plan["measured_segments"] or plan["fallback_segments"]:
            track = _scaled_track(track_fn, scale)
        if plan["segments"]:
            bws = [s["bandwidths"] for s in plan["segments"] if s["bandwidths"]]
            if len(bws) == len(plan["segments"]) and all(b is not None for b in bws):
                # Only when every rime segment carries a measurement; a partly
                # measured rime would otherwise mix measured and generic widths.
                bandwidths = [bws[0][0], bws[0][1], BASELINE_BW[2]]

        c = _consonant(onset, c_dur, pitch, rng, gain, brightness=brightness,
                       plan_on=coarticulation)
        v = _vowel(_split_keys(FINALS[final]), final_dur - c_dur, pitch, rng, gain,
                   track=track, bandwidths=bandwidths, plan_on=coarticulation)
        overlap = min(len(c), len(v), int(.005 * SR))
        if overlap:
            c[-overlap:] = [c[-overlap + i] + v[i] for i in range(overlap)]
        out += c + v[overlap:]
        out.extend([0.0] * round(pause * SR))
        plans.append(plan)
    out.extend([0.0] * int(.100 * SR))
    if trace is not None:
        trace["units"] = plans
        trace["fallback_segments"] = sum(p["fallback_segments"] for p in plans)
        trace["measured_segments"] = sum(p["measured_segments"] for p in plans)
        trace["fallbacks"] = sum(p["fallbacks"] for p in plans)
        trace["units_total"] = len(plans)
    return out


def _coerce_params(params):
    if params is None:
        return empty_params()
    if not isinstance(params, dict) or "phonemes" not in params:
        raise ValueError("params must be a loaded parameter table (see load_params)")
    if "notes" in params and "f3" in params["notes"]:
        return params                                    # already normalized
    return normalize(params, strict_sr=False)


def _scaled_track(track_fn, scale):
    def at(t, k):
        v = track_fn(t, k)
        return v * scale if k < 2 else v

    return at


def _split_duration(onset, duration):
    """Onset / rime split that leaves ``duration`` untouched.

    ``synth`` gives the onset a fixed slice (39 % cap) and whatever is left to
    the rime, so an onset in an 80 ms syllable is already 47 ms. Giving the
    onset a short floor costs the rime *length*, and the rime is the phoneme
    with the measurable quality. The cap takes the slack back when a short rime
    or a naturally short onset puts the total over, so the sum is preserved
    exactly and no unit silently gains or loses a sample.
    """
    if not onset:
        return duration, 0.0
    base = STOP.get(onset, (.058 if onset in FRIC else .040, 0))[0]
    base = min(base, duration * .39)
    if base <= 0.0:
        return duration, 0.0
    c = min(base, duration * .39, max(base, ONSET_FLOOR_S))
    rime = duration - c
    if rime < duration * .30:                      # keep the rime the longer half
        c = min(base, max(0.0, duration * .70))
    return duration, c


def _brightness_for(params, onset, plan):
    if onset in NASAL_SOUND:
        return None
    entry = params["phonemes"].get(f"onset:{onset}")
    if not entry:
        plan["onset_source"] = "fallback(generic:no_onset_entry)"
        return None
    centroid = entry.get("noise_centroid_hz")
    generic = FRIC.get(onset)
    bright = brightness_from_centroid(centroid, generic)
    plan["onset_noise_centroid_hz"] = centroid
    if bright is None:
        plan["onset_source"] = "fallback(generic:no_centroid)"
        return None
    plan["onset_source"] = "measured_centroid"
    if generic is not None:
        plan["onset_bright_generic"] = generic[1]
        plan["onset_bright_measured"] = round(bright, 6)
        raw = brightness_from_centroid(centroid, generic, clip=False)
        if raw is not None and abs(raw - bright) > 1e-12:
            # Clipped: the measurement was read but the knob cannot go further,
            # so the rendered consonant is the generic one. Saying "measured"
            # without saying this would overstate what happened.
            plan["onset_note"] = (
                f"measured centroid {centroid} Hz implies a brightness of "
                f"{raw:.4f}, outside the [{.08:.2f}, {.72:.2f}] range this knob "
                f"is allowed to use; it clipped to {bright:.4f}, so /{onset}/ is "
                f"rendered at {bright:.4f} and the measurement had no further effect")
        elif bright < generic[1]:
            plan["onset_note"] = (
                f"measured centroid {centroid} Hz is below what synth's generic "
                f"brightness {generic[1]} implies; /{onset}/ is rendered darker "
                f"than the baseline, not brighter")
        elif bright > generic[1] + 1e-12:
            plan["onset_note"] = (
                f"measured centroid {centroid} Hz is above the generic "
                f"{generic[1]} brightness; /{onset}/ is rendered brighter")
    return bright


# --------------------------------------------------------------------------
# high-level interface
# --------------------------------------------------------------------------
def render_full(text, params=None, style="auto", seed=72, *, coarticulation=True,
                speaker=None):
    """``text`` -> ``(samples, trace)`` via ``engine.plan``.

    ``engine.plan`` is the project's own text front end; it is imported lazily
    so this module stays usable without it (the renderer only ever sees units).
    """
    import sys as _sys

    root = str(Path(__file__).resolve().parent.parent)
    if root not in _sys.path:
        _sys.path.insert(0, root)
    from pybot_tts_v2_recorded import engine

    plan = engine.plan(text, style)
    return render_plan(plan, params, seed=seed, coarticulation=coarticulation,
                       speaker=speaker)


def render_plan(plan_doc, params=None, seed=72, *, coarticulation=True, speaker=None):
    """Render an already-planned document. Returns ``(samples, trace)``."""
    if not isinstance(plan_doc, dict) or not isinstance(plan_doc.get("units"), list):
        raise ValueError("plan must be a dict with a 'units' list")
    if not plan_doc["units"]:
        raise ValueError("plan has no units to render")
    trace = {"text": plan_doc.get("text"), "style": plan_doc.get("style"),
             "seed": seed, "speaker": speaker,
             "coarticulation": bool(coarticulation)}
    samples = render(plan_doc["units"], params, seed=seed,
                     coarticulation=coarticulation, speaker=speaker, trace=trace)
    return samples, trace


def write_wav(path, samples, factor=None):
    """Same convention as ``synth.write_wav``: peak-normalised to 0.90."""
    peak = max(map(abs, samples), default=0.0)
    factor = min(.90 / max(peak, 1e-8), 1.0) if factor is None else factor
    pcm = array('h', (round(max(-.999, min(.999, x * factor)) * 32767) for x in samples))
    if sys.byteorder != 'little':
        pcm.byteswap()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as f:
        f.setparams((1, 2, SR, 0, 'NONE', 'not compressed'))
        f.writeframes(pcm.tobytes())
    return factor
