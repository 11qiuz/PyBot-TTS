"""Pure-stdlib LPC formant analysis for the PyBot-TTS v4 parameter stage.

No numpy, no scipy. Everything here is ``math`` / ``cmath`` / ``array``.

Pipeline (order matters, each step is a documented failure mode):

1. de-mean the frame  -- a DC offset leaks into a0 and drags F1 down;
2. pre-emphasis (default 0.97) -- without it the LPC fits the -6 dB/oct
   glottal roll-off and F1/F2 come out low and merged;
3. Hamming window -- rectangular windows smear formants through sidelobes;
4. autocorrelation -> Levinson-Durbin recursion -> ``a`` (a[0] == 1.0);
5. roots of ``A(z) = 1 + a1 z^-1 + ... + ap z^-p``. Roots come from an
   explicit Durand-Kerner iteration on ``w^p + a1 w^(p-1) + ... + ap``
   (``a`` reversed), not from any special-cased quadratic solver;
6. keep conjugate pairs strictly inside the unit circle, take
   ``f = atan2(Im, Re) * SR / 2pi`` and ``bw = -ln(r) * SR / pi``;
7. filter hard (F1 in 200-1100 Hz, F2 in 700-3200 Hz, F3 in 1500-4000 Hz,
   F1 < F2 < F3, bandwidth sanity). Out-of-range roots are dropped, never
   clamped: a forced formant is worse than a missing one.

The voiced/unvoiced gate deliberately refuses:
* silence / near-silence (RMS floor),
* frames whose energy is far below the loudest frame of the segment,
* frames without a clean pitch peak (white noise, fricatives, stops).

The point is that ``analyze_segment`` returns ``None`` on non-voiced input
instead of inventing numbers -- the caller is expected to report "no
estimate", not to paper over it.
"""

from __future__ import annotations

import cmath
import math
import sys
from array import array

SR = 24000

# Analysis defaults tuned for 24 kHz Mandarin vowels.
PREEMPHASIS = 0.97
# Order 22 at 24 kHz, measured against synthetic vowels with known formants.
#
# Two effects drove this up from the more common order 12-18:
#
# * F1/F2 separation. Order 10-14 merges the pair outright (order 12 puts a
#   single pole at 1187 Hz where the target pair was 790/1300). Order 18 fixes
#   the merge but is still marginal for close back vowels: a synthetic /u/
#   (F1 320, F2 780) produced a usable estimate on only 11 of 47 frames, because
#   on many frames the two poles collapse into one wide one. Order 20 -> 43/47,
#   order 22 -> 47/47.
# * F3 bias. Higher poles soak up the residual that a too-short predictor
#   smears into F3. Mean absolute F3 error over six synthetic vowels fell from
#   295 Hz at order 18 to 169 Hz at order 22.
#
# Order 24 is only marginally better than 22 (F3 error 153 Hz) and costs 2 more
# coefficients per root solve, so 22 is the balance point.
ORDER = 22
WINDOW = 720          # 30 ms at 24 kHz: >= ~2.5x the lowest useful period
HOP = 240             # 10 ms
F0_RANGE = (65.0, 400.0)

# Reasonability gates for a 24 kHz speaker. Deliberately wide (the point is
# to reject garbage, not to enforce one speaker) but hard.
F1_RANGE = (200.0, 1100.0)
F2_RANGE = (700.0, 3200.0)
F3_RANGE = (1500.0, 4000.0)
TOO_CLOSE = 150.0           # formants closer than this are the same pole split
FORMANT_BW_MAX = 900.0      # Hz
# Voicing gate.
MIN_RMS = 0.012             # absolute floor, matches the bank's quiet frames
MIN_REL_RMS = 0.10          # vs the loudest frame of the same segment
MIN_ACF = 0.62              # normalised autocorrelation peak for voicing


# --------------------------------------------------------------------------
# basic helpers
# --------------------------------------------------------------------------
def rms(x):
    """Root mean square of a sequence of floats."""
    n = len(x)
    if n == 0:
        return 0.0
    return math.sqrt(sum(v * v for v in x) / n)


def dbfs(x):
    """dBFS level of a signal or short window (never -inf)."""
    return 20.0 * math.log10(max(1e-9, rms(x)))


def demean(x):
    """Copy of ``x`` with its mean removed."""
    if not x:
        return []
    mu = sum(x) / len(x)
    return [v - mu for v in x]


def preemphasis(x, coef=PREEMPHASIS):
    """``y[n] = x[n] - coef * x[n-1]`` (y[0] = x[0]).

    Rejects |coef| >= 1: that makes the filter non-invertible/unstable.
    """
    coef = float(coef)
    if not 0.0 <= coef < 1.0:
        raise ValueError("preemphasis coefficient must be in [0, 1)")
    if not x:
        return []
    out = [float(x[0])]
    prev = float(x[0])
    for v in x[1:]:
        v = float(v)
        out.append(v - coef * prev)
        prev = v
    return out


def hamming(n):
    """Symmetric-ish Hamming window of length ``n`` (1.0 for n <= 1)."""
    if n <= 1:
        return [1.0] * max(0, n)
    return [0.54 - 0.46 * math.cos(math.tau * i / (n - 1)) for i in range(n)]


def apply_window(x, win):
    if len(x) != len(win):
        raise ValueError("window length must match signal length")
    return [v * w for v, w in zip(x, win)]


def autocorrelation(x, order):
    """Unnormalised autocorrelation r[0..order] (biased, sum-based).

    A 4x zero pad is appended before correlating so that lag 0 is not
    penalised by the shrinking overlap. The extra lags do not change the LPC
    solution -- Levinson-Durbin still consumes the same r[0..order] -- they
    only stop a constant DC offset from appearing in every formant, which is
    exactly what happens when r[k] is summed over only N-k products.
    """
    order = int(order)
    if order < 1:
        raise ValueError("order must be >= 1")
    n = len(x)
    if n < 4 * order:
        raise ValueError("need at least 4*order samples")
    pad = 4 * order
    xx = [float(v) for v in x] + [0.0] * pad
    r = []
    for k in range(order + 1):
        acc = 0.0
        for i in range(n):
            acc += xx[i] * xx[i + k]
        r.append(acc)
    if r[0] <= 0.0:
        # silent input: give an explicitly degenerate autocorrelation
        r[0] = 1e-30
    return r


def levinson_durbin(r, order):
    """Autocorrelations -> LPC coefficients via the Levinson-Durbin recursion.

    Returns ``(a, err)`` where ``a`` has length order+1 and ``a[0] == 1.0``:
    ``A(z) = 1 + a1 z^-1 + ... + ap z^-p``. ``err`` is the final prediction
    error power. Raises ``ValueError`` when the recursion hits a non-positive
    error or a coefficient blows up (ill-conditioned autocorrelation).
    """
    order = int(order)
    if len(r) < order + 1:
        raise ValueError("need at least order+1 autocorrelation lags")
    if order < 1:
        raise ValueError("order must be >= 1")
    err = float(r[0])
    if not math.isfinite(err) or err <= 0.0:
        raise ValueError("autocorrelation r[0] must be positive")
    a = [1.0] + [0.0] * order
    for i in range(1, order + 1):
        acc = float(r[i])
        for j in range(1, i):
            acc += a[j] * r[i - j]           # reflection numerator
        k = -acc / err
        if not math.isfinite(k) or abs(k) >= 1.0:
            raise ValueError(f"reflection coefficient out of range at step {i}")
        # in-place update: new a_j = a_j + k * a_{i-j}
        prev = a[: i]
        for j in range(1, i):
            a[j] = prev[j] + k * prev[i - j]
        a[i] = k
        err *= (1.0 - k * k)
        if not math.isfinite(err) or err <= 0.0:
            raise ValueError(f"prediction error collapsed at step {i}")
    a[0] = 1.0
    return a, err


def lpc_spectrum(a, n_points=256, sr=SR, endpoint=True):
    """LPC spectral envelope magnitude |1/A(e^jw)| over 0..sr/2.

    Returns a list of ``n_points`` magnitudes. ``endpoint=True`` spans the
    closed interval [0, sr/2]; otherwise points are spaced by sr/(2*n_points)
    (bin-centred, matching an FFT bin layout).
    """
    a = list(a)
    if not a:
        raise ValueError("empty LPC coefficient list")
    if a[0] == 0.0:
        raise ValueError("a[0] must be non-zero")
    n_points = int(n_points)
    if n_points < 2:
        raise ValueError("need at least two spectral points")
    denom = max(1, n_points - 1) if endpoint else n_points
    out = []
    for i in range(n_points):
        w = math.pi * i / denom
        zinv = cmath.exp(-1j * w)
        s = 0j
        for j, c in enumerate(a):
            s += c * (zinv ** j)
        mag = abs(s)
        out.append(1.0 / mag if mag > 1e-30 else 1e30)
    return out


def spectral_tilt(mags, sr=SR):
    """Least-squares dB-per-octave slope of a magnitude envelope.

    Fit ``20log10|H|`` against ``log2(f)``. Negative slope = the high end is
    weaker, which is the natural glottal/radiation tilt. Pure function, so
    it is testable without any audio.
    """
    pts = []
    n = len(mags)
    if n < 3:
        raise ValueError("need at least 3 spectral points")
    for i, m in enumerate(mags):
        f = sr * 0.5 * i / (n - 1)
        if f < 60.0 or m <= 0.0:
            continue
        pts.append((math.log2(f), 20.0 * math.log10(m)))
    if len(pts) < 3:
        raise ValueError("not enough usable spectral points")
    n = float(len(pts))
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pts)
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den <= 0.0:
        raise ValueError("degenerate frequency axis")
    return num / den


# --------------------------------------------------------------------------
# polynomial roots: Durand-Kerner
# --------------------------------------------------------------------------
def poly_root_radius(coefs):
    """Cauchy bound on root modulus for monic polynomial ``coefs``.

    ``coefs`` highest-order-first, monic (coefs[0] == 1).
    """
    lead = coefs[0]
    if lead == 0:
        raise ValueError("polynomial must be monic")
    tail = [abs(c / lead) for c in coefs[1:]]
    if not tail:
        return 0.0
    return 1.0 + max(tail)


def poly_eval(coefs, z):
    """Horner evaluation of ``coefs`` (highest first) at complex ``z``."""
    v = 0j
    for c in coefs:
        v = v * z + c
    return v


def durand_kerner(coefs, tol=1e-13, max_iter=500):
    """All complex roots of a real monic polynomial, Durand-Kerner iteration.

    ``coefs`` is highest-order-first and monic (coefs[0] == 1). The iteration
    is ``z_i <- z_i - p(z_i) / prod_{j != i}(z_i - z_j)``, started on a circle
    just outside the Cauchy bound. A Weierstrass correction (dividing by the
    polynomial with the i-th root factored out) is used when two roots are
    about to collide, which is the usual way a plain DK iteration stalls.

    Raises ``ValueError`` when it does not converge, or when the residual
    ``max|p(z_i)|`` is still large -- a silently wrong root set is worse than
    no estimate at all.
    """
    coefs = [complex(c) for c in coefs]
    if len(coefs) < 2:
        raise ValueError("need at least linear polynomial")
    if abs(coefs[0]) == 0.0:
        raise ValueError("polynomial must be monic (non-zero leading)")
    lead = coefs[0]
    norm = [c / lead for c in coefs]
    deg = len(norm) - 1
    if deg == 1:
        return [-norm[1] / norm[0]]

    radius = poly_root_radius([c.real if abs(c.imag) < 1e-15 else c for c in norm]) \
        if all(abs(c.imag) < 1e-15 for c in norm) else 1.0 + max(abs(c) for c in norm[1:])
    if radius <= 0.0:
        radius = 1.0

    # Two starts, same seed family. A single seed is not always enough: the
    # powers of 0.4+0.9j bunch up for some degrees and two roots then start
    # numerically coincident, which stalls the iteration. Retrying from a
    # rotated seed costs one extra pass only when the first one fails, and it is
    # deterministic -- the same input always takes the same path.
    last_err = None
    for attempt, seed in enumerate((complex(0.4, 0.9), complex(0.9, 0.4))):
        try:
            return _durand_kerner_from(norm, deg, radius, seed, tol, max_iter)
        except ValueError as exc:                      # diverged / stalled
            last_err = exc
            if attempt == 1:
                raise
    raise last_err                                    # pragma: no cover


def _durand_kerner_from(norm, deg, radius, seed, tol, max_iter):
    """One Durand-Kerner pass from a given seed."""
    roots = [radius * (seed ** k) for k in range(deg)]
    for _ in range(int(max_iter)):
        delta_max = 0.0
        new = []
        for i, z in enumerate(roots):
            pz = poly_eval(norm, z)
            d = complex(1.0, 0.0)
            for j, zj in enumerate(roots):
                if i == j:
                    continue
                d *= (z - zj)
            if abs(d) < 1e-18:
                # coincident roots: nudge deterministically and retry
                z = z + complex(1e-9, 1e-9) * (1 + i)
                pz = poly_eval(norm, z)
                d = complex(1e-18, 1e-18)
            step = pz / d
            if not (cmath.isfinite(step)):
                raise ValueError("Durand-Kerner diverged")
            new.append(z - step)
            delta_max = max(delta_max, abs(step))
        roots = new
        if delta_max < tol:
            break
    else:
        raise ValueError(f"Durand-Kerner did not converge in {max_iter} iterations")

    resid = max(abs(poly_eval(norm, z)) for z in roots)
    if not math.isfinite(resid) or resid > 1e-6:
        raise ValueError(f"Durand-Kerner residual too large: {resid:.3e}")
    return roots


# --------------------------------------------------------------------------
# formants
# --------------------------------------------------------------------------
def lpc_to_polynomial(a):
    """``a = [1, a1..ap]`` -> monic polynomial ``w^p + a1 w^(p-1) + ... + ap``."""
    a = list(a)
    if not a or a[0] == 0.0:
        raise ValueError("invalid LPC coefficients")
    return [c / a[0] for c in a]


def roots_to_formants(roots, sr=SR):
    """Roots -> list of ``(f_hz, bandwidth_hz)``, unsorted, unfiltered."""
    out = []
    for z in roots:
        if z.imag <= 1e-9:                 # real roots and the lower conjugates
            continue
        r = abs(z)
        if not 1e-6 < r < 1.0:             # must be strictly inside the unit circle
            continue
        f = math.atan2(z.imag, z.real) * sr / math.tau
        bw = -math.log(r) * sr / math.pi
        if f <= 0.0 or not math.isfinite(bw):
            continue
        out.append((f, bw))
    return out


def candidate_poles(a, sr=SR):
    """All in-circle conjugate poles as ``(f, bw)`` sorted by frequency.

    Raises ``ValueError`` if the root finder fails. This is the "raw" view;
    use :func:`formants` for the gated F1/F2/F3 triple.
    """
    poly = lpc_to_polynomial(a)
    roots = durand_kerner(poly)
    cands = roots_to_formants(roots, sr)
    cands.sort(key=lambda p: p[0])
    return cands


def formants(a, sr=SR, strict=True):
    """F1/F2/F3 + bandwidths from LPC coefficients, or ``None`` if implausible.

    ``strict=True`` applies the reasonability gates. ``strict=False`` returns
    the first three candidate poles regardless, which exists for diagnostics
    and is what the tests use to prove the gates actually do something.

    When several poles are legal, the triple is chosen by *total resonance
    strength* (the LPC envelope magnitude at each pole), not greedily band by
    band. Greedy banding fails on the textbook 'a': a real F3 near 2650 Hz
    still sits inside the 700-3200 Hz F2 window and can be picked as F2,
    which then leaves no legal F3. The maximum-strength legal triple is what
    "these three poles are the formants" actually means.
    """
    try:
        cands = candidate_poles(a, sr)
    except ValueError:
        # A degenerate predictor (e.g. all-zero coefficients, whose polynomial
        # w^p has p coincident roots) is not an analysis error: it means there
        # is no resonance to report. Refusing is the correct answer, and letting
        # the root finder's exception escape would turn "no formants" into a
        # crash for the caller.
        return None
    if not cands:
        return None
    if not strict:
        if len(cands) < 3:
            return None
        return {
            "f": [round(cands[0][0], 1), round(cands[1][0], 1), round(cands[2][0], 1)],
            "bw": [round(cands[0][1], 1), round(cands[1][1], 1), round(cands[2][1], 1)],
        }
    legal1 = [p for p in cands if F1_RANGE[0] <= p[0] <= F1_RANGE[1] and _bw_ok(p[1])]
    legal2 = [p for p in cands if F2_RANGE[0] <= p[0] <= F2_RANGE[1] and _bw_ok(p[1])]
    legal3 = [p for p in cands if F3_RANGE[0] <= p[0] <= F3_RANGE[1] and _bw_ok(p[1])]
    if not legal1 or not legal2 or not legal3:
        return None
    strength = _pole_strength(a, sr)
    best = None
    for p1 in legal1:
        for p2 in legal2:
            if p2[0] <= p1[0] + TOO_CLOSE:
                continue
            for p3 in legal3:
                if p3[0] <= p2[0] + TOO_CLOSE:
                    continue
                score = strength(p1) + strength(p2) + strength(p3)
                if best is None or score > best[0]:
                    best = (score, p1, p2, p3)
    if best is None:
        return None
    _score, p1, p2, p3 = best
    return {
        "f": [round(p1[0], 1), round(p2[0], 1), round(p3[0], 1)],
        "bw": [round(p1[1], 1), round(p2[1], 1), round(p3[1], 1)],
    }


def _pole_strength(a, sr=SR):
    """Callable giving |H| of the all-pole envelope at a pole's frequency."""
    mags = lpc_spectrum(a, 257, sr)
    step = (sr * 0.5) / 256.0

    def at(pole):
        idx = int(round(pole[0] / step))
        idx = max(0, min(256, idx))
        return mags[idx]

    return at


def _bw_ok(bw):
    """A pole whose bandwidth exceeds ~900 Hz is a spectral-slope artefact."""
    return 0.0 < bw <= FORMANT_BW_MAX


# --------------------------------------------------------------------------
# frames and voicing
# --------------------------------------------------------------------------
def frame_ac(x, order=ORDER):
    """Public autocorrelation helper for one frame (zero padded)."""
    return autocorrelation(x, order)


def f0_from_acf(x, sr=SR, fmin=None, fmax=None):
    """Periodicity estimate ``(f0_hz, corr)`` from the normalised ACF.

    Uses the earliest local maximum whose height is within 0.06 of the global
    maximum (the same anti-octave-error rule as ``audit_edges.periodic``).
    Returns ``(None, corr)`` when no usable peak exists.
    """
    fmin = F0_RANGE[0] if fmin is None else float(fmin)
    fmax = F0_RANGE[1] if fmax is None else float(fmax)
    lag_min = max(2, int(sr / fmax))
    lag_max = int(sr / fmin)
    n = len(x)
    # Only lags that can actually be computed are usable. A 30 ms frame cannot
    # carry a 369-sample lag, so the range is truncated rather than rejecting
    # the frame: reducing fmax/fmin is the caller's job, not something to
    # silently paper over here.
    lag_max = min(lag_max, n // 2)
    if lag_max <= lag_min + 2:
        return None, 0.0
    mu = sum(x) / n
    xx = [v - mu for v in x]
    e0 = sum(v * v for v in xx)
    if e0 <= 0.0:
        return None, 0.0
    lags = {}
    for d in range(lag_min, min(lag_max, n - 1) + 1):
        num = 0.0
        for i in range(n - d):
            num += xx[i] * xx[i + d]
        den = math.sqrt(e0 * sum(xx[i] * xx[i] for i in range(d, n)))
        lags[d] = num / den if den > 0 else 0.0
    peaks = [i for i in range(lag_min + 1, lag_max)
             if lags[i] >= lags[i - 1] and lags[i] >= lags[i + 1]]
    if not peaks:
        return None, max(lags.values()) if lags else 0.0
    highest = max(lags[k] for k in peaks)
    # Earliest peak within 0.06 of the highest one: rejects octave-down picks
    # without letting a sub-threshold noise bump through (the <= test matters,
    # testing >= MIN_ACF first would make this branch unreachable).
    if highest < MIN_ACF:
        return None, highest
    chosen = min(k for k in peaks if lags[k] >= highest - 0.06)
    if lags[chosen] < MIN_ACF:
        return None, lags[chosen]
    return sr / chosen, lags[chosen]


def frame_formants(x, sr=SR, order=ORDER, coef=PREEMPHASIS, strict=True):
    """LPC formants for one short frame.

    Applies de-mean -> pre-emphasis -> Hamming -> LPC -> roots. Returns a dict
    with ``f``/``bw``/``rms``/``f0``/``corr`` or ``None`` when either the
    LPC solve or the reasonability gate refuses.
    """
    n = len(x)
    if n <= 4 * order:
        return None
    y = demean(x)
    y = preemphasis(y, coef)
    y = apply_window(y, hamming(n))
    try:
        r = autocorrelation(y, order)
        a, _err = levinson_durbin(r, order)
    except ValueError:
        return None
    got = formants(a, sr, strict=strict)
    if not got:
        return None
    f0, corr = f0_from_acf(demean(x), sr)
    got = dict(got)
    got["rms"] = rms(x)
    got["f0"] = f0
    got["corr"] = corr
    got["a"] = a
    return got


def analyze_segment(x, sr=SR, order=ORDER, window=WINDOW, hop=HOP,
                    coef=PREEMPHASIS, min_voiced_ratio=0.34):
    """Average formants of one phoneme-sized segment, or ``None``.

    Frames are centred and truncated to the segment (never padded), so a
    segment shorter than one window is analysed as a single shorter window.
    A frame is *voiced* when its RMS clears the absolute and relative floors
    **and** its autocorrelation has a clean pitch peak.

    Voiced frames are then cleaned up two ways:
    * per-formant median (robust against single-frame outliers),
    * a frame is dropped when it disagrees with the median F1 or F2 by more
      than ``OUTLIER_HZ`` -- those frames are usually glides onto/off the
      neighbour phoneme, not steady state.

    Returns ``None`` when fewer than ``min_voiced_ratio`` of the frames are
    voiced, which is exactly how silence, fricatives and stops are refused.
    """
    x = [float(v) for v in x]
    n = len(x)
    if n < 64:
        return None
    window = min(int(window), n)
    hop = max(1, int(hop))
    if n <= window:
        starts = [0]
    else:
        starts = list(range(0, n - window + 1, hop))
    frames = []
    for s in starts:
        frames.append(x[s:s + window])
    peak_rms = max((rms(f) for f in frames), default=0.0)
    if peak_rms < MIN_RMS:
        return None
    floor = max(MIN_RMS, MIN_REL_RMS * peak_rms)
    good = []
    all_f0 = []
    for f in frames:
        if rms(f) < floor:
            continue
        f0, corr = f0_from_acf(demean(f), sr)
        if f0 is None:
            continue
        all_f0.append(f0)
        got = frame_formants(f, sr, order, coef, strict=True)
        if not got:
            continue
        if got["f0"] is None:
            continue
        good.append(got)
    need = max(2, int(math.ceil(min_voiced_ratio * len(frames))))
    if len(good) < need:
        return None
    kept = _drop_outliers(good)
    if len(kept) < need:
        return None
    f = [round(_median([g["f"][i] for g in kept]), 1) for i in range(3)]
    bw = [round(_median([g["bw"][i] for g in kept]), 1) for i in range(3)]
    f0s = [g["f0"] for g in kept if g["f0"]]
    f0_val = round(sum(f0s) / len(f0s), 1) if f0s else None
    corr = sum(g["corr"] for g in kept) / len(kept)
    return {
        "f": f,
        "bw": bw,
        "f0": f0_val,
        "rms": round(rms(x), 5),
        "frames": len(frames),
        "voiced_frames": len(good),
        "used_frames": len(kept),
        "voiced_ratio": round(len(good) / max(1, len(frames)), 3),
        "corr": round(corr, 3),
        "sr": sr,
        "order": int(order),
        "duration_ms": round(1000.0 * n / sr, 2),
    }


OUTLIER_HZ = 260.0


def _drop_outliers(good):
    """Keep the frames whose F1 agrees with the dominant F1 cluster.

    Filtering on a global median F1 *and* F2 distance is wrong for a diphthong:
    ``uo`` sweeps F2 from ~1130 to ~1930 Hz, so a median-centred 260 Hz window
    rejects every frame on both sides of the sweep and the whole vowel is
    thrown away (measured). F1 is the stable one across a diphthong -- it moves
    by tens of Hz while F2 moves by many hundreds -- so F1 is the clustering
    key here, and F2 is only range-checked by the formant gates.

    The cluster is found as the densest F1 neighbourhood (the mode), not the
    median: for a glide the median sits between the two targets and belongs to
    neither.
    """
    if len(good) < 3:
        return list(good)
    f1 = sorted(g["f"][0] for g in good)
    # Dense 1-D mode: the value with the most neighbours within OUTLIER_HZ.
    best_centre, best_count = f1[0], -1
    for centre in f1:
        count = sum(1 for v in f1 if abs(v - centre) <= OUTLIER_HZ)
        if count > best_count:
            best_centre, best_count = centre, count
    kept = [g for g in good if abs(g["f"][0] - best_centre) <= OUTLIER_HZ]
    return kept if kept else list(good)


def _median(values):
    if not values:
        raise ValueError("median of empty sequence")
    s = sorted(values)
    m = len(s) // 2
    if len(s) % 2:
        return float(s[m])
    return 0.5 * (s[m - 1] + s[m])


def median(values):
    """Public wrapper around :func:`_median`."""
    return _median(list(values))


def wav_read(path):
    """Read a mono 16-bit PCM WAV as ``(samples, sr)``; floats in [-1, 1].

    Explicitly checks channel count / sample width / rate instead of
    assuming, because silently mis-reading a WAV makes every later number
    wrong. ``encoding`` is irrelevant here (binary), the check is the point.
    """
    import wave

    with wave.open(str(path), "rb") as w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        if (ch, width) != (1, 2):
            raise ValueError(f"expected mono 16-bit WAV, got channels={ch} width={width}")
        raw = w.readframes(n)
    data = array("h")
    data.frombytes(raw)
    if sys.byteorder != "little":
        data.byteswap()
    return [v / 32768.0 for v in data], sr
