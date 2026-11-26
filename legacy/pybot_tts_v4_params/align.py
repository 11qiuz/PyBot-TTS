"""Phoneme-level boundary proposal for already-cut Mandarin word recordings.

这不是语音识别，是时长与谱变化的启发式分割 (not speech recognition; heuristic
segmentation from duration and spectral change).

Why this module exists
----------------------
``pybot_tts_v2_recorded/cuts.json`` marks only word-level spans. Parametric
synthesis needs phone-level spans, otherwise LPC formants cannot be attributed to
a concrete phoneme and every unit is fitted as one blurred average. So this module
does exactly one job: given the *known* syllable structure of a clip (onset /
final / tone per syllable, as returned by ``engine.split_pinyin``) propose phone
boundaries in seconds, and attach an honest confidence label to every proposal.

What this module does NOT do
----------------------------
* No text reading, no character recognition, no forced alignment against an
  acoustic model. The syllable list is an input, never an inference.
* No claim that a boundary is acoustically observable. Most boundaries inside a
  coarticulated word are not marked by a silence, a stop closure or any other event
  a 20 ms energy/periodicity frame can see. Then the boundary is reported as
  ``low`` with its reasons in ``Segment.notes`` instead of being dressed up as a
  detection. Downstream code must read a ``low`` edge as "position unknown".
* No digitised formant tracking. Until the LPC/vowel-target step lands, the
  in-final split uses only an energy valley plus low-band spectral change, which
  is why it is capped at ``medium`` no matter how clean it looks.

Pure stdlib by design: this machine ships no numpy and the project must stay
dependency-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

SR = 24000
HOP = 240          # 10 ms analysis hop
WIN = 480          # 20 ms spectral window (f0/F1/F2 band-energy tracking)
# Periodicity needs >= 2 * LAG_MAX samples, i.e. >= 640 at 24 kHz, and in practice
# several pitch periods to be meaningful. 20 ms is too short (that bug silently
# returned None for every frame); use a separate, longer pitch window.
PITCH_WIN = round(0.100 * SR)   # 2400 samples = v3 audit_edges' 100 ms probe

# Lag search for the periodicity probe, 75-370 Hz, as in
# pybot_tts_v3_seams/audit_edges.periodic().
LAG_MIN = round(SR / 370)
LAG_MAX = round(SR / 75)

SILENCE_DB = -48.0     # below this a frame is digital silence, not speech
QUIET_DB = -46.0       # generic "quiet" threshold when a clip has no reference
# How far below the clip's own reference a stretch must sit before it may be called
# a real stop closure rather than a merely quieter vowel. Measured on the extended
# corpus: at 14 dB the selected stretches were still periodic (median corr 0.83),
# i.e. not closures at all. 30 dB reliably selects true glottal silence.
SILENCE_MARGIN_DB = 30.0

STOP_ONSETS = frozenset({'b', 'p', 'd', 't', 'g', 'k', 'z', 'c', 'zh', 'ch',
                         'j', 'q'})
FRICATIVE_ONSETS = frozenset({'f', 's', 'sh', 'x', 'h'})
SONORANT_ONSETS = frozenset({'m', 'n', 'l', 'r'})

MIN_GAP = 0.030        # shorter than this a level dip is coarticulation, not a join
MIN_DIP_DB = 4.0       # a level dip must be this deep to count as an event

# Vowel targets per final: how many nuclei/glides/offglides live in the final.
# Used only to decide *how many* in-final cuts to propose, never *where*.
FINAL_TARGETS = {
    'a': 1, 'o': 1, 'e': 1, 'i': 1, 'u': 1, 'v': 1, 'er': 1,
    'ir': 1, 'iz': 1, 'ii': 1, 'ih': 1, 'i_zh': 1, 'i_z': 1,
    'ai': 2, 'ei': 2, 'ao': 2, 'ou': 2, 'ia': 2, 'ie': 2, 'ua': 2, 'uo': 2,
    'an': 2, 'en': 2, 'ang': 2, 'eng': 2, 'ong': 2, 'in': 2, 'ing': 2,
    'un': 2, 'vn': 2, 've': 2,
    'iao': 3, 'iou': 3, 'uai': 3, 'uei': 3, 'ian': 3, 'iang': 3, 'iong': 3,
    'uan': 3, 'uang': 3, 'uen': 3, 'ueng': 3, 'van': 3,
}

RANK = {'low': 0, 'medium': 1, 'high': 2}


# --------------------------------------------------------------------------- #
# signal primitives
# --------------------------------------------------------------------------- #
def rms(x):
    """Root mean square of a sample sequence (empty input -> 0.0)."""
    return math.sqrt(sum(v * v for v in x) / len(x)) if len(x) else 0.0


def db(v):
    """Amplitude ratio to dBFS, floored so silence stays finite."""
    return 20 * math.log10(max(1e-9, v))


def _hann(win):
    return [0.5 - 0.5 * math.cos(math.tau * k / win) for k in range(win)]


def frames(x, hop=HOP, win=WIN):
    """Split ``x`` into hop-spaced, Hann-windowed analysis frames.

    The final frame is tail-aligned to the end of ``x`` so the last ``win``
    samples are always analysed. A clip shorter than one window yields no frames
    (callers must handle that), and no frame ever starts at a negative index.
    """
    if hop <= 0 or win <= 0:
        raise ValueError('hop and win must be positive')
    n = len(x)
    if n < win:
        return []
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] != n - win:
        starts.append(n - win)
    w = _hann(win)
    return [[v * h for v, h in zip(x[b:b + win], w)] for b in starts]


def zcr(x):
    """Fraction of adjacent sample pairs that change sign."""
    if len(x) < 2:
        return 0.0
    return sum(1 for a, b in zip(x, x[1:]) if (a < 0) != (b < 0)) / (len(x) - 1)


def periodic(x):
    """Peak normalized autocorrelation in the 75-370 Hz lag band.

    Same acceptance rule as pybot_tts_v3_seams/audit_edges.periodic(): take the
    highest local peak, then the earliest peak within 0.055 of it, and accept at
    >= 0.60. Returns ``(period_samples, corr)``, or ``None`` when the frame is too
    short, too quiet (< .011 RMS) or not periodic enough.
    """
    if len(x) < 2 * LAG_MAX or rms(x) < .011:
        return None
    mean = sum(x) / len(x)
    z = [v - mean for v in x]
    best = {}
    for d in range(LAG_MIN, LAG_MAX + 1):
        b, c = z[:-d], z[d:]
        den = math.sqrt(sum(v * v for v in b) * sum(v * v for v in c))
        best[d] = sum(m * n for m, n in zip(b, c)) / max(1e-10, den)
    peaks = [d for d in range(LAG_MIN + 1, LAG_MAX)
             if best[d] >= max(best[d - 1], best[d + 1])]
    if not peaks:
        return None
    top = max(best[d] for d in peaks)
    if top < .60:
        return None
    lag = min(d for d in peaks if best[d] >= max(.60, top - .055))
    return lag, best[lag]


# Low-order band edges over the first 21 DFT bins. With WIN=480 at 24 kHz that is
# 0 / 200 / 500 / 1050 Hz: f0 and the first two formants. Content above ~1 kHz is
# deliberately ignored: it tracks fricative noise and breath, while vowel identity
# lives low.
BANDS = (0, 4, 10, 21)
NCOEF = 21


def _goertzel(frame, k):
    """Energy of DFT bin ``k`` of ``frame`` (Goertzel, O(n) per bin)."""
    n = len(frame)
    coef = 2.0 * math.cos(math.tau * k / n)
    s1 = s2 = 0.0
    for v in frame:
        s0 = v + coef * s1 - s2
        s2, s1 = s1, s0
    return s1 * s1 + s2 * s2 - coef * s1 * s2


def _coefs(frame):
    return [_goertzel(frame, k) for k in range(NCOEF)]


def bin_change(a, b, bands=BANDS):
    """Relative change of band energies from coefficient list ``a`` to ``b``.

    Returns a number in [0, 1]; 0.0 when either frame is (near) silent, because a
    silence transition is an energy event, not a spectral one.
    """
    ea, eb = sum(a), sum(b)
    if ea < 1e-12 or eb < 1e-12:
        return 0.0
    total = 0.0
    for lo, hi in zip(bands, bands[1:]):
        total += abs(sum(b[lo:hi]) / eb - sum(a[lo:hi]) / ea)
    return total / max(1.0, len(bands) - 1)


# --------------------------------------------------------------------------- #
# result type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Segment:
    """One proposed phone span.

    ``confidence`` describes **the boundary at ``end``** (the cut to the following
    segment), not the interior. ``start`` is inherited from the previous segment
    because the boundary set is one-dimensional; for the first segment ``start``
    is the clip start and is not a proposal at all.

    A ``low`` confidence means: this position is a fallback, not a detection. Do
    not define an LPC window from a low edge without checking, and do not present
    per-phoneme formants behind it as measured.
    """
    index: int
    syllable: int
    role: str                  # 'onset' | 'final' | 'target'
    phone: str
    tone: int
    start: float               # seconds
    end: float                 # seconds
    confidence: str            # 'high' | 'medium' | 'low'
    event: str                 # which acoustic event justified the end edge
    evidence: dict = field(default_factory=dict)
    notes: tuple = ()

    @property
    def duration(self):
        return self.end - self.start

    def as_dict(self):
        return dict(index=self.index, syllable=self.syllable, role=self.role,
                    phone=self.phone, tone=self.tone,
                    start=round(self.start, 4), end=round(self.end, 4),
                    duration=round(self.duration, 4), confidence=self.confidence,
                    event=self.event, evidence=self.evidence, notes=self.notes)


# --------------------------------------------------------------------------- #
# frame features
# --------------------------------------------------------------------------- #
def features(x, hop=HOP, win=WIN, pitch_win=PITCH_WIN):
    """Parallel per-frame feature lists for ``x``.

    Keys: ``starts`` (frame start sample), ``hop``, ``win``, ``pitch_win``,
    ``nframes``, ``samples``, ``energy`` (RMS of the 20 ms spectral frame),
    ``level`` (dBFS), ``corr``/``hz`` (periodicity measured on a separate
    ``pitch_win``-sample frame centred on the same hop position, ``None`` when
    unvoiced), ``change`` (low-band band-energy change against the previous frame,
    ``None`` on the first frame), ``coefs`` (low-band DFT coefficients) and
    ``loud`` (dB of the clip's own 90th-percentile frame, the reference level for
    every "quiet" decision).

    The pitch window is separate on purpose: an autocorrelation peak search over
    lags up to SR/75 needs at least 2*LAG_MAX = 640 samples, so a 20 ms window can
    never produce a period at all.
    """
    fr = frames(x, hop, win)
    starts = list(range(0, len(x) - win + 1, hop))
    if fr and starts and starts[-1] != len(x) - win:
        starts.append(len(x) - win)
    if not fr:
        starts = []
    energy = [rms(f) for f in fr]
    level = [db(e) for e in energy]
    coefs = [_coefs(f) for f in fr]
    corr, hz = [], []
    half = pitch_win // 2
    for b in starts:
        c = b + win // 2                       # centre of the spectral frame
        lo = max(0, min(len(x) - pitch_win, c - half))
        p = periodic(x[lo:lo + pitch_win]) if len(x) >= pitch_win else None
        corr.append(None if p is None else p[1])
        hz.append(None if p is None else SR / p[0])
    change = [None] + [bin_change(a, b) for a, b in zip(coefs, coefs[1:])]
    loud = sorted(level)[round(0.9 * (len(level) - 1))] if level else -90.0
    return dict(starts=starts, hop=hop, win=win, pitch_win=pitch_win,
                nframes=len(fr), samples=len(x), energy=energy, level=level,
                corr=corr, hz=hz, change=change, coefs=coefs, loud=loud)


def _idx(feat, t):
    """Frame index whose start is closest to instant ``t``."""
    return max(0, min(feat['nframes'] - 1, int(round(t * SR / feat['hop']))))


def _level_at(feat, t):
    return feat['level'][_idx(feat, t)] if feat['nframes'] else -90.0


# --------------------------------------------------------------------------- #
# level scans
# --------------------------------------------------------------------------- #
def _quiet_floor(feat):
    """A clip-relative 'quiet' threshold: 12 dB under the reference, floored."""
    return max(SILENCE_DB, min(QUIET_DB, feat['loud'] - 12.0))


def quiet_runs(feat):
    """Maximal runs of frames below the quiet floor, as (start_s, end_s)."""
    floor = _quiet_floor(feat)
    out, start = [], None
    for i in range(feat['nframes'] + 1):
        quiet = i < feat['nframes'] and feat['level'][i] <= floor
        if quiet and start is None:
            start = i
        elif not quiet and start is not None:
            out.append((start * feat['hop'] / SR,
                        min(feat['samples'], (i - 1) * feat['hop'] + feat['win']) / SR))
            start = None
    return [r for r in out if r[1] - r[0] > 0.005]


def voiced_runs(feat, corr_min=0.55, margin_db=18.0, min_len=0.020):
    """Runs of frames that are periodic and near the clip's peak level."""
    ceiling = feat['loud'] - margin_db
    out, start = [], None
    for i in range(feat['nframes'] + 1):
        ok = (i < feat['nframes'] and feat['corr'][i] is not None
              and feat['corr'][i] >= corr_min and feat['level'][i] >= ceiling)
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            a = start * feat['hop'] / SR
            b = min(feat['samples'], (i - 1) * feat['hop'] + feat['win']) / SR
            if b - a >= min_len:
                out.append((a, b))
            start = None
    return out


def _silence_dial(feat, lo, hi, corr_min=0.5, margin_db=SILENCE_MARGIN_DB,
                  min_span=0.012):
    """Densest near-silent stretch inside [lo,hi]: the classic onset->final event.

    Walks 1 ms steps and returns the last frame of the best window that is (a) at
    least ``margin_db`` under the clip's own reference level and (b) at least
    ``min_span`` seconds wide. Every frame of the window must be under the ceiling,
    and the window's peak must be clearly under the reference too.

    ``margin_db`` defaults to :data:`SILENCE_MARGIN_DB` (30 dB), not a token 14 dB,
    because a merely *quieter* vowel stretch is still voiced: on the real corpus a
    14 dB margin selected 77 stretches whose periodicity at the boundary was a
    median 0.83, i.e. plainly still phonating. Only a deep margin means the glottis
    actually stopped. ``corr`` is reported for diagnosis only -- the 100 ms pitch
    window overhangs a short closure into surrounding voicing, so periodicity cannot
    certify silence.

    Returns ``(time, evidence)`` with the measured numbers, or ``(None, reason)``.
    """
    n = feat['nframes']
    if n < 2 or hi <= lo:
        return None, {'reason': 'clip_too_short'}
    ceiling = feat['loud'] - margin_db
    peak_ceiling = feat['loud'] - 0.75 * margin_db
    best = None
    step = 0.001
    span = max(min_span, 2 * feat['hop'] / SR)
    t = lo
    while t + span <= hi + 1e-9:
        i, j = _idx(feat, t), _idx(feat, t + span)
        if j - i < 1:
            t += step
            continue
        seg = feat['level'][i:j + 1]
        lvl = sum(seg) / len(seg)
        peak = max(seg)
        cs = [c for c in feat['corr'][i:j + 1] if c is not None]
        cmean = sum(cs) / len(cs) if cs else 0.0
        # A stop closure is not "a quieter vowel": every frame in the window must sit
        # far under the clip reference. Periodicity is deliberately NOT used as a
        # gate here: the 100 ms pitch window overhangs a short closure into the
        # surrounding voicing, so `corr` stays high across a real stop and a
        # periodicity test would reject every genuine closure. It is still reported
        # in the evidence for diagnosis.
        if lvl <= ceiling and peak <= peak_ceiling and min(seg) <= ceiling:
            score = lvl
            if best is None or score < best[0]:
                # The onset->final boundary is where quiet ends and voicing
                # resumes, i.e. the window *end*, not its centre.
                best = (score, t + span, lvl, cmean, span, peak)
        t += step
    if best is None:
        return None, {'reason': 'no_quiet_nonperiodic_window',
                      'ceiling_db': round(ceiling, 1), 'corr_min': corr_min}
    # Walk forward through the closure and place the boundary at the last frame
    # that is still under the ceiling: that is the release instant.
    j = _idx(feat, best[1])
    while j + 1 < n and feat['level'][j + 1] <= ceiling:
        j += 1
    end = (j * feat['hop'] + feat['win']) / SR
    end = min(max(end, best[1]), hi)
    return end, {'level_db': round(best[2], 1), 'peak_db': round(best[5], 1),
                 'corr': round(best[3], 2), 'span_ms': round(best[4] * 1000, 1),
                 'release_at_s': round(end, 4)}


def _dip_dial(feat, lo, hi, min_depth=MIN_DIP_DB):
    """Shallowest-interior level dip inside [lo,hi] (a sonorant/vowel join).

    Returns ``(time, evidence)`` or ``(None, reason)``. A dip qualifies only when
    it is at least ``min_depth`` dB below the louder of the two windows sitting
    60 ms either side of it, so a monotonically falling level never qualifies.
    """
    n = feat['nframes']
    if n < 2 or hi <= lo:
        return None, {'reason': 'span_too_short'}
    reach = round(0.060 * SR / feat['hop'])
    best = None
    for i in range(max(1, _idx(feat, lo)), min(n - 1, _idx(feat, hi)) + 1):
        left = feat['level'][max(0, i - reach):i]
        right = feat['level'][i + 1:i + 1 + reach]
        if not left or not right:
            continue
        flank = max(max(left), max(right))
        depth = flank - feat['level'][i]
        if depth >= min_depth and (best is None or depth > best[0]):
            best = (depth, i * feat['hop'] / SR, flank, feat['level'][i])
    if best is None:
        return None, {'reason': 'no_dip', 'min_depth_db': min_depth}
    return best[1], {'dip_db': round(best[0], 1), 'flank_db': round(best[2], 1),
                     'level_db': round(best[3], 1)}


def _change_dial(feat, lo, hi, min_change=0.030):
    """Point of largest low-band spectral change inside [lo,hi]."""
    n = feat['nframes']
    if n < 2 or hi <= lo:
        return None, {'reason': 'span_too_short'}
    best = None
    for i in range(max(1, _idx(feat, lo)), min(n - 1, _idx(feat, hi)) + 1):
        c = feat['change'][i]
        if c is not None and (best is None or c > best[0]):
            best = (c, i * feat['hop'] / SR)
    if best is None or best[0] < min_change:
        return None, {'reason': 'no_change', 'min_change': min_change,
                      'seen': None if best is None else round(best[0], 4)}
    return best[1], {'change': round(best[0], 4)}


# --------------------------------------------------------------------------- #
# layout: syllable slot boundaries and the onset/final knot positions
# --------------------------------------------------------------------------- #
def _nucleus_knots(n_targets, sonorant_onset):
    """Knot positions for in-final cuts, as fractions of the (onset-adjusted) span.

    A sonorant onset (m/n/l/r) fills a large share of the front of the syllable,
    so the nucleus has to be pulled back before dipping into the front vowel. For
    the rest the first knot anchors the *start* of the final (onset release ->
    vowel onset) and the remaining knots cut target->target.
    """
    if n_targets <= 1:
        return ([], [0.50] if sonorant_onset else [0.28])
    cuts = [i / n_targets for i in range(1, n_targets)]
    if sonorant_onset:
        return (cuts, [0.45])
    return (cuts, [0.34])


def _syllable_slots(feat, structure):
    """Speech span plus one join time per syllable boundary.

    Returns ``(speech_start, speech_end, joins)`` in seconds where ``joins`` has
    ``len(structure)-1`` entries of ``(time, confidence, evidence, notes)``.
    Joins are searched only in a provably quiet stretch: genuine inter-syllable
    occlusion is the one case where a join is acoustically visible at all.
    """
    n_s = feat['samples']
    duration = n_s / SR
    k = len(structure) - 1
    joins = []
    # "Quiet" for layout purposes means a genuine phonation break (silence, not a
    # -40 dB level dip), and a join must be wide enough to really separate two
    # syllables. A 40 ms level dip in the middle of a vowel is coarticulation.
    min_join = max(MIN_GAP, 0.045 * duration, 0.055)
    quiet = [r for r in quiet_runs(feat) if r[1] - r[0] >= min_join]
    # The first quiet stretch only counts as a lead-in if real speech follows it;
    # an initial silence that is itself the word's own stop closure must stay
    # inside the syllable span.
    lead, tail = 0.0, duration
    if quiet and quiet[0][0] <= 0.004 * duration:
        lead = quiet[0][1]
    if quiet and quiet[-1][1] >= 0.996 * duration:
        tail = quiet[-1][0]
    if tail - lead < 0.020:
        # No discernible speech at all (all-silence clip): keep the whole span so
        # the layout stays non-degenerate instead of inverting lead/tail.
        lead, tail = 0.0, duration
        quiet = []
    interior = [r for r in quiet if r[0] >= lead - 1e-9 and r[1] <= tail + 1e-9
                and r[0] > lead + 0.002 and r[1] < tail - 0.002]
    if k <= 0:
        # One syllable: the audio between lead-in and release is the syllable, and
        # any quiet stretch inside it is part of that syllable (typically the stop
        # closure being searched for later), never a word boundary.
        return lead, tail, joins
    if not interior or k != 1 or len(interior) != 1:
        # No single unambiguous break: distribute by syllable count.
        for i in range(1, k + 1):
            t = lead + (tail - lead) * i / (k + 1)
            joins.append((t, 'low',
                          {'quiet_candidates': len(interior),
                           'min_join_ms': round(min_join * 1000, 1),
                           'level_db': round(_level_at(feat, t), 1)},
                          ('no phonation break between syllables: join placed by '
                           'syllable-count proportion, not detected',)))
        return lead, tail, joins
    a, b = interior[0]
    # A real join must leave a plausible syllable on each side.
    min_syllable = max(0.060, 0.12 * duration)
    if a - lead < min_syllable or tail - b < min_syllable:
        for i in range(1, k + 1):
            t = lead + (tail - lead) * i / (k + 1)
            joins.append((t, 'low',
                          {'quiet_ms': round((b - a) * 1000, 1),
                           'level_db': round(_level_at(feat, t), 1)},
                          ('the only quiet stretch would starve one syllable, so the '
                           'join is placed by syllable-count proportion',)))
        return lead, tail, joins
    t = (a + b) / 2
    joins.append((t, 'high',
                  {'quiet_ms': round((b - a) * 1000, 1),
                   'level_db': round(_level_at(feat, t), 1),
                   'min_join_ms': round(min_join * 1000, 1)},
                  ()))
    return lead, tail, joins


def _onset_knot(feat, a, b, kind):
    """The onset->final change inside syllable span [a,b].

    Returns ``(time, confidence, event, evidence, notes)``. Strategy order is
    silence -> voicing onset -> duration proportion, and the confidence handed
    back is the confidence of whichever strategy actually fired.
    """
    span = b - a
    # Onsets are short by nature. Searching the whole syllable would let a silence
    # that actually belongs to the *next* onset (or to the clip's tail) be claimed
    # as this syllable's vowel onset. Cap the search at a generous onset budget and
    # always keep a minimum final length.
    budget = _onset_budget(kind)
    hi = min(a + budget, b - max(0.020, 0.20 * span))
    lo = a + max(0.006, 0.06 * span)
    # A clip cut from a longer recording starts with its own lead-in silence. That
    # silence is NOT a stop closure, so an event found inside it must never be
    # reported as the onset->final boundary. Compute the clip's leading silence and
    # require the boundary to sit clear of it.
    lead = _leading_silence(feat)
    if lead > 0.0:
        lo = max(lo, lead + 0.5 * feat['hop'] / SR)
    if hi <= lo:
        t = a + min(0.35 * span, 0.6 * budget)
        return (t, 'low', 'proportion',
                {'span_ms': round(span * 1000, 1),
                 'lead_silence_ms': round(lead * 1000, 1),
                 'onset_budget_ms': round(budget * 1000, 1)},
                ('syllable too short to search an onset->final change',))
    if kind in ('stop', 'fricative'):
        t, ev = _silence_dial(feat, lo, hi)
        if t is not None:
            ev = dict(ev, onset_budget_ms=round(budget * 1000, 1),
                      lead_silence_ms=round(lead * 1000, 1))
            return (t, 'high', 'silence', ev, ())
        t2, ev2 = _voicing_start(feat, lo, hi)
        if t2 is not None:
            return (t2, 'medium', 'voicing_onset',
                    dict(ev2, onset_budget_ms=round(budget * 1000, 1),
                         lead_silence_ms=round(lead * 1000, 1)), ())
        t = a + min(0.40 * span, 0.65 * budget)
        return (t, 'low', 'proportion', ev,
                ('obstruent onset present but neither its closure/silence nor a '
                 'voicing rise was found; cut is by duration only',))
    if kind == 'sonorant':
        t, ev = _dip_dial(feat, lo, hi)
        if t is not None:
            return (t, 'low', 'level_dip', ev,
                    ('sonorant onset never stops phonation; a level dip is a weak '
                     'hint at the nasal/liquid -> vowel change, not a reliable one',))
        t = a + min(0.42 * span, 0.7 * budget)
        return (t, 'low', 'proportion', ev,
                ('sonorant onset has no phonation break and no level dip; cut by '
                 'duration only',))
    # zero onset: no consonantal event exists at all.
    t = a + 0.10 * span
    return (t, 'low', 'proportion', {'span_ms': round(span * 1000, 1)},
            ('zero onset: the syllable begins directly with voiced vowel, so there '
             'is no onset segment to find and no boundary to detect',))


def _onset_budget(kind):
    """Upper bound on onset duration for each onset class, in seconds.

    These are timing facts about Mandarin articulation (stop closures and
    fricatives are short), not fitted constants of this corpus.
    """
    return {'stop': 0.140, 'fricative': 0.200, 'sonorant': 0.140}.get(kind, 0.140)


def _leading_silence(feat):
    """Length in seconds of the near-silent lead-in at the start of the clip.

    Starts at the first frame index 0 and stops at the first frame that is neither
    below the quiet floor nor clearly quieter than the clip reference. This is the
    recorder's own dead air or the trim margin, never a stop closure, so anything
    found inside it must not be sold as an onset->final boundary.
    """
    if not feat['nframes']:
        return 0.0
    floor = _quiet_floor(feat)
    n = 0
    for i, lv in enumerate(feat['level']):
        if lv <= floor:
            n = i + 1
        else:
            break
    return min(feat['samples'], n * feat['hop'] + feat['win']) / SR if n else 0.0


def _voicing_start(feat, lo, hi, corr_min=0.45, after_min=0.60, lookahead=2):
    """First frame in [lo,hi] where periodicity rises from unvoiced into voicing.

    The 'before' frame is looked up on the whole-clip feature list, so a rise that
    starts exactly at the window edge is still judged against what preceded it.
    """
    start = _idx(feat, lo)
    stop = min(feat['nframes'] - 1, _idx(feat, hi))
    for i in range(start, stop + 1):
        here = feat['corr'][i]
        j = min(i + lookahead, feat['nframes'] - 1)
        after = feat['corr'][j]
        before = feat['corr'][i - 1] if i > 0 else None
        if here is None or after is None:
            continue
        if here >= corr_min and after >= after_min and (before is None or before < corr_min):
            return i * feat['hop'] / SR, {'corr': round(here, 2),
                                          'corr_after': round(after, 2),
                                          'corr_before': None if before is None
                                          else round(before, 2)}
    return None, {'reason': 'no_voicing_rise'}


# --------------------------------------------------------------------------- #
# per-syllable segmentation
# --------------------------------------------------------------------------- #
def _onset_kind(onset):
    if not onset:
        return 'zero'
    if onset in STOP_ONSETS:
        return 'stop'
    if onset in FRICATIVE_ONSETS:
        return 'fricative'
    if onset in SONORANT_ONSETS:
        return 'sonorant'
    return 'sonorant'      # unknown latin onset: treat as continuation, be conservative


def syllable_segments(x, structure, feat=None):
    """Segment one syllable span of ``x`` into onset/final and in-final targets.

    ``structure`` is ``(onset, final, tone)`` exactly as ``engine.split_pinyin``
    returns. Returns a list of dicts with ``role``/``phone``/``start``/``end``/
    ``confidence``/``event``/``evidence``/``notes`` plus a trailing entry whose
    ``role == 'span'`` describes the whole syllable and how its extent was set.
    """
    if feat is None:
        feat = features(x)
    a, b = 0.0, len(x) / SR
    onset, final, tone = structure[0], structure[1], int(structure[2])
    kind = _onset_kind(onset)
    out = []
    if kind == 'zero':
        # No onset segment at all: the syllable starts on a voiced vowel. We keep
        # one honest placeholder so the clip is still tiled, but it is labelled
        # 'onset' with zero acoustic claim rather than pretending a boundary exists.
        out.append(dict(role='onset', phone='', tone=tone, start=a, end=a,
                        confidence='low', event='none', evidence={},
                        notes=('zero onset: no consonant, so there is no onset/final '
                               'boundary to detect; the final starts at the syllable '
                               'start',)))
        out.append(dict(role='final', phone=final, tone=tone, start=a, end=b,
                        confidence='low', event='inherited', evidence={},
                        notes=('zero-onset final owns the whole syllable span',)))
        start_final = a
    else:
        t, conf, event, ev, notes = _onset_knot(feat, a, b, kind)
        out.append(dict(role='onset', phone=onset, tone=tone, start=a, end=t,
                        confidence=conf, event=event, evidence=ev, notes=notes))
        out.append(dict(role='final', phone=final, tone=tone, start=t, end=b,
                        confidence='low', event='inherited',
                        evidence={},
                        notes=('final start inherits the onset edge; its own '
                               'in-final cuts are listed separately',)))
        start_final = t
    # in-final targets
    targets = FINAL_TARGETS.get(final, 1)
    cuts, anchor = _nucleus_knots(targets, kind == 'sonorant')
    if targets > 1:
        lo = start_final + (b - start_final) * anchor[0]
        hi = b - max(0.010, 0.06 * (b - start_final))
        edges = [start_final]
        for j, frac in enumerate(cuts):
            centre = start_final + (b - start_final) * frac
            half = max(0.012, 0.10 * (b - start_final))
            centre = max(lo, min(hi, centre))
            t, ev = _dip_dial(feat, max(lo, centre - half), min(hi, centre + half))
            conf, event = 'low', 'level_dip'
            notes = ()
            if t is None:
                t2, ev2 = _change_dial(feat, max(lo, centre - half),
                                       min(hi, centre + half))
                if t2 is not None:
                    t, ev, event = t2, ev2, 'spectral_change'
                    conf = 'medium'
                else:
                    t = centre
                    ev = dict(ev, **ev2)
                    event = 'proportion'
                    notes = ('neither a level dip nor a low-band spectral change '
                             'was found; cut is by duration only',)
            else:
                ev = dict(ev)
                t2, ev2 = _change_dial(feat, max(lo, t - 0.020),
                                       min(hi, t + 0.020))
                if t2 is not None:
                    ev['change_nearby'] = ev2['change']
            edges.append(t)
            out.append(dict(role=f'target{j + 1}', phone=final, tone=tone,
                            start=edges[-2], end=t, confidence=conf,
                            event=event, evidence=ev, notes=notes))
        out.append(dict(role=f'target{len(cuts) + 1}', phone=final, tone=tone,
                        start=edges[-1], end=b, confidence='low',
                        event='inherited', evidence={},
                        notes=('last final target runs to the end of the '
                               'syllable span; no further boundary',)))
    out.append(dict(role='span', phone=f'{onset or ""}{final}', tone=tone,
                    start=a, end=b, confidence='-', event='layout',
                    evidence={'n_targets': targets, 'onset_kind': kind},
                    notes=()))
    return out


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
def _structure(item):
    """Accept ``(onset, final, tone)`` tuples or dicts with those keys."""
    if isinstance(item, dict):
        return (str(item['onset']), str(item['final']), int(item['tone']))
    onset, final, tone = item
    return (str(onset), str(final), int(tone))


def refine_syllable(x, onset, final, tone, *, start=0.0, stop=None, hop=HOP,
                    win=WIN, feat=None):
    """Refine one *already-bounded* syllable into onset/final and intra-final cuts.

    This is the two-stage entry point. Stage one (syllable segmentation by energy
    dips) is somebody else's job: the caller has a window it believes holds exactly
    one syllable. Stage two -- this function -- only has to answer "where inside
    this window does the onset end and the final begin, and where do the final's
    own targets meet".

    Working inside a known syllable is much easier than searching a whole word: the
    span is short, it contains exactly one syllable, and the clip's lead-in silence
    is not in the search window unless the caller says so.

    Parameters
    ----------
    x : sequence of float
        Normalized audio, 24 kHz mono. May be the whole utterance; ``start`` and
        ``stop`` select the syllable window inside it.
    onset, final, tone : str, str, int
        The syllable, typically from ``engine.split_pinyin``.
    start, stop : float, optional
        Syllable window in seconds of ``x``. ``stop=None`` means end of ``x``.
    feat : dict, optional
        Pre-computed whole-clip features (avoids re-analysing per syllable).

    Returns
    -------
    dict with keys ``onset``, ``final``, ``targets`` (lists of :class:`Segment`),
    ``boundary`` (the onset->final time in seconds, or ``None`` if the syllable has
    no onset), ``boundary_confidence``, ``boundary_event``, ``evidence`` and
    ``notes``. Times are absolute seconds in ``x``, not window-relative.

    Raises ``ValueError`` for an unknown/invalid window or a window too short to
    analyse.
    """
    stop = len(x) / SR if stop is None else stop
    if stop <= start:
        raise ValueError(f'syllable window is empty: start={start}, stop={stop}')
    if stop * SR - start * SR < win + hop:
        raise ValueError(
            f'syllable window too short to analyse: {(stop - start) * 1000:.1f} ms '
            f'needs more than one {win}-sample window plus a hop')
    onset, final, tone = str(onset), str(final), int(tone)
    if feat is None or (feat.get('samples', 0) < len(x)):
        feat = features(x, hop, win)
    span = stop - start
    kind = _onset_kind(onset)
    kn = _onset_knot(feat, start, stop, kind)
    t, conf, event, ev = kn[0], kn[1], kn[2], kn[3]
    notes = kn[4]
    targets = []
    if _structure((onset, final, tone)) and final:
        rows = syllable_segments(x[int(round(start * SR)):int(round(stop * SR))],
                                 (onset, final, tone),
                                 feat=_slice_features(feat, start, stop))
        for row in rows[:-1]:                       # drop the 'span' bookkeeping row
            targets.append(Segment(
                index=len(targets), syllable=0, role=row['role'], phone=row['phone'],
                tone=row['tone'], start=start + row['start'], end=start + row['end'],
                confidence=row['confidence'], event=row['event'],
                evidence=dict(row['evidence']), notes=tuple(row['notes'])))
    has_onset = bool(onset) and kind != 'zero'
    return dict(onset=onset, final=final, tone=tone, start=start, stop=stop,
                span_ms=round(span * 1000, 1), onset_kind=kind,
                boundary=t if has_onset else None,
                boundary_confidence=conf if has_onset else 'low',
                boundary_event=event if has_onset else 'none',
                evidence=dict(ev), notes=tuple(notes), segments=targets)


def segment_word(x, syllable_structure, hop=HOP, win=WIN):
    """Propose phoneme-level segments for one already-cut word recording.

    ``x`` is normalized audio (floats in [-1, 1], 24 kHz mono) holding *only* this
    word; ``syllable_structure`` is the per-syllable list from
    ``engine.split_pinyin`` (tuples or dicts with onset/final/tone).

    Returns a list of :class:`Segment`. Segments tile the clip: consecutive
    ``end``/``start`` values are equal, the first ``start`` is 0.0 and the last
    ``end`` is the clip duration. Each segment's ``confidence`` describes the cut
    at its ``end`` -- read it as "how sure am I that a phone boundary sits here".

    Raises ``ValueError`` on an empty syllable list, malformed structure, or a clip
    too short to analyse (< one 20 ms window); no silent placeholder output.
    """
    structure = [_structure(s) for s in syllable_structure]
    if not structure:
        raise ValueError('syllable_structure must name at least one syllable')
    duration = len(x) / SR
    feat = features(x, hop, win)
    if feat['nframes'] < 2:
        raise ValueError(
            f'clip too short to analyse: {len(x)} samples ({duration:.4f}s) but '
            f'a {win}-sample window plus one hop is needed')
    speech_start, speech_end, joins = _syllable_slots(feat, structure)

    # syllable spans from the join points
    edges = [speech_start] + [j[0] for j in joins] + [speech_end]
    spans = [(edges[i], edges[i + 1]) for i in range(len(structure))]
    for i, (a, b) in enumerate(spans):
        if b - a <= 0.0:
            raise ValueError(
                f'syllable {i} span collapsed to {b - a:.4f}s; the join layout is '
                f'degenerate for {duration:.4f}s of audio and {len(structure)} '
                f'syllables')

    segs = []
    for i, ((a, b), struct) in enumerate(zip(spans, structure)):
        si, ei = int(round(a * SR)), int(round(b * SR))
        rows = syllable_segments(x[si:ei], struct, feat=_slice_features(feat, a, b))
        for row in rows[:-1]:                       # drop the 'span' bookkeeping row
            segs.append(Segment(
                index=len(segs), syllable=i, role=row['role'], phone=row['phone'],
                tone=row['tone'], start=a + row['start'], end=a + row['end'],
                confidence=row['confidence'], event=row['event'],
                evidence=dict(row['evidence']), notes=tuple(row['notes'])))
        if i < len(joins):                          # the cut into the next syllable
            t, conf, ev, notes = joins[i]
            segs.append(Segment(
                index=len(segs), syllable=i, role='join', phone='|', tone=0,
                start=t, end=t, confidence=conf, event='syllable_join',
                evidence=dict(ev), notes=tuple(notes)))
    # Clip the tiling to the exact clip extent: the first segment owns [0, a) as
    # unvoiced lead-in and the last owns [b, duration) as release, so no audio is
    # left unattributed.
    segs[0] = _replace(segs[0], start=0.0)
    segs[-1] = _replace(segs[-1], end=duration)
    return segs


def _replace(seg, **kw):
    data = dict(index=seg.index, syllable=seg.syllable, role=seg.role,
                phone=seg.phone, tone=seg.tone, start=seg.start, end=seg.end,
                confidence=seg.confidence, event=seg.event,
                evidence=seg.evidence, notes=seg.notes)
    data.update(kw)
    return Segment(**data)


def _slice_features(feat, a, b):
    """Re-index a whole-clip feature dict onto the window [a, b] (seconds).

    Frames are reused as-is (no re-analysis): only the time origin moves, so a
    syllable's own searches cannot reach outside its span.
    """
    i0 = max(0, int(round(a * SR / feat['hop'])))
    i1 = min(feat['nframes'], int(round(b * SR / feat['hop'])))
    i1 = max(i0 + 1, i1)
    # Absolute sample length of the slice. Frames are only ~nframes*hop samples
    # long, but the syllable's real extent is b-a, and cut searches must be allowed
    # to reach the true end of the span (the last 20 ms window overhangs it).
    samples = max(feat['win'], int(round((b - a) * SR)))
    keys = ('level', 'energy', 'corr', 'hz', 'change', 'coefs')
    sub = {k: list(feat[k][i0:i1]) for k in keys}
    sub.update(hop=feat['hop'], win=feat['win'], nframes=i1 - i0,
               samples=samples, loud=feat['loud'],
               starts=list(range(i0 * feat['hop'], i1 * feat['hop'], feat['hop'])))
    return sub


def boundaries(segments):
    """Collapse a segment list to explicit cuts.

    Each entry is ``(time_s, confidence, left_label, right_label)``. Every segment
    whose ``end`` is a genuine interior cut contributes one entry; ``'join'``
    markers contribute their own (the syllable-to-syllable cut); the clip's final
    edge is not reported because it is not a detection.
    """
    out = []
    for i, seg in enumerate(segments):
        if seg.role == 'join':
            out.append((seg.start, seg.confidence, 'syllable', 'syllable'))
            continue
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        if nxt is None:
            continue                     # clip end: not a detected boundary
        label_b = 'join' if nxt.role == 'join' else nxt.phone
        out.append((seg.end, seg.confidence, seg.phone, label_b))
    return out


def confidence_histogram(segments, roles=None):
    """Count ``high``/``medium``/``low`` over the interior cuts of a segment list.

    The final segment's edge is the clip end and is not counted; ``'join'``
    markers are counted because they are cuts the caller asked for.
    """
    hist = {'high': 0, 'medium': 0, 'low': 0}
    for i, seg in enumerate(segments):
        if seg.role != 'join' and i + 1 >= len(segments):
            continue
        if roles is not None and seg.role not in roles:
            continue
        hist[seg.confidence] = hist.get(seg.confidence, 0) + 1
    return hist
