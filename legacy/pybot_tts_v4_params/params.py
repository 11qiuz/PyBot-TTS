"""Phoneme parameter extraction from the recorded corpus, pure stdlib.

Reads syllable-aligned records from :mod:`pybot_tts_v4_params.corpus` (the
extended 32-utterance corpus plus the nine shipped words), runs
:mod:`pybot_tts_v4_params.lpc` on the voiced ones, and writes
``phoneme_params.json``.

Sample budget and why confidence is per-final
---------------------------------------------
By ``(onset, final, tone)`` triple almost every combination in this corpus has a
single sample, so a per-triple estimate would be a fiction. By **final**, the
core vowels are well covered while ``ia``/``o``/``ong``/``uai``/``uang``/``er``
have 1-2. Confidence is therefore keyed on the per-final sample count:

    n >= 5   -> "high"    (usable as a real parameter)
    n 3..4   -> "medium"  (usable, expect tens of Hz of error)
    n <  3   -> "low"     (evidence only; excluded from speaker estimates)

Fused finals (``ia`` = i + a, ``uo`` = u + o, ...) are deliberately *not*
collapsed to their nucleus: the measured F1 of a fused final differs from the
nucleus, so mixing them would corrupt both.

What this module refuses to do
------------------------------
* a per-speaker ``formant_scale`` is only emitted when the speaker has
  >= ``MIN_SPEAKER_TARGETS`` finals at ``medium``/``high`` confidence; otherwise
  it is ``null`` with a reason -- never invented;
* per-sample values are kept in ``provenance`` (utterance, syllable position,
  tone) instead of being averaged into one number that hides position drift,
  because the extended corpus is whole sentences with sentence-internal
  pitch/rate variation;
* unvoiced onsets (fricatives, aspirated stops) have no formants; only duration
  and a noise spectral centroid are recorded;
* syllable bounds at ``low`` confidence lower the final's own confidence. A
  formant averaged across a shaky cut is not a measurement of that final.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from . import corpus, lpc

SR = lpc.SR
ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "phoneme_params.json"

SCHEMA = 2

# Per-final sample-count thresholds. These are the contract, quoted in the JSON.
CONF_HIGH_MIN = 5
CONF_MEDIUM_MIN = 3
# A speaker needs this many medium/high finals before formant_scale is emitted.
MIN_SPEAKER_TARGETS = 3

# How far one sample's F1 may sit from its final's dense cluster before it is
# treated as a mis-cut. Measured: correct samples of one final agree on F1 to
# ~80 Hz while a cut that caught the neighbouring vowel lands 500+ Hz away, so
# this separates cleanly. Lives here (not in lpc.py) because it is a corpus
# property, not an analysis parameter.
CUT_OUTLIER_HZ = 260.0

# Mandarin phoneme classes. Voiced onsets can carry formants; the rest cannot.
VOICED_ONSETS = {"m", "n", "l", "r"}
UNVOICED_ONSETS = {
    "b", "p", "d", "t", "g", "k", "f", "s", "sh", "x", "h",
    "z", "c", "zh", "ch", "j", "q",
}
# Every final is voiced in Mandarin (nasal codas included).
VOICED_FINALS = True

# Finals whose medial is a glide onto a nucleus; their measured formants differ
# from the bare nucleus so they stay their own label.
FUSED_FINALS = {
    "ia", "ie", "iao", "iu", "iou", "ian", "in", "iang", "ing", "iong",
    "ua", "uo", "uai", "ui", "uei", "uan", "un", "uen", "uang", "ueng",
    "ve", "van", "vn", "ue",
}


# --------------------------------------------------------------------------
# pinyin
# --------------------------------------------------------------------------
def split_pinyin(pinyin):
    """Re-export of the project's own splitter, imported lazily.

    Imported from ``pybot_tts_v2_recorded.engine`` so the phoneme inventory can
    never drift from what the synthesiser actually renders.
    """
    import sys

    v2 = sys.modules.get("pybot_tts_v2_recorded.engine")
    if v2 is not None:
        return v2.split_pinyin(pinyin)
    sys.path.insert(0, str(ROOT))
    from pybot_tts_v2_recorded.engine import split_pinyin as _sp

    return _sp(pinyin)


def load_records(sources=("extended", "shipped")):
    """Syllable records from :mod:`corpus`, newest/most-trustworthy first."""
    rows = []
    if "extended" in sources:
        rows.extend(corpus.load_extended())
    if "shipped" in sources:
        rows.extend(corpus.load_shipped())
    if not rows:
        raise ValueError("corpus returned no syllable records")
    return rows


# --------------------------------------------------------------------------
# per-record measurement
# --------------------------------------------------------------------------
def is_voiced(phone, role):
    """True when a phoneme type is expected to carry formants."""
    if role == "final":
        return VOICED_FINALS
    return phone in VOICED_ONSETS


def measure_syllable(x, start, stop, pinyin, sr=SR):
    """Split one syllable into onset+final spans and measure each.

    Returns ``[(record)]``. The onset/final cut inside a syllable is not known
    from the corpus, so it is derived from the syllable's own energy profile:
    the boundary is the quietest frame in the middle 40-70 % of the syllable,
    which is where a Mandarin onset->final transition actually sits. When that
    search is not decisive the split falls back to 45 % of the syllable and is
    labelled ``low``.
    """
    onset, final, tone = split_pinyin(pinyin)
    start = max(0, int(start))
    stop = max(start + 32, int(stop))
    seg = x[start:stop]
    n = len(seg)
    spans = _split_syllable(seg, onset)
    out = []
    for role, a, b, conf in spans:
        sub = seg[a:b]
        phone = final if role == "final" else onset
        rec = {
            "phone": phone,
            "role": role,
            "final": final,
            "onset": onset,
            "start": start + a,
            "end": start + b,
            "duration_ms": round(1000.0 * (b - a) / sr, 1),
            "tone": tone,
            "split_confidence": conf,
            "rms": round(lpc.rms(sub), 5),
            "voiced_kind": is_voiced(phone, role),
        }
        if not rec["voiced_kind"]:
            rec["formants"] = None
            rec["bandwidths"] = None
            rec["f0_hz"] = None
            rec["noise_centroid_hz"] = _round_or_none(noise_centroid(sub, sr), 1)
            out.append(rec)
            continue
        res = lpc.analyze_segment(sub, sr=sr)
        if res is None:
            rec["formants"] = None
            rec["bandwidths"] = None
            rec["f0_hz"] = None
            rec["reject"] = "unvoiced_or_implausible"
        else:
            rec["formants"] = res["f"]
            rec["bandwidths"] = res["bw"]
            rec["f0_hz"] = res["f0"]
            rec["voiced_ratio"] = res["voiced_ratio"]
            rec["used_frames"] = res["used_frames"]
            rec["frames"] = res["frames"]
        out.append(rec)
    return out


def _split_syllable(seg, onset):
    """Where does the onset end inside one syllable?

    Returns ``[(role, a, b, confidence)]``. With no onset the whole syllable is
    the final. Otherwise the cut is the quietest 20 ms frame in the middle band
    of the syllable, which is where a Mandarin onset->final transition sits.
    """
    n = len(seg)
    if not onset:
        return [("final", 0, n, "high")]
    lo = int(n * 0.20)
    hi = int(n * 0.75)
    hop = max(1, SR // 100)                       # 10 ms
    win = max(hop * 2, min(n // 4, SR * 20 // 1000))
    if hi - lo < win:
        cut = int(n * 0.45)
        return [("onset", 0, cut, "low"), ("final", cut, n, "low")]
    levels = []
    pos = lo
    while pos + win <= hi:
        levels.append((lpc.rms(seg[pos:pos + win]), pos + win // 2))
        pos += hop
    if not levels:
        cut = int(n * 0.45)
        return [("onset", 0, cut, "low"), ("final", cut, n, "low")]
    low_energy, cut = min(levels)
    peak = max(r for r, _ in levels)
    # A decisive dip drops well under the local peak; otherwise it is a guess.
    decisive = peak > 0.0 and low_energy <= 0.55 * peak
    cut = max(hop, min(n - hop, cut))
    conf = "high" if decisive else "low"
    return [("onset", 0, cut, conf), ("final", cut, n, conf)]


def noise_centroid(seg, sr=SR):
    """Spectral centroid (Hz) of a short unvoiced segment via a DFT sweep.

    Coarse on purpose: 50 Hz steps over 200-6000 Hz. Returns ``None`` when the
    segment is too quiet or too short to say anything.
    """
    n = len(seg)
    if n < 256 or lpc.rms(seg) < 0.004:
        return None
    x = [float(v) for v in seg]
    mean = sum(x) / len(x)
    x = [v - mean for v in x]
    x = [v * w for v, w in zip(x, lpc.hamming(len(x)))]
    num = 0.0
    den = 0.0
    for f in range(200, 6000, 50):
        w = math.tau * f / sr
        re = im = 0.0
        for i, v in enumerate(x):
            re += v * math.cos(w * i)
            im -= v * math.sin(w * i)
        mag = math.hypot(re, im)
        num += f * mag
        den += mag
    return num / den if den > 0.0 else None


def _round_or_none(value, digits):
    return None if value is None else round(value, digits)


# --------------------------------------------------------------------------
# aggregation helpers
# --------------------------------------------------------------------------
def _median(values):
    return lpc.median(values)


def _std(values):
    n = len(values)
    if n < 2:
        return 0.0
    mu = sum(values) / n
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))


def confidence_of(n, split_confs, bound_confs):
    """Confidence for one final from sample count and boundary quality.

    The count decides the tier; a majority of ``low`` boundaries demotes one
    tier, because a formant smeared across a bad cut is not a clean sample.
    """
    if n >= CONF_HIGH_MIN:
        tier = 2
    elif n >= CONF_MEDIUM_MIN:
        tier = 1
    else:
        tier = 0
    bad = sum(1 for c in list(split_confs) + list(bound_confs) if c == "low")
    total = len(split_confs) + len(bound_confs)
    if total and bad * 2 > total and tier > 0:
        tier -= 1
    return ("low", "medium", "high")[tier]


def collect(sources=("extended", "shipped")):
    """Measure every syllable record. Returns ``(measurements, meta)``."""
    records = load_records(sources)
    cache = {}
    measurements = []
    meta = {"records": [], "n_syllables": 0, "by_source": {}}

    for row in records:
        wav = row["wav"]
        if wav not in cache:
            samples, sr = lpc.wav_read(wav)
            if sr != SR:
                raise ValueError(f"{wav}: expected {SR} Hz, got {sr}")
            cache[wav] = samples
        x = cache[wav]
        start = int(row["start_sample"])
        stop = int(row["stop_sample"])
        stop = min(stop, len(x))
        if stop - start < 64:
            continue
        pieces = measure_syllable(x, start, stop, row["pinyin"])
        for rec in pieces:
            rec["speaker"] = row["speaker"]
            rec["utterance"] = row["utterance"]
            rec["hanzi"] = row.get("hanzi")
            rec["pinyin"] = row["pinyin"]
            rec["source"] = row["source"]
            rec["bound_confidence"] = row["bound_confidence"]
            measurements.append(rec)
        meta["n_syllables"] += 1
        meta["by_source"][row["source"]] = meta["by_source"].get(row["source"], 0) + 1

    meta["records"] = sorted({m["utterance"] for m in measurements})
    return measurements, meta


def aggregate(measurements):
    """Group measurements into ``phonemes`` and ``speakers``."""
    finals = {}
    onsets = {}
    for m in measurements:
        if m["role"] == "final":
            finals.setdefault(m["final"], []).append(m)
        else:
            onsets.setdefault(m["phone"], []).append(m)

    phonemes = {}
    for phone, rows in sorted(finals.items()):
        phonemes[f"final:{phone}"] = _final_entry(phone, rows)
    for phone, rows in sorted(onsets.items()):
        phonemes[f"onset:{phone}"] = _onset_entry(phone, rows)

    speakers = _speaker_entries(measurements, phonemes)
    return phonemes, speakers


def _consistent(measured):
    """Split samples into (dense F1 cluster, rejected outliers).

    With fewer than three samples there is no cluster to speak of, so nothing is
    dropped -- judging one sample against itself would be circular.
    """
    if len(measured) < 3:
        return list(measured), []
    f1 = [r["formants"][0] for r in measured]
    best_centre, best_count = f1[0], -1
    for centre in f1:
        count = sum(1 for v in f1 if abs(v - centre) <= CUT_OUTLIER_HZ)
        if count > best_count:
            best_centre, best_count = centre, count
    kept = [r for r in measured
            if abs(r["formants"][0] - best_centre) <= CUT_OUTLIER_HZ]
    dropped = [r for r in measured if r not in kept]
    return kept, dropped


def _final_entry(phone, rows):
    """One final's aggregate: formants, spread, confidence, provenance.

    Samples first pass a per-final F1 consistency filter (``_consistent``). A
    final's samples come from many sentences, and a few cuts land on the
    neighbouring syllable; those describe a *different* vowel and would drag the
    median and inflate the spread. The filter is per-final, never global, and it
    keeps the dense cluster rather than trimming both tails -- a diphthong
    legitimately sweeps F2, so only F1 is used as the key.
    """
    measured = [r for r in rows if r["formants"]]
    ok, dropped = _consistent(measured)
    dur = [r["duration_ms"] for r in rows]
    conf = confidence_of(
        len(ok),
        [r["split_confidence"] for r in ok],
        [r["bound_confidence"] for r in ok],
    )
    entry = {
        "role": "final",
        "phone": phone,
        "n_samples": len(rows),
        "n_formant_samples": len(ok),
        "n_measured_before_filter": len(measured),
        "n_dropped_as_miscut": len(dropped),
        "confidence": conf,
        "confidence_rule": _rule_text(len(ok)),
        "duration_ms": {
            "mean": round(sum(dur) / len(dur), 1) if dur else None,
            "std": round(_std(dur), 1),
            "n": len(dur),
        },
        "fused": phone in FUSED_FINALS,
        "speakers": sorted({r["speaker"] for r in ok}),
    }
    if ok:
        entry["formants"] = [
            round(_median([r["formants"][i] for r in ok]), 1) for i in range(3)
        ]
        entry["bandwidths"] = [
            round(_median([r["bandwidths"][i] for r in ok]), 1) for i in range(3)
        ]
        f0s = [r["f0_hz"] for r in ok if r["f0_hz"]]
        entry["f0_hz"] = round(sum(f0s) / len(f0s), 1) if f0s else None
        entry["spread_hz"] = {
            name: round(_std([r["formants"][i] for r in ok]), 1)
            for i, name in enumerate(("f1", "f2", "f3"))
        }
        per_speaker = {}
        for r in ok:
            per_speaker.setdefault(r["speaker"], []).append(r["formants"])
        entry["per_speaker"] = {
            spk: {
                "n": len(v),
                "formants": [round(_median([f[i] for f in v]), 1) for i in range(3)],
            }
            for spk, v in sorted(per_speaker.items())
        }
        entry["provenance"] = [
            {
                "utterance": r["utterance"],
                "speaker": r["speaker"],
                "pinyin": r["pinyin"],
                "tone": r["tone"],
                "start": r["start"],
                "end": r["end"],
                "formants": r["formants"],
                "bound_confidence": r["bound_confidence"],
                "split_confidence": r["split_confidence"],
            }
            for r in ok
        ]
    else:
        entry["formants"] = None
        entry["bandwidths"] = None
        entry["f0_hz"] = None
        entry["reject"] = sorted({r.get("reject", "no_estimate") for r in rows})
    entry["usable_for_speaker"] = entry["confidence"] in ("medium", "high")
    return entry


def _onset_entry(phone, rows):
    """One onset's aggregate: duration, voicing, and noise centroid if unvoiced."""
    voiced = [r for r in rows if r["voiced_kind"]]
    ok = [r for r in rows if r["formants"]]
    dur = [r["duration_ms"] for r in rows]
    entry = {
        "role": "onset",
        "phone": phone,
        "n_samples": len(rows),
        "n_formant_samples": len(ok),
        "confidence": confidence_of(
            len(ok),
            [r["split_confidence"] for r in rows],
            [r["bound_confidence"] for r in rows],
        ),
        "confidence_rule": _rule_text(len(ok)),
        "kind": "voiced" if phone in VOICED_ONSETS else "unvoiced",
        "duration_ms": {
            "mean": round(sum(dur) / len(dur), 1) if dur else None,
            "std": round(_std(dur), 1),
            "n": len(dur),
        },
    }
    if ok:
        entry["formants"] = [
            round(_median([r["formants"][i] for r in ok]), 1) for i in range(3)
        ]
        f0s = [r["f0_hz"] for r in ok if r["f0_hz"]]
        entry["f0_hz"] = round(sum(f0s) / len(f0s), 1) if f0s else None
    else:
        entry["formants"] = None
        entry["f0_hz"] = None
        cents = [r["noise_centroid_hz"] for r in rows if r.get("noise_centroid_hz")]
        entry["noise_centroid_hz"] = (
            round(sum(cents) / len(cents), 1) if cents else None
        )
    entry["usable_for_speaker"] = False
    return entry


def _rule_text(n):
    if n >= CONF_HIGH_MIN:
        return f"n={n} >= {CONF_HIGH_MIN} -> high"
    if n >= CONF_MEDIUM_MIN:
        return f"n={n} in [{CONF_MEDIUM_MIN},{CONF_HIGH_MIN}) -> medium"
    return f"n={n} < {CONF_MEDIUM_MIN} -> low (evidence only)"


def _speaker_entries(measurements, phonemes):
    """Per-speaker summary plus a formant scale only when properly supported."""
    ref = {}
    for key, entry in phonemes.items():
        if entry["role"] == "final" and entry["formants"] and entry["usable_for_speaker"]:
            ref[key] = entry["formants"]

    per_speaker = {}
    for m in measurements:
        per_speaker.setdefault(m["speaker"], []).append(m)

    out = {}
    for spk, rows in sorted(per_speaker.items()):
        vo = [r for r in rows if r["formants"]]
        f0s = [r["f0_hz"] for r in vo if r["f0_hz"]]
        usable_keys = sorted({
            f"final:{r['phone']}" for r in vo
            if f"final:{r['phone']}" in ref
        })
        entry = {
            "utterances": len({r["utterance"] for r in rows}),
            "n_syllables": len({(r["utterance"], r["start"]) for r in rows}),
            "n_voiced_samples": len(vo),
            "n_usable_finals": len(usable_keys),
            "usable_finals": usable_keys,
            "f0_hz": round(sum(f0s) / len(f0s), 1) if f0s else None,
        }
        if len(usable_keys) < MIN_SPEAKER_TARGETS:
            entry["formant_scale"] = None
            entry["formant_scale_note"] = (
                f"only {len(usable_keys)} finals reach medium/high confidence "
                f"(need {MIN_SPEAKER_TARGETS}); a speaker scale would be a guess"
            )
            entry["confidence"] = "low"
        else:
            ratios = []
            for r in vo:
                key = f"final:{r['phone']}"
                base = ref.get(key)
                if not base:
                    continue
                ratios.extend(r["formants"][i] / base[i] for i in range(3) if base[i] > 0)
            entry["formant_scale"] = round(sum(ratios) / len(ratios), 4)
            entry["formant_scale_n"] = len(ratios)
            entry["confidence"] = "high" if len(usable_keys) >= CONF_HIGH_MIN else "medium"
        tilts = [r["spectral_tilt"] for r in vo if r.get("spectral_tilt") is not None]
        entry["spectral_tilt"] = round(sum(tilts) / len(tilts), 2) if tilts else None
        out[spk] = entry
    return out


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------
def build_params(sources=("extended", "shipped")):
    """Full pipeline -> the JSON-serialisable ``phoneme_params.json`` dict."""
    measurements, meta = collect(sources)
    phonemes, speakers = aggregate(measurements)

    by_conf = {}
    for m in measurements:
        c = m["bound_confidence"]
        by_conf[c] = by_conf.get(c, 0) + 1
    finals_conf = {}
    for key, entry in phonemes.items():
        if entry["role"] == "final":
            finals_conf[entry["confidence"]] = finals_conf.get(entry["confidence"], 0) + 1
    voiced = [m for m in measurements if m["formants"]]

    return {
        "schema": SCHEMA,
        "sr": SR,
        "confidence_contract": {
            "unit": "final (韵母); fused finals are their own unit",
            "high": f"n_formant_samples >= {CONF_HIGH_MIN}",
            "medium": f"{CONF_MEDIUM_MIN} <= n_formant_samples < {CONF_HIGH_MIN}",
            "low": f"n_formant_samples < {CONF_MEDIUM_MIN} -> evidence only, "
                   f"excluded from speaker estimates",
            "demotion": "majority of low-confidence boundaries demotes one tier",
            "speaker_scale": f"requires >= {MIN_SPEAKER_TARGETS} usable finals",
        },
        "phonemes": phonemes,
        "speakers": speakers,
        "provenance": {
            "sources": list(sources),
            "by_source": meta["by_source"],
            "n_syllables": meta["n_syllables"],
            "n_utterances": len(meta["records"]),
            "n_spans": len(measurements),
            "n_voiced_with_formants": len(voiced),
            "bound_confidence": by_conf,
            "finals_by_confidence": finals_conf,
            "analysis": {
                "order": lpc.ORDER,
                "preemphasis": lpc.PREEMPHASIS,
                "window_samples": lpc.WINDOW,
                "hop_samples": lpc.HOP,
                "method": "autocorrelation LPC + Durand-Kerner root finding",
                "f1_range_hz": list(lpc.F1_RANGE),
                "f2_range_hz": list(lpc.F2_RANGE),
                "f3_range_hz": list(lpc.F3_RANGE),
            },
            "warnings": _warnings(measurements, phonemes, speakers),
        },
    }


def _warnings(measurements, phonemes, speakers):
    """Honesty notes; part of the deliverable, not debug output."""
    out = []
    n_voiced = len([m for m in measurements if m["formants"]])
    out.append(
        f"{len(measurements)} phoneme spans from {len({(m['utterance']) for m in measurements})} "
        f"utterances; {n_voiced} produced a formant estimate."
    )
    finals = {k: v for k, v in phonemes.items() if v["role"] == "final"}
    low = sorted(k for k, v in finals.items() if v["confidence"] == "low")
    if low:
        out.append(
            f"{len(low)} finals are 'low' (<{CONF_MEDIUM_MIN} samples) and must not "
            f"feed a speaker estimate: {', '.join(k.split(':')[1] for k in low)}."
        )
    high = sorted(k for k, v in finals.items() if v["confidence"] == "high")
    out.append(
        f"{len(high)} finals reach 'high' (>= {CONF_HIGH_MIN} samples): "
        f"{', '.join(k.split(':')[1] for k in high)}."
    )
    no_scale = sorted(k for k, v in speakers.items() if v["formant_scale"] is None)
    if no_scale:
        out.append(
            f"formant_scale is null for {', '.join(no_scale)}: not enough usable "
            f"finals. Do not read speaker timbre from those entries."
        )
    lows = sum(1 for m in measurements if m["bound_confidence"] == "low")
    if lows:
        out.append(f"{lows}/{len(measurements)} spans sit on a low-confidence bound.")
    out.append(
        "The extended corpus is whole sentences: per-sample values are kept in "
        "each final's provenance so sentence-position drift stays visible rather "
        "than being averaged away."
    )
    return out


def write_params(path=OUT, sources=("extended", "shipped")):
    """Build and write ``phoneme_params.json``. Returns the dict written.

    Explicit ``encoding='utf-8'`` is mandatory: the host default codec is GBK
    and the payload contains non-ASCII text (hanzi, ``ü``).
    """
    data = build_params(sources=sources)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return data


def stats(sources=("extended", "shipped")):
    """Print the coverage table: samples per final, per speaker."""
    data = build_params(sources=sources)
    finals = {k.split(":")[1]: v for k, v in data["phonemes"].items()
              if v["role"] == "final"}
    print(f"{'final':8} {'n':>4} {'voiced':>6} {'conf':>7}  "
          f"{'F1':>7} {'F2':>7} {'F3':>7}  speakers")
    for name in sorted(finals, key=lambda k: -finals[k]["n_formant_samples"]):
        v = finals[name]
        f = v["formants"] or [None, None, None]
        print(f"{name:8} {v['n_samples']:>4} {v['n_formant_samples']:>6} "
              f"{v['confidence']:>7}  "
              f"{_fmt(f[0]):>7} {_fmt(f[1]):>7} {_fmt(f[2]):>7}  "
              f"{', '.join(v['speakers'])}")
    print()
    for spk, v in data["speakers"].items():
        print(f"{spk}: utterances={v['utterances']} syllables={v['n_syllables']} "
              f"voiced={v['n_voiced_samples']} usable_finals={v['n_usable_finals']} "
              f"scale={v['formant_scale']} conf={v['confidence']}")
    p = data["provenance"]
    print(f"\nspans={p['n_spans']} voiced_with_formants={p['n_voiced_with_formants']}")
    for w in p["warnings"]:
        print(f"  ! {w}")
    return data


def _fmt(v):
    return "-" if v is None else f"{v:.0f}"


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Build phoneme_params.json")
    p.add_argument("-o", "--output", type=Path, default=OUT)
    p.add_argument("--stats-only", action="store_true",
                   help="print the coverage table without writing")
    p.add_argument("--no-shipped", action="store_true",
                   help="ignore the nine shipped word crops")
    a = p.parse_args(argv)
    sources = ("extended",) if a.no_shipped else ("extended", "shipped")
    if a.stats_only:
        stats(sources)
        return 0
    data = write_params(a.output, sources)
    p2 = data["provenance"]
    print(f"wrote {a.output}")
    print(f"  syllables={p2['n_syllables']} utterances={p2['n_utterances']} "
          f"spans={p2['n_spans']} voiced_with_formants={p2['n_voiced_with_formants']}")
    for w in p2["warnings"]:
        print(f"  ! {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
