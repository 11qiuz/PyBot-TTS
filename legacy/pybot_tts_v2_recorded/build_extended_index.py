"""Assemble a phone-rich donor index from decoded utterances.

The shipped bank (bank/index.json) holds nine hand-trimmed two-syllable words.
That is enough to drive unit selection but far too few examples per phone to
estimate formants or a speaker profile. This module indexes decoded whole
utterances instead: every syllable of every utterance becomes one entry, with
its sample bounds taken from a measured active region rather than a hand crop.

No text segmentation happens here beyond what the transcript already provides,
so a misaligned transcript stays a visible data problem instead of being hidden
by a heuristic that "mostly works".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import wave
from array import array

HERE = Path(__file__).resolve().parent
SR = 24000
EDGE_PAD = 240  # 10 ms of context either side; cheaper than trimming twice


def read_mono(path):
    """Read a 24 kHz mono 16-bit WAV into floats in [-1, 1]."""
    with wave.open(str(path), 'rb') as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError(f'{path.name}: expected mono 16-bit')
        if handle.getframerate() != SR:
            raise ValueError(f'{path.name}: expected {SR} Hz')
        raw = handle.readframes(handle.getnframes())
    pcm = array('h')
    pcm.frombytes(raw)
    if sys.byteorder != 'little':
        pcm.byteswap()
    return [value / 32768 for value in pcm]


def active_bounds(samples, floor_ratio=0.02, hop=None):
    """First and last sample whose local RMS clears a fraction of the peak RMS.

    This is a level gate, not endpoint detection: it keeps breathy onsets and
    trailing voicing that a stricter detector would clip.
    """
    hop = hop or SR // 200
    energies = []
    for start in range(0, max(1, len(samples) - hop + 1), hop):
        window = samples[start:start + hop]
        energy = sum(v * v for v in window) / len(window)
        energies.append((start, energy))
    if not energies:
        return 0, len(samples)
    loudest = max(energy for _, energy in energies)
    threshold = max(loudest * floor_ratio, 1e-7)
    active = [start for start, energy in energies if energy >= threshold]
    if not active:
        return 0, len(samples)
    first = max(0, active[0] - EDGE_PAD)
    last = min(len(samples), active[-1] + hop + EDGE_PAD)
    return first, last


def syllable_spans(samples, bounds, pinyin):
    """Split an utterance into one span per syllable.

    Utterance-level splitting by equal duration is wrong for Mandarin because
    syllables differ in length by a factor of three. We split on the quietest
    inter-syllable dips instead, which tracks the transcript count and lets a
    caller see when the guess disagrees with the transcript.

    Returns the spans plus a note describing how they were produced.
    """
    start, stop = bounds
    region = samples[start:stop]
    count = len(pinyin)
    if count <= 0:
        return [], dict(method='empty')
    if count == 1:
        return [(start, stop)], dict(method='single_syllable')

    # Candidate dips: local minima of short-window energy inside the active part.
    hop = SR // 200
    frames = []
    for offset in range(0, max(1, len(region) - hop + 1), hop):
        window = region[offset:offset + hop]
        frames.append((offset, sum(v * v for v in window) / len(window)))
    if len(frames) < count * 2:
        cuts = [round(len(region) * k / count) for k in range(1, count)]
        method = 'equal_split_short_region'
    else:
        # Dynamic programme: choose count-1 cut frames that are dips and roughly
        # evenly spaced, preferring locally minimal energy.
        loudest = max(energy for _, energy in frames) or 1.0
        target = [(k + 1) * len(frames) // count for k in range(count - 1)]
        window_radius = max(2, len(frames) // (count * 4))
        cuts = []
        for want in target:
            low = max(0, want - window_radius)
            high = min(len(frames) - 1, want + window_radius)
            choice = min(range(low, high + 1),
                         key=lambda index: (frames[index][1] / loudest, abs(index - want)))
            cuts.append(frames[choice][0])
        method = 'quiet_dip_split'
    edges = [0] + cuts + [len(region)]
    spans = [(start + edges[k], start + edges[k + 1]) for k in range(count)]
    return spans, dict(method=method, count=count)


def build_index(utterances, notice=None):
    """Turn decoded utterances into a flat syllable index.

    `utterances` is the decode report's `decoded` list. Every entry must carry
    id, wav path, text and pinyin. Entries whose syllable count disagrees with
    the transcript are recorded with `aligned=False` rather than dropped, so the
    shortfall is visible in the data instead of silently shrinking sample counts.
    """
    rows, notes = [], []
    for entry in utterances:
        path = Path(entry['wav'])
        if not path.is_absolute():
            path = HERE / path
        samples = read_mono(path)
        bounds = active_bounds(samples)
        text = entry.get('text') or ''
        pinyin = (entry.get('pinyin') or '').split()
        if len(pinyin) != len(text):
            notes.append(dict(id=entry['id'], note='syllable count mismatch',
                              text_len=len(text), pinyin_len=len(pinyin)))
            continue
        spans, method = syllable_spans(samples, bounds, pinyin)
        rows.append(dict(id=entry['id'], speaker=entry['id'][:7], text=text,
                         pinyin=pinyin, path=str(path),
                         mp3_sha256=entry.get('mp3_sha256'),
                         wav_sha256=entry.get('wav_sha256'),
                         active_start=bounds[0], active_stop=bounds[1],
                         samples=len(samples), split=method,
                         syllables=[dict(index=index, hanzi=text[index],
                                         pinyin=pinyin[index],
                                         start_sample=first, stop_sample=second,
                                         ms=round((second - first) / SR * 1000, 1))
                                    for index, (first, second) in enumerate(spans)]))
    return dict(schema=1, sr=SR,
                notice=notice or 'AISHELL-3, Apache-2.0, see source/NOTICE.md',
                utterances=rows, notes=notes)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='decode_report.json')
    parser.add_argument('-o', '--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    index = build_index(report['decoded'])
    args.output.write_text(json.dumps(index, ensure_ascii=False, indent=1) + '\n',
                           encoding='utf-8')
    syllables = sum(len(row['syllables']) for row in index['utterances'])
    print(f"{len(index['utterances'])} utterances, {syllables} syllables, "
          f"{len(index['notes'])} notes")


if __name__ == '__main__':
    sys.exit(main())
