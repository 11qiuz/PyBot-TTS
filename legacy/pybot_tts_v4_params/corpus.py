"""Read the extended donor index in the shape the parameter builders want.

The shipped bank exposes two-syllable words with hand crop points. The extended
corpus exposes whole utterances split into syllables by energy dips. Both are
useful; they differ in structure. Rather than teach every consumer about both,
this module normalises them into one record type and states, per record, how
much the sample bounds can be trusted.

Confidence here describes *bounding quality*, never whether the audio matches
the transcript. That cannot be checked without listening.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SR = 24000

SHIPPED_INDEX = ROOT / 'pybot_tts_v2_recorded' / 'bank' / 'index.json'
EXTENDED_INDEX = ROOT / 'pybot_tts_v2_recorded' / 'extended' / 'index.json'


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_shipped(path=SHIPPED_INDEX):
    """The nine hand-trimmed words, as syllable records.

    Hand trimming is the most trustworthy bound available, so these are marked
    high confidence and carry the original crop numbers unchanged.
    """
    rows = []
    for entry in _read_json(path):
        wav = (ROOT / 'pybot_tts_v2_recorded' / 'bank' / entry['path'])
        count = len(entry['pinyin'])
        if count != len(entry['text']):
            continue
        span = entry['end_sample'] - entry['start_sample']
        for index in range(count):
            # A hand crop covers the whole word, so per-syllable bounds inside it
            # are unknown. Report the word span and say so.
            rows.append(dict(speaker=entry['speaker'], utterance=entry['id'],
                             wav=str(wav), hanzi=entry['text'][index],
                             pinyin=entry['pinyin'][index],
                             start_sample=entry['start_sample'],
                             stop_sample=entry['end_sample'],
                             ms=round(span / SR * 1000, 1),
                             bound='word_span',
                             bound_confidence='low',
                             source='shipped_bank'))
    return rows


def load_extended(path=EXTENDED_INDEX):
    """Decoded utterances, one record per energy-dip-delimited syllable."""
    rows = []
    for utterance in _read_json(path)['utterances']:
        method = (utterance.get('split') or {}).get('method')
        confidence = 'medium' if method == 'quiet_dip_split' else 'low'
        for syllable in utterance['syllables']:
            rows.append(dict(speaker=utterance['speaker'],
                             utterance=utterance['id'],
                             wav=utterance['path'],
                             hanzi=syllable['hanzi'],
                             pinyin=syllable['pinyin'],
                             start_sample=syllable['start_sample'],
                             stop_sample=syllable['stop_sample'],
                             ms=syllable.get('ms'),
                             bound='syllable',
                             bound_confidence=confidence,
                             source='extended'))
    return rows


def load_all(prefer='both'):
    """Return syllable records from both sources, de-duplicated by wav+syllable.

    `prefer` chooses which source wins when the same utterance appears twice.
    Shipped bank crops are shorter windows of the same recordings, so they are
    kept alongside the extended rows; a caller that wants only trustworthy ones
    should filter on `bound_confidence`.
    """
    if prefer == 'shipped':
        return load_shipped()
    if prefer == 'extended':
        return load_extended()
    return load_shipped() + load_extended()


def summary(rows):
    """Counts a caller can quote without re-reading the corpus."""
    from collections import Counter
    by_speaker = Counter(row['speaker'] for row in rows)
    by_bound = Counter(row['bound_confidence'] for row in rows)
    by_source = Counter(row['source'] for row in rows)
    return dict(records=len(rows), speakers=dict(by_speaker),
                bound_confidence=dict(by_bound), source=dict(by_source))
