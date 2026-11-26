"""Pick a minimal extra-donor set that covers every Mandarin phone N times, per speaker.

This is a coverage-selection helper, not a synthesizer. It reads the published
AISHELL-3 transcript tables for the two speakers already used by the bank, then
greedily picks the fewest recordings that bring each (onset, final, tone) triple
up to a wanted sample count. The point is to make downstream parameter
estimation have real samples instead of one recording per phone.

Nothing here edits or re-decodes audio; see build_bank.py for the crop step.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __package__:
    from .engine import split_pinyin
else:
    from engine import split_pinyin

HERE = Path(__file__).resolve().parent
SR = 24000


def load_transcripts(path):
    """Read a transcript table; tolerate both a list and a {speaker: [...]} map."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if "records" in data:
            return list(data["records"])
        raise ValueError("expected a JSON array of transcript records")
    if not isinstance(data, list):
        raise ValueError("expected a JSON array of transcript records")
    return data


def phone_triples(record):
    """Signed (onset, final, tone) triples plus positions, or [] if malformed."""
    text = record.get("text") or ""
    syllables = (record.get("pinyin") or "").split()
    if not syllables or len(syllables) != len(text):
        return []
    out = []
    for index, reading in enumerate(syllables):
        try:
            onset, final, tone = split_pinyin(reading)
        except ValueError:
            return []
        out.append(dict(index=index, onset=onset, final=final, tone=tone,
                        syllable=reading, hanzi=text[index]))
    return out


def estimate_seconds(record, syllables):
    """Rough speech duration from the transcript, used only for ordering.

    AISHELL-3 utterances run near 3.5 syllables/second; this is an ordering hint
    for cheap representative picks, never a measurement or a claim about audio.
    """
    return len(syllables) / 3.5 if syllables else 0.0


def choose(records, wanted, exclude=(), focus=(), max_syllables=0):
    """Greedy cover: prefer gain, then fewer syllables, then stable id order.

    Returns (picked, coverage) where coverage maps triple -> [record ids].
    """
    banned = set(exclude)
    pool = []
    for record in records:
        if record.get("id") in banned:
            continue
        triples = phone_triples(record)
        if not triples:
            continue
        if max_syllables and len(triples) > max_syllables:
            continue
        keys = sorted({(t["onset"], t["final"], t["tone"]) for t in triples})
        pool.append(dict(record=record, keys=keys,
                         syllables=len(triples),
                         seconds=estimate_seconds(record, triples)))

    coverage = {}
    picked = []
    focus = set(focus)

    def deficit(key):
        return max(0, wanted - len(coverage.get(key, ())))

    while True:
        best = None
        for entry in pool:
            if entry["record"]["id"] in {p["id"] for p in picked}:
                continue
            gain = sum(1 for key in entry["keys"] if deficit(key) > 0)
            if not gain:
                continue
            # Weighted: a phone that is still far from quota is worth more, and
            # finals named by focus carry extra weight so short phrases that hit
            # the target terminals win ties.
            score = sum(deficit(key) for key in entry["keys"] if deficit(key) > 0)
            score += sum(math.log1p(deficit(key)) for key in entry["keys"]
                         if deficit(key) > 0 and key[1] in focus)
            candidate = (-score, entry["syllables"], entry["seconds"], entry["record"]["id"])
            if best is None or candidate < best[0]:
                best = (candidate, entry, gain)
        if best is None:
            break
        _, entry, _ = best
        picked.append(entry["record"])
        for key in entry["keys"]:
            coverage.setdefault(key, []).append(entry["record"]["id"])
        # Stop as soon as every requested phone has enough support.
        if all(len(v) >= wanted for v in coverage.values()):
            keys_seen = {k for e in pool for k in e["keys"]}
            if all(len(coverage.get(k, ())) >= wanted for k in keys_seen):
                break

    return picked, coverage


def report(coverage, wanted):
    rows = []
    for key in sorted(coverage):
        ids = coverage[key]
        rows.append(dict(onset=key[0], final=key[1], tone=key[2], count=len(ids),
                         enough=len(ids) >= wanted, ids=ids))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", nargs="+", type=Path,
                        help="AISHELL-3 transcript tables (JSON arrays)")
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--want", type=int, default=3,
                        help="target samples per (onset, final, tone) triple")
    parser.add_argument("--exclude", default="",
                        help="comma-separated ids already in the bank")
    parser.add_argument("--focus-finals", default="",
                        help="comma-separated finals to favour")
    parser.add_argument("--limit", type=int, default=0,
                        help="cap on picked recordings (0 = no cap)")
    parser.add_argument("--max-syllables", type=int, default=0,
                        help="ignore recordings longer than this many syllables")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for path in args.transcript:
        records.extend(load_transcripts(path))
    records = [r for r in records if r.get("speaker") == args.speaker]
    if not records:
        parser.error(f"no transcripts for speaker {args.speaker}")

    picked, coverage = choose(records,
                              wanted=args.want,
                              exclude=[x for x in args.exclude.split(",") if x],
                              focus=[x for x in args.focus_finals.split(",") if x],
                              max_syllables=args.max_syllables)
    if args.limit and len(picked) > args.limit:
        picked = picked[:args.limit]

    result = dict(speaker=args.speaker, wanted=args.want,
                  transcripts=len(records), picked=len(picked),
                  estimated_seconds=round(sum(p["syllables"] for p in
                      (dict(syllables=len(phone_triples(r))) for r in picked)) / 3.5, 2),
                  records=[dict(id=r["id"], text=r.get("text"),
                                pinyin=r.get("pinyin"), audio=r.get("audio"),
                                license=r.get("license"))
                           for r in picked],
                  coverage=report(coverage, args.want),
                  caveat=("Greedy coverage over published transcripts only. "
                          "No audio was decoded here, durations are estimates, and "
                          "having a phone in the transcript does not prove it is "
                          "cleanly identifiable in the audio."))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")

    enough = sum(1 for row in result["coverage"] if row["enough"])
    print(f"{args.speaker}: picked {len(picked)} recordings for "
          f"{len(result['coverage'])} phones, {enough} at quota {args.want}")


if __name__ == "__main__":
    sys.exit(main())
