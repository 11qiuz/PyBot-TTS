"""Extract speaker 17's shared phone anchors from the local ASCEND cache."""
from __future__ import annotations

import json
import random
from collections import defaultdict

from exp_20_pybot_larynx import verify_stage10_bilingual as baseline

import numpy as np
import pyarrow.parquet as pq

from exp_20_pybot_larynx.crossling.fetch_bilingual_corpus import ASCEND_CACHE, _decode_audio_field_to_24k
from exp_20_pybot_larynx.verify_stage12_phone_matched import PHONES, REFERENCE_PATH, shape
from exp_v10_g2p_and_align.landmark_aligner import align_ipa_sequence_v10
from pybot_webui.crossling import _safe_plan_v13


def main():
    path = next(ASCEND_CACHE.rglob("test-*.parquet"))
    corpus = pq.ParquetFile(path)
    metadata = corpus.read(columns=["original_speaker_id", "language", "duration", "transcription", "id"]).to_pylist()
    chosen = {}
    for language in ("zh", "en"):
        pool = [index for index, row in enumerate(metadata) if (
            row["original_speaker_id"] == 17 and row["language"] == language
            and 1.3 <= row["duration"] <= 4.8 and "[" not in row["transcription"]
        )]
        random.Random(1725).shuffle(pool)
        chosen.update({index: (language, "fit" if idx < 32 else "check") for idx, index in enumerate(pool[:42])})

    audio = corpus.read(columns=["audio"])["audio"]
    groups, ids = defaultdict(list), defaultdict(set)
    for index, (language, split) in chosen.items():
        row = metadata[index]
        units = _safe_plan_v13(row["transcription"])["units"]
        if any(unit["lang"] != language for unit in units if unit.get("ipa_phones")):
            continue
        phones = [phone for unit in units for phone in unit.get("ipa_phones", [])]
        if not phones:
            continue
        wav, _ = _decode_audio_field_to_24k(audio[index].as_py())
        track = baseline.v6_voc.analyze(wav, sr=baseline.SR)
        bounds, info = align_ipa_sequence_v10(phones, track, sr=baseline.SR)
        if not info.get("dp_ok") or len(bounds) != len(phones):
            continue
        by_phone = defaultdict(list)
        for phone, (start, stop) in zip(phones, bounds):
            if phone not in PHONES:
                continue
            inner = max(1, int((stop - start) * 0.22))
            a, b = start + inner, stop - inner
            if b - a < 3:
                continue
            voiced = track.voiced[a:b]
            if voiced.sum() >= 3 and np.mean(voiced) >= 0.75:
                by_phone[phone].append(shape(np.median(track.log_env[a:b][voiced], axis=0)))
        for phone, envelopes in by_phone.items():
            key = f"{split}/{language}/{phone}"
            groups[key].append(np.median(envelopes, axis=0))
            ids[key].add(str(row["id"]))

    data = {
        "speaker": 17,
        "method": "ASCEND pure language clips, deterministic split, at most one median per phone and utterance",
        "groups": {key: {"count": len(value), "ids": sorted(ids[key]),
                         "log_env": np.round(np.median(value, axis=0), 6).tolist()}
                   for key, value in groups.items()},
    }
    REFERENCE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({key: len(value) for key, value in groups.items()}, flush=True)


if __name__ == "__main__":
    main()
