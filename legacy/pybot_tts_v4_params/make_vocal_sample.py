"""Build ``samples/vocal_01.wav`` + ``samples/vocal_01.json`` for text 现在回家？.

The JSON is the point of this file: it records, per rendered unit and per rime
segment, where every formant came from, so a reader can check the report's claims
instead of trusting them. It is generated from the same ``trace`` the renderer
returns, not re-derived, so the metadata cannot drift from the audio.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pybot_tts_v2_recorded import engine, synth          # noqa: E402
from pybot_tts_v4_params import lpc, vocal              # noqa: E402

TEXT = "现在回家？"
SEED = 72
SR = vocal.SR
OUT = HERE / "samples"


def rms(values):
    return (sum(v * v for v in values) / max(1, len(values))) ** .5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params = vocal.load_params(HERE / "phoneme_params.json")

    samples, trace = vocal.render_full(TEXT, params, seed=SEED, speaker="SSB0502")
    vocal.write_wav(OUT / "vocal_01.wav", samples)

    base = synth.render(engine.plan(TEXT, "auto")["units"], seed=SEED)
    # Measure sample 0 of the *unit* to locate each unit's level change; the
    # 75 ms lead-in is silence, so a whole-file ratio would be meaningless.
    lead = int(.075 * SR)

    units = []
    cursor = lead
    for plan, u in zip(trace["units"], engine.plan(TEXT, "auto")["units"]):
        n = int(round(float(u["duration_ms"]) / 1000 * SR))
        span = slice(cursor, cursor + n)
        units.append({
            "onset": plan["onset"],
            "requested_final": plan["requested_final"],
            "duration_ms": plan["duration_ms"],
            "duration_split": plan["duration_split"],
            "onset_s": plan["onset_s"],
            "n_samples": n,
            "onset_source": plan["onset_source"],
            "onset_noise_centroid_hz": plan.get("onset_noise_centroid_hz"),
            "onset_bright_generic": plan.get("onset_bright_generic"),
            "onset_bright_measured": plan.get("onset_bright_measured"),
            "onset_note": plan.get("onset_note"),
            "final_entry_source": plan["final_entry_source"],
            "fallbacks": plan["fallbacks"],
            "measured_segments": plan["measured_segments"],
            "fallback_segments": plan["fallback_segments"],
            "used_formants": list(plan["used_formants"]),
            "f3_used_from_table": plan["f3_used_from_table"],
            "segments": plan["segments"],
            "targets_hz": plan["targets_hz"],
            # Objective comparison against the baseline render, same seed.
            "rms_parametric": round(rms(samples[span]), 6),
            "rms_synth": round(rms(base[span]), 6),
            "rms_ratio": round(rms(samples[span]) / rms(base[span]), 4)
            if rms(base[span]) else None,
        })
        cursor += n

    f3_dropped = sorted({
        k.split(":", 1)[1]
        for k, v in params["phonemes"].items()
        if k.startswith("final:") and v.get("formants")
    })
    doc = {
        "text": TEXT,
        "sr": vocal.SR,
        "seed": SEED,
        "speaker": "SSB0502",
        "n_samples": len(samples),
        "duration_s": round(len(samples) / vocal.SR, 3),
        "peak": round(max(map(abs, samples)), 6),
        "used_formants": list(vocal.USED_FORMANTS),
        "f3_used_from_table": False,
        "f3_policy": params["notes"].get("f3"),
        "f3_note": (
            "F3 from the table is never rendered. Step 2 found measured F3 "
            "systematically 158-420 Hz high and unusable, so only F1/F2 are "
            "taken from a measurement; the third spectral_weight slot is the "
            "generic F3 shifted by (F2_measured - F2_generic), purely to keep "
            "the envelope shape. Voiced onsets likewise use synth's generic "
            "V['nasal']/V['schwa'] rather than the table."),
        "measured_segments": trace["measured_segments"],
        "fallback_segments": trace["fallback_segments"],
        "fallbacks": trace["fallbacks"],
        "units_total": trace["units_total"],
        "fallback_semantics": {
            "fallbacks": "phonemes of the requested final that came out "
                         "unmeasured (a 'low' final counts even when its "
                         "segments are individually measured, e.g. ei = eh+i)",
            "fallback_segments": "rime segments that kept synth's generic "
                                 "formant triple",
        },
        "finals_in_table": len(f3_dropped),
        "confidence_counts": params["provenance"].get("finals_by_confidence"),
        "confidence_contract": params.get("confidence_contract"),
        "limitations": [
            "Subjective sound quality is unknown: this was built without ears "
            "on the output. Only the objective numbers below are claimed.",
            "No claim is made that this sounds more natural than synth.py.",
            "With coarticulation on, output is NOT bit-identical to synth; "
            "equality holds only with coarticulation=False on an empty table.",
            "Measured F1/F2 with amp held fixed changes per-syllable RMS; see "
            "rms_ratio per unit. This is a loudness side effect, not a defect "
            "that was compensated for.",
            "Fricative brightness is a single knob driven by a measured "
            "centroid; it does not reproduce a fricative spectrum. /s/ ends up "
            "darker than synth made it and /h/ clips at the floor.",
        ],
        "units": units,
    }
    (OUT / "vocal_01.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {OUT / 'vocal_01.wav'} ({len(samples)} samples, "
          f"{doc['duration_s']} s, peak {doc['peak']})")
    print(f"wrote {OUT / 'vocal_01.json'}")
    print(f"measured_segments={trace['measured_segments']} "
          f"fallback_segments={trace['fallback_segments']} "
          f"fallbacks={trace['fallbacks']}")
    for u in units:
        print(f"  {u['onset'] or '-':>2}/{u['requested_final']:5s} "
              f"seg {u['measured_segments']}M/{u['fallback_segments']}F "
              f"rms_ratio {u['rms_ratio']}  {u['duration_split']}")


if __name__ == "__main__":
    main()
