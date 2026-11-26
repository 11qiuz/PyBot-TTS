"""Construct a Mandarin onset when the recording bank has no usable example."""
from functools import lru_cache
import math

import numpy as np

from exp_v5_arch.crossling_frontend import PINYIN_ONSET_TO_IPA
from exp_v6_crossling_dsp.arbiter import build_frame_track
from exp_v6_crossling_dsp.hnm_synth import _fricative_db, _vocal_tract_db
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS
from exp_v6_crossling_dsp.speaker_profile import neutral_female_reference
from exp_v10_integrity import N_FFT, SR
from exp_v10_integrity.gain_calibration import parseval_frame_db


_FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)
_REFERENCE = neutral_female_reference()
_DB_TO_LOG_AMPL = math.log(10) / 20


@lru_cache(maxsize=None)
def onset_template(onset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Convert the existing phone model's closure, release and voicing to HNM features."""
    ipa, = PINYIN_ONSET_TO_IPA[onset]
    spec = IPA_SPECS[ipa]
    track = build_frame_track({"units": [{
        "lang": "zh", "ipa_phones": [ipa], "duration_ms": 0.65 * spec.base_dur_ms,
        "pitch_st": [0.0] * 5, "gain": 1.0,
    }]}, _REFERENCE, lead_in_ms=0, lead_out_ms=0)
    # The v6 track includes one padding frame at each end even with zero padding requested.
    voiced = track.voiced_gain[1:-1] > 0
    gain = (track.voiced_gain if spec.voiced else track.noise_gain)[1:-1]
    env = np.empty((len(gain), len(_FREQS)))
    for i in range(len(gain)):
        k = i + 1
        if spec.voiced:
            shape_db = _vocal_tract_db(
                _FREQS, track.formants[k], track.bandwidths[k],
                track.zero_freq[k], track.zero_bw[k], track.zero_depth_db[k], _REFERENCE,
            )
        else:
            shape_db = _fricative_db(_FREQS, track.fric_p1[k], track.fric_p2[k], _REFERENCE)
        env[i] = shape_db * _DB_TO_LOG_AMPL
    level = -27.0 + 20 * np.log10(np.maximum(1e-4, gain / np.max(gain)))
    env += ((level - parseval_frame_db(env)) * _DB_TO_LOG_AMPL)[:, None]
    bands = [0.22, 0.35, 0.65, 0.86, 0.94] if spec.voiced else [0.995] * 5
    bap = np.tile(bands, (len(gain), 1))
    closure_frames = int(np.flatnonzero(gain > 0)[0])
    return env, bap, voiced, closure_frames
