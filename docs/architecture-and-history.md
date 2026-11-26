# Architecture & Historical Retrospect (`v2` → `Stage 13`)

## 1. Current Architecture: PyBot-Larynx (`exp_20_pybot_larynx/` + `pybot_webui/`)

PyBot-Larynx replaces waveform unit concatenation with a **zero-neural-weight parametric + physical acoustic synthesizer** running at `24 kHz` (`5 ms` hop, `513` FFT bins, `64` cepstral dimensions, `6` aperiodicity bands):

1. **Glottal & Harmonic/Noise Vocoder (`exp_20_pybot_larynx/larynx/`)**:
   - **TrueHarmEnv** spectral envelopes (`64-dim` cepstra + high-quefrency resolution) cleanly decouple harmonic vocal-fold source periodicity (`coherent_phasor` / dynamic `Rd`) from broadband turbulence (`bap6`).
2. **Linguistic Frontends (`exp_20_pybot_larynx/frontend/`, `crossling/`, `multilingual/`)**:
   - **Mandarin (`zh`)**: Polyphone disambiguation, sandhi rules, two-pass Xu-coarticulation + Ridge prosody predictor (`prosody_stage7`).
   - **English (`en`)**: Text normalizer, word stress & weak-form function word G2P (`en_frontend_v2`), conversational IRLS duration + qTA F0 predictor (`en_prosody_stage10`).
   - **Extensible Plugin Registry (`multilingual/schema.py`)**: Any language (`ja`, `ko`, etc.) plugs into `register_language_frontend` by emitting standardized IPA `PhoneEvent` spans + F0 + voicing flags.
3. **Unified Acoustic Selection & 4-Rule Physical Stack (`multilingual/`)**:
   - One cross-lingual Viterbi dynamic programming pass selects 5-phase medoids guided by hierarchical Bayesian Ridge regression across shared contextual tiers (`ctx_pns -> ctx_pn -> ctx_ps -> ctx_p -> ctx_base -> ipa_shared`).
   - Four language-agnostic articulatory rules refine glottal anti-buzz/BAP bounds (`IPAGlottalAntiBuzzRule`), oral nasal zeros + VN coarticulation (`IPANasalAntiresonanceRule`), stop closure & fricative peak control (`IPAObstruentCrispnessRule`), and `cep[0:18]` S-curve intra/cross-lingual seam bridging (`IPACoarticulationAndSwitchRule`).
4. **Expressive Synthesis & Single-Exit Cloning (`expressive/`, `clone/`)**:
   - Inline affect profiles, non-verbal vocalizations (breaths, sighs, laughs), and optional single-exit speaker projection via spectral tilt, VTLN, BAP matching, and glottal phase transfer.

---

## 2. Historical Note: Early Phoneme & Waveform Concatenation (`v2`–`v19`)

Before converging on PyBot-Larynx (`exp_20_pybot_larynx/`), the project systematically explored recording-based **phoneme / diphone / UTAU-style concatenation**:

- **`v2`–`v4` (`pybot_tts_v2_recorded/`, `pybot_tts_v3_*`, `pybot_tts_v4_params/`)**: Word/syllable unit concatenation from small recorded Mandarin corpora (`AISHELL-3`), pitch-contour overlays, seam trimming, and initial LPC formant estimation experiments (documented in [docs/v4-findings.md](v4-findings.md) and [docs/corpus-expansion.md](corpus-expansion.md)).
- **`v5`–`v10` (`exp_v5_*`–`exp_v10_*`)**: Expanded Mandarin+English phoneme concatenation with DSP fricative repairs, onset high-frequency boosting, forced alignment auditing, and timbre equalization.
- **`v11`–`v19` (`exp_v11_*`–`exp_19b_*`)**: GMM-HMM boundary aligners, diphone selection, natural vocoder hybrids, UTAU-style unit ceilings, and multi-speaker `BZNSYP` / `SSB0502` fluent Mandarin concatenation pipelines.

### Why We Transitioned Away From Waveform Concatenation
Direct time-domain / waveform-segment concatenation suffered from structural bottlenecks:
1. **Pitch & Duration Manipulation Artifacts**: Stretching or pitch-shifting raw recorded slices introduced phase hollowness and choppiness at arbitrary prosodic contours.
2. **Coarticulation & Cross-Lingual Seams**: Stitching independently recorded Mandarin and English slices produced audible vowel-formant jumps, loudness clicks, and speaker-switch sensations.
3. **Lack of Expressive Headroom**: Emotional modulation (whisper, sadness creak, excitement brightness) was severely constrained by frozen unit spectra.

Re-building synthesis around **TrueHarmEnv spectral-envelope + dynamic glottal excitation + explicit articulatory rules (`exp_20_pybot_larynx/`)** solved all three limitations while preserving strict **0.0 MB neural model weights** and ~30x real-time CPU speed. Selected shared DSP helpers, G2P aligners, and speaker indices from those earlier modules remain reused under clean boundaries by PyBot-Larynx.
