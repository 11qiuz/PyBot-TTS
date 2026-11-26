# PyBot-Larynx (`exp_20_pybot_larynx`) Architecture & Stage 0 Specification

## 1. Mission & Design Philosophy
Adapt the `huggingface/pybot` paradigm ("agent swarm writing massive explicit, modular, testable Python rules with zero neural weights") to **Mandarin electronic vocal cord synthesis**, replacing raw waveform concatenation (`v15`) with a **pure electronic vocal cord + explicit acoustic rules engine**:

- **Zero neural network weights** (`0.0 MB`)
- **Zero waveform concatenation / residual snippet pasting** (`100%` synthesized by the electronic vocal cord kernel)
- **Fast CPU synthesis** (target end-to-end `RTF <= 0.08`, kernel `RTF <= 0.05`)
- **Strict modular attribution & fault isolation** (every frame-level modification is traceable to a named rule module)

## 2. Five-Layer Architecture (Adapted from `huggingface/pybot`)
1. **`frontend/`** (Text -> Linguistic & Prosodic Plan): Reuses `plan_v13` + rule-driven prosody/breaks (Stage 0 reuses `ProsodyModelV15` in `C4_auto_e2e`; later swarm expands linguistic rules).
2. **`state/`** (Utterance & Cross-Syllable Context): Tracks syllable grid, onset/final phase progress `u in [0, 1]`, speaker F0 register, and boundary adjacency.
3. **`acoustics/`** (Explicit Acoustic Knowledge Base): Structured, human-inspectable cepstral (`DCT-64`), aperiodicity (`6-band BAP`), and onset-duration ratio knowledge extracted strictly from `TRAIN` (no audio slices, no neural weights).
4. **`rules/` + `arbiter.py`** (Independent Expert Rule Modules + Continuous Layered Arbiter):
   - Each rule module inspects `UtteranceContext` + base `LarynxControl` and emits a masked `RuleProposal` (additive deltas on `log_env`, `bap`, `glottal_tilt_db_per_oct`, `jitter_frac`, `shimmer_frac`, or hard `override_voiced` mask).
   - Unlike text chatbots where `arbiter` picks a single winning reply, `arbiter.py` fuses **base trajectory + masked weighted deltas**, isolates crashing modules without halting synthesis, enforces physical bounds, and logs exact per-module attribution.
5. **`larynx/`** (Frozen Vectorized Electronic Vocal Cord Kernel):
   - Operates in the unified **STFT / HNM coupled domain** (validated by `exp_19b` STFT noise findings and `exp_v12` coherent glottal phase findings — explicitly avoiding `exp_19b`'s broken LPC residual domain).
   - Synthesizes voiced glottal excitation using analytic Rosenberg/LF coherent glottal phase (`parametric_lf`) or speaker-coherent phase (`speaker_coherent`) below `3.5 kHz` + smooth dispersion above, coupled with dynamic `glottal_tilt_db_per_oct`, AR(1) `jitter_frac`/`shimmer_frac`, minimum-phase vocal tract filtering `H_min`, 5 ms voiced-tail gating, and 5 ms COLA STFT turbulence noise.

## 3. Zero-Leakage Data Split (Frozen)
From `exp_v15_bznsyp_tts/bznsyp_index_v15.pkl`:
- **`HELDOUT_20`**: `rz.HELD` (`20` utterances). Strictly read-only final judge; NEVER touched during knowledge extraction or swarm rule tuning.
- **`DEV_60`**: `sorted(random.Random(2026).sample(sorted(train_utt_metas), 60))` (`60` utterances). Swarm evolution and regression benchmark.
- **`TRAIN`**: Remaining `9,666` training utterances (`Stage 0` extracts from a deterministic `1,200`-utterance subset `seed=2026`).
- Pairwise disjointness is enforced by unit test (`test_zero_leakage_splits`).

## 4. Stage 0 Go / No-Go Gates
Evaluated on `HELDOUT_20` (same `rz.sf` Mel-LSD / MCD口径 as `v15` and `19b`):
- **`C0` (`larynx` copy-synth floor)**: `Mel-LSD <= 3.0 dB`, `CPU RTF <= 0.05`, `sanity 20/20`.
- **`C3` (`Oracle` timing/F0 + dynamic knowledge + 4 initial PyBot rules, 100% electronic)**:
  - Must decisively beat `v5_oracle` (`8.26 dB`), `19b aPN3` (`9.24 dB`), and `v15_auto` (`6.88 dB`);
  - **Go Gate**: `Mel-LSD <= 5.8 dB`, `CPU RTF <= 0.08`, `sanity 20/20`.
- **Subjective Check**: Export `-20 dBFS` normalized WAVs and `listen.html` for direct A/B comparison against `Real` and `v15_auto`.

## 5. Swarm Expansion Discipline (Post-Stage 0)
1. **No raw waveform slicing**: Rules may emit only parametric/spectral deltas or physical state controls.
2. **Keep mechanism separate from tables**: Do not mutate frozen `larynx/synth.py` when adding phonetic/prosodic rules.
3. **1:1 Unit Tests + Multi-Metric Non-Regression Gate**: Every new rule module ships with its own unit test and must not regress `DEV_60` (`Mel-LSD`, `MCD`, `uv_2_8k_vs_lo_db`, `clicks_gt8`, `RTF`).
4. **Never optimize on `HELDOUT_20`**: Swarm agents inspect only `TRAIN` statistics and `DEV_60` evaluation scores.
