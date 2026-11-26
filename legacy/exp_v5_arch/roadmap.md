# PyBotTTS Phase-2 Cross-Lingual & Few-Shot Voice Cloning Roadmap

**Author:** `crossling-arch`  
**Scope:** Architectural audit of current PyBotTTS (`v2`–`v4`), structural gap analysis for cross-lingual few-shot voice cloning, environment & corpus feasibility on the local RTX 4080 Laptop (12 GB VRAM), and a falsifiable Phase-2 experimental roadmap.  
**Scientific Standard:** Consistent with [`docs/v4-findings.md`](../docs/v4-findings.md), negative results are expected, preserved, and diagnosed rather than tuned away. Every milestone below defines explicit falsification and Go/No-Go criteria.

---

## 1. Current Codebase Audit: Mandarin-Hardcoded vs. Language-Agnostic

### 1.1 Frontend & Prosody (`pybot_tts_v2_recorded/engine.py`, `auto_rules.py`, `synth.py`)

| Component | Mandarin-Hardcoded Elements | Language-Agnostic / Reusable Elements |
|---|---|---|
| [`pybot_tts_v2_recorded/engine.py`](../pybot_tts_v2_recorded/engine.py) | • Closed lexicon: `WORDS` (100 entries) and `CHARS` (188 entries) mapped to numbered pinyin; unknown characters raise `ValueError`.<br>• `_segment()` enforces 1 Hanzi = 1 syllable.<br>• `split_pinyin()` hardcodes Mandarin `ONSET` regex (`zh\|ch\|sh\|[bpmfdtnlgkhjqxrzcsyw]`), `Y`/`W` orthographic rules, `j/q/x + u -> v`, apical vowels `ir`/`iz`, and tones `1..5`.<br>• `_sandhi()` hardcodes `"不"`, `"一"`, and within-foot T3+T3 sandhi.<br>• `CURVE` (tones 1–5 semitone trajectories) and `plan()` duration offsets (T5=185 ms vs 270 ms, aspirated/affricate +18 ms) assume syllable-timed Mandarin.<br>• `validate_plan()` requires `len(hanzi) == 1`, `tone in 1..5`, and `(onset, final) in _KNOWN_SEGMENTS` (150 pairs). | • IR separation between planner (`plan()`) and renderer (`units` dict with `duration_ms`, `gain`, `pitch_st`, `pause_after_ms`, `clauses`).<br>• Punctuation pause mapping (`PAUSE`, `TERMINAL`) is largely transferable once decoupled from Hanzi offsets. |
| [`pybot_tts_v2_recorded/auto_rules.py`](../pybot_tts_v2_recorded/auto_rules.py) | • `HAN = r"[\u3400-\u9fff]"`, `ACTION`, `WH`, `SURPRISE`, `POLAR` (`A不A` / `是不是`…), `NEGATION`, `CONTRAST` (`不是…而是…`), `LIMIT_RE`.<br>• `analyze_clause()` assumes Chinese surface cues for mood and focus spans. | • Clause-level `PROFILES` abstraction (8 communicative intents -> `(base_pitch, boundary_pitch, final_duration, gain)`) and quote masking (`_mask_quotes`) as a structural pattern, though numerical offsets are Mandarin-tuned. |
| [`pybot_tts_v2_recorded/synth.py`](../pybot_tts_v2_recorded/synth.py) | • `V` (12 vowel/tract targets including Mandarin `apical`, `retro`), `FINALS` (49 Pinyin rimes), `FRIC` (6 Mandarin fricatives), `STOP` (12 Mandarin stops/affricates), `NASAL_SOUND` (`m, n, l`).<br>• `render()` hardcodes CV syllable structure (`c_dur <= 0.39 * duration`), lacking English voiced obstruents (`/b, d, g, v, z, ʒ, ð, dʒ/`), dental/postalveolar contrasts (`/θ, ð/`), coda obstruents, and `CC-` / `-CCC` clusters. | • 24 kHz sample rate (`SR = 24000`), `interp()`, phase-continuous additive harmonic loop, Lorentzian `spectral_weight()`, and 16-bit PCM `write_wav()`. |

### 1.2 v4 Parameterization & Corpus (`pybot_tts_v4_params/`, `_transcripts/`, `_final_index.json`)

- **Hardcoded to Mandarin CV syllables:**
  - [`align.py`](../pybot_tts_v4_params/align.py) relies on `engine.split_pinyin`, `FINAL_TARGETS` (43 Pinyin finals), and energy-dip segmentation; on 346 real syllables, boundary detection achieved only **2.6% high confidence** (9/346), with **0% high** on zero-onset (54) and sonorant onsets `m/n/l/r` (39) and **0** high in-final vowel-transition cuts.
  - [`params.py`](../pybot_tts_v4_params/params.py) hardcodes Mandarin `VOICED_ONSETS = {"m", "n", "l", "r"}`, `UNVOICED_ONSETS` (17 Mandarin obstruents), `VOICED_FINALS = True`, `FUSED_FINALS` (24 glide+rime finals), and `_split_syllable()` (20–75% energy-valley split).
  - [`vocal.py`](../pybot_tts_v4_params/vocal.py) fixes `F0 = 152.0` Hz, ignores `spectral_tilt_db_per_oct` and per-speaker F0, ignores F3 (`USED_FORMANTS = ("f1", "f2")` due to +158..+420 Hz LPC bias), and adjusts fricatives via a single scalar brightness knob (`[.08, .72]`) that reverses Mandarin `s < x < sh < h` vs generic `f < x < sh < s` rank order.
  - [`corpus.py`](../pybot_tts_v4_params/corpus.py), `_transcripts/`, and `_final_index.json`:
    - `D:\ZaWu\PyBotTTS\_transcripts\SSB0502.json` (476 utterances) and `SSB1593.json` (488 utterances) are Mandarin AISHELL-3 transcripts with Pinyin annotations.
    - `D:\ZaWu\PyBotTTS\_final_index.json` maps 37 Mandarin finals per speaker (5,945 refs for `SSB0502`, 4,693 refs for `SSB1593`).
    - `pybot_tts_v2_recorded/extended/` contains 32 decoded utterances (346 syllables, **81,195 ms ≈ 1.35 min** total active syllable speech across both female speakers).
- **Language-agnostic reusable assets:**
  - [`lpc.py`](../pybot_tts_v4_params/lpc.py): 24 kHz order-22 Levinson-Durbin + Durand-Kerner root solver and ACF periodicity/F0 estimator (65–400 Hz) — reusable as a diagnostic analyzer for any language (subject to F3 bias caveat).
  - [`speaker.py`](../pybot_tts_v4_params/speaker.py): LTAS spectral tilt (`80–8000 Hz` dB/oct), Bark perceptual metric (`hz_to_bark`), and the **strict LOO-CV evaluation protocol** (comparing `model` vs `baseline_speaker_blind` vs `oracle_same_speaker` with zero train/test leakage).

---

## 2. Fundamental Architectural Gap Analysis

### 2.1 Can Pure-Stdlib Parametric Synthesis Meet "Natural & Fluent" + "Cross-Lingual Few-Shot Cloning"?

**Assessment: No — it hits structural acoustic, prosodic, and alignment ceilings.**  
Even after Phase 1 (`exp_v5_timbre`) adds speaker F0, LTAS spectral tilt, glottal pulse shaping (LF/Rosenberg + HNR aspiration), and LPC pole-tracking trajectories:

1. **Band-Limited & All-Pole Acoustic Ceiling:** `vocal.py` sums 26 harmonics with fixed phase (`0.39 * h`). At a male F0 of 110 Hz, 26 harmonics stop at **2.86 kHz** (5.72 kHz at 220 Hz female), leaving upper-band air/frication unmodeled. Nasals (`m, n, ŋ`), laterals (`l`), and rhotics (`r, ɹ`) require spectral zeros (anti-resonances), while obstruents require shaped multi-band turbulence rather than a 1-tap LP/HP noise mix.
2. **Timbre Identifiability Breakdown:** In v4 LOO-CV (`speaker_params.json`), scalar formant scaling yielded **0.6682 Bark** median pooled error vs **0.6531 Bark** speaker-blind baseline (`-0.0151 Bark`, worse than baseline; even same-speaker oracle was `0.6974 Bark`). While LTAS tilt separated `SSB0502` (`-2.71 dB/oct`) from `SSB1593` (`-4.41 dB/oct`), human speaker identity also depends on F3–F5 vocal-tract resonances (blocked by +158..+420 Hz LPC bias), sub-glottal/glottal open-quotient dynamics, band-wise aperiodicity, and micro-prosody—none of which can be reliably inverted from 5–10 s of arbitrary audio with 22-order LPC + 32-utterance heuristics.
3. **Cross-Lingual Phonotactic & Alignment Failure:** Extending `vocal.py` to English requires rewriting `V`/`FINALS`/`FRIC`/`STOP` for ~44 English phonemes, voiced obstruents, syllabic consonants, and `CCVCC` clusters, plus replacing `align.py`'s CV energy-dip splitter (2.6% high confidence) with a true neural/HMM forced aligner—without solving the robotic harmonic buzziness.

**Role of Pure-Stdlib Parametric Synthesis Going Forward:** Keep `exp_v5_timbre` as a **falsifiable DSP lower bound & interpretable diagnostic baseline** (measuring exact LTAS/F0/HNR/MCD contributions), not as the production synthesis backend for cross-lingual voice cloning.

### 2.2 Local Environment Reality Check (Observed)

From direct environment inspection on `D:\Anaconda\python.exe` (Python `3.13.9`):
- **Installed:** `numpy 2.3.5`, `scipy 1.16.3`, `numba 0.62.1`, `scikit-learn 1.7.2`, `huggingface_hub 1.30.0`, `pip 25.3`.
- **Missing:** `torch`, `torchaudio`, `transformers`, `librosa`, `soundfile`, `pyworld`, `pypinyin`, `g2p_en`, `phonemizer`, `onnxruntime`, `ffmpeg`, `espeak`, `espeak-ng`.
- **GPU Hardware:** `NVIDIA GeForce RTX 4080 Laptop GPU`, `12282 MiB` total VRAM (`10999 MiB` free).
- **Important Packaging Risk:** `conda` is absent from PATH (and `conda.exe` fails with a DLL error), while **`uv 0.12.3`** is installed (`C:\Users\MECHREVO\.local\bin\uv.EXE`). Furthermore, Python `3.13.9` frequently lacks prebuilt wheels for older audio/TTS packages (e.g., `pyworld`, `TTS`/`coqui`) and Windows C-extension builds (notably `pyworld` and `CosyVoice2` text-normalization `WeTextProcessing`/`tn`) may fail without MSVC build tools or wheel fallbacks. Any ML/DSP phase MUST use `uv venv --python 3.11` (or `3.10`) to isolate dependencies and verify wheel availability up front.

### 2.3 Feasible Cross-Lingual Architecture Components (12 GB VRAM Target)

#### A. Cross-Lingual G2P & Phoneme/Prosody Frontend (`ZH <-> EN`)
1. **Unified Phonological Representation (IPA + Prosody Tags):**
   - **Mandarin path:** Word segmentation + polyphone disambiguation (`pypinyin` + `jieba` or `g2pw`) -> Pinyin onset/final/tone (`T1..T5`) -> deterministic Pinyin-to-IPA mapping (preserving apical `/ʐ̩, ɹ̩/` and tonal contour tags attached to vowels).
   - **English path:** CMUdict + OOV neural/rule G2P (`g2p_en` or `misaki`) -> ARPAbet + lexical stress (`0, 1, 2`) -> deterministic ARPAbet-to-IPA mapping (eliminating external `espeak-ng` binary friction on Windows).
   - **Code-switching boundary:** Language-ID tag per token (`zh` vs `en`), normalized punctuation pauses, and clause prosody intent tags (`statement`, `question`, `continuation`, `exclamation`, `focus`) compatible with `auto_rules.py`'s IR design.
   - *(Note: If a zero-shot neural model with native BPE/text frontend such as F5-TTS or CosyVoice2 is used in Path C, the unified IPA frontend still serves two critical roles: (1) deterministic cross-lingual pronunciation normalization for heteronyms/loanwords/code-switching, and (2) phoneme-aligned diagnostic evaluation against `SSB` and English reference transcripts.)*

#### B. Speaker Representation (Few-Shot Timbre + Prosody)
- **Neural Speaker Embedding (Timbre Identity):** Pretrained 192-d / 256-d speaker verification encoder (e.g., `CAM++` / `WeSpeaker` ResNet / `SpeechBrain ECAPA-TDNN`, ~20–80 MB, <0.5 GB VRAM) to extract reference embeddings from 3–10 s prompts AND to score objective speaker cosine similarity without using the synthesis model's own encoder (avoiding circular evaluation).
- **Acoustic Prompt Tokens + Interpretable DSP Descriptors:** Combine neural reference audio/mel/codec prompting (which captures F3–F5, spectral envelope, and glottal micro-structure) with explicit v4/v5 DSP telemetry (`F0_median`, `F0_IQR_st`, `LTAS_tilt_dB_oct`, `HNR_dB`, `syllables_per_sec`) for failure diagnosis.

#### C. Synthesis & Vocoding Backends (Feasible Paths Comparison)

| Option | Concrete Stack | 12 GB VRAM Fit | Strengths | Failure Modes to Watch |
|---|---|---|---|---|
| **Path A: Enhanced DSP / WORLD Vocoder** | `pyworld` (Harvest + CheapTrick + D4C) or custom numpy/scipy harmonic+aperiodic vocoder + statistical/small-MLP acoustic predictor | CPU only (<0.1 GB) | Zero black-box magic; separates F0, spectral envelope, and aperiodicity cleanly. | Cannot synthesize fluent unseen EN/ZH coarticulated speech from 32 sentences; spectral envelope warping breaks across gender/wide F0 shifts. |
| **Path B: Modular Cross-Lingual TTS -> Few-Shot VC** | Base bilingual TTS (`Kokoro-82M` ONNX/PyTorch or `VITS`/piper ZH+EN) -> Few-shot Voice Conversion (`Seed-VC` / `FreeVC` / `kNN-VC` with `WavLM` features + `BigVGAN`/`HiFi-GAN`) | **Yes** (~2–4 GB VRAM FP16) | Cleanly decouples **fluency/G2P** (base TTS) from **timbre** (VC); strong diagnostic control over content vs timbre leakage. | Base TTS prosody/accent can leak into target speaker; two-stage artifacts at unvoiced fricatives. |
| **Path C: Native Zero-Shot Cross-Lingual TTS** | Flow-matching / LM+Flow TTS (`CosyVoice2-0.5B`, `F5-TTS`, or `Fish-Speech-1.5` in FP16) | **Yes** (~3–7 GB VRAM FP16) | Highest naturalness ceiling; native zero-shot cloning from 3–10 s audio across ZH, EN, and code-switching. | Heavy dependency footprint; risk of hallucinated words/duration drift on short/noisy prompts or tone errors on Mandarin sandhi. |

**Recommended Strategy:** Run **Path B (Modular TTS + Few-Shot VC)** and **Path C (Zero-Shot Cross-Lingual Flow/LM TTS)** as competing Phase-2 hypotheses against the **Phase-1 DSP baseline (`exp_v5_timbre`)**, evaluated on the exact same frozen benchmark and independent speaker-verification / ASR / blind-listening harness.

---

## 3. Strictly Required Corpus Expansion

The current corpus (**2 female Mandarin speakers, `SSB0502` @ 223.8 Hz and `SSB1593` @ 207.3 Hz, 32 extended utterances = 81.2 s**) is structurally insufficient to test voice cloning or cross-lingual synthesis:

1. **Cross-Gender & Multi-Timbre Mandarin Reference Set (AISHELL-3 / OpenSLR-93, Apache-2.0):**
   - Keep `SSB0502` and `SSB1593` as legacy continuity anchors. Note that `_transcripts/SSB0502.json` (`476` entries) and `SSB1593.json` (`488` entries) contain text/pinyin metadata only — local audio on disk is limited to the 32 WAVs in `extended/` (plus 9 crops in `bank/` and `test_15930257.wav`), so additional `SSB0502`/`SSB1593` utterances for clean disjoint prompt vs. held-out splits must be fetched from `fbngrm/mandarin-sentence-audio` / OpenSLR-93.
   - Add **4 diverse Mandarin speakers** (minimum **6 total**):
     - `2 Male` (1 deep baritone ~95–125 Hz, 1 mid-tenor ~135–165 Hz).
     - `2 Female` with contrasting spectral tilt / breathiness (1 bright/modal, 1 softer/breathy).
   - Minimum **15 utterances/speaker** (~45–60 s/speaker): 3 utterances reserved for few-shot prompt pools (3 s / 10 s), 12 held-out for intra-lingual oracle/similarity checks.
2. **English Reference Set (LibriTTS-R or VCTK clean subset, CC-BY-4.0):**
   - Add **4 English speakers** (`2 Male`, `2 Female`, native speakers, 24 kHz mono WAV, clean studio SNR > 25 dB), **15 utterances/speaker**.
   - Needed to establish: (a) native EN oracle speaker-similarity & ASR-WER ceilings, and (b) bidirectional cross-lingual cloning (`EN prompt -> ZH speech` as well as `ZH prompt -> EN speech`).
3. **Frozen Evaluation Text Suite (Zero Leakage, Fixed Seed):**
   - **10 Mandarin sentences:** tones 1–5, T3 sandhi (`你好` / `老虎`), `"不"/"一"` sandhi, polar/wh questions, statements, fricatives (`s/x/sh/h`), retroflex/apical (`ir/iz`).
   - **10 English sentences:** dental fricatives (`/θ, ð/`), sibilants (`/s, ʃ, z, ʒ/`), consonant clusters (`str-`, `-sts`), lexical stress minimal pairs (`REcord` vs `reCORD`), questions vs statements.
   - **6 Code-Switched (`ZH+EN`) sentences:** e.g., `"我今天用 Python 跑了一个 baseline。"` / `"Please check the 银行卡 balance before Tuesday."`.

---

## 4. Concrete, Falsifiable Phase-2 Experimental Roadmap

### Milestone 0: Environment, Reproducible Audio Harness & Corpus Expansion (Week 1)
- **Deliverables:**
  1. Isolated Python 3.11 (or 3.10) environment created via `uv venv --python 3.11` (since `conda.exe` is broken on this host and `uv 0.12.3` is available) with verified `torch` (CUDA 12.x on RTX 4080), `torchaudio`, `soundfile`, `librosa`, `pypinyin`, `g2p_en`, `onnxruntime-gpu`, plus fallback verification for Windows C-extension risks (`pyworld` wheel or pure-scipy WORLD-like harmonic/aperiodic fallback; regex/pure-Python TN fallback if `WeTextProcessing`/`tn` fails to build) — without touching `D:\Anaconda\python.exe` stdlib tests.
  2. Frozen evaluation dataset manifest in `exp_v5_arch/data_manifest.json` (6 ZH + 4 EN speakers, SHA-256 checksums, strict disjoint `prompt` vs `heldout_test` utterance IDs).
  3. Independent evaluation harness:
     - **Objective Speaker Similarity:** Pretrained `CAM++` / `ECAPA-TDNN` cosine similarity (calibrated on same-speaker real pairs vs different-speaker real pairs to establish Equal Error Rate threshold).
     - **Intelligibility:** `faster-whisper` / `FunASR (Paraformer-zh)` CER (ZH) and WER (EN) on synthesized outputs.
     - **Acoustic Fidelity:** F0 RMSE / Pearson $r$, LTAS tilt error (dB/oct), 13-dim MCD (dB), active speech rate, and waveform sanity checks (24 kHz, peak $< 0.99$, no NaN/silence).
- **Pass / Fail Gate:**
  - **Pass:** Independent SV encoder separates the 6 ZH + 4 EN real held-out speakers with **EER <= 5%** (and cleanly separates `SSB0502` vs `SSB1593` on held-out utterances), and ASR achieves **CER <= 8% (ZH)** / **WER <= 10% (EN)** on real ground-truth recordings.
  - **Fail / Abort:** If real-data SV calibration shows EER > 10% or ASR fails on clean ground truth, fix metric calibration before testing any TTS backend.

### Milestone 1: Unified Cross-Lingual Frontend & `engine` IR Bridge (Week 1–2)
- **Deliverables:**
  - `crossling_frontend.py`: Language-tagging tokenizer (`zh`, `en`, `punct`), `pypinyin`+lexicon -> IPA (`zh`) and CMUdict/`g2p_en` -> IPA (`en`), outputting an extended `units` IR (`token`, `lang`, `ipa_phones`, `tone_or_stress`, `duration_ms`, `pitch_st`, `pause_after_ms`) while preserving backward compatibility with `engine.py` unit schemas.
- **Minimal Viable Test Cases:**
  - `MV-FE-1`: Mandarin sandhi regression (`"你好"`, `"不对"`, `"不去"`, `"第一"`, `"银行行长"`).
  - `MV-FE-2`: English OOV + cluster + stress (`"strengths"`, `"record"`, `"PyBotTTS"`).
  - `MV-FE-3`: Mixed code-switch (`"今天我们测试 cross-lingual voice cloning。"`).
- **Pass / Fail Gate:**
  - **Pass:** 100% phoneme accuracy on a 50-item curated ZH/EN/CS golden G2P test set; 0 crashes on unknown Unicode/punctuation.
  - **Negative Result Criterion:** If code-switched boundaries misclassify > 5% of ASCII acronyms or polyphones, fix frontend segmentation before acoustic synthesis trials.

### Milestone 2: DSP Ceiling Verification (`exp_v5_timbre` + WORLD Vocoder Probe) (Week 2)
- **Experiment:** Compare (a) `v4 vocal.py` baseline, (b) `exp_v5_timbre` (tilt + F0 + LF glottal pulse + HNR + pole trajectories), and (c) `pyworld` analysis-resynthesis & linear spectral/F0 speaker adaptation across the 6 ZH speakers (especially cross-gender M<->F).
- **Pass / Fail Gate (Falsifies Pure-Parametric Route):**
  - **Expected Outcome / Negative Result Hypothesis:** Pure-parametric synthesis (`vocal.py` / `v5`) will show **MCD > 7.5 dB**, **ASR CER > 25%**, and fail human blind A/B naturalness against real speech (`win-rate < 20%`). `pyworld` cross-gender conversion will degrade dramatically (`SV cosine < 0.45` or severe buzziness/formant-shift artifacts).
  - **Decision:** Confirming this negative result formally closes pure-parametric synthesis as a primary voice-cloning engine and justifies Milestone 3 neural backends. (If `v5` unexpectedly achieves CER < 15% and SV cosine > 0.65 cross-gender, re-evaluate DSP hybrid viability.)

### Milestone 3: Competing Neural Backends — Modular TTS+VC vs. Unified Zero-Shot TTS (Week 3–4)
- **Candidates (12 GB VRAM FP16):**
  - **Candidate B (Modular):** Bilingual Base TTS (e.g., `Kokoro-82M` ZH+EN) -> Few-shot VC (`Seed-VC` or `kNN-VC`).
  - **Candidate C (Unified Zero-Shot):** `CosyVoice2-0.5B` and/or `F5-TTS` (both support ZH/EN zero-shot cloning from 3–10 s prompts within 12 GB VRAM).
- **Minimal Viable Cross-Lingual Matrix (evaluated at 3 s and 10 s prompt durations):**
  1. **Intra-lingual ZH->ZH:** 6 speakers × 10 ZH sentences (`60` clips), including `SSB0502` & `SSB1593`.
  2. **Intra-lingual EN->EN:** 4 speakers × 10 EN sentences (`40` clips).
  3. **Cross-lingual ZH->EN (clone Mandarin speaker speaking English):** 6 ZH prompts × 10 EN sentences (`60` clips).
  4. **Cross-lingual EN->ZH (clone English speaker speaking Mandarin):** 4 EN prompts × 10 ZH sentences (`40` clips).
  5. **Code-Switching (ZH/EN prompts -> 6 CS sentences):** `60` clips.
- **Quantitative Pass / Fail & Negative Result Criteria:**

| Metric | Pass Target (ZH->ZH / EN->EN) | Pass Target (Cross-Lingual ZH<->EN) | Explicit Failure / Negative Result Trigger |
|---|---|---|---|
| **Intelligibility (ASR CER / WER)** | CER ≤ 8% (ZH), WER ≤ 10% (EN) | CER ≤ 12% (EN->ZH), WER ≤ 15% (ZH->EN) | **FAIL** if CER > 18% or WER > 22% (tone collapse / heavy foreign accent / dropped words). |
| **Timbre Similarity (Indep. SV Cosine)** | ≥ 0.65 (and > same-gender impostor p95) | ≥ 0.55 (and > cross-lingual random pair p95) | **FAIL** if cross-lingual cosine drops below same-gender impostor median (timbre lost across language switch -> accent/speaker entanglement). |
| **F0 & Spectral Match** | F0 median err ≤ 1.5 st; LTAS tilt err ≤ 1.5 dB/oct | F0 median err ≤ 2.0 st; LTAS tilt err ≤ 2.0 dB/oct | **FAIL** if cloned male voice drifts > +3 st toward female pitch or vice versa (source prosody leakage in VC). |
| **Runtime (RTX 4080 12 GB)** | RTF ≤ 0.5, Peak VRAM ≤ 8 GB | RTF ≤ 0.6, Peak VRAM ≤ 8 GB | **FAIL** if OOM (> 11 GB) or RTF > 1.5. |
| **Blind Listening (Human A/B & MOS)** | ≥ 3.8 / 5.0 naturalness, > 70% win vs DSP | ≥ 3.5 / 5.0 fluency & accent acceptability | **FAIL** if listeners detect metallic vocoder buzz, skipped syllables, or Mandarin tone-1 flattening in EN->ZH. |

### Milestone 4: Diagnosis, Integration & Optional TUI Hook (Week 4–5)
- **Deliverables:**
  1. Failure-mode report documenting every negative result (e.g., EN->ZH tone-3 errors, short 3 s vs 10 s prompt degradation, cross-gender pitch leakage in VC vs flow-matching).
  2. Only if a candidate passes Milestone 3 gates: wrap behind a clean backend interface matching PyBotTTS's offline WAV generation workflow (`tui.py` / CLI pluggable backend), keeping `v2`–`v4` directories untouched as reproducible historical baselines.

---

## 5. Immediate Next Actions

1. **Await Phase-1 `exp_v5_timbre` verdict** (`timbre-lab` + `verifier`) to lock in the exact pure-DSP ceiling numbers on `SSB0502` vs `SSB1593`.
2. **Approve Milestone 0 setup** (isolated Python 3.10/3.11 ML env with PyTorch CUDA on RTX 4080 + 6-ZH / 4-EN Apache/CC-BY reference corpus download) before writing any neural synthesis code.
