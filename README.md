# PyBot-TTS (PyBot-Larynx)

**Zero-neural-weight (`0.0 MB` ML weights)**, pure physics + explicit deterministic acoustic rule engine for **24 kHz** speech synthesis in **Mandarin (`zh`)**, **English (`en`)**, **ZH+EN code-switching**, **expressive speech & inline vocalizations**, **zero-shot voice cloning**, and an **extensible unified IPA multilingual architecture** (Stage 13, ready for `ja`/`ko` language plugins).

> Read [docs/architecture-and-history.md](docs/architecture-and-history.md) for the full technical architecture and a concise retrospective of early phoneme-concatenation experiments (`v2`–`v19`).

---

## Key Capabilities

- **Zero ML / Zero Neural Weights**: All acoustic voicebanks are compressed deterministic NumPy archives (`allow_pickle=False`) paired with classical signal processing (513-bin TrueHarmEnv spectral envelopes + 6-band periodic/aperiodic `dynamic_coherent` glottal excitation).
- **Stage 13 Native Zero-Mapping Bilingual Engine (`exp_20_pybot_larynx/multilingual/`)**:
  - Stores Mandarin (`88,284` native tokens, `39` IPAs) and English (`41,521` tokens including `19,935` build-time studio `LJSpeech -> SF_EN` tokens) in one unified same-speaker IPA acoustic bank (`knowledge_sf_native_stage13.npz`, `22.13 MB`).
  - Synthesizes ZH, EN, and mid-clause code-switches in **one cross-lingual Viterbi DP pass** with **0 runtime cross-speaker projections** (`0.0%` cross-language fallback rate on studio EN) and **4 universal IPA articulatory rules** (`IPAGlottalAntiBuzzRule`, `IPANasalAntiresonanceRule`, `IPAObstruentCrispnessRule`, `IPACoarticulationAndSwitchRule`).
  - Supports **pluggable `LanguageFrontend` plugins** (`register_language_frontend`) for extending to new languages such as Japanese (`ja`) or Korean (`ko`) without modifying core acoustic selection or vocoder rules.
- **Zero-Shot Voice Cloning & Expressive Control (`exp_20_pybot_larynx/clone/`, `expressive/`)**:
  - Extract a target speaker's spectral envelope tilt, formant scaling (VTLN), coherent glottal phase, and pitch register from seconds of clean reference WAV audio.
  - Inline tags such as `[开心]`, `[激动]`, `[悲伤]`, `[耳语]`, `[笑]`, `[叹气]`, `[吸气]`, `[停顿 400ms]`, and `*focus*`.
- **Local WebUI Voice Studio (`webui.py`, `pybot_webui/`)**: Pure Python stdlib HTTP server + single-page studio UI.

---

## Verified Stage 13 Metrics

Measured on unseen held-out same-speaker bilingual recordings (`com_SF_ce996`–`ce1001`) and benchmark prompts:

| Benchmark Metric | Stage 12 (Projected Pivot) | **Stage 13 (Native Zero-Mapping)** |
|---|---:|---:|
| Held-out Mel-CD median (dB, lower is better) | `7.918` | **`6.655`** |
| Held-out HNR median (dB, Human = `14.962`) | — | **`14.782`** |
| Pure-EN / Mixed / Pure-ZH HNR median (dB) | — | **`13.194` / `13.766` / `14.160`** |
| Cross-speaker runtime projections | `1–2` | **`0`** (`1` optional single-exit clone) |
| Studio-EN cross-language fallback (`ipa_shared`) | — | **`0.0%`** (`87.39%` `ctx_pns` triphone hits) |
| Synthesis speed (median RTF, CPU single-thread) | — | **`0.0332`** (~30x real-time) |

---

## Quick Start

### 1. Environment

Python **3.10+** with `numpy` and `scipy`:

```bash
pip install -r requirements.txt
```

*(Note: this repository tracks the prebuilt inference voicebanks so you can synthesize immediately after cloning without downloading or re-aligning raw corpora.)*

### 2. Launch WebUI Studio

Windows:

```bat
run_webui.cmd
```

Cross-platform CLI:

```bash
python webui.py --host 127.0.0.1 --port 7860
```

### 3. Python API (Stage 13 Native Zero-Mapping)

```python
from exp_20_pybot_larynx import SR
from exp_20_pybot_larynx.multilingual import synthesize_multilingual_native
from exp_v14_fluent_zh_tts.eval_v14 import _write_pcm16_wav

wav, ctrl, meta = synthesize_multilingual_native(
    "今天用 PyBot 测试 this smooth cross lingual speech，表现非常自然。",
    seed=2026,
)
_write_pcm16_wav("out_stage13.wav", wav, SR)
```

---

## Verification Suites

```bash
python -m exp_20_pybot_larynx.verify_stage10_bilingual
python -m unittest exp_20_pybot_larynx.tests.test_stage13_multilingual exp_20_pybot_larynx.tests.test_stage10_bilingual
python -m unittest pybot_webui.test_webui
```

---

## Repository Structure

```text
PyBot-TTS/
├── webui.py                 # WebUI CLI entrypoint
├── run_webui.cmd            # Windows one-click launcher
├── pybot_webui/             # Stdlib HTTP server, presets, text splitter & WebUI frontend
├── exp_20_pybot_larynx/     # Active PyBot-Larynx engine (vocoder, Stage 7/10/12/13, clone, expressive)
│   ├── multilingual/        # Stage 13 zero-mapping unified IPA (`zh+en`, `ja`/`ko` plugin-ready)
│   ├── crossling/           # Stage 10-12 EN & cross-lingual prosody / acoustics
│   ├── acoustics/, rules/, frontend/, larynx/, clone/, expressive/, tests/
├── legacy/                  # Archived v2-v15 shared DSP/G2P/index helpers imported on sys.path
└── docs/                    # Architecture overview & concatenation retrospect
```

---

## Tracked Voicebanks & Artifacts

- `exp_20_pybot_larynx/multilingual/knowledge_sf_native_stage13.npz` (`22.13 MB`) — Stage 13 unified `zh+en` native IPA voicebank
- `exp_20_pybot_larynx/acoustics/knowledge_stage7.npz` (`73.94 MB`) — Stage 7 Mandarin `BZNSYP` voicebank
- `exp_20_pybot_larynx/crossling/knowledge_en_stage10.npz` (`17.02 MB`) — Stage 10 English acoustic bank
- `exp_20_pybot_larynx/crossling/knowledge_en_sf_stage12.npz` (`7.72 MB`) — Stage 12 SF-pivoted English bank
- `legacy/exp_v15_bznsyp_tts/bznsyp_speaker_assets.pkl` (`9.34 MB`) — BZNSYP reference glottal & speaker assets
