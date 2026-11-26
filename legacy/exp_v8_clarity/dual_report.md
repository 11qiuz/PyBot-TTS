# Phase 4 Part 2: Dual-Donor EN+ZH UTAU Bank + Decoupled V2 Cloner (`exp_v8_clarity`)

## 1. Architecture (100% Non-Neural, Zero Weights)
- **`exp_v8_clarity/en_utau_bank.py`**:
  1. **Zero-Neural DP Forced Aligner (`align_ipa_sequence`)**: Aligns `g2p_en`/CMUdict IPA phoneme sequences over 5 ms `VocoderFeatures` using acoustic cues `(voiced, HF/LF 2–8 kHz vs 80–1000 Hz, relative RMS)` and `IPA_SPECS` duration priors — extracting **`5,094` full-res 5 ms English phone tokens** across **`37` unique English IPAs** from **`140`** CMU Arctic training utterances (`slt[0:30]`, `bdl[0:30]`, `clb[0:40]`, `rms[0:40]`), strictly holding out `20` utterances (`slt[30:40]` & `bdl[30:40]`).
  2. **Unified `DualDonorBank`**: Pairs Mandarin `FullResBank` (`extended/`) with English `EnPhoneToken` bank (`exp_v8_data/corpus/manifest.json`) and unified `exp_v8_timbre_clone.timbre_cloner_v2` (`VTLN + V/UV-decoupled LTAS + HNR`) profiles.
  3. **Manner-Aware UTAU Rendering (`render_dual_utau_plan`)**: Target-First phone selection + `0 ms` V/UV obstruent seams / `10 ms` vowel joins, normalizing token energy to `-6.9` nepers and capping peaks at `0.89`.

## 2. Intra-Lingual `EN -> EN` Results (`n=20` Held-Out `slt[30:40]` & `bdl[30:40]`, Zero Leakage)
Target profiles extracted strictly from **3-shot prompts** (`[0:3]` per speaker). `v8_utau_en_cross_donor` excludes the entire target speaker from the bank (`bank_no_bdl` / `bank_no_slt`).

| Mode | Sanity | Mel-LSD (dB) | MCD [CMVN] (dB) | Spectral Contrast (dB) | `uv_HF/LF` (dB) | `HF_p95` Transient (dB) | `tilt_err` (dB/oct) | `F0_err` (st) | CPU RTF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `real_en` (ground truth) | 20/20 | `0.000` | `0.000` [`0.000`] | **`31.964`** | **`+9.573`** | `7.848` | `0.000` | `0.000` | — |
| `v7_warped_en` (ZH IPA warp) | 20/20 | `7.904` | `31.202` [`29.215`] | `28.814` | **`-8.429`** | `3.373` | `1.433` | `1.066` | `0.058` |
| **`v8_utau_en_target_first`** | **20/20** | **`7.652`** | **`29.077`** [`29.025`] | **`32.852`** | **`+3.976`** | **`8.915`** | **`0.672`** | `0.996` | **`0.050`** |
| **`v8_utau_en_cross_donor`** | **20/20** | `7.872` | `30.302` [`29.409`] | `31.217` | **`+3.643`** | `9.646` | `0.934` | `0.952` | `0.056` |

- **Real EN Concatenation Coverage**: **`686 / 686` (`100.0%`)** of held-out English phonemes used real English phone tokens (`en_target_phone=686` in target-first; `en_donor_phone=686` in cross-donor; **`0` synth-fill**).
- **Clarity & Tilt Gains over `v7_warped_en`**: `uv_HF/LF` consonant crispness jumps **`+12.405 dB`** (`-8.429 -> +3.976 dB`), `spectral_contrast` gains **`+4.038 dB`** (`28.814 -> 32.852 dB`), `MCD` drops **`-2.125 dB`**, and `tilt_err` drops **`-53%`** (`1.433 -> 0.672 dB/oct`).

## 3. Cross-Lingual `ZH <-> EN` & Male Mandarin `SSB0073` Results
Targets use 3-shot `TimbreProfileV2` (`SSB0502`, `SSB1593`, new Male Mandarin `SSB0073` from `manifest.json`, `slt`, `bdl`), evaluated on `5` held-out EN sentences and `5` held-out ZH sentences (with held-out ZH utterance IDs excluded from `zh_bank`).

| Cross-Lingual Case | Sanity | Target -> Out F0 (Hz) | `F0_err` (st) | Target -> Out Tilt (dB/oct) | `tilt_err` | `HNR_err` (dB) | Contrast (dB) | `uv_HF/LF` (dB) | Sources | CPU RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `SSB0502_speaks_EN` | 5/5 | `224.6 -> 225.0` | `0.034` | `-3.14 -> -3.58` | `0.44` | `0.99` | `32.89` | `+7.77` | `215 en_donor` (`100%`) | `0.0586` |
| `SSB1593_speaks_EN` | 5/5 | `203.7 -> 203.7` | `0.076` | `-4.01 -> -4.95` | `0.94` | `0.53` | `31.91` | `+4.25` | `215 en_donor` (`100%`) | `0.0601` |
| **`SSB0073_M133_speaks_EN`** | 5/5 | `134.5 -> 134.9` | **`0.053`** | `-6.08 -> -6.21` | **`0.37`** | `1.36` | `32.50` | `+4.28` | `215 en_donor` (`100%`) | `0.0535` |
| **`SSB0073_M133_speaks_ZH`** | 5/5 | `134.5 -> 135.0` | **`0.119`** | `-6.08 -> -5.96` | **`0.28`** | `2.33` | `35.08` | `-6.64` | `37 zh_donor + 17 concat` | `0.0423` |
| `slt_F189_speaks_ZH` | 5/5 | `185.2 -> 185.2` | `0.053` | `-3.76 -> -4.84` | `1.09` | `3.76` | `35.08` | `-4.36` | `37 zh_donor + 17 concat` | `0.0421` |
| `bdl_M122_speaks_ZH` | 5/5 | `127.2 -> 126.3` | `0.132` | `-5.05 -> -6.35` | `1.30` | `1.73` | `33.47` | `-5.81` | `37 zh_donor + 17 concat` | `0.0397` |

## 4. Honest Negative Results & Remaining Gaps
1. **Remaining `5.6 dB` English `uv_HF/LF` Gap (`+3.976` vs real `+9.573 dB`)**: Our zero-neural DP aligner (`align_ipa_sequence`) relies only on coarse `(voiced, HF/LF, RMS)` emissions without MFA/neural phone boundaries, so `±10–20 ms` stop/vowel boundary drift occasionally pulls adjacent voiced energy into fricative segments, and HNM resynthesis softens >8 kHz sibilant peaks.
2. **Cross-Corpus `HNR` & `Tilt` Residual (`HNR_err = 3.76 dB` on `slt_F189_speaks_ZH`, `tilt_err = 1.30 dB/oct` on `bdl_M122_speaks_ZH`)**: When CMU Arctic speakers render `AISHELL-3` Mandarin syllables (`zh_bank`) across corpus recording conditions and cross-gender `VTLN`, utterance-level consonant-to-vowel ratios cause residual `1–3.8 dB` HNR/tilt offsets vs the 3-shot English prompt.
3. **Intra-Speaker Utterance-to-Utterance `F0_err ~ 0.95–1.0 st`**: Comparing 3-shot prompt F0 (`[0:3]`) against individual held-out English test utterances (`[30:40]`) reflects natural ~1 st intra-speaker sentence pitch variation (whereas target-profile tracking `target -> out F0` is `0.034–0.132 st`).

## 5. Sample WAVs & Tests
- **Loudness-Matched Samples (`exp_v8_clarity/samples/`, `18/18` `sanity_ok=True`)**:
  - `AB_EN_slt_0030_real_v7warped_v8utau_v8xdom.wav`, `AB_EN_bdl_0030_real_v7warped_v8utau_v8xdom.wav`
  - `AB_CROSS_SSB0502_zhRef_speaks_EN.wav`, `AB_CROSS_bdl_M122_enRef_speaks_ZH.wav`
  - `AB_CROSS_SSB0073_M133_zhRef_speaks_EN.wav`, `AB_CROSS_SSB0073_M133_zhRef_speaks_ZH.wav`
- **Artifacts**: `exp_v8_clarity/en_utau_bank.py`, `exp_v8_clarity/dual_results.json`, `exp_v8_clarity/test_clarity.py`, `exp_v8_clarity/dual_report.md`.
- **Tests**: `D:\Anaconda\python.exe -B D:\ZaWu\PyBotTTS\PyBot-TTS\exp_v8_clarity\test_clarity.py -v` -> **7/7 passed (`21.345s`)** (`18/18` WAVs in `exp_v8_clarity/samples/` pass `check_wav_sanity`).
