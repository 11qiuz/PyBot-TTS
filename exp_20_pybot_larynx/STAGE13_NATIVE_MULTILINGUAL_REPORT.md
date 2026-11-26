# Stage 13 Native Multilingual Zero-Mapping Report

- **Artifact**: `knowledge_sf_native_stage13.npz` (`15.083 MB`, `ml_weights_mb=0.0`, `allow_pickle=False`)
- **Inventory**: `88284` native ZH tokens (`39` IPAs) + `45734` EN tokens (`+ 45734` studio EN->EN LJSpeech->SF_EN)
- **Static Telemetry**: `max_voiced_bap0=0.319`, `max_voiced_bap1=0.602`, `en_studio_ipa_shared_fallbacks=0`
- **Held-out 6 utterances (`com_SF_ce996..1001`)**: S13 Native Mel-CD median = **7.006 dB** (Stage 12 SF = 7.918 dB), HNR median = **14.332 dB** (Human = 14.962 dB)
- **Native EN Held-out 6 utterances (`en_stage10_heldout`)**: Mel-CD median = **8.061 dB**, `q8_24` = **11.511**, HNR = **13.678 dB**, F0 IQR = **4.12 st**
- **Showcase medians**: Mixed HNR=`14.412 dB`, Pure-EN HNR=`13.824 dB` (`q8_24=11.291`), Pure-ZH HNR=`14.081 dB`, RTF=`0.0399`, All Sanity OK=`True`
- **DP lookup tier distribution**: `{'ctx_base': 0.0, 'ctx_p': 0.13, 'ctx_pn': 1.69, 'ctx_pns': 58.65, 'ctx_ps': 39.53, 'fallback': 0.0, 'ipa_shared': 0.0}`

| Utterance | Stage 12 SF Mel-CD (dB) | Stage 13 Native Mel-CD (dB) | Human HNR (dB) | Stage 13 HNR (dB) | RTF |
|---|---:|---:|---:|---:|---:|
| `com_SF_ce996` | 7.633 | 6.956 | 15.262 | 13.96 | 0.749 |
| `com_SF_ce997` | 8.06 | 6.92 | 14.901 | 13.522 | 0.0388 |
| `com_SF_ce998` | 7.881 | 6.973 | 15.023 | 15.265 | 0.0361 |
| `com_SF_ce999` | 7.898 | 7.046 | 15.724 | 15.79 | 0.0341 |
| `com_SF_ce1000` | 7.964 | 7.365 | 10.607 | 14.121 | 0.0392 |
| `com_SF_ce1001` | 7.939 | 7.039 | 14.464 | 14.544 | 0.0375 |
