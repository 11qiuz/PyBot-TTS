# Stage 13 Native Multilingual Zero-Mapping Report

- **Artifact**: `knowledge_sf_native_stage13.npz` (`22.13 MB`, `ml_weights_mb=0.0`, `allow_pickle=False`)
- **Inventory**: `88284` native ZH tokens (`39` IPAs) + `41521` EN tokens (`+ 19935` studio EN->EN LJSpeech->SF_EN)
- **Static Telemetry**: `max_voiced_bap0=0.319`, `max_voiced_bap1=0.602`, `en_studio_ipa_shared_fallbacks=0`
- **Held-out 6 utterances (`com_SF_ce996..1001`)**: S13 Native Mel-CD median = **6.655 dB** (Stage 12 SF = 7.918 dB), HNR median = **14.782 dB** (Human = 14.962 dB)
- **Showcase medians**: Mixed HNR=`13.766 dB`, Pure-EN HNR=`13.194 dB`, Pure-ZH HNR=`14.16 dB`, RTF=`0.0332`, All Sanity OK=`True`
- **DP lookup tier distribution**: `{'ctx_base': 0.0, 'ctx_p': 0.13, 'ctx_pn': 5.85, 'ctx_pns': 87.39, 'ctx_ps': 6.63, 'fallback': 0.0, 'ipa_shared': 0.0}`

| Utterance | Stage 12 SF Mel-CD (dB) | Stage 13 Native Mel-CD (dB) | Human HNR (dB) | Stage 13 HNR (dB) | RTF |
|---|---:|---:|---:|---:|---:|
| `com_SF_ce996` | 7.633 | 6.645 | 15.262 | 14.41 | 0.6242 |
| `com_SF_ce997` | 8.06 | 6.666 | 14.901 | 13.572 | 0.0354 |
| `com_SF_ce998` | 7.881 | 6.922 | 15.023 | 16.003 | 0.0368 |
| `com_SF_ce999` | 7.898 | 6.803 | 15.724 | 15.506 | 0.0354 |
| `com_SF_ce1000` | 7.964 | 6.43 | 10.607 | 14.082 | 0.0339 |
| `com_SF_ce1001` | 7.939 | 6.54 | 14.464 | 15.155 | 0.0383 |
