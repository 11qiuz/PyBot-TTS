"""Build Stage 9B Speaker-Independent English Articulation, Coarticulation & Rhythm Table.

Zero-ML deterministic extractor:
1. Reads the 800-utterance local CMU-ARCTIC corpus (`exp_v12_data/corpus/manifest_v12.json`)
   and strictly excludes the 20 held-out IDs (`manifest["english"]["heldout_ids"]`, i.e.,
   `slt_0030..0039` & `bdl_0030..0039`) -> 780 training utterances.
2. Runs bootstrapped 3-state GMM-HMM Viterbi forced alignment (`exp_v11_align_gmm`).
3. Strips each speaker's VTLN vocal-tract proportions & LTAS spectral tilt, then aggregates:
   - 5-bin speaker-stripped articulatory profiles (`delta_cep`, `formants_5bin`, voicing, rel_db)
   - Empirical-Bayes shrunk left/right context coarticulation bends (`cep[0:20]` + `F1/F2/F3` shifts)
   - Positional allophones (`l_coda` dark-L, `p/t/k/b/d/ɡ_coda` unreleased coda stops)
   - Corpus-measured durations by stress, pre-fortis vowel clipping, cluster & phrase-final ratios
4. Saves compact `< 1 MB` `en_articulation_table.npz` (`allow_pickle=False`) and summary JSON.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.fft import rfftfreq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_20_pybot_larynx import HOP, N_BINS, N_FFT, SR  # noqa: E402
from exp_20_pybot_larynx.arbiter import NEPER_TO_DB  # noqa: E402
from exp_20_pybot_larynx.clone.adapter import _warp_log_env_pivot  # noqa: E402
from exp_20_pybot_larynx.crossling.en_articulation import (  # noqa: E402
    ART_CLASSES,
    EN_ARTICULATION_PATH,
    EN_ARTICULATION_SUMMARY_PATH,
    N_ART_BINS,
    N_ART_CEP,
    N_COART_CEP,
    VOICELESS_OBSTRUENTS,
    VOWEL_IPAS,
    classify_articulatory_class,
    clear_en_articulation_cache,
    resolve_en_allophone,
)
from exp_20_pybot_larynx.larynx.spectral_decomp import (  # noqa: E402
    decompose_frame_source_filter,
    lsf_to_formants,
)
from exp_20_pybot_larynx.larynx.synth import log_env_to_cep  # noqa: E402
from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS  # noqa: E402
from exp_v6_vocoder import vocoder as v6_voc  # noqa: E402
from exp_v8_clarity import en_utau_bank as v8_en  # noqa: E402
from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10  # noqa: E402
from exp_v10_g2p_and_align.landmark_aligner import score_token_purity  # noqa: E402
from exp_v11_align_gmm.gmm_hmm_aligner import (  # noqa: E402
    _align_with_trained_gmm,
    _train_and_align_gmm,
    _trim_utterance_active,
)

MANIFEST_PATH = REPO_ROOT / "exp_v12_data" / "corpus" / "manifest_v12.json"

_FREQS = rfftfreq(N_FFT, 1.0 / SR)
_HF_MASK = _FREQS >= 1500.0
_ARPA_RE = cf._ARPA_TOKEN_RE
_U_ART = np.clip((np.maximum(_FREQS, 80.0) - 3100.0) / 1000.0, 0.0, 1.0)
_W_ART = 0.5 * (1.0 + np.cos(math.pi * _U_ART))

_ANCHOR_VOWELS = ("i", "a", "ɑ", "u", "o", "æ", "ɛ", "ɔ", "ʌ")
_HELDOUT_CACHE_PATH = Path(
    r"C:\Users\MECHREVO\.gemini\antigravity\brain\f67e0437-e98d-4cb1-bbfd-da72d02478db\scratch\en_heldout_aligned_s9b.npz"
)


def _worker_extract_en_utt(task: tuple[str, str, str, str]) -> dict[str, Any]:
    """Worker: extract 24 kHz features + G2P phone metadata for one CMU-ARCTIC utterance."""
    for _ev in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_ev] = "1"
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    uid, spk, wav_rel, text = task
    x, file_sr = v6_voc.read_wav_mono(REPO_ROOT / wav_rel)
    if file_sr != SR:
        raise ValueError(f"Expected {SR} Hz, got {file_sr} for {uid}")
    s0, s1 = v8_en._trim_active_span(x, sr=SR)
    xa = x[s0:s1]
    feats_full = v6_voc.analyze(xa, sr=SR)
    cs, ce, cf0, cf1 = _trim_utterance_active(xa, feats_full, sr=SR)
    xa_t = xa[cs:ce]
    feats = v6_voc.VocoderFeatures(
        f0_hz=feats_full.f0_hz[cf0:cf1],
        voiced=feats_full.voiced[cf0:cf1],
        log_env=feats_full.log_env[cf0:cf1],
        bap=feats_full.bap[cf0:cf1],
        sr=SR,
        hop_samples=HOP,
        n_samples=len(xa_t),
        n_fft=feats_full.n_fft,
    )

    ir = plan_v10(text)
    ipas: list[str] = []
    meta_phones: list[dict[str, Any]] = []
    units = list(ir.get("units") or [])
    for w_idx, unit in enumerate(units):
        arps = list(unit.get("arpabet") or [])
        p_list = list(unit.get("ipa_phones") or [])
        p_pause = float(unit.get("pause_after_ms", 0.0))
        if len(arps) == len(p_list) and arps:
            pairs: list[tuple[str, str | None, int | None]] = []
            for arp_tok, ipa_p in zip(arps, p_list):
                m = _ARPA_RE.fullmatch(str(arp_tok).strip().upper())
                if m:
                    base, st = m.group(1), m.group(2)
                    stress = int(st) if (st is not None and base in cf.ARPABET_VOWEL_BASE) else None
                else:
                    stress = None
                pairs.append((ipa_p, str(arp_tok), stress))
        else:
            pairs = [(ipa_p, None, (1 if ipa_p in VOWEL_IPAS else None)) for ipa_p in p_list]

        for k, (ipa_p, arp_t, stress) in enumerate(pairs):
            ipas.append(ipa_p)
            meta_phones.append({
                "ipa": ipa_p,
                "arpabet": arp_t,
                "word": unit.get("token", ""),
                "word_idx": w_idx,
                "stress": stress,
                "word_pause_after_ms": p_pause if k == len(pairs) - 1 else 0.0,
            })

    return {
        "id": uid,
        "speaker": spk,
        "text": text,
        "wav": xa_t,
        "feats": feats,
        "ipas": ipas,
        "meta_phones": meta_phones,
    }


def _est_formants_lsf_bounded(env: np.ndarray, fb: tuple[float, float, float]) -> np.ndarray:
    """Estimate monotonic `(F1, F2, F3)` via LSF-22 around `IPA_SPECS` fallback."""
    try:
        sf = decompose_frame_source_filter(env, order=22)
        hz, _ = lsf_to_formants(sf.lsf, sr=SR, f_min=200.0, f_max=4500.0)
    except Exception:
        hz = np.zeros(0, dtype=np.float64)

    f1_fb, f2_fb, f3_fb = float(fb[0]), float(fb[1]), float(fb[2])
    c1 = hz[(hz >= 220.0) & (hz <= 1220.0)]
    f1 = float(c1[np.argmin(np.abs(c1 - f1_fb))]) if c1.size else f1_fb
    f1 = float(np.clip(f1, 230.0, 1180.0))

    c2 = hz[(hz >= max(f1 + 210.0, 720.0)) & (hz <= 3050.0)]
    f2 = float(c2[np.argmin(np.abs(c2 - f2_fb))]) if c2.size else max(f1 + 250.0, f2_fb)
    f2 = float(np.clip(f2, max(f1 + 210.0, 720.0), 3050.0))

    c3 = hz[(hz >= max(f2 + 260.0, 1780.0)) & (hz <= 4250.0)]
    f3 = float(c3[np.argmin(np.abs(c3 - f3_fb))]) if c3.size else max(f2 + 300.0, f3_fb)
    f3 = float(np.clip(f3, max(f2 + 260.0, 1780.0), 4250.0))

    return np.array([f1, f2, f3], dtype=np.float64)


def _bin_token_5(arr: np.ndarray, a: int, b: int) -> np.ndarray:
    """Average `(T, ...)` slice `[a:b]` into 5 uniform phase bins."""
    n = max(1, b - a)
    out = np.empty((N_ART_BINS, *arr.shape[1:]), dtype=np.float64)
    for bi in range(N_ART_BINS):
        lo = a + int(math.floor(bi * n / N_ART_BINS))
        hi = a + int(math.floor((bi + 1) * n / N_ART_BINS))
        if hi <= lo:
            lo = min(b - 1, a + min(n - 1, bi * n // N_ART_BINS))
            hi = lo + 1
        out[bi] = np.mean(arr[lo:hi], axis=0)
    return out


def _build_default_art_keys() -> list[str]:
    base_keys = sorted(IPA_SPECS.keys())
    allophones = ["l_coda", "p_coda", "t_coda", "k_coda", "b_coda", "d_coda", "ɡ_coda"]
    return base_keys + [a for a in allophones if a not in base_keys]


def build_en_articulation_table(*, n_em: int = 4, workers: int = 8) -> dict[str, Any]:
    """Run offline GMM-HMM alignment & build speaker-independent Stage 9B `.npz` artifact."""
    t_start = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    heldout_set = set(manifest["english"]["heldout_ids"])

    train_tasks: list[tuple[str, str, str, str]] = []
    held_tasks: list[tuple[str, str, str, str]] = []
    for u in manifest["english"]["utterances"]:
        item = (u["id"], u["speaker"], u["path"], u["text"])
        if u["id"] in heldout_set:
            held_tasks.append(item)
        else:
            train_tasks.append(item)

    print(f"[Stage9B] Extracting 24 kHz features: {len(train_tasks)} TRAIN, {len(held_tasks)} HELDOUT (isolated)...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        train_recs = list(ex.map(_worker_extract_en_utt, train_tasks))
        held_recs = list(ex.map(_worker_extract_en_utt, held_tasks))

    print(f"[Stage9B] Running 3-state GMM-HMM Viterbi alignment ({n_em} EM iters) strictly on {len(train_recs)} TRAIN utts...")
    gmm, train_bounds = _train_and_align_gmm(train_recs, n_em=n_em, sr=SR, hop=HOP)

    # Align isolated 20 held-out utterances ONLY for diagnostic Step 4 cache (never mixed into table)
    held_bounds: dict[str, list[tuple[int, int, int]]] = {}
    for rec in held_recs:
        held_bounds[rec["id"]] = _align_with_trained_gmm(gmm, rec["wav"], rec["feats"], rec["ipas"], sr=SR, hop=HOP)

    # Save lightweight held-out reference alignments in scratch/ for leakage-free Stage 9B verification
    _HELDOUT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ho_payload: dict[str, Any] = {"ids": np.asarray([r["id"] for r in held_recs])}
    for idx, rec in enumerate(held_recs):
        uid = rec["id"]
        feats: v6_voc.VocoderFeatures = rec["feats"]
        bnds = held_bounds[uid]
        vm = feats.voiced & (feats.f0_hz > 40.0)
        v_ltas = np.mean(feats.log_env[vm], axis=0) if np.any(vm) else np.zeros(N_BINS)
        ho_payload[f"u_{idx}_text"] = np.asarray([rec["text"]])
        ho_payload[f"u_{idx}_spk"] = np.asarray([rec["speaker"]])
        ho_payload[f"u_{idx}_ipas"] = np.asarray(rec["ipas"])
        ho_payload[f"u_{idx}_bounds"] = np.asarray(bnds, dtype=np.int32)
        ho_payload[f"u_{idx}_log_env"] = feats.log_env.astype(np.float32)
        ho_payload[f"u_{idx}_voiced"] = feats.voiced.astype(bool)
        ho_payload[f"u_{idx}_f0"] = feats.f0_hz.astype(np.float32)
        ho_payload[f"u_{idx}_v_ltas"] = v_ltas.astype(np.float32)
    np.savez_compressed(_HELDOUT_CACHE_PATH, **ho_payload)

    # Pass 1: Per-speaker F0 median, VTLN scale, and canonical LTAS baselines (strictly from TRAIN!)
    spk_f0s: dict[str, list[np.ndarray]] = defaultdict(list)
    spk_vtln_ratios: dict[str, list[float]] = defaultdict(list)
    spk_uv_frames: dict[str, list[np.ndarray]] = defaultdict(list)

    for rec in train_recs:
        uid = rec["id"]
        spk = rec["speaker"]
        feats: v6_voc.VocoderFeatures = rec["feats"]
        vm = feats.voiced & (feats.f0_hz > 45.0)
        if np.any(vm):
            spk_f0s[spk].append(feats.f0_hz[vm])
        um = (~feats.voiced) & (np.mean(feats.log_env[:, _HF_MASK], axis=1) > -10.5)
        if np.any(um):
            spk_uv_frames[spk].append(feats.log_env[um])

        for ipa, (ps, pm, pe) in zip(rec["ipas"], train_bounds[uid]):
            if ipa not in _ANCHOR_VOWELS or pe - ps < 6 or not feats.voiced[pm]:
                continue
            spec = IPA_SPECS.get(ipa)
            if spec is None:
                continue
            try:
                sf = decompose_frame_source_filter(feats.log_env[pm], order=22)
                hz, _ = lsf_to_formants(sf.lsf, sr=SR, f_min=220.0, f_max=4500.0)
                c2 = hz[(hz >= 700.0) & (hz <= 3000.0)]
                c3 = hz[(hz >= 1800.0) & (hz <= 4100.0)]
                if c2.size:
                    f2_m = float(c2[np.argmin(np.abs(c2 - spec.formants[1]))])
                    r2 = f2_m / max(400.0, spec.formants[1])
                    if 0.72 <= r2 <= 1.25:
                        spk_vtln_ratios[spk].append(r2)
                if c3.size:
                    f3_m = float(c3[np.argmin(np.abs(c3 - spec.formants[2]))])
                    r3 = f3_m / max(1500.0, spec.formants[2])
                    if 0.75 <= r3 <= 1.22:
                        spk_vtln_ratios[spk].append(r3)
            except Exception:
                continue

    spk_f0_med: dict[str, float] = {}
    spk_vtln: dict[str, float] = {}
    for spk in set(r["speaker"] for r in train_recs):
        spk_f0_med[spk] = float(np.median(np.concatenate(spk_f0s[spk]))) if spk_f0s[spk] else 160.0
        raw_v = float(np.median(spk_vtln_ratios[spk])) if spk_vtln_ratios[spk] else 1.0
        spk_vtln[spk] = float(np.clip(raw_v, 0.80, 1.10))

    # Compute canonical-warped voiced LTAS per speaker
    spk_canon_voiced_frames: dict[str, list[np.ndarray]] = defaultdict(list)
    for rec in train_recs:
        spk = rec["speaker"]
        feats = rec["feats"]
        vm = feats.voiced & (feats.f0_hz > 45.0)
        if not np.any(vm):
            continue
        inv_alpha = 1.0 / spk_vtln[spk]
        warped = _warp_log_env_pivot(feats.log_env[vm], inv_alpha, sr=SR, n_fft=N_FFT)
        spk_canon_voiced_frames[spk].append(warped)

    spk_ltas_v: dict[str, np.ndarray] = {}
    spk_ltas_u: dict[str, np.ndarray] = {}
    for spk in spk_f0_med:
        spk_ltas_v[spk] = (
            np.mean(np.vstack(spk_canon_voiced_frames[spk]), axis=0)
            if spk_canon_voiced_frames[spk]
            else np.zeros(N_BINS, dtype=np.float64)
        )
        spk_ltas_u[spk] = (
            np.mean(np.vstack(spk_uv_frames[spk]), axis=0)
            if spk_uv_frames[spk]
            else spk_ltas_v[spk]
        )

    print("[Stage9B] Speaker normalization params:", {s: {"f0": round(spk_f0_med[s], 1), "vtln": round(spk_vtln[s], 4)} for s in sorted(spk_f0_med)})

    # Pass 2: Extract speaker-stripped tokens across all 780 training utterances
    all_keys = _build_default_art_keys()
    key_to_id = {k: i for i, k in enumerate(all_keys)}

    strict_tokens: dict[str, list[dict[str, Any]]] = defaultdict(list)
    relaxed_tokens: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # Rhythm pools
    vowel_fortis_durs: dict[tuple[str, int], list[float]] = defaultdict(list)
    vowel_lenis_durs: dict[tuple[str, int], list[float]] = defaultdict(list)
    cluster2_ratios: list[float] = []
    cluster3_ratios: list[float] = []
    pfin_ratios: list[float] = []
    nonpfin_ratios: list[float] = []
    stress_f0_bins: dict[int, list[np.ndarray]] = defaultdict(list)
    stress_gain_pools: dict[int, list[float]] = defaultdict(list)

    n_tok_total = 0
    n_tok_strict = 0

    for rec in train_recs:
        uid = rec["id"]
        spk = rec["speaker"]
        wav = rec["wav"]
        feats = rec["feats"]
        bnds = train_bounds[uid]
        meta_p = rec["meta_phones"]
        n_p = len(meta_p)
        if n_p == 0 or len(bnds) != n_p:
            continue

        vm_u = feats.voiced & (feats.f0_hz > 45.0)
        utt_voiced_db = (
            float(np.mean(np.mean(feats.log_env[vm_u], axis=1) * NEPER_TO_DB))
            if np.any(vm_u)
            else -35.0
        )

        # Canonical-warped speaker-stripped log_env for whole utterance
        inv_alpha = 1.0 / spk_vtln[spk]
        canon_env_all = _warp_log_env_pivot(feats.log_env, inv_alpha, sr=SR, n_fft=N_FFT) - spk_ltas_v[spk][None, :]
        uv_stripped_all = feats.log_env - spk_ltas_u[spk][None, :]

        # Compute consonant cluster spans per word
        cluster_lens = [1] * n_p
        ii = 0
        while ii < n_p:
            if meta_p[ii]["ipa"] in VOWEL_IPAS:
                ii += 1
                continue
            jj = ii + 1
            while (
                jj < n_p
                and meta_p[jj]["ipa"] not in VOWEL_IPAS
                and meta_p[jj]["word_idx"] == meta_p[jj - 1]["word_idx"]
                and meta_p[jj - 1]["word_pause_after_ms"] < 20.0
            ):
                jj += 1
            cl = jj - ii
            for kk in range(ii, jj):
                cluster_lens[kk] = cl
            ii = jj

        f0_ref_spk = max(50.0, spk_f0_med[spk])

        for i in range(n_p):
            mp = meta_p[i]
            ipa = mp["ipa"]
            if ipa not in IPA_SPECS:
                continue
            sp = IPA_SPECS[ipa]
            ps, _pm, pe = bnds[i]
            if pe <= ps:
                continue

            next_mp = meta_p[i + 1] if i + 1 < n_p else None
            prev_mp = meta_p[i - 1] if i > 0 else None
            same_word_next = bool(
                next_mp is not None
                and next_mp["word_idx"] == mp["word_idx"]
                and mp["word_pause_after_ms"] < 20.0
            )
            same_word_prev = bool(
                prev_mp is not None
                and prev_mp["word_idx"] == mp["word_idx"]
                and prev_mp["word_pause_after_ms"] < 20.0
            )
            next_ipa = next_mp["ipa"] if (next_mp is not None and mp["word_pause_after_ms"] < 20.0) else None
            prev_ipa = prev_mp["ipa"] if (prev_mp is not None and prev_mp["word_pause_after_ms"] < 20.0) else None

            key = resolve_en_allophone(ipa, next_ipa, same_word_next)
            if key not in key_to_id:
                key = ipa
            p_cls = classify_articulatory_class(prev_ipa)
            n_cls = classify_articulatory_class(next_ipa)

            seg_w = wav[ps * HOP : min(len(wav), pe * HOP)]
            dur_ms = float((pe - ps) * HOP * 1000.0 / SR)
            ok, reason, met = score_token_purity(ipa, feats.log_env[ps:pe], feats.voiced[ps:pe], seg_w, sr=SR)
            n_tok_total += 1
            if ok:
                n_tok_strict += 1
            elif dur_ms < 20.0 or met.get("rms_dbfs", -99.0) < -42.0 or reason in ("near_silent", "too_short"):
                continue

            stress = mp["stress"]
            s_idx = 1 if stress == 1 else (2 if stress == 2 else 0)
            is_obs = sp.manner in ("fricative", "stop", "affricate")
            is_pfin = (i == n_p - 1) or (mp["word_pause_after_ms"] >= 45.0)

            # Collect rhythm stats from strictly clean tokens (stressed vowels avoid floor-clamping bias)
            if ok:
                if ipa in VOWEL_IPAS and not is_pfin and s_idx == 1:
                    if same_word_next and next_ipa in VOICELESS_OBSTRUENTS:
                        vowel_fortis_durs[(ipa, s_idx)].append(dur_ms)
                    elif same_word_next and next_ipa is not None:
                        vowel_lenis_durs[(ipa, s_idx)].append(dur_ms)
                if not is_pfin and ipa not in VOWEL_IPAS:
                    cl = cluster_lens[i]
                    if cl == 2:
                        cluster2_ratios.append(dur_ms / max(20.0, sp.base_dur_ms))
                    elif cl >= 3:
                        cluster3_ratios.append(dur_ms / max(20.0, sp.base_dur_ms))
                if ipa in VOWEL_IPAS:
                    r_base = dur_ms / max(30.0, sp.base_dur_ms)
                    if is_pfin:
                        pfin_ratios.append(r_base)
                    else:
                        nonpfin_ratios.append(r_base)

            # Compute 5-bin spectral & articulatory stats
            if not is_obs:
                env5 = _bin_token_5(canon_env_all, ps, pe)
                masked_env5 = env5 * _W_ART[None, :]
                cep5 = log_env_to_cep(masked_env5, n_cep=N_ART_CEP)
                f5 = np.empty((N_ART_BINS, 3), dtype=np.float64)
                for bi in range(N_ART_BINS):
                    f5[bi] = _est_formants_lsf_bounded(
                        _warp_log_env_pivot(feats.log_env[min(pe - 1, ps + int((bi + 0.5) * (pe - ps) / N_ART_BINS))][None, :], inv_alpha)[0],
                        sp.formants,
                    )
            else:
                env5 = _bin_token_5(uv_stripped_all, ps, pe)
                cep5 = log_env_to_cep(env5, n_cep=N_ART_CEP)
                f5 = np.tile(np.asarray(sp.formants, dtype=np.float64)[None, :], (N_ART_BINS, 1))

            v_prob5 = _bin_token_5(feats.voiced.astype(np.float64)[:, None], ps, pe)[:, 0]
            lvl_db5 = _bin_token_5(np.mean(feats.log_env, axis=1, keepdims=True) * NEPER_TO_DB, ps, pe)[:, 0] - utt_voiced_db

            if ok and ipa in VOWEL_IPAS and stress in (0, 1, 2):
                f0_seg = feats.f0_hz[ps:pe]
                vm_seg = feats.voiced[ps:pe] & (f0_seg > 45.0)
                if np.sum(vm_seg) >= 3:
                    st_arr = 12.0 * np.log2(np.where(vm_seg, f0_seg, f0_ref_spk) / f0_ref_spk)
                    if not np.all(vm_seg):
                        idx_v = np.where(vm_seg)[0]
                        st_arr = np.interp(np.arange(len(f0_seg)), idx_v, st_arr[idx_v])
                    st5 = _bin_token_5(st_arr[:, None], 0, len(st_arr))[:, 0]
                    stress_f0_bins[stress].append(st5)
                    stress_gain_pools[stress].append(float(np.mean(lvl_db5)))

            tok = {
                "dur_ms": dur_ms,
                "s_idx": s_idx,
                "p_cls": p_cls,
                "n_cls": n_cls,
                "cep5": cep5,
                "f5": f5,
                "v5": v_prob5,
                "rel_db5": lvl_db5,
                "is_obs": is_obs,
            }
            if ok:
                strict_tokens[key].append(tok)
            else:
                relaxed_tokens[key].append(tok)

    print(f"[Stage9B] Token extraction complete: strict_clean={n_tok_strict}/{n_tok_total} ({100.0*n_tok_strict/max(1,n_tok_total):.1f}%)")

    # Pass 3: Aggregate per-key tables cleanly
    K = len(all_keys)
    n_cls = len(ART_CLASSES)
    phone_is_voiced = np.zeros(K, dtype=bool)
    phone_counts = np.zeros(K, dtype=np.int32)
    delta_cep = np.zeros((K, N_ART_BINS, N_ART_CEP), dtype=np.float64)
    formants_5bin = np.zeros((K, N_ART_BINS, 3), dtype=np.float64)
    voicing_prob_5bin = np.zeros((K, N_ART_BINS), dtype=np.float64)
    rel_level_db_5bin = np.zeros((K, N_ART_BINS), dtype=np.float64)
    left_coart_cep = np.zeros((K, n_cls, N_COART_CEP), dtype=np.float64)
    right_coart_cep = np.zeros((K, n_cls, N_COART_CEP), dtype=np.float64)
    left_coart_f123 = np.zeros((K, n_cls, 3), dtype=np.float64)
    right_coart_f123 = np.zeros((K, n_cls, 3), dtype=np.float64)
    dur_ms_by_stress = np.zeros((K, 3), dtype=np.float64)

    for kid, key in enumerate(all_keys):
        base_ipa = key.replace("_coda", "")
        sp = IPA_SPECS.get(base_ipa, IPA_SPECS["ə"])
        phone_is_voiced[kid] = sp.voiced

        pool = list(strict_tokens.get(key, []))
        if len(pool) < 12:
            pool.extend(relaxed_tokens.get(key, []))
        if len(pool) < 4 and key != base_ipa:
            pool.extend(strict_tokens.get(base_ipa, [])[:16])

        phone_counts[kid] = len(pool)
        if not pool:
            f_start = np.asarray(sp.formants, dtype=np.float64)
            f_end = np.asarray(sp.end_formants or sp.formants, dtype=np.float64)
            for bi in range(N_ART_BINS):
                u = bi / float(N_ART_BINS - 1)
                formants_5bin[kid, bi] = (1.0 - u) * f_start + u * f_end
            voicing_prob_5bin[kid] = 1.0 if sp.voiced else 0.0
            dur_ms_by_stress[kid] = sp.base_dur_ms
            continue

        cep_arr = np.stack([t["cep5"] for t in pool], axis=0)      # (N, 5, 24)
        f_arr = np.stack([t["f5"] for t in pool], axis=0)          # (N, 5, 3)
        v_arr = np.stack([t["v5"] for t in pool], axis=0)          # (N, 5)
        db_arr = np.stack([t["rel_db5"] for t in pool], axis=0)    # (N, 5)

        delta_cep[kid] = np.clip(np.median(cep_arr, axis=0), -6.0, 6.0)
        formants_5bin[kid] = np.median(f_arr, axis=0)
        voicing_prob_5bin[kid] = np.clip(np.median(v_arr, axis=0), 0.0, 1.0)
        rel_level_db_5bin[kid] = np.clip(np.median(db_arr, axis=0), -18.0, 8.0)

        # Durations by stress slot {0, 1, 2}
        all_durs = [t["dur_ms"] for t in pool]
        med_all_dur = float(np.median(all_durs))
        for si in range(3):
            sd = [t["dur_ms"] for t in pool if t["s_idx"] == si]
            if len(sd) >= 5:
                dur_ms_by_stress[kid, si] = float(np.clip(np.median(sd), 26.0, 260.0))
            else:
                mul = 1.15 if (si == 1 and base_ipa in VOWEL_IPAS) else (0.78 if (si == 0 and base_ipa in VOWEL_IPAS) else 1.0)
                dur_ms_by_stress[kid, si] = float(np.clip(med_all_dur * mul, 26.0, 260.0))

        # Empirical-Bayes shrunk left/right coarticulation deltas on entry bin 0 & exit bin 4
        base_c0 = delta_cep[kid, 0, :N_COART_CEP]
        base_c4 = delta_cep[kid, -1, :N_COART_CEP]
        base_f0 = formants_5bin[kid, 0]
        base_f4 = formants_5bin[kid, -1]

        for cid in range(n_cls):
            l_idx = [j for j, t in enumerate(pool) if t["p_cls"] == cid]
            if len(l_idx) >= 3:
                w_c = len(l_idx) / (len(l_idx) + 15.0)
                w_f = len(l_idx) / (len(l_idx) + 18.0)
                d_c = np.median(cep_arr[l_idx, 0, :N_COART_CEP], axis=0) - base_c0
                d_f = np.median(f_arr[l_idx, 0], axis=0) - base_f0
                left_coart_cep[kid, cid] = np.clip(w_c * d_c, -1.8, 1.8)
                left_coart_f123[kid, cid] = np.clip(w_f * d_f, -320.0, 320.0)

            r_idx = [j for j, t in enumerate(pool) if t["n_cls"] == cid]
            if len(r_idx) >= 3:
                w_c = len(r_idx) / (len(r_idx) + 15.0)
                w_f = len(r_idx) / (len(r_idx) + 18.0)
                d_c = np.median(cep_arr[r_idx, -1, :N_COART_CEP], axis=0) - base_c4
                d_f = np.median(f_arr[r_idx, -1], axis=0) - base_f4
                right_coart_cep[kid, cid] = np.clip(w_c * d_c, -1.8, 1.8)
                right_coart_f123[kid, cid] = np.clip(w_f * d_f, -320.0, 320.0)

    # Rhythm scalars: pre-fortis clipping, cluster compression, phrase-final lengthening
    pf_pairs = []
    for key_pair, f_list in vowel_fortis_durs.items():
        l_list = vowel_lenis_durs.get(key_pair, [])
        if len(f_list) >= 5 and len(l_list) >= 5:
            pf_pairs.append(float(np.median(f_list)) / max(25.0, float(np.median(l_list))))
    pre_fortis_scale = float(np.clip(np.median(pf_pairs) if pf_pairs else 0.76, 0.65, 0.86))

    cluster2_scale = float(np.clip(np.median(cluster2_ratios) if cluster2_ratios else 0.78, 0.68, 0.90))
    cluster3_scale = float(np.clip(np.median(cluster3_ratios) if cluster3_ratios else 0.66, 0.56, 0.82))
    raw_pfin = (
        float(np.median(pfin_ratios)) / max(0.3, float(np.median(nonpfin_ratios)))
        if (pfin_ratios and nonpfin_ratios)
        else 1.25
    )
    phrase_final_scale = float(np.clip(raw_pfin, 1.12, 1.42))

    # Speaker-normalized relative stress F0 & gain
    stress_f0_st_5bin = np.zeros((3, N_ART_BINS), dtype=np.float64)
    stress_gain_db = np.zeros(3, dtype=np.float64)
    for si in (0, 1, 2):
        if stress_f0_bins[si]:
            stress_f0_st_5bin[si] = np.clip(np.median(np.stack(stress_f0_bins[si], axis=0), axis=0), -4.5, 4.5)
        if stress_gain_pools[si]:
            stress_gain_db[si] = float(np.clip(np.median(stress_gain_pools[si]), -4.0, 4.0))
    # Center stress_f0_st_5bin and stress_gain_db relative to weighted baseline
    f0_mid_dc = float(np.mean(stress_f0_st_5bin))
    stress_f0_st_5bin -= f0_mid_dc
    stress_gain_db -= float(np.mean(stress_gain_db))

    # Save compact compressed float32 `.npz` (`allow_pickle=False`)
    EN_ARTICULATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EN_ARTICULATION_PATH,
        keys=np.asarray(all_keys),
        phone_is_voiced=phone_is_voiced,
        phone_counts=phone_counts,
        delta_cep=delta_cep.astype(np.float32),
        formants_5bin=formants_5bin.astype(np.float32),
        voicing_prob_5bin=voicing_prob_5bin.astype(np.float32),
        rel_level_db_5bin=rel_level_db_5bin.astype(np.float32),
        left_coart_cep=left_coart_cep.astype(np.float32),
        right_coart_cep=right_coart_cep.astype(np.float32),
        left_coart_f123=left_coart_f123.astype(np.float32),
        right_coart_f123=right_coart_f123.astype(np.float32),
        dur_ms_by_stress=dur_ms_by_stress.astype(np.float32),
        pre_fortis_scale=np.asarray([pre_fortis_scale], dtype=np.float32),
        cluster2_scale=np.asarray([cluster2_scale], dtype=np.float32),
        cluster3_scale=np.asarray([cluster3_scale], dtype=np.float32),
        phrase_final_scale=np.asarray([phrase_final_scale], dtype=np.float32),
        stress_f0_st_5bin=stress_f0_st_5bin.astype(np.float32),
        stress_gain_db=stress_gain_db.astype(np.float32),
    )
    clear_en_articulation_cache()

    size_bytes = EN_ARTICULATION_PATH.stat().st_size
    l_on_f2 = float(formants_5bin[key_to_id["l"], 2, 1])
    l_cd_f2 = float(formants_5bin[key_to_id["l_coda"], 2, 1])

    summary = {
        "train_utterances": len(train_recs),
        "heldout_excluded": len(held_recs),
        "strict_clean_tokens": n_tok_strict,
        "total_tokens": n_tok_total,
        "strict_accept_rate": round(n_tok_strict / max(1, n_tok_total), 4),
        "n_keys": K,
        "artifact_path": str(EN_ARTICULATION_PATH),
        "artifact_size_bytes": size_bytes,
        "artifact_size_kb": round(size_bytes / 1024.0, 2),
        "zero_ml_weights": True,
        "pre_fortis_scale": round(pre_fortis_scale, 4),
        "cluster2_scale": round(cluster2_scale, 4),
        "cluster3_scale": round(cluster3_scale, 4),
        "phrase_final_scale": round(phrase_final_scale, 4),
        "l_onset_f2_hz": round(l_on_f2, 1),
        "l_coda_f2_hz": round(l_cd_f2, 1),
        "new_en_obstruent_counts": {k: int(phone_counts[key_to_id[k]]) for k in sorted(("θ", "ð", "v", "z", "ʒ", "b", "d", "ɡ", "dʒ", "æ", "ɪ", "ʊ", "l_coda")) if k in key_to_id},
        "build_time_s": round(time.perf_counter() - t_start, 2),
    }
    EN_ARTICULATION_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    build_en_articulation_table()
