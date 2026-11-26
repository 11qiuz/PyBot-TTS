"""Unified Phase-5 zero-leakage benchmark across all 4 testbeds for `TimbreClonerV3`.

Benchmarks:
1. 32-fold zero-leakage LOO-CV (SSB0502 <-> SSB1593, 1-shot & 3-shot) + male_115 stress test
2. Real 4-speaker English cross-gender & same-gender (bdl, rms, clb, slt; n=60/method)
3. Real 5-speaker Mandarin (SSB0073, SSB0139, SSB0005, SSB0502, SSB1593; n=60/method)
4. Real bidirectional cross-lingual ZH <-> EN (bdl, slt, SSB0073, SSB0502; n=24/method)
   with explicit cross-corpus EQ separation ablations (blind vs corpus_eq_keep={0.0, 0.65, 1.0}).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v5_eval import metrics as eval_m
from exp_v6_vocoder import cloner as v6_cloner, eval_vocoder as v6_eval, vocoder as v6_voc
from exp_v8_timbre_clone.ablation_clone import _eval_mode_fold, _summarize_rows
from exp_v8_timbre_clone.clarity_metrics import compute_clarity_metrics
from exp_v8_timbre_clone.eval_english_crossgender import _classify_4way, _pair_kind, _peak_limit
from exp_v8_timbre_clone.eval_multilingual_corpus import _load_zh_5speakers, _read_utt
from exp_v8_timbre_clone.timbre_cloner_v2 import (
    TimbreProfileV2,
    adapt_features_v2,
    extract_v2_profile,
    make_synthetic_male_v2_profile,
)
from exp_v9_timbre_v3 import EXP_DIR, EXTENDED_INDEX, MANIFEST_PATH, N_FFT, SR
from exp_v9_timbre_v3.eq_decoupler import estimate_corpus_baseline
from exp_v9_timbre_v3.timbre_cloner_v3 import (
    MANNER_NAMES,
    Q_ENV_STORE,
    Q_MANNER_STORE,
    TimbreProfileV3,
    _ceps_encode,
    adapt_features_v3,
    extract_timbre_v3,
)

SAMPLES_V9_DIR = EXP_DIR / "samples"


def _v2_to_v3_synthetic(v2: TimbreProfileV2, base_v3: TimbreProfileV3) -> TimbreProfileV3:
    """Wrap synthetic `male_115` V2 profile as TimbreProfileV3 without neural weights."""
    return TimbreProfileV3(
        name=v2.name,
        corpus=base_v3.corpus,
        sr=SR,
        n_fft=N_FFT,
        total_voiced_s=v2.total_voiced_s,
        f0_median_hz=v2.f0_median_hz,
        f0_log_std=v2.f0_log_std,
        f0_iqr_st=v2.f0_iqr_st,
        tilt_db_per_oct=v2.tilt_db_per_oct,
        hnr_median_db=v2.hnr_median_db,
        f1_median_hz=v2.f1_median_hz,
        f2_median_hz=v2.f2_median_hz,
        f3_median_hz=v2.f3_median_hz,
        n_formant_frames=v2.n_formant_frames,
        ope_h1_h2_db=max(-2.0, base_v3.ope_h1_h2_db - 4.0),
        ope_low_mid_db=base_v3.ope_low_mid_db,
        bri_formant_sharp_db=base_v3.bri_formant_sharp_db,
        bri_2_6k_db=base_v3.bri_2_6k_db,
        bap_voiced_median=[float(x) for x in v2.bap_voiced_median],
        voiced_ceps_q32=_ceps_encode(v2.voiced_env_mean, n_fft=N_FFT, q_max=Q_ENV_STORE),
        manner_deltas_q14={k: [0.0] * (Q_MANNER_STORE + 1) for k in MANNER_NAMES},
        manner_counts={k: 0 for k in MANNER_NAMES},
        unvoiced_tilt_slope=v2.unvoiced_tilt_slope,
        unvoiced_rel_level_np=v2.unvoiced_rel_level_np,
    )


def run_32fold_and_male_stress() -> dict:
    """Testbed 1: 32-fold LOO-CV (SSB0502 <-> SSB1593) + male_115 stress test matching Phase 4."""
    idx = json.loads(EXTENDED_INDEX.read_text(encoding="utf-8"))
    utts = idx["utterances"]
    real_wavs: dict[str, np.ndarray] = {}
    real_feats: dict[str, v6_voc.VocoderFeatures] = {}
    real_clf: dict[str, dict] = {}
    real_clarity: dict[str, dict] = {}
    by_spk: dict[str, list[dict]] = {"SSB0502": [], "SSB1593": []}
    for u in utts:
        uid = u["id"]
        x, sr = v6_voc.read_wav_mono(u["path"])
        x_act = x[u["active_start"] : u["active_stop"]]
        real_wavs[uid] = x_act
        real_feats[uid] = v6_voc.analyze(x_act, sr=SR)
        real_clf[uid] = v6_eval._extract_clf_vec(x_act, sr=SR)
        real_clarity[uid] = compute_clarity_metrics(x_act, sr=SR)
        by_spk[u["speaker"]].append(u)

    methods = ["v6_coupled", "v8_decoupled", "v9_blind", "v9_no_manner", "v9_no_utau"]
    summary: dict = {}
    unconv_rows: list[dict] = []
    modes_by_shot = {1: {m: [] for m in methods}, 3: {m: [] for m in methods}}

    for fold_i, held in enumerate(utts):
        uid = held["id"]
        tgt_spk = held["speaker"]
        src_spk = "SSB1593" if tgt_spk == "SSB0502" else "SSB0502"
        train_utts = [u for j, u in enumerate(utts) if j != fold_i]
        tgt_pool = [u for u in by_spk[tgt_spk] if u["id"] != uid]
        src_u = by_spk[src_spk][fold_i % len(by_spk[src_spk])]
        src_uid = src_u["id"]

        unconv_rows.append(
            _eval_mode_fold(
                "unconverted", real_wavs[src_uid], real_wavs[src_uid], real_feats[src_uid], real_feats[src_uid],
                tgt_spk, real_clf[uid], real_clf[src_uid], train_utts, real_clf, real_clarity[src_uid], 0.0, {},
            )
        )

        prof_src_v2 = extract_v2_profile([(real_wavs[src_uid], real_feats[src_uid])], sr=SR, name=src_spk)
        prof_src_v3 = extract_timbre_v3([(real_wavs[src_uid], real_feats[src_uid])], sr=SR, name=src_spk, corpus="aishell3")

        for n_shots in (1, 3):
            shot_utts = [tgt_pool[(fold_i + k) % len(tgt_pool)] for k in range(n_shots)]
            prof_tgt_v2 = extract_v2_profile([(real_wavs[x["id"]], real_feats[x["id"]]) for x in shot_utts], sr=SR, name=tgt_spk)
            prof_tgt_v3 = extract_timbre_v3([(real_wavs[x["id"]], real_feats[x["id"]]) for x in shot_utts], sr=SR, name=tgt_spk, corpus="aishell3")

            for m_name in methods:
                t0 = time.perf_counter()
                extra = {}
                if m_name == "v6_coupled":
                    conv_f = v6_cloner.adapt_features(real_feats[src_uid], prof_src_v2.to_v6_profile(), prof_tgt_v2.to_v6_profile())
                elif m_name == "v8_decoupled":
                    conv_f, info = adapt_features_v2(real_feats[src_uid], prof_src_v2, prof_tgt_v2)
                    extra["vtln_alpha"] = float(info["alpha"])
                elif m_name == "v9_blind":
                    conv_f, info = adapt_features_v3(real_feats[src_uid], prof_src_v3, prof_tgt_v3, eq_mode="blind_tilt_anchor")
                    extra["vtln_alpha"] = float(info["alpha"])
                elif m_name == "v9_no_manner":
                    conv_f, info = adapt_features_v3(real_feats[src_uid], prof_src_v3, prof_tgt_v3, use_manner_delta=False)
                    extra["vtln_alpha"] = float(info["alpha"])
                else:
                    conv_f, info = adapt_features_v3(real_feats[src_uid], prof_src_v3, prof_tgt_v3, use_utau_colors=False)
                    extra["vtln_alpha"] = float(info["alpha"])
                conv_w = v6_voc.synthesize(conv_f)
                dt = time.perf_counter() - t0
                modes_by_shot[n_shots][m_name].append(
                    _eval_mode_fold(
                        m_name, conv_w, real_wavs[src_uid], real_feats[src_uid], conv_f,
                        tgt_spk, real_clf[uid], real_clf[src_uid], train_utts, real_clf, real_clarity[src_uid], dt, extra,
                    )
                )

    summary["unconverted_donor"] = _summarize_rows(unconv_rows)
    for n_shots in (1, 3):
        summary[f"loo_cv_{n_shots}shot"] = {m: _summarize_rows(modes_by_shot[n_shots][m]) for m in methods}

    # Male 115 Hz stress test
    stress_rows = []
    for spk, sid in (("SSB0502", "SSB05020005"), ("SSB1593", "SSB15930031")):
        pool = [x for x in utts if x["speaker"] == spk and x["id"] != sid][:3]
        src_v2 = extract_v2_profile([(real_wavs[x["id"]], real_feats[x["id"]]) for x in pool], sr=SR, name=spk)
        src_v3 = extract_timbre_v3([(real_wavs[x["id"]], real_feats[x["id"]]) for x in pool], sr=SR, name=spk, corpus="aishell3")
        male_v2 = make_synthetic_male_v2_profile(src_v2, f0_hz=115.0, vtln_alpha=0.84, tilt_db_per_oct=-4.5)
        male_v3 = _v2_to_v3_synthetic(male_v2, src_v3)
        for m_name in ("v6_coupled_hard084", "v8_decoupled_autovtln", "v9_blind_autovtln"):
            if m_name == "v6_coupled_hard084":
                f_out = v6_cloner.adapt_features(real_feats[sid], src_v2.to_v6_profile(), male_v2.to_v6_profile(), vtln_alpha=0.84)
                a_used = 0.84
            elif m_name == "v8_decoupled_autovtln":
                f_out, info = adapt_features_v2(real_feats[sid], src_v2, male_v2)
                a_used = float(info["alpha"])
            else:
                f_out, info = adapt_features_v3(real_feats[sid], src_v3, male_v3)
                a_used = float(info["alpha"])
            w_out = v6_voc.synthesize(f_out)
            fh = eval_m.compute_f0_hnr_metrics(w_out, sr=SR)
            lt = eval_m.compute_ltas_metrics(w_out, sr=SR)
            sd = eval_m.compute_spectral_distances(real_wavs[sid], w_out, sr=SR)
            cl = compute_clarity_metrics(w_out, sr=SR)
            stress_rows.append({
                "source_id": sid,
                "method": m_name,
                "vtln_alpha": round(a_used, 4),
                "f0_med_hz": round(float(fh["f0_median_hz"]), 2),
                "tilt_80_8k_db_oct": round(float(lt["tilt_db_oct"]), 3),
                "target_tilt_err_db_oct": round(abs(float(lt["tilt_db_oct"]) - (-4.5)), 3),
                "content_mcd_cmn_db": round(float(sd["mcd_cmn_db"]), 3),
                "formant_prom_delta_db": round(cl["formant_prominence_db"] - real_clarity[sid]["formant_prominence_db"], 3),
                "sanity_ok": bool(eval_m.check_wav_sanity(w_out, SR)["ok"]),
            })
    summary["male_115hz_stress_test"] = stress_rows
    return summary


def _eval_corpus_suite(
    speaker_rows: dict[str, list[dict]],
    speaker_corpus: dict[str, str],
    pairs: list[tuple[str, str]],
    *,
    n_prompt: int = 3,
    n_clf_train: int = 12,
    n_held: int = 5,
    method_configs: list[tuple[str, dict]],
    export_pairs: set[tuple[str, str]] = (),
    tag_prefix: str = "",
) -> dict:
    """Zero-leakage evaluation helper shared across EN 4-way, ZH 5-way, and ZH<->EN."""
    spks = sorted(speaker_rows.keys())
    genders = {s: speaker_rows[s][0]["gender"] for s in spks}

    wavs: dict[str, np.ndarray] = {}
    feats: dict[str, v6_voc.VocoderFeatures] = {}
    clf_vecs: dict[str, dict[str, np.ndarray]] = {}
    clarity_ref: dict[str, dict] = {}

    prompt_rows = {s: speaker_rows[s][:n_prompt] for s in spks}
    clf_train_rows: list[tuple[str, dict[str, np.ndarray]]] = []
    held_rows = {s: speaker_rows[s][n_prompt + n_clf_train : n_prompt + n_clf_train + n_held] for s in spks}

    for s in spks:
        need = speaker_rows[s][: n_prompt + n_clf_train + n_held]
        for u in need:
            uid = u["id"]
            x = _read_utt(u)
            wavs[uid] = x
            feats[uid] = v6_voc.analyze(x, sr=SR)
            clf_vecs[uid] = v6_eval._extract_clf_vec(x, sr=SR)
            if u in held_rows[s]:
                clarity_ref[uid] = compute_clarity_metrics(x, sr=SR)
        for u in speaker_rows[s][n_prompt : n_prompt + n_clf_train]:
            clf_train_rows.append((s, clf_vecs[u["id"]]))

    real_acc = {"timbre4d": 0, "ltas": 0, "f0": 0, "all": 0, "n": 0}
    for s in spks:
        for u in held_rows[s]:
            qv = clf_vecs[u["id"]]
            real_acc["n"] += 1
            real_acc["timbre4d"] += int(_classify_4way(qv, clf_train_rows, "timbre_4d") == s)
            real_acc["ltas"] += int(_classify_4way(qv, clf_train_rows, "ltas_vec") == s)
            real_acc["f0"] += int(_classify_4way(qv, clf_train_rows, "f0_stats") == s)
            real_acc["all"] += int(_classify_4way(qv, clf_train_rows, "all_features") == s)

    profs_v2 = {
        s: extract_v2_profile([(wavs[u["id"]], feats[u["id"]]) for u in prompt_rows[s]], sr=SR, name=s)
        for s in spks
    }
    profs_v6 = {s: profs_v2[s].to_v6_profile() for s in spks}
    profs_v3 = {
        s: extract_timbre_v3([(wavs[u["id"]], feats[u["id"]]) for u in prompt_rows[s]], sr=SR, name=s, corpus=speaker_corpus[s])
        for s in spks
    }

    # Zero-leakage corpus baseline pools built strictly from prompt profiles grouped by corpus
    corp_pools: dict[str, dict[str, np.ndarray]] = {}
    for s in spks:
        corp_pools.setdefault(speaker_corpus[s], {})[s] = profs_v3[s].voiced_env_mean

    tgt_oracle = {
        s: {
            "tilt": float(np.mean([clf_vecs[u["id"]]["tilt"] for u in held_rows[s]])),
            "ltas_vec": np.mean([clf_vecs[u["id"]]["ltas_vec"] for u in held_rows[s]], axis=0),
            "f0_med": float(np.median([clf_vecs[u["id"]]["f0_med"] for u in held_rows[s]])),
            "hnr": float(np.mean([clf_vecs[u["id"]]["hnr"] for u in held_rows[s]])),
        }
        for s in spks
    }

    SAMPLES_V9_DIR.mkdir(parents=True, exist_ok=True)
    all_method_names = ["unconverted"] + [m for m, _ in method_configs]
    rows_by_method: dict[str, list[dict]] = {m: [] for m in all_method_names}

    for src_spk, tgt_spk in pairs:
        pkind = _pair_kind(genders[src_spk], genders[tgt_spk])
        t_ref = tgt_oracle[tgt_spk]
        s_corp_env = estimate_corpus_baseline(corp_pools[speaker_corpus[src_spk]], exclude_speaker=src_spk)
        t_corp_env = estimate_corpus_baseline(corp_pools[speaker_corpus[tgt_spk]], exclude_speaker=tgt_spk)

        for idx_u, src_u in enumerate(held_rows[src_spk]):
            uid = src_u["id"]
            src_w = wavs[uid]
            src_f = feats[uid]
            src_cl = clarity_ref[uid]

            # Unconverted baseline
            qv0 = clf_vecs[uid]
            rows_by_method["unconverted"].append({
                "pair": f"{src_spk}->{tgt_spk}",
                "kind": pkind,
                "sanity_ok": 1,
                "hit_timbre4d": int(_classify_4way(qv0, clf_train_rows, "timbre_4d") == tgt_spk),
                "hit_ltas": int(_classify_4way(qv0, clf_train_rows, "ltas_vec") == tgt_spk),
                "hit_f0": int(_classify_4way(qv0, clf_train_rows, "f0_stats") == tgt_spk),
                "hit_all": int(_classify_4way(qv0, clf_train_rows, "all_features") == tgt_spk),
                "tilt_err": abs(float(qv0["tilt"]) - t_ref["tilt"]),
                "ltas_rmse": float(np.sqrt(np.mean((qv0["ltas_vec"] - t_ref["ltas_vec"]) ** 2))),
                "f0_err_st": abs(12.0 * math.log2(max(1.0, float(qv0["f0_med"])) / max(1.0, t_ref["f0_med"]))),
                "hnr_err": abs(float(qv0["hnr"]) - t_ref["hnr"]),
                "mel_lsd": 0.0,
                "mcd_cmn": 0.0,
                "prom_d": 0.0,
                "hf_d": 0.0,
                "vtln": 1.0,
            })

            do_export = (src_spk, tgt_spk) in export_pairs and idx_u == 0
            if do_export:
                v6_voc.write_wav_mono(SAMPLES_V9_DIR / f"{tag_prefix}{src_spk}_{uid}_real_src.wav", _peak_limit(src_w), SR)
                ref_u = held_rows[tgt_spk][0]
                v6_voc.write_wav_mono(SAMPLES_V9_DIR / f"{tag_prefix}{tgt_spk}_real_ref.wav", _peak_limit(wavs[ref_u["id"]]), SR)

            for m_name, cfg in method_configs:
                kind = cfg["family"]
                if kind == "v6":
                    conv_f = v6_cloner.adapt_features(src_f, profs_v6[src_spk], profs_v6[tgt_spk])
                    raw_a = profs_v6[tgt_spk].vtln_anchor_f2_hz / max(100.0, profs_v6[src_spk].vtln_anchor_f2_hz)
                    alpha_used = 1.0 if abs(raw_a - 1.0) < 0.04 else float(np.clip(raw_a, 0.78, 1.22))
                elif kind == "v8":
                    conv_f, info = adapt_features_v2(src_f, profs_v2[src_spk], profs_v2[tgt_spk])
                    alpha_used = float(info["alpha"])
                else:
                    kw = dict(cfg.get("kwargs", {}))
                    if kw.get("eq_mode") == "corpus_eq_split":
                        kw["src_corpus_env"] = s_corp_env
                        kw["tgt_corpus_env"] = t_corp_env
                    conv_f, info = adapt_features_v3(src_f, profs_v3[src_spk], profs_v3[tgt_spk], **kw)
                    alpha_used = float(info["alpha"])

                conv_w = _peak_limit(v6_voc.synthesize(conv_f, seed=72 + idx_u, phase_dispersion=True))
                sanity = eval_m.check_wav_sanity(conv_w, SR)
                qv = v6_eval._extract_clf_vec(conv_w, sr=SR)
                sd = eval_m.compute_spectral_distances(src_w, conv_w, sr=SR)
                cl = compute_clarity_metrics(conv_w, sr=SR)
                rows_by_method[m_name].append({
                    "pair": f"{src_spk}->{tgt_spk}",
                    "kind": pkind,
                    "sanity_ok": int(sanity["ok"]),
                    "hit_timbre4d": int(_classify_4way(qv, clf_train_rows, "timbre_4d") == tgt_spk),
                    "hit_ltas": int(_classify_4way(qv, clf_train_rows, "ltas_vec") == tgt_spk),
                    "hit_f0": int(_classify_4way(qv, clf_train_rows, "f0_stats") == tgt_spk),
                    "hit_all": int(_classify_4way(qv, clf_train_rows, "all_features") == tgt_spk),
                    "tilt_err": abs(float(qv["tilt"]) - t_ref["tilt"]),
                    "ltas_rmse": float(np.sqrt(np.mean((qv["ltas_vec"] - t_ref["ltas_vec"]) ** 2))),
                    "f0_err_st": abs(12.0 * math.log2(max(1.0, float(qv["f0_med"])) / max(1.0, t_ref["f0_med"]))),
                    "hnr_err": abs(float(qv["hnr"]) - t_ref["hnr"]),
                    "mel_lsd": float(sd["mel_lsd_db"]),
                    "mcd_cmn": float(sd["mcd_cmn_db"]),
                    "prom_d": cl["formant_prominence_db"] - src_cl["formant_prominence_db"],
                    "hf_d": cl["consonant_hf_ratio_db"] - src_cl["consonant_hf_ratio_db"],
                    "vtln": alpha_used,
                })
                if do_export and m_name in ("v8_decoupled", "v9_blind", "v9_corp_eq_k065"):
                    v6_voc.write_wav_mono(SAMPLES_V9_DIR / f"{tag_prefix}{src_spk}_to_{tgt_spk}_{m_name}.wav", _peak_limit(conv_w), SR)

    def _agg(rs: list[dict]) -> dict:
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "sanity_ok": int(sum(r["sanity_ok"] for r in rs)),
            "clf_timbre4d": round(float(np.mean([r["hit_timbre4d"] for r in rs])), 4),
            "clf_ltas": round(float(np.mean([r["hit_ltas"] for r in rs])), 4),
            "clf_f0": round(float(np.mean([r["hit_f0"] for r in rs])), 4),
            "clf_all": round(float(np.mean([r["hit_all"] for r in rs])), 4),
            "tilt_err_db_oct": round(float(np.mean([r["tilt_err"] for r in rs])), 3),
            "ltas_env_rmse_db": round(float(np.mean([r["ltas_rmse"] for r in rs])), 3),
            "f0_err_st": round(float(np.mean([r["f0_err_st"] for r in rs])), 3),
            "hnr_err_db": round(float(np.mean([r["hnr_err"] for r in rs])), 3),
            "content_mel_lsd_db": round(float(np.mean([r["mel_lsd"] for r in rs])), 3),
            "content_mcd_cmn_db": round(float(np.mean([r["mcd_cmn"] for r in rs])), 3),
            "formant_prom_delta_db": round(float(np.mean([r["prom_d"] for r in rs])), 3),
            "consonant_hf_delta_db": round(float(np.mean([r["hf_d"] for r in rs])), 3),
            "vtln_alpha_mean": round(float(np.mean([r["vtln"] for r in rs])), 4),
        }

    return {
        "n_speakers": len(spks),
        "profile_bytes_v3": {s: profs_v3[s].serialized_size_bytes() for s in spks},
        "utau_params_v3": {s: profs_v3[s].utau_params() for s in spks},
        "real_oracle_acc": {
            "timbre4d": round(real_acc["timbre4d"] / max(1, real_acc["n"]), 4),
            "ltas": round(real_acc["ltas"] / max(1, real_acc["n"]), 4),
            "f0": round(real_acc["f0"] / max(1, real_acc["n"]), 4),
            "all": round(real_acc["all"] / max(1, real_acc["n"]), 4),
            "n": real_acc["n"],
        },
        "overall": {m: _agg(rows_by_method[m]) for m in all_method_names},
        "by_pair_kind": {
            m: {k: _agg([r for r in rows_by_method[m] if r["kind"] == k]) for k in ("M->F", "F->M", "M->M", "F->F")}
            for m in all_method_names
        },
        "bdl_ssb0073_only": {
            m: _agg([r for r in rows_by_method[m] if r["pair"] in ("bdl->SSB0073", "SSB0073->bdl")])
            for m in all_method_names
        },
    }


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    en_by_spk: dict[str, list[dict]] = {}
    for u in manifest["english"]["utterances"]:
        en_by_spk.setdefault(u["speaker"], []).append({
            "id": u["id"],
            "speaker": u["speaker"],
            "gender": u["gender"],
            "path": str(ROOT / u["path"]),
            "active_start": None,
            "active_stop": None,
        })
    for s in en_by_spk:
        en_by_spk[s].sort(key=lambda r: r["id"])
    zh_by_spk = _load_zh_5speakers(manifest)

    base_methods = [
        ("v6_coupled", {"family": "v6"}),
        ("v8_decoupled", {"family": "v8"}),
        ("v9_blind", {"family": "v9", "kwargs": {"eq_mode": "blind_tilt_anchor"}}),
        ("v9_no_manner", {"family": "v9", "kwargs": {"eq_mode": "blind_tilt_anchor", "use_manner_delta": False}}),
        ("v9_no_utau", {"family": "v9", "kwargs": {"eq_mode": "blind_tilt_anchor", "use_utau_colors": False}}),
    ]
    cross_methods = base_methods + [
        ("v9_corp_eq_k00", {"family": "v9", "kwargs": {"eq_mode": "corpus_eq_split", "corpus_eq_keep": 0.0}}),
        ("v9_corp_eq_k065", {"family": "v9", "kwargs": {"eq_mode": "corpus_eq_split", "corpus_eq_keep": 0.65}}),
        ("v9_corp_eq_k10", {"family": "v9", "kwargs": {"eq_mode": "corpus_eq_split", "corpus_eq_keep": 1.0}}),
    ]

    # 1. 32-fold ZH + male_115
    res_32 = run_32fold_and_male_stress()

    # 2. Real 4-speaker English (n=60/method)
    en_spks = sorted(en_by_spk.keys())
    en_pairs = [(s, t) for s in en_spks for t in en_spks if s != t]
    res_en = _eval_corpus_suite(
        en_by_spk,
        {s: "arctic" for s in en_spks},
        en_pairs,
        n_prompt=3,
        n_clf_train=16,
        n_held=5,
        method_configs=base_methods,
        export_pairs={("bdl", "slt"), ("slt", "rms")},
        tag_prefix="en_",
    )

    # 3. Real 5-speaker Mandarin (n=60/method)
    zh_spks = sorted(zh_by_spk.keys())
    zh_pairs = [(s, t) for s in zh_spks for t in zh_spks if s != t]
    res_zh = _eval_corpus_suite(
        zh_by_spk,
        {s: "aishell3" for s in zh_spks},
        zh_pairs,
        n_prompt=3,
        n_clf_train=9,
        n_held=3,
        method_configs=base_methods,
        export_pairs={("SSB0073", "SSB0502"), ("SSB0502", "SSB0073")},
        tag_prefix="zh5_",
    )

    # 4. Real Bidirectional ZH <-> EN (n=24/method; prompt corpus baselines include all 4 EN & 5 ZH prompts)
    cl_rows = {
        "SSB0073": zh_by_spk["SSB0073"],
        "SSB0502": zh_by_spk["SSB0502"],
        "bdl": en_by_spk["bdl"],
        "slt": en_by_spk["slt"],
    }
    cl_pairs = [(s, t) for s in ("SSB0073", "SSB0502") for t in ("bdl", "slt")] + [
        (s, t) for s in ("bdl", "slt") for t in ("SSB0073", "SSB0502")
    ]
    res_cl = _eval_corpus_suite(
        cl_rows,
        {"SSB0073": "aishell3", "SSB0502": "aishell3", "bdl": "arctic", "slt": "arctic"},
        cl_pairs,
        n_prompt=3,
        n_clf_train=9,
        n_held=3,
        method_configs=cross_methods,
        export_pairs={("bdl", "SSB0073"), ("SSB0073", "bdl"), ("bdl", "SSB0502"), ("SSB0502", "bdl")},
        tag_prefix="cl_",
    )

    out = {
        "zh_32fold_and_male115": res_32,
        "english_4way": res_en,
        "mandarin_5way": res_zh,
        "cross_lingual_zh_en": res_cl,
    }
    out_path = EXP_DIR / "results_v9.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("EN overall:", json.dumps(res_en["overall"], indent=2))
    print("ZH5 overall:", json.dumps(res_zh["overall"], indent=2))
    print("ZH<->EN overall:", json.dumps(res_cl["overall"], indent=2))
    print("bdl<->SSB0073:", json.dumps(res_cl["bdl_ssb0073_only"], indent=2))


if __name__ == "__main__":
    main()
