"""Fit and evaluate a fixed broadband EQ for the Stage 11 projected EN voice."""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from exp_20_pybot_larynx import verify_stage10_bilingual as baseline

import numpy as np
from scipy.ndimage import gaussian_filter1d

from exp_20_pybot_larynx import N_BAP, N_BINS, SR
from exp_20_pybot_larynx.crossling import en_stage10_renderer as renderer
from exp_20_pybot_larynx.larynx.synth import LarynxControl, cep_to_log_env, log_env_to_cep
from exp_20_pybot_larynx.verify_stage10_onsets import NEW_TEXTS, _human_switches, _render, _switch_features
from pybot_webui import crossling


PHONES = ("i", "u", "m", "n")
FREQ = np.linspace(0, SR / 2, N_BINS)
LEVEL = (FREQ >= 300) & (FREQ <= 2600)
DB = 20 / np.log(10)
REFERENCE_PATH = baseline.EXP_DIR / "crossling_data" / "stage12_ascend_phone_reference.json"
OUTPUT = baseline.EXP_DIR / "samples_stage12_phone_matched"


def shape(env):
    return env - np.mean(env[..., LEVEL], axis=-1, keepdims=True)


def smooth(env):
    return gaussian_filter1d(env, sigma=28.0, axis=-1)


def language_offset(groups, split, phone, minimum):
    zh, en = (groups.get(f"{split}/{lang}/{phone}") for lang in ("zh", "en"))
    if zh is None or en is None or min(zh["count"], en["count"]) < minimum:
        return None
    taper = 0.5 + 0.5 * np.cos(np.pi * np.clip((FREQ - 5200) / 2000, 0, 1))
    delta = shape(smooth(np.asarray(en["log_env"]) - np.asarray(zh["log_env"])))
    return 0.5 * taper * np.clip(delta, -2 / DB, 2 / DB)


def fit_eq(zh_kb, groups):
    en_kb = renderer.get_en_stage10_knowledge()
    bank = renderer.get_speaker_donor_bank(zh_kb)
    en_anchors, bap, zh_anchors = [], [], {}
    for phone in PHONES:
        medoids = en_kb.ctx_p[phone]
        cep = np.average(np.mean(medoids.cep[:, 1:4], axis=1), axis=0, weights=medoids.cluster_sz)
        en_anchors.append(cep_to_log_env(cep[None], n_bins=N_BINS)[0])
        bap.append(np.average(np.mean(medoids.bap[:, 1:4], axis=1), axis=0, weights=medoids.cluster_sz))
        zh_anchors[phone] = np.asarray(bank.get_sonorant_donor(phone).env)
    voiced = np.ones(len(PHONES), dtype=bool)
    control = LarynxControl(np.full(len(PHONES), en_kb.speaker_f0_med_hz), voiced, np.array(en_anchors), np.reshape(bap, (-1, N_BAP)))
    mapped, _, _ = renderer.apply_clone_to_control(
        control, renderer._load_en_source_profile(), renderer._load_base_profile_cached(),
        strength=0.92, clone_f0=False, clone_glottal=False,
    )
    cep = log_env_to_cep(control.log_env, n_cep=26)
    q = max(renderer._MIN_Q8_24_TARGET, 0.98 * np.median(np.linalg.norm(cep[:, 8:24], axis=1)))
    mapped_env = renderer._preserve_q8_24_sharpness(mapped.log_env, voiced, q)
    residuals = []
    for idx, phone in enumerate(PHONES):
        offset = language_offset(groups, "fit", phone, minimum=4)
        residual = shape(smooth(zh_anchors[phone] - mapped_env[idx]))
        residuals.append(residual if offset is None else residual + offset)
    eq = np.clip(0.7 * np.median(residuals, axis=0), -0.90, 0.70)
    hf_taper = 0.5 + 0.5 * np.cos(np.pi * np.clip((FREQ - 4500) / 3000, 0, 1))
    return np.minimum(eq, 0) + np.maximum(eq, 0) * hf_taper, zh_anchors


def collect_spans(control, plans, targets):
    rows = []
    for phone in plans:
        ipa = phone["ipa"]
        if ipa not in targets:
            continue
        start, end = phone["start_fr"], phone["end_fr"]
        margin = max(1, int(0.22 * (end - start)))
        a, b = start + margin, end - margin
        voiced = control.voiced[a:b]
        if voiced.sum() < 3 or np.mean(voiced) < 0.75:
            continue
        env = np.median(control.log_env[a:b][voiced], axis=0)
        error = shape(smooth(env)) - targets[ipa]
        bands = [(300, 7000), (300, 10500)]
        rows.append((ipa, [float(DB * np.sqrt(np.mean(error[(FREQ >= lo) & (FREQ < hi)] ** 2))) for lo, hi in bands]))
    return rows


def controls_agree(a, b):
    return all(np.array_equal(getattr(a, key), getattr(b, key)) for key in (
        "f0_hz", "voiced", "bap", "jitter_frac", "shimmer_frac", "glottal_tilt_db_per_oct", "rd",
    ))


def voiced_treble(wav, control):
    # Use the same known voiced-frame indices in each arm, avoiding F0 estimator bias.
    hop = control.hop
    window = 4 * hop
    stable = np.convolve(control.voiced, np.ones(5), mode="same") == 5
    indices = np.flatnonzero(stable & (control.f0_hz > 40))
    indices = indices[indices * hop + window // 2 < len(wav)]
    frames = np.pad(wav, window // 2)[indices[:, None] * hop + np.arange(window)]
    power = np.abs(np.fft.rfft(frames * np.hanning(window), n=2 * (N_BINS - 1))) ** 2
    ref = (FREQ >= 120) & (FREQ <= 2200)
    high = (FREQ >= 8000) & (FREQ <= 10500)
    return float(np.median(10 * np.log10((power[:, high].sum(axis=1) + 1e-16) / (power[:, ref].sum(axis=1) + 1e-16))))


def main():
    kb = baseline.AcousticKnowledgeStage7.load(baseline.STAGE7_KNOWLEDGE_PATH)
    arb, rules = baseline.Arbiter(), baseline.default_stage5_rules()
    params = baseline.VoiceParams.from_dict({"preset_id": "female_std", "auto_breath": False})
    phasor = np.asarray(baseline.load_speaker_assets_v15()["glottal_bank"].coherent_phasor)
    groups = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))["groups"]
    eq, anchors = fit_eq(kb, groups)
    upper_blend = 0.5 - 0.5 * np.cos(np.pi * np.clip((FREQ - 4000) / 3500, 0, 1))
    arms = {
        "before": None,
        "after": renderer.MatchedVoiceEQ(eq, np.ones(N_BINS)),
        "hybrid": renderer.MatchedVoiceEQ(eq, 1.0 - 0.65 * upper_blend),
    }
    targets = {}
    for phone in PHONES:
        offset = language_offset(groups, "check", phone, minimum=3)
        targets[phone] = shape(smooth(anchors[phone]))
        if offset is not None:
            targets[phone] += offset
    refs = json.loads(baseline.BILINGUAL_REF_STATS_PATH.read_text(encoding="utf-8"))["bilingual_ref_clips"]
    with baseline.STAGE10_HELDOUT_PKL.open("rb") as stream:
        held = pickle.load(stream)
    held = held["utterances"] if isinstance(held, dict) else held
    cases = [("english", str(row["id"]), row["text"], None) for row in held[:20]]
    cases += [("reference", row["ascend_id"], row["text"], row) for row in refs]
    cases += [("new", str(i), txt, None) for i, txt in enumerate(NEW_TEXTS)]
    cases += [("chinese", "zh", "早上的新消息已经发到了，你回去以后记得看一下。", None)]

    real_build = crossling.build_stage10_en_control
    predict_phones = renderer._predict_en_prosody_stage10_or_fallback
    samples, rows, all_spans, switches = [], [], defaultdict(list), defaultdict(list)
    OUTPUT.mkdir(exist_ok=True)
    for group, key, sentence, reference in cases:
        print(group, key, flush=True)
        uid = f"s12_{group}_{key}"
        raw_runs = {}
        metrics = {"group": group, "id": key, "text": sentence}
        ref_edges = _human_switches(reference) if reference else []
        variants = {}
        baseline_edges = None
        for arm, correction in arms.items():
            runs = []
            plans = []

            def capture_phones(*args, **kwargs):
                result = predict_phones(*args, **kwargs)
                plans[:] = result[0]
                return result

            def capture_run(*args, **kwargs):
                out = real_build(*args, **kwargs, matched_voice_eq=correction)
                runs.append((out[0].copy(), plans.copy()))
                return out

            with (
                patch.object(crossling, "build_stage10_en_control", side_effect=capture_run),
                patch.object(renderer, "_predict_en_prosody_stage10_or_fallback", side_effect=capture_phones),
            ):
                wav, ctrl, edges = _render(sentence, uid, kb, arb, rules, params, phasor, anchor_pitch=True)
            raw_runs[arm] = runs
            variants[arm] = wav, ctrl
            if baseline_edges is None:
                baseline_edges = edges
            else:
                assert edges == baseline_edges, uid
            for run, phones in runs:
                for ipa, error in collect_spans(run, phones, targets):
                    if group == "english":
                        all_spans[(arm, ipa)].append(error)
            if group == "english":
                cep = log_env_to_cep(ctrl.log_env[ctrl.voiced], n_cep=24)
                metrics[arm] = {
                    "voiced_8k_10k5_db": round(voiced_treble(wav, ctrl), 3),
                    "control_q8_24": round(float(np.mean(np.linalg.norm(cep[:, 8:24], axis=1))), 3),
                    "active_rms_db": round(baseline._active_rms_db(wav), 3),
                }
            ref_edges_match = [direction for direction, _ in ref_edges] == [edge[0] for edge in edges]
            for index, (direction, bounds, _, contiguous) in enumerate(edges):
                feature = _switch_features(wav, ctrl, *bounds)
                if feature is not None:
                    switches[(group, arm, direction)].append(feature)
                    if contiguous:
                        switches[(group, arm, "contiguous")].append(feature)
                    if ref_edges_match and ref_edges[index][1] is not None:
                        switches[(group, arm, "ref_error")].append(feature - ref_edges[index][1])

        for arm in ("after", "hybrid"):
            before, candidate = raw_runs["before"], raw_runs[arm]
            assert len(before) == len(candidate)
            for (src, plans_a), (out, plans_b) in zip(before, candidate):
                assert plans_a == plans_b and controls_agree(src, out), uid
                assert np.array_equal(src.log_env[~src.voiced], out.log_env[~out.voiced]), uid
            assert controls_agree(variants["before"][1], variants[arm][1]), uid
            assert len(variants["before"][0]) == len(variants[arm][0]), uid
            if group == "chinese":
                assert np.array_equal(variants["before"][0], variants[arm][0])
        assert all(np.all(np.isfinite(wav)) and np.max(np.abs(wav)) <= 1.0 for wav, _ in variants.values()), uid
        rows.append(metrics)

        if group == "reference" or (group, key) in {("new", "0"), ("new", "6"), ("new", "10"), ("chinese", "zh")} or (group == "english" and len(samples) < 4):
            clips = {}
            for arm, (wav, _) in variants.items():
                file = OUTPUT / f"{uid}_{arm}.wav"
                baseline._write_pcm16_wav(file, wav, SR)
                clips[arm] = file.name
            if reference:
                file = OUTPUT / f"{uid}_reference.wav"
                wav, _ = baseline.v6_voc.read_wav_mono(Path(reference["wav_path"]))
                baseline._write_pcm16_wav(file, baseline._rms_norm(wav, -19.5), SR)
                clips["reference"] = file.name
            samples.append({"group": group, "text": sentence, "tracks": clips})

    results = {
        "fit_counts": {k: v["count"] for k, v in groups.items()},
        "curve_db_at_hz": {hz: round(float(DB * np.interp(hz, FREQ, eq)), 3) for hz in (250, 500, 1000, 2000, 3500, 5000, 6500, 8500, 10000)},
        "matched_errors_by_phone": {f"{arm}/{ipa}": {"n": len(vals), "median_db_300_7000_300_10500": np.round(np.median(vals, axis=0), 3).tolist()} for (arm, ipa), vals in all_spans.items()},
        "matched_errors_pooled": {arm: np.round(np.median([row for (key, _), vals in all_spans.items() if key == arm for row in vals], axis=0), 3).tolist() for arm in arms},
        "mixed_switches": {"/".join(key): {"n": len(vals), "med_abs_level_db_pitch_st_band_distance_db": np.round(np.median(np.abs(vals), axis=0), 3).tolist()} for key, vals in switches.items()},
        "english_waveform_medians": {
            arm: {metric: round(float(np.median([row[arm][metric] for row in rows if row["group"] == "english"])), 3) for metric in ("voiced_8k_10k5_db", "control_q8_24", "active_rms_db")}
            for arm in arms
        },
        "cases": rows,
        "audition": samples,
    }
    (OUTPUT / "stage12_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k not in {"cases", "audition"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
