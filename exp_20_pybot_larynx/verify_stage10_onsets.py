"""Measure the effect of English onset anchoring in mixed speech, with paired renders."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from exp_20_pybot_larynx import verify_stage10_bilingual as stage10

import numpy as np

from exp_20_pybot_larynx.crossling import en_stage10_renderer
from exp_20_pybot_larynx.crossling.en_articulation import VOWEL_IPAS
from exp_v6_crossling_dsp.ipa_acoustics import IPA_SPECS
from exp_v10_g2p_and_align.landmark_aligner import align_ipa_sequence_v10
from pybot_webui import crossling


NEW_TEXTS = (
    "我把最新的 audio preview 放进文件夹了。",
    "你刚刚点的是 open file 还是 save as？",
    "这个 meeting schedule 更新以后，记得同步给我。",
    "Let me know 你今天几点出发。",
    "先打一个 tag，then run the nightly build，最后把 log 发给我。",
    "我说的是 ice cream，你听成了 I scream 吗？",
    "她把 red leather 和 blue ribbon 都放回柜子了。",
    "A quick note 这个名字叫 PyBot Studio，别拼错。",
    "我选了 green tea 按你的清单买的。",
    "Deliver it tomorrow 安排好了。",
    "我想要 blue 还是 white 你觉得呢？",
    "这个版本号是 seven 发出去就行。",
    "我们到了 play area 那边就下车。",
    "Next issue 看起来容易解决。",
)

_FFT = 1024
_HOP = stage10.SR // 200
_WINDOW = 4 * _HOP
_FREQ = np.fft.rfftfreq(_FFT, 1 / stage10.SR)
_BANDS = [(100, 500), (500, 1500), (1500, 3500), (3500, 6000), (6000, 10000)]
_BAND_MASKS = [(_FREQ >= lo) & (_FREQ < hi) for lo, hi in _BANDS]


def _voiced_features(wav, track, start, stop, *, tail):
    voiced = track.voiced[start:stop] & (track.f0_hz[start:stop] > 40)
    indices = np.flatnonzero(voiced) + start
    indices = indices[-16:] if tail else indices[:16]
    if len(indices) < 4:
        return None
    window = np.pad(wav, _WINDOW // 2)[indices[:, None] * _HOP + np.arange(_WINDOW)]
    level = np.median(10 * np.log10(np.mean(window**2, axis=1) + 1e-18))
    pitch = np.median(12 * np.log2(track.f0_hz[indices]))
    power = np.abs(np.fft.rfft(window * np.hanning(_WINDOW), n=_FFT))**2
    band_db = 10 * np.log10([np.mean(power[:, mask].sum(axis=1)) + 1e-18 for mask in _BAND_MASKS])
    return level, pitch, band_db - np.mean(band_db)


def _switch_features(wav, track, left_start, left_stop, right_start, right_stop):
    left = _voiced_features(wav, track, max(left_start, left_stop - 32), left_stop, tail=True)
    right = _voiced_features(wav, track, right_start, min(right_stop, right_start + 32), tail=False)
    if left is None or right is None:
        return None
    return np.array([
        right[0] - left[0],
        right[1] - left[1],
        np.sqrt(np.mean((right[2] - left[2])**2)),
    ])


def _render(text, uid, kb, arbiter, rules, params, phasor, *, anchor_pitch):
    regions = []
    roles = {}
    en_tail = None
    build_zh = crossling.build_zh_run_control
    build_en = crossling.build_en_run_control
    build_acoustics = en_stage10_renderer.build_en_stage10_control
    concat = crossling._concat_controls

    def track_phones(phones, *args, **kwargs):
        nonlocal en_tail
        en_tail = next((phone["ipa"] for phone in reversed(phones) if not phone["is_pause"]), None)
        return build_acoustics(phones, *args, **kwargs)

    def track_zh(*args, **kwargs):
        ctrl, meta = build_zh(*args, **kwargs)
        roles[id(ctrl)] = ("zh", None)
        return ctrl, meta

    def track_en(*args, **kwargs):
        nonlocal en_tail
        en_tail = None
        ctrl, subharm = build_en(*args, **kwargs)
        roles[id(ctrl)] = ("en", en_tail)
        return ctrl, subharm

    def track_regions(chunks):
        start = 0
        for i, (ctrl, pause) in enumerate(chunks):
            stop = start + len(ctrl.voiced)
            if id(ctrl) in roles:
                regions.append((i, start, stop, *roles[id(ctrl)]))
            start = stop + (round(pause / 5) if i + 1 < len(chunks) and pause >= 20 else 0)
        return concat(chunks)

    anchor = crossling._anchor_english_onset_pitch
    with (
        patch.object(crossling, "_anchor_english_onset_pitch", side_effect=anchor if anchor_pitch else None),
        patch.object(crossling, "build_zh_run_control", side_effect=track_zh),
        patch.object(crossling, "build_en_run_control", side_effect=track_en),
        patch.object(crossling, "_concat_controls", side_effect=track_regions),
        patch.object(en_stage10_renderer, "build_en_stage10_control", side_effect=track_phones),
    ):
        ctrl, wav, _, _ = stage10._synth_stage10_utt(text, uid, kb, arbiter, rules, params, phasor)
    switches = []
    for left, right in zip(regions, regions[1:]):
        li, la, lb, lang_left, tail = left
        ri, ra, rb, lang_right, _ = right
        if lang_left != lang_right:
            switches.append((lang_left + lang_right, (la, lb, ra, rb), tail, ri == li + 1 and lb == ra))
    return wav, ctrl, switches


def _human_switches(row):
    wav, _ = stage10.v6_voc.read_wav_mono(Path(row["wav_path"]))
    track = stage10.v6_voc.analyze(wav, sr=stage10.SR)
    units = [unit for unit in crossling._safe_plan_v13(row["text"])["units"] if unit.get("ipa_phones")]
    phones, switches = [], []
    for i, unit in enumerate(units):
        if i and units[i - 1]["lang"] != unit["lang"]:
            switches.append((units[i - 1]["lang"] + unit["lang"], len(phones)))
        phones.extend(unit["ipa_phones"])
    bounds, info = align_ipa_sequence_v10(phones, track, sr=stage10.SR)
    if not info.get("dp_ok") or len(bounds) != len(phones):
        return []
    return [
        (direction, _switch_features(wav, track, 0, bounds[i - 1][1], bounds[i][0], len(track.voiced)))
        for direction, i in switches
    ]


def _phone_group(ipa):
    if ipa in VOWEL_IPAS:
        return "vowel"
    spec = IPA_SPECS.get(ipa)
    return spec.manner if spec is not None else "other"


def _result(values):
    data = np.abs(np.asarray(values))
    return {"n": len(data), "median_abs": np.round(np.median(data, axis=0), 3).tolist()}


def main():
    kb = stage10.AcousticKnowledgeStage7.load(stage10.STAGE7_KNOWLEDGE_PATH)
    arbiter = stage10.Arbiter()
    rules = stage10.default_stage5_rules()
    phasor = np.asarray(stage10.load_speaker_assets_v15()["glottal_bank"].coherent_phasor)
    refs = json.loads(stage10.BILINGUAL_REF_STATS_PATH.read_text(encoding="utf-8"))["bilingual_ref_clips"]
    reference_switches = {row["ascend_id"]: _human_switches(row) for row in refs}
    results = {}
    for preset in ("female_std", "male_young"):
        params = stage10.VoiceParams.from_dict({"preset_id": preset, "auto_breath": False})
        for group, cases in (
            ("ascend", [(row["ascend_id"], row["text"]) for row in refs]),
            ("new", [(f"new{i}", text) for i, text in enumerate(NEW_TEXTS)]),
        ):
            paired, with_human = {}, {}
            for uid, text in cases:
                args = (text, uid, kb, arbiter, rules, params, phasor)
                before, tc_before, edges = _render(*args, anchor_pitch=False)
                after, tc_after, new_edges = _render(*args, anchor_pitch=True)
                assert edges == new_edges
                human = reference_switches.get(uid, [])
                measured_before = stage10.v6_voc.analyze(before, sr=stage10.SR) if human else None
                measured_after = stage10.v6_voc.analyze(after, sr=stage10.SR) if human else None
                human_directions_match = [direction for direction, _ in human] == [edge[0] for edge in edges]
                for index, (direction, bounds, phone, connected) in enumerate(edges):
                    old = _switch_features(before, tc_before, *bounds)
                    new = _switch_features(after, tc_after, *bounds)
                    if old is not None and new is not None and connected:
                        paired.setdefault(direction, []).append((old, new))
                        if direction == "enzh":
                            paired.setdefault(f"enzh_{_phone_group(phone)}", []).append((old, new))
                    if not human_directions_match:
                        continue
                    _, real = human[index]
                    old = _switch_features(before, measured_before, *bounds)
                    new = _switch_features(after, measured_after, *bounds)
                    if real is not None and old is not None and new is not None:
                        with_human.setdefault(direction, []).append((real, old, new))
            results[f"{preset}_{group}"] = {
                "connected_control_f0_before_after": {key: _result(vals) for key, vals in paired.items()},
                "waveform_f0_human_before_after": {key: _result(vals) for key, vals in with_human.items()},
            }
    print(json.dumps({
        "column_order": ["level_db", "pitch_st", "level_normalized_spectrum_rmse_db"],
        "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
