"""Phase 9 (v13) Virtual Singer / UTAU Ceiling Demo: Sustained Ping-Pong Loops, Portamento & Vibrato."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp_v5_arch import crossling_frontend as cf  # noqa: E402
from exp_v5_eval.metrics import check_wav_sanity  # noqa: E402
from exp_v6_vocoder import vocoder  # noqa: E402
from exp_v9_eval.v9_metrics import compute_v9_extended_metrics  # noqa: E402
from exp_v13_utau_ceiling import EXP_DIR, HOP, SAMPLES_DIR, SR  # noqa: E402
from exp_v13_utau_ceiling.eval_v13 import _rms_norm, _write_pcm16_wav  # noqa: E402
from exp_v13_utau_ceiling.v13_renderer import V13Config, render_singing_score_v13  # noqa: E402
from exp_v13_utau_ceiling.zh_bank_v13 import V13ZhBank, build_v13_zh_bank  # noqa: E402

LEAD_SIL_FRAMES = 4


# Two lyrical/rhythmic Mandarin singing scores whose syllables are 100% covered in SSB0502's recorded bank
DEMO_SCORES: dict[str, dict[str, Any]] = {
    "demo1_lyrical_ballad": {
        "title": "Demo 1 · 抒情长音吟唱（明夜在湖上，长歌和星华）",
        "description": "长音延展 420–860 ms、±6 半音音域、连音滑音 (40 ms portamento) 与延迟颤音 (5.4 Hz, 0.24 st)",
        "notes": [
            {"hanzi": "明", "pinyin": "ming2", "midi_note": 57.0, "dur_ms": 420.0, "portamento_ms": 0.0, "vibrato_depth_st": 0.15},
            {"hanzi": "夜", "pinyin": "ye4", "midi_note": 59.0, "dur_ms": 460.0, "portamento_ms": 40.0, "vibrato_depth_st": 0.18},
            {"hanzi": "在", "pinyin": "zai4", "midi_note": 62.0, "dur_ms": 440.0, "portamento_ms": 40.0, "vibrato_depth_st": 0.20},
            {"hanzi": "湖", "pinyin": "hu2", "midi_note": 64.0, "dur_ms": 540.0, "portamento_ms": 45.0, "vibrato_depth_st": 0.24},
            {"hanzi": "上", "pinyin": "shang4", "midi_note": 62.0, "dur_ms": 780.0, "portamento_ms": 45.0, "vibrato_depth_st": 0.28, "vibrato_delay_ms": 180.0},
            {"rest_ms": 180.0},
            {"hanzi": "长", "pinyin": "chang2", "midi_note": 60.0, "dur_ms": 440.0, "portamento_ms": 0.0, "vibrato_depth_st": 0.18},
            {"hanzi": "歌", "pinyin": "ge1", "midi_note": 64.0, "dur_ms": 500.0, "portamento_ms": 40.0, "vibrato_depth_st": 0.22},
            {"hanzi": "和", "pinyin": "he2", "midi_note": 65.0, "dur_ms": 460.0, "portamento_ms": 40.0, "vibrato_depth_st": 0.22},
            {"hanzi": "星", "pinyin": "xing1", "midi_note": 62.0, "dur_ms": 480.0, "portamento_ms": 40.0, "vibrato_depth_st": 0.22},
            {"hanzi": "华", "pinyin": "hua2", "midi_note": 59.0, "dur_ms": 860.0, "portamento_ms": 50.0, "vibrato_depth_st": 0.30, "vibrato_delay_ms": 200.0},
        ],
    },
    "demo2_rhythm_octave": {
        "title": "Demo 2 · 快慢交错 + 八度大跳压力测试（出发到地方，不止在今生）",
        "description": "150 ms 短促音节与 660 ms 延音交替，跨越一个八度 (F3=53 -> F4=65)，检验清辅音清晰度与循环稳定性",
        "notes": [
            {"hanzi": "出", "pinyin": "chu1", "midi_note": 55.0, "dur_ms": 200.0, "portamento_ms": 0.0, "vibrato_depth_st": 0.0},
            {"hanzi": "发", "pinyin": "fa1", "midi_note": 57.0, "dur_ms": 200.0, "portamento_ms": 25.0, "vibrato_depth_st": 0.0},
            {"hanzi": "到", "pinyin": "dao4", "midi_note": 60.0, "dur_ms": 280.0, "portamento_ms": 30.0, "vibrato_depth_st": 0.15},
            {"hanzi": "地", "pinyin": "di4", "midi_note": 62.0, "dur_ms": 180.0, "portamento_ms": 25.0, "vibrato_depth_st": 0.0},
            {"hanzi": "方", "pinyin": "fang1", "midi_note": 65.0, "dur_ms": 620.0, "portamento_ms": 40.0, "vibrato_depth_st": 0.25, "vibrato_delay_ms": 160.0},
            {"rest_ms": 140.0},
            {"hanzi": "不", "pinyin": "bu4", "midi_note": 53.0, "dur_ms": 170.0, "portamento_ms": 0.0, "vibrato_depth_st": 0.0},
            {"hanzi": "止", "pinyin": "zhi3", "midi_note": 57.0, "dur_ms": 220.0, "portamento_ms": 30.0, "vibrato_depth_st": 0.0},
            {"hanzi": "在", "pinyin": "zai4", "midi_note": 60.0, "dur_ms": 240.0, "portamento_ms": 30.0, "vibrato_depth_st": 0.0},
            {"hanzi": "今", "pinyin": "jin1", "midi_note": 64.0, "dur_ms": 360.0, "portamento_ms": 35.0, "vibrato_depth_st": 0.18},
            {"hanzi": "生", "pinyin": "sheng1", "midi_note": 65.0, "dur_ms": 680.0, "portamento_ms": 40.0, "vibrato_depth_st": 0.26, "vibrato_delay_ms": 180.0},
        ],
    },
}


def verify_score_bank_coverage(notes: list[dict[str, Any]], bank: V13ZhBank, speaker: str = "SSB0502") -> list[dict[str, Any]]:
    """Verify every sung note maps to an existing recorded syllable in `speaker`'s bank."""
    cov = []
    for item in notes:
        if "rest_ms" in item and "midi_note" not in item:
            continue
        o, f, t = cf.pinyin_to_ipa(str(item["pinyin"]))[:3]
        if o in ("y", "w"):
            o = ""
        if f == "i" and o in ("zh", "ch", "sh", "r", "z", "c", "s"):
            f = "ir" if o in ("zh", "ch", "sh", "r") else "iz"
        cands = [c for c in bank.zh_syllables.get((o, f), []) if c.speaker == speaker]
        if not cands:
            raise ValueError(f"Singing syllable {item['hanzi']} ({o},{f}) missing from {speaker} bank")
        cov.append({"hanzi": item["hanzi"], "key": f"{o}/{f}", "n_spk_tokens": len(cands)})
    return cov


def evaluate_singing_accuracy(y: np.ndarray, trace: dict[str, Any]) -> dict[str, Any]:
    """Measure pitch tracking accuracy (against exact score MIDI/vibrato curve) & acoustic sanity."""
    san = check_wav_sanity(y, SR)
    v9 = compute_v9_extended_metrics(y, sr=SR, fmin=75.0, fmax=450.0)
    n_fr = int(trace.get("n_frames", max(1, len(y) // HOP)))
    pad_half = 480
    f0_voc, voi_voc = vocoder._extract_f0_padded(np.pad(np.asarray(y, dtype=np.float64), (pad_half, pad_half)), n_fr, pad_half, sr=SR, hop=HOP, win_len=2 * pad_half)

    slots = trace.get("slot_timings_ms") or []
    score_notes = trace.get("score_notes") or []

    dur_errs: list[float] = []
    note_pitch_errs: list[float] = []
    steady_pitch_errs: list[float] = []

    for (s_ms, e_ms), sn in zip(slots, score_notes):
        dur_errs.append(abs((e_ms - s_ms) - float(sn["dur_ms"])))
        target_hz_track = np.asarray(sn["f0_hz_track"], dtype=np.float64)
        s_fr = int(round(s_ms / 5.0))
        e_fr = int(round(e_ms / 5.0))
        steady_fr = s_fr + max(11, int(round(0.25 * (e_fr - s_fr))))
        for fr in range(max(0, s_fr + 6), min(len(f0_voc), e_fr - 5)):
            if voi_voc[fr] and f0_voc[fr] > 40.0:
                rel_fr = int(np.clip(fr - s_fr, 0, len(target_hz_track) - 1))
                f_tgt = float(target_hz_track[rel_fr])
                if f_tgt > 40.0:
                    err_st = abs(12.0 * math.log2(float(f0_voc[fr]) / f_tgt))
                    note_pitch_errs.append(err_st)
                    if fr >= steady_fr:
                        steady_pitch_errs.append(err_st)

    return {
        "sanity_ok": bool(san["ok"]),
        "duration_s": san["duration_s"],
        "dur_mae_ms": round(float(np.mean(dur_errs)) if dur_errs else 0.0, 2),
        "f0_all_mae_st": round(float(np.mean(note_pitch_errs)) if note_pitch_errs else float("nan"), 4),
        "f0_steady_mae_st": round(float(np.mean(steady_pitch_errs)) if steady_pitch_errs else float("nan"), 4),
        "f0_steady_p95_st": round(float(np.percentile(steady_pitch_errs, 95)) if steady_pitch_errs else float("nan"), 4),
        "clicks_gt8": round(float(v9["click_outlier_rate_gt8"]), 5),
        "voiced_contrast_db": round(float(v9["clarity"]["voiced_spectral_contrast_db"]), 4),
        "pingpong_count": int(trace.get("pingpong_count", 0)),
        "source_counts": trace.get("source_counts", {}),
    }


def run_singing_demos(bank: V13ZhBank | None = None, v13_cfg: V13Config | None = None) -> dict[str, Any]:
    """Render both virtual-singer demos (v12 linear stretch vs v13 UTAU ping-pong) and export WAVs."""
    if bank is None:
        bank = build_v13_zh_bank(use_cache=True, fix_f_split=True)
    prof = bank.profiles_v5["SSB0502"]
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    cfg_v12_sing = V13Config(
        fix_viterbi_join=False,
        soft_context_and_f0_sel=False,
        joint_of_concat=False,
        gentle_gain=False,
        steady_nucleus_stretch=False,
        allow_pingpong=False,
        spectral_seam_join=False,
        f0_tilt_comp=False,
        use_sandhi_v13=False,
        use_learned_prosody=False,
        singing_mode=True,
    )
    if v13_cfg is None:
        cfg_v13_sing = V13Config(
            fix_viterbi_join=True,
            soft_context_and_f0_sel=True,
            joint_of_concat=True,
            gentle_gain=True,
            steady_nucleus_stretch=True,
            allow_pingpong=True,
            spectral_seam_join=True,
            f0_tilt_comp=False,
            use_sandhi_v13=False,
            use_learned_prosody=False,
            singing_mode=True,
        )
    else:
        cfg_v13_sing = V13Config(**{
            **v13_cfg.__dict__,
            "steady_nucleus_stretch": True,
            "allow_pingpong": True,
            "use_sandhi_v13": False,
            "use_learned_prosody": False,
            "singing_mode": True,
        })

    out_demos: dict[str, Any] = {}
    gap = np.zeros(int(0.45 * SR), dtype=np.float64)

    for key, spec in DEMO_SCORES.items():
        cov = verify_score_bank_coverage(spec["notes"], bank, speaker="SSB0502")
        y12, tr12 = render_singing_score_v13(spec["notes"], bank, prof, cfg=cfg_v12_sing, seed=72)
        m12 = evaluate_singing_accuracy(y12, tr12)

        y13, tr13 = render_singing_score_v13(spec["notes"], bank, prof, cfg=cfg_v13_sing, seed=72)
        m13 = evaluate_singing_accuracy(y13, tr13)

        fn12 = f"{key}_v12_linear.wav"
        fn13 = f"{key}_v13_utau.wav"
        fn_ab = f"{key}_AB_v12_vs_v13.wav"
        _write_pcm16_wav(SAMPLES_DIR / fn12, _rms_norm(y12))
        _write_pcm16_wav(SAMPLES_DIR / fn13, _rms_norm(y13))
        _write_pcm16_wav(SAMPLES_DIR / fn_ab, np.concatenate([_rms_norm(y12), gap, _rms_norm(y13)]))

        out_demos[key] = {
            "title": spec["title"],
            "description": spec["description"],
            "coverage": cov,
            "v12_linear": {"file": f"samples/{fn12}", "metrics": m12},
            "v13_utau": {"file": f"samples/{fn13}", "metrics": m13},
            "ab_file": f"samples/{fn_ab}",
        }

    (EXP_DIR / "singing_demo_results.json").write_text(json.dumps(out_demos, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== VIRTUAL SINGER DEMO METRICS (SSB0502, 100% Recorded Bank) ===")
    for k, d in out_demos.items():
        print(f"[{k}] {d['title']}")
        print("  v12_linear:", d["v12_linear"]["metrics"])
        print("  v13_utau:  ", d["v13_utau"]["metrics"])
    return out_demos


if __name__ == "__main__":
    run_singing_demos()
