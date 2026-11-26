"""Phase-8 Evaluation: `HistGradientBoosting` vs `Ridge` Prosody & CS Seam Bridge (`exp_v12_prosody`)."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_v6_vocoder import vocoder as v6_voc
from exp_v10_g2p_and_align.g2p_audit_and_fix import plan_v10
from exp_v11_align_gmm import SR
from exp_v11_align_gmm.gmm_hmm_aligner import EN_HELDOUT_DEFAULT, align_en_corpus_gmm, align_zh_corpus_gmm
from exp_v11_eval.v11_metrics import EN_FUNCTION_WORDS
from exp_v12_prosody import DEFAULT_MODEL_PATH, EXP_DIR
from exp_v12_prosody.cs_bridge import bridge_cs_boundary
from exp_v12_prosody.prosody_model import ProsodyModelV12, apply_prosody_v12, fit_prosody_v12


def eval_cs_bridge_on_real_units(zh_res: dict, en_res: dict) -> dict:
    """Measure CS `ZH <-> EN` seam `feature_join_log_env_l2_db` before vs after `bridge_cs_boundary`."""
    zh_items = list(zh_res["utterances"].values())[:10]
    en_items = list(en_res["utterances"].values())[:10]

    before_l2, after_l2 = [], []
    before_uv_l2, after_uv_l2 = [], []
    for zi, ei in zip(zh_items, en_items):
        zf = zi["feats"]
        ef = ei["feats"]
        z_au = zi["aligned"]
        e_au = ei["aligned"]
        # Concatenate a ZH syllable with an EN phone boundary (both ZH->EN and EN->ZH)
        if not z_au.syllables or len(e_au.phones) < 2:
            continue
        sy = z_au.syllables[len(z_au.syllables) // 2]
        pb_v = next((p for p in e_au.phones if p.manner in ("vowel", "diphthong")), e_au.phones[0])
        pb_uv = next((p for p in e_au.phones if p.manner in ("fricative", "stop")), e_au.phones[-1])

        # Voiced/mixed ZH->EN switch
        z_env = zf.log_env[sy.syl_start : sy.syl_end]
        z_voi = zf.voiced[sy.syl_start : sy.syl_end]
        e_env = ef.log_env[pb_v.p_start : pb_v.p_end]
        e_voi = ef.voiced[pb_v.p_start : pb_v.p_end]
        if len(z_env) >= 6 and len(e_env) >= 6:
            comb_env = np.concatenate([z_env, e_env], axis=0)
            comb_voi = np.concatenate([z_voi, e_voi], axis=0)
            _, diag = bridge_cs_boundary(comb_env, comb_voi, [len(z_env)], sr=SR, half_win_frames=3)
            if diag["join_l2_before_db"] is not None:
                before_l2.append(diag["join_l2_before_db"])
                after_l2.append(diag["join_l2_after_db"])

        # Unvoiced switch check
        e_uv_env = ef.log_env[pb_uv.p_start : pb_uv.p_end]
        e_uv_voi = ef.voiced[pb_uv.p_start : pb_uv.p_end]
        if len(z_env) >= 6 and len(e_uv_env) >= 4:
            comb_env = np.concatenate([z_env, e_uv_env], axis=0)
            comb_voi = np.concatenate([z_voi, e_uv_voi], axis=0)
            _, diag_u = bridge_cs_boundary(comb_env, comb_voi, [len(z_env)], sr=SR, half_win_frames=3)
            if diag_u["join_l2_before_db"] is not None:
                before_uv_l2.append(diag_u["join_l2_before_db"])
                after_uv_l2.append(diag_u["join_l2_after_db"])

    return {
        "n_voiced_switches": len(before_l2),
        "voiced_join_l2_before_median_db": round(float(np.median(before_l2)), 3) if before_l2 else None,
        "voiced_join_l2_after_median_db": round(float(np.median(after_l2)), 3) if after_l2 else None,
        "n_obstruent_switches": len(before_uv_l2),
        "obstruent_join_l2_before_median_db": round(float(np.median(before_uv_l2)), 3) if before_uv_l2 else None,
        "obstruent_join_l2_after_median_db": round(float(np.median(after_uv_l2)), 3) if after_uv_l2 else None,
    }


def eval_applied_rhythm_on_heldout_texts(en_res: dict, model: ProsodyModelV12) -> dict:
    """Compare rule-based `plan_v10` vs `apply_prosody_v12` on the 20 held-out EN texts."""
    held_texts = [item["aligned"].text for item in en_res.get("heldout_utterances", {}).values()]
    rule_ratios, v12_ratios = [], []
    rule_gaps, v12_gaps = [], []
    for txt in held_texts:
        ir0 = plan_v10(txt)
        ir1 = apply_prosody_v12(ir0, model)
        for ir, r_list, g_list in [(ir0, rule_ratios, rule_gaps), (ir1, v12_ratios, v12_gaps)]:
            f_d, c_d, f_g, c_g = [], [], [], []
            for u in ir["units"]:
                if u.get("lang") != "en":
                    continue
                is_f = str(u.get("token", "")).lower() in EN_FUNCTION_WORDS
                d = float(u.get("duration_ms", 160.0))
                g_db = float(u.get("gain_db", 20.0 * math.log10(max(1e-4, float(u.get("gain", 0.85)) / 0.85))))
                if is_f:
                    f_d.append(d)
                    f_g.append(g_db)
                else:
                    c_d.append(d)
                    c_g.append(g_db)
            if f_d and c_d:
                r_list.append(float(np.median(f_d)) / float(np.median(c_d)))
                g_list.append(float(np.median(c_g)) - float(np.median(f_g)))
    return {
        "n_heldout_texts": len(held_texts),
        "rule_plan_func_to_content_dur_ratio": round(float(np.median(rule_ratios)), 4) if rule_ratios else None,
        "v12_plan_func_to_content_dur_ratio": round(float(np.median(v12_ratios)), 4) if v12_ratios else None,
        "rule_plan_content_minus_func_db": round(float(np.median(rule_gaps)), 3) if rule_gaps else None,
        "v12_plan_content_minus_func_db": round(float(np.median(v12_gaps)), 3) if v12_gaps else None,
    }


def main() -> None:
    en_res = align_en_corpus_gmm(exclude_ids=EN_HELDOUT_DEFAULT, align_heldout=True, n_em=4)
    zh_res = align_zh_corpus_gmm(exclude_ids=None, n_em=4)

    _, ridge_metrics = fit_prosody_v12(en_res=en_res, zh_res=zh_res, model_kind="ridge", save_path=None)
    hgbr_model, hgbr_metrics = fit_prosody_v12(en_res=en_res, zh_res=zh_res, model_kind="hgbr", save_path=DEFAULT_MODEL_PATH)

    out = {
        "ridge_heldout": ridge_metrics,
        "hgbr_heldout": hgbr_metrics,
        "plan_rhythm_ablation": eval_applied_rhythm_on_heldout_texts(en_res, hgbr_model),
        "cs_bridge_real_units": eval_cs_bridge_on_real_units(zh_res, en_res),
    }
    p = EXP_DIR / "results_v12_prosody.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {p}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
