"""Phase 9 (v13) Mandarin Third-Tone Sandhi Fix (cross-word-group + prosodic-group aware)."""
from __future__ import annotations

import copy
import re
from typing import Any

from pybot_tts_v2_recorded import engine
from exp_v5_arch import crossling_frontend as cf
from exp_v10_g2p_and_align.g2p_audit_and_fix import (
    ZH_POLYPHONE_PHRASES_V10,
    _zh_phrase_v10,
    plan_v10,
    zh_g2p_hook_v10,
)

_HANZI_SPAN_RE = re.compile(r"[\u3400-\u9fff]+")


def _compute_zh_word_group_ids(text: str, units: list[dict[str, Any]]) -> list[int]:
    """Reconstruct word-group IDs per unit using the exact same segmenter as plan_v10."""
    # Map character offsets in `text` to word-group IDs
    char_to_wg: dict[int, int] = {}
    wg_id = 0
    for m in _HANZI_SPAN_RE.finditer(text):
        run_txt = m.group(0)
        base_off = m.start()
        groups = cf._segment_zh_run(
            run_txt,
            base_off,
            zh_hook=zh_g2p_hook_v10,
            zh_phrase_hook=_zh_phrase_v10,
            zh_priority_phrases=ZH_POLYPHONE_PHRASES_V10,
        )
        for g in groups:
            for gu in g["units"]:
                s0, s1 = gu["text_span"]
                for pos in range(s0, s1):
                    char_to_wg[pos] = wg_id
            wg_id += 1

    out_wg: list[int] = []
    for idx, u in enumerate(units):
        if u.get("lang") != "zh":
            out_wg.append(-1 - idx)
            continue
        ts = u.get("text_span")
        if isinstance(ts, list) and ts and ts[0] in char_to_wg:
            out_wg.append(char_to_wg[int(ts[0])])
        else:
            out_wg.append(1_000_000 + idx)
    return out_wg


def _update_zh_unit_tone(u: dict[str, Any], new_tone: int, *, is_tail_or_pause: bool) -> None:
    """Update tone and all synchronized phonological/prosodic fields on a Mandarin unit."""
    old_tone = int(u.get("tone", 1))
    if old_tone == new_tone:
        return
    u["tone"] = int(new_tone)
    py = str(u.get("pinyin", ""))
    if py and py[-1].isdigit():
        u["pinyin"] = f"{py[:-1]}{new_tone}"
    u["stress_or_tone"] = f"T{new_tone}"
    ipas = u.get("ipa_phones") or []
    if ipas:
        u["ipa"] = " ".join(ipas) + f" T{new_tone}"

    # Recompute pitch_lexical_st and pitch_st while preserving style/prosody curve
    spread = 1.05 if u.get("focus") else 1.0
    if "pitch_lexical_st" in u and "pitch_prosody_st" in u:
        lexical = engine.CURVE[new_tone]
        if new_tone == 3 and not is_tail_or_pause:
            lexical = [-1.4, -2.8, -4.0, -4.15, -3.8]
        tone_curve = [round(v * spread, 2) for v in lexical]
        style_curve = [float(x) for x in u["pitch_prosody_st"]]
        u["pitch_lexical_st"] = tone_curve
        u["pitch_st"] = [round(max(-18.0, min(18.0, a + b)), 2) for a, b in zip(tone_curve, style_curve)]


def apply_t3_sandhi_v13(plan_doc: dict[str, Any], *, pause_break_ms: float = 15.0) -> dict[str, Any]:
    """Apply two-pass word-aware + cross-word T3 sandhi on contiguous Mandarin breath groups.

    Pass 1 (word-internal): handles `3+3 -> 2+3` inside multi-character words first so `(1+2)`
    compounds yield `3-2-3` instead of `2-2-3`, and `(2+2)` yields `2-3-2-3`.
    Pass 2 (cross-word within breath group): handles `(1+1)` (`ye3 you3 -> ye2 you3`) and
    `(2+1)` (`shou3 fu3 you3 -> shou2 fu2 you3`) across adjacent word boundaries with no pause.
    """
    out = copy.deepcopy(plan_doc)
    units = out.get("units") or []
    if not units:
        return out

    text = str(out.get("text", ""))
    wg_ids = _compute_zh_word_group_ids(text, units) if text else [i for i in range(len(units))]

    # Split into contiguous Mandarin breath-group runs (stop at non-ZH or pause > pause_break_ms)
    runs: list[list[int]] = []
    cur: list[int] = []
    for i, u in enumerate(units):
        if u.get("lang") != "zh":
            if cur:
                runs.append(cur)
                cur = []
            continue
        cur.append(i)
        if float(u.get("pause_after_ms", 0.0)) > pause_break_ms:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)

    # Identify tail positions per clause/pause for T3 contour shaping
    is_tail_pos = [
        (i == len(units) - 1) or (float(u.get("pause_after_ms", 0.0)) > 0.0) or (units[i + 1].get("lang") != "zh")
        for i, u in enumerate(units)
    ]

    for run in runs:
        if len(run) < 2:
            continue
        # Reset only lexical-T3 syllables that aren't special sandhi ("一"/"不") back to lexical 3
        # before running our two-pass word-aware T3 sandhi so (1+2) vs (2+1) is computed cleanly.
        tones: dict[int, int] = {}
        for idx in run:
            u = units[idx]
            lex = int(u.get("lexical_tone", u.get("tone", 1)))
            cur_t = int(u.get("tone", lex))
            if lex == 3 and cur_t in (2, 3):
                tones[idx] = 3
            else:
                tones[idx] = cur_t

        # Pass 1: Word-internal T3+T3 sandhi (handles 2-char and 3-char words first)
        for pos in range(len(run) - 1):
            i0, i1 = run[pos], run[pos + 1]
            if wg_ids[i0] == wg_ids[i1] and tones[i0] == 3 and tones[i1] == 3:
                tones[i0] = 2

        # Pass 2: Cross-word-group T3+T3 within contiguous breath group
        for pos in range(len(run) - 1):
            i0, i1 = run[pos], run[pos + 1]
            if tones[i0] == 3 and tones[i1] == 3:
                tones[i0] = 2

        for idx in run:
            _update_zh_unit_tone(units[idx], tones[idx], is_tail_or_pause=is_tail_pos[idx])

    return out


def plan_v13(text: str, style: str = "auto") -> dict[str, Any]:
    """Plan ZH/EN/Code-switched text with cross-word-group Mandarin T3 sandhi fixed."""
    base = plan_v10(text, style=style)
    return apply_t3_sandhi_v13(base)
