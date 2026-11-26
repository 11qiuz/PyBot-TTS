"""Conservative written-Mandarin prosody rules, with auditable cue spans."""

from __future__ import annotations

import re

HAN = r"[\u3400-\u9fff]"
QUOTE_MARKS = frozenset('“”‘’「」『』"')
QUOTE = re.compile('“([^”]+)”|‘([^’]+)’|「([^」]+)」|『([^』]+)』|"([^"]+)"')
STOP_WORD = r"[^，,。.!！?？;；：:\s]+"
# A subset of transparent, unambiguous modal/verb heads, not an open tagger.
ACTION = r"(?:准备|知道|需要|应该|喜欢|同意|相信|记得|可以|愿意|想要|要|会|能|想|去|来|走|买|卖|看|听|说|做|吃|喝|开|关|给|等|试|见|送|拿|用|停|睡|跑|玩)"
WH = re.compile(r"为什么|怎么(?!回事的)|什么时候|什么|哪里|哪儿|哪个|哪[一几两]|谁|多少|几[个张次遍点人天]")
SURPRISE = re.compile(r"居然|竟然|怎么会|真的假的|没想到")
POLAR = re.compile(r"(?:" + HAN + r")不(?:" + HAN + r")|(?:是不是|能不能|会不会|要不要|可不可以|有没有)")
NEGATION = re.compile(r"(?:不|没有|没|别)(?:" + ACTION + r")")
CONTRAST = (
    re.compile(r"不是[^，,。.!！?？;；：:]+(?:[，,]\s*)?而是\s*(" + STOP_WORD + r")"),
    re.compile(r"(?:^|[，,]\s*)是\s*(" + STOP_WORD + r")[，,]\s*不是\s*" + STOP_WORD),
)
LIMIT_RE = re.compile(r"[吗呢吧呀啊了呀哦]+$")
# Clause F0 adjustment in semitones; not a replacement for syllable tones.
# baseline, end slope, final-duration multiplier, gain multiplier
PROFILES = {
    "statement": (0.0, -1.25, 1.13, .99),
    "continuation": (.10, .35, 1.08, 1.0),
    "neutral_question": (.15, 1.65, 1.15, 1.0),
    "polar_question": (.20, 1.95, 1.15, 1.0),
    "wh_question": (.20, .75, 1.13, 1.0),
    "rhetorical_question": (.37, 1.38, 1.15, 1.01),
    "exclamation": (.47, -.90, 1.20, 1.06),
    "surprise": (1.10, 3.00, 1.24, 1.03),
}


def _mask_quotes(text):
    spans = []
    chars = list(text)
    for match in QUOTE.finditer(text):
        g = next(i for i,s in enumerate(match.groups(),1) if s is not None)
        spans.append(match.span(g))
        chars[match.start():match.end()] = " " * (match.end()-match.start())
    return "".join(chars),spans


def _is_polar(text):
    if text.endswith("吗"):
        return True
    if re.search(r"是不是|能不能|会不会|要不要|可不可以|有没有",text):
        return True
    return any(m[0][0] == m[0][-1] for m in POLAR.finditer(text))


def _trim_focus(text, s, e):
    while s<e and not re.fullmatch(HAN,text[s]):
        s+=1
    while e>s and not re.fullmatch(HAN,text[e-1]):
        e-=1
    tail = LIMIT_RE.search(text[s:e])
    if tail and tail.start()>0:
        e=s+tail.start()
    return [s,e] if s<e else None


def analyze_clause(text, punctuation=""):
    """Analyze one clause; offsets are half-open Unicode spans relative to text.

    `text` may contain medial commas or quotes. Quoted contents cannot activate
    mood rules; a short quotation can itself receive focus. A missing cue never
    implies surprise or a focus-bearing word inferred from semantics.
    """
    outside,quotes = _mask_quotes(text)
    plain = outside.strip().rstrip("。.!！?？;；：:，,、…").rstrip()
    surprise = SURPRISE.search(outside)
    wh = WH.search(outside)
    marks = punctuation + (outside.rstrip()[-1:] if outside.rstrip() else "")
    if surprise:
        kind="surprise"
    elif "难道" in plain and any(x in marks for x in "?？"):
        kind="rhetorical_question"
    elif _is_polar(plain):
        kind="polar_question"
    elif wh and (any(x in marks for x in "?？") or plain.startswith(("谁","什么","为什么","怎么","哪里","哪儿","哪个"))):
        # Do not infer question mood from unchecked embedded questions.
        kind="wh_question"
    elif any(x in marks for x in "?？"):
        kind="neutral_question"
    elif any(x in marks for x in "!！"):
        kind="exclamation"
    elif any(x in marks for x in "，,、：:"):
        kind="continuation"
    else:
        kind="statement"
    focus,reason,break_at=None,None,None
    for s,e in quotes:
        # Avoid highlighting an entire long utterance indiscriminately.
        while s<e and not re.fullmatch(HAN,text[s]):
            s+=1
        while e>s and not re.fullmatch(HAN,text[e-1]):
            e-=1
        if 1 <= e-s <= 8 and re.fullmatch(HAN+"+",text[s:e]):
            focus,reason=[s,e],"quote"
            break
    if focus is None:
        for pat in CONTRAST:
            m=pat.search(outside)
            if not m:
                continue
            focus=_trim_focus(text,*m.span(1))
            reason="contrast" if focus else None
            j=outside.find("而是",m.start(),m.end())
            if j>0 and not any(c in "，," for c in outside[j-1:j]):
                break_at=j
            break
    if focus is None:
        candidates=list(NEGATION.finditer(outside))
        # A不A already expresses a choice: it is not a negated proposition.
        candidates=[m for m in candidates if not (
            m.start()>0 and outside[m.start()-1:m.start()] == outside[m.start()+1:m.start()+2])]
        if len(candidates)==1:
            focus,reason=list(candidates[0].span()),"negated_action"
    if focus is None and kind=="surprise":
        focus,reason=list(surprise.span()),"surprise_cue"
    if focus is None and kind=="wh_question":
        focus,reason=list(wh.span()),"question_word"
    base,tail,duration,gain=PROFILES[kind]
    pause_ms = (250 if any(x in marks for x in "?？!！") else
                235 if any(x in marks for x in "。.…") else
                195 if any(x in marks for x in ";；\n") else
                132 if kind=="continuation" else 0)
    return dict(clause_type=kind,focus_span=focus,focus_reason=reason,
                boundary_pitch=tail, pause_ms=pause_ms, break_at=break_at,
                base_pitch=base, final_duration=duration, gain=gain)
