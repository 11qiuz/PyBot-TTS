"""Table-driven Mandarin frontend; stdlib only, deliberately a closed vocabulary."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

if __package__:
    from .auto_rules import QUOTE, QUOTE_MARKS, analyze_clause
else:
    from auto_rules import QUOTE, QUOTE_MARKS, analyze_clause

DEMO = ("你真的要去银行啊？", "你不是说不要去银行吗？",
        "我想买一张票。", "好，我知道了。", "重新听一遍。",
        "你现在就走？", "你不去吗？", "银行还没开门。", "你好。")

# Spaces separate syllables; both readings and word/foot boundaries are authored.
# Neutral-tone syllables are explicitly marked 5. Punctuation ends a breath group.
WORDS = {
    "你怎么": "ni3 zen3 me5", "其实": "qi2 shi2", "只要": "zhi3 yao4", "一年": "yi1 nian2", "回家": "hui2 jia1",

    "银行": "yin2 hang2", "行长": "hang2 zhang3", "银行行长": "yin2 hang2 hang2 zhang3",
    "可以": "ke3 yi3", "不行": "bu4 xing2", "不对": "bu4 dui4", "不要": "bu4 yao4",
    "不会": "bu4 hui4", "不是": "bu4 shi4", "是不是": "shi4 bu5 shi4",
    "重新": "chong2 xin1", "重复": "chong2 fu4", "重要": "zhong4 yao4",
    "长大": "zhang3 da4", "长度": "chang2 du4", "重庆": "chong2 qing4",
    "音乐": "yin1 yue4", "快乐": "kuai4 le4", "乐器": "yue4 qi4",
    "听音乐": "ting1 yin1 yue4", "听一遍": "ting1 yi1 bian4",
    "一遍": "yi1 bian4", "一张": "yi1 zhang1", "一个": "yi1 ge4",
    "一样": "yi1 yang4", "一点": "yi1 dian3", "一起": "yi1 qi3",
    "第一次": "di4 yi1 ci4", "第一": "di4 yi1",
    "你好": "ni3 hao3", "很好": "hen3 hao3", "小狗": "xiao3 gou3",
    "老虎": "lao3 hu3", "小老虎": "xiao3 lao3 hu3",
    "我想": "wo3 xiang3", "想买": "xiang3 mai3", "我想买": "wo3 xiang3 mai3",
    "你想": "ni3 xiang3", "想要": "xiang3 yao4", "你想要": "ni3 xiang3 yao4",
    "知道": "zhi1 dao4", "真的": "zhen1 de5", "什么": "shen2 me5",
    "怎么": "zen3 me5", "这个": "zhe4 ge5", "那个": "na4 ge5",
    "怎么回事": "zen3 me5 hui2 shi4", "没问题": "mei2 wen4 ti2",
    "没关系": "mei2 guan1 xi5", "谢谢": "xie4 xie5", "明白": "ming2 bai5",
    "听见": "ting1 jian4", "听到了": "ting1 dao4 le5", "吃饭": "chi1 fan4",
    "吃过": "chi1 guo5", "现在": "xian4 zai4", "今天": "jin1 tian1",
    "马上": "ma3 shang4", "已经": "yi3 jing1", "准备": "zhun3 bei4",
    "当然": "dang1 ran2", "等等": "deng3 deng3", "觉得": "jue2 de5",
    "喜欢": "xi3 huan5", "行不行": "xing2 bu5 xing2",
    "好不好": "hao3 bu5 hao3", "看一看": "kan4 yi5 kan4",
    "试一试": "shi4 yi5 shi4", "重来": "chong2 lai2",
    "我知道": "wo3 zhi1 dao4", "我们": "wo3 men5", "你们": "ni3 men5",
    "对不起": "dui4 bu4 qi3", "不要紧": "bu4 yao4 jin3",
    "还好": "hai2 hao3", "还是": "hai2 shi4", "还钱": "huan2 qian2",
    "听我说": "ting1 wo3 shuo1",
    "太好了": "tai4 hao3 le5", "真的假的": "zhen1 de5 jia3 de5",
    "好久不见": "hao3 jiu3 bu4 jian4", "晚上好": "wan3 shang4 hao3",
    "早上好": "zao3 shang4 hao3", "再见": "zai4 jian4",
    "中国": "zhong1 guo2", "普通话": "pu3 tong1 hua4",
    "声音": "sheng1 yin1", "确认": "que4 ren4", "惊讶": "jing1 ya4",
    "小心": "xiao3 xin1", "没事": "mei2 shi4", "怎么了": "zen3 me5 le5",
    "说好了": "shuo1 hao3 le5", "明天": "ming2 tian1", "快点": "kuai4 dian3",
}
CHARS = dict(x.split(":") for x in (
    "我:wo3 你:ni3 他:ta1 她:ta1 它:ta1 好:hao3 很:hen3 想:xiang3 买:mai3 卖:mai4 票:piao4"
    " 去:qu4 来:lai2 会:hui4 能:neng2 真:zhen1 的:de5 了:le5 吗:ma5 呢:ne5 吧:ba5 啊:a5 呀:ya5"
    " 不:bu4 一:yi1 第:di4 是:shi4 说:shuo1 听:ting1 看:kan4 到:dao4 在:zai4 有:you3"
    " 要:yao4 就:jiu4 也:ye3 都:dou1 没:mei2 还:hai2 又:you4 让:rang4 做:zuo4 试:shi4"
    " 太:tai4 对:dui4 着:zhe5 这:zhe4 那:na4 个:ge4 先:xian1 请:qing3 等:deng3 别:bie2"
    " 小:xiao3 大:da4 狗:gou3 老:lao3 虎:hu3 人:ren2 口:kou3 气:qi4 给:gei3 水:shui3 茶:cha2"
    " 三:san1 两:liang3 二:er4 次:ci4 遍:bian4 张:zhang1 点:dian3 事:shi4 钱:qian2"
    " 行:xing2 长:chang2 重:zhong4 乐:le4 们:men5 新:xin1 开:kai1 关:guan1 门:men2"
    " 知:zhi1 道:dao4 些:xie1 现:xian4 明:ming2 白:bai2 准:zhun3 备:bei4 吃:chi1"
    " 饭:fan4 饿:e4 惊:jing1 讶:ya4 确:que4 认:ren4 什:shen2 么:me5 怎:zen3 过:guo4"
    " 打:da3 包:bao1 走:zou3 快:kuai4 慢:man4 送:song4 回:hui2 见:jian4 刚:gang1"
    " 日:ri4 字:zi4 四:si4 住:zhu4 店:dian4 家:jia1 和:he2 只:zhi3 自:zi4 己:ji3"
    " 银:yin2 音:yin1 喜:xi3 欢:huan1 能:neng2 起:qi3 今:jin1 天:tian1 当:dang1 然:ran2"
    " 早:zao3 上:shang4 晚:wan3 安:an1 中:zhong1 国:guo2 话:hua4 语:yu3 声:sheng1"
    " 谢:xie4 假:jia3 紧:jin3 心:xin1 去:qu4 最:zui4 后:hou4 久:jiu3 再:zai4"
    " 居:ju1 竟:jing4 而:er2 难:nan2 谁:shei2 哪:na3 为:wei4 里:li3 儿:er2"
    " 多:duo1 少:shao3 几:ji3 问:wen4 需:xu1 应:ying1 该:gai1 愿:yuan4 意:yi4"
    " 信:xin4 相:xiang1 记:ji4 得:de2 觉:jue2 拿:na2 用:yong4 停:ting2 睡:shui4"
    " 跑:pao3 玩:wan2 同:tong2 喝:he1 哦:o5 选:xuan3 杯:bei1 红:hong2 蓝:lan2"
    " 特:te4 刚:gang1 才:cai2 把:ba3 可:ke3 真:zhen1 衣:yi1 服:fu2"
).split())
ONSET = re.compile(r"^(zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw])")
Y = {"i": "i", "a": "ia", "ao": "iao", "e": "ie", "ou": "iou",
     "an": "ian", "in": "in", "ang": "iang", "ing": "ing", "ong": "iong",
     "u": "v", "ue": "ve", "uan": "van", "un": "vn"}
W = {"u": "u", "a": "ua", "o": "uo", "ai": "uai", "ei": "uei",
     "an": "uan", "en": "uen", "ang": "uang", "eng": "ueng"}
PAUSE = {"，": 155, ",": 155, "、": 115, "。": 280, ".": 280,
         "？": 255, "?": 255, "！": 255, "!": 255, "；": 210, ";": 210,
         "：": 160, ":": 160, "…": 285, "—": 190, "\n": 190}
TERMINAL = set("。.!！?？;；\n…")
MAX_CHUNK = max(map(len,WORDS))
STYLES = ("confirmation", "surprise")  # compatibility: explicit presets only
MODES = ("auto", *STYLES)
CURVE = {
    1: [1.8, 2.2, 2.1, 2.0, 1.8],
    2: [-1.3, -1.7, -.6, 1.0, 3.0],
    3: [-1.4, -3.4, -4.3, -2.6, -.6],
    4: [3.5, 2.4, .5, -1.7, -3.5],
    5: [0.0, -.2, -.35, -.55, -.75],
}


def split_pinyin(pinyin):
    """Return ASCII phonemic onset, final and tone. v = ü, ir/iz = apical vowels."""
    if not re.fullmatch(r"[a-zvü]+[1-5]", pinyin):
        raise ValueError(f"Invalid marked-suffix pinyin: {pinyin!r}")
    syllable, tone = pinyin[:-1].replace("ü", "v"), int(pinyin[-1])
    match = ONSET.match(syllable)
    o = match[0] if match else ""
    f = syllable[len(o):]
    if o == "y":
        return "", Y[f], tone
    if o == "w":
        return "", W[f], tone
    if o in ("j", "q", "x") and f.startswith("u"):
        f = "v" + f[1:]
    elif f == "i" and o in ("zh", "ch", "sh", "r"):
        f = "ir"
    elif f == "i" and o in ("z", "c", "s"):
        f = "iz"
    else:
        f = {"iu": "iou", "ui": "uei", "un": "uen"}.get(f, f)
    if not f:
        raise ValueError(f"Missing final: {pinyin!r}")
    return o, f, tone


def _segment(text):
    groups = []
    i = 0
    quote_inner = {i for m in QUOTE.finditer(text) for i in range(m.start()+1,m.end()-1)}
    while i < len(text):
        ch = text[i]
        if ch in PAUSE:
            if groups:
                prev = groups[-1]
                prev["pause"] = max(prev["pause"], PAUSE[ch])
                prev["boundary"] = True
                if i not in quote_inner:
                    prev["terminal"] |= ch in TERMINAL
                    prev["pmarks"] += ch
                    prev["punct_end"] = i+1
            i += 1
            continue
        if ch in QUOTE_MARKS:
            i += 1
            continue
        if ch.isspace():
            if groups:
                groups[-1]["boundary"] = True
            i += 1
            continue
        word = next((text[i:i+n] for n in range(min(MAX_CHUNK,len(text)-i),1,-1)
                     if text[i:i+n] in WORDS), ch)
        if word in WORDS:
            reading = WORDS[word]
        elif ch in CHARS:
            reading = CHARS[ch]
        else:
            raise ValueError(f"未收录字符 {ch!r}（第 {i+1} 字）；请先添加词表。")
        sounds = reading.split()
        if len(sounds) != len(word):
            raise ValueError(f"词表音节数不匹配：{word!r}")
        units = []
        for j,(char,py) in enumerate(zip(word,sounds)):
            onset, final, tone = split_pinyin(py)
            units.append(dict(hanzi=char,onset=onset,final=final,lexical_tone=tone,tone=tone,
                              text_span=[i+j,i+j+1]))
        groups.append(dict(word=word, units=units, boundary=False, terminal=False, pause=0,
                           pmarks="", punct_end=i+len(word)))
        i += len(word)
    return groups


def _sandhi(groups):
    for gi,g in enumerate(groups):
        us = g["units"]
        next_first = groups[gi+1]["units"][0] if gi+1 < len(groups) and not g["boundary"] else None
        for k,u in enumerate(us):
            nxt = us[k+1] if k+1 < len(us) else next_first
            if nxt is None:
                continue
            t = nxt["lexical_tone"]
            c = u["hanzi"]
            if c == "不" and u["lexical_tone"] == t == 4:
                u["tone"] = 2
            elif c == "一" and u["lexical_tone"] == 1 and not (k > 0 and us[k-1]["hanzi"] == "第"):
                if t in (1,2,3,4):
                    u["tone"] = 2 if t == 4 else 4
        # Deliberate narrow domain: longest lexicon entry/foot, not arbitrary
        # uninterrupted text. All but the last tone in a within-foot 3-run flip.
        for k in range(len(us)-1):
            if us[k]["tone"] == us[k+1]["tone"] == 3:
                us[k]["tone"] = 2


def plan(text, style="auto"):
    """Plan speech with five F0 offsets (semitones, at 0/25/50/75/100% duration).

    An independent renderer supplies the reference F0. gain is a linear scalar.
    Punctuation attaches pause_after_ms to the preceding syllable. Styles adjust
    phrasing without altering lexical readings or sandhi.
    """
    if style not in MODES:
        raise ValueError(f"style must be one of {MODES}")
    if not isinstance(text,str) or not text.strip():
        raise ValueError("text must contain supported Mandarin characters")
    groups = _segment(text)
    if not groups:
        raise ValueError("text must contain at least one syllable")
    _sandhi(groups)
    raw = []
    for g in groups:
        raw.extend((u, k == len(g["units"])-1, g) for k,u in enumerate(g["units"]))
    result=[]
    clause_start=0
    text_start=0
    clauses=[]
    ends=[i for i,(_,last,g) in enumerate(raw) if last and g["terminal"]] + [len(raw)-1]
    ends=sorted(set(ends))
    for end in ends:
        g_end=raw[end][2]
        text_end=g_end["punct_end"] if end < len(raw)-1 else len(text)
        # Main points of inference: spans stay absolute in original Unicode text,
        # including any punctuation/quotes, rather than offsets in normalized pinyin.
        chars=text[text_start:text_end]
        marks=g_end["pmarks"]
        info=analyze_clause(chars,marks) if style=="auto" else None
        focus=None
        if info:
            if info["focus_span"] is not None:
                focus=[text_start+x for x in info["focus_span"]]
            if info["break_at"] is not None:
                info["break_at"] += text_start
            info["focus_span"]=focus
            clauses.append(dict(info,text_span=[text_start,text_end],text=chars,
                                focus_text=text[focus[0]:focus[1]] if focus else None))
        for i in range(clause_start,end+1):
            u,last,g=raw[i]
            t=u["tone"]
            duration=185 if t==5 else 270
            duration += 18 if u["onset"] in {"p","t","k","q","ch","c","sh","x","s"} else 0
            duration += 20 if len(u["final"]) > 2 else 0
            duration += 20 if t==3 else 0
            fraction=(i-clause_start)/max(1,end-clause_start)
            tail = i==end
            focus_on=bool(focus and focus[0] <= u["text_span"][0] < focus[1])
            pause=g["pause"] if last else 0
            if style=="auto":
                shift=info["base_pitch"]-.32*fraction + (1.02 if focus_on else 0)
                slope=info["boundary_pitch"] if tail else (.32 if pause else 0)
                spread=1.05 if focus_on else 1.0
                gain=(.83 if t!=5 else .69)*info["gain"]*(1.11 if focus_on else 1.0)
                duration*=1.20 if focus_on else 1.0
                if tail:
                    duration*=info["final_duration"]
                    pause=info["pause_ms"] if marks else pause
                elif pause and last:
                    if not g["terminal"] and any(c in g["pmarks"] for c in "，,、：:"):
                        pause=132
                    duration*=1.07
                if u["text_span"][1]==info["break_at"] and not pause:
                    pause=88
                    duration*=1.05
            elif style=="confirmation":
                shift=.6-.9*fraction
                slope=-1.0 if tail else 0.0
                spread=1.0
                gain=(.83 if t!=5 else .69) * (.96 if tail else 1.0)
                duration *= 1.15 if tail else 1.0
            else:
                shift=1.15 + 1.15*fraction
                slope=4.0 if tail else .7
                spread=1.18
                gain=(.92 if t!=5 else .77) * (1.06 if tail else 1.0)
                duration *= 1.29 if tail else .94
            lexical = CURVE[t]
            if t == 3 and not (tail or pause):
                # The weak third tone stays low before another syllable.
                lexical = [-1.4, -2.8, -4.0, -4.15, -3.8]
            tone_curve = [round(v*spread,2) for v in lexical]
            style_curve = [round(shift+slope*(j/4)**2,2) for j in range(5)]
            u.update(duration_ms=round(duration),gain=round(gain,3),
                     pitch_st=[round(a+b,2) for a,b in zip(tone_curve,style_curve)],
                     pitch_lexical_st=tone_curve, pitch_prosody_st=style_curve,
                     pause_after_ms=pause,focus=focus_on)
            result.append(u)
        clause_start=end+1
        text_start=text_end
    return {"text":text,"style":style,"units":result,"clauses":clauses}


# Validate edited files without throwing away extra metadata or rewriting controls.
# The renderer should use duration_ms, pitch_st, gain and pause_after_ms directly.
_KNOWN_SEGMENTS = frozenset(split_pinyin(py)[:2]
    for reading in list(WORDS.values()) + list(CHARS.values())
    for py in reading.split())


def validate_plan(value):
    """Validate a mutable JSON plan; return it unchanged if valid."""
    def bounded(num, lo, hi):
        return type(num) in (int,float) and lo <= num <= hi and math.isfinite(num)

    if (not isinstance(value,dict) or not isinstance(value.get("units"),list)
            or not 1 <= len(value["units"]) <= 200):
        raise ValueError("plan needs a units list of 1–200 syllables")
    for i,u in enumerate(value["units"],1):
        if not isinstance(u,dict):
            raise ValueError(f"unit {i}: expected JSON object")
        if not isinstance(u.get("hanzi"),str) or len(u["hanzi"]) != 1:
            raise ValueError(f"unit {i}: expected one character")
        pair=u.get("onset"),u.get("final")
        if not all(isinstance(x,str) for x in pair) or pair not in _KNOWN_SEGMENTS:
            raise ValueError(f"unit {i}: unsupported onset/final")
        if type(u.get("tone")) is not int or not 1 <= u["tone"] <= 5:
            raise ValueError(f"unit {i}: invalid surface tone")
        if not bounded(u.get("duration_ms"),70,1500):
            raise ValueError(f"unit {i}: invalid duration_ms [70, 1500]")
        if not bounded(u.get("pause_after_ms"),0,2000):
            raise ValueError(f"unit {i}: invalid pause_after_ms [0, 2000]")
        if not bounded(u.get("gain"),0,2):
            raise ValueError(f"unit {i}: invalid gain [0, 2]")
        pitch=u.get("pitch_st")
        if not isinstance(pitch,list) or len(pitch)!=5 or not all(bounded(v,-18,18) for v in pitch):
            raise ValueError(f"unit {i}: pitch_st must hold five offsets in [-18, 18]")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?")
    parser.add_argument("--style", choices=MODES, default="auto")
    parser.add_argument("--validate", type=Path, help="Read an edited JSON plan instead of text.")
    parser.add_argument("-o", "--output", type=Path)
    args=parser.parse_args()
    if (args.text is None) == (args.validate is None):
        parser.error("provide text or --validate JSON, not both")
    try:
        value=validate_plan(json.loads(args.validate.read_text(encoding="utf-8"))) if args.validate else validate_plan(plan(args.text,args.style))
        out=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
        if args.output:
            args.output.write_text(out,encoding="utf-8")
        else:
            print(out,end="")
    except (OSError,ValueError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()

