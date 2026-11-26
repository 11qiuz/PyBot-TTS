"""Same text and frontend, stock formant vs licensed recorded units."""
import json
import math
import re
import shutil
import struct
import subprocess
from pathlib import Path
from engine import plan
from synth import render as formant, SR, write_wav
from recorded import render, rms

ROOT=Path(__file__).resolve().parent
CASES=[('你怎么知道？','SSB1593'),('我想回家。','SSB1593'),
       ('现在回家。','SSB1593'),('其实只要一年。','SSB0502')]
def active_level(x):
    n=SR//50
    levels=[rms(x[i:i+n]) for i in range(0,len(x)-n+1,n)]
    floor=max(levels)*.12
    return rms([v for i,v in enumerate(x) if i//n < len(levels) and levels[i//n]>=floor])

def lufs(x):
    if not shutil.which("ffmpeg"):
        return None
    proc=subprocess.run(["ffmpeg","-hide_banner","-f","f32le","-ar",str(SR),"-ac","1","-i","-",
                         "-af","loudnorm=print_format=json","-f","null","-"],
                        input=struct.pack(f"<{len(x)}f",*x),capture_output=True,check=True)
    return float(re.search(rb'"input_i"\s*:\s*"([^"]+)"',proc.stderr).group(1))

def pair_match(a,b):
    la,lb=lufs(a),lufs(b)
    if la is not None and lb is not None and all(map(math.isfinite,(la,lb))):
        ga,gb=10**((-21.0-la)/20),10**((-21.0-lb)/20)
    else:
        ga,gb=.12/active_level(a),.12/active_level(b)
    peak=max(max(map(abs,a))*ga,max(map(abs,b))*gb)
    common=min(1,.86/peak)
    return [a1*ga*common for a1 in a],[b1*gb*common for b1 in b]

def main():
    d=ROOT/'samples'
    d.mkdir(exist_ok=True)
    metrics=[]
    composite=[]
    for i,(text,speaker) in enumerate(CASES,1):
        p=plan(text)
        previous=formant(p['units'])
        actual,trace=render(p,speaker)
        assert len(trace)>=2 and len(set(r['id'] for r in trace))>=2
        assert {a['speaker'] for a in trace}=={speaker}
        assert ''.join(a['text'] for a in trace)==''.join(u['hanzi'] for u in p['units'])
        assert all(''.join(u['hanzi'] for u in p['units']) not in a['source_text'] for a in trace)
        assert actual==render(p,speaker)[0]
        a,b=pair_match(previous,actual)
        levels=[lufs(a),lufs(b)]
        if all(v is not None for v in levels):
            assert abs(levels[0]-levels[1])<=.15, levels
        write_wav(d/f'{i:02}_A_旧合成.wav',a,1)
        write_wav(d/f'{i:02}_B_真人片段拼接.wav',b,1)
        if i in (1,2,4):
            if composite:
                composite.extend([0.0]*int(SR*.70))
            composite.extend(a+[0.0]*int(SR*.38)+b)
        metrics.append({'text':text,'speaker':speaker,'layout':'A=formant then B=recorded',
                        'active_rms':[active_level(a),active_level(b)],'LUFS_I':levels,
                        'seconds':[len(a)/SR,len(b)/SR],'plan':p,'trace':trace})
    write_wav(d/'三句对照_旧合成再真人拼接.wav',composite,1)
    (d/'verification.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
if __name__=='__main__':
    main()
