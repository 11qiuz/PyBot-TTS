"""Local unit-selection synthesizer over a small licensed real-audio bank."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys
import wave

from engine import plan, split_pinyin, validate_plan
from synth import SR, write_wav

BANK=Path(__file__).resolve().parent/'bank'
def rms(x):
    return math.sqrt(sum(v*v for v in x)/max(1,len(x)))

def load_bank(speaker):
    items=[]
    for row in json.loads((BANK/'index.json').read_text(encoding='utf-8')):
        if row['speaker']!=speaker:
            continue
        f=(BANK/row['path'])
        with wave.open(str(f),'rb') as w:
            assert (w.getnchannels(),w.getsampwidth(),w.getframerate())==(1,2,SR)
            data=array('h',w.readframes(w.getnframes()))
        if sys.byteorder!='little':
            data.byteswap()
        x=[v/32768 for v in data]
        r=rms(x)
        gain=min(3.0,.13/max(r,1e-6))
        x=[v*gain for v in x]
        row.update(samples=x, rms=r,
                   left=rms(x[:SR//16]),right=rms(x[-SR//16:]),
                   signature=tuple(split_pinyin(y) for y in row['pinyin']),
                   sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        items.append(row)
    if not items:
        raise ValueError('unknown speaker/bank')
    return items

def choose(p, bank, strict_no_playback=True):
    u=p['units']
    han=''.join(t['hanzi'] for t in u)
    sig=[(t['onset'],t['final'],t['tone']) for t in u]
    states={0:[(0.0,[])]}
    for i in range(len(u)):
        if i not in states:
            continue
        for a in bank:
            j=i+len(a['text'])
            if not han.startswith(a['text'],i) or tuple(sig[i:j])!=a['signature']:
                continue
            if strict_no_playback and han in a['source_text']:
                continue
            if any(t['pause_after_ms'] for t in u[i:j-1]):
                continue
            dest=states.setdefault(j,[])
            dur=sum(t['duration_ms'] for t in u[i:j])/1000
            target=.30*abs(math.log(len(a['samples'])/SR/dur))
            for prior, path in states[i]:
                join=0 if not path else .16*abs(math.log((.015+path[-1][0]['right'])/(.015+a['left'])))
                dest.append((prior+1.0+target+join,path+[(a,i,j)]))
            dest.sort(key=lambda q:q[0])
            del dest[32:]
    if len(u) not in states or not states[len(u)]:
        furthest=max(states)
        raise ValueError(f'声库无法用精确声韵调覆盖：{han[furthest:]}。不把整句或合成音偷偷混进来。')
    return states[len(u)][0][1]

def tempo(x, ratio):
    """Small WSOLA duration edit. Retain the recording's pitch, limit amount."""
    if abs(1-ratio)<.035:
        return list(x)
    win,hop,search=720,360,150
    nominal=hop/ratio
    target=round(len(x)*ratio)
    if len(x)<3*win:
        return list(x)
    out=list(x[:win])
    for k in range(1,math.ceil(max(0,target-win)/hop)+2):
        expected=round(k*nominal)
        if expected+win+search>=len(x):
            break
        b=out[-hop:]
        scores=[]
        for j in range(max(0,expected-search),expected+search+1,6):
            w=x[j:j+hop:5]
            den=rms(w)*rms(b[::5])
            score=sum(a*b2 for a,b2 in zip(w,b[::5]))/(max(den,1e-8)*len(w))
            scores.append((score-.0008*abs(j-expected),j))
        j=max(scores)[1]
        out[-hop:]=[a*(1-q/hop)+b2*(q/hop) for q,(a,b2) in enumerate(zip(b,x[j:j+hop]))]
        out.extend(x[j+hop:j+win])
    # Keep donor release rather than truncating a final consonant/vowel.
    if len(out)<target:
        remaining=min(len(x),target-len(out)+hop)
        tail=x[-remaining:]
        ov=min(hop,len(tail),len(out))
        out[-ov:]=[a*(1-q/ov)+v*q/ov for q,(a,v) in enumerate(zip(out[-ov:],tail[:ov]))]
        out.extend(tail[ov:])
    return out[:target]

def overlap_append(out, part, n):
    n=min(n,len(out),len(part))
    if n==0:
        out.extend(part)
        return
    out[-n:]=[a*(1-w/n)+b*w/n for w,(a,b) in enumerate(zip(out[-n:],part[:n]))]
    out.extend(part[n:])

def render(p,speaker, adapt=True):
    path=choose(p,load_bank(speaker))
    raw=[0.0]*1800
    previous_pause=True
    trace=[]
    for a,i,j in path:
        u=p['units'][i:j]
        native=len(a['samples'])/SR
        requested=sum(v['duration_ms'] for v in u)/1000
        # Retain timing and coarticulation internal to each donor as much as possible.
        # Focus lengthens the word mildly; never force exact synthetic durations.
        focus=any(v['focus'] for v in u)
        ratio=max(.86,min(1.16, (requested/native)**.28*(1.035 if focus else 1.0))) if adapt else 1.0
        volume=1.10 if adapt and focus else 1.0
        q=tempo(a['samples'],ratio)
        edge=min(len(q)//3,120)
        q=[v*volume*min(1,k/edge,(len(q)-1-k)/edge) for k,v in enumerate(q)]
        overlap_append(raw,q,0 if previous_pause else 210)
        pause=u[-1]['pause_after_ms']
        if pause:
            raw.extend([0.0]*round(pause*SR/1000))
        previous_pause=bool(pause)
        trace.append({k:a[k] for k in ('text','id','speaker','source_text','pinyin','start_sample','end_sample','sha256')} |
                     {'target_span':[i,j],'realized_ms':round(len(q)/SR*1000,1),'focus':focus,'pause_ms':pause})
    return raw+[0.0]*2400,trace

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('text')
    p.add_argument('--speaker',default='SSB1593',choices=('SSB1593','SSB0502'))
    p.add_argument('-o',type=Path,required=True)
    p.add_argument('--trace',type=Path)
    p.add_argument('--native-timing',action='store_true')
    a=p.parse_args()
    try:
        s,t=render(validate_plan(plan(a.text)),a.speaker,not a.native_timing)
        write_wav(a.o,s)
        if a.trace:
            a.trace.write_text(json.dumps(t,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    except (ValueError,OSError) as e:
        p.error(str(e))
if __name__=='__main__':
    main()
