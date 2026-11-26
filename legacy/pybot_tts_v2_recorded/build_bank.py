"""Turn licensed AISHELL3 source files into compact recorded units."""
import json
import math
from pathlib import Path
from array import array
import wave

ROOT=Path(__file__).resolve().parent
B=ROOT/'bank'

def main():
    B.mkdir(exist_ok=True)
    rows={}
    for f in ('metadata.json','metadata_isolated.json'):
        rows.update({r['id']:r for r in json.loads((ROOT/'source'/f).read_text(encoding='utf-8'))})
    result=[]
    for i,(id,text,lo,hi) in enumerate(json.loads((ROOT/'cuts.json').read_text(encoding='utf-8')),1):
        m=rows[id]
        assert m['text'].count(text)==1, (m,text)
        offset=m['text'].index(text)
        py=m['pinyin'].split()
        assert len(py)==len(m['text']), (m,text)
        source=ROOT/'source'/'wav'/f'{id}.wav'
        with wave.open(str(source),'rb') as f:
            assert (f.getnchannels(),f.getsampwidth(),f.getframerate())==(1,2,24000)
            samples=array('h',f.readframes(f.getnframes()))
        if lo is None:
            hop=120
            e=[math.sqrt(sum(x*x for x in samples[j:j+hop])/hop) for j in range(0,len(samples),hop)]
            hit=[j for j,v in enumerate(e) if v>max(e)*.026 and v>30]
            a, b=max(0,(min(hit)-7)*hop),min(len(samples),(max(hit)+8)*hop)
        else:
            a,b=round(lo*24000), min(len(samples),round(hi*24000))
        assert b-a > 7200 and b-a < 42000,(id,a,b)
        file=f'u{i:02}.wav'
        with wave.open(str(B/file),'wb') as f:
            f.setparams((1,2,24000,0,'NONE','not compressed'))
            f.writeframes(samples[a:b].tobytes())
        result.append(dict(text=text,id=id,speaker=id[:7],source_text=m['text'],
                    pinyin=py[offset:offset+len(text)],path=file,start_sample=a,end_sample=b))
    (B/'index.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding='utf-8')
if __name__=='__main__':
    main()
