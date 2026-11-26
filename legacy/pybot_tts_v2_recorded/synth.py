"""Deterministic toy formant renderer. No recorded voice or trained weights."""
import argparse
from array import array
import json
import math
from pathlib import Path
import random
import sys
import wave

SR = 24000
F0 = 152.0
TAU = math.tau
# Broad spectral targets in Hz, not fitted to any speaker.
V = {
    'a': (790, 1300, 2650), 'aa': (730, 1180, 2550),
    'e': (470, 1220, 2700), 'eh': (570, 1880, 2740), 'o': (490, 880, 2630),
    'i': (285, 2200, 3050), 'u': (320, 780, 2200), 'v': (310, 1770, 2490),
    'schwa': (475, 1490, 2620), 'apical': (345, 1540, 2500),
    'retro': (390, 1390, 1900), 'nasal': (275, 1200, 2400),
}
# Piecewise vowel targets plus optional nasal tail.
FINALS = {
    'a':'a','o':'o','e':'e','i':'i','u':'u','v':'v','ü':'v',
    'ai':'a i','ei':'eh i','ao':'aa u','ou':'o u',
    'an':'a a nasal','en':'schwa schwa nasal','ang':'aa aa nasal',
    'eng':'e e nasal','ong':'o u nasal','er':'retro','ir':'retro',
    'iz':'apical','ii':'apical','ih':'apical','i_zh':'retro','i_z':'apical',
    'ia':'i a','ie':'i eh','iao':'i aa u','iu':'i o u','iou':'i o u',
    'ian':'i eh nasal','in':'i i nasal','iang':'i aa nasal','ing':'i i nasal',
    'iong':'v o nasal','ua':'u aa','uo':'u o','uai':'u aa i','ui':'u eh i',
    'uei':'u eh i','uan':'u aa nasal','un':'u schwa nasal','uen':'u schwa nasal',
    'uang':'u aa nasal','ueng':'u e nasal','ve':'v eh','üe':'v eh',
    'van':'v eh nasal','üan':'v eh nasal','vn':'v v nasal','ün':'v v nasal',
}
NASALS = {'n', 'm'}
FRIC = {'f':(.33,.18),'s':(.58,.56),'sh':(.38,.32),'x':(.52,.37),
        'h':(.34,.08),'r':(.23,.15)}
# onset timing and aspiration strength.
STOP = {'b':(.047,.15),'p':(.076,.44), 'd':(.040,.13),'t':(.070,.38),
        'g':(.043,.13),'k':(.075,.44),
        'z':(.051,.31),'c':(.080,.56),'zh':(.061,.31),'ch':(.091,.53),
        'j':(.052,.31),'q':(.083,.56)}
NASAL_SOUND = {'m', 'n', 'l'}

def interp(values, t):
    p = max(0.0, min(1.0, t))*(len(values)-1)
    i = min(int(p),len(values)-1)
    j = min(i+1,len(values)-1)
    return values[i]+(values[j]-values[i])*(p-i)

def spectral_weight(h, f0, formants, nasal):
    hz = h*f0
    peaks = sum(
        amp/(1.0+((hz-f)/bw)**2)
        for f,bw,amp in zip(formants, (100,140,195), (1.0,.73,.43))
    )
    return (.055+peaks)/h**.68 * ((1.0-.56*nasal) if hz>600 else 1.0)

def vowel(keys, duration, contour, rng, gain):
    n = max(1, int(duration*SR))
    target = tuple(V[k] for k in keys)
    phase = 0.0
    weights = ()
    samples = []
    for i in range(n):
        t = i/max(1,n-1)
        f0 = F0*2**((interp(contour,t)+.12*math.sin(t*TAU*1.7))/12)
        if i%72 == 0:
            slide = max(0.0, min(1.0, (t-.13)/.77))
            fs = [interp([f[k] for f in target],slide) for k in range(3)]
            nasality = max(0.0,(t-.62)/.38) if keys[-1]=='nasal' else 0.0
            weights = [spectral_weight(h,f0,fs,nasality) for h in range(1,27)]
        phase = (phase+TAU*f0/SR)%TAU
        pressure = sum(w*math.sin(h*phase + .39*h) for h,w in enumerate(weights,1))
        env = min(1.0, i/(SR*.017), (n-1-i)/(SR*.032))
        flutter = .95+.045*math.sin(t*TAU)
        samples.append(gain*env*flutter*(.61*pressure+.014*rng.uniform(-1,1)))
    return samples

def consonant(onset, duration, contour, rng, gain):
    n=int(duration*SR)
    if n<=0:
        return []
    if onset in NASAL_SOUND:
        # Short voiced pre-onset transitions into the vowel.
        key = 'nasal' if onset in NASALS else 'schwa'
        return vowel((key,),duration,[contour[0]],rng,gain*.47)
    out=[]
    last=0.0
    lp=0.0
    for i in range(n):
        x=rng.uniform(-1,1)
        lp=.70*lp+.30*x
        t=i/max(n-1,1)
        hp=x-last
        last=x
        if onset in FRIC:
            amp, bright=FRIC[onset]
            env=min(1,t*5,(1-t)*4)*amp
        else:
            _, amp = STOP.get(onset, (.04,.15))
            bright=.47 if onset in ('z','c','zh','ch','j','q','t','d') else .21
            env=0.0 if t < .27 else amp*max(0,1-t)**.40
            if .27 <= t <= .36:
                env=min(.75,env+.20)
        out.append(gain*env*(bright*hp+(1-bright)*lp)*.63)
    return out

def render(units, seed=72):
    rng=random.Random(seed)
    out=[0.0]*int(.075*SR)
    for u in units:
        onset=u['onset']
        final=u['final']
        pitch=[float(x) for x in u['pitch_st']]
        if len(pitch)<1 or any(not math.isfinite(x) or abs(x)>18 for x in pitch):
            raise ValueError('pitch_st must be finite semitone values within ±18')
        duration=float(u['duration_ms'])/1000
        gain=float(u['gain'])
        pause=float(u.get('pause_after_ms',0))/1000
        if not .07<=duration<=1.5 or not 0<=gain<=2 or not 0<=pause<=2:
            raise ValueError('invalid duration, gain or pause')
        if onset=='y' or onset=='w': # only for manually authored plans
            onset=''
        if final=='i' and onset in ('zh','ch','sh','r','z','c','s'):
            final='ir' if onset in ('zh','ch','sh','r') else 'iz'
        if final not in FINALS or onset not in (FRIC.keys()|STOP.keys()|NASAL_SOUND|{''}):
            raise ValueError(f'no sound recipe for onset={onset!r}, final={final!r}')
        c_dur = 0.0 if not onset else STOP.get(onset,(.058 if onset in FRIC else .040,0))[0]
        c_dur = min(c_dur,duration*.39)
        c=consonant(onset,c_dur,pitch,rng,gain)
        v=vowel(tuple(FINALS[final].split()),duration-c_dur,pitch,rng,gain)
        # Small overlap at the onset/rime seam; not natural coarticulation.
        overlap=min(len(c),len(v),int(.005*SR))
        if overlap:
            c[-overlap:]=[c[-overlap+i]+v[i] for i in range(overlap)]
        out+=c+v[overlap:]
        out.extend([0.0]*round(pause*SR))
    out.extend([0.0]*int(.100*SR))
    return out

def write_wav(path, samples, factor=None):
    peak=max(map(abs,samples),default=0.0)
    factor=min(.90/max(peak,1e-8),1.0) if factor is None else factor
    pcm=array('h',(round(max(-.999,min(.999,x*factor))*32767) for x in samples))
    if sys.byteorder!='little':
        pcm.byteswap()
    with wave.open(str(path),'wb') as f:
        f.setparams((1,2,SR,0,'NONE','not compressed'))
        f.writeframes(pcm.tobytes())

def main():
    if __package__:
        from .engine import MODES, plan, validate_plan
    else:
        from engine import MODES, plan, validate_plan
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('text',nargs='?')
    p.add_argument('--style',default='auto',choices=MODES)
    p.add_argument('--plan',type=Path)
    p.add_argument('--dump-plan',type=Path)
    p.add_argument('-o','--output',type=Path,required=True)
    a=p.parse_args()
    if (a.text is None) == (a.plan is None):
        p.error('give exactly one of text or --plan')
    try:
        d = validate_plan(json.loads(a.plan.read_text(encoding='utf-8')) if a.plan else plan(a.text,a.style))
        if a.dump_plan:
            a.dump_plan.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        write_wav(a.output,render(d['units']))
    except (OSError, ValueError) as exc:
        p.error(str(exc))

if __name__=='__main__':
    main()
