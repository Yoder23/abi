"""Prospective source-only design; annotation never sees the arm or workspace."""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'results/source_ceiling_r121'
REG = '057211db501b30f210f80d5e4eb5a5c21c0dbb54'
OLD = 'd81387e9a726e4ee93b7192cf89dd1b4d562aa1895554875c81cf4868136d427'
WP2 = '5be167cf386d0cb2e14fa2722870eeb0e3e8d325039bb62d0c514fd1a84ee9f3'
STATES = ('authorized','planned','occurred')
BANK = {
    'camera': ('rent the camera rig out to','rented the camera rig out to'),
    'kiosk': ('lease the market kiosk to','leased the market kiosk to'),
    'workshop': ('sublet the workshop to','sublet the workshop to'),
    'hoist': ('hire out the cargo hoist to','hired out the cargo hoist to'),
    'sale': ('sell the drafting table to','sold the drafting table to'),
    'loan': ('lend the field recorder to','lent the field recorder to'),
    'donation': ('donate the music stand to','donated the music stand to'),
    'handover': ('hand the parcel to','handed the parcel to'),
}
NAMES = 'Adele Bevan Cora Denis Elma Finn Gaia Hans Ida Joel Kaya Lars Mona Nico Olga Paul Rhea Seth Tina Uri Vito Wynn Xena Yves'.split()
DIRECT = ('Give one natural English restatement of the supplied note for {target}. '
          'Include the project ID and both names. Preserve the direction and whether the '
          'arrangement is authorized, planned or reported to have happened. Do not add other facts.')
EXTRA = ('\nBefore writing your restatement, identify the requested project, the two people in '
         'directional order, the event and its status. Output one short line in the form '
         'FIELDS: project | from | to | event | status. On the next line write FINAL: followed '
         'by your English restatement. Both lines must fit the response budget.')
ARMS = ('direct','fields')


def need(ok, why):
    if not ok:
        raise ValueError(why)


def dump(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',',':'), allow_nan=False)+'\n'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(p):
    return json.loads(p.read_bytes())


def put(p,x):
    p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write(dump(x))


def panel():
    rng=random.Random(int(REG,16))
    cases=[]
    seen=set()
    for family,(inf,past) in BANK.items():
        names=rng.sample(NAMES,12)
        for i in range(6):
            a,b=sorted(names[2*i:2*i+2])[::1 if i%2==0 else -1]
            t,f=[f'P{rng.getrandbits(32):08X}' for _ in range(2)]
            need(t!=f and not seen.intersection((t,f)),'duplicate ID')
            seen.update((t,f))
            def note(pid,left,right,state):
                body=(f'{left} {past} {right}.' if state=='occurred' else
                      f'The {state} arrangement is for {left} to {inf} {right}.')
                return pid+': '+body
            for j,state in enumerate(STATES):
                for count in (1,2):
                    notes=[note(t,a,b,state)]
                    if count==2:
                        notes.append(note(f,b,a,STATES[(j+1+i%2)%3]))
                        rng.shuffle(notes)
                    suffix='\nNotes:\n'+'\n'.join(notes)
                    base=DIRECT.format(target=t)
                    cases.append(dict(id=f'{family}-{i}-{count}-{state}',family=family,cluster=i,count=count,
                        state=state,target=t,foil=f,roles=[a,b],notes=notes,
                        prompts=dict(direct=base+suffix,fields=base+EXTRA+suffix)))
    rng.shuffle(cases)
    return cases


def schedule(cases):
    for i in range(0,len(cases),8):
        for arm in ARMS[::1 if i//8%2==0 else -1]:
            yield [(arm,c) for c in cases[i:i+8]]


def parse(arm,text,stop):
    if stop!='eos':
        return None,'length'
    if arm=='direct':
        return (text,'ok') if text.strip() else (None,'empty')
    need(arm=='fields','unknown arm')
    marks=list(re.finditer(r'(?m)^[ \t]*FINAL:[ \t]*',text))
    if len(marks)!=1:
        return None,'marker_count'
    marker=marks[0]
    if not text[:marker.start()].strip():
        return None,'no_workspace'
    final=text[marker.end():]
    return (final,'ok') if final.strip() else (None,'empty')


def bucket(c,arm,raw):
    final,reason=parse(arm,raw['text'],raw['stop'])
    if reason!='ok':
        return None,'Q',reason
    target=next(n for n in c['notes'] if n.startswith(c['target']+': '))
    if final in (target,'Project '+target,'Project ID '+target):
        return None,'V','source_copy'
    mapping=dict(zip([c['target'],c['foil'],*c['roles']],['<T>','<F>','<A>','<B>']))
    # Never merge a literal placeholder in the original output with a binding.
    rx=re.compile(r'\b(?:'+'|'.join(mapping)+r')\b')
    def norm(s):
        return rx.sub(lambda m:mapping[m[0]],s.replace('<','<<').replace('>','>>'))
    # Canonical target first; the full same-context foil is always included.
    notes=[target,*[n for n in c['notes'] if n!=target]]
    group=dict(notes=list(map(norm,notes)),final=norm(final))
    return group,None,'manual'


def classify(cases,rows,judgments):
    expected=[(a,c) for group in schedule(cases) for a,c in group]
    need(len(rows)==len(expected)==576 and len({c['id'] for c in cases})==288,'population')
    used=set()
    out=[]
    for (a,c),x in zip(expected,rows):
        need((x['id'],x['arm'])==(c['id'],a),'row mismatch')
        group,label,reason=bucket(c,a,x)
        if group is not None:
            key=sha(dump(group).encode())
            need(key in judgments and judgments[key] in ('V','U','O','Q'),'missing or bad blind label')
            used.add(key)
            label=judgments[key]
        out.append((c,a,label,reason))
    need(used==set(judgments),'extraneous judgment')
    return out


def metrics(scored):
    import numpy as np
    reports={}
    lookup={}
    for arm in ARMS:
        data=[(c,l,m) for c,a,l,m in scored if a==arm]
        need(len(data)==288,'arm population')
        lookup[arm]={c['id']:int(l=='V') for c,l,m in data}
        def count(rows):
            return {k:sum(l==k for _,l,_ in rows) for k in 'VUOQ'}
        cells={}
        for fam in BANK:
            for n in (1,2):
                for s in STATES:
                    u=[r for r in data if (r[0]['family'],r[0]['count'],r[0]['state'])==(fam,n,s)]
                    need(len(u)==6,'cell')
                    cells[f'{fam}/{n}/{s}']=count(u)
        windows={}
        for s in STATES[:2]:
            u=[r for r in data if r[0]['state'] in (s,'occurred')]
            need(len(u)==192,'window')
            windows[s]=dict(valid=count(u)['V'],nonempty=all(any(c['family']==f and l=='V' for c,l,_ in u) for f in BANK))
        totals=count(data)
        gates=dict(usable=totals['V']>=216,each_cell=all(x['V']>=3 for x in cells.values()),
                   windows=all(x['valid']>=144 and x['nonempty'] for x in windows.values()))
        reports[arm]=dict(totals=totals,cells=cells,windows=windows,gates=gates,ceiling_pass=all(gates.values()),
            by_family={f:count([r for r in data if r[0]['family']==f]) for f in BANK},
            by_status={s:count([r for r in data if r[0]['state']==s]) for s in STATES},
            reasons=dict(Counter(m for c,l,m in data)))
    paired={}
    cases=[c for c,a,l,m in scored if a=='direct']
    draw=np.random.default_rng(121001).integers(6,size=(10000,len(BANK),6))+np.arange(len(BANK))[None,:,None]*6
    for k,sub in [('all',cases),('occurred',[c for c in cases if c['state']=='occurred'])]:
        def diff(c):
            return lookup['fields'][c['id']]-lookup['direct'][c['id']]
        grid=np.array([np.mean([diff(c) for c in sub if c['family']==f and c['cluster']==i]) for f in sorted(BANK) for i in range(6)])
        paired[k]=dict(delta=float(grid.mean()),ci95=np.quantile(grid[draw].mean(axis=(1,2)),[.025,.975]).tolist(),
                       gained=sum(diff(c)>0 for c in sub),lost=sum(diff(c)<0 for c in sub))
    go=any(x['ceiling_pass'] for x in reports.values())
    return dict(arms=reports,paired=paired,supported_fields_gain=reports['fields']['ceiling_pass'] and paired['all']['ci95'][0]>0,
                verdict='SOURCE_CEILING_FEASIBLE_ONLY' if go else 'STOP_SOURCE_RECIPE',wp2='CLOSED')
