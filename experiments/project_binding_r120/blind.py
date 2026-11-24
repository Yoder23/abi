"""Prediction-blind annotation aid; no classifier access. No default labels."""
from collections import defaultdict
import re
from . import run as r


def groups(p,live):
    bucket=defaultdict(list)
    for c,x in zip(p['contexts'],live):
        bound=dict(c['notes'])[c['target']]
        ordered=sorted(c['roster'],key=bound.index)
        mapping=dict(zip(ordered,['A','B']))
        mapping[c['target']]='T'
        for pid,_ in c['notes']:
            mapping.setdefault(pid,'F')
        def sub(s):
            return re.sub(r'\b(?:'+'|'.join(map(re.escape,mapping))+r')\b',lambda m:mapping[m[0]],s)
        # Only group equivalent target facts; mentions of foils need the full
        # context. Sorting does not alter any candidate bytes or labeling units.
        context=tuple(sorted(sub(v) for k,v in c['notes'] if k!=c['target'] and k in x['text']))
        key=(c['family'],c['state'],sub(bound),context,sub(x['text']),x['stop'])
        bucket[key].append(x['id'])
    return sorted(bucket.items())


def show(f):
    p=r.read(r.OUT/'packet.json')
    live=r.lines(r.OUT/'teacher-initial/raw.jsonl')
    group=groups(p,live)
    for n,(key,ids) in enumerate(group):
        if key[0]==f:
            print(f'{n:03} {key[1][:3]} x{len(ids)} {key[4]!a}', flush=True)


if __name__=='__main__':
    import sys
    show(sys.argv[1])
