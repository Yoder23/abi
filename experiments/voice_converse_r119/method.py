"""Prospective text-only contrast. No event lexicon, role parser or panel labels."""
from hashlib import sha256
import math
import re
from experiments.modality_roles_r118 import method as frozen

ARMS=('one_way','r117','status_only','r118_full','alias_only','abi')
ALIASES=('Alice','Bob')


def views(target,notes,roster,text):
    baseline=frozen.prior.pairs(target,notes,text)
    fact=baseline[1][0]
    names=tuple(roster)
    if len(names)!=2 or len({n.casefold() for n in names})!=2 or not all(re.fullmatch('[A-Za-z]+',n) for n in names):
        raise ValueError('invalid roster')
    pattern=re.compile(r'\b(?:'+'|'.join(map(re.escape,names))+r')\b',re.I)
    def present(s):
        return {m[0].lower() for m in pattern.finditer(s)}=={n.lower() for n in names}
    safe=bool(fact and present(fact) and present(text) and not re.search(r'\b(?:Alice|Bob)\b',fact+' '+text,re.I))
    pairs=list(baseline)
    def rename(s,labels):
        mapping={n.lower():v for n,v in zip(names,labels)}
        return pattern.sub(lambda m:mapping[m[0].lower()],s)
    for labels in (ALIASES,ALIASES[::-1]):
        a,b,foil=rename(fact,labels),rename(text,labels),rename(text,labels[::-1])
        pairs.extend([(a,b),(b,a),(a,foil),(foil,a)])
    return pairs,safe


def pairs(target,notes,roster,text):
    return views(target,notes,roster,text)[0]


def decide(target,notes,roster,text,logits):
    if len(logits)!=11 or any(len(v)!=3 or not all(type(k) in (int,float) and math.isfinite(k) for k in v) for v in logits):
        raise ValueError('expected eleven finite logit triplets')
    _,safe=views(target,notes,roster,text)
    base=frozen.decide(target,notes,roster,text,logits[:3])
    arms={k:base['arms'][old] for k,old in zip(ARMS[:4],('one_way','r117','state_only','abi'))}
    margins=[v[1]-max(v[0],v[2]) for v in logits[3:]]
    aligned=[margins[i] for i in (0,1,4,5)]
    gaps=[margins[i]-margins[i+2] for i in (0,1,4,5)]
    src,cand=(s['state'] for s in base['slots'])
    why=('lexical' if base['arms']['lexical']['reason']!='accept' else 'missing_or_colliding_names' if not safe
         else 'state_unknown' if src is None or cand is None else 'state_mismatch' if src!=cand
         else 'aligned_not_entailed' if min(aligned)<=0 else 'accept')
    for arm,reason in [('alias_only',why),('abi',why if why!='accept' or min(gaps)>0 else 'swap_not_separated')]:
        arms[arm]={'reason':reason,'retained_sha256':sha256(text.encode()).hexdigest() if reason=='accept' else None}
    return dict(slots=base['slots'],safe=safe,aligned=aligned,gaps=gaps,arms=arms)
