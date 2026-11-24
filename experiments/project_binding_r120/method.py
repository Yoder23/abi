"""Separate an explicit project field from the text scored by frozen R119."""
import re
from hashlib import sha256
from experiments.voice_converse_r119 import method as old

ARMS=('status_only','r119_alias','r119','strict_raw','abi')
PID=r'P[0-9A-F]{8}'
PREFIX=re.compile(r'\A(?:(?i:project(?:[ \t]+id)?)[ \t]*:?[ \t]+)?('+PID+r')(?::|\.)[ \t]+',re.ASCII)
FOR=re.compile(r'\A(?i:for[ \t]+(?:project[ \t]+)?)('+PID+r'),[ \t]+',re.ASCII)
EXTRA=re.compile(r'\bP[0-9a-f]{6,}\b|\bproject[ \t]+id\s*:',re.I|re.ASCII)


def parse(target,text):
    if not re.fullmatch(PID,target,re.ASCII):
        raise ValueError('noncanonical target')
    match=PREFIX.match(text) or FOR.match(text)
    if not match:
        return None,'missing_or_malformed'
    if match[1]!=target:
        return None,'wrong_id'
    body=text[match.end():]
    if not body.strip() or EXTRA.search(body):
        return None,'empty_or_conflicting'
    return body,'bound'


def pairs(target,notes,roster,text):
    raw=old.pairs(target,notes,roster,text)
    return raw+[(parse(target,a)[0] or '',parse(target,b)[0] or '') for a,b in raw[3:]]


def decide(target,notes,roster,text,logits):
    if len(logits)!=19:
        raise ValueError('nineteen ordered triplets required')
    raw=old.decide(target,notes,roster,text,logits[:11])
    stripped=old.decide(target,notes,roster,text,logits[:3]+logits[11:])
    fact=dict(notes).get(target,'')
    bound=[parse(target,s) for s in (fact,text)]
    arms={k:raw['arms'][v] for k,v in zip(ARMS[:3],('status_only','alias_only','abi'))}
    for k,scores in [('strict_raw',raw),('abi',stripped)]:
        reason=next((why for _,why in bound if why!='bound'),scores['arms']['abi']['reason'])
        arms[k]=dict(reason=reason,retained_sha256=sha256(text.encode()).hexdigest() if reason=='accept' else None)
    return dict(slots=raw['slots'],bindings=[dict(reason=w,body_sha256=sha256(b.encode()).hexdigest() if b is not None else None)
                for b,w in bound],raw_gaps=raw['gaps'],gaps=stripped['gaps'],aligned=stripped['aligned'],arms=arms)
