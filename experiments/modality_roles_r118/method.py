"""Experimental finite English guards. No panel, expected state, or role labels."""
import re
from hashlib import sha256
from experiments.grounded_normalization_r117 import method as prior

ARMS = ('raw','lexical','one_way','r117','state_only','roles_only','abi')
UNSAFE = r"\b(?:not|never|unless|if|possibly|might|unclear|whether|alleged|denied|denies|refused|cancelled|canceled)\b|n't\b"
AUTH = r"\b(?:authori[sz](?:ed|ation)|permit(?:s|ted)?|permission|allows?|allowed|clearance|cleared|may|gave the go-ahead|green light|entitled)\b"
PLAN = r"\b(?:plans?|planned|planning|intends?|intended|intention|scheduled?|scheduling|slated|due to|aims?|expects? to|will|going to|on the calendar)\b"
PAST = r"\b(?:[a-z]{3,}ed|gave|sent|took|told|brought|wrote|read|met|lent|sold|paid|led|ran|held|made|spoke|shown|gave|got)\b"
RECEIVE = r"\b(?:receive[ds]?|got|obtained|accepted|was given|was sent|was told)\b"


def state(text):
    s=text.lower().replace('’',"'")
    if re.search(UNSAFE,s):
        return None
    a,p=bool(re.search(AUTH,s)),bool(re.search(PLAN,s))
    if a or p:
        return None if a and p else 'authorized' if a else 'planned'
    return 'occurred' if re.search(PAST,s) else None


def roles(text, roster):
    """Directed action: doer/source, other participant. Unknown fails closed."""
    if len(roster)!=2 or len(set(roster))!=2 or not all(re.fullmatch('[A-Za-z]+',n) for n in roster):
        raise ValueError('two different public names required')
    pat=r'\b(?:'+'|'.join(map(re.escape,roster))+r')\b'
    hits=list(re.finditer(pat,text,re.I))
    norm={n.lower():n for n in roster}
    if len(hits)!=2 or len(set(m[0].lower() for m in hits))!=2:
        return None
    if re.search(UNSAFE,text.lower()):
        return None
    actor,other=set(),set()
    for h in hits:
        n=norm[h[0].lower()]
        before=text[:h.start()].lower()
        if re.search(r'\bby\s*$',before):
            if re.search(r'\b(?:authorized|permitted|approved|granted)\s+by\s*$',before):
                return None
            actor.add(n)
        elif re.search(r'\bfrom\s*$',before):
            if not re.search(RECEIVE,text.lower()):
                return None
            actor.add(n)
        elif re.search(r'\b(?:to|toward|towards)\s*$',before):
            other.add(n)
    a,b=[norm[h[0].lower()] for h in hits]
    if not actor and not other:
        # Direct-object construction. Bare word order is not enough: require
        # evidence of a finite verb following the first participant.
        mid=text[hits[0].end():hits[1].start()].lower().strip()
        if re.match(r"(?:has|had|is|was|may|can|will|plans?|intends?|aims?|[a-z]+ed|gave|sent|told|lent|paid|brought)\b",mid):
            if re.match(RECEIVE,mid) or re.match(r'was\s+[a-z]+ed\b',mid) and not re.match(r'was (?:authorized|permitted|allowed|scheduled) to\b',mid):
                return None
            actor.add(a)
        else:
            return None
    possible=[(x,y) for x,y in ((a,b),(b,a)) if actor.issubset({x}) and other.issubset({y})]
    return list(possible[0]) if len(possible)==1 else None


def decide(target,notes,roster,text,logits):
    base=prior.decide(target,notes,text,logits)
    d={k:base[q] for k,q in zip(ARMS[:4],('r99','lexical','one_way','abi'))}
    bound=dict(notes).get(target,'')
    slots=[{'state':state(s),'roles':roles(s,roster)} for s in (bound,text)]
    for name,keys in [('state_only',['state']),('roles_only',['roles']),('abi',['state','roles'])]:
        why=base['abi']['reason']
        if why=='accept':
            for key in keys:
                a,b=(s[key] for s in slots)
                if a is None or b is None or a!=b:
                    why=key+('_unknown' if a is None or b is None else '_mismatch')
                    break
        d[name]={'reason':why,'retained_sha256':sha256(text.encode()).hexdigest() if why=='accept' else None}
    return dict(slots=slots,arms=d)
