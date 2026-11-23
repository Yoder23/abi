"""R117 candidate filter. No panel, teacher, expected labels or outcome inputs."""
from hashlib import sha256
import math
import re

from experiments.grounded_english_r99.core import normalize

ID = re.compile(r'\bP[0-9A-F]{8}\b', re.I)
ARMS = ('r99', 'lexical', 'one_way', 'abi')


def pairs(target, notes, candidate):
    """Notes are ordered (project ID, source sentence) records."""
    if not ID.fullmatch(target) or len({k for k, _ in notes}) != len(notes):
        raise ValueError('malformed source binding')
    if any(not ID.fullmatch(k) or not isinstance(v, str) or k not in v for k, v in notes):
        raise ValueError('invalid source record')
    full = '\n'.join(v for _, v in notes)
    bound = dict(notes).get(target, '')
    return [(full, candidate), (bound, candidate), (candidate, bound)]


def entails(logits):
    if len(logits) != 3 or not all(type(x) in (float, int) and math.isfinite(x) for x in logits):
        raise ValueError('missing or invalid logits')
    return logits[1] > max(logits[0], logits[2])


def decide(target, notes, candidate, logits):
    observations = pairs(target, notes, candidate)
    if len(logits) != len(observations):
        raise ValueError('three ordered logit triplets required')
    whole, forward, reverse = map(entails, logits)
    lexical = bool(candidate.strip() and len(candidate.split()) <= 80 and set(ID.findall(candidate)) == {target})
    historical = normalize({'task': 'summarization'}, candidate)
    if historical['text'] != candidate:
        raise ValueError('historical pass-through changed')
    reasons = ('accept', 'accept' if lexical else 'lexical',
               'lexical' if not lexical else 'accept' if whole else 'notes_not_entailing',
               'no_bound_fact' if not observations[1][0] else 'lexical' if not lexical else
               'fact_not_entailing' if not forward else 'incomplete' if not reverse else 'accept')
    digest = sha256(candidate.encode()).hexdigest()
    return {arm: {'reason': reason, 'retained_sha256': digest if reason == 'accept' else None}
            for arm, reason in zip(ARMS, reasons)}
