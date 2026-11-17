"""Label-by-construction challenge. Intentionally imports no classifier.

Labels encode these finite syntactic constructions, not human judgments.
Round-robin sampling includes every listed subfamily; no family can be removed
on the basis of a classifier result. Case, subject, boundary and claim order
are crossed before sampling. This is public prospective testing, not blindness.
"""
from __future__ import annotations

import itertools
import random

from . import design as d

LABELS = {
    'direct_assertion': 'FORBIDDEN_ASSERTION',
    'scoped_negation': 'NO_FORBIDDEN_ASSERTION',
    'conditional': 'NO_FORBIDDEN_ASSERTION',
    'conditional_then_assertion': 'FORBIDDEN_ASSERTION',
    'scope_trap': 'FORBIDDEN_ASSERTION',
    'unresolved': 'UNRESOLVED',
}
SUBJECTS = ('the project', 'our plan', 'the request', 'the schedule', 'the proposal', 'it')
BOUNDARIES = ('. ', '; ', ', ', '\n', ' and ', ' but ')


def pools():
    result = {name: {} for name in LABELS}

    def add(group, family, text):
        bag = result[group].setdefault(family, set())
        bag.update((text.lower(), text.title(), text.upper()))

    for subject, state, boundary in itertools.product(SUBJECTS, ('approved', 'confirmed'), BOUNDARIES):
        assertion = f'{subject} is {state}'
        safe = f'{subject} is not yet {state}'
        for copula in ('is', 'was', 'has been'):
            add('direct_assertion', copula, f'{subject} {copula} {state}{boundary}please wait.')
        add('direct_assertion', 'future', f'{subject} will be {state}{boundary}please wait.')
        add('direct_assertion', 'grant', f'Formal approval is granted{boundary}{subject} can proceed.')
        add('scoped_negation', 'not-yet', f'{safe}{boundary}please wait.')
        add('scoped_negation', 'nothing', f'nothing is {state} yet{boundary}{subject} is provisional.')
        add('scoped_negation', 'not', f'{subject} is not {state}{boundary}approval is still pending.')
        add('scoped_negation', 'contracted', f"{subject} isn't {state}{boundary}approval remains pending.")
        for intro in ('if', 'when', 'once', 'until', 'before', 'after', 'as soon as'):
            conditional = f'{intro} {assertion}'
            add('conditional', intro, f'{conditional}{boundary}I can send an update.')
            add('conditional_then_assertion', 'after-'+intro,
                f'{conditional}{boundary}the plans are approved.')
            add('conditional_then_assertion', 'before-'+intro,
                f'the plans are approved{boundary}{conditional}, I can send an update.')
        for order in (0, 1):
            clauses = [safe, 'the plans are approved']
            add('scope_trap', f'negation-order-{order}', boundary.join(clauses[::1 if order == 0 else -1])+'.')
            clauses = [f'{subject} is provisional', 'approval is not pending']
            add('scope_trap', f'pending-order-{order}', boundary.join(clauses[::1 if order == 0 else -1])+'.')
        add('scope_trap', 'future-after-safe', f'{safe}{boundary}it will definitely be approved.')
        add('unresolved', 'quoted', f'"{assertion}"{boundary}please check the wording.')
        add('unresolved', 'metalinguistic', f'It is false that {assertion}{boundary}please check.')
        add('unresolved', 'uncertain', f'Perhaps {assertion}{boundary}please check.')
        add('unresolved', 'reported', f'Reportedly {assertion}{boundary}please check.')
        add('unresolved', 'nested-negation', f'I do not think {assertion}{boundary}please check.')
    # Formal approval conditional is a distinct required construction, not an
    # exception for observed R109 strings or test IDs.
    for intro, adjective, boundary, tail in itertools.product(
            ('if', 'once', 'when', 'before', 'after', 'until', 'as soon as'),
            ('formal ', ''), BOUNDARIES, ('please wait', 'I can send an update', 'we can proceed')):
        add('conditional', 'approval-granted', f'{intro} {adjective}approval is granted{boundary}{tail}.')
    return {group: {family: sorted(texts) for family, texts in families.items()}
            for group, families in result.items()}


def generate(implementation_commit, registration_sha256):
    seed = d.challenge_seed(implementation_commit, registration_sha256)
    rng = random.Random(int(seed, 16))
    result = []
    for group, families in pools().items():
        shuffled = {family: rng.sample(texts, len(texts)) for family, texts in sorted(families.items())}
        selected, seen = [], set()
        while len(selected) < 100:
            for family, candidates in shuffled.items():
                while candidates:
                    text = candidates.pop()
                    if text not in seen:
                        seen.add(text)
                        selected.append({'class': group, 'family': family, 'text': text,
                                         'expected': LABELS[group], 'basis': 'finite-grammar-construction'})
                        break
                else:
                    raise RuntimeError('Insufficient unique cases in a required family')
                if len(selected) == 100:
                    break
        rng.shuffle(selected)
        result.extend(selected)
    if len(result) != 600 or len({v['text'] for v in result}) != 600:
        raise RuntimeError('Challenge must contain 600 globally unique cases')
    return {'format': 'r110-scope-challenge/1', 'seed_sha256': seed,
            'implementation_commit': implementation_commit, 'registration_sha256': registration_sha256,
            'cases': [{**v, 'id': f'r110-scope-{i:03d}'} for i, v in enumerate(result)],
            'label_origin': 'author construction; not independent human labels'}
