"""Construction-labeled R112 challenge; no evaluator import or score selection."""
from __future__ import annotations

import itertools
import random

from .. import design as d

LABELS = dict(zip(d.contract()['scope_classes'], (
    'NO_FORBIDDEN_ASSERTION', 'NO_FORBIDDEN_ASSERTION', 'NO_FORBIDDEN_ASSERTION',
    'NO_FORBIDDEN_ASSERTION', 'FORBIDDEN_ASSERTION', 'FORBIDDEN_ASSERTION',
    'FORBIDDEN_ASSERTION', 'UNRESOLVED',
)))


def pools():
    result = {name: {} for name in LABELS}
    def add(group, family, text):
        result[group].setdefault(family, set()).update((text.lower(), text.title(), text.upper()))
    boundaries = ('. ', '; ', ', ', '\n', ' and ', ' but ', ' \u2013 ')
    for noun, status, boundary in itertools.product(
            ('arrangements', 'details', 'plans', 'requests'),
            ('pending', 'provisional', 'unconfirmed'), boundaries):
        add('bare_nonfinal', 'with', f'With {noun} {status}{boundary}please wait.')
        add('bare_nonfinal', 'nominal', f'The {noun} {status}{boundary}please check.')
        for other in ('pending', 'provisional', 'unconfirmed'):
            add('purpose_modifiers', 'is', f'Approval for the {status} {noun} is still {other}{boundary}please wait.')
            add('purpose_modifiers', 'remains', f'Final approval for {status} {noun} remains {other}{boundary}please wait.')
        for family, phrase in (
                ('plural-process', 'approval processes progress'),
                ('single-process', 'the final approval process continues'),
                ('until-approval', f'all {noun} are subject to change until final approval')):
            add('process_and_prerequisite', family, f'{phrase}{boundary}the {noun} are {status}.')
        for family, phrase in (
                ('bare-no', f'no {noun} {status}'),
                ('purpose-not', f'approval for the {status} {noun} is not pending'),
                ('not-attributive', f'this is not a {status} arrangement')):
            add('negated_nonfinal', family, phrase + boundary + 'please check.')
        for state in ('approved', 'confirmed', 'granted'):
            positive = f'the {noun} are {state}'
            safe = f'approval for the {status} {noun} remains pending'
            add('affirmative_traps', 'modified-purpose',
                f'Approval for the {status} {noun} has been {state}{boundary}please proceed.')
            add('affirmative_traps', 'process-then-positive',
                f'Approval processes progress{boundary}{positive}.')
            for direction in (0, 1):
                add('mixed_scope', f'order-{direction}', boundary.join([safe, positive][::1 if direction == 0 else -1])+'.')
            for family, phrase in (
                    ('quote', f'"{safe}"'), ('perhaps', f'Perhaps {safe}'),
                    ('reported', f'Reportedly {safe}'),
                    ('metalinguistic', f'I do not think {positive}'),
                    ('double-negation', f'Nothing has not yet been {state}')):
                add('ambiguity', family, phrase + boundary + 'please check.')
    for state, boundary in itertools.product(('approved', 'confirmed', 'granted'), boundaries):
        for family, phrase in (
                ('not-yet-been', f'final approval has not yet been {state}'),
                ('contracted-not-yet', f"approval hasn't yet been {state}"),
                ('plural-not-yet', f'the arrangements have not yet been {state}'),
                ('been-yet', f'approval has not been yet {state}')):
            for suffix in ('please wait', 'please check', 'an update will follow'):
                add('negation_placement', family, phrase + boundary + suffix + '.')
    return {g: {f: sorted(v) for f, v in families.items()} for g, families in result.items()}


def generate(commit, protocol_sha):
    seed = d.challenge_seed(commit, protocol_sha)
    rng = random.Random(int(seed, 16))
    result, seen = [], set()
    for group, families in pools().items():
        shuffled = {f: rng.sample(v, len(v)) for f, v in sorted(families.items())}
        selected = []
        while len(selected) < 100:
            for family, candidates in shuffled.items():
                while candidates:
                    text = candidates.pop()
                    if text not in seen:
                        seen.add(text)
                        selected.append({'class': group, 'family': family, 'text': text,
                                         'expected': LABELS[group]})
                        break
                else:
                    raise RuntimeError('Required family exhausted')
                if len(selected) == 100:
                    break
        rng.shuffle(selected)
        result.extend(selected)
    return {'format': 'r112-scope-challenge/1', 'seed_sha256': seed,
            'implementation_commit': commit, 'protocol_sha256': protocol_sha,
            'label_origin': 'author construction, not independent human labels',
            'cases': [{**row, 'id': f'r112-scope-{i:03d}'} for i, row in enumerate(result)]}
