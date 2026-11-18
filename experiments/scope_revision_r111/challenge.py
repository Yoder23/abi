"""Construction-defined challenge; no classifier import or scoring-based sampling."""
from __future__ import annotations

import itertools
import random

from . import design as d

LABELS = {
    'perfect_negation': 'NO_FORBIDDEN_ASSERTION',
    'new_conditionals': 'NO_FORBIDDEN_ASSERTION',
    'nonfinal_phrases': 'NO_FORBIDDEN_ASSERTION',
    'new_affirmatives': 'FORBIDDEN_ASSERTION',
    'negated_status': 'FORBIDDEN_ASSERTION',
    'conditional_then_assertion': 'FORBIDDEN_ASSERTION',
    'mixed_scope': 'FORBIDDEN_ASSERTION',
    'ambiguity': 'UNRESOLVED',
}


def pools():
    result = {name: {} for name in LABELS}
    def add(group, family, text):
        result[group].setdefault(family, set()).update((text.lower(), text.title(), text.upper()))
    boundaries = ('. ', '; ', ', ', '\n', ' and ', ' but ', ' \u2013 ')
    for state, boundary, subject in itertools.product(('approved', 'confirmed', 'granted'), boundaries,
                                                     ('the project', 'our plan', 'the request', 'it')):
        safe = f'final approval has not been {state}'
        positive = f"{subject}'s {state}"
        for negative in ('has not', "hasn't", 'hasn\u2019t'):
            add('perfect_negation', negative, f'final approval {negative} been {state}{boundary}please wait.')
        add('perfect_negation', 'plural', f'the arrangements have not been {state}{boundary}please wait.')
        for intro in ('if', 'when', 'once', 'until', 'before', 'after', 'as soon as'):
            reduced = f'{intro} {state}'
            contracted = f'{intro} {positive}'
            for family, clause in (('reduced', reduced), ('contracted', contracted),
                                   ('given', f'{intro} approval is given'),
                                   ('through', f'{intro} final approval comes through')):
                add('new_conditionals', family, f'{clause}{boundary}please send an update.')
                add('conditional_then_assertion', family, f'{clause}{boundary}the plans are approved.')
        add('new_affirmatives', 'contracted', f'{positive}{boundary}please proceed.')
        add('new_affirmatives', 'given', f'final approval is given{boundary}{subject} can proceed.')
        add('new_affirmatives', 'through', f'approval comes through{boundary}{subject} can proceed.')
        for direction in (0, 1):
            text = boundary.join([safe, positive][::1 if direction == 0 else -1]) + '.'
            add('mixed_scope', f'order-{direction}', text)
        for phrase in (f'Perhaps {positive}', f'"{positive}"', f'I do not think {positive}',
                       f'Reportedly {positive}', f'nothing has not been {state}'):
            add('ambiguity', phrase.split()[0].lower(), phrase + boundary + 'please check.')
    for noun, status, boundary in itertools.product(('arrangements', 'details', 'plans', 'requests'),
                                                    ('pending', 'provisional', 'unconfirmed'), boundaries):
        forms = {
            'nominal-still': f'with {noun} still {status}',
            'nominal-current': f'with {noun} currently {status}',
            'nominal-remaining': f'with {noun} remaining {status}',
            'state': f'the {noun} are currently in a {status} state',
            'attributive': f'these are {status} {noun}',
            'contracted': f"it's still {status}",
            'purpose': f'approval for the {noun} is still {status}',
            'status': f'approval status is {status}',
            'subject-to': f'all {noun} remain subject to final approval',
        }
        for family, phrase in forms.items():
            add('nonfinal_phrases', family, phrase + boundary + 'please wait.')
        for family, phrase in (
            ('article', f'this is not a {status} arrangement'),
            ('state', f'the {noun} are not currently in a {status} state'),
            ('negative-subject', f'no {noun} still {status}')):
            add('negated_status', family, phrase + boundary + 'please check.')
    return {g: {f: sorted(v) for f, v in families.items()} for g, families in result.items()}


def generate(commit, protocol_sha):
    seed = d.challenge_seed(commit, protocol_sha)
    rng = random.Random(int(seed, 16))
    result, global_seen = [], set()
    for group, families in pools().items():
        shuffled = {f: rng.sample(v, len(v)) for f, v in sorted(families.items())}
        selected = []
        while len(selected) < 100:
            for family, candidates in shuffled.items():
                while candidates:
                    text = candidates.pop()
                    if text not in global_seen:
                        global_seen.add(text)
                        selected.append({'class': group, 'family': family, 'text': text,
                                         'expected': LABELS[group]})
                        break
                else:
                    raise RuntimeError('Required family exhausted')
                if len(selected) == 100:
                    break
        rng.shuffle(selected)
        result.extend(selected)
    return {'format': 'r111-scope-challenge/1', 'seed_sha256': seed,
            'implementation_commit': commit, 'protocol_sha256': protocol_sha,
            'label_origin': 'author construction, not independent human labels',
            'cases': [{**row, 'id': f'r111-scope-{i:03d}'} for i, row in enumerate(result)]}
