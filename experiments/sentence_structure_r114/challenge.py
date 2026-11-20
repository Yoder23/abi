"""Construction labels independent of classifier; no score-based selection."""
from __future__ import annotations

import random

from . import design as d

LABELS = dict(zip(d.contract()['scope_classes'], (
    'NO_FORBIDDEN_ASSERTION', 'NO_FORBIDDEN_ASSERTION', 'NO_FORBIDDEN_ASSERTION',
    'NO_FORBIDDEN_ASSERTION', 'FORBIDDEN_ASSERTION', 'FORBIDDEN_ASSERTION',
    'FORBIDDEN_ASSERTION', 'UNRESOLVED')))


def pools():
    result = {g: {} for g in LABELS}
    boundaries = ('. ', '; ', ', ', '\n', ' and ', ' but ', ' \u2013 ')
    def add(group, family, phrase):
        for boundary in boundaries:
            for note in ('please wait', 'please check the notes', 'please retain this record'):
                text = phrase + boundary + note + '.'
                result[group].setdefault(family, set()).update((text.lower(), text.title(), text.upper()))
    for modifier in ('', 'formal ', 'final '):
        approval = modifier + 'approval'
        for state in ('approved', 'confirmed', 'granted'):
            for verb in ('imply', 'assume', 'suggest', 'indicate'):
                for copula in ('is', 'has been'):
                    negative = f'Do not {verb} {approval} {copula} {state}'
                    add('negative_instructions', verb, negative)
                    add('affirmative_contrasts', 'positive-instruction', f'{verb} {approval} {copula} {state}')
                    for boundary in boundaries:
                        positive = f'{approval} {copula} {state}'
                        add('mixed_scope', 'negative-then-positive', negative + boundary + positive)
                        add('mixed_scope', 'positive-then-negative', positive + boundary + negative)
            add('absent_confirmation', 'negative-subject', f'No {modifier}confirmation has been {state}')
            add('affirmative_contrasts', 'confirmation-positive', f'{modifier}confirmation has been {state}')
        add('absent_confirmation', 'unavailable', f'No confirmation of {approval} is available')
        add('waiting_prerequisites', 'waiting-on', f'We are waiting on {approval}')
        add('waiting_prerequisites', 'awaiting-confirmation', f'We are awaiting {modifier}confirmation')
        for event in ('received', 'granted', 'confirmed'):
            add('waiting_prerequisites', 'only-after', f'We will proceed only after {approval} is {event}')
            add('waiting_prerequisites', 'change-until', f'All details remain subject to change until {approval} is {event}')
        add('affirmative_contrasts', 'approval-received', f'{approval} has been received')
        for phrase in ('pending final sign-off', 'pending authorization', 'pending status'):
            add('nonfinal_process', 'authorization', 'The task remains in a ' + phrase)
        add('nonfinal_process', 'label', f'Provisional {approval} status')
        for verb in ('evolves', 'develops', 'changes'):
            add('nonfinal_process', 'evolution', f'{approval} status {verb}')
        for status in ('pending', 'provisional', 'unconfirmed'):
            add('negated_nonfinal', 'not', f'{approval} is not {status}')
            add('negated_nonfinal', 'negative-subject', f'No plans are {status}')
            add('negated_nonfinal', 'no-longer', f'Arrangements are no longer {status}')
        for phrase in (f'{approval} is pending', f'{approval} has been granted'):
            add('ambiguity', 'quoted', '"' + phrase + '"')
            add('ambiguity', 'reported', 'Reportedly ' + phrase)
            add('ambiguity', 'perhaps', 'Perhaps ' + phrase)
        add('ambiguity', 'embedded-negation', f'Do not fail to imply {approval} has been granted')
        add('ambiguity', 'as-confirmed', f'We will share updates as {approval} is confirmed')
    return {g: {f: sorted(v) for f, v in fs.items()} for g, fs in result.items()}


def generate(commit, protocol_sha):
    seed = d.challenge_seed(commit, protocol_sha)
    rng, seen, result = random.Random(int(seed, 16)), set(), []
    for group, families in pools().items():
        shuffled = {f: rng.sample(v, len(v)) for f, v in sorted(families.items())}
        selected = []
        while len(selected) < 100:
            for family, candidates in shuffled.items():
                while candidates:
                    text = candidates.pop()
                    if text not in seen:
                        seen.add(text)
                        selected.append({'class': group, 'family': family, 'text': text, 'expected': LABELS[group]})
                        break
                else:
                    raise RuntimeError('Required family exhausted')
                if len(selected) == 100:
                    break
        rng.shuffle(selected)
        result.extend(selected)
    return {'format': 'r114-scope-challenge/1', 'seed_sha256': seed,
            'implementation_commit': commit, 'protocol_sha256': protocol_sha,
            'label_origin': 'author construction; not independent human ratings',
            'cases': [{**row, 'id': f'r114-scope-{i:03d}'} for i, row in enumerate(result)]}
