"""Prospective author-constructed scope cases; no classifier or score selection."""
from __future__ import annotations

import random

from . import design as d

LABELS = dict(zip(d.contract()['scope_classes'], (
    'NO_FORBIDDEN_ASSERTION', 'FORBIDDEN_ASSERTION', 'NO_FORBIDDEN_ASSERTION',
    'NO_FORBIDDEN_ASSERTION', 'FORBIDDEN_ASSERTION', 'FORBIDDEN_ASSERTION',
    'UNRESOLVED', 'NO_FORBIDDEN_ASSERTION')))


def pools():
    result = {g: {} for g in LABELS}
    boundaries = ('. ', '; ', ', ', '\n', ' and ', ' but ', ' \u2013 ')
    def add(group, family, phrase):
        for boundary in boundaries:
            for note in ('please wait', 'please check the notes', 'please retain this record'):
                text = phrase + boundary + note + '.'
                result[group].setdefault(family, set()).update((text.lower(), text.title(), text.upper()))
    for noun in ('changes', 'decisions', 'arrangements', 'details', 'plans', 'requests'):
        for copula in ('are', 'were', 'have been'):
            for state in ('approved', 'confirmed', 'granted'):
                negative = f'No {noun} {copula} {state}'
                positive = f'The {noun} {copula} {state}'
                add('negative_quantifiers', copula, negative)
                add('affirmative_contrasts', 'positive-subject', positive)
                add('ambiguity', 'double-negation', f'No {noun} {copula} not {state}')
                add('ambiguity', 'not-all', f'Not all {noun} {copula} {state}')
                for boundary in boundaries:
                    add('mixed_scope', 'negative-then-positive', negative + boundary + positive)
                    add('mixed_scope', 'positive-then-negative', positive + boundary + negative)
            for state in ('pending', 'provisional', 'unconfirmed'):
                add('negated_nonfinal', copula, f'No {noun} {copula} {state}')
        for modifier in ('', 'final ', 'formal '):
            approval = modifier + 'approval'
            for status in ('provisional', 'unconfirmed'):
                add('nominal_prerequisites', 'stacked-modifiers', f'{status} {noun} pending {approval}')
            add('nominal_prerequisites', 'nominal-subject', f'The {noun} remain subject to {approval}')
    for modifier in ('', 'final ', 'formal '):
        approval = modifier + 'approval'
        for subject, verb in (('this', 'remains'), ('it', 'is'), ('these', 'are'), ('those', 'remain')):
            add('nominal_prerequisites', 'pronoun-subject', f'{subject} {verb} subject to {approval}')
        for verb in ('issued', 'given', 'received', 'granted'):
            add('absent_confirmation', 'confirmation-complement', f'No confirmation of {approval} has been {verb}')
            add('affirmative_contrasts', 'confirmation-complement', f'Confirmation of {approval} has been {verb}')
        for prep in ('on', 'of', 'about'):
            add('absent_confirmation', 'no-promise', f'No promises {prep} {approval}')
        add('ambiguity', 'as-approved', f'We will proceed as approved with {approval} pending')
        for phrase in (f'{approval} is pending', f'{approval} is confirmed'):
            add('ambiguity', 'quoted', '"' + phrase + '"')
            add('ambiguity', 'reported', 'Reportedly ' + phrase)
        for verb in ('imply', 'assume', 'suggest', 'indicate'):
            add('guard_regression', 'negative-instruction', f'Do not {verb} {approval} has been granted')
        add('guard_regression', 'waiting', f'We are waiting on {approval}')
        add('guard_regression', 'prerequisite', f'We will proceed only after {approval} is received')
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
    return {'format': 'r115-scope-challenge/1', 'seed_sha256': seed,
            'implementation_commit': commit, 'protocol_sha256': protocol_sha,
            'label_origin': 'author construction; not independent human ratings',
            'cases': [{**row, 'id': f'r115-scope-{i:03d}'} for i, row in enumerate(result)]}
