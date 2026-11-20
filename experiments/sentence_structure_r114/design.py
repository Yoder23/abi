"""R114 prospective inputs and strict, non-repairing sentence codec."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import re
from pathlib import Path

from experiments.count_feedback_r112 import design as old

ROOT, HERE = old.ROOT, Path(__file__).resolve().parent
OUT = ROOT / 'results/sentence_structure_r114'
STAGES = ('draft', 'revision', 'structured')
TASKS = old.TASKS
sentence_count = old.sentence_count
STRUCTURE = (
    ' Serialize your replacement answer as one JSON array containing two or three strings. '
    'Each string must be exactly one complete sentence ending in a period, question mark, or exclamation mark. '
    'Combine related facts within sentences while preserving every requested fact and the original tone. '
    'No newline or leading/trailing whitespace inside a string. Return only the array, with no code fence or commentary.'
)


def contract():
    result = copy.deepcopy(old.contract())
    result.update(format='r114-contract/1', population_seed=114001, stages=list(STAGES),
        feedback='No score or count feedback; sentence serialization on 24 rows',
        codec='Strict 2-3 string JSON array, one sentence each, exact join; no repair',
        primary='structured minus revision sentence adherence; 12 paired clusters',
        scope_classes=['negative_instructions', 'absent_confirmation', 'waiting_prerequisites',
                       'nonfinal_process', 'affirmative_contrasts', 'negated_nonfinal', 'mixed_scope', 'ambiguity'])
    return result


def population():
    rng, result = random.Random(114001), []
    days = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')
    previous = old.population()
    for index in range(48):
        start = rng.randrange(5)
        for offset, item in enumerate(previous[index * 2:index * 2 + 2]):
            row = copy.deepcopy(item)
            identifiers = set(re.findall(r'\b(?:NOTE|Person)[A-Za-z0-9]+\b', row['prompt']))
            mapping = {v: v.replace('N112', 'N114') for v in identifiers}
            mapping[next(v for v in days if v in row['prompt'])] = days[(start + offset) % 5]
            pattern = re.compile(r'\b(?:' + '|'.join(re.escape(v) for v in sorted(mapping)) + r')\b')
            def replace(value):
                return pattern.sub(lambda m: mapping[m.group()], value)
            row.update(id=f'r114-{index:03d}-{row["panel"]}', pair=f'r114-{index:03d}',
                       split='fresh-instances-known-templates-not-hidden-families')
            row['prompt'], row['code'] = replace(row['prompt']), replace(row['code'])
            row['required'] = [replace(v) for v in row['required']]
            row['plan']['facts'] = [replace(v) for v in row['plan']['facts']]
            row['plan']['request'] = replace(row['plan']['request'])
            result.append(row)
    return result


def revision_request(row, stage, draft):
    if stage not in STAGES[1:] or not isinstance(draft, str):
        raise ValueError('Missing complete draft or unregistered stage')
    return old.previous.revision_request(row) + (STRUCTURE if stage == 'structured' and row['task'] in TASKS else '')


def render(row, stage, tok, draft=None):
    messages = [{'role': 'system', 'content': old.previous.previous.SYSTEM},
                {'role': 'user', 'content': row['prompt']}]
    if stage in STAGES[1:]:
        messages += [{'role': 'assistant', 'content': draft},
                     {'role': 'user', 'content': revision_request(row, stage, draft)}]
    elif stage != 'draft' or draft is not None:
        raise ValueError('Unregistered stage/draft')
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def decode(answer, active):
    if not isinstance(answer, str):
        raise TypeError('Codec requires text')
    if not active:
        return {'active': False, 'valid': True, 'sentences': None, 'payload': answer.strip(), 'error': None}
    value, error = None, None
    try:
        value = json.loads(answer, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
        if not isinstance(value, list) or len(value) not in (2, 3):
            error = 'not-two-or-three-element-array'
        elif any(not isinstance(v, str) or not v or v != v.strip() or len(v.splitlines()) != 1
                 or v[-1:] not in ('.', '!', '?') or sentence_count(v) != 1 for v in value):
            error = 'not-unmodified-single-sentence-strings'
        else:
            for v in value:
                v.encode('utf-8')
    except (ValueError, TypeError, RecursionError, UnicodeError):
        error = 'invalid-json'
    return {'active': True, 'valid': error is None, 'sentences': value if error is None else None,
            'payload': ' '.join(value) if error is None else '', 'error': error}


def challenge_seed(commit, protocol_sha):
    if not re.fullmatch('[0-9a-f]{40}', commit) or not re.fullmatch('[0-9a-f]{64}', protocol_sha):
        raise ValueError('Exact content identities required')
    return hashlib.sha256(f'r114-scope-v1\n{commit}\n{protocol_sha}\n'.encode()).hexdigest()
