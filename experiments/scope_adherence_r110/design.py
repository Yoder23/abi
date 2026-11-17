"""R110 input design only: no scorer implementation, learning or model execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import re

from experiments.trained_grounding_r108 import core as reference

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'results/scope_adherence_r110/preregistration'
INTERFACES = ('plain', 'response_contract')
INTERVENTION_TASKS = ('summarization', 'tone_control')
CATALOG_SEED = 110001
SYSTEM = reference.prior.SYSTEM


def contract():
    return json.loads((HERE/'contract.json').read_text(encoding='utf-8'))


def population():
    """New instantiations, not new task semantics or independently hidden data."""
    original = reference.catalog()
    rng = random.Random(CATALOG_SEED)
    days = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')
    result = []
    for index in range(48):
        first_day = rng.randrange(len(days))
        for offset, old in enumerate(original[2*index:2*index+2]):
            row = copy.deepcopy(old)
            identifiers = set(re.findall(r'\b(?:NOTE|Person)[A-Za-z0-9]+\b', row['prompt']))
            mapping = {v: v.replace('N108', 'N110') for v in identifiers}
            old_day = next(d for d in days if d in row['prompt'])
            mapping[old_day] = days[(first_day+offset) % len(days)]
            pattern = r'\b(?:'+'|'.join(re.escape(k) for k in sorted(mapping))+r')\b'

            def replace(value):
                return re.sub(pattern, lambda match: mapping[match.group()], value)

            row.update(id=f'r110-{index:03d}-{row["panel"]}', pair=f'r110-{index:03d}',
                       split='registered-instantiations-of-known-templates')
            row['prompt'], row['code'] = replace(row['prompt']), replace(row['code'])
            row['required'] = [replace(v) for v in row['required']]
            row['plan']['facts'] = [replace(v) for v in row['plan']['facts']]
            row['plan']['request'] = replace(row['plan']['request'])
            result.append(row)
    return result


def response_contract(row):
    if row['task'] not in INTERVENTION_TASKS:
        return None
    return {'required_literals': list(row['required']), 'approval_status': 'pending',
            'sentence_count': 'two or three', 'maximum_words': 'one hundred twenty',
            'format': 'connected prose'}


def user_message(row, interface):
    if interface not in INTERFACES:
        raise ValueError('Unregistered interface')
    payload = response_contract(row)
    if interface == 'plain' or payload is None:
        return row['prompt']
    return (row['prompt']+'\n\nThis metadata restates the requested response; do not reproduce the metadata:'
            '\n<response_contract>\n'+json.dumps(payload, sort_keys=True, separators=(',', ':'))
            +'\n</response_contract>')


def render(row, interface, tokenizer):
    return tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user_message(row, interface)}],
        tokenize=False, add_generation_prompt=True)


def challenge_seed(implementation_commit, registration_sha256):
    """Public deterministic commitment rule, not a claim of assessor blindness."""
    import hashlib
    if not re.fullmatch(r'[0-9a-f]{40}', implementation_commit) or not re.fullmatch(r'[0-9a-f]{64}', registration_sha256):
        raise ValueError('Exact implementation commit and registration SHA-256 required')
    text = f'r110-scope-challenge-v1\n{implementation_commit}\n{registration_sha256}\n'
    return hashlib.sha256(text.encode('ascii')).hexdigest()
