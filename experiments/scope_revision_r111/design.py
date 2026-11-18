"""Prospective R111 inputs and limits; no model calls or scorer feedback."""
from __future__ import annotations

import copy
import hashlib
import random
import re
from pathlib import Path

from experiments.scope_adherence_r110 import design as previous

ROOT = previous.ROOT
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/scope_revision_r111'
STAGES = ('draft', 'revision')
R110_DECISION = 'b29a4855de20f1fd23d3712720492bfad2f15826e3273b773803b95c343c5fa2'
REVISION = (
    'Review your draft against the original request and supplied notes. '
    'Write one complete replacement answer, returning only that answer, not a review. '
    'Keep every required identifier exactly, including case and suffixes, and retain all requested facts. '
    'Follow the original format, tone, and word limit. Do not invent information or imply that pending approval has been granted.'
)
SENTENCE_REVISION = (
    ' The original request requires two or three sentences. Count the sentences in your draft, '
    'combine related facts as needed, and return two or three complete sentences without losing any requested fact.'
)


def contract():
    return {
        'quality_gates': previous.contract()['quality_gates'],
        'source': {**previous.contract()['source'], 'context_tokens': 2048},
        'population_seed': 111001, 'rows': 96, 'pairs': 48,
        'stages': list(STAGES), 'repeats': ['initial', 'replay'],
        'revision_policy': 'always exactly one revision, no scorer feedback or best-of selection',
        'scope_cases_per_class': 100, 'scope_classes': 8, 'maximum_scope_errors': 0,
        'bootstrap_seed': 108001, 'bootstrap_draws': 10000, 'primary_lower_bound': 0,
        'quality_noninferiority_lower_bound': -0.10,
        'maximum_calls': 384, 'maximum_output_tokens': 384 * 640,
        'minimum_free_bytes': 10 * 1024**3, 'maximum_evidence_bytes': 1024**3,
        'scope_seconds': 1800, 'worker_seconds': 14400, 'controller_seconds': 30600,
        'new_download_bytes': 0, 'gpu_setting_changes': [], 'training_steps': 0,
        'recipient_calls': 0, 'automatic_retries': 0, 'automatic_follow_on': False,
    }


def population():
    rng = random.Random(111001)
    days = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')
    result = []
    for index in range(48):
        start = rng.randrange(5)
        for offset, old in enumerate(previous.population()[index * 2:index * 2 + 2]):
            row = copy.deepcopy(old)
            identifiers = set(re.findall(r'\b(?:NOTE|Person)[A-Za-z0-9]+\b', row['prompt']))
            mapping = {v: v.replace('N110', 'N111') for v in identifiers}
            mapping[next(v for v in days if v in row['prompt'])] = days[(start + offset) % 5]
            pattern = re.compile(r'\b(?:' + '|'.join(re.escape(v) for v in sorted(mapping)) + r')\b')
            def replace(value):
                return pattern.sub(lambda m: mapping[m.group()], value)
            row.update(id=f'r111-{index:03d}-{row["panel"]}', pair=f'r111-{index:03d}',
                       split='new-instantiations-of-known-templates-not-hidden-families')
            row['prompt'], row['code'] = replace(row['prompt']), replace(row['code'])
            row['required'] = [replace(v) for v in row['required']]
            row['plan']['facts'] = [replace(v) for v in row['plan']['facts']]
            row['plan']['request'] = replace(row['plan']['request'])
            result.append(row)
    return result


def revision_request(row):
    return REVISION + (SENTENCE_REVISION if row['task'] in previous.INTERVENTION_TASKS else '')


def render(row, stage, tokenizer, draft=None):
    messages = [{'role': 'system', 'content': previous.SYSTEM}, {'role': 'user', 'content': row['prompt']}]
    if stage == 'revision':
        if not isinstance(draft, str):
            raise ValueError('Revision requires the complete generated draft')
        messages += [{'role': 'assistant', 'content': draft}, {'role': 'user', 'content': revision_request(row)}]
    elif stage != 'draft' or draft is not None:
        raise ValueError('Unregistered stage or draft payload')
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def challenge_seed(commit, protocol_sha):
    if not re.fullmatch('[0-9a-f]{40}', commit) or not re.fullmatch('[0-9a-f]{64}', protocol_sha):
        raise ValueError('Exact commit and protocol identity required')
    return hashlib.sha256(f'r111-scope-v1\n{commit}\n{protocol_sha}\n'.encode()).hexdigest()
