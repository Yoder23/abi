"""Prospective R112 count-feedback contrast, without model or evaluator execution."""
from __future__ import annotations

import copy
import hashlib
import random
import re
from pathlib import Path

from experiments.scope_revision_r111 import design as previous

ROOT = previous.ROOT
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/count_feedback_r112'
STAGES = ('draft', 'revision', 'count_feedback')
TASKS = ('summarization', 'tone_control')
R111_DECISION = '2a26b2ec4751e81835a7d0c47fa2aed9ad96786f7b710247a48516b52e7f22ce'
R111_MANIFEST = '13a93229d5d9f13f19df38a8eeb60017d57e6a6ebc42ccd226647304287e0f76'
FEEDBACK = ' Observed sentence count in the draft: {count}. Required sentence count: two or three.'
SENTENCE_SPLIT = r'[.!?]+(?:\s+|$)'


def sentence_count(answer):
    """Exact historical mechanical counter, not a general linguistic parser."""
    if not isinstance(answer, str):
        raise TypeError('Complete decoded draft must be text')
    return sum(bool(part.strip()) for part in re.split(SENTENCE_SPLIT, answer.strip()))


def contract():
    return {
        'format': 'r112-contract/1',
        'quality_gates': copy.deepcopy(previous.contract()['quality_gates']),
        'source': copy.deepcopy(previous.contract()['source']),
        'population_seed': 112001, 'rows': 96, 'pairs': 48,
        'stages': list(STAGES), 'repeats': ['initial', 'replay'],
        'shared_draft': True, 'revision_selection': 'always emit the designated arm, never best-of',
        'feedback': 'sentence count only on 24 sentence-limited inputs; no semantic feedback',
        'unchanged_input_controls': 72, 'required_unchanged_token_matches_per_repeat': 72,
        'scope_classes': ['bare_nonfinal', 'negation_placement', 'purpose_modifiers',
                          'process_and_prerequisite', 'affirmative_traps', 'negated_nonfinal',
                          'mixed_scope', 'ambiguity'],
        'scope_cases_per_class': 100, 'maximum_scope_errors': 0,
        'bootstrap_seed': 108001, 'bootstrap_draws': 10000,
        'primary': 'count_feedback minus revision sentence adherence; 12 paired clusters',
        'primary_lower_bound_strictly_greater_than': 0,
        'quality_noninferiority_lower_bound': -0.10,
        'maximum_calls': 576, 'maximum_output_tokens': 576 * 640,
        'minimum_free_bytes': 10 * 1024**3, 'maximum_evidence_bytes': 1024**3,
        'scope_seconds': 1800, 'worker_seconds': 14400, 'controller_seconds': 30600,
        'new_download_bytes': 0, 'training_steps': 0, 'recipient_calls': 0,
        'gpu_setting_changes': [], 'automatic_retries': 0, 'automatic_follow_on': False,
    }


def population():
    rng = random.Random(112001)
    days = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')
    old_rows, result = previous.population(), []
    for index in range(48):
        start = rng.randrange(5)
        for offset, old in enumerate(old_rows[index * 2:index * 2 + 2]):
            row = copy.deepcopy(old)
            identifiers = set(re.findall(r'\b(?:NOTE|Person)[A-Za-z0-9]+\b', row['prompt']))
            mapping = {v: v.replace('N111', 'N112') for v in identifiers}
            mapping[next(v for v in days if v in row['prompt'])] = days[(start + offset) % 5]
            pattern = re.compile(r'\b(?:' + '|'.join(re.escape(v) for v in sorted(mapping)) + r')\b')
            def replace(value):
                return pattern.sub(lambda m: mapping[m.group()], value)
            row.update(id=f'r112-{index:03d}-{row["panel"]}', pair=f'r112-{index:03d}',
                       split='new-instantiations-of-known-templates-not-hidden-families')
            row['prompt'], row['code'] = replace(row['prompt']), replace(row['code'])
            row['required'] = [replace(v) for v in row['required']]
            row['plan']['facts'] = [replace(v) for v in row['plan']['facts']]
            row['plan']['request'] = replace(row['plan']['request'])
            result.append(row)
    return result


def revision_request(row, stage, draft):
    if stage not in ('revision', 'count_feedback') or not isinstance(draft, str):
        raise ValueError('Unregistered revision stage or missing complete draft')
    request = previous.revision_request(row)
    if stage == 'count_feedback' and row['task'] in TASKS:
        request += FEEDBACK.format(count=sentence_count(draft))
    return request


def render(row, stage, tokenizer, draft=None):
    messages = [{'role': 'system', 'content': previous.previous.SYSTEM},
                {'role': 'user', 'content': row['prompt']}]
    if stage in ('revision', 'count_feedback'):
        request = revision_request(row, stage, draft)
        messages += [{'role': 'assistant', 'content': draft}, {'role': 'user', 'content': request}]
    elif stage != 'draft' or draft is not None:
        raise ValueError('Unregistered stage or draft payload')
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def challenge_seed(commit, protocol_sha):
    if not re.fullmatch('[0-9a-f]{40}', commit) or not re.fullmatch('[0-9a-f]{64}', protocol_sha):
        raise ValueError('Exact implementation commit and protocol digest required')
    return hashlib.sha256(f'r112-scope-v1\n{commit}\n{protocol_sha}\n'.encode()).hexdigest()
