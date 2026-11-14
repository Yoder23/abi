"""Prospective paired trained-task screen; deliberately not a fluency judge."""
from __future__ import annotations

import copy
import hashlib
import math
import re
from pathlib import Path

from experiments.evaluator_control_r104 import core as prior
from experiments.evaluator_control_r104.evaluator_cases import cases as old_cases

ROOT, HOST = prior.ROOT, prior.HOST
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/trained_grounding_r108'
SOURCE = 'Qwen/Qwen3-4B-Instruct-2507'
REVISION = 'cdbee75f17c01a7cc42f958dc650907174af0554'
HOST_REVISION = '2c3c03fd749616d253798bd33bdb01dc8cc02b01'
ARMS, TASKS, ROUTES = prior.ARMS, prior.TASKS, prior.ROUTE_ORDER
SEED, MAX_NEW, CONTEXT = 108001, 640, 1024
sha, read, rows, write, append = prior.sha, prior.read, prior.rows, prior.write_once, prior.append
checkpoint = prior.checkpoint
FILES = ('metadata.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'vocab.json', 'merges.txt')
SOURCE_FILES = ('LICENSE', 'README.md', 'config.json', 'generation_config.json',
                'merges.txt', 'model-00001-of-00003.safetensors',
                'model-00002-of-00003.safetensors', 'model-00003-of-00003.safetensors',
                'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json')


def training_provenance():
    """Recompute matched historical exposure, not matched trainable parameter counts."""
    root = ROOT/'results/grounded_english_r99_fp32'
    train = read(root/'train.json')
    tasks = {r['id']: r['task'] for r in train}
    if len(tasks) != 192 or set(tasks.values()) != set(TASKS):
        raise RuntimeError('Wrong historical training population')
    output, orders = {}, []
    for arm in ('lora', 'distillation'):
        folder = checkpoint(arm)
        receipt, log = read(folder/'training_result.json'), rows(folder/'training.jsonl')
        order = [v for r in log for v in r['sample_ids']]
        digest = hashlib.sha256(''.join(v+'\n' for v in order).encode()).hexdigest()
        if (len(log) != 384 or [r['step'] for r in log] != list(range(1, 385)) or len(order) != 768
                or set(order) != set(tasks) or any(order.count(v) != 4 for v in tasks)
                or receipt['sample_order_sha256'] != digest or receipt['training_log_sha256'] != sha(folder/'training.jsonl')
                or receipt['checkpoint_sha256'] != sha(folder/'model.safetensors')
                or receipt['source_rows_sha256'] != sha(root/'source/rows.jsonl')
                or receipt['loss_tokens'] != sum(r['loss_tokens'] for r in log)
                or receipt['successful_steps'] != 384 or receipt['training_rows'] != 192
                or any(not math.isfinite(r[k]) for r in log for k in ('loss', 'gradient_norm'))):
            raise RuntimeError('Unrecomputable historical matched exposure')
        output[arm] = {'training_rows': 192, 'steps': 384, 'loss_tokens': receipt['loss_tokens'],
                       'sample_order_sha256': digest, 'source_rows_sha256': receipt['source_rows_sha256'],
                       'trained_task_rows': {t: sum(v == t for v in tasks.values()) for t in TASKS},
                       'trainable_parameters_historical': receipt['trainable_parameters']}
        orders.append(order)
    if orders[0] != orders[1] or output['lora']['loss_tokens'] != output['distillation']['loss_tokens']:
        raise RuntimeError('Controls not exposure matched')
    return output


def catalog():
    result = []
    days = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')
    for index, original in enumerate(prior.catalog('qualification')):
        old_day = next(d for d in days if d in original['prompt'])
        for panel in ('a', 'b'):
            row = copy.deepcopy(original)
            identifiers = set(re.findall(r'\b(?:NOTE|Person)[A-Za-z0-9]+\b', row['prompt']))
            mapping = {v: v.replace('N104Q', 'N108' + panel.upper()) for v in identifiers}
            mapping[old_day] = days[(index + (panel == 'b')) % 5]
            def replace(text):
                return re.sub(r'\b(?:' + '|'.join(re.escape(k) for k in mapping) + r')\b',
                              lambda m: mapping[m.group()], text)
            row.update(id=f'r108-{index:03d}-{panel}', pair=f'r108-{index:03d}', panel=panel, split='prospective')
            row['prompt'], row['code'] = replace(row['prompt']), replace(row['code'])
            row['required'] = [replace(v) for v in row['required']]
            row['plan']['facts'] = [replace(v) for v in row['plan']['facts']]
            row['plan']['request'] = replace(row['plan']['request'])
            result.append(row)
    return result


def screen(row, answer):
    result = prior.screen(row, answer)
    result['r104_mechanical_pass'] = result['mechanical_pass']
    result['exact_required_literals'] = all(re.search(r'\b' + re.escape(v) + r'\b', answer) for v in row['required'])
    result['negated_pending_status'] = re.findall(
        r"\b(?:not|no longer|isn't|aren't|wasn't|weren't|isn’t|aren’t)\s+(?:still\s+)?(?:pending|provisional|unconfirmed|awaiting)\b",
        answer, re.I)
    result['mechanical_pass'] = bool(result['r104_mechanical_pass'] and result['exact_required_literals']
                                     and not result['negated_pending_status'])
    return result


def audit():
    fixtures = copy.deepcopy(old_cases())
    ref = {'id': 'new-grounding-fixture', 'prompt': 'NOTEQA is coordinated by PersonQA for Monday. Approval is pending.',
           'task': 'rewriting', 'required': ['NOTEQA', 'PersonQA', 'Monday']}
    for index, (answer, expected) in enumerate((
        ('NOTEQA is coordinated by PersonQA for Monday. Approval is pending.', True),
        ('noteqa is coordinated by PersonQA for Monday. Approval is pending.', False),
        ('NOTEQA is coordinated by PersonQA for monday. Approval is pending.', False),
        ('NOTEQA is coordinated by PersonQA for Monday. Approval is not pending.', False),
        ("NOTEQA is coordinated by PersonQA for Monday. Approval isn't pending.", False),
        ('NOTEQA is coordinated by PersonQA for Monday. Approval is no longer pending.', False),
        ('NOTEQA is coordinated by PersonQA for Monday. Approval is pending, not confirmed.', True),
        ('NOTEQA is coordinated by PersonQB for Tuesday. Approval is pending.', False),
        ('NOTEQA is coordinated by PersonQA for Monday. Approval is not still pending.', False),
    )):
        fixtures.append({'id': f'r108-extra-{index}', 'row': ref, 'answer': answer, 'expected_pass': expected,
                         'basis': 'explicit prospective literal/polarity obligation'})
    result = [{**f, 'screen': screen(f['row'], f['answer'])} for f in fixtures]
    return {'cases': result, 'mismatches': [f['id'] for f in result if f['screen']['mechanical_pass'] != f['expected_pass']],
            'scope': 'Author-specified syntactic fixtures, not human fluency or general semantics'}


def render(row, arm, tok):
    if arm == 'control':
        return prior.render(row, arm, 'native', tok)
    if arm not in ARMS:
        raise ValueError('Unregistered arm')
    return prior.base.model_prompt(row, 'distillation')


def validate(observed, population, arm, tok):
    if len(observed) != len(population):
        raise RuntimeError('Incomplete raw population')
    size = len(tok)
    for actual, ref in zip(observed, population):
        prompt = render(ref, arm, tok)
        ids = tok.encode(prompt, add_special_tokens=False)
        tokens = actual['token_ids']
        if (any(actual[k] != ref[k] for k in ('id', 'task', 'pair', 'panel')) or actual['arm'] != arm
                or actual['prompt'] != prompt or actual['prompt_ids'] != ids or not tokens
                or len(tokens) > MAX_NEW or len(ids) + MAX_NEW > CONTEXT
                or any(type(v) is not int or not 0 <= v < size for v in tokens)):
            raise RuntimeError('Wrong raw identity, context or token matrix')
        stop = 'eos' if tokens[-1] == tok.eos_token_id else 'budget'
        if tok.eos_token_id in tokens[:-1] or actual['stop_reason'] != stop or (stop == 'budget' and len(tokens) != MAX_NEW):
            raise RuntimeError('Invalid termination')
        answer = tok.decode(tokens, skip_special_tokens=True).strip()
        if (actual['output'] != answer or actual['screen'] != screen(ref, answer)
                or actual['output_utf8_bytes'] != len(answer.encode()) or actual['output_characters'] != len(answer)
                or not math.isfinite(actual['wall_seconds']) or actual['wall_seconds'] <= 0):
            raise RuntimeError('Unrecomputable output, screen or accounting')
        route = ROUTES.index(ref['task'])
        expected = None if arm == 'control' else [{'cakes': [route], 'adapters': [[route]] * 6}] * len(tokens)
        if actual['execution_routes'] != expected:
            raise RuntimeError('Wrong physical routes')


def pair_rows(observed):
    if len(observed) != 96 or len({r['id'] for r in observed}) != 96:
        raise RuntimeError('Incomplete paired panel')
    result = []
    for a, b in zip(observed[::2], observed[1::2]):
        if a['panel'] != 'a' or b['panel'] != 'b' or a['pair'] != b['pair'] or a['task'] != b['task']:
            raise RuntimeError('Unpaired counterfactuals')
        result.append({'id': a['pair'], 'task': a['task'],
                       'screen': {'mechanical_pass': a['screen']['mechanical_pass'] and b['screen']['mechanical_pass'],
                                  'collapse': a['screen']['collapse'] or b['screen']['collapse']},
                       'stop_reason': 'budget' if 'budget' in (a['stop_reason'], b['stop_reason']) else 'eos'})
    if len({r['id'] for r in result}) != 48:
        raise RuntimeError('Duplicate pair identity')
    return result


def summarize(observed):
    paired = pair_rows(observed)
    panels = {s: [r for r in observed if r['panel'] == s] for s in ('a', 'b')}
    return {'panels': {s: prior.aggregate(v) for s, v in panels.items()}, 'both_pass': prior.aggregate(paired),
            'qualified_mechanical_only': all(prior.qualified(v) for v in (*panels.values(), paired)),
            'truncations': sum(r['stop_reason'] == 'budget' for r in observed),
            'input_tokens': sum(len(r['prompt_ids']) for r in observed),
            'output_tokens': sum(len(r['token_ids']) for r in observed),
            'output_utf8_bytes': sum(r['output_utf8_bytes'] for r in observed),
            'unique_prompt_utf8_bytes': sum(len(p.encode()) for p in {r['prompt'] for r in observed}),
            'unique_output_utf8_bytes': sum(len(p.encode()) for p in {r['output'] for r in observed})}


def contrast(first, second):
    import numpy as np
    pair_rows(first)
    pair_rows(second)
    if [r['id'] for r in first] != [r['id'] for r in second]:
        raise RuntimeError('Unpaired contrast')
    delta = np.array([int(a['screen']['mechanical_pass']) - int(b['screen']['mechanical_pass'])
                      for a, b in zip(first, second)], dtype=float).reshape(48, 2).mean(axis=1)
    rng = np.random.default_rng(SEED)
    draws = delta[rng.integers(0, 48, (10000, 48))].mean(axis=1)
    return {'pairs': 48, 'mean_difference': float(delta.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist()}


def binding():
    frozen = read(OUT / 'freeze.json')
    if not frozen.get('files') or frozen['source_revision'] != REVISION or frozen['source_model'] != SOURCE:
        raise RuntimeError('Missing or wrong frozen inputs')
    for name, digest in frozen['files'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f'Changed frozen input: {name}')
    if read(OUT / 'population.json') != catalog() or read(OUT / 'evaluator_audit.json') != audit() or audit()['mismatches']:
        raise RuntimeError('Changed input or evaluator contract')
    return frozen
