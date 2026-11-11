"""Deterministic BLiMP selection and evidence-only conditional likelihood scores."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from experiments.evaluator_control_r104 import core as old

ROOT, HOST = old.ROOT, old.HOST
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/grammar_interface_r106'
FAMILIES = ('anaphor_number_agreement', 'determiner_noun_agreement_1',
            'irregular_past_participle_verbs', 'regular_plural_subject_verb_agreement_1')
DATA_REVISION = '3e56b06fcabca9b30822fc66435fca6b1aa40bb1'
ARMS, INTERFACES = old.ARMS, old.INTERFACES
SEED, CONTEXT, ROUTE = 106001, 256, 6
CONTROL_PARAMETERS = 7615616512
REQUEST = 'Write one grammatical English sentence.'
sha, read, rows, write_once, append = old.sha, old.read, old.rows, old.write_once, old.append
canonical, checkpoint = old.base.canonical, old.checkpoint


def catalog(split):
    if split not in ('qualification', 'comparison'):
        raise ValueError('Unknown panel')
    result = []
    for family in FAMILIES:
        source = rows(OUT/'sources'/f'{family}.jsonl')
        if len(source) != 1000 or len({str(r['pairID']) for r in source}) != 1000:
            raise RuntimeError('Incomplete external population')
        if any(r['UID'] != family or r['simple_LM_method'] is not True for r in source):
            raise RuntimeError('Wrong external family/method')
        selected = sorted(source, key=lambda r: hashlib.sha256(f"r106:{SEED}:{family}:{r['pairID']}".encode()).hexdigest())
        for row in selected[:50] if split == 'qualification' else selected[50:150]:
            good, bad = row['sentence_good'], row['sentence_bad']
            if not isinstance(good, str) or not isinstance(bad, str) or not good or not bad or good == bad:
                raise RuntimeError('Invalid minimal pair')
            result.append({'id': f"{family}:{row['pairID']}", 'family': family, 'split': split,
                           'good': good, 'bad': bad})
    if len({(r['good'], r['bad']) for r in result}) != len(result):
        raise RuntimeError('Duplicate selected pair')
    return result


def prefix(arm, interface, tok):
    if arm not in ('control', *ARMS) or interface not in INTERFACES:
        raise ValueError('Unregistered cell')
    if interface == 'benchmark':
        return f'Task: rewriting\n{REQUEST}\n\nResponse:\n'
    if arm == 'control':
        return tok.apply_chat_template([{'role': 'system', 'content': 'Write English text as requested.'},
                                        {'role': 'user', 'content': REQUEST}], tokenize=False, add_generation_prompt=True)
    return REQUEST + '\n'


def encoding(text, prompt, tok):
    p, t = tok.encode(prompt, add_special_tokens=False), tok.encode(text, add_special_tokens=False)
    if not p or not t or len(p)+len(t) > CONTEXT or tok.decode(t) != text:
        raise RuntimeError('Invalid or over-budget untruncated encoding')
    return p, t


def score(item):
    ids, values, norms = item['target_ids'], item['target_logits'], item['log_normalizers']
    if not ids or not len(ids) == len(values) == len(norms):
        raise RuntimeError('Incomplete token scoring evidence')
    if any(type(i) is not int or i < 0 for i in ids):
        raise RuntimeError('Invalid token ID')
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in values+norms):
        raise RuntimeError('Nonfinite numeric evidence')
    if any(v > n + 1e-5 for v, n in zip(values, norms)):
        raise RuntimeError('Invalid log probability')
    return math.fsum(v-n for v, n in zip(values, norms))


def correct(row, mean=False):
    def value(side):
        item = row[side]
        return score(item) / (len(item['target_ids']) if mean else 1)
    return value('good') > value('bad')


def summary(observed):
    return {'pairs': len(observed), 'correct': sum(correct(r) for r in observed),
            'per_family': {f: {'pairs': sum(r['family'] == f for r in observed),
                              'correct': sum(correct(r) for r in observed if r['family'] == f)} for f in FAMILIES},
            'ties': sum(score(r['good']) == score(r['bad']) for r in observed),
            'mean_token_correct_descriptive': sum(correct(r, True) for r in observed),
            'shorter_token_preference_correct_descriptive': sum(len(r['good']['target_ids']) < len(r['bad']['target_ids']) for r in observed),
            'unequal_token_length_pairs': sum(len(r['good']['target_ids']) != len(r['bad']['target_ids']) for r in observed)}


def gate(observed, per_family):
    result = summary(observed)
    return (len({r['id'] for r in observed}) == result['pairs'] == 4*per_family
            and result['correct'] >= .95*result['pairs']
            and all(v['pairs'] == per_family and v['correct'] >= .9*per_family for v in result['per_family'].values()))


def paired(first, second):
    import numpy as np

    if not first or [(r['id'], r['family']) for r in first] != [(r['id'], r['family']) for r in second]:
        raise RuntimeError('Unpaired or missing observations')
    if len({r['id'] for r in first}) != len(first):
        raise RuntimeError('Duplicate observations')
    rng = np.random.default_rng(SEED)
    delta = np.array([int(correct(a))-int(correct(b)) for a, b in zip(first, second)])
    draws = np.zeros(10000)
    for family in FAMILIES:
        part = delta[[r['family'] == family for r in first]]
        if not len(part):
            raise RuntimeError('Missing family')
        draws += part[rng.integers(0, len(part), size=(10000, len(part)))].sum(axis=1)/len(delta)
    return {'pairs': len(delta), 'difference': float(delta.mean()),
            'ci95': np.quantile(draws, [.025, .975]).tolist(), 'ci97_5': np.quantile(draws, [.0125, .9875]).tolist()}


def validate(observed, population, arm, interfaces, tok):
    expected = [(r, i) for r in population for i in interfaces]
    if len(observed) != len(expected):
        raise RuntimeError('Incomplete pair matrix')
    for actual, (ref, interface) in zip(observed, expected):
        prompt = prefix(arm, interface, tok)
        if any(actual.get(k) != v for k, v in {'id': ref['id'], 'family': ref['family'], 'split': ref['split'],
                                              'arm': arm, 'interface': interface, 'prefix': prompt}.items()):
            raise RuntimeError('Wrong pair lineage')
        for side in ('good', 'bad'):
            p, t = encoding(ref[side], prompt, tok)
            item = actual[side]
            if item['text'] != ref[side] or item['prefix_ids'] != p or item['target_ids'] != t:
                raise RuntimeError('Wrong raw text/token binding')
            if item['utf8_bytes'] != len(ref[side].encode()):
                raise RuntimeError('Wrong byte accounting')
            if not math.isfinite(item['cross_entropy_max_error']) or not 0 <= item['cross_entropy_max_error'] <= 3e-5:
                raise RuntimeError('Scoring kernel discrepancy')
            expected_trace = None if arm == 'control' else {'cakes': [ROUTE], 'adapters': [[ROUTE]]*6}
            if item['route_trace'] != expected_trace:
                raise RuntimeError('Unexpected physical route')
            score(item)


def binding():
    frozen = read(OUT/'freeze.json')
    if not frozen.get('files') or not frozen.get('control_files'):
        raise RuntimeError('Missing input identity')
    for name, digest in frozen['files'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError(f'Missing or changed frozen input: {name}')
    return frozen
