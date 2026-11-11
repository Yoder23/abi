"""R105 input obligations and raw-logit recomputation. No model calls here."""

from __future__ import annotations

import csv
import io
import math
from collections import Counter
from pathlib import Path

from experiments.plan_english_r98.core import canonical as canonical
from experiments.plan_english_r98.core import read, sha
from experiments.plan_english_r98.core import rows as rows
from experiments.plan_english_r98.core import write_once as write_once

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/evaluator_scope_r105'
MODEL = ROOT / 'hf_models/evaluator_scope_r105'
SOURCE = 'cross-encoder/nli-deberta-v3-base'
REVISION = '6c749ce3425cd33b46d187e45b92bbf96ee12ec7'
DATA_REVISION = 'b02532a2f4185c6118a57a148455e0750592d8c8'
LABELS = ('contradiction', 'entailment', 'neutral')
SPLITS = ('no_negation', 'one_not_scoped', 'one_scoped', 'one_scoped_one_not_scoped', 'two_not_scoped', 'two_scoped')
MODEL_FILES = ('README.md', 'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'spm.model', 'special_tokens_map.json', 'added_tokens.json')
SEED = 105001


def external_catalog():
    result, keys = [], None
    for split in SPLITS:
        source = OUT / 'sources' / f'{split}.csv'
        records = list(csv.DictReader(io.StringIO(source.read_text(encoding='utf-8-sig'))))
        order = [r[''] for r in records]
        if len(records) != 200 or len(set(order)) != 200 or (keys is not None and order != keys):
            raise RuntimeError('Wrong external contrast-group coverage/order')
        keys = order
        suffix = '' if split == 'one_scoped' else '_edited'
        if Counter(r['gold_label' + suffix] for r in records) != {'entailment': 100, 'neutral': 100}:
            raise RuntimeError('Wrong external label coverage')
        for r in records:
            result.append({'id': f'scone-{split}-{r[""]}', 'family': split, 'group': r[''],
                           'population': 'external_scope', 'premise': r['sentence1' + suffix],
                           'hypothesis': r['sentence2' + suffix], 'gold_label': r['gold_label' + suffix],
                           'lexical_pair': [r['sentence1_lex'], r['sentence2_lex']]})
    for index in range(200):
        if len({tuple(result[j*200+index]['lexical_pair']) for j in range(6)}) != 1:
            raise RuntimeError('Mismatched external lexical contrast group')
    return result


def approval_catalog():
    # Authored before outputs; explicit Boolean contradiction obligations, not
    # human English-quality ratings. Separate from externally authored ScoNe.
    safe = (
        'Final approval for {p} is pending.',
        'If final approval for {p} is granted, we can proceed.',
        'Once final approval for {p} is granted, we can proceed.',
        'As soon as final approval for {p} is confirmed, we can proceed.',
        'It is confirmed that {p} is underway, but final approval is pending.',
        'The Monday meeting for {p} is confirmed, not its final approval.',
        '{p} has not been approved yet.',
        'Nobody has granted final approval for {p}.',
        'Final approval for {p} has not been decided.',
        'There is no final approval for {p} yet.',
        'Final approval for {p} might arrive later.',
        'Please wait until final approval for {p} is granted.',
        'Has final approval for {p} been granted?',
        'Someone asked whether final approval for {p} was granted.',
        'The claim that {p} has final approval is false.',
        'It is not true that {p} has final approval.',
    )
    conflict = (
        'Final approval for {p} is granted.',
        '{p} has been approved.',
        'The committee has granted final approval for {p}.',
        'Final approval for {p} has already arrived.',
        'The final approval of {p} is confirmed.',
        '{p} now has final approval.',
        'There is final approval for {p}.',
        'It is not true that final approval for {p} is pending.',
        'Final approval for {p} is no longer pending; it has been granted.',
        'Final approval for {p}: yes, granted.',
        'If the Monday meeting is confirmed, we can meet. Final approval for {p} is granted.',
        'Once the Monday meeting is confirmed, we can meet. {p} has final approval.',
        'Final approval for {p} is pending, but it has already been granted.',
        '{p} is underway and its final approval has been granted.',
        'The Monday meeting is confirmed and so is final approval for {p}.',
        'It is false that final approval for {p} has not been granted.',
    )
    result = []
    for entity in ('Project Cedar', 'Project Willow'):
        premise = f'{entity} is underway. Final approval for {entity} is still pending and has not been granted. The Monday meeting for {entity} is confirmed.'
        for category, templates, label in (('safe', safe, False), ('conflict', conflict, True)):
            for index, template in enumerate(templates):
                result.append({'id': f'approval-{entity.split()[-1]}-{category}-{index:02d}',
                               'family': category, 'group': f'{category}-{index:02d}',
                               'population': 'approval_scope', 'premise': premise,
                               'hypothesis': template.format(p=entity), 'gold_contradiction': label})
    return result


def inventory(folder, names):
    return {name: {'sha256': sha(folder/name), 'bytes': (folder/name).stat().st_size} for name in names}


def binding():
    frozen = read(OUT/'freeze.json')
    if frozen.get('format') != 'r105-freeze/1' or not frozen.get('files') or set(frozen.get('model_files', {})) != set(MODEL_FILES):
        raise RuntimeError('Missing frozen dependencies')
    for name, expected in frozen['files'].items():
        if sha(ROOT/name) != expected:
            raise RuntimeError(f'Changed dependency: {name}')
    if inventory(MODEL, MODEL_FILES) != frozen['model_files']:
        raise RuntimeError('Missing or changed evaluator payload')
    return frozen


def validate(observed, population, tokenizer):
    if len(observed) != len(population) or len({r['id'] for r in observed}) != len(observed):
        raise RuntimeError('Missing/duplicate raw observations')
    vocab_size = len(tokenizer)
    for row, case in zip(observed, population):
        features = tokenizer(case['premise'], case['hypothesis'], truncation=False)
        logits = row['logits']
        if (row['id'] != case['id'] or row['input_ids'] != features['input_ids']
                or row['attention_mask'] != features['attention_mask']
                or not row['input_ids'] or len(row['input_ids']) > 512
                or any(type(t) is not int or not 0 <= t < vocab_size for t in row['input_ids'])
                or len(logits) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in logits)):
            raise RuntimeError('Unrecomputable input or logits')
        label = LABELS[max(range(3), key=lambda j: logits[j])]
        if row['prediction'] != label:
            raise RuntimeError('Trusted/forged prediction')


def aggregate(observed, population):
    import numpy as np

    if len(observed) != len(population) or [r['id'] for r in observed] != [r['id'] for r in population]:
        raise RuntimeError('Unpaired raw evidence')
    external, approval = [], []
    for actual, ref in zip(observed, population):
        predicted = LABELS[max(range(3), key=lambda j: actual['logits'][j])]
        if ref['population'] == 'external_scope':
            external.append({'family': ref['family'], 'group': ref['group'], 'correct': predicted == ref['gold_label'],
                             'gold': ref['gold_label'], 'prediction': predicted})
        else:
            approval.append({'family': ref['family'], 'correct': (predicted == 'contradiction') == ref['gold_contradiction']})
    if len(external) != 1200 or len(approval) != 64:
        raise RuntimeError('Incomplete registered populations')
    groups = sorted({r['group'] for r in external})
    if len(groups) != 200 or any(sum(r['group'] == g for r in external) != 6 for g in groups):
        raise RuntimeError('Incomplete contrast clusters')
    means = np.array([sum(r['correct'] for r in external if r['group'] == g)/6 for g in groups])
    rng = np.random.default_rng(SEED)
    draws = means[rng.integers(0, 200, size=(10000, 200))].mean(axis=1)
    ci = np.quantile(draws, [.025, .975]).tolist()
    by_family = {f: sum(r['correct'] for r in external if r['family'] == f) for f in SPLITS}
    external_pass = sum(r['correct'] for r in external) >= 1140 and all(v >= 190 for v in by_family.values()) and ci[0] >= .90
    approval_pass = all(r['correct'] for r in approval)
    return {'format': 'r105-decision/1', 'external_correct': sum(r['correct'] for r in external), 'external_rows': 1200,
            'external_by_family_of_200': by_family, 'external_cluster_ci95': ci,
            'external_all_six_correct_groups': int(sum(v == 1 for v in means)),
            'external_confusion': {gold: {pred: sum(r['gold'] == gold and r['prediction'] == pred for r in external) for pred in LABELS} for gold in LABELS},
            'approval_correct': sum(r['correct'] for r in approval), 'approval_rows': 64,
            'approval_by_family_of_32': {f: sum(r['correct'] for r in approval if r['family'] == f) for f in ('safe', 'conflict')},
            'external_gate': external_pass, 'approval_gate': approval_pass,
            'verdict': 'BOUNDED_EVALUATOR_CANDIDATE_PASSED' if external_pass and approval_pass else 'EVALUATOR_CANDIDATE_REJECTED',
            'qualified_english_control': 'NOT_ESTABLISHED', 'recipient_comparison': 'NOT_RUN',
            'native_layercake_defect': 'NOT_ESTABLISHED', 'full_moonshot': 'NOT_PROVEN'}
