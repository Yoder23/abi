"""R105 preregistration invariants; no downloads, weights, or model calls."""

import copy

import pytest

from experiments.evaluator_scope_r105 import core as c


class Tokenizer:
    def __len__(self):
        return 256

    def __call__(self, first, second, truncation):
        assert not truncation
        ids = [ord(s) for s in first + '|' + second]
        return {'input_ids': ids, 'attention_mask': [1]*len(ids)}


def observations():
    tok = Tokenizer()
    ref = {'id': 'example', 'premise': 'Pending.', 'hypothesis': 'Approved.'}
    raw = {'id': ref['id'], **tok(ref['premise'], ref['hypothesis'], False),
           'logits': [1.0, .2, -.5], 'prediction': 'contradiction'}
    return tok, ref, raw


def full_fixture():
    population = [{'id': f'{split}-{i}', 'population': 'external_scope', 'family': split,
                   'group': str(i), 'gold_label': 'entailment' if i % 2 else 'neutral'}
                  for split in c.SPLITS for i in range(200)] + c.approval_catalog()
    records = []
    for ref in population:
        label = ref.get('gold_label', 'contradiction' if ref.get('gold_contradiction') else 'neutral')
        logits = [1.0 if s == label else -1.0 for s in c.LABELS]
        records.append({'id': ref['id'], 'logits': logits})
    return records, population


def test_scope_catalog_frozen_balanced():
    cases = c.approval_catalog()
    assert cases == c.approval_catalog()
    assert len(cases) == len({r['id'] for r in cases}) == 64
    assert sum(r['gold_contradiction'] for r in cases) == 32
    assert all('pending' in r['premise'] for r in cases)
    assert any('Once final approval' in r['hypothesis'] for r in cases)
    assert any('confirmed that' in r['hypothesis'] for r in cases)


def test_valid_raw_logits():
    tok, ref, raw = observations()
    c.validate([raw], [ref], tok)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'id', 'ids', 'mask', 'nan', 'infinity', 'label', 'dimension', 'invalid_token'])
def test_raw_mutations_rejected(mutation):
    tok, ref, raw = observations()
    records = [raw]
    if mutation == 'missing':
        records = []
    elif mutation == 'duplicate':
        records *= 2
    elif mutation == 'id':
        raw['id'] = 'wrong'
    elif mutation == 'ids':
        raw['input_ids'] = []
    elif mutation == 'mask':
        raw['attention_mask'][0] = 0
    elif mutation in ('nan', 'infinity'):
        raw['logits'][0] = float('nan' if mutation == 'nan' else 'inf')
    elif mutation == 'label':
        raw['prediction'] = 'neutral'
    elif mutation == 'dimension':
        raw['logits'] = [1.0]
    else:
        raw['input_ids'][0] = -1
    with pytest.raises(RuntimeError):
        c.validate(records, [ref], tok)


def test_all_gates_recomputed_from_logits():
    records, population = full_fixture()
    value = c.aggregate(records, population)
    assert value['verdict'] == 'BOUNDED_EVALUATOR_CANDIDATE_PASSED'
    assert value['external_cluster_ci95'] == [1., 1.]
    assert value['external_all_six_correct_groups'] == 200
    assert value['full_moonshot'] == 'NOT_PROVEN'
    assert value['qualified_english_control'] == 'NOT_ESTABLISHED'
    records[-1]['logits'] = [-1, 1, -1]
    assert c.aggregate(records, population)['verdict'] == 'EVALUATOR_CANDIDATE_REJECTED'


def test_each_external_family_must_pass():
    records, population = full_fixture()
    for row in records[:11]:
        row['logits'] = [2, -1, -1]
    result = c.aggregate(records, population)
    assert result['external_correct'] > 1140
    assert not result['external_gate']


def test_missing_or_unpaired_populations_rejected():
    rows, population = full_fixture()
    for other in (rows[:-1], rows[::-1]):
        with pytest.raises(RuntimeError):
            c.aggregate(other, population)
    broken = copy.deepcopy(population)
    broken[0]['group'] = 'unknown'
    with pytest.raises(RuntimeError):
        c.aggregate(rows, broken)


def test_missing_inventory_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'OUT', tmp_path)
    c.write_once(tmp_path/'freeze.json', {'format': 'r105-freeze/1', 'files': {}, 'model_files': {}})
    with pytest.raises(RuntimeError):
        c.binding()
