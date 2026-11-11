"""Scoring and integrity mechanics; no English-quality assertions."""

import copy
import math

import pytest

from experiments.grammar_interface_r106 import core as c
from experiments.grammar_interface_r106 import run


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return 'system: '+messages[0]['content']+'\nuser: '+messages[1]['content']+'\nassistant: '


def item(tokens, probability):
    return {'target_ids': tokens, 'target_logits': [math.log(probability)]*len(tokens),
            'log_normalizers': [0.0]*len(tokens)}


def fixture():
    tok = Tokenizer()
    ref = {'id': 'fixture', 'family': c.FAMILIES[0], 'split': 'qualification', 'good': 'They run.', 'bad': 'They runs.'}
    prompt = c.prefix('control', 'native', tok)
    observed = {k: ref[k] for k in ('id', 'family', 'split')}
    observed.update(arm='control', interface='native', prefix=prompt)
    for side, probability in (('good', .5), ('bad', .4)):
        p, t = c.encoding(ref[side], prompt, tok)
        observed[side] = {**item(t, probability), 'text': ref[side], 'prefix_ids': p, 'utf8_bytes': len(t),
                          'route_trace': None, 'cross_entropy_max_error': 0.0}
    return ref, observed, tok


def test_probability_product_not_mean_or_first_token_only():
    # .8*.8=.64 > .7*.7=.49, without using a model or our normalizer.
    assert math.exp(c.score(item([1, 2], .8))) == pytest.approx(.64)
    assert c.correct({'good': item([1, 2], .8), 'bad': item([1, 2], .7)})
    row = {'good': item([1, 2], .8), 'bad': item([1], .7)}
    assert not c.correct(row) and c.correct(row, mean=True)
    assert not c.correct({'good': item([1], .5), 'bad': item([2], .5)})


def test_causal_shift_and_independent_cross_entropy():
    import torch

    # Prefix has two tokens. Target [0,1] is predicted at positions [1,2].
    probabilities = torch.tensor([[.1, .9], [.8, .2], [.3, .7], [.99, .01]])
    picked, norms, error = run.token_evidence(probabilities.log()[1:-1], torch.tensor([0, 1]))
    result = {'target_ids': [0, 1], 'target_logits': picked, 'log_normalizers': norms}
    assert math.exp(c.score(result)) == pytest.approx(.56, abs=1e-7)
    wrong, wn, _ = run.token_evidence(probabilities.log()[2:], torch.tensor([0, 1]))
    assert math.exp(c.score({'target_ids': [0, 1], 'target_logits': wrong, 'log_normalizers': wn})) == pytest.approx(.003, abs=1e-7)
    assert error <= 3e-5


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_evidence_rejected(value):
    row = item([1], .5)
    row['target_logits'][0] = value
    with pytest.raises(RuntimeError):
        c.score(row)


@pytest.mark.parametrize('change', ['empty', 'missing_norm', 'positive_logprob', 'negative_id'])
def test_incomplete_numeric_evidence_rejected(change):
    row = item([1], .5)
    if change == 'empty':
        row['target_ids'] = []
    elif change == 'missing_norm':
        row['log_normalizers'] = []
    elif change == 'positive_logprob':
        row['target_logits'] = [1]
    else:
        row['target_ids'] = [-1]
    with pytest.raises(RuntimeError):
        c.score(row)


def test_interfaces_fixed_and_encoding_untruncated():
    tok = Tokenizer()
    assert c.prefix('parent', 'native', tok) == c.REQUEST+'\n'
    assert len({c.prefix(a, 'benchmark', tok) for a in ('control', *c.ARMS)}) == 1
    for args in (('new_model', 'native'), ('parent', 'best_of')):
        with pytest.raises(ValueError):
            c.prefix(*args, tok)
    with pytest.raises(RuntimeError):
        c.encoding('x'*257, 'p', tok)


def test_exact_raw_validation():
    ref, observed, tok = fixture()
    c.validate([observed], [ref], 'control', ('native',), tok)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'swap', 'prefix', 'token', 'text', 'family', 'interface', 'route', 'kernel', 'bytes'])
def test_raw_hostile_mutations(mutation):
    ref, observed, tok = fixture()
    changed = copy.deepcopy(observed)
    records = [changed]
    if mutation == 'missing':
        records = []
    elif mutation == 'duplicate':
        records *= 2
    elif mutation == 'swap':
        changed['good'], changed['bad'] = changed['bad'], changed['good']
    elif mutation == 'prefix':
        changed['good']['prefix_ids'][0] += 1
    elif mutation == 'token':
        changed['good']['target_ids'][0] += 1
    elif mutation == 'text':
        changed['good']['text'] += 'x'
    elif mutation in ('family', 'interface'):
        changed[mutation] = 'unregistered'
    elif mutation == 'route':
        changed['good']['route_trace'] = {'cakes': [0]}
    elif mutation == 'kernel':
        changed['good']['cross_entropy_max_error'] = .01
    else:
        changed['good']['utf8_bytes'] += 1
    with pytest.raises(RuntimeError):
        c.validate(records, [ref], 'control', ('native',), tok)


def panel():
    return [{'id': f'{f}:{i}', 'family': f, 'good': item([1], .9), 'bad': item([2], .1)}
            for f in c.FAMILIES for i in range(50)]


def test_new_narrow_gate_and_each_family_floor():
    values = panel()
    assert c.gate(values, 50)
    for i in (0, 1, 2, 3, 4, 50, 51, 52, 53, 54):
        values[i]['good'] = item([1], .05)
    assert c.gate(values, 50)
    values[100]['good'] = item([1], .05)
    assert not c.gate(values, 50)
    values = panel()
    for i in range(6):
        values[i]['good'] = item([1], .05)
    assert not c.gate(values, 50)
    assert not c.gate(panel()[:-1], 50)


def test_paired_bootstrap_keeps_pair_identity():
    values = panel()
    assert c.paired(values, values)['ci97_5'] == [0.0, 0.0]
    reverse = copy.deepcopy(values)
    for row in reverse:
        row['good'], row['bad'] = row['bad'], row['good']
    assert c.paired(values, reverse)['ci95'] == [1.0, 1.0]
    for invalid in (values[:-1], values[::-1], []):
        with pytest.raises(RuntimeError):
            c.paired(values, invalid)


def test_missing_replay_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'OUT', tmp_path)
    calls = []
    def fake_cell(arm, split, repeat):
        calls.append(repeat)
        if repeat == 'replay':
            raise FileNotFoundError('replay missing')
        return []
    monkeypatch.setattr(run, 'verify_cell', fake_cell)
    with pytest.raises(FileNotFoundError):
        run.verify_replays('control', 'qualification')
    assert calls == ['initial', 'replay']


def test_stage_budget_and_no_recipient_on_control_failure():
    assert run.stages(False) == [('control', 'qualification', 'initial'), ('control', 'qualification', 'replay')]
    assert len(run.stages(True)) == 12
