"""Synthetic mechanics and hostile evidence tests; no model-quality assertions."""

import copy
import math

import pytest

from experiments.retention_intervention_r107 import core as c
from experiments.retention_intervention_r107 import run


class Tokenizer:
    eos_token_id = 255

    def __len__(self):
        return 256

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(i) for i in ids if not skip_special_tokens or i != self.eos_token_id)


def fixture():
    ref = {'id': 'fixture', 'ids': [65]*128, 'target_bytes': 96}
    trace = {'cakes': [1], 'adapters': [[1]]*6}
    row = {**ref, 'cell': 'parent_active', 'route': 1, 'route_trace': trace,
           'target_logits': [-math.log(2)]*96, 'log_normalizers': [0.0]*96,
           'argmax_ids': [65]*96, 'cross_entropy_error': 0.0,
           'generation': {'ids': [65, 255], 'stop': 'eos', 'text': 'A', 'repeated4': 0.0,
                          'route_traces': [trace]*2}}
    return ref, row


def test_exact_scalar_accounting_and_weighting():
    _, row = fixture()
    c.validate([row], [fixture()[0]], 'parent_active', Tokenizer())
    assert c.nll(row) == pytest.approx(96*math.log(2))
    assert c.summary([row])['bpb'] == pytest.approx(1)
    second = copy.deepcopy(row)
    second['target_bytes'] = 192
    # Aggregate BPB is byte weighted, not (1+.5)/2.
    assert c.summary([row, second])['bpb'] == pytest.approx(2/3)


def test_causal_alignment_and_independent_kernel():
    import torch

    logits = torch.tensor([[.1, .9], [.8, .2], [.3, .7], [.99, .01]]).log()
    values, norms, argmax, error = run.token_evidence(logits[1:-1], torch.tensor([0, 1]))
    assert math.exp(sum(v-n for v, n in zip(values, norms))) == pytest.approx(.56)
    assert argmax == [0, 1] and error < 3e-5


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'id', 'cell', 'ids', 'bytes', 'route', 'trace',
                                     'short', 'nan', 'inf', 'positive', 'kernel', 'argmax', 'generation',
                                     'text', 'stop', 'eos_middle', 'gen_trace', 'repeat_stat'])
def test_hostile_raw_mutations(mutation):
    ref, row = fixture()
    observed = [row]
    if mutation == 'missing':
        observed = []
    elif mutation == 'duplicate':
        observed *= 2
    elif mutation in ('id', 'cell'):
        row[mutation] = 'wrong'
    elif mutation == 'ids':
        row['ids'] = [66]*128
    elif mutation == 'bytes':
        row['target_bytes'] = 97
    elif mutation == 'route':
        row['route'] = 6
    elif mutation == 'trace':
        row['route_trace'] = None
    elif mutation == 'short':
        row['target_logits'].pop()
    elif mutation in ('nan', 'inf'):
        row['target_logits'][0] = float(mutation)
    elif mutation == 'positive':
        row['target_logits'][0] = 1
    elif mutation == 'kernel':
        row['cross_entropy_error'] = .1
    elif mutation == 'argmax':
        row['argmax_ids'][0] = -1
    elif mutation == 'generation':
        row['generation']['ids'] = []
    elif mutation == 'text':
        row['generation']['text'] = 'B'
    elif mutation == 'stop':
        row['generation']['stop'] = 'budget'
    elif mutation == 'eos_middle':
        row['generation']['ids'] = [255, 65]
    elif mutation == 'gen_trace':
        row['generation']['route_traces'] = []
    else:
        row['generation']['repeated4'] = .9
    with pytest.raises((RuntimeError, KeyError)):
        c.validate(observed, [ref], 'parent_active', Tokenizer())


def test_fixed_selection_is_deterministic_not_score_based():
    texts = [f'Paragraph {i}: '+'word '*32 for i in range(100)]
    first = c.select(texts, Tokenizer())
    assert first == c.select(texts, Tokenizer()) and len(first) == 64
    assert len({r['id'] for r in first}) == 64
    assert [r['selection_key'] for r in first] == sorted(r['selection_key'] for r in first)
    with pytest.raises(RuntimeError):
        c.select(['too short']*100, Tokenizer())


def test_paired_direction_and_unpaired_rejection():
    _, row = fixture()
    better = copy.deepcopy(row)
    better['target_logits'] = [-math.log(2)/2]*96
    result = c.paired([row], [better])
    assert result['ci95'] == pytest.approx([.5, .5])
    assert c.paired([row], [row])['ci95'] == [0.0, 0.0]
    better['id'] = 'other'
    with pytest.raises(RuntimeError):
        c.paired([row], [better])


def test_no_silent_overwrite_or_missing_bindings(tmp_path, monkeypatch):
    c.write(tmp_path/'one.json', {'a': 1})
    with pytest.raises(FileExistsError):
        c.write(tmp_path/'one.json', {'a': 2})
    monkeypatch.setattr(c, 'OUT', tmp_path)
    c.write(tmp_path/'freeze.json', {'files': {}, 'source_files': {}})
    with pytest.raises(RuntimeError):
        c.binding()


def test_neutralization_only_changes_declared_residual_ups():
    import torch
    from torch import nn

    def cake():
        item = nn.Module()
        item.up = nn.Linear(2, 3, bias=False)
        item.down = nn.Linear(3, 2, bias=False)
        return item

    model = nn.Module()
    model.task_cakes = nn.ModuleList(cake() for _ in range(14))
    model.capability_layer_adapters = nn.ModuleList(nn.ModuleList(cake() for _ in range(14)) for _ in range(6))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    record = run.neutralize(model)
    assert len(record['tensors']) == 98
    for name, value in model.named_parameters():
        assert torch.equal(value, torch.zeros_like(value) if name.endswith('.up.weight') else before[name])
    with pytest.raises(RuntimeError):
        run.neutralize(nn.Linear(2, 3))


def test_repetition_is_descriptive():
    assert c.repeated4([1, 2, 3]) == 0
    assert c.repeated4([1]*8) == pytest.approx(.8)
