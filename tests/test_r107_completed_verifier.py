"""Post-execution hostile checks against the complete evidence verifier.

These run explicitly on the published packet; absence is not certification.
They do not mutate any saved scientific evidence.
"""

import copy

import pytest

from experiments.retention_intervention_r107 import core as c
from experiments.retention_intervention_r107 import complete as strict
from experiments.retention_intervention_r107 import run


@pytest.fixture(scope='module')
def complete():
    # Deliberately not skip-on-missing: this command requires the whole packet.
    strict.verify()
    actual = c.read(c.OUT/'decision.json')
    return actual, run.tokenizer(), c.binding()


@pytest.mark.parametrize('mutation', ['source', 'missing_receipt', 'missing_replay', 'missing_rows',
                                     'raw_hash', 'freeze_hash', 'missing_hash', 'state_replay',
                                     'cost_nan', 'cost_zero', 'cost_missing', 'budget', 'forwards',
                                     'optimizer', 'package', 'parameters', 'intervention', 'zero_hash',
                                     'untouched_hash', 'tensor_missing', 'row_nan', 'row_route',
                                     'raw_replay', 'generation_text', 'before_hash', 'both_state_hashes'])
def test_completed_verifier_rejects_hostile_mutations(complete, monkeypatch, mutation):
    _, tok, frozen = complete
    original_read, original_rows, original_sha = c.read, c.rows, c.sha
    monkeypatch.setattr(c, 'binding', lambda: frozen)
    monkeypatch.setattr(run, 'tokenizer', lambda: tok)
    monkeypatch.setattr(run, 'inventory', lambda folder: {} if mutation == 'source' else frozen['source_files'])

    def read(path):
        value = copy.deepcopy(original_read(path))
        name = str(path).replace('\\', '/')
        if not name.endswith('/receipt.json'):
            return value
        target = '/parent_neutral/initial/' in name
        if mutation == 'missing_receipt' and target:
            raise FileNotFoundError('hostile missing receipt')
        if mutation == 'missing_replay' and '/source/replay/' in name:
            raise FileNotFoundError('hostile missing replay')
        if mutation == 'state_replay' and '/source/replay/' in name:
            value['state_sha256'] = 'f'*64
        if mutation == 'raw_replay' and '/source/replay/' in name:
            value['raw_sha256'] = '0'*64
        if mutation == 'both_state_hashes' and '/parent_active/' in name:
            value['state_sha256'] = 'f'*64
        if target:
            if mutation == 'raw_hash':
                value['raw_sha256'] = '0'*64
            elif mutation == 'freeze_hash':
                value['freeze_sha256'] = '0'*64
            elif mutation == 'missing_hash':
                del value['raw_sha256']
            elif mutation == 'cost_nan':
                value['wall_seconds'] = float('nan')
            elif mutation == 'cost_zero':
                value['cpu_seconds'] = 0
            elif mutation == 'cost_missing':
                del value['peak_cuda_allocated_bytes']
            elif mutation == 'budget':
                value['wall_seconds'] = 1201
            elif mutation == 'forwards':
                value['forwards'] -= 1
            elif mutation == 'optimizer':
                value['optimizer_steps'] = 1
            elif mutation == 'package':
                value['new_capability_packages'] = 1
            elif mutation == 'parameters':
                value['parameters'] = -1
            elif mutation == 'intervention':
                value['intervention'] = None
            elif mutation == 'zero_hash':
                value['intervention']['tensors'][0]['after'] = 'f'*64
            elif mutation == 'untouched_hash':
                value['intervention']['untouched_sha256'] = ''
            elif mutation == 'tensor_missing':
                value['intervention']['tensors'].pop()
            elif mutation == 'before_hash':
                value['intervention']['tensors'][0]['before'] = 'f'*64
        if mutation == 'before_hash' and '/parent_neutral/replay/' in name:
            value['intervention']['tensors'][0]['before'] = 'f'*64
        return value

    def rows(path):
        value = copy.deepcopy(original_rows(path))
        name = str(path).replace('\\', '/')
        if '/source/initial/' in name:
            if mutation == 'missing_rows':
                return value[:-1]
            if mutation == 'row_nan':
                value[0]['target_logits'][0] = float('nan')
            if mutation == 'row_route':
                value[0]['route_trace'] = {'cakes': [0]}
            if mutation == 'generation_text':
                value[0]['generation']['text'] += 'fabricated'
        return value

    def sha(path):
        if mutation == 'raw_replay' and str(path).replace('\\', '/').endswith('/source/replay/raw.jsonl'):
            return '0'*64
        return original_sha(path)

    monkeypatch.setattr(c, 'read', read)
    monkeypatch.setattr(c, 'rows', rows)
    monkeypatch.setattr(c, 'sha', sha)
    with pytest.raises((RuntimeError, KeyError, FileNotFoundError, TypeError)):
        strict.verify()


def test_complete_claim_recomputed_not_promoted(complete):
    decision, _, _ = complete
    assert len(decision['summary']) == 9 and len(decision['contrasts']) == 4
    lower = decision['contrasts']['parent']['ci95'][0]
    expected = 'PARENT_RESIDUAL_INTERFERENCE_SUPPORTED' if lower > 0 else 'PARENT_RESIDUAL_INTERFERENCE_NOT_SUPPORTED'
    assert decision['outcome'] == expected
    assert 'no qualified English core' in decision['claim_boundary']


def test_immutable_size_cache_preserves_validation():
    class Tokenizer:
        eos_token_id = 255

        def __len__(self):
            return 256

        def decode(self, ids, skip_special_tokens=False):
            return ''.join(chr(i) for i in ids if not skip_special_tokens or i != 255)

    tok = Tokenizer()
    sized = strict.SizedTokenizer(tok)
    assert len(sized) == len(tok)
    ref = {'id': 'cache-fixture', 'ids': [65]*128, 'target_bytes': 96}
    trace = {'cakes': [1], 'adapters': [[1]]*6}
    row = {**ref, 'cell': 'parent_active', 'route': 1, 'route_trace': trace,
           'target_logits': [-1.0]*96, 'log_normalizers': [0.0]*96,
           'argmax_ids': [65]*96, 'cross_entropy_error': 0.0,
           'generation': {'ids': [65, 255], 'stop': 'eos', 'text': 'A', 'repeated4': 0.0,
                          'route_traces': [trace]*2}}
    c.validate([row], [ref], 'parent_active', tok)
    c.validate([row], [ref], 'parent_active', sized)
    broken = copy.deepcopy(row)
    broken['argmax_ids'][0] = len(tok)
    for variant in (tok, sized):
        with pytest.raises(RuntimeError):
            c.validate([broken], [ref], 'parent_active', variant)
