"""Explicit full-packet hostile audit; missing evidence is an error, not a skip."""
import copy

import pytest

from experiments.trained_grounding_r108 import core as c
from experiments.trained_grounding_r108 import run


@pytest.fixture(scope='module')
def packet():
    decision = run.verify()
    return {'decision': decision, 'frozen': c.binding(),
            'tokenizers': {arm: run.tokenizer(arm) for arm in decision['cells']},
            'parameters': run.source_parameter_count()}


@pytest.mark.parametrize('mutation', [
    'source_package', 'missing_receipt', 'missing_replay', 'missing_rows', 'raw_hash', 'freeze_hash',
    'missing_hash', 'wrong_arm', 'wrong_repeat', 'wrong_calls', 'wrong_precision', 'parameters',
    'cost_nan', 'cost_zero', 'cost_missing', 'budget', 'optimizer', 'text', 'screen', 'route',
    'token', 'prompt', 'pair', 'stop', 'bytes', 'decision',
])
def test_completed_verifier_fails_closed(packet, monkeypatch, mutation):
    original_read, original_rows = c.read, c.rows
    monkeypatch.setattr(c, 'binding', lambda: packet['frozen'])
    monkeypatch.setattr(run, 'tokenizer', lambda arm: packet['tokenizers'][arm])
    monkeypatch.setattr(run, 'source_parameter_count', lambda: packet['parameters'])
    monkeypatch.setattr(run, 'inventory', lambda folder: {} if mutation == 'source_package' else packet['frozen']['source_files'])

    def read(path):
        value = copy.deepcopy(original_read(path))
        name = path.as_posix()
        if name.endswith('/decision.json') and mutation == 'decision':
            value['product_promotion'] = 'FORGED_PASS'
        if '/control/replay/receipt.json' in name and mutation == 'missing_replay':
            raise FileNotFoundError('Missing replay')
        if '/control/initial/receipt.json' not in name:
            return value
        changes = {'raw_hash': ('raw_sha256', '0'*64), 'freeze_hash': ('freeze_sha256', '0'*64),
                   'wrong_arm': ('arm', 'lora'), 'wrong_repeat': ('repeat', 'replay'),
                   'wrong_calls': ('calls', 95), 'wrong_precision': ('precision', 'float16'),
                   'parameters': ('parameters', value['parameters']+1), 'cost_nan': ('wall_seconds', float('nan')),
                   'cost_zero': ('cpu_process_seconds', 0), 'budget': ('wall_seconds', 1501),
                   'optimizer': ('optimizer_steps', 1)}
        if mutation == 'missing_receipt':
            raise FileNotFoundError('Missing receipt')
        if mutation in changes:
            key, replacement = changes[mutation]
            value[key] = replacement
        if mutation == 'missing_hash':
            del value['raw_sha256']
        if mutation == 'cost_missing':
            del value['sampled_peak_rss_bytes']
        return value

    def rows(path):
        value = copy.deepcopy(original_rows(path))
        if '/control/initial/raw.jsonl' not in path.as_posix():
            return value
        if mutation == 'missing_rows':
            return value[:-1]
        row = value[0]
        if mutation == 'text':
            row['output'] += ' changed'
        elif mutation == 'screen':
            row['screen']['mechanical_pass'] = not row['screen']['mechanical_pass']
        elif mutation == 'route':
            row['execution_routes'] = []
        elif mutation == 'token':
            row['token_ids'][0] = -1
        elif mutation == 'prompt':
            row['prompt_ids'] = []
        elif mutation == 'pair':
            row['pair'] = 'wrong'
        elif mutation == 'stop':
            row['stop_reason'] = 'wrong'
        elif mutation == 'bytes':
            row['output_utf8_bytes'] += 1
        return value

    monkeypatch.setattr(c, 'read', read)
    monkeypatch.setattr(c, 'rows', rows)
    with pytest.raises((RuntimeError, KeyError, FileNotFoundError)):
        run.verify()


def test_missing_recipient_payload_is_not_optional(packet, monkeypatch):
    original = c.sha
    target = c.checkpoint('lora')/'model.safetensors'
    def sha(path):
        if path.resolve() == target.resolve():
            raise FileNotFoundError('Missing retained control payload')
        return original(path)
    monkeypatch.setattr(c, 'sha', sha)
    with pytest.raises(FileNotFoundError):
        c.binding()


def test_missing_decision_does_not_regenerate_silently(packet, monkeypatch):
    original = c.read
    monkeypatch.setattr(run, 'derive', lambda: packet['decision'])
    def read(path):
        if path.name == 'decision.json':
            raise FileNotFoundError('Missing decision')
        return original(path)
    monkeypatch.setattr(c, 'read', read)
    with pytest.raises(FileNotFoundError):
        run.verify()
