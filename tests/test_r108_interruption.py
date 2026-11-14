"""Explicit incomplete-packet verification, never a scientific pass or CI skip."""
import copy

import pytest

from experiments.trained_grounding_r108 import core as c
from experiments.trained_grounding_r108 import interruption as observer
from experiments.trained_grounding_r108 import run


@pytest.fixture(scope='module')
def packet():
    value = observer.verify()
    return {'value': value, 'frozen': c.binding(), 'tok': run.tokenizer('control'), 'history': c.training_provenance()}


@pytest.mark.parametrize('mutation', ['source', 'stop_missing', 'wrong_stop', 'short_timeout', 'budget',
    'empty_raw', 'text', 'screen', 'tokens', 'missing_telemetry', 'wrong_command', 'telemetry_error',
    'freeze_hash', 'missing_power', 'final_count', 'final_tokens', 'final_seconds', 'final_time',
    'tail', 'history', 'promoted_record'])
def test_interruption_requires_evidence_and_cannot_promote(packet, monkeypatch, mutation):
    original_read, original_rows = c.read, c.rows
    monkeypatch.setattr(c, 'binding', lambda: packet['frozen'])
    monkeypatch.setattr(c, 'training_provenance', lambda: packet['history'])
    monkeypatch.setattr(run, 'inventory', lambda path: {} if mutation == 'source' else packet['frozen']['source_files'])
    monkeypatch.setattr(run, 'tokenizer', lambda arm: packet['tok'])

    def read(path):
        value = copy.deepcopy(original_read(path))
        if path.name == 'stop.json':
            if mutation == 'stop_missing':
                raise FileNotFoundError('Missing stop')
            if mutation == 'wrong_stop':
                value['error'] = 'Unrelated error'
            if mutation == 'short_timeout':
                value['elapsed_seconds'] = 100
        if path.name == 'launch.json' and mutation == 'budget':
            value['budget_seconds'] = 9000
        if path.name == 'matched_training_provenance.json' and mutation == 'history':
            value['lora']['steps'] = 1
        if path.name == 'interruption.json' and mutation == 'promoted_record':
            value['qualification'] = 'PASS'
        if path.name == 'resource_final.json':
            if mutation == 'missing_telemetry':
                raise FileNotFoundError('Missing telemetry')
            if mutation == 'wrong_command':
                value['commands'][0]['argv'] = ['echo', 'fabricated']
            if mutation == 'telemetry_error':
                value['commands'][0]['returncode'] = 1
            if mutation == 'freeze_hash':
                value['freeze_sha256'] = '0'*64
            if mutation == 'missing_power':
                value['commands'][0]['stdout'] = ''
            if mutation == 'tail':
                value['incomplete_last_line'] = 'partial'
            fields = {'final_count': 'completed_rows', 'final_tokens': 'observed_output_tokens',
                      'final_seconds': 'observed_generation_seconds', 'final_time': 'launch_elapsed_seconds'}
            if mutation in fields:
                value[fields[mutation]] = 0
        return value

    def rows(path):
        value = copy.deepcopy(original_rows(path))
        if '/control/initial/raw.jsonl' in path.as_posix():
            if mutation == 'empty_raw':
                return []
            if mutation == 'text':
                value[0]['output'] += ' altered'
            if mutation == 'screen':
                value[0]['screen']['mechanical_pass'] = not value[0]['screen']['mechanical_pass']
            if mutation == 'tokens':
                value[0]['token_ids'][0] = -1
        return value
    monkeypatch.setattr(c, 'read', read)
    monkeypatch.setattr(c, 'rows', rows)
    with pytest.raises((RuntimeError, FileNotFoundError, KeyError)):
        observer.verify()


def test_complete_scientific_verifier_still_rejects(packet, monkeypatch):
    monkeypatch.setattr(c, 'binding', lambda: packet['frozen'])
    monkeypatch.setattr(c, 'training_provenance', lambda: packet['history'])
    with pytest.raises(RuntimeError, match='Interrupted run cannot be complete'):
        run.derive()


def test_interruption_missing_recipient_package_still_fails(packet, monkeypatch):
    original = c.sha
    def sha(path):
        if path.resolve() == (c.checkpoint('lora')/'model.safetensors').resolve():
            raise FileNotFoundError('Missing control checkpoint')
        return original(path)
    monkeypatch.setattr(c, 'sha', sha)
    with pytest.raises(FileNotFoundError):
        c.binding()
