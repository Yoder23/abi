"""Offline R113 integrity/hostile tests; synthetic evidence is never a result."""
import copy
import json

import pytest

from experiments.count_feedback_r113 import run as r


class Tokenizer:
    eos_token_id = 2

    def __len__(self):
        return 128

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages, ensure_ascii=False)

    def encode(self, text, **kwargs):
        return [len(text) % 100, 7]

    def decode(self, tokens, **kwargs):
        return ' Approval is pending. \n'


def synthetic_raw():
    tok, raw = Tokenizer(), []
    for ref in r.d.population():
        draft = None
        for stage in r.d.STAGES:
            prompt = r.d.render(ref, stage, tok, draft)
            row = r.score_record(ref, stage, prompt, tok.encode(prompt), [3, 2], tok, .1,
                                 {'free_bytes': 11 * 1024**3, 'evidence_bytes': 500}, draft)
            raw.append(row)
            if stage == 'draft':
                draft = row['decoded_output']
    return tok, raw


def test_scientific_functions_are_exact_parent_aliases():
    for name in ('score_record', 'validate_live', 'scientific', 'live_decision', 'unchanged_controls'):
        assert getattr(r, name) is getattr(r.parent, name)
    assert r.d is r.parent.d
    assert r.OUT != r.d.OUT
    assert r.d.contract()['quality_gates'] == r.d.previous.contract()['quality_gates']
    assert r.d.contract()['maximum_calls'] == 576
    assert r.d.contract()['minimum_free_bytes'] == 10 * 1024**3


def test_missing_freeze_and_changed_parent_manifest_reject(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        r.binding()
    path = tmp_path / 'manifest.json'
    r.write(path, {'files': {}})
    monkeypatch.setattr(r, 'MANIFEST', path)
    with pytest.raises(RuntimeError, match='manifest changed'):
        r.ancestry()


def test_freeze_refuses_existing_namespace(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'storage', lambda: {})
    with pytest.raises(RuntimeError, match='Preserve existing'):
        r.freeze()


def test_storage_floor_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    monkeypatch.setattr(r.shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 10 * 1024**3 - 1})())
    with pytest.raises(RuntimeError, match='Storage floor'):
        r.storage()


@pytest.fixture
def complete(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'binding', lambda: {})
    tok, raw = synthetic_raw()
    monkeypatch.setattr(r.runtime, 'tokenizer', lambda _: tok)
    monkeypatch.setattr(r.runtime, 'source_parameter_count', lambda: 100)
    r.write(tmp_path / 'freeze.json', {'synthetic': True})
    freeze_sha = r.sha(tmp_path / 'freeze.json')
    monkeypatch.setattr(r.subprocess, 'check_output', lambda *a, **k: (tmp_path / 'freeze.json').read_bytes())
    base = tmp_path / 'live'
    for pid, repeat in enumerate(('initial', 'replay'), 1):
        path = base / repeat / 'raw.jsonl'
        path.parent.mkdir(parents=True)
        with path.open('x', encoding='utf-8') as stream:
            for row in raw:
                r.append(stream, row)
        r.write(path.parent / 'receipt.json', {'pid': pid, 'repeat': repeat, 'calls': 288,
            'raw_sha256': r.sha(path), 'freeze_sha256': freeze_sha, 'wall_seconds': 40,
            'cpu_seconds': 10, 'sampled_peak_rss_bytes': 100, 'rss_interval_seconds': .05,
            'peak_cuda_allocated_bytes': 100, 'parameters': 100, 'precision': 'bfloat16',
            'device': 'synthetic', 'training_steps': 0, 'recipient_calls': 0, 'gpu_setting_changes': []})
        r.write(base / f'{repeat}_operation.json', {'pid': pid, 'returncode': 0, 'elapsed_seconds': 41,
            'timeout_seconds': 14400, 'raw_sha256': r.sha(path),
            'argv': ['python', '-u', '-m', r.MODULE, 'execute', '--repeat', repeat]})
        for when in ('before', 'after'):
            r.write(base / f'{repeat}_{when}.json', {'argv': ['nvidia-smi', '-q', '-d', 'PERFORMANCE,POWER,CLOCK'],
                'returncode': 0, 'stdout': 'synthetic read-only telemetry', 'settings_changed': [],
                'observed_unix': 1, 'storage': {'free_bytes': 11 * 1024**3, 'evidence_bytes': 500}})
    r.write(base / 'launch.json', {'freeze_sha256': freeze_sha, 'budget_seconds': 30600,
                                 'started_unix': 1, 'published_commit': 'a' * 40})
    r.write(base / 'complete.json', {'elapsed_seconds': 90})
    return base


def overwrite_test_fixture(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def test_complete_negative_fixture_recomputes_and_never_promotes(complete):
    value = r.derive()
    assert value['verdict'] == 'CONTROL_QUALIFICATION_FAILED'
    assert value['costs']['generation_calls'] == 576
    assert value['abi_acquisition'] == value['native_layercake'] == 'NOT_TESTED'
    assert value['full_moonshot'] == 'NOT_PROVEN'


@pytest.mark.parametrize('name', [
    'initial/raw.jsonl', 'replay/raw.jsonl', 'replay/receipt.json', 'replay_operation.json',
    'initial_before.json', 'replay_after.json', 'complete.json', 'launch.json',
])
def test_required_raw_and_receipts_cannot_be_missing(complete, name):
    (complete / name).unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        r.derive()


@pytest.mark.parametrize('name,key,value', [
    ('replay/receipt.json', 'pid', 1), ('replay/receipt.json', 'calls', 242),
    ('replay/receipt.json', 'raw_sha256', '0' * 64),
    ('replay/receipt.json', 'freeze_sha256', '0' * 64),
    ('replay/receipt.json', 'precision', 'float16'),
    ('replay/receipt.json', 'parameters', 99),
    ('replay/receipt.json', 'wall_seconds', 14401),
    ('replay/receipt.json', 'peak_cuda_allocated_bytes', 0),
    ('replay/receipt.json', 'training_steps', 1),
    ('replay/receipt.json', 'recipient_calls', 1),
    ('replay/receipt.json', 'gpu_setting_changes', ['changed']),
    ('replay_operation.json', 'returncode', 1),
    ('replay_operation.json', 'timeout_seconds', 14401),
    ('replay_operation.json', 'elapsed_seconds', 1),
    ('replay_operation.json', 'pid', 99),
    ('replay_operation.json', 'argv', ['python', 'old_worker.py']),
    ('launch.json', 'freeze_sha256', '0' * 64),
    ('launch.json', 'budget_seconds', 40000),
    ('complete.json', 'elapsed_seconds', 30601),
    ('complete.json', 'elapsed_seconds', 70),
    ('replay_after.json', 'returncode', 1),
    ('replay_after.json', 'stdout', ''),
    ('replay_after.json', 'settings_changed', ['power']),
    ('replay_after.json', 'storage', {'free_bytes': 1, 'evidence_bytes': 1}),
])
def test_hostile_receipt_changes_reject(complete, name, key, value):
    path = complete / name
    record = r.read(path)
    record[key] = value
    overwrite_test_fixture(path, record)
    with pytest.raises((RuntimeError, KeyError, IndexError)):
        r.derive()


def test_same_pid_replay_cannot_pass_even_with_rebound_operation(complete):
    for name in ('replay/receipt.json', 'replay_operation.json'):
        path = complete / name
        value = r.read(path)
        value['pid'] = 1
        overwrite_test_fixture(path, value)
    with pytest.raises(RuntimeError, match='fresh replay'):
        r.derive()


def test_stop_marker_always_rejects(complete):
    r.write(complete / 'stop.json', {'status': 'PASS'})
    with pytest.raises(RuntimeError, match='Interrupted'):
        r.derive()


@pytest.mark.parametrize('mutation', ['missing', 'reorder', 'output', 'score', 'prompt', 'tokens', 'storage'])
def test_raw_mutation_rejects_before_trusting_hashes(complete, mutation):
    path = complete / 'replay/raw.jsonl'
    raw = r.rows(path)
    if mutation == 'missing':
        raw.pop()
    elif mutation == 'reorder':
        raw.reverse()
    elif mutation == 'output':
        raw[0]['output'] = 'edited'
    elif mutation == 'score':
        raw[0]['screen']['mechanical_pass'] = True
    elif mutation == 'prompt':
        raw[0]['prompt'] = 'new prompt'
    elif mutation == 'tokens':
        raw[0]['token_ids'] = [True, 2]
    else:
        raw[0]['storage_before']['free_bytes'] = 1
    path.write_text(''.join(json.dumps(v) + '\n' for v in raw), encoding='utf-8')
    with pytest.raises(RuntimeError):
        r.derive()


def test_stale_decision_rejects(complete):
    value = r.derive()
    forged = copy.deepcopy(value)
    forged['verdict'] = 'PASS'
    r.write(complete / 'decision.json', forged)
    with pytest.raises(RuntimeError, match='Stale'):
        r.verify()


def test_new_evidence_write_is_once(tmp_path):
    path = tmp_path / 'receipt.json'
    r.write(path, {'value': 1})
    with pytest.raises((RuntimeError, FileExistsError)):
        r.write(path, {'value': 2})
