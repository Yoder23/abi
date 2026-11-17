"""R109 changes resource policy, never the R108 scientific experiment."""
import ast
import inspect
from types import SimpleNamespace

import pytest

from experiments.trained_grounding_r108 import core as old
from experiments.trained_grounding_r108 import run as old_run
from experiments.trained_grounding_r109 import core as c
from experiments.trained_grounding_r109 import run


@pytest.mark.parametrize('name', ['catalog', 'screen', 'audit', 'render', 'validate',
                                 'pair_rows', 'summarize', 'contrast', 'training_provenance'])
def test_scientific_functions_are_identical_objects(name):
    assert getattr(c, name) is getattr(old, name)


def test_generation_body_changes_only_resources():
    original = inspect.getsource(old_run.execute)
    successor = inspect.getsource(run.execute)
    successor = successor.replace('            guard_storage()\n', '')
    successor = successor.replace('c.process_budget(arm)', '1500')
    assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(successor))


@pytest.mark.parametrize('name', ['snapshot', 'inventory', 'tokenizer', 'source_parameter_count', 'verify_arm'])
def test_fixed_runtime_helpers(name):
    assert ast.dump(ast.parse(inspect.getsource(getattr(run, name)))) == ast.dump(
        ast.parse(inspect.getsource(getattr(old_run, name))))


def test_receipt_gate_changes_only_time_budget():
    successor = inspect.getsource(run.verify_cell).replace('c.process_budget(arm)', '1500')
    assert ast.dump(ast.parse(successor)) == ast.dump(ast.parse(inspect.getsource(old_run.verify_cell)))


def test_budget_arithmetic_and_unknown_arm():
    assert c.process_budget('control') == 14400
    assert all(c.process_budget(a) == 3600 for a in c.ARMS)
    assert sum(2*c.process_budget(a) for a in ('control', *c.ARMS)) == c.MODEL_PROCESS_SECONDS == 57600
    assert c.TOTAL_SECONDS == 59400 and c.resource_policy()['maximum_generation_calls'] == 960
    assert c.MAX_NEW == old.MAX_NEW == 640 and c.SEED == old.SEED
    with pytest.raises(ValueError):
        c.process_budget('sweep')


@pytest.mark.parametrize('free,used,passes', [(10*1024**3, 1, True), (10*1024**3-1, 1, False),
                                           (11*1024**3, 1024**3+1, False)])
def test_storage_guard_fails_closed(monkeypatch, free, used, passes):
    item = SimpleNamespace(is_file=lambda: True, stat=lambda: SimpleNamespace(st_size=used))
    monkeypatch.setattr(c, 'OUT', SimpleNamespace(rglob=lambda _: [item]))
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda _: SimpleNamespace(free=free))
    if passes:
        assert run.guard_storage() == {'free_bytes': free, 'evidence_bytes': used}
    else:
        with pytest.raises(RuntimeError, match='Storage guard'):
            run.guard_storage()


def test_missing_or_empty_freeze_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        c.binding()
    c.write(tmp_path/'freeze.json', {'files': {}})
    with pytest.raises(RuntimeError):
        c.binding()


def test_missing_completion_cannot_be_qualified(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'OUT', tmp_path)
    with pytest.raises(FileNotFoundError):
        run.derive()


def test_old_evidence_write_paths_are_not_used():
    source = inspect.getsource(run)
    assert 'c.write(c.previous' not in source
    assert 'prepare_source' not in source
    assert 'local_files_only=True' in source
    assert 'timeout=min(c.process_budget(arm), remaining)' in source
    assert "'settings_changed': []" in source
    assert all(flag not in source for flag in ('--power-limit', '--lock-gpu-clocks', '--reset-gpu-clocks'))


@pytest.mark.parametrize('mutation', ['hash', 'calls', 'precision', 'parameters', 'budget', 'cpu_nan', 'rss_zero'])
def test_execution_receipt_hostile_edits(monkeypatch, mutation):
    receipt = dict(raw_sha256='digest', freeze_sha256='digest', arm='parent', repeat='initial',
                   calls=96, optimizer_steps=0, precision='float32', parameters=87710428,
                   wall_seconds=10, cpu_process_seconds=1, sampled_peak_rss_bytes=1,
                   peak_cuda_allocated_bytes=1)
    if mutation == 'hash':
        receipt['raw_sha256'] = 'wrong'
    elif mutation == 'calls':
        receipt['calls'] = 95
    elif mutation == 'precision':
        receipt['precision'] = 'float16'
    elif mutation == 'parameters':
        receipt['parameters'] -= 1
    elif mutation == 'budget':
        receipt['wall_seconds'] = c.RECIPIENT_SECONDS + 1
    elif mutation == 'cpu_nan':
        receipt['cpu_process_seconds'] = float('nan')
    else:
        receipt['sampled_peak_rss_bytes'] = 0
    monkeypatch.setattr(c, 'read', lambda _: receipt)
    monkeypatch.setattr(c, 'rows', lambda _: [{'wall_seconds': .01}]*96)
    monkeypatch.setattr(c, 'sha', lambda _: 'digest')
    with pytest.raises(RuntimeError):
        run.verify_cell('parent', 'initial')


def test_absent_replay_is_not_a_complete_arm(monkeypatch):
    def cell(arm, repeat):
        if repeat == 'replay':
            raise FileNotFoundError('raw replay missing')
        return [{'token_ids': [1], 'wall_seconds': 1}]
    monkeypatch.setattr(run, 'verify_cell', cell)
    with pytest.raises(FileNotFoundError):
        run.verify_arm('control')


def test_changed_replay_rejected(monkeypatch):
    monkeypatch.setattr(run, 'verify_cell', lambda arm, repeat: [
        {'token_ids': [1 if repeat == 'initial' else 2], 'wall_seconds': 1}])
    with pytest.raises(RuntimeError, match='replay differs'):
        run.verify_arm('control')
