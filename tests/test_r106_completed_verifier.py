"""Post-run hostile checks against completed evidence; no model execution."""

import shutil

import pytest

from experiments.grammar_interface_r106 import core as c
from experiments.grammar_interface_r106 import run


@pytest.fixture(scope='module')
def validated_source():
    frozen = c.binding()
    actual = run.prior.inventory(run.prior.snapshot())
    assert actual == frozen['control_files']
    return actual


@pytest.fixture
def evidence(tmp_path, monkeypatch, validated_source):
    shutil.copytree(c.OUT, tmp_path/'evidence')
    monkeypatch.setattr(c, 'OUT', tmp_path/'evidence')
    # Payloads are checked once against their real bytes above, not downloaded
    # or hashed 20+ times. These cases mutate evidence, not source model files.
    monkeypatch.setattr(run.prior, 'inventory', lambda folder: validated_source)
    return c.OUT


def test_complete_recomputation(evidence):
    expected = c.read(evidence/'decision.json')
    assert run.verify() == expected
    assert expected['product_promotion'] == 'NONE'
    assert expected['qualified_free_generation_control'] == 'NOT_ESTABLISHED'


@pytest.mark.parametrize('mutation', ['missing_initial', 'missing_replay', 'missing_replay_directory', 'missing_receipt',
    'missing_decision', 'missing_operation', 'missing_launch', 'missing_execution', 'duplicate', 'drop', 'token',
    'logit_nan', 'swapped_pair', 'route', 'forged_gate', 'accounting', 'training', 'resource', 'kernel',
    'stale_catalog', 'stale_freeze', 'operation_status', 'extra_raw', 'replay_drift'])
def test_hostile_completed_evidence(evidence, mutation):
    directory = evidence/'qualification/control/initial'
    path, rp = directory/'raw.jsonl', directory/'receipt.json'
    receipt = c.read(rp)
    missing = {'missing_initial': path, 'missing_replay': evidence/'qualification/control/replay/raw.jsonl',
        'missing_replay_directory': evidence/'qualification/control/replay', 'missing_receipt': rp,
        'missing_decision': evidence/'decision.json', 'missing_operation': evidence/'operation-00.json',
        'missing_launch': evidence/'launch.json', 'missing_execution': evidence/'execution.json'}
    if mutation in missing:
        target = missing[mutation]
        target.rename(target.with_name(target.name+'.absent'))
    elif mutation in ('duplicate', 'drop', 'token', 'logit_nan', 'swapped_pair', 'route', 'kernel', 'replay_drift'):
        observed = c.rows(path)
        if mutation == 'duplicate':
            observed[1] = observed[0]
        elif mutation == 'drop':
            observed.pop()
        elif mutation == 'token':
            observed[0]['good']['target_ids'][0] = -1
        elif mutation == 'logit_nan':
            observed[0]['good']['target_logits'][0] = float('nan')
        elif mutation == 'swapped_pair':
            observed[0]['good'], observed[0]['bad'] = observed[0]['bad'], observed[0]['good']
        elif mutation == 'route':
            observed[0]['good']['route_trace'] = {'cakes': [0]}
        elif mutation == 'kernel':
            observed[0]['good']['cross_entropy_max_error'] = .01
        else:
            observed[0]['good']['target_logits'][0] -= .0001
        path.write_bytes(b''.join(c.canonical(r) for r in observed))
        receipt['raw_sha256'] = c.sha(path)
    elif mutation == 'forged_gate':
        target = evidence/'decision.json'
        value = c.read(target)
        value['product_promotion'] = 'FLUENT_ENGLISH'
        target.write_bytes(c.canonical(value))
    elif mutation in ('accounting', 'training', 'resource'):
        field = {'accounting': 'scored_tokens', 'training': 'optimizer_steps', 'resource': 'wall_seconds'}[mutation]
        receipt[field] = float('nan') if mutation == 'resource' else receipt[field]+1
    elif mutation in ('stale_catalog', 'stale_freeze', 'operation_status'):
        name = {'stale_catalog': 'qualification.json', 'stale_freeze': 'freeze.json', 'operation_status': 'operation-00.json'}[mutation]
        target = evidence/name
        value = c.read(target)
        if mutation == 'stale_catalog':
            value[0]['good'] += 'x'
        elif mutation == 'stale_freeze':
            value['files']['experiments/grammar_interface_r106/core.py'] = '0'*64
        else:
            value['returncode'] = 1
        target.write_bytes(c.canonical(value))
    else:
        extra = evidence/'unregistered/raw.jsonl'
        extra.parent.mkdir()
        extra.write_bytes(b'{}\n')
    if mutation != 'missing_receipt':
        rp.write_bytes(c.canonical(receipt))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        run.verify()
    if mutation == 'missing_decision':
        assert not (evidence/'decision.json').exists()


def test_missing_source_payload_rejected(evidence, monkeypatch):
    monkeypatch.setattr(run.prior, 'inventory', lambda folder: {})
    with pytest.raises(RuntimeError, match='control payload'):
        run.verify()
