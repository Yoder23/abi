"""Post-run completed-evidence adversarial checks; originals are read-only."""

import shutil

import pytest

from experiments.evaluator_scope_r105 import complete as run
from experiments.evaluator_scope_r105 import core as c


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    shutil.copytree(c.OUT, tmp_path/'evidence')
    monkeypatch.setattr(c, 'OUT', tmp_path/'evidence')
    return c.OUT


def test_complete_raw_recomputation_and_exact_live_replay(evidence):
    before = c.sha(c.OUT/'decision.json')
    result = run.verify()
    assert result['scientific_verdict'] in ('EVALUATOR_CANDIDATE_REJECTED', 'BOUNDED_EVALUATOR_CANDIDATE_PASSED')
    assert c.sha(c.OUT/'decision.json') == before
    assert c.sha(c.OUT/'initial/raw.jsonl') == c.sha(c.OUT/'replay/raw.jsonl')


@pytest.mark.parametrize('mutation', ['missing_raw', 'missing_receipt', 'missing_replay', 'missing_replay_directory', 'raw_hash',
    'drop_row', 'duplicate', 'token', 'prediction', 'logit_nan', 'decision', 'catalog',
    'resource', 'batch', 'training', 'freeze', 'stop', 'replay_drift'])
def test_actual_verifier_rejects_hostile_evidence(evidence, mutation):
    path, rp = c.OUT/'initial/raw.jsonl', c.OUT/'initial/receipt.json'
    receipt = c.read(rp)
    if mutation == 'missing_replay_directory':
        (c.OUT/'replay').rename(c.OUT/'replay.absent')
    elif mutation in ('missing_raw', 'missing_receipt', 'missing_replay'):
        target = {'missing_raw': path, 'missing_receipt': rp, 'missing_replay': c.OUT/'replay/raw.jsonl'}[mutation]
        target.rename(target.with_suffix('.absent'))
    elif mutation == 'raw_hash':
        receipt['raw_sha256'] = '0'*64
    elif mutation in ('drop_row', 'duplicate', 'token', 'prediction', 'logit_nan', 'replay_drift'):
        observed = c.rows(path)
        if mutation == 'drop_row':
            observed.pop()
        elif mutation == 'duplicate':
            observed[1] = observed[0]
        elif mutation == 'token':
            observed[0]['input_ids'][0] = -1
        elif mutation == 'prediction':
            observed[0]['prediction'] = 'FORCED_CORRECT'
        elif mutation == 'logit_nan':
            observed[0]['logits'][0] = float('nan')
        else:
            observed[0]['logits'][0] += .000001
        path.write_bytes(b''.join(c.canonical(r) for r in observed))
        receipt['raw_sha256'] = c.sha(path)
    elif mutation == 'decision':
        dp = c.OUT/'decision.json'
        value = c.read(dp)
        value['verdict'] = 'FORCED_PASS'
        dp.write_bytes(c.canonical(value))
    elif mutation == 'catalog':
        cp = c.OUT/'catalog.json'
        value = c.read(cp)
        value[0]['gold_label'] = 'contradiction'
        cp.write_bytes(c.canonical(value))
    elif mutation == 'resource':
        receipt['wall_seconds'] = float('nan')
    elif mutation == 'batch':
        receipt['batch_receipts'].pop()
    elif mutation == 'training':
        receipt['optimizer_steps'] = 1
    elif mutation == 'freeze':
        receipt['freeze_sha256'] = '0'*64
    elif mutation == 'stop':
        c.write_once(c.OUT/'stop.json', {'reason': 'interrupted'})
    if mutation != 'missing_receipt':
        rp.write_bytes(c.canonical(receipt))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        run.verify()


def test_missing_model_payload_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'MODEL', tmp_path)
    with pytest.raises(FileNotFoundError):
        c.binding()


@pytest.mark.parametrize('name', ['decision.json', 'operation-0.json', 'operation-1.json', 'launch.json'])
def test_missing_required_completion_record_is_not_recreated(evidence, name):
    path = c.OUT/name
    path.rename(path.with_suffix('.absent'))
    with pytest.raises(FileNotFoundError):
        run.verify()
    assert not path.exists()


def test_frozen_stage_verifier_is_not_a_full_completion_attestation(evidence):
    from experiments.evaluator_scope_r105 import run as stage_run

    (c.OUT/'replay').rename(c.OUT/'replay.absent')
    # Preserve the limitation explicitly; do not silently edit frozen code.
    assert stage_run.verify()['verdict'] == c.read(c.OUT/'decision.json')['verdict']
    with pytest.raises(FileNotFoundError):
        run.verify()
