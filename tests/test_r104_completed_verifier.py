"""Additive completed-evidence adversarial audit; no frozen files are edited."""

import shutil

import pytest

from experiments.evaluator_control_r104 import core as c
from experiments.evaluator_control_r104 import run


@pytest.fixture
def copied_evidence(tmp_path, monkeypatch):
    original = c.OUT
    shutil.copytree(original, tmp_path / 'evidence')
    monkeypatch.setattr(c, 'OUT', tmp_path / 'evidence')
    # Full binding remains active against the original frozen dependencies.
    return c.OUT


def test_completed_decision_recomputes(copied_evidence):
    before = c.sha(c.OUT / 'decision.json')
    value = run.verify()
    assert value['verdict'] in ('CONTROL_QUALIFICATION_FAILED', 'BOUNDED_RECIPIENT_INTERFACE_DIAGNOSTIC_COMPLETED')
    assert value['evaluator_cases'] == 155 and not value['evaluator_mismatches']
    assert c.sha(c.OUT / 'decision.json') == before


@pytest.mark.parametrize('mutation', [
    'missing_raw', 'missing_receipt', 'wrong_raw_hash', 'missing_row', 'wrong_text',
    'trusted_screen', 'legacy_screen', 'stale_decision', 'interrupted',
    'nan_resource', 'missing_evaluator', 'forged_evaluator', 'changed_prompt',
])
def test_completed_verifier_rejects_mutations(copied_evidence, mutation):
    raw = c.OUT / 'qualification/control/raw.jsonl'
    receipt_path = raw.with_name('receipt.json')
    receipt = c.read(receipt_path)
    if mutation == 'missing_raw':
        raw.rename(raw.with_suffix('.absent'))
    elif mutation == 'missing_receipt':
        receipt_path.rename(receipt_path.with_suffix('.absent'))
    elif mutation == 'wrong_raw_hash':
        receipt['raw_sha256'] = '0' * 64
    elif mutation in ('missing_row', 'wrong_text', 'trusted_screen', 'legacy_screen', 'changed_prompt'):
        values = c.rows(raw)
        if mutation == 'missing_row':
            values.pop()
        elif mutation == 'wrong_text':
            values[0]['output'] += ' Changed.'
        elif mutation == 'changed_prompt':
            values[0]['prompt'] += ' Changed.'
        else:
            field = 'legacy_screen' if mutation == 'legacy_screen' else 'screen'
            values[0][field]['mechanical_pass'] = not values[0][field]['mechanical_pass']
        raw.write_bytes(b''.join(c.base.canonical(r) for r in values))
        receipt['raw_sha256'] = c.sha(raw)
        receipt['calls'] = len(values)
    elif mutation == 'stale_decision':
        path = c.OUT / 'decision.json'
        value = c.read(path)
        value['verdict'] = 'FORCED_PASS'
        path.write_bytes(c.base.canonical(value))
    elif mutation == 'interrupted':
        c.write_once(c.OUT / 'stop.json', {'reason': 'incomplete'})
    elif mutation == 'nan_resource':
        receipt['wall_seconds'] = float('nan')
    elif mutation == 'missing_evaluator':
        path = c.OUT / 'evaluator_audit.json'
        path.rename(path.with_suffix('.absent'))
    elif mutation == 'forged_evaluator':
        path = c.OUT / 'evaluator_audit.json'
        value = c.read(path)
        value['cases'][0]['expected_pass'] = False
        path.write_bytes(c.base.canonical(value))
    if mutation != 'missing_receipt':
        receipt_path.write_bytes(c.base.canonical(receipt))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        run.verify()


def test_missing_frozen_package_fails_closed(tmp_path, monkeypatch):
    original = c.read(c.OUT / 'freeze.json')
    original['files']['results/missing-r104-package.safetensors'] = '0' * 64
    monkeypatch.setattr(c, 'OUT', tmp_path)
    c.write_once(tmp_path / 'freeze.json', original)
    with pytest.raises(FileNotFoundError):
        c.binding()


def test_comparison_custody_or_prerequisite_stop(copied_evidence):
    decision = c.read(c.OUT / 'decision.json')
    if decision['verdict'] == 'CONTROL_QUALIFICATION_FAILED':
        (c.OUT / 'comparison').mkdir()
    else:
        path = c.OUT / 'comparison/parent/raw.jsonl'
        values = c.rows(path)
        values[0]['execution_routes'] = []
        path.write_bytes(b''.join(c.base.canonical(r) for r in values))
        receipt_path = path.with_name('receipt.json')
        receipt = c.read(receipt_path)
        receipt['raw_sha256'] = c.sha(path)
        receipt_path.write_bytes(c.base.canonical(receipt))
    with pytest.raises(RuntimeError):
        run.verify()


def test_observed_granted_conditional_limitation_is_not_hidden():
    population = {r['id']: r for r in c.catalog('qualification')}
    row = next(r for r in c.rows(c.OUT / 'qualification/control/raw.jsonl')
               if r['id'] == 'r104-qualification-045-email_drafting')
    assert 'once the approval is granted' in row['output']
    # This freezes a disclosed limitation, not a repaired passing score.
    observed = c.screen(population[row['id']], row['output'])
    assert observed == row['screen']
    assert observed['explicit_approval_contradiction']
    assert not observed['nonempty_bounded']


def test_observed_confirmed_underway_scope_limitation_is_not_hidden():
    row = next(r for r in c.rows(c.OUT / 'qualification/control/raw.jsonl')
               if r['id'] == 'r104-qualification-030-conversation')
    assert 'it is confirmed that the project is underway' in row['output']
    assert 'approval is yet to be confirmed' in row['output']
    assert c.approval_evidence(row['output']) == row['screen']['approval_evidence']
    assert row['screen']['explicit_approval_contradiction']
