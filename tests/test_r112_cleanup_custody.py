"""Offline checks of cleanup custody; no deletion or scientific execution."""
import copy
import json
from pathlib import Path

import pytest

from scripts import r112_cleanup_receipt as c


def packet():
    root = Path(__file__).resolve().parents[1] / 'results/review_2026_11_19'
    values = [c.read(root / n) for n in ('r112_cleanup_plan.json', 'r112_cleanup_complete.json')]
    values.append([json.loads(s) for s in (root / 'r112_cleanup_actions.jsonl').read_text().splitlines()])
    values.append(c.read(root / 'r112_remaining_duplicate_audit.json'))
    return values


def test_cleanup_packet():
    plan, complete, journal, audit = packet()
    result = c.verify_documents(plan, complete, journal, audit)
    assert result['removed_files'] == 61
    assert result['removed_logical_bytes'] == 5925099441
    assert result['remaining_audit_groups'] == 240


@pytest.mark.parametrize('mutation', [
    'missing-row', 'missing-journal', 'stale-plan', 'changed-hash',
    'retained-is-removed', 'wrong-bytes', 'wrong-count', 'reordered-journal',
    'changed-audit', 'malformed-audit',
])
def test_tampered_custody_rejected(mutation):
    plan, complete, journal, audit = copy.deepcopy(packet())
    if mutation == 'missing-row':
        plan['deletions'].pop()
    elif mutation == 'missing-journal':
        journal.pop()
    elif mutation == 'stale-plan':
        complete['plan_sha256'] = '0' * 64
    elif mutation == 'changed-hash':
        journal[1]['retained_sha256'] = '0' * 64
    elif mutation == 'retained-is-removed':
        plan['deletions'][0]['retain'] = plan['deletions'][1]['remove']
    elif mutation == 'wrong-bytes':
        complete['removed_logical_bytes'] += 1
    elif mutation == 'wrong-count':
        complete['removed_files'] -= 1
    elif mutation == 'reordered-journal':
        journal[:2], journal[2:4] = journal[2:4], journal[:2]
    elif mutation == 'changed-audit':
        audit['minimum_bytes'] = 0
    elif mutation == 'malformed-audit':
        audit['groups'][0]['count'] += 1
    with pytest.raises((RuntimeError, KeyError, ValueError)):
        c.verify_documents(plan, complete, journal, audit)
