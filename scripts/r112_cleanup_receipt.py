"""Recheck exact duplicate retirement and record a new, non-scientific preflight."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from experiments.count_feedback_r112 import register as registration

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'results/review_2026_11_19'
ANCHORS = {
    'r112_cleanup_plan.json': '3d09bacb6059ceec83a479d74a3cc8bc575e6102557806ef44220c8928c76d23',
    'r112_cleanup_actions.jsonl': '316afe53ceabe2eae7894e3c022b336ba40a904028abcd718a40a60d5905bd45',
    'r112_cleanup_complete.json': '467a9d71b0cee021b7884a813042d0f96cb0dfdf38518bda67b69a7790f07c35',
    'r112_remaining_duplicate_audit.json': '2eea09c52caec9a54eaf8d85138eb73830e56695310d7fda9dd4c7cec24f6aca',
}
sha, read = registration.sha, registration.read


def verify_documents(plan, complete, journal, audit):
    """Check raw custody records, not physical free space in the past."""
    deletions = plan['deletions']
    targets = {r['remove'] for r in deletions}
    if (len(targets) != len(deletions) or len(deletions) != plan['selected_files']
            or sum(r['bytes'] for r in deletions) != plan['selected_logical_bytes']
            or complete['removed_files'] != len(deletions)
            or complete['removed_logical_bytes'] != plan['selected_logical_bytes']
            or complete['plan_sha256'] != ANCHORS['r112_cleanup_plan.json']
            or complete['journal_sha256'] != ANCHORS['r112_cleanup_actions.jsonl']
            or len(journal) != 2 * len(deletions)):
        raise RuntimeError('Missing/inconsistent cleanup custody records')
    for i, row in enumerate(deletions):
        before, after = journal[2*i:2*i+2]
        if (row['retain'] in targets or before != {
                'action': 'verified_before_remove', 'remove': row['remove'],
                'retain': row['retain'], 'sha256': row['sha256'], 'bytes': row['bytes']}
                or after['action'] != 'removed' or after['remove'] != row['remove']
                or after['retained_sha256'] != row['sha256']):
            raise RuntimeError('Changed/reordered cleanup action')
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{7}Z', after['utc']):
            raise RuntimeError('Malformed PowerShell UTC timestamp')
        # .NET round-trip timestamps have seven fractional digits; Python 3.10
        # accepts six and does not accept a terminal Z in fromisoformat.
        datetime.strptime(after['utc'][:26] + '+0000', '%Y-%m-%dT%H:%M:%S.%f%z')
    if audit['workspace'] != plan['workspace'] or audit['minimum_bytes'] != 1024**2:
        raise RuntimeError('Changed follow-up audit scope')
    for group in audit['groups']:
        if len(group['copies']) != group['count'] or group['count'] < 2 or group['bytes'] < 1024**2:
            raise RuntimeError('Malformed follow-up duplicate group')
    return {
        'removed_files': len(deletions), 'removed_logical_bytes': sum(r['bytes'] for r in deletions),
        'remaining_audit_groups': len(audit['groups']),
        'scope': 'Custody arithmetic only; no reconstruction of past physical disk observations',
    }


def verify_local():
    for name, digest in ANCHORS.items():
        if sha(REVIEW / name) != digest:
            raise RuntimeError(f'Missing/changed cleanup evidence: {name}')
    plan, complete, audit = (read(REVIEW / name) for name in (
        'r112_cleanup_plan.json', 'r112_cleanup_complete.json', 'r112_remaining_duplicate_audit.json'))
    journal = [json.loads(line) for line in (REVIEW / 'r112_cleanup_actions.jsonl').read_text().splitlines()]
    result = verify_documents(plan, complete, journal, audit)
    for script, digest in [('r112_duplicate_cleanup.ps1', plan['script_sha256']),
                           ('r112_remaining_duplicate_audit.ps1', audit['script_sha256'])]:
        if sha(ROOT / 'scripts' / script) != digest:
            raise RuntimeError('Changed cleanup/audit implementation')
    workspace = ROOT.parent.resolve()
    if Path(plan['workspace']).resolve() != workspace:
        raise RuntimeError('Local physical verification requires the original workspace')
    roots = [Path(p).resolve() for p in plan['target_roots']]
    for row in plan['deletions']:
        removed, retained = Path(row['remove']), Path(row['retain'])
        if (not removed.is_absolute() or not retained.is_absolute()
                or not removed.resolve().is_relative_to(workspace)
                or not retained.resolve().is_relative_to(workspace)
                or not any(removed.resolve().is_relative_to(p) for p in roots)
                or '.git' in removed.parts or removed.exists()
                or retained.stat().st_size != row['bytes'] or sha(retained) != row['sha256']):
            raise RuntimeError('Missing retained copy, changed bytes or incomplete retirement')
    registration.verify()
    return {**result, 'parent_custody': registration.audit_parent(),
            'evidence_sha256': ANCHORS, 'verifier_sha256': sha(Path(__file__))}


def observe():
    output = registration.d.OUT / 'preflight_002.json'
    if output.exists():
        raise RuntimeError('Preserve existing second preflight')
    custody = verify_local()
    resource = registration.resource_observation(shutil.disk_usage(ROOT).free)
    # This field concerns this observation, not the separately recorded cleanup.
    receipt = {
        **resource, 'format': 'r112-post-cleanup-preflight/1',
        'observed_utc': datetime.now(timezone.utc).isoformat(), 'observed_path': str(ROOT),
        'measurement': 'shutil.disk_usage(observed_path).free',
        'registration_sha256': sha(registration.d.OUT / 'registration.json'),
        'cleanup': custody,
        'scope': 'New storage/custody observation only; no implementation freeze or scientific execution',
    }
    registration.write_new(output, receipt)
    print(json.dumps(receipt))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'observe'))
    args = parser.parse_args()
    if args.command == 'observe':
        observe()
    else:
        print(json.dumps(verify_local()))
