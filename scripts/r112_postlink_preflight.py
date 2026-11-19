"""Verify actual hard-link custody, then record a new write-once storage preflight."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from experiments.count_feedback_r112 import register as r

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'results/review_2026_11_19'


def verify_pair(plan_name, complete_name, journal_name, plan_sha):
    if r.sha(REVIEW / plan_name) != plan_sha:
        raise RuntimeError('Changed hardlink plan')
    plan, complete = r.read(REVIEW / plan_name), r.read(REVIEW / complete_name)
    journal = [json.loads(x) for x in (REVIEW / journal_name).read_text().splitlines()]
    if (complete['plan_sha256'] != plan_sha or complete['completed_files'] != plan['files']
            or complete['journal_sha256'] != r.sha(REVIEW / journal_name)
            or len(journal) != 2 * plan['files'] or plan['files'] != len(plan['actions'])
            or complete['logical_duplicate_bytes'] != sum(x['bytes'] for x in plan['actions'])):
        raise RuntimeError('Incomplete/stale hardlink evidence')
    checked = set()
    for i, row in enumerate(plan['actions']):
        before, after = journal[2*i:2*i+2]
        target, retained = Path(row['target']), Path(row['retain'])
        if (before != {'action': 'before_link', **{k: row[k] for k in ('target', 'retain', 'sha256', 'bytes', 'backup')}}
                or after['action'] != 'linked' or after['target'] != row['target']
                or after['retain'] != row['retain'] or after['sha256'] != row['sha256']
                or not {row['target'], row['retain']}.issubset(after['physical_links'])
                or Path(row['backup']).exists()):
            raise RuntimeError('Changed action, missing link or leftover rollback file')
        for path in (target, retained):
            stat = path.stat()
            if (not path.resolve().is_relative_to(ROOT.parent.resolve()) or '.git' in path.parts
                    or stat.st_size != row['bytes'] or not stat.st_file_attributes & 1):
                raise RuntimeError('Missing, escaped, wrong-size or writable shared payload')
            key = stat.st_dev, stat.st_ino, row['sha256']
            if key not in checked:
                if r.sha(path) != row['sha256']:
                    raise RuntimeError('Changed shared payload bytes')
                checked.add(key)
        if not os.path.samefile(target, retained):
            raise RuntimeError('Paths do not share actual physical file identity')
    return {'files': plan['files'], 'logical_duplicate_bytes': complete['logical_duplicate_bytes'],
            'evidence': {n: r.sha(REVIEW / n) for n in (plan_name, complete_name, journal_name)}}


def main():
    output = r.d.OUT / 'preflight_003.json'
    if output.exists():
        raise RuntimeError('Preserve existing third preflight')
    plans = [
        verify_pair('r112_hardlink_plan.json', 'r112_hardlink_retry_complete.json',
                    'r112_hardlink_retry_actions.jsonl', '2bb5c4949eb98f1f590a7143a77b76ab27de175e6260023fe900d4c822b111fb'),
        verify_pair('r112_hardlink_aux_plan.json', 'r112_hardlink_aux_complete.json',
                    'r112_hardlink_aux_actions.jsonl', 'b44f3c66ae8264b9e91399fe327cf6366179c1cea3489142fe62f807cc5e0667'),
    ]
    r.verify()
    parent = r.audit_parent()
    receipt = {**r.resource_observation(shutil.disk_usage(ROOT).free),
               'format': 'r112-post-hardlink-preflight/1',
               'observed_utc': datetime.now(timezone.utc).isoformat(),
               'observed_path': str(ROOT), 'measurement': 'shutil.disk_usage(observed_path).free',
               'hardlink_plans': plans, 'parent_custody': parent,
               'registration_sha256': r.sha(r.d.OUT / 'registration.json'),
               'implementation_state': 'development_not_frozen',
               'scope': 'Local storage/custody observation, not scientific execution or promotion'}
    r.write_new(output, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
