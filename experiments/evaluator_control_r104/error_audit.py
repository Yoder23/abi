"""Post-run descriptive failure inventory, not a repaired qualification score."""

from __future__ import annotations

import json

from . import core as c
from .run import verify_cell


def build():
    c.binding()
    records = verify_cell('control', 'qualification')
    boolean_keys = ('required_literal_coverage', 'surface_form', 'nonempty_bounded', 'requested_sentence_count')
    forbidden_keys = ('unknown_identifiers', 'extra_numbers', 'unsupported_days', 'collapse')
    rows = []
    for row in records:
        score = row['screen']
        other_failures = [k for k in boolean_keys if not score[k]] + [k for k in forbidden_keys if score[k]]
        rows.append({'id': row['id'], 'task': row['task'], 'other_failures': other_failures,
                     'approval_flags': score['approval_evidence'], 'legacy_pass': row['legacy_screen']['mechanical_pass'],
                     'registered_pass': score['mechanical_pass'], 'stop_reason': row['stop_reason']})
    # Even removing EVERY approval-contradiction flag cannot exceed this ceiling.
    # This is a sensitivity bound, never the registered qualification score.
    ceiling = {task: sum(not row['other_failures'] and row['stop_reason'] == 'eos'
                         for row in rows if row['task'] == task) for task in c.TASKS}
    result = {'format': 'r104-postrun-error-inventory/1', 'rows': rows,
              'all_approval_flags_removed_upper_bound': sum(ceiling.values()),
              'all_approval_flags_removed_task_upper_bounds': ceiling,
              'interpretation': 'Descriptive sensitivity ceiling, NOT a repaired score or qualification',
              'bindings': {name: c.sha(c.OUT / name) for name in ('freeze.json', 'qualification.json', 'decision.json',
                           'qualification/control/raw.jsonl', 'qualification/control/receipt.json')},
              'audit_source_sha256': c.sha(c.HERE / 'error_audit.py')}
    path = c.OUT / 'postrun_error_inventory.json'
    if path.exists():
        if c.read(path) != result:
            raise RuntimeError('Stale descriptive audit')
    else:
        c.write_once(path, result)
    print(json.dumps({'upper_bound_not_score': result['all_approval_flags_removed_upper_bound'],
                      'task_upper_bounds': ceiling, 'audit_sha256': c.sha(path)}))
    return result


if __name__ == '__main__':
    build()
