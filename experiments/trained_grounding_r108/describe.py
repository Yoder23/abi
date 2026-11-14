"""Additive descriptive error inventory; cannot change frozen R108 decisions."""
import json

from . import core as c
from . import run


def describe():
    decision = run.verify()
    cells = {}
    good_keys = ('required_literal_coverage', 'exact_required_literals', 'surface_form',
                 'nonempty_bounded', 'requested_sentence_count')
    bad_keys = ('unknown_identifiers', 'extra_numbers', 'unsupported_days', 'collapse',
                'explicit_approval_contradiction', 'negated_pending_status')
    for arm in decision['cells']:
        rows = c.rows(c.OUT/arm/'initial/raw.jsonl')
        failures = [r for r in rows if not r['screen']['mechanical_pass']]
        cells[arm] = {'raw_sha256': c.sha(c.OUT/arm/'initial/raw.jsonl'),
            'overlapping_failure_counts': {**{k: sum(not r['screen'][k] for r in rows) for k in good_keys},
                                           **{k: sum(bool(r['screen'][k]) for r in rows) for k in bad_keys}},
            'first_three_failures_in_registered_order': [
                {'id': r['id'], 'task': r['task'], 'output': r['output'], 'screen': r['screen']} for r in failures[:3]],
            'asymmetric_pairs': sum(a['screen']['mechanical_pass'] != b['screen']['mechanical_pass']
                                    for a, b in zip(rows[::2], rows[1::2]))}
    value = {'format': 'r108-descriptive-errors/1', 'decision_sha256': c.sha(c.OUT/'decision.json'),
             'cells': cells, 'scope': 'Overlapping mechanical error counts only; no new score, semantic label or promotion'}
    path = c.OUT/'error_inventory.json'
    if not path.exists():
        c.write(path, value)
    if c.read(path) != value:
        raise RuntimeError('Stale descriptive inventory')
    print(json.dumps(value), flush=True)
    return value


if __name__ == '__main__':
    describe()
