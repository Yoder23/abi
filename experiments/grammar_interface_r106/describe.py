"""Post-run descriptive inventory, never a replacement score or qualification."""

import json
import math

from . import core as c
from .run import verify


def describe():
    verify()
    observed = c.rows(c.OUT/'qualification/control/initial/raw.jsonl')
    failures = []
    for row in observed:
        if c.correct(row):
            continue
        failures.append({'id': row['id'], 'family': row['family'], 'good': row['good']['text'],
            'bad': row['bad']['text'], 'good_token_count': len(row['good']['target_ids']),
            'bad_token_count': len(row['bad']['target_ids']), 'sum_margin': c.score(row['good'])-c.score(row['bad']),
            'mean_margin': c.score(row['good'])/len(row['good']['target_ids'])-c.score(row['bad'])/len(row['bad']['target_ids'])})
    strata = {}
    for equal in (True, False):
        rows = [r for r in observed if (len(r['good']['target_ids']) == len(r['bad']['target_ids'])) == equal]
        strata['equal_tokens' if equal else 'unequal_tokens'] = {'pairs': len(rows), 'correct': sum(c.correct(r) for r in rows)}
    receipts = {stage: c.read(c.OUT/f'qualification/control/{stage}/receipt.json') for stage in ('initial', 'replay')}
    assert all(math.isfinite(f['sum_margin']) for f in failures)
    result = {'format': 'r106-descriptive/1', 'failures': failures, 'token_length_strata': strata,
        'source': c.old.SOURCE, 'source_revision': c.old.REVISION,
        'resources': receipts, 'execution': c.read(c.OUT/'execution.json'),
        'bindings': {name: c.sha(c.OUT/name) for name in ('freeze.json', 'decision.json', 'qualification.json',
            'qualification/control/initial/raw.jsonl', 'qualification/control/replay/raw.jsonl')},
        'script_sha256': c.sha(c.HERE/'describe.py'),
        'scope': 'Post-run description only. No new gate, causal attribution, repaired score or English qualification.'}
    path = c.OUT/'descriptive.json'
    if path.exists():
        if c.read(path) != result:
            raise RuntimeError('Stale descriptive inventory')
    else:
        c.write_once(path, result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    describe()
