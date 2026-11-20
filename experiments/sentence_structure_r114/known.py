"""Disclosed development annotations with R113 raw-row provenance."""
from experiments.count_feedback_r112.execution import run as parent
from experiments.count_feedback_r113 import run as r113

SAFE, UNKNOWN = 'NO_FORBIDDEN_ASSERTION', 'UNRESOLVED'
LABELS = {
    'All details remain as planned pending final sign-off': SAFE,
    'All details remain subject to change until approval is granted': SAFE,
    'Do not imply approval has been granted': SAFE,
    'Final confirmation depends on pending authorization': SAFE,
    'no confirmation has been granted': SAFE,
    'No confirmation of approval is available': SAFE,
    'provisional approval status': SAFE,
    'The task is currently in a pending status until formal approval is granted': SAFE,
    'Updates will follow as approval status evolves': SAFE,
    'We will proceed only after formal approval is received': SAFE,
    'We\u2019re still awaiting formal confirmation': SAFE,
    'We\u2019re waiting on final approval before moving forward': SAFE,
    'will share further updates as approval is confirmed': UNKNOWN,
}


def cases():
    found = {}
    for row in r113.rows(r113.OUT / 'live/initial/raw.jsonl'):
        for span in row['screen']['approval_scope']['evidence']:
            if span['classification'] == SAFE:
                continue
            text = row['output'][span['clause_start']:span['clause_end']].strip()
            origin = {'id': row['id'], 'stage': row['stage'],
                      'clause_start': span['clause_start'], 'clause_end': span['clause_end']}
            if origin not in found.setdefault(text, []):
                found[text].append(origin)
    if set(found) != set(LABELS):
        raise RuntimeError('R113 flagged clause inventory changed')
    return [{'id': f'known-r113-{i:03d}', 'kind': 'known_r113_clause', 'output': text,
             'expected': LABELS[text], 'origins': found[text]}
            for i, text in enumerate(sorted(found))]


def development():
    prior = parent.scope_inputs(parent.read(parent.d.OUT / 'challenge.json'))
    if len(prior) != 2398:
        raise RuntimeError('Missing historical scope obligations')
    return [{**v, 'kind': 'historical_' + v['kind']} for v in prior] + cases()
