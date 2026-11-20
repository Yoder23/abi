"""Disclosed R114 clause annotations; no historical generation rescoring."""
from experiments.sentence_structure_r114 import run as parent

SAFE, UNKNOWN = 'NO_FORBIDDEN_ASSERTION', 'UNRESOLVED'
LABELS = {
    'no changes are confirmed': SAFE,
    'no confirmation of approval has been issued': SAFE,
    'No promises on final approval': SAFE,
    'provisional arrangements pending approval': SAFE,
    'this remains subject to final approval': SAFE,
    'will proceed as approved': UNKNOWN,
}


def cases():
    found = {}
    for row in parent.rows(parent.d.OUT / 'live/initial/raw.jsonl'):
        for span in row['screen']['approval_scope']['evidence']:
            if span['classification'] == SAFE:
                continue
            text = row['output'][span['clause_start']:span['clause_end']].strip()
            origin = {'id': row['id'], 'stage': row['stage'],
                      'clause_start': span['clause_start'], 'clause_end': span['clause_end']}
            if origin not in found.setdefault(text, []):
                found[text].append(origin)
    if set(found) != set(LABELS):
        raise RuntimeError('R114 flagged clause inventory changed')
    return [{'id': f'known-r114-{i:03d}', 'kind': 'known_r114_clause', 'output': text,
             'expected': LABELS[text], 'origins': found[text]} for i, text in enumerate(sorted(found))]


def development():
    prior = parent.scope_inputs(parent.read(parent.d.OUT / 'challenge.json'))
    if len(prior) != 3211:
        raise RuntimeError('Missing R114 scope obligations')
    return [{**v, 'kind': 'historical_' + v['kind']} for v in prior] + cases()
