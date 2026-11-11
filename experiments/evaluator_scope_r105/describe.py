"""Additive descriptive contrast audit, not a new promotion criterion."""

from __future__ import annotations

import json

from . import core as c
from .complete import verify


def main():
    verify()
    population = c.read(c.OUT/'catalog.json')
    raw = c.rows(c.OUT/'initial/raw.jsonl')
    paired = {a['id']: (a, b) for a, b in zip(population, raw)}
    contrasts = {}
    for split in c.SPLITS:
        actual_same, gold_same = 0, 0
        for index in range(200):
            ref, base = paired[f'scone-no_negation-{index}']
            other, observed = paired[f'scone-{split}-{index}']
            gold_same += ref['gold_label'] == other['gold_label']
            actual_same += base['prediction'] == observed['prediction']
        contrasts[split] = {'paired_groups': 200, 'same_gold_as_no_negation': gold_same,
                            'same_prediction_as_no_negation': actual_same}
    errors = [{'id': a['id'], 'premise': a['premise'], 'hypothesis': a['hypothesis'],
               'expected_contradiction': a['gold_contradiction'], 'logits': b['logits'], 'prediction': b['prediction']}
              for a, b in zip(population, raw) if a['population'] == 'approval_scope'
              and a['gold_contradiction'] != (b['prediction'] == 'contradiction')]
    result = {'format': 'r105-descriptive/1', 'contrasts': contrasts, 'approval_errors': errors,
              'bindings': {s: c.sha(c.OUT/s) for s in ('catalog.json', 'initial/raw.jsonl', 'decision.json', 'completion.json')},
              'source_sha256': c.sha(c.HERE/'describe.py'), 'new_scientific_claim': 'NONE'}
    path = c.OUT/'descriptive.json'
    if path.exists():
        if c.read(path) != result:
            raise RuntimeError('Stale descriptive audit')
    else:
        c.write_once(path, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
