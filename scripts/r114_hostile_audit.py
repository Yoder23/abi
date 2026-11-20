"""R114 actual-row hostile audit; same agent, not independent scientific review."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.sentence_structure_r114 import run as r  # noqa: E402

OUT = ROOT / 'results/review_2026_11_20/r114_hostile.json'


def derive(result):
    if result['verdict'] == 'SCOPE_PREREQUISITE_FAILED':
        stage = 'scope'
    else:
        stage = 'live'
    folder = r.d.OUT / stage
    raw = r.rows(folder / 'initial/raw.jsonl')
    if stage == 'scope':
        expected = r.scope_inputs(r.read(r.d.OUT / 'challenge.json'))
        def validate(values):
            return r.scope_summary(values, expected)
        changes = [
            ('missing_row', lambda v: v.pop()),
            ('reordered_rows', lambda v: v.reverse()),
            ('identity', lambda v: v[0].__setitem__('id', 'forged')),
            ('text', lambda v: v[0].__setitem__('output', 'forged')),
            ('gold_label', lambda v: v[0].__setitem__('expected', 'forged')),
            ('observed', lambda v: v[0].__setitem__('observed', {'classification': 'PASS'})),
            ('kind', lambda v: v[0].__setitem__('kind', 'forged')),
        ]
    else:
        tok = r.old.runtime.tokenizer('control')
        def validate(values):
            return r.validate_live(values, tok)
        changes = [
            ('missing_row', lambda v: v.pop()),
            ('reordered_rows', lambda v: v.reverse()),
            ('identity', lambda v: v[0].__setitem__('id', 'forged')),
            ('prompt', lambda v: v[0].__setitem__('prompt', 'forged')),
            ('input_tokens', lambda v: v[0].__setitem__('prompt_ids', [])),
            ('boolean_token', lambda v: v[0].__setitem__('token_ids', [True, tok.eos_token_id])),
            ('missing_output_tokens', lambda v: v[0].__setitem__('token_ids', [])),
            ('raw_text', lambda v: v[0].__setitem__('decoded_output', 'forged')),
            ('payload', lambda v: v[0].__setitem__('output', 'forged')),
            ('codec', lambda v: v[0]['codec'].__setitem__('valid', False)),
            ('score', lambda v: v[0]['screen'].__setitem__('mechanical_pass', not v[0]['screen']['mechanical_pass'])),
            ('raw_bytes', lambda v: v[0].__setitem__('raw_output_utf8_bytes', -1)),
            ('disk_floor', lambda v: v[0]['storage_before'].__setitem__('free_bytes', 1)),
            ('nonfinite_cost', lambda v: v[0].__setitem__('wall_seconds', float('nan'))),
        ]
    validate(raw)
    cases = []
    for name, mutate in changes:
        value = copy.deepcopy(raw)
        mutate(value)
        try:
            validate(value)
        except (RuntimeError, ValueError, KeyError) as error:
            cases.append({'name': name, 'rejected': True, 'exception': repr(error)})
        else:
            cases.append({'name': name, 'rejected': False})
    return {'format': 'r114-hostile-audit/1', 'stage': stage, 'cases': cases,
            'raw_sha256': r.sha(folder / 'initial/raw.jsonl'),
            'decision_sha256': r.sha(folder / 'decision.json'),
            'scope': 'Same-agent in-memory mutations of actual raw evidence; not independent review'}


def audit():
    if OUT.exists():
        raise RuntimeError('Preserve prior hostile audit')
    value = derive(r.verify())
    r.write(OUT, value)
    if not all(v['rejected'] for v in value['cases']):
        raise RuntimeError('Hostile mutation accepted; preserve failed audit')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    audit()
