"""Additive live-evidence validator mutations; not independent scientific review."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.count_feedback_r113 import run as r  # noqa: E402


def audit():
    output = ROOT / 'results/review_2026_11_20/r113_hostile.json'
    if output.exists():
        raise RuntimeError('Preserve the existing hostile audit')
    baseline = r.derive()
    if baseline != r.read(r.OUT / 'live/decision.json'):
        raise RuntimeError('Baseline decision not reproducible')
    tok = r.runtime.tokenizer('control')
    raw_path = r.OUT / 'live/initial/raw.jsonl'
    raw = r.rows(raw_path)
    cases = []
    mutations = [
        ('missing_row', lambda v: v.pop()),
        ('reordered_rows', lambda v: v.reverse()),
        ('wrong_identity', lambda v: v[0].__setitem__('id', 'forged')),
        ('changed_prompt', lambda v: v[0].__setitem__('prompt', 'forged')),
        ('changed_input_tokens', lambda v: v[0].__setitem__('prompt_ids', [0])),
        ('boolean_token', lambda v: v[0].__setitem__('token_ids', [True, tok.eos_token_id])),
        ('missing_output_tokens', lambda v: v[0].__setitem__('token_ids', [])),
        ('changed_output', lambda v: v[0].__setitem__('output', 'forged')),
        ('changed_decoded_output', lambda v: v[0].__setitem__('decoded_output', 'forged')),
        ('forged_quality_boolean', lambda v: v[0]['screen'].__setitem__('mechanical_pass', not v[0]['screen']['mechanical_pass'])),
        ('forged_counter', lambda v: v[1].__setitem__('draft_sentence_count', 999)),
        ('forged_feedback', lambda v: v[2].__setitem__('provided_sentence_count', 999)),
        ('disk_floor_violation', lambda v: v[0]['storage_before'].__setitem__('free_bytes', 1)),
        ('nonfinite_cost', lambda v: v[0].__setitem__('wall_seconds', float('nan'))),
    ]
    for name, mutate in mutations:
        changed = copy.deepcopy(raw)
        mutate(changed)
        try:
            r.validate_live(changed, tok)
        except (RuntimeError, ValueError, KeyError) as error:
            cases.append({'name': name, 'boundary': 'frozen raw validator', 'rejected': True, 'exception': repr(error)})
        else:
            cases.append({'name': name, 'boundary': 'frozen raw validator', 'rejected': False})
    value = {'format': 'r113-raw-hostile-audit/1', 'baseline_decision_sha256': r.sha(r.OUT / 'live/decision.json'),
             'raw_sha256': r.sha(raw_path), 'cases': cases,
             'rejected': sum(c['rejected'] for c in cases), 'attempted': len(cases),
             'scope': 'Same-agent targeted validator mutations of actual live rows; no model calls, not independent red-team or human review'}
    r.write(output, value)
    if value['rejected'] != value['attempted']:
        raise RuntimeError('Hostile mutation accepted; preserve failed audit')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    audit()
