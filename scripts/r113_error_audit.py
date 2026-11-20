"""Post-result descriptive failure inventory; never rescores or promotes R113."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.count_feedback_r113 import run as r  # noqa: E402


def audit():
    target = ROOT / 'results/review_2026_11_20/r113_error_audit.json'
    if target.exists():
        raise RuntimeError('Preserve prior descriptive audit')
    decision = r.derive()
    if decision != r.read(r.OUT / 'live/decision.json'):
        raise RuntimeError('Stored decision is stale')
    raw = r.rows(r.OUT / 'live/initial/raw.jsonl')
    arms = {}
    for stage in r.d.STAGES:
        arm = [v for v in raw if v['stage'] == stage]
        limited = [v for v in arm if v['task'] in r.d.TASKS]
        arms[stage] = {
            'rows': len(arm), 'sentence_limited_rows': len(limited),
            'sentence_passes': sum(v['screen']['requested_sentence_count'] for v in limited),
            'sentence_failures': [v['id'] for v in limited if not v['screen']['requested_sentence_count']],
            'unresolved': [v['id'] for v in arm if v['screen']['approval_scope']['classification'] == r.parent.e.UNKNOWN],
            'forbidden_flags': [v['id'] for v in arm if v['screen']['approval_scope']['classification'] == r.parent.e.UNSAFE],
            'collapses': sum(v['screen']['collapse'] for v in arm),
            'truncations': sum(v['stop_reason'] == 'budget' for v in arm),
            'failures': [{'id': v['id'], 'task': v['task'], 'output': v['output'],
                          'sentence_count': r.d.sentence_count(v['decoded_output']), 'screen': v['screen']}
                         for v in arm if not v['screen']['mechanical_pass']],
        }
    old = r.rows(r.d.OUT / 'live/initial/raw.jsonl')
    if len(old) != len(raw):
        raise RuntimeError('Missing historical comparison rows')
    differing = [a['id'] + '/' + a['stage'] for a, b in zip(r.scientific(raw), r.scientific(old)) if a != b]
    value = {'format': 'r113-post-result-error-audit/1',
        'decision_sha256': r.sha(r.OUT / 'live/decision.json'),
        'raw_sha256': r.sha(r.OUT / 'live/initial/raw.jsonl'), 'arms': arms,
        'r112_initial_raw_sha256': r.sha(r.d.OUT / 'live/initial/raw.jsonl'),
        'cross_attempt_differing_scientific_rows': differing,
        'scope': 'Post-result descriptive inventory of frozen scores, not new inference, rescoring, independent data or qualification'}
    r.write(target, value)
    print(json.dumps({'cross_attempt_differing_rows': len(differing),
                      'sentence_passes': {k: v['sentence_passes'] for k, v in arms.items()},
                      'decision_sha256': value['decision_sha256']}), flush=True)


if __name__ == '__main__':
    audit()
