"""Descriptive R115 failure inventory; never repairs, relabels or rescores outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.structural_decode_r115 import run as r  # noqa: E402


def audit():
    target = ROOT / 'results/review_2026_11_20/r115_error_audit.json'
    if target.exists():
        raise RuntimeError('Preserve prior descriptive audit')
    result = r.verify()
    stage = 'scope' if result['verdict'] == 'SCOPE_PREREQUISITE_FAILED' else 'live'
    folder = r.d.OUT / stage
    raw = r.rows(folder / 'initial/raw.jsonl')
    if stage == 'scope':
        details = {'failures': [v for v in raw if
            v['observed'].get('mechanical_pass', v['observed'].get('classification')) != v['expected']]}
    else:
        details = {}
        for arm in r.d.STAGES:
            values = [v for v in raw if v['stage'] == arm]
            limited = [v for v in values if v['task'] in r.d.TASKS]
            details[arm] = {
                'rows': len(values), 'sentence_limited_rows': len(limited),
                'sentence_passes': sum(v['screen']['requested_sentence_count'] for v in limited),
                'invalid_codec_rows': [v['id'] for v in values if not v['codec']['valid']],
                'unresolved': [v['id'] for v in values if v['screen']['approval_scope']['classification'] == r.e.UNKNOWN],
                'forbidden_flags': [v['id'] for v in values if v['screen']['approval_scope']['classification'] == r.e.UNSAFE],
                'failures': [{k: v[k] for k in ('id', 'task', 'decoded_output', 'output', 'codec', 'screen')}
                             for v in values if not v['screen']['mechanical_pass']],
            }
    value = {'format': 'r115-descriptive-error-audit/1', 'stage': stage, 'details': details,
             'decision_sha256': r.sha(folder / 'decision.json'), 'raw_sha256': r.sha(folder / 'initial/raw.jsonl'),
             'scope': 'Inventory of frozen scores, not new labels, corrected passes, human judgments or inference'}
    r.write(target, value)
    print(json.dumps({'stage': stage, 'verdict': result['verdict'],
                      'decision_sha256': value['decision_sha256']}), flush=True)


if __name__ == '__main__':
    audit()
