"""Post-result descriptive sensitivity ceiling; never replaces the frozen score."""
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.trained_grounding_r109 import core as c
from experiments.trained_grounding_r109 import run


def audit():
    run.verify()
    before = c.sha(c.OUT/'decision.json')
    raw = c.rows(c.OUT/'control/initial/raw.jsonl')
    flagged = [r for r in raw if r['screen']['explicit_approval_contradiction']]
    liberal = copy.deepcopy(raw)
    for row in liberal:
        if row['screen']['explicit_approval_contradiction']:
            row['screen']['mechanical_pass'] = True
    ceiling = c.summarize(liberal)
    result = {'format': 'r109-post-result-scope-audit/1', 'decision_sha256': before,
        'raw_sha256': c.sha(c.OUT/'control/initial/raw.jsonl'),
        'scope': 'Exploratory descriptive upper bound granting every approval-flagged row a pass, not a repaired or promoted score; no model calls or external human labels',
        'approval_flagged_rows': [{k: r[k] for k in ('id', 'task', 'output', 'screen')} for r in flagged],
        'sentence_count_failure_ids': [r['id'] for r in raw if not r['screen']['requested_sentence_count']],
        'optimistic_bound_panels': ceiling['panels'], 'optimistic_bound_pairs': ceiling['both_pass'],
        'optimistic_bound_qualifies': ceiling['qualified_mechanical_only']}
    path = c.OUT/'evaluator_scope_audit.json'
    if not path.exists():
        c.write(path, result)
    if c.read(path) != result or c.sha(c.OUT/'decision.json') != before:
        raise RuntimeError('Stale scope audit or changed frozen decision')
    print(json.dumps({'flagged_rows': len(flagged), 'sentence_count_failures': len(result['sentence_count_failure_ids']),
                      'optimistic_panels': {k: v['passing'] for k, v in ceiling['panels'].items()},
                      'optimistic_pairs': ceiling['both_pass']['passing'],
                      'optimistic_qualification': ceiling['qualified_mechanical_only']}))


if __name__ == '__main__':
    audit()
