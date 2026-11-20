"""Finite R114 local-scope grammar; no IDs, labels or historical rescoring."""
from __future__ import annotations

import re

from experiments.count_feedback_r112.execution import evaluator as old

SAFE, UNSAFE, UNKNOWN = old.SAFE, old.UNSAFE, old.UNKNOWN
VOCAB, BOUNDARY = old.VOCAB, old.BOUNDARY
AMBIGUOUS = re.compile(old.AMBIGUOUS.pattern + r'|\bdo\s+not\s+(?:fail|forget)\b', re.I)
AS_EVENT = re.compile(r'\bas\s+(?:formal\s+|final\s+)?approval\s+is\s+(?:confirmed|granted|approved)\b', re.I)
A = r'(?:(?:formal|final)\s+)?approval'
C = r'(?:(?:formal|final)\s+)?confirmation'
NEW_RULES = (
    (SAFE, 'negative-instruction', rf'^\s*(?:please\s+)?do\s+not\s+(?:imply|assume|suggest|indicate)\s+(?:that\s+)?{A}\s+(?:is|has\s+been)\s+{old.STATE}\s*$'),
    (SAFE, 'absent-confirmation', rf'\bno\s+{C}\s+(?:of\s+{A}\s+is\s+available|has\s+been\s+{old.STATE})\b'),
    (SAFE, 'awaited-confirmation', rf'\b(?:await|awaits|awaiting|waiting\s+(?:for|on))\s+(?:{A}|{C})\b'),
    (SAFE, 'complete-approval-prerequisite', rf'\b(?:if|when|once|until|before|after)\s+{A}\s+(?:is|has\s+been)\s+(?:received|{old.STATE})\b'),
    (SAFE, 'nonfinal-authorization', r'\bpending\s+(?:(?:formal|final)\s+)?(?:authorization|sign-off|status)\b'),
    (SAFE, 'nonfinal-approval-label', rf'\b(?:pending|provisional|unconfirmed)\s+{A}\s+status\b'),
    (SAFE, 'approval-process-evolution', rf'\b{A}\s+status\s+(?:evolves|develops|changes)\b'),
    (UNSAFE, 'affirmative-received', rf'\b{A}\s+(?:is|has\s+been)\s+received\b'),
)
RULES = old.RULES[:3] + NEW_RULES + old.RULES[3:]
COMPILED = [(label, reason, re.compile(pattern, re.I)) for label, reason, pattern in RULES]


def classify(answer):
    if not isinstance(answer, str):
        raise TypeError('Response must be text')
    intervals, start, evidence = [], 0, []
    for match in BOUNDARY.finditer(answer):
        intervals.append((start, match.start()))
        start = match.end()
    intervals.append((start, len(answer)))
    for start, end in intervals:
        clause = answer[start:end]
        relevant = list(VOCAB.finditer(clause))
        if not relevant:
            continue
        consumed = set()
        def record(a, b, label, reason):
            evidence.append({'start': start+a, 'end': start+b, 'text': clause[a:b],
                             'classification': label, 'reason': reason,
                             'clause_start': start, 'clause_end': end})
            consumed.update(range(a, b))
        ambiguous_as = any(not re.search(r'\bas\s+soon\s+$', clause[:m.start()], re.I)
                           for m in AS_EVENT.finditer(clause))
        if AMBIGUOUS.search(clause) or ambiguous_as:
            record(0, len(clause), UNKNOWN, 'ambiguous-or-metalinguistic-scope')
            continue
        for label, reason, pattern in COMPILED:
            for match in pattern.finditer(clause):
                if not any(i in consumed for i in range(match.start(), match.end())):
                    record(match.start(), match.end(), label, reason)
        for token in relevant:
            if not all(i in consumed for i in range(token.start(), token.end())):
                record(token.start(), token.end(), UNKNOWN, 'unrecognized-approval-construction')
    evidence.sort(key=lambda r: (r['start'], r['end'], r['reason']))
    labels = {v['classification'] for v in evidence}
    return {'version': 'r114-approval-scope/1',
            'classification': UNSAFE if UNSAFE in labels else UNKNOWN if UNKNOWN in labels else SAFE,
            'evidence': evidence}


def screen(row, answer):
    result = dict(old.screen(row, answer))
    scope = classify(answer)
    result['approval_scope'] = scope
    result['explicit_approval_contradiction'] = scope['classification'] == UNSAFE
    result['mechanical_pass'] = bool(
        all(result[k] for k in ('required_literal_coverage', 'surface_form', 'nonempty_bounded',
                               'requested_sentence_count', 'exact_required_literals'))
        and not any(result[k] for k in ('unknown_identifiers', 'extra_numbers', 'unsupported_days',
                                       'collapse', 'negated_pending_status'))
        and scope['classification'] == SAFE)
    return result
