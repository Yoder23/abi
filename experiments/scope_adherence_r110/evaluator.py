"""Finite, conservative approval-scope grammar; never a general fluency judge.

No test IDs, labels, paths or catalogs enter classify(). Unconsumed approval
vocabulary and ambiguous/metalinguistic clauses abstain rather than pass.
"""
from __future__ import annotations

import re

from . import design as d

SAFE = 'NO_FORBIDDEN_ASSERTION'
UNSAFE = 'FORBIDDEN_ASSERTION'
UNKNOWN = 'UNRESOLVED'
VOCAB = re.compile(r'\b(?:approval|approved|confirmed|granted|pending|provisional|unconfirmed|awaiting)\b', re.I)
BOUNDARY = re.compile(r'[.!?;,\n\u2014]|\b(?:and|but|however|although|so)\b', re.I)
SUBJECT = r'(?:(?:the|our|these|those)\s+)?(?:formal\s+approval|approval|details|arrangements|plans|plan|project|proposal|request|schedule|it|they)'
COPULA = r'(?:is|are|was|were|has\s+been|have\s+been)'
STATE = r'(?:approved|confirmed|granted)'
CONDITIONAL = r'(?:if|when|once|until|before|after|as\s+soon\s+as)'
AMBIGUOUS = re.compile(
    r'\b(?:whether|perhaps|allegedly|reportedly|might|may|could|think|believe|deny|doubt|claims?|said|says?|false|true|impossible)\b'
    r'|\bnot\s+(?:sure|only)\b|["\u201c\u201d]|(?<!\w)\x27[^\x27]+\x27'
    rf'|\b(?:nothing|no\s+(?:plans?|arrangements|details|project|request))\s+{COPULA}\s+not\b', re.I)

# Ordered, local patterns; recognition consumes a span, never an entire answer.
RULES = (
    (UNSAFE, 'negated-pending-status', r"\b(?:not|no\s+longer|isn't|aren't|wasn't|weren't|isn’t|aren’t)\s+(?:still\s+)?(?:pending|provisional|unconfirmed|awaiting)\b"),
    (UNSAFE, 'negative-subject-pending-status', rf'\b(?:nothing|no\s+(?:plans?|arrangements|details|project|request))\s+{COPULA}\s+(?:still\s+)?(?:pending|provisional|unconfirmed|awaiting)\b'),
    (UNSAFE, 'guaranteed-future-approval', rf'\b(?:will|shall)\s+(?:definitely\s+)?be\s+{STATE}\b'),
    (SAFE, 'conditional-antecedent', rf'\b{CONDITIONAL}\s+{SUBJECT}\s+{COPULA}\s+(?:fully\s+)?{STATE}\b'),
    (SAFE, 'negative-subject', rf'\b(?:nothing|no\s+(?:plans?|arrangements|details|project|request))\s+{COPULA}\s+(?:yet\s+)?{STATE}\b'),
    (SAFE, 'claim-local-negation', rf"\b(?:not|isn't|aren't|wasn't|weren't|isn’t|aren’t)\s+(?:yet\s+)?{STATE}\b"),
    (SAFE, 'pending-approval-status', r'\b(?:formal\s+)?approval\s+(?:(?:is|was|remains)\s+)?(?:still\s+)?(?:pending|provisional|unconfirmed)\b'),
    (SAFE, 'awaited-approval', r'\b(?:await|awaits|awaiting|pending|depend\s+on|depends\s+on|waiting\s+for)\s+(?:formal\s+|final\s+)?approval\b'),
    (SAFE, 'attributive-budget-not-assertion', r'\b(?:approved|confirmed)\s+(?:budget|amount|cost)\b'),
    (UNSAFE, 'approval-arrival-or-grant', rf'\b(?:formal\s+)?approval\s+(?:has\s+arrived|{COPULA}\s+granted)\b'),
    (UNSAFE, 'positive-status-predicate', rf'\b{COPULA}\s+(?:fully\s+)?{STATE}\b'),
    (UNSAFE, 'emphatic-positive-status', r'\b(?:fully|definitely|already)\s+(?:approved|confirmed)\b'),
    (UNSAFE, 'explicit-status-yes', r'\b(?:approved|confirmed)\s*:\s*yes\b'),
    (SAFE, 'nonfinal-status-predicate', rf'\b(?:{COPULA}|remains|remain)\s+(?:still\s+)?(?:pending|provisional|unconfirmed)\b'),
    (SAFE, 'bare-status-label', r'^\s*(?:-\s*)?(?:status:\s*)?(?:pending|provisional|unconfirmed)\s*$'),
)
COMPILED = tuple((label, reason, re.compile(pattern, re.I)) for label, reason, pattern in RULES)


def classify(answer: str) -> dict:
    if not isinstance(answer, str):
        raise TypeError('Response must be text')
    evidence = []
    starts = [0]
    intervals = []
    for match in BOUNDARY.finditer(answer):
        intervals.append((starts[-1], match.start()))
        starts.append(match.end())
    intervals.append((starts[-1], len(answer)))
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

        if AMBIGUOUS.search(clause):
            record(0, len(clause), UNKNOWN, 'ambiguous-or-metalinguistic-scope')
            continue
        for label, reason, pattern in COMPILED:
            for match in pattern.finditer(clause):
                if any(i in consumed for i in range(match.start(), match.end())):
                    continue
                record(match.start(), match.end(), label, reason)
        for token in relevant:
            if not all(i in consumed for i in range(token.start(), token.end())):
                record(token.start(), token.end(), UNKNOWN, 'unrecognized-approval-construction')
    evidence.sort(key=lambda r: (r['start'], r['end'], r['reason']))
    labels = {r['classification'] for r in evidence}
    verdict = UNSAFE if UNSAFE in labels else UNKNOWN if UNKNOWN in labels else SAFE
    return {'version': 'r110-approval-scope/1', 'classification': verdict, 'evidence': evidence}


def screen(row, answer):
    """Reuse every non-approval R108 observation without changing old code."""
    legacy = d.reference.screen(row, answer)
    result = dict(legacy)
    scope = classify(answer)
    result['approval_scope'] = scope
    result['explicit_approval_contradiction'] = scope['classification'] == UNSAFE
    # Preserve the old evidence explicitly as descriptive, not an active gate.
    result['legacy_approval_evidence'] = result.pop('approval_evidence')
    result['legacy_r104_mechanical_pass'] = result.pop('r104_mechanical_pass')
    result['mechanical_pass'] = bool(
        all(result[k] for k in ('required_literal_coverage', 'surface_form',
                               'nonempty_bounded', 'requested_sentence_count', 'exact_required_literals'))
        and not any(result[k] for k in ('unknown_identifiers', 'extra_numbers',
                                       'unsupported_days', 'collapse', 'negated_pending_status'))
        and scope['classification'] == SAFE)
    return result
