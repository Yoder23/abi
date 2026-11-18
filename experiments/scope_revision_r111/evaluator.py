"""Versioned finite clause grammar. Unknown/ambiguous forms never pass."""
from __future__ import annotations

import re

from experiments.scope_adherence_r110 import evaluator as old

SAFE, UNSAFE, UNKNOWN = old.SAFE, old.UNSAFE, old.UNKNOWN
VOCAB = old.VOCAB
BOUNDARY = re.compile(old.BOUNDARY.pattern + '|\u2013', re.I)
AMBIGUOUS = re.compile(old.AMBIGUOUS.pattern +
    r'|\b(?:nothing|no\s+\w+)\s+(?:has|have)\s+not\s+been\b', re.I)
STATE = old.STATE
NONFINAL = r'(?:pending|provisional|unconfirmed)'
NOUN = r'(?:arrangements?|details|elements|plans?|projects?|sessions?|requests?|schedules?|proposals?)'
APPROVAL = r'(?:(?:formal|final)\s+)?approval'
PERFECT_NEGATIVE = rf"(?:has\s+not|have\s+not|hasn['\u2019]t|haven['\u2019]t)\s+been\s+(?:yet\s+)?{STATE}"
NEW_RULES = (
    (UNSAFE, 'expanded-negated-nonfinal', rf'\bnot\s+(?:(?:still|currently)\s+)?(?:in\s+)?(?:a\s+)?{NONFINAL}\b'),
    (UNSAFE, 'negative-subject-nominal-status', rf'\bno\s+{NOUN}\s+(?:still|currently|remaining)\s+{NONFINAL}\b'),
    (SAFE, 'perfect-approval-negation', rf'\b(?:{APPROVAL}\s+)?{PERFECT_NEGATIVE}\b'),
    (SAFE, 'reduced-conditional', rf'\b{old.CONDITIONAL}\s+(?:fully\s+)?{STATE}\b'),
    (SAFE, 'contracted-conditional', rf"\b{old.CONDITIONAL}\s+{old.SUBJECT}['\u2019]s\s+(?:fully\s+)?{STATE}\b"),
    (SAFE, 'approval-event-conditional', rf'\b{old.CONDITIONAL}\s+{APPROVAL}\s+(?:is\s+given|comes\s+through)\b'),
    (SAFE, 'approval-status-or-purpose', rf'\b{APPROVAL}\s+(?:status\s+|for\s+(?:the\s+)?{NOUN}\s+)?(?:is|remains)\s+(?:still\s+)?{NONFINAL}\b'),
    (SAFE, 'subject-to-approval', rf'\b{NOUN}\s+(?:remain|remains|are|is)\s+subject\s+to\s+{APPROVAL}\b'),
    (SAFE, 'nominal-nonfinal-status', rf'\b{NOUN}\s+(?:still|currently|remaining)\s+{NONFINAL}\b'),
    (SAFE, 'nonfinal-state', rf'\b(?:is|are|remains|remain)\s+(?:currently\s+)?in\s+a\s+{NONFINAL}\s+state\b'),
    (SAFE, 'attributive-nonfinal', rf'\b{NONFINAL}\s+{NOUN}\b'),
    (SAFE, 'contracted-nonfinal', rf"\b(?:it|that|this)['\u2019]s\s+(?:still\s+)?{NONFINAL}\b"),
    (UNSAFE, 'contracted-affirmative', rf"\b{old.SUBJECT}['\u2019]s\s+(?:fully\s+)?{STATE}\b"),
    (UNSAFE, 'approval-event-affirmative', rf'\b{APPROVAL}\s+(?:is\s+given|comes\s+through|has\s+come\s+through)\b'),
)
# Retain historical unsafe guards before any newly recognized local safe span.
RULES = old.RULES[:3] + NEW_RULES + old.RULES[3:]
COMPILED = [(label, reason, re.compile(pattern, re.I)) for label, reason, pattern in RULES]


def classify(answer):
    if not isinstance(answer, str):
        raise TypeError('Response must be text')
    intervals, start = [], 0
    for match in BOUNDARY.finditer(answer):
        intervals.append((start, match.start()))
        start = match.end()
    intervals.append((start, len(answer)))
    evidence = []
    for start, end in intervals:
        clause = answer[start:end]
        relevant = list(VOCAB.finditer(clause))
        if not relevant:
            continue
        consumed = set()
        def record(a, b, label, reason):
            evidence.append({'start': start + a, 'end': start + b, 'text': clause[a:b],
                             'classification': label, 'reason': reason,
                             'clause_start': start, 'clause_end': end})
            consumed.update(range(a, b))
        if AMBIGUOUS.search(clause):
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
    labels = {r['classification'] for r in evidence}
    verdict = UNSAFE if UNSAFE in labels else UNKNOWN if UNKNOWN in labels else SAFE
    return {'version': 'r111-approval-scope/1', 'classification': verdict, 'evidence': evidence}


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
