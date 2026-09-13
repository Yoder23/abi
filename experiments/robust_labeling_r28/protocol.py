"""R28 selection and quorum rules."""

from __future__ import annotations

import hashlib
import random
from collections import Counter

from experiments.autonomous_labeling_r27.protocol import answer_key
from .facts import HIDDEN_FACTS


def selected_facts(secret_hex: str, commitment: str, per_domain: int):
    secret = bytes.fromhex(secret_hex)
    if len(secret) != 32 or hashlib.sha256(secret).hexdigest() != commitment: raise ValueError("R28 reveal mismatch")
    generator = random.Random(int.from_bytes(hashlib.sha256(b"abi-r28\0" + secret).digest(), "big")); result = []
    for domain in sorted({fact.oracle_domain for fact in HIDDEN_FACTS}):
        candidates = [fact for fact in HIDDEN_FACTS if fact.oracle_domain == domain]; result.extend(generator.sample(candidates, per_domain))
    generator.shuffle(result); return result


def unique_quorum(values, key=lambda value: value, minimum: int = 2):
    counts = Counter(key(value) for value in values)
    if not counts: return None
    top = max(counts.values()); winners = [value for value, count in counts.items() if count == top]
    return winners[0] if top >= minimum and len(winners) == 1 else None


def answer_packages(packages, query: str):
    """Execute arbitrary free-labeled factual packages without an ontology router."""
    lowered = query.casefold()
    matches = []
    for package in packages:
        for fact in package["facts"]:
            entity = fact["entity"].casefold()
            if entity in lowered:
                matches.append((len(entity), fact["value"]))
    if not matches:
        return None
    longest = max(length for length, _value in matches)
    values = {value for length, value in matches if length == longest}
    if len(values) != 1:
        return None
    return next(iter(values))
