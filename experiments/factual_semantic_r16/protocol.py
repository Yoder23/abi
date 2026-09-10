"""Held-out selection and rows for R16 factual extraction."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from experiments.preexisting_representation_r15b.public_qualification import (
    canonical_json_bytes,
    sha256_bytes,
)

from .facts import HELDOUT_FACTS, Fact, question


def heldout_facts(secret_hex: str, commitment: str, per_namespace: int) -> list[Fact]:
    secret = bytes.fromhex(secret_hex)
    if len(secret) != 32 or hashlib.sha256(secret).hexdigest() != commitment:
        raise ValueError("R16 held-out secret does not match its commitment")
    generator = random.Random(int.from_bytes(hashlib.sha256(b"abi-r16\0" + secret).digest(), "big"))
    selected = []
    for namespace in sorted({fact.namespace for fact in HELDOUT_FACTS}):
        candidates = [fact for fact in HELDOUT_FACTS if fact.namespace == namespace]
        if not 0 < per_namespace <= len(candidates):
            raise ValueError("R16 held-out selection count changed")
        selected.extend(generator.sample(candidates, per_namespace))
    generator.shuffle(selected)
    return selected


def candidate_values(fact: Fact, limit: int, secret_hex: str) -> tuple[str, ...]:
    values = sorted({item.value for item in HELDOUT_FACTS if item.relation == fact.relation})
    if not 1 < limit <= len(values):
        raise ValueError("R16 candidate limit changed")
    distractors = [value for value in values if value != fact.value]
    seed = hashlib.sha256(f"{secret_hex}:{fact.fact_id}:candidates".encode()).digest()
    generator = random.Random(int.from_bytes(seed, "big"))
    selected = generator.sample(distractors, limit - 1) + [fact.value]
    generator.shuffle(selected)
    return tuple(selected)


def evaluation_rows(facts: list[Fact]) -> list[dict[str, Any]]:
    rows = []
    for fact in facts:
        for view in range(3, 6):
            query = question(fact, view)
            identity = {"fact_id": fact.fact_id, "view": view, "query": query}
            rows.append(
                {
                    **identity,
                    "row_id": sha256_bytes(canonical_json_bytes(identity)),
                    "namespace": fact.namespace,
                    "relation": fact.relation,
                    "entity": fact.entity,
                    "answer": fact.value,
                }
            )
    return rows
