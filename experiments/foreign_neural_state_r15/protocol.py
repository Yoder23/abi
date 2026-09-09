"""Secret derivation and disjoint row construction for R15A."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from experiments.foreign_capability_r14.capability import (
    AffineCapability,
    capability_from_seed,
    generate_rows,
    order_counterfactual_rows,
)
from experiments.foreign_capability_r14.core import R14Error
from experiments.native_transfer_r8.capability_generator import canonical_json_bytes


def heldout_capabilities(
    secret_hex: str, *, expected_commitment: str, count: int
) -> list[AffineCapability]:
    try:
        secret = bytes.fromhex(secret_hex)
    except ValueError as exc:
        raise R14Error("R15 held-out secret is not hexadecimal") from exc
    if len(secret) != 32 or hashlib.sha256(secret).hexdigest() != expected_commitment:
        raise R14Error("R15 held-out secret does not match preregistration")
    result = []
    seen = set()
    candidate = 0
    while len(result) < int(count):
        digest = hashlib.sha256(
            b"abi-foreign-neural-state-r15a\0"
            + secret
            + candidate.to_bytes(8, "big")
        ).digest()
        capability = capability_from_seed(
            int.from_bytes(digest[:16], "big"),
            split="r15_heldout",
            index=len(result),
        )
        candidate += 1
        if capability.operations in seen:
            continue
        seen.add(capability.operations)
        result.append(capability)
    return result


def capability_rows(
    config: Mapping[str, Any], capabilities: Sequence[AffineCapability]
) -> list[dict[str, list[dict[str, Any]]]]:
    data = config["data"]
    result = []
    for index, capability in enumerate(capabilities):
        atomic = generate_rows(
            capability,
            split="r15_heldout_source_atomic",
            rows=24,
            depths=[1],
            seed=int(data["atomic_seed"]) + 1009 * index,
        )
        queries = generate_rows(
            capability,
            split="r15_heldout_black_box_query",
            rows=int(data["black_box_queries_per_capability"]),
            depths=data["black_box_query_depths"],
            seed=int(data["query_seed"]) + 2003 * index,
        )
        excluded = {str(row["program_key"]) for row in atomic + queries}
        evaluation = generate_rows(
            capability,
            split="r15_heldout_unseen_evaluation",
            rows=int(data["evaluation_rows_per_capability"]),
            depths=data["evaluation_depths"],
            seed=int(data["evaluation_seed"]) + 4001 * index,
            excluded_keys=excluded,
        )
        excluded.update(str(row["program_key"]) for row in evaluation)
        counterfactual = order_counterfactual_rows(
            capability,
            pairs=int(data["counterfactual_pairs_per_capability"]),
            depth=int(data["counterfactual_depth"]),
            seed=int(data["counterfactual_seed"]) + 8009 * index,
            excluded_keys=excluded,
        )
        keys = [
            str(row["program_key"])
            for row in atomic + queries + evaluation + counterfactual
        ]
        if len(keys) != len(set(keys)) or {len(row["program"]) for row in atomic} != {1}:
            raise R14Error("R15 source/evaluation split overlap or atomic drift")
        result.append(
            {
                "atomic": atomic,
                "queries": queries,
                "evaluation": evaluation,
                "counterfactual": counterfactual,
                "source_evaluation": evaluation[
                    : int(data["source_behavior_rows_per_capability"])
                ],
                "recipient": evaluation[: int(data["recipient_rows_per_capability"])],
            }
        )
    return result


def operations_commitment(capability: AffineCapability) -> str:
    return hashlib.sha256(
        canonical_json_bytes([list(operation) for operation in capability.operations])
    ).hexdigest()
