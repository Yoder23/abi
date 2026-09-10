"""Secret mapping and held-out rows for R15B."""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass
from typing import Any

from experiments.foreign_capability_r14.capability import AffineCapability, generate_rows

from .public_qualification import OPERATIONS, canonical_json_bytes, sha256_bytes


@dataclass(frozen=True)
class HeldoutCapability:
    capability: AffineCapability
    slot_order: tuple[int, int, int]

    @property
    def operation_labels(self) -> list[str]:
        return [OPERATIONS[index][0] for index in self.slot_order]


def heldout_capabilities(
    secret_hex: str,
    *,
    expected_commitment: str,
    count: int,
) -> list[HeldoutCapability]:
    try:
        secret = bytes.fromhex(secret_hex)
    except ValueError as exc:
        raise ValueError("R15B held-out secret is not hexadecimal") from exc
    if len(secret) != 32 or hashlib.sha256(secret).hexdigest() != expected_commitment:
        raise ValueError("R15B held-out secret does not match preregistration")
    permutations = list(itertools.permutations(range(3)))
    generator = random.Random(
        int.from_bytes(hashlib.sha256(b"abi-r15b\0" + secret).digest(), "big")
    )
    generator.shuffle(permutations)
    if count <= 0 or count > len(permutations):
        raise ValueError("R15B held-out capability count changed")
    result = []
    for index, slot_order in enumerate(permutations[:count]):
        operations = tuple((OPERATIONS[item][1], OPERATIONS[item][2]) for item in slot_order)
        identity = {
            "index": index,
            "slot_order_commitment": sha256_bytes(canonical_json_bytes(list(slot_order))),
            "secret_commitment": expected_commitment,
        }
        capability_id = "r15b-heldout-%02d-%s" % (
            index,
            sha256_bytes(canonical_json_bytes(identity))[:20],
        )
        result.append(
            HeldoutCapability(
                capability=AffineCapability(
                    capability_id=capability_id,
                    operations=operations,  # type: ignore[arg-type]
                    seed_commitment=expected_commitment,
                ),
                slot_order=slot_order,  # type: ignore[arg-type]
            )
        )
    return result


def evaluation_rows(
    config: dict[str, Any], heldout: list[HeldoutCapability]
) -> list[list[dict[str, Any]]]:
    data = config["data"]
    return [
        generate_rows(
            item.capability,
            split="r15b_heldout_package_evaluation",
            rows=int(data["evaluation_rows_per_capability"]),
            depths=data["evaluation_depths"],
            seed=int(data["evaluation_seed"]) + 4001 * index,
        )
        for index, item in enumerate(heldout)
    ]
