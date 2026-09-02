"""Opaque compositional capabilities and contamination-resistant row generation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
    render_prompt,
)

MODULUS = 8
OPERATORS = 3
AFFINE_MULTIPLIERS = (1, 3, 5, 7)
AFFINE_OPERATIONS = tuple(
    (multiplier, offset) for multiplier in AFFINE_MULTIPLIERS for offset in range(MODULUS)
)


class R14CapabilityError(ValueError):
    """Raised when an R14 capability or split violates its contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _compose(first: tuple[int, int], second: tuple[int, int]) -> tuple[int, int]:
    """Return second(first(x)) in the affine ring modulo eight."""
    a1, b1 = first
    a2, b2 = second
    return (a2 * a1) % MODULUS, (a2 * b1 + b2) % MODULUS


def operations_are_noncommutative(operations: Sequence[tuple[int, int]]) -> bool:
    return any(
        _compose(operations[left], operations[right])
        != _compose(operations[right], operations[left])
        for left in range(len(operations))
        for right in range(left + 1, len(operations))
    )


@dataclass(frozen=True)
class AffineCapability:
    """Three opaque, composable affine permutations over eight latent states."""

    capability_id: str
    operations: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    seed_commitment: str

    def __post_init__(self) -> None:
        if (
            len(self.operations) != OPERATORS
            or len(set(self.operations)) != OPERATORS
            or any(operation not in AFFINE_OPERATIONS for operation in self.operations)
            or not operations_are_noncommutative(self.operations)
        ):
            raise R14CapabilityError("capability operations are not admissible")
        if len(self.seed_commitment) != 64:
            raise R14CapabilityError("seed commitment is not SHA-256")

    def apply(self, start: int, program: Sequence[int]) -> int:
        value = int(start)
        if not 0 <= value < MODULUS or not program:
            raise R14CapabilityError("invalid state or empty program")
        for operator in program:
            if not 0 <= int(operator) < OPERATORS:
                raise R14CapabilityError("invalid operator")
            multiplier, offset = self.operations[int(operator)]
            value = (multiplier * value + offset) % MODULUS
        return value

    def transition(self) -> torch.Tensor:
        value = torch.zeros(OPERATORS, MODULUS, MODULUS, dtype=torch.float32)
        for operator in range(OPERATORS):
            for start in range(MODULUS):
                value[operator, start, self.apply(start, [operator])] = 1.0
        return value

    def private_document(self) -> dict[str, Any]:
        return {
            "format": "abi-r14-private-affine-capability/1",
            "capability_id": self.capability_id,
            "operations": [list(item) for item in self.operations],
            "seed_commitment": self.seed_commitment,
        }


def capability_from_seed(seed: int, *, split: str, index: int) -> AffineCapability:
    generator = random.Random(int(seed))
    while True:
        operations = tuple(generator.sample(AFFINE_OPERATIONS, OPERATORS))
        if operations_are_noncommutative(operations):
            break
    seed_commitment = sha256_bytes(int(seed).to_bytes(16, "big", signed=False))
    identity = sha256_bytes(
        canonical_json_bytes(
            {"split": split, "index": int(index), "seed_commitment": seed_commitment}
        )
    )[:20]
    return AffineCapability(
        capability_id=f"r14-{split}-{index:03d}-{identity}",
        operations=operations,  # type: ignore[arg-type]
        seed_commitment=seed_commitment,
    )


def _derived_seed(secret: bytes, split: str, index: int) -> int:
    digest = hashlib.sha256(
        b"abi-non-exhaustive-r14\0"
        + secret
        + b"\0"
        + split.encode("ascii")
        + int(index).to_bytes(8, "big")
    ).digest()
    return int.from_bytes(digest[:16], "big")


def heldout_capabilities(
    secret_hex: str, *, expected_commitment: str, count: int
) -> list[AffineCapability]:
    try:
        secret = bytes.fromhex(secret_hex)
    except ValueError as exc:
        raise R14CapabilityError("held-out secret is not hexadecimal") from exc
    if len(secret) != 32 or sha256_bytes(secret) != expected_commitment:
        raise R14CapabilityError("held-out secret does not match preregistration")
    return [
        capability_from_seed(_derived_seed(secret, "heldout", index), split="heldout", index=index)
        for index in range(int(count))
    ]


def public_capability(seed: int, *, index: int = 0) -> AffineCapability:
    return capability_from_seed(int(seed), split="public", index=index)


def behavior_space_size(depths: Sequence[int]) -> int:
    return MODULUS * sum(OPERATORS ** int(depth) for depth in set(depths))


def generate_rows(
    capability: AffineCapability,
    *,
    split: str,
    rows: int,
    depths: Sequence[int],
    seed: int,
    excluded_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Sample unique programs without materializing the multi-billion-case universe."""
    if rows <= 0 or not depths or any(int(depth) <= 0 for depth in depths):
        raise R14CapabilityError("invalid row request")
    excluded = set() if excluded_keys is None else set(excluded_keys)
    if rows + len(excluded) > behavior_space_size(depths):
        raise R14CapabilityError("row request exceeds behavior universe")
    generator = random.Random(int(seed))
    result: list[dict[str, Any]] = []
    seen = set(excluded)
    attempts = 0
    while len(result) < rows:
        attempts += 1
        if attempts > rows * 1000:
            raise R14CapabilityError("could not sample unique programs")
        depth = int(depths[generator.randrange(len(depths))])
        start = generator.randrange(MODULUS)
        program = tuple(generator.randrange(OPERATORS) for _ in range(depth))
        key = f"{start}:" + ",".join(str(value) for value in program)
        if key in seen:
            continue
        seen.add(key)
        prompt = render_prompt(start, program)
        identity = {
            "capability_id": capability.capability_id,
            "split": split,
            "start": start,
            "program": list(program),
        }
        result.append(
            {
                **identity,
                "row_id": sha256_bytes(canonical_json_bytes(identity)),
                "prompt": prompt,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "depth": depth,
                "answer": capability.apply(start, program),
                "program_key": key,
            }
        )
    return result


def order_counterfactual_rows(
    capability: AffineCapability,
    *,
    pairs: int,
    depth: int,
    seed: int,
    excluded_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Create order-swapped pairs whose oracle outputs differ."""
    if pairs <= 0 or depth < 2:
        raise R14CapabilityError("invalid counterfactual request")
    generator = random.Random(int(seed))
    excluded = set() if excluded_keys is None else set(excluded_keys)
    rows: list[dict[str, Any]] = []
    seen = set(excluded)
    while len(rows) < 2 * pairs:
        start = generator.randrange(MODULUS)
        program = [generator.randrange(OPERATORS) for _ in range(depth)]
        left = generator.randrange(depth)
        right = generator.randrange(depth)
        if left == right or program[left] == program[right]:
            continue
        swapped = list(program)
        swapped[left], swapped[right] = swapped[right], swapped[left]
        if capability.apply(start, program) == capability.apply(start, swapped):
            continue
        pair_rows = []
        for variant, candidate in (("original", program), ("swapped", swapped)):
            key = f"{start}:" + ",".join(str(value) for value in candidate)
            if key in seen:
                pair_rows = []
                break
            prompt = render_prompt(start, candidate)
            identity = {
                "capability_id": capability.capability_id,
                "split": "order_counterfactual",
                "pair_index": len(rows) // 2,
                "variant": variant,
                "start": start,
                "program": candidate,
            }
            pair_rows.append(
                {
                    **identity,
                    "row_id": sha256_bytes(canonical_json_bytes(identity)),
                    "prompt": prompt,
                    "prompt_sha256": sha256_bytes(prompt.encode()),
                    "depth": depth,
                    "answer": capability.apply(start, candidate),
                    "program_key": key,
                }
            )
        if len(pair_rows) == 2:
            rows.extend(pair_rows)
            seen.update(row["program_key"] for row in pair_rows)
    return rows
