"""Contamination-resistant synthetic capabilities for ABI R8.

This module is generator/scorer-side code. Recipient workers must not import it.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

OPERATORS = ("vok", "narel", "tem")
MODULUS = 8


class CapabilityGeneratorError(ValueError):
    """Raised when a synthetic capability or split violates the protocol."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class OpaqueCapability:
    capability_id: str
    offsets: tuple[int, int, int]
    seed_commitment: str

    def __post_init__(self) -> None:
        if len(self.offsets) != len(OPERATORS):
            raise CapabilityGeneratorError("one offset is required per opaque operator")
        if any(value < 0 or value >= MODULUS for value in self.offsets):
            raise CapabilityGeneratorError("opaque offsets must lie in Z/8Z")
        if len(self.seed_commitment) != 64:
            raise CapabilityGeneratorError("capability seed commitment must be SHA-256")

    def apply(self, start: int, program: Sequence[int]) -> int:
        value = int(start)
        if value < 0 or value >= MODULUS or not program:
            raise CapabilityGeneratorError("invalid start state or empty program")
        for operator in program:
            if operator < 0 or operator >= len(self.offsets):
                raise CapabilityGeneratorError("invalid opaque operator")
            value = (value + self.offsets[operator]) % MODULUS
        return value

    def private_document(self) -> dict[str, Any]:
        return {
            "format": "abi-r8-private-opaque-capability/1",
            "capability_id": self.capability_id,
            "offsets": list(self.offsets),
            "seed_commitment": self.seed_commitment,
        }


def _derive_seed(secret: bytes, label: str, index: int) -> int:
    digest = hashlib.sha256(
        b"abi-native-transfer-r8\0"
        + secret
        + b"\0"
        + label.encode("ascii")
        + index.to_bytes(8, "big")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def capability_from_seed(seed: int, *, split: str, index: int) -> OpaqueCapability:
    generator = random.Random(seed)
    offsets = tuple(generator.randrange(MODULUS) for _ in OPERATORS)
    if len(set(offsets)) == 1:
        offsets = (offsets[0], (offsets[1] + 1) % MODULUS, (offsets[2] + 3) % MODULUS)
    commitment = sha256_bytes(seed.to_bytes(8, "big"))
    identity = sha256_bytes(
        canonical_json_bytes({"split": split, "index": index, "seed_commitment": commitment})
    )[:20]
    return OpaqueCapability(
        capability_id=f"r8-{split}-{index:03d}-{identity}",
        offsets=offsets,
        seed_commitment=commitment,
    )


def committed_heldout_capabilities(
    secret_hex: str, *, expected_commitment: str, count: int
) -> list[OpaqueCapability]:
    try:
        secret = bytes.fromhex(secret_hex)
    except ValueError as exc:
        raise CapabilityGeneratorError("held-out secret is not hexadecimal") from exc
    if len(secret) != 32 or sha256_bytes(secret) != expected_commitment:
        raise CapabilityGeneratorError("held-out secret does not match preregistration")
    return _unique_capabilities(secret, split="heldout", count=count)


def public_capabilities(seed: int, *, split: str, count: int) -> list[OpaqueCapability]:
    if split not in {"meta_train", "development"}:
        raise CapabilityGeneratorError("public capability split is not permitted")
    secret = int(seed).to_bytes(16, "big", signed=False)
    return _unique_capabilities(secret, split=split, count=count)


def _unique_capabilities(secret: bytes, *, split: str, count: int) -> list[OpaqueCapability]:
    if count <= 0 or count > MODULUS ** len(OPERATORS):
        raise CapabilityGeneratorError("invalid unique capability count")
    result = []
    seen = set()
    candidate_index = 0
    while len(result) < count:
        capability = capability_from_seed(
            _derive_seed(secret, split, candidate_index),
            split=split,
            index=candidate_index,
        )
        candidate_index += 1
        if capability.offsets in seen:
            continue
        seen.add(capability.offsets)
        result.append(capability)
    return result


def render_prompt(start: int, program: Sequence[int]) -> str:
    if not program:
        raise CapabilityGeneratorError("cannot render an empty program")
    operations = " ".join(OPERATORS[index] for index in program)
    return f"Opaque program: start {int(start)} ; apply {operations} ; result ="


def _render_challenging_prompt(
    start: int,
    program: Sequence[int],
    *,
    capability: OpaqueCapability,
    flavor: str,
    answer: int,
) -> str:
    normal = render_prompt(start, program)
    if flavor == "counterfactual":
        neighbor = list(program)
        original = neighbor[-1]
        replacement = next(
            index
            for index in range(len(OPERATORS))
            if capability.offsets[index] != capability.offsets[original]
        )
        neighbor[-1] = replacement
        distractor = " ".join(OPERATORS[index] for index in neighbor)
        return (
            f"Counterfactual check: ignore near-neighbor '{distractor}'. "
            + normal
        )
    if flavor == "adversarial_near_neighbor":
        wrong = (answer + 1) % MODULUS
        return f"Incorrect proposal {wrong}. Return the corrected value. {normal}"
    return normal


def render_composed_prompt(
    start: int,
    first_program: Sequence[int],
    second_program: Sequence[int],
) -> str:
    if not first_program or not second_program:
        raise CapabilityGeneratorError("composed programs must have two nonempty stages")
    first = " ".join("a" + OPERATORS[index] for index in first_program)
    second = " ".join("b" + OPERATORS[index] for index in second_program)
    return (
        f"Opaque dual program: start {int(start)} ; A apply {first} ; "
        f"B apply {second} ; result ="
    )


def generate_composition_rows(
    first: OpaqueCapability,
    second: OpaqueCapability,
    *,
    split: str,
    rows: int,
    first_depths: Sequence[int],
    second_depths: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    if rows <= 0 or not first_depths or not second_depths:
        raise CapabilityGeneratorError("invalid composition row request")
    universe = [
        (start, tuple(first_program), tuple(second_program))
        for first_depth in sorted(set(int(value) for value in first_depths))
        for second_depth in sorted(set(int(value) for value in second_depths))
        for start in range(MODULUS)
        for first_program in itertools.product(range(len(OPERATORS)), repeat=first_depth)
        for second_program in itertools.product(range(len(OPERATORS)), repeat=second_depth)
    ]
    if rows > len(universe):
        raise CapabilityGeneratorError(
            f"requested {rows} composition rows but universe has {len(universe)}"
        )
    generator = random.Random(seed)
    generator.shuffle(universe)
    result = []
    for start, first_program, second_program in universe[:rows]:
        intermediate = first.apply(start, first_program)
        answer = second.apply(intermediate, second_program)
        prompt = render_composed_prompt(start, first_program, second_program)
        identity = {
            "first_capability_id": first.capability_id,
            "second_capability_id": second.capability_id,
            "split": split,
            "start": start,
            "first_program": list(first_program),
            "second_program": list(second_program),
        }
        result.append(
            {
                "row_id": sha256_bytes(canonical_json_bytes(identity)),
                **identity,
                "prompt": prompt,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "answer": answer,
            }
        )
    return result


def _program_key(start: int, program: Sequence[int]) -> str:
    return f"{start}:" + ",".join(str(value) for value in program)


def generate_rows(
    capability: OpaqueCapability,
    *,
    split: str,
    rows: int,
    depths: Sequence[int],
    seed: int,
) -> list[dict[str, Any]]:
    if rows <= 0 or not depths or any(depth <= 0 for depth in depths):
        raise CapabilityGeneratorError("invalid row count or program depth")
    generator = random.Random(seed)
    seen: set[str] = set()
    result: list[dict[str, Any]] = []

    universe = [
        (start, tuple(program))
        for depth in sorted(set(int(value) for value in depths))
        for start in range(MODULUS)
        for program in itertools.product(range(len(OPERATORS)), repeat=depth)
    ]
    if rows > len(universe):
        raise CapabilityGeneratorError(
            f"requested {rows} unique rows but depths permit only {len(universe)}"
        )

    # Source extraction requires complete atomic probe coverage. Everything
    # else is sampled without replacement from the finite program universe.
    atomic = [
        (start, (operator,))
        for operator in range(len(OPERATORS))
        for start in range(MODULUS)
    ] if split == "source_train" and 1 in depths else []
    atomic_keys = {_program_key(start, program) for start, program in atomic}
    remainder = [
        (start, program)
        for start, program in universe
        if _program_key(start, program) not in atomic_keys
    ]
    generator.shuffle(remainder)
    selected = [*atomic, *remainder[: rows - len(atomic)]]
    candidates = [
        (
            start,
            program,
            "atomic_coverage"
            if index < len(atomic)
            else "counterfactual"
            if index % 7 == 0
            else "adversarial_near_neighbor"
            if index % 11 == 0
            else "compositional",
        )
        for index, (start, program) in enumerate(selected)
    ]

    for start, program, flavor in candidates:
        key = _program_key(start, program)
        if key in seen:
            continue
        seen.add(key)
        answer = capability.apply(start, program)
        prompt = _render_challenging_prompt(
            start,
            program,
            capability=capability,
            flavor=flavor,
            answer=answer,
        )
        row_id = sha256_bytes(
            canonical_json_bytes(
                {
                    "capability_id": capability.capability_id,
                    "split": split,
                    "key": key,
                }
            )
        )
        result.append(
            {
                "row_id": row_id,
                "capability_id": capability.capability_id,
                "split": split,
                "prompt": prompt,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "start": start,
                "program": list(program),
                "depth": len(program),
                "flavor": flavor,
                "answer": answer,
            }
        )
        if len(result) == rows:
            break
    if len(result) != rows:
        raise CapabilityGeneratorError("could not construct the requested unique rows")
    return result


def worker_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {
        "row_id",
        "capability_id",
        "split",
        "prompt",
        "prompt_sha256",
        "depth",
        "flavor",
    }
    return [{key: row[key] for key in sorted(allowed)} for row in rows]


def write_jsonl_once(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    if path.exists():
        raise CapabilityGeneratorError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
