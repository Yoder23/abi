"""Custody, split, and metric helpers for the R14 held-out campaign."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .capability import (
    AffineCapability,
    behavior_space_size,
    generate_rows,
    order_counterfactual_rows,
)


class R14Error(RuntimeError):
    """Raised when R14 custody, evidence, or a registered gate fails."""


def json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R14Error(f"required JSON unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise R14Error(f"expected JSON object: {path}")
    return value


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise R14Error(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(dict(value), indent=2, sort_keys=True).encode() + b"\n")


def write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise R14Error(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) for row in rows))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def wilson_lower(correct: int, rows: int, *, z: float = 1.959963984540054) -> float:
    if rows <= 0 or not 0 <= correct <= rows:
        raise R14Error("invalid binomial observation")
    proportion = correct / rows
    denominator = 1.0 + z * z / rows
    center = proportion + z * z / (2.0 * rows)
    radius = z * math.sqrt(proportion * (1.0 - proportion) / rows + z * z / (4.0 * rows * rows))
    return (center - radius) / denominator


def capability_rows(
    config: Mapping[str, Any], capabilities: Sequence[AffineCapability]
) -> list[dict[str, list[dict[str, Any]]]]:
    data = config["data"]
    result = []
    for index, capability in enumerate(capabilities):
        training = generate_rows(
            capability,
            split="heldout_source_train",
            rows=int(data["training_rows_per_capability"]),
            depths=data["training_depths"],
            seed=int(data["training_seed"]) + 1009 * index,
        )
        excluded = {str(row["program_key"]) for row in training}
        queries = generate_rows(
            capability,
            split="heldout_extractor_query",
            rows=int(data["extractor_queries_per_capability"]),
            depths=data["extractor_query_depths"],
            seed=int(data["query_seed"]) + 2003 * index,
            excluded_keys=excluded,
        )
        excluded.update(str(row["program_key"]) for row in queries)
        evaluation = generate_rows(
            capability,
            split="heldout_unseen_evaluation",
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
        all_rows = [*training, *queries, *evaluation, *counterfactual]
        keys = [str(row["program_key"]) for row in all_rows]
        if len(keys) != len(set(keys)):
            raise R14Error("source/query/evaluation program overlap")
        if any(len(row["program"]) == 1 for row in queries):
            raise R14Error("atomic extractor query generated")
        result.append(
            {
                "training": training,
                "queries": queries,
                "evaluation": evaluation,
                "counterfactual": counterfactual,
                "recipient": evaluation[: int(data["recipient_rows_per_capability"])],
            }
        )
    return result


def source_metrics(
    observations: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {str(row["row_id"]): int(row["answer"]) for row in rows}
    if len(expected) != len(rows):
        raise R14Error("duplicate expected source rows")
    seen: set[str] = set()
    correct = 0
    nll = 0.0
    for observation in observations:
        row_id = str(observation["row_id"])
        probabilities = observation["canonical_probabilities"]
        if (
            row_id in seen
            or row_id not in expected
            or not isinstance(probabilities, list)
            or len(probabilities) != 8
            or any(not math.isfinite(float(value)) or float(value) < 0 for value in probabilities)
            or abs(sum(float(value) for value in probabilities) - 1.0) > 2e-5
        ):
            raise R14Error("source observation invalid")
        seen.add(row_id)
        answer = expected[row_id]
        correct += int(max(range(8), key=probabilities.__getitem__) == answer)
        nll -= math.log(max(float(probabilities[answer]), 1e-300))
    if seen != set(expected):
        raise R14Error("source observation coverage changed")
    return {
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "mean_canonical_nll": nll / len(rows),
        "wilson_95_lower": wilson_lower(correct, len(rows)),
    }


def behavior_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    depths = config["data"]["evaluation_depths"]
    space = behavior_space_size(depths)
    queries = int(config["data"]["extractor_queries_per_capability"])
    return {
        "evaluation_behavior_space": space,
        "extractor_query_budget": queries,
        "query_to_behavior_ratio": queries / space,
        "atomic_extractor_queries": 0,
    }
