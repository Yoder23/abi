"""Fixed model-selection frontend for non-exhaustive R14 observations."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .capability import (
    AFFINE_OPERATIONS,
    MODULUS,
    OPERATORS,
    operations_are_noncommutative,
)


class R14ExtractorError(RuntimeError):
    """Raised when extractor observations or selection are invalid."""


def candidate_operations() -> list[tuple[tuple[int, int], ...]]:
    return [
        tuple(items)
        for items in itertools.permutations(AFFINE_OPERATIONS, OPERATORS)
        if operations_are_noncommutative(items)
    ]


def extractor_spec() -> dict[str, Any]:
    value = {
        "format": "abi-r14-non-exhaustive-affine-model-selector/1",
        "input": "mixed_program_prompts_plus_source_canonical_probabilities",
        "query_answers_available": False,
        "atomic_queries_permitted": False,
        "hidden_capability_spec_available": False,
        "source_training_rows_available": False,
        "heldout_rows_available": False,
        "oracle_available": False,
        "adaptive_queries": False,
        "candidate_family": "three_distinct_noncommuting_affine_permutations_z8",
        "objective": "maximum_summed_log_source_probability",
        "learned_parameters": 0,
        "recipient_optimizer_steps": 0,
    }
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


def transition_from_operations(
    operations: Sequence[tuple[int, int]],
) -> torch.Tensor:
    transition = torch.zeros(OPERATORS, MODULUS, MODULUS, dtype=torch.float32)
    for operator, (multiplier, offset) in enumerate(operations):
        for start in range(MODULUS):
            target = (int(multiplier) * start + int(offset)) % MODULUS
            transition[operator, start, target] = 1.0
    return transition


def _validate_observations(observations: Sequence[Mapping[str, Any]]) -> None:
    if not observations:
        raise R14ExtractorError("no source observations")
    identities: set[str] = set()
    for row in observations:
        if set(row) != {
            "row_id",
            "prompt_sha256",
            "start",
            "program",
            "canonical_probabilities",
        }:
            raise R14ExtractorError("extractor observation schema changed")
        program = row["program"]
        probabilities = row["canonical_probabilities"]
        if (
            str(row["row_id"]) in identities
            or not isinstance(program, list)
            or len(program) < 2
            or any(not 0 <= int(item) < OPERATORS for item in program)
            or not 0 <= int(row["start"]) < MODULUS
            or not isinstance(probabilities, list)
            or len(probabilities) != MODULUS
            or any(not math.isfinite(float(item)) or float(item) < 0 for item in probabilities)
            or abs(sum(float(item) for item in probabilities) - 1.0) > 2e-5
        ):
            raise R14ExtractorError("invalid or duplicate extractor observation")
        identities.add(str(row["row_id"]))


def extract_transition(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select a latent compositional program using source probabilities only."""
    _validate_observations(observations)
    candidates = candidate_operations()
    operation_tensor = torch.tensor(candidates, dtype=torch.long)
    scores = torch.zeros(len(candidates), dtype=torch.float64)
    exact_matches = torch.zeros(len(candidates), dtype=torch.long)
    epsilon = torch.finfo(torch.float64).tiny
    for row in observations:
        states = torch.full((len(candidates),), int(row["start"]), dtype=torch.long)
        for operator in row["program"]:
            selected = operation_tensor[:, int(operator), :]
            states = (selected[:, 0] * states + selected[:, 1]) % MODULUS
        probabilities = torch.tensor(row["canonical_probabilities"], dtype=torch.float64)
        scores += probabilities.clamp_min(epsilon).log().index_select(0, states)
        source_prediction = int(probabilities.argmax())
        exact_matches += states.eq(source_prediction)
    order = torch.argsort(scores, descending=True, stable=True)
    best_index = int(order[0])
    runner_up = int(order[1])
    best = candidates[best_index]
    result = {
        "extractor": extractor_spec(),
        "observations": len(observations),
        "atomic_observations": sum(len(row["program"]) == 1 for row in observations),
        "candidate_programs": len(candidates),
        "selected_candidate_index": best_index,
        "selected_operations_commitment": hashlib.sha256(
            canonical_json_bytes([list(item) for item in best])
        ).hexdigest(),
        "best_log_likelihood": float(scores[best_index]),
        "runner_up_log_likelihood": float(scores[runner_up]),
        "log_likelihood_margin": float(scores[best_index] - scores[runner_up]),
        "best_hard_source_agreement": int(exact_matches[best_index]),
        "runner_up_hard_source_agreement": int(exact_matches[runner_up]),
        "query_identity_sha256": hashlib.sha256(
            canonical_json_bytes(
                [
                    {"row_id": row["row_id"], "prompt_sha256": row["prompt_sha256"]}
                    for row in observations
                ]
            )
        ).hexdigest(),
        "answers_consumed": 0,
        "oracle_calls": 0,
        "training_rows_consumed": 0,
        "heldout_rows_consumed": 0,
        "optimizer_steps": 0,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return transition_from_operations(best), result
