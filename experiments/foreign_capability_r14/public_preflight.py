"""Run the single bounded public R14 source/extractor development preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import torch

from experiments.foreign_teacher_r12.teacher import evaluate
from experiments.native_isa_r11.core import transition_accuracy
from experiments.native_transfer_r8.capability_generator import canonical_json_bytes
from experiments.native_transfer_r8.native_host import (
    SPECS,
    FrozenNeuralHost,
    GenericRecipientAdapterSet,
)

from .capability import (
    behavior_space_size,
    generate_rows,
    order_counterfactual_rows,
    public_capability,
)
from .extractor import extract_transition
from .source import observe_queries, train_fixed_schedule


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"immutable preflight output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _transition_predictions(transition: torch.Tensor, rows: list[dict[str, Any]]) -> list[int]:
    predictions = []
    for row in rows:
        state = torch.nn.functional.one_hot(torch.tensor(int(row["start"])), num_classes=8).float()
        for operator in row["program"]:
            state = torch.matmul(state, transition[int(operator)])
        predictions.append(int(state.argmax()))
    return predictions


def run(output: Path, *, formulation: str) -> dict[str, Any]:
    if formulation == "v1_full_vocabulary":
        training_rows = 8000
        training_depths = range(1, 9)
        development_depths = range(9, 13)
        query_depths = range(3, 9)
        evaluation_depths = range(10, 19)
        steps = 2000
        batch_size = 16
        objective = "full_vocabulary"
    elif formulation == "v2_canonical_curriculum":
        training_rows = 20000
        training_depths = range(1, 13)
        development_depths = range(13, 16)
        query_depths = range(4, 13)
        evaluation_depths = range(13, 19)
        steps = 4000
        batch_size = 24
        objective = "canonical_plus_native"
    else:
        raise RuntimeError(f"unknown preflight formulation: {formulation}")
    capability = public_capability(14014001)
    training = generate_rows(
        capability,
        split="public_source_train",
        rows=training_rows,
        depths=training_depths,
        seed=14014002,
    )
    excluded = {str(row["program_key"]) for row in training}
    development = generate_rows(
        capability,
        split="public_development",
        rows=2048,
        depths=development_depths,
        seed=14014003,
        excluded_keys=excluded,
    )
    excluded.update(str(row["program_key"]) for row in development)
    queries = generate_rows(
        capability,
        split="public_extractor_query",
        rows=256,
        depths=query_depths,
        seed=14014004,
        excluded_keys=excluded,
    )
    excluded.update(str(row["program_key"]) for row in queries)
    evaluation = generate_rows(
        capability,
        split="public_unseen_evaluation",
        rows=10000,
        depths=evaluation_depths,
        seed=14014005,
        excluded_keys=excluded,
    )
    excluded.update(str(row["program_key"]) for row in evaluation)
    counterfactual = order_counterfactual_rows(
        capability,
        pairs=500,
        depth=14,
        seed=14014006,
        excluded_keys=excluded,
    )
    host = FrozenNeuralHost(SPECS["qwen2"], device="cuda")
    base_sha = host.model_state_sha256
    before = evaluate(host, development, batch_size=64)
    adapters = GenericRecipientAdapterSet(host, rank=16)
    acquisition = train_fixed_schedule(
        host,
        adapters,
        training,
        development,
        steps=steps,
        learning_rate=0.0002,
        batch_size=batch_size,
        evaluation_interval=500,
        evaluation_batch_size=64,
        seed=14014007,
        objective=objective,
    )
    evaluation_observations, after = observe_queries(host, evaluation, batch_size=64)
    counterfactual_observations, after_counterfactual = observe_queries(
        host, counterfactual, batch_size=64
    )
    observations, query_metrics = observe_queries(host, queries, batch_size=64)
    transition, extraction = extract_transition(observations)
    package_predictions = _transition_predictions(transition, evaluation)
    source_predictions = [
        max(range(8), key=row["canonical_probabilities"].__getitem__)
        for row in evaluation_observations
    ]
    counterfactual_package_predictions = _transition_predictions(transition, counterfactual)
    counterfactual_source_predictions = [
        max(range(8), key=row["canonical_probabilities"].__getitem__)
        for row in counterfactual_observations
    ]
    result: dict[str, Any] = {
        "format": "abi-r14-public-preflight/1",
        "formulation": formulation,
        "scope": "PUBLIC_DEVELOPMENT_ONLY_NOT_HELDOUT_EVIDENCE",
        "source": {
            "model_id": host.spec.model_id,
            "revision": host.spec.revision,
            "base_sha256_before": base_sha,
            "base_sha256_after": adapters.base_state_sha256(),
            "before_public_development": before,
            "after_unseen_evaluation": after,
            "after_order_counterfactual": after_counterfactual,
            "query_metrics": query_metrics,
            "acquisition": acquisition,
        },
        "data": {
            "training_rows": len(training),
            "training_depths": list(training_depths),
            "extractor_queries": len(queries),
            "atomic_extractor_queries": 0,
            "evaluation_rows": len(evaluation),
            "evaluation_depths": list(evaluation_depths),
            "order_counterfactual_rows": len(counterfactual),
            "evaluation_behavior_space": behavior_space_size(evaluation_depths),
            "all_program_keys_distinct": len(excluded) + len(counterfactual)
            == len(training)
            + len(development)
            + len(queries)
            + len(evaluation)
            + len(counterfactual),
        },
        "extraction": extraction,
        "package": {
            "unseen_oracle_accuracy": transition_accuracy(transition, evaluation),
            "order_counterfactual_oracle_accuracy": transition_accuracy(transition, counterfactual),
            "unseen_source_agreement": sum(
                left == right for left, right in zip(package_predictions, source_predictions)
            )
            / len(evaluation),
            "counterfactual_source_agreement": sum(
                left == right
                for left, right in zip(
                    counterfactual_package_predictions, counterfactual_source_predictions
                )
            )
            / len(counterfactual),
        },
        "hardware": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "claim_ceiling": "PUBLIC_PREFLIGHT_NOT_CERTIFICATION",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_once(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--formulation",
        choices=("v1_full_vocabulary", "v2_canonical_curriculum"),
        default="v1_full_vocabulary",
    )
    args = parser.parse_args()
    result = run(Path(args.output).resolve(), formulation=args.formulation)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
