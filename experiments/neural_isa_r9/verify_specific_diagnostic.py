"""Fail-closed recomputation of the R9 capability-specific diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
    generate_rows,
    public_capabilities,
)
from experiments.native_transfer_r8.native_host import (
    SPECS,
    FrozenNeuralHost,
    sha256_file,
    tensor_state_sha256,
)

from .run_specific_diagnostic import _bind_implementation, _bind_r8, _json, _resolve


class R9VerificationError(RuntimeError):
    """Raised when Gate A evidence cannot be independently recomputed."""


def _evidence(value: Mapping[str, Any], label: str) -> None:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
        raise R9VerificationError(f"stale evidence hash: {label}")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        payload = path.read_bytes()
        rows = [json.loads(line) for line in payload.splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise R9VerificationError(f"raw rows unavailable: {path}") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise R9VerificationError("raw observation rows missing or malformed")
    if payload != b"".join(canonical_json_bytes(row) for row in rows):
        raise R9VerificationError("raw observations are not canonical")
    return rows


def _bootstrap(values: list[int], *, seed: int, replicates: int) -> dict[str, float]:
    if not values or replicates < 1000:
        raise R9VerificationError("bootstrap depth changed")
    generator = random.Random(seed)
    estimates = sorted(
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(replicates)
    )
    return {
        "point": sum(values) / len(values),
        "lower_95": estimates[math.floor(0.025 * replicates)],
        "upper_95": estimates[min(replicates - 1, math.ceil(0.975 * replicates) - 1)],
    }


def _teacher_distribution(latent: torch.Tensor, row: Mapping[str, Any]) -> torch.Tensor:
    state = F.one_hot(torch.tensor(int(row["start"])), num_classes=8).float()
    for operation in row["program"]:
        state = state @ latent[int(operation)].float()
    return state / state.sum().clamp_min(torch.finfo(torch.float32).tiny)


def verify(config_path: Path, run_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config = _json(config_path)
    latent_path, sealed_r8 = _bind_r8(root, config)
    implementation = _bind_implementation(root, config)
    receipt_path = run_dir / "receipt.json"
    receipt = _json(receipt_path)
    _evidence(receipt, "R9 Gate A run")
    if receipt.get("config_sha256") != sha256_file(config_path):
        raise R9VerificationError("run is not bound to this preregistration")
    reference = config["r8_reference"]
    expected_bindings = {
        "r8_config_sha256": sha256_file(_resolve(root, str(reference["config"]))),
        "r8_extraction_receipt_sha256": sha256_file(
            _resolve(root, str(reference["extraction_receipt"]))
        ),
        "r8_canonical_latents_sha256": sha256_file(latent_path),
    }
    if any(receipt.get(key) != value for key, value in expected_bindings.items()):
        raise R9VerificationError("R8 evidence binding changed")
    if receipt.get("implementation_sha256", {}) != implementation:
        raise R9VerificationError("implementation binding changed")
    if (
        receipt.get("capability_specific_weights_allowed") is not True
        or receipt.get("universal_decoder_claim_allowed") is not False
        or receipt.get("recipient_optimizer_steps") != 0
    ):
        raise R9VerificationError("diagnostic claim boundary changed")

    backend = receipt.get("backend")
    if not isinstance(backend, dict):
        raise R9VerificationError("backend receipt missing")
    backend_path = run_dir / str(backend.get("path"))
    if not backend_path.is_file() or sha256_file(backend_path) != backend.get("sha256"):
        raise R9VerificationError("backend artifact missing or changed")
    backend_state_hash = tensor_state_sha256(load_file(str(backend_path), device="cpu"))
    if (
        backend_state_hash != backend.get("state_sha256_before_evaluation")
        or backend_state_hash != backend.get("state_sha256_after_evaluation")
    ):
        raise R9VerificationError("backend state identity changed")

    gate = config["gate_a"]
    if receipt.get("host_key") != gate["host"] or receipt.get("host_revision") != SPECS[
        str(gate["host"])
    ].revision:
        raise R9VerificationError("host identity changed")
    host = FrozenNeuralHost(SPECS[str(gate["host"])], device="cuda")
    if (
        host.model_state_sha256 != receipt.get("host_model_state_sha256_before")
        or host.model_state_sha256 != receipt.get("host_model_state_sha256_after")
        or list(host.target_token_ids) != receipt.get("target_token_ids")
    ):
        raise R9VerificationError("live recipient identity or canonical output map changed")
    target_ids = list(host.target_token_ids)
    del host
    torch.cuda.empty_cache()

    r8_config = _json(_resolve(root, str(reference["config"])))
    split = r8_config["splits"]
    capabilities = public_capabilities(
        int(split["development_seed"]),
        split="development",
        count=int(split["development_capabilities"]),
    )
    capability = capabilities[int(gate["development_capability_index"])]
    if receipt.get("capability_id") != capability.capability_id:
        raise R9VerificationError("capability identity changed")
    expected_rows = generate_rows(
        capability,
        split="r9_specific_evaluation",
        rows=int(gate["evaluation_rows"]),
        depths=gate["evaluation_depths"],
        seed=int(gate["seed"]) + 2,
    )
    expected_by_id = {str(row["row_id"]): row for row in expected_rows}
    tensors = load_file(str(latent_path), device="cpu")
    after = tensors["development_after"][int(gate["development_capability_index"])].float()
    raw_info = receipt.get("observations")
    if not isinstance(raw_info, dict):
        raise R9VerificationError("observation receipt missing")
    raw_path = run_dir / str(raw_info.get("path"))
    if not raw_path.is_file() or sha256_file(raw_path) != raw_info.get("sha256"):
        raise R9VerificationError("raw observation artifact missing or changed")
    rows = _jsonl(raw_path)
    conditions = tuple(str(value) for value in gate["conditions"])
    expected_count = len(expected_rows) * len(conditions)
    if len(rows) != expected_count or raw_info.get("rows") != expected_count:
        raise R9VerificationError("observation depth changed")

    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("row_id"))
        condition = str(row.get("condition"))
        key = (row_id, condition)
        expected = expected_by_id.get(row_id)
        if expected is None or condition not in conditions or key in keyed:
            raise R9VerificationError("unknown or duplicate observation")
        if (
            row.get("capability_id") != capability.capability_id
            or row.get("prompt_sha256") != expected["prompt_sha256"]
            or row.get("depth") != expected["depth"]
            or row.get("flavor") != expected["flavor"]
        ):
            raise R9VerificationError("observation metadata changed")
        probabilities = row.get("canonical_output_probabilities")
        teacher = row.get("teacher_canonical_probabilities")
        if not isinstance(probabilities, list) or not isinstance(teacher, list):
            raise R9VerificationError("probability evidence missing")
        if len(probabilities) != 8 or len(teacher) != 8:
            raise R9VerificationError("probability width changed")
        values = [float(value) for value in probabilities]
        teacher_values = [float(value) for value in teacher]
        if (
            any(not math.isfinite(value) for value in values + teacher_values)
            or abs(sum(values) - 1.0) > 1e-4
            or abs(sum(teacher_values) - 1.0) > 1e-4
        ):
            raise R9VerificationError("probability row invalid")
        expected_teacher = _teacher_distribution(after, expected)
        if max(abs(left - float(right)) for left, right in zip(teacher_values, expected_teacher)) > 1e-6:
            raise R9VerificationError("teacher distribution is unrecomputable")
        expected_tv = 0.5 * sum(abs(left - right) for left, right in zip(values, teacher_values))
        if abs(float(row.get("teacher_recipient_tv")) - expected_tv) > 1e-6:
            raise R9VerificationError("stored distribution distance changed")
        if not isinstance(row.get("prediction_token_id"), int):
            raise R9VerificationError("prediction token missing")
        keyed[key] = row

    training_accuracy: float | None = None
    gates = config["gates"]
    if "training_accuracy_minimum" in gates:
        training_rows = generate_rows(
            capability,
            split="r9_specific_train",
            rows=int(gate["train_rows"]),
            depths=gate["train_depths"],
            seed=int(gate["seed"]) + 1,
        )
        training_by_id = {str(row["row_id"]): row for row in training_rows}
        training_info = receipt.get("training_observations")
        if not isinstance(training_info, dict):
            raise R9VerificationError("training-fit observations missing")
        training_path = run_dir / str(training_info.get("path"))
        if (
            not training_path.is_file()
            or sha256_file(training_path) != training_info.get("sha256")
        ):
            raise R9VerificationError("training-fit observations missing or changed")
        training_raw = _jsonl(training_path)
        if len(training_raw) != len(training_rows) or training_info.get("rows") != len(
            training_rows
        ):
            raise R9VerificationError("training-fit observation depth changed")
        training_correct = 0
        seen_training = set()
        for row in training_raw:
            row_id = str(row.get("row_id"))
            expected = training_by_id.get(row_id)
            if expected is None or row_id in seen_training or row.get("condition") != "AFTER":
                raise R9VerificationError("unknown or duplicate training-fit observation")
            seen_training.add(row_id)
            if (
                row.get("capability_id") != capability.capability_id
                or row.get("prompt_sha256") != expected["prompt_sha256"]
                or row.get("depth") != expected["depth"]
                or row.get("flavor") != expected["flavor"]
            ):
                raise R9VerificationError("training-fit observation metadata changed")
            probabilities = row.get("canonical_output_probabilities")
            teacher = row.get("teacher_canonical_probabilities")
            if (
                not isinstance(probabilities, list)
                or not isinstance(teacher, list)
                or len(probabilities) != 8
                or len(teacher) != 8
            ):
                raise R9VerificationError("training-fit probabilities missing")
            expected_teacher = _teacher_distribution(after, expected)
            if max(
                abs(float(left) - float(right))
                for left, right in zip(teacher, expected_teacher)
            ) > 1e-6:
                raise R9VerificationError("training-fit teacher distribution changed")
            target = target_ids[int(expected["answer"])]
            training_correct += int(int(row.get("prediction_token_id")) == target)
        training_accuracy = training_correct / len(training_rows)

    metrics: dict[str, dict[str, float]] = {}
    for condition in conditions:
        condition_rows = [keyed[(row_id, condition)] for row_id in sorted(expected_by_id)]
        correct = [
            int(int(row["prediction_token_id"]) == target_ids[int(expected_by_id[str(row["row_id"])]["answer"])])
            for row in condition_rows
        ]
        tv_values = [float(row["teacher_recipient_tv"]) for row in condition_rows]
        metrics[condition] = {
            "rows": len(condition_rows),
            "accuracy": sum(correct) / len(correct),
            "mean_teacher_recipient_tv": sum(tv_values) / len(tv_values),
            "maximum_teacher_recipient_tv": max(tv_values),
        }
    paired = []
    for row_id, expected in sorted(expected_by_id.items()):
        target = target_ids[int(expected["answer"])]
        paired.append(
            int(int(keyed[(row_id, "AFTER")]["prediction_token_id"]) == target)
            - int(int(keyed[(row_id, "BASE")]["prediction_token_id"]) == target)
        )
    bootstrap = _bootstrap(
        paired,
        seed=int(gate["seed"]) + 9001,
        replicates=int(gates["bootstrap_replicates"]),
    )
    after_accuracy = metrics["AFTER"]["accuracy"]
    base_accuracy = metrics["BASE"]["accuracy"]
    negative_names = [name for name in conditions if name not in {"BASE", "AFTER"}]
    negative_control_pass = all(
        metrics[name]["accuracy"] <= base_accuracy + float(gates["negative_control_gain_maximum"])
        for name in negative_names
    )
    gate_a_pass = (
        (training_accuracy is None or training_accuracy >= float(gates["training_accuracy_minimum"]))
        and after_accuracy >= float(gates["after_accuracy_minimum"])
        and after_accuracy - base_accuracy >= float(gates["after_minus_base_minimum"])
        and bootstrap["lower_95"] > 0
        and negative_control_pass
    )
    exact_output_equivalence = after_accuracy == float(gates["lossless_output_accuracy"])
    distribution_equivalence = (
        metrics["AFTER"]["maximum_teacher_recipient_tv"]
        <= float(gates["distribution_tv_maximum_for_equivalence"])
    )
    result = {
        "format": "abi-neural-isa-r9-gate-a-verification/1",
        "status": "PASS_CAPABILITY_SPECIFIC_EXPRESSIVITY" if gate_a_pass else "FAIL_CAPABILITY_SPECIFIC_EXPRESSIVITY",
        "exact_question_answer": "YES_GATE_A_ONLY" if gate_a_pass else "NO_GATE_A",
        "scope": "capability-specific Pythia recipient realization diagnostic; not a universal backend",
        "metrics": metrics,
        "training_fit_accuracy": training_accuracy,
        "after_minus_base_bootstrap": bootstrap,
        "gates": {
            "gate_a_capability_specific_expressivity": gate_a_pass,
            "training_fit": training_accuracy is None
            or training_accuracy >= float(gates["training_accuracy_minimum"]),
            "negative_control_causality": negative_control_pass,
            "recipient_frozen": True,
            "backend_frozen_during_evaluation": True,
            "exact_output_and_decision_equivalence": exact_output_equivalence,
            "distribution_equivalence": distribution_equivalence,
            "universal_capability_blind_backend": False,
            "heterogeneous_host_transfer": False,
        },
        "r8_controlling_result": sealed_r8["status"],
        "trusted_scientific_booleans_consumed": 0,
        "decision": config["decision_rule"]["gate_a_pass" if gate_a_pass else "gate_a_fail"],
        "claim_boundary": "A positive Gate A isolates recipient realization as feasible. It does not prove package sufficiency, capability-blind transfer, zero-shot transplantation, or lossless ABI.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        print(json.dumps({"status": "FAIL_CLOSED", "error": f"immutable output exists: {output}"}, indent=2))
        return 2
    try:
        value = verify(Path(args.config).resolve(), Path(args.run_dir).resolve())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    except (OSError, ValueError, R9VerificationError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
