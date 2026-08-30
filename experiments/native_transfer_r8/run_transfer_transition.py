"""Train held-out neural-transition sources, seal packages, then reveal evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .capability_generator import (
    OpaqueCapability,
    canonical_json_bytes,
    generate_rows,
    worker_rows,
    write_jsonl_once,
)
from .extract_capability import write_package_once
from .extract_capability_transition import extract_atomic_latent
from .native_host import SPECS, FrozenNeuralHost, NativeHostError, sha256_file
from .source_transition import (
    NeuralTransitionSource,
    SourceTransitionError,
    controller_state,
    ensure_source_base_frozen,
    load_controller_state,
    pack_controller_states,
)
from .train_source_transition import TransitionTrainingError, fit_one


class TransitionTransferError(RuntimeError):
    """Raised when held-out transition source custody is invalid."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TransitionTransferError(f"expected JSON object: {path}")
    return value


def _evidence_valid(value: Mapping[str, Any]) -> bool:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    return stored == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _capabilities(private: Mapping[str, Any]) -> list[OpaqueCapability]:
    result = [
        OpaqueCapability(
            capability_id=str(row["capability_id"]),
            offsets=tuple(int(value) for value in row["offsets"]),
            seed_commitment=str(row["seed_commitment"]),
        )
        for row in private.get("capabilities", [])
    ]
    if not result:
        raise TransitionTransferError("private held-out capability set is empty")
    return result


def _permuted(rows: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    shift = 1 + index % 7
    return [{**row, "answer": (int(row["answer"]) + shift) % 8} for row in rows]


@torch.inference_mode()
def _observations(
    controller: NeuralTransitionSource,
    rows: list[dict[str, Any]],
    *,
    capability_id: str,
    condition: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    correct = 0
    nll = 0.0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        logits = controller.logits(
            [str(row["prompt"]) for row in batch],
            [int(row["start"]) for row in batch],
            [[int(value) for value in row["program"]] for row in batch],
        )
        targets = controller.host.target_ids([int(row["answer"]) for row in batch])
        predicted = logits.argmax(dim=-1)
        probabilities = controller.host.canonical_probabilities(logits)
        correct += int((predicted == targets).sum().item())
        nll += float(F.cross_entropy(logits, targets, reduction="sum").item())
        for row, token, probability in zip(batch, predicted, probabilities):
            output.append(
                {
                    "capability_id": capability_id,
                    "row_id": row["row_id"],
                    "prompt_sha256": row["prompt_sha256"],
                    "condition": condition,
                    "prediction_token_id": int(token.item()),
                    "canonical_output_probabilities": [
                        float(value) for value in probability.cpu().tolist()
                    ],
                }
            )
    return output, {
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "mean_nll": nll / len(rows),
    }


def prepare(config_path: Path, campaign_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise TransitionTransferError(f"immutable held-out transition output exists: {output}")
    config = _json(config_path)
    freeze_path = campaign_root / "freeze_receipt.json"
    reveal_path = campaign_root / "heldout_reveal.json"
    private_path = campaign_root / "evaluator_private/capabilities.json"
    freeze, reveal, private = (_json(path) for path in (freeze_path, reveal_path, private_path))
    if not all(_evidence_valid(value) for value in (freeze, reveal, private)):
        raise TransitionTransferError("freeze/reveal/private evidence hash changed")
    if (
        reveal.get("freeze_receipt_sha256") != sha256_file(freeze_path)
        or int(reveal["created_unix_time_ns"]) <= int(freeze["created_unix_time_ns"])
    ):
        raise TransitionTransferError("held-out reveal does not postdate freeze")
    capabilities = _capabilities(private)
    split = config["splits"]
    train_rows = [
        generate_rows(
            capability,
            split="source_train",
            rows=int(split["source_train_rows_per_capability"]),
            depths=config["capability_family"]["source_train_depths"],
            seed=int(config["training"]["seed"]) + 8009 * index,
        )
        for index, capability in enumerate(capabilities)
    ]
    host = FrozenNeuralHost(SPECS["source"], device=config["training"]["device"])
    controller = NeuralTransitionSource(host, seed=int(config["training"]["seed"])).to(
        host.device
    )
    controller.reset()
    before_latent = extract_atomic_latent(controller)
    before_state = controller.state_sha256()
    after_states = []
    permuted_states = []
    training_receipts = []
    package_receipts = []
    output.mkdir(parents=True)
    packages_root = output / "packages"
    # Phase A: train and seal every package while private evaluation is absent.
    for index, capability in enumerate(capabilities):
        real_training = fit_one(
            controller,
            host,
            train_rows[index],
            steps=int(config["training"]["source_steps_per_capability"]),
            learning_rate=float(config["training"]["source_transition_learning_rate"]),
            batch_size=int(config["training"]["batch_size"]),
            seed=int(config["training"]["seed"]) + 17011 * index,
        )
        after_state = controller_state(controller)
        after_states.append(after_state)
        after_state_sha = controller.state_sha256()
        after_latent = extract_atomic_latent(controller)
        permuted_training = fit_one(
            controller,
            host,
            _permuted(train_rows[index], index),
            steps=int(config["training"]["source_steps_per_capability"]),
            learning_rate=float(config["training"]["source_transition_learning_rate"]),
            batch_size=int(config["training"]["batch_size"]),
            seed=int(config["training"]["seed"]) + 19001 * index,
        )
        permuted_state = controller_state(controller)
        permuted_states.append(permuted_state)
        permuted_state_sha = controller.state_sha256()
        permuted_latent = extract_atomic_latent(controller)
        directory = packages_root / capability.capability_id
        packages = {
            "before": write_package_once(
                directory / "before.abipkg",
                before_latent,
                capability_id=capability.capability_id,
                reveal_commitment_sha256=str(reveal["secret_sha256"]),
                source_before_sha256=before_state,
                source_after_sha256=before_state,
            ),
            "after": write_package_once(
                directory / "after.abipkg",
                after_latent,
                capability_id=capability.capability_id,
                reveal_commitment_sha256=str(reveal["secret_sha256"]),
                source_before_sha256=before_state,
                source_after_sha256=after_state_sha,
            ),
            "permuted_teacher_delta": write_package_once(
                directory / "permuted_teacher_delta.abipkg",
                permuted_latent,
                capability_id=capability.capability_id,
                reveal_commitment_sha256=str(reveal["secret_sha256"]),
                source_before_sha256=before_state,
                source_after_sha256=permuted_state_sha,
            ),
        }
        package_receipts.append(
            {"capability_id": capability.capability_id, "packages": packages}
        )
        training_receipts.append(
            {
                "capability_id": capability.capability_id,
                "after": real_training,
                "permuted_teacher_delta": permuted_training,
                "after_state_sha256": after_state_sha,
                "permuted_state_sha256": permuted_state_sha,
            }
        )
    state_path = output / "heldout_source_transition_states.safetensors"
    packed = {
        **{
            "after/" + key: value
            for key, value in pack_controller_states(after_states).items()
        },
        **{
            "permuted/" + key: value
            for key, value in pack_controller_states(permuted_states).items()
        },
    }
    save_file(packed, str(state_path))

    # Phase B: only after all immutable packages exist, generate private rows.
    evaluator_root = campaign_root / "evaluator_private/evaluation"
    worker_root = output / "worker_inputs"
    if evaluator_root.exists():
        raise TransitionTransferError("private evaluation existed before package sealing")
    raw_rows = []
    evaluations = []
    for index, capability in enumerate(capabilities):
        rows = generate_rows(
            capability,
            split="heldout_evaluation",
            rows=int(split["evaluation_rows_per_capability"]),
            depths=config["capability_family"]["evaluation_depths"],
            seed=int(config["training"]["seed"]) + 16001 * index,
        )
        write_jsonl_once(evaluator_root / f"{capability.capability_id}.jsonl", rows)
        write_jsonl_once(worker_root / f"{capability.capability_id}.jsonl", worker_rows(rows))
        controller.reset()
        before_rows, before_metrics = _observations(
            controller,
            rows,
            capability_id=capability.capability_id,
            condition="T_BEFORE",
            batch_size=int(config["training"]["batch_size"]),
        )
        load_controller_state(controller, after_states[index])
        after_rows, after_metrics = _observations(
            controller,
            rows,
            capability_id=capability.capability_id,
            condition="T_AFTER",
            batch_size=int(config["training"]["batch_size"]),
        )
        load_controller_state(controller, permuted_states[index])
        permuted_rows, permuted_metrics = _observations(
            controller,
            rows,
            capability_id=capability.capability_id,
            condition="T_PERMUTED_DELTA",
            batch_size=int(config["training"]["batch_size"]),
        )
        raw_rows.extend((*before_rows, *after_rows, *permuted_rows))
        evaluations.append(
            {
                "capability_id": capability.capability_id,
                "before": before_metrics,
                "after": after_metrics,
                "permuted_teacher_delta": permuted_metrics,
                "private_evaluation_sha256": sha256_file(
                    evaluator_root / f"{capability.capability_id}.jsonl"
                ),
                "worker_input_sha256": sha256_file(
                    worker_root / f"{capability.capability_id}.jsonl"
                ),
            }
        )
    raw_path = output / "source_observations.jsonl"
    raw_path.write_bytes(b"".join(canonical_json_bytes(row) for row in raw_rows))
    ensure_source_base_frozen(host, host.model_state_sha256)
    receipt = {
        "format": "abi-native-transfer-r8-heldout-transition-source-and-packages/1",
        "config_sha256": sha256_file(config_path),
        "freeze_receipt_sha256": sha256_file(freeze_path),
        "reveal_receipt_sha256": sha256_file(reveal_path),
        "source_model_state_sha256": host.model_state_sha256,
        "source_model_parameters_trainable": 0,
        "source_capability_parameters_trained": sum(
            parameter.numel() for parameter in controller.parameters()
        ),
        "source_training": training_receipts,
        "states": {"path": state_path.name, "sha256": sha256_file(state_path)},
        "packages": package_receipts,
        "source_evaluation": evaluations,
        "source_observations": {
            "path": raw_path.name,
            "sha256": sha256_file(raw_path),
            "rows": len(raw_rows),
            "target_token_ids": host.target_token_ids,
        },
        "package_sealed_before_private_evaluation_generation": True,
        "teacher_available_to_recipient_worker": False,
        "recipient_optimizer_steps": 0,
        "created_unix_time_ns": time.time_ns(),
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    (output / "receipt.json").write_bytes(
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        value = prepare(
            Path(args.config).resolve(),
            Path(args.campaign_root).resolve(),
            Path(args.output).resolve(),
        )
    except (
        TransitionTransferError,
        TransitionTrainingError,
        SourceTransitionError,
        NativeHostError,
    ) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
