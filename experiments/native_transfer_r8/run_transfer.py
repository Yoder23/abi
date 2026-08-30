"""Prepare held-out source training, sealed packages, and private R8 evaluations."""

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
from .extract_capability import extract_atomic_latent, write_package_once
from .native_host import SPECS, FrozenNeuralHost, NativeHostError, sha256_file
from .train_source import fit_prefixes


class TransferPreparationError(RuntimeError):
    """Raised when held-out source/package temporal order is invalid."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TransferPreparationError(f"expected JSON object: {path}")
    return value


def _evidence_valid(value: Mapping[str, Any]) -> bool:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    return stored == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _capabilities(private: Mapping[str, Any]) -> list[OpaqueCapability]:
    result = []
    for row in private.get("capabilities", []):
        result.append(
            OpaqueCapability(
                capability_id=str(row["capability_id"]),
                offsets=tuple(int(value) for value in row["offsets"]),
                seed_commitment=str(row["seed_commitment"]),
            )
        )
    if not result:
        raise TransferPreparationError("private held-out capability set is empty")
    return result


def _permuted_rows(rows: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    result = []
    for capability_index, capability_rows in enumerate(rows):
        shift = 1 + capability_index % 7
        result.append(
            [{**row, "answer": (int(row["answer"]) + shift) % 8} for row in capability_rows]
        )
    return result


def _source_observations(
    host: FrozenNeuralHost,
    rows: list[dict[str, Any]],
    prefix: torch.Tensor | None,
    *,
    capability_id: str,
    condition: str,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    correct = 0
    nll = 0.0
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=prefix)
            targets = host.target_ids([int(row["answer"]) for row in batch])
            predicted = logits.argmax(dim=-1)
            probabilities = host.canonical_probabilities(logits)
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
        raise TransferPreparationError(f"immutable held-out source output exists: {output}")
    config = _json(config_path)
    freeze_path = campaign_root / "freeze_receipt.json"
    reveal_path = campaign_root / "heldout_reveal.json"
    private_path = campaign_root / "evaluator_private/capabilities.json"
    freeze, reveal, private = (_json(path) for path in (freeze_path, reveal_path, private_path))
    if not all(_evidence_valid(value) for value in (freeze, reveal, private)):
        raise TransferPreparationError("freeze/reveal/private evidence hash changed")
    if (
        reveal.get("freeze_receipt_sha256") != sha256_file(freeze_path)
        or private.get("freeze_receipt_sha256") != sha256_file(freeze_path)
        or int(reveal["created_unix_time_ns"]) <= int(freeze["created_unix_time_ns"])
    ):
        raise TransferPreparationError("held-out reveal does not postdate the frozen components")
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
    steps = int(config["training"]["source_steps"]) * len(capabilities)
    prefixes, training = fit_prefixes(
        host,
        capabilities,
        train_rows,
        prefix_length=int(config["training"]["source_prefix_length"]),
        steps=steps,
        learning_rate=float(config["training"]["source_learning_rate"]),
        batch_size=int(config["training"]["batch_size"]),
        seed=int(config["training"]["seed"]) + 1,
    )
    permuted_prefixes, permuted_training = fit_prefixes(
        host,
        capabilities,
        _permuted_rows(train_rows),
        prefix_length=int(config["training"]["source_prefix_length"]),
        steps=steps,
        learning_rate=float(config["training"]["source_learning_rate"]),
        batch_size=int(config["training"]["batch_size"]),
        seed=int(config["training"]["seed"]) + 2,
    )
    output.mkdir(parents=True)
    prefix_path = output / "heldout_source_prefixes.safetensors"
    save_file(
        {
            "after": prefixes.contiguous(),
            "permuted_delta": permuted_prefixes.contiguous(),
        },
        str(prefix_path),
    )
    packages = []
    evaluations = []
    source_rows_out = []
    before_latent = extract_atomic_latent(host, None)
    before_state = hashlib.sha256(
        (host.model_state_sha256 + ":no-prefix").encode("ascii")
    ).hexdigest()
    packages_root = output / "packages"
    worker_root = output / "worker_inputs"
    evaluator_root = campaign_root / "evaluator_private/evaluation"
    if evaluator_root.exists():
        raise TransferPreparationError("private held-out evaluation already exists")
    for index, capability in enumerate(capabilities):
        capability_dir = packages_root / capability.capability_id
        after_prefix = prefixes[index].to(host.device)
        permuted_prefix = permuted_prefixes[index].to(host.device)
        after_latent = extract_atomic_latent(host, after_prefix)
        permuted_latent = extract_atomic_latent(host, permuted_prefix)
        after_state = hashlib.sha256(after_prefix.detach().cpu().numpy().tobytes()).hexdigest()
        permuted_state = hashlib.sha256(
            permuted_prefix.detach().cpu().numpy().tobytes()
        ).hexdigest()
        package_rows = {
            "before": write_package_once(
                capability_dir / "before.abipkg",
                before_latent,
                capability_id=capability.capability_id,
                reveal_commitment_sha256=str(reveal["secret_sha256"]),
                source_before_sha256=before_state,
                source_after_sha256=before_state,
            ),
            "after": write_package_once(
                capability_dir / "after.abipkg",
                after_latent,
                capability_id=capability.capability_id,
                reveal_commitment_sha256=str(reveal["secret_sha256"]),
                source_before_sha256=before_state,
                source_after_sha256=after_state,
            ),
            "permuted_teacher_delta": write_package_once(
                capability_dir / "permuted_teacher_delta.abipkg",
                permuted_latent,
                capability_id=capability.capability_id,
                reveal_commitment_sha256=str(reveal["secret_sha256"]),
                source_before_sha256=before_state,
                source_after_sha256=permuted_state,
            ),
        }
        packages.append({"capability_id": capability.capability_id, "packages": package_rows})

        # Private evaluation is generated only after the immutable packages exist.
        evaluation_rows = generate_rows(
            capability,
            split="heldout_evaluation",
            rows=int(split["evaluation_rows_per_capability"]),
            depths=config["capability_family"]["evaluation_depths"],
            seed=int(config["training"]["seed"]) + 16001 * index,
        )
        write_jsonl_once(evaluator_root / f"{capability.capability_id}.jsonl", evaluation_rows)
        write_jsonl_once(
            worker_root / f"{capability.capability_id}.jsonl",
            worker_rows(evaluation_rows),
        )
        raw_before, source_before = _source_observations(
            host,
            evaluation_rows,
            None,
            capability_id=capability.capability_id,
            condition="T_BEFORE",
            batch_size=int(config["training"]["batch_size"]),
        )
        raw_after, source_after = _source_observations(
            host,
            evaluation_rows,
            after_prefix,
            capability_id=capability.capability_id,
            condition="T_AFTER",
            batch_size=int(config["training"]["batch_size"]),
        )
        raw_permuted, source_permuted = _source_observations(
            host,
            evaluation_rows,
            permuted_prefix,
            capability_id=capability.capability_id,
            condition="T_PERMUTED_DELTA",
            batch_size=int(config["training"]["batch_size"]),
        )
        source_rows_out.extend((*raw_before, *raw_after, *raw_permuted))
        evaluations.append(
            {
                "capability_id": capability.capability_id,
                "before": source_before,
                "after": source_after,
                "permuted_teacher_delta": source_permuted,
                "private_evaluation_sha256": sha256_file(
                    evaluator_root / f"{capability.capability_id}.jsonl"
                ),
                "worker_input_sha256": sha256_file(
                    worker_root / f"{capability.capability_id}.jsonl"
                ),
            }
        )
    source_observations_path = output / "source_observations.jsonl"
    source_observations_path.write_bytes(
        b"".join(canonical_json_bytes(row) for row in source_rows_out)
    )
    receipt = {
        "format": "abi-native-transfer-r8-heldout-source-and-packages/1",
        "status": "HELDOUT_SOURCE_TRAINED_PACKAGES_SEALED",
        "config_sha256": sha256_file(config_path),
        "freeze_receipt_sha256": sha256_file(freeze_path),
        "reveal_receipt_sha256": sha256_file(reveal_path),
        "source_model_state_sha256": host.model_state_sha256,
        "source_model_parameters_trainable": 0,
        "source_capability_parameters_trained": int(prefixes.numel()),
        "source_training": training,
        "permuted_delta_training": permuted_training,
        "prefixes": {"path": prefix_path.name, "sha256": sha256_file(prefix_path)},
        "packages": packages,
        "source_evaluation": evaluations,
        "source_observations": {
            "path": source_observations_path.name,
            "sha256": sha256_file(source_observations_path),
            "rows": len(source_rows_out),
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
    del host
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
    except (TransferPreparationError, NativeHostError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
