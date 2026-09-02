"""Execute the preregistered R14 non-exhaustive capability campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from experiments.foreign_teacher_r12.custody import verify_r11_freeze
from experiments.native_isa_r11.core import (
    sha256_bytes,
    transition_accuracy,
    transition_bytes,
    write_package_once,
)
from experiments.native_transfer_r8.native_host import (
    SPECS,
    FrozenNeuralHost,
    GenericRecipientAdapterSet,
)

from .capability import heldout_capabilities
from .core import (
    R14Error,
    behavior_receipt,
    capability_rows,
    evidence_hash,
    json_object,
    sha256_file,
    source_metrics,
    write_json_once,
    write_jsonl_once,
)
from .extractor import extract_transition, extractor_spec
from .source import observe_queries, train_fixed_schedule


def _adapter_state_sha256(state: dict[str, torch.Tensor]) -> str:
    return hashlib.sha256(
        b"".join(
            key.encode() + state[key].detach().cpu().numpy().tobytes() for key in sorted(state)
        )
    ).hexdigest()


def _tag_observations(
    observations: list[dict[str, Any]], *, capability_id: str, split: str
) -> list[dict[str, Any]]:
    return [
        {"capability_id": capability_id, "split": split, **observation}
        for observation in observations
    ]


def _transition_predictions(transition: torch.Tensor, rows: list[dict[str, Any]]) -> list[int]:
    predictions = []
    for row in rows:
        state = torch.nn.functional.one_hot(torch.tensor(int(row["start"])), num_classes=8).float()
        for operator in row["program"]:
            state = torch.matmul(state, transition[int(operator)])
        predictions.append(int(state.argmax()))
    return predictions


def _source_gate(config: dict[str, Any], sources: list[dict[str, Any]]) -> bool:
    chance = float(config["gates"]["source_chance_accuracy"])
    return all(
        source["base_state_sha256_before"] == source["base_state_sha256_after"]
        and source["training"]["schedule_selection_used_heldout"] is False
        and source["query"]["wilson_95_lower"] > chance
        and source["evaluation"]["wilson_95_lower"] > chance
        and source["counterfactual"]["wilson_95_lower"] > chance
        and source["package_unseen_oracle_accuracy"]
        == float(config["gates"]["package_oracle_accuracy"])
        and source["package_counterfactual_oracle_accuracy"]
        == float(config["gates"]["package_oracle_accuracy"])
        and source["extraction"]["atomic_observations"] == 0
        and source["extraction"]["observations"]
        == int(config["data"]["extractor_queries_per_capability"])
        and source["extraction"]["answers_consumed"] == 0
        and source["extraction"]["oracle_calls"] == 0
        and source["extraction"]["log_likelihood_margin"]
        > float(config["gates"]["extractor_log_likelihood_margin_minimum"])
        for source in sources
    )


def _recipient_gate(config: dict[str, Any], receipts: list[dict[str, Any]]) -> bool:
    if len(receipts) != len(config["recipient_hosts"]):
        return False
    capabilities = int(config["data"]["heldout_capabilities"])
    maximum = float(config["gates"]["negative_control_accuracy_maximum"])
    for receipt in receipts:
        host = receipt["host_receipt"]
        if (
            receipt.get("source_adapter_loaded") is not False
            or host["recipient_optimizer_steps"] != 0
            or host["model_state_sha256_before"] != host["model_state_sha256_after"]
            or host["codec_sha256_before"] != host["codec_sha256_after"]
            or receipt["summary"]["removal_conditions_equal_base"] is not True
        ):
            return False
        accuracy = receipt["summary"]["accuracy"]
        capability_ids = sorted({key.split("/", 1)[0] for key in accuracy})
        if len(capability_ids) != capabilities:
            return False
        for capability_id in capability_ids:
            if (
                accuracy[f"{capability_id}/AFTER"] != 1.0
                or accuracy[f"{capability_id}/RESTORED"] != 1.0
                or any(
                    accuracy[f"{capability_id}/{condition}"] > maximum
                    for condition in config["negative_conditions"]
                )
            ):
                return False
    return True


def run(config_path: Path, reveal_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R14 output exists: {output}")
    config = json_object(config_path)
    if config.get("status") != "PREREGISTERED_BEFORE_HELDOUT_REVEAL":
        raise R14Error("R14 preregistration status changed")
    reveal = json_object(reveal_path)
    try:
        secret = bytes.fromhex(str(reveal["secret_hex"]))
    except ValueError as exc:
        raise R14Error("held-out reveal is not hexadecimal") from exc
    if (
        len(secret) != 32
        or sha256_bytes(secret) != config["heldout_seed_commitment"]
        or reveal.get("commitment") != config["heldout_seed_commitment"]
    ):
        raise R14Error("held-out reveal does not match preregistration")
    root = config_path.parents[3]
    r11_freeze = verify_r11_freeze(root, config)
    capabilities = heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    rows_by_capability = capability_rows(config, capabilities)
    output.mkdir(parents=True)
    preserved_reveal = output / "heldout_reveal.json"
    shutil.copyfile(reveal_path, preserved_reveal)
    all_observations: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    transitions: list[torch.Tensor] = []
    package_items: list[dict[str, Any]] = []
    source_config = config["source_acquisition"]
    for index, capability in enumerate(capabilities):
        rows = rows_by_capability[index]
        host = FrozenNeuralHost(SPECS["qwen2"], device=str(source_config["device"]))
        base_state = host.model_state_sha256
        before_observations, _ = observe_queries(
            host,
            rows["evaluation"][: int(config["data"]["before_rows_per_capability"])],
            batch_size=int(source_config["evaluation_batch_size"]),
        )
        adapters = GenericRecipientAdapterSet(host, rank=int(source_config["lora_rank"]))
        training = train_fixed_schedule(
            host,
            adapters,
            rows["training"],
            None,
            steps=int(source_config["steps"]),
            learning_rate=float(source_config["learning_rate"]),
            batch_size=int(source_config["batch_size"]),
            evaluation_interval=int(source_config["checkpoint_interval"]),
            evaluation_batch_size=int(source_config["evaluation_batch_size"]),
            seed=int(source_config["seed"]) + 16001 * index,
            objective=str(source_config["objective"]),
        )
        state = adapters.state()
        state_sha = _adapter_state_sha256(state)
        query_observations, _ = observe_queries(
            host, rows["queries"], batch_size=int(source_config["evaluation_batch_size"])
        )
        evaluation_observations, _ = observe_queries(
            host,
            rows["evaluation"],
            batch_size=int(source_config["evaluation_batch_size"]),
        )
        counterfactual_observations, _ = observe_queries(
            host,
            rows["counterfactual"],
            batch_size=int(source_config["evaluation_batch_size"]),
        )
        extraction_started = time.perf_counter()
        transition, extraction = extract_transition(query_observations)
        extraction_seconds = time.perf_counter() - extraction_started
        adapter_path = output / "source_adapters" / f"{capability.capability_id}.safetensors"
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state, str(adapter_path))
        package = write_package_once(
            output / "packages",
            transition,
            {"teacher_before_sha256": base_state, "teacher_after_sha256": state_sha},
        )
        package["capability_id"] = capability.capability_id
        package_items.append(package)
        transitions.append(transition)
        evaluation_predictions = _transition_predictions(transition, rows["evaluation"])
        evaluation_source = [
            max(range(8), key=item["canonical_probabilities"].__getitem__)
            for item in evaluation_observations
        ]
        counterfactual_predictions = _transition_predictions(transition, rows["counterfactual"])
        counterfactual_source = [
            max(range(8), key=item["canonical_probabilities"].__getitem__)
            for item in counterfactual_observations
        ]
        receipt = {
            "capability_id": capability.capability_id,
            "model_id": host.spec.model_id,
            "revision": host.spec.revision,
            "architecture_family": host.spec.architecture_family,
            "target_token_ids": list(host.target_token_ids),
            "base_state_sha256_before": base_state,
            "base_state_sha256_after": adapters.base_state_sha256(),
            "adapter_inventory": adapters.inventory(),
            "adapter_state_sha256": state_sha,
            "adapter_artifact": {
                "path": adapter_path.relative_to(output).as_posix(),
                "bytes": adapter_path.stat().st_size,
                "sha256": sha256_file(adapter_path),
            },
            "before": source_metrics(
                before_observations,
                rows["evaluation"][: int(config["data"]["before_rows_per_capability"])],
            ),
            "query": source_metrics(query_observations, rows["queries"]),
            "evaluation": source_metrics(evaluation_observations, rows["evaluation"]),
            "counterfactual": source_metrics(counterfactual_observations, rows["counterfactual"]),
            "training": training,
            "extraction": extraction,
            "extraction_wall_seconds": extraction_seconds,
            "package_unseen_oracle_accuracy": transition_accuracy(transition, rows["evaluation"]),
            "package_counterfactual_oracle_accuracy": transition_accuracy(
                transition, rows["counterfactual"]
            ),
            "package_source_unseen_agreement": sum(
                left == right for left, right in zip(evaluation_predictions, evaluation_source)
            )
            / len(evaluation_predictions),
            "package_source_counterfactual_agreement": sum(
                left == right
                for left, right in zip(counterfactual_predictions, counterfactual_source)
            )
            / len(counterfactual_predictions),
        }
        source_receipts.append(receipt)
        all_observations.extend(
            _tag_observations(
                before_observations,
                capability_id=capability.capability_id,
                split="BEFORE",
            )
        )
        all_observations.extend(
            _tag_observations(
                query_observations,
                capability_id=capability.capability_id,
                split="QUERY",
            )
        )
        all_observations.extend(
            _tag_observations(
                evaluation_observations,
                capability_id=capability.capability_id,
                split="EVALUATION",
            )
        )
        all_observations.extend(
            _tag_observations(
                counterfactual_observations,
                capability_id=capability.capability_id,
                split="COUNTERFACTUAL",
            )
        )
        print(json.dumps(receipt, sort_keys=True), flush=True)
        del adapters, host
        gc.collect()
        torch.cuda.empty_cache()
    uniform = torch.full((3, 8, 8), 1.0 / 8.0)
    before_package = write_package_once(
        output / "packages",
        uniform,
        {
            "teacher_before_sha256": sha256_bytes(transition_bytes(uniform)),
            "teacher_after_sha256": sha256_bytes(transition_bytes(uniform)),
        },
    )
    manifest = {"before": before_package, "after": package_items}
    manifest_path = output / "package_manifest.json"
    write_json_once(manifest_path, manifest)
    observations_path = output / "source_observations.jsonl"
    write_jsonl_once(observations_path, all_observations)
    source_pass = _source_gate(config, source_receipts)
    recipient_receipts = []
    if source_pass:
        for host_key in config["recipient_hosts"]:
            worker_output = output / "recipients" / str(host_key)
            command = [
                sys.executable,
                "-m",
                "experiments.foreign_capability_r14.recipient_worker",
                "--config",
                str(config_path),
                "--reveal",
                str(preserved_reveal),
                "--manifest",
                str(manifest_path),
                "--host",
                str(host_key),
                "--output",
                str(worker_output),
            ]
            environment = dict(os.environ)
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            logs = output / "worker_logs"
            logs.mkdir(exist_ok=True)
            (logs / f"{host_key}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (logs / f"{host_key}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise R14Error(f"recipient worker failed: {host_key}")
            recipient_receipts.append(json_object(worker_output / "receipt.json"))
    receipt = {
        "format": "abi-r14-non-exhaustive-capability-extraction/1",
        "status": "PASS" if source_pass and _recipient_gate(config, recipient_receipts) else "FAIL",
        "claim_target": "NON_EXHAUSTIVE_FOREIGN_CAPABILITY_RECOVERY",
        "claim_ceiling": "NOT_LOSSLESS_TEACHER_CLONING_NOT_PREEXISTING_KNOWLEDGE",
        "config_sha256": sha256_file(config_path),
        "reveal_sha256": sha256_file(preserved_reveal),
        "manifest_sha256": sha256_file(manifest_path),
        "r11_freeze": r11_freeze,
        "extractor": extractor_spec(),
        "behavior": behavior_receipt(config),
        "data": {
            "capabilities": [item.capability_id for item in capabilities],
            "training_rows_per_capability": len(rows_by_capability[0]["training"]),
            "query_rows_per_capability": len(rows_by_capability[0]["queries"]),
            "evaluation_rows_per_capability": len(rows_by_capability[0]["evaluation"]),
            "counterfactual_rows_per_capability": len(rows_by_capability[0]["counterfactual"]),
            "recipient_rows_per_capability": len(rows_by_capability[0]["recipient"]),
            "all_split_overlap": 0,
        },
        "source_acquisitions": source_receipts,
        "source_observations": {
            "path": observations_path.name,
            "rows": len(all_observations),
            "sha256": sha256_file(observations_path),
        },
        "packages": manifest,
        "recipient_workers": recipient_receipts,
        "heldout": {
            "commitment": config["heldout_seed_commitment"],
            "reveal_path": preserved_reveal.name,
            "revealed_after_preregistration": True,
        },
        "hardware": {
            "source_device": str(source_config["device"]),
            "cuda_device_name": torch.cuda.get_device_name(0),
        },
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = run(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.output).resolve(),
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
