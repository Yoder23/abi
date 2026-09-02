"""Run the preregistered R13-B held-out capability-extraction campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from experiments.foreign_teacher_r12.custody import verify_r11_freeze
from experiments.foreign_teacher_r12.extractor import extract_transition, extractor_spec
from experiments.native_isa_r11.core import (
    sha256_bytes,
    transition_accuracy,
    transition_bytes,
    write_package_once,
)
from experiments.native_transfer_r8.capability_generator import (
    committed_heldout_capabilities,
)
from experiments.native_transfer_r8.native_host import (
    SPECS,
    FrozenNeuralHost,
    GenericRecipientAdapterSet,
    sha256_file,
)

from .core import (
    R13Error,
    capability_rows,
    evidence_hash,
    json_object,
    observe_source,
    write_json_once,
    write_jsonl_once,
)
from .source import train_until_atomic_stable


def _adapter_state_sha256(state: dict[str, torch.Tensor]) -> str:
    return hashlib.sha256(
        b"".join(key.encode() + state[key].numpy().tobytes() for key in sorted(state))
    ).hexdigest()


def _recipient_pass(config: dict[str, Any], receipts: list[dict[str, Any]]) -> bool:
    capabilities = int(config["data"]["heldout_capabilities"])
    after = float(config["gates"]["recipient_after_accuracy"])
    restored = float(config["gates"]["recipient_restored_accuracy"])
    maximum = float(config["gates"]["negative_control_accuracy_maximum"])
    expected_ids = None
    for receipt in receipts:
        if receipt.get("source_adapter_loaded") is not False:
            return False
        host = receipt["host_receipt"]
        if (
            host["recipient_optimizer_steps"] != 0
            or host["model_state_sha256_before"] != host["model_state_sha256_after"]
            or host["codec_sha256_before"] != host["codec_sha256_after"]
            or receipt["summary"]["removal_conditions_equal_base"] is not True
        ):
            return False
        accuracy = receipt["summary"]["accuracy"]
        capability_ids = sorted({key.split("/", 1)[0] for key in accuracy})
        expected_ids = capability_ids if expected_ids is None else expected_ids
        if capability_ids != expected_ids or len(capability_ids) != capabilities:
            return False
        for capability_id in capability_ids:
            if accuracy[f"{capability_id}/AFTER"] != after:
                return False
            if accuracy[f"{capability_id}/RESTORED"] != restored:
                return False
            if any(
                accuracy[f"{capability_id}/{condition}"] > maximum
                for condition in config["negative_conditions"]
            ):
                return False
    return len(receipts) == len(config["recipient_hosts"])


def run(config_path: Path, reveal_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R13Error(f"immutable R13 output exists: {output}")
    config = json_object(config_path)
    if config.get("status") != "PREREGISTERED_BEFORE_HELDOUT_REVEAL":
        raise R13Error("R13 preregistration status changed")
    reveal = json_object(reveal_path)
    try:
        secret = bytes.fromhex(str(reveal["secret_hex"]))
    except ValueError as exc:
        raise R13Error("held-out reveal is not hexadecimal") from exc
    if (
        len(secret) != 32
        or sha256_bytes(secret) != config["heldout_seed_commitment"]
        or reveal.get("commitment") != config["heldout_seed_commitment"]
    ):
        raise R13Error("held-out reveal does not match preregistration")
    root = config_path.parents[3]
    r11_freeze = verify_r11_freeze(root, config)
    capabilities = committed_heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    training_rows, evaluation_rows, atomic_rows = capability_rows(config, capabilities)
    output.mkdir(parents=True)
    preserved_reveal = output / "heldout_reveal.json"
    shutil.copyfile(reveal_path, preserved_reveal)
    source_observations: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    transitions: list[torch.Tensor] = []
    after_packages: list[dict[str, Any]] = []
    source_config = config["source_acquisition"]
    for index, capability in enumerate(capabilities):
        host = FrozenNeuralHost(SPECS["qwen2"], device=str(source_config["device"]))
        base_state = host.model_state_sha256
        before, rows = observe_source(
            host,
            evaluation_rows[index],
            capability_id=capability.capability_id,
            condition="BEFORE_EVALUATION",
            batch_size=int(source_config["evaluation_batch_size"]),
        )
        source_observations.extend(rows)
        before_atomic, rows = observe_source(
            host,
            atomic_rows[index],
            capability_id=capability.capability_id,
            condition="BEFORE_ATOMIC",
            batch_size=int(source_config["evaluation_batch_size"]),
        )
        source_observations.extend(rows)
        adapters = GenericRecipientAdapterSet(host, rank=int(source_config["lora_rank"]))
        state, training = train_until_atomic_stable(
            host,
            adapters,
            training_rows[index],
            atomic_rows[index],
            maximum_steps=int(source_config["maximum_steps"]),
            evaluation_interval=int(source_config["evaluation_interval"]),
            stable_evaluations=int(source_config["stable_atomic_evaluations"]),
            learning_rate=float(source_config["learning_rate"]),
            batch_size=int(source_config["batch_size"]),
            evaluation_batch_size=int(source_config["evaluation_batch_size"]),
            seed=int(source_config["seed"]) + 4001 * index,
        )
        adapters.load_state(state)
        adapters.freeze()
        after, rows = observe_source(
            host,
            evaluation_rows[index],
            capability_id=capability.capability_id,
            condition="AFTER_EVALUATION",
            batch_size=int(source_config["evaluation_batch_size"]),
        )
        source_observations.extend(rows)
        after_atomic, rows = observe_source(
            host,
            atomic_rows[index],
            capability_id=capability.capability_id,
            condition="AFTER_ATOMIC",
            batch_size=int(source_config["evaluation_batch_size"]),
        )
        source_observations.extend(rows)
        transition, extraction = extract_transition(host)
        package_accuracy = transition_accuracy(transition, evaluation_rows[index])
        state_sha = _adapter_state_sha256(state)
        adapter_path = output / "source_adapters" / f"{capability.capability_id}.safetensors"
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(state, str(adapter_path))
        package = write_package_once(
            output / "packages",
            transition,
            {
                "teacher_before_sha256": base_state,
                "teacher_after_sha256": state_sha,
            },
        )
        package["capability_id"] = capability.capability_id
        after_packages.append(package)
        transitions.append(transition)
        source_receipts.append(
            {
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
                "before_evaluation": before,
                "before_atomic": before_atomic,
                "after_evaluation": after,
                "after_atomic": after_atomic,
                "package_oracle_accuracy": package_accuracy,
                "package_source_exact_agreement": after["canonical_accuracy"],
                "package_source_disagreements": after["rows"] - after["canonical_correct"],
                "training": training,
                "extraction": extraction,
            }
        )
        print(json.dumps(source_receipts[-1], sort_keys=True), flush=True)
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
    manifest = {"before": before_package, "after": after_packages}
    manifest_path = output / "package_manifest.json"
    write_json_once(manifest_path, manifest)
    source_rows_path = output / "source_observations.jsonl"
    write_jsonl_once(source_rows_path, source_observations)
    recipient_receipts = []
    for host_key in config["recipient_hosts"]:
        worker_output = output / "recipients" / str(host_key)
        command = [
            sys.executable,
            "-m",
            "experiments.foreign_capability_r13.recipient_worker",
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
        (output / "worker_logs").mkdir(exist_ok=True)
        (output / "worker_logs" / f"{host_key}.stdout.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        (output / "worker_logs" / f"{host_key}.stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise R13Error(f"recipient worker failed: {host_key}")
        recipient_receipts.append(json_object(worker_output / "receipt.json"))
    gates = config["gates"]
    source_pass = all(
        item["before_atomic"]["accuracy"] <= float(gates["source_before_atomic_maximum"])
        and item["after_atomic"]["accuracy"] == float(gates["source_after_atomic_accuracy"])
        and item["after_atomic"]["accuracy"] - item["before_atomic"]["accuracy"]
        >= float(gates["source_atomic_gain_minimum"])
        and item["base_state_sha256_before"] == item["base_state_sha256_after"]
        and item["package_oracle_accuracy"] == float(gates["package_oracle_accuracy"])
        for item in source_receipts
    )
    passed = source_pass and _recipient_pass(config, recipient_receipts)
    receipt = {
        "format": "abi-r13-bounded-capability-extraction/1",
        "status": "PASS" if passed else "FAIL",
        "claim_target": "BOUNDED_ENUMERABLE_CAPABILITY_EXTRACTION",
        "claim_ceiling": "NOT_BEHAVIORAL_TRANSPLANT_NOT_GENERAL_KNOWLEDGE",
        "config_sha256": sha256_file(config_path),
        "reveal_sha256": sha256_file(preserved_reveal),
        "manifest_sha256": sha256_file(manifest_path),
        "r11_freeze": r11_freeze,
        "extractor": extractor_spec(),
        "data": {
            "capabilities": [item.capability_id for item in capabilities],
            "training_rows_per_capability": len(training_rows[0]),
            "evaluation_rows_per_capability": len(evaluation_rows[0]),
            "atomic_rows_per_capability": len(atomic_rows[0]),
            "training_evaluation_prompt_overlap": 0,
        },
        "source_acquisitions": source_receipts,
        "source_observations": {
            "path": source_rows_path.name,
            "rows": len(source_observations),
            "sha256": sha256_file(source_rows_path),
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
