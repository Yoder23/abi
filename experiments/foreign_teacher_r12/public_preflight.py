"""Run the pre-reveal public conventional-Qwen feasibility gate for R12-A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from experiments.native_isa_r11.core import (
    sha256_bytes,
    transition_accuracy,
    transition_bytes,
    write_package_once,
)
from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
    generate_rows,
    public_capabilities,
    render_prompt,
)
from experiments.native_transfer_r8.native_host import (
    SPECS,
    FrozenNeuralHost,
    GenericRecipientAdapterSet,
    sha256_file,
)

from .extractor import extract_transition
from .teacher import R12TeacherError, evaluate, train


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R12TeacherError(f"expected JSON object: {path}")
    return value


def _atomic_rows(capability: Any) -> list[dict[str, Any]]:
    rows = []
    for operation in range(3):
        for start in range(8):
            rows.append(
                {
                    "prompt": render_prompt(start, (operation,)),
                    "answer": capability.apply(start, (operation,)),
                }
            )
    return rows


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R12TeacherError(f"immutable public preflight exists: {output}")
    config = _json(config_path)
    if config.get("status") != "PUBLIC_PREFLIGHT_ONLY":
        raise R12TeacherError("R12-A public preflight status changed")
    capability = public_capabilities(
        int(config["data"]["capability_seed"]), split="development", count=1
    )[0]
    training_rows = generate_rows(
        capability,
        split="source_train",
        rows=int(config["data"]["training_rows"]),
        depths=config["data"]["training_depths"],
        seed=int(config["data"]["training_seed"]),
    )
    evaluation_rows = generate_rows(
        capability,
        split="r12_public_evaluation",
        rows=int(config["data"]["evaluation_rows"]),
        depths=config["data"]["evaluation_depths"],
        seed=int(config["data"]["evaluation_seed"]),
    )
    atomic_rows = _atomic_rows(capability)
    host = FrozenNeuralHost(SPECS["qwen2"], device="cuda")
    base_state = host.model_state_sha256
    before = evaluate(
        host,
        evaluation_rows,
        batch_size=int(config["training"]["evaluation_batch_size"]),
    )
    before_atomic = evaluate(
        host,
        atomic_rows,
        batch_size=int(config["training"]["evaluation_batch_size"]),
    )
    adapters = GenericRecipientAdapterSet(host, rank=int(config["training"]["lora_rank"]))
    state, training = train(
        host,
        adapters,
        training_rows,
        evaluation_rows,
        atomic_rows,
        maximum_steps=int(config["training"]["maximum_steps"]),
        evaluation_interval=int(config["training"]["evaluation_interval"]),
        learning_rate=float(config["training"]["learning_rate"]),
        batch_size=int(config["training"]["batch_size"]),
        evaluation_batch_size=int(config["training"]["evaluation_batch_size"]),
        seed=int(config["training"]["seed"]),
    )
    adapters.load_state(state)
    after = evaluate(
        host,
        evaluation_rows,
        batch_size=int(config["training"]["evaluation_batch_size"]),
    )
    after_atomic = evaluate(
        host,
        atomic_rows,
        batch_size=int(config["training"]["evaluation_batch_size"]),
    )
    adapters.verify_base_frozen()
    transition, extraction = extract_transition(host)
    package_executor_accuracy = transition_accuracy(transition, evaluation_rows)
    adapter_state_sha256 = hashlib.sha256(
        b"".join(key.encode() + state[key].numpy().tobytes() for key in sorted(state))
    ).hexdigest()
    gates = config["gates"]
    passed = (
        before["accuracy"] <= float(gates["before_accuracy_maximum"])
        and after["accuracy"] == float(gates["after_accuracy"])
        and after_atomic["accuracy"] == float(gates["atomic_accuracy"])
        and package_executor_accuracy == 1.0
        and after["accuracy"] - before["accuracy"] >= float(gates["gain_minimum"])
        and adapters.base_state_sha256() == base_state
    )
    output.mkdir(parents=True)
    adapter_path = output / "qwen_after_lora.safetensors"
    save_file(state, str(adapter_path))
    package = write_package_once(
        output / "packages",
        transition,
        {
            "teacher_before_sha256": base_state,
            "teacher_after_sha256": adapter_state_sha256,
        },
    )
    receipt = {
        "format": "abi-r12a-public-qwen-feasibility/1",
        "status": "PASS" if passed else "FAIL",
        "claim_ceiling": "PUBLIC_FEASIBILITY_ONLY_NOT_R12_CERTIFICATION",
        "config_sha256": sha256_file(config_path),
        "capability_id": capability.capability_id,
        "teacher": {
            "model_id": host.spec.model_id,
            "revision": host.spec.revision,
            "architecture_family": host.spec.architecture_family,
            "base_state_sha256_before": base_state,
            "base_state_sha256_after": adapters.base_state_sha256(),
            "adapter": adapters.inventory(),
            "adapter_state_sha256": adapter_state_sha256,
        },
        "data": {
            "training_rows": len(training_rows),
            "evaluation_rows": len(evaluation_rows),
            "atomic_rows": len(atomic_rows),
            "training_depths": config["data"]["training_depths"],
            "evaluation_depths": config["data"]["evaluation_depths"],
        },
        "before": before,
        "before_atomic": before_atomic,
        "after": after,
        "after_atomic": after_atomic,
        "training": training,
        "extraction": extraction,
        "r11_package": package,
        "r11_transition_sha256": sha256_bytes(transition_bytes(transition)),
        "r11_package_executor_accuracy": package_executor_accuracy,
        "adapter_artifact": {
            "path": adapter_path.name,
            "bytes": adapter_path.stat().st_size,
            "sha256": sha256_file(adapter_path),
        },
        "heldout_secret_created": False,
        "r11_files_modified": False,
        "hardware": {
            "device": "cuda",
            "cuda_device_name": torch.cuda.get_device_name(0),
        },
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    (output / "receipt.json").write_bytes(
        json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = run(Path(args.config).resolve(), Path(args.output).resolve())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
