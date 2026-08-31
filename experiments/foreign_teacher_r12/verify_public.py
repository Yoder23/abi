"""Fail-closed live recomputation of the R12-A public Qwen feasibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from experiments.native_isa_r11.core import (
    load_package,
    sha256_file,
    transition_accuracy,
)
from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
    generate_rows,
    public_capabilities,
)
from experiments.native_transfer_r8.native_host import (
    SPECS,
    FrozenNeuralHost,
    GenericRecipientAdapterSet,
)

from .extractor import extract_transition
from .public_preflight import _atomic_rows, _json
from .teacher import R12TeacherError, evaluate


def _evidence(value: dict[str, Any]) -> None:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
        raise R12TeacherError("public receipt evidence hash changed")


def verify(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _json(config_path)
    receipt = _json(run_dir / "receipt.json")
    _evidence(receipt)
    if (
        receipt.get("format") != "abi-r12a-public-qwen-feasibility/1"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("heldout_secret_created") is not False
        or receipt.get("r11_files_modified") is not False
    ):
        raise R12TeacherError("public run identity or custody changed")
    adapter_path = run_dir / str(receipt["adapter_artifact"]["path"])
    if (
        not adapter_path.is_file()
        or adapter_path.stat().st_size != receipt["adapter_artifact"]["bytes"]
        or sha256_file(adapter_path) != receipt["adapter_artifact"]["sha256"]
    ):
        raise R12TeacherError("public Qwen adapter artifact changed")
    package_path = run_dir / "packages" / str(receipt["r11_package"]["path"])
    if (
        not package_path.is_file()
        or package_path.stat().st_size != receipt["r11_package"]["bytes"]
        or sha256_file(package_path) != receipt["r11_package"]["sha256"]
    ):
        raise R12TeacherError("public extracted package changed")
    _, package_transition = load_package(package_path)
    capability = public_capabilities(
        int(config["data"]["capability_seed"]), split="development", count=1
    )[0]
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
    adapters = GenericRecipientAdapterSet(
        host, rank=int(config["training"]["lora_rank"])
    )
    state = load_file(str(adapter_path), device="cpu")
    adapters.load_state(state)
    adapters.freeze()
    after = evaluate(
        host,
        evaluation_rows,
        batch_size=int(config["training"]["evaluation_batch_size"]),
    )
    atomic = evaluate(
        host,
        atomic_rows,
        batch_size=int(config["training"]["evaluation_batch_size"]),
    )
    extracted, extraction = extract_transition(host)
    adapters.verify_base_frozen()
    if not torch.equal(extracted, package_transition):
        raise R12TeacherError("live foreign extraction changed package transition")
    package_accuracy = transition_accuracy(package_transition, evaluation_rows)
    gates = config["gates"]
    passed = (
        before["accuracy"] <= float(gates["before_accuracy_maximum"])
        and after["accuracy"] == float(gates["after_accuracy"])
        and atomic["accuracy"] == float(gates["atomic_accuracy"])
        and after["accuracy"] - before["accuracy"] >= float(gates["gain_minimum"])
        and package_accuracy == 1.0
        and adapters.base_state_sha256() == base_state
    )
    result = {
        "format": "abi-r12a-public-qwen-feasibility-verification/1",
        "verdict": "PASS" if passed else "FAIL",
        "receipt_evidence_sha256": receipt["evidence_sha256"],
        "before": before,
        "after": after,
        "atomic": atomic,
        "gain": after["accuracy"] - before["accuracy"],
        "package_executor_accuracy": package_accuracy,
        "extraction_evidence_sha256": extraction["evidence_sha256"],
        "base_state_unchanged": adapters.base_state_sha256() == base_state,
        "heldout_secret_created": False,
        "claim_ceiling": "PUBLIC_FEASIBILITY_ONLY_NOT_R12_CERTIFICATION",
    }
    result["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    if not passed:
        raise R12TeacherError("R12-A public feasibility gate failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        if output.exists():
            raise R12TeacherError(f"immutable public verification exists: {output}")
        result = verify(Path(args.config).resolve(), Path(args.run_dir).resolve())
        output.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
