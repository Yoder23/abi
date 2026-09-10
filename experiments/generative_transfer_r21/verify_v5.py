"""Strict verification of an R21 manifest-repaired negative result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
)

from . import run as base_run
from . import verify_v4
from .hash_assurance_binding import selfless_evidence_hash
from .manifest_repair_binding import load_manifest_repair_config


def verify(config_path: Path, source_run: Path, result_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config = load_manifest_repair_config(root, config_path)
    assurance_config = root / str(config["hash_assurance_config"]["path"])
    binding = json_object(result_dir / "manifest_repair_binding.json")
    if (
        binding.get("format") != "abi-r21-manifest-repair-run/1"
        or binding.get("manifest_repair_config_sha256") != sha256_file(config_path)
        or binding.get("result_sha256") != sha256_file(result_dir / "result.json")
        or binding.get("evidence_sha256") != selfless_evidence_hash(binding)
        or binding.get("input_contract_addition") != config["input_contract_addition"]
        or binding.get("retraining_performed") is not True
        or any(
            binding.get(name) is not False
            for name in (
                "data_changes_performed",
                "model_changes_performed",
                "gate_changes_performed",
                "layercake_changes_performed",
            )
        )
    ):
        raise R14Error("R21 manifest-repair run binding failed")

    assurance = json_object(assurance_config)
    control = json_object(root / str(assurance["label_control_config"]["path"]))
    base_config = json_object(root / str(control["base_config"]["path"]))
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(base_config["layercake"]["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    api = base_run._layercake(root)
    signer = api["key_id"](public_pem)
    wrapper = json_object(result_dir / "result.json")
    engine = json_object(result_dir / str(wrapper["engine_result"]["path"]))
    checked = 0
    for seed in engine["systems"].values():
        for method in seed.values():
            for package in method["packages"]:
                loaded = api["load_package"](
                    root / package["path"], trust_store={signer: public_pem}
                )
                if (
                    loaded.manifest.input_contract.get("mode")
                    != config["input_contract_addition"]["mode"]
                ):
                    raise R14Error("R21 repaired package contract is absent")
                checked += 1
    if checked != 24:
        raise R14Error("R21 repaired package inventory is incomplete")

    underlying = verify_v4.verify(assurance_config, source_run, result_dir)
    verification = {
        "format": "abi-r21-manifest-repair-strict-verification/1",
        "status": underlying["status"],
        "manifest_repair_config_sha256": sha256_file(config_path),
        "manifest_repair_binding_sha256": sha256_file(result_dir / "manifest_repair_binding.json"),
        "underlying_verification": underlying,
        "packages_with_repaired_contract": checked,
        "full_abi_moonshot": "OPEN",
    }
    verification["evidence_sha256"] = selfless_evidence_hash(verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.config, args.source_run, args.result)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
