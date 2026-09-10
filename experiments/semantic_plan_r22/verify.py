"""Fail-closed stored-evidence verification for R22."""

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
from experiments.generative_transfer_r21 import run as base_run
from experiments.generative_transfer_r21 import verify as base_verify
from experiments.generative_transfer_r21.hash_assurance_binding import (
    selfless_evidence_hash,
)

from .binding import load_config
from .normalize import validate_source


def verify(config_path: Path, source_run: Path, result_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    source_run = source_run.resolve()
    result_dir = result_dir.resolve()
    config = load_config(root, config_path)
    source_rows = validate_source(root, config_path, source_run)
    wrapper = json_object(result_dir / "result.json")
    engine_dir = result_dir / "engine"
    engine = json_object(engine_dir / "result.json")
    if (
        wrapper.get("format") != "abi-r22-semantic-plan-bakeoff-result/1"
        or wrapper.get("config_sha256") != sha256_file(config_path)
        or wrapper.get("normalized_source_receipt_sha256")
        != sha256_file(source_run / "receipt.json")
        or wrapper.get("normalized_source_rows_sha256")
        != sha256_file(source_run / "source_observations.jsonl")
        or wrapper.get("engine_result", {}).get("sha256")
        != sha256_file(engine_dir / "result.json")
        or wrapper.get("engine_result", {}).get("evidence_sha256")
        != engine.get("evidence_sha256")
        or wrapper.get("evidence_sha256") != selfless_evidence_hash(wrapper)
        or wrapper.get("verdict") != engine.get("verdict")
        or wrapper.get("raw_teacher_responses_preserved") is not True
        or wrapper.get("normalized_targets_used_for_training") is not True
        or wrapper.get("teacher_present_at_student_training") is not False
        or wrapper.get("teacher_present_at_package_execution") is not False
        or wrapper.get("layercake_code_changes") != 0
    ):
        raise R14Error("R22 wrapper evidence failed")

    base_config = (root / str(config["base_training_config"]["path"])).resolve()
    original_validator = base_verify._validate_source
    original_hash = base_verify.evidence_hash

    def bound_validator(
        _root: Path, candidate_config: Path, candidate_source: Path
    ) -> list[dict[str, Any]]:
        if sha256_file(candidate_config) != config["base_training_config"]["sha256"]:
            raise R14Error("R22 verifier received a different base config")
        if candidate_source.resolve() != source_run:
            raise R14Error("R22 verifier received a different source directory")
        return source_rows

    if engine.get("verdict") == "PASS_PUBLIC_PREREQUISITE":
        raise R14Error("R22 positive result requires a separate fresh live verifier")
    base_verify._validate_source = bound_validator
    base_verify.evidence_hash = selfless_evidence_hash
    try:
        engine_verification = base_verify.verify(base_config, source_run, engine_dir)
    finally:
        base_verify._validate_source = original_validator
        base_verify.evidence_hash = original_hash

    original_config = json_object(base_config)
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(original_config["layercake"]["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    api = base_run._layercake(root)
    signer = api["key_id"](public_pem)
    packages = 0
    for seed in engine["systems"].values():
        for method in seed.values():
            for package in method["packages"]:
                loaded = api["load_package"](
                    root / package["path"], trust_store={signer: public_pem}
                )
                if (
                    loaded.manifest.input_contract.get("mode")
                    != "direct_selected_portable_decoder"
                ):
                    raise R14Error("R22 package input contract is incomplete")
                packages += 1
    if packages != 24:
        raise R14Error("R22 package inventory is incomplete")

    result = {
        "format": "abi-r22-stored-strict-verification/1",
        "status": engine_verification["status"],
        "scientific_verdict": wrapper["verdict"],
        "config_sha256": sha256_file(config_path),
        "wrapper_result_sha256": sha256_file(result_dir / "result.json"),
        "engine_result_sha256": sha256_file(engine_dir / "result.json"),
        "source_rows_revalidated": len(source_rows),
        "packages_verified": packages,
        "fresh_execution_performed": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = selfless_evidence_hash(result)
    return result


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
