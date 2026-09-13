"""Fail-closed fresh live verification of the R25 direct import."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.factual_semantic_r16.package import load_package as load_r16_package
from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
)
from experiments.generative_transfer_r21 import run as base_run
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash

from .binding import load_config
from .protocol import BUILD_INITIALIZATIONS, NAMESPACES
from .run import run


def _rows(directory: Path, result: dict[str, Any], name: str) -> list[dict[str, Any]]:
    ref = result.get("artifacts", {}).get(name, {})
    path = directory / str(ref.get("path", ""))
    if (
        not path.is_file()
        or path.stat().st_size != ref.get("bytes")
        or sha256_file(path) != ref.get("sha256")
    ):
        raise R14Error(f"R25 raw artifact changed: {name}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != ref.get("rows") or any(not line for line in lines):
        raise R14Error(f"R25 raw artifact row count changed: {name}")
    try:
        rows = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise R14Error(f"R25 raw artifact is invalid: {name}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise R14Error(f"R25 raw artifact has a non-object: {name}")
    return rows


def _metrics(
    config: dict[str, Any],
    package_builds: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    core_rows: list[dict[str, Any]],
    lifecycle: list[dict[str, Any]],
    leakage: list[dict[str, Any]],
) -> dict[str, int]:
    stored_english_rows = [
        json.loads(line)
        for line in (Path(__file__).resolve().parents[2] / config["r23_hidden_rows"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    stored_english = {
        row["record_id"]: row["output"]
        for row in stored_english_rows
        if row.get("method") == "abi_factorized" and row.get("seed") == 21022
    }
    return {
        "source_records_exact": sum(int(item["facts"]) for item in config["r16_packages"]),
        "reproducible_package_builds": sum(
            row["sha256"] == row["canonical_sha256"] for row in package_builds
        ),
        "domain_exact": sum(row["gpu_output"] == row["answer"] for row in observations),
        "teacher_agreement": sum(
            row["gpu_output"] == row["teacher_output"] for row in observations
        ),
        "namespace_route_exact": sum(
            row["namespace"] == row["routed_namespace"] for row in observations
        ),
        "other_domain_target_answers": sum(
            row["other_domain_output"] == row["answer"] for row in observations
        ),
        "cpu_gpu_exact": sum(
            row["cpu_output"] == row["gpu_output"] for row in observations
        ),
        "english_immutable": sum(
            row["output"] == stored_english[row["record_id"]] for row in core_rows
        ),
        "english_core_target_answers": sum(
            row["output"].strip() == row["target_answer"] for row in leakage
        ),
        "package_lifecycle_exact": sum(
            row["removed"]
            and row["absent_rejected"]
            and row["before_output"] == row["after_output"]
            and row["restored_archive_sha256"] == row["expected_archive_sha256"]
            for row in lifecycle
        ),
        "targeted_corruptions_rejected": sum(
            row["targeted_corruption_rejected"] for row in lifecycle
        ),
        "source_parameters_copied": 0,
        "receiver_training_steps": 0,
    }


def _validate_packages(
    root: Path, config: dict[str, Any], result: dict[str, Any]
) -> int:
    api = base_run._layercake(root)
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(config["layercake_abi"]["research_signing_seed_hex"])
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signer = api["key_id"](public)
    source = {
        item["namespace"]: load_r16_package(root / item["path"])
        for item in config["r16_packages"]
    }
    builds = result.get("builds")
    if not isinstance(builds, dict) or set(builds) != {
        str(value) for value in BUILD_INITIALIZATIONS
    }:
        raise R14Error("R25 build inventory changed")
    count = 0
    for initialization in BUILD_INITIALIZATIONS:
        entries = builds[str(initialization)]
        if set(entries) != set(NAMESPACES):
            raise R14Error("R25 build namespaces changed")
        for namespace in NAMESPACES:
            record = entries[namespace]
            path = root / record["path"]
            if (
                not path.is_file()
                or path.stat().st_size != record.get("bytes")
                or sha256_file(path) != record.get("sha256")
            ):
                raise R14Error("R25 package identity changed")
            package = api["load_package"](path, trust_store={signer: public})
            provenance = package.manifest.training_data_provenance
            if (
                not package.signed
                or package.signature_key_id != signer
                or package.manifest.domains != (namespace,)
                or package.manifest.architecture.get("name") != "canonical_factual_table"
                or package.manifest.architecture.get("facts") != source[namespace]["facts"]
                or provenance.get("source_parameters_copied") != 0
                or provenance.get("receiver_training_steps") != 0
                or provenance.get("teacher_at_inference") is not False
                or provenance.get("imported_records") != 8
                or record.get("parameters") != 0
                or record.get("state_bytes") != 32
            ):
                raise R14Error("R25 package semantics or provenance changed")
            count += 1
    for namespace in NAMESPACES:
        digests = {
            builds[str(initialization)][namespace]["sha256"]
            for initialization in BUILD_INITIALIZATIONS
        }
        if len(digests) != 1:
            raise R14Error("R25 independent package builds differ")
    return count


def verify(config_path: Path, stored_dir: Path, replay_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    stored_dir = stored_dir.resolve()
    replay_dir = replay_dir.resolve()
    config = load_config(root, config_path)
    stored_result_path = stored_dir / "result.json"
    stored_result = json_object(stored_result_path)
    if (
        stored_result.get("format") != "abi-r25-canonical-layercake-import-result/1"
        or stored_result.get("config_sha256") != sha256_file(config_path)
        or stored_result.get("evidence_sha256") != selfless_evidence_hash(stored_result)
    ):
        raise R14Error("R25 stored result identity changed")
    names = (
        "package_builds",
        "domain_observations",
        "english_immutability",
        "lifecycle",
        "english_leakage",
    )
    stored_rows = {name: _rows(stored_dir, stored_result, name) for name in names}
    packages = _validate_packages(root, config, stored_result)
    replay_result = run(config_path, replay_dir)
    replay_rows = {name: _rows(replay_dir, replay_result, name) for name in names}
    for name in names[1:]:
        if stored_rows[name] != replay_rows[name]:
            raise R14Error(f"R25 fresh live replay changed: {name}")
    stripped_stored = [{key: value for key, value in row.items() if key != "path"} for row in stored_rows["package_builds"]]
    stripped_replay = [{key: value for key, value in row.items() if key != "path"} for row in replay_rows["package_builds"]]
    if stripped_stored != stripped_replay:
        raise R14Error("R25 fresh package rebuild changed")
    metrics = _metrics(
        config,
        stored_rows["package_builds"],
        stored_rows["domain_observations"],
        stored_rows["english_immutability"],
        stored_rows["lifecycle"],
        stored_rows["english_leakage"],
    )
    gates = {name: metrics[name] == threshold for name, threshold in config["gates"].items()}
    gates["teacher_absent_at_execution"] = "transformers" not in sys.modules
    if (
        stored_result.get("metrics") != metrics
        or stored_result.get("gates") != gates
        or not all(gates.values())
        or stored_result.get("verdict") != "PASS_BOUNDED_DIRECT_IMPORT"
        or stored_result.get("claim")
        != "R25_REGISTERED_FACTUAL_DIRECT_IMPORT_AND_COMPOSITION"
        or stored_result.get("claim_ceiling")
        != "NOT_AUTONOMOUS_LABELING_OR_OPEN_WORLD_DOMAINS"
        or stored_result.get("full_abi_moonshot") != "OPEN"
    ):
        raise R14Error("R25 scientific gate failed")
    accounting = stored_result.get("information_accounting", {})
    if (
        accounting.get("new_source_calls") != 0
        or accounting.get("source_parameters_copied") != 0
        or accounting.get("receiver_training_steps") != 0
        or accounting.get("teacher_present_at_build") is not False
        or accounting.get("teacher_present_at_execution") is not False
        or accounting.get("imported_records") != 16
        or accounting.get("active_parameters") != 0
        or accounting.get("state_bytes_per_selected_package") != 32
    ):
        raise R14Error("R25 information accounting changed")
    verification = {
        "format": "abi-r25-strict-live-verification/1",
        "status": "PASS_STRICTLY_VERIFIED_BOUNDED_DIRECT_IMPORT",
        "scientific_claim": stored_result["claim"],
        "claim_ceiling": stored_result["claim_ceiling"],
        "config_sha256": sha256_file(config_path),
        "stored_result_sha256": sha256_file(stored_result_path),
        "replay_result_sha256": sha256_file(replay_dir / "result.json"),
        "packages_recomputed": packages,
        "rows_live_recomputed": sum(len(rows) for rows in replay_rows.values()),
        "package_archives_rebuilt": 6,
        "stored_scientific_booleans_trusted": False,
        "metrics": metrics,
        "gates": gates,
        "source_parameters_copied": 0,
        "receiver_training_steps": 0,
        "teacher_present_at_execution": False,
        "full_abi_moonshot": "OPEN",
    }
    verification["evidence_sha256"] = selfless_evidence_hash(verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stored", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.config, args.stored, args.replay)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
