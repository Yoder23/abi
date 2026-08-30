"""Independently inspect a clean ABI final-validation rehearsal.

The collector consumes only files produced inside a clean extracted release
tree plus the release archive itself.  It never imports a model or capability
runtime and never trusts the existing headline certificates.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .final_validation import (
    CAPABILITIES,
    HOSTS,
    evidence_hash,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)


class CleanReproductionError(RuntimeError):
    """Raised when the clean result is missing required raw evidence."""


def _self_hash(value: dict[str, Any]) -> bool:
    return value.get("evidence_sha256") == evidence_hash(value)


def _archive_verification(archive: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as handle:
        manifest_bytes = handle.read("abi-final-validation/manifest.json")
        manifest = json.loads(manifest_bytes)
        files = manifest["files"]
        exact = 0
        for binding in files:
            relative = binding["path"]
            payload = handle.read(f"abi-final-validation/{relative}")
            import hashlib

            if len(payload) != binding["bytes"]:
                raise CleanReproductionError(f"archive byte length changed: {relative}")
            if hashlib.sha256(payload).hexdigest() != binding["sha256"]:
                raise CleanReproductionError(f"archive hash changed: {relative}")
            exact += 1
        names = set(handle.namelist())
    forbidden_fragments = (
        "/.git/",
        "/__pycache__/",
        "/.pytest_cache/",
        "/external_reproduction/raw/",
        "/external_reproduction/models/",
        "/clean_run/",
    )
    forbidden = sorted(
        name for name in names if any(fragment in f"/{name}" for fragment in forbidden_fragments)
    )
    return {
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "manifest_sha256": __import__("hashlib").sha256(manifest_bytes).hexdigest(),
        "files_verified": exact,
        "files_required": len(files),
        "forbidden_development_paths": forbidden,
        "passed": exact == len(files) and not forbidden,
    }


def _clean_source_bindings(clean_root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    bindings: dict[str, dict[str, Any]] = {}
    rows = {
        **candidate["evaluator_and_data_bindings"],
        **{f"capability:{name}": value for name, value in candidate["capability_artifacts"].items()},
        **{f"adapter:{name}": value for name, value in candidate["host_adapters"].items()},
    }
    for name, row in rows.items():
        path = clean_root / row["path"]
        bindings[name] = {
            "path": row["path"],
            "present": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "expected_sha256": row["sha256"],
            "exact": path.is_file() and sha256_file(path) == row["sha256"],
        }
    return {
        "bindings": bindings,
        "exact": sum(row["exact"] for row in bindings.values()),
        "required": len(bindings),
    }


def _host_certifications(clean_root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    base = clean_root / "clean_run/host_certification"
    hosts = {}
    for host in HOSTS:
        result_path = base / host / "result.json"
        performance_path = base / host / "performance.json"
        adapter_path = base / host / "adapter.json"
        result = read_json(result_path)
        performance = read_json(performance_path)
        expected = candidate["host_adapters"][host]["sha256"]
        expected_host = candidate["host_checkpoints"][host]
        hosts[host] = {
            "status": result["status"],
            "device": result["device"],
            "result_self_hash_exact": _self_hash(result),
            "performance_sha256": sha256_file(performance_path),
            "adapter_sha256": sha256_file(adapter_path),
            "adapter_expected_sha256": expected,
            "adapter_exact": sha256_file(adapter_path) == expected,
            "adapter_parameters": result["adapter"]["trainable_parameters"],
            "capability_data_accessed": result["capability_blindness"][
                "capability_artifact_available_to_certification_logic"
            ],
            "capability_package_open_attempts": result["capability_blindness"][
                "package_open_attempts"
            ],
            "certification_examples": result["certification_data"]["examples"],
            "certification_utf8_bytes": result["certification_data"]["raw_utf8_bytes"],
            "wall_seconds": result["cost"]["wall_seconds"],
            "runtime_overhead_fraction": performance["overhead_fraction"],
            "snapshot_inventory_sha256": result["host"].get("snapshot_inventory_sha256"),
            "snapshot_inventory_exact": result["host"].get("snapshot_inventory_sha256")
            == expected_host.get("snapshot_inventory_sha256"),
            "checkpoint_sha256": result["host"].get("checkpoint_sha256"),
            "checkpoint_exact": result["host"].get("checkpoint_sha256")
            == expected_host.get("checkpoint_sha256"),
        }
    passed = all(
        row["status"].startswith("PASS")
        and row["result_self_hash_exact"]
        and row["adapter_exact"]
        and row["adapter_parameters"] == 0
        and row["capability_data_accessed"] is False
        and row["capability_package_open_attempts"] == 0
        and row["snapshot_inventory_exact"]
        and row["checkpoint_exact"]
        for row in hosts.values()
    )
    return {"hosts": hosts, "passed": passed}


def _matrix(clean_root: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    base = clean_root / "clean_run/matrix"
    hosts: dict[str, Any] = {}
    outputs: dict[str, dict[tuple[str, str], str]] = {}
    actions: dict[str, dict[tuple[str, str], tuple[int, ...]]] = {}
    for host in HOSTS:
        result = read_json(base / host / "result.json")
        rows = read_jsonl(base / host / "observations.jsonl")
        mathematical = read_json(base / host / "mathematical.json")
        keys = [(str(row["capability"]), str(row["probe_id"])) for row in rows]
        if len(keys) != len(set(keys)):
            raise CleanReproductionError(f"duplicate clean matrix row: {host}")
        outputs[host] = {key: str(row["output"]) for key, row in zip(keys, rows, strict=True)}
        actions[host] = {
            key: tuple(int(value) for value in row.get("actions", []))
            for key, row in zip(keys, rows, strict=True)
            if key[0] != "english"
        }
        per_capability = {}
        for capability in CAPABILITIES:
            selected = [row for row in rows if row["capability"] == capability]
            per_capability[capability] = {
                "tasks": len(selected),
                "functional_successes": sum(bool(row["functional_pass"]) for row in selected),
                "source_output_byte_exact": sum(
                    bool(row["source_output_byte_exact"]) for row in selected
                ),
            }
        causal = result["causal"]
        removal = causal["capability_removal_and_reinstall"]
        corrupt = causal["random_and_shuffled_capabilities"]
        hosts[host] = {
            "status": result["status"],
            "device": result["device"],
            "result_self_hash_exact": _self_hash(result),
            "mathematical_self_hash_exact": _self_hash(mathematical),
            "gates_passed": sum(bool(value) for value in result["gates"].values()),
            "gates_required": len(result["gates"]),
            "capabilities": per_capability,
            "adapter_hash_frozen": result["adapter"]["sha256_before"]
            == result["adapter"]["sha256_after"]
            == candidate["host_adapters"][host]["sha256"],
            "capability_hashes_frozen": all(
                result["package_hashes_after"][name]
                == candidate["capability_artifacts"][name]["sha256"]
                for name in CAPABILITIES
            ),
            "removal_reinstallation": sum(
                bool(value["absent_execution_rejected"])
                and bool(value["restored_output_byte_exact"])
                for value in removal.values()
            ),
            "removal_reinstallation_required": len(removal),
            "random_rejections": sum(
                bool(value["random_rejected_before_execution"]["rejected"])
                for value in corrupt.values()
            ),
            "shuffled_rejections": sum(
                bool(value["shuffled_rejected_before_execution"]["rejected"])
                for value in corrupt.values()
            ),
            "corrupt_rejections_required_each": len(corrupt),
            "teacher_loaded": result["teacher_loaded"],
            "source_model_loaded": result["source_model_loaded"],
            "training_performed": result["training_performed"],
            "calibration_performed": result["calibration_performed"],
            "wall_seconds": result["performance"]["wall_seconds_all_tests"],
            "peak_process_rss_bytes_lower_bound": result["performance"][
                "peak_process_rss_bytes_lower_bound"
            ],
            "peak_cuda_allocated_bytes": result["performance"]["peak_cuda_allocated_bytes"],
        }
    common = set.intersection(*(set(value) for value in outputs.values()))
    specialist = set.intersection(*(set(value) for value in actions.values()))
    aggregate = {
        "matrix_cells_passed": sum(
            row["functional_successes"] == row["source_output_byte_exact"] == row["tasks"]
            for host in hosts.values()
            for row in host["capabilities"].values()
        ),
        "matrix_cells_required": len(HOSTS) * len(CAPABILITIES),
        "receiver_successes": sum(
            row["functional_successes"]
            for host in hosts.values()
            for row in host["capabilities"].values()
        ),
        "receiver_tasks": sum(
            row["tasks"] for host in hosts.values() for row in host["capabilities"].values()
        ),
        "cross_host_outputs_equal": sum(
            len({outputs[host][key] for host in HOSTS}) == 1 for key in common
        ),
        "cross_host_outputs_total": len(common),
        "cross_host_actions_equal": sum(
            len({actions[host][key] for host in HOSTS}) == 1 for key in specialist
        ),
        "cross_host_actions_total": len(specialist),
    }
    passed = (
        aggregate["matrix_cells_passed"] == aggregate["matrix_cells_required"]
        and aggregate["receiver_successes"] == aggregate["receiver_tasks"]
        and aggregate["cross_host_outputs_equal"] == aggregate["cross_host_outputs_total"]
        and aggregate["cross_host_actions_equal"] == aggregate["cross_host_actions_total"]
        and all(
            host["status"].startswith("PASS")
            and host["result_self_hash_exact"]
            and host["mathematical_self_hash_exact"]
            and host["gates_passed"] == host["gates_required"]
            and host["adapter_hash_frozen"]
            and host["capability_hashes_frozen"]
            and host["removal_reinstallation"] == host["removal_reinstallation_required"]
            and host["random_rejections"] == host["corrupt_rejections_required_each"]
            and host["shuffled_rejections"] == host["corrupt_rejections_required_each"]
            and not any(
                host[name]
                for name in (
                    "teacher_loaded",
                    "source_model_loaded",
                    "training_performed",
                    "calibration_performed",
                )
            )
            for host in hosts.values()
        )
    )
    return {"hosts": hosts, "aggregate": aggregate, "passed": passed}


def collect(
    *,
    clean_root: Path,
    archive: Path,
    output: Path,
    command_receipt: Path,
    prior_command_receipts: list[Path] | None = None,
) -> dict[str, Any]:
    clean_root = clean_root.resolve()
    candidate = read_json(
        clean_root / "results/abi_final_validation_v2/frozen_release_candidate_r5.json"
    )
    archive_check = _archive_verification(archive.resolve())
    source = _clean_source_bindings(clean_root, candidate)
    certifications = _host_certifications(clean_root, candidate)
    matrix = _matrix(clean_root, candidate)
    hostile = read_json(clean_root / "clean_run/hostile_release_verification.json")
    commands = read_json(command_receipt.resolve())
    external_verify_path = clean_root / "external_reproduction/raw/final/verify.json"
    external_verify = read_json(external_verify_path)
    gates = {
        "archive_exact_and_development_state_absent": archive_check["passed"],
        "frozen_source_bindings_exact": source["exact"] == source["required"],
        "fresh_capability_blind_certification": certifications["passed"],
        "fresh_three_host_four_capability_matrix": matrix["passed"],
        "fresh_hostile_release_verification": hostile["status"]
        == "PASS_ALL_HOSTILE_RELEASE_MUTATIONS_REJECTED",
        "clean_commands_and_tests": commands["status"] == "PASS_CLEAN_COMMANDS_TESTS_AND_BUILD",
        "external_turnkey_byte_verifier": external_verify["status"]
        == "PASS_FINAL_VALIDATION_BYTES_AND_RAW_RECOMPUTATION",
        "source_teacher_absent": all(
            not host["teacher_loaded"] and not host["source_model_loaded"]
            for host in matrix["hosts"].values()
        ),
    }
    receipt = {
        "format": "abi-final-clean-checkout-reproduction/1",
        "status": "PASS_CLEAN_CHECKOUT_REPRODUCTION" if all(gates.values()) else "FAIL_CLEAN_CHECKOUT_REPRODUCTION",
        "source_lineage": {
            "technical_proof_commit": candidate["technical_proof_commit"],
            "technical_proof_tag": candidate["technical_proof_tag"],
            "clean_tree": "isolated_archive_extraction/abi_release",
            "development_git_metadata_reused": False,
            "development_caches_reused": False,
            "temporary_trainer_state_reused": False,
        },
        "gates": gates,
        "archive": archive_check,
        "frozen_source_bindings": source,
        "host_certification": certifications,
        "matrix": matrix,
        "hostile_release_verification": {
            "status": hostile["status"],
            "sha256": sha256_file(clean_root / "clean_run/hostile_release_verification.json"),
            "mutations_rejected": hostile["mutations_rejected"],
            "mutations_required": hostile["mutations_required"],
            "forbidden_attempts_rejected": hostile["forbidden_attempts_rejected"],
            "forbidden_attempts_required": hostile["forbidden_attempts_required"],
        },
        "commands_tests_and_build": commands,
        "external_turnkey_byte_verifier": {
            "status": external_verify["status"],
            "sha256": sha256_file(external_verify_path),
            "failures": external_verify["failures"],
        },
        "preserved_prior_failed_clean_attempts": [
            {
                "path": f"clean_attempt_{index}/command_receipt.json",
                "sha256": sha256_file(path.resolve()),
                "status": read_json(path.resolve())["status"],
                "failed_commands": [
                    {
                        "command": row["command"],
                        "exit_code": row["exit_code"],
                        "stdout_sha256": row["stdout_sha256"],
                        "stdout_tail": row["stdout_tail"].replace(
                            str(path.resolve().parents[1]), "<CLEAN_ATTEMPT_ROOT>"
                        ).replace(
                            str(path.resolve().parents[1]).replace("\\", "\\\\"),
                            "<CLEAN_ATTEMPT_ROOT>",
                        ),
                    }
                    for row in read_json(path.resolve())["commands"]
                    if not row["passed"]
                ],
            }
            for index, path in enumerate(prior_command_receipts or [], start=1)
        ],
        "independent_hardware_claimed": False,
        "human_review_claimed": False,
    }
    write_json(output.resolve(), receipt)
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--command-receipt", required=True)
    parser.add_argument("--prior-command-receipt", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    value = collect(
        clean_root=Path(args.clean_root),
        archive=Path(args.archive),
        command_receipt=Path(args.command_receipt),
        prior_command_receipts=[Path(value) for value in args.prior_command_receipt],
        output=Path(args.output),
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
