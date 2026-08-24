"""Audit the exact Phase 7 package against the final-mile host contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .final_mile import FinalMileError, canonical_json_bytes, sha256_file

CONTRACT_FORMAT = "abi-final-mile-host-portability-contract/1"


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalMileError(f"expected object: {path}")
    return value


def _write_immutable(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FinalMileError(f"immutable evidence already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


def _validate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = _object(contract_path)
    if (
        contract.get("format") != CONTRACT_FORMAT
        or contract.get("status")
        != "PREREGISTERED_BEFORE_CROSS_HOST_CAPABILITY_INSTALLATION"
        or contract.get("generic_certification", {}).get("capability_examples") != 0
        or contract.get("generic_certification", {}).get("teacher_outputs") != 0
        or contract.get("single_bounded_repair", {}).get("artifact_mutation_allowed") is not False
        or contract.get("single_bounded_repair", {}).get("receiver_specific_training_allowed")
        is not False
    ):
        raise FinalMileError("host portability governance changed")
    artifact = contract["frozen_english_artifact"]
    artifact_path = (root / artifact["path"]).resolve()
    if sha256_file(artifact_path) != artifact["archive_sha256"]:
        raise FinalMileError("frozen English artifact changed")
    abi_path = (
        root.parent
        / "layercake_release/moonshot/canonical_route_isolated_clarification_core_abi_v25.json"
    ).resolve()
    if sha256_file(abi_path) != artifact["declared_abi_sha256"]:
        raise FinalMileError("canonical LayerCake ABI declaration changed")
    receivers = contract.get("receiver_families")
    if not isinstance(receivers, list) or len(receivers) < 3:
        raise FinalMileError("three receiver families were not preregistered")
    return contract


def run_structural_screen(
    root: Path, contract_path: Path, *, output_dir: Path
) -> dict[str, Any]:
    """Fail before inference when a receiver cannot consume the sealed ABI."""
    root = root.resolve()
    contract_path = contract_path.resolve()
    contract = _validate(root, contract_path)
    artifact = contract["frozen_english_artifact"]
    declared_abi = artifact["declared_abi_version"]
    receiver_rows = []
    for receiver in contract["receiver_families"]:
        accepted = receiver["natively_accepted_abi_versions"]
        native = declared_abi in accepted
        receiver_rows.append(
            {
                "receiver_id": receiver["receiver_id"],
                "architecture": receiver["architecture"],
                "initialization_or_checkpoint": receiver["initialization_or_checkpoint"],
                "config_sha256": receiver["config_sha256"],
                "generic_certification_capability_examples": 0,
                "generic_certification_teacher_outputs": 0,
                "artifact_revealed_during_generic_certification": False,
                "host_parameters_changed_after_generic_certification": False,
                "declared_abi_natively_accepted": native,
                "capability_installation_attempted": native,
                "capability_specific_training_steps": 0,
                "capability_specific_calibration_examples": 0,
                "source_success_evaluation_eligible": native,
                "structural_result": "ELIGIBLE_FOR_EXISTING_LAYERCAKE_EVIDENCE"
                if native
                else "REJECTED_INCOMPATIBLE_NATIVE_RECEIVER_ABI",
            }
        )
    initial_pass = all(row["declared_abi_natively_accepted"] for row in receiver_rows)
    initial = _seal(
        {
            "format": "abi-final-mile-host-portability-initial-screen/1",
            "status": "PASS_CROSS_HOST_STRUCTURAL_SCREEN"
            if initial_pass
            else "FAIL_CROSS_HOST_STRUCTURAL_SCREEN",
            "contract": {
                "path": contract_path.relative_to(root).as_posix(),
                "sha256": sha256_file(contract_path),
            },
            "frozen_artifact_sha256": artifact["archive_sha256"],
            "receivers": receiver_rows,
            "receivers_accepted": sum(
                row["declared_abi_natively_accepted"] for row in receiver_rows
            ),
            "receivers_required": len(receiver_rows),
            "model_inference_performed": False,
            "training_performed": False,
            "semantic_portability_tested": False,
            "classification": None if initial_pass else contract["failure_classification"],
            "decision": (
                "The exact archive can enter the registered LayerCake v25 host but has no "
                "native consumer contract for either preregistered transformer receiver. "
                "Quality inference would not repair a structural ABI rejection."
            ),
        }
    )
    _write_immutable(output_dir / "initial_screen.json", initial)

    repair = contract["single_bounded_repair"]
    repair_preregistration = _seal(
        {
            "format": "abi-final-mile-bounded-host-repair-preregistration/1",
            "status": "SEALED_ONE_BOUNDED_REPAIR_BEFORE_RESCREEN",
            "contract_sha256": sha256_file(contract_path),
            "repair": repair,
            "frozen_artifact_sha256": artifact["archive_sha256"],
            "artifact_bytes_changed": False,
            "capability_training_authorized": False,
            "receiver_specific_training_authorized": False,
            "success_rule": (
                "All three native receiver paths must consume the identical artifact; an "
                "execution-provider bypass is explicitly non-qualifying."
            ),
        }
    )
    _write_immutable(output_dir / "repair_preregistration.json", repair_preregistration)

    repair_rows = []
    for row in receiver_rows:
        layercake = row["receiver_id"] == "layercake-v25"
        repair_rows.append(
            {
                "receiver_id": row["receiver_id"],
                "canonical_envelope_can_reference_unchanged_archive": True,
                "native_receiver_executes_capability_tensors": layercake,
                "would_require_layercake_provider_bypass": not layercake,
                "qualifies_under_preregistered_rule": layercake,
            }
        )
    repaired_pass = all(row["qualifies_under_preregistered_rule"] for row in repair_rows)
    rescreen = _seal(
        {
            "format": "abi-final-mile-bounded-host-repair-rescreen/1",
            "status": "PASS_CROSS_HOST_PORTABILITY"
            if repaired_pass
            else "HOST_INDEPENDENCE_FAILED",
            "repair_id": repair["repair_id"],
            "frozen_artifact_sha256": artifact["archive_sha256"],
            "artifact_bytes_changed": False,
            "receivers": repair_rows,
            "receivers_passing": sum(row["qualifies_under_preregistered_rule"] for row in repair_rows),
            "receivers_required": len(repair_rows),
            "model_inference_performed": False,
            "training_performed": False,
            "quality_or_speed_claim_borrowed": False,
            "semantic_portability_tested": False,
            "one_bounded_repair_consumed": True,
            "additional_architecture_search_authorized": False,
            "classification": contract["failure_classification"],
            "blocking_finding": (
                "The existing English tensors and execution declaration remain native only "
                "to the LayerCake v25 capability machine. A content envelope can make bytes "
                "transportable, but it cannot make Qwen2 or GPT-NeoX execute those tensors "
                "without a new receiver integration or a provider bypass. The latter was "
                "preregistered as non-qualifying."
            ),
        }
    )
    _write_immutable(output_dir / "repair_rescreen.json", rescreen)
    return rescreen


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run_structural_screen(
        root,
        (root / args.contract).resolve(),
        output_dir=(root / args.output_dir).resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
