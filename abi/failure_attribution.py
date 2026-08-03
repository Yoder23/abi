"""Fail-closed attribution for ABI-to-LayerCake transfer experiments.

The sealed LayerCake host, ABI extraction, and the bridge between them are
independent scientific claims.  This module makes that boundary executable so
an integration prototype cannot invalidate LayerCake or inherit its metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


CONTRACT_FORMAT = "abi-layercake-failure-attribution-contract/1"
EVIDENCE_FORMAT = "abi-layercake-failure-attribution-evidence/1"
CONTRACT_STATUS = "CONTROLLING_BEFORE_ANY_FURTHER_ABI_ENGLISH_EXPERIMENT"


class AttributionError(ValueError):
    """Raised when attribution evidence is incomplete, inconsistent, or unsafe."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttributionError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AttributionError(f"JSON must be an object: {path}")
    return value


def _bound_file(root: Path, specification: Mapping[str, Any]) -> Path:
    path = (root / str(specification.get("path", ""))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise AttributionError(f"bound path escapes repository: {path}") from exc
    if not path.is_file():
        raise AttributionError(f"bound file is missing: {path}")
    expected = specification.get("sha256")
    if not isinstance(expected, str) or _sha256_file(path) != expected:
        raise AttributionError(f"bound file hash mismatch: {path}")
    return path


def _require_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise AttributionError(f"{label} is missing fields: {', '.join(missing)}")


def verify_contract(
    contract_path: str | Path,
    *,
    layercake_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the attribution contract and, when supplied, its external control."""

    path = Path(contract_path).resolve()
    contract = _read_json(path)
    if contract.get("format") != CONTRACT_FORMAT:
        raise AttributionError("unsupported attribution contract format")
    if contract.get("status") != CONTRACT_STATUS:
        raise AttributionError("attribution contract is not controlling")
    if contract.get("historical_evidence", {}).get("changed") is not False:
        raise AttributionError("historical evidence must remain unchanged")

    required_controls = {
        "sealed_layercake_native",
        "capability_naive_receiver",
        "bridge_only",
        "shuffled_abi_artifact",
        "native_payload_same_path",
    }
    controls = contract.get("required_control_matrix")
    if not isinstance(controls, dict) or set(controls) != required_controls:
        raise AttributionError("required control matrix is incomplete or expanded")
    owners = contract.get("proof_owners")
    if not isinstance(owners, dict) or set(owners) != {
        "LAYERCAKE",
        "ABI",
        "INTEGRATION",
        "END_TO_END",
    }:
        raise AttributionError("proof owners are incomplete")

    extraction_gates = contract.get("required_abi_extraction_gates")
    integrated_gates = contract.get("required_integrated_gates")
    if not isinstance(extraction_gates, list) or len(set(extraction_gates)) != len(
        extraction_gates
    ):
        raise AttributionError("ABI extraction gates must be a unique list")
    if not isinstance(integrated_gates, list) or len(set(integrated_gates)) != len(
        integrated_gates
    ):
        raise AttributionError("integrated gates must be a unique list")
    if not extraction_gates or not integrated_gates:
        raise AttributionError("attribution gate lists cannot be empty")

    for specification in contract["historical_evidence"].values():
        if isinstance(specification, dict) and "path" in specification:
            _bound_file(path.parent, specification)

    external = {"verified": False}
    if layercake_root is not None:
        root = Path(layercake_root).resolve()
        if not root.is_dir():
            raise AttributionError("LayerCake control repository is missing")
        control = contract["sealed_layercake_control"]
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit != control["repository_commit"]:
            raise AttributionError("LayerCake control repository commit mismatch")
        phase2 = _read_json(_bound_file(root, control["phase2_certificate"]))
        phase8 = _read_json(_bound_file(root, control["phase8_certificate"]))
        final_core = _read_json(_bound_file(root, control["final_core_manifest"]))
        canonical_abi = _read_json(
            _bound_file(root, control["canonical_semantic_abi_file"])
        )
        checkpoint_metadata = _read_json(
            _bound_file(root, control["native_checkpoint_metadata"])
        )
        runtime_metadata = _read_json(
            _bound_file(root, control["native_runtime_artifact"]["metadata"])
        )
        _bound_file(root, control["native_runtime_artifact"]["graph"])
        _bound_file(root, control["native_runtime_artifact"]["tokenizer"])
        if (
            phase2.get("status") != control["phase2_certificate"]["required_status"]
            or phase2.get("scope") != control["phase2_certificate"]["required_scope"]
        ):
            raise AttributionError("LayerCake Phase 2 control certificate is not qualified")
        if (
            phase2.get("primary_checkpoint_sha256")
            != control["primary_checkpoint_sha256"]
            or phase2.get("lineage", {}).get("architecture_id")
            != control["architecture_id"]
            or phase2.get("lineage", {}).get("architecture_hash")
            != control["architecture_hash"]
            or phase2.get("lineage", {}).get("runtime_graph_sha256")
            != control["native_runtime_graph_sha256"]
            or phase2.get("lineage", {}).get("abi_hash")
            != control["canonical_semantic_abi_file"]["sha256"]
            or any(
                phase2.get("component_hashes", {}).get(name) != digest
                for name, digest in control["component_hashes"].items()
            )
        ):
            raise AttributionError("LayerCake Phase 2 control lineage mismatch")
        if (
            final_core.get("same_checkpoint_quality_and_speed")
            != control["primary_checkpoint_sha256"]
            or final_core.get("architecture_id") != control["architecture_id"]
            or final_core.get("architecture_hash") != control["architecture_hash"]
            or final_core.get("native_runtime_graph_sha256")
            != control["native_runtime_graph_sha256"]
            or final_core.get("canonical_semantic_abi_sha256")
            != control["canonical_semantic_abi_file"]["sha256"]
            or canonical_abi.get("status") != "LOCKED"
        ):
            raise AttributionError("LayerCake final core or canonical ABI mismatch")
        if (
            checkpoint_metadata.get("checkpoint", {}).get("sha256")
            != control["primary_checkpoint_sha256"]
            or checkpoint_metadata.get("architecture", {}).get("architecture_version")
            != control["architecture_id"]
            or runtime_metadata.get("source_checkpoint_sha256")
            != control["primary_checkpoint_sha256"]
            or runtime_metadata.get("runtime", {}).get("graph_sha256")
            != control["native_runtime_graph_sha256"]
            or runtime_metadata.get("canonical_semantic_abi", {}).get("sha256")
            != control["canonical_semantic_abi_file"]["sha256"]
        ):
            raise AttributionError("LayerCake native runtime artifact mismatch")
        if (
            phase8.get("status") != control["phase8_certificate"]["required_status"]
            or phase8.get("verification_summary", {}).get("status")
            != control["phase8_certificate"]["required_verification_status"]
            or phase8.get("verification_summary", {}).get("details", {}).get(
                "performance", {}
            ).get("cpu_cpu_throughput_ratio")
            != control["locked_reference_metrics"][
                "phase8_domain_workload_cpu_throughput_ratio"
            ]
        ):
            raise AttributionError("LayerCake Phase 8 control certificate is not qualified")
        external = {
            "verified": True,
            "repository_commit": commit,
            "primary_checkpoint_sha256": phase2["primary_checkpoint_sha256"],
            "phase2_status": phase2["status"],
            "phase8_status": phase8["verification_summary"]["status"],
        }

    return {
        "status": "PASS",
        "contract_format": CONTRACT_FORMAT,
        "external_layercake_control": external,
    }


def _incomplete(reason: str) -> dict[str, Any]:
    return {
        "classification": "INCOMPLETE_ATTRIBUTION_EVIDENCE",
        "owner": "UNASSIGNED",
        "promotion_eligible": False,
        "reasons": [reason],
    }


def classify_evidence(
    evidence: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify a result without allowing cross-claim inheritance."""

    if evidence.get("format") != EVIDENCE_FORMAT:
        raise AttributionError("unsupported attribution evidence format")
    controls = evidence.get("controls")
    if not isinstance(controls, dict):
        return _incomplete("control matrix is missing")

    sealed = controls.get("sealed_layercake_native")
    if not isinstance(sealed, dict) or sealed.get("executed") is not True:
        return _incomplete("exact native LayerCake positive control was not run")
    if sealed.get("exact_control_lineage") is not True:
        return _incomplete("native LayerCake control lineage is not exact")
    if sealed.get("result") != "PASS":
        return {
            "classification": "LAYERCAKE_HOST_REGRESSION",
            "owner": "LAYERCAKE",
            "promotion_eligible": False,
            "reasons": ["the exact sealed native LayerCake positive control failed"],
        }

    negative_names = (
        "capability_naive_receiver",
        "bridge_only",
        "shuffled_abi_artifact",
    )
    for name in negative_names:
        control = controls.get(name)
        if not isinstance(control, dict) or control.get("executed") is not True:
            return _incomplete(f"required negative control was not run: {name}")
        if control.get("english_quality_result") != "FAIL":
            return {
                "classification": "TRANSFER_CAUSALITY_FAILURE",
                "owner": "ABI",
                "promotion_eligible": False,
                "reasons": [f"negative control unexpectedly retained English: {name}"],
            }

    same_path = controls.get("native_payload_same_path")
    if not isinstance(same_path, dict) or same_path.get("executed") is not True:
        return _incomplete("native same-path integration control was not run")
    if same_path.get("result") != "PASS":
        return {
            "classification": "ABI_LAYERCAKE_INTEGRATION_FAILURE",
            "owner": "INTEGRATION",
            "promotion_eligible": False,
            "reasons": ["known-good native payload failed through the receiving path"],
        }

    extraction = evidence.get("abi_extraction")
    if not isinstance(extraction, dict) or extraction.get("executed") is not True:
        return _incomplete("ABI extraction evidence is missing")
    if extraction.get("artifact_sha256_before") != extraction.get(
        "artifact_sha256_after"
    ):
        return {
            "classification": "ABI_EXTRACTION_FAILURE",
            "owner": "ABI",
            "promotion_eligible": False,
            "reasons": ["ABI artifact changed during integration or evaluation"],
        }
    extraction_results = extraction.get("gates")
    if not isinstance(extraction_results, dict):
        return _incomplete("ABI extraction gate results are missing")
    required_extraction = set(contract["required_abi_extraction_gates"])
    missing = sorted(required_extraction - set(extraction_results))
    if missing:
        return _incomplete(f"ABI extraction gates are missing: {', '.join(missing)}")
    failed = sorted(
        gate for gate in required_extraction if extraction_results.get(gate) != "PASS"
    )
    if failed:
        return {
            "classification": "ABI_EXTRACTION_FAILURE",
            "owner": "ABI",
            "promotion_eligible": False,
            "reasons": [f"ABI extraction gates failed: {', '.join(failed)}"],
        }

    candidate = evidence.get("integrated_candidate")
    if not isinstance(candidate, dict) or candidate.get("executed") is not True:
        return _incomplete("integrated candidate evidence is missing")
    identity_requirements = {
        "exact_layercake_execution_contract": True,
        "canonical_abi_unchanged": True,
        "abi_artifact_unchanged": True,
        "teacher_present_at_inference": False,
    }
    identity_failures = sorted(
        key for key, expected in identity_requirements.items() if candidate.get(key) != expected
    )
    if identity_failures:
        return {
            "classification": "ABI_LAYERCAKE_INTEGRATION_FAILURE",
            "owner": "INTEGRATION",
            "promotion_eligible": False,
            "reasons": [
                "integration identity requirements failed: "
                + ", ".join(identity_failures)
            ],
        }
    integrated_results = candidate.get("gates")
    if not isinstance(integrated_results, dict):
        return _incomplete("integrated gate results are missing")
    required_integrated = set(contract["required_integrated_gates"])
    missing = sorted(required_integrated - set(integrated_results))
    if missing:
        return _incomplete(f"integrated gates are missing: {', '.join(missing)}")
    failed = sorted(
        gate for gate in required_integrated if integrated_results.get(gate) != "PASS"
    )
    if failed:
        return {
            "classification": "ABI_LAYERCAKE_INTEGRATION_FAILURE",
            "owner": "INTEGRATION",
            "promotion_eligible": False,
            "reasons": [f"integrated candidate gates failed: {', '.join(failed)}"],
        }
    return {
        "classification": "PASS",
        "owner": "END_TO_END",
        "promotion_eligible": True,
        "reasons": [],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise AttributionError(f"refusing to overwrite attribution evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-contract")
    verify.add_argument("--contract", required=True)
    verify.add_argument("--layercake-root")
    classify = subparsers.add_parser("classify")
    classify.add_argument("--contract", required=True)
    classify.add_argument("--evidence", required=True)
    classify.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract_path = Path(args.contract).resolve()
    verification = verify_contract(
        contract_path,
        layercake_root=getattr(args, "layercake_root", None),
    )
    if args.command == "verify-contract":
        result = verification
    else:
        contract = _read_json(contract_path)
        evidence = _read_json(Path(args.evidence).resolve())
        result = classify_evidence(evidence, contract)
        if args.output:
            _write_json(Path(args.output).resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("classification", "PASS") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
