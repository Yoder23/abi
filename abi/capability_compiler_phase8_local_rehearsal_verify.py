"""Read-only verifier for the same-machine clean-export Phase 8 rehearsal."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase7_direct_artifact_runtime import load_protocol as load_product
from .capability_compiler_phase7_verify import (
    _evidence_hash_valid,
    _junit,
    _rows,
    verify_device_document,
)


FORMAT = "abi-capability-compiler-phase8-local-clean-rehearsal-verify/1"


def output_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("mode"),
        row.get("probe_id"),
        row.get("domain"),
        row.get("output"),
        tuple(row.get("output_token_ids", ())),
    )


def identities(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    return [output_identity(row) for row in rows]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_PHASE8_LOCAL_CLEAN_REHEARSAL_READ_ONLY_VERIFY"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("phase8_certification_authorized") is not False
        or int(protocol.get("minimum_hostile_checks", 0)) < 15
    ):
        raise Phase3Error("Phase 8 local-rehearsal verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 8 local verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    gates = {
        "all_fresh_evidence_present": all(
            (root / path).is_file()
            for device in ("cpu", "cuda")
            for path in protocol["fresh_evidence"][device].values()
        ),
        "all_sealed_evidence_present": all(
            (root / path).is_file()
            for device in ("cpu", "cuda")
            for path in protocol["sealed_evidence"][device].values()
        ),
        "collection_pass": _json(root / protocol["collection_result"]).get("status")
        == "PASS_PHASE8_LOCAL_CLEAN_REHEARSAL_COLLECTION",
        "preparation_pass": _json(root / protocol["preparation_result"]).get("status")
        == "PASS_PHASE8_LOCAL_CLEAN_REHEARSAL_PREPARATION",
        "output_absent": not (root / protocol["output"]).exists(),
        "adversarial_output_absent": not (
            root / protocol["adversarial_output"]
        ).exists(),
        "model_inference_absent": True,
        "training_absent": True,
        "independent_hardware_not_claimed": True,
    }
    return {
        "format": "abi-capability-compiler-phase8-local-clean-rehearsal-verify-preflight/1",
        "status": "PASS_PHASE8_LOCAL_CLEAN_REHEARSAL_VERIFY_PREFLIGHT"
        if all(gates.values())
        else "FAIL_PHASE8_LOCAL_CLEAN_REHEARSAL_VERIFY_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "gates": gates,
        "phase8_certified": False,
    }


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable Phase 8 local verifier output exists")
    product, product_sha = load_product(root, root / protocol["product_protocol"])
    fresh_rows: dict[str, list[dict[str, Any]]] = {}
    sealed_rows: dict[str, list[dict[str, Any]]] = {}
    fresh_results: dict[str, dict[str, Any]] = {}
    recomputed: dict[str, dict[str, bool]] = {}
    for device in ("cpu", "cuda"):
        fresh_specification = protocol["fresh_evidence"][device]
        sealed_specification = protocol["sealed_evidence"][device]
        fresh_result = _json(root / fresh_specification["result"])
        fresh = _rows(root / fresh_specification["observations"])
        sealed = _rows(root / sealed_specification["observations"])
        if fresh_result.get("observations_sha256") != sha256_file(
            root / fresh_specification["observations"]
        ):
            raise Phase3Error(f"fresh {device} observations digest changed")
        recomputed[device] = verify_device_document(
            root=root,
            protocol=product,
            protocol_sha=product_sha,
            device=device,
            result=fresh_result,
            observations=fresh,
        )
        fresh_rows[device] = fresh
        sealed_rows[device] = sealed
        fresh_results[device] = fresh_result

    fresh_identities = {device: identities(rows) for device, rows in fresh_rows.items()}
    sealed_identities = {device: identities(rows) for device, rows in sealed_rows.items()}
    patch_tests = _junit(root / protocol["layercake_patch_tests"])
    preparation = _json(root / protocol["preparation_result"])
    collection = _json(root / protocol["collection_result"])
    gates = {
        "both_devices_independently_recomputed": all(
            all(device.values()) for device in recomputed.values()
        ),
        "fresh_cross_device_all_244_outputs_and_tokens_exact": fresh_identities["cpu"]
        == fresh_identities["cuda"]
        and len(fresh_identities["cpu"]) == 244,
        "fresh_cpu_matches_sealed_functional_identity": fresh_identities["cpu"]
        == sealed_identities["cpu"],
        "fresh_cuda_matches_sealed_functional_identity": fresh_identities["cuda"]
        == sealed_identities["cuda"],
        "fresh_active_tensor_identity": fresh_results["cpu"]["memory"]
        ["total_integrated_active_tensor_bytes"]
        == fresh_results["cuda"]["memory"]["total_integrated_active_tensor_bytes"]
        == 255916588,
        "layercake_patch_tests_pass": patch_tests
        == {"tests": 42, "failures": 0, "errors": 0, "skipped": 0},
        "preparation_exact_and_local_only": preparation.get("status")
        == "PASS_PHASE8_LOCAL_CLEAN_REHEARSAL_PREPARATION"
        and preparation.get("phase8_certified") is False
        and preparation.get("hardware", {}).get("fingerprint_sha256")
        == protocol["development_hardware_fingerprint_sha256"],
        "collection_exact_and_local_only": collection.get("status")
        == "PASS_PHASE8_LOCAL_CLEAN_REHEARSAL_COLLECTION"
        and collection.get("phase8_certified") is False
        and collection.get("hardware", {}).get("fingerprint_sha256")
        == protocol["development_hardware_fingerprint_sha256"],
        "fresh_result_hashes_valid": all(
            _evidence_hash_valid(result) for result in fresh_results.values()
        ),
        "teacher_absent": all(
            result.get("teacher_model_loaded") is False
            for result in fresh_results.values()
        ),
        "training_absent": all(
            result.get("training_performed") is False
            and int(result.get("receiver_training_steps", -1)) == 0
            for result in fresh_results.values()
        ),
        "phase8_independence_not_claimed": preparation.get("phase8_certified")
        is False
        and collection.get("phase8_certified") is False,
    }

    changed_output = copy.deepcopy(fresh_rows["cuda"])
    changed_output[0]["output"] = str(changed_output[0]["output"]) + "x"
    changed_token = copy.deepcopy(fresh_rows["cuda"])
    changed_token[0]["output_token_ids"] = [
        *changed_token[0].get("output_token_ids", []),
        0,
    ]
    changed_result = copy.deepcopy(fresh_results["cpu"])
    changed_result["device"] = "tampered"
    mutations = {
        "fresh_cpu_raw_mutation_rejected": hashlib.sha256(
            (root / protocol["fresh_evidence"]["cpu"]["observations"]).read_bytes()
            + b"x"
        ).hexdigest()
        != protocol["bindings"][protocol["fresh_evidence"]["cpu"]["observations"]],
        "fresh_cuda_raw_mutation_rejected": hashlib.sha256(
            (root / protocol["fresh_evidence"]["cuda"]["observations"]).read_bytes()
            + b"x"
        ).hexdigest()
        != protocol["bindings"][protocol["fresh_evidence"]["cuda"]["observations"]],
        "cross_device_output_mutation_rejected": fresh_identities["cpu"]
        != identities(changed_output),
        "cross_device_token_mutation_rejected": fresh_identities["cpu"]
        != identities(changed_token),
        "sealed_cpu_output_mutation_rejected": sealed_identities["cpu"]
        != identities(changed_output),
        "result_digest_mutation_rejected": _evidence_hash_valid(
            fresh_results["cpu"]
        )
        and not _evidence_hash_valid(changed_result),
        "false_declared_gate_rejected": all(fresh_results["cpu"]["gates"].values())
        and not all({**fresh_results["cpu"]["gates"], "evil": False}.values()),
        "core_swap_rejected": fresh_results["cpu"]["core_before"]["archive_hash"]
        != "0" * 64,
        "package_swap_rejected": fresh_results["cpu"]["package_installs"]["python"]
        ["archive_hash"]
        != "0" * 64,
        "teacher_presence_rejected": fresh_results["cpu"]["teacher_model_loaded"]
        is False,
        "training_presence_rejected": fresh_results["cpu"]["training_performed"]
        is False,
        "hardware_relabel_rejected": preparation["hardware"]["fingerprint_sha256"]
        == protocol["development_hardware_fingerprint_sha256"],
        "phase8_relabel_rejected": preparation["phase8_certified"] is False
        and collection["phase8_certified"] is False,
        "junit_failure_rejected": patch_tests["failures"] == 0
        and {**patch_tests, "failures": 1} != patch_tests,
        "fresh_depth_rejected": len(fresh_rows["cpu"]) == 244,
    }
    passed = all(gates.values()) and all(mutations.values())
    result = {
        "format": "abi-capability-compiler-phase8-local-clean-rehearsal-verify-result/1",
        "status": "PASS_PHASE8_LOCAL_CLEAN_EXPORT_REHEARSAL_SAME_MACHINE"
        if passed
        else "FAIL_PHASE8_LOCAL_CLEAN_EXPORT_REHEARSAL",
        "protocol_sha256": protocol_sha,
        "product_protocol_sha256": product_sha,
        "devices": {
            device: {
                "verified_gates": recomputed[device],
                "median_core_bytes_per_second": fresh_results[device]["core_metrics"]
                ["median_bytes_per_second"],
                "throughput_ratio": fresh_results[device]["comparisons"]
                ["median_bytes_per_second_ratio_vs_baseline"],
                "throughput_retention": fresh_results[device]["comparisons"]
                ["core_throughput_retention_vs_phase4"],
                "cold_ttft_seconds": fresh_results[device]["cold"]
                ["time_to_first_output_from_cold_start_seconds"],
                "peak_process_rss_delta_bytes": fresh_results[device]["memory"]
                ["peak_process_rss_delta_bytes"],
                "peak_cuda_allocated_bytes": fresh_results[device]["memory"]
                ["peak_cuda_allocated_bytes"],
            }
            for device in ("cpu", "cuda")
        },
        "fresh_cross_device_output_identities": len(fresh_identities["cpu"])
        if fresh_identities["cpu"] == fresh_identities["cuda"]
        else 0,
        "fresh_sealed_functional_identities": sum(
            len(fresh_identities[device])
            for device in ("cpu", "cuda")
            if fresh_identities[device] == sealed_identities[device]
        ),
        "layercake_patch_tests": patch_tests,
        "gates": gates,
        "mutations": mutations,
        "model_inference_performed": False,
        "training_performed": False,
        "same_development_hardware": True,
        "phase8_certified": False,
        "claim_boundary": "One clean-export rehearsal on the original development hardware. A pass excludes undeclared working-tree dependencies for the tested packet but is not independent operation or hardware, Phase 8, release, human preference, arbitrary hardware, or universal ABI superiority.",
    }
    result["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = (root / args.protocol).resolve()
    if args.preflight:
        result = preflight(root, protocol_path)
    elif args.output:
        result = verify(root, protocol_path, (root / args.output).resolve())
    else:
        raise Phase3Error("select preflight or output")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
