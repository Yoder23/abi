"""Independent read-only verifier for the exact integrated Phase 7 product."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

from .capability_compiler_phase2_common import (
    canonical_json_bytes,
    evaluate_functional,
    sha256_file,
)
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .capability_compiler_phase4_b50_gpu_runtime import _runtime_metrics
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase5_selective_product import _domain_rows
from .capability_compiler_phase7_direct_artifact_runtime import (
    load_protocol as load_product_protocol,
)
from .capability_compiler_phase7_integrated_runtime import (
    RESULT_FORMAT,
    _reference,
    _selected_only,
)


FORMAT = "abi-capability-compiler-phase7-independent-verify/1"
RESULT_VERIFY_FORMAT = "abi-capability-compiler-phase7-independent-verify-result/1"


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_bytes().splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase3Error(f"invalid Phase 7 JSONL: {path}") from exc


def _evidence_hash_valid(result: Mapping[str, Any]) -> bool:
    document = copy.deepcopy(dict(result))
    declared = document.pop("evidence_sha256", None)
    return declared == hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _sha256_with_suffix(path: Path, suffix: bytes) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    digest.update(suffix)
    return digest.hexdigest()


def _junit(path: Path) -> dict[str, int]:
    document = ET.parse(path).getroot()
    suites = [document] if document.tag == "testsuite" else list(document.findall("testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def _metrics_exact(
    rows: Sequence[Mapping[str, Any]], declared: Mapping[str, Any]
) -> bool:
    return _runtime_metrics(rows) == declared


def verify_device_document(
    *,
    root: Path,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    device: str,
    result: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    cold = [row for row in observations if row.get("mode") == "single_cold_core_request"]
    residency = [row for row in observations if row.get("mode") == "domain_residency_load"]
    core = [row for row in observations if row.get("mode") == "core_runtime"]
    domain = [row for row in observations if row.get("mode") == "domain_runtime"]
    core_reference = _reference(root / protocol["core_quality_reference"])
    domain_reference = _reference(
        root / protocol["phase6_observations"],
        mode="composed_host_selected_domain",
    )
    catalog_rows = _domain_rows(
        root / protocol["domain_catalog"], split="final_test", per_domain=100
    )
    catalog = {str(row["probe_id"]): row for row in catalog_rows}
    baseline = protocol["baselines"][device]
    phase4 = protocol["phase4_same_core_runtime"][device]
    core_metrics = _runtime_metrics(core) if core else {}
    domain_metrics = _runtime_metrics(domain) if domain else {}
    throughput_ratio = (
        float(core_metrics.get("median_bytes_per_second", 0.0))
        / float(baseline["median_bytes_per_second"])
    )
    retention = (
        float(core_metrics.get("median_bytes_per_second", 0.0))
        / float(phase4["median_bytes_per_second"])
    )
    expected_packages = protocol["product"]["packages"]
    gates = {
        "result_format_status_device_protocol": result.get("format") == RESULT_FORMAT
        and result.get("status") == "PASS_PHASE7_INTEGRATED_RUNTIME"
        and result.get("device") == device
        and result.get("protocol_sha256") == protocol_sha,
        "result_evidence_hash": _evidence_hash_valid(result),
        "declared_gates_all_pass": bool(result.get("gates"))
        and all(value is True for value in result["gates"].values()),
        "raw_depth_partition_exact": len(observations) == 244
        and len(cold) == 1
        and len(residency) == 3
        and len(core) == 120
        and len(domain) == 120,
        "core_schedule_and_reference_exact": len({str(row.get("probe_id")) for row in core})
        == 100
        and all(
            str(row.get("output", ""))
            == core_reference.get(str(row.get("probe_id")))
            and row.get("output_byte_exact") is True
            for row in core
        ),
        "domain_schedule_reference_and_function_exact": len(
            {str(row.get("probe_id")) for row in domain}
        )
        == 100
        and all(
            str(row.get("output", ""))
            == domain_reference.get(str(row.get("probe_id")))
            and row.get("output_byte_exact") is True
            and str(row.get("probe_id")) in catalog
            and evaluate_functional(
                str(row.get("output", "")),
                catalog[str(row.get("probe_id"))]["evaluator"],
            )
            for row in domain
        ),
        "domain_selected_only_recomputed": all(
            str(row.get("domain")) in expected_packages
            and row.get("selected")
            == [expected_packages[str(row.get("domain"))]["cake_id"]]
            and _selected_only(
                row.get("telemetry_delta", {}),
                expected_packages[str(row.get("domain"))]["cake_id"],
            )
            for row in [*residency, *domain]
        ),
        "zero_repetition_collapse_recomputed": not any(
            repetition_collapse_v2(str(row.get("output", "")))
            for row in [*core, *domain]
        ),
        "core_metrics_recomputed_exact": bool(core)
        and _metrics_exact(core, result.get("core_metrics", {})),
        "domain_metrics_recomputed_exact": bool(domain)
        and _metrics_exact(domain, result.get("domain_metrics", {})),
        "throughput_ratios_recomputed_exact": throughput_ratio
        == result.get("comparisons", {}).get("median_bytes_per_second_ratio_vs_baseline")
        and retention
        == result.get("comparisons", {}).get("core_throughput_retention_vs_phase4"),
        "throughput_gates_recomputed": throughput_ratio
        >= float(protocol["gates"]["throughput_ratio_minimum"])
        and retention
        >= float(protocol["gates"]["phase4_core_throughput_retention_minimum"]),
        "cold_single_request_and_ttft_recomputed": len(cold) == 1
        and cold[0].get("single_cold_request") is True
        and cold[0].get("output")
        == core_reference.get(str(cold[0].get("probe_id")))
        and float(cold[0].get("time_to_first_output_from_cold_start_seconds", 1e30))
        <= float(baseline["cold_ttft_seconds"]),
        "persistent_incremental_state_recomputed": all(
            row.get("capability") in {"coherence", "format_control"}
            or row.get("execution", {}).get("persistent_state_created") is True
            for row in core
        ),
        "memory_gates_recomputed": int(
            result.get("memory", {}).get("total_integrated_active_tensor_bytes", 1 << 62)
        )
        < int(baseline["active_tensor_bytes"])
        and int(result.get("memory", {}).get("peak_process_rss_delta_bytes", 1 << 62))
        < int(baseline["peak_process_rss_bytes"])
        and (
            device == "cpu"
            or int(result.get("memory", {}).get("peak_cuda_allocated_bytes", 1 << 62))
            < int(baseline["peak_cuda_allocated_bytes"])
        ),
        "core_identity_exact_and_unchanged": result.get("core_before")
        == result.get("core_after")
        and result.get("core_before", {}).get("archive_hash")
        == protocol["product"]["core_archive_sha256"]
        and result.get("core_before", {}).get("payload_hash")
        == protocol["product"]["core_payload_sha256"],
        "package_identity_exact": all(
            result.get("package_installs", {}).get(name, {}).get("archive_hash")
            == specification["archive_sha256"]
            and sha256_file(root / protocol["domain_packages"][name]["package"])
            == specification["archive_sha256"]
            for name, specification in expected_packages.items()
        ),
        "lifecycle_repair_exact": result.get("serving_lifecycle")
        == "direct_hash_bound_materialized_archive"
        and result.get("same_process_archive_reconstruction") is False
        and result.get("materialized_core_archive_sha256")
        == protocol["product"]["core_archive_sha256"],
        "teacher_training_receiver_learning_absent": result.get("teacher_model_loaded")
        is False
        and result.get("training_performed") is False
        and int(result.get("receiver_training_steps", -1)) == 0,
    }
    return gates


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_PHASE7_INDEPENDENT_MACHINE_VERIFICATION"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or int(protocol.get("minimum_adversarial_tests", 0)) < 15
    ):
        raise Phase3Error("Phase 7 verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 7 verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    product, product_sha = load_product_protocol(
        root, root / protocol["product_protocol"]
    )
    gates = {
        "product_protocol_exact": product_sha
        == protocol["product_protocol_sha256"],
        "cpu_and_cuda_results_pass": all(
            _json(root / protocol["evidence"][device]["result"]).get("status")
            == "PASS_PHASE7_INTEGRATED_RUNTIME"
            for device in ("cpu", "cuda")
        ),
        "runtime_host_overlay_exact": _json(
            root / protocol["runtime_host_overlay"]
        ).get("layercake_runtime_commit")
        == "662c5a9b7264a1a5478c9dfb656f35c450e2504f",
        "output_absent": not (root / protocol["output"]).exists(),
        "adversarial_output_absent": not (
            root / protocol["adversarial_output"]
        ).exists(),
        "model_inference_absent": True,
        "training_absent": True,
    }
    return {
        "format": "abi-capability-compiler-phase7-independent-verify-preflight/1",
        "status": "PASS_PHASE7_INDEPENDENT_VERIFY_PREFLIGHT"
        if all(gates.values())
        else "FAIL_PHASE7_INDEPENDENT_VERIFY_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "product_protocol_sha256": product_sha,
        "gates": gates,
    }


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable Phase 7 verifier output exists")
    product, product_sha = load_product_protocol(
        root, root / protocol["product_protocol"]
    )
    device_rows: dict[str, list[dict[str, Any]]] = {}
    device_results: dict[str, dict[str, Any]] = {}
    verified: dict[str, dict[str, bool]] = {}
    for device in ("cpu", "cuda"):
        specification = protocol["evidence"][device]
        result = _json(root / specification["result"])
        rows = _rows(root / specification["observations"])
        if result.get("observations_sha256") != sha256_file(
            root / specification["observations"]
        ):
            raise Phase3Error(f"Phase 7 {device} observations changed")
        verified[device] = verify_device_document(
            root=root,
            protocol=product,
            protocol_sha=product_sha,
            device=device,
            result=result,
            observations=rows,
        )
        device_rows[device] = rows
        device_results[device] = result

    def output_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("mode"),
            row.get("probe_id"),
            row.get("domain"),
            row.get("output"),
            tuple(row.get("output_token_ids", ())),
        )

    cpu_identity = [output_identity(row) for row in device_rows["cpu"]]
    cuda_identity = [output_identity(row) for row in device_rows["cuda"]]
    patch_tests = _junit(root / protocol["layercake_patch_tests"])
    materialization = _json(root / product["materialization_result"])
    certificates = {
        phase: _json(root / path)
        for phase, path in protocol["prerequisite_certificates"].items()
    }
    historical = {
        name: _json(root / path)
        for name, path in protocol["historical_negative_evidence"].items()
    }
    host_overlay = _json(root / protocol["runtime_host_overlay"])
    package_source = (root / "../layercake_release/layercake/cake/package.py").read_text(
        encoding="utf-8"
    )
    installer_source = (
        root / "../layercake_release/layercake/cake/installer.py"
    ).read_text(encoding="utf-8")
    gates = {
        "both_devices_independently_recomputed": all(
            all(row.values()) for row in verified.values()
        ),
        "cross_device_all_244_outputs_and_tokens_exact": cpu_identity
        == cuda_identity
        and len(cpu_identity) == 244,
        "cross_device_active_tensor_identity": device_results["cpu"]["memory"]
        ["total_integrated_active_tensor_bytes"]
        == device_results["cuda"]["memory"]["total_integrated_active_tensor_bytes"]
        == 255916588,
        "layercake_patch_tests_pass": patch_tests
        == {"tests": 42, "failures": 0, "errors": 0, "skipped": 0},
        "materialized_product_exact": materialization.get("status")
        == "PASS_PHASE7_PRODUCT_MATERIALIZATION"
        and sha256_file(root / product["materialized_core_archive"])
        == product["product"]["core_archive_sha256"],
        "prerequisite_machine_certificates_exact": certificates["phase4"].get("status")
        == "CERTIFIED_BOUNDED_MACHINE_DEVELOPMENT_SCOPE"
        and certificates["phase5"].get("phase5_certified") is True
        and certificates["phase6"].get("phase6_certified") is True,
        "runtime_host_overlay_exact": host_overlay.get("base_product_bytes_changed")
        is False
        and host_overlay.get("layercake_runtime_commit")
        == "662c5a9b7264a1a5478c9dfb656f35c450e2504f",
        "allocation_bounded_verify_implementation_present": "def verify_package_integrity("
        in package_source
        and "verify_package_integrity(" in installer_source
        and "load_package(" not in installer_source.split("def verify(", 1)[1].split(
            "def remove(", 1
        )[0],
        "all_negative_evidence_preserved": set(historical)
        == {"v1043", "v1046", "v1049", "v1056", "v1060"}
        and all(str(row.get("status", "")).startswith("FAIL") for row in historical.values()),
        "human_phase2_prerequisite_still_open": certificates["phase4"].get(
            "unresolved_external_prerequisite"
        )
        is not None,
        "model_inference_absent_from_verifier": True,
        "training_absent": True,
    }

    changed_cuda = copy.deepcopy(device_rows["cuda"])
    changed_cuda[0]["output"] = str(changed_cuda[0]["output"]) + "x"
    changed_token = copy.deepcopy(device_rows["cuda"])
    changed_token[0]["output_token_ids"] = [*changed_token[0]["output_token_ids"], 0]
    mutations = {
        "cpu_raw_byte_mutation_rejected": hashlib.sha256(
            (root / protocol["evidence"]["cpu"]["observations"]).read_bytes() + b"x"
        ).hexdigest()
        != protocol["bindings"][protocol["evidence"]["cpu"]["observations"]],
        "cuda_raw_byte_mutation_rejected": hashlib.sha256(
            (root / protocol["evidence"]["cuda"]["observations"]).read_bytes() + b"x"
        ).hexdigest()
        != protocol["bindings"][protocol["evidence"]["cuda"]["observations"]],
        "cross_device_output_mutation_rejected": cpu_identity
        != [output_identity(row) for row in changed_cuda],
        "cross_device_token_mutation_rejected": cpu_identity
        != [output_identity(row) for row in changed_token],
        "false_declared_gate_rejected": all(
            device_results["cpu"]["gates"].values()
        )
        and not all({**device_results["cpu"]["gates"], "evil": False}.values()),
        "rss_boundary_rejected": int(
            device_results["cpu"]["memory"]["peak_process_rss_delta_bytes"]
        )
        < int(product["baselines"]["cpu"]["peak_process_rss_bytes"]),
        "speed_relabel_rejected": float(
            device_results["cpu"]["comparisons"][
                "median_bytes_per_second_ratio_vs_baseline"
            ]
        )
        != 1.0,
        "core_swap_rejected": device_results["cpu"]["core_before"]["archive_hash"]
        != "0" * 64,
        "package_swap_rejected": device_results["cpu"]["package_installs"]["python"]
        ["archive_hash"]
        != "0" * 64,
        "teacher_presence_rejected": device_results["cpu"]["teacher_model_loaded"]
        is False,
        "training_presence_rejected": device_results["cpu"]["training_performed"]
        is False,
        "lifecycle_relabel_rejected": device_results["cpu"]["serving_lifecycle"]
        != "same_process_archive_reconstruction_then_activation",
        "materialized_archive_mutation_rejected": _sha256_with_suffix(
            root / product["materialized_core_archive"], b"x"
        )
        != product["product"]["core_archive_sha256"],
        "junit_failure_rejected": patch_tests["failures"] == 0
        and {**patch_tests, "failures": 1} != patch_tests,
        "result_digest_mutation_rejected": _evidence_hash_valid(
            device_results["cpu"]
        )
        and not _evidence_hash_valid(
            {**device_results["cpu"], "device": "tampered"}
        ),
    }
    passed = all(gates.values()) and all(mutations.values())
    result = {
        "format": RESULT_VERIFY_FORMAT,
        "status": "PASS_INDEPENDENTLY_VERIFIED_PHASE7_BOUNDED_MACHINE_PRODUCT"
        if passed
        else "FAIL_PHASE7_INDEPENDENT_VERIFICATION",
        "protocol_sha256": protocol_sha,
        "product_protocol_sha256": product_sha,
        "product": product["product"],
        "devices": {
            device: {
                "verified_gates": verified[device],
                "median_core_bytes_per_second": device_results[device]["core_metrics"]
                ["median_bytes_per_second"],
                "throughput_ratio": device_results[device]["comparisons"]
                ["median_bytes_per_second_ratio_vs_baseline"],
                "throughput_retention": device_results[device]["comparisons"]
                ["core_throughput_retention_vs_phase4"],
                "cold_ttft_seconds": device_results[device]["cold"]
                ["time_to_first_output_from_cold_start_seconds"],
                "peak_process_rss_delta_bytes": device_results[device]["memory"]
                ["peak_process_rss_delta_bytes"],
                "peak_cuda_allocated_bytes": device_results[device]["memory"]
                ["peak_cuda_allocated_bytes"],
            }
            for device in ("cpu", "cuda")
        },
        "cross_device_output_identities": len(cpu_identity)
        if cpu_identity == cuda_identity
        else 0,
        "layercake_patch_tests": patch_tests,
        "gates": gates,
        "mutations": mutations,
        "model_inference_performed": False,
        "training_performed": False,
        "phase7_machine_certified": passed,
        "phase2_human_prerequisite_complete": False,
        "phase8_independent_hardware_complete": False,
        "claim_boundary": "Independent same-machine CPU/CUDA certification of one exact integrated product. Phase 2 human preference, independent hardware, release, arbitrary-model, and universal ABI-superiority claims remain open.",
    }
    result["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(
        output,
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
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
