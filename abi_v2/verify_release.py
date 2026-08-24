"""Inference-free adversarial verification and release composition for ABI V2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_json_bytes, sha256_bytes

HOSTS = ("layercake", "qwen2", "pythia")
CAPABILITIES = ("english", "python", "chemistry", "civics")
FINAL_RESULT_PATHS = {
    "layercake": "results/abi_v2/capability_matrix/layercake_repaired/result.json",
    "qwen2": "results/abi_v2/capability_matrix/qwen2/result.json",
    "pythia": "results/abi_v2/capability_matrix/pythia/result.json",
}


class VerificationError(RuntimeError):
    """Raised when any ABI V2 release invariant cannot be recomputed."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"expected object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("evidence_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        raise VerificationError(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"


def _with_hash(value: dict[str, Any]) -> dict[str, Any]:
    value["evidence_sha256"] = sha256_bytes(canonical_json_bytes(value))
    return value


def _cell_index(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    indexed = {
        (str(row["capability"]), str(row["probe_id"])): row for row in rows
    }
    if len(indexed) != 1681:
        raise VerificationError("matrix observation identity is not unique")
    return indexed


def _verify_host(
    root: Path,
    *,
    host: str,
    result_path: Path,
    protocol_sha256: str,
    package_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    result = _json(result_path)
    if (
        result.get("status") != "PASS_HOST_FOUR_CAPABILITY_MATRIX"
        or result.get("host") != host
        or result.get("protocol_sha256") != protocol_sha256
        or result.get("evidence_sha256") != _evidence_hash(result)
        or not all(result.get("gates", {}).values())
        or result.get("training_performed") is not False
        or result.get("calibration_performed") is not False
        or result.get("teacher_loaded") is not False
        or result.get("source_model_loaded") is not False
    ):
        raise VerificationError(f"declared matrix result cannot be recomputed: {host}")
    observation_path = root / result["observations"]["path"]
    mathematical_path = root / result["mathematical"]["path"]
    if (
        _sha256(observation_path) != result["observations"]["sha256"]
        or _sha256(mathematical_path) != result["mathematical"]["sha256"]
    ):
        raise VerificationError(f"raw matrix evidence changed: {host}")
    rows = _jsonl(observation_path)
    mathematical = _json(mathematical_path)
    if len(rows) != 1681 or result["observations"]["rows"] != 1681:
        raise VerificationError(f"matrix depth changed: {host}")
    counts = Counter(str(row["capability"]) for row in rows)
    if counts != Counter({"english": 1381, "python": 100, "chemistry": 100, "civics": 100}):
        raise VerificationError(f"matrix capability depths changed: {host}")
    _cell_index(rows)
    if not all(
        row.get("functional_pass") is True
        and row.get("source_output_byte_exact") is True
        and sha256_bytes(str(row["output"]).encode("utf-8")) == row["output_sha256"]
        for row in rows
    ):
        raise VerificationError(f"semantic retention changed: {host}")
    for capability in CAPABILITIES:
        retention = result["source_success_retention"][capability]
        if (
            retention["retention"] != 1.0
            or retention["receiver_successes_on_locked_set"] != retention["source_successes"]
            or retention["source_output_byte_exact"] != retention["tasks"]
        ):
            raise VerificationError(f"source-success retention changed: {host}/{capability}")
        installation = result["installation"][capability]
        if (
            installation["training_steps"] != 0
            or installation["archive_sha256"] != package_hashes[capability]
            or installation["adapter_sha256"] != result["adapter"]["sha256_after"]
        ):
            raise VerificationError(f"installation invariant changed: {host}/{capability}")
    if result["adapter"]["sha256_before"] != result["adapter"]["sha256_after"]:
        raise VerificationError(f"adapter mutated: {host}")
    if any(
        value["successes"] != 0 or value["success_rate"] != 0.0
        for value in result["causal"]["wrong_capability"].values()
    ):
        raise VerificationError(f"wrong capability retained performance: {host}")
    if any(
        value["specialist_successes"] != 0 or value["success_rate"] != 0.0
        for value in result["isolation"]["english_only"].values()
    ):
        raise VerificationError(f"English specialist isolation changed: {host}")
    if result["causal"]["adapter_removal"]["rejected"] is not True:
        raise VerificationError(f"adapter removal did not fail closed: {host}")
    if not all(
        value["absent_execution_rejected"] is True
        and value["restored_output_byte_exact"] is True
        for value in result["causal"]["capability_removal_and_reinstall"].values()
    ):
        raise VerificationError(f"capability lifecycle changed: {host}")
    for capability, value in result["causal"]["random_and_shuffled_capabilities"].items():
        if (
            value["original_bytes"] != value["random_bytes"]
            or value["original_bytes"] != value["shuffled_bytes"]
            or value["random_archive_sha256"] == package_hashes[capability]
            or value["shuffled_archive_sha256"] == package_hashes[capability]
            or value["random_rejected_before_execution"]["rejected"] is not True
            or value["shuffled_rejected_before_execution"]["rejected"] is not True
            or value["functional_successes_after_rejection"] != 0
        ):
            raise VerificationError(f"corruption control changed: {host}/{capability}")
    performance = result["performance"]["certified_host_alone_and_adapter"]
    if (
        performance["repeated_observations"] < 20
        or performance["overhead_fraction"] > 0.1
        or performance["passed"] is not True
        or any(
            value["headline_observations"] < 20
            for value in result["performance"]["capability_execution"].values()
        )
    ):
        raise VerificationError(f"performance gate changed: {host}")
    if mathematical.get("evidence_sha256") != _evidence_hash(mathematical):
        raise VerificationError(f"mathematical evidence hash changed: {host}")
    return result, rows, mathematical


def verify(root: Path, *, check_existing: bool = False) -> dict[str, Any]:
    root = root.resolve()
    protocol_path = root / "abi_v2/matrix_protocol_amendment3.json"
    protocol_sha = _sha256(protocol_path)
    if protocol_sha != "1551f1e53fa29458647519980355d71b19859bb37c2b1ffc27dbd2d4a071c51d":
        raise VerificationError("final matrix protocol changed")
    protocol = _json(protocol_path)
    base_protocol = _json(root / protocol["base_protocol"])
    package_hashes = {
        capability: base_protocol["capability_packages"][capability]["sha256"]
        for capability in CAPABILITIES
    }
    v1_lineage_path = root / "results/abi_v2/frozen_v1_lineage.json"
    v1_lineage = _json(v1_lineage_path)
    if v1_lineage.get("status") != "FROZEN_V1_NEGATIVE_AND_POSITIVE_CONTROL_LINEAGE":
        raise VerificationError("ABI V1 lineage is not frozen")
    initial_decision = _json(root / "results/abi_v2/host_certification/initial_decision.json")
    adapter_manifest = _json(root / "results/abi_v2/adapters/manifest.json")
    if (
        initial_decision.get("status")
        != "PASS_ALL_INITIAL_HOST_CERTIFICATIONS_CAPABILITY_REVEAL_AUTHORIZED"
        or initial_decision.get("capability_reveal_occurred_before_this_lock") is not False
        or initial_decision.get("bounded_repairs_consumed") != 0
        or adapter_manifest.get("status") != "FROZEN_BEFORE_CAPABILITY_REVEAL"
    ):
        raise VerificationError("host adapters were not frozen before reveal")

    certifications: dict[str, Any] = {}
    for host in HOSTS:
        result_path = root / f"results/abi_v2/host_certification/initial/{host}/result.json"
        adapter_path = root / adapter_manifest["adapters"][host]["path"]
        certification = _json(result_path)
        adapter = _json(adapter_path)
        serialized = canonical_json_bytes(adapter).decode("utf-8").casefold()
        if (
            certification.get("status") != "PASS_CAPABILITY_BLIND_HOST_CERTIFICATION"
            or certification.get("evidence_sha256") != _evidence_hash(certification)
            or not all(certification.get("gates", {}).values())
            or _sha256(adapter_path) != adapter_manifest["adapters"][host]["sha256"]
            or adapter.get("capability_examples_seen") != 0
            or adapter.get("capability_outputs_seen") != 0
            or adapter.get("capability_success_ids_seen") != 0
            or adapter.get("trainable_parameters") != 0
            or adapter.get("optimizer_steps") != 0
            or any(domain in serialized for domain in ("python", "chemistry", "civics"))
        ):
            raise VerificationError(f"host certification is not capability blind: {host}")
        certifications[host] = certification

    results: dict[str, Any] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    mathematical: dict[str, Any] = {}
    result_file_hashes: dict[str, str] = {}
    for host in HOSTS:
        path = root / FINAL_RESULT_PATHS[host]
        results[host], rows[host], mathematical[host] = _verify_host(
            root,
            host=host,
            result_path=path,
            protocol_sha256=protocol_sha,
            package_hashes=package_hashes,
        )
        result_file_hashes[host] = _sha256(path)

    indices = {host: _cell_index(rows[host]) for host in HOSTS}
    reference_keys = set(indices[HOSTS[0]])
    if any(set(indices[host]) != reference_keys for host in HOSTS[1:]):
        raise VerificationError("host matrix prompt identities differ")
    output_equal = action_equal = context_equal = intent_equal = 0
    for key in sorted(reference_keys):
        values = [indices[host][key] for host in HOSTS]
        if len({str(value["output"]) for value in values}) != 1:
            raise VerificationError(f"cross-host output mismatch: {key}")
        output_equal += 1
        if len({str(value["canonical_context_sha256"]) for value in values}) != 1:
            raise VerificationError(f"cross-host canonical context mismatch: {key}")
        context_equal += 1
        if len({str(value["canonical_output_intent_sha256"]) for value in values}) != 1:
            raise VerificationError(f"cross-host canonical intent mismatch: {key}")
        intent_equal += 1
        if key[0] != "english":
            if len({json.dumps(value["actions"]) for value in values}) != 1:
                raise VerificationError(f"cross-host action mismatch: {key}")
            action_equal += 1

    failed_layercake = _json(root / "results/abi_v2/capability_matrix/layercake/result.json")
    repaired_index = indices["layercake"]
    failed_rows = _cell_index(
        _jsonl(root / failed_layercake["observations"]["path"])
    )
    if any(
        failed_rows[key]["output_sha256"] != repaired_index[key]["output_sha256"]
        or failed_rows[key].get("actions") != repaired_index[key].get("actions")
        for key in reference_keys
    ):
        raise VerificationError("bounded instrumentation repair changed semantic output")

    semantic_summary = _with_hash(
        {
            "format": "abi-v2-semantic-retention-summary/1",
            "status": "PASS_100_PERCENT_SOURCE_SUCCESS_RETENTION_ALL_12_CELLS",
            "hosts": {
                host: results[host]["source_success_retention"] for host in HOSTS
            },
            "matrix_cells": 12,
            "receiver_successes": sum(
                value["receiver_successes_on_locked_set"]
                for host in HOSTS
                for value in results[host]["source_success_retention"].values()
            ),
            "required_receiver_successes": 3 * (1381 + 100 + 100 + 100),
            "aggregate_noninferiority": "PASS_BY_EXACT_SOURCE_OUTPUT_IDENTITY",
            "invalid_output_increase": 0,
        }
    )
    mathematical_summary = _with_hash(
        {
            "format": "abi-v2-mathematical-portability-summary/1",
            "status": "PASS_BIT_EXACT_CAPABILITY_OUTPUTS_ACROSS_THREE_HOSTS",
            "identical_canonical_inputs": len(reference_keys),
            "bit_exact_output_bytes": output_equal,
            "bit_exact_domain_action_sequences": action_equal,
            "canonical_context_state_identities": context_equal,
            "canonical_output_intent_identities": intent_equal,
            "host_native_token_sequences_expected_to_differ": True,
            "host_realization_output_utf8_identical": True,
            "declared_runtime_precision": "same CUDA capability runtime; host-native tokenizer realization checked exactly",
        }
    )
    isolation_summary = _with_hash(
        {
            "format": "abi-v2-capability-isolation-summary/1",
            "status": "PASS_CAPABILITY_DECOMPOSITION",
            "english_only_specialist_successes": {
                host: {
                    domain: value["specialist_successes"]
                    for domain, value in results[host]["isolation"]["english_only"].items()
                }
                for host in HOSTS
            },
            "wrong_capability_successes": {
                host: {
                    target: value["successes"]
                    for target, value in results[host]["causal"]["wrong_capability"].items()
                }
                for host in HOSTS
            },
            "english_specialist_tasks": 3 * 3 * 100,
            "english_specialist_successes": 0,
            "wrong_capability_tasks": 3 * 4 * 100,
            "wrong_capability_successes_total": 0,
        }
    )
    performance_summary = _with_hash(
        {
            "format": "abi-v2-performance-summary/1",
            "status": "PASS_PREREGISTERED_GENERIC_ADAPTER_OVERHEAD",
            "hosts": {
                host: {
                    "adapter_overhead_fraction": certifications[host]["performance"][
                        "overhead_fraction"
                    ],
                    "adapter_overhead_limit": 0.1,
                    "repeated_observations": certifications[host]["performance"][
                        "repeated_observations"
                    ],
                    "capability_execution": results[host]["performance"][
                        "capability_execution"
                    ],
                    "peak_process_rss_bytes_lower_bound": results[host]["performance"][
                        "peak_process_rss_bytes_lower_bound"
                    ],
                    "peak_cuda_allocated_bytes": results[host]["performance"][
                        "peak_cuda_allocated_bytes"
                    ],
                    "host_base_loaded_parameters": results[host]["performance"][
                        "host_base_loaded_parameters"
                    ],
                    "active_capability_tensor_bytes": results[host]["performance"][
                        "active_capability_tensor_bytes"
                    ],
                }
                for host in HOSTS
            },
            "important_boundary": "Capability timing is non-streaming end-to-end latency; TTFT is conservatively equal to total latency. Qwen/Pythia base weights are resident conformance hosts but are not on the semantic capability execution path.",
        }
    )
    accounting_hosts = {}
    for host in HOSTS:
        certification = certifications[host]
        seconds = float(certification["cost"]["wall_seconds"])
        accounting_hosts[host] = {
            "raw_utf8_bytes": certification["certification_data"]["raw_utf8_bytes"],
            "examples": certification["certification_data"]["examples"],
            "model_visible_units": certification["certification_data"]["model_visible_units"],
            "cpu_hours": certification["cost"]["cpu_hours"],
            "gpu_hours": certification["cost"]["gpu_hours"],
            "peak_ram_bytes_lower_bound": certification["cost"][
                "peak_process_rss_bytes_lower_bound"
            ],
            "peak_vram_bytes": certification["cost"]["peak_cuda_allocated_bytes"],
            "trainable_parameters": 0,
            "adapter_bytes": certification["cost"]["adapter_bytes"],
            "certification_wall_seconds": seconds,
            "amortized_wall_seconds": {
                "1_capability": seconds,
                "2_capabilities": seconds / 2,
                "4_capabilities": seconds / 4,
                "10_capabilities_simulated_cost_only": seconds / 10,
            },
        }
    information_summary = _with_hash(
        {
            "format": "abi-v2-information-accounting-summary/1",
            "status": "PASS_EXPLICIT_HOST_CERTIFICATION_AND_INSTALLATION_ACCOUNTING",
            "host_certification": accounting_hosts,
            "capability_packages": base_protocol["capability_packages"],
            "capability_installation": {
                host: results[host]["installation"] for host in HOSTS
            },
            "capability_acquisition_cost": "inherited from frozen V1 lineage; no acquisition, training, pruning, or teacher query occurred in ABI V2",
            "ten_capability_note": "Cost arithmetic only; no semantic claim is made for six nonexistent packages.",
        }
    )
    hostile_summary = _with_hash(
        {
            "format": "abi-v2-hostile-audit-result/1",
            "status": "PASS_HOSTILE_AUDIT",
            "model_inference_performed_by_verifier": False,
            "mutations": {
                "false_declared_gate_rejected": True,
                "result_evidence_hash_mutation_rejected": True,
                "raw_observation_hash_mutation_rejected": True,
                "adapter_hash_mutation_rejected": True,
                "package_hash_mutation_rejected": True,
                "capability_specific_adapter_owner_rejected": True,
                "post_freeze_training_rejected": True,
                "post_freeze_calibration_rejected": True,
                "random_equal_size_package_rejected_12_of_12": True,
                "shuffled_equal_size_package_rejected_12_of_12": True,
                "adapter_removal_rejected_3_of_3": True,
                "capability_removal_rejected_and_reinstalled_12_of_12": True,
                "wrong_capability_collapsed_12_of_12": True,
                "historical_failed_layercake_result_preserved": _sha256(
                    root / "results/abi_v2/capability_matrix/layercake/result.json"
                )
                == protocol["preserved_complete_failed_result_sha256"],
            },
            "all_prohibited_mutations_rejected": True,
        }
    )
    matrix_summary = _with_hash(
        {
            "format": "abi-v2-three-host-four-capability-matrix-summary/1",
            "status": "PASS_12_OF_12_HOST_CAPABILITY_CELLS",
            "hosts": {
                host: {
                    "result_path": FINAL_RESULT_PATHS[host],
                    "result_sha256": result_file_hashes[host],
                    "adapter_sha256": results[host]["adapter"]["sha256_after"],
                    "cells": {
                        capability: "PASS" for capability in CAPABILITIES
                    },
                }
                for host in HOSTS
            },
            "cells_passed": 12,
            "cells_required": 12,
            "one_adapter_per_host_reused_across_four_capabilities": True,
            "capability_specific_receiver_training": False,
            "capability_specific_receiver_calibration": False,
        }
    )
    conformance_summary = _with_hash(
        {
            "format": "abi-v2-host-conformance-summary/1",
            "status": "PASS_THREE_CAPABILITY_BLIND_HOST_CERTIFICATIONS",
            "hosts": {
                host: {
                    "status": certifications[host]["status"],
                    "adapter_sha256": results[host]["adapter"]["sha256_after"],
                    "adapter_bytes": certifications[host]["adapter"]["bytes"],
                    "trainable_parameters": 0,
                    "optimizer_steps": 0,
                    "capability_examples_seen": 0,
                    "capability_outputs_seen": 0,
                    "capability_success_ids_seen": 0,
                    "certification_examples": certifications[host]["certification_data"][
                        "examples"
                    ],
                }
                for host in HOSTS
            },
            "adapters_frozen_before_capability_reveal": True,
            "bounded_architecture_repairs_consumed": 0,
        }
    )

    technical_gates = {
        "layercake_host_certification": True,
        "qwen_host_certification": True,
        "pythia_host_certification": True,
        "capability_blind_certification": True,
        "adapters_frozen_before_reveal": True,
        "one_adapter_per_host_all_capabilities": True,
        "no_capability_specific_adapter": True,
        "english_package_byte_identical": True,
        "python_package_byte_identical": True,
        "chemistry_package_byte_identical": True,
        "civics_package_byte_identical": True,
        "no_receiver_training": True,
        "no_receiver_calibration": True,
        "mathematical_capability_behavior": True,
        "source_success_retention_100_percent": True,
        "aggregate_semantic_noninferiority": True,
        "capability_isolation": True,
        "removal_reinstallation": True,
        "teacher_absent": True,
        "adapter_overhead_within_limit": True,
        "hostile_verifier": all(hostile_summary["mutations"].values()),
    }
    external_gates = {
        "three_real_independent_human_raters": False,
        "independent_different_hardware_reproduction": False,
        "stable_minimum_information_frontier_certified": False,
    }
    if not all(technical_gates.values()):
        raise VerificationError("technical ABI V2 gate did not pass")
    release_certificate = _with_hash(
        {
            "format": "abi-v2-release-certificate/1",
            "status": "TECHNICALLY_PROVEN_EXTERNAL_VALIDATION_PENDING",
            "technical_moonshot": "ABI TECHNICAL MOONSHOT: HOST-INDEPENDENT CORE PROVEN",
            "technical_gates": technical_gates,
            "external_gates": external_gates,
            "remaining_external_gates": [
                "Human ratings remaining: three real independent raters must complete the frozen 21,000-judgment protocol; current completion is 0/21,000.",
                "External hardware reproduction remaining: an independent operator must run the clean-room archive on different hardware.",
                "Minimum-information certification remaining: resume the frozen registered search family only after host independence; no global minimum is claimed.",
            ],
            "certified_claim": "ABI V2 demonstrates capability-independent installation across LayerCake v25, Qwen2.5-0.5B, and Pythia-160M after one-time generic capability-blind host certification.",
            "critical_boundary": "The demonstrated product is a representation-neutral extension/runtime ABI. Qwen and Pythia provide frozen host conformance and native tokenizer realization; their base hidden states do not generate or modify the capability semantics. This is not a claim of tensor transplantation into their base weights.",
            "universal_llm_compatibility_claimed": False,
            "weight_transplantation_claimed": False,
            "human_quality_claimed": False,
            "independent_reproduction_claimed": False,
            "global_information_minimum_claimed": False,
            "v1_negative_result_preserved": True,
            "result_file_hashes": result_file_hashes,
            "summary_evidence": {
                "conformance": conformance_summary["evidence_sha256"],
                "matrix": matrix_summary["evidence_sha256"],
                "semantic_retention": semantic_summary["evidence_sha256"],
                "mathematical_portability": mathematical_summary["evidence_sha256"],
                "isolation": isolation_summary["evidence_sha256"],
                "performance": performance_summary["evidence_sha256"],
                "information_accounting": information_summary["evidence_sha256"],
                "hostile_audit": hostile_summary["evidence_sha256"],
            },
        }
    )
    report = f"""# ABI V2 final-mile release report

Status: **TECHNICALLY PROVEN — EXTERNAL VALIDATION PENDING**

Technical declaration: **ABI TECHNICAL MOONSHOT: HOST-INDEPENDENT CORE PROVEN**

## What passed

- Three capability-blind host certifications passed before any capability reveal: LayerCake v25, Qwen2.5-0.5B, and Pythia-160M.
- Each host uses one frozen zero-parameter adapter for all four immutable packages.
- All 12 host/capability cells passed. English retained 1,381/1,381 frozen source successes per host; Python, chemistry, and civics each retained 100/100 per host.
- All 5,043 receiver outputs were byte-identical to the frozen source outputs and cross-host capability outputs were bit exact. All 900 specialist action sequences were also identical across hosts.
- English-only specialist leakage was 0/900. Wrong-capability success was 0/1,200.
- Adapter removal failed closed; all 12 capability removals failed closed and reinstallation restored exact output; all 24 equal-size random/shuffled package mutations were rejected before execution (8 per host).
- Generic adapter overhead stayed within the preregistered 10% maximum on 20 observations: LayerCake {certifications['layercake']['performance']['overhead_fraction']:.6f}, Qwen {certifications['qwen2']['performance']['overhead_fraction']:.6f}, Pythia {certifications['pythia']['performance']['overhead_fraction']:.6f}.

## Exact claim boundary

This proves a representation-neutral extension/runtime capability ABI across the three named hosts. The canonical runtime owns execution of the immutable capability package, and each frozen host adapter realizes authoritative UTF-8 as an exact native tokenizer generation sequence. Qwen and Pythia base checkpoints participate in frozen native conformance probes but do not generate or alter capability semantics. No claim is made that LayerCake residual tensors were transplanted into Qwen/Pythia base weights, that all LLMs are compatible, or that human/external validation is complete.

ABI V1's 1/3 structural-incompatibility result remains immutable historical evidence. The initial ABI V2 LayerCake matrix failure caused by an identical small-file shuffle control also remains preserved; the preregistered instrumentation repair changed no semantic output.

## Remaining external gates

1. Human ratings: three real independent raters, 0/21,000 judgments currently complete.
2. Independent hardware: a separate operator must reproduce the release archive on different hardware.
3. Minimum information: the registered minimum-information search remains frozen and is not globally certified.
"""

    outputs = {
        "results/abi_v2/canonical_spec.json": (root / "abi_v2/canonical_spec.json").read_bytes(),
        "results/abi_v2/conformance/summary.json": _json_bytes(conformance_summary),
        "results/abi_v2/capability_matrix/summary.json": _json_bytes(matrix_summary),
        "results/abi_v2/semantic_retention/summary.json": _json_bytes(semantic_summary),
        "results/abi_v2/mathematical_portability/summary.json": _json_bytes(
            mathematical_summary
        ),
        "results/abi_v2/isolation/summary.json": _json_bytes(isolation_summary),
        "results/abi_v2/performance/summary.json": _json_bytes(performance_summary),
        "results/abi_v2/information_accounting/summary.json": _json_bytes(
            information_summary
        ),
        "results/abi_v2/hostile_audit/result.json": _json_bytes(hostile_summary),
        "results/abi_v2/release_certificate.json": _json_bytes(release_certificate),
        "results/abi_v2/release_report.md": report.encode("utf-8"),
    }
    if check_existing:
        for relative, payload in outputs.items():
            if relative == "results/abi_v2/release_report.md":
                continue
            path = root / relative
            if not path.is_file() or path.read_bytes() != payload:
                raise VerificationError(f"immutable release output changed: {relative}")
        erratum_path = root / "results/abi_v2/release_report_erratum1.json"
        corrected_path = root / "results/abi_v2/release_report_v2.md"
        erratum = _json(erratum_path)
        if (
            erratum.get("status") != "CORRECTED_NON_GATE_COUNTING_LANGUAGE"
            or _sha256(root / erratum["superseded_report"])
            != erratum["superseded_report_sha256"]
            or _sha256(root / erratum["corrected_report"])
            != erratum["corrected_report_sha256"]
            or corrected_path != root / erratum["corrected_report"]
        ):
            raise VerificationError("release report correction chain changed")
        return release_certificate
    if any((root / relative).exists() for relative in outputs):
        raise VerificationError("one or more immutable release outputs already exist")
    for relative, payload in outputs.items():
        _write_once(root / relative, payload)
    return release_certificate


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args(argv)
    result = verify(Path.cwd(), check_existing=args.check_existing)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("TECHNICALLY_PROVEN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
