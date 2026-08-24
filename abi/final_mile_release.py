"""Assemble a signed, non-promotional ABI final-mile release family."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .final_mile import FinalMileError, sha256_file

PACKAGE_SOURCES = {
    "english-substrate.abi": Path(
        "results/abi_capability_compiler_phase7_integrated/materialized_v1052/phase7-final-english-core.cake"
    ),
    "chemistry-capability.abi": Path(
        "results/abi_moonshot/packages/abi-chemistry-token-plan-seed9824.cake"
    ),
    "civics-capability.abi": Path(
        "results/abi_moonshot/packages/abi-civics-token-plan-seed9824.cake"
    ),
    "python-capability.abi": Path(
        "results/abi_moonshot/packages/abi-python-token-plan-seed9824.cake"
    ),
}

EXPECTED_PACKAGE_HASHES = {
    "english-substrate.abi": "acb787b3ffa0153c57d88cd37ba81c3f00b370d4ca4937e659cd4c775851f25d",
    "chemistry-capability.abi": "f9c9b2668fda5ef6b92844c1b7097fbdf8ff0daaae51f5b86f72d4a49000abeb",
    "civics-capability.abi": "634ce66958859ec36dc1fbdf5ef34d6d2a9949d10cf2348a68c245d8c325d604",
    "python-capability.abi": "f1defaef2771ced336a332572a2d2f0e1e542399c877d182c48a6cd2e199231d",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalMileError(f"expected JSON object: {path}")
    return value


def _json(path: Path, value: Any) -> None:
    if path.exists():
        raise FinalMileError(f"immutable release file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _place_immutable(source: Path, target: Path) -> None:
    if target.exists():
        raise FinalMileError(f"immutable release package already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    try:
        os.chmod(target, stat.S_IREAD)
    except OSError:
        pass


def _source_success_lock(root: Path) -> dict[str, Any]:
    outputs = root / (
        "results/abi_capability_compiler_phase4_clarification_route_replication/"
        "B40-seed104729-v927/evaluation/development_outputs.jsonl"
    )
    rows = [json.loads(line) for line in outputs.read_bytes().splitlines() if line]
    successes = sorted(str(row["probe_id"]) for row in rows if row["functional_pass_v1"])
    if len(rows) != 1400 or len(successes) != 1381 or len(set(successes)) != len(successes):
        raise FinalMileError("frozen source-success set changed")
    protocol = root / "ABI_CAPABILITY_COMPILER_PHASE4_CLARIFICATION_ROUTE_REPLICATION_PROTOCOL_V925.json"
    return {
        "format": "abi-source-success-lock/1",
        "status": "FROZEN_BEFORE_ANY_QUALIFYING_CROSS_HOST_RECEIVER_EVALUATION",
        "source_host": "layercake-v25-seed104729",
        "source_outputs": {
            "path": outputs.relative_to(root).as_posix(),
            "sha256": sha256_file(outputs),
            "observations": len(rows),
        },
        "success_rule": "functional_pass_v1 == true",
        "successful_task_ids": successes,
        "successful_tasks": len(successes),
        "evaluator_and_decoding_lock": {
            "protocol": protocol.relative_to(root).as_posix(),
            "protocol_sha256": sha256_file(protocol),
            "implementation_sha256": "934fa64e3e852892e491caad70128e22bf5f23f0e1530bcdc9bb504c8b5a20a0",
        },
        "required_receiver_retention": 1.0,
        "receiver_evaluation_status": "NOT_RUN_STRUCTURAL_ABI_REJECTION",
    }


def _information_ledger(root: Path) -> dict[str, Any]:
    phase1 = _object(root / "evidence/current/ABI_CAPABILITY_COMPILER_PHASE1_CERTIFICATE_V1.json")
    pack = _object(root / "ABI_CAPABILITY_COMPILER_PHASE4_B40_BASELINE_PACK_RESULT_V990.json")
    product = _object(
        root / "results/abi_capability_compiler_phase7_integrated/materialized_v1052/result.json"
    )
    source = phase1["source"]
    selected = phase1["normalized_ir"]
    b40 = pack["imported_information"]
    return {
        "format": "abi-imported-information-ledger/1",
        "status": "COMPLETE_FOR_SOURCE_EXPOSURE_AND_DEPLOYED_BYTES_WITH_EXPLICIT_COST_GAPS",
        "source_exposure": {
            "source_model": phase1["source"]["model"],
            "checkpoint_revision": phase1["source"]["revision"],
            "source_manifest_sha256": phase1["source"]["source_manifest_sha256"],
            "source_layers_queried": "no hidden layer directly retained; full causal LM forward path used for generation",
            "source_forward_requests": source["teacher_outputs_generated"],
            "source_tokens_processed": {
                "input": source["teacher_input_tokens_all_attempts"],
                "output": source["authoritative_teacher_tokens_all_attempts"],
            },
            "raw_utf8_bytes_selected": {
                "prompts": selected["selected_raw_prompt_bytes"],
                "teacher_outputs": selected["selected_raw_teacher_output_bytes"],
            },
            "unique_documents_or_prompts_selected": selected["selected_records"],
        },
        "imported_information_full_selected_ir": {
            "teacher_output_bytes": selected["selected_raw_teacher_output_bytes"],
            "teacher_tokens": selected["selected_authoritative_teacher_tokens"],
            "retained_logits": selected["stored_logits"],
            "retained_hidden_scalars": selected["stored_activations"],
            "copied_frozen_source_parameters": selected["copied_source_parameters"],
            "normalized_artifact_bytes": selected["archive_bytes"],
        },
        "imported_information_b40_production_budget": {
            **b40,
            "artifact_archive_bytes": product["archive_bytes"],
            "artifact_parameters": product["total_parameters"],
            "component_parameters": product["component_parameters"],
            "routing_bytes": "included in english-substrate.abi",
            "bridge_bytes": "included in english-substrate.abi; not reported as a free channel",
        },
        "cost": {
            "one_time_teacher_generation_seconds": source["source_inference_seconds"],
            "one_time_source_load_seconds": source["source_load_seconds"],
            "one_time_extraction_wall_seconds": source["wall_seconds"],
            "normalization_seconds": selected["normalization_seconds"],
            "peak_cpu_ram_bytes": source["peak_process_rss_bytes"],
            "peak_vram_bytes": source["peak_cuda_allocated_bytes"],
            "external_hardware_used": False,
            "hardware": "registered development laptop RTX 3080 Laptop GPU",
            "energy": "NOT_MEASURED",
            "production_lineage_training_cpu_hours": "NOT_RECONSTRUCTED_IN_COMPACT_FINAL_MILE_LEDGER",
            "production_lineage_training_gpu_hours": "NOT_RECONSTRUCTED_IN_COMPACT_FINAL_MILE_LEDGER",
            "cost_gap_is_blocking_for_full_moonshot": True,
            "bound_full_history": "research-history-v1089 plus Phase 3/4 immutable receipts",
        },
        "deployment": {
            "installed_english_bytes": product["archive_bytes"],
            "active_english_tensor_bytes": 254495764,
            "active_english_parameters": product["total_parameters"],
            "phase7_process_rss_delta_bytes": 545308672,
            "phase7_cuda_allocation_bytes": 505427456,
            "operations_per_generated_unit": "NOT_INSTRUMENTED",
            "deployment_accounting_gap_is_blocking_for_full_moonshot": True,
        },
        "claim_boundary": (
            "No low-data, global-minimum, or exhaustive-cost claim is allowed while the two "
            "explicit production-lineage cost fields and operation count remain unresolved."
        ),
    }


def _baseline_comparison(root: Path) -> dict[str, Any]:
    verified = _object(
        root / "results/abi_capability_compiler_phase4_b40_frontier/verify_v1014/result.json"
    )
    rows = {}
    for system, evidence in verified["baseline_quality"].items():
        rows[system] = {
            "runs": len(evidence["runs"]),
            "all_seeds_pass_locked_quality": evidence["all_seeds_pass_locked_absolute_quality"],
            "functional_passes_v1": [row["functional_passes_v1"] for row in evidence["runs"]],
            "repetition_collapses_v2": [row["repetition_collapses_v2"] for row in evidence["runs"]],
            "training_seconds": sum(row["training_seconds"] for row in evidence["runs"]),
            "active_parameters": evidence["runs"][0]["active_parameters"],
            "installed_parameters": evidence["runs"][0]["complete_installed_parameters"],
            "source_base_present_at_inference": evidence["runs"][0]["source_base_present_at_inference"],
            "receiver_specific_retraining_required": True,
            "cross_host_reuse_tested": False,
        }
    return {
        "format": "abi-final-mile-matched-baseline-comparison/1",
        "status": "BOUNDED_SAME_HOST_BASELINES_COMPLETE_CROSS_HOST_PARETO_NOT_ESTABLISHED",
        "equal_information_budget": verified["fairness_views"]["equal_imported_information"],
        "systems": {
            "sequence_distillation_D0": rows["D0"],
            "LoRA_L0": rows["L0"],
            "routed_LoRA_L1": rows["L1"],
            "ABI_B40": {
                "all_seeds_pass_locked_quality": True,
                "teacher_absent_at_inference": True,
                "receiver_specific_retraining_required_on_layercake": False,
                "cross_host_reuse_tested": False,
                "cross_host_result": "HOST_INDEPENDENCE_FAILED",
                "active_tensor_bytes": 254495764,
            },
            "conventional_hidden_state_distillation": {
                "status": "NOT_REGISTERED_AT_EXACT_B40_FINAL_MILE_BUDGET",
                "blocking_for_full_baseline_claim": True,
            },
        },
        "same_host_runtime": verified["runtime_recomputed"],
        "pareto_conclusion": (
            "ABI has a strong bounded same-host quality/deployment advantage in this family. "
            "It has not established the required learned-once cross-host Pareto advantage."
        ),
    }


def _frontier(root: Path) -> dict[str, Any]:
    phase4 = _object(root / "evidence/current/ABI_CAPABILITY_COMPILER_PHASE4_CERTIFICATE_V1.json")
    host = _object(root / "results/abi_final_mile/host_portability_v1/repair_rescreen.json")
    return {
        "format": "abi-final-mile-stable-footprint-frontier/1",
        "status": "NO_FINAL_MILE_STABLE_FOOTPRINT",
        "registered_family": ["B20", "B40"],
        "same_host_result": {
            "B20": "FAIL_ALL_THREE_SEEDS",
            "B40": "PASS_ALL_THREE_SEEDS",
            "smallest_same_host_tested_pass": "B40",
            "global_minimum_claimed": False,
            "certificate_sha256": sha256_file(
                root / "evidence/current/ABI_CAPABILITY_COMPILER_PHASE4_CERTIFICATE_V1.json"
            ),
            "certificate_status": phase4["status"],
        },
        "all_receiver_requirement": {
            "receivers_passing": host["receivers_passing"],
            "receivers_required": host["receivers_required"],
            "B40": "FAIL_HOST_PORTABILITY",
            "minimum_stable_budget": None,
        },
        "post_acquisition_compression_run": False,
        "reason_compression_not_run": (
            "Compression is downstream of a stable all-receiver capability; no candidate "
            "satisfies the host-portability prerequisite."
        ),
    }


def _release_report() -> str:
    return """# ABI final-mile release report

This is a sealed **failed-candidate release family**, not a production or
moonshot certificate. It preserves the exact Phase 7 English and specialist
archives under explicit `.abi` filenames and adds an outer content-addressed
signature without changing their bytes.

Tier A remains a bounded same-machine pass. Tier B fails because the exact
English artifact is natively executable only by the LayerCake v25 receiver;
Qwen2 and GPT-NeoX receiver families have no native consumer for its tensor
contract. A wrapper that bypasses those receivers was preregistered as
non-qualifying. The one bounded repair is consumed.

Tier C is still external: no independent operator or different CPU/CUDA host
has returned evidence. The human packet remains 0/21,000. The registered
same-host B20/B40 frontier and matched LoRA/sequence-distillation results are
preserved, but there is no all-receiver stable information minimum and the
exact-B40 conventional hidden-state baseline is absent.

No file in this directory authorizes a universal, production, Phase 8, or
full-moonshot claim.
"""


def _key(path: Path) -> Ed25519PrivateKey:
    if path.exists():
        value = serialization.load_pem_private_key(path.read_bytes(), password=None)
        if not isinstance(value, Ed25519PrivateKey):
            raise FinalMileError("release custody key is not Ed25519")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    value = Ed25519PrivateKey.generate()
    path.write_bytes(
        value.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return value


def build_release(root: Path, *, output: Path, custody_key: Path) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FinalMileError(f"immutable release directory already exists: {output}")
    output.mkdir(parents=True)
    package_bindings = {}
    for name, relative in PACKAGE_SOURCES.items():
        source = root / relative
        if sha256_file(source) != EXPECTED_PACKAGE_HASHES[name]:
            raise FinalMileError(f"frozen package changed: {relative}")
        target = output / name
        _place_immutable(source, target)
        package_bindings[name] = {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "inner_format": "signed LayerCake cake archive",
            "outer_extension": ".abi",
        }

    canonical = _object(
        root.parent
        / "layercake_release/moonshot/canonical_route_isolated_clarification_core_abi_v25.json"
    )
    canonical = {
        "format": "abi-canonical-host-contract/1",
        "status": "LAYERCAKE_V25_ONLY_NOT_CROSS_HOST_CERTIFIED",
        "canonical_layercake_contract": canonical,
        "artifact_input": "strict UTF-8 bytes",
        "artifact_output": "strict UTF-8 bytes",
        "deterministic_tolerance": "bit-identical on the registered deterministic contract",
        "provider_bypass_counts_as_cross_architecture_portability": False,
    }
    _json(output / "canonical_host_abi.json", canonical)
    _json(
        output / "receiver-certification-spec.json",
        _object(root / "contracts/ABI_FINAL_MILE_HOST_PORTABILITY_V1.json"),
    )
    _json(output / "imported-information-ledger.json", _information_ledger(root))
    _json(output / "source-success-lock.json", _source_success_lock(root))
    _json(output / "compatibility-matrix.json", _object(root / "results/abi_final_mile/host_portability_v1/repair_rescreen.json"))
    _json(output / "baseline-comparisons/comparison.json", _baseline_comparison(root))
    _json(output / "minimum-stable-frontier.json", _frontier(root))
    _json(
        output / "human-evidence/status.json",
        {
            "format": "abi-human-evidence-status/1",
            "status": "BLOCKED_EXTERNAL_HUMAN_RATERS",
            "completed_preferences": 0,
            "required_preferences": 21000,
            "independent_raters_completed": 0,
            "command_per_rater": [
                "abi human-rate --rater R1",
                "abi human-rate --rater R2",
                "abi human-rate --rater R3",
            ],
            "codex_self_rated": False,
        },
    )
    _json(
        output / "external-reproduction/status.json",
        {
            "format": "abi-external-reproduction-status/1",
            "status": "BLOCKED_EXTERNAL_HARDWARE",
            "independent_operator_complete": False,
            "different_cpu_complete": False,
            "different_cuda_gpu_complete": False,
            "same_machine_substituted": False,
        },
    )
    certificate = {
        "format": "abi-final-mile-release-certificate/1",
        "status": "HOST_INDEPENDENCE_FAILED",
        "tier_a": "PASS_BOUNDED_SAME_MACHINE",
        "tier_b": "FAIL_1_OF_3_NATIVE_RECEIVER_FAMILIES",
        "tier_c": "BLOCKED_EXTERNAL_HARDWARE_AND_OPERATOR",
        "tier_d": "NOT_PROVEN",
        "human_gate": "BLOCKED_0_OF_21000",
        "minimum_stable_frontier": "NOT_ESTABLISHED_ACROSS_RECEIVERS",
        "matched_baselines": "SAME_HOST_BOUNDED_COMPLETE_FINAL_MILE_CROSS_HOST_INCOMPLETE",
        "teacher_absent_at_inference": True,
        "artifact_bytes_unchanged": True,
        "release_certified": False,
        "phase8_certified": False,
    }
    _json(output / "release-certificate.json", certificate)
    report = output / "release-report.md"
    report.write_text(_release_report(), encoding="utf-8", newline="\n")

    inventory = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"release-manifest.json", "release-signature.json"}:
            relative = path.relative_to(output).as_posix()
            inventory[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "format": "abi-final-mile-release-manifest/1",
        "status": "SEALED_FAILED_CANDIDATE_HOST_INDEPENDENCE_NOT_PRODUCTION",
        "created_utc": _utc_now(),
        "files": inventory,
        "file_count": len(inventory),
        "total_bytes": sum(row["bytes"] for row in inventory.values()),
        "package_bindings": package_bindings,
        "release_certified": False,
    }
    _json(output / "release-manifest.json", manifest)
    private = _key(custody_key.resolve())
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    manifest_bytes = (output / "release-manifest.json").read_bytes()
    signature = {
        "format": "abi-final-mile-release-signature/1",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "signature_ed25519_hex": private.sign(manifest_bytes).hex(),
        "public_key_pem": public.decode("ascii"),
        "inner_signature_boundary": (
            "The historical packages contain research signatures. This independent outer "
            "signature authenticates the exact final-mile inventory without changing package bytes."
        ),
    }
    _json(output / "release-signature.json", signature)
    return {"certificate": certificate, "manifest": manifest, "signature": signature}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--custody-key", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = build_release(
        root,
        output=(root / args.output).resolve(),
        custody_key=(root / args.custody_key).resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
