"""Coherence-only v19 rescreen of the frozen B40/B80 Phase 4 lineages."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from safetensors.torch import load_file
import torch

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_final_controls import evaluate_functional_v2
from .capability_compiler_phase3_guarded_screen import artifact_markers
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import (
    paired_stratified_bootstrap,
    wilson,
)
from .capability_compiler_phase3_weak_residual import WEAK_CAPABILITIES
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-v19-frontier-rescreen/1"
EXPECTED_COMPONENT_PARAMETERS = {
    "model": 61_655_050,
    "router": 1_058_040,
    "residual": 99_840,
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(value, dict) for value in values):
        raise Phase3Error(f"expected JSONL objects: {path}")
    return values


def _layercake_api(layercake_root: Path):
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.cake.manifest import CakeManifest
    from layercake.cake.package import build_package, load_package, tensor_specs
    from layercake.cake.signing import key_id
    from layercake_extensions.route_isolated_shallow_sparse_core import (
        CAPABILITY_TO_TASK_ROUTE,
    )
    from layercake_extensions.route_isolated_prompt_span_core_v19 import (
        ARCHITECTURE_V19_FORMAT,
        PROMPT_SPAN_FEATURE,
        PromptSpanRouteIsolatedShallowSparseCoreHost,
        ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256,
        ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION,
    )

    return {
        "CakeManifest": CakeManifest,
        "build_package": build_package,
        "load_package": load_package,
        "tensor_specs": tensor_specs,
        "key_id": key_id,
        "task_routes": CAPABILITY_TO_TASK_ROUTE,
        "architecture_format": ARCHITECTURE_V19_FORMAT,
        "prompt_span_feature": PROMPT_SPAN_FEATURE,
        "Host": PromptSpanRouteIsolatedShallowSparseCoreHost,
        "abi_sha256": ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256,
        "abi_version": ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION,
    }


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_SIX_SYSTEM_COHERENCE_ONLY_V19_RESCREEN"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("v19 frontier rescreen governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"v19 frontier rescreen binding changed: {relative}")
    return protocol, sha256_file(path)


def _states(root: Path, spec: Mapping[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: load_file(str((root / relative).resolve()), device="cpu")
        for name, relative in spec["components"].items()
    }


def _component_inventory(states: Mapping[str, Mapping[str, torch.Tensor]]) -> tuple[dict[str, int], dict[str, torch.Tensor]]:
    counts = {name: sum(tensor.numel() for tensor in state.values()) for name, state in states.items()}
    if counts != EXPECTED_COMPONENT_PARAMETERS:
        raise Phase3Error(f"candidate component inventory changed: {counts}")
    tensors = {
        f"{namespace}.{name}": tensor
        for namespace, state in states.items()
        for name, tensor in state.items()
    }
    namespaces = Counter(name.split(".", 1)[0] for name in tensors)
    if namespaces != Counter({"model": 82, "router": 3, "residual": 4}):
        raise Phase3Error(f"candidate tensor namespace inventory changed: {namespaces}")
    return counts, tensors


def _architecture(root: Path, protocol: Mapping[str, Any], spec: Mapping[str, Any], api: Mapping[str, Any]) -> dict[str, Any]:
    parent = _json(root / protocol["model_metadata"])
    model_tokenizer = _json(root / protocol["model_tokenizer"])
    model_tokenizer_raw = json.dumps(model_tokenizer, sort_keys=True, separators=(",", ":")).encode()
    router_tokenizer = _json(root / protocol["router_tokenizer"])
    router_config = _json(root / spec["router_config"])
    markers = artifact_markers(root / spec["guard_artifact"])
    return {
        "format": api["architecture_format"],
        "model": parent["architecture"],
        "model_tokenizer": {
            "format": "declarative-tokenizers-json/1",
            "tokenizers_json": model_tokenizer,
            "sha256": hashlib.sha256(model_tokenizer_raw).hexdigest(),
            "eos_token_id": 50256,
        },
        "router": {
            "vocabulary": int(router_config["vocabulary"]),
            "character_hash_buckets": int(router_config["character_hash_buckets"]),
            "character_ngram_minimum": int(router_config["character_ngram_minimum"]),
            "character_ngram_maximum": int(router_config["character_ngram_maximum"]),
            "hash_seed": int(router_config["hash_seed"]),
            "classes": len(CAPABILITIES) + 1,
        },
        "router_tokenizer": router_tokenizer,
        "residual": {"width": 768, "rank": 16, "routes": 4, "reuse": "before_each_transformer_block"},
        "capabilities": list(CAPABILITIES),
        "capability_to_task_route": api["task_routes"],
        "weak_capabilities": list(WEAK_CAPABILITIES),
        "guard": {
            "predicate": "contiguous_1_to_16_token_span_repeated_4_times_or_fourgram_diversity_below_0.35_at_32_tokens",
            "scope": "weak_capabilities_only",
            "stop_before_collapsing_token": True,
            "abstention_markers": list(markers),
            "abstention_clause": "I cannot determine that from the information given.",
        },
    }


def _build_package(
    root: Path,
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    output: Path,
    api: Mapping[str, Any],
    private: Ed25519PrivateKey,
    public_pem: bytes,
) -> tuple[dict[str, Any], bytes]:
    states = _states(root, spec)
    counts, tensors = _component_inventory(states)
    signer = api["key_id"](public_pem)
    manifest = api["CakeManifest"](
        schema_version="1",
        cake_id=f"abi-phase4-{spec['budget'].lower()}-seed{spec['seed']}-english-core",
        name=f"ABI Phase 4 {spec['budget']} seed {spec['seed']} English core",
        description="Exact frozen Phase 4 lineage packaged for governed v19 coherence-only rescreen",
        version="0.19.0-frontier-rescreen",
        publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer},
        abi_version=api["abi_version"],
        abi_hash=api["abi_sha256"],
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"},
        architecture=_architecture(root, protocol, spec, api),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": [
            "byte_input", "safe_tensors", "persistent_incremental_state",
            "physical_route_isolation", "declarative_runtime_guard",
            "strict_utf8_boundary", api["prompt_span_feature"],
        ]},
        tensor_payload_hash="",
        tensor_shapes=api["tensor_specs"](tensors),
        package_hash="",
        training_data_provenance={
            "phase4_budget": spec["budget"],
            "phase4_seed": int(spec["seed"]),
            "lineage_result_sha256": protocol["bindings"][spec["lineage_result"]],
            "component_sha256": {
                name: protocol["bindings"][relative]
                for name, relative in spec["components"].items()
            },
            "teacher_at_inference": False,
            "source_transformer_blocks": 0,
            "receiver_training_steps": 0,
        },
        evaluation_evidence={
            "authorization": protocol["authorization"],
            "authorization_sha256": protocol["bindings"][protocol["authorization"]],
            "status": "COHERENCE_ONLY_V19_RESCREEN",
        },
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": signer},
        domains=("english-core",),
        permissions=("local-inference",),
    )
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    api["build_package"](output, manifest, tensors, private_key=private_pem)
    loaded = api["load_package"](output, trust_store={signer: public_pem}, require_signature=True)
    gates = {
        "signature_valid": loaded.signed,
        "tensor_values_exact": set(loaded.tensors) == set(tensors) and all(
            torch.equal(loaded.tensors[name], tensors[name]) for name in tensors
        ),
        "interface_v19": loaded.manifest.abi_version == api["abi_version"] and loaded.manifest.abi_hash == api["abi_sha256"],
        "component_counts_exact": counts == EXPECTED_COMPONENT_PARAMETERS,
        "receiver_learning_zero": True,
        "teacher_absent": True,
    }
    if not all(gates.values()):
        raise Phase3Error(f"ephemeral package verification failed: {gates}")
    receipt = {
        "archive_sha256": loaded.archive_hash,
        "tensor_payload_hash": loaded.manifest.tensor_payload_hash,
        "package_hash": loaded.manifest.package_hash,
        "archive_bytes": output.stat().st_size,
        "component_parameters": counts,
        "total_parameters": sum(counts.values()),
        "tensor_count": len(tensors),
        "gates": gates,
    }
    return receipt, public_pem


def _merged_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        passed = sum(bool(row["functional_pass_v1"]) for row in values)
        per[capability] = {
            "passes_v1": passed,
            "observations": len(values),
            "wilson_v1": wilson(passed, len(values)),
            "collapses_v2": sum(bool(row["repetition_collapse_v2"]) for row in values),
        }
    return {
        "functional_passes_v1": sum(bool(row["functional_pass_v1"]) for row in rows),
        "functional_passes_v2": sum(bool(row["functional_pass_v2"]) for row in rows),
        "observations": len(rows),
        "per_capability": per,
        "repetition_collapses_v2": sum(bool(row["repetition_collapse_v2"]) for row in rows),
        "router_correct": sum(bool(row["capability_route_correct"]) for row in rows),
        "strong_routes_exact": sum(row["strong_parent_output_exact"] is True for row in rows),
        "guard_terminations": sum(bool(row["guard_terminated"]) for row in rows),
        "abstention_prefixes": sum(bool(row["abstention_clause_prefixed"]) for row in rows),
    }


def _quality_gates(
    protocol: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    merged_rows: list[dict[str, Any]],
    probes: Mapping[str, Mapping[str, Any]],
    teacher: Mapping[str, Mapping[str, Any]],
    bootstrap_seed: int,
) -> tuple[dict[str, bool], dict[str, Any]]:
    paired = [{
        "capability": row["capability"],
        "candidate_pass": bool(row["functional_pass_v1"]),
        "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probes[row["probe_id"]]["evaluator"]),
    } for row in merged_rows]
    relative = paired_stratified_bootstrap(
        paired,
        replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]),
        seed=bootstrap_seed,
    )
    threshold = protocol["absolute_screen"]
    per = evaluation["per_capability"]
    gates = {
        "per_capability_functional": all(
            value["wilson_v1"]["point"] >= float(threshold["per_capability_functional_point_estimate_minimum"])
            and value["wilson_v1"]["lower_95"] >= float(threshold["per_capability_functional_wilson_lower_minimum"])
            for value in per.values()
        ),
        "critical_capabilities": all(
            per[name]["wilson_v1"]["point"] >= float(threshold["critical_point_minimum"])
            and per[name]["wilson_v1"]["lower_95"] >= float(threshold["critical_wilson_lower_minimum"])
            for name in ("prompt_grounding", "instruction_following", "abstention")
        ),
        "zero_repetition_collapse": int(evaluation["repetition_collapses_v2"]) == 0,
        "router_exact": int(evaluation["router_correct"]) == 1400,
        "strong_parent_exact": int(evaluation["strong_routes_exact"]) == 1000,
        "teacher_noninferior": float(relative["lower_95"]) >= float(protocol["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"]),
        "teacher_absent": True,
        "training_absent": True,
        "final_test_not_accessed": True,
    }
    return gates, relative


def frontier_decision(systems: list[Mapping[str, Any]]) -> dict[str, bool]:
    b40 = [row for row in systems if row["budget"] == "B40"]
    b80 = [row for row in systems if row["budget"] == "B80"]
    return {
        "three_b40_seeds": len(b40) == 3,
        "three_b80_seeds": len(b80) == 3,
        "b80_all_seed_machine_gates": len(b80) == 3 and all(row["machine_gates_pass"] for row in b80),
        "b40_adjacent_lower_fails_all_seeds": len(b40) == 3 and all(not row["machine_gates_pass"] for row in b40),
    }


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if not torch.cuda.is_available():
        raise Phase3Error("preregistered CUDA device unavailable")
    probes = development_probes(root / protocol["development_catalog"])
    systems = []
    for spec in protocol["systems"]:
        historical = _rows(root / spec["historical_outputs"])
        coherence = [row for row in historical if row["capability"] == "coherence"]
        states = _states(root, spec)
        counts, tensors = _component_inventory(states)
        systems.append({
            "budget": spec["budget"],
            "seed": spec["seed"],
            "historical_rows": len(historical),
            "coherence_rows": len(coherence),
            "component_parameters": counts,
            "tensor_count": len(tensors),
        })
    gates = {
        "six_registered_systems": len(systems) == 6,
        "three_paired_seeds": sorted({row["seed"] for row in systems}) == [104729, 130363, 155921],
        "paired_budgets": Counter(row["budget"] for row in systems) == Counter({"B40": 3, "B80": 3}),
        "locked_development_depth": len(probes) == 1400 and all(row["historical_rows"] == 1400 for row in systems),
        "exact_coherence_depth": all(row["coherence_rows"] == 100 for row in systems),
        "component_inventory_exact": all(row["component_parameters"] == EXPECTED_COMPONENT_PARAMETERS and row["tensor_count"] == 89 for row in systems),
        "cuda_available": True,
        "training_prohibited": True,
        "teacher_model_loading_prohibited": True,
        "final_test_not_accessed": True,
    }
    return {
        "status": "PASS_V19_FRONTIER_RESCREEN_PREFLIGHT" if all(gates.values()) else "FAIL_V19_FRONTIER_RESCREEN_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "systems": systems,
        "authorized_model_inference_rows": 600,
        "gates": gates,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
    }


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable v19 frontier rescreen exists: {output}")
    check = preflight(root, protocol_path)
    if not check["status"].startswith("PASS"):
        raise Phase3Error("v19 frontier rescreen preflight failed")
    output.mkdir(parents=True)
    api = _layercake_api((root / protocol["layercake_root"]).resolve())
    if api["abi_version"] != protocol["interface"] or api["abi_sha256"] != protocol["interface_sha256"]:
        raise Phase3Error("LayerCake v19 interface changed")
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(protocol["research_signing_seed_hex"]))
    public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    signer = api["key_id"](public_pem)
    all_probes = development_probes(root / protocol["development_catalog"])
    probes = {str(row["probe_id"]): row for row in all_probes}
    teacher = {str(row["probe_id"]): row for row in _rows(root / protocol["teacher_reference"])}
    systems = []
    raw_evidence = []
    for spec in protocol["systems"]:
        historical = _rows(root / spec["historical_outputs"])
        historical_by_id = {str(row["probe_id"]): row for row in historical}
        coherence_probes = [row for row in all_probes if row["canonical_capability"] == "coherence"]
        if [row["probe_id"] for row in coherence_probes] != [row["probe_id"] for row in historical if row["capability"] == "coherence"]:
            raise Phase3Error("coherence prompt order changed")
        with tempfile.TemporaryDirectory(prefix=f"abi-v19-{spec['budget'].lower()}-{spec['seed']}-") as raw:
            temp = Path(raw)
            package_path = temp / "candidate.cake"
            package, _ = _build_package(root, protocol, spec, package_path, api, private, public_pem)
            host = api["Host"](temp / "registry", trust_store={signer: public_pem}, device="cuda")
            activation = host.activate(package_path)
            coherence_rows = []
            for probe in coherence_probes:
                output_text = host.generate(
                    str(probe["prompt"]),
                    maximum_tokens=int(probe["max_new_tokens"]),
                ).decode("utf-8", errors="strict")
                pointer = dict(host.last_pointer_execution or {})
                pointer.pop("wall_seconds", None)
                old = historical_by_id[str(probe["probe_id"])]
                row = {
                    **old,
                    "output": output_text,
                    "original_output": output_text,
                    "output_token_ids": [int(value) for value in host.model_tokenizer.encode(output_text)],
                    "automatic_capability_route": "coherence" if pointer else old["automatic_capability_route"],
                    "capability_route_correct": bool(pointer),
                    "guard_terminated": False,
                    "abstention_clause_prefixed": False,
                    "functional_pass_v1": evaluate_functional(output_text, probe["evaluator"]),
                    "functional_pass_v2": evaluate_functional_v2(output_text, probe["evaluator"], "coherence"),
                    "repetition_collapse_v2": repetition_collapse_v2(output_text),
                    "v19_pointer": pointer,
                }
                coherence_rows.append(row)
            verification = host.verify()
            activation_receipt = {
                "archive_sha256": activation["archive_hash"],
                "tensor_payload_hash": activation["payload_hash"],
                "receiver_training_steps": activation["receiver_training_steps"],
                "receiver_calibration_runs": activation["receiver_calibration_runs"],
                "verification": verification["status"],
            }
            del host
            gc.collect()
            torch.cuda.empty_cache()
        replacements = {row["probe_id"]: row for row in coherence_rows}
        merged = [replacements.get(row["probe_id"], row) for row in historical]
        if sum(left is right for left, right in zip(merged, historical)) != 1300:
            raise Phase3Error("v19 rescreen changed rows outside coherence")
        system_name = f"{spec['budget']}-seed{spec['seed']}"
        system_dir = output / system_name
        system_dir.mkdir()
        coherence_path = system_dir / "coherence_outputs.jsonl"
        merged_path = system_dir / "merged_development_outputs.jsonl"
        _write_immutable(coherence_path, b"".join(canonical_json_bytes(row) for row in coherence_rows))
        _write_immutable(merged_path, b"".join(canonical_json_bytes(row) for row in merged))
        evaluation = _merged_evaluation(merged)
        gates, relative = _quality_gates(
            protocol,
            evaluation,
            merged,
            probes,
            teacher,
            int(spec["seed"]) + 4_000_000,
        )
        pointer_gates = {
            "all_100_pointer_rows": all(bool(row["v19_pointer"]) for row in coherence_rows),
            "six_candidates": all(row["v19_pointer"].get("candidate_count") == 6 for row in coherence_rows),
            "one_scoring_forward": all(row["v19_pointer"].get("candidate_scoring_forward_passes") == 1 for row in coherence_rows),
            "one_active_route": all(row["v19_pointer"].get("active_residual_routes") == 1 for row in coherence_rows),
            "persistent_state_reused": all(row["v19_pointer"].get("persistent_prompt_state_reused") is True for row in coherence_rows),
            "evaluator_blind": all(row["v19_pointer"].get("evaluator_used") is False for row in coherence_rows),
            "package_identity": package["archive_sha256"] == activation_receipt["archive_sha256"] and package["tensor_payload_hash"] == activation_receipt["tensor_payload_hash"],
            "package_verified": activation_receipt["verification"] == "PASS",
            "receiver_learning_zero": activation_receipt["receiver_training_steps"] == activation_receipt["receiver_calibration_runs"] == 0,
        }
        machine_pass = all(gates.values()) and all(pointer_gates.values())
        receipt = {
            "budget": spec["budget"],
            "seed": int(spec["seed"]),
            "lineage_result_sha256": protocol["bindings"][spec["lineage_result"]],
            "historical_outputs_sha256": protocol["bindings"][spec["historical_outputs"]],
            "historical_functional_passes_v1": sum(bool(row["functional_pass_v1"]) for row in historical),
            "v19_functional_passes_v1": evaluation["functional_passes_v1"],
            "historical_coherence_passes_v1": sum(bool(row["functional_pass_v1"]) for row in historical if row["capability"] == "coherence"),
            "v19_coherence_passes_v1": sum(bool(row["functional_pass_v1"]) for row in coherence_rows),
            "noncoherence_rows_reused": 1300,
            "noncoherence_outputs_sha256": hashlib.sha256(b"".join(canonical_json_bytes(row) for row in historical if row["capability"] != "coherence")).hexdigest(),
            "coherence_outputs": {"path": str(coherence_path.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(coherence_path)},
            "merged_outputs": {"path": str(merged_path.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(merged_path)},
            "evaluation": evaluation,
            "teacher_comparison_v1": relative,
            "quality_gates": gates,
            "pointer_gates": pointer_gates,
            "machine_gates_pass": machine_pass,
            "package": package,
            "activation": activation_receipt,
        }
        systems.append(receipt)
        raw_evidence.extend({"system": system_name, **row} for row in coherence_rows)
        print(json.dumps({"system": system_name, "coherence_passes": receipt["v19_coherence_passes_v1"], "functional_passes": receipt["v19_functional_passes_v1"], "machine_pass": machine_pass}), flush=True)
    frontier = frontier_decision(systems)
    stable_sufficient = frontier["b80_all_seed_machine_gates"]
    stable_minimum = stable_sufficient and frontier["b40_adjacent_lower_fails_all_seeds"]
    if stable_minimum:
        status = "PASS_STABLE_B80_SUFFICIENT_FRONTIER_WITH_ADJACENT_B40_FAILURE"
    elif stable_sufficient:
        status = "COMPLETE_STABLE_B80_SUFFICIENT_UPPER_BOUND_MINIMUM_UNPROVEN"
    else:
        status = "COMPLETE_V19_B40_B80_RESCREEN_NO_STABLE_SUFFICIENT_BUDGET"
    result = {
        "format": "abi-capability-compiler-phase4-v19-frontier-rescreen-result/1",
        "status": status,
        "protocol_sha256": protocol_sha,
        "systems": systems,
        "frontier_gates": frontier,
        "stable_sufficient_b80": stable_sufficient,
        "stable_minimum_b80": stable_minimum,
        "model_inference_rows": 600,
        "noncoherence_rows_reused": 7800,
        "ephemeral_signed_packages": 6,
        "training_performed": False,
        "teacher_model_loaded": False,
        "teacher_present_at_inference": False,
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Development-only ABI-arm B40/B80 v19 coherence rescreen. A stable sufficient budget is not a minimum unless adjacent B40 failure reproduces across all seeds. Matched LoRA/distillation, final test, unconditional predecessor, Phase 4, and ABI superiority remain unproven.",
    }
    raw_path = output / "coherence_evidence.jsonl"
    _write_immutable(raw_path, b"".join(canonical_json_bytes(row) for row in raw_evidence))
    result["coherence_evidence_sha256"] = sha256_file(raw_path)
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    command = sub.add_parser("run")
    command.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = preflight(root, root / args.protocol) if args.command == "preflight" else run(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith(("PASS", "COMPLETE")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
