"""Signed LayerCake v25 product conformance for the three qualified B40 candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from safetensors.torch import load_file
import torch

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .capability_compiler_phase4_b20_v25_physical_screen import (
    _api,
    _architecture,
    _generate,
    _json,
    _route_for_capability,
)
from .capability_compiler_phase4_v19_frontier_rescreen import (
    _merged_evaluation,
    _quality_gates,
    _rows,
)


FORMAT = "abi-capability-compiler-phase4-b40-v25-product-conformance/1"
SEEDS = (104729, 130363, 155921)


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_THREE_SEED_B40_SIGNED_V25_PRODUCT_CONFORMANCE"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or tuple(int(spec["seed"]) for spec in protocol.get("systems", ())) != SEEDS
        or any(spec.get("budget") != "B40" for spec in protocol["systems"])
    ):
        raise Phase3Error("B40 v25 conformance governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B40 v25 conformance binding changed: {relative}")
    return protocol, sha256_file(path)


def _runtime_protocol(protocol: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(protocol)
    value["router_config"] = spec["router_config"]
    value["guard_artifact"] = spec["guard_artifact"]
    return value


def _package(
    root: Path,
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    path: Path,
    api: Mapping[str, Any],
    private: Ed25519PrivateKey,
    public_pem: bytes,
) -> dict[str, Any]:
    states = {
        "model": load_file(str(root / spec["components"]["model"]), device="cpu"),
        "router": load_file(str(root / spec["components"]["router"]), device="cpu"),
        "residual": load_file(str(root / spec["candidate_checkpoint"]), device="cpu"),
    }
    counts = {name: sum(value.numel() for value in state.values()) for name, state in states.items()}
    expected = {"model": 61655050, "router": 1058040, "residual": 124416}
    if counts != expected:
        raise Phase3Error(f"B40 v25 component inventory changed: {counts}")
    tensors = {
        f"{namespace}.{name}": value
        for namespace, state in states.items()
        for name, value in state.items()
    }
    if Counter(name.split(".", 1)[0] for name in tensors) != Counter({"model": 82, "router": 3, "residual": 4}):
        raise Phase3Error("B40 v25 tensor namespace inventory changed")
    runtime = _runtime_protocol(protocol, spec)
    signer = api["key_id"](public_pem)
    features = [
        "byte_input",
        "safe_tensors",
        "persistent_incremental_state",
        "physical_route_isolation",
        "declarative_runtime_guard",
        "strict_utf8_boundary",
        api["PROMPT_SPAN_FEATURE"],
        api["UNIVERSAL_GUARD_FEATURE"],
        api["EXACT_LEXICAL_GUARD_FEATURE"],
        api["FORMAT_LITERAL_FEATURE"],
        api["SINGLE_PARSE_ACTIVATION_FEATURE"],
        api["ALLOCATION_BOUNDED_ADOPTION_FEATURE"],
        api["CLARIFICATION_ROUTE_ISOLATION_FEATURE"],
    ]
    manifest = api["CakeManifest"](
        schema_version="1",
        cake_id=f"abi-phase4-b40-seed{spec['seed']}-v25-english-core",
        name=f"ABI Phase 4 B40 seed {spec['seed']} v25 English core",
        description="Frozen qualified B40 lineage with clarification-only fifth route",
        version="0.25.0-b40-conformance",
        publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer},
        abi_version=api["ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_VERSION"],
        abi_hash=api["ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_SHA256"],
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"},
        architecture=_architecture(root, runtime, api),
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": features},
        tensor_payload_hash="",
        tensor_shapes=api["tensor_specs"](tensors),
        package_hash="",
        training_data_provenance={
            "phase4_budget": "B40",
            "phase4_seed": int(spec["seed"]),
            "lineage_result_sha256": protocol["bindings"][spec["lineage_result"]],
            "clarification_checkpoint_sha256": protocol["bindings"][spec["candidate_checkpoint"]],
            "teacher_at_inference": False,
            "source_transformer_blocks": 0,
            "receiver_training_steps": 0,
        },
        evaluation_evidence={"authorization": protocol["authorization"], "status": "B40_V25_PRODUCT_CONFORMANCE"},
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
    api["build_package"](path, manifest, tensors, private_key=private_pem)
    loaded = api["load_package"](path, trust_store={signer: public_pem}, require_signature=True)
    gates = {
        "signature_valid": loaded.signed,
        "tensor_values_exact": set(loaded.tensors) == set(tensors)
        and all(torch.equal(loaded.tensors[name], tensors[name]) for name in tensors),
        "interface_v25": loaded.manifest.abi_version == api["ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_VERSION"]
        and loaded.manifest.abi_hash == api["ROUTE_ISOLATED_CLARIFICATION_CORE_V25_ABI_SHA256"],
        "component_counts_exact": counts == expected,
        "receiver_learning_zero": True,
        "teacher_absent": True,
    }
    if not all(gates.values()):
        raise Phase3Error(f"B40 v25 package verification failed: {gates}")
    return {
        "archive_sha256": loaded.archive_hash,
        "tensor_payload_hash": loaded.manifest.tensor_payload_hash,
        "package_hash": loaded.manifest.package_hash,
        "archive_bytes": path.stat().st_size,
        "component_parameters": counts,
        "total_parameters": sum(counts.values()),
        "tensor_count": len(tensors),
        "signer": signer,
        "gates": gates,
    }


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if not torch.cuda.is_available():
        raise Phase3Error("registered CUDA device unavailable")
    checks = []
    for spec in protocol["systems"]:
        expected = _rows(root / spec["qualified_outputs"])
        metadata = _json(root / spec["candidate_metadata"])
        state = load_file(str(root / spec["candidate_checkpoint"]), device="cpu")
        inherited = load_file(str(root / spec["components"]["inherited_residual"]), device="cpu")
        checks.append(
            {
                "seed": int(spec["seed"]),
                "qualified_rows": len(expected),
                "candidate_sha256": sha256_file(root / spec["candidate_checkpoint"]),
                "metadata_checkpoint_exact": metadata["checkpoint"]["sha256"]
                == sha256_file(root / spec["candidate_checkpoint"]),
                "inherited_four_routes_exact": torch.equal(state["norm.weight"], inherited["norm.weight"])
                and torch.equal(state["norm.bias"], inherited["norm.bias"])
                and torch.equal(state["down"][:4], inherited["down"])
                and torch.equal(state["up"][:4], inherited["up"]),
            }
        )
    gates = {
        "three_registered_seeds": [row["seed"] for row in checks] == list(SEEDS),
        "all_qualified_matrices_complete": all(row["qualified_rows"] == 1400 for row in checks),
        "all_candidate_metadata_exact": all(row["metadata_checkpoint_exact"] for row in checks),
        "all_inherited_routes_exact": all(row["inherited_four_routes_exact"] for row in checks),
        "training_absent": True,
        "teacher_loading_absent": True,
        "final_test_not_accessed": True,
    }
    return {
        "format": "abi-capability-compiler-phase4-b40-v25-product-conformance-preflight/1",
        "status": "PASS_B40_V25_PRODUCT_CONFORMANCE_PREFLIGHT" if all(gates.values()) else "FAIL_B40_V25_PRODUCT_CONFORMANCE_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "checks": checks,
        "gates": gates,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
    }


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("immutable B40 v25 output exists or CUDA unavailable")
    api = _api((root / protocol["layercake_root"]).resolve())
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(protocol["research_signing_seed_hex"]))
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    probes_list = development_probes(root / protocol["development_catalog"])
    probes = {str(row["probe_id"]): row for row in probes_list}
    teacher = {str(row["probe_id"]): row for row in _rows(root / protocol["teacher_reference"])}
    systems = []
    aggregate = []
    output.mkdir(parents=True)
    for spec in protocol["systems"]:
        qualified_rows = _rows(root / spec["qualified_outputs"])
        qualified = {str(row["probe_id"]): row for row in qualified_rows}
        with tempfile.TemporaryDirectory(prefix=f"abi-b40-v25-{spec['seed']}-") as raw:
            temporary = Path(raw)
            package = _package(root, protocol, spec, temporary / "candidate.cake", api, private, public_pem)
            host = api["ClarificationRouteAllocationBoundedCoreHost"](
                temporary / "registry", trust_store={package["signer"]: public_pem}, device="cuda"
            )
            active = host.activate(temporary / "candidate.cake")
            rows = []
            started = time.perf_counter()
            for index, probe in enumerate(probes_list, 1):
                probe_id = str(probe["probe_id"])
                capability = str(probe["canonical_capability"])
                expected = qualified[probe_id]
                value, terminated, physical_route, pointer, format_record = _generate(
                    host, str(probe["prompt"]), int(probe["max_new_tokens"]), capability
                )
                routed = host.route(str(probe["prompt"]))
                row = {
                    **expected,
                    "output": value,
                    "original_output": value,
                    "output_token_ids": [int(token) for token in host.model_tokenizer.encode(value)],
                    "automatic_capability_route": routed,
                    "capability_route_correct": routed == capability,
                    "physical_residual_route": physical_route,
                    "active_residual_routes": 0 if physical_route < 0 else 1,
                    "fifth_clarification_route_active": physical_route == 4,
                    "guard_terminated": terminated,
                    "functional_pass_v1": evaluate_functional(value, probe["evaluator"]),
                    "functional_pass_v2": evaluate_functional_v2(value, probe["evaluator"], capability),
                    "repetition_collapse_v2": repetition_collapse_v2(value),
                    "v25_pointer": pointer,
                    "v25_format": format_record,
                    "output_exact_to_qualified_matrix": value == str(expected["output"]),
                }
                rows.append(row)
                aggregate.append({"seed": int(spec["seed"]), **row})
                if index % 200 == 0:
                    print(json.dumps({"seed": spec["seed"], "evaluated": index}), flush=True)
            verified = host.verify()
            candidate_state = host.residual.state_dict()
            inherited = load_file(str(root / spec["components"]["inherited_residual"]), device="cpu")
            inherited_exact = (
                torch.equal(candidate_state["norm.weight"].cpu(), inherited["norm.weight"])
                and torch.equal(candidate_state["norm.bias"].cpu(), inherited["norm.bias"])
                and torch.equal(candidate_state["down"][:4].cpu(), inherited["down"])
                and torch.equal(candidate_state["up"][:4].cpu(), inherited["up"])
            )
            del host
            gc.collect()
            torch.cuda.empty_cache()
        raw_path = output / f"seed{spec['seed']}_outputs.jsonl"
        _write_immutable(raw_path, b"".join(canonical_json_bytes(row) for row in rows))
        evaluation = _merged_evaluation(rows)
        quality, relative = _quality_gates(
            protocol, evaluation, rows, probes, teacher, int(spec["seed"]) + 9_600_000
        )
        quality.pop("strong_parent_exact")
        quality.pop("training_absent")
        conformance = {
            "all_1400_outputs_byte_exact": len(rows) == 1400
            and all(bool(row["output_exact_to_qualified_matrix"]) for row in rows),
            "all_physical_routes_exact": all(
                int(row["physical_residual_route"]) == _route_for_capability(str(row["capability"]))
                for row in rows
            ),
            "clarification_route_four_100_of_100": sum(
                row["capability"] == "clarification" and row["physical_residual_route"] == 4
                for row in rows
            )
            == 100,
            "one_active_route_maximum": all(int(row["active_residual_routes"]) in {0, 1} for row in rows),
            "inherited_four_routes_exact": inherited_exact,
            "signed_package_identity": active["archive_hash"] == package["archive_sha256"]
            and active["payload_hash"] == package["tensor_payload_hash"],
            "package_verified": verified["status"] == "PASS",
            "one_authenticated_parse": active["authenticated_package_parses"] == 1,
            "strict_storage_adoption": active["strict_assigned_tensor_count"]
            == active["authenticated_tensor_count"]
            == 89
            and active["meta_tensors_after_adoption"] == 0,
            "receiver_learning_zero": active["receiver_training_steps"]
            == active["receiver_calibration_runs"]
            == 0,
            "teacher_absent": True,
            "final_test_not_accessed": True,
        }
        machine = all(quality.values()) and all(conformance.values())
        systems.append(
            {
                "budget": "B40",
                "seed": int(spec["seed"]),
                "status": "PASS" if machine else "FAIL",
                "machine_gates_pass": machine,
                "evaluation": evaluation,
                "teacher_comparison_v1": relative,
                "quality_gates": quality,
                "conformance_gates": conformance,
                "changed_outputs": sum(not row["output_exact_to_qualified_matrix"] for row in rows),
                "package": {key: value for key, value in package.items() if key != "signer"},
                "activation": active,
                "evaluation_wall_seconds": time.perf_counter() - started,
                "outputs": {
                    "path": str(raw_path.relative_to(root)).replace("\\", "/"),
                    "sha256": sha256_file(raw_path),
                },
            }
        )
    aggregate_path = output / "all_outputs.jsonl"
    _write_immutable(aggregate_path, b"".join(canonical_json_bytes(row) for row in aggregate))
    stable = len(systems) == 3 and all(system["machine_gates_pass"] for system in systems)
    result = {
        "format": "abi-capability-compiler-phase4-b40-v25-product-conformance-result/1",
        "status": "PASS_STABLE_B40_SIGNED_V25_PRODUCT_CONFORMANCE" if stable else "FAIL_B40_SIGNED_V25_PRODUCT_CONFORMANCE",
        "protocol_sha256": protocol_sha,
        "systems": systems,
        "topology": [bool(system["machine_gates_pass"]) for system in systems],
        "three_seed_all_pass": stable,
        "observations": len(aggregate),
        "model_inference_rows": len(aggregate),
        "aggregate_outputs_sha256": sha256_file(aggregate_path),
        "training_performed": False,
        "teacher_model_loaded": False,
        "receiver_training_steps": 0,
        "final_test_accessed": False,
        "phase4_certified": False,
        "stable_minimum_established": False,
        "claim_boundary": "Three-seed signed B40 V25 development product conformance only. Independent verification plus the B20 boundary are required for a tested-minimum claim; runtime, matched baselines, final test, Phase 4, and ABI superiority remain open.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = root / args.protocol
    if args.preflight:
        result = preflight(root, protocol_path)
    elif args.output_dir:
        result = run(root, protocol_path, root / args.output_dir)
    else:
        raise Phase3Error("select preflight or output-dir")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
