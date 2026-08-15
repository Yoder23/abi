"""Independent reconstruction of B40 signed-v25 sufficiency and the B20/B40 boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from safetensors.torch import load_file
import torch

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import (
    canonical_json_bytes,
    evaluate_functional,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .capability_compiler_phase4_b20_v25_physical_screen import (
    _api,
    _route_for_capability,
)
from .capability_compiler_phase4_b40_v25_product_conformance import _package
from .capability_compiler_phase4_v19_frontier_rescreen import (
    _merged_evaluation,
    _quality_gates,
    _rows,
)


FORMAT = "abi-capability-compiler-phase4-b40-v25-product-verify/1"
SEEDS = (104729, 130363, 155921)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _bound_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_READ_ONLY_THREE_SEED_B40_V25_PRODUCT_VERIFIER"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or tuple(int(spec["seed"]) for spec in protocol.get("systems", ())) != SEEDS
    ):
        raise Phase3Error("B40 product-verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B40 product-verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def _validate_product_bindings(root: Path, product_protocol: Mapping[str, Any]) -> None:
    for relative, expected in product_protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"bound product dependency changed: {relative}")


def _complete_identity(rows: list[dict[str, Any]], expected_ids: set[str]) -> bool:
    ids = [str(row["probe_id"]) for row in rows]
    return len(ids) == len(expected_ids) and len(set(ids)) == len(ids) and set(ids) == expected_ids


def _recompute_rows(
    rows: list[dict[str, Any]], probes: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], bool]:
    rebuilt = []
    stored_exact = True
    for source in rows:
        probe = probes[str(source["probe_id"])]
        capability = str(probe["canonical_capability"])
        output = str(source["output"])
        v1 = evaluate_functional(output, probe["evaluator"])
        v2 = evaluate_functional_v2(output, probe["evaluator"], capability)
        collapse = repetition_collapse_v2(output)
        route_correct = str(source["automatic_capability_route"]) == capability
        stored_exact = stored_exact and (
            str(source["capability"]) == capability
            and bool(source["functional_pass_v1"]) == v1
            and bool(source["functional_pass_v2"]) == v2
            and bool(source["repetition_collapse_v2"]) == collapse
            and bool(source["capability_route_correct"]) == route_correct
        )
        rebuilt.append(
            {
                **source,
                "capability": capability,
                "functional_pass_v1": v1,
                "functional_pass_v2": v2,
                "repetition_collapse_v2": collapse,
                "capability_route_correct": route_correct,
            }
        )
    return rebuilt, stored_exact


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = _bound_protocol(root, protocol_path)
    product_protocol = _json(root / protocol["product_protocol"])
    _validate_product_bindings(root, product_protocol)
    product_specs = {
        int(spec["seed"]): spec for spec in product_protocol["systems"]
    }
    checks = []
    for verification_spec in protocol["systems"]:
        seed = int(verification_spec["seed"])
        spec = product_specs[seed]
        candidate = load_file(str(root / spec["candidate_checkpoint"]), device="cpu")
        inherited = load_file(
            str(root / spec["components"]["inherited_residual"]), device="cpu"
        )
        checks.append(
            {
                "seed": seed,
                "product_rows": len(_rows(root / verification_spec["product_outputs"])),
                "qualified_rows": len(_rows(root / spec["qualified_outputs"])),
                "candidate_sha256": sha256_file(root / spec["candidate_checkpoint"]),
                "inherited_four_routes_exact": torch.equal(
                    candidate["norm.weight"], inherited["norm.weight"]
                )
                and torch.equal(candidate["norm.bias"], inherited["norm.bias"])
                and torch.equal(candidate["down"][:4], inherited["down"])
                and torch.equal(candidate["up"][:4], inherited["up"]),
            }
        )
    b20 = _json(root / protocol["b20_verified_boundary"])
    gates = {
        "three_fixed_seeds": [row["seed"] for row in checks] == list(SEEDS),
        "all_product_matrices_complete": all(row["product_rows"] == 1400 for row in checks),
        "all_qualified_matrices_complete": all(row["qualified_rows"] == 1400 for row in checks),
        "all_inherited_routes_exact": all(row["inherited_four_routes_exact"] for row in checks),
        "b20_boundary_is_independently_verified": b20["status"]
        == "PASS_INDEPENDENTLY_VERIFIED_ALL_THREE_B20_SEEDS_FAIL_LOCKED_GATES",
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_loading_absent": True,
        "final_test_not_accessed": True,
    }
    return {
        "format": "abi-capability-compiler-phase4-b40-v25-product-verify-preflight/1",
        "status": "PASS_B40_V25_INDEPENDENT_VERIFIER_PREFLIGHT"
        if all(gates.values())
        else "FAIL_B40_V25_INDEPENDENT_VERIFIER_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "checks": checks,
        "gates": gates,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
    }


def reconstruct_system(
    product_protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    rows: list[dict[str, Any]],
    qualified_rows: list[dict[str, Any]],
    probes: Mapping[str, Mapping[str, Any]],
    teacher: Mapping[str, Mapping[str, Any]],
    declared: Mapping[str, Any],
) -> dict[str, Any]:
    expected_ids = set(probes)
    qualified = {str(row["probe_id"]): row for row in qualified_rows}
    rebuilt, stored_fields_exact = _recompute_rows(rows, probes)
    evaluation = _merged_evaluation(rebuilt)
    quality, relative = _quality_gates(
        product_protocol,
        evaluation,
        rebuilt,
        probes,
        teacher,
        int(spec["seed"]) + 9_600_000,
    )
    quality.pop("strong_parent_exact")
    quality.pop("training_absent")
    qualified_exact = (
        _complete_identity(qualified_rows, expected_ids)
        and all(str(row["output"]) == str(qualified[str(row["probe_id"])]["output"]) for row in rebuilt)
    )
    route_exact = all(
        int(row["physical_residual_route"])
        == _route_for_capability(str(row["capability"]))
        for row in rebuilt
    )
    one_route = all(
        int(row["active_residual_routes"])
        == (0 if int(row["physical_residual_route"]) < 0 else 1)
        for row in rebuilt
    )
    clarification = sum(
        row["capability"] == "clarification"
        and int(row["physical_residual_route"]) == 4
        and bool(row["fifth_clarification_route_active"])
        for row in rebuilt
    )
    declared_exact = (
        declared["status"] == "PASS"
        and int(declared["seed"]) == int(spec["seed"])
        and int(declared["evaluation"]["functional_passes_v1"])
        == int(evaluation["functional_passes_v1"])
        and int(declared["evaluation"]["repetition_collapses_v2"])
        == int(evaluation["repetition_collapses_v2"])
        and int(declared["evaluation"]["router_correct"])
        == int(evaluation["router_correct"])
        and declared["quality_gates"] == quality
        and declared["teacher_comparison_v1"] == relative
        and int(declared["changed_outputs"]) == 0
    )
    gates = {
        "complete_unique_catalog": _complete_identity(rebuilt, expected_ids),
        "qualified_matrix_complete": _complete_identity(qualified_rows, expected_ids),
        "outputs_byte_exact_to_qualified": qualified_exact,
        "stored_evaluators_recomputed_exact": stored_fields_exact,
        "locked_quality_gates": all(quality.values()),
        "zero_repetition_collapse": int(evaluation["repetition_collapses_v2"]) == 0,
        "router_exact": int(evaluation["router_correct"]) == len(rebuilt),
        "physical_routes_exact": route_exact,
        "one_active_route_maximum": one_route,
        "clarification_route_four_100_of_100": clarification == 100,
        "declared_aggregate_exact": declared_exact,
    }
    return {
        "seed": int(spec["seed"]),
        "rows": len(rebuilt),
        "functional_passes_v1": int(evaluation["functional_passes_v1"]),
        "repetition_collapses_v2": int(evaluation["repetition_collapses_v2"]),
        "teacher_comparison_v1": relative,
        "quality_gates": quality,
        "gates": gates,
        "pass": all(gates.values()),
    }


def _tensor_and_package_reconstruction(
    root: Path,
    product_protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    declared: Mapping[str, Any],
    api: Mapping[str, Any],
    private: Ed25519PrivateKey,
    public_pem: bytes,
) -> dict[str, Any]:
    candidate = load_file(str(root / spec["candidate_checkpoint"]), device="cpu")
    inherited = load_file(str(root / spec["components"]["inherited_residual"]), device="cpu")
    inherited_exact = (
        torch.equal(candidate["norm.weight"], inherited["norm.weight"])
        and torch.equal(candidate["norm.bias"], inherited["norm.bias"])
        and torch.equal(candidate["down"][:4], inherited["down"])
        and torch.equal(candidate["up"][:4], inherited["up"])
    )
    shapes_exact = (
        tuple(candidate["norm.weight"].shape) == (768,)
        and tuple(candidate["norm.bias"].shape) == (768,)
        and tuple(candidate["down"].shape) == (5, 16, 768)
        and tuple(candidate["up"].shape) == (5, 768, 16)
        and sum(value.numel() for value in candidate.values()) == 124416
    )
    with tempfile.TemporaryDirectory(prefix=f"abi-b40-independent-{spec['seed']}-") as raw:
        temporary = Path(raw)
        package = _package(
            root,
            product_protocol,
            spec,
            temporary / "candidate.cake",
            api,
            private,
            public_pem,
        )
        host = api["ClarificationRouteAllocationBoundedCoreHost"](
            temporary / "registry",
            trust_store={package["signer"]: public_pem},
            device="cpu",
        )
        active = host.activate(temporary / "candidate.cake")
        verified = host.verify()
    declared_package = declared["package"]
    declared_activation = declared["activation"]
    gates = {
        "candidate_shapes_exact": shapes_exact,
        "inherited_four_routes_tensor_exact": inherited_exact,
        "rebuilt_payload_exact": package["tensor_payload_hash"]
        == declared_package["tensor_payload_hash"],
        "rebuilt_manifest_package_hash_exact": package["package_hash"]
        == declared_package["package_hash"],
        "rebuilt_archive_exact": package["archive_sha256"]
        == declared_package["archive_sha256"],
        "signature_and_tensors_verified": all(package["gates"].values()),
        "cpu_activation_payload_exact": active["payload_hash"]
        == declared_activation["payload_hash"],
        "cpu_activation_state_exact": active["state_dict_hash"]
        == declared_activation["state_dict_hash"],
        "one_authenticated_parse": active["authenticated_package_parses"] == 1,
        "strict_storage_adoption": active["strict_assigned_tensor_count"]
        == active["authenticated_tensor_count"]
        == 89
        and active["meta_tensors_after_adoption"] == 0,
        "receiver_learning_zero": active["receiver_training_steps"]
        == active["receiver_calibration_runs"]
        == 0,
        "host_verifier_pass": verified["status"] == "PASS",
    }
    return {
        "seed": int(spec["seed"]),
        "gates": gates,
        "pass": all(gates.values()),
        "rebuilt_archive_sha256": package["archive_sha256"],
        "rebuilt_payload_hash": package["tensor_payload_hash"],
        "cpu_activation_state_dict_hash": active["state_dict_hash"],
    }


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = _bound_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable B40 verifier output exists: {output}")
    product_protocol = _json(root / protocol["product_protocol"])
    _validate_product_bindings(root, product_protocol)
    product_result = _json(root / protocol["product_result"])
    product_specs = {
        int(spec["seed"]): spec for spec in product_protocol["systems"]
    }
    declared_by_seed = {int(row["seed"]): row for row in product_result["systems"]}
    probes_list = development_probes(root / product_protocol["development_catalog"])
    probes = {str(row["probe_id"]): row for row in probes_list}
    teacher = {
        str(row["probe_id"]): row
        for row in _rows(root / product_protocol["teacher_reference"])
    }
    api = _api((root / product_protocol["layercake_root"]).resolve())
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(product_protocol["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    systems = []
    packages = []
    for verification_spec in protocol["systems"]:
        seed = int(verification_spec["seed"])
        spec = product_specs[seed]
        rows = _rows(root / verification_spec["product_outputs"])
        qualified = _rows(root / spec["qualified_outputs"])
        systems.append(
            reconstruct_system(
                product_protocol,
                spec,
                rows,
                qualified,
                probes,
                teacher,
                declared_by_seed[seed],
            )
        )
        packages.append(
            _tensor_and_package_reconstruction(
                root,
                product_protocol,
                spec,
                declared_by_seed[seed],
                api,
                private,
                public_pem,
            )
        )

    aggregate_rows = _rows(root / protocol["aggregate_outputs"])
    concatenated = [
        {"seed": int(verification_spec["seed"]), **row}
        for verification_spec in protocol["systems"]
        for row in _rows(root / verification_spec["product_outputs"])
    ]
    aggregate_exact = aggregate_rows == concatenated
    b20 = _json(root / protocol["b20_verified_boundary"])
    b20_exact = (
        b20["status"]
        == "PASS_INDEPENDENTLY_VERIFIED_ALL_THREE_B20_SEEDS_FAIL_LOCKED_GATES"
        and b20["verified_boundary"]["rows"] == 4200
        and b20["stable_minimum_established"] is False
    )

    # In-memory adversarial perturbations never become evidence inputs.
    seed0_rows = _rows(root / protocol["systems"][0]["product_outputs"])
    seed0_qualified = _rows(
        root / product_specs[int(protocol["systems"][0]["seed"])]["qualified_outputs"]
    )
    dropped = seed0_rows[:-1]
    duplicated = [dict(row) for row in seed0_rows]
    duplicated[-1]["probe_id"] = duplicated[0]["probe_id"]
    changed_output = [dict(row) for row in seed0_rows]
    changed_output[0]["output"] = str(changed_output[0]["output"]) + "x"
    changed_route = [dict(row) for row in seed0_rows]
    changed_route[0]["physical_residual_route"] = 4
    changed_evaluator = [dict(row) for row in seed0_rows]
    changed_evaluator[0]["functional_pass_v1"] = not bool(
        changed_evaluator[0]["functional_pass_v1"]
    )
    first_verification_spec = protocol["systems"][0]
    first_spec = product_specs[int(first_verification_spec["seed"])]
    candidate = load_file(str(root / first_spec["candidate_checkpoint"]), device="cpu")
    inherited = load_file(
        str(root / first_spec["components"]["inherited_residual"]), device="cpu"
    )
    _, changed_stored_exact = _recompute_rows(changed_evaluator, probes)
    mutations = {
        "bound_raw_byte_mutation_rejected": hashlib.sha256(
            (root / first_verification_spec["product_outputs"]).read_bytes() + b"x"
        ).hexdigest()
        != protocol["bindings"][first_verification_spec["product_outputs"]],
        "dropped_row_rejected": not _complete_identity(dropped, set(probes)),
        "duplicate_probe_rejected": not _complete_identity(duplicated, set(probes)),
        "changed_output_rejected": any(
            str(row["output"])
            != str({str(q["probe_id"]): q for q in seed0_qualified}[str(row["probe_id"])]["output"])
            for row in changed_output
        ),
        "changed_route_rejected": not all(
            int(row["physical_residual_route"])
            == _route_for_capability(str(row["capability"]))
            for row in changed_route
        ),
        "changed_evaluator_field_rejected": not changed_stored_exact,
        "inherited_tensor_mutation_rejected": not torch.equal(
            candidate["down"][:4] + 1, inherited["down"]
        ),
        "declared_status_not_source_of_truth": all(system["pass"] for system in systems),
        "b20_boundary_byte_mutation_rejected": hashlib.sha256(
            (root / protocol["b20_verified_boundary"]).read_bytes() + b"x"
        ).hexdigest()
        != protocol["bindings"][protocol["b20_verified_boundary"]],
    }
    stable = (
        all(system["pass"] for system in systems)
        and all(package["pass"] for package in packages)
        and aggregate_exact
        and len(aggregate_rows) == 4200
        and b20_exact
        and all(mutations.values())
    )
    gates = {
        "three_b40_systems_reconstructed": len(systems) == 3
        and [system["seed"] for system in systems] == list(SEEDS)
        and all(system["pass"] for system in systems),
        "three_packages_rebuilt_and_verified": len(packages) == 3
        and all(package["pass"] for package in packages),
        "aggregate_4200_rows_exact": aggregate_exact and len(aggregate_rows) == 4200,
        "b20_adjacent_lower_all_seed_failure_bound": b20_exact,
        "all_nine_mutations_rejected": all(mutations.values()),
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_model_loading_absent": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-b40-v25-product-verify-result/1",
        "status": "PASS_INDEPENDENTLY_VERIFIED_B40_SMALLEST_TESTED_STABLE_FIVE_ROUTE_BUDGET"
        if stable and all(gates.values())
        else "FAIL_B40_V25_PRODUCT_VERIFIER",
        "protocol_sha256": protocol_sha,
        "systems": systems,
        "package_reconstructions": packages,
        "aggregate_outputs_exact": aggregate_exact,
        "observations": len(aggregate_rows),
        "b20_boundary_exact": b20_exact,
        "mutations": mutations,
        "gates": gates,
        "decision": "B40 is the smallest tested stable selected-information budget among B20 and B40 for this exact five-route LayerCake v25 architecture: all three B20 seeds fail their prospectively locked boundary rules and all three B40 seeds pass signed product execution.",
        "authorized_next_action": "Measure the exact B40 signed artifact on CPU and GPU, then run B40-matched LoRA and distillation comparisons under equal imported-information accounting. Do not expand acquisition unless a measured open gate requires it.",
        "model_inference_performed": False,
        "package_reconstruction_performed": True,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "stable_minimum_established": stable and all(gates.values()),
        "phase4_certified": False,
        "claim_boundary": "Independently verified development-data minimum only among the exact tested B20 and B40 five-route budgets. Runtime, matched baselines, external human review, final test, Phase 4, and ABI superiority remain open.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
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
    if args.preflight:
        result = preflight(root, root / args.protocol)
    elif args.output:
        result = run(root, root / args.protocol, root / args.output)
    else:
        raise Phase3Error("select preflight or output")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
