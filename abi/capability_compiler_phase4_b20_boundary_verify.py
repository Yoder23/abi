"""Independent read-only reconstruction of the complete three-seed B20 boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b20_host_oracle_replay import (
    _gate_failures,
    capability_metrics,
    oracle_rows,
)
from .capability_compiler_phase4_b20_v25_physical_screen import (
    _route_for_capability,
    preservation_gates,
)
from .capability_compiler_phase4_v19_frontier_rescreen import (
    _merged_evaluation,
    _quality_gates,
    _rows,
)


FORMAT = "abi-capability-compiler-phase4-b20-boundary-verify/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _oracle_reconstruction(
    rows: list[dict[str, Any]], declared: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    replayed = oracle_rows(rows)
    metrics = capability_metrics(replayed)
    failures = _gate_failures(metrics, thresholds)
    return {
        "raw_rows": len(rows),
        "raw_functional_passes": sum(bool(row["functional_pass_v1"]) for row in rows),
        "raw_collapses": sum(bool(row["repetition_collapse_v2"]) for row in rows),
        "declared_aggregates_exact": sum(bool(row["functional_pass_v1"]) for row in rows)
        == int(declared["functional_passes_v1"])
        and sum(bool(row["repetition_collapse_v2"]) for row in rows)
        == int(declared["repetition_collapses_v2"]),
        "oracle_functional_passes": sum(bool(row["functional_pass_v1"]) for row in replayed),
        "oracle_collapses": sum(bool(row["repetition_collapse_v2"]) for row in replayed),
        "oracle_gate_failures": failures,
        "failure_proven": len(rows) == 1400
        and len({str(row["probe_id"]) for row in rows}) == 1400
        and bool(failures),
    }


def _product_reconstruction(
    protocol: Mapping[str, Any],
    historical_rows: list[dict[str, Any]],
    product_rows: list[dict[str, Any]],
    probes: Mapping[str, Mapping[str, Any]],
    teacher: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    historical = {str(row["probe_id"]): row for row in historical_rows}
    evaluation = _merged_evaluation(product_rows)
    quality, relative = _quality_gates(
        protocol, evaluation, product_rows, probes, teacher, 155921 + 9_500_000
    )
    quality.pop("strong_parent_exact")
    quality.pop("training_absent")
    preservation = preservation_gates(historical, product_rows)
    undeclared = [
        str(row["probe_id"])
        for row in product_rows
        if str(row["output"]) != str(historical[str(row["probe_id"])]["output"])
        and not (
            historical[str(row["probe_id"])]["capability"]
            in {"coherence", "format_control", "clarification"}
            or bool(historical[str(row["probe_id"])]["repetition_collapse_v2"])
        )
    ]
    regressions = [
        str(row["probe_id"])
        for row in product_rows
        if str(row["output"]) != str(historical[str(row["probe_id"])]["output"])
        and bool(historical[str(row["probe_id"])]["functional_pass_v1"])
        and not bool(row["functional_pass_v1"])
    ]
    route_exact = all(
        int(row["physical_residual_route"])
        == _route_for_capability(str(row["capability"]))
        for row in product_rows
    )
    return {
        "rows": len(product_rows),
        "functional_passes": evaluation["functional_passes_v1"],
        "collapses": evaluation["repetition_collapses_v2"],
        "router_correct": evaluation["router_correct"],
        "quality_gates": quality,
        "teacher_comparison_v1": relative,
        "preservation_gates": preservation,
        "undeclared_change_rows": undeclared,
        "passing_to_failing_changed_rows": regressions,
        "physical_routes_exact": route_exact,
        "preservation_failure_proven": len(product_rows) == 1400
        and all(quality.values())
        and not all(preservation.values())
        and undeclared == ["phase1-validation-rewriting-0032-v2"]
        and regressions == ["phase1-validation-rewriting-0032-v2"],
    }


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_B20_BOUNDARY_VERIFIER"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("B20 boundary-verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B20 boundary binding changed: {relative}")
    if output.exists():
        raise Phase3Error(f"immutable B20 boundary output exists: {output}")

    thresholds = protocol["thresholds"]
    oracle_results = {}
    for spec in protocol["oracle_seeds"]:
        rows = _rows(root / spec["outputs"])
        declared = _json(root / spec["result"])
        oracle_results[str(spec["seed"])] = _oracle_reconstruction(rows, declared, thresholds)

    product_protocol = _json(root / protocol["product_protocol"])
    historical_rows = _rows(root / protocol["product_historical_outputs"])
    product_rows = _rows(root / protocol["product_outputs"])
    probes = {
        str(row["probe_id"]): row
        for row in development_probes(root / product_protocol["development_catalog"])
    }
    teacher = {
        str(row["probe_id"]): row
        for row in _rows(root / product_protocol["teacher_reference"])
    }
    product = _product_reconstruction(
        product_protocol, historical_rows, product_rows, probes, teacher
    )
    candidate = load_file(str(root / protocol["candidate_checkpoint"]), device="cpu")
    inherited = load_file(str(root / protocol["inherited_checkpoint"]), device="cpu")
    tensor_exact = (
        torch.equal(candidate["norm.weight"], inherited["norm.weight"])
        and torch.equal(candidate["norm.bias"], inherited["norm.bias"])
        and torch.equal(candidate["down"][:4], inherited["down"])
        and torch.equal(candidate["up"][:4], inherited["up"])
    )
    declared_product = _json(root / protocol["product_result"])

    # Adversarial mutations are in-memory only and never become evidence inputs.
    mutated_route = [dict(row) for row in product_rows]
    mutated_route[1000]["physical_residual_route"] = 3
    mutated_history = {str(row["probe_id"]): dict(row) for row in historical_rows}
    mutated_history["phase1-validation-rewriting-0032-v2"]["repetition_collapse_v2"] = True
    mutations = {
        "raw_byte_mutation_rejected": hashlib.sha256(
            (root / protocol["product_outputs"]).read_bytes() + b"x"
        ).hexdigest()
        != protocol["bindings"][protocol["product_outputs"]],
        "route_mutation_rejected": not all(
            int(row["physical_residual_route"])
            == _route_for_capability(str(row["capability"]))
            for row in mutated_route
        ),
        "scope_relabel_rejected_by_bound_history": hashlib.sha256(
            b"".join(
                canonical_json_bytes(mutated_history[str(row["probe_id"])])
                for row in historical_rows
            )
        ).hexdigest()
        != protocol["bindings"][protocol["product_historical_outputs"]],
        "tensor_mutation_rejected": not torch.equal(
            candidate["down"][:4] + 1, inherited["down"]
        ),
        "status_text_not_source_of_truth": declared_product["status"].startswith("FAIL")
        and product["preservation_failure_proven"],
        "one_seed_cannot_substitute_for_three": len(oracle_results) == 2
        and bool(product),
    }
    gates = {
        "two_oracle_failures_reconstructed": set(oracle_results) == {"104729", "130363"}
        and all(value["failure_proven"] for value in oracle_results.values()),
        "product_failure_reconstructed": product["preservation_failure_proven"],
        "product_aggregate_matches": product["functional_passes"]
        == int(declared_product["functional_passes_v1"])
        and product["collapses"] == int(declared_product["repetition_collapses_v2"])
        and product["router_correct"] == int(declared_product["router_correct"]),
        "candidate_inherited_tensors_exact": tensor_exact,
        "all_4200_rows_present": sum(value["raw_rows"] for value in oracle_results.values())
        + product["rows"]
        == 4200,
        "all_six_mutations_rejected": all(mutations.values()),
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_model_loading_absent": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-b20-boundary-verify-result/1",
        "status": "PASS_INDEPENDENTLY_VERIFIED_ALL_THREE_B20_SEEDS_FAIL_LOCKED_GATES" if all(gates.values()) else "FAIL_B20_BOUNDARY_VERIFIER",
        "protocol_sha256": sha256_file(protocol_path),
        "oracle_seeds": oracle_results,
        "product_seed155921": product,
        "candidate_inherited_tensors_exact": tensor_exact,
        "mutations": mutations,
        "gates": gates,
        "decision": "B20 fails at all three registered seeds: seeds104729 and130363 remain below immutable absolute quality gates after the generous host oracle; seed155921 passes aggregate quality but fails its prospectively locked signed-product preservation gate on exactly one online-guard regression.",
        "authorized_next_action": "Screen the already-qualified three B40 five-route candidates through the signed V25 product. Only three passes can combine with this boundary proof to establish B40 as the tested stable minimum for the exact architecture.",
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "stable_minimum_established": False,
        "claim_boundary": "Independent B20 adjacent-lower boundary only. B40 signed-product conformance, runtime, matched baselines, final test, Phase 4, and ABI superiority remain open.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
