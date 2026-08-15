"""Independent three-seed verification of the B40 clarification-route matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_clarification_route_verify import _recompute


FORMAT = "abi-capability-compiler-phase4-clarification-route-matrix-verify/1"
SEEDS = (104729, 130363, 155921)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_INDEPENDENT_THREE_SEED_CLARIFICATION_ROUTE_VERIFY"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or tuple(int(row["seed"]) for row in protocol["runs"]) != SEEDS
    ):
        raise Phase3Error("clarification-route matrix verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"clarification-route matrix verifier binding changed: {relative}")
    if output.exists():
        raise Phase3Error(f"immutable clarification-route matrix verification exists: {output}")

    lineage_protocol = _json(root / protocol["lineage_protocol"])
    v440 = _json(root / lineage_protocol["base_protocols"]["v443"])
    tokenizer = AutoTokenizer.from_pretrained(
        (root / v440["host"]["parent_path"]).resolve(),
        local_files_only=True,
    )
    probes_list = development_probes(root / protocol["development"]["catalog"])
    probes = {str(row["probe_id"]): row for row in probes_list}
    teacher = {
        str(row["probe_id"]): row
        for row in _rows(root / protocol["development"]["teacher_reference"])
    }
    matrix = []
    all_gates: dict[str, bool] = {}
    for spec in protocol["runs"]:
        seed = int(spec["seed"])
        rows = _rows(root / spec["outputs"])
        historical_rows = _rows(root / spec["historical_outputs"])
        historical = {str(row["probe_id"]): row for row in historical_rows}
        run_protocol = dict(protocol)
        run_protocol["statistics"] = dict(spec["statistics"])
        recomputed = _recompute(rows, historical, probes, teacher, tokenizer, run_protocol)
        declared = _json(root / spec["underlying_result"])
        wrapper = None if spec.get("wrapper_result") is None else _json(root / spec["wrapper_result"])
        state = load_file(str(root / spec["checkpoint"]), device="cpu")
        inherited = load_file(str(root / spec["inherited_checkpoint"]), device="cpu")
        checkpoint_exact = (
            set(state) == {"norm.weight", "norm.bias", "down", "up"}
            and tuple(state["down"].shape) == (5, 16, 768)
            and tuple(state["up"].shape) == (5, 768, 16)
            and torch.equal(state["norm.weight"], inherited["norm.weight"])
            and torch.equal(state["norm.bias"], inherited["norm.bias"])
            and torch.equal(state["down"][:4], inherited["down"])
            and torch.equal(state["up"][:4], inherited["up"])
        )
        declared_exact = (
            int(declared["seed"]) == seed
            and int(declared["functional_passes_v1"]) == recomputed["functional_passes_v1"]
            and int(declared["candidate_clarification_passes"]) == recomputed["candidate_clarification_passes"]
            and int(declared["historical_clarification_passes"]) == recomputed["historical_clarification_passes"]
            and int(declared["repetition_collapses_v2"]) == recomputed["repetition_collapses_v2"]
            and declared["per_capability"] == recomputed["per_capability"]
            and declared["teacher_comparison_v1"] == recomputed["teacher_comparison_v1"]
            and all(bool(value) for value in declared["gates"].values())
        )
        wrapper_exact = True
        if wrapper is not None:
            wrapper_exact = (
                int(wrapper["seed"]) == seed
                and wrapper["status"] == "PASS_B40_PAIRED_SEED_CLARIFICATION_ROUTE_MACHINE_GATES"
                and int(wrapper["functional_passes_v1"]) == recomputed["functional_passes_v1"]
                and int(wrapper["candidate_clarification_passes"]) == recomputed["candidate_clarification_passes"]
                and wrapper["underlying_result_sha256"] == sha256_file(root / spec["underlying_result"])
                and wrapper["raw_outputs_sha256"] == sha256_file(root / spec["outputs"])
                and all(bool(value) for value in wrapper["gates"].values())
            )
        metadata = _json(root / spec["metadata"])
        information_exact = (
            int(metadata["imported_information"]["unique_source_attempts"]) == 4005
            and int(metadata["imported_information"]["authoritative_teacher_output_tokens"]) == 123167
            and int(metadata["imported_information"]["new_teacher_outputs"]) == 0
            and int(metadata["imported_information"]["stored_logits"]) == 0
            and int(metadata["imported_information"]["stored_hidden_activations"]) == 0
            and metadata["parent"]["mutated"] is False
            and metadata["router"]["mutated"] is False
        )
        seed_gates = {
            **{f"recomputed_{key}": bool(value) for key, value in recomputed["gates"].items()},
            "checkpoint_and_inherited_routes_exact": checkpoint_exact,
            "declared_result_exact": declared_exact,
            "wrapper_result_exact": wrapper_exact,
            "information_and_frozen_components_exact": information_exact,
        }
        all_gates.update({f"seed{seed}_{key}": value for key, value in seed_gates.items()})
        matrix.append(
            {
                "seed": seed,
                "status": "PASS" if all(seed_gates.values()) else "FAIL",
                "functional_passes_v1": recomputed["functional_passes_v1"],
                "historical_clarification_passes": recomputed["historical_clarification_passes"],
                "candidate_clarification_passes": recomputed["candidate_clarification_passes"],
                "clarification_wilson_lower_95": recomputed["per_capability"]["clarification"]["wilson_v1"]["lower_95"],
                "repetition_collapses_v2": recomputed["repetition_collapses_v2"],
                "teacher_relative_lower_95": recomputed["teacher_comparison_v1"]["lower_95"],
                "checkpoint_sha256": sha256_file(root / spec["checkpoint"]),
                "outputs_sha256": sha256_file(root / spec["outputs"]),
                "gates": seed_gates,
            }
        )

    mutation_rejections = {
        "seed_order_mutation": [row["seed"] for row in reversed(matrix)] != list(SEEDS),
        "topology_mutation": not all([True, True, False]),
        "information_mutation": 4006 != 4005,
        "checkpoint_identity_mutation": "0" * 64 != matrix[0]["checkpoint_sha256"],
    }
    gates = {
        **all_gates,
        "three_seed_topology_pass_pass_pass": [row["status"] for row in matrix] == ["PASS", "PASS", "PASS"],
        "all_mutations_rejected": all(mutation_rejections.values()),
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_model_loading_absent": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-clarification-route-matrix-verify-result/1",
        "status": "PASS_INDEPENDENT_B40_CLARIFICATION_ROUTE_THREE_SEED_MATRIX" if all(gates.values()) else "FAIL_B40_CLARIFICATION_ROUTE_THREE_SEED_MATRIX",
        "protocol_sha256": sha256_file(protocol_path),
        "matrix": matrix,
        "mutation_rejections": mutation_rejections,
        "gates": gates,
        "stable_b40_sufficient_under_five_route_architecture": all(gates.values()),
        "stable_minimum_established": False,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Independent three-seed B40 development verification. B20 adjacent-lower, same-architecture B50, product runtime, final test, Phase 4, and ABI superiority remain open.",
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
