"""Independent replay verifier for the B40 clarification-route hard seed."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import (
    paired_stratified_bootstrap,
    wilson,
)
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-clarification-route-verify/1"
LEGACY_ROUTES = 4
CLARIFICATION_ROUTE = 4


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") == "abi-capability-compiler-phase4-clarification-route-verify-repair/1":
        if (
            protocol.get("status") != "BOUND_TOKENIZER_LOADER_REPAIR"
            or protocol.get("scientific_fields_changed") is not False
            or protocol.get("model_inference_authorized") is not False
            or protocol.get("training_authorized") is not False
            or protocol.get("final_test_access") != "PROHIBITED"
        ):
            raise Phase3Error("clarification-route verifier repair governance changed")
        base_path = (root / protocol["base_protocol"]["path"]).resolve()
        if sha256_file(base_path) != protocol["base_protocol"]["sha256"]:
            raise Phase3Error("clarification-route verifier base protocol changed")
        base = _json(base_path)
        ignored = {
            "abi/capability_compiler_phase4_clarification_route_verify.py",
            "tests/test_capability_compiler_phase4_clarification_route_verify.py",
        }
        for relative, expected in base["bindings"].items():
            if relative in ignored:
                continue
            target = (root / relative).resolve()
            if not target.is_file() or sha256_file(target) != expected:
                raise Phase3Error(f"clarification-route verifier base binding changed: {relative}")
        for relative, expected in protocol["bindings"].items():
            target = (root / relative).resolve()
            if not target.is_file() or sha256_file(target) != expected:
                raise Phase3Error(f"clarification-route verifier repair binding changed: {relative}")
        return base, sha256_file(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_INDEPENDENT_CLARIFICATION_ROUTE_VERIFY"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("clarification-route verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"clarification-route verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def _recompute(
    rows: Sequence[Mapping[str, Any]],
    historical: Mapping[str, Mapping[str, Any]],
    probes: Mapping[str, Mapping[str, Any]],
    teacher: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    semantic_consistency = []
    token_consistency = []
    for row in rows:
        probe = probes[str(row["probe_id"])]
        capability = str(row["capability"])
        value = str(row["output"])
        semantic_consistency.append(
            bool(row["functional_pass_v1"]) == evaluate_functional(value, probe["evaluator"])
            and bool(row["functional_pass_v2"])
            == evaluate_functional_v2(value, probe["evaluator"], capability)
            and bool(row["repetition_collapse_v2"]) == repetition_collapse_v2(value)
        )
        if capability == "clarification":
            token_consistency.append(
                list(row["output_token_ids"])
                == [int(value) for value in tokenizer.encode(value, add_special_tokens=False)]
            )
    per: dict[str, Any] = {}
    for capability in CAPABILITIES:
        selected = [row for row in rows if row["capability"] == capability]
        passed = sum(evaluate_functional(str(row["output"]), probes[str(row["probe_id"])]["evaluator"]) for row in selected)
        per[capability] = {
            "passes_v1": passed,
            "observations": len(selected),
            "wilson_v1": wilson(passed, len(selected)),
            "collapses_v2": sum(repetition_collapse_v2(str(row["output"])) for row in selected),
        }
    paired = [
        {
            "capability": str(row["capability"]),
            "candidate_pass": evaluate_functional(
                str(row["output"]), probes[str(row["probe_id"])]["evaluator"]
            ),
            "teacher_pass": evaluate_functional(
                str(teacher[str(row["probe_id"])]["output"]),
                probes[str(row["probe_id"])]["evaluator"],
            ),
        }
        for row in rows
    ]
    relative = paired_stratified_bootstrap(
        paired,
        replicates=int(protocol["statistics"]["bootstrap_replicates"]),
        seed=int(protocol["statistics"]["bootstrap_seed"]),
    )
    thresholds = protocol["thresholds"]
    nonclarification = [row for row in rows if row["capability"] != "clarification"]
    clarification = [row for row in rows if row["capability"] == "clarification"]
    gates = {
        "exact_1400_rows": len(rows) == 1400 and len({str(row["probe_id"]) for row in rows}) == 1400,
        "stored_semantics_recompute_exact": all(semantic_consistency),
        "clarification_token_accounting_exact": all(token_consistency) and len(token_consistency) == 100,
        "per_capability_functional": all(
            value["wilson_v1"]["point"] >= float(thresholds["per_capability_point"])
            and value["wilson_v1"]["lower_95"] >= float(thresholds["per_capability_lower"])
            for value in per.values()
        ),
        "critical_capabilities": all(
            per[name]["wilson_v1"]["point"] >= float(thresholds["critical_point"])
            and per[name]["wilson_v1"]["lower_95"] >= float(thresholds["critical_lower"])
            for name in protocol["critical_capabilities"]
        ),
        "zero_repetition_collapse": sum(value["collapses_v2"] for value in per.values()) == 0,
        "teacher_noninferior": relative["lower_95"] >= float(thresholds["teacher_relative_lower"]),
        "router_exact": sum(bool(row["capability_route_correct"]) for row in rows) == 1400,
        "all_nonclarification_rows_exact": (
            len(nonclarification) == 1300
            and all(dict(row) == historical[str(row["probe_id"])] for row in nonclarification)
        ),
        "new_route_only_on_clarification": (
            len(clarification) == 100
            and all(
                row.get("fifth_clarification_route_active") is True
                and int(row["control_residual_route"]) == CLARIFICATION_ROUTE
                and int(row["physical_residual_route"]) == CLARIFICATION_ROUTE
                and int(row["active_residual_routes"]) == 1
                for row in clarification
            )
        ),
    }
    return {
        "per_capability": per,
        "teacher_comparison_v1": relative,
        "functional_passes_v1": sum(value["passes_v1"] for value in per.values()),
        "repetition_collapses_v2": sum(value["collapses_v2"] for value in per.values()),
        "router_correct": sum(bool(row["capability_route_correct"]) for row in rows),
        "historical_clarification_passes": sum(
            bool(row["functional_pass_v1"])
            for row in historical.values()
            if row["capability"] == "clarification"
        ),
        "candidate_clarification_passes": per["clarification"]["passes_v1"],
        "gates": gates,
    }


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable clarification-route verification exists: {output}")
    rows = _rows(root / protocol["candidate_outputs"])
    historical_rows = _rows(root / protocol["historical_outputs"])
    historical = {str(row["probe_id"]): row for row in historical_rows}
    probes_list = development_probes(root / protocol["development"]["catalog"])
    probes = {str(row["probe_id"]): row for row in probes_list}
    teacher = {
        str(row["probe_id"]): row
        for row in _rows(root / protocol["development"]["teacher_reference"])
    }
    lineage_protocol = _json(root / protocol["lineage_protocol"])
    v440 = _json(root / lineage_protocol["base_protocols"]["v443"])
    tokenizer = AutoTokenizer.from_pretrained(
        (root / v440["host"]["parent_path"]).resolve(),
        local_files_only=True,
    )
    recomputed = _recompute(rows, historical, probes, teacher, tokenizer, protocol)
    declared = _json(root / protocol["candidate_result"])
    state = load_file(str(root / protocol["candidate_checkpoint"]), device="cpu")
    inherited = load_file(str(root / protocol["inherited_checkpoint"]), device="cpu")
    checkpoint_gates = {
        "checkpoint_schema_exact": set(state) == {"norm.weight", "norm.bias", "down", "up"},
        "checkpoint_geometry_exact": (
            tuple(state["norm.weight"].shape) == (768,)
            and tuple(state["norm.bias"].shape) == (768,)
            and tuple(state["down"].shape) == (5, 16, 768)
            and tuple(state["up"].shape) == (5, 768, 16)
        ),
        "inherited_normalization_exact": torch.equal(state["norm.weight"], inherited["norm.weight"])
        and torch.equal(state["norm.bias"], inherited["norm.bias"]),
        "inherited_four_routes_exact": torch.equal(state["down"][:LEGACY_ROUTES], inherited["down"])
        and torch.equal(state["up"][:LEGACY_ROUTES], inherited["up"]),
    }
    aggregate_gates = {
        "declared_functional_passes_exact": int(declared["functional_passes_v1"]) == recomputed["functional_passes_v1"],
        "declared_collapses_exact": int(declared["repetition_collapses_v2"]) == recomputed["repetition_collapses_v2"],
        "declared_router_exact": int(declared["router_correct"]) == recomputed["router_correct"],
        "declared_per_capability_exact": declared["per_capability"] == recomputed["per_capability"],
        "declared_teacher_comparison_exact": declared["teacher_comparison_v1"] == recomputed["teacher_comparison_v1"],
        "declared_historical_clarification_exact": int(declared["historical_clarification_passes"])
        == recomputed["historical_clarification_passes"],
        "declared_candidate_clarification_exact": int(declared["candidate_clarification_passes"])
        == recomputed["candidate_clarification_passes"],
        "declared_original_gates_all_pass": all(bool(value) for value in declared["gates"].values()),
    }

    wrong_nonclarification = copy.deepcopy(rows)
    first_nonclarification = next(index for index, row in enumerate(wrong_nonclarification) if row["capability"] != "clarification")
    wrong_nonclarification[first_nonclarification]["output"] += " mutation"
    wrong_route = copy.deepcopy(rows)
    first_clarification = next(index for index, row in enumerate(wrong_route) if row["capability"] == "clarification")
    wrong_route[first_clarification]["control_residual_route"] = 3
    wrong_semantics = copy.deepcopy(rows)
    wrong_semantics[first_clarification]["functional_pass_v1"] = not bool(
        wrong_semantics[first_clarification]["functional_pass_v1"]
    )
    wrong_tokens = copy.deepcopy(rows)
    wrong_tokens[first_clarification]["output_token_ids"] = list(wrong_tokens[first_clarification]["output_token_ids"]) + [0]
    mutated_legacy = {key: value.clone() for key, value in state.items()}
    mutated_legacy["down"][0, 0, 0] += 1
    mutation_rejections = {
        "nonclarification_output_mutation": not _recompute(
            wrong_nonclarification, historical, probes, teacher, tokenizer, protocol
        )["gates"]["all_nonclarification_rows_exact"],
        "clarification_route_mutation": not _recompute(
            wrong_route, historical, probes, teacher, tokenizer, protocol
        )["gates"]["new_route_only_on_clarification"],
        "stored_semantics_mutation": not _recompute(
            wrong_semantics, historical, probes, teacher, tokenizer, protocol
        )["gates"]["stored_semantics_recompute_exact"],
        "token_accounting_mutation": not _recompute(
            wrong_tokens, historical, probes, teacher, tokenizer, protocol
        )["gates"]["clarification_token_accounting_exact"],
        "inherited_tensor_mutation": not torch.equal(mutated_legacy["down"][:LEGACY_ROUTES], inherited["down"]),
        "declared_aggregate_mutation": int(declared["functional_passes_v1"]) + 1 != recomputed["functional_passes_v1"],
    }
    gates = {
        **recomputed["gates"],
        **checkpoint_gates,
        **aggregate_gates,
        "all_mutations_rejected": all(mutation_rejections.values()),
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_model_loading_absent": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-clarification-route-verify-result/1",
        "status": "PASS_INDEPENDENT_CLARIFICATION_ROUTE_VERIFICATION" if all(gates.values()) else "FAIL_CLARIFICATION_ROUTE_VERIFICATION",
        "protocol_sha256": protocol_sha,
        "recomputed": recomputed,
        "checkpoint_gates": checkpoint_gates,
        "aggregate_gates": aggregate_gates,
        "mutation_rejections": mutation_rejections,
        "gates": gates,
        "verified_rows": len(rows),
        "verified_clarification_rows": sum(row["capability"] == "clarification" for row in rows),
        "candidate_checkpoint_sha256": sha256_file(root / protocol["candidate_checkpoint"]),
        "candidate_outputs_sha256": sha256_file(root / protocol["candidate_outputs"]),
        "candidate_result_sha256": sha256_file(root / protocol["candidate_result"]),
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Independent hard-seed evidence verification only. No replication, stable minimum, product runtime, final test, Phase 4 certificate, or ABI-superiority claim.",
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
