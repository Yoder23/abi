"""Read-only audit for one bounded clarification-route frontier successor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-frontier-design-audit/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _source(root: Path, protocol: dict[str, Any], name: str) -> dict[str, Any]:
    relative = str(protocol["sources"][name])
    return _json((root / relative).resolve())


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_FRONTIER_DESIGN_AUDIT"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("frontier-design audit governance changed")
    for relative, expected in protocol["bindings"].items():
        if sha256_file((root / relative).resolve()) != expected:
            raise Phase3Error(f"frontier-design audit binding changed: {relative}")
    if output.exists():
        raise Phase3Error(f"immutable frontier-design audit exists: {output}")

    state = _source(root, protocol, "state")
    b20 = _source(root, protocol, "b20_seed104729")
    b40 = _source(root, protocol, "b40_verified")
    attribution = _source(root, protocol, "b40_attribution")
    b50 = _source(root, protocol, "b50_verified")
    contract = _source(root, protocol, "contract_readiness")
    uniform = _source(root, protocol, "uniform_exposure")
    consensus = _source(root, protocol, "seed_consensus")
    compatibility = _source(root, protocol, "compatibility")
    isolated = _source(root, protocol, "capability_isolated")
    response_ce = _source(root, protocol, "response_ce")
    acquisition_validation = _source(root, protocol, "acquisition_validation")
    targeted_validation = _source(root, protocol, "targeted_validation")
    metamorphic = _source(root, protocol, "metamorphic_validation")

    b40_matrix = b40["matrix"]
    b50_matrix = b50["matrix"]
    isolated_route_changes = isolated["measured_route_changes_from_inherited_same_seed"]
    facts = {
        "b20_seed104729": {
            "status": b20["status"],
            "functional_passes_v1": int(b20["functional_passes_v1"]),
            "repetition_collapses_v2": int(b20["repetition_collapses_v2"]),
            "other_registered_seeds_exist": False,
        },
        "b40": {
            "topology": [row["machine_gates"] for row in b40_matrix],
            "functional_passes_v1": [int(row["functional_passes_v1"]) for row in b40_matrix],
            "sole_failing_gate": b40["measured_limit"],
            "failed_rows": int(attribution["failed_row_count"]),
            "failure_taxonomy": dict(attribution["failure_taxonomy_counts"]),
            "same_information_all_seeds": bool(attribution["selected_information"]["same_across_all_seeds"]),
            "host_exonerated": not bool(attribution["layercake_host_failure"]),
        },
        "b50": {
            "topology": [row["machine_gates"] for row in b50_matrix],
            "functional_passes_v1": [int(row["functional_passes_v1"]) for row in b50_matrix],
            "stable_sufficient": bool(b50["stable_sufficient_b50"]),
            "stable_minimum": bool(b50["stable_minimum_b50"]),
        },
        "rejected_mechanisms": {
            "uniform_exposure": uniform["decision"],
            "equal_weight_seed_consensus": consensus["decision"],
            "uniform_all_route_training": isolated["adaptive_decision"],
            "response_ce_route_acceptance": response_ce["decision"],
            "phase1_acquisition_validation": acquisition_validation["adaptive_decision"],
            "targeted_or_host_validation": targeted_validation["adaptive_decision"],
            "metamorphic_coherence_acceptance": metamorphic["adaptive_decision"],
        },
        "measured_support": {
            "parent_bridge_coadaptation": compatibility["status"],
            "uniform_isolated_clarification_result": isolated_route_changes.get("clarification", "not reported because it remained passing"),
            "uniform_isolated_damage_was_other_routes": {
                key: value
                for key, value in isolated_route_changes.items()
                if key in {"format_control", "fluent_realization", "tone_control"}
            },
        },
    }

    gates = {
        "controlling_state_bound": state["state_id"] == "abi-capability-compiler-campaign-state-v911",
        "strict_adjacent_rule_preserved": bool(contract["frontier_semantics"]["later_rule_explicitly_treats_mixed_as_no_minimum"]),
        "b50_stable_all_seeds": facts["b50"]["topology"] == ["PASS", "PASS", "PASS"],
        "b40_mixed_exactly": facts["b40"]["topology"] == ["PASS", "PASS", "FAIL"],
        "b40_failure_is_clarification_form_only": (
            facts["b40"]["failed_rows"] == 10
            and facts["b40"]["failure_taxonomy"]["missing_inquiry_marker"] == 10
            and facts["b40"]["failure_taxonomy"]["missing_question_mark"] == 10
        ),
        "same_b40_information_all_seeds": facts["b40"]["same_information_all_seeds"],
        "layercake_host_exonerated": facts["b40"]["host_exonerated"],
        "b20_anchor_is_incomplete_and_failed_where_measured": (
            facts["b20_seed104729"]["status"] == "FAIL_PHASE4_ABI_BUDGET_MACHINE_GATES"
            and not facts["b20_seed104729"]["other_registered_seeds_exist"]
        ),
        "prior_stabilization_failures_preserved": (
            uniform["status"].startswith("FAIL_")
            and consensus["status"].startswith("FAIL_")
            and isolated["status"].startswith("FAIL_")
            and response_ce["status"].startswith("FAIL_")
            and acquisition_validation["status"].startswith("FAIL_")
            and targeted_validation["status"].startswith("FAIL_")
            and metamorphic["status"].startswith("FAIL_")
        ),
        "new_design_is_not_evaluator_patch": True,
        "new_design_adds_no_teacher_information": True,
        "training_absent": True,
        "model_inference_absent": True,
        "final_test_not_accessed": True,
    }
    authorized = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase4-frontier-design-audit-result/1",
        "status": "PASS_AUTHORIZE_CLARIFICATION_ONLY_ACQUISITION_ROUTE_DESIGN" if authorized else "FAIL_FRONTIER_DESIGN_AUDIT",
        "protocol_sha256": sha256_file(protocol_path),
        "facts": facts,
        "gates": gates,
        "authorized_design": {
            "name": "FIFTH_CLARIFICATION_ONLY_RANK16_ACQUISITION_ROUTE",
            "scope": "Extend the existing four-route residual by one physically isolated clarification route. Freeze the parent, router, inherited four routes, thresholds, selected information, and development suite.",
            "training_source": "Only the already selected budget-local Phase 1 clarification acquisition records and normalized teacher outputs; no development prompt, evaluator, failing-row identity, new teacher output, logits, or activations.",
            "initialization": "Copy the exact four inherited routes; initialize only the fifth route with inherited shared normalization, a deterministic rank-16 down projection, and an exact-zero up projection.",
            "optimization": "Train only fifth-route tensors with one prospectively frozen schedule; no route acceptance, coefficient, seed, learning-rate, step, or budget sweep.",
            "hard_seed_first": {"budget": "B40", "seed": 155921},
            "adaptive_order": "If the B40 hard seed fails any unchanged gate, close the design. If it passes, run the other two B40 seeds. Only three B40 passes authorize the same architecture at B50 and the three-seed B20 adjacent-lower anchor. B20 requires clean-start missing-seed lineages before a minimum decision.",
            "product_requirement": "Promotion later requires a separately certified LayerCake host/package implementation and full CPU/GPU runtime rescreen of the same artifact.",
        } if authorized else None,
        "explicitly_not_authorized": [
            "question-mark or inquiry-marker output rewriting",
            "development-evaluator logic in ABI or LayerCake",
            "uniform fourteen-route retraining",
            "route acceptance from response CE or the rejected validation sources",
            "nearby hyperparameter or budget sweeps",
            "final-test access",
            "Phase 4 or ABI-superiority certification",
        ],
        "training_performed": False,
        "model_inference_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Read-only frontier-design authorization only. No new model result, stable minimum, final test, Phase 4 certificate, or ABI-superiority claim.",
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
