"""Read-only reconciliation of Phase 4 frontier semantics and baseline readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-contract-baseline-readiness/2"


def frontier_semantics(
    campaign_exit: str,
    strict_mixed_rule: str,
    phase0_exit_requirements: dict[str, Any],
) -> dict[str, bool]:
    lower_campaign = campaign_exit.casefold()
    lower_strict = strict_mixed_rule.casefold()
    return {
        "phase0_freezes_mandatory_specs": all(
            phase0_exit_requirements.get(name) is True
            for name in (
                "mandatory_system_specs_frozen",
                "data_boundaries_frozen",
                "numeric_gates_frozen",
                "statistics_frozen",
                "accounting_frozen",
                "stop_rules_frozen",
            )
        ),
        "campaign_requires_smallest_passing_tested_budget": "smallest passing tested budget"
        in lower_campaign,
        "campaign_requires_adjacent_lower_failure": "adjacent lower failure"
        in lower_campaign,
        "campaign_requires_three_seeds": "three seeds" in lower_campaign,
        "later_rule_explicitly_treats_mixed_as_no_minimum": "mixed" in lower_strict
        and "no stable minimum" in lower_strict,
        "later_rule_is_stricter_than_campaign_text": "all three b40 seeds fail"
        in lower_strict,
    }


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_PHASE4_CONTRACT_BASELINE_AUDIT"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("Phase 4 contract/baseline audit governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 4 contract/baseline binding changed: {relative}")
    return protocol, sha256_file(path)


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable Phase 4 readiness result exists: {output}")
    phase0 = _json(root / protocol["phase0_protocol"])
    campaign = _json(root / protocol["campaign_contract"])
    b40_protocol = _json(root / protocol["strict_boundary_protocol"])
    b40_verified = _json(root / protocol["b40_verified"])
    attribution = _json(root / protocol["b40_attribution"])
    b50 = _json(root / protocol["b50_verified"])
    baseline_protocol = _json(root / protocol["phase2_protocol"])
    baseline_evidence = _json(root / protocol["phase2_machine_evidence"])

    semantics = frontier_semantics(
        str(campaign["phases"][4]["exit"]),
        str(b40_protocol["decision_rule"]),
        phase0["phase0_exit_requirements"],
    )
    headline = baseline_evidence["headline"]
    mandatory = ("L0", "L1", "D0", "D1", "D2")
    baseline_inventory = {}
    for system in mandatory:
        baseline_inventory[system] = {
            "three_seeds": headline[system]["seeds"] == [104729, 130363, 155921],
            "three_checkpoints": len(headline[system]["checkpoint_sha256"]) == 3,
            "functional_passes": headline[system]["functional_passes"],
            "repetition_collapses": headline[system]["repetition_collapses"],
            "zero_collapse_all_seeds": all(
                int(value) == 0 for value in headline[system]["repetition_collapses"]
            ),
            "phase2_pack_sha256": baseline_evidence["pack"]["sha256"],
        }

    fairness = {
        "equal_imported_information": {
            "ready": False,
            "reason": (
                "Phase 2 baselines used the V1 Phase 1-only pack, while B50 uses a "
                "different deterministic union of Phase 1, targeted, and host-conformance "
                "records. Exact prompt IDs, sequence bytes, tokens, and channel hashes "
                "have not been matched."
            ),
            "required_systems": ["L0", "L1", "D0"],
            "richer_channel_controls": ["D1", "D2"],
        },
        "equal_final_deployment_constraint": {
            "ready": True,
            "reason": (
                "The frozen accounting contracts already count the Phi-3 base and all "
                "LoRA adapters, the complete distilled student, and the complete B50 cake."
            ),
        },
        "matched_quality_frontier": {
            "ready": False,
            "reason": (
                "No Phase 2 headline baseline clears zero-collapse across all three seeds, "
                "and no exact-B50 baseline frontier has been trained."
            ),
        },
    }
    gates = {
        "contract_semantics_recomputed": all(semantics.values()),
        "verified_b50_stable": b50["stable_sufficient_b50"] is True,
        "verified_b40_mixed": [row["machine_gates"] for row in b40_verified["matrix"]]
        == ["PASS", "PASS", "FAIL"],
        "failure_owned_by_abi_not_layercake": attribution["measured_owner"].startswith(
            "ABI acquisition/optimization"
        ),
        "phase2_machine_evidence_complete": baseline_evidence[
            "machine_evidence_complete"
        ]
        is True,
        "phase2_human_gate_still_open": baseline_evidence["human_ratings"]["complete"]
        is False,
        "mandatory_baselines_present": set(headline) == set(mandatory),
        "all_baselines_three_seed": all(
            value["three_seeds"] and value["three_checkpoints"]
            for value in baseline_inventory.values()
        ),
        "no_baseline_zero_collapse_all_seeds": not any(
            value["zero_collapse_all_seeds"] for value in baseline_inventory.values()
        ),
        "phase2_and_b50_information_not_exact": baseline_evidence["pack"][
            "response_tokens"
        ]
        != b50["imported_information"]["authoritative_teacher_output_tokens"],
        "all_three_fairness_views_named": list(phase0["fairness_views"])
        == campaign["fairness_views"],
        "training_absent": True,
        "teacher_loading_absent": True,
        "final_test_not_accessed": True,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase4-contract-baseline-readiness-result/2",
        "status": "PASS_PHASE4_CONTRACT_RECONCILED_BASELINE_RETRAINING_REQUIRED"
        if passed
        else "FAIL_PHASE4_CONTRACT_BASELINE_READINESS_AUDIT",
        "protocol_sha256": protocol_sha,
        "frontier_semantics": semantics,
        "governance_decision": (
            "Preserve the stricter V801 no-minimum result. It cannot be weakened after "
            "observing B40. Use B50 only as the stable sufficient operating point for a "
            "prospective fairness campaign; do not label it a minimum."
        ),
        "baseline_inventory": baseline_inventory,
        "phase2_baseline_information": {
            "pack_sha256": baseline_evidence["pack"]["sha256"],
            "response_tokens": baseline_evidence["pack"]["response_tokens"],
            "phase1_teacher_output_tokens": baseline_protocol["phase1"][
                "selected_teacher_output_tokens"
            ],
        },
        "b50_information": b50["imported_information"],
        "fairness_views": fairness,
        "minimum_fair_next_campaign": {
            "operating_point": "B50_STABLE_SUFFICIENT_NOT_MINIMUM",
            "exact_information_methods": ["L0", "L1", "D0"],
            "richer_information_controls": ["D1", "D2"],
            "seeds": [104729, 130363, 155921],
            "selection_rule": (
                "Recreate the frozen Phase 2 development grids on seed104729 using only "
                "the exact B50 record/channel manifest, freeze one headline configuration "
                "per method, then replay it on all three paired seeds."
            ),
            "quality_rule": (
                "Use the unchanged 1,400-row suite, Wilson and paired bootstrap gates, "
                "zero-collapse rule, and teacher-relative threshold. Preserve every grid result."
            ),
            "deployment_rule": (
                "Count the entire source base, tokenizer, router, adapters, distilled student, "
                "and B50 package. No retained source or imported channel is free."
            ),
            "final_test": "PROHIBITED",
        },
        "gates": gates,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "stable_minimum_established": False,
        "claim_boundary": (
            "Read-only governance and readiness audit. It authorizes no claim that B50 is "
            "minimal and supplies no matched-frontier, final-test, Phase 4, or ABI-superiority result."
        ),
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
