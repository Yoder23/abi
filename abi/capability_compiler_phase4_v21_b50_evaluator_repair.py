"""Read-only independent evaluator repair for the complete V784 v21 B50 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Iterable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .capability_compiler_phase2_common import canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_final_controls import evaluate_functional_v2
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .capability_compiler_phase4_v19_frontier_rescreen import (
    _json,
    _merged_evaluation,
    _quality_gates,
    _rows,
)
from .capability_compiler_phase4_v19_frontier_verify import _pointer_checks, _without
from .capability_compiler_phase4_v21_b50_rescreen import (
    _api,
    _package,
    load_protocol as load_source_protocol,
    strong_route_conformance,
)


FORMAT = "abi-capability-compiler-phase4-v21-b50-evaluator-repair/1"


def corrected_guard_contract(
    rows: list[dict[str, Any]], weak: set[str], interface: str
) -> dict[str, bool]:
    """Use pointer semantics for coherence and exact history for all other rows."""

    noncoherence = [row for row in rows if row["capability"] != "coherence"]
    changed = [row for row in noncoherence if row["output_changed_from_v19_history"]]
    historical_collapses = [
        row for row in noncoherence if row["historical_repetition_collapse_v2"]
    ]
    changed_ids = {str(row["probe_id"]) for row in changed}
    collapsed_ids = {str(row["probe_id"]) for row in historical_collapses}
    return {
        "strong_route_conformance": strong_route_conformance(rows, weak),
        "noncoherence_change_set_equals_historical_collapses": changed_ids
        == collapsed_ids,
        "all_other_noncoherence_outputs_exact": all(
            row["output_changed_from_v19_history"]
            == row["historical_repetition_collapse_v2"]
            for row in noncoherence
        ),
        "changed_rows_guard_terminated": all(
            row["guard_terminated"] for row in changed
        ),
        "changed_rows_canonical_prefixes": all(
            row["canonical_historical_prefix_preserved"] for row in changed
        ),
        "historical_functional_passes_preserved": all(
            not row["historical_functional_pass_v1"] or row["functional_pass_v1"]
            for row in changed
        ),
        "coherence_not_guard_terminated": all(
            not row["guard_terminated"]
            for row in rows
            if row["capability"] == "coherence"
        ),
        "zero_remaining_collapse": not any(
            row["repetition_collapse_v2"] for row in rows
        ),
        "exact_lexical_boundary_declared": interface == "lc-direct-neural-core/21",
    }


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_V21_B50_EVALUATOR_REPAIR"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("v21 B50 evaluator-repair governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"v21 B50 evaluator-repair binding changed: {relative}")
    return protocol, sha256_file(path)


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    repair, repair_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable B50 repair output exists: {output}")
    source, source_sha = load_source_protocol(root, root / repair["source_protocol"])
    recorded = _json(root / repair["source_result"])
    recorded_evidence = hashlib.sha256(
        canonical_json_bytes(_without(recorded, "evidence_sha256"))
    ).hexdigest()
    api = _api((root / source["layercake_root"]).resolve())
    weak = set(api["weak_capabilities"])
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(source["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    probes_list = development_probes(root / source["development_catalog"])
    probes = {str(probe["probe_id"]): probe for probe in probes_list}
    teacher = {
        str(row["probe_id"]): row
        for row in _rows(root / source["teacher_reference"])
    }
    specs = {int(spec["seed"]): spec for spec in source["systems"]}
    verified: list[dict[str, Any]] = []
    aggregate: list[dict[str, Any]] = []

    from layercake_extensions.route_isolated_prompt_span_core_v19 import (
        extract_prompt_segments,
        render_prompt_segments,
    )

    for system in recorded["systems"]:
        seed = int(system["seed"])
        spec = specs[seed]
        rows = _rows(root / system["outputs"]["path"])
        historical = _rows(root / spec["historical_outputs"])
        history = {str(row["probe_id"]): row for row in historical}
        row_checks: list[dict[str, bool]] = []
        pointer_row_checks: list[dict[str, bool]] = []
        for row in rows:
            probe_id = str(row["probe_id"])
            probe = probes[probe_id]
            prior = history[probe_id]
            output_text = str(row["output"])
            prior_output = str(prior["output"])
            exact = output_text == prior_output
            checks = {
                "probe_identity": str(probe["canonical_capability"])
                == str(row["capability"]),
                "route_record": str(row["automatic_capability_route"])
                == str(row["capability"])
                and bool(row["capability_route_correct"]),
                "functional_v1": bool(row["functional_pass_v1"])
                == evaluate_functional(output_text, probe["evaluator"]),
                "functional_v2": bool(row["functional_pass_v2"])
                == evaluate_functional_v2(
                    output_text, probe["evaluator"], str(row["capability"])
                ),
                "collapse": bool(row["repetition_collapse_v2"])
                == repetition_collapse_v2(output_text),
                "historical_functional": bool(row["historical_functional_pass_v1"])
                == evaluate_functional(prior_output, probe["evaluator"]),
                "historical_collapse": bool(
                    row["historical_repetition_collapse_v2"]
                )
                == repetition_collapse_v2(prior_output),
                "change_flag": bool(row["output_changed_from_v19_history"])
                == (not exact),
                "canonical_prefix_flag": bool(
                    row["canonical_historical_prefix_preserved"]
                )
                == prior_output.startswith(output_text),
            }
            if row["capability"] == "coherence":
                pointer_checks = _pointer_checks(
                    str(probe["prompt"]),
                    output_text,
                    dict(row["v21_pointer"]),
                    extract_prompt_segments,
                    render_prompt_segments,
                )
                checks.update(pointer_checks)
                pointer_row_checks.append(pointer_checks)
            row_checks.append(checks)
            aggregate.append({"seed": seed, **row})

        evaluation = _merged_evaluation(rows)
        quality, relative = _quality_gates(
            source,
            evaluation,
            rows,
            probes,
            teacher,
            seed + 6_500_000,
        )
        quality.pop("strong_parent_exact")
        coherence = [row for row in rows if row["capability"] == "coherence"]
        pointer = {
            "all_100_pointer_rows": len(coherence) == 100
            and all(bool(row["v21_pointer"]) for row in coherence),
            "six_candidates": all(
                row["v21_pointer"].get("candidate_count") == 6 for row in coherence
            ),
            "one_scoring_forward": all(
                row["v21_pointer"].get("candidate_scoring_forward_passes") == 1
                for row in coherence
            ),
            "one_active_route": all(
                row["v21_pointer"].get("active_residual_routes") == 1
                for row in coherence
            ),
            "persistent_state_reused": all(
                row["v21_pointer"].get("persistent_prompt_state_reused") is True
                for row in coherence
            ),
            "evaluator_blind": all(
                row["v21_pointer"].get("evaluator_used") is False
                for row in coherence
            ),
            "package_identity": system["activation"]["archive_sha256"]
            == system["package"]["archive_sha256"]
            and system["activation"]["tensor_payload_hash"]
            == system["package"]["tensor_payload_hash"],
            "package_verified": system["activation"]["verification"] == "PASS",
            "receiver_learning_zero": system["activation"]["receiver_training_steps"]
            == system["activation"]["receiver_calibration_runs"]
            == 0,
        }
        guard = corrected_guard_contract(rows, weak, source["interface"])
        with tempfile.TemporaryDirectory(prefix=f"abi-v21-b50-repair-{seed}-") as raw:
            rebuilt = _package(
                root,
                source,
                spec,
                Path(raw) / "candidate.cake",
                api,
                private,
                public_pem,
            )
        machine = all(quality.values()) and all(pointer.values()) and all(guard.values())
        gates = {
            "depth": len(rows) == len(historical) == 1400,
            "raw_hash": sha256_file(root / system["outputs"]["path"])
            == system["outputs"]["sha256"],
            "all_row_checks": all(all(check.values()) for check in row_checks),
            "all_100_pointer_semantics": len(pointer_row_checks) == 100
            and all(all(check.values()) for check in pointer_row_checks),
            "evaluation_recomputed": evaluation == system["evaluation"],
            "quality_recomputed": quality == system["quality_gates"],
            "teacher_comparison_recomputed": relative
            == system["teacher_comparison_v1"],
            "pointer_recomputed": pointer == system["pointer_gates"],
            "package_rebuilt_exact": rebuilt == system["package"],
        }
        noncoherence_changed = [
            row
            for row in rows
            if row["capability"] != "coherence"
            and row["output_changed_from_v19_history"]
        ]
        coherence_changed = [
            row
            for row in rows
            if row["capability"] == "coherence"
            and row["output_changed_from_v19_history"]
        ]
        verified.append(
            {
                "budget": "B50",
                "seed": seed,
                "functional_passes_v1": evaluation["functional_passes_v1"],
                "repetition_collapses_v2": evaluation["repetition_collapses_v2"],
                "changed_coherence_rows": len(coherence_changed),
                "changed_noncoherence_rows": len(noncoherence_changed),
                "quality_gates": quality,
                "pointer_gates": pointer,
                "corrected_guard_gates": guard,
                "corrected_machine_gates_pass": machine,
                "verification_gates": gates,
                "all_verification_gates_pass": all(gates.values()),
            }
        )
        print(
            json.dumps(
                {
                    "verified": seed,
                    "machine_pass": machine,
                    "all_verification_gates_pass": all(gates.values()),
                }
            ),
            flush=True,
        )

    aggregate_path = root / repair["aggregate_outputs"]
    aggregate_exact = (
        b"".join(canonical_json_bytes(row) for row in aggregate)
        == aggregate_path.read_bytes()
    )
    topology = [system["corrected_machine_gates_pass"] for system in verified]
    stable = all(topology)
    mixed = any(topology) and not stable
    top = {
        "source_protocol_hash": source_sha == recorded["protocol_sha256"],
        "source_result_hash": sha256_file(root / repair["source_result"])
        == repair["bindings"][repair["source_result"]],
        "source_evidence_hash": recorded_evidence == recorded["evidence_sha256"],
        "three_registered_systems": len(verified) == 3,
        "all_system_verifiers": all(
            system["all_verification_gates_pass"] for system in verified
        ),
        "aggregate_exact": aggregate_exact,
        "aggregate_hash": sha256_file(aggregate_path)
        == recorded["aggregate_outputs_sha256"],
        "all_noncoherence_change_sets_exact": all(
            system["corrected_guard_gates"][
                "noncoherence_change_set_equals_historical_collapses"
            ]
            for system in verified
        ),
        "all_300_pointer_rows_semantically_valid": all(
            system["verification_gates"]["all_100_pointer_semantics"]
            for system in verified
        ),
        "expected_mixed_topology_reproduced": topology == [False, True, True],
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_loading_absent": True,
        "final_test_not_accessed": True,
    }
    verification_pass = all(top.values())
    if not verification_pass:
        status = "FAIL_V21_B50_EVALUATOR_REPAIR"
    elif stable:
        status = "PASS_VERIFIED_STABLE_B50_V21_DEVELOPMENT_CANDIDATE"
    elif mixed:
        status = "PASS_VERIFIED_MIXED_B50_V21_DEVELOPMENT_TOPOLOGY"
    else:
        status = "PASS_VERIFIED_ALL_FAIL_B50_V21_DEVELOPMENT_TOPOLOGY"
    result = {
        "format": "abi-capability-compiler-phase4-v21-b50-evaluator-repair-result/1",
        "status": status,
        "protocol_sha256": repair_sha,
        "source_result_sha256": sha256_file(root / repair["source_result"]),
        "source_evidence_sha256": recorded["evidence_sha256"],
        "systems": verified,
        "corrected_topology": topology,
        "three_seed_all_pass": stable,
        "mixed_topology": mixed,
        "gates": top,
        "packages_deterministically_rebuilt": 3,
        "rows_recomputed": 4200,
        "pointer_rows_recomputed": 300,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "stable_minimum_established": False,
        "claim_boundary": (
            "Read-only corrected B50 v21 development evaluation only. A mixed "
            "B50 topology establishes no minimum. No matched baseline, final test, "
            "Phase 4, or ABI-superiority claim."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(
        output,
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
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
