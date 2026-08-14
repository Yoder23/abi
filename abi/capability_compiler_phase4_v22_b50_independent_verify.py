"""Read-only independent verifier for the complete v22 B50 three-seed screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

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
from .capability_compiler_phase4_v22_b50_rescreen import (
    _api,
    _package,
    load_protocol as load_source_protocol,
)


FORMAT = "abi-capability-compiler-phase4-v22-b50-independent-verify/1"


def independent_strong_route_conformance(
    rows: list[dict[str, Any]], weak: set[str]
) -> bool:
    """Recompute the v22 strong-route contract without calling the screen helper."""

    strong = [row for row in rows if row["capability"] not in weak]
    return bool(strong) and all(
        bool(row["strong_parent_output_exact"])
        or (
            row["capability"] == "format_control"
            and bool(row["v22_format"])
            and bool(row["functional_pass_v1"])
            and not bool(row["repetition_collapse_v2"])
        )
        or (
            bool(row["guard_terminated"])
            and bool(row["strong_parent_prefix_preserved"])
            and (
                not bool(row["historical_functional_pass_v1"])
                or bool(row["functional_pass_v1"])
            )
            and not bool(row["repetition_collapse_v2"])
        )
        for row in strong
    )


def format_semantics(
    row: Mapping[str, Any],
    prompt: str,
    extract: Callable[[str], Any],
    render: Callable[[Any], str],
    expected_mode: str,
) -> dict[str, bool]:
    """Independently validate one declared deterministic format execution."""

    record = dict(row.get("v22_format") or {})
    literals = extract(prompt)
    return {
        "record_present": bool(record),
        "literals_extract": literals is not None,
        "exact_render": literals is not None and str(row["output"]) == render(literals),
        "mode_labeled": record.get("mode") == expected_mode,
        "deterministic_transducer": record.get("deterministic_transducer") is True,
        "one_prefill": record.get("prompt_prefill_forward_passes") == 1,
        "zero_scoring": record.get("candidate_scoring_forward_passes") == 0,
        "zero_decode": record.get("decode_forward_passes") == 0,
        "persistent_state_created": record.get("persistent_prompt_state_created") is True,
        "state_not_advanced": record.get("model_state_advanced_after_prefill") is False,
        "zero_residual": record.get("active_residual_routes") == 0,
        "evaluator_absent": record.get("evaluator_used") is False,
        "teacher_absent": record.get("teacher_used") is False,
    }


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_V22_B50_INDEPENDENT_VERIFIER"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("v22 B50 independent-verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"v22 B50 independent-verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    verifier, verifier_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable v22 B50 verifier output exists: {output}")
    source, source_sha = load_source_protocol(root, root / verifier["source_protocol"])
    recorded = _json(root / verifier["source_result"])
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
        str(row["probe_id"]): row for row in _rows(root / source["teacher_reference"])
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
        pointer_checks_all: list[dict[str, bool]] = []
        format_checks_all: list[dict[str, bool]] = []
        for row in rows:
            probe_id = str(row["probe_id"])
            probe = probes[probe_id]
            prior = history[probe_id]
            output_text = str(row["output"])
            prior_output = str(prior["output"])
            exact = output_text == prior_output
            capability = str(row["capability"])
            checks = {
                "probe_identity": str(probe["canonical_capability"]) == capability,
                "route_record": str(row["automatic_capability_route"]) == capability
                and bool(row["capability_route_correct"]),
                "output_alias": row["original_output"] == row["output"],
                "functional_v1": bool(row["functional_pass_v1"])
                == evaluate_functional(output_text, probe["evaluator"]),
                "functional_v2": bool(row["functional_pass_v2"])
                == evaluate_functional_v2(output_text, probe["evaluator"], capability),
                "collapse": bool(row["repetition_collapse_v2"])
                == repetition_collapse_v2(output_text),
                "historical_functional": bool(row["historical_functional_pass_v1"])
                == evaluate_functional(prior_output, probe["evaluator"]),
                "change_flag": bool(row["output_changed_from_v21_history"])
                == (not exact),
                "strong_exact_flag": capability in weak
                or bool(row["strong_parent_output_exact"]) == exact,
                "strong_prefix_flag": capability in weak
                or bool(row["strong_parent_prefix_preserved"])
                == prior_output.startswith(output_text),
                "abstention_prefix": bool(row["abstention_clause_prefixed"])
                == (
                    capability == "abstention"
                    and output_text.startswith(
                        "I cannot determine that from the information given."
                    )
                ),
            }
            if capability == "coherence":
                pointer_checks = _pointer_checks(
                    str(probe["prompt"]),
                    output_text,
                    dict(row["v22_pointer"]),
                    extract_prompt_segments,
                    render_prompt_segments,
                )
                checks.update(pointer_checks)
                pointer_checks_all.append(pointer_checks)
            elif row.get("v22_pointer"):
                checks["pointer_scope"] = False
            if capability == "format_control":
                format_checks = format_semantics(
                    row,
                    str(probe["prompt"]),
                    api["extract_format"],
                    api["render_format"],
                    api["format_literal_mode"],
                )
                checks.update(format_checks)
                format_checks_all.append(format_checks)
            elif row.get("v22_format"):
                checks["format_scope"] = False
            row_checks.append(checks)
            aggregate.append({"seed": seed, **row})

        evaluation = _merged_evaluation(rows)
        quality, relative = _quality_gates(
            source,
            evaluation,
            rows,
            probes,
            teacher,
            seed + 7_000_000,
        )
        quality.pop("strong_parent_exact")
        coherence = [row for row in rows if row["capability"] == "coherence"]
        formats = [row for row in rows if row["capability"] == "format_control"]
        changed = [row for row in rows if row["output_changed_from_v21_history"]]
        pointer = {
            "all_100_pointer_rows": len(coherence) == 100
            and all(bool(row["v22_pointer"]) for row in coherence),
            "six_candidates": all(
                row["v22_pointer"].get("candidate_count") == 6 for row in coherence
            ),
            "one_scoring_forward": all(
                row["v22_pointer"].get("candidate_scoring_forward_passes") == 1
                for row in coherence
            ),
            "one_active_route": all(
                row["v22_pointer"].get("active_residual_routes") == 1
                for row in coherence
            ),
            "persistent_state_reused": all(
                row["v22_pointer"].get("persistent_prompt_state_reused") is True
                for row in coherence
            ),
            "evaluator_blind": all(
                row["v22_pointer"].get("evaluator_used") is False for row in coherence
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
        format_gates = {
            "all_100_format_rows": len(formats) == 100
            and all(bool(row["v22_format"]) for row in formats),
            "exact_prompt_literals": all(
                checks["literals_extract"] and checks["exact_render"]
                for checks in format_checks_all
            ),
            "deterministic_transducer_labeled": all(
                checks["mode_labeled"] and checks["deterministic_transducer"]
                for checks in format_checks_all
            ),
            "one_prefill_zero_scoring_zero_decode": all(
                checks["one_prefill"]
                and checks["zero_scoring"]
                and checks["zero_decode"]
                and checks["persistent_state_created"]
                and checks["state_not_advanced"]
                for checks in format_checks_all
            ),
            "strong_path_zero_residual": all(
                checks["zero_residual"] for checks in format_checks_all
            ),
            "evaluator_and_teacher_absent": all(
                checks["evaluator_absent"] and checks["teacher_absent"]
                for checks in format_checks_all
            ),
            "all_format_rows_functional": all(
                bool(row["functional_pass_v1"]) for row in formats
            ),
        }
        preservation = {
            "strong_route_conformance": independent_strong_route_conformance(rows, weak),
            "changed_rows_format_only": all(
                row["capability"] == "format_control" for row in changed
            ),
            "all_nonformat_outputs_exact": all(
                not row["output_changed_from_v21_history"]
                for row in rows
                if row["capability"] != "format_control"
            ),
            "historical_format_passes_preserved": all(
                not row["historical_functional_pass_v1"] or row["functional_pass_v1"]
                for row in formats
            ),
            "zero_remaining_collapse": evaluation["repetition_collapses_v2"] == 0,
            "interface_v22_declared": source["interface"] == "lc-direct-neural-core/22",
        }
        with tempfile.TemporaryDirectory(prefix=f"abi-v22-b50-verify-{seed}-") as raw:
            rebuilt = _package(
                root,
                source,
                spec,
                Path(raw) / "candidate.cake",
                api,
                private,
                public_pem,
            )
        machine = (
            all(quality.values())
            and all(pointer.values())
            and all(format_gates.values())
            and all(preservation.values())
        )
        gates = {
            "depth": len(rows) == len(historical) == 1400,
            "raw_hash": sha256_file(root / system["outputs"]["path"])
            == system["outputs"]["sha256"],
            "all_row_checks": all(all(check.values()) for check in row_checks),
            "all_100_pointer_semantics": len(pointer_checks_all) == 100
            and all(all(check.values()) for check in pointer_checks_all),
            "all_100_format_semantics": len(format_checks_all) == 100
            and all(all(check.values()) for check in format_checks_all),
            "evaluation_recomputed": evaluation == system["evaluation"],
            "quality_recomputed": quality == system["quality_gates"],
            "teacher_comparison_recomputed": relative
            == system["teacher_comparison_v1"],
            "pointer_recomputed": pointer == system["pointer_gates"],
            "format_gates_recomputed": format_gates == system["format_gates"],
            "preservation_recomputed": preservation == system["preservation_gates"],
            "changed_count_recomputed": len(changed) == system["changed_rows"],
            "package_rebuilt_exact": rebuilt == system["package"],
            "recorded_machine_verdict_reproduced": machine
            == system["machine_gates_pass"],
        }
        verified.append(
            {
                "budget": "B50",
                "seed": seed,
                "functional_passes_v1": evaluation["functional_passes_v1"],
                "format_control_passes_v1": evaluation["per_capability"][
                    "format_control"
                ]["passes_v1"],
                "repetition_collapses_v2": evaluation["repetition_collapses_v2"],
                "changed_rows": len(changed),
                "quality_gates": quality,
                "pointer_gates": pointer,
                "format_gates": format_gates,
                "preservation_gates": preservation,
                "machine_gates_pass": machine,
                "verification_gates": gates,
                "all_verification_gates_pass": all(gates.values()),
            }
        )
        print(
            json.dumps(
                {
                    "verified": seed,
                    "machine_pass": machine,
                    "verification_pass": all(gates.values()),
                }
            ),
            flush=True,
        )

    aggregate_path = root / verifier["aggregate_outputs"]
    aggregate_exact = (
        b"".join(canonical_json_bytes(row) for row in aggregate)
        == aggregate_path.read_bytes()
    )
    stable = all(system["machine_gates_pass"] for system in verified)
    top = {
        "source_protocol_hash": source_sha == recorded["protocol_sha256"],
        "source_result_hash": sha256_file(root / verifier["source_result"])
        == verifier["bindings"][verifier["source_result"]],
        "source_evidence_hash": recorded_evidence == recorded["evidence_sha256"],
        "three_registered_systems": [system["seed"] for system in verified]
        == [104729, 130363, 155921],
        "all_system_verifiers": all(
            system["all_verification_gates_pass"] for system in verified
        ),
        "aggregate_exact": aggregate_exact,
        "aggregate_hash": sha256_file(aggregate_path)
        == recorded["aggregate_outputs_sha256"],
        "all_300_pointer_rows_semantically_valid": all(
            system["verification_gates"]["all_100_pointer_semantics"]
            for system in verified
        ),
        "all_300_format_rows_semantically_valid": all(
            system["verification_gates"]["all_100_format_semantics"]
            for system in verified
        ),
        "all_3900_nonformat_rows_exact": all(
            system["preservation_gates"]["all_nonformat_outputs_exact"]
            for system in verified
        ),
        "stable_topology_reproduced": stable and recorded["three_seed_all_pass"] is True,
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_loading_absent": True,
        "final_test_not_accessed": True,
    }
    verification_pass = all(top.values())
    status = (
        "PASS_INDEPENDENTLY_VERIFIED_STABLE_B50_V22_DEVELOPMENT_CANDIDATE"
        if verification_pass and stable
        else "FAIL_V22_B50_INDEPENDENT_VERIFICATION"
    )
    result = {
        "format": "abi-capability-compiler-phase4-v22-b50-independent-verify-result/1",
        "status": status,
        "protocol_sha256": verifier_sha,
        "source_result_sha256": sha256_file(root / verifier["source_result"]),
        "source_evidence_sha256": recorded["evidence_sha256"],
        "systems": verified,
        "three_seed_all_pass": stable,
        "gates": top,
        "packages_deterministically_rebuilt": 3,
        "rows_recomputed": 4200,
        "pointer_rows_recomputed": 300,
        "format_rows_recomputed": 300,
        "nonformat_rows_compared_exactly": 3900,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "stable_minimum_established": False,
        "claim_boundary": (
            "Independently verified three-seed B50 v22 development candidate only. "
            "The format path is a deterministic prompt-literal transducer, not broad "
            "neural generation. B50 is sufficient, not a minimum. No matched baseline, "
            "final test, Phase 4, or ABI-superiority claim."
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
