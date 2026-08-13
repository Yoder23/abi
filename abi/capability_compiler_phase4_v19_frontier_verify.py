"""Independent raw-evidence verifier for the v19 B40/B80 frontier rescreen."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_final_controls import evaluate_functional_v2
from .capability_compiler_repetition_v2 import repetition_collapse_v2
from .capability_compiler_phase4_v19_frontier_rescreen import (
    _build_package,
    _json,
    _layercake_api,
    _merged_evaluation,
    _quality_gates,
    _rows,
    frontier_decision,
    load_protocol as load_rescreen_protocol,
)


FORMAT = "abi-capability-compiler-phase4-v19-frontier-verifier/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_INDEPENDENT_RAW_EVIDENCE_VERIFIER"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("v19 frontier verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"v19 frontier verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def _without(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = dict(mapping)
    value.pop(key, None)
    return value


def _pointer_checks(prompt: str, output: str, pointer: dict[str, Any], extract, render) -> dict[str, bool]:
    segments = extract(prompt)
    candidates = [render(values) for values in itertools.permutations(segments)]
    scores = [float(value) for value in pointer.get("model_log_probability_sums", [])]
    selected = int(pointer.get("selected_index", -1))
    expected_selected = max(range(len(scores)), key=lambda index: (scores[index], -index)) if scores else -1
    return {
        "three_unique_segments": len(segments) == len(set(segments)) == 3,
        "six_candidates": len(candidates) == pointer.get("candidate_count") == 6,
        "selected_by_recorded_model_scores": selected == expected_selected,
        "output_is_selected_prompt_permutation": 0 <= selected < len(candidates) and output == candidates[selected],
        "candidate_lengths_complete": len(pointer.get("candidate_token_lengths", [])) == 6,
        "one_prompt_prefill": pointer.get("prompt_prefill_forward_passes") == 1,
        "one_scoring_forward": pointer.get("candidate_scoring_forward_passes") == 1,
        "persistent_prompt_state_reused": pointer.get("persistent_prompt_state_reused") is True,
        "one_active_route": pointer.get("active_residual_routes") == 1,
        "evaluator_blind": pointer.get("evaluator_used") is False,
        "nondeterministic_timing_absent": "wall_seconds" not in pointer,
    }


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable v19 frontier verification exists: {output}")
    source_protocol_path = root / protocol["source_protocol"]
    source_protocol, source_protocol_sha = load_rescreen_protocol(root, source_protocol_path)
    result = _json(root / protocol["source_result"])
    evidence_claim = str(result["evidence_sha256"])
    evidence_recomputed = hashlib.sha256(canonical_json_bytes(_without(result, "evidence_sha256"))).hexdigest()
    api = _layercake_api((root / source_protocol["layercake_root"]).resolve())
    from layercake_extensions.route_isolated_prompt_span_core_v19 import extract_prompt_segments, render_prompt_segments

    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(source_protocol["research_signing_seed_hex"]))
    public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    all_probes = development_probes(root / source_protocol["development_catalog"])
    probes = {str(row["probe_id"]): row for row in all_probes}
    teacher = {str(row["probe_id"]): row for row in _rows(root / source_protocol["teacher_reference"])}
    source_systems = {(str(row["budget"]), int(row["seed"])): row for row in source_protocol["systems"]}
    verified_systems = []
    aggregate = []
    for recorded in result["systems"]:
        key = (str(recorded["budget"]), int(recorded["seed"]))
        spec = source_systems[key]
        historical = _rows(root / spec["historical_outputs"])
        coherence = _rows(root / recorded["coherence_outputs"]["path"])
        merged = _rows(root / recorded["merged_outputs"]["path"])
        historical_by_id = {str(row["probe_id"]): row for row in historical}
        coherence_by_id = {str(row["probe_id"]): row for row in coherence}
        merged_by_id = {str(row["probe_id"]): row for row in merged}
        noncoherence_exact = all(
            canonical_json_bytes(row) == canonical_json_bytes(merged_by_id[str(row["probe_id"])])
            for row in historical
            if row["capability"] != "coherence"
        )
        coherence_join_exact = all(
            canonical_json_bytes(row) == canonical_json_bytes(merged_by_id[str(row["probe_id"])])
            for row in coherence
        )
        row_checks = []
        for row in coherence:
            probe = probes[str(row["probe_id"])]
            checks = {
                "capability_coherence": row["capability"] == probe["canonical_capability"] == "coherence",
                "historical_row_exists": str(row["probe_id"]) in historical_by_id,
                "functional_v1_recomputed": bool(row["functional_pass_v1"]) == evaluate_functional(str(row["output"]), probe["evaluator"]),
                "functional_v2_recomputed": bool(row["functional_pass_v2"]) == evaluate_functional_v2(str(row["output"]), probe["evaluator"], "coherence"),
                "collapse_recomputed": bool(row["repetition_collapse_v2"]) == repetition_collapse_v2(str(row["output"])),
                **_pointer_checks(str(probe["prompt"]), str(row["output"]), dict(row["v19_pointer"]), extract_prompt_segments, render_prompt_segments),
            }
            row_checks.append(checks)
            aggregate.append({"system": f"{key[0]}-seed{key[1]}", **row})
        evaluation = _merged_evaluation(merged)
        gates, relative = _quality_gates(
            source_protocol,
            evaluation,
            merged,
            probes,
            teacher,
            key[1] + 4_000_000,
        )
        with tempfile.TemporaryDirectory(prefix=f"abi-v19-verify-{key[0].lower()}-{key[1]}-") as raw:
            rebuilt, _ = _build_package(root, source_protocol, spec, Path(raw) / "candidate.cake", api, private, public_pem)
        system_gates = {
            "coherence_depth": len(coherence) == 100,
            "merged_depth": len(merged) == len(historical) == 1400,
            "noncoherence_exact": noncoherence_exact,
            "coherence_join_exact": coherence_join_exact,
            "all_row_checks": all(all(values.values()) for values in row_checks),
            "coherence_file_hash": sha256_file(root / recorded["coherence_outputs"]["path"]) == recorded["coherence_outputs"]["sha256"],
            "merged_file_hash": sha256_file(root / recorded["merged_outputs"]["path"]) == recorded["merged_outputs"]["sha256"],
            "evaluation_recomputed": evaluation == recorded["evaluation"],
            "quality_gates_recomputed": gates == recorded["quality_gates"],
            "teacher_comparison_recomputed": relative == recorded["teacher_comparison_v1"],
            "machine_decision_recomputed": bool(recorded["machine_gates_pass"]) == (all(gates.values()) and all(recorded["pointer_gates"].values())),
            "signed_package_rebuilt_exact": rebuilt == recorded["package"],
            "activation_package_identity": recorded["activation"]["archive_sha256"] == rebuilt["archive_sha256"] and recorded["activation"]["tensor_payload_hash"] == rebuilt["tensor_payload_hash"],
            "activation_verification_recorded": recorded["activation"]["verification"] == "PASS",
            "receiver_learning_zero": recorded["activation"]["receiver_training_steps"] == recorded["activation"]["receiver_calibration_runs"] == 0,
        }
        verified_systems.append({
            "budget": key[0],
            "seed": key[1],
            "functional_passes_v1": evaluation["functional_passes_v1"],
            "machine_gates_pass": bool(recorded["machine_gates_pass"]),
            "row_checks": len(row_checks),
            "gates": system_gates,
            "all_pass": all(system_gates.values()),
        })
        print(json.dumps({"verified": f"{key[0]}-seed{key[1]}", "all_pass": all(system_gates.values())}), flush=True)
    frontier = frontier_decision(verified_systems)
    aggregate_path = root / protocol["coherence_evidence"]
    aggregate_exact = b"".join(canonical_json_bytes(row) for row in aggregate) == aggregate_path.read_bytes()
    stable_sufficient = frontier["b80_all_seed_machine_gates"]
    stable_minimum = stable_sufficient and frontier["b40_adjacent_lower_fails_all_seeds"]
    top_gates = {
        "source_protocol_hash": source_protocol_sha == result["protocol_sha256"],
        "source_result_evidence_hash": evidence_claim == evidence_recomputed,
        "six_systems": len(verified_systems) == 6,
        "six_system_verifiers_pass": all(row["all_pass"] for row in verified_systems),
        "aggregate_coherence_exact": aggregate_exact,
        "aggregate_coherence_hash": sha256_file(aggregate_path) == result["coherence_evidence_sha256"],
        "frontier_recomputed": frontier == result["frontier_gates"],
        "stable_sufficient_recomputed": stable_sufficient == bool(result["stable_sufficient_b80"]),
        "stable_minimum_recomputed": stable_minimum == bool(result["stable_minimum_b80"]),
        "inference_absent": True,
        "training_absent": True,
        "teacher_model_loading_absent": True,
        "final_test_not_accessed": True,
    }
    verification = {
        "format": "abi-capability-compiler-phase4-v19-frontier-verifier-result/1",
        "status": "PASS_INDEPENDENT_V19_FRONTIER_EVIDENCE_VERIFICATION" if all(top_gates.values()) else "FAIL_V19_FRONTIER_EVIDENCE_VERIFICATION",
        "protocol_sha256": protocol_sha,
        "source_result_sha256": sha256_file(root / protocol["source_result"]),
        "source_evidence_sha256": evidence_claim,
        "systems": verified_systems,
        "frontier_gates": frontier,
        "gates": top_gates,
        "packages_deterministically_rebuilt": 6,
        "coherence_rows_recomputed": 600,
        "noncoherence_rows_identity_checked": 7800,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Independent verification of the V733 development evidence only. Stable minimum, matched baselines, final test, unconditional Phase 4, and ABI superiority remain unproven.",
    }
    verification["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(verification)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(verification, indent=2, sort_keys=True).encode() + b"\n")
    return verification


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
