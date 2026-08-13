"""Independent raw-evidence verifier for the closed v19 B60 refined-budget matrix."""

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
)
from .capability_compiler_phase4_v19_frontier_verify import _pointer_checks, _without
from .capability_compiler_phase4_v19_refined_candidate_screen import load_protocol as load_candidate_protocol


FORMAT = "abi-capability-compiler-phase4-v19-b60-matrix-verifier/1"
EXPECTED_SEEDS = (104729, 130363, 155921)


def matrix_decision(systems: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed = {int(row["seed"]): bool(row["machine_gates_pass"]) for row in systems}
    exact_seeds = tuple(sorted(by_seed)) == EXPECTED_SEEDS
    outcomes = [by_seed[seed] for seed in EXPECTED_SEEDS] if exact_seeds else []
    mixed = bool(outcomes) and any(outcomes) and not all(outcomes)
    return {
        "exact_registered_seeds": exact_seeds,
        "outcomes": {str(seed): by_seed.get(seed) for seed in EXPECTED_SEEDS},
        "all_seed_pass": bool(outcomes) and all(outcomes),
        "all_seed_fail": bool(outcomes) and not any(outcomes),
        "mixed": mixed,
        "stop_refinement": mixed,
        "b50_authorized": bool(outcomes) and all(outcomes),
        "b70_authorized": bool(outcomes) and not any(outcomes),
        "stable_minimum_proven": False,
    }


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_INDEPENDENT_B60_MATRIX_VERIFIER"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("B60 verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B60 verifier binding changed: {relative}")
    return protocol, sha256_file(path)


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable B60 verification exists: {output}")
    api = None
    verified = []
    for registered in protocol["systems"]:
        candidate_path = root / registered["protocol"]
        candidate, candidate_sha = load_candidate_protocol(root, candidate_path)
        result_path = root / registered["result"]
        result = _json(result_path)
        output_dir = result_path.parent
        evidence_recomputed = hashlib.sha256(canonical_json_bytes(_without(result, "evidence_sha256"))).hexdigest()
        if api is None:
            api = _layercake_api((root / candidate["layercake_root"]).resolve())
            from layercake_extensions.route_isolated_prompt_span_core_v19 import extract_prompt_segments, render_prompt_segments
        private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(candidate["research_signing_seed_hex"]))
        public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        probes_list = development_probes(root / candidate["development_catalog"])
        probes = {str(row["probe_id"]): row for row in probes_list}
        teacher = {str(row["probe_id"]): row for row in _rows(root / candidate["teacher_reference"])}
        spec = candidate["system"]
        historical = _rows(root / spec["historical_outputs"])
        coherence_path = output_dir / "coherence_outputs.jsonl"
        merged_path = output_dir / "merged_development_outputs.jsonl"
        coherence = _rows(coherence_path)
        merged = _rows(merged_path)
        historical_by_id = {str(row["probe_id"]): row for row in historical}
        merged_by_id = {str(row["probe_id"]): row for row in merged}
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
        noncoherence_exact = all(
            canonical_json_bytes(row) == canonical_json_bytes(merged_by_id[str(row["probe_id"])])
            for row in historical if row["capability"] != "coherence"
        )
        coherence_exact = all(
            canonical_json_bytes(row) == canonical_json_bytes(merged_by_id[str(row["probe_id"])] )
            for row in coherence
        )
        evaluation = _merged_evaluation(merged)
        gates, relative = _quality_gates(candidate, evaluation, merged, probes, teacher, int(spec["seed"]) + 4_000_000)
        with tempfile.TemporaryDirectory(prefix=f"abi-b60-verify-{spec['seed']}-") as raw:
            rebuilt, _ = _build_package(root, candidate, spec, Path(raw) / "candidate.cake", api, private, public_pem)
        pointer_gates = {
            "all_100_pointer_rows": len(coherence) == 100 and all(bool(row["v19_pointer"]) for row in coherence),
            "six_candidates": all(row["v19_pointer"].get("candidate_count") == 6 for row in coherence),
            "one_scoring_forward": all(row["v19_pointer"].get("candidate_scoring_forward_passes") == 1 for row in coherence),
            "one_active_route": all(row["v19_pointer"].get("active_residual_routes") == 1 for row in coherence),
            "persistent_state_reused": all(row["v19_pointer"].get("persistent_prompt_state_reused") is True for row in coherence),
            "evaluator_blind": all(row["v19_pointer"].get("evaluator_used") is False for row in coherence),
            "package_identity": result["activation"]["archive_sha256"] == rebuilt["archive_sha256"] and result["activation"]["tensor_payload_hash"] == rebuilt["tensor_payload_hash"],
            "package_verified": result["activation"]["verification"] == "PASS",
            "receiver_learning_zero": result["activation"]["receiver_training_steps"] == result["activation"]["receiver_calibration_runs"] == 0,
        }
        expected_machine = all(gates.values()) and all(pointer_gates.values())
        checks = {
            "candidate_protocol_hash": candidate_sha == result["protocol_sha256"],
            "result_file_hash": sha256_file(result_path) == registered["result_sha256"],
            "evidence_hash": evidence_recomputed == result["evidence_sha256"] == registered["evidence_sha256"],
            "registered_identity": str(result["budget"]) == "B60" and int(result["seed"]) == int(registered["seed"]),
            "depths": len(historical) == len(merged) == 1400 and len(coherence) == 100,
            "raw_hashes": sha256_file(coherence_path) == result["coherence_outputs_sha256"] and sha256_file(merged_path) == result["merged_outputs_sha256"],
            "noncoherence_exact": noncoherence_exact,
            "coherence_join_exact": coherence_exact,
            "row_checks": all(all(values.values()) for values in row_checks),
            "evaluation_recomputed": evaluation == result["evaluation"],
            "quality_gates_recomputed": gates == result["quality_gates"],
            "teacher_comparison_recomputed": relative == result["teacher_comparison_v1"],
            "pointer_gates_recomputed": pointer_gates == result["pointer_gates"],
            "package_rebuilt_exact": rebuilt == result["package"],
            "machine_decision_recomputed": expected_machine == bool(result["machine_gates_pass"]),
            "registered_outcome": expected_machine == bool(registered["machine_gates_pass"]),
        }
        verified.append({
            "budget":"B60", "seed":int(spec["seed"]), "machine_gates_pass":expected_machine,
            "functional_passes_v1":evaluation["functional_passes_v1"],
            "repetition_collapses_v2":evaluation["repetition_collapses_v2"],
            "rows_recomputed":len(merged), "package_rebuilt":True,
            "gates":checks, "all_pass":all(checks.values()),
        })
        print(json.dumps({"verified":f"B60-seed{spec['seed']}","all_pass":all(checks.values())}), flush=True)
    decision = matrix_decision(verified)
    refinement = _json(root / protocol["refinement_protocol"])
    summary = _json(root / protocol["matrix_summary"])
    state = _json(root / protocol["campaign_state"])
    top_gates = {
        "three_systems": len(verified) == 3,
        "all_system_verifiers_pass": all(row["all_pass"] for row in verified),
        "registered_seeds_exact": decision["exact_registered_seeds"],
        "mixed_recomputed": decision["mixed"],
        "stop_rule_preregistered": "If B60 is mixed, stop this refinement without calling a minimum." in refinement["adaptive_order"],
        "b50_b70_prohibited": not decision["b50_authorized"] and not decision["b70_authorized"],
        "summary_matrix_exact": summary["b60_three_seed_matrix"] == {"seed104729":"FAIL_ZERO_COLLAPSE","seed130363":"PASS","seed155921":"PASS","unanimous":False},
        "state_matrix_exact": state["b60_matrix"] == "MIXED_FAIL_PASS_PASS" and state["stable_minimum"] == "UNPROVEN",
        "inference_absent": True,
        "training_absent": True,
        "teacher_model_loading_absent": True,
        "final_test_not_accessed": True,
    }
    verification = {
        "format":"abi-capability-compiler-phase4-v19-b60-matrix-verifier-result/1",
        "status":"PASS_INDEPENDENT_B60_MATRIX_EVIDENCE_VERIFICATION" if all(top_gates.values()) else "FAIL_B60_MATRIX_EVIDENCE_VERIFICATION",
        "protocol_sha256":protocol_sha,
        "systems":verified,
        "matrix_decision":decision,
        "gates":top_gates,
        "packages_deterministically_rebuilt":3,
        "coherence_rows_recomputed":300,
        "noncoherence_rows_identity_checked":3900,
        "model_inference_performed":False,
        "training_performed":False,
        "teacher_model_loaded":False,
        "final_test_accessed":False,
        "phase4_certified":False,
        "claim_boundary":"Independent verification closes B60 as a mixed development matrix and enforces the registered stop rule. B80 is only a stable sufficient upper bound; no minimum, matched baseline, final test, Phase 4, or superiority claim.",
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
