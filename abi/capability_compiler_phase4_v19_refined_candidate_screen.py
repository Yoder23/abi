"""Apply the certified v19 coherence path to one refined-budget final lineage."""

from __future__ import annotations

import argparse
import gc
import hashlib
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


FORMAT = "abi-capability-compiler-phase4-v19-refined-candidate-screen/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if "base_protocol" in protocol:
        repair = protocol
        base = _json(root / str(repair["base_protocol"]))
        protocol = {**base, **repair}
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_ONE_REFINED_CANDIDATE_V19_SCREEN"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("refined candidate screen governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"refined candidate screen binding changed: {relative}")
    return protocol, sha256_file(path)


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable refined v19 screen exists: {output}")
    if not torch.cuda.is_available():
        raise Phase3Error("preregistered CUDA device unavailable")
    spec = protocol["system"]
    api = _layercake_api((root / protocol["layercake_root"]).resolve())
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(protocol["research_signing_seed_hex"]))
    public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    signer = api["key_id"](public_pem)
    all_probes = development_probes(root / protocol["development_catalog"])
    probes = {str(row["probe_id"]): row for row in all_probes}
    coherence_probes = [row for row in all_probes if row["canonical_capability"] == "coherence"]
    teacher = {str(row["probe_id"]): row for row in _rows(root / protocol["teacher_reference"])}
    historical = _rows(root / spec["historical_outputs"])
    if len(historical) != 1400 or len(coherence_probes) != 100:
        raise Phase3Error("locked development depth changed")
    historical_by_id = {str(row["probe_id"]): row for row in historical}
    with tempfile.TemporaryDirectory(prefix=f"abi-v19-{spec['budget'].lower()}-{spec['seed']}-") as raw:
        temp = Path(raw)
        package, _ = _build_package(root, protocol, spec, temp / "candidate.cake", api, private, public_pem)
        host = api["Host"](temp / "registry", trust_store={signer: public_pem}, device="cuda")
        activation = host.activate(temp / "candidate.cake")
        coherence = []
        for probe in coherence_probes:
            value = host.generate(str(probe["prompt"]), maximum_tokens=int(probe["max_new_tokens"])).decode("utf-8", errors="strict")
            pointer = dict(host.last_pointer_execution or {})
            pointer.pop("wall_seconds", None)
            old = historical_by_id[str(probe["probe_id"])]
            coherence.append({
                **old,
                "output": value,
                "original_output": value,
                "output_token_ids": [int(item) for item in host.model_tokenizer.encode(value)],
                "automatic_capability_route": "coherence" if pointer else old["automatic_capability_route"],
                "capability_route_correct": bool(pointer),
                "guard_terminated": False,
                "abstention_clause_prefixed": False,
                "functional_pass_v1": evaluate_functional(value, probe["evaluator"]),
                "functional_pass_v2": evaluate_functional_v2(value, probe["evaluator"], "coherence"),
                "repetition_collapse_v2": repetition_collapse_v2(value),
                "v19_pointer": pointer,
            })
        verification = host.verify()
        activation_receipt = {
            "archive_sha256": activation["archive_hash"],
            "tensor_payload_hash": activation["payload_hash"],
            "receiver_training_steps": activation["receiver_training_steps"],
            "receiver_calibration_runs": activation["receiver_calibration_runs"],
            "verification": verification["status"],
        }
        del host
        gc.collect()
        torch.cuda.empty_cache()
    replacements = {str(row["probe_id"]): row for row in coherence}
    merged = [replacements.get(str(row["probe_id"]), row) for row in historical]
    evaluation = _merged_evaluation(merged)
    gates, relative = _quality_gates(protocol, evaluation, merged, probes, teacher, int(spec["seed"]) + 4_000_000)
    pointer_gates = {
        "all_100_pointer_rows": len(coherence) == 100 and all(bool(row["v19_pointer"]) for row in coherence),
        "six_candidates": all(row["v19_pointer"].get("candidate_count") == 6 for row in coherence),
        "one_scoring_forward": all(row["v19_pointer"].get("candidate_scoring_forward_passes") == 1 for row in coherence),
        "one_active_route": all(row["v19_pointer"].get("active_residual_routes") == 1 for row in coherence),
        "persistent_state_reused": all(row["v19_pointer"].get("persistent_prompt_state_reused") is True for row in coherence),
        "evaluator_blind": all(row["v19_pointer"].get("evaluator_used") is False for row in coherence),
        "package_identity": package["archive_sha256"] == activation_receipt["archive_sha256"] and package["tensor_payload_hash"] == activation_receipt["tensor_payload_hash"],
        "package_verified": activation_receipt["verification"] == "PASS",
        "receiver_learning_zero": activation_receipt["receiver_training_steps"] == activation_receipt["receiver_calibration_runs"] == 0,
    }
    output.mkdir(parents=True)
    coherence_path = output / "coherence_outputs.jsonl"
    merged_path = output / "merged_development_outputs.jsonl"
    _write_immutable(coherence_path, b"".join(canonical_json_bytes(row) for row in coherence))
    _write_immutable(merged_path, b"".join(canonical_json_bytes(row) for row in merged))
    machine_pass = all(gates.values()) and all(pointer_gates.values())
    result = {
        "format": "abi-capability-compiler-phase4-v19-refined-candidate-screen-result/1",
        "status": "PASS_REFINED_CANDIDATE_V19_MACHINE_GATES" if machine_pass else "FAIL_REFINED_CANDIDATE_V19_MACHINE_GATES",
        "protocol_sha256": protocol_sha,
        "budget": spec["budget"],
        "seed": int(spec["seed"]),
        "imported_information": protocol["imported_information"],
        "historical_functional_passes_v1": sum(bool(row["functional_pass_v1"]) for row in historical),
        "historical_coherence_passes_v1": sum(bool(row["functional_pass_v1"]) for row in historical if row["capability"] == "coherence"),
        "v19_functional_passes_v1": evaluation["functional_passes_v1"],
        "v19_coherence_passes_v1": sum(bool(row["functional_pass_v1"]) for row in coherence),
        "evaluation": evaluation,
        "teacher_comparison_v1": relative,
        "quality_gates": gates,
        "pointer_gates": pointer_gates,
        "machine_gates_pass": machine_pass,
        "package": package,
        "activation": activation_receipt,
        "coherence_outputs_sha256": sha256_file(coherence_path),
        "merged_outputs_sha256": sha256_file(merged_path),
        "noncoherence_rows_reused": 1300,
        "model_inference_rows": 100,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "One refined-budget seed v19 development screen only. No three-seed result, minimum, matched baseline, final test, Phase 4, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
