"""Strictly reproduce and decompose the immutable failed R24 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
)
from experiments.generative_transfer_r21 import run as base_run
from experiments.generative_transfer_r21.hash_assurance_binding import (
    selfless_evidence_hash,
)
from experiments.layercake_composition_r24.binding import load_config as load_original
from experiments.layercake_composition_r24.protocol import NAMESPACES, SEEDS
from experiments.layercake_composition_r24.verify import (
    _assert_packages,
    _live_recompute,
    _metrics,
)

from .binding import load_config


def _stored_rows(run_dir: Path, result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    filenames = {
        "domain_observations": "domain_observations.jsonl",
        "english_immutability": "english_immutability.jsonl",
        "lifecycle": "lifecycle.jsonl",
        "english_leakage": "english_leakage.jsonl",
    }
    if set(result.get("artifacts", {})) != set(filenames):
        raise R14Error("R24 raw artifact inventory changed")
    output = {}
    for name, filename in filenames.items():
        ref = result["artifacts"][name]
        path = run_dir / filename
        if (
            ref.get("path") != filename
            or not path.is_file()
            or sha256_file(path) != ref.get("sha256")
        ):
            raise R14Error("R24 raw artifact binding changed")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != ref.get("rows") or any(not line for line in lines):
            raise R14Error("R24 raw artifact row count changed")
        try:
            rows = [json.loads(line) for line in lines]
        except json.JSONDecodeError as exc:
            raise R14Error("R24 raw artifact is invalid JSONL") from exc
        if any(not isinstance(row, dict) for row in rows):
            raise R14Error("R24 raw artifact contains a non-object")
        output[name] = rows
    return output


def verify(repair_config_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    repair_config_path = repair_config_path.resolve()
    repair = load_config(root, repair_config_path)
    original_config_path = root / repair["original_config"][0]["path"]
    original = load_original(root, original_config_path)
    run_dir = root / "results/layercake_composition_r24/public_v1"
    result_path = run_dir / "result.json"
    result = json_object(result_path)
    if (
        result.get("format") != "abi-r24-layercake-composition-result/1"
        or result.get("config_sha256") != sha256_file(original_config_path)
        or result.get("evidence_sha256") != selfless_evidence_hash(result)
    ):
        raise R14Error("R24 original result identity changed")
    stored = _stored_rows(run_dir, result)
    if not torch.cuda.is_available():
        raise R14Error("R24 negative-result live replay requires CUDA")
    api = base_run._layercake(root)
    r23_config = json_object(root / original["r23_config"]["path"])
    base_config = json_object(root / r23_config["base_config"]["path"])
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(base_config["layercake"]["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signer = api["key_id"](public_pem)
    packages, source_parameters, receiver_steps, teacher_present = _assert_packages(
        root, run_dir, original, result, api, public_pem, signer
    )
    live = _live_recompute(
        root, run_dir, original, result, api, base_config, public_pem, signer
    )
    names = (
        "domain_observations",
        "english_immutability",
        "lifecycle",
        "english_leakage",
    )
    for name, live_rows in zip(names, live, strict=True):
        if stored[name] != live_rows:
            raise R14Error(f"R24 negative-result live replay changed: {name}")
    metrics = _metrics(*live, packages)
    gates = {
        "domain_exact_per_seed": metrics["domain_exact"] == 48 * len(SEEDS),
        "teacher_agreement": metrics["teacher_agreement"] == 48 * len(SEEDS),
        "namespace_route_exact": metrics["namespace_route_exact"] == 48 * len(SEEDS),
        "other_domain_isolation": metrics["other_domain_target_answers"] == 0,
        "english_core_no_domain_answer": metrics["english_leakage_target_answers"] == 0,
        "english_immutable": metrics["english_immutable"] == 120 * len(SEEDS) * 2,
        "cpu_gpu_exact": metrics["cpu_gpu_exact"] == 48 * len(SEEDS),
        "packages_signed": metrics["packages_signed"] == 6,
        "package_lifecycle": metrics["package_lifecycle_exact"] == 6,
        "targeted_corruption": metrics["targeted_corruptions_rejected"] == 6,
        "teacher_absent_at_execution": (
            "transformers" not in sys.modules and not teacher_present
        ),
        "source_parameters_copied_zero": source_parameters == 0,
        "host_training_steps_zero": receiver_steps == 0,
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    if (
        result.get("metrics") != metrics
        or result.get("gates") != gates
        or failed != ["domain_exact_per_seed", "package_lifecycle"]
        or result.get("verdict") != "FAIL_BOUNDED_COMPOSITION"
        or result.get("claim") != "R24_LAYERCAKE_COMPOSITION_FAILED"
        or result.get("full_abi_moonshot") != "OPEN"
    ):
        raise R14Error("R24 negative scientific result changed")

    by_group: dict[tuple[int, str], dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "exact": 0}
    )
    for row in live[0]:
        group = by_group[(int(row["seed"]), str(row["namespace"]))]
        group["rows"] += 1
        group["exact"] += int(row["gpu_exact"])
    group_metrics = {
        f"{seed}:{namespace}": {
            **by_group[(seed, namespace)],
            "misses": by_group[(seed, namespace)]["rows"]
            - by_group[(seed, namespace)]["exact"],
        }
        for seed in SEEDS
        for namespace in NAMESPACES
    }
    first_output = {}
    for row in live[0]:
        key = (int(row["seed"]), str(row["namespace"]))
        first_output.setdefault(key, row["gpu_output"])
    restored_behavior_exact = sum(
        row["restored_output"]
        == first_output[(int(row["seed"]), str(row["namespace"]))]
        for row in live[2]
    )
    if restored_behavior_exact != 6:
        raise R14Error("R24 corrected lifecycle diagnostic failed")
    verification = {
        "format": "abi-r24-strict-negative-verification/1",
        "status": "PASS_STRICTLY_VERIFIED_NEGATIVE_DOMAIN_GENERALIZATION",
        "scientific_verdict": result["verdict"],
        "scientific_claim": result["claim"],
        "claim_ceiling": result["claim_ceiling"],
        "repair_config_sha256": sha256_file(repair_config_path),
        "original_config_sha256": sha256_file(original_config_path),
        "result_sha256": sha256_file(result_path),
        "rows_live_recomputed": sum(len(rows) for rows in live),
        "packages_recomputed": len(packages),
        "stored_scientific_booleans_trusted": False,
        "metrics": metrics,
        "gates": gates,
        "failed_gates": failed,
        "domain_exact_by_seed_and_namespace": group_metrics,
        "lifecycle_gate_note": (
            "The frozen gate compared restored output with gold task output; "
            "restoration itself reproduced pre-removal package behavior."
        ),
        "restored_behavior_exact": restored_behavior_exact,
        "restored_behavior_rows": 6,
        "candidate_changes": repair["candidate_changes"],
        "gate_changes": repair["gate_changes"],
        "scorer_changes": repair["scorer_changes"],
        "source_parameters_copied": source_parameters,
        "host_training_steps": receiver_steps,
        "teacher_present_at_execution": teacher_present,
        "full_abi_moonshot": "OPEN",
    }
    verification["evidence_sha256"] = selfless_evidence_hash(verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.config)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
