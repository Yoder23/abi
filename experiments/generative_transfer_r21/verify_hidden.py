"""Strict recomputation of the R21 fresh hidden replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
)

from .acquire_hidden import validate_hidden_source
from .evaluate_hidden import _aggregate
from .hash_assurance_binding import selfless_evidence_hash
from .hidden_binding import load_hidden_config, load_seed_reveal
from .hidden_protocol import hidden_rows
from .protocol import (
    SEEDS,
    WordLabeler,
    bootstrap_difference,
    instruction_from_prompt,
    score_output,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error(f"R21 hidden JSONL unreadable: {path}") from exc


def verify(
    config_path: Path,
    reveal_path: Path,
    source_run: Path,
    result_dir: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config = load_hidden_config(root, config_path)
    seed = load_seed_reveal(config, reveal_path)
    expected = hidden_rows(seed)
    expected_by_id = {row["record_id"]: row for row in expected}
    source_rows = validate_hidden_source(root, config_path, reveal_path, source_run)
    source_by_id = {row["record_id"]: row for row in source_rows}
    result = json_object(result_dir / "result.json")
    observations_path = result_dir / "observations.jsonl"
    cpu_path = result_dir / "cpu_observations.jsonl"
    observations = _jsonl(observations_path)
    cpu_rows = _jsonl(cpu_path)
    if (
        result.get("format") != "abi-r21-hidden-result/1"
        or result.get("config_sha256") != sha256_file(config_path)
        or result.get("seed_reveal_sha256") != sha256_file(reveal_path)
        or result.get("source_receipt_sha256") != sha256_file(source_run / "receipt.json")
        or result.get("source_rows_sha256") != sha256_file(source_run / "source_observations.jsonl")
        or result.get("evidence_sha256") != selfless_evidence_hash(result)
        or result.get("observations", {}).get("sha256") != sha256_file(observations_path)
        or result.get("cpu_observations", {}).get("sha256") != sha256_file(cpu_path)
        or len(observations) != 1080
        or len(cpu_rows) != 54
    ):
        raise R14Error("R21 hidden result binding failed")
    keys = {(row["method"], int(row["seed"]), row["record_id"]) for row in observations}
    expected_keys = {
        (method, seed_value, row["record_id"])
        for method in ("raw_sequence", "labeled_monolith", "abi_factorized")
        for seed_value in SEEDS
        for row in expected
    }
    if keys != expected_keys:
        raise R14Error("R21 hidden observation matrix changed")
    for row in observations:
        reference = expected_by_id[row["record_id"]]
        teacher = source_by_id[row["record_id"]]["teacher_output"]
        score = score_output(reference, row["output"], teacher)
        if any(row.get(name) != value for name, value in score.items()):
            raise R14Error("R21 hidden score changed")
        if (
            row.get("output_sha256") != hashlib.sha256(row["output"].encode()).hexdigest()
            or row.get("teacher_output_sha256") != hashlib.sha256(teacher.encode()).hexdigest()
        ):
            raise R14Error("R21 hidden output identity changed")
    aggregates = _aggregate(observations)
    if aggregates != result.get("aggregates"):
        raise R14Error("R21 hidden aggregate changed")
    teacher_scores = [
        score_output(
            row,
            source_by_id[row["record_id"]]["teacher_output"],
            source_by_id[row["record_id"]]["teacher_output"],
        )
        for row in expected
    ]
    teacher_functional = sum(row["functional_pass"] for row in teacher_scores)
    labeler = WordLabeler(json_object(root / str(config["public_candidate_labeler"]["path"])))
    label_exact = sum(
        labeler.predict(instruction_from_prompt(row["prompt"])) == row["task"] for row in expected
    )
    teacher_summary = {
        "functional": teacher_functional,
        "non_hallucinating": sum(row["hallucination_pass"] for row in teacher_scores),
        "non_collapsed": sum(not row["repetition_collapse"] for row in teacher_scores),
    }
    if result.get("teacher") != teacher_summary or result.get("label_exact") != label_exact:
        raise R14Error("R21 hidden teacher or label aggregate changed")
    comparisons = {}
    for seed_value in SEEDS:
        factor = [
            row
            for row in observations
            if row["method"] == "abi_factorized" and row["seed"] == seed_value
        ]
        for other in ("raw_sequence", "labeled_monolith"):
            control = [
                row for row in observations if row["method"] == other and row["seed"] == seed_value
            ]
            comparisons[f"seed{seed_value}:abi_minus_{other}"] = bootstrap_difference(
                [float(row["functional_pass"]) for row in factor],
                [float(row["functional_pass"]) for row in control],
                seed=70_000 + seed_value + len(other),
                samples=int(config["gates"]["bootstrap_samples"]),
            )
        comparisons[f"seed{seed_value}:abi_minus_teacher"] = bootstrap_difference(
            [float(row["functional_pass"]) for row in factor],
            [float(row["functional_pass"]) for row in teacher_scores],
            seed=80_000 + seed_value,
            samples=int(config["gates"]["bootstrap_samples"]),
        )
    if comparisons != result.get("comparisons"):
        raise R14Error("R21 hidden comparison changed")
    gates = {
        "label_exact": label_exact >= int(config["gates"]["minimum_label_exact"]),
        "each_seed_each_task": all(
            min(aggregates[f"abi_factorized:{seed_value}"]["functional_by_task"].values())
            >= int(config["gates"]["minimum_task_functional_per_seed"])
            for seed_value in SEEDS
        ),
        "each_seed_non_hallucinating": all(
            aggregates[f"abi_factorized:{seed_value}"]["non_hallucinating"]
            >= int(config["gates"]["minimum_non_hallucinating_per_seed"])
            for seed_value in SEEDS
        ),
        "each_seed_non_collapsed": all(
            aggregates[f"abi_factorized:{seed_value}"]["non_collapsed"]
            >= int(config["gates"]["minimum_non_collapsed_per_seed"])
            for seed_value in SEEDS
        ),
        "each_seed_no_worse_than_teacher": all(
            aggregates[f"abi_factorized:{seed_value}"]["functional"] >= teacher_functional
            for seed_value in SEEDS
        ),
        "each_seed_no_worse_than_controls": all(
            aggregates[f"abi_factorized:{seed_value}"]["functional"]
            >= aggregates[f"{method}:{seed_value}"]["functional"]
            for seed_value in SEEDS
            for method in ("raw_sequence", "labeled_monolith")
        ),
        "all_packages_frozen": True,
        "gpu_rows_complete": len(observations) == 1080,
        "cpu_gpu_rows_complete": len(cpu_rows) == 54,
        "teacher_absent_at_execution": result.get("teacher_present_at_execution") is False,
        "student_retraining_zero": result.get("student_training_steps") == 0,
    }
    if gates != result.get("gates"):
        raise R14Error("R21 hidden gate recomputation changed")
    passed = all(gates.values())
    if (result.get("verdict") == "PASS_HIDDEN_REPLICATION") != passed:
        raise R14Error("R21 hidden verdict does not follow gates")
    verification = {
        "format": "abi-r21-hidden-strict-verification/1",
        "status": "PASS_VERIFIED_HIDDEN_REPLICATION"
        if passed
        else "PASS_VERIFIED_NEGATIVE_HIDDEN_RESULT",
        "config_sha256": sha256_file(config_path),
        "result_sha256": sha256_file(result_dir / "result.json"),
        "source_receipt_sha256": sha256_file(source_run / "receipt.json"),
        "rows_recomputed": len(observations),
        "cpu_rows_recomputed": len(cpu_rows),
        "packages_rebound": len(config["packages"]),
        "label_exact": label_exact,
        "teacher_functional": teacher_functional,
        "gates": gates,
        "scientific_verdict": result["verdict"],
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_ABI_MOONSHOT",
        "full_abi_moonshot": "OPEN",
    }
    verification["evidence_sha256"] = selfless_evidence_hash(verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed-reveal", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.config, args.seed_reveal, args.source_run, args.result)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
