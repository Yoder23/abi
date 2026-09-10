"""Execute the frozen R21 packages on the fresh hidden split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tempfile
import time
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
    write_jsonl_once,
)

from . import run as base_run
from .acquire_hidden import validate_hidden_source
from .hash_assurance_binding import selfless_evidence_hash
from .hidden_binding import load_hidden_config, load_seed_reveal
from .hidden_protocol import hidden_rows
from .protocol import (
    LABELS,
    SEEDS,
    WordLabeler,
    bootstrap_difference,
    instruction_from_prompt,
    normalized_prompt,
    score_output,
)


def _package_for(system: dict[str, Any], method: str, seed: int, label: str):
    if method != "abi_factorized":
        return system["packages"][0]
    matches = [row for row in system["packages"] if row["cake_id"].endswith(f"-{label}-seed{seed}")]
    if len(matches) != 1:
        raise R14Error("R21 hidden factor selection changed")
    return matches[0]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return base_run._aggregate(rows)


def run(config_path: Path, reveal_path: Path, source_run: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R21 hidden result exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config = load_hidden_config(root, config_path)
    seed_hex = load_seed_reveal(config, reveal_path)
    expected = hidden_rows(seed_hex)
    source_rows = validate_hidden_source(root, config_path, reveal_path, source_run)
    source_by_id = {row["record_id"]: row for row in source_rows}
    engine = json_object(root / str(config["public_candidate_engine"]["path"]))
    labeler = WordLabeler(json_object(root / str(config["public_candidate_labeler"]["path"])))
    if not torch.cuda.is_available():
        raise R14Error("R21 hidden execution requires CUDA")
    base = json_object(root / str(config["base_config"]["path"]))
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(base["layercake"]["research_signing_seed_hex"])
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    api = base_run._layercake(root)
    signer = api["key_id"](public_pem)
    output.mkdir(parents=True)
    started = time.perf_counter()
    observations = []
    cpu_rows = []
    for seed in SEEDS:
        for method in ("raw_sequence", "labeled_monolith", "abi_factorized"):
            print(json.dumps({"hidden_system": method, "seed": seed}), flush=True)
            system = engine["systems"][str(seed)][method]
            with tempfile.TemporaryDirectory(prefix=f"r21-hidden-{method}-{seed}-") as raw:
                host = api["DirectCakeHost"](
                    Path(raw) / "gpu",
                    abi_version=base["layercake"]["abi_version"],
                    abi_hash=base["layercake"]["abi_sha256"],
                    trust_store={signer: public_pem},
                    device="cuda",
                )
                cpu_host = api["DirectCakeHost"](
                    Path(raw) / "cpu",
                    abi_version=base["layercake"]["abi_version"],
                    abi_hash=base["layercake"]["abi_sha256"],
                    trust_store={signer: public_pem},
                    device="cpu",
                )
                for package in system["packages"]:
                    if sha256_file(root / package["path"]) != package["sha256"]:
                        raise R14Error("R21 hidden package changed")
                    host.install(root / package["path"])
                    cpu_host.install(root / package["path"])
                for position, row in enumerate(expected):
                    predicted = labeler.predict(instruction_from_prompt(row["prompt"]))
                    prompt = (
                        row["prompt"]
                        if method == "raw_sequence"
                        else normalized_prompt(predicted, row["slots"])
                    )
                    package = _package_for(system, method, seed, predicted)
                    generated = host.generate(package["cake_id"], prompt, maximum_actions=256)
                    text = generated.output.decode("utf-8")
                    teacher = source_by_id[row["record_id"]]["teacher_output"]
                    observations.append(
                        {
                            "method": method,
                            "seed": seed,
                            "position": position,
                            "record_id": row["record_id"],
                            "task": row["task"],
                            "predicted_label": predicted,
                            "cake_id": package["cake_id"],
                            "output": text,
                            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "teacher_output_sha256": hashlib.sha256(teacher.encode()).hexdigest(),
                            **score_output(row, text, teacher),
                        }
                    )
                selected = [next(row for row in expected if row["task"] == task) for task in LABELS]
                for row in selected:
                    predicted = labeler.predict(instruction_from_prompt(row["prompt"]))
                    prompt = (
                        row["prompt"]
                        if method == "raw_sequence"
                        else normalized_prompt(predicted, row["slots"])
                    )
                    package = _package_for(system, method, seed, predicted)
                    cpu_text = cpu_host.generate(
                        package["cake_id"], prompt, maximum_actions=256
                    ).output.decode("utf-8")
                    gpu_text = next(
                        item["output"]
                        for item in observations
                        if item["method"] == method
                        and item["seed"] == seed
                        and item["record_id"] == row["record_id"]
                    )
                    if cpu_text != gpu_text:
                        raise R14Error("R21 hidden CPU/GPU output changed")
                    cpu_rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "record_id": row["record_id"],
                            "task": row["task"],
                            "output_sha256": hashlib.sha256(cpu_text.encode()).hexdigest(),
                        }
                    )
            del host, cpu_host
            gc.collect()
            torch.cuda.empty_cache()
    observations_path = output / "observations.jsonl"
    cpu_path = output / "cpu_observations.jsonl"
    write_jsonl_once(observations_path, observations)
    write_jsonl_once(cpu_path, cpu_rows)
    aggregates = _aggregate(observations)
    teacher_scores = [
        score_output(
            row,
            source_by_id[row["record_id"]]["teacher_output"],
            source_by_id[row["record_id"]]["teacher_output"],
        )
        for row in expected
    ]
    teacher_functional = sum(row["functional_pass"] for row in teacher_scores)
    label_exact = sum(
        labeler.predict(instruction_from_prompt(row["prompt"])) == row["task"] for row in expected
    )
    comparisons = {}
    for seed in SEEDS:
        factor = [
            row for row in observations if row["method"] == "abi_factorized" and row["seed"] == seed
        ]
        for other in ("raw_sequence", "labeled_monolith"):
            control = [
                row for row in observations if row["method"] == other and row["seed"] == seed
            ]
            comparisons[f"seed{seed}:abi_minus_{other}"] = bootstrap_difference(
                [float(row["functional_pass"]) for row in factor],
                [float(row["functional_pass"]) for row in control],
                seed=70_000 + seed + len(other),
                samples=int(config["gates"]["bootstrap_samples"]),
            )
        comparisons[f"seed{seed}:abi_minus_teacher"] = bootstrap_difference(
            [float(row["functional_pass"]) for row in factor],
            [float(row["functional_pass"]) for row in teacher_scores],
            seed=80_000 + seed,
            samples=int(config["gates"]["bootstrap_samples"]),
        )
    gates = {
        "label_exact": label_exact >= int(config["gates"]["minimum_label_exact"]),
        "each_seed_each_task": all(
            min(aggregates[f"abi_factorized:{seed}"]["functional_by_task"].values())
            >= int(config["gates"]["minimum_task_functional_per_seed"])
            for seed in SEEDS
        ),
        "each_seed_non_hallucinating": all(
            aggregates[f"abi_factorized:{seed}"]["non_hallucinating"]
            >= int(config["gates"]["minimum_non_hallucinating_per_seed"])
            for seed in SEEDS
        ),
        "each_seed_non_collapsed": all(
            aggregates[f"abi_factorized:{seed}"]["non_collapsed"]
            >= int(config["gates"]["minimum_non_collapsed_per_seed"])
            for seed in SEEDS
        ),
        "each_seed_no_worse_than_teacher": all(
            aggregates[f"abi_factorized:{seed}"]["functional"] >= teacher_functional
            for seed in SEEDS
        ),
        "each_seed_no_worse_than_controls": all(
            aggregates[f"abi_factorized:{seed}"]["functional"]
            >= aggregates[f"{method}:{seed}"]["functional"]
            for seed in SEEDS
            for method in ("raw_sequence", "labeled_monolith")
        ),
        "all_packages_frozen": True,
        "gpu_rows_complete": len(observations) == 1080,
        "cpu_gpu_rows_complete": len(cpu_rows) == 54,
        "teacher_absent_at_execution": "transformers" not in sys.modules,
        "student_retraining_zero": True,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-r21-hidden-result/1",
        "verdict": "PASS_HIDDEN_REPLICATION" if passed else "FAIL_HIDDEN_REPLICATION",
        "config_sha256": sha256_file(config_path),
        "seed_reveal_sha256": sha256_file(reveal_path),
        "source_receipt_sha256": sha256_file(source_run / "receipt.json"),
        "source_rows_sha256": sha256_file(source_run / "source_observations.jsonl"),
        "public_engine_sha256": config["public_candidate_engine"]["sha256"],
        "observations": {
            "path": observations_path.name,
            "rows": len(observations),
            "sha256": sha256_file(observations_path),
        },
        "cpu_observations": {
            "path": cpu_path.name,
            "rows": len(cpu_rows),
            "sha256": sha256_file(cpu_path),
        },
        "teacher": {
            "functional": teacher_functional,
            "non_hallucinating": sum(row["hallucination_pass"] for row in teacher_scores),
            "non_collapsed": sum(not row["repetition_collapse"] for row in teacher_scores),
        },
        "label_exact": label_exact,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "gates": gates,
        "student_training_steps": 0,
        "teacher_present_at_execution": False,
        "elapsed_seconds": time.perf_counter() - started,
        "claim": "R21_BOUNDED_HIDDEN_REPLICATED_LABEL_SEPARATED_GENERATIVE_TRANSFER"
        if passed
        else "R21_HIDDEN_REPLICATION_FAILED",
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_ABI_MOONSHOT",
        "full_abi_moonshot": "OPEN",
        "next_action": "strictly verify hidden evidence"
        if passed
        else "preserve failure and select a materially different representation",
    }
    result["evidence_sha256"] = selfless_evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed-reveal", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.config, args.seed_reveal, args.source_run, args.output)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
