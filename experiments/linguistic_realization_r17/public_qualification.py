"""Run the R17 public compositional English-realization prerequisite."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from experiments.factual_semantic_r16.public_qualification import (
    _generate,
    _load_source,
    _render_chat,
)
from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)

from .frames import SIGNATURES, public_rows
from .isolation import run_wsl_isolated_extraction
from .package import load_package, realize

SYSTEM = (
    "You are an exact English surface realizer. Follow the supplied semantic "
    "frame literally and return only the requested sentence."
)


def _write_bundle(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    value = {"format": "abi-r17-anonymous-teacher-realizations/1", "records": records}
    value["evidence_sha256"] = evidence_hash(value)
    write_json_once(path, value)
    return value


def _control_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    toggled = {}
    for sig in SIGNATURES:
        mood, tense, polarity, number = sig.split("|")
        other = "question" if mood == "declarative" else "declarative"
        toggled[sig] = "|".join((other, tense, polarity, number))
    return [{**record, "signature": toggled[str(record["signature"])]} for record in records]


def _source_observations(
    tokenizer: Any, model: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    observations = []
    compiler_records = []
    counters = {
        "raw_source_prompts": 0,
        "rendered_prompt_utf8_bytes": 0,
        "rendered_prompt_token_instances": 0,
        "teacher_generated_tokens": 0,
        "teacher_output_bytes": 0,
    }
    for split in ("extraction", "evaluation"):
        for row in public_rows(split):
            rendered = _render_chat(tokenizer, SYSTEM, str(row["prompt"]))
            input_tokens = tokenizer.encode(rendered, add_special_tokens=False)
            completion, token_count = _generate(tokenizer, model, rendered, 48)
            completion = completion.strip()
            exact = completion == row["expected"]
            observations.append(
                {
                    **row,
                    "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "rendered_prompt_utf8_bytes": len(rendered.encode()),
                    "input_token_count": len(input_tokens),
                    "output": completion,
                    "output_sha256": hashlib.sha256(completion.encode()).hexdigest(),
                    "output_token_count": token_count,
                    "functional_exact": exact,
                }
            )
            if split == "extraction":
                compiler_records.append(
                    {
                        "record_id": row["record_id"],
                        "signature": row["signature"],
                        "slots": row["slots"],
                        "output": completion,
                    }
                )
            counters["raw_source_prompts"] += 1
            counters["rendered_prompt_utf8_bytes"] += len(rendered.encode())
            counters["rendered_prompt_token_instances"] += len(input_tokens)
            counters["teacher_generated_tokens"] += token_count
            counters["teacher_output_bytes"] += len(completion.encode())
    return observations, compiler_records, counters


def _package_rows(
    observations: list[dict[str, Any]],
    package: dict[str, Any],
    control_package: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for row in observations:
        if row["split"] != "evaluation":
            continue
        package_output = realize(package, str(row["signature"]), row["slots"])
        control_output = realize(control_package, str(row["signature"]), row["slots"])
        result.append(
            {
                "record_id": row["record_id"],
                "signature": row["signature"],
                "slots": row["slots"],
                "expected": row["expected"],
                "source_output": row["output"],
                "source_functional_exact": row["functional_exact"],
                "package_output": package_output,
                "package_functional_exact": package_output == row["expected"],
                "package_source_exact": package_output == row["output"],
                "removed_output": realize(None, str(row["signature"]), row["slots"]),
                "control_output": control_output,
                "control_functional_exact": control_output == row["expected"],
            }
        )
    return result


def run(model_id: str, revision: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R17 public output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(model_id, revision)
    source_parameters = sum(parameter.numel() for parameter in model.parameters())
    observations, compiler_records, counters = _source_observations(tokenizer, model)
    source_rows_path = output / "source_observations.jsonl"
    write_jsonl_once(source_rows_path, observations)
    extraction_rows = [row for row in observations if row["split"] == "extraction"]
    evaluation_rows = [row for row in observations if row["split"] == "evaluation"]
    source_qualified = all(row["functional_exact"] for row in observations)
    source_identity = {
        "model_id": model_id,
        "revision": revision,
        "snapshot_path_name": snapshot.name,
        "parameters": source_parameters,
        "training_steps": 0,
    }
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if not source_qualified:
        receipt = {
            "format": "abi-r17-public-realization-qualification/1",
            "verdict": "FAIL_SOURCE_PREREQUISITE",
            "claim_ceiling": "PUBLIC_SOURCE_INTERFACE_FAILURE_ONLY",
            "source": source_identity,
            "metrics": {
                "extraction_source_exact": sum(row["functional_exact"] for row in extraction_rows),
                "extraction_rows": len(extraction_rows),
                "evaluation_source_exact": sum(row["functional_exact"] for row in evaluation_rows),
                "evaluation_rows": len(evaluation_rows),
            },
            "artifacts": {
                "source_rows": {
                    "path": source_rows_path.name,
                    "sha256": sha256_file(source_rows_path),
                }
            },
            "information_accounting": counters,
            "full_abi_moonshot": "OPEN",
        }
        receipt["evidence_sha256"] = evidence_hash(receipt)
        write_json_once(output / "receipt.json", receipt)
        return receipt
    generator = random.Random(17_001)
    generator.shuffle(compiler_records)
    bundle_path = output / "source_bundle.json"
    _write_bundle(bundle_path, compiler_records)
    extraction = run_wsl_isolated_extraction(root, bundle_path, output / "extraction")
    control_path = output / "permuted_signature_bundle.json"
    _write_bundle(control_path, _control_records(compiler_records))
    control_extraction = run_wsl_isolated_extraction(
        root, control_path, output / "control_extraction"
    )
    package_path = output / "extraction" / extraction["result"]["package"]["path"]
    control_package_path = (
        output / "control_extraction" / control_extraction["result"]["package"]["path"]
    )
    package = load_package(package_path)
    control_package = load_package(control_package_path)
    rows = _package_rows(observations, package, control_package)
    evaluation_path = output / "evaluation.jsonl"
    write_jsonl_once(evaluation_path, rows)
    metrics = {
        "signatures": len(SIGNATURES),
        "extraction_source_exact": sum(row["functional_exact"] for row in extraction_rows),
        "extraction_rows": len(extraction_rows),
        "evaluation_source_exact": sum(row["functional_exact"] for row in evaluation_rows),
        "evaluation_rows": len(evaluation_rows),
        "package_functional_exact": sum(row["package_functional_exact"] for row in rows),
        "package_source_exact": sum(row["package_source_exact"] for row in rows),
        "removed_abstain": sum(row["removed_output"] is None for row in rows),
        "control_functional_exact": sum(row["control_functional_exact"] for row in rows),
    }
    exact = len(rows)
    passed = (
        metrics["signatures"] == 24
        and metrics["extraction_rows"] == 72
        and metrics["extraction_source_exact"] == 72
        and metrics["evaluation_rows"] == 48
        and metrics["evaluation_source_exact"] == exact
        and metrics["package_functional_exact"] == exact
        and metrics["package_source_exact"] == exact
        and metrics["removed_abstain"] == exact
        and metrics["control_functional_exact"] / exact <= 0.10
    )
    elapsed = time.perf_counter() - started
    receipt = {
        "format": "abi-r17-public-realization-qualification/1",
        "verdict": "PASS" if passed else "FAIL",
        "claim": "PUBLIC_BOUNDED_COMPOSITIONAL_ENGLISH_REALIZATION_PREREQUISITE",
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_LAYERCAKE_ACCEPTANCE",
        "protocol_sha256": sha256_file(Path(__file__).with_name("PUBLIC_PROTOCOL.md")),
        "source": {**source_identity, "present_at_package_execution": False},
        "metrics": metrics,
        "packages": {
            "primary": {
                "path": str(package_path.relative_to(output)),
                "bytes": package_path.stat().st_size,
                "sha256": sha256_file(package_path),
            },
            "control": {
                "path": str(control_package_path.relative_to(output)),
                "bytes": control_package_path.stat().st_size,
                "sha256": sha256_file(control_package_path),
            },
        },
        "isolation": {
            "primary_result_sha256": sha256_file(output / "extraction/result.json"),
            "control_result_sha256": sha256_file(output / "control_extraction/result.json"),
        },
        "artifacts": {
            "source_rows": {"path": source_rows_path.name, "sha256": sha256_file(source_rows_path)},
            "source_bundle": {"path": bundle_path.name, "sha256": sha256_file(bundle_path)},
            "control_bundle": {"path": control_path.name, "sha256": sha256_file(control_path)},
            "evaluation_rows": {
                "path": evaluation_path.name,
                "sha256": sha256_file(evaluation_path),
            },
        },
        "information_accounting": {
            **counters,
            "compiler_source_records": len(compiler_records),
            "compiler_source_bundle_bytes": bundle_path.stat().st_size,
            "final_package_bytes": package_path.stat().st_size,
            "final_templates": len(package["templates"]),
            "frozen_source_parameters_in_package": 0,
            "bridge_parameters_trained": 0,
            "source_training_steps": 0,
            "host_training_steps": 0,
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "full_abi_moonshot": "OPEN",
        "unproven": [
            "unrestricted English fluency",
            "autonomous capability discovery",
            "production LayerCake ingestion",
            "global minimality",
            "superiority to LoRA or distillation",
        ],
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.model_id, args.revision, args.output), indent=2))


if __name__ == "__main__":
    main()
