"""Execute the preregistered R16 held-out factual acquisition campaign."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from experiments.preexisting_representation_r15b.public_qualification import sha256_bytes

from .facts import namespace_from_question, question
from .isolation import run_wsl_isolated_extraction
from .package import answer, load_package
from .protocol import candidate_values, evaluation_rows, heldout_facts
from .public_qualification import _generate, _load_source, _render_chat
from .public_sequence_scoring import _candidate_scores, normalized_text


def _verify_bindings(root: Path, config: dict[str, Any]) -> None:
    for relative, expected in config["code_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise R14Error(f"R16 code binding changed: {relative}")
    public = root / config["public_prerequisite"]["receipt"]
    if sha256_file(public) != config["public_prerequisite"]["sha256"]:
        raise R14Error("R16 public prerequisite changed")


def _source_rows(
    config: dict[str, Any],
    facts: list[Any],
    secret_hex: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[torch.Tensor], dict[str, int]]:
    tokenizer, model, _snapshot = _load_source(
        str(config["source"]["model_id"]), str(config["source"]["revision"])
    )
    system = "Answer the factual question with only the answer and no explanation."
    raw = []
    bundle = []
    residuals = []
    counters = {"generated_tokens": 0, "output_bytes": 0, "candidate_scores": 0}
    for fact in facts:
        candidates = candidate_values(
            fact, int(config["data"]["candidate_budget"]), secret_hex
        )
        for view in config["data"]["extraction_views"]:
            stem = question(fact, int(view))
            rendered = _render_chat(tokenizer, system, stem)
            completion, tokens = _generate(
                tokenizer, model, rendered, int(config["source"]["max_new_tokens"])
            )
            scores, residual = _candidate_scores(tokenizer, model, rendered, candidates)
            predicted = str(candidates[max(range(len(scores)), key=scores.__getitem__)])
            raw.append(
                {
                    "split": "extraction",
                    "fact_id": fact.fact_id,
                    "view": view,
                    "namespace": fact.namespace,
                    "entity": fact.entity,
                    "answer": fact.value,
                    "question": stem,
                    "completion": completion,
                    "completion_exact": normalized_text(completion) == normalized_text(fact.value),
                    "candidate_values": list(candidates),
                    "candidate_scores": scores,
                    "candidate_prediction": predicted,
                    "candidate_exact": predicted == fact.value,
                    "semantic_prediction": namespace_from_question(stem),
                    "residual_sha256": sha256_bytes(residual.numpy().tobytes()),
                }
            )
            bundle.append(
                {
                    "subject": fact.entity,
                    "question": stem,
                    "view": view,
                    "candidates": list(candidates),
                    "scores": scores,
                }
            )
            residuals.append(residual)
            counters["generated_tokens"] += tokens
            counters["output_bytes"] += len(completion.encode())
            counters["candidate_scores"] += len(scores)
        for view in config["data"]["evaluation_views"]:
            stem = question(fact, int(view))
            rendered = _render_chat(tokenizer, system, stem)
            completion, tokens = _generate(
                tokenizer, model, rendered, int(config["source"]["max_new_tokens"])
            )
            raw.append(
                {
                    "split": "evaluation",
                    "fact_id": fact.fact_id,
                    "view": view,
                    "namespace": fact.namespace,
                    "entity": fact.entity,
                    "answer": fact.value,
                    "question": stem,
                    "completion": completion,
                    "completion_exact": normalized_text(completion) == normalized_text(fact.value),
                }
            )
            counters["generated_tokens"] += tokens
            counters["output_bytes"] += len(completion.encode())
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    generator = random.Random(int(secret_hex[:16], 16))
    generator.shuffle(bundle)
    return raw, bundle, residuals, counters


def _write_source_bundle(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    value = {"format": "abi-r16-anonymous-source-scores/1", "records": records}
    value["evidence_sha256"] = evidence_hash(value)
    write_json_once(path, value)
    return value


def _rotated_bundle(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        scores = list(record["scores"])
        result.append({**record, "scores": scores[1:] + scores[:1]})
    return result


def _packages(extraction: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    return [load_package(extraction / item["path"]) for item in result["packages"]]


def _evaluation_matrix(
    facts: list[Any],
    packages: list[dict[str, Any]],
    control_packages: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source = {
        (row["fact_id"], int(row["view"])): row
        for row in source_rows
        if row["split"] == "evaluation"
    }
    chemistry = [item for item in packages if item["namespace"].startswith("chemistry/")]
    geography = [item for item in packages if item["namespace"].startswith("geography/")]
    rows = []
    for row in evaluation_rows(facts):
        source_row = source[(row["fact_id"], int(row["view"]))]
        target_only = chemistry if row["namespace"].startswith("chemistry/") else geography
        other_only = geography if target_only is chemistry else chemistry
        combined = answer(packages, row["query"])
        rows.append(
            {
                **row,
                "source_completion": source_row["completion"],
                "combined_prediction": combined,
                "combined_exact": combined == row["answer"],
                "source_agreement": normalized_text(str(combined or ""))
                == normalized_text(source_row["completion"]),
                "target_only_prediction": answer(target_only, row["query"]),
                "other_only_prediction": answer(other_only, row["query"]),
                "removed_prediction": answer([], row["query"]),
                "rotated_score_prediction": answer(control_packages, row["query"]),
            }
        )
    return rows


def run(config_path: Path, reveal_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R16 held-out output exists: {output}")
    root = config_path.resolve().parents[3]
    config = json_object(config_path)
    reveal = json_object(reveal_path)
    _verify_bindings(root, config)
    if sha256_file(reveal_path) != config["reveal_sha256"]:
        raise R14Error("R16 reveal file changed")
    facts = heldout_facts(
        str(reveal["secret_hex"]),
        str(config["heldout_seed_commitment"]),
        int(config["data"]["facts_per_namespace"]),
    )
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source_rows, source_bundle_rows, residuals, counters = _source_rows(
        config, facts, str(reveal["secret_hex"])
    )
    source_path = output / "source_observations.jsonl"
    write_jsonl_once(source_path, source_rows)
    residual_path = output / "source_prompt_end_residuals.safetensors"
    save_file({"residuals": torch.stack(residuals)}, str(residual_path))
    bundle_path = output / "source_bundle.json"
    _write_source_bundle(bundle_path, source_bundle_rows)
    extraction = run_wsl_isolated_extraction(
        root,
        bundle_path,
        output / "extraction",
        str(config["physical_extraction"]["distribution"]),
    )
    control_bundle_path = output / "rotated_score_bundle.json"
    _write_source_bundle(control_bundle_path, _rotated_bundle(source_bundle_rows))
    control_extraction = run_wsl_isolated_extraction(
        root,
        control_bundle_path,
        output / "control_extraction",
        str(config["physical_extraction"]["distribution"]),
    )
    packages = _packages(output / "extraction", extraction["result"])
    control_packages = _packages(output / "control_extraction", control_extraction["result"])
    matrix = _evaluation_matrix(facts, packages, control_packages, source_rows)
    matrix_path = output / "evaluation.jsonl"
    write_jsonl_once(matrix_path, matrix)
    metrics = {
        "selected_facts": len(facts),
        "extraction_open_exact": sum(
            int(row["completion_exact"]) for row in source_rows if row["split"] == "extraction"
        ),
        "extraction_rows": sum(row["split"] == "extraction" for row in source_rows),
        "candidate_exact": sum(
            int(row["candidate_exact"]) for row in source_rows if row["split"] == "extraction"
        ),
        "semantic_exact": sum(
            row.get("semantic_prediction") == row["namespace"]
            for row in source_rows
            if row["split"] == "extraction"
        ),
        "evaluation_source_exact": sum(
            int(row["completion_exact"]) for row in source_rows if row["split"] == "evaluation"
        ),
        "evaluation_rows": len(matrix),
        "package_exact": sum(int(row["combined_exact"]) for row in matrix),
        "source_agreement": sum(int(row["source_agreement"]) for row in matrix),
        "target_only_exact": sum(row["target_only_prediction"] == row["answer"] for row in matrix),
        "other_only_abstain": sum(row["other_only_prediction"] is None for row in matrix),
        "removed_abstain": sum(row["removed_prediction"] is None for row in matrix),
        "rotated_score_exact": sum(
            row["rotated_score_prediction"] == row["answer"] for row in matrix
        ),
    }
    exact = len(matrix)
    extraction_rows = metrics["extraction_rows"]
    gates = config["gates"]
    passed = (
        metrics["selected_facts"] == int(config["data"]["facts_per_namespace"]) * 2
        and metrics["extraction_open_exact"] == extraction_rows
        and metrics["candidate_exact"] == extraction_rows
        and metrics["semantic_exact"] == extraction_rows
        and metrics["evaluation_source_exact"] == exact
        and metrics["package_exact"] == exact
        and metrics["source_agreement"] == exact
        and metrics["target_only_exact"] == exact
        and metrics["other_only_abstain"] == exact
        and metrics["removed_abstain"] == exact
        and metrics["rotated_score_exact"] / exact <= float(gates["max_control_accuracy"])
    )
    package_records = []
    for item in extraction["result"]["packages"]:
        path = output / "extraction" / item["path"]
        package_records.append({**item, "sha256": sha256_file(path)})
    elapsed = time.perf_counter() - started
    receipt = {
        "format": "abi-r16-heldout-factual-acquisition/1",
        "verdict": "PASS" if passed else "FAIL",
        "claim": "BOUNDED_FACTUAL_EXTRACTION_AND_SEMANTIC_SEGREGATION",
        "claim_ceiling": "NOT_ENGLISH_OR_OPEN_WORLD_DOMAIN_EXTRACTION",
        "config_sha256": sha256_file(config_path),
        "reveal_sha256": sha256_file(reveal_path),
        "source": {
            "model_id": config["source"]["model_id"],
            "revision": config["source"]["revision"],
            "training_steps": 0,
            "present_at_package_execution": False,
        },
        "metrics": metrics,
        "packages": package_records,
        "isolation": {
            "result_sha256": sha256_file(output / "extraction/result.json"),
            "launcher_sha256": sha256_file(output / "extraction/launcher.json"),
            "control_result_sha256": sha256_file(output / "control_extraction/result.json"),
        },
        "artifacts": {
            "source_rows": {"path": source_path.name, "sha256": sha256_file(source_path)},
            "evaluation_rows": {"path": matrix_path.name, "sha256": sha256_file(matrix_path)},
            "source_bundle": {"path": bundle_path.name, "sha256": sha256_file(bundle_path)},
            "source_residuals": {"path": residual_path.name, "sha256": sha256_file(residual_path)},
        },
        "information_accounting": {
            **counters,
            "raw_source_prompts": len(source_rows),
            "source_bundle_bytes": bundle_path.stat().st_size,
            "source_residual_bytes": residual_path.stat().st_size,
            "final_package_bytes": sum(int(item["bytes"]) for item in package_records),
            "final_package_facts": sum(int(item["facts"]) for item in package_records),
            "source_parameters_in_final_packages": 0,
            "bridge_parameters_trained": 0,
            "source_training_steps": 0,
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "full_abi_moonshot": "OPEN",
        "unproven": [
            "autonomous open-world discovery",
            "English transfer",
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.reveal, args.output), indent=2))


if __name__ == "__main__":
    main()
