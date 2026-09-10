"""Recompute R16 imported-information and resource accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    write_json_once,
)


def account(run_dir: Path) -> dict[str, Any]:
    receipt = json_object(run_dir / "receipt.json")
    source_rows = [
        json.loads(line)
        for line in (run_dir / receipt["artifacts"]["source_rows"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if not source_rows:
        raise R14Error("R16 accounting source rows are empty")
    questions = {str(row["question"]) for row in source_rows}
    residual_path = run_dir / receipt["artifacts"]["source_residuals"]["path"]
    tensors = load_file(str(residual_path), device="cpu")
    if set(tensors) != {"residuals"}:
        raise R14Error("R16 accounting residual inventory changed")
    result = {
        "format": "abi-r16-imported-information-accounting/1",
        "claim_ceiling": "R16_BOUNDED_FACTUAL_ACQUISITION_ONLY",
        "raw_source_prompts": len(source_rows),
        "unique_source_prompts": len(questions),
        "unique_prompt_utf8_bytes": sum(len(value.encode()) for value in questions),
        "teacher_generated_tokens": receipt["information_accounting"]["generated_tokens"],
        "teacher_generated_output_bytes": sum(
            len(str(row["completion"]).encode()) for row in source_rows
        ),
        "candidate_score_values_imported": receipt["information_accounting"]["candidate_scores"],
        "hidden_activation_values_stored_for_evidence": int(tensors["residuals"].numel()),
        "hidden_activation_bytes_stored_for_evidence": int(
            tensors["residuals"].numel() * tensors["residuals"].element_size()
        ),
        "hidden_activations_consumed_by_compiler": 0,
        "source_bundle_disk_bytes": (run_dir / "source_bundle.json").stat().st_size,
        "source_residual_evidence_disk_bytes": residual_path.stat().st_size,
        "final_package_disk_bytes": sum(
            path.stat().st_size for path in (run_dir / "extraction/packages").glob("*.abipkg")
        ),
        "final_structured_fact_records": receipt["information_accounting"]["final_package_facts"],
        "frozen_source_parameters_in_final_packages": 0,
        "bridge_parameters_trained": 0,
        "source_training_steps": 0,
        "end_to_end_acquisition_seconds": receipt["information_accounting"]["elapsed_seconds"],
        "source_inference_seconds": "NOT_SEPARATELY_MEASURED",
        "peak_gpu_memory_bytes": receipt["information_accounting"]["peak_gpu_memory_bytes"],
        "peak_cpu_ram_bytes": "NOT_MEASURED",
        "external_hardware_used": False,
        "gpu_used": True,
    }
    if result["teacher_generated_output_bytes"] != receipt["information_accounting"]["output_bytes"]:
        raise R14Error("R16 accounting output bytes changed")
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = account(args.run_dir)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
