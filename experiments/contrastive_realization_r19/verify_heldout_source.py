"""Fail-closed recomputation of R19 hidden source acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.factual_semantic_r16.public_qualification import _render_chat
from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)
from experiments.functional_realization_r18.verify import _parse_template
from experiments.linguistic_realization_r17.public_qualification import SYSTEM
from experiments.linguistic_realization_r17.verify_source import _tokenizer

from .heldout_protocol import load_bound_inputs
from .hidden_frames import heldout_rows


def verify_source(config_path: Path, reveal_path: Path, source_run: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config, reveal = load_bound_inputs(root, config_path, reveal_path)
    receipt_path = source_run / "receipt.json"
    rows_path = source_run / "source_observations.jsonl"
    receipt = json_object(receipt_path)
    if receipt.get("evidence_sha256") != evidence_hash(
        {key: item for key, item in receipt.items() if key != "evidence_sha256"}
    ):
        raise R14Error("R19 source receipt hash changed")
    source = config["source"]
    if (
        receipt.get("format") != "abi-r19-heldout-source-acquisition/1"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("reveal_sha256") != sha256_file(reveal_path)
        or receipt.get("protocol_sha256")
        != sha256_file(Path(__file__).with_name("HOLDOUT_PROTOCOL.md"))
        or receipt.get("source", {}).get("model_id") != source["model_id"]
        or receipt.get("source", {}).get("revision") != source["revision"]
        or receipt.get("source", {}).get("snapshot_path_name") != source["revision"]
        or receipt.get("source", {}).get("training_steps") != 0
        or receipt.get("source", {}).get("device") != "cuda"
        or receipt.get("source", {}).get("present_at_compilation") is not False
        or receipt.get("source", {}).get("present_at_package_execution") is not False
        or receipt.get("artifacts", {}).get("source_rows", {}).get("sha256")
        != sha256_file(rows_path)
    ):
        raise R14Error("R19 source identity changed")
    commitment = config["heldout_seed_commitment"]
    expected = {
        row["record_id"]: row
        for split in ("extraction", "evaluation")
        for row in heldout_rows(reveal["secret_hex"], commitment, split)
    }
    try:
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R14Error("R19 source rows are unreadable") from exc
    if len(rows) != 120 or len({row.get("record_id") for row in rows}) != 120:
        raise R14Error("R19 source rows changed")
    tokenizer = _tokenizer()
    counters = {
        "raw_source_prompts": 0,
        "rendered_prompt_utf8_bytes": 0,
        "rendered_prompt_token_instances": 0,
        "teacher_generated_tokens": 0,
        "teacher_output_bytes": 0,
    }
    raw_prompts: list[str] = []
    teacher_outputs: list[str] = []
    exact = {"extraction": 0, "evaluation": 0}
    totals = {"extraction": 0, "evaluation": 0}
    parseable: dict[str, int] = {}
    for row in rows:
        expected_row = expected.get(row.get("record_id"))
        if expected_row is None:
            raise R14Error("R19 source row is not registered")
        for key in (
            "record_id",
            "split",
            "signature",
            "features",
            "slots",
            "prompt",
            "expected",
        ):
            if row.get(key) != expected_row[key]:
                raise R14Error(f"R19 registered source field changed: {key}")
        rendered = _render_chat(tokenizer, SYSTEM, row["prompt"])
        tokens = tokenizer.encode(rendered, add_special_tokens=False)
        output = row.get("output")
        if (
            not isinstance(output, str)
            or row.get("rendered_prompt_sha256") != hashlib.sha256(rendered.encode()).hexdigest()
            or row.get("rendered_prompt_utf8_bytes") != len(rendered.encode())
            or row.get("input_token_count") != len(tokens)
            or row.get("output_sha256") != hashlib.sha256(output.encode()).hexdigest()
            or not isinstance(row.get("output_token_count"), int)
            or row["output_token_count"] <= 0
            or row.get("functional_exact") != (output == row["expected"])
        ):
            raise R14Error("R19 generated source evidence changed")
        split = row["split"]
        totals[split] += 1
        exact[split] += int(row["functional_exact"])
        if split == "extraction":
            signature = row["signature"]
            parseable.setdefault(signature, 0)
            parseable[signature] += int(_parse_template(output, row["slots"]) is not None)
        counters["raw_source_prompts"] += 1
        counters["rendered_prompt_utf8_bytes"] += len(rendered.encode())
        counters["rendered_prompt_token_instances"] += len(tokens)
        counters["teacher_generated_tokens"] += row["output_token_count"]
        counters["teacher_output_bytes"] += len(output.encode())
        raw_prompts.append(str(row["prompt"]))
        teacher_outputs.append(output)
    minimum = min(parseable.values()) if parseable else 0
    authorized = (
        totals == {"extraction": 72, "evaluation": 48}
        and len(parseable) == 24
        and minimum >= config["gates"]["minimum_parseable_per_signature"]
    )
    metrics = {
        "source_rows": 120,
        "extraction_rows": totals["extraction"],
        "evaluation_rows": totals["evaluation"],
        "source_functional_exact": exact["extraction"] + exact["evaluation"],
        "extraction_functional_exact": exact["extraction"],
        "evaluation_functional_exact": exact["evaluation"],
        "parseable_by_signature": parseable,
        "minimum_parseable_per_signature": minimum,
    }
    accounting = receipt.get("information_accounting", {})
    if (
        receipt.get("metrics") != metrics
        or receipt.get("compiler_authorized") is not authorized
        or receipt.get("verdict")
        != ("PASS_SOURCE_COMPILABILITY" if authorized else "FAIL_SOURCE_COMPILABILITY")
        or any(accounting.get(key) != value for key, value in counters.items())
        or accounting.get("source_parameters") != receipt["source"]["parameters"]
        or accounting.get("source_parameters", 0) <= 0
        or accounting.get("logits_stored") != 0
        or accounting.get("hidden_activations_stored") != 0
        or accounting.get("frozen_source_parameters_copied") != 0
        or accounting.get("final_imported_substrate_parameters") != 0
        or accounting.get("bridge_parameters_trained") != 0
        or accounting.get("raw_source_prompt_utf8_bytes")
        != sum(len(value.encode()) for value in raw_prompts)
        or accounting.get("unique_source_prompt_utf8_bytes")
        != sum(len(value.encode()) for value in set(raw_prompts))
        or accounting.get("unique_teacher_output_utf8_bytes")
        != sum(len(value.encode()) for value in set(teacher_outputs))
        or accounting.get("peak_gpu_memory_bytes", 0) <= 0
        or accounting.get("peak_cpu_rss_bytes_observed", 0) <= 0
        or accounting.get("elapsed_seconds", 0) <= 0
        or accounting.get("cpu_process_seconds", 0) <= 0
        or accounting.get("cpu_process_hours") != accounting.get("cpu_process_seconds") / 3600
        or accounting.get("source_load_seconds", 0) <= 0
        or accounting.get("source_inference_seconds", 0) <= 0
        or accounting.get("source_model_inference_hours")
        != accounting.get("source_inference_seconds") / 3600
        or not isinstance(accounting.get("gpu_name"), str)
        or not accounting.get("gpu_name")
        or accounting.get("external_hardware_used") is not False
        or accounting.get("source_snapshot_files", 0) <= 0
        or accounting.get("source_snapshot_disk_bytes", 0) <= 0
        or accounting.get("source_observation_artifact_bytes") != rows_path.stat().st_size
        or any(
            (source_run / name).exists()
            for name in ("source_bundle.json", "evaluation.jsonl", "extraction")
        )
    ):
        raise R14Error("R19 source claims changed")
    result = {
        "format": "abi-r19-heldout-source-strict-verification/1",
        "verdict": receipt["verdict"],
        "source_rows_verified": 120,
        "metrics": metrics,
        "information_accounting": accounting,
        "source_rows_sha256": sha256_file(rows_path),
        "receipt_sha256": sha256_file(receipt_path),
        "compiler_authorized": authorized,
        "layercake_invoked": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_source(args.config, args.reveal, args.source_run)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
