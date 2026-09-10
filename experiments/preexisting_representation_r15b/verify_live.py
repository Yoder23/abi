"""Fresh live replay of the frozen R15B source, isolation, and recipients."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from experiments.native_isa_r11.core import load_package
from experiments.native_isa_r11.run import _host_matrix

from .isolation import run_wsl_isolated_extraction
from .protocol import evaluation_rows, heldout_capabilities
from .public_extraction import extract_with_source, load_source
from .verify import _jsonl, verify


def _scientific_extraction(result: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "format",
        "labels",
        "margins",
        "representation_sha256",
        "old_root_present",
        "windows_mount_present",
        "network_namespace_isolated",
        "reveal_files_present",
        "prompts_consumed",
        "answers_consumed",
        "semantic_labels_consumed",
        "candidate_search",
    }
    return {key: result[key] for key in sorted(keys)}


def run_live(config_path: Path, reveal_path: Path, run_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R15B live output exists: {output}")
    strict = verify(config_path, reveal_path, run_dir)
    config = json_object(config_path)
    reveal = json_object(reveal_path)
    original = json_object(run_dir / "receipt.json")
    heldout = heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    evaluations = evaluation_rows(config, heldout)
    recipient_rows = [
        rows[: int(config["data"]["recipient_rows_per_capability"])] for rows in evaluations
    ]
    output.mkdir(parents=True)
    source_config = config["source"]
    tokenizer, model, digit_ids, output_rows = load_source(
        model_id=str(source_config["model_id"]),
        model_revision=str(source_config["revision"]),
    )
    live_source = []
    for item in heldout:
        _residuals, observations, _metrics = extract_with_source(
            tokenizer,
            model,
            digit_ids,
            output_rows,
            max_new_tokens=int(source_config["max_new_tokens"]),
            slot_order=item.slot_order,
        )
        live_source.extend(
            {"capability_id": item.capability.capability_id, **row} for row in observations
        )
    original_ref = original["source"]["observations"]
    original_source = _jsonl(
        run_dir / original_ref["path"], original_ref["sha256"], int(original_ref["rows"])
    )
    if live_source != original_source:
        raise R14Error("R15B live source replay changed")
    source_path = output / "source_observations.jsonl"
    write_jsonl_once(source_path, live_source)
    del model, tokenizer, output_rows
    gc.collect()
    torch.cuda.empty_cache()

    extraction_records = []
    for index, source_item in enumerate(original["source"]["capability_receipts"]):
        destination = output / "extractions" / f"anonymous-{index:02d}"
        replay = run_wsl_isolated_extraction(
            config_path.parents[3],
            representation=run_dir / source_item["bundle"]["path"],
            destination=destination,
            distribution=str(config["physical_extraction"]["distribution"]),
        )
        original_result = json_object(
            run_dir / original["isolated_extractions"][index]["path"] / "result.json"
        )
        if _scientific_extraction(replay["result"]) != _scientific_extraction(original_result):
            raise R14Error("R15B live physical extraction changed")
        extraction_records.append(
            {
                "path": str(destination.relative_to(output)),
                "result_sha256": sha256_file(destination / "result.json"),
                "scientific_evidence_sha256": evidence_hash(
                    _scientific_extraction(replay["result"])
                ),
            }
        )

    manifest = json_object(run_dir / "package_manifest.json")
    _, before = load_package(run_dir / "packages" / manifest["before"]["path"])
    transitions = [
        load_package(run_dir / "packages" / item["path"])[1] for item in manifest["after"]
    ]
    codec_receipt = json_object(config_path.parents[3] / config["codec_freeze"]["receipt"])
    codec_tensors = load_file(
        str(config_path.parents[3] / config["codec_freeze"]["tensors"]), device="cpu"
    )
    capabilities = [item.capability for item in heldout]
    recipient_records = []
    for original_worker in original["recipient_workers"]:
        host = str(original_worker["host"])
        rows, host_receipt = _host_matrix(
            host,
            config,
            codec_tensors,
            codec_receipt,
            capabilities,
            recipient_rows,
            before,
            transitions,
            manifest,
        )
        worker = json_object(run_dir / original_worker["path"])
        row_ref = worker["observations"]
        original_rows = _jsonl(run_dir / row_ref["path"], row_ref["sha256"], int(row_ref["rows"]))
        if rows != original_rows:
            raise R14Error(f"R15B live recipient replay changed: {host}")
        host_dir = output / "recipients" / host
        host_dir.mkdir(parents=True)
        rows_path = host_dir / "observations.jsonl"
        write_jsonl_once(rows_path, rows)
        recipient_records.append(
            {
                "host": host,
                "rows": len(rows),
                "rows_sha256": sha256_file(rows_path),
                "model_state_sha256": host_receipt["model_state_sha256_after"],
                "codec_sha256": host_receipt["codec_sha256_after"],
            }
        )
    result = {
        "format": "abi-r15b-live-verification/1",
        "verdict": "PASS",
        "strict_claim": strict["claim"],
        "source_rows_replayed_byte_exact": len(live_source),
        "isolated_extractions_replayed": len(extraction_records),
        "recipient_rows_replayed_byte_exact": sum(item["rows"] for item in recipient_records),
        "source_rows_sha256": sha256_file(source_path),
        "extractions": extraction_records,
        "recipients": recipient_records,
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_live(args.config, args.reveal, args.run_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
