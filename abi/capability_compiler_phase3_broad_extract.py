"""Extract the preregistered Phase 3 broad-English search records on GPU."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

from .capability_compiler_phase1_extract import (
    Phase1ExtractionError,
    _append_rows,
    _attempt_row,
    _process_metrics,
    _sha256_file,
    load_journal,
    selected_attempts,
)
from .capability_pipeline import canonical_json_bytes
from .hf_extraction import HuggingFaceCausalSource, load_probe_catalog
from .validated_teacher_extraction import build_repair_prompt


PROTOCOL_FORMAT = "abi-capability-compiler-phase3-broad-extraction/1"
SUMMARY_FORMAT = "abi-capability-compiler-phase3-broad-source-evidence/1"


class BroadExtractionError(RuntimeError):
    """Raised when the broad extraction contract or runtime is violated."""


def _group_by_generation_budget(
    probes: Iterable[dict[str, Any]],
) -> list[tuple[int, list[dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for probe in probes:
        grouped.setdefault(int(probe["max_new_tokens"]), []).append(probe)
    return [(maximum, grouped[maximum]) for maximum in sorted(grouped)]


def _repair_candidates(
    failed: Iterable[str],
    probe_by_id: dict[str, dict[str, Any]],
    journal: dict[tuple[str, int], dict[str, Any]],
    *,
    maximum_rounds: int,
) -> list[dict[str, Any]]:
    if maximum_rounds not in {0, 1}:
        raise BroadExtractionError("repair rounds must be zero or one")
    if maximum_rounds == 0:
        return []
    return [probe_by_id[probe_id] for probe_id in failed if (probe_id, 1) not in journal]


def _canonical_sha(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def verify_extraction_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise BroadExtractionError("unsupported extraction protocol")
    root = path.resolve().parent
    for relative, expected in protocol.get("bindings", {}).items():
        target = root / relative
        if not target.is_file() or _sha256_file(target) != expected:
            raise BroadExtractionError(f"binding mismatch: {relative}")
    catalog_path = root / protocol["catalog"]["path"]
    if _sha256_file(catalog_path) != protocol["catalog"]["sha256"]:
        raise BroadExtractionError("catalog binding mismatch")
    catalog = load_probe_catalog(catalog_path)
    search = [row for row in catalog["probes"] if row["split"] == "search"]
    if len(search) != int(protocol["catalog"]["search_probe_count"]):
        raise BroadExtractionError("search probe depth changed")
    if any(row["split"] != "search" for row in search):
        raise BroadExtractionError("non-search row crossed extraction boundary")
    return {
        "protocol_sha256": _sha256_file(path),
        "catalog_sha256": _sha256_file(catalog_path),
        "search_probe_count": len(search),
    }


def _gpu_identity() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def run_extraction(
    *, protocol_path: Path, journal_path: Path, summary_path: Path
) -> dict[str, Any]:
    if summary_path.exists():
        raise BroadExtractionError(f"summary is immutable: {summary_path}")
    verified = verify_extraction_protocol(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    root = protocol_path.resolve().parent
    catalog = load_probe_catalog(root / protocol["catalog"]["path"])
    capability_mapping = protocol["capability_mapping"]
    probes = []
    for source_row in catalog["probes"]:
        if source_row["split"] != "search":
            continue
        row = dict(source_row)
        row["canonical_capability"] = capability_mapping[row["capability"]]
        probes.append(row)
    probes.sort(key=lambda row: str(row["probe_id"]))
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    protocol_sha = verified["protocol_sha256"]
    catalog_sha = verified["catalog_sha256"]
    journal = load_journal(
        journal_path,
        protocol_sha256=protocol_sha,
        catalog_sha256=catalog_sha,
    )
    if {key[0] for key in journal} - set(probe_by_id):
        raise BroadExtractionError("journal contains unknown probe IDs")

    started = time.perf_counter()
    peak_rss, peak_vram = _process_metrics()
    source_spec = protocol["source"]
    load_started = time.perf_counter()
    source = HuggingFaceCausalSource(
        source_spec["model"],
        revision=source_spec["revision"],
        license_id=source_spec["license"],
        device="cuda",
        local_files_only=True,
        trust_remote_code=False,
        use_chat_template=True,
        load_in_8bit=False,
    )
    source_load_seconds = time.perf_counter() - load_started
    if source.source_manifest["source_manifest_sha256"] != source_spec["source_manifest_sha256"]:
        raise BroadExtractionError("runtime source manifest changed")
    requested_attention = source_spec["generation"]["attention_implementation"]
    if hasattr(source.model, "set_attn_implementation"):
        source.model.set_attn_implementation(requested_attention)
    else:
        source.model.config._attn_implementation = requested_attention
    attention = str(getattr(source.model.config, "_attn_implementation", "eager"))
    if attention != requested_attention:
        raise BroadExtractionError("attention implementation changed")

    inference_seconds = 0.0
    batch_size = int(source_spec["generation"]["batch_size"])
    pending = [row for row in probes if (str(row["probe_id"]), 0) not in journal]
    for maximum, group in _group_by_generation_budget(pending):
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            requests = [
                {
                    "prompt": str(row["prompt"]),
                    "max_new_tokens": maximum,
                    "temperature": 0.0,
                    "seed": int(row["seed"]),
                }
                for row in chunk
            ]
            batch_started = time.perf_counter()
            samples = source.generate_batch(requests)
            inference_seconds += time.perf_counter() - batch_started
            completed = [
                _attempt_row(
                    protocol_sha256=protocol_sha,
                    catalog_sha256=catalog_sha,
                    probe=probe,
                    attempt_index=0,
                    kind="initial",
                    generation_prompt=str(probe["prompt"]),
                    sample=sample,
                )
                for probe, sample in zip(chunk, samples, strict=True)
            ]
            _append_rows(journal_path, completed)
            for row in completed:
                journal[(str(row["probe_id"]), 0)] = row
            rss, vram = _process_metrics()
            peak_rss, peak_vram = max(peak_rss, rss), max(peak_vram, vram)

    _, failed = selected_attempts(probes, journal)
    repairs = _repair_candidates(
        failed,
        probe_by_id,
        journal,
        maximum_rounds=int(protocol["repair_policy"]["maximum_rounds"]),
    )
    repair_maximum = int(protocol["repair_policy"]["max_new_tokens"])
    for start in range(0, len(repairs), batch_size):
        chunk = repairs[start : start + batch_size]
        prompts = [
            build_repair_prompt(
                original_prompt=str(probe["prompt"]),
                prior_output=str(journal[(str(probe["probe_id"]), 0)]["output"]),
                evaluator=probe["evaluator"],
            )
            for probe in chunk
        ]
        requests = [
            {
                "prompt": prompt,
                "max_new_tokens": repair_maximum,
                "temperature": 0.0,
                "seed": int(probe["seed"]) + 1,
            }
            for probe, prompt in zip(chunk, prompts, strict=True)
        ]
        batch_started = time.perf_counter()
        samples = source.generate_batch(requests)
        inference_seconds += time.perf_counter() - batch_started
        completed = [
            _attempt_row(
                protocol_sha256=protocol_sha,
                catalog_sha256=catalog_sha,
                probe=probe,
                attempt_index=1,
                kind="same_teacher_closed_evaluator_repair",
                generation_prompt=prompt,
                sample=sample,
            )
            for probe, prompt, sample in zip(chunk, prompts, samples, strict=True)
        ]
        _append_rows(journal_path, completed)
        for row in completed:
            journal[(str(row["probe_id"]), 1)] = row

    selected, failed = selected_attempts(probes, journal)
    selected_counts = Counter(probe_by_id[key]["canonical_capability"] for key in selected)
    required = set(capability_mapping.values())
    depth = int(protocol["selection"]["minimum_per_capability"])
    depth_pass = all(selected_counts[name] >= depth for name in required)
    all_initial = all((str(row["probe_id"]), 0) in journal for row in probes)
    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "format": SUMMARY_FORMAT,
        "status": "PASS_SOURCE_EVIDENCE_READY_FOR_NORMALIZATION" if depth_pass and all_initial else "FAIL_SOURCE_EVIDENCE_INADEQUATE",
        "protocol_sha256": protocol_sha,
        "catalog_sha256": catalog_sha,
        "source_manifest": source.source_manifest,
        "source_runtime": source.source_inference_runtime,
        "attention_implementation": attention,
        "gpu_identity": _gpu_identity(),
        "journal": {
            "path": str(journal_path.resolve()),
            "sha256": _sha256_file(journal_path),
            "bytes": journal_path.stat().st_size,
            "attempts": len(journal),
            "initial_attempts": sum(1 for key in journal if key[1] == 0),
            "repair_attempts": sum(1 for key in journal if key[1] == 1),
        },
        "selection": {
            "eligible_records": len(selected),
            "failed_probe_ids": sorted(failed),
            "selected_counts": dict(sorted(selected_counts.items())),
            "minimum_per_capability": depth,
            "depth_gate_pass": depth_pass,
        },
        "checks": {
            "all_initial_attempts_present": all_initial,
            "authoritative_token_ids_retained": all(len(row["authoritative_generated_token_ids"]) == row["teacher_tokens"] for row in journal.values()),
            "validation_teacher_outputs_generated": False,
            "final_teacher_outputs_generated": False,
            "candidate_training_performed": False,
        },
        "accounting": {
            "source_load_seconds": round(source_load_seconds, 6),
            "source_inference_seconds_this_process": round(inference_seconds, 6),
            "wall_seconds_this_process": round(elapsed, 6),
            "peak_process_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": peak_vram,
            "raw_generation_prompt_bytes": sum(len(row["generation_prompt"].encode("utf-8")) for row in journal.values()),
            "raw_teacher_output_bytes": sum(len(row["output"].encode("utf-8")) for row in journal.values()),
            "teacher_input_tokens": sum(row["teacher_input_tokens"] for row in journal.values()),
            "authoritative_teacher_tokens": sum(row["teacher_tokens"] for row in journal.values()),
            "stored_generated_token_ids": sum(len(row["authoritative_generated_token_ids"]) for row in journal.values()),
            "stored_logits": 0,
            "stored_activations": 0,
            "copied_source_parameters": 0,
            "bridge_or_student_parameters": 0,
        },
        "training_authorized": False,
    }
    summary["evidence_sha256"] = _canonical_sha(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_extraction(
        protocol_path=Path(args.protocol).resolve(),
        journal_path=Path(args.journal).resolve(),
        summary_path=Path(args.summary).resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
