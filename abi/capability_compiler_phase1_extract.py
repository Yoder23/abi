"""Run the preregistered Phase 1 teacher extraction with a durable journal."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Mapping

from .capability_compiler_phase1 import verify_protocol
from .capability_pipeline import canonical_json_bytes
from .hf_extraction import HuggingFaceCausalSource, evaluate_output, load_probe_catalog
from .validated_teacher_extraction import build_repair_prompt


JOURNAL_FORMAT = "abi-capability-compiler-phase1-source-attempt/1"
SUMMARY_FORMAT = "abi-capability-compiler-phase1-source-evidence/1"


class Phase1ExtractionError(RuntimeError):
    """Raised when a source attempt or resume journal violates the protocol."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _attempt_row(
    *,
    protocol_sha256: str,
    catalog_sha256: str,
    probe: Mapping[str, Any],
    attempt_index: int,
    kind: str,
    generation_prompt: str,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    ids = sample.get("authoritative_generated_token_ids")
    if (
        not isinstance(ids, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in ids)
        or len(ids) != int(sample.get("teacher_tokens", -1))
    ):
        raise Phase1ExtractionError("source attempt lacks authoritative token IDs")
    evaluator_pass, score = evaluate_output(str(sample.get("output", "")), dict(probe["evaluator"]))
    finish_reason = str(sample.get("finish_reason"))
    functional_pass = bool(evaluator_pass and finish_reason == "eos_token")
    row: dict[str, Any] = {
        "format": JOURNAL_FORMAT,
        "protocol_sha256": protocol_sha256,
        "catalog_sha256": catalog_sha256,
        "probe_id": str(probe["probe_id"]),
        "canonical_capability": str(probe["canonical_capability"]),
        "attempt_index": int(attempt_index),
        "kind": kind,
        "generation_prompt": generation_prompt,
        "generation_prompt_sha256": hashlib.sha256(generation_prompt.encode("utf-8")).hexdigest(),
        "rendered_prompt": str(sample["rendered_prompt"]),
        "rendered_prompt_sha256": hashlib.sha256(str(sample["rendered_prompt"]).encode("utf-8")).hexdigest(),
        "teacher_input_tokens": int(sample["input_tokens"]),
        "output": str(sample["output"]),
        "output_sha256": hashlib.sha256(str(sample["output"]).encode("utf-8")).hexdigest(),
        "teacher_tokens": int(sample["teacher_tokens"]),
        "teacher_token_counter": str(sample["teacher_token_counter"]),
        "authoritative_generated_token_ids": list(ids),
        "finish_reason": finish_reason,
        "generation_max_new_tokens": int(sample["generation_max_new_tokens"]),
        "functional_evaluator": dict(probe["evaluator"]),
        "functional_pass": functional_pass,
        "functional_score": float(score),
    }
    row["attempt_sha256"] = _canonical_sha(row)
    return row


def load_journal(
    path: Path, *, protocol_sha256: str, catalog_sha256: str
) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Phase1ExtractionError(f"invalid journal JSON at line {line_number}") from exc
            claimed = row.pop("attempt_sha256", None)
            actual = _canonical_sha(row)
            row["attempt_sha256"] = claimed
            if claimed != actual:
                raise Phase1ExtractionError(f"stale attempt hash at line {line_number}")
            if row.get("format") != JOURNAL_FORMAT:
                raise Phase1ExtractionError("unsupported journal row")
            if row.get("protocol_sha256") != protocol_sha256 or row.get("catalog_sha256") != catalog_sha256:
                raise Phase1ExtractionError("journal belongs to another protocol or catalog")
            key = (str(row["probe_id"]), int(row["attempt_index"]))
            if key in rows:
                raise Phase1ExtractionError(f"duplicate journal attempt: {key}")
            rows[key] = row
    return rows


def _append_rows(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def selected_attempts(
    probes: Iterable[Mapping[str, Any]],
    journal: Mapping[tuple[str, int], Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    selected: dict[str, Mapping[str, Any]] = {}
    failed: list[str] = []
    for probe in probes:
        probe_id = str(probe["probe_id"])
        attempts = [
            row
            for (candidate_id, _), row in journal.items()
            if candidate_id == probe_id
        ]
        attempts.sort(key=lambda row: int(row["attempt_index"]))
        passing = [row for row in attempts if row.get("functional_pass") is True]
        if passing:
            selected[probe_id] = passing[0]
        else:
            failed.append(probe_id)
    return selected, failed


def _process_metrics() -> tuple[int, int]:
    try:
        import psutil

        rss = int(psutil.Process().memory_info().rss)
    except Exception:
        rss = 0
    try:
        import torch

        vram = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except Exception:
        vram = 0
    return rss, vram


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
        raise Phase1ExtractionError(f"summary is immutable: {summary_path}")
    verification = verify_protocol(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    root = protocol_path.resolve().parent
    protocol_sha256 = verification["protocol_sha256"]
    catalog_sha256 = verification["catalog_sha256"]
    catalog = load_probe_catalog(root / protocol["catalog"]["path"])
    probes = sorted(
        (row for row in catalog["probes"] if row["split"] == "search"),
        key=lambda row: str(row["probe_id"]),
    )
    if len(probes) != 9_800:
        raise Phase1ExtractionError("search catalog depth changed")
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    journal = load_journal(
        journal_path,
        protocol_sha256=protocol_sha256,
        catalog_sha256=catalog_sha256,
    )
    unknown_ids = {key[0] for key in journal} - set(probe_by_id)
    if unknown_ids:
        raise Phase1ExtractionError("journal contains unknown probe IDs")

    started = time.perf_counter()
    maximum_rss, maximum_vram = _process_metrics()
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
        raise Phase1ExtractionError("runtime source manifest changed")
    requested_attention = source_spec["generation"]["attention_implementation"]
    if hasattr(source.model, "set_attn_implementation"):
        source.model.set_attn_implementation(requested_attention)
    else:
        source.model.config._attn_implementation = requested_attention
    attention = str(getattr(source.model.config, "_attn_implementation", "eager"))
    if attention != requested_attention:
        raise Phase1ExtractionError(f"attention implementation changed: {attention}")

    inference_seconds = 0.0
    batch_size = int(source_spec["generation"]["batch_size"])
    pending_initial = [row for row in probes if (str(row["probe_id"]), 0) not in journal]
    for start in range(0, len(pending_initial), batch_size):
        chunk = pending_initial[start : start + batch_size]
        requests = [
            {
                "prompt": str(row["prompt"]),
                "max_new_tokens": int(row["max_new_tokens"]),
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
                protocol_sha256=protocol_sha256,
                catalog_sha256=catalog_sha256,
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
        maximum_rss = max(maximum_rss, rss)
        maximum_vram = max(maximum_vram, vram)

    _, failed = selected_attempts(probes, journal)
    pending_repairs = [
        probe_by_id[probe_id]
        for probe_id in failed
        if (probe_id, 1) not in journal
    ]
    repair_maximum = int(protocol["repair_policy"]["max_new_tokens"])
    for start in range(0, len(pending_repairs), batch_size):
        chunk = pending_repairs[start : start + batch_size]
        prompts = []
        for probe in chunk:
            prior = journal[(str(probe["probe_id"]), 0)]
            prompts.append(
                build_repair_prompt(
                    original_prompt=str(probe["prompt"]),
                    prior_output=str(prior["output"]),
                    evaluator=probe["evaluator"],
                )
            )
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
                protocol_sha256=protocol_sha256,
                catalog_sha256=catalog_sha256,
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
        rss, vram = _process_metrics()
        maximum_rss = max(maximum_rss, rss)
        maximum_vram = max(maximum_vram, vram)

    selected, failed = selected_attempts(probes, journal)
    selected_counts = Counter(
        probe_by_id[probe_id]["canonical_capability"] for probe_id in selected
    )
    initial_passes = Counter(
        row["canonical_capability"]
        for (probe_id, attempt_index), row in journal.items()
        if attempt_index == 0 and row["functional_pass"] is True
    )
    repaired_passes = Counter(
        row["canonical_capability"]
        for (probe_id, attempt_index), row in journal.items()
        if attempt_index == 1 and row["functional_pass"] is True
    )
    length_terminations = Counter(
        row["canonical_capability"]
        for row in journal.values()
        if row["finish_reason"] == "length"
    )
    required = set(protocol["capability_mapping"].values())
    pass_depth = all(selected_counts[capability] >= 500 for capability in required)
    all_initial_present = all((str(row["probe_id"]), 0) in journal for row in probes)
    repaired_all_initial_failures = all(
        (probe_id, 1) in journal
        for probe_id in failed
        if journal[(probe_id, 0)]["functional_pass"] is False
    )
    status = "PASS_SOURCE_EVIDENCE_READY_FOR_NORMALIZATION" if pass_depth and all_initial_present else "FAIL_SOURCE_EVIDENCE_INADEQUATE"
    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "format": SUMMARY_FORMAT,
        "status": status,
        "protocol_sha256": protocol_sha256,
        "catalog_sha256": catalog_sha256,
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
            "initial_passes": dict(sorted(initial_passes.items())),
            "repaired_passes": dict(sorted(repaired_passes.items())),
            "length_terminations": dict(sorted(length_terminations.items())),
            "minimum_per_capability": 500,
            "depth_gate_pass": pass_depth,
        },
        "checks": {
            "all_initial_attempts_present": all_initial_present,
            "all_failed_initial_attempts_received_bounded_repair": repaired_all_initial_failures,
            "authoritative_token_ids_retained": all(
                len(row["authoritative_generated_token_ids"]) == row["teacher_tokens"]
                for row in journal.values()
            ),
            "length_terminated_records_selected": any(
                row["finish_reason"] == "length" for row in selected.values()
            ),
            "validation_teacher_outputs_generated": False,
            "final_teacher_outputs_generated": False,
            "candidate_training_performed": False,
        },
        "accounting": {
            "source_load_seconds": round(source_load_seconds, 6),
            "source_inference_seconds_this_process": round(inference_seconds, 6),
            "wall_seconds_this_process": round(elapsed, 6),
            "peak_process_rss_bytes": maximum_rss,
            "peak_cuda_allocated_bytes": maximum_vram,
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
        "phase1_ir_constructed": False,
        "training_authorized": False,
    }
    summary["evidence_sha256"] = _canonical_sha(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_bytes((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE1_PROTOCOL_V1.json")
    parser.add_argument("--journal", default="results/abi_capability_compiler_phase1/v1/source_search_attempts.jsonl")
    parser.add_argument("--summary", default="results/abi_capability_compiler_phase1/v1/source_search_summary.json")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_extraction(
            protocol_path=Path(args.protocol).resolve(),
            journal_path=Path(args.journal).resolve(),
            summary_path=Path(args.summary).resolve(),
        )
    except Exception as exc:
        raise SystemExit(f"Phase 1 extraction failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
