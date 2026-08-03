"""Verify and execute the bounded Phase 1 V2 abstention supplement."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

from .capability_compiler_phase1_abstention_catalog import (
    EXTRA_ABSTENTION_CONSTRUCTIONS,
    SUPPLEMENT_RECORDS,
)
from .capability_compiler_phase1_extract import (
    _append_rows,
    _attempt_row,
    _canonical_sha,
    _gpu_identity,
    _process_metrics,
    _sha256_file,
    load_journal,
)
from .hf_extraction import HuggingFaceCausalSource, load_probe_catalog


PROTOCOL_FORMAT = "abi-capability-compiler-phase1-abstention-protocol/2"


class AbstentionSuccessorError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AbstentionSuccessorError(message)


def verify_protocol(path: Path) -> dict[str, Any]:
    path = path.resolve()
    root = path.parent
    protocol = json.loads(path.read_text(encoding="utf-8"))
    _require(protocol.get("format") == PROTOCOL_FORMAT, "unsupported protocol format")
    _require(protocol.get("status") == "PREREGISTERED_BEFORE_V2_TEACHER_OUTPUTS", "retrospective protocol status")
    for binding in (
        protocol["parent_phase1_protocol"],
        protocol["v1_failure_decision"],
        protocol["v1_evidence"]["journal"],
        protocol["v1_evidence"]["summary"],
        protocol["fresh_catalog"],
        protocol["fresh_catalog"]["generator"],
    ):
        _require(_sha256_file(root / binding["path"]) == binding["sha256"], f"stale binding: {binding['path']}")
    parent = json.loads((root / protocol["parent_phase1_protocol"]["path"]).read_text(encoding="utf-8"))
    catalog = load_probe_catalog(root / protocol["fresh_catalog"]["path"])
    _require(len(catalog["probes"]) == SUPPLEMENT_RECORDS == 400, "fresh catalog depth changed")
    _require(all(row["capability"] == "abstention" and row["split"] == "search" for row in catalog["probes"]), "supplement scope expanded")
    _require(all(set(EXTRA_ABSTENTION_CONSTRUCTIONS).issubset(row["evaluator"]["values"]) for row in catalog["probes"]), "V2 evaluator construction changed")
    parent_catalog = load_probe_catalog(root / parent["catalog"]["path"])
    prior_prompts = {hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest() for row in parent_catalog["probes"]}
    fresh_prompts = {hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest() for row in catalog["probes"]}
    _require(len(fresh_prompts) == 400 and not prior_prompts & fresh_prompts, "fresh prompts overlap V1")
    prior_families = {row["phase1_template_family"] for row in parent_catalog["probes"]}
    fresh_families = {row["phase1_template_family"] for row in catalog["probes"]}
    _require(not prior_families & fresh_families, "fresh template family overlaps V1")
    selection = protocol["selection"]
    _require(selection["v1_accepted_records_retained"] == 237, "V1 pass count changed")
    _require(selection["minimum_fresh_v2_passes"] == 263, "V2 pass floor changed")
    _require(selection["repair_rounds"] == 0, "repairs became authorized")
    _require(selection["other_capability_generation"] == 0, "scope expanded beyond abstention")
    _require(protocol["evaluation_change"]["v1_failures_relabelled"] is False, "V1 failures may be relabelled")
    return {
        "status": "PASS",
        "protocol_sha256": _sha256_file(path),
        "catalog_sha256": protocol["fresh_catalog"]["sha256"],
        "fresh_prompts": len(fresh_prompts),
        "minimum_fresh_passes": 263,
        "training_authorized": False,
    }


def run(protocol_path: Path) -> dict[str, Any]:
    verification = verify_protocol(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    root = protocol_path.resolve().parent
    journal_path = root / protocol["outputs"]["journal"]
    summary_path = root / protocol["outputs"]["summary"]
    if summary_path.exists():
        raise AbstentionSuccessorError(f"summary is immutable: {summary_path}")
    catalog = load_probe_catalog(root / protocol["fresh_catalog"]["path"])
    probes = sorted(catalog["probes"], key=lambda row: row["probe_id"])
    journal = load_journal(
        journal_path,
        protocol_sha256=verification["protocol_sha256"],
        catalog_sha256=verification["catalog_sha256"],
    )
    source_spec = protocol["source"]
    started = time.perf_counter()
    rss_peak, vram_peak = _process_metrics()
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
    load_seconds = time.perf_counter() - load_started
    _require(source.source_manifest["source_manifest_sha256"] == source_spec["source_manifest_sha256"], "runtime source changed")
    source.model.set_attn_implementation(source_spec["attention_implementation"])
    _require(source.model.config._attn_implementation == "eager", "runtime attention changed")
    pending = [row for row in probes if (row["probe_id"], 0) not in journal]
    inference_seconds = 0.0
    batch_size = int(source_spec["batch_size"])
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        requests = [
            {"prompt": row["prompt"], "max_new_tokens": row["max_new_tokens"], "temperature": 0.0, "seed": row["seed"]}
            for row in chunk
        ]
        inference_started = time.perf_counter()
        samples = source.generate_batch(requests)
        inference_seconds += time.perf_counter() - inference_started
        rows = [
            _attempt_row(
                protocol_sha256=verification["protocol_sha256"],
                catalog_sha256=verification["catalog_sha256"],
                probe=probe,
                attempt_index=0,
                kind="fresh_v2_initial",
                generation_prompt=probe["prompt"],
                sample=sample,
            )
            for probe, sample in zip(chunk, samples, strict=True)
        ]
        _append_rows(journal_path, rows)
        for row in rows:
            journal[(row["probe_id"], 0)] = row
        rss, vram = _process_metrics()
        rss_peak, vram_peak = max(rss_peak, rss), max(vram_peak, vram)
    _require(len(journal) == 400, "V2 journal depth changed")
    passes = sum(row["functional_pass"] is True for row in journal.values())
    selected_total = 237 + passes
    status = "PASS_ABSTENTION_DEPTH_READY_FOR_IR" if passes >= 263 else "FAIL_ABSTENTION_DEPTH_INADEQUATE"
    summary: dict[str, Any] = {
        "format": "abi-capability-compiler-phase1-abstention-evidence/2",
        "status": status,
        "protocol_sha256": verification["protocol_sha256"],
        "catalog_sha256": verification["catalog_sha256"],
        "source_manifest": source.source_manifest,
        "source_runtime": source.source_inference_runtime,
        "attention_implementation": source.model.config._attn_implementation,
        "gpu_identity": _gpu_identity(),
        "journal": {"path": str(journal_path.resolve()), "sha256": _sha256_file(journal_path), "bytes": journal_path.stat().st_size, "attempts": len(journal)},
        "selection": {"v1_accepted": 237, "v2_passes": passes, "v2_failures": 400 - passes, "combined_eligible": selected_total, "minimum_combined": 500, "gate_pass": selected_total >= 500},
        "checks": {"v1_failed_outputs_reclassified": 0, "repair_attempts": 0, "validation_outputs_generated": False, "final_outputs_generated": False, "candidate_training_performed": False, "all_token_ids_authoritative": all(len(row["authoritative_generated_token_ids"]) == row["teacher_tokens"] for row in journal.values()), "selected_length_terminations": sum(row["finish_reason"] == "length" and row["functional_pass"] for row in journal.values())},
        "accounting": {"source_load_seconds": round(load_seconds, 6), "source_inference_seconds": round(inference_seconds, 6), "wall_seconds": round(time.perf_counter() - started, 6), "peak_process_rss_bytes": rss_peak, "peak_cuda_allocated_bytes": vram_peak, "teacher_input_tokens": sum(row["teacher_input_tokens"] for row in journal.values()), "authoritative_teacher_tokens": sum(row["teacher_tokens"] for row in journal.values()), "raw_generation_prompt_bytes": sum(len(row["generation_prompt"].encode("utf-8")) for row in journal.values()), "raw_teacher_output_bytes": sum(len(row["output"].encode("utf-8")) for row in journal.values()), "stored_logits": 0, "stored_activations": 0, "copied_source_parameters": 0, "bridge_or_student_parameters": 0},
        "phase1_ir_constructed": False,
        "training_authorized": False,
    }
    summary["evidence_sha256"] = _canonical_sha(summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_bytes((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE1_ABSTENTION_PROTOCOL_V2.json")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    path = Path(args.protocol).resolve()
    result = run(path) if args.run else verify_protocol(path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
