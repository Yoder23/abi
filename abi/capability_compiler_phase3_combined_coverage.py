"""Audit native structural coverage of original plus broad acquisition IR."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

from .capability_compiler_phase2_common import CAPABILITIES, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error
from .capability_compiler_phase3_acquisition_coverage import _coverage, _ngrams
from .capability_compiler_phase3_broad_ir import verify_broad_ir
from .capability_compiler_phase3_native_causal_core import load_protocol as load_candidate_protocol
from .capability_compiler_phase3_segment_router import _semantic_segments
from .capability_compiler_phase3_teacher_native_core import (
    _examples as original_examples,
    _json,
    _layercake_api,
    _tokenizer,
    controlled_prompt,
)


PROTOCOL_FORMAT = "abi-capability-compiler-phase3-combined-coverage/1"


def broad_examples(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    maximum_source_lexemes: int,
    maximum_target_actions: int,
) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if row.get("source_prompt_projection") not in {
            "full_normalized_acquisition_prompt_host_bound_selected",
            "fluent_realization_event_fields_without_redundant_context",
            "phase1_task_body_without_targeted_search_wrapper",
        }:
            raise Phase3Error("broad source projection changed")
        prompt = controlled_prompt(
            str(row["capability"]), str(row["host_conformant_acquisition_prompt"])
        )
        source_ids, _ = tokenizer.encode_source(prompt)
        target = tokenizer.encode_fixed_target(str(row["normalized_output"]))
        if len(source_ids) > maximum_source_lexemes or len(target) > maximum_target_actions:
            raise Phase3Error("broad example exceeds fixed host bound")
        if tokenizer.decode_actions(target, []) != str(row["normalized_output"]).encode("utf-8"):
            raise Phase3Error("broad target is not native-lossless")
        examples.append({
            "record_id": str(row["ir_record_id"]),
            "capability": str(row["capability"]),
            "source_ids": source_ids,
            "target_actions": target,
        })
    return examples


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("status") != "PREREGISTERED_NO_MODEL_COMBINED_COVERAGE_AUDIT"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("combined coverage governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"combined coverage binding changed: {relative}")
    candidate_protocol, _ = load_candidate_protocol(
        root, (root / protocol["candidate_protocol"]).resolve()
    )
    _, tokenizer_type, _, _ = _layercake_api(root, candidate_protocol)
    tokenizer = _tokenizer(root, candidate_protocol, tokenizer_type)
    original = original_examples(root, candidate_protocol, tokenizer)
    broad_path = (root / protocol["broad_ir"]).resolve()
    verify_broad_ir(broad_path)
    with zipfile.ZipFile(broad_path) as archive:
        broad_rows = [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line.strip()]
    architecture = candidate_protocol["architecture"]
    broad = broad_examples(
        broad_rows,
        tokenizer,
        maximum_source_lexemes=int(architecture["maximum_source_lexemes"]),
        maximum_target_actions=int(architecture["maximum_target_actions"]),
    )
    acquisition = original + broad
    if len({row["record_id"] for row in acquisition}) != len(acquisition):
        raise Phase3Error("combined acquisition record IDs overlap")

    probes = development_probes((root / candidate_protocol["development_catalog"]).resolve())
    teacher = {
        str(row["probe_id"]): row
        for row in map(json.loads, open(root / candidate_protocol["teacher_reference"], encoding="utf-8"))
    }
    heldout = []
    for probe in probes:
        capability = str(probe["canonical_capability"])
        source, _ = tokenizer.encode_source(
            controlled_prompt(capability, _semantic_segments(str(probe["prompt"]))[-1])
        )
        target = tokenizer.encode_fixed_target(str(teacher[str(probe["probe_id"])] ["output"]))
        heldout.append({
            "record_id": str(probe["probe_id"]),
            "capability": capability,
            "source_ids": source,
            "target_actions": target,
        })

    reports: dict[str, Any] = {}
    for field in ("source_ids", "target_actions"):
        reports[field] = {}
        for width in (1, 2, 3, 4):
            known = {name: set() for name in CAPABILITIES}
            for row in acquisition:
                known[str(row["capability"])].update(_ngrams(row[field], width))
            reports[field][str(width)] = _coverage(heldout, known, field, width)
    source_trigram = reports["source_ids"]["3"]["overall"]["coverage"]
    target_fourgram = reports["target_actions"]["4"]["overall"]["coverage"]
    threshold = float(protocol["decision_rule"]["minimum_coverage"])
    counts = Counter(str(row["capability"]) for row in acquisition)
    return {
        "format": PROTOCOL_FORMAT,
        "status": (
            "PASS_COMBINED_COVERAGE_GATE"
            if source_trigram >= threshold and target_fourgram >= threshold
            else "FAIL_COMBINED_COVERAGE_GATE"
        ),
        "records": {
            "original_acquisition": len(original),
            "broad_acquisition": len(broad),
            "combined_acquisition": len(acquisition),
            "heldout": len(heldout),
            "combined_per_capability": dict(sorted(counts.items())),
        },
        "ngram_coverage": reports,
        "headline": {
            "source_trigram_coverage": source_trigram,
            "target_fourgram_coverage": target_fourgram,
            "minimum_coverage": threshold,
            "gate_pass": source_trigram >= threshold and target_fourgram >= threshold,
        },
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = (root / args.output).resolve()
    if output.exists():
        raise Phase3Error("combined coverage output exists")
    result = run(root, (root / args.protocol).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
