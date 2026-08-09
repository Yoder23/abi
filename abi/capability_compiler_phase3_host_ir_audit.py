"""Independently recompute host conformance for a broad acquisition IR."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error
from .capability_compiler_phase3_broad_ir import host_prompt_projection, verify_broad_ir
from .capability_compiler_phase3_native_causal_core import load_protocol
from .capability_compiler_phase3_teacher_native_core import _layercake_api, _tokenizer, controlled_prompt


PROTOCOL_FORMAT = "abi-capability-compiler-phase3-host-ir-audit/1"


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != PROTOCOL_FORMAT or protocol.get("teacher_model_loading_authorized") is not False or protocol.get("neural_training_authorized") is not False:
        raise Phase3Error("host IR audit governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"host IR audit binding changed: {relative}")
    ir_path = root / protocol["ir"]
    verify_broad_ir(ir_path)
    candidate, _ = load_protocol(root, root / protocol["candidate_protocol"])
    _, tokenizer_type, _, _ = _layercake_api(root, candidate)
    tokenizer = _tokenizer(root, candidate, tokenizer_type)
    with zipfile.ZipFile(ir_path) as archive:
        rows = [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line.strip()]
    maximum_source = int(protocol["bounds"]["maximum_source_lexemes"])
    maximum_target = int(protocol["bounds"]["maximum_target_actions"])
    failures = Counter()
    maxima = {"source_lexemes": 0, "target_actions": 0}
    projections = Counter()
    for row in rows:
        expected_prompt, expected_projection = host_prompt_projection(
            str(row["capability"]), str(row["normalized_acquisition_prompt"])
        )
        projections[expected_projection] += 1
        if row.get("source_prompt_projection") != expected_projection or row.get("host_conformant_acquisition_prompt") != expected_prompt:
            failures["projection_mismatch"] += 1
        if row.get("host_conformant_acquisition_prompt_sha256") != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest():
            failures["projection_hash_mismatch"] += 1
        source_ids, _ = tokenizer.encode_source(controlled_prompt(str(row["capability"]), expected_prompt))
        target_actions = tokenizer.encode_fixed_target(str(row["normalized_output"]))
        maxima["source_lexemes"] = max(maxima["source_lexemes"], len(source_ids))
        maxima["target_actions"] = max(maxima["target_actions"], len(target_actions))
        if row.get("host_source_lexemes") != len(source_ids):
            failures["stored_source_length_mismatch"] += 1
        if row.get("host_target_actions") != len(target_actions):
            failures["stored_target_length_mismatch"] += 1
        if len(source_ids) > maximum_source:
            failures["source_bound_overflow"] += 1
        if len(target_actions) > maximum_target:
            failures["target_bound_overflow"] += 1
        if tokenizer.decode_actions(target_actions, []) != str(row["normalized_output"]).encode("utf-8"):
            failures["target_losslessness_failure"] += 1
    return {
        "format": "abi-capability-compiler-phase3-host-ir-audit-result/1",
        "status": "PASS_HOST_CONFORMANT_IR" if not failures else "FAIL_HOST_CONFORMANT_IR",
        "ir_sha256": sha256_file(ir_path),
        "records": len(rows),
        "bounds": protocol["bounds"],
        "maxima": maxima,
        "projection_counts": dict(sorted(projections.items())),
        "failures": dict(sorted(failures.items())),
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "final_test_accessed": False,
        "claim_boundary": "Independent host-conformance audit only; no model quality or Phase 3 pass is claimed."
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve(); output = root / args.output
    if output.exists():
        raise Phase3Error("host IR audit output exists")
    result = run(root, root / args.protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
