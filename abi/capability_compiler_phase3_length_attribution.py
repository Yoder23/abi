"""Read-only record-level attribution for the nine V29 length failures."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_direct_core import _json
from .capability_compiler_phase3_representation_bakeoff import inventory, rows, tokenizer_type


FORMAT = "abi-capability-compiler-phase3-length-attribution/1"


def load_protocol(root: Path, path: Path):
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY" or protocol.get("training_allowed") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("length attribution governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"length attribution binding changed: {relative}")
    return protocol, sha256_file(path)


def analyze_row(row, split, lexemes):
    source = split(row["prompt"]); output = split(row["output"]); source_counts = Counter(source)
    action_types = Counter(); fallbacks = []; actions = 1
    for piece in output:
        if source_counts[piece] == 1:
            action_types["pointer"] += 1; actions += 1
        elif piece in lexemes:
            action_types["lexeme"] += 1; actions += 1
        else:
            characters = [char.encode("utf-8") for char in piece.decode("utf-8", errors="strict")]
            action_types["character"] += len(characters); actions += len(characters)
            fallbacks.append({
                "lexeme_utf8": piece.decode("utf-8", errors="strict"),
                "lexeme_sha256": hashlib.sha256(piece).hexdigest(),
                "character_actions": len(characters),
                "excess_over_one_compact_action": max(0, len(characters) - 1),
            })
    return {
        "record_id": row["record_id"], "capability": row["capability"],
        "actions": actions, "excess_over_320": max(0, actions - 320),
        "output_utf8_bytes": len(row["output"].encode("utf-8")), "output_characters": len(row["output"]),
        "action_types": dict(action_types), "fallback_lexemes": fallbacks,
        "fallback_excess_actions": sum(value["excess_over_one_compact_action"] for value in fallbacks),
    }


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    training, development = rows(root, protocol); tokenizer = tokenizer_type(root, protocol)
    lexemes, characters = inventory(training, tokenizer.split)
    printable_ascii = {bytes((value,)) for value in range(0x20, 0x7F)}; lexemes |= printable_ascii; characters |= printable_ascii
    records = [analyze_row(row, tokenizer.split, lexemes) for row in development]
    failing = sorted((row for row in records if row["actions"] > 320), key=lambda row: (-row["actions"], row["record_id"]))
    if len(failing) != protocol["expected_failures"]:
        raise Phase3Error("V29 failing-record depth changed")
    result: dict[str, Any] = {
        "format": "abi-capability-compiler-phase3-length-attribution-result/1", "status": "PASS_READ_ONLY_ATTRIBUTION",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha}, "training_performed": False,
        "final_test_accessed": False, "phase3_certified": False, "host_changed": False,
        "failing_records": failing,
        "aggregate": {
            "records": len(failing), "maximum_actions": max(row["actions"] for row in failing),
            "total_excess_over_320": sum(row["excess_over_320"] for row in failing),
            "total_fallback_excess_actions": sum(row["fallback_excess_actions"] for row in failing),
            "all_failures_have_sufficient_fallback_excess_to_clear": all(row["fallback_excess_actions"] >= row["excess_over_320"] for row in failing),
        },
        "decision": "If every failure has enough fallback expansion to explain its excess, preregister one training-derived compact-sublexeme representation. Otherwise stop and attribute the residual to the host length contract.",
        "claim_boundary": "Read-only development length attribution; no model, host, quality, or superiority result."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("execute", "verify")); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_LENGTH_ATTRIBUTION_PROTOCOL_V31.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_length_attribution/length_attribution_v31.json"); args = parser.parse_args(argv)
    root = Path.cwd().resolve(); expected = execute(root, (root / args.protocol).resolve()); output = (root / args.output).resolve()
    if args.command == "execute":
        if output.exists(): raise Phase3Error(f"V31 output is immutable: {output}")
        _write_immutable(output, json.dumps(expected, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    elif _json(output) != expected: raise Phase3Error("stored V31 result differs from recomputation")
    print(json.dumps({"status": expected["status"], "aggregate": expected["aggregate"], "evidence_sha256": expected["evidence_sha256"]}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
