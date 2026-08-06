"""V29 universal-syntax representation successor; no model training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_direct_core import _json
from .capability_compiler_phase3_representation_bakeoff import evaluate, inventory, rows, tokenizer_type


FORMAT = "abi-capability-compiler-phase3-universal-syntax/1"


def load_protocol(root: Path, path: Path):
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_NO_TRAINING" or protocol.get("training_allowed") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("V29 governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"V29 binding changed: {relative}")
    return protocol, sha256_file(path)


def execute(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    training_rows, development_rows = rows(root, protocol)
    tokenizer = tokenizer_type(root, protocol)
    lexemes, characters = inventory(training_rows, tokenizer.split)
    printable_ascii = {bytes((value,)) for value in range(0x20, 0x7F)}
    lexemes |= printable_ascii
    characters |= printable_ascii
    specification = {"mode": "hybrid", "exposure_modulus": None}
    training = evaluate(training_rows, tokenizer.split, lexemes, characters, specification)
    development = evaluate(development_rows, tokenizer.split, lexemes, characters, specification)
    dev_chars = {char.encode("utf-8") for row in development_rows for char in row["output"]}
    unsupported = sorted(value.hex() for value in dev_chars - characters)
    curriculum_repetitions = int(protocol["syntax_curriculum"]["repetitions_per_character"])
    fixed_actions = len(lexemes | characters) + 4
    qualifying = (
        training["representable"] == 7000 and development["representable"] == 1400
        and training["source_over_limit"] == 0 and development["source_over_limit"] == 0
        and training["target_over_limit"] == 0 and development["target_over_limit"] == 0
        and not unsupported and fixed_actions <= int(protocol["qualification"]["maximum_fixed_actions"])
    )
    result: dict[str, Any] = {
        "format": "abi-capability-compiler-phase3-universal-syntax-result/1",
        "status": "PASS_REPRESENTATION_ONLY" if qualifying else "FAIL_REPRESENTATION",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "qualifying": qualifying,
        "training_performed": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "phase4_through_8": "LOCKED",
        "representation": {
            "fixed_actions": fixed_actions,
            "printable_ascii_actions": len(printable_ascii),
            "unsupported_development_characters": unsupported,
            "syntax_curriculum": {
                "teacher_generated": False,
                "knowledge_payload": False,
                "characters": len(characters),
                "repetitions_per_character": curriculum_repetitions,
                "actions": len(characters) * curriculum_repetitions
            },
            "training": training,
            "development": development
        },
        "host_change_authorized": False,
        "model_training_authorized": False,
        "decision": "If qualifying, separately preregister host-conformance feasibility before any model fit. If failing, preserve the exact limiting gate; do not add development-derived exceptions.",
        "claim_boundary": "Bounded representation feasibility only; no learned generation, transfer, performance, Phase 3 pass, or superiority claim."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("execute", "verify"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_UNIVERSAL_SYNTAX_PROTOCOL_V29.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_universal_syntax/universal_syntax_v29.json")
    args = parser.parse_args(argv); root = Path.cwd().resolve(); protocol = (root / args.protocol).resolve(); output = (root / args.output).resolve(); expected = execute(root, protocol)
    if args.command == "execute":
        if output.exists(): raise Phase3Error(f"V29 output is immutable: {output}")
        _write_immutable(output, json.dumps(expected, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    elif _json(output) != expected: raise Phase3Error("stored V29 result differs from recomputation")
    print(json.dumps({"status": expected["status"], "qualifying": expected["qualifying"], "evidence_sha256": expected["evidence_sha256"]}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
