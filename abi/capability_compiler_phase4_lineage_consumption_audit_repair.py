"""Implementation-only repair of the V560 no-final-access boolean polarity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from . import capability_compiler_phase4_lineage_consumption_audit as base
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-lineage-consumption-audit-repair/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_IMPLEMENTATION_ONLY_BOOLEAN_POLARITY_REPAIR"
        or protocol.get("scientific_fields_changed") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("consumption-audit repair governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"consumption-audit repair binding changed: {relative}")
    return protocol, sha256_file(path)


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    result = base.run(root, root / protocol["base_protocol"])
    if result["status"] != "FAIL_CLOSED_CONSUMPTION_LINEAGE":
        raise Phase3Error("V561 failure signature changed")
    original = dict(result["gates"])
    positive = {key: value for key, value in original.items() if key != "final_test_accessed"}
    corrected = dict(positive)
    corrected["final_test_not_accessed"] = result["final_test_accessed"] is False
    passed = all(value is True for value in corrected.values())
    result["format"] = "abi-capability-compiler-phase4-lineage-consumption-audit-repair-result/1"
    result["status"] = "PASS_CONSUMED_INFORMATION_LINEAGE_FRONTIER_PROTOCOL_OPEN" if passed else "FAIL_CLOSED_CONSUMPTION_LINEAGE_REPAIR"
    result["protocol_sha256"] = protocol_sha
    result["superseded_v561_gate_vector"] = original
    result["gates"] = corrected
    result["implementation_repair"] = {
        "scientific_fields_changed": False,
        "from": "a false final_test_accessed diagnostic was incorrectly included in all(gates.values())",
        "to": "the positive final_test_not_accessed assertion equals true",
    }
    result["decision"] = "The consumed-information lineage passes after correcting only boolean polarity. Use 9596 unique consumed source attempts and 294212 unique teacher-output tokens for Phase 4 budget design."
    result["claim_boundary"] = "Implementation-only audit repair; no training, minimum-information, Phase 4 certificate, final-test result, or superiority claim."
    result.pop("evidence_sha256", None)
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / args.output
    if output.exists():
        raise Phase3Error(f"immutable repair output exists: {output}")
    result = run(root, root / args.protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
