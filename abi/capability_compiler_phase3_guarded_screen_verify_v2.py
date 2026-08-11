"""Corrected hostile verifier for V494 with a guaranteed route mutation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_guarded_screen_verify import (
    _artifact_markers,
    _json,
    _jsonl,
    _must_reject,
    verify_payload,
)


FORMAT = "abi-capability-compiler-phase3-guarded-screen-hostile-verifier-v2/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_CORRECTED_HOSTILE_READ_ONLY_VERIFICATION"
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("corrected hostile-verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"corrected hostile-verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def choose_wrong_route(actual: str) -> str:
    return next(capability for capability in CAPABILITIES if capability != actual)


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable corrected-verifier output exists: {output}")
    probes = development_probes(root / protocol["development"]["catalog"])
    rows = _jsonl(root / protocol["evidence"]["outputs"])
    teacher = _jsonl(root / protocol["development"]["teacher_reference"])
    parent = _jsonl(root / protocol["parent"]["development_outputs"])
    raw_result = _json(root / protocol["evidence"]["result"])
    manifest = _json(root / protocol["evidence"]["manifest"])
    markers = _artifact_markers(root / protocol["guard"]["artifact"])
    recomputed = verify_payload(protocol, probes, rows, teacher, parent, raw_result, manifest, markers)

    rejected: list[str] = []
    duplicate = copy.deepcopy(rows); duplicate[-1] = copy.deepcopy(duplicate[0])
    rejected.append(_must_reject("duplicate_probe", lambda: verify_payload(protocol, probes, duplicate, teacher, parent, raw_result, manifest, markers)))
    mutated_output = copy.deepcopy(rows); mutated_output[0]["output"] += " mutation"
    rejected.append(_must_reject("output_mutation", lambda: verify_payload(protocol, probes, mutated_output, teacher, parent, raw_result, manifest, markers)))
    mutated_route = copy.deepcopy(rows); actual = str(mutated_route[0]["capability"]); wrong = choose_wrong_route(actual); mutated_route[0]["automatic_capability_route"] = wrong
    if wrong == actual:
        raise Phase3Error("route mutation construction remained identity")
    rejected.append(_must_reject("nonidentity_route_mutation", lambda: verify_payload(protocol, probes, mutated_route, teacher, parent, raw_result, manifest, markers)))
    mutated_result = copy.deepcopy(raw_result); mutated_result["functional_passes_v1"] += 1
    rejected.append(_must_reject("aggregate_mutation", lambda: verify_payload(protocol, probes, rows, teacher, parent, mutated_result, manifest, markers)))
    mutated_manifest = copy.deepcopy(manifest); mutated_manifest["canonical_abstention_clause"] += " changed"
    rejected.append(_must_reject("manifest_mutation", lambda: verify_payload(protocol, probes, rows, teacher, parent, raw_result, mutated_manifest, markers)))

    result = {
        "format": FORMAT,
        "status": "PASS_CORRECTED_HOSTILE_RAW_EVIDENCE_VERIFICATION",
        "protocol_sha256": protocol_sha,
        "observations_verified": len(rows),
        "functional_passes_v1_recomputed": recomputed["functional_passes_v1"],
        "functional_passes_v2_recomputed": recomputed["functional_passes_v2"],
        "teacher_comparison_v1_recomputed": recomputed["teacher_comparison_v1"],
        "markers_independently_derived": list(markers),
        "adversarial_mutations_rejected": rejected,
        "adversarial_mutations_rejected_count": len(rejected),
        "route_mutation": {"actual": actual, "mutated": wrong, "nonidentity": wrong != actual},
        "historical_evidence_changed": False,
        "promotion_eligible": False,
        "phase3_certified": False,
        "final_test_accessed": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.mkdir(parents=True)
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_GUARDED_SCREEN_VERIFY_V2_PROTOCOL_V497.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_guarded_screen_verify/verification_v498")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
