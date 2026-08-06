"""Independent verifier for the bounded Phase 3 failure-attribution screen."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase3_failure_attribution import (
    FORMAT,
    SYSTEMS,
    DiagnosticError,
    classify_attribution,
    load_protocol,
)


def verify(root: Path, protocol_path: Path, evidence_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("format") != FORMAT or evidence.get("status") != "COMPLETE_DIAGNOSTIC_NO_PROMOTION":
        raise DiagnosticError("unsupported or incomplete attribution evidence")
    if evidence.get("protocol_sha256") != protocol_sha:
        raise DiagnosticError("attribution protocol identity mismatch")
    if evidence.get("training_performed") is not False or evidence.get("final_test_accessed") is not False:
        raise DiagnosticError("diagnostic crossed its training or final-data boundary")
    claimed = evidence.get("evidence_sha256")
    unsigned = dict(evidence)
    unsigned.pop("evidence_sha256", None)
    derived = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if claimed != derived:
        raise DiagnosticError("internal attribution evidence hash mismatch")
    systems = evidence.get("systems")
    if not isinstance(systems, dict) or set(systems) != set(SYSTEMS):
        raise DiagnosticError("matched system matrix changed")
    expected_prompts = int(protocol["diagnostic_prompts_per_capability"]) * len(CAPABILITIES)
    expected_tokens = None
    for name in SYSTEMS:
        teacher_forced = systems[name].get("teacher_forced", {})
        if teacher_forced.get("prompts") != expected_prompts:
            raise DiagnosticError(f"teacher-forced prompt count changed: {name}")
        if set(teacher_forced.get("per_capability", {})) != set(CAPABILITIES):
            raise DiagnosticError(f"capability coverage changed: {name}")
        if expected_tokens is None:
            expected_tokens = teacher_forced.get("tokens")
        if teacher_forced.get("tokens") != expected_tokens:
            raise DiagnosticError("matched target token accounting changed")
        if name != "P0" and systems[name].get("autonomous") != protocol["sealed_autonomous_results"][name]:
            raise DiagnosticError(f"sealed autonomous result changed: {name}")
    external = evidence.get("external_layercake_control", {})
    if external.get("identity_pass") is not True or external.get("sealed_certificates_pass") is not True:
        raise DiagnosticError("exact LayerCake positive-control identity failed")
    recovery = evidence.get("c0_corruption_recovery", {})
    observations = recovery.get("observations")
    if not isinstance(observations, int) or not 0 < observations <= expected_prompts:
        raise DiagnosticError("corruption recovery coverage is invalid")
    if set(recovery.get("by_horizon", {})) != {"1", "2", "4", "8", "16"}:
        raise DiagnosticError("corruption horizons changed")
    recomputed = classify_attribution(evidence, protocol["attribution_rules"])
    if recomputed != evidence.get("attribution"):
        raise DiagnosticError("stored attribution is not derivable")
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "evidence_file_sha256": sha256_file(evidence_path),
        "evidence_internal_sha256": claimed,
        "systems": list(SYSTEMS),
        "prompts_per_system": expected_prompts,
        "matched_target_tokens_per_system": expected_tokens,
        "corruption_observations": observations,
        "attribution": recomputed,
        "promotion_eligible": False,
    }


def adversarial_checks(root: Path, protocol_path: Path, evidence_path: Path) -> dict[str, Any]:
    original = json.loads(evidence_path.read_text(encoding="utf-8"))
    mutations = []
    cases = {
        "claim_training": lambda v: v.__setitem__("training_performed", True),
        "claim_final_access": lambda v: v.__setitem__("final_test_accessed", True),
        "alter_protocol": lambda v: v.__setitem__("protocol_sha256", "0" * 64),
        "drop_system": lambda v: v["systems"].pop("C3"),
        "borrow_autonomous": lambda v: v["systems"]["C0"].__setitem__("autonomous", v["systems"]["C1"]["autonomous"]),
        "rewrite_attribution": lambda v: v.__setitem__("attribution", {"primary": "PASS"}),
    }
    temp = evidence_path.parent / ".adversarial-attribution.tmp.json"
    for name, mutate in cases.items():
        value = deepcopy(original)
        mutate(value)
        # Rehash to prove structural checks reject more than byte corruption.
        unsigned = dict(value)
        unsigned.pop("evidence_sha256", None)
        value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        temp.write_text(json.dumps(value), encoding="utf-8")
        rejected = False
        try:
            verify(root, protocol_path, temp)
        except DiagnosticError:
            rejected = True
        finally:
            temp.unlink(missing_ok=True)
        mutations.append({"attack": name, "rejected": rejected})
    if not all(item["rejected"] for item in mutations):
        raise DiagnosticError("one or more attribution attacks were accepted")
    return {"status": "PASS", "attacks": mutations, "rejected": len(mutations)}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_FAILURE_ATTRIBUTION_PROTOCOL_V16.json")
    parser.add_argument("--evidence", default="results/abi_capability_compiler_phase3_failure_attribution/v16/evidence.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = verify(root, (root / args.protocol).resolve(), (root / args.evidence).resolve())
    result["adversarial"] = adversarial_checks(root, (root / args.protocol).resolve(), (root / args.evidence).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
