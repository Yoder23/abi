"""Independent hostile verifier for the sealed V494 guarded screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import zipfile

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_contract_guard_audit import _contains_any_values
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import paired_stratified_bootstrap, wilson
from .capability_compiler_phase3_weak_residual import WEAK_CAPABILITIES
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase3-guarded-screen-hostile-verifier/1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_HOSTILE_READ_ONLY_VERIFICATION"
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("hostile-verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"hostile-verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def _artifact_markers(path: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(path, "r") as archive:
        rows = [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line]
    contracts = [
        set(_contains_any_values(row["functional_evaluator"]))
        for row in rows
        if row["capability"] == "abstention" and _contains_any_values(row["functional_evaluator"])
    ]
    if not contracts:
        raise Phase3Error("artifact abstention contracts absent")
    return tuple(sorted(set.intersection(*contracts)))


def verify_payload(
    protocol: Mapping[str, Any],
    probes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    parent_rows: list[dict[str, Any]],
    raw_result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    markers: tuple[str, ...],
) -> dict[str, Any]:
    if len(probes) != 1400 or len(rows) != 1400:
        raise Phase3Error("development depth changed")
    probe_ids = [str(row["probe_id"]) for row in probes]
    row_ids = [str(row.get("probe_id")) for row in rows]
    if row_ids != probe_ids or len(set(row_ids)) != len(row_ids):
        raise Phase3Error("probe identity, order, or uniqueness changed")
    teacher = {str(row["probe_id"]): row for row in teacher_rows}
    parent = {str(row["probe_id"]): row for row in parent_rows}
    if set(teacher) != set(row_ids) or set(parent) != set(row_ids):
        raise Phase3Error("teacher or parent coverage changed")

    clause = str(protocol["guard"]["canonical_abstention_clause"])
    verified: list[dict[str, Any]] = []
    for probe, row in zip(probes, rows):
        capability = str(probe["canonical_capability"])
        output = str(row.get("output", ""))
        if row.get("capability") != capability:
            raise Phase3Error("capability label changed")
        if row.get("automatic_capability_route") != capability or row.get("capability_route_correct") is not True:
            raise Phase3Error("qualified route evidence changed")
        if bool(row.get("weak_route_active")) != (capability in WEAK_CAPABILITIES):
            raise Phase3Error("weak route activation changed")
        v1 = evaluate_functional(output, probe["evaluator"])
        v2 = evaluate_functional_v2(output, probe["evaluator"], capability)
        collapse = repetition_collapse_v2(output)
        if row.get("functional_pass_v1") is not v1 or row.get("functional_pass_v2") is not v2 or row.get("repetition_collapse_v2") != collapse:
            raise Phase3Error("functional or collapse claim changed")
        if collapse:
            raise Phase3Error("guard emitted a collapsing output")
        prefixed = bool(row.get("abstention_clause_prefixed"))
        if prefixed and (capability != "abstention" or not output.startswith(clause)):
            raise Phase3Error("abstention prefix provenance changed")
        if capability == "abstention" and not any(marker.casefold() in output.casefold() for marker in markers):
            raise Phase3Error("abstention marker contract not satisfied")
        if capability not in WEAK_CAPABILITIES:
            exact = output == str(parent[str(row["probe_id"])]["output"])
            if not exact or row.get("strong_parent_output_exact") is not True:
                raise Phase3Error("strong parent identity changed")
            if bool(row.get("guard_terminated")) or prefixed:
                raise Phase3Error("guard touched strong route")
        verified.append({"probe_id": row["probe_id"], "capability": capability, "v1": v1, "v2": v2})

    per: dict[str, dict[str, Any]] = {}
    for capability in CAPABILITIES:
        selected = [row for row in verified if row["capability"] == capability]
        if len(selected) != 100:
            raise Phase3Error("capability depth changed")
        passes_v1 = sum(row["v1"] for row in selected)
        passes_v2 = sum(row["v2"] for row in selected)
        per[capability] = {"passes_v1": passes_v1, "passes_v2": passes_v2, "observations": 100, "collapses_v2": 0, "wilson_v1": wilson(passes_v1, 100)}
    probe_map = {str(row["probe_id"]): row for row in probes}
    paired = [
        {
            "capability": row["capability"],
            "candidate_pass": row["v1"],
            "teacher_pass": evaluate_functional(str(teacher[str(row["probe_id"])]["output"]), probe_map[str(row["probe_id"])]["evaluator"]),
        }
        for row in verified
    ]
    comparison = paired_stratified_bootstrap(
        paired,
        replicates=int(protocol["teacher_relative_screen"]["bootstrap_replicates"]),
        seed=int(protocol["teacher_relative_screen"]["bootstrap_seed"]),
    )
    strong_count = sum(row["capability"] not in WEAK_CAPABILITIES for row in verified)
    recomputed = {
        "functional_passes_v1": sum(row["v1"] for row in verified),
        "functional_passes_v2": sum(row["v2"] for row in verified),
        "observations": len(verified),
        "per_capability": per,
        "repetition_collapses_v2": 0,
        "guard_terminations": sum(bool(row.get("guard_terminated")) for row in rows),
        "abstention_prefixes": sum(bool(row.get("abstention_clause_prefixed")) for row in rows),
        "strong_routes_exact": strong_count,
        "strong_route_observations": strong_count,
        "router_correct": len(rows),
        "teacher_comparison_v1": comparison,
    }
    for key, value in recomputed.items():
        if raw_result.get(key) != value:
            raise Phase3Error(f"raw result aggregate changed: {key}")
    claimed = dict(raw_result)
    evidence = claimed.pop("evidence_sha256", None)
    if evidence != hashlib.sha256(canonical_json_bytes(claimed)).hexdigest():
        raise Phase3Error("raw result evidence hash changed")
    expected_manifest = {
        "format": "abi-capability-contract-guard-manifest/1",
        "artifact_sha256": protocol["guard"]["artifact_sha256"],
        "markers": list(markers),
        "canonical_abstention_clause": clause,
        "repetition_predicate_module_sha256": protocol["bindings"]["abi/capability_compiler_repetition_v2.py"],
        "teacher_required_at_inference": False,
    }
    if manifest != expected_manifest:
        raise Phase3Error("guard manifest changed")
    if raw_result.get("passed") is not True or not all(raw_result.get("gates", {}).values()):
        raise Phase3Error("sealed pass gates changed")
    return recomputed


def _must_reject(name: str, callback: Any) -> str:
    try:
        callback()
    except Phase3Error:
        return name
    raise Phase3Error(f"adversarial mutation accepted: {name}")


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable verifier output exists: {output}")
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
    mutated_route = copy.deepcopy(rows); mutated_route[0]["automatic_capability_route"] = "grammar"
    rejected.append(_must_reject("route_mutation", lambda: verify_payload(protocol, probes, mutated_route, teacher, parent, raw_result, manifest, markers)))
    mutated_result = copy.deepcopy(raw_result); mutated_result["functional_passes_v1"] += 1
    rejected.append(_must_reject("aggregate_mutation", lambda: verify_payload(protocol, probes, rows, teacher, parent, mutated_result, manifest, markers)))
    mutated_manifest = copy.deepcopy(manifest); mutated_manifest["canonical_abstention_clause"] += " changed"
    rejected.append(_must_reject("manifest_mutation", lambda: verify_payload(protocol, probes, rows, teacher, parent, raw_result, mutated_manifest, markers)))

    result = {
        "format": FORMAT,
        "status": "PASS_HOSTILE_RAW_EVIDENCE_VERIFICATION",
        "protocol_sha256": protocol_sha,
        "observations_verified": len(rows),
        "functional_passes_v1_recomputed": recomputed["functional_passes_v1"],
        "functional_passes_v2_recomputed": recomputed["functional_passes_v2"],
        "teacher_comparison_v1_recomputed": recomputed["teacher_comparison_v1"],
        "markers_independently_derived": list(markers),
        "adversarial_mutations_rejected": rejected,
        "adversarial_mutations_rejected_count": len(rejected),
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
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_GUARDED_SCREEN_VERIFY_PROTOCOL_V495.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_guarded_screen_verify/verification_v496")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
