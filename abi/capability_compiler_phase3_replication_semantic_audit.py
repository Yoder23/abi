"""Read-only three-way semantic identity audit for V494/H1/H2 evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-replication-semantic-identity-audit/1"
ALLOWED_NONDETERMINISTIC_FIELDS = frozenset({"guard_check_seconds"})


def _json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle: return [json.loads(line) for line in handle]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_THREE_WAY_SEMANTIC_IDENTITY_AUDIT" or protocol.get("neural_training_authorized") is not False or protocol.get("new_generation_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("semantic audit governance changed")
    if set(protocol.get("allowed_nondeterministic_fields", [])) != ALLOWED_NONDETERMINISTIC_FIELDS: raise Phase3Error("semantic comparator exclusion changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"semantic audit binding changed: {relative}")
    return protocol, sha256_file(path)


def semantic_projection(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key not in ALLOWED_NONDETERMINISTIC_FIELDS} for row in rows]


def compare_three(reference: list[dict[str, Any]], first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, Any]:
    if len(reference) != 1400 or len(first) != 1400 or len(second) != 1400: raise Phase3Error("replication depth changed")
    projections = [semantic_projection(rows) for rows in (reference, first, second)]
    hashes = [hashlib.sha256(b"".join(canonical_json_bytes(row) for row in rows)).hexdigest() for rows in projections]
    if len(set(hashes)) != 1: raise Phase3Error("semantic replication fields differ")
    timing_difference_count = 0
    for triplet in zip(reference, first, second):
        if not all(set(row) == set(reference[0]) for row in triplet): raise Phase3Error("row schema changed")
        values = [row["guard_check_seconds"] for row in triplet]
        timing_difference_count += int(len(set(values)) > 1)
    return {"semantic_sha256": hashes[0], "semantic_hashes": hashes, "timing_difference_rows": timing_difference_count, "observations_per_host": 1400}


def _must_reject(name: str, callback: Any) -> str:
    try: callback()
    except Phase3Error: return name
    raise Phase3Error(f"semantic audit accepted hostile mutation: {name}")


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists(): raise Phase3Error(f"immutable semantic audit output exists: {output}")
    sources = [_jsonl(root / item["path"]) for item in protocol["hosts"]]
    comparison = compare_three(*sources)
    timing_mutation = copy.deepcopy(sources[1]); timing_mutation[0]["guard_check_seconds"] += 1000.0
    if compare_three(sources[0], timing_mutation, sources[2])["semantic_sha256"] != comparison["semantic_sha256"]: raise Phase3Error("authorized timing exclusion changed semantic identity")
    rejected = []
    output_mutation = copy.deepcopy(sources[1]); output_mutation[0]["output"] += " mutation"
    rejected.append(_must_reject("output_mutation", lambda: compare_three(sources[0], output_mutation, sources[2])))
    token_mutation = copy.deepcopy(sources[1]); token_mutation[0]["output_token_ids"][0] += 1
    rejected.append(_must_reject("completed_token_mutation", lambda: compare_three(sources[0], token_mutation, sources[2])))
    guard_mutation = copy.deepcopy(sources[1]); guard_mutation[0]["guard_terminated"] = not guard_mutation[0]["guard_terminated"]
    rejected.append(_must_reject("guard_action_mutation", lambda: compare_three(sources[0], guard_mutation, sources[2])))
    missing = copy.deepcopy(sources[1][:-1])
    rejected.append(_must_reject("missing_observation", lambda: compare_three(sources[0], missing, sources[2])))
    result = {"format": FORMAT, "status": "PASS_THREE_HOST_SEMANTIC_IDENTITY_RUNTIME_OPEN", "protocol_sha256": protocol_sha, **comparison, "allowed_nondeterministic_fields": sorted(ALLOWED_NONDETERMINISTIC_FIELDS), "authorized_timing_mutation_accepted": True, "hostile_semantic_mutations_rejected": rejected, "hostile_semantic_mutations_rejected_count": len(rejected), "new_generation_performed": False, "neural_training_performed": False, "historical_evidence_changed": False, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); output.mkdir(parents=True); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_REPLICATION_SEMANTIC_AUDIT_PROTOCOL_V501.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_replication_semantic_audit/audit_v502"); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
