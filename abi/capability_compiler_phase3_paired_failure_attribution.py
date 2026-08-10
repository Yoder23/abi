"""Read-only paired V443/teacher functional-failure attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluator_literals(evaluator: dict[str, Any]) -> list[str]:
    kind = str(evaluator["kind"])
    if kind in {"contains_all", "contains_any", "ordered_contains"}:
        return [str(value) for value in evaluator["values"]]
    if kind == "exact":
        return [str(evaluator["value"])]
    if kind == "all_of":
        values: list[str] = []
        for rule in evaluator["rules"]:
            values.extend(evaluator_literals(rule))
        return values
    return []


def run(root: Path, protocol: Path, output: Path) -> dict[str, Any]:
    catalog_path = root / "catalogs/capability_compiler_phase1_frozen_v1.json"
    teacher_path = root / "results/abi_capability_compiler_phase2/teacher/T0/development_outputs.jsonl"
    candidate_path = root / "results/abi_capability_compiler_phase3_qualified_transition_control/evaluation_v443/development_outputs.jsonl"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    probes = {
        str(row["probe_id"]): row
        for row in catalog["probes"]
        if row["split"] == "validation"
    }
    teacher = {str(row["probe_id"]): row for row in _rows(teacher_path)}
    candidate = {str(row["probe_id"]): row for row in _rows(candidate_path)}
    if set(teacher) != set(candidate) or set(candidate) != set(probes):
        raise RuntimeError("paired development identities changed")

    outcome = Counter()
    owners = Counter()
    per_capability: dict[str, Counter[str]] = defaultdict(Counter)
    attributed_records: list[dict[str, Any]] = []
    for probe_id in sorted(candidate):
        probe = probes[probe_id]
        teacher_pass = bool(teacher[probe_id]["functional_pass"])
        candidate_pass = bool(candidate[probe_id]["functional_pass"])
        pair = f"candidate_{'pass' if candidate_pass else 'fail'}__teacher_{'pass' if teacher_pass else 'fail'}"
        outcome[pair] += 1
        capability = str(candidate[probe_id]["capability"])
        per_capability[capability][pair] += 1
        if candidate_pass or not teacher_pass:
            continue

        prompt = str(probe["prompt"]).casefold()
        generated = str(candidate[probe_id]["output"]).casefold()
        prompt_literals = [
            value
            for value in evaluator_literals(probe["evaluator"])
            if value.casefold() in prompt
        ]
        missing = [value for value in prompt_literals if value.casefold() not in generated]
        if missing:
            owner = "missing_prompt_derived_literal"
        elif str(probe["evaluator"]["kind"]) == "regex":
            owner = "format_regex"
        elif capability == "abstention":
            owner = "abstention_semantics"
        else:
            owner = "other_candidate_specific"
        owners[owner] += 1
        per_capability[capability][owner] += 1
        attributed_records.append(
            {
                "probe_id": probe_id,
                "capability": capability,
                "owner": owner,
                "missing_prompt_literals": missing,
                "candidate_output_sha256": hashlib.sha256(
                    str(candidate[probe_id]["output"]).encode("utf-8")
                ).hexdigest(),
            }
        )

    candidate_specific = outcome["candidate_fail__teacher_pass"]
    prompt_literal = owners["missing_prompt_derived_literal"]
    prompt_literal_share = prompt_literal / candidate_specific
    gates = {
        "paired_identity_exact": len(candidate) == 1400,
        "candidate_specific_failures_exact": candidate_specific == 81,
        "prompt_literal_failures_exact": prompt_literal == 56,
        "prompt_literal_majority": prompt_literal_share >= 0.5,
        "no_generation_or_training": True,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase3-paired-failure-attribution/1",
        "status": "PASS_PROMPT_LITERAL_FIDELITY_MATERIAL_OWNER" if passed else "FAIL_NO_MATERIAL_OWNER",
        "protocol_sha256": _sha256(protocol),
        "bindings": {
            "catalog_sha256": _sha256(catalog_path),
            "teacher_outputs_sha256": _sha256(teacher_path),
            "candidate_outputs_sha256": _sha256(candidate_path),
        },
        "observations": len(candidate),
        "paired_outcomes": dict(sorted(outcome.items())),
        "candidate_specific_failure_owners": dict(sorted(owners.items())),
        "prompt_literal_share_of_candidate_specific_failures": prompt_literal_share,
        "per_capability": {
            capability: dict(sorted(values.items()))
            for capability, values in sorted(per_capability.items())
        },
        "attributed_records": attributed_records,
        "gates": gates,
        "attribution_pass": passed,
        "training_performed": False,
        "model_loaded": False,
        "artifact_mutated": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only development attribution. It selects a measured engineering target but cannot certify quality, authorize final access, or promote V443.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("ABI_CAPABILITY_COMPILER_PHASE3_PAIRED_FAILURE_ATTRIBUTION_PROTOCOL_V448.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/abi_capability_compiler_phase3_paired_failure_attribution/attribution_v449/result.json"),
    )
    args = parser.parse_args()
    result = run(args.root, args.root / args.protocol, args.root / args.output)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
