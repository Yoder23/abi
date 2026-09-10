"""Fail-closed strict verification for the R16 held-out factual campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
)

from .facts import namespace_from_question, question
from .package import load_package
from .protocol import candidate_values, evaluation_rows, heldout_facts
from .public_sequence_scoring import normalized_text
from .run import _evaluation_matrix, _verify_bindings


def _jsonl(path: Path, expected_sha: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file() or (expected_sha is not None and sha256_file(path) != expected_sha):
        raise R14Error(f"R16 required raw rows changed: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise R14Error("R16 raw row is not an object")
        rows.append(value)
    if not rows:
        raise R14Error("R16 raw rows are empty")
    return rows


def _verified_object(path: Path, name: str) -> dict[str, Any]:
    value = json_object(path)
    stored = value.get("evidence_sha256")
    scientific = {key: item for key, item in value.items() if key != "evidence_sha256"}
    if not isinstance(stored, str) or evidence_hash(scientific) != stored:
        raise R14Error(f"R16 {name} evidence hash changed")
    return value


def verify(config_path: Path, reveal_path: Path, run_dir: Path) -> dict[str, Any]:
    root = config_path.resolve().parents[3]
    config = json_object(config_path)
    reveal = json_object(reveal_path)
    _verify_bindings(root, config)
    if sha256_file(reveal_path) != config["reveal_sha256"]:
        raise R14Error("R16 reveal changed")
    facts = heldout_facts(
        str(reveal["secret_hex"]),
        str(config["heldout_seed_commitment"]),
        int(config["data"]["facts_per_namespace"]),
    )
    fact_by_id = {fact.fact_id: fact for fact in facts}
    receipt = _verified_object(run_dir / "receipt.json", "receipt")
    if (
        receipt.get("format") != "abi-r16-heldout-factual-acquisition/1"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("reveal_sha256") != sha256_file(reveal_path)
    ):
        raise R14Error("R16 receipt identity changed")
    source_ref = receipt["artifacts"]["source_rows"]
    source_rows = _jsonl(run_dir / source_ref["path"], source_ref["sha256"])
    if len(source_rows) != len(facts) * 6:
        raise R14Error("R16 source row count changed")
    for row in source_rows:
        fact = fact_by_id.get(str(row.get("fact_id")))
        view = int(row.get("view", -1))
        if fact is None or row.get("question") != question(fact, view):
            raise R14Error("R16 source row identity changed")
        if (
            row.get("namespace") != fact.namespace
            or row.get("entity") != fact.entity
            or row.get("answer") != fact.value
            or normalized_text(str(row.get("completion", ""))) != normalized_text(fact.value)
            or row.get("completion_exact") is not True
        ):
            raise R14Error("R16 source factual evidence changed")
        if row["split"] == "extraction":
            candidates = candidate_values(
                fact, int(config["data"]["candidate_budget"]), str(reveal["secret_hex"])
            )
            scores = row.get("candidate_scores")
            if (
                row.get("candidate_values") != list(candidates)
                or not isinstance(scores, list)
                or len(scores) != len(candidates)
                or row.get("candidate_prediction")
                != candidates[max(range(len(scores)), key=scores.__getitem__)]
                or row.get("candidate_prediction") != fact.value
                or row.get("candidate_exact") is not True
                or row.get("semantic_prediction") != namespace_from_question(row["question"])
                or row.get("semantic_prediction") != fact.namespace
                or not isinstance(row.get("residual_sha256"), str)
            ):
                raise R14Error("R16 extraction evidence changed")
    bundle_ref = receipt["artifacts"]["source_bundle"]
    bundle = _verified_object(run_dir / bundle_ref["path"], "source bundle")
    if sha256_file(run_dir / bundle_ref["path"]) != bundle_ref["sha256"]:
        raise R14Error("R16 source bundle file changed")
    forbidden = {"answer", "oracle", "namespace", "fact_id", "secret", "reveal", "success_id"}
    if len(bundle.get("records", [])) != len(facts) * 3 or any(
        forbidden.intersection(record) for record in bundle.get("records", [])
    ):
        raise R14Error("R16 source bundle contains forbidden evidence")
    extraction = _verified_object(run_dir / "extraction/result.json", "isolated extraction")
    if (
        sha256_file(run_dir / "extraction/result.json")
        != receipt["isolation"]["result_sha256"]
        or extraction.get("old_root_present") is not False
        or extraction.get("windows_mount_present") is not False
        or extraction.get("network_namespace_isolated") is not True
        or any(extraction.get(key) != 0 for key in ("oracle_fields_consumed", "fact_ids_consumed", "success_ids_consumed", "reveal_files_present"))
    ):
        raise R14Error("R16 physical extraction evidence changed")
    packages = []
    expected_by_namespace: dict[str, list[dict[str, str]]] = {}
    for fact in facts:
        expected_by_namespace.setdefault(fact.namespace, []).append(
            {"relation": fact.relation, "entity": fact.entity, "value": fact.value}
        )
    for item in receipt["packages"]:
        path = run_dir / "extraction" / item["path"]
        if sha256_file(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise R14Error("R16 package identity changed")
        package = load_package(path)
        expected = sorted(
            expected_by_namespace[package["namespace"]],
            key=lambda value: (value["relation"], value["entity"].casefold()),
        )
        if package["facts"] != expected or item["facts"] != len(expected):
            raise R14Error("R16 package facts changed")
        packages.append(package)
    if {item["namespace"] for item in packages} != set(expected_by_namespace):
        raise R14Error("R16 package namespace coverage changed")
    control_extraction = _verified_object(
        run_dir / "control_extraction/result.json", "control extraction"
    )
    control_packages = [
        load_package(run_dir / "control_extraction" / item["path"])
        for item in control_extraction["packages"]
    ]
    matrix_ref = receipt["artifacts"]["evaluation_rows"]
    matrix = _jsonl(run_dir / matrix_ref["path"], matrix_ref["sha256"])
    recomputed = _evaluation_matrix(facts, packages, control_packages, source_rows)
    if matrix != recomputed or len(matrix) != len(evaluation_rows(facts)):
        raise R14Error("R16 evaluation matrix changed")
    exact = len(matrix)
    extraction_rows = len(facts) * 3
    metrics = {
        "selected_facts": len(facts),
        "extraction_open_exact": extraction_rows,
        "extraction_rows": extraction_rows,
        "candidate_exact": extraction_rows,
        "semantic_exact": extraction_rows,
        "evaluation_source_exact": exact,
        "evaluation_rows": exact,
        "package_exact": sum(row["combined_prediction"] == row["answer"] for row in matrix),
        "source_agreement": sum(bool(row["source_agreement"]) for row in matrix),
        "target_only_exact": sum(row["target_only_prediction"] == row["answer"] for row in matrix),
        "other_only_abstain": sum(row["other_only_prediction"] is None for row in matrix),
        "removed_abstain": sum(row["removed_prediction"] is None for row in matrix),
        "rotated_score_exact": sum(
            row["rotated_score_prediction"] == row["answer"] for row in matrix
        ),
    }
    if metrics != receipt["metrics"]:
        raise R14Error("R16 stored metrics changed")
    if not (
        all(metrics[key] == exact for key in ("evaluation_source_exact", "package_exact", "source_agreement", "target_only_exact", "other_only_abstain", "removed_abstain"))
        and all(metrics[key] == extraction_rows for key in ("extraction_open_exact", "candidate_exact", "semantic_exact"))
        and metrics["rotated_score_exact"] / exact <= float(config["gates"]["max_control_accuracy"])
    ):
        raise R14Error("R16 strict scientific gate failed")
    if receipt.get("verdict") != "PASS" or receipt.get("full_abi_moonshot") != "OPEN":
        raise R14Error("R16 stored verdict changed")
    result = {
        "format": "abi-r16-strict-verification/1",
        "verdict": "PASS",
        "claim": receipt["claim"],
        "claim_ceiling": receipt["claim_ceiling"],
        "facts_exact": len(facts),
        "evaluation_rows_exact": exact,
        "namespaces": sorted(expected_by_namespace),
        "source_training_steps": 0,
        "teacher_present_at_package_execution": False,
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.config, args.reveal, args.run_dir)
    write = {**result}
    from experiments.foreign_capability_r14.core import write_json_once

    write_json_once(args.output, write)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
