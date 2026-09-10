"""Fail-closed recomputation of R21 public evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)

from .binding import load_config
from .protocol import (
    SEEDS,
    WordLabeler,
    evaluation_rows,
    instruction_from_prompt,
    score_output,
)
from .run import _aggregate, _evaluation_teacher, _jsonl, _validate_source


def verify(config_path: Path, source_run: Path, result_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root, config_path)
    _validate_source(root, config_path, source_run)
    teacher_rows = _evaluation_teacher(root, config)
    result = json_object(result_dir / "result.json")
    observations_path = result_dir / "observations.jsonl"
    labeler_path = result_dir / "labeler.json"
    observations = _jsonl(observations_path)
    labeler_document = json_object(labeler_path)
    labeler = WordLabeler(labeler_document)
    if (
        result.get("format") != "abi-r21-public-generative-bakeoff-result/1"
        or result.get("config_sha256") != sha256_file(config_path)
        or result.get("source_receipt_sha256") != sha256_file(source_run / "receipt.json")
        or result.get("source_rows_sha256") != sha256_file(source_run / "source_observations.jsonl")
        or result.get("evidence_sha256") != evidence_hash(result)
        or result.get("observations", {}).get("sha256") != sha256_file(observations_path)
        or result.get("labeler", {}).get("sha256") != sha256_file(labeler_path)
        or len(observations) != 1080
    ):
        raise R14Error("R21 top-level evidence binding failed")
    expected = evaluation_rows()
    expected_by_id = {str(row["record_id"]): row for row in expected}
    teacher_by_id = {str(row["record_id"]): str(row["output"]) for row in teacher_rows}
    keys = {(str(row["method"]), int(row["seed"]), str(row["record_id"])) for row in observations}
    expected_keys = {
        (method, seed, str(row["record_id"]))
        for method in ("raw_sequence", "labeled_monolith", "abi_factorized")
        for seed in SEEDS
        for row in expected
    }
    if keys != expected_keys:
        raise R14Error("R21 observation matrix is incomplete or duplicated")
    for row in observations:
        reference = expected_by_id[str(row["record_id"])]
        teacher = teacher_by_id[str(row["record_id"])]
        score = score_output(reference, str(row["output"]), teacher)
        for key, value in score.items():
            if row.get(key) != value:
                raise R14Error(f"R21 score changed: {row['record_id']} {key}")
        if (
            row.get("output_sha256") != hashlib.sha256(str(row["output"]).encode()).hexdigest()
            or row.get("teacher_output_sha256") != hashlib.sha256(teacher.encode()).hexdigest()
            or row.get("task") != reference["task"]
        ):
            raise R14Error("R21 row identity or output hash changed")
    aggregates = _aggregate(observations)
    if aggregates != result.get("aggregates"):
        raise R14Error("R21 aggregate recomputation changed")
    label_exact = sum(
        labeler.predict(instruction_from_prompt(str(row["prompt"]))) == row["task"]
        for row in expected
    )
    if label_exact != result.get("labeler", {}).get("evaluation_exact"):
        raise R14Error("R21 labeler recomputation changed")
    package_rows = []
    for seed in SEEDS:
        for method in ("raw_sequence", "labeled_monolith", "abi_factorized"):
            system = result["systems"][str(seed)][method]
            for package in system["packages"]:
                path = root / str(package["path"])
                if (
                    not path.is_file()
                    or path.suffix != ".cake"
                    or path.stat().st_size != int(package["bytes"])
                    or sha256_file(path) != package["sha256"]
                    or package.get("signed") is not True
                ):
                    raise R14Error("R21 package identity changed")
                package_rows.append(
                    {"path": package["path"], "bytes": package["bytes"], "sha256": package["sha256"]}
                )
    quality_gate_names = {
        "label_exact",
        "headline_each_task",
        "headline_non_hallucinating",
        "headline_non_collapsed",
        "no_worse_than_teacher",
        "no_worse_than_raw_sequence",
        "no_worse_than_labeled_monolith",
        "factor_parameter_budget",
    }
    quality_pass = all(bool(result["gates"].get(name)) for name in quality_gate_names)
    claimed_pass = result.get("verdict") == "PASS_PUBLIC_PREREQUISITE"
    if claimed_pass != all(bool(value) for value in result.get("gates", {}).values()):
        raise R14Error("R21 verdict does not follow recorded gates")
    if claimed_pass:
        raise R14Error(
            "R21 positive promotion requires a separate fresh live verifier; none is bound"
        )
    verification = {
        "format": "abi-r21-public-strict-verification/1",
        "status": "PASS_VERIFIED_NEGATIVE_RESULT",
        "result_sha256": sha256_file(result_dir / "result.json"),
        "observations_sha256": sha256_file(observations_path),
        "source_receipt_sha256": sha256_file(source_run / "receipt.json"),
        "source_rows_sha256": sha256_file(source_run / "source_observations.jsonl"),
        "rows_recomputed": len(observations),
        "packages_verified": len(package_rows),
        "package_inventory_sha256": hashlib.sha256(
            json.dumps(package_rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "label_decisions_recomputed": len(expected),
        "label_exact": label_exact,
        "quality_gate_pass": quality_pass,
        "scientific_verdict": result["verdict"],
        "full_abi_moonshot": "OPEN",
    }
    verification["evidence_sha256"] = evidence_hash(verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.config, args.source_run, args.result)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
