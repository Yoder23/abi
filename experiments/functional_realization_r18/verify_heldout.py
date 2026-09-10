"""Fail-closed independent verification of R18 held-out replication."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)

from .heldout_protocol import load_bound_inputs
from .package import load_package, realize
from .verify import (
    _control_records,
    _derive_package,
    _fill,
    _independent_modal,
    _jsonl,
    _verified_object,
    _verify_bundle,
    _verify_isolation,
)
from .verify_heldout_source import verify_source


def _records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = [
        {
            "record_id": row["record_id"],
            "signature": row["signature"],
            "slots": row["slots"],
            "output": row["output"],
        }
        for row in rows
        if row["split"] == "extraction"
    ]
    random.Random(18_002).shuffle(values)
    return values


def verify(
    config_path: Path,
    reveal_path: Path,
    source_run: Path,
    run_dir: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config, _ = load_bound_inputs(root, config_path, reveal_path)
    source_strict = verify_source(config_path, reveal_path, source_run)
    stored_source_strict = json_object(source_run / "strict_verification.json")
    if source_strict != stored_source_strict or not source_strict["compiler_authorized"]:
        raise R14Error("R18 held-out source verification changed")
    receipt = _verified_object(run_dir / "receipt.json", "held-out receipt")
    if (
        receipt.get("format") != "abi-r18-heldout-factorized-realization/1"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("reveal_sha256") != sha256_file(reveal_path)
        or receipt.get("protocol_sha256")
        != sha256_file(Path(__file__).with_name("HELDOUT_PROTOCOL.md"))
        or receipt.get("full_abi_moonshot") != "OPEN"
    ):
        raise R14Error("R18 held-out receipt identity changed")
    source_rows_path = source_run / "source_observations.jsonl"
    rows = _jsonl(source_rows_path, receipt["artifacts"]["source_rows_sha256"])
    if len(rows) != 120:
        raise R14Error("R18 held-out row count changed")
    records = _records(rows)
    control_records = _control_records(records)
    source_bundle = run_dir / receipt["artifacts"]["source_bundle"]["path"]
    control_bundle = run_dir / receipt["artifacts"]["control_bundle"]["path"]
    _verify_bundle(source_bundle, records, receipt["artifacts"]["source_bundle"]["sha256"])
    _verify_bundle(
        control_bundle,
        control_records,
        receipt["artifacts"]["control_bundle"]["sha256"],
    )
    expected_package, rejected = _derive_package(records)
    expected_control, control_rejected = _derive_package(control_records)
    primary_result, package_path = _verify_isolation(
        root,
        run_dir,
        "extraction",
        sha256_file(source_bundle),
        _verified_object(source_bundle, "held-out source bundle")["evidence_sha256"],
        expected_package,
        rejected,
    )
    control_result, control_path = _verify_isolation(
        root,
        run_dir,
        "control_extraction",
        sha256_file(control_bundle),
        _verified_object(control_bundle, "held-out control bundle")["evidence_sha256"],
        expected_control,
        control_rejected,
    )
    package = load_package(package_path)
    control_package = load_package(control_path)
    modal = _independent_modal(records)
    evaluation = []
    for row in rows:
        if row["split"] != "evaluation":
            continue
        package_output = realize(package, row["signature"], row["slots"])
        control_output = realize(control_package, row["signature"], row["slots"])
        modal_output = _fill(modal[str(row["signature"])], row["slots"])
        evaluation.append(
            {
                "record_id": row["record_id"],
                "signature": row["signature"],
                "slots": row["slots"],
                "expected": row["expected"],
                "source_output": row["output"],
                "source_functional_exact": row["functional_exact"],
                "package_output": package_output,
                "package_functional_exact": package_output == row["expected"],
                "package_source_exact": package_output == row["output"],
                "modal_output": modal_output,
                "modal_functional_exact": modal_output == row["expected"],
                "control_output": control_output,
                "control_functional_exact": control_output == row["expected"],
                "removed_output": None,
            }
        )
    evaluation_ref = receipt["artifacts"]["evaluation"]
    if _jsonl(run_dir / evaluation_ref["path"], evaluation_ref["sha256"]) != evaluation:
        raise R14Error("R18 held-out evaluation changed")
    exact = len(evaluation)
    metrics = {
        "source_extraction_rows": len(records),
        "evaluation_rows": exact,
        "source_functional_exact": sum(row["source_functional_exact"] for row in evaluation),
        "package_functional_exact": sum(row["package_functional_exact"] for row in evaluation),
        "package_source_exact": sum(row["package_source_exact"] for row in evaluation),
        "independent_modal_functional_exact": sum(
            row["modal_functional_exact"] for row in evaluation
        ),
        "source_exact_regressions": sum(
            row["source_functional_exact"] and not row["package_functional_exact"]
            for row in evaluation
        ),
        "control_functional_exact": sum(row["control_functional_exact"] for row in evaluation),
        "removed_abstain": sum(row["removed_output"] is None for row in evaluation),
    }
    if metrics != receipt.get("metrics"):
        raise R14Error("R18 held-out metrics changed")
    gates = config["gates"]
    if not (
        len(records) == config["data"]["extraction_rows"] == 72
        and exact == config["data"]["evaluation_rows"] == 48
        and primary_result["templates_learned"] == 24
        and control_result["templates_learned"] == 24
        and metrics["package_functional_exact"] == exact
        and metrics["source_exact_regressions"] == 0
        and metrics["package_functional_exact"] >= metrics["independent_modal_functional_exact"]
        and metrics["removed_abstain"] == exact
        and metrics["control_functional_exact"] / exact <= gates["max_control_accuracy"]
        and receipt.get("verdict") == "PASS"
    ):
        raise R14Error("R18 held-out scientific gate failed")
    source_receipt = json_object(source_run / "receipt.json")
    if (
        receipt["artifacts"]["source_receipt_sha256"] != sha256_file(source_run / "receipt.json")
        or receipt["artifacts"]["source_strict_sha256"]
        != sha256_file(source_run / "strict_verification.json")
        or receipt["artifacts"]["extraction_result_sha256"]
        != sha256_file(run_dir / "extraction/result.json")
        or receipt["artifacts"]["control_extraction_result_sha256"]
        != sha256_file(run_dir / "control_extraction/result.json")
        or receipt["package"]["sha256"] != sha256_file(package_path)
        or receipt["package"]["bytes"] != package_path.stat().st_size
        or receipt["package"]["templates"] != 24
        or receipt["control_package"]["sha256"] != sha256_file(control_path)
        or receipt["control_package"]["bytes"] != control_path.stat().st_size
    ):
        raise R14Error("R18 held-out artifact identity changed")
    accounting = receipt.get("information_accounting", {})
    inherited = source_receipt["information_accounting"]
    if (
        any(accounting.get(key) != value for key, value in inherited.items())
        or accounting.get("compiler_source_records") != 72
        or accounting.get("compiler_source_bundle_bytes") != source_bundle.stat().st_size
        or accounting.get("final_package_bytes") != package_path.stat().st_size
        or accounting.get("final_templates") != 24
        or any(
            accounting.get(key) != 0
            for key in (
                "frozen_source_parameters_in_package",
                "source_training_steps",
                "host_training_steps",
                "bridge_parameters_trained",
            )
        )
        or accounting.get("compiler_elapsed_seconds", 0) <= 0
    ):
        raise R14Error("R18 held-out accounting changed")
    result = {
        "format": "abi-r18-heldout-strict-verification/1",
        "verdict": "PASS",
        "claim": receipt["claim"],
        "claim_ceiling": receipt["claim_ceiling"],
        "source_rows_verified": len(rows),
        "extraction_rows_verified": len(records),
        "evaluation_rows_verified": exact,
        "package_functional_exact": metrics["package_functional_exact"],
        "modal_functional_exact": metrics["independent_modal_functional_exact"],
        "source_functional_exact": metrics["source_functional_exact"],
        "control_functional_exact": metrics["control_functional_exact"],
        "teacher_present_at_compilation": False,
        "teacher_present_at_package_execution": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.config, args.reveal, args.source_run, args.run_dir)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
