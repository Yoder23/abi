"""Strictly recompute and certify the negative R18 held-out result."""

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
from .verify_heldout import _records
from .verify_heldout_source import verify_source


def verify_failure(
    config_path: Path,
    reveal_path: Path,
    source_run: Path,
    run_dir: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config, _ = load_bound_inputs(root, config_path, reveal_path)
    source_strict = verify_source(config_path, reveal_path, source_run)
    if source_strict != json_object(source_run / "strict_verification.json"):
        raise R14Error("R18 failed-run source verification changed")
    receipt = _verified_object(run_dir / "receipt.json", "failed held-out receipt")
    if (
        receipt.get("format") != "abi-r18-heldout-factorized-realization/1"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("reveal_sha256") != sha256_file(reveal_path)
        or receipt.get("protocol_sha256")
        != sha256_file(Path(__file__).with_name("HELDOUT_PROTOCOL.md"))
        or receipt.get("verdict") != "FAIL"
        or receipt.get("full_abi_moonshot") != "OPEN"
    ):
        raise R14Error("R18 failed-run receipt identity changed")
    rows = _jsonl(
        source_run / "source_observations.jsonl",
        receipt["artifacts"]["source_rows_sha256"],
    )
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
    _, package_path = _verify_isolation(
        root,
        run_dir,
        "extraction",
        sha256_file(source_bundle),
        _verified_object(source_bundle, "failed source bundle")["evidence_sha256"],
        expected_package,
        rejected,
    )
    _, control_path = _verify_isolation(
        root,
        run_dir,
        "control_extraction",
        sha256_file(control_bundle),
        _verified_object(control_bundle, "failed control bundle")["evidence_sha256"],
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
        raise R14Error("R18 failed evaluation rows changed")
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
    gates = config["gates"]
    gate_passed = (
        len(records) == config["data"]["extraction_rows"]
        and exact == config["data"]["evaluation_rows"]
        and metrics["package_functional_exact"] == gates["package_functional_exact"]
        and metrics["source_exact_regressions"] <= gates["max_source_exact_regressions"]
        and metrics["package_functional_exact"] >= metrics["independent_modal_functional_exact"]
        and metrics["removed_abstain"] == gates["removed_abstain"]
        and metrics["control_functional_exact"] / exact <= gates["max_control_accuracy"]
    )
    if metrics != receipt.get("metrics") or gate_passed:
        raise R14Error("R18 negative scientific verdict changed")
    failing = [
        {
            "record_id": row["record_id"],
            "signature": row["signature"],
            "source_functional_exact": row["source_functional_exact"],
            "package_output": row["package_output"],
            "expected": row["expected"],
        }
        for row in evaluation
        if not row["package_functional_exact"]
    ]
    if len(failing) != exact - metrics["package_functional_exact"]:
        raise R14Error("R18 negative row diagnosis changed")
    result = {
        "format": "abi-r18-heldout-failure-verification/1",
        "verdict": "FAIL_CONFIRMED",
        "metrics": metrics,
        "failed_rows": failing,
        "failed_signatures": sorted({row["signature"] for row in failing}),
        "gate_passed": False,
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
    result = verify_failure(args.config, args.reveal, args.source_run, args.run_dir)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
