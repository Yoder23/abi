"""Compile and evaluate the preregistered R18 hidden lexical replication."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)

from .heldout_protocol import load_bound_inputs
from .isolation import run_wsl_isolated_extraction
from .package import load_package, realize
from .run_public import (
    _compiler_records,
    _control,
    _independent_modal,
    _modal_realize,
    _rows,
    _write_bundle,
)
from .verify_heldout_source import verify_source


def run(
    config_path: Path,
    reveal_path: Path,
    source_run: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R18 held-out output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config, _ = load_bound_inputs(root, config_path, reveal_path)
    strict_source = verify_source(config_path, reveal_path, source_run)
    if strict_source["verdict"] != "PASS_SOURCE_COMPILABILITY":
        raise R14Error("R18 held-out source did not authorize compilation")
    output.mkdir(parents=True)
    started = time.perf_counter()
    source_rows_path = source_run / "source_observations.jsonl"
    rows = _rows(source_rows_path)
    records = _compiler_records(rows)
    random.Random(18_002).shuffle(records)
    source_bundle = output / "source_bundle.json"
    _write_bundle(source_bundle, records)
    primary = run_wsl_isolated_extraction(root, source_bundle, output / "extraction")
    control_bundle = output / "mood_permutation_bundle.json"
    _write_bundle(control_bundle, _control(records))
    control = run_wsl_isolated_extraction(root, control_bundle, output / "control_extraction")
    package_path = output / "extraction" / primary["result"]["package"]["path"]
    control_path = output / "control_extraction" / control["result"]["package"]["path"]
    package = load_package(package_path)
    control_package = load_package(control_path)
    modal = _independent_modal(records)
    evaluation = []
    for row in rows:
        if row["split"] != "evaluation":
            continue
        package_output = realize(package, row["signature"], row["slots"])
        control_output = realize(control_package, row["signature"], row["slots"])
        modal_output = _modal_realize(modal, row)
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
                "removed_output": realize(None, row["signature"], row["slots"]),
            }
        )
    evaluation_path = output / "evaluation.jsonl"
    write_jsonl_once(evaluation_path, evaluation)
    metrics = {
        "source_extraction_rows": len(records),
        "evaluation_rows": len(evaluation),
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
    exact = len(evaluation)
    gates = config["gates"]
    passed = (
        len(records) == int(config["data"]["extraction_rows"])
        and exact == int(config["data"]["evaluation_rows"])
        and primary["result"]["templates_learned"] == 24
        and primary["result"]["parseable_records"] >= 48
        and metrics["package_functional_exact"] == exact
        and metrics["source_exact_regressions"] == 0
        and metrics["package_functional_exact"] >= metrics["independent_modal_functional_exact"]
        and metrics["removed_abstain"] == exact
        and metrics["control_functional_exact"] / exact <= float(gates["max_control_accuracy"])
    )
    source_receipt = json.loads((source_run / "receipt.json").read_text(encoding="utf-8"))
    receipt = {
        "format": "abi-r18-heldout-factorized-realization/1",
        "verdict": "PASS" if passed else "FAIL",
        "claim": "BOUNDED_REPLICATED_FACTORIZED_ENGLISH_REALIZATION",
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_LAYERCAKE_ACCEPTANCE",
        "config_sha256": sha256_file(config_path),
        "reveal_sha256": sha256_file(reveal_path),
        "protocol_sha256": sha256_file(Path(__file__).with_name("HELDOUT_PROTOCOL.md")),
        "source": {
            **source_receipt["source"],
            "present_at_compilation": False,
            "present_at_package_execution": False,
        },
        "metrics": metrics,
        "package": {
            "path": str(package_path.relative_to(output)),
            "bytes": package_path.stat().st_size,
            "sha256": sha256_file(package_path),
            "templates": len(package["templates"]),
        },
        "control_package": {
            "path": str(control_path.relative_to(output)),
            "bytes": control_path.stat().st_size,
            "sha256": sha256_file(control_path),
        },
        "artifacts": {
            "source_receipt_sha256": sha256_file(source_run / "receipt.json"),
            "source_strict_sha256": sha256_file(source_run / "strict_verification.json"),
            "source_rows_sha256": sha256_file(source_rows_path),
            "source_bundle": {"path": source_bundle.name, "sha256": sha256_file(source_bundle)},
            "control_bundle": {
                "path": control_bundle.name,
                "sha256": sha256_file(control_bundle),
            },
            "evaluation": {"path": evaluation_path.name, "sha256": sha256_file(evaluation_path)},
            "extraction_result_sha256": sha256_file(output / "extraction/result.json"),
            "control_extraction_result_sha256": sha256_file(
                output / "control_extraction/result.json"
            ),
        },
        "information_accounting": {
            **source_receipt["information_accounting"],
            "compiler_source_records": len(records),
            "compiler_source_bundle_bytes": source_bundle.stat().st_size,
            "final_package_bytes": package_path.stat().st_size,
            "final_templates": len(package["templates"]),
            "frozen_source_parameters_in_package": 0,
            "source_training_steps": 0,
            "host_training_steps": 0,
            "bridge_parameters_trained": 0,
            "compiler_elapsed_seconds": time.perf_counter() - started,
        },
        "full_abi_moonshot": "OPEN",
        "unproven": [
            "unrestricted English fluency",
            "prompt understanding or conversation",
            "autonomous capability discovery or labeling",
            "arbitrary-domain extraction",
            "production LayerCake ingestion",
            "global minimality",
            "superiority to LoRA or distillation",
        ],
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.reveal, args.source_run, args.output), indent=2))


if __name__ == "__main__":
    main()
