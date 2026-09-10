"""Compile and evaluate R18 from the frozen R17-v2 public source evidence."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from experiments.linguistic_realization_r17.isolated_worker import (
    IsolatedR17Error,
)
from experiments.linguistic_realization_r17.isolated_worker import (
    _template as r17_template,
)
from experiments.linguistic_realization_r17.verify_source import verify_source

from .isolation import run_wsl_isolated_extraction
from .package import load_package, realize


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R14Error("R18 inherited source rows are unavailable") from exc
    if len(values) != 120 or any(not isinstance(row, dict) for row in values):
        raise R14Error("R18 inherited source row count changed")
    return values


def _write_bundle(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    value = {"format": "abi-r18-anonymous-teacher-realizations/1", "records": records}
    value["evidence_sha256"] = evidence_hash(value)
    write_json_once(path, value)
    return value


def _compiler_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "record_id": row["record_id"],
            "signature": row["signature"],
            "slots": row["slots"],
            "output": row["output"],
        }
        for row in rows
        if row["split"] == "extraction"
    ]


def _control(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        mood, tense, polarity, number = str(record["signature"]).split("|")
        toggled = "question" if mood == "declarative" else "declarative"
        result.append({**record, "signature": "|".join((toggled, tense, polarity, number))})
    return result


def _independent_modal(records: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        try:
            template = r17_template(str(record["output"]), record["slots"])
        except IsolatedR17Error:
            continue
        grouped[str(record["signature"])][template] += 1
    if len(grouped) != 24:
        raise R14Error("R18 modal baseline lacks a signature")
    return {
        signature: sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for signature, counts in grouped.items()
    }


def _modal_realize(templates: dict[str, str], row: dict[str, Any]) -> str:
    result = str(templates[str(row["signature"])])
    for key, value in sorted(row["slots"].items(), key=lambda item: len(item[1]), reverse=True):
        result = result.replace("{" + key + "}", str(value))
    return result


def run(source_run: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R18 public output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    strict = verify_source(source_run, "v2")
    stored_strict = json.loads((source_run / "strict_source.json").read_text(encoding="utf-8"))
    if strict != stored_strict:
        raise R14Error("R18 inherited strict source receipt changed")
    output.mkdir(parents=True)
    started = time.perf_counter()
    source_rows_path = source_run / "source_observations.jsonl"
    rows = _rows(source_rows_path)
    records = _compiler_records(rows)
    random.Random(18_001).shuffle(records)
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
    passed = (
        len(records) == 72
        and exact == 48
        and primary["result"]["templates_learned"] == 24
        and primary["result"]["parseable_records"] >= 48
        and metrics["package_functional_exact"] == exact
        and metrics["source_exact_regressions"] == 0
        and metrics["package_functional_exact"] > metrics["independent_modal_functional_exact"]
        and metrics["removed_abstain"] == exact
        and metrics["control_functional_exact"] / exact <= 0.10
    )
    inherited = json.loads((source_run / "receipt.json").read_text(encoding="utf-8"))
    receipt = {
        "format": "abi-r18-public-factorized-realization/1",
        "verdict": "PASS" if passed else "FAIL",
        "claim": "PUBLIC_BOUNDED_FACTORIZED_ENGLISH_REALIZATION_PREREQUISITE",
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_LAYERCAKE_ACCEPTANCE",
        "protocol_sha256": sha256_file(Path(__file__).with_name("PUBLIC_PROTOCOL.md")),
        "source": {
            **inherited["source"],
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
            "inherited_source_rows_sha256": sha256_file(source_rows_path),
            "inherited_strict_source_sha256": sha256_file(source_run / "strict_source.json"),
            "source_bundle": {"path": source_bundle.name, "sha256": sha256_file(source_bundle)},
            "control_bundle": {"path": control_bundle.name, "sha256": sha256_file(control_bundle)},
            "evaluation": {"path": evaluation_path.name, "sha256": sha256_file(evaluation_path)},
            "extraction_result_sha256": sha256_file(output / "extraction/result.json"),
            "control_extraction_result_sha256": sha256_file(
                output / "control_extraction/result.json"
            ),
        },
        "information_accounting": {
            **inherited["information_accounting"],
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
            "autonomous capability discovery",
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
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source_run, args.output), indent=2))


if __name__ == "__main__":
    main()
