"""Derive R21 labels with the registered ABI-side ontology classifier."""

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
    write_jsonl_once,
)

from .label_control_binding import load_control_config
from .protocol import WordLabeler, instruction_from_prompt, training_rows


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error(f"R21 label-control JSONL unreadable: {path}") from exc


def _classifier() -> WordLabeler:
    return WordLabeler.fit(
        {"instruction": row["instruction"], "label": row["task"]} for row in training_rows()
    )


def _registered_pair_count() -> int:
    return len({(row["task"], row["instruction"].lower()) for row in training_rows()})


def validate_control_source(
    root: Path, control_config_path: Path, source_run: Path
) -> list[dict[str, Any]]:
    config = load_control_config(root, control_config_path)
    receipt = json_object(source_run / "receipt.json")
    rows_path = source_run / "source_observations.jsonl"
    labeler_path = source_run / "labeler.json"
    rows = _jsonl(rows_path)
    original = _jsonl(root / str(config["failed_source_rows"]["path"]))
    labeler = WordLabeler(json_object(labeler_path))
    response_identity = [
        (row.get("record_id"), row.get("teacher_output"), row.get("teacher_output_sha256"))
        for row in rows
    ]
    original_identity = [
        (row.get("record_id"), row.get("teacher_output"), row.get("teacher_output_sha256"))
        for row in original
    ]
    recomputed = [
        labeler.predict(instruction_from_prompt(str(row.get("prompt", "")))) for row in rows
    ]
    if (
        receipt.get("format") != "abi-r21-public-source-acquisition/3"
        or receipt.get("verdict") != "PASS_REGISTERED_LABEL_CONTROL"
        or receipt.get("control_config_sha256") != sha256_file(control_config_path)
        or receipt.get("evidence_sha256") != evidence_hash(receipt)
        or receipt.get("artifacts", {}).get("source_rows", {}).get("sha256")
        != sha256_file(rows_path)
        or receipt.get("artifacts", {}).get("labeler", {}).get("sha256")
        != sha256_file(labeler_path)
        or len(rows) != 600
        or len(original) != 600
        or response_identity != original_identity
        or receipt.get("response_rows_reused_byte_exact") is not True
        or recomputed != [row.get("teacher_label") for row in rows]
        or sum(label == row.get("task") for label, row in zip(recomputed, rows)) != 600
    ):
        raise R14Error("R21 registered label-control evidence failed")
    return rows


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R21 label control exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config = load_control_config(root, config_path)
    original_receipt = json_object(root / str(config["failed_source_receipt"]["path"]))
    original_rows = _jsonl(root / str(config["failed_source_rows"]["path"]))
    expected = training_rows()
    if (
        len(original_rows) != 600
        or original_receipt.get("verdict") != "FAIL_SOURCE"
        or [row.get("record_id") for row in original_rows] != [row["record_id"] for row in expected]
    ):
        raise R14Error("R21 failed source prerequisite changed")

    labeler = _classifier()
    if _registered_pair_count() != 36:
        raise R14Error("R21 registered label seed budget changed")
    rows = []
    for original, registered in zip(original_rows, expected):
        predicted = labeler.predict(instruction_from_prompt(str(original["prompt"])))
        row = dict(original)
        row["generated_teacher_label_original"] = original["teacher_label"]
        row["generated_teacher_label_raw_original"] = original["teacher_label_raw"]
        row["teacher_label"] = predicted
        row["teacher_label_raw"] = predicted
        row["teacher_label_token_count"] = 0
        row["teacher_label_valid"] = True
        row["teacher_label_exact"] = predicted == registered["task"]
        row["teacher_label_source"] = "abi_registered_ontology_classifier"
        rows.append(row)

    output.mkdir(parents=True)
    rows_path = output / "source_observations.jsonl"
    labeler_path = output / "labeler.json"
    write_jsonl_once(rows_path, rows)
    write_json_once(labeler_path, labeler.document)
    train_exact = sum(row["teacher_label_exact"] for row in rows)
    response_exact = all(
        source["record_id"] == derived["record_id"]
        and source["teacher_output"] == derived["teacher_output"]
        and source["teacher_output_sha256"] == derived["teacher_output_sha256"]
        for source, derived in zip(original_rows, rows)
    )
    passed = train_exact == int(config["required_training_exact"])
    receipt = {
        "format": "abi-r21-public-source-acquisition/3",
        "verdict": "PASS_REGISTERED_LABEL_CONTROL" if passed else "FAIL_REGISTERED_LABEL_CONTROL",
        "control_config_sha256": sha256_file(config_path),
        "base_config_sha256": config["base_config"]["sha256"],
        "original_source_receipt_sha256": config["failed_source_receipt"]["sha256"],
        "original_source_rows_sha256": config["failed_source_rows"]["sha256"],
        "response_rows_reused_byte_exact": response_exact,
        "metrics": {
            "training_rows": len(rows),
            "training_label_exact": train_exact,
            "unique_registered_seed_pairs": _registered_pair_count(),
        },
        "artifacts": {
            "source_rows": {"path": rows_path.name, "sha256": sha256_file(rows_path)},
            "labeler": {"path": labeler_path.name, "sha256": sha256_file(labeler_path)},
        },
        "information_accounting": {
            "teacher_responses_reused": 600,
            "teacher_response_tokens_reused": original_receipt["information_accounting"][
                "teacher_generated_tokens"
            ],
            "teacher_response_bytes_reused": original_receipt["information_accounting"][
                "teacher_output_bytes"
            ],
            "new_teacher_calls": 0,
            "new_teacher_tokens": 0,
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "source_parameters_copied": 0,
            "registered_label_seed_pairs": 36,
            "registered_label_classes": 6,
            "labeler_document_sha256": hashlib.sha256(
                json.dumps(labeler.document, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "teacher_side_labeling": False,
        "autonomous_ontology_discovery": False,
        "evaluation_access_during_derivation": False,
        "full_abi_moonshot": "OPEN",
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
