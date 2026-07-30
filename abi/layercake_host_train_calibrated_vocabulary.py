"""Build a compact long-form token reserve from train-only LayerCake output.

The frozen full-vocabulary LayerCake host is calibrated on hash-ordered
general-English *training* prompts.  Calibration contributes token IDs only.
No calibration output, validation row, benchmark row, logit, activation, or
weight is copied into the deployed host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

from .layercake_host import _canonical_json_bytes
from .layercake_host_preservation import _load_general_rows
from .layercake_host_runtime import (
    NativeHostRuntime,
    RUNTIME_FORMAT,
    _canonical_sha,
    _quality,
    _sha256_file,
    generate_native_host_bytes,
)


CALIBRATION_FORMAT = "abi-layercake-train-longform-token-calibration/1"
CALIBRATED_VOCABULARY_FORMAT = (
    "abi-layercake-train-calibrated-output-vocabulary/1"
)
CALIBRATION_SUFFIX = (
    " Continue for at least 220 words so sustained decoding can be measured."
)
BUDGET_DEPTHS = {
    "train_longform_32": 32,
    "train_longform_128": 128,
}


class TrainCalibratedVocabularyError(RuntimeError):
    """Raised when train-only calibration or derivation is not auditable."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _payload_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _prompt_hashes_from_manifest(path: Path) -> set[str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    prompts = document.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise TrainCalibratedVocabularyError(
            "speed prompt manifest has no prompts"
        )
    hashes: set[str] = set()
    for row in prompts:
        text = str(row["text"])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        expected = row.get("sha256", row.get("prompt_sha256"))
        if expected is not None and digest != str(expected):
            raise TrainCalibratedVocabularyError(
                "speed prompt manifest hash changed"
            )
        hashes.add(digest)
    return hashes


def calibrate_train_longform_reserve(
    *,
    full_vocabulary_artifact: str | Path,
    base_dynamic_artifact: str | Path,
    general_curriculum_path: str | Path,
    speed_prompt_manifest_path: str | Path,
    output_path: str | Path,
    threads: int = 16,
    output_bytes: int = 1024,
) -> dict[str, Any]:
    """Collect nested token-ID reserves from train-split generations."""

    full_vocabulary_artifact = Path(full_vocabulary_artifact).resolve()
    base_dynamic_artifact = Path(base_dynamic_artifact).resolve()
    general_curriculum_path = Path(general_curriculum_path).resolve()
    speed_prompt_manifest_path = Path(
        speed_prompt_manifest_path
    ).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise TrainCalibratedVocabularyError(
            f"calibration evidence is immutable: {output_path}"
        )
    if threads <= 0 or output_bytes <= 0:
        raise TrainCalibratedVocabularyError(
            "threads and output byte target must be positive"
        )

    train_rows = _load_general_rows(
        general_curriculum_path, split="train"
    )
    validation_rows = _load_general_rows(
        general_curriculum_path, split="instruction_validation"
    )
    validation_hashes = {
        str(row["prompt_sha256"]) for row in validation_rows
    }
    speed_hashes = _prompt_hashes_from_manifest(
        speed_prompt_manifest_path
    )
    ordered = sorted(
        train_rows,
        key=lambda row: (
            hashlib.sha256(str(row["id"]).encode("utf-8")).hexdigest(),
            str(row["prompt_sha256"]),
        ),
    )
    train_hashes = {str(row["prompt_sha256"]) for row in ordered}
    validation_overlap = sorted(train_hashes & validation_hashes)
    speed_overlap = sorted(train_hashes & speed_hashes)
    if validation_overlap or speed_overlap:
        raise TrainCalibratedVocabularyError(
            "calibration train prompts overlap a locked evaluation prompt"
        )

    base_metadata = json.loads(
        (base_dynamic_artifact / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    output_contract = base_metadata["runtime"]["output_vocabulary"]
    base_vocabulary_path = (
        base_dynamic_artifact / output_contract["path"]
    )
    if _sha256_file(base_vocabulary_path) != output_contract["sha256"]:
        raise TrainCalibratedVocabularyError(
            "base dynamic vocabulary changed"
        )
    base_vocabulary = json.loads(
        base_vocabulary_path.read_text(encoding="utf-8")
    )
    base_ids = [
        int(value) for value in base_vocabulary["global_token_ids"]
    ]
    if (
        base_ids != sorted(set(base_ids))
        or len(base_ids) != 1469
        or output_contract.get("mode")
        != "train_base_union_prompt_tokens"
    ):
        raise TrainCalibratedVocabularyError(
            "locked 1,469-token dynamic base changed"
        )

    runtime = NativeHostRuntime(
        full_vocabulary_artifact, threads=threads
    )
    if runtime.output_token_ids is not None:
        raise TrainCalibratedVocabularyError(
            "calibration host is not the full-vocabulary runtime"
        )
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    required = max(BUDGET_DEPTHS.values())
    for row in ordered:
        prompt = str(row["prompt"]) + CALIBRATION_SUFFIX
        transformed_sha = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()
        try:
            generated = generate_native_host_bytes(
                runtime, prompt, output_bytes=output_bytes
            )
        except Exception as exc:
            message = str(exc)
            if (
                "fixed symbolic output" not in message
                and "context ended" not in message
            ):
                raise
            skipped.append(
                {
                    "row_id": str(row["id"]),
                    "source_prompt_sha256": str(row["prompt_sha256"]),
                    "transformed_prompt_sha256": transformed_sha,
                    "reason": message,
                }
            )
            continue
        payload = generated.pop("payload")
        token_ids = [int(value) for value in generated.pop("generated_ids")]
        eligible.append(
            {
                "ordinal": len(eligible),
                "row_id": str(row["id"]),
                "task": str(row.get("task", "")),
                "topic": str(row.get("topic", "")),
                "source_split": str(row["split"]),
                "source_prompt_sha256": str(row["prompt_sha256"]),
                "transformed_prompt_sha256": transformed_sha,
                "output_sha256": hashlib.sha256(payload).hexdigest(),
                "output_utf8_bytes": len(payload),
                "authoritative_generated_token_ids": token_ids,
                "authoritative_generated_tokens": len(token_ids),
                "route": int(generated["route"]),
                "prompt_tokens": int(generated["prompt_tokens"]),
                "timing": generated["timing"],
                "quality": _quality(payload),
            }
        )
        if len(eligible) % 16 == 0:
            print(
                json.dumps(
                    {
                        "calibrated": len(eligible),
                        "required": required,
                        "skipped": len(skipped),
                    }
                ),
                flush=True,
            )
        if len(eligible) == required:
            break
    if len(eligible) != required:
        raise TrainCalibratedVocabularyError(
            f"only {len(eligible)} eligible train prompts were available"
        )

    budgets: dict[str, Any] = {}
    previous = set(base_ids)
    for budget_id, depth in BUDGET_DEPTHS.items():
        selected = set(base_ids)
        for record in eligible[:depth]:
            selected.update(record["authoritative_generated_token_ids"])
        if not previous.issubset(selected):
            raise TrainCalibratedVocabularyError(
                "calibrated budgets are not nested"
            )
        previous = selected
        selected_ids = sorted(selected)
        budgets[budget_id] = {
            "calibration_prompt_count": depth,
            "base_token_count": len(base_ids),
            "added_token_count": len(selected_ids) - len(base_ids),
            "selected_token_count": len(selected_ids),
            "global_token_ids": selected_ids,
            "global_token_ids_sha256": hashlib.sha256(
                _canonical_json_bytes(selected_ids)
            ).hexdigest(),
        }

    document: dict[str, Any] = {
        "format": CALIBRATION_FORMAT,
        "status": "CALIBRATED_FROM_TRAIN_SPLIT_ONLY",
        "protocol": (
            "NATIVE_HOST_TRAIN_CALIBRATED_TOKEN_RESERVE_PROTOCOL.json"
        ),
        "sources": {
            "full_vocabulary_artifact": str(full_vocabulary_artifact),
            "full_vocabulary_metadata_sha256": _sha256_file(
                full_vocabulary_artifact / "metadata.json"
            ),
            "full_vocabulary_runtime_graph_sha256": runtime.metadata[
                "runtime"
            ]["graph_sha256"],
            "base_dynamic_artifact": str(base_dynamic_artifact),
            "base_dynamic_metadata_sha256": _sha256_file(
                base_dynamic_artifact / "metadata.json"
            ),
            "base_dynamic_runtime_graph_sha256": base_metadata["runtime"][
                "graph_sha256"
            ],
            "base_vocabulary_sha256": _sha256_file(
                base_vocabulary_path
            ),
            "general_curriculum": str(general_curriculum_path),
            "general_curriculum_sha256": _sha256_file(
                general_curriculum_path
            ),
            "speed_prompt_manifest": str(speed_prompt_manifest_path),
            "speed_prompt_manifest_sha256": _sha256_file(
                speed_prompt_manifest_path
            ),
        },
        "selection": {
            "source_split": "train",
            "train_rows_available": len(train_rows),
            "instruction_validation_rows_available": len(validation_rows),
            "order": "sha256_utf8_row_id_then_prompt_sha256",
            "prompt_suffix": CALIBRATION_SUFFIX,
            "output_target_bytes": output_bytes,
            "runtime_threads": threads,
            "eligible_rows_used": len(eligible),
            "skipped_rows": skipped,
            "train_validation_prompt_overlap_count": len(
                validation_overlap
            ),
            "train_speed_prompt_overlap_count": len(speed_overlap),
            "validation_outputs_seen": 0,
            "benchmark_outputs_seen": 0,
            "final_test_rows_seen": 0,
        },
        "budgets": budgets,
        "records": eligible,
        "imported_information_accounting": {
            "source_teacher_tokens": 0,
            "source_teacher_output_bytes": 0,
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "weights_changed": 0,
            "calibration_layercake_output_token_ids_stored": sum(
                row["authoritative_generated_tokens"]
                for row in eligible
            ),
            "deployed_calibration_outputs": 0,
            "deployed_payload": (
                "sorted token-ID reserve only; weights and graph unchanged"
            ),
        },
        "claim_boundary": (
            "The budgets are nested tested reserves learned from this "
            "LayerCake host's own train-split output. They are not a global "
            "minimum and are not promoted without every locked quality and "
            "speed certificate."
        ),
        "final_test_accessed": False,
    }
    document["evidence_sha256"] = _payload_hash(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, document)
    return document


def derive_train_calibrated_dynamic_artifact(
    *,
    base_dynamic_artifact: str | Path,
    calibration_evidence_path: str | Path,
    budget_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind one calibrated base to the already-proved dynamic ONNX graph."""

    base_dynamic_artifact = Path(base_dynamic_artifact).resolve()
    calibration_evidence_path = Path(
        calibration_evidence_path
    ).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise TrainCalibratedVocabularyError(
            f"calibrated artifact is immutable: {output_path}"
        )
    if budget_id not in BUDGET_DEPTHS:
        raise TrainCalibratedVocabularyError(
            f"unknown calibrated budget: {budget_id}"
        )
    calibration = json.loads(
        calibration_evidence_path.read_text(encoding="utf-8")
    )
    if (
        calibration.get("format") != CALIBRATION_FORMAT
        or calibration.get("evidence_sha256")
        != _payload_hash(
            {
                key: value
                for key, value in calibration.items()
                if key != "evidence_sha256"
            }
        )
    ):
        raise TrainCalibratedVocabularyError(
            "calibration evidence is invalid or changed"
        )
    budget = calibration["budgets"][budget_id]
    selected_ids = [
        int(value) for value in budget["global_token_ids"]
    ]
    if selected_ids != sorted(set(selected_ids)):
        raise TrainCalibratedVocabularyError(
            "calibrated token IDs are not canonical"
        )

    metadata = json.loads(
        (base_dynamic_artifact / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if metadata.get("format") != RUNTIME_FORMAT:
        raise TrainCalibratedVocabularyError(
            "base dynamic artifact format changed"
        )
    contract = metadata["runtime"]["output_vocabulary"]
    source_paths = {
        "graph": base_dynamic_artifact / metadata["runtime"]["graph"],
        "vocabulary": base_dynamic_artifact / contract["path"],
        "tokenizer": base_dynamic_artifact
        / metadata["tokenizer"]["path"],
        "symbolic": base_dynamic_artifact
        / metadata["symbolic_surface"]["path"],
    }
    expected_hashes = {
        "graph": metadata["runtime"]["graph_sha256"],
        "vocabulary": contract["sha256"],
        "tokenizer": metadata["tokenizer"]["sha256"],
        "symbolic": metadata["symbolic_surface"]["sha256"],
    }
    for name, path in source_paths.items():
        if _sha256_file(path) != expected_hashes[name]:
            raise TrainCalibratedVocabularyError(
                f"base dynamic component changed: {name}"
            )

    base_vocabulary = json.loads(
        source_paths["vocabulary"].read_text(encoding="utf-8")
    )
    original_base_ids = [
        int(value) for value in base_vocabulary["global_token_ids"]
    ]
    if (
        len(original_base_ids) != 1469
        or not set(original_base_ids).issubset(selected_ids)
    ):
        raise TrainCalibratedVocabularyError(
            "calibrated budget does not contain the locked base"
        )

    output_path.mkdir(parents=True, exist_ok=False)
    graph_path = output_path / "model-int8.onnx"
    tokenizer_path = output_path / "tokenizer.json"
    symbolic_path = output_path / "symbolic-surface.json"
    shutil.copyfile(source_paths["graph"], graph_path)
    shutil.copyfile(source_paths["tokenizer"], tokenizer_path)
    shutil.copyfile(source_paths["symbolic"], symbolic_path)

    vocabulary = json.loads(json.dumps(base_vocabulary))
    vocabulary["format"] = CALIBRATED_VOCABULARY_FORMAT
    vocabulary["status"] = "DERIVED_TRAIN_CALIBRATED_BASE_UNION_PROMPT_IDS"
    vocabulary["budget_id"] = budget_id
    vocabulary["global_token_ids"] = selected_ids
    vocabulary["base_global_token_ids"] = selected_ids
    vocabulary["selected_token_count"] = len(selected_ids)
    vocabulary["base_token_count"] = len(selected_ids)
    vocabulary["train_calibration"] = {
        "evidence_path_at_derivation": str(calibration_evidence_path),
        "evidence_file_sha256": _sha256_file(
            calibration_evidence_path
        ),
        "evidence_sha256": calibration["evidence_sha256"],
        "calibration_prompt_count": int(
            budget["calibration_prompt_count"]
        ),
        "added_token_count": int(budget["added_token_count"]),
        "validation_outputs_seen": 0,
        "benchmark_outputs_seen": 0,
        "final_test_rows_seen": 0,
    }
    vocabulary["claim_boundary"] = (
        "The runtime candidate set is a train-calibrated base union exact "
        "token IDs in the current prompt. Calibration contributed IDs only; "
        "weights, graph operations, decoding, and ABI are unchanged."
    )
    vocabulary.pop("evidence_sha256", None)
    vocabulary["evidence_sha256"] = _payload_hash(vocabulary)
    vocabulary_path = output_path / "output-vocabulary.json"
    _write_json(vocabulary_path, vocabulary)

    derived = json.loads(json.dumps(metadata))
    derived["status"] = "EXPORTED_NOT_YET_CERTIFIED"
    runtime = derived["runtime"]
    runtime["graph"] = graph_path.name
    runtime["graph_sha256"] = _sha256_file(graph_path)
    runtime["graph_bytes"] = graph_path.stat().st_size
    runtime["output_vocabulary"] = {
        "format": CALIBRATED_VOCABULARY_FORMAT,
        "mode": "train_base_union_prompt_tokens",
        "path": vocabulary_path.name,
        "sha256": _sha256_file(vocabulary_path),
        "budget_id": budget_id,
        "selected_token_count": len(selected_ids),
        "base_token_count": len(selected_ids),
        "full_token_count": 50_257,
        "byte_fallback_ids_complete": True,
        "global_runtime_token_ids": True,
        "allowed_output_ids_graph_input": "allowed_output_ids",
        "prompt_token_ids_added_at_runtime": True,
        "duplicate_output_projection_removed": True,
        "train_calibration_evidence_sha256": calibration[
            "evidence_sha256"
        ],
    }
    derived["tokenizer"]["path"] = tokenizer_path.name
    derived["tokenizer"]["sha256"] = _sha256_file(tokenizer_path)
    derived["symbolic_surface"]["path"] = symbolic_path.name
    derived["symbolic_surface"]["sha256"] = _sha256_file(symbolic_path)
    derived["evidence_sha256"] = _canonical_sha(derived)
    _write_json(output_path / "metadata.json", derived)
    return derived


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--full-vocabulary-artifact", required=True)
    calibrate.add_argument("--base-dynamic-artifact", required=True)
    calibrate.add_argument("--general-curriculum", required=True)
    calibrate.add_argument("--speed-prompt-manifest", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--threads", type=int, default=16)
    calibrate.add_argument("--output-bytes", type=int, default=1024)

    derive = commands.add_parser("derive")
    derive.add_argument("--base-dynamic-artifact", required=True)
    derive.add_argument("--calibration-evidence", required=True)
    derive.add_argument(
        "--budget-id", choices=tuple(BUDGET_DEPTHS), required=True
    )
    derive.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "calibrate":
        result = calibrate_train_longform_reserve(
            full_vocabulary_artifact=args.full_vocabulary_artifact,
            base_dynamic_artifact=args.base_dynamic_artifact,
            general_curriculum_path=args.general_curriculum,
            speed_prompt_manifest_path=args.speed_prompt_manifest,
            output_path=args.output,
            threads=args.threads,
            output_bytes=args.output_bytes,
        )
    else:
        result = derive_train_calibrated_dynamic_artifact(
            base_dynamic_artifact=args.base_dynamic_artifact,
            calibration_evidence_path=args.calibration_evidence,
            budget_id=args.budget_id,
            output_path=args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
