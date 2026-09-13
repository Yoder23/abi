"""Fail-closed recomputation of the negative R46 acquisition result."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from safetensors import safe_open

from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output
from experiments.causal_english_transfer_r46 import run_v1 as campaign
from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once


EXPECTED_RESULT_SHA256 = "637323763db55a5f802db0101e5da98633a347609d970571a33bb3572f12cc40"


class VerificationError(RuntimeError):
    """Raised when any required R46 evidence cannot be recomputed."""


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"required JSON is absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"required JSON is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"required JSON is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise VerificationError(f"required JSONL is absent: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"required JSONL is unreadable: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise VerificationError(f"required JSONL contains a non-object row: {path}")
    return rows


def verify(
    run_dir: Path,
    layercake_root: Path,
    training_archive: Path,
    evaluation_archive: Path,
    parent_dir: Path,
) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    if sha256_file(result_path) != EXPECTED_RESULT_SHA256:
        raise VerificationError("R46 result file differs from the completed frozen run")
    result = _json(result_path)
    unsigned = dict(result)
    stored_evidence = unsigned.pop("evidence_sha256", None)
    if not isinstance(stored_evidence, str) or evidence_hash(unsigned) != stored_evidence:
        raise VerificationError("R46 result evidence digest changed")
    if (
        result.get("format") != "abi-r46-causal-english-transfer-feasibility/1"
        or result.get("verdict") != "FAIL_R46_CAUSAL_ENGLISH_TRANSFER_FEASIBILITY"
        or result.get("full_abi_moonshot") != "OPEN"
    ):
        raise VerificationError("R46 result scope or negative verdict changed")

    frozen_files = (
        (training_archive, campaign.TRAINING_SHA256, "training_archive_sha256"),
        (evaluation_archive, campaign.EVALUATION_SHA256, "evaluation_archive_sha256"),
        (parent_dir / "model.safetensors", campaign.PARENT_SHA256, "parent_checkpoint_sha256"),
        (parent_dir / "tokenizer.json", campaign.TOKENIZER_SHA256, "parent_tokenizer_sha256"),
    )
    for path, digest, receipt in frozen_files:
        if not path.is_file() or sha256_file(path) != digest:
            raise VerificationError(f"R46 frozen input changed: {path}")
        if result.get("inputs", {}).get(receipt) != digest:
            raise VerificationError(f"R46 frozen input receipt changed: {receipt}")

    artifacts = result.get("artifacts", {})
    required_artifacts = {
        "evaluation": "evaluation.jsonl",
        "excluded_training_rows": "excluded_training_rows.jsonl",
        "training_trace": "training_trace.jsonl",
        "checkpoint": "model.safetensors",
    }
    if set(artifacts) != set(required_artifacts):
        raise VerificationError("R46 artifact inventory changed")
    for name, filename in required_artifacts.items():
        path = run_dir / filename
        receipt = artifacts[name]
        if (
            not path.is_file()
            or receipt.get("path") != filename
            or receipt.get("sha256") != sha256_file(path)
            or receipt.get("bytes") != path.stat().st_size
        ):
            raise VerificationError(f"R46 artifact receipt changed: {name}")

    training = campaign._read_bundle(training_archive, campaign.TRAINING_SHA256)
    evaluation = campaign._read_bundle(evaluation_archive, campaign.EVALUATION_SHA256)
    validation = [row for row in evaluation["records"] if row.get("split") == "validation"]
    selected: list[dict[str, Any]] = []
    for capability in campaign.CAPABILITIES:
        population = sorted(
            (row for row in validation if row.get("capability") == capability),
            key=lambda row: row["record_id"],
        )
        if len(population) != 100:
            raise VerificationError(f"R46 evaluation stratum changed: {capability}")
        selected.extend(population[:20])
    training_ids = {row["record_id"] for row in training["records"]}
    training_prompts = {row["prompt_sha256"] for row in training["records"]}
    training_outputs = {row["output_sha256"] for row in training["records"]}
    overlaps = {
        "record_ids": len(training_ids & {row["record_id"] for row in validation}),
        "prompt_sha256": len(training_prompts & {row["prompt_sha256"] for row in validation}),
        "output_sha256": len(training_outputs & {row["output_sha256"] for row in validation}),
    }
    if len(selected) != 280 or any(overlaps.values()):
        raise VerificationError("R46 train/evaluation separation changed")
    if result.get("inputs", {}).get("training_evaluation_overlap") != overlaps:
        raise VerificationError("R46 overlap receipt changed")

    sys.path.insert(0, str(layercake_root))
    from layercake.models.sparse_bpe_layercake import LayerCakeSparseBPECore, SparseBPELayerCakeConfig
    from layercake.training.phase2_sparse_bpe import _tokenizer

    parent_metadata = _json(parent_dir / "metadata.json")
    tokenizer = _tokenizer(parent_dir / "tokenizer.json")
    prepared, expected_excluded, prepared_response_tokens = campaign._prepare_training(
        training["records"], tokenizer, int(parent_metadata["architecture"]["max_tokens"])
    )
    excluded = _jsonl(run_dir / "excluded_training_rows.jsonl")
    if excluded != expected_excluded:
        raise VerificationError("R46 excluded training population changed")

    rng = random.Random(campaign.SEED)
    orders: dict[str, list[dict[str, Any]]] = {}
    cursors: dict[str, int] = {}
    for capability in campaign.CAPABILITIES:
        orders[capability] = list(prepared[capability])
        rng.shuffle(orders[capability])
        cursors[capability] = 0
    sequence_digest = hashlib.sha256()
    for _ in range(campaign.STEPS):
        for capability in campaign.CAPABILITIES:
            position = cursors[capability]
            values = orders[capability]
            if position == len(values):
                rng.shuffle(values)
                position = 0
            row = values[position]
            cursors[capability] = position + 1
            sequence_digest.update((capability + "\0" + row["record_id"] + "\n").encode())
    training_receipt = result.get("training", {})
    if (
        training_receipt.get("record_sequence_sha256") != sequence_digest.hexdigest()
        or training_receipt.get("prepared_response_tokens") != prepared_response_tokens
        or training_receipt.get("source_records_context_eligible")
        != sum(len(values) for values in prepared.values())
        or training_receipt.get("source_records_context_excluded") != len(excluded)
        or training_receipt.get("examples_seen") != campaign.STEPS * campaign.BATCH_SIZE
    ):
        raise VerificationError("R46 training population accounting changed")

    trace = _jsonl(run_dir / "training_trace.jsonl")
    expected_steps = [1, *range(100, 1_701, 100), campaign.STEPS]
    if (
        [row.get("step") for row in trace] != expected_steps
        or trace != training_receipt.get("curves")
        or any(not isinstance(row.get("training_loss"), (int, float)) for row in trace)
        or any(not isinstance(row.get("wall_seconds"), (int, float)) for row in trace)
        or any(row["wall_seconds"] <= 0 for row in trace)
    ):
        raise VerificationError("R46 training trace changed or is incomplete")

    architecture = parent_metadata["architecture"]
    state_shapes: dict[str, list[int]] = {}
    with safe_open(run_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        for key in handle.keys():
            state_shapes[key] = list(handle.get_slice(key).get_shape())
    reference = LayerCakeSparseBPECore(SparseBPELayerCakeConfig(**architecture))
    reference_shapes = {key: list(value.shape) for key, value in reference.state_dict().items()}
    if state_shapes != reference_shapes:
        raise VerificationError("R46 checkpoint is incompatible with the declared LayerCake graph")
    del reference

    evaluator_map = campaign._evaluator_by_record(evaluation)
    selected_map = {row["record_id"]: row for row in selected}
    raw = _jsonl(run_dir / "evaluation.jsonl")
    by_system: dict[str, list[dict[str, Any]]] = {
        system: [row for row in raw if row.get("system") == system]
        for system in ("parent", "candidate", "source")
    }
    expected_order = [row["record_id"] for row in selected]
    if (
        len(raw) != 840
        or any(len(rows) != 280 for rows in by_system.values())
        or any([row.get("record_id") for row in rows] != expected_order for rows in by_system.values())
    ):
        raise VerificationError("R46 raw evaluation matrix changed")

    for system, rows in by_system.items():
        for row in rows:
            source = selected_map[row["record_id"]]
            evaluator = evaluator_map[row["record_id"]]["evaluator"]
            output = row.get("output")
            if not isinstance(output, str):
                raise VerificationError("R46 raw output is absent")
            output.encode("utf-8", errors="strict")
            passed, score = evaluate_output(output, evaluator)
            if (
                row.get("capability") != source["capability"]
                or row.get("prompt_sha256") != source["prompt_sha256"]
                or row.get("evaluator") != evaluator
                or row.get("functional_pass") != passed
                or row.get("functional_score") != score
                or row.get("output_sha256") != hashlib.sha256(output.encode()).hexdigest()
            ):
                raise VerificationError(f"R46 raw functional row changed: {system}/{row['record_id']}")
            if system == "source":
                if (
                    output != source["output"]
                    or row.get("teacher_tokens") != source["teacher_tokens"]
                    or row.get("teacher_token_count_authoritative") is not True
                ):
                    raise VerificationError("R46 source row differs from the immutable evaluation archive")
                continue
            token_ids = row.get("output_token_ids")
            calls = row.get("decode_expert_forward_calls")
            collapse = _collapse_metrics(
                token_ids,
                output,
                tokenizer.encode(source["prompt"]),
                source["prompt"],
            )
            if (
                not isinstance(token_ids, list)
                or not isinstance(calls, list)
                or row.get("generated_tokens") != len(token_ids)
                or row.get("decode_expert_invocations") != sum(calls)
                or row.get("collapse") != collapse
                or not isinstance(row.get("latency_seconds"), (int, float))
                or row["latency_seconds"] <= 0
            ):
                raise VerificationError(f"R46 decode evidence changed: {system}/{row['record_id']}")

    by_capability = {
        capability: {
            system: sum(
                row["functional_pass"]
                for row in by_system[system]
                if row["capability"] == capability
            )
            for system in by_system
        }
        for capability in campaign.CAPABILITIES
    }
    metrics = {
        "evaluation_rows_per_system": 280,
        "functional": {system: sum(row["functional_pass"] for row in rows) for system, rows in by_system.items()},
        "functional_rate": {system: sum(row["functional_pass"] for row in rows) / len(rows) for system, rows in by_system.items()},
        "by_capability": by_capability,
        "candidate_minus_parent": campaign._bootstrap(
            [row["functional_pass"] for row in by_system["candidate"]],
            [row["functional_pass"] for row in by_system["parent"]],
            campaign.SEED + 1,
        ),
        "candidate_minus_source": campaign._bootstrap(
            [row["functional_pass"] for row in by_system["candidate"]],
            [row["functional_pass"] for row in by_system["source"]],
            campaign.SEED + 2,
        ),
        "candidate_collapses": sum(row["collapse"]["collapse_detected"] for row in by_system["candidate"]),
        "parent_collapses": sum(row["collapse"]["collapse_detected"] for row in by_system["parent"]),
        "candidate_generation_errors": sum(row["generation_error"] is not None for row in by_system["candidate"]),
        "parent_generation_errors": sum(row["generation_error"] is not None for row in by_system["parent"]),
        "candidate_terminated": sum(row["terminated"] for row in by_system["candidate"]),
        "parent_terminated": sum(row["terminated"] for row in by_system["parent"]),
        "candidate_physical_sparse_rows": sum(
            row["maximum_active_decode_experts_per_token"] == 1 for row in by_system["candidate"]
        ),
        "training_expert_assignments": result["metrics"]["training_expert_assignments"],
        "training_experts_used": sum(value > 0 for value in result["metrics"]["training_expert_assignments"]),
    }
    if metrics != result.get("metrics"):
        raise VerificationError("R46 published metrics differ from raw recomputation")
    gates = {
        "matrix_complete": all(len(values) == 280 for values in by_system.values()),
        "train_evaluation_disjoint": not any(overlaps.values()),
        "candidate_overall_functional": metrics["functional_rate"]["candidate"] >= 0.50,
        "candidate_per_capability": all(by_capability[value]["candidate"] >= 6 for value in campaign.CAPABILITIES),
        "candidate_parent_improvement": metrics["candidate_minus_parent"]["point"] >= 0.20,
        "candidate_source_gap": metrics["candidate_minus_source"]["point"] >= -0.25,
        "candidate_utf8": metrics["candidate_generation_errors"] == 0,
        "candidate_noncollapse": metrics["candidate_collapses"] <= 2,
        "all_sparse_experts_trained": metrics["training_experts_used"] == 8,
        "physical_top1_decode": metrics["candidate_physical_sparse_rows"] == 280,
        "frozen_inputs_unchanged": True,
    }
    if gates != result.get("gates") or all(gates.values()):
        raise VerificationError("R46 gate vector changed or no longer demonstrates the registered failure")
    return {
        "format": "abi-r46-strict-verification/1",
        "verdict": "PASS_R46_NEGATIVE_EVIDENCE_VERIFICATION",
        "scientific_verdict": result["verdict"],
        "result_sha256": sha256_file(result_path),
        "evidence_sha256": stored_evidence,
        "raw_rows_recomputed": len(raw),
        "training_sequence_records_recomputed": campaign.STEPS * campaign.BATCH_SIZE,
        "checkpoint_tensors_shape_checked": len(state_shapes),
        "candidate_functional": metrics["functional"]["candidate"],
        "source_functional": metrics["functional"]["source"],
        "candidate_collapses": metrics["candidate_collapses"],
        "full_abi_moonshot": "OPEN",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--training-archive", type=Path, required=True)
    parser.add_argument("--evaluation-archive", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(
        args.run_dir.resolve(),
        args.layercake_root.resolve(),
        args.training_archive.resolve(),
        args.evaluation_archive.resolve(),
        args.parent.resolve(),
    )
    write_json_once(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
