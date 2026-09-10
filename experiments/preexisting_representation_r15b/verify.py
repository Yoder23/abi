"""Fail-closed recomputation of the R15B held-out claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from experiments.copy_paste_r10.runtime import canonical_prediction
from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)
from experiments.foreign_neural_state_r15.recipient_worker import summarize
from experiments.foreign_teacher_r12.custody import verify_r11_freeze
from experiments.native_isa_r11.core import (
    RecurrentTransitionNeuralISA,
    load_package,
    transition_accuracy,
    transition_bytes,
)
from experiments.native_isa_r11.run import _conditions

from .protocol import evaluation_rows, heldout_capabilities
from .public_extraction import LABEL_NAMESPACE, _anchor_rows, _control_accuracy
from .public_qualification import OPERATIONS
from .representation import decode_labels, decode_transition, labels_to_operations


def _verify_evidence(value: dict[str, Any], label: str) -> None:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    if stored != evidence_hash(payload):
        raise R14Error(f"{label} evidence hash changed")


def _jsonl(path: Path, expected_sha: str, expected_rows: int) -> list[dict[str, Any]]:
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise R14Error(f"required R15B raw rows unavailable: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R14Error(f"required R15B raw rows unreadable: {path}") from exc
    if len(rows) != expected_rows or not all(isinstance(row, dict) for row in rows):
        raise R14Error(f"R15B raw row count changed: {path}")
    return rows


def _close(left: list[float], right: list[float], *, tolerance: float = 2e-6) -> bool:
    return len(left) == len(right) and all(
        math.isfinite(float(a))
        and math.isfinite(float(b))
        and abs(float(a) - float(b)) <= tolerance
        for a, b in zip(left, right)
    )


def _verify_isolated(
    extraction: Path, expected_bundle_sha: str, expected_labels: list[int]
) -> None:
    result = json_object(extraction / "result.json")
    launcher = json_object(extraction / "launcher.json")
    manifest = json_object(extraction / "manifest.json")
    _verify_evidence(result, "R15B isolated result")
    _verify_evidence(launcher, "R15B isolated launcher")
    _verify_evidence(manifest, "R15B isolated manifest")
    declared = {str(row["path"]): str(row["sha256"]) for row in manifest["files"]}
    if (
        result.get("format") != "abi-r15b-isolated-representation-extraction/1"
        or result.get("labels") != expected_labels
        or result.get("representation_sha256") != expected_bundle_sha
        or min(float(value) for value in result.get("margins", [])) <= 0
        or result.get("prompts_consumed") != 0
        or result.get("answers_consumed") != 0
        or result.get("semantic_labels_consumed") != 0
        or result.get("candidate_search") is not False
        or result.get("old_root_present") is not False
        or result.get("windows_mount_present") is not False
        or result.get("network_namespace_isolated") is not True
        or manifest.get("prompts_included") != 0
        or manifest.get("answers_included") != 0
        or manifest.get("semantic_labels_included") != 0
        or declared.get("representation.safetensors") != expected_bundle_sha
        or launcher.get("result_sha256") != sha256_file(extraction / "result.json")
        or launcher.get("mountinfo_sha256") != sha256_file(extraction / "mountinfo.txt")
    ):
        raise R14Error("R15B isolated extraction boundary changed")


def _verify_recipient(
    run_dir: Path,
    worker_ref: dict[str, Any],
    config: dict[str, Any],
    capabilities: list[Any],
    recipient_rows: list[list[dict[str, Any]]],
    before: torch.Tensor,
    transitions: list[torch.Tensor],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    worker_path = run_dir / str(worker_ref["path"])
    if sha256_file(worker_path) != worker_ref["sha256"]:
        raise R14Error("R15B recipient receipt hash changed")
    worker = json_object(worker_path)
    _verify_evidence(worker, "R15B recipient")
    row_ref = worker["observations"]
    rows = _jsonl(run_dir / str(row_ref["path"]), row_ref["sha256"], int(row_ref["rows"]))
    capability_ids = [item.capability_id for item in capabilities]
    expected = {
        str(row["row_id"]): row for capability_rows in recipient_rows for row in capability_rows
    }
    condition_transitions = {
        capability_id: _conditions(
            before,
            transitions[index],
            transitions[(index + 1) % len(transitions)],
            capability_id,
        )
        for index, capability_id in enumerate(capability_ids)
    }
    executor = RecurrentTransitionNeuralISA().eval()
    seen = set()
    token_ids = [int(value) for value in worker["host_receipt"]["target_token_ids"]]
    for row in rows:
        key = (str(row["capability_id"]), str(row["condition"]), str(row["row_id"]))
        if key in seen or key[0] not in capability_ids or key[2] not in expected:
            raise R14Error("R15B recipient row identity changed")
        seen.add(key)
        index = capability_ids.index(key[0])
        wrong = (index + 1) % len(capability_ids)
        package_sha = {
            "BASE": None,
            "AFTER": manifest["after"][index]["sha256"],
            "BEFORE": manifest["before"]["sha256"],
            "WRONG": manifest["after"][wrong]["sha256"],
            "ZERO": "CONTROL_ZERO",
            "RANDOM": "CONTROL_RANDOM",
            "SHUFFLED": "CONTROL_SHUFFLED",
            "REMOVED": None,
            "BACKEND_REMOVED": manifest["after"][index]["sha256"],
            "CODEC_REMOVED": manifest["after"][index]["sha256"],
            "RESTORED": manifest["after"][index]["sha256"],
        }[key[1]]
        active = key[1] not in {"BASE", "REMOVED", "BACKEND_REMOVED", "CODEC_REMOVED"}
        canonical_probabilities = [float(value) for value in row["canonical_probabilities"]]
        neural_state = [float(value) for value in row["neural_state"]]
        if len(canonical_probabilities) != 8 or len(neural_state) != 8:
            raise R14Error("R15B recipient probability width changed")
        transition = condition_transitions[key[0]][key[1]]
        if active:
            if transition is None:
                raise R14Error("R15B active recipient transition missing")
            recomputed = executor(transition, [str(expected[key[2]]["prompt"])])[0].tolist()
        else:
            recomputed = canonical_probabilities
        if (
            row["prompt_sha256"] != expected[key[2]]["prompt_sha256"]
            or row["package_sha256"] != package_sha
            or row["backend_active"] is not active
            or row["codec_active"] is not active
            or canonical_prediction(int(row["prediction_token_id"]), token_ids)
            != row["canonical_prediction"]
            or not _close(neural_state, recomputed)
        ):
            raise R14Error("R15B recipient row content changed")
    expected_keys = {
        (capability.capability_id, condition, str(row["row_id"]))
        for capability, rows_for_capability in zip(capabilities, recipient_rows)
        for condition in config["conditions"]
        for row in rows_for_capability
    }
    summary = summarize(rows, recipient_rows)
    host = worker["host_receipt"]
    if (
        seen != expected_keys
        or summary != worker["summary"]
        or host["recipient_optimizer_steps"] != 0
        or host["source_model_loaded"] is not False
        or host["model_state_sha256_before"] != host["model_state_sha256_after"]
        or host["codec_sha256_before"] != host["codec_sha256_after"]
        or summary["removal_conditions_equal_base"] is not True
    ):
        raise R14Error("R15B recipient invariant changed")
    maximum = float(config["gates"]["negative_control_accuracy_maximum"])
    for capability_id in capability_ids:
        if (
            summary["accuracy"][f"{capability_id}/AFTER"] != 1.0
            or summary["accuracy"][f"{capability_id}/RESTORED"] != 1.0
            or any(
                summary["accuracy"][f"{capability_id}/{condition}"] > maximum
                for condition in config["negative_conditions"]
            )
        ):
            raise R14Error("R15B recipient accuracy gate failed")
    return summary


def verify(config_path: Path, reveal_path: Path, run_dir: Path) -> dict[str, Any]:
    config = json_object(config_path)
    receipt = json_object(run_dir / "receipt.json")
    _verify_evidence(receipt, "R15B run")
    if (
        config.get("status") != "PREREGISTERED_BEFORE_HELDOUT_REVEAL"
        or receipt.get("format") != "abi-r15b-preexisting-representation-run/1"
        or receipt.get("verification_status") != "UNVERIFIED_RUN_OUTPUT"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("reveal_sha256") != sha256_file(reveal_path)
    ):
        raise R14Error("R15B run identity changed")
    reveal = json_object(reveal_path)
    secret = bytes.fromhex(str(reveal["secret_hex"]))
    if (
        len(secret) != 32
        or hashlib.sha256(secret).hexdigest() != config["heldout_seed_commitment"]
        or reveal.get("commitment") != config["heldout_seed_commitment"]
        or (run_dir / "heldout_reveal.json").read_bytes() != reveal_path.read_bytes()
    ):
        raise R14Error("R15B reveal custody changed")
    root = config_path.parents[3]
    for relative, expected_hash in config["code_bindings"].items():
        if sha256_file(root / relative) != expected_hash:
            raise R14Error(f"R15B code binding changed: {relative}")
    for relative, expected_hash in config["public_preflight"]["files"].items():
        if sha256_file(root / relative) != expected_hash:
            raise R14Error("R15B public preflight binding changed")
    r11 = verify_r11_freeze(root, config)
    if receipt.get("r11_freeze") != r11:
        raise R14Error("R15B R11 freeze changed")
    heldout = heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    capabilities = [item.capability for item in heldout]
    evaluations = evaluation_rows(config, heldout)
    recipient_rows = [
        rows[: int(config["data"]["recipient_rows_per_capability"])] for rows in evaluations
    ]
    if receipt["capabilities"] != [item.capability_id for item in capabilities]:
        raise R14Error("R15B held-out capability inventory changed")
    source_ref = receipt["source"]["observations"]
    source_rows = _jsonl(
        run_dir / source_ref["path"], source_ref["sha256"], int(source_ref["rows"])
    )
    grouped = {
        capability.capability_id: [
            row for row in source_rows if row["capability_id"] == capability.capability_id
        ]
        for capability in capabilities
    }
    manifest = json_object(run_dir / "package_manifest.json")
    if (
        manifest != receipt["packages"]
        or sha256_file(run_dir / "package_manifest.json") != receipt["package_manifest_sha256"]
    ):
        raise R14Error("R15B package manifest changed")
    before_package, before = load_package(run_dir / "packages" / manifest["before"]["path"])
    if before_package["transition_sha256"] != manifest["before"]["transition_sha256"]:
        raise R14Error("R15B before package changed")
    transitions = []
    recomputed_controls = []
    for index, (item, evaluation, source_item, extraction_item, label_item) in enumerate(
        zip(
            heldout,
            evaluations,
            receipt["source"]["capability_receipts"],
            receipt["isolated_extractions"],
            receipt["semantic_labels"],
        )
    ):
        expected_anchors = _anchor_rows(item.slot_order)
        observed = grouped[item.capability.capability_id]
        if len(observed) != 6 or source_item["anchor_accuracy"] != 1.0:
            raise R14Error("R15B source anchor completeness changed")
        for reference, row in zip(expected_anchors, observed):
            probabilities = [float(value) for value in row["canonical_probabilities"]]
            if (
                row["row_id"] != reference["row_id"]
                or row["prompt_sha256"] != reference["prompt_sha256"]
                or int(row["answer"]) != int(reference["answer"])
                or int(row["prediction"]) != int(reference["answer"])
                or len(probabilities) != 8
                or max(range(8), key=lambda position: probabilities[position])
                != int(reference["answer"])
            ):
                raise R14Error("R15B source anchor changed")
        bundle_ref = source_item["bundle"]
        bundle_path = run_dir / bundle_ref["path"]
        if (
            not bundle_path.is_file()
            or bundle_path.stat().st_size != bundle_ref["bytes"]
            or sha256_file(bundle_path) != bundle_ref["sha256"]
        ):
            raise R14Error("R15B anonymous representation bundle changed")
        with safe_open(str(bundle_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        bundle = load_file(str(bundle_path), device="cpu")
        if metadata != {"format": "abi-r15b-anonymous-pre-answer-representation/1"} or set(
            bundle
        ) != {"residuals", "output_rows"}:
            raise R14Error("R15B representation bundle schema changed")
        residuals = bundle["residuals"].float().contiguous()
        output_rows = bundle["output_rows"].float().contiguous()
        labels = decode_labels(residuals, output_rows)
        expected_operations = [
            (OPERATIONS[semantic][1], OPERATIONS[semantic][2]) for semantic in item.slot_order
        ]
        if labels_to_operations(labels) != expected_operations:
            raise R14Error("R15B representation decoder changed")
        extraction = run_dir / extraction_item["path"]
        if (
            sha256_file(extraction / "result.json") != extraction_item["result_sha256"]
            or sha256_file(extraction / "launcher.json") != extraction_item["launcher_sha256"]
        ):
            raise R14Error("R15B isolated extraction receipt changed")
        _verify_isolated(extraction, bundle_ref["sha256"], labels)
        package_ref = manifest["after"][index]
        package_path = run_dir / "packages" / package_ref["path"]
        package, transition = load_package(package_path)
        decoded = decode_transition(residuals, output_rows)
        if (
            sha256_file(package_path) != package_ref["sha256"]
            or package["transition_sha256"] != package_ref["transition_sha256"]
            or transition_bytes(transition) != transition_bytes(decoded)
            or transition_accuracy(transition, evaluation) != 1.0
        ):
            raise R14Error("R15B package identity or evaluation changed")
        label_path = run_dir / label_item["path"]
        if sha256_file(label_path) != label_item["sha256"]:
            raise R14Error("R15B semantic label file changed")
        label = json_object(label_path)
        _verify_evidence(label, "R15B semantic label")
        if (
            label.get("namespace") != LABEL_NAMESPACE
            or label.get("package_sha256") != package_ref["sha256"]
            or label.get("capability_id") != item.capability.capability_id
            or label.get("operation_labels") != item.operation_labels
        ):
            raise R14Error("R15B semantic label binding changed")
        values = {
            name: _control_accuracy(residuals, output_rows, evaluation, control=name)
            for name in ("zero", "shuffle", "random", "head_shuffle")
        }
        transitions.append(transition)
        recomputed_controls.append(values)
    for index, values in enumerate(recomputed_controls):
        values["wrong"] = transition_accuracy(
            transitions[(index + 1) % len(transitions)], evaluations[index]
        )
        stored = {
            key: value
            for key, value in receipt["controls"][index].items()
            if key != "capability_id"
        }
        if values != stored or max(values.values()) > float(
            config["gates"]["negative_control_accuracy_maximum"]
        ):
            raise R14Error("R15B negative-control recomputation failed")
    summaries = {
        item["host"]: _verify_recipient(
            run_dir,
            item,
            config,
            capabilities,
            recipient_rows,
            before,
            transitions,
            manifest,
        )
        for item in receipt["recipient_workers"]
    }
    if sorted(summaries) != sorted(config["recipient_hosts"]):
        raise R14Error("R15B recipient inventory changed")
    return {
        "format": "abi-r15b-strict-verification/1",
        "verdict": "PASS",
        "claim": "BOUNDED_PREEXISTING_REPRESENTATION_CAPABILITY_RECOVERY",
        "capabilities_exact": len(transitions),
        "package_evaluation_rows": sum(len(rows) for rows in evaluations),
        "recipient_hosts": sorted(summaries),
        "source_training_steps": 0,
        "teacher_present_at_recipient_execution": False,
        "claim_ceiling": "NOT_ENGLISH_DOMAIN_OR_TEACHER_BEHAVIOR_CLONING",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.config, args.reveal, args.run_dir)
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
