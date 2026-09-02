"""Fail-closed recomputation of the R14 non-exhaustive recovery claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from experiments.copy_paste_r10.runtime import canonical_prediction
from experiments.foreign_teacher_r12.custody import verify_r11_freeze
from experiments.native_isa_r11.core import (
    load_package,
    sha256_bytes,
    transition_accuracy,
    transition_bytes,
)
from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .capability import heldout_capabilities
from .core import (
    R14Error,
    behavior_receipt,
    capability_rows,
    evidence_hash,
    json_object,
    sha256_file,
    source_metrics,
    write_json_once,
)
from .extractor import extract_transition, extractor_spec
from .recipient_worker import summarize
from .run import _adapter_state_sha256, _recipient_gate, _source_gate

SOURCE_FIELDS = {
    "capability_id",
    "split",
    "row_id",
    "prompt_sha256",
    "start",
    "program",
    "canonical_probabilities",
}


def _verify_evidence(value: dict[str, Any], label: str) -> None:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    if stored != evidence_hash(payload):
        raise R14Error(f"{label} evidence hash changed")


def _jsonl(path: Path, expected_sha: str, expected_rows: int) -> list[dict[str, Any]]:
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise R14Error(f"raw evidence unavailable: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R14Error(f"raw evidence unreadable: {path}") from exc
    if len(rows) != expected_rows or not all(isinstance(row, dict) for row in rows):
        raise R14Error(f"raw row count or schema changed: {path}")
    return rows


def _probabilities(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 8:
        raise R14Error("probability width changed")
    numbers = [float(item) for item in value]
    if (
        any(not math.isfinite(item) or item < 0 or item > 1.000001 for item in numbers)
        or abs(sum(numbers) - 1.0) > 2e-5
    ):
        raise R14Error("probabilities invalid")


def _operation_commitment(operations: Any) -> str:
    return hashlib.sha256(canonical_json_bytes([list(item) for item in operations])).hexdigest()


def verify(config_path: Path, reveal_path: Path, run_dir: Path) -> dict[str, Any]:
    config = json_object(config_path)
    receipt = json_object(run_dir / "receipt.json")
    _verify_evidence(receipt, "run receipt")
    if config.get("status") != "PREREGISTERED_BEFORE_HELDOUT_REVEAL":
        raise R14Error("R14 preregistration status changed")
    for preflight in config.get("public_preflights", []):
        path = config_path.parents[3] / str(preflight["file"])
        value = json_object(path)
        payload = dict(value)
        stored = payload.pop("evidence_sha256", None)
        if (
            not path.is_file()
            or sha256_file(path) != str(preflight["file_sha256"])
            or stored != str(preflight["evidence_sha256"])
            or stored != evidence_hash(payload)
        ):
            raise R14Error("bound public preflight changed")
    if (
        receipt.get("format") != "abi-r14-non-exhaustive-capability-extraction/1"
        or receipt.get("claim_target") != "NON_EXHAUSTIVE_FOREIGN_CAPABILITY_RECOVERY"
        or receipt.get("claim_ceiling") != "NOT_LOSSLESS_TEACHER_CLONING_NOT_PREEXISTING_KNOWLEDGE"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("extractor") != extractor_spec()
        or receipt.get("behavior") != behavior_receipt(config)
    ):
        raise R14Error("run identity or claim boundary changed")
    preserved_reveal = run_dir / str(receipt["heldout"]["reveal_path"])
    if (
        not preserved_reveal.is_file()
        or sha256_file(preserved_reveal) != receipt["reveal_sha256"]
        or preserved_reveal.read_bytes() != reveal_path.read_bytes()
    ):
        raise R14Error("held-out reveal changed")
    reveal = json_object(reveal_path)
    try:
        secret = bytes.fromhex(str(reveal["secret_hex"]))
    except ValueError as exc:
        raise R14Error("held-out reveal malformed") from exc
    if (
        len(secret) != 32
        or sha256_bytes(secret) != config["heldout_seed_commitment"]
        or reveal.get("commitment") != config["heldout_seed_commitment"]
    ):
        raise R14Error("held-out commitment changed")
    root = config_path.parents[3]
    r11 = verify_r11_freeze(root, config)
    if receipt.get("r11_freeze") != r11:
        raise R14Error("R11 freeze receipt changed")
    capabilities = heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    generated = capability_rows(config, capabilities)
    capability_ids = [item.capability_id for item in capabilities]
    if receipt["data"]["capabilities"] != capability_ids:
        raise R14Error("held-out capability inventory changed")
    source_reference = receipt["source_observations"]
    raw_source = _jsonl(
        run_dir / source_reference["path"],
        source_reference["sha256"],
        source_reference["rows"],
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_source:
        if set(row) != SOURCE_FIELDS:
            raise R14Error("source observation schema changed")
        if row["capability_id"] not in capability_ids or row["split"] not in {
            "BEFORE",
            "QUERY",
            "EVALUATION",
            "COUNTERFACTUAL",
        }:
            raise R14Error("source observation identity changed")
        _probabilities(row["canonical_probabilities"])
        grouped[(str(row["capability_id"]), str(row["split"]))].append(
            {key: row[key] for key in SOURCE_FIELDS - {"capability_id", "split"}}
        )
    manifest_path = run_dir / "package_manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != receipt["manifest_sha256"]:
        raise R14Error("package manifest changed")
    manifest = json_object(manifest_path)
    if manifest != receipt["packages"]:
        raise R14Error("package manifest receipt mismatch")
    declared_packages = [manifest["before"], *manifest["after"]]
    if {path.name for path in (run_dir / "packages").glob("*.abipkg")} != {
        str(item["path"]) for item in declared_packages
    }:
        raise R14Error("undeclared or missing package")
    before_path = run_dir / "packages" / str(manifest["before"]["path"])
    before_package, _ = load_package(before_path)
    if (
        sha256_file(before_path) != manifest["before"]["sha256"]
        or before_package["transition_sha256"] != manifest["before"]["transition_sha256"]
    ):
        raise R14Error("before package changed")
    recomputed_sources = []
    package_accuracy: dict[str, float] = {}
    for index, capability in enumerate(capabilities):
        capability_id = capability.capability_id
        rows = generated[index]
        split_rows = {
            "BEFORE": rows["evaluation"][: int(config["data"]["before_rows_per_capability"])],
            "QUERY": rows["queries"],
            "EVALUATION": rows["evaluation"],
            "COUNTERFACTUAL": rows["counterfactual"],
        }
        for split, expected in split_rows.items():
            actual = grouped[(capability_id, split)]
            reference = {str(row["row_id"]): row for row in expected}
            if len(actual) != len(expected):
                raise R14Error(f"source coverage changed: {capability_id}/{split}")
            seen = set()
            for observation in actual:
                row_id = str(observation["row_id"])
                target = reference.get(row_id)
                if (
                    row_id in seen
                    or target is None
                    or observation["prompt_sha256"] != target["prompt_sha256"]
                    or int(observation["start"]) != int(target["start"])
                    or [int(item) for item in observation["program"]]
                    != [int(item) for item in target["program"]]
                ):
                    raise R14Error("source row content changed")
                seen.add(row_id)
        safe_queries = grouped[(capability_id, "QUERY")]
        transition, extraction = extract_transition(safe_queries)
        source_receipt = receipt["source_acquisitions"][index]
        _verify_evidence(source_receipt["extraction"], f"extraction {capability_id}")
        if (
            source_receipt["capability_id"] != capability_id
            or extraction != source_receipt["extraction"]
            or extraction["selected_operations_commitment"]
            != _operation_commitment(capability.operations)
        ):
            raise R14Error("extracted latent program changed or is incorrect")
        package_item = manifest["after"][index]
        package_path = run_dir / "packages" / str(package_item["path"])
        package, package_transition = load_package(package_path)
        if (
            package_item["capability_id"] != capability_id
            or sha256_file(package_path) != package_item["sha256"]
            or package["transition_sha256"] != package_item["transition_sha256"]
            or transition_bytes(package_transition) != transition_bytes(transition)
            or package["provenance"]["teacher_before_sha256"]
            != source_receipt["base_state_sha256_before"]
            or package["provenance"]["teacher_after_sha256"]
            != source_receipt["adapter_state_sha256"]
        ):
            raise R14Error("package identity, transition, or provenance changed")
        adapter = run_dir / source_receipt["adapter_artifact"]["path"]
        if (
            not adapter.is_file()
            or adapter.stat().st_size != source_receipt["adapter_artifact"]["bytes"]
            or sha256_file(adapter) != source_receipt["adapter_artifact"]["sha256"]
        ):
            raise R14Error("source adapter unavailable")
        adapter_state = load_file(str(adapter), device="cpu")
        inventory = source_receipt["adapter_inventory"]
        inventory_rows = inventory["modules"]
        if (
            _adapter_state_sha256(adapter_state) != source_receipt["adapter_state_sha256"]
            or sum(value.numel() for value in adapter_state.values())
            != int(source_receipt["training"]["trainable_parameters"])
            or inventory["module_count"] != len(inventory_rows)
            or inventory["trainable_parameters"]
            != int(source_receipt["training"]["trainable_parameters"])
            or inventory["inventory_sha256"]
            != hashlib.sha256(canonical_json_bytes(inventory_rows)).hexdigest()
            or len(adapter_state) != 2 * int(inventory["module_count"])
        ):
            raise R14Error("source adapter tensor identity changed")
        training = source_receipt["training"]
        source_config = config["source_acquisition"]
        expected_steps = int(source_config["steps"])
        if (
            source_receipt["model_id"] != "Qwen/Qwen2.5-0.5B"
            or source_receipt["revision"] != "060db6499f32faf8b98477b0a26969ef7d8b9987"
            or source_receipt["base_state_sha256_before"]
            != source_receipt["base_state_sha256_after"]
            or training["steps"] != expected_steps
            or training["objective"] != source_config["objective"]
            or training["batch_size"] != int(source_config["batch_size"])
            or training["learning_rate"] != float(source_config["learning_rate"])
            or training["sampling"] != "depth_balanced"
            or training["schedule_selection_used_heldout"] is not False
            or [item["step"] for item in training["checkpoints"]]
            != list(
                range(
                    int(source_config["checkpoint_interval"]),
                    expected_steps + 1,
                    int(source_config["checkpoint_interval"]),
                )
            )
            or any("public_development" in item for item in training["checkpoints"])
            or not all(
                math.isfinite(float(value))
                for value in (
                    training["first_loss"],
                    training["final_loss"],
                    training["wall_seconds"],
                    source_receipt["extraction_wall_seconds"],
                )
            )
            or float(training["wall_seconds"]) <= 0
            or float(source_receipt["extraction_wall_seconds"]) <= 0
        ):
            raise R14Error("source acquisition custody changed")
        metrics = {
            "before": source_metrics(grouped[(capability_id, "BEFORE")], split_rows["BEFORE"]),
            "query": source_metrics(safe_queries, split_rows["QUERY"]),
            "evaluation": source_metrics(
                grouped[(capability_id, "EVALUATION")], split_rows["EVALUATION"]
            ),
            "counterfactual": source_metrics(
                grouped[(capability_id, "COUNTERFACTUAL")],
                split_rows["COUNTERFACTUAL"],
            ),
        }
        for label, value in metrics.items():
            if value != source_receipt[label]:
                raise R14Error(f"source metric changed: {capability_id}/{label}")
        unseen_accuracy = transition_accuracy(package_transition, rows["evaluation"])
        counterfactual_accuracy = transition_accuracy(package_transition, rows["counterfactual"])
        if (
            unseen_accuracy != source_receipt["package_unseen_oracle_accuracy"]
            or counterfactual_accuracy != source_receipt["package_counterfactual_oracle_accuracy"]
            or source_receipt["package_source_unseen_agreement"]
            != metrics["evaluation"]["accuracy"]
            or source_receipt["package_source_counterfactual_agreement"]
            != metrics["counterfactual"]["accuracy"]
        ):
            raise R14Error("package oracle metric changed")
        package_accuracy[capability_id] = unseen_accuracy
        recomputed_sources.append({**source_receipt, **metrics, "extraction": extraction})
    if not _source_gate(config, recomputed_sources):
        raise R14Error("source or extraction gate failed")
    recipient_receipts = receipt["recipient_workers"]
    if [item["host"] for item in recipient_receipts] != list(config["recipient_hosts"]):
        raise R14Error("recipient host inventory changed")
    if len({int(item["pid"]) for item in recipient_receipts}) != len(recipient_receipts):
        raise R14Error("recipient workers did not use distinct processes")
    expected_recipient = {
        str(row["row_id"]): row for item in generated for row in item["recipient"]
    }
    for worker in recipient_receipts:
        _verify_evidence(worker, f"recipient {worker.get('host')}")
        if (
            worker.get("config_sha256") != sha256_file(config_path)
            or worker.get("reveal_sha256") != sha256_file(reveal_path)
            or worker.get("manifest_sha256") != sha256_file(manifest_path)
            or worker.get("source_adapter_argument_present") is not False
            or worker.get("source_adapter_loaded") is not False
        ):
            raise R14Error("recipient custody changed")
        host_dir = run_dir / "recipients" / str(worker["host"])
        rows = _jsonl(
            host_dir / worker["observations"]["path"],
            worker["observations"]["sha256"],
            worker["observations"]["rows"],
        )
        host = worker["host_receipt"]
        token_ids = [int(item) for item in host["target_token_ids"]]
        seen = set()
        for row in rows:
            key = (
                str(row["capability_id"]),
                str(row["condition"]),
                str(row["row_id"]),
            )
            reference = expected_recipient.get(key[2])
            capability_index = capability_ids.index(key[0]) if key[0] in capability_ids else -1
            wrong_index = (capability_index + 1) % len(capability_ids)
            expected_package = {
                "BASE": None,
                "AFTER": manifest["after"][capability_index]["sha256"],
                "BEFORE": manifest["before"]["sha256"],
                "WRONG": manifest["after"][wrong_index]["sha256"],
                "ZERO": "CONTROL_ZERO",
                "RANDOM": "CONTROL_RANDOM",
                "SHUFFLED": "CONTROL_SHUFFLED",
                "REMOVED": None,
                "BACKEND_REMOVED": manifest["after"][capability_index]["sha256"],
                "CODEC_REMOVED": manifest["after"][capability_index]["sha256"],
                "RESTORED": manifest["after"][capability_index]["sha256"],
            }.get(key[1])
            active = key[1] not in {
                "BASE",
                "REMOVED",
                "BACKEND_REMOVED",
                "CODEC_REMOVED",
            }
            if (
                key in seen
                or reference is None
                or row["prompt_sha256"] != reference["prompt_sha256"]
                or row["package_sha256"] != expected_package
                or row["backend_active"] is not active
                or row["codec_active"] is not active
                or canonical_prediction(int(row["prediction_token_id"]), token_ids)
                != row["canonical_prediction"]
            ):
                raise R14Error("recipient row identity changed")
            seen.add(key)
            _probabilities(row["canonical_probabilities"])
            if not isinstance(row["neural_state"], list) or len(row["neural_state"]) != 8:
                raise R14Error("recipient neural state changed")
            _probabilities(row["neural_state"])
        expected_keys = {
            (capability_id, str(condition), str(row["row_id"]))
            for capability_id, item in zip(capability_ids, generated)
            for condition in config["conditions"]
            for row in item["recipient"]
        }
        if seen != expected_keys:
            raise R14Error("recipient observation coverage changed")
        evaluations = [item["recipient"] for item in generated]
        if worker["summary"] != summarize(rows, evaluations):
            raise R14Error("recipient summary changed")
    if not _recipient_gate(config, recipient_receipts):
        raise R14Error("recipient gate failed")
    result = {
        "format": "abi-r14-strict-verification/1",
        "verdict": "PASS",
        "claim": "NON_EXHAUSTIVE_FOREIGN_CAPABILITY_RECOVERY",
        "capabilities": len(capabilities),
        "package_unseen_oracle_accuracy": package_accuracy,
        "evaluation_behavior_space_per_capability": behavior_receipt(config)[
            "evaluation_behavior_space"
        ],
        "extractor_queries_per_capability": int(config["data"]["extractor_queries_per_capability"]),
        "atomic_extractor_queries": 0,
        "source_evaluation_accuracy": {
            item["capability_id"]: item["evaluation"]["accuracy"] for item in recomputed_sources
        },
        "source_counterfactual_accuracy": {
            item["capability_id"]: item["counterfactual"]["accuracy"] for item in recomputed_sources
        },
        "r11_freeze_evidence_sha256": r11["evidence_sha256"],
        "run_evidence_sha256": receipt["evidence_sha256"],
        "stored_status_booleans_consumed": 0,
        "claim_ceiling": "NOT_LOSSLESS_TEACHER_CLONING_NOT_PREEXISTING_KNOWLEDGE",
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        result = verify(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.run_dir).resolve(),
        )
        write_json_once(output, result)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
