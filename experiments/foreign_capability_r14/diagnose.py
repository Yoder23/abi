"""Fail-closed recomputation and diagnosis of an R14 pass or negative result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

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
from .run import _adapter_state_sha256
from .verify import SOURCE_FIELDS, _jsonl, _probabilities, _verify_evidence


def _operation_commitment(operations: Any) -> str:
    return hashlib.sha256(canonical_json_bytes([list(item) for item in operations])).hexdigest()


def diagnose(config_path: Path, reveal_path: Path, run_dir: Path) -> dict[str, Any]:
    config = json_object(config_path)
    receipt = json_object(run_dir / "receipt.json")
    _verify_evidence(receipt, "run receipt")
    root = config_path.parents[3]
    if (
        config.get("status") != "PREREGISTERED_BEFORE_HELDOUT_REVEAL"
        or receipt.get("format") != "abi-r14-non-exhaustive-capability-extraction/1"
        or receipt.get("claim_target") != "NON_EXHAUSTIVE_FOREIGN_CAPABILITY_RECOVERY"
        or receipt.get("claim_ceiling") != "NOT_LOSSLESS_TEACHER_CLONING_NOT_PREEXISTING_KNOWLEDGE"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("extractor") != extractor_spec()
        or receipt.get("behavior") != behavior_receipt(config)
    ):
        raise R14Error("run identity or claim boundary changed")
    for preflight in config["public_preflights"]:
        path = root / str(preflight["file"])
        value = json_object(path)
        payload = dict(value)
        stored = payload.pop("evidence_sha256", None)
        if (
            sha256_file(path) != str(preflight["file_sha256"])
            or stored != str(preflight["evidence_sha256"])
            or stored != evidence_hash(payload)
        ):
            raise R14Error("bound public preflight changed")
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
    r11 = verify_r11_freeze(root, config)
    if receipt.get("r11_freeze") != r11:
        raise R14Error("R11 freeze changed")
    capabilities = heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    generated = capability_rows(config, capabilities)
    capability_ids = [item.capability_id for item in capabilities]
    expected_data = {
        "capabilities": capability_ids,
        "training_rows_per_capability": len(generated[0]["training"]),
        "query_rows_per_capability": len(generated[0]["queries"]),
        "evaluation_rows_per_capability": len(generated[0]["evaluation"]),
        "counterfactual_rows_per_capability": len(generated[0]["counterfactual"]),
        "recipient_rows_per_capability": len(generated[0]["recipient"]),
        "all_split_overlap": 0,
    }
    if receipt.get("data") != expected_data:
        raise R14Error("data receipt changed")
    source_reference = receipt["source_observations"]
    raw_source = _jsonl(
        run_dir / source_reference["path"],
        source_reference["sha256"],
        source_reference["rows"],
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_source = set()
    for row in raw_source:
        if set(row) != SOURCE_FIELDS:
            raise R14Error("source observation schema changed")
        key = (str(row["capability_id"]), str(row["split"]), str(row["row_id"]))
        if (
            key in seen_source
            or key[0] not in capability_ids
            or key[1] not in {"BEFORE", "QUERY", "EVALUATION", "COUNTERFACTUAL"}
        ):
            raise R14Error("source observation identity changed")
        seen_source.add(key)
        _probabilities(row["canonical_probabilities"])
        grouped[key[:2]].append(
            {field: row[field] for field in SOURCE_FIELDS - {"capability_id", "split"}}
        )
    manifest_path = run_dir / "package_manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != receipt["manifest_sha256"]:
        raise R14Error("package manifest changed")
    manifest = json_object(manifest_path)
    if manifest != receipt["packages"] or len(manifest.get("after", [])) != len(capabilities):
        raise R14Error("package manifest receipt changed")
    declared = [manifest["before"], *manifest["after"]]
    if {path.name for path in (run_dir / "packages").glob("*.abipkg")} != {
        str(item["path"]) for item in declared
    }:
        raise R14Error("package inventory changed")
    before_path = run_dir / "packages" / str(manifest["before"]["path"])
    before_package, _ = load_package(before_path)
    if (
        sha256_file(before_path) != manifest["before"]["sha256"]
        or before_package["transition_sha256"] != manifest["before"]["transition_sha256"]
    ):
        raise R14Error("before package changed")
    if len(receipt["source_acquisitions"]) != len(capabilities):
        raise R14Error("source receipt count changed")
    chance = float(config["gates"]["source_chance_accuracy"])
    per_capability = []
    all_failures = []
    for index, capability in enumerate(capabilities):
        capability_id = capability.capability_id
        rows = generated[index]
        source = receipt["source_acquisitions"][index]
        if source["capability_id"] != capability_id:
            raise R14Error("source capability order changed")
        expected_splits = {
            "BEFORE": rows["evaluation"][: int(config["data"]["before_rows_per_capability"])],
            "QUERY": rows["queries"],
            "EVALUATION": rows["evaluation"],
            "COUNTERFACTUAL": rows["counterfactual"],
        }
        metrics = {}
        for split, expected_rows in expected_splits.items():
            observations = grouped[(capability_id, split)]
            reference = {str(row["row_id"]): row for row in expected_rows}
            if len(observations) != len(reference):
                raise R14Error("source observation coverage changed")
            for observation in observations:
                expected = reference.get(str(observation["row_id"]))
                if (
                    expected is None
                    or observation["prompt_sha256"] != expected["prompt_sha256"]
                    or int(observation["start"]) != int(expected["start"])
                    or [int(item) for item in observation["program"]]
                    != [int(item) for item in expected["program"]]
                ):
                    raise R14Error("source observation content changed")
            metrics[split.lower()] = source_metrics(observations, expected_rows)
        for label in ("before", "query", "evaluation", "counterfactual"):
            if metrics[label] != source[label]:
                raise R14Error(f"source metric changed: {capability_id}/{label}")
        query_observations = grouped[(capability_id, "QUERY")]
        transition, extraction = extract_transition(query_observations)
        _verify_evidence(source["extraction"], f"extraction {capability_id}")
        if extraction != source["extraction"]:
            raise R14Error("extraction receipt changed")
        package_item = manifest["after"][index]
        package_path = run_dir / "packages" / str(package_item["path"])
        package, package_transition = load_package(package_path)
        if (
            package_item["capability_id"] != capability_id
            or sha256_file(package_path) != package_item["sha256"]
            or transition_bytes(package_transition) != transition_bytes(transition)
            or package["provenance"]["teacher_before_sha256"] != source["base_state_sha256_before"]
            or package["provenance"]["teacher_after_sha256"] != source["adapter_state_sha256"]
        ):
            raise R14Error("package identity or provenance changed")
        adapter = run_dir / source["adapter_artifact"]["path"]
        if (
            not adapter.is_file()
            or adapter.stat().st_size != source["adapter_artifact"]["bytes"]
            or sha256_file(adapter) != source["adapter_artifact"]["sha256"]
        ):
            raise R14Error("source adapter unavailable")
        adapter_state = load_file(str(adapter), device="cpu")
        inventory = source["adapter_inventory"]
        if (
            _adapter_state_sha256(adapter_state) != source["adapter_state_sha256"]
            or sum(value.numel() for value in adapter_state.values())
            != int(source["training"]["trainable_parameters"])
            or inventory["inventory_sha256"]
            != hashlib.sha256(canonical_json_bytes(inventory["modules"])).hexdigest()
        ):
            raise R14Error("source adapter tensor identity changed")
        training = source["training"]
        source_config = config["source_acquisition"]
        if (
            source["model_id"] != "Qwen/Qwen2.5-0.5B"
            or source["revision"] != "060db6499f32faf8b98477b0a26969ef7d8b9987"
            or source["base_state_sha256_before"] != source["base_state_sha256_after"]
            or training["steps"] != int(source_config["steps"])
            or training["objective"] != source_config["objective"]
            or training["batch_size"] != int(source_config["batch_size"])
            or training["schedule_selection_used_heldout"] is not False
            or any("public_development" in item for item in training["checkpoints"])
            or not all(
                math.isfinite(float(value))
                for value in (
                    training["first_loss"],
                    training["final_loss"],
                    training["wall_seconds"],
                    source["extraction_wall_seconds"],
                )
            )
        ):
            raise R14Error("source training custody changed")
        unseen_accuracy = transition_accuracy(package_transition, rows["evaluation"])
        counterfactual_accuracy = transition_accuracy(package_transition, rows["counterfactual"])
        if (
            unseen_accuracy != source["package_unseen_oracle_accuracy"]
            or counterfactual_accuracy != source["package_counterfactual_oracle_accuracy"]
        ):
            raise R14Error("package score changed")
        selected_correct = extraction["selected_operations_commitment"] == _operation_commitment(
            capability.operations
        )
        failures = []
        for split in ("query", "evaluation", "counterfactual"):
            if metrics[split]["wilson_95_lower"] <= chance:
                failures.append(f"SOURCE_{split.upper()}_NOT_ABOVE_CHANCE")
        if not selected_correct:
            failures.append("LATENT_PROGRAM_NOT_RECOVERED")
        if extraction["log_likelihood_margin"] <= float(
            config["gates"]["extractor_log_likelihood_margin_minimum"]
        ):
            failures.append("NONPOSITIVE_SELECTION_MARGIN")
        if unseen_accuracy != 1.0:
            failures.append("UNSEEN_PACKAGE_NOT_EXACT")
        if counterfactual_accuracy != 1.0:
            failures.append("COUNTERFACTUAL_PACKAGE_NOT_EXACT")
        all_failures.extend(f"{capability_id}/{failure}" for failure in failures)
        per_capability.append(
            {
                "capability_id": capability_id,
                "source_query": metrics["query"],
                "source_unseen": metrics["evaluation"],
                "source_counterfactual": metrics["counterfactual"],
                "selected_program_correct": selected_correct,
                "selection_log_likelihood_margin": extraction["log_likelihood_margin"],
                "package_unseen_oracle_accuracy": unseen_accuracy,
                "package_counterfactual_oracle_accuracy": counterfactual_accuracy,
                "failures": failures,
            }
        )
    passed = not all_failures
    if passed and not receipt["recipient_workers"]:
        raise R14Error("passing source result omitted recipient evidence")
    if not passed and receipt["recipient_workers"]:
        raise R14Error("failed source result unexpectedly ran recipients")
    result = {
        "format": "abi-r14-outcome-diagnosis/1",
        "verdict": "PASS" if passed else "FAIL",
        "claim": "NON_EXHAUSTIVE_FOREIGN_CAPABILITY_RECOVERY",
        "capabilities_passing": sum(not item["failures"] for item in per_capability),
        "capabilities_total": len(per_capability),
        "per_capability": per_capability,
        "failures": all_failures,
        "recipient_execution": "COMPLETED" if passed else "NOT_RUN_SOURCE_GATE_FAILED",
        "evaluation_behavior_space_per_capability": behavior_receipt(config)[
            "evaluation_behavior_space"
        ],
        "extractor_queries_per_capability": int(config["data"]["extractor_queries_per_capability"]),
        "atomic_extractor_queries": 0,
        "stored_status_booleans_consumed": 0,
        "run_evidence_sha256": receipt["evidence_sha256"],
        "r11_freeze_evidence_sha256": r11["evidence_sha256"],
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
        result = diagnose(
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
