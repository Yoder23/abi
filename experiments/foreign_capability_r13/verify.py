"""Fail-closed recomputation of the R13-B bounded capability-extraction claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.copy_paste_r10.runtime import canonical_prediction
from experiments.foreign_teacher_r12.custody import verify_r11_freeze
from experiments.foreign_teacher_r12.extractor import extractor_spec
from experiments.native_isa_r11.core import (
    load_package,
    sha256_bytes,
    sha256_file,
    transition_accuracy,
)
from experiments.native_transfer_r8.capability_generator import (
    committed_heldout_capabilities,
)

from .core import R13Error, capability_rows, evidence_hash, json_object, write_json_once
from .recipient_worker import _summarize
from .run import _recipient_pass

SOURCE_FIELDS = {
    "capability_id",
    "condition",
    "row_id",
    "prompt_sha256",
    "answer",
    "native_prediction_token_id",
    "canonical_prediction",
    "canonical_probabilities",
}


def _evidence(value: dict[str, Any], label: str) -> None:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    if stored != evidence_hash(payload):
        raise R13Error(f"{label} evidence hash changed")


def _jsonl(path: Path, expected_sha: str, expected_rows: int) -> list[dict[str, Any]]:
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise R13Error(f"raw evidence unavailable: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R13Error(f"raw evidence unreadable: {path}") from exc
    if len(rows) != expected_rows or not all(isinstance(row, dict) for row in rows):
        raise R13Error(f"raw evidence row count or schema changed: {path}")
    return rows


def _probabilities(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 8:
        raise R13Error("canonical probability width changed")
    numbers = [float(item) for item in value]
    if (
        any(not math.isfinite(item) or item < 0 or item > 1.000001 for item in numbers)
        or abs(sum(numbers) - 1.0) > 2e-5
    ):
        raise R13Error("canonical probabilities invalid")


def _source_metrics(
    rows: list[dict[str, Any]],
    target_ids: list[int],
) -> dict[str, Any]:
    correct = sum(
        int(int(row["native_prediction_token_id"]) == target_ids[int(row["answer"])])
        for row in rows
    )
    canonical = sum(
        int(int(row["canonical_prediction"]) == int(row["answer"])) for row in rows
    )
    prediction_hash = hashlib.sha256(
        ",".join(str(int(row["native_prediction_token_id"])) for row in rows).encode()
    ).hexdigest()
    return {
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "canonical_correct": canonical,
        "canonical_accuracy": canonical / len(rows),
        "prediction_ids_sha256": prediction_hash,
    }


def verify(config_path: Path, reveal_path: Path, run_dir: Path) -> dict[str, Any]:
    config = json_object(config_path)
    receipt = json_object(run_dir / "receipt.json")
    _evidence(receipt, "run receipt")
    if config.get("status") != "PREREGISTERED_BEFORE_HELDOUT_REVEAL":
        raise R13Error("R13 preregistration status changed")
    if (
        receipt.get("format") != "abi-r13-bounded-capability-extraction/1"
        or receipt.get("claim_target") != "BOUNDED_ENUMERABLE_CAPABILITY_EXTRACTION"
        or receipt.get("claim_ceiling") != "NOT_BEHAVIORAL_TRANSPLANT_NOT_GENERAL_KNOWLEDGE"
        or receipt.get("config_sha256") != sha256_file(config_path)
    ):
        raise R13Error("run identity or claim ceiling changed")
    preserved_reveal = run_dir / str(receipt["heldout"]["reveal_path"])
    if (
        not preserved_reveal.is_file()
        or sha256_file(preserved_reveal) != receipt["reveal_sha256"]
        or preserved_reveal.read_bytes() != reveal_path.read_bytes()
    ):
        raise R13Error("held-out reveal changed")
    reveal = json_object(reveal_path)
    try:
        secret = bytes.fromhex(str(reveal["secret_hex"]))
    except ValueError as exc:
        raise R13Error("held-out reveal malformed") from exc
    if (
        len(secret) != 32
        or sha256_bytes(secret) != config["heldout_seed_commitment"]
        or reveal.get("commitment") != config["heldout_seed_commitment"]
    ):
        raise R13Error("held-out commitment changed")
    root = config_path.parents[3]
    r11 = verify_r11_freeze(root, config)
    if receipt.get("r11_freeze") != r11:
        raise R13Error("R11 freeze receipt changed")
    capabilities = committed_heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    training, evaluation, atomic = capability_rows(config, capabilities)
    if any(
        {row["prompt_sha256"] for row in train}
        & {row["prompt_sha256"] for row in test}
        for train, test in zip(training, evaluation)
    ):
        raise R13Error("source/evaluation prompt overlap recomputed")
    capability_ids = [item.capability_id for item in capabilities]
    if receipt["data"]["capabilities"] != capability_ids:
        raise R13Error("held-out capability inventory changed")
    manifest_path = run_dir / "package_manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != receipt["manifest_sha256"]:
        raise R13Error("package manifest changed")
    manifest = json_object(manifest_path)
    if manifest != receipt["packages"]:
        raise R13Error("package manifest receipt mismatch")
    package_items = [manifest["before"], *manifest["after"]]
    actual_package_files = {path.name for path in (run_dir / "packages").glob("*.abipkg")}
    if actual_package_files != {str(item["path"]) for item in package_items}:
        raise R13Error("undeclared or missing package")
    before_path = run_dir / "packages" / str(manifest["before"]["path"])
    if (
        not before_path.is_file()
        or before_path.stat().st_size != manifest["before"]["bytes"]
        or sha256_file(before_path) != manifest["before"]["sha256"]
    ):
        raise R13Error("before package identity changed")
    before_package, _before_transition = load_package(before_path)
    if before_package["transition_sha256"] != manifest["before"]["transition_sha256"]:
        raise R13Error("before package transition changed")
    package_accuracy: dict[str, float] = {}
    for index, item in enumerate(manifest["after"]):
        path = run_dir / "packages" / str(item["path"])
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise R13Error("package identity changed")
        package, transition = load_package(path)
        if package["transition_sha256"] != item["transition_sha256"]:
            raise R13Error("package transition changed")
        capability_id = capability_ids[index]
        if item["capability_id"] != capability_id:
            raise R13Error("package capability order changed")
        source = receipt["source_acquisitions"][index]
        if (
            package["provenance"]["teacher_before_sha256"]
            != source["base_state_sha256_before"]
            or package["provenance"]["teacher_after_sha256"]
            != source["adapter_state_sha256"]
        ):
            raise R13Error("package provenance changed")
        package_accuracy[capability_id] = transition_accuracy(transition, evaluation[index])
    source_reference = receipt["source_observations"]
    source_rows = _jsonl(
        run_dir / source_reference["path"],
        source_reference["sha256"],
        source_reference["rows"],
    )
    expected_rows = {
        str(row["row_id"]): row
        for rows in [*evaluation, *atomic]
        for row in rows
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for row in source_rows:
        if set(row) != SOURCE_FIELDS:
            raise R13Error("source observation schema changed")
        key = (str(row["capability_id"]), str(row["condition"]), str(row["row_id"]))
        reference = expected_rows.get(key[2])
        if (
            key in seen
            or key[0] not in capability_ids
            or key[1]
            not in {"BEFORE_EVALUATION", "BEFORE_ATOMIC", "AFTER_EVALUATION", "AFTER_ATOMIC"}
            or reference is None
            or row["prompt_sha256"] != reference["prompt_sha256"]
            or int(row["answer"]) != int(reference["answer"])
        ):
            raise R13Error("source observation identity changed")
        seen.add(key)
        _probabilities(row["canonical_probabilities"])
        grouped[(key[0], key[1])].append(row)
    for capability_id in capability_ids:
        for condition, expected_count in (
            ("BEFORE_EVALUATION", len(evaluation[0])),
            ("AFTER_EVALUATION", len(evaluation[0])),
            ("BEFORE_ATOMIC", len(atomic[0])),
            ("AFTER_ATOMIC", len(atomic[0])),
        ):
            if len(grouped[(capability_id, condition)]) != expected_count:
                raise R13Error("source observation coverage changed")
    gates = config["gates"]
    for index, source in enumerate(receipt["source_acquisitions"]):
        capability_id = capability_ids[index]
        if source["capability_id"] != capability_id:
            raise R13Error("source capability order changed")
        target_ids = [int(value) for value in source["target_token_ids"]]
        if len(target_ids) != 8 or len(set(target_ids)) != 8:
            raise R13Error("source target token map changed")
        for condition, field in (
            ("BEFORE_EVALUATION", "before_evaluation"),
            ("BEFORE_ATOMIC", "before_atomic"),
            ("AFTER_EVALUATION", "after_evaluation"),
            ("AFTER_ATOMIC", "after_atomic"),
        ):
            actual = _source_metrics(grouped[(capability_id, condition)], target_ids)
            if actual != source[field]:
                raise R13Error(f"source metrics changed: {capability_id}/{condition}")
        adapter = run_dir / source["adapter_artifact"]["path"]
        if (
            not adapter.is_file()
            or adapter.stat().st_size != source["adapter_artifact"]["bytes"]
            or sha256_file(adapter) != source["adapter_artifact"]["sha256"]
        ):
            raise R13Error("source adapter unavailable")
        extraction = source["extraction"]["extractor"]
        _evidence(source["extraction"], f"source extraction {capability_id}")
        if extraction != extractor_spec():
            raise R13Error("extractor access specification changed")
        training_receipt = source["training"]
        if (
            training_receipt["selection_used_heldout_rows"] is not False
            or training_receipt["selected_step"] is None
            or training_receipt["checkpoints"][-1]["consecutive_exact_atomic_evaluations"]
            < int(config["source_acquisition"]["stable_atomic_evaluations"])
            or source["base_state_sha256_before"] != source["base_state_sha256_after"]
            or source["before_atomic"]["accuracy"]
            > float(gates["source_before_atomic_maximum"])
            or source["after_atomic"]["accuracy"]
            != float(gates["source_after_atomic_accuracy"])
            or source["after_atomic"]["accuracy"] - source["before_atomic"]["accuracy"]
            < float(gates["source_atomic_gain_minimum"])
            or package_accuracy[capability_id] != float(gates["package_oracle_accuracy"])
            or source["package_oracle_accuracy"] != package_accuracy[capability_id]
            or source["package_source_exact_agreement"]
            != source["after_evaluation"]["canonical_accuracy"]
        ):
            raise R13Error(f"source or package gate failed: {capability_id}")
    recipient_receipts = receipt["recipient_workers"]
    if [item["host"] for item in recipient_receipts] != list(config["recipient_hosts"]):
        raise R13Error("recipient inventory changed")
    if len({int(item["pid"]) for item in recipient_receipts}) != len(recipient_receipts):
        raise R13Error("recipient workers did not use distinct processes")
    for worker in recipient_receipts:
        _evidence(worker, f"recipient {worker.get('host')}")
        if (
            worker.get("config_sha256") != sha256_file(config_path)
            or worker.get("reveal_sha256") != sha256_file(reveal_path)
            or worker.get("manifest_sha256") != sha256_file(manifest_path)
            or worker.get("source_adapter_argument_present") is not False
            or worker.get("source_adapter_loaded") is not False
        ):
            raise R13Error("recipient worker custody changed")
        host_dir = run_dir / "recipients" / str(worker["host"])
        rows = _jsonl(
            host_dir / worker["observations"]["path"],
            worker["observations"]["sha256"],
            worker["observations"]["rows"],
        )
        host_receipt = worker["host_receipt"]
        token_ids = [int(value) for value in host_receipt["target_token_ids"]]
        expected = {str(row["row_id"]): row for values in evaluation for row in values}
        keys = set()
        expected_row_count = len(expected) * len(config["conditions"])
        if len(rows) != expected_row_count:
            raise R13Error("recipient observation coverage changed")
        for row in rows:
            key = (
                str(row["capability_id"]),
                str(row["condition"]),
                str(row["row_id"]),
            )
            reference = expected.get(key[2])
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
                key in keys
                or key[0] not in capability_ids
                or key[1] not in config["conditions"]
                or reference is None
                or row["prompt_sha256"] != reference["prompt_sha256"]
                or row["package_sha256"] != expected_package
                or row["backend_active"] is not active
                or row["codec_active"] is not active
                or canonical_prediction(int(row["prediction_token_id"]), token_ids)
                != row["canonical_prediction"]
            ):
                raise R13Error("recipient row identity changed")
            keys.add(key)
            _probabilities(row["canonical_probabilities"])
            if not isinstance(row["neural_state"], list) or len(row["neural_state"]) != 8:
                raise R13Error("recipient neural state changed")
            if any(not math.isfinite(float(value)) for value in row["neural_state"]):
                raise R13Error("recipient neural state is non-finite")
        expected_keys = {
            (capability_id, str(condition), row_id)
            for capability_id in capability_ids
            for condition in config["conditions"]
            for row_id, row in expected.items()
            if row["capability_id"] == capability_id
        }
        if keys != expected_keys:
            raise R13Error("recipient observation matrix changed")
        if worker["summary"] != _summarize(rows, evaluation):
            raise R13Error("recipient summary changed")
    if not _recipient_pass(config, recipient_receipts):
        raise R13Error("recipient gate failed")
    result = {
        "format": "abi-r13-strict-verification/1",
        "verdict": "PASS",
        "claim": "BOUNDED_ENUMERABLE_CAPABILITY_EXTRACTION",
        "capabilities": len(capability_ids),
        "package_oracle_accuracy": package_accuracy,
        "source_package_agreement": {
            item["capability_id"]: item["package_source_exact_agreement"]
            for item in receipt["source_acquisitions"]
        },
        "r11_freeze_evidence_sha256": r11["evidence_sha256"],
        "run_evidence_sha256": receipt["evidence_sha256"],
        "stored_status_booleans_consumed": 0,
        "live_replay_required_for_final_seal": True,
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
