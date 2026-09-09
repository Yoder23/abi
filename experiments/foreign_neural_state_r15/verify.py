"""Fail-closed recomputation of the R15A neural-state recovery claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
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
    source_metrics,
    write_json_once,
)
from experiments.foreign_capability_r14.extractor import extract_transition
from experiments.foreign_teacher_r12.custody import verify_r11_freeze
from experiments.native_isa_r11.core import (
    RecurrentTransitionNeuralISA,
    load_package,
    transition_accuracy,
    transition_bytes,
)
from experiments.native_isa_r11.run import _conditions

from .frontend import (
    frontend_spec,
    labels_for_capability,
    load_frozen_frontend,
    predict_labels,
    train_affine_table_frontend,
    transition_from_labels,
)
from .protocol import capability_rows, heldout_capabilities, operations_commitment
from .public_preflight import _public_capabilities
from .recipient_worker import summarize
from .run import _delta_controls

SOURCE_FIELDS = {
    "capability_id",
    "condition",
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
        raise R14Error(f"raw R15A evidence unavailable: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R14Error(f"raw R15A evidence unreadable: {path}") from exc
    if len(rows) != expected_rows or not all(isinstance(row, dict) for row in rows):
        raise R14Error(f"raw R15A row count or schema changed: {path}")
    return rows


def _probabilities(values: Any, *, normalized: bool = True) -> list[float]:
    if not isinstance(values, list) or len(values) != 8:
        raise R14Error("R15A probability width changed")
    numbers = [float(item) for item in values]
    if any(not math.isfinite(item) for item in numbers) or (
        normalized
        and (
            any(item < -1e-6 or item > 1.000001 for item in numbers)
            or abs(sum(numbers) - 1.0) > 2e-5
        )
    ):
        raise R14Error("R15A probabilities invalid")
    return numbers


def _close(left: list[float], right: list[float]) -> bool:
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= 2e-6 for a, b in zip(left, right)
    )


def _adapter_state_sha256(state: dict[str, torch.Tensor]) -> str:
    if set(state) != {"a", "b"}:
        raise R14Error("R15A adapter state inventory changed")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise R14Error("R15A adapter state contains non-finite values")
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _effective_delta_sha256(delta: torch.Tensor) -> str:
    value = delta.detach().cpu().float().reshape(8, -1).contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(str(list(value.shape)).encode("ascii") + b"\0")
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def verify_public_frontend(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    bound = config["public_frontend"]
    receipt_path = root / str(bound["receipt"])
    frontend_path = root / str(bound["tensors"])
    dataset_path = root / str(bound["dataset"])
    if (
        not receipt_path.is_file()
        or not frontend_path.is_file()
        or not dataset_path.is_file()
        or sha256_file(receipt_path) != bound["receipt_sha256"]
        or sha256_file(frontend_path) != bound["tensors_sha256"]
        or sha256_file(dataset_path) != bound["dataset_sha256"]
    ):
        raise R14Error("bound R15A public frontend evidence unavailable")
    receipt = json_object(receipt_path)
    _verify_evidence(receipt, "R15A public receipt")
    if receipt["evidence_sha256"] != bound["evidence_sha256"]:
        raise R14Error("bound R15A public evidence identity changed")
    dataset = load_file(str(dataset_path), device="cpu")
    with safe_open(str(dataset_path), framework="pt", device="cpu") as handle:
        dataset_metadata = handle.metadata()
    if set(dataset) != {"features"} or dataset_metadata != {
        "format": "abi-r15a-public-weight-delta-dataset/1",
        "receipt_sha256": bound["receipt_sha256"],
        "receipt_evidence_sha256": bound["evidence_sha256"],
    }:
        raise R14Error("R15A public dataset contract changed")
    features_tensor = dataset["features"].float().contiguous()
    if tuple(features_tensor.shape) != (320, 7168):
        raise R14Error("R15A public dataset shape changed")
    capabilities = _public_capabilities(
        320, seed=int(bound["public_seed"]), split="r15_public"
    )
    features = []
    labels = []
    source_events = receipt["source"]["events"]
    if len(source_events) != 320:
        raise R14Error("R15A public source event count changed")
    for index, capability in enumerate(capabilities):
        event = source_events[index]
        feature = features_tensor[index]
        if (
            feature.numel() != 7168
            or event.get("capability_id") != capability.capability_id
            or event.get("operations_commitment") != operations_commitment(capability)
            or event.get("acquisition", {}).get("after_delta_sha256")
            != _effective_delta_sha256(feature)
        ):
            raise R14Error("R15A public event binding changed")
        features.append(feature)
        labels.append(labels_for_capability(capability))
    x = torch.stack(features)
    y = torch.stack(labels)
    retrained = train_affine_table_frontend(x[:256], y[:256])
    frozen = load_frozen_frontend(frontend_path, receipt["frontend"])
    if (
        frontend_spec(retrained) != frontend_spec(frozen)
        or not torch.equal(retrained["readout"], frozen["readout"])
        or not torch.equal(retrained["bias"], frozen["bias"])
    ):
        raise R14Error("R15A public frontend is not exactly reproducible")
    predictions = torch.stack([predict_labels(frozen, feature) for feature in x[256:]])
    exact = int(predictions.eq(y[256:]).all(dim=1).sum())
    heads = int(predictions.eq(y[256:]).sum())
    if exact != 64 or heads != 384:
        raise R14Error("R15A public frontend qualification does not recompute")
    return {
        "events_recomputed": 320,
        "development_exact": exact,
        "development_heads_exact": heads,
        "frontend_spec_sha256": receipt["frontend"]["evidence_sha256"],
        "receipt_sha256": sha256_file(receipt_path),
        "tensors_sha256": sha256_file(frontend_path),
        "dataset_sha256": sha256_file(dataset_path),
    }


def _group_source_rows(
    raw: list[dict[str, Any]], capabilities: list[Any]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    capability_ids = {item.capability_id for item in capabilities}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        if set(row) != SOURCE_FIELDS or row["capability_id"] not in capability_ids:
            raise R14Error("R15A source observation schema or identity changed")
        _probabilities(row["canonical_probabilities"])
        grouped[(str(row["capability_id"]), str(row["condition"]))].append(
            {key: row[key] for key in SOURCE_FIELDS - {"capability_id", "condition"}}
        )
    return grouped


def _verify_rows(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]], label: str
) -> None:
    reference = {str(row["row_id"]): row for row in expected}
    if len(actual) != len(expected) or len(reference) != len(expected):
        raise R14Error(f"R15A source coverage changed: {label}")
    seen = set()
    for row in actual:
        row_id = str(row["row_id"])
        target = reference.get(row_id)
        if (
            row_id in seen
            or target is None
            or row["prompt_sha256"] != target["prompt_sha256"]
            or int(row["start"]) != int(target["start"])
            or [int(item) for item in row["program"]]
            != [int(item) for item in target["program"]]
        ):
            raise R14Error(f"R15A source row identity changed: {label}")
        seen.add(row_id)


def _verify_isolation(
    run_dir: Path, item: dict[str, Any], expected_labels: list[int]
) -> dict[str, Any]:
    result_path = run_dir / item["isolated_result"]["path"]
    launcher_path = run_dir / item["isolated_launcher"]["path"]
    if (
        not result_path.is_file()
        or not launcher_path.is_file()
        or sha256_file(result_path) != item["isolated_result"]["sha256"]
        or sha256_file(launcher_path) != item["isolated_launcher"]["sha256"]
    ):
        raise R14Error("R15A isolated extraction evidence unavailable")
    result = json_object(result_path)
    launcher = json_object(launcher_path)
    _verify_evidence(result, "R15A isolated extraction")
    _verify_evidence(launcher, "R15A isolated launcher")
    mount_path = result_path.parent / "mountinfo.txt"
    mount_text = mount_path.read_text(encoding="utf-8")
    if (
        result.get("format") != "abi-r15a-isolated-weight-delta-extraction/1"
        or result.get("labels") != expected_labels
        or result.get("behavioral_queries") != 0
        or result.get("answers_consumed") != 0
        or result.get("oracle_calls") != 0
        or result.get("candidate_program_search") is not False
        or result.get("old_root_present") is not False
        or result.get("windows_mount_present") is not False
        or result.get("capability_reveal_files_present") != 0
        or result.get("operation_tables_present") != 0
        or len(result.get("operator_margins", [])) != 3
        or any(float(value) <= 0 for value in result.get("operator_margins", []))
        or launcher.get("sandbox_policy") != "linux-pivot-root-no-network/1"
        or launcher.get("worker_exit_code") != 0
        or sha256_file(mount_path) != launcher.get("mountinfo_sha256")
        or hashlib.sha256(mount_text.encode()).hexdigest() != result.get("mountinfo_sha256")
        or "/mnt/c " in mount_text
        or "/oldroot " in mount_text
    ):
        raise R14Error("R15A physical isolation gate changed")
    return result


def _verify_recipient(
    config: dict[str, Any],
    reveal_path: Path,
    run_dir: Path,
    manifest: dict[str, Any],
    capabilities: list[Any],
    rows_by_capability: list[dict[str, list[dict[str, Any]]]],
    worker: dict[str, Any],
) -> dict[str, Any]:
    _verify_evidence(worker, f"R15A recipient {worker.get('host')}")
    host_dir = run_dir / "recipients" / str(worker["host"])
    rows = _jsonl(
        host_dir / worker["observations"]["path"],
        worker["observations"]["sha256"],
        worker["observations"]["rows"],
    )
    if (
        worker.get("format") != "abi-r15a-recipient-worker/1"
        or worker.get("config_sha256") != config["config_sha256"]
        or worker.get("reveal_sha256") != sha256_file(reveal_path)
        or worker.get("manifest_sha256") != sha256_file(run_dir / "package_manifest.json")
        or worker.get("source_adapter_argument_present") is not False
        or worker.get("source_adapter_loaded") is not False
    ):
        raise R14Error("R15A recipient custody changed")
    expected = {
        str(row["row_id"]): row
        for item in rows_by_capability
        for row in item["recipient"]
    }
    capability_ids = [item.capability_id for item in capabilities]
    package_dir = run_dir / "packages"
    _, before_transition = load_package(
        package_dir / str(manifest["before"]["path"])
    )
    after_transitions = [
        load_package(package_dir / str(item["path"]))[1]
        for item in manifest["after"]
    ]
    condition_transitions = {
        capability_id: _conditions(
            before_transition,
            after_transitions[index],
            after_transitions[(index + 1) % len(after_transitions)],
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
            raise R14Error("R15A recipient row identity changed")
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
        }.get(key[1])
        active = key[1] not in {"BASE", "REMOVED", "BACKEND_REMOVED", "CODEC_REMOVED"}
        reference = expected[key[2]]
        canonical_probabilities = _probabilities(row["canonical_probabilities"])
        neural_state = _probabilities(
            row["neural_state"], normalized=key[1] != "ZERO"
        )
        transition = condition_transitions[key[0]].get(key[1])
        if active:
            if transition is None:
                raise R14Error("R15A active recipient transition missing")
            recomputed_state = executor(transition, [str(reference["prompt"])])[0].tolist()
        else:
            recomputed_state = canonical_probabilities
        if (
            row["prompt_sha256"] != reference["prompt_sha256"]
            or row["package_sha256"] != package_sha
            or row["backend_active"] is not active
            or row["codec_active"] is not active
            or canonical_prediction(int(row["prediction_token_id"]), token_ids)
            != row["canonical_prediction"]
            or not _close(neural_state, recomputed_state)
        ):
            raise R14Error("R15A recipient row content changed")
    expected_keys = {
        (capability.capability_id, condition, str(row["row_id"]))
        for capability, item in zip(capabilities, rows_by_capability)
        for condition in config["conditions"]
        for row in item["recipient"]
    }
    summary = summarize(rows, [item["recipient"] for item in rows_by_capability])
    host = worker["host_receipt"]
    maximum = float(config["gates"]["negative_control_accuracy_maximum"])
    if (
        seen != expected_keys
        or summary != worker["summary"]
        or host["recipient_optimizer_steps"] != 0
        or host["model_state_sha256_before"] != host["model_state_sha256_after"]
        or host["codec_sha256_before"] != host["codec_sha256_after"]
        or summary["removal_conditions_equal_base"] is not True
    ):
        raise R14Error("R15A recipient invariant changed")
    for capability_id in capability_ids:
        if (
            summary["accuracy"][f"{capability_id}/AFTER"] != 1.0
            or summary["accuracy"][f"{capability_id}/RESTORED"] != 1.0
            or any(
                summary["accuracy"][f"{capability_id}/{condition}"] > maximum
                for condition in config["negative_conditions"]
            )
        ):
            raise R14Error("R15A recipient accuracy gate failed")
    return summary


def verify(config_path: Path, reveal_path: Path, run_dir: Path) -> dict[str, Any]:
    config = json_object(config_path)
    receipt = json_object(run_dir / "receipt.json")
    _verify_evidence(receipt, "R15A run receipt")
    config_with_sha = {**config, "config_sha256": sha256_file(config_path)}
    if (
        config.get("status") != "PREREGISTERED_BEFORE_HELDOUT_REVEAL"
        or receipt.get("format") != "abi-r15a-foreign-weight-delta-extraction/1"
        or receipt.get("verification_status") != "UNVERIFIED_RUN_OUTPUT"
        or receipt.get("config_sha256") != config_with_sha["config_sha256"]
        or receipt.get("reveal_sha256") != sha256_file(reveal_path)
    ):
        raise R14Error("R15A run identity changed")
    reveal = json_object(reveal_path)
    try:
        secret = bytes.fromhex(str(reveal["secret_hex"]))
    except ValueError as exc:
        raise R14Error("R15A reveal malformed") from exc
    if (
        len(secret) != 32
        or hashlib.sha256(secret).hexdigest() != config["heldout_seed_commitment"]
        or reveal.get("commitment") != config["heldout_seed_commitment"]
    ):
        raise R14Error("R15A reveal commitment changed")
    preserved_reveal = run_dir / receipt["heldout"]["reveal_path"]
    if not preserved_reveal.is_file() or preserved_reveal.read_bytes() != reveal_path.read_bytes():
        raise R14Error("R15A preserved reveal changed")
    root = config_path.parents[3]
    r11 = verify_r11_freeze(root, config)
    if receipt.get("r11_freeze") != r11:
        raise R14Error("R15A R11 freeze changed")
    public = verify_public_frontend(root, config)
    capabilities = heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    rows_by_capability = capability_rows(config, capabilities)
    capability_ids = [item.capability_id for item in capabilities]
    if receipt["data"]["capabilities"] != capability_ids:
        raise R14Error("R15A held-out capability inventory changed")
    raw_reference = receipt["source_observations"]
    raw = _jsonl(
        run_dir / raw_reference["path"], raw_reference["sha256"], raw_reference["rows"]
    )
    grouped = _group_source_rows(raw, capabilities)
    deltas = []
    recomputed_sources = []
    source_conditions = {
        "BEFORE": "atomic",
        "AFTER": "atomic",
        "REMOVED": "atomic",
        "RESTORED": "atomic",
        "WRONG": "atomic",
        "RANDOM": "atomic",
        "SHUFFLED": "atomic",
        "QUERY": "queries",
        "SOURCE_EVALUATION": "source_evaluation",
        "SOURCE_COUNTERFACTUAL": "counterfactual",
    }
    source_receipts = receipt["source_acquisitions"]
    if len(source_receipts) != len(capabilities):
        raise R14Error("R15A source acquisition count changed")
    for index, (capability, rows, source) in enumerate(
        zip(capabilities, rows_by_capability, source_receipts)
    ):
        if (
            source["capability_id"] != capability.capability_id
            or source["operations_commitment"] != operations_commitment(capability)
            or source["model_id"] != "Qwen/Qwen2.5-0.5B"
            or source["revision"] != "060db6499f32faf8b98477b0a26969ef7d8b9987"
            or source["base_model_sha256_before"] != source["base_model_sha256_after"]
        ):
            raise R14Error("R15A source identity changed")
        for condition, split in source_conditions.items():
            _verify_rows(
                grouped[(capability.capability_id, condition)],
                rows[split],
                f"{capability.capability_id}/{condition}",
            )
        state_path = run_dir / source["adapter_artifact"]["path"]
        delta_path = run_dir / source["delta_artifact"]["path"]
        if (
            not state_path.is_file()
            or not delta_path.is_file()
            or state_path.stat().st_size != source["adapter_artifact"]["bytes"]
            or delta_path.stat().st_size != source["delta_artifact"]["bytes"]
            or sha256_file(state_path) != source["adapter_artifact"]["sha256"]
            or sha256_file(delta_path) != source["delta_artifact"]["sha256"]
        ):
            raise R14Error("R15A source neural-state artifact unavailable")
        state = load_file(str(state_path), device="cpu")
        delta_state = load_file(str(delta_path), device="cpu")
        if (
            _adapter_state_sha256(state) != source["adapter_state_sha256"]
            or set(delta_state) != {"delta"}
        ):
            raise R14Error("R15A source neural-state identity changed")
        delta = delta_state["delta"].float().contiguous()
        derived = torch.matmul(state["b"].t(), state["a"].t()).float().flatten() / int(
            config["source_acquisition"]["lora_rank"]
        )
        if (
            not torch.allclose(
                delta,
                derived,
                atol=float(config["gates"]["effective_delta_absolute_tolerance"]),
                rtol=float(config["gates"]["effective_delta_relative_tolerance"]),
            )
            or _effective_delta_sha256(delta) != source["delta_artifact"]["effective_delta_sha256"]
            or delta.numel() != source["delta_artifact"]["elements"]
        ):
            raise R14Error("R15A effective before/after weight delta changed")
        metrics = {
            "before_atomic": source_metrics(
                grouped[(capability.capability_id, "BEFORE")], rows["atomic"]
            ),
            "after_atomic": source_metrics(
                grouped[(capability.capability_id, "AFTER")], rows["atomic"]
            ),
            "source_evaluation": source_metrics(
                grouped[(capability.capability_id, "SOURCE_EVALUATION")],
                rows["source_evaluation"],
            ),
            "source_counterfactual": source_metrics(
                grouped[(capability.capability_id, "SOURCE_COUNTERFACTUAL")],
                rows["counterfactual"],
            ),
        }
        control_metrics = {
            condition: source_metrics(
                grouped[(capability.capability_id, condition)], rows["atomic"]
            )
            for condition in ("WRONG", "RANDOM", "SHUFFLED")
        }
        if (
            any(metrics[key] != source[key] for key in metrics)
            or control_metrics != source["state_controls"]
            or grouped[(capability.capability_id, "BEFORE")]
            != grouped[(capability.capability_id, "REMOVED")]
            or grouped[(capability.capability_id, "AFTER")]
            != grouped[(capability.capability_id, "RESTORED")]
            or metrics["after_atomic"]["accuracy"] != 1.0
            or source["training"]["source_atomic_accuracy"] != 1.0
            or source["training"]["steps"]
            != int(config["source_acquisition"]["steps"])
            or source["training"]["learning_rate"]
            != float(config["source_acquisition"]["learning_rate"])
            or source["training"]["batch_size"] != 24
            or source["training"]["cache_available_to_extractor"] is not False
        ):
            raise R14Error("R15A source learning or intervention gate failed")
        deltas.append(delta)
        recomputed_sources.append(metrics)
    frontend_receipt = json_object(root / config["public_frontend"]["receipt"])
    frontend = load_frozen_frontend(
        root / config["public_frontend"]["tensors"], frontend_receipt["frontend"]
    )
    manifest_path = run_dir / "package_manifest.json"
    if sha256_file(manifest_path) != receipt["manifest_sha256"]:
        raise R14Error("R15A package manifest changed")
    manifest = json_object(manifest_path)
    if manifest != receipt["packages"]:
        raise R14Error("R15A package manifest receipt mismatch")
    package_dir = run_dir / "packages"
    declared = [manifest["before"], *manifest["after"]]
    if {path.name for path in package_dir.glob("*.abipkg")} != {
        item["path"] for item in declared
    }:
        raise R14Error("R15A package inventory changed")
    extraction_items = receipt["neural_state_extractions"]
    black_box_items = receipt["black_box_baseline"]
    if len(extraction_items) != len(capabilities) or len(black_box_items) != len(capabilities):
        raise R14Error("R15A extraction evidence count changed")
    package_accuracy = {}
    black_box_exact = 0
    for index, (capability, rows, source, item, baseline) in enumerate(
        zip(capabilities, rows_by_capability, source_receipts, extraction_items, black_box_items)
    ):
        predicted = predict_labels(frontend, deltas[index]).tolist()
        expected = labels_for_capability(capability).tolist()
        isolated = _verify_isolation(run_dir, item, predicted)
        if (
            predicted != expected
            or item["capability_id"] != capability.capability_id
            or item["labels"] != predicted
            or isolated["weight_delta_sha256"] != source["delta_artifact"]["sha256"]
            or isolated["frontend_spec_sha256"] != public["frontend_spec_sha256"]
        ):
            raise R14Error("R15A neural-state extraction is not exact")
        transition = transition_from_labels(predicted)
        package_item = manifest["after"][index]
        package_path = package_dir / package_item["path"]
        package, package_transition = load_package(package_path)
        unseen = transition_accuracy(package_transition, rows["evaluation"])
        counterfactual = transition_accuracy(package_transition, rows["counterfactual"])
        if (
            package_item["capability_id"] != capability.capability_id
            or sha256_file(package_path) != package_item["sha256"]
            or transition_bytes(package_transition) != transition_bytes(transition)
            or package["provenance"]["teacher_before_sha256"]
            != source["base_model_sha256_before"]
            or package["provenance"]["teacher_after_sha256"]
            != source["adapter_state_sha256"]
            or unseen != item["package_unseen_accuracy"]
            or counterfactual != item["package_counterfactual_accuracy"]
            or unseen != 1.0
            or counterfactual != 1.0
        ):
            raise R14Error("R15A package identity or oracle behavior changed")
        package_accuracy[capability.capability_id] = {
            "unseen": unseen,
            "counterfactual": counterfactual,
        }
        queries = grouped[(capability.capability_id, "QUERY")]
        baseline_transition, extraction = extract_transition(queries)
        baseline_exact = (
            extraction["selected_operations_commitment"] == operations_commitment(capability)
        )
        if (
            baseline["capability_id"] != capability.capability_id
            or baseline["extraction"] != extraction
            or baseline["operations_exact"] is not baseline_exact
            or baseline["package_unseen_accuracy"]
            != transition_accuracy(baseline_transition, rows["evaluation"])
            or baseline["package_counterfactual_accuracy"]
            != transition_accuracy(baseline_transition, rows["counterfactual"])
        ):
            raise R14Error("R15A black-box baseline changed")
        black_box_exact += int(baseline_exact)
    controls = _delta_controls(frontend, deltas, capabilities, rows_by_capability)
    maximum = float(config["gates"]["negative_control_accuracy_maximum"])
    if controls != receipt["delta_controls"] or any(
        not item["zero_delta_rejected"]
        or any(
            item[condition][metric] > maximum
            for condition in ("WRONG", "RANDOM", "SHUFFLED")
            for metric in ("unseen_accuracy", "counterfactual_accuracy")
        )
        for item in controls
    ):
        raise R14Error("R15A delta intervention gate failed")
    recipient_workers = receipt["recipient_workers"]
    if (
        [item["host"] for item in recipient_workers] != config["recipient_hosts"]
        or len({int(item["pid"]) for item in recipient_workers}) != len(recipient_workers)
    ):
        raise R14Error("R15A recipient worker inventory changed")
    recipient_summaries = {
        worker["host"]: _verify_recipient(
            config_with_sha,
            reveal_path,
            run_dir,
            manifest,
            capabilities,
            rows_by_capability,
            worker,
        )
        for worker in recipient_workers
    }
    result = {
        "format": "abi-r15a-strict-verification/1",
        "verdict": "PASS",
        "claim": "BOUNDED_FOREIGN_NEURAL_STATE_CAPABILITY_RECOVERY",
        "capabilities": len(capabilities),
        "public_frontend": public,
        "package_oracle_accuracy": package_accuracy,
        "black_box_exact_capabilities": black_box_exact,
        "recipient_hosts": sorted(recipient_summaries),
        "source_atomic_exact_capabilities": sum(
            item["after_atomic"]["accuracy"] == 1.0 for item in recomputed_sources
        ),
        "extractor_behavioral_queries": 0,
        "extractor_answers": 0,
        "extractor_oracle_calls": 0,
        "r11_freeze_evidence_sha256": r11["evidence_sha256"],
        "run_evidence_sha256": receipt["evidence_sha256"],
        "stored_scientific_status_booleans_consumed": 0,
        "claim_ceiling": "NOT_TEACHER_BEHAVIOR_CLONING_NOT_PREEXISTING_KNOWLEDGE",
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
