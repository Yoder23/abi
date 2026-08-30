"""Fail-closed raw-evidence verifier for ABI R8.

Stored status, pass, and gate booleans are never scientific inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from safetensors.torch import load_file

from .capability_generator import (
    OpaqueCapability,
    canonical_json_bytes,
    generate_composition_rows,
    generate_rows,
    public_capabilities,
    worker_rows,
)
from .extract_capability import ExtractionError, load_package
from .native_host import sha256_file, tensor_state_sha256
from .recipient_worker import CONDITIONS
from .run_noninterference import unrelated_tasks


class R8VerificationError(RuntimeError):
    """Raised when required R8 evidence is stale, malformed, or ambiguous."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R8VerificationError(f"required JSON unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise R8VerificationError(f"required JSON object changed: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise R8VerificationError(f"required JSONL unavailable: {path}") from exc
    if not lines or any(not line.strip() for line in lines):
        raise R8VerificationError(f"required JSONL is empty or contains blanks: {path}")
    rows = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise R8VerificationError(f"invalid row {index + 1}: {path}") from exc
        if not isinstance(value, dict):
            raise R8VerificationError(f"non-object row {index + 1}: {path}")
        rows.append(value)
    if path.read_bytes() != b"".join(canonical_json_bytes(row) for row in rows):
        raise R8VerificationError(f"noncanonical raw evidence: {path}")
    return rows


def _evidence(value: Mapping[str, Any], label: str) -> None:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
        raise R8VerificationError(f"stale evidence hash: {label}")


def _accuracy(rows: Sequence[Mapping[str, Any]], targets: Mapping[str, int]) -> float:
    if not rows:
        raise R8VerificationError("cannot score an empty condition")
    return sum(int(row.get("prediction_token_id") == targets[str(row["row_id"])]) for row in rows) / len(rows)


def _paired_bootstrap(
    after: Sequence[Mapping[str, Any]],
    base: Sequence[Mapping[str, Any]],
    targets: Mapping[str, int],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    base_by_id = {str(row["row_id"]): row for row in base}
    values = []
    for row in after:
        row_id = str(row["row_id"])
        if row_id not in base_by_id:
            raise R8VerificationError("paired bootstrap row is missing from BASE")
        target = targets[row_id]
        values.append(
            int(row.get("prediction_token_id") == target)
            - int(base_by_id[row_id].get("prediction_token_id") == target)
        )
    if len(base_by_id) != len(values):
        raise R8VerificationError("paired bootstrap row sets differ")
    generator = random.Random(seed)
    means = []
    for _ in range(replicates):
        means.append(sum(values[generator.randrange(len(values))] for _ in values) / len(values))
    means.sort()
    lower = means[max(0, math.floor(0.025 * replicates))]
    upper = means[min(replicates - 1, math.ceil(0.975 * replicates) - 1)]
    return {
        "point": sum(values) / len(values),
        "lower_95": lower,
        "upper_95": upper,
    }


def _private_capabilities(campaign_root: Path) -> list[OpaqueCapability]:
    private = _json(campaign_root / "evaluator_private/capabilities.json")
    _evidence(private, "private-capabilities")
    capabilities = [
        OpaqueCapability(
            capability_id=str(row["capability_id"]),
            offsets=tuple(int(value) for value in row["offsets"]),
            seed_commitment=str(row["seed_commitment"]),
        )
        for row in private.get("capabilities", [])
    ]
    if not capabilities or len({row.capability_id for row in capabilities}) != len(capabilities):
        raise R8VerificationError("private held-out capability inventory changed")
    return capabilities


def _labels(
    campaign_root: Path,
    config: Mapping[str, Any],
    capabilities: Sequence[OpaqueCapability],
) -> dict[str, dict[str, int]]:
    values = {}
    source_dir = campaign_root / "heldout_source"
    for index, capability in enumerate(capabilities):
        path = campaign_root / "evaluator_private/evaluation" / f"{capability.capability_id}.jsonl"
        rows = _jsonl(path)
        expected = generate_rows(
            capability,
            split="heldout_evaluation",
            rows=int(config["splits"]["evaluation_rows_per_capability"]),
            depths=config["capability_family"]["evaluation_depths"],
            seed=int(config["training"]["seed"]) + 16001 * index,
        )
        if rows != expected:
            raise R8VerificationError(f"private evaluator rows changed: {capability.capability_id}")
        public_path = source_dir / "worker_inputs" / f"{capability.capability_id}.jsonl"
        if _jsonl(public_path) != worker_rows(expected):
            raise R8VerificationError(f"worker input differs from answer-free projection: {capability.capability_id}")
        row_index = {str(row["row_id"]): int(row["answer"]) for row in rows}
        if len(row_index) != len(rows):
            raise R8VerificationError(f"duplicate evaluator row: {capability.capability_id}")
        values[capability.capability_id] = row_index
    return values


def _token_targets(labels: Mapping[str, int], token_ids: Sequence[int]) -> dict[str, int]:
    if len(token_ids) != 8 or len(set(int(value) for value in token_ids)) != 8:
        raise R8VerificationError("canonical target-token map changed")
    return {row_id: int(token_ids[answer]) for row_id, answer in labels.items()}


def _group(
    rows: Iterable[Mapping[str, Any]], keys: Sequence[str]
) -> dict[tuple[str, ...], list[Mapping[str, Any]]]:
    result: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(field)) for field in keys)
        result.setdefault(key, []).append(row)
    return result


def _validate_probabilities(row: Mapping[str, Any], *, allow_absent: bool = False) -> None:
    values = row.get("canonical_output_probabilities")
    if allow_absent and values is None:
        return
    if (
        not isinstance(values, list)
        or len(values) != 8
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values)
        or abs(sum(float(value) for value in values) - 1.0) > 1e-4
    ):
        raise R8VerificationError("canonical probability row is malformed")


def _jensen_shannon(left: Sequence[float], right: Sequence[float]) -> float:
    epsilon = 1e-12
    midpoint = [(float(a) + float(b)) / 2.0 for a, b in zip(left, right)]

    def divergence(values: Sequence[float]) -> float:
        return sum(
            float(value) * math.log((float(value) + epsilon) / (middle + epsilon), 2)
            for value, middle in zip(values, midpoint)
            if float(value) > 0
        )

    return 0.5 * divergence(left) + 0.5 * divergence(right)


def verify(root: Path, config_path: Path, campaign_root: Path) -> dict[str, Any]:
    config = _json(config_path)
    freeze_path = campaign_root / "freeze_receipt.json"
    reveal_path = campaign_root / "heldout_reveal.json"
    freeze, reveal = _json(freeze_path), _json(reveal_path)
    _evidence(freeze, "freeze")
    _evidence(reveal, "reveal")
    if (
        freeze.get("config_sha256") != sha256_file(config_path)
        or reveal.get("freeze_receipt_sha256") != sha256_file(freeze_path)
        or int(reveal.get("created_unix_time_ns", 0)) <= int(freeze.get("created_unix_time_ns", 0))
    ):
        raise R8VerificationError("freeze/reveal temporal binding changed")
    frozen_inputs = freeze.get("inputs")
    if not isinstance(frozen_inputs, list) or len(frozen_inputs) != freeze.get("input_count"):
        raise R8VerificationError("freeze input inventory changed")
    if hashlib.sha256(canonical_json_bytes(frozen_inputs)).hexdigest() != freeze.get(
        "input_aggregate_sha256"
    ):
        raise R8VerificationError("freeze input aggregate changed")
    for row in frozen_inputs:
        relative = str(row.get("path", ""))
        path = (
            root / "experiments/native_transfer_r8" / relative.removeprefix("@code/")
            if relative.startswith("@code/")
            else campaign_root / relative
        )
        if (
            not path.is_file()
            or path.stat().st_size != int(row.get("bytes", -1))
            or sha256_file(path) != row.get("sha256")
        ):
            raise R8VerificationError(f"frozen input changed: {relative}")
    private_path = campaign_root / "evaluator_private/capabilities.json"
    if (
        reveal.get("private_capabilities_sha256") != sha256_file(private_path)
        or reveal.get("secret_sha256") != config["splits"]["heldout_secret_commitment_sha256"]
        or reveal.get("private_capability_rules_disclosed_to_worker") is not False
    ):
        raise R8VerificationError("held-out private/reveal binding changed")
    capabilities = _private_capabilities(campaign_root)
    capability_ids = [row.capability_id for row in capabilities]
    revealed_ids = [str(row["capability_id"]) for row in reveal.get("capabilities", [])]
    if revealed_ids != capability_ids:
        raise R8VerificationError("public and private held-out inventories differ")
    public_rules = {
        capability.offsets
        for capability in (
            *public_capabilities(
                int(config["splits"]["meta_seed"]),
                split="meta_train",
                count=int(config["splits"]["meta_train_capabilities"]),
            ),
            *public_capabilities(
                int(config["splits"]["development_seed"]),
                split="development",
                count=int(config["splits"]["development_capabilities"]),
            ),
        )
    }
    if any(capability.offsets in public_rules for capability in capabilities):
        raise R8VerificationError("held-out rule table overlaps a public capability")
    if len(capability_ids) != int(config["splits"]["heldout_capabilities"]) or len(set(capability_ids)) != len(capability_ids):
        raise R8VerificationError("held-out capability inventory changed")
    if len(capability_ids) < int(config["gates"]["heldout_capabilities_minimum"]):
        raise R8VerificationError("held-out capability depth is below the registered minimum")
    labels = _labels(campaign_root, config, capabilities)
    required_depth = int(config["splits"]["evaluation_rows_per_capability"])
    if required_depth < int(config["gates"]["evaluation_rows_per_capability_minimum"]):
        raise R8VerificationError("evaluation depth is below the registered minimum")
    if any(len(value) != required_depth for value in labels.values()):
        raise R8VerificationError("held-out evaluation depth changed")

    source_dir = campaign_root / "heldout_source"
    source_receipt = _json(source_dir / "receipt.json")
    _evidence(source_receipt, "heldout-source")
    if (
        source_receipt.get("config_sha256") != sha256_file(config_path)
        or source_receipt.get("freeze_receipt_sha256") != sha256_file(freeze_path)
        or source_receipt.get("reveal_receipt_sha256") != sha256_file(reveal_path)
        or source_receipt.get("source_model_parameters_trainable") != 0
        or source_receipt.get("recipient_optimizer_steps") != 0
    ):
        raise R8VerificationError("held-out source freeze receipt changed")
    source_raw_path = source_dir / source_receipt["source_observations"]["path"]
    if sha256_file(source_raw_path) != source_receipt["source_observations"]["sha256"]:
        raise R8VerificationError("source raw observation binding changed")
    source_rows = _jsonl(source_raw_path)
    source_expected = len(capability_ids) * required_depth * 3
    if len(source_rows) != source_expected:
        raise R8VerificationError("source raw observation depth changed")
    if len({(row.get("capability_id"), row.get("condition"), row.get("row_id")) for row in source_rows}) != len(source_rows):
        raise R8VerificationError("source observation keys are duplicated")
    for row in source_rows:
        _validate_probabilities(row)
    source_grouped = _group(source_rows, ("capability_id", "condition"))
    source_metrics = {}
    source_token_ids = source_receipt["source_observations"]["target_token_ids"]
    for capability_id in capability_ids:
        targets = _token_targets(labels[capability_id], source_token_ids)
        metrics = {}
        for condition in ("T_BEFORE", "T_AFTER", "T_PERMUTED_DELTA"):
            rows = source_grouped.get((capability_id, condition), [])
            if len(rows) != required_depth or len({str(row["row_id"]) for row in rows}) != required_depth:
                raise R8VerificationError(f"source condition depth changed: {capability_id}/{condition}")
            metrics[condition] = _accuracy(rows, targets)
        source_metrics[capability_id] = metrics

    package_receipts = {
        str(row["capability_id"]): row["packages"]
        for row in source_receipt.get("packages", [])
    }
    for capability_id in capability_ids:
        package_dir = source_dir / "packages" / capability_id
        if capability_id not in package_receipts:
            raise R8VerificationError(f"package receipt missing: {capability_id}")
        states = {}
        for name in ("before", "after", "permuted_teacher_delta"):
            path = package_dir / f"{name}.abipkg"
            receipt = package_receipts[capability_id][name]
            if (
                path.stat().st_size != int(receipt["bytes"])
                or sha256_file(path) != receipt["sha256"]
            ):
                raise R8VerificationError(f"sealed package changed: {capability_id}/{name}")
            try:
                document, _ = load_package(path)
            except ExtractionError as exc:
                raise R8VerificationError(f"package neutrality failed: {capability_id}/{name}") from exc
            states[name] = document
        if states["after"]["source_after_state_sha256"] == states["before"]["source_after_state_sha256"]:
            raise R8VerificationError(f"source-trained package has no learned state delta: {capability_id}")
        if states["after"]["latent_sha256"] in {
            states["before"]["latent_sha256"],
            states["permuted_teacher_delta"]["latent_sha256"],
        }:
            raise R8VerificationError(f"source-trained latent is not distinct: {capability_id}")

    gates_config = config["gates"]
    source_gate = all(
        values["T_BEFORE"] <= float(gates_config["source_before_accuracy_maximum"])
        and values["T_AFTER"] >= float(gates_config["source_after_accuracy_minimum"])
        and values["T_AFTER"] - values["T_BEFORE"] >= float(gates_config["source_gain_minimum"])
        and values["T_PERMUTED_DELTA"] <= float(
            gates_config["source_permuted_accuracy_maximum"]
        )
        for values in source_metrics.values()
    )

    recipients = {}
    package_identity: dict[str, set[str]] = {capability_id: set() for capability_id in capability_ids}
    recipient_gate = True
    specificity_gate = True
    host_families = set()
    missing = []
    for host_index, host in enumerate(sorted(config["models"]["recipients"])):
        base = campaign_root / "recipients" / host
        if not (base / "manifest.json").is_file():
            missing.append(f"recipient:{host}")
            recipient_gate = False
            specificity_gate = False
            continue
        manifest = _json(base / "manifest.json")
        _evidence(manifest, f"recipient:{host}")
        bridge_path = campaign_root / "pre_reveal/bridges" / host / "bridge.safetensors"
        expected_revision = str(config["models"]["recipients"][host]["revision"])
        if (
            manifest.get("config_sha256") != sha256_file(config_path)
            or manifest.get("freeze_receipt_sha256") != sha256_file(freeze_path)
            or manifest.get("revision") != expected_revision
            or manifest.get("bridge_sha256_before") != sha256_file(bridge_path)
            or manifest.get("bridge_sha256_after") != sha256_file(bridge_path)
            or manifest.get("host_model_state_sha256_before")
            != manifest.get("host_model_state_sha256_after")
            or tuple(manifest.get("conditions", ())) != CONDITIONS
        ):
            raise R8VerificationError(f"recipient identity/freeze binding changed: {host}")
        raw_path = base / "observations.jsonl"
        if manifest.get("observations_sha256") != sha256_file(raw_path):
            raise R8VerificationError(f"recipient raw binding changed: {host}")
        rows = _jsonl(raw_path)
        expected_rows = len(capability_ids) * required_depth * len(CONDITIONS)
        if len(rows) != expected_rows or manifest.get("rows") != expected_rows:
            raise R8VerificationError(f"recipient raw depth changed: {host}")
        if len({(row.get("capability_id"), row.get("condition"), row.get("row_id")) for row in rows}) != len(rows):
            raise R8VerificationError(f"recipient observation keys are duplicated: {host}")
        if (
            manifest.get("recipient_parameters_trainable") != 0
            or manifest.get("recipient_optimizer_steps") != 0
            or manifest.get("bridge_optimizer_steps_after_reveal") != 0
        ):
            raise R8VerificationError(f"recipient freeze/leakage receipt failed: {host}")
        absent_conditions = {"BRIDGE_REMOVED", "MODEL_REMOVED", "RUNTIME_ONLY"}
        for row in rows:
            condition = str(row.get("condition"))
            if condition in absent_conditions:
                if (
                    row.get("prediction_token_id") is not None
                    or row.get("canonical_output_probabilities") is not None
                    or not str(row.get("exception_type", "")).endswith(
                        "_INTENTIONALLY_ABSENT"
                    )
                ):
                    raise R8VerificationError(f"removed-component condition produced output: {host}")
            else:
                if row.get("prediction_token_id") is None or row.get("exception_type") is not None:
                    raise R8VerificationError(f"live recipient condition lacks logits: {host}/{condition}")
                _validate_probabilities(row)
        host_families.add(str(manifest["architecture_family"]))
        grouped = _group(rows, ("capability_id", "condition"))
        host_metrics = {}
        host_after = []
        host_base = []
        host_targets = {}
        for capability_id in capability_ids:
            targets = _token_targets(labels[capability_id], manifest["target_token_ids"])
            host_targets.update(targets)
            metrics = {}
            for condition in CONDITIONS:
                condition_rows = grouped.get((capability_id, condition), [])
                if len(condition_rows) != required_depth or len({str(row["row_id"]) for row in condition_rows}) != required_depth:
                    raise R8VerificationError(f"recipient condition depth changed: {host}/{capability_id}/{condition}")
                if set(str(row["row_id"]) for row in condition_rows) != set(targets):
                    raise R8VerificationError(f"recipient/evaluator row identity changed: {host}/{capability_id}/{condition}")
                metrics[condition] = _accuracy(condition_rows, targets)
            host_metrics[capability_id] = metrics
            host_after.extend(grouped[(capability_id, "AFTER")])
            host_base.extend(grouped[(capability_id, "BASE")])
            package_identity[capability_id].add(
                str(manifest["packages"][capability_id]["after"])
            )
            actual_hashes = {
                "before": sha256_file(source_dir / "packages" / capability_id / "before.abipkg"),
                "after": sha256_file(source_dir / "packages" / capability_id / "after.abipkg"),
                "permuted_teacher_delta": sha256_file(
                    source_dir / "packages" / capability_id / "permuted_teacher_delta.abipkg"
                ),
            }
            if manifest["packages"][capability_id] != actual_hashes:
                raise R8VerificationError(f"recipient package manifest changed: {host}/{capability_id}")
            wrong_id = capability_ids[(capability_ids.index(capability_id) + 1) % len(capability_ids)]
            expected_package_by_condition = {
                "BASE": None,
                "BEFORE": actual_hashes["before"],
                "AFTER": actual_hashes["after"],
                "PERMUTED_TEACHER_DELTA": actual_hashes["permuted_teacher_delta"],
                "ZERO": actual_hashes["after"],
                "RANDOM": actual_hashes["after"],
                "SHUFFLED": actual_hashes["after"],
                "WRONG": sha256_file(source_dir / "packages" / wrong_id / "after.abipkg"),
                "REMOVED": actual_hashes["after"],
                "BRIDGE_REMOVED": actual_hashes["after"],
                "MODEL_REMOVED": actual_hashes["after"],
                "RUNTIME_ONLY": actual_hashes["after"],
            }
            for condition, expected_hash in expected_package_by_condition.items():
                if any(
                    row.get("package_sha256") != expected_hash
                    for row in grouped[(capability_id, condition)]
                ):
                    raise R8VerificationError(
                        f"condition/package binding changed: {host}/{capability_id}/{condition}"
                    )
        bootstrap = _paired_bootstrap(
            host_after,
            host_base,
            host_targets,
            replicates=int(gates_config["bootstrap_replicates"]),
            seed=int(config["training"]["seed"]) + host_index,
        )
        gains = [values["AFTER"] - values["BASE"] for values in host_metrics.values()]
        host_transfer_gate = (
            min(gains) >= float(gates_config["recipient_gain_minimum"])
            and bootstrap["lower_95"] > 0
        )
        negative_conditions = (
            "BEFORE",
            "PERMUTED_TEACHER_DELTA",
            "ZERO",
            "RANDOM",
            "SHUFFLED",
            "WRONG",
            "REMOVED",
        )
        host_specificity_gate = all(
            values[condition] <= values["BASE"] + float(gates_config["negative_control_gain_maximum"])
            for values in host_metrics.values()
            for condition in negative_conditions
        )
        recipient_gate = recipient_gate and host_transfer_gate
        specificity_gate = specificity_gate and host_specificity_gate
        recipients[host] = {
            "architecture_family": manifest["architecture_family"],
            "per_capability": host_metrics,
            "paired_bootstrap_after_minus_base": bootstrap,
            "minimum_capability_gain": min(gains),
            "minimum_capability_gain_strict_pass": min(gains)
            >= float(gates_config["recipient_gain_strict"]),
            "transfer_gate": host_transfer_gate,
            "specificity_gate": host_specificity_gate,
            "physical_filesystem_isolation": bool(manifest.get("physical_filesystem_isolation")),
        }

    package_identity_gate = all(len(values) == 1 for values in package_identity.values()) and all(
        values for values in package_identity.values()
    )
    host_diversity_gate = len(host_families) >= int(gates_config["recipient_families_minimum"])
    freeze_gate = True
    for host in sorted(config["models"]["recipients"]):
        receipt_path = campaign_root / "pre_reveal/bridges" / host / "receipt.json"
        receipt = _json(receipt_path)
        _evidence(receipt, f"pre-reveal-bridge:{host}")
        bridge_path = receipt_path.parent / receipt["bridge"]["path"]
        freeze_gate = freeze_gate and (
            receipt.get("config_sha256") == sha256_file(config_path)
            and receipt.get("host_model_state_sha256_before")
            == receipt.get("host_model_state_sha256_after")
            and receipt.get("training", {}).get("heldout_capabilities") == 0
            and receipt.get("recipient_parameters_trainable") == 0
            and receipt.get("recipient_optimizer_steps") == 0
            and sha256_file(bridge_path) == receipt["bridge"]["sha256"]
            and freeze.get("bridges", {}).get(host, {}).get("bridge_sha256")
            == sha256_file(bridge_path)
        )
    physical_isolation_gate = False
    missing.append("physical-filesystem-isolation")

    causal_gate = False
    causal = {}
    for host in sorted(config["models"]["recipients"]):
        base = campaign_root / "causal" / host
        if not (base / "manifest.json").is_file():
            missing.append(f"causal:{host}")
            continue
        manifest = _json(base / "manifest.json")
        _evidence(manifest, f"causal:{host}")
        raw_path = base / "observations.jsonl"
        bridge_path = campaign_root / "pre_reveal/bridges" / host / "bridge.safetensors"
        if (
            manifest.get("observations_sha256") != sha256_file(raw_path)
            or manifest.get("config_sha256") != sha256_file(config_path)
            or manifest.get("bridge_sha256_before") != sha256_file(bridge_path)
            or manifest.get("bridge_sha256_after") != sha256_file(bridge_path)
            or manifest.get("host_model_state_sha256_before")
            != manifest.get("host_model_state_sha256_after")
            or manifest.get("recipient_parameters_trainable") != 0
            or manifest.get("recipient_optimizer_steps") != 0
            or manifest.get("bridge_optimizer_steps_after_reveal") != 0
        ):
            raise R8VerificationError(f"causal raw binding changed: {host}")
        rows = _jsonl(raw_path)
        layers = [int(value) for value in manifest.get("layers", [])]
        expected_per_capability = int(config["splits"]["causal_rows_per_capability"]) * (
            3 + 3 * len(layers)
        )
        if (
            not layers
            or len(rows) != len(capability_ids) * expected_per_capability
            or manifest.get("rows") != len(rows)
        ):
            raise R8VerificationError(f"causal row depth changed: {host}")
        for row in rows:
            _validate_probabilities(row)
        grouped = _group(rows, ("capability_id", "condition", "layer"))
        package_fractions = []
        best_patch_fractions = []
        best_rescue_fractions = []
        best_destruction_fractions = []
        per_capability = {}
        for capability_id in capability_ids:
            targets = _token_targets(labels[capability_id], manifest["target_token_ids"])
            count = int(config["splits"]["causal_rows_per_capability"])
            selected_targets = {
                str(row["row_id"]): targets[str(row["row_id"])]
                for row in _jsonl(
                    source_dir / "worker_inputs" / f"{capability_id}.jsonl"
                )[:count]
            }
            base_rows = grouped.get((capability_id, "BASE", "None"), [])
            real_rows = grouped.get((capability_id, "AFTER", "None"), [])
            ablated_rows = grouped.get(
                (capability_id, "PACKAGE_PATH_ABLATION", "None"), []
            )
            if any(len(value) != count for value in (base_rows, real_rows, ablated_rows)):
                raise R8VerificationError(f"causal primary rows missing: {host}/{capability_id}")
            if any(
                set(str(row["row_id"]) for row in values) != set(selected_targets)
                for values in (base_rows, real_rows, ablated_rows)
            ):
                raise R8VerificationError(f"causal row identities changed: {host}/{capability_id}")
            base_accuracy = _accuracy(base_rows, selected_targets)
            real_accuracy = _accuracy(real_rows, selected_targets)
            ablated_accuracy = _accuracy(ablated_rows, selected_targets)
            denominator = real_accuracy - base_accuracy
            package_fraction = (
                0.0
                if denominator <= 0
                else 1.0 - ((ablated_accuracy - base_accuracy) / denominator)
            )
            patch_fractions = []
            rescue_fractions = []
            destruction_fractions = []
            layer_metrics = {}
            for layer in layers:
                patched = grouped.get((capability_id, "CLEAN_STATE_PATCH", str(layer)), [])
                rescued = grouped.get(
                    (capability_id, "CAPABILITY_STATE_RESCUE", str(layer)), []
                )
                destroyed = grouped.get(
                    (capability_id, "DOWNSTREAM_NEURAL_DESTRUCTION", str(layer)), []
                )
                if any(len(value) != count for value in (patched, rescued, destroyed)):
                    raise R8VerificationError(f"causal patch rows missing: {host}/{capability_id}/{layer}")
                patched_accuracy = _accuracy(patched, selected_targets)
                rescued_accuracy = _accuracy(rescued, selected_targets)
                destroyed_accuracy = _accuracy(destroyed, selected_targets)
                patch_fraction = 0.0 if denominator <= 0 else 1.0 - (
                    (patched_accuracy - base_accuracy) / denominator
                )
                rescue_fraction = 0.0 if denominator <= 0 else (
                    rescued_accuracy - base_accuracy
                ) / denominator
                destruction_fraction = 0.0 if denominator <= 0 else 1.0 - (
                    (destroyed_accuracy - base_accuracy) / denominator
                )
                patch_fractions.append(patch_fraction)
                rescue_fractions.append(rescue_fraction)
                destruction_fractions.append(destruction_fraction)
                layer_metrics[str(layer)] = {
                    "clean_patch_causal_fraction": patch_fraction,
                    "rescue_fraction": rescue_fraction,
                    "downstream_destruction_causal_fraction": destruction_fraction,
                }
            package_fractions.append(package_fraction)
            best_patch_fractions.append(max(patch_fractions))
            best_rescue_fractions.append(max(rescue_fractions))
            best_destruction_fractions.append(max(destruction_fractions))
            per_capability[capability_id] = {
                "base_accuracy": base_accuracy,
                "after_accuracy": real_accuracy,
                "package_path_causal_fraction": package_fraction,
                "layers": layer_metrics,
            }
        causal[host] = {
            "observations": len(rows),
            "median_package_path_causal_fraction": statistics.median(package_fractions),
            "median_best_layer_clean_patch_causal_fraction": statistics.median(
                best_patch_fractions
            ),
            "median_best_layer_rescue_fraction": statistics.median(best_rescue_fractions),
            "median_best_layer_destruction_causal_fraction": statistics.median(
                best_destruction_fractions
            ),
            "per_capability": per_capability,
        }
    if len(causal) == int(gates_config["recipient_families_minimum"]):
        threshold = float(gates_config["causal_fraction_median_minimum"])
        rescue_threshold = float(gates_config["causal_rescue_fraction_median_minimum"])
        causal_gate = all(
            value["median_package_path_causal_fraction"] >= threshold
            and value["median_best_layer_clean_patch_causal_fraction"] >= threshold
            and value["median_best_layer_destruction_causal_fraction"] >= threshold
            and value["median_best_layer_rescue_fraction"] >= rescue_threshold
            for value in causal.values()
        )

    noninterference = {}
    noninterference_gate = True
    public_tasks = unrelated_tasks()
    task_answers = {str(row["task_id"]): int(row["answer"]) for row in public_tasks}
    for host in sorted(config["models"]["recipients"]):
        base = campaign_root / "noninterference" / host
        if not (base / "manifest.json").is_file():
            missing.append(f"noninterference:{host}")
            noninterference_gate = False
            continue
        manifest = _json(base / "manifest.json")
        _evidence(manifest, f"noninterference:{host}")
        raw_path = base / "observations.jsonl"
        if (
            manifest.get("observations_sha256") != sha256_file(raw_path)
            or manifest.get("task_inventory_sha256")
            != hashlib.sha256(canonical_json_bytes(public_tasks)).hexdigest()
            or int(manifest.get("task_rows", -1)) != len(public_tasks)
        ):
            raise R8VerificationError(f"non-interference evidence binding changed: {host}")
        rows = _jsonl(raw_path)
        expected = len(capability_ids) * len(public_tasks) * 2
        if len(rows) != expected or manifest.get("rows") != expected:
            raise R8VerificationError(f"non-interference row depth changed: {host}")
        for row in rows:
            _validate_probabilities(row)
        grouped = _group(rows, ("capability_id", "condition"))
        targets = _token_targets(task_answers, manifest["target_token_ids"])
        per_capability = {}
        for capability_id in capability_ids:
            base_rows = grouped.get((capability_id, "BASE"), [])
            after_rows = grouped.get((capability_id, "AFTER"), [])
            if any(len(value) != len(public_tasks) for value in (base_rows, after_rows)):
                raise R8VerificationError(
                    f"non-interference conditions missing: {host}/{capability_id}"
                )
            base_by_id = {str(row["task_id"]): row for row in base_rows}
            after_by_id = {str(row["task_id"]): row for row in after_rows}
            if set(base_by_id) != set(task_answers) or set(after_by_id) != set(task_answers):
                raise R8VerificationError(
                    f"non-interference task identity changed: {host}/{capability_id}"
                )
            base_accuracy = _accuracy(
                [{**row, "row_id": row["task_id"]} for row in base_rows], targets
            )
            after_accuracy = _accuracy(
                [{**row, "row_id": row["task_id"]} for row in after_rows], targets
            )
            divergences = [
                _jensen_shannon(
                    base_by_id[task_id]["canonical_output_probabilities"],
                    after_by_id[task_id]["canonical_output_probabilities"],
                )
                for task_id in sorted(task_answers)
            ]
            mean_jsd = sum(divergences) / len(divergences)
            passed = (
                after_accuracy
                >= base_accuracy
                - float(gates_config["noninterference_accuracy_drop_maximum"])
                and mean_jsd <= float(gates_config["noninterference_jsd_maximum"])
            )
            noninterference_gate = noninterference_gate and passed
            per_capability[capability_id] = {
                "base_accuracy": base_accuracy,
                "after_accuracy": after_accuracy,
                "accuracy_change": after_accuracy - base_accuracy,
                "mean_canonical_jsd_bits": mean_jsd,
                "gate": passed,
            }
        noninterference[host] = per_capability

    baseline_manifest_path = campaign_root / "baselines/manifest.json"
    baseline_raw_path = campaign_root / "baselines/observations.jsonl"
    if baseline_manifest_path.is_file() and baseline_raw_path.is_file():
        baseline_manifest = _json(baseline_manifest_path)
        _evidence(baseline_manifest, "baselines")
        if (
            baseline_manifest.get("observations_sha256") != sha256_file(baseline_raw_path)
            or baseline_manifest.get("config_sha256") != sha256_file(config_path)
            or baseline_manifest.get("methods") != config["required_baselines"]
        ):
            raise R8VerificationError("baseline raw binding changed")
        baseline_rows = _jsonl(baseline_raw_path)
        reconstructed_rows = []
        for host in sorted(config["models"]["recipients"]):
            shard_dir = campaign_root / "baselines/shards" / host
            shard_manifest_path = shard_dir / "manifest.json"
            shard_manifest = _json(shard_manifest_path)
            _evidence(shard_manifest, f"baseline-shard:{host}")
            shard_raw_path = shard_dir / "observations.jsonl"
            parameter_path = shard_dir / "baseline_parameters.safetensors"
            root_shard = baseline_manifest.get("shards", {}).get(host, {})
            if (
                shard_manifest.get("config_sha256") != sha256_file(config_path)
                or shard_manifest.get("revision")
                != config["models"]["recipients"][host]["revision"]
                or shard_manifest.get("methods") != config["required_baselines"]
                or shard_manifest.get("lora_base_weight_sha256_before")
                != shard_manifest.get("lora_base_weight_sha256_after")
                or shard_manifest.get("observations_sha256") != sha256_file(shard_raw_path)
                or shard_manifest.get("parameter_artifact_sha256")
                != sha256_file(parameter_path)
                or shard_manifest.get("parameter_state_sha256")
                != tensor_state_sha256(load_file(str(parameter_path), device="cpu"))
                or root_shard.get("manifest_sha256") != sha256_file(shard_manifest_path)
                or root_shard.get("observations_sha256") != sha256_file(shard_raw_path)
            ):
                raise R8VerificationError(f"baseline shard binding changed: {host}")
            reconstructed_rows.extend(_jsonl(shard_raw_path))
        if reconstructed_rows != baseline_rows:
            raise R8VerificationError("consolidated baseline rows differ from immutable shards")
        required_methods = [str(value) for value in config["required_baselines"]]
        expected = (
            len(config["models"]["recipients"])
            * len(capability_ids)
            * required_depth
            * len(required_methods)
        )
        if len(baseline_rows) != expected or baseline_manifest.get("rows") != expected:
            raise R8VerificationError("baseline raw depth changed")
        for row in baseline_rows:
            _validate_probabilities(row)
        grouped = _group(baseline_rows, ("host", "capability_id", "method"))
        baseline_scores = {}
        efficiencies = []
        baseline_gate = True
        for host in sorted(config["models"]["recipients"]):
            token_ids = baseline_manifest.get("target_token_ids", {}).get(host)
            if token_ids is None:
                raise R8VerificationError(f"baseline target tokens missing: {host}")
            host_scores = {}
            for capability_id in capability_ids:
                targets = _token_targets(labels[capability_id], token_ids)
                capability_scores = {}
                for method in required_methods:
                    method_rows = grouped.get((host, capability_id, method), [])
                    if (
                        len(method_rows) != required_depth
                        or {str(row["row_id"]) for row in method_rows} != set(targets)
                    ):
                        raise R8VerificationError(
                            f"baseline condition missing: {host}/{capability_id}/{method}"
                        )
                    capability_scores[method] = _accuracy(method_rows, targets)
                host_scores[capability_id] = capability_scores
                base_accuracy = recipients[host]["per_capability"][capability_id]["BASE"]
                abi_accuracy = recipients[host]["per_capability"][capability_id]["AFTER"]
                lora_accuracy = capability_scores["target_specific_lora"]
                denominator = lora_accuracy - base_accuracy
                efficiency = None if denominator <= 0 else (
                    abi_accuracy - base_accuracy
                ) / denominator
                if efficiency is not None:
                    efficiencies.append(efficiency)
            baseline_scores[host] = host_scores
        median_efficiency = statistics.median(efficiencies) if efficiencies else None
        transfer_efficiency_gate = (
            median_efficiency is not None
            and median_efficiency
            >= float(gates_config["transfer_efficiency_median_minimum"])
        )
        baseline = {
            "per_host": baseline_scores,
            "median_transfer_efficiency": median_efficiency,
            "strict_transfer_efficiency_pass": median_efficiency is not None
            and median_efficiency >= float(gates_config["transfer_efficiency_strict"]),
        }
    else:
        missing.append("baselines")
        baseline = {"per_host": {}}
        baseline_gate = False
        transfer_efficiency_gate = False

    composition = {}
    composition_gate = True
    composition_inputs = campaign_root / "composition/worker_inputs"
    pair_path = composition_inputs / "pair.json"
    cross_private_path = campaign_root / "evaluator_private/composition/cross.jsonl"
    if not pair_path.is_file() or not cross_private_path.is_file():
        missing.append("composition")
        composition_gate = False
    else:
        pair = _json(pair_path)
        first_id, second_id = (str(value) for value in pair["capability_ids"])
        if [first_id, second_id] != capability_ids[:2]:
            raise R8VerificationError("composition pair identity changed")
        first_targets_by_answer = labels[first_id]
        second_targets_by_answer = labels[second_id]
        cross_rows = _jsonl(cross_private_path)
        expected_cross = generate_composition_rows(
            capabilities[0],
            capabilities[1],
            split="heldout_composition",
            rows=required_depth,
            first_depths=config["capability_family"]["composition_evaluation_depths"],
            second_depths=config["capability_family"]["composition_evaluation_depths"],
            seed=int(config["training"]["seed"]) + 99173,
        )
        if cross_rows != expected_cross:
            raise R8VerificationError("private composition rows changed")
        cross_targets_by_answer = {
            str(row["row_id"]): int(row["answer"]) for row in cross_rows
        }
        for host in sorted(config["models"]["recipients"]):
            base = campaign_root / "composition" / host
            if not (base / "manifest.json").is_file():
                missing.append(f"composition:{host}")
                composition_gate = False
                continue
            manifest = _json(base / "manifest.json")
            _evidence(manifest, f"composition:{host}")
            raw_path = base / "observations.jsonl"
            if (
                manifest.get("observations_sha256") != sha256_file(raw_path)
                or manifest.get("pair_sha256") != sha256_file(pair_path)
                or manifest.get("host_model_state_sha256_before")
                != manifest.get("host_model_state_sha256_after")
                or manifest.get("bridge_sha256_before")
                != manifest.get("bridge_sha256_after")
            ):
                raise R8VerificationError(f"composition evidence binding changed: {host}")
            rows = _jsonl(raw_path)
            if len(rows) != 11 * required_depth or manifest.get("rows") != len(rows):
                raise R8VerificationError(f"composition row depth changed: {host}")
            for row in rows:
                _validate_probabilities(row)
            grouped = _group(rows, ("condition",))
            token_ids = manifest["target_token_ids"]
            first_targets = _token_targets(first_targets_by_answer, token_ids)
            second_targets = _token_targets(second_targets_by_answer, token_ids)
            cross_targets = _token_targets(cross_targets_by_answer, token_ids)
            conditions = {
                "BASE_FIRST": first_targets,
                "FIRST_ONLY": first_targets,
                "COMBINED_ON_FIRST": first_targets,
                "REMOVED_FIRST": first_targets,
                "BASE_SECOND": second_targets,
                "SECOND_ONLY": second_targets,
                "COMBINED_ON_SECOND": second_targets,
                "BASE_CROSS": cross_targets,
                "FIRST_ONLY_CROSS": cross_targets,
                "SECOND_ONLY_CROSS": cross_targets,
                "COMBINED_CROSS": cross_targets,
            }
            scores = {}
            for condition, targets in conditions.items():
                condition_rows = grouped.get((condition,), [])
                if (
                    len(condition_rows) != required_depth
                    or {str(row["row_id"]) for row in condition_rows} != set(targets)
                ):
                    raise R8VerificationError(f"composition condition missing: {host}/{condition}")
                scores[condition] = _accuracy(condition_rows, targets)
            drop = float(gates_config["composition_preservation_drop_maximum"])
            cross_gain = float(gates_config["composition_cross_gain_minimum"])
            host_gate = (
                scores["COMBINED_ON_FIRST"] >= scores["FIRST_ONLY"] - drop
                and scores["COMBINED_ON_SECOND"] >= scores["SECOND_ONLY"] - drop
                and abs(scores["REMOVED_FIRST"] - scores["BASE_FIRST"]) <= drop
                and scores["COMBINED_CROSS"]
                >= max(
                    scores["BASE_CROSS"],
                    scores["FIRST_ONLY_CROSS"],
                    scores["SECOND_ONLY_CROSS"],
                )
                + cross_gain
            )
            composition_gate = composition_gate and host_gate
            composition[host] = {"scores": scores, "gate": host_gate}

    # These higher-level gates intentionally remain false until dedicated raw
    # evidence validators are implemented and exercised. File existence alone
    # is never a scientific input.
    new_recipient_gate = False
    external_gate = False
    blind_gate = False
    second_family_gate = False
    for label, path in (
        ("new-recipient", campaign_root / "new_recipient"),
        ("independent-hardware", campaign_root / "external"),
        ("blind-review", campaign_root / "blind_review"),
        ("second-capability-family", campaign_root / "second_family"),
    ):
        if not path.is_dir():
            missing.append(label)

    gates = {
        "source_acquisition": source_gate,
        "native_transfer": recipient_gate and host_diversity_gate,
        "specificity": specificity_gate,
        "neural_causality": causal_gate,
        "noninterference": noninterference_gate,
        "package_identity": package_identity_gate,
        "freeze": freeze_gate,
        "physical_isolation": physical_isolation_gate,
        "matched_baselines": baseline_gate,
        "transfer_efficiency": transfer_efficiency_gate,
        "composition_reversibility": composition_gate,
        "new_after_package_recipient": new_recipient_gate,
        "independent_hardware": external_gate,
        "blind_hostile_review": blind_gate,
        "second_capability_family": second_family_gate,
    }
    level = 0
    if any(
        value.get("paired_bootstrap_after_minus_base", {}).get("lower_95", 0.0) > 0
        and value.get("minimum_capability_gain", -1.0) > 0
        for value in recipients.values()
    ):
        level = 1
    if all(
        gates[key]
        for key in (
            "source_acquisition",
            "native_transfer",
            "specificity",
            "neural_causality",
            "noninterference",
            "package_identity",
            "freeze",
            "physical_isolation",
        )
    ):
        level = 2
    if level >= 2 and gates["composition_reversibility"] and gates["new_after_package_recipient"]:
        level = 3
    if (
        level >= 3
        and gates["matched_baselines"]
        and gates["transfer_efficiency"]
        and gates["independent_hardware"]
        and gates["blind_hostile_review"]
        and gates["second_capability_family"]
    ):
        level = 4
    primary_execution_complete = (
        len(recipients) == int(gates_config["recipient_families_minimum"])
        and len(causal) == int(gates_config["recipient_families_minimum"])
        and len(noninterference) == int(gates_config["recipient_families_minimum"])
    )
    exact_answer = (
        "YES"
        if level >= 2
        else "NO"
        if primary_execution_complete and level == 0
        else "NOT YET ESTABLISHED"
    )
    result = {
        "format": "abi-native-transfer-r8-verification/1",
        "status": "PASS_R8_LEVEL_4_BREAKTHROUGH_CANDIDATE" if level == 4 else "FAIL_CLOSED_R8_NOT_FULLY_ESTABLISHED",
        "exact_question_answer": exact_answer,
        "verdict_level": level,
        "source": source_metrics,
        "recipients": recipients,
        "causal": causal,
        "noninterference": noninterference,
        "baseline": baseline,
        "composition": composition,
        "package_identity_sha256_sets": {key: sorted(value) for key, value in package_identity.items()},
        "gates": gates,
        "missing_required_evidence": sorted(set(missing)),
        "trusted_scientific_booleans_consumed": 0,
        "claim_boundary": "R8 only. R7 is unchanged. No teacher extraction, native transfer, or breakthrough claim may exceed this recomputed verdict level.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        value = verify(
            Path(args.root).resolve(),
            Path(args.config).resolve(),
            Path(args.campaign_root).resolve(),
        )
        if args.output:
            path = Path(args.output).resolve()
            if path.exists():
                raise R8VerificationError(f"immutable verifier output exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    except R8VerificationError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["verdict_level"] == 4 else 2


if __name__ == "__main__":
    raise SystemExit(main())
