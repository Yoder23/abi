"""Fail-closed recomputation of the R10 copy/paste result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .run import (
    R10RunError,
    _bind_inputs,
    _condition_latents,
    _evaluation_rows,
    _json,
    _resolve,
)
from .runtime import (
    CanonicalTransitionVM,
    CopyPasteRuntimeError,
    canonical_prediction,
    latent_bytes,
    load_package,
    sha256_file,
)


class R10VerificationError(RuntimeError):
    """Raised when R10 evidence is absent, stale, malformed, or fails a gate."""


SOURCE_FIELDS = {
    "capability_id",
    "condition",
    "row_id",
    "prompt_sha256",
    "prediction_token_id",
    "canonical_prediction",
    "canonical_output_probabilities",
}
RECIPIENT_FIELDS = {
    "host",
    "capability_id",
    "condition",
    "row_id",
    "prompt_sha256",
    "package_sha256",
    "interpreter_active",
    "prediction_token_id",
    "canonical_prediction",
    "canonical_output_utf8_hex",
    "canonical_output_probabilities",
    "vm_output_probabilities",
}


def _fail(message: str) -> None:
    raise R10VerificationError(message)


def _check_evidence_hash(value: Mapping[str, Any], *, label: str) -> None:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if not isinstance(stored, str) or stored != actual:
        _fail(f"{label} evidence hash changed")


def _jsonl(path: Path, *, expected_sha256: str, expected_rows: int) -> list[dict[str, Any]]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        _fail(f"raw evidence absent or changed: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    _fail(f"raw row is not an object: {path}")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R10VerificationError(f"raw evidence is unreadable: {path}") from exc
    if len(rows) != expected_rows:
        _fail(f"raw row count changed: {path}")
    return rows


def _probabilities(value: Any, *, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 8:
        _fail(f"{label} probability width changed")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in result):
        _fail(f"{label} probabilities invalid")
    if abs(sum(result) - 1.0) > 2e-5:
        _fail(f"{label} probabilities are not normalized")
    return result


def _close(left: Sequence[float], right: Sequence[float], tolerance: float = 2e-6) -> bool:
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right)
    )


def _metrics(rows: Sequence[Mapping[str, Any]], answers: Mapping[str, int]) -> dict[str, Any]:
    correct = sum(
        int(row.get("canonical_prediction") == answers.get(str(row.get("row_id")))) for row in rows
    )
    return {"correct": correct, "rows": len(rows), "accuracy": correct / len(rows)}


def _verify_packages(
    root: Path,
    run_dir: Path,
    config: Mapping[str, Any],
    receipt: Mapping[str, Any],
    capability_ids: Sequence[str],
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, str]]:
    package_manifest = receipt.get("packages")
    if not isinstance(package_manifest, dict):
        _fail("package manifest missing")
    before_item = package_manifest.get("before")
    after_items = package_manifest.get("after")
    if not isinstance(before_item, dict) or not isinstance(after_items, list):
        _fail("package manifest malformed")
    if [item.get("capability_id") for item in after_items] != list(capability_ids):
        _fail("package capability order changed")
    if len(after_items) != len(capability_ids):
        _fail("package count changed")

    expected_latents = load_file(
        str(_resolve(root, str(config["r8_reference"]["canonical_latents"]))),
        device="cpu",
    )
    items = [before_item, *after_items]
    expected = [expected_latents["before"], *list(expected_latents["development_after"])]
    loaded: list[torch.Tensor] = []
    hashes: dict[str, str] = {}
    declared_files: set[str] = set()
    for index, (item, expected_latent) in enumerate(zip(items, expected)):
        required = {"path", "sha256", "bytes", "latent_sha256"}
        if index > 0:
            required.add("capability_id")
        if set(item) != required:
            _fail("package manifest fields changed")
        path = run_dir / "packages" / str(item["path"])
        declared_files.add(path.name)
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or sha256_file(path) != item["sha256"]
        ):
            _fail(f"package identity changed: {path.name}")
        package, latent = load_package(path)
        if package["latent_sha256"] != item["latent_sha256"]:
            _fail(f"package latent hash changed: {path.name}")
        if latent_bytes(latent) != latent_bytes(expected_latent):
            _fail(f"package is not the registered R8 extracted latent: {path.name}")
        loaded.append(latent)
        if index > 0:
            hashes[capability_ids[index - 1]] = str(item["sha256"])
    actual_files = {path.name for path in (run_dir / "packages").glob("*.abipkg")}
    if actual_files != declared_files:
        _fail("undeclared or missing package file")
    return loaded[0], loaded[1:], hashes


def verify(config_path: Path, run_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config = _json(config_path)
    if config.get("status") != "PREREGISTERED_BEFORE_EXECUTION":
        _fail("preregistration is not frozen")
    receipt = _json(run_dir / "receipt.json")
    _check_evidence_hash(receipt, label="run receipt")
    if receipt.get("format") != "abi-copy-paste-r10-run/1":
        _fail("run receipt format changed")
    if receipt.get("config_sha256") != sha256_file(config_path):
        _fail("run does not bind the registered config")
    if receipt.get("claim_target") != "ABI-C4" or receipt.get("claim_ceiling") != (
        "RUNTIME_OWNED_COPY_PASTE_EXECUTION_ONLY"
    ):
        _fail("claim boundary changed")
    bindings = _bind_inputs(root, config)
    if receipt.get("bindings") != bindings:
        _fail("run dependency bindings changed")

    r8 = _json(_resolve(root, str(config["r8_reference"]["config"])))
    capabilities, generated_rows = _evaluation_rows(config, r8)
    capability_ids = [item.capability_id for item in capabilities]
    expected_rows = {str(row["row_id"]): row for rows in generated_rows for row in rows}
    if len(expected_rows) != sum(len(rows) for rows in generated_rows):
        _fail("registered evaluation rows are not globally unique")
    answers = {row_id: int(row["answer"]) for row_id, row in expected_rows.items()}
    before, after, after_hashes = _verify_packages(root, run_dir, config, receipt, capability_ids)

    source = receipt.get("source")
    source_execution = receipt.get("source_execution")
    recipient_execution = receipt.get("recipient_execution")
    if not all(
        isinstance(value, dict) for value in (source, source_execution, recipient_execution)
    ):
        _fail("execution receipt section missing")
    source_token_ids = [int(value) for value in source.get("target_token_ids", [])]
    if len(source_token_ids) != 8 or len(set(source_token_ids)) != 8:
        _fail("source token map missing or invalid")
    source_reference = source_execution.get("observations")
    recipient_reference = recipient_execution.get("observations")
    if not isinstance(source_reference, dict) or not isinstance(recipient_reference, dict):
        _fail("raw observation manifest missing")
    source_rows = _jsonl(
        run_dir / str(source_reference.get("path")),
        expected_sha256=str(source_reference.get("sha256")),
        expected_rows=int(source_reference.get("rows", -1)),
    )
    expected_source_count = len(expected_rows) * 2
    if len(source_rows) != expected_source_count or source.get("rows") != expected_source_count:
        _fail("source observation count changed")
    source_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    source_keys: set[tuple[str, str, str]] = set()
    for row in source_rows:
        if set(row) != SOURCE_FIELDS:
            _fail("source raw row schema changed")
        capability_id = str(row["capability_id"])
        condition = str(row["condition"])
        row_id = str(row["row_id"])
        key = (capability_id, condition, row_id)
        if (
            key in source_keys
            or capability_id not in capability_ids
            or condition not in {"BEFORE", "AFTER"}
        ):
            _fail("source raw row identity changed")
        source_keys.add(key)
        expected = expected_rows.get(row_id)
        if expected is None or row["prompt_sha256"] != expected["prompt_sha256"]:
            _fail("source prompt commitment changed")
        prediction = canonical_prediction(int(row["prediction_token_id"]), source_token_ids)
        if prediction != row["canonical_prediction"]:
            _fail("source token accounting changed")
        _probabilities(row["canonical_output_probabilities"], label="source")
        source_groups[(capability_id, condition)].append(row)
    if len(source_keys) != expected_source_count:
        _fail("source matrix incomplete")

    source_metrics: dict[str, Any] = {}
    gates = config["gates"]
    source_pass = True
    for capability_id in capability_ids:
        before_metric = _metrics(source_groups[(capability_id, "BEFORE")], answers)
        after_metric = _metrics(source_groups[(capability_id, "AFTER")], answers)
        gain = after_metric["accuracy"] - before_metric["accuracy"]
        source_metrics[capability_id] = {
            "BEFORE": before_metric,
            "AFTER": after_metric,
            "gain": gain,
        }
        source_pass &= after_metric["accuracy"] >= float(
            gates["source_after_accuracy_minimum"]
        ) and gain >= float(gates["source_gain_minimum"])

    host_receipts = recipient_execution.get("hosts")
    hosts = list(config["public_matrix"]["hosts"])
    if not isinstance(host_receipts, list) or [item.get("host") for item in host_receipts] != hosts:
        _fail("recipient host receipt inventory changed")
    host_tokens: dict[str, list[int]] = {}
    for host_receipt in host_receipts:
        host = str(host_receipt["host"])
        tokens = [int(value) for value in host_receipt.get("target_token_ids", [])]
        if len(tokens) != 8 or len(set(tokens)) != 8:
            _fail(f"recipient token map invalid: {host}")
        if (
            host_receipt.get("model_state_sha256_before")
            != host_receipt.get("model_state_sha256_after")
            or host_receipt.get("recipient_optimizer_steps") != gates["recipient_optimizer_steps"]
            or host_receipt.get("interpreter_learned_parameters")
            != gates["interpreter_learned_parameters"]
            or host_receipt.get("source_model_loaded") is not False
        ):
            _fail(f"recipient immutability boundary failed: {host}")
        host_tokens[host] = tokens

    recipient_rows = _jsonl(
        run_dir / str(recipient_reference.get("path")),
        expected_sha256=str(recipient_reference.get("sha256")),
        expected_rows=int(recipient_reference.get("rows", -1)),
    )
    conditions = list(config["public_matrix"]["conditions"])
    expected_recipient_count = (
        len(hosts)
        * len(capability_ids)
        * len(expected_rows)
        // len(capability_ids)
        * len(conditions)
    )
    if len(recipient_rows) != expected_recipient_count:
        _fail("recipient observation count changed")
    if any(
        item.get("rows") != len(expected_rows) // len(capability_ids) * len(conditions)
        for item in host_receipts
    ):
        _fail("per-host observation count changed")

    vm = CanonicalTransitionVM()
    recipient_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    recipient_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in recipient_rows:
        if set(row) != RECIPIENT_FIELDS:
            _fail("recipient raw row schema changed")
        host = str(row["host"])
        capability_id = str(row["capability_id"])
        condition = str(row["condition"])
        row_id = str(row["row_id"])
        key = (host, capability_id, condition, row_id)
        if (
            key in recipient_index
            or host not in hosts
            or capability_id not in capability_ids
            or condition not in conditions
        ):
            _fail("recipient raw row identity changed")
        expected = expected_rows.get(row_id)
        if expected is None or row["prompt_sha256"] != expected["prompt_sha256"]:
            _fail("recipient prompt commitment changed")
        prediction = canonical_prediction(int(row["prediction_token_id"]), host_tokens[host])
        if prediction != row["canonical_prediction"]:
            _fail("recipient token accounting changed")
        canonical_probabilities = _probabilities(
            row["canonical_output_probabilities"], label="recipient"
        )
        vm_probabilities = _probabilities(row["vm_output_probabilities"], label="VM")
        active = condition not in {"BASE", "REMOVED", "INTERPRETER_REMOVED"}
        if row["interpreter_active"] is not active:
            _fail("interpreter condition accounting changed")
        capability_index = capability_ids.index(capability_id)
        wrong_index = (capability_index + 1) % len(after)
        condition_latents = _condition_latents(
            after[capability_index], before, after[wrong_index], capability_id
        )
        expected_hash: str | None = None
        if condition in {"AFTER", "RESTORED", "INTERPRETER_REMOVED"}:
            expected_hash = after_hashes[capability_id]
        elif condition == "BEFORE":
            expected_hash = str(receipt["packages"]["before"]["sha256"])
        elif condition == "WRONG":
            expected_hash = after_hashes[capability_ids[wrong_index]]
        elif condition in {"ZERO", "RANDOM", "SHUFFLED"}:
            expected_hash = "CONTROL_" + condition
        if row["package_sha256"] != expected_hash:
            _fail("package invocation identity changed")
        if active:
            expected_vm = vm.execute(condition_latents[condition], [str(expected["prompt"])])[
                0
            ].tolist()
            if not _close(vm_probabilities, expected_vm):
                _fail("stored VM output does not recompute")
            if not _close(canonical_probabilities, vm_probabilities):
                _fail("host codec changed canonical capability distribution")
            if row["canonical_prediction"] is None or bytes.fromhex(
                str(row["canonical_output_utf8_hex"])
            ) != str(row["canonical_prediction"]).encode("utf-8"):
                _fail("canonical output realization changed")
        else:
            if not _close(canonical_probabilities, vm_probabilities):
                _fail("inactive VM evidence does not equal base host evidence")
            try:
                bytes.fromhex(str(row["canonical_output_utf8_hex"]))
            except ValueError as exc:
                raise R10VerificationError("output byte encoding changed") from exc
        recipient_index[key] = row
        recipient_groups[(host, capability_id, condition)].append(row)
    if len(recipient_index) != expected_recipient_count:
        _fail("recipient matrix incomplete")

    recipient_metrics: dict[str, Any] = {}
    recipient_pass = True
    negative_conditions = {"BEFORE", "WRONG", "ZERO", "RANDOM", "SHUFFLED"}
    for host in hosts:
        recipient_metrics[host] = {}
        for capability_id in capability_ids:
            values = {
                condition: _metrics(recipient_groups[(host, capability_id, condition)], answers)
                for condition in conditions
            }
            recipient_metrics[host][capability_id] = values
            recipient_pass &= values["AFTER"]["accuracy"] == float(
                gates["recipient_after_accuracy"]
            )
            recipient_pass &= values["RESTORED"]["accuracy"] == float(
                gates["recipient_restored_accuracy"]
            )
            recipient_pass &= all(
                values[condition]["accuracy"] <= float(gates["negative_control_accuracy_maximum"])
                for condition in negative_conditions
            )
            for row_id in expected_rows:
                # Only compare rows belonging to this capability.
                after_key = (host, capability_id, "AFTER", row_id)
                if after_key not in recipient_index:
                    continue
                restored = recipient_index[(host, capability_id, "RESTORED", row_id)]
                after_row = recipient_index[after_key]
                if after_row != {**restored, "condition": "AFTER"}:
                    _fail("restored output is not an exact paired restoration")
                base = recipient_index[(host, capability_id, "BASE", row_id)]
                for condition in ("REMOVED", "INTERPRETER_REMOVED"):
                    removed = recipient_index[(host, capability_id, condition, row_id)]
                    ignored = {"condition", "package_sha256"}
                    if {k: v for k, v in removed.items() if k not in ignored} != {
                        k: v for k, v in base.items() if k not in ignored
                    }:
                        _fail(f"{condition} does not exactly restore BASE")

    source_started = float(source_execution.get("started"))
    source_finished = float(source_execution.get("finished"))
    recipient_started = float(recipient_execution.get("started"))
    recipient_finished = float(recipient_execution.get("finished"))
    temporal_pass = (
        math.isfinite(source_started)
        and math.isfinite(source_finished)
        and math.isfinite(recipient_started)
        and math.isfinite(recipient_finished)
        and source_started < source_finished <= recipient_started < recipient_finished
    )
    if recipient_execution.get("physical_source_file_absence_claimed") is not False:
        _fail("physical source-file absence was improperly claimed")
    package_reuse_pass = all(
        recipient_index[(host, capability_id, condition, row_id)]["package_sha256"]
        == after_hashes[capability_id]
        for host in hosts
        for capability_id in capability_ids
        for condition in ("AFTER", "RESTORED")
        for row_id in expected_rows
        if (host, capability_id, condition, row_id) in recipient_index
    )
    passed = bool(source_pass and recipient_pass and temporal_pass and package_reuse_pass)
    result = {
        "format": "abi-copy-paste-r10-verification/1",
        "claim_target": "ABI-C4",
        "claim_ceiling": "RUNTIME_OWNED_COPY_PASTE_EXECUTION_ONLY",
        "verdict": "PASS" if passed else "FAIL",
        "source_metrics": source_metrics,
        "recipient_metrics": recipient_metrics,
        "recomputed_gates": {
            "source_learning": bool(source_pass),
            "recipient_matrix": bool(recipient_pass),
            "temporal_separation": bool(temporal_pass),
            "same_package_reuse": bool(package_reuse_pass),
        },
        "stronger_claims_open": [
            "native_recipient_internalization",
            "arbitrary_teacher_extraction",
            "english_or_domain_transfer",
            "information_minimality",
            "teacher_quality_parity",
            "lora_or_distillation_superiority",
        ],
        "receipt_evidence_sha256": receipt["evidence_sha256"],
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    if not passed:
        _fail("one or more recomputed R10 gates failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        result = verify(Path(args.config).resolve(), Path(args.run_dir).resolve())
        if args.output:
            output = Path(args.output).resolve()
            if output.exists():
                raise R10VerificationError(f"immutable verification exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        CopyPasteRuntimeError,
        R10RunError,
        R10VerificationError,
    ) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
