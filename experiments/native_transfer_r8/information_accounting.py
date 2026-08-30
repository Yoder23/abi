"""Recompute R8 information and compute accounting from immutable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import zlib
from pathlib import Path
from typing import Any, Mapping

from safetensors import safe_open

from .capability_generator import (
    OpaqueCapability,
    canonical_json_bytes,
    generate_rows,
)
from .native_host import sha256_file


class AccountingError(RuntimeError):
    """Raised when an accounting input is missing, stale, or ambiguous."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AccountingError(f"required accounting input unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise AccountingError(f"accounting input is not an object: {path}")
    return value


def _valid_evidence(value: Mapping[str, Any]) -> bool:
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    return stored == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _heldout_capabilities(value: Mapping[str, Any]) -> list[OpaqueCapability]:
    capabilities = []
    for row in value.get("capabilities", []):
        capabilities.append(
            OpaqueCapability(
                capability_id=str(row["capability_id"]),
                offsets=tuple(int(item) for item in row["offsets"]),
                seed_commitment=str(row["seed_commitment"]),
            )
        )
    if not capabilities:
        raise AccountingError("held-out capability inventory is empty")
    return capabilities


def _canonical_dataset_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def account(config_path: Path, campaign_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise AccountingError(f"immutable accounting output exists: {output}")
    config = _json(config_path)
    private = _json(campaign_root / "evaluator_private/capabilities.json")
    source = _json(campaign_root / "heldout_source/receipt.json")
    if not _valid_evidence(private) or not _valid_evidence(source):
        raise AccountingError("private or source receipt hash is stale")
    capabilities = _heldout_capabilities(private)
    if len(capabilities) != int(config["splits"]["heldout_capabilities"]):
        raise AccountingError("held-out capability count changed")

    prefix_path = campaign_root / "heldout_source" / source["prefixes"]["path"]
    if sha256_file(prefix_path) != source["prefixes"]["sha256"]:
        raise AccountingError("held-out source prefix tensor changed")
    with safe_open(str(prefix_path), framework="pt", device="cpu") as handle:
        prefix_shape = tuple(int(value) for value in handle.get_slice("after").get_shape())
    if prefix_shape[0] != len(capabilities) or prefix_shape[1] != int(
        config["training"]["source_prefix_length"]
    ):
        raise AccountingError("held-out source prefix shape changed")
    # One float32 prefix slice is the learned per-capability source delta.
    teacher_delta_bytes = prefix_shape[1] * prefix_shape[2] * 4
    bridge_rows = {}
    bridge_total = 0
    bridge_wall = 0.0
    for host in sorted(config["models"]["recipients"]):
        receipt_path = campaign_root / "pre_reveal/bridges" / host / "receipt.json"
        receipt = _json(receipt_path)
        if not _valid_evidence(receipt):
            raise AccountingError(f"bridge receipt hash is stale: {host}")
        bridge_path = receipt_path.parent / receipt["bridge"]["path"]
        if sha256_file(bridge_path) != receipt["bridge"]["sha256"]:
            raise AccountingError(f"bridge bytes changed: {host}")
        size = bridge_path.stat().st_size
        wall = float(receipt["training"]["wall_seconds"])
        bridge_rows[host] = {
            "one_time_bridge_bytes": size,
            "one_time_bridge_parameters": int(receipt["bridge"]["parameters"]),
            "one_time_bridge_optimizer_steps": int(receipt["training"]["optimizer_steps"]),
            "one_time_bridge_training_wall_seconds": wall,
        }
        bridge_total += size
        bridge_wall += wall

    source_by_capability = {
        str(row["capability_id"]): row for row in source.get("source_evaluation", [])
    }
    source_training = source.get("source_training", {})
    source_wall_total = float(source_training.get("wall_seconds", 0.0))
    per_capability = []
    for index, capability in enumerate(capabilities):
        train_rows = generate_rows(
            capability,
            split="source_train",
            rows=int(config["splits"]["source_train_rows_per_capability"]),
            depths=config["capability_family"]["source_train_depths"],
            seed=int(config["training"]["seed"]) + 8009 * index,
        )
        raw = _canonical_dataset_bytes(train_rows)
        package_dir = campaign_root / "heldout_source/packages" / capability.capability_id
        packages = {}
        for name in ("before", "after", "permuted_teacher_delta"):
            path = package_dir / f"{name}.abipkg"
            packages[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        if capability.capability_id not in source_by_capability:
            raise AccountingError(f"source evaluation missing: {capability.capability_id}")
        per_capability.append(
            {
                "capability_id": capability.capability_id,
                "source_training_rows": len(train_rows),
                "source_training_dataset_bytes": len(raw),
                "source_training_dataset_zlib9_bytes": len(zlib.compress(raw, level=9)),
                "teacher_parameter_delta_float32_bytes": teacher_delta_bytes,
                "capability_packages": packages,
                "per_capability_source_optimizer_steps": int(
                    source_training.get("optimizer_steps", 0)
                ) // len(capabilities),
                "per_capability_source_training_wall_seconds_allocated": source_wall_total
                / len(capabilities),
                "recipient_specific_optimizer_steps": 0,
            }
        )

    recipient_inference = {}
    for host in sorted(config["models"]["recipients"]):
        manifest_path = campaign_root / "recipients" / host / "manifest.json"
        if not manifest_path.is_file():
            recipient_inference[host] = {"available": False}
            continue
        manifest = _json(manifest_path)
        recipient_inference[host] = {
            "available": True,
            "raw_rows": int(manifest["rows"]),
            "wall_seconds": float(manifest["wall_seconds"]),
            "recipient_optimizer_steps": int(manifest["recipient_optimizer_steps"]),
        }

    value = {
        "format": "abi-native-transfer-r8-information-accounting/1",
        "config_sha256": sha256_file(config_path),
        "machine": platform.node(),
        "per_capability": per_capability,
        "one_time_host_bridges": bridge_rows,
        "one_time_bridge_bytes_total": bridge_total,
        "one_time_bridge_training_wall_seconds_total": bridge_wall,
        "recipient_inference": recipient_inference,
        "performance_size_curve": {
            "status": "NOT_MEASURED" if not (campaign_root / "budgets/manifest.json").is_file() else "RAW_EVIDENCE_AVAILABLE",
            "manifest": "budgets/manifest.json",
        },
        "scope_note": "Counts are empirical artifact and dataset sizes; no global information-minimality claim is made.",
    }
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        value = account(
            Path(args.config).resolve(),
            Path(args.campaign_root).resolve(),
            Path(args.output).resolve(),
        )
    except AccountingError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
