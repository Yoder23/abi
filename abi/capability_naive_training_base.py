"""Materialize a hash-bound training view of the capability-naive receiver.

The receiver package is immutable control evidence and intentionally uses a
receiver manifest rather than acquisition metadata.  This module creates a
new ABI-owned directory with bit-exact checkpoint/tokenizer bytes plus the
minimum metadata required by the acquisition trainer.  It never changes the
receiver and never introduces learned parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


FORMAT = "abi-layercake-capability-naive-training-base/1"
RECEIVER_FORMAT = "abi-layercake-capability-naive-receiver/1"


class CapabilityNaiveTrainingBaseError(RuntimeError):
    """Raised when the immutable receiver cannot be materialized safely."""


def _canonical_sha(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_training_metadata(
    receiver: Mapping[str, Any], *, receiver_manifest_sha256: str
) -> dict[str, Any]:
    """Create an honest acquisition-base metadata record from a receiver."""

    if (
        receiver.get("format") != RECEIVER_FORMAT
        or receiver.get("status") != "SEALED_CAUSAL_NEGATIVE_CONTROL"
        or receiver.get("role") != "capability_naive_receiver"
    ):
        raise CapabilityNaiveTrainingBaseError("receiver role is invalid")
    imported = receiver.get("imported_information")
    if not isinstance(imported, Mapping) or any(
        imported.get(key) != 0
        for key in (
            "foreign_teacher_parameters_copied",
            "layercake_learned_parameters_copied",
            "bridge_parameters",
            "training_steps",
            "training_tokens",
        )
    ):
        raise CapabilityNaiveTrainingBaseError(
            "receiver contains learned or imported information"
        )
    checkpoint = receiver.get("checkpoint")
    host = receiver.get("layercake_host")
    if not isinstance(checkpoint, Mapping) or not isinstance(host, Mapping):
        raise CapabilityNaiveTrainingBaseError("receiver metadata is incomplete")
    tensor_rows = checkpoint.get("tensors")
    if not isinstance(tensor_rows, list) or not tensor_rows:
        raise CapabilityNaiveTrainingBaseError("receiver tensor ledger is missing")
    total = int(checkpoint.get("parameter_count", 0))
    cakes: dict[str, int] = {}
    for row in tensor_rows:
        name = str(row.get("name", ""))
        if name.startswith("task_cakes."):
            route = name.split(".", 2)[1]
            cakes[route] = cakes.get(route, 0) + int(row.get("parameters", 0))
    if total <= 0 or len(cakes) != int(host["architecture"]["task_cakes"]):
        raise CapabilityNaiveTrainingBaseError("receiver parameter ledger changed")
    cake_sizes = set(cakes.values())
    if len(cake_sizes) != 1:
        raise CapabilityNaiveTrainingBaseError("task-cake sizes are inconsistent")
    active = total - (len(cakes) - 1) * next(iter(cake_sizes))
    metadata: dict[str, Any] = {
        "format": FORMAT,
        "status": "SEALED_CAPABILITY_NAIVE_ACQUISITION_BASE",
        "architecture": dict(host["architecture"]),
        "checkpoint": {
            "path": "model.safetensors",
            "sha256": str(checkpoint["sha256"]),
            "bytes": int(checkpoint["bytes"]),
        },
        "parameters": {"total": total, "active": active},
        "canonical_semantic_abi": {
            "sha256": str(host["canonical_semantic_abi_sha256"])
        },
        "source_receiver": {
            "format": RECEIVER_FORMAT,
            "seed": int(receiver["seed"]),
            "manifest_file_sha256": receiver_manifest_sha256,
            "manifest_sha256": str(receiver["manifest_sha256"]),
            "checkpoint_copied_bit_exact": True,
            "tokenizer_copied_bit_exact": True,
        },
        "imported_information": dict(imported),
        "claim_boundary": (
            "This is only a bit-exact training view of the capability-naive "
            "LayerCake receiver. It contains no learned English, foreign "
            "teacher parameters, LayerCake learned parameters, or bridge state."
        ),
    }
    return metadata


def materialize_training_base(
    *, receiver_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    receiver_path = Path(receiver_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise CapabilityNaiveTrainingBaseError(
            f"training base is immutable: {output_path}"
        )
    manifest_path = receiver_path / "manifest.json"
    receiver = json.loads(manifest_path.read_text(encoding="utf-8"))
    claimed = receiver.get("manifest_sha256")
    unsigned = dict(receiver)
    unsigned.pop("manifest_sha256", None)
    if claimed != _canonical_sha(unsigned):
        raise CapabilityNaiveTrainingBaseError("receiver manifest hash changed")
    manifest_file_sha = _sha256_file(manifest_path)
    metadata = build_training_metadata(
        receiver, receiver_manifest_sha256=manifest_file_sha
    )
    checkpoint_source = receiver_path / str(receiver["checkpoint"]["path"])
    if _sha256_file(checkpoint_source) != receiver["checkpoint"]["sha256"]:
        raise CapabilityNaiveTrainingBaseError("receiver checkpoint changed")
    assets = receiver.get("tokenizer", {}).get("assets")
    if not isinstance(assets, list) or not assets:
        raise CapabilityNaiveTrainingBaseError("receiver tokenizer ledger missing")
    for asset in assets:
        source = receiver_path / str(asset["path"])
        if source.parent != receiver_path or _sha256_file(source) != asset["sha256"]:
            raise CapabilityNaiveTrainingBaseError("receiver tokenizer changed")
    output_path.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(checkpoint_source, output_path / "model.safetensors")
    for asset in assets:
        source = receiver_path / str(asset["path"])
        shutil.copyfile(source, output_path / source.name)
    if _sha256_file(output_path / "model.safetensors") != receiver["checkpoint"]["sha256"]:
        raise CapabilityNaiveTrainingBaseError("training-base copy is not exact")
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    metadata = materialize_training_base(
        receiver_path=args.receiver, output_path=args.output
    )
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "checkpoint_sha256": metadata["checkpoint"]["sha256"],
                "active_parameters": metadata["parameters"]["active"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
