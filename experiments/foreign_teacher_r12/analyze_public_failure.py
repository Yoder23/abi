"""Live train/depth analysis for an immutable failed R12-A public run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from experiments.native_isa_r11.core import sha256_file
from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
    generate_rows,
    public_capabilities,
)
from experiments.native_transfer_r8.native_host import (
    SPECS,
    FrozenNeuralHost,
    GenericRecipientAdapterSet,
)

from .public_preflight import _json, build_training_rows
from .teacher import R12TeacherError, evaluate
from .verify_public import _evidence


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _json(config_path)
    receipt = _json(run_dir / "receipt.json")
    _evidence(receipt)
    artifact = receipt["adapter_artifact"]
    adapter_path = run_dir / str(artifact["path"])
    if (
        not adapter_path.is_file()
        or adapter_path.stat().st_size != int(artifact["bytes"])
        or sha256_file(adapter_path) != artifact["sha256"]
    ):
        raise R12TeacherError("failed-run adapter artifact changed")
    capability = public_capabilities(
        int(config["data"]["capability_seed"]), split="development", count=1
    )[0]
    evaluation_rows = generate_rows(
        capability,
        split="r12_public_evaluation",
        rows=int(config["data"]["evaluation_rows"]),
        depths=config["data"]["evaluation_depths"],
        seed=int(config["data"]["evaluation_seed"]),
    )
    training_rows = build_training_rows(config, capability, evaluation_rows)
    host = FrozenNeuralHost(SPECS["qwen2"], device="cuda")
    adapters = GenericRecipientAdapterSet(
        host, rank=int(config["training"]["lora_rank"])
    )
    adapters.load_state(load_file(str(adapter_path), device="cpu"))
    adapters.freeze()
    batch_size = int(config["training"]["evaluation_batch_size"])

    def grouped(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            str(depth): evaluate(
                host,
                [row for row in rows if int(row["depth"]) == depth],
                batch_size=batch_size,
            )
            for depth in sorted({int(row["depth"]) for row in rows})
        }

    result = {
        "format": "abi-r12a-public-failure-analysis/1",
        "source_receipt_evidence_sha256": receipt["evidence_sha256"],
        "source_status": receipt.get("status"),
        "adapter_sha256": artifact["sha256"],
        "training": {
            "overall": evaluate(host, training_rows, batch_size=batch_size),
            "by_depth": grouped(training_rows),
        },
        "evaluation": {
            "overall": evaluate(host, evaluation_rows, batch_size=batch_size),
            "by_depth": grouped(evaluation_rows),
        },
        "base_state_unchanged": adapters.base_state_sha256()
        == receipt["teacher"]["base_state_sha256_before"],
        "claim_ceiling": "FAILED_PUBLIC_TEACHER_DIAGNOSTIC_ONLY",
    }
    result["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        if output.exists():
            raise R12TeacherError(f"immutable failure analysis exists: {output}")
        result = run(Path(args.config).resolve(), Path(args.run_dir).resolve())
        output.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
