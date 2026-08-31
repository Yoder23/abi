"""Freshly replay the frozen R10 source and recipient executions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes
from experiments.native_transfer_r8.recipient_worker import _disable_network

from .run import (
    _evaluation_rows,
    _host_observations,
    _json,
    _resolve,
    _source_observations,
)
from .runtime import sha256_file


class R10ReplayError(RuntimeError):
    """Raised when live R10 behavior differs from immutable raw evidence."""


def _rows_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def replay(config_path: Path, run_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config = _json(config_path)
    receipt = _json(run_dir / "receipt.json")
    receipt_payload = dict(receipt)
    stored_evidence = receipt_payload.pop("evidence_sha256", None)
    if (
        receipt.get("config_sha256") != sha256_file(config_path)
        or stored_evidence != hashlib.sha256(canonical_json_bytes(receipt_payload)).hexdigest()
    ):
        raise R10ReplayError("immutable R10 receipt changed")
    r8 = _json(_resolve(root, str(config["r8_reference"]["config"])))
    capabilities, rows = _evaluation_rows(config, r8)
    manifest = receipt["packages"]
    _disable_network()

    source_rows, source_receipt = _source_observations(
        config,
        r8,
        _resolve(root, str(config["r8_reference"]["source_states"])),
        capabilities,
        rows,
    )
    stored_source = (run_dir / receipt["source_execution"]["observations"]["path"]).read_bytes()
    source_bytes = _rows_bytes(source_rows)
    if source_bytes != stored_source:
        raise R10ReplayError("fresh source execution differs from immutable raw rows")

    host_results = []
    recipient_rows = []
    for host in config["public_matrix"]["hosts"]:
        host_rows, host_receipt = _host_observations(
            str(host), config, run_dir, manifest, capabilities, rows
        )
        recipient_rows.extend(host_rows)
        host_results.append(
            {
                "host": host,
                "rows": len(host_rows),
                "model_state_sha256_before": host_receipt["model_state_sha256_before"],
                "model_state_sha256_after": host_receipt["model_state_sha256_after"],
                "exact_raw_replay": True,
            }
        )
    stored_recipient = (
        run_dir / receipt["recipient_execution"]["observations"]["path"]
    ).read_bytes()
    recipient_bytes = _rows_bytes(recipient_rows)
    if recipient_bytes != stored_recipient:
        raise R10ReplayError("fresh recipient execution differs from immutable raw rows")

    result = {
        "format": "abi-copy-paste-r10-live-replay/1",
        "status": "PASS_EXACT_LIVE_REPLAY",
        "claim_boundary": "R10 runtime-owned synthetic copy/paste component only",
        "source_rows": len(source_rows),
        "source_rows_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_model_state_sha256_before": source_receipt["model_state_sha256_before"],
        "source_model_state_sha256_after": source_receipt["model_state_sha256_after"],
        "recipient_rows": len(recipient_rows),
        "recipient_rows_sha256": hashlib.sha256(recipient_bytes).hexdigest(),
        "hosts": host_results,
        "packages_reused_from_run": True,
        "optimizer_steps": 0,
        "interpreter_learned_parameters": 0,
        "r10_overall_verdict_changed": False,
        "r10_overall_verdict": "FAIL_SOURCE_NATIVE_GENERALIZATION",
        "hardware": {
            "device": "cuda",
            "cuda_device_name": torch.cuda.get_device_name(0),
        },
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    del source_rows, recipient_rows
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = replay(Path(args.config).resolve(), Path(args.run_dir).resolve())
        output = Path(args.output).resolve()
        if output.exists():
            raise R10ReplayError(f"immutable replay output exists: {output}")
        output.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
