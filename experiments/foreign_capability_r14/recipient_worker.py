"""Execute R14 packages in one clean frozen R11 recipient process."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from experiments.native_isa_r11.core import load_package
from experiments.native_isa_r11.run import _host_matrix

from .capability import heldout_capabilities
from .core import (
    R14Error,
    capability_rows,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)


def summarize(rows: list[dict[str, Any]], evaluation: list[list[dict[str, Any]]]) -> dict[str, Any]:
    answers = {
        str(row["row_id"]): int(row["answer"]) for capability in evaluation for row in capability
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    indexed = set()
    for row in rows:
        key = (str(row["capability_id"]), str(row["condition"]), str(row["row_id"]))
        if key in indexed or key[2] not in answers:
            raise R14Error("duplicate or unknown recipient observation")
        indexed.add(key)
        grouped[key[:2]].append(row)
    accuracy = {
        f"{capability_id}/{condition}": sum(
            int(int(row["canonical_prediction"]) == answers[str(row["row_id"])]) for row in values
        )
        / len(values)
        for (capability_id, condition), values in grouped.items()
    }
    removal_equal_base = True
    for capability_id in {key[0] for key in grouped}:
        base = {
            str(row["row_id"]): int(row["prediction_token_id"])
            for row in grouped[(capability_id, "BASE")]
        }
        for condition in ("REMOVED", "BACKEND_REMOVED", "CODEC_REMOVED"):
            removed = {
                str(row["row_id"]): int(row["prediction_token_id"])
                for row in grouped[(capability_id, condition)]
            }
            removal_equal_base = removal_equal_base and removed == base
    return {"accuracy": accuracy, "removal_conditions_equal_base": removal_equal_base}


def run(
    config_path: Path,
    reveal_path: Path,
    manifest_path: Path,
    host_key: str,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable recipient output exists: {output}")
    config = json_object(config_path)
    reveal = json_object(reveal_path)
    manifest = json_object(manifest_path)
    capabilities = heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    rows_by_capability = capability_rows(config, capabilities)
    evaluation = [item["recipient"] for item in rows_by_capability]
    root = config_path.parents[3]
    package_dir = manifest_path.parent / "packages"
    before_path = package_dir / str(manifest["before"]["path"])
    before_package, before = load_package(before_path)
    if sha256_file(before_path) != manifest["before"]["sha256"]:
        raise R14Error("before package changed")
    after = []
    for item in manifest["after"]:
        path = package_dir / str(item["path"])
        package, transition = load_package(path)
        if (
            sha256_file(path) != item["sha256"]
            or package["transition_sha256"] != item["transition_sha256"]
        ):
            raise R14Error("after package changed")
        after.append(transition)
    if before_package["transition_sha256"] != manifest["before"]["transition_sha256"]:
        raise R14Error("before transition changed")
    observations, host_receipt = _host_matrix(
        host_key,
        config,
        load_file(str(root / config["codec_freeze"]["tensors"]), device="cpu"),
        json_object(root / config["codec_freeze"]["receipt"]),
        capabilities,
        evaluation,
        before,
        after,
        manifest,
    )
    output.mkdir(parents=True)
    rows_path = output / "observations.jsonl"
    write_jsonl_once(rows_path, observations)
    receipt = {
        "format": "abi-r14-recipient-worker/1",
        "host": host_key,
        "pid": os.getpid(),
        "config_sha256": sha256_file(config_path),
        "reveal_sha256": sha256_file(reveal_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_adapter_argument_present": False,
        "source_adapter_loaded": False,
        "host_receipt": host_receipt,
        "observations": {
            "path": rows_path.name,
            "rows": len(observations),
            "sha256": sha256_file(rows_path),
        },
        "summary": summarize(observations, evaluation),
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = run(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.manifest).resolve(),
            str(args.host),
            Path(args.output).resolve(),
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
