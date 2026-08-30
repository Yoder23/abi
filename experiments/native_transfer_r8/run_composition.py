"""Prepare private composition labels and launch a separate label-free worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .capability_generator import (
    OpaqueCapability,
    canonical_json_bytes,
    generate_composition_rows,
    worker_rows,
    write_jsonl_once,
)
from .native_host import sha256_file


class CompositionError(RuntimeError):
    """Raised when composition labels and worker custody are not separated."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CompositionError(f"composition input unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise CompositionError(f"composition input is not an object: {path}")
    return value


def _capabilities(path: Path) -> list[OpaqueCapability]:
    value = _json(path)
    return [
        OpaqueCapability(
            capability_id=str(row["capability_id"]),
            offsets=tuple(int(item) for item in row["offsets"]),
            seed_commitment=str(row["seed_commitment"]),
        )
        for row in value["capabilities"]
    ]


def _write_or_verify(path: Path, rows: list[dict[str, Any]]) -> None:
    expected = b"".join(canonical_json_bytes(row) for row in rows)
    if path.is_file():
        if path.read_bytes() != expected:
            raise CompositionError(f"existing immutable composition input changed: {path}")
        return
    write_jsonl_once(path, rows)


def prepare_and_run(
    config_path: Path,
    campaign_root: Path,
    output: Path,
    *,
    host_key: str,
) -> dict[str, Any]:
    config = _json(config_path)
    capabilities = _capabilities(campaign_root / "evaluator_private/capabilities.json")
    if len(capabilities) < 2:
        raise CompositionError("composition requires two held-out capabilities")
    first, second = capabilities[:2]
    evaluator_dir = campaign_root / "evaluator_private/composition"
    inputs_dir = campaign_root / "composition/worker_inputs"
    pair = {
        "format": "abi-native-transfer-r8-composition-pair/1",
        "capability_ids": [first.capability_id, second.capability_id],
        "config_sha256": sha256_file(config_path),
    }
    pair_path = inputs_dir / "pair.json"
    pair_payload = json.dumps(pair, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if pair_path.is_file() and pair_path.read_bytes() != pair_payload:
        raise CompositionError("existing immutable composition pair changed")
    if not pair_path.exists():
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        pair_path.write_bytes(pair_payload)
    source_dir = campaign_root / "heldout_source"
    first_private = (campaign_root / "evaluator_private/evaluation" / f"{first.capability_id}.jsonl")
    second_private = (campaign_root / "evaluator_private/evaluation" / f"{second.capability_id}.jsonl")
    first_rows = [json.loads(line) for line in first_private.read_bytes().splitlines()]
    second_rows = [json.loads(line) for line in second_private.read_bytes().splitlines()]
    cross_rows = generate_composition_rows(
        first,
        second,
        split="heldout_composition",
        rows=int(config["splits"]["evaluation_rows_per_capability"]),
        first_depths=config["capability_family"]["composition_evaluation_depths"],
        second_depths=config["capability_family"]["composition_evaluation_depths"],
        seed=int(config["training"]["seed"]) + 99173,
    )
    _write_or_verify(evaluator_dir / "cross.jsonl", cross_rows)
    _write_or_verify(inputs_dir / "first.jsonl", worker_rows(first_rows))
    _write_or_verify(inputs_dir / "second.jsonl", worker_rows(second_rows))
    public_cross = [
        {key: row[key] for key in ("row_id", "prompt", "prompt_sha256")}
        for row in cross_rows
    ]
    _write_or_verify(inputs_dir / "cross.jsonl", public_cross)
    command = [
        sys.executable,
        "-B",
        "-m",
        "experiments.native_transfer_r8.composition_worker",
        "--config",
        str(config_path),
        "--campaign-root",
        str(campaign_root),
        "--source-dir",
        str(source_dir),
        "--inputs-dir",
        str(inputs_dir),
        "--output",
        str(output),
        "--host",
        host_key,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise CompositionError(
            f"label-free composition worker failed ({completed.returncode}): {completed.stderr or completed.stdout}"
        )
    manifest = _json(output / "manifest.json")
    manifest["launcher_stdout_sha256"] = hashlib.sha256(
        completed.stdout.encode("utf-8")
    ).hexdigest()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", required=True, choices=("qwen2", "pythia", "t5"))
    args = parser.parse_args()
    try:
        value = prepare_and_run(
            Path(args.config).resolve(),
            Path(args.campaign_root).resolve(),
            Path(args.output).resolve(),
            host_key=args.host,
        )
    except CompositionError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
