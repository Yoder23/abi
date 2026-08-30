"""Freeze every R8 pre-reveal component and bind the temporal order."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from .capability_generator import canonical_json_bytes
from .native_host import sha256_file


class FreezeError(RuntimeError):
    """Raised if held-out material exists or a required pre-reveal input is stale."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError(f"required JSON unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise FreezeError(f"expected JSON object: {path}")
    return value


def freeze(root: Path, config_path: Path, campaign_root: Path) -> dict[str, Any]:
    output = campaign_root / "freeze_receipt.json"
    reveal = campaign_root / "heldout_reveal.json"
    private = campaign_root / "evaluator_private"
    if output.exists() or reveal.exists() or private.exists():
        raise FreezeError("freeze requires held-out reveal and private evaluator data absent")
    if list(campaign_root.rglob("*.abipkg")):
        raise FreezeError("a held-out capability package exists before freeze")
    config = _json(config_path)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    campaign_status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "experiments/native_transfer_r8",
            "tests/test_native_transfer_r8.py",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if campaign_status:
        raise FreezeError("R8 code must be committed and clean before temporal freeze")
    source_dir = campaign_root / "pre_reveal/source_public"
    extraction_dir = campaign_root / "pre_reveal/meta_extraction"
    component_paths = [
        source_dir / "receipt.json",
        source_dir / "meta_source_prefixes.safetensors",
        extraction_dir / "receipt.json",
        extraction_dir / "meta_canonical_latents.safetensors",
    ]
    bridge_receipts = {}
    for host in sorted(config["models"]["recipients"]):
        base = campaign_root / "pre_reveal/bridges" / host
        component_paths.extend((base / "receipt.json", base / "bridge.safetensors"))
        receipt = _json(base / "receipt.json")
        if (
            receipt.get("frozen") is not True
            or receipt.get("heldout_reveal_present_during_training") is not False
            or receipt.get("capability_packages_present_during_training") != 0
            or receipt.get("recipient_parameters_trainable") != 0
            or receipt.get("recipient_optimizer_steps") != 0
        ):
            raise FreezeError(f"bridge receipt is not capability-blind: {host}")
        bridge_receipts[host] = receipt
    files = []
    for path in component_paths:
        if not path.is_file():
            raise FreezeError(f"required pre-reveal component missing: {path}")
        files.append(
            {
                "path": path.relative_to(campaign_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    code_root = root / "experiments/native_transfer_r8"
    for path in sorted(code_root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_file() and path.suffix in {".py", ".json", ".md"}:
            files.append(
                {
                    "path": "@code/" + path.relative_to(code_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if len({row["path"] for row in files}) != len(files):
        raise FreezeError("pre-reveal input inventory is duplicated")
    files.sort(key=lambda row: row["path"])
    value = {
        "format": "abi-native-transfer-r8-freeze-receipt/1",
        "status": "FROZEN_BEFORE_HELDOUT_REVEAL",
        "created_unix_time_ns": time.time_ns(),
        "machine": platform.node(),
        "git_commit": commit,
        "r8_worktree_status": "CLEAN",
        "config_sha256": sha256_file(config_path),
        "heldout_secret_commitment_sha256": config["splits"]["heldout_secret_commitment_sha256"],
        "heldout_reveal_present": False,
        "private_evaluator_data_present": False,
        "capability_packages_present": 0,
        "bridges": {
            host: {
                "receipt_sha256": sha256_file(
                    campaign_root / "pre_reveal/bridges" / host / "receipt.json"
                ),
                "bridge_sha256": receipt["bridge"]["sha256"],
                "optimizer_steps": receipt["training"]["optimizer_steps"],
                "heldout_optimizer_steps": 0,
            }
            for host, receipt in bridge_receipts.items()
        },
        "extractor_learned_parameters": 0,
        "inputs": files,
        "input_count": len(files),
        "input_aggregate_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()
    try:
        value = freeze(
            Path(args.root).resolve(),
            Path(args.config).resolve(),
            Path(args.campaign_root).resolve(),
        )
    except FreezeError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
