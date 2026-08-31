"""Fresh live R11 execution using exact frozen package bytes from a prior run."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
    committed_heldout_capabilities,
)
from experiments.native_transfer_r8.recipient_worker import _disable_network

from .core import R11Error, load_package, sha256_bytes, sha256_file
from .run import _bind, _host_matrix, _json, _resolve, _rows, _write_json_once, _write_jsonl_once
from .verify import verify


def _load_original_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8")]


def _assert_exact_rows(
    original: Sequence[Mapping[str, Any]], replay: Sequence[Mapping[str, Any]], label: str
) -> None:
    if len(original) != len(replay):
        raise R11Error(f"{label} replay row count changed")
    for index, (left, right) in enumerate(zip(original, replay)):
        if dict(left) != dict(right):
            raise R11Error(f"{label} replay changed raw row {index}")


def replay(
    config_path: Path, reveal_path: Path, original_dir: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise R11Error(f"immutable replay output exists: {output}")
    original_verification = verify(config_path, reveal_path, original_dir)
    root = Path(__file__).resolve().parents[2]
    config = _json(config_path)
    bindings = _bind(root, config)
    reveal = _json(reveal_path)
    try:
        secret = bytes.fromhex(str(reveal["secret_hex"]))
    except ValueError as exc:
        raise R11Error("held-out reveal is not hexadecimal") from exc
    if (
        len(secret) != 32
        or sha256_bytes(secret) != config["heldout_seed_commitment"]
        or reveal.get("commitment") != config["heldout_seed_commitment"]
    ):
        raise R11Error("held-out reveal does not match preregistration")
    capabilities = committed_heldout_capabilities(
        reveal["secret_hex"],
        expected_commitment=config["heldout_seed_commitment"],
        count=int(config["data"]["heldout_capabilities"]),
    )
    _, evaluation_rows = _rows(config, capabilities)
    original_receipt = _json(original_dir / "receipt.json")
    manifest = original_receipt["packages"]
    package_items = [manifest["before"], *manifest["after"]]
    transitions = []
    for item in package_items:
        package_path = original_dir / "packages" / str(item["path"])
        if sha256_file(package_path) != item["sha256"]:
            raise R11Error("original frozen package identity changed")
        _, transition = load_package(package_path)
        transitions.append(transition)
    before, after = transitions[0], transitions[1:]
    output.mkdir(parents=True)
    package_dir = output / "packages"
    package_dir.mkdir()
    for item in package_items:
        source = original_dir / "packages" / str(item["path"])
        target = package_dir / str(item["path"])
        shutil.copyfile(source, target)
        if sha256_file(target) != item["sha256"]:
            raise R11Error("frozen package changed during replay copy")

    codec_receipt = _json(_resolve(root, config["codec_freeze"]["receipt"]))
    codec_tensors = load_file(str(_resolve(root, config["codec_freeze"]["tensors"])), device="cpu")
    _disable_network()
    teacher_started = time.perf_counter()
    source_rows, source_receipt = _host_matrix(
        "source",
        config,
        codec_tensors,
        codec_receipt,
        capabilities,
        evaluation_rows,
        before,
        after,
        manifest,
    )
    teacher_finished = time.perf_counter()
    original_source_rows = _load_original_rows(original_dir / "teacher_observations.jsonl")
    _assert_exact_rows(original_source_rows, source_rows, "teacher")
    source_row_count = len(source_rows)
    source_path = output / "teacher_observations.jsonl"
    _write_jsonl_once(source_path, source_rows)
    del source_rows, original_source_rows
    gc.collect()

    recipient_started = time.perf_counter()
    recipient_rows = []
    recipient_receipts = []
    for host_key in config["recipient_hosts"]:
        rows, host_receipt = _host_matrix(
            str(host_key),
            config,
            codec_tensors,
            codec_receipt,
            capabilities,
            evaluation_rows,
            before,
            after,
            manifest,
        )
        recipient_rows.extend(rows)
        recipient_receipts.append(host_receipt)
        print(json.dumps(host_receipt, sort_keys=True), flush=True)
    recipient_finished = time.perf_counter()
    original_recipient_rows = _load_original_rows(original_dir / "recipient_observations.jsonl")
    _assert_exact_rows(original_recipient_rows, recipient_rows, "recipient")
    recipient_row_count = len(recipient_rows)
    recipient_path = output / "recipient_observations.jsonl"
    _write_jsonl_once(recipient_path, recipient_rows)
    receipt = {
        "format": "abi-native-neural-isa-r11-run/1",
        "execution_mode": "FROZEN_PACKAGE_FRESH_LIVE_REPLAY",
        "replay_of_receipt_sha256": sha256_file(original_dir / "receipt.json"),
        "replay_of_verification_evidence_sha256": original_verification["evidence_sha256"],
        "config_sha256": sha256_file(config_path),
        "reveal_sha256": sha256_file(reveal_path),
        "bindings": bindings,
        "claim_target": "LOSSLESS_OUTPUT_EQUIVALENCE_SYNTHETIC_ABI_NATIVE_TEACHER",
        "claim_ceiling": "LEVEL_1_OUTPUT_EQUIVALENCE_ONLY",
        "packages": manifest,
        "teacher_training": original_receipt["teacher_training"],
        "teacher_execution": {
            "started": teacher_started,
            "finished": teacher_finished,
            "host": source_receipt,
            "observations": {
                "path": source_path.name,
                "sha256": sha256_file(source_path),
                "rows": source_row_count,
            },
        },
        "recipient_execution": {
            "started": recipient_started,
            "finished": recipient_finished,
            "teacher_finished_before_recipient_started": teacher_finished <= recipient_started,
            "teacher_loaded_in_recipient_process": False,
            "hosts": recipient_receipts,
            "observations": {
                "path": recipient_path.name,
                "sha256": sha256_file(recipient_path),
                "rows": recipient_row_count,
            },
        },
        "heldout": original_receipt["heldout"],
        "hardware": {
            "device": "cuda",
            "cuda_device_name": torch.cuda.get_device_name(0),
        },
        "exact_raw_row_replay": True,
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    _write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = replay(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.original_dir).resolve(),
            Path(args.output).resolve(),
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
