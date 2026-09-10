"""Rerun R21 live verification with content-targeted package corruption."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, sha256_file, write_json_once

from . import live_verify_v6
from .hash_assurance_binding import selfless_evidence_hash
from .hostile_repair_binding import load_hostile_repair_config


def targeted_tensor_corruption(final_byte_mutated_archive: bytes | bytearray) -> bytes:
    payload = bytearray(final_byte_mutated_archive)
    if not payload:
        raise R14Error("R21 hostile candidate archive is empty")
    payload[-1] ^= 1
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            member = archive.getinfo("tensors.safetensors")
            if member.compress_type != zipfile.ZIP_STORED:
                raise R14Error("R21 tensor member is unexpectedly compressed")
            header = int(member.header_offset)
            if payload[header : header + 4] != b"PK\x03\x04":
                raise R14Error("R21 tensor member local header is invalid")
            name_length = int.from_bytes(payload[header + 26 : header + 28], "little")
            extra_length = int.from_bytes(payload[header + 28 : header + 30], "little")
            data_offset = header + 30 + name_length + extra_length
            offset = data_offset + int(member.file_size) // 2
    except (zipfile.BadZipFile, KeyError) as exc:
        raise R14Error("R21 tensor member is unavailable for targeted mutation") from exc
    if offset < 0 or offset >= len(payload):
        raise R14Error("R21 targeted mutation offset is outside archive")
    payload[offset] ^= 1
    return bytes(payload)


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R21 hostile-repaired verification exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config = load_hostile_repair_config(root, config_path)
    live_config = root / str(config["live_verification_config"]["path"])
    original_write = Path.write_bytes
    targeted_writes = 0

    def repaired_write(path: Path, data: bytes | bytearray) -> int:
        nonlocal targeted_writes
        if path.suffix == ".cake" and path.name.startswith("corrupt-"):
            data = targeted_tensor_corruption(data)
            targeted_writes += 1
        return original_write(path, data)

    Path.write_bytes = repaired_write
    try:
        receipt = live_verify_v6.run(live_config, output)
    finally:
        Path.write_bytes = original_write
    if targeted_writes != 24:
        raise R14Error("R21 targeted mutation count changed")
    binding = {
        "format": "abi-r21-hostile-repair-run/1",
        "hostile_repair_config_sha256": sha256_file(config_path),
        "live_receipt_sha256": sha256_file(output / "receipt.json"),
        "live_receipt_evidence_sha256": receipt["evidence_sha256"],
        "targeted_tensor_mutations": targeted_writes,
        "candidate_changes": 0,
        "gate_changes": 0,
        "full_abi_moonshot": "OPEN",
    }
    binding["evidence_sha256"] = selfless_evidence_hash(binding)
    write_json_once(output / "hostile_repair_binding.json", binding)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
