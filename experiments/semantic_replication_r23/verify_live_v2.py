"""Strictly verify the R23 repaired live replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
)
from experiments.generative_transfer_r21.hash_assurance_binding import (
    selfless_evidence_hash,
)

from . import verify_live as base_verify
from .live_repair_binding import load_repair_config


def verify(config_path: Path, result_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    repair = load_repair_config(root, config_path)
    live_path = (root / str(repair["live_config"]["path"])).resolve()
    live = json_object(live_path)
    r23 = json_object(root / str(repair["r23_config"]["path"]))
    hidden_engine_path = (root / str(live["candidate_engine"]["path"])).resolve()
    public_engine_path = (
        root / str(r23["public_candidate_engine"]["path"])
    ).resolve()
    binding = json_object(result_dir / "repair_binding.json")
    if (
        binding.get("format") != "abi-r23-live-repair-run/1"
        or binding.get("repair_config_sha256") != sha256_file(config_path)
        or binding.get("live_receipt_sha256")
        != sha256_file(result_dir / "receipt.json")
        or binding.get("evidence_sha256") != selfless_evidence_hash(binding)
        or binding.get("systems_joins") != 1
        or binding.get("systems_source_sha256") != sha256_file(public_engine_path)
        or any(
            binding.get(name) != 0
            for name in (
                "candidate_changes",
                "package_changes",
                "scorer_changes",
                "gate_changes",
            )
        )
    ):
        raise R14Error("R23 live-repair binding failed")
    original_json = base_verify.json_object
    joins = 0

    def joined_json(path: Path) -> dict[str, Any]:
        nonlocal joins
        document = original_json(path)
        if path.resolve() == hidden_engine_path:
            public = original_json(public_engine_path)
            if "systems" in document or not isinstance(public.get("systems"), dict):
                raise R14Error("R23 strict systems repair precondition changed")
            document = {**document, "systems": public["systems"]}
            joins += 1
        return document

    base_verify.json_object = joined_json
    try:
        underlying = base_verify.verify(live_path, result_dir)
    finally:
        base_verify.json_object = original_json
    if joins != 1:
        raise R14Error("R23 strict systems metadata was not joined exactly once")
    result = {
        "format": "abi-r23-live-repair-strict-verification/1",
        "status": underlying["status"],
        "repair_config_sha256": sha256_file(config_path),
        "repair_binding_sha256": sha256_file(result_dir / "repair_binding.json"),
        "live_receipt_sha256": sha256_file(result_dir / "receipt.json"),
        "systems_joins": joins,
        "underlying_verification": underlying,
        "candidate_changes": 0,
        "package_changes": 0,
        "scorer_changes": 0,
        "gate_changes": 0,
        "scientific_claim": "R23_BOUNDED_FRESH_SEMANTIC_SUPPLIED_CONTENT_TRANSFER",
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_ABI_MOONSHOT",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = selfless_evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.config, args.result)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
