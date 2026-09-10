"""Run R23 live replay with the frozen public-engine systems binding."""

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

from . import live_verify as base_live
from .live_repair_binding import load_repair_config


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R23 repaired live result exists: {output}")
    root = Path(__file__).resolve().parents[2]
    repair = load_repair_config(root, config_path)
    live_path = (root / str(repair["live_config"]["path"])).resolve()
    live = json_object(live_path)
    r23 = json_object(root / str(repair["r23_config"]["path"]))
    hidden_engine_path = (root / str(live["candidate_engine"]["path"])).resolve()
    public_engine_path = (
        root / str(r23["public_candidate_engine"]["path"])
    ).resolve()
    original_json = base_live.json_object
    joins = 0

    def joined_json(path: Path) -> dict[str, Any]:
        nonlocal joins
        document = original_json(path)
        if path.resolve() == hidden_engine_path:
            public = original_json(public_engine_path)
            if "systems" in document or not isinstance(public.get("systems"), dict):
                raise R14Error("R23 live systems repair precondition changed")
            document = {**document, "systems": public["systems"]}
            joins += 1
        return document

    base_live.json_object = joined_json
    try:
        receipt = base_live.run(live_path, output)
    finally:
        base_live.json_object = original_json
    if joins != 1:
        raise R14Error("R23 live systems metadata was not joined exactly once")
    binding = {
        "format": "abi-r23-live-repair-run/1",
        "repair_config_sha256": sha256_file(config_path),
        "live_receipt_sha256": sha256_file(output / "receipt.json"),
        "live_receipt_evidence_sha256": receipt["evidence_sha256"],
        "systems_joins": joins,
        "systems_source_sha256": sha256_file(public_engine_path),
        "candidate_changes": 0,
        "package_changes": 0,
        "scorer_changes": 0,
        "gate_changes": 0,
        "full_abi_moonshot": "OPEN",
    }
    binding["evidence_sha256"] = selfless_evidence_hash(binding)
    write_json_once(output / "repair_binding.json", binding)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.config, args.output)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
