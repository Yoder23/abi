"""Fail-closed fresh replay of the R26 direct-import replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.canonical_layercake_import_r25.verify import _rows, _validate_packages
from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
)
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash

from .binding import load_config
from .run import NAMES, corrected_metrics, run


def verify(config_path: Path, stored_dir: Path, replay_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    stored_dir = stored_dir.resolve()
    replay_dir = replay_dir.resolve()
    config = load_config(root, config_path)
    stored = json_object(stored_dir / "result.json")
    if (
        stored.get("format") != "abi-r26-canonical-import-replication-result/1"
        or stored.get("config_sha256") != sha256_file(config_path)
        or stored.get("evidence_sha256") != selfless_evidence_hash(stored)
    ):
        raise R14Error("R26 stored result identity changed")
    import_config_path = root / config["import_config"]["path"]
    import_config = json.loads(import_config_path.read_text(encoding="utf-8"))
    stored_engine_dir = stored_dir / "engine"
    stored_engine = json_object(stored_engine_dir / "result.json")
    stored_rows = {
        name: _rows(stored_engine_dir, stored_engine, name) for name in NAMES
    }
    packages = _validate_packages(root, import_config, stored_engine)
    run(config_path, replay_dir)
    replay_engine_dir = replay_dir / "engine"
    replay_engine = json_object(replay_engine_dir / "result.json")
    replay_rows = {
        name: _rows(replay_engine_dir, replay_engine, name) for name in NAMES
    }
    for name in NAMES[1:]:
        if stored_rows[name] != replay_rows[name]:
            raise R14Error(f"R26 fresh live replay changed: {name}")
    for left, right in zip(
        stored_rows["package_builds"], replay_rows["package_builds"], strict=True
    ):
        if {key: value for key, value in left.items() if key != "path"} != {
            key: value for key, value in right.items() if key != "path"
        }:
            raise R14Error("R26 fresh package rebuild changed")
    metrics, gates = corrected_metrics(
        root, import_config, stored_engine_dir, stored_engine
    )
    if (
        stored.get("metrics") != metrics
        or stored.get("gates") != gates
        or not all(gates.values())
        or stored.get("verdict") != "PASS_FRESH_BOUNDED_DIRECT_IMPORT"
        or stored.get("claim")
        != "R26_FRESH_REGISTERED_FACTUAL_DIRECT_IMPORT_AND_COMPOSITION"
        or stored.get("full_abi_moonshot") != "OPEN"
        or packages != 6
    ):
        raise R14Error("R26 strict scientific gate failed")
    result = {
        "format": "abi-r26-strict-live-verification/1",
        "status": "PASS_STRICTLY_VERIFIED_FRESH_BOUNDED_DIRECT_IMPORT",
        "scientific_claim": stored["claim"],
        "claim_ceiling": stored["claim_ceiling"],
        "config_sha256": sha256_file(config_path),
        "stored_result_sha256": sha256_file(stored_dir / "result.json"),
        "replay_result_sha256": sha256_file(replay_dir / "result.json"),
        "source_receipt_sha256": stored["source_receipt_sha256"],
        "packages_recomputed": packages,
        "rows_live_recomputed": sum(len(rows) for rows in replay_rows.values()),
        "package_archives_rebuilt": 6,
        "stored_scientific_booleans_trusted": False,
        "metrics": metrics,
        "gates": gates,
        "candidate_changes_from_r25": 0,
        "gate_changes_from_r25": 0,
        "source_parameters_copied": 0,
        "receiver_training_steps": 0,
        "teacher_present_at_execution": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = selfless_evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stored", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.config, args.stored, args.replay)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
