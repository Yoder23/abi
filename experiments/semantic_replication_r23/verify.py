"""Strict recomputation of the R23 semantic replication."""

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
from experiments.generative_transfer_r21 import verify_hidden as base_verify
from experiments.generative_transfer_r21.hash_assurance_binding import (
    selfless_evidence_hash,
)

from .acquire import validate_source
from .binding import load_config, load_seed_reveal
from .protocol import hidden_rows, semantic_score


def verify(
    config_path: Path,
    reveal_path: Path,
    source_run: Path,
    result_dir: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    reveal_path = reveal_path.resolve()
    source_run = source_run.resolve()
    result_dir = result_dir.resolve()
    load_config(root, config_path)
    source_rows = validate_source(root, config_path, reveal_path, source_run)
    wrapper = json_object(result_dir / "result.json")
    engine_dir = result_dir / "engine"
    engine = json_object(engine_dir / "result.json")
    if (
        wrapper.get("format") != "abi-r23-semantic-replication-result/1"
        or wrapper.get("config_sha256") != sha256_file(config_path)
        or wrapper.get("seed_reveal_sha256") != sha256_file(reveal_path)
        or wrapper.get("source_receipt_sha256")
        != sha256_file(source_run / "receipt.json")
        or wrapper.get("source_rows_sha256")
        != sha256_file(source_run / "source_observations.jsonl")
        or wrapper.get("engine_result", {}).get("sha256")
        != sha256_file(engine_dir / "result.json")
        or wrapper.get("engine_result", {}).get("evidence_sha256")
        != engine.get("evidence_sha256")
        or wrapper.get("evidence_sha256") != selfless_evidence_hash(wrapper)
        or wrapper.get("student_training_steps") != 0
        or wrapper.get("package_changes") != 0
        or wrapper.get("frozen_packages") != 24
    ):
        raise R14Error("R23 result wrapper failed")
    replacements = {
        "load_hidden_config": load_config,
        "load_seed_reveal": load_seed_reveal,
        "hidden_rows": hidden_rows,
        "validate_hidden_source": validate_source,
        "score_output": semantic_score,
    }
    originals = {name: getattr(base_verify, name) for name in replacements}
    for name, value in replacements.items():
        setattr(base_verify, name, value)
    try:
        underlying = base_verify.verify(
            config_path, reveal_path, source_run, engine_dir
        )
    finally:
        for name, value in originals.items():
            setattr(base_verify, name, value)
    passed = underlying["scientific_verdict"] == "PASS_HIDDEN_REPLICATION"
    expected_verdict = (
        "PASS_SEMANTIC_REPLICATION" if passed else "FAIL_SEMANTIC_REPLICATION"
    )
    if wrapper.get("verdict") != expected_verdict:
        raise R14Error("R23 wrapper verdict changed")
    result = {
        "format": "abi-r23-strict-verification/1",
        "status": (
            "PASS_VERIFIED_SEMANTIC_REPLICATION"
            if passed
            else "PASS_VERIFIED_NEGATIVE_SEMANTIC_REPLICATION"
        ),
        "scientific_verdict": wrapper["verdict"],
        "config_sha256": sha256_file(config_path),
        "result_sha256": sha256_file(result_dir / "result.json"),
        "engine_result_sha256": sha256_file(engine_dir / "result.json"),
        "source_receipt_sha256": sha256_file(source_run / "receipt.json"),
        "source_rows_recomputed": len(source_rows),
        "rows_recomputed": underlying["rows_recomputed"],
        "cpu_rows_recomputed": underlying["cpu_rows_recomputed"],
        "packages_rebound": underlying["packages_rebound"],
        "gates": underlying["gates"],
        "underlying_verification": underlying,
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_ABI_MOONSHOT",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = selfless_evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed-reveal", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.config, args.seed_reveal, args.source_run, args.result)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
