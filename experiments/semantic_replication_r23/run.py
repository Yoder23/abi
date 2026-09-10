"""Execute exact frozen R21 packages on the R23 semantic split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    sha256_file,
    write_json_once,
)
from experiments.generative_transfer_r21 import evaluate_hidden as base_evaluate
from experiments.generative_transfer_r21.hash_assurance_binding import (
    selfless_evidence_hash,
)

from .acquire import validate_source
from .binding import load_config, load_seed_reveal
from .protocol import hidden_rows, semantic_score


def run(
    config_path: Path,
    reveal_path: Path,
    source_run: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R23 result exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    reveal_path = reveal_path.resolve()
    source_run = source_run.resolve()
    output = output.resolve()
    if not output.is_relative_to(root):
        raise R14Error("R23 result must remain inside the ABI repository")
    config = load_config(root, config_path)
    load_seed_reveal(config, reveal_path)
    source_rows = validate_source(root, config_path, reveal_path, source_run)
    engine_dir = output / "engine"
    replacements = {
        "load_hidden_config": load_config,
        "load_seed_reveal": load_seed_reveal,
        "hidden_rows": hidden_rows,
        "validate_hidden_source": validate_source,
        "score_output": semantic_score,
    }
    originals = {name: getattr(base_evaluate, name) for name in replacements}
    for name, value in replacements.items():
        setattr(base_evaluate, name, value)
    try:
        engine = base_evaluate.run(
            config_path, reveal_path, source_run, engine_dir
        )
    finally:
        for name, value in originals.items():
            setattr(base_evaluate, name, value)
    result = {
        "format": "abi-r23-semantic-replication-result/1",
        "verdict": (
            "PASS_SEMANTIC_REPLICATION"
            if engine["verdict"] == "PASS_HIDDEN_REPLICATION"
            else "FAIL_SEMANTIC_REPLICATION"
        ),
        "scientific_claim": (
            "R23_BOUNDED_FRESH_SEMANTIC_SUPPLIED_CONTENT_TRANSFER"
            if engine["verdict"] == "PASS_HIDDEN_REPLICATION"
            else "R23_BOUNDED_SEMANTIC_REPLICATION_FAILED"
        ),
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_ABI_MOONSHOT",
        "config_sha256": sha256_file(config_path),
        "seed_reveal_sha256": sha256_file(reveal_path),
        "source_receipt_sha256": sha256_file(source_run / "receipt.json"),
        "source_rows_sha256": sha256_file(source_run / "source_observations.jsonl"),
        "engine_result": {
            "path": "engine/result.json",
            "sha256": sha256_file(engine_dir / "result.json"),
            "evidence_sha256": engine["evidence_sha256"],
        },
        "frozen_packages": len(config["packages"]),
        "student_training_steps": 0,
        "package_changes": 0,
        "source_rows_revalidated": len(source_rows),
        "full_abi_moonshot": "OPEN",
        "next_action": (
            "strict recomputation and bounded claim update"
            if engine["verdict"] == "PASS_HIDDEN_REPLICATION"
            else "preserve negative and select a materially different architecture"
        ),
    }
    result["evidence_sha256"] = selfless_evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed-reveal", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.config, args.seed_reveal, args.source_run, args.output)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
