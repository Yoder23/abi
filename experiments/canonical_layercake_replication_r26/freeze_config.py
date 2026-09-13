"""Bind the fresh R26 source result to the unchanged R25 implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.foreign_capability_r14.core import json_object, sha256_file

from .binding import CODE_PATHS


def _binding(root: Path, relative: str) -> dict[str, object]:
    target = root / relative
    return {
        "path": relative,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-strict", type=Path, required=True)
    parser.add_argument("--import-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source_config_relative = "experiments/factual_semantic_r16/configs/r26_fresh_v1.json"
    source_reveal_relative = "experiments/factual_semantic_r16/reveals/r26_fresh_v1.json"
    source_config = json_object(root / source_config_relative)
    protocol_commit = source_config["replication"]["implementation_freeze_commit"]
    config = {
        "format": "abi-r26-config/1",
        "protocol_freeze_commit": protocol_commit,
        "code_sha256": {
            relative: sha256_file(root / relative) for relative in CODE_PATHS
        },
        "r25_implementation_config": _binding(
            root, "experiments/canonical_layercake_import_r25/configs/public_v1.json"
        ),
        "r25_posthoc_repair": _binding(
            root, "results/canonical_layercake_import_r25/semantic_repair_v2.json"
        ),
        "source_config": _binding(root, source_config_relative),
        "source_reveal": _binding(root, source_reveal_relative),
        "source_receipt": _binding(
            root, (args.source_run / "receipt.json").as_posix()
        ),
        "source_strict_verification": _binding(root, args.source_strict.as_posix()),
        "import_config": _binding(root, args.import_config.as_posix()),
        "candidate_changes_from_r25": 0,
        "gate_changes_from_r25": 0,
        "prospective_scorer": "r16-normalized-text",
        "source_selection_generated_after_protocol_freeze": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R26 config")
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
