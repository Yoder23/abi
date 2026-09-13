"""Strict R29 verifier using only the additive path-resolution wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file, write_json_once
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash

from . import verify as frozen_verify
from .import_run_v2 import run as resolved_import


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--source-stored", type=Path, required=True)
    parser.add_argument("--import-config", type=Path, required=True)
    parser.add_argument("--import-stored", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen_verify.run_import = resolved_import
    result = frozen_verify.verify(args.config, args.reveal, args.source_stored, args.import_config, args.import_stored, args.replay)
    root = Path(__file__).resolve().parents[2]
    result["path_resolution_repair"] = {
        "scope": "resolve config and output paths only",
        "frozen_import_sha256": sha256_file(root / "experiments/validated_labeling_r29/import_run.py"),
        "wrapper_sha256": sha256_file(root / "experiments/validated_labeling_r29/import_run_v2.py"),
        "verifier_sha256": sha256_file(root / "experiments/validated_labeling_r29/verify_v2.py"),
    }
    result["evidence_sha256"] = selfless_evidence_hash(result)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
