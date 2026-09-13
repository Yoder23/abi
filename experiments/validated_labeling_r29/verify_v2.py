"""Strict R29 verifier using only the additive path-resolution wrapper."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file, write_json_once
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash

from . import verify as frozen_verify


def clean_import(config: Path, output: Path):
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.validated_labeling_r29.import_run_v2",
            "--config",
            str(config.resolve()),
            "--output",
            str(output.resolve()),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        from experiments.foreign_capability_r14.core import R14Error
        raise R14Error(f"clean R29 import subprocess failed: {completed.stderr[-2000:]}")


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
    frozen_verify.run_import = clean_import
    result = frozen_verify.verify(args.config, args.reveal, args.source_stored, args.import_config, args.import_stored, args.replay)
    root = Path(__file__).resolve().parents[2]
    result["path_resolution_repair"] = {
        "scope": "resolve paths and execute LayerCake in a clean subprocess",
        "frozen_import_sha256": sha256_file(root / "experiments/validated_labeling_r29/import_run.py"),
        "wrapper_sha256": sha256_file(root / "experiments/validated_labeling_r29/import_run_v2.py"),
        "verifier_sha256": sha256_file(root / "experiments/validated_labeling_r29/verify_v2.py"),
    }
    result["evidence_sha256"] = selfless_evidence_hash(result)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
