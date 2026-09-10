"""Bind R16 strict verification output to the exact replication artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    evidence_hash,
    sha256_file,
    write_json_once,
)

from .verify_strict_v3 import verify_v3


def verify_v4(
    config_path: Path,
    reveal_path: Path,
    run_dir: Path,
    live_path: Path,
    public_requalification: Path,
    source_inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = verify_v3(
        config_path,
        reveal_path,
        run_dir,
        live_path,
        public_requalification,
        source_inventory=source_inventory,
    )
    result = {
        **{key: value for key, value in base.items() if key != "evidence_sha256"},
        "format": "abi-r16-strict-verification/4",
        "verified_identity": {
            "config_sha256": sha256_file(config_path),
            "reveal_sha256": sha256_file(reveal_path),
            "run_receipt_sha256": sha256_file(run_dir / "receipt.json"),
            "live_receipt_sha256": sha256_file(live_path),
            "public_requalification_receipt_sha256": sha256_file(
                public_requalification / "receipt.json"
            ),
        },
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--public-requalification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_v4(
        args.config,
        args.reveal,
        args.run_dir,
        args.live,
        args.public_requalification,
    )
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
