"""Bind the expanded R16 hostile audit to one exact replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once

from .hostile_audit_v3 import audit_v3
from .verify_strict_v4 import verify_v4


def audit_v4(
    config: Path,
    reveal: Path,
    run_dir: Path,
    live: Path,
    public_requalification: Path,
) -> dict[str, Any]:
    base = audit_v3(config, reveal, run_dir, live, public_requalification)
    strict = verify_v4(config, reveal, run_dir, live, public_requalification)
    result = {
        **{key: value for key, value in base.items() if key != "evidence_sha256"},
        "format": "abi-r16-hostile-audit/4",
        "base_audit_evidence_sha256": base["evidence_sha256"],
        "strict_v4_evidence_sha256": strict["evidence_sha256"],
        "verified_identity": {
            "config_sha256": sha256_file(config),
            "reveal_sha256": sha256_file(reveal),
            "run_receipt_sha256": sha256_file(run_dir / "receipt.json"),
            "live_receipt_sha256": sha256_file(live),
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
    result = audit_v4(
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
