"""Create the private R26 source reveal and public commitment config."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    args = parser.parse_args()
    if args.config.exists() or args.reveal.exists():
        raise RuntimeError("refusing to overwrite R26 source commitment")
    root = Path(__file__).resolve().parents[2]
    template_path = root / "experiments/factual_semantic_r16/configs/heldout_v2.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    secret = secrets.token_bytes(32)
    reveal = {
        "format": "abi-r16-heldout-reveal/1",
        "commitment": hashlib.sha256(secret).hexdigest(),
        "secret_hex": secret.hex(),
    }
    args.reveal.parent.mkdir(parents=True, exist_ok=True)
    args.reveal.write_text(
        json.dumps(reveal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config = {
        **template,
        "heldout_seed_commitment": reveal["commitment"],
        "reveal_sha256": sha256_file(args.reveal),
        "replication": {
            "implementation_freeze_commit": commit,
            "purpose": "R26 fresh prospective canonical LayerCake import replication",
            "same_metrics_and_gates_as": (
                "experiments/factual_semantic_r16/configs/heldout_v2.json"
            ),
        },
    }
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(reveal["commitment"])


if __name__ == "__main__":
    main()
