"""Generate an unrevealed R13 secret and print only its SHA-256 commitment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to replace held-out secret: {output}")
    secret = os.urandom(32)
    commitment = hashlib.sha256(secret).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        json.dumps(
            {
                "format": "abi-r13-heldout-reveal/1",
                "secret_hex": secret.hex(),
                "commitment": commitment,
            },
            indent=2,
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    print(json.dumps({"commitment": commitment, "secret_printed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
