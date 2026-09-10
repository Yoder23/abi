"""Create the R16 held-out secret while printing only its commitment."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"immutable R16 reveal exists: {args.output}")
    secret = secrets.token_bytes(32)
    value = {
        "format": "abi-r16-heldout-reveal/1",
        "commitment": hashlib.sha256(secret).hexdigest(),
        "secret_hex": secret.hex(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
    print(value["commitment"])


if __name__ == "__main__":
    main()
