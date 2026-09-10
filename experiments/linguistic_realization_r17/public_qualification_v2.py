"""Run the single evidence-driven R17 public source-interface repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .frames_v2 import public_rows_v2
from .public_qualification import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.model_id,
                args.revision,
                args.output,
                row_builder=public_rows_v2,
                protocol_name="PUBLIC_PROTOCOL_V2.md",
                interface_revision="v2",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
