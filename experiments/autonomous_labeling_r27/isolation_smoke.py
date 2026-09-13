"""Disclosed physical-isolation smoke test for the R27 compiler."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from experiments.foreign_capability_r14.core import evidence_hash, write_json_once

from .isolation import run_isolated


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="r27-smoke-") as raw:
        directory = Path(raw)
        records = [
            {"subject": "helium", "question": f"question {view}", "view": view, "answer": "2", "label": "chemistry"}
            for view in range(3)
        ]
        bundle = {"format": "abi-r27-anonymous-free-label-observations/1", "records": records}
        bundle["evidence_sha256"] = evidence_hash(bundle)
        path = directory / "bundle.json"; write_json_once(path, bundle)
        result = run_isolated(root, path, directory / "extraction")
        value = result["result"]
        if value["records_consumed"] != 3 or value["oracle_fields_consumed"] != 0 or len(value["packages"]) != 1:
            raise RuntimeError("R27 physical isolation smoke failed")
        print(json.dumps(value, indent=2))


if __name__ == "__main__": main()
