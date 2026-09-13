"""Audit R59 output changes against the historical R55 raw matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.foreign_capability_r14.core import write_json_once


def _rows(path: Path) -> dict[str, dict]:
    return {
        row["probe_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r55-validation", type=Path, required=True)
    parser.add_argument("--r55-final", type=Path, required=True)
    parser.add_argument("--r59-validation", type=Path, required=True)
    parser.add_argument("--r59-final", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = {
        "validation": (args.r55_validation.resolve(), args.r59_validation.resolve()),
        "final_test": (args.r55_final.resolve(), args.r59_final.resolve()),
    }
    splits = {}
    for split, (before_path, after_path) in pairs.items():
        before = _rows(before_path)
        after = _rows(after_path)
        if set(before) != set(after) or len(before) != 1_400:
            raise RuntimeError("R59 parity matrix changed")
        changed = [key for key in sorted(before) if before[key]["output_sha256"] != after[key]["output_sha256"]]
        functional_changes = [key for key in changed if before[key]["functional_pass"] != after[key]["functional_pass"]]
        new_collapses = [key for key in after if after[key]["collapse"]["collapse_detected"]]
        changed_old_collapses = [key for key in changed if before[key]["collapse"]["collapse_detected"]]
        splits[split] = {
            "rows": len(before),
            "changed_outputs": len(changed),
            "changed_probe_ids": changed,
            "functional_verdict_changes": len(functional_changes),
            "new_collapses": len(new_collapses),
            "changed_historical_collapses": len(changed_old_collapses),
            "r55_raw_sha256": _sha(before_path),
            "r59_raw_sha256": _sha(after_path),
        }
    passed = (
        splits["validation"]["changed_outputs"] == 0
        and splits["final_test"]["changed_outputs"] == 12
        and splits["final_test"]["functional_verdict_changes"] == 0
        and splits["final_test"]["new_collapses"] == 0
    )
    receipt = {
        "format": "abi-r59-disclosed-parity-audit/1",
        "verdict": "PASS_R59_PARITY_AUDIT" if passed else "FAIL_R59_PARITY_AUDIT",
        "splits": splits,
        "finding": (
            "R55 postprocessed token re-encoding undercounted raw runtime token runs; "
            "R59 changed no functional verdict."
        ),
        "full_abi_moonshot": "OPEN",
    }
    if not passed:
        raise RuntimeError(json.dumps(receipt, sort_keys=True))
    write_json_once(args.output.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
