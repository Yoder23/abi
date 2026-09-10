"""Diagnose source-derived modal templates on a completed R17 public run."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .isolated_worker import IsolatedR17Error, _template
from .package import PACKAGE_FORMAT, realize


def diagnose(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    grouped: dict[str, list[str]] = defaultdict(list)
    unparseable: list[str] = []
    for row in rows:
        if row["split"] == "extraction":
            try:
                learned = _template(str(row["output"]), row["slots"])
            except IsolatedR17Error:
                unparseable.append(str(row["record_id"]))
            else:
                grouped[str(row["signature"])].append(learned)
    counts = {signature: Counter(values) for signature, values in grouped.items()}
    templates = {
        signature: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for signature, counter in counts.items()
    }
    package = {
        "format": PACKAGE_FORMAT,
        "namespace": "english/core/realization",
        "templates": templates,
    }
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    result = {
        "format": "abi-r17-consensus-diagnostic/1",
        "source_rows": len(rows),
        "signatures": len(grouped),
        "unparseable_extraction_rows": unparseable,
        "signatures_with_multiple_templates": sum(len(counter) > 1 for counter in counts.values()),
        "package_oracle_exact": sum(
            realize(package, row["signature"], row["slots"]) == row["expected"]
            for row in evaluation
        ),
        "package_source_exact": sum(
            realize(package, row["signature"], row["slots"]) == row["output"] for row in evaluation
        ),
        "evaluation_rows": len(evaluation),
        "ambiguous_templates": {
            signature: dict(counter)
            for signature, counter in sorted(counts.items())
            if len(counter) > 1
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rows", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.source_rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
