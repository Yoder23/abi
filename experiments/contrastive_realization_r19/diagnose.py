"""Evaluate R19 contrastive inference on disclosed R17/R18 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.functional_realization_r18.run_public import _compiler_records, _rows
from experiments.functional_realization_r18.verify import _fill

from .compiler import infer_templates


def diagnose(source_rows: Path) -> dict[str, Any]:
    rows = _rows(source_rows)
    records = _compiler_records(rows)
    templates, diagnostics = infer_templates(records)
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    failed = []
    for row in evaluation:
        output = _fill(templates[str(row["signature"])], row["slots"])
        if output != row["expected"]:
            failed.append(
                {
                    "record_id": row["record_id"],
                    "signature": row["signature"],
                    "output": output,
                    "expected": row["expected"],
                }
            )
    return {
        "format": "abi-r19-contrastive-diagnostic/1",
        "source_rows": len(rows),
        "extraction_rows": len(records),
        "evaluation_rows": len(evaluation),
        "package_functional_exact": len(evaluation) - len(failed),
        "failed_rows": failed,
        "parseable_records": diagnostics["parseable_records"],
        "rejected_record_ids": diagnostics["rejected_record_ids"],
        "templates": templates,
        "support": diagnostics["support"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rows", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.source_rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
