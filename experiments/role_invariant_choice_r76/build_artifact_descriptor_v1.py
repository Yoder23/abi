"""Build the immutable R76 selective-artifact descriptor from sealed evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.conditional_choice_artifact_v2 import DESCRIPTOR_SCHEMA, _canonical_sha, validate_descriptor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable descriptor exists: {args.output}")
    value = {
        "schema_version": DESCRIPTOR_SCHEMA,
        "campaign": "r76",
        "catalog_sha256": "b7bf96a48d2c51fc08948ff1f92a5211c22fd24e0f83d5749a61523c22ee2387",
        "result_file_sha256": "f01dced22dc3a4469d41677f178e8735e19e0d6e16d301da4fc1342b30771d10",
        "raw_scores_sha256": "7866ecba2026be86c1840efb3b236fb4953d4cb19a2288290a40eaaa21100d42",
        "result_evidence_sha256": "841a1a7205d228543fd3416781660312c379cc2d1b587faf13aa5b290176276f",
        "source_archive_sha256": "82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150",
        "source_manifest_sha256": "3bac528a1825e77dcb35963f5c78946fb14400e1cd832e0db40c2d964360c310",
        "source_rows": 8_400,
        "minimum_packaged_rows": 8_000,
        "minimum_packaged_rows_per_family": 900,
        "result_format": "abi-role-invariant-conditional-choice-source-score/1",
        "accepted_result_verdicts": ["FAIL_R76_SOURCE"],
        "allow_selective_packaging": True,
        "scientific_interpretation": (
            "R76 failed its broad family floor. This descriptor authorizes only the "
            "8,129 independently correct source selections as disclosed development "
            "training material; it does not reinterpret R76 as a source pass."
        ),
    }
    value["descriptor_sha256"] = _canonical_sha(value)
    validate_descriptor(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
