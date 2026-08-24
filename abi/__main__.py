"""Command-line entry point for the supported ABI release surface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from . import __version__
from .capability_pipeline import build_source_model_manifest
from .capability_segregation import (
    LINGUISTIC_FORM,
    build_segregated_extraction_record,
    validate_segregated_extraction_record,
)

STATUS: dict[str, Any] = {
    "project": "ABI",
    "version": __version__,
    "maturity": "alpha_research",
    "campaign_state": "V1089",
    "phase8_certified": False,
    "release_certified": False,
    "external_human_preferences": {"complete": 0, "required": 21_000},
    "supported_boundary": (
        "source manifests, bounded capability inventories, semantic labeling, "
        "quarantine, selection, accounting, and immutable acquisition artifacts"
    ),
}


def _self_check() -> dict[str, Any]:
    revision = "a" * 40
    source = build_source_model_manifest(
        model_id="self-check/teacher",
        revision=revision,
        revision_is_immutable=True,
        architecture="SelfCheckForCausalLM",
        parameter_count=1,
        tokenizer_id="self-check/tokenizer",
        tokenizer_revision=revision,
        license_id="test-only",
        weight_files=[
            {
                "relative_path": "model.safetensors",
                "sha256": "0" * 64,
                "bytes": 1,
            }
        ],
    )
    record = build_segregated_extraction_record(
        destination_scope="english_core",
        capability="rewriting",
        domain="domain_independent",
        provenance="self-check:rewrite-1",
        split="validation",
        source_model=source["model_id"],
        source_model_revision=source["revision"],
        prompt="Rewrite this politely: Send the file.",
        output="Could you please send the file?",
        teacher_tokens=7,
        teacher_token_counter="authoritative_generated_token_ids",
        knowledge_class=LINGUISTIC_FORM,
        content_basis="domain_free_instruction",
        domain_labels=[],
        domain_claims=[],
        label_method="human_review",
        label_evidence_sha256="1" * 64,
        output_introduces_unsupplied_facts=False,
    )
    validate_segregated_extraction_record(record)
    return {
        "status": "PASS",
        "source_manifest_sha256": source["source_manifest_sha256"],
        "record_id": record["record_id"],
        "english_domain_labels": record["domain_labels"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abi",
        description="Inspect and validate the supported ABI release surface.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status", help="print the current claim boundary")
    status.add_argument("--json", action="store_true", help="emit JSON")
    subparsers.add_parser("self-check", help="run a dependency-light integrity check")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "status":
        if args.json:
            print(json.dumps(STATUS, indent=2, sort_keys=True))
        else:
            print(f"ABI {__version__} ({STATUS['maturity']})")
            print("Phase 8 certified: no")
            print("Release certified: no")
            print(f"Supported boundary: {STATUS['supported_boundary']}")
        return 0
    result = _self_check()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
