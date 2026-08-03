"""Build fresh validation-only probes for comprehensive abstention wording.

V2 development evidence showed that valid grounded abstention spans more
ordinary English constructions than its lexical evaluator admitted.  This
catalog broadens that evaluator once and uses new prompts so the observations
that motivated the correction cannot qualify the source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .broad_english_catalog import DEFAULT_CORE_EXCLUSION_MARKERS
from .capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    load_probe_catalog,
    probe_label_evidence_sha256,
)
from .natural_gap_catalog import FICTIONAL_NOTES


CATALOG_CONTRACT = "abi-natural-abstention-catalog/2"
DEFAULT_SEED = 89_824

VALIDATION_OPENINGS = (
    "Using just the fictional sentence, is the requested detail stated? Answer briefly.",
    "Does this note supply enough evidence to answer the question? Be concise.",
    "Give the most careful response licensed by the passage and nothing external.",
    "Say directly whether the requested information appears in the supplied text.",
    "Without assuming anything beyond the note, answer the question in one sentence.",
    "Judge only from this passage and explicitly recognize an omitted answer.",
    "Provide a concise text-grounded response, with no invented detail.",
    "Is the requested detail recoverable from the note alone? Respond honestly.",
    "Read the sentence literally and explain briefly if it leaves the question open.",
    "Answer only to the extent supported by the fictional evidence.",
)

ABSTENTION_MARKERS = (
    "not supplied",
    "not provided",
    "not given",
    "not specified",
    "not mentioned",
    "not stated",
    "not available",
    "does not mention",
    "doesn't mention",
    "does not provide",
    "doesn't provide",
    "does not contain",
    "doesn't contain",
    "does not state",
    "doesn't state",
    "does not supply",
    "doesn't supply",
    "does not include",
    "doesn't include",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "cannot be known",
    "cannot be answered",
    "can't be answered",
    "no information",
    "no evidence-based answer",
    "insufficient information",
    "insufficient evidence",
    "insufficient to determine",
    "unavailable",
    "unknown",
    "absent",
    "omitted",
    "lacks the",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluator() -> dict[str, Any]:
    return {
        "kind": "all_of",
        "rules": [
            {"kind": "nonempty", "minimum_characters": 8},
            {"kind": "maximum_characters", "value": 600},
            {
                "kind": "contains_none",
                "values": list(DEFAULT_CORE_EXCLUSION_MARKERS),
            },
            {"kind": "contains_any", "values": list(ABSTENTION_MARKERS)},
        ],
    }


def build_catalog(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    for opening_index, opening in enumerate(VALIDATION_OPENINGS):
        for note_index, (note, missing) in enumerate(FICTIONAL_NOTES):
            ordinal = opening_index * len(FICTIONAL_NOTES) + note_index
            probe: dict[str, Any] = {
                "probe_id": f"natural-abstention-validation-{ordinal:03d}-v3",
                "destination_scope": "english_core",
                "capability": "abstention",
                "domain": "domain_independent",
                "split": "validation",
                "prompt": (
                    f'Fictional note: "{note}"\n'
                    f"Question: What is the {missing}?\n{opening}"
                ),
                "max_new_tokens": 80,
                "temperature": 0,
                "seed": seed + ordinal,
                "evaluator": _evaluator(),
                "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": LINGUISTIC_FORM,
                "content_basis": "domain_free_instruction",
                "domain_labels": [],
                "domain_claims": [],
                "label_method": "preregistered_catalog",
                "output_introduces_unsupplied_facts": False,
            }
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            probes.append(probe)
    prompt_hashes = [
        hashlib.sha256(probe["prompt"].encode("utf-8")).hexdigest()
        for probe in probes
    ]
    if len(prompt_hashes) != len(set(prompt_hashes)):
        raise RuntimeError("v3 abstention catalog contains duplicate prompts")
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "abi-natural-abstention-validation-v3",
        "catalog_contract_schema": CATALOG_CONTRACT,
        "status": "PREREGISTERED_FRESH_VALIDATION_AFTER_V2_EVALUATOR_DEVELOPMENT",
        "claim_boundary": (
            "This bounded catalog tests recognition of information absent from "
            "supplied fictional text. V1 and V2 results remain unchanged."
        ),
        "generation": {
            "generator": "abi.natural_abstention_catalog_v3",
            "seed": seed,
            "split_counts": {"validation": 100},
            "total_probes": 100,
            "closed_book_world_fact_prompts": 0,
            "specialist_domain_prompts": 0,
            "prompt_sha256_set_sha256": hashlib.sha256(
                "\n".join(sorted(prompt_hashes)).encode("ascii")
            ).hexdigest(),
        },
        "core_exclusion_markers": list(DEFAULT_CORE_EXCLUSION_MARKERS),
        "probes": probes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"catalog is immutable: {output}")
    catalog = build_catalog(seed=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    load_probe_catalog(output)
    print(
        json.dumps(
            {
                "catalog_id": catalog["catalog_id"],
                "output": str(output),
                "probes": len(catalog["probes"]),
                "sha256": _sha256_file(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
