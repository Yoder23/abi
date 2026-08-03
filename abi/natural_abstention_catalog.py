"""Build disjoint search/validation probes for grounded English abstention.

The v1 gap catalog used an overly narrow lexical evaluator.  This successor
keeps the behavioral requirement unchanged (acknowledge that a requested
detail is absent from supplied fictional text) while accepting the ordinary
English ways a source can express that behavior.
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


CATALOG_CONTRACT = "abi-natural-abstention-catalog/1"
DEFAULT_SEED = 79_824

SEARCH_OPENINGS = (
    "Answer from the note alone. Say plainly when the requested detail is absent.",
    "Use only the supplied text; acknowledge any missing information.",
    "Do not guess. Explain briefly if the note lacks the answer.",
    "Give a grounded answer and identify when the detail is unavailable.",
    "Reply in one sentence using only what the fictional note states.",
    "State whether the requested fact can be determined from the note.",
    "Avoid assumptions and say when the text contains no answer.",
    "Be concise and honest about any information missing from the note.",
    "Answer solely from the fictional note, without filling in gaps.",
    "If the requested detail is not in the note, make that clear.",
)

VALIDATION_OPENINGS = (
    "Can the question be answered from this passage alone? Respond briefly.",
    "Ground the response only in the text and never invent an omitted detail.",
    "Report concisely whether the passage supplies the requested information.",
    "Use no outside knowledge; indicate if the evidence is insufficient.",
    "What careful answer is justified by the fictional passage?",
    "Respond without speculation and mention when the answer is unavailable.",
    "Determine whether the note actually states the requested detail.",
    "Give only an evidence-based answer from the supplied sentence.",
    "If the text leaves the question unresolved, say so directly.",
    "Answer faithfully from the note, explicitly recognizing missing evidence.",
)

ABSTENTION_MARKERS = (
    "not supplied",
    "not provided",
    "not given",
    "not specified",
    "not mentioned",
    "does not mention",
    "doesn't mention",
    "does not provide",
    "doesn't provide",
    "does not contain",
    "doesn't contain",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "cannot be known",
    "no information",
    "insufficient information",
    "unavailable",
    "unknown",
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


def _probe(
    *,
    split: str,
    opening_index: int,
    note_index: int,
    opening: str,
    note: str,
    missing: str,
    seed: int,
) -> dict[str, Any]:
    ordinal = opening_index * len(FICTIONAL_NOTES) + note_index
    probe: dict[str, Any] = {
        "probe_id": f"natural-abstention-{split}-{ordinal:03d}-v2",
        "destination_scope": "english_core",
        "capability": "abstention",
        "domain": "domain_independent",
        "split": split,
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
    return probe


def build_catalog(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    for split, openings, split_offset in (
        ("search", SEARCH_OPENINGS, 0),
        ("validation", VALIDATION_OPENINGS, 10_000),
    ):
        for opening_index, opening in enumerate(openings):
            for note_index, (note, missing) in enumerate(FICTIONAL_NOTES):
                probes.append(
                    _probe(
                        split=split,
                        opening_index=opening_index,
                        note_index=note_index,
                        opening=opening,
                        note=note,
                        missing=missing,
                        seed=seed + split_offset,
                    )
                )
    prompt_hashes = [
        hashlib.sha256(probe["prompt"].encode("utf-8")).hexdigest()
        for probe in probes
    ]
    if len(prompt_hashes) != len(set(prompt_hashes)):
        raise RuntimeError("natural abstention catalog contains duplicate prompts")
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "abi-natural-abstention-search-validation-v2",
        "catalog_contract_schema": CATALOG_CONTRACT,
        "status": "PREREGISTERED_SEARCH_VALIDATION_EVALUATOR_CORRECTION",
        "claim_boundary": (
            "This bounded catalog tests whether a response recognizes that a "
            "requested detail is absent from supplied fictional text. It does "
            "not test broad English fluency or prove zero latent knowledge."
        ),
        "generation": {
            "generator": "abi.natural_abstention_catalog",
            "seed": seed,
            "split_counts": {"search": 100, "validation": 100},
            "total_probes": 200,
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
