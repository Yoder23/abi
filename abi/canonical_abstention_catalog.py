"""Build a fresh exact-behavior validation for grounded abstention.

This is a separate behavioral protocol from the closed lexical classifiers in
natural abstention v1-v3.  Every supplied fictional note omits the requested
detail, and the source must follow an explicit canonical response contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    load_probe_catalog,
    probe_label_evidence_sha256,
)
from .natural_gap_catalog import FICTIONAL_NOTES


CATALOG_CONTRACT = "abi-canonical-abstention-catalog/1"
DEFAULT_SEED = 99_824
CANONICAL_RESPONSE = "INSUFFICIENT INFORMATION"

VALIDATION_OPENINGS = (
    "If the answer is absent, reply with exactly INSUFFICIENT INFORMATION and no other text.",
    "When the note omits the detail, output only INSUFFICIENT INFORMATION.",
    "Use the exact response INSUFFICIENT INFORMATION if the passage does not answer the question.",
    "For a missing answer, write INSUFFICIENT INFORMATION exactly, without punctuation.",
    "If the evidence is missing, your entire reply must be INSUFFICIENT INFORMATION.",
    "Return only INSUFFICIENT INFORMATION when the requested detail is not stated.",
    "Respond with exactly INSUFFICIENT INFORMATION if the note is inadequate.",
    "When the question cannot be answered from the note, say only INSUFFICIENT INFORMATION.",
    "If the passage lacks the answer, provide the exact string INSUFFICIENT INFORMATION.",
    "Output INSUFFICIENT INFORMATION alone whenever the supplied sentence omits the detail.",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_catalog(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    for opening_index, opening in enumerate(VALIDATION_OPENINGS):
        for note_index, (note, missing) in enumerate(FICTIONAL_NOTES):
            ordinal = opening_index * len(FICTIONAL_NOTES) + note_index
            probe: dict[str, Any] = {
                "probe_id": f"canonical-abstention-validation-{ordinal:03d}-v1",
                "destination_scope": "english_core",
                "capability": "abstention",
                "domain": "domain_independent",
                "split": "validation",
                "prompt": (
                    f'Fictional note: "{note}"\n'
                    f"Question: What is the {missing}?\n{opening}"
                ),
                "max_new_tokens": 16,
                "temperature": 0,
                "seed": seed + ordinal,
                "evaluator": {
                    "kind": "exact",
                    "value": CANONICAL_RESPONSE,
                },
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
        raise RuntimeError("canonical abstention catalog contains duplicate prompts")
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "abi-canonical-abstention-validation-v1",
        "catalog_contract_schema": CATALOG_CONTRACT,
        "status": "PREREGISTERED_FRESH_EXACT_BEHAVIOR_VALIDATION",
        "claim_boundary": (
            "This catalog qualifies exact adherence to a canonical abstention "
            "contract on bounded fictional notes. It does not erase the failed "
            "v1-v3 natural-language lexical evaluators."
        ),
        "generation": {
            "generator": "abi.canonical_abstention_catalog",
            "seed": seed,
            "split_counts": {"validation": 100},
            "total_probes": 100,
            "closed_book_world_fact_prompts": 0,
            "specialist_domain_prompts": 0,
            "prompt_sha256_set_sha256": hashlib.sha256(
                "\n".join(sorted(prompt_hashes)).encode("ascii")
            ).hexdigest(),
        },
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
