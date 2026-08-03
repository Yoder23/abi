"""Build the three capability-completion rows absent from the natural corpus.

The main natural-instruction catalog intentionally rejects closed-book and
underspecified prompts, so it cannot supply enough clarification, abstention,
or abstract-reasoning rows.  This small deterministic supplement uses only
fictional/nonce content and varied natural phrasings.  It completes acquisition
coverage without importing specialist facts or corpus reference answers.
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


CATALOG_CONTRACT = "abi-natural-instruction-gap-catalog/1"
DEFAULT_SEED = 69_824

CLARIFICATION_OPENINGS = (
    "Before acting, ask one brief question that would resolve the ambiguity.",
    "Reply only with the single most useful clarification question.",
    "Do not guess; ask one concise follow-up question.",
    "What one question should be answered before this request is carried out?",
    "Respond with a natural clarification question and nothing else.",
    "Ask for the missing detail in one short sentence.",
    "State one polite question that would make the request actionable.",
    "Identify the ambiguity by asking one direct follow-up question.",
    "Seek clarification instead of choosing an interpretation.",
    "Write the shortest courteous question that resolves what is missing.",
)
AMBIGUOUS_REQUESTS = (
    "Please send the draft to them when it is ready.",
    "Move the meeting to the other room.",
    "Use the newer version for the review.",
    "Leave the package beside the usual door.",
    "Tell the group that the date has changed.",
    "Replace the item with the preferred one.",
    "Add the note to the correct folder.",
    "Book the visit for the better time.",
    "Share the summary with the appropriate person.",
    "Update the message before the next session.",
)

ABSTENTION_OPENINGS = (
    "Answer from the note alone. If the detail is absent, say so briefly.",
    "Use only the supplied text and do not invent the missing detail.",
    "Can this be known from the note? Give a concise answer.",
    "State whether the requested detail is supplied, without guessing.",
    "Reply in one sentence and explicitly acknowledge missing information.",
    "Do not rely on outside knowledge; answer only from the note.",
    "If the note is insufficient, say that it cannot be determined.",
    "Give a careful answer grounded solely in the fictional note.",
    "Avoid assumptions and explain briefly when the answer is unavailable.",
    "Respond honestly about whether the note contains the requested fact.",
)
FICTIONAL_NOTES = (
    ("Mira placed the silver key inside a wooden box.", "exact weight"),
    ("Tovan left the folded scarf beside the quiet window.", "serial number"),
    ("Nela carried a blue cup into the garden.", "purchase date"),
    ("Orin moved the small lamp onto the round table.", "manufacturer"),
    ("Savi found a paper kite near the old gate.", "exact price"),
    ("Luma put a green book beneath the chair.", "page count"),
    ("Peren took the red bag into the empty hall.", "owner's age"),
    ("Vela set a glass jar beside the bench.", "factory location"),
    ("Romi brought a soft blanket to the porch.", "model number"),
    ("Kavi stored a brass bell inside the cabinet.", "warranty length"),
)

REASONING_OPENINGS = (
    "Reason only from these invented rules and explain the conclusion briefly.",
    "Using the nonce rules alone, what must follow?",
    "Treat every term as fictional. Derive the supported conclusion.",
    "Do not use outside facts; apply the stated relations step by step.",
    "Which conclusion is licensed by the supplied imaginary rules?",
    "Follow the invented category chain and answer in one sentence.",
    "Infer only what is guaranteed by the fictional premises.",
    "Apply the two nonce rules to the named item and state the result.",
    "Give the shortest valid deduction from these made-up statements.",
    "From the supplied relations alone, identify what is necessarily true.",
)
NONCE_RULES = (
    "Every zorin is a pel. Every pel is a naku. Luma is a zorin.",
    "All mavens are tiri. Every tiri is a sorn. Kavi is a maven.",
    "Each belu is a rax. All rax are feni. Tovan is a belu.",
    "Every cordan is a vesh. Each vesh is a lomi. Nela is a cordan.",
    "All prils are jotas. Every jota is a wex. Mira is a pril.",
    "Each daven is a quor. All quors are simi. Orin is a daven.",
    "Every halen is a brix. Each brix is a noro. Savi is a halen.",
    "All yoris are kepa. Every kepa is a tul. Vela is a yori.",
    "Each fenor is a gavi. All gavi are pex. Romi is a fenor.",
    "Every sulen is a tari. Each tari is a miv. Peren is a sulen.",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluator(capability: str) -> dict[str, Any]:
    rules: list[dict[str, Any]] = [
        {"kind": "nonempty", "minimum_characters": 8},
        {"kind": "maximum_characters", "value": 600},
        {
            "kind": "contains_none",
            "values": list(DEFAULT_CORE_EXCLUSION_MARKERS),
        },
    ]
    if capability == "clarification":
        rules.append({"kind": "contains_all", "values": ["?"]})
    elif capability == "abstention":
        rules.append(
            {
                "kind": "contains_any",
                "values": [
                    "not supplied",
                    "not provided",
                    "not given",
                    "not specified",
                    "cannot determine",
                    "can't determine",
                    "cannot be known",
                    "unknown",
                ],
            }
        )
    return {"kind": "all_of", "rules": rules}


def _probe(
    *,
    capability: str,
    ordinal: int,
    prompt: str,
    seed: int,
) -> dict[str, Any]:
    probe: dict[str, Any] = {
        "probe_id": f"natural-gap-search-{capability}-{ordinal:03d}-v1",
        "destination_scope": "english_core",
        "capability": capability,
        "domain": "domain_independent",
        "split": "search",
        "prompt": prompt,
        "max_new_tokens": 80 if capability != "domain_independent_reasoning" else 112,
        "temperature": 0,
        "seed": seed + ordinal,
        "evaluator": _evaluator(capability),
        "record_schema": SEGREGATED_RECORD_SCHEMA,
        "knowledge_class": LINGUISTIC_FORM,
        "content_basis": (
            "abstract_or_nonce_content"
            if capability == "domain_independent_reasoning"
            else "domain_free_instruction"
        ),
        "domain_labels": [],
        "domain_claims": [],
        "label_method": "preregistered_catalog",
        "output_introduces_unsupplied_facts": False,
    }
    probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
    return probe


def build_catalog(*, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    capability_offsets = {
        "clarification": 0,
        "abstention": 10_000,
        "domain_independent_reasoning": 20_000,
    }
    for opening_index, opening in enumerate(CLARIFICATION_OPENINGS):
        for request_index, request in enumerate(AMBIGUOUS_REQUESTS):
            ordinal = opening_index * len(AMBIGUOUS_REQUESTS) + request_index
            probes.append(
                _probe(
                    capability="clarification",
                    ordinal=ordinal,
                    prompt=f"Ambiguous request: \"{request}\"\n{opening}",
                    seed=seed + capability_offsets["clarification"],
                )
            )
    for opening_index, opening in enumerate(ABSTENTION_OPENINGS):
        for note_index, (note, missing) in enumerate(FICTIONAL_NOTES):
            ordinal = opening_index * len(FICTIONAL_NOTES) + note_index
            probes.append(
                _probe(
                    capability="abstention",
                    ordinal=ordinal,
                    prompt=(
                        f"Fictional note: \"{note}\"\n"
                        f"Question: What is the {missing}?\n{opening}"
                    ),
                    seed=seed + capability_offsets["abstention"],
                )
            )
    for opening_index, opening in enumerate(REASONING_OPENINGS):
        for rule_index, rules in enumerate(NONCE_RULES):
            ordinal = opening_index * len(NONCE_RULES) + rule_index
            probes.append(
                _probe(
                    capability="domain_independent_reasoning",
                    ordinal=ordinal,
                    prompt=f"Invented premises: {rules}\n{opening}",
                    seed=seed
                    + capability_offsets["domain_independent_reasoning"],
                )
            )
    prompt_hashes = [
        hashlib.sha256(probe["prompt"].encode("utf-8")).hexdigest()
        for probe in probes
    ]
    if len(prompt_hashes) != len(set(prompt_hashes)):
        raise RuntimeError("natural gap catalog contains duplicate prompts")
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "abi-natural-instruction-gap-search-v1",
        "catalog_contract_schema": CATALOG_CONTRACT,
        "status": "PREREGISTERED_SEARCH_ONLY_CAPABILITY_COMPLETION",
        "claim_boundary": (
            "This finite supplement fills three capabilities absent from the "
            "natural prompt corpus with varied fictional or nonce prompts. It "
            "does not prove broad English fluency or zero latent knowledge."
        ),
        "generation": {
            "generator": "abi.natural_gap_catalog",
            "seed": seed,
            "split": "search",
            "capability_counts": {
                "clarification": 100,
                "abstention": 100,
                "domain_independent_reasoning": 100,
            },
            "total_probes": 300,
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
