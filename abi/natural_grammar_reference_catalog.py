"""Build a broad, natural, domain-free grammar acquisition catalog.

The catalog isolates subject-verb agreement across sixteen ordinary English
structures.  Search and validation sentences are distinct, every expected
correction is preregistered, and the exact evaluator is bound to the raw
prompt.  These are ABI source-acquisition probes, not LayerCake runtime rules.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from .capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    load_probe_catalog,
    probe_label_evidence_sha256,
    prompt_contract_sha256,
)


CATALOG_ID = "abi-natural-grammar-reference-search-validation-v1"
SEARCH_PER_STRUCTURE = 10
VALIDATION_PER_STRUCTURE = 4

_ADJECTIVES = (
    "careful",
    "cheerful",
    "calm",
    "curious",
    "eager",
    "gentle",
    "helpful",
    "patient",
    "quiet",
    "steady",
    "thoughtful",
    "watchful",
    "friendly",
    "polite",
    "brisk",
    "focused",
    "attentive",
    "kind",
    "lively",
    "orderly",
    "prompt",
    "relaxed",
    "skillful",
    "warm",
)

_AGENTS = (
    ("baker", "bakers"),
    ("courier", "couriers"),
    ("dancer", "dancers"),
    ("driver", "drivers"),
    ("gardener", "gardeners"),
    ("musician", "musicians"),
    ("neighbor", "neighbors"),
    ("painter", "painters"),
    ("reader", "readers"),
    ("singer", "singers"),
    ("student", "students"),
    ("teacher", "teachers"),
    ("traveler", "travelers"),
    ("visitor", "visitors"),
    ("worker", "workers"),
    ("writer", "writers"),
)

_VERBS = (
    ("carry", "carries"),
    ("place", "places"),
    ("watch", "watches"),
    ("study", "studies"),
    ("prepare", "prepares"),
    ("organize", "organizes"),
    ("deliver", "delivers"),
    ("open", "opens"),
    ("close", "closes"),
    ("move", "moves"),
    ("polish", "polishes"),
    ("wash", "washes"),
    ("sort", "sorts"),
    ("collect", "collects"),
    ("fold", "folds"),
    ("pack", "packs"),
)

_OBJECTS = (
    "canvas bag",
    "ceramic bowl",
    "cotton blanket",
    "glass jar",
    "green folder",
    "linen cloth",
    "maple box",
    "navy parcel",
    "orange ribbon",
    "paper packet",
    "purple card",
    "reed basket",
    "silver tray",
    "small envelope",
    "teal notebook",
    "wooden case",
)

_PLACES = (
    "front doorway",
    "garden path",
    "inner hallway",
    "open courtyard",
    "quiet balcony",
    "reading room",
    "rear entrance",
    "side window",
    "stone landing",
    "sunny porch",
    "upper corridor",
    "waiting area",
)

_TIMES = (
    "each morning",
    "every afternoon",
    "before sunset",
    "after lunch",
    "on quiet evenings",
    "during the week",
    "before the meeting",
    "after the break",
)


def _lexical_row(index: int) -> dict[str, str]:
    # Mixed-radix selection keeps the full lexical tuple injective for the
    # bounded catalog while avoiding an identifier token inside the sentence.
    remainder = index
    adjective_index = remainder % len(_ADJECTIVES)
    remainder //= len(_ADJECTIVES)
    agent_index = remainder % len(_AGENTS)
    remainder //= len(_AGENTS)
    verb_index = remainder % len(_VERBS)
    remainder //= len(_VERBS)
    object_index = remainder % len(_OBJECTS)
    remainder //= len(_OBJECTS)
    place_index = remainder % len(_PLACES)
    remainder //= len(_PLACES)
    time_index = remainder % len(_TIMES)
    first = _AGENTS[agent_index]
    second = _AGENTS[(agent_index + 7) % len(_AGENTS)]
    if second == first:
        second = _AGENTS[(agent_index + 8) % len(_AGENTS)]
    base, third = _VERBS[verb_index]
    return {
        "adjective": _ADJECTIVES[adjective_index],
        "second_adjective": _ADJECTIVES[
            (adjective_index + 9) % len(_ADJECTIVES)
        ],
        "singular": first[0],
        "plural": first[1],
        "second_singular": second[0],
        "second_plural": second[1],
        "base": base,
        "third": third,
        "object": _OBJECTS[object_index],
        "place": _PLACES[place_index],
        "time": _TIMES[time_index],
    }


def _singular_simple(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"The {x['adjective']} {x['singular']} {x['base']} the {x['object']} {x['time']}."
    right = f"The {x['adjective']} {x['singular']} {x['third']} the {x['object']} {x['time']}."
    return wrong, right


def _plural_simple(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"The {x['adjective']} {x['plural']} {x['third']} the {x['object']} {x['time']}."
    right = f"The {x['adjective']} {x['plural']} {x['base']} the {x['object']} {x['time']}."
    return wrong, right


def _singular_be(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"The {x['adjective']} {x['singular']} are ready near the {x['place']}."
    right = f"The {x['adjective']} {x['singular']} is ready near the {x['place']}."
    return wrong, right


def _plural_be(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"The {x['adjective']} {x['plural']} is ready near the {x['place']}."
    right = f"The {x['adjective']} {x['plural']} are ready near the {x['place']}."
    return wrong, right


def _singular_have(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"The {x['adjective']} {x['singular']} have the {x['object']} by the {x['place']}."
    right = f"The {x['adjective']} {x['singular']} has the {x['object']} by the {x['place']}."
    return wrong, right


def _plural_have(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"The {x['adjective']} {x['plural']} has the {x['object']} by the {x['place']}."
    right = f"The {x['adjective']} {x['plural']} have the {x['object']} by the {x['place']}."
    return wrong, right


def _singular_do(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"The {x['adjective']} {x['singular']} do not {x['base']} the {x['object']} {x['time']}."
    right = f"The {x['adjective']} {x['singular']} does not {x['base']} the {x['object']} {x['time']}."
    return wrong, right


def _plural_do(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"The {x['adjective']} {x['plural']} does not {x['base']} the {x['object']} {x['time']}."
    right = f"The {x['adjective']} {x['plural']} do not {x['base']} the {x['object']} {x['time']}."
    return wrong, right


def _each_of(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"Each of the {x['adjective']} {x['plural']} {x['base']} one {x['object']} {x['time']}."
    right = f"Each of the {x['adjective']} {x['plural']} {x['third']} one {x['object']} {x['time']}."
    return wrong, right


def _one_of(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"One of the {x['adjective']} {x['plural']} {x['base']} the {x['object']} near the {x['place']}."
    right = f"One of the {x['adjective']} {x['plural']} {x['third']} the {x['object']} near the {x['place']}."
    return wrong, right


def _compound_subject(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"The {x['adjective']} {x['singular']} and the {x['second_adjective']} {x['second_singular']} {x['third']} the {x['object']} {x['time']}."
    right = f"The {x['adjective']} {x['singular']} and the {x['second_adjective']} {x['second_singular']} {x['base']} the {x['object']} {x['time']}."
    return wrong, right


def _singular_intervener(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"The {x['adjective']} {x['singular']} beside the {x['second_adjective']} {x['second_plural']} {x['base']} the {x['object']} {x['time']}."
    right = f"The {x['adjective']} {x['singular']} beside the {x['second_adjective']} {x['second_plural']} {x['third']} the {x['object']} {x['time']}."
    return wrong, right


def _plural_intervener(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"The {x['adjective']} {x['plural']} beside the {x['second_adjective']} {x['second_singular']} {x['third']} the {x['object']} {x['time']}."
    right = f"The {x['adjective']} {x['plural']} beside the {x['second_adjective']} {x['second_singular']} {x['base']} the {x['object']} {x['time']}."
    return wrong, right


def _every_subject(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"Every {x['adjective']} {x['singular']} {x['base']} the {x['object']} near the {x['place']}."
    right = f"Every {x['adjective']} {x['singular']} {x['third']} the {x['object']} near the {x['place']}."
    return wrong, right


def _gerund_subject(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"Sorting the {x['adjective']} {x['object']} near the {x['place']} require patience."
    right = f"Sorting the {x['adjective']} {x['object']} near the {x['place']} requires patience."
    return wrong, right


def _singular_pronoun(x: dict[str, str]) -> tuple[str, str]:
    wrong = f"She {x['base']} the {x['adjective']} {x['object']} near the {x['place']} {x['time']}."
    right = f"She {x['third']} the {x['adjective']} {x['object']} near the {x['place']} {x['time']}."
    return wrong, right


STRUCTURES: tuple[tuple[str, Callable[[dict[str, str]], tuple[str, str]]], ...] = (
    ("singular_simple", _singular_simple),
    ("plural_simple", _plural_simple),
    ("singular_be", _singular_be),
    ("plural_be", _plural_be),
    ("singular_have", _singular_have),
    ("plural_have", _plural_have),
    ("singular_do", _singular_do),
    ("plural_do", _plural_do),
    ("each_of", _each_of),
    ("one_of", _one_of),
    ("compound_subject", _compound_subject),
    ("singular_intervener", _singular_intervener),
    ("plural_intervener", _plural_intervener),
    ("every_subject", _every_subject),
    ("gerund_subject", _gerund_subject),
    ("singular_pronoun", _singular_pronoun),
)


def _probe(
    *,
    structure: str,
    builder: Callable[[dict[str, str]], tuple[str, str]],
    split: str,
    local_index: int,
    lexical_index: int,
) -> dict[str, Any]:
    wrong, expected = builder(_lexical_row(lexical_index))
    prompt = (
        "Correct the single subject-verb agreement error below. Reply with "
        "exactly one corrected sentence. Do not add quotation marks or an "
        f"explanation.\nSentence: {wrong}"
    )
    evaluator = {
        "kind": "exact",
        "value": expected,
        "case_sensitive": True,
        "prompt_contract_sha256": prompt_contract_sha256(prompt),
    }
    probe: dict[str, Any] = {
        "probe_id": f"natural-grammar-{structure}-{split}-{local_index:03d}-v1",
        "destination_scope": "english_core",
        "capability": "grammar",
        "domain": "domain_independent",
        "split": split,
        "prompt": prompt,
        "max_new_tokens": 48,
        "temperature": 0,
        "seed": 66_000_000 + lexical_index,
        "evaluator": evaluator,
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


def build_natural_grammar_reference_catalog() -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    for structure_index, (structure, builder) in enumerate(STRUCTURES):
        for local_index in range(SEARCH_PER_STRUCTURE):
            lexical_index = structure_index * 1000 + local_index
            probes.append(
                _probe(
                    structure=structure,
                    builder=builder,
                    split="search",
                    local_index=local_index,
                    lexical_index=lexical_index,
                )
            )
        for local_index in range(VALIDATION_PER_STRUCTURE):
            lexical_index = 100_000 + structure_index * 1000 + local_index
            probes.append(
                _probe(
                    structure=structure,
                    builder=builder,
                    split="validation",
                    local_index=local_index,
                    lexical_index=lexical_index,
                )
            )

    expected = {
        (structure, split): count
        for structure, _ in STRUCTURES
        for split, count in (
            ("search", SEARCH_PER_STRUCTURE),
            ("validation", VALIDATION_PER_STRUCTURE),
        )
    }
    counts = Counter(
        (
            probe["probe_id"].removeprefix("natural-grammar-").split(
                f"-{probe['split']}-", 1
            )[0],
            probe["split"],
        )
        for probe in probes
    )
    if dict(counts) != expected:
        raise RuntimeError("natural grammar structure/split depth drift")
    if len({probe["prompt"] for probe in probes}) != len(probes):
        raise RuntimeError("natural grammar prompts must be distinct")
    if len(
        {
            json.dumps(probe["evaluator"], sort_keys=True, separators=(",", ":"))
            for probe in probes
        }
    ) != len(probes):
        raise RuntimeError("natural grammar evaluator contracts must be distinct")

    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": CATALOG_ID,
        "status": "PREREGISTERED_REFERENCE_ADEQUACY_CANDIDATE",
        "claim_boundary": (
            "This catalog measures source behavior and supplies domain-free "
            "grammar acquisition material. It is not a symbolic LayerCake "
            "repair, does not access final-test data, and cannot prove ABI-to-"
            "LayerCake transfer or general English competence by itself."
        ),
        "generation": {
            "generator": "abi.natural_grammar_reference_catalog",
            "generator_version": "v1",
            "capability": "grammar",
            "structures": [name for name, _ in STRUCTURES],
            "structure_count": len(STRUCTURES),
            "search_per_structure": SEARCH_PER_STRUCTURE,
            "validation_per_structure": VALIDATION_PER_STRUCTURE,
            "search_probes": len(STRUCTURES) * SEARCH_PER_STRUCTURE,
            "validation_probes": len(STRUCTURES) * VALIDATION_PER_STRUCTURE,
            "final_test_probes": 0,
            "total_probes": len(probes),
            "unique_prompts": len(probes),
            "unique_prompt_specific_evaluator_contracts": len(probes),
            "closed_book_specialist_prompts": 0,
            "domain_labels_present": 0,
            "domain_claims_present": 0,
        },
        "probes": probes,
    }


def build_natural_grammar_preflight_catalog() -> dict[str, Any]:
    parent = build_natural_grammar_reference_catalog()
    selected = []
    for structure, _ in STRUCTURES:
        prefix = f"natural-grammar-{structure}-search-"
        selected.append(
            next(
                probe
                for probe in parent["probes"]
                if str(probe["probe_id"]).startswith(prefix)
            )
        )
    preflight = dict(parent)
    generation = dict(parent["generation"])
    generation.update(
        {
            "parent_catalog_id": parent["catalog_id"],
            "parent_total_probes": len(parent["probes"]),
            "search_probes": len(selected),
            "validation_probes": 0,
            "total_probes": len(selected),
            "preflight_only": True,
        }
    )
    preflight.update(
        {
            "catalog_id": f"{parent['catalog_id']}-gpu-preflight",
            "status": "GPU_RUNTIME_AND_EXACT_BEHAVIOR_PREFLIGHT_ONLY",
            "claim_boundary": (
                "This one-row-per-structure subset gates the frozen GPU source "
                "survey. It is never acquisition or promotion material."
            ),
            "generation": generation,
            "probes": selected,
        }
    )
    return preflight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error(f"catalog is immutable: {output}")
    catalog = (
        build_natural_grammar_preflight_catalog()
        if args.preflight
        else build_natural_grammar_reference_catalog()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    load_probe_catalog(output)
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
