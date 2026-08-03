"""Build the search-only object/action/location/count repair catalog.

The catalog targets the exact schema class absent from both existing
acquisition streams. Its values, wrappers, and numeric range are disjoint from
the existing natural-English validation generator. It contains no validation
or final-test probes and earns no evaluation credit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .capability_segregation import (
    LINGUISTIC_FORM,
    SEGREGATED_RECORD_SCHEMA,
)
from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    load_probe_catalog,
    probe_label_evidence_sha256,
)


CATALOG_ID = "abi-english-realization-object-fields-search-v1"
DEFAULT_PROBE_COUNT = 512

OBJECTS = (
    "amber crate",
    "canvas pouch",
    "ceramic tray",
    "cobalt packet",
    "copper case",
    "coral envelope",
    "cotton bundle",
    "glass cylinder",
    "ivory carton",
    "linen packet",
    "maple box",
    "navy container",
    "ochre package",
    "paper capsule",
    "pewter bin",
    "plum carrier",
    "reed basket",
    "rose satchel",
    "silver sleeve",
    "slate parcel",
    "teal canister",
    "tin vessel",
    "umber packet",
    "velvet bag",
    "violet case",
    "waxed bundle",
    "willow hamper",
    "wool carton",
    "woven holder",
    "yellow capsule",
    "zinc container",
    "cedar carrier",
)

LOCATIONS = (
    "north alcove",
    "south vestibule",
    "upper landing",
    "lower gallery",
    "west annex",
    "central atrium",
    "rear pavilion",
    "side chamber",
    "inner courtyard",
    "outer lobby",
    "stone arcade",
    "covered terrace",
    "sunlit bay",
    "narrow passage",
    "round foyer",
    "open veranda",
)

ACTIONS = (
    "arrived",
    "reached",
    "entered",
    "appeared in",
)

WRAPPERS = (
    (
        "Convert this field record into one fluent English sentence. Copy "
        "every supplied value literally and add no facts: {fields}"
    ),
    (
        "Express the structured values below as a single natural sentence; "
        "retain the digit, object, action, and location exactly: {fields}"
    ),
    (
        "Realize this record in one grammatical sentence with no invented "
        "detail. Each field value must appear verbatim: {fields}"
    ),
    (
        "Write exactly one ordinary English sentence from these fields. Keep "
        "all four values unchanged, including the numeric count: {fields}"
    ),
    (
        "Turn the following data into a concise fluent statement. Use the "
        "given digit and literal values, and introduce nothing else: {fields}"
    ),
    (
        "Render this supplied record as one natural sentence while preserving "
        "the object, action, location, and count strings: {fields}"
    ),
    (
        "Produce one grammatical sentence that communicates only this field "
        "record and contains every value exactly as written: {fields}"
    ),
    (
        "State the provided structured event in one fluent sentence. Do not "
        "omit, rename, spell out, or supplement any supplied value: {fields}"
    ),
)


def build_targeted_realization_catalog(
    probe_count: int = DEFAULT_PROBE_COUNT,
) -> dict[str, Any]:
    if (
        isinstance(probe_count, bool)
        or not isinstance(probe_count, int)
        or probe_count < 100
    ):
        raise ValueError("probe_count must be an integer of at least 100")
    probes: list[dict[str, Any]] = []
    for index in range(probe_count):
        object_name = OBJECTS[index % len(OBJECTS)]
        location = LOCATIONS[
            (index // len(OBJECTS)) % len(LOCATIONS)
        ]
        action = ACTIONS[(index // len(WRAPPERS)) % len(ACTIONS)]
        count = 11 + (
            (
                index * 17
                + index // len(OBJECTS)
            )
            % 86
        )
        fields = (
            f"object={object_name}; action={action}; "
            f"location={location}; count={count}"
        )
        prompt = WRAPPERS[index % len(WRAPPERS)].format(fields=fields)
        probe: dict[str, Any] = {
            "probe_id": f"realization-object-fields-search-{index:04d}-v1",
            "destination_scope": "english_core",
            "capability": "cake_output_realization",
            "domain": "domain_independent",
            "split": "search",
            "prompt": prompt,
            "max_new_tokens": 64,
            "temperature": 0,
            "seed": 27_310_000 + index,
            "evaluator": {
                "kind": "contains_all",
                "values": [
                    object_name,
                    action,
                    location,
                    str(count),
                ],
            },
            "record_schema": SEGREGATED_RECORD_SCHEMA,
            "knowledge_class": LINGUISTIC_FORM,
            "content_basis": "supplied_non_domain_context",
            "domain_labels": [],
            "domain_claims": [],
            "label_method": "preregistered_catalog",
            "output_introduces_unsupplied_facts": False,
        }
        probe["label_evidence_sha256"] = (
            probe_label_evidence_sha256(probe)
        )
        probes.append(probe)
    if len({probe["prompt"] for probe in probes}) != len(probes):
        raise RuntimeError("targeted realization prompts are not distinct")
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": CATALOG_ID,
        "status": "PREREGISTERED_SEARCH_ONLY_TARGETED_ACQUISITION",
        "claim_boundary": (
            "This catalog supplies a schema class measured absent from prior "
            "search artifacts. It is training/search material only, contains "
            "no validation or final-test probes, and cannot certify English "
            "quality or a moonshot pass."
        ),
        "generation": {
            "generator": "abi.targeted_realization_catalog",
            "generator_version": "v1",
            "probe_count": probe_count,
            "capabilities": ["cake_output_realization"],
            "destination_scopes": ["english_core"],
            "splits": ["search"],
            "schema_family": "object_action_location_count",
            "wrapper_count": len(WRAPPERS),
            "object_count": len(OBJECTS),
            "location_count": len(LOCATIONS),
            "action_count": len(ACTIONS),
            "count_minimum": 11,
            "count_maximum": 96,
            "specialist_or_closed_book_prompts": 0,
            "validation_probes": 0,
            "final_test_probes": 0,
        },
        "probes": probes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--probe-count",
        type=int,
        default=DEFAULT_PROBE_COUNT,
    )
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error(f"catalog is immutable: {output}")
    try:
        catalog = build_targeted_realization_catalog(args.probe_count)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            catalog,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    load_probe_catalog(output)
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
