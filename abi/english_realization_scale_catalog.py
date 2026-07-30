"""Build the bounded search-only English realization scaling catalog.

The catalog targets four measured weak capabilities without adding closed-book
questions. Every required fact is supplied in the prompt, and every record is
labeled as linguistic form for the English core.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capability_segregation import (
    LINGUISTIC_FORM,
    SEGREGATED_RECORD_SCHEMA,
)
from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    probe_label_evidence_sha256,
)


CATALOG_ID = "abi-english-realization-scale-v6"
PROBES_PER_CAPABILITY = 500
CAPABILITIES = (
    "rewriting",
    "email_drafting",
    "tone_control",
    "cake_output_realization",
)

NAMES = (
    "Adira",
    "Belen",
    "Cato",
    "Dara",
    "Eli",
    "Farah",
    "Galen",
    "Hana",
    "Idris",
    "Juno",
    "Kira",
    "Leif",
    "Maren",
    "Niko",
    "Orla",
    "Pavel",
    "Rina",
    "Soren",
    "Talia",
    "Vik",
)
ADJECTIVES = (
    "amber",
    "brisk",
    "calm",
    "clear",
    "compact",
    "crimson",
    "gentle",
    "golden",
    "lilac",
    "neat",
    "quiet",
    "silver",
    "small",
    "smooth",
    "teal",
)
NOUNS = (
    "binder",
    "card",
    "draft",
    "envelope",
    "folder",
    "form",
    "ledger",
    "memo",
    "note",
    "packet",
    "parcel",
    "report",
    "sketch",
    "summary",
    "worksheet",
)
PLACES = (
    "Cedar Alcove",
    "East Gallery",
    "Harbor Room",
    "Juniper Hall",
    "Lake Annex",
    "Maple Desk",
    "North Studio",
    "Orchid Corner",
    "Pine Lounge",
    "River Office",
    "South Atrium",
    "Willow Bay",
)
DAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
)
TIMES = (
    "08:10",
    "09:25",
    "10:40",
    "11:55",
    "13:15",
    "14:30",
    "15:45",
    "17:05",
)
ACTIONS = (
    "delivered",
    "filed",
    "moved",
    "placed",
    "prepared",
    "returned",
    "reviewed",
    "sealed",
    "shared",
    "stored",
)
PURPOSES = (
    "a brief review",
    "a careful handoff",
    "a final check",
    "a short discussion",
    "an orderly update",
)


def _all_of(*rules: Mapping[str, Any]) -> dict[str, Any]:
    return {"kind": "all_of", "rules": [dict(rule) for rule in rules]}


def _contains_all(*values: str) -> dict[str, Any]:
    return {"kind": "contains_all", "values": list(values)}


def _contains_any(*values: str) -> dict[str, Any]:
    return {"kind": "contains_any", "values": list(values)}


def _case(index: int) -> dict[str, Any]:
    return {
        "sender": NAMES[index % len(NAMES)],
        "recipient": NAMES[(index * 7 + 3) % len(NAMES)],
        "object": (
            f"{ADJECTIVES[(index * 11 + 2) % len(ADJECTIVES)]} "
            f"{NOUNS[(index * 13 + 4) % len(NOUNS)]}"
        ),
        "place": PLACES[(index * 5 + index // 11) % len(PLACES)],
        "day": DAYS[(index * 5 + index // 17) % len(DAYS)],
        "time": TIMES[(index * 3 + index // 19) % len(TIMES)],
        "action": ACTIONS[(index * 7 + index // 23) % len(ACTIONS)],
        "purpose": PURPOSES[(index * 3 + index // 29) % len(PURPOSES)],
        "count": str(index % 17 + 2),
        "code": f"RV{index:04d}-{chr(65 + index % 26)}{chr(65 + (index * 7) % 26)}",
    }


def _rewriting(index: int) -> tuple[str, dict[str, Any], int]:
    case = _case(index)
    family = index % 5
    statements = (
        (
            f"{case['sender']} {case['action']} the {case['object']}. "
            f"It is now in {case['place']}. The recorded time is {case['time']}."
        ),
        (
            f"The item marked {case['code']} is the {case['object']}. "
            f"{case['recipient']} will review it on {case['day']}."
        ),
        (
            f"{case['sender']} prepared {case['count']} copies. "
            f"{case['recipient']} will collect them from {case['place']} at {case['time']}."
        ),
        (
            f"The {case['object']} is ready for {case['purpose']}. "
            f"{case['sender']} will send it to {case['recipient']} on {case['day']}."
        ),
        (
            f"Record {case['code']} was updated by {case['sender']}. "
            f"The update moved to {case['place']} and is due at {case['time']}."
        ),
    )[family]
    required = (
        (case["sender"], case["action"], case["object"], case["place"], case["time"]),
        (case["code"], case["object"], case["recipient"], case["day"]),
        (case["sender"], case["count"], case["recipient"], case["place"], case["time"]),
        (case["object"], case["purpose"], case["sender"], case["recipient"], case["day"]),
        (case["code"], case["sender"], case["place"], case["time"]),
    )[family]
    required = (*required, case["code"])
    prompt = (
        "Combine the supplied statements into one polished, grammatical "
        "sentence. Preserve every supplied detail and add no facts: "
        + statements
        + f" Tracking reference: {case['code']}."
    )
    return (
        prompt,
        _all_of(
            _contains_all(*required),
            {"kind": "maximum_characters", "value": 300},
        ),
        96,
    )


def _email(index: int) -> tuple[str, dict[str, Any], int]:
    case = _case(index)
    family = index % 5
    request = (
        f"thank {case['recipient']} for {case['action']} the {case['object']} and ask for {case['code']} by {case['day']}",
        f"tell {case['recipient']} that {case['code']} is ready at {case['place']} at {case['time']} and ask for confirmation",
        f"ask {case['recipient']} to bring {case['count']} copies of the {case['object']} to {case['place']} on {case['day']}",
        f"thank {case['recipient']} for helping with the {case['object']} and propose {case['purpose']} at {case['time']}",
        f"tell {case['recipient']} that the {case['object']} was {case['action']} under {case['code']} and invite a reply by {case['day']}",
    )[family]
    required = (
        (case["recipient"], case["object"], case["code"], case["day"]),
        (case["recipient"], case["code"], case["place"], case["time"], "confirm"),
        (case["recipient"], case["count"], case["object"], case["place"], case["day"]),
        (case["recipient"], case["object"], case["purpose"], case["time"], "thank"),
        (case["recipient"], case["object"], case["action"], case["code"], case["day"]),
    )[family]
    required = (*required, case["code"])
    prompt = (
        f"Write a concise, courteous email from {case['sender']} using only "
        f"these notes: {request}. Include a subject, greeting, and closing. "
        "Do not introduce any new event, date, place, or reason. "
        f"Preserve this tracking reference: {case['code']}."
    )
    evaluator = _all_of(
        _contains_all(*required),
        _contains_any("hello", "hi", "dear"),
        _contains_any("regards", "best", "sincerely", "thank"),
        {"kind": "maximum_characters", "value": 700},
    )
    return prompt, evaluator, 160


def _tone(index: int) -> tuple[str, dict[str, Any], int]:
    case = _case(index)
    family = index % 5
    blunt = (
        f"{case['recipient']}, move the {case['object']} to {case['place']} now.",
        f"Send {case['code']} to {case['recipient']} by {case['day']}.",
        f"Bring {case['count']} copies to {case['place']} at {case['time']}.",
        f"{case['recipient']} must {case['action']} the {case['object']}.",
        f"Schedule {case['purpose']} with {case['recipient']} at {case['time']}.",
    )[family]
    required = (
        (case["recipient"], case["object"], case["place"]),
        (case["code"], case["recipient"], case["day"]),
        (case["count"], case["place"], case["time"]),
        (case["recipient"], case["action"], case["object"]),
        (case["purpose"], case["recipient"], case["time"]),
    )[family]
    required = (*required, case["code"])
    prompt = (
        "Rewrite the quoted request as one natural, professional, courteous "
        f"sentence while preserving every supplied detail: \"{blunt}\" "
        f"Keep the tracking reference {case['code']}."
    )
    return (
        prompt,
        _all_of(
            _contains_all(*required),
            _contains_any("please", "could", "would", "appreciate", "thank"),
        ),
        80,
    )


def _realization(index: int) -> tuple[str, dict[str, Any], int]:
    case = _case(index)
    family = index % 5
    fields = (
        f"actor={case['sender']}; action={case['action']}; object={case['object']}; location={case['place']}; time={case['time']}",
        f"record={case['code']}; state=ready; owner={case['recipient']}; day={case['day']}; purpose={case['purpose']}",
        f"actor={case['recipient']}; action=collected; object={case['count']} copies; source={case['place']}; time={case['time']}",
        f"speaker={case['sender']}; action=thanked; recipient={case['recipient']}; reason={case['purpose']}",
        f"object={case['object']}; action={case['action']}; identifier={case['code']}; destination={case['place']}; day={case['day']}",
    )[family]
    required = (
        (case["sender"], case["action"], case["object"], case["place"], case["time"]),
        (case["code"], "ready", case["recipient"], case["day"], case["purpose"]),
        (case["recipient"], "collected", case["count"], case["place"], case["time"]),
        (case["sender"], "thanked", case["recipient"], case["purpose"]),
        (case["object"], case["action"], case["code"], case["place"], case["day"]),
    )[family]
    required = (*required, case["code"])
    prompt = (
        "Turn the supplied fields into one fluent English sentence. Include "
        f"every field value exactly and add no information: {fields}; "
        f"tracking_reference={case['code']}"
    )
    return prompt, _contains_all(*required), 96


BUILDERS = {
    "rewriting": _rewriting,
    "email_drafting": _email,
    "tone_control": _tone,
    "cake_output_realization": _realization,
}


def build_catalog() -> dict[str, Any]:
    probes = []
    for capability, builder in BUILDERS.items():
        for index in range(PROBES_PER_CAPABILITY):
            prompt, evaluator, maximum = builder(index)
            probe: dict[str, Any] = {
                "probe_id": (
                    f"realization-v6-{capability}-search-{index:04d}"
                ),
                "destination_scope": "english_core",
                "capability": capability,
                "domain": "domain_independent",
                "split": "search",
                "prompt": prompt,
                "max_new_tokens": maximum,
                "temperature": 0,
                "seed": 9_860_000 + index,
                "evaluator": evaluator,
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
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": CATALOG_ID,
        "status": "PREREGISTERED_SEARCH_ONLY_MEASURED_REALIZATION_REPAIR",
        "claim_boundary": (
            "This search-only catalog targets four measured realization "
            "weaknesses. It contains supplied-context linguistic-form tasks, "
            "not final evidence, specialist facts, or proof of absolute zero "
            "world knowledge."
        ),
        "generation": {
            "generator": "abi.english_realization_scale_catalog",
            "capabilities": list(CAPABILITIES),
            "probes_per_capability": PROBES_PER_CAPABILITY,
            "probe_count": len(probes),
            "search_only": True,
            "validation_or_final_probes": 0,
            "closed_book_questions": 0,
            "new_evaluator_kinds": 0,
        },
        "probes": probes,
    }


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    catalog = build_catalog()
    _write(Path(args.output), catalog)
    print(
        json.dumps(
            {
                "catalog_id": catalog["catalog_id"],
                "probe_count": len(catalog["probes"]),
                "status": catalog["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
