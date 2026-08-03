"""Build a compact, prompt-grounded English reference catalog.

Every probe supplies its own non-specialist facts or nonce symbols and carries
an evaluator whose contract is unique to that prompt.  Search rows are source
material; validation rows qualify the frozen source independently.  No final
test prompts are created or accessed here.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    load_probe_catalog,
    prompt_contract_sha256,
    probe_label_evidence_sha256,
)
from .layercake_acquisition import ENGLISH_CORE_CAPABILITIES


CATALOG_ID = "abi-grounded-english-reference-search-validation-v1"
CATALOG_ID_V2 = "abi-grounded-english-reference-search-validation-v2"
CATALOG_ID_V3 = "abi-grounded-english-reference-search-validation-v3"
SEARCH_PER_CAPABILITY = 160
VALIDATION_PER_CAPABILITY = 64
CAPABILITIES = tuple(sorted(ENGLISH_CORE_CAPABILITIES))

_STEMS = (
    "mavora",
    "kelune",
    "sorali",
    "tavren",
    "nelora",
    "briwen",
    "calira",
    "dovena",
    "elaris",
    "fenora",
    "galune",
    "helori",
    "ivaren",
    "jorali",
    "kasira",
    "lorven",
    "merali",
    "novira",
    "oralen",
    "pelora",
    "quorin",
    "ravela",
    "selune",
    "toravi",
    "uloren",
    "valira",
    "welora",
    "xavren",
    "yelune",
    "zorali",
)

_OBJECTS = (
    "amber parcel",
    "canvas pouch",
    "cedar box",
    "coral folder",
    "cotton bundle",
    "glass token",
    "indigo packet",
    "ivory card",
    "linen envelope",
    "maple case",
    "navy satchel",
    "ochre note",
    "paper capsule",
    "pewter tag",
    "plum ribbon",
    "reed basket",
    "rose sleeve",
    "silver tray",
    "slate carton",
    "teal canister",
    "umber label",
    "velvet bag",
    "violet ticket",
    "willow hamper",
)

_LOCATIONS = (
    "amber alcove",
    "blue vestibule",
    "cedar landing",
    "coral gallery",
    "east atrium",
    "green veranda",
    "inner courtyard",
    "linen foyer",
    "north annex",
    "open terrace",
    "quiet pavilion",
    "rear arcade",
    "silver lobby",
    "south passage",
    "upper chamber",
    "west balcony",
)

_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
_TIMES = (
    "early morning",
    "late morning",
    "just after noon",
    "midafternoon",
    "early evening",
)
_FEELINGS = ("uneasy", "frustrated", "uncertain", "overwhelmed", "hopeful")
_PLANS = (
    "the upcoming visit",
    "the delayed meeting",
    "the shared presentation",
    "the room change",
    "the group conversation",
)
_VERBS = (
    ("carry", "carries"),
    ("place", "places"),
    ("watch", "watches"),
    ("study", "studies"),
    ("prepare", "prepares"),
    ("organize", "organizes"),
)


def _letters(index: int) -> str:
    value = index + 1
    output = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        output = chr(ord("a") + remainder) + output
    return output


def _nonce(index: int, offset: int = 0) -> str:
    stem = _STEMS[(index + offset) % len(_STEMS)]
    return f"{stem}-{_letters(index + offset * 997)}"


def _all_of(*rules: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "all_of", "rules": list(rules)}


def _contains(*values: str) -> dict[str, Any]:
    return {"kind": "contains_all", "values": list(values)}


def _maximum(value: int) -> dict[str, Any]:
    return {"kind": "maximum_characters", "value": value}


def _probe_payload(capability: str, index: int) -> tuple[str, dict[str, Any], int, str]:
    name = _nonce(index, 1).capitalize()
    second = _nonce(index, 2).capitalize()
    third = _nonce(index, 3).capitalize()
    reference = _nonce(index, 4)
    obj = _OBJECTS[index % len(_OBJECTS)]
    location = _LOCATIONS[(index * 3) % len(_LOCATIONS)]
    day = _DAYS[index % len(_DAYS)]
    time_name = _TIMES[(index // len(_DAYS)) % len(_TIMES)]

    if capability == "grammar":
        base, corrected = _VERBS[index % len(_VERBS)]
        sentence = f"{name} {base} the {obj} to the {location} every {day}."
        prompt = (
            "Correct the subject-verb agreement. Return one corrected sentence "
            f"and preserve every other detail: {sentence}"
        )
        evaluator = _all_of(
            _contains(name, corrected, obj, location, day),
            {"kind": "contains_none", "values": [f"{name} {base} "]},
            _maximum(240),
        )
        return prompt, evaluator, 64, "domain_free_instruction"

    if capability == "coherence":
        first = f"{reference}-ready"
        middle = f"{reference}-moved"
        last = f"{reference}-settled"
        prompt = (
            "Turn these shuffled events into one coherent short paragraph. "
            "Keep each event label verbatim and place them in the actual order: "
            f"[{middle}] {name} carried the {obj} to the {location}. "
            f"[{last}] {name} closed the door. "
            f"[{first}] {name} wrapped the {obj}."
        )
        evaluator = _all_of(
            {"kind": "ordered_contains", "values": [first, middle, last]},
            _contains(name, obj, location),
            {"kind": "nonempty", "minimum_characters": 80},
            _maximum(480),
        )
        return prompt, evaluator, 112, "supplied_non_domain_context"

    if capability == "prompt_grounding":
        texture = ("smooth", "woven", "matte", "soft")[index % 4]
        prompt = (
            "Use only the supplied card. Reply in one sentence and include its "
            f"reference word. Card: reference={reference}; object={obj}; "
            f"texture={texture}; location={location}. What object, texture, and "
            "location does the card state?"
        )
        evaluator = _all_of(
            _contains(reference, obj, texture, location),
            _maximum(300),
        )
        return prompt, evaluator, 80, "supplied_non_domain_context"

    if capability == "instruction_following":
        top = _nonce(index, 5)
        bottom = _nonce(index, 6)
        prompt = (
            "Follow all constraints: output exactly two lines, no bullets and no "
            f"extra text. The first line must be `First: {top}` and the second "
            f"must be `Second: {bottom}`."
        )
        evaluator = {
            "kind": "regex",
            "pattern": f"^First:\\s*{top}\\s*\\nSecond:\\s*{bottom}\\s*$",
        }
        return prompt, evaluator, 48, "domain_free_instruction"

    if capability == "conversation":
        feeling = _FEELINGS[index % len(_FEELINGS)]
        plan = _PLANS[(index * 2) % len(_PLANS)]
        prompt = (
            f"{name} says, 'I feel {feeling} about {plan}.' Respond naturally "
            f"and supportively in one or two sentences. Mention {name} and the "
            f"private reference word {reference} so the reply stays grounded."
        )
        evaluator = _all_of(
            _contains(name, reference),
            {
                "kind": "contains_any",
                "values": ["understand", "sounds", "sorry", "support", "help"],
            },
            _maximum(420),
        )
        return prompt, evaluator, 96, "interpersonal_pragmatics"

    if capability == "summarization":
        result = ("became quieter", "opened sooner", "felt more welcoming", "ran smoothly")[index % 4]
        prompt = (
            "Summarize the supplied note in one sentence under forty words; keep "
            "the reference, object, place, and result. Note: "
            f"During {time_name}, {name} moved the {obj} into the {location}. "
            f"Afterward, the room {result}. The note reference is {reference}."
        )
        evaluator = _all_of(
            _contains(reference, obj, location, result),
            _maximum(300),
        )
        return prompt, evaluator, 80, "supplied_non_domain_context"

    if capability == "rewriting":
        prompt = (
            "Rewrite this as one concise, natural sentence without losing the "
            f"reference or supplied details: Due to the fact that {name} was in "
            f"possession of the {obj}, {name} proceeded to go to the {location} "
            f"on {day}, with {reference} being the reference."
        )
        evaluator = _all_of(
            _contains(name, obj, location, day, reference),
            _maximum(280),
        )
        return prompt, evaluator, 80, "domain_free_instruction"

    if capability == "email_drafting":
        prompt = (
            "Draft a brief, polite email using only these notes and no invented "
            f"details: recipient={name}; thank them for the {obj}; ask them to "
            f"bring it to the {location} on {day} during {time_name}; reference="
            f"{reference}. Keep it under seventy words."
        )
        evaluator = _all_of(
            _contains(name, obj, location, day, time_name, reference),
            _maximum(520),
        )
        return prompt, evaluator, 128, "interpersonal_pragmatics"

    if capability == "tone_control":
        prompt = (
            "Rewrite the message in a courteous professional tone, as one or two "
            f"sentences, preserving all details: Hey {name}, get the {obj} to the "
            f"{location} by {time_name} on {day}. Reference {reference}."
        )
        evaluator = _all_of(
            _contains(name, obj, location, time_name, day, reference),
            {
                "kind": "contains_any",
                "values": ["please", "could you", "would you", "thank you"],
            },
            _maximum(420),
        )
        return prompt, evaluator, 96, "interpersonal_pragmatics"

    if capability == "format_control":
        prompt = (
            "Return only one raw JSON object, without a code fence. Use exactly "
            f"these string values: reference={reference}, holder={name}, "
            f"object={obj}, location={location}."
        )
        evaluator = {
            "kind": "json_object",
            "required_keys": ["reference", "holder", "object", "location"],
            "expected_values": {
                "reference": reference,
                "holder": name,
                "object": obj,
                "location": location,
            },
        }
        return prompt, evaluator, 80, "domain_free_instruction"

    if capability == "clarification":
        option_a = _nonce(index, 7)
        option_b = _nonce(index, 8)
        prompt = (
            f"The request says, 'Move {reference} there,' but 'there' could mean "
            f"the {option_a} room or the {option_b} room. Ask one concise "
            "clarifying question. Do not choose an option."
        )
        evaluator = _all_of(
            _contains(reference, option_a, option_b),
            {"kind": "regex", "pattern": "\\?\\s*$"},
            _maximum(300),
        )
        return prompt, evaluator, 72, "interpersonal_pragmatics"

    if capability == "abstention":
        prompt = (
            f"No information is supplied about which sealed envelope {name} "
            f"privately chose. Answer without guessing and mention case {reference}."
        )
        evaluator = _all_of(
            _contains(reference),
            {
                "kind": "contains_any",
                "values": [
                    "cannot determine",
                    "can't determine",
                    "cannot know",
                    "can't know",
                    "not enough information",
                    "no information",
                ],
            },
            _maximum(300),
        )
        return prompt, evaluator, 72, "interpersonal_pragmatics"

    if capability == "domain_independent_reasoning":
        prompt = (
            f"Use only this ordering: {name} arrived before {second}; {second} "
            f"arrived before {third}. Who arrived first? Answer in one sentence "
            f"and include reasoning tag {reference}."
        )
        evaluator = _all_of(
            _contains(name, reference),
            {"kind": "contains_none", "values": [f"{second} arrived first", f"{third} arrived first"]},
            _maximum(300),
        )
        return prompt, evaluator, 72, "abstract_or_nonce_content"

    if capability == "cake_output_realization":
        action = ("arrived", "rested", "waited", "remained")[index % 4]
        prompt = (
            "Realize the supplied fields as one fluent English sentence. Copy "
            "every value literally and add no facts: "
            f"actor={name}; object={obj}; action={action}; location={location}; "
            f"reference={reference}."
        )
        evaluator = _all_of(
            _contains(name, obj, action, location, reference),
            _maximum(320),
        )
        return prompt, evaluator, 80, "supplied_non_domain_context"

    raise ValueError(f"unsupported English capability: {capability}")


def _probe_payload_v2(
    capability: str, index: int
) -> tuple[str, dict[str, Any], int, str]:
    """Apply only changes supported by the disclosed V1 GPU preflight."""

    name = _nonce(index, 1).capitalize()
    second = _nonce(index, 2).capitalize()
    reference = _nonce(index, 4)
    obj = _OBJECTS[index % len(_OBJECTS)]
    location = _LOCATIONS[(index * 3) % len(_LOCATIONS)]
    day = _DAYS[index % len(_DAYS)]

    if capability == "coherence":
        prompt = (
            "Write one coherent short paragraph that puts these shuffled events "
            f"in their logical order: {name} carried the {obj} to the {location}. "
            f"{name} closed the door. {name} first wrapped the {obj}. Use clear "
            "sequence words and preserve the supplied details."
        )
        return (
            prompt,
            _all_of(
                _contains(name, obj, location),
                {
                    "kind": "contains_any",
                    "values": ["first", "then", "after", "finally", "once"],
                },
                {"kind": "nonempty", "minimum_characters": 70},
                _maximum(480),
            ),
            112,
            "supplied_non_domain_context",
        )

    if capability == "conversation":
        feeling = _FEELINGS[index % len(_FEELINGS)]
        plan = _PLANS[(index * 2) % len(_PLANS)]
        prompt = (
            f"{name} says, 'I feel {feeling} about {plan}.' Respond naturally "
            "and supportively in one or two sentences, grounded in what was said."
        )
        return (
            prompt,
            _all_of(
                _contains(name, plan),
                {
                    "kind": "contains_any",
                    "values": [
                        "understand",
                        "sounds",
                        "sorry",
                        "support",
                        "help",
                        "normal",
                    ],
                },
                _maximum(420),
            ),
            96,
            "interpersonal_pragmatics",
        )

    if capability == "prompt_grounding":
        texture = ("smooth", "woven", "matte", "soft")[index % 4]
        prompt = (
            "Use only the supplied card. Reply in one sentence. Card: "
            f"holder={name}; object={obj}; texture={texture}; location={location}. "
            "What holder, object, texture, and location does the card state?"
        )
        return (
            prompt,
            _all_of(_contains(name, obj, texture, location), _maximum(300)),
            80,
            "supplied_non_domain_context",
        )

    if capability == "summarization":
        result = (
            "became quieter",
            "opened sooner",
            "felt more welcoming",
            "ran smoothly",
        )[index % 4]
        result_key = result.split()[-1]
        prompt = (
            "Summarize the supplied note in one sentence under forty words; "
            f"preserve the person, object, place, and result. Note: {name} moved "
            f"the {obj} into the {location}. Afterward, the room {result}."
        )
        return (
            prompt,
            _all_of(
                _contains(name, obj, result_key),
                {
                    "kind": "contains_any",
                    "values": [location, location.split()[-1]],
                },
                _maximum(300),
            ),
            80,
            "supplied_non_domain_context",
        )

    if capability == "tone_control":
        prompt = (
            "Rewrite the message in a courteous professional tone, as one or two "
            f"sentences. Copy the name, object, place, day, and reference verbatim: "
            f"Hey {name}, get the {obj} to the {location} on {day}. Reference "
            f"{reference}."
        )
        return (
            prompt,
            _all_of(
                _contains(name, obj, location, day, reference),
                {
                    "kind": "contains_any",
                    "values": ["please", "could you", "would you", "thank you"],
                },
                _maximum(420),
            ),
            96,
            "interpersonal_pragmatics",
        )

    if capability == "format_control":
        prompt = (
            "Return one JSON code block and no prose. The object must use exactly "
            f"these string values: reference={reference}, holder={name}, "
            f"object={obj}, location={location}."
        )
        return (
            prompt,
            {
                "kind": "json_code_block",
                "required_keys": ["reference", "holder", "object", "location"],
                "expected_values": {
                    "reference": reference,
                    "holder": name,
                    "object": obj,
                    "location": location,
                },
            },
            80,
            "domain_free_instruction",
        )

    if capability == "clarification":
        option_a = _nonce(index, 7)
        option_b = _nonce(index, 8)
        expected = (
            f"For {reference}, should 'there' mean the {option_a} room or the "
            f"{option_b} room?"
        )
        prompt = (
            f"The request 'Move {reference} there' is ambiguous. Ask exactly this "
            f"single clarifying question and nothing else: {expected}"
        )
        return prompt, {"kind": "exact", "value": expected}, 64, "interpersonal_pragmatics"

    if capability == "abstention":
        expected = f"Cannot determine {reference} from the supplied information."
        prompt = (
            f"No information is supplied about which sealed envelope {name} "
            f"privately chose. Do not guess. Reply with exactly: {expected}"
        )
        return prompt, {"kind": "exact", "value": expected}, 48, "interpersonal_pragmatics"

    return _probe_payload(capability, index)


def _probe_payload_v3(
    capability: str, index: int
) -> tuple[str, dict[str, Any], int, str]:
    """Correct the two semantically passing V2 false negatives."""

    name = _nonce(index, 1).capitalize()
    reference = _nonce(index, 4)
    if capability == "conversation":
        feeling = _FEELINGS[index % len(_FEELINGS)]
        plan = _PLANS[(index * 2) % len(_PLANS)]
        prompt = (
            f"{name} says, 'I feel {feeling} about {plan}.' Respond naturally "
            "and supportively in one or two sentences, grounded in what was said."
        )
        evaluator = _all_of(
            _contains(feeling, plan.split()[-1]),
            {
                "kind": "contains_any",
                "values": [
                    "understand",
                    "sounds",
                    "sorry",
                    "support",
                    "help",
                    "normal",
                ],
            },
            _maximum(420),
        )
        return prompt, evaluator, 96, "interpersonal_pragmatics"
    if capability == "clarification":
        option_a = _nonce(index, 7)
        option_b = _nonce(index, 8)
        expected = (
            f"For {reference}, should 'there' mean the {option_a} room or the "
            f"{option_b} room?"
        )
        prompt = (
            f"The request 'Move {reference} there' is ambiguous. Ask exactly this "
            f"single clarifying question and nothing else: {expected}"
        )
        evaluator = _all_of(
            _contains(reference, option_a, option_b),
            {"kind": "regex", "pattern": "\\?\\s*$"},
            _maximum(300),
        )
        return prompt, evaluator, 64, "interpersonal_pragmatics"
    return _probe_payload_v2(capability, index)


def build_grounded_english_reference_catalog(
    version: str = "v1",
) -> dict[str, Any]:
    if version not in {"v1", "v2", "v3"}:
        raise ValueError("grounded reference catalog version must be v1, v2, or v3")
    probes: list[dict[str, Any]] = []
    total_per_capability = SEARCH_PER_CAPABILITY + VALIDATION_PER_CAPABILITY
    for capability_index, capability in enumerate(CAPABILITIES):
        for local_index in range(total_per_capability):
            global_index = capability_index * total_per_capability + local_index
            split = (
                "search"
                if local_index < SEARCH_PER_CAPABILITY
                else "validation"
            )
            if version == "v3":
                prompt, evaluator, maximum, content_basis = _probe_payload_v3(
                    capability, global_index
                )
            elif version == "v2":
                prompt, evaluator, maximum, content_basis = _probe_payload_v2(
                    capability, global_index
                )
            else:
                prompt, evaluator, maximum, content_basis = _probe_payload(
                    capability, global_index
                )
            if version == "v3":
                evaluator = dict(evaluator)
                evaluator["prompt_contract_sha256"] = prompt_contract_sha256(
                    prompt
                )
            probe: dict[str, Any] = {
                "probe_id": f"grounded-{capability}-{split}-{local_index:04d}-{version}",
                "destination_scope": "english_core",
                "capability": capability,
                "domain": "domain_independent",
                "split": split,
                "prompt": prompt,
                "max_new_tokens": maximum,
                "temperature": 0,
                "seed": 57_000_000 + global_index,
                "evaluator": evaluator,
                "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": LINGUISTIC_FORM,
                "content_basis": content_basis,
                "domain_labels": [],
                "domain_claims": [],
                "label_method": "preregistered_catalog",
                "output_introduces_unsupplied_facts": False,
            }
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            probes.append(probe)

    prompt_count = len({probe["prompt"] for probe in probes})
    evaluator_count = len(
        {
            json.dumps(probe["evaluator"], sort_keys=True, separators=(",", ":"))
            for probe in probes
        }
    )
    if prompt_count != len(probes) or evaluator_count != len(probes):
        raise RuntimeError("grounded prompts and evaluator contracts must be unique")
    counts = Counter((probe["capability"], probe["split"]) for probe in probes)
    expected = {
        **{(capability, "search"): SEARCH_PER_CAPABILITY for capability in CAPABILITIES},
        **{
            (capability, "validation"): VALIDATION_PER_CAPABILITY
            for capability in CAPABILITIES
        },
    }
    if dict(counts) != expected:
        raise RuntimeError("grounded capability/split depth drift")
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": (
            CATALOG_ID_V3
            if version == "v3"
            else CATALOG_ID_V2 if version == "v2" else CATALOG_ID
        ),
        "status": "PREREGISTERED_REFERENCE_ADEQUACY_CANDIDATE",
        "claim_boundary": (
            "This deterministic catalog tests supplied-context English behavior "
            "with unique prompt-specific evaluators. It contains no specialist "
            "claims or final-test prompts and cannot by itself prove teacher-to-"
            "LayerCake transfer, natural-language generalization, or global core purity."
        ),
        "generation": {
            "generator": "abi.grounded_english_reference_catalog",
            "generator_version": version,
            "supersedes": (
                CATALOG_ID_V2
                if version == "v3"
                else CATALOG_ID if version == "v2" else None
            ),
            "capabilities": list(CAPABILITIES),
            "search_per_capability": SEARCH_PER_CAPABILITY,
            "validation_per_capability": VALIDATION_PER_CAPABILITY,
            "final_test_probes": 0,
            "total_probes": len(probes),
            "unique_prompts": prompt_count,
            "unique_prompt_specific_evaluator_contracts": evaluator_count,
            "closed_book_specialist_prompts": 0,
            "domain_labels_present": 0,
            "domain_claims_present": 0,
        },
        "probes": probes,
    }


def build_grounded_preflight_catalog(version: str = "v1") -> dict[str, Any]:
    """Select one immutable search probe per capability for GPU transport tests."""

    parent = build_grounded_english_reference_catalog(version)
    selected = []
    for capability in CAPABILITIES:
        selected.append(
            next(
                probe
                for probe in parent["probes"]
                if probe["capability"] == capability
                and probe["split"] == "search"
            )
        )
    preflight = dict(parent)
    generation = dict(parent["generation"])
    generation.update(
        {
            "parent_catalog_id": parent["catalog_id"],
            "parent_total_probes": len(parent["probes"]),
            "total_probes": len(selected),
            "search_per_capability": 1,
            "validation_per_capability": 0,
            "preflight_only": True,
        }
    )
    preflight.update(
        {
            "catalog_id": f"{parent['catalog_id']}-gpu-preflight",
            "status": "GPU_RUNTIME_EVIDENCE_PREFLIGHT_ONLY",
            "claim_boundary": (
                "This exact subset tests source runtime transport and evidence "
                "retention only. It is never acquisition or promotion material."
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
    parser.add_argument("--version", choices=("v1", "v2", "v3"), default="v1")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error(f"catalog is immutable: {output}")
    catalog = (
        build_grounded_preflight_catalog(args.version)
        if args.preflight
        else build_grounded_english_reference_catalog(args.version)
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
