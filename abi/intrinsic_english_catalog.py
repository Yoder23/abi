"""Build a broad, construction-segregated fourteen-capability English catalog."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from .grounded_english_reference_catalog import (
    CAPABILITIES,
    _DAYS,
    _FEELINGS,
    _LOCATIONS,
    _OBJECTS,
    _PLANS,
    _TIMES,
    _VERBS,
    _nonce,
)
from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    load_probe_catalog,
    probe_label_evidence_sha256,
    prompt_contract_sha256,
)


CATALOG_ID = "abi-intrinsic-english-search-validation-v83"
PREFLIGHT_CATALOG_ID = f"{CATALOG_ID}-gpu-preflight"
SEARCH_PER_CAPABILITY = 640
VALIDATION_PER_CAPABILITY = 96
FAMILIES_PER_CAPABILITY = 12

_SURFACES: dict[str, tuple[str, ...]] = {
    "grammar": (
        "Correct only the agreement error and return the corrected sentence: {sentence}",
        "Proofread this sentence for subject-verb agreement. Output one corrected sentence only: {sentence}",
        "Make the verb agree with its subject without changing any other detail: {sentence}",
        "Return a grammatical version of the following, preserving every supplied word except the verb form: {sentence}",
        "Fix the single agreement mistake. Give no explanation: {sentence}",
        "Edit this line so the subject and verb agree; keep the name, object, place, and day: {sentence}",
        "What is the corrected sentence if only subject-verb agreement may change? {sentence}",
        "Rewrite this with correct agreement and nothing else altered: {sentence}",
        "Supply only the proofread sentence. The sole error is agreement: {sentence}",
        "Repair the verb form in this sentence while retaining its meaning: {sentence}",
        "One agreement error appears below. Correct it in one sentence: {sentence}",
        "Produce the clean final sentence after fixing subject-verb agreement: {sentence}",
    ),
    "coherence": (
        "Arrange these labeled events into one coherent paragraph in the stated order {order}: {events}",
        "Write a short paragraph that follows {order}. Preserve every event label: {events}",
        "The events are shuffled. Retell them coherently in this sequence—{order}: {events}",
        "Use clear sequence words to combine the supplied events, following {order}: {events}",
        "Turn the fragments into a logical mini-narrative. Required order: {order}. Fragments: {events}",
        "Reorder and connect these events as {order}, keeping all labels verbatim: {events}",
        "Make one cohesive paragraph from the supplied actions. The chronology is {order}: {events}",
        "Present the following events in their actual progression, {order}: {events}",
        "Join these shuffled sentences smoothly while respecting {order}: {events}",
        "Create a coherent account in the order {order}; do not omit a label: {events}",
        "Retell the supplied sequence as one paragraph, beginning with {first} and ending with {last}: {events}",
        "Restore the logical order {order} and write one concise paragraph: {events}",
    ),
    "prompt_grounding": (
        "Using only this card, state the holder, object, texture, place, and reference in one sentence: {card}",
        "Read the supplied record and report every field without adding information: {card}",
        "What five values does this private card provide? Answer from the card alone: {card}",
        "Ground your answer exclusively in the following fields and mention each value: {card}",
        "Turn this card into one factual sentence using only its contents: {card}",
        "Extract the holder, item, texture, location, and reference from: {card}",
        "Answer with a single sentence that faithfully restates this supplied card: {card}",
        "Which holder, object, texture, place, and reference are written here? {card}",
        "Restate all values from the record, with no outside facts: {card}",
        "Use the card as the complete source for a one-sentence answer: {card}",
        "Report exactly what the supplied entry says about its five fields: {card}",
        "Summarize this private record in one grounded sentence while preserving every value: {card}",
    ),
    "instruction_following": (
        "Follow the exact output contract: {contract}",
        "Return only the requested text, with no explanation. {contract}",
        "Obey every formatting constraint in this instruction: {contract}",
        "Your complete response must satisfy this literal specification: {contract}",
        "Produce the exact constrained answer described here: {contract}",
        "No preface or suffix is allowed. {contract}",
        "Apply this output rule precisely and emit nothing else: {contract}",
        "The response format is strict. {contract}",
        "Complete the following deterministic text transformation: {contract}",
        "Honor the order, spelling, and separators in this requirement: {contract}",
        "Respond under this exact two-part constraint: {contract}",
        "Use only the specified content and punctuation: {contract}",
    ),
    "conversation": (
        "Reply naturally and supportively in one or two sentences to this message: {message}",
        "How would you respond with empathy, using only what this person shared? {message}",
        "Continue the conversation with a brief, considerate reply: {message}",
        "Offer a warm and grounded response to the following speaker: {message}",
        "Write one or two supportive sentences in reply to: {message}",
        "Respond as a thoughtful conversation partner without inventing details: {message}",
        "Acknowledge the feeling and situation in a concise reply: {message}",
        "Give a natural, compassionate response grounded in this message: {message}",
        "Reply in a calm, helpful tone to what was said: {message}",
        "Write the next supportive turn in this conversation: {message}",
        "Answer the speaker directly and empathetically: {message}",
        "Provide a brief conversational response that recognizes both the feeling and plan: {message}",
    ),
    "summarization": (
        "Summarize this supplied note in one sentence under forty words: {note}",
        "Give a concise one-sentence summary that preserves the person, object, place, result, and reference: {note}",
        "Reduce the following note to one accurate sentence without adding facts: {note}",
        "Write a brief summary using only the details in this note: {note}",
        "Capture the essential supplied details in a single sentence: {note}",
        "Condense this note while retaining all named fields and the outcome: {note}",
        "State the note's main event and result in fewer than forty words: {note}",
        "Produce one grounded summary sentence for: {note}",
        "Summarize the following private note faithfully and concisely: {note}",
        "Turn this multi-sentence note into one short sentence: {note}",
        "Write a compact summary that keeps the reference and every supplied detail: {note}",
        "In one sentence, report what happened and what followed: {note}",
    ),
    "rewriting": (
        "Rewrite this as one concise, natural sentence while preserving every supplied detail: {text}",
        "Make the following sentence clearer and less wordy without dropping information: {text}",
        "Rephrase this awkward line in fluent English, retaining all names and details: {text}",
        "Edit the supplied text for clarity and concision: {text}",
        "Produce a smooth one-sentence version of this wording: {text}",
        "Improve the naturalness of the sentence but keep its exact meaning: {text}",
        "Turn this verbose statement into a direct, fluent sentence: {text}",
        "Paraphrase the following in clear English with no added facts: {text}",
        "Rewrite this sentence so it reads naturally and preserves the reference: {text}",
        "Give one polished sentence based only on this supplied wording: {text}",
        "Remove needless phrasing while keeping every factual detail: {text}",
        "Return a concise final edit of: {text}",
    ),
    "email_drafting": (
        "Draft a brief, polite email using only these notes: {notes}",
        "Turn the supplied notes into a concise email with a greeting and closing: {notes}",
        "Write a professional email from the following details without inventing anything: {notes}",
        "Compose a short, courteous message that includes every supplied field: {notes}",
        "Create an email under seventy words from these notes alone: {notes}",
        "Use the notes to draft a clear request email: {notes}",
        "Prepare a polite email that thanks the recipient and makes the stated request: {notes}",
        "Convert these private notes into a complete but compact email: {notes}",
        "Write the email implied by the supplied recipient, item, place, day, time, and reference: {notes}",
        "Draft a friendly professional email without adding dates or facts: {notes}",
        "Produce a concise email that preserves all six note values: {notes}",
        "Write a subject-free email body using only: {notes}",
    ),
    "tone_control": (
        "Rewrite this message in a courteous professional tone while preserving every detail: {message}",
        "Make the supplied message polite and respectful without changing its request: {message}",
        "Adjust only the tone of this note so it sounds professional: {message}",
        "Turn this blunt message into one or two diplomatic sentences: {message}",
        "Rephrase the request warmly while retaining the name, item, place, day, time, and reference: {message}",
        "Produce a tactful version of the following message: {message}",
        "Edit this note for a courteous workplace tone, adding no facts: {message}",
        "Rewrite the wording so it is considerate but still direct: {message}",
        "Make this request sound calm and professional: {message}",
        "Return a polite version that preserves all supplied values: {message}",
        "Soften the tone of this message without weakening or expanding the request: {message}",
        "Write one professional alternative to: {message}",
    ),
    "format_control": (
        "Return only this exact compact JSON object: {expected}",
        "Output the supplied fields as this raw JSON and nothing else: {expected}",
        "Use no code fence or prose; your entire response must be {expected}",
        "Copy this valid JSON object exactly: {expected}",
        "The required output is one line of JSON: {expected}",
        "Produce exactly the following structured object and no commentary: {expected}",
        "Honor this literal JSON response contract: {expected}",
        "Respond with these fields in the shown order: {expected}",
        "Return the exact machine-readable line below: {expected}",
        "Emit only the supplied JSON, preserving spelling and punctuation: {expected}",
        "Your answer must match this JSON object character for character: {expected}",
        "Format control task—output only: {expected}",
    ),
    "clarification": (
        "Ask one concise question that resolves this ambiguity: {request}",
        "Do not choose an interpretation; ask a single clarifying question about: {request}",
        "What one question should be asked before acting on this request? {request}",
        "Reply only with a brief clarification question for: {request}",
        "The destination is ambiguous. Ask which supplied option is intended: {request}",
        "Request the missing choice in one question and do not assume an answer: {request}",
        "Clarify the referent of 'there' by naming both options: {request}",
        "Write one direct question that distinguishes the two possible rooms: {request}",
        "Before proceeding, ask exactly one grounded clarification about: {request}",
        "Respond with a question mark-terminated clarification for: {request}",
        "Ask which of the two supplied destinations the requester means: {request}",
        "Generate a concise clarification rather than completing the ambiguous command: {request}",
    ),
    "abstention": (
        "Answer without guessing. The supplied note does not reveal {missing}; mention reference {reference}.",
        "State that the requested information cannot be determined from the supplied material: missing={missing}; reference={reference}.",
        "The record omits {missing}. Give a concise grounded abstention containing {reference}.",
        "Do not invent the absent value {missing}; explain briefly that it is not supplied and retain {reference}.",
        "Reply that there is not enough information to know {missing}. Include case {reference}.",
        "Use one sentence to abstain from answering the missing field {missing}; cite {reference}.",
        "The private choice is unavailable. Say so without speculation and mention {reference}.",
        "Provide a concise non-answer because {missing} was not given; preserve {reference}.",
        "Refuse to guess the omitted detail {missing} and ground the response with {reference}.",
        "Indicate that the supplied information is insufficient to determine {missing}; include {reference}.",
        "Return a brief uncertainty statement rather than a fabricated answer for {missing}, case {reference}.",
        "Acknowledge that {missing} cannot be known from the record and mention {reference}.",
    ),
    "domain_independent_reasoning": (
        "Use only this ordering and identify who came first, explaining briefly and including {reference}: {ordering}",
        "Given the supplied before-relations, who is earliest? Answer in one sentence with tag {reference}: {ordering}",
        "Reason from these two ordering facts only and name the first arrival; retain {reference}: {ordering}",
        "Determine the earliest person in the chain and state why, using marker {reference}: {ordering}",
        "Which name precedes the other two according to the supplied relations? Include {reference}: {ordering}",
        "Infer the first arrival from this transitive order, with no outside facts: {ordering}; tag={reference}",
        "Answer the abstract ordering question in one grounded sentence and copy {reference}: {ordering}",
        "From the given sequence constraints, identify the first name and include reasoning token {reference}: {ordering}",
        "Solve only this nonce ordering: {ordering}. Who is first? Mention {reference}.",
        "Use the two premises to choose the earliest arrival; include {reference}: {ordering}",
        "State the first person in this supplied chain and a short reason, preserving {reference}: {ordering}",
        "Follow the before-relations to their beginning and answer with tag {reference}: {ordering}",
    ),
    "cake_output_realization": (
        "Realize these fields as one fluent sentence, copying every value and adding no facts: {fields}",
        "Turn the supplied structured values into a natural English sentence: {fields}",
        "Write one sentence that verbalizes all five fields exactly: {fields}",
        "Convert this record into fluent prose without omitting a value: {fields}",
        "Produce a grounded sentence from the following actor, object, action, location, and reference: {fields}",
        "Express this field bundle in ordinary English and add nothing: {fields}",
        "Make one clear sentence that includes each literal value in: {fields}",
        "Render the supplied record as natural language while preserving the reference: {fields}",
        "Verbalize this compact event description in one sentence: {fields}",
        "Use fluent English to state exactly what these fields encode: {fields}",
        "Create one natural sentence solely from the supplied values: {fields}",
        "Transform the structured cake output into a concise English sentence: {fields}",
    ),
}


def _all_of(*rules: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "all_of", "rules": list(rules)}


def _contains(*values: str) -> dict[str, Any]:
    return {"kind": "contains_all", "values": list(values)}


def _maximum(value: int) -> dict[str, Any]:
    return {"kind": "maximum_characters", "value": value}


def _payload(
    capability: str,
    index: int,
    family: int,
) -> tuple[str, dict[str, Any], int, str]:
    name = _nonce(index, 1).capitalize()
    second = _nonce(index, 2).capitalize()
    third = _nonce(index, 3).capitalize()
    reference = _nonce(index, 4)
    obj = _OBJECTS[index % len(_OBJECTS)]
    location = _LOCATIONS[(index * 3) % len(_LOCATIONS)]
    day = _DAYS[index % len(_DAYS)]
    time_name = _TIMES[(index // len(_DAYS)) % len(_TIMES)]
    surface = _SURFACES[capability][family]

    if capability == "grammar":
        base, corrected = _VERBS[index % len(_VERBS)]
        sentence = f"{name} {base} the {obj} to the {location} every {day}."
        return (
            surface.format(sentence=sentence),
            _all_of(
                _contains(name, corrected, obj, location, day),
                {"kind": "contains_none", "values": [f"{name} {base} "]},
                _maximum(260),
            ),
            72,
            "domain_free_instruction",
        )
    if capability == "coherence":
        first, middle, last = (
            f"{reference}-ready",
            f"{reference}-moved",
            f"{reference}-settled",
        )
        order = f"{first}, then {middle}, then {last}"
        events = (
            f"[{middle}] {name} carried the {obj} to the {location}. "
            f"[{last}] {name} closed the door. "
            f"[{first}] {name} wrapped the {obj}."
        )
        return (
            surface.format(
                order=order,
                events=events,
                first=first,
                last=last,
            ),
            _all_of(
                {"kind": "ordered_contains", "values": [first, middle, last]},
                _contains(name, obj, location),
                _maximum(520),
            ),
            128,
            "supplied_non_domain_context",
        )
    if capability == "prompt_grounding":
        texture = ("smooth", "woven", "matte", "soft")[index % 4]
        card = (
            f"holder={name}; object={obj}; texture={texture}; "
            f"location={location}; reference={reference}"
        )
        return (
            surface.format(card=card),
            _all_of(
                _contains(name, obj, texture, location, reference),
                _maximum(340),
            ),
            88,
            "supplied_non_domain_context",
        )
    if capability == "instruction_following":
        top, bottom = _nonce(index, 5), _nonce(index, 6)
        expected = f"First: {top}\nSecond: {bottom}"
        contract = (
            f"Output exactly two lines. Line one is `First: {top}` and line "
            f"two is `Second: {bottom}`."
        )
        return (
            surface.format(contract=contract),
            {"kind": "exact", "value": expected},
            48,
            "domain_free_instruction",
        )
    if capability == "conversation":
        feeling = _FEELINGS[index % len(_FEELINGS)]
        plan = _PLANS[(index * 2) % len(_PLANS)]
        message = (
            f"{name} says, 'I feel {feeling} about {plan}.' "
            f"Private reference: {reference}."
        )
        return (
            surface.format(message=message),
            _all_of(
                _contains(feeling, plan.split()[-1], reference),
                {
                    "kind": "contains_any",
                    "values": [
                        "understand",
                        "sounds",
                        "sorry",
                        "support",
                        "help",
                        "normal",
                        "reasonable",
                    ],
                },
                _maximum(440),
            ),
            104,
            "interpersonal_pragmatics",
        )
    if capability == "summarization":
        result = (
            "became quieter",
            "opened sooner",
            "felt more welcoming",
            "ran smoothly",
        )[index % 4]
        note = (
            f"During {time_name}, {name} moved the {obj} into the {location}. "
            f"Afterward, the room {result}. The reference is {reference}."
        )
        return (
            surface.format(note=note),
            _all_of(
                _contains(name, obj, location, result.split()[-1], reference),
                _maximum(320),
            ),
            88,
            "supplied_non_domain_context",
        )
    if capability == "rewriting":
        text = (
            f"Due to the fact that {name} was in possession of the {obj}, "
            f"{name} proceeded to go to the {location} on {day}, with "
            f"{reference} being the reference."
        )
        return (
            surface.format(text=text),
            _all_of(
                _contains(name, obj, location, day, reference),
                _maximum(300),
            ),
            88,
            "domain_free_instruction",
        )
    if capability == "email_drafting":
        notes = (
            f"recipient={name}; thank them for the {obj}; ask them to bring it "
            f"to the {location}; day={day}; time={time_name}; "
            f"reference={reference}"
        )
        return (
            surface.format(notes=notes),
            _all_of(
                _contains(name, obj, location, day, time_name, reference),
                _maximum(560),
            ),
            136,
            "interpersonal_pragmatics",
        )
    if capability == "tone_control":
        message = (
            f"Hey {name}, get the {obj} to the {location} by {time_name} on "
            f"{day}. Reference {reference}."
        )
        return (
            surface.format(message=message),
            _all_of(
                _contains(name, obj, location, time_name, day, reference),
                {
                    "kind": "contains_any",
                    "values": ["please", "could you", "would you", "thank you"],
                },
                _maximum(460),
            ),
            104,
            "interpersonal_pragmatics",
        )
    if capability == "format_control":
        expected = json.dumps(
            {
                "reference": reference,
                "holder": name,
                "object": obj,
                "location": location,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return (
            surface.format(expected=expected),
            {"kind": "exact", "value": expected},
            96,
            "domain_free_instruction",
        )
    if capability == "clarification":
        option_a, option_b = _nonce(index, 7), _nonce(index, 8)
        request = (
            f"Move {reference} there; 'there' could mean the {option_a} room "
            f"or the {option_b} room."
        )
        return (
            surface.format(request=request),
            _all_of(
                _contains(reference, option_a, option_b),
                {"kind": "regex", "pattern": r"\?\s*$"},
                _maximum(320),
            ),
            80,
            "interpersonal_pragmatics",
        )
    if capability == "abstention":
        missing = f"which sealed envelope {name} privately chose"
        return (
            surface.format(missing=missing, reference=reference),
            _all_of(
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
                        "not supplied",
                        "insufficient",
                    ],
                },
                _maximum(320),
            ),
            80,
            "interpersonal_pragmatics",
        )
    if capability == "domain_independent_reasoning":
        ordering = (
            f"{name} arrived before {second}; {second} arrived before {third}."
        )
        return (
            surface.format(ordering=ordering, reference=reference),
            _all_of(
                _contains(name, reference),
                {
                    "kind": "contains_none",
                    "values": [
                        f"{second} arrived first",
                        f"{third} arrived first",
                    ],
                },
                _maximum(320),
            ),
            80,
            "abstract_or_nonce_content",
        )
    if capability == "cake_output_realization":
        action = ("arrived", "rested", "waited", "remained")[index % 4]
        fields = (
            f"actor={name}; object={obj}; action={action}; "
            f"location={location}; reference={reference}"
        )
        return (
            surface.format(fields=fields),
            _all_of(
                _contains(name, obj, action, location, reference),
                _maximum(340),
            ),
            88,
            "supplied_non_domain_context",
        )
    raise ValueError(f"unsupported capability: {capability}")


def build_intrinsic_english_catalog() -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    total_per_capability = SEARCH_PER_CAPABILITY + VALIDATION_PER_CAPABILITY
    for capability_index, capability in enumerate(CAPABILITIES):
        surfaces = _SURFACES.get(capability)
        if surfaces is None or len(surfaces) != FAMILIES_PER_CAPABILITY:
            raise RuntimeError(f"surface-family contract changed: {capability}")
        for local_index in range(total_per_capability):
            split = (
                "search"
                if local_index < SEARCH_PER_CAPABILITY
                else "validation"
            )
            global_index = capability_index * total_per_capability + local_index
            family = local_index % FAMILIES_PER_CAPABILITY
            prompt, evaluator, maximum, content_basis = _payload(
                capability,
                global_index,
                family,
            )
            evaluator = dict(evaluator)
            evaluator["prompt_contract_sha256"] = prompt_contract_sha256(prompt)
            probe: dict[str, Any] = {
                "probe_id": (
                    f"intrinsic-{capability}-{split}-{local_index:05d}-v83"
                ),
                "destination_scope": "english_core",
                "capability": capability,
                "domain": "domain_independent",
                "split": split,
                "prompt": prompt,
                "max_new_tokens": maximum,
                "temperature": 0,
                "seed": 83_000_000 + global_index,
                "evaluator": evaluator,
                "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": LINGUISTIC_FORM,
                "content_basis": content_basis,
                "domain_labels": [],
                "domain_claims": [],
                # Construction is the evidence basis; the vocabulary-level
                # segregation contract deliberately admits only registered
                # catalog labels here.
                "label_method": "preregistered_catalog",
                "output_introduces_unsupplied_facts": False,
                "surface_family": family,
            }
            probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
            probes.append(probe)

    prompt_hashes = [prompt_contract_sha256(row["prompt"]) for row in probes]
    evaluator_hashes = [
        hashlib.sha256(
            json.dumps(
                row["evaluator"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for row in probes
    ]
    if len(set(prompt_hashes)) != len(probes):
        raise RuntimeError("intrinsic prompts are not unique")
    if len(set(evaluator_hashes)) != len(probes):
        raise RuntimeError("prompt-specific evaluators are not unique")
    counts = Counter((row["capability"], row["split"]) for row in probes)
    expected = {
        **{
            (capability, "search"): SEARCH_PER_CAPABILITY
            for capability in CAPABILITIES
        },
        **{
            (capability, "validation"): VALIDATION_PER_CAPABILITY
            for capability in CAPABILITIES
        },
    }
    if dict(counts) != expected:
        raise RuntimeError("capability/split depth drift")
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": CATALOG_ID,
        "status": "PREREGISTERED_INTRINSIC_DOMAIN_SEGREGATION_CANDIDATE",
        "claim_boundary": (
            "Every prompt is constructed from nonce names, supplied abstract "
            "fields, linguistic transformations, interpersonal pragmatics, or "
            "domain-independent ordering. This establishes record-level "
            "construction segregation, not zero factual content in source "
            "weights, source-output adequacy, transfer, or LayerCake quality."
        ),
        "generation": {
            "generator": "abi.intrinsic_english_catalog",
            "capabilities": list(CAPABILITIES),
            "surface_families_per_capability": FAMILIES_PER_CAPABILITY,
            "search_per_capability": SEARCH_PER_CAPABILITY,
            "validation_per_capability": VALIDATION_PER_CAPABILITY,
            "final_test_probes": 0,
            "total_probes": len(probes),
            "unique_prompts": len(set(prompt_hashes)),
            "unique_prompt_specific_evaluator_contracts": len(
                set(evaluator_hashes)
            ),
            "domain_labels_present": 0,
            "domain_claims_present": 0,
            "closed_book_specialist_prompts": 0,
            "prompt_sha256_set_sha256": hashlib.sha256(
                "\n".join(sorted(prompt_hashes)).encode("ascii")
            ).hexdigest(),
        },
        "probes": probes,
    }


def build_intrinsic_preflight_catalog() -> dict[str, Any]:
    parent = build_intrinsic_english_catalog()
    selected = [
        row
        for row in parent["probes"]
        if row["split"] == "search"
        and int(row["surface_family"]) == (
            int(row["probe_id"].split("-")[-2]) % FAMILIES_PER_CAPABILITY
        )
        and int(row["probe_id"].split("-")[-2]) < FAMILIES_PER_CAPABILITY
    ]
    counts = Counter(row["capability"] for row in selected)
    if counts != Counter({capability: FAMILIES_PER_CAPABILITY for capability in CAPABILITIES}):
        raise RuntimeError("preflight did not retain one row per surface family")
    generation = dict(parent["generation"])
    generation.update(
        {
            "parent_catalog_id": parent["catalog_id"],
            "parent_total_probes": len(parent["probes"]),
            "preflight_only": True,
            "search_per_capability": FAMILIES_PER_CAPABILITY,
            "validation_per_capability": 0,
            "total_probes": len(selected),
        }
    )
    return {
        **{
            key: value
            for key, value in parent.items()
            if key not in {"catalog_id", "status", "claim_boundary", "generation", "probes"}
        },
        "catalog_id": PREFLIGHT_CATALOG_ID,
        "status": "GPU_SOURCE_RUNTIME_AND_FUNCTIONAL_PREFLIGHT_ONLY",
        "claim_boundary": (
            "This 168-row search subset tests all twelve prompt families in "
            "every capability. It is not training material or promotion evidence."
        ),
        "generation": generation,
        "probes": selected,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"catalog is immutable: {output}")
    catalog = (
        build_intrinsic_preflight_catalog()
        if args.preflight
        else build_intrinsic_english_catalog()
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
