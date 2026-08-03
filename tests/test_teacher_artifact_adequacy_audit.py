from abi.teacher_artifact_adequacy_audit import (
    _evaluator_is_content_specific,
    _has_terminal_marker,
    _summarize_records,
)
from abi.hf_extraction import prompt_contract_sha256


def test_generic_hygiene_evaluator_is_not_content_specific():
    assert not _evaluator_is_content_specific(
        {
            "kind": "all_of",
            "rules": [
                {"kind": "nonempty", "minimum_characters": 1},
                {"kind": "maximum_characters", "value": 100},
                {"kind": "contains_none", "values": ["forbidden"]},
            ],
        }
    )
    assert _evaluator_is_content_specific(
        {"kind": "contains_all", "values": ["Friday", "agenda"]}
    )


def test_terminal_marker_is_only_a_surface_heuristic():
    assert _has_terminal_marker("Complete sentence.\")")
    assert _has_terminal_marker("```text\ncomplete\n```")
    assert not _has_terminal_marker("The unfinished response can")


def test_summary_distinguishes_ceiling_and_evaluator_adequacy():
    rows = [
        {
            "record_id": "one",
            "capability": "email_drafting",
            "prompt": "Draft from Friday notes",
            "response": "Please meet Friday.",
            "teacher_tokens": 8,
        },
        {
            "record_id": "two",
            "capability": "email_drafting",
            "prompt": "Draft from invoice notes",
            "response": "The invoice can",
            "teacher_tokens": 8,
        },
    ]
    originals = {
        "one": {
            "record_id": "one",
            "finish_reason": "eos_token",
            "authoritative_generated_token_ids": [1, 2],
        },
        "two": {"record_id": "two"},
    }
    results = {
        "one": {
            "evaluator": {
                "kind": "contains_all",
                "values": ["Friday"],
                "prompt_contract_sha256": prompt_contract_sha256(
                    "Draft from Friday notes"
                ),
            }
        },
        "two": {
            "evaluator": {
                "kind": "all_of",
                "rules": [{"kind": "nonempty", "minimum_characters": 1}],
            }
        },
    }
    summary = _summarize_records(
        rows=rows,
        original_records=originals,
        probe_results=results,
        prompt_caps={
            "Draft from Friday notes": 8,
            "Draft from invoice notes": 8,
        },
    )
    assert summary["generation_ceiling_saturated"] == 2
    assert summary["ceiling_saturated_without_terminal_marker"] == 1
    assert summary["content_specific_evaluators"] == 1
    assert summary["finish_reason_present"] == 1
    assert summary["authoritative_generated_token_ids_present"] == 1
    assert summary["length_terminated"] == 0
    assert summary["distinct_content_specific_evaluator_signatures"] == 1
    assert summary["prompt_contract_bindings_valid"] == 1


def test_summary_binds_raw_catalog_prompt_when_record_prompt_is_chat_rendered():
    raw_prompt = "Rewrite the supplied sentence in a courteous tone."
    rendered_prompt = f"<chat><user>{raw_prompt}</user><assistant>"
    contract = prompt_contract_sha256(raw_prompt)
    summary = _summarize_records(
        rows=[
            {
                "record_id": "chat-one",
                "capability": "rewriting",
                "prompt": rendered_prompt,
                "response": "Please send the note.",
                "teacher_tokens": 5,
            }
        ],
        original_records={
            "chat-one": {
                "record_id": "chat-one",
                "finish_reason": "eos_token",
                "authoritative_generated_token_ids": [1, 2, 3, 4, 5],
            }
        },
        probe_results={
            "chat-one": {
                "evaluator": {
                    "kind": "independent_semantic_judge",
                    "prompt_contract_sha256": contract,
                    "judge_observation_sha256": "a" * 64,
                }
            }
        },
        prompt_caps={contract: 32},
    )
    assert summary["prompt_contract_bindings_valid"] == 1
    assert summary["unique_prompt_hashes"] == 1


def test_summary_accepts_bound_contrastive_source_selection_without_generation_ids():
    raw_prompt = "Correct this sentence.\nSentence: The baker carry the bag."
    output = "The baker carries the bag."
    contract = prompt_contract_sha256(raw_prompt)
    import hashlib

    summary = _summarize_records(
        rows=[
            {
                "record_id": "contrastive-one",
                "capability": "grammar",
                "prompt": raw_prompt,
                "response": output,
                "teacher_tokens": 6,
            }
        ],
        original_records={
            "contrastive-one": {
                "record_id": "contrastive-one",
                "teacher_token_counter": "authoritative_source_tokenizer_posthoc_on_contrastive_selection",
            }
        },
        probe_results={
            "contrastive-one": {
                "evaluator": {
                    "kind": "counterbalanced_source_preference",
                    "prompt_contract_sha256": contract,
                    "contrastive_evidence_sha256": "a" * 64,
                    "contrastive_observation_sha256": "b" * 64,
                    "selected_output_sha256": hashlib.sha256(
                        output.encode("utf-8")
                    ).hexdigest(),
                    "ab_margin": 1.0,
                    "ba_margin": 2.0,
                    "teacher_generated_output": False,
                }
            }
        },
        prompt_caps={contract: 48},
    )
    assert summary["contrastive_source_selected_rows"] == 1
    assert summary["valid_contrastive_source_evidence_rows"] == 1
    assert summary["authoritative_generated_token_ids_present"] == 0
    assert summary["finish_reason_present"] == 0
    assert summary["generation_ceiling_saturated"] == 0
