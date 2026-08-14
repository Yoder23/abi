from abi.capability_compiler_phase4_b50_baseline_pack import (
    ARTIFACT_ORDER,
    FORMAT,
    membership_id,
    normalize_memberships,
    source_attempt_accounting,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<user>{messages[0]['content']}</user><assistant>"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        assert text == "Fixed."
        return type("Tokenized", (), {"input_ids": [1]})()


def _teacher_row(record_id, attempt, capability="grammar"):
    return {
        "ir_record_id": record_id,
        "source_attempt_sha256": attempt,
        "capability": capability,
        "destination": "english_core",
        "normalized_generation_prompt": "Fix this.",
        "rendered_generation_prompt": "<user>Fix this.</user><assistant>",
        "normalized_output": "Fixed.",
        "raw_generation_prompt": "Fix this.",
        "raw_output": "Fixed.",
        "raw_output_sha256": "f" * 64,
        "authoritative_generated_token_ids": [1, 2],
        "authoritative_teacher_tokens": 2,
        "teacher_input_tokens": 3,
    }


def test_format_and_artifact_order_are_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-baseline-pack/2"
    assert ARTIFACT_ORDER == (
        "phase1_ir",
        "v138_targeted_ir",
        "v480_host_supervision",
    )


def test_membership_ids_are_artifact_namespaced():
    assert membership_id("phase1_ir", "same") != membership_id(
        "v138_targeted_ir", "same"
    )
    assert membership_id("phase1_ir", "same") == membership_id(
        "phase1_ir", "same"
    )


def test_host_membership_uses_host_prompt_and_preserves_source_attempt():
    attempt = "a" * 64
    targeted = _teacher_row("targeted", attempt, "abstention")
    host = {
        "record_id": "host",
        "source_attempt_sha256": attempt,
        "capability": "abstention",
        "host_prompt": "Host-safe question?",
        "output": "Fixed.",
        "source_authoritative_generated_token_ids": [1, 2],
        "source_teacher_output_tokens": 2,
        "source_teacher_input_tokens": 3,
        "source_output_sha256": "f" * 64,
        "source_generation_prompt_sha256": "e" * 64,
        "surface_normalization_steps": [],
    }
    selected = {
        "phase1_ir": [],
        "v138_targeted_ir": [targeted],
        "v480_host_supervision": [host],
    }
    records = normalize_memberships(selected, FakeTokenizer())
    assert len(records) == 2
    host_record = next(row for row in records if row["source_artifact"] == "v480_host_supervision")
    assert host_record["normalized_generation_prompt"] == "Host-safe question?"
    assert host_record["source_attempt_sha256"] == attempt
    journal = [
        {
            "attempt_sha256": attempt,
            "teacher_tokens": 2,
            "teacher_input_tokens": 3,
            "authoritative_generated_token_ids": [1, 2],
            "output": "Fixed.",
            "output_sha256": "f" * 64,
            "generation_prompt": "Fix this.",
            "generation_prompt_sha256": "e" * 64,
        }
    ]
    accounting = source_attempt_accounting(selected, journal)
    assert accounting["record_memberships"] == 2
    assert accounting["unique_source_attempts"] == 1
    assert accounting["duplicate_memberships"] == 1
    assert accounting["authoritative_teacher_output_tokens"] == 2
    assert accounting["membership_teacher_output_tokens"] == 4


def test_surface_normalized_host_output_is_retokenized():
    attempt = "b" * 64
    host = {
        "record_id": "host-repaired",
        "source_attempt_sha256": attempt,
        "capability": "tone_control",
        "host_prompt": "Rewrite politely.",
        "output": "Fixed.",
        "source_authoritative_generated_token_ids": [9, 2],
        "source_teacher_output_tokens": 2,
        "surface_normalization_steps": ["closed_surface_repair"],
    }
    selected = {
        "phase1_ir": [],
        "v138_targeted_ir": [],
        "v480_host_supervision": [host],
    }
    record = normalize_memberships(selected, FakeTokenizer())[0]
    assert record["authoritative_generated_token_ids"] == [1, 2]
    assert (
        record["sequence_normalization"]
        == "retokenized_after_closed_v479_surface_normalization"
    )
