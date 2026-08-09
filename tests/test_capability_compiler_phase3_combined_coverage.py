import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_combined_coverage import broad_examples


class _Tokenizer:
    def encode_source(self, prompt):
        return list(prompt.encode("utf-8")), []

    def encode_fixed_target(self, output):
        return list(output.encode("utf-8"))

    def decode_actions(self, actions, _):
        return bytes(actions)


def test_broad_examples_preserve_full_prompt_and_lossless_target() -> None:
    rows = [{
        "source_prompt_projection": "full_normalized_acquisition_prompt_host_bound_selected",
        "host_conformant_acquisition_prompt": "Full supplied context.",
        "capability": "grammar",
        "normalized_acquisition_prompt": "Full supplied context.",
        "normalized_output": "Clean output.",
        "ir_record_id": "r1",
    }]
    result = broad_examples(
        rows, _Tokenizer(), maximum_source_lexemes=200, maximum_target_actions=100
    )
    assert result[0]["target_actions"] == list(b"Clean output.")
    assert len(result[0]["source_ids"]) > len(b"Full supplied context.")


def test_broad_examples_fail_if_projection_is_ambiguous() -> None:
    rows = [{
        "source_prompt_projection": "strip_first_line",
        "host_conformant_acquisition_prompt": "Context.",
        "capability": "grammar",
        "normalized_acquisition_prompt": "Context.",
        "normalized_output": "Output.",
        "ir_record_id": "r1",
    }]
    with pytest.raises(Phase3Error, match="projection"):
        broad_examples(rows, _Tokenizer(), maximum_source_lexemes=200, maximum_target_actions=100)
