from abi.capability_compiler_phase3_failure_to_supervision_audit import (
    builder_index,
    prompt_projection_exact,
)


def test_builder_index_recovers_task_family_identity():
    assert builder_index("targeted:tone_control:builder-3:wrapper-7") == 3
    assert builder_index("broad_corpus_grounded_v108:abstention") is None


def test_prompt_projection_exact_distinguishes_repair_meta_prompt():
    direct = {
        "normalized_generation_prompt": "Rewrite this sentence.",
        "host_conformant_acquisition_prompt": "Rewrite this sentence.",
    }
    repair = {
        "normalized_generation_prompt": "Repair one answer. <prior_answer>bad</prior_answer>",
        "host_conformant_acquisition_prompt": "Rewrite this sentence.",
    }
    assert prompt_projection_exact(direct)
    assert not prompt_projection_exact(repair)
