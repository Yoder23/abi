from abi.capability_compiler_phase3_broad_ir import host_prompt_projection


def test_nonfluent_projection_is_identity() -> None:
    prompt = "Keep this complete supplied context."
    projected, method = host_prompt_projection("grammar", prompt)
    assert projected == prompt
    assert method == "full_normalized_acquisition_prompt_host_bound_selected"
