from abi.capability_compiler_phase4_v19_frontier_audit import eligible


def test_eligibility_requires_request_and_three_unique_spans():
    assert eligible("Return the labels in order: [A] a; [B] b; [C] c")
    assert not eligible("Return the labels in order: [A] a; [A] b; [C] c")
    assert not eligible("Summarize: [A] a; [B] b; [C] c")
