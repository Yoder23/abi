from abi.capability_compiler_phase4_v20_b60_rescreen import strong_route_conformance


def test_strong_conformance_allows_only_functional_safe_guard_prefix():
    rows=[{"capability":"grammar","strong_parent_output_exact":True,"guard_terminated":False,"strong_parent_prefix_preserved":True,"functional_pass_v1":True,"repetition_collapse_v2":False},{"capability":"supplied_text_summarization","strong_parent_output_exact":False,"guard_terminated":True,"strong_parent_prefix_preserved":True,"functional_pass_v1":True,"repetition_collapse_v2":False}]
    assert strong_route_conformance(rows,set())
    rows[-1]["functional_pass_v1"]=False
    assert not strong_route_conformance(rows,set())


def test_weak_rows_are_outside_strong_conformance():
    rows=[{"capability":"coherence","strong_parent_output_exact":False,"guard_terminated":False,"strong_parent_prefix_preserved":False,"functional_pass_v1":False,"repetition_collapse_v2":True},{"capability":"grammar","strong_parent_output_exact":True,"guard_terminated":False,"strong_parent_prefix_preserved":True,"functional_pass_v1":True,"repetition_collapse_v2":False}]
    assert strong_route_conformance(rows,{"coherence"})
