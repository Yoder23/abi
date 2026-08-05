from abi.capability_compiler_phase2_human_packet import CANDIDATES, FORM_COUNT, candidate_first


def test_three_form_counterbalance_is_exact_and_reversed():
    pairs = len(CANDIDATES) * 1_400
    for form_index in range(FORM_COUNT):
        assignments = [candidate_first(index, form_index) for index in range(pairs)]
        assert sum(assignments) == pairs // 2
    for index in range(pairs):
        assert candidate_first(index, 0) is not candidate_first(index, 1)
        assert {candidate_first(index, form) for form in range(FORM_COUNT)} == {False, True}
