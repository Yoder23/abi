from abi.capability_compiler_phase3_length_attribution import analyze_row


def test_length_attribution_counts_fallback_excess():
    split = lambda value: [str(value).encode("utf-8")]
    row = {"record_id": "x", "capability": "grammar", "prompt": "p", "output": "abcd"}
    result = analyze_row(row, split, {b"p"})
    assert result["actions"] == 5
    assert result["fallback_excess_actions"] == 3
