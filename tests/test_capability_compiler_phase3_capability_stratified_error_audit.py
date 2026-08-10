from abi.capability_compiler_phase3_capability_stratified_error_audit import FORMAT, _summary


def test_capability_stratified_format_is_versioned() -> None:
    assert "capability-stratified-error" in FORMAT
    assert FORMAT.endswith("/1")


def test_summary_is_deterministic_and_exact() -> None:
    result = _summary({"b": [0.5], "a": [0.25, 0.75]})
    assert list(result) == ["a", "b"]
    assert result["a"]["records"] == 2
    assert result["a"]["mean_output_cosine"] == 0.5
