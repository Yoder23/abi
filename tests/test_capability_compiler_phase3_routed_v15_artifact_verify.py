from abi.capability_compiler_phase3_routed_v15_artifact_verify import FORMAT


def test_routed_v15_artifact_verifier_format_is_versioned() -> None:
    assert "artifact-verifier" in FORMAT
    assert FORMAT.endswith("/2")
