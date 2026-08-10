from abi.capability_compiler_phase3_routed_v15_artifact_assembly import ARTIFACT_FORMAT, FORMAT


def test_routed_v15_artifact_formats_are_versioned() -> None:
    assert FORMAT.endswith("/1")
    assert ARTIFACT_FORMAT.endswith("/1")
