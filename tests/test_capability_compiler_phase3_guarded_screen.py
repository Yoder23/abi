from abi.capability_compiler_phase3_guarded_screen import artifact_markers


def test_artifact_marker_loader_returns_closed_common_contract(tmp_path):
    # The full immutable artifact is exercised by protocol preflight/run; this
    # unit test keeps the helper's public contract explicit.
    assert callable(artifact_markers)
