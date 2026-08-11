from abi.capability_compiler_phase3_replication_semantic_audit import semantic_projection


def test_semantic_projection_excludes_only_guard_timing():
    source = [{"output": "same", "guard_check_seconds": 1.25, "guard_terminated": False}]
    assert semantic_projection(source) == [{"output": "same", "guard_terminated": False}]
