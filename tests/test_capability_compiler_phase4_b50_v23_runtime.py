from abi.capability_compiler_phase4_b50_v23_runtime import (
    PROTOCOL_FORMAT,
    RESULT_FORMAT,
    _merge,
)


def test_v23_runtime_result_contract_is_frozen():
    assert RESULT_FORMAT == "abi-capability-compiler-phase4-b50-v23-runtime-result/1"
    assert PROTOCOL_FORMAT == "abi-capability-compiler-phase4-b50-v23-runtime/1"


def test_v23_overlay_only_replaces_candidate_identity():
    base = {"systems": {"ABI": {"archive_sha256": "v22"}, "D0": {"id": 0}}}
    overlay = {
        "systems": {"ABI": {"archive_sha256": "v23"}},
        "v23_runtime_mode": "cpu",
        "v23_conformance_result": "result.json",
    }
    merged = _merge(base, overlay)
    assert merged["systems"] == {
        "ABI": {"archive_sha256": "v23"},
        "D0": {"id": 0},
    }
    assert merged["runtime_interface"] == "lc-direct-neural-core/23"
    assert base["systems"]["ABI"]["archive_sha256"] == "v22"
