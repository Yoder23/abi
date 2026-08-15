from abi.capability_compiler_phase4_b50_v24_runtime import (
    PROTOCOL_FORMAT,
    RESULT_FORMAT,
    _merge,
)


def test_v24_runtime_contract_is_frozen():
    assert PROTOCOL_FORMAT == "abi-capability-compiler-phase4-b50-v24-runtime/1"
    assert RESULT_FORMAT == "abi-capability-compiler-phase4-b50-v24-runtime-result/1"


def test_v24_overlay_replaces_candidate_and_retention_reference():
    base = {
        "systems": {"ABI": {"archive_sha256": "old"}, "D0": {"id": 0}},
        "locked_phase2_runtime": {"median_bytes_per_second": 1.0},
    }
    overlay = {
        "systems": {"ABI": {"archive_sha256": "v24"}},
        "v24_runtime_mode": "cpu",
        "v24_conformance_result": "result.json",
        "locked_phase2_runtime": {"median_bytes_per_second": 2.0},
    }
    merged = _merge(base, overlay)
    assert merged["systems"]["ABI"]["archive_sha256"] == "v24"
    assert merged["systems"]["D0"] == {"id": 0}
    assert merged["locked_phase2_runtime"]["median_bytes_per_second"] == 2.0
    assert base["locked_phase2_runtime"]["median_bytes_per_second"] == 1.0
