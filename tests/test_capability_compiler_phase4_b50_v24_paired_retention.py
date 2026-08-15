from abi.capability_compiler_phase4_b50_v24_paired_retention import (
    FORMAT,
    RESULT_FORMAT,
    _identity,
)


def test_v24_paired_retention_contract_is_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-v24-paired-retention/1"
    assert RESULT_FORMAT == (
        "abi-capability-compiler-phase4-b50-v24-paired-retention-result/1"
    )


def test_identity_requires_output_and_token_identity():
    reference = {"p": {"output": "ok", "output_token_ids": [1]}}
    assert _identity([{"probe_id": "p", **reference["p"]}], reference) == 1
    assert _identity(
        [{"probe_id": "p", "output": "ok", "output_token_ids": [2]}], reference
    ) == 0
