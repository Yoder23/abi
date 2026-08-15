from abi.capability_compiler_phase4_b40_v25_product_verify import (
    _complete_identity,
    SEEDS,
)


def test_b40_verifier_seed_order_is_fixed():
    assert SEEDS == (104729, 130363, 155921)


def test_complete_identity_rejects_missing_and_duplicate_rows():
    expected = {"a", "b"}
    assert _complete_identity([{"probe_id": "a"}, {"probe_id": "b"}], expected)
    assert not _complete_identity([{"probe_id": "a"}], expected)
    assert not _complete_identity([{"probe_id": "a"}, {"probe_id": "a"}], expected)
