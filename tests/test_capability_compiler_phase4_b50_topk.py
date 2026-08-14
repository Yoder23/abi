import numpy as np

from abi.capability_compiler_phase4_b50_topk import FORMAT, cache_shape_valid


def test_topk_format_is_versioned():
    assert FORMAT == "abi-capability-compiler-phase4-b50-topk/1"


def test_cache_shape_contract_rejects_wrong_shape_or_dtype():
    positions = np.asarray([0, 1], dtype=np.int32)
    indices = np.zeros((2, 64), dtype=np.int32)
    values = np.zeros((2, 64), dtype=np.float16)
    assert cache_shape_valid(
        positions=positions,
        indices=indices,
        values=values,
        expected=2,
        topk=64,
    )
    assert not cache_shape_valid(
        positions=positions,
        indices=indices[:, :32],
        values=values,
        expected=2,
        topk=64,
    )
    assert not cache_shape_valid(
        positions=positions,
        indices=indices,
        values=values.astype(np.float32),
        expected=2,
        topk=64,
    )
