import numpy as np

from abi.capability_compiler_phase4_b50_topk_verify import FORMAT, compare_arrays


def test_topk_verifier_format_is_independent():
    assert FORMAT == "abi-capability-compiler-phase4-b50-topk-verify/1"


def test_exact_array_comparison_rejects_value_and_index_changes():
    positions = np.asarray([0, 1], dtype=np.int32)
    indices = np.zeros((2, 64), dtype=np.int32)
    values = np.zeros((2, 64), dtype=np.float16)
    assert compare_arrays(positions, indices, values, positions, indices, values)["pass"]
    changed_indices = indices.copy()
    changed_indices[0, 0] = 1
    assert not compare_arrays(
        positions, changed_indices, values, positions, indices, values
    )["pass"]
    changed_values = values.copy()
    changed_values[0, 0] = np.float16(1.0)
    assert not compare_arrays(
        positions, indices, changed_values, positions, indices, values
    )["pass"]
