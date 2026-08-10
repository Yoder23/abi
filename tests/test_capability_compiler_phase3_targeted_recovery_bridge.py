import torch

from abi.capability_compiler_phase3_targeted_recovery_bridge import (
    _batch_with_prefixes,
)


def _row():
    return {
        "input_ids": [10, 11, 20, 21, 22, 99],
        "labels": [-100, -100, 20, 21, 22, 99],
        "prompt_tokens": 2,
        "response_tokens": 4,
        "route": 3,
        "capability": "coherence",
    }


def test_generated_prefix_replaces_inputs_but_not_teacher_targets():
    ids, labels, attention, prompt_lengths, routes = _batch_with_prefixes(
        [_row()], 99, torch.device("cpu"), [[30, 31]]
    )
    assert ids.tolist() == [[10, 11, 30, 31, 22, 99]]
    assert labels.tolist() == [[-100, -100, 20, 21, 22, 99]]
    assert attention.tolist() == [[1, 1, 1, 1, 1, 1]]
    assert prompt_lengths.tolist() == [2]
    assert routes.tolist() == [3]


def test_prefix_is_bounded_before_terminal_target():
    ids, labels, *_ = _batch_with_prefixes(
        [_row()], 99, torch.device("cpu"), [[30, 31, 32, 33, 34, 35]]
    )
    assert ids.tolist() == [[10, 11, 30, 31, 32, 99]]
    assert labels.tolist() == [[-100, -100, 20, 21, 22, 99]]
