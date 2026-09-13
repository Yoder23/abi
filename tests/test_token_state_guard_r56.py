import torch

from experiments.token_state_guard_r56.screen_v1 import NO_REPEAT_NGRAM_SIZE
from abi.layercake_host import _select_next_token


def test_r56_blocks_only_a_repeated_response_fourgram_continuation():
    scores = torch.zeros(16)
    scores[4] = 10.0
    scores[9] = 9.0
    history = [1, 2, 3, 4, 1, 2, 3]
    selected = _select_next_token(
        scores,
        generated=history,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
    )
    assert int(selected) == 9


def test_r56_does_not_block_an_unseen_response_fourgram():
    scores = torch.zeros(16)
    scores[4] = 10.0
    selected = _select_next_token(
        scores,
        generated=[5, 6, 7],
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
    )
    assert int(selected) == 4
