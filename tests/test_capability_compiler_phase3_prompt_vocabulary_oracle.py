import torch

from abi.capability_compiler_phase3_prompt_vocabulary_oracle import _rank_within_prompt


def test_rank_uses_unique_prompt_token_vocabulary():
    logits = torch.tensor([0.0, 4.0, 2.0, 3.0, 1.0])
    prompt = torch.tensor([3, 2, 3, 4])
    assert _rank_within_prompt(logits, prompt, 3) == 1
    assert _rank_within_prompt(logits, prompt, 2) == 2
    assert _rank_within_prompt(logits, prompt, 4) == 3
