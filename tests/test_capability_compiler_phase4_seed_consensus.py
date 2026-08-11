from pathlib import Path

from safetensors.torch import save_file
import torch

from abi.capability_compiler_phase4_seed_consensus import mean_states


def test_mean_states_uses_exact_equal_weights_and_float64_accumulation(tmp_path: Path) -> None:
    paths = []
    for index, value in enumerate((1.0, 2.0, 6.0)):
        path = tmp_path / f"state-{index}.safetensors"
        save_file({"weight": torch.tensor([value, value + 3], dtype=torch.float32)}, str(path))
        paths.append(path)
    state, receipt = mean_states(paths)
    assert torch.equal(state["weight"], torch.tensor([3.0, 6.0], dtype=torch.float32))
    assert receipt["accumulation_dtype"] == "float64"
    assert receipt["parameters"] == 2


def test_mean_states_is_order_invariant_for_three_float32_sources(tmp_path: Path) -> None:
    paths = []
    for index, value in enumerate((0.125, 0.5, 0.875)):
        path = tmp_path / f"state-{index}.safetensors"
        save_file({"weight": torch.tensor([value], dtype=torch.float32)}, str(path))
        paths.append(path)
    left, _ = mean_states(paths)
    right, _ = mean_states(tuple(reversed(paths)))
    assert torch.equal(left["weight"], right["weight"])
