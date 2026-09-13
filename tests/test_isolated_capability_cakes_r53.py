import torch

from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER
from experiments.broad_payload_reconstruction_r51.screen_v1a import (
    _router_matrix,
    _router_score,
)
from experiments.isolated_capability_cakes_r53.fit_router import (
    CAPABILITY_TO_INDEX,
)


def test_generic_screen_router_supports_all_fourteen_capabilities():
    rows = [
        {"prompt": f"task {capability}", "capability": capability}
        for capability in CAPABILITY_CAKE_ORDER
    ]
    inputs, labels = _router_matrix(rows, CAPABILITY_TO_INDEX)
    assert inputs.shape == (14, 1024)
    assert labels.tolist() == list(range(14))

    router = torch.nn.Linear(1024, 14)
    with torch.no_grad():
        router.weight.zero_()
        router.bias.copy_(torch.arange(14, dtype=torch.float32))
    result = _router_score(router, inputs, labels)
    assert result["rows"] == 14
    assert result["correct"] == 1
    assert result["rotated_correct"] == 1
