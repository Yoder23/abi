import torch
from abi.capability_compiler_phase3_candidate_device_audit import device_set


def test_device_set_reports_module_device():
    assert device_set(torch.nn.Linear(2, 2)) == ["cpu"]
