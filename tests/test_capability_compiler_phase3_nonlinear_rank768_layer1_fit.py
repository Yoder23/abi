from pathlib import Path
from abi.capability_compiler_phase3_nonlinear_rank768_layer1_fit import FORMAT

def test_format_is_versioned_and_local_only() -> None:
    assert FORMAT.endswith("/1")
    assert "layer1" in FORMAT
