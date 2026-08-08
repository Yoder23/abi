import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_teacher_native_core import controlled_prompt


def test_controlled_prompt_is_explicit_and_stable():
    assert controlled_prompt("grammar", "  Fix this.  ") == "Capability route: grammar\nFix this."


def test_controlled_prompt_rejects_unknown_route():
    with pytest.raises(Phase3Error, match="unknown route"):
        controlled_prompt("chemistry", "x")
