from abi.capability_compiler_phase2_common import CAPABILITIES
from abi.capability_compiler_phase3_guarded_screen_verify_v2 import choose_wrong_route


def test_wrong_route_is_nonidentity_for_every_capability():
    for capability in CAPABILITIES:
        assert choose_wrong_route(capability) != capability
