from abi.capability_compiler_phase3_three_route_feasibility import accounting


def test_three_route_accounting_is_exact_and_under_one_gib() -> None:
    values = accounting()
    assert values["extra_expert_output_parameters"] == 18_874_368
    assert values["router_parameters"] == 9_219
    assert values["deployed_parameters"] == 536_758_275
    assert values["fp16_payload_bytes"] == 1_073_516_550
    assert values["fp16_payload_bytes"] < 1024**3
    assert values["active_routes_per_request"] == 1
