from abi.capability_compiler_phase3_lexical_substrate_feasibility import projected_accounting


def test_projected_accounting_is_complete():
    value = projected_accounting(32011, 3072, 192, 14587728)
    assert value["projected_payload_bytes_fp16"] == 24_584_448
    assert value["final_imported_substrate_parameters"] == 12_292_224
    assert value["bridge_and_special_parameters"] == 2_295_504
    assert value["source_to_projected_payload_ratio"] == 16
