from abi.capability_compiler_phase7_materialize_product import FORMAT, RESULT_FORMAT


def test_phase7_materialization_formats_are_versioned():
    assert FORMAT == "abi-capability-compiler-phase7-product-materialization/1"
    assert RESULT_FORMAT == "abi-capability-compiler-phase7-product-materialization-result/1"
