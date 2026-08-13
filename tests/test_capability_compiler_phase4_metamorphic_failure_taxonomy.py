from abi.capability_compiler_phase4_metamorphic_failure_taxonomy import classify


EVALUATOR = {"kind": "ordered_contains", "values": ["N1-PREP", "N1-ACT", "N1-DONE"]}


def test_classifies_pass_wrong_order_and_copy_failure() -> None:
    passed = classify("[N1-PREP] x [N1-ACT] y [N1-DONE] z", EVALUATOR)
    assert passed["primary"] == "pass"
    wrong = classify("[N1-ACT] y [N1-PREP] x [N1-DONE] z", EVALUATOR)
    assert wrong["primary"] == "complete_exact_labels_wrong_order"
    partial = classify("[N1-PREP] x [N1-ACT] y", EVALUATOR)
    assert partial["primary"] == "partial_exact_identifier_copy"
    absent = classify("[N2-PREP] x [N2-ACT] y [N2-DONE] z", EVALUATOR)
    assert absent["primary"] == "identifier_stem_absent"


def test_detects_consecutive_surface_loop() -> None:
    row = classify("[N1-PREP] x inguishedinguishedinguishedinguished", EVALUATOR)
    assert row["surface_loop_suspected"] is True
