from abi.capability_compiler_phase3_teacher_representation_feasibility import _candidate


def test_candidate_accounting_is_explicit() -> None:
    row = _candidate("x", payload_bytes=12, vectors=2, scalars=6, per_record=True, covers_prompt=True, covers_response=True, tokenizer_independent=True, direct_layercake_alignment=True, standard_method="representation distillation")
    assert row["payload_bytes"] == 12
    assert row["vectors"] == 2
    assert row["scalars"] == 6
    assert row["standard_method"] == "representation distillation"
