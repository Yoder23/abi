from experiments.factual_semantic_r16.public_sequence_scoring import normalized_text


def test_r26_prospective_teacher_agreement_normalizes_unicode_surface_form() -> None:
    assert normalized_text("Reykjavík") == normalized_text("Reykjavik")
    assert normalized_text("VIENNA.") == normalized_text("Vienna")
    assert normalized_text("Warsaw") != normalized_text("Dublin")
