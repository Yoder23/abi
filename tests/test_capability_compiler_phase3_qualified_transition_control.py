from abi.capability_compiler_phase3_qualified_transition_control import (
    FROZEN_PREFIXES,
    TRAINABLE_PREFIXES,
)


def test_transition_and_embedding_boundaries_are_disjoint():
    names = (
        "transformer.wte.weight",
        "transformer.wpe.weight",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.ln_f.weight",
        "task_classifier.weight",
        "task_cakes.0.down.weight",
    )
    for name in names:
        trainable = name.startswith(TRAINABLE_PREFIXES)
        frozen = name.startswith(FROZEN_PREFIXES)
        assert trainable != frozen


def test_control_does_not_add_a_parameter_namespace():
    assert TRAINABLE_PREFIXES == (
        "transformer.h.",
        "transformer.ln_f.",
        "task_classifier.",
        "task_cakes.",
    )
    assert FROZEN_PREFIXES == ("transformer.wte.", "transformer.wpe.")
