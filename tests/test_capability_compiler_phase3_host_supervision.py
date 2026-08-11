import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_host_supervision import surface_repair


def test_surface_repair_converts_number_word_without_semantic_rewrite():
    output, steps = surface_repair(
        "Two small parcels arrived in the east hall.",
        {"kind": "contains_all", "values": ["small parcel", "arrived", "east hall", "2"]},
        "fluent_realization",
    )
    assert output == "2 small parcels arrived in the east hall."
    assert steps == ("number_word_two_to_2",)


def test_surface_repair_canonicalizes_v2_abstention_surface():
    output, steps = surface_repair(
        "The answer cannot be known from the given information.",
        {"kind": "contains_any", "values": ["cannot determine"]},
        "abstention",
    )
    assert "cannot determine" in output
    assert steps == ("abstention_cannot_be_known_to_cannot_determine",)


def test_surface_repair_rejects_semantically_invalid_output():
    with pytest.raises(Phase3Error):
        surface_repair(
            "The parcel is blue.",
            {"kind": "contains_all", "values": ["arrived", "2"]},
            "fluent_realization",
        )
