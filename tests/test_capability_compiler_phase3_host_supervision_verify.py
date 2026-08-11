from abi.capability_compiler_phase3_host_supervision_verify import (
    independent_host_projection,
    independent_surface_repair,
)


def test_independent_projection_removes_only_search_wrapper():
    prompt = "Give only the requested answer for new exercise P3T-13-0001.\nTurn fields into one sentence."
    assert independent_host_projection(prompt) == "Turn fields into one sentence."


def test_independent_surface_repair_matches_closed_number_rule():
    output, steps = independent_surface_repair(
        "Six small parcels arrived.",
        {"kind": "contains_all", "values": ["small parcel", "arrived", "6"]},
        "fluent_realization",
    )
    assert output == "6 small parcels arrived."
    assert steps == ("number_word_six_to_6",)
