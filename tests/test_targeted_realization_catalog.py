from abi.hf_extraction import probe_label_evidence_sha256
from abi.targeted_realization_catalog import (
    ACTIONS,
    LOCATIONS,
    OBJECTS,
    WRAPPERS,
    build_targeted_realization_catalog,
)


def test_targeted_realization_catalog_is_distinct_search_only_and_bound():
    catalog = build_targeted_realization_catalog(512)
    probes = catalog["probes"]
    assert len(probes) == 512
    assert len({probe["probe_id"] for probe in probes}) == 512
    assert len({probe["prompt"] for probe in probes}) == 512
    assert {probe["split"] for probe in probes} == {"search"}
    assert {
        probe["capability"] for probe in probes
    } == {"cake_output_realization"}
    assert {
        probe["destination_scope"] for probe in probes
    } == {"english_core"}
    assert all(
        probe["label_evidence_sha256"]
        == probe_label_evidence_sha256(probe)
        for probe in probes
    )
    assert catalog["generation"]["validation_probes"] == 0
    assert catalog["generation"]["final_test_probes"] == 0


def test_targeted_realization_values_are_disjoint_from_legacy_generator():
    legacy_objects = {
        "blue folder",
        "small parcel",
        "meeting note",
        "green notebook",
        "draft",
    }
    legacy_locations = {
        "the quiet room",
        "the east hall",
        "the garden",
        "the reading area",
        "the front desk",
    }
    assert set(OBJECTS).isdisjoint(legacy_objects)
    assert set(LOCATIONS).isdisjoint(legacy_locations)
    assert set(range(11, 97)).isdisjoint(range(1, 10))
    assert len(WRAPPERS) == 8
    assert "arrived" in ACTIONS
