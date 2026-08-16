from pathlib import Path

from abi.capability_compiler_phase5_construct_screen import (
    DOMAINS,
    _catalog_rows,
    is_explicit_abstention,
    project_catalog_prompt,
)
from abi.capability_compiler_phase3 import Phase3Error
import pytest


def test_abstention_classifier_is_explicit_and_not_generic_uncertainty():
    assert is_explicit_abstention("I cannot determine that from the information given.")
    assert is_explicit_abstention("I don't have the capability installed.")
    assert not is_explicit_abstention("Hydrogen is element one.")
    assert not is_explicit_abstention("Maybe the answer is hydrogen.")


def test_construct_schedule_is_three_distinct_domain_slices():
    root = Path(__file__).resolve().parents[1]
    rows = _catalog_rows(
        root / "catalogs/english_and_first_domains_certification_v6.json",
        split="validation",
        per_domain=20,
    )
    assert len(rows) == 60
    assert len({row["probe_id"] for row in rows}) == 60
    assert {row["domain"] for row in rows} == set(DOMAINS)
    assert all(row["destination_scope"] == "domain_cake" for row in rows)
    assert all(row["split"] == "validation" for row in rows)


def test_catalog_projection_removes_only_one_exact_wrapper():
    prompt = (
        "Evaluation case V6-python-generation-100: "
        "Write only Python code defining `calculate_100(a, b)`."
    )
    assert project_catalog_prompt(prompt) == (
        "Write only Python code defining `calculate_100(a, b)`."
    )
    with pytest.raises(Phase3Error):
        project_catalog_prompt("Write only Python code.")
    with pytest.raises(Phase3Error):
        project_catalog_prompt(
            "Evaluation case V6-x: Evaluation case V6-y: hidden"
        )
