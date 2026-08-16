from abi.capability_compiler_phase6_composition import (
    DOMAINS,
    SEEDS,
    _historical_selected_domain_rows,
    _selected_only,
)
from abi.capability_compiler_phase3 import Phase3Error
import pytest


def test_phase6_matrix_is_paired_and_domain_complete():
    assert SEEDS == (104729, 130363, 155921)
    assert DOMAINS == ("chemistry", "civics", "python")


def test_selected_only_requires_one_selected_prefill_and_zero_inactive_work():
    delta = {
        "chemistry": {
            "module_load_calls": 1,
            "prefill_calls": 1,
            "decode_step_calls": 10,
        },
        "python": {
            "module_load_calls": 0,
            "prefill_calls": 0,
            "decode_step_calls": 0,
        },
    }
    assert _selected_only(delta, "chemistry")
    delta["python"]["prefill_calls"] = 1
    assert not _selected_only(delta, "chemistry")
    delta["python"]["prefill_calls"] = 0
    delta["chemistry"]["prefill_calls"] = 0
    assert not _selected_only(delta, "chemistry")


def _historical_bundle():
    return {
        "verification": {
            "verified": True,
            "artifact_role": "selected_layercake_training_material_v2",
            "training_eligible": False,
            "historical_manifest_training_eligible": True,
        },
        "selection": {
            "selected_items": [
                {
                    "destination_scope": "domain_cake",
                    "domain": "chemistry",
                    "source_model": "teacher",
                    "source_model_revision": "a" * 40,
                }
            ]
        },
        "budgets": [{"split": "search", "record_ids": ["selected", "wrong"]}],
        "probe_results": [
            {"record_id": "selected", "passed": True},
            {"record_id": "wrong", "passed": True},
        ],
        "records": [
            {
                "record_id": "selected",
                "destination_scope": "domain_cake",
                "domain": "chemistry",
                "split": "search",
                "source_model": "teacher",
                "source_model_revision": "a" * 40,
            },
            {
                "record_id": "wrong",
                "destination_scope": "domain_cake",
                "domain": "chemistry",
                "split": "search",
                "source_model": "other",
                "source_model_revision": "b" * 40,
            },
        ],
    }


def test_historical_lineage_selects_only_frozen_source_and_passing_budget_rows():
    rows, budget, source = _historical_selected_domain_rows(
        _historical_bundle(), domain="chemistry", budget_index=0
    )
    assert [row["record_id"] for row in rows] == ["selected"]
    assert budget["split"] == "search"
    assert source["source_model"] == "teacher"


def test_historical_lineage_never_reauthorizes_current_training_material():
    bundle = _historical_bundle()
    bundle["verification"]["training_eligible"] = True
    with pytest.raises(Phase3Error, match="requires a retired training archive"):
        _historical_selected_domain_rows(bundle, domain="chemistry", budget_index=0)
