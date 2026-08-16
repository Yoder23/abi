import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase5_selective_product import (
    DOMAINS,
    SEEDS,
    SYSTEMS,
    _domain_rows,
    _explicit_product_abstention,
    _zero_execution,
)


def test_campaign_matrix_is_exact_and_paired():
    assert SYSTEMS == ("ABI", "L1", "D0")
    assert SEEDS == (104729, 130363, 155921)
    assert DOMAINS == ("chemistry", "civics", "python")


def test_product_abstention_requires_an_explicit_failure_message():
    assert _explicit_product_abstention(
        "The requested chemistry capability is not installed."
    )
    assert _explicit_product_abstention(
        "The requested capability is unavailable or ambiguous."
    )
    assert not _explicit_product_abstention("Hydrogen is probably element one.")


def test_zero_execution_rejects_any_module_prefill_or_decode():
    assert _zero_execution({"cake": {"module_load_calls": 0, "prefill_calls": 0}})
    assert not _zero_execution({"cake": {"module_load_calls": 0, "prefill_calls": 1}})


def test_domain_rows_reject_unbalanced_or_wrong_destination(tmp_path):
    rows = []
    for domain in DOMAINS:
        rows.append(
            {
                "probe_id": f"{domain}-1",
                "domain": domain,
                "split": "synthetic",
                "destination_scope": "domain_cake",
            }
        )
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"probes": rows}), encoding="utf-8")
    assert len(_domain_rows(path, split="synthetic", per_domain=1)) == 3
    rows[0]["destination_scope"] = "english_core"
    path.write_text(json.dumps({"probes": rows}), encoding="utf-8")
    with pytest.raises(Phase3Error):
        _domain_rows(path, split="synthetic", per_domain=1)
