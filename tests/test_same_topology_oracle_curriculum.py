import hashlib

import pytest

from abi.natural_instruction_catalog import _prompt
from abi.same_topology_oracle_curriculum import (
    OracleCurriculumError,
    PROVENANCE,
    build_oracle_curriculum,
)


class _Encoded:
    def __init__(self, ids):
        self.ids = ids


class _Tokenizer:
    @staticmethod
    def encode(value):
        return _Encoded(value.split())


def _fixture():
    instruction = "Rewrite the supplied sentence clearly."
    supplied = "Aster placed the quiet note beside the blue folder."
    prompt = _prompt(instruction, supplied)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    catalog = {
        "catalog_id": "abi-natural-domain-filtered-instruction-search-v1",
        "probes": [
            {
                "probe_id": "natural-search-rewriting-00000-v1",
                "capability": "rewriting",
                "prompt": prompt,
                "natural_prompt_sha256": digest,
            }
        ],
    }
    source = [
        {
            "instruction": instruction,
            "input": supplied,
            "output": "Aster put the quiet note next to the blue folder.",
        }
    ]
    return catalog, source


def test_oracle_curriculum_maps_exact_prompt_and_accounts_tokens():
    catalog, source = _fixture()
    rows, accounting = build_oracle_curriculum(
        catalog=catalog,
        source_rows=source,
        tokenizer=_Tokenizer(),
    )
    assert rows[0]["provenance"] == PROVENANCE
    assert rows[0]["teacher_tokens"] == 10
    assert rows[0]["oracle_only"] is True
    assert rows[0]["production_eligible"] is False
    assert accounting["records"] == 1
    assert accounting["capability_counts"] == {"rewriting": 1}


def test_oracle_curriculum_fails_if_frozen_prompt_cannot_be_reproduced():
    catalog, source = _fixture()
    source[0]["input"] = "Different supplied text."
    with pytest.raises(OracleCurriculumError, match="no exact source row"):
        build_oracle_curriculum(
            catalog=catalog,
            source_rows=source,
            tokenizer=_Tokenizer(),
        )


def test_oracle_curriculum_collapses_only_identical_source_duplicates():
    catalog, source = _fixture()
    source.append(dict(source[0]))
    rows, accounting = build_oracle_curriculum(
        catalog=catalog,
        source_rows=source,
        tokenizer=_Tokenizer(),
    )
    assert len(rows) == 1
    assert accounting["identical_duplicate_source_rows_collapsed"] == 1


def test_oracle_curriculum_rejects_conflicting_duplicate_answers():
    catalog, source = _fixture()
    conflicting = dict(source[0])
    conflicting["output"] = "A conflicting direct target."
    source.append(conflicting)
    with pytest.raises(OracleCurriculumError, match="conflicting reference"):
        build_oracle_curriculum(
            catalog=catalog,
            source_rows=source,
            tokenizer=_Tokenizer(),
        )
