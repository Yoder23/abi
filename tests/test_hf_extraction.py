import json

import pytest

from abi.capability_pipeline import (
    CapabilityPipelineError,
    build_source_model_manifest,
    validate_probe_result,
)
from abi.capability_segregation import (
    LINGUISTIC_FORM,
    SEGREGATED_RECORD_SCHEMA,
)
from abi.hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    evaluate_output,
    load_probe_catalog,
    probe_label_evidence_sha256,
    run_probe_catalog,
)
from abi.layercake_acquisition import validate_labeled_extraction_record


def test_schema_closed_output_evaluators():
    assert evaluate_output("The answer is Blue.", {"kind": "contains_all", "values": ["blue"]}) == (
        True,
        1.0,
    )
    assert evaluate_output("first ... second", {"kind": "ordered_contains", "values": ["first", "second"]}) == (
        True,
        1.0,
    )
    passed, score = evaluate_output(
        '{"summary": "short"}',
        {"kind": "json_object", "required_keys": ["summary"]},
    )
    assert passed is True and score == 1.0
    assert evaluate_output("The result is 42.", {"kind": "numeric_equal", "value": 42})[0]
    assert evaluate_output(
        "```python\ndef add(a, b):\n    return a + b\n```",
        {"kind": "python_compiles", "contains": ["def add"]},
    )[0]
    assert evaluate_output(
        "```python\ndef add(a, b):\n    return a + b\n```",
        {
            "kind": "python_function_expression",
            "function_name": "add",
            "arguments": ["a", "b"],
            "expression": "a + b",
        },
    ) == (True, 1.0)
    assert evaluate_output(
        "def add(a, b):\n    return a - b",
        {
            "kind": "python_function_expression",
            "function_name": "add",
            "arguments": ["a", "b"],
            "expression": "a + b",
        },
    ) == (False, 0.5)
    assert evaluate_output(
        '{"item": "book", "count": 3}',
        {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "json_object",
                    "required_keys": ["item", "count"],
                    "expected_values": {"item": "book", "count": 3},
                },
                {"kind": "maximum_characters", "value": 50},
            ],
        },
    )[0]
    with pytest.raises(CapabilityPipelineError, match="unsupported"):
        evaluate_output("x", {"kind": "execute_python"})


def test_probe_catalog_rejects_duplicate_ids_and_domain_leakage(tmp_path):
    catalog = {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "test-v1",
        "probes": [
            {
                "probe_id": "same",
                "destination_scope": "english_core",
                "capability": "rewriting",
                "domain": "python",
                "split": "validation",
                "prompt": "Rewrite this.",
                "evaluator": {"kind": "nonempty"},
            }
        ],
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(CapabilityPipelineError, match="domain-independent"):
        load_probe_catalog(path)

    catalog["probes"][0]["domain"] = "domain_independent"
    catalog["probes"].append(dict(catalog["probes"][0]))
    path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(CapabilityPipelineError, match="duplicate probe_id"):
        load_probe_catalog(path)


class _FakeSource:
    def __init__(self):
        self.batch_calls = 0
        self.source_manifest = build_source_model_manifest(
            model_id="local/fake",
            revision="a" * 40,
            revision_is_immutable=True,
            architecture="FakeCausalLM",
            parameter_count=10,
            tokenizer_id="local/fake",
            tokenizer_revision="a" * 40,
            license_id="test-only",
            weight_files=[
                {"relative_path": "model.bin", "sha256": "b" * 64, "bytes": 10}
            ],
        )

    def generate(self, prompt, *, max_new_tokens, seed, temperature):
        return {
            "rendered_prompt": f"<user>{prompt}</user>",
            "output": "Please revise this sentence.",
            "input_tokens": 4,
            "teacher_tokens": 5,
            "teacher_token_counter": "authoritative_generated_token_ids",
        }

    def generate_batch(self, requests):
        self.batch_calls += 1
        return [
            self.generate(
                request["prompt"],
                max_new_tokens=request["max_new_tokens"],
                seed=request["seed"],
                temperature=request["temperature"],
            )
            for request in requests
        ]


def test_probe_run_binds_rendered_prompt_runtime_token_count_and_result():
    catalog = {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "test-v1",
        "probes": [
            {
                "probe_id": "rewrite-1",
                "destination_scope": "english_core",
                "capability": "rewriting",
                "domain": "domain_independent",
                "split": "validation",
                "prompt": "Rewrite this.",
                "max_new_tokens": 16,
                "temperature": 0,
                "seed": 7,
                "evaluator": {
                    "kind": "contains_all",
                    "values": ["revise", "sentence"],
                },
            }
        ],
    }
    source = _FakeSource()
    records, results = run_probe_catalog(source, catalog, batch_size=8)
    assert len(records) == len(results) == 1
    assert records[0]["prompt"] == "<user>Rewrite this.</user>"
    assert records[0]["teacher_tokens"] == 5
    assert records[0]["teacher_token_count_authoritative"] is True
    assert results[0]["passed"] is True
    assert source.batch_calls == 1
    validate_labeled_extraction_record(records[0])
    validate_probe_result(results[0])


def test_probe_catalog_requires_complete_segregation_metadata(tmp_path):
    catalog = {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "segregated-v1",
        "probes": [
            {
                "probe_id": "rewrite-1",
                "destination_scope": "english_core",
                "capability": "rewriting",
                "domain": "domain_independent",
                "split": "search",
                "prompt": "Rewrite this.",
                "evaluator": {"kind": "nonempty"},
                "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": LINGUISTIC_FORM,
            }
        ],
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(CapabilityPipelineError, match="incomplete"):
        load_probe_catalog(path)


def test_probe_run_builds_semantically_labeled_v2_record():
    catalog = {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "segregated-v1",
        "probes": [
            {
                "probe_id": "rewrite-1",
                "destination_scope": "english_core",
                "capability": "rewriting",
                "domain": "domain_independent",
                "split": "search",
                "prompt": "Rewrite this.",
                "evaluator": {"kind": "nonempty"},
                "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": LINGUISTIC_FORM,
                "content_basis": "domain_free_instruction",
                "domain_labels": [],
                "domain_claims": [],
                "label_method": "human_review",
                "label_evidence_sha256": "c" * 64,
                "output_introduces_unsupplied_facts": False,
            }
        ],
    }
    records, _ = run_probe_catalog(_FakeSource(), catalog)
    assert records[0]["schema_version"] == SEGREGATED_RECORD_SCHEMA
    assert records[0]["knowledge_class"] == LINGUISTIC_FORM
    assert records[0]["domain_labels"] == []
    validate_labeled_extraction_record(records[0])


def test_preregistered_label_evidence_binds_exact_probe(tmp_path):
    probe = {
        "probe_id": "rewrite-locked-1",
        "destination_scope": "english_core",
        "capability": "rewriting",
        "domain": "domain_independent",
        "split": "search",
        "prompt": "Rewrite this nonce sentence.",
        "evaluator": {"kind": "nonempty"},
        "record_schema": SEGREGATED_RECORD_SCHEMA,
        "knowledge_class": LINGUISTIC_FORM,
        "content_basis": "abstract_or_nonce_content",
        "domain_labels": [],
        "domain_claims": [],
        "label_method": "preregistered_catalog",
        "output_introduces_unsupplied_facts": False,
    }
    probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
    catalog = {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": "locked-labels-v1",
        "probes": [probe],
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    load_probe_catalog(path)

    probe["prompt"] = "Tampered domain fact prompt."
    path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(CapabilityPipelineError, match="label evidence"):
        load_probe_catalog(path)
