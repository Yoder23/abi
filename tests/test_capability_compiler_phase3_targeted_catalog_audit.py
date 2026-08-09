import json

from abi.capability_compiler_phase3_targeted_catalog import build_catalog
from abi.capability_compiler_phase3_targeted_catalog_audit import audit


def _write(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_targeted_catalog_audit_passes_clean_catalog(tmp_path) -> None:
    candidate = tmp_path / "candidate.json"
    prior = tmp_path / "prior.json"
    document = build_catalog()
    _write(candidate, document)
    _write(prior, {**document, "probes": []})
    result = audit(candidate, [prior])
    assert result["status"] == "PASS_TARGETED_CATALOG_AUDIT"
    assert result["exact_prior_prompt_overlap"] == 0


def test_targeted_catalog_audit_rejects_overlap(tmp_path) -> None:
    candidate = tmp_path / "candidate.json"
    prior = tmp_path / "prior.json"
    document = build_catalog()
    _write(candidate, document)
    _write(prior, {**document, "probes": [document["probes"][0]]})
    result = audit(candidate, [prior])
    assert result["status"] == "FAIL_TARGETED_CATALOG_AUDIT"
    assert result["exact_prior_prompt_overlap"] == 1
