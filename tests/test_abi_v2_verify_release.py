from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from abi_v2.verify_release import (
    VerificationError,
    _cell_index,
    _evidence_hash,
    _verify_host,
    verify,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "results/abi_v2/capability_matrix/qwen2/result.json"
PROTOCOL_SHA256 = "1551f1e53fa29458647519980355d71b19859bb37c2b1ffc27dbd2d4a071c51d"
PACKAGE_HASHES = {
    "english": "acb787b3ffa0153c57d88cd37ba81c3f00b370d4ca4937e659cd4c775851f25d",
    "python": "f1defaef2771ced336a332572a2d2f0e1e542399c877d182c48a6cd2e199231d",
    "chemistry": "f9c9b2668fda5ef6b92844c1b7097fbdf8ff0daaae51f5b86f72d4a49000abeb",
    "civics": "634ce66958859ec36dc1fbdf5ef34d6d2a9949d10cf2348a68c245d8c325d604",
}


def _result() -> dict[str, object]:
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def _write_mutation(tmp_path: Path, value: dict[str, object]) -> Path:
    path = tmp_path / "mutated-result.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _rehash(value: dict[str, object]) -> None:
    value["evidence_sha256"] = _evidence_hash(value)


@pytest.mark.parametrize(
    "mutation",
    [
        "false_gate",
        "training_present",
        "adapter_hash",
        "package_hash",
        "raw_observation_hash",
        "retention",
        "wrong_capability_success",
        "corruption_accepted",
    ],
)
def test_verifier_rejects_semantically_rehashed_mutations(
    tmp_path: Path, mutation: str
) -> None:
    value = _result()
    if mutation == "false_gate":
        value["gates"]["teacher_absent"] = False
    elif mutation == "training_present":
        value["training_performed"] = True
    elif mutation == "adapter_hash":
        value["adapter"]["sha256_after"] = "0" * 64
    elif mutation == "package_hash":
        value["installation"]["python"]["archive_sha256"] = "0" * 64
    elif mutation == "raw_observation_hash":
        value["observations"]["sha256"] = "0" * 64
    elif mutation == "retention":
        value["source_success_retention"]["english"]["retention"] = 0.999
    elif mutation == "wrong_capability_success":
        value["causal"]["wrong_capability"]["python"]["successes"] = 1
    elif mutation == "corruption_accepted":
        value["causal"]["random_and_shuffled_capabilities"]["python"][
            "shuffled_rejected_before_execution"
        ]["rejected"] = False
    _rehash(value)
    with pytest.raises(VerificationError):
        _verify_host(
            ROOT,
            host="qwen2",
            result_path=_write_mutation(tmp_path, value),
            protocol_sha256=PROTOCOL_SHA256,
            package_hashes=PACKAGE_HASHES,
        )


def test_verifier_rejects_evidence_digest_mutation(tmp_path: Path) -> None:
    value = _result()
    value["status"] = "PASS_BUT_RELABELLED"
    with pytest.raises(VerificationError):
        _verify_host(
            ROOT,
            host="qwen2",
            result_path=_write_mutation(tmp_path, value),
            protocol_sha256=PROTOCOL_SHA256,
            package_hashes=PACKAGE_HASHES,
        )


def test_cell_index_rejects_duplicate_identity() -> None:
    rows = [
        {"capability": "english", "probe_id": str(index)} for index in range(1681)
    ]
    duplicate = deepcopy(rows)
    duplicate[-1] = duplicate[0]
    with pytest.raises(VerificationError, match="not unique"):
        _cell_index(duplicate)


def test_existing_immutable_release_recomputes_exactly() -> None:
    result = verify(ROOT, check_existing=True)
    assert result["status"] == "TECHNICALLY_PROVEN_EXTERNAL_VALIDATION_PENDING"
