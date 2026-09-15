import hashlib
import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase5_selective_product import load_protocol

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "reviews/independent_2026-09-14/REPLAY_SURFACE_REPAIR_MANIFEST_V1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert document["format"] == "abi-independent-review-replay-surface-repair/1"
    assert document["status"] == "READY_FOR_BOUNDED_REVIEW_FULL_MOONSHOT_NOT_PROVEN"
    return document


def _assert_inventory(entries: list[dict]) -> None:
    for entry in entries:
        path = ROOT / entry["path"]
        assert path.is_file(), entry["path"]
        assert path.stat().st_size == entry["bytes"], entry["path"]
        assert _sha256(path) == entry["sha256"], entry["path"]


def test_preserved_independent_audit_and_receipts_are_exact() -> None:
    _assert_inventory(_load_manifest()["preserved_audit"])


def test_historical_compatibility_copies_are_exact_and_content_identical() -> None:
    for entry in _load_manifest()["compatibility_copies"]:
        target = ROOT / entry["path"]
        source = ROOT / entry["source"]
        assert target.is_file(), entry["path"]
        assert source.is_file(), entry["source"]
        assert target.stat().st_size == source.stat().st_size == entry["bytes"]
        assert _sha256(target) == _sha256(source) == entry["sha256"]


def test_review_publication_payloads_are_complete_and_exact() -> None:
    _assert_inventory(_load_manifest()["publication_payloads"])


def test_phase5_gap_is_physically_absent_and_bound_to_cleanup_manifest() -> None:
    document = _load_manifest()
    cleanup_path = ROOT / document["cleanup_manifest"]["path"]
    assert cleanup_path.is_file()
    assert _sha256(cleanup_path) == document["cleanup_manifest"]["sha256"]
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
    targets = {entry["path"]: entry for entry in cleanup["targets"]}
    missing = document["unresolved_phase5_payloads"]
    assert len(missing) == 6
    assert sum(entry["bytes"] for entry in missing) == 1_191_079_704
    for entry in missing:
        assert not (ROOT / entry["path"]).exists(), entry["path"]
        cleanup_entry = targets[entry["path"]]
        assert cleanup_entry["size_bytes"] == entry["bytes"]
        assert cleanup_entry["sha256"] == entry["sha256"]


def test_phase5_verifier_reaches_the_declared_tensor_boundary() -> None:
    protocol = ROOT / (
        "ABI_CAPABILITY_COMPILER_PHASE5_SELECTIVE_PRODUCT_REPAIR_PROTOCOL_V1026.json"
    )
    base_protocol = ROOT / (
        "ABI_CAPABILITY_COMPILER_PHASE5_SELECTIVE_PRODUCT_PROTOCOL_V1023.json"
    )
    assert protocol.is_file()
    assert base_protocol.is_file()
    first_missing = (
        "results/abi_capability_compiler_phase4_b40_baselines/headline_v997/"
        "L1_r8_lr1e-4_exp4_seed104729/adapters.safetensors"
    )
    with pytest.raises(Phase3Error, match="Phase 5 selective-product binding changed") as error:
        load_protocol(ROOT, protocol)
    assert first_missing in str(error.value)


def test_full_moonshot_is_not_claimed_by_repair_manifest() -> None:
    document = _load_manifest()
    assert "FULL_MOONSHOT_NOT_PROVEN" in document["status"]
    assert len(document["external_gates_remaining"]) == 6
