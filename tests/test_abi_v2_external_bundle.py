from __future__ import annotations

import hashlib
import json
from pathlib import Path

from abi_v2.build_external_bundle import CAPABILITY_FILES, collect_bundle_files
from abi_v2.verify_external_bundle import verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def test_clean_room_inventory_is_explicit_and_cache_free() -> None:
    files = collect_bundle_files(ROOT, ROOT.parent / "layercake_release")
    names = [record.archive_path for record in files]
    assert len(names) == len(set(names))
    assert all("__pycache__" not in name and "/.git/" not in name for name in names)
    for relative in CAPABILITY_FILES:
        assert f"abi_release/{relative}" in names
    assert "abi_release/results/abi_v2/external_reproduction/raw_evidence.schema.json" in names


def test_extracted_bundle_verifier_checks_exact_members(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("canonical", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = {
        "format": "abi-v2-clean-room-manifest/1",
        "files": [
            {
                "path": "payload.txt",
                "bytes": payload.stat().st_size,
                "sha256": digest,
                "classification": "test",
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_bundle(tmp_path, strict=True)["passed"] is True
    payload.write_text("mutated", encoding="utf-8")
    result = verify_bundle(tmp_path, strict=True)
    assert result["passed"] is False
    assert result["failures"][0]["reason"] == "identity_mismatch"
