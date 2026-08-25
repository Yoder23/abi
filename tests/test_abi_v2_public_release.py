from __future__ import annotations

import json
from pathlib import Path

import pytest

from abi_v2.public_release import PublicReleaseError, sha256_file, verify_manifest

REPOSITORY = "example/abi"
TAG = "abi-test-v1"
COMMIT = "a" * 40
ROLES = (
    "definitive_repaired_validation_archive",
    "immutable_chemistry_capability_package",
    "immutable_civics_capability_package",
    "immutable_english_capability_package",
    "immutable_python_capability_package",
)


def _manifest(assets: list[dict[str, object]]) -> dict[str, object]:
    return {
        "format": "abi-v2-public-release-assets/1",
        "repository": REPOSITORY,
        "release_tag": TAG,
        "release_commit": COMMIT,
        "assets": assets,
    }


def test_public_manifest_verifies_all_bound_assets(tmp_path: Path) -> None:
    assets = []
    for index in range(5):
        path = tmp_path / f"asset-{index}.bin"
        path.write_bytes(f"asset-{index}".encode())
        digest = sha256_file(path)
        assets.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "content_address": f"sha256:{digest}",
                "role": ROLES[index],
                "download_url": (
                    f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{path.name}"
                ),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(_manifest(assets)),
        encoding="utf-8",
    )
    result = verify_manifest(manifest, tmp_path)
    assert result["status"] == "PASS_EXACT_PUBLIC_ASSET_IDENTITY"
    assert len(result["assets_verified"]) == 5


def test_public_manifest_fails_closed_on_missing_or_changed_asset(tmp_path: Path) -> None:
    assets = []
    for index in range(5):
        path = tmp_path / f"asset-{index}.bin"
        path.write_bytes(f"asset-{index}".encode())
        digest = sha256_file(path)
        assets.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "content_address": f"sha256:{digest}",
                "role": ROLES[index],
                "download_url": (
                    f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{path.name}"
                ),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(_manifest(assets)),
        encoding="utf-8",
    )
    (tmp_path / "asset-2.bin").write_bytes(b"changed")
    with pytest.raises(PublicReleaseError, match="identity mismatch"):
        verify_manifest(manifest, tmp_path)
