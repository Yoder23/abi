from __future__ import annotations

import json
from pathlib import Path

import pytest

from abi_v2.public_reconstruction import (
    PublicReconstructionError,
    download_assets,
)


def test_public_download_rejects_non_github_urls_before_network(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "name": "candidate.zip",
                        "download_url": "https://example.invalid/candidate.zip",
                        "bytes": 1,
                        "sha256": "0" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PublicReconstructionError, match="non-GitHub"):
        download_assets(manifest, tmp_path / "assets")
