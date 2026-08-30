from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

from abi_v2.build_final_validation_bundle import PREFIX
from abi_v2.public_reconstruction import (
    PublicReconstructionError,
    _io_path,
    _run,
    download_assets,
    extract_archive,
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


def test_public_extraction_supports_long_member_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "release.zip"
    long_relative = "/".join(("segment" * 8 for _ in range(4))) + "/receipt.json"
    member = f"{PREFIX}/abi_release/{long_relative}"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(member, b"{}\n")
    monkeypatch.setattr(
        "abi_v2.public_reconstruction.verify_archive",
        lambda _path: {"status": "PASS_EXACT_ARCHIVE_IDENTITY"},
    )
    root = extract_archive(archive, tmp_path / "fresh")
    assert _io_path(root / long_relative).read_bytes() == b"{}\n"
    module_root = _io_path(root / long_relative).parent
    (module_root / "long_path_probe.py").write_text("VALUE = 7\n", encoding="utf-8")
    process = _run(
        module_root,
        [sys.executable, "-B", "-c", "import long_path_probe; print(long_path_probe.VALUE)"],
    )
    assert process["exit_code"] == 0
