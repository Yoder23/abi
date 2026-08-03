from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from abi.source_snapshot_import import SourceSnapshotImportError, import_snapshot


def test_import_snapshot_is_exact_and_immutable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    output = tmp_path / "output"
    result = import_snapshot(
        source_path=source,
        output_path=output,
        model_id="example/model",
        revision="abc",
        expected_files={
            "model.safetensors": {"bytes": 7, "sha256": digest}
        },
    )
    assert (output / "model.safetensors").read_bytes() == b"weights"
    assert result["transformations"] == 0
    with pytest.raises(SourceSnapshotImportError):
        import_snapshot(
            source_path=source,
            output_path=output,
            model_id="example/model",
            revision="abc",
            expected_files={
                "model.safetensors": {"bytes": 7, "sha256": digest}
            },
        )


def test_import_snapshot_rejects_unlocked_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "extra.bin").write_bytes(b"extra")
    with pytest.raises(SourceSnapshotImportError):
        import_snapshot(
            source_path=source,
            output_path=tmp_path / "output",
            model_id="example/model",
            revision="abc",
            expected_files={
                "model.safetensors": {
                    "bytes": 7,
                    "sha256": hashlib.sha256(b"weights").hexdigest(),
                }
            },
        )
