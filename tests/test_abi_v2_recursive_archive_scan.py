from __future__ import annotations

import bz2
import gzip
import io
import json
import lzma
import tarfile
import zipfile
from pathlib import Path

from abi_v2.isolated_certification import _capability_archive_signatures


def _cake_zip() -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps({"cake_id": "test", "cake_type": "domain", "abi_hash": "0" * 64}),
        )
        archive.writestr("tensors.safetensors", b"tensor")
        archive.writestr("signature.json", b"{}")
    return payload.getvalue()


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    return _capability_archive_signatures(path)


def _zstd_raw_frame(payload: bytes) -> bytes:
    # A standards-conforming single-segment frame with one final raw block.
    assert 256 <= len(payload) <= 65_791
    descriptor = b"\x60"
    content_size = (len(payload) - 256).to_bytes(2, "little")
    block_header = ((len(payload) << 3) | 1).to_bytes(3, "little")
    return b"(\xb5/\xfd" + descriptor + content_size + block_header + payload


def test_prefixed_renamed_zip_is_content_detected(tmp_path: Path) -> None:
    value = _write(tmp_path / "innocent.bin", b"launcher-prefix" + _cake_zip())
    assert any("layercake-capability-package" in row for row in value["signatures"])
    assert value["members_scanned"] == 3


def test_nested_zip_is_recursively_content_detected(tmp_path: Path) -> None:
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("payload.bin", _cake_zip())
    value = _write(tmp_path / "runtime.data", outer.getvalue())
    assert any("layercake-capability-package" in row for row in value["signatures"])
    assert value["members_scanned"] == 4
    assert value["maximum_depth"] >= 2


def test_gzip_wrapped_tar_zip_is_recursively_content_detected(tmp_path: Path) -> None:
    tar_payload = io.BytesIO()
    with tarfile.open(fileobj=tar_payload, mode="w") as archive:
        info = tarfile.TarInfo("payload.bin")
        cake = _cake_zip()
        info.size = len(cake)
        archive.addfile(info, io.BytesIO(cake))
    value = _write(tmp_path / "runtime.cache", gzip.compress(tar_payload.getvalue()))
    assert any("layercake-capability-package" in row for row in value["signatures"])
    assert value["members_scanned"] >= 4
    assert value["maximum_depth"] >= 3


def test_unsupported_archive_magic_is_fail_closed_evidence(tmp_path: Path) -> None:
    value = _write(tmp_path / "opaque.bin", b"prefix" + _zstd_raw_frame(_cake_zip()))
    assert value["unsupported_signatures"] or any(
        "layercake-capability-package" in row for row in value["signatures"]
    )


def test_prefixed_compressed_archives_are_detected_after_old_probe_boundary(
    tmp_path: Path,
) -> None:
    for name, payload in (
        ("gzip", gzip.compress(_cake_zip())),
        ("bzip2", bz2.compress(_cake_zip())),
        ("xz", lzma.compress(_cake_zip())),
    ):
        value = _write(tmp_path / f"{name}.bin", b"P" * 513 + payload)
        assert any("layercake-capability-package" in row for row in value["signatures"])


def test_unsupported_archive_magic_is_detected_after_old_probe_boundary(
    tmp_path: Path,
) -> None:
    value = _write(
        tmp_path / "opaque-after-boundary.bin",
        b"P" * 513 + _zstd_raw_frame(_cake_zip()),
    )
    assert value["unsupported_signatures"] or any(
        "layercake-capability-package" in row for row in value["signatures"]
    )


def test_incidental_unsupported_magic_literal_is_not_a_container(tmp_path: Path) -> None:
    value = _write(
        tmp_path / "literal.bin",
        b"program data mentions 7z\xbc\xaf'\x1c and (\xb5/\xfd but has no valid frame",
    )
    assert value["unsupported_signatures"] == []


def test_archive_magic_split_across_streaming_block_boundary(tmp_path: Path) -> None:
    from abi_v2.isolated_certification import CONTENT_SCAN_BLOCK_BYTES

    payload = b"P" * (CONTENT_SCAN_BLOCK_BYTES - 1) + gzip.compress(_cake_zip())
    value = _write(tmp_path / "split-signature.bin", payload)
    assert any("layercake-capability-package" in row for row in value["signatures"])
