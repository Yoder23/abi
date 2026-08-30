from __future__ import annotations

import gzip
import io
import json
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
    value = _write(tmp_path / "opaque.bin", b"prefix" + b"7z\xbc\xaf'\x1c" + b"opaque")
    assert value["unsupported_signatures"]
