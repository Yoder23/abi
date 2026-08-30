"""Build and execute a physically isolated ABI host-certification capsule.

The certification worker is intentionally incapable of seeing capability
packages, source-success locks, matrix observations, or development evidence.
The parent process builds an allowlisted capsule containing only the canonical
ABI, the generic conformance corpus, certification code, and one host.  The
worker executes after ``pivot_root`` into a tmpfs filesystem containing only
that capsule and read-only runtime dependencies.  The old root, Windows
mounts, network namespace, and every development path are unreachable.
"""

from __future__ import annotations

import argparse
import bz2
import ctypes
import ctypes.util
import hashlib
import io
import json
import lzma
import os
import re
import shutil
import site
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
import zipfile
import zlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_json_bytes, sha256_bytes

CAPSULE_FORMAT = "abi-v2-physical-certification-capsule/1"
ISOLATION_FORMAT = "abi-v2-physical-certification-isolation/2"
FILESYSTEM_INVENTORY_FORMAT = "abi-v2-reachable-filesystem-inventory/2"
FORBIDDEN_SUFFIXES = {".abi", ".cake", ".abix", ".abicir"}
ALLOWED_CLASSIFICATIONS = {
    "abi_specification",
    "adapter_certification_code",
    "generic_certification_corpus",
    "host_code",
    "host_checkpoint",
}

# This deliberately describes the *shape* of a campaign identifier without
# embedding any successful identifier or success ledger in the blind capsule.
# The longest current IDs are far below the retained streaming overlap.
CAMPAIGN_IDENTIFIER_PATTERN = re.compile(
    rb"(?<![a-z0-9])(?:[a-z][a-z0-9]*-){2,}[0-9]{3,4}-v[0-9]+(?![a-z0-9])",
    flags=re.IGNORECASE,
)
CONTENT_SCAN_OVERLAP = 1024
CONTENT_SCAN_BLOCK_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_RECURSION_DEPTH = 8
MAX_ARCHIVE_MEMBER_BYTES = 2 * 1024 * 1024 * 1024

# Containers for which the frozen certification runtime has no safe standard-
# library decoder are rejected if their magic is present anywhere in an
# admitted file or expanded member. This is deliberately content based.
UNSUPPORTED_ARCHIVE_MAGICS = {
    b"7z\xbc\xaf'\x1c": "7z",
    b"Rar!\x1a\x07": "rar",
    b"(\xb5/\xfd": "zstd",
}


class IsolatedCertificationError(RuntimeError):
    """Raised when certification is not physically capability-blind."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _copy(
    records: list[dict[str, Any]],
    *,
    source: Path,
    capsule: Path,
    relative: str,
    classification: str,
) -> None:
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise IsolatedCertificationError(f"invalid capsule classification: {classification}")
    if not source.is_file():
        raise IsolatedCertificationError(f"required capsule input missing: {source}")
    target = capsule / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=True)
    records.append(
        {
            "path": Path(relative).as_posix(),
            "classification": classification,
            "bytes": target.stat().st_size,
            "sha256": _sha256_file(target),
        }
    )


def build_capsule(
    root: Path,
    *,
    host_key: str,
    destination: Path,
    snapshot: Path | None = None,
) -> dict[str, Any]:
    """Create one immutable, allowlisted certification filesystem."""

    root = root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise IsolatedCertificationError(f"capsule destination already exists: {destination}")
    if host_key not in {"layercake", "qwen2", "pythia"}:
        raise IsolatedCertificationError(f"unsupported host: {host_key}")
    destination.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    code = (
        "abi_v2/__init__.py",
        "abi_v2/canonical.py",
        "abi_v2/certification_pivot_runner.sh",
        "abi_v2/host_certification.py",
        "abi_v2/isolated_certification.py",
    )
    for relative in code:
        _copy(
            records,
            source=root / relative,
            capsule=destination,
            relative=f"abi_release/{relative}",
            classification="adapter_certification_code",
        )
    for relative in ("abi_v2/canonical_spec.json", "abi_v2/conformance_suite.json"):
        _copy(
            records,
            source=root / relative,
            capsule=destination,
            relative=f"abi_release/{relative}",
            classification="abi_specification",
        )
    _copy(
        records,
        source=root / "abi_v2/reference_vectors/core_vectors.json",
        capsule=destination,
        relative="abi_release/abi_v2/reference_vectors/core_vectors.json",
        classification="generic_certification_corpus",
    )
    if host_key == "layercake":
        _copy(
            records,
            source=(
                root.parent
                / "layercake_release/layercake_extensions/route_isolated_clarification_core_v25.py"
            ),
            capsule=destination,
            relative=(
                "layercake_release/layercake_extensions/"
                "route_isolated_clarification_core_v25.py"
            ),
            classification="host_code",
        )
    else:
        if snapshot is None or not snapshot.resolve().is_dir():
            raise IsolatedCertificationError(f"{host_key} requires an exact snapshot")
        for source in sorted(snapshot.resolve().iterdir(), key=lambda value: value.name):
            if source.is_file():
                _copy(
                    records,
                    source=source,
                    capsule=destination,
                    relative=f"abi_release/host_snapshot/{source.name}",
                    classification="host_checkpoint",
                )
    if any(Path(row["path"]).suffix.casefold() in FORBIDDEN_SUFFIXES for row in records):
        raise IsolatedCertificationError("capability archive entered certification capsule")
    if any("source_success" in str(row["path"]).casefold() for row in records):
        raise IsolatedCertificationError("source-success data entered certification capsule")
    manifest: dict[str, Any] = {
        "format": CAPSULE_FORMAT,
        "host_key": host_key,
        "files": sorted(records, key=lambda row: row["path"]),
        "allowed_classifications": sorted(ALLOWED_CLASSIFICATIONS),
        "capability_archives_included": 0,
        "source_success_ledgers_included": 0,
    }
    manifest["evidence_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    _write_json(destination / "abi_release/certification-capsule-manifest.json", manifest)
    return manifest


def _capsule_inventory(capsule: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(capsule.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(capsule).as_posix()
        if relative == "abi_release/certification-capsule-manifest.json":
            continue
        if relative.startswith("abi_release/output/"):
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def verify_capsule(capsule: Path) -> dict[str, Any]:
    capsule = capsule.resolve()
    manifest_path = capsule / "abi_release/certification-capsule-manifest.json"
    if not manifest_path.is_file():
        raise IsolatedCertificationError("certification capsule manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != CAPSULE_FORMAT:
        raise IsolatedCertificationError("certification capsule format changed")
    expected_hash = manifest.get("evidence_sha256")
    payload = {key: value for key, value in manifest.items() if key != "evidence_sha256"}
    if expected_hash != sha256_bytes(canonical_json_bytes(payload)):
        raise IsolatedCertificationError("certification capsule manifest hash changed")
    if manifest.get("allowed_classifications") != sorted(ALLOWED_CLASSIFICATIONS):
        raise IsolatedCertificationError("certification capsule classifications changed")
    expected = {
        row["path"]: {"bytes": int(row["bytes"]), "sha256": str(row["sha256"])}
        for row in manifest.get("files", [])
        if row.get("classification") in ALLOWED_CLASSIFICATIONS
    }
    if len(expected) != len(manifest.get("files", [])):
        raise IsolatedCertificationError("unclassified file in certification capsule")
    actual_rows = _capsule_inventory(capsule)
    actual = {
        row["path"]: {"bytes": int(row["bytes"]), "sha256": str(row["sha256"])}
        for row in actual_rows
    }
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            path for path in set(actual) & set(expected) if actual[path] != expected[path]
        )
        raise IsolatedCertificationError(
            f"certification capsule inventory changed: missing={missing}, extra={extra}, changed={changed}"
        )
    capability_archives = [
        row["path"]
        for row in actual_rows
        if Path(row["path"]).suffix.casefold() in FORBIDDEN_SUFFIXES
    ]
    source_success_ledgers = [
        row["path"] for row in actual_rows if "source_success" in row["path"].casefold()
    ]
    forbidden = [
        row["path"]
        for row in actual_rows
        if Path(row["path"]).suffix.casefold() in FORBIDDEN_SUFFIXES
        or "source_success" in row["path"].casefold()
    ]
    if forbidden:
        raise IsolatedCertificationError(f"forbidden certification payload present: {forbidden}")
    return {
        "manifest_sha256": _sha256_file(manifest_path),
        "files_verified": len(actual_rows),
        "inventory_sha256": sha256_bytes(canonical_json_bytes(actual_rows)),
        "capability_archives_present": len(capability_archives),
        "source_success_ledgers_present": len(source_success_ledgers),
        "inventory": actual_rows,
    }


def _decode_mount_path(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _mount_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.split()
        if "-" not in fields or len(fields) < 10:
            raise IsolatedCertificationError("malformed kernel mount record")
        separator = fields.index("-")
        if separator + 2 >= len(fields):
            raise IsolatedCertificationError("incomplete kernel mount record")
        rows.append(
            {
                "mount_point": _decode_mount_path(fields[4]),
                "mount_options": fields[5].split(","),
                "filesystem_type": fields[separator + 1],
                "source": fields[separator + 2],
                "super_options": fields[separator + 3].split(",")
                if separator + 3 < len(fields)
                else [],
            }
        )
    return rows


def _mount_state() -> dict[str, Any]:
    mountinfo = Path("/proc/self/mountinfo")
    text = mountinfo.read_text(encoding="utf-8") if mountinfo.is_file() else ""
    rows = _mount_rows(text)
    runtime_site = os.environ.get("PYTHONPATH", "")
    allowed_exact = {
        "/",
        "/capsule",
        "/dev",
        "/dev/null",
        "/dev/random",
        "/dev/urandom",
        "/dev/zero",
        "/etc",
        "/proc",
        "/usr",
        runtime_site,
    }
    unexpected = sorted(
        row["mount_point"]
        for row in rows
        if row["mount_point"] not in allowed_exact
    )
    required = {"/", "/capsule", "/dev", "/etc", "/proc", "/usr", runtime_site}
    present = {row["mount_point"] for row in rows}
    root_rows = [row for row in rows if row["mount_point"] == "/"]
    capsule_rows = [row for row in rows if row["mount_point"] == "/capsule"]
    if (
        os.environ.get("ABI_CERTIFICATION_PIVOT_ROOT") != "1"
        or unexpected
        or required - present
        or len(root_rows) != 1
        or root_rows[0]["filesystem_type"] != "tmpfs"
        or len(capsule_rows) != 1
        or Path("/oldroot").exists()
        or Path("/mnt/c").exists()
    ):
        raise IsolatedCertificationError(
            "worker filesystem is not the exact pivot-root certification sandbox"
        )
    return {
        "platform": sys.platform,
        "mountinfo_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "sandbox_policy": "abi-certification-pivot-root/1",
        "runtime_site_mount": runtime_site,
        "allowed_mount_points": sorted(allowed_exact),
        "mounts": rows,
        "unexpected_mount_points": unexpected,
        "old_root_present": Path("/oldroot").exists(),
        "windows_mount_present": Path("/mnt/c").exists(),
    }


def _stream_campaign_identifier_count(handle: Any) -> tuple[int, int]:
    """Scan every byte in one stream while retaining boundary matches."""

    scanned = 0
    matches = 0
    overlap = b""
    while True:
        block = handle.read(CONTENT_SCAN_BLOCK_BYTES)
        if not block:
            break
        scanned += len(block)
        combined = overlap + block
        matches += sum(
            int(match.end() > len(overlap))
            for match in CAMPAIGN_IDENTIFIER_PATTERN.finditer(combined)
        )
        overlap = combined[-CONTENT_SCAN_OVERLAP:]
    return scanned, matches


def _empty_archive_scan() -> dict[str, Any]:
    return {
        "signatures": [],
        "members_scanned": 0,
        "member_bytes_scanned": 0,
        "member_identifier_matches": 0,
        "forbidden_members": [],
        "maximum_depth": 0,
        "unsupported_signatures": [],
    }


def _merge_archive_scan(target: dict[str, Any], child: Mapping[str, Any]) -> None:
    target["signatures"].extend(child["signatures"])
    target["members_scanned"] += int(child["members_scanned"])
    target["member_bytes_scanned"] += int(child["member_bytes_scanned"])
    target["member_identifier_matches"] += int(child["member_identifier_matches"])
    target["forbidden_members"].extend(child["forbidden_members"])
    target["maximum_depth"] = max(int(target["maximum_depth"]), int(child["maximum_depth"]))
    target["unsupported_signatures"].extend(child["unsupported_signatures"])


def _read_bounded(handle: Any, *, label: str) -> bytes:
    payload = handle.read(MAX_ARCHIVE_MEMBER_BYTES + 1)
    if len(payload) > MAX_ARCHIVE_MEMBER_BYTES:
        raise IsolatedCertificationError(f"expanded archive member exceeds bound: {label}")
    return payload


def _archive_member_families(names: set[str]) -> list[tuple[str, str]]:
    """Recognize capability layouts at any directory prefix."""

    normalized = {name.replace("\\", "/").lstrip("/") for name in names}
    by_parent: dict[str, set[str]] = {}
    for name in normalized:
        parent, _, base = name.rpartition("/")
        by_parent.setdefault(parent, set()).add(base)
    families: list[tuple[str, str]] = []
    cake = {"manifest.json", "tensors.safetensors", "signature.json"}
    extraction = {"manifest.json", "inventory.json", "ledger.json", "records.jsonl"}
    abix = {"budgets.json", "segregation.json", "selection.json", "sources.json"}
    abicir = {"accounting.json", "normalization.json", "source_identity.json", "split_manifest.json"}
    for parent, bases in by_parent.items():
        if cake <= bases:
            families.append(("layercake-capability-package", parent))
        if extraction <= bases and abix <= bases:
            families.append(("abi-extraction-archive", parent))
        if extraction <= bases and abicir <= bases:
            families.append(("abi-normalized-ir-archive", parent))
    return families


def _valid_7z_stream(payload: bytes, offset: int) -> bool:
    """Validate the fixed 7z signature header and next-header CRC."""

    if offset + 32 > len(payload):
        return False
    header = payload[offset : offset + 32]
    if header[:6] != b"7z\xbc\xaf'\x1c":
        return False
    if zlib.crc32(header[12:32]) & 0xFFFFFFFF != int.from_bytes(header[8:12], "little"):
        return False
    next_offset = int.from_bytes(header[12:20], "little")
    next_size = int.from_bytes(header[20:28], "little")
    start = offset + 32 + next_offset
    end = start + next_size
    return (
        next_size > 0
        and start >= offset + 32
        and end <= len(payload)
        and zlib.crc32(payload[start:end]) & 0xFFFFFFFF
        == int.from_bytes(header[28:32], "little")
    )


def _valid_rar4_stream(payload: bytes, offset: int) -> bool:
    """Validate a RAR4 marker/header rather than trusting an incidental literal."""

    signature = b"Rar!\x1a\x07\x00"
    header_start = offset + len(signature)
    if payload[offset:header_start] != signature or header_start + 7 > len(payload):
        return False
    header_size = int.from_bytes(payload[header_start + 5 : header_start + 7], "little")
    header_end = header_start + header_size
    if header_size < 7 or header_end > len(payload):
        return False
    expected_crc = int.from_bytes(payload[header_start : header_start + 2], "little")
    actual_crc = zlib.crc32(payload[header_start + 2 : header_end]) & 0xFFFF
    return expected_crc == actual_crc and 0x72 <= payload[header_start + 2] <= 0x7B


def _read_rar5_vint(payload: bytes, position: int, limit: int) -> tuple[int, int] | None:
    value = 0
    shift = 0
    while position < limit and shift <= 63:
        byte = payload[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7
    return None


def _valid_rar5_stream(payload: bytes, offset: int) -> bool:
    """Validate the RAR5 signature, bounded header size, and header CRC32."""

    signature = b"Rar!\x1a\x07\x01\x00"
    crc_start = offset + len(signature)
    if payload[offset:crc_start] != signature or crc_start + 5 > len(payload):
        return False
    expected_crc = int.from_bytes(payload[crc_start : crc_start + 4], "little")
    size_start = crc_start + 4
    decoded = _read_rar5_vint(payload, size_start, min(len(payload), size_start + 10))
    if decoded is None:
        return False
    header_size, body_start = decoded
    header_end = body_start + header_size
    return (
        header_size > 0
        and header_end <= len(payload)
        and zlib.crc32(payload[size_start:header_end]) & 0xFFFFFFFF == expected_crc
    )


def _valid_zstd_frame(payload: bytes, offset: int) -> bool:
    """Validate Zstandard frame structure without decoding its content."""

    if payload[offset : offset + 4] != b"(\xb5/\xfd" or offset + 6 > len(payload):
        return False
    position = offset + 4
    descriptor = payload[position]
    position += 1
    if descriptor & 0x18:  # reserved and unused bits must both be zero
        return False
    single_segment = bool(descriptor & 0x20)
    checksum = bool(descriptor & 0x04)
    dictionary_flag = descriptor & 0x03
    content_size_flag = descriptor >> 6
    if not single_segment:
        if position >= len(payload):
            return False
        position += 1  # window descriptor
    dictionary_size = (0, 1, 2, 4)[dictionary_flag]
    content_size_size = (
        1 if single_segment else 0,
        2,
        4,
        8,
    )[content_size_flag]
    position += dictionary_size + content_size_size
    if position > len(payload):
        return False
    while position + 3 <= len(payload):
        block_header = int.from_bytes(payload[position : position + 3], "little")
        position += 3
        last_block = bool(block_header & 1)
        block_type = (block_header >> 1) & 0x03
        block_size = block_header >> 3
        if block_type == 3:
            return False
        stored_size = 1 if block_type == 1 else block_size
        position += stored_size
        if position > len(payload):
            return False
        if last_block:
            if checksum:
                position += 4
            return position <= len(payload)
    return False


def _unsupported_archive_streams(payload: bytes, *, label: str) -> list[str]:
    """Return only structurally valid unsupported containers at any offset."""

    findings: list[str] = []
    validators = (
        (b"7z\xbc\xaf'\x1c", "7z", _valid_7z_stream),
        (b"Rar!\x1a\x07\x00", "rar4", _valid_rar4_stream),
        (b"Rar!\x1a\x07\x01\x00", "rar5", _valid_rar5_stream),
    )
    for magic, kind, validator in validators:
        offset = payload.find(magic)
        while offset >= 0:
            if validator(payload, offset):
                findings.append(f"{label}:{kind}@{offset}")
            offset = payload.find(magic, offset + 1)
    return findings


def _valid_gzip_header(payload: bytes, offset: int) -> bool:
    if payload[offset : offset + 2] != b"\x1f\x8b" or offset + 18 > len(payload):
        return False
    if payload[offset + 2] != 8 or payload[offset + 3] & 0xE0:
        return False
    flags = payload[offset + 3]
    position = offset + 10
    if flags & 0x04:
        if position + 2 > len(payload):
            return False
        extra_size = int.from_bytes(payload[position : position + 2], "little")
        position += 2 + extra_size
    for flag in (0x08, 0x10):
        if flags & flag:
            terminator = payload.find(b"\x00", position)
            if terminator < 0:
                return False
            position = terminator + 1
    if flags & 0x02:
        if position + 2 > len(payload):
            return False
        expected_crc = int.from_bytes(payload[position : position + 2], "little")
        if zlib.crc32(payload[offset:position]) & 0xFFFF != expected_crc:
            return False
        position += 2
    return position + 8 <= len(payload)


def _valid_bzip2_header(payload: bytes, offset: int) -> bool:
    return (
        offset + 10 <= len(payload)
        and payload[offset : offset + 3] == b"BZh"
        and payload[offset + 3 : offset + 4] in b"123456789"
        and payload[offset + 4 : offset + 10]
        in {b"1AY&SY", b"\x17rE8P\x90"}
    )


def _valid_xz_header(payload: bytes, offset: int) -> bool:
    if payload[offset : offset + 6] != b"\xfd7zXZ\x00" or offset + 12 > len(payload):
        return False
    flags = payload[offset + 6 : offset + 8]
    return (
        flags[0] == 0
        and flags[1] & 0xF0 == 0
        and zlib.crc32(flags) & 0xFFFFFFFF
        == int.from_bytes(payload[offset + 8 : offset + 12], "little")
    )


def _decompress_supported_stream(
    payload: bytes, *, offset: int, kind: str
) -> bytes | None:
    """Decode one supported stream at an arbitrary offset with an output cap."""

    source = payload[offset:]
    try:
        if kind == "gzip":
            decompressor: Any = zlib.decompressobj(wbits=31)
        elif kind == "bzip2":
            decompressor = bz2.BZ2Decompressor()
        elif kind == "xz":
            decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
        else:
            raise IsolatedCertificationError(f"unknown supported archive stream: {kind}")
        expanded = decompressor.decompress(source, max_length=MAX_ARCHIVE_MEMBER_BYTES + 1)
    except (OSError, EOFError, ValueError, lzma.LZMAError, zlib.error):
        return None
    if len(expanded) > MAX_ARCHIVE_MEMBER_BYTES or not decompressor.eof:
        return None
    return expanded


def _zstd_library() -> Any | None:
    name = ctypes.util.find_library("zstd")
    if not name:
        return None
    try:
        library = ctypes.CDLL(name)
    except OSError:
        return None
    size_t = ctypes.c_size_t
    void_pointer = ctypes.c_void_p
    library.ZSTD_createDStream.argtypes = []
    library.ZSTD_createDStream.restype = void_pointer
    library.ZSTD_freeDStream.argtypes = [void_pointer]
    library.ZSTD_freeDStream.restype = size_t
    library.ZSTD_initDStream.argtypes = [void_pointer]
    library.ZSTD_initDStream.restype = size_t
    library.ZSTD_DStreamOutSize.argtypes = []
    library.ZSTD_DStreamOutSize.restype = size_t
    library.ZSTD_isError.argtypes = [size_t]
    library.ZSTD_isError.restype = ctypes.c_uint
    return library


class _ZstdInputBuffer(ctypes.Structure):
    _fields_ = [
        ("src", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]


class _ZstdOutputBuffer(ctypes.Structure):
    _fields_ = [
        ("dst", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("pos", ctypes.c_size_t),
    ]


def _decompress_zstd_frame(payload: bytes, offset: int) -> tuple[str, bytes | None]:
    """Streaming-decode one zstd frame and distinguish malformed literals."""

    if not _valid_zstd_frame(payload, offset):
        return "invalid", None
    library = _zstd_library()
    if library is None:
        return "unsupported", None
    library.ZSTD_decompressStream.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ZstdOutputBuffer),
        ctypes.POINTER(_ZstdInputBuffer),
    ]
    library.ZSTD_decompressStream.restype = ctypes.c_size_t
    source = payload[offset:]
    source_buffer = ctypes.create_string_buffer(source)
    incoming = _ZstdInputBuffer(
        ctypes.cast(source_buffer, ctypes.c_void_p),
        len(source),
        0,
    )
    stream = library.ZSTD_createDStream()
    if not stream:
        return "unsupported", None
    expanded = bytearray()
    try:
        initialized = int(library.ZSTD_initDStream(stream))
        if library.ZSTD_isError(initialized):
            return "unsupported", None
        chunk_size = max(64 * 1024, int(library.ZSTD_DStreamOutSize()))
        while True:
            destination = ctypes.create_string_buffer(chunk_size)
            outgoing = _ZstdOutputBuffer(
                ctypes.cast(destination, ctypes.c_void_p),
                chunk_size,
                0,
            )
            before = int(incoming.pos)
            remaining = int(library.ZSTD_decompressStream(stream, outgoing, incoming))
            if library.ZSTD_isError(remaining):
                return "invalid", None
            expanded.extend(destination.raw[: int(outgoing.pos)])
            if len(expanded) > MAX_ARCHIVE_MEMBER_BYTES:
                return "unsupported", None
            if remaining == 0:
                return "decoded", bytes(expanded)
            if int(incoming.pos) == before and int(outgoing.pos) == 0:
                return "invalid", None
            if int(incoming.pos) >= int(incoming.size) and remaining > 0:
                return "invalid", None
    finally:
        library.ZSTD_freeDStream(stream)


def _scan_expanded_payload(payload: bytes, *, label: str, depth: int) -> dict[str, Any]:
    """Recursively content-scan readable ZIP/tar/gzip/bzip2/xz payloads."""

    if depth > MAX_ARCHIVE_RECURSION_DEPTH:
        raise IsolatedCertificationError(f"archive recursion depth exceeded: {label}")
    result = _empty_archive_scan()
    result["maximum_depth"] = depth
    result["member_identifier_matches"] = sum(
        1 for _ in CAMPAIGN_IDENTIFIER_PATTERN.finditer(payload)
    )
    result["unsupported_signatures"].extend(
        _unsupported_archive_streams(payload, label=label)
    )

    source = io.BytesIO(payload)
    if zipfile.is_zipfile(source):
        source.seek(0)
        try:
            with zipfile.ZipFile(source) as archive:
                infos = [info for info in archive.infolist() if not info.is_dir()]
                names = {info.filename for info in infos}
                for family, parent in _archive_member_families(names):
                    result["signatures"].append(f"{label}:{parent or '.'}:{family}")
                for info in infos:
                    member_label = f"{label}!/{info.filename}"
                    result["members_scanned"] += 1
                    result["forbidden_members"].extend(
                        [member_label]
                        if Path(info.filename).suffix.casefold() in FORBIDDEN_SUFFIXES
                        else []
                    )
                    if info.flag_bits & 1:
                        result["unsupported_signatures"].append(f"{member_label}:encrypted-zip")
                        continue
                    if int(info.file_size) > MAX_ARCHIVE_MEMBER_BYTES:
                        raise IsolatedCertificationError(
                            f"expanded archive member exceeds bound: {member_label}"
                        )
                    with archive.open(info, "r") as member:
                        member_payload = _read_bounded(member, label=member_label)
                    result["member_bytes_scanned"] += len(member_payload)
                    child = _scan_expanded_payload(
                        member_payload, label=member_label, depth=depth + 1
                    )
                    _merge_archive_scan(result, child)
        except (OSError, zipfile.BadZipFile, RuntimeError):
            # is_zipfile can false-positive on binary blobs that happen to
            # contain ZIP trailers (for example CPython bytecode). Such bytes
            # are still fully hashed/scanned, but are not a readable archive.
            return result
        return result

    # A tar header is authoritative at byte 257. Compressed single streams are
    # decoded and recursively inspected, so tar.gz/tar.bz2/tar.xz and nested
    # containers cannot hide capability layouts or campaign identifiers.
    if len(payload) >= 262 and payload[257:262] == b"ustar":
        try:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
                infos = [info for info in archive.getmembers() if info.isfile()]
                names = {info.name for info in infos}
                for family, parent in _archive_member_families(names):
                    result["signatures"].append(f"{label}:{parent or '.'}:{family}")
                for info in infos:
                    member_label = f"{label}!/{info.name}"
                    result["members_scanned"] += 1
                    if Path(info.name).suffix.casefold() in FORBIDDEN_SUFFIXES:
                        result["forbidden_members"].append(member_label)
                    if int(info.size) > MAX_ARCHIVE_MEMBER_BYTES:
                        raise IsolatedCertificationError(
                            f"expanded archive member exceeds bound: {member_label}"
                        )
                    extracted = archive.extractfile(info)
                    if extracted is None:
                        raise IsolatedCertificationError(f"tar member unreadable: {member_label}")
                    with extracted:
                        member_payload = _read_bounded(extracted, label=member_label)
                    result["member_bytes_scanned"] += len(member_payload)
                    _merge_archive_scan(
                        result,
                        _scan_expanded_payload(
                            member_payload, label=member_label, depth=depth + 1
                        ),
                    )
        except (OSError, tarfile.TarError) as exc:
            raise IsolatedCertificationError(f"readable tar expansion failed: {label}") from exc
        return result

    decompressors: tuple[tuple[bytes, str, Any], ...] = (
        (b"\x1f\x8b", "gzip", _valid_gzip_header),
        (b"BZh", "bzip2", _valid_bzip2_header),
        (b"\xfd7zXZ\x00", "xz", _valid_xz_header),
    )
    expanded_offsets: set[tuple[str, int]] = set()
    for magic, kind, validator in decompressors:
        offset = payload.find(magic)
        while offset >= 0:
            if not validator(payload, offset):
                offset = payload.find(magic, offset + 1)
                continue
            expanded = _decompress_supported_stream(payload, offset=offset, kind=kind)
            if expanded is None:
                # Header-shaped bytes can still occur incidentally. Only a
                # stream that reaches a valid end is treated as a container.
                offset = payload.find(magic, offset + 1)
                continue
            if len(expanded) > MAX_ARCHIVE_MEMBER_BYTES:
                raise IsolatedCertificationError(
                    f"expanded {kind} stream exceeds bound: {label}@{offset}"
                )
            identity = (hashlib.sha256(expanded).hexdigest(), len(expanded))
            if identity not in expanded_offsets:
                expanded_offsets.add(identity)
                result["members_scanned"] += 1
                result["member_bytes_scanned"] += len(expanded)
                _merge_archive_scan(
                    result,
                    _scan_expanded_payload(
                        expanded,
                        label=f"{label}!/{kind}@{offset}",
                        depth=depth + 1,
                    ),
                )
            offset = payload.find(magic, offset + 1)

    zstd_magic = b"(\xb5/\xfd"
    offset = payload.find(zstd_magic)
    while offset >= 0:
        if _valid_zstd_frame(payload, offset):
            zstd_state, expanded = _decompress_zstd_frame(payload, offset)
            if zstd_state == "unsupported":
                result["unsupported_signatures"].append(f"{label}:zstd@{offset}")
            elif zstd_state == "decoded" and expanded is not None:
                identity = (hashlib.sha256(expanded).hexdigest(), len(expanded))
                if identity not in expanded_offsets:
                    expanded_offsets.add(identity)
                    result["members_scanned"] += 1
                    result["member_bytes_scanned"] += len(expanded)
                    _merge_archive_scan(
                        result,
                        _scan_expanded_payload(
                            expanded,
                            label=f"{label}!/zstd@{offset}",
                            depth=depth + 1,
                        ),
                    )
        offset = payload.find(zstd_magic, offset + 1)
    return result


def _capability_archive_signatures(path: Path) -> dict[str, Any]:
    """Inspect renamed, prefixed, compressed, and recursively nested containers."""

    if zipfile.is_zipfile(path):
        # zipfile handles arbitrary prefixes/self-extracting ZIPs. Reading here
        # is bounded member-by-member by the recursive scanner.
        with path.open("rb") as handle:
            payload = _read_bounded(handle, label=path.as_posix())
        return _scan_expanded_payload(payload, label=path.as_posix(), depth=1)

    # A container may be hidden behind an arbitrary executable/data prefix.
    # Inspect the complete byte stream for every supported or fail-closed
    # signature before deciding that a regular file is not a container.  Keep
    # enough overlap to detect a signature split across scan blocks.
    magics = tuple(UNSUPPORTED_ARCHIVE_MAGICS) + (
        b"\x1f\x8b",
        b"BZh",
        b"\xfd7zXZ\x00",
    )
    maximum_magic = max(len(magic) for magic in magics)
    overlap = b""
    recognized = False
    with path.open("rb") as handle:
        while True:
            block = handle.read(CONTENT_SCAN_BLOCK_BYTES)
            if not block:
                break
            window = overlap + block
            if any(magic in window for magic in magics):
                recognized = True
                break
            overlap = window[-(maximum_magic - 1) :]

    # Uncompressed tar has its authoritative ustar marker at byte 257.  This
    # check is deliberately separate because incidental "ustar" bytes later
    # in an ordinary file do not make it a tar stream.
    if not recognized:
        with path.open("rb") as handle:
            prefix = handle.read(262)
        recognized = len(prefix) >= 262 and prefix[257:262] == b"ustar"
    if not recognized:
        return _empty_archive_scan()
    with path.open("rb") as handle:
        payload = _read_bounded(handle, label=path.as_posix())
    return _scan_expanded_payload(payload, label=path.as_posix(), depth=1)


def _inspect_regular_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    content_bytes = 0
    identifier_matches = 0
    overlap = b""
    with path.open("rb") as handle:
        while True:
            block = handle.read(CONTENT_SCAN_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            content_bytes += len(block)
            combined = overlap + block
            identifier_matches += sum(
                int(match.end() > len(overlap))
                for match in CAMPAIGN_IDENTIFIER_PATTERN.finditer(combined)
            )
            overlap = combined[-CONTENT_SCAN_OVERLAP:]
    archive = _capability_archive_signatures(path)
    return {
        "bytes": content_bytes,
        "sha256": digest.hexdigest(),
        "content_bytes_scanned": content_bytes,
        "campaign_identifier_matches": identifier_matches,
        "embedded_archive_members_scanned": archive["members_scanned"],
        "embedded_archive_uncompressed_bytes_scanned": archive["member_bytes_scanned"],
        "embedded_campaign_identifier_matches": archive["member_identifier_matches"],
        "embedded_archive_maximum_depth": archive["maximum_depth"],
        "unsupported_archive_signatures": sorted(set(archive["unsupported_signatures"])),
        "capability_archive_signatures": sorted(set(archive["signatures"])),
        "forbidden_archive_member_paths": sorted(set(archive["forbidden_members"])),
    }


def _mount_classification(path: str, runtime_site: str) -> str:
    if path == "/capsule" or path.startswith("/capsule/"):
        return "certification_capsule"
    if path == "/usr" or path.startswith("/usr/"):
        return "python_system_runtime"
    if path == "/etc" or path.startswith("/etc/"):
        return "system_runtime_configuration"
    if path == runtime_site or path.startswith(runtime_site.rstrip("/") + "/"):
        return "python_package_runtime"
    if path in {"/bin", "/sbin", "/lib", "/lib64"}:
        return "runtime_alias"
    raise IsolatedCertificationError(f"unclassified reachable filesystem entry: {path}")


def _physical_forbidden_scan() -> tuple[dict[str, Any], bytes]:
    """Hash and content-scan every reachable non-virtual filesystem entry."""

    runtime_site = os.environ.get("PYTHONPATH", "")
    if not runtime_site.startswith("/home/") or not runtime_site.endswith("/site-packages"):
        raise IsolatedCertificationError("runtime site is unavailable for physical inventory")
    rows: list[dict[str, Any]] = []
    directories_scanned = 0
    special_entries: list[str] = []
    forbidden_path_matches: list[str] = []
    excluded_virtual_roots = {"/dev", "/proc"}
    allowed_symlink_roots = ("/usr", "/etc", runtime_site, "/dev", "/proc", "/capsule")

    def inspect_path(path: Path) -> None:
        normalized = path.as_posix()
        lowered = normalized.casefold()
        if (
            path.suffix.casefold() in FORBIDDEN_SUFFIXES
            or "source_" + "success" in lowered
            or CAMPAIGN_IDENTIFIER_PATTERN.search(normalized.encode("utf-8"))
        ):
            forbidden_path_matches.append(normalized)
        mode = os.lstat(path).st_mode
        classification = _mount_classification(normalized, runtime_site)
        if stat.S_ISLNK(mode):
            target = os.readlink(path)
            resolved = Path(os.path.realpath(path)).as_posix()
            target_in_admitted_root = any(
                resolved == root or resolved.startswith(root.rstrip("/") + "/")
                for root in allowed_symlink_roots
            )
            if target_in_admitted_root:
                target_state = "reachable_admitted_root"
            elif not Path(resolved).exists():
                target_state = "absent_in_namespace"
            else:
                raise IsolatedCertificationError(
                    f"symlink escapes admitted filesystem roots: {normalized} -> {resolved}"
                )
            rows.append(
                {
                    "path": normalized,
                    "kind": "symlink",
                    "mount_classification": classification,
                    "link_target": target,
                    "resolved_path": resolved,
                    "target_state": target_state,
                    "sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                }
            )
            return
        if not stat.S_ISREG(mode):
            special_entries.append(normalized)
            return
        inspection = _inspect_regular_file(path)
        rows.append(
            {
                "path": normalized,
                "kind": "regular_file",
                "mount_classification": classification,
                **inspection,
            }
        )

    for directory, names, filenames in os.walk("/", topdown=True, followlinks=False):
        directory_path = Path(directory)
        names[:] = sorted(
            name
            for name in names
            if (directory_path / name).as_posix() not in excluded_virtual_roots
        )
        filenames.sort()
        directories_scanned += 1
        normalized_directory = directory_path.as_posix()
        if CAMPAIGN_IDENTIFIER_PATTERN.search(normalized_directory.encode("utf-8")):
            forbidden_path_matches.append(normalized_directory)
        symlink_directories = [name for name in names if (directory_path / name).is_symlink()]
        for name in symlink_directories:
            inspect_path(directory_path / name)
        names[:] = [name for name in names if name not in symlink_directories]
        for name in filenames:
            inspect_path(directory_path / name)

    rows.sort(key=lambda row: row["path"])
    duplicate_paths = len(rows) != len({row["path"] for row in rows})
    regular_rows = [row for row in rows if row["kind"] == "regular_file"]
    symlink_rows = [row for row in rows if row["kind"] == "symlink"]
    capability_signatures = sum(
        len(row["capability_archive_signatures"]) for row in regular_rows
    )
    forbidden_members = sum(
        len(row["forbidden_archive_member_paths"]) for row in regular_rows
    )
    campaign_matches = sum(
        int(row["campaign_identifier_matches"])
        + int(row["embedded_campaign_identifier_matches"])
        for row in regular_rows
    )
    unsupported_archives = sum(
        len(row["unsupported_archive_signatures"]) for row in regular_rows
    )
    unsupported_archive_examples = [
        signature
        for row in regular_rows
        for signature in row["unsupported_archive_signatures"]
    ][:30]
    if (
        duplicate_paths
        or special_entries
        or forbidden_path_matches
        or capability_signatures
        or forbidden_members
        or campaign_matches
        or unsupported_archives
    ):
        raise IsolatedCertificationError(
            "forbidden or unaccounted content is reachable inside certification root: "
            f"duplicate_paths={duplicate_paths}, special_entries={special_entries[:5]}, "
            f"forbidden_paths={forbidden_path_matches[:5]}, "
            f"capability_signatures={capability_signatures}, "
            f"forbidden_archive_members={forbidden_members}, "
            f"campaign_identifier_matches={campaign_matches}, "
            f"unsupported_archive_signatures={unsupported_archives}, "
            f"unsupported_archive_examples={unsupported_archive_examples}"
        )
    inventory_bytes = b"".join(canonical_json_bytes(row) for row in rows)
    summary: dict[str, Any] = {
        "format": FILESYSTEM_INVENTORY_FORMAT,
        "scan_root": "/",
        "excluded_virtual_roots": sorted(excluded_virtual_roots),
        "runtime_site_root": runtime_site,
        "directories_scanned": directories_scanned,
        "inventory_rows": len(rows),
        "regular_files_scanned": len(regular_rows),
        "symlinks_scanned": len(symlink_rows),
        "regular_file_bytes": sum(int(row["bytes"]) for row in regular_rows),
        "content_bytes_scanned": sum(
            int(row["content_bytes_scanned"]) for row in regular_rows
        ),
        "embedded_archive_members_scanned": sum(
            int(row["embedded_archive_members_scanned"]) for row in regular_rows
        ),
        "embedded_archive_uncompressed_bytes_scanned": sum(
            int(row["embedded_archive_uncompressed_bytes_scanned"])
            for row in regular_rows
        ),
        "embedded_archive_maximum_depth": max(
            (int(row["embedded_archive_maximum_depth"]) for row in regular_rows),
            default=0,
        ),
        "unsupported_archive_signatures": unsupported_archives,
        "capability_archive_signature_matches": capability_signatures,
        "forbidden_archive_member_paths": forbidden_members,
        "campaign_identifier_matches": campaign_matches,
        "forbidden_path_matches": len(forbidden_path_matches),
        "special_entries": len(special_entries),
        "inventory_jsonl_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
    }
    summary["evidence_sha256"] = sha256_bytes(canonical_json_bytes(summary))
    return summary, inventory_bytes


def run_worker(capsule: Path, *, host_key: str, device: str) -> dict[str, Any]:
    """Execute certification after proving capsule identity and mount isolation."""

    capsule = capsule.resolve()
    root = capsule / "abi_release"
    inventory = verify_capsule(capsule)
    mount = _mount_state()
    forbidden_scan, filesystem_inventory_bytes = _physical_forbidden_scan()
    if sys.platform != "linux" or mount["unexpected_mount_points"]:
        raise IsolatedCertificationError(
            "worker requires the exact Linux pivot-root isolation policy"
        )
    from .host_certification import certify_host

    output = root / "output"
    snapshot = root / "host_snapshot" if host_key != "layercake" else None
    isolation = {
        "format": ISOLATION_FORMAT,
        "host_key": host_key,
        "capsule": inventory,
        "mount": mount,
        "reachable_filesystem_forbidden_scan": forbidden_scan,
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "capsule_files_physically_present": inventory["files_verified"],
        "capability_archives_physically_present": (
            forbidden_scan["capability_archive_signature_matches"]
            + forbidden_scan["forbidden_archive_member_paths"]
        ),
        "source_success_ledgers_physically_present": forbidden_scan[
            "campaign_identifier_matches"
        ],
    }
    isolation["evidence_sha256"] = sha256_bytes(canonical_json_bytes(isolation))
    result = certify_host(
        root,
        host_key=host_key,
        output_dir=output / "certification",
        snapshot=snapshot,
        device=device,
        physical_isolation=isolation,
    )
    _write_json(output / "physical-isolation.json", isolation)
    (output / "reachable-filesystem-inventory.jsonl").write_bytes(
        filesystem_inventory_bytes
    )
    shutil.copy2(
        root / "certification-capsule-manifest.json",
        output / "certification-capsule-manifest.json",
    )
    mountinfo = Path("/proc/self/mountinfo")
    mountinfo_bytes = mountinfo.read_bytes() if mountinfo.is_file() else b""
    (output / "mountinfo.txt").write_bytes(mountinfo_bytes)
    receipt: dict[str, Any] = {
        "format": "abi-v2-isolated-host-certification-receipt/1",
        "host_key": host_key,
        "device": device,
        "isolation_evidence_sha256": _sha256_file(output / "physical-isolation.json"),
        "result_sha256": _sha256_file(output / "certification/result.json"),
        "adapter_sha256": _sha256_file(output / "certification/adapter.json"),
        "performance_sha256": _sha256_file(output / "certification/performance.json"),
        "capsule_manifest_sha256": _sha256_file(
            output / "certification-capsule-manifest.json"
        ),
        "mountinfo_sha256": _sha256_file(output / "mountinfo.txt"),
        "reachable_filesystem_inventory_sha256": _sha256_file(
            output / "reachable-filesystem-inventory.jsonl"
        ),
        "executed_isolated_certification_sha256": _sha256_file(
            root / "abi_v2/isolated_certification.py"
        ),
        "executed_pivot_runner_sha256": _sha256_file(
            root / "abi_v2/certification_pivot_runner.sh"
        ),
        "executed_host_certification_sha256": _sha256_file(
            root / "abi_v2/host_certification.py"
        ),
        "worker_exit_basis": result["status"],
    }
    receipt["evidence_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    _write_json(output / "receipt.json", receipt)
    return receipt


def run_wsl_capsule(
    root: Path,
    *,
    host_key: str,
    destination: Path,
    snapshot: Path | None = None,
    device: str = "cpu",
    distribution: str = "Ubuntu",
) -> dict[str, Any]:
    """Run a capsule in WSL2 after detaching the Windows development drive."""

    root = root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise IsolatedCertificationError(f"immutable output exists: {destination}")
    capsule = Path(os.environ.get("TEMP", str(root))) / (
        f"abi-certification-{host_key}-{uuid.uuid4().hex}"
    )
    build_capsule(root, host_key=host_key, destination=capsule, snapshot=snapshot)

    def wsl_path(path: Path) -> str:
        resolved = path.resolve()
        drive = resolved.drive.rstrip(":").casefold()
        if len(drive) != 1:
            raise IsolatedCertificationError(f"cannot map path into WSL: {resolved}")
        tail = resolved.as_posix().split(":", 1)[1]
        return f"/mnt/{drive}{tail}"

    capsule_source = wsl_path(capsule)
    destination_target = wsl_path(destination)
    runtime_probe = subprocess.run(
        [
            "wsl.exe",
            "-d",
            distribution,
            "--",
            "python3",
            "-c",
            "import site; print(site.getusersitepackages())",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    runtime_site = runtime_probe.stdout.strip()
    if runtime_probe.returncode != 0 or not runtime_site.startswith("/home/"):
        raise IsolatedCertificationError("cannot identify the WSL Python runtime site")
    linux_capsule = f"/tmp/abi-certification-{host_key}-{uuid.uuid4().hex}"
    sandbox_root = f"/tmp/abi-certification-root-{host_key}-{uuid.uuid4().hex}"
    completed = subprocess.run(
        [
            "wsl.exe",
            "-d",
            distribution,
            "-u",
            "root",
            "--",
            "bash",
            "-lc",
            (
                "set -euo pipefail; "
                f"cp -aL '{capsule_source}' '{linux_capsule}'; "
                f"chmod 500 '{linux_capsule}/abi_release/abi_v2/certification_pivot_runner.sh'; "
                f"ABI_CAPSULE_PATH='{linux_capsule}' "
                f"ABI_SANDBOX_ROOT='{sandbox_root}' "
                f"ABI_HOST_KEY='{host_key}' ABI_DEVICE='{device}' "
                f"ABI_RUNTIME_SITE_PATH='{runtime_site}' "
                "unshare --mount --pid --fork --ipc --uts --net "
                "--propagation private "
                f"'{linux_capsule}/abi_release/abi_v2/certification_pivot_runner.sh'; "
                f"mkdir -p '{Path(destination_target).parent.as_posix()}'; "
                f"cp -a '{linux_capsule}/abi_release/output' '{destination_target}'; "
                f"printf '%s' '{linux_capsule}'"
            ),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise IsolatedCertificationError(
            "WSL isolated certification failed "
            f"({completed.returncode}): {completed.stderr[-4000:]}"
        )
    receipt_path = destination / "receipt.json"
    if not receipt_path.is_file():
        raise IsolatedCertificationError("isolated certification receipt missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["launcher"] = {
        "distribution": distribution,
        "exit_code": completed.returncode,
        "sandbox_policy": "abi-certification-pivot-root/1",
        "wsl_capsule_path_sha256": hashlib.sha256(
            completed.stdout.strip().encode("utf-8")
        ).hexdigest(),
        "windows_capsule_path_sha256": hashlib.sha256(
            str(capsule).encode("utf-8")
        ).hexdigest(),
    }
    receipt["evidence_sha256"] = sha256_bytes(
        canonical_json_bytes({key: value for key, value in receipt.items() if key != "evidence_sha256"})
    )
    _write_json(destination / "launcher-receipt.json", receipt)
    return receipt


def run_linux_capsule(
    root: Path,
    *,
    host_key: str,
    destination: Path,
    snapshot: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a capsule in a private Linux mount namespace."""

    root = root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise IsolatedCertificationError(f"immutable output exists: {destination}")
    if root == Path(root.anchor):
        raise IsolatedCertificationError("refusing to mask a filesystem root")
    staging = Path(tempfile.mkdtemp(prefix=f"abi-certification-{host_key}-"))
    capsule = staging / "capsule"
    build_capsule(root, host_key=host_key, destination=capsule, snapshot=snapshot)
    runner = capsule / "abi_release/abi_v2/certification_pivot_runner.sh"
    runner.chmod(0o500)
    sandbox_root = f"/tmp/abi-certification-root-{host_key}-{uuid.uuid4().hex}"
    environment = {
        **os.environ,
        "ABI_CAPSULE_PATH": capsule.as_posix(),
        "ABI_SANDBOX_ROOT": sandbox_root,
        "ABI_HOST_KEY": host_key,
        "ABI_DEVICE": device,
        "ABI_RUNTIME_SITE_PATH": site.getusersitepackages(),
    }
    completed = subprocess.run(
        [
            "unshare",
            "--user",
            "--map-root-user",
            "--mount",
            "--pid",
            "--fork",
            "--ipc",
            "--uts",
            "--net",
            "--propagation",
            "private",
            str(runner),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise IsolatedCertificationError(
            "Linux isolated certification failed "
            f"({completed.returncode}): {completed.stderr[-4000:]}"
        )
    shutil.copytree(capsule / "abi_release/output", destination)
    receipt = json.loads((destination / "receipt.json").read_text(encoding="utf-8"))
    receipt["launcher"] = {
        "kind": "linux-user-mount-namespace",
        "exit_code": completed.returncode,
        "capsule_path_sha256": hashlib.sha256(str(capsule).encode("utf-8")).hexdigest(),
        "masked_release_root_sha256": hashlib.sha256(str(root).encode("utf-8")).hexdigest(),
    }
    receipt["evidence_sha256"] = sha256_bytes(
        canonical_json_bytes({key: value for key, value in receipt.items() if key != "evidence_sha256"})
    )
    _write_json(destination / "launcher-receipt.json", receipt)
    return receipt


def run_isolated_capsule(
    root: Path,
    *,
    host_key: str,
    destination: Path,
    snapshot: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if os.name == "nt":
        return run_wsl_capsule(
            root,
            host_key=host_key,
            destination=destination,
            snapshot=snapshot,
            device=device,
        )
    if sys.platform == "linux":
        return run_linux_capsule(
            root,
            host_key=host_key,
            destination=destination,
            snapshot=snapshot,
            device=device,
        )
    raise IsolatedCertificationError("no supported physical certification sandbox is available")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--host", required=True, choices=("layercake", "qwen2", "pythia"))
    build_parser.add_argument("--destination", required=True)
    build_parser.add_argument("--snapshot")
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--capsule", required=True)
    worker_parser.add_argument("--host", required=True, choices=("layercake", "qwen2", "pythia"))
    worker_parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    run_parser = subparsers.add_parser("run-wsl")
    run_parser.add_argument("--host", required=True, choices=("layercake", "qwen2", "pythia"))
    run_parser.add_argument("--destination", required=True)
    run_parser.add_argument("--snapshot")
    run_parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    if args.command == "build":
        value = build_capsule(
            Path.cwd(),
            host_key=args.host,
            destination=Path(args.destination),
            snapshot=Path(args.snapshot) if args.snapshot else None,
        )
    elif args.command == "worker":
        value = run_worker(Path(args.capsule), host_key=args.host, device=args.device)
    else:
        value = run_wsl_capsule(
            Path.cwd(),
            host_key=args.host,
            destination=Path(args.destination),
            snapshot=Path(args.snapshot) if args.snapshot else None,
            device=args.device,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
