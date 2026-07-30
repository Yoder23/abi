"""Read Windows CPU-set topology without changing process scheduling.

This module is evidence tooling for the ABI native-host campaign.  Windows
exposes an efficiency class for every schedulable logical processor through
GetSystemCpuSetInformation.  Recording that mapping prevents a hybrid-core
affinity experiment from assuming that processor numbers imply P/E identity.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Any


class WindowsCpuTopologyError(RuntimeError):
    """Raised when the Windows CPU-set API cannot be queried safely."""


class _CpuSet(ctypes.Structure):
    _fields_ = [
        ("Id", wintypes.DWORD),
        ("Group", wintypes.WORD),
        ("LogicalProcessorIndex", wintypes.BYTE),
        ("CoreIndex", wintypes.BYTE),
        ("LastLevelCacheIndex", wintypes.BYTE),
        ("NumaNodeIndex", wintypes.BYTE),
        ("EfficiencyClass", wintypes.BYTE),
        ("AllFlags", wintypes.BYTE),
        ("Reserved", wintypes.DWORD),
        ("AllocationTag", ctypes.c_ulonglong),
    ]


class _CpuSetUnion(ctypes.Union):
    _fields_ = [
        ("CpuSet", _CpuSet),
        ("reserved", wintypes.BYTE * 24),
    ]


class _SystemCpuSetInformation(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [
        ("Size", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("data", _CpuSetUnion),
    ]


def _canonical_sha(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("evidence_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def query_windows_cpu_sets() -> dict[str, Any]:
    """Return the kernel-reported logical processor and efficiency mapping."""

    if os.name != "nt":
        raise WindowsCpuTopologyError("Windows CPU sets require Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel32.GetSystemCpuSetInformation
    function.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.HANDLE,
        wintypes.DWORD,
    ]
    function.restype = wintypes.BOOL
    required = wintypes.DWORD()
    function(None, 0, ctypes.byref(required), None, 0)
    error = ctypes.get_last_error()
    if required.value == 0:
        raise WindowsCpuTopologyError(
            f"CPU-set size query failed with Windows error {error}"
        )
    buffer = ctypes.create_string_buffer(required.value)
    if not function(
        buffer, required.value, ctypes.byref(required), None, 0
    ):
        raise WindowsCpuTopologyError(
            "CPU-set query failed with Windows error "
            f"{ctypes.get_last_error()}"
        )
    rows: list[dict[str, Any]] = []
    offset = 0
    header_size = ctypes.sizeof(wintypes.DWORD) * 2
    while offset < required.value:
        if offset + header_size > required.value:
            raise WindowsCpuTopologyError("truncated CPU-set record header")
        record = _SystemCpuSetInformation.from_buffer(buffer, offset)
        if record.Size < header_size or offset + record.Size > required.value:
            raise WindowsCpuTopologyError("invalid CPU-set record size")
        # CpuSetInformation is the zero-valued union member.
        if record.Type == 0:
            flags = int(record.CpuSet.AllFlags)
            rows.append(
                {
                    "id": int(record.CpuSet.Id),
                    "group": int(record.CpuSet.Group),
                    "logical_processor_index": int(
                        record.CpuSet.LogicalProcessorIndex
                    ),
                    "core_index": int(record.CpuSet.CoreIndex),
                    "last_level_cache_index": int(
                        record.CpuSet.LastLevelCacheIndex
                    ),
                    "numa_node_index": int(
                        record.CpuSet.NumaNodeIndex
                    ),
                    "efficiency_class": int(
                        record.CpuSet.EfficiencyClass
                    ),
                    "parked": bool(flags & 0x01),
                    "allocated": bool(flags & 0x02),
                    "allocated_to_target_process": bool(flags & 0x04),
                    "realtime": bool(flags & 0x08),
                }
            )
        offset += int(record.Size)
    if not rows:
        raise WindowsCpuTopologyError("Windows returned no CPU-set records")
    rows.sort(
        key=lambda row: (
            row["group"],
            row["logical_processor_index"],
            row["id"],
        )
    )
    classes: dict[str, list[int]] = {}
    for row in rows:
        classes.setdefault(str(row["efficiency_class"]), []).append(
            row["logical_processor_index"]
        )
    document: dict[str, Any] = {
        "format": "abi-windows-cpu-topology/1",
        "status": "PASS",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_processor_count": os.cpu_count(),
        "cpu_sets": rows,
        "logical_processors_by_efficiency_class": classes,
        "claim_boundary": (
            "Kernel-reported topology only. This evidence does not establish "
            "that any affinity configuration is faster."
        ),
        "final_test_accessed": False,
    }
    document["evidence_sha256"] = _canonical_sha(document)
    return document


def _write_json(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise WindowsCpuTopologyError(
            f"topology evidence is immutable: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = query_windows_cpu_sets()
    _write_json(args.output.resolve(), document)
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
