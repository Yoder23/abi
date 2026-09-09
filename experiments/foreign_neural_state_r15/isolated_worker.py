"""Pure-stdlib worker for physically isolated R15A delta extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

MODULUS = 8
OPERATORS = 3
FORMAT = "abi-r15a-isolated-weight-delta-extraction/1"
FRONTEND_FORMAT = "abi-r15a-frozen-weight-delta-affine-table-frontend/4"


class IsolatedR15Error(RuntimeError):
    """Raised when the isolated extractor input or mount contract changes."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IsolatedR15Error(f"required JSON unavailable: {path.name}") from exc
    if not isinstance(value, dict):
        raise IsolatedR15Error(f"required JSON is not an object: {path.name}")
    return value


def _f32_safetensors(path: Path) -> tuple[dict[str, str], dict[str, tuple[list[int], list[float]]]]:
    payload = path.read_bytes()
    if len(payload) < 10:
        raise IsolatedR15Error(f"safetensors artifact truncated: {path.name}")
    header_bytes = int.from_bytes(payload[:8], "little")
    data_start = 8 + header_bytes
    try:
        header = json.loads(payload[8:data_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IsolatedR15Error(f"safetensors header invalid: {path.name}") from exc
    if not isinstance(header, dict) or data_start > len(payload):
        raise IsolatedR15Error(f"safetensors bounds invalid: {path.name}")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()
    ):
        raise IsolatedR15Error(f"safetensors metadata invalid: {path.name}")
    tensors: dict[str, tuple[list[int], list[float]]] = {}
    intervals = []
    for name, record in header.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise IsolatedR15Error(f"safetensors tensor schema invalid: {path.name}")
        shape = record.get("shape")
        offsets = record.get("data_offsets")
        if (
            record.get("dtype") != "F32"
            or not isinstance(shape, list)
            or not shape
            or any(not isinstance(item, int) or item <= 0 for item in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise IsolatedR15Error(f"safetensors tensor contract invalid: {path.name}")
        begin, end = (int(offsets[0]), int(offsets[1]))
        elements = math.prod(shape)
        if begin < 0 or end - begin != 4 * elements or data_start + end > len(payload):
            raise IsolatedR15Error(f"safetensors tensor bounds invalid: {path.name}")
        intervals.append((begin, end))
        values = [
            item[0]
            for item in struct.iter_unpack("<f", payload[data_start + begin : data_start + end])
        ]
        if not all(math.isfinite(value) for value in values):
            raise IsolatedR15Error(f"safetensors tensor numerics invalid: {path.name}")
        tensors[name] = (shape, values)
    if (
        sorted(intervals) != intervals
        or any(left[1] != right[0] for left, right in zip(intervals, intervals[1:]))
        or (intervals and (intervals[0][0] != 0 or data_start + intervals[-1][1] != len(payload)))
    ):
        raise IsolatedR15Error(f"safetensors layout invalid: {path.name}")
    return metadata, tensors


def _verify_manifest(capsule: Path) -> dict[str, Any]:
    manifest = _json(capsule / "manifest.json")
    stored = manifest.pop("evidence_sha256", None)
    if manifest.get("format") != "abi-r15a-isolated-extraction-capsule/1" or stored != hashlib.sha256(
        _canonical(manifest)
    ).hexdigest():
        raise IsolatedR15Error("capsule manifest identity changed")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise IsolatedR15Error("capsule inventory missing")
    declared = {str(row["path"]): str(row["sha256"]) for row in files}
    allowed = {
        "delta.safetensors",
        "frontend.safetensors",
        "frontend_spec.json",
        "isolated_worker.py",
    }
    actual = {
        path.name: _sha256_file(path)
        for path in capsule.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    if set(declared) != allowed or actual != declared:
        raise IsolatedR15Error("capsule file inventory changed")
    if any(term in name.casefold() for name in actual for term in ("secret", "reveal", "answer")):
        raise IsolatedR15Error("forbidden held-out material entered extraction capsule")
    return {**manifest, "evidence_sha256": stored}


def _decode(capsule: Path) -> dict[str, Any]:
    spec = _json(capsule / "frontend_spec.json")
    spec_stored = spec.pop("evidence_sha256", None)
    if (
        spec.get("format") != FRONTEND_FORMAT
        or spec.get("candidate_program_search") is not False
        or spec.get("behavioral_queries") != 0
        or spec.get("oracle_calls_at_extraction") != 0
        or spec_stored != hashlib.sha256(_canonical(spec)).hexdigest()
    ):
        raise IsolatedR15Error("frozen frontend specification changed")
    spec["evidence_sha256"] = spec_stored
    frontend_metadata, frontend = _f32_safetensors(capsule / "frontend.safetensors")
    if set(frontend) != {"readout", "bias"} or frontend_metadata != {
        "format": FRONTEND_FORMAT,
        "frontend_spec_sha256": spec_stored,
    }:
        raise IsolatedR15Error("frozen frontend tensor inventory changed")
    readout_shape, readout = frontend["readout"]
    bias_shape, bias = frontend["bias"]
    width = int(spec.get("feature_elements", 0)) // MODULUS
    if readout_shape != [OPERATORS, MODULUS, width] or bias_shape != [
        OPERATORS,
        MODULUS,
        MODULUS,
    ]:
        raise IsolatedR15Error("frozen frontend tensor shape changed")
    tensor_hash = hashlib.sha256()
    for name in ("readout", "bias"):
        tensor_hash.update(name.encode() + b"\0")
        tensor_hash.update(
            struct.pack(f"<{len(frontend[name][1])}f", *frontend[name][1])
        )
    if tensor_hash.hexdigest() != spec.get("tensor_sha256"):
        raise IsolatedR15Error("frozen frontend tensor hash changed")
    delta_metadata, delta = _f32_safetensors(capsule / "delta.safetensors")
    if delta_metadata or set(delta) != {"delta"} or delta["delta"][0] != [MODULUS * width]:
        raise IsolatedR15Error("anonymous weight-delta contract changed")
    flat = delta["delta"][1]
    rows = [flat[index * width : (index + 1) * width] for index in range(MODULUS)]
    means = [sum(rows[index][column] for index in range(MODULUS)) / MODULUS for column in range(width)]
    rows = [[value - means[column] for column, value in enumerate(row)] for row in rows]
    norm = math.sqrt(sum(value * value for row in rows for value in row))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise IsolatedR15Error("anonymous weight delta is degenerate")
    rows = [[value / norm for value in row] for row in rows]

    def readout_value(operator: int, state: int, column: int) -> float:
        return readout[(operator * MODULUS + state) * width + column]

    def bias_value(operator: int, state: int, output: int) -> float:
        return bias[(operator * MODULUS + state) * MODULUS + output]

    scores = []
    for operator in range(OPERATORS):
        operator_scores = []
        for state in range(MODULUS):
            operator_scores.append(
                [
                    sum(
                        rows[output][column] * readout_value(operator, state, column)
                        for column in range(width)
                    )
                    + bias_value(operator, state, output)
                    for output in range(MODULUS)
                ]
            )
        scores.append(operator_scores)
    labels = []
    margins = []
    for operator in range(OPERATORS):
        candidates = []
        for at_zero in range(MODULUS):
            for multiplier in (1, 3, 5, 7):
                score = sum(
                    scores[operator][state][(multiplier * state + at_zero) % MODULUS]
                    for state in range(MODULUS)
                )
                candidates.append((score, at_zero, (at_zero + multiplier) % MODULUS))
        candidates.sort(reverse=True)
        labels.extend(candidates[0][1:])
        margins.append(candidates[0][0] - candidates[1][0])
    return {
        "labels": labels,
        "operator_margins": margins,
        "frontend_spec_sha256": spec_stored,
        "weight_delta_sha256": _sha256_file(capsule / "delta.safetensors"),
        "behavioral_queries": 0,
        "answers_consumed": 0,
        "oracle_calls": 0,
        "candidate_program_search": False,
    }


def run(capsule: Path) -> dict[str, Any]:
    capsule = capsule.resolve()
    manifest = _verify_manifest(capsule)
    mountinfo = Path("/proc/self/mountinfo")
    mount_bytes = mountinfo.read_bytes() if mountinfo.is_file() else b""
    isolated = (
        os.environ.get("ABI_R15_ISOLATED") == "1"
        and os.name != "nt"
        and not Path("/oldroot").exists()
        and not Path("/mnt/c").exists()
    )
    if not isolated:
        raise IsolatedR15Error("worker is not inside the physical extraction sandbox")
    result = {
        "format": FORMAT,
        "capsule_manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(),
        "mountinfo_sha256": hashlib.sha256(mount_bytes).hexdigest(),
        "old_root_present": Path("/oldroot").exists(),
        "windows_mount_present": Path("/mnt/c").exists(),
        "network_namespace_isolated": True,
        "capability_reveal_files_present": 0,
        "operation_tables_present": 0,
        **_decode(capsule),
    }
    result["evidence_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    output = capsule / "output"
    output.mkdir()
    (output / "result.json").write_bytes(
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"
    )
    (output / "mountinfo.txt").write_bytes(mount_bytes)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", required=True)
    args = parser.parse_args()
    try:
        result = run(Path(args.capsule))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
