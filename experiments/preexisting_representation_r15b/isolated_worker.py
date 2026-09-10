"""Pure-stdlib decoder for an isolated anonymous R15B representation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any


class IsolatedR15BError(RuntimeError):
    """Raised when the physical R15B capsule or bundle is invalid."""


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
        raise IsolatedR15BError(f"required JSON unavailable: {path.name}") from exc
    if not isinstance(value, dict):
        raise IsolatedR15BError(f"required JSON is not an object: {path.name}")
    return value


def _f32_safetensors(path: Path) -> tuple[dict[str, str], dict[str, tuple[list[int], list[float]]]]:
    payload = path.read_bytes()
    if len(payload) < 10:
        raise IsolatedR15BError("representation bundle is truncated")
    header_bytes = int.from_bytes(payload[:8], "little")
    data_start = 8 + header_bytes
    try:
        header = json.loads(payload[8:data_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IsolatedR15BError("representation header is invalid") from exc
    metadata = header.pop("__metadata__", {})
    tensors = {}
    intervals = []
    for name, record in header.items():
        shape = record.get("shape")
        offsets = record.get("data_offsets")
        if (
            record.get("dtype") != "F32"
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
        ):
            raise IsolatedR15BError("representation tensor schema changed")
        begin, end = map(int, offsets)
        elements = math.prod(shape)
        if end - begin != 4 * elements or data_start + end > len(payload):
            raise IsolatedR15BError("representation tensor bounds changed")
        intervals.append((begin, end))
        values = [
            item[0]
            for item in struct.iter_unpack("<f", payload[data_start + begin : data_start + end])
        ]
        if not all(math.isfinite(value) for value in values):
            raise IsolatedR15BError("representation tensor is non-finite")
        tensors[name] = (shape, values)
    if sorted(intervals) != intervals or any(
        a[1] != b[0] for a, b in zip(intervals, intervals[1:])
    ):
        raise IsolatedR15BError("representation tensor layout changed")
    return metadata, tensors


def _manifest(capsule: Path) -> dict[str, Any]:
    manifest = _json(capsule / "manifest.json")
    stored = manifest.pop("evidence_sha256", None)
    if (
        manifest.get("format") != "abi-r15b-isolated-representation-capsule/1"
        or stored != hashlib.sha256(_canonical(manifest)).hexdigest()
    ):
        raise IsolatedR15BError("capsule manifest identity changed")
    expected = {str(row["path"]): str(row["sha256"]) for row in manifest.get("files", [])}
    actual = {
        path.name: _sha256_file(path)
        for path in capsule.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    if (
        set(expected) != {"representation.safetensors", "spec.json", "isolated_worker.py"}
        or actual != expected
    ):
        raise IsolatedR15BError("capsule inventory changed")
    forbidden = ("secret", "reveal", "answer", "prompt", "label", "operation")
    if any(term in name.casefold() for name in actual for term in forbidden):
        raise IsolatedR15BError("forbidden semantic material entered capsule")
    return {**manifest, "evidence_sha256": stored}


def _decode(capsule: Path) -> dict[str, Any]:
    spec = _json(capsule / "spec.json")
    stored = spec.pop("evidence_sha256", None)
    if (
        spec.get("format") != "abi-r15b-generic-representation-decoder/1"
        or spec.get("prompts") != 0
        or spec.get("answers") != 0
        or spec.get("semantic_labels") != 0
        or spec.get("candidate_search") is not False
        or stored != hashlib.sha256(_canonical(spec)).hexdigest()
    ):
        raise IsolatedR15BError("generic decoder specification changed")
    metadata, tensors = _f32_safetensors(capsule / "representation.safetensors")
    if (
        metadata.get("format") != "abi-r15b-anonymous-pre-answer-representation/1"
        or not set(metadata).issubset({"format", "source_revision"})
        or set(tensors) != {"residuals", "output_rows"}
    ):
        raise IsolatedR15BError("anonymous representation inventory changed")
    residual_shape, residuals = tensors["residuals"]
    head_shape, head = tensors["output_rows"]
    if residual_shape[:2] != [3, 2] or head_shape[0] != 8 or residual_shape[2] != head_shape[1]:
        raise IsolatedR15BError("anonymous representation shapes changed")
    width = residual_shape[2]
    labels = []
    margins = []
    for slot in range(3):
        for anchor in range(2):
            vector = residuals[(slot * 2 + anchor) * width : (slot * 2 + anchor + 1) * width]
            scores = [
                sum(vector[column] * head[output * width + column] for column in range(width))
                for output in range(8)
            ]
            order = sorted(range(8), key=lambda index: scores[index], reverse=True)
            labels.append(order[0])
            margins.append(scores[order[0]] - scores[order[1]])
    for slot in range(3):
        if (labels[2 * slot + 1] - labels[2 * slot]) % 8 not in {1, 3, 5, 7}:
            raise IsolatedR15BError("decoded labels are not affine permutations")
    return {
        "labels": labels,
        "margins": margins,
        "representation_sha256": _sha256_file(capsule / "representation.safetensors"),
        "prompts_consumed": 0,
        "answers_consumed": 0,
        "semantic_labels_consumed": 0,
        "candidate_search": False,
    }


def run(capsule: Path) -> dict[str, Any]:
    capsule = capsule.resolve()
    manifest = _manifest(capsule)
    mountinfo = Path("/proc/self/mountinfo").read_bytes()
    if (
        os.environ.get("ABI_R15B_ISOLATED") != "1"
        or os.name == "nt"
        or Path("/oldroot").exists()
        or Path("/mnt/c").exists()
    ):
        raise IsolatedR15BError("worker is not inside the physical sandbox")
    result = {
        "format": "abi-r15b-isolated-representation-extraction/1",
        "capsule_manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(),
        "mountinfo_sha256": hashlib.sha256(mountinfo).hexdigest(),
        "old_root_present": False,
        "windows_mount_present": False,
        "network_namespace_isolated": True,
        "reveal_files_present": 0,
        **_decode(capsule),
    }
    result["evidence_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    output = capsule / "output"
    output.mkdir()
    (output / "result.json").write_bytes(
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"
    )
    (output / "mountinfo.txt").write_bytes(mountinfo)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule", required=True)
    args = parser.parse_args()
    run(Path(args.capsule))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
