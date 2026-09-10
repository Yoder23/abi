"""Pure-stdlib R20 instruction/program compiler for the physical capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PACKAGE_FORMAT = "abi-r20-instructional-realization-package/1"


class IsolatedR20Error(RuntimeError):
    """Raised when the isolated R20 compiler fails closed."""


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
        raise IsolatedR20Error(f"required JSON unavailable: {path.name}") from exc
    if not isinstance(value, dict):
        raise IsolatedR20Error(f"required JSON is not an object: {path.name}")
    return value


def _evidence(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _parse_prompt(prompt: str) -> tuple[str, dict[str, str]]:
    lines = prompt.splitlines()
    if len(lines) != 5 or not lines[0].startswith("INSTRUCTION: ") or lines[1] != "DATA:":
        raise IsolatedR20Error("R20 prompt schema changed")
    instruction = lines[0].removeprefix("INSTRUCTION: ").strip()
    slots: dict[str, str] = {}
    for line in lines[2:]:
        key, separator, value = line.partition("=")
        if not separator or not key or not value or key in slots:
            raise IsolatedR20Error("R20 data schema changed")
        slots[key] = value
    if set(slots) != {"field_a", "field_b", "field_c"}:
        raise IsolatedR20Error("R20 slot inventory changed")
    return instruction, slots


def _features(instruction: str) -> dict[str, int]:
    normalized = " ".join(re.findall(r"[a-z0-9]+", instruction.casefold()))
    padded = f"^^{normalized}$$"
    result: dict[str, int] = {}
    for width in (3, 4, 5):
        for index in range(len(padded) - width + 1):
            value = padded[index : index + width]
            result[value] = result.get(value, 0) + 1
    return result


def _parse_program(output: str, slots: dict[str, str]) -> str:
    program = output.strip().replace("\r\n", "\n")
    for key, value in sorted(slots.items(), key=lambda item: -len(item[1])):
        if program.count(value) != 1:
            raise IsolatedR20Error("teacher output omitted or duplicated supplied content")
        program = program.replace(value, f"{{{key}}}")
    return program


def infer_package(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(records) != 72 or len({record.get("record_id") for record in records}) != 72:
        raise IsolatedR20Error("R20 extraction row inventory changed")
    parsed: list[tuple[str, str]] = []
    rejected = 0
    for record in records:
        if set(record) != {"record_id", "prompt", "output"}:
            raise IsolatedR20Error("R20 extraction record schema changed")
        try:
            instruction, slots = _parse_prompt(str(record["prompt"]))
            program = _parse_program(str(record["output"]), slots)
        except (IsolatedR20Error, ValueError):
            rejected += 1
            continue
        parsed.append((instruction, program))
    support = Counter(program for _, program in parsed)
    selected = [program for program, count in support.most_common(6) if count >= 6]
    if len(selected) != 6:
        raise IsolatedR20Error("insufficient teacher support for six output programs")
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for instruction, program in parsed:
        if program in selected:
            grouped[program].update(_features(instruction))
    classes = [
        {
            "program": program,
            "support": support[program],
            "centroid": dict(sorted(grouped[program].items())),
        }
        for program in sorted(selected)
    ]
    package = {
        "format": PACKAGE_FORMAT,
        "namespace": "english/core/instructional-realization",
        "representation": "instruction-ngram-centroid-plus-output-program",
        "classes": classes,
    }
    return package, {
        "records_consumed": len(records),
        "records_parsed": len(parsed),
        "records_rejected": rejected,
        "programs_selected": len(classes),
        "selected_support": {program: support[program] for program in sorted(selected)},
        "task_labels_consumed": 0,
        "expected_outputs_consumed": 0,
        "evaluation_rows_consumed": 0,
        "training_steps": 0,
    }


def _manifest(capsule: Path) -> dict[str, Any]:
    manifest = _json(capsule / "manifest.json")
    stored = manifest.pop("evidence_sha256", None)
    expected = {str(item["path"]): str(item["sha256"]) for item in manifest.get("files", [])}
    actual = {
        path.name: _sha256_file(path)
        for path in capsule.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    if (
        manifest.get("format") != "abi-r20-isolated-instructional-capsule/1"
        or stored != _evidence(manifest)
        or set(expected) != {"isolated_worker.py", "source_bundle.json", "spec.json"}
        or actual != expected
        or manifest.get("prompt_rows_included") != 72
        or any(
            manifest.get(key) != 0
            for key in (
                "task_labels_included",
                "expected_outputs_included",
                "evaluation_rows_included",
                "success_ids_included",
            )
        )
        or manifest.get("network") is not False
    ):
        raise IsolatedR20Error("R20 capsule inventory changed")
    return {**manifest, "evidence_sha256": stored}


def _compile(capsule: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = _json(capsule / "spec.json")
    stored_spec = spec.pop("evidence_sha256", None)
    if spec != {
        "format": "abi-r20-instructional-compiler/1",
        "expected_classes": 6,
        "records": 72,
        "task_labels": 0,
        "expected_outputs": 0,
        "evaluation_rows": 0,
        "success_ids": 0,
    } or stored_spec != _evidence(spec):
        raise IsolatedR20Error("R20 compiler specification changed")
    bundle = _json(capsule / "source_bundle.json")
    stored_bundle = bundle.pop("evidence_sha256", None)
    records = bundle.get("records")
    if (
        bundle.get("format") != "abi-r20-anonymous-instruction-realizations/1"
        or stored_bundle != _evidence(bundle)
        or not isinstance(records, list)
        or len(records) != 72
    ):
        raise IsolatedR20Error("R20 source bundle changed")
    forbidden = {
        "task",
        "expected",
        "oracle",
        "evaluator",
        "evaluation",
        "secret",
        "reveal",
        "success_id",
    }
    if any(not isinstance(record, dict) or forbidden.intersection(record) for record in records):
        raise IsolatedR20Error("forbidden R20 source field entered capsule")
    package, diagnostics = infer_package(records)
    diagnostics["bundle_evidence_sha256"] = stored_bundle
    return package, diagnostics


def run(capsule: Path) -> dict[str, Any]:
    capsule = capsule.resolve()
    manifest = _manifest(capsule)
    mountinfo = Path("/proc/self/mountinfo").read_bytes()
    if (
        os.environ.get("ABI_R20_ISOLATED") != "1"
        or os.name == "nt"
        or Path("/oldroot").exists()
        or Path("/mnt/c").exists()
    ):
        raise IsolatedR20Error("worker is not inside the physical sandbox")
    package, extraction = _compile(capsule)
    output = capsule / "output"
    packages = output / "packages"
    packages.mkdir(parents=True)
    package_bytes = _canonical(package)
    package_sha = hashlib.sha256(package_bytes).hexdigest()
    package_path = packages / f"sha256-{package_sha}.abipkg"
    package_path.write_bytes(package_bytes)
    result = {
        "format": "abi-r20-isolated-instructional-extraction/1",
        "capsule_manifest_evidence_sha256": manifest["evidence_sha256"],
        "mountinfo_sha256": hashlib.sha256(mountinfo).hexdigest(),
        "old_root_present": False,
        "windows_mount_present": False,
        "network_namespace_isolated": True,
        "package": {
            "path": f"packages/{package_path.name}",
            "bytes": len(package_bytes),
            "sha256": package_sha,
        },
        **extraction,
    }
    result["evidence_sha256"] = _evidence(result)
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
