"""Pure-stdlib R17 template learner for the physical extraction capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

PACKAGE_FORMAT = "abi-r17-canonical-english-realization-package/1"
SLOT_KEYS = ("subject", "verb_base", "verb_3sg", "verb_past", "object")


class IsolatedR17Error(RuntimeError):
    """Raised when isolated R17 extraction fails closed."""


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
        raise IsolatedR17Error(f"required JSON unavailable: {path.name}") from exc
    if not isinstance(value, dict):
        raise IsolatedR17Error(f"required JSON is not an object: {path.name}")
    return value


def _evidence(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


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
        manifest.get("format") != "abi-r17-isolated-realization-capsule/1"
        or stored != _evidence(manifest)
        or set(expected) != {"isolated_worker.py", "source_bundle.json", "spec.json"}
        or actual != expected
    ):
        raise IsolatedR17Error("R17 capsule inventory changed")
    return {**manifest, "evidence_sha256": stored}


def _template(output: str, slots: dict[str, str]) -> str:
    template = output.strip()
    matches = []
    for key, value in slots.items():
        occurrences = list(re.finditer(r"(?<!\w)" + re.escape(value) + r"(?!\w)", template))
        if occurrences:
            if len(occurrences) != 1:
                raise IsolatedR17Error("slot value occurs more than once")
            start, end = occurrences[0].span()
            matches.append((start, end, key))
    if "subject" not in {item[2] for item in matches} or "object" not in {
        item[2] for item in matches
    }:
        raise IsolatedR17Error("teacher output omitted a required lexical slot")
    if len({"verb_base", "verb_3sg", "verb_past"} & {item[2] for item in matches}) != 1:
        raise IsolatedR17Error("teacher output must use exactly one supplied verb form")
    for start, end, key in sorted(matches, reverse=True):
        template = template[:start] + "{" + key + "}" + template[end:]
    return template


def _compile(capsule: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = _json(capsule / "spec.json")
    stored_spec = spec.pop("evidence_sha256", None)
    signatures = spec.get("signatures")
    if (
        spec.get("format") != "abi-r17-generic-template-compiler/1"
        or not isinstance(signatures, list)
        or len(signatures) != 24
        or len(set(signatures)) != 24
        or spec.get("observations_per_signature") != 3
        or stored_spec != _evidence(spec)
    ):
        raise IsolatedR17Error("R17 compiler specification changed")
    bundle = _json(capsule / "source_bundle.json")
    stored_bundle = bundle.pop("evidence_sha256", None)
    records = bundle.get("records")
    if (
        bundle.get("format") != "abi-r17-anonymous-teacher-realizations/1"
        or stored_bundle != _evidence(bundle)
        or not isinstance(records, list)
        or len(records) != 72
    ):
        raise IsolatedR17Error("R17 source bundle changed")
    forbidden = {
        "prompt",
        "expected",
        "oracle",
        "evaluator",
        "evaluation",
        "secret",
        "reveal",
        "success_id",
    }
    learned: dict[str, list[str]] = {}
    for record in records:
        if not isinstance(record, dict) or forbidden.intersection(record):
            raise IsolatedR17Error("forbidden R17 source field entered capsule")
        if set(record) != {"record_id", "signature", "slots", "output"}:
            raise IsolatedR17Error("R17 source record schema changed")
        sig = record["signature"]
        slots = record["slots"]
        if (
            sig not in signatures
            or not isinstance(record["record_id"], str)
            or not isinstance(record["output"], str)
            or not isinstance(slots, dict)
            or set(slots) != set(SLOT_KEYS)
            or any(not isinstance(value, str) or not value for value in slots.values())
        ):
            raise IsolatedR17Error("R17 source record values changed")
        learned.setdefault(sig, []).append(_template(record["output"], slots))
    templates = {}
    for sig in signatures:
        observed = learned.get(sig, [])
        if len(observed) != spec["observations_per_signature"] or len(set(observed)) != 1:
            raise IsolatedR17Error("teacher outputs do not identify one stable template")
        templates[sig] = observed[0]
    package = {
        "format": PACKAGE_FORMAT,
        "namespace": "english/core/realization",
        "templates": templates,
    }
    return package, {
        "records_consumed": len(records),
        "templates_learned": len(templates),
        "bundle_evidence_sha256": stored_bundle,
        "oracle_fields_consumed": 0,
        "evaluation_rows_consumed": 0,
    }


def run(capsule: Path) -> dict[str, Any]:
    capsule = capsule.resolve()
    manifest = _manifest(capsule)
    mountinfo = Path("/proc/self/mountinfo").read_bytes()
    if (
        os.environ.get("ABI_R17_ISOLATED") != "1"
        or os.name == "nt"
        or Path("/oldroot").exists()
        or Path("/mnt/c").exists()
    ):
        raise IsolatedR17Error("worker is not inside the physical sandbox")
    package, extraction = _compile(capsule)
    output = capsule / "output"
    packages = output / "packages"
    packages.mkdir(parents=True)
    package_bytes = _canonical(package)
    package_sha = hashlib.sha256(package_bytes).hexdigest()
    package_path = packages / f"sha256-{package_sha}.abipkg"
    package_path.write_bytes(package_bytes)
    result = {
        "format": "abi-r17-isolated-realization-extraction/1",
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
