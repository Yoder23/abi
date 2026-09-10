"""Pure-stdlib feature-factorized R18 compiler for the physical capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

PACKAGE_FORMAT = "abi-r18-factorized-english-realization-package/1"
SLOT_KEYS = ("subject", "verb_base", "verb_3sg", "verb_past", "object")


class IsolatedR18Error(RuntimeError):
    """Raised when isolated R18 extraction fails closed."""


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
        raise IsolatedR18Error(f"required JSON unavailable: {path.name}") from exc
    if not isinstance(value, dict):
        raise IsolatedR18Error(f"required JSON is not an object: {path.name}")
    return value


def _evidence(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _template(output: str, slots: dict[str, str]) -> str:
    template = output.strip()
    matches = []
    for key, value in slots.items():
        occurrences = list(re.finditer(r"(?<!\w)" + re.escape(value) + r"(?!\w)", template))
        if occurrences:
            if len(occurrences) != 1:
                raise IsolatedR18Error("slot value occurs more than once")
            start, end = occurrences[0].span()
            matches.append((start, end, key))
    used = {item[2] for item in matches}
    if "subject" not in used or "object" not in used:
        raise IsolatedR18Error("teacher output omitted a required lexical slot")
    if len({"verb_base", "verb_3sg", "verb_past"} & used) != 1:
        raise IsolatedR18Error("teacher output did not expose one supplied verb form")
    for start, end, key in sorted(matches, reverse=True):
        template = template[:start] + "{" + key + "}" + template[end:]
    return template


def _number_pair(signature: str) -> str:
    mood, tense, polarity, number = signature.split("|")
    other = "plural" if number == "singular" else "singular"
    return "|".join((mood, tense, polarity, other))


def infer_templates(
    records: list[dict[str, Any]], signatures: list[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    parsed: dict[str, Counter[str]] = {signature: Counter() for signature in signatures}
    rejected: list[str] = []
    for record in records:
        sig = record.get("signature")
        slots = record.get("slots")
        if (
            not isinstance(record, dict)
            or set(record) != {"record_id", "signature", "slots", "output"}
            or sig not in parsed
            or not isinstance(record.get("record_id"), str)
            or not isinstance(record.get("output"), str)
            or not isinstance(slots, dict)
            or set(slots) != set(SLOT_KEYS)
            or any(not isinstance(value, str) or not value for value in slots.values())
        ):
            raise IsolatedR18Error("R18 source record schema changed")
        try:
            value = _template(str(record["output"]), slots)
        except IsolatedR18Error:
            rejected.append(str(record["record_id"]))
        else:
            parsed[str(sig)][value] += 1
    if any(sum(counter.values()) < 2 for counter in parsed.values()):
        raise IsolatedR18Error("insufficient parseable teacher support")
    selected = {}
    support = {}
    for sig in signatures:
        pair = _number_pair(sig)
        candidates = set(parsed[sig]) | set(parsed[pair])
        ranked = sorted(
            candidates,
            key=lambda value: (
                -(parsed[sig][value] + parsed[pair][value]),
                -parsed[sig][value],
                value,
            ),
        )
        chosen = ranked[0]
        selected[sig] = chosen
        support[sig] = {
            "local": parsed[sig][chosen],
            "paired_number": parsed[pair][chosen],
            "combined": parsed[sig][chosen] + parsed[pair][chosen],
        }
    return selected, {
        "support": support,
        "rejected_record_ids": sorted(rejected),
        "parseable_records": sum(sum(counter.values()) for counter in parsed.values()),
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
        manifest.get("format") != "abi-r18-isolated-factorized-capsule/1"
        or stored != _evidence(manifest)
        or set(expected) != {"isolated_worker.py", "source_bundle.json", "spec.json"}
        or actual != expected
    ):
        raise IsolatedR18Error("R18 capsule inventory changed")
    return {**manifest, "evidence_sha256": stored}


def _compile(capsule: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = _json(capsule / "spec.json")
    stored_spec = spec.pop("evidence_sha256", None)
    signatures = spec.get("signatures")
    if (
        spec.get("format") != "abi-r18-factorized-template-compiler/1"
        or spec.get("factorized_axis") != "subject_number"
        or not isinstance(signatures, list)
        or len(signatures) != 24
        or len(set(signatures)) != 24
        or stored_spec != _evidence(spec)
    ):
        raise IsolatedR18Error("R18 compiler specification changed")
    bundle = _json(capsule / "source_bundle.json")
    stored_bundle = bundle.pop("evidence_sha256", None)
    records = bundle.get("records")
    if (
        bundle.get("format") != "abi-r18-anonymous-teacher-realizations/1"
        or stored_bundle != _evidence(bundle)
        or not isinstance(records, list)
        or len(records) != 72
    ):
        raise IsolatedR18Error("R18 source bundle changed")
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
    if any(not isinstance(record, dict) or forbidden.intersection(record) for record in records):
        raise IsolatedR18Error("forbidden R18 source field entered capsule")
    templates, diagnostics = infer_templates(records, signatures)
    package = {
        "format": PACKAGE_FORMAT,
        "namespace": "english/core/realization",
        "factorized_axis": "subject_number",
        "templates": templates,
        "support": diagnostics["support"],
    }
    return package, {
        "records_consumed": len(records),
        "templates_learned": len(templates),
        "parseable_records": diagnostics["parseable_records"],
        "rejected_record_ids": diagnostics["rejected_record_ids"],
        "bundle_evidence_sha256": stored_bundle,
        "oracle_fields_consumed": 0,
        "evaluation_rows_consumed": 0,
    }


def run(capsule: Path) -> dict[str, Any]:
    capsule = capsule.resolve()
    manifest = _manifest(capsule)
    mountinfo = Path("/proc/self/mountinfo").read_bytes()
    if (
        os.environ.get("ABI_R18_ISOLATED") != "1"
        or os.name == "nt"
        or Path("/oldroot").exists()
        or Path("/mnt/c").exists()
    ):
        raise IsolatedR18Error("worker is not inside the physical sandbox")
    package, extraction = _compile(capsule)
    output = capsule / "output"
    packages = output / "packages"
    packages.mkdir(parents=True)
    package_bytes = _canonical(package)
    package_sha = hashlib.sha256(package_bytes).hexdigest()
    package_path = packages / f"sha256-{package_sha}.abipkg"
    package_path.write_bytes(package_bytes)
    result = {
        "format": "abi-r18-isolated-factorized-extraction/1",
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
