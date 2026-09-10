"""Pure-stdlib structure-contrastive R19 compiler for the physical capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

PACKAGE_FORMAT = "abi-r19-contrastive-english-realization-package/1"
SLOT_KEYS = ("subject", "verb_base", "verb_3sg", "verb_past", "object")


class IsolatedR19Error(RuntimeError):
    """Raised when isolated R19 extraction fails closed."""


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
        raise IsolatedR19Error(f"required JSON unavailable: {path.name}") from exc
    if not isinstance(value, dict):
        raise IsolatedR19Error(f"required JSON is not an object: {path.name}")
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
                raise IsolatedR19Error("slot value occurs more than once")
            start, end = occurrences[0].span()
            matches.append((start, end, key))
    used = {item[2] for item in matches}
    if "subject" not in used or "object" not in used:
        raise IsolatedR19Error("teacher output omitted a required lexical slot")
    if len({"verb_base", "verb_3sg", "verb_past"} & used) != 1:
        raise IsolatedR19Error("teacher output did not expose one supplied verb form")
    for start, end, key in sorted(matches, reverse=True):
        template = template[:start] + "{" + key + "}" + template[end:]
    return template


def _partner(signature: str) -> str | None:
    mood, tense, polarity, number = signature.split("|")
    if mood != "question" and tense != "future":
        return None
    other = "negative" if polarity == "positive" else "positive"
    return "|".join((mood, tense, other, number))


def _positive_negative(left: str, right: str) -> tuple[str, str]:
    return (left, right) if "|positive|" in f"|{left}|" else (right, left)


def _compatible(positive: str, negative: str) -> bool:
    positive_tokens = positive.split()
    negative_tokens = negative.split()
    if len(negative_tokens) != len(positive_tokens) + 1:
        return False
    return any(
        negative_tokens[:index] + negative_tokens[index + 1 :] == positive_tokens
        and not negative_tokens[index].startswith("{")
        for index in range(len(negative_tokens))
    )


def infer_templates(
    records: list[dict[str, Any]], signatures: list[str]
) -> tuple[dict[str, str], dict[str, Any]]:
    parsed: dict[str, Counter[str]] = {signature: Counter() for signature in signatures}
    rejected = []
    for record in records:
        signature = record.get("signature")
        slots = record.get("slots")
        if (
            not isinstance(record, dict)
            or set(record) != {"record_id", "signature", "slots", "output"}
            or signature not in parsed
            or not isinstance(record.get("record_id"), str)
            or not isinstance(record.get("output"), str)
            or not isinstance(slots, dict)
            or set(slots) != set(SLOT_KEYS)
            or any(not isinstance(value, str) or not value for value in slots.values())
        ):
            raise IsolatedR19Error("R19 source record schema changed")
        try:
            value = _template(str(record["output"]), slots)
        except IsolatedR19Error:
            rejected.append(str(record["record_id"]))
        else:
            parsed[str(signature)][value] += 1
    if any(sum(counter.values()) < 2 for counter in parsed.values()):
        raise IsolatedR19Error("insufficient parseable teacher support")
    templates: dict[str, str] = {}
    support: dict[str, Any] = {}
    completed: set[tuple[str, str]] = set()
    for signature in signatures:
        partner = _partner(signature)
        if partner is None:
            chosen, count = sorted(parsed[signature].items(), key=lambda item: (-item[1], item[0]))[
                0
            ]
            templates[signature] = chosen
            support[signature] = {"mode": "local", "local": count, "joint": count}
            continue
        positive_signature, negative_signature = _positive_negative(signature, partner)
        pair = (positive_signature, negative_signature)
        if pair in completed:
            continue
        candidates = [
            (positive_template, negative_template, positive_count, negative_count)
            for positive_template, positive_count in parsed[positive_signature].items()
            for negative_template, negative_count in parsed[negative_signature].items()
            if _compatible(positive_template, negative_template)
        ]
        if not candidates:
            raise IsolatedR19Error("no structure-compatible polarity contrast")
        positive_template, negative_template, positive_count, negative_count = sorted(
            candidates,
            key=lambda item: (
                -(item[2] + item[3]),
                -min(item[2], item[3]),
                -item[2],
                item[0],
                item[1],
            ),
        )[0]
        joint = positive_count + negative_count
        templates[positive_signature] = positive_template
        templates[negative_signature] = negative_template
        support[positive_signature] = {
            "mode": "polarity_contrast",
            "local": positive_count,
            "paired_polarity": negative_count,
            "joint": joint,
        }
        support[negative_signature] = {
            "mode": "polarity_contrast",
            "local": negative_count,
            "paired_polarity": positive_count,
            "joint": joint,
        }
        completed.add(pair)
    if set(templates) != set(signatures):
        raise IsolatedR19Error("R19 template coverage changed")
    return templates, {
        "support": support,
        "parseable_records": sum(sum(counter.values()) for counter in parsed.values()),
        "rejected_record_ids": sorted(rejected),
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
        manifest.get("format") != "abi-r19-isolated-contrastive-capsule/1"
        or stored != _evidence(manifest)
        or set(expected) != {"isolated_worker.py", "source_bundle.json", "spec.json"}
        or actual != expected
    ):
        raise IsolatedR19Error("R19 capsule inventory changed")
    return {**manifest, "evidence_sha256": stored}


def _compile(capsule: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = _json(capsule / "spec.json")
    stored_spec = spec.pop("evidence_sha256", None)
    signatures = spec.get("signatures")
    if (
        spec.get("format") != "abi-r19-contrastive-template-compiler/1"
        or spec.get("factorization") != "same-number-polarity-contrast"
        or not isinstance(signatures, list)
        or len(signatures) != 24
        or len(set(signatures)) != 24
        or stored_spec != _evidence(spec)
    ):
        raise IsolatedR19Error("R19 compiler specification changed")
    bundle = _json(capsule / "source_bundle.json")
    stored_bundle = bundle.pop("evidence_sha256", None)
    records = bundle.get("records")
    if (
        bundle.get("format") != "abi-r19-anonymous-teacher-realizations/1"
        or stored_bundle != _evidence(bundle)
        or not isinstance(records, list)
        or len(records) != 72
    ):
        raise IsolatedR19Error("R19 source bundle changed")
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
        raise IsolatedR19Error("forbidden R19 source field entered capsule")
    templates, diagnostics = infer_templates(records, signatures)
    package = {
        "format": PACKAGE_FORMAT,
        "namespace": "english/core/realization",
        "factorization": "same-number-polarity-contrast",
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
        os.environ.get("ABI_R19_ISOLATED") != "1"
        or os.name == "nt"
        or Path("/oldroot").exists()
        or Path("/mnt/c").exists()
    ):
        raise IsolatedR19Error("worker is not inside the physical sandbox")
    package, extraction = _compile(capsule)
    output = capsule / "output"
    packages = output / "packages"
    packages.mkdir(parents=True)
    package_bytes = _canonical(package)
    package_sha = hashlib.sha256(package_bytes).hexdigest()
    package_path = packages / f"sha256-{package_sha}.abipkg"
    package_path.write_bytes(package_bytes)
    result = {
        "format": "abi-r19-isolated-contrastive-extraction/1",
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
