"""Pure-stdlib free-label compiler executed inside the R27 pivot-root capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any


class IsolationError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def evidence(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IsolationError(f"unreadable JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise IsolationError(f"non-object JSON: {path.name}")
    return value


def answer_key(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = " ".join(value.strip().rstrip(".!?").split()).casefold()
    if value.endswith("()"):
        value = value[:-2].rstrip()
    if value.endswith(" degrees") and value[:-8].strip().replace(".", "", 1).isdigit():
        value = value[:-8].strip()
    return value


def verify_capsule(capsule: Path) -> dict[str, Any]:
    manifest = load_object(capsule / "manifest.json")
    stored = manifest.pop("evidence_sha256", None)
    expected = {item["path"]: item["sha256"] for item in manifest.get("files", [])}
    actual = {path.name: file_hash(path) for path in capsule.iterdir() if path.is_file() and path.name != "manifest.json"}
    if (
        manifest.get("format") != "abi-r27-isolated-open-label-capsule/1"
        or stored != evidence(manifest)
        or set(expected) != {"isolated_worker.py", "source_bundle.json", "spec.json"}
        or actual != expected
    ):
        raise IsolationError("R27 capsule inventory changed")
    return {**manifest, "evidence_sha256": stored}


def compile_packages(capsule: Path) -> tuple[list[dict[str, Any]], int]:
    spec = load_object(capsule / "spec.json")
    spec_hash = spec.pop("evidence_sha256", None)
    if spec.get("format") != "abi-r27-generic-open-label-compiler/1" or spec.get("views_per_fact") != 3 or spec_hash != evidence(spec):
        raise IsolationError("R27 compiler spec changed")
    bundle = load_object(capsule / "source_bundle.json")
    bundle_hash = bundle.pop("evidence_sha256", None)
    if bundle.get("format") != "abi-r27-anonymous-free-label-observations/1" or bundle_hash != evidence(bundle) or not isinstance(bundle.get("records"), list):
        raise IsolationError("R27 source bundle changed")
    forbidden = {"oracle", "oracle_domain", "oracle_answer", "fact_id", "secret", "reveal", "success_id", "accepted_labels"}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in bundle["records"]:
        if not isinstance(row, dict) or forbidden.intersection(row) or set(row) != {"subject", "question", "view", "answer", "label"}:
            raise IsolationError("forbidden or changed R27 record schema")
        if (
            not isinstance(row["subject"], str) or not row["subject"].strip()
            or not isinstance(row["question"], str) or not row["question"].strip()
            or not isinstance(row["view"], int)
            or not isinstance(row["answer"], str) or not row["answer"].strip()
            or not isinstance(row["label"], str) or re.fullmatch(r"[a-z]+", row["label"]) is None
        ):
            raise IsolationError("invalid R27 source observation")
        groups.setdefault(row["subject"], []).append(row)
    by_label: dict[str, list[dict[str, str]]] = {}
    for subject, rows in groups.items():
        if len(rows) != spec["views_per_fact"] or {row["view"] for row in rows} != {0, 1, 2}:
            raise IsolationError("R27 fact lacks the registered independent views")
        labels = {row["label"] for row in rows}
        answers = {answer_key(row["answer"]) for row in rows}
        if len(labels) != 1 or len(answers) != 1 or "" in answers:
            raise IsolationError("R27 teacher views do not reach exact semantic consensus")
        label = next(iter(labels))
        answer = min((row["answer"].strip() for row in rows), key=lambda value: (len(value), value))
        if answer_key(answer) != next(iter(answers)):
            raise IsolationError("R27 answer canonicalization changed")
        by_label.setdefault(label, []).append({"relation": "teacher_fact", "entity": subject, "value": answer})
    packages = [
        {
            "format": "abi-r16-canonical-factual-package/1",
            "namespace": f"teacher/{label}",
            "facts": sorted(facts, key=lambda item: item["entity"].casefold()),
        }
        for label, facts in sorted(by_label.items())
    ]
    return packages, len(bundle["records"])


def run(capsule: Path) -> dict[str, Any]:
    capsule = capsule.resolve()
    manifest = verify_capsule(capsule)
    mountinfo = Path("/proc/self/mountinfo").read_bytes()
    if os.environ.get("ABI_R27_ISOLATED") != "1" or os.name == "nt" or Path("/oldroot").exists() or Path("/mnt/c").exists():
        raise IsolationError("R27 worker is not physically isolated")
    payloads, records = compile_packages(capsule)
    package_dir = capsule / "output/packages"
    package_dir.mkdir(parents=True)
    inventory = []
    for payload in payloads:
        raw = canonical(payload)
        digest = hashlib.sha256(raw).hexdigest()
        path = package_dir / f"sha256-{digest}.abipkg"
        path.write_bytes(raw)
        inventory.append({"namespace": payload["namespace"], "path": f"packages/{path.name}", "bytes": len(raw), "sha256": digest, "facts": len(payload["facts"])})
    result = {
        "format": "abi-r27-isolated-open-label-extraction/1",
        "capsule_manifest_evidence_sha256": manifest["evidence_sha256"],
        "mountinfo_sha256": hashlib.sha256(mountinfo).hexdigest(),
        "old_root_present": False, "windows_mount_present": False,
        "network_namespace_isolated": True, "oracle_fields_consumed": 0,
        "fact_ids_consumed": 0, "success_ids_consumed": 0,
        "accepted_labels_consumed": 0, "records_consumed": records,
        "packages": inventory,
    }
    result["evidence_sha256"] = evidence(result)
    (capsule / "output/result.json").write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    (capsule / "output/mountinfo.txt").write_bytes(mountinfo)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule", type=Path, required=True)
    args = parser.parse_args()
    run(args.capsule)


if __name__ == "__main__":
    main()

