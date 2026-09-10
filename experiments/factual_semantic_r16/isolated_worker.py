"""Pure-stdlib R16 factual package compiler for the physical capsule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


class IsolatedR16Error(RuntimeError):
    """Raised when isolated R16 extraction fails closed."""


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
        raise IsolatedR16Error(f"required JSON unavailable: {path.name}") from exc
    if not isinstance(value, dict):
        raise IsolatedR16Error(f"required JSON is not an object: {path.name}")
    return value


def _evidence(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _namespace(question: str) -> tuple[str, str]:
    lowered = " ".join(question.casefold().split())
    chemistry = "atomic number" in lowered or "proton-count" in lowered
    geography = "capital" in lowered
    if chemistry == geography:
        raise IsolatedR16Error("semantic namespace is ambiguous")
    if chemistry:
        return "chemistry/periodic-table", "atomic_number"
    return "geography/national-capitals", "national_capital"


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
        manifest.get("format") != "abi-r16-isolated-factual-capsule/1"
        or stored != _evidence(manifest)
        or set(expected) != {"isolated_worker.py", "source_bundle.json", "spec.json"}
        or actual != expected
    ):
        raise IsolatedR16Error("R16 capsule inventory changed")
    return {**manifest, "evidence_sha256": stored}


def _compile(capsule: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = _json(capsule / "spec.json")
    stored_spec = spec.pop("evidence_sha256", None)
    if (
        spec.get("format") != "abi-r16-generic-factual-compiler/1"
        or spec.get("views_per_fact") != 3
        or spec.get("candidate_budget") != 12
        or stored_spec != _evidence(spec)
    ):
        raise IsolatedR16Error("R16 compiler specification changed")
    bundle = _json(capsule / "source_bundle.json")
    stored_bundle = bundle.pop("evidence_sha256", None)
    if (
        bundle.get("format") != "abi-r16-anonymous-source-scores/1"
        or stored_bundle != _evidence(bundle)
        or not isinstance(bundle.get("records"), list)
    ):
        raise IsolatedR16Error("R16 source bundle changed")
    forbidden = {"answer", "oracle", "namespace", "fact_id", "secret", "reveal", "success_id"}
    groups: dict[tuple[str, str, str], list[tuple[int, str]]] = {}
    for record in bundle["records"]:
        if not isinstance(record, dict) or forbidden.intersection(record):
            raise IsolatedR16Error("forbidden R16 source field entered capsule")
        if set(record) != {"subject", "question", "view", "candidates", "scores"}:
            raise IsolatedR16Error("R16 source record schema changed")
        candidates = record["candidates"]
        scores = record["scores"]
        if (
            not isinstance(record["subject"], str)
            or not record["subject"].strip()
            or not isinstance(record["question"], str)
            or not isinstance(record["view"], int)
            or not isinstance(candidates, list)
            or not isinstance(scores, list)
            or len(candidates) != len(scores)
            or len(scores) != spec["candidate_budget"]
            or len(set(candidates)) != len(candidates)
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in scores
            )
            or not all(isinstance(value, str) and value for value in candidates)
        ):
            raise IsolatedR16Error("R16 candidate score contract changed")
        namespace, relation = _namespace(str(record["question"]))
        predicted = str(candidates[max(range(len(scores)), key=scores.__getitem__)])
        groups.setdefault((namespace, relation, str(record["subject"])), []).append(
            (int(record["view"]), predicted)
        )
    packages: dict[str, list[dict[str, str]]] = {}
    for (namespace, relation, subject), observations in groups.items():
        views = {view for view, _value in observations}
        values = {value for _view, value in observations}
        if views != {0, 1, 2} or len(observations) != spec["views_per_fact"] or len(values) != 1:
            raise IsolatedR16Error("R16 source views do not identify one fact")
        packages.setdefault(namespace, []).append(
            {"relation": relation, "entity": subject, "value": next(iter(values))}
        )
    if set(packages) != set(spec["namespaces"]):
        raise IsolatedR16Error("R16 namespace coverage changed")
    payloads = []
    for namespace in sorted(packages):
        payloads.append(
            {
                "format": "abi-r16-canonical-factual-package/1",
                "namespace": namespace,
                "facts": sorted(
                    packages[namespace], key=lambda item: (item["relation"], item["entity"].casefold())
                ),
            }
        )
    return payloads, {"records_consumed": len(bundle["records"]), "bundle_evidence_sha256": stored_bundle}


def run(capsule: Path) -> dict[str, Any]:
    capsule = capsule.resolve()
    manifest = _manifest(capsule)
    mountinfo = Path("/proc/self/mountinfo").read_bytes()
    if (
        os.environ.get("ABI_R16_ISOLATED") != "1"
        or os.name == "nt"
        or Path("/oldroot").exists()
        or Path("/mnt/c").exists()
    ):
        raise IsolatedR16Error("worker is not inside the physical sandbox")
    payloads, extraction = _compile(capsule)
    output = capsule / "output"
    packages_dir = output / "packages"
    packages_dir.mkdir(parents=True)
    packages = []
    for payload in payloads:
        package_bytes = _canonical(payload)
        package_sha = hashlib.sha256(package_bytes).hexdigest()
        path = packages_dir / f"sha256-{package_sha}.abipkg"
        path.write_bytes(package_bytes)
        packages.append(
            {
                "namespace": payload["namespace"],
                "path": f"packages/{path.name}",
                "bytes": len(package_bytes),
                "sha256": package_sha,
                "facts": len(payload["facts"]),
            }
        )
    result = {
        "format": "abi-r16-isolated-factual-extraction/1",
        "capsule_manifest_evidence_sha256": manifest["evidence_sha256"],
        "mountinfo_sha256": hashlib.sha256(mountinfo).hexdigest(),
        "old_root_present": False,
        "windows_mount_present": False,
        "network_namespace_isolated": True,
        "oracle_fields_consumed": 0,
        "fact_ids_consumed": 0,
        "success_ids_consumed": 0,
        "reveal_files_present": 0,
        "packages": packages,
        **extraction,
    }
    result["evidence_sha256"] = _evidence(result)
    (output / "result.json").write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
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
