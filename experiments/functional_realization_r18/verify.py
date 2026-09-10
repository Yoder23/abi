"""Fail-closed, independent verification of an R18 public run."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)
from experiments.linguistic_realization_r17.frames import SIGNATURES, SLOT_KEYS
from experiments.linguistic_realization_r17.verify_source import verify_source

from .package import PACKAGE_FORMAT, R18PackageError, load_package, realize

_SLOT_PATTERN = re.compile(r"\{([a-z0-9_]+)\}")


def _verified_object(path: Path, name: str) -> dict[str, Any]:
    value = json_object(path)
    stored = value.get("evidence_sha256")
    scientific = {key: item for key, item in value.items() if key != "evidence_sha256"}
    if not isinstance(stored, str) or evidence_hash(scientific) != stored:
        raise R14Error(f"R18 {name} evidence hash changed")
    return value


def _jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise R14Error(f"R18 required raw rows changed: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R14Error("R18 required raw rows are unreadable") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise R14Error("R18 required raw rows changed")
    return rows


def _parse_template(output: str, slots: dict[str, str]) -> str | None:
    """Independent implementation of slot-template recovery."""
    result = output.strip()
    matches: list[tuple[int, int, str]] = []
    for key in SLOT_KEYS:
        value = slots[key]
        occurrences = list(re.finditer(r"(?<!\w)" + re.escape(value) + r"(?!\w)", result))
        if occurrences:
            if len(occurrences) != 1:
                return None
            start, end = occurrences[0].span()
            matches.append((start, end, key))
    used = {key for _, _, key in matches}
    if "subject" not in used or "object" not in used:
        return None
    if len({"verb_base", "verb_3sg", "verb_past"} & used) != 1:
        return None
    for start, end, key in sorted(matches, reverse=True):
        result = result[:start] + "{" + key + "}" + result[end:]
    return result


def _paired_number(signature: str) -> str:
    mood, tense, polarity, number = signature.split("|")
    other = "plural" if number == "singular" else "singular"
    return "|".join((mood, tense, polarity, other))


def _derive_package(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    grouped: dict[str, Counter[str]] = {signature: Counter() for signature in SIGNATURES}
    rejected = []
    for record in records:
        template = _parse_template(str(record["output"]), record["slots"])
        if template is None:
            rejected.append(str(record["record_id"]))
        else:
            grouped[str(record["signature"])][template] += 1
    if any(sum(counter.values()) < 2 for counter in grouped.values()):
        raise R14Error("R18 independent compiler lacks source support")
    templates: dict[str, str] = {}
    support: dict[str, Any] = {}
    for signature in SIGNATURES:
        paired = _paired_number(signature)
        candidates = set(grouped[signature]) | set(grouped[paired])
        chosen = sorted(
            candidates,
            key=lambda value: (
                -(grouped[signature][value] + grouped[paired][value]),
                -grouped[signature][value],
                value,
            ),
        )[0]
        templates[signature] = chosen
        support[signature] = {
            "local": grouped[signature][chosen],
            "paired_number": grouped[paired][chosen],
            "combined": grouped[signature][chosen] + grouped[paired][chosen],
        }
    return (
        {
            "format": PACKAGE_FORMAT,
            "namespace": "english/core/realization",
            "factorized_axis": "subject_number",
            "templates": templates,
            "support": support,
        },
        sorted(rejected),
    )


def _canonical_package_bytes(package: dict[str, Any]) -> bytes:
    return json.dumps(package, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _expected_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = [
        {
            "record_id": row["record_id"],
            "signature": row["signature"],
            "slots": row["slots"],
            "output": row["output"],
        }
        for row in rows
        if row["split"] == "extraction"
    ]
    random.Random(18_001).shuffle(records)
    return records


def _control_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        mood, tense, polarity, number = str(record["signature"]).split("|")
        toggled = "question" if mood == "declarative" else "declarative"
        result.append({**record, "signature": "|".join((toggled, tense, polarity, number))})
    return result


def _verify_bundle(path: Path, expected: list[dict[str, Any]], expected_sha: str) -> None:
    if sha256_file(path) != expected_sha:
        raise R14Error("R18 bundle file hash changed")
    bundle = _verified_object(path, "source bundle")
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
    if (
        bundle.get("format") != "abi-r18-anonymous-teacher-realizations/1"
        or bundle.get("records") != expected
        or any(forbidden.intersection(record) for record in bundle.get("records", []))
    ):
        raise R14Error("R18 source bundle changed")


def _expected_spec_sha256() -> str:
    spec = {
        "format": "abi-r18-factorized-template-compiler/1",
        "signatures": list(SIGNATURES),
        "factorized_axis": "subject_number",
        "oracle_fields": 0,
        "evaluation_rows": 0,
        "success_ids": 0,
    }
    spec["evidence_sha256"] = evidence_hash(spec)
    raw = json.dumps(spec, indent=2, sort_keys=True).encode() + b"\n"
    return hashlib.sha256(raw).hexdigest()


def _verify_isolation(
    root: Path,
    run_dir: Path,
    name: str,
    bundle_sha: str,
    bundle_evidence_sha: str,
    expected_package: dict[str, Any],
    rejected: list[str],
) -> tuple[dict[str, Any], Path]:
    directory = run_dir / name
    result_path = directory / "result.json"
    launcher_path = directory / "launcher.json"
    manifest_path = directory / "manifest.json"
    mountinfo_path = directory / "mountinfo.txt"
    result = _verified_object(result_path, f"{name} result")
    launcher = _verified_object(launcher_path, f"{name} launcher")
    manifest = _verified_object(manifest_path, f"{name} manifest")
    if (
        result.get("format") != "abi-r18-isolated-factorized-extraction/1"
        or result.get("records_consumed") != 72
        or result.get("templates_learned") != 24
        or result.get("parseable_records") != 72 - len(rejected)
        or result.get("rejected_record_ids") != rejected
        or result.get("oracle_fields_consumed") != 0
        or result.get("evaluation_rows_consumed") != 0
        or result.get("bundle_evidence_sha256") != bundle_evidence_sha
        or result.get("old_root_present") is not False
        or result.get("windows_mount_present") is not False
        or result.get("network_namespace_isolated") is not True
    ):
        raise R14Error(f"R18 {name} extraction claims changed")
    if (
        launcher.get("format") != "abi-r18-isolated-extraction-launcher/1"
        or launcher.get("worker_exit_code") != 0
        or launcher.get("sandbox_policy") != "linux-pivot-root-no-network/1"
        or launcher.get("result_sha256") != sha256_file(result_path)
        or launcher.get("mountinfo_sha256") != sha256_file(mountinfo_path)
        or result.get("mountinfo_sha256") != sha256_file(mountinfo_path)
    ):
        raise R14Error(f"R18 {name} launcher changed")
    mountinfo = mountinfo_path.read_text(encoding="utf-8")
    if (
        " /capsule " not in mountinfo
        or " /proc " not in mountinfo
        or " /mnt/c " in mountinfo.casefold()
        or " /oldroot " in mountinfo.casefold()
    ):
        raise R14Error(f"R18 {name} physical mount evidence changed")
    expected_files = {
        "isolated_worker.py": sha256_file(
            root / "experiments/functional_realization_r18/isolated_worker.py"
        ),
        "source_bundle.json": bundle_sha,
        "spec.json": _expected_spec_sha256(),
    }
    manifest_files = {
        str(item["path"]): str(item["sha256"])
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }
    if (
        manifest.get("format") != "abi-r18-isolated-factorized-capsule/1"
        or manifest_files != expected_files
        or any(
            manifest.get(key) != 0
            for key in (
                "prompts_included",
                "oracle_fields_included",
                "evaluation_rows_included",
                "success_ids_included",
            )
        )
        or manifest.get("network") is not False
        or result.get("capsule_manifest_evidence_sha256") != manifest["evidence_sha256"]
        or launcher.get("manifest_evidence_sha256") != manifest["evidence_sha256"]
    ):
        raise R14Error(f"R18 {name} capsule manifest changed")
    package_ref = result.get("package")
    if not isinstance(package_ref, dict):
        raise R14Error(f"R18 {name} package reference missing")
    package_path = directory / str(package_ref.get("path"))
    try:
        package = load_package(package_path)
    except R18PackageError as exc:
        raise R14Error(f"R18 {name} package is unavailable") from exc
    package_raw = _canonical_package_bytes(expected_package)
    expected_sha = hashlib.sha256(package_raw).hexdigest()
    if (
        package != expected_package
        or package_path.read_bytes() != package_raw
        or package_ref.get("sha256") != expected_sha
        or package_ref.get("bytes") != len(package_raw)
        or package_path.name != f"sha256-{expected_sha}.abipkg"
        or result.get("bundle_evidence_sha256") is None
    ):
        raise R14Error(f"R18 {name} package changed")
    return result, package_path


def _independent_modal(records: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, Counter[str]] = {signature: Counter() for signature in SIGNATURES}
    for record in records:
        template = _parse_template(str(record["output"]), record["slots"])
        if template is not None:
            grouped[str(record["signature"])][template] += 1
    return {
        signature: sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for signature, counter in grouped.items()
    }


def _fill(template: str, slots: dict[str, str]) -> str:
    return _SLOT_PATTERN.sub(lambda match: slots[match.group(1)], template)


def verify(source_run: Path, run_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    source_run = source_run.resolve()
    run_dir = run_dir.resolve()
    source_strict = verify_source(source_run, "v2")
    if source_strict != json_object(source_run / "strict_source.json"):
        raise R14Error("R18 inherited strict source evidence changed")
    receipt = _verified_object(run_dir / "receipt.json", "receipt")
    if (
        receipt.get("format") != "abi-r18-public-factorized-realization/1"
        or receipt.get("verdict") != "PASS"
        or receipt.get("full_abi_moonshot") != "OPEN"
        or receipt.get("protocol_sha256")
        != sha256_file(root / "experiments/functional_realization_r18/PUBLIC_PROTOCOL.md")
    ):
        raise R14Error("R18 receipt identity changed")
    source_path = source_run / "source_observations.jsonl"
    rows = _jsonl(source_path, receipt["artifacts"]["inherited_source_rows_sha256"])
    if len(rows) != 120:
        raise R14Error("R18 inherited row count changed")
    records = _expected_records(rows)
    if len(records) != 72:
        raise R14Error("R18 extraction row count changed")
    control_records = _control_records(records)
    source_bundle = run_dir / receipt["artifacts"]["source_bundle"]["path"]
    control_bundle = run_dir / receipt["artifacts"]["control_bundle"]["path"]
    _verify_bundle(source_bundle, records, receipt["artifacts"]["source_bundle"]["sha256"])
    _verify_bundle(
        control_bundle,
        control_records,
        receipt["artifacts"]["control_bundle"]["sha256"],
    )
    expected_package, rejected = _derive_package(records)
    expected_control, control_rejected = _derive_package(control_records)
    primary_result, package_path = _verify_isolation(
        root,
        run_dir,
        "extraction",
        sha256_file(source_bundle),
        _verified_object(source_bundle, "source bundle")["evidence_sha256"],
        expected_package,
        rejected,
    )
    control_result, control_path = _verify_isolation(
        root,
        run_dir,
        "control_extraction",
        sha256_file(control_bundle),
        _verified_object(control_bundle, "control bundle")["evidence_sha256"],
        expected_control,
        control_rejected,
    )
    package = load_package(package_path)
    control_package = load_package(control_path)
    modal = _independent_modal(records)
    evaluation = []
    for row in rows:
        if row["split"] != "evaluation":
            continue
        package_output = realize(package, row["signature"], row["slots"])
        control_output = realize(control_package, row["signature"], row["slots"])
        modal_output = _fill(modal[str(row["signature"])], row["slots"])
        evaluation.append(
            {
                "record_id": row["record_id"],
                "signature": row["signature"],
                "slots": row["slots"],
                "expected": row["expected"],
                "source_output": row["output"],
                "source_functional_exact": row["functional_exact"],
                "package_output": package_output,
                "package_functional_exact": package_output == row["expected"],
                "package_source_exact": package_output == row["output"],
                "modal_output": modal_output,
                "modal_functional_exact": modal_output == row["expected"],
                "control_output": control_output,
                "control_functional_exact": control_output == row["expected"],
                "removed_output": None,
            }
        )
    evaluation_ref = receipt["artifacts"]["evaluation"]
    stored_evaluation = _jsonl(run_dir / evaluation_ref["path"], evaluation_ref["sha256"])
    if stored_evaluation != evaluation:
        raise R14Error("R18 evaluation rows changed")
    exact = len(evaluation)
    metrics = {
        "source_extraction_rows": len(records),
        "evaluation_rows": exact,
        "source_functional_exact": sum(row["source_functional_exact"] for row in evaluation),
        "package_functional_exact": sum(row["package_functional_exact"] for row in evaluation),
        "package_source_exact": sum(row["package_source_exact"] for row in evaluation),
        "independent_modal_functional_exact": sum(
            row["modal_functional_exact"] for row in evaluation
        ),
        "source_exact_regressions": sum(
            row["source_functional_exact"] and not row["package_functional_exact"]
            for row in evaluation
        ),
        "control_functional_exact": sum(row["control_functional_exact"] for row in evaluation),
        "removed_abstain": sum(row["removed_output"] is None for row in evaluation),
    }
    if metrics != receipt.get("metrics"):
        raise R14Error("R18 stored metrics changed")
    if not (
        len(records) == 72
        and exact == 48
        and primary_result["templates_learned"] == 24
        and control_result["templates_learned"] == 24
        and metrics["package_functional_exact"] == exact
        and metrics["source_exact_regressions"] == 0
        and metrics["package_functional_exact"] > metrics["independent_modal_functional_exact"]
        and metrics["removed_abstain"] == exact
        and metrics["control_functional_exact"] / exact <= 0.10
    ):
        raise R14Error("R18 strict scientific gate failed")
    package_ref = receipt.get("package", {})
    control_ref = receipt.get("control_package", {})
    if (
        package_ref.get("sha256") != sha256_file(package_path)
        or package_ref.get("bytes") != package_path.stat().st_size
        or package_ref.get("templates") != 24
        or control_ref.get("sha256") != sha256_file(control_path)
        or control_ref.get("bytes") != control_path.stat().st_size
        or receipt["artifacts"]["extraction_result_sha256"]
        != sha256_file(run_dir / "extraction/result.json")
        or receipt["artifacts"]["control_extraction_result_sha256"]
        != sha256_file(run_dir / "control_extraction/result.json")
    ):
        raise R14Error("R18 receipt artifact identity changed")
    accounting = receipt.get("information_accounting", {})
    inherited_accounting = source_strict["information_accounting"]
    if (
        any(accounting.get(key) != value for key, value in inherited_accounting.items())
        or accounting.get("compiler_source_records") != 72
        or accounting.get("compiler_source_bundle_bytes") != source_bundle.stat().st_size
        or accounting.get("final_package_bytes") != package_path.stat().st_size
        or accounting.get("final_templates") != 24
        or any(
            accounting.get(key) != 0
            for key in (
                "frozen_source_parameters_in_package",
                "source_training_steps",
                "host_training_steps",
                "bridge_parameters_trained",
            )
        )
        or not isinstance(accounting.get("compiler_elapsed_seconds"), (int, float))
        or accounting["compiler_elapsed_seconds"] <= 0
    ):
        raise R14Error("R18 information accounting changed")
    result = {
        "format": "abi-r18-strict-verification/1",
        "verdict": "PASS",
        "claim": receipt["claim"],
        "claim_ceiling": receipt["claim_ceiling"],
        "source_rows_verified": len(rows),
        "extraction_rows_verified": len(records),
        "evaluation_rows_verified": exact,
        "package_functional_exact": metrics["package_functional_exact"],
        "modal_functional_exact": metrics["independent_modal_functional_exact"],
        "source_functional_exact": metrics["source_functional_exact"],
        "control_functional_exact": metrics["control_functional_exact"],
        "teacher_present_at_package_execution": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.source_run, args.run_dir)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
