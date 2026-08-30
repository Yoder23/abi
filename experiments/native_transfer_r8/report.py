"""Render a compact R8 report strictly from the fail-closed verifier output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .capability_generator import canonical_json_bytes


class ReportError(RuntimeError):
    """Raised when the verifier output is absent or stale."""


def _verified(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"verifier output unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise ReportError("verifier output is not an object")
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
        raise ReportError("verifier output evidence hash is stale")
    return value


def _percentage(value: Any) -> str:
    return "--" if value is None else f"{100.0 * float(value):.1f}%"


def render(value: Mapping[str, Any]) -> str:
    lines = [
        "# ABI R8 native neural transfer result",
        "",
        f"Exact answer: **{value['exact_question_answer']}**",
        "",
        f"Verdict level: **{value['verdict_level']}**",
        "",
        "This is an additive R8 result. It does not alter or broaden R7.",
        "",
        "## Gate ledger",
        "",
        "| Gate | Recomputed result |",
        "|---|---:|",
    ]
    for name, passed in value.get("gates", {}).items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL / INCOMPLETE'} |")
    lines.extend(
        [
            "",
            "## Decisive pattern",
            "",
            "| Condition | Source teacher | "
            + " | ".join(value.get("recipients", {}).keys())
            + " |",
            "|---|---:|" + "---:|" * len(value.get("recipients", {})),
        ]
    )
    source = value.get("source", {})
    source_before = (
        sum(row.get("T_BEFORE", 0.0) for row in source.values()) / len(source)
        if source
        else None
    )
    source_after = (
        sum(row.get("T_AFTER", 0.0) for row in source.values()) / len(source)
        if source
        else None
    )
    conditions = (
        ("BASE", None),
        ("P_before", "BEFORE"),
        ("WRONG", "WRONG"),
        ("RANDOM", "RANDOM"),
        ("P_after", "AFTER"),
        ("P_after + neural ablation", None),
        ("target-specific LoRA", None),
    )
    for label, condition in conditions:
        if label == "BASE":
            source_value = source_before
            condition = "BASE"
        elif label == "P_after":
            source_value = source_after
        else:
            source_value = None
        cells = []
        for host_name, host in value.get("recipients", {}).items():
            per_capability = host.get("per_capability", {})
            if label == "P_after + neural ablation":
                causal = value.get("causal", {}).get(host_name, {}).get(
                    "per_capability", {}
                )
                scores = [
                    row["after_accuracy"]
                    - row["package_path_causal_fraction"]
                    * (row["after_accuracy"] - row["base_accuracy"])
                    for row in causal.values()
                ]
            elif label == "target-specific LoRA":
                baseline = value.get("baseline", {}).get("per_host", {}).get(
                    host_name, {}
                )
                scores = [
                    row["target_specific_lora"] for row in baseline.values()
                ]
            else:
                scores = (
                    [row.get(condition) for row in per_capability.values()]
                    if condition
                    else []
                )
            cells.append(_percentage(sum(scores) / len(scores)) if scores else "--")
        lines.append(f"| {label} | {_percentage(source_value)} | " + " | ".join(cells) + " |")
    missing = value.get("missing_required_evidence", [])
    lines.extend(
        [
            "",
            "## Missing or failed evidence",
            "",
            *(f"- `{item}`" for item in missing),
            "" if missing else "- None recorded by the verifier.",
            "",
            "## Claim boundary",
            "",
            str(value.get("claim_boundary", "")),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        print(json.dumps({"status": "FAIL_CLOSED", "error": f"immutable report exists: {output}"}, indent=2))
        return 2
    try:
        value = _verified(Path(args.verification).resolve())
        report = render(value)
    except ReportError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "REPORT_WRITTEN", "path": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
