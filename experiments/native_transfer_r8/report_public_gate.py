"""Render the immutable R8 public-falsification certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .capability_generator import canonical_json_bytes


class PublicReportError(RuntimeError):
    """Raised when the public verifier result is absent or stale."""


def _verified(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PublicReportError("verification is not a JSON object")
    payload = dict(value)
    stored = payload.pop("evidence_sha256", None)
    if stored != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
        raise PublicReportError("verification evidence hash is stale")
    if value.get("format") != "abi-native-transfer-r8-public-falsification/1":
        raise PublicReportError("verification format changed")
    return value


def render(value: dict[str, Any]) -> str:
    bootstrap = value["after_minus_base_bootstrap"]
    lines = [
        "# ABI R8 native neural transfer: public falsification certificate",
        "",
        f"Exact answer: **{value['exact_question_answer']}**",
        "",
        f"Verdict: **LEVEL {value['verdict_level']} — FAIL**",
        "",
        str(value["scope"]),
        "",
        "R7 is unchanged. This certificate neither relabels nor broadens R7.",
        "",
        "## Recomputed recipient evidence",
        "",
        "| Public capability | BASE | BEFORE | AFTER | ZERO | RANDOM | WRONG | AFTER−BASE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for capability_id, metrics in value["metrics"].items():
        gain = metrics["AFTER"] - metrics["BASE"]
        lines.append(
            f"| `{capability_id}` | {metrics['BASE']:.4f} | {metrics['BEFORE']:.4f} | "
            f"{metrics['AFTER']:.4f} | {metrics['ZERO']:.4f} | "
            f"{metrics['RANDOM']:.4f} | {metrics['WRONG']:.4f} | {gain:+.4f} |"
        )
    lines.extend(
        [
            "",
            "Paired AFTER−BASE across 1,024 raw rows: "
            f"{bootstrap['point']:+.4f}; 95% bootstrap CI "
            f"[{bootstrap['lower_95']:+.4f}, {bootstrap['upper_95']:+.4f}].",
            "",
            "Canonical atomic extraction accuracy was "
            f"{value['extraction_meta_atomic_accuracy']:.1%} on meta-train and "
            f"{value['extraction_development_atomic_accuracy']:.1%} on development.",
            "",
            "## Gate ledger",
            "",
            "| Gate | Result |",
            "| --- | ---: |",
        ]
    )
    for name, passed in value["gates"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL / NOT RUN'} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            str(value["decision"]),
            "",
            "The held-out commitment was never revealed. Qwen, T5, causal, "
            "baseline, composition, and external gates were not run because the "
            "smallest recipient failed the public prerequisite. The source summary "
            "is hash-bound but lacks raw public source rows, so the verifier "
            "correctly leaves that raw-recomputation gate failed.",
            "",
            "## Exact required answer",
            "",
            "> Did information acquired by training one neural model become an "
            "immutable capability object that caused previously absent, "
            "generalizing behavior to emerge in several independently trained "
            "neural models, without capability-specific training of those recipient "
            "models, and with the recipient neural computation causally necessary "
            "for the behavior?",
            "",
            f"**{value['exact_question_answer']}**",
            "",
            str(value["claim_boundary"]),
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
        print(json.dumps({"status": "FAIL_CLOSED", "error": f"immutable output exists: {output}"}, indent=2))
        return 2
    try:
        value = _verified(Path(args.verification).resolve())
        report = render(value)
    except (OSError, json.JSONDecodeError, PublicReportError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8", newline="\n")
    print(json.dumps({"status": "REPORT_WRITTEN", "path": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
