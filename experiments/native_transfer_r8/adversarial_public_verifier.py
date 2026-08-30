"""Exercise hostile mutations against the R8 public fail-closed verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .capability_generator import canonical_json_bytes
from .report_public_gate import PublicReportError, _verified
from .verify_public_gate import PublicVerificationError, verify


class AdversarialAuditError(RuntimeError):
    """Raised when a hostile mutation is accepted unexpectedly."""


def _expect_failure(name: str, action: Callable[[], Any]) -> dict[str, Any]:
    try:
        action()
    except (OSError, ValueError, PublicVerificationError, PublicReportError) as exc:
        return {"case": name, "outcome": "REJECTED", "exception": type(exc).__name__}
    raise AdversarialAuditError(f"hostile case was accepted: {name}")


def audit(
    config_path: Path,
    campaign_root: Path,
    gate_dir: Path,
    verification_path: Path,
) -> dict[str, Any]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="abi-r8-hostile-") as temporary:
        root = Path(temporary)

        missing_raw = root / "missing_raw"
        shutil.copytree(gate_dir, missing_raw)
        (missing_raw / "observations.jsonl").unlink()
        rows.append(
            _expect_failure(
                "missing_raw_rows",
                lambda: verify(config_path, campaign_root, missing_raw),
            )
        )

        mutated_raw = root / "mutated_raw"
        shutil.copytree(gate_dir, mutated_raw)
        raw_path = mutated_raw / "observations.jsonl"
        raw_path.write_bytes(raw_path.read_bytes() + b"{}\n")
        rows.append(
            _expect_failure(
                "raw_row_hash_mismatch",
                lambda: verify(config_path, campaign_root, mutated_raw),
            )
        )

        stale_receipt = root / "stale_receipt"
        shutil.copytree(gate_dir, stale_receipt)
        receipt_path = stale_receipt / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["rows"] = int(receipt["rows"]) + 1
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        rows.append(
            _expect_failure(
                "stale_receipt_hash",
                lambda: verify(config_path, campaign_root, stale_receipt),
            )
        )

        missing_source = root / "missing_source"
        shutil.copytree(campaign_root, missing_source)
        source_receipt = json.loads(
            (missing_source / "pre_reveal/source_public/receipt.json").read_text(
                encoding="utf-8"
            )
        )
        (missing_source / "pre_reveal/source_public" / source_receipt["states"]["path"]).unlink()
        rows.append(
            _expect_failure(
                "missing_source_state",
                lambda: verify(config_path, missing_source, gate_dir),
            )
        )

        contaminated = root / "heldout_contamination"
        shutil.copytree(campaign_root, contaminated)
        (contaminated / "forbidden.abipkg").write_bytes(b"not-a-package")
        rows.append(
            _expect_failure(
                "heldout_package_present",
                lambda: verify(config_path, contaminated, gate_dir),
            )
        )

        changed_config = root / "changed_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["gates"]["recipient_gain_minimum"] = 0.0
        changed_config.write_text(json.dumps(config), encoding="utf-8")
        rows.append(
            _expect_failure(
                "changed_threshold_config",
                lambda: verify(changed_config, campaign_root, gate_dir),
            )
        )

        forged = root / "forged_boolean"
        shutil.copytree(gate_dir, forged)
        forged_receipt_path = forged / "receipt.json"
        forged_receipt = json.loads(forged_receipt_path.read_text(encoding="utf-8"))
        forged_receipt["scientific_pass"] = True
        forged_receipt.pop("evidence_sha256")
        forged_receipt["evidence_sha256"] = hashlib.sha256(
            canonical_json_bytes(forged_receipt)
        ).hexdigest()
        forged_receipt_path.write_bytes(
            json.dumps(forged_receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        forged_result = verify(config_path, campaign_root, forged)
        if (
            forged_result.get("exact_question_answer") != "NO"
            or forged_result.get("trusted_scientific_booleans_consumed") != 0
        ):
            raise AdversarialAuditError("forged scientific boolean changed verdict")
        rows.append(
            {
                "case": "forged_scientific_boolean_ignored",
                "outcome": "NO_VERDICT_UNCHANGED",
                "trusted_booleans_consumed": 0,
            }
        )

        stale_verification = root / "stale_verification.json"
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        verification["exact_question_answer"] = "YES"
        stale_verification.write_text(json.dumps(verification), encoding="utf-8")
        rows.append(
            _expect_failure(
                "stale_verification_report",
                lambda: _verified(stale_verification),
            )
        )

    value = {
        "format": "abi-native-transfer-r8-public-hostile-audit/1",
        "cases": rows,
        "case_count": len(rows),
        "unexpected_acceptances": 0,
        "trusted_scientific_booleans_consumed": 0,
    }
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--gate-dir", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        print(json.dumps({"status": "FAIL_CLOSED", "error": f"immutable output exists: {output}"}, indent=2))
        return 2
    try:
        value = audit(
            Path(args.config).resolve(),
            Path(args.campaign_root).resolve(),
            Path(args.gate_dir).resolve(),
            Path(args.verification).resolve(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    except (OSError, ValueError, AdversarialAuditError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
