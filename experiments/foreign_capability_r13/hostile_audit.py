"""Adversarial fail-closed audit of the hardened R13 verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .core import R13Error, evidence_hash, json_object, write_json_once
from .verify import verify
from .verify_live import verify_final


def _expect_rejection(label: str, operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        operation()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration) as exc:
        return {"case": label, "expected": "REJECT", "actual": "REJECT", "error": str(exc)}
    raise R13Error(f"hostile case was accepted: {label}")


def run(
    config_path: Path,
    reveal_path: Path,
    run_dir: Path,
    live_dir: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise R13Error(f"immutable hostile audit exists: {output}")
    baseline_static = verify(config_path, reveal_path, run_dir)
    baseline_final = verify_final(config_path, reveal_path, run_dir, live_dir)
    cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="abi-r13-hostile-") as temporary:
        temporary_root = Path(temporary)
        work_run = temporary_root / "run"
        work_live = temporary_root / "live"
        shutil.copytree(run_dir, work_run)
        shutil.copytree(live_dir, work_live)

        receipt = work_run / "receipt.json"
        receipt_bytes = receipt.read_bytes()
        receipt.rename(work_run / "receipt.removed")
        cases.append(
            _expect_rejection(
                "missing_run_receipt",
                lambda: verify(config_path, reveal_path, work_run),
            )
        )
        (work_run / "receipt.removed").rename(receipt)

        run_value = json_object(receipt)
        adapter = work_run / run_value["source_acquisitions"][0]["adapter_artifact"]["path"]
        removed_adapter = adapter.with_suffix(".removed")
        adapter.rename(removed_adapter)
        cases.append(
            _expect_rejection(
                "missing_source_adapter",
                lambda: verify(config_path, reveal_path, work_run),
            )
        )
        removed_adapter.rename(adapter)

        package = work_run / "packages" / run_value["packages"]["after"][0]["path"]
        package_bytes = package.read_bytes()
        corrupted = bytearray(package_bytes)
        corrupted[len(corrupted) // 2] ^= 1
        package.write_bytes(corrupted)
        cases.append(
            _expect_rejection(
                "corrupted_package",
                lambda: verify(config_path, reveal_path, work_run),
            )
        )
        package.write_bytes(package_bytes)

        source_rows_path = work_run / run_value["source_observations"]["path"]
        source_rows_bytes = source_rows_path.read_bytes()
        rows = [json.loads(line) for line in source_rows_bytes.decode().splitlines()]
        rows[0]["canonical_prediction"] = (int(rows[0]["canonical_prediction"]) + 1) % 8
        source_rows_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
        forged = json_object(receipt)
        forged["source_observations"]["sha256"] = hashlib.sha256(
            source_rows_path.read_bytes()
        ).hexdigest()
        forged.pop("evidence_sha256", None)
        forged["evidence_sha256"] = evidence_hash(forged)
        receipt.write_bytes(json.dumps(forged, indent=2, sort_keys=True).encode() + b"\n")
        cases.append(
            _expect_rejection(
                "hash_consistent_forged_source_row",
                lambda: verify(config_path, reveal_path, work_run),
            )
        )
        source_rows_path.write_bytes(source_rows_bytes)
        receipt.write_bytes(receipt_bytes)

        recipient_rows = work_run / "recipients" / "pythia" / "observations.jsonl"
        removed_rows = recipient_rows.with_suffix(".removed")
        recipient_rows.rename(removed_rows)
        cases.append(
            _expect_rejection(
                "missing_recipient_rows",
                lambda: verify(config_path, reveal_path, work_run),
            )
        )
        removed_rows.rename(recipient_rows)

        forged_status = json_object(receipt)
        forged_status["status"] = "FORGED_SCIENTIFIC_PASS"
        forged_status.pop("evidence_sha256", None)
        forged_status["evidence_sha256"] = evidence_hash(forged_status)
        receipt.write_bytes(
            json.dumps(forged_status, indent=2, sort_keys=True).encode() + b"\n"
        )
        status_result = verify(config_path, reveal_path, work_run)
        if status_result != baseline_static:
            raise R13Error("forged status changed recomputed result")
        cases.append(
            {
                "case": "forged_scientific_boolean_ignored",
                "expected": "UNCHANGED_PASS",
                "actual": "UNCHANGED_PASS",
            }
        )
        receipt.write_bytes(receipt_bytes)

        bad_reveal = temporary_root / "bad_reveal.json"
        bad = json_object(reveal_path)
        bad["secret_hex"] = "00" * 32
        bad_reveal.write_bytes(json.dumps(bad, indent=2, sort_keys=True).encode() + b"\n")
        cases.append(
            _expect_rejection(
                "wrong_heldout_reveal",
                lambda: verify(config_path, bad_reveal, work_run),
            )
        )

        live_receipt = work_live / "receipt.json"
        live_receipt_bytes = live_receipt.read_bytes()
        live_source = next((work_live / "sources").glob("*/observations.jsonl"))
        live_source_bytes = live_source.read_bytes()
        live_source.write_bytes(live_source_bytes + b"\n")
        live_value = json_object(live_receipt)
        source_item = next(
            item
            for item in live_value["source_replays"]
            if item["capability_id"] == live_source.parent.name
        )
        source_item["actual_sha256"] = hashlib.sha256(live_source.read_bytes()).hexdigest()
        source_item["byte_exact"] = True
        live_value.pop("evidence_sha256", None)
        live_value["evidence_sha256"] = evidence_hash(live_value)
        live_receipt.write_bytes(
            json.dumps(live_value, indent=2, sort_keys=True).encode() + b"\n"
        )
        cases.append(
            _expect_rejection(
                "hash_consistent_forged_live_replay",
                lambda: verify_final(config_path, reveal_path, work_run, work_live),
            )
        )
        live_source.write_bytes(live_source_bytes)
        live_receipt.write_bytes(live_receipt_bytes)

        live_recipient = work_live / "recipients" / "qwen2" / "observations.jsonl"
        removed_live = live_recipient.with_suffix(".removed")
        live_recipient.rename(removed_live)
        cases.append(
            _expect_rejection(
                "missing_live_recipient_rows",
                lambda: verify_final(config_path, reveal_path, work_run, work_live),
            )
        )
        removed_live.rename(live_recipient)

        if verify(config_path, reveal_path, work_run) != baseline_static:
            raise R13Error("static baseline did not restore after hostile audit")
        if verify_final(config_path, reveal_path, work_run, work_live) != baseline_final:
            raise R13Error("final baseline did not restore after hostile audit")

    result = {
        "format": "abi-r13-hostile-audit/1",
        "verdict": "PASS",
        "cases": cases,
        "expected_outcomes": len(cases),
        "matched_outcomes": sum(item["expected"] == item["actual"] for item in cases),
        "original_evidence_modified": False,
        "baseline_static_evidence_sha256": baseline_static["evidence_sha256"],
        "baseline_final_evidence_sha256": baseline_final["evidence_sha256"],
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reveal", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--live-replay-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = run(
            Path(args.config).resolve(),
            Path(args.reveal).resolve(),
            Path(args.run_dir).resolve(),
            Path(args.live_replay_dir).resolve(),
            Path(args.output).resolve(),
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
