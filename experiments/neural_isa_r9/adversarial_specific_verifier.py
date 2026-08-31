"""Run hostile fail-closed controls against the R9 Gate A verifier and live replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes
from experiments.native_transfer_r8.native_host import sha256_file

from .live_replay_specific import replay
from .verify_specific_diagnostic import verify


class R9AdversarialError(RuntimeError):
    """Raised when hostile evidence is unexpectedly accepted."""


def _rehash_receipt(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    payload.pop("evidence_sha256", None)
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _expect_rejection(name: str, operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        operation()
    except Exception as exc:  # hostile controls intentionally exercise multiple failure types
        return {
            "control": name,
            "expected": "REJECT",
            "observed": "REJECT",
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    raise R9AdversarialError(f"hostile control was accepted: {name}")


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    controls: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="abi-r9-hostile-") as temporary:
        base = Path(temporary)

        missing_receipt = base / "missing-receipt"
        shutil.copytree(run_dir, missing_receipt)
        (missing_receipt / "receipt.json").unlink()
        controls.append(
            _expect_rejection(
                "missing_receipt", lambda: verify(config_path, missing_receipt)
            )
        )

        stale_receipt = base / "stale-receipt"
        shutil.copytree(run_dir, stale_receipt)
        value = json.loads((stale_receipt / "receipt.json").read_text(encoding="utf-8"))
        value["backend_optimizer_steps"] = int(value["backend_optimizer_steps"]) + 1
        (stale_receipt / "receipt.json").write_bytes(
            json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        controls.append(
            _expect_rejection("stale_receipt_hash", lambda: verify(config_path, stale_receipt))
        )

        missing_backend = base / "missing-backend"
        shutil.copytree(run_dir, missing_backend)
        receipt = json.loads((missing_backend / "receipt.json").read_text(encoding="utf-8"))
        (missing_backend / receipt["backend"]["path"]).unlink()
        controls.append(
            _expect_rejection("missing_backend", lambda: verify(config_path, missing_backend))
        )

        corrupt_backend = base / "corrupt-backend"
        shutil.copytree(run_dir, corrupt_backend)
        receipt = json.loads((corrupt_backend / "receipt.json").read_text(encoding="utf-8"))
        backend_path = corrupt_backend / receipt["backend"]["path"]
        payload = bytearray(backend_path.read_bytes())
        payload[len(payload) // 2] ^= 0x01
        backend_path.write_bytes(payload)
        controls.append(
            _expect_rejection("corrupt_backend", lambda: verify(config_path, corrupt_backend))
        )

        forged_raw = base / "forged-raw"
        shutil.copytree(run_dir, forged_raw)
        receipt_path = forged_raw / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        raw_path = forged_raw / receipt["observations"]["path"]
        rows = [json.loads(line) for line in raw_path.read_bytes().splitlines()]
        rows[0]["prediction_token_id"] = int(rows[0]["prediction_token_id"]) + 1
        raw_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
        receipt["observations"]["sha256"] = sha256_file(raw_path)
        receipt_path.write_bytes(
            json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        _rehash_receipt(receipt_path)
        controls.append(
            _expect_rejection(
                "forged_raw_with_recomputed_hashes",
                lambda: replay(config_path, forged_raw),
            )
        )

    result = {
        "format": "abi-neural-isa-r9-adversarial-verifier/1",
        "status": "PASS_HOSTILE_CONTROLS",
        "config_sha256": sha256_file(config_path),
        "run_receipt_sha256": sha256_file(run_dir / "receipt.json"),
        "controls": controls,
        "expected_rejections": len(controls),
        "observed_rejections": sum(row["observed"] == "REJECT" for row in controls),
        "unexpected_acceptances": 0,
        "trusted_scientific_booleans_consumed": 0,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        print(json.dumps({"status": "FAIL_CLOSED", "error": f"immutable output exists: {output}"}, indent=2))
        return 2
    try:
        value = audit(Path(args.config).resolve(), Path(args.run_dir).resolve())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    except (OSError, ValueError, R9AdversarialError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
