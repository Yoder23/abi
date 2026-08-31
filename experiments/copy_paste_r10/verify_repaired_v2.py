"""Registered one-line arithmetic repair for the frozen R10 v1 verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .runtime import sha256_file

ORIGINAL_VERIFIER_SHA256 = "fff1faf1febe6d25e91012c6b022613877df480e0c52873f4053b7087ef28037"
ORIGINAL_EXPRESSION = "len(expected_rows) // len(capability_ids) * len(conditions)"
REPAIRED_EXPRESSION = "len(expected_rows) * len(conditions)"


class R10VerifierRepairError(RuntimeError):
    """Raised when the bounded verifier repair or its bindings change."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R10VerifierRepairError(f"required JSON unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise R10VerifierRepairError(f"expected JSON object: {path}")
    return value


def _evidence_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("evidence_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _repaired_verify_function(root: Path) -> Any:
    path = root / "experiments/copy_paste_r10/verify.py"
    if sha256_file(path) != ORIGINAL_VERIFIER_SHA256:
        raise R10VerifierRepairError("frozen v1 verifier changed")
    source = path.read_text(encoding="utf-8")
    if source.count(ORIGINAL_EXPRESSION) != 1:
        raise R10VerifierRepairError("registered arithmetic expression inventory changed")
    repaired = source.replace(ORIGINAL_EXPRESSION, REPAIRED_EXPRESSION)
    namespace = {
        "__file__": str(path),
        "__name__": "experiments.copy_paste_r10._verified_repaired_v2",
        "__package__": "experiments.copy_paste_r10",
    }
    exec(compile(repaired, str(path), "exec"), namespace)
    return namespace["verify"]


def verify(amendment_path: Path, run_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    amendment = _json(amendment_path)
    if (
        amendment.get("format") != "abi-copy-paste-r10-verifier-amendment/2"
        or amendment.get("status") != "PREREGISTERED_POST_RUN_VERIFIER_ARITHMETIC_REPAIR"
        or amendment.get("scientific_thresholds_changed") is not False
        or amendment.get("raw_evidence_changed") is not False
    ):
        raise R10VerifierRepairError("verifier amendment boundary changed")
    bindings = amendment.get("bindings")
    if not isinstance(bindings, dict):
        raise R10VerifierRepairError("verifier amendment bindings missing")
    for relative, expected in bindings.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise R10VerifierRepairError("amendment binding escapes repository") from exc
        if not path.is_file() or sha256_file(path) != expected:
            raise R10VerifierRepairError(f"amendment binding changed: {relative}")
    if (
        amendment.get("original_expression") != ORIGINAL_EXPRESSION
        or amendment.get("repaired_expression") != REPAIRED_EXPRESSION
    ):
        raise R10VerifierRepairError("registered verifier repair changed")

    base_config = (root / str(amendment["base_config"]["path"])).resolve()
    receipt_path = run_dir / "receipt.json"
    receipt = _json(receipt_path)
    if (
        sha256_file(base_config) != amendment["base_config"]["sha256"]
        or sha256_file(receipt_path) != amendment["run_receipt"]["sha256"]
        or receipt.get("evidence_sha256") != amendment["run_receipt"]["evidence_sha256"]
        or _evidence_hash(receipt) != receipt.get("evidence_sha256")
    ):
        raise R10VerifierRepairError("base config or immutable run receipt changed")

    verifier = _repaired_verify_function(root)
    result = verifier(base_config, run_dir)
    result.pop("evidence_sha256", None)
    result["format"] = "abi-copy-paste-r10-verification/2"
    result["verifier_amendment"] = {
        "path": amendment_path.relative_to(root).as_posix(),
        "sha256": sha256_file(amendment_path),
        "repair_scope": "per-host expected-row arithmetic only",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        result = verify(Path(args.amendment).resolve(), Path(args.run_dir).resolve())
        if args.output:
            output = Path(args.output).resolve()
            if output.exists():
                raise R10VerifierRepairError(f"immutable verification exists: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    except (OSError, ValueError, KeyError, TypeError, R10VerifierRepairError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
