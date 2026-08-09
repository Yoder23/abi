"""Recover an append-only extraction journal containing identical duplicates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from .capability_pipeline import canonical_json_bytes


PROTOCOL_FORMAT = "abi-capability-compiler-phase3-journal-recovery/1"
RESULT_FORMAT = "abi-capability-compiler-phase3-journal-recovery-result/1"


class JournalRecoveryError(RuntimeError):
    """Raised when recovery cannot be performed without changing evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recover_journal(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise JournalRecoveryError(f"recovery output is immutable: {output}")
    retained: dict[tuple[str, int], tuple[dict[str, Any], str]] = {}
    source_rows = 0
    duplicate_rows = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            source_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JournalRecoveryError(f"invalid JSON at line {line_number}") from exc
            claimed = row.pop("attempt_sha256", None)
            actual = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
            row["attempt_sha256"] = claimed
            if claimed != actual:
                raise JournalRecoveryError(f"invalid attempt hash at line {line_number}")
            key = (str(row.get("probe_id")), int(row.get("attempt_index", -1)))
            canonical_line = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            prior = retained.get(key)
            if prior is None:
                retained[key] = (row, canonical_line)
            elif prior[1] != canonical_line:
                raise JournalRecoveryError(f"conflicting duplicate attempt: {key}")
            else:
                duplicate_rows += 1
    if not retained:
        raise JournalRecoveryError("source journal is empty")
    ordered = sorted(
        (value[0] for value in retained.values()),
        key=lambda row: (str(row["probe_id"]), int(row["attempt_index"])),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in ordered:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    counts = Counter(int(row["attempt_index"]) for row in ordered)
    return {
        "format": RESULT_FORMAT,
        "status": "PASS_IDENTICAL_DUPLICATES_REMOVED",
        "source": {"path": source.as_posix(), "sha256": _sha256(source), "bytes": source.stat().st_size, "rows": source_rows},
        "output": {"path": output.as_posix(), "sha256": _sha256(output), "bytes": output.stat().st_size, "rows": len(ordered)},
        "duplicate_rows_removed": duplicate_rows,
        "conflicting_duplicate_keys": 0,
        "invalid_json_lines": 0,
        "invalid_attempt_hashes": 0,
        "attempt_counts": {str(key): value for key, value in sorted(counts.items())},
        "selection_rule": "retain first occurrence only when the complete canonical signed attempt is byte-equivalent",
        "claim_boundary": "Journal recovery only; no source response was quality-selected and no acquisition or Phase 3 pass is claimed."
    }


def _verify_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise JournalRecoveryError("unsupported recovery protocol")
    root = path.resolve().parent
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or _sha256(target) != expected:
            raise JournalRecoveryError(f"binding mismatch: {relative}")
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = _verify_protocol(protocol_path)
    root = protocol_path.parent
    result_path = Path(args.result).resolve()
    if result_path.exists():
        raise JournalRecoveryError(f"result is immutable: {result_path}")
    result = recover_journal(root / protocol["source_journal"], Path(args.output).resolve())
    gates = protocol["pass_gates"]
    if (
        result["source"]["rows"] != int(gates["source_rows"])
        or result["output"]["rows"] != int(gates["unique_rows"])
        or result["duplicate_rows_removed"] != int(gates["identical_duplicate_rows"])
    ):
        raise JournalRecoveryError("recovery counts changed from preregistration")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
