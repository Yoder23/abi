import hashlib
import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3_journal_recovery import JournalRecoveryError, recover_journal
from abi.capability_pipeline import canonical_json_bytes


def _row(probe: str, output: str) -> dict:
    row = {"probe_id": probe, "attempt_index": 0, "output": output}
    row["attempt_sha256"] = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
    return row


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_recovery_removes_only_identical_signed_duplicates(tmp_path: Path) -> None:
    source, output = tmp_path / "source.jsonl", tmp_path / "output.jsonl"
    first, second = _row("a", "one"), _row("b", "two")
    _write(source, [first, second, first])
    result = recover_journal(source, output)
    assert result["duplicate_rows_removed"] == 1
    assert result["output"]["rows"] == 2


def test_recovery_rejects_conflicting_duplicate_key(tmp_path: Path) -> None:
    source, output = tmp_path / "source.jsonl", tmp_path / "output.jsonl"
    _write(source, [_row("a", "one"), _row("a", "changed")])
    with pytest.raises(JournalRecoveryError, match="conflicting duplicate"):
        recover_journal(source, output)
