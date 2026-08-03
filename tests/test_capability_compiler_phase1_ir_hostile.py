from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from abi.capability_compiler_phase1_extract import _canonical_sha
from abi.capability_compiler_phase1_ir import Phase1IRError, verify_ir
from abi.capability_pipeline import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
IR = ROOT / "results/abi_capability_compiler_phase1/final/normalized_acquisition_ir_v1.abicir"


def _members() -> dict[str, bytes]:
    with zipfile.ZipFile(IR, "r") as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _rebind_and_write(path: Path, members: dict[str, bytes]) -> None:
    manifest = json.loads(members["manifest.json"])
    for name, data in members.items():
        if name == "manifest.json":
            continue
        manifest["members"][name] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest["content_set_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest["members"])
    ).hexdigest()
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    members["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)


def _jsonl(data: bytes):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def _jsonl_bytes(rows) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )


def test_real_phase1_ir_verifies():
    result = verify_ir(IR)
    assert result["status"] == "PASS"
    assert result["record_count"] == 7_000


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("english_to_domain", "specialist data leaked"),
        ("token_count", "authoritative token accounting"),
        ("domain_training", "domain reference became acquisition"),
        ("final_access", "final isolation changed"),
        ("reclassify_failure", "reclassified"),
    ],
)
def test_semantic_tampering_fails_after_attacker_rebinds_hashes(tmp_path, mutation, message):
    members = _members()
    if mutation in {"english_to_domain", "token_count"}:
        rows = _jsonl(members["records.jsonl"])
        row = rows[0]
        row.pop("ir_record_id")
        if mutation == "english_to_domain":
            row["destination"] = "domain_cake"
            row["domain"] = "python"
            row["domain_labels"] = ["python"]
        else:
            row["authoritative_generated_token_ids"] = row[
                "authoritative_generated_token_ids"
            ][:-1]
        row["ir_record_id"] = _canonical_sha(row)
        members["records.jsonl"] = _jsonl_bytes(rows)
    elif mutation == "domain_training":
        rows = _jsonl(members["domain_reference.jsonl"])
        row = rows[0]
        row.pop("reference_record_sha256")
        row["training_eligible"] = True
        row["reference_record_sha256"] = _canonical_sha(row)
        members["domain_reference.jsonl"] = _jsonl_bytes(rows)
    elif mutation == "final_access":
        split = json.loads(members["split_manifest.json"])
        split["final_used_for_normalization_selection_or_repairs"] = True
        members["split_manifest.json"] = (
            json.dumps(split, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    else:
        rows = _jsonl(members["rejections.jsonl"])
        rows[0]["v1_failure_reclassified"] = True
        members["rejections.jsonl"] = _jsonl_bytes(rows)
    candidate = tmp_path / f"{mutation}.abicir"
    _rebind_and_write(candidate, members)
    with pytest.raises(Phase1IRError, match=message):
        verify_ir(candidate)


def test_path_traversal_and_member_extension_fail(tmp_path):
    members = _members()
    candidate = tmp_path / "traversal.abicir"
    with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr("../escape.json", b"{}")
    with pytest.raises(Phase1IRError, match="member set"):
        verify_ir(candidate)
