"""Build the exact B40 sequence-information pack for matched baselines."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    pack_examples,
    pack_manifest,
    sha256_file,
    tokenize_records,
)
from .capability_compiler_phase2_prepare import _tokenizer, _verified_snapshot
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_abi_lineage import _selected_rows
from . import capability_compiler_phase4_b50_baseline_pack as b50
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-b40-baseline-pack/1"
BUDGET = "B40"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_EXACT_B40_BASELINE_PACK"
        or protocol.get("budget") != BUDGET
        or protocol.get("artifact_construction_authorized") is not True
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("exact B40 baseline-pack governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B40 baseline-pack binding changed: {relative}")
    return protocol, sha256_file(path)


def normalize_memberships(
    selected: Mapping[str, Sequence[Mapping[str, Any]]], tokenizer: Any
) -> list[dict[str, Any]]:
    records = b50.normalize_memberships(selected, tokenizer)
    for row in records:
        row["format"] = "abi-phase4-exact-b40-baseline-membership/1"
        row["ir_record_id"] = hashlib.sha256(
            (
                f"b40-baseline:{row['source_artifact']}:{row['native_record_id']}"
            ).encode()
        ).hexdigest()
    return records


def _zip_member(name: str, payload: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info, payload


def write_records_archive(
    output: Path,
    records: Sequence[Mapping[str, Any]],
    accounting: Mapping[str, Any],
    selection_sha256: str,
) -> dict[str, Any]:
    if output.exists():
        raise Phase3Error(f"immutable exact B40 records archive exists: {output}")
    records_bytes = b"".join(canonical_json_bytes(dict(row)) for row in records)
    accounting_bytes = canonical_json_bytes(dict(accounting))
    manifest = {
        "format": "abi-phase4-exact-b40-baseline-records-manifest/1",
        "budget": BUDGET,
        "selection_sha256": selection_sha256,
        "records": len(records),
        "records_jsonl_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "accounting_sha256": hashlib.sha256(accounting_bytes).hexdigest(),
        "final_test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, payload in (
                ("accounting.json", accounting_bytes),
                ("manifest.json", canonical_json_bytes(manifest)),
                ("records.jsonl", records_bytes),
            ):
                info, value = _zip_member(name, payload)
                archive.writestr(info, value)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def _reconstruct(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[Any], dict[str, Any]]:
    lineage = _json(root / protocol["lineage_protocol"])
    for relative, expected in lineage["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B40 lineage binding changed: {relative}")
    manifest = _json(root / protocol["budget_manifest"])
    selected, budget = _selected_rows(root, lineage, manifest, BUDGET)
    tokenizer = _tokenizer(_verified_snapshot(root))
    records = normalize_memberships(selected, tokenizer)
    attempts = [
        json.loads(line)
        for line in (root / protocol["source_attempt_journal"]).read_bytes().splitlines()
        if line.strip()
    ]
    accounting = b50.source_attempt_accounting(selected, attempts)
    examples = tokenize_records(records, tokenizer)
    maximum_record_tokens = max(len(row.input_ids) for row in examples)
    if maximum_record_tokens > int(protocol["packing_context"]):
        raise Phase3Error("exact B40 record exceeds frozen baseline context")
    packs = pack_examples(
        examples,
        max_tokens=int(protocol["packing_context"]),
        seed=int(protocol["packing_seed"]),
    )
    packed = pack_manifest(packs)
    packed["maximum_record_tokens"] = maximum_record_tokens
    return records, accounting, budget, packs, packed


def _gates(
    protocol: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    accounting: Mapping[str, Any],
    budget: Mapping[str, Any],
    packed: Mapping[str, Any],
) -> dict[str, bool]:
    expected = protocol["expected"]
    capabilities = Counter(str(row["capability"]) for row in records)
    sources = Counter(str(row["source_artifact"]) for row in records)
    destinations = Counter(str(row["destination"]) for row in records)
    normalized = Counter(str(row["sequence_normalization"]) for row in records)
    return {
        "selection_exact": str(budget["selection_sha256"])
        == expected["selection_sha256"],
        "membership_count_exact": len(records) == int(expected["record_memberships"]),
        "source_memberships_exact": dict(sources) == expected["source_memberships"],
        "unique_attempt_count_exact": int(accounting["unique_source_attempts"])
        == int(expected["unique_source_attempts"]),
        "duplicate_count_exact": int(accounting["duplicate_memberships"])
        == int(expected["duplicate_memberships"]),
        "teacher_output_tokens_exact": int(
            accounting["authoritative_teacher_output_tokens"]
        )
        == int(expected["authoritative_teacher_output_tokens"]),
        "surface_normalized_host_memberships_exact": normalized[
            "retokenized_after_closed_v479_surface_normalization"
        ]
        == int(expected["surface_normalized_host_memberships"]),
        "all_memberships_namespaced_unique": len(
            {str(row["ir_record_id"]) for row in records}
        )
        == len(records),
        "all_capabilities_present": set(capabilities) == set(CAPABILITIES),
        "english_core_only": destinations == {"english_core": len(records)},
        "sequence_channels_nonempty": all(
            row["rendered_generation_prompt"]
            and row["authoritative_generated_token_ids"]
            for row in records
        ),
        "context_conformant": int(packed["maximum_record_tokens"])
        <= int(protocol["packing_context"]),
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_model_loading_absent": True,
        "final_test_not_accessed": True,
    }


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    records, accounting, budget, packs, packed = _reconstruct(root, protocol)
    gates = _gates(protocol, records, accounting, budget, packed)
    return {
        "format": "abi-capability-compiler-phase4-b40-baseline-pack-preflight/1",
        "status": "PASS_EXACT_B40_BASELINE_PACK_PREFLIGHT"
        if all(gates.values())
        else "FAIL_EXACT_B40_BASELINE_PACK_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "record_memberships": len(records),
        "pack_count": len(packs),
        "pack_manifest": {
            key: packed[key]
            for key in (
                "format",
                "pack_count",
                "record_count",
                "input_tokens",
                "response_tokens",
                "content_sha256",
                "maximum_record_tokens",
            )
        },
        "imported_information": accounting,
        "gates": gates,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
    }


def run(
    root: Path,
    protocol_path: Path,
    records_output: Path,
    pack_output: Path,
    result_output: Path,
) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if any(path.exists() for path in (records_output, pack_output, result_output)):
        raise Phase3Error("immutable exact B40 baseline-pack output exists")
    records, accounting, budget, packs, packed = _reconstruct(root, protocol)
    gates = _gates(protocol, records, accounting, budget, packed)
    if not all(gates.values()):
        raise Phase3Error(f"exact B40 baseline-pack reconstruction failed: {gates}")
    archive_manifest = write_records_archive(
        records_output, records, accounting, str(budget["selection_sha256"])
    )
    packed.update(
        {
            "status": "PASS",
            "format": "abi-phase4-exact-b40-baseline-pack-manifest/1",
            "budget": BUDGET,
            "selection_sha256": str(budget["selection_sha256"]),
            "records_archive": records_output.relative_to(root).as_posix(),
            "records_archive_sha256": sha256_file(records_output),
            "packing_seed": int(protocol["packing_seed"]),
            "packing_context": int(protocol["packing_context"]),
            "source_model": protocol["source_model"],
            "source_revision": protocol["source_revision"],
            "candidate_training_performed": False,
        }
    )
    _write_immutable(pack_output, canonical_json_bytes(packed))
    result = {
        "format": "abi-capability-compiler-phase4-b40-baseline-pack-result/1",
        "status": "PASS_EXACT_B40_BASELINE_SEQUENCE_PACK_READY",
        "protocol_sha256": protocol_sha,
        "budget": budget,
        "source_memberships": dict(sorted(Counter(str(row["source_artifact"]) for row in records).items())),
        "capability_memberships": dict(sorted(Counter(str(row["capability"]) for row in records).items())),
        "destination_memberships": {"english_core": len(records)},
        "imported_information": accounting,
        "records_archive": {
            "path": records_output.relative_to(root).as_posix(),
            "sha256": sha256_file(records_output),
            "bytes": records_output.stat().st_size,
            "manifest": archive_manifest,
        },
        "sequence_pack": {
            "path": pack_output.relative_to(root).as_posix(),
            "sha256": sha256_file(pack_output),
            **{key: packed[key] for key in ("pack_count", "record_count", "input_tokens", "response_tokens", "content_sha256", "maximum_record_tokens")},
        },
        "baseline_channels": {
            "L0": ["teacher_generated_sequence"],
            "L1": ["teacher_generated_sequence"],
            "D0": ["teacher_generated_sequence"],
        },
        "gates": gates,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Exact B40 equal-sequence baseline pack only. No baseline training, matched comparison, final test, Phase 4, or ABI-superiority result.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    result_output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(result_output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--records-output")
    parser.add_argument("--pack-output")
    parser.add_argument("--result-output")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    if args.preflight:
        result = preflight(root, root / args.protocol)
    elif args.records_output and args.pack_output and args.result_output:
        result = run(
            root,
            root / args.protocol,
            root / args.records_output,
            root / args.pack_output,
            root / args.result_output,
        )
    else:
        raise Phase3Error("select preflight or all B40 pack outputs")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if str(result["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
