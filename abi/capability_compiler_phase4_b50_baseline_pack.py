"""Build the exact B50 sequence-information pack for matched Phase 4 baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    Phase2Error,
    canonical_json_bytes,
    pack_examples,
    pack_manifest,
    sha256_file,
    tokenize_records,
)
from .capability_compiler_phase2_prepare import _tokenizer, _verified_snapshot
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_abi_lineage import _selected_rows
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-b50-baseline-pack/2"
ARTIFACT_ORDER = ("phase1_ir", "v138_targeted_ir", "v480_host_supervision")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_EXACT_B50_BASELINE_PACK"
        or protocol.get("artifact_construction_authorized") is not True
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("budget") != "B50"
    ):
        raise Phase3Error("exact B50 baseline-pack governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B50 baseline-pack binding changed: {relative}")
    return protocol, sha256_file(path)


def _archive_rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        return [json.loads(line) for line in archive.read("records.jsonl").splitlines()]


def _native_id(artifact: str, row: Mapping[str, Any]) -> str:
    key = "record_id" if artifact == "v480_host_supervision" else "ir_record_id"
    value = str(row.get(key, ""))
    if not value:
        raise Phase3Error(f"{artifact} record lacks {key}")
    return value


def membership_id(artifact: str, native_id: str) -> str:
    return hashlib.sha256(f"b50-baseline:{artifact}:{native_id}".encode()).hexdigest()


def _render_host_prompt(tokenizer: Any, prompt: str) -> str:
    return str(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def normalize_memberships(
    selected: Mapping[str, Sequence[Mapping[str, Any]]], tokenizer: Any
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for artifact in ARTIFACT_ORDER:
        for row in selected[artifact]:
            native = _native_id(artifact, row)
            if artifact == "v480_host_supervision":
                prompt = str(row["host_prompt"])
                rendered = _render_host_prompt(tokenizer, prompt)
                source_response_ids = [
                    int(value) for value in row["source_authoritative_generated_token_ids"]
                ]
                output = str(row["output"])
                teacher_tokens = int(row["source_teacher_output_tokens"])
                destination = "english_core"
                if not source_response_ids:
                    raise Phase3Error("empty host source response tokens")
                rendered_output_ids = [
                    int(value)
                    for value in tokenizer(output, add_special_tokens=False).input_ids
                ] + [source_response_ids[-1]]
                surface_normalized = bool(row.get("surface_normalization_steps"))
                if not surface_normalized and rendered_output_ids != source_response_ids:
                    raise Phase3Error("unchanged host output no longer matches source tokens")
                response_ids = (
                    rendered_output_ids if surface_normalized else source_response_ids
                )
                sequence_normalization = (
                    "retokenized_after_closed_v479_surface_normalization"
                    if surface_normalized
                    else "none"
                )
            else:
                prompt = str(row["normalized_generation_prompt"])
                rendered = str(row["rendered_generation_prompt"])
                response_ids = [
                    int(value) for value in row["authoritative_generated_token_ids"]
                ]
                output = str(row["normalized_output"])
                teacher_tokens = int(row["authoritative_teacher_tokens"])
                destination = str(row["destination"])
                sequence_normalization = "none"
            if not prompt or not rendered or not output or not response_ids:
                raise Phase3Error(f"empty exact B50 sequence channel: {artifact}:{native}")
            normalized.append(
                {
                    "format": "abi-phase4-exact-b50-baseline-membership/1",
                    "ir_record_id": membership_id(artifact, native),
                    "native_record_id": native,
                    "source_artifact": artifact,
                    "source_attempt_sha256": str(row["source_attempt_sha256"]),
                    "capability": str(row["capability"]),
                    "destination": destination,
                    "normalized_generation_prompt": prompt,
                    "normalized_generation_prompt_sha256": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "rendered_generation_prompt": rendered,
                    "rendered_generation_prompt_sha256": hashlib.sha256(
                        rendered.encode("utf-8")
                    ).hexdigest(),
                    "normalized_output": output,
                    "normalized_output_sha256": hashlib.sha256(
                        output.encode("utf-8")
                    ).hexdigest(),
                    "authoritative_generated_token_ids": response_ids,
                    "authoritative_teacher_tokens": teacher_tokens,
                    "baseline_sequence_tokens": len(response_ids),
                    "sequence_normalization": sequence_normalization,
                    "channel": "teacher_generated_sequence",
                    "split": "acquisition",
                }
            )
    return sorted(
        normalized,
        key=lambda row: (
            ARTIFACT_ORDER.index(str(row["source_artifact"])),
            str(row["capability"]),
            str(row["native_record_id"]),
        ),
    )


def source_attempt_accounting(
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    source_attempt_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempts_by_hash = {
        str(row["attempt_sha256"]): row for row in source_attempt_rows
    }
    union: dict[str, dict[str, Any]] = {}
    memberships = 0
    membership_tokens = 0
    for artifact in ARTIFACT_ORDER:
        for row in selected[artifact]:
            memberships += 1
            attempt = str(row["source_attempt_sha256"])
            if artifact == "v480_host_supervision":
                canonical = attempts_by_hash.get(attempt)
                if canonical is None:
                    raise Phase3Error("host membership lacks source-attempt lineage")
                token_count = int(row["source_teacher_output_tokens"])
                if token_count != int(canonical["teacher_tokens"]):
                    raise Phase3Error("host/source teacher-token accounting diverged")
                if (
                    [int(value) for value in row["source_authoritative_generated_token_ids"]]
                    != [int(value) for value in canonical["authoritative_generated_token_ids"]]
                    or str(row["source_output_sha256"])
                    != str(canonical["output_sha256"])
                    or str(row["source_generation_prompt_sha256"])
                    != str(canonical["generation_prompt_sha256"])
                    or int(row["source_teacher_input_tokens"])
                    != int(canonical["teacher_input_tokens"])
                ):
                    raise Phase3Error("host/source attempt provenance diverged")
                prompt = str(canonical["generation_prompt"])
                output = str(canonical["output"])
                output_sha = str(canonical["output_sha256"])
                input_tokens = int(canonical["teacher_input_tokens"])
            else:
                token_count = int(row["authoritative_teacher_tokens"])
                prompt = str(row["raw_generation_prompt"])
                output = str(row["raw_output"])
                output_sha = str(row["raw_output_sha256"])
                input_tokens = int(row["teacher_input_tokens"])
            membership_tokens += token_count
            facts = {
                "teacher_input_tokens": input_tokens,
                "teacher_output_tokens": token_count,
                "raw_prompt_bytes": len(prompt.encode("utf-8")),
                "raw_teacher_output_bytes": len(output.encode("utf-8")),
                "teacher_output_sha256": output_sha,
            }
            previous = union.setdefault(attempt, facts)
            if previous != facts:
                raise Phase3Error("duplicate source attempt carries different information")
    duplicates = memberships - len(union)
    return {
        "record_memberships": memberships,
        "unique_source_attempts": len(union),
        "duplicate_memberships": duplicates,
        "membership_teacher_output_tokens": membership_tokens,
        "authoritative_teacher_input_tokens": sum(
            row["teacher_input_tokens"] for row in union.values()
        ),
        "authoritative_teacher_output_tokens": sum(
            row["teacher_output_tokens"] for row in union.values()
        ),
        "unique_raw_prompt_utf8_bytes": sum(
            row["raw_prompt_bytes"] for row in union.values()
        ),
        "unique_raw_teacher_output_utf8_bytes": sum(
            row["raw_teacher_output_bytes"] for row in union.values()
        ),
        "source_attempt_manifest_sha256": hashlib.sha256(
            canonical_json_bytes({key: union[key] for key in sorted(union)})
        ).hexdigest(),
        "stored_logits": 0,
        "stored_hidden_activations": 0,
        "copied_source_parameters": 0,
    }


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
        raise Phase3Error(f"immutable exact B50 records archive exists: {output}")
    records_bytes = b"".join(canonical_json_bytes(dict(row)) for row in records)
    accounting_bytes = canonical_json_bytes(dict(accounting))
    manifest = {
        "format": "abi-phase4-exact-b50-baseline-records-manifest/1",
        "budget": "B50",
        "selection_sha256": selection_sha256,
        "records": len(records),
        "records_jsonl_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "accounting_sha256": hashlib.sha256(accounting_bytes).hexdigest(),
        "final_test_accessed": False,
    }
    manifest_bytes = canonical_json_bytes(manifest)
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
                ("manifest.json", manifest_bytes),
                ("records.jsonl", records_bytes),
            ):
                info, value = _zip_member(name, payload)
                archive.writestr(info, value)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def run(
    root: Path,
    protocol_path: Path,
    records_output: Path,
    pack_output: Path,
    result_output: Path,
) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    for path in (records_output, pack_output, result_output):
        if path.exists():
            raise Phase3Error(f"immutable exact B50 baseline-pack output exists: {path}")
    lineage = _json(root / protocol["lineage_protocol"])
    manifest = _json(root / protocol["budget_manifest"])
    selected, budget = _selected_rows(root, lineage, manifest, "B50")
    tokenizer = _tokenizer(_verified_snapshot(root))
    records = normalize_memberships(selected, tokenizer)
    accounting = source_attempt_accounting(
        selected, [
            json.loads(line)
            for line in (root / protocol["source_attempt_journal"])
            .read_bytes()
            .splitlines()
        ]
    )
    source_counts = Counter(str(row["source_artifact"]) for row in records)
    capability_counts = Counter(str(row["capability"]) for row in records)
    destination_counts = Counter(str(row["destination"]) for row in records)
    sequence_normalization_counts = Counter(
        str(row["sequence_normalization"]) for row in records
    )
    expected_counts = {key: int(value) for key, value in budget["records"].items()}
    if dict(source_counts) != expected_counts:
        raise Phase3Error("exact B50 membership counts changed")
    if (
        accounting["record_memberships"] != int(budget["record_memberships"])
        or accounting["unique_source_attempts"] != int(budget["unique_source_attempts"])
        or accounting["duplicate_memberships"] != int(budget["duplicate_memberships"])
        or accounting["authoritative_teacher_output_tokens"]
        != int(budget["authoritative_teacher_output_tokens"])
    ):
        raise Phase3Error("exact B50 imported-information accounting changed")
    if set(capability_counts) != set(CAPABILITIES) or destination_counts != {
        "english_core": len(records)
    }:
        raise Phase3Error("exact B50 English/capability segregation changed")
    archive_manifest = write_records_archive(
        records_output, records, accounting, str(budget["selection_sha256"])
    )
    examples = tokenize_records(records, tokenizer)
    maximum_record_tokens = max(len(row.input_ids) for row in examples)
    if maximum_record_tokens > int(protocol["packing_context"]):
        raise Phase2Error("exact B50 record exceeds frozen baseline context")
    packs = pack_examples(
        examples,
        max_tokens=int(protocol["packing_context"]),
        seed=int(protocol["packing_seed"]),
    )
    packed = pack_manifest(packs)
    packed.update(
        {
            "status": "PASS",
            "format": "abi-phase4-exact-b50-baseline-pack-manifest/1",
            "budget": "B50",
            "selection_sha256": str(budget["selection_sha256"]),
            "records_archive": records_output.relative_to(root).as_posix(),
            "records_archive_sha256": sha256_file(records_output),
            "packing_seed": int(protocol["packing_seed"]),
            "packing_context": int(protocol["packing_context"]),
            "maximum_record_tokens": maximum_record_tokens,
            "source_model": protocol["source_model"],
            "source_revision": protocol["source_revision"],
            "candidate_training_performed": False,
        }
    )
    _write_immutable(pack_output, canonical_json_bytes(packed))
    gates = {
        "selection_exact": str(budget["selection_sha256"])
        == protocol["expected"]["selection_sha256"],
        "membership_count_exact": len(records)
        == int(protocol["expected"]["record_memberships"]),
        "unique_attempt_count_exact": accounting["unique_source_attempts"]
        == int(protocol["expected"]["unique_source_attempts"]),
        "teacher_output_tokens_exact": accounting[
            "authoritative_teacher_output_tokens"
        ]
        == int(protocol["expected"]["authoritative_teacher_output_tokens"]),
        "all_memberships_namespaced_unique": len(
            {str(row["ir_record_id"]) for row in records}
        )
        == len(records),
        "all_source_attempts_bound": all(
            len(str(row["source_attempt_sha256"])) == 64 for row in records
        ),
        "all_capabilities_present": set(capability_counts) == set(CAPABILITIES),
        "english_core_only": destination_counts == {"english_core": len(records)},
        "all_sequence_channels_nonempty": all(
            row["rendered_generation_prompt"]
            and row["authoritative_generated_token_ids"]
            for row in records
        ),
        "surface_normalized_rows_retokenized": sequence_normalization_counts[
            "retokenized_after_closed_v479_surface_normalization"
        ]
        == int(protocol["expected"]["surface_normalized_host_memberships"]),
        "context_conformant": maximum_record_tokens
        <= int(protocol["packing_context"]),
        "records_archive_self_consistent": archive_manifest["records"]
        == len(records),
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_model_loading_absent": True,
        "final_test_absent": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-b50-baseline-pack-result/1",
        "status": "PASS_EXACT_B50_BASELINE_SEQUENCE_PACK_READY"
        if all(gates.values())
        else "FAIL_EXACT_B50_BASELINE_SEQUENCE_PACK",
        "protocol_sha256": protocol_sha,
        "budget": budget,
        "source_memberships": dict(sorted(source_counts.items())),
        "capability_memberships": dict(sorted(capability_counts.items())),
        "destination_memberships": dict(sorted(destination_counts.items())),
        "sequence_normalization_memberships": dict(
            sorted(sequence_normalization_counts.items())
        ),
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
            "pack_count": packed["pack_count"],
            "record_count": packed["record_count"],
            "input_tokens": packed["input_tokens"],
            "response_tokens": packed["response_tokens"],
            "content_sha256": packed["content_sha256"],
            "maximum_record_tokens": maximum_record_tokens,
        },
        "baseline_channels": {
            "L0": ["teacher_generated_sequence"],
            "L1": ["teacher_generated_sequence"],
            "D0": ["teacher_generated_sequence"],
            "D1": ["teacher_generated_sequence", "prospective_top64_teacher_logits"],
            "D2": ["teacher_generated_sequence", "prospective_top64_teacher_logits"],
        },
        "gates": gates,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "stable_minimum_established": False,
        "claim_boundary": (
            "Exact B50 sequence-pack construction only. D1/D2 top-64 logits do not "
            "yet exist for this pack. No baseline training, matched frontier, minimum, "
            "final-test, Phase 4, or ABI-superiority result."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    _write_immutable(
        result_output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--records-output", required=True)
    parser.add_argument("--pack-output", required=True)
    parser.add_argument("--result-output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(
        root,
        root / args.protocol,
        root / args.records_output,
        root / args.pack_output,
        root / args.result_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
