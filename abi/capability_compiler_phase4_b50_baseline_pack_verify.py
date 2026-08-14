"""Independent hostile verification of the exact B50 baseline sequence pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-b50-baseline-pack-verify/2"
ARTIFACT_ORDER = ("phase1_ir", "v138_targeted_ir", "v480_host_supervision")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_INDEPENDENT_EXACT_B50_PACK_VERIFY"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("independent exact B50 pack verification governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"independent exact B50 pack binding changed: {relative}")
    return protocol, sha256_file(path)


def _archive_entries(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {"accounting.json", "manifest.json", "records.jsonl"}:
            raise Phase3Error("exact B50 records archive member set changed")
        return {name: archive.read(name) for name in archive.namelist()}


def _rows(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        return [
            json.loads(line)
            for line in archive.read("records.jsonl").splitlines()
            if line.strip()
        ]


def rank_within_strata(
    rows: Sequence[Mapping[str, Any]],
    *,
    artifact: str,
    salt: str,
    groups: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[name]) for name in groups)].append(row)
    ordered: list[dict[str, Any]] = []
    for key in sorted(grouped):
        ordered.extend(
            dict(row)
            for row in sorted(
                grouped[key],
                key=lambda row: hashlib.sha256(
                    f"{salt}:{artifact}:{':'.join(key)}:{row['source_attempt_sha256']}".encode(
                        "ascii"
                    )
                ).hexdigest(),
            )
        )
    return ordered


def prefix_per_stratum(
    rows: Sequence[Mapping[str, Any]], groups: tuple[str, ...], depth: int
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, ...]] = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(str(row[name]) for name in groups)
        if counts[key] < depth:
            selected.append(dict(row))
            counts[key] += 1
    if not counts or set(counts.values()) != {depth}:
        raise Phase3Error("independent exact B50 stratum depth changed")
    return selected


def independent_selection(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    salt: str,
    weak_capabilities: set[str],
    budget: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    filtered = {
        key: [dict(row) for row in values] for key, values in source_rows.items()
    }
    filtered["v138_targeted_ir"] = [
        row
        for row in filtered["v138_targeted_ir"]
        if str(row["capability"]) in weak_capabilities
    ]
    ranked = {
        "phase1_ir": rank_within_strata(
            filtered["phase1_ir"],
            artifact="phase1_ir",
            salt=salt,
            groups=("capability",),
        ),
        "v138_targeted_ir": rank_within_strata(
            filtered["v138_targeted_ir"],
            artifact="v138_targeted_ir",
            salt=salt,
            groups=("capability",),
        ),
        "v480_host_supervision": rank_within_strata(
            filtered["v480_host_supervision"],
            artifact="v480_host_supervision",
            salt=salt,
            groups=("capability", "builder"),
        ),
    }
    return {
        "phase1_ir": prefix_per_stratum(
            ranked["phase1_ir"], ("capability",), int(budget["phase1_per_capability"])
        ),
        "v138_targeted_ir": prefix_per_stratum(
            ranked["v138_targeted_ir"],
            ("capability",),
            int(budget["targeted_per_weak_capability"]),
        ),
        "v480_host_supervision": prefix_per_stratum(
            ranked["v480_host_supervision"],
            ("capability", "builder"),
            int(budget["host_per_capability_builder"]),
        ),
    }


def _membership_id(artifact: str, native: str) -> str:
    return hashlib.sha256(f"b50-baseline:{artifact}:{native}".encode()).hexdigest()


def expected_memberships(
    selected: Mapping[str, Sequence[Mapping[str, Any]]], tokenizer: Any
) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    for artifact in ARTIFACT_ORDER:
        for row in selected[artifact]:
            key = "record_id" if artifact == "v480_host_supervision" else "ir_record_id"
            native = str(row[key])
            if artifact == "v480_host_supervision":
                prompt = str(row["host_prompt"])
                rendered = str(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                output = str(row["output"])
                source_ids = [
                    int(value) for value in row["source_authoritative_generated_token_ids"]
                ]
                rebuilt_ids = [
                    int(value)
                    for value in tokenizer(output, add_special_tokens=False).input_ids
                ] + [source_ids[-1]]
                surface = bool(row.get("surface_normalization_steps"))
                if not surface and rebuilt_ids != source_ids:
                    raise Phase3Error("independent unchanged host token check failed")
                response_ids = rebuilt_ids if surface else source_ids
                destination = "english_core"
                teacher_tokens = int(row["source_teacher_output_tokens"])
                normalization = (
                    "retokenized_after_closed_v479_surface_normalization"
                    if surface
                    else "none"
                )
            else:
                prompt = str(row["normalized_generation_prompt"])
                rendered = str(row["rendered_generation_prompt"])
                output = str(row["normalized_output"])
                response_ids = [int(value) for value in row["authoritative_generated_token_ids"]]
                destination = str(row["destination"])
                teacher_tokens = int(row["authoritative_teacher_tokens"])
                normalization = "none"
            expected.append(
                {
                    "format": "abi-phase4-exact-b50-baseline-membership/1",
                    "ir_record_id": _membership_id(artifact, native),
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
                    "sequence_normalization": normalization,
                    "channel": "teacher_generated_sequence",
                    "split": "acquisition",
                }
            )
    return sorted(
        expected,
        key=lambda row: (
            ARTIFACT_ORDER.index(str(row["source_artifact"])),
            str(row["capability"]),
            str(row["native_record_id"]),
        ),
    )


def expected_accounting(
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    journal_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    journal = {str(row["attempt_sha256"]): row for row in journal_rows}
    union: dict[str, dict[str, Any]] = {}
    memberships = membership_tokens = 0
    for artifact in ARTIFACT_ORDER:
        for row in selected[artifact]:
            memberships += 1
            attempt = str(row["source_attempt_sha256"])
            if artifact == "v480_host_supervision":
                source = journal.get(attempt)
                if source is None:
                    raise Phase3Error("independent host journal join failed")
                token_count = int(row["source_teacher_output_tokens"])
                if (
                    token_count != int(source["teacher_tokens"])
                    or [int(value) for value in row["source_authoritative_generated_token_ids"]]
                    != [int(value) for value in source["authoritative_generated_token_ids"]]
                    or str(row["source_output_sha256"]) != str(source["output_sha256"])
                    or str(row["source_generation_prompt_sha256"])
                    != str(source["generation_prompt_sha256"])
                ):
                    raise Phase3Error("independent host provenance check failed")
                input_tokens = int(source["teacher_input_tokens"])
                prompt = str(source["generation_prompt"])
                output = str(source["output"])
                output_sha = str(source["output_sha256"])
            else:
                token_count = int(row["authoritative_teacher_tokens"])
                input_tokens = int(row["teacher_input_tokens"])
                prompt = str(row["raw_generation_prompt"])
                output = str(row["raw_output"])
                output_sha = str(row["raw_output_sha256"])
            membership_tokens += token_count
            facts = {
                "teacher_input_tokens": input_tokens,
                "teacher_output_tokens": token_count,
                "raw_prompt_bytes": len(prompt.encode("utf-8")),
                "raw_teacher_output_bytes": len(output.encode("utf-8")),
                "teacher_output_sha256": output_sha,
            }
            if attempt in union and union[attempt] != facts:
                raise Phase3Error("independent duplicate-attempt check failed")
            union[attempt] = facts
    return {
        "record_memberships": memberships,
        "unique_source_attempts": len(union),
        "duplicate_memberships": memberships - len(union),
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


def _verify_payload(
    *,
    entries: Mapping[str, bytes],
    observed_pack: Mapping[str, Any],
    expected_records: Sequence[Mapping[str, Any]],
    expected_information: Mapping[str, Any],
    tokenizer: Any,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = json.loads(entries["manifest.json"])
    accounting = json.loads(entries["accounting.json"])
    records = [json.loads(line) for line in entries["records.jsonl"].splitlines()]
    if hashlib.sha256(entries["records.jsonl"]).hexdigest() != manifest["records_jsonl_sha256"]:
        raise Phase3Error("records manifest hash mismatch")
    if hashlib.sha256(entries["accounting.json"]).hexdigest() != manifest["accounting_sha256"]:
        raise Phase3Error("accounting manifest hash mismatch")
    if records != list(expected_records):
        raise Phase3Error("records differ from independent B50 reconstruction")
    if accounting != dict(expected_information):
        raise Phase3Error("accounting differs from independent source-attempt reconstruction")
    packs = pack_examples(
        tokenize_records(records, tokenizer),
        max_tokens=int(protocol["packing_context"]),
        seed=int(protocol["packing_seed"]),
    )
    rebuilt = pack_manifest(packs)
    for key in (
        "packs",
        "pack_count",
        "record_count",
        "input_tokens",
        "response_tokens",
        "content_sha256",
    ):
        if observed_pack[key] != rebuilt[key]:
            raise Phase3Error(f"pack reconstruction mismatch: {key}")
    return {
        "records": len(records),
        "packs": rebuilt["pack_count"],
        "response_tokens": rebuilt["response_tokens"],
        "content_sha256": rebuilt["content_sha256"],
    }


def _reject(callable_object: Any) -> bool:
    try:
        callable_object()
    except (Phase3Error, KeyError, ValueError, TypeError):
        return True
    return False


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable independent B50 pack verifier result exists: {output}")
    result_under_test = _json(root / protocol["result_under_test"])
    evidence_copy = dict(result_under_test)
    observed_evidence = str(evidence_copy.pop("evidence_sha256"))
    if hashlib.sha256(canonical_json_bytes(evidence_copy)).hexdigest() != observed_evidence:
        raise Phase3Error("B50 pack result evidence hash changed")
    lineage = _json(root / protocol["lineage_protocol"])
    budget_manifest = _json(root / protocol["budget_manifest"])
    source_specs = {str(row["id"]): row for row in lineage["teacher_artifacts"]}
    source_paths = {
        key: root / str(row["path"]) for key, row in source_specs.items()
    }
    source_rows = {key: _rows(path) for key, path in source_paths.items()}
    if any(
        len(source_rows[key]) != int(source_specs[key]["records"])
        for key in source_specs
    ):
        raise Phase3Error("independent source logical record count changed")
    budget = next(row for row in lineage["budgets"] if row["id"] == "B50")
    selected = independent_selection(
        source_rows,
        salt=str(budget_manifest["selection_salt"]),
        weak_capabilities=set(protocol["weak_capabilities"]),
        budget=budget,
    )
    selected_ids = {
        artifact: sorted(
            str(row.get("ir_record_id", row.get("record_id"))) for row in rows
        )
        for artifact, rows in selected.items()
    }
    selection_sha = hashlib.sha256(canonical_json_bytes(selected_ids)).hexdigest()
    if selection_sha != protocol["expected"]["selection_sha256"]:
        raise Phase3Error("independent B50 selection hash mismatch")
    tokenizer = _tokenizer(_verified_snapshot(root))
    expected_records = expected_memberships(selected, tokenizer)
    journal_rows = [
        json.loads(line)
        for line in (root / protocol["source_attempt_journal"]).read_bytes().splitlines()
    ]
    information = expected_accounting(selected, journal_rows)
    entries = _archive_entries(root / protocol["records_archive"])
    observed_pack = _json(root / protocol["pack_manifest"])
    verified = _verify_payload(
        entries=entries,
        observed_pack=observed_pack,
        expected_records=expected_records,
        expected_information=information,
        tokenizer=tokenizer,
        protocol=protocol,
    )
    mutated_records = dict(entries)
    changed = [json.loads(line) for line in entries["records.jsonl"].splitlines()]
    changed[0]["source_attempt_sha256"] = "0" * 64
    mutated_records["records.jsonl"] = b"".join(canonical_json_bytes(row) for row in changed)
    mutated_accounting = dict(entries)
    changed_accounting = dict(json.loads(entries["accounting.json"]))
    changed_accounting["authoritative_teacher_output_tokens"] += 1
    mutated_accounting["accounting.json"] = canonical_json_bytes(changed_accounting)
    mutated_pack = dict(observed_pack)
    mutated_pack["response_tokens"] += 1
    attacks = {
        "record_mutation_rejected": _reject(
            lambda: _verify_payload(
                entries=mutated_records,
                observed_pack=observed_pack,
                expected_records=expected_records,
                expected_information=information,
                tokenizer=tokenizer,
                protocol=protocol,
            )
        ),
        "accounting_mutation_rejected": _reject(
            lambda: _verify_payload(
                entries=mutated_accounting,
                observed_pack=observed_pack,
                expected_records=expected_records,
                expected_information=information,
                tokenizer=tokenizer,
                protocol=protocol,
            )
        ),
        "pack_mutation_rejected": _reject(
            lambda: _verify_payload(
                entries=entries,
                observed_pack=mutated_pack,
                expected_records=expected_records,
                expected_information=information,
                tokenizer=tokenizer,
                protocol=protocol,
            )
        ),
    }
    gates = {
        "producer_result_passed": result_under_test["status"]
        == "PASS_EXACT_B50_BASELINE_SEQUENCE_PACK_READY",
        "selection_independently_reconstructed": selection_sha
        == protocol["expected"]["selection_sha256"],
        "all_records_independently_reconstructed": verified["records"] == 5140,
        "surface_normalization_count_reconstructed": sum(
            row["sequence_normalization"]
            == "retokenized_after_closed_v479_surface_normalization"
            for row in expected_records
        )
        == int(protocol["expected"]["surface_normalized_host_memberships"]),
        "source_attempt_accounting_independently_reconstructed": information[
            "unique_source_attempts"
        ]
        == 4953
        and information["authoritative_teacher_output_tokens"] == 152266,
        "pack_independently_reconstructed": verified["content_sha256"]
        == observed_pack["content_sha256"]
        == protocol["expected"]["pack_content_sha256"]
        and verified["packs"] == int(protocol["expected"]["pack_count"])
        and verified["response_tokens"]
        == int(protocol["expected"]["response_tokens"]),
        "producer_result_artifact_links_exact": result_under_test["records_archive"][
            "sha256"
        ]
        == protocol["bindings"][protocol["records_archive"]]
        and result_under_test["sequence_pack"]["sha256"]
        == protocol["bindings"][protocol["pack_manifest"]],
        "all_attacks_rejected": all(attacks.values()),
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_model_loading_absent": True,
        "final_test_absent": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-b50-baseline-pack-verify-result/1",
        "status": "PASS_INDEPENDENT_EXACT_B50_BASELINE_PACK_VERIFICATION"
        if all(gates.values())
        else "FAIL_INDEPENDENT_EXACT_B50_BASELINE_PACK_VERIFICATION",
        "protocol_sha256": protocol_sha,
        "result_under_test_sha256": sha256_file(root / protocol["result_under_test"]),
        "selection_sha256": selection_sha,
        "verified": verified,
        "imported_information": information,
        "attacks": attacks,
        "gates": gates,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": (
            "Independent exact-B50 sequence-pack verification only. No baseline training, "
            "matched frontier, minimum, final-test, Phase 4, or ABI-superiority result."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
