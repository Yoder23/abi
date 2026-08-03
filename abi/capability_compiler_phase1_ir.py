"""Build and verify the immutable normalized Phase 1 acquisition IR."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import time
import unicodedata
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from .capability_compiler_phase1_extract import (
    _canonical_sha,
    _sha256_file,
    load_journal,
)
from .capability_pipeline import canonical_json_bytes
from .hf_extraction import load_probe_catalog


IR_FORMAT = "abi-normalized-acquisition-ir/1"
IR_RECORD_FORMAT = "abi-normalized-acquisition-record/1"
REQUIRED_MEMBERS = {
    "manifest.json",
    "source_identity.json",
    "normalization.json",
    "records.jsonl",
    "inventory.json",
    "split_manifest.json",
    "domain_reference.jsonl",
    "rejections.jsonl",
    "accounting.json",
    "ledger.json",
}
CANONICAL_CAPABILITIES = (
    "grammar",
    "coherence",
    "prompt_grounding",
    "instruction_following",
    "conversation",
    "supplied_text_summarization",
    "rewriting",
    "email_drafting_from_notes",
    "tone_control",
    "format_control",
    "clarification",
    "abstention",
    "fact_free_reasoning",
    "fluent_realization",
)
DOMAINS = ("chemistry", "civics", "mathematics", "python")


class Phase1IRError(RuntimeError):
    pass


def normalize_text(value: str) -> tuple[str, list[str]]:
    transformations: list[str] = []
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value:
        transformations.append("unicode_nfc")
    line_normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if line_normalized != normalized:
        transformations.append("line_endings_lf")
    stripped_lines = "\n".join(line.rstrip(" \t") for line in line_normalized.split("\n"))
    if stripped_lines != line_normalized:
        transformations.append("strip_trailing_horizontal_whitespace_per_line")
    outer = stripped_lines.strip("\n")
    if outer != stripped_lines:
        transformations.append("strip_outer_blank_lines")
    return outer, transformations


def simhash64(value: str) -> int:
    words = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFC", value).lower())
    shingles = (
        [" ".join(words[index : index + 5]) for index in range(len(words) - 4)]
        if len(words) >= 5
        else [" ".join(words)]
    )
    accumulators = [0] * 64
    for shingle in shingles:
        hashed = int.from_bytes(hashlib.sha256(shingle.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            accumulators[bit] += 1 if (hashed >> bit) & 1 else -1
    return sum(1 << bit for bit, value in enumerate(accumulators) if value >= 0)


def cross_split_near_duplicates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    collisions: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        value = simhash64(str(row["prompt"]))
        candidates: set[tuple[int, int]] = set()
        for band in range(4):
            candidates.update(index[(band, (value >> (band * 16)) & 0xFFFF)])
        for prior_index, prior_hash in candidates:
            prior = rows[prior_index]
            distance = (value ^ prior_hash).bit_count()
            if prior["split"] != row["split"] and distance <= 3:
                collisions.append(
                    {
                        "left_probe_id": prior["probe_id"],
                        "left_split": prior["split"],
                        "right_probe_id": row["probe_id"],
                        "right_split": row["split"],
                        "hamming_distance": distance,
                    }
                )
        for band in range(4):
            index[(band, (value >> (band * 16)) & 0xFFFF)].append((row_index, value))
    return collisions


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def _tokenizer(protocol: Mapping[str, Any]):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    source = protocol["source"]
    snapshot = snapshot_download(
        repo_id=source["model"],
        revision=source["revision"],
        local_files_only=True,
    )
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


def _candidate_rows(
    *,
    root: Path,
    v1_protocol: Mapping[str, Any],
    v2_protocol: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    v1_protocol_sha = _sha256_file(root / "ABI_CAPABILITY_COMPILER_PHASE1_PROTOCOL_V1.json")
    v2_protocol_sha = _sha256_file(root / "ABI_CAPABILITY_COMPILER_PHASE1_ABSTENTION_PROTOCOL_V2.json")
    v1_catalog_sha = v1_protocol["catalog"]["sha256"]
    v2_catalog_sha = v2_protocol["fresh_catalog"]["sha256"]
    v1_catalog = load_probe_catalog(root / v1_protocol["catalog"]["path"])
    v2_catalog = load_probe_catalog(root / v2_protocol["fresh_catalog"]["path"])
    probes = {
        (v1_protocol_sha, row["probe_id"]): row
        for row in v1_catalog["probes"]
        if row["split"] == "search"
    }
    probes.update({(v2_protocol_sha, row["probe_id"]): row for row in v2_catalog["probes"]})
    journals = (
        (
            v1_protocol_sha,
            load_journal(
                root / "results/abi_capability_compiler_phase1/v1/source_search_attempts.jsonl",
                protocol_sha256=v1_protocol_sha,
                catalog_sha256=v1_catalog_sha,
            ),
        ),
        (
            v2_protocol_sha,
            load_journal(
                root / "results/abi_capability_compiler_phase1/v2/abstention_source_attempts.jsonl",
                protocol_sha256=v2_protocol_sha,
                catalog_sha256=v2_catalog_sha,
            ),
        ),
    )
    first_pass: dict[tuple[str, str], Mapping[str, Any]] = {}
    all_attempts: list[tuple[str, Mapping[str, Any]]] = []
    for protocol_sha, journal in journals:
        for (_, _), attempt in journal.items():
            all_attempts.append((protocol_sha, attempt))
            key = (protocol_sha, attempt["probe_id"])
            if attempt["functional_pass"] and (
                key not in first_pass
                or attempt["attempt_index"] < first_pass[key]["attempt_index"]
            ):
                first_pass[key] = attempt
    candidates: list[dict[str, Any]] = []
    for (protocol_sha, probe_id), attempt in first_pass.items():
        probe = probes[(protocol_sha, probe_id)]
        candidates.append(
            {
                "protocol_sha256": protocol_sha,
                "probe": probe,
                "attempt": attempt,
                "selection_key": hashlib.sha256(
                    f"{protocol_sha}:{probe_id}:{attempt['attempt_sha256']}".encode("ascii")
                ).hexdigest(),
            }
        )
    selected: list[dict[str, Any]] = []
    for capability in CANONICAL_CAPABILITIES:
        rows = sorted(
            (row for row in candidates if row["probe"]["canonical_capability"] == capability),
            key=lambda row: row["selection_key"],
        )
        if len(rows) < 500:
            raise Phase1IRError(f"insufficient eligible records: {capability}={len(rows)}")
        selected.extend(rows[:500])
    selected_attempts = {
        (row["protocol_sha256"], row["attempt"]["attempt_sha256"]) for row in selected
    }
    rejections: list[dict[str, Any]] = []
    for protocol_sha, attempt in all_attempts:
        key = (protocol_sha, attempt["attempt_sha256"])
        if key in selected_attempts:
            continue
        reason = (
            "functional_or_finish_failure"
            if not attempt["functional_pass"]
            else "fixed_depth_or_first_pass_selection_not_selected"
        )
        rejections.append(
            {
                "source_protocol_sha256": protocol_sha,
                "probe_id": attempt["probe_id"],
                "attempt_sha256": attempt["attempt_sha256"],
                "canonical_capability": attempt["canonical_capability"],
                "reason": reason,
                "v1_failure_reclassified": False,
            }
        )
    return selected, rejections, {
        "v1_protocol_sha256": v1_protocol_sha,
        "v2_protocol_sha256": v2_protocol_sha,
        "candidate_passes": Counter(row["probe"]["canonical_capability"] for row in candidates),
        "attempts": len(all_attempts),
    }


def build_ir(*, root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise Phase1IRError(f"IR is immutable: {output_path}")
    started = time.perf_counter()
    v1_protocol = json.loads((root / "ABI_CAPABILITY_COMPILER_PHASE1_PROTOCOL_V1.json").read_text(encoding="utf-8"))
    v2_protocol = json.loads((root / "ABI_CAPABILITY_COMPILER_PHASE1_ABSTENTION_PROTOCOL_V2.json").read_text(encoding="utf-8"))
    selected, rejections, selection_meta = _candidate_rows(root=root, v1_protocol=v1_protocol, v2_protocol=v2_protocol)
    tokenizer = _tokenizer(v1_protocol)
    records: list[dict[str, Any]] = []
    for candidate in selected:
        probe = candidate["probe"]
        attempt = candidate["attempt"]
        acquisition_prompt, acquisition_transformations = normalize_text(str(probe["prompt"]))
        generation_prompt, generation_transformations = normalize_text(str(attempt["generation_prompt"]))
        output, output_transformations = normalize_text(str(attempt["output"]))
        if not output:
            raise Phase1IRError("selected record normalized to an empty output")
        source_record_id = hashlib.sha256(
            f"{candidate['protocol_sha256']}:{attempt['attempt_sha256']}".encode("ascii")
        ).hexdigest()
        record: dict[str, Any] = {
            "format": IR_RECORD_FORMAT,
            "source_record_id": source_record_id,
            "source_protocol_sha256": candidate["protocol_sha256"],
            "source_catalog_sha256": attempt["catalog_sha256"],
            "source_attempt_sha256": attempt["attempt_sha256"],
            "source_model": v1_protocol["source"]["model"],
            "source_revision": v1_protocol["source"]["revision"],
            "source_license": v1_protocol["source"]["license"],
            "probe_id": probe["probe_id"],
            "split": "acquisition",
            "destination": "english_core",
            "capability": probe["canonical_capability"],
            "domain": "domain_independent",
            "knowledge_class": "english_linguistic_form",
            "content_basis": probe["content_basis"],
            "domain_labels": [],
            "domain_claims": [],
            "label_method": "preregistered_catalog",
            "label_confidence": 1.0,
            "label_evidence_sha256": probe["label_evidence_sha256"],
            "template_family": probe["phase1_template_family"],
            "selection_key": candidate["selection_key"],
            "raw_acquisition_prompt": probe["prompt"],
            "raw_acquisition_prompt_sha256": hashlib.sha256(probe["prompt"].encode("utf-8")).hexdigest(),
            "normalized_acquisition_prompt": acquisition_prompt,
            "normalized_acquisition_prompt_sha256": hashlib.sha256(acquisition_prompt.encode("utf-8")).hexdigest(),
            "normalized_acquisition_prompt_token_ids": tokenizer.encode(acquisition_prompt, add_special_tokens=False),
            "raw_generation_prompt": attempt["generation_prompt"],
            "raw_generation_prompt_sha256": attempt["generation_prompt_sha256"],
            "normalized_generation_prompt": generation_prompt,
            "normalized_generation_prompt_sha256": hashlib.sha256(generation_prompt.encode("utf-8")).hexdigest(),
            "rendered_generation_prompt": attempt["rendered_prompt"],
            "rendered_generation_prompt_sha256": attempt["rendered_prompt_sha256"],
            "teacher_input_tokens": attempt["teacher_input_tokens"],
            "raw_output": attempt["output"],
            "raw_output_sha256": attempt["output_sha256"],
            "normalized_output": output,
            "normalized_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "normalized_output_token_ids": tokenizer.encode(output, add_special_tokens=False),
            "authoritative_generated_token_ids": attempt["authoritative_generated_token_ids"],
            "authoritative_teacher_tokens": attempt["teacher_tokens"],
            "authoritative_token_counter": attempt["teacher_token_counter"],
            "finish_reason": attempt["finish_reason"],
            "generation_max_new_tokens": attempt["generation_max_new_tokens"],
            "functional_evaluator": attempt["functional_evaluator"],
            "functional_score": attempt["functional_score"],
            "functional_pass": attempt["functional_pass"],
            "attempt_kind": attempt["kind"],
            "normalization": {
                "normalization_id": "abi-phase1-text-normalization-v1",
                "acquisition_prompt_transformations": acquisition_transformations,
                "generation_prompt_transformations": generation_transformations,
                "output_transformations": output_transformations,
                "semantic_rewrite": False,
            },
        }
        record["ir_record_id"] = _canonical_sha(record)
        records.append(record)
    records.sort(key=lambda row: row["ir_record_id"])

    full_catalog = json.loads((root / v1_protocol["catalog"]["path"]).read_text(encoding="utf-8"))
    near_duplicates = cross_split_near_duplicates(full_catalog["probes"])
    prompt_hashes: dict[str, list[str]] = defaultdict(list)
    template_families: dict[str, set[str]] = defaultdict(set)
    for row in full_catalog["probes"]:
        prompt_hashes[row["split"]].append(hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest())
        template_families[row["split"]].add(row["phase1_template_family"])
    split_manifest = {
        "format": "abi-phase1-split-manifest/1",
        "evaluation_catalog_sha256": v1_protocol["catalog"]["sha256"],
        "counts": Counter(row["split"] for row in full_catalog["probes"]),
        "prompt_set_sha256": {
            split: hashlib.sha256("\n".join(sorted(values)).encode("ascii")).hexdigest()
            for split, values in prompt_hashes.items()
        },
        "template_family_set_sha256": {
            split: hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()
            for split, values in template_families.items()
        },
        "cross_split_exact_prompt_overlap": sum(
            len(set(prompt_hashes[left]) & set(prompt_hashes[right]))
            for left, right in itertools.combinations(prompt_hashes, 2)
        ),
        "near_duplicate_algorithm": "unicode_nfc_lower_word_5gram_simhash64",
        "near_duplicate_hamming_distance_maximum": 3,
        "cross_split_near_duplicate_clusters": near_duplicates,
        "domain_isolation_counts": Counter(row["domain"] for row in full_catalog["domain_isolation_probes"]),
        "adversarial_counts": Counter(row["family"] for row in full_catalog["adversarial_probes"]),
        "validation_teacher_outputs_generated": False,
        "final_teacher_outputs_generated": False,
        "final_candidate_outputs_generated": False,
        "final_used_for_normalization_selection_or_repairs": False,
    }

    domain_bundle_path = root / v1_protocol["declared_domain_reference"]["source_bundle"]["path"]
    with zipfile.ZipFile(domain_bundle_path, "r") as archive:
        source_records = [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line.strip()]
        passing_ids = {row["record_id"] for row in json.loads(archive.read("probe_results.json")) if row["passed"]}
    domain_rows: list[dict[str, Any]] = []
    for source_row in source_records:
        if source_row["destination_scope"] != "domain_cake" or source_row["domain"] not in DOMAINS or source_row["record_id"] not in passing_ids:
            continue
        prompt, prompt_steps = normalize_text(source_row["prompt"])
        output, output_steps = normalize_text(source_row["output"])
        row = {
            "format": "abi-phase1-domain-reference/1",
            "role": "evaluation_only_not_acquisition",
            "training_eligible": False,
            "source_record_id": source_row["record_id"],
            "source_bundle_sha256": v1_protocol["declared_domain_reference"]["source_bundle"]["sha256"],
            "source_model": source_row["source_model"],
            "source_revision": source_row["source_model_revision"],
            "source_license": v1_protocol["source"]["license"],
            "destination": "domain_cake",
            "domain": source_row["domain"],
            "capability": source_row["capability"],
            "raw_prompt": source_row["prompt"],
            "normalized_prompt": prompt,
            "normalized_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_output": source_row["output"],
            "normalized_output": output,
            "normalized_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "posthoc_output_token_ids": tokenizer.encode(output, add_special_tokens=False),
            "authoritative_generated_token_ids_available": False,
            "exclusion_reason": "historical_validation_reference_lacks_authoritative_ids_and_is_not_acquisition_eligible",
            "normalization": {"prompt_transformations": prompt_steps, "output_transformations": output_steps, "semantic_rewrite": False},
        }
        row["reference_record_sha256"] = _canonical_sha(row)
        domain_rows.append(row)
    domain_rows.sort(key=lambda row: row["reference_record_sha256"])
    domain_counts = Counter(row["domain"] for row in domain_rows)
    if domain_counts != {domain: 100 for domain in DOMAINS}:
        raise Phase1IRError(f"domain reference depth changed: {domain_counts}")

    inventory_rows = []
    for capability in CANONICAL_CAPABILITIES:
        rows = [row for row in records if row["capability"] == capability]
        inventory_rows.append(
            {
                "destination": "english_core",
                "capability": capability,
                "selected_records": len(rows),
                "distinct_source_records": len({row["source_record_id"] for row in rows}),
                "distinct_normalized_prompt_output_pairs": len({(row["normalized_acquisition_prompt_sha256"], row["normalized_output_sha256"]) for row in rows}),
                "authoritative_teacher_tokens": sum(row["authoritative_teacher_tokens"] for row in rows),
                "raw_prompt_bytes": sum(len(row["raw_acquisition_prompt"].encode("utf-8")) for row in rows),
                "raw_output_bytes": sum(len(row["raw_output"].encode("utf-8")) for row in rows),
                "template_families": sorted({row["template_family"] for row in rows}),
                "initial_attempt_records": sum(row["attempt_kind"] in {"initial", "fresh_v2_initial"} for row in rows),
                "repair_attempt_records": sum(row["attempt_kind"] == "same_teacher_closed_evaluator_repair" for row in rows),
            }
        )
    inventory = {
        "format": "abi-phase1-capability-inventory/1",
        "english": inventory_rows,
        "domains": [
            {"domain": domain, "reference_records": domain_counts[domain], "acquisition_records": 0, "role": "evaluation_only_not_acquisition"}
            for domain in DOMAINS
        ],
        "quarantine_destinations": v1_protocol["capability_ontology"]["fail_closed_destinations"] if "capability_ontology" in v1_protocol else json.loads((root / v1_protocol["phase0_protocol"]["path"]).read_text(encoding="utf-8"))["capability_ontology"]["fail_closed_destinations"],
    }
    selected_prompt_bytes = [row["normalized_acquisition_prompt"].encode("utf-8") for row in records]
    selected_output_bytes = [row["normalized_output"].encode("utf-8") for row in records]
    accounting = {
        "format": "abi-phase1-information-accounting/1",
        "selected_records": len(records),
        "selected_raw_prompt_bytes": sum(len(row["raw_acquisition_prompt"].encode("utf-8")) for row in records),
        "selected_unique_normalized_prompt_utf8_bytes": sum(len(value) for value in {value for value in selected_prompt_bytes}),
        "selected_raw_teacher_output_bytes": sum(len(row["raw_output"].encode("utf-8")) for row in records),
        "selected_normalized_teacher_output_bytes": sum(len(value) for value in selected_output_bytes),
        "selected_unique_normalized_output_utf8_bytes": sum(len(value) for value in {value for value in selected_output_bytes}),
        "selected_authoritative_teacher_tokens": sum(row["authoritative_teacher_tokens"] for row in records),
        "selected_teacher_input_tokens": sum(row["teacher_input_tokens"] for row in records),
        "selected_stored_authoritative_token_ids": sum(len(row["authoritative_generated_token_ids"]) for row in records),
        "selected_posthoc_normalized_output_token_ids": sum(len(row["normalized_output_token_ids"]) for row in records),
        "source_attempts_considered": selection_meta["attempts"],
        "rejected_or_unselected_attempts": len(rejections),
        "domain_reference_records": len(domain_rows),
        "stored_logits": 0,
        "stored_activations": 0,
        "copied_source_parameters": 0,
        "copied_source_parameter_bytes": 0,
        "bridge_or_student_parameters": 0,
        "source_parameter_count_read": v1_protocol["source"]["parameter_count"],
        "source_weight_bytes_read": v1_protocol["source"]["weight_bytes"],
        "normalization_seconds": round(time.perf_counter() - started, 6),
        "artifact_disk_footprint": "recorded_in_external_build_receipt",
    }
    source_identity = {
        "format": "abi-phase1-source-identity/1",
        "source": v1_protocol["source"],
        "v1_protocol_sha256": selection_meta["v1_protocol_sha256"],
        "v2_protocol_sha256": selection_meta["v2_protocol_sha256"],
        "v1_journal_sha256": "41d556ad1ff9b8e0f875a0d76299bedfce2dc8de756d088618fb85fb0a7f8648",
        "v2_journal_sha256": "930ea01334d02d5fe1adef4643ce4e9e7000127ade2f3fa3e211ab0012174cd3",
        "teacher_required_at_deployment": False,
        "source_parameters_copied": 0,
    }
    normalization = {
        "format": "abi-phase1-normalization/1",
        "normalization_id": "abi-phase1-text-normalization-v1",
        "unicode": "NFC",
        "line_endings": "LF",
        "strip_trailing_horizontal_whitespace_per_line": True,
        "strip_outer_blank_lines": True,
        "semantic_rewrite_allowed": False,
        "raw_forms_retained": True,
        "idempotence_required": True,
    }
    ledger = {
        "format": "abi-phase1-ledger/1",
        "status": "NORMALIZED_ACQUISITION_IR_ONLY_NOT_A_MODEL_OR_PACKAGE",
        "historical_evidence_changed": False,
        "candidate_training_performed": False,
        "validation_or_final_outputs_generated": False,
        "final_data_influenced_normalization_or_selection": False,
        "domain_acquisition_performed": False,
        "claim_boundary": "This artifact certifies bounded data suitability only. It does not prove teacher-to-LayerCake transfer, fluency, ABI superiority, or exhaustive domain discovery.",
    }
    members = {
        "source_identity.json": _json_bytes(source_identity),
        "normalization.json": _json_bytes(normalization),
        "records.jsonl": _jsonl_bytes(records),
        "inventory.json": _json_bytes(inventory),
        "split_manifest.json": _json_bytes(split_manifest),
        "domain_reference.jsonl": _jsonl_bytes(domain_rows),
        "rejections.jsonl": _jsonl_bytes(sorted(rejections, key=lambda row: (row["source_protocol_sha256"], row["attempt_sha256"]))),
        "accounting.json": _json_bytes(accounting),
        "ledger.json": _json_bytes(ledger),
    }
    member_bindings = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in sorted(members.items())}
    manifest: dict[str, Any] = {
        "format": IR_FORMAT,
        "artifact_id": "abi-capability-compiler-phase1-normalized-ir-v1",
        "status": "PHASE1_DATA_ARTIFACT_PENDING_CERTIFICATE",
        "contains_teacher_material": True,
        "installable_as_layercake_package": False,
        "candidate_training_performed": False,
        "record_count": len(records),
        "domain_reference_count": len(domain_rows),
        "rejection_count": len(rejections),
        "members": member_bindings,
        "content_set_sha256": hashlib.sha256(canonical_json_bytes(member_bindings)).hexdigest(),
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    members["manifest.json"] = _json_bytes(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as archive:
        for name in sorted(members):
            archive.writestr(_zip_info(name), members[name])
    verification = verify_ir(output_path)
    return {
        **verification,
        "output": str(output_path),
        "normalization_seconds": accounting["normalization_seconds"],
    }


def verify_ir(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if set(names) != REQUIRED_MEMBERS or len(names) != len(set(names)):
                raise Phase1IRError("IR member set is incomplete, duplicated, or extended")
            if any(Path(name).is_absolute() or ".." in Path(name).parts or "\\" in name for name in names):
                raise Phase1IRError("IR contains unsafe member path")
            members = {name: archive.read(name) for name in names}
    except zipfile.BadZipFile as exc:
        raise Phase1IRError("invalid IR ZIP") from exc
    manifest = json.loads(members["manifest.json"])
    if manifest.get("format") != IR_FORMAT:
        raise Phase1IRError("unsupported IR format")
    claimed_manifest = manifest.pop("manifest_sha256", None)
    if claimed_manifest != _canonical_sha(manifest):
        raise Phase1IRError("manifest hash changed")
    manifest["manifest_sha256"] = claimed_manifest
    for name, binding in manifest["members"].items():
        if len(members[name]) != binding["bytes"] or hashlib.sha256(members[name]).hexdigest() != binding["sha256"]:
            raise Phase1IRError(f"member binding changed: {name}")
    if manifest["content_set_sha256"] != hashlib.sha256(canonical_json_bytes(manifest["members"])).hexdigest():
        raise Phase1IRError("content set hash changed")
    records = [json.loads(line) for line in members["records.jsonl"].splitlines() if line.strip()]
    if len(records) != 7_000 or manifest["record_count"] != 7_000:
        raise Phase1IRError("IR record depth changed")
    counts = Counter(row["capability"] for row in records)
    if counts != {capability: 500 for capability in CANONICAL_CAPABILITIES}:
        raise Phase1IRError(f"capability depth changed: {counts}")
    source_ids: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for row in records:
        claimed = row.pop("ir_record_id", None)
        if claimed != _canonical_sha(row):
            raise Phase1IRError("IR record hash changed")
        row["ir_record_id"] = claimed
        if row["source_record_id"] in source_ids:
            raise Phase1IRError("duplicate source record ID")
        source_ids.add(row["source_record_id"])
        pair = (row["normalized_acquisition_prompt_sha256"], row["normalized_output_sha256"])
        if pair in pairs:
            raise Phase1IRError("duplicate normalized prompt/output pair")
        pairs.add(pair)
        if row["destination"] != "english_core" or row["domain"] != "domain_independent" or row["domain_labels"] or row["domain_claims"]:
            raise Phase1IRError("specialist data leaked into English IR")
        if row["finish_reason"] != "eos_token" or row["functional_pass"] is not True:
            raise Phase1IRError("ineligible source attempt selected")
        if len(row["authoritative_generated_token_ids"]) != row["authoritative_teacher_tokens"]:
            raise Phase1IRError("authoritative token accounting changed")
        normalized_prompt, _ = normalize_text(row["normalized_acquisition_prompt"])
        normalized_output, _ = normalize_text(row["normalized_output"])
        if normalized_prompt != row["normalized_acquisition_prompt"] or normalized_output != row["normalized_output"]:
            raise Phase1IRError("normalization is not idempotent")
        if hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest() != row["normalized_acquisition_prompt_sha256"] or hashlib.sha256(normalized_output.encode("utf-8")).hexdigest() != row["normalized_output_sha256"]:
            raise Phase1IRError("normalized text hash changed")
    split = json.loads(members["split_manifest.json"])
    if dict(split["counts"]) != {"final_test": 1400, "search": 9800, "validation": 1400}:
        raise Phase1IRError("evaluation split depth changed")
    if split["cross_split_exact_prompt_overlap"] != 0 or split["cross_split_near_duplicate_clusters"]:
        raise Phase1IRError("cross-split duplicate contamination detected")
    if split["final_used_for_normalization_selection_or_repairs"] is not False or split["final_teacher_outputs_generated"] is not False:
        raise Phase1IRError("final isolation changed")
    if dict(split["domain_isolation_counts"]) != {domain: 100 for domain in DOMAINS}:
        raise Phase1IRError("domain isolation depth changed")
    domain_rows = [json.loads(line) for line in members["domain_reference.jsonl"].splitlines() if line.strip()]
    if Counter(row["domain"] for row in domain_rows) != {domain: 100 for domain in DOMAINS}:
        raise Phase1IRError("domain reference depth changed")
    if any(row["training_eligible"] is not False or row["destination"] != "domain_cake" for row in domain_rows):
        raise Phase1IRError("domain reference became acquisition eligible or misrouted")
    accounting = json.loads(members["accounting.json"])
    if accounting["selected_records"] != len(records) or accounting["selected_authoritative_teacher_tokens"] != sum(row["authoritative_teacher_tokens"] for row in records):
        raise Phase1IRError("IR accounting changed")
    ledger = json.loads(members["ledger.json"])
    if ledger["candidate_training_performed"] is not False or ledger["final_data_influenced_normalization_or_selection"] is not False:
        raise Phase1IRError("Phase 1 boundary changed")
    return {
        "status": "PASS",
        "archive_sha256": _sha256_file(path),
        "archive_bytes": path.stat().st_size,
        "manifest_sha256": claimed_manifest,
        "record_count": len(records),
        "records_per_capability": 500,
        "domain_reference_records": len(domain_rows),
        "cross_split_near_duplicates": 0,
        "candidate_training_performed": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--root", default=".")
    build.add_argument("--output", default="results/abi_capability_compiler_phase1/final/normalized_acquisition_ir_v1.abicir")
    verify = sub.add_parser("verify")
    verify.add_argument("path")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    result = (
        build_ir(root=Path(args.root), output_path=Path(args.output))
        if args.command == "build"
        else verify_ir(Path(args.path))
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
