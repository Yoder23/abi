"""Normalize a fixed-depth broad-English teacher journal into immutable IR."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping
import zipfile

from .capability_compiler_phase1_extract import _canonical_sha, _sha256_file, load_journal
from .capability_compiler_phase1_ir import (
    CANONICAL_CAPABILITIES,
    IR_FORMAT,
    IR_RECORD_FORMAT,
    REQUIRED_MEMBERS,
    _json_bytes,
    _jsonl_bytes,
    _tokenizer as _teacher_tokenizer,
    _zip_info,
    normalize_text,
)
from .hf_extraction import load_probe_catalog
from .capability_compiler_phase3_native_causal_core import load_protocol as load_candidate_protocol
from .capability_compiler_phase3_teacher_native_core import (
    _layercake_api,
    _tokenizer as _host_tokenizer,
    controlled_prompt,
)


PROTOCOL_FORMAT = "abi-capability-compiler-phase3-broad-ir/1"


class BroadIRError(RuntimeError):
    """Raised when selection, normalization, or artifact integrity fails."""


def host_prompt_projection(capability: str, prompt: str) -> tuple[str, str]:
    first, separator, remainder = prompt.partition("\n")
    targeted_prefixes = (
        "Complete this new bounded English practice task.",
        "Use only the material supplied in search item",
        "Follow the exact requested wording and format for item",
        "Respond directly to this independent English search task",
        "Work only from the text below. Search case",
        "Give only the requested answer for new exercise",
        "Read the supplied context, then answer search ticket",
        "Handle this language task without outside facts. Search ref",
    )
    if separator and first.startswith(targeted_prefixes):
        if not remainder.strip():
            raise BroadIRError("targeted prompt projection removed the complete task")
        return remainder.strip(), "phase1_task_body_without_targeted_search_wrapper"
    if capability != "fluent_realization":
        return prompt, "full_normalized_acquisition_prompt_host_bound_selected"
    marker = "</fictional_context>\n"
    if marker not in prompt:
        raise BroadIRError("fluent realization prompt lacks context boundary")
    projected = prompt.split(marker, 1)[1]
    prefix = "Use the context only to disambiguate the fields. "
    if not projected.startswith(prefix):
        raise BroadIRError("fluent realization projection prefix changed")
    projected = projected[len(prefix):]
    if "event_one=" not in projected or "event_two=" not in projected:
        raise BroadIRError("fluent realization event fields changed")
    return projected, "fluent_realization_event_fields_without_redundant_context"


def select_candidates(
    probes: Iterable[Mapping[str, Any]],
    journal: Mapping[tuple[str, int], Mapping[str, Any]],
    capability_mapping: Mapping[str, str],
    *,
    source_protocol_sha256: str,
    per_capability: int,
    capabilities: Iterable[str] = CANONICAL_CAPABILITIES,
    eligibility: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    probe_by_id = {str(row["probe_id"]): row for row in probes}
    first_pass: dict[str, Mapping[str, Any]] = {}
    for (probe_id, _), attempt in journal.items():
        if probe_id not in probe_by_id:
            raise BroadIRError("journal contains an unknown probe")
        if attempt.get("functional_pass") is True and (
            probe_id not in first_pass
            or int(attempt["attempt_index"]) < int(first_pass[probe_id]["attempt_index"])
        ):
            first_pass[probe_id] = attempt
    candidates: list[dict[str, Any]] = []
    for probe_id, attempt in first_pass.items():
        probe = probe_by_id[probe_id]
        capability = capability_mapping[str(probe["capability"])]
        if eligibility is not None and not eligibility(probe, attempt, capability):
            continue
        selection_key = hashlib.sha256(
            f"{source_protocol_sha256}:{probe_id}:{attempt['attempt_sha256']}".encode("ascii")
        ).hexdigest()
        candidates.append({
            "probe": probe,
            "attempt": attempt,
            "capability": capability,
            "selection_key": selection_key,
        })
    selected: list[dict[str, Any]] = []
    selected_capabilities = tuple(capabilities)
    for capability in selected_capabilities:
        rows = sorted(
            (row for row in candidates if row["capability"] == capability),
            key=lambda row: row["selection_key"],
        )
        if len(rows) < per_capability:
            raise BroadIRError(f"insufficient eligible records: {capability}={len(rows)}")
        selected.extend(rows[:per_capability])
    selected_attempts = {row["attempt"]["attempt_sha256"] for row in selected}
    rejections = []
    for attempt in journal.values():
        if attempt["attempt_sha256"] in selected_attempts:
            continue
        rejections.append({
            "source_protocol_sha256": source_protocol_sha256,
            "probe_id": attempt["probe_id"],
            "attempt_sha256": attempt["attempt_sha256"],
            "canonical_capability": attempt["canonical_capability"],
            "reason": (
                "functional_or_finish_failure"
                if not attempt["functional_pass"]
                else "fixed_depth_or_first_pass_selection_not_selected"
            ),
        })
    return selected, sorted(rejections, key=lambda row: row["attempt_sha256"])


def _verify_protocol(path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise BroadIRError("unsupported broad IR protocol")
    root = path.resolve().parent
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or _sha256_file(target) != expected:
            raise BroadIRError(f"binding mismatch: {relative}")
    return protocol, _sha256_file(path)


def build_ir(*, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    output_path = output_path.resolve()
    if output_path.exists():
        raise BroadIRError(f"IR is immutable: {output_path}")
    started = time.perf_counter()
    protocol, ir_protocol_sha = _verify_protocol(protocol_path)
    root = protocol_path.resolve().parent
    candidate_protocol, _ = load_candidate_protocol(
        root, (root / protocol["candidate_protocol"]).resolve()
    )
    _, host_tokenizer_type, _, _ = _layercake_api(root, candidate_protocol)
    host_tokenizer = _host_tokenizer(root, candidate_protocol, host_tokenizer_type)
    maximum_source = int(protocol["host_bounds"]["maximum_source_lexemes"])
    maximum_target = int(protocol["host_bounds"]["maximum_target_actions"])

    def host_eligible(
        probe: Mapping[str, Any], attempt: Mapping[str, Any], capability: str
    ) -> bool:
        projected, _ = host_prompt_projection(capability, str(probe["prompt"]))
        source_ids, _ = host_tokenizer.encode_source(controlled_prompt(capability, projected))
        target_actions = host_tokenizer.encode_fixed_target(str(attempt["output"]))
        return len(source_ids) <= maximum_source and len(target_actions) <= maximum_target
    selected: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    source_runtime: list[dict[str, Any]] = []
    for source_spec in protocol["sources"]:
        extraction_path = root / source_spec["extraction_protocol"]["path"]
        extraction_sha = _sha256_file(extraction_path)
        if extraction_sha != source_spec["extraction_protocol"]["sha256"]:
            raise BroadIRError("extraction protocol changed")
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        catalog = load_probe_catalog(root / extraction["catalog"]["path"])
        probes = [row for row in catalog["probes"] if row["split"] == "search"]
        journal_path = root / source_spec["journal"]["path"]
        if _sha256_file(journal_path) != source_spec["journal"]["sha256"]:
            raise BroadIRError("teacher journal changed")
        journal = load_journal(
            journal_path,
            protocol_sha256=extraction_sha,
            catalog_sha256=extraction["catalog"]["sha256"],
        )
        source_selected, source_rejections = select_candidates(
            probes,
            journal,
            extraction["capability_mapping"],
            source_protocol_sha256=extraction_sha,
            per_capability=int(protocol["selection"]["per_capability"]),
            capabilities=source_spec["capabilities"],
            eligibility=host_eligible,
        )
        for candidate in source_selected:
            candidate["extraction"] = extraction
            candidate["source_protocol_sha256"] = extraction_sha
        selected.extend(source_selected)
        rejections.extend(source_rejections)
        source_runtime.append({
            "extraction": extraction,
            "extraction_sha256": extraction_sha,
            "journal_sha256": source_spec["journal"]["sha256"],
            "attempts": len(journal),
        })
    if Counter(row["capability"] for row in selected) != {
        capability: int(protocol["selection"]["per_capability"])
        for capability in CANONICAL_CAPABILITIES
    }:
        raise BroadIRError("combined source selection is not balanced")
    tokenizer = _teacher_tokenizer(source_runtime[0]["extraction"])
    records: list[dict[str, Any]] = []
    for candidate in selected:
        probe, attempt = candidate["probe"], candidate["attempt"]
        extraction = candidate["extraction"]
        extraction_sha = candidate["source_protocol_sha256"]
        acquisition, acquisition_steps = normalize_text(str(probe["prompt"]))
        generation, generation_steps = normalize_text(str(attempt["generation_prompt"]))
        output, output_steps = normalize_text(str(attempt["output"]))
        host_prompt, host_projection = host_prompt_projection(
            candidate["capability"], acquisition
        )
        host_source_ids, _ = host_tokenizer.encode_source(
            controlled_prompt(candidate["capability"], host_prompt)
        )
        host_target_actions = host_tokenizer.encode_fixed_target(output)
        if len(host_source_ids) > maximum_source or len(host_target_actions) > maximum_target:
            raise BroadIRError("selected record crossed host bound after normalization")
        if not output:
            raise BroadIRError("selected output normalized empty")
        source_record_id = hashlib.sha256(
            f"{extraction_sha}:{attempt['attempt_sha256']}".encode("ascii")
        ).hexdigest()
        record: dict[str, Any] = {
            "format": IR_RECORD_FORMAT,
            "source_record_id": source_record_id,
            "source_protocol_sha256": extraction_sha,
            "source_catalog_sha256": extraction["catalog"]["sha256"],
            "source_attempt_sha256": attempt["attempt_sha256"],
            "source_model": extraction["source"]["model"],
            "source_revision": extraction["source"]["revision"],
            "source_license": extraction["source"]["license"],
            "probe_id": probe["probe_id"],
            "split": "acquisition",
            "destination": "english_core",
            "capability": candidate["capability"],
            "domain": "domain_independent",
            "knowledge_class": probe["knowledge_class"],
            "content_basis": probe["content_basis"],
            "domain_labels": [],
            "domain_claims": [],
            "label_method": probe["label_method"],
            "label_confidence": 1.0,
            "label_evidence_sha256": probe["label_evidence_sha256"],
            "template_family": str(probe.get("phase3_targeted_template_family", f"broad_corpus_grounded_v108:{probe['capability']}")),
            "raw_context_sha256": str(probe.get("raw_context_sha256", hashlib.sha256(str(probe["prompt"]).encode("utf-8")).hexdigest())),
            "source_prompt_projection": host_projection,
            "host_conformant_acquisition_prompt": host_prompt,
            "host_conformant_acquisition_prompt_sha256": hashlib.sha256(host_prompt.encode("utf-8")).hexdigest(),
            "host_source_lexemes": len(host_source_ids),
            "host_target_actions": len(host_target_actions),
            "selection_key": candidate["selection_key"],
            "raw_acquisition_prompt": probe["prompt"],
            "raw_acquisition_prompt_sha256": hashlib.sha256(probe["prompt"].encode("utf-8")).hexdigest(),
            "normalized_acquisition_prompt": acquisition,
            "normalized_acquisition_prompt_sha256": hashlib.sha256(acquisition.encode("utf-8")).hexdigest(),
            "normalized_acquisition_prompt_token_ids": tokenizer.encode(acquisition, add_special_tokens=False),
            "raw_generation_prompt": attempt["generation_prompt"],
            "raw_generation_prompt_sha256": attempt["generation_prompt_sha256"],
            "normalized_generation_prompt": generation,
            "normalized_generation_prompt_sha256": hashlib.sha256(generation.encode("utf-8")).hexdigest(),
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
                "acquisition_prompt_transformations": acquisition_steps,
                "generation_prompt_transformations": generation_steps,
                "output_transformations": output_steps,
                "semantic_rewrite": False,
            },
        }
        record["ir_record_id"] = _canonical_sha(record)
        records.append(record)
    records.sort(key=lambda row: row["ir_record_id"])
    pairs = {(row["normalized_acquisition_prompt_sha256"], row["normalized_output_sha256"]) for row in records}
    if len(pairs) != len(records):
        raise BroadIRError("duplicate normalized prompt/output pair selected")

    counts = Counter(row["capability"] for row in records)
    inventory = {
        "format": "abi-phase3-broad-capability-inventory/1",
        "english": [
            {
                "destination": "english_core",
                "capability": capability,
                "selected_records": counts[capability],
                "distinct_source_records": len({row["source_record_id"] for row in records if row["capability"] == capability}),
                "authoritative_teacher_tokens": sum(row["authoritative_teacher_tokens"] for row in records if row["capability"] == capability),
            }
            for capability in CANONICAL_CAPABILITIES
        ],
        "domains": [],
    }
    split_manifest = {
        "format": "abi-phase3-broad-split-manifest/1",
        "catalog_sha256": extraction["catalog"]["sha256"],
        "selected_search_records": len(records),
        "validation_teacher_outputs_generated": False,
        "final_teacher_outputs_generated": False,
        "final_used_for_selection": False,
    }
    accounting = {
        "format": "abi-phase3-broad-information-accounting/1",
        "selected_records": len(records),
        "source_attempts_considered": sum(row["attempts"] for row in source_runtime),
        "rejected_or_unselected_attempts": len(rejections),
        "selected_raw_prompt_bytes": sum(len(row["raw_acquisition_prompt"].encode("utf-8")) for row in records),
        "selected_raw_teacher_output_bytes": sum(len(row["raw_output"].encode("utf-8")) for row in records),
        "selected_authoritative_teacher_tokens": sum(row["authoritative_teacher_tokens"] for row in records),
        "selected_teacher_input_tokens": sum(row["teacher_input_tokens"] for row in records),
        "stored_logits": 0,
        "stored_activations": 0,
        "copied_source_parameters": 0,
        "bridge_or_student_parameters": 0,
        "normalization_seconds": round(time.perf_counter() - started, 6),
    }
    source_identity = {
        "format": "abi-phase3-broad-source-identity/1",
        "sources": [
            {
                "source": row["extraction"]["source"],
                "extraction_protocol_sha256": row["extraction_sha256"],
                "journal_sha256": row["journal_sha256"],
                "attempts": row["attempts"],
            }
            for row in source_runtime
        ],
        "ir_protocol_sha256": ir_protocol_sha,
        "teacher_required_at_deployment": False,
        "source_parameters_copied": 0,
    }
    normalization = {
        "format": "abi-phase1-normalization/1",
        "normalization_id": "abi-phase1-text-normalization-v1",
        "unicode": "NFC", "line_endings": "LF",
        "strip_trailing_horizontal_whitespace_per_line": True,
        "strip_outer_blank_lines": True,
        "semantic_rewrite_allowed": False, "raw_forms_retained": True,
        "idempotence_required": True,
    }
    ledger = {
        "format": "abi-phase3-broad-ledger/1",
        "status": "NORMALIZED_COVERAGE_EXPANSION_IR_NOT_A_MODEL_OR_PACKAGE",
        "historical_evidence_changed": False,
        "candidate_training_performed": False,
        "validation_or_final_outputs_generated": False,
        "claim_boundary": "Data artifact only; no LayerCake quality or Phase 3 pass is claimed.",
    }
    members = {
        "source_identity.json": _json_bytes(source_identity),
        "normalization.json": _json_bytes(normalization),
        "records.jsonl": _jsonl_bytes(records),
        "inventory.json": _json_bytes(inventory),
        "split_manifest.json": _json_bytes(split_manifest),
        "domain_reference.jsonl": b"",
        "rejections.jsonl": _jsonl_bytes(rejections),
        "accounting.json": _json_bytes(accounting),
        "ledger.json": _json_bytes(ledger),
    }
    member_bindings = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in sorted(members.items())}
    manifest: dict[str, Any] = {
        "format": IR_FORMAT,
        "artifact_id": "abi-capability-compiler-phase3-broad-expansion-ir-v1",
        "status": "PHASE3_COVERAGE_EXPANSION_DATA_PENDING_COMBINED_COVERAGE_AUDIT",
        "contains_teacher_material": True,
        "installable_as_layercake_package": False,
        "candidate_training_performed": False,
        "record_count": len(records),
        "domain_reference_count": 0,
        "rejection_count": len(rejections),
        "members": member_bindings,
        "content_set_sha256": hashlib.sha256(json.dumps(member_bindings, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    members["manifest.json"] = _json_bytes(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as archive:
        for name in sorted(members):
            archive.writestr(_zip_info(name), members[name])
    result = verify_broad_ir(output_path)
    result["output"] = str(output_path)
    return result


def verify_broad_ir(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if set(names) != REQUIRED_MEMBERS or len(names) != len(set(names)):
            raise BroadIRError("IR member set changed")
        members = {name: archive.read(name) for name in names}
    manifest = json.loads(members["manifest.json"])
    claimed = manifest.pop("manifest_sha256", None)
    if claimed != _canonical_sha(manifest):
        raise BroadIRError("manifest hash changed")
    for name, binding in manifest["members"].items():
        if len(members[name]) != binding["bytes"] or hashlib.sha256(members[name]).hexdigest() != binding["sha256"]:
            raise BroadIRError(f"member binding changed: {name}")
    rows = [json.loads(line) for line in members["records.jsonl"].splitlines() if line.strip()]
    if len(rows) != 7000 or Counter(row["capability"] for row in rows) != {name: 500 for name in CANONICAL_CAPABILITIES}:
        raise BroadIRError("record depth or balance changed")
    ids, sources, pairs = set(), set(), set()
    for row in rows:
        row_copy = dict(row)
        claimed_id = row_copy.pop("ir_record_id", None)
        if claimed_id != _canonical_sha(row_copy):
            raise BroadIRError("record hash changed")
        pair = (row["normalized_acquisition_prompt_sha256"], row["normalized_output_sha256"])
        if claimed_id in ids or row["source_record_id"] in sources or pair in pairs:
            raise BroadIRError("duplicate record, source, or normalized pair")
        ids.add(claimed_id); sources.add(row["source_record_id"]); pairs.add(pair)
        if not (
            row["destination"] == "english_core"
            and row["domain"] == "domain_independent"
            and row["domain_labels"] == []
            and row["domain_claims"] == []
            and row["functional_pass"] is True
            and row["finish_reason"] == "eos_token"
        ):
            raise BroadIRError("ineligible record crossed IR boundary")
    return {
        "status": "PASS",
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "record_count": len(rows),
        "counts": dict(sorted(Counter(row["capability"] for row in rows).items())),
        "manifest_sha256": claimed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = (
        verify_broad_ir(Path(args.output).resolve())
        if args.verify_only
        else build_ir(protocol_path=Path(args.protocol).resolve(), output_path=Path(args.output).resolve())
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
