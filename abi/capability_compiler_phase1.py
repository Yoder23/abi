"""Verify the preregistered ABI capability-compiler Phase 1 protocol."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase1_catalog import (
    ADVERSARIAL_FAMILIES,
    CAPABILITY_ALIASES,
    DOMAINS,
)
from .hf_extraction import load_probe_catalog


PROTOCOL_FORMAT = "abi-capability-compiler-phase1-protocol/1"


class Phase1ProtocolError(RuntimeError):
    """Raised when the frozen Phase 1 protocol is incomplete or stale."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase1ProtocolError(message)


def verify_protocol(protocol_path: str | Path) -> dict[str, Any]:
    path = Path(protocol_path).resolve()
    root = path.parent
    protocol = json.loads(path.read_text(encoding="utf-8"))
    _require(protocol.get("format") == PROTOCOL_FORMAT, "unsupported protocol format")
    _require(
        protocol.get("status") == "PREREGISTERED_BEFORE_TEACHER_EXTRACTION_OR_IR_CONSTRUCTION",
        "protocol status does not prevent retrospective registration",
    )
    for binding_name in ("phase0_protocol",):
        binding = protocol[binding_name]
        bound_path = root / binding["path"]
        _require(bound_path.is_file(), f"missing {binding_name}")
        _require(_sha256_file(bound_path) == binding["sha256"], f"stale {binding_name}")
    source = protocol["source"]
    _require(len(source["revision"]) == 40, "source revision must be immutable")
    _require(sum(row["bytes"] for row in source["weight_files"]) == source["weight_bytes"], "source weight byte accounting mismatch")
    _require(len(source["weight_files"]) > 0, "source weight files are not pinned")
    _require(len(source["tokenizer_files"]) >= 4, "tokenizer files are not pinned")
    for row in source["weight_files"] + source["tokenizer_files"]:
        _require(len(row["sha256"]) == 64 and row["bytes"] > 0, "invalid source file binding")
    generation = source["generation"]
    _require(generation["device"] == "cuda", "Phase 1 source extraction must use CUDA")
    _require(generation["precision"] == "float16", "source precision changed")
    _require(generation["sampling"] == "greedy" and generation["temperature"] == 0.0, "source generation must be deterministic")
    _require(generation["authoritative_generated_token_ids_required"] is True, "runtime token IDs are not mandatory")
    _require(generation["finish_reason_required"] == "eos_token", "completion gate changed")
    _require(generation["length_terminated_records_eligible"] is False, "length-terminated rows became eligible")

    catalog_binding = protocol["catalog"]
    catalog_path = root / catalog_binding["path"]
    _require(_sha256_file(catalog_path) == catalog_binding["sha256"], "catalog identity changed")
    generator = catalog_binding["generator"]
    _require(_sha256_file(root / generator["path"]) == generator["sha256"], "catalog generator identity changed")
    catalog = load_probe_catalog(catalog_path)
    raw_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    counts = Counter((row["canonical_capability"], row["split"]) for row in catalog["probes"])
    canonical_capabilities = tuple(CAPABILITY_ALIASES.values())
    for capability in canonical_capabilities:
        _require(counts[(capability, "search")] == 700, f"wrong search depth: {capability}")
        _require(counts[(capability, "validation")] == 100, f"wrong development depth: {capability}")
        _require(counts[(capability, "final_test")] == 100, f"wrong final depth: {capability}")
    _require(
        Counter(row["domain"] for row in raw_catalog["domain_isolation_probes"])
        == {domain: 100 for domain in DOMAINS},
        "domain-isolation depth changed",
    )
    _require(
        Counter(row["family"] for row in raw_catalog["adversarial_probes"])
        == {family: 100 for family in ADVERSARIAL_FAMILIES},
        "adversarial depth changed",
    )
    families: dict[str, set[str]] = defaultdict(set)
    prompts: dict[str, set[str]] = defaultdict(set)
    for row in catalog["probes"]:
        families[row["split"]].add(row["phase1_template_family"])
        prompt_hash = hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
        _require(prompt_hash not in prompts[row["split"]], "duplicate prompt within split")
        prompts[row["split"]].add(prompt_hash)
        _require(row["destination_scope"] == "english_core", "non-English acquisition row")
        _require(row["domain"] == "domain_independent", "specialist row in English catalog")
        _require(row["domain_labels"] == [] and row["domain_claims"] == [], "domain label leaked into English")
        _require(row["output_introduces_unsupplied_facts"] is False, "English row permits new facts")
    for left, right in (("search", "validation"), ("search", "final_test"), ("validation", "final_test")):
        _require(not prompts[left] & prompts[right], "exact prompt overlap across splits")
        _require(not families[left] & families[right], "template family overlap across splits")

    acceptance = protocol["record_acceptance"]
    _require(acceptance["minimum_selected_per_canonical_capability"] == 500, "acquisition floor changed")
    _require(acceptance["selected_per_canonical_capability"] == 500, "selection depth changed")
    _require(acceptance["functional_evaluator_must_pass"] is True, "functional gate disabled")
    _require(acceptance["finish_reason"] == "eos_token", "finish gate changed")
    repair = protocol["repair_policy"]
    _require(repair["maximum_rounds"] == 1, "repair schedule is not bounded")
    _require(repair["additional_teacher_expansion_without_new_protocol"] is False, "unbounded source expansion allowed")
    split = protocol["split_and_contamination"]
    _require(split["final_used_for_normalization_or_selection"] is False, "final data may influence Phase 1")
    _require(split["final_candidate_or_teacher_outputs_generated_in_phase1"] is False, "final output generation allowed")
    phase1_ir = protocol["phase1_ir"]
    _require(len(phase1_ir["required_members"]) == len(set(phase1_ir["required_members"])), "duplicate IR members")
    _require(phase1_ir["candidate_training_performed"] is False, "candidate training allowed")
    _require(protocol["declared_domain_reference"]["selected_domain_acquisition_records"] == 0, "domain reference became acquisition material")
    for binding in (
        protocol["declared_domain_reference"]["source_bundle"],
        protocol["declared_domain_reference"]["labeling_certificate"],
    ):
        _require(_sha256_file(root / binding["path"]) == binding["sha256"], "domain reference binding changed")
    return {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256_file(path),
        "catalog_sha256": catalog_binding["sha256"],
        "english_capabilities": len(canonical_capabilities),
        "search_prompts": sum(1 for row in catalog["probes"] if row["split"] == "search"),
        "development_prompts": sum(1 for row in catalog["probes"] if row["split"] == "validation"),
        "final_prompts": sum(1 for row in catalog["probes"] if row["split"] == "final_test"),
        "domain_isolation_prompts": len(raw_catalog["domain_isolation_probes"]),
        "adversarial_prompts": len(raw_catalog["adversarial_probes"]),
        "training_authorized": False,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", nargs="?", default="ABI_CAPABILITY_COMPILER_PHASE1_PROTOCOL_V1.json")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_protocol(args.protocol)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, Phase1ProtocolError) as exc:
        raise SystemExit(f"Phase 1 protocol verification failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
