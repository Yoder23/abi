"""Read-only attribution of the verified B40 seed155921 clarification misses."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_abi_lineage import _selected_rows
from .capability_compiler_phase4_v19_frontier_rescreen import _json, _rows
from .capability_compiler_phase4_v19_frontier_verify import _without


FORMAT = "abi-capability-compiler-phase4-v22-b40-clarification-attribution/1"
INQUIRY_MARKERS = ("what", "which", "when", "where", "how", "could you", "clarify")


def clarification_failure_taxonomy(output: str) -> dict[str, bool]:
    lower = output.lower()
    return {
        "missing_question_mark": "?" not in output,
        "missing_inquiry_marker": not any(marker in lower for marker in INQUIRY_MARKERS),
        "empty_output": not output.strip(),
    }


def _object_phrase(prompt: str) -> str:
    match = re.search(r"Make the ([^”\"]+) better", prompt, flags=re.IGNORECASE)
    return match.group(1).strip().lower() if match else ""


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_B40_CLARIFICATION_ATTRIBUTION"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("B40 clarification-attribution governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B40 clarification-attribution binding changed: {relative}")
    return protocol, sha256_file(path)


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable B40 clarification attribution exists: {output}")
    screen = _json(root / protocol["source_result"])
    screen_evidence = hashlib.sha256(
        canonical_json_bytes(_without(screen, "evidence_sha256"))
    ).hexdigest()
    probes_list = development_probes(root / protocol["development_catalog"])
    probes = {str(probe["probe_id"]): probe for probe in probes_list}
    teacher = {
        str(row["probe_id"]): row for row in _rows(root / protocol["teacher_reference"])
    }
    rows_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for system in screen["systems"]:
        rows_by_seed[int(system["seed"])] = {
            str(row["probe_id"]): row
            for row in _rows(root / system["outputs"]["path"])
        }

    lineage_protocol = _json(root / protocol["lineage_protocol"])
    manifest = _json(root / protocol["budget_manifest"])
    selected, budget = _selected_rows(root, lineage_protocol, manifest, "B40")
    selected_clarification = [
        row for row in selected["phase1_ir"] if row["capability"] == "clarification"
    ]
    selection_hashes = {
        int(seed): _json(root / path)["selection_sha256"]
        for seed, path in protocol["lineage_results"].items()
    }
    failed = [
        row
        for row in rows_by_seed[155921].values()
        if row["capability"] == "clarification" and not row["functional_pass_v1"]
    ]
    records: list[dict[str, Any]] = []
    for row in sorted(failed, key=lambda value: str(value["probe_id"])):
        probe_id = str(row["probe_id"])
        probe = probes[probe_id]
        prompt = str(probe["prompt"])
        phrase = _object_phrase(prompt)
        matching_source = [
            source
            for source in selected_clarification
            if phrase
            and phrase
            in (
                str(source.get("normalized_generation_prompt", ""))
                + "\n"
                + str(source.get("normalized_output", ""))
            ).lower()
        ]
        records.append(
            {
                "probe_id": probe_id,
                "object_phrase": phrase,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "failure_taxonomy": clarification_failure_taxonomy(str(row["output"])),
                "host_output_unchanged_from_v19": not bool(
                    row["output_changed_from_v19_history"]
                ),
                "route_correct_all_seeds": all(
                    rows_by_seed[seed][probe_id]["automatic_capability_route"]
                    == "clarification"
                    and rows_by_seed[seed][probe_id]["capability_route_correct"]
                    for seed in (104729, 130363, 155921)
                ),
                "passing_seed104729": bool(
                    rows_by_seed[104729][probe_id]["functional_pass_v1"]
                ),
                "passing_seed130363": bool(
                    rows_by_seed[130363][probe_id]["functional_pass_v1"]
                ),
                "cached_teacher_pass": evaluate_functional(
                    str(teacher[probe_id]["output"]), probe["evaluator"]
                ),
                "selected_clarification_records_mentioning_object": len(matching_source),
                "selected_object_records_with_functional_target": sum(
                    evaluate_functional(
                        str(source.get("normalized_output", "")), probe["evaluator"]
                    )
                    for source in matching_source
                ),
            }
        )

    unique_prompt_hashes = {record["prompt_sha256"] for record in records}
    taxonomy_counts = {
        key: sum(bool(record["failure_taxonomy"][key]) for record in records)
        for key in ("missing_question_mark", "missing_inquiry_marker", "empty_output")
    }
    gates = {
        "source_result_hash": sha256_file(root / protocol["source_result"])
        == protocol["bindings"][protocol["source_result"]],
        "source_evidence_hash": screen_evidence == screen["evidence_sha256"],
        "verified_mixed_topology_bound": screen["topology"] == [True, True, False],
        "exact_ten_failed_rows": len(records) == 10,
        "all_failures_interrogative_form": taxonomy_counts["missing_question_mark"] == 10
        and taxonomy_counts["missing_inquiry_marker"] == 10,
        "all_outputs_nonempty": taxonomy_counts["empty_output"] == 0,
        "all_host_outputs_unchanged": all(
            record["host_output_unchanged_from_v19"] for record in records
        ),
        "all_routes_correct": all(record["route_correct_all_seeds"] for record in records),
        "both_other_seeds_pass_all_ten": all(
            record["passing_seed104729"] and record["passing_seed130363"]
            for record in records
        ),
        "cached_teacher_passes_all_ten": all(
            record["cached_teacher_pass"] for record in records
        ),
        "same_selection_all_seeds": len(set(selection_hashes.values())) == 1
        and next(iter(selection_hashes.values())) == budget["selection_sha256"],
        "b40_selection_recomputed": budget["selection_sha256"]
        == "e494d251f8517495ab216128ca323755cade8015754f64aaacf28ad252e5bee6",
        "selected_clarification_depth": len(selected_clarification) == 200,
        "every_failed_object_covered_by_functional_source_targets": all(
            record["selected_object_records_with_functional_target"] > 0
            for record in records
        ),
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_loading_absent": True,
        "final_test_not_accessed": True,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase4-v22-b40-clarification-attribution-result/1",
        "status": "PASS_B40_CLARIFICATION_FAILURE_ATTRIBUTED_TO_SEED_DEPENDENT_ACQUISITION_REALIZATION"
        if passed
        else "FAIL_B40_CLARIFICATION_ATTRIBUTION",
        "protocol_sha256": protocol_sha,
        "source_result_sha256": sha256_file(root / protocol["source_result"]),
        "source_evidence_sha256": screen["evidence_sha256"],
        "failed_rows": records,
        "failed_row_count": len(records),
        "distinct_failed_prompt_count": len(unique_prompt_hashes),
        "distinct_failed_object_phrases": sorted(
            {record["object_phrase"] for record in records}
        ),
        "failure_taxonomy_counts": taxonomy_counts,
        "selected_information": {
            "selection_sha256": budget["selection_sha256"],
            "same_across_all_seeds": len(set(selection_hashes.values())) == 1,
            "selected_clarification_records": len(selected_clarification),
            "b40_unique_source_attempts": budget["unique_source_attempts"],
            "b40_authoritative_teacher_output_tokens": budget[
                "authoritative_teacher_output_tokens"
            ],
        },
        "gates": gates,
        "measured_owner": (
            "Seed-dependent B40 acquisition/optimization realization of interrogative "
            "clarification form. The same frozen information and v22 host pass every "
            "affected row in both other seeds; the failing host outputs are unchanged "
            "from v19, correctly routed, nonempty, and covered by selected functional "
            "clarification targets."
        ),
        "layercake_host_failure": False,
        "missing_imported_object_coverage": False,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "stable_minimum_established": False,
        "claim_boundary": (
            "Read-only prompt-level B40 failure attribution. No repair, stable minimum, "
            "matched baseline, final test, Phase 4, or ABI-superiority claim."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
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
