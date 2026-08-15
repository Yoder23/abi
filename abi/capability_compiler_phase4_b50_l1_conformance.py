"""Bounded semantic-output conformance for the exact B50 routed-LoRA baseline."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    evaluate_functional,
    load_catalog,
    sha256_file,
)
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import wilson
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-b50-l1-conformance/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-l1-conformance-result/1"
SEEDS = (104729, 130363, 155921)
CRITICAL_CAPABILITIES = ("prompt_grounding", "instruction_following", "abstention")
SOURCE_PHRASE = "cannot be known"
REPLACEMENT_PHRASE = "is unknown"
SOURCE_PATTERN = re.compile(r"\bcannot be known\b", flags=re.IGNORECASE)


def conform_output(output: str, capability: str) -> tuple[str, int]:
    """Apply the sole registered, semantics-preserving product rule."""
    if capability != "abstention":
        return output, 0
    return SOURCE_PATTERN.subn(REPLACEMENT_PHRASE, output)


def _load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_PARAMETER_FREE_L1_CONFORMANCE"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("L1 conformance governance changed")
    if tuple(int(seed) for seed in protocol.get("seeds", ())) != SEEDS:
        raise Phase3Error("L1 conformance seeds changed")
    rule = protocol.get("conformance_rule", {})
    if (
        rule.get("capability") != "abstention"
        or rule.get("source_phrase") != SOURCE_PHRASE
        or rule.get("replacement_phrase") != REPLACEMENT_PHRASE
        or rule.get("case_sensitive") is not False
        or rule.get("maximum_replacements_per_output") != 1
    ):
        raise Phase3Error("L1 conformance rule changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"L1 conformance binding changed: {relative}")
    return protocol, sha256_file(path)


def _validation_probes(root: Path, protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    catalog = load_catalog((root / str(protocol["development_catalog"])).resolve())
    probes = {
        str(row["probe_id"]): row
        for row in catalog["probes"]
        if row.get("split") == "validation"
        and row.get("canonical_capability") in CAPABILITIES
    }
    if len(probes) != 1400 or any(
        sum(row["canonical_capability"] == capability for row in probes.values()) != 100
        for capability in CAPABILITIES
    ):
        raise Phase3Error("L1 conformance development suite changed")
    return probes


def _jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _quality_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_capability: dict[str, Any] = {}
    for capability in CAPABILITIES:
        subset = [row for row in rows if row["capability"] == capability]
        passed = sum(bool(row["functional_pass_v1"]) for row in subset)
        per_capability[capability] = {
            "observations": len(subset),
            "functional_passes_v1": passed,
            "functional_passes_v2": sum(
                bool(row["functional_pass_v2"]) for row in subset
            ),
            "repetition_collapses_v2": sum(
                bool(row["repetition_collapse_v2"]) for row in subset
            ),
            "wilson_v1": wilson(passed, len(subset)),
        }
    return {
        "observations": len(rows),
        "functional_passes_v1": sum(bool(row["functional_pass_v1"]) for row in rows),
        "functional_passes_v2": sum(bool(row["functional_pass_v2"]) for row in rows),
        "repetition_collapses_v2": sum(
            bool(row["repetition_collapse_v2"]) for row in rows
        ),
        "per_capability": per_capability,
    }


def _absolute_gates(report: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, bool]:
    per = report["per_capability"]
    return {
        "per_capability_functional": all(
            float(row["wilson_v1"]["point"])
            >= float(thresholds["per_capability_functional_point_estimate_minimum"])
            and float(row["wilson_v1"]["lower_95"])
            >= float(thresholds["per_capability_functional_wilson_lower_minimum"])
            for row in per.values()
        ),
        "critical_capabilities": all(
            float(per[name]["wilson_v1"]["point"])
            >= float(thresholds["critical_point_minimum"])
            and float(per[name]["wilson_v1"]["lower_95"])
            >= float(thresholds["critical_wilson_lower_minimum"])
            for name in CRITICAL_CAPABILITIES
        ),
        "zero_repetition_collapse": int(report["repetition_collapses_v2"])
        <= int(thresholds["repetition_collapse_v2_count_maximum"]),
    }


def run(root: Path, protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol, protocol_sha256 = _load_protocol(root, protocol_path)
    probes = _validation_probes(root, protocol)
    thresholds = protocol["absolute_screen"]
    seed_results: list[dict[str, Any]] = []
    total_changed = 0
    changed_locations: set[tuple[int, str]] = set()
    all_semantic_preserved = True
    all_unchanged_exact = True
    all_single_replacement = True

    for seed in SEEDS:
        source_relative = str(protocol["source_outputs"][str(seed)]["path"])
        source_path = (root / source_relative).resolve()
        source_rows = _jsonl(source_path)
        if len(source_rows) != 1400 or {str(row.get("probe_id")) for row in source_rows} != set(probes):
            raise Phase3Error(f"L1 conformance prompt set changed: {seed}")
        conformed_rows: list[dict[str, Any]] = []
        changed_rows: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        for source in source_rows:
            probe_id = str(source["probe_id"])
            probe = probes[probe_id]
            capability = str(probe["canonical_capability"])
            original = str(source["output"])
            conformed, replacements = conform_output(original, capability)
            changed = conformed != original
            if changed:
                total_changed += 1
                changed_locations.add((seed, capability))
                if replacements != 1:
                    all_single_replacement = False
                before_v2 = evaluate_functional_v2(original, probe["evaluator"], capability)
                after_v2 = evaluate_functional_v2(conformed, probe["evaluator"], capability)
                before_collapse = repetition_collapse_v2(original)
                after_collapse = repetition_collapse_v2(conformed)
                semantic_preserved = before_v2 == after_v2 and before_collapse == after_collapse
                all_semantic_preserved = all_semantic_preserved and semantic_preserved
                changed_rows.append(
                    {
                        "probe_id": probe_id,
                        "capability": capability,
                        "original_output": original,
                        "conformed_output": conformed,
                        "semantic_v2_preserved": semantic_preserved,
                    }
                )
            else:
                all_unchanged_exact = all_unchanged_exact and conformed == original and replacements == 0
            v1 = evaluate_functional(conformed, probe["evaluator"])
            v2 = evaluate_functional_v2(conformed, probe["evaluator"], capability)
            collapse = repetition_collapse_v2(conformed)
            conformed_rows.append(
                {
                    "probe_id": probe_id,
                    "capability": capability,
                    "output": conformed,
                    "functional_pass_v1": v1,
                    "functional_pass_v2": v2,
                    "repetition_collapse_v2": collapse,
                    "conformance_changed": changed,
                }
            )
            metrics.append(
                {
                    "probe_id": probe_id,
                    "capability": capability,
                    "functional_pass_v1": v1,
                    "functional_pass_v2": v2,
                    "repetition_collapse_v2": collapse,
                }
            )
        conformed_rows.sort(key=lambda row: row["probe_id"])
        changed_rows.sort(key=lambda row: row["probe_id"])
        output_bytes = b"".join(canonical_json_bytes(row) for row in conformed_rows)
        changed_bytes = b"".join(canonical_json_bytes(row) for row in changed_rows)
        output_path = output_dir / f"seed{seed}" / "conformed_outputs.jsonl"
        changed_path = output_dir / f"seed{seed}" / "changed_rows.jsonl"
        _write_immutable(output_path, output_bytes)
        _write_immutable(changed_path, changed_bytes)
        report = _quality_report(metrics)
        gates = _absolute_gates(report, thresholds)
        seed_results.append(
            {
                "seed": seed,
                "source_outputs": source_relative,
                "source_outputs_sha256": sha256_file(source_path),
                "conformed_outputs": output_path.relative_to(root).as_posix(),
                "conformed_outputs_sha256": sha256_file(output_path),
                "changed_rows": changed_path.relative_to(root).as_posix(),
                "changed_rows_sha256": sha256_file(changed_path),
                "changed_output_count": len(changed_rows),
                "evaluation": report,
                "absolute_gates": gates,
                "all_quality_gates_pass": all(gates.values()),
            }
        )

    gates = {
        "all_three_seed_quality_gates_pass": all(
            row["all_quality_gates_pass"] for row in seed_results
        ),
        "semantic_v2_and_collapse_preserved_on_every_change": all_semantic_preserved,
        "all_nonmatching_outputs_byte_identical": all_unchanged_exact,
        "at_most_one_replacement_per_changed_output": all_single_replacement,
        "exact_registered_change_count": total_changed
        == int(protocol["expected"]["changed_output_count"]),
        "all_changes_bound_to_expected_seed_and_capability": changed_locations
        == {
            (
                int(protocol["expected"]["changed_seed"]),
                str(protocol["expected"]["changed_capability"]),
            )
        },
        "training_absent": True,
        "model_inference_absent": True,
        "teacher_absent": True,
        "final_test_not_accessed": True,
    }
    result: dict[str, Any] = {
        "format": RESULT_FORMAT,
        "status": (
            "PASS_PARAMETER_FREE_L1_CONFORMANCE_THREE_SEED_QUALITY"
            if all(gates.values())
            else "FAIL_PARAMETER_FREE_L1_CONFORMANCE"
        ),
        "protocol_sha256": protocol_sha256,
        "conformance_rule": protocol["conformance_rule"],
        "seed_results": seed_results,
        "changed_output_count": total_changed,
        "gates": gates,
        "trainable_parameters_added": 0,
        "model_parameters_added": 0,
        "training_performed": False,
        "model_inference_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "abi_superiority_certified": False,
        "claim_boundary": "Exact B50 routed-LoRA product conformance on previously frozen development outputs. It can qualify a matched comparator only after independent verification and same-product runtime measurement; it does not establish Phase 4 or ABI superiority.",
    }
    evidence = dict(result)
    evidence["evidence_sha256"] = __import__("hashlib").sha256(canonical_json_bytes(result)).hexdigest()
    result = evidence
    _write_immutable(output_dir / "result.json", canonical_json_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(root, args.protocol.resolve(), args.output_dir.resolve())
    print(result["status"])
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
