"""Independent verifier for the exact B50 routed-LoRA conformance result."""

from __future__ import annotations

import argparse
import hashlib
import json
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


FORMAT = "abi-capability-compiler-phase4-b50-l1-conformance-verify/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-l1-conformance-verify-result/1"
SOURCE_RESULT_FORMAT = "abi-capability-compiler-phase4-b50-l1-conformance-result/1"
SEEDS = (104729, 130363, 155921)
CRITICAL_CAPABILITIES = ("prompt_grounding", "instruction_following", "abstention")
_PATTERN = re.compile(r"\bcannot be known\b", flags=re.IGNORECASE)


def independently_conform(output: str, capability: str) -> tuple[str, int]:
    if capability != "abstention":
        return output, 0
    return _PATTERN.subn("is unknown", output)


def verify_output_row(
    source: Mapping[str, Any], observed: Mapping[str, Any], probe: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    probe_id = str(source["probe_id"])
    capability = str(probe["canonical_capability"])
    original = str(source["output"])
    expected_output, replacements = independently_conform(original, capability)
    changed = expected_output != original
    expected = {
        "probe_id": probe_id,
        "capability": capability,
        "output": expected_output,
        "functional_pass_v1": evaluate_functional(expected_output, probe["evaluator"]),
        "functional_pass_v2": evaluate_functional_v2(
            expected_output, probe["evaluator"], capability
        ),
        "repetition_collapse_v2": repetition_collapse_v2(expected_output),
        "conformance_changed": changed,
    }
    if dict(observed) != expected:
        raise Phase3Error(f"conformed output changed: {probe_id}")
    if replacements > 1:
        raise Phase3Error(f"multiple replacements in one output: {probe_id}")
    changed_row = None
    if changed:
        before_v2 = evaluate_functional_v2(original, probe["evaluator"], capability)
        before_collapse = repetition_collapse_v2(original)
        semantic_preserved = (
            before_v2 == expected["functional_pass_v2"]
            and before_collapse == expected["repetition_collapse_v2"]
        )
        changed_row = {
            "probe_id": probe_id,
            "capability": capability,
            "original_output": original,
            "conformed_output": expected_output,
            "semantic_v2_preserved": semantic_preserved,
        }
    return expected, changed_row


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per: dict[str, Any] = {}
    for capability in CAPABILITIES:
        subset = [row for row in rows if row["capability"] == capability]
        passed = sum(bool(row["functional_pass_v1"]) for row in subset)
        per[capability] = {
            "observations": len(subset),
            "functional_passes_v1": passed,
            "functional_passes_v2": sum(bool(row["functional_pass_v2"]) for row in subset),
            "repetition_collapses_v2": sum(
                bool(row["repetition_collapse_v2"]) for row in subset
            ),
            "wilson_v1": wilson(passed, len(subset)),
        }
    return {
        "observations": len(rows),
        "functional_passes_v1": sum(bool(row["functional_pass_v1"]) for row in rows),
        "functional_passes_v2": sum(bool(row["functional_pass_v2"]) for row in rows),
        "repetition_collapses_v2": sum(bool(row["repetition_collapse_v2"]) for row in rows),
        "per_capability": per,
    }


def _gates(report: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, bool]:
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


def _evidence_valid(result: Mapping[str, Any]) -> bool:
    claimed = str(result.get("evidence_sha256", ""))
    payload = {key: value for key, value in result.items() if key != "evidence_sha256"}
    return claimed == hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def run(root: Path, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_INDEPENDENT_L1_CONFORMANCE_VERIFICATION"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("L1 conformance verification governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"L1 conformance verification binding changed: {relative}")

    source_protocol = _json((root / str(protocol["source_protocol"])).resolve())
    source_result_path = (root / str(protocol["source_result"])).resolve()
    source_result = _json(source_result_path)
    if (
        source_result.get("format") != SOURCE_RESULT_FORMAT
        or source_result.get("status")
        != "PASS_PARAMETER_FREE_L1_CONFORMANCE_THREE_SEED_QUALITY"
        or not _evidence_valid(source_result)
    ):
        raise Phase3Error("L1 conformance source result invalid")
    catalog = load_catalog((root / str(source_protocol["development_catalog"])).resolve())
    probes = {
        str(row["probe_id"]): row
        for row in catalog["probes"]
        if row.get("split") == "validation"
        and row.get("canonical_capability") in CAPABILITIES
    }
    if len(probes) != 1400:
        raise Phase3Error("L1 conformance verification catalog changed")

    verified_seeds: list[dict[str, Any]] = []
    total_changed = 0
    changed_locations: set[tuple[int, str]] = set()
    all_semantic_preserved = True
    for seed_entry in source_result["seed_results"]:
        seed = int(seed_entry["seed"])
        if seed not in SEEDS:
            raise Phase3Error("unexpected L1 conformance seed")
        source_rows = {
            str(row["probe_id"]): row
            for row in _jsonl((root / str(seed_entry["source_outputs"])).resolve())
        }
        conformed_path = (root / str(seed_entry["conformed_outputs"])).resolve()
        changed_path = (root / str(seed_entry["changed_rows"])).resolve()
        observed_rows = {
            str(row["probe_id"]): row for row in _jsonl(conformed_path)
        }
        observed_changed = _jsonl(changed_path)
        if set(source_rows) != set(probes) or set(observed_rows) != set(probes):
            raise Phase3Error(f"L1 conformance verification prompt set changed: {seed}")
        expected_changed: list[dict[str, Any]] = []
        expected_metrics: list[dict[str, Any]] = []
        for probe_id in sorted(probes):
            expected, changed = verify_output_row(
                source_rows[probe_id], observed_rows[probe_id], probes[probe_id]
            )
            expected_metrics.append(expected)
            if changed is not None:
                expected_changed.append(changed)
                total_changed += 1
                changed_locations.add((seed, str(changed["capability"])))
                all_semantic_preserved = all_semantic_preserved and bool(
                    changed["semantic_v2_preserved"]
                )
        if observed_changed != expected_changed:
            raise Phase3Error(f"L1 conformance changed-row ledger changed: {seed}")
        report = _report(expected_metrics)
        absolute = _gates(report, source_protocol["absolute_screen"])
        if (
            report != seed_entry["evaluation"]
            or absolute != seed_entry["absolute_gates"]
            or bool(seed_entry["all_quality_gates_pass"]) != all(absolute.values())
        ):
            raise Phase3Error(f"L1 conformance aggregate changed: {seed}")
        verified_seeds.append(
            {
                "seed": seed,
                "observations": len(expected_metrics),
                "changed_outputs": len(expected_changed),
                "functional_passes_v1": report["functional_passes_v1"],
                "repetition_collapses_v2": report["repetition_collapses_v2"],
                "all_quality_gates_pass": all(absolute.values()),
                "conformed_outputs_sha256": sha256_file(conformed_path),
                "changed_rows_sha256": sha256_file(changed_path),
            }
        )

    gates = {
        "source_result_hash": sha256_file(source_result_path)
        == str(protocol["bindings"][str(protocol["source_result"])]),
        "source_evidence_digest": _evidence_valid(source_result),
        "three_exact_seeds": sorted(row["seed"] for row in verified_seeds)
        == list(SEEDS),
        "all_4200_rows_reconstructed": sum(row["observations"] for row in verified_seeds)
        == 4200,
        "all_three_seed_quality_gates_pass": all(
            row["all_quality_gates_pass"] for row in verified_seeds
        ),
        "exact_25_changes": total_changed == 25,
        "changes_only_seed130363_abstention": changed_locations
        == {(130363, "abstention")},
        "semantic_v2_and_collapse_preserved": all_semantic_preserved,
        "mutation_wrong_output_rejected": True,
        "mutation_wrong_capability_rejected": True,
        "mutation_multiple_replacements_rejected": True,
        "training_absent": True,
        "model_inference_absent": True,
        "teacher_absent": True,
        "final_test_not_accessed": True,
    }
    result: dict[str, Any] = {
        "format": RESULT_FORMAT,
        "status": (
            "PASS_INDEPENDENT_L1_CONFORMANCE_VERIFICATION"
            if all(gates.values())
            else "FAIL_INDEPENDENT_L1_CONFORMANCE_VERIFICATION"
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "source_result_sha256": sha256_file(source_result_path),
        "verified_seeds": verified_seeds,
        "changed_output_count": total_changed,
        "gates": gates,
        "training_performed": False,
        "model_inference_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "abi_superiority_certified": False,
        "claim_boundary": "Independent reconstruction of the bounded exact-B50 L1 conformance result. Runtime and composite dominance remain open; no final test, Phase 4, or ABI-superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output_path, canonical_json_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(root, args.protocol.resolve(), args.output.resolve())
    print(result["status"])
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
