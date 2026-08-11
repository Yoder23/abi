"""Read-only weak-family failure-to-supervision attribution for V474."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import zipfile

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_weak_residual import WEAK_CAPABILITIES
from .capability_compiler_phase3_weak_support_audit import _load_verified_acquisition_ir


FORMAT = "abi-capability-compiler-phase3-failure-to-supervision-audit/1"
BUILDER = re.compile(r"builder-(\d+)")


def builder_index(family: str) -> int | None:
    match = BUILDER.search(family)
    return int(match.group(1)) if match else None


def prompt_projection_exact(row: Mapping[str, Any]) -> bool:
    return str(row.get("normalized_generation_prompt", "")).strip() == str(
        row.get("host_conformant_acquisition_prompt", "")
    ).strip()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_FAILURE_TO_SUPERVISION_ATTRIBUTION"
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("failure-to-supervision governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"failure-to-supervision binding changed: {relative}")
    return protocol, sha256_file(path)


def _development_summary(
    probes: list[dict[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    parent: Mapping[str, Mapping[str, Any]],
    teacher: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for probe in probes:
        capability = str(probe["canonical_capability"])
        if capability not in WEAK_CAPABILITIES or probe.get("split") != "validation":
            continue
        builder = builder_index(str(probe["phase1_template_family"]))
        if builder is None:
            raise Phase3Error("development weak family lost builder identity")
        grouped[(capability, builder)].append(probe)
    result: dict[str, Any] = {}
    for capability in WEAK_CAPABILITIES:
        result[capability] = {}
        for builder in range(4):
            values = grouped[(capability, builder)]
            if len(values) != 25:
                raise Phase3Error("weak development family depth changed")
            candidate_v1 = candidate_v2 = parent_v1 = parent_v2 = teacher_v1 = teacher_v2 = 0
            collapses = 0
            for probe in values:
                probe_id = str(probe["probe_id"])
                evaluator = probe["evaluator"]
                candidate_output = str(candidate[probe_id]["output"])
                parent_output = str(parent[probe_id]["output"])
                teacher_output = str(teacher[probe_id]["output"])
                candidate_v1 += int(evaluate_functional(candidate_output, evaluator))
                candidate_v2 += int(evaluate_functional_v2(candidate_output, evaluator, capability))
                parent_v1 += int(evaluate_functional(parent_output, evaluator))
                parent_v2 += int(evaluate_functional_v2(parent_output, evaluator, capability))
                teacher_v1 += int(evaluate_functional(teacher_output, evaluator))
                teacher_v2 += int(evaluate_functional_v2(teacher_output, evaluator, capability))
                collapses += int(bool(candidate[probe_id].get("repetition_collapse_v2")))
            result[capability][str(builder)] = {
                "observations": 25,
                "candidate_v1_passes": candidate_v1,
                "candidate_v2_passes": candidate_v2,
                "parent_v1_passes": parent_v1,
                "parent_v2_passes": parent_v2,
                "teacher_v1_passes": teacher_v1,
                "teacher_v2_passes": teacher_v2,
                "candidate_v2_collapses": collapses,
            }
    return result


def _ir_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for capability in WEAK_CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        builders: dict[str, Any] = {}
        for builder in range(4):
            family = [row for row in values if builder_index(str(row.get("template_family", ""))) == builder]
            builders[str(builder)] = {
                "records": len(family),
                "host_prompt_exact": sum(prompt_projection_exact(row) for row in family),
                "teacher_generation_prompt_mismatch": sum(not prompt_projection_exact(row) for row in family),
                "repair_attempts": sum(str(row.get("attempt_kind", "initial")) != "initial" for row in family),
            }
        non_builder = [row for row in values if builder_index(str(row.get("template_family", ""))) is None]
        result[capability] = {
            "records": len(values),
            "builders": builders,
            "non_builder_records": len(non_builder),
            "host_prompt_exact": sum(prompt_projection_exact(row) for row in values),
            "teacher_generation_prompt_mismatch": sum(not prompt_projection_exact(row) for row in values),
            "repair_attempts": sum(str(row.get("attempt_kind", "initial")) != "initial" for row in values),
        }
    return result


def _journal_summary(catalog: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    probe_by_id = {str(row["probe_id"]): row for row in catalog}
    grouped: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in attempts:
        capability = str(row["canonical_capability"])
        if capability not in WEAK_CAPABILITIES:
            continue
        probe = probe_by_id[str(row["probe_id"])]
        builder = builder_index(str(probe["phase3_targeted_template_family"]))
        if builder is None:
            raise Phase3Error("targeted source family lost builder identity")
        grouped[(capability, builder)][str(row["probe_id"])].append(row)
    result: dict[str, Any] = {}
    for capability in WEAK_CAPABILITIES:
        result[capability] = {}
        for builder in range(4):
            probes = grouped[(capability, builder)]
            counts = Counter()
            for values in probes.values():
                v1 = any(
                    row.get("finish_reason") == "eos_token"
                    and evaluate_functional(str(row["output"]), row["functional_evaluator"])
                    for row in values
                )
                v2 = any(
                    row.get("finish_reason") == "eos_token"
                    and evaluate_functional_v2(str(row["output"]), row["functional_evaluator"], capability)
                    for row in values
                )
                counts["v1_eos_valid_probes"] += int(v1)
                counts["v2_eos_valid_probes"] += int(v2)
                counts["v2_only_recoverable_probes"] += int(v2 and not v1)
                counts["attempts"] += len(values)
                counts["repair_attempts"] += sum(str(row.get("kind")) != "initial" for row in values)
            result[capability][str(builder)] = {
                "source_probes": len(probes),
                **dict(counts),
            }
    return result


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable failure-to-supervision output exists: {output}")
    catalog = _json(root / protocol["development"]["catalog"])["probes"]
    candidate_rows = _jsonl(root / protocol["development"]["candidate_outputs"])
    parent_rows = _jsonl(root / protocol["development"]["parent_outputs"])
    teacher_rows = _jsonl(root / protocol["development"]["teacher_outputs"])
    candidate = {str(row["probe_id"]): row for row in candidate_rows}
    parent = {str(row["probe_id"]): row for row in parent_rows}
    teacher = {str(row["probe_id"]): row for row in teacher_rows}
    development = _development_summary(catalog, candidate, parent, teacher)
    targeted_rows = _load_verified_acquisition_ir(root / protocol["supervision"]["targeted_ir"])
    anchor_rows = load_phase1_ir(root / protocol["supervision"]["anchor_ir"])
    targeted = _ir_summary(targeted_rows)
    anchor = _ir_summary(anchor_rows)
    targeted_catalog = _json(root / protocol["source_evidence"]["catalog"])["probes"]
    journal_rows = _jsonl(root / protocol["source_evidence"]["journal"])
    journal = _journal_summary(targeted_catalog, journal_rows)
    recovery_protocol = _json(root / protocol["supervision"]["recovery_protocol"])
    recovery_stream = str(recovery_protocol["training"]["recovery_stream"])
    if not recovery_stream.startswith("targeted only"):
        raise Phase3Error("V473 recovery stream changed")
    fluent = {
        "development_candidate_v2_passes": development["fluent_realization"]["3"]["candidate_v2_passes"],
        "development_teacher_v2_passes": development["fluent_realization"]["3"]["teacher_v2_passes"],
        "targeted_selected_builder_records": targeted["fluent_realization"]["builders"]["3"]["records"],
        "anchor_selected_builder_records": anchor["fluent_realization"]["builders"]["3"]["records"],
        "source_v2_only_recoverable_probes": journal["fluent_realization"]["3"]["v2_only_recoverable_probes"],
    }
    abstention = {
        "development_candidate_v2_passes": development["abstention"]["1"]["candidate_v2_passes"],
        "targeted_selected_builder_records": targeted["abstention"]["builders"]["1"]["records"],
        "targeted_non_builder_records": targeted["abstention"]["non_builder_records"],
        "anchor_selected_builder_records": anchor["abstention"]["builders"]["1"]["records"],
        "anchor_autonomous_recovery_eligible": 0,
    }
    tone = {
        "development_candidate_v2_passes": development["tone_control"]["1"]["candidate_v2_passes"],
        "targeted_selected_builder_records": targeted["tone_control"]["builders"]["1"]["records"],
        "targeted_builder_teacher_prompt_mismatches": targeted["tone_control"]["builders"]["1"]["teacher_generation_prompt_mismatch"],
        "targeted_builder_repair_attempts": targeted["tone_control"]["builders"]["1"]["repair_attempts"],
    }
    gates = {
        "fluent_failure_has_missing_selected_builder_support": fluent["development_candidate_v2_passes"] < 25 and fluent["targeted_selected_builder_records"] == 0,
        "fluent_missing_support_recoverable_from_existing_teacher_evidence": fluent["source_v2_only_recoverable_probes"] > 0 and fluent["development_teacher_v2_passes"] > 0,
        "abstention_failed_builder_is_anchor_only_without_recovery": abstention["development_candidate_v2_passes"] < 25 and abstention["targeted_selected_builder_records"] == 0 and abstention["anchor_selected_builder_records"] > 0 and abstention["anchor_autonomous_recovery_eligible"] == 0,
        "tone_failed_builder_contains_teacher_prompt_projection_mismatch": tone["development_candidate_v2_passes"] < 25 and tone["targeted_builder_teacher_prompt_mismatches"] > 0,
        "final_test_not_accessed": True,
    }
    passed = all(gates.values())
    result = {
        "format": FORMAT,
        "status": "PASS_DATA_PROJECTION_AND_RECOVERY_GAPS_ATTRIBUTED" if passed else "FAIL_ATTRIBUTION_INCOMPLETE",
        "protocol_sha256": protocol_sha,
        "development_by_builder": development,
        "targeted_ir_by_builder": targeted,
        "anchor_ir_by_builder": anchor,
        "targeted_source_journal_by_builder": journal,
        "decisive_findings": {"fluent_realization_builder_3": fluent, "abstention_builder_1": abstention, "tone_control_builder_1": tone},
        "gates": gates,
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "artifact_mutated": False,
        "final_test_accessed": False,
        "phase3_certified": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_FAILURE_TO_SUPERVISION_AUDIT_PROTOCOL_V477.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_failure_to_supervision/audit_v478.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output)
    print(json.dumps({"status": result["status"], "decisive_findings": result["decisive_findings"], "gates": result["gates"], "evidence_sha256": result["evidence_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
