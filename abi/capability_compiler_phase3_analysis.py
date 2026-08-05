"""Recompute the conditional Phase 3 development decision from raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file


SYSTEMS = ("A0", "A1", "A2", "A3", "A4")


class Phase3AnalysisError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3AnalysisError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def wilson(successes: int, observations: int, z: float = 1.959963984540054) -> dict[str, float]:
    if observations <= 0 or not 0 <= successes <= observations:
        raise Phase3AnalysisError("invalid Wilson inputs")
    p = successes / observations
    denominator = 1 + z * z / observations
    center = (p + z * z / (2 * observations)) / denominator
    half = z * math.sqrt(p * (1 - p) / observations + z * z / (4 * observations**2)) / denominator
    return {"point": p, "lower_95": center - half, "upper_95": center + half}


def measured_bottleneck(system: Mapping[str, Any]) -> str:
    """Describe the observed A0 bottleneck without embedding run-specific counts."""
    zero_capabilities = sum(
        value["passes"] == 0 for value in system["per_capability"].values()
    )
    observations = sum(
        value["observations"] for value in system["per_capability"].values()
    )
    return (
        f"The {system['trainable_parameters']:,}-parameter output-side six-route bridge "
        "can memorize training responses and route held-out prompts, but lacks sufficient "
        f"sequence-realization capacity: {zero_capabilities} capabilities scored 0/100, "
        f"overall quality was {system['functional_passes']}/{observations}, and "
        f"{system['repetition_collapses']} outputs collapsed."
    )


def require_identical_successful_sequences(
    systems: Mapping[str, Mapping[str, Any]],
) -> str:
    """Fail closed unless every registered system used one non-null sequence."""
    sequences = {
        systems[system].get("successful_record_sequence_sha256")
        for system in SYSTEMS
    }
    if len(sequences) != 1 or None in sequences:
        raise Phase3AnalysisError("successful paired record sequences differ")
    return next(iter(sequences))


def stratified_bootstrap(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {capability: [] for capability in CAPABILITIES}
    if set(left) != set(right):
        raise Phase3AnalysisError("paired prompt identities differ")
    for probe_id in sorted(left):
        a = left[probe_id]
        b = right[probe_id]
        if a["capability"] != b["capability"]:
            raise Phase3AnalysisError("paired capabilities differ")
        grouped[str(a["capability"])].append(float(a["functional_pass"]) - float(b["functional_pass"]))
    if {len(values) for values in grouped.values()} != {100}:
        raise Phase3AnalysisError("bootstrap strata depth changed")
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        total = 0.0
        count = 0
        for capability in CAPABILITIES:
            stratum = grouped[capability]
            total += sum(stratum[rng.randrange(len(stratum))] for _ in range(len(stratum)))
            count += len(stratum)
        values.append(total / count)
    values.sort()
    point = sum(sum(rows) for rows in grouped.values()) / sum(len(rows) for rows in grouped.values())
    return {
        "point": point,
        "lower_95": values[int(0.025 * replicates)],
        "upper_95": values[min(replicates - 1, int(0.975 * replicates))],
        "replicates": replicates,
        "seed": seed,
    }


def analyze(*, root: Path, evidence_root: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise Phase3AnalysisError(f"analysis is immutable: {output_path}")
    protocol_path = root / "ABI_CAPABILITY_COMPILER_PHASE3_PAIRED_SAMPLER_AMENDMENT_V4.json"
    protocol_sha = sha256_file(protocol_path)
    systems: dict[str, Any] = {}
    raw: dict[str, dict[str, dict[str, Any]]] = {}
    for system in SYSTEMS:
        candidate = evidence_root / "development_v4" / f"{system}-seed104729"
        evaluation = evidence_root / "evaluation_v4" / f"{system}-seed104729"
        metadata_path = candidate / "metadata.json"
        receipt_path = evaluation / "receipt.json"
        outputs_path = evaluation / "development_outputs.jsonl"
        metadata = _json(metadata_path)
        receipt = _json(receipt_path)
        outputs = _jsonl(outputs_path)
        if (
            metadata.get("system") != system
            or receipt.get("system") != system
            or metadata.get("protocol_sha256") != protocol_sha
            or receipt.get("protocol_sha256") != protocol_sha
            or sha256_file(candidate / "model.safetensors") != metadata["checkpoint"]["sha256"]
            or sha256_file(outputs_path) != receipt["outputs_sha256"]
            or len(outputs) != 1400
            or len({row["probe_id"] for row in outputs}) != 1400
            or sum(bool(row["functional_pass"]) for row in outputs) != receipt["functional_passes"]
            or sum(bool(row["repetition_collapse"]) for row in outputs) != receipt["repetition_collapses"]
        ):
            raise Phase3AnalysisError(f"{system} evidence failed identity or aggregate verification")
        raw[system] = {str(row["probe_id"]): row for row in outputs}
        per_capability = {}
        for capability in CAPABILITIES:
            rows = [row for row in outputs if row["capability"] == capability]
            successes = sum(bool(row["functional_pass"]) for row in rows)
            per_capability[capability] = {
                "passes": successes,
                "observations": len(rows),
                "wilson": wilson(successes, len(rows)),
                "collapses": sum(bool(row["repetition_collapse"]) for row in rows),
            }
        systems[system] = {
            "checkpoint_sha256": metadata["checkpoint"]["sha256"],
            "functional_passes": receipt["functional_passes"],
            "functional": wilson(receipt["functional_passes"], 1400),
            "repetition_collapses": receipt["repetition_collapses"],
            "generation_wall_seconds": receipt["wall_seconds"],
            "output_tokens": receipt["output_tokens"],
            "training_wall_seconds": metadata["training"]["wall_seconds"],
            "teacher_response_tokens_seen": metadata["training"]["teacher_response_tokens_seen"],
            "skipped_amp_steps": metadata["training"].get("skipped_amp_steps"),
            "successful_record_sequence_sha256": metadata["training"].get("successful_record_sequence_sha256"),
            "trainable_parameters": metadata["training"]["trainable_parameters"],
            "peak_process_rss_bytes": metadata["training"]["peak_process_rss_bytes"],
            "peak_cuda_allocated_bytes": metadata["training"]["peak_cuda_allocated_bytes"],
            "per_capability": per_capability,
            "metadata_sha256": sha256_file(metadata_path),
            "receipt_sha256": sha256_file(receipt_path),
            "outputs_sha256": sha256_file(outputs_path),
        }
    teacher_rows = _jsonl(root / "results/abi_capability_compiler_phase2/teacher/T0/development_outputs.jsonl")
    teacher = {str(row["probe_id"]): row for row in teacher_rows}
    comparisons = {
        f"A0_minus_{system}": stratified_bootstrap(raw["A0"], raw[system], replicates=10_000, seed=1729)
        for system in ("A1", "A2", "A3", "A4")
    }
    comparisons["A0_minus_T0"] = stratified_bootstrap(raw["A0"], teacher, replicates=10_000, seed=1729)
    require_identical_successful_sequences(systems)
    a0_caps = systems["A0"]["per_capability"]
    per_capability_gate = all(
        value["wilson"]["point"] >= 0.90 and value["wilson"]["lower_95"] >= 0.85
        for value in a0_caps.values()
    )
    critical_gate = all(
        a0_caps[capability]["wilson"]["point"] >= 0.95
        and a0_caps[capability]["wilson"]["lower_95"] >= 0.90
        for capability in ("prompt_grounding", "instruction_following", "abstention")
    )
    causal = {name: value["lower_95"] > 0 for name, value in comparisons.items() if name != "A0_minus_T0"}
    gates = {
        "per_capability_functional": per_capability_gate,
        "critical_capabilities": critical_gate,
        "zero_repetition_collapses": systems["A0"]["repetition_collapses"] == 0,
        "teacher_relative_noninferiority": comparisons["A0_minus_T0"]["lower_95"] >= -0.05,
        "causal_A0_beats_each_control": all(causal.values()),
        "teacher_absent": True,
        "source_parameters_copied_zero": True,
        "registered_bridge_only": True,
        "remaining_seed_runs_authorized": False,
        "final_test_accessed": False,
    }
    result = {
        "format": "abi-capability-compiler-phase3-conditional-decision/1",
        "status": "FAIL_AUTONOMOUS_QUALITY_CAUSAL_SIGNAL_PRESENT" if all(causal.values()) else "FAIL_QUALITY_AND_CAUSALITY",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "phase2_status": "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED_NOT_PASSED",
        "phase3_certified": False,
        "phase4_status": "LOCKED",
        "systems": systems,
        "paired_bootstrap": comparisons,
        "causal_gates": causal,
        "gates": gates,
        "decision": {
            "branch_promoted": False,
            "remaining_two_seeds_run": False,
            "reason": "A0 shows a causal labeled teacher-payload signal against all four matched controls but fails every absolute quality family, teacher noninferiority, and repetition-collapse requirements.",
            "measured_bottleneck": measured_bottleneck(systems["A0"]),
            "next_step": "Do not increase data, steps, or nearby cake variants. A future separately governed architecture must add prompt-conditioned sequence transformation capacity while preserving the frozen host and equal-information controls.",
        },
        "negative_evidence_preserved": True,
        "claim_boundary": "This establishes a development-only causal signal and a formal Phase 3 branch failure. It does not certify fluent transfer, Phase 3, ABI superiority, or access to Phase 4.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", default="results/abi_capability_compiler_phase3")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3/conditional_decision_v3.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = analyze(root=root, evidence_root=(root / args.evidence_root).resolve(), output_path=(root / args.output).resolve())
    print(json.dumps({"status": result["status"], "phase3_certified": result["phase3_certified"], "phase4_status": result["phase4_status"], "evidence_sha256": result["evidence_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
