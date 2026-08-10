"""Numerically stable replay of the fixed rank-1044 coefficient audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from . import capability_compiler_phase3_rank1044_coefficient_audit as original
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


RESULT_FORMAT = "abi-capability-compiler-phase3-rank1044-stable-solver-audit/1"


def _stable_solver_factory(*, chunk_tokens: int, diagnostics: list[dict]):
    def solve(features: torch.Tensor, targets: torch.Tensor, relative_ridge: float):
        columns = features.shape[1]
        outputs = targets.shape[1]
        device = features.device
        gram = torch.zeros(columns, columns, dtype=torch.float64, device=device)
        cross = torch.zeros(columns, outputs, dtype=torch.float64, device=device)
        for start in range(0, features.shape[0], chunk_tokens):
            stop = min(start + chunk_tokens, features.shape[0])
            feature_chunk = features[start:stop].float()
            target_chunk = targets[start:stop].float()
            gram.add_((feature_chunk.transpose(0, 1) @ feature_chunk).double())
            cross.add_((feature_chunk.transpose(0, 1) @ target_chunk).double())
        scale = float(torch.trace(gram) / columns)
        ridge = float(relative_ridge) * scale
        system = gram + ridge * torch.eye(columns, dtype=torch.float64, device=device)
        solution = torch.linalg.solve(system, cross).float()
        zero_sse = 0.0
        solution_sse = 0.0
        for start in range(0, features.shape[0], chunk_tokens):
            stop = min(start + chunk_tokens, features.shape[0])
            feature_chunk = features[start:stop].double()
            target_chunk = targets[start:stop].double()
            zero_sse += float(target_chunk.square().sum())
            solution_sse += float((feature_chunk @ solution.double() - target_chunk).square().sum())
        penalty = ridge * float(solution.double().square().sum())
        objective = solution_sse + penalty
        diagnostics.append(
            {
                "features": int(features.shape[0]),
                "columns": int(columns),
                "outputs": int(outputs),
                "ridge": ridge,
                "zero_solution_sse": zero_sse,
                "solution_sse": solution_sse,
                "ridge_penalty": penalty,
                "penalized_objective": objective,
                "penalized_objective_ratio_to_zero": objective / max(zero_sse, 1e-300),
                "unpenalized_sse_ratio_to_zero": solution_sse / max(zero_sse, 1e-300),
            }
        )
        return solution, ridge

    return solve


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    numeric = protocol.get("numeric_solver", {})
    if (
        protocol.get("format") != original.FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_RANK1044_COEFFICIENT_AUDIT"
        or numeric.get("method") != "FP32_CHUNK_PRODUCTS_FP64_ACCUMULATION_AND_SOLVE"
        or int(numeric.get("chunk_tokens", 0)) != 8192
        or float(numeric.get("objective_tolerance", -1)) != 1e-6
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("rank1044 stable-solver governance changed")
    if output.exists():
        raise Phase3Error("rank1044 stable-solver output exists")

    diagnostics: list[dict] = []
    replacement = _stable_solver_factory(
        chunk_tokens=int(numeric["chunk_tokens"]), diagnostics=diagnostics
    )
    prior = original.closed.solve_ridge
    original.closed.solve_ridge = replacement
    try:
        source_result = original.execute(root, protocol_path, output / "solver_replay")
    finally:
        original.closed.solve_ridge = prior

    source_path = output / "solver_replay" / "result.json"
    tolerance = float(numeric["objective_tolerance"])
    objective_valid = len(diagnostics) == 4 and all(
        row["penalized_objective_ratio_to_zero"] <= 1.0 + tolerance for row in diagnostics
    )
    training_bound_valid = (
        float(source_result["training_coefficient_relative_rmse"]) <= 1.0 + tolerance
    )
    gates = dict(source_result["gates"])
    gates["numeric_objectives_valid"] = objective_valid
    gates["sequential_training_bound_valid"] = training_bound_valid
    passed = all(gates.values())
    result = dict(source_result)
    result.update(
        {
            "format": RESULT_FORMAT,
            "status": "PASS_RANK1044_STABLE_COEFFICIENT_REALIZATION_AUDIT"
            if passed
            else "FAIL_RANK1044_STABLE_COEFFICIENT_REALIZATION_AUDIT",
            "source_solver_replay": "solver_replay/result.json",
            "source_solver_replay_sha256": sha256_file(source_path),
            "source_solver_evidence_sha256": source_result["evidence_sha256"],
            "numeric_solver": numeric,
            "solve_diagnostics": diagnostics,
            "gates": gates,
            "passed": passed,
            "phase3_certified": False,
            "claim_boundary": "Stable-solver replay of the single fixed rank-1044 coefficient realizability audit only; no installed host, checkpoint, physical runtime, autonomous quality, Phase 3, or superiority claim.",
        }
    )
    result.pop("evidence_sha256", None)
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_RANK1044_STABLE_SOLVER_AUDIT_PROTOCOL_V401.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results/abi_capability_compiler_phase3_native_trajectory/rank1044_stable_solver_v402",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
