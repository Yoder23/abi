"""Paired, prompt-level analysis for Phase 2 development evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .capability_compiler_phase2_common import CAPABILITIES, Phase2Error, canonical_json_bytes, sha256_file


def read_outputs(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def wilson(successes: int, observations: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if observations <= 0 or not 0 <= successes <= observations:
        raise Phase2Error("invalid Wilson inputs")
    p = successes / observations
    denominator = 1.0 + z * z / observations
    center = (p + z * z / (2.0 * observations)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / observations + z * z / (4.0 * observations**2)) / denominator
    return center - radius, center + radius


def validate_output_suite(rows: Sequence[Mapping[str, Any]], *, expected_per_capability: int) -> None:
    if len(rows) != len(CAPABILITIES) * expected_per_capability:
        raise Phase2Error("headline output depth changed")
    ids: set[str] = set()
    counts = {capability: 0 for capability in CAPABILITIES}
    for row in rows:
        probe_id = str(row.get("probe_id"))
        capability = str(row.get("capability"))
        if probe_id in ids or capability not in counts:
            raise Phase2Error("duplicate probe or invalid capability")
        ids.add(probe_id)
        counts[capability] += 1
    if set(counts.values()) != {expected_per_capability}:
        raise Phase2Error("per-capability output depth changed")


def stratified_paired_bootstrap(
    candidate: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    resamples: int = 10_000,
    seed: int = 1729,
) -> dict[str, float]:
    candidate_by_id = {str(row["probe_id"]): row for row in candidate}
    reference_by_id = {str(row["probe_id"]): row for row in reference}
    if set(candidate_by_id) != set(reference_by_id):
        raise Phase2Error("paired prompt identities changed")
    strata: dict[str, list[float]] = defaultdict(list)
    for probe_id in sorted(candidate_by_id):
        left = candidate_by_id[probe_id]
        right = reference_by_id[probe_id]
        if left["capability"] != right["capability"]:
            raise Phase2Error("paired capability identity changed")
        strata[str(left["capability"])].append(float(bool(left["functional_pass"])) - float(bool(right["functional_pass"])))
    if set(strata) != set(CAPABILITIES):
        raise Phase2Error("bootstrap strata changed")
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        stratum_means = []
        for capability in CAPABILITIES:
            values = np.asarray(strata[capability], dtype=np.float64)
            sampled = rng.choice(values, size=values.size, replace=True)
            stratum_means.append(float(sampled.mean()))
        draws[index] = float(np.mean(stratum_means))
    observed = float(np.mean([np.mean(values) for values in strata.values()]))
    return {
        "observed_difference": observed,
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "resamples": resamples,
        "seed": seed,
    }


def summarize_system(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observations = len(rows)
    passes = sum(bool(row["functional_pass"]) for row in rows)
    lower, upper = wilson(passes, observations)
    return {
        "observations": observations,
        "functional_passes": passes,
        "functional_rate": passes / observations,
        "functional_wilson95": [lower, upper],
        "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows),
        "output_utf8_bytes": sum(len(str(row["output"]).encode("utf-8")) for row in rows),
        "output_tokens": sum(len(row.get("output_token_ids", [])) for row in rows),
    }


def analyze(*, systems: Mapping[str, Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise Phase2Error("immutable analysis already exists")
    if "T0" not in systems:
        raise Phase2Error("T0 reference is mandatory")
    loaded = {name: read_outputs(path) for name, path in systems.items()}
    for rows in loaded.values():
        validate_output_suite(rows, expected_per_capability=100)
    result = {
        "format": "abi-capability-compiler-phase2-paired-analysis/1",
        "status": "PASS",
        "systems": {name: summarize_system(rows) for name, rows in loaded.items()},
        "paired_vs_T0": {
            name: stratified_paired_bootstrap(rows, loaded["T0"])
            for name, rows in loaded.items()
            if name != "T0"
        },
        "input_bindings": {name: {"path": path.as_posix(), "sha256": sha256_file(path)} for name, path in systems.items()},
        "final_prompts_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(result))
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", required=True, help="NAME=outputs.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    systems = {}
    for value in args.system:
        name, separator, path = value.partition("=")
        if not separator or name in systems:
            raise Phase2Error("systems must be unique NAME=PATH values")
        systems[name] = Path(path).resolve()
    result = analyze(systems=systems, output=Path(args.output).resolve())
    print(json.dumps({"status": result["status"], "systems": sorted(result["systems"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
