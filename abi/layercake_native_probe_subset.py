"""Run one preregistered native-runtime probe subset without promoting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .english_generalization_evaluation import _collapse_metrics
from .hf_extraction import evaluate_output, load_probe_catalog
from .layercake_host_runtime import (
    NativeHostRuntime,
    _canonical_sha,
    _sha256_file,
    _write_json,
    generate_native_host,
)


def evaluate_probe_subset(
    *,
    artifact: str | Path,
    catalog_path: str | Path,
    probe_ids: Sequence[str],
    output_path: str | Path,
    threads: int = 14,
) -> dict[str, Any]:
    artifact = Path(artifact).resolve()
    catalog_path = Path(catalog_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise RuntimeError(f"probe-subset evidence is immutable: {output_path}")
    if not probe_ids or len(set(probe_ids)) != len(probe_ids):
        raise ValueError("probe IDs must be nonempty and unique")
    catalog = load_probe_catalog(catalog_path)
    by_id = {
        str(probe["probe_id"]): probe
        for probe in catalog["probes"]
        if probe["split"] == "validation"
    }
    missing = sorted(set(probe_ids) - set(by_id))
    if missing:
        raise ValueError(f"probe IDs are absent from validation: {missing}")
    runtime = NativeHostRuntime(artifact, threads=threads)
    observations = []
    for probe_id in probe_ids:
        probe = by_id[probe_id]
        result = generate_native_host(
            runtime,
            str(probe["prompt"]),
            max_new_tokens=int(probe["max_new_tokens"]),
        )
        passed, score = evaluate_output(
            result["output"], probe["evaluator"]
        )
        collapse = _collapse_metrics(
            result["authoritative_generated_token_ids"],
            result["output"],
            runtime.encode(str(probe["prompt"]) + "\n"),
            str(probe["prompt"]),
        )
        observations.append(
            {
                "probe_id": probe_id,
                "capability": str(probe["capability"]),
                "prompt": str(probe["prompt"]),
                "evaluator": probe["evaluator"],
                "output": result["output"],
                "output_sha256": result["output_sha256"],
                "passed": passed,
                "score": score,
                "route": result["route"],
                "symbolic_handler_used": result[
                    "symbolic_handler_used"
                ],
                "collapse": collapse,
                "timing": result["timing"],
                "memory": result["memory"],
                "persistent_state": result["persistent_state"],
            }
        )
    evidence = {
        "format": "abi-layercake-native-probe-subset/1",
        "status": "DIAGNOSTIC_NOT_PROMOTABLE",
        "artifact": str(artifact),
        "runtime_metadata_evidence_sha256": runtime.metadata[
            "evidence_sha256"
        ],
        "runtime_graph_sha256": runtime.metadata["runtime"][
            "graph_sha256"
        ],
        "catalog": str(catalog_path),
        "catalog_sha256": _sha256_file(catalog_path),
        "probe_ids": list(probe_ids),
        "threads": int(threads),
        "observations": observations,
        "passes": sum(bool(row["passed"]) for row in observations),
        "collapse_count": sum(
            bool(row["collapse"]["collapse_detected"])
            for row in observations
        ),
        "final_test_accessed": False,
        "claim_boundary": (
            "This bounded diagnostic selects a native precision profile. It "
            "cannot replace the full locked semantic suite."
        ),
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--probe-id", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=14)
    args = parser.parse_args(argv)
    evidence = evaluate_probe_subset(
        artifact=args.artifact,
        catalog_path=args.catalog,
        probe_ids=args.probe_id,
        output_path=args.output,
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "observations": len(evidence["observations"]),
                "passes": evidence["passes"],
                "collapse_count": evidence["collapse_count"],
                "evidence_sha256": evidence["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
