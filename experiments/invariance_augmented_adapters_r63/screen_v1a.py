"""Run the repaired R63 payload-only development screen with sealed labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from abi.capability_compiler_phase2_common import sha256_file
from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import load_layercake_core
from experiments.external_router_cake_r49 import run_v1 as r49
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    write_json_once,
    write_jsonl_once,
)
from experiments.invariance_augmented_adapters_r63 import screen_v1 as frozen
from experiments.irreversible_collapse_invariant_r59 import screen_v1 as r59
from experiments.isolated_capability_cakes_r53 import fit_router


class ScreenError(RuntimeError):
    pass


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        raise ScreenError(f"required source rows absent: {path}")
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScreenError(f"source rows unreadable: {path}") from error
    if any(not isinstance(row, dict) for row in values):
        raise ScreenError("source rows contain a non-object")
    return values


def run(
    candidate: Path,
    catalog_path: Path,
    source_dir: Path,
    layercake_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ScreenError(f"immutable R63-v1a output exists: {output}")
    frozen._preflight(candidate)
    inputs = (
        (candidate / "model.safetensors", frozen.CANDIDATE_SHA256),
        (candidate / "metadata.json", frozen.METADATA_SHA256),
        (candidate / "augmentation_receipt.json", frozen.AUGMENTATION_RECEIPT_SHA256),
        (catalog_path, frozen.CATALOG_SHA256),
        (source_dir / "receipt.json", frozen.SOURCE_RECEIPT_SHA256),
        (source_dir / "source_outputs.jsonl", frozen.SOURCE_OUTPUTS_SHA256),
    )
    for path, digest in inputs:
        if not path.is_file() or sha256_file(path) != digest:
            raise ScreenError(f"R63-v1a frozen input changed: {path}")
    if not torch.cuda.is_available():
        raise ScreenError("R63-v1a requires CUDA")
    probes = list(load_probe_catalog(catalog_path)["probes"])
    source_rows = _jsonl(source_dir / "source_outputs.jsonl")
    source = {str(row["probe_id"]): row for row in source_rows}
    counts = Counter(str(probe["capability"]) for probe in probes)
    if (
        len(probes) != 1_400
        or len(source) != 1_400
        or len(counts) != 14
        or set(counts.values()) != {100}
        or set(source) != {str(probe["probe_id"]) for probe in probes}
    ):
        raise ScreenError("R63-v1a matrix changed")

    device = torch.device("cuda")
    model, tokenizer, _ = load_layercake_core(
        candidate, layercake_root=layercake_root, device=device
    )
    model.eval()
    telemetry = {
        "candidate_transitions_checked": 0,
        "accepted_tokens": 0,
        "rejected_boundary_tokens": 0,
        "rejections_by_condition": {
            "maximum_identical_token_run": 0,
            "repeated_novel_lexical_fourgrams": 0,
        },
        "language_model_eos_stops": 0,
        "inherited_r55_lexical_truncations": 0,
    }
    rows = []
    started = time.perf_counter()
    for ordinal, probe in enumerate(probes, 1):
        prompt = str(probe["prompt"])
        route = fit_router.CAPABILITY_TO_INDEX[str(probe["capability"])]
        output_text, token_ids, latency, physical = r59._generate(
            model,
            tokenizer,
            prompt,
            route,
            int(probe["max_new_tokens"]),
            device,
            telemetry=telemetry,
        )
        if not physical or not all(calls == (route,) for calls in physical):
            raise ScreenError(f"R63-v1a sparse execution changed: {probe['probe_id']}")
        passed, score = evaluate_output(output_text, probe["evaluator"])
        collapse = _collapse_metrics(
            token_ids, output_text, tokenizer.encode(prompt + "\n"), prompt
        )
        source_row = source[str(probe["probe_id"])]
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "capability": probe["capability"],
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "evaluator": probe["evaluator"],
                "route": route,
                "route_source": "sealed_catalog_capability_label",
                "output": output_text,
                "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
                "output_token_ids": token_ids,
                "functional_pass": bool(passed),
                "functional_score": float(score),
                "final_collapse": collapse,
                "latency_seconds": latency,
                "physical_sparse": max(map(len, physical)) == 1,
                "source": {
                    "passed": bool(source_row["passed"]),
                    "score": float(source_row["score"]),
                    "output_sha256": str(source_row["output_sha256"]),
                },
                "source_passing_regression": bool(source_row["passed"] and not passed),
            }
        )
        if ordinal % 100 == 0:
            print(
                json.dumps(
                    {
                        "evaluated": ordinal,
                        "functional": sum(row["functional_pass"] for row in rows),
                        "source_functional": sum(row["source"]["passed"] for row in rows),
                        "collapses": sum(
                            row["final_collapse"]["collapse_detected"] for row in rows
                        ),
                    }
                ),
                flush=True,
            )

    output.mkdir(parents=True)
    raw_path = output / "evaluation.jsonl"
    write_jsonl_once(raw_path, rows)
    functional = sum(row["functional_pass"] for row in rows)
    source_functional = sum(row["source"]["passed"] for row in rows)
    regressions = sum(row["source_passing_regression"] for row in rows)
    by_capability = {
        capability: {
            "functional": sum(
                row["capability"] == capability and row["functional_pass"] for row in rows
            ),
            "source_functional": sum(
                row["capability"] == capability and row["source"]["passed"] for row in rows
            ),
            "collapses": sum(
                row["capability"] == capability
                and row["final_collapse"]["collapse_detected"]
                for row in rows
            ),
        }
        for capability in sorted(counts)
    }
    comparison = r49._bootstrap(
        [row["functional_pass"] for row in rows],
        [row["source"]["passed"] for row in rows],
    )
    metrics = {
        "rows": len(rows),
        "functional": functional,
        "r60_baseline_functional": 813,
        "functional_delta_from_r60": functional - 813,
        "source_functional": source_functional,
        "source_passing_regressions": regressions,
        "source_retention": (source_functional - regressions) / source_functional,
        "candidate_minus_source": comparison,
        "final_collapses": sum(
            row["final_collapse"]["collapse_detected"] for row in rows
        ),
        "physical_sparse_rows": sum(row["physical_sparse"] for row in rows),
        "by_capability": by_capability,
        "wall_seconds": time.perf_counter() - started,
    }
    gates = {
        "matrix": len(rows) == 1_400,
        "material_improvement": functional > 813,
        "functional": functional >= 1_260,
        "per_capability": all(value["functional"] >= 65 for value in by_capability.values()),
        "source_noninferior_point": functional >= source_functional,
        "source_retention": metrics["source_retention"] >= 0.94,
        "zero_final_collapse": metrics["final_collapses"] == 0,
        "physical_sparse": metrics["physical_sparse_rows"] == 1_400,
        "artifacts_unchanged": all(sha256_file(path) == digest for path, digest in inputs),
    }
    if not math.isfinite(metrics["wall_seconds"]) or metrics["wall_seconds"] <= 0:
        raise ScreenError("R63-v1a wall time invalid")
    passed = all(gates.values())
    result = {
        "format": "abi-r63-invariance-augmented-oracle-route-development-screen/1",
        "verdict": "PASS_R63_PAYLOAD_SCREEN" if passed else "FAIL_R63_PAYLOAD_SCREEN",
        "candidate_checkpoint_sha256": frozen.CANDIDATE_SHA256,
        "candidate_metadata_sha256": frozen.METADATA_SHA256,
        "augmentation_receipt_sha256": frozen.AUGMENTATION_RECEIPT_SHA256,
        "catalog_sha256": frozen.CATALOG_SHA256,
        "source_outputs_sha256": frozen.SOURCE_OUTPUTS_SHA256,
        "routing_condition": "sealed_catalog_label_oracle_not_operational_router",
        "metrics": metrics,
        "gates": gates,
        "runtime_telemetry": telemetry,
        "artifacts": {
            "evaluation": {
                "path": raw_path.name,
                "sha256": sha256_file(raw_path),
                "bytes": raw_path.stat().st_size,
            }
        },
        "teacher_calls": 0,
        "training_steps_during_screen": 0,
        "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.candidate.resolve(),
        args.catalog.resolve(),
        args.source_dir.resolve(),
        args.layercake_root.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()

