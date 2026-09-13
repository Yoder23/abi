"""Measure the exact R60 loss attributable to its 90 route errors."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from abi.capability_compiler_phase2_common import sha256_file
from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import load_layercake_core
from experiments.deep_sparse_adapters_r55 import screen_v1 as r55
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    write_json_once,
    write_jsonl_once,
)
from experiments.irreversible_collapse_invariant_r59 import screen_v1 as r59
from experiments.isolated_capability_cakes_r53 import fit_router


CATALOG_SHA256 = "4b0087c9a7fa0e0fd6f607fdbd94fffbdd3cb375f5880a0588c97414583a2ad7"
R60_RESULT_SHA256 = "755c285d52bf50b3a6826ef5112cc111346740e04b8f75f74e15fa60d6ef4140"
R60_RAW_SHA256 = "e509484babfab685c91fee3a6165bfea0b2881213fdb7b3a71fde9266e2a25cf"
R60_LIVE_RECEIPT_SHA256 = "9e9070d820ed241d6c6f14590110338c38914a9935840c0817c864e64a513db2"


class AttributionError(RuntimeError):
    pass


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        raise AttributionError(f"required raw rows absent: {path}")
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AttributionError(f"raw rows unreadable: {path}") from error
    if any(not isinstance(row, dict) for row in values):
        raise AttributionError(f"raw rows contain a non-object: {path}")
    return values


def run(
    candidate: Path,
    catalog_path: Path,
    r60_dir: Path,
    layercake_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise AttributionError(f"immutable R61 output exists: {output}")
    frozen = (
        (candidate / "model.safetensors", r55.CANDIDATE_SHA256),
        (candidate / "metadata.json", r55.METADATA_SHA256),
        (catalog_path, CATALOG_SHA256),
        (r60_dir / "result.json", R60_RESULT_SHA256),
        (r60_dir / "evaluation.jsonl", R60_RAW_SHA256),
        (r60_dir / "strict_verification_live.json", R60_LIVE_RECEIPT_SHA256),
    )
    for path, digest in frozen:
        if not path.is_file() or sha256_file(path) != digest:
            raise AttributionError(f"R61 frozen input changed: {path}")
    r55._preflight(candidate)
    if not torch.cuda.is_available():
        raise AttributionError("R61 requires CUDA")
    probes = list(load_probe_catalog(catalog_path)["probes"])
    baseline = _jsonl(r60_dir / "evaluation.jsonl")
    if (
        len(probes) != 1_400
        or len(baseline) != 1_400
        or [row["probe_id"] for row in baseline]
        != [probe["probe_id"] for probe in probes]
    ):
        raise AttributionError("R61 matrix identity changed")

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
    for probe, parent in zip(probes, baseline, strict=True):
        expected_route = fit_router.CAPABILITY_TO_INDEX[str(probe["capability"])]
        changed = not bool(parent["route_correct"])
        if changed:
            output_text, token_ids, latency, physical = r59._generate(
                model,
                tokenizer,
                str(probe["prompt"]),
                expected_route,
                int(probe["max_new_tokens"]),
                device,
                telemetry=telemetry,
            )
            if not physical or not all(calls == (expected_route,) for calls in physical):
                raise AttributionError(f"R61 sparse execution changed: {probe['probe_id']}")
            physical_sparse = max(map(len, physical)) == 1
        else:
            output_text = str(parent["candidate_output"])
            token_ids = list(parent["candidate_output_token_ids"])
            latency = None
            physical_sparse = bool(parent["candidate_selected_route_only"])
        passed, score = evaluate_output(output_text, probe["evaluator"])
        collapse = _collapse_metrics(
            token_ids,
            output_text,
            tokenizer.encode(str(probe["prompt"]) + "\n"),
            str(probe["prompt"]),
        )
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "capability": probe["capability"],
                "prompt_sha256": hashlib.sha256(
                    str(probe["prompt"]).encode("utf-8")
                ).hexdigest(),
                "baseline_route": parent["route"],
                "expected_route": expected_route,
                "route_changed": changed,
                "executed_live": changed,
                "output": output_text,
                "output_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest(),
                "output_token_ids": token_ids,
                "functional_pass": bool(passed),
                "functional_score": float(score),
                "final_collapse": collapse,
                "latency_seconds": latency,
                "physical_sparse": physical_sparse,
                "baseline_output_exact_when_route_unchanged": (
                    output_text == parent["candidate_output"] if not changed else None
                ),
            }
        )
        if changed and sum(row["executed_live"] for row in rows) % 10 == 0:
            print(
                json.dumps(
                    {
                        "changed_rows_executed": sum(row["executed_live"] for row in rows),
                        "changed_rows_functional": sum(
                            row["route_changed"] and row["functional_pass"] for row in rows
                        ),
                    }
                ),
                flush=True,
            )

    output.mkdir(parents=True)
    raw_path = output / "evaluation.jsonl"
    write_jsonl_once(raw_path, rows)
    by_capability = {
        capability: {
            "functional": sum(
                row["capability"] == capability and row["functional_pass"] for row in rows
            ),
            "changed_rows": sum(
                row["capability"] == capability and row["route_changed"] for row in rows
            ),
        }
        for capability in sorted(Counter(row["capability"] for row in rows))
    }
    baseline_functional = sum(row["candidate_functional_pass"] for row in baseline)
    functional = sum(row["functional_pass"] for row in rows)
    result = {
        "format": "abi-r61-route-failure-attribution/1",
        "verdict": "PASS_R61_ROUTE_EFFECT_QUANTIFIED",
        "metrics": {
            "rows": len(rows),
            "changed_rows_executed_live": sum(row["executed_live"] for row in rows),
            "unchanged_rows_bound_to_r60_live_receipt": sum(
                not row["executed_live"] for row in rows
            ),
            "baseline_functional": baseline_functional,
            "forced_route_functional": functional,
            "functional_delta": functional - baseline_functional,
            "baseline_format_control": 0,
            "forced_route_format_control": by_capability["format_control"]["functional"],
            "final_collapses": sum(
                row["final_collapse"]["collapse_detected"] for row in rows
            ),
            "physical_sparse_rows": sum(row["physical_sparse"] for row in rows),
            "by_capability": by_capability,
            "wall_seconds": time.perf_counter() - started,
        },
        "runtime_telemetry_changed_rows": telemetry,
        "artifacts": {
            "evaluation": {
                "path": raw_path.name,
                "sha256": sha256_file(raw_path),
                "bytes": raw_path.stat().st_size,
            }
        },
        "training_steps": 0,
        "teacher_calls": 0,
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
    parser.add_argument("--r60-dir", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.candidate.resolve(),
        args.catalog.resolve(),
        args.r60_dir.resolve(),
        args.layercake_root.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()

