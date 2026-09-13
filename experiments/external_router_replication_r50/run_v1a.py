"""Evaluate the unchanged R49 host against sealed live R50 final-test source rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import load_layercake_core
from abi.layercake_host import CAPABILITY_TO_ROUTE, _sha256_file
from experiments.external_router_cake_r49 import run_v1 as r49
from experiments.external_router_replication_r50.verify_source_v1a import verify as verify_source
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once


R49_RESULT_SHA256 = "a9014281204b1b1c05be8df28b414c2df1c2a524b2a223609aa8fc31275446a1"
ROUTER_SHA256 = "20ae6562432cc59ece5b9cb552caaa16da4824323e62460c6c33feb0579df863"
SOURCE_RECEIPT_SHA256 = "fabfeb5e151b190721054f49644ebbf285741b19db76e492f65e6d824025843e"
SOURCE_OUTPUTS_SHA256 = "8ead4c8fb110a9dff7f88122d6ce0271d9eb74c63afca485f43df4cc24213f95"


class R50RepairError(RuntimeError):
    pass


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise R50RepairError(f"required source rows missing: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise R50RepairError("source rows are unreadable") from error
    if any(not isinstance(row, dict) for row in rows):
        raise R50RepairError("source rows contain a non-object")
    return rows


def run(
    candidate: Path,
    router_path: Path,
    r49_result: Path,
    catalog_path: Path,
    source_dir: Path,
    layercake_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise R50RepairError(f"immutable R50-v1a output exists: {output}")
    source_receipt_path = source_dir / "receipt.json"
    source_outputs_path = source_dir / "source_outputs.jsonl"
    frozen = (
        (candidate / "model.safetensors", r49.CHECKPOINT_SHA256),
        (candidate / "metadata.json", r49.METADATA_SHA256),
        (router_path, ROUTER_SHA256),
        (r49_result, R49_RESULT_SHA256),
        (catalog_path, r49.CATALOG_SHA256),
        (source_receipt_path, SOURCE_RECEIPT_SHA256),
        (source_outputs_path, SOURCE_OUTPUTS_SHA256),
    )
    for path, digest in frozen:
        if not path.is_file() or _sha256_file(path) != digest:
            raise R50RepairError(f"R50-v1a frozen input changed: {path}")
    if not torch.cuda.is_available():
        raise R50RepairError("R50-v1a requires CUDA")
    # This recomputes the source corpus from raw rows and re-hashes the pinned
    # source snapshot before any candidate model is loaded.
    verify_source(Path.cwd().resolve(), catalog_path, source_dir)

    catalog = load_probe_catalog(catalog_path)
    probes = [dict(row) for row in catalog["probes"] if row["split"] == "final_test"]
    if (
        len(probes) != 1_400
        or any(sum(row["capability"] == capability for row in probes) != 100 for capability in CAPABILITY_TO_ROUTE)
    ):
        raise R50RepairError("R50-v1a final-test matrix changed")
    source_rows = _jsonl(source_outputs_path)
    source = {str(row["probe_id"]): row for row in source_rows}
    selected_ids = {str(row["probe_id"]) for row in probes}
    if len(source) != 1_400 or set(source) != selected_ids:
        raise R50RepairError("R50-v1a source coverage changed")

    router = torch.nn.Linear(r49.FEATURES, 10)
    router.load_state_dict(load_file(str(router_path), device="cpu"), strict=True)
    router.eval()
    features, labels = r49._matrix(probes)
    router_score = r49._score(router, features, labels)
    if router_score["accuracy"] != 1.0 or router_score["rotated_correct"] != 0:
        raise R50RepairError("R50-v1a frozen router failed final-test prerequisite")

    device = torch.device("cuda")
    model, tokenizer, _ = load_layercake_core(candidate, layercake_root=layercake_root, device=device)
    model.eval()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, probe in enumerate(probes, 1):
        prompt = str(probe["prompt"])
        route = int(router(r49._feature(prompt)).argmax())
        text, tokens, latency, physical = r49._generate(
            model, tokenizer, prompt, route, int(probe["max_new_tokens"]), device
        )
        passed, score = evaluate_output(text, probe["evaluator"])
        source_row = source[str(probe["probe_id"])]
        rows.append({
            "probe_id": probe["probe_id"],
            "capability": probe["capability"],
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "evaluator": probe["evaluator"],
            "output": text,
            "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "output_token_ids": tokens,
            "functional_pass": bool(passed),
            "functional_score": float(score),
            "route": route,
            "expected_route": CAPABILITY_TO_ROUTE[str(probe["capability"])],
            "route_correct": route == CAPABILITY_TO_ROUTE[str(probe["capability"])],
            "maximum_cakes_called_per_model_invocation": max(map(len, physical)),
            "all_calls_selected_only": all(value == (route,) for value in physical),
            "latency_seconds": latency,
            "collapse": _collapse_metrics(tokens, text, tokenizer.encode(prompt + "\n"), prompt),
            "generation_error": None,
            "source_probe_id": source_row["probe_id"],
            "source_output_sha256": source_row["output_sha256"],
            "source_passed": bool(source_row["passed"]),
            "source_score": float(source_row["score"]),
            "source_passing_regression": bool(source_row["passed"] and not passed),
        })
        if index % 100 == 0:
            print(json.dumps({
                "evaluated": index,
                "functional": sum(row["functional_pass"] for row in rows),
                "source_functional": sum(row["source_passed"] for row in rows),
                "collapses": sum(row["collapse"]["collapse_detected"] for row in rows),
            }), flush=True)

    output.mkdir(parents=True)
    raw_path = output / "evaluation.jsonl"
    write_jsonl_once(raw_path, rows)
    by_capability = {
        capability: {
            "rows": sum(row["capability"] == capability for row in rows),
            "functional": sum(row["capability"] == capability and row["functional_pass"] for row in rows),
            "source_functional": sum(row["capability"] == capability and row["source_passed"] for row in rows),
            "collapses": sum(row["capability"] == capability and row["collapse"]["collapse_detected"] for row in rows),
            "route_correct": sum(row["capability"] == capability and row["route_correct"] for row in rows),
        }
        for capability in sorted(CAPABILITY_TO_ROUTE)
    }
    functional = sum(row["functional_pass"] for row in rows)
    source_functional = sum(row["source_passed"] for row in rows)
    regressions = sum(row["source_passing_regression"] for row in rows)
    if source_functional == 0:
        raise R50RepairError("R50-v1a source has no passing rows")
    metrics = {
        "rows": len(rows),
        "functional": functional,
        "source_functional": source_functional,
        "source_passing_regressions": regressions,
        "source_retention": (source_functional - regressions) / source_functional,
        "candidate_minus_source": r49._bootstrap(
            [row["functional_pass"] for row in rows],
            [row["source_passed"] for row in rows],
        ),
        "collapses": sum(row["collapse"]["collapse_detected"] for row in rows),
        "generation_errors": sum(row["generation_error"] is not None for row in rows),
        "route_correct": sum(row["route_correct"] for row in rows),
        "physical_sparse_rows": sum(
            row["all_calls_selected_only"]
            and row["maximum_cakes_called_per_model_invocation"] == 1
            for row in rows
        ),
        "by_capability": by_capability,
        "generation_wall_seconds": time.perf_counter() - started,
    }
    gates = {
        "matrix": len(rows) == 1_400,
        "functional": functional >= 1_260,
        "per_capability": all(value["functional"] >= 65 for value in by_capability.values()),
        "source_noninferior_point": functional >= source_functional,
        "source_retention": metrics["source_retention"] >= 0.94,
        "zero_collapse": metrics["collapses"] == 0,
        "zero_generation_error": metrics["generation_errors"] == 0,
        "route_exact": metrics["route_correct"] == 1_400,
        "physical_sparse": metrics["physical_sparse_rows"] == 1_400,
        "artifacts_unchanged": all(_sha256_file(path) == digest for path, digest in frozen),
    }
    if not math.isfinite(metrics["generation_wall_seconds"]) or metrics["generation_wall_seconds"] <= 0:
        raise R50RepairError("R50-v1a generation wall time is invalid")
    passed = all(gates.values())
    result = {
        "format": "abi-r50-external-router-final-test-live-source/1",
        "verdict": "PASS_R50_HELD_SPLIT_LIVE_SOURCE" if passed else "FAIL_R50_HELD_SPLIT_LIVE_SOURCE",
        "inputs": {
            "candidate_checkpoint_sha256": r49.CHECKPOINT_SHA256,
            "candidate_metadata_sha256": r49.METADATA_SHA256,
            "router_sha256": ROUTER_SHA256,
            "r49_result_sha256": R49_RESULT_SHA256,
            "catalog_sha256": r49.CATALOG_SHA256,
            "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
            "source_outputs_sha256": SOURCE_OUTPUTS_SHA256,
        },
        "router_final_test": router_score,
        "metrics": metrics,
        "gates": gates,
        "artifacts": {
            "evaluation": {
                "path": raw_path.name,
                "sha256": _sha256_file(raw_path),
                "bytes": raw_path.stat().st_size,
            }
        },
        "candidate_training_steps": 0,
        "candidate_teacher_calls": 0,
        "source_capture_teacher_sequences": 1_400,
        "teacher_present_at_candidate_inference": False,
        "symbolic_output_calls": 0,
        "planner_calls": 0,
        "claim_ceiling": "SAME_MACHINE_SAME_CATALOG_HELD_SPLIT_LIVE_SOURCE_REPLICATION",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--r49-result", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.candidate.resolve(),
        args.router.resolve(),
        args.r49_result.resolve(),
        args.catalog.resolve(),
        args.source_dir.resolve(),
        args.layercake_root.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
