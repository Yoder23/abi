"""Evaluate the frozen R49 construction on the untouched final-test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file

from abi.english_generalization_evaluation import _collapse_metrics, _source_by_probe
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import load_layercake_core
from abi.layercake_host import CAPABILITY_TO_ROUTE, _sha256_file
from experiments.external_router_cake_r49 import run_v1 as r49
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once


R49_RESULT_SHA256 = "a9014281204b1b1c05be8df28b414c2df1c2a524b2a223609aa8fc31275446a1"
ROUTER_SHA256 = "20ae6562432cc59ece5b9cb552caaa16da4824323e62460c6c33feb0579df863"


class R50Error(RuntimeError):
    pass


def run(candidate: Path, router_path: Path, r49_result: Path, catalog_path: Path, source_paths: Sequence[Path], layercake_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R50Error(f"immutable R50 output exists: {output}")
    if (
        _sha256_file(candidate / "model.safetensors") != r49.CHECKPOINT_SHA256
        or _sha256_file(candidate / "metadata.json") != r49.METADATA_SHA256
        or _sha256_file(router_path) != ROUTER_SHA256
        or _sha256_file(r49_result) != R49_RESULT_SHA256
        or _sha256_file(catalog_path) != r49.CATALOG_SHA256
        or tuple(_sha256_file(path) for path in source_paths) != r49.SOURCE_SHA256
    ):
        raise R50Error("R50 frozen input changed")
    if not torch.cuda.is_available():
        raise R50Error("R50 requires CUDA")
    catalog = load_probe_catalog(catalog_path)
    probes = [dict(row) for row in catalog["probes"] if row["split"] == "final_test"]
    if len(probes) != 1_400 or any(sum(row["capability"] == capability for row in probes) != 100 for capability in CAPABILITY_TO_ROUTE):
        raise R50Error("R50 final-test matrix changed")
    router = torch.nn.Linear(r49.FEATURES, 10)
    router.load_state_dict(load_file(str(router_path), device="cpu"), strict=True)
    router.eval()
    features, labels = r49._matrix(probes)
    router_score = r49._score(router, features, labels)
    if router_score["accuracy"] != 1.0:
        raise R50Error("R50 frozen router failed final-test prerequisite")
    source, identities = _source_by_probe(source_paths, split="final_test")
    selected = {str(row["probe_id"]) for row in probes}
    if selected - set(source):
        raise R50Error("R50 source evidence is incomplete")
    device = torch.device("cuda")
    model, tokenizer, _ = load_layercake_core(candidate, layercake_root=layercake_root, device=device)
    model.eval()
    rows = []
    started = time.perf_counter()
    for index, probe in enumerate(probes, 1):
        prompt = str(probe["prompt"])
        route = int(router(r49._feature(prompt)).argmax())
        text, tokens, latency, physical = r49._generate(model, tokenizer, prompt, route, int(probe["max_new_tokens"]), device)
        passed, score = evaluate_output(text, probe["evaluator"])
        source_row = source[str(probe["probe_id"])]
        rows.append({
            "probe_id": probe["probe_id"], "capability": probe["capability"], "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "evaluator": probe["evaluator"],
            "output": text, "output_sha256": hashlib.sha256(text.encode()).hexdigest(), "output_token_ids": tokens,
            "functional_pass": passed, "functional_score": score, "route": route,
            "expected_route": CAPABILITY_TO_ROUTE[str(probe["capability"])],
            "route_correct": route == CAPABILITY_TO_ROUTE[str(probe["capability"])],
            "maximum_cakes_called_per_model_invocation": max(map(len, physical)),
            "all_calls_selected_only": all(value == (route,) for value in physical),
            "latency_seconds": latency,
            "collapse": _collapse_metrics(tokens, text, tokenizer.encode(prompt + "\n"), prompt),
            "generation_error": None, "source": source_row,
            "source_passing_regression": bool(source_row["passed"] and not passed),
        })
        if index % 100 == 0:
            print(json.dumps({"evaluated": index, "functional": sum(row["functional_pass"] for row in rows), "collapses": sum(row["collapse"]["collapse_detected"] for row in rows)}), flush=True)
    output.mkdir(parents=True)
    raw_path = output / "evaluation.jsonl"
    write_jsonl_once(raw_path, rows)
    by_capability = {
        capability: {
            "rows": sum(row["capability"] == capability for row in rows),
            "functional": sum(row["capability"] == capability and row["functional_pass"] for row in rows),
            "source_functional": sum(row["capability"] == capability and row["source"]["passed"] for row in rows),
            "collapses": sum(row["capability"] == capability and row["collapse"]["collapse_detected"] for row in rows),
            "route_correct": sum(row["capability"] == capability and row["route_correct"] for row in rows),
        }
        for capability in sorted(CAPABILITY_TO_ROUTE)
    }
    functional = sum(row["functional_pass"] for row in rows)
    source_functional = sum(row["source"]["passed"] for row in rows)
    regressions = sum(row["source_passing_regression"] for row in rows)
    metrics = {
        "rows": len(rows), "functional": functional, "source_functional": source_functional,
        "source_passing_regressions": regressions, "source_retention": (source_functional - regressions) / source_functional,
        "candidate_minus_source": r49._bootstrap([row["functional_pass"] for row in rows], [row["source"]["passed"] for row in rows]),
        "collapses": sum(row["collapse"]["collapse_detected"] for row in rows), "generation_errors": 0,
        "route_correct": sum(row["route_correct"] for row in rows),
        "physical_sparse_rows": sum(row["all_calls_selected_only"] and row["maximum_cakes_called_per_model_invocation"] == 1 for row in rows),
        "by_capability": by_capability, "generation_wall_seconds": time.perf_counter() - started,
    }
    gates = {
        "matrix": len(rows) == 1_400, "functional": functional >= 1_260,
        "per_capability": all(value["functional"] >= 65 for value in by_capability.values()),
        "source_noninferior_point": functional >= source_functional, "source_retention": metrics["source_retention"] >= 0.94,
        "zero_collapse": metrics["collapses"] == 0, "zero_generation_error": True,
        "route_exact": metrics["route_correct"] == 1_400, "physical_sparse": metrics["physical_sparse_rows"] == 1_400,
        "artifacts_unchanged": _sha256_file(candidate / "model.safetensors") == r49.CHECKPOINT_SHA256 and _sha256_file(router_path) == ROUTER_SHA256,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-r50-external-router-final-test/1", "verdict": "PASS_R50_HELD_SPLIT_REPLICATION" if passed else "FAIL_R50_HELD_SPLIT_REPLICATION",
        "inputs": {"candidate_checkpoint_sha256": r49.CHECKPOINT_SHA256, "candidate_metadata_sha256": r49.METADATA_SHA256, "router_sha256": ROUTER_SHA256, "r49_result_sha256": R49_RESULT_SHA256, "catalog_sha256": r49.CATALOG_SHA256, "source_sha256": list(r49.SOURCE_SHA256)},
        "router_final_test": router_score, "source_identities": identities, "metrics": metrics, "gates": gates,
        "artifacts": {"evaluation": {"path": raw_path.name, "sha256": _sha256_file(raw_path), "bytes": raw_path.stat().st_size}},
        "training_steps": 0, "teacher_calls": 0, "teacher_present_at_inference": False, "symbolic_output_calls": 0, "planner_calls": 0,
        "claim_ceiling": "SAME_LINEAGE_HELD_SPLIT_REPLICATION_NOT_NEW_TASKS_OR_UNRESTRICTED_ENGLISH", "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True); parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--r49-result", type=Path, required=True); parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, action="append", required=True); parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    run(args.candidate.resolve(), args.router.resolve(), args.r49_result.resolve(), args.catalog.resolve(), [value.resolve() for value in args.source_bundle], args.layercake_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
