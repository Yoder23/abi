"""Fail-closed raw recomputation and optional live replay for R50-v1a."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import load_layercake_core
from abi.layercake_host import CAPABILITY_TO_ROUTE, _sha256_file
from experiments.external_router_cake_r49 import run_v1 as r49
from experiments.external_router_replication_r50 import run_v1a as campaign
from experiments.external_router_replication_r50.verify_source_v1a import verify as verify_source
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once


EXPECTED_RESULT_SHA256 = "b9e50631736a98ee52f5155f9a90dbd49e5f71d65732a5c932a73220881d9bb9"


class VerificationError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"required JSON absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"required JSON unreadable: {path}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"required JSON is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise VerificationError(f"required JSONL absent: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"required JSONL unreadable: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise VerificationError(f"required JSONL contains non-object row: {path}")
    return rows


def verify(
    run_dir: Path,
    candidate: Path,
    router_path: Path,
    r49_result: Path,
    catalog_path: Path,
    source_dir: Path,
    layercake_root: Path,
    live: bool,
) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    raw_path = run_dir / "evaluation.jsonl"
    if not result_path.is_file() or _sha256_file(result_path) != EXPECTED_RESULT_SHA256:
        raise VerificationError("R50-v1a frozen result changed")
    result = _json(result_path)
    unsigned = dict(result)
    stored_evidence = unsigned.pop("evidence_sha256", None)
    if not isinstance(stored_evidence, str) or evidence_hash(unsigned) != stored_evidence:
        raise VerificationError("R50-v1a evidence digest changed")
    if (
        result.get("format") != "abi-r50-external-router-final-test-live-source/1"
        or result.get("verdict") != "FAIL_R50_HELD_SPLIT_LIVE_SOURCE"
        or result.get("full_abi_moonshot") != "OPEN"
    ):
        raise VerificationError("R50-v1a result scope changed")

    source_verification = verify_source(Path.cwd().resolve(), catalog_path, source_dir)
    frozen = (
        (candidate / "model.safetensors", r49.CHECKPOINT_SHA256),
        (candidate / "metadata.json", r49.METADATA_SHA256),
        (router_path, campaign.ROUTER_SHA256),
        (r49_result, campaign.R49_RESULT_SHA256),
        (catalog_path, r49.CATALOG_SHA256),
        (source_dir / "receipt.json", campaign.SOURCE_RECEIPT_SHA256),
        (source_dir / "source_outputs.jsonl", campaign.SOURCE_OUTPUTS_SHA256),
    )
    for path, digest in frozen:
        if not path.is_file() or _sha256_file(path) != digest:
            raise VerificationError(f"R50-v1a frozen input changed: {path}")
    artifact = result.get("artifacts", {}).get("evaluation", {})
    if (
        not raw_path.is_file()
        or artifact.get("path") != raw_path.name
        or artifact.get("sha256") != _sha256_file(raw_path)
        or artifact.get("bytes") != raw_path.stat().st_size
    ):
        raise VerificationError("R50-v1a raw evaluation receipt changed")

    catalog = load_probe_catalog(catalog_path)
    probes = [dict(row) for row in catalog["probes"] if row["split"] == "final_test"]
    source_rows = _jsonl(source_dir / "source_outputs.jsonl")
    source = {str(row["probe_id"]): row for row in source_rows}
    rows = _jsonl(raw_path)
    if (
        len(probes) != 1_400
        or len(source) != 1_400
        or len(rows) != 1_400
        or [row.get("probe_id") for row in rows] != [row["probe_id"] for row in probes]
        or set(source) != {str(row["probe_id"]) for row in probes}
    ):
        raise VerificationError("R50-v1a matrix identity changed")
    router = torch.nn.Linear(r49.FEATURES, 10)
    router.load_state_dict(load_file(str(router_path), device="cpu"), strict=True)
    router.eval()
    features, labels = r49._matrix(probes)
    router_score = r49._score(router, features, labels)
    if router_score != result.get("router_final_test"):
        raise VerificationError("R50-v1a router predictions changed")

    device = torch.device("cuda" if live else "cpu")
    if live and not torch.cuda.is_available():
        raise VerificationError("R50-v1a live replay requires CUDA")
    model = None
    if live:
        model, tokenizer, _ = load_layercake_core(candidate, layercake_root=layercake_root, device=device)
        model.eval()
    else:
        _, tokenizer, _ = load_layercake_core(candidate, layercake_root=layercake_root, device="cpu")
    live_exact = 0
    for ordinal, (probe, row) in enumerate(zip(probes, rows, strict=True), 1):
        prompt = str(probe["prompt"])
        route = int(router(r49._feature(prompt)).argmax())
        output = row.get("output")
        tokens = row.get("output_token_ids")
        source_row = source[str(probe["probe_id"])]
        if not isinstance(output, str) or not isinstance(tokens, list):
            raise VerificationError(f"R50-v1a output absent: {probe['probe_id']}")
        passed, score = evaluate_output(output, probe["evaluator"])
        collapse = _collapse_metrics(tokens, output, tokenizer.encode(prompt + "\n"), prompt)
        if (
            row.get("capability") != probe["capability"]
            or row.get("prompt") != prompt
            or row.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or row.get("evaluator") != probe["evaluator"]
            or row.get("output_sha256") != hashlib.sha256(output.encode("utf-8")).hexdigest()
            or row.get("functional_pass") is not passed
            or row.get("functional_score") != score
            or row.get("route") != route
            or row.get("expected_route") != CAPABILITY_TO_ROUTE[str(probe["capability"])]
            or row.get("route_correct") != (route == CAPABILITY_TO_ROUTE[str(probe["capability"])] )
            or row.get("collapse") != collapse
            or row.get("source_probe_id") != source_row["probe_id"]
            or row.get("source_output_sha256") != source_row["output_sha256"]
            or row.get("source_passed") is not source_row["passed"]
            or row.get("source_score") != source_row["score"]
            or row.get("source_passing_regression") is not bool(source_row["passed"] and not passed)
            or row.get("generation_error") is not None
            or not isinstance(row.get("latency_seconds"), (int, float))
            or not math.isfinite(row["latency_seconds"])
            or row["latency_seconds"] <= 0
        ):
            raise VerificationError(f"R50-v1a raw row changed: {probe['probe_id']}")
        if live:
            live_output, live_tokens, _, physical = r49._generate(
                model, tokenizer, prompt, route, int(probe["max_new_tokens"]), device
            )
            if (
                live_output != output
                or live_tokens != tokens
                or not physical
                or max(map(len, physical)) != 1
                or not all(value == (route,) for value in physical)
            ):
                raise VerificationError(f"R50-v1a live replay changed: {probe['probe_id']}")
            live_exact += 1
            if ordinal % 100 == 0:
                print(json.dumps({"live_replayed": ordinal, "exact": live_exact}), flush=True)

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
        "generation_wall_seconds": result["metrics"]["generation_wall_seconds"],
    }
    if metrics != result.get("metrics"):
        raise VerificationError("R50-v1a metrics differ from raw recomputation")
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
    if gates != result.get("gates") or gates["source_retention"] is not False or sum(not value for value in gates.values()) != 1:
        raise VerificationError("R50-v1a recomputed gate vector changed")
    return {
        "format": "abi-r50-strict-verification/1",
        "verdict": "PASS_STRICT_VERIFICATION_OF_FAILED_R50",
        "result_sha256": _sha256_file(result_path),
        "evidence_sha256": stored_evidence,
        "source_verification": source_verification,
        "raw_rows_recomputed": len(rows),
        "live_rows_exact": live_exact,
        "functional": functional,
        "source_functional": source_functional,
        "source_retention": metrics["source_retention"],
        "failed_gates": [name for name, value in gates.items() if not value],
        "full_abi_moonshot": "OPEN",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--r49-result", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    receipt = verify(
        args.run_dir.resolve(), args.candidate.resolve(), args.router.resolve(),
        args.r49_result.resolve(), args.catalog.resolve(), args.source_dir.resolve(),
        args.layercake_root.resolve(), args.live,
    )
    write_json_once(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
