"""Fail-closed raw recomputation and live replay for R49."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file

from abi.english_generalization_evaluation import _collapse_metrics, _source_by_probe
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import load_layercake_core
from abi.layercake_host import CAPABILITY_TO_ROUTE, _sha256_file
from experiments.external_router_cake_r49 import run_v1 as campaign
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once


EXPECTED_RESULT_SHA256 = "a9014281204b1b1c05be8df28b414c2df1c2a524b2a223609aa8fc31275446a1"


class VerificationError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"required JSON absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"required JSON unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"required JSON is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise VerificationError(f"required JSONL absent: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"required JSONL unreadable: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise VerificationError(f"required JSONL contains non-object row: {path}")
    return rows


def verify(
    run_dir: Path,
    candidate: Path,
    catalog_path: Path,
    source_paths: Sequence[Path],
    layercake_root: Path,
    live: bool,
) -> dict[str, Any]:
    result_path = run_dir / "result.json"
    if not result_path.is_file() or _sha256_file(result_path) != EXPECTED_RESULT_SHA256:
        raise VerificationError("R49 frozen result changed")
    result = _json(result_path)
    unsigned = dict(result)
    stored_evidence = unsigned.pop("evidence_sha256", None)
    if not isinstance(stored_evidence, str) or evidence_hash(unsigned) != stored_evidence:
        raise VerificationError("R49 evidence digest changed")
    if (
        result.get("format") != "abi-r49-external-router-cake/1"
        or result.get("verdict") != "PASS_R49_EXTERNAL_ROUTER_CAKE"
        or result.get("full_abi_moonshot") != "OPEN"
    ):
        raise VerificationError("R49 result scope changed")
    frozen = (
        (candidate / "model.safetensors", campaign.CHECKPOINT_SHA256),
        (candidate / "metadata.json", campaign.METADATA_SHA256),
        (catalog_path, campaign.CATALOG_SHA256),
        *zip(source_paths, campaign.SOURCE_SHA256, strict=True),
    )
    for path, digest in frozen:
        if not path.is_file() or _sha256_file(path) != digest:
            raise VerificationError(f"R49 frozen input changed: {path}")
    artifacts = result.get("artifacts", {})
    for name, filename in (("router", "router.safetensors"), ("evaluation", "evaluation.jsonl")):
        path = run_dir / filename
        receipt = artifacts.get(name, {})
        if (
            not path.is_file()
            or receipt.get("path") != filename
            or receipt.get("sha256") != _sha256_file(path)
            or receipt.get("bytes") != path.stat().st_size
        ):
            raise VerificationError(f"R49 artifact receipt changed: {name}")
    router_receipt = _json(run_dir / "router.json")
    if router_receipt != result.get("router"):
        raise VerificationError("R49 router receipt differs from result")
    router = torch.nn.Linear(campaign.FEATURES, 10)
    router.load_state_dict(load_file(str(run_dir / "router.safetensors"), device="cpu"), strict=True)
    router.eval()
    catalog = load_probe_catalog(catalog_path)
    search = [dict(row) for row in catalog["probes"] if row["split"] == "search"]
    validation = [dict(row) for row in catalog["probes"] if row["split"] == "validation"]
    search_x, search_y = campaign._matrix(search)
    validation_x, validation_y = campaign._matrix(validation)
    if (
        len(search) != 1_400
        or len(validation) != 1_400
        or campaign._score(router, search_x, search_y) != router_receipt["search"]
        or campaign._score(router, validation_x, validation_y) != router_receipt["validation"]
    ):
        raise VerificationError("R49 router predictions changed")
    source, _ = _source_by_probe(source_paths, split="validation")
    selected_ids = {str(row["probe_id"]) for row in validation}
    missing = selected_ids - set(source)
    surplus = set(source) - selected_ids
    source_inventory = {
        "selected": len(selected_ids),
        "available": len(source),
        "missing": len(missing),
        "surplus_not_evaluated": len(surplus),
        "surplus_ids_sha256": hashlib.sha256("\n".join(sorted(surplus)).encode()).hexdigest(),
    }
    if missing or source_inventory != result.get("source_inventory"):
        raise VerificationError("R49 source inventory changed")
    rows = _jsonl(run_dir / "evaluation.jsonl")
    if len(rows) != 1_400 or [row.get("probe_id") for row in rows] != [row["probe_id"] for row in validation]:
        raise VerificationError("R49 raw matrix identity changed")

    tokenizer = None
    model = None
    device = torch.device("cuda" if live else "cpu")
    if live:
        if not torch.cuda.is_available():
            raise VerificationError("R49 live verification requires CUDA")
        model, tokenizer, _ = load_layercake_core(candidate, layercake_root=layercake_root, device=device)
        model.eval()
    else:
        _, tokenizer, _ = load_layercake_core(candidate, layercake_root=layercake_root, device="cpu")
    recomputed = []
    live_exact = 0
    for ordinal, (probe, row) in enumerate(zip(validation, rows, strict=True), 1):
        prompt = str(probe["prompt"])
        route = int(router(campaign._feature(prompt)).argmax())
        output = row.get("output")
        tokens = row.get("output_token_ids")
        if not isinstance(output, str) or not isinstance(tokens, list):
            raise VerificationError("R49 raw output or token IDs absent")
        passed, score = evaluate_output(output, probe["evaluator"])
        collapse = _collapse_metrics(tokens, output, tokenizer.encode(prompt + "\n"), prompt)
        source_row = source[str(probe["probe_id"])]
        if (
            row.get("capability") != probe["capability"]
            or row.get("prompt") != prompt
            or row.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
            or row.get("evaluator") != probe["evaluator"]
            or row.get("output_sha256") != hashlib.sha256(output.encode()).hexdigest()
            or row.get("functional_pass") != passed
            or row.get("functional_score") != score
            or row.get("route") != route
            or row.get("expected_route") != CAPABILITY_TO_ROUTE[str(probe["capability"])]
            or row.get("route_correct") != (route == CAPABILITY_TO_ROUTE[str(probe["capability"])])
            or row.get("collapse") != collapse
            or row.get("source") != source_row
            or row.get("source_passing_regression") != bool(source_row["passed"] and not passed)
            or row.get("generation_error") is not None
            or not isinstance(row.get("latency_seconds"), (int, float))
            or row["latency_seconds"] <= 0
        ):
            raise VerificationError(f"R49 raw row changed: {probe['probe_id']}")
        if live:
            live_output, live_tokens, _, physical = campaign._generate(
                model, tokenizer, prompt, route, int(probe["max_new_tokens"]), device
            )
            if live_output != output or live_tokens != tokens or not all(value == (route,) for value in physical):
                raise VerificationError(f"R49 live replay changed: {probe['probe_id']}")
            live_exact += 1
            if ordinal % 100 == 0:
                print(json.dumps({"live_replayed": ordinal, "exact": live_exact}), flush=True)
        recomputed.append(row)
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
    # Wall time is measured, not a scientific gate; preserve the raw value.
    metrics = {
        "rows": len(rows),
        "functional": functional,
        "source_functional": source_functional,
        "source_passing_regressions": regressions,
        "source_retention": (source_functional - regressions) / source_functional,
        "candidate_minus_source": campaign._bootstrap([row["functional_pass"] for row in rows], [row["source"]["passed"] for row in rows]),
        "collapses": sum(row["collapse"]["collapse_detected"] for row in rows),
        "generation_errors": sum(row["generation_error"] is not None for row in rows),
        "route_correct": sum(row["route_correct"] for row in rows),
        "physical_sparse_rows": sum(row["all_calls_selected_only"] and row["maximum_cakes_called_per_model_invocation"] == 1 for row in rows),
        "by_capability": by_capability,
        "generation_wall_seconds": result["metrics"]["generation_wall_seconds"],
    }
    if metrics != result.get("metrics"):
        raise VerificationError("R49 metrics differ from raw recomputation")
    gates = {
        "router": router_receipt["passed"],
        "matrix": len(rows) == 1_400,
        "functional": functional >= 1_260,
        "per_capability": all(value["functional"] >= 65 for value in by_capability.values()),
        "source_noninferior_point": functional >= source_functional,
        "source_retention": metrics["source_retention"] >= 0.94,
        "zero_collapse": metrics["collapses"] == 0,
        "zero_generation_error": metrics["generation_errors"] == 0,
        "route_exact": metrics["route_correct"] == 1_400,
        "physical_sparse": metrics["physical_sparse_rows"] == 1_400,
        "candidate_unchanged": True,
    }
    if gates != result.get("gates") or not all(gates.values()):
        raise VerificationError("R49 recomputed gate vector failed")
    return {
        "format": "abi-r49-strict-verification/1",
        "verdict": "PASS_R49_STRICT_VERIFICATION",
        "result_sha256": _sha256_file(result_path),
        "evidence_sha256": stored_evidence,
        "raw_rows_recomputed": len(rows),
        "router_search_recomputed": len(search),
        "router_validation_recomputed": len(validation),
        "live_rows_exact": live_exact,
        "functional": functional,
        "source_functional": source_functional,
        "full_abi_moonshot": "OPEN",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, action="append", required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    receipt = verify(args.run_dir.resolve(), args.candidate.resolve(), args.catalog.resolve(), [value.resolve() for value in args.source_bundle], args.layercake_root.resolve(), args.live)
    write_json_once(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
