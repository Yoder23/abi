"""Separate R78 candidate selection from autonomous output realization."""

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

from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER, load_layercake_core
from abi.layercake_host import _sha256_file
from abi.hf_extraction import load_probe_catalog
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once


CANDIDATE_SHA256 = "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e"
CANDIDATE_METADATA_SHA256 = "1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e"
CATALOG_SHA256 = "b89a473da82b66297f0cd9f7796b341f873d1bb3d1b2c7f82e94255b5b61330a"
SCREEN_RESULT_SHA256 = "3322605ccd34166c0a58ba2ad9ea4a8ddbea464c2ad59979fff3a8d51a99826"
SCREEN_RAW_SHA256 = "c8c1494a2a89469986ded391fcc5ff38da5ac3e8bca382f1f56ba12dd5ede29d"
ROWS = 1_400
ROUTE = CAPABILITY_CAKE_ORDER.index("domain_independent_reasoning")


class DiagnosticError(RuntimeError):
    pass


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DiagnosticError(f"required raw evidence is absent: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise DiagnosticError("raw evidence is malformed")
    return rows


@torch.inference_mode()
def run(
    *, candidate: Path, layercake_root: Path, catalog_path: Path,
    screen_result_path: Path, screen_raw_path: Path, output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise DiagnosticError(f"immutable diagnostic output exists: {output}")
    bindings = (
        (candidate / "model.safetensors", CANDIDATE_SHA256),
        (candidate / "metadata.json", CANDIDATE_METADATA_SHA256),
        (catalog_path, CATALOG_SHA256),
        (screen_result_path, SCREEN_RESULT_SHA256),
        (screen_raw_path, SCREEN_RAW_SHA256),
    )
    for path, expected in bindings:
        if not path.is_file() or _sha256_file(path) != expected:
            raise DiagnosticError(f"bound input changed: {path}")
    screen_result = json.loads(screen_result_path.read_text(encoding="utf-8"))
    if screen_result.get("verdict") != "FAIL_R78_HOST_TRANSFER":
        raise DiagnosticError("diagnostic requires the frozen R78 failure")
    screen_rows = {str(row["probe_id"]): row for row in _load_rows(screen_raw_path)}
    probes = list(load_probe_catalog(catalog_path)["probes"])
    if len(probes) != ROWS or len(screen_rows) != ROWS or set(screen_rows) != {str(row["probe_id"]) for row in probes}:
        raise DiagnosticError("diagnostic matrix is incomplete")
    if not torch.cuda.is_available():
        raise DiagnosticError("R78 likelihood diagnostic requires CUDA")
    device = torch.device("cuda")
    model, tokenizer, _ = load_layercake_core(candidate, layercake_root=layercake_root, device=device)
    model.eval()
    route = torch.tensor([ROUTE], dtype=torch.long, device=device)
    rows = []
    started = time.perf_counter()
    physical_forwards = 0
    scored_tokens = 0
    for index, probe in enumerate(probes, 1):
        prompt = str(probe["prompt"]) + "\n"
        prompt_ids = tokenizer.encode(prompt)
        choices = list(probe["conditional_candidates"])
        scores = []
        for choice in choices:
            full_ids = tokenizer.encode(prompt + choice)
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise DiagnosticError("candidate changed LayerCake prompt token prefix")
            candidate_ids = full_ids[len(prompt_ids):]
            if not candidate_ids:
                raise DiagnosticError("candidate has no LayerCake tokens")
            ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            result = model(
                ids,
                prompt_lengths=torch.tensor([len(prompt_ids)], device=device),
                task_routes=route,
                use_cache=False,
            )
            if tuple(model.last_cake_calls) != (ROUTE,) or any(
                tuple(getattr(block, "_abi_last_deep_adapter_routes", ())) != (ROUTE,)
                for block in model.transformer.h
            ):
                raise DiagnosticError("likelihood diagnostic executed an inactive route")
            physical_forwards += 1
            log_probs = torch.log_softmax(result["logits"].float(), dim=-1)
            values = [
                float(log_probs[0, len(prompt_ids) + local - 1, token].item())
                for local, token in enumerate(candidate_ids)
            ]
            if not values or any(not math.isfinite(value) for value in values):
                raise DiagnosticError("non-finite LayerCake candidate score")
            scored_tokens += len(values)
            scores.append({"mean_log_probability": sum(values) / len(values), "token_count": len(values)})
        means = [row["mean_log_probability"] for row in scores]
        selected_index = max(range(3), key=means.__getitem__)
        selected = choices[selected_index]
        expected = str(probe["evaluator"]["value"])
        generation = screen_rows[str(probe["probe_id"])]
        rows.append({
            "probe_id": probe["probe_id"],
            "premise_family": int(str(probe["probe_id"]).split("-")[2][1:]),
            "prompt_sha256": hashlib.sha256(str(probe["prompt"]).encode()).hexdigest(),
            "candidate_codes": choices,
            "candidate_scores": scores,
            "selected_index": selected_index,
            "selected_output": selected,
            "selected_output_sha256": hashlib.sha256(selected.encode()).hexdigest(),
            "selection_passed": selected == expected,
            "generation_passed": bool(generation["candidate_passed"]),
            "generation_output_sha256": generation["candidate_output_sha256"],
            "selection_correct_generation_failed": selected == expected and not generation["candidate_passed"],
        })
        if index % 200 == 0:
            print(json.dumps({"scored": index, "selection_passing": sum(row["selection_passed"] for row in rows), "generation_passing": sum(row["generation_passed"] for row in rows)}), flush=True)
    output.mkdir(parents=True)
    raw_path = output / "candidate_choice_scores.jsonl"
    write_jsonl_once(raw_path, rows)
    family = Counter(str(row["premise_family"]) for row in rows if row["selection_passed"])
    selection_passing = sum(row["selection_passed"] for row in rows)
    generation_passing = sum(row["generation_passed"] for row in rows)
    isolated = sum(row["selection_correct_generation_failed"] for row in rows)
    result = {
        "format": "abi-r78-selection-versus-realization-diagnostic/1",
        "verdict": (
            "SELECTION_TRANSFERRED_REALIZATION_LIMITING"
            if selection_passing >= 1_330 and min(family.values(), default=0) >= 180 and isolated >= 1_200
            else "SELECTION_AND_REALIZATION_BOTH_LIMITING"
        ),
        "candidate_checkpoint_sha256": CANDIDATE_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "screen_result_sha256": SCREEN_RESULT_SHA256,
        "screen_raw_sha256": SCREEN_RAW_SHA256,
        "metrics": {
            "rows": len(rows),
            "selection_passing": selection_passing,
            "selection_pass_rate": selection_passing / len(rows),
            "generation_passing": generation_passing,
            "generation_pass_rate": generation_passing / len(rows),
            "selection_correct_generation_failed": isolated,
            "family_selection_passing": dict(sorted(family.items())),
            "physical_sparse_forwards": physical_forwards,
            "candidate_tokens_scored": scored_tokens,
            "wall_seconds": time.perf_counter() - started,
        },
        "source_teacher_loaded": False,
        "promotion_eligible": False,
        "post_failure_diagnostic": True,
        "full_abi_moonshot": "OPEN",
        "artifacts": {"candidate_choice_scores": {"path": raw_path.name, "sha256": _sha256_file(raw_path), "bytes": raw_path.stat().st_size}},
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--layercake-root", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--screen-result", required=True, type=Path)
    parser.add_argument("--screen-raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(
        candidate=args.candidate.resolve(), layercake_root=args.layercake_root.resolve(),
        catalog_path=args.catalog.resolve(), screen_result_path=args.screen_result.resolve(),
        screen_raw_path=args.screen_raw.resolve(), output=args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
