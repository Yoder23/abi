"""Execute the two preregistered R62 prompt-only counterfactuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
WRAPPERS = (
    "Please solve this fresh request and follow every constraint: ",
    "For this turn, carry out the following instruction exactly: ",
    "Read the request below, then respond accordingly: ",
    "Complete this user request carefully and directly: ",
)
CONDITIONS = ("new_wrapper_only", "body_only")


class AttributionError(RuntimeError):
    pass


def _prompts(original: str) -> dict[str, str]:
    without_ordinal, count = re.subn(r"^Evaluation item [0-9]+\. ", "", original)
    if count != 1:
        raise AttributionError("R62 ordinal boundary changed")
    matches = [prefix for prefix in WRAPPERS if without_ordinal.startswith(prefix)]
    if len(matches) != 1:
        raise AttributionError("R62 wrapper boundary changed")
    body = without_ordinal[len(matches[0]) :]
    if not body:
        raise AttributionError("R62 task body is empty")
    return {"new_wrapper_only": without_ordinal, "body_only": body}


def run(
    candidate: Path,
    catalog_path: Path,
    r60_dir: Path,
    layercake_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise AttributionError(f"immutable R62 output exists: {output}")
    frozen = (
        (candidate / "model.safetensors", r55.CANDIDATE_SHA256),
        (candidate / "metadata.json", r55.METADATA_SHA256),
        (catalog_path, CATALOG_SHA256),
        (r60_dir / "result.json", R60_RESULT_SHA256),
        (r60_dir / "evaluation.jsonl", R60_RAW_SHA256),
    )
    for path, digest in frozen:
        if not path.is_file() or sha256_file(path) != digest:
            raise AttributionError(f"R62 frozen input changed: {path}")
    r55._preflight(candidate)
    if not torch.cuda.is_available():
        raise AttributionError("R62 requires CUDA")
    probes = list(load_probe_catalog(catalog_path)["probes"])
    if len(probes) != 1_400:
        raise AttributionError("R62 matrix changed")
    for probe in probes:
        _prompts(str(probe["prompt"]))

    device = torch.device("cuda")
    model, tokenizer, _ = load_layercake_core(
        candidate, layercake_root=layercake_root, device=device
    )
    model.eval()
    rows = []
    telemetry = {
        condition: {
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
        for condition in CONDITIONS
    }
    started = time.perf_counter()
    for condition in CONDITIONS:
        for ordinal, probe in enumerate(probes, 1):
            inference_prompt = _prompts(str(probe["prompt"]))[condition]
            route = fit_router.CAPABILITY_TO_INDEX[str(probe["capability"])]
            output_text, token_ids, latency, physical = r59._generate(
                model,
                tokenizer,
                inference_prompt,
                route,
                int(probe["max_new_tokens"]),
                device,
                telemetry=telemetry[condition],
            )
            if not physical or not all(calls == (route,) for calls in physical):
                raise AttributionError(f"R62 sparse execution changed: {probe['probe_id']}")
            passed, score = evaluate_output(output_text, probe["evaluator"])
            collapse = _collapse_metrics(
                token_ids,
                output_text,
                tokenizer.encode(inference_prompt + "\n"),
                inference_prompt,
            )
            rows.append(
                {
                    "condition": condition,
                    "probe_id": probe["probe_id"],
                    "capability": probe["capability"],
                    "original_prompt_sha256": hashlib.sha256(
                        str(probe["prompt"]).encode("utf-8")
                    ).hexdigest(),
                    "inference_prompt": inference_prompt,
                    "inference_prompt_sha256": hashlib.sha256(
                        inference_prompt.encode("utf-8")
                    ).hexdigest(),
                    "forced_route": route,
                    "output": output_text,
                    "output_sha256": hashlib.sha256(
                        output_text.encode("utf-8")
                    ).hexdigest(),
                    "output_token_ids": token_ids,
                    "functional_pass": bool(passed),
                    "functional_score": float(score),
                    "final_collapse": collapse,
                    "latency_seconds": latency,
                    "physical_sparse": max(map(len, physical)) == 1,
                }
            )
            if ordinal % 100 == 0:
                current = [row for row in rows if row["condition"] == condition]
                print(
                    json.dumps(
                        {
                            "condition": condition,
                            "evaluated": ordinal,
                            "functional": sum(row["functional_pass"] for row in current),
                            "collapses": sum(
                                row["final_collapse"]["collapse_detected"] for row in current
                            ),
                        }
                    ),
                    flush=True,
                )

    output.mkdir(parents=True)
    raw_path = output / "evaluation.jsonl"
    write_jsonl_once(raw_path, rows)
    summaries = {}
    for condition in CONDITIONS:
        values = [row for row in rows if row["condition"] == condition]
        summaries[condition] = {
            "rows": len(values),
            "functional": sum(row["functional_pass"] for row in values),
            "final_collapses": sum(
                row["final_collapse"]["collapse_detected"] for row in values
            ),
            "physical_sparse_rows": sum(row["physical_sparse"] for row in values),
            "by_capability": {
                capability: sum(
                    row["capability"] == capability and row["functional_pass"]
                    for row in values
                )
                for capability in sorted(Counter(row["capability"] for row in values))
            },
        }
    result = {
        "format": "abi-r62-prompt-invariance-attribution/1",
        "verdict": "PASS_R62_PROMPT_EFFECT_QUANTIFIED",
        "r60_baseline_functional": 813,
        "summaries": summaries,
        "telemetry": telemetry,
        "artifacts": {
            "evaluation": {
                "path": raw_path.name,
                "sha256": sha256_file(raw_path),
                "bytes": raw_path.stat().st_size,
            }
        },
        "wall_seconds": time.perf_counter() - started,
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

