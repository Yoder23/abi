"""Run the frozen R88 bridge on the disclosed R60 reasoning surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import load_layercake_core
from abi.layercake_host import _sha256_file
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    write_json_once,
    write_jsonl_once,
)
from experiments.generic_prompt_identity_r83.screen_host_v1 import _bootstrap, _jsonl
from experiments.joint_span_r88.core_v1 import JointSpanBridge, load_joint_span_package
from experiments.joint_span_r88.screen_host_v1 import _joint_outputs


CATALOG_SHA256 = "4b0087c9a7fa0e0fd6f607fdbd94fffbdd3cb375f5880a0588c97414583a2ad7"
R60_RAW_SHA256 = "e509484babfab685c91fee3a6165bfea0b2881213fdb7b3a71fde9266e2a25cf"
CANDIDATE_SHA256 = "15c28bdceea49e61e771092ce74be1cc30233e8bef53d87171d36f5a2487372f"
CANDIDATE_METADATA_SHA256 = "d98c53ff713ead136b9a659743553b40fed2fa709d0127b93550f8fef7bfaabe"
PARENT_SHA256 = "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e"
PARENT_METADATA_SHA256 = "1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e"
ARTIFACT_SHA256 = "292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc"
CAPABILITY = "domain_independent_reasoning"
ROWS = 100


class R90Error(RuntimeError):
    pass


def _require_hash(path: Path, digest: str) -> None:
    if not path.is_file() or _sha256_file(path) != digest:
        raise R90Error(f"missing or changed frozen input: {path}")


def run(
    *,
    candidate: Path,
    parent: Path,
    artifact: Path,
    layercake_root: Path,
    catalog: Path,
    r60_raw: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise R90Error(f"immutable R90 output exists: {output}")
    for path, digest in (
        (catalog, CATALOG_SHA256),
        (r60_raw, R60_RAW_SHA256),
        (candidate / "bridge.safetensors", CANDIDATE_SHA256),
        (candidate / "metadata.json", CANDIDATE_METADATA_SHA256),
        (parent / "model.safetensors", PARENT_SHA256),
        (parent / "metadata.json", PARENT_METADATA_SHA256),
        (artifact, ARTIFACT_SHA256),
    ):
        _require_hash(path, digest)
    probes = [
        row for row in load_probe_catalog(catalog)["probes"]
        if row["capability"] == CAPABILITY
    ]
    r60 = {
        str(row["probe_id"]): row for row in _jsonl(r60_raw)
        if row["capability"] == CAPABILITY
    }
    if (
        len(probes) != ROWS
        or len(r60) != ROWS
        or {str(row["probe_id"]) for row in probes} != set(r60)
    ):
        raise R90Error("R90 reasoning matrix coverage changed")
    for probe in probes:
        prior = r60[str(probe["probe_id"])]
        if prior.get("prompt_sha256") != hashlib.sha256(
            str(probe["prompt"]).encode()
        ).hexdigest():
            raise R90Error("R60 row is not bound to the catalog prompt")
    if not torch.cuda.is_available():
        raise R90Error("R90 requires CUDA")
    device = torch.device("cuda")
    process = psutil.Process()
    model, tokenizer, _ = load_layercake_core(
        parent, layercake_root=layercake_root, device=device
    )
    model.eval()
    trained, metadata = load_joint_span_package(candidate, device=device)
    torch.manual_seed(90_002)
    randomized = JointSpanBridge().to(device).eval()
    rows = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for index, probe in enumerate(probes, 1):
        generated = _joint_outputs(
            model=model,
            tokenizer=tokenizer,
            trained=trained,
            randomized=randomized,
            prompt=str(probe["prompt"]),
            device=device,
        )
        passed, score = evaluate_output(generated["output"], probe["evaluator"])
        random_passed, random_score = evaluate_output(
            generated["random_output"], probe["evaluator"]
        )
        prompt_ids = tokenizer.encode(
            str(probe["prompt"]) + "\n", add_special_tokens=False
        )
        collapse = _collapse_metrics(
            generated["token_ids"], generated["output"], prompt_ids,
            str(probe["prompt"]),
        )
        prior = r60[str(probe["probe_id"])]
        rows.append({
            "probe_id": probe["probe_id"],
            "prompt_sha256": prior["prompt_sha256"],
            "source_passed": bool(prior["source"]["passed"]),
            "r60_endpoint_passed": bool(prior["candidate_functional_pass"]),
            "candidate_output": generated["output"],
            "candidate_token_ids": generated["token_ids"],
            "candidate_start": generated["start"],
            "candidate_end": generated["end"],
            "candidate_passed": bool(passed),
            "candidate_score": float(score),
            "candidate_collapse": collapse,
            "candidate_physical_sparse": bool(generated["physical_sparse"]),
            "random_output": generated["random_output"],
            "random_start": generated["random_start"],
            "random_end": generated["random_end"],
            "random_passed": bool(random_passed),
            "random_score": float(random_score),
            "teacher_called": False,
        })
        if index % 20 == 0:
            print(json.dumps({"evaluated": index}), flush=True)
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True)
    raw = output / "evaluation.jsonl"
    write_jsonl_once(raw, rows)
    candidate_passes = sum(row["candidate_passed"] for row in rows)
    source_passes = sum(row["source_passed"] for row in rows)
    r60_passes = sum(row["r60_endpoint_passed"] for row in rows)
    random_passes = sum(row["random_passed"] for row in rows)
    retained = sum(
        row["source_passed"] and row["candidate_passed"] for row in rows
    )
    collapses = sum(
        row["candidate_collapse"]["collapse_detected"] for row in rows
    )
    sparse = sum(row["candidate_physical_sparse"] for row in rows)
    metrics = {
        "rows": len(rows),
        "candidate_passing": candidate_passes,
        "source_passing": source_passes,
        "r60_endpoint_passing": r60_passes,
        "random_bridge_passing": random_passes,
        "source_passing_retained": retained,
        "source_retention": retained / source_passes if source_passes else 1.0,
        "candidate_collapses": collapses,
        "candidate_physical_sparse_rows": sparse,
        "candidate_minus_r60": _bootstrap(
            [row["candidate_passed"] for row in rows],
            [row["r60_endpoint_passed"] for row in rows], 90_001,
        ),
        "candidate_minus_random": _bootstrap(
            [row["candidate_passed"] for row in rows],
            [row["random_passed"] for row in rows], 90_002,
        ),
        "generation_seconds": elapsed,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "process_rss_bytes": int(process.memory_info().rss),
    }
    gates = {
        "matrix": len(rows) == ROWS,
        "candidate_quality": candidate_passes >= 95,
        "source_retention": metrics["source_retention"] >= 0.95,
        "r60_gain": candidate_passes - r60_passes >= 50,
        "random_bridge_fails": random_passes <= 35,
        "zero_collapse": collapses == 0,
        "physical_sparse": sparse == ROWS,
        "teacher_absent": metadata["source_boundary"]["teacher_present_at_inference"] is False,
        "artifacts_unchanged": all(
            _sha256_file(path) == digest for path, digest in (
                (candidate / "bridge.safetensors", CANDIDATE_SHA256),
                (candidate / "metadata.json", CANDIDATE_METADATA_SHA256),
                (parent / "model.safetensors", PARENT_SHA256),
                (parent / "metadata.json", PARENT_METADATA_SHA256),
                (artifact, ARTIFACT_SHA256),
                (catalog, CATALOG_SHA256),
                (r60_raw, R60_RAW_SHA256),
            )
        ),
    }
    if not math.isfinite(elapsed):
        raise R90Error("runtime accounting is not finite")
    passed = all(gates.values())
    result = {
        "format": "abi-r90-broad-surface-joint-span-diagnostic/1",
        "verdict": "PASS_R90_DISCLOSED_DIAGNOSTIC" if passed else "FAIL_R90_DISCLOSED_DIAGNOSTIC",
        "candidate_checkpoint_sha256": CANDIDATE_SHA256,
        "candidate_metadata_sha256": CANDIDATE_METADATA_SHA256,
        "parent_checkpoint_sha256": PARENT_SHA256,
        "imported_artifact_sha256": ARTIFACT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "r60_raw_sha256": R60_RAW_SHA256,
        "metrics": metrics,
        "gates": gates,
        "artifacts": {"evaluation": {
            "path": raw.name,
            "bytes": raw.stat().st_size,
            "sha256": _sha256_file(raw),
        }},
        "candidate_retrained_or_calibrated": False,
        "teacher_present_at_inference": False,
        "source_parameters_retained": 0,
        "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
        "claim_boundary": "Disclosed R60 reasoning-surface diagnostic only.",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate", "parent", "artifact", "layercake-root", "catalog",
        "r60-raw", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    run(
        candidate=args.candidate.resolve(),
        parent=args.parent.resolve(),
        artifact=args.artifact.resolve(),
        layercake_root=args.layercake_root.resolve(),
        catalog=args.catalog.resolve(),
        r60_raw=args.r60_raw.resolve(),
        output=args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
