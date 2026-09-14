"""Screen frozen R91 on the disclosed R60 reasoning surface."""

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

from abi.conditional_choice_artifact_v2 import _canonical_sha
from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import load_layercake_core
from abi.layercake_host import _sha256_file
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once
from experiments.generic_prompt_identity_r83.screen_host_v1 import _bootstrap, _jsonl
from experiments.joint_span_r88.core_v1 import JointSpanBridge
from experiments.joint_span_r88.screen_host_v1 import _joint_outputs
from .core_v1 import load_package


CATALOG_SHA256 = "4b0087c9a7fa0e0fd6f607fdbd94fffbdd3cb375f5880a0588c97414583a2ad7"
R60_RAW_SHA256 = "e509484babfab685c91fee3a6165bfea0b2881213fdb7b3a71fde9266e2a25cf"
PARENT_SHA256 = "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e"
PARENT_METADATA_SHA256 = "1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e"
ARTIFACT_SHA256 = "292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc"
CAPABILITY = "domain_independent_reasoning"
ROWS = 100


class R91ScreenError(RuntimeError):
    pass


def _binding(path: Path, candidate: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.get("binding_sha256")
    unsigned = {key: item for key, item in value.items() if key != "binding_sha256"}
    root = Path(__file__).resolve().parents[2]
    implementation = {
        "core_sha256": root / "experiments/structural_span_r91/core_v1.py",
        "trainer_sha256": root / "experiments/structural_span_r91/train_v1.py",
        "screen_sha256": root / "experiments/structural_span_r91/screen_development_v1.py",
        "protocol_sha256": root / "experiments/structural_span_r91/PROTOCOL.md",
    }
    if (
        value.get("format") != "abi-r91-frozen-development-candidate-binding/1"
        or claimed != _canonical_sha(unsigned)
        or value.get("candidate_generation_observations_before_binding") != 0
        or value.get("r60_outputs_used_for_training") != 0
        or value.get("development_catalog_sha256") != CATALOG_SHA256
        or value.get("development_raw_sha256") != R60_RAW_SHA256
        or value.get("candidate_checkpoint_sha256")
        != _sha256_file(candidate / "bridge.safetensors")
        or value.get("candidate_metadata_sha256")
        != _sha256_file(candidate / "metadata.json")
        or any(value.get(name) != _sha256_file(file) for name, file in implementation.items())
    ):
        raise R91ScreenError("R91 candidate binding is stale")
    return value


def run(*, candidate: Path, parent: Path, artifact: Path, binding_path: Path,
        layercake_root: Path, catalog: Path, r60_raw: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R91ScreenError(f"immutable R91 screen exists: {output}")
    for path, digest in (
        (catalog, CATALOG_SHA256), (r60_raw, R60_RAW_SHA256),
        (parent / "model.safetensors", PARENT_SHA256),
        (parent / "metadata.json", PARENT_METADATA_SHA256),
        (artifact, ARTIFACT_SHA256),
    ):
        if not path.is_file() or _sha256_file(path) != digest:
            raise R91ScreenError(f"missing or changed input: {path}")
    binding = _binding(binding_path, candidate)
    probes = [row for row in load_probe_catalog(catalog)["probes"] if row["capability"] == CAPABILITY]
    prior = {str(row["probe_id"]): row for row in _jsonl(r60_raw) if row["capability"] == CAPABILITY}
    if len(probes) != ROWS or len(prior) != ROWS or {str(row["probe_id"]) for row in probes} != set(prior):
        raise R91ScreenError("R91 matrix coverage changed")
    for probe in probes:
        if prior[str(probe["probe_id"])]["prompt_sha256"] != hashlib.sha256(str(probe["prompt"]).encode()).hexdigest():
            raise R91ScreenError("R91 prompt binding changed")
    if not torch.cuda.is_available():
        raise R91ScreenError("R91 requires CUDA")
    device = torch.device("cuda")
    process = psutil.Process()
    model, tokenizer, _ = load_layercake_core(parent, layercake_root=layercake_root, device=device)
    model.eval()
    trained, metadata = load_package(candidate, device=device)
    torch.manual_seed(91_002)
    random_bridge = JointSpanBridge().to(device).eval()
    rows = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for index, probe in enumerate(probes, 1):
        generated = _joint_outputs(
            model=model, tokenizer=tokenizer, trained=trained, randomized=random_bridge,
            prompt=str(probe["prompt"]), device=device,
        )
        passed, score = evaluate_output(generated["output"], probe["evaluator"])
        random_passed, random_score = evaluate_output(generated["random_output"], probe["evaluator"])
        prompt_ids = tokenizer.encode(str(probe["prompt"]) + "\n", add_special_tokens=False)
        collapse = _collapse_metrics(
            generated["token_ids"], generated["output"], prompt_ids, str(probe["prompt"])
        )
        old = prior[str(probe["probe_id"])]
        rows.append({
            "probe_id": probe["probe_id"], "prompt_sha256": old["prompt_sha256"],
            "source_passed": bool(old["source"]["passed"]),
            "r60_endpoint_passed": bool(old["candidate_functional_pass"]),
            "candidate_output": generated["output"], "candidate_token_ids": generated["token_ids"],
            "candidate_start": generated["start"], "candidate_end": generated["end"],
            "candidate_passed": bool(passed), "candidate_score": float(score),
            "candidate_collapse": collapse, "candidate_physical_sparse": bool(generated["physical_sparse"]),
            "random_output": generated["random_output"], "random_passed": bool(random_passed),
            "random_score": float(random_score), "teacher_called": False,
        })
        if index % 20 == 0:
            print(json.dumps({"evaluated": index}), flush=True)
    elapsed = time.perf_counter() - started
    output.mkdir(parents=True)
    raw = output / "evaluation.jsonl"
    write_jsonl_once(raw, rows)
    candidate_passes = sum(row["candidate_passed"] for row in rows)
    source_passes = sum(row["source_passed"] for row in rows)
    old_passes = sum(row["r60_endpoint_passed"] for row in rows)
    random_passes = sum(row["random_passed"] for row in rows)
    retained = sum(row["source_passed"] and row["candidate_passed"] for row in rows)
    collapses = sum(row["candidate_collapse"]["collapse_detected"] for row in rows)
    sparse = sum(row["candidate_physical_sparse"] for row in rows)
    metrics = {
        "rows": len(rows), "candidate_passing": candidate_passes,
        "source_passing": source_passes, "r60_endpoint_passing": old_passes,
        "random_bridge_passing": random_passes, "source_passing_retained": retained,
        "source_retention": retained / source_passes if source_passes else 1.0,
        "candidate_collapses": collapses, "candidate_physical_sparse_rows": sparse,
        "candidate_minus_r60": _bootstrap(
            [row["candidate_passed"] for row in rows],
            [row["r60_endpoint_passed"] for row in rows], 91_001,
        ),
        "candidate_minus_random": _bootstrap(
            [row["candidate_passed"] for row in rows],
            [row["random_passed"] for row in rows], 91_002,
        ),
        "generation_seconds": elapsed,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "process_rss_bytes": int(process.memory_info().rss),
    }
    gates = {
        "matrix": len(rows) == ROWS, "candidate_quality": candidate_passes >= 95,
        "source_retention": metrics["source_retention"] >= 0.95,
        "r60_gain": candidate_passes - old_passes >= 50,
        "random_bridge_fails": random_passes <= 35,
        "zero_collapse": collapses == 0, "physical_sparse": sparse == ROWS,
        "teacher_absent": metadata["source_boundary"]["teacher_present_at_inference"] is False,
        "artifacts_unchanged": all(_sha256_file(path) == digest for path, digest in (
            (candidate / "bridge.safetensors", binding["candidate_checkpoint_sha256"]),
            (candidate / "metadata.json", binding["candidate_metadata_sha256"]),
            (parent / "model.safetensors", PARENT_SHA256),
            (parent / "metadata.json", PARENT_METADATA_SHA256), (artifact, ARTIFACT_SHA256),
            (catalog, CATALOG_SHA256), (r60_raw, R60_RAW_SHA256),
        )),
    }
    if not math.isfinite(elapsed):
        raise R91ScreenError("R91 runtime accounting is non-finite")
    passed = all(gates.values())
    result = {
        "format": "abi-r91-structure-invariant-development-screen/1",
        "verdict": "PASS_R91_DEVELOPMENT" if passed else "FAIL_R91_DEVELOPMENT",
        "candidate_checkpoint_sha256": binding["candidate_checkpoint_sha256"],
        "candidate_metadata_sha256": binding["candidate_metadata_sha256"],
        "candidate_binding_sha256": binding["binding_sha256"],
        "parent_checkpoint_sha256": PARENT_SHA256, "imported_artifact_sha256": ARTIFACT_SHA256,
        "catalog_sha256": CATALOG_SHA256, "r60_raw_sha256": R60_RAW_SHA256,
        "metrics": metrics, "gates": gates,
        "artifacts": {"evaluation": {"path": raw.name, "bytes": raw.stat().st_size, "sha256": _sha256_file(raw)}},
        "candidate_retrained_or_calibrated_during_screen": False,
        "teacher_present_at_inference": False, "source_parameters_retained": 0,
        "promotion_eligible": False, "prospective_test_authorized": passed,
        "full_abi_moonshot": "OPEN",
        "claim_boundary": "Disclosed R60 reasoning development screen only.",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "parent", "artifact", "binding", "layercake-root", "catalog", "r60-raw", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    run(
        candidate=args.candidate.resolve(), parent=args.parent.resolve(),
        artifact=args.artifact.resolve(), binding_path=args.binding.resolve(),
        layercake_root=args.layercake_root.resolve(), catalog=args.catalog.resolve(),
        r60_raw=args.r60_raw.resolve(), output=args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
