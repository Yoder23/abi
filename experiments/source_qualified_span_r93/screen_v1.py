"""Prospectively execute frozen R91 on the source-qualified R93 surface."""

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
from experiments.mixture_pointer_r84.screen_host_v1 import _generate as _parent_generate
from experiments.structural_span_r91.core_v1 import load_package


PARENT_SHA256 = "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e"
PARENT_METADATA_SHA256 = "1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e"
ARTIFACT_SHA256 = "292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc"
ROWS = 1_400
CAMPAIGN = "r93"
BINDING_FORMAT = "abi-r93-source-qualified-candidate-binding/1"
SOURCE_VERDICT = "PASS_R93_SOURCE"
SOURCE_PASS_FIELD = "prior_corrected_passed"
FAMILIES = 4
FAMILY_FLOOR = 315
SEED_BASE = 93_000
RESULT_FORMAT = "abi-r93-source-qualified-prospective-screen/1"
PASS_VERDICT = "PASS_R93_BOUNDED_PROSPECTIVE_TRANSFER"
FAIL_VERDICT = "FAIL_R93_PROSPECTIVE_TRANSFER"
SOURCE_QUALITY_FLOOR = 1_330
REQUIRE_SOURCE_POINT_NONINFERIOR = False
SOURCE_NONINFERIOR_CI95_FLOOR = -1.0
CLAIM_BOUNDARY = "Bounded source-qualified prospective two-hop reasoning transfer only."
PACKAGE_LOADER = load_package
EXPECTED_RETRAINED_AFTER_R91 = False


class R93Error(RuntimeError):
    pass


def _load_binding(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.get("binding_sha256")
    unsigned = {key: item for key, item in value.items() if key != "binding_sha256"}
    if value.get("format") != BINDING_FORMAT or claimed != _canonical_sha(unsigned) or value.get("candidate_observations_before_binding") != 0 or value.get("candidate_retrained_after_r91_development") is not EXPECTED_RETRAINED_AFTER_R91:
        raise R93Error("R93 binding is invalid")
    paths = {}
    for name, item in value.get("files", {}).items():
        target = root / str(item.get("path", ""))
        if not target.is_file() or target.stat().st_size != item.get("bytes") or _sha256_file(target) != item.get("sha256"):
            raise R93Error(f"R93 bound file changed: {name}")
        paths[name] = target
    return value, paths


def run(*, binding_path: Path, parent: Path, artifact: Path, layercake_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R93Error(f"immutable R93 output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    binding, files = _load_binding(binding_path, root)
    if _sha256_file(parent / "model.safetensors") != PARENT_SHA256 or _sha256_file(parent / "metadata.json") != PARENT_METADATA_SHA256 or _sha256_file(artifact) != ARTIFACT_SHA256:
        raise R93Error("R93 parent or artifact changed")
    source_result = json.loads(files["source_result"].read_text(encoding="utf-8"))
    unsigned_source = dict(source_result)
    source_claim = unsigned_source.pop("evidence_sha256", None)
    if source_result.get("verdict") != SOURCE_VERDICT or source_claim != binding.get("source_evidence_sha256") or source_claim != evidence_hash(unsigned_source) or not all(source_result.get("gates", {}).values()):
        raise R93Error("R93 source authorization failed")
    probes = list(load_probe_catalog(files["catalog"])["probes"])
    source_rows = {str(row["probe_id"]): row for row in _jsonl(files["source_raw"])}
    if len(probes) != ROWS or len(source_rows) != ROWS or set(source_rows) != {str(row["probe_id"]) for row in probes}:
        raise R93Error("R93 matrix coverage changed")
    for probe in probes:
        source = source_rows[str(probe["probe_id"])]
        if source["prompt_sha256"] != hashlib.sha256(str(probe["prompt"]).encode()).hexdigest() or source["expected_output_sha256"] != hashlib.sha256(str(probe["evaluator"]["value"]).encode()).hexdigest():
            raise R93Error("R93 source row is not catalog bound")
    if not torch.cuda.is_available():
        raise R93Error("R93 requires CUDA")
    device = torch.device("cuda")
    process = psutil.Process()
    model, tokenizer, _ = load_layercake_core(parent, layercake_root=layercake_root, device=device)
    model.eval()
    trained, metadata = PACKAGE_LOADER(files["candidate_checkpoint"].parent, device=device)
    torch.manual_seed(SEED_BASE + 2)
    random_bridge = JointSpanBridge().to(device).eval()
    observed = {str(row["probe_id"]): {} for row in probes}
    parent_started = time.perf_counter()
    for index, probe in enumerate(probes, 1):
        generation = _parent_generate(model, tokenizer, str(probe["prompt"]), int(probe["max_new_tokens"]), device, require_pointer=False)
        passed, score = evaluate_output(generation["output"], probe["evaluator"])
        observed[str(probe["probe_id"])]["parent"] = {**generation, "passed": bool(passed), "score": float(score)}
        if index % 200 == 0:
            print(json.dumps({"system": "parent", "evaluated": index}), flush=True)
    parent_seconds = time.perf_counter() - parent_started
    torch.cuda.reset_peak_memory_stats()
    candidate_started = time.perf_counter()
    for index, probe in enumerate(probes, 1):
        generation = _joint_outputs(model=model, tokenizer=tokenizer, trained=trained, randomized=random_bridge, prompt=str(probe["prompt"]), device=device)
        passed, score = evaluate_output(generation["output"], probe["evaluator"])
        random_passed, _ = evaluate_output(generation["random_output"], probe["evaluator"])
        collapse = _collapse_metrics(generation["token_ids"], generation["output"], tokenizer.encode(str(probe["prompt"]) + "\n", add_special_tokens=False), str(probe["prompt"]))
        observed[str(probe["probe_id"])]["candidate"] = {**generation, "passed": bool(passed), "score": float(score), "random_passed": bool(random_passed), "collapse": collapse}
        if index % 200 == 0:
            print(json.dumps({"system": "candidate", "evaluated": index}), flush=True)
    candidate_seconds = time.perf_counter() - candidate_started
    rows = []
    for probe in probes:
        probe_id = str(probe["probe_id"])
        source = source_rows[probe_id]
        parent_row = observed[probe_id]["parent"]
        candidate = observed[probe_id]["candidate"]
        rows.append({
            "probe_id": probe_id, "premise_family": int(source["premise_family"]),
            "prompt_sha256": source["prompt_sha256"], "source_passed": bool(source[SOURCE_PASS_FIELD]),
            "parent_output": parent_row["output"], "parent_passed": parent_row["passed"],
            "candidate_output": candidate["output"], "candidate_token_ids": candidate["token_ids"],
            "candidate_start": candidate["start"], "candidate_end": candidate["end"],
            "candidate_passed": candidate["passed"], "candidate_collapse": candidate["collapse"],
            "random_output": candidate["random_output"], "random_passed": candidate["random_passed"],
            "candidate_physical_sparse": candidate["physical_sparse"],
            "source_passing_retained": bool(source[SOURCE_PASS_FIELD] and candidate["passed"]),
            "candidate_model_invocations": 1, "candidate_task_cake_invocations": 1,
            "candidate_deep_adapter_invocations": 6, "candidate_joint_span_bridge_invocations": 1,
        })
    output.mkdir(parents=True)
    raw = output / "evaluation.jsonl"
    write_jsonl_once(raw, rows)
    source_passing = sum(row["source_passed"] for row in rows)
    candidate_passing = sum(row["candidate_passed"] for row in rows)
    parent_passing = sum(row["parent_passed"] for row in rows)
    random_passing = sum(row["random_passed"] for row in rows)
    retained = sum(row["source_passing_retained"] for row in rows)
    families = {str(index): {
        "source_passing": sum(row["premise_family"] == index and row["source_passed"] for row in rows),
        "candidate_passing": sum(row["premise_family"] == index and row["candidate_passed"] for row in rows),
        "parent_passing": sum(row["premise_family"] == index and row["parent_passed"] for row in rows),
        "random_passing": sum(row["premise_family"] == index and row["random_passed"] for row in rows),
    } for index in range(FAMILIES)}
    metrics = {
        "rows": len(rows), "source_passing": source_passing, "candidate_passing": candidate_passing,
        "parent_passing": parent_passing, "random_bridge_passing": random_passing,
        "source_passing_retained": retained, "source_retention": retained / source_passing,
        "candidate_collapses": sum(row["candidate_collapse"]["collapse_detected"] for row in rows),
        "candidate_physical_sparse_rows": sum(row["candidate_physical_sparse"] for row in rows),
        "by_family": families,
        "candidate_minus_source": _bootstrap([row["candidate_passed"] for row in rows], [row["source_passed"] for row in rows], SEED_BASE + 1),
        "candidate_minus_parent": _bootstrap([row["candidate_passed"] for row in rows], [row["parent_passed"] for row in rows], SEED_BASE + 2),
        "candidate_minus_random": _bootstrap([row["candidate_passed"] for row in rows], [row["random_passed"] for row in rows], SEED_BASE + 3),
        "parent_generation_seconds": parent_seconds, "candidate_and_random_generation_seconds": candidate_seconds,
        "candidate_peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "process_rss_bytes": int(process.memory_info().rss),
    }
    gates = {
        "matrix": len(rows) == ROWS, "source_quality": source_passing >= SOURCE_QUALITY_FLOOR,
        "candidate_quality": candidate_passing >= 1_330,
        "candidate_family_floor": min(value["candidate_passing"] for value in families.values()) >= FAMILY_FLOOR,
        "source_retention": metrics["source_retention"] >= 0.95,
        "causal_parent_gain": (candidate_passing - parent_passing) / ROWS >= 0.50,
        "random_bridge_fails": random_passing <= 500,
        "trained_beats_random": (candidate_passing - random_passing) / ROWS >= 0.50,
        "zero_collapse": metrics["candidate_collapses"] == 0,
        "physical_sparse": metrics["candidate_physical_sparse_rows"] == ROWS,
        "teacher_absent": metadata["source_boundary"]["teacher_present_at_inference"] is False,
        "artifacts_unchanged": all(_sha256_file(root / item["path"]) == item["sha256"] for item in binding["files"].values()),
    }
    if REQUIRE_SOURCE_POINT_NONINFERIOR:
        gates["candidate_at_least_source_point"] = candidate_passing >= source_passing
        gates["candidate_source_noninferior_bootstrap"] = (
            metrics["candidate_minus_source"]["ci95_low"] >= SOURCE_NONINFERIOR_CI95_FLOOR
        )
    if not all(math.isfinite(float(value)) for value in (parent_seconds, candidate_seconds)):
        raise R93Error("R93 timing is non-finite")
    passed = all(gates.values())
    result = {
        "format": RESULT_FORMAT,
        "campaign": CAMPAIGN,
        "source_pass_field": SOURCE_PASS_FIELD,
        "verdict": PASS_VERDICT if passed else FAIL_VERDICT,
        "binding_sha256": binding["binding_sha256"],
        "candidate_checkpoint_sha256": binding["files"]["candidate_checkpoint"]["sha256"],
        "candidate_metadata_sha256": binding["files"]["candidate_metadata"]["sha256"],
        "parent_checkpoint_sha256": PARENT_SHA256, "imported_artifact_sha256": ARTIFACT_SHA256,
        "catalog_sha256": binding["files"]["catalog"]["sha256"],
        "source_result_sha256": binding["files"]["source_result"]["sha256"],
        "source_raw_sha256": binding["files"]["source_raw"]["sha256"],
        "metrics": metrics, "gates": gates,
        "artifacts": {"evaluation": {"path": raw.name, "bytes": raw.stat().st_size, "sha256": _sha256_file(raw)}},
        "candidate_retrained_or_calibrated": False, "teacher_present_at_inference": False,
        "source_parameters_retained": 0, "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--layercake-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(binding_path=args.binding.resolve(), parent=args.parent.resolve(), artifact=args.artifact.resolve(), layercake_root=args.layercake_root.resolve(), output=args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
