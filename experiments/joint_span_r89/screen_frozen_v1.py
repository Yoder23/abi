"""Live R89 screen of the unchanged, already frozen R88 package."""

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
from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER, load_layercake_core
from abi.layercake_host import _sha256_file
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    write_json_once,
    write_jsonl_once,
)
from experiments.generic_prompt_identity_r83.screen_host_v1 import _bootstrap, _jsonl
from experiments.joint_span_r88.core_v1 import JointSpanBridge, load_joint_span_package
from experiments.joint_span_r88.screen_host_v1 import (
    _binding as _candidate_binding,
    _joint_outputs,
    _preflight as _candidate_preflight,
)
from experiments.mixture_pointer_r84.screen_host_v1 import _generate as _parent_generate


CATALOG_SHA256 = "e21bf7e3a2f3c72d5b82307009fb05c3e1e0623208d9023a13baeaf1e265a118"
SOURCE_RESULT_SHA256 = "8384d590e5d66a12fdab938f776b8fa0fa5cf091fe40778b17a5e0d20b47fdbb"
SOURCE_RAW_SHA256 = "70850638029b6e238a25ba63543ad8111b586fc0a69926863fdf7831b0628994"
SOURCE_EVIDENCE_SHA256 = "a1e7c2c3887bb3cf900d010a572955392a7f51a4a68988541f422b4a81731e48"
CANDIDATE_SHA256 = "15c28bdceea49e61e771092ce74be1cc30233e8bef53d87171d36f5a2487372f"
CANDIDATE_METADATA_SHA256 = "d98c53ff713ead136b9a659743553b40fed2fa709d0127b93550f8fef7bfaabe"
PARENT_SHA256 = "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e"
ROWS = 1_400
ROUTE = CAPABILITY_CAKE_ORDER.index("domain_independent_reasoning")


class HoldoutError(RuntimeError):
    pass


def _holdout_binding(path: Path, files: dict[str, Path]) -> dict[str, Any]:
    if not path.is_file():
        raise HoldoutError("R89 holdout binding is absent")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.get("binding_sha256")
    unsigned = {key: item for key, item in value.items() if key != "binding_sha256"}
    bound = value.get("files", {})
    if (
        value.get("format") != "abi-r89-frozen-package-holdout-binding/1"
        or value.get("holdout_observations_before_binding") != 0
        or _canonical_sha(unsigned) != expected
        or set(bound) != set(files)
    ):
        raise HoldoutError("R89 holdout binding structure changed")
    for name, file in files.items():
        item = bound.get(name, {})
        if (
            not file.is_file()
            or item.get("sha256") != _sha256_file(file)
            or item.get("bytes") != file.stat().st_size
        ):
            raise HoldoutError(f"R89 bound file changed: {name}")
    return value


def run(
    *,
    candidate: Path,
    parent: Path,
    candidate_binding_path: Path,
    candidate_screen_path: Path,
    candidate_protocol_path: Path,
    holdout_binding_path: Path,
    holdout_screen_path: Path,
    holdout_protocol_path: Path,
    layercake_root: Path,
    catalog_path: Path,
    source_result_path: Path,
    source_raw_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise HoldoutError(f"immutable R89 screen exists: {output}")
    files = {
        "candidate_checkpoint": candidate / "bridge.safetensors",
        "candidate_metadata": candidate / "metadata.json",
        "candidate_binding": candidate_binding_path,
        "candidate_screen": candidate_screen_path,
        "candidate_protocol": candidate_protocol_path,
        "candidate_core": candidate_screen_path.with_name("core_v1.py"),
        "candidate_trainer": candidate_screen_path.with_name("train_candidate_v1.py"),
        "holdout_screen": holdout_screen_path,
        "holdout_protocol": holdout_protocol_path,
        "catalog": catalog_path,
        "source_result": source_result_path,
        "source_raw": source_raw_path,
    }
    holdout = _holdout_binding(holdout_binding_path, files)
    if (
        files["candidate_checkpoint"].is_file() is not True
        or _sha256_file(files["candidate_checkpoint"]) != CANDIDATE_SHA256
        or _sha256_file(files["candidate_metadata"]) != CANDIDATE_METADATA_SHA256
        or _sha256_file(catalog_path) != CATALOG_SHA256
        or _sha256_file(source_result_path) != SOURCE_RESULT_SHA256
        or _sha256_file(source_raw_path) != SOURCE_RAW_SHA256
    ):
        raise HoldoutError("R89 frozen package or evidence identity changed")
    candidate_binding = _candidate_binding(
        candidate_binding_path,
        screen=candidate_screen_path,
        protocol=candidate_protocol_path,
    )
    metadata = _candidate_preflight(candidate, parent, candidate_binding)
    if candidate_binding["candidate_checkpoint_sha256"] != CANDIDATE_SHA256:
        raise HoldoutError("R89 did not receive the frozen R88 checkpoint")
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    unsigned_source = dict(source_result)
    unsigned_source.pop("evidence_sha256", None)
    if (
        source_result.get("verdict") != "PASS_R89_SOURCE"
        or not all(source_result.get("gates", {}).values())
        or source_result.get("evidence_sha256") != SOURCE_EVIDENCE_SHA256
        or evidence_hash(unsigned_source) != SOURCE_EVIDENCE_SHA256
        or source_result.get("metrics", {}).get("prior_corrected_passing", 0) < 1_330
    ):
        raise HoldoutError("R89 source result is invalid or unrecomputable")
    probes = list(load_probe_catalog(catalog_path)["probes"])
    source_rows = {str(row["probe_id"]): row for row in _jsonl(source_raw_path)}
    if (
        len(probes) != ROWS
        or len(source_rows) != ROWS
        or set(source_rows) != {str(row["probe_id"]) for row in probes}
    ):
        raise HoldoutError("R89 source row coverage changed")
    for probe in probes:
        source = source_rows[str(probe["probe_id"])]
        if (
            source.get("prompt_sha256")
            != hashlib.sha256(str(probe["prompt"]).encode()).hexdigest()
            or source.get("expected_output_sha256")
            != hashlib.sha256(str(probe["evaluator"]["value"]).encode()).hexdigest()
        ):
            raise HoldoutError("R89 raw source row is not catalog-bound")
    if not torch.cuda.is_available():
        raise HoldoutError("R89 live holdout requires CUDA")
    device = torch.device("cuda")
    process = psutil.Process()
    model, tokenizer, _ = load_layercake_core(
        parent, layercake_root=layercake_root, device=device
    )
    model.eval()
    trained, _ = load_joint_span_package(candidate, device=device)
    torch.manual_seed(89_002)
    randomized = JointSpanBridge().to(device).eval()
    observed: dict[str, dict[str, Any]] = {
        str(probe["probe_id"]): {} for probe in probes
    }

    parent_started = time.perf_counter()
    for index, probe in enumerate(probes, 1):
        generated = _parent_generate(
            model, tokenizer, str(probe["prompt"]),
            int(probe["max_new_tokens"]), device, require_pointer=False,
        )
        passed, score = evaluate_output(generated["output"], probe["evaluator"])
        observed[str(probe["probe_id"])]["parent"] = {
            **generated, "passed": bool(passed), "score": float(score)
        }
        if index % 200 == 0:
            print(json.dumps({"system": "parent", "evaluated": index}), flush=True)
    parent_seconds = time.perf_counter() - parent_started

    torch.cuda.reset_peak_memory_stats()
    candidate_started = time.perf_counter()
    for index, probe in enumerate(probes, 1):
        generated = _joint_outputs(
            model=model, tokenizer=tokenizer, trained=trained,
            randomized=randomized, prompt=str(probe["prompt"]), device=device,
        )
        passed, score = evaluate_output(generated["output"], probe["evaluator"])
        random_passed, _ = evaluate_output(generated["random_output"], probe["evaluator"])
        collapse = _collapse_metrics(
            generated["token_ids"], generated["output"],
            tokenizer.encode(str(probe["prompt"]) + "\n", add_special_tokens=False),
            str(probe["prompt"]),
        )
        observed[str(probe["probe_id"])]["candidate"] = {
            **generated, "passed": bool(passed), "score": float(score),
            "random_passed": bool(random_passed), "collapse": collapse,
        }
        if index % 200 == 0:
            print(json.dumps({"system": "candidate", "evaluated": index}), flush=True)
    candidate_seconds = time.perf_counter() - candidate_started

    rows = []
    for probe in probes:
        probe_id = str(probe["probe_id"])
        source = source_rows[probe_id]
        parent_row = observed[probe_id]["parent"]
        candidate_row = observed[probe_id]["candidate"]
        rows.append({
            "probe_id": probe_id,
            "premise_family": int(source["premise_family"]),
            "prompt_sha256": source["prompt_sha256"],
            "source_passed": bool(source["prior_corrected_passed"]),
            "parent_output": parent_row["output"],
            "parent_passed": parent_row["passed"],
            "candidate_output": candidate_row["output"],
            "candidate_token_ids": candidate_row["token_ids"],
            "candidate_start": candidate_row["start"],
            "candidate_end": candidate_row["end"],
            "candidate_passed": candidate_row["passed"],
            "candidate_collapse": candidate_row["collapse"],
            "random_output": candidate_row["random_output"],
            "random_passed": candidate_row["random_passed"],
            "candidate_physical_sparse": candidate_row["physical_sparse"],
            "source_passing_retained": bool(
                source["prior_corrected_passed"] and candidate_row["passed"]
            ),
        })
    output.mkdir(parents=True)
    raw_path = output / "evaluation.jsonl"
    write_jsonl_once(raw_path, rows)
    source_passing = sum(row["source_passed"] for row in rows)
    candidate_passing = sum(row["candidate_passed"] for row in rows)
    parent_passing = sum(row["parent_passed"] for row in rows)
    random_passing = sum(row["random_passed"] for row in rows)
    retained = sum(row["source_passing_retained"] for row in rows)
    family = {
        str(index): {
            "source_passing": sum(row["premise_family"] == index and row["source_passed"] for row in rows),
            "candidate_passing": sum(row["premise_family"] == index and row["candidate_passed"] for row in rows),
            "parent_passing": sum(row["premise_family"] == index and row["parent_passed"] for row in rows),
            "random_passing": sum(row["premise_family"] == index and row["random_passed"] for row in rows),
        }
        for index in range(7)
    }
    metrics = {
        "rows": ROWS,
        "source_passing": source_passing,
        "candidate_passing": candidate_passing,
        "parent_passing": parent_passing,
        "random_bridge_passing": random_passing,
        "source_passing_retained": retained,
        "source_retention": retained / source_passing,
        "candidate_collapses": sum(row["candidate_collapse"]["collapse_detected"] for row in rows),
        "candidate_physical_sparse_rows": sum(row["candidate_physical_sparse"] for row in rows),
        "by_family": family,
        "candidate_minus_source": _bootstrap(
            [row["candidate_passed"] for row in rows],
            [row["source_passed"] for row in rows], 89_001,
        ),
        "candidate_minus_parent": _bootstrap(
            [row["candidate_passed"] for row in rows],
            [row["parent_passed"] for row in rows], 89_002,
        ),
        "candidate_minus_random": _bootstrap(
            [row["candidate_passed"] for row in rows],
            [row["random_passed"] for row in rows], 89_003,
        ),
        "parent_generation_seconds": parent_seconds,
        "candidate_and_random_generation_seconds": candidate_seconds,
        "candidate_peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "process_rss_bytes": int(process.memory_info().rss),
    }
    gates = {
        "matrix": len(rows) == ROWS,
        "source_quality": source_passing >= 1_330,
        "candidate_quality": candidate_passing >= 1_330,
        "candidate_family_floor": min(value["candidate_passing"] for value in family.values()) >= 180,
        "source_retention": metrics["source_retention"] >= 0.95,
        "causal_parent_gain": (candidate_passing - parent_passing) / ROWS >= 0.50,
        "random_bridge_fails": random_passing <= 500,
        "trained_beats_random": (candidate_passing - random_passing) / ROWS >= 0.50,
        "zero_collapse": metrics["candidate_collapses"] == 0,
        "physical_sparse": metrics["candidate_physical_sparse_rows"] == ROWS,
        "teacher_absent": metadata["source_boundary"]["teacher_present_at_inference"] is False,
        "artifacts_unchanged": all(
            _sha256_file(path) == digest for path, digest in (
                (candidate / "bridge.safetensors", CANDIDATE_SHA256),
                (candidate / "metadata.json", CANDIDATE_METADATA_SHA256),
                (parent / "model.safetensors", PARENT_SHA256),
                (catalog_path, CATALOG_SHA256),
                (source_result_path, SOURCE_RESULT_SHA256),
                (source_raw_path, SOURCE_RAW_SHA256),
            )
        ),
    }
    if not all(math.isfinite(float(value)) for value in (parent_seconds, candidate_seconds)):
        raise HoldoutError("R89 runtime accounting is non-finite")
    passed = all(gates.values())
    result = {
        "format": "abi-r89-frozen-r88-package-holdout-screen/1",
        "verdict": "PASS_R89_FROZEN_PACKAGE_HOLDOUT" if passed else "FAIL_R89_FROZEN_PACKAGE_HOLDOUT",
        "candidate_checkpoint_sha256": CANDIDATE_SHA256,
        "candidate_metadata_sha256": CANDIDATE_METADATA_SHA256,
        "candidate_binding_sha256": candidate_binding["binding_sha256"],
        "holdout_binding_sha256": holdout["binding_sha256"],
        "parent_checkpoint_sha256": PARENT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_raw_sha256": SOURCE_RAW_SHA256,
        "metrics": metrics,
        "gates": gates,
        "artifacts": {"evaluation": {
            "path": raw_path.name, "sha256": _sha256_file(raw_path),
            "bytes": raw_path.stat().st_size,
        }},
        "candidate_retrained_or_calibrated": False,
        "teacher_present_at_inference": False,
        "source_parameters_retained": 0,
        "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
        "claim_boundary": "Frozen R88 package holdout under new seven-digit nonce identities only.",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate", "parent", "candidate-binding", "candidate-screen",
        "candidate-protocol", "holdout-binding", "holdout-screen",
        "holdout-protocol", "layercake-root", "catalog", "source-result",
        "source-raw", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    run(
        candidate=args.candidate.resolve(), parent=args.parent.resolve(),
        candidate_binding_path=args.candidate_binding.resolve(),
        candidate_screen_path=args.candidate_screen.resolve(),
        candidate_protocol_path=args.candidate_protocol.resolve(),
        holdout_binding_path=args.holdout_binding.resolve(),
        holdout_screen_path=args.holdout_screen.resolve(),
        holdout_protocol_path=args.holdout_protocol.resolve(),
        layercake_root=args.layercake_root.resolve(), catalog_path=args.catalog.resolve(),
        source_result_path=args.source_result.resolve(), source_raw_path=args.source_raw.resolve(),
        output=args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
