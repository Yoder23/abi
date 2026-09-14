"""Live prospective R88 quality, causality, and sparse-execution screen."""

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
from experiments.generic_prompt_identity_r83.screen_host_v1 import (
    _bootstrap,
    _jsonl,
)
from experiments.joint_span_r88.core_v1 import (
    JointSpanBridge,
    load_joint_span_package,
    realize_joint_span,
    validate_metadata,
)
from experiments.mixture_pointer_r84.screen_host_v1 import _generate as _parent_generate


CATALOG_SHA256 = "5dba762eec849f2e6531f5f1283417d38e9470a9b37ca69cf3576bb25286a177"
SOURCE_RESULT_SHA256 = "facac7a81732805ad7c2dece8160ce22e74532805d5b5da4fe2f8a8aec1baf4e"
SOURCE_RAW_SHA256 = "08245820235fe4c553be5c7c961dda2ec8c21938374f241b84fc0a94a9123868"
SOURCE_EVIDENCE_SHA256 = "de446e86df020b29732c011185c48f6e0120bac2b47b9299ea95336dfacd8e29"
PARENT_SHA256 = "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e"
PARENT_METADATA_SHA256 = "1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e"
ARTIFACT_SHA256 = "292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc"
ROWS = 1_400
ROUTE = CAPABILITY_CAKE_ORDER.index("domain_independent_reasoning")


class ScreenError(RuntimeError):
    pass


def _binding(path: Path, *, screen: Path, protocol: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ScreenError("R88 binding is absent")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.get("binding_sha256")
    unsigned = {key: item for key, item in value.items() if key != "binding_sha256"}
    implementation = {
        "screen_sha256": screen,
        "protocol_sha256": protocol,
        "core_sha256": screen.with_name("core_v1.py"),
        "trainer_sha256": screen.with_name("train_candidate_v1.py"),
    }
    if (
        value.get("format") != "abi-r88-joint-span-candidate-binding/1"
        or _canonical_sha(unsigned) != expected
        or value.get("candidate_generation_observations_before_binding") != 0
        or any(value.get(key) != _sha256_file(file) for key, file in implementation.items())
    ):
        raise ScreenError("R88 binding or frozen implementation changed")
    return value


def _preflight(candidate: Path, parent: Path, binding: dict[str, Any]) -> dict[str, Any]:
    for path, digest in (
        (candidate / "bridge.safetensors", binding["candidate_checkpoint_sha256"]),
        (candidate / "metadata.json", binding["candidate_metadata_sha256"]),
        (parent / "model.safetensors", PARENT_SHA256),
        (parent / "metadata.json", PARENT_METADATA_SHA256),
    ):
        if not path.is_file() or _sha256_file(path) != digest:
            raise ScreenError(f"R88 frozen input changed: {path}")
    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
    validate_metadata(metadata, candidate)
    training = metadata.get("training", {})
    imported = metadata.get("imported_artifact", {})
    normalized = metadata.get("normalization", {})
    parent_contract = metadata.get("parent_layercake", {})
    validation = metadata.get("prospective_validation", {})
    implementation = metadata.get("implementation", {})
    if (
        training.get("seed") != 88_001
        or training.get("successful_optimizer_steps") != 3_000
        or training.get("batch_size") != 32
        or training.get("learning_rate") != 1.0e-3
        or imported.get("sha256") != ARTIFACT_SHA256
        or imported.get("records") != 8_129
        or imported.get("all_records_seen") is not True
        or normalized.get("method") != "deterministic_bijective_nonce_code_renaming"
        or normalized.get("adds_teacher_claims") is not False
        or normalized.get("synthetic_teacher_tokens") != 0
        or normalized.get("normalized_text_stored_in_package") is not False
        or parent_contract.get("checkpoint_sha256") != PARENT_SHA256
        or parent_contract.get("all_parameters_frozen") is not True
        or parent_contract.get("changed_on_disk") is not False
        or validation.get("catalog_sha256") != CATALOG_SHA256
        or validation.get("candidate_outputs_observed") != 0
        or implementation.get("screen_sha256") != binding["screen_sha256"]
        or implementation.get("protocol_sha256") != binding["protocol_sha256"]
        or implementation.get("core_sha256") != binding["core_sha256"]
        or implementation.get("trainer_sha256") != binding["trainer_sha256"]
    ):
        raise ScreenError("R88 training or package contract changed")
    return metadata


def _check_sparse(model: Any) -> None:
    if tuple(model.last_cake_calls) != (ROUTE,):
        raise ScreenError("R88 executed an inactive capability cake")
    traces = [
        tuple(getattr(block, "_abi_last_deep_adapter_routes", ()))
        for block in model.transformer.h
    ]
    if len(traces) != 6 or any(trace != (ROUTE,) for trace in traces):
        raise ScreenError("R88 executed an inactive or missing deep adapter")


@torch.inference_mode()
def _joint_outputs(
    *,
    model: Any,
    tokenizer: Any,
    trained: JointSpanBridge,
    randomized: JointSpanBridge,
    prompt: str,
    device: torch.device,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt + "\n", add_special_tokens=False)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(ids, dtype=torch.bool)
    routes = torch.tensor([ROUTE], dtype=torch.long, device=device)
    started = time.perf_counter()
    result = model(
        ids,
        attention_mask=attention,
        prompt_lengths=attention.long().sum(dim=1),
        task_routes=routes,
        use_cache=False,
    )
    _check_sparse(model)
    output, token_ids, start, end = realize_joint_span(
        bridge=trained,
        hidden=result["hidden"],
        attention_mask=attention,
        routes=routes,
        input_ids=ids,
        tokenizer=tokenizer,
    )
    random_output, _, random_start, random_end = realize_joint_span(
        bridge=randomized,
        hidden=result["hidden"],
        attention_mask=attention,
        routes=routes,
        input_ids=ids,
        tokenizer=tokenizer,
    )
    return {
        "output": output,
        "token_ids": token_ids,
        "start": start,
        "end": end,
        "random_output": random_output,
        "random_start": random_start,
        "random_end": random_end,
        "latency_seconds": time.perf_counter() - started,
        "physical_sparse": True,
    }


def run(
    *,
    candidate: Path,
    parent: Path,
    binding_path: Path,
    screen_path: Path,
    protocol_path: Path,
    layercake_root: Path,
    catalog_path: Path,
    source_result_path: Path,
    source_raw_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ScreenError(f"immutable R88 screen exists: {output}")
    for path, digest in (
        (catalog_path, CATALOG_SHA256),
        (source_result_path, SOURCE_RESULT_SHA256),
        (source_raw_path, SOURCE_RAW_SHA256),
    ):
        if not path.is_file() or _sha256_file(path) != digest:
            raise ScreenError("R88 source evidence or catalog changed")
    binding = _binding(binding_path, screen=screen_path, protocol=protocol_path)
    metadata = _preflight(candidate, parent, binding)
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    unsigned_source = dict(source_result)
    unsigned_source.pop("evidence_sha256", None)
    if (
        source_result.get("verdict") != "PASS_R88_SOURCE"
        or not all(source_result.get("gates", {}).values())
        or source_result.get("evidence_sha256") != SOURCE_EVIDENCE_SHA256
        or evidence_hash(unsigned_source) != SOURCE_EVIDENCE_SHA256
        or source_result.get("metrics", {}).get("prior_corrected_passing", 0) < 1_330
    ):
        raise ScreenError("R88 source result is invalid or unrecomputable")
    probes = list(load_probe_catalog(catalog_path)["probes"])
    source_rows = {str(row["probe_id"]): row for row in _jsonl(source_raw_path)}
    if (
        len(probes) != ROWS
        or len(source_rows) != ROWS
        or set(source_rows) != {str(row["probe_id"]) for row in probes}
    ):
        raise ScreenError("R88 source row coverage changed")
    for probe in probes:
        source = source_rows[str(probe["probe_id"])]
        if (
            source.get("prompt_sha256")
            != hashlib.sha256(str(probe["prompt"]).encode()).hexdigest()
            or source.get("expected_output_sha256")
            != hashlib.sha256(str(probe["evaluator"]["value"]).encode()).hexdigest()
        ):
            raise ScreenError("R88 raw source row is not catalog-bound")
    if not torch.cuda.is_available():
        raise ScreenError("R88 live screen requires CUDA")
    device = torch.device("cuda")
    process = psutil.Process()
    model, tokenizer, _ = load_layercake_core(
        parent, layercake_root=layercake_root, device=device
    )
    model.eval()
    trained, _ = load_joint_span_package(candidate, device=device)
    torch.manual_seed(88_002)
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
            **generated,
            "passed": bool(passed),
            "score": float(score),
            "random_passed": bool(random_passed),
            "collapse": collapse,
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
            "candidate_model_invocations": 1,
            "candidate_task_cake_invocations": 1,
            "candidate_deep_adapter_invocations": 6,
            "candidate_joint_span_bridge_invocations": 1,
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
            [row["source_passed"] for row in rows], 88_001,
        ),
        "candidate_minus_parent": _bootstrap(
            [row["candidate_passed"] for row in rows],
            [row["parent_passed"] for row in rows], 88_002,
        ),
        "candidate_minus_random": _bootstrap(
            [row["candidate_passed"] for row in rows],
            [row["random_passed"] for row in rows], 88_003,
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
            _sha256_file(path) == digest
            for path, digest in (
                (candidate / "bridge.safetensors", binding["candidate_checkpoint_sha256"]),
                (candidate / "metadata.json", binding["candidate_metadata_sha256"]),
                (parent / "model.safetensors", PARENT_SHA256),
                (parent / "metadata.json", PARENT_METADATA_SHA256),
                (catalog_path, CATALOG_SHA256),
            )
        ),
    }
    if not all(math.isfinite(float(value)) for value in (parent_seconds, candidate_seconds)):
        raise ScreenError("R88 runtime accounting is non-finite")
    passed = all(gates.values())
    result = {
        "format": "abi-r88-joint-span-host-screen/1",
        "verdict": "PASS_R88_BOUNDED_JOINT_SPAN_TRANSFER" if passed else "FAIL_R88_JOINT_SPAN_TRANSFER",
        "candidate_checkpoint_sha256": binding["candidate_checkpoint_sha256"],
        "candidate_metadata_sha256": binding["candidate_metadata_sha256"],
        "candidate_binding_sha256": binding["binding_sha256"],
        "parent_checkpoint_sha256": PARENT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_raw_sha256": SOURCE_RAW_SHA256,
        "metrics": metrics,
        "gates": gates,
        "artifacts": {"evaluation": {
            "path": raw_path.name,
            "sha256": _sha256_file(raw_path),
            "bytes": raw_path.stat().st_size,
        }},
        "teacher_present_at_inference": False,
        "source_parameters_retained": 0,
        "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
        "claim_boundary": "Bounded extractive teacher-to-ABI-to-joint-span-package-to-LayerCake transfer only.",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate", "parent", "binding", "screen", "protocol",
        "layercake-root", "catalog", "source-result", "source-raw", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    run(
        candidate=args.candidate.resolve(), parent=args.parent.resolve(),
        binding_path=args.binding.resolve(), screen_path=args.screen.resolve(),
        protocol_path=args.protocol.resolve(), layercake_root=args.layercake_root.resolve(),
        catalog_path=args.catalog.resolve(), source_result_path=args.source_result.resolve(),
        source_raw_path=args.source_raw.resolve(), output=args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
