"""Live R78 candidate, untouched-parent, and pinned-source R77 screen."""

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
from abi.layercake_full_core_acquisition import _manifest_sha
from abi.layercake_host import CAPABILITY_TO_ROUTE, _sha256_file
from experiments.conditional_choice_host_r74.screen_candidate_v1 import _bootstrap, _generate, _jsonl
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once


CATALOG_SHA256 = "b89a473da82b66297f0cd9f7796b341f873d1bb3d1b2c7f82e94255b5b61330a"
SOURCE_RESULT_SHA256 = "172646f0602f9336aa815c1ead96147e6e469711c6213b78781f0215dd70bd6e"
SOURCE_RAW_SHA256 = "2e1f4a2f9b081fdbc0d00442bc1940f844792a7581683fc604286fa2c2df4b4b"
SOURCE_EVIDENCE_SHA256 = "6f9e4051f10673c5c7598c206b8c8d2b41595bbefb0b0f9d150520b54664f29b"
PARENT_SHA256 = "65b1aae5e1aed947f2bc086281a3c7388b22a4260c562363aff2e8cc59f9661b"
PARENT_METADATA_SHA256 = "9590374b0afd3184dd75bcf08d8b9f0876ed7bcc0db7f7ec1f4abf147093d1d6"
MAIN_ARCHIVE_SHA256 = "292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc"
ANCHOR_ARCHIVE_SHA256 = "82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150"
ROWS = 1_400
ROWS_PER_FAMILY = 200
REASONING_ROUTE = CAPABILITY_CAKE_ORDER.index("domain_independent_reasoning")
PARENT_ROUTE = CAPABILITY_TO_ROUTE["domain_independent_reasoning"]


class ScreenError(RuntimeError):
    pass


def _load_binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ScreenError("candidate binding is absent")
    binding = json.loads(path.read_text(encoding="utf-8"))
    expected = binding.get("binding_sha256")
    payload = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if binding.get("format") != "abi-r78-candidate-screen-binding/1" or expected != _canonical_sha(payload):
        raise ScreenError("candidate binding is invalid or stale")
    for field in ("candidate_checkpoint_sha256", "candidate_metadata_sha256"):
        value = binding.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ScreenError(f"candidate binding hash is invalid: {field}")
    return binding


def _preflight(candidate: Path, parent: Path, binding: dict[str, Any]) -> dict[str, Any]:
    frozen = (
        (candidate / "model.safetensors", binding["candidate_checkpoint_sha256"]),
        (candidate / "metadata.json", binding["candidate_metadata_sha256"]),
        (parent / "model.safetensors", PARENT_SHA256),
        (parent / "metadata.json", PARENT_METADATA_SHA256),
    )
    for path, digest in frozen:
        if not path.is_file() or _sha256_file(path) != digest:
            raise ScreenError(f"frozen host input changed: {path}")
    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
    unsigned = dict(metadata)
    manifest_sha = unsigned.pop("manifest_sha256", None)
    training = metadata.get("training", {})
    imported = metadata.get("imported_artifact", {})
    anchor = metadata.get("broad_behavior_anchor", {})
    boundary = metadata.get("foreign_source_boundary", {})
    acquired = metadata.get("acquired_core", {})
    expansion = acquired.get("capability_cake_expansion", {})
    if (
        _manifest_sha(unsigned) != manifest_sha
        or metadata.get("status") != "TRAINED_NOT_YET_SEMANTICALLY_OR_OPERATIONALLY_CERTIFIED"
        or training.get("seed") != 74_001
        or training.get("successful_optimizer_steps") != 6_000
        or training.get("batch_size") != 8
        or training.get("anchor_batch_size") != 8
        or training.get("balanced_terminal_loss") is not True
        or training.get("source_teacher_forward_tokens") != 0
        or training.get("source_model_inference_seconds") != 0.0
        or imported.get("archive_sha256_after") != MAIN_ARCHIVE_SHA256
        or imported.get("selected_english_records") != 8_129
        or imported.get("all_selected_records_seen") is not True
        or anchor.get("archive_sha256_after") != ANCHOR_ARCHIVE_SHA256
        or boundary.get("teacher_present_at_inference") is not False
        or boundary.get("source_parameters_copied") != 0
        or boundary.get("source_transformer_blocks_retained") != 0
        or acquired.get("physical_sparse_topology_preserved") is not True
        or expansion.get("installed_deep_adapters") != 84
        or expansion.get("maximum_active_deep_adapters_per_sequence") != 6
        or expansion.get("maximum_active_capability_cakes_per_sequence") != 1
    ):
        raise ScreenError("R78 acquisition or deployment contract changed")
    return metadata


def run(
    *,
    candidate: Path,
    parent: Path,
    binding_path: Path,
    layercake_root: Path,
    catalog_path: Path,
    source_result_path: Path,
    source_raw_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ScreenError(f"immutable R77 screen exists: {output}")
    if (
        _sha256_file(catalog_path) != CATALOG_SHA256
        or _sha256_file(source_result_path) != SOURCE_RESULT_SHA256
        or _sha256_file(source_raw_path) != SOURCE_RAW_SHA256
    ):
        raise ScreenError("R77 catalog or source evidence changed")
    binding = _load_binding(binding_path)
    metadata = _preflight(candidate, parent, binding)
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if (
        source_result.get("verdict") != "PASS_R77_SOURCE"
        or source_result.get("evidence_sha256") != SOURCE_EVIDENCE_SHA256
        or source_result.get("metrics", {}).get("passing") != 1_370
        or source_result.get("artifacts", {}).get("choice_scores", {}).get("sha256") != SOURCE_RAW_SHA256
    ):
        raise ScreenError("R77 source result changed")
    probes = list(load_probe_catalog(catalog_path)["probes"])
    source_rows = {str(row["probe_id"]): row for row in _jsonl(source_raw_path)}
    if len(probes) != ROWS or len(source_rows) != ROWS or set(source_rows) != {str(row["probe_id"]) for row in probes}:
        raise ScreenError("R77 source coverage changed")
    if not torch.cuda.is_available():
        raise ScreenError("R77 host screen requires CUDA")
    device = torch.device("cuda")
    process = psutil.Process()
    rows_by_id: dict[str, dict[str, Any]] = {str(row["probe_id"]): {"probe": row} for row in probes}

    parent_model, parent_tokenizer, _ = load_layercake_core(parent, layercake_root=layercake_root, device=device)
    parent_model.eval()
    parent_started = time.perf_counter()
    for index, probe in enumerate(probes, 1):
        generated = _generate(parent_model, parent_tokenizer, str(probe["prompt"]), PARENT_ROUTE, int(probe["max_new_tokens"]), device, require_adapters=False)
        passed, score = evaluate_output(generated["output"], probe["evaluator"])
        rows_by_id[str(probe["probe_id"])]["parent"] = {**generated, "passed": bool(passed), "score": float(score)}
        if index % 200 == 0:
            print(json.dumps({"system": "parent", "evaluated": index, "passing": sum(row.get("parent", {}).get("passed", False) for row in rows_by_id.values())}), flush=True)
    parent_seconds = time.perf_counter() - parent_started
    del parent_model
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    candidate_model, candidate_tokenizer, _ = load_layercake_core(candidate, layercake_root=layercake_root, device=device)
    candidate_model.eval()
    candidate_started = time.perf_counter()
    for index, probe in enumerate(probes, 1):
        generated = _generate(candidate_model, candidate_tokenizer, str(probe["prompt"]), REASONING_ROUTE, int(probe["max_new_tokens"]), device, require_adapters=True)
        passed, score = evaluate_output(generated["output"], probe["evaluator"])
        collapse = _collapse_metrics(generated["token_ids"], generated["output"], candidate_tokenizer.encode(str(probe["prompt"]) + "\n"), str(probe["prompt"]))
        rows_by_id[str(probe["probe_id"])]["candidate"] = {**generated, "passed": bool(passed), "score": float(score), "collapse": collapse}
        if index % 200 == 0:
            print(json.dumps({"system": "candidate", "evaluated": index, "passing": sum(row.get("candidate", {}).get("passed", False) for row in rows_by_id.values()), "collapses": sum(row.get("candidate", {}).get("collapse", {}).get("collapse_detected", False) for row in rows_by_id.values())}), flush=True)
    candidate_seconds = time.perf_counter() - candidate_started

    rows = []
    for probe in probes:
        probe_id = str(probe["probe_id"])
        source, parent_row, candidate_row = source_rows[probe_id], rows_by_id[probe_id]["parent"], rows_by_id[probe_id]["candidate"]
        rows.append({
            "probe_id": probe_id,
            "premise_family": int(probe_id.split("-")[2][1:]),
            "destination_display_index": int(probe["destination_display_index"]),
            "prompt_sha256": hashlib.sha256(str(probe["prompt"]).encode()).hexdigest(),
            "evaluator": probe["evaluator"],
            "source_passed": bool(source["passed"]),
            "source_selected_output_sha256": source["selected_output_sha256"],
            "parent_output": parent_row["output"],
            "parent_output_token_ids": parent_row["token_ids"],
            "parent_passed": parent_row["passed"],
            "parent_latency_seconds": parent_row["latency_seconds"],
            "candidate_output": candidate_row["output"],
            "candidate_output_sha256": hashlib.sha256(candidate_row["output"].encode()).hexdigest(),
            "candidate_output_token_ids": candidate_row["token_ids"],
            "candidate_passed": candidate_row["passed"],
            "candidate_latency_seconds": candidate_row["latency_seconds"],
            "candidate_language_model_eos": candidate_row["language_model_eos"],
            "candidate_collapse": candidate_row["collapse"],
            "candidate_model_invocations": candidate_row["model_invocations"],
            "candidate_adapter_invocations": candidate_row["adapter_invocations"],
            "candidate_physical_sparse": candidate_row["physical_sparse"],
            "source_passing_retained": bool(source["passed"] and candidate_row["passed"]),
        })
    output.mkdir(parents=True)
    raw_path = output / "evaluation.jsonl"
    write_jsonl_once(raw_path, rows)
    source_passing = sum(row["source_passed"] for row in rows)
    candidate_passing = sum(row["candidate_passed"] for row in rows)
    parent_passing = sum(row["parent_passed"] for row in rows)
    retained = sum(row["source_passing_retained"] for row in rows)
    family = {
        str(index): {
            "rows": ROWS_PER_FAMILY,
            "source_passing": sum(row["premise_family"] == index and row["source_passed"] for row in rows),
            "candidate_passing": sum(row["premise_family"] == index and row["candidate_passed"] for row in rows),
            "parent_passing": sum(row["premise_family"] == index and row["parent_passed"] for row in rows),
        }
        for index in range(7)
    }
    metrics = {
        "rows": len(rows),
        "source_passing": source_passing,
        "candidate_passing": candidate_passing,
        "parent_passing": parent_passing,
        "source_passing_retained": retained,
        "source_retention": retained / source_passing,
        "candidate_collapses": sum(row["candidate_collapse"]["collapse_detected"] for row in rows),
        "candidate_physical_sparse_rows": sum(row["candidate_physical_sparse"] for row in rows),
        "by_family": family,
        "candidate_minus_source": _bootstrap([row["candidate_passed"] for row in rows], [row["source_passed"] for row in rows], 77_001),
        "candidate_minus_parent": _bootstrap([row["candidate_passed"] for row in rows], [row["parent_passed"] for row in rows], 77_002),
        "parent_generation_seconds": parent_seconds,
        "candidate_generation_seconds": candidate_seconds,
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
        "zero_collapse": metrics["candidate_collapses"] == 0,
        "physical_sparse": metrics["candidate_physical_sparse_rows"] == ROWS,
        "source_absent_at_inference": metadata["foreign_source_boundary"]["teacher_present_at_inference"] is False,
        "artifacts_unchanged": _sha256_file(candidate / "model.safetensors") == binding["candidate_checkpoint_sha256"] and _sha256_file(candidate / "metadata.json") == binding["candidate_metadata_sha256"],
    }
    if not all(math.isfinite(float(value)) for value in (parent_seconds, candidate_seconds)):
        raise ScreenError("R77 runtime accounting is non-finite")
    passed = all(gates.values())
    result = {
        "format": "abi-r78-role-invariant-capability-host-screen/1",
        "verdict": "PASS_R78_BOUNDED_ROLE_INVARIANT_HOST_TRANSFER" if passed else "FAIL_R78_HOST_TRANSFER",
        "candidate_checkpoint_sha256": binding["candidate_checkpoint_sha256"],
        "candidate_metadata_sha256": binding["candidate_metadata_sha256"],
        "candidate_binding_sha256": binding["binding_sha256"],
        "parent_checkpoint_sha256": PARENT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_raw_sha256": SOURCE_RAW_SHA256,
        "metrics": metrics,
        "gates": gates,
        "artifacts": {"evaluation": {"path": raw_path.name, "sha256": _sha256_file(raw_path), "bytes": raw_path.stat().st_size}},
        "teacher_present_at_inference": False,
        "source_parameters_retained": 0,
        "candidate_accessed_only_after_protocol_and_screen_code_frozen": True,
        "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
        "claim_boundary": "Bounded causal transfer of one role-invariant conditional-weight-selected nonce reasoning capability; not unrestricted English.",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--layercake-root", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--source-result", required=True, type=Path)
    parser.add_argument("--source-raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(
        candidate=args.candidate.resolve(), parent=args.parent.resolve(), binding_path=args.binding.resolve(),
        layercake_root=args.layercake_root.resolve(), catalog_path=args.catalog.resolve(),
        source_result_path=args.source_result.resolve(), source_raw_path=args.source_raw.resolve(), output=args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
