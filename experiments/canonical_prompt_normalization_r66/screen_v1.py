"""Screen R66 on adjudicable R60 development rows and all stability rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from abi.capability_compiler_phase2_common import sha256_file
from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import (
    CAPABILITY_CAKE_ORDER,
    SIX_BLOCK_DEEP_CAPABILITY_ADAPTER_ARCHITECTURE,
    load_layercake_core,
)
from experiments.evaluator_lineage_audit_r64.audit_v1 import repaired_evaluator
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once
from experiments.irreversible_collapse_invariant_r59 import screen_v1 as r59
from experiments.isolated_capability_cakes_r53 import fit_router


CANDIDATE_SHA256 = "421777ed99459391cf36b528350169998f6bec67220d717212c32955591d26de"
METADATA_SHA256 = "2aee3cc21a1a248f4d91171461cbf273c2f8bae1d29ec08f3dfb0cb4f692b14b"
NORMALIZATION_RECEIPT_SHA256 = "d790c8cf324ed8befcd0af3da2e389793bfe417185317628a557f8c84ec2311b"
CATALOG_SHA256 = "4b0087c9a7fa0e0fd6f607fdbd94fffbdd3cb375f5880a0588c97414583a2ad7"
SOURCE_OUTPUTS_SHA256 = "35a5a6e1035035db1663d793257274e3be2b766aab79a00a0476478d0428b596"
EXPECTED_NORMALIZATION = {
    "broad_english_form": 19_337,
    "supplied_text_linguistic": 4_769,
    "none": 315,
}


class ScreenError(RuntimeError):
    pass


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        raise ScreenError(f"missing raw rows: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScreenError(f"unreadable raw rows: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise ScreenError(f"non-object raw row: {path}")
    return rows


def _preflight(candidate: Path) -> None:
    receipt_path = candidate / "normalization_receipt.json"
    if not NORMALIZATION_RECEIPT_SHA256:
        raise ScreenError("R66 screen is not bound to the normalization receipt")
    if (
        sha256_file(candidate / "model.safetensors") != CANDIDATE_SHA256
        or sha256_file(candidate / "metadata.json") != METADATA_SHA256
        or sha256_file(receipt_path) != NORMALIZATION_RECEIPT_SHA256
    ):
        raise ScreenError("R66 candidate identity changed")
    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    acquired = metadata.get("acquired_core", {})
    architecture = metadata.get("architecture", {})
    expansion = acquired.get("capability_cake_expansion", {})
    training = metadata.get("training", {})
    if (
        architecture.get("architecture_version") != SIX_BLOCK_DEEP_CAPABILITY_ADAPTER_ARCHITECTURE
        or architecture.get("layers") != 6
        or architecture.get("task_cakes") != 14
        or tuple(architecture.get("capability_cake_order", ())) != CAPABILITY_CAKE_ORDER
        or architecture.get("capability_adapter_rank") != 32
        or architecture.get("capability_adapter_shared_across_layers") is not False
        or acquired.get("trainable_scope") != "deep_capability_adapter_cakes"
        or acquired.get("trainable_parameter_count") != 5_787_086
        or acquired.get("frozen_parameter_count") != 81_923_342
        or acquired.get("frozen_shared_state_preserved_exact") is not True
        or acquired.get("frozen_shared_state_sha256_before")
        != acquired.get("frozen_shared_state_sha256_after")
        or acquired.get("maximum_active_task_cakes_per_sequence") != 1
        or expansion.get("installed_capability_cakes") != 14
        or expansion.get("installed_deep_adapters") != 84
        or expansion.get("active_adapter_parameters") != 304_128
        or expansion.get("maximum_active_deep_adapters_per_sequence") != 6
        or expansion.get("extra_kv_positions") != 0
        or training.get("successful_optimizer_steps") != 6_000
        or training.get("seed") != 66_001
        or training.get("parent_logit_preservation_weight") != 0.5
        or receipt.get("normalization_counts") != EXPECTED_NORMALIZATION
        or receipt.get("audited_records") != 24_421
        or receipt.get("new_teacher_outputs") != 0
        or receipt.get("r60_outputs_used_for_training") != 0
        or receipt.get("teacher_calls") != 0
        or receipt.get("checkpoint_sha256") != CANDIDATE_SHA256
        or receipt.get("metadata_sha256") != METADATA_SHA256
    ):
        raise ScreenError("R66 training or normalization contract changed")


def run(
    candidate: Path,
    catalog_path: Path,
    source_path: Path,
    layercake_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ScreenError(f"immutable R66 screen exists: {output}")
    _preflight(candidate)
    if sha256_file(catalog_path) != CATALOG_SHA256 or sha256_file(source_path) != SOURCE_OUTPUTS_SHA256:
        raise ScreenError("R66 development catalog or source changed")
    if not torch.cuda.is_available():
        raise ScreenError("R66 development screen requires CUDA")
    probes = list(load_probe_catalog(catalog_path)["probes"])
    source = {str(row["probe_id"]): row for row in _jsonl(source_path)}
    counts = Counter(str(row["capability"]) for row in probes)
    if (
        len(probes) != 1_400 or len(source) != 1_400 or len(counts) != 14
        or set(counts.values()) != {100}
        or set(source) != {str(row["probe_id"]) for row in probes}
    ):
        raise ScreenError("R66 development matrix identity changed")

    device = torch.device("cuda")
    model, tokenizer, _ = load_layercake_core(candidate, layercake_root=layercake_root, device=device)
    model.eval()
    telemetry = {
        "candidate_transitions_checked": 0, "accepted_tokens": 0,
        "rejected_boundary_tokens": 0,
        "rejections_by_condition": {
            "maximum_identical_token_run": 0,
            "repeated_novel_lexical_fourgrams": 0,
        },
        "language_model_eos_stops": 0, "inherited_r55_lexical_truncations": 0,
    }
    rows = []
    started = time.perf_counter()
    for ordinal, probe in enumerate(probes, 1):
        prompt = str(probe["prompt"])
        route = fit_router.CAPABILITY_TO_INDEX[str(probe["capability"])]
        text, tokens, latency, physical = r59._generate(
            model, tokenizer, prompt, route, int(probe["max_new_tokens"]), device,
            telemetry=telemetry,
        )
        if not physical or not all(calls == (route,) for calls in physical):
            raise ScreenError(f"R66 sparse execution changed: {probe['probe_id']}")
        classification, evaluator, reason = repaired_evaluator(probe)
        candidate_pass = candidate_score = source_pass = source_score = None
        if evaluator is not None:
            candidate_pass, candidate_score = evaluate_output(text, evaluator)
            source_pass, source_score = evaluate_output(
                str(source[str(probe["probe_id"])]["output"]), evaluator
            )
        collapse = _collapse_metrics(tokens, text, tokenizer.encode(prompt + "\n"), prompt)
        rows.append({
            "probe_id": probe["probe_id"], "capability": probe["capability"],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "classification": classification, "classification_reason": reason,
            "evaluator": evaluator, "route": route,
            "route_source": "sealed_catalog_capability_label",
            "output": text, "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "output_token_ids": tokens, "functional_pass": candidate_pass,
            "functional_score": candidate_score,
            "source_pass": source_pass, "source_score": source_score,
            "source_output_sha256": source[str(probe["probe_id"])]["output_sha256"],
            "source_passing_regression": bool(source_pass and not candidate_pass)
                if evaluator is not None else None,
            "final_collapse": collapse, "latency_seconds": latency,
            "physical_sparse": max(map(len, physical)) == 1,
        })
        if ordinal % 100 == 0:
            print(json.dumps({
                "evaluated": ordinal,
                "adjudicable_passes": sum(row["functional_pass"] is True for row in rows),
                "source_passes": sum(row["source_pass"] is True for row in rows),
                "collapses": sum(row["final_collapse"]["collapse_detected"] for row in rows),
            }), flush=True)

    output.mkdir(parents=True)
    raw_path = output / "evaluation.jsonl"
    write_jsonl_once(raw_path, rows)
    adjudicable = [row for row in rows if row["functional_pass"] is not None]
    unchanged = [row for row in rows if row["classification"] == "unchanged"]
    source_passes = sum(row["source_pass"] is True for row in adjudicable)
    regressions = sum(row["source_passing_regression"] is True for row in adjudicable)
    by_capability = {
        capability: {
            "classification": next(row["classification"] for row in rows if row["capability"] == capability),
            "candidate_passes": sum(row["capability"] == capability and row["functional_pass"] is True for row in rows),
            "source_passes": sum(row["capability"] == capability and row["source_pass"] is True for row in rows),
            "collapses": sum(row["capability"] == capability and row["final_collapse"]["collapse_detected"] for row in rows),
        }
        for capability in sorted(counts)
    }
    metrics = {
        "rows": len(rows), "adjudicable_rows": len(adjudicable),
        "unchanged_rows": len(unchanged),
        "adjudicable_candidate_passes": sum(row["functional_pass"] is True for row in adjudicable),
        "adjudicable_source_passes": source_passes,
        "unchanged_candidate_passes": sum(row["functional_pass"] is True for row in unchanged),
        "r59_unchanged_baseline": 778,
        "source_passing_regressions": regressions,
        "source_retention": (source_passes - regressions) / source_passes,
        "final_collapses": sum(row["final_collapse"]["collapse_detected"] for row in rows),
        "physical_sparse_rows": sum(row["physical_sparse"] for row in rows),
        "by_capability": by_capability,
        "wall_seconds": time.perf_counter() - started,
    }
    unchanged_caps = [value for value in by_capability.values() if value["classification"] == "unchanged"]
    gates = {
        "matrix": len(rows) == 1_400 and len(adjudicable) == 1_200 and len(unchanged) == 1_000,
        "source_retention": metrics["source_retention"] >= 0.94,
        "unchanged_per_capability": all(value["candidate_passes"] >= 65 for value in unchanged_caps),
        "materially_exceeds_r59_unchanged": metrics["unchanged_candidate_passes"] > 778,
        "zero_final_collapse": metrics["final_collapses"] == 0,
        "physical_sparse": metrics["physical_sparse_rows"] == 1_400,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-r66-canonical-prompt-normalization-development-screen/1",
        "verdict": "PASS_R66_AUTHORIZE_NEW_PROSPECTIVE" if passed else "FAIL_R66_CLOSE_NORMALIZATION_BRANCH",
        "candidate_checkpoint_sha256": CANDIDATE_SHA256,
        "candidate_metadata_sha256": METADATA_SHA256,
        "normalization_receipt_sha256": NORMALIZATION_RECEIPT_SHA256,
        "catalog_sha256": CATALOG_SHA256, "source_outputs_sha256": SOURCE_OUTPUTS_SHA256,
        "routing_condition": "sealed_catalog_label_oracle_not_operational_router",
        "metrics": metrics, "gates": gates, "runtime_telemetry": telemetry,
        "artifacts": {"evaluation": {"path": raw_path.name, "bytes": raw_path.stat().st_size, "sha256": sha256_file(raw_path)}},
        "teacher_calls": 0, "training_steps_during_screen": 0,
        "promotion_eligible": False, "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.candidate.resolve(), args.catalog.resolve(), args.source.resolve(),
        args.layercake_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
