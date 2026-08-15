"""Train a frozen matched baseline on the independently verified exact B40 pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from unittest.mock import patch

import numpy as np

from . import capability_compiler_phase2_common as common
from . import capability_compiler_phase2_lora as lora
from . import capability_compiler_phase2_student as student
from .capability_compiler_phase2_common import (
    CAPABILITIES,
    pack_examples,
    pack_manifest,
    sha256_file,
    tokenize_records,
)
from .capability_compiler_phase2_prepare import _tokenizer, _verified_snapshot
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b50_baselines import (
    evaluate_lora,
    evaluate_student,
)
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-b40-baselines/1"
SYSTEMS = ("L0", "L1", "D0")
CLI_STAGES = ("headline",)
EXACT_B40_ROUTER_MEMBERSHIPS = {
    "abstention": 528,
    "clarification": 200,
    "coherence": 528,
    "conversation": 200,
    "email_drafting_from_notes": 200,
    "fact_free_reasoning": 200,
    "fluent_realization": 528,
    "format_control": 200,
    "grammar": 200,
    "instruction_following": 200,
    "prompt_grounding": 200,
    "rewriting": 200,
    "supplied_text_summarization": 200,
    "tone_control": 528,
}


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_EXACT_B40_MATCHED_BASELINE_CAMPAIGN"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not True
        or protocol.get("teacher_query_generation_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("exact B40 baseline campaign governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B40 baseline binding changed: {relative}")
    return protocol, sha256_file(path)


def load_exact_records(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("records.jsonl")
        manifest = json.loads(archive.read("manifest.json"))
    if hashlib.sha256(raw).hexdigest() != manifest["records_jsonl_sha256"]:
        raise Phase3Error("exact B40 records archive changed")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != 4112:
        raise Phase3Error("exact B40 record depth changed")
    return rows


def exact_packs(
    root: Path, protocol: Mapping[str, Any]
) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    records = load_exact_records(root / str(protocol["records_archive"]))
    tokenizer = _tokenizer(_verified_snapshot(root))
    packs = pack_examples(
        tokenize_records(records, tokenizer),
        max_tokens=int(protocol["packing_context"]),
        seed=int(protocol["packing_seed"]),
    )
    rebuilt = pack_manifest(packs)
    observed = _json(root / str(protocol["pack_manifest"]))
    if any(
        observed[key] != rebuilt[key]
        for key in (
            "packs",
            "pack_count",
            "record_count",
            "input_tokens",
            "response_tokens",
            "content_sha256",
        )
    ):
        raise Phase3Error("exact B40 pack content changed")
    return packs, observed, records


def configuration_allowed(
    protocol: Mapping[str, Any],
    *,
    system: str,
    rank: int | None,
    learning_rate: float,
    exposures: int,
) -> bool:
    spec = protocol["systems"][system]
    if learning_rate not in spec["learning_rates"] or exposures not in spec["exposures"]:
        return False
    return rank in spec["ranks"] if "ranks" in spec else rank is None


def stage_authorizes_system(protocol: Mapping[str, Any], *, stage: str, system: str) -> bool:
    contract = protocol.get("stages", {}).get(stage)
    return bool(isinstance(contract, dict) and system in contract.get("authorized_systems", []))


def validate_exact_b40_router_records(records: Sequence[Mapping[str, Any]]) -> None:
    observed = {capability: 0 for capability in CAPABILITIES}
    for row in records:
        capability = str(row.get("capability"))
        if capability not in observed:
            raise Phase3Error("exact B40 router record has an unknown capability")
        observed[capability] += 1
    if observed != EXACT_B40_ROUTER_MEMBERSHIPS:
        raise Phase3Error("exact B40 router membership depth changed")


def train_exact_b40_router(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    validate_exact_b40_router_records(records)
    grouped: dict[str, list[np.ndarray]] = {capability: [] for capability in CAPABILITIES}
    for row in records:
        grouped[str(row["capability"])].append(
            lora._feature_vector(str(row["normalized_acquisition_prompt"]))
        )
    centroids: dict[str, np.ndarray] = {}
    for capability in CAPABILITIES:
        centroid = np.mean(grouped[capability], axis=0)
        norm = float(np.linalg.norm(centroid))
        centroids[capability] = centroid / norm if norm else centroid
    return centroids


def run_one(
    root: Path,
    protocol_path: Path,
    *,
    stage: str,
    system: str,
    rank: int | None,
    learning_rate: float,
    exposures: int,
    seed: int,
    development_per_capability: int,
    output_dir: Path,
    save_checkpoint: bool,
) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    result_path = output_dir / "phase4_result.json"
    if result_path.exists():
        raise Phase3Error(f"immutable exact B40 baseline result exists: {result_path}")
    if system not in SYSTEMS or not configuration_allowed(
        protocol,
        system=system,
        rank=rank,
        learning_rate=learning_rate,
        exposures=exposures,
    ):
        raise Phase3Error("exact B40 baseline configuration is outside the frozen campaign")
    if not stage_authorizes_system(protocol, stage=stage, system=system):
        raise Phase3Error("exact B40 baseline system is unauthorized for this stage")
    contract = protocol["stages"][stage]
    if (
        seed not in contract["seeds"]
        or development_per_capability != int(contract["development_per_capability"])
        or save_checkpoint is not bool(contract["save_checkpoint"])
    ):
        raise Phase3Error("exact B40 baseline stage contract changed")
    packs, manifest, records = exact_packs(root, protocol)
    router_records = [
        dict(row, normalized_acquisition_prompt=row["normalized_generation_prompt"])
        for row in records
    ]

    def patched_reconstruct(*, root: Path, manifest_path: Path):
        if manifest_path.resolve() != (root / protocol["pack_manifest"]).resolve():
            raise Phase3Error("baseline pack path changed")
        return packs, manifest

    started = time.perf_counter()
    with ExitStack() as stack:
        if system in {"L0", "L1"}:
            stack.enter_context(patch.object(lora, "reconstruct_packs", patched_reconstruct))
            stack.enter_context(patch.object(lora, "evaluate_development", evaluate_lora))
            stack.enter_context(patch.object(lora, "train_router", train_exact_b40_router))
            stack.enter_context(
                patch.object(common, "load_phase1_records", lambda _path: router_records)
            )
            receipt = lora.train_lora(
                root=root,
                manifest_path=root / protocol["pack_manifest"],
                system=system,
                rank=int(rank),
                learning_rate=learning_rate,
                exposures=exposures,
                seed=seed,
                development_per_capability=development_per_capability,
                output_dir=output_dir,
                save_checkpoint=save_checkpoint,
            )
        else:
            stack.enter_context(patch.object(student, "reconstruct_packs", patched_reconstruct))
            stack.enter_context(patch.object(student, "evaluate_development", evaluate_student))
            receipt = student.train_student(
                root=root,
                manifest_path=root / protocol["pack_manifest"],
                topk_summary_path=None,
                method=system,
                learning_rate=learning_rate,
                exposures=exposures,
                seed=seed,
                development_per_capability=development_per_capability,
                output_dir=output_dir,
                save_checkpoint=save_checkpoint,
            )
    base_parameters = int(receipt.get("base_parameters", 0))
    adapter_parameters = int(receipt.get("adapter_parameters_per_capability", 0))
    router_parameters = int(receipt.get("router_parameters", 0))
    complete_lora_parameters = base_parameters + adapter_parameters * len(CAPABILITIES) + router_parameters
    active_parameters = (
        base_parameters + adapter_parameters + router_parameters
        if system in {"L0", "L1"}
        else int(receipt["student_parameters"])
    )
    imported = _json(root / protocol["pack_result"])["imported_information"]
    result = {
        "format": "abi-capability-compiler-phase4-b40-baseline-run-result/1",
        "status": "PASS_EXACT_B40_BASELINE_RUN_COMPLETE",
        "protocol_sha256": protocol_sha,
        "stage": stage,
        "system": system,
        "configuration": {
            "rank": rank,
            "learning_rate": learning_rate,
            "exposures": exposures,
            "seed": seed,
            "development_per_capability": development_per_capability,
        },
        "information_role": "EXACT_B40_SEQUENCE_EQUAL_INFORMATION",
        "imported_information": {**imported, "stored_top64_values": 0},
        "deployment": {
            "source_base_present_at_inference": bool(receipt.get("source_base_present_at_inference", False)),
            "complete_installed_parameters": complete_lora_parameters if system in {"L0", "L1"} else int(receipt["student_parameters"]),
            "active_parameters": active_parameters,
            "all_fourteen_lora_adapters_counted": system not in {"L0", "L1"} or complete_lora_parameters == base_parameters + adapter_parameters * 14 + router_parameters,
        },
        "training": {
            "optimizer_steps": int(receipt["successful_optimizer_steps"]),
            "response_tokens_seen": int(receipt["response_tokens_seen"]),
            "training_seconds": float(receipt["training_seconds"]),
            "run_wall_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(receipt["peak_cuda_allocated_bytes"]),
            "peak_process_rss_bytes": int(receipt["peak_process_rss_bytes"]),
        },
        "development": receipt["development"],
        "receipt": {
            "path": (output_dir / "receipt.json").relative_to(root).as_posix(),
            "sha256": sha256_file(output_dir / "receipt.json"),
        },
        "checkpoint": {
            "path": receipt.get("checkpoint_path"),
            "sha256": receipt.get("checkpoint_sha256"),
        },
        "teacher_query_generation_performed": False,
        "source_base_present_at_inference": system in {"L0", "L1"},
        "candidate_training_performed": True,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "One exact-B40 frozen matched baseline run. No matched frontier, final test, Phase 4, or ABI-superiority result.",
    }
    result["evidence_sha256"] = hashlib.sha256(common.canonical_json_bytes(result)).hexdigest()
    _write_immutable(result_path, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--stage", choices=CLI_STAGES, required=True)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--exposures", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--development-per-capability", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run_one(
        root,
        root / args.protocol,
        stage=args.stage,
        system=args.system,
        rank=args.rank,
        learning_rate=args.learning_rate,
        exposures=args.exposures,
        seed=args.seed,
        development_per_capability=args.development_per_capability,
        output_dir=root / args.output_dir,
        save_checkpoint=args.save_checkpoint,
    )
    print(json.dumps({
        "status": result["status"],
        "system": result["system"],
        "configuration": result["configuration"],
        "functional_passes": result["development"]["functional_passes"],
        "functional_passes_v2": result["development"]["functional_passes_v2"],
        "repetition_collapses_v2": result["development"]["repetition_collapses_v2"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
