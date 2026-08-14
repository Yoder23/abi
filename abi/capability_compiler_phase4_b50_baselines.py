"""Train one frozen-grid baseline run on the independently verified exact B50 pack."""

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
import torch

from . import capability_compiler_phase2_common as common
from . import capability_compiler_phase2_lora as lora
from . import capability_compiler_phase2_student as student
from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import (
    CAPABILITIES,
    Phase2Error,
    canonical_json_bytes,
    load_catalog,
    pack_examples,
    pack_manifest,
    sha256_file,
    tokenize_records,
)
from .capability_compiler_phase2_prepare import _tokenizer, _verified_snapshot
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-b50-baselines/1"
SYSTEMS = ("L0", "L1", "D0", "D1", "D2")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_EXACT_B50_MATCHED_BASELINE_CAMPAIGN"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not True
        or protocol.get("teacher_query_generation_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("exact B50 baseline campaign governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B50 baseline binding changed: {relative}")
    return protocol, sha256_file(path)


def load_exact_records(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("records.jsonl")
        manifest = json.loads(archive.read("manifest.json"))
    if hashlib.sha256(raw).hexdigest() != manifest["records_jsonl_sha256"]:
        raise Phase3Error("exact B50 records archive changed")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != 5140:
        raise Phase3Error("exact B50 record depth changed")
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
        raise Phase3Error("exact B50 pack content changed")
    return packs, observed, records


def exact_topk(
    root: Path, protocol: Mapping[str, Any], packs: Sequence[Any]
) -> tuple[dict[str, Path], dict[str, Any]]:
    summary = _json(root / str(protocol["topk_result"]))
    if (
        summary.get("status") != "PASS_EXACT_B50_TOP64_CACHE_READY"
        or int(summary.get("topk", 0)) != 64
        or float(summary.get("temperature", 0.0)) != 2.0
        or summary.get("pack_content_sha256")
        != protocol["expected"]["pack_content_sha256"]
    ):
        raise Phase3Error("verified exact B50 top-64 summary changed")
    files: dict[str, Path] = {}
    for row in summary["files"]:
        target = (root / str(row["path"])).resolve()
        if root.resolve() not in target.parents or not target.is_file():
            raise Phase3Error("unsafe or missing exact B50 top-64 cache")
        if sha256_file(target) != row["sha256"]:
            raise Phase3Error("exact B50 top-64 cache hash changed")
        pack_id = str(row["pack_id"])
        if pack_id in files:
            raise Phase3Error("duplicate exact B50 top-64 cache identity")
        files[pack_id] = target
    if set(files) != {pack.pack_id for pack in packs}:
        raise Phase3Error("exact B50 top-64 cache coverage changed")
    compatible = dict(summary)
    compatible["status"] = "PASS"
    return files, compatible


def _development_probes(catalog_path: Path, per_capability: int) -> list[dict[str, Any]]:
    catalog = load_catalog(catalog_path)
    grouped = {capability: [] for capability in CAPABILITIES}
    for probe in catalog["probes"]:
        capability = str(probe.get("canonical_capability"))
        if probe.get("split") == "validation" and capability in grouped:
            grouped[capability].append(probe)
    selected: list[dict[str, Any]] = []
    for capability in CAPABILITIES:
        rows = sorted(grouped[capability], key=lambda row: str(row["probe_id"]))
        if len(rows) != 100 or not 1 <= per_capability <= 100:
            raise Phase3Error("exact B50 development suite depth changed")
        selected.extend(rows[:per_capability])
    return selected


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_capability: dict[str, Any] = {}
    for capability in CAPABILITIES:
        subset = [row for row in rows if row["capability"] == capability]
        per_capability[capability] = {
            "observations": len(subset),
            "functional_passes": sum(bool(row["functional_pass"]) for row in subset),
            "functional_passes_v2": sum(
                bool(row["functional_pass_v2"]) for row in subset
            ),
            "repetition_collapses": sum(
                bool(row["repetition_collapse"]) for row in subset
            ),
            "repetition_collapses_v2": sum(
                bool(row["repetition_collapse_v2"]) for row in subset
            ),
        }
    return {
        "observations": len(rows),
        "functional_passes": sum(bool(row["functional_pass"]) for row in rows),
        "functional_passes_v2": sum(bool(row["functional_pass_v2"]) for row in rows),
        "repetition_collapses": sum(bool(row["repetition_collapse"]) for row in rows),
        "repetition_collapses_v2": sum(
            bool(row["repetition_collapse_v2"]) for row in rows
        ),
        "per_capability": per_capability,
    }


def evaluate_lora(
    model: Any,
    adapter_states: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    system: str,
    centroids: Mapping[str, np.ndarray],
    tokenizer: Any,
    catalog_path: Path,
    per_capability: int,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise Phase2Error(f"immutable development outputs already exist: {output_path}")
    probes = _development_probes(catalog_path, per_capability)
    rows: list[dict[str, Any]] = []
    current_capability = None
    router_correct = 0
    model.eval()
    model.config.use_cache = True
    started = time.perf_counter()
    for probe in probes:
        capability = str(probe["canonical_capability"])
        routed = capability if system == "L0" else lora.route_prompt(str(probe["prompt"]), centroids)
        router_correct += int(routed == capability)
        if routed != current_capability:
            lora.load_lora(model, adapter_states[routed])
            current_capability = routed
        prompt_ids = lora._render_prompt(tokenizer, str(probe["prompt"]))
        inputs = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs,
                do_sample=False,
                max_new_tokens=int(probe["max_new_tokens"]),
                eos_token_id=int(tokenizer.eos_token_id),
                pad_token_id=int(tokenizer.eos_token_id),
                use_cache=True,
            )[0, len(prompt_ids) :]
        output_ids = [
            int(value)
            for value in generated.tolist()
            if int(value) != int(tokenizer.eos_token_id)
        ]
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "capability": capability,
                "routed_capability": routed,
                "output": output,
                "output_token_ids": output_ids,
                "functional_pass": common.evaluate_functional(output, probe["evaluator"]),
                "functional_pass_v2": evaluate_functional_v2(
                    output, probe["evaluator"], capability
                ),
                "repetition_collapse": common.repetition_collapse(output),
                "repetition_collapse_v2": repetition_collapse_v2(output),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    result = _metrics(rows)
    result.update(
        {
            "router_correct": router_correct,
            "seconds": time.perf_counter() - started,
            "outputs_path": output_path.as_posix(),
            "outputs_sha256": sha256_file(output_path),
        }
    )
    return result


def evaluate_student(
    model: Any,
    *,
    tokenizer: Any,
    catalog_path: Path,
    per_capability: int,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise Phase2Error(f"immutable development outputs already exist: {output_path}")
    probes = _development_probes(catalog_path, per_capability)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for probe in probes:
        capability = str(probe["canonical_capability"])
        prompt_ids = student._render_prompt(tokenizer, str(probe["prompt"]))
        output_ids = student._generate_student(
            model,
            prompt_ids,
            max_new_tokens=int(probe["max_new_tokens"]),
            eos_token_id=int(tokenizer.eos_token_id),
            device=next(model.parameters()).device,
        )
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "capability": capability,
                "output": output,
                "output_token_ids": output_ids,
                "functional_pass": common.evaluate_functional(output, probe["evaluator"]),
                "functional_pass_v2": evaluate_functional_v2(
                    output, probe["evaluator"], capability
                ),
                "repetition_collapse": common.repetition_collapse(output),
                "repetition_collapse_v2": repetition_collapse_v2(output),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    result = _metrics(rows)
    result.update(
        {
            "seconds": time.perf_counter() - started,
            "outputs_path": output_path.as_posix(),
            "outputs_sha256": sha256_file(output_path),
        }
    )
    return result


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


def stage_authorizes_system(protocol: dict[str, Any], *, stage: str, system: str) -> bool:
    stage_contract = protocol.get("stages", {}).get(stage)
    return bool(
        isinstance(stage_contract, dict)
        and system in stage_contract.get("authorized_systems", [])
    )


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
        raise Phase3Error(f"immutable exact B50 baseline result exists: {result_path}")
    if system not in SYSTEMS or not configuration_allowed(
        protocol,
        system=system,
        rank=rank,
        learning_rate=learning_rate,
        exposures=exposures,
    ):
        raise Phase3Error("exact B50 baseline configuration is outside the frozen grid")
    if not stage_authorizes_system(protocol, stage=stage, system=system):
        raise Phase3Error("exact B50 baseline system is unauthorized for this stage")
    stage_contract = protocol["stages"][stage]
    if (
        seed not in stage_contract["seeds"]
        or development_per_capability != int(stage_contract["development_per_capability"])
        or save_checkpoint is not bool(stage_contract["save_checkpoint"])
    ):
        raise Phase3Error("exact B50 baseline stage contract changed")
    packs, manifest, records = exact_packs(root, protocol)
    router_records = [
        dict(row, normalized_acquisition_prompt=row["normalized_generation_prompt"])
        for row in records
    ]

    def patched_reconstruct(*, root: Path, manifest_path: Path):
        if manifest_path.resolve() != (root / protocol["pack_manifest"]).resolve():
            raise Phase3Error("baseline pack path changed")
        return packs, manifest

    def patched_topk(summary_path: Path, *, root: Path):
        if summary_path.resolve() != (root / protocol["topk_result"]).resolve():
            raise Phase3Error("baseline top-64 path changed")
        return exact_topk(root, protocol, packs)

    started = time.perf_counter()
    with ExitStack() as stack:
        if system in {"L0", "L1"}:
            stack.enter_context(patch.object(lora, "reconstruct_packs", patched_reconstruct))
            stack.enter_context(patch.object(lora, "evaluate_development", evaluate_lora))
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
            stack.enter_context(
                patch.object(student, "reconstruct_packs", patched_reconstruct)
            )
            stack.enter_context(patch.object(student, "_load_topk", patched_topk))
            stack.enter_context(
                patch.object(student, "evaluate_development", evaluate_student)
            )
            receipt = student.train_student(
                root=root,
                manifest_path=root / protocol["pack_manifest"],
                topk_summary_path=(
                    root / protocol["topk_result"] if system in {"D1", "D2"} else None
                ),
                method=system,
                learning_rate=learning_rate,
                exposures=exposures,
                seed=seed,
                development_per_capability=development_per_capability,
                output_dir=output_dir,
                save_checkpoint=save_checkpoint,
            )
    base_parameters = int(receipt.get("base_parameters", 0))
    adapter_per_capability = int(receipt.get("adapter_parameters_per_capability", 0))
    complete_lora_parameters = (
        base_parameters + adapter_per_capability * len(CAPABILITIES)
        + int(receipt.get("router_parameters", 0))
        if system in {"L0", "L1"}
        else 0
    )
    active_parameters = (
        base_parameters
        + adapter_per_capability
        + int(receipt.get("router_parameters", 0))
        if system in {"L0", "L1"}
        else int(receipt["student_parameters"])
    )
    topk = _json(root / protocol["topk_result"])
    imported = _json(root / protocol["pack_result"])["imported_information"]
    result = {
        "format": "abi-capability-compiler-phase4-b50-baseline-run-result/1",
        "status": "PASS_EXACT_B50_BASELINE_RUN_COMPLETE",
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
        "information_role": (
            "EXACT_B50_SEQUENCE_PLUS_TOP64_RICHER_CONTROL"
            if system in {"D1", "D2"}
            else "EXACT_B50_SEQUENCE_EQUAL_INFORMATION"
        ),
        "imported_information": {
            **imported,
            "stored_top64_values": int(topk["stored_logit_values"])
            if system in {"D1", "D2"}
            else 0,
            "stored_top64_value_bytes": int(topk["stored_logit_value_bytes"])
            if system in {"D1", "D2"}
            else 0,
            "stored_top64_index_bytes": int(topk["stored_logit_index_bytes"])
            if system in {"D1", "D2"}
            else 0,
        },
        "deployment": {
            "source_base_present_at_inference": bool(
                receipt.get("source_base_present_at_inference", False)
            ),
            "complete_installed_parameters": complete_lora_parameters
            if system in {"L0", "L1"}
            else int(receipt["student_parameters"]),
            "active_parameters": active_parameters,
            "all_fourteen_lora_adapters_counted": system not in {"L0", "L1"}
            or complete_lora_parameters
            == base_parameters
            + adapter_per_capability * 14
            + int(receipt.get("router_parameters", 0)),
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
        "teacher_present_at_student_inference": system in {"L0", "L1"},
        "candidate_training_performed": True,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": (
            "One exact-B50 frozen-grid baseline run. No configuration promotion, "
            "matched frontier, minimum, final-test, Phase 4, or ABI-superiority result."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(
        result_path, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--stage", choices=("grid", "full", "headline"), required=True)
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
    print(
        json.dumps(
            {
                "status": result["status"],
                "system": result["system"],
                "configuration": result["configuration"],
                "functional_passes": result["development"]["functional_passes"],
                "functional_passes_v2": result["development"]["functional_passes_v2"],
                "repetition_collapses_v2": result["development"][
                    "repetition_collapses_v2"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
