"""Prepare and extract the exact-B50 top-64 teacher-logit control channel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import psutil
import torch

from .capability_compiler_phase2_common import (
    Phase2Error,
    canonical_json_bytes,
    pack_examples,
    pack_manifest,
    sha256_file,
    tokenize_records,
)
from .capability_compiler_phase2_prepare import _tokenizer, _verified_snapshot
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-b50-topk/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_EXACT_B50_TOP64_EXTRACTION"
        or protocol.get("device") != "cuda"
        or protocol.get("model_inference_authorized")
        != "FROZEN_TEACHER_TOP64_EXTRACTION_ONLY"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not True
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("exact B50 top-64 extraction governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B50 top-64 binding changed: {relative}")
    return protocol, sha256_file(path)


def _records(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        entries = set(archive.namelist())
        if entries != {"accounting.json", "manifest.json", "records.jsonl"}:
            raise Phase3Error("exact B50 records archive member set changed")
        manifest = json.loads(archive.read("manifest.json"))
        raw = archive.read("records.jsonl")
        if hashlib.sha256(raw).hexdigest() != manifest["records_jsonl_sha256"]:
            raise Phase3Error("exact B50 records archive manifest changed")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != 5140:
        raise Phase3Error("exact B50 records depth changed")
    return rows


def reconstruct_packs(root: Path, protocol: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    records = _records(root / protocol["records_archive"])
    tokenizer = _tokenizer(_verified_snapshot(root))
    packs = pack_examples(
        tokenize_records(records, tokenizer),
        max_tokens=int(protocol["packing_context"]),
        seed=int(protocol["packing_seed"]),
    )
    rebuilt = pack_manifest(packs)
    observed = _json(root / protocol["pack_manifest"])
    for key in (
        "packs",
        "pack_count",
        "record_count",
        "input_tokens",
        "response_tokens",
        "content_sha256",
    ):
        if observed[key] != rebuilt[key]:
            raise Phase3Error(f"exact B50 pack reconstruction changed: {key}")
    return packs, observed


def cache_shape_valid(
    *, positions: np.ndarray, indices: np.ndarray, values: np.ndarray, expected: int, topk: int
) -> bool:
    return (
        positions.shape == (expected,)
        and indices.shape == (expected, topk)
        and values.shape == (expected, topk)
        and positions.dtype == np.int32
        and indices.dtype == np.int32
        and values.dtype == np.float16
        and np.isfinite(values).all()
    )


def preflight(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable exact B50 top-64 preflight exists: {output}")
    packs, manifest = reconstruct_packs(root, protocol)
    gates = {
        "cuda_available": torch.cuda.is_available(),
        "pack_count_exact": len(packs) == int(protocol["expected"]["pack_count"]),
        "response_positions_exact": sum(len(pack.response_positions) for pack in packs)
        == int(protocol["expected"]["response_positions"]),
        "pack_content_exact": manifest["content_sha256"]
        == protocol["expected"]["pack_content_sha256"],
        "all_positions_in_range": all(
            all(0 <= position < len(pack.input_ids) for position in pack.response_positions)
            for pack in packs
        ),
        "teacher_model_not_loaded": True,
        "model_inference_absent": True,
        "training_absent": True,
        "final_test_absent": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-b50-topk-preflight-result/1",
        "status": "PASS_EXACT_B50_TOP64_EXTRACTION_READY"
        if all(gates.values())
        else "FAIL_EXACT_B50_TOP64_EXTRACTION_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "packs": len(packs),
        "response_positions": sum(len(pack.response_positions) for pack in packs),
        "pack_content_sha256": manifest["content_sha256"],
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gates": gates,
        "teacher_model_loaded": False,
        "model_inference_performed": False,
        "training_performed": False,
        "final_test_accessed": False,
        "claim_boundary": (
            "Top-64 extraction preflight only. No teacher weights were loaded and no "
            "inference, candidate training, baseline result, or Phase 4 claim exists."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def extract(
    root: Path,
    protocol_path: Path,
    preflight_path: Path,
    cache_dir: Path,
    output: Path,
) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable exact B50 top-64 result exists: {output}")
    preflight_result = _json(preflight_path)
    if (
        preflight_result.get("status") != "PASS_EXACT_B50_TOP64_EXTRACTION_READY"
        or preflight_result.get("protocol_sha256") != protocol_sha
    ):
        raise Phase3Error("exact B50 top-64 preflight changed")
    packs, manifest = reconstruct_packs(root, protocol)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot = _verified_snapshot(root)
    from transformers import AutoModelForCausalLM

    run_started = time.perf_counter()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    load_seconds = time.perf_counter() - load_started
    source_parameters = sum(parameter.numel() for parameter in model.parameters())
    if source_parameters != int(protocol["expected"]["source_parameters"]):
        raise Phase3Error("frozen teacher parameter count changed")
    topk = int(protocol["topk"])
    inference_seconds = 0.0
    response_positions = stored_values = reused_files = new_files = 0
    evidence_rows: list[dict[str, Any]] = []
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    for index, pack in enumerate(packs):
        target = cache_dir / f"{index:05d}-{pack.pack_id}.npz"
        expected_positions = len(pack.response_positions)
        reused = target.exists()
        if reused:
            with np.load(target, allow_pickle=False) as cached:
                if str(cached["pack_id"].item()) != pack.pack_id:
                    raise Phase3Error("stale exact B50 top-64 cache identity")
                positions = cached["positions"]
                indices = cached["indices"]
                values = cached["values"]
            reused_files += 1
        else:
            inputs = torch.tensor([pack.input_ids], dtype=torch.long, device="cuda")
            position_tensor = torch.tensor(
                pack.response_positions, dtype=torch.long, device="cuda"
            )
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(inputs).logits[0, position_tensor]
                value_tensor, index_tensor = torch.topk(
                    logits, k=topk, dim=-1, sorted=True
                )
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - started
            positions = position_tensor.cpu().numpy().astype(np.int32, copy=False)
            indices = index_tensor.cpu().numpy().astype(np.int32, copy=False)
            values = value_tensor.cpu().numpy().astype(np.float16, copy=False)
            temporary = target.with_suffix(".tmp.npz")
            np.savez(
                temporary,
                pack_id=np.asarray(pack.pack_id),
                positions=positions,
                indices=indices,
                values=values,
            )
            os.replace(temporary, target)
            new_files += 1
        if not cache_shape_valid(
            positions=positions,
            indices=indices,
            values=values,
            expected=expected_positions,
            topk=topk,
        ):
            raise Phase3Error("exact B50 top-64 cache shape or value changed")
        if positions.tolist() != list(pack.response_positions):
            raise Phase3Error("exact B50 top-64 response positions changed")
        response_positions += expected_positions
        stored_values += int(indices.size)
        peak_rss = max(peak_rss, process.memory_info().rss)
        evidence_rows.append(
            {
                "pack_id": pack.pack_id,
                "path": target.relative_to(root).as_posix(),
                "sha256": sha256_file(target),
                "positions": expected_positions,
                "topk": topk,
                "index_bytes": int(indices.nbytes),
                "value_bytes": int(values.nbytes),
                "reused_from_interrupted_attempt": reused,
            }
        )
    gates = {
        "pack_count_exact": len(evidence_rows) == int(protocol["expected"]["pack_count"]),
        "response_positions_exact": response_positions
        == int(protocol["expected"]["response_positions"]),
        "stored_values_exact": stored_values
        == int(protocol["expected"]["response_positions"]) * topk,
        "all_cache_files_unique": len({row["pack_id"] for row in evidence_rows})
        == len(evidence_rows),
        "source_parameters_exact": source_parameters
        == int(protocol["expected"]["source_parameters"]),
        "pack_content_exact": manifest["content_sha256"]
        == protocol["expected"]["pack_content_sha256"],
        "candidate_training_absent": True,
        "final_test_absent": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-b50-topk-result/1",
        "status": "PASS_EXACT_B50_TOP64_CACHE_READY"
        if all(gates.values())
        else "FAIL_EXACT_B50_TOP64_EXTRACTION",
        "protocol_sha256": protocol_sha,
        "preflight_sha256": sha256_file(preflight_path),
        "source_model": protocol["source_model"],
        "source_revision": protocol["source_revision"],
        "source_parameters": source_parameters,
        "precision": "float16",
        "attention_implementation": "eager",
        "device": torch.cuda.get_device_name(0),
        "topk": topk,
        "temperature": float(protocol["temperature"]),
        "pack_manifest_sha256": sha256_file(root / protocol["pack_manifest"]),
        "pack_content_sha256": manifest["content_sha256"],
        "pack_count": len(evidence_rows),
        "response_positions": response_positions,
        "stored_logit_values": stored_values,
        "stored_logit_value_bytes": stored_values * 2,
        "stored_logit_index_bytes": stored_values * 4,
        "source_load_seconds": load_seconds,
        "source_inference_seconds": inference_seconds,
        "wall_seconds": time.perf_counter() - run_started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_process_rss_bytes": peak_rss,
        "resume": {
            "reused_cache_files": reused_files,
            "new_cache_files": new_files,
        },
        "hardware": {
            "machine": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "files": evidence_rows,
        "files_content_sha256": hashlib.sha256(
            canonical_json_bytes(evidence_rows)
        ).hexdigest(),
        "gates": gates,
        "teacher_model_loaded": True,
        "teacher_model_present_at_candidate_inference": False,
        "candidate_training_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": (
            "Exact-B50 top-64 cached teacher channel only. No candidate was trained "
            "and no baseline quality, matched frontier, minimum, final-test, Phase 4, "
            "or ABI-superiority result exists."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ready = subparsers.add_parser("preflight")
    ready.add_argument("--protocol", required=True)
    ready.add_argument("--output", required=True)
    run_parser = subparsers.add_parser("extract")
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--preflight", required=True)
    run_parser.add_argument("--cache-dir", required=True)
    run_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    if args.command == "preflight":
        result = preflight(root, root / args.protocol, root / args.output)
    else:
        result = extract(
            root,
            root / args.protocol,
            root / args.preflight,
            root / args.cache_dir,
            root / args.output,
        )
    print(
        json.dumps(
            {
                key: result[key]
                for key in result
                if key
                in {
                    "status",
                    "packs",
                    "pack_count",
                    "response_positions",
                    "stored_logit_values",
                    "source_inference_seconds",
                    "wall_seconds",
                }
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
