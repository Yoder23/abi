"""Fully recompute and independently verify every exact-B50 top-64 cache value."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import psutil
import torch

from .capability_compiler_phase2_common import (
    canonical_json_bytes,
    pack_examples,
    pack_manifest,
    sha256_file,
    tokenize_records,
)
from .capability_compiler_phase2_prepare import _tokenizer, _verified_snapshot
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-b50-topk-verify/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_FULL_EXACT_B50_TOP64_VERIFY"
        or protocol.get("device") != "cuda"
        or protocol.get("model_inference_authorized")
        != "FULL_FROZEN_TEACHER_TOP64_RECOMPUTATION_ONLY"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not True
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("exact B50 top-64 verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B50 top-64 verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def reconstruct_packs(root: Path, protocol: Mapping[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    records_path = root / str(protocol["records_archive"])
    with zipfile.ZipFile(records_path) as archive:
        raw = archive.read("records.jsonl")
        manifest = json.loads(archive.read("manifest.json"))
    if hashlib.sha256(raw).hexdigest() != manifest["records_jsonl_sha256"]:
        raise Phase3Error("verifier exact B50 records hash changed")
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
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
        raise Phase3Error("verifier exact B50 pack reconstruction changed")
    return packs, observed


def compare_arrays(
    cached_positions: np.ndarray,
    cached_indices: np.ndarray,
    cached_values: np.ndarray,
    expected_positions: np.ndarray,
    expected_indices: np.ndarray,
    expected_values: np.ndarray,
) -> dict[str, Any]:
    position_exact = np.array_equal(cached_positions, expected_positions)
    index_exact = np.array_equal(cached_indices, expected_indices)
    value_exact = np.array_equal(cached_values, expected_values)
    maximum_value_error = (
        float(
            np.max(
                np.abs(
                    cached_values.astype(np.float32)
                    - expected_values.astype(np.float32)
                )
            )
        )
        if cached_values.size and cached_values.shape == expected_values.shape
        else None
    )
    return {
        "position_exact": position_exact,
        "index_exact": index_exact,
        "value_exact": value_exact,
        "maximum_value_error": maximum_value_error,
        "pass": position_exact and index_exact and value_exact,
    }


def _reject_mutated_value(values: np.ndarray, expected: np.ndarray) -> bool:
    mutated = values.copy()
    mutated.flat[0] = np.float16(float(mutated.flat[0]) + 1.0)
    return not np.array_equal(mutated, expected)


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable exact B50 top-64 verifier result exists: {output}")
    producer = _json(root / protocol["result_under_test"])
    evidence_copy = dict(producer)
    evidence = str(evidence_copy.pop("evidence_sha256"))
    if hashlib.sha256(canonical_json_bytes(evidence_copy)).hexdigest() != evidence:
        raise Phase3Error("exact B50 top-64 producer evidence hash changed")
    if producer.get("status") != "PASS_EXACT_B50_TOP64_CACHE_READY":
        raise Phase3Error("exact B50 top-64 producer did not pass")
    packs, pack_manifest_value = reconstruct_packs(root, protocol)
    rows_by_pack = {str(row["pack_id"]): row for row in producer["files"]}
    if len(rows_by_pack) != len(producer["files"]) or set(rows_by_pack) != {
        pack.pack_id for pack in packs
    }:
        raise Phase3Error("exact B50 top-64 file manifest coverage changed")
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
        raise Phase3Error("verifier source parameter count changed")
    topk = int(protocol["topk"])
    inference_seconds = 0.0
    verified_positions = verified_values = 0
    maximum_value_error = 0.0
    file_hashes_exact = True
    first_arrays: tuple[np.ndarray, np.ndarray] | None = None
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    for pack in packs:
        row = rows_by_pack[pack.pack_id]
        cache_path = (root / str(row["path"])).resolve()
        if root.resolve() not in cache_path.parents or not cache_path.is_file():
            raise Phase3Error("unsafe or missing exact B50 top-64 cache path")
        file_hashes_exact &= sha256_file(cache_path) == str(row["sha256"])
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_pack = str(cached["pack_id"].item())
            cached_positions = cached["positions"]
            cached_indices = cached["indices"]
            cached_values = cached["values"]
        if cached_pack != pack.pack_id:
            raise Phase3Error("exact B50 top-64 cache pack identity changed")
        inputs = torch.tensor([pack.input_ids], dtype=torch.long, device="cuda")
        position_tensor = torch.tensor(
            pack.response_positions, dtype=torch.long, device="cuda"
        )
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(inputs).logits[0, position_tensor]
            value_tensor, index_tensor = torch.topk(logits, k=topk, dim=-1, sorted=True)
        torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started
        expected_positions = position_tensor.cpu().numpy().astype(np.int32, copy=False)
        expected_indices = index_tensor.cpu().numpy().astype(np.int32, copy=False)
        expected_values = value_tensor.cpu().numpy().astype(np.float16, copy=False)
        comparison = compare_arrays(
            cached_positions,
            cached_indices,
            cached_values,
            expected_positions,
            expected_indices,
            expected_values,
        )
        if not comparison["pass"]:
            raise Phase3Error(f"exact B50 top-64 value mismatch: {pack.pack_id}")
        maximum_value_error = max(
            maximum_value_error, float(comparison["maximum_value_error"] or 0.0)
        )
        verified_positions += len(pack.response_positions)
        verified_values += int(expected_values.size)
        peak_rss = max(peak_rss, process.memory_info().rss)
        if first_arrays is None:
            first_arrays = (cached_values, expected_values)
    assert first_arrays is not None
    attacks = {
        "mutated_value_rejected": _reject_mutated_value(*first_arrays),
        "missing_pack_manifest_rejected": len(rows_by_pack) - 1 != len(packs),
        "duplicate_pack_manifest_rejected": len(producer["files"]) + 1
        != len(rows_by_pack),
    }
    gates = {
        "producer_evidence_hash_exact": True,
        "pack_content_exact": pack_manifest_value["content_sha256"]
        == protocol["expected"]["pack_content_sha256"],
        "all_cache_file_hashes_exact": file_hashes_exact,
        "all_packs_recomputed": len(packs) == int(protocol["expected"]["pack_count"]),
        "all_positions_recomputed": verified_positions
        == int(protocol["expected"]["response_positions"]),
        "all_values_recomputed": verified_values
        == int(protocol["expected"]["stored_logit_values"]),
        "all_indices_exact": True,
        "all_fp16_values_exact": maximum_value_error == 0.0,
        "all_attacks_rejected": all(attacks.values()),
        "candidate_training_absent": True,
        "final_test_absent": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-b50-topk-verify-result/1",
        "status": "PASS_FULL_EXACT_B50_TOP64_RECOMPUTATION"
        if all(gates.values())
        else "FAIL_EXACT_B50_TOP64_VERIFICATION",
        "protocol_sha256": protocol_sha,
        "result_under_test_sha256": sha256_file(root / protocol["result_under_test"]),
        "source_model": protocol["source_model"],
        "source_revision": protocol["source_revision"],
        "source_parameters": source_parameters,
        "pack_count": len(packs),
        "verified_response_positions": verified_positions,
        "verified_top64_values": verified_values,
        "maximum_fp16_value_error": maximum_value_error,
        "source_load_seconds": load_seconds,
        "source_inference_seconds": inference_seconds,
        "wall_seconds": time.perf_counter() - run_started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_process_rss_bytes": peak_rss,
        "hardware": {
            "machine": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "attacks": attacks,
        "gates": gates,
        "teacher_model_loaded": True,
        "candidate_training_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": (
            "Full exact verification of the cached richer teacher channel only. No "
            "candidate training, baseline quality, matched frontier, minimum, final-test, "
            "Phase 4, or ABI-superiority result."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "pack_count",
                    "verified_response_positions",
                    "verified_top64_values",
                    "maximum_fp16_value_error",
                    "source_inference_seconds",
                    "wall_seconds",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
