"""Hostile verifier for the compact causal probability-field artifact."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file
import torch

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_causal_field_extract import _load_source, _prepared, _topk_field


FORMAT = "abi-capability-compiler-phase3-causal-field-verifier/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error("expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_HOSTILE_VERIFICATION"
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("causal-field verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"causal-field verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def _selected_indices(rows: list[Mapping[str, Any]], *, seed: int, per_capability: int) -> list[int]:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = hashlib.sha256(f"{seed}:{row['record_id']}".encode()).hexdigest()
        grouped[str(row["capability"])].append((key, index))
    if set(grouped) != set(CAPABILITIES):
        raise Phase3Error("verifier capability inventory changed")
    return sorted(index for capability in CAPABILITIES for _, index in sorted(grouped[capability])[:per_capability])


@torch.inference_mode()
def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("causal-field verification output exists")
    artifact = (root / protocol["artifact"]["directory"]).resolve()
    metadata = _json(artifact / "metadata.json")
    tensor_path = artifact / metadata["artifact"]["path"]
    record_path = artifact / metadata["records"]["path"]
    if sha256_file(tensor_path) != metadata["artifact"]["sha256"] or sha256_file(record_path) != metadata["records"]["sha256"]:
        raise Phase3Error("causal-field artifact identity changed")
    tensors = load_file(str(tensor_path), device="cpu")
    required = {"token_ids", "probabilities", "residual_mass", "offsets"}
    if set(tensors) != required:
        raise Phase3Error("causal-field tensor set changed")
    token_ids = tensors["token_ids"]
    probabilities = tensors["probabilities"]
    residual = tensors["residual_mass"]
    offsets = tensors["offsets"]
    positions = int(protocol["expected"]["prediction_positions"])
    top_k = int(protocol["expected"]["top_k"])
    if token_ids.shape != (positions, top_k) or token_ids.dtype != torch.uint16 or probabilities.shape != (positions, top_k) or probabilities.dtype != torch.float16 or residual.shape != (positions,) or residual.dtype != torch.float16 or offsets.shape != (int(protocol["expected"]["records"]) + 1,) or offsets.dtype != torch.int64:
        raise Phase3Error("causal-field tensor structure changed")
    if int(offsets[0]) != 0 or int(offsets[-1]) != positions or not bool((offsets[1:] >= offsets[:-1]).all()):
        raise Phase3Error("causal-field offsets are invalid")
    allowed = int(protocol["expected"]["allowed_external_vocabulary"])
    ids64 = token_ids.to(torch.int64)
    sorted_probabilities = probabilities[:, :-1] >= probabilities[:, 1:]
    unique_rows = torch.tensor([len(set(row)) == top_k for row in ids64.tolist()], dtype=torch.bool)
    sums = probabilities.float().sum(dim=-1) + residual.float()
    static_checks = {
        "tensor_sha256": sha256_file(tensor_path) == protocol["artifact"]["tensor_sha256"],
        "records_sha256": sha256_file(record_path) == protocol["artifact"]["records_sha256"],
        "tensor_payload_bytes": int(metadata["artifact"]["tensor_payload_bytes"]) == int(protocol["expected"]["tensor_payload_bytes"]),
        "token_ids_in_allowed_vocabulary": bool((ids64 < allowed).all()),
        "token_ids_unique_per_position": bool(unique_rows.all()),
        "probabilities_finite_nonnegative": bool(torch.isfinite(probabilities).all() and (probabilities >= 0).all()),
        "probabilities_sorted": bool(sorted_probabilities.all()),
        "residual_finite_bounded": bool(torch.isfinite(residual).all() and (residual >= 0).all() and (residual <= 1).all()),
        "grouped_mass_rounds_to_one": bool(torch.max(torch.abs(sums - 1.0)) <= float(protocol["expected"]["mass_absolute_tolerance"])),
        "teacher_absent": metadata.get("teacher_present_in_artifact") is False and metadata.get("source_manifest", {}).get("parameter_count", 0) > 0 and metadata.get("imported_information", {}).get("source_parameters_copied") == 0,
        "no_hidden_activations": metadata.get("imported_information", {}).get("hidden_activations_stored") == 0,
        "no_training": metadata.get("neural_training_performed") is False,
    }
    if not all(static_checks.values()):
        raise Phase3Error("causal-field hostile static checks failed")
    record_rows = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines() if line]
    tokenizer = AutoTokenizer.from_pretrained(protocol["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False)
    prepared, accounting = _prepared(root, protocol, tokenizer)
    if accounting != protocol["expected_inventory"] or len(record_rows) != len(prepared):
        raise Phase3Error("causal-field verifier inventory changed")
    for index, (stored, source) in enumerate(zip(record_rows, prepared)):
        expected = {"record_id": source["record_id"], "capability": source["capability"], "prompt_count": source["prompt_count"], "output_count": source["output_count"]}
        if stored != expected or int(offsets[index + 1] - offsets[index]) != int(source["output_count"]):
            raise Phase3Error(f"causal-field provenance join failed: {index}")
    selected = _selected_indices(prepared, seed=int(protocol["sampling"]["seed"]), per_capability=int(protocol["sampling"]["records_per_capability"]))
    partitions = sorted({index // int(protocol["runtime"]["original_batch_size"]) for index in selected})
    set_determinism(int(protocol["runtime"]["seed"]))
    model, loaded_tokenizer, manifest = _load_source(protocol)
    if manifest["source_manifest_sha256"] != protocol["source"]["source_manifest_sha256"]:
        raise Phase3Error("causal-field verifier source changed")
    exact_ids = exact_probabilities = exact_residual = sampled_positions = 0
    batch_size = int(protocol["runtime"]["original_batch_size"])
    pad = int(protocol["source"]["pad_token_id"])
    selected_set = set(selected)
    for partition in partitions:
        start = partition * batch_size
        batch = prepared[start : start + batch_size]
        width = max(len(row["input_ids"]) for row in batch)
        input_ids = torch.full((len(batch), width), pad, dtype=torch.long, device="cuda")
        mask = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
        for local, row in enumerate(batch):
            count = len(row["input_ids"])
            input_ids[local, :count] = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda")
            mask[local, :count] = 1
        logits = model(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True).logits
        for local, row in enumerate(batch):
            global_index = start + local
            if global_index not in selected_set:
                continue
            first = int(row["prompt_count"]) - 1
            count = int(row["output_count"])
            expected_ids, expected_probabilities, expected_residual = _topk_field(logits[local, first : first + count], top_k=top_k, allowed_vocabulary=allowed)
            lo = int(offsets[global_index]); hi = int(offsets[global_index + 1])
            exact_ids += int((token_ids[lo:hi] == expected_ids.cpu()).sum())
            exact_probabilities += int((probabilities[lo:hi] == expected_probabilities.cpu()).sum())
            exact_residual += int((residual[lo:hi] == expected_residual.cpu()).sum())
            sampled_positions += count
    recomputation = {
        "selected_records": len(selected),
        "selected_record_ids": [prepared[index]["record_id"] for index in selected],
        "original_batch_partitions": len(partitions),
        "sampled_positions": sampled_positions,
        "token_ids_exact": exact_ids,
        "token_ids_expected": sampled_positions * top_k,
        "probability_scalars_exact": exact_probabilities,
        "probability_scalars_expected": sampled_positions * top_k,
        "residual_scalars_exact": exact_residual,
        "residual_scalars_expected": sampled_positions,
    }
    recomputation_pass = exact_ids == sampled_positions * top_k and exact_probabilities == sampled_positions * top_k and exact_residual == sampled_positions
    result = {
        "format": "abi-capability-compiler-phase3-causal-field-verification-result/1",
        "status": "PASS_VERIFIED_TRAINING_PROTOCOL_MAY_BE_DESIGNED" if recomputation_pass else "FAIL_VERIFICATION_TRAINING_PROHIBITED",
        "protocol_sha256": protocol_sha,
        "artifact": {"tensor_sha256": sha256_file(tensor_path), "records_sha256": sha256_file(record_path), "metadata_sha256": sha256_file(artifact / "metadata.json")},
        "static_checks": static_checks,
        "provenance_joins": len(prepared),
        "recomputation": recomputation,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "teacher_loaded_for_verification_only": True,
        "neural_training_performed": False,
        "phase3_certified": False,
        "phase4_open": False,
        "final_test_accessed": False,
        "claim_boundary": "Hostile artifact verification only; no learned transfer, quality, performance, or Phase 3 certification claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_json(output, result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CAUSAL_FIELD_VERIFIER_PROTOCOL_V180.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3/causal_field_verification_v180/result_v181.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = verify(root, (root / args.protocol).resolve(), (root / args.output).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
