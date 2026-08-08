"""Hostile V68 verifier for the action-aligned causal teacher substrate."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from safetensors.torch import load_file
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_action_aligned_extract import (
    _load_source,
    _prepare,
    _projection,
    _tensor_hash,
)


FORMAT = "abi-capability-compiler-phase3-action-aligned-verifier/1"
ARTIFACT_FORMAT = "abi-capability-compiler-phase3-action-aligned-teacher-substrate/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error("expected JSON object")
    return value


def _evidence_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("evidence_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_HOSTILE_VERIFICATION_ONLY" or protocol.get("training_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("V68 verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"V68 verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise Phase3Error("V68 record manifest contains a non-object")
    return rows


def _verify_offsets(offsets: torch.Tensor, counts: Sequence[int], final: int) -> None:
    if offsets.dtype != torch.int64 or tuple(offsets.shape) != (len(counts) + 1,):
        raise Phase3Error("V68 ragged offsets shape or dtype changed")
    expected = [0]
    for count in counts:
        if int(count) <= 0:
            raise Phase3Error("V68 record has empty action vectors")
        expected.append(expected[-1] + int(count))
    if offsets.tolist() != expected or expected[-1] != final:
        raise Phase3Error("V68 ragged offsets do not match record counts")


def _verify_static(
    tensors: Mapping[str, torch.Tensor], rows: Sequence[Mapping[str, Any]], expected: Mapping[str, Any]
) -> dict[str, Any]:
    if set(tensors) != {"source_values", "source_offsets", "target_values", "target_offsets"}:
        raise Phase3Error("V68 tensor keys changed")
    source = tensors["source_values"]
    target = tensors["target_values"]
    if source.dtype != torch.float16 or tuple(source.shape) != (int(expected["source_vectors"]), int(expected["width"])):
        raise Phase3Error("V68 source tensor changed")
    if target.dtype != torch.float16 or tuple(target.shape) != (int(expected["target_vectors"]), int(expected["width"])):
        raise Phase3Error("V68 target tensor changed")
    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(target).all()):
        raise Phase3Error("V68 tensors contain nonfinite values")
    _verify_offsets(tensors["source_offsets"], [int(row["source_count"]) for row in rows], source.shape[0])
    _verify_offsets(tensors["target_offsets"], [int(row["target_count"]) for row in rows], target.shape[0])
    summaries = {}
    for name, value in (("source_values", source), ("target_values", target)):
        norms = value.float().norm(dim=1)
        if bool((norms <= 0).any()) or float(norms.std()) <= 0:
            raise Phase3Error(f"V68 {name} is degenerate")
        summaries[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "minimum_norm": float(norms.min()), "maximum_norm": float(norms.max()), "mean_norm": float(norms.mean()), "norm_stddev": float(norms.std())}
    return summaries


def _select(rows: Sequence[Mapping[str, Any]], seed: int, per_capability: int) -> list[int]:
    grouped: dict[str, list[tuple[str, int]]] = {}
    for index, row in enumerate(rows):
        score = hashlib.sha256(f"{seed}\0{row['record_id']}".encode("ascii")).hexdigest()
        grouped.setdefault(str(row["capability"]), []).append((score, index))
    if len(grouped) != 14:
        raise Phase3Error("V68 capability inventory changed")
    return sorted(index for capability in sorted(grouped) for _, index in sorted(grouped[capability])[:per_capability])


@torch.inference_mode()
def _recompute(
    protocol: Mapping[str, Any], prepared: Sequence[Mapping[str, Any]], selected: Sequence[int], tensors: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise Phase3Error("V68 exact recomputation requires CUDA")
    model, _, manifest = _load_source(protocol)
    projection_cpu = _projection(int(protocol["projection"]["source_width"]), int(protocol["projection"]["target_width"]), int(protocol["projection"]["seed"]))
    if _tensor_hash(projection_cpu) != protocol["projection"]["sha256"]:
        raise Phase3Error("V68 projection changed")
    projection = projection_cpu.to("cuda")
    selected_set = set(selected)
    batch_size = int(protocol["sampling"]["original_batch_size"])
    batch_starts = sorted({(index // batch_size) * batch_size for index in selected})
    compared = exact = 0
    maximum_error = 0.0
    minimum_cosine = 1.0
    for start in batch_starts:
        batch = prepared[start:start + batch_size]
        width = max(len(row["input_ids"]) for row in batch)
        ids = torch.full((len(batch), width), int(protocol["source"]["pad_token_id"]), dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        for offset, row in enumerate(batch):
            ids[offset, :len(row["input_ids"])] = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda")
            mask[offset, :len(row["input_ids"])] = 1
        hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        for offset, row in enumerate(batch):
            index = start + offset
            if index not in selected_set:
                continue
            expected_source = torch.stack([hidden[offset, group].float().mean(0) @ projection for group in [row["semantic_tokens"], *row["source_groups"]]]).half().cpu()
            expected_target = torch.stack([hidden[offset, group].float().mean(0) @ projection for group in row["target_groups"]]).half().cpu()
            s0, s1 = (int(tensors["source_offsets"][index]), int(tensors["source_offsets"][index + 1]))
            t0, t1 = (int(tensors["target_offsets"][index]), int(tensors["target_offsets"][index + 1]))
            for stored, recomputed in ((tensors["source_values"][s0:s1], expected_source), (tensors["target_values"][t0:t1], expected_target)):
                delta = (stored.float() - recomputed.float()).abs()
                maximum_error = max(maximum_error, float(delta.max()))
                cosine = torch.nn.functional.cosine_similarity(stored.float(), recomputed.float(), dim=-1)
                minimum_cosine = min(minimum_cosine, float(cosine.min()))
                exact += int((stored == recomputed).sum())
                compared += stored.numel()
    if maximum_error != 0.0 or exact != compared:
        raise Phase3Error("V68 sampled action vectors do not reproduce exactly")
    return {"sample_records": len(selected), "sample_actions": sum(int(prepared[index]["source_count"]) + int(prepared[index]["target_count"]) for index in selected), "compared_scalars": compared, "exact_scalars": exact, "exact_scalar_fraction": exact / compared, "maximum_absolute_error": maximum_error, "minimum_cosine_similarity": minimum_cosine, "source_manifest_sha256": manifest["source_manifest_sha256"]}


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    protocol, protocol_sha = _load_protocol(root, protocol_path)
    artifact_dir = (root / protocol["artifact"]["directory"]).resolve()
    metadata_path = artifact_dir / "metadata.json"
    metadata = _json(metadata_path)
    if metadata.get("format") != ARTIFACT_FORMAT or metadata.get("status") != "EXTRACTED_UNVERIFIED_TRAINING_PROHIBITED" or metadata.get("evidence_sha256") != _evidence_hash(metadata):
        raise Phase3Error("V68 metadata changed")
    tensor_path = artifact_dir / metadata["artifact"]["path"]
    records_path = artifact_dir / metadata["records"]["path"]
    for path, expected in ((metadata_path, protocol["artifact"]["metadata_sha256"]), (tensor_path, protocol["artifact"]["tensor_sha256"]), (records_path, protocol["artifact"]["records_sha256"])):
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"V68 artifact changed: {path.name}")
    rows = _read_rows(records_path)
    phase1 = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    if len(rows) != 7000 or len(phase1) != len(rows):
        raise Phase3Error("V68 record count changed")
    capabilities = Counter()
    for source, row in zip(phase1, rows):
        if row["record_id"] != source["ir_record_id"] or row["capability"] != source["capability"] or row["prompt_sha256"] != source["normalized_acquisition_prompt_sha256"] or row["output_sha256"] != source["normalized_output_sha256"]:
            raise Phase3Error("V68 provenance join changed")
        capabilities[row["capability"]] += 1
    if set(capabilities.values()) != {500} or len(capabilities) != 14:
        raise Phase3Error("V68 capability balance changed")
    tensors = load_file(str(tensor_path), device="cpu")
    structure = _verify_static(tensors, rows, protocol["expected"])
    teacher = AutoTokenizer.from_pretrained(protocol["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False)
    prepared, accounting = _prepare(root, protocol, teacher)
    del teacher
    selected = _select(rows, int(protocol["sampling"]["seed"]), int(protocol["sampling"]["records_per_capability"]))
    reproduction = _recompute(protocol, prepared, selected, tensors)
    result = {"format": "abi-capability-compiler-phase3-action-aligned-verification-result/1", "status": "PASS_ARTIFACT_VERIFIED_TRAINING_PROTOCOL_DESIGN_AUTHORIZED", "protocol_sha256": protocol_sha, "artifact": {"tensor_sha256": sha256_file(tensor_path), "records_sha256": sha256_file(records_path), "metadata_sha256": sha256_file(metadata_path)}, "tensor_structure": structure, "records": len(rows), "capabilities": dict(sorted(capabilities.items())), "inventory": accounting, "sample_reproduction": reproduction, "stored_logits": 0, "copied_source_parameters": 0, "teacher_present_in_artifact": False, "training_performed": False, "layercake_host_changed": False, "phase3_certified": False, "phase4_open": False, "final_test_accessed": False, "claim_boundary": "V68 verifies only the action-aligned teacher substrate; learned transfer and Phase 3 remain unproven."}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ACTION_ALIGNED_VERIFIER_PROTOCOL_V68.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_action_aligned/verification_v68.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    print(json.dumps(verify(root, (root / args.protocol).resolve(), (root / args.output).resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
