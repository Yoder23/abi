"""Extract the preregistered causal teacher states aligned to LayerCake actions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import psutil
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_action_aligned_feasibility import _overlaps, _piece_spans
from .capability_compiler_phase3_bpe_core import _layercake_api, _tokenizer
from .capability_compiler_phase3_teacher_representation_extract import _load_source


FORMAT = "abi-capability-compiler-phase3-action-aligned-extraction/1"


def _projection(source_width: int, target_width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return F.normalize(torch.randn((source_width, target_width), generator=generator, dtype=torch.float32), dim=0)


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _ragged_offsets(counts: Sequence[int]) -> torch.Tensor:
    offsets = [0]
    for count in counts:
        if int(count) <= 0:
            raise Phase3Error("action-aligned record has an empty ragged sequence")
        offsets.append(offsets[-1] + int(count))
    return torch.tensor(offsets, dtype=torch.int64)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error("expected JSON object")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_EXTRACTION_ONLY" or protocol.get("training_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED" or protocol.get("source", {}).get("device") != "cuda":
        raise Phase3Error("V66 action-aligned extraction governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"V66 action-aligned extraction binding changed: {relative}")
    snapshot = Path(protocol["source"]["snapshot_path"])
    for filename, field in (("config.json", "config_sha256"), ("tokenizer_config.json", "tokenizer_config_sha256")):
        source_file = snapshot / filename
        if not source_file.is_file() or sha256_file(source_file) != protocol["source"][field]:
            raise Phase3Error(f"V66 source {filename} changed")
    return protocol, sha256_file(path)


def _prepare(root: Path, protocol: Mapping[str, Any], teacher: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    _, _, tokenizer_type, _, _ = _layercake_api(root, protocol)
    layercake = _tokenizer(root, protocol, tokenizer_type)
    prepared: list[dict[str, Any]] = []
    totals = Counter()
    order = hashlib.sha256()
    terminal = int(protocol["source"]["terminal_response_token_id"])
    for row in rows:
        rendered = str(row["rendered_generation_prompt"])
        semantic = str(row["normalized_acquisition_prompt"])
        output_text = str(row["normalized_output"])
        prompt_ids = [int(value) for value in teacher(rendered, add_special_tokens=False)["input_ids"]]
        output_ids = [int(value) for value in row["authoritative_generated_token_ids"]]
        if not output_ids or output_ids[-1] != terminal:
            raise Phase3Error("V66 response terminal changed")
        combined = teacher(rendered + output_text, add_special_tokens=False, return_offsets_mapping=True)
        combined_ids = [int(value) for value in combined["input_ids"]]
        if combined_ids != prompt_ids + output_ids[:-1]:
            raise Phase3Error("V66 contextual token identity changed")
        semantic_start = rendered.find(semantic)
        if semantic_start < 0 or rendered.find(semantic, semantic_start + 1) >= 0:
            raise Phase3Error("V66 semantic span changed")
        semantic_tokens = _overlaps(combined["offset_mapping"], semantic_start, semantic_start + len(semantic))
        lines = semantic.splitlines()
        body = "\n" + "\n".join(lines[1:]).strip()
        body_relative = semantic.find(body)
        if body_relative < 0 or semantic.find(body, body_relative + 1) >= 0:
            raise Phase3Error("V66 routed body span changed")
        source_pieces = layercake.split(body)
        target_pieces = layercake.split(output_text)
        source_spans = _piece_spans(body, source_pieces, semantic_start + body_relative)
        target_spans = _piece_spans(output_text, target_pieces, len(rendered))
        source_groups = [_overlaps(combined["offset_mapping"], left, right) for left, right in source_spans]
        target_tokens = [_overlaps(combined["offset_mapping"], left, right) for left, right in target_spans]
        target_groups = [[index - 1 for index in indices] for indices in target_tokens]
        if any(min(group) < 0 for group in target_groups):
            raise Phase3Error("V66 target lacks causal predecessor")
        source_count = 1 + len(source_groups)
        target_count = len(target_groups)
        totals["source_vectors"] += source_count
        totals["target_vectors"] += target_count
        prepared.append({"record_id": str(row["ir_record_id"]), "capability": str(row["capability"]), "prompt_sha256": str(row["normalized_acquisition_prompt_sha256"]), "output_sha256": str(row["normalized_output_sha256"]), "input_ids": combined_ids, "semantic_tokens": semantic_tokens, "source_groups": source_groups, "target_groups": target_groups, "source_count": source_count, "target_count": target_count})
        order.update(str(row["ir_record_id"]).encode("ascii") + b"\n")
    accounting = {"records": len(prepared), "source_vectors": totals["source_vectors"], "target_vectors": totals["target_vectors"], "vectors": totals["source_vectors"] + totals["target_vectors"], "record_order_sha256": order.hexdigest()}
    return prepared, accounting


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer
    protocol, protocol_sha = load_protocol(root, protocol_path)
    teacher = AutoTokenizer.from_pretrained(protocol["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False)
    prepared, accounting = _prepare(root, protocol, teacher)
    for key, expected in protocol["expected"].items():
        if key in accounting and accounting[key] != expected:
            raise Phase3Error(f"V66 inventory changed: {key}")
    projection = _projection(int(protocol["projection"]["source_width"]), int(protocol["projection"]["target_width"]), int(protocol["projection"]["seed"]))
    if _tensor_hash(projection) != protocol["projection"]["sha256"]:
        raise Phase3Error("V66 projection changed")
    source_offsets = _ragged_offsets([row["source_count"] for row in prepared])
    target_offsets = _ragged_offsets([row["target_count"] for row in prepared])
    payload = accounting["vectors"] * int(protocol["projection"]["target_width"]) * 2
    return {"status": "PASS", "protocol_sha256": protocol_sha, **accounting, "activation_payload_bytes": payload, "offset_payload_bytes": source_offsets.numel() * 8 + target_offsets.numel() * 8, "projection_sha256": _tensor_hash(projection), "teacher_model_loaded": False, "training_performed": False, "final_test_accessed": False}


@torch.inference_mode()
def extract(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("V66 output exists or CUDA unavailable")
    set_determinism(int(protocol["runtime"]["seed"]))
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    loaded = time.perf_counter()
    model, teacher, source_manifest = _load_source(protocol)
    load_seconds = time.perf_counter() - loaded
    prepared, accounting = _prepare(root, protocol, teacher)
    for key, expected in protocol["expected"].items():
        if key in accounting and accounting[key] != expected:
            raise Phase3Error(f"V66 post-load inventory changed: {key}")
    projection_cpu = _projection(int(protocol["projection"]["source_width"]), int(protocol["projection"]["target_width"]), int(protocol["projection"]["seed"]))
    if _tensor_hash(projection_cpu) != protocol["projection"]["sha256"]:
        raise Phase3Error("V66 projection changed")
    projection = projection_cpu.to("cuda")
    source_values = torch.empty((accounting["source_vectors"], projection.shape[1]), dtype=torch.float16)
    target_values = torch.empty((accounting["target_vectors"], projection.shape[1]), dtype=torch.float16)
    source_offsets = _ragged_offsets([row["source_count"] for row in prepared])
    target_offsets = _ragged_offsets([row["target_count"] for row in prepared])
    batch_size = int(protocol["runtime"]["batch_size"])
    pad = int(protocol["source"]["pad_token_id"])
    started = time.perf_counter()
    base = model.model
    for batch_start in range(0, len(prepared), batch_size):
        batch = prepared[batch_start:batch_start + batch_size]
        width = max(len(row["input_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        for offset, row in enumerate(batch):
            ids[offset, :len(row["input_ids"])] = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda")
            mask[offset, :len(row["input_ids"])] = 1
        hidden = base(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        for offset, row in enumerate(batch):
            record_index = batch_start + offset
            source_groups = [row["semantic_tokens"], *row["source_groups"]]
            source_projected = torch.stack([hidden[offset, group].float().mean(0) @ projection for group in source_groups]).half().cpu()
            target_projected = torch.stack([hidden[offset, group].float().mean(0) @ projection for group in row["target_groups"]]).half().cpu()
            source_values[source_offsets[record_index]:source_offsets[record_index + 1]] = source_projected
            target_values[target_offsets[record_index]:target_offsets[record_index + 1]] = target_projected
        peak_rss = max(peak_rss, process.memory_info().rss)
        completed = batch_start + len(batch)
        if completed % 1000 == 0 or completed == len(prepared):
            print(json.dumps({"extracted_records": completed, "records": len(prepared), "wall_seconds": time.perf_counter() - started}), flush=True)
    forward_seconds = time.perf_counter() - started
    if not bool(torch.isfinite(source_values).all()) or not bool(torch.isfinite(target_values).all()):
        raise Phase3Error("V66 extracted nonfinite values")
    activation_bytes = source_values.numel() * 2 + target_values.numel() * 2
    if activation_bytes != int(protocol["expected"]["activation_payload_bytes"]):
        raise Phase3Error("V66 activation payload changed")
    output.mkdir(parents=True)
    tensor_path = output / "action_aligned_causal_fp16.safetensors"
    save_file({"source_values": source_values, "source_offsets": source_offsets, "target_values": target_values, "target_offsets": target_offsets}, str(tensor_path))
    records_path = output / "records.jsonl"
    records_path.write_bytes(b"".join(canonical_json_bytes({key: row[key] for key in ("record_id", "capability", "prompt_sha256", "output_sha256", "source_count", "target_count")}) for row in prepared))
    metadata = {"format": "abi-capability-compiler-phase3-action-aligned-teacher-substrate/1", "status": "EXTRACTED_UNVERIFIED_TRAINING_PROHIBITED", "protocol_sha256": protocol_sha, "source_manifest": source_manifest, "artifact": {"path": tensor_path.name, "sha256": sha256_file(tensor_path), "bytes": tensor_path.stat().st_size}, "records": {"path": records_path.name, "sha256": sha256_file(records_path), "bytes": records_path.stat().st_size, **accounting}, "representation": {"source_values": list(source_values.shape), "target_values": list(target_values.shape), "activation_payload_bytes": activation_bytes, "offset_payload_bytes": source_offsets.numel() * 8 + target_offsets.numel() * 8, "projection_sha256": protocol["projection"]["sha256"], "causal_target_indexing": "teacher token index minus one"}, "runtime": {"seed": int(protocol["runtime"]["seed"]), "batch_size": batch_size, "model_load_seconds": load_seconds, "teacher_forward_seconds": forward_seconds, "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)}}, "imported_information": {"records": 7000, "teacher_input_tokens": 576925, "teacher_output_tokens": 215647, "stored_logits": 0, "hidden_activations_stored": accounting["vectors"], "hidden_activation_scalars": source_values.numel() + target_values.numel(), "hidden_activation_bytes": activation_bytes, "source_parameters_copied": 0}, "teacher_present_in_artifact": False, "teacher_required_at_inference": False, "training_performed": False, "layercake_host_changed": False, "phase3_certified": False, "phase4_open": False, "final_test_accessed": False}
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "extract"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ACTION_ALIGNED_EXTRACTION_PROTOCOL_V66.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_action_aligned/extraction_v66")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = inventory(root, (root / args.protocol).resolve()) if args.command == "inventory" else extract(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
