"""Extract fixed-projection teacher predecessor states for native host actions."""
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

from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_action_aligned_extract import _projection, _ragged_offsets, _tensor_hash
from .capability_compiler_phase3_teacher_representation_extract import _load_source


FORMAT = "abi-capability-compiler-phase3-native-causal-extraction/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error("expected JSON object")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_EXTRACTION_ONLY"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("source", {}).get("device") != "cuda"
    ):
        raise Phase3Error("native causal extraction governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"native causal extraction binding changed: {relative}")
    snapshot = Path(protocol["source"]["snapshot_path"])
    if not snapshot.is_dir() or snapshot.name != protocol["source"]["revision"]:
        raise Phase3Error("frozen source snapshot changed")
    for filename, field in (
        ("config.json", "config_sha256"),
        ("tokenizer_config.json", "tokenizer_config_sha256"),
    ):
        source_file = snapshot / filename
        if not source_file.is_file() or sha256_file(source_file) != protocol["source"][field]:
            raise Phase3Error(f"frozen source {filename} changed")
    return protocol, sha256_file(path)


def _predecessor_indices(prompt_count: int, action_count: int) -> list[int]:
    if prompt_count <= 0 or action_count <= 0:
        raise Phase3Error("native causal sequence is empty")
    return list(range(prompt_count - 1, prompt_count + action_count - 1))


def _prepare(
    root: Path, protocol: Mapping[str, Any], tokenizer: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    prepared: list[dict[str, Any]] = []
    order = hashlib.sha256()
    totals = Counter()
    terminal = int(protocol["source"]["terminal_response_token_id"])
    maximum = 0
    for row in rows:
        rendered = str(row["rendered_generation_prompt"])
        output_text = str(row["normalized_output"])
        prompt_ids = [int(value) for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]]
        normalized_ids = [int(value) for value in row["normalized_output_token_ids"]]
        authoritative = [int(value) for value in row["authoritative_generated_token_ids"]]
        native_ids = [int(value) for value in tokenizer(output_text, add_special_tokens=False)["input_ids"]]
        contextual_ids = [int(value) for value in tokenizer(rendered + output_text, add_special_tokens=False)["input_ids"]]
        if native_ids != normalized_ids or authoritative != normalized_ids + [terminal]:
            raise Phase3Error("native response identity changed")
        if contextual_ids != prompt_ids + normalized_ids:
            raise Phase3Error("contextual native response identity changed")
        predecessor = _predecessor_indices(len(prompt_ids), len(normalized_ids))
        combined = prompt_ids + normalized_ids
        if predecessor[-1] >= len(combined):
            raise Phase3Error("native causal predecessor index is invalid")
        maximum = max(maximum, len(combined))
        totals["teacher_input_tokens"] += len(prompt_ids)
        totals["target_actions"] += len(normalized_ids)
        totals["teacher_output_tokens_including_terminal"] += len(authoritative)
        prepared.append(
            {
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "prompt_sha256": str(row["normalized_acquisition_prompt_sha256"]),
                "output_sha256": str(row["normalized_output_sha256"]),
                "input_ids": combined,
                "predecessor_indices": predecessor,
                "target_count": len(normalized_ids),
            }
        )
        order.update(
            (str(row["ir_record_id"]) + ":" + ",".join(map(str, normalized_ids)) + "\n").encode()
        )
    accounting = {
        "records": len(prepared),
        **totals,
        "teacher_forward_tokens": totals["teacher_input_tokens"] + totals["target_actions"],
        "maximum_sequence_tokens": maximum,
        "record_action_order_sha256": order.hexdigest(),
    }
    return prepared, accounting


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    protocol, protocol_sha = load_protocol(root, protocol_path)
    tokenizer = AutoTokenizer.from_pretrained(
        protocol["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False
    )
    prepared, accounting = _prepare(root, protocol, tokenizer)
    for key, expected in protocol["expected"].items():
        if key in accounting and accounting[key] != expected:
            raise Phase3Error(f"native causal extraction inventory changed: {key}")
    projection = _projection(
        int(protocol["projection"]["source_width"]),
        int(protocol["projection"]["target_width"]),
        int(protocol["projection"]["seed"]),
    )
    if _tensor_hash(projection) != protocol["projection"]["sha256"]:
        raise Phase3Error("native causal projection changed")
    offsets = _ragged_offsets([row["target_count"] for row in prepared])
    payload = accounting["target_actions"] * int(protocol["projection"]["target_width"]) * 2
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        **accounting,
        "activation_payload_bytes": payload,
        "offset_payload_bytes": offsets.numel() * offsets.element_size(),
        "projection_sha256": _tensor_hash(projection),
        "teacher_model_loaded": False,
        "training_performed": False,
        "final_test_accessed": False,
    }


@torch.inference_mode()
def extract(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("native causal output exists or CUDA unavailable")
    set_determinism(int(protocol["runtime"]["seed"]))
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    loaded = time.perf_counter()
    model, tokenizer, source_manifest = _load_source(protocol)
    load_seconds = time.perf_counter() - loaded
    prepared, accounting = _prepare(root, protocol, tokenizer)
    for key, expected in protocol["expected"].items():
        if key in accounting and accounting[key] != expected:
            raise Phase3Error(f"native causal post-load inventory changed: {key}")
    projection_cpu = _projection(
        int(protocol["projection"]["source_width"]),
        int(protocol["projection"]["target_width"]),
        int(protocol["projection"]["seed"]),
    )
    if _tensor_hash(projection_cpu) != protocol["projection"]["sha256"]:
        raise Phase3Error("native causal projection changed")
    projection = projection_cpu.to("cuda")
    offsets = _ragged_offsets([row["target_count"] for row in prepared])
    values = torch.empty((accounting["target_actions"], projection.shape[1]), dtype=torch.float16)
    batch_size = int(protocol["runtime"]["batch_size"])
    pad = int(protocol["source"]["pad_token_id"])
    started = time.perf_counter()
    base = model.model
    for batch_start in range(0, len(prepared), batch_size):
        batch = prepared[batch_start : batch_start + batch_size]
        width = max(len(row["input_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        for index, row in enumerate(batch):
            count = len(row["input_ids"])
            ids[index, :count] = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda")
            mask[index, :count] = 1
        hidden = base(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        if hidden.shape[-1] != projection.shape[0] or not bool(torch.isfinite(hidden).all()):
            raise Phase3Error("native causal teacher hidden state is invalid")
        for index, row in enumerate(batch):
            record_index = batch_start + index
            selected = hidden[index, row["predecessor_indices"]].float() @ projection
            values[offsets[record_index] : offsets[record_index + 1]] = selected.half().cpu()
        peak_rss = max(peak_rss, process.memory_info().rss)
        completed = batch_start + len(batch)
        if completed % int(protocol["runtime"]["progress_interval_records"]) == 0 or completed == len(prepared):
            print(json.dumps({"extracted_records": completed, "records": len(prepared), "wall_seconds": time.perf_counter() - started}), flush=True)
    forward_seconds = time.perf_counter() - started
    if not bool(torch.isfinite(values).all()):
        raise Phase3Error("native causal substrate is nonfinite")
    payload = values.numel() * values.element_size()
    if payload != int(protocol["expected"]["activation_payload_bytes"]):
        raise Phase3Error("native causal payload changed")
    output.mkdir(parents=True)
    tensor_path = output / "native_causal_predecessor_fp16.safetensors"
    save_file({"target_values": values.contiguous(), "target_offsets": offsets.contiguous()}, str(tensor_path))
    records_path = output / "records.jsonl"
    records_path.write_bytes(
        b"".join(
            canonical_json_bytes({key: row[key] for key in ("record_id", "capability", "prompt_sha256", "output_sha256", "target_count")})
            for row in prepared
        )
    )
    metadata = {
        "format": "abi-capability-compiler-phase3-native-causal-teacher-substrate/1",
        "status": "EXTRACTED_UNVERIFIED_TRAINING_PROHIBITED",
        "protocol_sha256": protocol_sha,
        "source_manifest": source_manifest,
        "artifact": {"path": tensor_path.name, "sha256": sha256_file(tensor_path), "bytes": tensor_path.stat().st_size},
        "records": {"path": records_path.name, "sha256": sha256_file(records_path), "bytes": records_path.stat().st_size, **accounting},
        "representation": {
            "target_values": list(values.shape),
            "activation_payload_bytes": payload,
            "offset_payload_bytes": offsets.numel() * offsets.element_size(),
            "projection_sha256": protocol["projection"]["sha256"],
            "causal_target_indexing": "final base-model hidden state immediately preceding each native response token",
        },
        "runtime": {
            "seed": int(protocol["runtime"]["seed"]),
            "batch_size": batch_size,
            "model_load_seconds": load_seconds,
            "teacher_forward_seconds": forward_seconds,
            "teacher_forward_tokens_per_second": accounting["teacher_forward_tokens"] / forward_seconds,
            "peak_process_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
        },
        "imported_information": {
            "raw_source_prompts": accounting["records"],
            "teacher_input_tokens": accounting["teacher_input_tokens"],
            "teacher_output_tokens": accounting["teacher_output_tokens_including_terminal"],
            "logits_stored": 0,
            "hidden_activations_stored": accounting["target_actions"],
            "hidden_activation_scalars": values.numel(),
            "hidden_activation_bytes": payload,
            "source_parameters_copied": 0,
        },
        "teacher_present_in_artifact": False,
        "teacher_required_at_inference": False,
        "training_performed": False,
        "layercake_host_changed": False,
        "phase3_certified": False,
        "phase4_open": False,
        "final_test_accessed": False,
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "extract"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_NATIVE_CAUSAL_EXTRACTION_PROTOCOL_V90.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_native_causal/extraction_v90")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    result = inventory(root, protocol) if args.command == "inventory" else extract(root, protocol, (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
