"""Extract a compact top-k causal probability field from the frozen teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import psutil
from safetensors.torch import save_file
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_causal_field_feasibility import _rows
from .capability_pipeline import build_source_model_manifest


FORMAT = "abi-capability-compiler-phase3-causal-field-extraction/1"


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
        or protocol.get("status") != "PREREGISTERED_EXTRACTION_ONLY"
        or protocol.get("source", {}).get("device") != "cuda"
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("causal-field extraction governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"causal-field extraction binding changed: {relative}")
    snapshot = Path(protocol["source"]["snapshot_path"])
    if not snapshot.is_dir() or snapshot.name != protocol["source"]["revision"]:
        raise Phase3Error("frozen source snapshot changed")
    for filename, field in (("config.json", "config_sha256"), ("tokenizer_config.json", "tokenizer_config_sha256")):
        if sha256_file(snapshot / filename) != protocol["source"][field]:
            raise Phase3Error(f"frozen source {filename} changed")
    return protocol, sha256_file(path)


def _prepared(root: Path, protocol: Mapping[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _rows(root, protocol)
    prepared: list[dict[str, Any]] = []
    order = hashlib.sha256()
    input_tokens = output_tokens = maximum = 0
    terminal = int(protocol.get("probability_field", {}).get("terminal_token_id", protocol.get("expected", {}).get("terminal_token_id", -1)))
    if terminal < 0:
        raise Phase3Error("causal-field terminal token is not bound")
    for row in rows:
        prompt_ids = tokenizer.encode(str(row["rendered_prompt"]), add_special_tokens=False)
        generated = [int(value) for value in row["generated_ids"]]
        if len(prompt_ids) != int(row["teacher_input_tokens"]) or len(generated) != int(row["teacher_output_tokens"]) or generated[-1] != terminal:
            raise Phase3Error(f"causal-field record boundary changed: {row['record_id']}")
        combined = prompt_ids + generated
        if len(combined) > int(protocol["source"]["maximum_sequence_tokens"]):
            raise Phase3Error("causal-field source sequence exceeds teacher bound")
        prepared.append(
            {
                "record_id": row["record_id"],
                "capability": row["capability"],
                "input_ids": combined,
                "prompt_count": len(prompt_ids),
                "output_count": len(generated),
            }
        )
        order.update(str(row["record_id"]).encode("ascii") + b"\n")
        input_tokens += len(prompt_ids)
        output_tokens += len(generated)
        maximum = max(maximum, len(combined))
    accounting = {
        "records": len(prepared),
        "teacher_input_tokens": input_tokens,
        "teacher_output_tokens": output_tokens,
        "teacher_forward_tokens": input_tokens + output_tokens,
        "maximum_sequence_tokens": maximum,
        "record_order_sha256": order.hexdigest(),
    }
    return prepared, accounting


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer

    protocol, protocol_sha = load_protocol(root, protocol_path)
    tokenizer = AutoTokenizer.from_pretrained(protocol["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False)
    _, accounting = _prepared(root, protocol, tokenizer)
    if accounting != protocol["expected_inventory"]:
        raise Phase3Error("causal-field extraction inventory changed")
    positions = accounting["teacher_output_tokens"]
    top_k = int(protocol["probability_field"]["top_k"])
    tensor_payload = positions * (top_k * 2 + top_k * 2 + 2) + (accounting["records"] + 1) * 8
    if tensor_payload != int(protocol["probability_field"]["expected_tensor_payload_bytes"]):
        raise Phase3Error("causal-field payload accounting changed")
    return {
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        **accounting,
        "prediction_positions": positions,
        "top_k": top_k,
        "tensor_payload_bytes": tensor_payload,
        "teacher_model_loaded": False,
        "teacher_forward_passes": 0,
        "neural_training_performed": False,
        "final_test_accessed": False,
    }


def _load_source(protocol: Mapping[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot = Path(protocol["source"]["snapshot_path"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16, attn_implementation="eager")
    model.to("cuda").eval()
    weight_files = [
        {"relative_path": path.relative_to(snapshot).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(snapshot.glob("*.safetensors"))
    ]
    manifest = build_source_model_manifest(
        model_id=protocol["source"]["model"],
        revision=protocol["source"]["revision"],
        revision_is_immutable=True,
        architecture=model.config.architectures[0],
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        tokenizer_id=protocol["source"]["model"],
        tokenizer_revision=protocol["source"]["revision"],
        license_id=protocol["source"]["license"],
        weight_files=weight_files,
        trust_remote_code=False,
    )
    if manifest["source_manifest_sha256"] != protocol["source"]["source_manifest_sha256"]:
        raise Phase3Error("loaded teacher identity changed")
    return model, tokenizer, manifest


def _topk_field(logits: torch.Tensor, *, top_k: int, allowed_vocabulary: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = logits.float()
    log_z = torch.logsumexp(values, dim=-1, keepdim=True)
    top_values, top_ids = torch.topk(values[:, :allowed_vocabulary], top_k, dim=-1, sorted=True)
    probabilities = torch.exp(top_values - log_z)
    residual = (1.0 - probabilities.sum(dim=-1)).clamp(min=0.0, max=1.0)
    return top_ids.to(torch.uint16), probabilities.to(torch.float16), residual.to(torch.float16)


@torch.inference_mode()
def extract(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("causal-field output exists or CUDA unavailable")
    set_determinism(int(protocol["runtime"]["seed"]))
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model, tokenizer, manifest = _load_source(protocol)
    load_seconds = time.perf_counter() - load_started
    prepared, accounting = _prepared(root, protocol, tokenizer)
    if accounting != protocol["expected_inventory"]:
        raise Phase3Error("causal-field inventory changed after teacher load")
    positions = accounting["teacher_output_tokens"]
    top_k = int(protocol["probability_field"]["top_k"])
    allowed = int(protocol["probability_field"]["allowed_external_vocabulary"])
    token_ids = torch.empty((positions, top_k), dtype=torch.uint16)
    probabilities = torch.empty((positions, top_k), dtype=torch.float16)
    residual_mass = torch.empty((positions,), dtype=torch.float16)
    offsets = torch.empty((len(prepared) + 1,), dtype=torch.int64)
    offsets[0] = 0
    cursor = 0
    batch_size = int(protocol["runtime"]["batch_size"])
    pad = int(protocol["source"]["pad_token_id"])
    forward_started = time.perf_counter()
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        width = max(len(row["input_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device="cuda")
        mask = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
        for index, row in enumerate(batch):
            count = len(row["input_ids"])
            ids[index, :count] = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda")
            mask[index, :count] = 1
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits
        if logits.shape[-1] < allowed or not bool(torch.isfinite(logits).all()):
            raise Phase3Error("teacher causal logits are invalid")
        for index, row in enumerate(batch):
            first = int(row["prompt_count"]) - 1
            count = int(row["output_count"])
            selected = logits[index, first : first + count]
            top_ids, top_probabilities, residual = _topk_field(selected, top_k=top_k, allowed_vocabulary=allowed)
            token_ids[cursor : cursor + count] = top_ids.cpu()
            probabilities[cursor : cursor + count] = top_probabilities.cpu()
            residual_mass[cursor : cursor + count] = residual.cpu()
            cursor += count
            offsets[start + index + 1] = cursor
        del logits
        peak_rss = max(peak_rss, process.memory_info().rss)
        completed = start + len(batch)
        if completed % int(protocol["runtime"]["progress_interval_records"]) == 0 or completed == len(prepared):
            print(json.dumps({"extracted_records": completed, "records": len(prepared), "positions": cursor, "wall_seconds": time.perf_counter() - forward_started}), flush=True)
    forward_seconds = time.perf_counter() - forward_started
    if cursor != positions or int(offsets[-1]) != positions:
        raise Phase3Error("causal-field ragged accounting changed")
    if not bool(torch.isfinite(probabilities).all()) or not bool(torch.isfinite(residual_mass).all()):
        raise Phase3Error("causal-field probabilities are nonfinite")
    output.mkdir(parents=True)
    tensor_path = output / "top32_probability_field.safetensors"
    save_file({"token_ids": token_ids.contiguous(), "probabilities": probabilities.contiguous(), "residual_mass": residual_mass.contiguous(), "offsets": offsets.contiguous()}, str(tensor_path))
    record_path = output / "records.jsonl"
    record_path.write_bytes(
        b"".join(
            canonical_json_bytes({"record_id": row["record_id"], "capability": row["capability"], "prompt_count": row["prompt_count"], "output_count": row["output_count"]})
            for row in prepared
        )
    )
    tensor_payload = token_ids.numel() * token_ids.element_size() + probabilities.numel() * probabilities.element_size() + residual_mass.numel() * residual_mass.element_size() + offsets.numel() * offsets.element_size()
    if tensor_payload != int(protocol["probability_field"]["expected_tensor_payload_bytes"]):
        raise Phase3Error("causal-field tensor payload changed")
    metadata = {
        "format": "abi-capability-compiler-phase3-causal-probability-field/1",
        "status": "EXTRACTED_UNVERIFIED_TRAINING_PROHIBITED",
        "protocol_sha256": protocol_sha,
        "source_manifest": manifest,
        "artifact": {"path": tensor_path.name, "sha256": sha256_file(tensor_path), "file_bytes": tensor_path.stat().st_size, "tensor_payload_bytes": tensor_payload},
        "records": {"path": record_path.name, "sha256": sha256_file(record_path), "bytes": record_path.stat().st_size, **accounting},
        "representation": {
            "top_k": top_k,
            "allowed_external_vocabulary": allowed,
            "teacher_logit_vocabulary": int(model.config.vocab_size),
            "probability": "softmax over full teacher vocabulary at temperature 1.0",
            "residual_mass": "all unselected allowed tokens plus all source-only excluded rows",
            "prediction_positions": positions,
            "token_id_dtype": "uint16",
            "probability_dtype": "float16",
            "residual_mass_dtype": "float16",
            "offset_dtype": "int64",
        },
        "runtime": {
            "device": "cuda",
            "weight_dtype": "float16",
            "batch_size": batch_size,
            "model_load_seconds": load_seconds,
            "teacher_forward_seconds": forward_seconds,
            "teacher_forward_tokens_per_second": accounting["teacher_forward_tokens"] / forward_seconds,
            "peak_process_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
        },
        "imported_information": {
            "raw_source_prompts": len(prepared),
            "teacher_input_tokens": accounting["teacher_input_tokens"],
            "teacher_output_tokens": accounting["teacher_output_tokens"],
            "stored_logits": positions * top_k,
            "stored_probability_scalars": positions * (top_k + 1),
            "hidden_activations_stored": 0,
            "source_parameters_copied": 0,
            "tensor_payload_bytes": tensor_payload,
        },
        "teacher_required_at_final_inference": False,
        "teacher_present_in_artifact": False,
        "neural_training_performed": False,
        "phase3_certified": False,
        "phase4_open": False,
        "final_test_accessed": False,
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_json(output / "metadata.json", metadata)
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "extract"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CAUSAL_FIELD_EXTRACTION_PROTOCOL_V179.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3/causal_field_extraction_v179")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    result = inventory(root, protocol) if args.command == "inventory" else extract(root, protocol, (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
