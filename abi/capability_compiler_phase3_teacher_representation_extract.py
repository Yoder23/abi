"""Extract one preregistered pooled hidden-state substrate from the frozen teacher."""
from __future__ import annotations

import argparse
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
from .capability_pipeline import build_source_model_manifest


FORMAT = "abi-capability-compiler-phase3-teacher-representation-extraction/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error("expected JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_EXTRACTION_ONLY" or protocol.get("final_test_access") != "PROHIBITED" or protocol.get("training_authorized") is not False or protocol.get("source", {}).get("device") != "cuda":
        raise Phase3Error("teacher representation extraction governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"teacher representation extraction binding changed: {relative}")
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


def _find_unique(haystack: Sequence[int], needle: Sequence[int]) -> int:
    matches = [index for index in range(len(haystack) - len(needle) + 1) if list(haystack[index:index + len(needle)]) == list(needle)]
    if len(matches) != 1:
        raise Phase3Error("semantic prompt tokens do not have one rendered-prompt span")
    return matches[0]


def _find_unique_text(haystack: str, needle: str) -> tuple[int, int]:
    start = haystack.find(needle)
    if start < 0 or haystack.find(needle, start + 1) >= 0:
        raise Phase3Error("semantic prompt text does not have one rendered-prompt span")
    return start, start + len(needle)


def _offset_token_span(
    offsets: Sequence[Sequence[int]], char_start: int, char_end: int
) -> tuple[int, int]:
    selected = [
        index
        for index, (start, end) in enumerate(offsets)
        if int(end) > char_start and int(start) < char_end and int(end) > int(start)
    ]
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        raise Phase3Error("semantic prompt token offsets are noncontiguous")
    covered = [(int(offsets[index][0]), int(offsets[index][1])) for index in selected]
    if covered[0][0] != char_start or covered[-1][1] != char_end:
        raise Phase3Error("semantic prompt token offsets straddle its character boundary")
    if any(left[1] != right[0] for left, right in zip(covered, covered[1:])):
        raise Phase3Error("semantic prompt token offsets do not exactly cover its text")
    return selected[0], len(selected)


def _inventory_rows(root: Path, protocol: Mapping[str, Any], tokenizer: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    prepared = []
    sequence_hash = hashlib.sha256()
    input_tokens = output_tokens = maximum = 0
    terminal = int(protocol["pooling"]["terminal_response_token_id"])
    for row in rows:
        rendered_prompt = str(row["rendered_generation_prompt"])
        encoded = tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        prompt_ids = encoded["input_ids"]
        if len(prompt_ids) != int(row["teacher_input_tokens"]):
            raise Phase3Error("rendered prompt token count changed")
        semantic_prompt = str(row["normalized_acquisition_prompt"])
        if hashlib.sha256(semantic_prompt.encode("utf-8")).hexdigest() != str(
            row["normalized_acquisition_prompt_sha256"]
        ):
            raise Phase3Error("semantic prompt text hash changed")
        try:
            char_start, char_end = _find_unique_text(rendered_prompt, semantic_prompt)
            prompt_start, prompt_count = _offset_token_span(
                encoded["offset_mapping"], char_start, char_end
            )
        except Phase3Error as exc:
            raise Phase3Error(
                f"semantic prompt boundary changed for {row['ir_record_id']}"
            ) from exc
        output_ids = [int(value) for value in row["authoritative_generated_token_ids"]]
        if len(output_ids) != int(row["authoritative_teacher_tokens"]) or not output_ids or output_ids[-1] != terminal:
            raise Phase3Error("authoritative response token boundary changed")
        response_ids = output_ids[:-1]
        if not response_ids:
            raise Phase3Error("response has no semantic token")
        combined = prompt_ids + output_ids
        if len(combined) > int(protocol["source"]["maximum_sequence_tokens"]):
            raise Phase3Error("teacher extraction sequence exceeds source bound")
        prepared.append({"record_id": str(row["ir_record_id"]), "capability": str(row["capability"]), "prompt_sha256": str(row["normalized_acquisition_prompt_sha256"]), "output_sha256": str(row["normalized_output_sha256"]), "input_ids": combined, "prompt_start": prompt_start, "prompt_count": prompt_count, "response_start": len(prompt_ids), "response_count": len(response_ids)})
        sequence_hash.update(str(row["ir_record_id"]).encode("ascii") + b"\n")
        input_tokens += len(prompt_ids)
        output_tokens += len(output_ids)
        maximum = max(maximum, len(combined))
    return prepared, {"records": len(prepared), "teacher_input_tokens": input_tokens, "teacher_output_tokens": output_tokens, "teacher_forward_tokens": input_tokens + output_tokens, "maximum_sequence_tokens": maximum, "record_order_sha256": sequence_hash.hexdigest()}


def inventory(root: Path, protocol_path: Path) -> dict[str, Any]:
    from transformers import AutoTokenizer
    protocol, protocol_sha = load_protocol(root, protocol_path)
    tokenizer = AutoTokenizer.from_pretrained(protocol["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False)
    _, accounting = _inventory_rows(root, protocol, tokenizer)
    expected = protocol["expected_inventory"]
    for key, value in expected.items():
        if accounting.get(key) != value:
            raise Phase3Error(f"teacher extraction inventory changed: {key}")
    width = int(protocol["pooling"]["hidden_width"])
    vectors = accounting["records"] * 2
    return {"status": "PASS", "protocol_sha256": protocol_sha, **accounting, "vectors": vectors, "hidden_width": width, "tensor_payload_bytes": vectors * width * 2, "teacher_model_loaded": False, "teacher_forward_passes": 0, "training_performed": False, "final_test_accessed": False}


def _load_source(protocol: Mapping[str, Any]):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snapshot = Path(protocol["source"]["snapshot_path"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float16, attn_implementation="eager")
    model.to("cuda").eval()
    weight_paths = sorted(snapshot.glob("*.safetensors"))
    weight_files = [{"relative_path": path.relative_to(snapshot).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in weight_paths]
    manifest = build_source_model_manifest(model_id=protocol["source"]["model"], revision=protocol["source"]["revision"], revision_is_immutable=True, architecture=model.config.architectures[0], parameter_count=sum(parameter.numel() for parameter in model.parameters()), tokenizer_id=protocol["source"]["model"], tokenizer_revision=protocol["source"]["revision"], license_id=protocol["source"]["license"], weight_files=weight_files, trust_remote_code=False)
    if manifest["source_manifest_sha256"] != protocol["source"]["source_manifest_sha256"]:
        raise Phase3Error("loaded teacher source manifest changed")
    return model, tokenizer, manifest


@torch.inference_mode()
def extract(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("teacher representation output exists or CUDA unavailable")
    seed = int(protocol["runtime"]["seed"])
    set_determinism(seed)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model, tokenizer, source_manifest = _load_source(protocol)
    load_seconds = time.perf_counter() - load_started
    prepared, accounting = _inventory_rows(root, protocol, tokenizer)
    for key, value in protocol["expected_inventory"].items():
        if accounting.get(key) != value:
            raise Phase3Error(f"teacher extraction inventory changed after load: {key}")
    hidden_width = int(protocol["pooling"]["hidden_width"])
    prompt_pooled = torch.empty((len(prepared), hidden_width), dtype=torch.float16)
    response_pooled = torch.empty((len(prepared), hidden_width), dtype=torch.float16)
    batch_size = int(protocol["runtime"]["batch_size"])
    pad = int(protocol["source"]["pad_token_id"])
    forward_started = time.perf_counter()
    base = model.model
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        width = max(len(row["input_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device="cuda")
        mask = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
        for index, row in enumerate(batch):
            count = len(row["input_ids"])
            ids[index, :count] = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda")
            mask[index, :count] = 1
        hidden = base(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        if hidden.shape[-1] != hidden_width or not bool(torch.isfinite(hidden).all()):
            raise Phase3Error("teacher hidden state is invalid")
        for index, row in enumerate(batch):
            p0 = int(row["prompt_start"]); p1 = p0 + int(row["prompt_count"])
            r0 = int(row["response_start"]); r1 = r0 + int(row["response_count"])
            prompt_pooled[start + index] = hidden[index, p0:p1].float().mean(0).half().cpu()
            response_pooled[start + index] = hidden[index, r0:r1].float().mean(0).half().cpu()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if (start + len(batch)) % int(protocol["runtime"]["progress_interval_records"]) == 0 or start + len(batch) == len(prepared):
            print(json.dumps({"extracted_records": start + len(batch), "records": len(prepared), "wall_seconds": time.perf_counter() - forward_started}), flush=True)
    forward_seconds = time.perf_counter() - forward_started
    if not bool(torch.isfinite(prompt_pooled).all()) or not bool(torch.isfinite(response_pooled).all()):
        raise Phase3Error("pooled teacher substrate is nonfinite")
    output.mkdir(parents=True)
    tensor_path = output / "dual_pooled_final_hidden_fp16.safetensors"
    save_file({"prompt_pooled": prompt_pooled.contiguous(), "response_pooled": response_pooled.contiguous()}, str(tensor_path))
    rows_path = output / "records.jsonl"
    rows_path.write_bytes(b"".join(canonical_json_bytes({key: row[key] for key in ("record_id", "capability", "prompt_sha256", "output_sha256", "prompt_count", "response_count")}) for row in prepared))
    tensor_payload = prompt_pooled.numel() * prompt_pooled.element_size() + response_pooled.numel() * response_pooled.element_size()
    if tensor_payload != int(protocol["pooling"]["expected_tensor_payload_bytes"]):
        raise Phase3Error("teacher substrate payload changed")
    metadata = {"format": "abi-capability-compiler-phase3-dual-pooled-teacher-substrate/1", "status": "EXTRACTED_UNVERIFIED_TRAINING_PROHIBITED", "protocol_sha256": protocol_sha, "source_manifest": source_manifest, "representation": {"name": "dual_pooled_final_hidden_fp16", "pooling": "arithmetic mean in fp32 then cast fp16", "prompt_span": "normalized acquisition prompt tokens inside rendered teacher prompt", "response_span": "authoritative generated tokens excluding terminal <|end|>", "hidden_layer": "final normalized base-model hidden state", "vectors": len(prepared) * 2, "hidden_width": hidden_width, "tensor_payload_bytes": tensor_payload}, "artifact": {"path": tensor_path.name, "sha256": sha256_file(tensor_path), "file_bytes": tensor_path.stat().st_size}, "records": {"path": rows_path.name, "sha256": sha256_file(rows_path), "bytes": rows_path.stat().st_size, **accounting}, "runtime": {"seed": seed, "device": "cuda", "weight_dtype": "float16", "pooling_accumulation_dtype": "float32", "batch_size": batch_size, "model_load_seconds": load_seconds, "teacher_forward_seconds": forward_seconds, "teacher_forward_tokens_per_second": accounting["teacher_forward_tokens"] / forward_seconds, "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)}}, "imported_information": {"raw_source_prompts": len(prepared), "teacher_input_tokens": accounting["teacher_input_tokens"], "teacher_output_tokens": accounting["teacher_output_tokens"], "logits_stored": 0, "hidden_activations_stored": len(prepared) * 2, "hidden_activation_scalars": len(prepared) * 2 * hidden_width, "hidden_activation_bytes": tensor_payload, "source_parameters_copied": 0}, "teacher_required_at_final_inference": False, "teacher_present_in_artifact": False, "training_performed": False, "layercake_host_changed": False, "phase3_certified": False, "phase4_open": False, "final_test_accessed": False}
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_json(output / "metadata.json", metadata)
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "extract"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_REPRESENTATION_EXTRACTION_PROTOCOL_V58.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_teacher_representation/extraction_v58")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol = (root / args.protocol).resolve()
    result = inventory(root, protocol) if args.command == "inventory" else extract(root, protocol, (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
