"""Read-only layerwise error-accumulation audit for the failed routed-v15 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_routed_v15_layer0_extract as layer0
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-routed-v15-accumulation-attribution/1"
HOST_EXTERNAL_OFFSET = 4
HOST_EOS = 2


def source_token_id(action: int, terminal: int) -> int:
    if action == HOST_EOS:
        return terminal
    if action < HOST_EXTERNAL_OFFSET:
        raise Phase3Error("unmappable host-special action in source sequence")
    return action - HOST_EXTERNAL_OFFSET


def metrics(candidate: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    left, right = candidate.float(), target.float()
    return {
        "cosine": float(F.cosine_similarity(left.reshape(1, -1), right.reshape(1, -1)).item()),
        "relative_rmse": float(torch.sqrt(torch.mean((left - right) ** 2)).div(torch.sqrt(torch.mean(right**2)).clamp_min(1e-8)).item()),
    }


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_LAYERWISE_ACCUMULATION_ATTRIBUTION"
        or protocol.get("device") != "cuda"
        or protocol.get("training") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("routed-v15 accumulation governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"routed-v15 accumulation binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("attribution output exists or CUDA unavailable")
    output.mkdir(parents=True)
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    artifact = (root / protocol["artifact"]["directory"]).resolve()
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve()
    sys.path.insert(0, str(layercake_root))
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"])
    candidate = PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer)
    incompatible = candidate.load_state_dict(load_file(str(artifact / "model.safetensors"), device="cuda"), strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Phase3Error("attribution artifact strict load failed")
    candidate = candidate.cuda().eval()
    for parameter in candidate.parameters(): parameter.requires_grad_(False)
    examples = sequential.field._examples(root, base, tokenizer)
    example_by_id = {str(row["record_id"]): row for row in examples}
    cfg = base["calibration"]
    _, validation, _ = dual._calibration_examples(examples, seed=int(base["training"]["seed"]), train_per_capability=int(cfg["train_records_per_capability"]), validation_per_capability=int(cfg["validation_records_per_capability"]), maximum_tokens=int(cfg["maximum_sequence_tokens"]))
    if len(validation) != 28:
        raise Phase3Error("attribution validation population changed")
    teacher = AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in teacher.parameters()) != int(base["source"]["parameter_count"]):
        raise Phase3Error("attribution source identity changed")
    terminal = int(base["source"]["terminal_token_id"])
    layer_values: dict[int, list[dict[str, float]]] = {index: [] for index in range(32)}
    records = []
    started = time.perf_counter(); process = psutil.Process(); peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for row in validation:
            example = example_by_id[str(row["record_id"])]
            host_ids = list(example["source_ids"])
            source_ids = [source_token_id(value, terminal) for value in host_ids]
            host_tensor = torch.tensor([host_ids], dtype=torch.long, device="cuda")
            source_tensor = torch.tensor([source_ids], dtype=torch.long, device="cuda")
            route = candidate._select_route(host_tensor)
            expected_route = layer0._route(str(row["capability"]))
            positions = torch.arange(len(host_ids), device="cuda")
            student = candidate.token_embedding(host_tensor)
            teacher_hidden = teacher.model.embed_tokens(source_tensor)
            initial = metrics(student, teacher_hidden)
            per_layer = []
            for index, (student_layer, _) in enumerate(zip(candidate.layers, teacher.model.layers)):
                student_before = student
                student, _, _ = student_layer.forward_with_cache(student, positions, route)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, teacher_hidden = dual._teacher_components(teacher, index, teacher_hidden)
                    _, local_oracle = dual._teacher_components(teacher, index, student_before)
                global_value = metrics(student, teacher_hidden)
                local_value = metrics(student, local_oracle)
                value = {"layer": index, "global_cosine": global_value["cosine"], "global_relative_rmse": global_value["relative_rmse"], "local_cosine": local_value["cosine"], "local_relative_rmse": local_value["relative_rmse"]}
                per_layer.append(value); layer_values[index].append(value)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_logits = teacher.lm_head(teacher.model.norm(teacher_hidden[:, -1]))
            candidate_logits = candidate.lm_head(candidate.final_norm(student[:, -1]))
            candidate_common = candidate_logits[:, HOST_EXTERNAL_OFFSET:]
            source_common = teacher_logits[:, : candidate_common.shape[-1]]
            logits = metrics(candidate_common, source_common)
            source_top = int(teacher_logits.argmax(dim=-1).item())
            candidate_action = int(candidate_logits.argmax(dim=-1).item())
            try: candidate_top = source_token_id(candidate_action, terminal)
            except Phase3Error: candidate_top = -1
            records.append({"record_id": str(row["record_id"]), "capability": str(row["capability"]), "source_actions": len(host_ids), "expected_route": expected_route, "predicted_route": route, "route_correct": route == expected_route, "initial": initial, "layers": per_layer, "final_common_logit_cosine": logits["cosine"], "final_common_logit_relative_rmse": logits["relative_rmse"], "source_top_token": source_top, "candidate_top_token": candidate_top, "top1_agreement": candidate_top == source_top})
            peak_rss = max(peak_rss, process.memory_info().rss)
            print(json.dumps({"records": len(records), "top1_agreement": sum(value["top1_agreement"] for value in records)}), flush=True)
    raw = output / "records.jsonl"
    raw.write_bytes(b"".join(canonical_json_bytes(value) for value in records))
    layers = []
    for index in range(32):
        values = layer_values[index]
        layers.append({"layer": index, "mean_global_cosine": sum(value["global_cosine"] for value in values) / len(values), "mean_global_relative_rmse": sum(value["global_relative_rmse"] for value in values) / len(values), "mean_local_cosine": sum(value["local_cosine"] for value in values) / len(values), "mean_local_relative_rmse": sum(value["local_relative_rmse"] for value in values) / len(values)})
    first_global_below = next((value["layer"] for value in layers if value["mean_global_cosine"] < protocol["diagnostic_thresholds"]["global_cosine_minimum"]), None)
    result = {
        "format": FORMAT, "status": "PASS_ATTRIBUTION_COMPLETE", "protocol_sha256": sha256_file(protocol_path),
        "artifact_model_sha256": protocol["artifact"]["model_sha256"], "records": len(records), "layers": layers,
        "first_layer_global_cosine_below_threshold": first_global_below,
        "final_mean_global_cosine": layers[-1]["mean_global_cosine"], "final_mean_local_cosine": layers[-1]["mean_local_cosine"],
        "mean_final_common_logit_cosine": sum(value["final_common_logit_cosine"] for value in records) / len(records),
        "prompt_first_token_top1_agreement": sum(value["top1_agreement"] for value in records) / len(records),
        "route_correct": sum(value["route_correct"] for value in records), "raw_sha256": sha256_file(raw),
        "wall_seconds": time.perf_counter() - started, "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "training_performed": False, "artifact_mutated": False, "source_model_loaded_for_attribution": True,
        "final_test_accessed": False, "phase3_certified": False,
        "claim_boundary": "Read-only development-independent calibration attribution; no quality promotion, final-test, runtime, or superiority claim."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V15_ACCUMULATION_ATTRIBUTION_PROTOCOL_V333.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_routed_v15/accumulation_attribution_v334"); args = parser.parse_args()
    root = Path.cwd().resolve(); print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
