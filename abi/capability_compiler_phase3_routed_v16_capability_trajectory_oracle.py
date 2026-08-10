"""No-artifact 14-capability trajectory-factorization capacity oracle."""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_routed_v15_layer0_extract as layer0
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as prior
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-routed-v16-capability-trajectory-oracle/1"


def _load(root: Path, path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_NO_ARTIFACT_FOURTEEN_CAPABILITY_TRAJECTORY_FACTORING"
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("capability-trajectory oracle governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"capability-trajectory binding changed: {name}")
    return protocol


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = _load(root, protocol_path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("capability-trajectory output exists or CUDA unavailable")
    output.mkdir(parents=True)
    set_determinism(int(protocol["seed"]))
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    artifact = (root / protocol["artifact"]["directory"]).resolve()
    model_path = artifact / "model.safetensors"
    artifact_before = sha256_file(model_path)
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str((root / protocol["layercake_host"]["repository"]).resolve()))
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"])
    candidate = PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer)
    incompatible = candidate.load_state_dict(load_file(str(model_path), device="cpu"), strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Phase3Error("capability-trajectory strict artifact load failed")
    candidate = candidate.cuda().eval()
    for parameter in candidate.parameters():
        parameter.requires_grad_(False)

    examples = sequential.field._examples(root, base, tokenizer)
    cfg = base["calibration"]
    train_rows, validation_rows, calibration_tokens = dual._calibration_examples(
        examples, seed=int(base["training"]["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )
    if len(train_rows) != 420 or len(validation_rows) != 28:
        raise Phase3Error("capability-trajectory population changed")
    rows = train_rows + validation_rows
    validation_ids = {str(row["record_id"]) for row in validation_rows}
    capability_index = {name: index for index, name in enumerate(CAPABILITIES)}

    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in teacher.parameters()) != int(base["source"]["parameter_count"]):
        raise Phase3Error("capability-trajectory source identity changed")
    terminal = int(base["source"]["terminal_token_id"])
    student_cache, teacher_cache, route_by_id = {}, {}, {}
    with torch.inference_mode():
        for row in rows:
            record_id = str(row["record_id"])
            host_ids = list(row["input_ids"])
            source_ids = [prior.source_token_id(value, terminal) for value in host_ids]
            host = torch.tensor([host_ids], dtype=torch.long, device=device)
            source = torch.tensor([source_ids], dtype=torch.long, device=device)
            route_by_id[record_id] = capability_index[str(row["capability"])]
            student_cache[record_id] = candidate.token_embedding(host).squeeze(0).half().cpu()
            teacher_cache[record_id] = teacher.model.embed_tokens(source).squeeze(0).to(torch.bfloat16).cpu()

    process = psutil.Process(); peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    layers = []; fitted_values = 0
    for layer_index, candidate_layer in enumerate(candidate.layers):
        width, rank = candidate_layer.sparse_width, candidate_layer.residual_rank
        grams = [torch.zeros((width, width), device=device) for _ in CAPABILITIES]
        crosses = [torch.zeros((width, rank), device=device) for _ in CAPABILITIES]
        observations = [0 for _ in CAPABILITIES]
        teacher_next = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                record_id = str(row["record_id"])
                student = student_cache[record_id].unsqueeze(0).to(device)
                source = teacher_cache[record_id].unsqueeze(0).to(device)
                _, source_final = dual._teacher_components(teacher, layer_index, source)
                teacher_next[record_id] = source_final.squeeze(0).to(torch.bfloat16).cpu()
                if record_id not in validation_ids:
                    positions = torch.arange(student.shape[1], device=device)
                    attention = layer0._attention(candidate_layer, student, positions)
                    feature = candidate_layer.post_attention_norm(attention)
                    gate, up = candidate_layer.sparse_gate_up_projection(feature).float().chunk(2, dim=-1)
                    sparse = (F.silu(gate) * up).reshape(-1, width)
                    linear = candidate_layer.linear_coefficient_projection(feature).float().reshape(-1, rank)
                    target = source_final.float() - attention.float() - candidate_layer.mlp_residual_mean.float()
                    coefficients = target.reshape(-1, target.shape[-1]) @ candidate_layer.mlp_output_projection.weight.float()
                    desired = coefficients - linear
                    route = route_by_id[record_id]
                    grams[route].add_(sparse.T @ sparse); crosses[route].add_(sparse.T @ desired)
                    observations[route] += sparse.shape[0]
        weights, ridges = [], []
        for route in range(len(CAPABILITIES)):
            if observations[route] < width:
                raise Phase3Error(f"insufficient capability-trajectory observations for {CAPABILITIES[route]}")
            solution, ridge = prior.solve_from_moments(grams[route], crosses[route], float(protocol["relative_ridge"]))
            weights.append(solution); ridges.append(ridge); fitted_values += solution.numel()
        student_next = {}; train_cosines = []; validation_cosines = []; validation_rmses = []
        per_capability = defaultdict(list)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                record_id = str(row["record_id"]); route = route_by_id[record_id]
                student = student_cache[record_id].unsqueeze(0).to(device)
                positions = torch.arange(student.shape[1], device=device)
                attention = layer0._attention(candidate_layer, student, positions)
                feature = candidate_layer.post_attention_norm(attention)
                gate, up = candidate_layer.sparse_gate_up_projection(feature).float().chunk(2, dim=-1)
                sparse = F.silu(gate) * up
                coefficients = candidate_layer.linear_coefficient_projection(feature).float() + sparse @ weights[route]
                value = attention.float() + candidate_layer.mlp_residual_mean.float() + F.linear(coefficients, candidate_layer.mlp_output_projection.weight.float())
                target = teacher_next[record_id].unsqueeze(0).to(device)
                cosine, rmse = prior._metrics(value, target)
                if record_id in validation_ids:
                    validation_cosines.append(cosine); validation_rmses.append(rmse)
                    per_capability[str(row["capability"])].append(cosine)
                else:
                    train_cosines.append(cosine)
                student_next[record_id] = value.squeeze(0).half().cpu()
        student_cache, teacher_cache = student_next, teacher_next
        value = {
            "layer": layer_index,
            "observations": dict(zip(CAPABILITIES, observations)),
            "effective_ridge": dict(zip(CAPABILITIES, ridges)),
            "train_mean_global_cosine": sum(train_cosines) / len(train_cosines),
            "validation_mean_global_cosine": sum(validation_cosines) / len(validation_cosines),
            "validation_minimum_global_cosine": min(validation_cosines),
            "validation_mean_global_relative_rmse": sum(validation_rmses) / len(validation_rmses),
            "validation_per_capability_mean_cosine": {name: sum(items) / len(items) for name, items in sorted(per_capability.items())},
        }
        layers.append(value); peak_rss = max(peak_rss, process.memory_info().rss)
        print(json.dumps(value), flush=True)

    top1 = 0; logit_cosines = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            record_id = str(row["record_id"])
            student = student_cache[record_id].unsqueeze(0).to(device)
            source = teacher_cache[record_id].unsqueeze(0).to(device)
            candidate_logits = candidate.lm_head(candidate.final_norm(student[:, -1]))
            teacher_logits = teacher.lm_head(teacher.model.norm(source[:, -1]))
            common = candidate_logits[:, prior.HOST_EXTERNAL_OFFSET:]
            logit_cosines.append(float(F.cosine_similarity(common.float(), teacher_logits[:, :common.shape[-1]].float()).item()))
            action = int(candidate_logits.argmax(dim=-1).item())
            try: predicted = prior.source_token_id(action, terminal)
            except Phase3Error: predicted = -1
            top1 += int(predicted == int(teacher_logits.argmax(dim=-1).item()))
    artifact_after = sha256_file(model_path)
    gates = {
        "every_layer_mean_global_cosine": min(value["validation_mean_global_cosine"] for value in layers) >= float(protocol["gates"]["every_layer_mean_global_cosine_minimum"]),
        "final_mean_global_cosine": layers[-1]["validation_mean_global_cosine"] >= float(protocol["gates"]["final_mean_global_cosine_minimum"]),
        "first_token_top1_agreement": top1 / len(validation_rows) >= float(protocol["gates"]["first_token_top1_agreement_minimum"]),
        "artifact_unchanged": artifact_before == artifact_after,
    }
    passed = all(gates.values())
    result = {
        "format": FORMAT, "status": "PASS_CAPABILITY_TRAJECTORY_ORACLE" if passed else "FAIL_CAPABILITY_TRAJECTORY_ORACLE",
        "protocol_sha256": sha256_file(protocol_path), "artifact_model_sha256_before": artifact_before,
        "artifact_model_sha256_after": artifact_after, "calibration_train_records": len(train_rows),
        "calibration_validation_records": len(validation_rows), "calibration_tokens": calibration_tokens,
        "oracle_routes": len(CAPABILITIES), "analytically_fitted_parameter_values": fitted_values,
        "additional_values_over_v15_artifact_if_naively_deployed": fitted_values - 32 * 3 * 384 * 768,
        "naive_additional_fp16_bytes": 2 * (fitted_values - 32 * 3 * 384 * 768),
        "persisted_parameter_values": 0, "layers": layers,
        "final_mean_global_cosine": layers[-1]["validation_mean_global_cosine"],
        "final_mean_common_logit_cosine": sum(logit_cosines) / len(logit_cosines),
        "first_token_top1_agreement": top1 / len(validation_rows), "gates": gates, "passed": passed,
        "wall_seconds": time.perf_counter() - started, "peak_process_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "teacher_loaded_for_read_only_calibration": True,
        "gradient_training_performed": False, "artifact_written": False, "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Fourteen-label oracle factorization feasibility only. This does not prove integrated routing, autonomous quality, runtime, Phase 3, or superiority."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V16_CAPABILITY_TRAJECTORY_ORACLE_PROTOCOL_V347.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_routed_v16/capability_trajectory_oracle_v348"); args = parser.parse_args()
    root = Path.cwd().resolve(); print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
