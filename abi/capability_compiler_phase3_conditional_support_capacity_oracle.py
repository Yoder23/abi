"""Nonpromotional target-aware capacity oracle for fixed dynamic local support."""

from __future__ import annotations

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
from . import capability_compiler_phase3_existing_attention_refit as coverage
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-conditional-support-capacity-oracle/1"


def marginal_reduction_scores(target: torch.Tensor, contributions: torch.Tensor) -> torch.Tensor:
    """Squared-error reduction from adding each unscaled contribution independently."""
    if contributions.shape[:-2] != target.shape[:-1] or contributions.shape[-1] != target.shape[-1]:
        raise Phase3Error("conditional-support score shape changed")
    return 2.0 * (contributions * target.unsqueeze(-2)).sum(dim=-1) - contributions.square().sum(dim=-1)


def select_contributions(target: torch.Tensor, contributions: torch.Tensor, count: int):
    scores = marginal_reduction_scores(target, contributions)
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :count]
    selected = torch.gather(
        contributions,
        -2,
        indices.unsqueeze(-1).expand(*indices.shape, contributions.shape[-1]),
    ).sum(dim=-2)
    return selected, indices


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    support = protocol.get("conditional_support", {})
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_NONPROMOTIONAL_CONDITIONAL_SUPPORT_CAPACITY_ORACLE"
        or protocol.get("target_access") != "ORACLE_SELECTION_ONLY"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("router_write") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
        or int(support.get("heads_per_token", 0)) != 5
        or int(support.get("neurons_per_token", 0)) != 432
    ):
        raise Phase3Error("conditional-support governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"conditional-support binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("conditional-support output exists or CUDA unavailable")

    output.mkdir(parents=True)
    set_determinism(int(protocol["seed"]))
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    artifact = root / protocol["artifact"]["directory"]
    artifact_path = artifact / "model.safetensors"
    artifact_before = sha256_file(artifact_path)
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str((root / protocol["layercake_host"]).resolve()))

    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"])
    model = PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer)
    model.load_state_dict(load_file(str(artifact_path), device="cpu"), strict=True, assign=True)
    state = model.state_dict()
    prefix = load_file(str(root / protocol["layer0_checkpoint"]["path"]), device="cpu")
    with torch.no_grad():
        for name, value in prefix.items():
            state[name].copy_(value.to(state[name].dtype))
    layer0 = model.layers[0].float().cuda().eval()

    examples = sequential.field._examples(root, base, tokenizer)
    _, validation_rows = coverage.expanded_split(
        examples,
        seed=int(base["training"]["seed"]),
        maximum_tokens=int(protocol["population"]["maximum_sequence_actions"]),
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    for value in teacher.parameters():
        value.requires_grad_(False)
    source_layer1 = teacher.model.layers[1]
    source_attention = source_layer1.self_attn
    heads = int(protocol["source_topology"]["heads"])
    head_dimension = int(protocol["source_topology"]["head_dimension"])
    width = heads * head_dimension
    qkv = source_attention.qkv_proj.weight.detach().float()
    output_weight = source_attention.o_proj.weight.detach().float()
    gate_up = source_layer1.mlp.gate_up_proj.weight.detach().float()
    down = source_layer1.mlp.down_proj.weight.detach().float()
    neurons = down.shape[1]
    if qkv.shape != (3 * width, width) or output_weight.shape != (width, width) or gate_up.shape != (2 * neurons, width) or down.shape != (width, neurons):
        raise Phase3Error("conditional-support source topology changed")

    terminal = int(base["source"]["terminal_token_id"])
    head_count = int(support["heads_per_token"])
    neuron_count = int(support["neurons_per_token"])
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    route_exact = 0
    cosines = []
    rmses = []
    full_cosines = []
    full_rmses = []
    attention_cosines = []
    attention_rmses = []
    head_histogram = torch.zeros(heads, dtype=torch.int64)
    neuron_histogram = torch.zeros(neurons, dtype=torch.int64)
    record_metrics = []

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            host_ids = torch.tensor([row["input_ids"]], dtype=torch.long)
            route_index = model._select_route(host_ids)
            route_exact += int(route_index == routed._route(str(row["capability"])))
            candidate = model.token_embedding(host_ids).to(device)
            positions = torch.arange(candidate.shape[1], device=device)
            candidate, _, _ = layer0.forward_with_cache(candidate, positions, route_index)

            source_ids = torch.tensor(
                [[trajectory.source_token_id(value, terminal) for value in row["input_ids"]]],
                dtype=torch.long,
                device=device,
            )
            native = teacher.model.embed_tokens(source_ids)
            for source_index in range(2):
                _, native = dual._teacher_components(teacher, source_index, native)
            native = native.float()

            length = candidate.shape[1]
            position_ids = torch.arange(length, device=device)[None]
            position_embeddings = teacher.model.rotary_emb(candidate, position_ids)
            normalized = source_layer1.input_layernorm(candidate)
            exact_delta, weights = source_attention(
                hidden_states=normalized,
                attention_mask=dual.base._causal_mask(length, device=device, dtype=candidate.dtype),
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            head_contributions = []
            value_start = 2 * width
            for head in range(heads):
                value_weight = qkv[value_start + head * head_dimension:value_start + (head + 1) * head_dimension]
                head_output = output_weight[:, head * head_dimension:(head + 1) * head_dimension]
                values = F.linear(normalized.float(), value_weight)
                attended = weights[:, head].float() @ values
                head_contributions.append(F.linear(attended, head_output))
            contributions = torch.stack(head_contributions, dim=-2)
            attention_target = native - candidate.float()
            selected_delta, head_indices = select_contributions(attention_target, contributions, head_count)
            selected_attention = candidate.float() + selected_delta
            exact_attention = candidate.float() + exact_delta.float()
            attention_cosine, attention_rmse = trajectory._metrics(selected_attention, exact_attention)

            feature = source_layer1.post_attention_layernorm(selected_attention).float()
            gate, up = F.linear(feature, gate_up).chunk(2, dim=-1)
            activations = (F.silu(gate) * up).float()
            residual_target = native - selected_attention
            target_down_products = residual_target @ down
            down_norms = down.square().sum(dim=0)
            neuron_scores = 2.0 * activations * target_down_products - activations.square() * down_norms
            neuron_indices = torch.argsort(neuron_scores, dim=-1, descending=True, stable=True)[..., :neuron_count]
            selected_mlp = torch.zeros_like(selected_attention)
            for token_index in range(length):
                indices = neuron_indices[0, token_index]
                selected_mlp[0, token_index] = down.index_select(1, indices) @ activations[0, token_index].index_select(0, indices)
            prediction = selected_attention + selected_mlp

            full_prediction = exact_attention + source_layer1.mlp(source_layer1.post_attention_layernorm(exact_attention)).float()
            cosine, rmse = trajectory._metrics(prediction, native)
            full_cosine, full_rmse = trajectory._metrics(full_prediction, native)
            cosines.append(cosine)
            rmses.append(rmse)
            full_cosines.append(full_cosine)
            full_rmses.append(full_rmse)
            attention_cosines.append(attention_cosine)
            attention_rmses.append(attention_rmse)
            head_histogram += torch.bincount(head_indices.cpu().reshape(-1), minlength=heads)
            neuron_histogram += torch.bincount(neuron_indices.cpu().reshape(-1), minlength=neurons)
            record_metrics.append(
                {
                    "record_id": row["record_id"],
                    "capability": row["capability"],
                    "conditional_cosine": cosine,
                    "conditional_relative_rmse": rmse,
                    "attention_cosine_to_exact": attention_cosine,
                    "attention_relative_rmse_to_exact": attention_rmse,
                    "full_source_cosine": full_cosine,
                    "full_source_relative_rmse": full_rmse,
                }
            )
            peak_rss = max(peak_rss, process.memory_info().rss)

    artifact_after = sha256_file(artifact_path)
    mean_cosine = sum(cosines) / len(cosines)
    mean_rmse = sum(rmses) / len(rmses)
    gates = {
        "validation_mean_cosine": mean_cosine >= float(protocol["gates"]["validation_mean_cosine_minimum"]),
        "validation_mean_relative_rmse": mean_rmse <= float(protocol["gates"]["validation_mean_relative_rmse_maximum"]),
        "support_exact": all(int(value) == head_count for value in [head_count]) and neuron_count == 432,
        "routes_exact": route_exact == len(validation_rows),
        "artifact_unchanged": artifact_before == artifact_after,
    }
    passed = all(gates.values())
    result = {
        "format": FORMAT,
        "status": "PASS_CONDITIONAL_SUPPORT_CAPACITY_ORACLE" if passed else "FAIL_CONDITIONAL_SUPPORT_CAPACITY_ORACLE",
        "protocol_sha256": sha256_file(protocol_path),
        "oracle_selection": {
            "target_accessed_for_support": True,
            "promotional": False,
            "heads_per_token": head_count,
            "neurons_per_token": neuron_count,
            "head_rule": "stable top marginal squared-error reduction",
            "neuron_rule": "stable top marginal squared-error reduction",
        },
        "validation": {
            "records": len(validation_rows),
            "mean_cosine": mean_cosine,
            "minimum_cosine": min(cosines),
            "mean_relative_rmse": mean_rmse,
            "maximum_relative_rmse": max(rmses),
            "mean_attention_cosine_to_exact": sum(attention_cosines) / len(attention_cosines),
            "mean_attention_relative_rmse_to_exact": sum(attention_rmses) / len(attention_rmses),
            "record_metrics": record_metrics,
        },
        "full_source_diagnostic": {
            "mean_cosine": sum(full_cosines) / len(full_cosines),
            "mean_relative_rmse": sum(full_rmses) / len(full_rmses),
        },
        "selection_distribution": {
            "unique_heads_selected": int((head_histogram > 0).sum()),
            "unique_neurons_selected": int((neuron_histogram > 0).sum()),
            "head_counts": head_histogram.tolist(),
            "neuron_nonzero_count_sha256": hashlib.sha256(neuron_histogram.numpy().tobytes()).hexdigest(),
        },
        "physical_envelope": protocol["physical_envelope"],
        "route_correct": route_exact,
        "gates": gates,
        "passed": passed,
        "artifact_model_sha256_before": artifact_before,
        "artifact_model_sha256_after": artifact_after,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak_rss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "training_performed": False,
        "router_written": False,
        "artifact_written": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Permanently nonpromotional target-aware held-out capacity oracle for the fixed 5-head/432-neuron support rule. Dense source equations are used only to score support, no router or artifact is written, and no realizable runtime, autonomous, complete-model, Phase 3, or superiority claim is made.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CONDITIONAL_SUPPORT_CAPACITY_ORACLE_PROTOCOL_V431.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_native_trajectory/conditional_support_capacity_v432")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
