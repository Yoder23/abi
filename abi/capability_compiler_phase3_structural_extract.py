"""GPU structural extraction of a compact Phi-compatible causal substrate."""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import platform
import time
from typing import Any, Mapping

import psutil
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-structural-extraction/1"


def _tensor(snapshot: Path, weight_map: Mapping[str, str], key: str) -> torch.Tensor:
    relative = weight_map.get(key)
    if relative is None:
        raise Phase3Error(f"source index lacks required tensor: {key}")
    with safe_open(str(snapshot / relative), framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def stable_top_indices(scores: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    if scores.ndim != 1 or count <= 0 or count >= scores.numel():
        raise Phase3Error("invalid structural selection inventory")
    ranking = torch.argsort(scores, descending=True, stable=True)
    selected_ranked = ranking[:count]
    ordered = selected_ranked.sort().values
    margin = float((scores[ranking[count - 1]] - scores[ranking[count]]).item())
    return selected_ranked, ordered, margin


def target_qkv(source: torch.Tensor, residual: torch.Tensor, heads: torch.Tensor, *, source_width: int, head_dim: int, scale: float) -> torch.Tensor:
    rows = []
    for section in range(3):
        rows.extend(range(section * source_width + int(head) * head_dim, section * source_width + (int(head) + 1) * head_dim) for head in heads)
    flat_rows = torch.tensor([value for group in rows for value in group], dtype=torch.long, device=source.device)
    return source.index_select(0, flat_rows).index_select(1, residual) * scale


def target_o(source: torch.Tensor, residual: torch.Tensor, heads: torch.Tensor, *, head_dim: int, scale: float) -> torch.Tensor:
    columns = torch.tensor([value for head in heads for value in range(int(head) * head_dim, (int(head) + 1) * head_dim)], dtype=torch.long, device=source.device)
    return source.index_select(0, residual).index_select(1, columns) * scale


def target_gate_up(source: torch.Tensor, residual: torch.Tensor, neurons: torch.Tensor, *, source_intermediate: int, scale: float) -> torch.Tensor:
    rows = torch.cat((neurons, neurons + source_intermediate))
    return source.index_select(0, rows).index_select(1, residual) * scale


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_GPU_STRUCTURAL_EXTRACTION"
        or protocol.get("device") != "cuda"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("structural extraction governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"structural extraction binding changed: {name}")
    return protocol, sha256_file(path)


def execute(root: Path, protocol_path: Path, output_directory: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output_directory.exists():
        raise Phase3Error("structural extraction output already exists")
    if not torch.cuda.is_available():
        raise Phase3Error("preregistered CUDA device unavailable")
    source = protocol["source"]
    target = protocol["target"]
    selection = protocol["selection"]
    transforms = protocol["transforms"]
    snapshot = Path(source["snapshot_path"])
    weight_map = json.loads(Path(source["index_path"]).read_text(encoding="utf-8"))["weight_map"]
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss

    external_rows = torch.arange(int(target["external_actions"]), dtype=torch.long, device=device)
    residual_scores = torch.zeros(int(source["hidden_size"]), dtype=torch.float64, device=device)
    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        table = _tensor(snapshot, weight_map, key).to(device)
        residual_scores += table.index_select(0, external_rows).double().square().sum(dim=0)
        del table
        peak_rss = max(peak_rss, process.memory_info().rss)
    ranked_residual, residual, residual_margin = stable_top_indices(residual_scores, int(target["width"]))

    source_width = int(source["hidden_size"])
    source_intermediate = int(source["intermediate_size"])
    head_dim = int(source["head_dim"])
    selected_heads: dict[int, torch.Tensor] = {}
    selected_neurons: dict[int, torch.Tensor] = {}
    selection_margins: dict[str, float] = {"residual": residual_margin}
    for layer in target["source_layers"]:
        layer = int(layer)
        qkv = _tensor(snapshot, weight_map, f"model.layers.{layer}.self_attn.qkv_proj.weight").to(device)
        o = _tensor(snapshot, weight_map, f"model.layers.{layer}.self_attn.o_proj.weight").to(device)
        head_scores = []
        for head in range(int(source["num_attention_heads"])):
            row_groups = [qkv[section * source_width + head * head_dim:section * source_width + (head + 1) * head_dim].index_select(1, residual) for section in range(3)]
            o_slice = o.index_select(0, residual)[:, head * head_dim:(head + 1) * head_dim]
            head_scores.append(sum(value.double().square().sum() for value in row_groups) + o_slice.double().square().sum())
        head_score_tensor = torch.stack(head_scores)
        _, heads, margin = stable_top_indices(head_score_tensor, int(target["num_attention_heads"]))
        selected_heads[layer] = heads
        selection_margins[f"layer_{layer}_heads"] = margin
        del qkv, o, head_score_tensor

        gate_up = _tensor(snapshot, weight_map, f"model.layers.{layer}.mlp.gate_up_proj.weight").to(device)
        down = _tensor(snapshot, weight_map, f"model.layers.{layer}.mlp.down_proj.weight").to(device)
        gate = gate_up[:source_intermediate].index_select(1, residual).double().square().sum(dim=1)
        up = gate_up[source_intermediate:].index_select(1, residual).double().square().sum(dim=1)
        down_score = down.index_select(0, residual).double().square().sum(dim=0)
        _, neurons, margin = stable_top_indices(gate + up + down_score, int(target["intermediate_size"]))
        selected_neurons[layer] = neurons
        selection_margins[f"layer_{layer}_neurons"] = margin
        del gate_up, down, gate, up, down_score
        peak_rss = max(peak_rss, process.memory_info().rss)

    tensors: dict[str, torch.Tensor] = {}
    special_source_rows = torch.tensor(selection["host_special_source_rows"], dtype=torch.long, device=device)
    host_source_rows = torch.cat((special_source_rows, external_rows))
    embedding = _tensor(snapshot, weight_map, "model.embed_tokens.weight").to(device)
    tensors["token_embedding.weight"] = embedding.index_select(0, host_source_rows).index_select(1, residual)
    del embedding
    lm_head = _tensor(snapshot, weight_map, "lm_head.weight").to(device)
    tensors["lm_head.weight"] = lm_head.index_select(0, host_source_rows).index_select(1, residual) * float(transforms["lm_head_scale"])
    del lm_head

    for target_layer, source_layer in enumerate(target["source_layers"]):
        source_layer = int(source_layer)
        prefix = f"layers.{target_layer}"
        qkv = _tensor(snapshot, weight_map, f"model.layers.{source_layer}.self_attn.qkv_proj.weight").to(device)
        tensors[f"{prefix}.qkv_proj.weight"] = target_qkv(qkv, residual, selected_heads[source_layer], source_width=source_width, head_dim=head_dim, scale=float(transforms["qkv_scale"]))
        del qkv
        o = _tensor(snapshot, weight_map, f"model.layers.{source_layer}.self_attn.o_proj.weight").to(device)
        tensors[f"{prefix}.o_proj.weight"] = target_o(o, residual, selected_heads[source_layer], head_dim=head_dim, scale=float(transforms["o_scale"]))
        del o
        gate_up = _tensor(snapshot, weight_map, f"model.layers.{source_layer}.mlp.gate_up_proj.weight").to(device)
        tensors[f"{prefix}.gate_up_proj.weight"] = target_gate_up(gate_up, residual, selected_neurons[source_layer], source_intermediate=source_intermediate, scale=float(transforms["gate_up_scale"]))
        del gate_up
        down = _tensor(snapshot, weight_map, f"model.layers.{source_layer}.mlp.down_proj.weight").to(device)
        tensors[f"{prefix}.down_proj.weight"] = down.index_select(0, residual).index_select(1, selected_neurons[source_layer]) * float(transforms["down_scale"])
        del down
        for source_name, target_name in (("input_layernorm", "input_norm"), ("post_attention_layernorm", "post_attention_norm")):
            norm = _tensor(snapshot, weight_map, f"model.layers.{source_layer}.{source_name}.weight").to(device)
            tensors[f"{prefix}.{target_name}.weight"] = norm.index_select(0, residual)
            del norm
    final_norm = _tensor(snapshot, weight_map, "model.norm.weight").to(device)
    tensors["final_norm.weight"] = final_norm.index_select(0, residual)
    del final_norm

    torch.cuda.synchronize()
    extraction_seconds = time.perf_counter() - started
    peak_cuda = torch.cuda.max_memory_allocated()
    peak_rss = max(peak_rss, process.memory_info().rss)
    serializable = {key: value.to(torch.float16).cpu().contiguous() for key, value in tensors.items()}
    scalar_count = sum(value.numel() for value in serializable.values())
    if scalar_count != int(target["deployed_parameters"]):
        raise Phase3Error(f"structural scalar accounting changed: {scalar_count}")
    exact_lexical_entries = int(target["vocabulary"]) * int(target["width"])
    exact_norm_entries = (2 * len(target["source_layers"]) + 1) * int(target["width"])
    exact_source_scalar_entries = exact_lexical_entries + exact_norm_entries
    duplicate_special_entries = len(selection["host_special_source_rows"]) * int(target["width"])
    output_directory.mkdir(parents=True)
    tensor_path = output_directory / "structural_core.safetensors"
    save_file(serializable, str(tensor_path), metadata={"format": "abi-structural-core/1", "protocol_sha256": protocol_sha})
    result = {
        "format": FORMAT,
        "status": "COMPLETE_UNVERIFIED",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "source": {"model": source["model"], "revision": source["revision"], "complete_blocks_retained": 0},
        "selection": {
            "residual_ranked": ranked_residual.cpu().tolist(),
            "residual_ordered": residual.cpu().tolist(),
            "attention_heads": {str(key): value.cpu().tolist() for key, value in selected_heads.items()},
            "mlp_neurons": {str(key): value.cpu().tolist() for key, value in selected_neurons.items()},
            "boundary_score_margins": selection_margins,
        },
        "accounting": {
            "source_parameters": int(source["parameter_count"]),
            "final_imported_substrate_parameters": scalar_count,
            "source_derived_parameter_scalars": scalar_count,
            "exact_source_scalar_entries_in_deployed_artifact": exact_source_scalar_entries,
            "unique_exact_source_scalar_coordinates": exact_source_scalar_entries - duplicate_special_entries,
            "transformed_source_derived_scalar_entries": scalar_count - exact_source_scalar_entries,
            "duplicate_special_scalar_entries": duplicate_special_entries,
            "complete_source_blocks_retained": 0,
            "stored_logits": 0,
            "stored_activations": 0,
            "teacher_forward_tokens": 0,
            "teacher_inference_seconds": 0.0,
            "tensor_file_bytes": tensor_path.stat().st_size,
            "extraction_seconds": extraction_seconds,
            "peak_cuda_allocated_bytes": peak_cuda,
            "peak_process_rss_bytes": peak_rss,
            "external_hardware_used": True,
            "device": torch.cuda.get_device_name(0),
        },
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in serializable.items()},
        "tensor_sha256": sha256_file(tensor_path),
        "teacher_present_at_inference": False,
        "training_performed": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "next_gate": "Hostile independent recomputation of selections, transforms, tensor values, payload identity, and accounting.",
        "claim_boundary": "Unverified structural extraction only; no quality, transfer, or superiority claim.",
    }
    result_path = output_directory / "result.json"
    _write_immutable(result_path, json.dumps(result, indent=2, sort_keys=True) .encode("utf-8") + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_STRUCTURAL_EXTRACTION_PROTOCOL_V193.json")
    parser.add_argument("--output-directory", default="results/abi_capability_compiler_phase3_structural/extraction_v194")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, (root / args.protocol).resolve(), (root / args.output_directory).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
