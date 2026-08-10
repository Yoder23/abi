"""One bounded source-aligned coefficient repair for routed v15 layer 2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_closed_form_coefficient_audit as closed
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_routed_v15_layer0_extract as layer0
from . import capability_compiler_phase3_routed_v15_progressive_extract as progressive
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-routed-v15-layer2-source-aligned-repair/2"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_ONE_IDENTITY_BOUND_SOURCE_ALIGNED_LAYER2_REPAIR"
        or protocol.get("device") != "cuda"
        or protocol.get("gradient_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("layer2 source-aligned repair governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"layer2 source-aligned binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("output exists or CUDA unavailable")
    output.mkdir(parents=True)
    extraction, extraction_sha = progressive._load_protocol(
        root, root / protocol["extraction_protocol"]
    )
    if extraction_sha != protocol["extraction_protocol_sha256"]:
        raise Phase3Error("extraction protocol identity changed")
    device = torch.device("cuda")
    set_determinism(int(extraction["training"]["seed"]))
    model, tokenizer, base, _, _ = progressive._instantiate(root, extraction, device)
    state = model.state_dict()
    failed = load_file(str(root / protocol["failed_checkpoint"]["path"]), device="cpu")
    expected = {
        name for name, _ in model.named_parameters() if name.startswith("layers.2.")
    }
    if set(failed) != expected:
        raise Phase3Error("failed layer2 checkpoint boundary changed")
    with torch.no_grad():
        for name, value in failed.items():
            state[name].copy_(value.to(state[name].dtype))
    model.eval()
    examples = progressive.sequential.field._examples(root, base, tokenizer)
    example_by_id = {str(row["record_id"]): row for row in examples}
    cfg = base["calibration"]
    train_rows, validation_rows, calibration_tokens = dual._calibration_examples(
        examples,
        seed=int(base["training"]["seed"]),
        train_per_capability=int(cfg["train_records_per_capability"]),
        validation_per_capability=int(cfg["validation_records_per_capability"]),
        maximum_tokens=int(cfg["maximum_sequence_tokens"]),
    )
    all_rows = train_rows + validation_rows
    cache = progressive._initial_cache(model, all_rows, example_by_id, device)
    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in teacher.parameters()) != int(
        base["source"]["parameter_count"]
    ):
        raise Phase3Error("loaded source parameter count changed")
    targets = progressive._targets(teacher, 2, all_rows, cache, device)
    layer = model.layers[2]
    source_layer = teacher.model.layers[2]
    source_gate_up = source_layer.mlp.gate_up_proj.weight.float()
    source_down = source_layer.mlp.down_proj.weight.float()
    source_neurons = source_down.shape[1]
    identity = json.loads((root / protocol["identity_manifest"]["path"]).read_text(encoding="utf-8"))
    recovered = identity.get("ordered_source_neuron_indices")
    sparse_width = int(extraction["architecture"]["sparse_width"])
    if (
        identity.get("format") != "abi-capability-compiler-source-neuron-identity/1"
        or identity.get("source_layer") != 2
        or identity.get("canonical_dtype") != "float16"
        or not isinstance(recovered, list)
        or len(recovered) != sparse_width
        or len(set(recovered)) != sparse_width
        or any(not isinstance(value, int) or not 0 <= value < source_neurons for value in recovered)
    ):
        raise Phase3Error("source-neuron identity manifest changed")
    selected = torch.tensor(recovered, dtype=torch.long, device=device)
    selected_gate = source_gate_up[:source_neurons].index_select(0, selected)
    selected_up = source_gate_up[source_neurons:].index_select(0, selected)
    expected_sparse = torch.cat((selected_gate, selected_up), dim=0).to(torch.float16)
    stored_sparse = layer.sparse_gate_up_projection.weight.detach().to(torch.float16)
    selected_neurons_reproduced = bool(torch.equal(expected_sparse, stored_sparse))
    if not selected_neurons_reproduced:
        raise Phase3Error("identity-bound source neurons do not match the failed checkpoint")
    basis = layer.mlp_output_projection.weight.float()
    exact_sparse_coefficients = source_down.index_select(1, selected).T @ basis
    features_cpu = []
    remaining_cpu = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            record_id = str(row["record_id"])
            hidden = cache[record_id][0].unsqueeze(0).to(device)
            positions = torch.arange(hidden.shape[1], device=device)
            attention = layer0._attention(layer, hidden, positions)
            feature = layer.post_attention_norm(attention)
            gate = F.linear(feature.float(), selected_gate)
            up = F.linear(feature.float(), selected_up)
            selected_output = (F.silu(gate) * up) @ source_down.index_select(1, selected).T
            target_residual = targets[record_id][1].unsqueeze(0).to(device).float() - attention.float()
            features_cpu.append(feature.squeeze(0).float().cpu())
            remaining_cpu.append((target_residual - selected_output).squeeze(0).cpu())
    remaining = torch.cat(remaining_cpu).to(device)
    mean = remaining.mean(dim=0)
    features = torch.cat(features_cpu).to(device)
    coefficients = (remaining - mean) @ basis
    linear_weights, effective_ridge = closed.solve_ridge(
        features, coefficients, float(protocol["relative_ridge"])
    )
    with torch.no_grad():
        layer.mlp_residual_mean.copy_(mean)
        layer.linear_coefficient_projection.weight.copy_(linear_weights.T)
        for route_projection in layer.route_coefficient_projections:
            route_projection.weight.copy_(exact_sparse_coefficients.T)
    validation, records = progressive._validate(
        layer, validation_rows, cache, targets, extraction, device
    )
    gate = protocol["gate"]
    passed = (
        validation["mean_relative_rmse"] <= float(gate["mean_relative_rmse_maximum"])
        and validation["mean_output_cosine"] >= float(gate["mean_output_cosine_minimum"])
        and validation["exact_routes"] == int(gate["exact_validation_routes_required"])
    )
    prefix = "layers.2."
    checkpoint = {
        name: parameter.detach().to(torch.float16).cpu().contiguous()
        for name, parameter in model.named_parameters()
        if name.startswith(prefix)
    }
    checkpoint_path = output / "routed_v15_layer_02_source_aligned.safetensors"
    save_file(
        checkpoint,
        str(checkpoint_path),
        metadata={"format": FORMAT, "protocol_sha256": sha256_file(protocol_path)},
    )
    result = {
        "format": FORMAT,
        "status": "PASS_IDENTITY_BOUND_LAYER2_REPAIR" if passed else "FAIL_IDENTITY_BOUND_LAYER2_REPAIR",
        "protocol_sha256": sha256_file(protocol_path),
        "extraction_protocol_sha256": extraction_sha,
        "layer": 2,
        "calibration_tokens": calibration_tokens,
        "selected_neurons_reproduced": selected_neurons_reproduced,
        "selected_neurons": int(selected.numel()),
        "identity_manifest_sha256": sha256_file(root / protocol["identity_manifest"]["path"]),
        "basis_rank_unchanged": int(basis.shape[1]),
        "effective_ridge": effective_ridge,
        "route_maps_identical_by_construction": True,
        "validation": validation,
        "record_metrics": records,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "parameters": sum(value.numel() for value in checkpoint.values()),
        },
        "source_blocks_in_checkpoint": 0,
        "gradient_training_performed": False,
        "analytic_fit_performed": True,
        "artifact_promoted": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "One identity-bound source-aligned analytic layer2 coefficient repair only; no assembled artifact, English quality, runtime, certificate, or superiority claim.",
    }
    _write_immutable(
        output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V15_LAYER2_IDENTITY_BOUND_REPAIR_PROTOCOL_V317.json",
    )
    parser.add_argument(
        "--output", default="results/abi_capability_compiler_phase3_routed_v15/layer2_identity_repair_v318"
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    result = execute(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
