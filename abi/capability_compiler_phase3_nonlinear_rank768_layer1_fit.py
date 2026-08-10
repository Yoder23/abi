"""Bounded layer-1 fit of the v14 nonlinear rank-768 coefficient path."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from .capability_compiler_phase2_common import set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable

FORMAT = "abi-capability-compiler-phase3-nonlinear-rank768-layer1-fit/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_BOUNDED_LAYER1_GPU_FIT" or protocol.get("device") != "cuda" or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("nonlinear layer1 governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"nonlinear layer1 binding changed: {name}")
    if output.exists() or not torch.cuda.is_available(): raise Phase3Error("output exists or CUDA unavailable")
    output.mkdir(parents=True); device = torch.device("cuda"); base_protocol = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    prefix_model, tokenizer, _, _, _ = sequential._model(root, base_protocol, device); state = prefix_model.state_dict()
    for layer_index in (0, 1):
        checkpoint = load_file(str(root / protocol["checkpoints"][str(layer_index)]["path"]), device="cpu")
        for name, value in checkpoint.items():
            if name in state: state[name].copy_(value.to(state[name].dtype))
    prefix_model.eval(); layer_index = 1
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve(); sys.path.insert(0, str(layercake_root))
    from layercake.nonlinear_rank768_progressive_core import NonlinearRank768ProgressiveLayer
    set_determinism(int(protocol["training"]["seed"]))
    layer = NonlinearRank768ProgressiveLayer(3072, 192, 2, 768, residual_rank=768, nonlinear_hidden=384, rms_epsilon=1e-5, rope_theta=10000.0)
    source_layer = prefix_model.layers[layer_index]
    source_state = source_layer.state_dict(); target_state = layer.state_dict()
    copied_names = []
    with torch.no_grad():
        for name in target_state:
            if name in source_state and target_state[name].shape == source_state[name].shape and not name.startswith("mlp_"):
                target_state[name].copy_(source_state[name].to(target_state[name].dtype)); copied_names.append(name)
        layer.mlp_output_projection.weight.zero_(); layer.mlp_residual_mean.zero_()
    layer.to(device)
    examples = sequential.field._examples(root, base_protocol, tokenizer); cfg = base_protocol["calibration"]
    train_rows, validation_rows, tokens = dual._calibration_examples(examples, seed=int(protocol["training"]["seed"]), train_per_capability=int(cfg["train_records_per_capability"]), validation_per_capability=int(cfg["validation_records_per_capability"]), maximum_tokens=int(cfg["maximum_sequence_tokens"]))
    teacher = AutoModelForCausalLM.from_pretrained(base_protocol["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    deltas = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device); hidden = dual.base._prefix_hidden(prefix_model, ids, layer_index)
            attention_target, final_target = dual._teacher_components(teacher, layer_index, hidden); deltas.append((final_target - attention_target).squeeze(0).float().cpu())
    mean, covariance, observations = rank_audit.centered_covariance(deltas, 3072, device); eigenvalues, eigenvectors = torch.linalg.eigh(covariance); eigenvalues = eigenvalues.clamp_min(0).flip(0); basis = eigenvectors.flip(1)[:, :768].contiguous()
    with torch.no_grad(): layer.mlp_residual_mean.copy_(mean); layer.mlp_output_projection.weight.copy_(basis)
    trainable_names = {"mlp_gate_up_projection.weight", "mlp_coefficient_projection.weight"}
    for name, parameter in layer.named_parameters(): parameter.requires_grad_(name in trainable_names)
    parameters = [parameter for parameter in layer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(protocol["training"]["learning_rate"]), betas=(0.9, 0.95), weight_decay=float(protocol["training"]["weight_decay"]))
    curves = []; steps = int(protocol["training"]["steps"]); layer.train()
    for step in range(steps):
        row = train_rows[(step + 256) % len(train_rows)]; ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = dual.base._prefix_hidden(prefix_model, ids, layer_index); attention_target, final_target = dual._teacher_components(teacher, layer_index, hidden); attention = sequential._student_attention(source_layer, hidden, torch.arange(ids.shape[1], device=device)); coefficient_target = (final_target.float() - attention_target.float() - mean) @ basis
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gate, up = layer.mlp_gate_up_projection(layer.post_attention_norm(attention)).chunk(2, dim=-1); coefficients = layer.mlp_coefficient_projection(F.silu(gate) * up); final = attention.float() + mean + coefficients.float() @ basis.T
            coefficient_rmse = torch.sqrt((coefficients.float() - coefficient_target).square().mean() / coefficient_target.square().mean().clamp_min(1e-8)); final_rmse, final_cosine = dual.base._metrics(final, final_target.float(), hidden.float()); loss = coefficient_rmse.square() + final_rmse.square() + float(protocol["training"]["cosine_weight"]) * (1.0 - final_cosine)
        loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, float(protocol["training"]["gradient_clip_norm"])); optimizer.step()
        if step == 0 or (step + 1) % int(protocol["training"]["curve_interval"]) == 0: curves.append({"step": step + 1, "coefficient_relative_rmse": float(coefficient_rmse.detach()), "final_relative_rmse": float(final_rmse.detach()), "final_cosine": float(final_cosine.detach()), "loss": float(loss.detach())})
    layer.eval(); rmses = []; cosines = []; coefficient_rmses = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device); hidden = dual.base._prefix_hidden(prefix_model, ids, layer_index); attention_target, final_target = dual._teacher_components(teacher, layer_index, hidden); attention = sequential._student_attention(source_layer, hidden, torch.arange(ids.shape[1], device=device)); target = (final_target.float() - attention_target.float() - mean) @ basis; gate, up = layer.mlp_gate_up_projection(layer.post_attention_norm(attention)).chunk(2, dim=-1); coefficients = layer.mlp_coefficient_projection(F.silu(gate) * up); final = attention.float() + mean + coefficients.float() @ basis.T; rmse, cosine = dual.base._metrics(final, final_target.float(), hidden.float()); rmses.append(float(rmse)); cosines.append(float(cosine)); coefficient_rmses.append(float(torch.sqrt((coefficients.float() - target).square().mean() / target.square().mean().clamp_min(1e-8))))
    mean_rmse = sum(rmses) / len(rmses); mean_cosine = sum(cosines) / len(cosines); gate_cfg = protocol["gate"]; passed = mean_rmse <= gate_cfg["mean_relative_rmse_maximum"] and mean_cosine >= gate_cfg["mean_output_cosine_minimum"]
    checkpoint = {name: parameter.detach().to(torch.float16).cpu().contiguous() for name, parameter in layer.named_parameters() if name.startswith("mlp_")}; checkpoint_path = output / "layer1_nonlinear_weights.safetensors"; save_file(checkpoint, str(checkpoint_path), metadata={"format": FORMAT, "protocol_sha256": sha256_file(protocol_path)})
    result = {"format": FORMAT, "status": "PASS_LAYER1_NONLINEAR_MAP" if passed else "FAIL_LAYER1_NONLINEAR_MAP", "protocol_sha256": sha256_file(protocol_path), "layer": 1, "steps": steps, "basis_rank": 768, "basis_energy_explained": float(eigenvalues[:768].sum() / eigenvalues.sum().clamp_min(1e-12)), "train_observations": observations, "curves": curves, "validation": {"mean_coefficient_relative_rmse": sum(coefficient_rmses) / len(coefficient_rmses), "mean_relative_rmse": mean_rmse, "maximum_relative_rmse": max(rmses), "mean_output_cosine": mean_cosine, "minimum_output_cosine": min(cosines), "passed": passed}, "checkpoint": {"path": checkpoint_path.name, "sha256": sha256_file(checkpoint_path), "parameters": sum(value.numel() for value in checkpoint.values())}, "copied_attention_tensor_keys": len(copied_names), "artifact_promoted": False, "teacher_required_at_inference": False, "training_performed": True, "final_test_accessed": False, "phase3_certified": False, "claim_boundary": "Bounded layer-1 nonlinear map fit only; no 32-layer artifact, autonomous English quality, runtime, certificate, or superiority claim."}
    _write_immutable(output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_NONLINEAR_RANK768_LAYER1_FIT_PROTOCOL_V269.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_nonlinear_rank768/layer1_fit_v270"); args = parser.parse_args(); root = Path.cwd().resolve(); result = execute(root, root / args.protocol, root / args.output); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"].startswith("PASS") else 1
if __name__ == "__main__": raise SystemExit(main())
