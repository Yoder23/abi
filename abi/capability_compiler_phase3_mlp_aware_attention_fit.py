"""Bounded MLP-aware fine-tune of the existing compact layer-1 attention."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from safetensors.torch import load_file, save_file
import torch
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
FORMAT = "abi-capability-compiler-phase3-mlp-aware-attention-fit/1"

def execute(root: Path, path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_BOUNDED_MLP_AWARE_ATTENTION_FIT" or protocol.get("device") != "cuda" or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("MLP-aware attention governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"MLP-aware binding changed: {name}")
    if output.exists() or not torch.cuda.is_available(): raise Phase3Error("output exists or CUDA unavailable")
    output.mkdir(parents=True); base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8")); device = torch.device("cuda"); model, tokenizer, _, attention_keys, _ = sequential._model(root, base, device); state = model.state_dict()
    for layer_index in (0, 1):
        checkpoint = load_file(str(root / protocol["checkpoints"][str(layer_index)]["path"]), device="cpu")
        for name, value in checkpoint.items():
            if name in state: state[name].copy_(value.to(state[name].dtype))
    model.eval(); layer_index = 1; layer = model.layers[layer_index]
    examples = sequential.field._examples(root, base, tokenizer); cfg = base["calibration"]; train_rows, validation_rows, tokens = dual._calibration_examples(examples, seed=int(base["training"]["seed"]), train_per_capability=int(cfg["train_records_per_capability"]), validation_per_capability=int(cfg["validation_records_per_capability"]), maximum_tokens=int(cfg["maximum_sequence_tokens"]))
    teacher = AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval(); source_layer = teacher.model.layers[layer_index]
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    current_prefix = f"layers.{layer_index}."
    for name, parameter in model.named_parameters(): parameter.requires_grad_(name.startswith(current_prefix) and name in attention_keys)
    parameters = [parameter for parameter in layer.parameters() if parameter.requires_grad]; optimizer = torch.optim.AdamW(parameters, lr=float(protocol["training"]["learning_rate"]), betas=(0.9, 0.95), weight_decay=float(protocol["training"]["weight_decay"])); curves = []; steps = int(protocol["training"]["steps"]); layer.train()
    for step in range(steps):
        row = train_rows[(step + 256) % len(train_rows)]; ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = dual.base._prefix_hidden(model, ids, layer_index); attention_target, final_target = dual._teacher_components(teacher, layer_index, hidden); feature_target = source_layer.post_attention_layernorm(attention_target)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            attention = sequential._student_attention(layer, hidden, torch.arange(ids.shape[1], device=device)); feature = layer.post_attention_norm(attention); final = attention + source_layer.mlp(feature); attention_rmse, attention_cosine = dual.base._metrics(attention, attention_target, hidden); final_rmse, final_cosine = dual.base._metrics(final, final_target, hidden); feature_rmse = torch.sqrt((feature.float() - feature_target.float()).square().mean() / feature_target.float().square().mean().clamp_min(1e-8)); loss = attention_rmse.square() + final_rmse.square() + feature_rmse.square() + float(protocol["training"]["cosine_weight"]) * (2.0 - attention_cosine - final_cosine)
        loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, float(protocol["training"]["gradient_clip_norm"])); optimizer.step()
        if step == 0 or (step + 1) % int(protocol["training"]["curve_interval"]) == 0: curves.append({"step": step + 1, "attention_relative_rmse": float(attention_rmse.detach()), "feature_relative_rmse": float(feature_rmse.detach()), "final_relative_rmse": float(final_rmse.detach()), "final_cosine": float(final_cosine.detach()), "loss": float(loss.detach())})
    layer.eval(); attention_rmses = []; feature_rmses = []; rmses = []; cosines = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device); hidden = dual.base._prefix_hidden(model, ids, layer_index); attention_target, final_target = dual._teacher_components(teacher, layer_index, hidden); feature_target = source_layer.post_attention_layernorm(attention_target); attention = sequential._student_attention(layer, hidden, torch.arange(ids.shape[1], device=device)); feature = layer.post_attention_norm(attention); final = attention + source_layer.mlp(feature); armse, _ = dual.base._metrics(attention, attention_target, hidden); rmse, cosine = dual.base._metrics(final, final_target, hidden); frmse = torch.sqrt((feature.float() - feature_target.float()).square().mean() / feature_target.float().square().mean().clamp_min(1e-8)); attention_rmses.append(float(armse)); feature_rmses.append(float(frmse)); rmses.append(float(rmse)); cosines.append(float(cosine))
    mr = sum(rmses) / len(rmses); mc = sum(cosines) / len(cosines); gate = protocol["gate"]; passed = mr <= gate["mean_relative_rmse_maximum"] and mc >= gate["mean_output_cosine_minimum"]
    checkpoint = {name: parameter.detach().to(torch.float16).cpu().contiguous() for name, parameter in model.named_parameters() if name.startswith(current_prefix) and name in attention_keys}; checkpoint_path = output / "layer1_attention_weights.safetensors"; save_file(checkpoint, str(checkpoint_path), metadata={"format": FORMAT, "protocol_sha256": sha256_file(path)})
    result = {"format": FORMAT, "status": "PASS_MLP_AWARE_ATTENTION_INTERFACE" if passed else "FAIL_MLP_AWARE_ATTENTION_INTERFACE", "protocol_sha256": sha256_file(path), "layer": 1, "steps": steps, "curves": curves, "validation": {"mean_attention_relative_rmse": sum(attention_rmses) / len(attention_rmses), "mean_feature_relative_rmse": sum(feature_rmses) / len(feature_rmses), "mean_relative_rmse": mr, "maximum_relative_rmse": max(rmses), "mean_output_cosine": mc, "minimum_output_cosine": min(cosines), "passed": passed}, "checkpoint": {"path": checkpoint_path.name, "sha256": sha256_file(checkpoint_path), "parameters": sum(value.numel() for value in checkpoint.values())}, "complete_source_mlp_promoted": False, "artifact_promoted": False, "training_performed": True, "final_test_accessed": False, "phase3_certified": False, "claim_boundary": "Bounded layer-1 MLP-aware attention interface fit only; no deployable artifact, English quality, runtime, certificate, or superiority claim."}; _write_immutable(output / "metadata.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_MLP_AWARE_ATTENTION_FIT_PROTOCOL_V277.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_mlp_aware_attention/layer1_fit_v278"); args = parser.parse_args(); root = Path.cwd().resolve(); result = execute(root, root / args.protocol, root / args.output); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"].startswith("PASS") else 1
if __name__ == "__main__": raise SystemExit(main())
