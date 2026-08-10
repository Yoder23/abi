"""Read-only audit of nested matched source-MLP neuron subsets."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from safetensors.torch import load_file
import torch
import torch.nn.functional as F
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable

FORMAT = "abi-capability-compiler-phase3-source-neuron-subset-audit/1"


def deployment_accounting(width: int, *, full: int = 3072, layers: int = 32) -> dict:
    copied = 196_899_840; attention = 42_473_472; residual_per_layer = 3 * full * width + full
    deployed = copied + attention + layers * residual_per_layer
    direct_active = 184_857_600; direct_residual_per_layer = 2 * full * 192
    active = direct_active + layers * (3 * full * width - direct_residual_per_layer)
    return {"selected_neurons": width, "imported_residual_parameters": layers * residual_per_layer, "deployed_parameters": deployed, "fp16_payload_bytes": 2 * deployed, "active_incremental_macs_at_maximum_context": active, "source_to_target_active_mac_ratio": 3_823_042_560 / active}


def execute(root: Path, protocol_path: Path) -> dict:
    from transformers import AutoModelForCausalLM
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_NESTED_SOURCE_NEURON_AUDIT" or protocol.get("training_authorized") is not False or protocol.get("artifact_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("source-neuron audit governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"source-neuron binding changed: {name}")
    if not torch.cuda.is_available(): raise Phase3Error("source-neuron audit requires CUDA")
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8")); device = torch.device("cuda")
    model, tokenizer, _, _, _ = sequential._model(root, base, device); state = model.state_dict()
    for layer_index in (0, 1):
        checkpoint = load_file(str(root / protocol["checkpoints"][str(layer_index)]["path"]), device="cpu")
        for name, value in checkpoint.items():
            if name in state: state[name].copy_(value.to(state[name].dtype))
    model.eval(); layer_index = 1; student_layer = model.layers[layer_index]
    examples = sequential.field._examples(root, base, tokenizer); cfg = base["calibration"]
    train_rows, validation_rows, tokens = dual._calibration_examples(examples, seed=int(base["training"]["seed"]), train_per_capability=int(cfg["train_records_per_capability"]), validation_per_capability=int(cfg["validation_records_per_capability"]), maximum_tokens=int(cfg["maximum_sequence_tokens"]))
    teacher = AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    teacher_layer = teacher.model.layers[layer_index]; gate_up_weight = teacher_layer.mlp.gate_up_proj.weight.float(); down_weight = teacher_layer.mlp.down_proj.weight.float(); neurons = down_weight.shape[1]
    if gate_up_weight.shape != (2 * neurons, 3072) or down_weight.shape[0] != 3072: raise Phase3Error("source MLP topology changed")
    importance = torch.zeros(neurons, device=device); observations = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device); hidden = dual.base._prefix_hidden(model, ids, layer_index); _, _ = dual._teacher_components(teacher, layer_index, hidden); student_attention = sequential._student_attention(student_layer, hidden, torch.arange(ids.shape[1], device=device)); normalized = student_layer.post_attention_norm(student_attention).float(); gate, up = F.linear(normalized, gate_up_weight).chunk(2, dim=-1); activation = F.silu(gate) * up; importance += activation.square().sum(dim=(0, 1)); observations += activation.shape[0] * activation.shape[1]
    importance *= down_weight.square().sum(dim=0)
    ordering = torch.argsort(importance, descending=True, stable=True); widths = [int(value) for value in protocol["widths"]]; selected = {width: ordering[:width] for width in widths}
    omitted_sums = {width: torch.zeros(3072, device=device) for width in widths}; omitted_count = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device); hidden = dual.base._prefix_hidden(model, ids, layer_index); teacher_attention, teacher_final = dual._teacher_components(teacher, layer_index, hidden); student_attention = sequential._student_attention(student_layer, hidden, torch.arange(ids.shape[1], device=device)); normalized = student_layer.post_attention_norm(student_attention).float(); gate, up = F.linear(normalized, gate_up_weight).chunk(2, dim=-1); activation = F.silu(gate) * up; target_delta = teacher_final.float() - teacher_attention.float()
            for width in widths:
                indices = selected[width]; contribution = F.linear(activation.index_select(-1, indices), down_weight.index_select(1, indices)); omitted_sums[width] += (target_delta - contribution).sum(dim=(0, 1))
            omitted_count += target_delta.shape[0] * target_delta.shape[1]
    means = {width: omitted_sums[width] / omitted_count for width in widths}; accumulators = {width: [[], []] for width in widths}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device); hidden = dual.base._prefix_hidden(model, ids, layer_index); _, teacher_final = dual._teacher_components(teacher, layer_index, hidden); student_attention = sequential._student_attention(student_layer, hidden, torch.arange(ids.shape[1], device=device)); normalized = student_layer.post_attention_norm(student_attention).float(); gate, up = F.linear(normalized, gate_up_weight).chunk(2, dim=-1); activation = F.silu(gate) * up
            for width in widths:
                indices = selected[width]; contribution = F.linear(activation.index_select(-1, indices), down_weight.index_select(1, indices)); prediction = student_attention.float() + means[width] + contribution; rmse, cosine = dual.base._metrics(prediction, teacher_final.float(), hidden.float()); accumulators[width][0].append(float(rmse)); accumulators[width][1].append(float(cosine))
    gate = protocol["gate"]; results = []
    for width in widths:
        rmses, cosines = accumulators[width]; mean_rmse = sum(rmses) / len(rmses); mean_cosine = sum(cosines) / len(cosines); accounting = deployment_accounting(width); passed = mean_rmse <= gate["mean_relative_rmse_maximum"] and mean_cosine >= gate["mean_output_cosine_minimum"] and accounting["source_to_target_active_mac_ratio"] >= protocol["minimum_theoretical_active_mac_ratio"]
        results.append({"selected_neurons": width, "mean_relative_rmse": mean_rmse, "maximum_relative_rmse": max(rmses), "mean_output_cosine": mean_cosine, "minimum_output_cosine": min(cosines), "local_and_compute_gate_passed": passed, "accounting": accounting})
    passing = [row for row in results if row["local_and_compute_gate_passed"]]; selected_width = passing[0]["selected_neurons"] if passing else None
    return {"format": FORMAT, "status": "PASS_SOURCE_NEURON_SUBSET_FEASIBLE_NO_ARTIFACT" if selected_width is not None else "FAIL_SOURCE_NEURON_SUBSET_AUDIT", "protocol_sha256": sha256_file(protocol_path), "layer": 1, "source_mlp_neurons": neurons, "importance_observations": observations, "calibration_tokens": tokens, "budgets": results, "smallest_passing_width": selected_width, "copied_source_parameters_in_artifact": 0, "artifact_written": False, "training_performed": False, "teacher_present_in_artifact": False, "final_test_accessed": False, "phase3_certified": False, "claim_boundary": "Read-only layer-1 matched source-neuron subset audit only; no artifact, autonomous English quality, measured runtime, certificate, or superiority claim."}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_SOURCE_NEURON_SUBSET_AUDIT_PROTOCOL_V271.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_source_neurons/layer1_audit_v272.json"); args = parser.parse_args(); root = Path.cwd().resolve(); output = root / args.output
    if output.exists(): raise Phase3Error("source-neuron output exists")
    result = execute(root, root / args.protocol); _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"].startswith("PASS") else 1
if __name__ == "__main__": raise SystemExit(main())
