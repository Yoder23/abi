"""Read-only diagnostic of the complete source MLP on replacement attention."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from safetensors.torch import load_file
import torch
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
FORMAT = "abi-capability-compiler-phase3-complete-mlp-interface-audit/1"

def execute(root: Path, path: Path) -> dict:
    from transformers import AutoModelForCausalLM
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_COMPLETE_MLP_INTERFACE" or protocol.get("artifact_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("complete MLP audit governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"complete MLP audit binding changed: {name}")
    if not torch.cuda.is_available(): raise Phase3Error("CUDA required")
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8")); device = torch.device("cuda"); model, tokenizer, _, _, _ = sequential._model(root, base, device); state = model.state_dict()
    for layer_index in (0, 1):
        checkpoint = load_file(str(root / protocol["checkpoints"][str(layer_index)]["path"]), device="cpu")
        for name, value in checkpoint.items():
            if name in state: state[name].copy_(value.to(state[name].dtype))
    model.eval(); examples = sequential.field._examples(root, base, tokenizer); cfg = base["calibration"]; _, rows, tokens = dual._calibration_examples(examples, seed=int(base["training"]["seed"]), train_per_capability=int(cfg["train_records_per_capability"]), validation_per_capability=int(cfg["validation_records_per_capability"]), maximum_tokens=int(cfg["maximum_sequence_tokens"]))
    teacher = AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval(); layer_index = 1; source_layer = teacher.model.layers[layer_index]; student_layer = model.layers[layer_index]
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    names = ("teacher_interface", "student_interface", "exact_delta_on_student_attention"); metrics = {name: [[], []] for name in names}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device); hidden = dual.base._prefix_hidden(model, ids, layer_index); teacher_attention, teacher_final = dual._teacher_components(teacher, layer_index, hidden); student_attention = sequential._student_attention(student_layer, hidden, torch.arange(ids.shape[1], device=device)); teacher_delta = teacher_final - teacher_attention; teacher_prediction = teacher_attention + source_layer.mlp(source_layer.post_attention_layernorm(teacher_attention)); student_prediction = student_attention + source_layer.mlp(source_layer.post_attention_layernorm(student_attention)); exact_prediction = student_attention + teacher_delta
            for name, prediction in zip(names, (teacher_prediction, student_prediction, exact_prediction)):
                rmse, cosine = dual.base._metrics(prediction, teacher_final, hidden); metrics[name][0].append(float(rmse)); metrics[name][1].append(float(cosine))
    gate = protocol["gate"]; summaries = {}
    for name, (rmses, cosines) in metrics.items():
        mr = sum(rmses) / len(rmses); mc = sum(cosines) / len(cosines); summaries[name] = {"mean_relative_rmse": mr, "maximum_relative_rmse": max(rmses), "mean_output_cosine": mc, "minimum_output_cosine": min(cosines), "passed": mr <= gate["mean_relative_rmse_maximum"] and mc >= gate["mean_output_cosine_minimum"]}
    interface_pass = summaries["student_interface"]["passed"]; diagnosis = "COMPRESSION_PRIMARY" if interface_pass else "ATTENTION_TO_MLP_INTERFACE_PRIMARY"
    return {"format": FORMAT, "status": "PASS_DIAGNOSTIC_COMPLETE", "protocol_sha256": sha256_file(path), "layer": 1, "calibration_tokens": tokens, "paths": summaries, "diagnosis": diagnosis, "complete_source_block_promoted": False, "artifact_written": False, "training_performed": False, "final_test_accessed": False, "phase3_certified": False, "claim_boundary": "Read-only complete-source-MLP interface diagnostic only; no source block may be promoted and no artifact, certificate, or superiority claim is made."}
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_COMPLETE_MLP_INTERFACE_AUDIT_PROTOCOL_V273.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_source_neurons/complete_mlp_interface_v274.json"); args = parser.parse_args(); root = Path.cwd().resolve(); output = root / args.output
    if output.exists(): raise Phase3Error("output exists")
    result = execute(root, root / args.protocol); _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); print(json.dumps(result, indent=2, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
