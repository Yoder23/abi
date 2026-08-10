"""Read-only nested low-rank attention-to-MLP feature-alignment audit."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from safetensors.torch import load_file
import torch
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_layer1_error_decomposition as decomposition
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
FORMAT = "abi-capability-compiler-phase3-feature-alignment-audit/1"

def execute(root: Path, path: Path) -> dict:
    from transformers import AutoModelForCausalLM
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_NESTED_FEATURE_ALIGNMENT" or protocol.get("artifact_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("feature alignment governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"alignment binding changed: {name}")
    if not torch.cuda.is_available(): raise Phase3Error("CUDA required")
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8")); device = torch.device("cuda"); model, tokenizer, _, _, _ = sequential._model(root, base, device); state = model.state_dict()
    for layer_index in (0, 1):
        checkpoint = load_file(str(root / protocol["checkpoints"][str(layer_index)]["path"]), device="cpu")
        for name, value in checkpoint.items():
            if name in state: state[name].copy_(value.to(state[name].dtype))
    model.eval(); examples = sequential.field._examples(root, base, tokenizer); cfg = base["calibration"]; train_rows, validation_rows, tokens = dual._calibration_examples(examples, seed=int(base["training"]["seed"]), train_per_capability=int(cfg["train_records_per_capability"]), validation_per_capability=int(cfg["validation_records_per_capability"]), maximum_tokens=int(cfg["maximum_sequence_tokens"]))
    teacher = AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval(); layer_index = 1; source_layer = teacher.model.layers[layer_index]; student_layer = model.layers[layer_index]
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    features = []; differences = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in train_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device); hidden = dual.base._prefix_hidden(model, ids, layer_index); teacher_attention, _ = dual._teacher_components(teacher, layer_index, hidden); student_attention = sequential._student_attention(student_layer, hidden, torch.arange(ids.shape[1], device=device)); student_feature = student_layer.post_attention_norm(student_attention).squeeze(0).float().cpu(); teacher_feature = source_layer.post_attention_layernorm(teacher_attention).squeeze(0).float().cpu(); features.append(student_feature); differences.append(teacher_feature - student_feature)
    mean, covariance, observations = rank_audit.centered_covariance(differences, 3072, device); eigenvalues, eigenvectors = torch.linalg.eigh(covariance); eigenvalues = eigenvalues.clamp_min(0).flip(0); eigenvectors = eigenvectors.flip(1).contiguous(); ranks = [int(value) for value in protocol["ranks"]]; maximum_rank = max(ranks); basis = eigenvectors[:, :maximum_rank]; x = torch.cat(features).to(device); target = (torch.cat(differences).to(device) - mean) @ basis; weights, ridge = decomposition.solve_map(x, target, float(protocol["relative_ridge"])); del features, differences, x, target
    accumulators = {rank: {"mapped": [[], []], "oracle": [[], []]} for rank in ranks}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_rows:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device); hidden = dual.base._prefix_hidden(model, ids, layer_index); teacher_attention, teacher_final = dual._teacher_components(teacher, layer_index, hidden); student_attention = sequential._student_attention(student_layer, hidden, torch.arange(ids.shape[1], device=device)); student_feature = student_layer.post_attention_norm(student_attention).float(); teacher_feature = source_layer.post_attention_layernorm(teacher_attention).float(); difference = teacher_feature - student_feature
            for rank in ranks:
                current_basis = basis[:, :rank]; mapped_feature = student_feature + mean + (student_feature @ weights[:, :rank]) @ current_basis.T; oracle_feature = student_feature + mean + ((difference - mean) @ current_basis) @ current_basis.T
                for name, aligned in (("mapped", mapped_feature), ("oracle", oracle_feature)):
                    prediction = student_attention.float() + source_layer.mlp(aligned).float(); rmse, cosine = dual.base._metrics(prediction, teacher_final.float(), hidden.float()); accumulators[rank][name][0].append(float(rmse)); accumulators[rank][name][1].append(float(cosine))
    gate = protocol["gate"]; rows = []
    for rank in ranks:
        result = {"rank": rank, "training_alignment_energy_explained": float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))}
        for name, (rmses, cosines) in accumulators[rank].items():
            mr = sum(rmses) / len(rmses); mc = sum(cosines) / len(cosines); result[name] = {"mean_relative_rmse": mr, "maximum_relative_rmse": max(rmses), "mean_output_cosine": mc, "minimum_output_cosine": min(cosines), "passed": mr <= gate["mean_relative_rmse_maximum"] and mc >= gate["mean_output_cosine_minimum"]}
        rows.append(result)
    passing = [row for row in rows if row["mapped"]["passed"]]
    return {"format": FORMAT, "status": "PASS_ALIGNMENT_FEASIBLE_NO_ARTIFACT" if passing else "FAIL_ALIGNMENT_AUDIT", "protocol_sha256": sha256_file(path), "layer": 1, "train_observations": observations, "calibration_tokens": tokens, "effective_ridge": ridge, "ranks": rows, "smallest_passing_rank": passing[0]["rank"] if passing else None, "complete_source_mlp_promoted": False, "artifact_written": False, "training_performed": False, "final_test_accessed": False, "phase3_certified": False, "claim_boundary": "Read-only low-rank feature-alignment diagnostic using a non-promotable complete source MLP; no artifact, runtime, certificate, or superiority claim."}
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_FEATURE_ALIGNMENT_AUDIT_PROTOCOL_V275.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_feature_alignment/layer1_audit_v276.json"); args = parser.parse_args(); root = Path.cwd().resolve(); output = root / args.output
    if output.exists(): raise Phase3Error("output exists")
    result = execute(root, root / args.protocol); _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"].startswith("PASS") else 1
if __name__ == "__main__": raise SystemExit(main())
