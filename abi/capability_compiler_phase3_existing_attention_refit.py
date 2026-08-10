"""Existing-architecture layer-zero attention refit plus analytic decoder extraction."""

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
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_closed_form_coefficient_audit as closed
from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-existing-attention-refit/1"


def expanded_split(examples: list[dict], *, seed: int, maximum_tokens: int) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in examples:
        grouped[str(row["capability"])].append(row)
    train = []; validation = []
    for capability in CAPABILITIES:
        ranked = sorted(grouped[capability], key=lambda row: hashlib.sha256(f"{seed}:{row['record_id']}".encode()).digest())
        train_source = ranked[:30] + ranked[32:302]
        validation_source = ranked[30:32]
        if len(train_source) != 300 or len(validation_source) != 2:
            raise Phase3Error("fixed attention-refit coverage unavailable")
        for source, destination in ((train_source, train), (validation_source, validation)):
            for row in source:
                packed = (list(row["source_ids"]) + list(row["target_actions"])[:-1])[:maximum_tokens]
                destination.append({"record_id": row["record_id"], "capability": capability, "input_ids": packed})
    train.sort(key=lambda row: hashlib.sha256(f"train:{seed}:{row['record_id']}".encode()).digest())
    validation.sort(key=lambda row: hashlib.sha256(f"validation:{seed}:{row['record_id']}".encode()).digest())
    return train, validation


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_FAIL_FAST_EXISTING_ATTENTION_REFIT"
        or protocol.get("device") != "cuda"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("existing-attention refit governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"existing-attention refit binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("existing-attention refit output exists or CUDA unavailable")
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
    layer = model.layers[0].float().cuda()
    model.router = model.router.float().cuda().eval()
    examples = sequential.field._examples(root, base, tokenizer)
    train, validation = expanded_split(examples, seed=int(base["training"]["seed"]), maximum_tokens=128)
    teacher = AutoModelForCausalLM.from_pretrained(
        base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    terminal = int(base["source"]["terminal_token_id"])
    process = psutil.Process(); peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()

    def capture(rows: list[dict], label: str) -> list[dict]:
        nonlocal peak_rss
        values = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index, row in enumerate(rows):
                source = torch.tensor([[trajectory.source_token_id(value, terminal) for value in row["input_ids"]]], device=device)
                hidden = teacher.model.embed_tokens(source)
                attention, target = dual._teacher_components(teacher, 0, hidden)
                values.append({"record_id": row["record_id"], "capability": row["capability"], "input_ids": row["input_ids"], "attention": attention.squeeze(0).float().cpu(), "target": target.squeeze(0).float().cpu()})
                peak_rss = max(peak_rss, process.memory_info().rss)
                if (index + 1) % 500 == 0:
                    print(json.dumps({"capture": label, "records": index + 1}), flush=True)
        return values

    train_cache = capture(train, "train"); validation_cache = capture(validation, "validation")
    del teacher; torch.cuda.empty_cache()
    embedding = model.token_embedding.cuda().eval()
    attention_prefixes = ("input_norm", "attention_input_projection", "attention_norm", "qkv_proj", "o_proj", "attention_output_projection", "secondary_")
    trainable = [parameter for name, parameter in layer.named_parameters() if name.startswith(attention_prefixes)]
    for parameter in layer.parameters(): parameter.requires_grad_(False)
    for parameter in trainable: parameter.requires_grad_(True)
    training = protocol["training"]
    optimizer = torch.optim.AdamW(trainable, lr=float(training["learning_rate"]), betas=(0.9, 0.95), weight_decay=float(training["weight_decay"]))
    curves = []; layer.train()
    for step, row in enumerate(train_cache, start=1):
        ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
        with torch.no_grad(): hidden = embedding(ids)
        target = row["attention"].unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = routed._attention(layer, hidden, torch.arange(ids.shape[1], device=device)).float()
            relative_mse = torch.mean((prediction - target).square()) / torch.mean(target.square()).clamp_min(1e-8)
            cosine = F.cosine_similarity(prediction.reshape(1, -1), target.reshape(1, -1)).mean()
            loss = relative_mse + float(training["cosine_weight"]) * (1 - cosine)
        if not torch.isfinite(loss):
            raise Phase3Error(f"existing-attention refit became nonfinite at step {step}")
        loss.backward(); gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, float(training["gradient_clip_norm"]))
        optimizer.step(); peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(training["curve_interval"]) == 0:
            point = {"step": step, "loss": float(loss), "relative_rmse": float(torch.sqrt(relative_mse)), "cosine": float(cosine), "gradient_norm": float(gradient_norm)}
            curves.append(point); print(json.dumps(point), flush=True)
    layer.eval()
    features_cpu = []; sparse_cpu = []; residuals_cpu = []; token_routes = []
    route_exact = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, row in enumerate(train_cache):
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            route = model._select_route(ids); route_exact += int(route == routed._route(str(row["capability"])))
            hidden = embedding(ids)
            attention = routed._attention(layer, hidden, torch.arange(ids.shape[1], device=device))
            feature = layer.post_attention_norm(attention)
            gate, up = layer.sparse_gate_up_projection(feature).float().chunk(2, dim=-1)
            features_cpu.append(feature.squeeze(0).float().cpu())
            sparse_cpu.append((F.silu(gate) * up).squeeze(0).cpu())
            residuals_cpu.append((row["target"].unsqueeze(0).to(device) - attention).squeeze(0).float().cpu())
            token_routes.extend([route] * ids.shape[1])
            if (index + 1) % 500 == 0:
                print(json.dumps({"extract": "decoder", "records": index + 1}), flush=True)
    mean, covariance, observations = rank_audit.centered_covariance(residuals_cpu, int(protocol["full_width"]), device)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance); eigenvalues = eigenvalues.clamp_min(0).flip(0)
    rank = int(protocol["rank"]); basis = eigenvectors.flip(1)[:, :rank].contiguous()
    energy = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))
    features = torch.cat(features_cpu).to(device); residuals = torch.cat(residuals_cpu).to(device)
    sparse = torch.cat(sparse_cpu).to(device); routes = torch.tensor(token_routes, dtype=torch.long, device=device)
    targets = (residuals - mean) @ basis
    linear_map, linear_ridge = closed.solve_ridge(features, targets, float(protocol["relative_ridge"]))
    correction = targets - features @ linear_map
    route_maps = []; route_ridges = []; route_observations = []
    for route in range(3):
        indices = torch.nonzero(routes == route, as_tuple=False).squeeze(1)
        mapping, ridge = closed.solve_ridge(sparse.index_select(0, indices), correction.index_select(0, indices), float(protocol["relative_ridge"]))
        route_maps.append(mapping); route_ridges.append(ridge); route_observations.append(int(indices.numel()))
    with torch.no_grad():
        layer.mlp_residual_mean.copy_(mean.to(layer.mlp_residual_mean.dtype))
        layer.mlp_output_projection.weight.copy_(basis.to(layer.mlp_output_projection.weight.dtype))
        layer.linear_coefficient_projection.weight.copy_(linear_map.T.to(layer.linear_coefficient_projection.weight.dtype))
        for index, mapping in enumerate(route_maps): layer.route_coefficient_projections[index].weight.copy_(mapping.T.to(layer.route_coefficient_projections[index].weight.dtype))
    attention_cosines = []; attention_rmses = []; cosines = []; rmses = []; records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for row in validation_cache:
            ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            route = model._select_route(ids); route_exact += int(route == routed._route(str(row["capability"])))
            hidden = embedding(ids); attention = routed._attention(layer, hidden, torch.arange(ids.shape[1], device=device))
            ac, ar = trajectory._metrics(attention, row["attention"].unsqueeze(0).to(device))
            prediction = attention + layer._mlp_delta(attention, route)
            cosine, relative_rmse = trajectory._metrics(prediction, row["target"].unsqueeze(0).to(device))
            attention_cosines.append(ac); attention_rmses.append(ar); cosines.append(cosine); rmses.append(relative_rmse)
            records.append({"record_id": row["record_id"], "capability": row["capability"], "cosine": cosine, "relative_rmse": relative_rmse, "attention_cosine": ac, "attention_relative_rmse": ar})
    artifact_after = sha256_file(artifact_path); mean_cosine = sum(cosines) / len(cosines); mean_rmse = sum(rmses) / len(rmses)
    gates = {"rank_energy": energy >= float(protocol["gates"]["rank_energy_minimum"]), "validation_mean_cosine": mean_cosine >= float(protocol["gates"]["validation_mean_cosine_minimum"]), "validation_mean_relative_rmse": mean_rmse <= float(protocol["gates"]["validation_mean_relative_rmse_maximum"]), "routes_exact": route_exact == len(train_cache) + len(validation_cache), "artifact_unchanged": artifact_before == artifact_after}
    passed = all(gates.values()); checkpoint = None
    if passed:
        tensors = {f"layers.0.{name}": value.detach().half().cpu().contiguous() for name, value in layer.state_dict().items()}
        checkpoint_path = output / "layer_00.safetensors"; save_file(tensors, str(checkpoint_path), metadata={"format": FORMAT, "protocol_sha256": sha256_file(protocol_path)})
        checkpoint = {"path": checkpoint_path.name, "sha256": sha256_file(checkpoint_path), "parameters": sum(value.numel() for value in tensors.values())}
    result = {"format": FORMAT, "status": "PASS_EXISTING_ATTENTION_REFIT_LAYER_ZERO" if passed else "FAIL_EXISTING_ATTENTION_REFIT_LAYER_ZERO", "protocol_sha256": sha256_file(protocol_path), "training":{"records":len(train_cache),"steps":len(train_cache),"trainable_existing_parameters":sum(value.numel() for value in trainable),"curves":curves}, "rank":rank,"rank_energy":energy,"train_observations":observations,"linear_effective_ridge":linear_ridge,"route_effective_ridge":dict(zip(routed.ROUTES,route_ridges)),"route_train_observations":dict(zip(routed.ROUTES,route_observations)),"validation":{"records":len(validation_cache),"mean_attention_cosine":sum(attention_cosines)/len(attention_cosines),"mean_attention_relative_rmse":sum(attention_rmses)/len(attention_rmses),"mean_cosine":mean_cosine,"minimum_cosine":min(cosines),"mean_relative_rmse":mean_rmse,"maximum_relative_rmse":max(rmses),"record_metrics":records},"route_correct":route_exact,"gates":gates,"passed":passed,"checkpoint":checkpoint,"artifact_model_sha256_before":artifact_before,"artifact_model_sha256_after":artifact_after,"wall_seconds":time.perf_counter()-started,"peak_process_rss_bytes":peak_rss,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"source_blocks_in_checkpoint":0,"teacher_activations_persisted":0,"final_test_accessed":False,"phase3_certified":False,"claim_boundary":"Existing-architecture layer-zero attention refit and analytic decoder extraction only; no multi-layer artifact, autonomous quality, runtime, Phase 3, or superiority claim."}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_EXISTING_ATTENTION_REFIT_PROTOCOL_V381.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_native_trajectory/existing_attention_refit_v382")
    args = parser.parse_args(); root = Path.cwd().resolve(); print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
