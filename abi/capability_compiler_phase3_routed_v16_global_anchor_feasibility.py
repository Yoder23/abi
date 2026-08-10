"""No-artifact feasibility audit for an in-budget layer-29 global-state anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil
from safetensors.torch import load_file
import torch

from . import capability_compiler_phase3_closed_form_coefficient_audit as closed
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_mlp_residual_rank_audit as rank_audit
from . import capability_compiler_phase3_routed_v15_layer0_extract as layer0
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_accumulation_attribution import metrics, source_token_id


FORMAT = "abi-capability-compiler-phase3-routed-v16-global-anchor-feasibility/1"
OFFSET = 4


def _advance_source(teacher, hidden: torch.Tensor, stop: int) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(stop):
            _, hidden = dual._teacher_components(teacher, index, hidden)
    return hidden


def _advance_candidate(model, hidden: torch.Tensor, positions: torch.Tensor, route: int, stop: int) -> torch.Tensor:
    for index in range(stop):
        hidden, _, _ = model.layers[index].forward_with_cache(hidden, positions, route)
    return hidden


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_NO_ARTIFACT_GLOBAL_ANCHOR_FEASIBILITY" or protocol.get("training") != "ANALYTIC_IN_MEMORY_ONLY" or protocol.get("artifact_write") != "PROHIBITED" or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("global-anchor feasibility governance changed")
    for name, expected in protocol["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"global-anchor feasibility binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("global-anchor output exists or CUDA unavailable")
    output.mkdir(parents=True)
    base = json.loads((root / protocol["base_protocol"]).read_text(encoding="utf-8"))
    artifact = (root / protocol["artifact"]["directory"]).resolve(); config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    layercake_root = (root / protocol["layercake_host"]["repository"]).resolve(); sys.path.insert(0, str(layercake_root))
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer

    tokenizer = DecoderAwareExternalTokenizer.from_document(config["tokenizer"])
    model = PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer)
    model.load_state_dict(load_file(str(artifact / "model.safetensors"), device="cuda"), strict=True, assign=True); model = model.cuda().eval()
    examples = dual.field._examples(root, base, tokenizer); example_by_id = {str(row["record_id"]): row for row in examples}
    cfg = base["calibration"]
    train, validation, _ = dual._calibration_examples(examples, seed=int(base["training"]["seed"]), train_per_capability=int(cfg["train_records_per_capability"]), validation_per_capability=int(cfg["validation_records_per_capability"]), maximum_tokens=int(cfg["maximum_sequence_tokens"]))
    if len(train) != 420 or len(validation) != 28: raise Phase3Error("global-anchor population changed")
    teacher = AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    terminal, anchor = int(base["source"]["terminal_token_id"]), int(protocol["anchor_layer"])
    features_cpu, residuals_cpu = [], []
    started = time.perf_counter(); process = psutil.Process(); peak_rss = process.memory_info().rss; torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, row in enumerate(train):
            host_ids = list(example_by_id[str(row["record_id"])]["source_ids"]); source_ids = [source_token_id(value, terminal) for value in host_ids]
            host = torch.tensor([host_ids], dtype=torch.long, device="cuda"); source = torch.tensor([source_ids], dtype=torch.long, device="cuda")
            route = model._select_route(host); positions = torch.arange(len(host_ids), device="cuda")
            student = _advance_candidate(model, model.token_embedding(host), positions, route, anchor)
            teacher_hidden = _advance_source(teacher, teacher.model.embed_tokens(source), anchor + 1)
            attention = layer0._attention(model.layers[anchor], student, positions)
            features_cpu.append(model.layers[anchor].post_attention_norm(attention).squeeze(0).float().cpu())
            residuals_cpu.append((teacher_hidden - attention).squeeze(0).float().cpu())
            if (index + 1) % 60 == 0: print(json.dumps({"train_records": index + 1}), flush=True)
    width, rank = int(config["model"]["full_width"]), int(config["model"]["residual_rank"])
    mean, covariance, observations = rank_audit.centered_covariance(residuals_cpu, width, torch.device("cuda"))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance); eigenvalues = eigenvalues.clamp_min(0).flip(0); basis = eigenvectors.flip(1)[:, :rank].contiguous()
    energy = float(eigenvalues[:rank].sum() / eigenvalues.sum().clamp_min(1e-12))
    features = torch.cat(features_cpu).cuda(); residuals = torch.cat(residuals_cpu).cuda(); coefficients = (residuals - mean) @ basis
    weights, ridge = closed.solve_ridge(features, coefficients, float(protocol["relative_ridge"]))
    train_prediction = mean + (features @ weights) @ basis.T
    train_rmse = float(torch.sqrt(torch.mean((train_prediction - residuals) ** 2) / torch.mean(residuals**2).clamp_min(1e-8)))
    layer = model.layers[anchor]
    with torch.no_grad():
        layer.mlp_residual_mean.copy_(mean.to(layer.mlp_residual_mean.dtype)); layer.mlp_output_projection.weight.copy_(basis.to(layer.mlp_output_projection.weight.dtype)); layer.linear_coefficient_projection.weight.copy_(weights.T.to(layer.linear_coefficient_projection.weight.dtype)); layer.sparse_gate_up_projection.weight.zero_()
        for projection in layer.route_coefficient_projections: projection.weight.zero_()
    records = []
    with torch.inference_mode():
        for row in validation:
            host_ids = list(example_by_id[str(row["record_id"])]["source_ids"]); source_ids = [source_token_id(value, terminal) for value in host_ids]
            host = torch.tensor([host_ids], dtype=torch.long, device="cuda"); source = torch.tensor([source_ids], dtype=torch.long, device="cuda"); positions = torch.arange(len(host_ids), device="cuda")
            route = model._select_route(host); student = _advance_candidate(model, model.token_embedding(host), positions, route, anchor)
            teacher_hidden = _advance_source(teacher, teacher.model.embed_tokens(source), 32)
            student, _, _ = model.layers[anchor].forward_with_cache(student, positions, route); anchor_target = _advance_source(teacher, teacher.model.embed_tokens(source), anchor + 1)
            anchor_metric = metrics(student, anchor_target)
            for index in range(anchor + 1, 32): student, _, _ = model.layers[index].forward_with_cache(student, positions, route)
            final_metric = metrics(student, teacher_hidden)
            with torch.autocast("cuda", dtype=torch.bfloat16): teacher_logits = teacher.lm_head(teacher.model.norm(teacher_hidden[:, -1]))
            candidate_logits = model.lm_head(model.final_norm(student[:, -1])); common = metrics(candidate_logits[:, OFFSET:], teacher_logits[:, : candidate_logits.shape[-1] - OFFSET])
            source_top = int(teacher_logits.argmax(-1)); action = int(candidate_logits.argmax(-1)); candidate_top = source_token_id(action, terminal) if action == 2 or action >= 4 else -1
            records.append({"record_id": str(row["record_id"]), "capability": str(row["capability"]), "route_correct": route == layer0._route(str(row["capability"])), "anchor_cosine": anchor_metric["cosine"], "anchor_relative_rmse": anchor_metric["relative_rmse"], "final_cosine": final_metric["cosine"], "final_relative_rmse": final_metric["relative_rmse"], "logit_cosine": common["cosine"], "top1_agreement": candidate_top == source_top})
            peak_rss = max(peak_rss, process.memory_info().rss)
    raw = output / "records.jsonl"; raw.write_bytes(b"".join(canonical_json_bytes(row) for row in records))
    aggregate = {"mean_anchor_cosine": sum(row["anchor_cosine"] for row in records) / len(records), "mean_final_cosine": sum(row["final_cosine"] for row in records) / len(records), "mean_logit_cosine": sum(row["logit_cosine"] for row in records) / len(records), "top1_agreement": sum(row["top1_agreement"] for row in records) / len(records), "route_correct": sum(row["route_correct"] for row in records)}
    gates = {"rank_energy": energy >= float(protocol["gates"]["rank_energy_minimum"]), "anchor_global_cosine": aggregate["mean_anchor_cosine"] >= float(protocol["gates"]["anchor_global_cosine_minimum"]), "final_global_cosine": aggregate["mean_final_cosine"] >= float(protocol["gates"]["final_global_cosine_minimum"]), "top1_agreement": aggregate["top1_agreement"] >= float(protocol["gates"]["top1_agreement_minimum"]), "route_exact": aggregate["route_correct"] == len(records)}
    result = {"format": FORMAT, "status": "PASS_GLOBAL_ANCHOR_FEASIBILITY" if all(gates.values()) else "FAIL_GLOBAL_ANCHOR_FEASIBILITY", "protocol_sha256": sha256_file(protocol_path), "anchor_layer": anchor, "train_records": len(train), "validation_records": len(validation), "observations": observations, "rank": rank, "rank_energy": energy, "ridge": ridge, "training_residual_relative_rmse": train_rmse, "validation": aggregate, "gates": gates, "passed": all(gates.values()), "raw_sha256": sha256_file(raw), "wall_seconds": time.perf_counter() - started, "peak_process_rss_bytes": peak_rss, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "artifact_written": False, "stored_activations_written": 0, "training_performed": False, "final_test_accessed": False, "phase3_certified": False, "claim_boundary": "In-memory analytic feasibility only; no artifact, autonomous quality, runtime, Phase 3, or superiority claim."}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n"); return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V16_GLOBAL_ANCHOR_FEASIBILITY_PROTOCOL_V335.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_routed_v16/global_anchor_feasibility_v336"); args = parser.parse_args(); root = Path.cwd().resolve(); print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
