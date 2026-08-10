"""Derive one source-MLP neuron subset from train-only contribution energy."""

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
from . import capability_compiler_phase3_factorized_attention_residual_span as span
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-source-neuron-threshold-audit/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_SOURCE_NEURON_THRESHOLD_AUDIT"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("source_block_promotion") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("neuron_count_sweep_authorized") is not False
        or protocol.get("importance_method_sweep_authorized") is not False
    ):
        raise Phase3Error("source-neuron threshold governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"source-neuron threshold binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("source-neuron threshold output exists or CUDA unavailable")

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
        for name, value in prefix.items(): state[name].copy_(value.to(state[name].dtype))
    layer0 = model.layers[0].float().cuda().eval()
    examples = sequential.field._examples(root, base, tokenizer)
    train_rows, validation_rows = coverage.expanded_split(examples, seed=int(base["training"]["seed"]), maximum_tokens=int(protocol["population"]["maximum_sequence_actions"]))
    teacher = AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    for value in teacher.parameters(): value.requires_grad_(False)
    source_layer1 = teacher.model.layers[1]
    source_attention = source_layer1.self_attn
    attention_cfg = protocol["factorized_attention"]
    ranks, attention_energy, runtime_factors = span._factors(source_attention, int(attention_cfg["source_heads"]), int(attention_cfg["source_head_dimension"]), float(attention_cfg["operator_energy_threshold"]))
    if ranks != [int(value) for value in attention_cfg["locked_rank_schedule"]]: raise Phase3Error("source-neuron attention schedule changed")
    gate_up = source_layer1.mlp.gate_up_proj.weight.detach().float()
    down = source_layer1.mlp.down_proj.weight.detach().float()
    neurons = down.shape[1]
    width = int(protocol["full_width"])
    if gate_up.shape != (2 * neurons, width) or down.shape != (width, neurons): raise Phase3Error("source-neuron MLP topology changed")

    terminal = int(base["source"]["terminal_token_id"])
    process = psutil.Process(); peak_rss = process.memory_info().rss; torch.cuda.reset_peak_memory_stats(); started=time.perf_counter(); route_exact=0

    def factorized_attention(candidate):
        length=candidate.shape[1]; position_ids=torch.arange(length,device=device)[None]; position_embeddings=teacher.model.rotary_emb(candidate,position_ids); normalized=source_layer1.input_layernorm(candidate)
        _,weights=source_attention(hidden_states=normalized,attention_mask=dual.base._causal_mask(length,device=device,dtype=candidate.dtype),position_ids=position_ids,use_cache=False,position_embeddings=position_embeddings)
        delta=torch.zeros_like(candidate,dtype=torch.float32)
        for head,(vf,of) in enumerate(runtime_factors): delta.add_(F.linear(weights[:,head].float() @ F.linear(normalized.float(),vf),of))
        return candidate.float()+delta

    def common(row):
        nonlocal route_exact,peak_rss
        host_ids=torch.tensor([row["input_ids"]],dtype=torch.long); route_index=model._select_route(host_ids); route_exact += int(route_index==routed._route(str(row["capability"])))
        candidate=model.token_embedding(host_ids).to(device); positions=torch.arange(candidate.shape[1],device=device); candidate,_,_=layer0.forward_with_cache(candidate,positions,route_index); attention=factorized_attention(candidate)
        feature=source_layer1.post_attention_layernorm(attention).float(); source_ids=torch.tensor([[trajectory.source_token_id(value,terminal) for value in row["input_ids"]]],dtype=torch.long,device=device); native=teacher.model.embed_tokens(source_ids)
        for source_index in range(2): _,native=dual._teacher_components(teacher,source_index,native)
        peak_rss=max(peak_rss,process.memory_info().rss); return attention,feature,native.float()

    importance=torch.zeros(neurons,dtype=torch.float64,device=device); importance_observations=0
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for index,row in enumerate(train_rows):
            _,feature,_=common(row); gate,up=F.linear(feature,gate_up).chunk(2,dim=-1); activation=(F.silu(gate)*up).float(); importance.add_(activation.square().sum(dim=(0,1)).double()); importance_observations += activation.shape[0]*activation.shape[1]
            if (index+1)%500==0: print(json.dumps({"importance_records":index+1}),flush=True)
    importance *= down.double().square().sum(dim=0)
    order=torch.argsort(importance,descending=True,stable=True); ordered=importance.index_select(0,order); cumulative=torch.cumsum(ordered,dim=0)/ordered.sum().clamp_min(1e-300); threshold=float(protocol["contribution_energy_threshold"]); selected_count=int(torch.searchsorted(cumulative,torch.tensor(threshold,dtype=torch.float64,device=device),right=False).item())+1; selected=order[:selected_count]; achieved=float(cumulative[selected_count-1]); selected_gate=gate_up[:neurons].index_select(0,selected); selected_up=gate_up[neurons:].index_select(0,selected); selected_down=down.index_select(1,selected)

    subset_cosines=[];subset_rmses=[];full_cosines=[];full_rmses=[];records=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for row in validation_rows:
            attention,feature,native=common(row); activation=F.silu(F.linear(feature,selected_gate))*F.linear(feature,selected_up); subset=attention+F.linear(activation,selected_down); full=attention+source_layer1.mlp(feature)
            sc,sr=trajectory._metrics(subset.float(),native); fc,fr=trajectory._metrics(full.float(),native); subset_cosines.append(sc);subset_rmses.append(sr);full_cosines.append(fc);full_rmses.append(fr);records.append({"record_id":row["record_id"],"capability":row["capability"],"subset_cosine":sc,"subset_relative_rmse":sr,"full_cosine":fc,"full_relative_rmse":fr})
    artifact_after=sha256_file(artifact_path); mean_cosine=sum(subset_cosines)/len(subset_cosines);mean_rmse=sum(subset_rmses)/len(subset_rmses)
    gates={"fixed_attention_energy":attention_energy>=float(attention_cfg["operator_energy_threshold"]),"derived_contribution_energy":achieved>=threshold,"strict_neuron_compression":selected_count<neurons,"validation_mean_cosine":mean_cosine>=float(protocol["gates"]["validation_mean_cosine_minimum"]),"validation_mean_relative_rmse":mean_rmse<=float(protocol["gates"]["validation_mean_relative_rmse_maximum"]),"routes_exact":route_exact==len(train_rows)+len(validation_rows),"artifact_unchanged":artifact_before==artifact_after};passed=all(gates.values())
    result={"format":FORMAT,"status":"PASS_SOURCE_NEURON_THRESHOLD" if passed else "FAIL_SOURCE_NEURON_THRESHOLD","protocol_sha256":sha256_file(protocol_path),"source_neurons":neurons,"selected_neurons":selected_count,"contribution_energy_threshold":threshold,"achieved_contribution_energy":achieved,"importance_observations":importance_observations,"factorized_attention_total_rank":sum(ranks),"subset_validation":{"records":len(validation_rows),"mean_cosine":mean_cosine,"minimum_cosine":min(subset_cosines),"mean_relative_rmse":mean_rmse,"maximum_relative_rmse":max(subset_rmses)},"full_source_mlp_diagnostic":{"mean_cosine":sum(full_cosines)/len(full_cosines),"mean_relative_rmse":sum(full_rmses)/len(full_rmses)},"physical_accounting":{"selected_mlp_parameters":selected_count*3*width,"selected_mlp_multiply_adds_per_token":selected_count*3*width,"factorized_attention_multiply_adds_per_token":33441792,"combined_projection_multiply_adds_per_token":33441792+selected_count*3*width},"record_metrics":records,"route_correct":route_exact,"gates":gates,"passed":passed,"artifact_model_sha256_before":artifact_before,"artifact_model_sha256_after":artifact_after,"wall_seconds":time.perf_counter()-started,"peak_process_rss_bytes":peak_rss,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"neuron_count_sweep_performed":False,"importance_method_sweep_performed":False,"training_performed":False,"artifact_written":False,"source_blocks_promoted":0,"final_test_accessed":False,"phase3_certified":False,"claim_boundary":"Read-only training-derived single source-neuron contribution-threshold subset behind the fixed factorized attention; no weights installed and no physical runtime, autonomous, complete-model, Phase 3, or superiority claim."};result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest();_write_immutable(output/"result.json",json.dumps(result,indent=2,sort_keys=True).encode()+b"\n");return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_SOURCE_NEURON_THRESHOLD_PROTOCOL_V428.json");parser.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_native_trajectory/source_neuron_threshold_v429");args=parser.parse_args();root=Path.cwd().resolve();result=execute(root,(root/args.protocol).resolve(),(root/args.output_dir).resolve());print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
