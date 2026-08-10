"""Read-only shared-rank source-weight factorization preserving every head slot."""

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
import torch.nn as nn
import torch.nn.functional as F

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_existing_attention_refit as coverage
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-shared-rank256-all-head-attention-oracle/1"


def stable_factor(weight: torch.Tensor, rank: int):
    values = weight.detach().double()
    left, singular, right_h = torch.linalg.svd(values, full_matrices=False)
    if rank <= 0 or rank >= singular.numel():
        raise Phase3Error("shared attention rank is not a strict compression")
    root = torch.sqrt(singular[:rank])
    output_factor = (left[:, :rank] * root.unsqueeze(0)).float()
    input_factor = (root.unsqueeze(1) * right_h[:rank]).float()
    energy = float(singular[:rank].square().sum() / singular.square().sum().clamp_min(1e-300))
    return output_factor, input_factor, energy


class FactoredLinear(nn.Module):
    def __init__(self, output_factor: torch.Tensor, input_factor: torch.Tensor):
        super().__init__()
        self.register_buffer("output_factor", output_factor)
        self.register_buffer("input_factor", input_factor)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(values.float(), self.input_factor), self.output_factor)


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    factor = protocol.get("factorization", {})
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_SHARED_RANK256_ALL_HEAD_ATTENTION_ORACLE"
        or int(factor.get("qkv_rank", 0)) != 256
        or int(factor.get("output_rank", 0)) != 256
        or factor.get("decomposition_precision") != "fp64"
        or factor.get("runtime_precision") != "fp32"
        or protocol.get("training_authorized") is not False
        or protocol.get("artifact_write") != "PROHIBITED"
        or protocol.get("source_block_promotion") != "PROHIBITED"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("sweeps_authorized") is not False
    ):
        raise Phase3Error("shared rank256 attention governance changed")
    for name, expected in protocol["bindings"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"shared rank256 attention binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("shared rank256 attention output exists or CUDA unavailable")

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
        for name, value in prefix.items():
            state[name].copy_(value.to(state[name].dtype))
    layer0 = model.layers[0].float().cuda().eval()

    examples = sequential.field._examples(root, base, tokenizer)
    _, validation_rows = coverage.expanded_split(examples, seed=int(base["training"]["seed"]), maximum_tokens=int(protocol["population"]["maximum_sequence_actions"]))
    teacher = AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"], local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    for value in teacher.parameters(): value.requires_grad_(False)
    source_layer1 = teacher.model.layers[1]
    source_attention = source_layer1.self_attn
    original_qkv = source_attention.qkv_proj
    original_output = source_attention.o_proj
    qkv_output_factor, qkv_input_factor, qkv_energy = stable_factor(original_qkv.weight, int(factor["qkv_rank"]))
    output_output_factor, output_input_factor, output_energy = stable_factor(original_output.weight, int(factor["output_rank"]))
    factored_qkv = FactoredLinear(qkv_output_factor, qkv_input_factor).to(device).eval()
    factored_output = FactoredLinear(output_output_factor, output_input_factor).to(device).eval()

    terminal=int(base["source"]["terminal_token_id"]);process=psutil.Process();peak_rss=process.memory_info().rss;torch.cuda.reset_peak_memory_stats();started=time.perf_counter();route_exact=0
    attention_cosines=[];attention_rmses=[];final_cosines=[];final_rmses=[];full_cosines=[];full_rmses=[];records=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for row in validation_rows:
            host_ids=torch.tensor([row["input_ids"]],dtype=torch.long);route_index=model._select_route(host_ids);route_exact+=int(route_index==routed._route(str(row["capability"])))
            candidate=model.token_embedding(host_ids).to(device);positions=torch.arange(candidate.shape[1],device=device);candidate,_,_=layer0.forward_with_cache(candidate,positions,route_index)
            source_ids=torch.tensor([[trajectory.source_token_id(value,terminal) for value in row["input_ids"]]],dtype=torch.long,device=device);native=teacher.model.embed_tokens(source_ids)
            for source_index in range(2): _,native=dual._teacher_components(teacher,source_index,native)
            native=native.float();length=candidate.shape[1];position_ids=torch.arange(length,device=device)[None];position_embeddings=teacher.model.rotary_emb(candidate,position_ids);mask=dual.base._causal_mask(length,device=device,dtype=candidate.dtype);normalized=source_layer1.input_layernorm(candidate)
            source_attention.qkv_proj=original_qkv;source_attention.o_proj=original_output
            exact_delta,_=source_attention(hidden_states=normalized,attention_mask=mask,position_ids=position_ids,use_cache=False,position_embeddings=position_embeddings)
            source_attention.qkv_proj=factored_qkv;source_attention.o_proj=factored_output
            compressed_delta,_=source_attention(hidden_states=normalized,attention_mask=mask,position_ids=position_ids,use_cache=False,position_embeddings=position_embeddings)
            source_attention.qkv_proj=original_qkv;source_attention.o_proj=original_output
            exact_attention=candidate.float()+exact_delta.float();compressed_attention=candidate.float()+compressed_delta.float()
            compressed_final=compressed_attention+source_layer1.mlp(source_layer1.post_attention_layernorm(compressed_attention)).float();full_final=exact_attention+source_layer1.mlp(source_layer1.post_attention_layernorm(exact_attention)).float()
            ac,ar=trajectory._metrics(compressed_attention,exact_attention);fc,fr=trajectory._metrics(compressed_final,native);xc,xr=trajectory._metrics(full_final,native)
            attention_cosines.append(ac);attention_rmses.append(ar);final_cosines.append(fc);final_rmses.append(fr);full_cosines.append(xc);full_rmses.append(xr);records.append({"record_id":row["record_id"],"capability":row["capability"],"attention_cosine":ac,"attention_relative_rmse":ar,"final_cosine":fc,"final_relative_rmse":fr,"full_source_cosine":xc,"full_source_relative_rmse":xr});peak_rss=max(peak_rss,process.memory_info().rss)
    source_attention.qkv_proj=original_qkv;source_attention.o_proj=original_output
    artifact_after=sha256_file(artifact_path);mean_cosine=sum(final_cosines)/len(final_cosines);mean_rmse=sum(final_rmses)/len(final_rmses)
    gates={"qkv_strict_compression":int(factor["qkv_rank"])<original_qkv.weight.shape[1],"output_strict_compression":int(factor["output_rank"])<original_output.weight.shape[1],"all_head_slots_present":factored_qkv.output_factor.shape[0]==3*int(protocol["source_topology"]["full_width"]),"validation_mean_cosine":mean_cosine>=float(protocol["gates"]["validation_mean_cosine_minimum"]),"validation_mean_relative_rmse":mean_rmse<=float(protocol["gates"]["validation_mean_relative_rmse_maximum"]),"routes_exact":route_exact==len(validation_rows),"artifact_unchanged":artifact_before==artifact_after};passed=all(gates.values())
    result={"format":FORMAT,"status":"PASS_SHARED_RANK256_ALL_HEAD_ATTENTION_ORACLE" if passed else "FAIL_SHARED_RANK256_ALL_HEAD_ATTENTION_ORACLE","protocol_sha256":sha256_file(protocol_path),"factorization":{"qkv_rank":int(factor["qkv_rank"]),"output_rank":int(factor["output_rank"]),"qkv_operator_energy":qkv_energy,"output_operator_energy":output_energy,"all_head_slots_present":True},"validation":{"records":len(validation_rows),"mean_attention_cosine":sum(attention_cosines)/len(attention_cosines),"mean_attention_relative_rmse":sum(attention_rmses)/len(attention_rmses),"mean_final_cosine":mean_cosine,"minimum_final_cosine":min(final_cosines),"mean_final_relative_rmse":mean_rmse,"maximum_final_relative_rmse":max(final_rmses),"record_metrics":records},"full_source_diagnostic":{"mean_cosine":sum(full_cosines)/len(full_cosines),"mean_relative_rmse":sum(full_rmses)/len(full_rmses)},"physical_envelope":protocol["physical_envelope"],"route_correct":route_exact,"gates":gates,"passed":passed,"artifact_model_sha256_before":artifact_before,"artifact_model_sha256_after":artifact_after,"wall_seconds":time.perf_counter()-started,"peak_process_rss_bytes":peak_rss,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"rank_sweep_performed":False,"training_performed":False,"artifact_written":False,"source_blocks_promoted":0,"final_test_accessed":False,"phase3_certified":False,"claim_boundary":"Read-only shared rank-256 QKV/O all-head source-attention factorization with exact source MLP diagnostic only; no residual realization, component, host installation, physical runtime, autonomous, complete-model, Phase 3, or superiority claim."};result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest();_write_immutable(output/"result.json",json.dumps(result,indent=2,sort_keys=True).encode()+b"\n");return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_SHARED_RANK256_ALL_HEAD_ATTENTION_ORACLE_PROTOCOL_V434.json");parser.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_native_trajectory/shared_rank256_all_head_attention_v435");args=parser.parse_args();root=Path.cwd().resolve();result=execute(root,(root/args.protocol).resolve(),(root/args.output_dir).resolve());print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
