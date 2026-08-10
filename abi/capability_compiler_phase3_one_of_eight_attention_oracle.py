"""Read-only fixed 1:8 structured source-attention quality oracle."""

from __future__ import annotations

import argparse, hashlib, json, os, sys, time
from pathlib import Path
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import psutil
from safetensors.torch import load_file
import torch
import torch.nn as nn

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_existing_attention_refit as coverage
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable

FORMAT="abi-capability-compiler-phase3-one-of-eight-attention-oracle/1"

def one_of_eight(weight:torch.Tensor):
    values=weight.detach().float()
    if values.shape[1]%8: raise Phase3Error("1:8 input width changed")
    groups=values.reshape(values.shape[0],values.shape[1]//8,8)
    indices=torch.argmax(groups.abs(),dim=-1,keepdim=True)
    mask=torch.zeros_like(groups).scatter_(-1,indices,1.0)
    retained=(groups*mask).reshape_as(values)
    energy=float(retained.square().sum()/values.square().sum().clamp_min(1e-30))
    return retained,energy

def linear_from(weight:torch.Tensor):
    layer=nn.Linear(weight.shape[1],weight.shape[0],bias=False,device=weight.device,dtype=weight.dtype)
    with torch.no_grad(): layer.weight.copy_(weight)
    return layer.eval()

def execute(root:Path,protocol_path:Path,output:Path)->dict:
    from transformers import AutoModelForCausalLM
    p=json.loads(protocol_path.read_text(encoding="utf-8"));structure=p.get("structured_support",{})
    if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_READ_ONLY_FIXED_ONE_OF_EIGHT_SOURCE_ATTENTION_ORACLE" or int(structure.get("group_size",0))!=8 or int(structure.get("retained_per_group",0))!=1 or structure.get("selection")!="STABLE_MAXIMUM_ABSOLUTE_SOURCE_WEIGHT" or p.get("training_authorized") is not False or p.get("artifact_write")!="PROHIBITED" or p.get("physical_runtime_claim")!="PROHIBITED" or p.get("final_test_access")!="PROHIBITED" or p.get("sweeps_authorized") is not False: raise Phase3Error("1:8 attention governance changed")
    for name,expected in p["bindings"].items():
        path=Path(name) if Path(name).is_absolute() else root/name
        if not path.is_file() or sha256_file(path)!=expected: raise Phase3Error(f"1:8 attention binding changed: {name}")
    if output.exists() or not torch.cuda.is_available(): raise Phase3Error("1:8 output exists or CUDA unavailable")
    output.mkdir(parents=True);set_determinism(int(p["seed"]));torch.use_deterministic_algorithms(True);device=torch.device("cuda");base=json.loads((root/p["base_protocol"]).read_text(encoding="utf-8"));artifact=root/p["artifact"]["directory"];artifact_path=artifact/"model.safetensors";artifact_before=sha256_file(artifact_path);config=json.loads((artifact/"config.json").read_text(encoding="utf-8"));sys.path.insert(0,str((root/p["layercake_host"]).resolve()))
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
    tokenizer=DecoderAwareExternalTokenizer.from_document(config["tokenizer"]);model=PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer);model.load_state_dict(load_file(str(artifact_path),device="cpu"),strict=True,assign=True);state=model.state_dict();prefix=load_file(str(root/p["layer0_checkpoint"]["path"]),device="cpu")
    with torch.no_grad():
        for name,value in prefix.items(): state[name].copy_(value.to(state[name].dtype))
    layer0=model.layers[0].float().cuda().eval();examples=sequential.field._examples(root,base,tokenizer);_,validation_rows=coverage.expanded_split(examples,seed=int(base["training"]["seed"]),maximum_tokens=int(p["population"]["maximum_sequence_actions"]));teacher=AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation="eager").cuda().eval()
    for value in teacher.parameters(): value.requires_grad_(False)
    source_layer1=teacher.model.layers[1];attention=source_layer1.self_attn;original_qkv=attention.qkv_proj;original_output=attention.o_proj;masked_qkv,qkv_energy=one_of_eight(original_qkv.weight);masked_output,output_energy=one_of_eight(original_output.weight);sparse_qkv=linear_from(masked_qkv);sparse_output=linear_from(masked_output);terminal=int(base["source"]["terminal_token_id"]);process=psutil.Process();peak=process.memory_info().rss;torch.cuda.reset_peak_memory_stats();started=time.perf_counter();route_exact=0;acos=[];arms=[];fcos=[];frms=[];xcos=[];xrms=[];records=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for row in validation_rows:
            host_ids=torch.tensor([row["input_ids"]],dtype=torch.long);route=model._select_route(host_ids);route_exact+=int(route==routed._route(str(row["capability"])));candidate=model.token_embedding(host_ids).to(device);positions=torch.arange(candidate.shape[1],device=device);candidate,_,_=layer0.forward_with_cache(candidate,positions,route);source_ids=torch.tensor([[trajectory.source_token_id(value,terminal) for value in row["input_ids"]]],dtype=torch.long,device=device);native=teacher.model.embed_tokens(source_ids)
            for source_index in range(2): _,native=dual._teacher_components(teacher,source_index,native)
            native=native.float();length=candidate.shape[1];position_ids=torch.arange(length,device=device)[None];position_embeddings=teacher.model.rotary_emb(candidate,position_ids);mask=dual.base._causal_mask(length,device=device,dtype=candidate.dtype);normalized=source_layer1.input_layernorm(candidate);attention.qkv_proj=original_qkv;attention.o_proj=original_output;exact_delta,_=attention(hidden_states=normalized,attention_mask=mask,position_ids=position_ids,use_cache=False,position_embeddings=position_embeddings);attention.qkv_proj=sparse_qkv;attention.o_proj=sparse_output;sparse_delta,_=attention(hidden_states=normalized,attention_mask=mask,position_ids=position_ids,use_cache=False,position_embeddings=position_embeddings);attention.qkv_proj=original_qkv;attention.o_proj=original_output;exact_attention=candidate.float()+exact_delta.float();sparse_attention=candidate.float()+sparse_delta.float();sparse_final=sparse_attention+source_layer1.mlp(source_layer1.post_attention_layernorm(sparse_attention)).float();full_final=exact_attention+source_layer1.mlp(source_layer1.post_attention_layernorm(exact_attention)).float();ac,ar=trajectory._metrics(sparse_attention,exact_attention);fc,fr=trajectory._metrics(sparse_final,native);xc,xr=trajectory._metrics(full_final,native);acos.append(ac);arms.append(ar);fcos.append(fc);frms.append(fr);xcos.append(xc);xrms.append(xr);records.append({"record_id":row["record_id"],"capability":row["capability"],"attention_cosine":ac,"attention_relative_rmse":ar,"final_cosine":fc,"final_relative_rmse":fr,"full_source_cosine":xc,"full_source_relative_rmse":xr});peak=max(peak,process.memory_info().rss)
    attention.qkv_proj=original_qkv;attention.o_proj=original_output;artifact_after=sha256_file(artifact_path);mean_cos=sum(fcos)/len(fcos);mean_rmse=sum(frms)/len(frms);qkv_nonzero=int(torch.count_nonzero(masked_qkv));output_nonzero=int(torch.count_nonzero(masked_output));expected=(masked_qkv.numel()+masked_output.numel())//8;gates={"exact_density":qkv_nonzero+output_nonzero==expected,"all_rows_present":bool((masked_qkv.abs().sum(1)>0).all() and (masked_output.abs().sum(1)>0).all()),"validation_mean_cosine":mean_cos>=float(p["gates"]["validation_mean_cosine_minimum"]),"validation_mean_relative_rmse":mean_rmse<=float(p["gates"]["validation_mean_relative_rmse_maximum"]),"routes_exact":route_exact==len(validation_rows),"artifact_unchanged":artifact_before==artifact_after};passed=all(gates.values());result={"format":FORMAT,"status":"PASS_ONE_OF_EIGHT_SOURCE_ATTENTION_ORACLE" if passed else "FAIL_ONE_OF_EIGHT_SOURCE_ATTENTION_ORACLE","protocol_sha256":sha256_file(protocol_path),"structured_support":{"group_size":8,"retained_per_group":1,"qkv_nonzero":qkv_nonzero,"output_nonzero":output_nonzero,"total_nonzero":qkv_nonzero+output_nonzero,"expected_nonzero":expected,"qkv_retained_weight_energy":qkv_energy,"output_retained_weight_energy":output_energy},"validation":{"records":len(validation_rows),"mean_attention_cosine":sum(acos)/len(acos),"mean_attention_relative_rmse":sum(arms)/len(arms),"mean_final_cosine":mean_cos,"minimum_final_cosine":min(fcos),"mean_final_relative_rmse":mean_rmse,"maximum_final_relative_rmse":max(frms),"record_metrics":records},"full_source_diagnostic":{"mean_cosine":sum(xcos)/len(xcos),"mean_relative_rmse":sum(xrms)/len(xrms)},"physical_envelope":p["physical_envelope"],"route_correct":route_exact,"gates":gates,"passed":passed,"artifact_model_sha256_before":artifact_before,"artifact_model_sha256_after":artifact_after,"wall_seconds":time.perf_counter()-started,"peak_process_rss_bytes":peak,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"dense_masked_diagnostic_execution":True,"physical_sparse_execution_verified":False,"support_sweep_performed":False,"training_performed":False,"artifact_written":False,"final_test_accessed":False,"phase3_certified":False,"claim_boundary":"Read-only fixed source-weight 1:8 QKV/O support quality oracle executed with equivalent dense masked matrices; no physical sparse execution, residual realization, component, autonomous, complete-model, Phase 3, or superiority claim."};result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest();_write_immutable(output/"result.json",json.dumps(result,indent=2,sort_keys=True).encode()+b"\n");return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_ONE_OF_EIGHT_ATTENTION_ORACLE_PROTOCOL_V437.json");parser.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_native_trajectory/one_of_eight_attention_v438");args=parser.parse_args();root=Path.cwd().resolve();result=execute(root,(root/args.protocol).resolve(),(root/args.output_dir).resolve());print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
