"""Jointly fit the first two replacement layers with both native boundaries locked."""

from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import sys, time
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import psutil
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from . import capability_compiler_phase3_direct_linear_sequential_fit as sequential
from . import capability_compiler_phase3_dual_path_local_fit as dual
from . import capability_compiler_phase3_existing_attention_refit as coverage
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable

FORMAT="abi-capability-compiler-phase3-two-boundary-joint-prefix/1"


def execute(root:Path,protocol_path:Path,output:Path)->dict:
    from transformers import AutoModelForCausalLM
    p=json.loads(protocol_path.read_text(encoding="utf-8"))
    if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_FAIL_FAST_TWO_BOUNDARY_JOINT_PREFIX" or p.get("device")!="cuda" or p.get("final_test_access")!="PROHIBITED" or p.get("sweeps_authorized") is not False:raise Phase3Error("two-boundary governance changed")
    for name,expected in p["bindings"].items():
        path=Path(name) if Path(name).is_absolute() else root/name
        if not path.is_file() or sha256_file(path)!=expected:raise Phase3Error(f"two-boundary binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():raise Phase3Error("two-boundary output exists or CUDA unavailable")
    output.mkdir(parents=True);set_determinism(int(p["training"]["seed"]));torch.use_deterministic_algorithms(True);device=torch.device("cuda")
    base=json.loads((root/p["base_protocol"]).read_text(encoding="utf-8"));artifact=root/p["artifact"]["directory"];artifact_path=artifact/"model.safetensors";artifact_before=sha256_file(artifact_path);config=json.loads((artifact/"config.json").read_text(encoding="utf-8"));sys.path.insert(0,str((root/p["layercake_host"]).resolve()))
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
    tokenizer=DecoderAwareExternalTokenizer.from_document(config["tokenizer"]);model=PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer);model.load_state_dict(load_file(str(artifact_path),device="cpu"),strict=True,assign=True)
    prefix=load_file(str(root/p["initial_layer0_checkpoint"]["path"]),device="cpu");state=model.state_dict()
    with torch.no_grad():
        for name,value in prefix.items():state[name].copy_(value.to(state[name].dtype))
    examples=sequential.field._examples(root,base,tokenizer);train,validation=coverage.expanded_split(examples,seed=int(base["training"]["seed"]),maximum_tokens=128);rows=train+validation
    teacher=AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation="eager").cuda().eval()
    for value in teacher.parameters():value.requires_grad_(False)
    terminal=int(base["source"]["terminal_token_id"]);cache={};route_exact=0;process=psutil.Process();peak=process.memory_info().rss;torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for index,row in enumerate(rows):
            host=torch.tensor([row["input_ids"]],dtype=torch.long);route=model._select_route(host);route_exact+=int(route==routed._route(str(row["capability"])));candidate_input=model.token_embedding(host).squeeze(0).half();source=torch.tensor([[trajectory.source_token_id(value,terminal) for value in row["input_ids"]]],device=device);native=teacher.model.embed_tokens(source);targets=[]
            for layer_index in range(2):_,native=dual._teacher_components(teacher,layer_index,native);targets.append(native.squeeze(0).to(torch.bfloat16).cpu())
            cache[str(row["record_id"])]=(candidate_input,targets[0],targets[1],route);peak=max(peak,process.memory_info().rss)
            if (index+1)%500==0:print(json.dumps({"capture_records":index+1}),flush=True)
    del teacher;torch.cuda.empty_cache();layers=[model.layers[index].float().cuda() for index in range(2)];trainable=[value for layer in layers for value in layer.parameters()]
    if sum(value.numel() for value in trainable)!=int(p["training"]["trainable_parameters"]):raise Phase3Error("two-boundary trainable boundary changed")
    optimizer=torch.optim.AdamW(trainable,lr=float(p["training"]["learning_rate"]),betas=(0.9,0.95),weight_decay=float(p["training"]["weight_decay"]));curves=[]
    for layer in layers:layer.train()
    for step,row in enumerate(train,start=1):
        hidden,target0,target1,route=cache[str(row["record_id"])];hidden=hidden.unsqueeze(0).to(device);targets=[target0.unsqueeze(0).to(device),target1.unsqueeze(0).to(device)];optimizer.zero_grad(set_to_none=True);losses=[];cosines=[];rmses=[]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            for layer,target in zip(layers,targets):
                hidden,_,_=layer.forward_with_cache(hidden,torch.arange(hidden.shape[1],device=device),route);left,right=hidden.float(),target.float();relative_mse=torch.mean((left-right).square())/torch.mean(right.square()).clamp_min(1e-8);cosine=F.cosine_similarity(left.reshape(1,-1),right.reshape(1,-1)).mean();losses.append(relative_mse+float(p["training"]["cosine_weight"])*(1-cosine));cosines.append(cosine);rmses.append(torch.sqrt(relative_mse))
            loss=losses[0]+losses[1]
        if not torch.isfinite(loss):raise Phase3Error(f"two-boundary fit became nonfinite at step {step}")
        loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(trainable,float(p["training"]["gradient_clip_norm"]));optimizer.step();peak=max(peak,process.memory_info().rss)
        if step==1 or step%int(p["training"]["curve_interval"])==0:
            point={"step":step,"loss":float(loss),"layer0_cosine":float(cosines[0]),"layer0_relative_rmse":float(rmses[0]),"layer1_cosine":float(cosines[1]),"layer1_relative_rmse":float(rmses[1]),"gradient_norm":float(gradient_norm)};curves.append(point);print(json.dumps(point),flush=True)
    for layer in layers:layer.eval()
    def evaluate(population):
        boundary=[{"cosines":[],"rmses":[],"records":[]} for _ in range(2)]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            for row in population:
                hidden,target0,target1,route=cache[str(row["record_id"])];hidden=hidden.unsqueeze(0).to(device)
                for index,(layer,target) in enumerate(zip(layers,(target0,target1))):
                    hidden,_,_=layer.forward_with_cache(hidden,torch.arange(hidden.shape[1],device=device),route);cosine,rmse=trajectory._metrics(hidden,target.unsqueeze(0).to(device));boundary[index]["cosines"].append(cosine);boundary[index]["rmses"].append(rmse);boundary[index]["records"].append({"record_id":row["record_id"],"capability":row["capability"],"cosine":cosine,"relative_rmse":rmse})
        return [{"records":len(population),"mean_cosine":sum(item["cosines"])/len(item["cosines"]),"minimum_cosine":min(item["cosines"]),"mean_relative_rmse":sum(item["rmses"])/len(item["rmses"]),"maximum_relative_rmse":max(item["rmses"]),"record_metrics":item["records"]} for item in boundary]
    train_metrics=evaluate(train);validation_metrics=evaluate(validation);artifact_after=sha256_file(artifact_path);gates={}
    for index,item in enumerate(validation_metrics):gates[f"layer{index}_mean_cosine"]=item["mean_cosine"]>=float(p["gates"]["validation_mean_cosine_minimum"]);gates[f"layer{index}_mean_relative_rmse"]=item["mean_relative_rmse"]<=float(p["gates"]["validation_mean_relative_rmse_maximum"])
    gates["routes_exact"]=route_exact==len(rows);gates["artifact_unchanged"]=artifact_before==artifact_after;passed=all(gates.values());checkpoint=None
    if passed:
        checkpoint_path=output/"native_trajectory_layers_00_01.safetensors";tensors={f"layers.{index}.{name}":value.detach().half().cpu().contiguous() for index,layer in enumerate(layers) for name,value in layer.named_parameters()};save_file(tensors,str(checkpoint_path),metadata={"format":FORMAT,"protocol_sha256":sha256_file(protocol_path)});checkpoint={"path":checkpoint_path.name,"sha256":sha256_file(checkpoint_path),"parameters":sum(value.numel() for value in tensors.values())}
    result={"format":FORMAT,"status":"PASS_TWO_BOUNDARY_JOINT_PREFIX" if passed else "FAIL_TWO_BOUNDARY_JOINT_PREFIX","protocol_sha256":sha256_file(protocol_path),"initial_layer0_sha256":p["initial_layer0_checkpoint"]["sha256"],"training":{"steps":len(train),"trainable_existing_parameters":sum(value.numel() for value in trainable),"curves":curves},"train_boundaries":train_metrics,"validation_boundaries":validation_metrics,"route_correct":route_exact,"gates":gates,"passed":passed,"checkpoint":checkpoint,"artifact_model_sha256_before":artifact_before,"artifact_model_sha256_after":artifact_after,"wall_seconds":time.perf_counter()-started,"peak_process_rss_bytes":peak,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"source_blocks_in_checkpoint":0,"teacher_activations_persisted":0,"final_test_accessed":False,"phase3_certified":False,"claim_boundary":"Realizable two-boundary joint prefix conformance only; no complete artifact, autonomous quality, runtime, Phase 3, or superiority claim."};result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest();_write_immutable(output/"result.json",json.dumps(result,indent=2,sort_keys=True).encode()+b"\n");return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_TWO_BOUNDARY_JOINT_PREFIX_PROTOCOL_V389.json");parser.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_native_trajectory/two_boundary_joint_v390");args=parser.parse_args();root=Path.cwd().resolve();print(json.dumps(execute(root,(root/args.protocol).resolve(),(root/args.output_dir).resolve()),indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
