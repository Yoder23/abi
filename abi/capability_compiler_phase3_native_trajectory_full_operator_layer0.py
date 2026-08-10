"""Fail-fast full-operator native-trajectory conformance for replacement layer 0."""

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
from . import capability_compiler_phase3_routed_v15_layer0_extract as routed
from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as trajectory
from .capability_compiler_phase2_common import canonical_json_bytes, set_determinism, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable

FORMAT="abi-capability-compiler-phase3-native-trajectory-full-operator-layer0/1"


def execute(root:Path,protocol_path:Path,output:Path)->dict:
    from transformers import AutoModelForCausalLM
    p=json.loads(protocol_path.read_text(encoding="utf-8"))
    if p.get("format")!=FORMAT or p.get("status")!="PREREGISTERED_FAIL_FAST_NATIVE_TRAJECTORY_FULL_OPERATOR_LAYER_ZERO" or p.get("device")!="cuda" or p.get("final_test_access")!="PROHIBITED" or p.get("sweeps_authorized") is not False:raise Phase3Error("native-trajectory layer0 governance changed")
    for name,expected in p["bindings"].items():
        path=Path(name) if Path(name).is_absolute() else root/name
        if not path.is_file() or sha256_file(path)!=expected:raise Phase3Error(f"native-trajectory layer0 binding changed: {name}")
    if output.exists() or not torch.cuda.is_available():raise Phase3Error("native-trajectory output exists or CUDA unavailable")
    output.mkdir(parents=True);set_determinism(int(p["training"]["seed"]));torch.use_deterministic_algorithms(True);device=torch.device("cuda")
    base=json.loads((root/p["base_protocol"]).read_text(encoding="utf-8"));artifact=(root/p["artifact"]["directory"]).resolve();artifact_path=artifact/"model.safetensors";artifact_before=sha256_file(artifact_path);config=json.loads((artifact/"config.json").read_text(encoding="utf-8"));sys.path.insert(0,str((root/p["layercake_host"]["repository"]).resolve()))
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
    tokenizer=DecoderAwareExternalTokenizer.from_document(config["tokenizer"]);model=PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer);incompatible=model.load_state_dict(load_file(str(artifact_path),device="cpu"),strict=True,assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:raise Phase3Error("native-trajectory artifact strict load failed")
    examples=sequential.field._examples(root,base,tokenizer);cfg=base["calibration"];train,validation,tokens=dual._calibration_examples(examples,seed=int(base["training"]["seed"]),train_per_capability=int(cfg["train_records_per_capability"]),validation_per_capability=int(cfg["validation_records_per_capability"]),maximum_tokens=int(cfg["maximum_sequence_tokens"]));rows=train+validation
    teacher=AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation="eager").cuda().eval()
    for value in teacher.parameters():value.requires_grad_(False)
    terminal=int(base["source"]["terminal_token_id"]);cache={};route_exact=0
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for row in rows:
            host=torch.tensor([row["input_ids"]],dtype=torch.long);route=model._select_route(host);route_exact+=int(route==routed._route(str(row["capability"])));candidate_hidden=model.token_embedding(host).squeeze(0).half();source=torch.tensor([[trajectory.source_token_id(v,terminal) for v in row["input_ids"]]],dtype=torch.long,device=device);teacher_hidden=teacher.model.embed_tokens(source);_,target=dual._teacher_components(teacher,0,teacher_hidden);cache[str(row["record_id"])]=(candidate_hidden,target.squeeze(0).to(torch.bfloat16).cpu(),route)
    if route_exact!=len(rows):raise Phase3Error("native-trajectory layer0 router failed")
    del teacher;torch.cuda.empty_cache();layer=model.layers[0].float().cuda();trainable=list(layer.parameters())
    if sum(v.numel() for v in trainable)!=int(p["training"]["trainable_parameters"]):raise Phase3Error("native-trajectory trainable boundary changed")
    optimizer=torch.optim.AdamW(trainable,lr=float(p["training"]["learning_rate"]),betas=(0.9,0.95),weight_decay=float(p["training"]["weight_decay"]));curves=[];started=time.perf_counter();process=psutil.Process();peak=process.memory_info().rss;torch.cuda.reset_peak_memory_stats();layer.train()
    for step in range(int(p["training"]["steps"])):
        row=train[step%len(train)];hidden,target,route=cache[str(row["record_id"])];hidden=hidden.unsqueeze(0).to(device);target=target.unsqueeze(0).to(device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):pred,_,_=layer.forward_with_cache(hidden,torch.arange(hidden.shape[1],device=device),route);left,right=pred.float(),target.float();rmse=torch.sqrt(torch.mean((left-right)**2)/torch.mean(right**2).clamp_min(1e-8));cos=F.cosine_similarity(left.reshape(1,-1),right.reshape(1,-1)).mean();loss=rmse.square()+float(p["training"]["cosine_weight"])*(1-cos)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,float(p["training"]["gradient_clip_norm"]));optimizer.step()
        if not torch.isfinite(loss):raise Phase3Error("nonfinite native-trajectory loss")
        if step==0 or (step+1)%int(p["training"]["curve_interval"])==0:curves.append({"step":step+1,"loss":float(loss),"relative_rmse":float(rmse),"cosine":float(cos),"gradient_norm":float(norm)});print(json.dumps(curves[-1]),flush=True)
        peak=max(peak,process.memory_info().rss)
    def evaluate(values):
        cosines=[];rmses=[];records=[]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            for row in values:
                hidden,target,route=cache[str(row["record_id"])];hidden=hidden.unsqueeze(0).to(device);target=target.unsqueeze(0).to(device);pred,_,_=layer.forward_with_cache(hidden,torch.arange(hidden.shape[1],device=device),route);cos,rmse=trajectory._metrics(pred,target);cosines.append(cos);rmses.append(rmse);records.append({"record_id":row["record_id"],"capability":row["capability"],"cosine":cos,"relative_rmse":rmse})
        return {"records":len(values),"mean_cosine":sum(cosines)/len(cosines),"minimum_cosine":min(cosines),"mean_relative_rmse":sum(rmses)/len(rmses),"maximum_relative_rmse":max(rmses),"record_metrics":records}
    layer.eval();tr=evaluate(train);va=evaluate(validation);artifact_after=sha256_file(artifact_path);gates={"validation_mean_cosine":va["mean_cosine"]>=float(p["gates"]["validation_mean_cosine_minimum"]),"validation_mean_relative_rmse":va["mean_relative_rmse"]<=float(p["gates"]["validation_mean_relative_rmse_maximum"]),"routes_exact":route_exact==len(rows),"artifact_unchanged":artifact_before==artifact_after};passed=all(gates.values());checkpoint=None
    if passed:
        path=output/"native_trajectory_layer_00.safetensors";save_file({f"layers.0.{name}":value.detach().half().cpu().contiguous() for name,value in layer.named_parameters()},str(path),metadata={"format":FORMAT});checkpoint={"path":path.name,"sha256":sha256_file(path),"parameters":sum(v.numel() for v in trainable)}
    result={"format":FORMAT,"status":"PASS_NATIVE_TRAJECTORY_LAYER_ZERO" if passed else "FAIL_NATIVE_TRAJECTORY_LAYER_ZERO","protocol_sha256":sha256_file(protocol_path),"artifact_model_sha256_before":artifact_before,"artifact_model_sha256_after":artifact_after,"calibration_tokens":tokens,"training":{"steps":int(p["training"]["steps"]),"trainable_parameters":sum(v.numel() for v in trainable),"curves":curves,"wall_seconds":time.perf_counter()-started},"train":tr,"validation":va,"route_correct":route_exact,"gates":gates,"passed":passed,"checkpoint":checkpoint,"peak_process_rss_bytes":peak,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"teacher_activations_persisted":0,"source_blocks_in_checkpoint":0,"final_test_accessed":False,"phase3_certified":False,"claim_boundary":"Structured weight transfer plus hidden-state-distillation conformance at layer0 only; no autonomous quality, runtime, Phase 3, or superiority claim."};result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest();_write_immutable(output/"result.json",json.dumps(result,indent=2,sort_keys=True).encode()+b"\n");return result

def main():
    q=argparse.ArgumentParser();q.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_NATIVE_TRAJECTORY_FULL_OPERATOR_LAYER0_PROTOCOL_V354.json");q.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_native_trajectory/layer0_v355");a=q.parse_args();root=Path.cwd().resolve();print(json.dumps(execute(root,(root/a.protocol).resolve(),(root/a.output_dir).resolve()),indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
