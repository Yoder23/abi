"""Fail-fast progressive native-trajectory joint conformance for one replacement layer."""

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


FORMAT = "abi-capability-compiler-phase3-progressive-joint-conformance/1"


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    from transformers import AutoModelForCausalLM
    protocol=json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format")!=FORMAT or protocol.get("status")!="PREREGISTERED_FAIL_FAST_PROGRESSIVE_JOINT_CONFORMANCE" or protocol.get("device")!="cuda" or protocol.get("final_test_access")!="PROHIBITED" or protocol.get("sweeps_authorized") is not False:raise Phase3Error("progressive joint governance changed")
    for name,expected in protocol["bindings"].items():
        path=Path(name) if Path(name).is_absolute() else root/name
        if not path.is_file() or sha256_file(path)!=expected:raise Phase3Error(f"progressive joint binding changed: {name}")
    layer_index=int(protocol["target_layer"])
    if layer_index<1 or len(protocol["prefix_checkpoints"])!=layer_index:raise Phase3Error("progressive prefix boundary changed")
    if output.exists() or not torch.cuda.is_available():raise Phase3Error("progressive output exists or CUDA unavailable")
    output.mkdir(parents=True);set_determinism(int(protocol["training"]["seed"]));torch.use_deterministic_algorithms(True);device=torch.device("cuda")
    base=json.loads((root/protocol["base_protocol"]).read_text(encoding="utf-8"));artifact=root/protocol["artifact"]["directory"];artifact_path=artifact/"model.safetensors";artifact_before=sha256_file(artifact_path);config=json.loads((artifact/"config.json").read_text(encoding="utf-8"));sys.path.insert(0,str((root/protocol["layercake_host"]).resolve()))
    from layercake.routed_sparse_rank768_progressive_core_fp16 import PrecisionConformantRoutedSparseRank768ProgressiveCore
    from layercake_extensions.decoder_direct_neural_core import DecoderAwareExternalTokenizer
    tokenizer=DecoderAwareExternalTokenizer.from_document(config["tokenizer"]);model=PrecisionConformantRoutedSparseRank768ProgressiveCore(**config["model"]).bind_tokenizer(tokenizer);model.load_state_dict(load_file(str(artifact_path),device="cpu"),strict=True,assign=True)
    state=model.state_dict()
    for expected_layer,item in enumerate(protocol["prefix_checkpoints"]):
        values=load_file(str(root/item["path"]),device="cpu")
        if any(not name.startswith(f"layers.{expected_layer}.") for name in values):raise Phase3Error("progressive checkpoint layer identity changed")
        with torch.no_grad():
            for name,value in values.items():state[name].copy_(value.to(state[name].dtype))
    examples=sequential.field._examples(root,base,tokenizer);train,validation=coverage.expanded_split(examples,seed=int(base["training"]["seed"]),maximum_tokens=128);rows=train+validation
    teacher=AutoModelForCausalLM.from_pretrained(base["source"]["snapshot_path"],local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation="eager").cuda().eval()
    for value in teacher.parameters():value.requires_grad_(False)
    prefix_layers=[model.layers[index].float().cuda().eval() for index in range(layer_index)];terminal=int(base["source"]["terminal_token_id"]);cache={};route_exact=0;process=psutil.Process();peak=process.memory_info().rss;torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for row_index,row in enumerate(rows):
            host=torch.tensor([row["input_ids"]],dtype=torch.long);route=model._select_route(host);route_exact+=int(route==routed._route(str(row["capability"])));candidate=model.token_embedding(host).to(device);positions=torch.arange(candidate.shape[1],device=device)
            for prefix in prefix_layers:candidate,_,_=prefix.forward_with_cache(candidate,positions,route)
            source=torch.tensor([[trajectory.source_token_id(value,terminal) for value in row["input_ids"]]],device=device);native=teacher.model.embed_tokens(source)
            for source_index in range(layer_index+1):_,native=dual._teacher_components(teacher,source_index,native)
            cache[str(row["record_id"])]=(candidate.squeeze(0).half().cpu(),native.squeeze(0).to(torch.bfloat16).cpu(),route);peak=max(peak,process.memory_info().rss)
            if (row_index+1)%500==0:print(json.dumps({"capture_records":row_index+1}),flush=True)
    del teacher,prefix_layers;torch.cuda.empty_cache();layer=model.layers[layer_index].float().cuda();trainable=list(layer.parameters())
    if sum(value.numel() for value in trainable)!=int(protocol["training"]["trainable_parameters"]):raise Phase3Error("progressive trainable boundary changed")
    optimizer=torch.optim.AdamW(trainable,lr=float(protocol["training"]["learning_rate"]),betas=(0.9,0.95),weight_decay=float(protocol["training"]["weight_decay"]));curves=[];layer.train()
    for step,row in enumerate(train,start=1):
        hidden,target,route=cache[str(row["record_id"])];hidden=hidden.unsqueeze(0).to(device);target=target.unsqueeze(0).to(device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            prediction,_,_=layer.forward_with_cache(hidden,torch.arange(hidden.shape[1],device=device),route);left,right=prediction.float(),target.float();relative_mse=torch.mean((left-right).square())/torch.mean(right.square()).clamp_min(1e-8);cosine=F.cosine_similarity(left.reshape(1,-1),right.reshape(1,-1)).mean();loss=relative_mse+float(protocol["training"]["cosine_weight"])*(1-cosine)
        if not torch.isfinite(loss):raise Phase3Error(f"progressive layer {layer_index} became nonfinite at step {step}")
        loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(trainable,float(protocol["training"]["gradient_clip_norm"]));optimizer.step();peak=max(peak,process.memory_info().rss)
        if step==1 or step%int(protocol["training"]["curve_interval"])==0:
            point={"step":step,"loss":float(loss),"relative_rmse":float(torch.sqrt(relative_mse)),"cosine":float(cosine),"gradient_norm":float(gradient_norm)};curves.append(point);print(json.dumps(point),flush=True)
    def evaluate(population):
        cosines=[];rmses=[];metrics=[]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            for row in population:
                hidden,target,route=cache[str(row["record_id"])];hidden=hidden.unsqueeze(0).to(device);target=target.unsqueeze(0).to(device);prediction,_,_=layer.forward_with_cache(hidden,torch.arange(hidden.shape[1],device=device),route);cosine,rmse=trajectory._metrics(prediction,target);cosines.append(cosine);rmses.append(rmse);metrics.append({"record_id":row["record_id"],"capability":row["capability"],"cosine":cosine,"relative_rmse":rmse})
        return {"records":len(population),"mean_cosine":sum(cosines)/len(cosines),"minimum_cosine":min(cosines),"mean_relative_rmse":sum(rmses)/len(rmses),"maximum_relative_rmse":max(rmses),"record_metrics":metrics}
    layer.eval();training_metrics=evaluate(train);validation_metrics=evaluate(validation);artifact_after=sha256_file(artifact_path);gates={"validation_mean_cosine":validation_metrics["mean_cosine"]>=float(protocol["gates"]["validation_mean_cosine_minimum"]),"validation_mean_relative_rmse":validation_metrics["mean_relative_rmse"]<=float(protocol["gates"]["validation_mean_relative_rmse_maximum"]),"routes_exact":route_exact==len(rows),"artifact_unchanged":artifact_before==artifact_after};passed=all(gates.values());checkpoint=None
    if passed:
        checkpoint_path=output/f"native_trajectory_layer_{layer_index:02d}.safetensors";tensors={f"layers.{layer_index}.{name}":value.detach().half().cpu().contiguous() for name,value in layer.named_parameters()};save_file(tensors,str(checkpoint_path),metadata={"format":FORMAT,"protocol_sha256":sha256_file(protocol_path)});checkpoint={"path":checkpoint_path.name,"sha256":sha256_file(checkpoint_path),"parameters":sum(value.numel() for value in tensors.values())}
    result={"format":FORMAT,"status":f"PASS_PROGRESSIVE_LAYER_{layer_index}" if passed else f"FAIL_PROGRESSIVE_LAYER_{layer_index}","protocol_sha256":sha256_file(protocol_path),"target_layer":layer_index,"prefix_checkpoint_sha256s":[item["sha256"] for item in protocol["prefix_checkpoints"]],"training":{"steps":len(train),"trainable_existing_parameters":sum(value.numel() for value in trainable),"curves":curves},"train":training_metrics,"validation":validation_metrics,"route_correct":route_exact,"gates":gates,"passed":passed,"checkpoint":checkpoint,"artifact_model_sha256_before":artifact_before,"artifact_model_sha256_after":artifact_after,"wall_seconds":time.perf_counter()-started,"peak_process_rss_bytes":peak,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"source_blocks_in_checkpoint":0,"teacher_activations_persisted":0,"final_test_accessed":False,"phase3_certified":False,"claim_boundary":f"Progressive realizable layer-{layer_index} conformance only; no complete artifact, autonomous quality, runtime, Phase 3, or superiority claim."};result["evidence_sha256"]=hashlib.sha256(canonical_json_bytes(result)).hexdigest();_write_immutable(output/"result.json",json.dumps(result,indent=2,sort_keys=True).encode()+b"\n");return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--protocol",default="ABI_CAPABILITY_COMPILER_PHASE3_PROGRESSIVE_LAYER1_PROTOCOL_V386.json");parser.add_argument("--output-dir",default="results/abi_capability_compiler_phase3_native_trajectory/progressive_layer1_v387");args=parser.parse_args();root=Path.cwd().resolve();print(json.dumps(execute(root,(root/args.protocol).resolve(),(root/args.output_dir).resolve()),indent=2,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
